"""Separate-process CHP sender / receiver. Receiver arguments have no RGB input."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import export_decoder, load_checkpoint
from .model import Geometry
from .protocol import Packet, PrefixReceiver, Session
from .stream import DecoderState, ProgressiveDecoder, ProgressiveEncoder


def read_rgb(path, max_frames=None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    images = []
    while max_frames is None or len(images) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not images:
        raise ValueError("empty input video")
    return torch.from_numpy(np.stack(images)).permute(3, 0, 1, 2).unsqueeze(
        0
    ).float() / 127.5 - 1, fps


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def send(args):
    from fractions import Fraction

    model, identity = load_checkpoint(
        args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype)
    )
    video, fps = read_rgb(args.video, args.frames)
    geometry = Geometry.from_video(video)
    video = video.to(args.device, dtype=getattr(torch, args.dtype))
    rate = Fraction(fps).limit_denominator(1001)
    session = Session(
        args.model_id,
        identity,
        model.config.sha256,
        geometry.width,
        geometry.height,
        geometry.padded_width,
        geometry.padded_height,
        kmax=model.config.kmax,
        fps_num=rate.numerator,
        fps_den=rate.denominator,
        timebase_num=rate.denominator,
        timebase_den=rate.numerator,
    )
    budgets = [int(k) for k in args.budgets.split(",")]
    if (
        budgets != sorted(set(budgets))
        or budgets[0] < 256
        or budgets[-1] > model.config.kmax
    ):
        raise ValueError("increments must be unique and increasing, within [256,Kmax]")
    sender, state, records, all_ids = ProgressiveEncoder(model), None, [], []
    nblocks = 1 + (video.shape[2] - 1 + 3) // 4
    start = time.perf_counter()
    with open(args.stream, "xb") as target:
        target.write(session.to_bytes())
        for block in range(nblocks):
            position = 0 if block == 0 else 1 + 4 * (block - 1)
            count = 1 if block == 0 else min(4, video.shape[2] - position)
            synchronize(args.device)
            t0 = time.perf_counter()
            if block == 0:
                ids, state = sender.encode_bootstrap(video[:, :, :1])
            else:
                ids = sender.encode_block(
                    video[:, :, position : position + count],
                    state,
                    eos=block == nblocks - 1,
                )
            synchronize(args.device)
            latency = time.perf_counter() - t0
            all_ids.append(ids.cpu())
            offset = 0
            for k in budgets:
                packet = Packet(
                    block,
                    position,
                    count,
                    offset,
                    tuple(ids[0, offset:k].cpu().tolist()),
                    eos=block == nblocks - 1 and k == budgets[-1],
                )
                wire = packet.to_bytes()
                target.write(wire)
                records.append(
                    dict(
                        block_id=block,
                        budget=k,
                        offset=offset,
                        new_ids=k - offset,
                        packet_bytes=len(wire),
                        encode_seconds=latency,
                        cache_bytes=state.backbone.bytes,
                    )
                )
                offset = k
    torch.save(torch.stack(all_ids, 1), args.stream + ".sender_ids.pt")
    Path(args.stream + ".sender.json").write_text(
        json.dumps(
            dict(
                packets=records,
                elapsed_seconds=time.perf_counter() - start,
                actual_bytes=Path(args.stream).stat().st_size,
                source=str(Path(args.video).resolve()),
            ),
            indent=2,
        )
    )


@torch.no_grad()
def receive(args):
    # Decoder-only export is mandatory for process-isolation validation.
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if bundle.get("kind") != "chp_decoder":
        raise ValueError("receiver requires a public decoder export")
    del bundle
    model, identity = load_checkpoint(
        args.checkpoint,
        decoder_only=True,
        device=args.device,
        dtype=getattr(torch, args.dtype),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    state, records, received, frames = DecoderState(), [], [], []
    synchronize(args.device)
    begin = time.perf_counter()
    with open(args.stream, "rb") as source:
        session = Session.read(source)
        receiver = PrefixReceiver(
            session,
            model_id=args.model_id,
            checkpoint_sha256=identity,
            config_sha256=model.config.sha256,
        )
        decoder = ProgressiveDecoder(model, session.geometry)
        packet = receiver.read_packet(source)
        while packet is not None:
            next_packet = Packet.read(source)
            commit = next_packet is None or next_packet.block_id != packet.block_id
            prefix = torch.tensor(
                [receiver.blocks[-1]], dtype=torch.int32, device=args.device
            )
            synchronize(args.device)
            t0 = time.perf_counter()
            rgb = decoder.decode_block(
                prefix,
                state,
                frame_count=packet.frame_count,
                commit=commit,
                eos=packet.eos,
            )
            synchronize(args.device)
            latency = time.perf_counter() - t0
            # Save exact tensor, not a lossy video, so independent reconstruction can be checked.
            torch.save(
                rgb.cpu(),
                output / f"block_{packet.block_id:04}_k{prefix.shape[1]:04}.pt",
            )
            records.append(
                dict(
                    block_id=packet.block_id,
                    budget=prefix.shape[1],
                    decode_seconds=latency,
                    received_bytes=receiver.actual_bytes,
                    reference_cache_bytes=state.reference.backbone.bytes,
                    committed=commit,
                    shape=list(rgb.shape),
                )
            )
            if commit:
                frames.append(rgb.cpu())
                received.append(prefix.cpu())
            if next_packet is None:
                if not receiver.finished:
                    raise ValueError("missing explicit EOS")
                packet = None
            else:
                receiver.append_tokens(next_packet)
                packet = next_packet
    torch.save(torch.cat(frames, 2), output / "reconstruction.pt")
    torch.save(received, output / "received_ids.pt")
    Path(output / "receiver.json").write_text(
        json.dumps(
            dict(
                packets=records,
                actual_bytes=receiver.actual_bytes,
                elapsed_seconds=time.perf_counter() - begin,
                independent_process=True,
                inputs=[
                    str(Path(args.checkpoint).resolve()),
                    str(Path(args.stream).resolve()),
                ],
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("send", "receive"):
        p = sub.add_parser(mode)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--stream", required=True)
        p.add_argument("--model-id", default="CHP-VTok-v1")
        p.add_argument("--device", default="cuda")
        p.add_argument(
            "--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"]
        )
        if mode == "send":
            p.add_argument("--video", required=True)
            p.add_argument("--frames", type=int, default=17)
            p.add_argument("--budgets", default="256,512,1024,2048")
        else:
            p.add_argument("--output", required=True)
    export = sub.add_parser("export-decoder")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.mode == "export-decoder":
        export_decoder(args.checkpoint, args.output)
    else:
        (send if args.mode == "send" else receive)(args)


if __name__ == "__main__":
    main()
