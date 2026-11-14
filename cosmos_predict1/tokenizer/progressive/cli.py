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
from .transport import open_transport


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


def _send_session(args, model, identity, video, fps):
    from fractions import Fraction

    geometry = Geometry.from_video(video)
    rate = Fraction(fps).limit_denominator(1001)
    return Session(
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


def _transmission_plan(args, kmax, nblocks):
    budgets = [int(k) for k in args.budgets.split(",")]
    if budgets != sorted(set(budgets)):
        raise ValueError("increments must be unique and increasing")
    if budgets[0] < 256 or budgets[-1] > kmax:
        raise ValueError("increments outside [256,Kmax]")
    text = getattr(args, "block_budgets", None)
    limits = [int(k) for k in text.split(",")] if text else [budgets[-1]] * nblocks
    if len(limits) != nblocks:
        raise ValueError(
            "one block budget per bootstrap and temporal block is required"
        )
    if any(k < 256 or k > kmax for k in limits):
        raise ValueError("invalid per-block budget")
    return [
        sorted(set([k for k in budgets if k < limit] + [limit])) for limit in limits
    ]


def _sender_output_prefix(args):
    prefix = getattr(args, "log_prefix", None) or args.stream
    if prefix.startswith("tcp://"):
        raise ValueError("TCP sender requires --log-prefix for local measurements")
    return prefix


def _reset_peak(device):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def _peak_bytes(device):
    return torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0


def _encode_time_block(sender, video, block, state, device):
    position = 0 if block == 0 else 1 + 4 * (block - 1)
    count = 1 if block == 0 else min(4, video.shape[2] - position)
    synchronize(device)
    start = time.monotonic_ns()
    if block == 0:
        ids, state = sender.encode_bootstrap(video[:, :, :1])
    else:
        ids = sender.encode_block(
            video[:, :, position : position + count],
            state,
            eos=position + count == video.shape[2],
        )
    synchronize(device)
    return ids, state, position, count, start, (time.monotonic_ns() - start) / 1e9


def _write_increments(
    target, ids, state, block, position, count, start, latency, increments, last
):
    records, offset = [], 0
    for k in increments:
        packet = Packet(
            block,
            position,
            count,
            offset,
            tuple(ids[0, offset:k].cpu().tolist()),
            eos=last and k == increments[-1],
            block_end=k == increments[-1],
            tail=last and block > 0 and count < 4,
        )
        wire = packet.to_bytes()
        target.write(wire)
        target.flush()
        records.append(
            dict(
                block_id=block,
                budget=k,
                offset=offset,
                new_ids=k - offset,
                packet_bytes=len(wire),
                encode_seconds=latency,
                cache_bytes=state.backbone.bytes,
                encode_start_ns=start,
                packet_sent_ns=time.monotonic_ns(),
            )
        )
        offset = k
    return records


@torch.no_grad()
def send(args):
    model, identity = load_checkpoint(
        args.checkpoint, device=args.device, dtype=getattr(torch, args.dtype)
    )
    video, fps = read_rgb(args.video, args.frames)
    session = _send_session(args, model, identity, video, fps)
    video = video.to(args.device, dtype=getattr(torch, args.dtype))
    nblocks = 1 + (video.shape[2] - 1 + 3) // 4
    plans = _transmission_plan(args, model.config.kmax, nblocks)
    log_prefix = _sender_output_prefix(args)
    sender, state, records, all_ids = ProgressiveEncoder(model), None, [], []
    begin = time.perf_counter()
    _reset_peak(args.device)
    with open_transport(args.stream, "xb") as target:
        target.write(session.to_bytes())
        target.flush()
        for block, increments in enumerate(plans):
            ids, state, position, count, start, latency = _encode_time_block(
                sender, video, block, state, args.device
            )
            all_ids.append(ids.cpu())
            records.extend(
                _write_increments(
                    target,
                    ids,
                    state,
                    block,
                    position,
                    count,
                    start,
                    latency,
                    increments,
                    block == nblocks - 1,
                )
            )
    torch.save(torch.stack(all_ids, 1), log_prefix + ".sender_ids.pt")
    metadata = dict(
        packets=records,
        elapsed_seconds=time.perf_counter() - begin,
        actual_bytes=len(session.to_bytes()) + sum(p["packet_bytes"] for p in records),
        protocol_version=session.protocol_version,
        transport="tcp" if args.stream.startswith("tcp://") else "file",
        peak_allocated_bytes=_peak_bytes(args.device),
        source=str(Path(args.video).resolve()),
    )
    Path(log_prefix + ".sender.json").write_text(json.dumps(metadata, indent=2))


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
    _reset_peak(args.device)
    with open_transport(args.stream, "rb") as source:
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
            commit = packet.block_end
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
                eos=packet.eos or packet.tail,
            )
            synchronize(args.device)
            latency = time.perf_counter() - t0
            ready_ns = time.monotonic_ns()
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
                    reconstruction_ready_ns=ready_ns,
                    shape=list(rgb.shape),
                )
            )
            if commit:
                frames.append(rgb.cpu())
                received.append(prefix.cpu())
            packet = receiver.read_packet(source)
    torch.save(torch.cat(frames, 2), output / "reconstruction.pt")
    torch.save(received, output / "received_ids.pt")
    Path(output / "receiver.json").write_text(
        json.dumps(
            dict(
                packets=records,
                actual_bytes=receiver.actual_bytes,
                elapsed_seconds=time.perf_counter() - begin,
                independent_process=True,
                protocol_version=session.protocol_version,
                transport="tcp" if args.stream.startswith("tcp://") else "file",
                peak_allocated_bytes=_peak_bytes(args.device),
                inputs=[
                    str(Path(args.checkpoint).resolve()),
                    args.stream
                    if args.stream.startswith("tcp://")
                    else str(Path(args.stream).resolve()),
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
            p.add_argument(
                "--block-budgets", help="one final budget per bootstrap/block"
            )
            p.add_argument(
                "--log-prefix", help="local measurement path for a TCP stream"
            )
        else:
            p.add_argument("--output", required=True)
    export = sub.add_parser("export-decoder")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.mode == "export-decoder":
        export_decoder(args.checkpoint, args.output)
    else:
        (send if args.mode == "send" else receive)(args)


if __name__ == "__main__":
    main()
