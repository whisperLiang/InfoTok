#!/usr/bin/env python3
"""Native RGB rate-distortion, actual wire bytes, and history intervention.

Small engineering samples are labeled as such. No monotonic smoothing, value
clipping of gains, or random-weight model is accepted as a checkpoint result.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from cosmos_predict1.tokenizer.progressive.checkpoint import load_checkpoint
from cosmos_predict1.tokenizer.progressive.cli import read_rgb
from cosmos_predict1.tokenizer.progressive.model import Geometry
from cosmos_predict1.tokenizer.progressive.protocol import Packet, Session
from cosmos_predict1.tokenizer.progressive.stream import (
    DecoderState,
    ProgressiveDecoder,
    ProgressiveEncoder,
)
from scripts.chp_validate import measured


class Metrics:
    def __init__(self):
        import lpips
        from DISTS_pytorch import DISTS

        self.lpips = lpips.LPIPS(net="alex").cuda().eval()
        self.dists = DISTS().cuda().eval()

    @torch.no_grad()
    def __call__(self, source, reconstruction):
        source = source.float().clamp(-1, 1)
        reconstruction = reconstruction.float().clamp(-1, 1)
        psnr, ssim, lp, di = [], [], [], []
        for i in range(source.shape[2]):
            x, y = source[:, :, i], reconstruction[:, :, i]
            mse = (x - y).square().mean() / 4
            psnr.append(float(-10 * torch.log10(mse)))
            a, b = [z[0].permute(1, 2, 0).cpu().numpy() / 2 + 0.5 for z in (x, y)]
            ssim.append(
                float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
            )
            lp.append(float(self.lpips(x, y)))
            # DISTS.forward receives the full native tensor. Its optional PIL
            # prepare_image(resize=True) preprocessing is deliberately not used.
            di.append(
                float(
                    self.dists(
                        (x + 1) / 2, (y + 1) / 2, require_grad=False, batch_average=True
                    )
                )
            )
        return dict(
            psnr=float(np.mean(psnr)),
            ssim=float(np.mean(ssim)),
            lpips=float(np.mean(lp)),
            dists=float(np.mean(di)),
            per_frame=dict(psnr=psnr, ssim=ssim, lpips=lp, dists=di),
        )


def wire_bytes(session, tokens, budgets):
    packets = []
    for i, k in enumerate(budgets):
        packet = Packet(
            i,
            0 if i == 0 else 1 + 4 * (i - 1),
            1 if i == 0 else 4,
            0,
            tuple(tokens[0, i, :k].cpu().tolist()),
            eos=i == len(budgets) - 1,
            block_end=True,
        )
        packets.append(packet)
    return len(session.to_bytes()) + sum(len(p.to_bytes()) for p in packets)


@torch.no_grad()
def decode(model, geometry, tokens, budgets):
    decoder, state = ProgressiveDecoder(model, geometry), DecoderState()
    frames = []
    for i, k in enumerate(budgets):
        frames.append(
            decoder.decode_block(
                tokens[:, i, :k], state, frame_count=1 if i == 0 else 4
            )
        )
    return torch.cat(frames, 2), state


def comparison(source, reconstructions, path):
    # Exact RGB outputs, reduced only for this display contact sheet. Full geometry
    # tensors and PNGs are preserved separately; no fabricated/simulated image.
    frame = min(8, source.shape[2] - 1)
    images = [("Original", source)] + [(str(k), v) for k, v in reconstructions.items()]
    cellw, cellh = 640, 360
    canvas = Image.new("RGB", (cellw * len(images), cellh + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, tensor) in enumerate(images):
        pixels = (
            ((tensor[0, :, frame].float().clamp(-1, 1) + 1) * 127.5)
            .round()
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        image = Image.fromarray(pixels)
        image.save(path.with_name(path.stem + f"_{label}_native.png"))
        canvas.paste(image.resize((cellw, cellh)), (cellw * i, 28))
        draw.text((cellw * i + 10, 8), label, fill="black")
    canvas.save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--output", required=True)
    p.add_argument("--history-intervention", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, identity = load_checkpoint(
        args.checkpoint, device="cuda", dtype=getattr(torch, args.dtype)
    )
    x, fps = read_rgb(args.video, args.frames)
    x = x.cuda().to(getattr(torch, args.dtype))
    geometry = Geometry.from_video(x)
    encoder = ProgressiveEncoder(model)
    enc_state = None

    def encode_all():
        nonlocal enc_state
        ids, enc_state = encoder.encode_bootstrap(x[:, :, :1])
        tokens = [ids]
        for start in range(1, x.shape[2], 4):
            tokens.append(encoder.encode_block(x[:, :, start : start + 4], enc_state))
        return torch.stack(tokens, 1)

    tokens, encode_perf = measured(encode_all)
    torch.save(tokens.cpu(), out / "token_ids.pt")
    session = Session(
        "CHP-VTok-v1",
        identity,
        model.config.sha256,
        geometry.width,
        geometry.height,
        geometry.padded_width,
        geometry.padded_height,
        kmax=model.config.kmax,
        fps_num=round(fps),
        timebase_den=round(fps),
    )
    metrics = Metrics()
    results, recons = [], {}
    for k in (256, 512, 1024, 2048):
        budgets = [k] * tokens.shape[1]
        (rgb, state), perf = measured(lambda: decode(model, geometry, tokens, budgets))
        quality = metrics(x, rgb)
        byte_count = wire_bytes(session, tokens, budgets)
        result = dict(
            video=Path(args.video).stem,
            model="morton"
            if model.config.anchor_order == "morton"
            else ("multi" if model.config.multiscale else "single"),
            checkpoint_sha256=identity,
            config_sha256=model.config.sha256,
            budget=k,
            protocol_version=session.protocol_version,
            actual_bytes=byte_count,
            bpp=byte_count * 8 / (geometry.width * geometry.height * x.shape[2]),
            mbps=byte_count * 8 * fps / x.shape[2] / 1e6,
            encode_seconds=encode_perf["seconds"],
            decode_seconds=perf["seconds"],
            peak_vram_bytes=max(
                encode_perf["peak_allocated_bytes"], perf["peak_allocated_bytes"]
            ),
            cache_bytes=state.reference.backbone.bytes,
            frames=x.shape[2],
            height=geometry.height,
            width=geometry.width,
            fps=fps,
            training_status="engineering-smoke-until-validated",
            **quality,
        )
        results.append(result)
        torch.save(rgb.cpu(), out / f"reconstruction_k{k}.pt")
        recons[k] = rgb.cpu()
        print(json.dumps(result), flush=True)
    comparison(x.cpu(), recons, out / "comparison.png")
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    fields = [k for k in results[0] if k != "per_frame"]
    with (out / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [{k: v for k, v in row.items() if k in fields} for row in results]
        )
    if args.history_intervention:
        rows = []
        # Keep every target ID and every future input unchanged; only change a
        # historical source's received budget, and report negative gains intact.
        base, _ = decode(model, geometry, tokens, [256] * tokens.shape[1])
        for source in range(1, tokens.shape[1] - 1):
            for target in range(source + 1, tokens.shape[1]):
                budgets = [256] * tokens.shape[1]
                budgets[source] = 2048
                enhanced, _ = decode(model, geometry, tokens, budgets)
                start = 1 + 4 * (target - 1)
                base_quality = metrics(
                    x[:, :, start : start + 4], base[:, :, start : start + 4]
                )
                new_quality = metrics(
                    x[:, :, start : start + 4], enhanced[:, :, start : start + 4]
                )
                rows.append(
                    dict(
                        source_block=source,
                        target_block=target,
                        target_ids_unchanged=True,
                        baseline=base_quality,
                        enhanced=new_quality,
                        psnr_gain=new_quality["psnr"] - base_quality["psnr"],
                        additional_payload_bytes=(2048 - 256) * 2,
                        max_rgb_change=float(
                            (
                                base[:, :, start : start + 4]
                                - enhanced[:, :, start : start + 4]
                            )
                            .abs()
                            .max()
                        ),
                    )
                )
        (out / "history_intervention.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
