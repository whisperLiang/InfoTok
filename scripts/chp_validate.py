#!/usr/bin/env python3
"""Measured GPU validation; never labels initialization as a trained model."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from cosmos_predict1.tokenizer.progressive.causal_state import (
    StreamState,
    install_dispatch,
)
from cosmos_predict1.tokenizer.progressive.model import (
    Geometry,
    file_sha256,
    official_backbone,
)


def measured(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    begin = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, dict(
        seconds=time.perf_counter() - begin,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        baseline_bytes=before,
        incremental_peak_bytes=torch.cuda.max_memory_allocated() - before,
    )


def read_video(path, height=None, frames=17):
    from cosmos_predict1.tokenizer.progressive.cli import read_rgb

    video, fps = read_rgb(path, frames)
    if height:
        b, c, t, h, w = video.shape
        video = (
            F.interpolate(
                video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w),
                (height, round(w * height / h)),
                mode="area",
            )
            .reshape(b, t, c, height, -1)
            .permute(0, 2, 1, 3, 4)
        )
    return video, fps


@torch.no_grad()
def upstream(args):
    from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
    from cosmos_predict1.tokenizer.networks import TokenizerConfigs

    model = CausalVideoTokenizer(
        checkpoint=args.backbone,
        tokenizer_config=TokenizerConfigs["OURS4x8x8_mse_256p_88"].value,
        device="cuda",
        dtype=args.dtype,
    )
    x, _ = read_video(args.video, args.height, args.frames)
    geometry = Geometry.from_video(x)
    x = geometry.pad(x.to("cuda", dtype=getattr(torch, args.dtype)))
    ((rgb, ratios), perf) = measured(
        lambda: model.autoencode(x, strategy="static", avg_rate=1.0)
    )
    torch.save(
        geometry.crop(rgb).cpu(), Path(args.output).with_suffix(".reconstruction.pt")
    )
    return dict(
        mode="official_infotok",
        input_shape=list(x.shape),
        output_shape=list(rgb.shape),
        performance=perf,
        ratios=ratios.tolist(),
        params=sum(p.numel() for p in model.parameters()),
    )


@torch.no_grad()
def backbone(args):
    model = (
        official_backbone(args.backbone)
        .to(device="cuda", dtype=getattr(torch, args.dtype))
        .eval()
    )
    enc, dec = model.encoder, model.decoder
    if args.window:
        install_dispatch(enc, window=args.window)
        install_dispatch(dec, window=args.window)
    video, _ = read_video(args.video, args.height, args.frames)
    geometry = Geometry.from_video(video)
    x = geometry.pad(video.to("cuda", dtype=getattr(torch, args.dtype)))
    z, encode = measured(lambda: enc(x))
    rgb, decode = measured(lambda: dec(z))
    base = dict(
        mode="backbone",
        architecture="local-window-change" if args.window else "global-SDPA-equivalent",
        window=args.window,
        params=sum(p.numel() for p in enc.parameters())
        + sum(p.numel() for p in dec.parameters()),
        input_shape=list(x.shape),
        latent_shape=list(z.shape),
        output_shape=list(geometry.crop(rgb).shape),
        encode=encode,
        decode=decode,
        pretrained=True,
        quantization_applied=False,
        note="continuous backbone geometry test; no discrete tokenizer quality claim",
    )
    if args.stream:
        install_dispatch(enc, window=args.window)
        install_dispatch(dec, window=args.window)
        es, ds = StreamState(), StreamState()
        chunks = [x[:, :, :1]] + [x[:, :, i : i + 4] for i in range(1, x.shape[2], 4)]
        if any(c.shape[2] != 4 for c in chunks[1:]):
            raise ValueError("stream comparison needs 1+4n frames")
        zs, ys = [], []

        def streamed():
            for chunk in chunks:
                with es.activate():
                    part = enc(chunk)
                zs.append(part)
                with ds.activate():
                    ys.append(dec(part))
            return torch.cat(ys, 2)

        output, perf = measured(streamed)
        z_stream = torch.cat(zs, 2)
        ids = model.quantizer(model.quant_conv(z))[0]
        streamed_ids = model.quantizer(model.quant_conv(z_stream))[0]
        base.update(
            stream=perf,
            encoder_max_error=float((z - z_stream).abs().max()),
            decoder_max_error=float((rgb - output).abs().max()),
            token_difference_fraction=float((ids != streamed_ids).float().mean()),
            encoder_cache_bytes=es.bytes,
            decoder_cache_bytes=ds.bytes,
            cache_entries=len(es.cache) + len(ds.cache),
        )
    return base


@torch.no_grad()
def progressive(args):
    from cosmos_predict1.tokenizer.progressive.checkpoint import load_checkpoint
    from cosmos_predict1.tokenizer.progressive.stream import (
        ProgressiveDecoder,
        ProgressiveEncoder,
    )

    if not args.checkpoint:
        raise ValueError(
            "progressive reconstruction requires an explicitly trained checkpoint"
        )
    model, identity = load_checkpoint(
        args.checkpoint, device="cuda", dtype=getattr(torch, args.dtype)
    )
    video, fps = read_video(args.video, None, args.frames)
    x = video.to("cuda", dtype=getattr(torch, args.dtype))
    encoder = ProgressiveEncoder(model)
    shape = Geometry.from_video(x)
    ids, state = encoder.encode_bootstrap(x[:, :, :1])
    first = ids
    block, encode = measured(lambda: encoder.encode_block(x[:, :, 1:5], state))
    decoder = ProgressiveDecoder(model, shape)
    records = []
    for k in (256, 512, 1024, 2048):
        _, ds = decoder.decode_bootstrap(first[:, :k])
        rgb, perf = measured(lambda: decoder.decode_block(block[:, :k], ds))
        torch.save(
            rgb.cpu(), Path(args.output).with_name(Path(args.output).stem + f"_k{k}.pt")
        )
        mse = (rgb.float().clamp(-1, 1) - x[:, :, 1:5].float()).square().mean() / 4
        records.append(
            dict(
                budget=k,
                psnr=float(-10 * torch.log10(mse)),
                decode=perf,
                input_shape=list(x[:, :, 1:5].shape),
                output_shape=list(rgb.shape),
                payload_bytes=2 * k,
                encoder_cache_bytes=state.backbone.bytes,
                decoder_cache_bytes=ds.reference.backbone.bytes,
            )
        )
    return dict(
        mode="progressive",
        checkpoint_sha256=identity,
        config_sha256=model.config.sha256,
        encode=encode,
        budgets=records,
        training_status="engineering-smoke-until-validated",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["upstream", "backbone", "progressive"])
    p.add_argument("--backbone", default="checkpoints/infotok-flex/infotok_mse.pt")
    p.add_argument("--checkpoint")
    p.add_argument("--video", required=True)
    p.add_argument("--height", type=int)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument(
        "--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        result = globals()[args.mode](args)
        result["status"] = "completed"
    except Exception as e:
        traceback.print_exc()
        result = dict(
            status="failed",
            error=type(e).__name__,
            message=str(e),
            traceback=traceback.format_exc(),
        )
    result.update(
        argv=sys.argv,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        dtype=args.dtype,
        backbone_sha256=file_sha256(args.backbone),
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
