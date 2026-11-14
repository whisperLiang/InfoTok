#!/usr/bin/env python3
"""A full causal graph, B legal-history recomputation, C actual state reuse."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_predict1.tokenizer.progressive.checkpoint import load_checkpoint
from cosmos_predict1.tokenizer.progressive.stream import (
    DecoderState,
    ProgressiveDecoder,
    ProgressiveEncoder,
)
from scripts.chp_validate import measured, read_video


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--height", type=int)
    p.add_argument("--frames", type=int, default=17)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, sha = load_checkpoint(
        args.checkpoint, device="cuda", dtype=getattr(torch, args.dtype)
    )
    x, _ = read_video(args.video, args.height, args.frames)
    x = x.cuda().to(getattr(torch, args.dtype))
    (full, geometry), full_enc = measured(lambda: model.encode(x))

    # B: repeat legal history, never truncate/re-bootstrap the raw past window.
    def recompute():
        return torch.stack(
            [
                model.encode(x[:, :, :end])[0][:, -1]
                for end in range(1, x.shape[2] + 1, 4)
            ],
            1,
        )

    repeated, repeat_enc = measured(recompute)
    enc = ProgressiveEncoder(model)
    state = None

    def cached():
        nonlocal state
        first, state = enc.encode_bootstrap(x[:, :, :1])
        return torch.stack(
            [first]
            + [
                enc.encode_block(x[:, :, i : i + 4], state)
                for i in range(1, x.shape[2], 4)
            ],
            1,
        )

    streamed, stream_enc = measured(cached)
    budgets = [(256, 512, 1024, 2048)[i % 4] for i in range(full.shape[1])]
    causal, full_dec = measured(lambda: model.decode(full, geometry, budgets))
    decoder = ProgressiveDecoder(model, geometry)
    ds = DecoderState()

    def cached_decode():
        return torch.cat(
            [
                decoder.decode_block(full[:, i, :k], ds, frame_count=1 if i == 0 else 4)
                for i, k in enumerate(budgets)
            ],
            2,
        )

    rgb, stream_dec = measured(cached_decode)
    # Independent interventions on future raw frames and future integer IDs.
    changed = x.clone()
    changed[:, :, 5:] = torch.flip(changed[:, :, 5:], [-1]) + 0.2
    altered, _ = model.encode(changed)
    future = full.clone()
    future[:, 2:] = (future[:, 2:] + 1237) % 64000
    changed_rgb = model.decode(future, geometry, budgets)
    result = dict(
        checkpoint_sha256=sha,
        dtype=args.dtype,
        shape=list(x.shape),
        budgets=budgets,
        encoder_future_past_ids_equal=bool(torch.equal(full[:, :2], altered[:, :2])),
        decoder_future_past_max_abs=float(
            (causal[:, :, :5] - changed_rgb[:, :, :5]).abs().max()
        ),
        full_vs_recompute_ids_equal=bool(torch.equal(full, repeated)),
        full_vs_stream_ids_equal=bool(torch.equal(full, streamed)),
        full_vs_recompute_id_fraction=float((full != repeated).float().mean()),
        full_vs_stream_id_fraction=float((full != streamed).float().mean()),
        full_vs_stream_rgb_max_abs=float((causal - rgb).abs().max()),
        full_vs_stream_rgb_mean_abs=float((causal - rgb).abs().mean()),
        full_encode=full_enc,
        repeated_encode=repeat_enc,
        cached_encode=stream_enc,
        full_decode=full_dec,
        cached_decode=stream_dec,
        encoder_cache_bytes=state.backbone.bytes,
        decoder_cache_bytes=ds.reference.backbone.bytes,
        encoder_cache_entries=len(state.backbone.cache),
        decoder_cache_entries=len(ds.reference.backbone.cache),
        actual_new_frames=state.backbone.input_frames,
        history_convolutions_reused=True,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
