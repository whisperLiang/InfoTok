#!/usr/bin/env python3
"""Original bmm vs SDPA error, gradients, memory, latency and checkpoint ID drift."""

import argparse
import copy
import json
import runpy
from pathlib import Path
from types import MethodType

import torch

from cosmos_predict1.tokenizer.modules.layers3d import (
    CausalAttnBlock,
    CausalTemporalAttnBlock,
)
from cosmos_predict1.tokenizer.progressive.model import Geometry, official_backbone
from scripts.chp_validate import measured, read_video

explicit = runpy.run_path("tests/test_chp_attention.py")["explicit_attention"]


def compare(dtype, cls):
    torch.manual_seed(11)
    module = cls(128, 1).to(device="cuda", dtype=dtype)
    ref = copy.deepcopy(module)
    ref.forward = MethodType(explicit, ref)
    x = torch.randn(1, 128, 5, 16, 24, device="cuda", dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    with torch.no_grad():
        for _ in range(3):
            module(x)
            ref(y)
    a, ap = measured(lambda: module(x))
    b, bp = measured(lambda: ref(y))
    a.square().mean().backward()
    b.square().mean().backward()
    return dict(
        dtype=str(dtype),
        class_name=cls.__name__,
        sdpa=ap,
        explicit=bp,
        forward_max_abs=float((a - b).abs().max()),
        input_gradient_max_abs=float((x.grad - y.grad).abs().max()),
        parameter_gradient_max_abs=max(
            float((p.grad - q.grad).abs().max())
            for p, q in zip(module.parameters(), ref.parameters())
        ),
    )


@torch.no_grad()
def checkpoint_compare(args, dtype):
    model = official_backbone(args.backbone).to(device="cuda", dtype=dtype).eval()
    video, _ = read_video(args.video, 256, 5)
    geometry = Geometry.from_video(video)
    x = geometry.pad(video.to(device="cuda", dtype=dtype))
    for _ in range(2):
        model.decode(model.encode(x)[1])
    (ids, codes, _), enc = measured(lambda: model.encode(x))
    recon, dec = measured(lambda: model.decode(codes))
    for m in model.modules():
        if isinstance(m, (CausalAttnBlock, CausalTemporalAttnBlock)):
            m.forward = MethodType(explicit, m)
    for _ in range(2):
        model.decode(model.encode(x)[1])
    (old_ids, old_codes, _), old_enc = measured(lambda: model.encode(x))
    old_recon, old_dec = measured(lambda: model.decode(old_codes))
    return dict(
        dtype=str(dtype),
        id_difference_fraction=float((ids != old_ids).float().mean()),
        final_reconstruction_max_abs=float((recon - old_recon).abs().max()),
        final_reconstruction_mean_abs=float((recon - old_recon).abs().mean()),
        sdpa_encode=enc,
        sdpa_decode=dec,
        explicit_encode=old_enc,
        explicit_decode=old_dec,
        total_params=sum(p.numel() for p in model.parameters()),
        encoder_params=sum(p.numel() for p in model.encoder.parameters()),
        decoder_params=sum(p.numel() for p in model.decoder.parameters()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="checkpoints/infotok-flex/infotok_mse.pt")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="results/chp_vtok/attention_numerics.json")
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result = dict(
        attention=[
            compare(d, c)
            for d in (torch.float32, torch.float16, torch.bfloat16)
            for c in (CausalAttnBlock, CausalTemporalAttnBlock)
        ],
        checkpoint=[],
    )
    for d in (torch.float32, torch.bfloat16):
        torch.cuda.empty_cache()
        result["checkpoint"].append(checkpoint_compare(a, d))
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
