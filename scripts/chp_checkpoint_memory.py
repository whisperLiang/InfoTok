#!/usr/bin/env python3
"""Measure frozen native Decoder input gradients, with and without checkpointing."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from cosmos_predict1.tokenizer.progressive.cli import read_rgb
from cosmos_predict1.tokenizer.progressive.config import CHPConfig
from cosmos_predict1.tokenizer.progressive.model import CHPVTokenizer, Geometry
from scripts.chp_validate import measured

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--video", required=True)
p.add_argument("--protection", choices=["on", "off"], required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(5)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
raw = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
cfg = replace(
    CHPConfig(**raw["chp_config"]),
    backbone_checkpoint=None,
    checkpoint_decoder=a.protection == "on",
)
model = CHPVTokenizer(cfg)
weights = {
    k.removeprefix("network."): v
    for k, v in raw["model"].items()
    if k.startswith("network.")
}
model.load_state_dict(weights, strict=False)
model = model.to(device="cuda", dtype=torch.bfloat16).eval()
x, _ = read_rgb(a.video, 5)
x = x.cuda().bfloat16()
geometry = Geometry.from_video(x)
with torch.no_grad():
    ids, _ = model.encode(x)
    dense = torch.stack(
        [
            model.dense_from_ids(ids[:, i], geometry.latent_shape)
            for i in range(ids.shape[1])
        ],
        2,
    )
z = dense.detach().requires_grad_(True)


def run():
    result = geometry.crop(model.decoder(z))
    loss = (result.float() - x.float()).square().mean()
    loss.backward()
    return result.detach(), loss.detach()


(rgb, loss), perf = measured(run)
assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
assert all(p.grad is None for p in model.decoder.parameters())
info = dict(
    protection=a.protection,
    geometry=list(x.shape),
    loss=float(loss),
    latent_gradient_norm=float(z.grad.float().norm()),
    finite=True,
    decoder_parameter_gradients=False,
    performance=perf,
)
Path(a.output).write_text(json.dumps(info, indent=2) + "\n")
torch.save(
    dict(rgb=rgb.cpu(), gradient=z.grad.cpu()), Path(a.output).with_suffix(".pt")
)
print(json.dumps(info, indent=2))
