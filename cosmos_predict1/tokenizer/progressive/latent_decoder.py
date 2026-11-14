"""Prefix-conditioned learned spatial queries with bounded slot neighborhoods.

Only IDs actually received are converted to codes. The fixed slot ordering is
regenerated from resolution metadata. Missing is a separate learned embedding;
ID zero is never a sentinel. Residual references use these same decoded codes.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from .anchors import anchors


@lru_cache(maxsize=48)
def neighbor_map(height, width, count, order, neighbors):
    _, positions = anchors(height, width, count, order)
    y = (torch.arange(height) + 0.5) / height
    x = (torch.arange(width) + 0.5) / width
    grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), -1).reshape(-1, 2)
    # Stable sort defines deterministic tie-break by slot ID, on CPU in float64.
    result = []
    for tile in grid.double().split(256):
        distances = (tile[:, None] - positions.double()[None]).square().sum(-1)
        result.append(
            torch.argsort(distances, dim=-1, stable=True)[:, : min(count, neighbors)]
        )
    return torch.cat(result), grid


class PrefixLatentDecoder(nn.Module):
    def __init__(self, channels, count, config):
        super().__init__()
        self.count, self.config = count, config
        d = config.hidden_dim
        self.slot = nn.Parameter(torch.randn(count, d) * 0.02)
        self.missing = nn.Parameter(torch.randn(d) * 0.02)
        self.code = nn.Linear(6, d)
        self.query = nn.Linear(2, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.output = nn.Linear(d, channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, codes, shape):
        b, received, _ = codes.shape
        if received > self.count:
            raise ValueError("prefix exceeds group length")
        h, w = shape
        indices, grid = neighbor_map(
            h, w, self.count, self.config.anchor_order, self.config.decoder_neighbors
        )
        indices, grid = (
            indices.to(codes.device),
            grid.to(device=codes.device, dtype=self.slot.dtype),
        )
        visible = torch.arange(self.count, device=codes.device) < received
        encoded = self.code(codes.to(self.slot.dtype))
        padded = F.pad(encoded, (0, 0, 0, self.count - received))
        slots = self.slot[None] + torch.where(
            visible[None, :, None], padded, self.missing[None, None]
        )
        # Bounded neighbors, evaluated in tiles. The learned missing representation is
        # intentionally retained; no unreceived integer or feature enters this graph.
        output = []
        d, heads = self.config.hidden_dim, self.config.heads
        for idx, xy in zip(indices.split(512), grid.split(512)):
            n = idx.shape[0]
            features = slots[:, idx]
            q = (
                self.query(xy)[None]
                .expand(b, -1, -1)
                .reshape(b * n, heads, 1, d // heads)
            )

            def project(module):
                return (
                    module(features)
                    .reshape(b * n, -1, heads, d // heads)
                    .transpose(1, 2)
                )

            out = F.scaled_dot_product_attention(
                q, project(self.key), project(self.value), dropout_p=0.0
            )
            output.append(self.output(out.reshape(b, n, d)))
        dense = torch.cat(output, 1).transpose(1, 2).reshape(b, -1, h, w)
        return dense + self.refine(dense)
