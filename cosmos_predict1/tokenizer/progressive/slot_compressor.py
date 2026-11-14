"""Fixed learned 1D queries attend only to bounded neighborhoods at shared anchors."""

import torch
import torch.nn.functional as F
from torch import nn

from .anchors import anchors


class SlotCompressor(nn.Module):
    def __init__(self, channels, count, config):
        super().__init__()
        self.count, self.config = count, config
        d = config.hidden_dim
        self.query = nn.Parameter(torch.randn(count, d) * 0.02)
        self.position = nn.Linear(2, d)
        self.key = nn.Linear(channels, d)
        self.value = nn.Linear(channels, d)
        self.norm = nn.LayerNorm(d)
        self.to_code = nn.Linear(d, 6)

    def forward(self, features):
        b, c, h, w = features.shape
        coords, positions = anchors(
            h, w, self.count, self.config.anchor_order, features.device
        )
        radius = self.config.local_radius
        delta = torch.arange(-radius, radius + 1, device=features.device)
        offsets = torch.stack(torch.meshgrid(delta, delta, indexing="ij"), -1).reshape(
            -1, 2
        )
        indices = coords[:, None] + offsets[None]
        valid = (
            (indices[..., 0] >= 0)
            & (indices[..., 0] < h)
            & (indices[..., 1] >= 0)
            & (indices[..., 1] < w)
        )
        flat = indices[..., 0].clamp(0, h - 1) * w + indices[..., 1].clamp(0, w - 1)
        patches = features.flatten(2).transpose(1, 2)[:, flat]
        d, heads = self.config.hidden_dim, self.config.heads
        q = (self.query + self.position(positions.to(self.query.dtype)))[None].expand(
            b, -1, -1
        )
        q = q.reshape(b * self.count, heads, 1, d // heads)

        def project(module):
            return (
                module(patches)
                .reshape(b * self.count, -1, heads, d // heads)
                .transpose(1, 2)
            )

        allowed = (
            valid[None, :, None, None]
            .expand(b, -1, 1, 1, -1)
            .reshape(b * self.count, 1, 1, -1)
        )
        out = F.scaled_dot_product_attention(
            q, project(self.key), project(self.value), attn_mask=allowed, dropout_p=0.0
        )
        out = out.reshape(b, self.count, d)
        return self.to_code(self.norm(out + self.query[None]))
