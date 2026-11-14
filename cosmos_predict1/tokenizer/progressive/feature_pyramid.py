"""Spatial-only F8/F16/F32 pyramid; no future-time statistics or pooling."""

import torch.nn.functional as F


def feature_pyramid(f8):
    if f8.ndim != 4:
        raise ValueError("one temporal latent per block, [B,C,H8,W8]")
    return {
        8: f8,
        16: F.avg_pool2d(f8, 2, 2, ceil_mode=True),
        32: F.avg_pool2d(f8, 4, 4, ceil_mode=True),
    }
