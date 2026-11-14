"""Opt-in high-resolution sampling of the official perceptual training loss.

Full RGB and temporal objectives remain full resolution. A uniform rectangular
source-pixel crop is shared by all frames of one sample. This estimates a
patch-based perceptual objective, not full-frame LPIPS. Final evaluation always
uses complete 1080p frames and the separately standardized LPIPS/DISTS metrics.
"""

import torch
from cosmos_predict1.tokenizer.training.losses.continuous import PerceptualLoss
from cosmos_predict1.tokenizer.training.datasets.utils import (
    INPUT_KEY,
    MASK_KEY,
    RECON_KEY,
)


class NativeCropPerceptualLoss(PerceptualLoss):
    def __init__(self, config, crop_size):
        super().__init__(config)
        if crop_size < 32:
            raise ValueError("perceptual crop must be at least 32 pixels")
        self.crop_size = crop_size
        self.last_region = None

    def forward(self, inputs, output_batch, iteration):
        h, w = inputs[INPUT_KEY].shape[-2:]
        ch, cw = min(self.crop_size, h), min(self.crop_size, w)
        y = int(torch.randint(h - ch + 1, ()).item())
        x = int(torch.randint(w - cw + 1, ()).item())
        self.last_region = dict(
            y=y,
            x=x,
            height=ch,
            width=cw,
            source_height=h,
            source_width=w,
            strategy="uniform source pixels; shared across time; no resize",
        )
        region = (..., slice(y, y + ch), slice(x, x + cw))
        cropped = {
            INPUT_KEY: inputs[INPUT_KEY][region],
            MASK_KEY: inputs[MASK_KEY][region],
        }
        prediction = {RECON_KEY: output_batch[RECON_KEY][region]}
        return super().forward(cropped, prediction, iteration)
