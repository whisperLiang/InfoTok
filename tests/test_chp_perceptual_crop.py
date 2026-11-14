"""Reuse official perceptual features on actual selected pixels, with input gradients."""

import torch

from cosmos_predict1.tokenizer.training.chp_losses import NativeCropPerceptualLoss
from cosmos_predict1.tokenizer.training.configs.base.loss import PerceptualConfig
from cosmos_predict1.tokenizer.training.datasets.utils import (
    INPUT_KEY,
    MASK_KEY,
    RECON_KEY,
)
from cosmos_predict1.tokenizer.training.losses.continuous import PerceptualLoss


def test_native_crop_is_exact_pixel_selection_and_preserves_rgb_gradient():
    torch.manual_seed(5)
    config = PerceptualConfig(
        gram_enabled=False,
        lpips_boundaries=[0],
        lpips_values=[0.1],
        checkpoint_activations=True,
    )
    loss = NativeCropPerceptualLoss(config, crop_size=32)
    x = torch.randn(1, 3, 1, 64, 96)
    reconstruction = torch.randn_like(x, requires_grad=True)
    output = loss(
        {INPUT_KEY: x, MASK_KEY: torch.ones_like(x)}, {RECON_KEY: reconstruction}, 0
    )["lpips"]
    region = loss.last_region
    assert region["height"] == 32 and region["width"] == 32
    expected = PerceptualLoss.forward(
        loss,
        {
            INPUT_KEY: x[
                ..., region["y"] : region["y"] + 32, region["x"] : region["x"] + 32
            ],
            MASK_KEY: torch.ones(1, 3, 1, 32, 32),
        },
        {
            RECON_KEY: reconstruction[
                ..., region["y"] : region["y"] + 32, region["x"] : region["x"] + 32
            ]
        },
        0,
    )["lpips"]
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    output.mean().backward()
    assert (
        reconstruction.grad.abs().sum() > 0
        and torch.isfinite(reconstruction.grad).all()
    )
    assert reconstruction.grad.count_nonzero() <= 3 * 32 * 32
