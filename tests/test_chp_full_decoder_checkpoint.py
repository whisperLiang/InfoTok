"""Full temporal prefix checkpointing cannot sever RGB-to-slot gradients."""

from dataclasses import replace

import torch

from cosmos_predict1.tokenizer.progressive.config import CHPConfig
from cosmos_predict1.tokenizer.progressive.model import CHPVTokenizer


def test_full_prefix_checkpoint_equal_with_gradients():
    torch.manual_seed(2)
    model = CHPVTokenizer(
        CHPConfig(debug_backbone=True, hidden_dim=16, heads=2, checkpoint_decoder=False)
    )
    enabled = CHPVTokenizer(replace(model.config, checkpoint_decoder=True))
    enabled.load_state_dict(model.state_dict())
    x = torch.randn(1, 3, 9, 32, 64)
    a = model(x, budgets=[256, 512, 1024])
    b = enabled(x, budgets=[256, 512, 1024])
    torch.testing.assert_close(
        a["reconstructions"], b["reconstructions"], atol=0, rtol=0
    )
    a["reconstructions"].square().mean().backward()
    b["reconstructions"].square().mean().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            q = dict(enabled.named_parameters())[name]
            assert p.grad is not None
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
