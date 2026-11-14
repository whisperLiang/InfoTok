"""Pin upstream attention math before substituting memory-efficient SDPA."""

import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from cosmos_predict1.tokenizer.modules.layers3d import (
    CausalAttnBlock,
    CausalTemporalAttnBlock,
)
from cosmos_predict1.tokenizer.modules.utils import CausalNormalize


def explicit_attention(module, x):
    h = module.norm(x)
    q, k, v = [getattr(module, name)(h) for name in ("q", "k", "v")]
    b, c, t, h, w = q.shape
    temporal = isinstance(module, CausalTemporalAttnBlock)
    if temporal:
        q, k, v = [a.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c) for a in (q, k, v)]
    else:
        q, k, v = [a.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c) for a in (q, k, v)]
    weights = (q @ k.transpose(-1, -2)) * c**-0.5
    if temporal:
        allowed = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        weights = weights.masked_fill(~allowed, -torch.inf)
    out = weights.softmax(-1) @ v
    out = (
        out.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
        if temporal
        else out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)
    )
    return x + module.proj_out(out)


@pytest.mark.parametrize("cls", [CausalAttnBlock, CausalTemporalAttnBlock])
def test_sdpa_is_used_without_parameter_changes(cls):
    m = cls(16, 1)
    keys = tuple(m.state_dict())
    with patch.object(
        F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
    ) as spy:
        m(torch.randn(1, 16, 3, 4, 6))
        assert spy.call_count == 1
    assert tuple(m.state_dict()) == keys


@pytest.mark.parametrize("cls", [CausalAttnBlock, CausalTemporalAttnBlock])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attention_forward_backward(cls, dtype):
    if dtype != torch.float32 and not torch.cuda.is_available():
        pytest.skip("CUDA low precision test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)
    m = cls(32, 1).to(device=device, dtype=dtype)
    reference = copy.deepcopy(m)
    x = torch.randn(2, 32, 5, 4, 6, device=device, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    y, expected = m(x), explicit_attention(reference, x_ref)
    tolerance = {torch.float32: 3e-6, torch.float16: 3e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(y, expected, atol=tolerance, rtol=tolerance)
    y.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad, atol=tolerance, rtol=tolerance)
    for p, p_ref in zip(m.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, p_ref.grad, atol=tolerance, rtol=tolerance)


def test_group_one_normalization_has_no_future_statistics():
    m = CausalNormalize(16, num_groups=1)
    x = torch.randn(1, 16, 9, 4, 6)
    changed = x.clone()
    changed[:, :, 5:] += 50
    torch.testing.assert_close(m(x)[:, :, :5], m(changed)[:, :, :5], atol=0, rtol=0)


def test_upstream_positional_to_preserves_quantizer_dtype():
    from cosmos_predict1.tokenizer.progressive.model import official_backbone

    model = official_backbone(debug=True).to(torch.float32).eval()
    with torch.no_grad():
        ids, codes, _ = model.encode(torch.randn(1, 3, 5, 32, 64))
        assert codes.dtype == torch.float32
        assert model.decode(codes).shape == (1, 3, 5, 32, 64)
