"""Pin temporal slicing math before bounding cuDNN's full-window workspace."""

import pytest
import torch
from cosmos_predict1.tokenizer.modules.layers3d import CausalConv3d
from cosmos_predict1.tokenizer.progressive.causal_state import bounded_convolution


@pytest.mark.parametrize(
    "kernel,stride,dilation", [(1, 1, 1), (3, 1, 1), (3, 2, 1), (5, 3, 2)]
)
def test_chunked_convolution_retains_stride_dilation_gradients(
    kernel, stride, dilation
):
    torch.manual_seed(4)
    m = CausalConv3d(
        4, 6, (kernel, 3, 3), time_stride=stride, time_dilation=dilation, padding=1
    ).double()
    x = torch.randn(2, 4, 17, 9, 13, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a = m(x)
    b = bounded_convolution(m, y, temporal_outputs=3)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    weight = torch.randn_like(a)
    ag = torch.autograd.grad((a * weight).sum(), (x, m.conv3d.weight, m.conv3d.bias))
    bg = torch.autograd.grad((b * weight).sum(), (y, m.conv3d.weight, m.conv3d.bias))
    for left, right in zip(ag, bg):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
