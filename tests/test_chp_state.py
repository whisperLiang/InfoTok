import copy

import pytest
import torch

from cosmos_predict1.tokenizer.modules.layers3d import (
    CausalConv3d,
)
from cosmos_predict1.tokenizer.progressive.causal_state import (
    StreamState,
    install_dispatch,
)
from cosmos_predict1.tokenizer.progressive.model import official_backbone


@pytest.mark.parametrize(
    "kernel,dilation,stride", [(3, 1, 1), (5, 2, 1), (3, 1, 2), (5, 2, 3), (1, 1, 1)]
)
@torch.no_grad()
def test_stride_dilation_cache_phase(kernel, dilation, stride):
    m = CausalConv3d(
        2, 2, (kernel, 1, 1), time_stride=stride, time_dilation=dilation, padding=0
    )
    x = torch.randn(1, 2, 25, 2, 3)
    expected = m(x)
    install_dispatch(m)
    state = StreamState()
    outputs = []
    # Sufficient aligned/un-aligned chunks, including a nontrivial stride phase.
    for chunk in x.split([7, 8, 10], 2):
        with state.activate():
            outputs.append(m(chunk))
    torch.testing.assert_close(torch.cat(outputs, 2), expected, atol=2e-6, rtol=2e-6)
    assert state.bytes < x.numel() * x.element_size()


@torch.no_grad()
def test_official_graph_stream_and_recompute_low_precision_guard():
    torch.manual_seed(1)
    original = official_backbone(debug=True).eval()
    encoder, decoder = original.encoder, original.decoder
    x = torch.randn(1, 3, 17, 32, 64)
    z = encoder(x)
    expected = decoder(z)
    install_dispatch(encoder)
    install_dispatch(decoder)
    enc, dec = StreamState(), StreamState()
    encoded, decoded = [], []
    for i, frames in enumerate(
        [x[:, :, :1]] + [x[:, :, j : j + 4] for j in (1, 5, 9, 13)]
    ):
        with enc.activate():
            part = encoder(frames)
        assert part.shape[2] == 1
        encoded.append(part)
        with dec.activate():
            decoded.append(decoder(part))
        assert decoded[-1].shape[2] == (1 if i == 0 else 4)
    torch.testing.assert_close(torch.cat(encoded, 2), z, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(torch.cat(decoded, 2), expected, atol=2e-5, rtol=2e-5)
    with pytest.raises(RuntimeError, match="inference-only"):
        with torch.enable_grad(), enc.activate():
            pass


def test_checkpoint_preserves_latent_gradient_and_forward():
    torch.manual_seed(3)
    ordinary = official_backbone(debug=True).decoder.requires_grad_(False)
    protected = copy.deepcopy(ordinary)
    install_dispatch(protected, checkpoint_blocks=True)
    x = torch.randn(1, 16, 2, 4, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a, b = ordinary(x), protected(y)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
    assert y.grad.abs().sum() > 0
