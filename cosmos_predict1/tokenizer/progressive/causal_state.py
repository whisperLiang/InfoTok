"""Exact incremental execution for InfoTok's FACTORIZED 4x8x8 convolutional graph.

Dispatch is installed on a CHP model instance only. Upstream models, parameter
names, and their TorchScript exports are untouched. No finite attention horizon.
RGB history is never retained. KV history is retained at temporal attention sites.
This inference state must not be used inside activation checkpoint recomputation.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MethodType

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from cosmos_predict1.tokenizer.modules.layers3d import (
    CausalAttnBlock,
    CausalConv3d,
    CausalHybridDownsample3d,
    CausalHybridUpsample3d,
    CausalResnetBlockFactorized3d,
    CausalTemporalAttnBlock,
)
from cosmos_predict1.tokenizer.modules.patching import Patcher3D, UnPatcher3D
from cosmos_predict1.tokenizer.modules.utils import (
    batch2space,
    batch2time,
    space2batch,
    time2batch,
)

_ACTIVE = ContextVar("chp_stream", default=None)


@dataclass
class StreamState:
    cache: dict = field(default_factory=dict)
    calls: int = 0
    input_frames: int = 0

    @property
    def bytes(self):
        def size(v):
            if torch.is_tensor(v):
                return v.numel() * v.element_size()
            if isinstance(v, (dict, tuple, list)):
                return sum(size(x) for x in (v.values() if isinstance(v, dict) else v))
            return 0

        return size(self.cache)

    def fork(self):
        # Inference cache entries are immutable tensors; operations always replace entries.
        return StreamState(dict(self.cache), self.calls, self.input_frames)

    @contextmanager
    def activate(self):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "StreamState is inference-only; train with the full causal graph"
            )
        if _ACTIVE.get() is not None:
            raise RuntimeError("nested stream execution")
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)
        self.calls += 1

    def conv(self, name, m, x):
        """Track the next convolution window's absolute start, including stride phase.

        Upstream left pad = dilation*(kernel-1)+1-stride. The cache retains
        exactly input >= next window start, not an unconditional kernel-1 tail.
        """
        old = self.cache.get(name)
        if old is None:
            padding = x[:, :, :1].expand(-1, -1, m.time_pad, -1, -1)
            buffer = torch.cat((padding, x), 2)
        else:
            buffer = torch.cat((old, x), 2)
        receptive = m.conv3d.dilation[0] * (m.conv3d.kernel_size[0] - 1) + 1
        stride = m.conv3d.stride[0]
        count = max(0, (buffer.shape[2] - receptive) // stride + 1)
        if count == 0:
            raise ValueError(
                f"chunk too short at {name}; supply aligned bootstrap/4-frame chunks"
            )
        out = m.conv3d(
            F.pad(buffer, m.spatial_pad + (0, 0), mode=m.pad_mode, value=0.0)
        )
        self.cache[name] = buffer[:, :, count * stride :].detach().clone()
        return out

    def downsample(self, name, m, x):
        if m.spatial_down:
            x = F.pad(x, (0, 1, 0, 1, 0, 0))
            x = m.conv1(x) + F.avg_pool3d(x, (1, 2, 2), (1, 2, 2))
        if m.temporal_down:
            if name not in self.cache:
                x = torch.cat((x[:, :, :1], x), 2)
                self.cache[name] = True
            # Aligned chunks yield an even pool input after the one-time bootstrap.
            if x.shape[2] % 2:
                raise ValueError("temporal pool phase requires an aligned chunk")
            x = m.conv2(x) + F.avg_pool3d(x, (2, 1, 1), (2, 1, 1))
        return m.conv3(x) if m.spatial_down or m.temporal_down else x

    def upsample(self, name, m, x):
        if m.temporal_up:
            first = name not in self.cache
            self.cache[name] = True
            x = x.repeat_interleave(2, 2)
            if first:
                x = x[:, :, 1:]
            x = m.conv1(x) + x
        if m.spatial_up:
            x = x.repeat_interleave(2, 3).repeat_interleave(2, 4)
            x = m.conv2(x) + x
        return m.conv3(x) if m.spatial_up or m.temporal_up else x

    def patch(self, name, m, x):
        first = name not in self.cache
        self.cache[name] = True
        if first:
            x = torch.cat(
                (x[:, :, :1].repeat_interleave(m.patch_size, 2), x[:, :, 1:]), 2
            )
        if m.patch_method == "haar":
            for _ in m.range:
                x = m._dwt(x, "haar", rescale=True)
            return x
        return rearrange(
            x,
            "b c (t p) (h q) (w r) -> b (c p q r) t h w",
            p=m.patch_size,
            q=m.patch_size,
            r=m.patch_size,
        )

    def unpatch(self, name, m, x):
        first = name not in self.cache
        self.cache[name] = True
        if m.patch_method == "haar":
            for _ in m.range:
                x = m._idwt(x, "haar", rescale=True)
        else:
            x = rearrange(
                x,
                "b (c p q r) t h w -> b c (t p) (h q) (w r)",
                p=m.patch_size,
                q=m.patch_size,
                r=m.patch_size,
            )
        return x[:, :, m.patch_size - 1 :] if first else x

    def temporal_attention(self, name, m, x):
        normalized = m.norm(x)
        q, b, height = space2batch(m.q(normalized))
        k, _, _ = space2batch(m.k(normalized))
        v, _, _ = space2batch(m.v(normalized))
        q, k, v = [a.transpose(1, 2).unsqueeze(1) for a in (q, k, v)]
        old = self.cache.get(name)
        history = 0 if old is None else old[0].shape[2]
        if old is not None:
            k, v = torch.cat((old[0], k), 2), torch.cat((old[1], v), 2)
        self.cache[name] = (k.detach(), v.detach())
        # Non-square causal SDPA's upper-left alignment would be wrong with a KV cache.
        allowed = torch.arange(k.shape[2], device=x.device)[None] <= (
            history + torch.arange(q.shape[2], device=x.device)[:, None]
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        return x + m.proj_out(batch2space(out.squeeze(1).transpose(1, 2), b, height))


# Separate architecture change: bounded spatial windows for CHP only.
def window_attention(m, x, window):
    z = m.norm(x)
    q, b = time2batch(m.q(z))
    k, _ = time2batch(m.k(z))
    v, _ = time2batch(m.v(z))
    n, c, h, w = q.shape
    ph, pw = (-h) % window, (-w) % window

    def partition(a):
        return rearrange(
            F.pad(a, (0, pw, 0, ph)),
            "b c (h u) (w v) -> (b h w) 1 (u v) c",
            u=window,
            v=window,
        )

    valid = torch.ones(n, 1, h, w, device=x.device, dtype=torch.bool)
    allowed = partition(valid).squeeze(-1).unsqueeze(-2)
    out = F.scaled_dot_product_attention(
        partition(q), partition(k), partition(v), attn_mask=allowed, dropout_p=0.0
    )
    out = rearrange(
        out.squeeze(1),
        "(b h w) (u v) c -> b c (h u) (w v)",
        b=n,
        h=(h + ph) // window,
        w=(w + pw) // window,
        u=window,
        v=window,
    )
    return x + m.proj_out(batch2time(out[:, :, :h, :w], b))


def install_dispatch(root, *, window=0, checkpoint_blocks=False):
    """Install once on a private CHP backbone instance; keep the state_dict intact."""
    methods = {
        CausalConv3d: "conv",
        CausalHybridDownsample3d: "downsample",
        CausalHybridUpsample3d: "upsample",
        Patcher3D: "patch",
        UnPatcher3D: "unpatch",
        CausalTemporalAttnBlock: "temporal_attention",
    }
    for name, m in root.named_modules():
        if getattr(m, "_chp_dispatch", False):
            continue
        method = methods.get(type(m))
        local = isinstance(m, CausalAttnBlock) and window > 0
        protect = checkpoint_blocks and isinstance(
            m,
            (
                CausalResnetBlockFactorized3d,
                CausalAttnBlock,
                CausalTemporalAttnBlock,
                CausalHybridUpsample3d,
            ),
        )
        if method is None and not local and not protect:
            continue
        m.forward = MethodType(
            _bind_forward(name, method, m.forward, local, protect, window), m
        )
        m._chp_dispatch = True


def _bind_forward(name, method, original, local, protect, window):
    def forward(self, x):
        state = _ACTIVE.get()
        if state is not None and method is not None:
            return getattr(state, method)(name, self, x)

        def compute(value):
            return window_attention(self, value, window) if local else original(value)

        if protect and state is None and torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(compute, x, use_reentrant=False)
        return compute(x)

    return forward
