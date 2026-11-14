"""CHP-VTok over the official InfoTok causal convolutional backbone.

The pretrained ViT and 2D bottleneck are deliberately not on the new model's
path. Their replacement is a separately trained 1D bottleneck, not an equivalent
checkpoint conversion. No dense latent is serialized or required by a receiver.
"""

import hashlib
from collections import namedtuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from cosmos_predict1.tokenizer.networks.configs import ours_discrete_video_4x8x8_256p_88
from cosmos_predict1.tokenizer.networks.ours_discrete_video import (
    OursDiscreteVideoTokenizer,
)

from .causal_state import install_dispatch
from .config import CHPConfig
from .feature_pyramid import feature_pyramid
from .latent_decoder import PrefixLatentDecoder
from .quantizer import make_fsq, quantize, validate_ids
from .slot_compressor import SlotCompressor

Evaluation = namedtuple(
    "CHPEvaluation",
    "reconstructions quant_loss latent allocated_ratios token_ids latent_loss",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Geometry:
    height: int
    width: int
    padded_height: int
    padded_width: int

    def __post_init__(self):
        if min(self.height, self.width) < 1:
            raise ValueError("invalid dimensions")
        if (
            self.padded_height != (self.height + 31) // 32 * 32
            or self.padded_width != (self.width + 31) // 32 * 32
        ):
            raise ValueError("CHP requires the canonical padding to multiples of 32")

    @classmethod
    def from_video(cls, x):
        h, w = x.shape[-2:]
        return cls(h, w, (h + 31) // 32 * 32, (w + 31) // 32 * 32)

    @property
    def latent_shape(self):
        return self.padded_height // 8, self.padded_width // 8

    def pad(self, x):
        if tuple(x.shape[-2:]) != (self.height, self.width):
            raise ValueError("geometry changed inside a session")
        return F.pad(
            x, (0, self.padded_width - self.width, 0, self.padded_height - self.height)
        )

    def crop(self, x):
        return x[..., : self.height, : self.width]


def official_backbone(checkpoint=None, debug=False):
    config = dict(ours_discrete_video_4x8x8_256p_88)
    if debug:
        config.update(
            channels=8,
            channels_mult=[1, 2, 2],
            num_res_blocks=1,
            z_channels=16,
            use_vit=False,
            attn_resolutions=[],
        )
    model = OursDiscreteVideoTokenizer(**config)
    if checkpoint is not None:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw = raw.get("model", raw)
        weights = {
            k.removeprefix("network."): v
            for k, v in raw.items()
            if k.startswith("network.")
        }
        result = model.load_state_dict(weights, strict=False)
        # Official checkpoints omit fixed patch/FSQ buffers, but no learned weight may be absent.
        missing_parameters = set(result.missing_keys) & set(
            dict(model.named_parameters())
        )
        if missing_parameters or result.unexpected_keys:
            raise ValueError(
                f"checkpoint mismatch: missing={missing_parameters}, unexpected={result.unexpected_keys}"
            )
    return model


class CHPVTokenizer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = (
            config if isinstance(config, CHPConfig) else CHPConfig(**(config or {}))
        )
        cfg = self.config
        original = official_backbone(cfg.backbone_checkpoint, cfg.debug_backbone)
        # Reuse objects and their weights, do not copy the implementations or keep unused ViTs.
        self.encoder, self.decoder = original.encoder, original.decoder
        self.channels = 16 if cfg.debug_backbone else 256
        self.quantizer = make_fsq()
        counts = (
            [cfg.base_tokens, cfg.structure_tokens, cfg.kmax - 1024]
            if cfg.multiscale
            else [cfg.kmax]
        )
        self.strides = [32, 16, 8] if cfg.multiscale else [8]
        self.compressors = nn.ModuleList(
            [SlotCompressor(self.channels, k, cfg) for k in counts]
        )
        self.latent_decoders = nn.ModuleList(
            [PrefixLatentDecoder(self.channels, k, cfg) for k in counts]
        )
        self.reference_projection = nn.Conv2d(
            self.channels, self.channels, 1, bias=False
        )
        self.display_projection = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        nn.init.zeros_(self.reference_projection.weight)
        nn.init.zeros_(self.display_projection.weight)
        if cfg.freeze_backbone:
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
        install_dispatch(self.encoder, window=cfg.spatial_window)
        install_dispatch(
            self.decoder,
            window=cfg.spatial_window,
            checkpoint_blocks=cfg.checkpoint_decoder,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            for name in ("encoder", "decoder"):
                module = self._modules.get(name)
                if module is not None:
                    module.eval()
        return self

    def backbone_encode(self, x):
        # Frozen encoder needs no gradients; frozen decoder MUST retain latent gradients.
        if self.config.freeze_backbone:
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)

    def compress_features(self, f8, return_references=False):
        pyramid = feature_pyramid(f8)
        codes_all, ids_all, references = [], [], []
        reconstruction = torch.zeros_like(f8)
        losses = []
        for stride, compressor, decoder in zip(
            self.strides, self.compressors, self.latent_decoders
        ):
            target = pyramid[stride]
            # Quantized and receiver-recoverable previous groups only. No private
            # unquantized base features are used to predict a residual reference.
            reference = F.adaptive_avg_pool2d(reconstruction, target.shape[-2:])
            ids, codes, loss = quantize(self.quantizer, compressor(target - reference))
            delta = decoder(codes, target.shape[-2:])
            reconstruction = reconstruction + F.interpolate(
                delta, size=f8.shape[-2:], mode="bilinear", align_corners=False
            )
            ids_all.append(ids)
            codes_all.append(codes)
            losses.append(loss)
            references.append(reconstruction)
        result = torch.cat(ids_all, 1), torch.cat(codes_all, 1), sum(losses)
        return (*result, references) if return_references else result

    def dense_from_codes(self, codes, shape):
        b, received, _ = codes.shape
        h, w = shape
        dense = codes.new_zeros(b, self.channels, h, w).to(
            self.latent_decoders[0].slot.dtype
        )
        offset = 0
        for stride, decoder in zip(self.strides, self.latent_decoders):
            group = codes[:, offset : min(received, offset + decoder.count)]
            if group.shape[1]:
                divisor = stride // 8
                delta = decoder(
                    group, ((h + divisor - 1) // divisor, (w + divisor - 1) // divisor)
                )
                dense = dense + F.interpolate(
                    delta, size=(h, w), mode="bilinear", align_corners=False
                )
            offset += decoder.count
        return dense

    def dense_from_ids(self, ids, shape):
        validate_ids(ids)
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.config.kmax:
            raise ValueError("expected a nonempty [B,K] prefix")
        codes = self.quantizer.indices_to_codes(ids.to(torch.int32))
        return self.dense_from_codes(codes, shape)

    def condition(self, dense, reference=None, display=None):
        if reference is not None:
            dense = dense + self.reference_projection(reference)
        if display is not None:
            dense = dense + self.display_projection(display)
        return dense

    def encode(self, video):
        if video.ndim != 5 or video.shape[1] != 3 or (video.shape[2] - 1) % 4:
            raise ValueError(
                "encode expects [B,3,1+4n,H,W]; use stream API for EOS tails"
            )
        geometry = Geometry.from_video(video)
        features = self.backbone_encode(geometry.pad(video))
        ids = [self.compress_features(frame)[0] for frame in features.unbind(2)]
        return torch.stack(ids, 1), geometry

    def decode(self, tokens, geometry, budgets=None):
        if tokens.ndim != 3:
            raise ValueError("tokens must be [B,blocks,K] including bootstrap")
        budgets = budgets or [tokens.shape[-1]] * tokens.shape[1]
        if len(budgets) != tokens.shape[1] or any(
            k < 256 or k > tokens.shape[-1] for k in budgets
        ):
            raise ValueError("each committed block requires a received base prefix")
        reference, display, dense, bases = None, None, [], []
        for t, k in enumerate(budgets):
            prefix = tokens[:, t, :k]
            current = self.condition(
                self.dense_from_ids(prefix, geometry.latent_shape), reference, display
            )
            reference = self.condition(
                self.dense_from_ids(prefix[:, :256], geometry.latent_shape), reference
            )
            display = current
            dense.append(current)
            bases.append(reference)
        return self.decode_latent_sequence(dense, bases, geometry)

    def decode_latent_sequence(self, dense, bases, geometry):
        # Reference semantics: historical backbone activations derive only from
        # base IDs. This full causal reference implementation repeats legal
        # history; the stream API instead reuses exactly those base caches.
        frames = []
        for t, current in enumerate(dense):
            latent = torch.stack(bases[:t] + [current], 2)
            if (
                self.config.checkpoint_decoder
                and torch.is_grad_enabled()
                and latent.requires_grad
            ):
                # The repeated full-prefix reference graph otherwise retains
                # multiple high-resolution decoder graphs simultaneously.
                frame = checkpoint(
                    self._decode_current, latent, geometry, t, use_reentrant=False
                )
            else:
                frame = self._decode_current(latent, geometry, t)
            frames.append(frame)
        return torch.cat(frames, 2)

    def _decode_current(self, latent, geometry, block):
        decoded = geometry.crop(self.decoder(latent))
        return decoded if block == 0 else decoded[:, :, -4:]

    def forward(self, video, budgets=None):
        if (video.shape[2] - 1) % 4:
            raise ValueError("training requires bootstrap + complete four-frame blocks")
        geometry = Geometry.from_video(video)
        features = self.backbone_encode(geometry.pad(video))
        steps = features.shape[2]
        if budgets is None:
            # Independently sample each historical block, not only the last block.
            budgets = [
                self.config.budgets[
                    int(torch.randint(len(self.config.budgets), ()).item())
                ]
                for _ in range(steps)
            ]
        if len(budgets) != steps or any(
            k < 256 or k > self.config.kmax for k in budgets
        ):
            raise ValueError("invalid per-block budgets")
        ids_all, dense_all, base_all, quant_losses = [], [], [], []
        reference, display = None, None
        for frame, k in zip(features.unbind(2), budgets):
            ids, codes, quant_loss = self.compress_features(frame)
            dense = self.condition(
                self.dense_from_codes(codes[:, :k], geometry.latent_shape),
                reference,
                display,
            )
            reference = self.condition(
                self.dense_from_codes(codes[:, :256], geometry.latent_shape), reference
            )
            display = dense
            dense_all.append(dense)
            base_all.append(reference)
            ids_all.append(ids)
            quant_losses.append(quant_loss)
        dense = torch.stack(dense_all, 2)
        # No no_grad / detach here: frozen RGB decoder backpropagates to new modules.
        reconstruction = self.decode_latent_sequence(dense_all, base_all, geometry)
        result = dict(
            reconstructions=reconstruction,
            quant_loss=sum(quant_losses),
            latent=dense,
            allocated_ratios=video.new_tensor(budgets)[None] / self.config.kmax,
            token_ids=torch.stack(ids_all, 1),
            latent_loss=F.mse_loss(dense.float(), features.float()),
        )
        return result if self.training else Evaluation(**result)
