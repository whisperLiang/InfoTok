"""Independent sender / receiver states for ordered progressive transmission."""

from dataclasses import dataclass, field

import torch

from .causal_state import StreamState
from .model import Geometry
from .quantizer import validate_ids


@dataclass
class EncoderState:
    owner: int | None = None
    geometry: Geometry | None = None
    backbone: StreamState = field(default_factory=StreamState)
    next_block: int = 0
    eos: bool = False


@dataclass
class ReferenceState:
    backbone: StreamState = field(default_factory=StreamState)
    latent: torch.Tensor | None = None
    version: int = -1


@dataclass
class DisplayContext:
    latent: torch.Tensor | None = None
    budget: int = 0


@dataclass
class DecoderState:
    owner: int | None = None
    reference: ReferenceState = field(default_factory=ReferenceState)
    display: DisplayContext = field(default_factory=DisplayContext)
    next_block: int = 0
    eos: bool = False


class ProgressiveEncoder:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def encode_bootstrap(self, frame, state=None):
        state = state or EncoderState()
        if state.next_block != 0 or frame.ndim != 5 or frame.shape[2] != 1:
            raise ValueError("bootstrap must be exactly one frame, initialized once")
        state.owner = id(self.model)
        state.geometry = Geometry.from_video(frame)
        return self._encode(frame, state, False), state

    @torch.no_grad()
    def encode_block(self, frames, encoder_state, *, eos=False):
        if encoder_state.next_block == 0 or encoder_state.owner != id(self.model):
            raise ValueError("bootstrap or model identity mismatch")
        count = frames.shape[2]
        if not 1 <= count <= 4 or (count < 4 and not eos):
            raise ValueError("four frames required, except explicit EOS tail")
        return self._encode(frames, encoder_state, eos)

    def _encode(self, frames, state, eos):
        if state.eos:
            raise ValueError("encoder already reached EOS")
        count = frames.shape[2]
        frames = state.geometry.pad(frames)
        if state.next_block and count < 4:
            frames = torch.cat(
                (frames, frames[:, :, -1:].expand(-1, -1, 4 - count, -1, -1)), 2
            )
        with state.backbone.activate():
            features = self.model.encoder(frames)
        if features.shape[2] != 1:
            raise RuntimeError("backbone temporal phase mismatch")
        ids = self.model.compress_features(features[:, :, 0])[0]
        state.backbone.input_frames += count
        state.next_block += 1
        state.eos = eos
        return ids


class ProgressiveDecoder:
    """The only input data is integer prefixes and public geometry/weights.

    Reference caches commit once, from the first 256 IDs only. Repeated display
    refinements fork the same pre-block cache and cannot change reference version.
    Historical display enhancements can condition the next latent via a learned
    projection, never the acknowledged base convolution cache.
    """

    def __init__(self, model, geometry):
        self.model, self.geometry = model, geometry

    @torch.no_grad()
    def decode_bootstrap(self, tokens, decoder_state=None, *, commit=True):
        state = decoder_state or DecoderState(owner=id(self.model))
        if state.next_block != 0:
            raise ValueError("bootstrap already decoded")
        return self.decode_block(tokens, state, frame_count=1, commit=commit), state

    def _validate_decode(self, prefix, state, frame_count, commit, eos):
        if state.owner is None:
            state.owner = id(self.model)
        if state.owner != id(self.model) or state.eos:
            raise ValueError("decoder identity mismatch or EOS")
        validate_ids(prefix)
        self._validate_frame_count(state.next_block, frame_count, eos)
        if commit and prefix.shape[1] < 256:
            raise ValueError(
                "the base prefix must arrive before advancing to the next block"
            )

    @staticmethod
    def _validate_frame_count(block, frame_count, eos):
        if block == 0:
            if frame_count != 1:
                raise ValueError("bootstrap is required")
            return
        if frame_count == 4:
            return
        if not eos or not 1 <= frame_count < 4:
            raise ValueError("invalid EOS/frame count")

    @torch.no_grad()
    def decode_block(
        self, prefix, decoder_state, *, frame_count=4, commit=True, eos=False
    ):
        state = decoder_state
        self._validate_decode(prefix, state, frame_count, commit, eos)
        dense = self.model.condition(
            self.model.dense_from_ids(prefix, self.geometry.latent_shape),
            state.reference.latent,
            state.display.latent,
        )
        display_cache = state.reference.backbone.fork()
        with display_cache.activate():
            rgb = self.geometry.crop(self.model.decoder(dense.unsqueeze(2)))[
                :, :, :frame_count
            ]
        if commit:
            base = self.model.condition(
                self.model.dense_from_ids(prefix[:, :256], self.geometry.latent_shape),
                state.reference.latent,
            )
            with state.reference.backbone.activate():
                self.model.decoder(base.unsqueeze(2))
            state.reference.latent = base
            state.reference.version += 1
            state.display = DisplayContext(dense, prefix.shape[1])
            state.next_block += 1
            state.eos = eos
        return rgb
