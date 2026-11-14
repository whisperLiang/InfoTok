import hashlib
import json
from dataclasses import asdict, dataclass

FSQ_LEVELS = (8, 8, 8, 5, 5, 5)
CODEBOOK_SIZE = 64000


@dataclass(frozen=True)
class CHPConfig:
    kmax: int = 2048
    budgets: tuple[int, ...] = (256, 512, 1024, 2048)
    multiscale: bool = False
    base_tokens: int = 256
    structure_tokens: int = 768
    hidden_dim: int = 128
    heads: int = 4
    local_radius: int = 2
    # Global convolutional attention is replaced only for the new architecture.
    spatial_window: int = 16
    anchor_order: str = "halton"
    decoder_neighbors: int = 8
    checkpoint_decoder: bool = True
    freeze_backbone: bool = True
    backbone_checkpoint: str | None = None
    # Only used by tests; never presented as pretrained or trained weights.
    debug_backbone: bool = False

    def __post_init__(self):
        self._validate_budgets()
        self._validate_attention()
        if (self.base_tokens, self.structure_tokens) != (256, 768):
            raise ValueError("protocol v1 uses Base=256 and Structure=768")
        if self.anchor_order not in ("halton", "morton"):
            raise ValueError("unknown anchor order")

    def _validate_budgets(self):
        if self.kmax not in (2048, 4096, 8192):
            raise ValueError("Kmax must be 2048, 4096 or 8192")
        if not self.budgets or any(k < 256 or k > self.kmax for k in self.budgets):
            raise ValueError("invalid training budgets")

    def _validate_attention(self):
        if self.heads < 1 or self.hidden_dim < 1 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by positive heads")
        if (
            self.spatial_window < 1
            or self.local_radius < 0
            or self.decoder_neighbors < 1
        ):
            raise ValueError("attention bounds must be positive")

    def to_dict(self):
        result = asdict(self)
        result["budgets"] = list(self.budgets)
        return result

    @property
    def sha256(self):
        # A pathname is not model identity. The checkpoint itself is hashed separately.
        data = self.to_dict()
        data.pop("backbone_checkpoint")
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
