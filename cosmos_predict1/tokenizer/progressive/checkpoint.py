"""Training checkpoints use InfoTok's manager; inference exports only public decoding weights."""

from dataclasses import replace

import torch

from .config import CHPConfig
from .model import CHPVTokenizer, file_sha256


def load_checkpoint(path, *, decoder_only=False, device="cpu", dtype=torch.float32):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    config = CHPConfig(**raw["chp_config"])
    # Construction must not depend on the sender's original checkpoint pathname.
    config = replace(config, backbone_checkpoint=None)
    model = CHPVTokenizer(config)
    decoder_only = decoder_only or raw.get("kind") == "chp_decoder"
    if decoder_only:
        del model.encoder
        del model.compressors
    if "model" in raw:
        weights = {
            k.removeprefix("network."): v
            for k, v in raw["model"].items()
            if k.startswith("network.")
        }
    else:
        weights = raw["network"]
    if decoder_only:
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith(("encoder.", "compressors."))
        }
    result = model.load_state_dict(weights, strict=False)
    missing = set(result.missing_keys) & set(dict(model.named_parameters()))
    if missing or result.unexpected_keys:
        raise ValueError(
            f"CHP checkpoint mismatch: {missing}, {result.unexpected_keys}"
        )
    _validate_identity(raw, config)
    identity = raw.get("source_checkpoint_sha256", file_sha256(path))
    return model.to(device=device, dtype=dtype).eval(), identity


def _validate_identity(raw, config):
    if raw["config_sha256"] != config.sha256:
        raise ValueError("checkpoint config identity mismatch")


def export_decoder(checkpoint_path, destination):
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = {
        k.removeprefix("network."): v
        for k, v in raw["model"].items()
        if k.startswith("network.")
    }
    weights = {
        k: v
        for k, v in weights.items()
        if not k.startswith(("encoder.", "compressors."))
    }
    torch.save(
        dict(
            kind="chp_decoder",
            network=weights,
            chp_config=raw["chp_config"],
            config_sha256=raw["config_sha256"],
            source_checkpoint_sha256=file_sha256(checkpoint_path),
            global_step=raw["global_step"],
            training_status=raw.get("training_status"),
        ),
        destination,
    )
