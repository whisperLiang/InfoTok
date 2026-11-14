"""Public decoder reconstruction and checkpoint boundary behavior (tiny test weights)."""

import dataclasses

import pytest
import torch

from cosmos_predict1.tokenizer.progressive.checkpoint import (
    export_decoder,
    load_checkpoint,
)
from cosmos_predict1.tokenizer.progressive.config import CHPConfig
from cosmos_predict1.tokenizer.progressive.model import (
    CHPVTokenizer,
    Geometry,
)
from cosmos_predict1.tokenizer.progressive.stream import (
    ProgressiveDecoder,
    ProgressiveEncoder,
)


def make_test_checkpoint(path, config=None):
    config = config or CHPConfig(
        debug_backbone=True, hidden_dim=16, heads=2, checkpoint_decoder=False
    )
    model = CHPVTokenizer(config).eval()
    torch.save(
        dict(
            model={"network." + k: v for k, v in model.state_dict().items()},
            chp_config=config.to_dict(),
            config_sha256=config.sha256,
            global_step=0,
        ),
        path,
    )
    return model


@torch.no_grad()
def test_public_decoder_export_matches_sender_and_contains_no_private_modules(tmp_path):
    source = tmp_path / "model.pt"
    make_test_checkpoint(source)
    public = tmp_path / "decoder.pt"
    export_decoder(source, public)
    sender, identity = load_checkpoint(source)
    receiver, other_identity = load_checkpoint(public, decoder_only=True)
    assert not hasattr(receiver, "encoder") and not hasattr(receiver, "compressors")
    receiver.eval()
    assert identity == other_identity
    ids = torch.randint(64000, (1, 512), dtype=torch.int32)
    geometry = Geometry(32, 64, 32, 64)
    a, _ = ProgressiveDecoder(sender, geometry).decode_bootstrap(ids)
    b, _ = ProgressiveDecoder(receiver, geometry).decode_bootstrap(ids)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    raw = torch.load(source, weights_only=False)
    raw["config_sha256"] = "bad"
    torch.save(raw, source)
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(source)


@pytest.mark.parametrize(
    "changes",
    [
        dict(kmax=10),
        dict(budgets=[]),
        dict(budgets=[256, 9000]),
        dict(base_tokens=1),
        dict(heads=0),
        dict(hidden_dim=7),
        dict(anchor_order="random"),
        dict(local_radius=-1),
    ],
)
def test_invalid_model_config_rejected(changes):
    with pytest.raises(ValueError):
        CHPConfig(**changes)


def test_config_identity_excludes_local_weight_path_but_includes_architecture():
    config = CHPConfig()
    assert (
        config.sha256
        == dataclasses.replace(
            config, backbone_checkpoint="/another/public/path"
        ).sha256
    )
    assert config.sha256 != dataclasses.replace(config, multiscale=True).sha256
    assert config.sha256 == CHPConfig(**config.to_dict()).sha256


@torch.no_grad()
def test_tail_initialization_and_ownership_guards(tmp_path):
    model = make_test_checkpoint(tmp_path / "test.pt")
    encoder = ProgressiveEncoder(model)
    x = torch.randn(1, 3, 4, 31, 63)
    ids, state = encoder.encode_bootstrap(x[:, :, :1])
    with pytest.raises(ValueError, match="bootstrap"):
        encoder.encode_bootstrap(x[:, :, :1], state)
    with pytest.raises(ValueError, match="four frames"):
        encoder.encode_block(x[:, :, 1:3], state)
    tail = encoder.encode_block(x[:, :, 1:4], state, eos=True)
    decoder = ProgressiveDecoder(model, state.geometry)
    first, ds = decoder.decode_bootstrap(ids[:, :256])
    last = decoder.decode_block(tail[:, :512], ds, frame_count=3, eos=True)
    assert torch.cat((first, last), 2).shape == x.shape
    with pytest.raises(ValueError, match="EOS"):
        encoder.encode_block(x, state)
    with pytest.raises(ValueError, match="EOS"):
        decoder.decode_block(tail, ds)


@pytest.mark.parametrize("kmax", [4096, 8192])
@torch.no_grad()
def test_extended_kmax_keeps_all_ids_discrete_and_prefixes_decodable(kmax, tmp_path):
    model = make_test_checkpoint(
        tmp_path / "extended.pt",
        CHPConfig(
            debug_backbone=True,
            hidden_dim=8,
            heads=2,
            kmax=kmax,
            budgets=(256, 512, 1024, 2048, kmax),
            checkpoint_decoder=False,
            multiscale=True,
        ),
    )
    ids, _ = model.encode(torch.randn(1, 3, 1, 32, 64))
    assert ids.shape == (1, 1, kmax)
    assert int(ids.min()) >= 0 and int(ids.max()) < 64000
    for k in (256, 512, 1024, 2048, kmax):
        assert model.dense_from_ids(ids[:, 0, :k], (4, 8)).shape == (1, 16, 4, 8)
