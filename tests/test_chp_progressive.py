import io

import pytest
import torch

from cosmos_predict1.tokenizer.progressive.anchors import anchors, coverage
from cosmos_predict1.tokenizer.progressive.config import CHPConfig
from cosmos_predict1.tokenizer.progressive.model import CHPVTokenizer, Geometry
from cosmos_predict1.tokenizer.progressive.protocol import (
    Packet,
    PrefixReceiver,
    Session,
)
from cosmos_predict1.tokenizer.progressive.quantizer import make_fsq
from cosmos_predict1.tokenizer.progressive.stream import (
    ProgressiveDecoder,
    ProgressiveEncoder,
)


@pytest.fixture
def model():
    torch.manual_seed(5)
    return CHPVTokenizer(
        CHPConfig(
            debug_backbone=True,
            hidden_dim=16,
            heads=2,
            local_radius=1,
            checkpoint_decoder=False,
        )
    ).eval()


def session():
    return Session("test", "a" * 64, "b" * 64, 64, 32, 64, 32)


def receiver(s):
    return PrefixReceiver(
        s, model_id="test", checkpoint_sha256="a" * 64, config_sha256="b" * 64
    )


def test_fsq_all_64000_ids_are_reversible():
    q = make_fsq()
    ids = torch.arange(64000, dtype=torch.int32)[None]
    torch.testing.assert_close(
        q.codes_to_indices(q.indices_to_codes(ids)), ids, atol=0, rtol=0
    )
    assert (q.indices_to_codes(ids[:, :1]) != 0).any()


@pytest.mark.parametrize("order", ["halton", "morton"])
def test_rectangular_anchor_prefix_coverage(order):
    coords, _ = anchors(68, 120, 768, order)
    assert torch.unique(coords, dim=0).shape[0] == 768
    assert torch.equal(coords[:256], anchors(68, 120, 256, order)[0])
    stats = coverage(68, 120, 256, order)
    assert stats["covered_partition_fraction"] == 1.0


@torch.no_grad()
def test_encoder_causality_and_exact_prefix(model):
    x = torch.randn(1, 3, 9, 32, 64)
    y = x.clone()
    y[:, :, 5:] += 3
    ids, geometry = model.encode(x)
    altered, _ = model.encode(y)
    assert torch.equal(ids[:, :2], altered[:, :2])
    assert ids.shape == (1, 3, 2048)
    for low, high in zip((256, 512, 1024), (512, 1024, 2048)):
        assert torch.equal(ids[:, :, :low], ids[:, :, :high][:, :, :low])
    assert geometry == Geometry(32, 64, 32, 64)


@torch.no_grad()
def test_decode_uses_only_prefix_zero_is_valid(model):
    zero = torch.zeros(1, 256, dtype=torch.int32)
    one = zero.clone()
    one[:, 0] = 1
    z = model.dense_from_ids(zero, (4, 8))
    assert not torch.equal(z, model.dense_from_ids(one, (4, 8)))
    assert not torch.equal(
        z,
        model.dense_from_codes(model.quantizer.indices_to_codes(zero[:, :255]), (4, 8)),
    )
    with pytest.raises(ValueError):
        model.dense_from_ids(torch.tensor([[64000]]), (4, 8))


@torch.no_grad()
def test_stream_full_encoder_and_decoder_alignment(model):
    x = torch.randn(1, 3, 9, 32, 64)
    full_ids, geometry = model.encode(x)
    sender = ProgressiveEncoder(model)
    first, state = sender.encode_bootstrap(x[:, :, :1])
    streamed = [
        first,
        sender.encode_block(x[:, :, 1:5], state),
        sender.encode_block(x[:, :, 5:9], state, eos=True),
    ]
    assert torch.equal(full_ids, torch.stack(streamed, 1))
    assert state.backbone.input_frames == 9
    assert state.backbone.bytes > 0
    decoder = ProgressiveDecoder(model, geometry)
    budget = [256, 512, 1024]
    r0, recv = decoder.decode_bootstrap(first[:, :256])
    outputs = [r0]
    for i in (1, 2):
        prefix = streamed[i][:, : budget[i]]
        outputs.append(decoder.decode_block(prefix, recv, commit=True))
    result = torch.cat(outputs, 2)
    full = model.decode(full_ids, geometry, budget)
    assert result.shape == x.shape
    torch.testing.assert_close(result, full, atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_decoder_future_intervention_and_display_reference(model):
    ids = torch.randint(64000, (1, 3, 2048), dtype=torch.int32)
    altered = ids.clone()
    altered[:, 2] = torch.randint(64000, (1, 2048), dtype=torch.int32)
    geometry = Geometry(32, 64, 32, 64)
    a, b = model.decode(ids, geometry), model.decode(altered, geometry)
    torch.testing.assert_close(a[:, :, :5], b[:, :, :5], atol=0, rtol=0)
    decoder = ProgressiveDecoder(model, geometry)
    first, state = decoder.decode_bootstrap(ids[:, 0, :256], commit=False)
    assert state.reference.version == -1
    for k in (512, 1024, 2048):
        decoder.decode_block(ids[:, 0, :k], state, frame_count=1, commit=False)
        assert state.reference.version == -1 and state.reference.backbone.bytes == 0
    decoder.decode_block(ids[:, 0], state, frame_count=1)
    assert state.reference.version == 0


def test_gradient_survives_frozen_rgb_decoder(model):
    model.train()
    x = torch.randn(1, 3, 5, 32, 64)
    output = model(x, budgets=[256, 512])
    loss = (output["reconstructions"] - x).square().mean()
    loss.backward()
    assert all(p.grad is None for p in model.decoder.parameters())
    grads = [p.grad for p in model.latent_decoders.parameters() if p.grad is not None]
    assert grads and sum(g.abs().sum().item() for g in grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert output["token_ids"].dtype == torch.int32


def test_incremental_wire_bytes_crc_identity_offsets():
    s = session()
    wire = s.to_bytes()
    assert Session.read(io.BytesIO(wire)) == s
    r = receiver(s)
    ids = tuple(range(2048))
    offset = 0
    for k in (256, 512, 1024, 2048):
        packet = Packet(0, 0, 1, offset, ids[offset:k], k == 2048)
        data = packet.to_bytes()
        r.append_tokens(Packet.read(io.BytesIO(data)))
        wire += data
        offset = k
    assert r.blocks[0] == list(ids)
    assert r.actual_bytes == len(wire)
    assert r.actual_bytes > 4096
    with pytest.raises(ValueError, match="EOS"):
        r.append_tokens(Packet(0, 0, 1, 2048, (0,)))
    bad = bytearray(wire[len(s.to_bytes()) :])
    bad[35] ^= 1
    with pytest.raises(ValueError, match="CRC32"):
        Packet.read(io.BytesIO(bad))
    with pytest.raises(ValueError, match="identity"):
        PrefixReceiver(
            s, model_id="other", checkpoint_sha256="a" * 64, config_sha256="b" * 64
        )
    r = receiver(s)
    with pytest.raises(ValueError, match="prefix"):
        r.append_tokens(Packet(0, 0, 1, 1, (0,)))
    r.append_tokens(Packet(0, 0, 1, 0, tuple([0] * 256)))
    with pytest.raises(ValueError, match="offset"):
        r.append_tokens(Packet(0, 0, 1, 0, (1,)))
    with pytest.raises(ValueError, match="FSQ"):
        Packet(0, 0, 1, 0, (64000,))


@torch.no_grad()
def test_quantized_multiscale_staircase_reference():
    m = CHPVTokenizer(
        CHPConfig(
            debug_backbone=True,
            hidden_dim=16,
            heads=2,
            multiscale=True,
            checkpoint_decoder=False,
            local_radius=1,
        )
    ).eval()
    features = torch.randn(1, 16, 4, 8)
    ids, codes, _, refs = m.compress_features(features, return_references=True)
    for i, k in enumerate((256, 1024, 2048)):
        receiver_dense = m.dense_from_ids(ids[:, :k], (4, 8))
        torch.testing.assert_close(refs[i], receiver_dense, atol=0, rtol=0)
    for k in (1, 255, 256, 512, 1024, 1500, 2048):
        assert torch.isfinite(m.dense_from_ids(ids[:, :k], (4, 8))).all()


@pytest.mark.parametrize("multiscale", [False, True])
@pytest.mark.parametrize("budget", [256, 512, 1024, 2048])
def test_every_training_budget_has_finite_loss_discrete_ids_and_effective_gradients(
    multiscale, budget
):
    model = CHPVTokenizer(
        CHPConfig(
            debug_backbone=True,
            hidden_dim=16,
            heads=2,
            multiscale=multiscale,
            checkpoint_decoder=True,
        )
    ).train()
    video = torch.randn(1, 3, 5, 32, 64)
    output = model(video, budgets=[budget, budget])
    loss = (output["reconstructions"] - video).abs().mean() + output["latent_loss"]
    assert torch.isfinite(loss)
    assert output["token_ids"].dtype == torch.int32
    assert 0 <= output["token_ids"].min() <= output["token_ids"].max() < 64000
    loss.backward()
    for modules in (model.compressors, model.latent_decoders):
        gradients = [p.grad for p in modules.parameters() if p.grad is not None]
        assert gradients and sum(g.abs().sum() for g in gradients) > 0
        assert all(torch.isfinite(g).all() for g in gradients)
    assert all(p.grad is None for p in model.decoder.parameters())
