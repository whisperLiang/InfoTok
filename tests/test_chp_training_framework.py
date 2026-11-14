"""Exercise upstream training integration and the extra recoverable metadata."""

from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from cosmos_predict1.tokenizer.progressive.config import CHPConfig
from cosmos_predict1.tokenizer.progressive.model import official_backbone
from cosmos_predict1.tokenizer.training.chp_model import (
    CHPTrainingModel,
    CHPCheckpointMetadata,
)
from cosmos_predict1.tokenizer.training.configs.chp_vtok import CHPModelConfig
from cosmos_predict1.tokenizer.training.configs.base.loss import ColorConfig
from cosmos_predict1.tokenizer.training.losses.continuous import (
    ColorLoss,
    TokenizerLoss,
)
from cosmos_predict1.tokenizer.training.metrics import TokenizerMetric
from cosmos_predict1.tokenizer.progressive.model import CHPVTokenizer
from cosmos_predict1.utils.config import EMAConfig
from cosmos_predict1.utils.lazy_config import LazyCall as L, LazyDict
from cosmos_predict1.utils.scheduler import WarmupCosineLR


def test_framework_losses_optimizer_gradients_and_rng_resume(tmp_path, monkeypatch):
    config = CHPModelConfig(
        network=L(CHPVTokenizer)(
            config=CHPConfig(debug_backbone=True, hidden_dim=16, heads=2).to_dict()
        ),
        loss=L(TokenizerLoss)(
            config=LazyDict(
                dict(color=L(ColorLoss)(config=ColorConfig()), reduce="MEAN")
            )
        ),
        metric=L(TokenizerMetric)(config=LazyDict({})),
        ema=EMAConfig(enabled=False),
        precision="float32",
    )
    model = CHPTrainingModel(config).train()
    output, loss = model.training_step({"video": torch.randn(1, 3, 5, 32, 64)}, 0)
    assert torch.isfinite(loss) and output["prediction"].shape == (1, 3, 5, 32, 64)
    loss.backward()
    assert CHPCheckpointMetadata._collect_gradients(model.network)
    optimizer, scheduler = model.init_optimizer_scheduler(
        L(torch.optim.AdamW)(params=None, lr=1e-4),
        L(WarmupCosineLR)(
            optimizer=None, warmup_iters=2, lr_decay_iters=10, min_lr=1e-6
        ),
    )
    optimizer.step()
    scheduler.step()
    assert optimizer.state_dict()["state"]
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda state: None)
    (tmp_path / "config.yaml").write_text("training: test\n")
    cb = CHPCheckpointMetadata()
    cb.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    cb.on_after_backward(model, iteration=1)
    state = {"iteration": 1}
    cb.on_save_checkpoint(model, state)
    assert state["global_step"] == 1 and len(state["training_config_sha256"]) == 64
    expected = (random.random(), np.random.random(), torch.rand(2))
    cb.on_load_checkpoint(model, state)
    actual = (random.random(), np.random.random(), torch.rand(2))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    with pytest.raises(ValueError, match="resume"):
        cb.on_load_checkpoint(model, {})
    state["config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        cb.on_load_checkpoint(model, state)
    for p in model.network.parameters():
        p.grad = None
    with pytest.raises(FloatingPointError, match="missing"):
        CHPCheckpointMetadata._collect_gradients(model.network)


def test_original_checkpoint_loader_requires_all_learned_parameters(tmp_path):
    original = official_backbone(debug=True)
    path = tmp_path / "upstream.pt"
    weights = {"network." + k: v for k, v in original.state_dict().items()}
    torch.save({"model": weights}, path)
    loaded = official_backbone(str(path), debug=True)
    assert torch.equal(
        original.encoder.conv_in[0].conv3d.weight,
        loaded.encoder.conv_in[0].conv3d.weight,
    )
    del weights["network.encoder.conv_in.0.conv3d.weight"]
    torch.save({"model": weights}, path)
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        official_backbone(str(path), debug=True)


def test_debug_loss_without_perceptual_still_records_training_step(
    tmp_path, monkeypatch
):
    import time

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    cb = CHPCheckpointMetadata()
    cb.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    cb.step_start = time.perf_counter()
    model = SimpleNamespace(loss=SimpleNamespace(loss_modules={}))
    cb.on_training_step_end(
        model,
        {"video": torch.zeros(1, 3, 1, 32, 64)},
        {"budgets": torch.tensor([[256]]), "loss": {"color": torch.tensor(0.1)}},
        torch.tensor(0.1),
        iteration=1,
    )
    import json

    row = json.loads((tmp_path / "steps.jsonl").read_text())
    assert row["perceptual_region"] is None and row["iteration"] == 1


def test_optional_backbone_finetuning_receives_rgb_gradients():
    model = CHPVTokenizer(
        CHPConfig(debug_backbone=True, hidden_dim=16, heads=2, freeze_backbone=False)
    ).train()
    video = torch.randn(1, 3, 5, 32, 64)
    output = model(video, budgets=[256, 512])
    (output["reconstructions"] - video).square().mean().backward()
    for backbone in (model.encoder, model.decoder):
        gradients = [p.grad for p in backbone.parameters() if p.grad is not None]
        assert gradients and sum(g.abs().sum() for g in gradients) > 0
        assert all(torch.isfinite(g).all() for g in gradients)
