"""Native input pixels and explicit debug/crop training configuration boundaries."""

import cv2
import numpy as np
import pytest
import torch

from cosmos_predict1.tokenizer.progressive.cli import read_rgb
from cosmos_predict1.tokenizer.training.configs.chp_vtok import make_config
from cosmos_predict1.tokenizer.training.datasets.chp_video import CHPVideoDataset


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "source.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 32))
    assert writer.isOpened()
    generator = np.random.default_rng(7)
    for _ in range(5):
        writer.write(generator.integers(0, 256, (32, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def test_native_pixels_debug_resize_and_real_pixel_crop(video):
    dataset = CHPVideoDataset(video, num_video_frames=5)
    assert len(dataset) == 1
    native = dataset[0]["video"]
    decoded, _ = read_rgb(video, 5)
    torch.testing.assert_close(native, decoded[0], atol=0, rtol=0)
    debug = CHPVideoDataset(video, num_video_frames=5, debug_height=16)[0]["video"]
    assert debug.shape == (3, 5, 16, 32)
    crop = CHPVideoDataset(video, num_video_frames=5, crop_size=16)[0]["video"]
    assert crop.shape == (3, 5, 16, 16)
    assert any(
        torch.equal(crop, native[..., y : y + 16, x : x + 16])
        for y in range(17)
        for x in range(49)
    )
    with pytest.raises(ValueError, match="crop"):
        CHPVideoDataset(video, num_video_frames=5, crop_size=64)[0]
    with pytest.raises(ValueError, match="shorter"):
        CHPVideoDataset(video, num_video_frames=9)[0]


def test_dataset_rejects_empty_source_and_invalid_temporal_geometry(tmp_path, video):
    with pytest.raises(ValueError, match="no videos"):
        CHPVideoDataset(str(tmp_path / "missing*.mp4"))
    with pytest.raises(ValueError, match="1\\+4n"):
        CHPVideoDataset(video, num_video_frames=4)


@pytest.mark.parametrize("stage", ["A", "B", "C", "D"])
def test_stage_configuration_uses_original_trainer_and_declared_backbone_policy(
    stage, monkeypatch
):
    monkeypatch.setenv("CHP_STAGE", stage)
    config = make_config()
    assert config.trainer.type.__name__ == "TokenizerTrainer"
    network = config.model.config.network.config
    assert network.multiscale == (stage != "A")
    assert network.freeze_backbone == (stage != "D")
    assert "perceptual" in config.model.config.loss.config


def test_debug_and_native_crop_policies_are_explicit(monkeypatch):
    monkeypatch.setenv("CHP_STAGE", "B")
    monkeypatch.setenv("CHP_DEBUG_HEIGHT", "256")
    monkeypatch.setenv("CHP_NATIVE_CROP", "128")
    monkeypatch.setenv("CHP_LPIPS", "0")
    config = make_config()
    assert config.dataloader_train.dataset.debug_height == 256
    assert config.dataloader_train.dataset.crop_size == 128
    assert "perceptual" not in config.model.config.loss.config
    monkeypatch.setenv("CHP_LPIPS_CROP", "256")
    config = make_config()
    assert config.model.config.loss.config.perceptual.crop_size == 256
    monkeypatch.setenv("CHP_STAGE", "E")
    with pytest.raises(ValueError, match="CHP_STAGE"):
        make_config()
