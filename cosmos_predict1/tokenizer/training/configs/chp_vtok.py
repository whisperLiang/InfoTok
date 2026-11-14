"""Use the official training entry point, Trainer, AdamW, scheduler and checkpointer.

CHP_STAGE=A single-scale; B multi-scale; C randomized history; D backbone finetune.
All stages sample independent historical budgets. Debug resizing is opt-in.
"""

import os

import attrs

import torch
from torch.utils.data import DataLoader

from cosmos_predict1.tokenizer.progressive.config import CHPConfig
from cosmos_predict1.tokenizer.progressive.model import CHPVTokenizer
from cosmos_predict1.tokenizer.training.callbacks import GradClipCallback
from cosmos_predict1.tokenizer.training.chp_model import (
    CHPCheckpointMetadata,
    CHPTrainingModel,
)
from cosmos_predict1.tokenizer.training.configs.base.loss import (
    ColorConfig,
    PerceptualConfig,
)
from cosmos_predict1.tokenizer.training.configs.base.model import ModelConfig
from cosmos_predict1.tokenizer.training.datasets.chp_video import CHPVideoDataset
from cosmos_predict1.tokenizer.training.losses.continuous import (
    ColorLoss,
    PerceptualLoss,
    TokenizerLoss,
)
from cosmos_predict1.tokenizer.training.metrics import TokenizerMetric
from cosmos_predict1.tokenizer.training.trainer import TokenizerTrainer
from cosmos_predict1.utils.callback import LowPrecisionCallback, ProgressBarCallback
from cosmos_predict1.utils.config import Config, EMAConfig, JobConfig
from cosmos_predict1.utils.lazy_config import LazyCall as L
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.utils.scheduler import WarmupCosineLR


@attrs.define(slots=False)
class CHPModelConfig(ModelConfig):
    latent_weight: float = 1.0
    temporal_weight: float = 0.1


def make_config():
    stage = os.environ.get("CHP_STAGE", "A")
    if stage not in "ABCD":
        raise ValueError("CHP_STAGE must be A/B/C/D")
    net = CHPConfig(
        multiscale=stage != "A",
        freeze_backbone=stage != "D",
        backbone_checkpoint=os.environ.get(
            "CHP_BACKBONE", "checkpoints/infotok-flex/infotok_mse.pt"
        ),
    )
    loss_config = LazyDict(
        dict(color=L(ColorLoss)(config=ColorConfig()), reduce="MEAN")
    )
    if os.environ.get("CHP_LPIPS", "1") == "1":
        loss_config.perceptual = L(PerceptualLoss)(
            config=PerceptualConfig(
                gram_enabled=False,
                checkpoint_activations=True,
                lpips_boundaries=[0],
                lpips_values=[0.1],
            )
        )
    model = L(CHPTrainingModel)(
        config=CHPModelConfig(
            network=L(CHPVTokenizer)(config=net.to_dict()),
            loss=L(TokenizerLoss)(config=loss_config),
            metric=L(TokenizerMetric)(config=LazyDict({})),
            ema=EMAConfig(enabled=False),
            precision="bfloat16",
        )
    )
    height = os.environ.get("CHP_DEBUG_HEIGHT")
    crop = os.environ.get("CHP_NATIVE_CROP")
    dataset = L(CHPVideoDataset)(
        video_pattern=os.environ.get("CHP_VIDEOS", "data/chp_vtok/*.mp4"),
        num_video_frames=17,
        debug_height=int(height) if height else None,
        crop_size=int(crop) if crop else None,
    )
    loader = L(DataLoader)(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
    config = Config(
        model=model,
        optimizer=L(torch.optim.AdamW)(params=None, lr=1e-5 if stage == "D" else 1e-4),
        scheduler=L(WarmupCosineLR)(
            optimizer=None, warmup_iters=5, lr_decay_iters=10000, min_lr=1e-6
        ),
        dataloader_train=loader,
        dataloader_val=loader,
        job=JobConfig(project="chp_vtok", group="tokenizer", name=f"stage_{stage}"),
    )
    config.trainer.type = TokenizerTrainer
    config.trainer.max_iter = 1000
    config.trainer.run_validation = False
    config.trainer.logging_iter = 1
    config.trainer.ddp.find_unused_parameters = True
    config.trainer.ddp.static_graph = False
    config.trainer.cudnn.benchmark = False
    config.trainer.cudnn.deterministic = True
    config.trainer.callbacks = dict(
        low_precision=L(LowPrecisionCallback)(update_iter=1),
        grad_clip=L(GradClipCallback)(
            grad_clip_norm=1, config=None, trainer=None, verbose=False
        ),
        progress=L(ProgressBarCallback)(),
        chp_metadata=L(CHPCheckpointMetadata)(),
    )
    config.checkpoint.jit.enabled = False
    config.checkpoint.save_iter = 100
    config.checkpoint.strict_resume = True
    return config
