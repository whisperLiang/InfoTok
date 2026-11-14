"""CHP-specific losses and training metadata within the existing TokenizerModel."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cosmos_predict1.tokenizer.progressive.model import file_sha256
from cosmos_predict1.tokenizer.training.model import PREDICTION, TokenizerModel
from cosmos_predict1.utils.callback import Callback


class CHPTrainingModel(TokenizerModel):
    def training_step(self, data_batch, iteration):
        output = self._network_forward(data_batch)
        video = data_batch[self.get_input_key(data_batch)]
        # Use upstream RGB / perceptual loss definitions, then add new bottleneck losses.
        from cosmos_predict1.tokenizer.training.datasets.utils import (
            INPUT_KEY,
            MASK_KEY,
        )

        inputs = {INPUT_KEY: video, MASK_KEY: torch.ones_like(video)}
        losses, value = self.loss(inputs, output, iteration)
        latent = output["latent_loss"] * self.config.latent_weight
        quant = output["quant_loss"]
        rgb = output["reconstructions"]
        temporal = (
            video.new_zeros(())
            if video.shape[2] == 1
            else (
                (rgb[:, :, 1:] - rgb[:, :, :-1]) - (video[:, :, 1:] - video[:, :, :-1])
            )
            .abs()
            .mean()
        )
        value = value + latent + quant + self.config.temporal_weight * temporal
        return {
            PREDICTION: rgb,
            "loss": {
                **losses["loss"],
                "latent": latent,
                "temporal": temporal,
                "quant": quant,
            },
            "budgets": output["allocated_ratios"] * self.network.config.kmax,
        }, value

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        # Reuse framework optimizer/scheduler construction; frozen backbone parameters
        # stay excluded. A fine-tuning stage can set a lower global LR in its config.
        optimizer_config.params = [
            p for p in self.network.parameters() if p.requires_grad
        ]
        from cosmos_predict1.utils.lazy_config import instantiate

        optimizer = instantiate(optimizer_config)
        scheduler_config.optimizer = optimizer
        return optimizer, instantiate(scheduler_config)


class CHPCheckpointMetadata(Callback):
    """Extend the existing checkpoint with exact RNG and content identities."""

    def on_save_checkpoint(self, model, state_dict):
        state_dict["rng"] = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
        )
        state_dict["chp_config"] = model.network.config.to_dict()
        state_dict["config_sha256"] = model.network.config.sha256
        state_dict["training_status"] = "engineering-smoke-until-validated"
        state_dict["global_step"] = state_dict["iteration"]
        config_file = Path(self.config.job.path_local) / "config.yaml"
        state_dict["training_config_sha256"] = file_sha256(config_file)

    def on_load_checkpoint(self, model, state_dict):
        if "chp_config" not in state_dict:
            raise ValueError(
                "CHP resume requires a CHP checkpoint; initialize the backbone in network.config"
            )
        if state_dict["config_sha256"] != model.network.config.sha256:
            raise ValueError("resume config hash mismatch")
        rng = state_dict["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])

    def on_save_checkpoint_success(self, iteration=0):
        path = (
            Path(self.trainer.checkpointer.checkpoint_dir_local)
            / f"iter_{iteration:09}.pt"
        )
        identity = dict(
            path=str(path.resolve()),
            checkpoint_sha256=file_sha256(path),
            global_step=iteration,
        )
        path.with_suffix(".sha256.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )

    def on_training_step_start(self, model, data_batch, iteration=0):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.step_start = time.perf_counter()

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        torch.cuda.synchronize()
        row = dict(
            iteration=iteration,
            loss=float(loss),
            budgets=output_batch["budgets"].tolist(),
            components={k: float(v) for k, v in output_batch["loss"].items()},
            seconds=time.perf_counter() - self.step_start,
            shape=list(data_batch["video"].shape),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        perceptual = (
            model.loss.loss_modules["perceptual"]
            if "perceptual" in model.loss.loss_modules
            else None
        )
        row["perceptual_region"] = getattr(perceptual, "last_region", None)
        with (Path(self.config.job.path_local) / "steps.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print("CHP_TRAIN_STEP " + json.dumps(row), flush=True)

    def on_after_backward(self, model, iteration=0):
        network = model.module.network if hasattr(model, "module") else model.network
        gradients = self._collect_gradients(network)
        latent_norm = (
            sum(v * v for k, v in gradients.items() if k.startswith("latent_decoders."))
            ** 0.5
        )
        if latent_norm == 0:
            raise RuntimeError("RGB decoder severed latent decoder gradients")
        row = dict(
            iteration=iteration,
            norm=sum(v * v for v in gradients.values()) ** 0.5,
            latent_decoder_norm=latent_norm,
        )
        print("CHP_GRADIENTS " + json.dumps(row), flush=True)

    @staticmethod
    def _collect_gradients(network):
        gradients = {
            name: float(p.grad.float().norm())
            for name, p in network.named_parameters()
            if p.grad is not None
        }
        if not gradients or not all(np.isfinite(v) for v in gradients.values()):
            raise FloatingPointError("missing or nonfinite progressive gradients")
        if network.config.freeze_backbone and any(
            p.grad is not None for p in network.decoder.parameters()
        ):
            raise RuntimeError("frozen decoder acquired parameter gradients")
        return gradients
