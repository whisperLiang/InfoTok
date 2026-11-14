#!/usr/bin/env bash
set -u
export PYTHONPATH=.
for CHP_VIDEO in uvg_beauty basketball; do
  for CHP_SCALE in single multi; do
    .venv/bin/python scripts/chp_run.py --gpu --name "rd_${CHP_SCALE}_${CHP_VIDEO}" -- \
      .venv/bin/python scripts/chp_evaluate.py \
      --checkpoint "checkpoints/chp_vtok/tokenizer/${CHP_SCALE}_native/checkpoints/iter_000000042.pt" \
      --video "data/chp_vtok/${CHP_VIDEO}.mp4" --frames 17 \
      --output "results/chp_vtok/reconstructions/rd_${CHP_SCALE}_${CHP_VIDEO}"
  done
done
.venv/bin/python scripts/chp_run.py --gpu --name morton_debug_training -- \
  env CHP_STAGE=B CHP_DEBUG_HEIGHT=256 CHP_LPIPS=1 OMP_NUM_THREADS=4 WANDB_MODE=offline \
  bash exp_scripts/chp_vtok_train.sh job.name=morton_debug trainer.max_iter=40 checkpoint.save_iter=40 \
  model.config.network.config.anchor_order=morton \
  dataloader_train.dataset.num_video_frames=5 dataloader_val.dataset.num_video_frames=5
.venv/bin/python scripts/chp_run.py --gpu --name morton_native_training -- \
  env CHP_STAGE=B CHP_LPIPS=1 OMP_NUM_THREADS=4 WANDB_MODE=offline \
  bash exp_scripts/chp_vtok_train.sh job.name=morton_native trainer.max_iter=42 checkpoint.save_iter=42 \
  model.config.network.config.anchor_order=morton \
  checkpoint.load_path=checkpoints/chp_vtok/tokenizer/morton_debug/checkpoints/iter_000000040.pt \
  checkpoint.load_training_state=True \
  dataloader_train.dataset.num_video_frames=5 dataloader_val.dataset.num_video_frames=5
.venv/bin/python scripts/chp_run.py --gpu --name rd_morton_mcl -- \
  .venv/bin/python scripts/chp_evaluate.py \
  --checkpoint checkpoints/chp_vtok/tokenizer/morton_native/checkpoints/iter_000000042.pt \
  --video data/chp_vtok/mcl_jcv01.mp4 --frames 17 \
  --output results/chp_vtok/reconstructions/rd_morton_mcl
.venv/bin/python scripts/chp_run.py --gpu --name warm_attention_uncontended -- \
  .venv/bin/python scripts/chp_numerics.py --video data/chp_vtok/mcl_jcv01.mp4 \
  --output results/chp_vtok/attention_uncontended.json
