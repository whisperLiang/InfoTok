# CHP-VTok

This is an isolated experimental model in the InfoTok repository. It reuses the official factorized causal convolutional encoder/decoder, FSQ, losses, Trainer, AdamW/scheduler construction, and checkpoint manager. The pretrained ViT bottleneck is replaced with new learned modules; this is **not a numerically equivalent conversion of InfoTok-Flex**. Upstream models and configuration names remain available.

The current checkpoints are engineering smoke training on three short public clips, including native 1920×1080 optimization steps. They are not converged production or benchmark checkpoints. See [the measured technical report](results/chp_vtok/report.md) and its gate table before interpreting quality or performance.

## Model and geometry

RGB `[B,3,1+4n,H,W]` → official causal Encoder → F8 (optionally spatially pooled F16/F32) → fixed learned 1D queries over bounded anchor neighborhoods → official FSQ `[8,8,8,5,5,5]` → `[B,blocks,Kmax]` integer IDs. There are 64000 symbols; fixed packing is 16 bits/ID.

The receiver reconstructs from those integer prefixes using the same deterministic Halton or breadth-first Morton anchors, learned missing embeddings, bounded slot cross-attention and local refinement, followed by the official causal Decoder. ID zero is a real symbol. Dense latents and RGB are never sent. Native 1080p is padded to 1088×1920 and cropped after decoding.

Single-scale uses Kmax learned queries on F8. Multi-scale emits Base 256 (F32), Structure 768 (F16), then Detail Kmax−1024 (F8). Structure residuals use **decoded quantized Base**, and Detail residuals use **decoded quantized Base+Structure**. K=512 uses the first 256 Structure IDs immediately. All budgets slice the one maximum sequence; there is no budget-dependent re-encoding.

The new model uses bounded spatial attention windows. It preserves the original convolutional weights, but changes their spatial attention semantics and requires training. The separate SDPA edit to the original attention preserves its mathematical definition and parameters, with measured floating-point differences.

## Setup and tests

Use Python 3.10 and the pinned environment in `results/chp_vtok/dependencies.txt`. The original `requirements.txt` is untouched. Official weights are from `qyoo/infotok-flex` revision `7b9f0d4d5b7ce128e7de0d4252f34199b9ac25f5`; their SHA256 is `bd69629a8f8fcfc4808d29eec5bed68177e22aa14e3b9b799599791e26af935a`.

```bash
export PYTHONPATH=.
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/chp_prepare_data.py
```

The data preparation script reads existing public 1080p YUV sources without altering the benchmark repository and writes 17-frame lossless RGB clips under this checkout's ignored `data/chp_vtok/`. For another installation, supply your own video paths through `CHP_VIDEOS`.

## Training and resume

The standard InfoTok training entry point is used with a new config:

```bash
CHP_STAGE=A CHP_VIDEOS='data/chp_vtok/*.mp4' \
  bash exp_scripts/chp_vtok_train.sh job.name=single_native \
  trainer.max_iter=1000 checkpoint.save_iter=100
```

A=single-scale frozen backbone, B=multi-scale frozen backbone, C=multi-scale history curriculum, D=backbone fine-tuning at a smaller global learning rate. Independent historical prefix budgets and reference/display conditioning are enabled from A onward. C can increase the number of blocks without introducing a duplicate implementation.

`CHP_DEBUG_HEIGHT=256` explicitly enables a debug resize, never a final-resolution result. `CHP_NATIVE_CROP=512` samples true source pixels without resizing; crop geometry must be reported. Omit both for full native inputs. `CHP_LPIPS=0` explicitly disables perceptual loss for dependency/debug checks. The default uses the upstream perceptual loss, RGB, temporal differences and latent MSE; FSQ returns its native zero auxiliary quantization loss.

For the 17-frame native curriculum, full-frame perceptual training exceeded 80 GB. `CHP_LPIPS_CROP=256` instead evaluates the upstream perceptual training objective on a uniform 256×256 source-pixel region shared across frames, with its coordinates recorded at every step. The RGB output, RGB loss, temporal loss and final evaluation remain complete 1920×1080. This patch objective is not described as full-frame LPIPS training. Non-reentrant checkpointing also covers each stateless repeated-prefix RGB decoder call; inference cache mutation is never replayed during backward.

```bash
CHP_STAGE=B bash exp_scripts/chp_vtok_train.sh job.name=resume_multi \
  checkpoint.load_path=checkpoints/chp_vtok/tokenizer/multi_native/checkpoints/iter_000000042.pt \
  checkpoint.load_training_state=True trainer.max_iter=1042
```

Checkpoints contain network, optimizer, scheduler, scaler, Python/NumPy/CPU/CUDA RNG states, global step, model config hash, full training config hash, plus a checkpoint SHA sidecar. Resume restores optimization and RNG; changing the dataset/curriculum is intentional, and arbitrary dataloader mid-epoch replay is not claimed bit-identical.

Frozen Decoder parameters stay frozen while gradients flow from RGB loss to the new latent decoder. Non-reentrant activation checkpointing does not run stateful inference caches. Stage D is optional and is not a completed experimental result.

## Stateful inference

`ProgressiveEncoder.encode_bootstrap(frame)` initializes once; `encode_block(frames,state,eos=...)` consumes four *new* frames. A shorter final block requires EOS and repeats only its last available frame internally. The valid frame count is transmitted and decoded output is trimmed accordingly.

`ProgressiveDecoder.decode_bootstrap(ids)` and `decode_block(prefix,state,commit=...)` reconstruct from public weights and IDs only. `ReferenceState` commits once per block from acknowledged first 256 IDs. Display refinements fork this reference and cannot rewrite it. `DisplayContext` can carry historical enhancements through a separately trained projection. Encoder caches contain legal past activations; Decoder caches are derived only from base IDs.

The full graph, legal-history recomputation and cache reuse have separate measurements. The cache covers Haar patching, convolution stride/dilation phase, pooling, upsampling/unpatching and unlimited temporal attention KV history. High-resolution convolutions are sliced along exact output-time intervals to bound cuDNN workspace. This is mathematical equivalence, not guaranteed bitwise equivalence in low precision.

## Independent receiver and incremental packets

CHP2 sessions and CHPB packets are independent of upstream inference formats. They include model/checkpoint/config identities, geometry, timing, FSQ levels, uint16 IDs, frame counts, offsets, payload lengths and CRC32. A `block_end` flag finalizes the current prefix without reading a future packet. EOS is explicit. Partial tail increments carry a `tail` flag and their actual frame count; only the final increment sets EOS. Late, duplicated, skipped and mixed-identity tokens are rejected.

```bash
.venv/bin/python -m cosmos_predict1.tokenizer.progressive.cli export-decoder \
  --checkpoint checkpoints/chp_vtok/tokenizer/multi_native/checkpoints/iter_000000042.pt \
  --output checkpoints/chp_vtok/multi_decoder.pt

.venv/bin/python -m cosmos_predict1.tokenizer.progressive.cli send \
  --checkpoint checkpoints/chp_vtok/tokenizer/multi_native/checkpoints/iter_000000042.pt \
  --video data/chp_vtok/mcl_jcv01.mp4 --frames 17 \
  --stream results/chp_vtok/example.chp --budgets 256,512,1024,2048

.venv/bin/python scripts/chp_isolated_receiver.py \
  --checkpoint checkpoints/chp_vtok/multi_decoder.pt \
  --stream results/chp_vtok/example.chp --output results/chp_vtok/reconstructions/example
```

The exported decoder excludes encoder and compressor weights. The audit wrapper denies reads of repository videos and private sender tensors. It verifies process independence, not a hostile operating-system sandbox.

For actual ordered TCP, start receiver with `--stream tcp://127.0.0.1:29641`, then run sender with the same endpoint and `--log-prefix results/chp_vtok/tcp_run`. `--block-budgets 256,512,1024,2048,256` gives independent final budgets to bootstrap and four blocks, while sending only legal increments within each block. These are loopback measurements; no WAN or real-time source pacing claim is implied.

## Measured experiments

```bash
.venv/bin/python scripts/chp_run.py --gpu --name native_consistency -- \
  .venv/bin/python scripts/chp_consistency.py \
  --checkpoint checkpoints/chp_vtok/tokenizer/multi_native/checkpoints/iter_000000042.pt \
  --video data/chp_vtok/mcl_jcv01.mp4 --frames 17 --dtype bfloat16 \
  --output results/chp_vtok/native_consistency.json

.venv/bin/python scripts/chp_run.py --gpu --name native_rd -- \
  .venv/bin/python scripts/chp_evaluate.py \
  --checkpoint checkpoints/chp_vtok/tokenizer/multi_native/checkpoints/iter_000000042.pt \
  --video data/chp_vtok/mcl_jcv01.mp4 --frames 17 --history-intervention \
  --output results/chp_vtok/reconstructions/new_native_rd
```

`chp_run.py` saves verbatim argument arrays, code SHA, dirty state, timestamps, logs and exit codes. `--gpu` serializes GPU experiments so memory/latency comparisons are not contaminated by another local experiment. Output directories and stream files refuse to overwrite existing runs. Quality values, including negative incremental gains, are preserved. PSNR/SSIM/LPIPS/DISTS are computed from complete native frames; training-loss perceptual features are not mislabeled as the standardized evaluation LPIPS metric.
