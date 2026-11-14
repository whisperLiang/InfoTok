---
license: apache-2.0
library_name: pytorch
tags:
  - video-tokenization
  - adaptive-tokenization
  - discrete-tokenizer
  - infotok
  - infotok-flex
  - video-reconstruction
  - cosmos
arxiv: "2512.16975"
---

<p align="center">
  <img src="assets/infotok-header.png" width="100%" alt="InfoTok adaptive video tokenization overview">
</p>

<div align="center">

# InfoTok-Flex

### Adaptive Discrete Video Tokenizer via Information-Theoretic Compression

**ICLR 2026 Oral**

[Project page](https://research.nvidia.com/labs/cosmos-lab/infotok/) ·
[Paper](https://arxiv.org/abs/2512.16975) ·
[Code](https://github.com/YWolfeee/InfoTok) ·
[TokenBench data](https://huggingface.co/datasets/qyoo/tokenbench_240p) ·
[DAVIS data](https://huggingface.co/datasets/qyoo/davis_240p)

</div>

Fixed-rate video tokenizers spend the same number of tokens on a static scene
and a rapidly changing one. **InfoTok** instead allocates token length according
to informational content: predictable regions receive fewer tokens, while
motion and fine detail receive more. An ELBO-based router estimates the budget,
and a transformer-based adaptive compressor turns fixed-length embeddings into
a variable-length sequence of discrete tokens.

This repository contains **InfoTok-Flex**, the flexible-router checkpoint
post-trained on 81-frame temporal windows at varied resolutions and on
non-square videos.

## At a glance

| Field | Released checkpoint |
|---|---|
| Weight file | `infotok_mse.pt` |
| Temporal window | 81 frames |
| Overlap used by reference inference | 3 frames |
| Token allocation | `elbo` per clip or `global_elbo` across frames |
| Average token-use control | `--avg_rate`, from 0.0625 to 1.0 |
| Reference architecture name | `OURS4x8x8-mse-256p-88` |
| Base implementation | NVIDIA Cosmos Predict1 tokenizer stack |

The checkpoint uses the native PyTorch format expected by the official
InfoTok implementation. Use it only through trusted code and a trusted
downloaded revision.

## Why adaptive tokenization?

The project is built around two questions:

1. **How many tokens should a video receive?** The router uses the base
   tokenizer's ELBO as a tractable proxy for predictability.
2. **How should a variable budget encode the video?** The adaptive compressor
   and decompressor learn to preserve the most useful information at the chosen
   token length.

The result is one tokenizer that can move continuously along the
compression–fidelity trade-off without training a separate model for every
rate.

## Installation and download

```bash
git clone https://github.com/YWolfeee/InfoTok.git
cd InfoTok

conda env create --file infotok.yaml
conda activate infotok
export PYTHONPATH="$PWD"
pip install -r requirements.txt

hf download qyoo/infotok-flex infotok_mse.pt \
  --local-dir checkpoints/infotok-flex
```

## Reconstruction

```bash
python -m cosmos_predict1.tokenizer.inference.video_cli \
  --video_pattern "/path/to/videos/*.mp4" \
  --checkpoint checkpoints/infotok-flex/infotok_mse.pt \
  --output_dir outputs/infotok \
  --tokenizer_type OURS4x8x8-mse-256p-88 \
  --temporal_window 81 \
  --overlap_window 3 \
  --strategy global_elbo \
  --avg_rate 0.5 \
  --mode torch
```

Use `global_elbo` to distribute one budget across temporal frames. With `elbo`,
each clip uses the requested average rate while the router still allocates
tokens adaptively inside that clip. The official repository also provides
`exp_scripts/infotok_inference.sh` and a token-mask visualization mode.

## Results

<p align="center">
  <img src="assets/infotok-table.png" width="100%" alt="InfoTok and InfoTok-Flex paper results on TokenBench and DAVIS">
</p>

The figure above is the paper's 256 × 256 comparison. For the released
81-frame checkpoint, the official codebase provides the following 240p
reconstruction targets:

| Dataset | Average token rate | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| TokenBench 240p | 0.75 | 29.7088 | 0.8786 |
| TokenBench 240p | 0.50 | 28.9674 | 0.8522 |
| DAVIS 240p | 0.75 | 26.1951 | 0.7994 |
| DAVIS 240p | 0.50 | 25.1283 | 0.7529 |

Metric reproduction depends on the published preprocessing, temporal window,
overlap, rate strategy, and TokenBench evaluation implementation.

## Intended use and limitations

InfoTok-Flex is intended for research on adaptive discrete video
representations, reconstruction, token allocation, and efficient downstream
video modeling. It is not a semantic video-understanding model or a perceptual
quality guarantee. Very aggressive compression can remove fine detail or rapid
motion, and behavior outside the documented temporal and resolution profile is
not characterized by this release. The model also inherits data and domain
biases from its base tokenizer and post-training corpus.

## Citation

```bibtex
@inproceedings{ye2026infotok,
  title     = {InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression},
  author    = {Ye, Haotian and He, Qiyuan and Han, Jiaqi and Li, Puheng and Fan, Jiaojiao and Hao, Zekun and Reda, Fitsum and Balaji, Yogesh and Chen, Huayu and Liu, Sheng and Yao, Angela and Zou, James and Ermon, Stefano and Wang, Haoxiang and Liu, Ming-Yu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

## License

Apache 2.0. See the official codebase's `ATTRIBUTIONS.md` for third-party
components and notices.
