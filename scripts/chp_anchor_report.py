#!/usr/bin/env python3
"""Geometry-only anchor diagnostics; no reconstruction-quality inference."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_predict1.tokenizer.progressive.anchors import anchors, coverage

out = Path("results/chp_vtok/figures")
out.mkdir(parents=True, exist_ok=True)
rows = []
fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
for y, order in enumerate(("halton", "morton")):
    for x, k in enumerate((256, 512, 1024, 2048)):
        points = []
        for scale, h, w, start, count in [
            (32, 34, 60, 0, 256),
            (16, 68, 120, 256, 768),
            (8, 136, 240, 1024, 1024),
        ]:
            received = max(0, min(count, k - start))
            if received:
                stats = coverage(h, w, received, order)
                stats.update(budget=k, scale=scale)
                rows.append(stats)
                points.append(anchors(h, w, received, order)[1].numpy())
        p = np.concatenate(points)
        heat, _, _ = np.histogram2d(
            p[:, 0], p[:, 1], bins=(8, 8), range=((0, 1), (0, 1))
        )
        axes[y, x].imshow(heat, origin="upper", cmap="viridis", aspect=16 / 9)
        axes[y, x].set_title(f"{order}, K={k}, occupied={np.mean(heat > 0):.0%}")
        axes[y, x].set_xticks([])
        axes[y, x].set_yticks([])
fig.suptitle("1088 x 1920 geometry; token-anchor counts in 8 x 8 partitions")
fig.savefig(out / "anchor_coverage.png", dpi=160)
Path("results/chp_vtok/anchor_coverage.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
print(json.dumps(rows, indent=2))
