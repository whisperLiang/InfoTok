#!/usr/bin/env python3
"""Geometry-only anchor diagnostics; no reconstruction-quality inference."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cosmos_predict1.tokenizer.progressive.anchors import anchors, coverage

p = argparse.ArgumentParser()
p.add_argument("--output", default="results/chp_vtok/figures/anchors_final")
p.add_argument(
    "--figure-tools", default=str(Path.home() / ".codex/skills/nature-figure/scripts")
)
args = p.parse_args()
sys.path.insert(0, args.figure_tools)
from audit_panel_alignment import require_matplotlib_panel_alignment

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
rows = []
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
fig, axes = plt.subplots(2, 4, figsize=(180 / 25.4, 95 / 25.4))
fig.subplots_adjust(
    left=0.03, right=0.98, bottom=0.16, top=0.84, wspace=0.17, hspace=0.50
)
maps = []
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
        # Normalize by the mean anchor count so different budgets are comparable.
        maps.append((axes[y, x], heat / (k / 64)))
        axes[y, x].set_title(f"{order.title()}, K={k}")
        axes[y, x].set_xticks([])
        axes[y, x].set_yticks([])
maximum = max(float(heat.max()) for _, heat in maps)
for ax, heat in maps:
    ax.imshow(
        heat, origin="upper", cmap="viridis", aspect=1088 / 1920, vmin=0, vmax=maximum
    )
fig.suptitle("Native padded geometry: 1088 x 1920", fontsize=8, y=0.98)
fig.text(
    0.03,
    0.03,
    f"8 x 8 spatial bins; anchor count / mean count per bin. Shared color range: 0 to {maximum:.2f}.\nCoverage is a geometry diagnostic, not a reconstruction-quality claim.",
    fontsize=7,
)
require_matplotlib_panel_alignment(
    fig,
    axes=list(axes.flat),
    json_out=out / "anchor_coverage.alignment.json",
    panel_ids=list("abcdefgh"),
    strict=True,
)
fig.savefig(out / "anchor_coverage.pdf")
fig.savefig(out / "anchor_coverage.svg")
fig.savefig(out / "anchor_coverage.png", dpi=300)
(out / "anchor_coverage.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
