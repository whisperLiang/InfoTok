#!/usr/bin/env python3
"""Measured rate/distortion; no monotonic fitting or invented uncertainty.

Figure contract: do additional *actual bytes* improve early CHP reconstructions,
including failures of monotonicity and multi-scale versus single-scale? This
engineering quantitative grid has four complementary distortion definitions.
Every point is retained; separate thin video curves expose between-clip variation.
The thick mean is descriptive only (three related smoke clips, no seed CI).
Exports: 180x135 mm, editable PDF/SVG plus PNG; minimum text 7 pt. Python only.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def export_figure(figure, path):
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=300)


p = argparse.ArgumentParser()
p.add_argument("--root", default="results/chp_vtok")
p.add_argument("--metrics-root", help="explicit source of completed quality records")
p.add_argument(
    "--figure-tools", default=str(Path.home() / ".codex/skills/nature-figure/scripts")
)
a = p.parse_args()
sys.path.insert(0, a.figure_tools)
from audit_panel_alignment import require_matplotlib_panel_alignment

root = Path(a.root)
figures = root / "figures"
figures.mkdir(parents=True, exist_ok=True)
rows = []
metrics_root = Path(a.metrics_root) if a.metrics_root else root / "reconstructions"
for path in sorted(metrics_root.glob("rd_*/metrics.json")):
    rows.extend(json.loads(path.read_text()))
if not rows:
    raise ValueError("no completed measured rate-distortion runs")
if any(row["actual_bytes"] <= 0 for row in rows):
    raise ValueError("log-scale byte coordinates must be strictly positive")
fields = [k for k in rows[0] if k not in ("per_frame",)]
fields = sorted(set(fields) | set(k for r in rows for k in r if k != "per_frame"))
with (root / "summary.csv").open("w") as target:
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()
    writer.writerows(
        [{k: v for k, v in row.items() if k != "per_frame"} for row in rows]
    )
(root / "figure_source_data.json").write_text(json.dumps(rows, indent=2) + "\n")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
    }
)
models = [
    ("single", "Single-scale", "#456F9D", "o"),
    ("multi", "Multi-scale residual", "#C6814D", "s"),
]
fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 135 / 25.4))
fig.subplots_adjust(
    left=0.10, right=0.98, bottom=0.14, top=0.88, wspace=0.28, hspace=0.38
)
for index, (ax, metric, label) in enumerate(
    zip(
        axes.flat,
        ["psnr", "ssim", "lpips", "dists"],
        ["PSNR (dB)", "SSIM", "LPIPS", "DISTS"],
    )
):
    for name, display, color, marker in models:
        selected = [r for r in rows if r["model"] == name]
        videos = sorted({r["video"] for r in selected})
        for video in videos:
            values = sorted(
                [r for r in selected if r["video"] == video], key=lambda r: r["budget"]
            )
            ax.plot(
                [r["actual_bytes"] for r in values],
                [r[metric] for r in values],
                color=color,
                linewidth=0.6,
                alpha=0.35,
            )
        x, y = [], []
        for k in (256, 512, 1024, 2048):
            group = [r for r in selected if r["budget"] == k]
            if group:
                x.append(np.mean([r["actual_bytes"] for r in group]))
                y.append(np.mean([r[metric] for r in group]))
        ax.plot(
            x, y, marker=marker, color=color, linewidth=1.4, markersize=3, label=display
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([3200, 5800, 10900, 21100], labels=["3.2k", "5.8k", "10.9k", "21.1k"])
    ax.set_xlabel("Actual bytes / 17 frames")
    ax.set_ylabel(label)
    ax.tick_params(labelsize=7)
    ax.text(
        -0.13,
        1.08,
        chr(ord("a") + index),
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="upper center",
    bbox_to_anchor=(0.54, 0.995),
    ncol=2,
    fontsize=8,
)
fig.text(
    0.10,
    0.035,
    "Thin lines: individual clips. Thick lines: descriptive mean; no seed uncertainty estimate.\nEarly smoke checkpoints; source pixels 1920 x 1080. Negative gains retained.",
    fontsize=7,
)
require_matplotlib_panel_alignment(
    fig,
    json_out=figures / "rate_distortion.alignment.json",
    panel_ids=["a", "b", "c", "d"],
    strict=True,
)
export_figure(fig, figures / "rate_distortion")
plt.close(fig)
# A matched-byte ablation is paired by video and K, not by nominal model bitrate.
pairs = []
for row in rows:
    if row["model"] != "multi":
        continue
    matches = [
        r
        for r in rows
        if (r["model"], r["video"], r["budget"])
        == ("single", row["video"], row["budget"])
    ]
    if not matches:
        continue
    baseline = matches[0]
    if row["actual_bytes"] != baseline["actual_bytes"]:
        raise ValueError("ablation does not use identical actual bytes")
    pairs.append(
        dict(
            video=row["video"],
            budget=row["budget"],
            actual_bytes=row["actual_bytes"],
            psnr_delta=row["psnr"] - baseline["psnr"],
            lpips_delta=row["lpips"] - baseline["lpips"],
            dists_delta=row["dists"] - baseline["dists"],
        )
    )
(root / "multiscale_ablation.json").write_text(json.dumps(pairs, indent=2) + "\n")
fig, ax = plt.subplots(figsize=(120 / 25.4, 75 / 25.4))
fig.subplots_adjust(left=0.15, bottom=0.22, right=0.97, top=0.82)
for video in sorted({r["video"] for r in pairs}):
    group = sorted([r for r in pairs if r["video"] == video], key=lambda r: r["budget"])
    ax.plot(
        [r["budget"] for r in group],
        [r["psnr_delta"] for r in group],
        marker="o",
        markersize=3,
        linewidth=1,
        label=video,
    )
ax.axhline(0, color="#666666", linewidth=0.6)
ax.set_xscale("log", base=2)
ax.set_xticks([256, 512, 1024, 2048], labels=["256", "512", "1024", "2048"])
ax.set_xlabel("Tokens per temporal block")
ax.set_ylabel("Multi-scale minus single-scale PSNR (dB)")
fig.legend(
    *ax.get_legend_handles_labels(),
    fontsize=7,
    loc="upper center",
    bbox_to_anchor=(0.55, 0.99),
    ncol=3,
)
export_figure(fig, figures / "multiscale_ablation")
plt.close(fig)
# Anchor order is a separate, identically budgeted training ablation.
anchor_rows = [
    r for r in rows if r["video"] == "mcl_jcv01" and r["model"] in ("multi", "morton")
]
if any(r["model"] == "morton" for r in anchor_rows):
    fig, ax = plt.subplots(figsize=(120 / 25.4, 75 / 25.4))
    fig.subplots_adjust(left=0.15, bottom=0.22, right=0.97, top=0.82)
    for model, label, color in [
        ("multi", "Halton", "#456F9D"),
        ("morton", "Hierarchical Morton", "#C6814D"),
    ]:
        group = sorted(
            [r for r in anchor_rows if r["model"] == model], key=lambda r: r["budget"]
        )
        ax.plot(
            [r["actual_bytes"] for r in group],
            [r["psnr"] for r in group],
            marker="o",
            markersize=3,
            color=color,
            linewidth=1,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([3200, 5800, 10900, 21100], labels=["3.2k", "5.8k", "10.9k", "21.1k"])
    ax.set_xlabel("Actual bytes / 17 frames")
    ax.set_ylabel("PSNR (dB), MCL-JCV01")
    fig.legend(
        *ax.get_legend_handles_labels(),
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
        ncol=2,
    )
    export_figure(fig, figures / "anchor_order_ablation")
    plt.close(fig)
    (root / "anchor_quality_ablation.json").write_text(
        json.dumps(anchor_rows, indent=2) + "\n"
    )
print(
    json.dumps(
        dict(
            observations=len(rows),
            paired_ablation_observations=len(pairs),
            videos=sorted({r["video"] for r in rows}),
        )
    )
)
