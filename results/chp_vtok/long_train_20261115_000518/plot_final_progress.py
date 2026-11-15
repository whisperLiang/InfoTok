"""Measured matched-byte training progress; no smoothing or observations removed."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.home() / ".codex/skills/nature-figure/scripts"))
from audit_panel_alignment import require_matplotlib_panel_alignment

root = Path(__file__).resolve().parent
rows = json.loads((root / "final_averages.json").read_text())
assert len(rows) == 20 and all(r["actual_bytes_mean"] > 0 for r in rows)
figures = root / "figures"
figures.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "legend.frameon": False,
})
fig, ax = plt.subplots(figsize=(180 / 25.4, 105 / 25.4))
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.28, top=0.81)
stages = [
    ("baseline44", "Step 44", "#999999", "o"),
    ("step100", "Step 100", "#9BB5CB", "^"),
    ("step300", "Step 300", "#5C8FA5", "s"),
    ("step600", "Step 600", "#526F8C", "v"),
    ("step1044", "Step 1044", "#C6814D", "D"),
]
for phase, label, color, marker in stages:
    group = sorted([r for r in rows if r["phase"] == phase], key=lambda r: r["budget"])
    assert len(group) == 4
    ax.plot(
        [r["actual_bytes_mean"] for r in group],
        [r["psnr"] for r in group],
        marker=marker, color=color, markersize=3.5,
        linewidth=1.8 if phase == "step1044" else 1.0, label=label,
    )
ax.set_xscale("log", base=2)
ax.set_xticks([3216.667, 5776.667, 10896.667, 21136.667],
              labels=["3.2k", "5.8k", "10.9k", "21.1k"])
ax.set_xlabel("Actual bytes per 17 frames (bootstrap included)")
ax.set_ylabel("Mean PSNR (dB)")
ax.set_ylim(18.0, 22.1)
ax.grid(axis="y", alpha=0.16, linewidth=0.5)
fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc="upper center",
           bbox_to_anchor=(0.54, 0.96), fontsize=7.5)
fig.text(0.12, 0.08,
         "Three fixed same-scene temporal holdouts; native 1920 x 1080, 17 frames.\n"
         "Arithmetic means; no seed confidence interval. All individual records and negative changes retained.",
         fontsize=7)
require_matplotlib_panel_alignment(
    fig, json_out=figures / "training_rate_distortion.alignment.json", strict=True
)
fig.savefig(figures / "training_rate_distortion.pdf")
fig.savefig(figures / "training_rate_distortion.svg")
fig.savefig(figures / "training_rate_distortion.png", dpi=600)
plt.close(fig)
