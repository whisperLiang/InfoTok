#!/usr/bin/env python3
"""Native-pixel ROI stress measurements, evaluated after full-frame decoding.

Regions are fixed manual spatial boxes, not segmentation or tracked-object masks.
They localize examples of small text, texture, static background and moving edges.
They are not additional independent videos or evidence of generalization.
"""

import argparse
import json
from pathlib import Path
import torch
from scripts.chp_evaluate import Metrics
from cosmos_predict1.tokenizer.progressive.cli import read_rgb

REGIONS = {
    "mcl_jcv01": {
        "static_background": (950, 0, 1450, 250),
        "textured_shelves": (0, 100, 450, 650),
    },
    "uvg_beauty": {
        "small_watermark_text": (1600, 960, 1920, 1080),
        "hair_texture": (1120, 100, 1650, 780),
    },
    "basketball": {
        "moving_player_region": (0, 150, 600, 1000),
        "small_wall_sign": (900, 260, 1400, 500),
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/chp_vtok")
    a = p.parse_args()
    root = Path(a.root)
    torch.set_num_threads(4)
    metrics = Metrics()
    rows = []
    with torch.no_grad():
        for directory in sorted((root / "reconstructions").glob("rd_*")):
            summary = directory / "metrics.json"
            if not summary.exists():
                continue
            records = json.loads(summary.read_text())
            video = records[0]["video"]
            source, _ = read_rgb("data/chp_vtok/" + video + ".mp4", 17)
            for record in records:
                recon = torch.load(
                    directory / f"reconstruction_k{record['budget']}.pt",
                    weights_only=True,
                )
                for name, (x0, y0, x1, y1) in REGIONS[video].items():
                    quality = metrics(
                        source[..., y0:y1, x0:x1].cuda(),
                        recon[..., y0:y1, x0:x1].cuda(),
                    )
                    rows.append(
                        dict(
                            video=video,
                            model=record["model"],
                            budget=record["budget"],
                            region=name,
                            box=[x0, y0, x1, y1],
                            actual_full_video_bytes=record["actual_bytes"],
                            **quality,
                        )
                    )
    (root / "region_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (root / "regions.json").write_text(json.dumps(REGIONS, indent=2) + "\n")
    print(json.dumps(dict(measurements=len(rows), description=__doc__)))


if __name__ == "__main__":
    main()
