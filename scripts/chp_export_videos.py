#!/usr/bin/env python3
"""Export native RGB reconstruction tensors with the existing InfoTok writer.

MP4 is a display export. Scientific metrics use the saved RGB tensors before
this lossy encoding. A completed export is verified by decoding its geometry and
frame count; existing exports are never replaced.
"""

import argparse
import json
from pathlib import Path

import cv2
import torch

from cosmos_predict1.tokenizer.inference.utils import write_video
from cosmos_predict1.tokenizer.progressive.model import file_sha256


def export(record, directory):
    source = directory / f"reconstruction_k{record['budget']}.pt"
    destination = directory / f"native_k{record['budget']}.mp4"
    if destination.exists():
        raise FileExistsError(destination)
    rgb = torch.load(source, map_location="cpu", weights_only=True)
    pixels = (
        ((rgb[0].float().clamp(-1, 1) + 1) * 127.5)
        .round()
        .byte()
        .permute(1, 2, 3, 0)
        .numpy()
    )
    write_video(str(destination), pixels, fps=record["fps"])
    cap = cv2.VideoCapture(str(destination))
    shape = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    cap.release()
    if shape != (record["frames"], record["height"], record["width"]):
        raise ValueError(f"display export geometry mismatch: {destination}: {shape}")
    return dict(
        source_tensor=str(source),
        mp4=str(destination),
        sha256=file_sha256(destination),
        geometry=list(shape),
        fps=record["fps"],
        scientific_metrics_source="RGB tensor before display-only MP4 encoding",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/chp_vtok")
    args = parser.parse_args()
    root = Path(args.root)
    records = []
    torch.set_num_threads(4)
    for summary in sorted((root / "reconstructions").glob("rd_*/metrics.json")):
        for record in json.loads(summary.read_text()):
            records.append(export(record, summary.parent))
    with (root / "video_exports.json").open("x") as target:
        target.write(json.dumps(records, indent=2) + "\n")
    print(json.dumps(dict(native_display_exports=len(records))))


if __name__ == "__main__":
    main()
