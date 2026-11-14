#!/usr/bin/env python3
"""Lossless 17-frame extracts from existing public 1080p YUV sources, no resizing."""

import json
import subprocess
from pathlib import Path

from cosmos_predict1.tokenizer.progressive.model import file_sha256

root = Path("data/chp_vtok")
root.mkdir(parents=True, exist_ok=True)
sources = [
    (
        "mcl_jcv01",
        Path(
            "/home/whisperliang/video-tokenizer-benchmark/data/compression_sources/MCL-JCV/videoSRC01_1920x1080_30.yuv"
        ),
        30,
    ),
    (
        "uvg_beauty",
        Path(
            "/home/whisperliang/video-tokenizer-benchmark/data/compression_sources/UVG/Beauty_1920x1080_120fps_420_8bit_YUV.yuv"
        ),
        120,
    ),
    (
        "basketball",
        Path(
            "/home/whisperliang/video-tokenizer-benchmark/data/compression_sources/HEVC-B/BasketballDrive_1920x1080_50.yuv"
        ),
        50,
    ),
]
records = []
for name, source, fps in sources:
    if not source.is_file():
        continue
    output = root / (name + ".mp4")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "yuv420p",
        "-video_size",
        "1920x1080",
        "-framerate",
        str(fps),
        "-i",
        str(source),
        "-frames:v",
        "17",
        "-c:v",
        "libx264rgb",
        "-crf",
        "0",
        "-preset",
        "fast",
        str(output),
    ]
    if not output.exists():
        subprocess.run(command, check=True)
    records.append(
        dict(
            name=name,
            source=str(source),
            source_read_only=True,
            geometry=[1920, 1080],
            fps=fps,
            frame_count=17,
            command=command,
            sha256=file_sha256(output),
            purpose="engineering-smoke; not a held-out benchmark",
        )
    )
(root / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
print(json.dumps(records, indent=2))
