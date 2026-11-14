"""Prepare a recorded temporal holdout from the three existing native sources.

This is a same-scene temporal split, not a disjoint-video test set. Source
files remain read only. No resizing, new reference images, or lossy encoding.
"""

import json
import subprocess
from pathlib import Path

from cosmos_predict1.tokenizer.progressive.model import file_sha256


root = Path(__file__).resolve().parent
data_root = Path("data/chp_vtok") / root.name
manifest = []
frame_bytes = 1920 * 1080 * 3 // 2
for source in json.loads(Path("results/chp_vtok/data_manifest.json").read_text()):
    path = Path(source["source"])
    total, remainder = divmod(path.stat().st_size, frame_bytes)
    assert remainder == 0 and total >= 85, (path, total, remainder)
    train_count = 1 + 4 * ((int(total * 0.7) - 1) // 4)
    validation_start = train_count + 16
    assert train_count >= 17 and validation_start + 17 <= total
    assert validation_start > 17  # Previous smoke training used only frames 0:17.
    source_sha = file_sha256(path)
    for split, start, count in (
        ("train", 0, train_count),
        ("validation", validation_start, 17),
    ):
        output = data_root / split / (source["name"] + ".mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-v", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "yuv420p",
            "-video_size", "1920x1080", "-framerate", str(source["fps"]),
            "-skip_initial_bytes", str(start * frame_bytes), "-i", str(path),
            "-frames:v", str(count), "-c:v", "libx264rgb", "-crf", "0",
            "-preset", "fast", "-threads", "4", str(output),
        ]
        print(json.dumps({"preparing": str(output), "argv": command}), flush=True)
        subprocess.run(command, check=True)
        probe_command = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,nb_read_frames,pix_fmt",
            "-of", "json", str(output),
        ]
        probe = json.loads(subprocess.check_output(probe_command))["streams"][0]
        assert (probe["width"], probe["height"], int(probe["nb_read_frames"])) == (
            1920, 1080, count
        ), probe
        # Compare the first encoded RGB frame with a fresh RGB conversion of
        # the exact raw source offset, independently of container metadata.
        source_command = command[:command.index("-frames:v")] + [
            "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"
        ]
        decoded_command = [
            "ffmpeg", "-v", "error", "-i", str(output), "-frames:v", "1",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ]
        assert subprocess.check_output(source_command) == subprocess.check_output(decoded_command)
        manifest.append(dict(
            source=str(path), source_sha256=source_sha, source_read_only=True,
            source_frames=total, split=split, source_start=start,
            source_end_exclusive=start + count, frame_count=count,
            geometry=[1920, 1080], fps=source["fps"],
            output=str(output), output_sha256=file_sha256(output),
            command=command, probe_command=probe_command, probe=probe,
            first_rgb_frame_exact=True,
        ))

record = dict(
    split_type="same-scene-temporal-holdout-not-disjoint-video",
    temporal_gap_frames=16, validation_clip_frames=17,
    previous_smoke_frames_per_scene=17,
    training_sampling="uniform temporal start within each scene; scenes cycled equally",
    caveat="Three source scenes only; no cross-video generalization claim.",
    files=manifest,
)
(root / "data_split.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"prepared": str(root / "data_split.json")}), flush=True)
