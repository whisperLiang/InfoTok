#!/usr/bin/env python3
"""Re-measure saved reconstructions against exact FP32 native RGB references.

Earlier inference cast the metric reference to BF16. Preserve those observations
and their performance logs; this independent derived directory replaces only
quality values. No model inference is needed for the existing RD tensors.
History interventions re-use received integer IDs and the shared decoder API.
"""

import argparse
import json
from pathlib import Path

import torch

from cosmos_predict1.tokenizer.progressive.checkpoint import load_checkpoint
from cosmos_predict1.tokenizer.progressive.cli import read_rgb
from cosmos_predict1.tokenizer.progressive.model import Geometry
from scripts.chp_evaluate import Metrics, decode


@torch.no_grad()
def remeasure(directory, output, metrics, resume=False):
    records = json.loads((directory / "metrics.json").read_text())
    destination = output / "metrics.json"
    if resume and destination.exists():
        completed = json.loads(destination.read_text())
        identity = lambda row: (
            row["checkpoint_sha256"],
            row["budget"],
            row["actual_bytes"],
        )
        if [identity(row) for row in completed] != [
            identity(row) for row in records
        ] or any(row.get("reference_dtype") != "float32" for row in completed):
            raise ValueError(f"incompatible quality resume: {output}")
        print(f"Verified completed FP32-reference record: {output}", flush=True)
        return
    source, _ = read_rgb(f"data/chp_vtok/{records[0]['video']}.mp4", 17)
    reference = source.cuda()  # Deliberately no model-dtype cast.
    output.mkdir(parents=True, exist_ok=resume)
    corrected = []
    for record in records:
        path = directory / f"reconstruction_k{record['budget']}.pt"
        reconstruction = torch.load(path, map_location="cuda", weights_only=True)
        row = {
            **record,
            **metrics(reference, reconstruction),
            "reference_dtype": "float32",
            "inference_dtype": str(reconstruction.dtype),
            "source_performance_record": str(directory / "metrics.json"),
            "reconstruction_tensor": str(path),
        }
        corrected.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "per_frame"}), flush=True
        )
    destination.write_text(json.dumps(corrected, indent=2) + "\n")


@torch.no_grad()
def history_interventions(root, output, metrics):
    source, _ = read_rgb("data/chp_vtok/mcl_jcv01.mp4", 17)
    reference = source.cuda()
    geometry = Geometry.from_video(source)
    model, identity = load_checkpoint(
        "checkpoints/chp_vtok/tokenizer/multi_native/checkpoints/iter_000000042.pt",
        device="cuda",
        dtype=torch.bfloat16,
    )
    tokens = torch.load(
        root / "reconstructions/rd_multi_mcl_verified/token_ids.pt",
        map_location="cuda",
        weights_only=True,
    )
    base, _ = decode(model, geometry, tokens, [256] * tokens.shape[1])
    rows = []
    for source_block in range(1, 4):
        budgets = [256] * tokens.shape[1]
        budgets[source_block] = 2048
        enhanced, _ = decode(model, geometry, tokens, budgets)
        for target in range(source_block + 1, 5):
            start = 1 + 4 * (target - 1)
            original = reference[:, :, start : start + 4]
            baseline = metrics(original, base[:, :, start : start + 4])
            quality = metrics(original, enhanced[:, :, start : start + 4])
            rows.append(
                dict(
                    source_block=source_block,
                    target_block=target,
                    target_ids_unchanged=True,
                    checkpoint_sha256=identity,
                    reference_dtype="float32",
                    additional_payload_bytes=3584,
                    baseline=baseline,
                    enhanced=quality,
                    delta={
                        k: quality[k] - baseline[k]
                        for k in ("psnr", "ssim", "lpips", "dists")
                    },
                )
            )
    (output / "history_intervention.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/chp_vtok")
    parser.add_argument("--output", default="results/chp_vtok/quality_reference_fp32")
    parser.add_argument("--history", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify and reuse completed derived quality files",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    metrics = Metrics()
    for summary in sorted((root / "reconstructions").glob("rd_*/metrics.json")):
        remeasure(summary.parent, output / summary.parent.name, metrics, args.resume)
    if args.history:
        history_interventions(root, output, metrics)


if __name__ == "__main__":
    main()
