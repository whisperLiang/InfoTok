#!/usr/bin/env python3
"""Build delivery provenance from actual files; never manufacture measurements.

The manifest is derived metadata and may be regenerated. Experiment outputs,
weights, configs and logs are only read. Large tensors remain local and ignored
by Git, but have content hashes and explicit paths in artifact_manifest.json.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import cv2
import torch

from cosmos_predict1.tokenizer.progressive.config import CHPConfig
from cosmos_predict1.tokenizer.progressive.model import file_sha256

UPSTREAM = "042fc5815717ee34d7281e7eb2971ad9c9d76421"
REVISION = "7b9f0d4d5b7ce128e7de0d4252f34199b9ac25f5"
OFFICIAL_SHA = "bd69629a8f8fcfc4808d29eec5bed68177e22aa14e3b9b799599791e26af935a"


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checkpoint_record(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    config = CHPConfig(**raw["chp_config"])
    if raw["config_sha256"] != config.sha256:
        raise ValueError(f"checkpoint config hash mismatch: {path}")
    return dict(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
        kind=raw.get("kind", "training"),
        global_step=raw["global_step"],
        config_sha256=raw["config_sha256"],
        training_config_sha256=raw.get("training_config_sha256"),
        source_checkpoint_sha256=raw.get("source_checkpoint_sha256"),
        saved_states={
            key: key in raw
            for key in ("model", "optimizer", "scheduler", "grad_scaler", "rng")
        },
        config=raw["chp_config"],
        training_status=raw.get("training_status"),
        distribution="local ignored checkpoint; not uploaded",
    )


def preserve_training_configs(root):
    destination = root / "training"
    destination.mkdir(exist_ok=True)
    for job in sorted(Path("checkpoints/chp_vtok/tokenizer").iterdir()):
        target = destination / job.name
        target.mkdir(exist_ok=True)
        for name in ("config.yaml", "steps.jsonl"):
            source = job / name
            if source.exists():
                shutil.copyfile(source, target / name)


def verify_video_exports(root):
    exports = json.loads((root / "video_exports.json").read_text())
    for record in exports:
        path = Path(record["mp4"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"display video hash changed: {path}")
        cap = cv2.VideoCapture(str(path))
        shape = [
            int(cap.get(key))
            for key in (
                cv2.CAP_PROP_FRAME_COUNT,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FRAME_WIDTH,
            )
        ]
        cap.release()
        if shape != record["geometry"]:
            raise ValueError(f"display video geometry changed: {path}")
    return len(exports)


def packet_tables(root):
    for experiment in ("tcp_incremental_v2", "tcp_mixed_v2"):
        data = json.loads((root / experiment / "result.json").read_text())
        rows = data["packets"]
        with (root / experiment / "packets.csv").open("w") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def source_identity(root):
    changes = git(
        "diff", "--name-only", UPSTREAM, "HEAD", "--", ".", ":(exclude)results"
    ).splitlines()
    files = [Path(name) for name in changes if Path(name).is_file()]
    hashes = {str(path): file_sha256(path) for path in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (root / "implementation.diff").write_text(
        git("diff", UPSTREAM, "HEAD", "--", ".", ":(exclude)results") + "\n"
    )
    save(root / "source_hashes.json", hashes)
    return dict(
        implementation_commit=git("rev-parse", "HEAD"),
        upstream_commit=UPSTREAM,
        branch=git("branch", "--show-current"),
        remotes=git("remote", "-v"),
        source_files_sha256=digest,
        changed_paths=changes,
        provenance_limit="Earlier commands honestly recorded dirty=true. Their HEAD is not claimed to be an exact source snapshot; the final implementation diff and source hashes are archived here.",
    )


def artifact_index(root):
    files = sorted(root.rglob("*"))
    records = []
    for path in files:
        if (
            not path.is_file()
            or path.name in (".gpu.lock", "artifact_manifest.json", "commands.jsonl")
            or "logs" in path.relative_to(root).parts
        ):
            continue
        records.append(
            dict(
                path=str(path.resolve()),
                bytes=path.stat().st_size,
                sha256=file_sha256(path),
                local_tensor_excluded_from_git=path.suffix == ".pt",
            )
        )
    save(root / "artifact_manifest.json", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/chp_vtok")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="refresh derived artifact hashes after writing the report",
    )
    args = parser.parse_args()
    root = Path(args.root)
    if args.index_only:
        artifact_index(root)
        print(
            "Artifact index refreshed; mutable command logs are excluded.", flush=True
        )
        return
    quality_files = sorted((root / "quality_reference_fp32").glob("rd_*/metrics.json"))
    source_files = sorted((root / "reconstructions").glob("rd_*/metrics.json"))
    if len(quality_files) != len(source_files):
        raise ValueError("complete FP32-reference quality records are required")
    quality_rows = [
        row for path in quality_files for row in json.loads(path.read_text())
    ]
    if not quality_rows or any(
        row.get("reference_dtype") != "float32" for row in quality_rows
    ):
        raise ValueError("quality references must be exact FP32 source RGB")
    official = Path("checkpoints/infotok-flex/infotok_mse.pt")
    actual_sha = file_sha256(official)
    if actual_sha != OFFICIAL_SHA:
        raise ValueError("official checkpoint hash changed")
    weights = [
        checkpoint_record(p) for p in sorted(Path("checkpoints/chp_vtok").rglob("*.pt"))
    ]
    save(root / "checkpoint_manifest.json", weights)
    preserve_training_configs(root)
    shutil.copyfile("data/chp_vtok/manifest.json", root / "data_manifest.json")
    packet_tables(root)
    commands = [
        json.loads(line) for line in (root / "commands.jsonl").read_text().splitlines()
    ]
    save(
        root / "failures.json",
        [command for command in commands if command["exit_code"] != 0],
    )
    audit = dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        code=source_identity(root),
        device=dict(
            name=torch.cuda.get_device_name(),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            python=platform.python_version(),
            driver=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
            cuda_devices=torch.cuda.device_count(),
        ),
        official_checkpoint=dict(
            path=str(official.resolve()),
            sha256=actual_sha,
            bytes=official.stat().st_size,
            source=f"https://huggingface.co/qyoo/infotok-flex/resolve/{REVISION}/infotok_mse.pt",
            revision=REVISION,
            architecture="OURS4x8x8-mse-256p-88",
            published_temporal_window=81,
            published_spatial_training_profile="varied, non-square; no verified 1080p training claim",
        ),
        licenses=dict(
            repository="Apache-2.0",
            model_card="Apache-2.0",
            inconsistent_legacy_headers=[
                "cosmos_predict1/tokenizer/networks/ours_discrete_video.py",
            ],
        ),
        upstream_findings=dict(
            A_model="OursDiscreteVideoTokenizer and official config retained. Checkpoint loads all learned parameters; omitted fixed FSQ/patch buffers are not treated as missing learned weights.",
            B_backbone="Factorized 4x8x8 causal convolution encoder and decoder are reused as objects. CHP replaces their spatial attention with bounded 16x16 windows; this is an architecture change.",
            C_vit="vit_encode/vit_decode keep dense time/space organization and global block-causal attention. The old ViT path is excluded only from the separately configured CHP model, not deleted upstream.",
            D_masks="adaptive_tokenize/mask_tokens zero continuous dense positions. MSE ordering requires position metadata. These masked latents are not the CHP integer prefix stream.",
            E_FSQ="Exact upstream FSQuantizer with 64000 symbols and bidirectional index/code mapping; no 8192/13-bit assumption.",
            F_latents="Original reshaped dense 2D latents differ from new fixed learned 1D slots. Missing slots have an independent mask/embedding; slot order is deterministic geometry, with no full ordering table on the wire.",
            G_attention="Original block mask marks future blocks as forbidden. SDPA boolean masks mark allowed positions True. The optimized temporal block uses causal SDPA; cache offset masking is explicit.",
            H_checkpoint="Flex model card reports 81-frame windows at varied non-square resolutions. Config label 256p is not evidence of native 1080p training. Original ViT spatial rotary tables are 96x96 and temporal embeddings have 49 positions.",
            I_gradients="Frozen encoder may use no_grad. Frozen decoder does not: RGB gradients reach new latent recovery modules. Native checkpoint on/off gradient equality is measured; finite nonzero training gradients are logged.",
            J_streaming="Original video_lib/video_cli use window execution and overlap; ViT cache implementation is commented out. CHP implements actual per-module convolution/phase/bootstrap/pooling/upsample/KV state. Recomputed history and reused caches have independent measurements.",
            normalization="Actual factorized backbone CausalNormalize uses num_groups=1 and time2batch GroupNorm. Its implementation and pretrained normalization semantics are unchanged.",
        ),
        quantizer=dict(
            type="upstream FSQ",
            levels=[8, 8, 8, 5, 5, 5],
            symbols=64000,
            bits_per_id=16,
            zero_is_a_valid_symbol=True,
            exhaustive_roundtrip_ids=64000,
        ),
        geometry=dict(
            source=[1, 3, 17, 1080, 1920],
            padded=[1, 3, 17, 1088, 1920],
            latent=[1, 256, 5, 136, 240],
            temporal_protocol="one bootstrap + four 4-frame blocks; explicit EOS valid tail",
        ),
        feature_acceptance=dict(
            native_1080_geometry="passed on trained smoke checkpoints",
            temporal_causality="passed encoder and decoder interventions",
            variable_length="passed mixed 256/512/1024/2048; 4096/8192 unit-only",
            strict_prefix="passed same maximum encoding, incremental uint16 IDs, independent receiver",
            streaming_fp32="native 17-frame IDs equal; RGB max error 1.0132789611816406e-6",
            streaming_bfloat16="not integer-equivalent; native ID difference fraction 0.0027343749534338713",
            realtime_30fps="failed; measured seconds per block",
        ),
        training=dict(
            status="engineering smoke only; no convergence/generalization claim",
            single_steps=42,
            multi_steps=42,
            multi_native17_resume_step=44,
            morton_steps=42,
            optional_finetune_steps=2,
            held_out_evaluation=False,
            native17_perceptual="uniform real 256x256 source-pixel crop, full RGB/temporal losses",
            dataloader_replay_bit_exact=False,
        ),
        evidence=dict(
            final_tests="tests_delivery_final.xml",
            rd_observations=len(quality_rows),
            quality_rows="summary.csv",
            quality_reference_dtype="float32",
            quality_record_directory="quality_reference_fp32",
            native_display_exports=verify_video_exports(root),
            receiver_protocol="CHP2/CHPB",
            final_tcp="tcp_incremental_v2/result.json",
            mixed_tcp="tcp_mixed_v2/result.json",
            full_fp32_consistency="consistency_multi_1080_fp32.json",
            full_bf16_consistency="consistency_multi_1080_bounded_serial.json",
            old_chp1_status="superseded; not final causality proof",
            dependency_freeze="dependencies.txt",
        ),
        limitations=[
            "No converged model or held-out benchmark",
            "BF16 full-window versus cache discrete IDs differ",
            "Not real-time",
            "No controlled global-vs-local attention quality ablation",
            "Only ordered blocks; no late enhancement replay or WAN impairment test",
            "Unlimited temporal attention KV grows with session duration",
            "FP16 native full-graph equivalence and extended budgets not GPU-qualified",
            "Some early experiments ran from dirty trees; see recorded command provenance",
        ],
    )
    save(root / "audit.json", audit)
    artifact_index(root)
    print(
        json.dumps(
            dict(
                checkpoints=len(weights),
                native_video_exports=audit["evidence"]["native_display_exports"],
                implementation_commit=audit["code"]["implementation_commit"],
            )
        )
    )


if __name__ == "__main__":
    main()
