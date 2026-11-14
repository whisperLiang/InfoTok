"""End-user file interfaces, using labeled tiny test weights and real video I/O."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from test_chp_checkpoint import make_test_checkpoint

from cosmos_predict1.tokenizer.progressive.checkpoint import export_decoder
from cosmos_predict1.tokenizer.progressive.cli import (
    _sender_output_prefix,
    _transmission_plan,
    main,
    read_rgb,
    receive,
    send,
)


def test_file_sender_receiver_mixed_budgets_and_tail(tmp_path):
    checkpoint = tmp_path / "test.pt"
    make_test_checkpoint(checkpoint)
    decoder = tmp_path / "public.pt"
    export_decoder(checkpoint, decoder)
    video = tmp_path / "real_io.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 32))
    assert writer.isOpened()
    rng = np.random.default_rng(1)
    for _ in range(4):
        writer.write(rng.integers(0, 256, (32, 64, 3), dtype=np.uint8))
    writer.release()
    wire = tmp_path / "video.chp"
    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        video=str(video),
        frames=4,
        device="cpu",
        dtype="float32",
        model_id="test",
        stream=str(wire),
        budgets="256,512,1024,2048",
        block_budgets="512,1024",
        log_prefix=None,
    )
    send(args)
    sent = torch.load(str(wire) + ".sender_ids.pt", weights_only=True)
    output = tmp_path / "receiver"
    receive(
        SimpleNamespace(
            checkpoint=str(decoder),
            stream=str(wire),
            output=str(output),
            device="cpu",
            dtype="float32",
            model_id="test",
        )
    )
    recovered = torch.load(output / "received_ids.pt", weights_only=True)
    rgb = torch.load(output / "reconstruction.pt", weights_only=True)
    assert [v.shape[1] for v in recovered] == [512, 1024]
    assert torch.equal(recovered[0], sent[:, 0, :512]) and torch.equal(
        recovered[1], sent[:, 1, :1024]
    )
    assert rgb.shape == (1, 3, 4, 32, 64)
    sender = json.loads(Path(str(wire) + ".sender.json").read_text())
    receiver = json.loads((output / "receiver.json").read_text())
    assert sender["actual_bytes"] == receiver["actual_bytes"] == wire.stat().st_size
    assert len(receiver["packets"]) == 5 and receiver["protocol_version"] == 2
    assert (
        receiver["packets"][1]["committed"] and not receiver["packets"][0]["committed"]
    )
    with pytest.raises(ValueError, match="public decoder"):
        receive(SimpleNamespace(checkpoint=str(checkpoint)))


@pytest.mark.parametrize(
    "budgets,limits",
    [
        ("512,256", None),
        ("128,256", None),
        ("256,9000", None),
        ("256", "256"),
        ("256", "2,256"),
    ],
)
def test_invalid_transmission_plans(budgets, limits):
    with pytest.raises(ValueError):
        _transmission_plan(
            SimpleNamespace(budgets=budgets, block_budgets=limits), 2048, 2
        )


def test_tcp_requires_local_log_prefix():
    with pytest.raises(ValueError, match="log-prefix"):
        _sender_output_prefix(SimpleNamespace(stream="tcp://127.0.0.1:1000"))


def test_missing_video_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        read_rgb(str(tmp_path / "does_not_exist.mp4"))


def test_export_decoder_cli(tmp_path, monkeypatch):
    source = tmp_path / "test.pt"
    make_test_checkpoint(source)
    output = tmp_path / "decoder.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["chp", "export-decoder", "--checkpoint", str(source), "--output", str(output)],
    )
    # Public CLI setup should not initialize CUDA for an export-only command.
    main()
    assert torch.load(output, weights_only=False)["kind"] == "chp_decoder"
