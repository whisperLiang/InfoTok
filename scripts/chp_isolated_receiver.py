#!/usr/bin/env python3
"""Run the receiver with an OS-level Python file audit denying sender inputs.

This is a process independence check, not a hostile-process sandbox. Its CLI has
no raw-video or private-latent argument; every read of a repository tensor file
except the public decoder export is rejected, and attempted video reads fail.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--stream", required=True)
p.add_argument("--output", required=True)
p.add_argument("--dtype", default="bfloat16")
p.add_argument("--device", default="cuda")
a = p.parse_args()
repo = Path.cwd().resolve()
public = Path(a.checkpoint).resolve()
wire = Path(a.stream).resolve()
reads = []


def audit(event, arguments):
    if event != "open" or not isinstance(arguments[0], (str, bytes)):
        return
    path = Path(arguments[0]).resolve()
    mode = arguments[1]
    if not isinstance(mode, str) or any(c in mode for c in "wax+"):
        return
    if repo in path.parents and (path.suffix in (".pt", ".mp4", ".yuv", ".chp")):
        if path not in (public, wire):
            raise PermissionError(f"independent receiver cannot read {path}")
        reads.append(str(path))


sys.addaudithook(audit)
from cosmos_predict1.tokenizer.progressive.cli import receive
import torch

torch.set_num_threads(4)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

receive(SimpleNamespace(**vars(a), model_id="CHP-VTok-v1"))
Path(a.output, "input_access_audit.json").write_text(
    json.dumps(
        dict(
            permitted_repository_inputs=sorted(set(reads)),
            raw_video_access=False,
            sender_latent_access=False,
            process_id=__import__("os").getpid(),
        ),
        indent=2,
    )
    + "\n"
)
