#!/usr/bin/env python3
"""Run a command verbatim, saving command, commit, timestamps, output and exit status."""

import argparse
import datetime as dt
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--root", default="results/chp_vtok")
p.add_argument("--gpu", action="store_true", help="serialize GPU experiments on this checkout")
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ["--"] else a.command
root = Path(a.root)
(root / "logs").mkdir(parents=True, exist_ok=True)
gpu_lock = None
if a.gpu:
    gpu_lock = (root / ".gpu.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
start = dt.datetime.now(dt.timezone.utc).isoformat()
stamp = time.time_ns()
log = root / "logs" / f"{a.name}_{stamp}.log"
record = dict(
    name=a.name,
    argv=command,
    cwd=str(Path.cwd()),
    start_utc=start,
    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
    log=str(log),
)
t0 = time.perf_counter()
with log.open("w") as f:
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        f.write(line)
        f.flush()
        print(line, end="", flush=True)
    record["exit_code"] = proc.wait()
record["seconds"] = time.perf_counter() - t0
with (root / "commands.jsonl").open("a") as f:
    f.write(json.dumps(record) + "\n")
if gpu_lock is not None:
    fcntl.flock(gpu_lock, fcntl.LOCK_UN)
    gpu_lock.close()
sys.exit(record["exit_code"])
