"""Execute recorded commands; model training stays in InfoTok's Trainer."""

import datetime as dt
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
plan = json.loads((root / "plan.json").read_text())
assert (root / "data_split.json").is_file(), "prepare and verify the split first"
env = dict(os.environ, **plan["environment"])
status = dict(start_utc=dt.datetime.now(dt.timezone.utc).isoformat(), tasks=[])
for task in plan["tasks"]:
    status["active"] = task["name"]
    status["state"] = "running"
    (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    command = [
        ".venv/bin/python", "scripts/chp_run.py", "--root", str(root),
        "--name", task["name"], "--", *task["argv"],
    ]
    result = subprocess.run(command, env=env)
    status["tasks"].append(dict(name=task["name"], exit_code=result.returncode))
    if result.returncode:
        status["state"] = "failed"
        (root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        raise SystemExit(result.returncode)
status.update(state="completed", active=None, end_utc=dt.datetime.now(dt.timezone.utc).isoformat())
(root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
