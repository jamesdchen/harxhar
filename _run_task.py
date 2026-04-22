"""Per-task runner invoked as the SLURM array EXECUTOR.

Reads `_hpc_dispatch.json` and runs the cmd for the task identified by the
`TASK_ID` env var (set by the SLURM array script). Supports two manifest
formats:

  - String: `{"tasks": {"0": "python ... --output-file foo/results_0.csv"}}`
    The output dir is inferred from `--output-file` and created.
  - Dict:   `{"tasks": {"0": {"cmd": "...", "result_dir": "results/foo"}}}`
    `result_dir` is exported as `RESULT_DIR` to the subprocess and created.

`TASK_ID` is always re-exported to the subprocess so commands that fall back
to `${RESULT_DIR}/results_chunk_${TASK_ID}.csv` resolve correctly.
"""

import json
import os
import re
import subprocess
import sys

manifest_path = "_hpc_dispatch.json"
with open(manifest_path) as f:
    manifest = json.load(f)

task_id = int(os.environ.get("TASK_ID", 0))
task_key = str(task_id)

if task_key not in manifest["tasks"]:
    print(f"ERROR: Task {task_id} not in manifest", file=sys.stderr)
    sys.exit(1)

task_spec = manifest["tasks"][task_key]

# Handle both formats: plain string or dict.
if isinstance(task_spec, str):
    cmd = task_spec
    result_dir = None
    chunk_id = None
else:
    cmd = task_spec.get("cmd", "")
    result_dir = task_spec.get("result_dir")
    chunk_id = task_spec.get("chunk_id")

# Build subprocess env: harden TASK_ID, propagate RESULT_DIR/CHUNK_ID if specified.
env = os.environ.copy()
env["TASK_ID"] = str(task_id)

if result_dir:
    os.makedirs(result_dir, exist_ok=True)
    env["RESULT_DIR"] = result_dir

if chunk_id is not None:
    env["CHUNK_ID"] = str(chunk_id)

# Backward compat: also infer dir from --output-file if present.
m = re.search(r"--output-file\s+([^\s]+)", cmd)
if m:
    out_dir = os.path.dirname(m.group(1))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

print(f"[Task {task_id}] RESULT_DIR={env.get('RESULT_DIR', '<unset>')} {cmd}", file=sys.stderr)
result = subprocess.run(cmd, shell=True, env=env)
sys.exit(result.returncode)
