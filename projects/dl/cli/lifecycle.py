"""
DL experiment status reporter and job submitter (SLURM + SGE).

Subcommands:
    status  — Query chunk CSVs + scheduler, print JSON status report to stdout.
    submit  — Submit an array job, print job IDs as JSON, exit.

Usage:
    python -m projects.dl.cli.lifecycle status \
        --result-dir results/dl_patchts --job-ids 12345678 --total-chunks 10
    python -m projects.dl.cli.lifecycle status \
        --result-dir results/dl_patchts --job-ids 12345678 --total-chunks 10 --scheduler sge
    python -m projects.dl.cli.lifecycle submit \
        --experiment patchts --total-chunks 10 --result-dir results/dl_patchts
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from core.backends import PROJECT_ROOT, get_backend
from projects.dl.cli.submit import DL_SGE_PASS_ENV_KEYS, DL_SGE_SCRIPT, DL_SLURM_SCRIPT, build_job_env

logger = logging.getLogger(__name__)

DEFAULT_LOG_DIR = "/scratch1/jc_905/logs"

_CHUNK_RE = re.compile(r"(?:results_)?chunk_(\d+)\.csv$", re.IGNORECASE)


def _extract_chunk_id(filename: str) -> int | None:
    """Return the integer chunk id embedded in *filename*, or None."""
    m = _CHUNK_RE.search(filename)
    if m:
        return int(m.group(1))
    m2 = re.search(r"_chunk_(\d+)\.csv$", filename, re.IGNORECASE)
    return int(m2.group(1)) if m2 else None


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def check_result_csvs(result_dir: str, total_chunks: int) -> dict[int, dict]:
    """Check result CSVs on disk. Validate header + at least 1 data row."""
    results: dict[int, dict] = {}
    seen: set[str] = set()
    rdir = Path(result_dir).resolve()

    for pattern in [str(rdir / "results_chunk_*.csv"), str(rdir / "*_chunk_*.csv")]:
        for path in glob.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            chunk_id = _extract_chunk_id(os.path.basename(path))
            if chunk_id is None or chunk_id < 1 or chunk_id > total_chunks:
                continue
            try:
                with open(path, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header is None:
                        continue
                    row_count = sum(1 for _ in reader)
                    if row_count < 1:
                        continue
                results[chunk_id] = {"status": "complete", "csv_rows": row_count}
            except OSError:
                continue
    return results


def query_sacct(job_ids: list[str]) -> dict[int, dict]:
    """Query sacct for all job IDs. Returns {task_id: {state, exit_code, job_id}}."""
    task_info: dict[int, dict] = {}

    for job_id in job_ids:
        cmd = [
            "sacct",
            "--clusters=discovery",
            "-j",
            job_id,
            "--format=JobID,State,ExitCode",
            "--noheader",
            "--parsable2",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {"error": "sacct_unavailable"}

        if result.returncode != 0 or not result.stdout.strip():
            if not task_info:
                return {"error": "sacct_unavailable"}
            continue

        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 3:
                continue
            job_field, state, exit_code = parts[0], parts[1], parts[2]
            if "_" not in job_field:
                continue
            try:
                tid = int(job_field.split("_")[1])
            except (IndexError, ValueError):
                continue
            task_info[tid] = {
                "state": state,
                "exit_code": exit_code,
                "job_id": job_id,
            }

    if not task_info:
        return {"error": "sacct_unavailable"}
    return task_info


# SGE state code → normalized state
_SGE_STATE_MAP: dict[str, str] = {
    "r": "RUNNING",
    "t": "RUNNING",
    "Rr": "RUNNING",
    "Rt": "RUNNING",
    "qw": "PENDING",
    "hqw": "PENDING",
    "Eqw": "FAILED",
    "Ehqw": "FAILED",
    "dr": "CANCELLED",
    "dt": "CANCELLED",
    "dRr": "CANCELLED",
    "dRt": "CANCELLED",
    "ds": "CANCELLED",
    "dS": "CANCELLED",
    "dT": "CANCELLED",
}


def _expand_task_range(spec: str) -> list[int]:
    """Expand an SGE task range like '3-10:1' or '5' into a list of ints."""
    spec = spec.strip()
    if not spec or spec == "undefined":
        return []
    m = re.match(r"(\d+)(?:-(\d+)(?::(\d+))?)?", spec)
    if not m:
        return []
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else start
    step = int(m.group(3)) if m.group(3) else 1
    return list(range(start, end + 1, step))


def query_sge(job_ids: list[str]) -> dict[int, dict]:
    """Query SGE via qstat + qacct. Returns {task_id: {state, exit_code, job_id}}.

    Uses the same return schema as query_sacct so report_status can
    consume either interchangeably.
    """
    task_info: dict[int, dict] = {}

    # --- Phase 1: qstat for running/pending tasks ---
    try:
        result = subprocess.run(
            ["qstat", "-u", os.environ.get("USER", "jamesdc1")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        qstat_out = result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        qstat_out = ""

    job_id_set = set(job_ids)
    for line in qstat_out.strip().splitlines():
        cols = line.split()
        if len(cols) < 5:
            continue
        jid = cols[0].strip()
        if jid not in job_id_set:
            continue
        state_code = cols[4].strip()
        normalized = _SGE_STATE_MAP.get(state_code, "UNKNOWN")
        # ja-task-ID is the last column
        task_spec = cols[-1].strip() if len(cols) >= 9 else ""
        for tid in _expand_task_range(task_spec):
            task_info[tid] = {
                "state": normalized,
                "exit_code": None,
                "job_id": jid,
            }

    # --- Phase 2: qacct for finished tasks not yet seen ---
    for job_id in job_ids:
        try:
            result = subprocess.run(
                ["qacct", "-j", job_id],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        if result.returncode != 0:
            continue

        # Parse blocks separated by ===== lines
        current: dict[str, str] = {}
        for raw_line in result.stdout.splitlines():
            if raw_line.startswith("====="):
                if current:
                    _process_qacct_block(current, job_id, task_info)
                    current = {}
                continue
            parts = raw_line.split(None, 1)
            if len(parts) == 2:
                current[parts[0]] = parts[1].strip()
        if current:
            _process_qacct_block(current, job_id, task_info)

    if not task_info:
        return {"error": "sge_unavailable"}
    return task_info


def _process_qacct_block(
    block: dict[str, str],
    job_id: str,
    task_info: dict[int, dict],
) -> None:
    """Extract task status from a single qacct block, skip if already in task_info."""
    tid_str = block.get("taskid", "")
    if not tid_str or tid_str == "undefined":
        return
    try:
        tid = int(tid_str)
    except ValueError:
        return
    if tid in task_info:
        return  # qstat data takes precedence (more current)

    exit_status = block.get("exit_status", "0")
    failed = block.get("failed", "0")
    try:
        exit_int = int(exit_status)
        failed_int = int(failed.split()[0]) if failed else 0
    except ValueError:
        exit_int, failed_int = -1, -1

    if exit_int == 0 and failed_int == 0:
        state = "COMPLETED"
    elif failed_int == 100:
        state = "TIMEOUT"
    elif failed_int != 0:
        state = "NODE_FAIL"
    else:
        state = "FAILED"

    task_info[tid] = {
        "state": state,
        "exit_code": exit_status,
        "job_id": job_id,
    }


def log_event(audit_path: str | Path, action: str, **details) -> None:
    """Append a JSON event to the lifecycle audit trail."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "action": action,
        **details,
    }
    try:
        with open(audit_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError as exc:
        logger.warning("Failed to write audit log: %s", exc)


SGE_SCRATCH_DIR = os.environ.get("SCRATCH", "/u/scratch/j/jamesdc1")


def get_err_log_paths(
    job_ids: list[str],
    total_chunks: int,
    log_dir: str = DEFAULT_LOG_DIR,
    scheduler: str = "slurm",
    job_name: str = "",
) -> dict[int, str]:
    """For each chunk, find the most recent log path that exists on disk."""
    paths: dict[int, str] = {}
    for tid in range(1, total_chunks + 1):
        for job_id in reversed(job_ids):
            if scheduler == "sge":
                # SGE: <job_name>.o<job_id>.<task_id> in $SCRATCH
                p = os.path.join(SGE_SCRATCH_DIR, f"{job_name}.o{job_id}.{tid}")
            else:
                p = os.path.join(log_dir, f"slurm-{job_id}_{tid}.err")
            if os.path.isfile(p):
                paths[tid] = p
                break
    return paths


def _detect_scheduler(result_dir: str) -> str:
    """Auto-detect scheduler from experiment metadata, falling back to probing."""
    meta_path = Path(result_dir) / "experiment_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            backend = meta.get("backend", "")
            if "sge" in backend:
                return "sge"
            if "slurm" in backend:
                return "slurm"
        except (json.JSONDecodeError, OSError):
            pass
    # Probe: try sacct (SLURM-only command)
    try:
        result = subprocess.run(
            ["sacct", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "slurm"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "sge"


def report_status(
    result_dir: str,
    job_ids: list[str],
    total_chunks: int,
    experiment: str = "",
    scheduler: str | None = None,
) -> dict:
    """Assemble full JSON status report."""
    csv_results = check_result_csvs(result_dir, total_chunks)

    if scheduler is None:
        scheduler = _detect_scheduler(result_dir)

    if scheduler == "sge":
        job_info = query_sge(job_ids) if job_ids else {}
    else:
        job_info = query_sacct(job_ids) if job_ids else {}
    query_error = job_info.pop("error", None)

    complete_ids = set(csv_results)
    chunks: dict[str, dict] = {}
    summary = {"complete": 0, "running": 0, "pending": 0, "failed": 0, "unknown": 0}

    active_states = {"RUNNING", "REQUEUED", "CONFIGURING"}
    pending_states = {"PENDING"}
    failed_states = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}

    for tid in range(1, total_chunks + 1):
        if tid in complete_ids:
            chunks[str(tid)] = csv_results[tid]
            summary["complete"] += 1
        elif tid in job_info:
            info = job_info[tid]
            state = info["state"]
            if state in active_states:
                cat = "running"
            elif state in pending_states:
                cat = "pending"
            elif state in failed_states or state.startswith("CANCELLED"):
                cat = "failed"
            else:
                cat = "unknown"
            chunks[str(tid)] = {"status": cat, **info}
            summary[cat] += 1
        else:
            chunks[str(tid)] = {"status": "unknown"}
            summary["unknown"] += 1

    # Error log paths for non-complete chunks
    failed_or_unknown = [tid for tid in range(1, total_chunks + 1) if tid not in complete_ids]
    job_name = f"dl_{experiment}" if experiment else ""
    all_err = (
        get_err_log_paths(job_ids, total_chunks, scheduler=scheduler, job_name=job_name)
        if job_ids
        else {}
    )
    err_paths = {str(tid): all_err[tid] for tid in failed_or_unknown if tid in all_err}

    report = {
        "experiment": experiment,
        "result_dir": str(Path(result_dir).resolve()),
        "total_chunks": total_chunks,
        "scheduler": scheduler,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunks": chunks,
        "summary": summary,
        "err_log_paths": err_paths,
    }
    if query_error:
        report["query_error"] = query_error
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DL experiment status reporter and job submitter (SLURM + SGE).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- status ---
    st = sub.add_parser("status", help="Print JSON status report to stdout.")
    st.add_argument("--result-dir", required=True, help="Result directory.")
    st.add_argument("--job-ids", type=str, default="", help="Comma-separated SLURM job IDs.")
    st.add_argument("--total-chunks", type=int, required=True)
    st.add_argument("--experiment", type=str, default="")
    st.add_argument(
        "--scheduler",
        choices=["slurm", "sge"],
        default=None,
        help="Scheduler type (auto-detected from metadata if omitted).",
    )

    # --- submit ---
    sm = sub.add_parser("submit", help="Submit array job, print job IDs, exit.")
    sm.add_argument("--experiment", required=True, choices=["patchts", "ae_ridge"])
    sm.add_argument("--total-chunks", type=int, default=10)
    sm.add_argument("--result-dir", default=None)
    sm.add_argument("--batch-size", type=int, default=None)
    sm.add_argument("--epochs", type=int, default=None)
    sm.add_argument("--learning-rate", type=float, default=None)
    sm.add_argument("--train-window", type=int, default=None)
    sm.add_argument("--input-path", type=str, default=None)
    sm.add_argument("--weights-dir", type=str, default=None)
    sm.add_argument("--backend", type=str, default="slurm", help="HPC backend (slurm, sge, sge-remote).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.command == "status":
        job_ids = [j.strip() for j in args.job_ids.split(",") if j.strip()]
        report = report_status(
            args.result_dir,
            job_ids,
            args.total_chunks,
            args.experiment,
            scheduler=args.scheduler,
        )
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")

    elif args.command == "submit":
        result_dir = args.result_dir or str(PROJECT_ROOT / "results" / f"dl_{args.experiment}")
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        env_kwargs: dict = {}
        for key in ("batch_size", "epochs", "learning_rate", "train_window", "input_path", "weights_dir"):
            val = getattr(args, key, None)
            if val is not None:
                env_kwargs[key] = val

        job_env = build_job_env(args.experiment, result_dir, args.total_chunks, **env_kwargs)
        backend_name = args.backend
        if backend_name in ("sge", "sge-remote"):
            backend = get_backend(backend_name, script=DL_SGE_SCRIPT, pass_env_keys=DL_SGE_PASS_ENV_KEYS)
        else:
            backend = get_backend(backend_name, script=DL_SLURM_SCRIPT)
        job_name = f"dl_{args.experiment}"

        submissions = backend.submit_array_tracked(job_name, args.total_chunks, args.total_chunks, job_env)
        job_ids = [job_id for _, job_id in submissions]

        # Write experiment metadata for scheduler auto-detection
        meta_path = Path(result_dir) / "experiment_meta.json"
        meta = {
            "backend": backend_name,
            "experiment": args.experiment,
            "total_chunks": args.total_chunks,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

        audit_path = Path(result_dir) / "lifecycle.jsonl"
        log_event(audit_path, "submitted", job_ids=job_ids, total_chunks=args.total_chunks)

        json.dump({"job_ids": job_ids}, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
