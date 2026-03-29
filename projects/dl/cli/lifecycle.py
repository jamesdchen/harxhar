"""
SLURM DL experiment status reporter and job submitter.

Subcommands:
    status  — Query chunk CSVs + sacct, print JSON status report to stdout.
    submit  — Submit an array job, print job IDs as JSON, exit.

Usage:
    python -m projects.dl.cli.lifecycle status \
        --result-dir results/dl_patchts --job-ids 12345678 --total-chunks 10
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
    """Query sacct for all job IDs. Returns {task_id: {sacct_state, exit_code, job_id}}."""
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
                "sacct_state": state,
                "exit_code": exit_code,
                "job_id": job_id,
            }

    if not task_info:
        return {"error": "sacct_unavailable"}
    return task_info


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


def get_err_log_paths(
    job_ids: list[str],
    total_chunks: int,
    log_dir: str = DEFAULT_LOG_DIR,
) -> dict[int, str]:
    """For each chunk, find the most recent .err log path that exists on disk."""
    paths: dict[int, str] = {}
    for tid in range(1, total_chunks + 1):
        for job_id in reversed(job_ids):
            p = os.path.join(log_dir, f"slurm-{job_id}_{tid}.err")
            if os.path.isfile(p):
                paths[tid] = p
                break
    return paths


def report_status(
    result_dir: str,
    job_ids: list[str],
    total_chunks: int,
    experiment: str = "",
) -> dict:
    """Assemble full JSON status report."""
    csv_results = check_result_csvs(result_dir, total_chunks)
    sacct = query_sacct(job_ids) if job_ids else {}
    sacct_error = sacct.pop("error", None)

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
        elif tid in sacct:
            info = sacct[tid]
            state = info["sacct_state"]
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
    all_err = get_err_log_paths(job_ids, total_chunks) if job_ids else {}
    err_paths = {str(tid): all_err[tid] for tid in failed_or_unknown if tid in all_err}

    report = {
        "experiment": experiment,
        "result_dir": str(Path(result_dir).resolve()),
        "total_chunks": total_chunks,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunks": chunks,
        "summary": summary,
        "err_log_paths": err_paths,
    }
    if sacct_error:
        report["sacct_error"] = sacct_error
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLURM DL experiment status reporter and job submitter.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- status ---
    st = sub.add_parser("status", help="Print JSON status report to stdout.")
    st.add_argument("--result-dir", required=True, help="Result directory.")
    st.add_argument("--job-ids", type=str, default="", help="Comma-separated SLURM job IDs.")
    st.add_argument("--total-chunks", type=int, required=True)
    st.add_argument("--experiment", type=str, default="")

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
        report = report_status(args.result_dir, job_ids, args.total_chunks, args.experiment)
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

        audit_path = Path(result_dir) / "lifecycle.jsonl"
        log_event(audit_path, "submitted", job_ids=job_ids, total_chunks=args.total_chunks)

        json.dump({"job_ids": job_ids}, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
