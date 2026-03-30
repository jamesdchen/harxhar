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
import json
import logging
import os
import sys
from pathlib import Path

from hpc import load_clusters_config, load_project_config
from hpc.lifecycle import (
    check_results,
    detect_scheduler,
    log_event,
)
from hpc.lifecycle import (
    report_status as _hpc_report_status,
)

from core.backends import PROJECT_ROOT, get_backend
from projects.dl.cli.submit import DL_SGE_PASS_ENV_KEYS, _resolve_dl_template, build_job_env

logger = logging.getLogger(__name__)


def _get_cluster_paths() -> tuple[str, str]:
    """Return (log_dir, scratch_dir) from cluster config."""
    project_cfg = load_project_config()
    cluster_name = project_cfg.get("cluster", "hoffman2")
    clusters = load_clusters_config()
    cluster = clusters[cluster_name]
    scratch = cluster.get("scratch", "")
    log_dir = f"{scratch}/logs" if cluster.get("scheduler") == "slurm" else ""
    return log_dir, scratch


def check_result_csvs(result_dir: str, total_chunks: int) -> dict[int, dict]:
    """Check result CSVs on disk. Backward-compatible wrapper."""
    return check_results(result_dir, total_chunks)


def report_status(
    result_dir: str,
    job_ids: list[str],
    total_chunks: int,
    experiment: str = "",
    scheduler: str | None = None,
) -> dict:
    """Assemble full JSON status report with experiment field."""
    if scheduler is None:
        scheduler = detect_scheduler(result_dir)

    log_dir, scratch_dir = _get_cluster_paths()

    project_cfg = load_project_config()
    cluster_name = project_cfg.get("cluster", "hoffman2")
    clusters = load_clusters_config()
    cluster = clusters[cluster_name]

    report = _hpc_report_status(
        result_dir,
        job_ids,
        total_chunks,
        scheduler=scheduler,
        log_dir=log_dir,
        scratch_dir=scratch_dir,
        job_name=f"dl_{experiment}" if experiment else "",
        slurm_cluster=cluster_name if cluster.get("scheduler") == "slurm" else "",
        sge_user=cluster.get("user", os.environ.get("USER", "")),
    )
    report["experiment"] = experiment
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
        script = _resolve_dl_template(backend_name)
        if backend_name in ("sge", "sge-remote"):
            backend = get_backend(backend_name, script=script, pass_env_keys=DL_SGE_PASS_ENV_KEYS)
        else:
            backend = get_backend(backend_name, script=script)
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
