"""
SLURM Job Lifecycle Manager for DL experiments.

Monitors array jobs, diagnoses failures, resubmits with appropriate fixes,
and runs aggregation once all chunks complete.

Usage:
    python -m projects.dl.cli.lifecycle --experiment patchts
    python -m projects.dl.cli.lifecycle --experiment ae_ridge --result-dir results/dl_ae_ridge \
        --no-submit --job-ids 12345678,12345679
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import glob
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from core.backends import PROJECT_ROOT, get_backend
from projects.dl.cli.submit import DL_SLURM_SCRIPT, build_job_env

logger = logging.getLogger(__name__)

AGGREGATE_SCRIPT = str(PROJECT_ROOT / "projects" / "ml" / "scripts" / "aggregate.py")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FailureType(enum.Enum):
    OOM = "oom"
    TIMEOUT = "timeout"
    NODE_FAILURE = "node_failure"
    CODE_BUG = "code_bug"
    UNKNOWN = "unknown"


@dataclasses.dataclass
class ChunkStatus:
    task_id: int
    status: str
    retries: int = 0
    failure_type: FailureType | None = None
    last_job_id: str | None = None


# ---------------------------------------------------------------------------
# Chunk-ID extraction
# ---------------------------------------------------------------------------

_CHUNK_RE = re.compile(r"(?:results_)?chunk_(\d+)\.csv$", re.IGNORECASE)


def _extract_chunk_id(filename: str) -> int | None:
    """Return the integer chunk id embedded in *filename*, or None."""
    m = _CHUNK_RE.search(filename)
    if m:
        return int(m.group(1))
    # Fallback: any *_chunk_<N>.csv pattern
    m2 = re.search(r"_chunk_(\d+)\.csv$", filename, re.IGNORECASE)
    return int(m2.group(1)) if m2 else None


# ---------------------------------------------------------------------------
# LifecycleManager
# ---------------------------------------------------------------------------


class LifecycleManager:
    """Monitors, diagnoses, and retries a SLURM DL array job."""

    def __init__(
        self,
        experiment: str,
        result_dir: str,
        total_chunks: int,
        poll_interval: int = 120,
        max_retries: int = 3,
        backend_name: str = "slurm",
        dry_run: bool = False,
        no_submit: bool = False,
        job_ids: list[str] | None = None,
        **env_kwargs,
    ) -> None:
        self.experiment = experiment
        self.result_dir = Path(result_dir).resolve()
        self.total_chunks = total_chunks
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.backend_name = backend_name
        self.dry_run = dry_run
        self.no_submit = no_submit
        self.env_kwargs = env_kwargs

        # Track per-chunk state (1-indexed)
        self.chunks: dict[int, ChunkStatus] = {
            tid: ChunkStatus(task_id=tid, status="pending") for tid in range(1, total_chunks + 1)
        }

        # All job IDs we need to poll (initial + resubmissions)
        self.tracked_job_ids: list[str] = list(job_ids or [])

        # Original batch size for OOM retry scaling
        self._original_batch_size: int | None = env_kwargs.get("batch_size")

        # Backend (resolved lazily for dry-run)
        if dry_run:
            self._backend = get_backend("dry-run")
        else:
            self._backend = get_backend(backend_name, script=DL_SLURM_SCRIPT)

        # Ensure output directory exists
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # Audit trail path
        self._audit_path = self.result_dir / "lifecycle.jsonl"

        # Shutdown flag
        self._shutting_down = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main lifecycle loop."""
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)

        if self.dry_run:
            self._initial_submit()
            logger.info("Dry run complete — exiting.")
            return

        if not self.no_submit:
            self._initial_submit()
        elif not self.tracked_job_ids:
            logger.error("--no-submit requires --job-ids")
            sys.exit(1)
        else:
            self._log_event("resumed", job_ids=self.tracked_job_ids)
            logger.info("Monitoring existing jobs: %s", ", ".join(self.tracked_job_ids))

        # Poll loop
        while True:
            time.sleep(self.poll_interval)
            if self._shutting_down:
                return

            completed = self._check_results()
            logger.info("Completed %d / %d chunks", len(completed), self.total_chunks)

            if len(completed) == self.total_chunks:
                self._log_event("all_complete")
                logger.info("All %d chunks complete.", self.total_chunks)
                break

            # Query SLURM for running / failed states
            states = self._poll_status()
            self._log_event("waiting", completed=len(completed), states=dict(states))

            # If anything is still running or pending, keep waiting
            active_states = {"RUNNING", "PENDING", "REQUEUED", "CONFIGURING"}
            if any(s in active_states for s in states.values()):
                still_active = sum(1 for s in states.values() if s in active_states)
                logger.info("%d tasks still active, continuing.", still_active)
                continue

            # Identify failed / missing chunks
            failed_tids: list[int] = []
            for tid in range(1, self.total_chunks + 1):
                if tid in completed:
                    continue
                chunk = self.chunks[tid]
                if chunk.retries >= self.max_retries:
                    if chunk.status != "give_up":
                        chunk.status = "give_up"
                        self._log_event(
                            "give_up",
                            task_id=tid,
                            retries=chunk.retries,
                            failure_type=chunk.failure_type.value if chunk.failure_type else None,
                        )
                        logger.warning(
                            "Chunk %d: retries exhausted (%d), giving up.",
                            tid,
                            chunk.retries,
                        )
                    continue
                self._diagnose_failure(tid)
                failed_tids.append(tid)

            if failed_tids:
                self._resubmit_failed(failed_tids)
            else:
                # Nothing running, nothing to resubmit — all remaining gave up
                break

        # Final status
        completed = self._check_results()
        if len(completed) == self.total_chunks:
            self._run_aggregation()
        else:
            missing = sorted(set(range(1, self.total_chunks + 1)) - completed)
            self._log_event("incomplete", missing_chunks=missing)
            logger.warning(
                "Incomplete: %d / %d chunks missing (%s).",
                len(missing),
                self.total_chunks,
                ", ".join(str(c) for c in missing),
            )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def _initial_submit(self) -> None:
        """Submit the initial array job and capture job IDs."""
        job_name = f"dl_{self.experiment}"
        job_env = build_job_env(
            self.experiment,
            str(self.result_dir),
            self.total_chunks,
            **self.env_kwargs,
        )

        if self.dry_run:
            self._backend.submit_array(job_name, self.total_chunks, self.total_chunks, job_env)
            return

        submissions = self._backend.submit_array_tracked(job_name, self.total_chunks, self.total_chunks, job_env)
        for task_range, job_id in submissions:
            self.tracked_job_ids.append(job_id)
            logger.info("Submitted %s array=%s job_id=%s", job_name, task_range, job_id)

        self._log_event(
            "submitted",
            job_ids=self.tracked_job_ids,
            total_chunks=self.total_chunks,
        )

    # ------------------------------------------------------------------
    # Result checking
    # ------------------------------------------------------------------

    def _check_results(self) -> set[int]:
        """Return set of chunk IDs whose result CSVs are complete on disk."""
        now = time.time()
        mtime_cutoff = now - 30  # ignore files still being written
        completed: set[int] = set()

        patterns = [
            str(self.result_dir / "results_chunk_*.csv"),
            str(self.result_dir / "*_chunk_*.csv"),
        ]
        seen_paths: set[str] = set()

        for pattern in patterns:
            for path in glob.glob(pattern):
                if path in seen_paths:
                    continue
                seen_paths.add(path)

                basename = os.path.basename(path)
                chunk_id = _extract_chunk_id(basename)
                if chunk_id is None:
                    continue

                try:
                    st = os.stat(path)
                except OSError:
                    continue

                if st.st_size == 0:
                    continue
                if st.st_mtime > mtime_cutoff:
                    continue

                completed.add(chunk_id)
                self.chunks[chunk_id].status = "complete"

        return completed

    # ------------------------------------------------------------------
    # SLURM status polling
    # ------------------------------------------------------------------

    def _poll_status(self) -> dict[int, str]:
        """Query sacct for all tracked jobs, return {task_id: state}."""
        task_states: dict[int, str] = {}

        for job_id in self.tracked_job_ids:
            cmd = [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode",
                "--noheader",
                "--parsable2",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("sacct call failed for job %s", job_id)
                continue

            for line in result.stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                job_field, state = parts[0], parts[1]
                # Only array task entries contain '_'
                if "_" not in job_field:
                    continue
                try:
                    tid = int(job_field.split("_")[1])
                except (IndexError, ValueError):
                    continue
                # Keep the most severe state per task
                task_states[tid] = state
                # Update internal tracking
                if tid in self.chunks:
                    self.chunks[tid].last_job_id = job_id

        return task_states

    # ------------------------------------------------------------------
    # Failure diagnosis
    # ------------------------------------------------------------------

    def _diagnose_failure(self, tid: int) -> None:
        """Read .err log for *tid* and classify the failure type."""
        chunk = self.chunks[tid]

        # Check sacct state first
        sacct_state = self._get_sacct_state(tid)

        if sacct_state == "TIMEOUT":
            chunk.failure_type = FailureType.TIMEOUT
        elif sacct_state in ("NODE_FAIL", "CANCELLED"):
            chunk.failure_type = FailureType.NODE_FAILURE
        else:
            # Read stderr log for diagnosis
            err_content = self._read_err_log(tid)
            if err_content is None:
                chunk.failure_type = FailureType.UNKNOWN
            elif "CUDA out of memory" in err_content or "OutOfMemoryError" in err_content:
                chunk.failure_type = FailureType.OOM
            elif "exceeded timeout" in err_content:
                chunk.failure_type = FailureType.TIMEOUT
            elif "Traceback" in err_content:
                chunk.failure_type = FailureType.CODE_BUG
            else:
                chunk.failure_type = FailureType.UNKNOWN

        self._log_event(
            "diagnosed",
            task_id=tid,
            failure_type=chunk.failure_type.value,
            retries=chunk.retries,
        )
        logger.info(
            "Chunk %d: diagnosed as %s (retry %d/%d)",
            tid,
            chunk.failure_type.value,
            chunk.retries,
            self.max_retries,
        )

    def _get_sacct_state(self, tid: int) -> str | None:
        """Return the latest sacct state for a specific task ID, or None."""
        for job_id in reversed(self.tracked_job_ids):
            cmd = [
                "sacct",
                "-j",
                f"{job_id}_{tid}",
                "--format=State",
                "--noheader",
                "--parsable2",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                state = result.stdout.strip().splitlines()
                if state:
                    return state[-1].strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return None

    def _read_err_log(self, tid: int) -> str | None:
        """Read the .err log for a given task ID from all tracked jobs."""
        log_dir = self._backend.log_dir if hasattr(self._backend, "log_dir") else "/scratch1/jc_905/logs"

        for job_id in reversed(self.tracked_job_ids):
            err_path = os.path.join(log_dir, f"slurm-{job_id}_{tid}.err")
            try:
                with open(err_path) as f:
                    return f.read()
            except FileNotFoundError:
                continue
        return None

    # ------------------------------------------------------------------
    # Resubmission
    # ------------------------------------------------------------------

    def _resubmit_failed(self, tids: list[int]) -> None:
        """Group failed tasks by failure type and resubmit with fixes."""
        # Group by failure type
        groups: dict[FailureType, list[int]] = {}
        for tid in tids:
            ft = self.chunks[tid].failure_type or FailureType.UNKNOWN
            groups.setdefault(ft, []).append(tid)

        backend = self._backend
        cluster = getattr(backend, "cluster", "discovery")
        account = getattr(backend, "account", "pollok_1603")
        log_dir = getattr(backend, "log_dir", "/scratch1/jc_905/logs")
        script = getattr(backend, "script", DL_SLURM_SCRIPT)
        job_name = f"dl_{self.experiment}"

        for failure_type, chunk_ids in groups.items():
            array_spec = ",".join(str(t) for t in sorted(chunk_ids))

            # Build sbatch command with overrides
            cmd = [
                "sbatch",
                f"--clusters={cluster}",
                f"--array={array_spec}",
                f"--job-name={job_name}",
                f"--account={account}",
                f"--output={log_dir}/slurm-%A_%a.out",
                f"--error={log_dir}/slurm-%A_%a.err",
            ]

            # Build env with potential modifications
            env_kwargs = dict(self.env_kwargs)

            if failure_type == FailureType.OOM:
                retry_num = max(self.chunks[t].retries for t in chunk_ids)
                if retry_num >= 1:
                    # 2nd+ retry: 256G, quarter batch size
                    cmd.append("--mem=256G")
                    if self._original_batch_size is not None:
                        env_kwargs["batch_size"] = self._original_batch_size // 4
                else:
                    # 1st retry: 192G, halve batch size
                    cmd.append("--mem=192G")
                    if self._original_batch_size is not None:
                        env_kwargs["batch_size"] = self._original_batch_size // 2

            elif failure_type == FailureType.TIMEOUT:
                cmd.append("--time=10:00:00")

            # Append the script
            cmd.append(script)

            # Build job env
            job_env = build_job_env(
                self.experiment,
                str(self.result_dir),
                self.total_chunks,
                **env_kwargs,
            )

            logger.info(
                "Resubmitting chunks [%s] (failure=%s): %s",
                array_spec,
                failure_type.value,
                " ".join(cmd),
            )

            result = subprocess.run(
                cmd,
                env=job_env,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(
                    "Resubmission failed for %s chunks [%s]: %s",
                    failure_type.value,
                    array_spec,
                    result.stderr.strip(),
                )
                continue

            # Parse new job ID
            match = re.search(r"(\d+)", result.stdout)
            new_job_id = match.group(1) if match else "unknown"
            self.tracked_job_ids.append(new_job_id)

            # Update chunk state
            for tid in chunk_ids:
                self.chunks[tid].retries += 1
                self.chunks[tid].status = "resubmitted"
                self.chunks[tid].last_job_id = new_job_id

            self._log_event(
                "resubmitted",
                job_id=new_job_id,
                failure_type=failure_type.value,
                task_ids=sorted(chunk_ids),
                array_spec=array_spec,
            )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _run_aggregation(self) -> None:
        """Run the aggregation script over completed results."""
        cmd = [
            sys.executable,
            AGGREGATE_SCRIPT,
            "--base_dir",
            str(self.result_dir),
        ]
        logger.info("Running aggregation: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

        if result.returncode != 0:
            logger.error("Aggregation failed:\n%s", result.stderr.strip())
            self._log_event("aggregation_failed", stderr=result.stderr.strip())
        else:
            logger.info("Aggregation complete.")
            self._log_event("aggregated")

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def _log_event(self, action: str, **details) -> None:
        """Append a JSON event to the lifecycle audit trail."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "action": action,
            "experiment": self.experiment,
            **details,
        }
        try:
            with open(self._audit_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit log: %s", exc)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _graceful_shutdown(self, signum, frame) -> None:
        """Handle SIGINT / SIGTERM by logging state and exiting."""
        self._shutting_down = True
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully.", sig_name)

        completed = self._check_results()
        pending = sorted(set(range(1, self.total_chunks + 1)) - completed)

        self._log_event(
            "shutdown",
            signal=sig_name,
            completed_chunks=len(completed),
            pending_chunks=pending,
            tracked_job_ids=self.tracked_job_ids,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLURM Job Lifecycle Manager for DL experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["patchts", "ae_ridge"],
        required=True,
        help="DL experiment to manage.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Output directory for chunk results. Defaults to results/dl_<experiment>.",
    )
    parser.add_argument("--total-chunks", type=int, default=10, help="Number of array tasks.")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between status polls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max resubmissions per chunk.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="slurm",
        help="HPC backend (slurm, sge, dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted, then exit.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Skip initial submission; use with --job-ids to monitor existing jobs.",
    )
    parser.add_argument(
        "--job-ids",
        type=str,
        default=None,
        help="Comma-separated SLURM job IDs to monitor (use with --no-submit).",
    )

    # Pass-through env kwargs
    parser.add_argument("--batch-size", type=int, default=None, help="Windows per batch.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--train-window", type=int, default=None, help="Training window size.")
    parser.add_argument("--input-path", type=str, default=None, help="Data directory.")
    parser.add_argument("--weights-dir", type=str, default=None, help="AE weights directory.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Collect pass-through kwargs
    env_kwargs: dict = {}
    for key in ("batch_size", "epochs", "learning_rate", "train_window", "input_path", "weights_dir"):
        val = getattr(args, key, None)
        if val is not None:
            env_kwargs[key] = val

    # Parse job IDs
    job_ids: list[str] | None = None
    if args.job_ids:
        job_ids = [jid.strip() for jid in args.job_ids.split(",") if jid.strip()]

    result_dir = args.result_dir or str(PROJECT_ROOT / "results" / f"dl_{args.experiment}")

    manager = LifecycleManager(
        experiment=args.experiment,
        result_dir=result_dir,
        total_chunks=args.total_chunks,
        poll_interval=args.poll_interval,
        max_retries=args.max_retries,
        backend_name=args.backend,
        dry_run=args.dry_run,
        no_submit=args.no_submit,
        job_ids=job_ids,
        **env_kwargs,
    )
    manager.run()


if __name__ == "__main__":
    main()
