"""Local backend — runs chunk backtests sequentially or with multiprocessing."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.cli.backends import PROJECT_ROOT, HPCBackend, register
from src.core.log import get_logger

logger = get_logger(__name__)


def _run_chunk(chunk_id: int, job_env: dict[str, str]) -> tuple[int, bool]:
    """Run a single chunk as a subprocess."""
    env = {**job_env, "SLURM_ARRAY_TASK_ID": str(chunk_id)}
    result_dir = env["RESULT_DIR"]
    model_type = env["MODEL_TYPE"]
    total_chunks = env["TOTAL_CHUNKS"]
    exog_cols = env.get("EXOG_COLS", "None")
    extra_args = env.get("EXTRA_ARGS", "")

    output_file = os.path.join(result_dir, f"results_chunk_{chunk_id}.csv")

    cmd = [
        sys.executable,
        "-m",
        "src.cli.executor",
        "--model",
        model_type,
        "--output-file",
        output_file,
        "--chunk-id",
        str(chunk_id),
        "--total-chunks",
        total_chunks,
    ]

    if exog_cols and exog_cols != "None":
        cmd.extend(["--exog-cols", exog_cols])

    if extra_args:
        cmd.extend(extra_args.split())

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        return chunk_id, True
    except subprocess.CalledProcessError as e:
        logger.error("Chunk %d failed: %s", chunk_id, e.stderr[-500:] if e.stderr else str(e))
        return chunk_id, False


@register("local")
class LocalBackend(HPCBackend):
    """Run experiments locally with optional parallelism."""

    def __init__(self, max_workers: int = 1):
        self.max_workers = max_workers

    def submit_array(self, job_name, total_chunks, tasks_per_array, job_env):
        logger.info("Running %d chunks locally (workers=%d): %s", total_chunks, self.max_workers, job_name)

        if self.max_workers == 1:
            for chunk_id in range(1, total_chunks + 1):
                cid, ok = _run_chunk(chunk_id, job_env)
                status = "OK" if ok else "FAILED"
                logger.info("  Chunk %d/%d: %s", cid, total_chunks, status)
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(_run_chunk, i, job_env): i for i in range(1, total_chunks + 1)}
                for done, future in enumerate(as_completed(futures), 1):
                    cid, ok = future.result()
                    status = "OK" if ok else "FAILED"
                    logger.info("  Chunk %d/%d: %s (%d/%d done)", cid, total_chunks, status, done, total_chunks)

        logger.info("Local run complete: %s", job_name)


@register("dry-run")
class DryRunBackend(HPCBackend):
    """Print what would be submitted without actually running anything."""

    def submit_array(self, job_name, total_chunks, tasks_per_array, job_env):
        result_dir = job_env.get("RESULT_DIR", "?")
        model_type = job_env.get("MODEL_TYPE", "?")
        extra_args = job_env.get("EXTRA_ARGS", "")
        exog_cols = job_env.get("EXOG_COLS", "None")

        print(f"  [DRY RUN] Job: {job_name}")
        print(f"            Model: {model_type}")
        print(f"            Chunks: 1-{total_chunks} (batches of {tasks_per_array})")
        print(f"            Output: {result_dir}")
        if exog_cols != "None":
            n_vars = len(exog_cols.split("|"))
            print(f"            Exog vars: {n_vars}")
        if extra_args:
            print(f"            Extra args: {extra_args}")
        print()
