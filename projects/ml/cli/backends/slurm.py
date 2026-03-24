"""SLURM backend — submits array jobs via sbatch."""

import os
import subprocess

from projects.ml.cli.backends import PROJECT_ROOT, HPCBackend, register

DEFAULT_SLURM_ACCOUNT = "pollok_1603"
DEFAULT_SLURM_CLUSTER = "discovery"
DEFAULT_SLURM_LOG_DIR = "/scratch1/jc_905/logs"
DEFAULT_SUBMISSION_SCRIPT = str(PROJECT_ROOT / "projects" / "ml" / "infra" / "slurm" / "submit_carc.slurm")


@register("slurm")
class SlurmBackend(HPCBackend):
    def __init__(
        self,
        script: str = DEFAULT_SUBMISSION_SCRIPT,
        account: str | None = None,
        cluster: str | None = None,
        log_dir: str | None = None,
    ):
        self.script = script
        self.account = account or os.environ.get("SLURM_ACCOUNT", DEFAULT_SLURM_ACCOUNT)
        self.cluster = cluster or os.environ.get("SLURM_CLUSTER", DEFAULT_SLURM_CLUSTER)
        self.log_dir = log_dir or os.environ.get("SLURM_LOG_DIR", DEFAULT_SLURM_LOG_DIR)

    def submit_array(self, job_name, total_chunks, tasks_per_array, job_env):
        os.makedirs(self.log_dir, exist_ok=True)

        start_task = 1
        while start_task <= total_chunks:
            end_task = min(start_task + tasks_per_array - 1, total_chunks)
            task_range = f"{start_task}-{end_task}"
            cmd = [
                "sbatch",
                f"--clusters={self.cluster}",
                "--array",
                task_range,
                "--job-name",
                job_name,
                "--account",
                self.account,
                "--output",
                f"{self.log_dir}/slurm-%A_%a.out",
                self.script,
            ]
            result = subprocess.run(
                cmd,
                env=job_env,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr_msg = result.stderr.strip() if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"sbatch failed (exit {result.returncode}) for array {task_range}:\n"
                    f"  command: {' '.join(cmd)}\n"
                    f"  stderr:  {stderr_msg}"
                )
            start_task = end_task + 1
