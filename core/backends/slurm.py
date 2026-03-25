"""SLURM backend — submits array jobs via sbatch."""

import os

from core.backends import HPCBackend, register

DEFAULT_SLURM_ACCOUNT = "pollok_1603"
DEFAULT_SLURM_CLUSTER = "discovery"
DEFAULT_SLURM_LOG_DIR = os.path.join(os.environ.get("SCRATCH1", "/scratch1"), os.environ.get("USER", "jc_905"), "logs")


@register("slurm")
class SlurmBackend(HPCBackend):
    def __init__(
        self,
        script: str | None = None,
        account: str | None = None,
        cluster: str | None = None,
        log_dir: str | None = None,
    ):
        if script is None:
            raise ValueError("SlurmBackend requires a 'script' path")
        self.script = script
        self.account = account or os.environ.get("SLURM_ACCOUNT", DEFAULT_SLURM_ACCOUNT)
        self.cluster = cluster or os.environ.get("SLURM_CLUSTER", DEFAULT_SLURM_CLUSTER)
        self.log_dir = log_dir or os.environ.get("SLURM_LOG_DIR", DEFAULT_SLURM_LOG_DIR)

    def _build_command(self, task_range: str, job_name: str, job_env: dict[str, str]) -> list[str]:
        return [
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
            "--error",
            f"{self.log_dir}/slurm-%A_%a.err",
            self.script,
        ]
