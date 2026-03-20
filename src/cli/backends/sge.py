"""SGE (Sun/Univa Grid Engine) backend — submits array jobs via qsub."""

import os
import subprocess

from src.cli.backends import HPCBackend, PROJECT_ROOT, register

DEFAULT_SGE_LOG_DIR = str(PROJECT_ROOT / "logs")
DEFAULT_SUBMISSION_SCRIPT = str(PROJECT_ROOT / "infra" / "sge" / "submit_hoffman2.sh")


@register("sge")
class SGEBackend(HPCBackend):
    def __init__(
        self,
        script: str = DEFAULT_SUBMISSION_SCRIPT,
        log_dir: str | None = None,
    ):
        self.script = script
        self.log_dir = log_dir or os.environ.get("SGE_LOG_DIR", DEFAULT_SGE_LOG_DIR)

    def submit_array(self, job_name, total_chunks, tasks_per_array, job_env):
        os.makedirs(self.log_dir, exist_ok=True)

        # Build the -v flag to pass env vars into the job script
        pass_vars = ",".join(
            f"{k}={v}"
            for k, v in job_env.items()
            if k in ("TOTAL_CHUNKS", "EXOG_COLS", "RESULT_DIR", "MODEL_TYPE", "EXTRA_ARGS")
        )

        start_task = 1
        while start_task <= total_chunks:
            end_task = min(start_task + tasks_per_array - 1, total_chunks)
            task_range = f"{start_task}-{end_task}"
            cmd = [
                "qsub",
                "-t",
                task_range,
                "-N",
                job_name,
                "-o",
                self.log_dir,
                "-j",
                "y",
                "-v",
                pass_vars,
                self.script,
            ]
            subprocess.run(cmd, env=job_env, check=True, cwd=PROJECT_ROOT)
            start_task = end_task + 1
