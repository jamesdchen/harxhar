"""SGE (Sun/Univa Grid Engine) backend — submits array jobs via qsub."""

import os
import subprocess

from core.backends import PROJECT_ROOT, HPCBackend, register


@register("sge")
class SGEBackend(HPCBackend):
    def __init__(
        self,
        script: str | None = None,
        log_dir: str | None = None,
        pass_env_keys: tuple[str, ...] = (),
    ):
        if script is None:
            raise ValueError("SGEBackend requires a 'script' path")
        self.script = script
        self.log_dir = log_dir or os.environ.get("SGE_LOG_DIR", str(PROJECT_ROOT / "logs"))
        self.pass_env_keys = pass_env_keys

    def submit_array(self, job_name, total_chunks, tasks_per_array, job_env):
        os.makedirs(self.log_dir, exist_ok=True)

        # Build the -v flag to pass env vars into the job script
        pass_vars = ",".join(f"{k}={v}" for k, v in job_env.items() if k in self.pass_env_keys)

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
            ]
            if pass_vars:
                cmd += ["-v", pass_vars]
            cmd.append(self.script)
            subprocess.run(cmd, env=job_env, check=True, cwd=PROJECT_ROOT)
            start_task = end_task + 1
