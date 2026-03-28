"""Remote SGE backend — submits array jobs via qsub over SSH."""

from __future__ import annotations

import re

from core.backends import HPCBackend, register
from core.remote import REMOTE_REPO, ssh_run


@register("sge-remote")
class RemoteSGEBackend(HPCBackend):
    """SGE backend that runs qsub on the cluster via SSH.

    Unlike the local ``SGEBackend``, this backend does not call ``qsub``
    directly — it wraps each command in ``ssh_run()`` so submissions
    happen on the Hoffman2 login node.
    """

    def __init__(
        self,
        script: str | None = None,
        log_dir: str | None = None,
        pass_env_keys: tuple[str, ...] = (),
    ):
        if script is None:
            raise ValueError("RemoteSGEBackend requires a 'script' path")
        self.script = script
        self.log_dir = log_dir or f"{REMOTE_REPO}/logs"
        self.pass_env_keys = pass_env_keys

    def _build_command(
        self,
        task_range: str,
        job_name: str,
        job_env: dict[str, str],
    ) -> str:  # type: ignore[override]
        """Return qsub command as a single string for SSH execution."""
        parts = [
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
        pass_vars = ",".join(
            f"{k}={v}" for k, v in job_env.items() if k in self.pass_env_keys
        )
        if pass_vars:
            parts += ["-v", pass_vars]
        parts.append(self.script)
        return " ".join(parts)

    def submit_array(
        self,
        job_name: str,
        total_chunks: int,
        tasks_per_array: int,
        job_env: dict[str, str],
    ) -> None:
        """Submit array jobs in batches via SSH."""
        ssh_run(f"mkdir -p {self.log_dir}")

        start_task = 1
        while start_task <= total_chunks:
            end_task = min(start_task + tasks_per_array - 1, total_chunks)
            task_range = f"{start_task}-{end_task}"
            cmd_str = self._build_command(task_range, job_name, job_env)
            remote_cmd = f"cd {REMOTE_REPO} && {cmd_str}"
            result = ssh_run(remote_cmd)
            if result.returncode != 0:
                stderr_msg = result.stderr.strip() if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"Remote job submission failed (exit {result.returncode}) "
                    f"for array {task_range}:\n"
                    f"  command: {cmd_str}\n"
                    f"  stderr:  {stderr_msg}"
                )
            start_task = end_task + 1

    def submit_array_tracked(
        self,
        job_name: str,
        total_chunks: int,
        tasks_per_array: int,
        job_env: dict[str, str],
    ) -> list[tuple[str, str]]:
        """Like submit_array but returns (task_range, job_id) pairs."""
        ssh_run(f"mkdir -p {self.log_dir}")
        submissions: list[tuple[str, str]] = []

        start_task = 1
        while start_task <= total_chunks:
            end_task = min(start_task + tasks_per_array - 1, total_chunks)
            task_range = f"{start_task}-{end_task}"
            cmd_str = self._build_command(task_range, job_name, job_env)
            remote_cmd = f"cd {REMOTE_REPO} && {cmd_str}"
            result = ssh_run(remote_cmd)
            if result.returncode != 0:
                stderr_msg = result.stderr.strip() if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"Remote job submission failed (exit {result.returncode}) "
                    f"for array {task_range}:\n"
                    f"  command: {cmd_str}\n"
                    f"  stderr:  {stderr_msg}"
                )
            match = re.search(r"(\d+)", result.stdout)
            if not match:
                raise RuntimeError(
                    f"Could not parse job ID from qsub output: {result.stdout!r}"
                )
            submissions.append((task_range, match.group(1)))
            start_task = end_task + 1

        return submissions
