"""Submit one iteration of a `tune_<model>_<bucket>` campaign.

Driver for the tune-campaign closed-loop. Each invocation:

1. Parses ``campaign_id`` into (model, bucket).
2. Imports the local ``.hpc/tasks.py`` with ``HPC_CAMPAIGN_ID`` set so the
   campaign branch fires Optuna.ask + materializes
   ``params/<cid>/iter_<N>/{trial_*.json, manifest.json}``.
3. Computes the iteration's ``run_id``, builds the per-iteration sidecar,
   pushes ``tasks.py`` + sidecar + the new params dir to the cluster.
4. Submits an SGE/SLURM array job (100 tasks = K=1 trial × 100 chunks).

Caller invokes this once per iteration. The next iteration is submitted
**after** ``_score_iter.py`` has scored the previous one (so the next
``study.ask`` sees the new QLIKE).

Usage::

    python .hpc/campaigns/_submit_tune_iter.py <campaign_id> [--cluster h2|carc] [--dry-run]

Cluster default: H2 for xgb, CARC for lgbm — picked from prior empirical
fairness on each cluster (xgb ran fine on H2; lgbm starved there and
landed cleanly on CARC). Override with ``--cluster``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]

# ── Cluster profiles ──────────────────────────────────────────────────────
# Hardcoded here rather than pulled from clusters.yaml because we've
# already corrected several stale clusters.yaml fields by hand for this
# session (H2 conda 2023.03 not 2024.06, MODULES="" not python/3.11.9).
_CLUSTERS = {
    "h2": {
        "ssh_alias": "jamesdc1@hoffman2.idre.ucla.edu",
        "scratch": "/u/scratch/j/jamesdc1/harxhar",
        "scheduler": "sge",
        "template": ".hpc/templates/cpu_array.sh",
        "conda_source": "/u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh",
        "conda_env": "harxhar-dl",
        "modules": "",  # H2 has no python/3.11 module; conda env supplies python
        "ssh_opts": ["-o", "IdentitiesOnly=yes"],
    },
    "carc": {
        "ssh_alias": "usc-discovery",
        "scratch": "/scratch1/jc_905/harxhar",
        "scheduler": "slurm",
        "template": ".hpc/templates/cpu_array.slurm",
        "conda_source": "/apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh",
        "conda_env": "project-cucuringu",
        "modules": "",
        "account": "pollok_1603",
        "ssh_opts": ["-o", "IdentitiesOnly=yes"],
    },
}
_DEFAULT_CLUSTER_BY_MODEL = {"xgb": "carc", "lgbm": "carc", "ridge": "h2", "rf": "h2", "pcr": "h2"}

_MODEL_TO_MODULE = {
    "xgb": "src.ml_xgboost",
    "lgbm": "src.ml_lightgbm",
    "ridge": "src.ml_ridge",
    "rf": "src.ml_random_forest",
    "pcr": "src.ml_pcr",
}
_MODEL_PREFIXES = tuple(_MODEL_TO_MODULE)


def _parse_campaign(cid: str) -> tuple[str, str]:
    if not cid.startswith("tune_"):
        raise ValueError(f"campaign_id must start with 'tune_': {cid!r}")
    rest = cid[len("tune_") :]
    for m in _MODEL_PREFIXES:
        if rest.startswith(m + "_"):
            return m, rest[len(m) + 1 :]
    raise ValueError(f"unknown model prefix in {cid!r}")


def _load_tasks_with_campaign(campaign_id: str):
    """Import .hpc/tasks.py with HPC_CAMPAIGN_ID set, so the campaign
    branch fires Optuna ask + materializes params files at module load."""
    os.environ["HPC_CAMPAIGN_ID"] = campaign_id
    sys.path.insert(0, str(EXPERIMENT_DIR))
    spec = importlib.util.spec_from_file_location("hpc_tasks", EXPERIMENT_DIR / ".hpc/tasks.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load .hpc/tasks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_executor_cmd(model: str) -> str:
    """Shell command in the per-run sidecar. $RESULT_DIR is set by the
    cluster-side dispatcher to the per-task wip_dir; $HPC_KW_CHUNK_ID,
    $EXOG_COLS, $START, $END, $PARAMS_FILE come from tasks.resolve()."""
    return (
        f"python -m cli {_MODEL_TO_MODULE[model]} "
        f'--output-file "$RESULT_DIR/results_chunk_$HPC_KW_CHUNK_ID.csv" '
        f"--data-path all30min "
        f'--exog-cols "$EXOG_COLS" '
        f'--start "$START" --end "$END" '
        f'--params-file "$PARAMS_FILE"'
    )


def _result_dir_template(campaign_id: str, model: str, bucket: str) -> str:
    """Layout matches src.tune_tree.score_trials's bucket-scoped lookup:
    ``{results_dir}/{model}_{bucket}_{trial_id}/results_chunk_*.csv``.

    iter_idx + trial_idx are filled per-task by the dispatcher's format()
    against tasks.resolve()'s kwargs.
    """
    return f"results/tune/{campaign_id}/iter_{{iter_idx:03d}}/{model}_{bucket}_{{trial_idx}}"


def _ssh(cluster_cfg, *args: str) -> subprocess.CompletedProcess:
    cmd = ["C:/Windows/System32/OpenSSH/ssh.exe", *cluster_cfg["ssh_opts"], cluster_cfg["ssh_alias"], *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def _scp_to(cluster_cfg, src: Path, dst_rel: str) -> subprocess.CompletedProcess:
    """scp src -> {scratch}/{dst_rel} via the configured ssh alias."""
    target = f"{cluster_cfg['ssh_alias']}:{cluster_cfg['scratch']}/{dst_rel}"
    cmd = ["C:/Windows/System32/OpenSSH/scp.exe", *cluster_cfg["ssh_opts"], str(src), target]
    return subprocess.run(cmd, capture_output=True, text=True)


def _scp_dir_to(cluster_cfg, src: Path, dst_rel: str) -> subprocess.CompletedProcess:
    """Recursive scp."""
    target = f"{cluster_cfg['ssh_alias']}:{cluster_cfg['scratch']}/{dst_rel}"
    cmd = ["C:/Windows/System32/OpenSSH/scp.exe", "-r", *cluster_cfg["ssh_opts"], str(src), target]
    return subprocess.run(cmd, capture_output=True, text=True)


def _build_qsub_cmd(
    cluster_cfg,
    *,
    run_id: str,
    cmd_sha: str,
    campaign_id: str,
    iter_idx: int,
    job_name: str,
    total_tasks: int,
    walltime: str = "4:00:00",
    mem: str = "32G",
    cpus_per_task: int = 16,
) -> list[str]:
    pass_vars = ",".join(
        [
            "EXECUTOR=python3 .hpc/_hpc_dispatch.py",
            f"HPC_RUN_ID={run_id}",
            f"HPC_CMD_SHA={cmd_sha}",
            f"HPC_TASK_COUNT={total_tasks}",
            f"HPC_CAMPAIGN_ID={campaign_id}",
            f"HPC_ITER_IDX={iter_idx}",
            f"REPO_DIR={cluster_cfg['scratch']}",
            f"MODULES={cluster_cfg['modules']}",
            f"CONDA_SOURCE={cluster_cfg['conda_source']}",
            f"CONDA_ENV={cluster_cfg['conda_env']}",
        ]
    )
    template = f"{cluster_cfg['scratch']}/{cluster_cfg['template']}"
    if cluster_cfg["scheduler"] == "sge":
        return [
            "qsub",
            "-t",
            f"1-{total_tasks}",
            "-N",
            job_name,
            "-o",
            f"{cluster_cfg['scratch']}/logs",
            "-j",
            "y",
            "-l",
            f"h_rt={walltime},h_data={mem}",
            "-v",
            pass_vars,
            template,
        ]
    elif cluster_cfg["scheduler"] == "slurm":
        return [
            "sbatch",
            f"--array=1-{total_tasks}",
            f"--job-name={job_name}",
            f"--account={cluster_cfg['account']}",
            f"--time={walltime}",
            f"--mem={mem}",
            f"--cpus-per-task={cpus_per_task}",
            f"--output={cluster_cfg['scratch']}/logs/%x_%A_%a.out",
            f"--error={cluster_cfg['scratch']}/logs/%x_%A_%a.err",
            f"--export=ALL,{pass_vars}",
            template,
        ]
    raise ValueError(f"unknown scheduler: {cluster_cfg['scheduler']}")


def submit_iter(campaign_id: str, *, cluster: str | None = None, dry_run: bool = False) -> dict:
    from claude_hpc import compute_cmd_sha, compute_tasks_py_sha, write_run_sidecar

    model, bucket = _parse_campaign(campaign_id)
    cluster_key = cluster or _DEFAULT_CLUSTER_BY_MODEL[model]
    cfg = _CLUSTERS[cluster_key]

    # Step 1: import tasks.py with the campaign env set — fires Optuna ask
    # for this iteration and materializes params/<cid>/iter_<N>/.
    tasks = _load_tasks_with_campaign(campaign_id)
    n = tasks.total()
    if n == 0:
        raise RuntimeError(f"campaign {campaign_id!r} budget exhausted (tasks.total()==0)")

    cmd_sha = compute_cmd_sha(tasks)
    tasks_py_sha = compute_tasks_py_sha(EXPERIMENT_DIR / ".hpc/tasks.py")
    git_sha = subprocess.run(
        ["git", "-C", str(EXPERIMENT_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    iter_idx = tasks.resolve(0)["iter_idx"]
    run_id = f"{campaign_id}-iter{iter_idx:03d}-{ts}-{cmd_sha[:8]}"

    # Step 2: write sidecar
    executor_cmd = _build_executor_cmd(model)
    template = _result_dir_template(campaign_id, model, bucket)
    write_run_sidecar(
        EXPERIMENT_DIR,
        run_id=run_id,
        cmd_sha=cmd_sha,
        claude_hpc_version=__import__("claude_hpc").__version__,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        executor=executor_cmd,
        result_dir_template=template,
        task_count=n,
        tasks_py_sha=tasks_py_sha,
        cluster=cluster_key,
        profile=campaign_id,
        project="harxhar",
        campaign_id=campaign_id,
        remote_path=cfg["scratch"],
        resources={"cpus": 4, "mem": "16G", "walltime": "2:00:00"},
        env={"modules": cfg["modules"], "conda_source": cfg["conda_source"], "conda_env": cfg["conda_env"]},
        extra={"git_sha": git_sha, "model": model, "bucket": bucket, "iter_idx": iter_idx},
    )
    sidecar_local = EXPERIMENT_DIR / ".hpc/runs" / f"{run_id}.json"
    iter_dir_rel = f"params/{campaign_id}/iter_{iter_idx:03d}"
    iter_dir_local = EXPERIMENT_DIR / iter_dir_rel

    if dry_run:
        return {
            "dry_run": True,
            "run_id": run_id,
            "cluster": cluster_key,
            "tasks": n,
            "iter_idx": iter_idx,
            "qsub_cmd": _build_qsub_cmd(
                cfg,
                run_id=run_id,
                cmd_sha=cmd_sha,
                campaign_id=campaign_id,
                iter_idx=iter_idx,
                job_name=f"tu_{model}_{bucket}",
                total_tasks=n,
            ),
        }

    # Steps 3+4 wrapped in try/except: if SSH/scp/sbatch fails (transient
    # DNS hiccup, fail2ban, etc), roll back the just-asked Optuna trial and
    # delete the just-created iter_dir so a retry doesn't leave phantom
    # RUNNING trials in the study or holes in the iter sequence.
    def _rollback() -> None:
        import shutil

        try:
            from src.tune_tree import _load_or_create_study, _study_name
            import optuna.trial

            study = _load_or_create_study(
                model, storage_path=str(EXPERIMENT_DIR / ".hpc/optuna.db"),
                study_name=_study_name(model, bucket),
            )
            for t in study.trials:
                if t.state == optuna.trial.TrialState.RUNNING:
                    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        except Exception as e:
            print(f"  [rollback] could not mark trials FAIL: {e}", file=sys.stderr)
        try:
            shutil.rmtree(iter_dir_local)
        except Exception as e:
            print(f"  [rollback] could not delete {iter_dir_local}: {e}", file=sys.stderr)
        try:
            sidecar_local.unlink()
        except Exception:
            pass

    try:
        # Step 3: push tasks.py + sidecar + iter_dir to the cluster.
        for src, dst in [
            (EXPERIMENT_DIR / ".hpc/tasks.py", ".hpc/tasks.py"),
            (sidecar_local, f".hpc/runs/{run_id}.json"),
        ]:
            rc = _scp_to(cfg, src, dst)
            if rc.returncode != 0:
                raise RuntimeError(f"scp failed for {src}: {rc.stderr}")
        _ssh(cfg, f"mkdir -p {shlex.quote(cfg['scratch'])}/params/{shlex.quote(campaign_id)}")
        rc = _scp_dir_to(cfg, iter_dir_local, f"params/{campaign_id}/")
        if rc.returncode != 0:
            raise RuntimeError(f"scp params dir failed: {rc.stderr}")

        # Step 4: submit
        qsub_cmd = _build_qsub_cmd(
            cfg,
            run_id=run_id,
            cmd_sha=cmd_sha,
            campaign_id=campaign_id,
            iter_idx=iter_idx,
            job_name=f"tu_{model}_{bucket}",
            total_tasks=n,
        )
        remote_cmd = " ".join(shlex.quote(a) for a in qsub_cmd)
        submit_rc = _ssh(cfg, remote_cmd)
        if submit_rc.returncode != 0:
            raise RuntimeError(f"submit failed: {submit_rc.stderr}\n{submit_rc.stdout}")
    except Exception:
        _rollback()
        raise

    return {
        "run_id": run_id,
        "cluster": cluster_key,
        "tasks": n,
        "iter_idx": iter_idx,
        "submit_stdout": submit_rc.stdout.strip(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("campaign_id")
    p.add_argument("--cluster", choices=tuple(_CLUSTERS), default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    result = submit_iter(args.campaign_id, cluster=args.cluster, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
