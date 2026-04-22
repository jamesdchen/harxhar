#!/usr/bin/env python3
"""
Pre-sync safety check: Prevent manifest overwrites during active HPC jobs.

Failure mode (2026-04-19):
  - Manifest _hpc_dispatch.json was overwritten while job 8167467 was running
  - Old tasks used old config, new tasks used new config
  - Chunk IDs became misaligned (overlapping ranges: trial 1 chunks 100-199, trial 2 chunks 156-255)
  - Results were corrupted and unreliable

Solution: Lock manifest during active jobs. Check before any rsync/deploy.
"""

import subprocess
import sys
from pathlib import Path


def get_active_jobs(hpc_config_path: str = "hpc.yaml") -> dict:
    """Get active HPC jobs for this project via qstat/squeue."""
    result = {}

    # Try Hoffman2 (SGE)
    try:
        out = subprocess.check_output(
            ["ssh", "jamesdc1@hoffman2.idre.ucla.edu", "qstat | grep tune"], stderr=subprocess.DEVNULL, text=True
        )
        result["hoffman2"] = [line.strip() for line in out.split("\n") if line.strip()]
    except Exception:
        pass

    # Try CARC (SLURM)
    try:
        out = subprocess.check_output(
            [
                "ssh",
                "-i",
                str(Path.home() / ".ssh" / "id_carc"),
                "jc_905@discovery2.usc.edu",
                "squeue -u jc_905 | grep tune",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        result["carc"] = [line.strip() for line in out.split("\n") if line.strip()]
    except Exception:
        pass

    return result


def check_manifest_safety() -> bool:
    """
    Check if it's safe to sync. Return False if active job would be corrupted.

    RULE: If a manifest exists on any active cluster, refuse rsync without confirmation.
    """
    active_jobs = get_active_jobs()

    if not active_jobs or all(not v for v in active_jobs.values()):
        print("✓ No active HPC jobs found. Safe to sync.")
        return True

    print("⚠ Active HPC jobs detected:")
    for cluster, jobs in active_jobs.items():
        if jobs:
            print(f"  {cluster}: {len(jobs)} job(s)")
            for job in jobs[:3]:
                print(f"    {job}")

    print("\n⛔ MANIFEST CORRUPTION RISK")
    print("Syncing will overwrite _hpc_dispatch.json on active clusters.")
    print("This causes chunk ID misalignment → corrupted results.")
    print("\nOptions:")
    print("  1. Wait for jobs to complete: qstat -j <job_id> (until no running tasks)")
    print("  2. Force sync (not recommended): python pre_sync_check.py --force")

    return False


if __name__ == "__main__":
    if "--force" in sys.argv:
        print("⚠ Forced sync bypassing safety check. Results may be corrupted.")
        sys.exit(0)

    if not check_manifest_safety():
        sys.exit(1)

    print("\n✓ Safe to proceed with sync/deploy.")
    sys.exit(0)
