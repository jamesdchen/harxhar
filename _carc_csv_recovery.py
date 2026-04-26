#!/usr/bin/env python3
"""Move stray ./results_chunk_<TID>.csv files into their per-trial result dirs.

Lookup uses the CARC _hpc_dispatch.json manifest cmd strings (canonical), so it
works regardless of whether the manifest stores cmd strings or full task dicts.

Safety: never overwrites. Skips collisions and reports them.
Scope: only touches files matching ./results_chunk_*.csv in the repo root and
creates dirs under ./results/tune_exog/.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path("/scratch1/jc_905/harxhar")
MANIFEST = REPO / "_hpc_dispatch.json"
TARGET_ROOT = REPO / "results" / "tune_exog"

CMD_RE = {
    "chunk": re.compile(r"--chunk-id (\d+)"),
    "model": re.compile(r"--model (\S+)"),
    "trial": re.compile(r"--trial-id (\d+)"),
    "bucket": re.compile(r"--exog-bucket (\S+)"),
}


def parse_cmd(cmd: str) -> tuple[int, str] | None:
    try:
        chunk = int(CMD_RE["chunk"].search(cmd).group(1))
        model = CMD_RE["model"].search(cmd).group(1)
        trial = int(CMD_RE["trial"].search(cmd).group(1))
        bucket = CMD_RE["bucket"].search(cmd).group(1)
    except (AttributeError, ValueError):
        return None
    return chunk, f"results/tune_exog/{model}_{bucket}_{trial}"


def build_map(manifest_path: Path) -> dict[str, tuple[int, str]]:
    raw = json.loads(manifest_path.read_text())
    tasks = raw["tasks"]
    out: dict[str, tuple[int, str]] = {}
    for tid, val in tasks.items():
        if isinstance(val, dict):
            cmd = val.get("cmd", "")
        else:
            cmd = val
        parsed = parse_cmd(cmd)
        if parsed is not None:
            out[str(tid)] = parsed
    return out


def main() -> int:
    if not MANIFEST.exists():
        print(f"ERROR: manifest missing at {MANIFEST}", file=sys.stderr)
        return 2
    tid_map = build_map(MANIFEST)
    print(f"manifest entries parsed: {len(tid_map)}")

    stray = sorted(REPO.glob("results_chunk_*.csv"))
    print(f"stray csvs found in repo root: {len(stray)}")

    moved = 0
    skipped_collision: list[tuple[str, str]] = []
    skipped_unmapped: list[str] = []
    sample_moves: list[tuple[str, str]] = []

    for src in stray:
        m = re.match(r"results_chunk_(\d+)\.csv$", src.name)
        if not m:
            skipped_unmapped.append(src.name)
            continue
        tid = m.group(1)
        info = tid_map.get(tid)
        if info is None:
            skipped_unmapped.append(src.name)
            continue
        chunk_id, rel_dir = info
        dst_dir = REPO / rel_dir
        dst = dst_dir / f"results_chunk_{chunk_id}.csv"
        if dst.exists():
            skipped_collision.append((src.name, str(dst.relative_to(REPO))))
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
        moved += 1
        if len(sample_moves) < 5:
            sample_moves.append((src.name, str(dst.relative_to(REPO))))

    print("\n=== SUMMARY ===")
    print(f"moved:              {moved}")
    print(f"skipped (collision):{len(skipped_collision)}")
    print(f"skipped (unmapped): {len(skipped_unmapped)}")
    print("\nsample moves:")
    for s, d in sample_moves:
        print(f"  {s}  ->  {d}")
    if skipped_collision:
        print("\nfirst 5 collisions (left in place):")
        for s, d in skipped_collision[:5]:
            print(f"  {s}  X  {d}")
    if skipped_unmapped:
        print("\nfirst 5 unmapped (left in place):")
        for s in skipped_unmapped[:5]:
            print(f"  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
