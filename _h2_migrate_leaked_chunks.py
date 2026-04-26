#!/usr/bin/env python3
"""Cascade-migrate cross-trial-leaked chunk files in results/tune_exog/xgb_*/.

Background:
  _cpu_array.sh used to export a single global RESULT_DIR for every task in
  the SGE array. Tasks for trial N (TIDs N*100..N*100+99) all wrote into the
  trial-1 dir, naming files results_chunk_<TID>.csv (with the global TID).
  Mapping rule: a file results_chunk_<TID>.* sitting in any xgb_<bucket>_*/
  dir actually belongs to xgb_<bucket>_<TID//100>/results_chunk_<TID%100>.*

Behavior:
  - Walks results/tune_exog/xgb_*/ dirs.
  - Plans moves for results_chunk_<TID>.csv, results_chunk_<TID>_reduce.json,
    and _wip_<TID> markers.
  - Skips when src == dest, when dest already exists (no overwrites), or
    when the planned move is a no-op.
  - Processes moves in DESCENDING order of trial-bucket so trial-2 files
    land in (currently empty) trial-2 slots BEFORE trial-1 files vacate
    those slots — avoiding self-clobber.
  - Logs all planned moves, then executes (no human confirmation prompt).
  - Reports per-trial-dir counts before/after, plus totals.

Compatible with Python 3.6+ (H2 login node).
"""

import os
import re
import sys
from collections import defaultdict

ROOT = "results/tune_exog"

DIR_RE = re.compile(r"^xgb_(?P<bucket>[a-zA-Z0-9]+)_(?P<trial>\d+)$")

FILE_PATTERNS = [
    re.compile(r"^results_chunk_(?P<tid>\d+)\.csv$"),
    re.compile(r"^results_chunk_(?P<tid>\d+)_reduce\.json$"),
    re.compile(r"^_wip_(?P<tid>\d+)$"),
]


def classify(name):
    for p in FILE_PATTERNS:
        m = p.match(name)
        if m:
            return int(m.group("tid"))
    return None


def rename_for_chunk(name, new_chunk):
    if name.endswith("_reduce.json"):
        return "results_chunk_{}_reduce.json".format(new_chunk)
    if name.endswith(".csv"):
        return "results_chunk_{}.csv".format(new_chunk)
    if name.startswith("_wip_"):
        return "_wip_{}".format(new_chunk)
    raise ValueError("unrecognised name: " + name)


def per_dir_counts(buckets):
    counts = {}
    for bucket, trials in buckets.items():
        for trial in trials:
            d = os.path.join(ROOT, "xgb_{}_{}".format(bucket, trial))
            if os.path.isdir(d):
                counts[os.path.basename(d)] = len(os.listdir(d))
            else:
                counts[os.path.basename(d)] = 0
    return counts


def discover_buckets():
    buckets = defaultdict(list)
    for name in os.listdir(ROOT):
        m = DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(ROOT, name)):
            buckets[m.group("bucket")].append(int(m.group("trial")))
    for b in buckets:
        buckets[b].sort()
    return buckets


def main():
    if not os.path.isdir(ROOT):
        sys.stderr.write("ERROR: {} does not exist\n".format(ROOT))
        return 2

    buckets = discover_buckets()

    print("=== Discovered trial dirs ===")
    for b, trials in sorted(buckets.items()):
        print("  {}: trials {}".format(b, trials))
    print()

    # Plan moves: (src_path, dest_path, src_trial, target_trial)
    plans = []
    unmapped = []

    for bucket, trials in buckets.items():
        for src_trial in trials:
            src_dir = os.path.join(ROOT, "xgb_{}_{}".format(bucket, src_trial))
            for entry in os.listdir(src_dir):
                src_path = os.path.join(src_dir, entry)
                tid = classify(entry)
                if tid is None:
                    unmapped.append(src_path)
                    continue
                target_trial = tid // 100
                new_chunk = tid % 100
                dest_dir = os.path.join(ROOT, "xgb_{}_{}".format(bucket, target_trial))
                dest = os.path.join(dest_dir, rename_for_chunk(entry, new_chunk))
                if dest == src_path:
                    continue  # already correct
                plans.append((src_path, dest, src_trial, target_trial))

    # Process highest target_trial first so trial-N files land in trial-N
    # slots before trial-(N-1) files vacate those slots.
    plans.sort(key=lambda t: (-t[3], -t[2], t[0]))

    print("=== Planned moves: {} ===".format(len(plans)))
    for src, dest, st, tt in plans[:20]:
        print("  [{}->{}] {} -> {}".format(st, tt, src, dest))
    if len(plans) > 20:
        print("  ... ({} more)".format(len(plans) - 20))
    print()

    print("=== Unmapped entries: {} ===".format(len(unmapped)))
    for u in unmapped[:10]:
        print("  {}".format(u))
    if len(unmapped) > 10:
        print("  ... ({} more)".format(len(unmapped) - 10))
    print()

    before = per_dir_counts(buckets)
    print("=== Per-dir counts (before) ===")
    for name in sorted(before):
        print("  {}: {}".format(name, before[name]))
    print()

    moved = 0
    skipped_collision = []
    skipped_other = []
    for src, dest, st, tt in plans:
        if not os.path.exists(src):
            skipped_other.append((src, dest, "src disappeared"))
            continue
        if os.path.exists(dest):
            skipped_collision.append((src, dest))
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            os.rename(src, dest)
            moved += 1
        except OSError as e:
            skipped_other.append((src, dest, "OSError: {}".format(e)))

    print("=== Execution summary ===")
    print("  total moved:       {}".format(moved))
    print("  skipped collision: {}".format(len(skipped_collision)))
    print("  skipped other:     {}".format(len(skipped_other)))
    print("  unmapped:          {}".format(len(unmapped)))
    print()
    if skipped_collision:
        print("--- Collisions (first 20) ---")
        for s, d in skipped_collision[:20]:
            print("  {} -> {} (dest exists)".format(s, d))
        if len(skipped_collision) > 20:
            print("  ... ({} more)".format(len(skipped_collision) - 20))
        print()
    if skipped_other:
        print("--- Other skips (first 20) ---")
        for s, d, why in skipped_other[:20]:
            print("  {} -> {}: {}".format(s, d, why))
        if len(skipped_other) > 20:
            print("  ... ({} more)".format(len(skipped_other) - 20))
        print()

    buckets_after = discover_buckets()
    after = per_dir_counts(buckets_after)
    print("=== Per-dir counts (after) ===")
    all_names = sorted(set(before) | set(after))
    for name in all_names:
        b = before.get(name, 0)
        a = after.get(name, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        print("  {}: {} -> {} ({}{})".format(name, b, a, sign, delta))

    return 0


if __name__ == "__main__":
    sys.exit(main())
