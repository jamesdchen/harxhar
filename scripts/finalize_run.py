"""Finalize a backtest run: write metrics.json and optionally upsert MANIFEST."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation import calculate_metrics, mz_regression  # noqa: E402


def _upsert_manifest_entry(manifest_path: Path, entry: dict) -> None:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": "1.0", "entries": [], "notes": []}
    for k, default in (("version", "1.0"), ("entries", []), ("notes", [])):
        manifest.setdefault(k, default)
    entries = manifest["entries"]
    for i, e in enumerate(entries):
        if e.get("results_dir") == entry["results_dir"]:
            entries[i] = entry
            break
    else:
        entries.append(entry)

    fd, tmp = tempfile.mkstemp(prefix=manifest_path.name, dir=str(manifest_path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)


def main() -> int:
    """Compute metrics.json from results.csv and (optionally) upsert MANIFEST."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--update-manifest", type=Path, default=None)
    p.add_argument("--feature-set", default="default")
    p.add_argument("--config", default="default")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--segment", default="global")
    p.add_argument("--summary-csv", default="metrics.json")
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Run dir of the naive baseline (with results.csv). When set, "
        "writes oos_r2 = 1 - SS_model / SS_baseline (raw scale) into metrics.json.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.run_dir / "results.csv")
    metrics = calculate_metrics(df)

    # Mincer-Zarnowitz on raw scale
    mz = mz_regression(df["true_raw"].to_numpy(), df["pred_raw"].to_numpy())
    metrics["mz_alpha"] = mz["alpha"]
    metrics["mz_alpha_se"] = mz["alpha_se"]
    metrics["mz_beta"] = mz["beta"]
    metrics["mz_beta_se"] = mz["beta_se"]
    metrics["mz_t_alpha_eq_0"] = mz["t_alpha_eq_0"]
    metrics["mz_t_beta_eq_1"] = mz["t_beta_eq_1"]
    metrics["mz_r2"] = mz["r2"]

    # OOS R^2 vs naive baseline (QLIKE-based: 1 - QLIKE_model / QLIKE_naive)
    if args.baseline_dir is not None:
        base_df = pd.read_csv(args.baseline_dir / "results.csv")
        if (df["date"].values == base_df["date"].values).all():
            qlike_base = calculate_metrics(base_df)["qlike"]
            metrics["oos_r2"] = 1.0 - metrics["qlike"] / qlike_base if qlike_base > 0 else float("nan")
        else:
            print(f"[warn] date mismatch with baseline {args.baseline_dir} — skipping oos_r2", file=sys.stderr)

    metrics_path = args.run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    upserted = ""
    if args.update_manifest is not None:
        entry = {
            "method": args.method,
            "feature_set": args.feature_set,
            "config": args.config,
            "results_dir": str(args.run_dir),
            "summary_csv": args.summary_csv,
            "segment": args.segment,
            "horizon": args.horizon,
        }
        _upsert_manifest_entry(args.update_manifest, entry)
        upserted = " and upserted manifest entry"

    print(f"finalize_run: wrote {metrics_path} (qlike={metrics.get('qlike')}, n={metrics.get('n_samples')}){upserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
