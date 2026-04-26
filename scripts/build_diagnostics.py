"""Build the standardized per-method diagnostic plot bundle.

For each entry in `results/MANIFEST.json` with a `predictions_glob`, produces:

    results/diagnostics/<method>_<feature_set>_<config>/
        mz_scatter.png        — MZ regression in raw RV levels (mainstream axes,
                                MZ line + 45° reference always drawn)
        timeseries.png        — full-range Y vs Ŷ
        crash_2008.png        — Lehman / Oct-2008 window
        crash_2020.png        — COVID Feb-Apr 2020 window
        qlike_by_slot.png
        qlike_by_slot.csv
        mz_stats.json         — alpha, beta, SEs, R², N, t-stat for β=1

This is the only sanctioned diagnostic-plot entry point.

Usage:
    python scripts/build_diagnostics.py                            # all entries
    python scripts/build_diagnostics.py --entry ridge_har_paper    # one entry
    python scripts/build_diagnostics.py --skip-existing            # idempotent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation import (  # noqa: E402
    mz_regression,
    plot_crash_window,
    plot_mz_scatter,
    plot_qlike_by_slot,
    plot_y_yhat_timeseries,
    qlike_by_slot,
)

CRASH_WINDOWS = {
    "2008": ("2008-08-01", "2009-01-31"),
    "2020": ("2020-02-01", "2020-04-30"),
}


def entry_id(entry: dict) -> str:
    return f"{entry['method']}_{entry['feature_set']}_{entry['config']}"


def load_predictions(repo: Path, entry: dict) -> pd.DataFrame | None:
    """Concatenate all per-chunk prediction CSVs for one entry.

    Returns a DataFrame with columns: date, true_raw, pred_raw (sorted by date,
    non-positive / non-finite rows dropped). Returns None if predictions are
    not available.
    """
    glob = entry.get("predictions_glob")
    if not glob:
        return None
    rdir = repo / entry["results_dir"]
    paths = sorted(rdir.glob(glob))
    if not paths:
        return None

    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p, parse_dates=["date"], usecols=["date", "true_raw", "pred_raw"]))
        except (ValueError, KeyError) as e:
            print(f"  [warn] could not read {p}: {e}", file=sys.stderr)
            return None

    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    y, yhat = df["true_raw"].to_numpy(), df["pred_raw"].to_numpy()
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
    return df.loc[mask].reset_index(drop=True)


def render_mz_scatter(df: pd.DataFrame, out_dir: Path, label: str) -> dict:
    y = df["true_raw"].to_numpy()
    yhat = df["pred_raw"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    mz = plot_mz_scatter(y, yhat, ax, title=f"{label}\nMZ regression (raw RV)")
    fig.tight_layout()
    fig.savefig(out_dir / "mz_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mz


def render_timeseries(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    fig, (ax_raw, ax_log) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    plot_y_yhat_timeseries(
        df["date"],
        df["true_raw"].to_numpy(),
        df["pred_raw"].to_numpy(),
        ax_raw,
        ax_log,
        title=f"{label}: full-range Y vs Ŷ (N={len(df):,})",
    )
    fig.savefig(out_dir / "timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_crash(df: pd.DataFrame, out_dir: Path, label: str, tag: str, start: str, end: str) -> None:
    fig, (ax_raw, ax_log) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    plot_crash_window(df, start, end, ax_raw, ax_log, title=f"{label}: crash window {tag} ({start} – {end})")
    fig.savefig(out_dir / f"crash_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_qlike_slot(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    slot_df = qlike_by_slot(df)
    slot_df.to_csv(out_dir / "qlike_by_slot.csv", index=False)

    y, yhat = df["true_raw"].to_numpy(), df["pred_raw"].to_numpy()
    r = y / yhat
    global_qlike = float(np.mean(r - np.log(r) - 1.0))

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_qlike_by_slot(slot_df, ax, global_qlike=global_qlike, title=f"{label}: QLIKE by 30-min intraday slot")
    fig.tight_layout()
    fig.savefig(out_dir / "qlike_by_slot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_entry(repo: Path, entry: dict, out_root: Path, skip_existing: bool) -> bool:
    eid = entry_id(entry)
    out_dir = out_root / eid

    if skip_existing and (out_dir / "mz_stats.json").exists():
        print(f"[{eid}] skip (already built)")
        return True

    df = load_predictions(repo, entry)
    if df is None or df.empty:
        print(f"[{eid}] no predictions available — skipping")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{entry['method']} / {entry['feature_set']} / {entry['config']}"
    print(f"[{eid}] N={len(df):,}  range=[{df['date'].min()} .. {df['date'].max()}]")

    mz_in_plot = render_mz_scatter(df, out_dir, label)
    render_timeseries(df, out_dir, label)
    for tag, (start, end) in CRASH_WINDOWS.items():
        render_crash(df, out_dir, label, tag, start, end)
    render_qlike_slot(df, out_dir, label)

    mz_stats = mz_regression(df["true_raw"].to_numpy(), df["pred_raw"].to_numpy())
    assert mz_stats == mz_in_plot, "MZ regression drifted between scatter and stats!"
    (out_dir / "mz_stats.json").write_text(json.dumps(mz_stats, indent=2), encoding="utf-8")
    print(f"  -> {out_dir}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO / "results" / "MANIFEST.json"))
    parser.add_argument("--out-root", default=str(REPO / "results" / "diagnostics"))
    parser.add_argument("--entry", default=None, help="Process only this <method>_<feature_set>_<config>")
    parser.add_argument("--skip-existing", action="store_true", help="Skip entries with mz_stats.json already present.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(args.out_root)

    entries = manifest["entries"]
    if args.entry:
        entries = [e for e in entries if entry_id(e) == args.entry]
        if not entries:
            print(f"ERROR: no entry matched '{args.entry}'", file=sys.stderr)
            return 1

    n_ok = sum(process_entry(REPO, e, out_root, args.skip_existing) for e in entries)
    print(f"\nDone: {n_ok}/{len(entries)} entries processed -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
