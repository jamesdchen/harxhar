"""Meeting deliverables for the v4 master_table:
  1. results/mz_scatter_ridge_naive_xgb.png  (3-panel MZ comparison)
  2. results/<method>_monthly_y_vs_yhat.pdf  (one per method, monthly mean line)
  3. (master_table.tex already refreshed by scripts/build_master_table.py)

Pure orchestration of src.evaluation.plot_mz_scatter; no plot logic re-implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation import plot_mz_scatter  # noqa: E402

ENTRIES = [
    ("ridge", "ml_ridge", "Ridge"),
    ("naive", "ml_baseline", "Naive"),
    ("xgb", "ml_xgboost_tuned", "XGBoost (tuned)"),
    ("lgbm", "tune/lgbm_replay_carc_apr30", "LightGBM (tuned)"),
]


def load_preds(results_dir: str) -> pd.DataFrame:
    paths = sorted(
        (REPO / "results" / results_dir).glob("results_chunk_*.csv"),
        key=lambda p: int(p.stem.removeprefix("results_chunk_")),
    )
    df = (
        pd.concat(
            [pd.read_csv(p, parse_dates=["date"], usecols=["date", "true_raw", "pred_raw"]) for p in paths],
            ignore_index=True,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    y, yhat = df["true_raw"].to_numpy(), df["pred_raw"].to_numpy()
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
    return df.loc[mask].reset_index(drop=True)


def make_mz_panel(dfs: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, (key, _, label) in zip(axes.flat, ENTRIES, strict=True):
        df = dfs[key]
        plot_mz_scatter(
            df["true_raw"].to_numpy(),
            df["pred_raw"].to_numpy(),
            ax,
            title=f"{label}\nMZ regression (raw RV)",
        )
    fig.tight_layout()
    out = REPO / "results" / "mz_scatter_ridge_naive_xgb_lgbm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def make_monthly_pdf(df: pd.DataFrame, label: str, out_path: Path) -> None:
    """Multi-page PDF — one page per calendar month, showing the 30-min granular
    Y and Ŷ series within that month."""
    from matplotlib.backends.backend_pdf import PdfPages

    df = df.assign(month=df["date"].dt.to_period("M"))
    months = sorted(df["month"].unique())
    with PdfPages(str(out_path)) as pdf:
        for m in months:
            sub = df.loc[df["month"] == m]
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(sub["date"], sub["true_raw"], color="#1f77b4", lw=0.8, label="actual (Y)")
            ax.plot(sub["date"], sub["pred_raw"], color="#d62728", lw=0.8, alpha=0.85, label="predicted (Ŷ)")
            ax.set_xlabel("date (30-min bars)")
            ax.set_ylabel("RV (raw scale)")
            ax.set_title(f"{label} — {m}  (N={len(sub):,} bars)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {out_path}  ({len(months)} pages)")


def main() -> int:
    dfs = {key: load_preds(rd) for key, rd, _ in ENTRIES}
    for key, _, _ in ENTRIES:
        print(f"  {key}: N={len(dfs[key]):,}")
    make_mz_panel(dfs)
    for key, _, label in ENTRIES:
        out = REPO / "results" / f"{key}_monthly_y_vs_yhat.pdf"
        make_monthly_pdf(dfs[key], label, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
