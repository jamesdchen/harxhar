"""Merge the per-method monthly y-vs-yhat PDFs into a single PDF where each
page shows naive / ridge / xgb / lgbm stacked vertically for the same calendar
month (shared x-axis), so you can read the same time-of-month across models in
one glance.

Output: results/all3_monthly_y_vs_yhat.pdf (~207 pages).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parents[1]

# Order of stacking, top-to-bottom. Naive on top so the eye starts at the
# baseline and works "down" through the models.
ENTRIES = [
    ("Naive", "ml_baseline"),
    ("Ridge", "ml_ridge"),
    ("XGBoost (tuned)", "ml_xgboost_tuned"),
    ("LightGBM (tuned)", "tune/lgbm_replay_carc_apr30"),
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


def main() -> int:
    dfs = [(label, load_preds(rd).assign(month=lambda d: d["date"].dt.to_period("M"))) for label, rd in ENTRIES]

    all_months = sorted(set().union(*(set(df["month"].unique()) for _, df in dfs)))
    out = REPO / "results" / "all3_monthly_y_vs_yhat.pdf"
    print(f"writing {len(all_months)} pages to {out} ...")

    with PdfPages(str(out)) as pdf:
        for m in all_months:
            fig, axes = plt.subplots(
                len(ENTRIES), 1, figsize=(12, 3.3 * len(ENTRIES)), sharex=True, constrained_layout=True
            )
            for ax, (label, df) in zip(axes, dfs, strict=True):
                sub = df.loc[df["month"] == m]
                if len(sub) == 0:
                    ax.set_title(f"{label} — {m} (no data)")
                    continue
                ax.plot(sub["date"], sub["true_raw"], color="#1f77b4", lw=0.7, label="actual (Y)")
                ax.plot(sub["date"], sub["pred_raw"], color="#d62728", lw=0.7, alpha=0.85, label="predicted (Ŷ)")
                ax.set_ylabel("RV (raw)")
                ax.set_title(f"{label} — {m}  (N={len(sub):,})")
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel("date (30-min bars)")
            fig.autofmt_xdate()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
