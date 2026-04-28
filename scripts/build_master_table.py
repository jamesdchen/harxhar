# Auto-generated from notebooks/build_master_table.ipynb. Do not edit by hand.

"""Build the master results table from results/MANIFEST.json.

Outputs:
    results/master_table.csv          — full, queryable, one row per manifest entry
    writeup/sections/master_table.tex — compact LaTeX for the writeup body
    writeup/sections/master_table_full.tex — full LaTeX for the appendix

Cross-meeting comparison: `git diff results/master_table.csv` between meetings.

Usage:
    python scripts/build_master_table.py
    python scripts/build_master_table.py --manifest results/MANIFEST.json
    python scripts/build_master_table.py --no-tex   # skip LaTeX outputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation import calculate_metrics, mz_regression  # noqa: E402

# Canonical column order for the master CSV. Tree-story columns are populated
# from results/diagnostics/<entry>/tree_story_stats.json when that file exists
# (NaN / "—" otherwise). Provides cross-meeting comparability for tree internals.
MASTER_COLUMNS = [
    "method",
    "feature_set",
    "config",
    "segment",
    "horizon",
    "n",
    "qlike",
    "w_qlike",
    "mse",
    "mae",
    "oos_r2",
    "delta_qlike",
    "mz_alpha",
    "mz_alpha_se",
    "mz_beta",
    "mz_beta_se",
    "mz_t_beta_eq_1",
    "mz_t_alpha_eq_0",
    "mz_r2",
    "mean_tree_depth",
    "top_shap_feature",
    "top_shap_share",
    "rank_stability_rho",
    "top_interaction_pair",
    "results_dir",
]

# Compact-LaTeX columns: keeps the body-table narrow.
COMPACT_TEX_COLS = [
    "method",
    "feature_set",
    "config",
    "n",
    "qlike",
    "w_qlike",
    "mse",
    "oos_r2",
    "mz_alpha",
    "mz_beta",
    "mz_t_beta_eq_1",
    "mz_r2",
]
# Full-LaTeX columns: appendix table picks up all tree-story extras.
FULL_TEX_COLS = COMPACT_TEX_COLS + ["top_shap_share", "rank_stability_rho", "top_interaction_pair"]


_NAN_METRICS = {
    "n": 0,
    "qlike": float("nan"),
    "w_qlike": float("nan"),
    "mse": float("nan"),
    "mae": float("nan"),
    "oos_r2": float("nan"),
    "delta_qlike": float("nan"),
    "segment": "GLOBAL",
    "horizon": 1,
}


def _load_summary_row(repo: Path, entry: dict) -> dict | None:
    """Load metrics from a summary file. Three flavours supported:
    - CSV with optional row filter (e.g., experiment_name=...)
    - JSON with a ``qlike`` key (per `tune_tree.reduce` convention)
    - JSON without metrics (e.g., best_xgb.json hyperparams) → metrics
      deferred to predictions-glob computation; returns NaN scaffold.
    """
    rdir = repo / entry["results_dir"]
    if "qlike_json" in entry:
        path = rdir / entry["qlike_json"]
        if not path.exists():
            print(f"  [skip] missing qlike: {path}", file=sys.stderr)
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        return {**_NAN_METRICS, "n": int(d.get("n", 0)), "qlike": float(d.get("qlike", float("nan")))}

    summary_name = entry.get("summary_csv")
    if summary_name is None:
        print(f"  [skip] entry has neither summary_csv nor qlike_json: {entry}", file=sys.stderr)
        return None
    path = rdir / summary_name
    if not path.exists():
        print(f"  [skip] missing summary: {path}", file=sys.stderr)
        return None

    if path.suffix.lower() == ".json":
        d = json.loads(path.read_text(encoding="utf-8"))
        if "qlike" in d:
            row = dict(_NAN_METRICS)
            row["n"] = int(d.get("n_samples", d.get("n", 0)))
            row["qlike"] = float(d["qlike"])
            for k in (
                "w_qlike",
                "mse",
                "mae",
                "oos_r2",
                "delta_qlike",
                "mz_alpha",
                "mz_alpha_se",
                "mz_beta",
                "mz_beta_se",
                "mz_t_beta_eq_1",
                "mz_t_alpha_eq_0",
                "mz_r2",
            ):
                if k in d:
                    row[k] = float(d[k])
            if "segment" in d:
                row["segment"] = str(d["segment"])
            if "horizon" in d:
                row["horizon"] = int(d["horizon"])
            return row
        # JSON without metrics (e.g., a best-params file): defer to predictions.
        return dict(_NAN_METRICS)

    # CSV summary (the common case).
    df = pd.read_csv(path)
    flt = entry.get("filter") or {}
    if flt:
        mask = pd.Series(True, index=df.index)
        for k, v in flt.items():
            mask &= df[k] == v
        df = df[mask]
    if df.empty:
        print(f"  [skip] filter {flt} matched 0 rows in {path}", file=sys.stderr)
        return None
    if len(df) > 1:
        print(f"  [warn] filter matched {len(df)} rows in {path}; using first", file=sys.stderr)
    row = df.iloc[0]
    return {
        "n": int(row.get("n_samples", 0)),
        "qlike": float(row.get("qlike", float("nan"))),
        "w_qlike": float(row.get("w_qlike", float("nan"))),
        "mse": float(row.get("mse", float("nan"))),
        "mae": float(row.get("mae", float("nan"))),
        "oos_r2": float(row.get("oos_r2", float("nan"))),
        "delta_qlike": float(row.get("delta_qlike", float("nan"))),
        "segment": str(row.get("segment", "GLOBAL")),
        "horizon": int(row.get("horizon", 1)),
    }


def _metrics_from_predictions(y: np.ndarray, yhat: np.ndarray) -> dict:
    """Compute QLIKE / MSE / MAE on raw scale from predictions when summary
    file lacks metrics (tuned-trial entries with hyperparams-only JSON)."""
    df = pd.DataFrame(
        {
            "true_adj": y,
            "pred_adj": yhat,
            "true_raw": y,
            "pred_raw": yhat,
        }
    )
    m = calculate_metrics(df)
    return {
        "n": int(m["n_samples"]),
        "qlike": float(m["qlike"]),
        "w_qlike": float(m["w_qlike"]),
        "mse": float(m["mse"]),
        "mae": float(m["mae"]),
    }


def _load_predictions(repo: Path, entry: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate per-chunk prediction CSVs if `predictions_glob` is present.

    Returns (y_raw, yhat_raw) with non-positive / non-finite rows dropped, or
    None if predictions are unavailable.
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
            frames.append(pd.read_csv(p, usecols=["true_raw", "pred_raw"]))
        except (ValueError, KeyError):
            return None
    df = pd.concat(frames, ignore_index=True)
    y = df["true_raw"].to_numpy()
    yhat = df["pred_raw"].to_numpy()
    m = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
    return y[m], yhat[m]


_NAN_TREE_STORY = {
    "mean_tree_depth": float("nan"),
    "top_shap_feature": "—",
    "top_shap_share": float("nan"),
    "rank_stability_rho": float("nan"),
    "top_interaction_pair": "—",
}


def _entry_id(entry: dict) -> str:
    return f"{entry['method']}_{entry['feature_set']}_{entry['config']}"


def _load_tree_story_row(repo: Path, entry: dict) -> dict:
    """Pull tree-story headline numbers from results/diagnostics/<entry>/tree_story_stats.json.

    Returns NaN/"—" defaults when the file does not exist (i.e., non-tree models
    or tree models we haven't built diagnostics for yet).
    """
    path = repo / "results" / "diagnostics" / _entry_id(entry) / "tree_story_stats.json"
    if not path.exists():
        return dict(_NAN_TREE_STORY)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [warn] could not read {path}: {e}", file=sys.stderr)
        return dict(_NAN_TREE_STORY)
    return {
        "mean_tree_depth": float(d.get("mean_tree_depth", float("nan"))),
        "top_shap_feature": str(d.get("top_shap_feature", "—")),
        "top_shap_share": float(d.get("top_shap_share", float("nan"))),
        "rank_stability_rho": float(d.get("rank_stability_rho", float("nan"))),
        "top_interaction_pair": str(d.get("top_interaction_pair", "—")),
    }


def build_rows(repo: Path, manifest: dict) -> pd.DataFrame:
    rows = []
    for entry in manifest["entries"]:
        method = entry["method"]
        feature_set = entry["feature_set"]
        config = entry["config"]
        print(f"[{method}/{feature_set}/{config}] loading...")

        metrics = _load_summary_row(repo, entry)
        if metrics is None:
            continue

        # MZ stats: prefer summary-file values (written by finalize_run), fall
        # back to predictions-glob computation if summary didn't carry them.
        mz_keys = ("mz_alpha", "mz_alpha_se", "mz_beta", "mz_beta_se", "mz_t_beta_eq_1", "mz_t_alpha_eq_0", "mz_r2")
        mz_stats = {k: float(metrics.pop(k, float("nan"))) for k in mz_keys}
        need_mz = not np.isfinite(mz_stats["mz_alpha"])
        need_metrics = not np.isfinite(metrics.get("qlike", float("nan")))
        if need_mz or need_metrics:
            preds = _load_predictions(repo, entry)
            if preds is not None:
                y, yhat = preds
                if len(y) >= 30:
                    if need_mz:
                        mz = mz_regression(y, yhat)
                        mz_stats = {
                            "mz_alpha": mz["alpha"],
                            "mz_alpha_se": mz["alpha_se"],
                            "mz_beta": mz["beta"],
                            "mz_beta_se": mz["beta_se"],
                            "mz_t_beta_eq_1": mz["t_beta_eq_1"],
                            "mz_t_alpha_eq_0": mz["t_alpha_eq_0"],
                            "mz_r2": mz["r2"],
                        }
                    if need_metrics:
                        metrics.update(_metrics_from_predictions(y, yhat))

        tree_story = _load_tree_story_row(repo, entry)

        rows.append(
            {
                "method": method,
                "feature_set": feature_set,
                "config": config,
                **metrics,
                **mz_stats,
                **tree_story,
                "results_dir": entry["results_dir"],
            }
        )

    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    return df


def write_compact_tex(df: pd.DataFrame, out_path: Path) -> None:
    """Headline rows: naive baseline + each method's best feature_set by QLIKE.

    Per-method "best" picks the row with min QLIKE (after dropna). Always include
    the naive baseline if present.
    """
    rows = []
    if (df["method"] == "naive").any():
        rows.append(df[df["method"] == "naive"].iloc[0])
    for method in [m for m in df["method"].unique() if m != "naive"]:
        sub = df[(df["method"] == method) & df["qlike"].notna()]
        if sub.empty:
            continue
        rows.append(sub.loc[sub["qlike"].idxmin()])
    if not rows:
        out_path.write_text("% no rows\n", encoding="utf-8")
        return
    tab = pd.DataFrame(rows)
    _write_tex(
        tab,
        out_path,
        caption="Headline: best feature set per method (by QLIKE).",
        label="tab:master_compact",
        cols=COMPACT_TEX_COLS,
    )


def write_full_tex(df: pd.DataFrame, out_path: Path) -> None:
    _write_tex(
        df,
        out_path,
        caption="Full master table across (method × feature set × config).",
        label="tab:master_full",
        cols=FULL_TEX_COLS,
    )


def _write_tex(df: pd.DataFrame, out_path: Path, *, caption: str, label: str, cols: list[str]) -> None:
    keep = [c for c in cols if c in df.columns]
    sub = df[keep].copy()

    fmts = {
        "n": "{:,.0f}",
        "qlike": "{:.4f}",
        "w_qlike": "{:.4f}",
        "mse": "{:.4f}",
        "mae": "{:.4f}",
        "oos_r2": "{:+.4f}",
        "mz_alpha": "{:+.3g}",
        "mz_alpha_se": "{:.3g}",
        "mz_beta": "{:.4f}",
        "mz_beta_se": "{:.4f}",
        "mz_t_alpha_eq_0": "{:+.2f}",
        "mz_t_beta_eq_1": "{:+.2f}",
        "mz_r2": "{:.4f}",
        "mean_tree_depth": "{:.2f}",
        "top_shap_share": "{:.3f}",
        "rank_stability_rho": "{:.3f}",
    }
    for col, fmt in fmts.items():
        if col in sub.columns:
            sub[col] = sub[col].apply(lambda v, f=fmt: f.format(v) if pd.notna(v) else "—")
    for col in ("method", "feature_set", "config", "top_shap_feature", "top_interaction_pair"):
        if col in sub.columns:
            sub[col] = sub[col].astype(str).str.replace("_", r"\_", regex=False)

    align = "l" * len(keep)
    headers = [c.replace("_", r"\_") for c in keep]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        lines.append(" & ".join(str(row[c]) for c in keep) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO / "results" / "MANIFEST.json"))
    parser.add_argument("--out-csv", default=str(REPO / "results" / "master_table.csv"))
    parser.add_argument("--out-tex-compact", default=str(REPO / "writeup" / "sections" / "master_table.tex"))
    parser.add_argument("--out-tex-full", default=str(REPO / "writeup" / "sections" / "master_table_full.tex"))
    parser.add_argument("--no-tex", action="store_true", help="Skip LaTeX outputs.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    df = build_rows(REPO, manifest)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(df)} rows)")

    if not args.no_tex:
        out_compact = Path(args.out_tex_compact)
        out_full = Path(args.out_tex_full)
        out_compact.parent.mkdir(parents=True, exist_ok=True)
        write_compact_tex(df, out_compact)
        write_full_tex(df, out_full)
        print(f"wrote {out_compact}")
        print(f"wrote {out_full}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
