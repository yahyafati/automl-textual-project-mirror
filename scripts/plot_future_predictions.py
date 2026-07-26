#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# IO (JSON array or JSONL)
# ----------------------------
def load_trials(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Detect JSON array vs JSONL
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array at top-level.")
        return data

    # JSONL
    trials = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i}: {e}") from e
        trials.append(obj)
    return trials


def stable_config_id(config: Dict[str, Any]) -> str:
    # Stable across runs/machines (unlike Python's hash()).
    s = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ----------------------------
# Extraction helpers
# ----------------------------
def epoch_history_to_series(trial: Dict[str, Any], metric: str) -> pd.Series:
    """
    Returns a Series indexed by epoch (int) -> metric value (float).
    metric usually 'val_accuracy' per your schema.
    """
    hist = trial.get("epoch_history", []) or []
    rows = []
    for r in hist:
        if "epoch" not in r:
            continue
        if metric not in r:
            # allow fallback to val_accuracy if metric missing
            if metric != "val_accuracy" and "val_accuracy" in r:
                val = r["val_accuracy"]
            else:
                continue
        else:
            val = r[metric]
        rows.append((int(r["epoch"]), float(val)))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    s.index.name = "epoch"
    s.name = metric
    return s


def predictions_frame(trial: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      epoch, mean, median, q05, q25, q50, q75, q95
    (only those available).
    """
    pred = trial.get("predictions", {}) or {}
    q_epochs = pred.get("query_epochs", []) or []
    df = pd.DataFrame({"epoch": [int(e) for e in q_epochs]})
    for k in ["mean", "median", "mode", "std", "variance", "ucb", "ei", "pi"]:
        if k in pred and isinstance(pred[k], list) and len(pred[k]) == len(df):
            df[k] = pd.to_numeric(pred[k], errors="coerce")

    q = pred.get("quantiles", {}) or {}
    # Quantile keys are strings like "0.05"
    qmap = {
        "0.05": "q05",
        "0.25": "q25",
        "0.5": "q50",
        "0.75": "q75",
        "0.95": "q95",
    }
    for qk, col in qmap.items():
        if qk in q and isinstance(q[qk], list) and len(q[qk]) == len(df):
            df[col] = pd.to_numeric(q[qk], errors="coerce")

    df = df.sort_values("epoch").reset_index(drop=True)
    return df


@dataclass
class TrialMeta:
    idx: int
    trialNo: Optional[int]
    seed: Optional[int]
    budget: Optional[float]
    val_error: Optional[float]
    best_so_far: Optional[bool]
    timestamp: Optional[str]
    config_id: str


def get_trial_meta(trials: List[Dict[str, Any]]) -> List[TrialMeta]:
    metas = []
    for i, t in enumerate(trials):
        cfg = t.get("config", {}) or {}
        metas.append(
            TrialMeta(
                idx=i,
                trialNo=(
                    int(t["trialNo"])
                    if "trialNo" in t and t["trialNo"] is not None
                    else None
                ),
                seed=(
                    int(t["seed"]) if "seed" in t and t["seed"] is not None else None
                ),
                budget=(
                    float(t["budget"])
                    if "budget" in t and t["budget"] is not None
                    else None
                ),
                val_error=(
                    float(t["val_error"])
                    if "val_error" in t and t["val_error"] is not None
                    else None
                ),
                best_so_far=(bool(t["best_so_far"]) if "best_so_far" in t else None),
                timestamp=(str(t["timestamp"]) if "timestamp" in t else None),
                config_id=stable_config_id(cfg),
            )
        )
    return metas


# ----------------------------
# Plotting
# ----------------------------
def plot_trial_timeline(
    trial: Dict[str, Any],
    meta: TrialMeta,
    metric: str,
    outpath: Optional[str] = None,
    show: bool = False,
):
    pred = trial.get("predictions", {}) or {}
    last_obs = pred.get("last_observed_epoch", None)
    target_epoch = pred.get("target_epoch", None)

    actual = epoch_history_to_series(trial, metric=metric)
    pdf = predictions_frame(trial)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Real curve
    if not actual.empty:
        ax.plot(
            actual.index, actual.values, color="black", lw=2, label=f"actual ({metric})"
        )

    # Prediction mean/median + interval
    if not pdf.empty:
        if "q05" in pdf.columns and "q95" in pdf.columns:
            ax.fill_between(
                pdf["epoch"].values,
                pdf["q05"].values,
                pdf["q95"].values,
                alpha=0.2,
                label="pred 5–95%",
            )
        if "q25" in pdf.columns and "q75" in pdf.columns:
            ax.fill_between(
                pdf["epoch"].values,
                pdf["q25"].values,
                pdf["q75"].values,
                alpha=0.25,
                label="pred 25–75%",
            )

        if "mean" in pdf.columns:
            ax.plot(pdf["epoch"], pdf["mean"], lw=2, label="pred mean")
        elif "median" in pdf.columns:
            ax.plot(pdf["epoch"], pdf["median"], lw=2, label="pred median")
        elif "q50" in pdf.columns:
            ax.plot(pdf["epoch"], pdf["q50"], lw=2, label="pred q50")

    # Mark observed boundary + target epoch
    if last_obs is not None:
        ax.axvline(
            int(last_obs),
            color="tab:blue",
            ls="--",
            lw=1.5,
            label="last observed epoch",
        )
    if target_epoch is not None:
        ax.axvline(
            int(target_epoch), color="tab:orange", ls=":", lw=1.5, label="target epoch"
        )

    title_bits = [f"config_id={meta.config_id[:10]}…", f"trialNo={meta.trialNo}"]
    if meta.seed is not None:
        title_bits.append(f"seed={meta.seed}")
    if meta.val_error is not None:
        title_bits.append(f"val_error={meta.val_error:.4g}")
    ax.set_title(" | ".join(title_bits))

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_target_epoch_calibration(
    trials: List[Dict[str, Any]],
    metas: List[TrialMeta],
    metric: str,
    outpath: Optional[str] = None,
    show: bool = False,
):
    xs, ys = [], []
    labels = []

    for t, m in zip(trials, metas):
        pred = t.get("predictions", {}) or {}
        target_epoch = pred.get("target_epoch", None)
        if target_epoch is None:
            continue
        target_epoch = int(target_epoch)

        actual = epoch_history_to_series(t, metric=metric)
        if actual.empty or target_epoch not in actual.index:
            continue

        pdf = predictions_frame(t)
        if pdf.empty:
            continue
        row = pdf[pdf["epoch"] == target_epoch]
        if row.empty:
            continue

        # choose a prediction column
        if "mean" in row.columns and not np.isnan(row["mean"].iloc[0]):
            x = float(row["mean"].iloc[0])
        elif "median" in row.columns and not np.isnan(row["median"].iloc[0]):
            x = float(row["median"].iloc[0])
        elif "q50" in row.columns and not np.isnan(row["q50"].iloc[0]):
            x = float(row["q50"].iloc[0])
        else:
            continue

        y = float(actual.loc[target_epoch])
        xs.append(x)
        ys.append(y)
        labels.append(m.config_id[:8])

    if not xs:
        print(
            "No trials had both prediction and actual value at target_epoch; skipping calibration plot."
        )
        return

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, alpha=0.7)

    lo = float(np.nanmin([xs.min(), ys.min()]))
    hi = float(np.nanmax([xs.max(), ys.max()]))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="ideal")

    mae = float(np.nanmean(np.abs(xs - ys)))
    rmse = float(np.sqrt(np.nanmean((xs - ys) ** 2)))

    ax.set_title(f"Target-epoch calibration (MAE={mae:.4g}, RMSE={rmse:.4g})")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def plot_interval_coverage(
    trials: List[Dict[str, Any]],
    metas: List[TrialMeta],
    metric: str,
    outpath: Optional[str] = None,
    show: bool = False,
):
    """
    For each trial, checks for epochs in query_epochs where actual exists,
    and computes fraction of times actual lies within [q05, q95].
    """
    coverages = []
    for t in trials:
        actual = epoch_history_to_series(t, metric=metric)
        pdf = predictions_frame(t)
        if actual.empty or pdf.empty:
            continue
        if not ("q05" in pdf.columns and "q95" in pdf.columns):
            continue

        merged = pdf.merge(
            actual.rename("actual").reset_index(),
            on="epoch",
            how="inner",
        )
        if merged.empty:
            continue
        inside = (merged["actual"] >= merged["q05"]) & (
            merged["actual"] <= merged["q95"]
        )
        coverages.append(float(inside.mean()))

    if not coverages:
        print(
            "No trials had both actuals and q05/q95 intervals on overlapping epochs; skipping coverage plot."
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(coverages, bins=12, alpha=0.8)
    ax.axvline(
        np.mean(coverages),
        color="black",
        lw=1.5,
        label=f"mean={np.mean(coverages):.3f}",
    )
    ax.axvline(0.90, color="tab:orange", lw=1.5, ls="--", label="ideal 90%")
    ax.set_title("Coverage of predicted 5–95% interval (per trial)")
    ax.set_xlabel("fraction of epochs where actual ∈ [q05, q95]")
    ax.set_ylabel("count of trials")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def ensure_outdir(out: Optional[str]) -> Optional[str]:
    if out is None:
        return None
    # If user passes a file path ending with an image extension, keep it as file;
    # otherwise treat as directory.
    lower = out.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
        return out
    os.makedirs(out, exist_ok=True)
    return out


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="Path to JSON or JSONL file of trials."
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory (or a single image path). If omitted, only --show displays.",
    )
    ap.add_argument("--show", action="store_true", help="Display plots interactively.")
    ap.add_argument(
        "--metric",
        default="val_accuracy",
        help="Metric in epoch_history to plot (default: val_accuracy).",
    )

    ap.add_argument("--trial", type=int, default=None, help="Plot only this trialNo.")
    ap.add_argument(
        "--config-id", default=None, help="Plot only trials with this config_id (sha1)."
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Select top-k trials by lowest val_error (after filters).",
    )

    args = ap.parse_args()

    trials = load_trials(args.input)
    if not trials:
        raise SystemExit("No trials found in input file.")

    metas = get_trial_meta(trials)

    # Print a small index to help the user find config IDs
    dfm = pd.DataFrame([m.__dict__ for m in metas])
    print("\nFound trials:", len(trials))
    print("Unique config_ids:", dfm["config_id"].nunique())
    print("\nSample (first 10):")
    cols = ["idx", "trialNo", "seed", "budget", "val_error", "best_so_far", "config_id"]
    print(dfm[cols].head(10).to_string(index=False))

    # Filtering
    keep = np.ones(len(trials), dtype=bool)
    if args.trial is not None:
        keep &= (dfm["trialNo"] == args.trial).fillna(False).values
    if args.config_id is not None:
        keep &= (dfm["config_id"] == args.config_id).values

    sel_idx = np.where(keep)[0].tolist()
    if not sel_idx:
        raise SystemExit("No trials match the given filters.")

    # Rank by val_error if available
    sel_df = dfm.loc[sel_idx].copy()
    if sel_df["val_error"].notna().any():
        sel_df = sel_df.sort_values("val_error", ascending=True, na_position="last")
    else:
        sel_df = sel_df.sort_values("idx")

    sel_df = sel_df.head(args.top_k)
    selected_trials = [trials[i] for i in sel_df["idx"].tolist()]
    selected_metas = [metas[i] for i in sel_df["idx"].tolist()]

    out = ensure_outdir(args.out)

    # 1) per-trial timelines
    if out is None or (
        out and not out.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg"))
    ):
        outdir = out
        for t, m in zip(selected_trials, selected_metas):
            fname = f"timeline_config-{m.config_id[:10]}_trial-{m.trialNo}.png"
            outpath = os.path.join(outdir, fname) if outdir else None
            plot_trial_timeline(
                t, m, metric=args.metric, outpath=outpath, show=args.show
            )
    else:
        # If a single file path is provided, just plot the best/first selected trial into it.
        t0, m0 = selected_trials[0], selected_metas[0]
        plot_trial_timeline(t0, m0, metric=args.metric, outpath=out, show=args.show)

    # 2) calibration at target epoch (over selected trials)
    calib_out = None
    coverage_out = None
    if out and not out.lower().endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
        calib_out = os.path.join(out, "calibration_target_epoch.png")
        coverage_out = os.path.join(out, "coverage_q05_q95.png")

    plot_target_epoch_calibration(
        selected_trials,
        selected_metas,
        metric=args.metric,
        outpath=calib_out,
        show=args.show,
    )
    plot_interval_coverage(
        selected_trials,
        selected_metas,
        metric=args.metric,
        outpath=coverage_out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
