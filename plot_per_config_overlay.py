#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_trials(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    s = text.lstrip()
    if not s:
        return []
    if s[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Top-level JSON must be an array.")
        return data

    trials = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            trials.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {i}: {e}") from e
    return trials


def stable_config_id(config: Dict[str, Any]) -> str:
    # Stable across runs/machines (unlike Python's built-in hash()).
    s = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def epoch_history_to_series(trial: Dict[str, Any], metric: str) -> pd.Series:
    rows = []
    for r in trial.get("epoch_history", []) or []:
        if "epoch" not in r:
            continue
        if metric in r:
            rows.append((int(r["epoch"]), float(r[metric])))
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(dict(rows)).sort_index()
    s.index.name = "epoch"
    s.name = metric
    return s


def predictions_frame(trial: Dict[str, Any]) -> pd.DataFrame:
    pred = trial.get("predictions", {}) or {}
    q_epochs = pred.get("query_epochs", []) or []
    df = pd.DataFrame({"epoch": [int(e) for e in q_epochs]})

    for k in ["mean", "median"]:
        v = pred.get(k, None)
        if isinstance(v, list) and len(v) == len(df):
            df[k] = pd.to_numeric(v, errors="coerce")

    q = pred.get("quantiles", {}) or {}
    qmap = {"0.05": "q05", "0.5": "q50", "0.95": "q95"}
    for qk, col in qmap.items():
        v = q.get(qk, None)
        if isinstance(v, list) and len(v) == len(df):
            df[col] = pd.to_numeric(v, errors="coerce")

    return df.sort_values("epoch").reset_index(drop=True)


def pick_pred_column(pdf: pd.DataFrame) -> Optional[str]:
    for col in ["mean", "median", "q50"]:
        if col in pdf.columns and pdf[col].notna().any():
            return col
    return None


def config_title(config: Dict[str, Any]) -> str:
    # Keep short; extend if you want more fields.
    mt = config.get("model_type", "unknown_model")
    lr = config.get("learning_rate", None)
    hd = config.get("hidden_dim", None)
    bits = [str(mt)]
    if lr is not None:
        bits.append(f"lr={lr}")
    if hd is not None:
        bits.append(f"hidden={hd}")
    return " | ".join(bits)


def plot_one_config(
    cfg_id: str,
    cfg_trials: List[Dict[str, Any]],
    metric: str,
    outpath: Optional[str],
    show: bool,
    max_trials_per_config: Optional[int] = None,
):
    if max_trials_per_config is not None:
        cfg_trials = cfg_trials[:max_trials_per_config]

    cfg = cfg_trials[0].get("config", {}) or {}
    fig, ax = plt.subplots(figsize=(11, 5.5))

    cmap = plt.get_cmap("tab10")
    n = len(cfg_trials)

    for i, t in enumerate(cfg_trials):
        color = cmap(i % 10)

        trial_no = t.get("trialNo", None)
        seed = t.get("seed", None)
        label_suffix = []
        if trial_no is not None:
            label_suffix.append(f"trial={int(trial_no)}")
        if seed is not None:
            label_suffix.append(f"seed={int(seed)}")
        label_suffix = ", ".join(label_suffix) if label_suffix else f"run{i}"

        actual = epoch_history_to_series(t, metric=metric)
        if not actual.empty:
            ax.plot(
                actual.index,
                actual.values,
                color=color,
                lw=2.0,
                alpha=0.75,
                label=f"actual ({label_suffix})" if n <= 8 else None,
            )

        pdf = predictions_frame(t)
        if not pdf.empty:
            pred_col = pick_pred_column(pdf)
            if pred_col is not None:
                pred = t.get("predictions", {}) or {}
                last_obs = pred.get("last_observed_epoch", None)
                if last_obs is not None:
                    mask = pdf["epoch"].values > int(last_obs)
                else:
                    mask = np.ones(len(pdf), dtype=bool)

                # intervals (if present)
                if "q05" in pdf.columns and "q95" in pdf.columns and mask.any():
                    ax.fill_between(
                        pdf.loc[mask, "epoch"].values,
                        pdf.loc[mask, "q05"].values,
                        pdf.loc[mask, "q95"].values,
                        color=color,
                        alpha=0.12,
                        label="pred q05–q95" if (i == 0 and n <= 8) else None,
                    )

                ax.plot(
                    pdf.loc[mask, "epoch"].values,
                    pdf.loc[mask, pred_col].values,
                    color=color,
                    lw=2.0,
                    ls="--",
                    alpha=0.95,
                    label=f"pred {pred_col} ({label_suffix})" if n <= 8 else None,
                )

    ax.set_title(f"Config {cfg_id[:10]}…  |  {config_title(cfg)}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)

    # If many trials, avoid massive legend.
    if n <= 8:
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()

    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="JSON array file or JSONL file of trials."
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output directory for one image per config. If omitted, use --show.",
    )
    ap.add_argument("--show", action="store_true", help="Show plots interactively.")
    ap.add_argument(
        "--metric",
        default="val_accuracy",
        help="epoch_history metric to plot (default: val_accuracy).",
    )

    ap.add_argument(
        "--top-k-configs",
        type=int,
        default=None,
        help="Only plot top K configs (ranked by best val_error).",
    )
    ap.add_argument(
        "--max-trials-per-config",
        type=int,
        default=None,
        help="Limit number of trials overlaid per config.",
    )

    args = ap.parse_args()

    trials = load_trials(args.input)
    if not trials:
        raise SystemExit("No trials found.")

    # Group by config_id
    groups: Dict[str, List[Dict[str, Any]]] = {}
    best_val_error: Dict[str, float] = {}

    for t in trials:
        cfg = t.get("config", {}) or {}
        cid = stable_config_id(cfg)
        groups.setdefault(cid, []).append(t)
        ve = t.get("val_error", None)
        if ve is not None:
            ve = float(ve)
            best_val_error[cid] = min(best_val_error.get(cid, float("inf")), ve)

    # Order configs by best val_error (if present), else by size desc
    if best_val_error:
        ordered = sorted(
            groups.keys(), key=lambda c: best_val_error.get(c, float("inf"))
        )
    else:
        ordered = sorted(groups.keys(), key=lambda c: len(groups[c]), reverse=True)

    if args.top_k_configs is not None:
        ordered = ordered[: args.top_k_configs]

    if args.out is None and not args.show:
        raise SystemExit("Provide --out to save images or --show to display plots.")

    if args.out is not None:
        os.makedirs(args.out, exist_ok=True)

    print(
        f"Found {len(trials)} trials across {len(groups)} configs. Plotting {len(ordered)} configs..."
    )

    for cid in ordered:
        cfg_trials = groups[cid]
        # Optional: sort trials within config (best first)
        cfg_trials = sorted(
            cfg_trials, key=lambda t: float(t.get("val_error", float("inf")))
        )

        outpath = None
        if args.out is not None:
            outpath = os.path.join(args.out, f"config_{cid[:12]}.png")

        plot_one_config(
            cfg_id=cid,
            cfg_trials=cfg_trials,
            metric=args.metric,
            outpath=outpath,
            show=args.show,
            max_trials_per_config=args.max_trials_per_config,
        )


if __name__ == "__main__":
    main()
