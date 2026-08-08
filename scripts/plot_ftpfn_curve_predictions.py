#!/usr/bin/env python3
"""
Evaluate how well the FT-PFN surrogate (used by the `ifbo` optimizer) predicts
freeze-thaw learning curves, replayed against a real `history.log.jsonl`.

For each trial (in the order it actually ran), this reconstructs the exact
in-context-learning setup ifBO used at that point in the run:

  - context: every other candidate's curve as it was known *right before*
    this trial ran, plus this trial's own candidate curve prior state
    (its "observed prefix"), each Curve keyed by a hyperparameter encoding
    derived from the observed configs in this history file.
  - query: the epochs this trial newly trained (the genuinely unseen
    "future" of that candidate's curve at the time).

It then asks FT-PFN for its mean prediction (and optionally 10-90% quantile
band) over that query, and plots it against what actually happened -- one
subplot per trial.

Usage:
  python scripts/plot_ftpfn_curve_predictions.py \
      actual.ignore.results/ag_news/20260802_163658_e65e5e30/history.log.jsonl

  # with uncertainty band, capped to 24 trials/page
  python scripts/plot_ftpfn_curve_predictions.py <history.log.jsonl> \
      --show-quantiles --trials-per-page 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import torch

from ifbo import Curve
from ifbo.surrogate import FTPFN

from automl.core.optimizers.ifbo.hp_space import (
    Categorical,
    Float,
    HPSpec,
    HyperparameterSpace,
    Integer,
)

# Hyperparameters known (from configspacehelper.py) to be searched on a log scale.
LOG_SCALE_KEYS = {"learning_rate", "weight_decay"}
# Excluded from the encoding: uninformative/degenerate for this diagnostic
# (constant across the whole file, or would need special text handling).
EXCLUDE_KEYS = {"model_type"}


def load_trials(path: str) -> list[dict[str, Any]]:
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))
    trials.sort(key=lambda t: float(t.get("trialNo", 0)))
    return trials


def config_id(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


def build_hp_space(trials: list[dict[str, Any]]) -> HyperparameterSpace:
    """
    Build a HyperparameterSpace by inspecting the actual observed configs in
    this history file, rather than importing configspacehelper.build_config_space
    -- the search space definition can (and does) drift across commits, so a
    historical history.log.jsonl may contain keys/choices that no longer match
    the current config space (e.g. a hyperparameter that used to be tunable and
    is now fixed). Reading the ranges straight from the data keeps this script
    correct for any past run.
    """
    keys: set[str] = set()
    for t in trials:
        keys.update(t["config"].keys())
    keys -= EXCLUDE_KEYS

    values_by_key: dict[str, list[Any]] = {k: [] for k in keys}
    for t in trials:
        cfg = t["config"]
        for k in keys:
            if k in cfg and cfg[k] is not None:
                values_by_key[k].append(cfg[k])

    specs: dict[str, HPSpec] = {}
    for k in sorted(keys):
        vals = values_by_key[k]
        if not vals:
            continue
        is_numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals
        )
        unique_vals = sorted(set(vals)) if is_numeric else list(dict.fromkeys(vals))

        if not is_numeric or len(unique_vals) <= 8:
            # Small/short discrete choice set (or non-numeric): treat as categorical,
            # matching how batch_size/hidden_dim/etc. are actually defined as
            # Categorical hyperparameters in configspacehelper.py.
            specs[k] = Categorical(tuple(unique_vals))
            continue

        lo, hi = float(min(vals)), float(max(vals))
        if lo == hi:
            hi = lo + 1e-8
        log = k in LOG_SCALE_KEYS and lo > 0
        all_int = all(float(v).is_integer() for v in vals)
        specs[k] = (
            Integer(int(lo), int(hi), log=log) if all_int else Float(lo, hi, log=log)
        )

    return HyperparameterSpace(**specs)


def epoch_series(trial: dict[str, Any]) -> list[tuple[int, float]]:
    hist = trial.get("epoch_history") or []
    pairs = [
        (int(e["epoch"]), float(e["val_accuracy"]))
        for e in hist
        if e.get("val_accuracy") is not None
    ]
    pairs.sort(key=lambda p: p[0])
    return pairs


def normalize(epochs: list[int], min_budget: int, b_max: int) -> np.ndarray:
    return np.array([(e - min_budget + 1) / b_max for e in epochs], dtype=np.float32)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("history_path", help="Path to a history.log.jsonl file")
    p.add_argument(
        "--runtime-config",
        default=None,
        help="Path to runtime_config.json (default: runtime_config.json next to history_path)",
    )
    p.add_argument(
        "--min-budget", type=int, default=None, help="Override min_budget (epochs)"
    )
    p.add_argument(
        "--max-budget", type=int, default=None, help="Override max_budget (epochs)"
    )
    p.add_argument("--model-version", default="0.0.1")
    p.add_argument(
        "--model-path", default=".model", help="Directory containing FT-PFN weights"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--show-quantiles", action="store_true", help="Shade 10-90%% predictive band"
    )
    p.add_argument("--trials-per-page", type=int, default=24)
    p.add_argument("--max-cols", type=int, default=4)
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path prefix (default: <history_dir>/ftpfn_curve_predictions.png)",
    )
    args = p.parse_args()

    trials = load_trials(args.history_path)
    if not trials:
        raise ValueError(f"No trials found in {args.history_path}")

    min_budget, max_budget = args.min_budget, args.max_budget
    if min_budget is None or max_budget is None:
        rc_path = args.runtime_config or os.path.join(
            os.path.dirname(os.path.abspath(args.history_path)), "runtime_config.json"
        )
        if os.path.exists(rc_path):
            with open(rc_path, "r", encoding="utf-8") as f:
                rc = json.load(f)
            min_budget = min_budget if min_budget is not None else int(rc["min_budget"])
            max_budget = max_budget if max_budget is not None else int(rc["max_budget"])
            print(
                f"[info] loaded min_budget={min_budget}, max_budget={max_budget} from {rc_path}"
            )
        else:
            raise ValueError(
                "Could not find runtime_config.json next to the history file, and "
                "--min-budget/--max-budget were not given. Please pass them explicitly."
            )
    b_max = max_budget - min_budget + 1

    hp_space = build_hp_space(trials)
    print(f"[info] hyperparameter encoding dims ({hp_space.dim}): {hp_space.names}")
    if hp_space.dropped_names:
        print(f"[info] dropped (>10-dim cap): {hp_space.dropped_names}")

    print(f"[info] loading FT-PFN surrogate (version={args.model_version}) ...")
    model = FTPFN(
        version=args.model_version,
        target_path=args.model_path,
        device=torch.device(args.device),
    )
    model.eval()

    # First pass: since epoch_history is cumulative per config (a later trial for
    # the same config repeats and extends its earlier epoch_history), the last
    # trial seen for a given config holds its full, final known curve. Recording
    # that lets each trial's forecast reach past just the one epoch it newly
    # trained (often only 1, under the default ifbo_thaw_step) out to everything
    # eventually known about that candidate -- a far more legible "predicted vs.
    # actual" comparison than a single dangling point.
    final_pairs: dict[str, list[tuple[int, float]]] = {}
    final_all_pairs: dict[str, list[tuple[int, float]]] = {}
    for trial in trials:
        cid = config_id(trial["config"])
        all_pairs = epoch_series(trial)
        if not all_pairs:
            continue
        final_all_pairs[cid] = all_pairs
        ft_pairs = [pr for pr in all_pairs if pr[0] >= min_budget]
        if ft_pairs:
            final_pairs[cid] = ft_pairs

    state: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for trial in trials:
        cfg = trial["config"]
        cid = config_id(cfg)
        pairs = [pr for pr in epoch_series(trial) if pr[0] >= min_budget]
        if not pairs:
            continue

        prev = state.get(cid)
        prev_max_epoch = prev["epochs"][-1] if prev else 0
        future_pairs = [pr for pr in final_pairs[cid] if pr[0] > prev_max_epoch]

        z = hp_space.encode(cfg)

        if future_pairs:
            context_curves: list[Curve] = []
            for other_cid, s in state.items():
                if other_cid == cid:
                    continue
                context_curves.append(
                    Curve(
                        hyperparameters=s["z"],
                        t=torch.tensor(
                            normalize(s["epochs"], min_budget, b_max),
                            dtype=torch.float32,
                        ),
                        y=torch.tensor(s["accs"], dtype=torch.float32),
                    )
                )
            if prev is not None:
                context_curves.append(
                    Curve(
                        hyperparameters=prev["z"],
                        t=torch.tensor(
                            normalize(prev["epochs"], min_budget, b_max),
                            dtype=torch.float32,
                        ),
                        y=torch.tensor(prev["accs"], dtype=torch.float32),
                    )
                )

            # FT-PFN's tokenizer can't handle a fully-empty context (torch.stack on
            # an empty list) -- this only happens for the very first trial of the
            # very first candidate in the whole run, before anything else has been
            # observed. The real ifBO optimizer never calls model.predict() at that
            # point either (it always samples the first candidate unconditionally),
            # so there is nothing meaningful to compare against here.
            if not context_curves:
                state[cid] = {
                    "epochs": [e for e, _ in pairs],
                    "accs": [a for _, a in pairs],
                    "z": z,
                }
                continue

            query_epochs = [pr[0] for pr in future_pairs]
            query_curve = Curve(
                hyperparameters=z,
                t=torch.tensor(
                    normalize(query_epochs, min_budget, b_max), dtype=torch.float32
                ),
            )

            with torch.no_grad():
                pred = model.predict(context=context_curves, query=[query_curve])[0]
                mean = pred.criterion.mean(pred.logits).squeeze(-1).cpu().numpy()
                lo = hi = None
                if args.show_quantiles:
                    lo = pred.quantile(0.1).cpu().numpy()
                    hi = pred.quantile(0.9).cpu().numpy()

            actual = np.array([a for _, a in future_pairs], dtype=np.float32)
            mae = float(np.mean(np.abs(mean - actual)))

            records.append(
                {
                    "trial_no": int(float(trial.get("trialNo", 0))),
                    "config_id": cid,
                    "all_epochs": [e for e, _ in final_all_pairs[cid]],
                    "all_accs": [a for _, a in final_all_pairs[cid]],
                    "obs_boundary_epoch": prev_max_epoch if prev else None,
                    "n_ctx_curves": len(context_curves),
                    "pred_epochs": query_epochs,
                    "pred_mean": mean,
                    "pred_lo": lo,
                    "pred_hi": hi,
                    "mae": mae,
                }
            )

        state[cid] = {
            "epochs": [e for e, _ in pairs],
            "accs": [a for _, a in pairs],
            "z": z,
        }

    if not records:
        raise ValueError("No trials produced a new-epoch prediction (nothing to plot).")

    all_mae = np.array([r["mae"] for r in records])
    print(
        f"[info] {len(records)} trials evaluated | mean MAE={all_mae.mean():.4f} "
        f"| median MAE={np.median(all_mae):.4f} | worst MAE={all_mae.max():.4f}"
    )

    out_prefix = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.history_path)),
        "ftpfn_curve_predictions.png",
    )
    root, ext = os.path.splitext(out_prefix)
    ext = ext or ".png"

    n_pages = math.ceil(len(records) / args.trials_per_page)
    for page in range(n_pages):
        page_records = records[
            page * args.trials_per_page : (page + 1) * args.trials_per_page
        ]
        n = len(page_records)
        cols = min(args.max_cols, n)
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(
            rows, cols, figsize=(4.6 * cols, 3.2 * rows), squeeze=False
        )
        axes_flat = axes.flatten()

        for ax, r in zip(axes_flat, page_records):
            ax.plot(
                r["all_epochs"],
                r["all_accs"],
                lw=2.0,
                marker="o",
                ms=3,
                color="C0",
                label="actual",
            )
            ax.plot(
                r["pred_epochs"],
                r["pred_mean"],
                lw=2.0,
                ls="--",
                marker="x",
                ms=6,
                color="C1",
                label="FT-PFN predicted",
            )
            if r["pred_lo"] is not None:
                ax.fill_between(
                    r["pred_epochs"], r["pred_lo"], r["pred_hi"], color="C1", alpha=0.15
                )
            if r["obs_boundary_epoch"] is not None:
                obs_epoch = r["obs_boundary_epoch"]
                ax.axvline(obs_epoch, color="k", alpha=0.2, lw=1.2)
                obs_e = [e for e in r["all_epochs"] if e <= obs_epoch]
                obs_a = r["all_accs"][: len(obs_e)]
                ax.scatter(obs_e, obs_a, s=16, color="C0", alpha=0.7)
            ax.set_title(
                f"trial {r['trial_no']} | cfg {r['config_id']} | ctx={r['n_ctx_curves']} "
                f"| MAE={r['mae']:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("epoch")
            ax.set_ylabel("val_accuracy")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=7)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            f"FT-PFN actual vs predicted curves | {os.path.basename(args.history_path)} "
            f"| page {page + 1}/{n_pages} | mean MAE={all_mae.mean():.4f}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        out_path = f"{root}{ext}" if n_pages == 1 else f"{root}_p{page + 1}{ext}"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[saved] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
