from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Standardization: colors, sizes, resolution
# ---------------------------------------------------------------------------

DEFAULT_DPI = 300
FIGSIZE = (10, 6)
HEATMAP_CMAP = "viridis"

COLORS = {
    "train_loss": "#1f77b4",  # blue
    "val_loss": "#d62728",  # red
    "val_accuracy": "#2ca02c",  # green
    "val_error": "#d62728",  # red
    "best_marker": "#FFD700",  # gold star fill
    "best_edge": "#B8860B",  # dark gold star edge
    "candidate_line": "#9467bd",  # purple
    "duplicate_marker": "#d62728",
    "ideal_line": "#999999",
    "grid": "#e0e0e0",
    "missing_cell": "#f0f0f0",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    }
)


# ---------------------------------------------------------------------------
# Data loading (.json array or .jsonl)
# ---------------------------------------------------------------------------


def load_trials(path) -> list[dict]:
    """
    Load trial records from a .json (array) or .jsonl (one JSON object per
    line) file. Also auto-detects newline-delimited JSON saved with a
    .json extension.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        stripped = text.strip()
        if stripped.startswith("["):
            records = json.loads(stripped)
        else:
            # newline-delimited JSON objects, just saved as .json
            records = [
                json.loads(line) for line in stripped.splitlines() if line.strip()
            ]

    if not isinstance(records, list):
        raise ValueError(f"Expected a list of trial records, got {type(records)}")
    return records


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Best-effort timestamp parser -> datetime; returns None if unparsable."""
    if not ts:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def config_hash(config: dict) -> str:
    """Stable short hash identifying a config, used to detect duplicate candidates."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Standardized axis bounds, computed once across the whole dataset
# ---------------------------------------------------------------------------


def _bounds(values, pad_frac=0.05):
    vals = [v for v in values if v is not None]
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def compute_global_bounds(trials: list[dict]) -> dict:
    """
    Single pass over all trials/epochs to compute standardized min/max
    ranges. Pass the result into each plot function's `bounds=` argument
    so every plot uses identical axis scaling for a given metric.
    """
    trial_nos, val_errors, exec_times = [], [], []
    epochs, train_losses, val_accuracies, val_losses = [], [], [], []

    for t in trials:
        trial_nos.append(t.get("trialNo"))
        val_errors.append(t.get("val_error"))
        exec_times.append(t.get("execution_time"))
        for e in t.get("epoch_history", []) or []:
            epochs.append(e.get("epoch"))
            train_losses.append(e.get("train_loss"))
            val_accuracies.append(e.get("val_accuracy"))
            if "val_loss" in e:
                val_losses.append(e.get("val_loss"))

    return {
        "trial_no": _bounds(trial_nos, pad_frac=0.0),
        "epoch": _bounds(epochs, pad_frac=0.0),
        "train_loss": _bounds(train_losses),
        "val_accuracy": _bounds(val_accuracies),
        "val_loss": _bounds(val_losses) if val_losses else None,
        "val_error": _bounds(val_errors),
        "execution_time": _bounds(exec_times),
    }


# ---------------------------------------------------------------------------
# Plot 1: Epoch Heatmap (Epoch vs Trial No)
# ---------------------------------------------------------------------------


def plot_epoch_heatmap(
    trials, outpath, dpi=DEFAULT_DPI, metric="val_accuracy", bounds=None
):
    """
    Heatmap with Trial No on the y-axis and Epoch on the x-axis, colored by
    `metric` (default 'val_accuracy'; also accepts 'train_loss' or
    'val_loss' if present).
    """
    if bounds is None:
        bounds = compute_global_bounds(trials)

    trial_nos = sorted({t["trialNo"] for t in trials})
    all_epochs = sorted(
        {e["epoch"] for t in trials for e in (t.get("epoch_history") or [])}
    )

    if not trial_nos or not all_epochs:
        raise ValueError("No epoch_history data available to build the heatmap.")

    trial_idx = {tn: i for i, tn in enumerate(trial_nos)}
    epoch_idx = {ep: i for i, ep in enumerate(all_epochs)}

    grid = np.full((len(trial_nos), len(all_epochs)), np.nan)
    for t in trials:
        ti = trial_idx[t["trialNo"]]
        for e in t.get("epoch_history") or []:
            ei = epoch_idx[e["epoch"]]
            val = e.get(metric)
            if val is not None:
                grid[ti, ei] = val

    vmin, vmax = bounds.get(metric) or (None, None)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color=COLORS["missing_cell"])

    im = ax.imshow(
        masked,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[
            min(all_epochs) - 0.5,
            max(all_epochs) + 0.5,
            min(trial_nos) - 0.5,
            max(trial_nos) + 0.5,
        ],
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Trial No")
    ax.set_title(f"Epoch Heatmap ({metric.replace('_', ' ').title()})")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.replace("_", " ").title())

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return str(outpath)


# ---------------------------------------------------------------------------
# Plot 2: Learning Curves (Train Loss vs Epoch, Validation metric vs Epoch)
# ---------------------------------------------------------------------------


def _val_metric_for_epoch(epoch_entry: dict):
    """Prefer val_loss if the data actually has it, else fall back to val_accuracy."""
    if "val_loss" in epoch_entry and epoch_entry["val_loss"] is not None:
        return "val_loss", epoch_entry["val_loss"]
    return "val_accuracy", epoch_entry.get("val_accuracy")


def plot_learning_curves(
    trials, outpath, dpi=DEFAULT_DPI, bounds=None, max_labeled_trials=15
):
    """
    Two side-by-side panels, one line per trial:
      left  = Train Loss vs Epoch
      right = Validation Loss vs Epoch if val_loss is present in the data,
              otherwise Validation Accuracy vs Epoch (schema default).
    """
    if bounds is None:
        bounds = compute_global_bounds(trials)

    sorted_trials = sorted(trials, key=lambda t: t["trialNo"])

    # Decide which validation metric this dataset actually has.
    val_key = "val_accuracy"
    for t in sorted_trials:
        for e in t.get("epoch_history") or []:
            k, _ = _val_metric_for_epoch(e)
            if k == "val_loss":
                val_key = "val_loss"
            break
        if val_key == "val_loss":
            break

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 1.8, FIGSIZE[1]))
    cmap = plt.get_cmap("tab20")
    n = len(sorted_trials)

    for i, t in enumerate(sorted_trials):
        hist = sorted(t.get("epoch_history") or [], key=lambda e: e["epoch"])
        if not hist:
            continue
        epochs = [e["epoch"] for e in hist]
        train_loss = [e.get("train_loss") for e in hist]
        val_vals = [e.get(val_key) for e in hist]

        color = cmap(i % 20)
        label = f"Trial {t['trialNo']}" if n <= max_labeled_trials else None

        ax_train.plot(
            epochs, train_loss, color=color, alpha=0.8, linewidth=1.5, label=label
        )
        ax_val.plot(
            epochs, val_vals, color=color, alpha=0.8, linewidth=1.5, label=label
        )

    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train Loss")
    ax_train.set_title("Train Loss vs Epoch")
    if bounds.get("epoch"):
        ax_train.set_xlim(bounds["epoch"])
    if bounds.get("train_loss"):
        ax_train.set_ylim(bounds["train_loss"])

    val_bounds = bounds.get(val_key)
    val_label = "Validation Loss" if val_key == "val_loss" else "Validation Accuracy"
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel(val_label)
    ax_val.set_title(f"{val_label} vs Epoch")
    if bounds.get("epoch"):
        ax_val.set_xlim(bounds["epoch"])
    if val_key in ["val_loss", "val_accuracy"]:
        ax_val.set_ylim(0, 1)
    elif val_bounds:
        ax_val.set_ylim(val_bounds)

    if n <= max_labeled_trials:
        ax_val.legend(loc="best", fontsize=8, ncol=2)
    else:
        ax_val.text(
            0.02,
            0.98,
            f"{n} trials (legend hidden)",
            transform=ax_val.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#666666",
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return str(outpath)


# ---------------------------------------------------------------------------
# Plot 3: Validation Error vs Time (best_so_far starred)
# ---------------------------------------------------------------------------


def plot_val_error_vs_time(
    trials, outpath, dpi=DEFAULT_DPI, bounds=None, time_field="execution_time"
):
    """
    Validation error over time. `time_field` is 'timestamp' (wall-clock,
    parsed from the `timestamp` field) or 'execution_time' (cumulative sum
    of each trial's execution_time). Falls back to execution_time
    automatically if timestamps can't be parsed.

    Trials with best_so_far=True are marked with a gold star.
    """
    if bounds is None:
        bounds = compute_global_bounds(trials)

    sorted_trials = sorted(trials, key=lambda t: t["trialNo"])

    use_dates = False
    if time_field == "timestamp":
        x_vals = [_parse_timestamp(t.get("timestamp")) for t in sorted_trials]
        use_dates = all(x is not None for x in x_vals) and len(x_vals) > 0
        if not use_dates:
            time_field = "execution_time"

    if time_field == "execution_time":
        x_vals = list(np.cumsum([t.get("execution_time") or 0 for t in sorted_trials]))

    y_vals = [t.get("val_error") for t in sorted_trials]
    is_best = [bool(t.get("best_so_far")) for t in sorted_trials]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(x_vals, y_vals, color=COLORS["val_error"], alpha=0.3, linewidth=1, zorder=1)

    reg_x = [x for x, b in zip(x_vals, is_best) if not b]
    reg_y = [y for y, b in zip(y_vals, is_best) if not b]
    ax.scatter(
        reg_x,
        reg_y,
        color=COLORS["val_error"],
        s=35,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        label="Trial",
        zorder=2,
    )

    best_x = [x for x, b in zip(x_vals, is_best) if b]
    best_y = [y for y, b in zip(y_vals, is_best) if b]
    ax.scatter(
        best_x,
        best_y,
        marker="*",
        s=280,
        color=COLORS["best_marker"],
        edgecolor=COLORS["best_edge"],
        linewidth=1.2,
        label="Best so far",
        zorder=3,
    )

    ax.set_ylabel("Validation Error")
    ax.set_xlabel("Timestamp" if use_dates else "Cumulative Execution Time (s)")
    ax.set_title("Validation Error vs Time")
    if bounds.get("val_error"):
        ax.set_ylim(bounds["val_error"])

    if use_dates:
        fig.autofmt_xdate()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return str(outpath)


# ---------------------------------------------------------------------------
# Plot 4: Unique candidates vs Trial (hash(config) dedup check)
# ---------------------------------------------------------------------------


def plot_unique_candidates_vs_trial(trials, outpath, dpi=DEFAULT_DPI, bounds=None):
    """
    Number of unique candidate configs seen so far (up to each trialNo).
    A duplicate trial is marked with a red X.
    """
    if bounds is None:
        bounds = compute_global_bounds(trials)

    sorted_trials = sorted(trials, key=lambda t: t["trialNo"])

    seen = set()
    unique_so_far = []
    is_duplicate = []

    for t in sorted_trials:
        h = config_hash(t.get("config", {}))
        dup = h in seen
        is_duplicate.append(dup)
        if not dup:
            seen.add(h)
        unique_so_far.append(len(seen))

    trial_nos = [t["trialNo"] for t in sorted_trials]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # "seen so far at trial N" -> regular line is unambiguous
    ax.plot(
        trial_nos,
        unique_so_far,
        color=COLORS["candidate_line"],
        linewidth=2,
        label="Unique candidates (seen so far)",
    )

    # mark duplicates (same y as previous, because unique count didn't increase)
    dup_x = [tn for tn, d in zip(trial_nos, is_duplicate) if d]
    dup_y = [y for y, d in zip(unique_so_far, is_duplicate) if d]
    if dup_x:
        ax.scatter(
            dup_x,
            dup_y,
            color=COLORS["duplicate_marker"],
            marker="x",
            s=70,
            linewidth=2,
            zorder=3,
            label="Duplicate config",
        )

    ax.set_xlabel("Trial No")
    ax.set_ylabel("Unique candidates seen so far")
    ax.set_title("Unique Candidates Seen So Far vs Trial")

    if bounds.get("trial_no"):
        lo, hi = bounds["trial_no"]
        ax.set_xlim(lo - 0.5, hi + 0.5)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return str(outpath)


def plot_learning_curves_by_config(
    trials,
    outpath,
    dpi=DEFAULT_DPI,
    bounds=None,
    max_labeled_configs=15,
):
    """
    Like plot_learning_curves(), but colors/labels lines by *config* (hash(config))
    instead of by trial number. Trials that share the same config get the same color.

    Two side-by-side panels:
      left  = Train Loss vs Epoch
      right = Validation Loss vs Epoch if val_loss is present, else Val Accuracy vs Epoch
    """
    if bounds is None:
        bounds = compute_global_bounds(trials)

    sorted_trials = sorted(trials, key=lambda t: t["trialNo"])

    # Decide which validation metric this dataset actually has.
    val_key = "val_accuracy"
    for t in sorted_trials:
        for e in t.get("epoch_history") or []:
            k, _ = _val_metric_for_epoch(e)
            if k == "val_loss":
                val_key = "val_loss"
            break
        if val_key == "val_loss":
            break

    # Group trials by config hash
    groups = {}  # cfg_hash -> list[trial]
    for t in sorted_trials:
        h = config_hash(t.get("config", {}))
        groups.setdefault(h, []).append(t)

    # Stable ordering of config groups by first-seen trialNo
    cfg_order = sorted(
        groups.keys(), key=lambda h: min(tt["trialNo"] for tt in groups[h])
    )

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 1.8, FIGSIZE[1]))

    cmap = plt.get_cmap("tab20")
    cfg_to_color = {h: cmap(i % 20) for i, h in enumerate(cfg_order)}

    # Plot: every trial is a line, but colored by its config
    for h in cfg_order:
        color = cfg_to_color[h]
        trials_for_cfg = sorted(groups[h], key=lambda t: t["trialNo"])

        # Label each config once (legend shows configs, not trials)
        label = None
        if len(cfg_order) <= max_labeled_configs:
            label = f"Config {h[:8]} ({len(trials_for_cfg)} trial{'s' if len(trials_for_cfg) != 1 else ''})"

        first_line_for_label = True
        for t in trials_for_cfg:
            hist = sorted(t.get("epoch_history") or [], key=lambda e: e["epoch"])
            if not hist:
                continue

            epochs = [e["epoch"] for e in hist]
            train_loss = [e.get("train_loss") for e in hist]
            val_vals = [e.get(val_key) for e in hist]

            ax_train.plot(
                epochs,
                train_loss,
                color=color,
                alpha=0.8,
                linewidth=1.5,
                label=(label if first_line_for_label else None),
            )
            ax_val.plot(
                epochs,
                val_vals,
                color=color,
                alpha=0.8,
                linewidth=1.5,
                label=(label if first_line_for_label else None),
            )
            first_line_for_label = False

    # Axes formatting (same as original)
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Train Loss")
    ax_train.set_title("Train Loss vs Epoch")
    if bounds.get("epoch"):
        ax_train.set_xlim(bounds["epoch"])
    if bounds.get("train_loss"):
        ax_train.set_ylim(bounds["train_loss"])

    val_bounds = bounds.get(val_key)
    val_label = "Validation Loss" if val_key == "val_loss" else "Validation Accuracy"
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel(val_label)
    ax_val.set_title(f"{val_label} vs Epoch")
    if bounds.get("epoch"):
        ax_val.set_xlim(bounds["epoch"])
    if val_bounds:
        ax_val.set_ylim(val_bounds)

    # Legend handling
    if len(cfg_order) <= max_labeled_configs:
        ax_val.legend(loc="best", fontsize=8, ncol=1)
    else:
        ax_val.text(
            0.02,
            0.98,
            f"{len(cfg_order)} unique configs (legend hidden)",
            transform=ax_val.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="#666666",
        )

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return str(outpath)


def plot_config_hash_vs_time(
    trials,
    outpath,
    dpi=DEFAULT_DPI,
    bounds=None,
    hash_len=12,
    show_trial_labels=False,
    overlay_val_error=True,
):
    """
    Gantt-like plot: config-hash (y, ordered by first appearance) vs cumulative time (x).
    No "trialNo" needed for y-ordering; we use first-seen order of config hashes.

    - Each trial is a horizontal bar from its start time to end time.
    - y-axis categories are config hashes in order of first appearance.
    - Optionally overlays val_error points at trial end time (and best_so_far stars).
    """
    if bounds is None:
        bounds = compute_global_bounds(trials)

    # keep chronological order by trialNo if present; else preserve given order
    if trials and isinstance(trials[0], dict) and "trialNo" in trials[0]:
        sorted_trials = sorted(trials, key=lambda t: t.get("trialNo", 0))
    else:
        sorted_trials = list(trials)

    # cumulative execution time starting at 0 (trial start times)
    durations = [t.get("execution_time") or 0 for t in sorted_trials]
    start_times = [0] + list(np.cumsum(durations[:-1]))
    end_times = list(np.cumsum(durations))

    # config hash in order of first appearance
    hashes = [config_hash(t.get("config", {}))[:hash_len] for t in sorted_trials]
    hash_to_y = {}
    y_labels = []
    for h in hashes:
        if h not in hash_to_y:
            hash_to_y[h] = len(y_labels)
            y_labels.append(h)
    y_vals = [hash_to_y[h] for h in hashes]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # color per config hash (stable across plot)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(hash_to_y[h] % cmap.N) for h in hashes]

    bar_h = 0.7
    for i, (y, x0, x1, c) in enumerate(zip(y_vals, start_times, end_times, colors)):
        ax.barh(
            y=y,
            width=max(0, x1 - x0),
            left=x0,
            height=bar_h,
            color=c,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            zorder=1,
        )
        if show_trial_labels:
            ax.text(
                x0 + 0.01 * (end_times[-1] if end_times else 1),
                y,
                f"trial {sorted_trials[i].get('trialNo', i)}",
                va="center",
                ha="left",
                fontsize=9,
                color="#222222",
                zorder=3,
            )

    ax.set_xlabel("Cumulative Execution Time (s)")
    ax.set_ylabel("Config hash (first-seen order)")
    ax.set_title("Config Hash vs Time")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.7)
    ax.grid(False, axis="y")

    # optional val_error overlay at trial end time
    if overlay_val_error:
        ax2 = ax.twinx()
        val_err = [t.get("val_error") for t in sorted_trials]
        ax2.scatter(
            end_times,
            val_err,
            s=35,
            color=COLORS["val_error"],
            alpha=0.9,
            label="val_error",
            zorder=4,
        )
        best_x = [et for et, t in zip(end_times, sorted_trials) if t.get("best_so_far")]
        best_y = [t.get("val_error") for t in sorted_trials if t.get("best_so_far")]
        if best_x:
            ax2.scatter(
                best_x,
                best_y,
                marker="*",
                s=220,
                facecolor=COLORS["best_marker"],
                edgecolor=COLORS["best_edge"],
                linewidth=1.2,
                label="best_so_far",
                zorder=5,
            )

        ax2.set_ylabel("val_error")
        # reuse standardized bounds if present
        if bounds.get("val_error"):
            lo, hi = bounds["val_error"]
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                ax2.set_ylim(lo, hi)

        # combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h2 or h1:
            ax2.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return str(outpath)


# ---------------------------------------------------------------------------
# Save all plots
# ---------------------------------------------------------------------------


def save_all_plots(trials, outdir=".", dpi=DEFAULT_DPI, prefix="", fmt="png") -> dict:
    """
    Generate and save all four diagnostic plots into `outdir`.
    Returns a dict mapping plot name -> saved file path.
    """
    os.makedirs(outdir, exist_ok=True)
    bounds = compute_global_bounds(trials)

    jobs = {
        "epoch_heatmap": plot_epoch_heatmap,
        "learning_curves": plot_learning_curves,
        "val_error_vs_time": plot_val_error_vs_time,
        "unique_candidates_vs_trial": plot_unique_candidates_vs_trial,
        "config_space": plot_learning_curves_by_config,
        "config_vs_time": plot_config_hash_vs_time,
    }

    outputs = {}
    for name, fn in jobs.items():
        outpath = os.path.join(outdir, f"{prefix}{name}.{fmt}")
        outputs[name] = fn(trials, outpath, dpi=dpi, bounds=bounds)

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate standardized trial diagnostic plots."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to a .json or .jsonl trial log file"
    )
    parser.add_argument(
        "--outdir", "-o", default="./plots", help="Output directory for plot images"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Resolution (dots per inch) of saved images",
    )
    parser.add_argument("--prefix", default="", help="Filename prefix for outputs")
    parser.add_argument(
        "--format", default="png", help="Image format, e.g. png, pdf, svg"
    )
    args = parser.parse_args()

    trials = load_trials(args.input)
    if not trials:
        raise SystemExit("No trial records found in input file.")

    paths = save_all_plots(
        trials, outdir=args.outdir, dpi=args.dpi, prefix=args.prefix, fmt=args.format
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
