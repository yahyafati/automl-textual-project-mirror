#!/usr/bin/env python3
"""Plot wallclock time vs. best-validation-accuracy-so-far for one or more
history.log.jsonl files, overlaid on a single figure.

Wallclock is derived from each trial's completion `timestamp`, relative to the
first trial's timestamp in that file -- not from summing `execution_time` --
so that runs using overlapping/parallel trials (e.g. ifbo with
num_parallel_trials > 1) still get a real elapsed-time axis instead of an
inflated sum of concurrent durations.

Usage:
    python plot_wallclock_vs_best_accuracy.py --inputs results/a/history.log.jsonl \
        results/b/history.log.jsonl --out wallclock_vs_best_accuracy.png

    python plot_wallclock_vs_best_accuracy.py --inputs run1.jsonl run2.jsonl \
        --labels "SMAC" "ifBO" --time-unit minutes --show
"""
import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

TIME_UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
}

TIMESTAMP_FMT = "%Y%m%d_%H%M%S,%f"


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
            raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e
    return trials


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, TIMESTAMP_FMT)


def wallclock_vs_best_accuracy(
    trials: List[Dict[str, Any]],
) -> Tuple[List[float], List[float]]:
    """Returns (wallclock_seconds, best_val_accuracy_so_far), sorted by completion time."""
    records = []
    for t in trials:
        ts = t.get("timestamp")
        val_error = t.get("val_error")
        if ts is None or val_error is None:
            continue
        records.append((parse_timestamp(ts), float(val_error)))

    if not records:
        return [], []

    records.sort(key=lambda r: r[0])
    t0 = records[0][0]

    wallclock: List[float] = []
    best_acc: List[float] = []
    running_best = float("-inf")
    for ts, val_error in records:
        wallclock.append((ts - t0).total_seconds())
        running_best = max(running_best, 1.0 - val_error)
        best_acc.append(running_best)

    return wallclock, best_acc


def raw_accuracies(
    trials: List[Dict[str, Any]],
) -> Tuple[List[float], List[float]]:
    """Per-trial (non-cumulative) accuracy points, for a faint scatter overlay."""
    records = []
    for t in trials:
        ts = t.get("timestamp")
        val_error = t.get("val_error")
        if ts is None or val_error is None:
            continue
        records.append((parse_timestamp(ts), 1.0 - float(val_error)))

    if not records:
        return [], []

    records.sort(key=lambda r: r[0])
    t0 = records[0][0]
    wallclock = [(ts - t0).total_seconds() for ts, _ in records]
    acc = [a for _, a in records]
    return wallclock, acc


def plot_wallclock_vs_best_accuracy(
    inputs: List[str],
    labels: Optional[List[str]],
    save_path: Optional[str],
    show: bool,
    time_unit: str,
    log_x: bool,
) -> None:
    if labels is not None and len(labels) != len(inputs):
        raise ValueError("--labels must have the same length as --inputs.")

    divisor = TIME_UNIT_SECONDS[time_unit]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, path in enumerate(inputs):
        color = cmap(i % 10)
        label = labels[i] if labels else os.path.basename(os.path.dirname(path) or path) or path

        trials = load_trials(path)
        if not trials:
            print(f"Warning: no trials found in {path}, skipping.")
            continue

        raw_x, raw_y = raw_accuracies(trials)
        best_x, best_y = wallclock_vs_best_accuracy(trials)
        if not best_x:
            print(f"Warning: no usable (timestamp, val_error) pairs in {path}, skipping.")
            continue

        raw_x_scaled = [x / divisor for x in raw_x]
        best_x_scaled = [x / divisor for x in best_x]

        ax.scatter(raw_x_scaled, raw_y, color=color, alpha=0.2, s=20, zorder=1)
        ax.step(
            best_x_scaled,
            best_y,
            where="post",
            color=color,
            linewidth=2.0,
            label=label,
            zorder=2,
        )

        print(
            f"{label}: {len(best_x)} trials, "
            f"final best val_accuracy={best_y[-1]:.4f}, "
            f"total wallclock={best_x[-1] / divisor:.2f} {time_unit}"
        )

    ax.set_xlabel(f"Wallclock time ({time_unit})")
    ax.set_ylabel("Best validation accuracy so far")
    ax.set_title("Wallclock time vs. best validation accuracy")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more history.log.jsonl (or JSON-array) files to overlay.",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels, one per --inputs entry (default: parent directory name).",
    )
    ap.add_argument(
        "--out",
        default="wallclock_vs_best_accuracy.png",
        help="Output image path (default: wallclock_vs_best_accuracy.png).",
    )
    ap.add_argument("--show", action="store_true", help="Show the plot interactively.")
    ap.add_argument(
        "--time-unit",
        choices=list(TIME_UNIT_SECONDS.keys()),
        default="minutes",
        help="Unit for the wallclock axis (default: minutes).",
    )
    ap.add_argument(
        "--log-x", action="store_true", help="Use a log scale for the wallclock axis."
    )

    args = ap.parse_args()

    plot_wallclock_vs_best_accuracy(
        inputs=args.inputs,
        labels=args.labels,
        save_path=args.out,
        show=args.show,
        time_unit=args.time_unit,
        log_x=args.log_x,
    )


if __name__ == "__main__":
    main()
