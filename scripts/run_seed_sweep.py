#!/usr/bin/env python3
"""Run a matrix of (optimizer variant x seed) AutoML searches on one dataset,
so results can be averaged/compared across seeds per variant afterwards.

Each cell of the matrix is a separate `python -m automl` subprocess, launched
against a generated YAML config that overlays --base-config (default
runconfig.yml) with the variant's optimizer-specific overrides and that
cell's seed/dataset. Output lands in the normal `results/<dataset>/<runtime_id>/`
layout, so existing tools (train_top_k_from_history.py,
plot_wallclock_vs_best_accuracy.py) work unchanged on each individual run.

This script only runs things - it does not plot or average anything itself.
Point plot_wallclock_vs_best_accuracy.py (or your own aggregation) at the
history.log.jsonl files listed in the sweep's manifest.json afterwards.

Usage:
    # Plan only, run nothing (always do this first):
    python scripts/run_seed_sweep.py --dataset amazon --seeds 0 1 2 --dry-run

    # Actually launch the default matrix (random, ifbo-greedy, ifbo-softmax,
    # ifbo-random) x seeds 0,1,2 on amazon, sequentially:
    python scripts/run_seed_sweep.py --dataset amazon --seeds 0 1 2

    # Just the ifbo variants, 5 seeds, on a different dataset:
    python scripts/run_seed_sweep.py --dataset dbpedia --seeds 0 1 2 3 4 \
        --variants ifbo-greedy ifbo-softmax ifbo-random

    # Quick smoke test of the sweep machinery itself (tiny budget/trials):
    python scripts/run_seed_sweep.py --seeds 0 --variants random \
        --override n_trials=2 evaluation_budget=2 max_budget=2
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each entry overrides fields on top of --base-config for that arm of the
# sweep. Edit this dict to add, remove, or rename arms.
#
# ifbo_greedy_candidate_selection is omitted from "ifbo-random" on purpose:
# once ifbo_use_random_selection is True, _select_next_candidate returns
# before ever consulting greedy_selection (see
# automl/core/optimizers/ifbo/optimizer.py), so sweeping it there would just
# be two identical runs under different names.
VARIANTS: dict[str, dict[str, Any]] = {
    "random": {
        "optimizer": "random",
    },
    "ifbo-greedy": {
        "optimizer": "ifbo",
        "ifbo_use_random_selection": False,
        "ifbo_greedy_candidate_selection": True,
    },
    "ifbo-softmax": {
        "optimizer": "ifbo",
        "ifbo_use_random_selection": False,
        "ifbo_greedy_candidate_selection": False,
    },
    "ifbo-random": {
        "optimizer": "ifbo",
        "ifbo_use_random_selection": True,
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Base config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--override entries must be KEY=VALUE, got: {pair!r}")
        key, raw_value = pair.split("=", 1)
        parsed[key] = yaml.safe_load(raw_value)
    return parsed


def build_plan(
    base_cfg: dict[str, Any],
    dataset: str,
    seeds: list[int],
    variants: list[str],
    generic_overrides: dict[str, Any],
    output_root: Path,
    batch_ts: str,
) -> list[dict[str, Any]]:
    plan = []
    for variant in variants:
        for seed in seeds:
            runtime_id = f"[{variant}]seed{seed}_{batch_ts}"
            cfg = copy.deepcopy(base_cfg)
            cfg.update(generic_overrides)
            cfg.update(VARIANTS[variant])
            cfg["dataset"] = dataset
            cfg["seed"] = seed
            cfg["runtime_id"] = runtime_id
            plan.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "runtime_id": runtime_id,
                    "config": cfg,
                    "output_path": output_root / dataset / runtime_id,
                }
            )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default="amazon",
        choices=["ag_news", "imdb", "amazon", "dbpedia", "yelp"],
        help="Dataset all runs in this sweep use (default: amazon).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to repeat every variant with (default: 0 1 2).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=sorted(VARIANTS, key=list(VARIANTS).index),
        help="Which named variants to run (default: all of them).",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "runconfig.yml",
        help="YAML config every variant overlays on top of (default: runconfig.yml). "
        "Controls everything not swept here: n_trials, budgets, approach, "
        "max_num_rows, num_parallel_trials, etc.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra config overrides applied to every run before the variant's "
        "own overrides (e.g. --override n_trials=2 for a quick smoke test).",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Override the base config's output_path (default: whatever "
        "output_path says, normally 'results').",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to launch `-m automl` with (default: current interpreter).",
    )
    parser.add_argument(
        "--stop-on-error",
        dest="continue_on_error",
        action="store_false",
        default=True,
        help="Abort the sweep on the first failed run (default: keep going "
        "and report all failures at the end).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned matrix only. Writes nothing to disk and "
        "launches nothing.",
    )
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    if args.output_path:
        base_cfg["output_path"] = args.output_path
    output_root = Path(base_cfg.get("output_path", "results"))
    generic_overrides = parse_overrides(args.override)

    batch_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_plan(
        base_cfg,
        args.dataset,
        args.seeds,
        args.variants,
        generic_overrides,
        output_root,
        batch_ts,
    )

    print(
        f"Planned {len(plan)} run(s): {len(args.variants)} variant(s) x "
        f"{len(args.seeds)} seed(s), dataset={args.dataset!r}, "
        f"base_config={args.base_config}"
    )
    for entry in plan:
        print(
            f"  {entry['variant']:<14} seed={entry['seed']:<4} -> {entry['output_path']}"
        )

    if args.dry_run:
        print("\n--dry-run: nothing written or launched.")
        return

    sweep_dir = output_root / args.dataset / "_sweeps" / batch_ts
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.json"
    manifest: list[dict[str, Any]] = []

    for entry in plan:
        config_path = sweep_dir / f"{entry['runtime_id']}.yml"
        with config_path.open("w") as f:
            yaml.safe_dump(entry["config"], f, sort_keys=False)

        cmd = [args.python, "-m", "automl", "--config", str(config_path)]
        print(f"\n=== {entry['variant']} seed={entry['seed']} ===\n$ {' '.join(cmd)}")

        log_path = sweep_dir / f"{entry['runtime_id']}.stdout.log"
        record = {
            "variant": entry["variant"],
            "seed": entry["seed"],
            "runtime_id": entry["runtime_id"],
            "config_path": str(config_path),
            "output_path": str(entry["output_path"]),
            "history_path": str(entry["output_path"] / "history.log.jsonl"),
            "log_path": str(log_path),
        }

        start = time.monotonic()
        with log_path.open("w") as log_f:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT
            )
        duration = time.monotonic() - start

        record["status"] = "ok" if proc.returncode == 0 else "failed"
        record["returncode"] = proc.returncode
        record["duration_seconds"] = round(duration, 1)
        manifest.append(record)
        manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"  -> {record['status']} in {duration / 60:.1f} min (log: {log_path})")

        if proc.returncode != 0 and not args.continue_on_error:
            print(
                "Stopping sweep on first failure (omit --stop-on-error to keep going)."
            )
            break

    n_ok = sum(1 for r in manifest if r["status"] == "ok")
    n_failed = sum(1 for r in manifest if r["status"] == "failed")
    print(f"\nSweep complete: {n_ok} ok, {n_failed} failed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
