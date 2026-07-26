#!/usr/bin/env python3
"""
ifbo_predict.py
================

Generate learning-curve predictions (with full uncertainty quantification)
for a set of HPO trials using the FT-PFN surrogate from ifBO
(In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization,
Rakotoarison et al., ICML 2024 - https://arxiv.org/html/2404.16795v3).

WHAT THIS DOES
--------------
1. Loads a .json (array) or .jsonl (line-delimited) file where every record
   matches the trial schema (config, seed, budget, trialNo, execution_time,
   timestamp, val_error, best_so_far, epoch_history).
2. Encodes each trial's hyperparameter config into a normalized vector and
   turns its observed `epoch_history` into an ifbo.Curve (the "context").
3. For every trial, builds query points for the epochs it hasn't reached yet
   (up to its `budget`, optionally extended by --extra-epochs) and asks the
   FT-PFN surrogate for the Predictive Posterior Distribution (PPD) at those
   points, using every other trial's curve as in-context information -
   exactly the grey-box extrapolation ifBO uses inside its BO loop.
4. Writes out a copy of the input where every record has a new
   "predictions" key holding the predicted mean/median/mode/quantiles/
   variance/std plus acquisition-style scores (ucb, ei, pi).

INSTALL
-------
    pip install ifbo

The first run downloads the pretrained FT-PFN weights (a few hundred MB)
from figshare, so machines with restricted egress must allow that host
(or pre-populate the --model-path cache directory).

USAGE
-----
    python ifbo_predict.py --input trials.jsonl --output trials_with_predictions.jsonl

    # extrapolate 20 epochs past each trial's own budget, custom quantiles
    python ifbo_predict.py --input trials.json --extra-epochs 20 \
        --quantiles 0.1,0.5,0.9

NOTES / ASSUMPTIONS (documented because the schema doesn't fully pin these
down - adjust the constants below if your data means something different):
  * The curve metric modeled is `val_accuracy` from `epoch_history`
    (FT-PFN expects a "higher is better" signal normalized to [0, 1]).
    If your `val_accuracy` is on a 0-100 scale it is auto-detected and
    rescaled; predictions are converted back to your original scale.
  * Hyperparameter encoding is delegated to `hp_space.py` (HyperparameterSpace /
    Float / Integer / Categorical). It always excludes `model_type` and caps at
    hp_space.MAX_HYPERPARAMETERS=10 dims (keeping the first N by priority order
    if more are supplied). By default the bounds for each field (low/high,
    log-scale) are inferred from the min/max observed in your data; pass
    --hp-space-json to supply exact ConfigSpace-consistent bounds instead. The
    dims actually used are reported per-record under
    predictions["hyperparameter_dims_used"] and printed at start-up.
  * Each JSON record is treated as one independent curve (its own
    `epoch_history` = the observed portion of that curve). If your logs
    contain several snapshot rows for the *same* run, de-duplicate/merge
    them before running this script, otherwise the same run will appear
    multiple times in the context.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import torch

from automl.core.optimizers.ifbo.hp_space import (
    Categorical,
    Float,
    HyperparameterSpace,
    Integer,
    MAX_HYPERPARAMETERS,
)

# Fields present in `config` that are genuine hyperparameters. `model_type` is
# deliberately left out here: hp_space.HyperparameterSpace hard-drops it
# regardless (it treats it as a fixed architecture choice, not worth a dim).
INTEGER_FIELDS = {
    "batch_size",
    "hidden_dim",
    "max_seq_length",
    "seq_embed_dim",
    "seq_num_layers",
}
FLOAT_FIELDS = {"dropout", "learning_rate", "weight_decay"}
CATEGORICAL_FIELDS = ["optimizer", "scheduler"]

# Priority order: most-important hyperparameter first. Only matters if the
# space exceeds hp_space.MAX_HYPERPARAMETERS dims (it won't for this schema -
# 10 fields remain once `model_type` is dropped - but keeps behavior sane if
# extra fields are ever added to `config`).
PRIORITY_ORDER = [
    "learning_rate",
    "optimizer",
    "scheduler",
    "dropout",
    "weight_decay",
    "hidden_dim",
    "seq_num_layers",
    "seq_embed_dim",
    "batch_size",
    "max_seq_length",
]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_records(path: Path) -> tuple[list[dict], str]:
    """Load a .json array or a .jsonl file. Returns (records, format)."""
    text = path.read_text()
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"{path} is empty")

    if stripped[0] == "[":
        try:
            records = json.loads(stripped)
            if not isinstance(records, list):
                raise ValueError("Top-level JSON is not an array")
            return records, "json"
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse {path} as a JSON array: {e}")

    # Otherwise assume JSONL (one JSON object per line)
    records = []
    for i, line in enumerate(stripped.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse line {i + 1} of {path} as JSON: {e}")
    return records, "jsonl"


def save_records(records: list[dict], path: Path, fmt: str) -> None:
    if fmt == "jsonl":
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    else:
        with path.open("w") as f:
            json.dump(records, f, indent=2)


# --------------------------------------------------------------------------- #
# Hyperparameter encoding
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _infer_default_spec(field: str, records: list[dict], is_categorical: bool):
    """Best-effort HPSpec inferred from the observed data (used when
    --hp-space-json isn't supplied)."""
    if is_categorical:
        raw = [(r.get("config") or {}).get(field) for r in records]
        raw = ["<missing>" if v is None else str(v) for v in raw]
        choices = tuple(sorted(set(raw)))
        if not choices:
            return None
        return Categorical(choices)

    vals = [_to_float((r.get("config") or {}).get(field)) for r in records]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        pad = max(abs(lo) * 1e-6, 1e-9)
        lo, hi = lo - pad, hi + pad
    # Heuristic: log-scale if strictly positive and spans an order of
    # magnitude or more (typical for learning_rate/weight_decay/dims).
    log = lo > 0 and (hi / lo) >= 10.0
    cls = Integer if field in INTEGER_FIELDS else Float
    return cls(lo, hi, log=log)


def _space_from_json(path: str) -> HyperparameterSpace:
    """Build a HyperparameterSpace from an explicit user-supplied JSON spec, e.g.:
    {"learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": true},
     "optimizer": {"type": "categorical", "choices": ["adam", "sgd"]}, ...}
    Field order in the JSON = priority order for the 10-dim cap.
    """
    payload = json.loads(Path(path).read_text())
    specs = {}
    for field, spec in payload.items():
        t = str(spec.get("type", "float")).lower()
        if t == "categorical":
            specs[field] = Categorical(tuple(spec["choices"]))
        elif t in ("int", "integer"):
            specs[field] = Integer(
                spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
        else:
            specs[field] = Float(
                spec["low"], spec["high"], log=bool(spec.get("log", False))
            )
    return HyperparameterSpace(**specs)


def build_hyperparameter_space(
    records: list[dict], hp_space_json: str | None
) -> HyperparameterSpace:
    if hp_space_json:
        print(
            f"[ifbo_predict] Building hyperparameter space from {hp_space_json}",
            file=sys.stderr,
        )
        return _space_from_json(hp_space_json)

    specs = {}
    for field in PRIORITY_ORDER:
        is_cat = field in CATEGORICAL_FIELDS
        spec = _infer_default_spec(field, records, is_categorical=is_cat)
        if spec is not None:
            specs[field] = spec
    print(
        "[ifbo_predict] No --hp-space-json given; inferring hyperparameter bounds "
        "from the min/max observed in the data (pass --hp-space-json for exact "
        "ConfigSpace-consistent bounds instead).",
        file=sys.stderr,
    )
    return HyperparameterSpace(**specs)


def _config_for_encode(config: dict, space: HyperparameterSpace) -> dict:
    """Coerce categorical values to str (to match how choices were built) and
    null out any categorical value the space doesn't recognize (encoded as
    the neutral 0.5 by HyperparameterSpace.encode, same as a missing value)."""
    cfg = dict(config or {})
    for name in space.names:
        spec = space.specs[name]
        if name not in cfg or cfg[name] is None:
            continue
        if isinstance(spec, Categorical):
            cfg[name] = str(cfg[name])
            if cfg[name] not in spec.choices:
                print(
                    f"[ifbo_predict] Warning: value {cfg[name]!r} for '{name}' is not "
                    f"in the configured choices {spec.choices}; treating as missing.",
                    file=sys.stderr,
                )
                cfg[name] = None
    return cfg


# --------------------------------------------------------------------------- #
# Curve construction
# --------------------------------------------------------------------------- #
def detect_metric_scale(records: list[dict], metric: str) -> float:
    """Return 100.0 if the metric looks like a 0-100 scale, else 1.0."""
    vmax = 0.0
    for r in records:
        for pt in r.get("epoch_history") or []:
            v = _to_float(pt.get(metric))
            if v is not None:
                vmax = max(vmax, v)
    return 100.0 if vmax > 1.5 else 1.0


def compute_global_horizon(records: list[dict], extra_epochs: int) -> float:
    base = 1.0
    for r in records:
        budget = _to_float(r.get("budget")) or 0.0
        epochs = [
            _to_float(pt.get("epoch")) or 0.0 for pt in (r.get("epoch_history") or [])
        ]
        last_epoch = max(epochs) if epochs else 0.0
        base = max(base, budget, last_epoch)
    return base + max(extra_epochs, 0)


# --------------------------------------------------------------------------- #
# Tensor helpers
# --------------------------------------------------------------------------- #
def _squeeze_stat(t: torch.Tensor) -> torch.Tensor:
    """FT-PFN's own PredictionResult methods do `.squeeze(1)` on raw
    BarDistribution outputs before returning; mirror that for the stats
    (mean/median/mode/variance) we pull straight from the criterion."""
    if t.dim() >= 2 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def _to_list(t: torch.Tensor, scale: float = 1.0, ndigits: int = 6) -> list[float]:
    t = _squeeze_stat(t.detach()).flatten()
    return [round(float(x) * scale, ndigits) for x in t]


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> None:
    # Imports here so `--help` doesn't require torch/ifbo to be installed yet.
    from ifbo import Curve
    from ifbo.surrogate import FTPFN

    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_name(in_path.stem + "_predictions" + in_path.suffix)
    )

    records, fmt = load_records(in_path)
    if not records:
        raise ValueError("Input file contains no records")
    print(
        f"[ifbo_predict] Loaded {len(records)} records from {in_path} ({fmt})",
        file=sys.stderr,
    )

    space = build_hyperparameter_space(records, args.hp_space_json)
    hp_fields_used = space.names
    if space.dropped_names:
        print(
            f"[ifbo_predict] Dropped hyperparameter dim(s) beyond the "
            f"{MAX_HYPERPARAMETERS}-dim FT-PFN limit: {space.dropped_names}",
            file=sys.stderr,
        )
    print(
        f"[ifbo_predict] Using hyperparameter dims (priority order): {hp_fields_used} "
        f"- 'model_type' is always excluded per hp_space.py.",
        file=sys.stderr,
    )
    hp_matrix = [
        space.encode(_config_for_encode(r.get("config") or {}, space)) for r in records
    ]

    metric = args.metric
    scale = detect_metric_scale(records, metric)
    if scale != 1.0:
        print(
            f"[ifbo_predict] '{metric}' looks like a 0-100 scale; "
            f"normalizing to [0,1] for the model and rescaling predictions back.",
            file=sys.stderr,
        )

    global_T = compute_global_horizon(records, args.extra_epochs)
    print(
        f"[ifbo_predict] Global time horizon = {global_T} epochs "
        f"(t is normalized as epoch / {global_T})",
        file=sys.stderr,
    )

    # -- build one Curve per record for context, tracking which records have any data --
    context_curves = []
    context_index_for_record = [None] * len(records)  # record idx -> context list idx
    all_y_scaled: list[float] = []

    for i, r in enumerate(records):
        pts = sorted(
            (
                (_to_float(p.get("epoch")), _to_float(p.get(metric)))
                for p in (r.get("epoch_history") or [])
            ),
            key=lambda p: p[0] if p[0] is not None else -1,
        )
        pts = [(e, v) for e, v in pts if e is not None and v is not None]
        if not pts:
            continue
        t_vals = [e / global_T for e, _ in pts]
        y_vals = [max(0.0, min(1.0, v / scale)) for _, v in pts]
        all_y_scaled.extend(y_vals)
        context_curves.append(
            Curve(
                hyperparameters=hp_matrix[i],
                t=torch.tensor(t_vals, dtype=torch.float32),
                y=torch.tensor(y_vals, dtype=torch.float32),
            )
        )
        context_index_for_record[i] = len(context_curves) - 1

    if len(context_curves) > args.max_context_curves:
        print(
            f"[ifbo_predict] {len(context_curves)} curves exceed "
            f"--max-context-curves={args.max_context_curves}; truncating context "
            f"(all trials are still predicted, just with a smaller context set).",
            file=sys.stderr,
        )
        context_curves = context_curves[: args.max_context_curves]

    global_best_y = max(all_y_scaled) if all_y_scaled else 0.0

    # -- build query points (future epochs) per record --
    query_curves = []
    query_meta = []  # parallel list: dict with bookkeeping for each query curve
    for i, r in enumerate(records):
        pts = [
            (_to_float(p.get("epoch")), _to_float(p.get(metric)))
            for p in (r.get("epoch_history") or [])
        ]
        pts = [(e, v) for e, v in pts if e is not None]
        observed_epochs = sorted(int(e) for e, _ in pts)
        last_observed = observed_epochs[-1] if observed_epochs else 0

        budget = _to_float(r.get("budget")) or 0.0
        target = max(budget, last_observed)
        target += max(args.extra_epochs, 0)

        if target <= last_observed:
            query_meta.append(
                {
                    "record_idx": i,
                    "observed_epochs": observed_epochs,
                    "last_observed_epoch": last_observed,
                    "target_epoch": int(target),
                    "query_epochs": [],
                    "note": "No future epochs to predict (trial already at/above "
                    "its budget and --extra-epochs=0).",
                }
            )
            continue

        step = max(args.query_step, 1)
        future_epochs = list(range(last_observed + step, int(target) + 1, step))
        if not future_epochs:
            future_epochs = [int(target)]
        if len(future_epochs) > args.max_query_points:
            # keep it evenly spaced but capped
            idx = torch.linspace(0, len(future_epochs) - 1, args.max_query_points)
            future_epochs = sorted({future_epochs[int(round(j.item()))] for j in idx})

        t_query = torch.tensor(
            [e / global_T for e in future_epochs], dtype=torch.float32
        )
        query_curves.append(Curve(hyperparameters=hp_matrix[i], t=t_query))
        query_meta.append(
            {
                "record_idx": i,
                "observed_epochs": observed_epochs,
                "last_observed_epoch": last_observed,
                "target_epoch": int(target),
                "query_epochs": future_epochs,
                "note": None,
            }
        )

    n_active_queries = sum(1 for m in query_meta if m["query_epochs"])
    print(
        f"[ifbo_predict] Built {len(context_curves)} context curves and "
        f"{n_active_queries} query curves ({len(records) - n_active_queries} "
        f"record(s) have nothing left to predict).",
        file=sys.stderr,
    )

    # -- load surrogate & predict --
    device = None if args.device == "auto" else torch.device(args.device)
    print(
        f"[ifbo_predict] Loading FT-PFN (version={args.model_version})... "
        f"this downloads pretrained weights on first use.",
        file=sys.stderr,
    )
    try:
        model = FTPFN(
            target_path=args.model_path,
            version=args.model_version,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize the FT-PFN surrogate. This usually means the "
            "pretrained weights could not be downloaded (they come from figshare.com "
            "on first run) - check your network/proxy allowlist, or pre-populate "
            f"--model-path. Original error: {e}"
        ) from e

    # query_curves was only appended for records that still have future epochs to
    # predict, so it is already aligned with the subset of query_meta entries
    # whose "query_epochs" is non-empty.
    if query_curves:
        with torch.no_grad():
            predictions = model.predict(context=context_curves, query=query_curves)
    else:
        predictions = []

    quantile_levels = [float(q) for q in args.quantiles.split(",") if q.strip()]

    # -- assemble output --
    pred_by_record_idx: dict[int, dict] = {}

    for meta, pred in zip((m for m in query_meta if m["query_epochs"]), predictions):
        logits = pred.logits
        crit = pred.criterion
        entry = {
            "metric": metric,
            "observed_epochs": meta["observed_epochs"],
            "last_observed_epoch": meta["last_observed_epoch"],
            "target_epoch": meta["target_epoch"],
            "query_epochs": meta["query_epochs"],
            "mean": _to_list(crit.mean(logits), scale=scale),
            "median": _to_list(crit.median(logits), scale=scale),
            "mode": _to_list(crit.mode(logits), scale=scale),
            "variance": _to_list(crit.variance(logits), scale=scale * scale),
            "std": _to_list(
                torch.sqrt(_squeeze_stat(crit.variance(logits)).clamp(min=0)),
                scale=scale,
            ),
            "quantiles": {
                str(q): _to_list(pred.quantile(q), scale=scale) for q in quantile_levels
            },
            "ucb": _to_list(pred.ucb(), scale=scale),
            "ei": _to_list(pred.ei(torch.tensor(global_best_y)), scale=scale),
            "pi": _to_list(pred.pi(torch.tensor(global_best_y))),
            "context_size": len(context_curves),
            "hyperparameter_dims_used": hp_fields_used,
            "model_version": args.model_version,
        }
        pred_by_record_idx[meta["record_idx"]] = entry

    for meta in query_meta:
        if meta["record_idx"] in pred_by_record_idx:
            continue
        pred_by_record_idx[meta["record_idx"]] = {
            "metric": metric,
            "observed_epochs": meta["observed_epochs"],
            "last_observed_epoch": meta["last_observed_epoch"],
            "target_epoch": meta["target_epoch"],
            "query_epochs": [],
            "mean": [],
            "median": [],
            "mode": [],
            "variance": [],
            "std": [],
            "quantiles": {str(q): [] for q in quantile_levels},
            "ucb": [],
            "ei": [],
            "pi": [],
            "context_size": len(context_curves),
            "hyperparameter_dims_used": hp_fields_used,
            "model_version": args.model_version,
            "note": meta["note"],
        }

    out_records = []
    for i, r in enumerate(records):
        r_copy = dict(r)
        r_copy["predictions"] = pred_by_record_idx[i]
        out_records.append(r_copy)

    save_records(out_records, out_path, fmt)
    print(
        f"[ifbo_predict] Wrote {len(out_records)} records with predictions to {out_path}",
        file=sys.stderr,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate IFBO (FT-PFN) learning-curve predictions for HPO trial logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True, help="Path to input .json or .jsonl file"
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output file (default: <input>_predictions.<ext>)",
    )
    p.add_argument(
        "--hp-space-json",
        default=None,
        help="Path to a JSON file with explicit ConfigSpace-consistent bounds "
        "per hyperparameter, e.g. "
        '{"learning_rate": {"type": "float", "low": 1e-5, "high": 0.1, '
        '"log": true}, "optimizer": {"type": "categorical", "choices": '
        '["adam", "sgd"]}}. If omitted, bounds are inferred from the '
        "min/max observed in --input.",
    )
    p.add_argument(
        "--metric",
        default="val_accuracy",
        help="Field in epoch_history to model (default: val_accuracy). "
        "Must be a 'higher is better' signal.",
    )
    p.add_argument(
        "--extra-epochs",
        type=int,
        default=0,
        help="Extrapolate this many epochs beyond each trial's own budget "
        "(default: 0, i.e. predict up to budget only)",
    )
    p.add_argument(
        "--query-step",
        type=int,
        default=1,
        help="Epoch step size between predicted points (default: 1)",
    )
    p.add_argument(
        "--max-query-points",
        type=int,
        default=50,
        help="Cap on predicted points per trial (default: 50)",
    )
    p.add_argument(
        "--max-context-curves",
        type=int,
        default=1000,
        help="Cap on context curves, per FT-PFN v0.0.1's limit (default: 1000)",
    )
    p.add_argument(
        "--quantiles",
        default="0.05,0.25,0.5,0.75,0.95",
        help="Comma-separated quantile levels to report (default: 0.05,0.25,0.5,0.75,0.95)",
    )
    p.add_argument("--model-version", default="0.0.1", help="FT-PFN checkpoint version")
    p.add_argument(
        "--model-path",
        default=None,
        help="Directory to cache/load pretrained FT-PFN weights",
    )
    p.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="ifbo.*")
    run(args)


if __name__ == "__main__":
    main()
