#!/usr/bin/env python3
"""
Plot actual vs predicted Freeze-Thaw curves per config, with a subplot per trial.

Updates vs previous:
- Supports JSON and JSONL/NDJSON (auto-detected) for both results and predictions.
- If neither --pred-path nor --model-loader is provided, it will try to instantiate FTPFN()
  (you can control where to import it from via --ftpfm-import).

Results input: JSON array or JSONL where each item matches your schema (has config, trialNo, epoch_history[..].val_accuracy)

Prediction input (optional): JSON array or JSONL with items keyed by:
  - (config_id, trialNo) OR (config, trialNo)
and containing any of:
  - epoch_history_pred: [{epoch, val_accuracy}, ...]
  - predicted_epoch_history: [...]
  - epoch_history: [...]  (if your pred file mirrors schema)

If computing predictions via a model:
- The script will include the *observed prefix of the same trial* as a context curve, and query the whole epoch grid
  (vertical line marks the observed boundary).
- Context can additionally include prior/all_other trials (global or same_config scope).

Examples:
  # List configs
  python plot_ft_curves.py results.jsonl --list-configs

  # Plot one config and auto-create FTPFN()
  python plot_ft_curves.py results.jsonl --config-id 7c1d4f2a9b --show-only

  # Plot random 3 configs, save files
  python plot_ft_curves.py results.jsonl --config-id random --random-n 3 --out plots/ft.png

  # Use predictions file (jsonl)
  python plot_ft_curves.py results.jsonl --pred-path preds.jsonl --config-id all --out plots/pred.png

  # Use your own model loader
  python plot_ft_curves.py results.jsonl --model-loader mypkg:load_model --config-to-z mypkg:config_to_z --config-id all
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib
import io
import json
import math
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None


# -----------------------------
# Import helpers
# -----------------------------
def _import_from_path(path: str) -> Any:
    """Import 'module:attr' and return attr."""
    if ":" not in path:
        raise ValueError(f"Expected 'module:attr', got: {path}")
    mod_name, attr = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _try_import_first(paths: List[str]) -> Any:
    last_err = None
    for p in paths:
        try:
            return _import_from_path(p)
        except Exception as e:
            last_err = e
    raise ImportError(f"Failed to import any of: {paths}. Last error: {last_err}")


# -----------------------------
# JSON / JSONL loading
# -----------------------------
def _open_maybe_gzip(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _sniff_format(path: str) -> str:
    """
    Return 'jsonl' or 'json'.
    - If extension indicates jsonl/ndjson => jsonl
    - Else sniff first non-whitespace char: '[' or '{' => json; otherwise jsonl
    """
    lower = path.lower()
    if lower.endswith((".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")):
        return "jsonl"
    if lower.endswith((".json", ".json.gz")):
        return "json"

    with _open_maybe_gzip(path) as f:
        while True:
            ch = f.read(1)
            if ch == "":
                return "json"  # empty -> treat as json
            if not ch.isspace():
                if ch in "[{":
                    return "json"
                return "jsonl"


def load_items(path: str) -> List[Dict[str, Any]]:
    fmt = _sniff_format(path)
    items: List[Dict[str, Any]] = []
    with _open_maybe_gzip(path) as f:
        if fmt == "json":
            obj = json.load(f)
            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict):
                items = [obj]
            else:
                raise ValueError("JSON must be an array or object")
        else:
            for line_no, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {path}: {e}"
                    ) from e
                if not isinstance(obj, dict):
                    raise ValueError(f"JSONL line {line_no} must be an object")
                items.append(obj)
    return items


# -----------------------------
# Config id
# -----------------------------
def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _config_id_sha1(config: Dict[str, Any], n: int = 10) -> str:
    return hashlib.sha1(_canonical_json(config).encode("utf-8")).hexdigest()[:n]


def _config_id_pyhash(config: Dict[str, Any]) -> str:
    # mirrors hash(str(config)) but not stable unless PYTHONHASHSEED fixed
    return str(hash(str(config)))


def _match_id(requested: str, available_ids: List[str]) -> List[str]:
    if requested in available_ids:
        return [requested]
    hits = [cid for cid in available_ids if cid.startswith(requested)]
    return hits


# -----------------------------
# Data extraction
# -----------------------------
def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _extract_epoch_series(trial_obj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    hist = trial_obj.get("epoch_history", []) or []
    rows = []
    for r in hist:
        e = _safe_float(r.get("epoch"))
        a = _safe_float(r.get("val_accuracy"))
        if not (math.isnan(e) or math.isnan(a)):
            rows.append((e, a))
    if not rows:
        return np.array([]), np.array([])
    rows.sort(key=lambda x: x[0])
    return (
        np.array([x[0] for x in rows], dtype=np.float32),
        np.array([x[1] for x in rows], dtype=np.float32),
    )


def _infer_b_max(trials: List[Dict[str, Any]], b_max_arg: Optional[float]) -> float:
    if b_max_arg is not None:
        return float(b_max_arg)
    max_epoch = 0.0
    max_budget = 0.0
    for t in trials:
        epochs, _ = _extract_epoch_series(t)
        if epochs.size:
            max_epoch = max(max_epoch, float(np.nanmax(epochs)))
        max_budget = max(max_budget, _safe_float(t.get("budget", float("nan"))))
    if max_epoch > 0:
        return max_epoch
    if max_budget > 0 and not math.isnan(max_budget):
        return max_budget
    return 1.0


# -----------------------------
# Prediction file support
# -----------------------------
def _extract_pred_series(pred_obj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    for k in ("epoch_history_pred", "predicted_epoch_history", "epoch_history"):
        if k in pred_obj and isinstance(pred_obj[k], list):
            rows = []
            for r in pred_obj[k]:
                e = _safe_float(r.get("epoch"))
                a = _safe_float(r.get("val_accuracy", r.get("val_accuracy_pred")))
                if not (math.isnan(e) or math.isnan(a)):
                    rows.append((e, a))
            rows.sort(key=lambda x: x[0])
            if rows:
                return (
                    np.array([x[0] for x in rows], dtype=np.float32),
                    np.array([x[1] for x in rows], dtype=np.float32),
                )
    return np.array([]), np.array([])


def _index_predictions(
    pred_items: List[Dict[str, Any]],
    cfg_id_fn: Callable[[Dict[str, Any]], str],
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    idx: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for it in pred_items:
        if "config_id" in it:
            cid = str(it["config_id"])
        elif "config" in it:
            cid = cfg_id_fn(it["config"])
        else:
            raise ValueError("Prediction item must have 'config_id' or 'config'")
        if "trialNo" not in it:
            raise ValueError("Prediction item must have 'trialNo'")
        tn = int(float(it["trialNo"]))
        idx[(cid, tn)] = it
    return idx


# -----------------------------
# Model-side prediction support
# -----------------------------
@dataclasses.dataclass
class SimpleCurve:
    hyperparameters: Any
    t: Any
    y: Optional[Any] = None


def _prediction_mean_from_predobj(
    predobj: Any, expected_len: int
) -> Optional[np.ndarray]:
    """
    Try to extract a vector mean of length expected_len, else None.
    """
    # common attrs that might store mean
    for attr in ("mean", "mu", "loc", "y_mean", "pred_mean"):
        if hasattr(predobj, attr):
            v = getattr(predobj, attr)
            try:
                if callable(v):
                    v = v()
                if torch is not None and isinstance(v, torch.Tensor):
                    v = v.detach().cpu().reshape(-1).numpy()
                elif isinstance(v, np.ndarray):
                    v = v.reshape(-1)
                else:
                    v = np.array(v).reshape(-1)
                if v.size == expected_len:
                    return v.astype(np.float32)
            except Exception:
                pass
    return None


def _compute_predictions_with_model(
    model: Any,
    all_trials: List[Dict[str, Any]],
    target_trial: Dict[str, Any],
    config_to_z: Callable[[Dict[str, Any]], Any],
    CurveCls: Any,
    b_max: float,
    context_scope: str,  # global | same_config
    context_mode: str,  # prior | all_others
    order_key: str,  # trialNo | timestamp
    observed_frac: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Freeze-thaw style: include observed prefix of THIS trial as a context curve, then query times.
    Returns (epochs, y_pred, n_obs).
    """
    if torch is None:
        raise RuntimeError("torch is required to compute predictions with a model")

    epochs, y_true = _extract_epoch_series(target_trial)
    if epochs.size == 0:
        return epochs, np.array([]), 0

    n_obs = max(1, int(round(observed_frac * len(epochs))))
    n_obs = min(n_obs, len(epochs))

    def _key(t: Dict[str, Any]) -> Any:
        if order_key == "timestamp":
            return str(t.get("timestamp", ""))
        return float(t.get("trialNo", 0))

    # Choose pool for "other trials" context
    if context_scope == "same_config":
        cid = target_trial["_config_id"]
        pool = [t for t in all_trials if t.get("_config_id") == cid]
    else:
        pool = all_trials

    # Choose which of those to include (excluding the target itself)
    if context_mode == "prior":
        cutoff = _key(target_trial)
        other_trials = [t for t in pool if t is not target_trial and _key(t) < cutoff]
    else:
        other_trials = [t for t in pool if t is not target_trial]

    context_curves: List[Any] = []

    # Add other trial curves (full histories)
    for t in other_trials:
        e, a = _extract_epoch_series(t)
        if e.size == 0:
            continue
        z = config_to_z(t["config"])
        tt = (e / b_max).astype(np.float32)
        context_curves.append(
            CurveCls(
                hyperparameters=z,
                t=torch.tensor(tt, dtype=torch.float32, device=device),
                y=torch.tensor(
                    a.astype(np.float32), dtype=torch.float32, device=device
                ),
            )
        )

    # Add observed prefix of this target trial into context
    zt = config_to_z(target_trial["config"])
    t_obs = (epochs[:n_obs] / b_max).astype(np.float32)
    y_obs = y_true[:n_obs].astype(np.float32)
    context_curves.append(
        CurveCls(
            hyperparameters=zt,
            t=torch.tensor(t_obs, dtype=torch.float32, device=device),
            y=torch.tensor(y_obs, dtype=torch.float32, device=device),
        )
    )

    # Query the full epoch grid
    tq = (epochs / b_max).astype(np.float32)
    query_curve = CurveCls(
        hyperparameters=zt,
        t=torch.tensor(tq, dtype=torch.float32, device=device),
    )

    preds = model.predict(context=context_curves, query=[query_curve])
    pred0 = preds[0]

    vec = _prediction_mean_from_predobj(pred0, expected_len=len(epochs))
    if vec is not None:
        return epochs, vec, n_obs

    # Fallback: pointwise queries
    y_pred_list: List[float] = []
    for ti in tq:
        q = CurveCls(
            hyperparameters=zt, t=torch.tensor([ti], dtype=torch.float32, device=device)
        )
        p = model.predict(context=context_curves, query=[q])[0]
        # scalar extraction
        mu = None
        if hasattr(p, "mean"):
            try:
                m = p.mean() if callable(p.mean) else p.mean
                if isinstance(m, torch.Tensor):
                    mu = float(m.detach().cpu().reshape(-1)[0].item())
                else:
                    mu = float(np.array(m).reshape(-1)[0])
            except Exception:
                mu = None
        y_pred_list.append(float("nan") if mu is None else mu)

    return epochs, np.array(y_pred_list, dtype=np.float32), n_obs


def _make_default_ftpfm(ftpfm_import: Optional[str], device: str) -> Any:
    """
    Try to instantiate FTPFN() if user didn't provide a model.
    You can override import location via --ftpfm-import, e.g.:
      --ftpfm-import ft_pfn:FTPFN
      --ftpfm-import yourpkg.models:FTPFN
    """
    if torch is None:
        raise RuntimeError("torch is required to instantiate and run FTPFN()")

    if ftpfm_import:
        FTPFN = _import_from_path(ftpfm_import)
    else:
        # common guesses; adjust via --ftpfm-import if needed
        FTPFN = _try_import_first(
            [
                "ft_pfn:FTPFN",
                "ftpfn:FTPFN",
                "ifbo:FTPFN",
                "ifbo.models:FTPFN",
                "ifbo_impl:FTPFN",
            ]
        )

    try:
        model = FTPFN()
    except TypeError:
        # some constructors might want device or other kwargs
        model = FTPFN(device=device)

    # Move / eval if available
    if hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:
            pass
    if hasattr(model, "eval"):
        try:
            model.eval()
        except Exception:
            pass
    return model


# -----------------------------
# Plotting
# -----------------------------
def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _plot_config_trials(
    config_id: str,
    trials_for_config: List[Dict[str, Any]],
    all_trials: List[Dict[str, Any]],
    out_path: Optional[str],
    show_only: bool,
    max_cols: int,
    fig_scale: float,
    pred_lookup: Optional[Dict[Tuple[str, int], Dict[str, Any]]],
    model: Optional[Any],
    config_to_z: Optional[Callable[[Dict[str, Any]], Any]],
    CurveCls: Any,
    b_max: float,
    context_scope: str,
    context_mode: str,
    order_key: str,
    observed_frac: float,
    device: str,
    title_extra: str = "",
) -> None:
    trials_for_config = sorted(
        trials_for_config, key=lambda t: float(t.get("trialNo", 0))
    )
    n = len(trials_for_config)
    cols = min(max_cols, max(n, 1))
    rows = int(math.ceil(n / cols)) if n else 1

    figsize = (fig_scale * 5.0 * cols, fig_scale * 3.6 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for i, trial in enumerate(trials_for_config):
        ax = axes_flat[i]
        trial_no = int(float(trial.get("trialNo", i)))
        seed = trial.get("seed", None)

        epochs, y_true = _extract_epoch_series(trial)
        if epochs.size == 0:
            ax.set_title(f"trialNo={trial_no} (empty)")
            ax.axis("off")
            continue

        ax.plot(epochs, y_true, lw=2.0, label="actual", color="C0")

        y_pred = None
        n_obs = 0

        # (A) Predictions from file
        if pred_lookup is not None:
            key = (config_id, trial_no)
            if key in pred_lookup:
                ep, yp = _extract_pred_series(pred_lookup[key])
                if ep.size and yp.size:
                    if not np.array_equal(ep, epochs):
                        yp = np.interp(epochs, ep, yp).astype(np.float32)
                    y_pred = yp

        # (B) Predictions from model
        elif model is not None:
            if config_to_z is None:
                raise ValueError(
                    "Model provided but config_to_z is None. Provide --config-to-z."
                )
            epochs2, y_pred2, n_obs = _compute_predictions_with_model(
                model=model,
                all_trials=all_trials,
                target_trial=trial,
                config_to_z=config_to_z,
                CurveCls=CurveCls,
                b_max=b_max,
                context_scope=context_scope,
                context_mode=context_mode,
                order_key=order_key,
                observed_frac=observed_frac,
                device=device,
            )
            if epochs2.size:
                y_pred = y_pred2

        if y_pred is not None:
            ax.plot(epochs, y_pred, lw=2.0, ls="--", label="predicted", color="C1")
            if n_obs > 0:
                ax.axvline(epochs[n_obs - 1], color="k", alpha=0.15, lw=1.5)
                ax.scatter(
                    epochs[:n_obs],
                    y_true[:n_obs],
                    s=18,
                    color="C0",
                    alpha=0.8,
                    label="observed",
                )
        else:
            ax.text(
                0.02,
                0.98,
                "no predictions",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                alpha=0.8,
            )

        ax.set_title(f"config={config_id} | trialNo={trial_no} | seed={seed}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_accuracy")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    suptitle = f"Actual vs Predicted Freeze-Thaw Curves | config_id={config_id}"
    if title_extra:
        suptitle += f" | {title_extra}"
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if show_only:
        plt.show()
    else:
        if out_path is None:
            out_path = f"plot_{config_id}.png"
        _ensure_dir_for_file(out_path)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[saved] {out_path}")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    p = argparse.ArgumentParser()

    p.add_argument(
        "results_path", help="Results file (.json/.jsonl/.ndjson; .gz supported)"
    )
    p.add_argument(
        "--pred-path",
        default=None,
        help="Optional predictions file (.json/.jsonl; .gz supported)",
    )

    p.add_argument("--config-id", default="all", help="Config id: <id>|random|all")
    p.add_argument(
        "--id-mode",
        default="sha1",
        choices=["sha1", "pyhash"],
        help="Config id function. 'pyhash' mirrors hash(str(config)) but is not stable unless PYTHONHASHSEED fixed.",
    )
    p.add_argument(
        "--list-configs",
        action="store_true",
        help="Print available config ids and exit",
    )
    p.add_argument("--random-n", type=int, default=1)
    p.add_argument("--random-seed", type=int, default=0)

    # Model options
    p.add_argument(
        "--model-loader",
        default=None,
        help="Import path 'module:callable' -> model with .predict(context, query)",
    )
    p.add_argument(
        "--config-to-z",
        default=None,
        help="Import path 'module:callable' -> torch tensor encoding for config",
    )
    p.add_argument(
        "--curve-class",
        default=None,
        help="Import path 'module:CurveClass'. If omitted, tries to import Curve; else uses SimpleCurve.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--ftpfm-import",
        default=None,
        help="Where to import FTPFN from, e.g. 'ft_pfn:FTPFN'. Used if no pred-path and no model-loader.",
    )

    # Context / FT plotting behavior
    p.add_argument(
        "--context-scope",
        default="global",
        choices=["global", "same_config"],
        help="Which trial pool to use for context curves when predicting.",
    )
    p.add_argument(
        "--context-mode",
        default="prior",
        choices=["prior", "all_others"],
        help="Which trials from the pool are included as context.",
    )
    p.add_argument("--order-key", default="trialNo", choices=["trialNo", "timestamp"])
    p.add_argument(
        "--observed-frac",
        type=float,
        default=0.3,
        help="Prefix fraction treated as observed for the same-trial context curve (vertical line).",
    )

    p.add_argument(
        "--b-max",
        type=float,
        default=None,
        help="Override b_max for t=epoch/b_max; else inferred.",
    )

    # Plot options
    p.add_argument("--max-cols", type=int, default=3)
    p.add_argument("--fig-scale", type=float, default=1.0)
    p.add_argument(
        "--out",
        default=None,
        help="Output file path. If plotting multiple configs, used as prefix: out_<config>.png (or root_<config>.ext).",
    )
    p.add_argument("--show-only", action="store_true")

    args = p.parse_args()

    # Load results
    trials = load_items(args.results_path)
    if not isinstance(trials, list) or not all(isinstance(x, dict) for x in trials):
        raise ValueError("Results must be a list of objects")

    # Config id function
    cfg_id_fn = _config_id_sha1 if args.id_mode == "sha1" else _config_id_pyhash

    # Attach computed ids
    for t in trials:
        if "config" not in t:
            raise ValueError("Each trial must have 'config'")
        t["_config_id"] = cfg_id_fn(t["config"])

    # Group
    by_cfg: Dict[str, List[Dict[str, Any]]] = {}
    for t in trials:
        by_cfg.setdefault(t["_config_id"], []).append(t)
    available_ids = sorted(by_cfg.keys())

    if args.list_configs:
        print("Available config ids:")
        for cid in available_ids:
            print(f"  {cid}  (trials={len(by_cfg[cid])})")
        if args.id_mode == "pyhash":
            print(
                "\nNote: pyhash ids depend on PYTHONHASHSEED; set it (e.g., PYTHONHASHSEED=0) to reproduce."
            )
        return 0

    # Select configs
    rng = random.Random(args.random_seed)
    if args.config_id == "all":
        selected_ids = available_ids
    elif args.config_id == "random":
        if not available_ids:
            raise ValueError("No configs found")
        selected_ids = rng.sample(
            available_ids, k=min(args.random_n, len(available_ids))
        )
    else:
        hits = _match_id(args.config_id, available_ids)
        if not hits:
            raise ValueError(
                f"No config id matching '{args.config_id}'. Use --list-configs."
            )
        if len(hits) > 1:
            raise ValueError(f"Ambiguous id prefix '{args.config_id}' matches: {hits}")
        selected_ids = hits

    # Load predictions OR build model
    pred_lookup = None
    model = None
    config_to_z = None

    # Curve class: user provided, else try import Curve, else SimpleCurve
    CurveCls = SimpleCurve
    if args.curve_class:
        CurveCls = _import_from_path(args.curve_class)
    else:
        # best-effort attempt to find a Curve type if your model requires it
        try:
            CurveCls = _try_import_first(
                [
                    "ft_pfn:Curve",
                    "ft_pfn.curve:Curve",
                    "ifbo:Curve",
                    "ifbo.types:Curve",
                ]
            )
        except Exception:
            CurveCls = SimpleCurve

    if args.pred_path:
        pred_items = load_items(args.pred_path)
        pred_lookup = _index_predictions(pred_items, cfg_id_fn=cfg_id_fn)
    else:
        # Model path:
        if args.model_loader:
            if torch is None:
                raise RuntimeError("torch is required for model prediction")
            load_model = _import_from_path(args.model_loader)
            model = load_model()
        else:
            # Default: FTPFN()
            model = _make_default_ftpfm(args.ftpfm_import, device=args.device)

        # config_to_z:
        if args.config_to_z:
            config_to_z = _import_from_path(args.config_to_z)
        else:
            # Generic fallback; may not match your trained encoding.
            # Keep it minimal (numeric + one-hot for categoricals present in your schema).
            if torch is None:
                raise RuntimeError("torch is required for model prediction")

            numeric_keys = [
                "batch_size",
                "dropout",
                "hidden_dim",
                "learning_rate",
                "max_seq_length",
                "seq_embed_dim",
                "seq_num_layers",
                "weight_decay",
            ]
            categorical_keys = ["model_type", "optimizer", "scheduler"]

            # build category maps from observed configs
            configs = [t["config"] for t in trials]
            cat_maps: Dict[str, Dict[str, int]] = {}
            for k in categorical_keys:
                values = sorted({str(c.get(k)) for c in configs})
                cat_maps[k] = {v: i for i, v in enumerate(values)}

            def config_to_z(cfg: Dict[str, Any]) -> Any:
                feats: List[float] = []
                for k in numeric_keys:
                    feats.append(_safe_float(cfg.get(k)))
                for k in categorical_keys:
                    mapping = cat_maps[k]
                    onehot = [0.0] * len(mapping)
                    v = str(cfg.get(k))
                    if v in mapping:
                        onehot[mapping[v]] = 1.0
                    feats.extend(onehot)
                z = np.array(feats, dtype=np.float32)
                return torch.tensor(z, dtype=torch.float32, device=args.device)

            print(
                "[warn] --config-to-z not provided; using generic numeric+onehot encoding derived from the results file."
            )

        # best-effort: move/eval
        if hasattr(model, "to"):
            try:
                model = model.to(args.device)
            except Exception:
                pass
        if hasattr(model, "eval"):
            try:
                model.eval()
            except Exception:
                pass

    b_max = _infer_b_max(trials, args.b_max)
    title_extra = f"id_mode={args.id_mode}, b_max={b_max:g}, ctx={args.context_scope}/{args.context_mode}/{args.order_key}"

    for cid in selected_ids:
        out_path = args.out
        if not args.show_only:
            if out_path is None:
                out_path = f"plot_{cid}.png"
            elif args.config_id == "all" or args.config_id == "random":
                root, ext = os.path.splitext(out_path)
                if ext.lower() in (".png", ".pdf", ".svg"):
                    out_path = f"{root}_{cid}{ext}"
                else:
                    out_path = f"{out_path}_{cid}.png"

        _plot_config_trials(
            config_id=cid,
            trials_for_config=by_cfg[cid],
            all_trials=trials,
            out_path=out_path,
            show_only=args.show_only,
            max_cols=args.max_cols,
            fig_scale=args.fig_scale,
            pred_lookup=pred_lookup,
            model=model,
            config_to_z=config_to_z,
            CurveCls=CurveCls,
            b_max=b_max,
            context_scope=args.context_scope,
            context_mode=args.context_mode,
            order_key=args.order_key,
            observed_frac=args.observed_frac,
            device=args.device,
            title_extra=title_extra,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
