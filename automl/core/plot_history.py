import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from automl.core.types import TrialResult
from automl.logger import get_logger

logger = get_logger()


def _get_model_colors(history: List[TrialResult]) -> Dict[str, Any]:
    """Finds unique model_types in the history and maps them to a matplotlib color."""
    unique_models = list(
        set([trial.get("config", {}).get("model_type", "Unknown") for trial in history])
    )
    cmap = plt.get_cmap("tab10")
    return {model: cmap(i % 10) for i, model in enumerate(unique_models)}


def _smooth_curve(points: List[float], weight: float = 0.8) -> List[float]:
    """
    Smooths a list of points using an Exponential Moving Average (EMA).
    Weight should be between 0 (no smoothing) and 1 (flat line).
    """
    if not points:
        return []

    smoothed = []
    last = points[0]
    for point in points:
        # EMA formula
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


# --- Function 1: Plot Learning Curves ---
def plot_learning_curves(
    history: List[TrialResult],
    save_path: Union[str, Path] = "learning_curves.png",
    smooth_weight: float = 0.8,  # Add a smoothing parameter (0.0 to disable)
) -> None:
    """
    Plots Training Loss and Validation Accuracy across epochs.
    Lines are color-coded based on the config['model_type'].
    Noisy raw data is plotted faintly, with a smoothed trendline on top.
    """
    save_path = Path(save_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    model_colors = _get_model_colors(history)

    for trial in history:
        model_type = trial.get("config", {}).get("model_type", "Unknown")
        color = model_colors[model_type]
        epoch_hist = trial.get("epoch_history", [])

        # --- Process Training Loss ---
        epochs_loss = [e["epoch"] for e in epoch_hist if e["train_loss"] is not None]
        train_losses: list[float] = [
            cast(float, e["train_loss"])
            for e in epoch_hist
            if e["train_loss"] is not None
        ]

        if epochs_loss and train_losses:
            # Plot raw noisy data faintly
            ax1.plot(epochs_loss, train_losses, color=color, alpha=0.2, linewidth=1.0)
            # Plot smoothed data prominently
            smoothed_loss = _smooth_curve(train_losses, weight=smooth_weight)
            ax1.plot(epochs_loss, smoothed_loss, color=color, alpha=0.9, linewidth=2.0)

        # --- Process Validation Accuracy ---
        epochs_acc = [e["epoch"] for e in epoch_hist if e["val_accuracy"] is not None]
        val_accuracies = [
            cast(float, e["val_accuracy"])
            for e in epoch_hist
            if e["val_accuracy"] is not None
        ]

        if epochs_acc and val_accuracies:
            # Plot raw noisy data faintly
            ax2.plot(epochs_acc, val_accuracies, color=color, alpha=0.2, linewidth=1.0)
            # Plot smoothed data prominently
            smoothed_acc = _smooth_curve(val_accuracies, weight=smooth_weight)
            ax2.plot(epochs_acc, smoothed_acc, color=color, alpha=0.9, linewidth=2.0)

    # --- Formatting & Legends ---
    legend_patches = [
        mpatches.Patch(color=color, label=f"Approach: {m_type}")
        for m_type, color in model_colors.items()
    ]

    ax1.set_title(f"Training Loss vs. Epoch (Smoothing: {smooth_weight})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(handles=legend_patches)

    ax2.set_title(f"Validation Accuracy vs. Epoch (Smoothing: {smooth_weight})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(handles=legend_patches)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # Assuming logger is defined elsewhere in your script
    # logger.debug(f"Learning curves saved to {save_path}")


# --- Function 2: Plot Optimization History ---
def plot_optimization_history(
    history: List[TrialResult],
    save_path: Union[str, Path] = "optimization_history.png",
) -> None:
    """
    Plots the final validation error for each trial.
    Scatter points and 'best' stars are color-coded by config['model_type'].
    """
    save_path = Path(save_path)
    sorted_history = sorted(history, key=lambda x: x.get("trialNo", 0))
    model_colors = _get_model_colors(sorted_history)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot individual trials grouped by model_type
    for m_type, color in model_colors.items():
        m_trials = [
            t
            for t in sorted_history
            if t.get("config", {}).get("model_type", "Unknown") == m_type
        ]

        m_x = [t.get("trialNo") for t in m_trials]
        m_y = [t.get("val_error") for t in m_trials]

        if m_x and m_y:
            ax.scatter(
                m_x, m_y, color=color, alpha=0.6, s=50, label=f"Approach: {m_type}"
            )

    # 2. Calculate and plot the "best so far" trajectory line
    val_errors = [trial.get("val_error") for trial in sorted_history]
    trial_nos = [trial.get("trialNo") for trial in sorted_history]

    best_errors = []
    current_best = float("inf")
    for err in val_errors:
        if err is not None and err < current_best:
            current_best = err
        best_errors.append(current_best)

    ax.plot(
        trial_nos,
        best_errors,
        color="black",
        linewidth=1.5,
        linestyle="--",
        zorder=1,
        label="Best Error Trajectory",
    )

    # 3. Plot specific trials flagged as 'best_so_far' with model-specific star colors
    best_trials = [t for t in sorted_history if t.get("best_so_far")]

    for t in best_trials:
        m_type = t.get("config", {}).get("model_type", "Unknown")
        color = model_colors.get(
            m_type, "gold"
        )  # Fallback to gold if something goes wrong

        ax.scatter(
            t.get("trialNo"),
            t.get("val_error"),
            color=color,
            edgecolor="black",
            marker="*",
            s=250,
            zorder=3,
        )

    # Add a colorless dummy star to the legend so we don't have multiple entries
    if best_trials:
        ax.scatter(
            [],
            [],
            color="white",
            edgecolor="black",
            marker="*",
            s=150,
            label="New Best Found",
        )

    ax.set_title("Optimization Search History by Approach")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Error")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.debug(f"Optimization history saved to {save_path}")


# --- Function 3: Plot Budget vs Performance ---
def plot_budget_vs_performance(
    runs: list[TrialResult],
    metric: str = "val_error",
    agg: str = "mean",
    save_path: Union[str, Path] = "budget_vs_performance.png",
):
    """
    Plots budget vs performance overall and grouped by model_type,
    then saves the figure to disk.

    Parameters
    ----------
    runs : list[dict]
        HPO result list matching schema.
    metric : str
        Performance metric (default: val_error).
    agg : str
        Aggregation: "min", "mean", "median".
    save_path : str | Path
        Output image file path.
    """

    save_path = Path(save_path)

    # -----------------------------
    # Flatten structure
    # -----------------------------
    df = pd.json_normalize(runs)  # type: ignore[arg-type]

    df["budget"] = pd.to_numeric(df["budget"])
    df[metric] = pd.to_numeric(df[metric])

    # -----------------------------
    # Aggregate overall
    # -----------------------------
    if agg == "min":
        overall = df.groupby("budget")[metric].min().reset_index()
    elif agg == "mean":
        overall = df.groupby("budget")[metric].mean().reset_index()
    elif agg == "median":
        overall = df.groupby("budget")[metric].median().reset_index()
    else:
        raise ValueError("agg must be one of: min, mean, median")

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure()

    plt.plot(
        overall["budget"],
        overall[metric],
        marker="o",
        label="overall",
    )

    # -----------------------------
    # Grouped by model_type
    # -----------------------------
    for model_type, sub in df.groupby("config.model_type"):
        if agg == "min":
            g = sub.groupby("budget")[metric].min()
        elif agg == "mean":
            g = sub.groupby("budget")[metric].mean()
        else:
            g = sub.groupby("budget")[metric].median()

        g = g.sort_index()

        plt.plot(
            g.index,
            g.values,
            linestyle="--",
            alpha=0.7,
            label=str(model_type),
        )

    plt.xlabel("Budget")
    plt.ylabel(metric)
    plt.title(f"Budget vs {metric} (overall + model_type)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # -----------------------------
    # Save instead of show
    # -----------------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return df, overall


# --- Function 4: Plot Epoch Heatmap ---
def plot_epoch_heatmap(
    runs: List[TrialResult],
    metric: str = "val_accuracy",
    save_path: Union[str, Path] = "epoch_heatmap.png",
    agg: str = "mean",
):
    """
    Creates a heatmap of (trialNo x epoch) for a given metric
    extracted from epoch_history and saves it to disk.

    Parameters
    ----------
    runs : list[TrialResult]
        HPO results following the provided schema.
    metric : str
        Metric inside epoch_history (e.g. val_accuracy, train_loss).
    save_path : str | Path
        Output image path.
    agg : str
        How to aggregate duplicate (trial, epoch) pairs: max/mean/last.
    """

    save_path = Path(save_path)

    rows = []

    for r in runs:
        trial = r["trialNo"]
        history = r.get("epoch_history", [])

        for h in history:
            # metric = "val_accuracy"
            rows.append(
                {
                    "trial": trial,
                    "epoch": h["epoch"],
                    metric: h[metric],
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No epoch_history data found.")

    # -----------------------------
    # Aggregate duplicates if needed
    # -----------------------------
    if agg == "max":
        df = df.groupby(["trial", "epoch"], as_index=False)[metric].max()
    elif agg == "mean":
        df = df.groupby(["trial", "epoch"], as_index=False)[metric].mean()
    elif agg == "last":
        df = (
            df.sort_values(["trial", "epoch"])
            .groupby(["trial", "epoch"], as_index=False)
            .last()
        )
    else:
        raise ValueError("agg must be one of: max, mean, last")

    # -----------------------------
    # Pivot to matrix form
    # -----------------------------
    mat = df.pivot(index="trial", columns="epoch", values=metric)

    # Sort axes for stability
    mat = mat.sort_index().sort_index(axis=1)

    # -----------------------------
    # Plot heatmap
    # -----------------------------
    plt.figure(figsize=(10, 6))

    plt.imshow(mat, aspect="auto", origin="lower", interpolation="nearest")

    plt.colorbar(label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Trial No")
    plt.title(f"Epoch Heatmap ({metric})")

    # tick control (avoid clutter)
    plt.xticks(
        ticks=range(0, len(mat.columns), max(1, len(mat.columns) // 10)),
        labels=mat.columns[:: max(1, len(mat.columns) // 10)],
        rotation=45,
    )

    plt.yticks(
        ticks=range(0, len(mat.index), max(1, len(mat.index) // 10)),
        labels=mat.index[:: max(1, len(mat.index) // 10)],
    )

    plt.tight_layout()

    # -----------------------------
    # Save
    # -----------------------------
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return mat


def _runs_to_dataframe(runs: list[TrialResult]) -> pd.DataFrame:
    """Flatten runs so that config.* becomes top-level columns with prefix 'config_'."""
    records = []
    for r in runs:
        base = {k: v for k, v in r.items() if k != "config"}
        cfg = r.get("config", {})
        flat_cfg = {f"config_{k}": v for k, v in cfg.items()}
        base.update(flat_cfg)
        records.append(base)
    return pd.DataFrame.from_records(records)


def plot_config_corr_overall(
    runs: list[TrialResult],
    metric: str = "val_error",
    save_path: Union[str, Path] = "config_corr_overall.png",
    *,
    annotate: bool = True,
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "coolwarm",
    dpi: int = 300,
) -> None:
    """
    Plot signed Pearson correlation between each numeric config parameter and `metric`.
    """
    df = _runs_to_dataframe(runs)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in runs DataFrame.")

    # numeric config_* columns only
    config_cols = [c for c in df.columns if c.startswith("config_")]
    if not config_cols:
        raise ValueError("No config_* columns found in runs data.")

    numeric_cols = [c for c in config_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric config_* columns found to compute correlation.")

    corr = df[numeric_cols + [metric]].corr(method="pearson")[metric].drop(metric)
    # sort by signed value (you could also sort by abs if you want strongest first)
    corr = corr.sort_values(ascending=False)

    plt.figure(figsize=figsize, dpi=dpi)
    # use a diverging palette to emphasize sign
    sns.barplot(
        x=corr.values,
        y=[c.replace("config_", "") for c in corr.index],
        palette=sns.color_palette(cmap, as_cmap=False),
    )

    plt.xlabel(f"Pearson corr with {metric}")
    plt.ylabel("config parameter")
    plt.title(f"Correlation of config parameters with {metric} (overall)")

    if annotate:
        for i, v in enumerate(corr.values):
            plt.text(
                v,
                i,
                f"{v:.2f}",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
            )

    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_config_corr_by_model_type(
    runs: list[TrialResult],
    metric: str = "val_error",
    save_path: Union[str, Path] = "config_corr_by_model_type.png",
    *,
    annotate: bool = True,
    figsize_per_model: tuple[float, float] = (8, 6),
    cmap: str = "coolwarm",
    dpi: int = 300,
    min_runs_per_model: int = 3,
) -> None:
    """
    Plot signed Pearson correlation between each numeric config parameter and `metric`,
    separately for each `config_model_type`.
    """
    df = _runs_to_dataframe(runs)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in runs DataFrame.")

    model_col = "config_model_type"
    if model_col not in df.columns:
        raise ValueError("`config.model_type` not found in runs data.")

    config_cols = [c for c in df.columns if c.startswith("config_")]
    numeric_cols = [
        c
        for c in config_cols
        if pd.api.types.is_numeric_dtype(df[c]) and c != model_col
    ]
    if not numeric_cols:
        raise ValueError(
            "No numeric config_* columns (besides model_type) to correlate."
        )

    model_types = (
        df[model_col]
        .value_counts()
        .loc[lambda s: s >= min_runs_per_model]
        .index.tolist()
    )
    if not model_types:
        raise ValueError(f"No model_type has at least {min_runs_per_model} runs.")

    n_models = len(model_types)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(figsize_per_model[0] * n_models, figsize_per_model[1]),
        dpi=dpi,
        squeeze=False,
    )
    axes = axes[0]

    for ax, mt in zip(axes, model_types):
        sub = df[df[model_col] == mt]
        corr = sub[numeric_cols + [metric]].corr(method="pearson")[metric].drop(metric)
        corr = corr.sort_values(ascending=False)

        sns.barplot(
            x=corr.values,
            y=[c.replace("config_", "") for c in corr.index],
            ax=ax,
            palette=sns.color_palette(cmap, as_cmap=False),
        )

        ax.set_title(f"model_type = {mt}")
        ax.set_xlabel(f"Pearson corr with {metric}")
        ax.set_ylabel("config parameter")

        if annotate:
            for i, v in enumerate(corr.values):
                ax.text(
                    v,
                    i,
                    f"{v:.2f}",
                    va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=7,
                )

        ax.axvline(0.0, color="black", linewidth=0.8)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_config_corr_heatmap_overall(
    runs: list[TrialResult],
    metric: str = "val_error",
    save_path: Union[str, Path] = "config_corr_heatmap_overall.png",
    *,
    figsize: tuple[float, float] = (6, 8),
    cmap: str = "coolwarm",
    dpi: int = 300,
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".2f",
) -> None:
    """
    Heatmap of signed Pearson correlation between each numeric config parameter and `metric`.
    Each cell is corr(config_param, metric).
    """
    df = _runs_to_dataframe(runs)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in runs DataFrame.")

    # numeric config_* columns only
    config_cols = [c for c in df.columns if c.startswith("config_")]
    if not config_cols:
        raise ValueError("No config_* columns found in runs data.")

    numeric_cols = [c for c in config_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric config_* columns found to compute correlation.")

    # Compute correlations: we’ll create a 2D frame (rows: config params, col: [metric])
    corr_series = (
        df[numeric_cols + [metric]].corr(method="pearson")[metric].drop(metric)
    )
    corr_df = corr_series.to_frame(name=metric)
    corr_df.index = [idx.replace("config_", "") for idx in corr_df.index]

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        corr_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0.0,
        annot=annot,
        fmt=fmt,
        cbar_kws={"label": f"Pearson corr with {metric}"},
    )
    plt.title(f"Correlation of config parameters with {metric} (overall)")
    plt.ylabel("config parameter")
    plt.xlabel("")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_config_corr_heatmap_by_model_type(
    runs: list[TrialResult],
    metric: str = "val_error",
    save_path: Union[str, Path] = "config_corr_heatmap_by_model_type.png",
    *,
    figsize_per_model: tuple[float, float] = (4, 8),
    cmap: str = "coolwarm",
    dpi: int = 300,
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
    fmt: str = ".2f",
    min_runs_per_model: int = 3,
) -> None:
    """
    Heatmap of signed Pearson correlation between numeric config parameters and `metric`,
    separately for each `config_model_type`.

    Produces a grid where:
    - rows: config parameters
    - columns: model types
    each cell is corr(config_param, metric) for that model_type.
    """
    df = _runs_to_dataframe(runs)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in runs DataFrame.")

    model_col = "config_model_type"
    if model_col not in df.columns:
        raise ValueError("`config.model_type` not found in runs data.")

    # numeric config_* columns except model_type
    config_cols = [c for c in df.columns if c.startswith("config_")]
    numeric_cols = [
        c
        for c in config_cols
        if pd.api.types.is_numeric_dtype(df[c]) and c != model_col
    ]
    if not numeric_cols:
        raise ValueError(
            "No numeric config_* columns (besides model_type) to correlate."
        )

    # Only keep model_types with enough runs
    valid_models = (
        df[model_col]
        .value_counts()
        .loc[lambda s: s >= min_runs_per_model]
        .index.tolist()
    )
    if not valid_models:
        raise ValueError(f"No model_type has at least {min_runs_per_model} runs.")

    # Build a matrix: rows = config params, cols = model_types
    corr_mats = {}
    for mt in valid_models:
        sub = df[df[model_col] == mt]
        # If degenerate after filtering, skip
        if len(sub) < min_runs_per_model:
            continue
        corr_series = (
            sub[numeric_cols + [metric]].corr(method="pearson")[metric].drop(metric)
        )
        corr_mats[mt] = corr_series

    if not corr_mats:
        raise ValueError("No correlations could be computed for any model_type.")

    corr_df = pd.DataFrame(corr_mats)
    corr_df.index = [idx.replace("config_", "") for idx in corr_df.index]
    corr_df = corr_df.sort_index(axis=0)

    # Keep consistent model_type ordering
    corr_df = corr_df[valid_models]

    # Figure size scales with number of model types
    figsize = (figsize_per_model[0] * len(valid_models), figsize_per_model[1])

    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        corr_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0.0,
        annot=annot,
        fmt=fmt,
        cbar_kws={"label": f"Pearson corr with {metric}"},
    )
    plt.title(f"Correlation of config parameters with {metric} by model_type")
    plt.ylabel("config parameter")
    plt.xlabel("model_type")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# --- Dummy Data Generator ---
def generate_dummy_history(num_trials: int = 40) -> List[Dict[str, Any]]:
    history = []
    approaches = ["SMAC3", "Hyperband", "Random Search"]
    current_best_error = float("inf")

    for trial_no in range(1, num_trials + 1):
        approach = random.choice(approaches)
        num_epochs = random.randint(15, 50)

        epoch_history = []
        start_loss = random.uniform(1.5, 3.0)
        start_acc = random.uniform(0.2, 0.5)

        # Generate realistic curves
        for epoch in range(1, num_epochs + 1):
            train_loss = start_loss * math.exp(-0.1 * epoch) + random.uniform(
                0.01, 0.05
            )
            val_accuracy = min(
                0.98, start_acc + 0.15 * math.log(epoch)
            ) + random.uniform(-0.02, 0.02)

            epoch_history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_accuracy}
            )

        # Base the val_error on the final accuracy with some noise
        final_acc = epoch_history[-1]["val_accuracy"]
        val_error = 1.0 - final_acc + random.uniform(0.01, 0.05)

        is_best_yet = False
        if val_error < current_best_error:
            current_best_error = val_error
            is_best_yet = True

        trial_dict = {
            "config": {
                "model_type": approach,
                "learning_rate": random.uniform(1e-4, 1e-1),
            },
            "seed": random.randint(1, 9999),
            "budget": num_epochs,
            "trialNo": trial_no,
            "execution_time": random.uniform(15.0, 120.0),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S,%f"),
            "val_error": val_error,
            "best_so_far": is_best_yet,
            "epoch_history": epoch_history,
        }
        history.append(trial_dict)

    return history


def load_history_log(log_file: str | Path) -> List[TrialResult]:
    log_file = Path(log_file)

    with open(log_file, "r") as f:
        if log_file.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

            # normalize to list
            if isinstance(data, list):
                return data
            else:
                return [data]


# --- Main Execution ---
def main():
    # logger.info("Generating dummy training history...")
    # history_data = generate_dummy_history(num_trials=40)
    history_data = load_history_log(
        "results/amazon/[Transformer] 20260708_134328_98b38ac7/history.log.jsonl"
    )

    Path("temp-plots").mkdir(exist_ok=True)

    logger.info("Generating learning curves...")
    plot_learning_curves(
        history_data, save_path="temp-plots/learning_curves.ignore.png"
    )

    logger.info("Generating optimization history...")
    plot_optimization_history(
        history_data, save_path="temp-plots/optimization_history.ignore.png"
    )

    logger.info("Generating budget vs performance plot...")
    plot_budget_vs_performance(
        history_data, save_path="temp-plots/budget_vs_performance.ignore.png"
    )

    logger.info("Generating epoch heatmap...")
    plot_epoch_heatmap(history_data, save_path="temp-plots/epoch_heatmap.ignore.png")

    logger.info("Generating correlation plot...")
    plot_config_corr_overall(
        history_data, save_path="temp-plots/correlation.ignore.png"
    )
    plot_config_corr_by_model_type(
        history_data, save_path="temp-plots/correlation_by_model.ignore.png"
    )
    plot_config_corr_heatmap_overall(
        history_data, save_path="temp-plots/correlation_heatmap.ignore.png"
    )
    plot_config_corr_heatmap_by_model_type(
        history_data, save_path="temp-plots/correlation_heatmap_by_model.ignore.png"
    )

    logger.info("All tasks complete. Images saved to your current directory.")


if __name__ == "__main__":
    main()
