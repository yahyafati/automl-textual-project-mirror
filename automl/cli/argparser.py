import argparse
import datetime
import uuid
from pathlib import Path
from typing import Optional, Any

import yaml

from automl.cli.types import RuntimeConfig

DEFAULT_CONFIG: RuntimeConfig = RuntimeConfig(
    runtime_id="",
    device="auto",
    dataset="amazon",
    output_path=Path("results"),
    load_path=None,
    data_path=Path("data"),
    seed=int(datetime.datetime.now().timestamp() % 1e6),
    approach="sequence-dl",
    evaluation_budget=5,
    max_budget=40,
    min_budget=5,
    n_trials=10,
    max_trial_time_seconds=None,
    enable_jsonl_history=True,
    evaluate_incumbent=True,
    max_num_rows=40_000,
    val_size=0.2,
    optimizer="ifbo",
    num_workers=2,
    num_parallel_trials=1,
    stochastic_epochs=False,
    stochastic_epoch_fraction=0.25,
    log_level="INFO",
    ifbo_use_random_selection=False,
    ifbo_greedy_candidate_selection=False,
    ifbo_incumbent_ensemble_top_k=3,
    ifbo_incumbent_ensemble_accuracy_threshold=0.05,
    ifbo_thaw_step=1,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, default="runconfig.yml")

    parser.add_argument("--runtime-id", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ag_news", "imdb", "amazon", "dbpedia", "yelp"],
    )
    parser.add_argument("--device", type=str)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--load-path", type=Path)
    parser.add_argument("--data-path", type=Path)

    parser.add_argument("--seed", type=int)

    parser.add_argument(
        "--approach",
        type=str,
        choices=[
            "transformer",
            "sequence-dl",
        ],
    )

    parser.add_argument("--evaluation-budget", type=int)
    parser.add_argument("--max-budget", type=int)
    parser.add_argument("--min-budget", type=int)
    parser.add_argument("--n-trials", type=int)
    parser.add_argument(
        "--max-trial-time-seconds",
        type=float,
        help="Wall-clock cap (in seconds) on a single trial's training call. "
        "Checked once per epoch boundary. Default: None (no cap).",
    )
    parser.add_argument("--max-num-rows", type=int)
    parser.add_argument("--val-size", type=float)

    # parser.add_argument("--data-fraction", type=float)

    parser.add_argument(
        "--enable-jsonl-history",
        action="store_true",
        help="Enable writing JSONL history (overrides config to True)",
    )
    parser.add_argument(
        "--disable-jsonl-history",
        action="store_true",
        help="Disable writing JSONL history (overrides config to False)",
    )

    parser.add_argument(
        "--no-evaluate-incumbent",
        action="store_true",
        help="Skip retraining/evaluating the incumbent on held-out test data "
        "after optimization finishes (no predictions.npy / incumbent.json "
        "produced). Default: incumbent evaluation is enabled.",
    )

    parser.add_argument(
        "--ifbo-use-random-selection",
        action="store_true",
        help="Use random selection for ifBO candidates.",
    )

    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--stochastic-epochs",
        action="store_true",
        default=None,
        help="Sample a random fraction of the training batches each epoch "
        "instead of iterating the full dataset (see "
        "--stochastic-epoch-fraction). Default: False (full epochs).",
    )
    parser.add_argument(
        "--stochastic-epoch-fraction",
        type=float,
        help="Fraction of training batches to draw per epoch when "
        "--stochastic-epochs is set, in (0, 1]. Default: 0.25.",
    )
    parser.add_argument(
        "--num-parallel-trials",
        type=int,
        help="Number of ifBO trials to run concurrently, one per visible "
        "GPU (or round-robin across GPUs if this exceeds the device "
        "count). Default: 1 (sequential, today's behavior).",
    )
    parser.add_argument(
        "--optimizer", choices=["smac", "random", "ifbo", "rl_freeze_thaw"]
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--ifbo-incumbent-ensemble-top-k",
        type=int,
        help="Maximum number of near-best ifBO incumbents to ensemble.",
    )
    parser.add_argument(
        "--ifbo-incumbent-ensemble-accuracy-threshold",
        type=float,
        help=(
            "Absolute validation-accuracy tolerance for including an ifBO "
            "incumbent in the final ensemble."
        ),
    )
    parser.add_argument(
        "--ifbo-thaw-step",
        type=int,
        help="Number of steps to thaw each candidate in ifBO.",
    )

    return parser


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_config(
    yaml_cfg: Optional[dict[str, Any]] = None,
    cli_args: Optional[argparse.Namespace] = None,
    defaults: Optional[RuntimeConfig] = None,
) -> RuntimeConfig:
    if defaults is None:
        defaults = DEFAULT_CONFIG
    cfg = defaults.copy()

    # 1. apply YAML
    if yaml_cfg is not None:
        cfg.update({k: v for k, v in yaml_cfg.items() if v is not None})

    # 2. apply CLI overrides (only non-None / used flags)
    if cli_args is not None:
        cli_dict = vars(cli_args)

        # In case the key are named different in the RuntimeConfigDict and argparse
        key_map = {}

        # handle the mutually-exclusive history flags
        if cli_dict.get("enable_jsonl_history"):
            cfg["enable_jsonl_history"] = True
        if cli_dict.get("disable_jsonl_history"):
            cfg["enable_jsonl_history"] = False

        if cli_dict.get("no_evaluate_incumbent"):
            cfg["evaluate_incumbent"] = False

        for k, v in cli_dict.items():
            if k in (
                "config",
                "enable_jsonl_history",
                "disable_jsonl_history",
                "no_evaluate_incumbent",
            ):
                continue
            if v is None:
                continue

            cfg_key = key_map.get(k, k)
            cfg[cfg_key] = v

    return RuntimeConfig(**cfg)


def load_runtime_config(config_path: Optional[Path | str] = None):
    parser = create_parser()
    args = parser.parse_args()

    yaml_cfg = load_yaml(config_path or Path(args.config))
    cfg = merge_config(yaml_cfg, args, defaults=DEFAULT_CONFIG)

    if cfg["runtime_id"].strip() == "":
        cfg["runtime_id"] = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )

    cfg["output_path"] = Path(cfg["output_path"]) / cfg["dataset"] / cfg["runtime_id"]
    cfg["output_path"].mkdir(exist_ok=True, parents=True)

    cfg["data_path"] = Path(cfg["data_path"])
    load_path = cfg["load_path"]
    if load_path:
        cfg["load_path"] = Path(load_path)

    return cfg
