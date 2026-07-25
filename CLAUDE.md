# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An AutoML system for text classification (SS26 AutoML exam, Freiburg). Given one of
5 datasets (`ag_news`, `imdb`, `amazon`, `dbpedia` — practice; `yelp` — held-out
exam/test set), it runs a hyperparameter-optimization search over model configs and
produces a `predictions.npy` for the test split. See `docs/PROJECT_OVERVIEW.md` for a
full technical deep-dive, `docs/IFBO_METHOD.md` for the flagship optimizer's algorithm,
and `docs/PARALLELISM.md` for the concurrency model — read those before making
non-trivial changes to optimizers or the training loop.

## Commands

```bash
# Setup
pip install -e .                    # or: poetry install
make init                           # download datasets + tokenizers + FT-PFN weights (one-shot bootstrap)

# Run an HPO search
python -m automl --config runconfig.yml       # uses runconfig.yml (checked in, targets yelp+ifbo)
make run DATASET=amazon ARGS="--seed 42"      # shorthand; DATASET must be one of amazon/ag_news/imdb/dbpedia
make run-amazon ARGS="--seed 42"              # per-dataset shorthand target
make run-all ARGS="--seed 42"                 # loop over amazon/ag_news/imdb/dbpedia sequentially

# Produce the final submission artifact from a saved search
python train_top_k_from_history.py --history results/<dataset>/<runtime_id>/history.log.jsonl \
    --dataset <dataset> --top-k <k> --epochs <full_training_epochs> --output-dir <out_dir>
# -> writes predictions.npy (majority-vote ensemble of the retrained top-k configs)

# Formatting
make check                          # black --check .
make fix                            # black .

# Tests
make test                           # runs pytest — NOTE: no test files currently exist in this repo
```

There is no `run.py` on this branch despite `README.md` referencing it as the main
entrypoint — that script only exists on `main` (pre-refactor). The real entrypoint is
`python -m automl` (`automl/__main__.py`).

Config precedence for any run: `DEFAULT_CONFIG` (`automl/cli/argparser.py`) → YAML
file (`--config`, defaults to `runconfig.yml`) → CLI flags. `runconfig.local.yml` is a
gitignored personal-scale override some contributors keep locally.

## Architecture

Three pluggable layers, wired together by a `RuntimeConfig` (a `TypedDict`,
`automl/cli/types.py`):

```
Optimizer   (automl/core/optimizers/*)         decides WHICH config to try next, and for how long
     -> Approach  (automl/core/approaches/*)   turns a sampled Configuration into model + data pipeline
          -> Trainer (automl/core/trainers/*)  generic PyTorch train/eval loop
```

- **Optimizers** (`--optimizer`): `random` (`baselines/random.py`, no multi-fidelity,
  baseline), `smac` (`baselines/smac.py`, SMAC3 + Hyperband), `rl_freeze_thaw`
  (`baselines/rl_freeze_thaw.py`, from-scratch PPO scheduler), `ifbo`
  (`ifbo/optimizer.py`, **the flagship strategy** — in-context freeze-thaw BO using
  the pretrained FT-PFN surrogate; only optimizer that supports
  `num_parallel_trials > 1`). All registered in `automl/core/optimizers/__init__.py`
  and selected by name in `automl/__main__.py:main()`.
- **Approaches** (`--approach`): `sequence-dl` (BiLSTM over a DistilBERT WordPiece
  vocab, trained from scratch with an SVD-projected pretrained-embedding warm start)
  and `transformer` (a pretrained encoder fine-tuned end-to-end, optionally with
  `freeze_base` for linear-probing). Registered via `@register_approach("name")`
  (`automl/core/registry.py`) and auto-discovered at startup by
  `register_all_approaches()`, which imports every submodule under
  `automl/core/approaches/`. Shared tokenization/caching/dataset code for both
  approaches lives in `automl/core/approaches/text_encoding.py`.
- **Trainer**: `TorchTrainer` (`automl/core/trainers/torch_trainer.py`) — the one
  generic supervised-training loop every approach delegates to (optimizer/scheduler
  construction, stochastic epochs, linear LR warmup, checkpoint save/resume).

Each hyperparameter search space is defined per-approach in
`automl/core/configspacehelper.py:build_config_space(fixed_model_type=...)` — adding a
new approach means adding a branch there too, not just an `approaches/` module.

### Data pipeline (`automl/core/datasets.py`)

`BaseTextDataset` subclasses load `data_path/<dataset>/{train,test}.csv` once
(cached, always returns copies). `create_dataloaders(...)` does stratified
subsampling to `max_num_rows` (bounds per-trial HPO cost independent of raw dataset
size — `dbpedia`/`yelp` are 500k+ rows), then a stratified train/val split.

### Why the caching machinery in `text_encoding.py` matters

HPO runs hundreds of trials over the same fixed text pool, so: tokenizer instances
are **thread-local** (a fast HF tokenizer mutates in-place per call and isn't safe
to share across concurrently-running trial threads under `ifbo`'s
`num_parallel_trials`), and full-text tokenization is cached process-wide keyed by
`tokenizer_path -> {text: input_ids}` so each text is tokenized once regardless of
how many trials resample it — trial-specific truncation to `max_seq_length` is
applied lazily on top of the cached full ids. If you touch tokenization, preserve
this once-per-text-per-tokenizer invariant or HPO wall-clock cost regresses sharply.

### Optimizer-level shared infra (`automl/core/optimizers/base_optimizer.py`)

`train_single_configuration` is the one trial-execution routine every optimizer
calls. Configuration checkpoints are keyed by `sha256(sorted config dict)`
(`_config_to_hash_id`) under `checkpoints/trainers/<hash>/` — this is what lets
SMAC/Hyperband intensification and ifBO's freeze-thaw "thaw" (resume) a
partially-trained config instead of restarting from epoch 0. `_finalize_optimization`
handles both a single incumbent and a `list[Configuration]` (ensemble) uniformly,
majority-voting predictions when an optimizer returns a list — currently only `ifbo`
does. History is appended to `results/<dataset>/<runtime_id>/history.log.jsonl`
(line-delimited, `FileLock`-guarded for concurrent writers).

### Concurrency gotcha

Under `num_parallel_trials > 1`, `DataLoader(num_workers>0)` forking while another
thread holds the history `FileLock` can deadlock/crash (`filelock` refuses forks
mid-acquire on Python 3.12+). Effective `num_workers` is forced to `0` whenever
`parallelism > 1` — tokenization is already cached upfront so this costs little. See
`docs/PARALLELISM.md` before changing anything about `num_parallel_trials` or worker
counts.

### Producing the final submission

`train_top_k_from_history.py` is the script that actually produces
`predictions.npy` for submission: reads a saved `history.log.jsonl`, dedupes/ranks
trials, retrains the top-k distinct configs on full training data, and
majority-votes their test predictions. It mirrors `_finalize_optimization`'s
protocol standalone and is resumable (`--output-dir` reuse skips already-trained
incumbents via `manifest.json`). To submit a test score, the resulting
`predictions.npy` must be copied to `data/exam_dataset/predictions.npy` and pushed
to the `test` branch (see `README.md`'s auto-evaluation section) — that branch has
an unrelated git history from `main`/`dev-bilstm-only`, so moving the file across
requires `git checkout main -- data/exam_dataset/predictions.npy` from within the
`test` branch, not a merge.
