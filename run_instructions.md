# Run Instructions

## Requirements

- `python = ">=3.10,<3.13"`

## Setup

```bash
make uninstall-existing-torch
make install
make init          # downloads datasets, tokenizers, and FT-PFN weights
```

## Step 1: Search for a tuned configuration (yelp)

```bash
python -m automl --config runconfig.yml --seed 42 --runtime-id <run_id>
```

`runconfig.yml` already targets `dataset: yelp` with `optimizer: ifbo`. This
runs within the 24h budget (governed by `n_trials` / `max_trial_time_seconds`
/ `evaluation_budget` in `runconfig.yml`) and writes:

- `results/yelp/<run_id>/history.log.jsonl` — every trial's config and
  learning curve (the "(i) hyperparameter configuration" artifact)
- `results/yelp/<run_id>/checkpoints/` — per-config checkpoints keyed by a
  hash of the config, enabling ifBO's freeze-thaw resume

This command is safely interruptible: re-running it with the same
`--runtime-id` resumes partially-trained configs from their checkpoints
instead of restarting from scratch (the "(ii) partially trained model" case).

## Step 2: Retrain the top-k configs on full data and generate predictions

```bash
python scripts/train_top_k_from_history.py \
  --history results/yelp/<run_id>/history.log.jsonl \
  --dataset yelp \
  --top-k 5 \
  --epochs 50 \
  --output-dir results_final/yelp/<run_id>
```

Retrains the top-k distinct configs (ranked by validation accuracy in the
history file) on the full yelp training split and majority-votes their test
predictions, writing `results_final/yelp/<run_id>/predictions.npy`. This step
is resumable too: re-running with the same `--output-dir` skips
already-trained incumbents via its `manifest.json`.

## Produce the submission file

```bash
cp results_final/yelp/<run_id>/predictions.npy final_test_preds.npy
```
