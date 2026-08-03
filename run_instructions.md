# Run Instructions

## TL;DR

```bash
make uninstall-existing-torch && make install && make init

python -m automl --config runconfig.yml --seed 67 --runtime-id run_id

cp results/yelp/run_id/predictions.npy final_test_preds.npy
```

That's it for the default config. Everything below covers the details,
resuming, and the (only if needed) top-k retraining fallback.

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
python -m automl --config runconfig.yml --seed 67 --runtime-id <run_id>
```

`runconfig.yml` targets `dataset: yelp` with `optimizer: ifbo` and
`evaluate_incumbent: true`. This runs within the 24h budget (governed by
`n_trials` / `max_trial_time_seconds` / `evaluation_budget` in
`runconfig.yml`) and writes, under `results/yelp/<run_id>/`:

- `history.log.jsonl` — every trial's config and learning curve (the "(i)
  hyperparameter configuration" artifact)
- `checkpoints/` — per-config checkpoints keyed by a hash of the config,
  enabling ifBO's freeze-thaw resume
- `predictions.npy` — since `evaluate_incumbent: true`, the incumbent(s) are
  already retrained on the full training split and evaluated on the
  held-out test set here, so **no further step is needed**

Safely interruptible: re-running with the same `--runtime-id` resumes
partially-trained configs from their checkpoints instead of restarting from
scratch (the "(ii) partially trained model" case).

## Reproducing the submitted results for the other datasets

`yelp` is the only dataset graded on the held-out test set, and it's
reproduced via `runconfig.yml` above. The submitted results for the
remaining datasets (`ag_news`, `imdb`, `amazon`, `dbpedia`) were produced
with `runconfig.submitted.yml` instead — pass `--dataset` explicitly since
that config doesn't pin one:

```bash
python -m automl --config runconfig.submitted.yml --dataset <ag_news|imdb|amazon|dbpedia> --seed 67 --runtime-id <run_id>
```

This writes under `actual-results/<dataset>/<run_id>/` (`runconfig.submitted.yml`
sets `output_path: actual-results`), following the same
`history.log.jsonl` / `checkpoints/` / `predictions.npy` layout as Step 1.

## Step 2 — only if `evaluate_incumbent: false` was used

If Step 1 was run with `evaluate_incumbent: false` / `--no-evaluate-incumbent`
(no `predictions.npy` written yet), retrain the top-k configs on full data:

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
predictions, writing `results_final/yelp/<run_id>/predictions.npy`. Resumable:
re-running with the same `--output-dir` skips already-trained incumbents via
its `manifest.json`.

## Produce the submission file

```bash
# evaluate_incumbent: true (default) -> Step 1 already wrote predictions.npy
cp results/yelp/<run_id>/predictions.npy final_test_preds.npy

# evaluate_incumbent: false -> used Step 2 instead
cp results_final/yelp/<run_id>/predictions.npy final_test_preds.npy
```
