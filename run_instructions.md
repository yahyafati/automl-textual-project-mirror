# Run Instructions

## Requirements:

- `python = ">=3.10,<3.13"`

## Setup

```bash
make uninstall-existing-torch
make install
make init

make run DATASET=yelp
```

Or in one command: `make streamline`

## Train on the final dataset (yelp)

```bash
make run DATASET=yelp ARGS="--seed 42 --run-id <run_id>"
```

Trains within a 24h budget, saving config/checkpoint to
`results/yelp/<run_id>`.

## Generate predictions

Re-run the same command above — it picks up saved artifacts and writes
`results/yelp/<run_id>/predictions.npy`. Then:

```bash
cp results/yelp/<run_id>/predictions.npy final_test_preds.npy
```