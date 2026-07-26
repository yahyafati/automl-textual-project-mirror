# Project Overview — AutoML for Text Classification (SS26 Exam)

This document is a detailed technical record of what has actually been built in this
repository, on branch `dev-bilstm-only`, as of 2026-07-24. It exists so the design
decisions, workarounds, and non-obvious behavior don't get lost. It is not a tutorial
for the exam grader — see `README.md` for the official assignment description and
submission requirements.

**Task**: predict the class label of a text review/document across 5 datasets
(`ag_news`, `imdb`, `amazon`, `dbpedia` — practice; `yelp` — held-out exam/test set).
The system must run an AutoML search (HPO) over model configurations and produce a
`predictions.npy` file for the test split.

---

## 1. High-level architecture

```
CLI (automl/cli) → RuntimeConfig
        │
        ▼
automl/__main__.py (`python -m automl`)
        │
        ▼
Optimizer (automl/core/optimizers/*)  ── samples/schedules Configurations
        │
        ▼
Approach (automl/core/approaches/sequence_dl.py)  ── owns model + data prep
        │
        ▼
Trainer (automl/core/trainers/torch_trainer.py)  ── owns the actual train loop
```

Three pluggable layers:

- **Optimizer** — decides *which hyperparameter configuration* to try next and for how
  long (budget/fidelity). Four strategies are implemented: `random`, `smac`, `ifbo`,
  `rl_freeze_thaw` (registered in `automl/core/optimizers/__init__.py`).
- **Approach** — turns a sampled configuration into a concrete model + data pipeline.
  Registered via a decorator into a global registry (`automl/core/registry.py`) and
  auto-discovered at startup. **Only one approach is actually implemented:
  `sequence-dl`** (a BiLSTM classifier), despite the CLI listing four other names — see
  [Known gaps](#10-known-gaps--dead-code--things-to-double-check) below.
- **Trainer** — a generic PyTorch supervised-training loop (optimizer/scheduler
  creation, epoch loop, checkpointing, resume). Only one implementation exists:
  `TorchTrainer`.

Everything is glued together by a `RuntimeConfig` (`automl/cli/types.py`), a `TypedDict`
merged from defaults → YAML config file → CLI flags (`automl/cli/argparser.py`).

---

## 2. Repository map (code only)

```
automl/
  __main__.py                 entrypoint: `python -m automl`
  logger.py                   colored logging + a "fix up already-created loggers" pattern
  trial_plots.py              production diagnostic plots, called at end of every HPO run
  cli/
    argparser.py              DEFAULT_CONFIG, argparse, YAML+CLI merge logic
    types.py                  RuntimeConfig TypedDict (the config schema)
  core/
    registry.py                decorator-based Approach registry + auto-discovery
    types.py                   ApproachName, DatasetSplit, TrainResult, TrialResult, etc.
    configspacehelper.py       ConfigSpace search-space definition for sequence-dl
    datasets.py                per-dataset loaders (ag_news/imdb/amazon/dbpedia/yelp)
    plot_history.py            broader/manual diagnostics (correlation plots etc.)
    approaches/
      base_approach.py         abstract Approach interface
      constants.py             default hyperparameters per approach
      sequence_dl.py           THE approach: BiLSTM + DistilBERT tokenizer
    trainers/
      base_trainer.py          abstract Trainer interface
      torch_trainer.py         THE trainer: generic PyTorch training loop
    optimizers/
      base_optimizer.py        shared HPO infra: checkpointing, history, ensembling
      random.py                RandomSearch baseline
      smac.py                  SMAC3 (Hyperband multi-fidelity Bayesian opt)
      rl_freeze_thaw.py         from-scratch PPO freeze-thaw scheduler
      ifbo/
        hp_space.py             encodes ConfigSpace → [0,1]^d for the FT-PFN surrogate
        candidate.py             IfBOCandidate dataclass (learning curve state)
        optimizer.py             IfboOptimizer: the flagship strategy
    utils/
      misc.py                  set_seed, get_device, JSON encoders, save_incumbent
      timer.py                 Timer context manager, duration formatting
  environment/
    device_info.py             hardware/software snapshot per run (never raises)
    save_requirements.py       `pip freeze` snapshot per run

Root-level scripts:
  run_config.py                debug: run ONE config for N epochs, print/​save metrics
  run_incumbant.py             manual: train + fully retrain + plot ONE incumbent config
  train_top_k_from_history.py  ★ PRODUCES THE FINAL predictions.npy (top-k ensemble)
  ifbo_predict.py              offline FT-PFN posterior/UQ analysis over a history file
  plot_future_predictions.py   calibration + coverage plots from ifbo_predict.py output
  plot_ifbo_freeze_thaw_predictions.py   flexible actual-vs-predicted curve plots
  plot_per_config_overlay.py   overlay all trials of the same config
  save_tokenizer.py            cache a HF tokenizer locally (offline use during HPO)
  download-datasets.py         fetch Phase 1 + Phase 2 dataset zips into ./data
  load-ftpfn.py                warm the FT-PFN surrogate weights cache
  test_approaches.py           DEAD — imports a pre-refactor API that no longer exists

Config/data files:
  runconfig.yml                saved experiment config (yelp, ifbo, cluster-scale)
  runconfig.local.yml          saved experiment config (dbpedia, smaller/local scale) — if present
  logging.yaml                 dictConfig-style logging setup (colored console handler)
  Makefile                     canonical commands: init, run, run-all, check, test
  pyproject.toml               Poetry deps: smac, configspace, neural-pipeline-search,
                                transformers, datasets, colorlog, psutil, filelock...
```

---

## 3. Data pipeline — `automl/core/datasets.py`

`BaseTextDataset(ABC)` is the base class for all five datasets. `load_data()` reads
`data_path/<subdir>/{train,test}.csv` once and caches the result, always returning
**copies** so downstream mutation never corrupts the cache.

Dataset subclasses (`AGNewsDataset`=4 classes, `IMDBDataset`=2, `AmazonReviewsDataset`=5,
`DBpediaDataset`=14, `YelpDataset`=5). `DBpediaDataset` remaps a raw `-1` label sentinel
to `num_classes - 1` found in that dataset's CSVs. `get_dataset_class(name)` is a
`match`-based factory keyed by the same strings as the `--dataset` CLI choice.

**`create_dataloaders(val_size, random_state, train_fraction, max_num_rows)`** is the
pipeline entry point:
1. Loads data, optionally subsamples to `train_fraction` (stratified by label).
2. Optionally caps to `max_num_rows` via `_uniform_sample` — a **stratified subsampling**
   that distributes the row budget across classes proportionally, then tops up from
   leftover rows if small classes couldn't fill their quota. This exists to bound HPO
   trial cost (`max_num_rows` defaults to 40,000) while preserving class balance.
3. Splits train into train/val via `sklearn.train_test_split` (stratified if
   `val_size > 0`).
4. Lowercases text via `preprocess_text()` (currently just `.lower()` — punctuation
   cleanup is commented out / TODO'd).
5. Returns `{train_df, val_df, test_df, num_classes}`.

---

## 4. The `sequence-dl` approach — `automl/core/approaches/sequence_dl.py`

This is the only working approach: a **bidirectional LSTM** classifier over a
**DistilBERT WordPiece tokenizer's vocabulary**, with a PCA/SVD-projected pretrained
embedding warm start.

### 4.1 Caching machinery (the intricate part — exists because HPO runs hundreds of trials)

- **Thread-local tokenizer cache** (`_tokenizer_cache = threading.local()`,
  `_load_tokenizer()`): a fast Rust-backed HF tokenizer mutates its own internal
  truncation config in place per call (since `max_seq_length` varies per trial). Sharing
  one instance across concurrently-running trial threads (under `ifbo`'s
  `num_parallel_trials`) causes `RuntimeError: Already borrowed`. Thread-local storage
  keeps "load once, reuse many times" *within* a thread while isolating concurrent
  threads from each other.
- **Full-encoding cache** (`_full_encoding_cache`, `_encode_texts_cached()`): a
  process-wide dict keyed by `tokenizer_path -> {text: input_ids}`, guarded by a
  `threading.Lock`. Every HPO trial resamples train/val from the *same* fixed text pool,
  so the same texts recur across trials. Each text is tokenized **fully untruncated**
  exactly once; trial-specific truncation to `max_seq_length` is deferred to
  `TextSequenceDataset`, applied to the cached full ids via `_truncate_ids()` — which
  reproduces the tokenizer's native truncate-then-append-`[SEP]` behavior exactly, rather
  than naively slicing. This turns O(trials × corpus) tokenization cost into
  effectively O(corpus) plus cheap slicing (added specifically to fix HPO wall-clock
  cost — see git history: "Cache tokenization across HPO trials and cap max_seq_length
  search space").
- **Pretrained-embedding warm start via SVD** (three chained `lru_cache`d functions:
  `_load_pretrained_word_embeddings` → `_pretrained_svd` → `_pretrained_embedding_init`):
  since the tunable `seq_embed_dim` (32–512) essentially never matches DistilBERT's
  native 768-dim hidden size, a straight copy of pretrained embeddings is impossible.
  Instead the pretrained embedding matrix is mean-centered, SVD'd once per
  `(model_name, vocab_size)`, and projected onto the top `target_dim` right-singular
  vectors (PCA down to `target_dim`) — preserving the highest-variance semantic
  directions so nearby tokens stay close after projection. This is a much better BiLSTM
  embedding init than random. If `target_dim` exceeds the native dim, extra dims are
  padded with small random noise. Because of `lru_cache`, the expensive DistilBERT load
  + SVD happens **at most once** per unique `(model_name, vocab_size)` across the entire
  HPO run.

### 4.2 Dynamic per-batch padding

`_collate_sequences()` — `TextSequenceDataset` stores un-padded (only truncated)
sequences; padding to a common length happens **per batch**, to that batch's longest
sequence, not to a fixed dataset-wide `max_seq_length`. Avoids wasting compute on
padding when most texts are shorter than the configured max.

### 4.3 Model — `BiLSTMClassifier(nn.Module)`

`nn.Embedding(padding_idx=0)` (initialized from the SVD-projected pretrained matrix, row
0 explicitly zeroed) → `nn.LSTM(bidirectional=True, batch_first=True)` → dropout →
`nn.Linear`. `forward()` derives true sequence lengths from where `input_ids !=
padding_idx` and uses `pack_padded_sequence(enforce_sorted=False)` so the LSTM doesn't
compute/backprop through padded positions — a real compute/memory win when
`max_seq_length` is much larger than typical text length. Takes the last layer's
forward+backward final hidden states, concatenates, projects to logits.

### 4.4 `TextSequenceDataset`

Storage optimization: instead of a Python list of per-sample tensors, all samples' token
ids are concatenated into **one contiguous `int32` buffer** plus a cumulative-offsets
index, sliced per `__getitem__` and upcast to `int64` lazily. Two reasons: (1) int32
halves memory vs int64 for a ~30k vocab; (2) avoids the classic multiprocessing
`DataLoader` pitfall where many individually-refcounted Python objects trigger
copy-on-write page duplication across worker processes (multiplying memory by roughly
`num_workers`).

**Label masking**: labels use `DEFAULT_LABEL_MASK = -100` (the PyTorch/HF "ignore this
label" sentinel) for missing/NaN labels via `pd.Series(labels).fillna(-100)`.
⚠️ **Latent inconsistency**: `TorchTrainer`'s `nn.CrossEntropyLoss()` is constructed
**without** `ignore_index=-100`, so -100 would currently be treated as a real (invalid)
class index rather than being ignored, if it ever actually reaches the loss. Worth
fixing or double-checking this path is never exercised in practice (e.g. only used for
the true unlabeled `yelp` test split, which never goes through `train()`).

### 4.5 `SequenceDLApproach(Approach)`

- `TOKENIZER_PATH = "./tokenizers/distilbert-base-uncased"` — a locally-vendored
  tokenizer (see `save_tokenizer.py`/`make tokenizers`), not downloaded per run.
- `prepare(train, val)`: resolves hyperparameters, builds datasets/dataloaders (pinned
  memory + persistent workers when applicable), builds the model with the SVD-projected
  embedding init.
- `train(prepared_result, epochs, load_path=None)`: lazily constructs a `TorchTrainer`
  (only if `self.trainer is None`, so continued-training calls reuse optimizer state),
  delegates the loop.
- `predict(data)`: accepts a raw DataFrame or a prebuilt DataLoader; moves predictions to
  CPU **immediately after each batch** rather than accumulating GPU tensors — bounds
  peak GPU memory to ~one batch instead of the whole prediction set.

### 4.6 Hyperparameter search space — `automl/core/configspacehelper.py`

`build_config_space(seed, fixed_model_type="sequence-dl")`:

| Hyperparameter | Type | Range/Choices | Default | Purpose |
|---|---|---|---|---|
| `model_type` | Constant | `"sequence-dl"` | — | vestige of an intended multi-approach space |
| `hidden_dim` | Categorical | `{32,64,128,256}` | 128 | LSTM hidden size |
| `dropout` | Float | `(0.0, 0.5)` | 0.1 | dropout before final linear layer |
| `learning_rate` | Float, log | `(1e-4, 1e-2)` | 1e-3 | optimizer LR |
| `optimizer` | Categorical | `{adam, adamw, sgd}` | adam | (SGD gets momentum=0.9 internally) |
| `scheduler` | Categorical | `{steplr, cosineannealinglr, exponentiallr, reducelronplateau}` | steplr | LR schedule |
| `weight_decay` | Float, log | `(1e-6, 1e-2)` | 1e-4 | optimizer weight decay |
| `batch_size` | Categorical | `{32,64,128,256}` | 64 | DataLoader batch size |
| `max_seq_length` | Categorical | `{64,128,256}` | 128 | token truncation length (capped from a previous max of 1024 — packed-LSTM cost scales ~linearly with token count, and most signal in a review lives in the first ~256 tokens) |
| `warmup_ratio` | Float | `(0.0, 0.2)` | 0.1 | fraction of epochs spent on linear LR warmup |
| `seq_embed_dim` | Integer, log | `(32, 512)` | 128 | embedding dim (drives SVD projection target) |
| `seq_num_layers` | Integer | `(1, 3)` | 1 | number of stacked LSTM layers |

No conditions/forbidden clauses are actually added (imports exist but are unused).
`max_grad_norm` is used by `TorchTrainer` but is **not** in this space — it's a fixed
default (1.0), not tuned.

---

## 5. Trainer — `automl/core/trainers/torch_trainer.py`

`TorchTrainer` is a generic PyTorch supervised-classification loop.

- **Optimizer/scheduler factories**: string-keyed maps to `torch.optim`/scheduler
  classes (`adam/adamw/sgd/rmsprop`, `steplr/cosineannealinglr/exponentiallr/
  reducelronplateau`), each built with sensible defaults if args aren't given.
- **Stochastic epochs**: if `stochastic_epochs=True`, each "epoch" only iterates a
  random `fraction` (default 0.25) of the batches (`itertools.islice` over a freshly
  reshuffled `DataLoader`), rather than the full dataset. Since the loader reshuffles
  every fresh iteration, each epoch sees a *different* random subsample — so a fixed
  epoch budget still covers the dataset in expectation over several epochs, while each
  individual epoch is cheaper. This lets a multi-fidelity/freeze-thaw scheduler treat
  "epoch budget" as a finer-grained, cheaper cost knob.
- **Linear LR warmup**: for `epoch < warmup_epochs` (`round(warmup_ratio * epochs)`),
  each param group's LR is scaled directly (`base_lr * (epoch+1)/warmup_epochs`) before
  the scheduler steps.
- **Resume**: `train(load_path=...)` restores model/optimizer/scheduler/history/
  `start_epoch`/`best_val_acc` from a `torch.save`'d checkpoint dict — this is the
  mechanism that underlies freeze-thaw's "thaw and continue training" behavior.
- **Checkpointing on improvement**: saves whenever a new best `val_accuracy` is found —
  but only if `save_path` is passed; `SequenceDLApproach.train()` currently always calls
  with `save_path=None`, so this particular in-loop checkpoint path is effectively
  unused there (checkpointing instead happens at the optimizer level, see below).
- **`evaluate()`**: `no_grad`, argmax logits, `sklearn.metrics.accuracy_score`.
- Gracefully handles `KeyboardInterrupt` (logs best-so-far before re-raising) and always
  clears CUDA cache in a `finally` block.

---

## 6. Optimizers — `automl/core/optimizers/`

Four pluggable HPO strategies behind a common `Optimizer` ABC
(`base_optimizer.py`), registered in `automl/core/optimizers/__init__.py`
(`RandomSearch`, `SmacOptimizer`, `RLFreezeThawOptimizer`, `IfboOptimizer`).

### 6.1 Shared infrastructure — `base_optimizer.py`

- Builds the `ConfigurationSpace`, resolves devices (`self.devices` = every visible GPU
  when `device="auto"` + CUDA present, so parallel optimizers can dispatch one trial per
  GPU), creates `checkpoints/`, `checkpoints/trainers/`, `history.log.json(l)`.
- **Concurrency primitives**: one `threading.Lock` guards shared mutable state
  (history, trial counters, best-so-far), plus a per-config-hash lock dict so two
  concurrent trials that hash to the same config don't corrupt each other's checkpoint
  file. No-op when running sequentially.
- **`train_single_configuration`** — the single universal trial-execution routine every
  optimizer calls: builds dataloaders, seeds RNGs, calls `approach.prepare()`/`train()`,
  times it, tracks "best so far," saves a checkpoint if so, appends a `TrialResult` to
  history (+ live JSONL log). Accepts optional device/num_workers overrides so parallel
  callers (ifBO) can pin a trial to a specific GPU.
  - **Deliberate reproducibility trade-off** (documented in-code): only `set_seed()` is
    lock-protected, not the (expensive, CPU-bound) `approach.prepare()` that follows,
    because serializing `prepare()` behind a lock would starve every GPU but one under
    `num_parallel_trials > 1`. Consequence: exact "same seed → same result"
    reproducibility does not hold under parallelism. Accepted because ifBO already
    treats accuracy as a noisy observation.
- **Freeze-thaw checkpoint reuse**: `_config_to_hash_id` (sha256 of the sorted config
  dict) identifies a configuration; trainer checkpoints are saved/loaded per config hash
  — this is the actual mechanism that lets SMAC/Hyperband intensification and ifBO's
  step-based training "thaw" (resume) a partially-trained config instead of restarting.
- **`_finalize_optimization`**: common post-run logic for every optimizer. Handles both
  a single-`Configuration` incumbent and a `list[Configuration]` (ensemble) incumbent
  uniformly — retrains/evaluates each on held-out test data via `evaluate_incumbent`,
  and when multiple incumbents are returned, builds a **majority-vote ensemble**
  prediction (`_majority_vote`, ties broken by `np.unique`'s natural sort order). Saves
  `predictions.npy`. This ensembling logic is generic — any optimizer returning a list
  of configs gets it "for free" — but in practice only `IfboOptimizer` returns a list.
- **`evaluate_incumbent`**: retrains the chosen config from scratch on the **full**
  training data for `evaluation_budget` epochs, predicts on the true test set, saves
  `predictions.npy` (or `predictions_incumbent_{i}.npy` for multi-incumbent runs).

### 6.2 `RandomSearch` (`random.py`)

Simplest baseline: for `n_trials` iterations, uniformly samples a config, always trains
at `max_budget` (no multi-fidelity), tracks best-seen, evaluates at the end. No
freeze-thaw, no surrogate, no parallelism.

### 6.3 `SmacOptimizer` (`smac.py`)

Delegates to the external `smac` (SMAC3) library: builds a `Scenario`
(`min_budget`/`max_budget`/`n_trials`), wraps it in a `Hyperband` intensifier (`eta=3`)
inside a `MultiFidelityFacade`. Classic BOHB-style Hyperband/SMAC — a Bayesian model
(random forest, SMAC3 default) proposes configs, Hyperband's successive-halving
decides which get promoted to higher budgets. A commented-out line hints an earlier
attempt to plug ifBO's intensification directly into SMAC's facade, abandoned in favor
of the separate hand-rolled `IfboOptimizer`.

### 6.4 `RLFreezeThawOptimizer` (`rl_freeze_thaw.py`)

A from-scratch RL environment + PPO policy, no Gym/Stable-Baselines dependency.

- `FreezeThawEnvironment`: `n_initial_configs` candidates sampled once at reset (fixed
  pool, unlike ifBO's dynamically-growing pool). Action space = "start" actions (train
  an unstarted candidate for `start_budgets[action]` epochs, e.g. 1/2/4/8) + "thaw"
  actions (resume a specific started candidate for one more epoch).
- Optionally trades a real training epoch for a cheap "surrogate" epoch
  (`LastValueSurrogate.predict_next` just repeats the last observed accuracy), governed
  by `surrogate_probability` (off by default).
- `_train_ppo`: a minimal PPO implementation (~100 lines) — 2-layer MLP policy over
  masked action logits, matching value network, discounted-return advantages, clipped
  PPO loss, 4 gradient updates/episode. Alternative: `RandomFreezeThawController`
  (uniform-random baseline). After training, runs one final deterministic episode and
  picks the single highest-accuracy candidate as incumbent — **no ensembling here**;
  always returns a single `Configuration`.

### 6.5 `IfboOptimizer` (`ifbo/`) — the flagship strategy

A reimplementation of **ifBO** (in-context freeze-thaw Bayesian optimization,
Rakotoarison et al. ICML 2024) using the pretrained **FT-PFN** surrogate from the
`ifbo` PyPI package.

- **`hp_space.py`**: encodes ConfigSpace hyperparameters into `[0,1]^d` for the
  surrogate (`Float`/`Integer` support log-scale, `Categorical` uses equal-width-bin
  encoding). Drops `model_type` and `warmup_ratio` unconditionally, then caps at
  `MAX_HYPERPARAMETERS = 10` (FT-PFN's hard architectural limit), keeping the first N
  by declaration order and warning about anything dropped. Missing/inactive
  hyperparameters encode to the neutral value `0.5`.
- **`candidate.py`**: `IfBOCandidate` — holds the `Configuration`, its encoded vector,
  `steps_done`, and the learning curve (`ts`/`ys`) FT-PFN conditions on. Carries an
  explicit `uid` because the dataclass (containing a tensor) isn't hashable, and `uid`
  lets a parallel batch dedupe candidate selection.
- **`optimizer.py`**:
  - Budget mapping: `min/max_budget` epochs → discrete freeze-thaw "steps."
  - **Dynamic candidate pool** — unlike RL-freeze-thaw, starts empty and grows via
    `_sample_new_candidate()` on demand.
  - **Acquisition — "dynamic epsilon-greedy MFPI-random"**: with probability `epsilon`
    (polynomial decay toward `eps_min=0.1` over the run), sample a brand-new candidate
    (exploration). Otherwise, among *pending* candidates, use Multi-Fidelity
    Probability-of-Improvement with a **randomized** lookahead horizon (`h_rand ∈
    [1,3]`) and randomized improvement target (`τ = 10^Uniform(-4,-1)`); FT-PFN is
    queried for each pending candidate's PI at that horizon, using all already-observed
    candidates' curves as in-context "training data." Selection is then either greedy
    (argmax PI, `ifbo_greedy_candidate_selection=True`) or stochastic (softmax over PI
    scores). A `ifbo_use_random_selection` flag bypasses the surrogate entirely (uniform
    baseline).
  - The selected candidate is "thawed" for `h_rand` more steps, resuming from its saved
    trainer checkpoint.
  - **Sequential vs. parallel**: `num_parallel_trials` controls batch-synchronous
    parallel dispatch — each round builds one shared FT-PFN context, greedily selects
    up to `parallelism` distinct candidates against that (slightly stale) context
    (documented as standard batch-BO staleness, not a bug), dispatches concurrently via
    a `ThreadPoolExecutor`, round-robins GPUs, waits for the whole round before
    rebuilding context. Explicitly calls `torch.cuda.set_device(device)` per thread
    because PyTorch's "current CUDA device" is thread-local.
  - **Incumbent/ensemble selection**: ranks all evaluated candidates by best observed
    accuracy, keeps the top-`k` (`ifbo_incumbent_ensemble_top_k`, default 5) within
    `ifbo_incumbent_ensemble_accuracy_threshold` (default 0.05 absolute accuracy) of the
    single best. Returns a single `Configuration` if only one survives the threshold,
    else a `list[Configuration]` — triggering `base_optimizer.py`'s majority-vote
    ensembling at finalize time.
  - **Memory cleanup**: explicit `gc.collect()` + CUDA/MPS `empty_cache()` after every
    step/round — FT-PFN inference under `no_grad` still accumulates enough activation
    memory across hundreds of steps to warrant this.

- **The `os.fork` / filelock workaround** (git history: commits `bfb729e`, `50c739a`,
  `a782677`, `5f88832`): `DataLoader(num_workers>0)` forks worker subprocesses. Python
  3.12+'s `filelock` refuses to let a fork happen while any `FileLock` in the process is
  mid-acquire — with concurrent trials, one thread can be inside the JSONL-history
  `FileLock` exactly when another thread's `DataLoader` tries to fork, raising `"os.fork
  is unsafe while filelock is changing descriptor ownership"`. **Fix**: effective
  `num_workers` is forced to `0` whenever `parallelism > 1`, sidestepping forking
  entirely. Cheap because tokenization already happens once upfront via the caching
  described in §4.1.
- **Prewarming**: when running in parallel, the dataset and pretrained embeddings are
  loaded once up front in the constructor (before threads race on a
  check-then-set cache). Deliberately does *not* prewarm the tokenizer, since that
  cache is thread-local by design.

### 6.6 History / checkpoint persistence (shared)

- `history.log.jsonl`: append-only, line-delimited `TrialResult`, written under a
  `FileLock` so concurrent threads don't interleave partial writes.
- `checkpoints/`: best-model checkpoint (`replace_best=True`) plus per-config-hash
  trainer checkpoints (`checkpoints/trainers/<config_hash>/trainer.pth`) backing
  freeze-thaw resume.
- `predictions.npy` / `predictions_incumbent_{i}.npy`: final held-out test predictions.

### 6.7 Offline analysis / plotting toolchain

- `load-ftpfn.py`: 3-line smoke test forcing a download of the pretrained FT-PFN
  weights (feeds `.model/` cache, primed by `make init`).
- `ifbo_predict.py`: standalone CLI — given a `history.log.jsonl`, builds an `ifbo.Curve`
  per trial and queries FT-PFN for full posterior predictive statistics
  (mean/median/mode/variance/quantiles) plus acquisition scores (UCB/EI/PI) beyond what
  was actually observed — exposing ifBO's internal grey-box extrapolation for offline
  inspection.
- `plot_future_predictions.py`: reads `ifbo_predict.py` output, produces per-trial
  timeline plots (predicted mean/quantile bands vs actual), a **calibration plot**
  (predicted vs actual at target epoch, MAE/RMSE), and an **interval-coverage
  histogram** (fraction of times actual fell inside the predicted 5–95% interval,
  compared to the ideal 90%) — a direct surrogate uncertainty-quantification diagnostic.
- `plot_ifbo_freeze_thaw_predictions.py` / `plot_per_config_overlay.py`: more
  flexible/standalone variants for inspecting actual-vs-predicted curves and per-config
  run-to-run variance.
- `automl/trial_plots.py` (the one actually invoked automatically by every optimizer's
  `_finalize_optimization`): epoch heatmap, per-trial and per-config-group learning
  curves, val-error-vs-cumulative-time (best-so-far starred), unique-candidates-seen
  chart (flags exact-duplicate configs — a sampler-diversity sanity check), and a
  Gantt-style config-hash-vs-time chart.

### 6.8 Strategy comparison

| Optimizer | Surrogate/model | Fidelity handling | Parallelism | Ensembling |
|---|---|---|---|---|
| `random` | none | fixed at max_budget | no | no |
| `smac` | SMAC3 random-forest EI | Hyperband successive halving | no (SMAC-internal only) | no |
| `rl_freeze_thaw` | from-scratch PPO policy | fixed candidate pool, 1..8-epoch start budgets | no | no (single incumbent) |
| `ifbo` | FT-PFN in-context transformer | continuous freeze-thaw steps, MFPI-random acquisition | yes (thread pool, GPU round-robin) | yes (top-k threshold, majority vote) |

---

## 7. CLI & runtime configuration

`RuntimeConfig` (`automl/cli/types.py`) is the single schema for all runtime config.
Precedence: `DEFAULT_CONFIG` → YAML file (`--config`, default `runconfig.yml`) → CLI
flags (`automl/cli/argparser.py:merge_config`).

Notable fields: `dataset`, `approach` (only `sequence-dl` works), `optimizer`
(`smac`/`random`/`ifbo`/`rl_freeze_thaw`), `evaluation_budget`/`max_budget`/`min_budget`/
`n_trials`, `max_num_rows`, `val_size`, `num_workers`, `num_parallel_trials`,
`stochastic_epochs`/`stochastic_epoch_fraction`, ifBO-specific
(`ifbo_use_random_selection`, `ifbo_greedy_candidate_selection`,
`ifbo_incumbent_ensemble_top_k`, `ifbo_incumbent_ensemble_accuracy_threshold`).

The repo root's `runconfig.yml` is checked in with: `dataset: yelp`, `seed: 67`,
`optimizer: ifbo` (comment: "use rl_freeze_thaw for the PPO Freeze-Thaw scheduler"),
`n_trials: 50`, `max_budget: 30`, `min_budget: 5`, `evaluation_budget: 50`,
`num_parallel_trials: 2`, `max_num_rows: 50000`, `val_size: 0.2`,
`ifbo_greedy_candidate_selection: true`, `stochastic_epochs: false` — this is the
current cluster/exam-scale configuration targeting the held-out `yelp` dataset.

⚠️ `runconfig.yml` also sets `max_trainers_in_memory: 2` and has a comment
`num_workers: 1 # TODO: I don't think this is being properly used` — both flagged as
**not present** in `RuntimeConfig`'s `TypedDict` (`automl/cli/types.py`)/not obviously
consumed — worth checking whether they're silently ignored or read via a different,
unaudited path before relying on them.

`automl/__main__.py:main()` registers all approaches, snapshots `device_info.json` and
`requirements.txt` into the run's output dir, seeds RNGs, picks the optimizer class by
name, and runs it.

---

## 8. Environment / logging utilities

- **`automl/environment/device_info.py`**: collects a full hardware/software snapshot
  (OS, CPU, RAM, disk, GPU via `pynvml`/`nvidia-smi`/`rocm-smi`/`system_profiler`,
  torch/CUDA/cuDNN/MPS info, relevant package versions, relevant env vars) into
  `device_info.json` per run. Every section is independently try/excepted — **the file
  is designed to never raise**, since environment logging is diagnostic sugar and
  should never block a training run.
- **`automl/environment/save_requirements.py`**: runs `sys.executable -m pip freeze`
  (reflects whichever env — pip/poetry/conda — is actually active) into
  `requirements.txt` per run, for full reproducibility alongside `device_info.json`.
- **`automl/logger.py`**: `TruncatingFormatter` (colored, truncates long logger
  names/filenames for column alignment). `get_logger(name)` is a cached-logger factory
  (module-level `_LOGGERS` dict) so repeated `get_logger()` calls at import time don't
  re-add handlers. `setup_logging(output_path, level)` is the run-time reconfiguration
  entrypoint: since many modules call `get_logger()` at import time (before
  `RuntimeConfig` is parsed and the real level/output path are known), this function
  re-runs `get_logger(..., force_new=True)` for every previously-cached logger name plus
  root — retroactively fixing up already-created loggers.

---

## 9. Root-level driver scripts — how to actually run things

| Script | Purpose | Produces |
|---|---|---|
| `make init` (→ `download-datasets.py`, `save_tokenizer.py` ×2, `load-ftpfn.py`) | one-shot environment bootstrap | `data/`, `tokenizers/`, `.model/` |
| `python -m automl --config runconfig.yml` (or `make run DATASET=... ARGS=...`) | run an HPO search | `results/<dataset>/<runtime_id>/{history.log.jsonl, checkpoints/, device_info.json, requirements.txt, plots}` |
| `run_config.py` | debug: run ONE config for N epochs, no checkpointing | optional result JSON |
| `run_incumbant.py` (typo intentional in filename) | manually train+retrain+plot ONE incumbent config (mirrors `SmacOptimizer`'s evaluation protocol) | model checkpoint, `train_result.json`, loss/accuracy plots |
| **`train_top_k_from_history.py`** | ★ reads a `history.log.jsonl`, dedupes/ranks trials, retrains the **top-k distinct configs** on full training data, majority-votes their test predictions | **`predictions.npy`** (the exam submission artifact), `manifest.json`, `incumbent.json`, per-incumbent checkpoints |
| `ifbo_predict.py` + `plot_future_predictions.py` / `plot_ifbo_freeze_thaw_predictions.py` / `plot_per_config_overlay.py` | offline FT-PFN posterior/UQ analysis and diagnostic plotting over a saved history file | plots, augmented history JSON |

**`train_top_k_from_history.py` is the most important script** — it's the one that
actually mirrors `Optimizer._finalize_optimization`'s ensembling protocol standalone.
Its resumability is a first-class feature: re-running with the same `--output-dir`
resumes from `manifest.json` (already-trained incumbents are skipped); `Ctrl-C` is
caught at multiple levels with best-effort partial-checkpoint saves so an interrupted
run doesn't lose progress. Notably it special-cases `tfidf-linear`
(`can_resume_training = model_type != "tfidf-linear"`) to avoid handing a stale
checkpoint to a trainer that always refits from scratch — dead code today since
`tfidf-linear` isn't implemented, but documents an intended constraint if it ever is.

---

## 10. Known gaps / dead code / things to double check

These are worth resolving (or consciously deciding to leave) before final submission:

1. **Only `sequence-dl` (BiLSTM) is implemented.** `tfidf-ffnn` (the CLI *default*),
   `transformer`, `tfidf-linear`, `bpe-rnn` are listed as `--approach` choices and even
   referenced by name in `constants.py`/`train_top_k_from_history.py`, but no
   corresponding module exists under `automl/core/approaches/`. Passing any of those
   names crashes at `get_approach()` with `Unknown approach`. `automl/core/types.py`'s
   `ApproachName = Literal["sequence-dl",]` is the one place that's honest about this.
2. **`run.py` does not exist on this branch**, despite `README.md` documenting it as the
   main entrypoint ("trains an AutoML-System... generates predictions"). It exists only
   on `main` (an old pre-refactor script using a now-deleted `automl.core.TextAutoML`
   API) and was deleted here as part of the CLI restructure. The real entrypoint today
   is `python -m automl` (`automl/__main__.py`) / `make run`. **The README is stale on
   this point** — worth fixing before submission, since `run_instructions.md` (a
   required deliverable) needs to point at something that actually exists.
3. **`test_approaches.py` is dead code** — imports `TextAutoML`/`AGNewsDataset` from
   `automl` top-level, neither of which exists anymore (confirmed `ImportError`). Either
   delete it or rewrite against the current `registry`/`Approach` API.
4. **Label-mask/loss mismatch**: `TextSequenceDataset` maps missing labels to `-100`
   (the standard "ignore" sentinel) but `TorchTrainer`'s `CrossEntropyLoss()` doesn't
   set `ignore_index=-100`. Currently likely harmless in practice (labeled train/val
   data never has NaNs; the real unlabeled yelp test split never goes through `train()`)
   but is a latent inconsistency worth a defensive fix if label-masking is ever relied
   on for real.
5. **`constants.py` names the LR default `lr`, but the actual ConfigSpace/`sequence_dl.py`
   code path uses `learning_rate`** — the `lr` default in `SEQUENCE_DL_DEFAULT_CONFIG`
   is effectively dead; the fallback-with-WARNING path in `get_param_value` would fire
   if `learning_rate` were ever missing from a sampled config (it isn't, in practice,
   since it's always in the ConfigSpace).
6. **`runconfig.yml` sets `max_trainers_in_memory` and has a `# TODO` next to
   `num_workers`** — neither is confirmed to be consumed by `RuntimeConfig`/the
   optimizer code paths actually read in this pass; worth a follow-up check.
7. Untracked `topk_results/` and `yelp-topk-results/` directories were present in the
   working tree at time of writing — these are `train_top_k_from_history.py` outputs,
   presumably the actual candidate submission artifacts. Make sure the right one's
   `predictions.npy` gets copied to `data/exam_dataset/predictions.npy` before pushing
   to the `test` branch (see `README.md`'s auto-evaluation section).

---

## 11. Suggested reproduction path for final submission

Based on how the pieces actually fit together:

1. `make init` — fetch data, tokenizers, FT-PFN weights.
2. `python -m automl --config runconfig.yml` — run the ifBO HPO search on `yelp`
   (or whichever config/dataset is finalized), producing `results/yelp/<runtime_id>/
   history.log.jsonl`.
3. `python train_top_k_from_history.py --history results/yelp/<runtime_id>/
   history.log.jsonl --dataset yelp --top-k <k> --epochs <full-training-epochs>
   --output-dir yelp-topk-results` — retrain the top-k distinct configs on full data
   and majority-vote ensemble their predictions into `yelp-topk-results/predictions.npy`.
4. Copy that `predictions.npy` into `data/exam_dataset/predictions.npy` and follow the
   `test`-branch push procedure documented in `README.md`.

This two-command sequence (`python -m automl ...` then `train_top_k_from_history.py
...`) is the natural candidate for the two commands required in `run_instructions.md`.
