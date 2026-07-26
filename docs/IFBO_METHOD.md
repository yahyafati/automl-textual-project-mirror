# In-Context Freeze-Thaw Bayesian Optimization (ifBO): Method and Implementation

This document is a rigorous technical description of the optimizer this project
actually submits: `IfboOptimizer` (`automl/core/optimizers/ifbo/`). The other three
optimizers in the repository (`random`, `smac`, `rl_freeze_thaw`, see
`PROJECT_OVERVIEW.md`) are baselines kept for comparison; ifBO is the method used to
produce the final submission.

The implementation follows **ifBO** as introduced in:

> H. Rakotoarison, S. Adriaensen, N. Mallik, S. Garibov, E. Bergman, F. Hutter.
> *In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization.*
> ICML 2024. [arXiv:2404.16795](https://arxiv.org/abs/2404.16795)

Below, §1–§4 restate the method as defined in the paper (with equation numbers matching
the paper's own numbering where applicable), and §5 onward map it precisely onto this
repository's code, with explicit call-outs wherever the implementation extends or
deviates from the original algorithm.

---

## 1. Problem formulation: freeze-thaw hyperparameter optimization

Let Λ be a hyperparameter search space and f(λ, b) the performance (validation
accuracy, in this project) of configuration λ ∈ Λ after being trained for b discrete
training steps (epochs, here). A configuration's *learning curve* is the sequence
{f(λ, 1), f(λ, 2), …} as b grows.

Freeze-thaw HPO relaxes the usual "pick a config, train it to completion, observe one
number" protocol. Instead, resources are spent **incrementally**: at every iteration
the optimizer either **thaws** (resumes/continues) a previously partially-trained,
currently-frozen configuration for one more step, or starts a brand-new configuration.
Formally, the goal is to find a resource allocation {b_λ}_{λ∈Λ}, b_λ ≥ 0,
Σ_λ b_λ ≤ B, that maximizes

```
max_{λ ∈ Λ, 1 ≤ b ≤ b_λ}  f(λ, b)
```

subject to a total budget B expressed in training steps. The history of everything
observed so far is H = {(λ, b, f(λ, b))} — a set of **partial** learning curves, since
most configurations in H will not have been trained to the maximum budget.

**Mapping onto this project**: Λ = the `ConfigurationSpace` built by
`build_config_space()` (`automl/core/configspacehelper.py`, described in
`PROJECT_OVERVIEW.md` §4.6) for the `sequence-dl` BiLSTM approach — 11 tunable
hyperparameters (`hidden_dim`, `dropout`, `learning_rate`, `optimizer`, `scheduler`,
`weight_decay`, `batch_size`, `max_seq_length`, `warmup_ratio`, `seq_embed_dim`,
`seq_num_layers`) plus the fixed constant `model_type`. f(λ, b) is validation accuracy
after b epochs of BiLSTM training. B is `n_trials` (`RuntimeConfig["n_trials"]`) — the
total number of freeze-thaw *steps* the optimizer is allowed to spend, **not** total
epochs.

---

## 2. The FT-PFN surrogate

Rather than fitting a Gaussian Process or random forest online (as SMAC/BOHB do), ifBO
uses **FT-PFN**, a *Prior-data Fitted Network*: a transformer trained once, offline, on
large quantities of **synthetic** learning-curve data sampled from a hand-designed
prior, and then used purely for inference at HPO time via **in-context learning** — no
weights are updated during the actual HPO run.

### 2.1 What the surrogate models

FT-PFN approximates the posterior predictive distribution

```
p( f(λ_test, b_test) | λ_test, b_test, H )
```

i.e., "given everything observed so far about *other* (and this) configurations'
partial learning curves, what is the distribution over this configuration's performance
at some future training step b_test?" The set H is passed to the network as a sequence
of tokens (its **context**) at inference time; the transformer's attention mechanism
implicitly performs Bayesian updating over this context in a single forward pass,
rather than through iterative model refitting. This is what makes each acquisition
query cheap enough to run every freeze-thaw step (the paper reports 10–100× speedups
over refitting-based grey-box surrogates such as DPL and DyHPO).

### 2.2 How it was trained (offline, not part of this repo)

FT-PFN is meta-trained on curves drawn from a two-level generative prior:

- A **config model** π_config(λ; θ): a *randomly initialized* (untrained) neural
  network mapping a hyperparameter vector λ to a set of curve parameters — asymptotic
  performance y∞, K=4 basis-function weights w_k and shape parameters Ψ_k, and noise
  variance σ². Randomizing θ per sampled synthetic "task" emulates drawing from a prior
  over plausible relationships between hyperparameters and training dynamics.
- A **curve model** π_curve(λ, t) ~ N(f_comb(t; E), σ²), where

  ```
  f_comb(t; E) = y0 + (y∞ − y0) · Σ_{k=1}^{4} w_k · f_k(t; Ψ_k)
  ```

  combines four basis functions chosen so the family subsumes common empirical
  learning-curve shapes (power-law, exponential, and curves with "breaking points").

The network is trained by minimizing the negative log-likelihood of held-out points
under its predicted posterior, over many such synthetic datasets — i.e., it is trained
to *be* a Bayesian predictor for this family of curve priors, once, and never again.

**In this repository**, FT-PFN is not trained — the pretrained weights (version
`"0.0.1"`) are loaded directly from the `ifbo` PyPI package:

```python
self.model = FTPFN(version="0.0.1", target_path="../.model")
```

(`automl/core/optimizers/ifbo/optimizer.py:128`). `load-ftpfn.py` /
`make load-ftpfn` exists solely to pre-warm this download into the local `.model/`
cache so the first real HPO trial doesn't stall on a network fetch.

---

## 3. Encoding the search space for the surrogate

FT-PFN's architecture consumes hyperparameters as a fixed-size real vector in
[0, 1]^d — it has no notion of ConfigSpace types. `automl/core/optimizers/ifbo/hp_space.py`
implements this encoding.

Three `HPSpec` subclasses cover every hyperparameter type present in `build_config_space()`:

- **`Float(low, high, log)`** / **`Integer(low, high, log)`**: linearly (or, if
  `log=True`, log-linearly) rescale the value into [0, 1]:

  ```
  encode(v) = clip( (φ(v) − φ(low)) / (φ(high) − φ(low)),  0, 1 )
  ```

  where φ = log if `log=True` else the identity. Decoding is the exact inverse. This is
  the standard PFN-style normalization.
  reference implementation the paper's authors distribute.
- **`Categorical(choices)`**: maps each choice to the **center** of its equal-width bin,
  `encode(choice) = (index(choice) + 0.5) / len(choices)`, deliberately avoiding exact
  0 or 1 (which would sit on the space's boundary).

`HyperparameterSpace` (the container) enforces FT-PFN's hard architectural limit of
**`MAX_HYPERPARAMETERS = 10`** input dimensions. Given the 11 real hyperparameters plus
the constant `model_type` in this project's search space:

1. `model_type` (a `ConfigSpace.Constant`) is dropped unconditionally — a constant
   carries no information for the surrogate.
2. `warmup_ratio` is also dropped unconditionally (hard-coded in
   `hyperparams_to_drop`, `hp_space.py:140`) — chosen because it consistently has the
   smallest measured effect on validation accuracy versus the other 10 hyperparameters,
   getting the count under the 10-dimension cap without needing to drop something with
   larger expected impact (e.g. `learning_rate` or `hidden_dim`).
3. The remaining hyperparameters are kept in the order they were declared to
   `HyperparameterSpace(**specs)`, which — since Python kwargs preserve insertion
   order — is exactly the declaration order in `_build_hp_space()`'s loop over
   `cs.get_hyperparameters()`. Anything still beyond the cap would be dropped with a
   logged warning, but with the current 10-hyperparameter search space this never
   triggers.

The resulting encoder has `dim = 10`. A configuration λ (a `dict`) is encoded to
z ∈ [0,1]^10 by `HyperparameterSpace.encode()`; any hyperparameter absent or `None`
(inactive under some `ConfigSpace` condition — not currently used in this space, but
handled defensively) maps to the neutral value 0.5.

---

## 4. Candidate state and the discretized budget axis

`IfBOCandidate` (`automl/core/optimizers/ifbo/candidate.py`) is the unit of state ifBO
tracks per configuration:

```python
@dataclass
class IfBOCandidate:
    config: Configuration      # the raw ConfigSpace configuration λ
    z: torch.Tensor            # its [0,1]^10 encoding (computed once, reused)
    steps_done: int = 0        # freeze-thaw steps executed so far
    ts: list[float]            # normalized time coordinates of every observation
    ys: list[float]            # observed accuracy at each of those times
    uid: int = -1              # process-unique identity (see §7)
```

**Budget discretization.** The optimizer does not operate directly in epochs; it
operates on an integer step index in `{1, …, b_max}`, where

```
b_max = max_budget − min_budget + 1
step_to_budget(step) = min_budget + step − 1        # step=1 → min_budget, step=b_max → max_budget
```

(`optimizer.py:65`, `_step_to_budget`, `optimizer.py:306`). Each call to `_step()`
increments a candidate's `steps_done` by (typically) the acquisition function's chosen
horizon, trains it for `step_to_budget(steps_done)` **total** epochs — i.e. the
approach's trainer resumes from its last checkpoint and trains up to this new epoch
count, not from scratch — and records one new observation:

```
t = steps_done / b_max      # normalized time in [0, 1], matching the paper's convention
y = 1 − val_error            # accuracy in [0, 1]
```

appended to `cand.ts` / `cand.ys`. This is the "thaw" operation: pausing every other
candidate ("frozen") and spending the step budget entirely on the selected one.

---

## 5. Acquisition function: MFPI and MFPI-random

### 5.1 The paper's definition

Multi-fidelity Probability of Improvement, as defined in the paper (Eq. 3):

```
MFPI(λ; h, T) = P( M(λ, min(b_λ + h, b_max)) > T )
```

— the surrogate-predicted probability that configuration λ, if thawed for h more
steps, would exceed a target performance T. Two "hyper-hyperparameters" govern it: the
lookahead horizon h and the improvement target T. Rather than fixing these, the paper
proposes **MFPI-random** (Eq. 4), redrawing both **every single freeze-thaw iteration**:

```
MFPI-random(λ) = MFPI(λ; h_rand, T_rand)
h_rand        ~ U(1, b_max)
T_rand        = f_best + τ_rand · (1 − f_best)
log10(τ_rand) ~ U(−4, −1)
```

where f_best is the best performance observed so far. This amounts to sampling an
acquisition function from an implicit *portfolio* of MFPI instances at every step,
which the paper's ablations (Figure 4) show is necessary — fixed-horizon or
fixed-threshold variants, and standard Expected Improvement paired with FT-PFN's
heavy-tailed posteriors, both underperform substantially.

### 5.2 This implementation — `_select_next_candidate()` (`optimizer.py:392`)

The core computation matches the paper closely, with one explicit, documented
deviation:

```python
MAX_LOOKAHEAD = 3
f_best  = self._best_so_far_accuracy()
h_rand  = self._rng.randint(1, MAX_LOOKAHEAD)          # deviates: capped at 3, not b_max
tau_rand = 10 ** self._rng.uniform(-4, -1)
T_rand  = f_best + tau_rand * (1.0 - f_best)            # matches Eq. 4 exactly
```

- **τ_rand / T_rand match the paper's Eq. 4 exactly** — a log-uniform improvement
  fraction over the remaining headroom (1 − f_best) above the current best.
- **h_rand deviates from the paper**: the paper samples `h_rand ~ U(1, b_max)`; the
  implementation samples from `U{1, …, 3}` (`MAX_LOOKAHEAD = 3`), with the in-code
  rationale *"to prevent it from running to max budget"* — i.e., without a cap, a large
  sampled h could immediately promote a barely-explored candidate all the way to
  `max_budget` epochs on a single (possibly lucky) PI estimate, which is expensive
  (each additional epoch of BiLSTM training is real wall-clock cost) and defeats the
  incremental, many-small-steps character freeze-thaw is meant to have. This is a
  practical compute-budget safeguard, not a reproduction of the paper's exact sampling
  distribution.

For every **pending** candidate (one that hasn't yet reached `b_max` steps), the query
time is `t' = min(steps_done + h_rand, b_max) / b_max`, and FT-PFN is queried in a
single batched call:

```python
predictions = self.model.predict(context=context, query=query)
pi_scores = torch.stack([pred.pi(T_rand_tensor).squeeze() for pred in predictions])
```

where `context` is built by `_build_context()` — one `ifbo.Curve` per candidate that
has ≥ 1 observation, i.e. exactly the freeze-thaw history H from §1, encoded as
(z, ts, ys) triples. This is the in-context conditioning set described in §2.1: FT-PFN
never "fits" anything, it consumes H as tokens on every call.

### 5.3 Candidate selection given PI scores — an added selection-mode switch

The paper's Algorithm 1 selects λ = arg max_λ MFPI-random(λ) deterministically. This
implementation exposes both modes via the `ifbo_greedy_candidate_selection` runtime
flag:

```python
if self.greedy_selection:
    idx = torch.argmax(pi_scores)
    selected = pending[idx]                       # matches paper's arg max
else:
    weights = torch.softmax(pi_scores, dim=0).tolist()
    selected = self._rng.choices(pending, weights=weights, k=1)[0]   # stochastic variant
```

The softmax-sampling branch is **not part of the published ifBO algorithm** — it is an
implementation-level extension intended to inject additional exploration among *already
pending* candidates (as opposed to only exploring by spawning brand-new candidates,
§6), trading off some of the pure-exploitation character of arg-max PI selection for
more diverse coverage of the pending pool. `runconfig.yml` in this repo sets
`ifbo_greedy_candidate_selection: true`, i.e. the exam configuration actually runs the
paper-faithful arg-max variant.

A third mode, `ifbo_use_random_selection`, bypasses the surrogate entirely (uniform choice
among pending candidates) — a debugging/ablation baseline, not used in the final config.

---

## 6. Growing the candidate pool: an added ε-greedy exploration layer

The paper's Algorithm 1 (see §7 for the literal loop) is stated over the *whole* search
space Λ implicitly — at each iteration the acquisition is (conceptually) maximized over
all λ ∈ Λ, which naturally lets brand-new, never-tried configurations compete against
partially-trained ones for being selected next.

This implementation instead maintains an **explicit, dynamically growing list**
`self.candidates` (starting empty) and makes the "propose a new configuration vs.
continue an existing one" decision via an **explicit decaying ε-greedy rule** —
`_epsilon()` (`optimizer.py:272`):

```
ε(t) = ε_min + (ε₀ − ε_min) · (1 − t/T)^p,     ε_min = 0.1,  p = 2,  T = total_steps
```

At each iteration, with probability ε(t) a brand-new configuration is sampled uniformly
from Λ (`self.space.sample_configuration()`) and added to the pool with zero
observations; with probability 1 − ε(t), MFPI-random (§5) selects among the existing
*pending* candidates instead. `ε₀` (`ifbo_initial_epsilon`, default 1.0) means the very
first steps are pure exploration (every early step spawns a new candidate, since with
an empty/small pool there's nothing meaningful yet for MFPI-random to discriminate
between), decaying polynomially toward a floor of 0.1 so some exploration always
remains, even late in the run.

**This ε-greedy layer is a deliberate engineering addition, not part of the published
ifBO method.** It exists because this implementation manages a discrete, growing
candidate list rather than treating "propose a new λ" as one more option scored by the
same acquisition function — a pragmatic simplification, since scoring an *unobserved*
configuration's PI is degenerate (its curve is empty; there's nothing for FT-PFN to
extrapolate from except the population-level prior baked into its weights, which the
paper's own architecture does support via zero-context queries, but this implementation
does not attempt to score new-candidate proposals against pending ones on the same
acquisition scale — it decides "explore vs. exploit" first, structurally, then applies
MFPI-random only within whichever branch is chosen).

---

## 7. The full algorithm loop

### 7.1 The paper's Algorithm 1 (restated)

```
Initialize b_λ ← 0 for all λ ∈ Λ
while budget B not exhausted:
    λ ← arg max_λ  MFPI-random(λ)      # acquisition over all of Λ
    b_λ ← b_λ + 1                       # thaw one more step
    y ← f(λ, b_λ)                       # observe (train one more step)
    H ← H ∪ {(λ, b_λ, y)}
return λ* = arg max_{(λ,b) ∈ H} f(λ, b)
```

### 7.2 This implementation — sequential mode (`_perform_ifbo_sequential`, `optimizer.py:566`)

```python
used_steps = 0
while used_steps < self.total_steps:
    context = self._build_context()                    # H, encoded, §5.2
    next_cand, steps = self._select_next_candidate(     # ε-greedy (§6) + MFPI-random (§5)
        context, used_steps + 1
    )
    if next_cand.steps_done >= self.b_max:
        break                                            # every candidate already maxed out
    self._step(next_cand, steps)                         # thaw for `steps` steps, observe
    used_steps += 1
    gc.collect(); torch.cuda.empty_cache()                # §9
return self._select_incumbent()                           # §8
```

Structurally this is the paper's loop, with two differences already covered above: (1)
new-vs-existing selection is decided by the explicit ε-greedy gate before MFPI-random
ever runs, and (2) a selected step can advance a candidate by `h_rand` steps at once
(not always exactly 1) — a genuine implementation of "freeze-thaw look-ahead," since the
paper's own MFPI-random already reasons about an h-step-ahead horizon when scoring a
candidate; advancing by that same h when the candidate is chosen keeps the acquisition
target and the actual training increment consistent, rather than only ever training one
step regardless of what horizon was used to justify the choice.

### 7.3 Parallel / batch-synchronous mode (`_perform_ifbo_parallel`, `optimizer.py:612`)

When `num_parallel_trials > 1` (`runconfig.yml` sets this to 2 for the exam run), the
loop dispatches whole **rounds** of up to `num_parallel_trials` candidates at once:

```python
with ThreadPoolExecutor(max_workers=self._parallelism) as executor:
    while used_steps < self.total_steps:
        context = self._build_context()                 # ONE context per round
        batch = []
        selected_uids = set()
        for _ in range(round_size):
            cand, steps = self._select_next_candidate(
                context, used_steps + 1, exclude=selected_uids   # no duplicate picks
            )
            selected_uids.add(cand.uid)
            batch.append((cand, steps))

        futures = [
            executor.submit(self._step_on_device, cand, steps,
                             self.devices[i % len(self.devices)],
                             self._effective_num_workers)
            for i, (cand, steps) in enumerate(batch)
        ]
        for fut in futures:
            fut.result()                                   # wait for the whole round
        used_steps += len(batch)
```

This is **batch Bayesian optimization**: every candidate within a round is selected
against the *same* context (i.e. candidates 2..N in a round don't see candidate 1's
outcome, only that it was excluded from re-selection via `selected_uids`). The code
comment explicitly frames this as "standard batch-BO staleness, not a bug" — a
well-known, accepted approximation whenever true sequential (one-at-a-time,
observe-then-reselect) BO is parallelized: batch members are chosen under a *slightly
stale* posterior compared to what a strictly sequential run would have used, in
exchange for actually using multiple GPUs concurrently. Devices are assigned
round-robin (`self.devices[i % len(self.devices)]`), and each worker thread explicitly
sets its own CUDA "current device" before training (`_step_on_device`,
`optimizer.py:365`) since that state is thread-local in PyTorch and not inherited from
the thread that spawned the pool.

### 7.4 Freeze/thaw checkpointing mechanism (shared with SMAC, in `base_optimizer.py`)

A configuration is identified by a stable hash of its (sorted, JSON-serialized)
hyperparameter dict (`_config_to_hash_id`, sha256, first 16 hex chars). "Thawing" a
candidate concretely means: look up `checkpoints/trainers/<config_hash>/trainer.pth`
(`_trainer_checkpoint_path`); if present, pass it to `approach.train(load_path=...,
trainer_load_path=...)`, which — per `TorchTrainer.train()`'s resume logic (see
`PROJECT_OVERVIEW.md` §5) — restores model/optimizer/scheduler/history/`best_val_acc`
and continues from `start_epoch` rather than reinitializing. After training, the
(possibly updated) trainer state is written back to the same path
(`_save_trainer_checkpoint`), so the next thaw of this exact configuration resumes
correctly. A per-config-hash lock (`_checkpoint_lock_for`) guards this
read-then-train-then-write sequence so two concurrently-running trials never race on
the same checkpoint file — relevant because `_select_next_candidate`'s `exclude` set
already prevents the *same* candidate from being picked twice in one round, but a
different round could still overlap with an in-flight save from the previous round
under high parallelism.

---

## 8. Incumbent selection and ensembling — a deliberate extension beyond vanilla ifBO

The paper returns a **single** incumbent, λ* = arg max over all observations in H.

This implementation's `_select_incumbent()` (`optimizer.py:528`) generalizes this to an
**ensemble of near-best incumbents**:

1. Rank every candidate by its best-ever observed accuracy,
   `_candidate_best_accuracy(c) = max({y ∈ c.ys : y is finite})`.
2. Keep the top-`ifbo_incumbent_ensemble_top_k` (default 5) candidates whose accuracy is
   within `ifbo_incumbent_ensemble_accuracy_threshold` (default 0.01, i.e. one
   percentage point of absolute validation accuracy in `runconfig.yml`'s override) of
   the single best candidate:

   ```
   incumbents = { c ∈ top_k(candidates, by=best_accuracy) :
                  best_accuracy − best_accuracy(c) ≤ threshold }
   ```

3. If exactly one candidate survives, return it as a plain `Configuration` (the
   single-incumbent, paper-faithful case). If more than one survives, return a
   `list[Configuration]`.

`Optimizer._finalize_optimization()` (`base_optimizer.py:144`, shared infrastructure —
see `PROJECT_OVERVIEW.md` §6.1) treats a `list[Configuration]` specially: each
incumbent is independently retrained on the full training data and evaluated on the
true held-out test set via `evaluate_incumbent()` (`base_optimizer.py:536` — full
retrain for `evaluation_budget` epochs from scratch, `val_size=0.0`, i.e. the previous
freeze-thaw checkpoints are *not* reused here — this is a clean, final retrain, not a
"continue thawing" step), and their per-sample class predictions are combined by
**majority vote** (`_majority_vote`, ties broken by `np.unique`'s ascending natural sort
order) into the final `predictions.npy`.

**Why this deviates from the paper, and why deliberately**: ifBO as published optimizes
purely for finding the single best configuration under a fixed evaluation budget; this
project's actual deliverable is a *prediction file*, evaluated by held-out accuracy, not
a configuration recommendation. Several configurations that ended freeze-thaw search
within a hair of each other's best observed accuracy are plausibly making
*different, partially uncorrelated errors* (different BiLSTM hidden sizes / embedding
dims / regularization settings), so a majority-vote ensemble over near-tied incumbents
is a standard, low-risk way to trade a small amount of extra compute (top_k full
retrains instead of 1) for a typically-more-robust final prediction than committing to
whichever single trial happened to log the highest validation accuracy — especially
given the acknowledged reproducibility caveat in §9 (validation accuracy is a somewhat
noisy signal under concurrent trials).

---

## 9. Practical/engineering considerations specific to running ifBO here

These are covered in full in `PROJECT_OVERVIEW.md`, cross-referenced briefly here since
they materially affect how faithfully the freeze-thaw loop above executes in practice:

- **Memory cleanup** (`gc.collect()` + `torch.cuda.empty_cache()`/`torch.mps.empty_cache()`
  after every step or round, `optimizer.py:593-599` / `690-696`): FT-PFN inference
  under `torch.no_grad()` still accumulates enough activation memory across hundreds of
  steps to require explicit cleanup, since the acquisition step (§5.2) runs once per
  freeze-thaw iteration, not just once per HPO run.
- **The `os.fork` / `filelock` interaction under parallel trials**: `DataLoader(
  num_workers>0)` forks worker subprocesses; Python 3.12+'s `filelock` refuses a fork
  while any `FileLock` (e.g. the JSONL-history writer) is mid-acquire elsewhere in the
  process. Under `num_parallel_trials > 1` this is resolved by forcing
  `_effective_num_workers = 0` (`optimizer.py:160`) rather than merely reducing it —
  affordable because tokenization is already cached upfront (see next point), so losing
  `DataLoader` worker parallelism costs little.
- **Cross-trial tokenization caching and thread-local tokenizer state**
  (`automl/core/approaches/sequence_dl.py`): since ifBO resamples train/val from the
  same fixed text pool on every `_step()` call (each is a fresh call to
  `train_single_configuration` → `dataset.create_dataloaders(...)`), the same texts
  recur across hundreds of freeze-thaw steps; a process-wide encoding cache plus a
  thread-local tokenizer instance (to avoid two concurrent trials racing on one
  Rust-tokenizer's mutable truncation state) turn what would be O(steps × corpus)
  tokenization cost into effectively O(corpus).
- **Prewarming under parallelism** (`_prewarm_shared_resources`, `optimizer.py:174`):
  the dataset and the pretrained DistilBERT embedding matrix (needed for the
  SVD-projected embedding warm start described in `PROJECT_OVERVIEW.md` §4.1) are
  loaded once in the constructor before any worker thread can race on populating those
  caches redundantly.
- **Reproducibility under parallelism**: only the `set_seed()` call inside
  `train_single_configuration` is lock-protected, not the CPU-bound `approach.prepare()`
  that follows it (`base_optimizer.py:435-460`) — serializing `prepare()` would starve
  every GPU but one under `num_parallel_trials > 1`. Consequence: exact "same seed →
  same result" reproducibility does not hold when running with parallel trials, which
  is an accepted trade-off since ifBO already treats each observed accuracy as a noisy
  sample of the true learning curve, not a ground truth to be reproduced bit-for-bit.

---

## 10. Summary: paper vs. this implementation

| Component | ifBO (Rakotoarison et al., 2024) | This implementation |
|---|---|---|
| Surrogate | FT-PFN, meta-trained offline on synthetic curves | Pretrained FT-PFN v0.0.1, loaded as-is from the `ifbo` package — never retrained here |
| Search space dim | Not architecturally capped in the paper's presentation | Hard-capped at 10 dims (`MAX_HYPERPARAMETERS`); `model_type` and `warmup_ratio` dropped to fit |
| Acquisition | MFPI-random, Eq. 3–4 | Same τ_rand/T_rand formula (Eq. 4, exact); h_rand capped at `U{1,2,3}` instead of `U(1, b_max)` for compute-cost control |
| Candidate selection given PI | arg max (deterministic) | Configurable: arg max (`ifbo_greedy_candidate_selection=True`, used in the exam config) **or** softmax-stochastic sampling |
| New-vs-continue decision | Implicit in maximizing acquisition over all of Λ | Explicit decaying ε-greedy gate (`ε_min=0.1`, `p=2`) that structurally separates "explore a new λ" from "exploit via MFPI-random" |
| Parallelism | Not part of the core algorithm as presented | Batch-synchronous multi-GPU dispatch (`num_parallel_trials`), with documented batch-BO staleness |
| Final output | Single incumbent λ* | Optionally an ensemble of up to `ifbo_incumbent_ensemble_top_k` near-tied incumbents (accuracy within a configurable threshold), combined via majority vote over their held-out predictions |
| Freeze/thaw mechanics | Conceptual (train b_λ+1 steps) | Concrete PyTorch checkpoint round-trip per config hash (`trainer.pth`), reused across freeze-thaw steps within the search and discarded in favor of a clean full retrain at final evaluation |

---

## 11. Where to look in the code

| Concept | File |
|---|---|
| Optimizer entrypoint, main loop, acquisition | `automl/core/optimizers/ifbo/optimizer.py` |
| Hyperparameter → [0,1]^d encoding | `automl/core/optimizers/ifbo/hp_space.py` |
| Per-candidate learning-curve state | `automl/core/optimizers/ifbo/candidate.py` |
| Shared trial execution, checkpointing, ensembling, final evaluation | `automl/core/optimizers/base_optimizer.py` |
| The BiLSTM model being optimized | `automl/core/approaches/sequence_dl.py` |
| Search space definition (Λ) | `automl/core/configspacehelper.py` |
| Exam-run configuration | `runconfig.yml` (`optimizer: ifbo`, `num_parallel_trials: 2`, `n_trials: 50`, `min_budget: 5`, `max_budget: 30`) |
| Turns a finished `history.log.jsonl` into the final `predictions.npy` | `train_top_k_from_history.py` (see `PROJECT_OVERVIEW.md` §9) |
