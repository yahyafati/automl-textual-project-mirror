# Poster Content — Freeze-Thaw HPO with FT-PFN Surrogate for Text Classification

Content draft, laid out panel-by-panel to match the structure of `poster-example.ignore.pdf`
("Towards Automatically-Tuned Neural Networks", ML4AAD/Uni Freiburg poster: header band →
"In a nutshell" strip → 4–5 wide content panels → figures with captions → footer branding).
Drop each panel's text into the corresponding box of the official poster template. Anything
in `[brackets]` is a placeholder you must fill in before printing — pull the real numbers from
your final `train_top_k_from_history.py` / `evaluate_incumbent` run, not the provisional ones
noted here.

---

## Header band

**Title** (pick one — see `docs/TITLE_OPTIONS.md` for more options):
> Freeze-Thaw HPO with FT-PFN Surrogate for Text Classification

**Authors**: `[Name 1, Name 2, ...]`
**Affiliation**: Department of Computer Science, University of Freiburg — SS26 AutoML Exam
**Contact**: `[emails]`

---

## Panel 1 — In a nutshell

- Text classification across 5 datasets (`ag_news`, `imdb`, `amazon`, `dbpedia`, held-out **`yelp`**), graded on **methodological rigor**, not raw accuracy.
- We treat model training itself as a resource to spend **incrementally**: instead of training one config to completion before judging it, our optimizer trains many configs *partially and in parallel* and predicts which partial learning curve is worth continuing.
- **ifBO** (in-context freeze-thaw Bayesian optimization, Rakotoarison et al., ICML 2024) drives the search, using a pretrained transformer surrogate (**FT-PFN**) to forecast learning curves with no per-run retraining of the surrogate itself.
- Implemented and empirically compared **four** HPO strategies — Random Search, SMAC/Hyperband, a from-scratch PPO freeze-thaw controller, and ifBO — on one shared model/data harness, so the gap between them is attributable to method, not implementation.
- Final prediction: a **top-k majority-vote ensemble** of the best distinct configurations found, retrained on full data.
- Single-command pipeline: `python -m automl --config runconfig.yml`.

---

## Panel 2 — Method / System Architecture

Three cleanly separated, independently swappable layers — built this way specifically so
four HPO strategies could be **fairly compared** against the same model/data code, rather
than each optimizer bringing its own bespoke training loop.

```
CLI (RuntimeConfig: defaults → YAML → CLI flags)
        │
        ▼
Optimizer   ── decides WHICH config to try next and for HOW LONG (fidelity/budget)
        │        implementations: RandomSearch · SMAC (Hyperband) · RL-Freeze-Thaw (PPO) · ifBO ★
        ▼
Approach    ── turns a config into a concrete model + data pipeline
        │        implementation: sequence-dl = BiLSTM over a DistilBERT WordPiece vocabulary
        ▼
Trainer     ── generic PyTorch train loop (optimizer/scheduler, resume, checkpointing)
```

**Talking point**: "We didn't just pick one HPO method — we built the harness so we could
swap optimizers and prove ifBO wins for a reason, not by accident of implementation."

### Architecture choice

- **Tokenizer**: DistilBERT's pretrained WordPiece vocabulary, used only for subword
  tokenization (not the DistilBERT model). Vendored locally so HPO trials never hit the network.
- **Model**: bidirectional LSTM, chosen over a full transformer fine-tune because (a) the
  exam spec states these datasets are solvable without large-scale transformers, (b) a
  BiLSTM is cheap enough to train hundreds of times across 5 datasets inside a 24h budget,
  and (c) it keeps the tuned-hyperparameter budget on architecture/optimization choices
  instead of "which frozen layers to unfreeze."
- **Pretrained embedding warm start**: the tuned embedding dimension (32–512) never equals
  DistilBERT's native 768-dim table, so the pretrained embedding matrix is SVD-projected
  onto its top-`k` principal directions before initializing the BiLSTM's embedding layer —
  a transfer-learning component even though the classifier itself isn't a pretrained
  transformer.
- **Efficiency engineering**: full-text tokenization cached once across all HPO trials
  (O(trials × corpus) → O(corpus)); `pack_padded_sequence` so the LSTM never computes over
  padding; per-batch dynamic padding; token ids stored as one contiguous `int32` buffer to
  avoid `DataLoader` worker memory multiplication.

---

## Panel 3 — Search Space (11 tuned hyperparameters)

| Hyperparameter | Range / Choices | Why it's in the space |
|---|---|---|
| `hidden_dim` | {32, 64, 128, 256} | LSTM capacity vs. overfitting/compute tradeoff |
| `seq_num_layers` | 1–3 | depth — a lightweight stand-in for architecture search |
| `seq_embed_dim` | 32–512 (log) | representation capacity; drives the SVD projection target |
| `dropout` | 0.0–0.5 | regularization, especially for small/noisy classes |
| `learning_rate` | 1e-4–1e-2 (log) | single most sensitive hyperparameter empirically |
| `weight_decay` | 1e-6–1e-2 (log) | regularization |
| `optimizer` | {adam, adamw, sgd} | algorithm selection within the space |
| `scheduler` | {steplr, cosine, exponential, reduce-on-plateau} | LR-schedule selection |
| `warmup_ratio` | 0.0–0.2 | early-training stability |
| `batch_size` | {32, 64, 128, 256} | compute / statistical-efficiency tradeoff |
| `max_seq_length` | {64, 128, 256} | **key fidelity/cost knob** — capped down from an earlier 1024 after finding packed-LSTM cost scales ~linearly with token count and most signal lives in the first ~256 tokens |

**Talking point**: every bound has a stated reason — directly answers the "search strategy
justification" rubric line.

---

## Panel 4 — HPO Methodology (centerpiece — most weight, most poster space)

Four optimizers implemented on the *same* model/data harness:

| Optimizer | Surrogate / model | Fidelity handling | Parallel? | Ensembling |
|---|---|---|---|---|
| Random Search (baseline) | none | fixed, full budget every trial | no | no |
| SMAC (BOHB-style) | random-forest EI (SMAC3) | Hyperband successive halving | no | no |
| RL Freeze-Thaw (from scratch) | PPO policy (~100 lines, no Gym/SB3) | fixed candidate pool, 1/2/4/8-epoch start actions + thaw actions | no | no (single incumbent) |
| **ifBO ★ (submitted)** | **FT-PFN** — pretrained in-context transformer | continuous freeze-thaw steps, dynamic pool | **yes** (thread pool, GPU round-robin) | **yes** (top-k threshold, majority vote) |

This table alone demonstrates **multi-fidelity optimization, Bayesian optimization,
meta-learning** (a network pretrained offline on synthetic curves), and **reinforcement
learning**, applied to the same problem so their tradeoffs are directly comparable rather
than asserted.

### Why ifBO is the flagship method

Reference: Rakotoarison et al., *In-Context Freeze-Thaw Bayesian Optimization for
Hyperparameter Optimization*, ICML 2024 (arXiv:2404.16795).

- **Problem framing**: instead of "pick config → train to completion → observe one number,"
  freeze-thaw HPO spends a *budget of training steps* incrementally — each iteration either
  starts a new config or resumes ("thaws") a partially-trained, currently-frozen one.
- **The surrogate (FT-PFN)**: a Prior-data Fitted Network — a transformer pretrained once,
  offline, on millions of *synthetic* learning curves (power-law / exponential /
  breaking-point shapes). At HPO time no weights are updated: observed curves so far are fed
  in as **in-context** tokens, and one forward pass approximates the Bayesian posterior over
  "how will this curve continue?" (paper reports 10–100× speedup over refitting-based
  surrogates like DPL/DyHPO). This *is* the meta-learning component of the project.
- **Acquisition — "dynamic epsilon-greedy MFPI-random"**: with probability ε (polynomial
  decay toward 0.1 over the run) sample a brand-new candidate (exploration); otherwise pick
  among pending candidates via Multi-Fidelity Probability-of-Improvement with a
  **randomized lookahead horizon** (1–3 steps) and **randomized improvement target**
  (log-uniform 1e-4–1e-1) — a deliberate diversity mechanism.
- **Ensembling**: keeps every candidate within a fixed accuracy threshold of the best
  observed (top-k, threshold-gated) and majority-votes their test predictions, instead of
  committing to a single incumbent.
- **Parallelism**: batch-synchronous — one shared FT-PFN context per round, up to N
  distinct candidates selected against it, dispatched concurrently across GPUs. The
  resulting staleness for candidates 2..N is a documented, accepted standard-batch-BO
  tradeoff, not a bug.

*(Suggested figure here: freeze-thaw loop diagram — start vs. thaw actions, FT-PFN
in-context surrogate, MFPI-random acquisition. Or reuse a `trial_plots.py` learning-curve
panel from a real run.)*

### Validation protocol

- Stratified train/val split per trial; stratified subsampling (`max_num_rows`) bounds
  trial cost while preserving class balance — matters for imbalanced/large datasets like
  `yelp` and `dbpedia`.
- Final incumbent(s) retrained **from scratch on the full training set** for a larger
  evaluation budget, then evaluated on true held-out test data — validation selects, full
  data + held-out test reports.
- `train_top_k_from_history.py` independently re-derives the top-k configs from a saved
  trial history and reproduces the same ensembling protocol standalone — the final
  submission artifact is reproducible from a frozen log alone, not only a live run.

---

## Panel 5 — Lecture-Concept Mapping

*(Exam requirement: "denote on poster the weeks from which concepts were used." Fill in
your course's actual week numbers against each row before printing.)*

| Concept used | Where in our system | Lecture week |
|---|---|---|
| Multi-fidelity optimization / successive halving / Hyperband | SMAC(Hyperband) optimizer | `[week ]` |
| Bayesian optimization (surrogate + acquisition function) | SMAC (random forest) and ifBO (FT-PFN) | `[week ]` |
| Meta-learning / learning-curve extrapolation / transfer across tasks | FT-PFN surrogate; SVD-projected pretrained embedding warm start | `[week ]` |
| Reinforcement learning for scheduling/control | From-scratch PPO freeze-thaw controller | `[week ]` |
| Ensemble methods | Top-k majority-vote ensembling (ifBO incumbent selection) | `[week ]` |
| Algorithm / model selection within a search space | `optimizer` / `scheduler` as tuned categoricals | `[week ]` |
| Random search baseline | RandomSearch optimizer | `[week ]` |

---

## Panel 6 — Results: Phase I Analysis + Test Score

Best **validation accuracy** found during HPO search vs. the official reference baseline
(`README.md`; reference obtained via "a rather simple HPO on a crudely constructed search
space" — an undisclosed budget/compute, i.e. a soft target, not a hard bar):

| Dataset | Classes | Reference test acc. (baseline) | Best val. acc. found (our HPO) |
|---|---|---|---|
| ag_news | 4 | 90.265% | `[fill in from final run]` |
| imdb | 2 | 86.993% | `[fill in from final run]` |
| dbpedia | 14 | 97.882% | `[fill in from final run]` |
| amazon | 3 | 81.799% | `[fill in from final run]` |
| **yelp** (exam test set) | 5 | 62.082% | `[fill in from final run]` |

**Final held-out test score on `yelp`**: `[X.XX%]`
**Number of test-branch submissions used**: `[N of ≤3]`

> ⚠️ Before printing: replace the placeholders above with numbers from the **final full
> retrain + held-out test evaluation** (`evaluate_incumbent` / `train_top_k_from_history.py`
> output), not validation-split numbers from the live search — those are two different
> quantities and mixing them up will misstate results on a printed poster.

**Framing note**: the exam explicitly rewards disciplined methodology over chasing the test
number — if `yelp` search hasn't converged as far as the practice datasets, say so plainly
("still tuning, here's the trajectory and why we expect it to close the gap with more
budget") rather than only showing a favorable number.

*(Suggested figure here: one `trial_plots.py` panel — e.g. val-error-vs-cumulative-time with
best-so-far starred — per dataset, or just for `yelp`.)*

---

## Panel 7 — Compute Budget & Resource Tracking

- **Hardware**: `[fill in final hardware — e.g. Apple M2 Max, 12-core CPU, MPS GPU, 32GB RAM / cluster GPU model + count]`, captured automatically per run in `device_info.json`.
- **Wall-clock spent per dataset**: `[pull from history.log.jsonl execution_time / run-folder timestamps]`, against the 24h ceiling.
- Every run also snapshots exact `pip freeze` output (`requirements.txt`) for full dependency reproducibility.
- **Cost-saving design decisions** (appropriate-use-of-compute evidence):
  - Tokenization caching: O(trials × corpus) → O(corpus).
  - Stratified `max_num_rows` subsampling caps per-trial cost independent of raw dataset size (`dbpedia`=560k rows, `yelp`=650k rows would otherwise dominate).
  - `max_seq_length` capped at 256 (down from an earlier 1024) after empirically finding most signal lives in the first ~256 tokens — a direct fidelity/cost-accuracy tradeoff.
  - Freeze-thaw checkpoint reuse (per-config-hash trainer checkpoints): "resuming" a candidate never repeats wasted compute from epoch 0.
  - `num_parallel_trials` (thread-pool, GPU round-robin): `[state N used]` — tradeoff is more parallelism = staler shared FT-PFN context per round.

---

## Panel 8 — Limitations & Future Work

- Only one **approach** (BiLSTM) is implemented end-to-end; the CLI still lists
  `tfidf-ffnn` / `transformer` / `tfidf-linear` / `bpe-rnn` but they aren't wired up — the
  variation explored was in HPO *strategy*, not model *family*.
- Preprocessing is currently just lowercasing; punctuation cleanup was scoped but not
  finished — legitimate future-work item.
- Exact reproducibility (same seed → same result) is deliberately relaxed under parallel
  trials: `set_seed()` is lock-protected, but the CPU-bound data prep after it isn't, to
  avoid starving GPUs. A documented, deliberate tradeoff, not an oversight.
- Future work: longer/full-budget ifBO search on `yelp`; finish punctuation-aware
  preprocessing; extend the approach registry beyond BiLSTM.

---

## Suggested panel order (matches example poster's flow: nutshell → method → space → results → footer)

1. In a nutshell
2. Method / architecture (one diagram)
3. Search space (table)
4. HPO methodology comparison (centerpiece — most space)
5. Lecture-concept mapping (can fold into panel 4's margin)
6. Results (Phase I table + yelp test score)
7. Compute budget
8. Limitations & future work (small, bottom corner)

## Figures to source before finalizing

- Freeze-thaw loop / start-vs-thaw diagram for the ifBO panel (hand-drawn or from `docs/IFBO_METHOD.md`).
- One `automl/trial_plots.py` output (val-error-vs-cumulative-time, best-so-far starred) per dataset or just `yelp`.
- Optional: `plot_future_predictions.py` calibration/coverage plot as an FT-PFN "we checked our surrogate's uncertainty" aside.
