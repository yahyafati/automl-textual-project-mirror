# Poster Content — Freeze-Thaw HPO with an FT-PFN Surrogate

Content draft, laid out panel-by-panel to match the structure of `poster-example.ignore.pdf`
("Towards Automatically-Tuned Neural Networks", ML4AAD/Uni Freiburg poster: header band →
"In a nutshell" strip → 4–5 wide content panels → figures with captions → footer branding).

**Framing (read this before editing panels below)**: this is an **AutoML project**, not a
text-classification project. The contribution being graded is the *hyperparameter
optimization system* — how it decides which configuration to train, for how long, and when
to stop — not the accuracy of a BiLSTM. Text classification is the **testbed**, used to
generate real learning curves the optimizer has to make decisions on; keep it in the
background. Every panel below leads with the AutoML concept (multi-fidelity optimization,
freeze-thaw scheduling, meta-learned surrogates, exploration/exploitation, compute-aware
design, ensembling) and only reaches for text-classification specifics when a concept needs
a concrete anchor. Drop each panel's text into the corresponding box of the official poster
template. Anything in `[brackets]` is a placeholder to fill in before printing.

---

## Header band

**Title** (pick one — see `docs/TITLE_OPTIONS.md` for more options):
> Freeze-Thaw HPO with an FT-PFN Surrogate: Multi-Fidelity AutoML Under a Fixed Compute Budget

**Authors**: `[Name 1, Name 2, ...]`
**Affiliation**: Department of Computer Science, University of Freiburg — SS26 AutoML Exam
**Contact**: `[emails]`

---

## Panel 1 — In a nutshell

- **The AutoML question we address**: given a fixed compute budget and a hyperparameter
  search space too large to explore exhaustively, how do you decide *which configurations
  deserve more training and which should be abandoned early* — without wasting budget
  training every candidate to completion just to find out?
- Our answer is **multi-fidelity, freeze-thaw Bayesian optimization**: configurations are
  trained only *partially*, in parallel, free to pause ("freeze") or resume ("thaw") at any
  point. The candidate pool isn't fixed or handed in upfront — it **grows dynamically**:
  with a decaying exploration probability, sample a brand-new configuration; otherwise
  resume whichever already-started candidate a **pretrained meta-learned surrogate
  (FT-PFN)** judges most likely to improve, based on its partial learning curve so far.
- **Efficiency is a first-class design objective**, not an afterthought: every expensive
  step in the pipeline (tokenization, embedding init, per-trial training) is cached,
  checkpointed, or fidelity-capped specifically to fit more HPO trials inside the budget.
- The testbed problem: 5 text-classification datasets, one held out (`yelp`) as the final
  exam evaluation — used only to give the optimizer real, noisy learning curves to reason
  about, not the object of study itself.
- Single-command pipeline: `python -m automl --config runconfig.yml`.

---

## Panel 2 — The AutoML Problem & System Design

**Formal picture**: at each step, the optimizer chooses an action `a ∈ {start a new
configuration θ, resume ("thaw") a partially-trained configuration}` and a **fidelity**
(how many more training steps to spend), trying to maximize final validation performance
under a total-step budget `B`. This is the freeze-thaw variant of multi-fidelity HPO —
fidelity is continuous and *interruptible*, not fixed brackets chosen up front.

Three cleanly separated, independently swappable layers — built this way specifically so
four HPO strategies could be **fairly compared** against the same model/data code, rather
than each optimizer bringing its own bespoke training loop. The **optimizer layer is the
actual research contribution**; approach/trainer exist to give it something real to search
over.

```
CLI (RuntimeConfig: defaults → YAML → CLI flags)
        │
        ▼
Optimizer   ★ the contribution — decides WHICH config to try next and for HOW LONG
        │        implementations: RandomSearch · SMAC (Hyperband) · RL-Freeze-Thaw (PPO) · ifBO ★★
        ▼
Approach    ── turns a config into a concrete model + data pipeline (the testbed)
        │        implementation: sequence-dl = BiLSTM over a DistilBERT WordPiece vocabulary
        ▼
Trainer     ── generic PyTorch train loop with checkpoint/resume — the mechanism
                 that makes "thaw" actually possible (resume ≠ restart)
```

**Talking point**: "We didn't just pick one HPO method — we built the harness so we could
swap optimizers and prove ifBO wins for a reason, not by accident of implementation."

### Testbed model (kept deliberately simple — the optimizer is where the complexity lives)

- BiLSTM classifier over a DistilBERT WordPiece vocabulary (tokenizer only, not the
  DistilBERT model), with an SVD-projected pretrained-embedding warm start. Chosen because
  it's cheap enough to train **hundreds of times** per dataset — a necessary condition for
  any multi-fidelity method to have enough trials to reason over. A heavier model
  (transformer fine-tune) would have starved the optimizer of trials within budget and
  turned this into a single-config training exercise instead of an HPO one.
- `[N]` trials / `[N]` freeze-thaw steps executed for the reported `yelp` run — cite this
  number on the poster as evidence the budget bought a genuinely multi-trial search, not a
  handful of expensive runs.

---

## Panel 3 — Search & Fidelity Space Design

Two design questions every multi-fidelity method needs answered, and how we answered them:

**1. What is the fidelity dimension (the "cheap-but-informative" axis)?**
Training epochs / steps — cheap to extend incrementally, and the standard choice for
learning-curve-based methods (this is *why* freeze-thaw applies at all: partial training
is informative about final performance). `min_budget`/`max_budget` bound it per run.

**2. What is the configuration space the optimizer searches over?**
11 tuned hyperparameters, `ConfigSpace`-defined, capped at FT-PFN's architectural limit of
10 encoded dimensions (`model_type` dropped as constant):

| Hyperparameter | Range / Choices | Why it's in the space |
|---|---|---|
| `hidden_dim` | {32, 64, 128, 256} | capacity vs. overfitting/compute tradeoff |
| `seq_num_layers` | 1–3 | depth — lightweight architecture search |
| `seq_embed_dim` | 32–512 (log) | representation capacity |
| `dropout` | 0.0–0.5 | regularization |
| `learning_rate` | 1e-4–1e-2 (log) | most sensitive hyperparameter empirically |
| `weight_decay` | 1e-6–1e-2 (log) | regularization |
| `optimizer` | {adam, adamw, sgd} | **algorithm selection** inside the space |
| `scheduler` | {steplr, cosine, exponential, reduce-on-plateau} | LR-schedule selection |
| `warmup_ratio` | 0.0–0.2 | early-training stability |
| `batch_size` | {32, 64, 128, 256} | compute / statistical-efficiency tradeoff |
| `max_seq_length` | {64, 128, 256} | a **second, orthogonal cost knob** alongside the epoch-fidelity axis — capped from an earlier 1024 after finding packed-LSTM cost scales ~linearly with token count |

**Talking point**: every bound has a stated reason — directly answers the "search strategy
justification" rubric line. Note `optimizer`/`scheduler` are tuned as *categoricals*, i.e.
algorithm selection is folded into the same HPO loop rather than fixed by hand.

---

## Panel 4 — HPO Methodology (centerpiece — most weight, most poster space)

Four optimizers, four different answers to "how do you allocate a fixed budget across many
candidates," implemented on the *same* model/data harness so they're directly comparable:

| Optimizer | AutoML paradigm | Fidelity handling | Parallel? | Ensembling |
|---|---|---|---|---|
| Random Search | naive baseline | fixed, full budget every trial | no | no |
| SMAC | classical Bayesian opt. (random-forest EI) + Hyperband | successive-halving brackets | no | no |
| RL Freeze-Thaw | learned scheduling policy (PPO, from scratch) | fixed candidate pool, discrete 1/2/4/8-epoch start/thaw actions | no | no (single incumbent) |
| **ifBO ★ (submitted)** | **in-context meta-learned surrogate BO** | continuous freeze-thaw steps, dynamic pool | **yes** (thread pool, GPU round-robin) | **yes** (top-k threshold, majority vote) |

This table alone demonstrates **multi-fidelity optimization, Bayesian optimization,
meta-learning, and reinforcement learning** applied to the *same* resource-allocation
problem — so their tradeoffs are measured, not asserted.

### Multi-fidelity, the freeze-thaw way

- **Successive halving / Hyperband (SMAC)** commits to fixed brackets: allocate a budget,
  train a batch of configs to that budget, discard the worst half, repeat. Fidelity levels
  are pre-defined and coarse.
- **Freeze-thaw (RL and ifBO)** removes that rigidity: any partially-trained configuration
  can be frozen (paused) and thawed (resumed) at any point, one training chunk at a time,
  based on an online judgment of "is this curve still worth it?" — a strictly more flexible
  fidelity schedule, at the cost of needing a *cheap-to-query* judgment mechanism at every
  step. That's exactly the role FT-PFN plays for ifBO, and the PPO policy for RL-freeze-thaw.

### Why ifBO is the flagship method

Reference: Rakotoarison et al., *In-Context Freeze-Thaw Bayesian Optimization for
Hyperparameter Optimization*, ICML 2024 (arXiv:2404.16795).

- **The surrogate (FT-PFN) is the meta-learning core of the whole project**: a Prior-data
  Fitted Network — a transformer pretrained *once, offline*, on millions of *synthetic*
  learning curves (power-law / exponential / breaking-point shapes), never touching our
  data. At HPO time no weights are updated — observed curves so far are fed in as
  **in-context tokens**, and one forward pass approximates the Bayesian posterior over "how
  will this curve continue?" (10–100× cheaper per acquisition than refitting-based
  surrogates like DPL/DyHPO). This is knowledge about *how training curves behave in
  general*, transferred zero-shot into our search — the textbook meta-learning move.
- **Acquisition — "dynamic epsilon-greedy MFPI-random"**: with probability ε (polynomial
  decay toward 0.1 over the run) sample a brand-new candidate (**explore**); otherwise pick
  among pending candidates via Multi-Fidelity Probability-of-Improvement with a
  **randomized lookahead horizon** (1–3 steps) and **randomized improvement target**
  (log-uniform 1e-4–1e-1) (**exploit**, with built-in diversity so the search doesn't
  always chase one fixed target).
- **Ensembling as a multi-objective step beyond accuracy-maximization**: keeps every
  candidate within a fixed accuracy threshold of the best observed (top-k, threshold-gated)
  and majority-votes their test predictions, instead of betting everything on a single
  incumbent — a small robustness gain "for free" from the search history.
- **Parallelism**: batch-synchronous — one shared FT-PFN context per round, up to N
  distinct candidates selected against it, dispatched concurrently across GPUs. The
  resulting staleness for candidates 2..N is a documented, *accepted* standard-batch-BO
  tradeoff, not a bug — a "we understand the tradeoff we made" talking point.

*(Suggested figure here: freeze-thaw loop diagram — start vs. thaw actions, FT-PFN
in-context surrogate, MFPI-random acquisition. Or reuse a `trial_plots.py` learning-curve
panel from a real run — e.g. the epoch heatmap or Gantt-style config-hash-vs-time chart,
which visually *is* the freeze-thaw schedule.)*

### Validation protocol (experimental rigor)

- Stratified train/val split per trial; stratified subsampling bounds trial cost while
  preserving class balance — a controlled-cost design choice, not incidental.
- Final incumbent(s) retrained **from scratch on the full training set** for a larger
  evaluation budget, then evaluated on held-out test data — validation selects, held-out
  test reports, never the reverse.
- The ensembling protocol is independently reproducible standalone from a frozen trial-history
  log (`train_top_k_from_history.py`) — the final submission artifact doesn't depend on a
  live run, a reproducibility strength worth stating explicitly.
- Random Search runs on the identical harness as every other method — any gap over it is
  attributable to the optimization *method*, not to different code paths.

---

## Panel 5 — Lecture-Concept Mapping

*(Exam requirement: "denote on poster the weeks from which concepts were used." Fill in
your course's actual week numbers against each row before printing.)*

| AutoML concept | Where in our system | Lecture week |
|---|---|---|
| Multi-fidelity optimization / successive halving / Hyperband | SMAC(Hyperband) optimizer | `[week ]` |
| Bayesian optimization (surrogate + acquisition function) | SMAC (random forest EI) and ifBO (FT-PFN + MFPI-random) | `[week ]` |
| Meta-learning / learning-curve extrapolation / transfer across tasks | FT-PFN surrogate (pretrained offline on synthetic curves); SVD-projected pretrained embedding warm start | `[week ]` |
| Reinforcement learning for scheduling/control | From-scratch PPO freeze-thaw controller | `[week ]` |
| Ensemble methods | Top-k majority-vote ensembling (ifBO incumbent selection) | `[week ]` |
| Algorithm/model selection within a search space | `optimizer` / `scheduler` as tuned categoricals | `[week ]` |
| Random search as a baseline | RandomSearch optimizer | `[week ]` |
| Exploration/exploitation tradeoff | ε-greedy candidate sampling in ifBO's acquisition | `[week ]` |

---

## Panel 6 — Testbed Evaluation (secondary panel — validates the method works, not the headline)

The optimizer's job is to find good configurations under budget; this panel shows it did,
on real (noisy, imbalanced, variable-length) text data. Best **validation accuracy** found
during HPO search vs. the official reference baseline (`README.md`; obtained via "a rather
simple HPO on a crudely constructed search space" — a soft target, not a hard bar):

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
> output), not validation-split numbers from the live search — mixing those up will
> misstate results on a printed poster.

**Framing note**: the exam explicitly rewards disciplined methodology over chasing the test
number. If it's true for your timeline, this is good material: "we deliberately didn't
over-invest search budget chasing the `yelp` number early, to avoid overfitting design
choices to Phase I data — we validated methodology breadth (four optimizers) before
committing full compute to the final dataset." Otherwise present the trajectory honestly.

*(Suggested figure here: one `trial_plots.py` panel — val-error-vs-cumulative-time with
best-so-far starred — for `yelp`, since it's also a multi-fidelity-search visualization,
not just a results chart.)*

---

## Panel 7 — Efficiency & Compute-Aware Design (appropriate use of compute)

Efficiency isn't just "we didn't waste money" — it's what *let the multi-fidelity search
have enough trials to work with* in the first place. Each item below is a deliberate
cost/accuracy tradeoff, tied to what it bought the optimizer:

- **Tokenization caching** (O(trials × corpus) → O(corpus)): every trial resamples the same
  fixed text pool, so full-text tokenization runs once, not once per trial — reclaims budget
  that would otherwise be spent redoing identical work every single freeze-thaw step.
- **Freeze-thaw checkpoint reuse** (per-config-hash trainer checkpoints): "thaw" literally
  *is* this mechanism — resuming a candidate never repeats wasted compute from epoch 0. No
  checkpointing, no freeze-thaw.
- **Stratified `max_num_rows` subsampling**: bounds per-trial cost independent of raw
  dataset size (`dbpedia`=560k rows, `yelp`=650k rows would otherwise dominate the budget on
  a handful of trials), while preserving class balance.
- **`max_seq_length` capped at 256** (down from an earlier 1024): an explicit fidelity/cost
  ablation — found most classification signal lives in the first ~256 tokens, so the extra
  compute bought little accuracy.
- **`num_parallel_trials`** (thread-pool, GPU round-robin): `[state N used]` more trials per
  wall-clock hour, at the cost of a slightly stale shared FT-PFN context per round — an
  explicit parallelism/staleness tradeoff, not free.
- **Hardware & tracking**: `[fill in final hardware — e.g. Apple M2 Max, 12-core CPU, MPS
  GPU, 32GB RAM / cluster GPU model + count]`, captured automatically per run in
  `device_info.json`; exact `pip freeze` snapshot per run for dependency reproducibility.
- **Wall-clock spent per dataset**: `[pull from history.log.jsonl execution_time /
  run-folder timestamps]`, against the 24h ceiling — state this explicitly on the poster.

---

## Panel 8 — Limitations & Future Work

- Only one testbed **model** (BiLSTM) is wired up end-to-end; the variation we explored was
  in HPO *strategy* (four optimizers), not model *family* — say this plainly if asked
  "did you compare architectures," rather than implying a NAS-style search happened.
- **ifBO's parallel dispatch trades exact reproducibility for throughput**: `set_seed()` is
  lock-protected but the CPU-bound data prep after it isn't, to avoid starving GPUs under
  `num_parallel_trials > 1` — a documented, deliberate tradeoff (ifBO already treats
  accuracy as a noisy observation), not an oversight.
- Batch-synchronous parallelism means candidates 2..N in a round are selected against a
  slightly stale FT-PFN context — accepted standard batch-BO behavior.
- Future work: longer/full-budget ifBO search on `yelp`; a learned (rather than fixed) 
  lookahead-horizon/target distribution for MFPI-random; extending the harness to a second
  testbed model to test whether the same optimizer ranking holds.

---

## Suggested panel order (matches example poster's flow: nutshell → method → space → results → footer)

1. In a nutshell (AutoML framing first, testbed mentioned last)
2. The AutoML problem & system design (one diagram; testbed model kept as a small aside)
3. Search & fidelity space design (table)
4. HPO methodology comparison (centerpiece — most space, most time in the pitch)
5. Lecture-concept mapping (can fold into panel 4's margin)
6. Testbed evaluation (secondary — Phase I table + yelp test score)
7. Efficiency & compute-aware design
8. Limitations & future work (small, bottom corner)

## Figures to source before finalizing

- Freeze-thaw loop / start-vs-thaw diagram for the ifBO panel (hand-drawn or from `docs/IFBO_METHOD.md`).
- A `automl/trial_plots.py` Gantt-style config-hash-vs-time chart or epoch heatmap — this
  *is* a picture of the freeze-thaw schedule in action, arguably the single most
  AutoML-relevant figure available.
- One `trial_plots.py` val-error-vs-cumulative-time panel (best-so-far starred) for `yelp`.
- Optional: `plot_future_predictions.py` calibration/coverage plot as a "we checked our
  surrogate's uncertainty calibration" aside — direct evidence FT-PFN's forecasts are
  trustworthy, not just fast.
