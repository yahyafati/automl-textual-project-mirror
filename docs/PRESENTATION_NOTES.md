# Presentation Prep — Text AutoML (SS26 Exam)

Working notes to build the poster + talking points from. Organized around the exam's
own grading rubric (`SS26_AutoML_Exam-textual.ignore.md`) so nothing you present is
disconnected from what's actually being graded. Pull sections directly into the poster
template; use the "talking points" as spoken narration during Q&A.

---

## 0. One-paragraph summary (elevator pitch)

We built an AutoML pipeline for text classification that treats **model training
itself as a resource to be spent incrementally**: instead of picking a hyperparameter
configuration and training it to completion before judging it, our optimizer
(**ifBO** — in-context freeze-thaw Bayesian optimization) trains many configurations
*partially and in parallel*, uses a pretrained transformer surrogate (FT-PFN) to
predict which partial learning curve is most likely to keep improving, and spends
the remaining budget only on the most promising ones. The pipeline is a single
`python -m automl` command, wraps a BiLSTM classifier with an 11-dimensional tuned
search space, and produces a top-k majority-vote ensemble as the final prediction.

---

## 1. What the exam actually asks for (context for framing everything else)

- **Goal**: best top-1 accuracy on a held-out `yelp` test set, but graded primarily
  on **methodological rigor**, not the number itself.
- **Budget**: ≤ 24h wall-clock for "given a new dataset → test predictions"
  (final full retrain + eval excluded from that budget).
- **Grading weights** (from the spec, use these as your poster's section headers):
  1. **Scientific rigor in HPO methodology** (High weight)
  2. **Creative use of AutoML concepts** (High weight) — multi-fidelity opt, meta-learning,
     NAS-ish, algorithm selection, ensembling
  3. **Appropriate use of compute resources** — cost tracking + cost/performance tradeoff
- **Explicit requirement**: "denote on poster the weeks from which concepts were used" —
  you'll need to map the bullet list in §4 back to your course's lecture schedule
  (multi-fidelity HPO, Bayesian optimization / surrogates, meta-learning, RL, ensembling).
- **Trap to avoid mentioning you avoided**: "overfitting to Phase I practice data" and
  "test score rabbit hole" (≤3 test submissions recommended) — good material for a
  "we were disciplined about X" talking point.

---

## 2. System architecture (poster: "Method" section)

```
CLI (RuntimeConfig: defaults → YAML → CLI flags)
        │
        ▼
Optimizer  ── decides WHICH config to try next and for HOW LONG (fidelity/budget)
        │        implementations: RandomSearch, SMAC(Hyperband), RL-Freeze-Thaw(PPO), ifBO ★
        ▼
Approach   ── turns a config into a concrete model + data pipeline
        │        implementation: sequence-dl = BiLSTM over a DistilBERT WordPiece vocab
        ▼
Trainer    ── generic PyTorch train loop (optimizer/scheduler, resume, checkpointing)
```

Three cleanly separated, independently swappable layers — this separation is itself
a design decision worth stating explicitly: it's what let us implement and *fairly
compare* four different HPO strategies against the same model/data code, rather than
each optimizer having its own bespoke training loop.

**Talking point**: "We didn't just pick one HPO method — we built the harness so we
could swap optimizers and prove ifBO is winning for a reason, not by accident of
implementation."

### 2.1 Text representation & model (poster: "Architecture choice" + justification)

- **Tokenizer**: DistilBERT's pretrained WordPiece vocabulary, used *only* for
  subword tokenization — not the DistilBERT model itself. Vendored locally
  (`./tokenizers/distilbert-base-uncased`) so HPO trials never hit the network.
- **Model**: bidirectional LSTM (`BiLSTMClassifier`) — chosen over a full transformer
  fine-tune because (a) the exam footnote explicitly states these datasets are
  "solvable without large-scale transformers," (b) a BiLSTM is cheap enough to train
  hundreds of times inside a 24h budget across 5 datasets, which a transformer
  fine-tune is not, and (c) it leaves the tunable-hyperparameter budget free for
  architecture/optimization choices rather than "which frozen layers to unfreeze."
- **Pretrained embedding warm start**: since the tuned embedding dimension
  (`seq_embed_dim`, searched 32–512) essentially never equals DistilBERT's native
  768-dim embedding table, we SVD-project the pretrained embedding matrix onto its
  top-`k` principal directions before initializing the BiLSTM's embedding layer —
  cheap, done once per unique `(vocab, target_dim)` via `lru_cache`, and gives a much
  better init than random. This *is* a transfer-learning component even though the
  final classifier is not a pretrained transformer.
- **Efficiency engineering** (good "appropriate use of compute" material):
  - Full-text tokenization cached once across all HPO trials (same fixed text pool
    resampled every trial) — turns O(trials × corpus) tokenization into O(corpus).
  - `pack_padded_sequence` so the LSTM never computes over padding.
  - Dynamic per-batch padding (pad to the batch's longest sequence, not a global max).
  - Token ids stored as one contiguous `int32` buffer (not a Python list of tensors)
    to avoid DataLoader worker copy-on-write memory multiplication.

### 2.2 Search space (poster: "Hyperparameter space" + justification)

11 tuned hyperparameters, `ConfigSpace`-defined:

| Hyperparameter | Range | Why it's in the space |
|---|---|---|
| `hidden_dim` | {32,64,128,256} | LSTM capacity vs. overfitting/compute tradeoff |
| `seq_num_layers` | 1–3 | depth — a lightweight stand-in for architecture search |
| `seq_embed_dim` | 32–512 (log) | representation capacity, drives the SVD projection target |
| `dropout` | 0.0–0.5 | regularization, esp. important on small/noisy classes |
| `learning_rate` | 1e-4–1e-2 (log) | single most sensitive hyperparameter empirically |
| `weight_decay` | 1e-6–1e-2 (log) | regularization |
| `optimizer` | {adam, adamw, sgd} | algorithm selection within the space |
| `scheduler` | {steplr, cosine, exponential, reduce-on-plateau} | LR-schedule selection |
| `warmup_ratio` | 0.0–0.2 | training stability early on |
| `batch_size` | {32,64,128,256} | compute/statistical-efficiency tradeoff |
| `max_seq_length` | {64,128,256} | **the key fidelity/cost knob** — capped down from an earlier 1024, since packed-LSTM cost scales ~linearly with token count and most signal in a review lives in the first ~256 tokens |

**Talking point**: every bound has a stated reason — this directly answers the
"Search strategy justification" rubric line.

---

## 3. HPO methodology — the scientific-rigor core of the poster

We implemented and empirically compared **four** optimizers on the *same* model/data
harness:

| Optimizer | Surrogate/model | Fidelity handling | Parallel? | Ensembling |
|---|---|---|---|---|
| Random Search (baseline) | none | fixed, full budget every trial | no | no |
| SMAC (BOHB-style) | random-forest EI (SMAC3) | Hyperband successive halving | no | no |
| RL Freeze-Thaw (from scratch) | PPO policy (~100 lines, no Gym/SB3) | fixed candidate pool, 1/2/4/8-epoch start actions + thaw actions | no | no (single incumbent) |
| **ifBO** ★ (submitted) | **FT-PFN** — pretrained in-context transformer | continuous freeze-thaw steps, dynamic pool | **yes** (thread pool, GPU round-robin) | **yes** (top-k threshold, majority vote) |

This table alone demonstrates "creative use of AutoML concepts" — multi-fidelity
optimization, Bayesian optimization, meta-learning (a network pretrained offline on
synthetic curves), and reinforcement learning, all applied to the *same* problem so
their tradeoffs are directly comparable rather than asserted.

### 3.1 Why ifBO is the flagship method (poster: main HPO panel)

Reference: Rakotoarison et al., *In-Context Freeze-Thaw Bayesian Optimization for
Hyperparameter Optimization*, ICML 2024 (arXiv:2404.16795). Full technical writeup in
`docs/IFBO_METHOD.md` — use that as your source of truth for equations/details.

- **Problem framing**: rather than "pick config → train to completion → observe one
  number," freeze-thaw HPO spends a *budget of training steps* incrementally: at each
  iteration, either start a new config or resume ("thaw") a partially-trained,
  currently-frozen one for one more chunk of training.
- **The surrogate (FT-PFN)**: a *Prior-data Fitted Network* — a transformer trained
  once, offline (not by us — pretrained weights loaded from the `ifbo` PyPI package),
  on millions of *synthetic* learning curves sampled from a hand-designed generative
  prior (power-law / exponential / breaking-point curve shapes). At HPO time, no
  weights are updated — the observed learning curves so far are just fed in as
  **in-context** tokens, and one transformer forward pass approximates the Bayesian
  posterior over "how will this curve continue?" This is what makes each acquisition
  step cheap enough to run every freeze-thaw round (paper reports 10-100× speedup over
  refitting-based surrogates like DPL/DyHPO).
- **This is a meta-learning component** — the surrogate encodes prior knowledge about
  *how neural network training curves tend to look*, learned from a totally separate
  synthetic-data process, then transferred zero-shot to our BiLSTM/text problem.
- **Acquisition strategy** (our specific implementation choice, worth narrating):
  "dynamic epsilon-greedy MFPI-random" — with probability ε (polynomial decay toward
  0.1 over the run) sample a fresh candidate (exploration); otherwise pick among
  pending candidates via Multi-Fidelity Probability-of-Improvement with a
  **randomized lookahead horizon** (1–3 steps) and **randomized improvement target**
  (log-uniform 1e-4–1e-1) — the randomization is a deliberate diversity mechanism so
  the acquisition doesn't always chase the same fixed-horizon target.
- **Ensembling**: keeps all candidates within a fixed accuracy threshold of the best
  observed (top-k, threshold-gated) and majority-votes their test predictions, rather
  than committing to a single incumbent — a small, principled multi-objective/
  robustness step beyond raw accuracy-maximization.
- **Parallelism**: batch-synchronous — builds one shared FT-PFN context per round,
  selects up to N distinct candidates against it, dispatches them concurrently across
  GPUs via a thread pool. The staleness this introduces (candidates 2..N selected
  against a slightly-outdated context) is accepted and documented as standard batch-BO
  behavior, not treated as a bug — a good "we understand the tradeoff we made"
  talking point.

### 3.2 Validation protocol / experimental design

- Stratified train/val split per trial; stratified subsampling (`max_num_rows`) to
  bound trial cost while preserving class balance — matters especially for
  imbalanced datasets like `yelp` (5-way, naturally skewed) and `dbpedia` (14 classes).
- Final incumbent(s) are **retrained from scratch on the full training set** for a
  larger `evaluation_budget`, then evaluated on true held-out test data — the
  standard "validation selects, full data + held-out test reports" separation.
- `train_top_k_from_history.py` independently re-derives the top-k distinct configs
  from a saved trial history and reproduces the same majority-vote ensembling
  protocol standalone — i.e., the *final submission* artifact is producible from a
  frozen log, not only from a live run, which is a reproducibility strength worth
  stating explicitly.

---

## 4. Mapping to lecture concepts (fill in your course's actual week numbers)

Use this list as a checklist against your syllabus — I don't have your week-by-week
schedule, so match each bullet to whichever week covered it:

- Multi-fidelity optimization / successive halving / Hyperband → SMAC panel
- Bayesian optimization (surrogate + acquisition function) → SMAC (random forest) and
  ifBO (FT-PFN) both instantiate this generically
- Meta-learning / learning-curve extrapolation / transfer of a pretrained model
  across tasks → FT-PFN itself, plus the SVD-projected embedding warm start
- Reinforcement learning for scheduling/control → the from-scratch PPO freeze-thaw
  controller
- Ensemble methods → top-k majority-vote ensembling (ifBO's incumbent selection)
- Algorithm/model selection within a search space → `optimizer`/`scheduler` choices
  as first-class tuned categoricals, not fixed defaults
- Random search as a baseline → required, implemented, and comparable on the same
  harness

---

## 5. Results so far (poster: "Analysis of Phase I datasets" + test score)

Best **validation accuracy observed during HPO search** (i.e. the incumbent found by
search, not yet the final full-retrain test number) on each Phase I dataset, vs. the
official reference baseline from `README.md`:

| Dataset | Classes | Reference baseline (test acc.) | Best val. acc. found (this HPO run) |
|---|---|---|---|
| ag_news | 4 | 90.27% | **94.8%** |
| imdb | 2 | 86.99% | **95.96%** |
| dbpedia | 14 | 97.88% | **98.54%** |
| amazon | 5 | 81.80% | **93.3%** |
| **yelp** (exam/test dataset) | 5 | 62.08% | **~58.5%** (best HPO-search run so far; see caveat below) |

⚠️ **Important caveats to sort out before you finalize poster numbers:**

1. These are **validation-split** accuracies from the HPO search itself, not the
   final held-out **test**-split numbers from a full retrain — get the actual
   `evaluate_incumbent` / `train_top_k_from_history.py` test-set numbers before
   printing final figures on the poster.
2. On `yelp` specifically, the strongest completed search run in the repo currently
   sits **below** the reference baseline (58.5% vs 62.1%) — several other `yelp` run
   folders in `results/yelp/` are 1-trial debug runs, not real searches, so don't
   average those in. This is worth a real conversation before the poster: either (a)
   run ifBO longer/at full budget on `yelp` before the deadline, since `yelp` is
   harder (5-way, noisiest of the five datasets, per the sequence-length stats in
   `README.md`), or (b) present it honestly as "still tuning, here's the trajectory
   and why we expect it to close the gap with more budget" — the exam explicitly says
   modest test performance with excellent methodology outscores the reverse.
3. I found two `train_top_k_from_history.py` output folders, `topk_results/` and
   `yelp-topk-results/` — **both say `"dataset": "imdb"` inside their `manifest.json`**,
   not yelp, despite the second folder's name. Double check which folder actually holds
   your intended yelp submission artifacts before copying anything into
   `data/exam_dataset/predictions.npy` — as-is, neither looks like the final yelp
   submission.

**Talking point that turns caveat #2 into a strength**: "We deliberately didn't
over-invest search budget chasing the yelp number early, to avoid the exam's explicitly
flagged pitfall of overfitting design choices to Phase I data — we validated methodology
breadth (4 optimizers) before committing full compute to the final dataset."
(Only use this if it's true for your actual timeline — adjust honestly.)

---

## 6. Compute budget & resource tracking (poster: required "documentation of computational resources")

- **Hardware actually used for development**: Apple M2 Max (12-core CPU, integrated
  GPU via MPS), 32GB RAM — captured automatically per-run in
  `device_info.json` (never-raises design: every section independently
  try/excepted, since environment logging must never block a training run).
- Every run also snapshots exact `pip freeze` output (`requirements.txt`) — full
  reproducibility of the exact dependency versions used, not just the code.
- **Cost-saving design decisions to cite as "appropriate use of compute"**:
  - Tokenization caching (O(trials × corpus) → O(corpus))
  - `max_num_rows` stratified subsampling caps per-trial cost independent of raw
    dataset size (`dbpedia`=560k rows, `yelp`=650k rows would otherwise dominate)
  - `max_seq_length` capped at 256 (down from an earlier 1024) after empirically
    finding most signal lives in the first ~256 tokens — a direct fidelity-dimension
    cost/accuracy tradeoff decision, worth showing as a mini ablation if you have the
    old-vs-new numbers
  - Freeze-thaw checkpoint reuse (per-config-hash trainer checkpoints) means
    "resuming" a candidate never repeats wasted compute from epoch 0
  - `num_parallel_trials` (thread-pool, GPU round-robin) — parallel dispatch was
    engineered but is genuinely optional; state whatever `N` you used for the final
    run and why (tradeoff: more parallelism = staler shared FT-PFN context per round)
- **State the budget math explicitly on the poster**: total wall-clock spent per
  dataset for the reported HPO run (pull from each run's directory timestamp deltas
  or `execution_time` fields in `history.log.jsonl`), against the 24h ceiling.

---

## 7. Honest limitations / anticipate these questions

Good practice per the grading rubric ("methodological rigor" rewards knowing your own
system's edges, not hiding them). Keep a short "limitations" box on the poster or in
your back pocket for Q&A:

- Only one **approach** (BiLSTM) is actually implemented end-to-end; the CLI still
  lists `tfidf-ffnn`/`transformer`/`tfidf-linear`/`bpe-rnn` as options but they aren't
  wired up. If asked "did you compare architectures," the honest answer is: the
  variation we explored was in HPO *strategy*, not model *family* — say this plainly
  rather than implying an architecture search happened.
- Preprocessing is currently just lowercasing (punctuation cleanup was scoped but not
  finished) — a legitimate "future work" bullet.
- Exact-reproducibility (same seed → same result) is deliberately relaxed under
  parallel trials — `set_seed()` is lock-protected but the CPU-bound data prep after
  it isn't, to avoid starving GPUs. This is a documented, deliberate tradeoff, not an
  oversight — say so if asked about determinism.
- `run.py`, which `README.md` documents as the main entrypoint, does not exist on this
  branch — the real entrypoint is `python -m automl` / `make run`. Make sure
  `run_instructions.md` (a required submission file) points at commands that actually
  exist on the branch you submit.

---

## 8. Suggested poster narrative arc (order of panels)

1. **Motivation / task**: 5-dataset text classification, 24h budget, methodology > raw score.
2. **Architecture**: 3-layer pluggable design (optimizer / approach / trainer) — one
   diagram (§2 above).
3. **Search space**: the 11-hyperparameter table with one-line justifications (§2.1).
4. **HPO methodology**: the 4-optimizer comparison table (§3) as the centerpiece —
   this is where most of your grading weight lives. Spend the most poster real estate
   and presentation time here.
5. **ifBO deep dive**: one diagram of the freeze-thaw loop (start vs. thaw actions,
   FT-PFN in-context surrogate, MFPI-random acquisition).
6. **Results**: Phase I dataset table + yelp trajectory (§5), framed honestly per the
   caveats above.
7. **Compute budget**: hardware, wall-clock per dataset, cost-saving engineering (§6).
8. **Limitations & future work** (§7) — brief, confident, not defensive.

---

## 9. Likely Q&A and short answers

- **"Why not just fine-tune a transformer?"** → footnote 2 of the exam spec says these
  datasets are solvable without large transformers; a BiLSTM lets us afford hundreds
  of HPO trials across 5 datasets within budget, which fine-tuning repeatedly would not.
- **"Why ifBO over SMAC/Hyperband?"** → same multi-fidelity idea, but ifBO's surrogate
  is a meta-learned in-context model instead of a refit-every-time random forest —
  cheaper per acquisition and (per the original paper) empirically stronger at low
  budgets, which matters under a 24h ceiling.
- **"How do you know your HPO methodology actually helps?"** → because Random Search
  is implemented on the exact same harness as a baseline — any gap over Random Search
  is attributable to the method, not to different code paths.
- **"What guards against overfitting to Phase I data?"** → held-out test evaluation
  after HPO selects on validation only; final incumbent retrained from scratch on
  full data before touching test; explicit awareness of the exam's own warning not to
  over-invest in practice-dataset quirks.
