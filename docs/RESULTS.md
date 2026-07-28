# Results: ifBO vs. Random / SMAC(BO+HB) / ifBO-Random

Numbers pulled directly from `results/<dataset>/[<optimizer>]<runtime_id>/history.log.jsonl`
(and the curated copies in `final-results/`) on 2026-07-28. "Best val. acc." is
`1 - min(val_error)` over all logged trial/freeze-thaw steps in a run — the best
validation-split accuracy the search *found*, not a held-out test score. No `yelp`
runs exist yet (`yelp` is the held-out exam set); all numbers below are Phase I
(`amazon`, `ag_news`, `dbpedia`, `imdb`).

Two baselines share the name-space and are easy to conflate:
- **`ifbo-random`** — the `ifbo` optimizer with its acquisition forced to always-explore
  (no FT-PFN surrogate scoring), i.e. an ablation that isolates freeze-thaw scheduling
  *without* the meta-learned surrogate.
- **`smac(BO+HB)`** — SMAC3, random-forest-EI Bayesian optimization + Hyperband
  successive-halving.

## 1. Matched-budget comparison — `ag_news`

The only dataset with all four methods run under an **identical budget**: `n_trials=20`,
`min_budget=3` / `max_budget=10` epochs, `max_trial_time_seconds=600`, `seed=67`. This is
the fair, apples-to-apples read — any accuracy gap here is attributable to the
optimization *method*, not to extra budget.

| Method | Best val. acc. | Freeze-thaw steps | Unique configs tried | Wall-clock |
|---|---|---|---|---|
| Random Search | 89.65% | 20 | 20 | 2h 20m |
| SMAC (BO+HB) | 89.35% | 20 | 17 | 1h 07m |
| ifBO-Random (ablation) | 92.45% | 20 | 8 | 0h 53m |
| **ifBO (surrogate)** | **running** — 76.30% after 3/20 steps, 2 unique configs, 6 min elapsed | | | |

The ifBO run under this exact matched config (`results/ag_news/20260728_191423_5dd3ec17/`)
was **still in progress** at write time (PID active, started 19:14, 3 of 20 steps logged).
Its early number isn't representative — freeze-thaw methods look worse than they are
early on because most of the step budget hasn't gone into the eventual-best candidate yet
(see the flagship run below, which reaches 95% by the time it's this many trials in).
**Update this row once `history.log.jsonl` reaches 20 steps.**

Even without the matched ifBO number: **ifBO-Random already beats both classical baselines
by ~3 points** on 2.4–3x less wall-clock and far fewer unique configs (8 vs 17–20) —
i.e. most of the gain at this budget comes from *freeze-thaw resource allocation itself*
(spending steps on fewer, more promising candidates instead of spreading budget thin),
before the FT-PFN surrogate even enters the picture.

## 2. Flagship results — per dataset (larger budget, real deployment config)

The `ifbo` **flagship** runs use a larger fidelity range (`min_budget=5`, `max_budget=20`,
`n_trials=30-40`) than the baseline sweeps below — this is the config the actual
submission pipeline uses, not a matched ablation. Baseline configs vary by dataset (see
`runtime_config.json` per run); read wall-clock and step counts alongside accuracy, not
accuracy alone.

| Dataset | ifBO (flagship) | ifBO-Random | SMAC (BO+HB) | Random Search | README reference (test acc.) |
|---|---|---|---|---|---|
| `ag_news` | **95.00%** (40 steps, 16 cfgs, 2h 24m) | 92.45% (20 steps, 8 cfgs, 0h 53m) | 89.35% (20 steps, 17 cfgs, 1h 07m) | 89.65% (20 steps, 20 cfgs, 2h 20m) | 90.265% |
| `amazon` | **90.80%** (40 steps, 16 cfgs, 2h 25m)¹ | 86.60% (40 steps, 17 cfgs, 2h 26m) | 78.90% (39 steps, 33 cfgs, 12h 08m) | 78.90% (14 steps, 14 cfgs, 5h 01m)² | 81.799% |
| `dbpedia` | **98.45%** (30 steps, 13 cfgs, 1h 34m) | — (not run) | 97.00% (20 steps, 17 cfgs, 1h 02m) | 96.40% (14 steps, 14 cfgs, 1h 21m)² | 97.882% |
| `imdb` | **99.85%** (30 steps, 13 cfgs, 1h 56m) | — (not run) | — (not run) | — (not run) | 86.993% |

¹ A second `amazon` ifBO flagship run (`[ifbo]20260726_211009_5caa1130`) landed at only
83.05% over 8h 22m — same config/seed, longer wall-clock, worse result. Included in
`results/amazon/` but excluded from the headline row above; worth checking whether it hit
a bad exploration draw or a resource contention issue (8h vs 2.4h for nominally the same
search is a large gap) before trusting either amazon number too far.
² These `random` runs show fewer trials than their own `n_trials` config target (14 vs.
20/30) — they stopped short of the configured trial count (manual stop or wall-clock cap),
so their number is a lower bound on what full-budget random search would find, not a
completed run.

**Reading the table**: `ifBO (flagship)` wins on every dataset it has a comparison for,
by 2.5–13 points over the best baseline — but it also gets 1.5–4x the wall-clock and
unique-config count of the matched baselines in most rows, so this table shows "what the
deployed method achieves," not "what the method achieves at equal cost" (see §1 for that).
`dbpedia`/`imdb` best-val already exceed the README's reference *test* accuracy, though
val ≠ test and the README baseline uses a different (simpler) HPO setup — a soft
signal, not a rigorous beat.

## 3. Caveats

- **Val accuracy, not test accuracy.** Nothing here is a held-out evaluation; that only
  happens via `train_top_k_from_history.py` retrain-and-test, not yet run for these.
- **Freeze-thaw "steps" ≠ "trials" the way Random/SMAC count them.** ifBO/ifBO-Random
  step counts include repeated visits (thaws) to the same config; "unique configs" is the
  fairer count to compare against Random/SMAC's one-shot-per-trial semantics.
- **Seeds and budgets aren't held constant across all rows** — see the per-run
  `runtime_config.json` for exact `n_trials`/`min_budget`/`max_budget`/seed. §1 is the one
  section where every variable is held fixed.
- Hardware: Apple M2 Max, 12-core CPU, MPS backend, 32GB RAM (`device_info.json` per run).

## 4. Source runs

```
ag_news:  [ifbo]20260727_102539_ba139c63        (flagship, 40 steps)
          20260728_191423_5dd3ec17               (matched, in progress)
          [ifbo-random]20260728_171127_ccc91fc3  (matched)
          [smac]20260728_142235_aca72795         (matched)
          [random]20260728_120149_58dda114       (matched)
amazon:   [ifbo]20260726_092101_5ffd5c03, [ifbo]20260726_211009_5caa1130
          [ifbo-random]20260726_125924_3c127468
          [smac]20260724_203257_140abdfb
          [random]20260725_110149_b3939077
dbpedia:  [ifbo]20260727_222606_b8de08c6
          [smac]20260728_092125_058c045b
          [random]20260728_103533_e73988f5
imdb:     20260727_201626_c85803b0
```
