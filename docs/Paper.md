## Problem formulation: freeze-thaw hyperparameter optimization

Let $\Lambda$ be a hyperparameter search space and $f(\lambda, b)$ the performance (validation
accuracy, in this project) of configuration $\lambda \in \Lambda$ after being trained for $b$ discrete
training steps (epochs, here). A configuration's *learning curve* is the sequence
$\{f(\lambda, 1), f(\lambda, 2), …\}$ as $b$ grows.

Freeze-thaw HPO relaxes the usual "pick a config, train it to completion, observe one
number" protocol. Instead, resources are spent **incrementally**: at every iteration
the optimizer either **thaws** (resumes/continues) a previously partially-trained,
currently-frozen configuration for one more step, or starts a brand-new configuration.

Formally, the goal is to find a resource allocation that maximizes

$$
\max_{\lambda \in \Lambda,\ 1 \le b \le b_\lambda} f(\lambda, b)
$$
subject to

$$
N = |\{\text{decisions made}\}| \le B, \qquad
b_\lambda^{\min} \le\  b_\lambda \le b_\lambda^{\max}, \qquad 
\forall\, \lambda \in \Lambda \ \text{with } b_\lambda > 0, \qquad
b_\lambda \in \mathbb{Z}_{\ge 0}, \qquad 
\forall\, \lambda \in \Lambda
$$

- a total trial budget, $B$ - the number of freeze/thaw decisions the optimizer is allowed to make.
- $b_\lambda^{\min}$ and $b_\lambda^{\max}$ - per-configuration epoch bounds that prevent premature judgments on too 
little training and wasted spend on configs that have already converged.

## The FT-PFN surrogate

Rather than fitting a Gaussian Process or random forest online (as SMAC/BOHB do), ifBO
uses **FT-PFN**, a *Prior-data Fitted Network*: a transformer trained once, offline, on
large quantities of **synthetic** learning-curve data sampled from a hand-designed
prior, and then used purely for inference at HPO time via **in-context learning** — no
weights are updated during the actual HPO run.

The implementation follows **ifBO** as introduced in:

> H. Rakotoarison, S. Adriaensen, N. Mallik, S. Garibov, E. Bergman, F. Hutter.
> *In-Context Freeze-Thaw Bayesian Optimization for Hyperparameter Optimization.*
> ICML 2024. [arXiv:2404.16795](https://arxiv.org/abs/2404.16795)

### What the surrogate models

FT-PFN approximates the posterior predictive distribution

$$
p( f(\lambda_\text{test}, b_\text{test}) \mid \lambda_\text{test}, b_\text{test}, H )
$$

i.e., "given everything observed so far about *other* (and this) configurations'
partial learning curves, what is the distribution over this configuration's performance
at some future training step $b_\text{test}$?" The set $H$ is passed to the network as a sequence
of tokens (its **context**) at inference time; the transformer's attention mechanism
implicitly performs Bayesian updating over this context in a single forward pass,
rather than through iterative model refitting. This is what makes each acquisition
query cheap enough to run every freeze-thaw step (the paper reports 10–100× speedups
over refitting-based gray-box surrogates such as DPL and DyHPO).

## Acquisition function: MFPI and MFPI-random

### 5.1 The paper's definition

Multi-fidelity Probability of Improvement, as defined in the paper (Eq. 3):

$$
\operatorname{MFPI}(\lambda; h, T) = P( M(\lambda, \min(b_\lambda + h, b_\max)) > T )
$$

— the surrogate-predicted probability that configuration $\lambda$, if thawed for $h$ more
steps, would exceed a target performance $T$. Two "hyper-hyperparameters" govern it: the
lookahead horizon $h$ and the improvement target $T$. Rather than fixing these, the paper
proposes **MFPI-random** (Eq. 4), redrawing both **every single freeze-thaw iteration**:

$$
\begin{align*} 
\operatorname{MFPI-random}(\lambda) &= \operatorname{MFPI}(\lambda; h_\text{rand}, T_\text{rand}) \\
h_\text{rand}        &\sim U(1, b_\max) \\
T_\text{rand}        &= f_\text{best} + \tau_\text{rand} · (1 − f_\text{best}) \\
\log_{10}(\tau_\text{rand}) &\sim U(−4, −1)
\end{align*}
$$

where $f_\text{best}$ is the best performance observed so far. This amounts to sampling an
acquisition function from an implicit *portfolio* of MFPI instances at every step,
which the paper's ablations _(Figure 4)_ show is necessary — fixed-horizon or
fixed-threshold variants, and standard Expected Improvement paired with FT-PFN's
heavy-tailed posteriors, both underperform substantially.

## A growing the candidate pool

The paper's Algorithm 1 (see §7 for the literal loop) is stated over the *whole* search
space $\Lambda$ implicitly — at each iteration the acquisition is (conceptually) maximized over
all $\lambda \in \Lambda$, which naturally lets brand-new, never-tried configurations compete against
partially-trained ones for being selected next.

This implementation instead maintains an **explicit, dynamically growing list** of candidates
(starting empty), and layers an **explicit decaying $\epsilon$-greedy exploration floor** on top
of MFPI-random rather than substituting for it:

$$
\begin{align*}
\epsilon(t) = \epsilon_{\min} + (\epsilon_0 - \epsilon_{\min}) \cdot \left(1 - \frac{t}{T}\right)^p.\\
\text{where } \qquad \epsilon_{\min} = 0.05, \quad p = 2, \quad T = N
\end{align*}
$$

$\epsilon_0$ (`ifbo_initial_epsilon`, default $1.0$) means the very first steps are pure
exploration — with an empty/small pool there's nothing meaningful yet for MFPI-random to
discriminate between — decaying polynomially toward a floor of $0.05$ so some unconditional
exploration always remains, even late in the run. Each iteration then branches:

- **With probability $\epsilon(t)$** (the exploration floor): a brand-new configuration is
  sampled uniformly from $\Lambda$, added to the pool with zero observations, and selected
  directly — bypassing the surrogate entirely.
- **With probability $1-\epsilon(t)$** (an exploitation round): the *pending* pool (candidates
  not yet at $b_\max$, minus any already claimed earlier in the same parallel batch) is assembled, 
  **plus one additional fresh configuration** sampled
  uniformly from $\Lambda$ for this round only. MFPI-random ($h_\text{rand}, T_\text{rand}$
  redrawn as in §5.1) then scores *every* contender — pending and fresh alike — against the
  FT-PFN context in a single batched query, and either takes the arg max
  (`ifbo_greedy_candidate_selection`) or samples from a softmax over the PI scores. The fresh
  candidate is only appended to the persistent pool if it *wins* this round; otherwise it is
  discarded and never referenced again. A `ifbo_use_random_selection` flag swaps this scoring
  step for a uniform choice among the same contenders (pending + fresh), which is what backs
  the "freeze-thaw random" baseline (see Results, below). 
  **TODO: Add some experiment results for greedy and softmax methods**

One more rule guards the exploitation branch: once the current best-so-far candidate has
accumulated at least `ifbo_incumbent_exclusion_min_observations` (default $2$) freeze-thaw
steps, it is temporarily dropped from the pending pool for that round. Left unchecked, it tends
to keep re-winning $\mathrm{PI}(T_\text{rand})$ against its own already-confirmed best (since
$T_\text{rand}$ sits just above $f_\text{best}$), sinking budget into repeatedly re-thawing
itself instead of advancing or discovering other candidates.

**This layer is a deliberate engineering addition, not part of the published ifBO method.** It
exists because this implementation manages a discrete, growing candidate list rather than
treating "propose a new $\lambda$" as one more option scored unconditionally by the same
acquisition function every round — doing so on *every* round would grow the pool (and hence the
FT-PFN context) without bound. Note, though, that a fresh candidate *is* scored against the
surrogate on every exploitation round, competing on the same $\mathrm{PI}(T_\text{rand})$ scale
as pending candidates for a chance to enter the pool; the surrogate is only fully bypassed on
the explicit $\epsilon(t)$-floor rounds, which exist to guarantee a baseline injection rate of
new configurations independent of what the surrogate currently believes.

## Methodology

![ifbo_diagram](./ifbo_diagram.svg)


## Results

Budget:

```yaml
max_budget: 10
min_budget: 3
n_trials: 20
```

### Baselines

- random - non multifidelity random optimizer which trains a random configuration from a search space till its budget 
completion.
- multifidelity bayesian optimization with Hyperband intensifier using SMAC. SMAC3, random-forest-EI Bayesian optimization + Hyperband
  successive-halving.
- freeze thaw random - a system that follows the same pipeline but instead of using the FT-PFN surrogate to predict what
candidate to select, it selects one randomly.


### Testbed Results

All four methods below were run under a **matched budget** (`n_trials=20`, `min_budget=3`,
`max_budget=10` epochs, `sequence-dl`/BiLSTM search space) across all five datasets —
`ag_news`, `amazon`, `dbpedia`, `imdb`, and the held-out exam set `yelp`. Numbers are
**best validation accuracy** (`1 - min(val_error)`) reached anywhere in the run, i.e. search
quality, not a held-out test score. Raw histories are in `sample-results/all_results/`; the
figures below are regenerated straight from those `.jsonl` files (see
`sample-results/final_results_analysis.ipynb`).

#### Final best validation accuracy

| Dataset | ifBO | SMAC (BO+HB) | Random Search | ifBO (random selection) |
|---|---|---|---|---|
| AG News | 0.9135 | 0.8935 | 0.8965 | **0.9245** |
| Amazon | **0.8915** | 0.7820 | 0.7925 | 0.7930 |
| DBpedia | **0.9785** | 0.9700 | 0.9780 | 0.9770 |
| IMDB | **0.9355** | 0.8525 | 0.8670 | 0.8770 |
| Yelp | **0.5735** | 0.5050 | 0.5510 | 0.5435 |

The full ifBO (FT-PFN-guided) surrogate wins on 4 of 5 datasets, with the largest margins on
the hardest tasks — Amazon (+9.9pp over the best baseline) and Yelp (+2.3pp over Random, +6.9pp
over SMAC). On AG News, its own random-selection ablation edges it out (92.45% vs. 91.35%) —
on the easiest dataset in the suite (all methods clear 89%), the FT-PFN surrogate's
acquisition doesn't have much signal to exploit over unguided freeze-thaw scheduling, so the
two land within noise of each other. Everywhere else, guided candidate selection is what
separates ifBO from its own ablation.

![final accuracy bars](./figures/03_final_accuracy_bars.png)

![accuracy heatmap](./figures/04_accuracy_heatmap.png)

#### Sample efficiency and wall-clock cost

Freeze-thaw's structural advantage shows up earliest here: both ifBO variants pull ahead of
Random/SMAC within the first 5–7 trials on every dataset, because a handful of thaw steps into
one promising candidate teach the surrogate (or even just the epsilon-floor exploration alone)
more than one epoch each spent across many never-revisited configs.

![best accuracy vs trial number](./figures/02_best_vs_trial.png)

![best accuracy vs wallclock time](./figures/01_best_vs_wallclock.png)

Wall-clock tells a second story: Random Search's fixed per-trial budget means it always pays
for `max_budget` epochs regardless of how a config is doing, so it consistently takes 2–4x
longer than either ifBO variant for a 20-trial run, without a corresponding accuracy payoff.

![total wallclock time](./figures/08_total_wallclock.png)

#### Where the budget goes

The per-trial accuracy distribution below shows *every* sampled trial, not just the incumbent —
a tighter, higher box means an optimizer is spending steps on consistently good configs rather
than wasting them on poor ones. The epoch-budget plot shows the multi-fidelity behavior
directly: Random Search always trains to the same fixed budget, while SMAC's Hyperband rungs
and ifBO's continuous freeze-thaw both concentrate later, larger epoch budgets on the
configurations that already looked promising early.

![per-trial accuracy distribution](./figures/05_accuracy_boxplots.png)

![epoch budget allocated per trial](./figures/06_fidelity_allocation.png)

Finally, the training curve of each method's single best-found configuration per dataset:

![best trial training curves](./figures/07_best_trial_curves.png)
