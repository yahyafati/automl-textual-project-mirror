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

This implementation instead maintains an **explicit, dynamically growing list** of candidates (starting empty) 
and makes the "propose a new configuration vs. continue an existing one" decision via an **explicit decaying 
$\epsilon$-greedy rule**:

$$
\begin{align*}
\epsilon(t) = \epsilon_{\min} + (\epsilon_0 - \epsilon_{\min}) \cdot \left(1 - \frac{t}{T}\right)^p.\\
\text{where } \qquad \epsilon_{\min} = 0.1, \quad p = 2, \quad T = N
\end{align*}
$$

At each iteration, with probability $\epsilon(t)$ a brand-new configuration is sampled uniformly
from $\Lambda$ and added to the pool with zero observations; 
with probability $1 - \epsilon(t)$, MFPI-random selects among the existing *pending* candidates instead. 
$\epsilon_0$ means the very first steps are pure exploration (every early step spawns a new candidate, since with
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


