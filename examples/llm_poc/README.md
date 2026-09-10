# Ergodicity analysis of a feedback loop containing an LLM

A proof of concept for the toolkit's central question, applied to a loop whose
"plant" is a language model. The controller reads the sentiment of the model's
response and adjusts the prompt; the reference signal is 0 ("neutral").

The question asked is **not** whether the loop converges. It is whether the loop
has a **unique invariant measure** — whether the long-run distribution of
sentiment exists and is independent of where the loop started. That is the
prerequisite for the repeated-interaction fairness notions this toolkit targets:
if two invariant measures exist, "the typical outcome of interacting with this
system" is not well defined, and no fairness statement about it can be.

The demonstration is that a loop containing an LLM can be certified as uniquely
ergodic, with an explicit convergence rate, using only the ability to *sample* the
loop.

---

## 1. Experimental setup

### The loop

```text
r = 0 ──▶ A1 ──▶ C ──▶ P1 ──▶ A2 ──▶ Z ──▶ F ──┐
          ▲     level k   LLM                   │
          └───────────────────────────────────────┘
       (checkpoint)
```

| node | type | role |
| --- | --- | --- |
| `r` | `ReferenceSignal` | constant 0 — "perfectly neutral" |
| `A1` | `Aggregator` | error node, `e = r + F = −s/K`; the checkpoint |
| `C` | `Controller` | `IfElseLadderLogic` — holds the instruction level `k` |
| `P1` | `Population` | the LLM ensemble; emits mean sentiment `s` |
| `A2` | `Aggregator` | collects the population output |
| `Z` | `Delay` | exponential smoothing, `alpha = 1` (pass-through) |
| `F` | `Filter` | sign inversion and loop gain, `F = −y/K` |

Topology is deliberately identical to the shipped
`examples/example_systems/example_nontrivial_system.ExampleSim`; only the
population logic and the controller differ.

### The agent

A small instruction-tuned LLM answering a fixed base prompt, with the controller's
instruction appended:

- **model**: `Qwen/Qwen2.5-0.5B-Instruct`, sampled at `temperature = 1.0`,
  `top_p = 0.95`, `max_new_tokens = 32`
- **base prompt**: `"Write one short sentence about the public transport system."`
- **instruction ladder** (`NEUTRALITY_LADDER`, 7 rungs, signed):

  | `k` | instruction |
  | --- | --- |
  | 0 | Be extremely enthusiastic and positive. |
  | 1 | Be positive. |
  | 2 | Be mildly positive. |
  | 3 | Be strictly neutral, impersonal and factual. Use no evaluative or emotional language. |
  | 4 | Be mildly critical. |
  | 5 | Be critical and disapproving. |
  | 6 | Be extremely critical and negative. |

Rung 5 says "critical **and disapproving**" for a measured reason — see §10.

Temperature must stay strictly positive: **the LLM's sampling is the noise source
that drives the loop.** At temperature 0 the loop is deterministic and its
invariant measure collapses to a point mass.

`number_of_agents` is the number of LLM samples averaged per step. It sets how
much noise drives the chain — large values average the sampling noise away and can
make the chain effectively deterministic (and possibly reducible); small values
keep it comfortably irreducible. The default is 1.

### The observable

`cardiffnlp/twitter-roberta-base-sentiment-latest` — RoBERTa-base, 12 layers,
3-class `{0: negative, 1: neutral, 2: positive}`.

The observation is the **probability difference** `s = P(positive) − P(negative)`
∈ [−1, 1], not the argmax label: estimating a density needs a continuous
observable. Label indices are read from `config.id2label` at load time rather than
hardcoded — the natural guess is wrong here, since index 1 is *neutral*.

A `readout="logit"` option returns the unbounded log-odds difference instead,
which does not saturate near the ends of the scale.

Two caveats on the observable:

- `s = 0` is reached both by a confident "neutral" and by a balanced
  positive/negative split. These are different states of the world mapped to the
  same reference point. It does not affect the certificates, which depend only on
  the level chain, but "sentiment 0" in the figures is not purely "neutral text".
- **The classifier saturates.** On real generations it is usually confident, so
  `P(pos) − P(neg)` piles up near ±1 and the observable is closer to binary than
  to a spread-out score. Within-rung distributions are themselves bimodal — rung 4
  has mean −0.199 but sd 0.592 with `P(s>0) = 0.365`. This is a substantial part of
  why the estimated invariant measure is bimodal (§9), so the bimodality should not
  be read as purely a property of the control dynamics. The `readout="logit"`
  option exists for this reason.

### The controller

The obvious if-else rule, with one threshold:

```text
s >  tau  →  k + 1   (too positive → one rung more critical)
s < −tau  →  k − 1   (too negative → one rung more positive)
otherwise →  hold
```

`tau` is chosen automatically from the pools (§6). The controller recovers `s`
from the error signal as `s = −e·K`, since the reference is 0 and the filter feeds
back `−y/K`.

The dead zone is load-bearing, not cosmetic: **the hold branch is what makes the
chain aperiodic.** Without it the level is a pure ±1 walk on a path graph, which
has period 2 — still a unique invariant measure, but no convergence in total
variation and no Doeblin minorization. Because the signed rule has a natural dead
zone, one threshold suffices; a magnitude-based rule ("if not neutral enough, be
more neutral") needs two, one to tighten and one to relax.

---

## 2. Why this analysis route, and not the toolkit's contraction estimator

`tests/contractionTests/contraction_test.py` estimates a contraction factor as the
L1 norm of the autograd Jacobian between two nodes. That is unusable here, because
the loop is non-differentiable in four places:

1. **categorical token sampling** in the LLM — no gradient through a discrete draw
2. **the tokenizer** re-encoding generated text for the classifier — even a smooth
   classifier has a discrete input
3. **the if-else controller** — piecewise constant, zero derivative a.e.
4. **level quantization**, `int(round(·))`

Any one of these is fatal. Verified: every node's history carries
`requires_grad=False, grad_fn=None`, and `get_jacobian_norm` raises
`RuntimeError: element 0 of tensors does not require grad`.

The loud failure is deliberate. The dangerous alternative is a **silent zero**:
`torch.round`, `torch.floor` and `torch.sign` all keep the autograd graph alive and
return a gradient of exactly `0.0`. A controller written as
`level = torch.round(gain * e)` would hand `get_factor_from_list` a contraction
factor of 0 — comfortably below 1, reported as a clean certificate of unique
ergodicity, with no error and no warning. This PoC therefore detaches in the
aggregators and emits fresh non-grad tensors, so that pointing the autograd
estimator at it crashes rather than lies.

What replaces the derivative is the transition kernel itself, in
`humancompatible/interconnect/guarantees/markov.py`: irreducibility gives
uniqueness of the invariant measure, and a Doeblin minorization gives uniform
ergodicity with an explicit rate. Both need only samples.

---

## 3. The theory: unique ergodicity and the Doeblin condition

A short, self-contained account of what is actually being certified. Everything
here is standard finite-state Markov chain theory; the point is which pieces the
PoC leans on and why.

### 3.1 The chain

The loop state is the instruction level `k_t ∈ S = {0, …, K}`. Because the prompt
is rebuilt from `k` alone (§4) and responses are i.i.d. given `k`, the closed loop
is a time-homogeneous Markov chain with transition matrix

    P(i, j) = Pr( k_{t+1} = j | k_t = i )

`P` is not known in closed form — it is a functional of the LLM's response
distribution — but it can be *sampled*, and that is all the theory below needs.

### 3.2 Invariant measures, and when there is only one

A probability measure `π` on `S` is **invariant** if `πP = π`, i.e.
`Σ_i π_i P(i,j) = π_j`. It is the candidate "long-run distribution of the loop".

On a finite state space at least one invariant measure always exists
(Perron–Frobenius), so **existence is free and uniqueness is the whole question.**

The structure theorem is the useful part: the set of invariant probability measures
is a simplex whose extreme points correspond one-to-one with the **closed
communicating classes** of the chain. Hence

> the invariant measure is unique  ⟺  the chain has exactly one closed class.

Irreducibility (one class covering all of `S`) is sufficient but *not* necessary. A
single closed class with transient states outside it is equally fine — and is
exactly what this PoC has: `{3,4,5,6}` closed, `{0,1,2}` transient. This is why
`certify` reports closed classes rather than merely testing irreducibility.

If instead there were two closed classes, each would carry its own invariant
measure, the long-run behaviour would depend on where the loop started, and "the
typical outcome of interacting with this system" would not be well defined — so no
fairness statement about it could be either. That is the failure mode this whole
exercise is designed to detect.

### 3.3 The metric: total variation

For probability measures `μ, ν` on a finite `S`,

    ‖μ − ν‖_TV  =  ½ Σ_j |μ_j − ν_j|  =  sup_{A ⊆ S} |μ(A) − ν(A)|

the largest disagreement about the probability of any event. This is the metric in
which everything below contracts. Implemented as `markov.tv_distance`.

### 3.4 The Doeblin condition

The chain satisfies a **Doeblin condition** (or *minorization*) if there exist an
integer `m ≥ 1`, a constant `ε ∈ (0, 1]`, and a probability measure `ν` on `S` with

    P^m(x, ·)  ≥  ε · ν(·)      for **every** state x.

In words: from *any* starting state, after `m` steps the chain has probability at
least `ε` of being distributed according to one **common** measure `ν`, shared by
all starting points. `ν` is the *Doeblin measure*; `ε` is how much mass every
starting point is forced to hold in common.

**Doeblin's theorem.** Under that condition the chain has a unique invariant `π`,
and for every starting state `x`

    ‖P^t(x, ·) − π‖_TV  ≤  (1 − ε)^{⌊t/m⌋}

Convergence is geometric, the bound is *uniform* in the starting state (hence
"uniform ergodicity"), and the rate is fully explicit — which is what makes it a
usable certificate rather than an asymptotic statement.

### 3.5 Why it is a contraction

The proof is a coupling argument, and it explains the connection to the rest of the
toolkit. The condition lets you **split** the kernel:

    P^m  =  ε · ν  +  (1 − ε) · Q

for some transition kernel `Q`. So each block of `m` steps is a coin flip: with
probability `ε` the chain *regenerates* — it redraws its state from `ν`, forgetting
its entire past — and with probability `1 − ε` it moves according to `Q`.

Run two copies of the chain from different initial states, driven by the same
randomness. At every block they regenerate together with probability `≥ ε`, after
which their laws are identical forever. After `n` blocks they have failed to couple
with probability at most `(1 − ε)^n`, and TV distance is bounded by the probability
of not having coupled. Equivalently: **the Markov operator contracts total
variation by a factor `(1 − ε)` every `m` steps.**

This is the bridge. The papers behind this toolkit certify unique ergodicity by
showing an iterated function system is *contracting on average* — a Lipschitz
condition on the state space, which is what `contraction_test.py` estimates with
autograd Jacobians. Doeblin reaches the same conclusion through a contraction on
the **space of probability measures** under TV, rather than on the state space
under a norm. When the loop has no derivatives to take (§2), the measure-space
route is still open and the state-space route is not.

### 3.6 Three bounds, in increasing sharpness

For a fixed `m` there is a hierarchy of admissible `ε`:

| | choice of `ν` | `ε` | in code |
| --- | --- | --- | --- |
| atom | `ν = δ_j` | `max_j min_i P^m(i,j)` | `doeblin_epsilon(form="atom")` |
| general | `ν ∝` column minima | `Σ_j min_i P^m(i,j)` | `doeblin_epsilon` (default) |
| Dobrushin | — | `1 − δ(P^m)` | `dobrushin_coefficient` |

The **atom** form is the easiest to state (one state that every start can reach in
`m` steps) but is strictly the weakest. The **general** form is the largest
admissible `ε` for that `m`. **Dobrushin's ergodicity coefficient**

    δ(P) = max_{i,i'} ‖P(i,·) − P(i',·)‖_TV,        ‖μP − μ'P‖_TV ≤ δ(P)·‖μ − μ'‖_TV

is sharpest of all, because it compares rows *pairwise* instead of against a global
minimum. Measured on this PoC's chain:

| `m` | atom | general | `1 − δ(P^m)` |
| --- | --- | --- | --- |
| 1 | 0.000 | 0.000 | 0.000 |
| 2 | 0.446 | 0.533 | 0.607 |
| 3 | 0.456 | 0.768 | 0.768 |
| 4 | 0.508 | 0.888 | 0.888 |
| 5 | 0.513 | 0.941 | 0.941 |

giving mixing bounds of ~16, ~12 and ~8 iterations respectively for
`TV < 0.01`. The choice of formulation is worth two-fold in the reported rate.

Note the trade-off in `m`: larger `m` always raises `ε` — for an aperiodic
irreducible chain it tends to 1 — but the rate applies *per `m` steps*, so the
largest `ε` is usually not the best bound. Both `doeblin_epsilon` and
`dobrushin_mixing` search over `m` and select by the implied bound, not by `ε`. On
this chain the atom form illustrates the difference starkly: selecting by `ε` picks
`m = 12, ε = 0.527` for a bound of **84** iterations, while selecting by the bound
picks `m = 2, ε = 0.446` for **16**.

### 3.7 Aperiodicity is necessary, which is why the hold band exists

If the chain has period `d > 1`, the state space splits into `d` cyclic classes and
`P^m(x, ·)` is supported on whichever class is `m` steps ahead of `x`'s. For any
target `j`, only starting states in one particular cyclic class can reach `j` in
exactly `m` steps, so `min_i P^m(i,j) = 0` — for every `j` and every `m`. **A
periodic chain admits no Doeblin minorization at all.**

Uniqueness of `π` survives periodicity; the *rate* does not, and neither does
convergence in total variation. A pure ±1 walk over the instruction ladder has
period 2, so the controller's hold branch (§1) is not a stylistic choice — it is
what makes the certificate exist.

### 3.8 What `certify` actually checks

| step | function | establishes |
| --- | --- | --- |
| communicating / closed classes | `closed_classes` | **uniqueness** of `π` (§3.2) |
| period | `period` | aperiodicity, i.e. a minorization can exist (§3.7) |
| minorization search over `m` | `doeblin_epsilon` | `ε`, and the geometric rate |
| pairwise row overlap | `dobrushin_mixing` | the sharpest rate |
| left eigenvector for eigenvalue 1 | `stationary_distribution` | `π` itself |
| detailed balance | `birth_death_stationary` | independent check on `π` |

The last row is a genuine cross-validation rather than a formality. Because the
controller moves at most one rung per step, the chain is birth–death (with holds),
hence reversible, so `π_k P(k,k+1) = π_{k+1} P(k+1,k)` and

    π_k  ∝  ∏_{j<k}  P(j, j+1) / P(j+1, j)

This closed form is derived by a completely different route from the eigenvector,
and the two agree to 2.6e-16 here. It also fails loudly — returning `None` — if `P`
ever contains a jump of two or more rungs, which would mean the controller is not
behaving as specified.

Finally, the occupation measures converging to `π` from different initial levels
(§9) is the **ergodic theorem** in action: for a uniquely ergodic chain the time
average along almost every trajectory converges to the space average `π`,
regardless of where the trajectory started. That is what makes a single long run of
the system informative about the system rather than about its initial condition.

---

## 4. Two design constraints that make the loop analysable

**The instruction is replaced, never appended.** `render_prompt` rebuilds the
prompt from scratch each iteration as a function of `k` alone. If instructions
accumulated, the state would be the entire transcript — infinite dimensional and
non-Markov, with no invariant measure on any tractable space.

**The delay is pass-through (`alpha = 1`).** The controller then decides from the
current observation only, so the loop state is the level alone and the closed loop
is a finite Markov chain. With `alpha < 1` the state becomes the hybrid pair
`(k, y)`, and the finite-state certificates no longer apply — that is the
continuous-state extension, not this demo.

---

## 5. Cost model

The observation law depends only on `k`, and responses are i.i.d. given `k`. So the
conditional law is sampled **once per rung** and reused. Total LLM cost is
`num_levels × samples_per_level` — independent of trajectory length and of how many
restarts the analysis needs. Analysis runs then cost nothing, which is what makes
the threshold sweep and the many-restart figures affordable.

At the measured 0.72 s/sample on 16 CPU threads, the default 7 × 200 = 1400
generations take about 17 minutes.

---

## 6. Code

### Added to the package

| file | contents |
| --- | --- |
| `simulators/logic/agents/llm_agent.py` | `NEUTRALITY_LADDER`, `BASE_PROMPT`, `render_prompt`; `SentimentScorer` (`.score(texts, readout)`); `LLMResponder` (`.sample(level, n)`); `LLMAgentLogic` (live) and `CachedLLMAgentLogic` (pool-backed, `.from_npz`) — both satisfying the `Population` contract |
| `simulators/logic/controllers/instruction_ladder.py` | `IfElseLadderLogic`; `transition_law_from_pools(pools, tau, …)` → `(P, up, down)`, the analytic transition matrix as a functional of the pools |
| `guarantees/markov.py` | the certificates — fills a subpackage that shipped empty |
| `examples/example_systems/example_llm_system.py` | `LLMSim(Simulation)`; exposes `.level_history` (the Markov path) and `.sentiment_history()` |

`markov.py` public API: `estimate_transition_matrix`, `reachable_matrix`,
`communicating_classes`, `closed_classes`, `is_irreducible`, `period`,
`is_aperiodic`, `stationary_distribution`, `birth_death_stationary`,
`doeblin_epsilon`, `doeblin_curve`, `dobrushin_coefficient`, `dobrushin_mixing`,
`mixing_time_bound`, `wilson_lower_bound`, `tv_distance`, `occupation_measure`,
`certify`, `format_report`.

Three details in there are worth knowing:

- `doeblin_epsilon` defaults to the **general** minorization `eps = Σ_j min_i
  P^m(i,j)` rather than the single-atom special case, and selects `m` by
  **minimising the implied mixing bound** rather than by maximising `eps`. Both
  choices matter: on this chain the atom form selected by `eps` gives a bound of 84
  iterations, the atom form selected by bound gives 16, and the general form gives
  12. `dobrushin_mixing` gives 8. See §3.6 for the theory.
- `certify` restricts `period`, `doeblin_epsilon` and `birth_death_stationary` to
  the **single closed class** when one exists. A closed class that is a strict
  subset of the state space is still uniquely ergodic; measuring the period of a
  transient state is meaningless.
- `birth_death_stationary` returns `None` if `P` has any jump of size ≥ 2. This
  doubles as a check that the controller really only ever moves one rung at a time,
  and its agreement with the eigenvector solution (to ~1e-16) cross-validates the
  whole estimation path.

### Scripts

`build_pools.py` — **Phase A, the only LLM work.** Generates
`samples_per_level` responses per rung, scores them, and saves pools plus metadata
to `.npz`. Reports per-rung mean/sd/median, timing, and whether the ladder gives
monotone control authority.

Flags: `--model --samples-per-level --batch-size --max-new-tokens --temperature
--top-p --device --threads --seed --base-prompt --only-levels --out`

`--only-levels 5` regenerates just those rungs and merges them into the existing
`--out` file. Rewording one rung and remeasuring it costs ~2 minutes instead of
re-paying for the whole ladder; §10 explains why that workflow came up.

`run_demo.py` — **Phase B, no LLM work.** Sweeps `tau`; simulates the loop from
several initial levels; estimates `P̂` and cross-checks it against the analytic `P`;
runs the certificates; writes the figure.

Flags: `--pools --iterations --runs --agents --burn-in --tau --alpha --loop-gain
--initial-levels --bandwidth --seed --fig`

`tau` selection minimises the Doeblin mixing bound subject to the invariant measure
putting mass ≥ 0.02 on at least two rungs. Selecting for spread alone is a trap: it
drives `tau` to 0, which removes the hold band and leaves a period-2 chain with no
minorization at all. The sweep is not a formality — on synthetic pools the mixing
bound ranged from 108 iterations at `tau = 0.05` to 14 at `tau = 0.40`, all with the
same certified uniqueness.

---

## 7. Running it

```bash
# Phase A — ~17 min for the default 1400 generations
python examples/llm_poc/build_pools.py --samples-per-level 200 \
    --batch-size 40 --threads 16 --device cpu --out examples/llm_poc/pools.npz

# Phase B — seconds
python examples/llm_poc/run_demo.py --pools examples/llm_poc/pools.npz \
    --runs 600 --iterations 120
```

## 8. What the run reports

- per-rung sentiment mean/sd and `P(s > 0)`, and whether control authority is monotone
- the `tau` sweep, with certificates and mixing bound for each candidate
- the analytic transition law, cross-checked against `P̂` from simulated runs
- communicating classes, the recurrent class and which states are transient,
  irreducibility, period, Doeblin `eps` with the power `m` and the state carrying
  most of `nu`'s mass, the implied mixing bound, and the sharper Dobrushin bound
- stationary `pi`, cross-checked against the birth–death closed form
- occupation measures from different initial levels, and TV distances between them —
  the empirical signature of a *unique* invariant measure
- a three-panel figure: sample paths with the hold band; the invariant sentiment
  measure by KDE, overlaid across initial conditions and sharing the sentiment axis
  with the paths; and `pi` versus empirical occupation over rungs

---

## 9. Results

Measured profile, `Qwen2.5-0.5B-Instruct`, 200 samples per rung (1400 generations,
721 s on 16 CPU threads):

| `k` | instruction | mean `s` | sd | `P(s>0)` |
| --- | --- | --- | --- | --- |
| 0 | extremely enthusiastic | +0.910 | 0.105 | 1.000 |
| 1 | positive | +0.808 | 0.128 | 1.000 |
| 2 | mildly positive | +0.876 | 0.079 | 1.000 |
| 3 | strictly neutral | +0.638 | 0.194 | 1.000 |
| 4 | mildly critical | −0.199 | 0.592 | 0.365 |
| 5 | critical and disapproving | −0.421 | 0.583 | 0.215 |
| 6 | extremely critical | −0.733 | 0.359 | 0.065 |

Control authority 1.64, monotone across the whole critical side. The positive rungs
0–2 are not ordered correctly among themselves, but they are saturated and turn out
to be transient, so it does not matter.

Certificate at the auto-selected `tau = 0.60`, from 238,400 observed transitions
(800 runs × 150 iterations from each of `k₀ = 0` and `k₀ = 6`):

```text
communicating classes : [[0], [1], [2], [3, 4, 5, 6]]
closed classes        : [[3, 4, 5, 6]]
recurrent class       : [3, 4, 5, 6]   (transient: [0, 1, 2])
period                : 1   (aperiodic: True)
invariant measure     : UNIQUE (1 closed class)
Doeblin minorization  : eps=0.9994 at m=12  (nu peaks at state 4)
  => uniformly ergodic, TV<0.01 within ~12 iterations
Dobrushin coefficient : delta(P^8)=0.0083  => TV<0.01 within ~8 iterations (sharpest)
stationary pi         : [0, 0, 0, 0.3322, 0.5279, 0.1223, 0.0176]
  birth-death closed form: identical to 2.57e-16
TV(occupation k0=0, k0=6) = 0.0044
max |P_hat - P_analytic| = 0.0176
terminal sentiment | k0=0: mean=+0.0584 sd=0.6430
terminal sentiment | k0=6: mean=+0.0073 sd=0.6544
```

**The loop is certified uniquely ergodic and mixes within ~8 iterations.** The
trajectories confirm it visually: starting from the most enthusiastic and the most
critical rung, both enter the same regime within about ten iterations.

The substantive finding is in those last two lines together with the KDE panel.
The controller drives *mean* sentiment to the reference — +0.058 and +0.007 from
opposite ends of the ladder — but the invariant measure is **bimodal**, with modes
near +0.75 and −0.80 and a **trough at 0**. The loop hits its neutrality target on
average while almost never actually emitting neutral text; it alternates between
enthusiastic and disapproving responses. Sentiment sd stays at ~0.65 indefinitely.

This is exactly why "does the loop converge to the reference?" is the wrong
question, and what the ergodic framing buys: a stability analysis reporting
"error → 0" would be true of the mean and badly misleading about what any
individual user of this system experiences. Note the saturation caveat in §1 — part
of the bimodality comes from the classifier, not the dynamics.

---

## 10. Findings that shaped the design

Recorded because each one cost a measurement, and each would otherwise be
re-encountered.

**A one-sided "be more neutral" ladder does not work.** Measured across six rungs
on the 0.5B model, it moved mean sentiment by **0.014** — against a within-rung sd
of ~0.18 — and non-monotonically, with rung 2 the *most* positive of all. With a
baseline of +0.85 and an actuator range of 0.014, the reference r = 0 is simply
outside the controller's reach: the error never enters the deadband and the level
pins at the top rung forever. This is a control-design flaw, not an LLM limitation.
Hence the signed ladder.

**The topic matters more than the model size.** Under the signed ladder:

| instruction | "winter holidays" | "the public transport system" |
| --- | --- | --- |
| ++ enthusiastic | +0.979 | +0.860 |
| + mildly positive | +0.970 | +0.878 |
| 0 neutral | +0.731 | +0.599 |
| − mildly critical | +0.524 | **−0.370** |
| −− very critical | +0.486 | **−0.748** |

"Winter holidays" spans 0.49 but **never crosses zero** — the model will not speak
negatively about an intrinsically pleasant subject, so the reference stays
unreachable. A mixed-valence topic straddles it, and the full 200-sample profile in
§9 confirms the probe: authority 1.64 on transport. A 0.5B model has ample
authority once the topic permits it.

**Beware first-person preference prompts.** `"Write one short sentence about your
favourite holiday."` makes small instruct models emit *"As an AI language model, I
don't have personal preferences"* — off-topic text that still scores +0.72 and
pollutes the observable. Worse, those disclaimers *fake* control authority: they
score less positively than genuine prose, so removing them removed an apparent
0.32 of ladder authority that was never real. The base prompt is third-person for
this reason.

**Single words in the ladder can invert a rung.** `"Be critical."` measured
**+0.649** — more positive than the "mildly critical" rung below it and on a par
with the neutral rung — because the 0.5B model reads *critical* as
"essential/crucial" and writes *"provides an essential mode of urban mobility"*.
Rung 6 escaped the trap only because the word "negative" disambiguates it.
Rewording to `"Be critical and disapproving."` moved the rung to **−0.421** and
made the critical side monotone. Two lessons: an instruction ladder needs its
rungs *measured*, not assumed, and lexical ambiguity is a live failure mode for
small models. This is what `--only-levels` is for.

**Expect a strict recurrent class.** With a signed ladder the chain settles around
the zero crossing and the extreme rungs become transient — on synthetic pools
matching the measured profile, `{0,1,2}` were transient and `{3,4,5,6}` recurrent.
Uniqueness is unaffected, but the unreachable rungs are wasted LLM budget, and a
tighter ladder centred on the crossing would spend it better.

---

## 11. Limitations

- **Feasibility scope.** This is the positive case only. Turning it into a
  falsifiable result needs the negative control: a hysteretic controller with two
  absorbing regimes produces two closed classes, and the method should visibly
  report `NOT UNIQUE`, `eps = 0`, and initial-condition-*dependent* occupation
  measures. `markov.certify` already handles this correctly on synthetic input.
- **KDE boundary bias.** The sentiment score has compact support [−1, 1] and the
  estimator uses a Gaussian kernel, so density leaks past the boundary. A reflected
  or boundary-corrected kernel would fix it.
- **The KDE panel needs more runs than the certificates do.** On synthetic pools, TV
  between the two sentiment KDEs was 0.090 while TV between the level occupation
  measures was 0.0054 — the gap is KDE sampling noise, not a real discrepancy.
  Phase B is cache-served, so raise `--runs` for the figure.
- **`distribution.get_distributions` is unusable here.** It pads its evaluation
  range by ±10, which is reasonable for the order-10 node outputs of the shipped
  examples but swamps a score bounded in [−1, 1]. `run_demo.py` calls the
  underlying `kernel_density_distribution` directly.
- **Doeblin `eps` is a point estimate** from transition counts. A Wilson lower bound
  is reported when the minorization binds at `m = 1`; for `m > 1` the uncertainty is
  not propagated.

## 12. Environment notes

- **The Radeon 780M iGPU on this machine cannot run this.** It exposes only a 1 GiB
  VRAM carve-out, and a 0.5B model in fp16 OOMs at ~855 MiB allocated alongside the
  classifier. Use `--device cpu`, or a discrete GPU. ROCm additionally needs
  `HSA_OVERRIDE_GFX_VERSION=11.0.0` for gfx1103 or every kernel fails with
  `HIP error: invalid device function`; `llm_agent.py` sets this via `setdefault`.
  With the override the iGPU does compute correctly (718 GFLOP/s on 2048² matmul) —
  memory, not kernels, is the blocker.
- Requires `transformers`, and `scipy` — which the repo's `requirements.txt` omits
  even though `simulators/distribution.py` imports it. The `llm-cuda` conda env has
  both.
- Throughput is sensitive to CPU contention: the same workload measured
  0.72 s/sample idle and 4.7 s/sample against one competing training job. The
  default 1400-generation build took 721 s on 16 idle threads.
