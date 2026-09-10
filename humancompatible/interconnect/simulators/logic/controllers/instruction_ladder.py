"""Signed if-else controller over a ladder of prompt instructions.

The controller's only actuator is an index ``k`` into
``logic.agents.llm_agent.NEUTRALITY_LADDER``, which runs from "be extremely
enthusiastic" (k = 0) through a neutrality directive to "be extremely critical"
(k = last).  The rule is the obvious one: if the response came out too positive,
push one rung toward critical; too negative, push one rung toward positive;
otherwise hold.

Why signed, and why a hold band
-------------------------------
* A one-sided "be more neutral" ladder has no authority to straddle the
  reference.  Measured on a 0.5B instruct model it moved sentiment by 0.014
  across six rungs while the baseline sat at +0.85, so the error never entered
  the deadband and the level pinned at the top rung forever.
* The hold band is what makes the chain aperiodic.  Without it the level is a
  pure +-1 walk, which has period 2: still a unique invariant measure, but no
  convergence in total variation and no Doeblin minorization.  Because the signed
  rule has a natural dead zone ``|s| <= tau``, one threshold suffices -- the
  magnitude formulation needed two.
"""

import numpy as np
import torch


class IfElseLadderLogic:
    """:param tau: dead zone half-width on the sentiment observation.
    :param num_levels: size of the instruction ladder.
    :param initial_level: level at ``t = 0``; varying it is how the PoC probes
        whether the invariant measure is independent of the initial condition.
    :param loop_gain: the ``K`` used by the filter node, needed to recover the
        sentiment from the error signal.
    """

    def __init__(self, num_levels=7, tau=0.25, initial_level=None, loop_gain=1.0):
        if tau < 0:
            raise ValueError("tau must be non-negative")
        self.tensors = {"e": torch.tensor([0.0])}
        self.variables = ["e"]
        self.num_levels = int(num_levels)
        self.tau = float(tau)
        self.loop_gain = float(loop_gain)
        self.initial_level = (self.num_levels // 2 if initial_level is None
                              else int(initial_level))
        self.level = self.initial_level
        self.level_history = []

    def reset(self, level=None):
        self.level = self.initial_level if level is None else int(level)
        self.level_history = []

    def forward(self, values):
        self.tensors["e"] = values["e"]
        # Sign convention: the reference is 0 and the filter feeds back -y / K,
        # so the error node emits e = -s / K.  Recover the sentiment itself.
        sentiment = -float(values["e"].item()) * self.loop_gain
        if sentiment > self.tau:            # too positive -> more critical
            self.level = min(self.level + 1, self.num_levels - 1)
        elif sentiment < -self.tau:         # too negative -> more positive
            self.level = max(self.level - 1, 0)
        # otherwise: hold, which is what makes the chain aperiodic
        self.level_history.append(self.level)
        return torch.tensor([float(self.level)])


def transition_law_from_pools(pools, tau, num_levels=None, number_of_agents=1,
                              rng=None, mc_samples=20000):
    """Analytic transition matrix of the closed loop, straight from the pools.

    Given the per-level sentiment pools there is no need to simulate to learn the
    chain: the up/down/hold probabilities at level ``k`` are tail masses of the
    level-``k`` observation law.  For ``number_of_agents > 1`` the observation is a
    mean of that many draws, so the tails are estimated by Monte Carlo over the
    pool rather than counted directly.

    :return: ``(P, up, down)``, where ``up[k]`` is the probability of moving one
        rung toward critical and ``down[k]`` one rung toward positive.
    """
    pools = [np.asarray(p, dtype=np.float64) for p in pools]
    n_levels = num_levels or len(pools)
    rng = rng if rng is not None else np.random.default_rng(0)

    up = np.empty(n_levels)
    down = np.empty(n_levels)
    for k in range(n_levels):
        pool = pools[k]
        if number_of_agents == 1:
            obs = pool
        else:
            obs = rng.choice(pool, size=(mc_samples, number_of_agents),
                             replace=True).mean(axis=1)
        up[k] = float((obs > tau).mean())
        down[k] = float((obs < -tau).mean())

    P = np.zeros((n_levels, n_levels))
    for k in range(n_levels):
        P[k, min(k + 1, n_levels - 1)] += up[k]
        P[k, max(k - 1, 0)] += down[k]
        P[k, k] += 1.0 - up[k] - down[k]
    return P, up, down
