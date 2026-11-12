"""Invariant-measure plots for persona_steering_loop.py, one figure per target R: trajectories of y, and the
long-run densities of y and alpha per starting point alpha0 (KDE over seeds and steps after BURN)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, wasserstein_distance

BURN = 20   # by step ~15 the runs from different alpha0 have merged
FILES = {0.3: "persona_steering_runs_r03.csv", 0.7: "persona_steering_runs_r0.7.csv"}

for R, path in FILES.items():
    d = pd.read_csv(path).query("agent == 0")   # alpha and y repeat on every agent's row
    a0s = sorted(d.alpha0.unique())
    late = d[d.step >= BURN]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    for a0, color in zip(a0s, ["C0", "C1", "C2"]):
        run = d[(d.alpha0 == a0) & (d.seed == 0)]
        ax[0].plot(run.step, run.y, color=color, lw=1, label=rf"$\alpha_0={a0:g}$")
        for k, col in [(1, "y"), (2, "alpha")]:
            x = late[late.alpha0 == a0][col].to_numpy()
            grid = np.linspace(x.min() - 0.1, x.max() + 0.1, 300)
            ax[k].plot(gaussian_kde(x)(grid), grid, color=color, label=rf"$\alpha_0={a0:g}$")
    # largest W1 distance between the long-run samples of two starting points: ~0 if the limit forgets alpha0
    w1 = {col: max(wasserstein_distance(late[late.alpha0 == p][col], late[late.alpha0 == q][col])
                   for p in a0s for q in a0s if p < q) for col in ["y", "alpha"]}
    ax[0].axhline(R, color="k", lw=1); ax[0].axvline(BURN, color="grey", ls=":", lw=1)
    ax[0].set_xlabel("step"); ax[0].set_ylabel("y (mean P(Yes))"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Population output $y_k$ (seed 0), target R = {R}")
    ax[1].axhline(R, color="k", ls=":", lw=1)
    ax[1].set_title(f"Invariant measure of y (KDE, step $\\geq$ {BURN})\nmax $W_1$ = {w1['y']:.4f}")
    ax[2].set_title(f"Invariant measure of $\\alpha$ (KDE, step $\\geq$ {BURN})\nmax $W_1$ = {w1['alpha']:.4f}")
    ax[1].set_ylabel("y"); ax[2].set_ylabel(r"$\alpha$")
    for a in ax[1:]:
        a.set_xlabel("density"); a.legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figs/persona_steering_invariant_r{R}.png", bbox_inches="tight", dpi=150)
