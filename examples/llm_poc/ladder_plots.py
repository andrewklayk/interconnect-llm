"""Figures for ladder_results.ipynb, in the same style as orchestrator_plots."""
import numpy as np
from scipy.stats import gaussian_kde
from orchestrator_plots import plt, COLORS, INK, GREY, DOTTED   # importing also sets the shared rcParams

SHOW = 3   # runs drawn per starting rung


def ladder_runs(paths, pi, burn):
    """left: rung k_t of a few runs per start k0; right: stationary pi against the occupation of each start"""
    fig, (ax, ab) = plt.subplots(1, 2, figsize=(10, 4.1), gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.08},
                                 sharey=True)
    K, h = len(pi), 0.8 / (len(paths) + 1)
    ab.barh(np.arange(K) - h * len(paths) / 2, pi, h, color=GREY, label="π (from P)")
    for j, ((k0, p), c) in enumerate(zip(paths.items(), [COLORS[0], COLORS[2]])):
        for r in range(SHOW):
            ax.step(np.arange(p.shape[1]), p[r] + 0.08 * (r - 1), where="post", color=c, lw=1.1,
                    alpha=0.8, label=rf"from $k_0 = {k0}$" if r == 0 else None)
        occ = np.bincount(p[:, burn:].ravel(), minlength=K) / p[:, burn:].size
        ab.barh(np.arange(K) + h * (j + 1 - len(paths) / 2), occ, h, color=c, label=rf"time in $k$, $k_0 = {k0}$")
    ax.set_xlim(0, 60); ax.set_ylim(K - 0.5, -0.5)   # rung 0 (most enthusiastic) on top
    ax.set_yticks(range(K)); ax.set_xlabel("round $t$"); ax.set_ylabel("instruction rung $k_t$")
    ax.legend(frameon=False, fontsize=12, loc="upper right")   # rungs 0-2 are transient, so the top is empty
    ab.set_xlabel("probability"); ab.tick_params(left=False)
    ab.legend(frameon=False, fontsize=11, loc="upper right")
    return fig


def sentiment_density(sents, tau, burn):
    """long-run density of the sentiment score s, one curve per start"""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    grid = np.linspace(-1, 1, 400)
    ax.axvspan(-tau, tau, color=GREY, alpha=0.25, lw=0)
    for (k0, s), c in zip(sents.items(), [COLORS[0], COLORS[2]]):
        ax.plot(grid, gaussian_kde(s[:, burn:].ravel(), bw_method=0.15)(grid), color=c, lw=1.8,
                label=rf"from $k_0 = {k0}$")
    ax.text(0, ax.get_ylim()[1] * 0.95, "hold band", ha="center", va="top", color=INK, fontsize=13)
    ax.set_xlim(-1, 1); ax.set_yticks([])
    ax.set_xlabel("sentiment score $s_t$, after the burn-in"); ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    return fig


def absorption(paths, closed, H, P):
    """left: a few runs per start, coloured by the closed class they end in; right: P(in the last class | k0),
    in the limit (H) and after as many rounds as the runs (P^T), which differ where the chain leaks slowly"""
    fig, (ax, aa) = plt.subplots(1, 2, figsize=(10, 4.1), gridspec_kw={"width_ratios": [3, 1.6], "wspace": 0.3})
    K = len(paths)
    for k0, p in paths.items():
        for r in range(SHOW):
            c = COLORS[2] if p[r, -1] in closed[-1] else COLORS[0]
            ax.step(np.arange(p.shape[1]), p[r] + 0.08 * (r - 1), where="post", color=c, lw=1.1, alpha=0.7)
    ax.set_xlim(0, 30); ax.set_ylim(K - 0.5, -0.5)
    ax.set_yticks(range(K)); ax.set_xlabel("round $t$"); ax.set_ylabel("instruction rung $k_t$")
    emp = [np.isin(p[:, -1], closed[-1]).mean() for p in paths.values()]
    T = p.shape[1] - 1   # the run's last rung is k_T, T moves after k_0
    aa.plot(range(K), H[:, -1], color=GREY, lw=1.4, label=r"$t \to \infty$")
    aa.plot(range(K), np.linalg.matrix_power(P, T)[:, closed[-1]].sum(1), color=INK, lw=1.4, ls=(0, (4, 2)),
            label=f"$t = {T}$")
    aa.scatter(range(K), emp, color=COLORS[2], s=30, zorder=3, label="simulated runs")
    aa.set_xticks(range(K)); aa.set_ylim(-0.05, 1.05)
    aa.set_xlabel("starting rung $k_0$"); aa.set_ylabel(f"P($k_t$ = {closed[-1][0]})")
    aa.legend(frameon=False, fontsize=11, loc="upper left")
    return fig


def escape(paths, k0, drift, K_signed):
    """rung k_t on the endless ladder: a few runs, their mean, and the predicted drift k0 + (p - q) t"""
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    t = np.arange(paths.shape[1])   # run() holds in round 0, so paths[:, 0] = k0
    ax.axhspan(0, K_signed - 1, color=GREY, alpha=0.25, lw=0)
    ax.text(3, K_signed + 2, "range of the signed ladder", color=INK, fontsize=12)
    for r in range(SHOW):
        ax.step(t, paths[r], where="post", color=COLORS[2], lw=1, alpha=0.6)
    ax.plot(t, paths.mean(0), color=COLORS[2], lw=2, label="mean over runs")
    ax.plot(t, k0 + drift * t, color=INK, ls=(0, (4, 2)), lw=1.4, label=r"$k_0 + (p - q)\,t$")
    ax.set_xlim(0, t[-1]); ax.set_ylim(0, None)
    ax.set_xlabel("round $t$"); ax.set_ylabel("instruction rung $k_t$")
    ax.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(0, 0.9))
    return fig
