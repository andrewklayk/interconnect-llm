"""Figures for orchestrator_loop.py's CSVs, in the deck's style (used by orchestrator_results.ipynb)."""
import glob, logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import gaussian_kde

COLORS = ["#235CA6", "#CB8E16", "#C8302C"]   # the deck's blue, gold, red
INK, GREY = "#42424D", "#B2B2B6"
BURN = 5   # rounds dropped before the long-run statistics
DOTTED = (0, (1, 2.5))
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # harmless "cursive" fallback warning
for f in glob.glob("/usr/share/fonts/opentype/linux-libertine/LinLibertine_R*.otf"):   # serif close to the deck's figures
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Linux Libertine O", "font.size": 16, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GREY, "mathtext.fontset": "custom",
                     "mathtext.it": "Linux Libertine O:italic", "mathtext.rm": "Linux Libertine O",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
                     "savefig.bbox": "tight", "savefig.dpi": 300})


def scores(runs):
    """one row per (y0, seed, round): y_prev is the score the manager read, y the score the team then produced"""
    d = runs.query("agent == 0")[["y0", "seed", "step", "pi_req", "y"]].copy()   # y, pi_req repeat on every agent's row
    d["y_prev"] = d.groupby(["y0", "seed"]).y.shift(fill_value=np.nan)
    d.loc[d.step == 0, "y_prev"] = d.y0
    return d


def sweep(sw, R):
    """manager's word probabilities and the expected next score over the last score y"""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for col, c, lab in [("pi_required", COLORS[0], "π(required | y)"), ("pi_optional", COLORS[1], "π(optional | y)"),
                        ("E_next_y", INK, "E[next score]")]:
        ax.plot(sw.y, sw[col], color=c, lw=1.8, ls=(0, (4, 2)) if col == "E_next_y" else "-")
        ax.text(1.02, sw[col].iloc[-1], lab, color=c, va="center", fontsize=14, clip_on=False)   # direct labels
    ax.axvline(R, color=INK, ls=DOTTED, lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0.35, 0.82)
    ax.set_xticks([0, 0.2, 0.4, R, 0.8, 1], ["0", "0.2", "0.4", f"target {R:g}", "0.8", "1"])
    ax.set_xlabel("last team score $y$"); ax.set_ylabel("probability / score")
    return fig


def runs(runs, R):
    """team score per run, coloured by the starting score y0, and its long-run density per y0"""
    d = scores(runs)
    y0s, T = sorted(d.y0.unique()), d.step.max() + 1
    fig, (ax, ad) = plt.subplots(1, 2, figsize=(9, 4.1), sharey=True, gridspec_kw={"width_ratios": [4, 1], "wspace": 0.06})
    grid = np.linspace(0, 1, 400)
    for y0, c in zip(y0s, COLORS):
        for k, (_, run) in enumerate(d[d.y0 == y0].groupby("seed")):
            ax.plot(np.r_[0, run.step + 1], np.r_[y0, run.y], color=c, lw=1.4, alpha=0.8,
                    label=rf"from $y_0 = {y0:g}$" if k == 0 else None)
        dens = gaussian_kde(d[(d.y0 == y0) & (d.step >= BURN)].y, bw_method=0.3)(grid)
        dens /= dens.max()   # shape only: a narrow band would otherwise flatten the others
        ad.fill_betweenx(grid, 0, dens, color=c, alpha=0.15, lw=0)
        ad.plot(dens, grid, color=c, lw=1.6)
    for a in (ax, ad):
        a.axhline(R, color=INK, ls=DOTTED, lw=1)
    ax.set_xlim(0, T); ax.set_ylim(0, 1.02)
    ax.set_xticks(range(0, T + 1, 5)); ax.set_yticks([0, R, 1], ["0", f"target {R:g}", "1"])
    ax.set_xlabel("round $t$"); ax.set_ylabel("team security score $y_t$")
    ax.legend(frameon=False, fontsize=13, loc="lower right", handlelength=1.4)
    ad.set_xticks([]); ad.spines["bottom"].set_visible(False); ad.tick_params(left=False)
    ad.set_title("long run", style="italic", fontsize=15, color=INK)
    return fig


def round_map(runs, R, sw=None):
    """every observed round: the manager's P(required) and the next score against the score it read;
    with a sweep, its curves are drawn on top (what the rounds scatter around)"""
    d = scores(runs)
    fig, (am, ay) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.3})
    for y0, c in zip(sorted(d.y0.unique()), COLORS):
        s = d[d.y0 == y0]
        am.scatter(s.y_prev, s.pi_req, s=14, color=c, alpha=0.6, lw=0)
        ay.scatter(s.y_prev, s.y, s=14, color=c, alpha=0.6, lw=0, label=rf"from $y_0 = {y0:g}$")
    if sw is not None:
        am.plot(sw.y, sw.pi_required, color=INK, lw=1.2)
        ay.plot(sw.y, sw.E_next_y, color=INK, lw=1.2, ls=(0, (4, 2)), label="E[next score]")
    ay.plot([0, 1], [0, 1], color=GREY, lw=1)   # fixed points of the round map sit on the diagonal
    for a in (am, ay):
        a.axvline(R, color=INK, ls=DOTTED, lw=1)
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel("score the manager read, $y_t$")
    am.set_ylabel("π(required | $y_t$)"); ay.set_ylabel("next score $y_{t+1}$")
    ay.legend(frameon=False, fontsize=12, loc="lower right", handlelength=1.4)
    return fig


def long_run(runs):
    """mean and std of the score after BURN rounds, per starting score"""
    d = scores(runs)
    return d[d.step >= BURN].groupby("y0").y.agg(["mean", "std"]).round(3)
