"""Figures for E22 -- characteristic-frequency multi-level initialization (core innovation).

Reads saved_data/E22_summary.csv. Writes figures/.
  fig1  9-panel grid: per case, L2 vs w_char scale {0.6,1.0,1.5}, one line per level
        {3-level/2-level/1-level}; E11 anchor (3-level @ 1.0) starred.
  fig2  robustness summary: per level, geo-mean degradation factor under wrong w_char
        + count of catastrophic cells (L2 > 0.1) across all 9x3 cells.
  fig3  compact level x scale heatmap (geo-mean L2 across cases).
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "saved_data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CASES = [f"case{i}" for i in range(1, 10)]
LEVELS = ["default", "S2_two", "S1_single"]
LEVEL_LABEL = {"default": "3-level (multi, core)", "S2_two": "2-level", "S1_single": "1-level"}
LEVEL_COLOR = {"default": "#c0392b", "S2_two": "#2c7fb8", "S1_single": "#7f8c8d"}
SCALES = [0.6, 1.0, 1.5]
CAT = 0.1   # catastrophic-failure threshold on relative L2


def load():
    d = {}
    with open(os.path.join(SD, "E22_summary.csv")) as f:
        for r in csv.DictReader(f):
            d[(r["case"], r["strategy"], float(r["scale"]))] = (
                float(r["best_l2_mean"]), float(r["best_l2_std"]), float(r["best_l2_min"]))
    return d


def fig1_panels(d):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, case in zip(axes.ravel(), CASES):
        for strat in LEVELS:
            ys = [d.get((case, strat, s), (np.nan, 0, 0))[0] for s in SCALES]
            es = [d.get((case, strat, s), (np.nan, 0, 0))[1] for s in SCALES]
            ax.errorbar(SCALES, ys, yerr=es, marker="o", capsize=3, lw=1.8,
                        color=LEVEL_COLOR[strat], label=LEVEL_LABEL[strat])
        anc = d.get((case, "default", 1.0), (np.nan,))[0]
        ax.scatter([1.0], [anc], marker="*", s=240, color="#f1c40f",
                   edgecolor="k", zorder=5, label="E11 best (anchor)")
        ax.set_yscale("log"); ax.set_title(case, fontsize=11)
        ax.set_xticks(SCALES); ax.set_xlabel(r"$w_{char}$ scale (1.0 = correct)")
        ax.set_ylabel("rel. $L_2$"); ax.grid(alpha=0.3, which="both")
        ax.axvline(1.0, color="k", ls=":", lw=0.8, alpha=0.5)
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.suptitle("E22: characteristic-frequency MULTI-LEVEL initialization across cases 1-9\n"
                 "(only knob = frequency init; 3-level stays accurate & robust to wrong "
                 "$w_{char}$, 1-level collapses)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "E22_fig1_panels.png"), dpi=140)
    plt.close(fig)
    print("saved E22_fig1_panels.png")


def _geomean(xs):
    xs = [x for x in xs if x > 0 and np.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def fig2_robustness(d):
    degr = {}; ncat = {}
    for strat in LEVELS:
        factors = []
        cat = 0
        for case in CASES:
            base = d.get((case, strat, 1.0), (np.nan,))[0]
            worst = max(d.get((case, strat, s), (np.nan,))[0] for s in (0.6, 1.5))
            if base and np.isfinite(base) and np.isfinite(worst) and base > 0:
                factors.append(worst / base)
            for s in SCALES:
                v = d.get((case, strat, s), (np.nan,))[0]
                if np.isfinite(v) and v > CAT:
                    cat += 1
        degr[strat] = _geomean(factors); ncat[strat] = cat

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    xs = np.arange(len(LEVELS))
    a1.bar(xs, [degr[s] for s in LEVELS], color=[LEVEL_COLOR[s] for s in LEVELS],
           edgecolor="k", alpha=0.9)
    for i, s in enumerate(LEVELS):
        a1.text(i, degr[s] * 1.05, f"{degr[s]:.1f}x", ha="center", fontsize=10)
    a1.set_yscale("log")
    a1.set_xticks(xs); a1.set_xticklabels([LEVEL_LABEL[s] for s in LEVELS], fontsize=9)
    a1.set_ylabel("degradation factor under wrong $w_{char}$\n(geo-mean over cases, lower=robust)")
    a1.set_title("Robustness to inaccurate $w_{char}$")
    a1.grid(axis="y", alpha=0.3, which="both")

    a2.bar(xs, [ncat[s] for s in LEVELS], color=[LEVEL_COLOR[s] for s in LEVELS],
           edgecolor="k", alpha=0.9)
    for i, s in enumerate(LEVELS):
        a2.text(i, ncat[s] + 0.2, str(ncat[s]), ha="center", fontsize=11)
    a2.set_xticks(xs); a2.set_xticklabels([LEVEL_LABEL[s] for s in LEVELS], fontsize=9)
    a2.set_ylabel(f"# catastrophic cells (L2 > {CAT})\nout of 9 cases x 3 scales = 27")
    a2.set_title("Catastrophic failures across the grid")
    a2.grid(axis="y", alpha=0.3)
    fig.suptitle("E22: multi-level (3-level) characteristic-frequency init is the robust core default",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG, "E22_fig2_robustness.png"), dpi=140)
    plt.close(fig)
    print("saved E22_fig2_robustness.png  degr=", {k: round(v, 2) for k, v in degr.items()},
          " ncat=", ncat)
    return degr, ncat


def fig3_heatmap(d):
    M = np.full((len(LEVELS), len(SCALES)), np.nan)
    for i, strat in enumerate(LEVELS):
        for j, s in enumerate(SCALES):
            M[i, j] = _geomean([d.get((c, strat, s), (np.nan,))[0] for c in CASES])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(np.log10(M), cmap="viridis_r", aspect="auto")
    for i in range(len(LEVELS)):
        for j in range(len(SCALES)):
            ax.text(j, i, f"{M[i, j]:.1e}", ha="center", va="center", color="w", fontsize=9)
    ax.set_xticks(range(len(SCALES))); ax.set_xticklabels([f"x{s}" for s in SCALES])
    ax.set_yticks(range(len(LEVELS))); ax.set_yticklabels([LEVEL_LABEL[s] for s in LEVELS])
    ax.set_xlabel(r"$w_{char}$ scale"); ax.set_ylabel("frequency-init levels")
    ax.set_title("E22: geo-mean rel. $L_2$ across cases 1-9\n(log color; lower=better)")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ rel. $L_2$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "E22_fig3_heatmap.png"), dpi=140)
    plt.close(fig)
    print("saved E22_fig3_heatmap.png")


def main():
    d = load()
    fig1_panels(d)
    fig2_robustness(d)
    fig3_heatmap(d)


if __name__ == "__main__":
    main()
