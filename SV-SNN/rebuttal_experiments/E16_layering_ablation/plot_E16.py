"""Aggregate + plot the E16 frequency-layering ablation.

Reads saved_data/{case}_{strategy}_seed{seed}.json, writes:
  figures/E16_layering_bars.png   (3x3 grid, best-L2 per strategy, log scale)
  saved_data/E16_summary.csv      (mean / std / min best-L2 per case x strategy)
"""
import os, json, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(HERE, "saved_data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

CASES = [f"case{i}" for i in range(1, 10)]
TITLES = {
    "case1": "C1  1D Heat ($\\kappa{=}20\\pi$)",
    "case2": "C2  2D Helmholtz ($\\kappa{=}24\\pi$)",
    "case3": "C3  2D Nonlinear Elliptic",
    "case4": "C4  1D Heat ($\\kappa{=}500\\pi$)",
    "case5": "C5  2D Helmholtz (cylinder)",
    "case6": "C6  2D Helmholtz ($\\kappa{=}48\\pi$)",
    "case7": "C7  2D Poisson (multi-hole)",
    "case8": "C8  Taylor-Green NS",
    "case9": "C9  Steady NS (2 cylinders)",
}
STRATS = ["default", "S1_single", "S2_two", "S4_continuous", "S5_40_40_20"]
LABELS = {
    "default": "3-level\n(ours)", "S1_single": "single", "S2_two": "two-level",
    "S4_continuous": "continuous", "S5_40_40_20": "40/40/20",
}


def load(case, strat):
    vals = []
    for f in sorted(glob.glob(os.path.join(SAVE, f"{case}_{strat}_seed*.json"))):
        with open(f) as fh:
            vals.append(json.load(fh)["best_l2"])
    return np.array(vals, dtype=float)


def main():
    rows = []
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, case in zip(axes.flat, CASES):
        means, stds, present = [], [], []
        for s in STRATS:
            v = load(case, s)
            if v.size == 0:
                means.append(np.nan); stds.append(np.nan)
            else:
                means.append(float(np.mean(v))); stds.append(float(np.std(v)))
                rows.append([case, s, len(v), np.mean(v), np.std(v), np.min(v)])
            present.append(s)
        x = np.arange(len(STRATS))
        colors = ["#1f77b4" if s == "default" else "#9aa7b5" for s in STRATS]
        ax.bar(x, means, yerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in STRATS], fontsize=8)
        ax.set_title(TITLES[case], fontsize=11)
        ax.set_ylabel("Relative $L_2$ error")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        # mark the best strategy
        finite = [(m, i) for i, m in enumerate(means) if np.isfinite(m)]
        if finite:
            bi = min(finite)[1]
            ax.annotate("best", (x[bi], means[bi]), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=7, color="green")
    fig.suptitle("E16  SV-SNN Frequency-Layering Ablation (best relative $L_2$, mean$\\pm$std over 3 seeds)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, "E16_layering_bars.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)

    with open(os.path.join(SAVE, "E16_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "strategy", "n_seeds", "best_l2_mean", "best_l2_std", "best_l2_min"])
        w.writerows(rows)
    print("wrote E16_summary.csv  (%d rows)" % len(rows))

    # markdown table to stdout
    print("\n| case | " + " | ".join(LABELS[s].replace("\n", " ") for s in STRATS) + " |")
    print("|" + "---|" * (len(STRATS) + 1))
    for case in CASES:
        cells = []
        for s in STRATS:
            v = load(case, s)
            cells.append(f"{np.mean(v):.3e}" if v.size else "-")
        print(f"| {case} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
