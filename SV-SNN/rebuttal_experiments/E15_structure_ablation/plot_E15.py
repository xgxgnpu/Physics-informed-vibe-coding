"""Aggregate + plot the E15 structural / w_char ablation.

Panel A (figures/E15_wchar_curves.png): per-case best-L2 vs frequency scale.
Panel B (figures/E15_structure_bars.png): per-case structural decomposition at the
  SV-SNN-matched parameter budget, reusing E11:
    SV-SNN (full)      = separation + Fourier basis + multi-level freq init
    FourierPINN(match) = Fourier features, NO variable separation
    SPINN(match)       = variable separation, NO Fourier (tanh basis)
    PINN(match)        = neither (plain MLP)
Also writes saved_data/E15_wchar_summary.csv and saved_data/E15_structure.csv.
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
E11_CSV = os.path.join(os.path.dirname(HERE), "E11_grand_fair_comparison",
                       "saved_data", "summary_meanstd.csv")

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
SCALES = [0.6, 0.8, 1.0, 1.2, 1.5]


def tagof(s):
    return f"{s:.1f}".replace(".", "p")


def load_wchar(case, s):
    vals = []
    for f in sorted(glob.glob(os.path.join(SAVE, f"{case}_wchar{tagof(s)}_seed*.json"))):
        with open(f) as fh:
            vals.append(json.load(fh)["best_l2"])
    return np.array(vals, dtype=float)


def read_e11():
    d = {}
    with open(E11_CSV) as f:
        for r in csv.DictReader(f):
            d[(r["case"], r["method"], r["budget"])] = r
    return d


def plot_wchar():
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    rows = []
    for ax, case in zip(axes.flat, CASES):
        ms, ss = [], []
        for s in SCALES:
            v = load_wchar(case, s)
            if v.size:
                ms.append(np.mean(v)); ss.append(np.std(v))
                rows.append([case, s, len(v), np.mean(v), np.std(v), np.min(v)])
            else:
                ms.append(np.nan); ss.append(np.nan)
        ms = np.array(ms); ss = np.array(ss)
        ax.errorbar(SCALES, ms, yerr=ss, marker="o", capsize=3, color="#1f77b4")
        i1 = SCALES.index(1.0)
        if np.isfinite(ms[i1]):
            ax.scatter([1.0], [ms[i1]], color="red", zorder=5, label="ours (1.0)")
        ax.set_yscale("log")
        ax.set_title(TITLES[case], fontsize=11)
        ax.set_xlabel("frequency scale  ($\\times w_{char}$)")
        ax.set_ylabel("Relative $L_2$ error")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("E15a  SV-SNN Characteristic-Frequency Magnitude Sweep (mean$\\pm$std, 3 seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, "E15_wchar_curves.png")
    fig.savefig(out, dpi=150); print("wrote", out)
    with open(os.path.join(SAVE, "E15_wchar_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "scale", "n_seeds", "best_l2_mean", "best_l2_std", "best_l2_min"])
        w.writerows(rows)
    return rows


def plot_structure(e11):
    methods = [("SVSNN", "best", "SV-SNN\n(full)"),
               ("FourierPINN", "matched", "Fourier\n(no sep.)"),
               ("SPINN", "matched", "separable\n(no Fourier)"),
               ("PINN", "matched", "MLP\n(neither)")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    rows = []
    for ax, case in zip(axes.flat, CASES):
        ms, ss, labs = [], [], []
        for meth, bud, lab in methods:
            r = e11.get((case, meth, bud))
            if r:
                ms.append(float(r["best_l2_mean"])); ss.append(float(r["best_l2_std"]))
            else:
                ms.append(np.nan); ss.append(np.nan)
            labs.append(lab)
            rows.append([case, meth, bud, ms[-1], ss[-1]])
        x = np.arange(len(methods))
        colors = ["#1f77b4", "#9aa7b5", "#9aa7b5", "#9aa7b5"]
        ax.bar(x, ms, yerr=ss, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
        ax.set_title(TITLES[case], fontsize=11)
        ax.set_ylabel("Relative $L_2$ error")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.suptitle("E15b  Structural Decomposition at SV-SNN-Matched Budget (E11; mean$\\pm$std, 3 seeds)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, "E15_structure_bars.png")
    fig.savefig(out, dpi=150); print("wrote", out)
    with open(os.path.join(SAVE, "E15_structure.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "method", "budget", "best_l2_mean", "best_l2_std"])
        w.writerows(rows)
    return rows


def main():
    plot_wchar()
    e11 = read_e11()
    plot_structure(e11)
    # markdown w_char table
    print("\nw_char sweep (best L2 mean):")
    print("| case | " + " | ".join(f"x{s}" for s in SCALES) + " |")
    print("|" + "---|" * (len(SCALES) + 1))
    for case in CASES:
        cells = []
        for s in SCALES:
            v = load_wchar(case, s)
            cells.append(f"{np.mean(v):.3e}" if v.size else "-")
        print(f"| {case} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
