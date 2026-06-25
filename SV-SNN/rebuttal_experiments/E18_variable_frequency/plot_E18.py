"""Plot the E18 variable/chirp-frequency comparison.

  figures/E18_curves.png : best relative L2 vs chirp ratio r, per method (log-y).
  figures/E18_fields.png : spatial |error| maps at the hardest ratio (r=8, seed 0),
                           showing where each method's error concentrates.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = json.load(open(os.path.join(HERE, "saved_data", "summary.json")))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

RATIOS = [1, 2, 4, 8]
METHODS = ["SVSNN", "FourierPINN", "SIREN", "SPINN", "PINN"]
MLABEL = {"SVSNN": "SV-SNN", "FourierPINN": "FourierPINN", "SIREN": "SIREN",
          "SPINN": "SPINN", "PINN": "PINN"}
COLORS = {"SVSNN": "#d62728", "FourierPINN": "#1f77b4", "SIREN": "#2ca02c",
          "SPINN": "#9467bd", "PINN": "#7f7f7f"}
MARK = {"SVSNN": "o", "FourierPINN": "s", "SIREN": "^", "SPINN": "D", "PINN": "x"}


def curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in METHODS:
        ms = [SUM[str(r)].get(m, {}).get("best_l2_mean", np.nan) for r in RATIOS]
        ss = [SUM[str(r)].get(m, {}).get("best_l2_std", 0.0) for r in RATIOS]
        ax.errorbar(RATIOS, ms, yerr=ss, marker=MARK[m], capsize=3,
                    color=COLORS[m], label=MLABEL[m], linewidth=1.8)
    ax.set_yscale("log"); ax.set_xscale("log", base=2)
    ax.set_xticks(RATIOS); ax.set_xticklabels([str(r) for r in RATIOS])
    ax.set_xlabel("chirp ratio  $r = k_{max}/k_{min}$  ($r{=}1$: single frequency)")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("E18  Variable-Frequency Poisson (matched params, mean$\\pm$std, 3 seeds)")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "E18_curves.png"); fig.savefig(out, dpi=150)
    print("wrote", out)


def fields():
    p = os.path.join(HERE, "saved_data", "fields_r8.npz")
    if not os.path.exists(p):
        print("no fields_r8.npz"); return
    d = np.load(p)
    X, Y = d["X"], d["Y"]
    present = [m for m in METHODS if f"{m}_err" in d.files]
    vmax = max(float(np.percentile(d[f"{m}_err"], 99)) for m in present)
    fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 4))
    if len(present) == 1:
        axes = [axes]
    for ax, m in zip(axes, present):
        im = ax.pcolormesh(X, Y, d[f"{m}_err"], cmap="inferno", vmin=0, vmax=vmax,
                           shading="auto")
        ax.set_title(f"{MLABEL[m]}"); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=axes, fraction=0.025, label="|error|")
    fig.suptitle("E18  Spatial Error at $r{=}8$ (local wavenumber grows left$\\to$right / bottom$\\to$top)",
                 fontsize=13)
    out = os.path.join(FIG, "E18_fields.png"); fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    curves()
    fields()
