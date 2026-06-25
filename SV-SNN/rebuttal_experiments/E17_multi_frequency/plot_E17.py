"""Plot the E17 multi-frequency comparison.

  figures/E17_bars.png         : best relative L2 per method, grouped by spectrum (log).
  figures/E17_per_component.png: per-component projected error (how well each band is
                                 captured) for the 4-component spectrum, per method.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = json.load(open(os.path.join(HERE, "saved_data", "summary.json")))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

SPECTRA = ["2comp", "3comp", "4comp"]
SLABEL = {"2comp": "2 components\n[8,24]$\\pi$", "3comp": "3 components\n[8,16,30]$\\pi$",
          "4comp": "4 components\n[6,14,24,36]$\\pi$"}
METHODS = ["SVSNN", "FourierPINN", "SIREN", "SPINN", "PINN"]
MLABEL = {"SVSNN": "SV-SNN", "FourierPINN": "FourierPINN", "SIREN": "SIREN",
          "SPINN": "SPINN", "PINN": "PINN"}
COLORS = {"SVSNN": "#d62728", "FourierPINN": "#1f77b4", "SIREN": "#2ca02c",
          "SPINN": "#9467bd", "PINN": "#7f7f7f"}


def bars():
    x = np.arange(len(SPECTRA)); w = 0.16
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(METHODS):
        ms = [SUM[s].get(m, {}).get("best_l2_mean", np.nan) for s in SPECTRA]
        ss = [SUM[s].get(m, {}).get("best_l2_std", 0.0) for s in SPECTRA]
        ax.bar(x + (i - 2) * w, ms, w, yerr=ss, capsize=3, label=MLABEL[m],
               color=COLORS[m], edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([SLABEL[s] for s in SPECTRA])
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("E17  Multi-Frequency Poisson (matched params, mean$\\pm$std over 3 seeds)")
    ax.legend(ncol=5, fontsize=9, loc="upper left")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "E17_bars.png"); fig.savefig(out, dpi=150)
    print("wrote", out)


def per_component():
    s = "4comp"
    rec = SUM[s]
    bands = None
    for m in METHODS:
        pc = rec.get(m, {}).get("per_comp_mean")
        if pc:
            bands = list(pc.keys()); break
    if not bands:
        print("no per-component data"); return
    xlabels = [f"{float(b.split('_')[1])/np.pi:.0f}$\\pi$" for b in bands]
    x = np.arange(len(bands)); w = 0.16
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(METHODS):
        pc = rec.get(m, {}).get("per_comp_mean", {})
        vals = [pc.get(b, np.nan) for b in bands]
        ax.bar(x + (i - 2) * w, vals, w, label=MLABEL[m], color=COLORS[m],
               edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_xlabel("frequency component  $k_j$")
    ax.set_ylabel("per-component relative amplitude error")
    ax.set_title("E17  Per-Component Capture (4-component spectrum, mean over 3 seeds)")
    ax.legend(ncol=5, fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "E17_per_component.png"); fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    bars()
    per_component()
