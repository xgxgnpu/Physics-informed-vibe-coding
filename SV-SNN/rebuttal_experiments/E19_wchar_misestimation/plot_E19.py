"""Plot the E19 inaccurate-w_char comparison.

  figures/E19_ucurves.png : best relative L2 vs prior scale rho per frequency-aware
                            method (U-shaped), with the PINN floor (flat, no prior) and
                            the SV-SNN-FFTauto recovery point overlaid.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = json.load(open(os.path.join(HERE, "saved_data", "summary.json")))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

RHOS = SUM["rhos"]
SWEPT = ["SVSNN", "FourierPINN", "SPINN", "SIREN"]
MLABEL = {"SVSNN": "SV-SNN", "FourierPINN": "FourierPINN", "SIREN": "SIREN", "SPINN": "SPINN"}
COLORS = {"SVSNN": "#d62728", "FourierPINN": "#1f77b4", "SIREN": "#2ca02c", "SPINN": "#9467bd"}
MARK = {"SVSNN": "o", "FourierPINN": "s", "SIREN": "^", "SPINN": "D"}


def main():
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in SWEPT:
        d = SUM["swept"].get(m, {})
        ms = [d.get(str(r), {}).get("best_l2_mean", np.nan) for r in RHOS]
        ss = [d.get(str(r), {}).get("best_l2_std", 0.0) for r in RHOS]
        ax.errorbar(RHOS, ms, yerr=ss, marker=MARK[m], capsize=3, color=COLORS[m],
                    label=MLABEL[m], linewidth=1.8)
    pinn = SUM["indep"].get("PINN")
    if pinn:
        ax.axhline(pinn["best_l2_mean"], color="#7f7f7f", ls="--", linewidth=1.6,
                   label="PINN (no prior)")
    fft = SUM["indep"].get("SVSNN_FFTauto")
    if fft:
        rho_hat = SUM["wc_hat"] / SUM["kappa"]
        ax.scatter([rho_hat], [fft["best_l2_mean"]], s=140, marker="*", color="black",
                   zorder=6, label=f"SV-SNN-FFTauto (recovers $\\rho{{\\approx}}{rho_hat:.2f}$)")
    ax.axvline(1.0, color="gray", ls=":", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("prior scale  $\\rho = w_{char}/\\kappa$   ($\\rho{=}1$: correct prior)")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("E19  Robustness to an Inaccurate Frequency Prior "
                 "($\\kappa{=}24\\pi$, matched params, mean$\\pm$std, 3 seeds)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "E19_ucurves.png"); fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
