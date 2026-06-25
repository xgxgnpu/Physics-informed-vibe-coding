"""Plot E12 noisy-BC robustness."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SD = os.path.join(HERE, "saved_data"); FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})
R = json.load(open(os.path.join(SD, "E12_results.json")))
eps_keys = sorted(R.keys(), key=lambda k: float(k.split("_")[1])); eps = [float(k.split("_")[1]) for k in eps_keys]
methods = ["SVSNN", "FourierPINN", "PINN"]
colors = {"SVSNN": "#d62728", "FourierPINN": "#ff7f0e", "PINN": "#9467bd"}
labels = {"SVSNN": "SV-SNN", "FourierPINN": "FourierPINN", "PINN": "PINN"}

# degradation curve
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for m in methods:
    mean = [R[k][m]["best_l2_mean"] for k in eps_keys]; std = [R[k][m]["best_l2_std"] for k in eps_keys]
    p = R[eps_keys[0]][m]["params"]
    ax.errorbar(np.array(eps) * 100, mean, yerr=std, marker="o", capsize=4, lw=2,
                color=colors[m], label=f"{labels[m]} ({p:,} params)")
ax.set_yscale("log"); ax.set_xlabel("boundary-measurement noise level  $\\epsilon$ (% of solution RMS)")
ax.set_ylabel("best relative $L_2$ error (vs clean solution)")
ax.set_title("E12: Robustness to noisy boundary data\nHelmholtz $\\kappa=24\\pi$, 3 seeds mean$\\pm$std")
ax.legend(); fig.savefig(os.path.join(FIG, "E12_degradation.png")); plt.close(fig)

# fields at eps=0.05
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
d = np.load(os.path.join(SD, "SVSNN_eps05_pred.npz")); up, ue, X, Y = d["u_pred"], d["ue"], d["X"], d["Y"]; err = np.abs(up - ue)
for ax, F, ttl, cmap in zip(axes, [ue, up, err], ["exact", "SV-SNN pred ($\\epsilon$=5%)", "abs error"], ["RdBu_r", "RdBu_r", "magma"]):
    im = ax.pcolormesh(X, Y, F, cmap=cmap, shading="auto"); ax.set_aspect("equal"); ax.set_title(ttl)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("E12: SV-SNN under 5% boundary noise (seed 0)", y=1.02)
fig.savefig(os.path.join(FIG, "E12_fields.png")); plt.close(fig)
print("E12 figures written")
