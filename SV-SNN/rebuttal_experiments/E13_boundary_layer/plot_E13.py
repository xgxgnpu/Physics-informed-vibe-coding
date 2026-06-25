"""Plot E13 boundary-layer results (honest)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SD = os.path.join(HERE, "saved_data"); FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})
R = json.load(open(os.path.join(SD, "E13_results.json")))
eps_keys = sorted(R.keys(), key=lambda k: -float(k.split("_")[1])); epsv = [float(k.split("_")[1]) for k in eps_keys]
methods = ["SVSNN", "FourierPINN", "PINN"]
colors = {"SVSNN": "#d62728", "FourierPINN": "#ff7f0e", "PINN": "#9467bd"}
labels = {"SVSNN": "SV-SNN", "FourierPINN": "FourierPINN", "PINN": "PINN"}

# L2 vs eps: mean+/-std (lines) and min (markers)
fig, ax = plt.subplots(figsize=(7.8, 5.4))
xx = np.arange(len(eps_keys))
for m in methods:
    mean = [R[k][m]["best_l2_mean"] for k in eps_keys]; std = [R[k][m]["best_l2_std"] for k in eps_keys]
    mn = [R[k][m]["best_l2_min"] for k in eps_keys]; p = R[eps_keys[0]][m]["params"]
    ax.errorbar(xx, mean, yerr=std, marker="o", capsize=4, lw=2, color=colors[m], label=f"{labels[m]} mean ({p:,} p)")
    ax.plot(xx, mn, marker="*", ls="--", ms=11, color=colors[m], alpha=0.7, label=f"{labels[m]} best-seed")
ax.axhline(1.0, color="gray", ls=":", lw=1); ax.text(0.02, 1.05, "L2=1 (=trivial zero)", color="gray", fontsize=8)
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels([f"$\\epsilon$={e}\n(layer width)" for e in epsv])
ax.set_ylabel("best relative $L_2$ error"); ax.set_title("E13: Boundary-layer problem (3 seeds). Thinner layer = harder.")
ax.legend(fontsize=8, ncol=3); fig.savefig(os.path.join(FIG, "E13_vs_eps.png")); plt.close(fig)

# slice profile at y=0.5 for eps=0.02 (seed 0 predictions)
ek = "eps_0.020"
fig, ax = plt.subplots(figsize=(8, 5))
d = np.load(os.path.join(SD, f"{ek}_SVSNN_pred.npz")); x1 = d["x1"]; gex = d["g_slice"]
ax.plot(x1, gex, "k-", lw=2.5, label="exact $g(x)$ (layer)")
for m in methods:
    sl = np.array(R[ek][m]["slice"]); ax.plot(x1, sl, lw=1.6, color=colors[m], alpha=0.85, label=f"{labels[m]} (seed0, L2={R[ek][m]['best_l2_mean']:.2f})")
ax.set_xlabel("x  (y=0.5 slice)"); ax.set_ylabel("u(x, 0.5)")
ax.set_title("E13: solution slice through the layer, $\\epsilon$=0.02"); ax.legend(fontsize=9)
fig.savefig(os.path.join(FIG, "E13_profile.png")); plt.close(fig)
print("E13 figures written")
