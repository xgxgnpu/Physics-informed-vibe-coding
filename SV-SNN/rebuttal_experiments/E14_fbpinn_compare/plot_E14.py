"""Plot E14: SV-SNN vs FBPINN vs FourierPINN on perforated domain."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SD = os.path.join(HERE, "saved_data"); FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150, "savefig.bbox": "tight"})
R = json.load(open(os.path.join(SD, "E14_results.json")))
order = ["SVSNN", "FBPINN", "FBPINN-rich", "FourierPINN"]
colors = {"SVSNN": "#d62728", "FBPINN": "#1f77b4", "FBPINN-rich": "#17becf", "FourierPINN": "#ff7f0e"}
labels = {"SVSNN": "SV-SNN", "FBPINN": "FBPINN (4x4)", "FBPINN-rich": "FBPINN (6x6, rich)", "FourierPINN": "FourierPINN"}

# error vs distance-to-boundary
fig, ax = plt.subplots(figsize=(7.8, 5.2))
for m in order:
    c = R[m]["dist_centers"]; e = R[m]["err_means"]
    ax.plot(c, e, marker="o", ms=4, lw=2, color=colors[m],
            label=f"{labels[m]}  (L2={R[m]['best_l2_mean']:.3f}, {R[m]['params']:,}p)")
ax.set_yscale("log"); ax.set_xlabel("distance to nearest boundary (outer or hole)")
ax.set_ylabel("mean absolute error"); ax.set_title("E14: error vs distance-to-boundary\nPoisson $\\mu=6\\pi$ on perforated domain, 3 seeds")
ax.legend(fontsize=9); fig.savefig(os.path.join(FIG, "E14_error_vs_distance.png")); plt.close(fig)

# bar chart accuracy/params
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
xs = np.arange(len(order))
ax1.bar(xs, [R[m]["best_l2_mean"] for m in order], yerr=[R[m]["best_l2_std"] for m in order],
        color=[colors[m] for m in order], capsize=4)
ax1.set_yscale("log"); ax1.set_xticks(xs); ax1.set_xticklabels([labels[m] for m in order], rotation=15, ha="right")
ax1.axhline(1.0, color="gray", ls=":"); ax1.set_ylabel("best masked $L_2$"); ax1.set_title("Accuracy (lower better)")
ax2.bar(xs, [R[m]["params"] for m in order], color=[colors[m] for m in order])
ax2.set_xticks(xs); ax2.set_xticklabels([labels[m] for m in order], rotation=15, ha="right")
ax2.set_ylabel("# parameters"); ax2.set_title("Parameter count")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "E14_summary_bars.png")); plt.close(fig)

# fields (SV-SNN vs FBPINN-rich vs FourierPINN error)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
files = {"SVSNN": "SVSNN_field.npz", "FBPINN-rich": "FBPINN_field.npz", "FourierPINN": "FourierPINN_field.npz"}
# FBPINN-rich field not saved separately (saved under FBPINN name overwrite); use available npz
import glob
npzs = {os.path.basename(p).replace("_field.npz", ""): p for p in glob.glob(os.path.join(SD, "*_field.npz"))}
show = [k for k in ["SVSNN", "FBPINN", "FourierPINN"] if k in npzs]
for col, m in enumerate(show):
    d = np.load(npzs[m]); up = d["u_pred"].copy(); ue = d["ue"].copy(); X = d["X"]; Y = d["Y"]; ih = d["inside_hole"]
    up[ih] = np.nan; err = np.abs(up - ue)
    im0 = axes[0, col].pcolormesh(X, Y, up, cmap="RdBu_r", shading="auto"); axes[0, col].set_title(f"{labels.get(m,m)} pred"); axes[0, col].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0, col], fraction=0.046)
    im1 = axes[1, col].pcolormesh(X, Y, err, cmap="magma", shading="auto"); axes[1, col].set_title(f"{labels.get(m,m)} |error|"); axes[1, col].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1, col], fraction=0.046)
fig.suptitle("E14: prediction (top) and error (bottom) on perforated domain", y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "E14_fields.png")); plt.close(fig)
print("E14 figures written")
