"""Plot E1 results (English, journal style). Reads saved_data/, writes figures/."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "saved_data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

with open(os.path.join(SD, "per_seed_records.json")) as f:
    rec = json.load(f)

# read summary csv
import csv
rows = []
with open(os.path.join(SD, "summary_meanstd.csv")) as f:
    for r in csv.DictReader(f):
        rows.append(r)

cases = list(rec.keys())
method_order = ["SVSNN_accel", "SPINN", "SIREN", "FourierPINN", "PINN", "ClassicalSpectral"]
labels = {"SVSNN_accel": "SV-SNN (ours)", "SPINN": "SPINN", "SIREN": "SIREN",
          "FourierPINN": "FourierPINN", "PINN": "PINN", "ClassicalSpectral": "Classical Spectral"}
colors = {"SVSNN_accel": "#d62728", "SPINN": "#1f77b4", "SIREN": "#2ca02c",
          "FourierPINN": "#ff7f0e", "PINN": "#9467bd", "ClassicalSpectral": "#7f7f7f"}

def get(case, method, key):
    for r in rows:
        if r["case"] == case and r["method"] == method:
            return float(r[key])
    return np.nan

# ---------- Fig 1: best L2 (mean +- std) bar per case ----------
fig, axes = plt.subplots(1, len(cases), figsize=(6*len(cases), 4.5))
if len(cases) == 1: axes = [axes]
for ax, case in zip(axes, cases):
    ms = [m for m in method_order if any(r["case"]==case and r["method"]==m for r in rows)]
    means = [get(case, m, "best_l2_mean") for m in ms]
    stds = [get(case, m, "best_l2_std") for m in ms]
    x = np.arange(len(ms))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[colors[m] for m in ms], alpha=0.85, edgecolor="black", lw=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in ms], rotation=30, ha="right")
    ax.set_ylabel("Relative $L_2$ error")
    title = case.replace("helmholtz", "Helmholtz $\\kappa=").replace("pi", "\\pi$")
    ax.set_title(title)
    for xi, mn in zip(x, means):
        ax.text(xi, mn, f"{mn:.1e}", ha="center", va="bottom", fontsize=8)
fig.suptitle("E1: Fair comparison (5 seeds, mean$\\pm$std) — same frequency init for all baselines", y=1.02)
fig.savefig(os.path.join(FIG, "E1_accuracy_bars.png"))
plt.close(fig)

# ---------- Fig 2: compute metrics (time, memory) ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
case0 = cases[0]
ms = [m for m in method_order if m != "ClassicalSpectral" and any(r["case"]==case0 and r["method"]==m for r in rows)]
x = np.arange(len(ms))
times = [get(case0, m, "time_mean_s") for m in ms]
tstd = [get(case0, m, "time_std_s") for m in ms]
mems = [get(case0, m, "peak_mem_mb_mean") for m in ms]
axes[0].bar(x, times, yerr=tstd, capsize=4, color=[colors[m] for m in ms], alpha=0.85, edgecolor="black", lw=0.6)
axes[0].set_xticks(x); axes[0].set_xticklabels([labels[m] for m in ms], rotation=30, ha="right")
axes[0].set_ylabel("Training wall-clock (s)")
axes[0].set_title(f"Training time ({case0})")
for xi, t in zip(x, times): axes[0].text(xi, t, f"{t:.1f}", ha="center", va="bottom", fontsize=8)
axes[1].bar(x, mems, color=[colors[m] for m in ms], alpha=0.85, edgecolor="black", lw=0.6)
axes[1].set_xticks(x); axes[1].set_xticklabels([labels[m] for m in ms], rotation=30, ha="right")
axes[1].set_ylabel("Peak GPU memory (MB)")
axes[1].set_title(f"Peak GPU memory ({case0})")
for xi, mm in zip(x, mems): axes[1].text(xi, mm, f"{mm:.0f}", ha="center", va="bottom", fontsize=8)
fig.savefig(os.path.join(FIG, "E1_compute_metrics.png"))
plt.close(fig)

# ---------- Fig 3: params vs accuracy scatter ----------
fig, ax = plt.subplots(figsize=(7, 5.5))
for case in cases:
    ms = [m for m in method_order if any(r["case"]==case and r["method"]==m for r in rows)]
    for m in ms:
        p = get(case, m, "params"); e = get(case, m, "best_l2_mean")
        mk = "o" if "24" in case else "s"
        ax.scatter(p, e, color=colors[m], marker=mk, s=90, edgecolor="black", lw=0.6,
                   label=f"{labels[m]}" if case == cases[0] else None)
        ax.annotate(labels[m], (p, e), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of parameters"); ax.set_ylabel("Relative $L_2$ error (best, mean)")
ax.set_title("E1: Parameter efficiency vs accuracy\n(circle: $\\kappa{=}24\\pi$, square: $\\kappa{=}48\\pi$)")
ax.legend(fontsize=8, loc="lower left")
fig.savefig(os.path.join(FIG, "E1_params_vs_accuracy.png"))
plt.close(fig)

# ---------- Fig 4: prediction & error fields for SV-SNN (kappa=48pi) ----------
target_case = "helmholtz48pi" if "helmholtz48pi" in cases else cases[-1]
fp = os.path.join(SD, f"{target_case}_SVSNN_accel_pred.npz")
if os.path.exists(fp):
    d = np.load(fp)
    up, ue, X, Y = d["u_pred"], d["u_exact"], d["X"], d["Y"]
    err = np.abs(up - ue)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    im0 = axes[0].pcolormesh(X, Y, ue, cmap="RdBu_r", shading="auto"); axes[0].set_title("Exact $u$")
    im1 = axes[1].pcolormesh(X, Y, up, cmap="RdBu_r", shading="auto"); axes[1].set_title("SV-SNN prediction")
    im2 = axes[2].pcolormesh(X, Y, err, cmap="magma", shading="auto"); axes[2].set_title("Absolute error")
    for ax, im in zip(axes, [im0, im1, im2]):
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"SV-SNN on {target_case} (representative seed 0)", y=1.03)
    fig.savefig(os.path.join(FIG, "E1_svsnn_field.png"))
    plt.close(fig)

print("E1 figures written to", FIG)
