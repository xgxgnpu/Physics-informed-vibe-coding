"""
E11 publication-quality figures (English).

Reads:
  saved_data/summary_meanstd.csv   (aggregated mean+-std + best over seeds)
  saved_data/per_run_records.json  (every individual run)
  saved_data/pred_*_seed0.npz      (optional prediction fields for qualitative panel)

Produces (figures/):
  F1_best_l2_grouped.png      per-case best-L2 grouped bars (SV-SNN vs baselines matched/rich)
  F2_accuracy_vs_params.png   accuracy vs parameter count Pareto scatter
  F3_ms_per_100epoch.png      training cost (ms / 100 steps), matched vs rich
  F4_peak_gpu_mem.png         peak GPU memory, matched vs rich
  F5_accuracy_vs_walltime.png accuracy vs wall-clock training time Pareto
  F6_fields_seed0.png         (optional) SV-SNN predicted vs exact fields

Run AFTER run_all.py finishes (or partway; it tolerates missing rows).
"""
import os
import csv
import json
import math
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(HERE, "saved_data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

CASES = [f"case{i}" for i in range(1, 10)]
CASE_LABEL = {
    "case1": "C1: Heat 20\u03c0",
    "case2": "C2: Helmholtz 24\u03c0",
    "case3": "C3: Nonlinear elliptic",
    "case4": "C4: Heat 500\u03c0",
    "case5": "C5: Helmholtz cylinder",
    "case6": "C6: Helmholtz 48\u03c0",
    "case7": "C7: Poisson porous",
    "case8": "C8: Taylor-Green",
    "case9": "C9: Double-cylinder NS",
}

# series = (method, budget) -> display
SERIES = [
    ("SVSNN", "best"),
    ("SPINN", "matched"), ("SPINN", "rich"),
    ("SIREN", "matched"), ("SIREN", "rich"),
    ("FourierPINN", "matched"), ("FourierPINN", "rich"),
    ("PINN", "matched"), ("PINN", "rich"),
    ("ClassicalSpectral", "reference"),
]
SERIES_LABEL = {
    ("SVSNN", "best"): "SV-SNN",
    ("SPINN", "matched"): "SPINN (matched)",
    ("SPINN", "rich"): "SPINN (rich)",
    ("SIREN", "matched"): "SIREN (matched)",
    ("SIREN", "rich"): "SIREN (rich)",
    ("FourierPINN", "matched"): "FourierPINN (matched)",
    ("FourierPINN", "rich"): "FourierPINN (rich)",
    ("PINN", "matched"): "PINN (matched)",
    ("PINN", "rich"): "PINN (rich)",
    ("ClassicalSpectral", "reference"): "Classical spectral",
}
SERIES_COLOR = {
    ("SVSNN", "best"): "#d62728",
    ("SPINN", "matched"): "#aec7e8", ("SPINN", "rich"): "#1f77b4",
    ("SIREN", "matched"): "#98df8a", ("SIREN", "rich"): "#2ca02c",
    ("FourierPINN", "matched"): "#ffbb78", ("FourierPINN", "rich"): "#ff7f0e",
    ("PINN", "matched"): "#c5b0d5", ("PINN", "rich"): "#9467bd",
    ("ClassicalSpectral", "reference"): "#7f7f7f",
}
METHOD_MARKER = {
    "SVSNN": "*", "SPINN": "o", "SIREN": "s",
    "FourierPINN": "^", "PINN": "D", "ClassicalSpectral": "X",
}


def load_summary():
    p = os.path.join(SAVED, "summary_meanstd.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}; run run_all.py first")
    rows = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            def g(k):
                v = r.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")
            rows[(r["case"], r["method"], r["budget"])] = {
                "best_l2": g("best_l2_mean"), "best_l2_std": g("best_l2_std"),
                "best_l2_min": g("best_l2_min"),
                "params": g("total_params_mean"),
                "ms100": g("ms_per_100_epoch_mean"), "ms100_std": g("ms_per_100_epoch_std"),
                "mem": g("peak_gpu_mem_mb_mean"), "mem_std": g("peak_gpu_mem_mb_std"),
                "wall": g("wall_clock_train_sec_mean"), "wall_std": g("wall_clock_train_sec_std"),
                "infer": g("inference_time_ms_mean"),
                "within": r.get("matched_within_tol", ""),
            }
    return rows


def _present_cases(rows):
    return [c for c in CASES if any((c, m, b) in rows for (m, b) in SERIES)]


# ----------------------------------------------------------------------
# F1: per-case grouped best-L2 bars
# ----------------------------------------------------------------------
def fig1_grouped_l2(rows):
    cases = _present_cases(rows)
    if not cases:
        return
    series = [s for s in SERIES if any((c, *s) in rows for c in cases)]
    n_s = len(series)
    x = np.arange(len(cases))
    w = 0.8 / max(n_s, 1)
    fig, ax = plt.subplots(figsize=(max(11, 1.4 * len(cases)), 5.5))
    for i, s in enumerate(series):
        vals, errs = [], []
        for c in cases:
            d = rows.get((c, *s))
            vals.append(d["best_l2"] if d else np.nan)
            errs.append(d["best_l2_std"] if d else np.nan)
        vals = np.array(vals); errs = np.nan_to_num(np.array(errs))
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs,
               label=SERIES_LABEL[s], color=SERIES_COLOR[s],
               edgecolor="black", linewidth=0.3, error_kw=dict(lw=0.6))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABEL[c] for c in cases], rotation=30, ha="right")
    ax.set_ylabel("Relative $L_2$ error (mean over seeds, log)")
    ax.set_title("Best relative $L_2$ error per case: SV-SNN vs baselines (matched & rich budgets)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(os.path.join(FIG, "F1_best_l2_grouped.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
# F2: accuracy vs params scatter (Pareto)
# ----------------------------------------------------------------------
def fig2_acc_vs_params(rows):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    seen = set()
    for (case, method, budget), d in rows.items():
        if not (np.isfinite(d["params"]) and np.isfinite(d["best_l2"])):
            continue
        key = (method, budget)
        color = SERIES_COLOR.get(key, "#333333")
        ax.scatter(d["params"], d["best_l2"], s=120 if method == "SVSNN" else 55,
                   marker=METHOD_MARKER.get(method, "o"), color=color,
                   edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
        seen.add(key)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Parameter count (log)")
    ax.set_ylabel("Best relative $L_2$ error (log)")
    ax.set_title("Accuracy vs. parameter budget (all cases, all methods)")
    handles = [Line2D([0], [0], marker=METHOD_MARKER.get(k[0], "o"), color="w",
                      markerfacecolor=SERIES_COLOR.get(k, "#333"), markeredgecolor="black",
                      markersize=10, label=SERIES_LABEL[k])
               for k in SERIES if k in seen]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="best")
    fig.savefig(os.path.join(FIG, "F2_accuracy_vs_params.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
# generic per-case grouped metric bar (F3 ms/100ep, F4 peak mem)
# ----------------------------------------------------------------------
def _grouped_metric(rows, key, ylabel, title, fname, logy=True, std_key=None):
    cases = _present_cases(rows)
    if not cases:
        return
    series = [s for s in SERIES if any((c, *s) in rows for c in cases)]
    n_s = len(series)
    x = np.arange(len(cases))
    w = 0.8 / max(n_s, 1)
    fig, ax = plt.subplots(figsize=(max(11, 1.4 * len(cases)), 5.5))
    for i, s in enumerate(series):
        vals, errs = [], []
        for c in cases:
            d = rows.get((c, *s))
            vals.append(d[key] if d else np.nan)
            errs.append(d[std_key] if (d and std_key) else 0.0)
        vals = np.array(vals); errs = np.nan_to_num(np.array(errs))
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs,
               label=SERIES_LABEL[s], color=SERIES_COLOR[s],
               edgecolor="black", linewidth=0.3, error_kw=dict(lw=0.6))
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABEL[c] for c in cases], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


# ----------------------------------------------------------------------
# F5: accuracy vs wall-clock Pareto
# ----------------------------------------------------------------------
def fig5_acc_vs_time(rows):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    seen = set()
    for (case, method, budget), d in rows.items():
        if not (np.isfinite(d["wall"]) and np.isfinite(d["best_l2"])):
            continue
        key = (method, budget)
        ax.scatter(d["wall"], d["best_l2"], s=120 if method == "SVSNN" else 55,
                   marker=METHOD_MARKER.get(method, "o"),
                   color=SERIES_COLOR.get(key, "#333"),
                   edgecolor="black", linewidth=0.4, alpha=0.85, zorder=3)
        seen.add(key)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Wall-clock training time (s, log)")
    ax.set_ylabel("Best relative $L_2$ error (log)")
    ax.set_title("Accuracy vs. training time (all cases, all methods)")
    handles = [Line2D([0], [0], marker=METHOD_MARKER.get(k[0], "o"), color="w",
                      markerfacecolor=SERIES_COLOR.get(k, "#333"), markeredgecolor="black",
                      markersize=10, label=SERIES_LABEL[k])
               for k in SERIES if k in seen]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="best")
    fig.savefig(os.path.join(FIG, "F5_accuracy_vs_walltime.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
# F6: SV-SNN predicted fields (qualitative), seed0
# ----------------------------------------------------------------------
def _as_2d_pair(d):
    """Return (exact2d, pred2d) if a clean 2D comparison is available, else None."""
    if "u_pred" not in d.files or "u_exact" not in d.files:
        return None
    up = np.asarray(d["u_pred"]); ue = np.asarray(d["u_exact"])
    if up.shape != ue.shape:
        return None
    if up.ndim == 3:  # spatiotemporal (e.g. Taylor-Green) -> middle time slice
        mid = up.shape[2] // 2
        up = up[:, :, mid]; ue = ue[:, :, mid]
    if up.ndim != 2:
        return None
    return ue, up


def fig6_fields():
    files = sorted(glob.glob(os.path.join(SAVED, "pred_*_SVSNN_seed0.npz")))
    panels = []
    for fp in files:
        case = os.path.basename(fp).split("_")[1]
        pair = _as_2d_pair(np.load(fp))
        if pair is not None:
            panels.append((case, pair[0], pair[1]))
    if not panels:
        return
    n = len(panels)
    fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8.5), squeeze=False)
    for j, (case, ue, up) in enumerate(panels):
        err = np.abs(up - ue)
        for row, (img, ttl, cmap) in enumerate([
                (ue, "exact", "viridis"), (up, "SV-SNN", "viridis"), (err, "|err|", "magma")]):
            ax = axes[row][j]
            im = ax.imshow(img.T, origin="lower", cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(CASE_LABEL.get(case, case), fontsize=9)
            if j == 0:
                ax.set_ylabel(ttl)
    fig.suptitle("SV-SNN predicted fields vs exact (seed 0)", y=1.0)
    fig.savefig(os.path.join(FIG, "F6_fields_seed0.png"))
    plt.close(fig)


def main():
    rows = load_summary()
    fig1_grouped_l2(rows)
    fig2_acc_vs_params(rows)
    _grouped_metric(rows, "ms100", "Time per 100 steps (ms, log)",
                    "Training cost per 100 steps: SV-SNN vs baselines",
                    "F3_ms_per_100epoch.png", logy=True, std_key="ms100_std")
    _grouped_metric(rows, "mem", "Peak GPU memory (MB)",
                    "Peak GPU memory per run (isolated subprocess)",
                    "F4_peak_gpu_mem.png", logy=False, std_key="mem_std")
    fig5_acc_vs_time(rows)
    fig6_fields()
    print(f"[plot_E11] figures written to {FIG}")
    for f in sorted(os.listdir(FIG)):
        print("  -", f)


if __name__ == "__main__":
    main()
