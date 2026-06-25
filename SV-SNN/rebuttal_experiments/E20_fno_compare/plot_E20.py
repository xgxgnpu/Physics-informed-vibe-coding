"""Plot the E20 FNO (data-driven) vs SV-SNN (physics-informed) comparison.

  figures/E20_data_efficiency.png : held-out relative L2 vs FNO training-set size,
        with SV-SNN's data-free accuracy as a horizontal reference (SV-SNN uses 0
        solution samples).
  figures/E20_amortization.png    : total wall-clock vs number of query instances Q.
        FNO = one-off training + Q * inference; SV-SNN = Q * per-instance solve.
        The crossover Q* marks where amortized FNO becomes cheaper.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = json.load(open(os.path.join(HERE, "saved_data", "summary.json")))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

NTRAINS = SUM["ntrains"]
FNO = SUM["fno"]
SV = SUM["svsnn"]


def data_efficiency():
    fig, ax = plt.subplots(figsize=(8, 6))
    ms = [FNO[str(n)]["rel_l2_mean"] for n in NTRAINS]
    ss = [FNO[str(n)]["rel_l2_std"] for n in NTRAINS]
    ax.errorbar(NTRAINS, ms, yerr=ss, marker="s", capsize=4, color="#1f77b4",
                linewidth=2, label="FNO (data-driven)")
    if SV:
        sv = SV["rel_l2_mean"]; svs = SV["rel_l2_std"]
        ax.axhline(sv, color="#d62728", linewidth=2, label="SV-SNN (0 solution data)")
        ax.fill_between([min(NTRAINS), max(NTRAINS)], sv - svs, sv + svs,
                        color="#d62728", alpha=0.15)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(NTRAINS); ax.set_xticklabels([str(n) for n in NTRAINS])
    ax.set_xlabel("FNO training-set size  $N_{train}$  (labeled (f,u) pairs)")
    ax.set_ylabel("held-out relative $L_2$ error")
    ax.set_title("E20  Data efficiency: FNO needs labeled solutions, SV-SNN needs none")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "E20_data_efficiency.png"); fig.savefig(out, dpi=150)
    print("wrote", out)


def amortization():
    # use the best-data FNO (largest Ntrain) for the timing model
    nt = str(max(NTRAINS))
    fno_train = FNO[nt]["train_time_s_mean"]
    fno_infer = FNO[nt]["infer_ms_per_instance_mean"] / 1000.0
    sv_solve = SV["solve_time_s_mean"] if SV else None
    Q = np.arange(1, 4001)
    fno_total = fno_train + fno_infer * Q
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Q, fno_total, color="#1f77b4", linewidth=2,
            label=f"FNO: train {fno_train:.0f}s + {fno_infer*1000:.2f}ms/query")
    if sv_solve:
        sv_total = sv_solve * Q
        ax.plot(Q, sv_total, color="#d62728", linewidth=2,
                label=f"SV-SNN: {sv_solve:.1f}s/instance solve")
        # crossover
        diff = fno_total - sv_total
        sign = np.sign(diff)
        idx = np.where(np.diff(sign) != 0)[0]
        if idx.size:
            qstar = int(Q[idx[0]])
            ax.axvline(qstar, color="gray", ls=":", alpha=0.8)
            ax.annotate(f"crossover $Q^*\\approx${qstar}", (qstar, sv_solve * qstar),
                        textcoords="offset points", xytext=(8, 10), fontsize=10)
    ax.set_xlabel("number of query instances  $Q$")
    ax.set_ylabel("total wall-clock time (s)")
    ax.set_title("E20  Amortized compute: FNO amortizes over many queries, "
                 "SV-SNN is data-free per instance")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "E20_amortization.png"); fig.savefig(out, dpi=150)
    print("wrote", out)


def summary_table():
    print("\n| metric | FNO (Ntrain=%s) | SV-SNN |" % max(NTRAINS))
    print("|---|---|---|")
    nt = str(max(NTRAINS))
    f = FNO[nt]
    print(f"| held-out rel L2 | {f['rel_l2_mean']:.3e} | {SV['rel_l2_mean']:.3e} |")
    print(f"| params | {f['params']:,} | {SV['params']:,} |")
    print(f"| solution data needed | {nt} pairs | 0 |")
    print(f"| train/solve time | {f['train_time_s_mean']:.1f}s (once) | "
          f"{SV['solve_time_s_mean']:.1f}s/instance |")
    print(f"| inference/instance | {f['infer_ms_per_instance_mean']:.3f}ms | "
          f"{SV['solve_time_s_mean']:.1f}s |")


if __name__ == "__main__":
    data_efficiency()
    amortization()
    summary_table()
