"""E17 - multiple-frequency fair comparison (matched budget, full baseline set).

Poisson  -Lap u = f  with a manufactured multi-component solution
    u = sum_j a_j sin(k_j x) sin(k_j y),   f = sum_j a_j (2 k_j^2) sin(k_j x) sin(k_j y).
Sweep spectrum richness (2/3/4 components). All frequency-aware methods receive the
same multi-level prior with w_char = k_max; PINN is the no-prior floor.

5 methods x 3 spectra x 3 seeds. Saves raw JSON (incl. per-component projected error)
to saved_data/raw and an aggregated mean+-std summary to saved_data/summary.json.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_fair_freq_common"))
import fair_engine as fe

fe.EPOCHS = 8000
fe.NC = 160          # finer tensor grid for the high-frequency components
fe.N_PDE = 6000
fe.N_TEST = 240

PI = np.pi
SPECTRA = {
    "2comp": [(8 * PI, 1.0), (24 * PI, 1.0)],
    "3comp": [(8 * PI, 1.0), (16 * PI, 1.0), (30 * PI, 1.0)],
    "4comp": [(6 * PI, 1.0), (14 * PI, 1.0), (24 * PI, 1.0), (36 * PI, 1.0)],
}
METHODS = ["SVSNN", "FourierPINN", "SIREN", "SPINN", "PINN"]
SEEDS = [0, 1, 2]

RAW = os.path.join(HERE, "saved_data", "raw")
os.makedirs(RAW, exist_ok=True)


def make_problem(name, comps):
    kmax = max(k for k, _ in comps)

    def u_exact(x, y):
        return sum(a * np.sin(k * x) * np.sin(k * y) for k, a in comps)

    def source(x, y):  # f = -Lap u  (so residual u_xx+u_yy+f = 0 at exact)
        return sum(a * (2.0 * k * k) * np.sin(k * x) * np.sin(k * y) for k, a in comps)

    return {"name": name, "u_exact": u_exact, "source": source,
            "w_char": float(kmax), "domain": (0.0, 1.0), "components": comps}


def main():
    summary = {}
    for sname, comps in SPECTRA.items():
        prob = make_problem(sname, comps)
        wc = prob["w_char"]
        target = fe.svsnn_target(prob, wc)
        print(f"\n==== spectrum {sname}  (k_max={wc/PI:.0f}pi, target params={target}) ====", flush=True)
        for m in METHODS:
            for seed in SEEDS:
                out = os.path.join(RAW, f"{sname}_{m}_seed{seed}.json")
                if os.path.exists(out):
                    print(f"  skip {os.path.basename(out)}"); continue
                if m == "SVSNN":
                    r = fe.run_svsnn(prob, seed, wc)
                else:
                    r = fe.run_baseline(m, prob, seed, wc, target)
                rec = {"spectrum": sname, "method": m, "seed": seed,
                       "best_l2": r["best_l2"], "params": r["params"],
                       "time_s": r["time_s"], "target": target,
                       "per_comp": r.get("per_comp")}
                with open(out, "w") as f:
                    json.dump(rec, f, indent=2)
                print(f"  {sname:6s} {m:12s} seed{seed}: L2={r['best_l2']:.4e} "
                      f"params={r['params']}", flush=True)

    # aggregate
    for sname in SPECTRA:
        summary[sname] = {}
        for m in METHODS:
            vals, pars, comp_errs = [], [], []
            for seed in SEEDS:
                p = os.path.join(RAW, f"{sname}_{m}_seed{seed}.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))
                vals.append(d["best_l2"]); pars.append(d["params"])
                if d.get("per_comp"):
                    comp_errs.append({k: v["rel_err"] for k, v in d["per_comp"].items()})
            if not vals:
                continue
            agg = {"best_l2_mean": float(np.mean(vals)), "best_l2_std": float(np.std(vals)),
                   "best_l2_min": float(np.min(vals)), "params": int(np.round(np.mean(pars)))}
            if comp_errs:
                keys = comp_errs[0].keys()
                agg["per_comp_mean"] = {k: float(np.mean([c[k] for c in comp_errs])) for k in keys}
            summary[sname][m] = agg
    with open(os.path.join(HERE, "saved_data", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[E17] wrote summary.json")


if __name__ == "__main__":
    main()
