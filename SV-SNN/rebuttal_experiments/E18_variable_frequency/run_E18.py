"""E18 - variable / chirp-frequency fair comparison (matched budget, full baselines).

Chirp solution on [0,1]^2:
    phi(x) = k0 * x * (1 + c*x)  ->  local wavenumber phi'(x) = k0 (1 + 2 c x),
    u = sin(phi(x)) sin(phi(y)),   f = -Lap u  (analytic).
The local wavenumber grows from k0 (at 0) to k0(1+2c) (at 1); chirp ratio
    r = k_max / k_min = 1 + 2c  ->  c = (r-1)/2.
Sweep r in {1,2,4,8} (r=1 is the single-frequency control). w_char = k_max,
wideband multi-level prior. 5 methods x 4 ratios x 3 seeds.

Saves raw JSON to saved_data/raw, aggregated summary.json, and error fields for the
hardest ratio (r=8, seed 0) to saved_data/fields_r8.npz for the spatial error map.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_fair_freq_common"))
import fair_engine as fe

fe.EPOCHS = 8000
fe.NC = 200
fe.N_PDE = 6000
fe.N_TEST = 240

PI = np.pi
K0 = 6 * PI
RATIOS = [1, 2, 4, 8]
METHODS = ["SVSNN", "FourierPINN", "SIREN", "SPINN", "PINN"]
SEEDS = [0, 1, 2]

RAW = os.path.join(HERE, "saved_data", "raw")
os.makedirs(RAW, exist_ok=True)


def make_problem(r):
    c = (r - 1) / 2.0
    kmax = K0 * (1 + 2 * c)

    def phi(t):
        return K0 * t * (1 + c * t)

    def dphi(t):
        return K0 * (1 + 2 * c * t)

    ddphi = 2 * K0 * c

    def u_exact(x, y):
        return np.sin(phi(x)) * np.sin(phi(y))

    def source(x, y):
        sx, cx = np.sin(phi(x)), np.cos(phi(x))
        sy, cy = np.sin(phi(y)), np.cos(phi(y))
        u = sx * sy
        # u_xx = -dphi(x)^2 u + ddphi * cx * sy ; u_yy analogous
        lap = (-dphi(x) ** 2 * u + ddphi * cx * sy) + (-dphi(y) ** 2 * u + ddphi * sx * cy)
        return -lap

    return {"name": f"r{r}", "u_exact": u_exact, "source": source,
            "w_char": float(kmax), "domain": (0.0, 1.0)}


def main():
    fields = {}
    for r in RATIOS:
        prob = make_problem(r)
        wc = prob["w_char"]
        target = fe.svsnn_target(prob, wc)
        print(f"\n==== chirp ratio r={r}  (k_min={K0/PI:.0f}pi, k_max={wc/PI:.0f}pi, "
              f"target={target}) ====", flush=True)
        for m in METHODS:
            for seed in SEEDS:
                out = os.path.join(RAW, f"r{r}_{m}_seed{seed}.json")
                want_field = (r == 8 and seed == 0)
                if os.path.exists(out) and not want_field:
                    print(f"  skip {os.path.basename(out)}"); continue
                if m == "SVSNN":
                    res = fe.run_svsnn(prob, seed, wc, return_pred=want_field)
                else:
                    res = fe.run_baseline(m, prob, seed, wc, target, return_pred=want_field)
                rec = {"ratio": r, "method": m, "seed": seed, "best_l2": res["best_l2"],
                       "params": res["params"], "time_s": res["time_s"], "target": target}
                with open(out, "w") as f:
                    json.dump(rec, f, indent=2)
                if want_field:
                    fields[f"{m}_err"] = np.abs(res["u_pred"] - res["UE"])
                    fields["X"] = res["X"]; fields["Y"] = res["Y"]; fields["UE"] = res["UE"]
                print(f"  r{r} {m:12s} seed{seed}: L2={res['best_l2']:.4e} "
                      f"params={res['params']}", flush=True)
    if fields:
        np.savez_compressed(os.path.join(HERE, "saved_data", "fields_r8.npz"), **fields)
        print("[E18] wrote fields_r8.npz")

    summary = {}
    for r in RATIOS:
        summary[str(r)] = {}
        for m in METHODS:
            vals, pars = [], []
            for seed in SEEDS:
                p = os.path.join(RAW, f"r{r}_{m}_seed{seed}.json")
                if os.path.exists(p):
                    d = json.load(open(p)); vals.append(d["best_l2"]); pars.append(d["params"])
            if vals:
                summary[str(r)][m] = {"best_l2_mean": float(np.mean(vals)),
                                      "best_l2_std": float(np.std(vals)),
                                      "best_l2_min": float(np.min(vals)),
                                      "params": int(np.round(np.mean(pars)))}
    with open(os.path.join(HERE, "saved_data", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[E18] wrote summary.json")


if __name__ == "__main__":
    main()
