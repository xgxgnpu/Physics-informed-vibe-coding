"""E19 - inaccurate w_char (frequency prior) fair comparison.

Fixed true frequency  kappa = 24*pi,  u = sin(kappa x) sin(kappa y),  Poisson.
Every frequency-aware method is handed the SAME deliberately mis-scaled prior
    w_char = rho * kappa,   rho in {0.5,0.7,0.85,1.0,1.2,1.5,2.0}.
We measure how gracefully each architecture degrades under a wrong prior.

Methods:
  rho-swept (matched):  SVSNN, FourierPINN, SPINN, SIREN
  rho-independent:      PINN          (no prior; flat reference floor)
                        SVSNN_FFTauto (estimates w_char from the source FFT -> recovers
                                       kappa, the "rescue" baseline)

3 seeds. Saves raw JSON to saved_data/raw and aggregated summary.json.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_fair_freq_common"))
import fair_engine as fe

fe.EPOCHS = 8000
fe.NC = 160
fe.N_PDE = 6000
fe.N_TEST = 240

PI = np.pi
KAPPA = 24 * PI
RHOS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
SWEPT = ["SVSNN", "FourierPINN", "SPINN", "SIREN"]
SEEDS = [0, 1, 2]

RAW = os.path.join(HERE, "saved_data", "raw")
os.makedirs(RAW, exist_ok=True)


def problem():
    def u_exact(x, y):
        return np.sin(KAPPA * x) * np.sin(KAPPA * y)

    def source(x, y):
        return 2.0 * KAPPA * KAPPA * np.sin(KAPPA * x) * np.sin(KAPPA * y)

    return {"name": "wmis", "u_exact": u_exact, "source": source,
            "w_char": float(KAPPA), "domain": (0.0, 1.0)}


def estimate_wchar_from_source(prob, n=512):
    """FFT of the source along a non-nodal slice -> dominant angular wavenumber."""
    x = np.linspace(0, 1, n, endpoint=False)
    y0 = 0.37  # avoid sin(kappa*y)=0 nodes
    f = prob["source"](x.reshape(-1, 1), np.full((n, 1), y0)).reshape(-1)
    F = np.abs(np.fft.rfft(f - f.mean()))
    freqs = np.fft.rfftfreq(n, d=1.0 / n)  # cycles over [0,1]
    f0 = freqs[1 + int(np.argmax(F[1:]))]
    return float(2 * np.pi * f0)


def run(method, prob, seed, wc, target):
    if method.startswith("SVSNN"):
        return fe.run_svsnn(prob, seed, wc)
    return fe.run_baseline(method, prob, seed, wc, target)


def main():
    prob = problem()
    target = fe.svsnn_target(prob, KAPPA)
    wc_hat = estimate_wchar_from_source(prob)
    print(f"true kappa={KAPPA:.3f}  ({KAPPA/PI:.1f}pi), FFT estimate={wc_hat:.3f} "
          f"({wc_hat/PI:.2f}pi), target params={target}", flush=True)

    # rho-swept methods
    for rho in RHOS:
        wc = rho * KAPPA
        for m in SWEPT:
            for seed in SEEDS:
                out = os.path.join(RAW, f"rho{rho}_{m}_seed{seed}.json")
                if os.path.exists(out):
                    print(f"  skip {os.path.basename(out)}"); continue
                r = run(m, prob, seed, wc, target)
                json.dump({"rho": rho, "method": m, "seed": seed, "wc": wc,
                           "best_l2": r["best_l2"], "params": r["params"],
                           "time_s": r["time_s"], "target": target},
                          open(out, "w"), indent=2)
                print(f"  rho={rho:<4} {m:12s} seed{seed}: L2={r['best_l2']:.4e} "
                      f"params={r['params']}", flush=True)

    # rho-independent: PINN (no prior) and SVSNN_FFTauto (recovered prior)
    for m, wc in [("PINN", KAPPA), ("SVSNN_FFTauto", wc_hat)]:
        for seed in SEEDS:
            out = os.path.join(RAW, f"indep_{m}_seed{seed}.json")
            if os.path.exists(out):
                print(f"  skip {os.path.basename(out)}"); continue
            if m == "PINN":
                r = fe.run_baseline("PINN", prob, seed, wc, target)
            else:
                r = fe.run_svsnn(prob, seed, wc)
            json.dump({"method": m, "seed": seed, "wc": wc, "best_l2": r["best_l2"],
                       "params": r["params"], "time_s": r["time_s"], "wc_hat": wc_hat},
                      open(out, "w"), indent=2)
            print(f"  indep {m:14s} seed{seed}: L2={r['best_l2']:.4e}", flush=True)

    # aggregate
    summary = {"kappa": KAPPA, "wc_hat": wc_hat, "target": target,
               "rhos": RHOS, "swept": {}, "indep": {}}
    for m in SWEPT:
        summary["swept"][m] = {}
        for rho in RHOS:
            vals, pars = [], []
            for seed in SEEDS:
                p = os.path.join(RAW, f"rho{rho}_{m}_seed{seed}.json")
                if os.path.exists(p):
                    d = json.load(open(p)); vals.append(d["best_l2"]); pars.append(d["params"])
            if vals:
                summary["swept"][m][str(rho)] = {
                    "best_l2_mean": float(np.mean(vals)), "best_l2_std": float(np.std(vals)),
                    "best_l2_min": float(np.min(vals)), "params": int(np.round(np.mean(pars)))}
    for m in ["PINN", "SVSNN_FFTauto"]:
        vals, pars = [], []
        for seed in SEEDS:
            p = os.path.join(RAW, f"indep_{m}_seed{seed}.json")
            if os.path.exists(p):
                d = json.load(open(p)); vals.append(d["best_l2"]); pars.append(d["params"])
        if vals:
            summary["indep"][m] = {"best_l2_mean": float(np.mean(vals)),
                                   "best_l2_std": float(np.std(vals)),
                                   "best_l2_min": float(np.min(vals)),
                                   "params": int(np.round(np.mean(pars)))}
    json.dump(summary, open(os.path.join(HERE, "saved_data", "summary.json"), "w"), indent=2)
    print("[E19] wrote summary.json")


if __name__ == "__main__":
    main()
