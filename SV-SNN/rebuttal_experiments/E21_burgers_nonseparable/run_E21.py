"""E21 orchestrator -- same-paradigm fair comparison on the non-separable Burgers shock.

Shared substrate (identical for every method): spatial sine-spectral separation +
classical Galerkin-projection ODE -> beta_k(t) targets (computed ONCE and cached).
The comparison isolates the TEMPORAL REPRESENTATION that compresses beta_k(t):

  neural (matched budget, 3 seeds): svsnn / pinn / fourier / siren_permode
  classical reference (1 run):      chebyshev

Outputs:
  saved_data/galerkin_beta.npz       cached Galerkin targets
  saved_data/records_hybrid/*.json   per-run records
  saved_data/fields_hybrid/*.npz     seed-0 prediction fields (for plotting)
  saved_data/summary_hybrid.json     aggregated mean+-std + pure-residual appendix
"""
import os, sys, json, subprocess, statistics, time

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "saved_data")
RECDIR = os.path.join(SD, "records_hybrid")
os.makedirs(RECDIR, exist_ok=True)

NEURAL = ["svsnn", "pinn", "fourier", "siren_permode"]
SEEDS = [0, 1, 2]
CLASSICAL = ["chebyshev"]

ENV = dict(os.environ)
ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
ENV["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".6"


def precompute_galerkin():
    print(">>> Phase 0: computing/caching shared Galerkin beta_k(t)...", flush=True)
    sys.path.insert(0, HERE)
    import hybrid_engine as he
    t, b = he.get_galerkin()
    print(f"    cached: t{t.shape}, beta{b.shape}", flush=True)


def run_sweep():
    t0 = time.time()
    for m in NEURAL:
        for s in SEEDS:
            cmd = [sys.executable, os.path.join(HERE, "run_one_hybrid.py"),
                   "--method", m, "--seed", str(s)]
            if s == SEEDS[0]:
                cmd.append("--save_pred")
            print(f">>> {m} seed={s}", flush=True)
            r = subprocess.run(cmd, env=ENV)
            if r.returncode != 0:
                print(f"!!! FAILED: {m} seed={s} (rc={r.returncode})", flush=True)
    for m in CLASSICAL:
        cmd = [sys.executable, os.path.join(HERE, "run_one_hybrid.py"),
               "--method", m, "--seed", "0", "--save_pred"]
        print(f">>> {m} (classical reference)", flush=True)
        subprocess.run(cmd, env=ENV)
    print(f"Sweep wall time: {time.time() - t0:.1f}s", flush=True)


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(statistics.mean(vals)),
            "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)), "max": float(max(vals)),
            "n": len(vals), "vals": [float(v) for v in vals]}


def aggregate():
    keys = ["best_l2", "l2_exact", "wall_clock_train_sec", "ms_per_100_epoch",
            "peak_gpu_mem_mb", "inference_time_ms"]
    per_method = {}
    for m in NEURAL:
        recs = []
        for s in SEEDS:
            p = os.path.join(RECDIR, f"{m}_s{s}.json")
            if os.path.exists(p):
                recs.append(json.load(open(p)))
        if not recs:
            continue
        per_method[m] = {
            "kind": "neural (3 seeds)",
            "total_params": recs[0]["total_params"],
            "target_params": recs[0]["target_params"],
            "matched_within_tol": all(r["matched_within_tol"] for r in recs),
            "sizes": recs[0].get("sizes", {}),
            **{k: _agg([r[k] for r in recs]) for k in keys},
        }
    for m in CLASSICAL:
        p = os.path.join(RECDIR, f"{m}_s0.json")
        if os.path.exists(p):
            r = json.load(open(p))
            per_method[m] = {"kind": "classical reference (no NN, 1 run)",
                             "total_params": r["total_params"],
                             "matched_within_tol": r["matched_within_tol"],
                             **{k: r[k] for k in keys}}

    # pure-residual appendix (from the earlier residual sweep, if present)
    appendix = None
    res = os.path.join(SD, "summary.json")
    if os.path.exists(res):
        rs = json.load(open(res))
        appendix = {mm: {"best_l2_mean": d["best_l2"]["mean"],
                         "total_params": d["total_params"]}
                    for mm, d in rs.get("track_A", {}).items()}

    summary = {
        "experiment": "E21 same-paradigm Burgers shock (SV-SNN = v10 hybrid)",
        "pde": "u_t + u*u_x = nu*u_xx, nu=0.01/pi, x in [-1,1], t in [0,1]",
        "shared_substrate": ("spatial sine-spectral separation u=sum beta_k(t) sin(k pi x) "
                             "+ classical Galerkin-projection ODE (RK4) for beta_k(t). "
                             "ALL methods share this; the NN only COMPRESSES beta_k(t). "
                             "This is a spectral-neural HYBRID, NOT a pure-residual PINN "
                             "(see v10_critical_analysis.md)."),
        "comparison": "temporal representation that compresses beta_k(t)",
        "seeds": SEEDS,
        "reference": "ETDRK4 hi-res (burgers_reference_hires.npz)",
        "methods": per_method,
        "pure_residual_appendix": appendix,
        "appendix_note": ("Pure-residual PINNs (no Galerkin substrate) on the same shock "
                          "all struggle (SV-SNN residual ~0.33, best baseline ~5.6e-3); "
                          "this motivates the shared Galerkin substrate. See burgers_engine.py."),
    }
    with open(os.path.join(SD, "summary_hybrid.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    precompute_galerkin()
    run_sweep()
    s = aggregate()
    print("\n================ E21 SAME-PARADIGM SUMMARY ================")
    for m, d in s["methods"].items():
        bl = d["best_l2"]
        if isinstance(bl, dict):
            print(f"  {m:14s} [{d['kind']:24s}] params={d['total_params']:7d} "
                  f"L2={bl['mean']:.3e} +/- {bl['std']:.1e}  matched={d['matched_within_tol']}")
        else:
            print(f"  {m:14s} [{d['kind']:24s}] params={d['total_params']:7d} "
                  f"L2={bl:.3e}  matched={d['matched_within_tol']}")
    print("==========================================================")


if __name__ == "__main__":
    main()
