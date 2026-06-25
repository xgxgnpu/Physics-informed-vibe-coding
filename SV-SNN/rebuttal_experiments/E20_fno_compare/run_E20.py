"""E20 - data-driven FNO vs physics-informed SV-SNN.

Operator-learning task: parametric Poisson  -Lap u = f  on [0,1]^2 with random
multi-frequency manufactured solutions (see fno_jax.make_problem_bank). The FNO
learns f -> u from Ntrain labeled (f,u) pairs and is tested on a fixed held-out
set of 100 instances. SV-SNN solves a subset of the SAME held-out instances from
the PDE residual alone (no solution data).

We report, honestly, the paradigm trade-off:
  - accuracy (held-out relative L2),
  - DATA requirement (FNO Ntrain sweep vs SV-SNN's zero solution data),
  - amortized compute (FNO one-off train + fast inference per query vs SV-SNN
    per-instance solve) and the crossover number of queries,
  - params / peak GPU mem / inference time.

Each config runs in its own subprocess (clean memory / peak-mem). Modes:
  --mode fno    --ntrain N --seed S --out f.json
  --mode svsnn  --out s.json
  (no mode)     orchestrate all + aggregate -> saved_data/summary.json
"""
import os, sys, json, argparse, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(os.path.dirname(HERE), "_fair_freq_common")
for p in (HERE, COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)

N_GRID = 64
TRAIN_POOL = 1024
N_TEST = 100
NTRAINS = [64, 256, 1024]
SEEDS = [0, 1, 2]
SVSNN_SUBSET = 16          # held-out instances solved physics-informed
SVSNN_EPOCHS = 5000
FNO_EPOCHS = 300

RAW = os.path.join(HERE, "saved_data", "raw")


def run_fno(ntrain, seed, out):
    import jax
    import fno_jax as F
    pool = F.make_problem_bank(TRAIN_POOL, seed=0)          # fixed pool (nested subsets)
    test = F.make_problem_bank(N_TEST, seed=999)            # fixed held-out set
    Ftr, Utr, X, Y = F.build_grid_dataset(pool[:ntrain], N_GRID)
    Fte, Ute, _, _ = F.build_grid_dataset(test, N_GRID)
    key = jax.random.PRNGKey(seed)
    rec = F.train_fno(key, Ftr, Utr, Fte, Ute, X, Y,
                      modes=16, width=32, n_layers=4, epochs=FNO_EPOCHS, batch=32)
    rec.update({"method": "FNO", "seed": seed})
    json.dump(rec, open(out, "w"), indent=2)
    print(f"[E20] FNO ntrain={ntrain} seed{seed}: rel_l2={rec['test_rel_l2_mean']:.4e} "
          f"params={rec['params']} train={rec['train_time_s']:.1f}s "
          f"infer={rec['infer_ms_per_instance']:.3f}ms", flush=True)


def run_svsnn(out):
    import fair_engine as fe
    import fno_jax as F
    fe.N_TEST = N_GRID
    fe.NC = 120
    fe.EPOCHS = SVSNN_EPOCHS
    test = F.make_problem_bank(N_TEST, seed=999)
    recs = []
    for i in range(SVSNN_SUBSET):
        prob = F.make_svsnn_problem(test[i])
        r = fe.run_svsnn(prob, seed=0, wc=prob["w_char"])
        recs.append({"idx": i, "best_l2": r["best_l2"], "params": r["params"],
                     "time_s": r["time_s"], "kmax": prob["w_char"]})
        print(f"[E20] SVSNN inst {i:2d}: best_l2={r['best_l2']:.4e} "
              f"params={r['params']} t={r['time_s']:.1f}s", flush=True)
    l2 = np.array([x["best_l2"] for x in recs])
    out_rec = {"method": "SVSNN", "n_instances": len(recs),
               "rel_l2_mean": float(l2.mean()), "rel_l2_std": float(l2.std()),
               "rel_l2_min": float(l2.min()), "rel_l2_max": float(l2.max()),
               "params": int(recs[0]["params"]),
               "solve_time_s_mean": float(np.mean([x["time_s"] for x in recs])),
               "per_instance": recs}
    json.dump(out_rec, open(out, "w"), indent=2)
    print(f"[E20] SVSNN subset: rel_l2={out_rec['rel_l2_mean']:.4e}"
          f"+-{out_rec['rel_l2_std']:.4e} solve={out_rec['solve_time_s_mean']:.1f}s/inst",
          flush=True)


def orchestrate():
    os.makedirs(RAW, exist_ok=True)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=".6")
    for nt in NTRAINS:
        for seed in SEEDS:
            out = os.path.join(RAW, f"fno_nt{nt}_seed{seed}.json")
            if os.path.exists(out):
                print("skip", os.path.basename(out)); continue
            subprocess.run([sys.executable, __file__, "--mode", "fno",
                            "--ntrain", str(nt), "--seed", str(seed), "--out", out],
                           check=False, env=env)
    sout = os.path.join(RAW, "svsnn.json")
    if not os.path.exists(sout):
        subprocess.run([sys.executable, __file__, "--mode", "svsnn", "--out", sout],
                       check=False, env=env)
    aggregate()


def aggregate():
    summary = {"task": "parametric Poisson f->u, [0,1]^2, multi-frequency MMS",
               "n_grid": N_GRID, "n_test": N_TEST, "ntrains": NTRAINS,
               "fno": {}, "svsnn": None}
    for nt in NTRAINS:
        vals, tr, inf, par, mem = [], [], [], [], []
        for seed in SEEDS:
            p = os.path.join(RAW, f"fno_nt{nt}_seed{seed}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            vals.append(d["test_rel_l2_mean"]); tr.append(d["train_time_s"])
            inf.append(d["infer_ms_per_instance"]); par.append(d["params"])
            if d.get("peak_gpu_mb"):
                mem.append(d["peak_gpu_mb"])
        if vals:
            summary["fno"][str(nt)] = {
                "rel_l2_mean": float(np.mean(vals)), "rel_l2_std": float(np.std(vals)),
                "train_time_s_mean": float(np.mean(tr)),
                "infer_ms_per_instance_mean": float(np.mean(inf)),
                "params": int(np.mean(par)),
                "peak_gpu_mb": float(np.mean(mem)) if mem else None}
    sp = os.path.join(RAW, "svsnn.json")
    if os.path.exists(sp):
        summary["svsnn"] = json.load(open(sp))
    json.dump(summary, open(os.path.join(HERE, "saved_data", "summary.json"), "w"), indent=2)
    print("[E20] wrote summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="orchestrate")
    ap.add_argument("--ntrain", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.mode == "fno":
        run_fno(args.ntrain, args.seed, args.out)
    elif args.mode == "svsnn":
        run_svsnn(args.out)
    else:
        orchestrate()


if __name__ == "__main__":
    main()
