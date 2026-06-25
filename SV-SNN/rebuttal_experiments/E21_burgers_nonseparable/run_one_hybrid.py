"""Run ONE (method, seed) same-paradigm Burgers config in an isolated process.

Isolation gives an accurate per-process peak_gpu_mem_mb. Writes a JSON record and
(optionally) the seed-0 prediction field for plotting.
"""
import os, sys, json, argparse

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".6")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hybrid_engine as he


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--save_pred", action="store_true")
    a = ap.parse_args()

    rec = he.run(a.method, a.seed, return_pred=a.save_pred)

    recdir = os.path.join(HERE, "saved_data", "records_hybrid")
    os.makedirs(recdir, exist_ok=True)
    if a.save_pred:
        fdir = os.path.join(HERE, "saved_data", "fields_hybrid")
        os.makedirs(fdir, exist_ok=True)
        np.savez(os.path.join(fdir, f"{a.method}.npz"),
                 u_pred=rec.pop("u_pred"), UE=rec.pop("UE"),
                 xe=rec.pop("xe"), te=rec.pop("te"))
    else:
        for k in ("u_pred", "UE", "xe", "te"):
            rec.pop(k, None)

    out = os.path.join(recdir, f"{a.method}_s{a.seed}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[{a.method} seed={a.seed}] best_l2={rec['best_l2']:.4e} "
          f"params={rec['total_params']} matched={rec['matched_within_tol']} "
          f"t={rec['wall_clock_train_sec']:.1f}s -> {out}")


if __name__ == "__main__":
    main()
