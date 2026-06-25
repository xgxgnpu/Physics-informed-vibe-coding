"""
E11 - run ONE (case, method, budget, seed) in an isolated process.

Process isolation gives an accurate per-run peak GPU memory
(memory_stats["peak_bytes_in_use"]) and isolates any single failure.

Usage:
  python run_one.py --case case2 --method PINN --budget matched --seed 0 \
      [--epochs 10000] [--target 1170] [--save_pred path.npz]
"""
import os
import sys
import json
import argparse
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")
for p in (HERE, CASES):
    if p not in sys.path:
        sys.path.insert(0, p)

RAW = os.path.join(HERE, "saved_data", "raw")
os.makedirs(RAW, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--budget", default="rich")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--save_pred", default=None)
    ap.add_argument("--tag", default=None, help="optional filename tag")
    args = ap.parse_args()

    import importlib
    mod = importlib.import_module(args.case)

    tag = args.tag or f"{args.case}_{args.method}_{args.budget}_seed{args.seed}"
    out_path = os.path.join(RAW, tag + ".json")

    try:
        rec = mod.E11_run(args.method, args.budget, args.seed,
                          epochs=args.epochs, target=args.target,
                          save_pred_path=args.save_pred)
        rec["case"] = args.case
        rec["status"] = "ok"
        # refresh peak gpu (cumulative max for this isolated process)
        import harness
        rec["peak_gpu_mem_mb"] = harness.gpu_peak_mb()
    except Exception as e:
        rec = {"case": args.case, "method": args.method, "budget": args.budget,
               "seed": args.seed, "status": "error", "error": repr(e),
               "trace": traceback.format_exc()}
        print(rec["trace"], flush=True)

    with open(out_path, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[run_one] wrote {out_path}", flush=True)
    if rec.get("status") == "ok":
        print(f"  {args.case} {args.method} {args.budget} seed{args.seed}: "
              f"params={rec['total_params']} best_l2={rec['best_l2']:.4e} "
              f"time={rec['wall_clock_train_sec']:.1f}s "
              f"mem={rec['peak_gpu_mem_mb']:.0f}MB "
              f"ms/100ep={rec['ms_per_100_epoch']:.1f}", flush=True)


if __name__ == "__main__":
    main()
