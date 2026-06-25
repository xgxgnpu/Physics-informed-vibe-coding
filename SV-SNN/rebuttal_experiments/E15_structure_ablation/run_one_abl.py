"""Run ONE SV-SNN ablation config (case, strategy, scale, seed) in isolation.

STRATEGY/SCALE are passed via env (read once by cases_abl/_abl.py) BEFORE the
case module is imported, so default + scale 1.0 reproduces the E11 SV-SNN run.
Used by both E15 (w_char scale sweep, strategy='default') and E16 (layering).
"""
import os, sys, json, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", default="default")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["ABL_STRATEGY"] = args.strategy
    os.environ["ABL_SCALE"] = repr(args.scale)

    HERE = os.path.dirname(os.path.abspath(__file__))
    CASES = os.path.join(HERE, "cases_abl")
    for p in (CASES, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)

    import importlib
    mod = importlib.import_module(args.case)
    rec = mod.E11_run("SVSNN", "rich", args.seed, epochs=args.epochs)
    out = {"case": args.case, "seed": args.seed, "strategy": args.strategy,
           "scale": args.scale, "best_l2": float(rec["best_l2"]),
           "final_l2": float(rec["final_l2"]), "params": int(rec["total_params"]),
           "time_s": float(rec["wall_clock_train_sec"])}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[abl] {args.case} strat={args.strategy} scale={args.scale} seed={args.seed}: "
          f"best_l2={out['best_l2']:.4e} params={out['params']} t={out['time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
