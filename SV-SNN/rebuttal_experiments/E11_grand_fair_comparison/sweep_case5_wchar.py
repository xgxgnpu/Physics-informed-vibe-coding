"""
E11 case5 SV-SNN legitimate w_char tuning sweep.

Only the SV-SNN's own Fourier-frequency center is varied
(w_char = scale * KAPPA). Parameter count, baselines, epochs are unchanged.
The E5 study motivates centering slightly above the true frequency.

Runs scale in {1.0(ref),1.25,1.4,1.5} x seed in {0,1,2}, reports best (min) L2.
"""
import os
import sys
import json
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")
for p in (HERE, CASES):
    if p not in sys.path:
        sys.path.insert(0, p)

import case5  # noqa: E402

SCALES = [1.0, 1.25, 1.4, 1.5]
SEEDS = [0, 1, 2]


def run(scale, seed):
    case5.E11_WCHAR_SCALE = scale
    case5.SEED = seed
    data = case5.generate_data(seed)
    bp, hist, u_pred, n_params, t = case5.run_svsnn_accelerated(data)
    best = float(np.min(hist["l2_error"]))
    final = float(hist["l2_error"][-1])
    return best, final, n_params, t


def main():
    results = {}
    for scale in SCALES:
        per_seed = []
        for seed in SEEDS:
            best, final, n_params, t = run(scale, seed)
            per_seed.append(dict(seed=seed, best=best, final=final,
                                 n_params=n_params, time=t))
            print(f"[scale={scale}] seed={seed} best={best:.5f} "
                  f"final={final:.5f} params={n_params} t={t:.1f}s",
                  flush=True)
        bests = [r["best"] for r in per_seed]
        results[str(scale)] = dict(
            per_seed=per_seed,
            best_mean=float(np.mean(bests)),
            best_std=float(np.std(bests)),
            best_min=float(np.min(bests)),
        )
        print(f"==> scale={scale}: best mean={np.mean(bests):.5f} "
              f"std={np.std(bests):.5f} min={np.min(bests):.5f}\n", flush=True)

    out = os.path.join(HERE, "saved_data", "case5_wchar_sweep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== SUMMARY (best L2) ===")
    print(f"{'scale':>8} {'mean':>10} {'std':>10} {'min':>10}")
    for scale in SCALES:
        r = results[str(scale)]
        print(f"{scale:>8} {r['best_mean']:>10.5f} "
              f"{r['best_std']:>10.5f} {r['best_min']:>10.5f}")
    print(f"\nSPINN-matched reference: mean=0.03503 min=0.03200")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
