"""
E11 orchestrator
================
Enumerates every (case, method, budget, seed) configuration and launches each as
an isolated `run_one.py` subprocess (accurate per-run peak GPU memory + failure
isolation). The SV-SNN run for each (case, seed) is executed first so its param
count becomes the `--target` for the matched-budget baselines.

Aggregates all per-run JSON into:
  - saved_data/per_run_records.json   (every run)
  - saved_data/summary_meanstd.csv    (mean +- std + best over seeds)
  - saved_data/config_used.json       (what was actually run)

Usage:
  python run_all.py                       # full run (3 seeds, default epochs)
  python run_all.py --cases case2 case6   # subset
  python run_all.py --seeds 0 1 2 --smoke 200   # quick pipeline check
"""
import os
import sys
import json
import csv
import time
import argparse
import subprocess
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "saved_data", "raw")
SAVED = os.path.join(HERE, "saved_data")
FIG = os.path.join(HERE, "figures")
os.makedirs(RAW, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

ALL_CASES = [f"case{i}" for i in range(1, 10)]
BASELINES = ["SPINN", "SIREN", "FourierPINN", "PINN"]
CLASSICAL_CASES = {"case2", "case6"}  # clean square Helmholtz -> classical spectral ref


def run_subprocess(case, method, budget, seed, epochs, target, tag, save_pred):
    cmd = [sys.executable, os.path.join(HERE, "run_one.py"),
           "--case", case, "--method", method, "--budget", budget,
           "--seed", str(seed), "--tag", tag]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if target is not None:
        cmd += ["--target", str(int(target))]
    if save_pred is not None:
        cmd += ["--save_pred", save_pred]
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    t0 = time.time()
    subprocess.run(cmd, env=env)
    return time.time() - t0


def load_rec(tag):
    p = os.path.join(RAW, tag + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=ALL_CASES)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", type=int, default=None,
                    help="override epochs for a quick pipeline check")
    ap.add_argument("--skip_done", action="store_true", default=True)
    ap.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    args = ap.parse_args()

    records = []
    config_used = {}
    t_start = time.time()

    for case in args.cases:
        for seed in args.seeds:
            # 1) SV-SNN first -> gives matched target
            sv_tag = f"{case}_SVSNN_best_seed{seed}"
            save_pred = (os.path.join(SAVED, f"pred_{case}_SVSNN_seed{seed}.npz")
                         if seed == 0 else None)
            if not (args.skip_done and load_rec(sv_tag)):
                run_subprocess(case, "SVSNN", "best", seed, args.smoke, None,
                               sv_tag, save_pred)
            sv = load_rec(sv_tag)
            if sv is None or sv.get("status") != "ok":
                print(f"[run_all] WARNING SV-SNN failed for {case} seed{seed}; "
                      f"skipping its baselines.")
                if sv is not None:
                    records.append(sv)
                continue
            records.append(sv)
            target = sv["total_params"]
            config_used.setdefault(case, {})["svsnn_params"] = target

            # 2) baselines x {matched, rich}
            for method in BASELINES:
                for budget in ("matched", "rich"):
                    tag = f"{case}_{method}_{budget}_seed{seed}"
                    sp = (os.path.join(SAVED, f"pred_{case}_{method}_{budget}_seed{seed}.npz")
                          if seed == 0 else None)
                    if not (args.skip_done and load_rec(tag)):
                        run_subprocess(case, method, budget, seed,
                                       args.smoke, target if budget == "matched" else None,
                                       tag, sp)
                    r = load_rec(tag)
                    if r is not None:
                        records.append(r)

            # 3) classical spectral reference (seed-independent; run once at seed0)
            if case in CLASSICAL_CASES and seed == args.seeds[0]:
                tag = f"{case}_ClassicalSpectral_ref"
                if not (args.skip_done and load_rec(tag)):
                    run_subprocess(case, "ClassicalSpectral", "reference", 0,
                                   None, None, tag, None)
                r = load_rec(tag)
                if r is not None:
                    records.append(r)

    # ---- persist raw records ----
    with open(os.path.join(SAVED, "per_run_records.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(SAVED, "config_used.json"), "w") as f:
        json.dump(config_used, f, indent=2)

    # ---- aggregate mean+-std over seeds ----
    groups = defaultdict(list)
    for r in records:
        if r.get("status") != "ok":
            continue
        groups[(r["case"], r["method"], r["budget"])].append(r)

    metrics = ["total_params", "best_l2", "final_l2", "wall_clock_train_sec",
               "ms_per_100_epoch", "peak_gpu_mem_mb", "inference_time_ms", "n_collocation"]
    rows = []
    for (case, method, budget), rs in sorted(groups.items()):
        row = {"case": case, "method": method, "budget": budget,
               "n_seeds": len(rs),
               "target_params": rs[0].get("target_params"),
               "matched_within_tol": rs[0].get("matched_within_tol")}
        for m in metrics:
            vals = np.array([x[m] for x in rs if x.get(m) is not None], dtype=float)
            vals = vals[~np.isnan(vals)] if vals.size else vals
            if vals.size:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
        l2s = [x["best_l2"] for x in rs]
        row["best_l2_min"] = float(np.min(l2s))
        rows.append(row)

    csv_path = os.path.join(SAVED, "summary_meanstd.csv")
    if rows:
        fields = (["case", "method", "budget", "n_seeds", "target_params",
                   "matched_within_tol"]
                  + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
                  + ["best_l2_min"])
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)

    print(f"\n[run_all] DONE in {(time.time()-t_start)/60:.1f} min")
    print(f"  records: {len(records)}  groups: {len(rows)}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
