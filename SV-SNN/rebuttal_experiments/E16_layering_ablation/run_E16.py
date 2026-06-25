"""E16 - SV-SNN frequency-layering ablation, all 9 cases x 5 strategies x 3 seeds.

Strategies:
  default        : each case's original three-level sampler == E11 (reused, not rerun)
  S1_single      : all frequencies at the characteristic frequency wc
  S2_two         : low band + characteristic band (no high tail)
  S4_continuous  : single uniform band [1, fmax]
  S5_40_40_20    : 40% low / 40% characteristic / 20% high

The 'default' point is taken verbatim from the E11 SV-SNN runs (consistency),
the others run the real case code path with only the sampler swapped.
"""
import os, sys, json, subprocess, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
E15 = os.path.join(os.path.dirname(HERE), "E15_structure_ablation")
RUN_ONE = os.path.join(E15, "run_one_abl.py")
E11_RAW = os.path.join(os.path.dirname(HERE), "E11_grand_fair_comparison",
                       "saved_data", "raw")
SAVE = os.path.join(HERE, "saved_data")
os.makedirs(SAVE, exist_ok=True)

CASES = [f"case{i}" for i in range(1, 10)]
STRATEGIES = ["default", "S1_single", "S2_two", "S4_continuous", "S5_40_40_20"]
SEEDS = [0, 1, 2]


def reuse_default(case, seed, out):
    src = os.path.join(E11_RAW, f"{case}_SVSNN_best_seed{seed}.json")
    with open(src) as f:
        r = json.load(f)
    rec = {"case": case, "seed": seed, "strategy": "default", "scale": 1.0,
           "best_l2": float(r["best_l2"]), "final_l2": float(r["final_l2"]),
           "params": int(r["total_params"]), "time_s": float(r["wall_clock_train_sec"]),
           "source": "E11"}
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[E16] {case} default seed{seed}: reused E11 best_l2={rec['best_l2']:.4e}")


def main():
    for case in CASES:
        for strat in STRATEGIES:
            for seed in SEEDS:
                out = os.path.join(SAVE, f"{case}_{strat}_seed{seed}.json")
                if os.path.exists(out):
                    print(f"[E16] skip {os.path.basename(out)}")
                    continue
                if strat == "default":
                    reuse_default(case, seed, out)
                    continue
                cmd = [sys.executable, RUN_ONE, "--case", case, "--seed", str(seed),
                       "--strategy", strat, "--scale", "1.0", "--out", out]
                print("[E16] RUN", " ".join(cmd[2:]), flush=True)
                subprocess.run(cmd, check=False)
    print("[E16] done")


if __name__ == "__main__":
    main()
