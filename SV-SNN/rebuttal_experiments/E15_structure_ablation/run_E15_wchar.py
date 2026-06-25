"""E15a - SV-SNN characteristic-frequency (w_char) magnitude sweep.

All 9 cases x scale {0.6,0.8,1.0,1.2,1.5} x 3 seeds. The scale multiplies every
(frozen) frequency, i.e. shifts the whole spectrum while keeping the layering and
all other settings fixed. scale==1.0 is the E11 SV-SNN run (reused, not rerun).
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ONE = os.path.join(HERE, "run_one_abl.py")
E11_RAW = os.path.join(os.path.dirname(HERE), "E11_grand_fair_comparison",
                       "saved_data", "raw")
SAVE = os.path.join(HERE, "saved_data")
os.makedirs(SAVE, exist_ok=True)

CASES = [f"case{i}" for i in range(1, 10)]
SCALES = [0.6, 0.8, 1.0, 1.2, 1.5]
SEEDS = [0, 1, 2]


def reuse_e11(case, seed, out):
    with open(os.path.join(E11_RAW, f"{case}_SVSNN_best_seed{seed}.json")) as f:
        r = json.load(f)
    rec = {"case": case, "seed": seed, "strategy": "default", "scale": 1.0,
           "best_l2": float(r["best_l2"]), "final_l2": float(r["final_l2"]),
           "params": int(r["total_params"]), "time_s": float(r["wall_clock_train_sec"]),
           "source": "E11"}
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[E15w] {case} scale=1.0 seed{seed}: reused E11 best_l2={rec['best_l2']:.4e}")


def main():
    for case in CASES:
        for s in SCALES:
            tag = f"{s:.1f}".replace(".", "p")
            for seed in SEEDS:
                out = os.path.join(SAVE, f"{case}_wchar{tag}_seed{seed}.json")
                if os.path.exists(out):
                    print(f"[E15w] skip {os.path.basename(out)}")
                    continue
                if abs(s - 1.0) < 1e-9:
                    reuse_e11(case, seed, out)
                    continue
                cmd = [sys.executable, RUN_ONE, "--case", case, "--seed", str(seed),
                       "--strategy", "default", "--scale", str(s), "--out", out]
                print("[E15w] RUN", " ".join(cmd[2:]), flush=True)
                subprocess.run(cmd, check=False)
    print("[E15w] done")


if __name__ == "__main__":
    main()
