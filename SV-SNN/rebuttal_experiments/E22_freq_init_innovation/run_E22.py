"""E22 -- Characteristic-frequency MULTI-LEVEL initialization (core innovation ablation).

Two ablation axes per case (cases 1-9), fully fair (same accelerated SV-SNN engine,
same matched budget, same seeds; the ONLY knob is the frequency initialization):

  levels   (--strategy):  default = 3-level (multi-level, 25/50/25)   <- core innovation
                          S2_two  = 2-level (low + characteristic)
                          S1_single = 1-level (all freqs pinned to w_char)
  w_char   (--scale):     0.6 (under) / 1.0 (correct) / 1.5 (over)    <- inaccurate prior

Matrix = 9 cases x 3 levels x 3 scales x 3 seeds = 243 cells.
Only 108 are NEW ({S1_single,S2_two} x {0.6,1.5}); the rest are REUSED bit-exactly:
  - scale 1.0, any level  -> E16_layering_ablation/saved_data/case{N}_{strategy}_seed{S}.json
  - strategy default, 0.6/1.5 -> E15_structure_ablation/saved_data/case{N}_wchar{tag}_seed{S}.json
    (default + scale 1.0 == E11 SV-SNN best, so consistency is guaranteed.)

Every cell ends up as saved_data/raw/case{N}_{strategy}_scale{tag}_seed{S}.json with a
uniform schema, aggregated into saved_data/E22_summary.csv.
"""
import os, sys, json, csv, shutil, subprocess, statistics, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(HERE, "saved_data", "raw")
os.makedirs(RAW, exist_ok=True)

E15 = os.path.join(ROOT, "E15_structure_ablation")
E16 = os.path.join(ROOT, "E16_layering_ablation", "saved_data")
RUN_ONE = os.path.join(E15, "run_one_abl.py")

CASES = [f"case{i}" for i in range(1, 10)]
LEVELS = ["default", "S2_two", "S1_single"]   # 3-level / 2-level / 1-level
SCALES = [0.6, 1.0, 1.5]                       # under / correct / over
SEEDS = [0, 1, 2]

LEVEL_NAME = {"default": "3-level (multi)", "S2_two": "2-level", "S1_single": "1-level"}

ENV = dict(os.environ)
ENV["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
ENV.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".6")


def scale_tag(s):
    return ("%.1f" % s).replace(".", "p")         # 0.6->0p6, 1.0->1p0, 1.5->1p5


def target_path(case, strat, scale, seed):
    return os.path.join(RAW, f"{case}_{strat}_scale{scale_tag(scale)}_seed{seed}.json")


def reuse_source(case, strat, scale, seed):
    """Return path to an existing bit-identical raw record, or None."""
    if abs(scale - 1.0) < 1e-9:
        p = os.path.join(E16, f"{case}_{strat}_seed{seed}.json")
        return p if os.path.exists(p) else None
    if strat == "default":
        p = os.path.join(E15, "saved_data", f"{case}_wchar{scale_tag(scale)}_seed{seed}.json")
        return p if os.path.exists(p) else None
    return None


def ensure_cell(case, strat, scale, seed, run_new=True):
    tgt = target_path(case, strat, scale, seed)
    if os.path.exists(tgt):
        return "cached"
    src = reuse_source(case, strat, scale, seed)
    if src is not None:
        rec = json.load(open(src))
        rec["case"], rec["strategy"], rec["scale"], rec["seed"] = case, strat, scale, seed
        rec["source"] = os.path.relpath(src, ROOT)
        json.dump(rec, open(tgt, "w"), indent=2)
        return "reused"
    if not run_new:
        return "missing"
    cmd = [sys.executable, RUN_ONE, "--case", case, "--seed", str(seed),
           "--strategy", strat, "--scale", str(scale), "--out", tgt]
    r = subprocess.run(cmd, env=ENV)
    return "ran" if r.returncode == 0 else "FAILED"


def build(run_new=True):
    counts = {"cached": 0, "reused": 0, "ran": 0, "missing": 0, "FAILED": 0}
    total = len(CASES) * len(LEVELS) * len(SCALES) * len(SEEDS)
    i = 0
    for case in CASES:
        for strat in LEVELS:
            for scale in SCALES:
                for seed in SEEDS:
                    i += 1
                    st = ensure_cell(case, strat, scale, seed, run_new=run_new)
                    counts[st] += 1
                    if st in ("ran", "FAILED"):
                        print(f"[{i}/{total}] {case} {strat} x{scale} s{seed}: {st}", flush=True)
    print("build summary:", counts, flush=True)
    return counts


def aggregate():
    rows = []
    for case in CASES:
        for strat in LEVELS:
            for scale in SCALES:
                vals = []
                for seed in SEEDS:
                    p = target_path(case, strat, scale, seed)
                    if os.path.exists(p):
                        vals.append(float(json.load(open(p))["best_l2"]))
                if not vals:
                    continue
                rows.append({
                    "case": case, "strategy": strat, "level": LEVEL_NAME[strat],
                    "scale": scale, "n_seeds": len(vals),
                    "best_l2_mean": statistics.mean(vals),
                    "best_l2_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    "best_l2_min": min(vals),
                })
    out = os.path.join(HERE, "saved_data", "E22_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "strategy", "level", "scale", "n_seeds",
                                          "best_l2_mean", "best_l2_std", "best_l2_min"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--reuse-only", action="store_true",
                    help="only copy reusable cells, do not run new ones")
    a = ap.parse_args()
    if not a.aggregate_only:
        build(run_new=not a.reuse_only)
    aggregate()
