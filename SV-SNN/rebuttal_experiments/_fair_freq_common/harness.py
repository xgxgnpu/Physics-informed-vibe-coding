"""
E11 Grand Fair Comparison - shared harness
==========================================
Shared utilities used by every case module and the run_one / run_all drivers.

This module is intentionally dependency-light (numpy + jax) and is *internal*
to the E11 directory; case modules in ./cases/ contain the validated physics
copied from svsnn_acceleration/caseX/run_accelerated.py so that the whole
E11 directory is self-contained and reproduces SV-SNN's best accuracy/speed.

Reviewer-facing metrics collected per run:
  - total_params
  - wall_clock_train_sec
  - ms_per_100_epoch         (explicitly requested: time per 100 steps)
  - peak_gpu_mem_mb          (peak_bytes_in_use, isolated per subprocess)
  - best_l2 / final_l2
  - inference_time_ms
  - n_collocation
"""

import time
import numpy as np
import jax


# ------------------------------------------------------------------
# Accuracy / size helpers
# ------------------------------------------------------------------
def l2_rel(u_pred, u_exact):
    u_pred = np.asarray(u_pred)
    u_exact = np.asarray(u_exact)
    return float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))


def count_params(params):
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(p.size for p in leaves if hasattr(p, "size")))


def gpu_peak_mb():
    """Peak GPU memory in MB. Accurate per-process because run_one isolates
    each (case, method, budget, seed) in its own subprocess."""
    try:
        s = jax.devices()[0].memory_stats()
        return float(s.get("peak_bytes_in_use", 0)) / 1e6
    except Exception:
        return float("nan")


# ------------------------------------------------------------------
# Parameter-matching search (strict +-10% target)
# ------------------------------------------------------------------
def search_width(make_params, target, lo=2, hi=512, tol=0.10):
    """Search a single integer width `w` so that count_params(make_params(w))
    is as close as possible to `target`. Returns (best_w, best_count, within_tol).

    `make_params(w)` must build a parameter pytree for width w (using the
    case's *real* init functions, so the count is exact)."""
    best_w, best_count, best_err = None, None, float("inf")
    for w in range(lo, hi + 1):
        try:
            n = count_params(make_params(w))
        except Exception:
            continue
        err = abs(n - target)
        if err < best_err:
            best_err, best_w, best_count = err, w, n
        # early exit once we start moving away monotonically past target
        if n > target * 1.6 and best_count is not None and best_count >= target:
            break
    within = best_count is not None and abs(best_count - target) <= tol * target
    return best_w, best_count, within


# ------------------------------------------------------------------
# Closed-form parameter counters (used to CHOOSE matched widths for the
# monolithic cases whose init functions are local). The ACTUAL param count is
# always re-read from the trained model and reported honestly; these formulas
# only pick a candidate width.
# ------------------------------------------------------------------
def _mlp_count(dims):
    return int(sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)))


def count_pinn(hidden, n_hidden, in_dim, out_dim):
    return _mlp_count([in_dim] + [hidden] * n_hidden + [out_dim])


def count_fourier(hidden, n_hidden, out_dim, n_coord, ff, siren=False):
    inp = n_coord * 2 * ff
    base = _mlp_count([inp] + [hidden] * n_hidden + [out_dim])
    return int(base + n_coord * ff)  # + frozen frequency vectors


def count_spinn(features, n_layers, r, n_branch, ff, out_dim=1, per_out_weight=False):
    inp = 2 * ff
    branch = (inp * features + features)
    branch += (n_layers - 1) * (features * features + features)
    branch += (features * r + r)
    total = n_branch * branch + n_branch * ff
    if per_out_weight:
        total += out_dim * r
    return int(total)


def choose_matched(method, target, *, in_dim, out_dim, n_coord, ff=64,
                   n_hidden_pinn=4, n_hidden_fourier=3, n_hidden_siren=4,
                   spinn_n_branch=2, spinn_per_out_weight=False, tol=0.10):
    """Pick a width that brings the closed-form param count closest to target.
    Returns (sizes_dict, est_count, within_tol_est)."""
    def best_width(fn, lo, hi):
        bw, bc, be = lo, None, float("inf")
        for w in range(lo, hi + 1):
            c = fn(w)
            e = abs(c - target)
            if e < be:
                be, bw, bc = e, w, c
            if c > target * 1.5 and bc is not None and bc >= target:
                break
        return bw, bc

    if method == "PINN":
        w, c = best_width(lambda h: count_pinn(h, n_hidden_pinn, in_dim, out_dim), 1, 400)
        sizes = dict(hidden=w, n_hidden=n_hidden_pinn)
    elif method == "FourierPINN":
        w, c = best_width(lambda h: count_fourier(h, n_hidden_fourier, out_dim, n_coord, ff), 1, 300)
        sizes = dict(hidden=w, n_hidden=n_hidden_fourier, ff=ff)
    elif method == "SIREN":
        w, c = best_width(lambda h: count_fourier(h, n_hidden_siren, out_dim, n_coord, ff, True), 1, 300)
        sizes = dict(hidden=w, n_hidden=n_hidden_siren, ff=ff)
    elif method == "SPINN":
        # grid over (ff, features) with n_layers=2; SPINN has a high floor.
        best = None
        for ffx in (8, 12, 16, 24, 32, 48, 64):
            for feat in range(1, 65):
                c = count_spinn(feat, 2, feat, spinn_n_branch, ffx, out_dim, spinn_per_out_weight)
                e = abs(c - target)
                if best is None or e < best[0]:
                    best = (e, dict(features=feat, n_layers=2, r=feat, ff=ffx), c)
                if c > target * 1.3:
                    break
        _, sizes, c = best
    else:
        raise ValueError(method)
    within = abs(c - target) <= tol * target
    return sizes, int(c), within


# ------------------------------------------------------------------
# Inference timing (full evaluation grid forward pass)
# ------------------------------------------------------------------
def time_inference(fwd_callable, n_repeat=10):
    """fwd_callable() -> jax array (one full-grid forward). Returns ms/forward."""
    out = fwd_callable()
    try:
        out.block_until_ready()
    except Exception:
        pass
    t0 = time.time()
    for _ in range(n_repeat):
        out = fwd_callable()
    try:
        out.block_until_ready()
    except Exception:
        pass
    return (time.time() - t0) / n_repeat * 1000.0


def normalize_record(method, budget, seed, *, params, best_l2, final_l2,
                     train_time_sec, n_epochs, n_collocation,
                     inference_ms, target_params=None, matched_within_tol=None,
                     extra=None):
    """Assemble a uniform per-run record (JSON-serializable, no big arrays)."""
    rec = {
        "method": method,
        "budget": budget,
        "seed": int(seed),
        "total_params": int(params),
        "target_params": None if target_params is None else int(target_params),
        "matched_within_tol": matched_within_tol,
        "best_l2": float(best_l2),
        "final_l2": float(final_l2),
        "wall_clock_train_sec": float(train_time_sec),
        "ms_per_100_epoch": float(train_time_sec / max(n_epochs, 1) * 1000.0 * 100.0),
        "ms_per_epoch": float(train_time_sec / max(n_epochs, 1) * 1000.0),
        "peak_gpu_mem_mb": gpu_peak_mb(),
        "inference_time_ms": float(inference_ms),
        "n_collocation": int(n_collocation),
    }
    if extra:
        rec.update(extra)
    return rec
