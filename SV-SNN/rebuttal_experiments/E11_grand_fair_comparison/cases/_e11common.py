"""
E11 shared driver helpers for the MODULAR ELLIPTIC family (cases 2, 5, 6, 7).

These cases share an identical public API in their copied modules:
  - generate_data(seed)
  - run_svsnn_accelerated(data) -> (best_params, history, u_pred, n_params, total_time)
  - train_spinn(data, epochs=, params=None) -> result dict
  - init_spinn(key, features, n_layers, r), spinn_forward
  - init_siren(key, ff_dim, hidden, n_hidden), siren_u, siren_loss_fn
  - init_fourier_pinn(key, ff_dim, hidden_layers), fourier_pinn_u, fourier_pinn_loss_fn
  - init_pinn(key, layers_list), pinn_u, pinn_loss_fn
  - train_pointwise_method(name, params, loss_fn, predict_fn, data, epochs=)

run_modular_elliptic() dispatches one (method, budget, seed) and returns a
normalized record. SV-SNN keeps its best (modes,freqs) config untouched.
"""
import os
import sys
import time

import numpy as np
import jax
from jax import random

_HERE = os.path.dirname(os.path.abspath(__file__))
_E11 = os.path.dirname(_HERE)
if _E11 not in sys.path:
    sys.path.insert(0, _E11)
import harness  # noqa: E402


# Default ("rich") baseline configs == the originally published large networks.
RICH = {
    "spinn": dict(features=64, n_layers=4, r=64),
    "siren": dict(ff_dim=64, hidden=128, n_hidden=4),
    "fourier": dict(ff_dim=64, hidden_layers=[128, 128, 128, 1]),
    "pinn": dict(layers_list=[2, 128, 128, 128, 128, 1]),
}


def _matched_sizes(mod, method, target, seed):
    """Search a width that lands within +-10% of `target` SV-SNN params,
    using the case's REAL init functions (exact param count)."""
    key = random.PRNGKey(seed)

    if method == "SPINN":
        # SPINN has a high parameter floor (U/V/H input projections). Grid-search
        # over (ff_dim, features, n_layers) to land closest to target. The Fourier
        # embedding (ff_dim) may shrink for the matched budget; recorded honestly.
        best = None  # (err, sizes, count)
        for nl in (1, 2):
            for ff in (8, 12, 16, 24, 32, 48, 64):
                mod.FF_DIM_SPINN = ff
                for feat in range(1, 65):
                    try:
                        n = harness.count_params(
                            mod.init_spinn(key, features=feat, n_layers=nl, r=feat))
                    except Exception:
                        continue
                    err = abs(n - target)
                    if best is None or err < best[0]:
                        best = (err, dict(features=feat, n_layers=nl, r=feat, ff_dim=ff), n)
                    if n > target * 1.3:
                        break
        mod.FF_DIM_SPINN = best[1]["ff_dim"]
        ok = abs(best[2] - target) <= 0.10 * target
        return best[1], best[2], ok
    if method == "SIREN":
        make = lambda w: mod.init_siren(key, ff_dim=64, hidden=int(w), n_hidden=4)
        w, n, ok = harness.search_width(make, target, lo=1, hi=256)
        return dict(ff_dim=64, hidden=int(w), n_hidden=4), n, ok
    if method == "FourierPINN":
        make = lambda w: mod.init_fourier_pinn(
            key, ff_dim=64, hidden_layers=[int(w), int(w), int(w), 1])
        w, n, ok = harness.search_width(make, target, lo=1, hi=256)
        return dict(ff_dim=64, hidden_layers=[int(w), int(w), int(w), 1]), n, ok
    if method == "PINN":
        make = lambda w: mod.init_pinn(key, layers_list=[2, int(w), int(w), int(w), int(w), 1])
        w, n, ok = harness.search_width(make, target, lo=1, hi=400)
        return dict(layers_list=[2, int(w), int(w), int(w), int(w), 1]), n, ok
    raise ValueError(method)


def _infer_ms(fn):
    return harness.time_inference(fn)


def run_modular_elliptic(mod, method, budget, seed, epochs, target=None,
                         save_pred_path=None):
    mod.SEED = seed
    if epochs is not None:
        mod.EPOCHS = epochs
    EP = mod.EPOCHS
    n_steps = EP - 2
    data = mod.generate_data(seed)
    key = random.PRNGKey(seed)
    NC = mod.NC_SPINN
    matched_within = None
    tgt = None

    if method == "SVSNN":
        bp, hist, u_pred, n_params, t = mod.run_svsnn_accelerated(data)
        best = float(np.min(hist["l2_error"]))
        final = float(hist["l2_error"][-1])
        infer = _infer_ms(lambda: mod.svsnn_forward(bp, data["x_test_flat"], data["y_test_flat"]))
        n_coll = NC * NC
        u_pred_arr = u_pred

    elif method == "ClassicalSpectral":
        rec = _classical_helmholtz(mod, data)
        return rec

    else:
        # choose sizes
        if budget == "matched":
            assert target is not None, "matched needs target params"
            sizes, nfit, matched_within = _matched_sizes(mod, method, target, seed)
            tgt = target
        else:
            sizes = RICH[{"SPINN": "spinn", "SIREN": "siren",
                          "FourierPINN": "fourier", "PINN": "pinn"}[method]]

        if method == "SPINN":
            params = mod.init_spinn(key, features=sizes["features"],
                                    n_layers=sizes["n_layers"], r=sizes["r"])
            res = mod.train_spinn(data, epochs=EP, params=params)
            n_coll = NC * NC
            infer = _infer_ms(lambda: mod.spinn_forward(res["params"],
                                                        data["x_test_1d"], data["y_test_1d"]))
        elif method == "SIREN":
            params = mod.init_siren(key, ff_dim=sizes["ff_dim"],
                                    hidden=sizes["hidden"], n_hidden=sizes["n_hidden"])
            res = mod.train_pointwise_method("SIREN", params, mod.siren_loss_fn,
                                             mod.siren_u, data, epochs=EP)
            n_coll = int(mod.N_PDE)
            infer = _infer_ms(lambda: mod.siren_u(res["params"],
                                                  data["x_test_flat"], data["y_test_flat"]))
        elif method == "FourierPINN":
            params = mod.init_fourier_pinn(key, ff_dim=sizes["ff_dim"],
                                           hidden_layers=sizes["hidden_layers"])
            res = mod.train_pointwise_method("FourierPINN", params, mod.fourier_pinn_loss_fn,
                                             mod.fourier_pinn_u, data, epochs=EP)
            n_coll = int(mod.N_PDE)
            infer = _infer_ms(lambda: mod.fourier_pinn_u(res["params"],
                                                         data["x_test_flat"], data["y_test_flat"]))
        elif method == "PINN":
            params = mod.init_pinn(key, layers_list=sizes["layers_list"])
            res = mod.train_pointwise_method("PINN", params, mod.pinn_loss_fn,
                                             mod.pinn_u, data, epochs=EP)
            n_coll = int(mod.N_PDE)
            infer = _infer_ms(lambda: mod.pinn_u(res["params"],
                                                 data["x_test_flat"], data["y_test_flat"]))
        else:
            raise ValueError(method)

        best = float(res["best_l2_error"])
        final = float(res["final_l2_error"])
        n_params = int(res["total_params"])
        t = float(res["total_time_sec"])
        u_pred_arr = None

    rec = harness.normalize_record(
        method, budget, seed, params=n_params, best_l2=best, final_l2=final,
        train_time_sec=t, n_epochs=n_steps, n_collocation=n_coll,
        inference_ms=infer, target_params=tgt, matched_within_tol=matched_within)

    if save_pred_path is not None and u_pred_arr is not None:
        np.savez(save_pred_path, u_pred=np.asarray(u_pred_arr),
                 u_exact=np.asarray(data["u_exact_test"]),
                 X=np.asarray(data["X_test"]), Y=np.asarray(data["Y_test"]))
    return rec


# ==================================================================
# Monolithic-case helpers (cases 1, 3, 4, 8, 9)
# ==================================================================
def set_matched_ovr(mod, method, target, seed, arch):
    """Choose matched sizes via closed-form counters and write them into
    mod.E11_OVR (consumed by the monolithic run_* functions). Returns
    (est_count, within_tol_est)."""
    sizes, est, within = harness.choose_matched(
        method, target,
        in_dim=arch["in_dim"], out_dim=arch["out_dim"], n_coord=arch["n_coord"],
        ff=arch.get("ff", 64),
        n_hidden_pinn=arch.get("n_hidden_pinn", 4),
        n_hidden_fourier=arch.get("n_hidden_fourier", 3),
        n_hidden_siren=arch.get("n_hidden_siren", 4),
        spinn_n_branch=arch.get("spinn_n_branch", 2),
        spinn_per_out_weight=arch.get("spinn_per_out_weight", False))
    ovr = {}
    if method in ("PINN", "FourierPINN", "SIREN"):
        ovr["hidden"] = sizes["hidden"]
        ovr["n_hidden"] = sizes["n_hidden"]
        if "ff" in sizes:
            ovr["ff"] = sizes["ff"]
    elif method == "SPINN":
        ovr["spinn_features"] = sizes["features"]
        ovr["spinn_n_layers"] = sizes["n_layers"]
        ovr["spinn_r"] = sizes["r"]
        ovr["spinn_ff"] = sizes["ff"]
    mod.E11_OVR = ovr
    return est, within


def _classical_helmholtz(mod, data):
    """Classical sine-basis Galerkin reference for Helmholtz on the unit square
    (cases 2 & 6). Non-learning accuracy upper bound."""
    kappa = float(mod.KAPPA)
    X, Y, u_ex = data["X_test"], data["Y_test"], data["u_exact_test"]
    Ng = mod.N_TEST
    x = X[:, 0]; y = Y[0, :]
    t0 = time.time()
    f = (kappa ** 2) * np.sin(kappa * X) * np.sin(kappa * Y)
    M = 64
    ms = np.arange(1, M + 1)
    Sx = np.sin(np.outer(x, ms * np.pi))
    Sy = np.sin(np.outer(y, ms * np.pi))
    norm = (Sx.T @ Sx) / Ng
    Fmn = (Sx.T @ f @ Sy) * (2.0 / Ng) * (2.0 / Ng) / (4 * norm[0, 0] * norm[0, 0])
    lam = (np.add.outer(ms ** 2, ms ** 2)) * (np.pi ** 2)
    denom = lam - kappa ** 2
    Cmn = np.where(np.abs(denom) > 1e-8, Fmn / denom, 0.0)
    u_pred = Sx @ Cmn @ Sy.T
    solve_t = time.time() - t0
    err = harness.l2_rel(u_pred, u_ex)
    return harness.normalize_record(
        "ClassicalSpectral", "reference", 0, params=M * M, best_l2=err, final_l2=err,
        train_time_sec=solve_t, n_epochs=1, n_collocation=Ng * Ng,
        inference_ms=solve_t * 1000.0, target_params=None, matched_within_tol=None)
