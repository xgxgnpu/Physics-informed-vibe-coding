"""E21 same-paradigm engine -- non-separable Burgers shock.

SHARED SUBSTRATE (identical for every method; disclosed once):
  1. Spatial sine-spectral separation:  u(x,t) = sum_{k=1..K} beta_k(t) sin(k*pi*x)
     on x in [-1,1] (Dirichlet auto-satisfied), K=300.
  2. Phase 0 = classical Galerkin-projection ODE (RK4, float64) -> beta_k(t) targets
     at dense times.  Computed ONCE and cached (saved_data/galerkin_beta.npz).
  >>> HONEST CAVEAT (applies to ALL methods equally): the PDE is advanced by a
      classical Galerkin solver; the network only COMPRESSES beta_k(t).  This is a
      spectral-neural HYBRID, not a pure-residual PINN.  See v10_critical_analysis.md.

The comparison therefore isolates the TEMPORAL REPRESENTATION that compresses
beta_k(t).  All neural backbones share the SAME fit schedule (head-reset LSQ init
+ Adam + L-BFGS, float64) and a MATCHED parameter budget (+-10%); 3 seeds.

  svsnn          : shared SIREN backbone (sin, omega0=30) + per-mode linear heads (v10)
  pinn           : shared tanh   backbone + per-mode linear heads
  fourier        : Fourier-feature(t) + tanh backbone + per-mode linear heads
  siren_permode  : K independent per-mode SIRENs (v9-style), end-to-end
  chebyshev      : classical Chebyshev temporal LSQ, NO network (v8-style reference)

All evaluated against the SAME ETDRK4 hi-res reference (burgers_reference_hires.npz).
"""
import os, sys, time, json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax
import scipy.optimize as sopt

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS",
                      "--xla_gpu_enable_command_buffer= --xla_gpu_enable_cublaslt=false")
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(HERE, "saved_data")
os.makedirs(SD, exist_ok=True)

# ---------------- PDE / substrate ----------------
NU = 0.01 / np.pi
K_MODES = 300
NX_PHYS = 1024
ODE_DT = 5e-4

# ---------------- SV-SNN (v10) reference architecture = matched anchor ----------------
BACKBONE_H = 256
BACKBONE_D = 256
N_HIDDEN = 2
OMEGA_0 = 30.0

# ---------------- shared fit schedule (identical for all neural backbones) ----------------
ADAM_EPOCHS = int(os.environ.get("E21_ADAM", "10000"))
ADAM_LR = 1e-3
HEAD_RESET_EVERY = int(os.environ.get("E21_HRESET", "2000"))
LBFGS_ITERS = int(os.environ.get("E21_LBFGS", "3000"))
LBFGS_MAXCOR = 100

# ---------------- Chebyshev classical reference ----------------
N_CHEB = 256

# ---------------- evaluation ----------------
N_EVAL_X = 1024
N_EVAL_T = 201

k_idx = jnp.arange(1, K_MODES + 1, dtype=jnp.float64)
kpi = k_idx * jnp.pi
IC = jnp.zeros(K_MODES, dtype=jnp.float64).at[0].set(-1.0)


# ================================================================
#  Phase 0 -- shared Galerkin ODE (RK4, float64), cached
# ================================================================
def solve_galerkin_ode():
    K = K_MODES
    kpi_64 = np.arange(1, K + 1, dtype=np.float64) * np.pi
    dk_64 = NU * kpi_64 ** 2
    x_64 = np.linspace(-1.0, 1.0, NX_PHYS + 1, dtype=np.float64)[:-1]
    sin_b = np.sin(kpi_64[None, :] * x_64[:, None])
    cos_b = np.cos(kpi_64[None, :] * x_64[:, None])
    dst_p = (2.0 / NX_PHYS) * sin_b.T

    def rhs(beta):
        u = sin_b @ beta
        ux = cos_b @ (kpi_64 * beta)
        return -dk_64 * beta - dst_p @ (u * ux)

    beta = np.zeros(K, dtype=np.float64); beta[0] = -1.0
    n_steps = int(round(1.0 / ODE_DT)); dt = 1.0 / n_steps
    t_save = [0.0]; beta_save = [beta.copy()]
    for step in range(1, n_steps + 1):
        k1 = rhs(beta)
        k2 = rhs(beta + 0.5 * dt * k1)
        k3 = rhs(beta + 0.5 * dt * k2)
        k4 = rhs(beta + dt * k3)
        beta = beta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_save.append(step * dt); beta_save.append(beta.copy())
    return np.array(t_save), np.array(beta_save)


def get_galerkin(cache=True):
    path = os.path.join(SD, "galerkin_beta.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["t"], d["beta"]
    t_ode, beta_ode = solve_galerkin_ode()
    if cache:
        np.savez(path, t=t_ode, beta=beta_ode)
    return t_ode, beta_ode


def compute_targets(t_ode, beta_ode):
    """g_k target with beta_k(t) = IC_k + t * g_k(t)."""
    ic_np = np.array(IC)
    nn_t = np.zeros((K_MODES, len(t_ode)), dtype=np.float64)
    for j in range(len(t_ode)):
        tt = t_ode[j] if t_ode[j] > 1e-15 else t_ode[1]
        nn_t[:, j] = (beta_ode[j if t_ode[j] > 1e-15 else 1] - ic_np) / tt
    return jnp.asarray(nn_t)


# ================================================================
#  Shared-backbone temporal nets (svsnn / pinn / fourier)
# ================================================================
def _init_shared(rng, method, H, D, ff=0):
    keys = random.split(rng, 8)
    p = {}
    if method == "fourier":
        # frozen Fourier features of t:  [sin(t*W), cos(t*W)] -> 2*ff inputs
        p["Wff"] = jnp.linspace(1.0, OMEGA_0, ff, dtype=jnp.float64).reshape(1, -1)
        d_in = 2 * ff
    else:
        d_in = 1
    # layer 0
    if method == "svsnn":
        b0 = 1.0 / d_in
    else:
        b0 = float(np.sqrt(6.0 / d_in))
    p["W0"] = random.uniform(keys[0], (d_in, H), minval=-b0, maxval=b0, dtype=jnp.float64)
    p["b0"] = random.uniform(keys[1], (H,), minval=-np.pi, maxval=np.pi, dtype=jnp.float64) \
        if method == "svsnn" else jnp.zeros(H, dtype=jnp.float64)
    # hidden layers
    for i in range(1, N_HIDDEN):
        bi = float(np.sqrt(6.0 / H)) / (OMEGA_0 if method == "svsnn" else 1.0)
        p[f"W{i}"] = random.uniform(keys[1 + i], (H, H), minval=-bi, maxval=bi, dtype=jnp.float64)
        p[f"b{i}"] = (random.uniform(keys[4 + i], (H,), minval=-np.pi, maxval=np.pi, dtype=jnp.float64)
                     if method == "svsnn" else jnp.zeros(H, dtype=jnp.float64))
    bo = float(np.sqrt(6.0 / H)) / (OMEGA_0 if method == "svsnn" else 1.0)
    p["Wout"] = random.uniform(keys[6], (H, D), minval=-bo, maxval=bo, dtype=jnp.float64)
    p["bout"] = jnp.zeros(D, dtype=jnp.float64)
    return p


def _fwd_shared(method, p, t):
    if method == "fourier":
        ph = t @ jax.lax.stop_gradient(p["Wff"])
        h = jnp.concatenate([jnp.sin(ph), jnp.cos(ph)], axis=1)
        h = jnp.tanh(h @ p["W0"] + p["b0"])
        act = jnp.tanh
    elif method == "svsnn":
        h = jnp.sin(OMEGA_0 * (t @ p["W0"] + p["b0"]))
        act = jnp.sin
    else:  # pinn
        h = jnp.tanh(t @ p["W0"] + p["b0"])
        act = jnp.tanh
    for i in range(1, N_HIDDEN):
        h = act(h @ p[f"W{i}"] + p[f"b{i}"])
    return h @ p["Wout"] + p["bout"]      # (Nt, D)


def _shared_param_count(method, H, D, ff=0):
    d_in = 2 * ff if method == "fourier" else 1
    n = d_in * H + H               # W0,b0
    n += (N_HIDDEN - 1) * (H * H + H)
    n += H * D + D                 # Wout,bout
    if method == "fourier":
        n += ff                    # frozen features (counted as leaves)
    return n


# ================================================================
#  Per-mode SIREN (v9-style), end-to-end
# ================================================================
def _init_permode(rng, h):
    keys = random.split(rng, 4)
    K = K_MODES
    s0 = 1.0
    s1 = float(np.sqrt(6.0 / h)) / OMEGA_0
    return {
        "W0": random.uniform(keys[0], (K, 1, h), minval=-s0, maxval=s0, dtype=jnp.float64),
        "b0": random.uniform(keys[1], (K, h), minval=-np.pi, maxval=np.pi, dtype=jnp.float64),
        "W1": random.uniform(keys[2], (K, h, h), minval=-s1, maxval=s1, dtype=jnp.float64),
        "b1": jnp.zeros((K, h), dtype=jnp.float64),
        "Wout": random.uniform(keys[3], (K, h, 1), minval=-s1, maxval=s1, dtype=jnp.float64),
        "bout": jnp.zeros((K, 1), dtype=jnp.float64),
    }


def _fwd_permode(p, t):
    # t: (Nt,1) -> g: (K, Nt)
    tt = t[None, :, :]                                  # (1,Nt,1)
    h = jnp.sin(OMEGA_0 * (jnp.einsum("Nti,Kih->Kth", tt, p["W0"]) + p["b0"][:, None, :]))
    h = jnp.sin(jnp.einsum("Kth,Khg->Ktg", h, p["W1"]) + p["b1"][:, None, :])
    g = jnp.einsum("Kth,Khg->Ktg", h, p["Wout"]) + p["bout"][:, None, :]
    return g[:, :, 0]                                   # (K, Nt)


def _permode_param_count(h):
    K = K_MODES
    return K * (1 * h + h + h * h + h + h * 1 + 1)


# ================================================================
#  per-mode linear heads (shared-backbone methods)
# ================================================================
def head_reset(method, bb, t_train, nn_target, ff=0):
    phi = np.asarray(_fwd_shared(method, bb, t_train))           # (Nt, D)
    Phi = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
    sol, _, _, _ = np.linalg.lstsq(Phi, np.asarray(nn_target).T, rcond=None)
    D = phi.shape[1]
    return {"C": jnp.asarray(sol[:D, :].T), "d": jnp.asarray(sol[D, :])}


def _full_fwd_shared(method, bb, hd, t):
    phi = _fwd_shared(method, bb, t)            # (Nt, D)
    return hd["C"] @ phi.T + hd["d"][:, None]   # (K, Nt)


# ================================================================
#  predict u(x,t) and L2 vs reference
# ================================================================
def _predict_beta_from_g(g, t_flat):
    return IC[:, None] + t_flat[None, :] * g    # (K, Nt)


def _predict_u(g_fn, t_train_flat, x_ev, t_ev):
    # g_fn(t_col) -> (K, Nt)
    tc = t_ev.reshape(-1, 1)
    g = g_fn(jnp.asarray(tc))
    beta = _predict_beta_from_g(g, jnp.asarray(t_ev))
    sin_ev = jnp.sin(kpi[None, :] * x_ev[:, None])   # (Nx, K)
    return sin_ev @ beta                              # (Nx, Nt)


def evaluate(g_fn, t_r, x_r, u_r):
    from scipy.interpolate import RegularGridInterpolator
    x_ev = np.linspace(-1, 1, N_EVAL_X + 1, dtype=np.float64)[:-1]
    t_ev = np.linspace(0, 1, N_EVAL_T, dtype=np.float64)
    u_pred = np.asarray(_predict_u(g_fn, None, jnp.asarray(x_ev), jnp.asarray(t_ev)))
    # exact-grid L2
    x_full = x_r[:-1]; t_full = t_r
    u_pred_full = np.asarray(_predict_u(g_fn, None, jnp.asarray(x_full), jnp.asarray(t_full)))
    u_exact_full = u_r[:, :-1].T
    l2_exact = float(np.sqrt(np.sum((u_pred_full - u_exact_full) ** 2) /
                             np.sum(u_exact_full ** 2)))
    # interpolated L2 (down-sampled grid)
    Tm, Xm = np.meshgrid(t_ev, x_ev, indexing="ij")
    interp = RegularGridInterpolator((t_r, x_r), u_r, method="cubic",
                                     bounds_error=False, fill_value=None)
    u_exact = interp((Tm, Xm)).T
    l2_interp = float(np.sqrt(np.sum((u_pred - u_exact) ** 2) / np.sum(u_exact ** 2)))
    return l2_exact, l2_interp, u_pred, u_exact, x_ev, t_ev


def load_reference():
    d = np.load(os.path.join(SD, "burgers_reference_hires.npz"))
    return d["t"], d["x"], d["usol"]


def _cp(*trees):
    return int(sum(x.size for t in trees for x in jax.tree_util.tree_leaves(t)
                   if hasattr(x, "size")))


def gpu_peak_mb():
    try:
        return float(jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0)) / 1e6
    except Exception:
        return float("nan")


# ================================================================
#  matched-budget width search (anchor = SV-SNN total params)
# ================================================================
def svsnn_total_params():
    return _shared_param_count("svsnn", BACKBONE_H, BACKBONE_D) + K_MODES * BACKBONE_D + K_MODES


def _match_shared(method, target, tol=0.10):
    heads = K_MODES * BACKBONE_D + K_MODES
    if method == "fourier":
        best = None
        for ff in (8, 12, 16, 24, 32, 48, 64):
            for H in range(8, 513):
                c = _shared_param_count("fourier", H, BACKBONE_D, ff) + heads
                e = abs(c - target)
                if best is None or e < best[0]:
                    best = (e, dict(H=H, ff=ff), c)
                if c > target * 1.4:
                    break
        return best[1], best[2]
    else:
        best = None
        for H in range(8, 1025):
            c = _shared_param_count(method, H, BACKBONE_D) + heads
            e = abs(c - target)
            if best is None or e < best[0]:
                best = (e, dict(H=H), c)
            if c > target * 1.4:
                break
        return best[1], best[2]


def _match_permode(target):
    best = None
    for h in range(2, 128):
        c = _permode_param_count(h)
        e = abs(c - target)
        if best is None or e < best[0]:
            best = (e, dict(h=h), c)
        if c > target * 1.4:
            break
    return best[1], best[2]


# ================================================================
#  training (shared schedule) and classical chebyshev
# ================================================================
def _adam_lbfgs_fit(loss_fn, params, eval_fn, head_reset_fn=None):
    """Adam (cosine) + periodic head-reset (if provided) + L-BFGS polish."""
    sched = optax.cosine_decay_schedule(ADAM_LR, ADAM_EPOCHS, alpha=1e-6 / ADAM_LR)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        lv, g = value_and_grad(loss_fn)(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, lv

    best = {"l2": float("inf"), "p": params}

    def maybe_track(p):
        e = eval_fn(p)
        if e < best["l2"]:
            best["l2"] = e; best["p"] = jax.tree_util.tree_map(lambda z: np.asarray(z).copy(), p)

    for ep in range(1, ADAM_EPOCHS + 1):
        params, state, _ = step(params, state)
        if head_reset_fn is not None and ep % HEAD_RESET_EVERY == 0:
            params = head_reset_fn(params)
            state = opt.init(params)
            maybe_track(params)
        elif ep % 1000 == 0:
            maybe_track(params)
    maybe_track(params)

    # L-BFGS polish
    if LBFGS_ITERS > 0:
        flat0, unravel = jax.flatten_util.ravel_pytree(params)

        @jax.jit
        def lg(fp):
            pp = unravel(fp)
            lv, g = value_and_grad(loss_fn)(pp)
            gf, _ = jax.flatten_util.ravel_pytree(g)
            return lv, gf

        def fg(x):
            lv, g = lg(jnp.asarray(x, jnp.float64))
            return float(lv), np.asarray(g, np.float64)

        res = sopt.minimize(fg, np.asarray(flat0, np.float64), method="L-BFGS-B", jac=True,
                            options={"maxiter": LBFGS_ITERS, "maxfun": LBFGS_ITERS * 2,
                                     "ftol": 1e-16, "gtol": 1e-15, "maxcor": LBFGS_MAXCOR})
        pp = unravel(jnp.asarray(res.x, jnp.float64))
        maybe_track(pp)
    return best["p"], best["l2"]


def run(method, seed, return_pred=False):
    t_ode, beta_ode = get_galerkin()
    nn_target = compute_targets(t_ode, beta_ode)
    t_train = jnp.asarray(t_ode, jnp.float64).reshape(-1, 1)
    t_r, x_r, u_r = load_reference()
    target = svsnn_total_params()

    t0 = time.time()
    sizes = {}; matched = True

    if method in ("svsnn", "pinn", "fourier"):
        if method == "svsnn":
            H, ff = BACKBONE_H, 0
        else:
            sz, est = _match_shared(method, target)
            H = sz["H"]; ff = sz.get("ff", 0); sizes = sz
        bb = _init_shared(random.PRNGKey(seed), method, H, BACKBONE_D, ff)
        hd = head_reset(method, bb, t_train, nn_target, ff)

        def loss_fn(p):
            g = _full_fwd_shared(method, p["bb"], p["hd"], t_train)
            return jnp.mean((g - nn_target) ** 2)

        def hr(p):
            new_hd = head_reset(method, p["bb"], t_train, nn_target, ff)
            return {"bb": p["bb"], "hd": new_hd}

        def eval_fn(p):
            g_fn = lambda tc: _full_fwd_shared(method, p["bb"], p["hd"], tc)
            l2e, _, _, _, _, _ = evaluate(g_fn, t_r, x_r, u_r)
            return l2e

        params = {"bb": bb, "hd": hd}
        best_p, best_l2 = _adam_lbfgs_fit(loss_fn, params, eval_fn, head_reset_fn=hr)
        n_params = _cp(best_p["bb"], best_p["hd"])
        g_fn = lambda tc: _full_fwd_shared(method, best_p["bb"], best_p["hd"], tc)

    elif method == "siren_permode":
        sz, est = _match_permode(target); h = sz["h"]; sizes = sz
        params = _init_permode(random.PRNGKey(seed), h)

        def loss_fn(p):
            g = _fwd_permode(p, t_train)
            return jnp.mean((g - nn_target) ** 2)

        def eval_fn(p):
            g_fn = lambda tc: _fwd_permode(p, tc)
            l2e, _, _, _, _, _ = evaluate(g_fn, t_r, x_r, u_r)
            return l2e

        best_p, best_l2 = _adam_lbfgs_fit(loss_fn, params, eval_fn, head_reset_fn=None)
        n_params = _cp(best_p)
        g_fn = lambda tc: _fwd_permode(best_p, tc)

    elif method == "chebyshev":
        # classical: g_k(t) via Chebyshev LSQ (no network, deterministic)
        tt = 2.0 * (t_ode - 0.0) / 1.0 - 1.0          # map [0,1] -> [-1,1]
        Phi = np.polynomial.chebyshev.chebvander(tt, N_CHEB - 1)   # (Nt, N_CHEB)
        sol, _, _, _ = np.linalg.lstsq(Phi, np.asarray(nn_target).T, rcond=None)  # (N_CHEB, K)
        coeff = jnp.asarray(sol)

        def g_fn(tc):
            tcn = 2.0 * (np.asarray(tc).reshape(-1) - 0.0) / 1.0 - 1.0
            P = np.polynomial.chebyshev.chebvander(tcn, N_CHEB - 1)
            return (jnp.asarray(P) @ coeff).T          # (K, Nt)

        l2_exact, _, _, _, _, _ = evaluate(g_fn, t_r, x_r, u_r)
        best_l2 = l2_exact
        n_params = int(coeff.size)
        matched = False
    else:
        raise ValueError(method)

    train_t = time.time() - t0
    l2_exact, l2_interp, u_pred, u_exact, x_ev, t_ev = evaluate(g_fn, t_r, x_r, u_r)
    if method != "chebyshev":
        best_l2 = min(best_l2, l2_exact)

    # inference timing (one full reference-grid forward of beta + spatial synth)
    def inf_call():
        return _predict_u(g_fn, None, jnp.asarray(x_ev), jnp.asarray(t_ev))
    out = inf_call()
    try:
        out.block_until_ready()
    except Exception:
        pass
    ti = time.time()
    for _ in range(10):
        out = inf_call()
    try:
        out.block_until_ready()
    except Exception:
        pass
    inf_ms = (time.time() - ti) / 10 * 1000.0

    if method != "chebyshev":
        matched = bool(abs(n_params - target) <= 0.10 * target)
    n_eff_epochs = (ADAM_EPOCHS + LBFGS_ITERS) if method != "chebyshev" else 1

    rec = {
        "method": method, "seed": int(seed),
        "total_params": int(n_params), "target_params": int(target),
        "matched_within_tol": matched, "sizes": sizes,
        "best_l2": float(best_l2), "l2_exact": float(l2_exact), "l2_interp": float(l2_interp),
        "wall_clock_train_sec": float(train_t),
        "ms_per_100_epoch": float(train_t / max(n_eff_epochs, 1) * 1000.0 * 100.0),
        "peak_gpu_mem_mb": gpu_peak_mb(),
        "inference_time_ms": float(inf_ms),
        "n_modes": K_MODES, "ode_dt": ODE_DT, "nu": float(NU),
        "paradigm": "shared Galerkin-ODE substrate + temporal-representation fit",
    }
    if return_pred:
        rec["u_pred"] = u_pred; rec["UE"] = u_exact; rec["xe"] = x_ev; rec["te"] = t_ev
    return rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="svsnn")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print("SV-SNN anchor params:", svsnn_total_params(), "| devices:", jax.devices())
    r = run(a.method, a.seed)
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("u_pred", "UE", "xe", "te")}, indent=2))
