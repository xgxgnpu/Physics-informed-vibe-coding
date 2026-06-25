"""
SV-SNN Burgers v10 — Galerkin ODE + Shared SIREN Backbone + Per-Mode Linear Heads
===================================================================================
PDE:  u_t + u * u_x = nu * u_xx,  nu = 0.01/pi
IC:   u(x,0) = -sin(pi*x),  x in [-1,1]

Strategy:
  1. Solve K-mode Galerkin ODE with RK4 (float64) → beta_k(t) at dense times.
  2. Train a *shared* SIREN backbone  phi(t) in R^D  plus per-mode linear
     heads  g_k(t) = w_k^T phi(t) + b_k  so that
         beta_k(t) = IC_k + t * g_k(t).
     Phase A: Adam warm-up (float64) with periodic head-reset (least squares).
     Phase B: Joint L-BFGS polish (float64) of backbone + all heads.

Key innovation over v9 (independent per-mode SIRENs):
  - Shared nonlinear feature extractor captures common temporal structure.
  - Per-mode heads are cheap linear readouts, kept optimal via head-reset.
  - Joint optimisation over all modes simultaneously → better gradients, faster.
"""

import os, sys, time, json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import optax
import scipy.optimize as sopt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_command_buffer= "
    "--xla_gpu_enable_cublaslt=false"
)
jax.config.update("jax_enable_x64", True)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── PDE parameters ──────────────────────────────────────────────
NU       = 0.01 / np.pi
K_MODES  = 300
NX_PHYS  = 1024
ODE_DT   = 5e-4
SEED     = 42

# ── Architecture ────────────────────────────────────────────────
BACKBONE_H = 256          # hidden width of shared SIREN backbone
BACKBONE_D = 256          # output feature dimension
N_HIDDEN   = 2            # number of hidden SIREN layers
OMEGA_0    = 30.0         # first-layer SIREN frequency

# ── Training ────────────────────────────────────────────────────
ADAM_EPOCHS   = 30_000
ADAM_LR       = 1e-3
HEAD_RESET_EVERY = 5_000  # recompute heads via least-squares
LBFGS_ITERS   = 100_000
LBFGS_MAXCOR  = 100

# ── Evaluation ──────────────────────────────────────────────────
N_EVAL_X = 1024
N_EVAL_T = 201

# ── Globals ─────────────────────────────────────────────────────
k_idx = jnp.arange(1, K_MODES + 1, dtype=jnp.float64)
kpi   = k_idx * jnp.pi
dk    = NU * kpi**2
IC    = jnp.zeros(K_MODES, dtype=jnp.float64).at[0].set(-1.0)


# ================================================================
#  Phase 0 — Galerkin ODE solver (RK4, float64)
# ================================================================
def solve_galerkin_ode():
    print("  Phase 0: Solving Galerkin ODE (RK4, float64)...")
    t0 = time.time()
    K = K_MODES
    kpi_64 = np.arange(1, K + 1, dtype=np.float64) * np.pi
    dk_64  = NU * kpi_64**2
    x_64   = np.linspace(-1.0, 1.0, NX_PHYS + 1, dtype=np.float64)[:-1]
    sin_b  = np.sin(kpi_64[None, :] * x_64[:, None])
    cos_b  = np.cos(kpi_64[None, :] * x_64[:, None])
    dst_p  = (2.0 / NX_PHYS) * sin_b.T

    def rhs(beta):
        u  = sin_b @ beta
        ux = cos_b @ (kpi_64 * beta)
        return -dk_64 * beta - dst_p @ (u * ux)

    beta = np.zeros(K, dtype=np.float64)
    beta[0] = -1.0
    n_steps = int(round(1.0 / ODE_DT))
    dt = 1.0 / n_steps
    t_save = [0.0];  beta_save = [beta.copy()]
    for step in range(1, n_steps + 1):
        k1 = rhs(beta)
        k2 = rhs(beta + 0.5 * dt * k1)
        k3 = rhs(beta + 0.5 * dt * k2)
        k4 = rhs(beta + dt * k3)
        beta = beta + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_save.append(step * dt)
        beta_save.append(beta.copy())
    elapsed = time.time() - t0
    print(f"    {n_steps} steps, saved {len(t_save)} snapshots, {elapsed:.1f}s")
    return np.array(t_save), np.array(beta_save)


# ================================================================
#  Shared SIREN backbone
# ================================================================
def init_backbone(rng, H, D, n_hid, omega_0):
    """Initialise a shared SIREN:  1 -> [H]*n_hid -> D (linear output)."""
    keys = random.split(rng, 2 * n_hid + 2)
    params = {}
    d_in = 1;  ki = 0
    for layer in range(n_hid):
        if layer == 0:
            bound = 1.0 / d_in
        else:
            bound = float(np.sqrt(6.0 / d_in)) / omega_0
        params[f"W{layer}"] = random.uniform(
            keys[ki], (d_in, H), minval=-bound, maxval=bound,
            dtype=jnp.float64)
        params[f"b{layer}"] = random.uniform(
            keys[ki+1], (H,), minval=-float(np.pi), maxval=float(np.pi),
            dtype=jnp.float64)
        ki += 2;  d_in = H
    bound_out = float(np.sqrt(6.0 / d_in)) / omega_0
    params["W_out"] = random.uniform(
        keys[ki], (H, D), minval=-bound_out, maxval=bound_out,
        dtype=jnp.float64)
    params["b_out"] = jnp.zeros(D, dtype=jnp.float64)
    return params


def backbone_fwd(params, t):
    """Evaluate shared SIREN backbone.  t: (Nt, 1) -> phi: (Nt, D)."""
    h = jnp.sin(OMEGA_0 * (t @ params["W0"] + params["b0"]))
    for i in range(1, N_HIDDEN):
        h = jnp.sin(h @ params[f"W{i}"] + params[f"b{i}"])
    phi = h @ params["W_out"] + params["b_out"]
    return phi


# ── Per-mode heads ──────────────────────────────────────────────
def init_heads(K, D):
    """Zero-initialised per-mode linear readout."""
    return {
        "C": jnp.zeros((K, D), dtype=jnp.float64),
        "d": jnp.zeros(K, dtype=jnp.float64),
    }


def head_reset(backbone_params, head_params, t_train, nn_target):
    """Recompute per-mode heads via least-squares given current backbone."""
    phi = np.asarray(backbone_fwd(backbone_params, t_train))     # (Nt, D)
    Phi_aug = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)  # (Nt, D+1)
    sol, _, _, _ = np.linalg.lstsq(Phi_aug, np.asarray(nn_target).T, rcond=None)
    new_C = jnp.array(sol[:BACKBONE_D, :].T)       # (K, D)
    new_d = jnp.array(sol[BACKBONE_D, :])           # (K,)
    return {"C": new_C, "d": new_d}


# ── Full forward pass ───────────────────────────────────────────
def full_fwd(backbone_params, head_params, t):
    """Predict g_k(t) for all K modes.  Returns (K, Nt)."""
    phi = backbone_fwd(backbone_params, t)   # (Nt, D)
    C = head_params["C"]                     # (K, D)
    d = head_params["d"]                     # (K,)
    return C @ phi.T + d[:, None]            # (K, Nt)


def predict_beta(backbone_params, head_params, t):
    """Predict beta_k(t).  Returns (K, Nt)."""
    t_flat = t.squeeze(-1)                   # (Nt,)
    g = full_fwd(backbone_params, head_params, t)  # (K, Nt)
    return IC[:, None] + t_flat[None, :] * g


def predict_u(backbone_params, head_params, x_ev, t_ev):
    """Predict u(x, t) on evaluation grid.  Returns (Nx, Nt)."""
    tc = t_ev.reshape(-1, 1)
    beta = predict_beta(backbone_params, head_params, tc)  # (K, Nt)
    sin_ev = jnp.sin(kpi[None, :] * x_ev[:, None])        # (Nx, K)
    return sin_ev @ beta


def count_params(*trees):
    total = 0
    for tree in trees:
        total += sum(x.size for x in jax.tree.leaves(tree))
    return total


# ================================================================
#  Phase 1 — Initialisation
# ================================================================
def initialise(t_ode, beta_ode):
    print("\n  Phase 1: Initialisation...")
    t0 = time.time()
    rng = random.PRNGKey(SEED)

    ic_np = np.array(IC)
    nn_target = np.zeros((K_MODES, len(t_ode)), dtype=np.float64)
    for j in range(len(t_ode)):
        if t_ode[j] < 1e-15:
            nn_target[:, j] = (beta_ode[1] - ic_np) / t_ode[1]
        else:
            nn_target[:, j] = (beta_ode[j] - ic_np) / t_ode[j]
    nn_target_jnp = jnp.array(nn_target)

    bb_params = init_backbone(rng, BACKBONE_H, BACKBONE_D, N_HIDDEN, OMEGA_0)
    hd_params = init_heads(K_MODES, BACKBONE_D)

    t_train = jnp.array(t_ode, dtype=jnp.float64).reshape(-1, 1)
    hd_params = head_reset(bb_params, hd_params, t_train, nn_target_jnp)

    g_pred = full_fwd(bb_params, hd_params, t_train)
    init_loss = float(jnp.mean((g_pred - nn_target_jnp)**2))
    print(f"    Backbone params: {count_params(bb_params)}")
    print(f"    Head params:     {count_params(hd_params)}")
    print(f"    Total params:    {count_params(bb_params, hd_params)}")
    print(f"    Init MSE loss:   {init_loss:.6e}")
    print(f"    Time: {time.time() - t0:.1f}s")
    return bb_params, hd_params, nn_target_jnp, t_train


# ================================================================
#  Phase 2 — Adam warm-up (float64)
# ================================================================
def adam_train(bb_params, hd_params, nn_target, t_train):
    print(f"\n  Phase 2: Adam warm-up ({ADAM_EPOCHS} epochs, lr={ADAM_LR}, f64)...")
    t0 = time.time()

    all_params = {"bb": bb_params, "hd": hd_params}

    def loss_fn(params):
        g = full_fwd(params["bb"], params["hd"], t_train)
        return jnp.mean((g - nn_target)**2)

    sched = optax.cosine_decay_schedule(ADAM_LR, ADAM_EPOCHS, alpha=1e-6 / ADAM_LR)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adam(learning_rate=sched))
    opt_state = opt.init(all_params)

    @jax.jit
    def step(params, state):
        lv, grads = value_and_grad(loss_fn)(params)
        updates, new_state = opt.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, lv

    for ep in range(1, ADAM_EPOCHS + 1):
        all_params, opt_state, lv = step(all_params, opt_state)

        if ep % HEAD_RESET_EVERY == 0:
            new_hd = head_reset(all_params["bb"], all_params["hd"],
                                t_train, nn_target)
            all_params = {"bb": all_params["bb"], "hd": new_hd}
            opt_state = opt.init(all_params)
            lv_after = loss_fn(all_params)
            print(f"    ep {ep:6d} | loss {float(lv):.6e} → "
                  f"head-reset → {float(lv_after):.6e}")
        elif ep % 5000 == 0 or ep == 1:
            print(f"    ep {ep:6d} | loss {float(lv):.6e}")

    elapsed = time.time() - t0
    print(f"    Adam done: {elapsed:.1f}s, final loss={float(lv):.6e}")
    return all_params["bb"], all_params["hd"]


# ================================================================
#  Phase 3 — LBFGS polish (float64)
# ================================================================
def lbfgs_polish(bb_params, hd_params, nn_target, t_train):
    print(f"\n  Phase 3: LBFGS polish (f64, maxiter={LBFGS_ITERS}, "
          f"maxcor={LBFGS_MAXCOR})...")
    t0 = time.time()

    all_params = {"bb": bb_params, "hd": hd_params}

    def loss_fn(params):
        g = full_fwd(params["bb"], params["hd"], t_train)
        return jnp.mean((g - nn_target)**2)

    flat0, unravel = jax.flatten_util.ravel_pytree(all_params)
    print(f"    Flat parameter vector length: {len(flat0)}")

    @jax.jit
    def loss_and_grad_flat(fp):
        p = unravel(fp)
        lv, g = value_and_grad(loss_fn)(p)
        gf, _ = jax.flatten_util.ravel_pytree(g)
        return lv, gf

    best = {"loss": float("inf"), "flat": None}
    ctr = [0]

    def callback(xk):
        ctr[0] += 1
        if ctr[0] % 5000 == 0 or ctr[0] == 1:
            lv, _ = loss_and_grad_flat(jnp.array(xk, dtype=jnp.float64))
            lv_f = float(lv)
            if lv_f < best["loss"]:
                best["loss"] = lv_f
                best["flat"] = xk.copy()
            print(f"    step {ctr[0]:6d} | loss {lv_f:.6e}")

    def scipy_fg(x):
        lv, g = loss_and_grad_flat(jnp.array(x, dtype=jnp.float64))
        lv_f = float(lv)
        if lv_f < best["loss"]:
            best["loss"] = lv_f
            best["flat"] = x.copy()
        return lv_f, np.array(g, dtype=np.float64)

    res = sopt.minimize(
        scipy_fg, np.array(flat0, dtype=np.float64),
        method="L-BFGS-B", jac=True, callback=callback,
        options={"maxiter": LBFGS_ITERS, "maxfun": LBFGS_ITERS * 2,
                 "ftol": 1e-16, "gtol": 1e-15, "maxcor": LBFGS_MAXCOR})

    elapsed = time.time() - t0
    print(f"    LBFGS done: {res.nit} it, {elapsed:.1f}s — {res.message}")
    print(f"    Final loss: {res.fun:.6e},  Best loss: {best['loss']:.6e}")

    best_flat = best["flat"] if best["flat"] is not None else res.x
    return unravel(jnp.array(best_flat, dtype=jnp.float64))


# ================================================================
#  Evaluation helpers
# ================================================================
def evaluate(bb_params, hd_params, t_r, x_r, u_r):
    """Evaluate L2 errors on interpolated and exact grids."""
    from scipy.interpolate import RegularGridInterpolator

    x_ev = np.linspace(-1, 1, N_EVAL_X + 1, dtype=np.float64)[:-1]
    t_ev = np.linspace(0, 1, N_EVAL_T, dtype=np.float64)

    u_pred = np.asarray(predict_u(bb_params, hd_params,
                                  jnp.array(x_ev), jnp.array(t_ev)))

    Tm, Xm = np.meshgrid(t_ev, x_ev, indexing="ij")
    interp = RegularGridInterpolator(
        (t_r, x_r), u_r, method="cubic",
        bounds_error=False, fill_value=None)
    u_exact = interp((Tm, Xm)).T
    l2_interp = float(np.sqrt(np.sum((u_pred - u_exact)**2) /
                               np.sum(u_exact**2)))

    x_full = x_r[:-1]
    t_full = t_r
    u_pred_full = np.asarray(predict_u(bb_params, hd_params,
                                       jnp.array(x_full), jnp.array(t_full)))
    u_exact_full = u_r[:, :-1].T
    l2_exact = float(np.sqrt(np.sum((u_pred_full - u_exact_full)**2) /
                              np.sum(u_exact_full**2)))

    return l2_interp, l2_exact, u_pred, u_exact, x_ev, t_ev


def per_mode_errors(bb_params, hd_params, t_train, nn_target):
    """Return per-mode RMS fitting error."""
    g_pred = np.asarray(full_fwd(bb_params, hd_params, t_train))
    nn_tgt = np.asarray(nn_target)
    return np.sqrt(np.mean((g_pred - nn_tgt)**2, axis=1))


# ================================================================
#  Figures
# ================================================================
def make_figures(bb_params, hd_params, u_pred, u_exact, x_ev, t_ev,
                 t_r, x_r, u_r, l2_exact, n_params, mode_errs):
    Tm, Xm = np.meshgrid(t_ev, x_ev, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, tl, cm in [
        (axes[0], u_pred, "SV-SNN v10", "jet"),
        (axes[1], u_exact, "ETDRK4 hi-res", "jet"),
        (axes[2], np.abs(u_pred - u_exact),
         f"|err| max={np.max(np.abs(u_pred - u_exact)):.2e}", "hot"),
    ]:
        pc = ax.pcolormesh(Tm, Xm, data.T, shading="gouraud", cmap=cm)
        ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_title(tl)
        fig.colorbar(pc, ax=ax)
    fig.suptitle(f"SV-SNN v10 — L2={l2_exact:.4e}, params={n_params}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "v10_comparison.png"), dpi=150)
    plt.close(fig); print("  Saved v10_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = np.argmin(np.abs(t_ev - tv))
        ax.plot(x_ev, u_pred[:, i], "-",  lw=2, label=f"v10 t={t_ev[i]:.2f}")
        ax.plot(x_ev, u_exact[:, i], "--", lw=1.5, label=f"Ref t={t_ev[i]:.2f}")
    ax.set_xlabel("x"); ax.set_ylabel("u"); ax.set_title("Time Slices")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "v10_slices.png"), dpi=150)
    plt.close(fig); print("  Saved v10_slices.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, K_MODES + 1), mode_errs + 1e-20, "b-", lw=1)
    ax.set_xlabel("Mode k"); ax.set_ylabel("RMS fitting error (g_k)")
    ax.set_title("Per-mode fitting error"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "v10_mode_errors.png"), dpi=150)
    plt.close(fig); print("  Saved v10_mode_errors.png")


# ================================================================
#  Main
# ================================================================
def main():
    print("=" * 65)
    print("  SV-SNN v10 — Shared SIREN Backbone + Per-Mode Linear Heads")
    print(f"  K={K_MODES}, backbone {N_HIDDEN}×{BACKBONE_H}, D={BACKBONE_D}")
    print(f"  ω₀={OMEGA_0}, Device: {jax.devices()}")
    print("=" * 65)

    ref_path = os.path.join(SAVE_DIR, "burgers_reference_hires.npz")
    if not os.path.exists(ref_path):
        print("  ERROR: hi-res reference not found. Run gen_hires_ref.py first.")
        return
    d = np.load(ref_path)
    t_r, x_r, u_r = d["t"], d["x"], d["usol"]
    print(f"  Reference: {u_r.shape}")

    t0_total = time.time()

    # Phase 0 — Galerkin ODE
    t_ode, beta_ode = solve_galerkin_ode()

    # Phase 1 — Initialisation
    bb_params, hd_params, nn_target, t_train = initialise(t_ode, beta_ode)

    # Phase 2 — Adam
    bb_params, hd_params = adam_train(bb_params, hd_params, nn_target, t_train)

    # Phase 3 — LBFGS
    best_all = lbfgs_polish(bb_params, hd_params, nn_target, t_train)
    bb_params, hd_params = best_all["bb"], best_all["hd"]

    total_t = time.time() - t0_total
    n_params = count_params(bb_params, hd_params)

    # Evaluation
    print("\n  Evaluating on reference grid...")
    l2_interp, l2_exact, u_pred, u_exact, x_ev, t_ev = evaluate(
        bb_params, hd_params, t_r, x_r, u_r)
    print(f"    L2 (interpolated): {l2_interp:.6e}")
    print(f"    L2 (exact grid):   {l2_exact:.6e}")

    mode_errs = per_mode_errors(bb_params, hd_params, t_train, nn_target)
    print(f"    Mode errors — max: {mode_errs.max():.4e}, "
          f"mean: {mode_errs.mean():.4e}, median: {np.median(mode_errs):.4e}")

    # Figures
    make_figures(bb_params, hd_params, u_pred, u_exact, x_ev, t_ev,
                 t_r, x_r, u_r, l2_exact, n_params, mode_errs)

    # Summary
    summary = {
        "method": "SV-SNN v10 Shared SIREN backbone + per-mode linear heads",
        "pde": "u_t + u*u_x = nu*u_xx", "nu": float(NU),
        "K_modes": K_MODES,
        "backbone_hidden": BACKBONE_H, "backbone_D": BACKBONE_D,
        "n_hidden": N_HIDDEN, "omega_0": OMEGA_0,
        "total_params": n_params,
        "adam_epochs": ADAM_EPOCHS, "lbfgs_iters": LBFGS_ITERS,
        "total_time_sec": round(total_t, 2),
        "best_l2_interp": l2_interp,
        "best_l2_exact": l2_exact,
        "target_met": l2_exact <= 1e-5,
    }
    with open(os.path.join(SAVE_DIR, "v10_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tag = "ACHIEVED" if l2_exact <= 1e-5 else "NOT YET"
    print(f"\n{'='*65}")
    print(f"  RESULT: L2 (exact) = {l2_exact:.6e}  [{tag}]")
    print(f"  Params: {n_params}, Time: {total_t:.1f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
