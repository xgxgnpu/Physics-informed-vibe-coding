"""
Multi-scale Fourier Feature PINN for 1D Wave Equation (JAX)
============================================================
Reproduces the wave1D case (Section 4.3, Table 2) from:
  Wang, Wang & Perdikaris, "On the eigenvector bias of Fourier feature
  networks", CMAME 384, 113938 (2021).

Six model variants = 3 architectures x 2 weighting strategies:
  Architectures:
    1) Plain  — standard MLP
    2) MFF    — Multi-scale Fourier Feature (sigma=1,10 joint embedding)
    3) ST_MFF — Spatio-Temporal Multi-scale FF (sigma_x=1; sigma_t=1,10)
  Weighting:
    a) Fixed   — equal weights (lambda=1)
    b) Adaptive — NTK-based adaptive weights (Wang et al. 2021)

PDE:  u_tt - 100*u_xx = 0,  (x,t) in (0,1)^2
BC:   u(0,t) = u(1,t) = 0
IC:   u(x,0) = sin(pi*x) + sin(2*pi*x),  u_t(x,0) = 0
Exact: u(x,t) = sin(pi*x)*cos(10*pi*t) + sin(2*pi*x)*cos(20*pi*t)

Self-contained single file.  Activate env:
    source /root/autodl-tmp/pinn_env/bin/activate
"""

import os
import time
import pickle
import datetime

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacrev
from jax.flatten_util import ravel_pytree
import optax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# Global plot style — journal quality
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 2.0
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# ============================================================
# Output directories
# ============================================================
WORKDIR = os.path.dirname(os.path.abspath(__file__))
TIMESTAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join(WORKDIR, f'results_{TIMESTAMP}')
DATA_DIR = os.path.join(RESULTS_DIR, 'data')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
for d in [DATA_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# PDE parameters
# ============================================================
C_SPEED = 10.0  # wave speed: u_tt - c^2 * u_xx = 0, c^2=100

def u_exact(t, x):
    return (np.sin(np.pi * x) * np.cos(10.0 * np.pi * t)
            + np.sin(2.0 * np.pi * x) * np.cos(20.0 * np.pi * t))

def u_t_exact(t, x):
    return (-10.0 * np.pi * np.sin(np.pi * x) * np.sin(10.0 * np.pi * t)
            - 20.0 * np.pi * np.sin(2.0 * np.pi * x) * np.sin(20.0 * np.pi * t))

# ============================================================
# Sampler
# ============================================================
class Sampler:
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N, rng=None):
        if rng is None:
            r = np.random.rand(N, self.dim)
        else:
            r = rng.random((N, self.dim))
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * r
        y = self.func(x)
        return x, y

# ============================================================
# Normalization statistics
# ============================================================
def compute_norm_stats(res_sampler, n_sample=100000):
    X, _ = res_sampler.sample(n_sample)
    mu = X.mean(0)
    sigma = X.std(0)
    return mu, sigma

# ============================================================
# Network initialization (Xavier)
# ============================================================
def xavier_init(key, fan_in, fan_out):
    std = 1.0 / np.sqrt((fan_in + fan_out) / 2.0)
    k1, k2 = random.split(key)
    W = std * random.normal(k1, (fan_in, fan_out))
    b = random.normal(k2, (1, fan_out))
    return W, b

def init_mlp(layers, key):
    params = []
    for i in range(len(layers) - 1):
        key, subkey = random.split(key)
        W, b = xavier_init(subkey, layers[i], layers[i + 1])
        params.append((W, b))
    return params

# ============================================================
# Model 1: Plain MLP (Plain)
# ============================================================
def apply_plain(params, tx):
    """tx: (2,) array [t_norm, x_norm] -> scalar u."""
    H = tx.reshape(1, -1)
    for (W, b) in params[:-1]:
        H = jnp.tanh(H @ W + b)
    W, b = params[-1]
    return (H @ W + b)[0, 0]

# ============================================================
# Model 2: Multi-scale Fourier Feature (MFF)
# Joint embedding of (t,x) with sigma=1 and sigma=10, concatenation merge.
# ============================================================
def apply_mff(params, W_ff1, W_ff2, tx):
    """tx: (2,) -> scalar u.  W_ff1(sigma=1), W_ff2(sigma=10) frozen."""
    H_in = tx.reshape(1, -1)
    H1 = jnp.concatenate([jnp.sin(H_in @ W_ff1), jnp.cos(H_in @ W_ff1)], axis=1)
    H2 = jnp.concatenate([jnp.sin(H_in @ W_ff2), jnp.cos(H_in @ W_ff2)], axis=1)

    for (W, b) in params[:-1]:
        H1 = jnp.tanh(H1 @ W + b)
        H2 = jnp.tanh(H2 @ W + b)

    H = jnp.concatenate([H1, H2], axis=1)
    W_last, b_last = params[-1]
    return (H @ W_last + b_last)[0, 0]

# ============================================================
# Model 3: Spatio-Temporal Multi-scale FF (ST_MFF)
# Spatial: sigma_x=1; Temporal: sigma_t1=1, sigma_t2=10
# Point-wise multiplication merge (Eq. 3.33-3.34 in paper).
# ============================================================
def apply_st_mff(params, W_x, W_t1, W_t2, tx):
    """tx: (2,) -> scalar u.  W_x, W_t1, W_t2 frozen."""
    t_val = tx[0:1].reshape(1, 1)
    x_val = tx[1:2].reshape(1, 1)

    H_x = jnp.concatenate([jnp.sin(x_val @ W_x), jnp.cos(x_val @ W_x)], axis=1)
    H_t1 = jnp.concatenate([jnp.sin(t_val @ W_t1), jnp.cos(t_val @ W_t1)], axis=1)
    H_t2 = jnp.concatenate([jnp.sin(t_val @ W_t2), jnp.cos(t_val @ W_t2)], axis=1)

    for (W, b) in params[:-1]:
        H_x = jnp.tanh(H_x @ W + b)
        H_t1 = jnp.tanh(H_t1 @ W + b)
        H_t2 = jnp.tanh(H_t2 @ W + b)

    # Point-wise multiplication then linear combination
    H = H_x * H_t1 + H_x * H_t2
    W_last, b_last = params[-1]
    return (H @ W_last + b_last)[0, 0]

# ============================================================
# PDE residual: u_tt - 100*u_xx  (wave equation)
# ============================================================
def make_residual_fn(apply_fn, c_sq, sigma_t, sigma_x):
    def residual_single(params, *frozen_and_tx):
        *frozen, t_norm, x_norm = frozen_and_tx

        def u_of_tx(t_n, x_n):
            return apply_fn(params, *frozen, jnp.array([t_n, x_n]))

        # u_tt in physical space
        du_dt = grad(u_of_tx, argnums=0)
        d2u_dt2_norm = grad(du_dt, argnums=0)(t_norm, x_norm)
        u_tt = d2u_dt2_norm / (sigma_t ** 2)

        # u_xx in physical space
        du_dx = grad(u_of_tx, argnums=1)
        d2u_dx2_norm = grad(du_dx, argnums=1)(t_norm, x_norm)
        u_xx = d2u_dx2_norm / (sigma_x ** 2)

        return u_tt - c_sq * u_xx

    return residual_single

# ============================================================
# u_t for initial velocity condition
# ============================================================
def make_u_t_fn(apply_fn, sigma_t):
    def u_t_single(params, *frozen_and_tx):
        *frozen, t_norm, x_norm = frozen_and_tx

        def u_of_tx(t_n, x_n):
            return apply_fn(params, *frozen, jnp.array([t_n, x_n]))

        du_dt_norm = grad(u_of_tx, argnums=0)(t_norm, x_norm)
        return du_dt_norm / sigma_t

    return u_t_single

# ============================================================
# Loss functions (3 components: L_u, L_ut, L_r)
# ============================================================
def make_loss_fns(apply_fn, residual_fn, u_t_fn, n_frozen):

    def u_pred_single(params, *frozen_and_tx):
        *frozen, t_n, x_n = frozen_and_tx
        return apply_fn(params, *frozen, jnp.array([t_n, x_n]))

    u_pred_batch = vmap(u_pred_single, in_axes=(None,) + (None,) * n_frozen + (0, 0))
    res_batch = vmap(residual_fn, in_axes=(None,) + (None,) * n_frozen + (0, 0))
    u_t_batch = vmap(u_t_fn, in_axes=(None,) + (None,) * n_frozen + (0, 0))

    def loss_u(params, *frozen, t_ic, x_ic, u_ic, t_bc1, x_bc1, t_bc2, x_bc2):
        u_ic_pred = u_pred_batch(params, *frozen, t_ic, x_ic)
        u_bc1_pred = u_pred_batch(params, *frozen, t_bc1, x_bc1)
        u_bc2_pred = u_pred_batch(params, *frozen, t_bc2, x_bc2)
        return (jnp.mean((u_ic_pred - u_ic) ** 2)
                + jnp.mean(u_bc1_pred ** 2)
                + jnp.mean(u_bc2_pred ** 2))

    def loss_ut(params, *frozen, t_ic, x_ic):
        ut_pred = u_t_batch(params, *frozen, t_ic, x_ic)
        return jnp.mean(ut_pred ** 2)

    def loss_r(params, *frozen, t_r, x_r):
        r = res_batch(params, *frozen, t_r, x_r)
        return jnp.mean(r ** 2)

    def loss_total(params, *frozen, t_ic, x_ic, u_ic, t_bc1, x_bc1, t_bc2, x_bc2,
                   t_r, x_r, lam_u, lam_ut, lam_r):
        l_u = loss_u(params, *frozen, t_ic=t_ic, x_ic=x_ic, u_ic=u_ic,
                     t_bc1=t_bc1, x_bc1=x_bc1, t_bc2=t_bc2, x_bc2=x_bc2)
        l_ut = loss_ut(params, *frozen, t_ic=t_ic, x_ic=x_ic)
        l_r = loss_r(params, *frozen, t_r=t_r, x_r=x_r)
        return lam_u * l_u + lam_ut * l_ut + lam_r * l_r, (l_u, l_ut, l_r)

    return loss_total, u_pred_batch, res_batch, u_t_batch

# ============================================================
# NTK computation for adaptive weights
# ============================================================
def compute_jacobian(fn_batch, params, frozen_args, t_pts, x_pts):
    flat_params, unravel = ravel_pytree(params)
    def f_flat(fp):
        p = unravel(fp)
        return fn_batch(p, *frozen_args, t_pts, x_pts)
    return jacrev(f_flat)(flat_params)


def compute_ntk_blocks(u_pred_batch, u_t_batch, res_batch,
                       params, frozen_args,
                       t_bc, x_bc, t_ic, x_ic, t_r, x_r):
    """Compute diagonal NTK blocks: K_u, K_ut, K_r."""
    J_u = compute_jacobian(u_pred_batch, params, frozen_args, t_bc, x_bc)
    J_ut = compute_jacobian(u_t_batch, params, frozen_args, t_ic, x_ic)
    J_r = compute_jacobian(res_batch, params, frozen_args, t_r, x_r)
    K_u = J_u @ J_u.T
    K_ut = J_ut @ J_ut.T
    K_r = J_r @ J_r.T
    return K_u, K_ut, K_r

# ============================================================
# Training one model
# ============================================================
def train_model(model_name, apply_fn, params, frozen_args, n_frozen,
                mu_X, sigma_X, ics_sampler, bcs_sampler, res_sampler,
                ut_sampler,
                use_adaptive_weights=False,
                n_iter=40000, batch_size=360, log_every=100,
                ntk_every=100, ntk_n_pts=128, seed=42):

    tag = f"{model_name}_adapt" if use_adaptive_weights else f"{model_name}_fixed"
    print(f"\n{'='*70}")
    print(f"  Training: {tag}")
    print(f"  Adaptive weights: {'ON' if use_adaptive_weights else 'OFF'}")
    print(f"{'='*70}")

    sigma_t = float(sigma_X[0])
    sigma_x = float(sigma_X[1])
    c_sq = C_SPEED ** 2

    residual_fn = make_residual_fn(apply_fn, c_sq, sigma_t, sigma_x)
    u_t_fn = make_u_t_fn(apply_fn, sigma_t)
    loss_total_fn, u_pred_batch_fn, res_batch_fn, u_t_batch_fn = make_loss_fns(
        apply_fn, residual_fn, u_t_fn, n_frozen
    )

    flat0, _ = ravel_pytree(params)
    n_params = flat0.shape[0]
    print(f"  Trainable parameters: {n_params}")

    lr_schedule = optax.exponential_decay(
        init_value=1e-3, transition_steps=1000,
        decay_rate=0.9, staircase=False)
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    lam_u = 1.0
    lam_ut = 1.0
    lam_r = 1.0

    @jit
    def train_step(params, opt_state,
                   t_ic, x_ic, u_ic,
                   t_bc1, x_bc1, t_bc2, x_bc2,
                   t_r, x_r, lam_u_v, lam_ut_v, lam_r_v):
        (loss_val, (l_u, l_ut, l_r)), grads = jax.value_and_grad(
            lambda p: loss_total_fn(p, *frozen_args,
                                    t_ic=t_ic, x_ic=x_ic, u_ic=u_ic,
                                    t_bc1=t_bc1, x_bc1=x_bc1,
                                    t_bc2=t_bc2, x_bc2=x_bc2,
                                    t_r=t_r, x_r=x_r,
                                    lam_u=lam_u_v, lam_ut=lam_ut_v, lam_r=lam_r_v),
            has_aux=True
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, l_u, l_ut, l_r

    def predict_u(params, X_pts):
        X_norm = (X_pts - mu_X) / sigma_X
        t_n = jnp.array(X_norm[:, 0])
        x_n = jnp.array(X_norm[:, 1])
        return np.array(u_pred_batch_fn(params, *frozen_args, t_n, x_n))

    # Test grid
    nn_test = 200
    t_test = np.linspace(0, 1, nn_test)
    x_test = np.linspace(0, 1, nn_test)
    tt, xx = np.meshgrid(t_test, x_test)
    X_test = np.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
    u_test = u_exact(X_test[:, 0], X_test[:, 1])

    # Logs
    loss_u_log, loss_ut_log, loss_res_log, l2_error_log = [], [], [], []
    iters_log = []
    lam_u_log, lam_ut_log, lam_r_log = [], [], []
    ntk_K_log, ntk_iters_log = [], []
    best_l2 = 1e10

    rng_train = np.random.RandomState(seed)
    start_time = time.time()

    for it in range(n_iter):
        X_ic, u_ic_batch = ics_sampler.sample(batch_size, rng_train)
        X_bc1, _ = bcs_sampler[0].sample(batch_size, rng_train)
        X_bc2, _ = bcs_sampler[1].sample(batch_size, rng_train)
        X_res, _ = res_sampler.sample(batch_size, rng_train)

        X_ic_n = (X_ic - mu_X) / sigma_X
        X_bc1_n = (X_bc1 - mu_X) / sigma_X
        X_bc2_n = (X_bc2 - mu_X) / sigma_X
        X_res_n = (X_res - mu_X) / sigma_X

        params, opt_state, loss_val, l_u, l_ut, l_r = train_step(
            params, opt_state,
            jnp.array(X_ic_n[:, 0]), jnp.array(X_ic_n[:, 1]),
            jnp.array(u_ic_batch.flatten()),
            jnp.array(X_bc1_n[:, 0]), jnp.array(X_bc1_n[:, 1]),
            jnp.array(X_bc2_n[:, 0]), jnp.array(X_bc2_n[:, 1]),
            jnp.array(X_res_n[:, 0]), jnp.array(X_res_n[:, 1]),
            jnp.float32(lam_u), jnp.float32(lam_ut), jnp.float32(lam_r),
        )

        if it % log_every == 0:
            u_pred = predict_u(params, X_test)
            l2_err = float(np.linalg.norm(u_test - u_pred) / np.linalg.norm(u_test))
            loss_u_log.append(float(l_u))
            loss_ut_log.append(float(l_ut))
            loss_res_log.append(float(l_r))
            l2_error_log.append(l2_err)
            iters_log.append(it)
            lam_u_log.append(lam_u)
            lam_ut_log.append(lam_ut)
            lam_r_log.append(lam_r)
            if l2_err < best_l2:
                best_l2 = l2_err

            elapsed = time.time() - start_time
            print(f"  It: {it:5d}, Loss: {float(loss_val):.3e}, "
                  f"L_u: {float(l_u):.3e}, L_ut: {float(l_ut):.3e}, "
                  f"L_r: {float(l_r):.3e}, L2: {l2_err:.3e}, Time: {elapsed:.1f}s")
            if use_adaptive_weights:
                print(f"    lam_u={lam_u:.3e}, lam_ut={lam_ut:.3e}, lam_r={lam_r:.3e}")

        # NTK adaptive weight update
        if use_adaptive_weights and it % ntk_every == 0:
            n_k = min(ntk_n_pts, batch_size)
            t_bc_ntk = jnp.concatenate([
                jnp.array(X_ic_n[:n_k, 0]),
                jnp.array(X_bc1_n[:n_k, 0]),
                jnp.array(X_bc2_n[:n_k, 0])
            ])
            x_bc_ntk = jnp.concatenate([
                jnp.array(X_ic_n[:n_k, 1]),
                jnp.array(X_bc1_n[:n_k, 1]),
                jnp.array(X_bc2_n[:n_k, 1])
            ])
            t_ic_ntk = jnp.array(X_ic_n[:n_k, 0])
            x_ic_ntk = jnp.array(X_ic_n[:n_k, 1])
            t_r_ntk = jnp.array(X_res_n[:n_k, 0])
            x_r_ntk = jnp.array(X_res_n[:n_k, 1])

            K_u_val, K_ut_val, K_r_val = compute_ntk_blocks(
                u_pred_batch_fn, u_t_batch_fn, res_batch_fn,
                params, frozen_args,
                t_bc_ntk, x_bc_ntk, t_ic_ntk, x_ic_ntk, t_r_ntk, x_r_ntk
            )

            tr_u = float(jnp.trace(K_u_val))
            tr_ut = float(jnp.trace(K_ut_val))
            tr_r = float(jnp.trace(K_r_val))
            tr_total = tr_u + tr_ut + tr_r

            if tr_u > 0 and tr_ut > 0 and tr_r > 0:
                lam_u = tr_total / tr_u
                lam_ut = tr_total / tr_ut
                lam_r = tr_total / tr_r

            ntk_K_log.append({
                'K_u': np.array(K_u_val), 'K_ut': np.array(K_ut_val),
                'K_r': np.array(K_r_val),
            })
            ntk_iters_log.append(it)

        # For non-adaptive models, compute NTK less frequently for analysis
        if not use_adaptive_weights and it % 5000 == 0:
            n_k = min(64, batch_size)
            t_bc_ntk = jnp.concatenate([
                jnp.array(X_ic_n[:n_k, 0]),
                jnp.array(X_bc1_n[:n_k, 0]),
                jnp.array(X_bc2_n[:n_k, 0])
            ])
            x_bc_ntk = jnp.concatenate([
                jnp.array(X_ic_n[:n_k, 1]),
                jnp.array(X_bc1_n[:n_k, 1]),
                jnp.array(X_bc2_n[:n_k, 1])
            ])
            t_ic_ntk = jnp.array(X_ic_n[:n_k, 0])
            x_ic_ntk = jnp.array(X_ic_n[:n_k, 1])
            t_r_ntk = jnp.array(X_res_n[:n_k, 0])
            x_r_ntk = jnp.array(X_res_n[:n_k, 1])

            K_u_val, K_ut_val, K_r_val = compute_ntk_blocks(
                u_pred_batch_fn, u_t_batch_fn, res_batch_fn,
                params, frozen_args,
                t_bc_ntk, x_bc_ntk, t_ic_ntk, x_ic_ntk, t_r_ntk, x_r_ntk
            )
            ntk_K_log.append({
                'K_u': np.array(K_u_val), 'K_ut': np.array(K_ut_val),
                'K_r': np.array(K_r_val),
            })
            ntk_iters_log.append(it)
            print(f"  [NTK computed at iter {it}]")

    total_time = time.time() - start_time

    u_pred_final = predict_u(params, X_test)
    final_l2 = float(np.linalg.norm(u_test - u_pred_final) / np.linalg.norm(u_test))

    print(f"\n  {tag} training complete.")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Best L2 error:    {best_l2:.3e}")
    print(f"  Final L2 error:   {final_l2:.3e}")
    print(f"  Parameters:       {n_params}")

    return {
        'model_name': tag,
        'arch_name': model_name,
        'adaptive': use_adaptive_weights,
        'n_params': n_params,
        'total_time': total_time,
        'best_l2': best_l2,
        'final_l2': final_l2,
        'loss_u_log': loss_u_log,
        'loss_ut_log': loss_ut_log,
        'loss_res_log': loss_res_log,
        'l2_error_log': l2_error_log,
        'iters_log': iters_log,
        'lam_u_log': lam_u_log,
        'lam_ut_log': lam_ut_log,
        'lam_r_log': lam_r_log,
        'ntk_K_log': ntk_K_log,
        'ntk_iters_log': ntk_iters_log,
        'u_pred_final': u_pred_final,
        'u_test': u_test,
        'X_test': X_test,
        'tt': tt, 'xx': xx,
        'nn_test': nn_test,
        'params': params,
    }

# ============================================================
# Save data for one model
# ============================================================
def save_model_data(results):
    name = results['model_name']
    iters = np.array(results['iters_log'])

    loss_data = np.column_stack([
        iters, results['loss_u_log'], results['loss_ut_log'],
        results['loss_res_log'], results['l2_error_log'],
        results['lam_u_log'], results['lam_ut_log'], results['lam_r_log'],
    ])
    np.savetxt(os.path.join(DATA_DIR, f'loss_history_{name}.txt'), loss_data,
               header='iteration  loss_u  loss_ut  loss_res  l2_error  lam_u  lam_ut  lam_r',
               fmt='%.6e')

    X = results['X_test']
    pred_data = np.column_stack([
        X[:, 0], X[:, 1], results['u_test'], results['u_pred_final'],
        np.abs(results['u_test'] - results['u_pred_final']),
    ])
    np.savetxt(os.path.join(DATA_DIR, f'prediction_{name}.txt'), pred_data,
               header='t  x  u_exact  u_pred  abs_error', fmt='%.6e')

    if results['adaptive']:
        lam_data = np.column_stack([
            iters, results['lam_u_log'], results['lam_ut_log'], results['lam_r_log'],
        ])
        np.savetxt(os.path.join(DATA_DIR, f'lambda_history_{name}.txt'), lam_data,
                   header='iteration  lambda_u  lambda_ut  lambda_r', fmt='%.6e')

    # NTK eigenvalues
    ntk_iters = results['ntk_iters_log']
    ntk_logs = results['ntk_K_log']
    eig_Ku_all, eig_Kut_all, eig_Kr_all = [], [], []

    for i, nt in enumerate(ntk_logs):
        eig_u = np.sort(np.real(np.linalg.eigvalsh(nt['K_u'])))[::-1]
        eig_ut = np.sort(np.real(np.linalg.eigvalsh(nt['K_ut'])))[::-1]
        eig_r = np.sort(np.real(np.linalg.eigvalsh(nt['K_r'])))[::-1]
        eig_Ku_all.append(eig_u)
        eig_Kut_all.append(eig_ut)
        eig_Kr_all.append(eig_r)

        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Ku_{name}_iter{ntk_iters[i]}.txt'),
                   eig_u, fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Kut_{name}_iter{ntk_iters[i]}.txt'),
                   eig_ut, fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Kr_{name}_iter{ntk_iters[i]}.txt'),
                   eig_r, fmt='%.6e')

    with open(os.path.join(CKPT_DIR, f'params_{name}.pkl'), 'wb') as f:
        pickle.dump(results['params'], f)

    results['eig_Ku_all'] = eig_Ku_all
    results['eig_Kut_all'] = eig_Kut_all
    results['eig_Kr_all'] = eig_Kr_all

# ============================================================
# Plotting helpers
# ============================================================
def _label(ax, label, x=-0.10, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')

def _style(ax, xlabel=None, ylabel=None, fs=14):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs, fontweight='bold')
    ax.tick_params(labelsize=12, width=1.5, length=4)
    for s in ax.spines.values():
        s.set_linewidth(2.0)

# ============================================================
# Plot: single model — prediction (Exact / Predicted / Error)
# ============================================================
def plot_prediction_single(results):
    name = results['model_name']
    tt, xx = results['tt'], results['xx']
    nn = results['nn_test']

    u_star = results['u_test'].reshape(nn, nn)
    u_pred = results['u_pred_final'].reshape(nn, nn)
    err = np.abs(u_star - u_pred)
    vmin_u, vmax_u = u_star.min(), u_star.max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    im0 = axes[0].pcolormesh(tt, xx, u_star, cmap='jet', shading='auto',
                              vmin=vmin_u, vmax=vmax_u)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title('Exact $u(t,x)$', fontsize=16, fontweight='bold')
    _style(axes[0], '$t$', '$x$')
    _label(axes[0], '(a)')

    im1 = axes[1].pcolormesh(tt, xx, u_pred, cmap='jet', shading='auto',
                              vmin=vmin_u, vmax=vmax_u)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    l2_str = f'{results["best_l2"]:.2e}'
    axes[1].set_title(f'Predicted ({name}, $L^2$={l2_str})', fontsize=14, fontweight='bold')
    _style(axes[1], '$t$', '$x$')
    _label(axes[1], '(b)')

    im2 = axes[2].pcolormesh(tt, xx, err, cmap='jet', shading='auto')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title('Absolute Error', fontsize=16, fontweight='bold')
    _style(axes[2], '$t$', '$x$')
    _label(axes[2], '(c)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Plot: single model — loss components
# ============================================================
def plot_loss_single(results):
    name = results['model_name']
    iters = np.array(results['iters_log'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(iters, results['loss_res_log'], lw=2, label=r'$\mathcal{L}_{r}$')
    ax.semilogy(iters, results['loss_u_log'], lw=2, label=r'$\mathcal{L}_{u}$')
    ax.semilogy(iters, results['loss_ut_log'], lw=2, label=r'$\mathcal{L}_{u_t}$')
    ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black')
    _style(ax, 'Iterations', 'Loss')
    ax.set_title(f'{name} — Loss Components', fontsize=14, fontweight='bold')
    _label(ax, '(a)')

    ax = axes[1]
    ax.semilogy(iters, results['l2_error_log'], lw=2, color='tab:red')
    _style(ax, 'Iterations', 'Relative $L^2$ Error')
    ax.set_title(f'{name} — $L^2$ Error', fontsize=14, fontweight='bold')
    _label(ax, '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Plot: adaptive weights history
# ============================================================
def plot_adaptive_weights(results):
    if not results['adaptive']:
        return
    name = results['model_name']
    iters = np.array(results['iters_log'])
    n = len(results['lam_u_log'])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(iters[:n], results['lam_u_log'][:n], lw=2, label=r'$\lambda_u$')
    ax.semilogy(iters[:n], results['lam_ut_log'][:n], lw=2, label=r'$\lambda_{u_t}$')
    ax.semilogy(iters[:n], results['lam_r_log'][:n], lw=2, label=r'$\lambda_r$')
    ax.legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    _style(ax, 'Iterations', r'$\lambda$')
    ax.set_title(f'{name} — Adaptive Weights', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_adaptive_weights.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison: Table 2 heatmap
# ============================================================
ARCH_ORDER = ['Plain', 'MFF', 'ST_MFF']
WEIGHT_ORDER = ['fixed', 'adapt']

def plot_table2(all_results):
    """Reproduce Table 2: 3 architectures x 2 weighting strategies."""
    table = np.ones((2, 3)) * np.nan
    labels = [['' for _ in range(3)] for _ in range(2)]

    for res in all_results:
        arch = res['arch_name']
        w_idx = 1 if res['adaptive'] else 0
        a_idx = ARCH_ORDER.index(arch)
        table[w_idx, a_idx] = res['best_l2']
        labels[w_idx][a_idx] = f"{res['best_l2']:.2e}"

    fig, ax = plt.subplots(figsize=(10, 4))
    log_table = np.log10(np.clip(table, 1e-10, None))
    im = ax.imshow(log_table, cmap='RdYlGn_r', aspect='auto', vmin=-4, vmax=0.5)
    fig.colorbar(im, ax=ax, label=r'$\log_{10}$ (Relative $L^2$ Error)', fraction=0.046, pad=0.04)

    ax.set_xticks(range(3))
    ax.set_xticklabels(ARCH_ORDER, fontsize=14, fontweight='bold')
    ax.set_yticks(range(2))
    ax.set_yticklabels(['No adaptive\nweights', 'With adaptive\nweights'],
                       fontsize=13, fontweight='bold')

    for i in range(2):
        for j in range(3):
            if labels[i][j]:
                color = 'white' if log_table[i, j] > -1.5 else 'black'
                ax.text(j, i, labels[i][j], ha='center', va='center',
                        fontsize=14, fontweight='bold', color=color)

    ax.set_title('1D Wave Equation — Table 2 (Relative $L^2$ Error)',
                 fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_table2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison: L2 convergence
# ============================================================
MODEL_COLORS = {
    'Plain_fixed': 'tab:blue', 'Plain_adapt': 'tab:blue',
    'MFF_fixed': 'tab:orange', 'MFF_adapt': 'tab:orange',
    'ST_MFF_fixed': 'tab:green', 'ST_MFF_adapt': 'tab:green',
}
MODEL_LS = {
    'Plain_fixed': '-', 'Plain_adapt': '--',
    'MFF_fixed': '-', 'MFF_adapt': '--',
    'ST_MFF_fixed': '-', 'ST_MFF_adapt': '--',
}

def plot_comparison_l2(all_results):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: without adaptive
    ax = axes[0]
    for res in all_results:
        if not res['adaptive']:
            n = res['model_name']
            ax.semilogy(res['iters_log'], res['l2_error_log'], lw=2.5,
                        color=MODEL_COLORS[n], label=f"{res['arch_name']} (L2={res['best_l2']:.2e})")
    ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor='black')
    _style(ax, 'Iterations', 'Relative $L^2$ Error', fs=15)
    ax.set_title('Without Adaptive Weights', fontsize=16, fontweight='bold')
    _label(ax, '(a)')

    # Right: with adaptive
    ax = axes[1]
    for res in all_results:
        if res['adaptive']:
            n = res['model_name']
            ax.semilogy(res['iters_log'], res['l2_error_log'], lw=2.5,
                        color=MODEL_COLORS[n], label=f"{res['arch_name']} (L2={res['best_l2']:.2e})")
    ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor='black')
    _style(ax, 'Iterations', 'Relative $L^2$ Error', fs=15)
    ax.set_title('With Adaptive Weights', fontsize=16, fontweight='bold')
    _label(ax, '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_l2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison: prediction — best vs worst
# ============================================================
def plot_comparison_prediction(all_results):
    sorted_res = sorted(all_results, key=lambda r: r['best_l2'])
    best = sorted_res[0]
    worst = sorted_res[-1]

    nn = best['nn_test']
    tt, xx = best['tt'], best['xx']
    u_star = best['u_test'].reshape(nn, nn)
    vmin_u, vmax_u = u_star.min(), u_star.max()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for row, res in enumerate([best, worst]):
        u_pred = res['u_pred_final'].reshape(nn, nn)
        err = np.abs(u_star - u_pred)

        im0 = axes[row, 0].pcolormesh(tt, xx, u_star, cmap='jet', shading='auto',
                                        vmin=vmin_u, vmax=vmax_u)
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        _style(axes[row, 0], '$t$', '$x$')
        axes[row, 0].set_ylabel(f"{res['model_name']}\n$x$", fontsize=13, fontweight='bold')
        if row == 0:
            axes[row, 0].set_title('Exact', fontsize=16, fontweight='bold')

        im1 = axes[row, 1].pcolormesh(tt, xx, u_pred, cmap='jet', shading='auto',
                                        vmin=vmin_u, vmax=vmax_u)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        _style(axes[row, 1], '$t$')
        if row == 0:
            axes[row, 1].set_title('Predicted', fontsize=16, fontweight='bold')

        im2 = axes[row, 2].pcolormesh(tt, xx, err, cmap='jet', shading='auto')
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        _style(axes[row, 2], '$t$')
        if row == 0:
            axes[row, 2].set_title('Absolute Error', fontsize=16, fontweight='bold')

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for idx, ax in enumerate(axes.flat):
        _label(ax, labels[idx])

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison: loss convergence grouped
# ============================================================
def plot_comparison_loss(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    titles = [r'$\mathcal{L}_{u}$ (BC+IC)', r'$\mathcal{L}_{u_t}$ (Init. velocity)',
              r'$\mathcal{L}_{r}$ (PDE residual)']
    keys = ['loss_u_log', 'loss_ut_log', 'loss_res_log']

    for col, (ax, title, key) in enumerate(zip(axes, titles, keys)):
        for res in all_results:
            n = res['model_name']
            ls = '--' if res['adaptive'] else '-'
            ax.semilogy(res['iters_log'], res[key], lw=2,
                        color=MODEL_COLORS[n], linestyle=ls,
                        label=n, alpha=0.8)
        ax.set_title(title, fontsize=15, fontweight='bold')
        _style(ax, 'Iterations', 'Loss')
        _label(ax, f'({"abc"[col]})')

    handles, labels_l = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_l, loc='upper center', ncol=3,
               fontsize=11, frameon=True, fancybox=False, edgecolor='black',
               bbox_to_anchor=(0.5, 1.04))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison: adaptive weights evolution (adapt models only)
# ============================================================
def plot_comparison_adaptive_weights(all_results):
    adapt_results = [r for r in all_results if r['adaptive']]
    if not adapt_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    titles = [r'$\lambda_u$', r'$\lambda_{u_t}$', r'$\lambda_r$']
    keys = ['lam_u_log', 'lam_ut_log', 'lam_r_log']

    for col, (ax, title, key) in enumerate(zip(axes, titles, keys)):
        for res in adapt_results:
            n = res['model_name']
            iters = np.array(res['iters_log'])
            vals = res[key]
            n_pts = min(len(iters), len(vals))
            ax.semilogy(iters[:n_pts], vals[:n_pts], lw=2.5,
                        color=MODEL_COLORS[n], label=res['arch_name'])
        ax.set_title(title, fontsize=16, fontweight='bold')
        _style(ax, 'Iterations', r'$\lambda$')
        ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor='black')
        _label(ax, f'({"abc"[col]})')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_adaptive_weights.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison: NTK eigenvalues
# ============================================================
def plot_comparison_ntk(all_results):
    adapt_results = [r for r in all_results if r['adaptive'] and len(r.get('eig_Kr_all', [])) > 0]
    if not adapt_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, (ax, title, idx) in enumerate(zip(
            axes, ['$K_r$ — Initial ($n=0$)', '$K_r$ — Final'], [0, -1])):
        for res in adapt_results:
            eig = res.get('eig_Kr_all', [])
            if not eig:
                continue
            it_n = res['ntk_iters_log'][idx]
            ax.loglog(np.arange(1, len(eig[idx]) + 1), np.clip(eig[idx], 1e-30, None),
                      lw=2.5, color=MODEL_COLORS[res['model_name']],
                      label=f"{res['arch_name']} (iter {it_n})")
        ax.set_title(title, fontsize=16, fontweight='bold')
        _style(ax, 'Index', 'Eigenvalue')
        ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor='black')
        _label(ax, f'({"ab"[col]})')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_ntk.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison: bar chart
# ============================================================
def plot_comparison_bar(all_results):
    names = [r['model_name'] for r in all_results]
    best_l2 = [r['best_l2'] for r in all_results]
    times = [r['total_time'] for r in all_results]
    x = np.arange(len(names))
    w = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    colors = [MODEL_COLORS[n] for n in names]
    hatch = ['///' if r['adaptive'] else '' for r in all_results]

    ax = axes[0]
    bars = ax.bar(x, best_l2, w, color=colors, edgecolor='black', linewidth=1.5)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight='bold', rotation=30, ha='right')
    _style(ax, ylabel='Best Relative $L^2$ Error')
    ax.set_title('$L^2$ Error Comparison', fontsize=16, fontweight='bold')
    for b, val in zip(bars, best_l2):
        ax.text(b.get_x() + b.get_width() / 2, val * 1.5, f'{val:.1e}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    _label(ax, '(a)')

    ax = axes[1]
    bars = ax.bar(x, times, w, color=colors, edgecolor='black', linewidth=1.5)
    for b, h in zip(bars, hatch):
        b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight='bold', rotation=30, ha='right')
    _style(ax, ylabel='Training Time (s)')
    ax.set_title('Training Time', fontsize=16, fontweight='bold')
    for b, val in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, val + max(times) * 0.02,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    _label(ax, '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_bar.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Save comparison table
# ============================================================
def save_comparison_table(all_results):
    lines = []
    lines.append("=" * 90)
    lines.append("TABLE 2 — 1D Wave Equation (Multi-scale PINN)")
    lines.append("Reproducing Wang, Wang & Perdikaris, CMAME 384 (2021), Section 4.3")
    lines.append("=" * 90)
    lines.append(f"{'Model':<18} {'Adaptive':>10} {'Params':>10} "
                 f"{'Best L2':>14} {'Final L2':>14} {'Time (s)':>12}")
    lines.append("-" * 90)
    for res in all_results:
        adaptive_str = 'YES' if res['adaptive'] else 'NO'
        lines.append(f"{res['model_name']:<18} {adaptive_str:>10} {res['n_params']:>10d} "
                     f"{res['best_l2']:>14.4e} {res['final_l2']:>14.4e} "
                     f"{res['total_time']:>12.1f}")
    lines.append("-" * 90)

    # Table 2 format
    lines.append("")
    lines.append("Table 2 format (Architecture x Weighting):")
    lines.append(f"{'':>20} {'Plain':>14} {'MFF':>14} {'ST-MFF':>14}")
    lines.append("-" * 62)

    for wt_label, is_adapt in [('No adaptive', False), ('With adaptive', True)]:
        row = f"{wt_label:>20}"
        for arch in ARCH_ORDER:
            matched = [r for r in all_results if r['arch_name'] == arch and r['adaptive'] == is_adapt]
            if matched:
                row += f" {matched[0]['best_l2']:>14.2e}"
            else:
                row += f" {'N/A':>14}"
        lines.append(row)

    lines.append("=" * 90)
    table_str = "\n".join(lines)
    print("\n" + table_str)

    with open(os.path.join(DATA_DIR, 'comparison_table.txt'), 'w') as f:
        f.write(table_str + "\n")

    header = 'model  adaptive  n_params  best_l2  final_l2  time_s'
    with open(os.path.join(DATA_DIR, 'comparison_data.txt'), 'w') as f:
        f.write(header + "\n")
        for r in all_results:
            f.write(f"{r['model_name']}  {r['adaptive']}  {r['n_params']}  "
                    f"{r['best_l2']:.6e}  {r['final_l2']:.6e}  {r['total_time']:.1f}\n")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Multi-scale Fourier Feature PINN — 1D Wave Equation")
    print(f"PDE: u_tt - {C_SPEED**2:.0f} * u_xx = 0")
    print(f"Exact: u = sin(pi*x)*cos(10*pi*t) + sin(2*pi*x)*cos(20*pi*t)")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    # --------------------------------------------------------
    # Problem setup
    # --------------------------------------------------------
    ics_coords = np.array([[0.0, 0.0], [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    bc2_coords = np.array([[0.0, 1.0], [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    # IC sampler: t=0, x in [0,1], u = sin(pi*x) + sin(2*pi*x)
    ics_sampler = Sampler(2, ics_coords,
                          lambda X: u_exact(X[:, 0:1], X[:, 1:2]), name='IC')
    # ut IC sampler (for u_t(x,0)=0 — same domain as IC)
    ut_sampler = Sampler(2, ics_coords,
                         lambda X: np.zeros((X.shape[0], 1)), name='ut_IC')
    bc1_sampler = Sampler(2, bc1_coords,
                          lambda X: np.zeros((X.shape[0], 1)), name='BC1')
    bc2_sampler = Sampler(2, bc2_coords,
                          lambda X: np.zeros((X.shape[0], 1)), name='BC2')
    res_sampler = Sampler(2, dom_coords,
                          lambda X: np.zeros((X.shape[0], 1)), name='Residual')

    bcs_sampler = [bc1_sampler, bc2_sampler]

    mu_X, sigma_X = compute_norm_stats(res_sampler)
    print(f"Normalization: mu={mu_X}, sigma={sigma_X}")

    N_ITER = 40000
    BATCH_SIZE = 360
    SEED = 1234
    NTK_EVERY_ADAPT = 100
    NTK_N_PTS = 128

    all_results = []

    # --------------------------------------------------------
    # Plain MLP: 3-layer, 200 hidden units
    # --------------------------------------------------------
    layers_plain = [2, 200, 200, 200, 1]

    for use_adapt in [False, True]:
        key = random.PRNGKey(SEED)
        params = init_mlp(layers_plain, key)

        res = train_model(
            model_name='Plain', apply_fn=apply_plain,
            params=params, frozen_args=(), n_frozen=0,
            mu_X=mu_X, sigma_X=sigma_X,
            ics_sampler=ics_sampler, bcs_sampler=bcs_sampler,
            res_sampler=res_sampler, ut_sampler=ut_sampler,
            use_adaptive_weights=use_adapt,
            n_iter=N_ITER, batch_size=BATCH_SIZE,
            ntk_every=NTK_EVERY_ADAPT, ntk_n_pts=NTK_N_PTS, seed=SEED,
        )
        save_model_data(res)
        plot_prediction_single(res)
        plot_loss_single(res)
        plot_adaptive_weights(res)
        all_results.append(res)

    # --------------------------------------------------------
    # MFF: Multi-scale Fourier Feature (sigma=1,10 joint on (t,x))
    # Shared hidden layers + doubled final layer input
    # --------------------------------------------------------
    hidden = 200
    n_ff_half = hidden // 2  # 100 Fourier features per branch -> 200 after sin/cos

    for use_adapt in [False, True]:
        key = random.PRNGKey(SEED)
        layers_mff_hidden = [hidden, hidden, hidden]  # shared hidden
        params_hidden = init_mlp(layers_mff_hidden, key)
        key, subkey = random.split(key)
        W_final, b_final = xavier_init(subkey, hidden * 2, 1)  # concat of 2 branches
        params_mff = params_hidden + [(W_final, b_final)]

        key_ff = random.PRNGKey(SEED + 10)
        k1, k2 = random.split(key_ff)
        W_ff1 = random.normal(k1, (2, n_ff_half)) * 1.0   # sigma=1
        W_ff2 = random.normal(k2, (2, n_ff_half)) * 10.0   # sigma=10

        res = train_model(
            model_name='MFF', apply_fn=apply_mff,
            params=params_mff, frozen_args=(W_ff1, W_ff2), n_frozen=2,
            mu_X=mu_X, sigma_X=sigma_X,
            ics_sampler=ics_sampler, bcs_sampler=bcs_sampler,
            res_sampler=res_sampler, ut_sampler=ut_sampler,
            use_adaptive_weights=use_adapt,
            n_iter=N_ITER, batch_size=BATCH_SIZE,
            ntk_every=NTK_EVERY_ADAPT, ntk_n_pts=NTK_N_PTS, seed=SEED,
        )
        save_model_data(res)
        plot_prediction_single(res)
        plot_loss_single(res)
        plot_adaptive_weights(res)
        all_results.append(res)

    # --------------------------------------------------------
    # ST_MFF: Spatio-Temporal Multi-scale FF
    # Spatial: sigma_x=1; Temporal: sigma_t1=1, sigma_t2=10
    # --------------------------------------------------------
    for use_adapt in [False, True]:
        key = random.PRNGKey(SEED)
        layers_stmff_hidden = [hidden, hidden, hidden]
        params_hidden = init_mlp(layers_stmff_hidden, key)
        key, subkey = random.split(key)
        W_final, b_final = xavier_init(subkey, hidden, 1)
        params_stmff = params_hidden + [(W_final, b_final)]

        key_st = random.PRNGKey(SEED + 20)
        k1, k2, k3 = random.split(key_st, 3)
        W_x = random.normal(k1, (1, n_ff_half)) * 1.0     # sigma_x = 1
        W_t1 = random.normal(k2, (1, n_ff_half)) * 1.0    # sigma_t1 = 1
        W_t2 = random.normal(k3, (1, n_ff_half)) * 10.0   # sigma_t2 = 10

        res = train_model(
            model_name='ST_MFF', apply_fn=apply_st_mff,
            params=params_stmff, frozen_args=(W_x, W_t1, W_t2), n_frozen=3,
            mu_X=mu_X, sigma_X=sigma_X,
            ics_sampler=ics_sampler, bcs_sampler=bcs_sampler,
            res_sampler=res_sampler, ut_sampler=ut_sampler,
            use_adaptive_weights=use_adapt,
            n_iter=N_ITER, batch_size=BATCH_SIZE,
            ntk_every=NTK_EVERY_ADAPT, ntk_n_pts=NTK_N_PTS, seed=SEED,
        )
        save_model_data(res)
        plot_prediction_single(res)
        plot_loss_single(res)
        plot_adaptive_weights(res)
        all_results.append(res)

    # --------------------------------------------------------
    # Comparison plots and table
    # --------------------------------------------------------
    plot_table2(all_results)
    plot_comparison_l2(all_results)
    plot_comparison_prediction(all_results)
    plot_comparison_loss(all_results)
    plot_comparison_adaptive_weights(all_results)
    plot_comparison_ntk(all_results)
    plot_comparison_bar(all_results)
    save_comparison_table(all_results)

    # L2 comparison data
    max_len = max(len(r['iters_log']) for r in all_results)
    cols = [np.array(all_results[0]['iters_log'][:max_len])]
    for res in all_results:
        l2 = list(res['l2_error_log'])
        while len(l2) < max_len:
            l2.append(l2[-1])
        cols.append(np.array(l2[:max_len]))

    l2_data = np.column_stack(cols)
    header = 'iteration  ' + '  '.join([f'l2_{r["model_name"]}' for r in all_results])
    np.savetxt(os.path.join(DATA_DIR, 'l2_error_comparison.txt'), l2_data,
               header=header, fmt='%.6e')

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
