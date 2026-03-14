"""
Case 2: 1D Wave Equation — NTK-Adaptive Weight PINN (JAX)
WITH z-score input normalization + gradient chain-rule correction

PDE: u_tt = c^2 * u_xx,  (t, x) in [0, 1]^2,  c = 2
Exact: u = sin(pi*x)*cos(c*pi*t) + a*sin(2*c*pi*x)*cos(4*c*pi*t)
       with a = 0.5, c = 2

Normalization: input z-score ((t,x)_norm = ((t,x) - mu) / sigma)
               gradient correction (d/dt_phys = (1/sigma_t) * d/dt_norm, etc.)
"""

import os
import time
import pickle

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
from scipy.interpolate import griddata

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 2.0
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5

WORKDIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKDIR, 'data')
FIG_DIR = os.path.join(WORKDIR, 'figures')
CKPT_DIR = os.path.join(WORKDIR, 'checkpoints')
for d in [DATA_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Problem parameters
# ============================================================
A_PARAM = 0.5
C_PARAM = 2.0

def u_exact_np(t, x):
    return (np.sin(np.pi * x) * np.cos(C_PARAM * np.pi * t) +
            A_PARAM * np.sin(2 * C_PARAM * np.pi * x) * np.cos(4 * C_PARAM * np.pi * t))

# ============================================================
# Normalization statistics
# Computed from uniform samples over the residual domain [0,1]^2
# ============================================================
rng_stat = np.random.RandomState(0)
_N_STAT = 100000
_T_stat = rng_stat.uniform(0, 1, _N_STAT)
_X_stat = rng_stat.uniform(0, 1, _N_STAT)
MU_T = float(np.mean(_T_stat))
MU_X = float(np.mean(_X_stat))
SIGMA_T = float(np.std(_T_stat))
SIGMA_X = float(np.std(_X_stat))

def normalize_t(t):
    return (t - MU_T) / SIGMA_T

def normalize_x(x):
    return (x - MU_X) / SIGMA_X

# ============================================================
# Network
# ============================================================
LAYERS = [2, 500, 500, 500, 1]

def init_params(layers, key):
    params = []
    for i in range(len(layers) - 1):
        k1, k2, key = random.split(key, 3)
        fan_in = layers[i]
        fan_out = layers[i + 1]
        xavier_std = 1.0 / np.sqrt((fan_in + fan_out) / 2.0)
        W = xavier_std * random.normal(k1, (layers[i], layers[i + 1]))
        b = jnp.zeros((layers[i + 1],))
        params.append((W, b))
    return params


def apply_net(params, tx):
    h = tx
    for (W, b) in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)


def net_u_single(params, t_norm, x_norm):
    """Scalar (t_norm, x_norm) -> scalar u."""
    inp = jnp.array([[t_norm, x_norm]])
    return apply_net(params, inp)[0, 0]


def net_u_t_single(params, t_norm, x_norm):
    """du/dt in physical space = (1/sigma_t) * d(net)/d(t_norm)."""
    du_dt_norm = grad(net_u_single, argnums=1)(params, t_norm, x_norm)
    return du_dt_norm / SIGMA_T


def net_residual_single(params, t_norm, x_norm):
    """Residual: u_tt - c^2 * u_xx in physical space.
    Chain rule: d^2/dt^2_phys = (1/sigma_t^2) * d^2/dt_norm^2, etc.
    """
    du_dt_norm = grad(net_u_single, argnums=1)
    du_dx_norm = grad(net_u_single, argnums=2)
    d2u_dt2_norm = grad(du_dt_norm, argnums=1)
    d2u_dx2_norm = grad(du_dx_norm, argnums=2)
    u_tt_phys = d2u_dt2_norm(params, t_norm, x_norm) / (SIGMA_T ** 2)
    u_xx_phys = d2u_dx2_norm(params, t_norm, x_norm) / (SIGMA_X ** 2)
    return u_tt_phys - C_PARAM ** 2 * u_xx_phys

net_u_batch = jit(vmap(net_u_single, in_axes=(None, 0, 0)))
net_u_t_batch = jit(vmap(net_u_t_single, in_axes=(None, 0, 0)))
net_residual_batch = jit(vmap(net_residual_single, in_axes=(None, 0, 0)))

# ============================================================
# Batch prediction (non-vmap, fast)
# ============================================================
def predict_u_batch(params, t_norm_arr, x_norm_arr):
    inp = jnp.stack([t_norm_arr, x_norm_arr], axis=-1)
    return apply_net(params, inp)[:, 0]

predict_u_batch_jit = jit(predict_u_batch)

# ============================================================
# Losses
# ============================================================
def loss_ics_u_fn(params, t_ic_n, x_ic_n, u_ic):
    u_pred = net_u_batch(params, t_ic_n, x_ic_n)
    return jnp.mean((u_pred - u_ic) ** 2)


def loss_ics_ut_fn(params, t_ic_n, x_ic_n):
    u_t_pred = net_u_t_batch(params, t_ic_n, x_ic_n)
    return jnp.mean(u_t_pred ** 2)


def loss_bc_fn(params, t_bc_n, x_bc_n):
    u_pred = net_u_batch(params, t_bc_n, x_bc_n)
    return jnp.mean(u_pred ** 2)


def loss_res_fn(params, t_r_n, x_r_n):
    r_pred = net_residual_batch(params, t_r_n, x_r_n)
    return jnp.mean(r_pred ** 2)


def total_loss_fn(params, t_ic_n, x_ic_n, u_ic, t_bc1_n, x_bc1_n, t_bc2_n, x_bc2_n,
                  t_r_n, x_r_n, lam_u, lam_ut, lam_r):
    l_ics_u = loss_ics_u_fn(params, t_ic_n, x_ic_n, u_ic)
    l_ics_ut = loss_ics_ut_fn(params, t_ic_n, x_ic_n)
    l_bc1 = loss_bc_fn(params, t_bc1_n, x_bc1_n)
    l_bc2 = loss_bc_fn(params, t_bc2_n, x_bc2_n)

    l_bcs = l_ics_u + l_bc1 + l_bc2
    l_res = loss_res_fn(params, t_r_n, x_r_n)

    total = lam_u * l_bcs + lam_ut * l_ics_ut + lam_r * l_res
    return total, (l_bcs, l_ics_ut, l_res)

# ============================================================
# NTK computation
# ============================================================
def compute_jacobian_u(params, t_pts, x_pts):
    flat_params, unravel = ravel_pytree(params)
    def f_flat(fp):
        return net_u_batch(unravel(fp), t_pts, x_pts)
    return jacrev(f_flat)(flat_params)


def compute_jacobian_ut(params, t_pts, x_pts):
    flat_params, unravel = ravel_pytree(params)
    def f_flat(fp):
        return net_u_t_batch(unravel(fp), t_pts, x_pts)
    return jacrev(f_flat)(flat_params)


def compute_jacobian_r(params, t_pts, x_pts):
    flat_params, unravel = ravel_pytree(params)
    def f_flat(fp):
        return net_residual_batch(unravel(fp), t_pts, x_pts)
    return jacrev(f_flat)(flat_params)


def compute_ntk_diag_blocks(params, t_bc_n, x_bc_n, t_ic_n, x_ic_n, t_r_n, x_r_n):
    J_u = compute_jacobian_u(params, t_bc_n, x_bc_n)
    J_ut = compute_jacobian_ut(params, t_ic_n, x_ic_n)
    J_r = compute_jacobian_r(params, t_r_n, x_r_n)
    K_u = J_u @ J_u.T
    K_ut = J_ut @ J_ut.T
    K_r = J_r @ J_r.T
    return K_u, K_ut, K_r

# ============================================================
# Sampling (returns NORMALIZED coordinates)
# ============================================================
def sample_ics(key, N):
    t_phys = jnp.zeros(N)
    x_phys = random.uniform(key, (N,), minval=0.0, maxval=1.0)
    u = jnp.sin(jnp.pi * x_phys) + A_PARAM * jnp.sin(2 * C_PARAM * jnp.pi * x_phys)
    t_n = (t_phys - MU_T) / SIGMA_T
    x_n = (x_phys - MU_X) / SIGMA_X
    return t_n, x_n, u


def sample_bc1(key, N):
    t_phys = random.uniform(key, (N,), minval=0.0, maxval=1.0)
    x_phys = jnp.zeros(N)
    return (t_phys - MU_T) / SIGMA_T, (x_phys - MU_X) / SIGMA_X


def sample_bc2(key, N):
    t_phys = random.uniform(key, (N,), minval=0.0, maxval=1.0)
    x_phys = jnp.ones(N)
    return (t_phys - MU_T) / SIGMA_T, (x_phys - MU_X) / SIGMA_X


def sample_res(key, N):
    k1, k2 = random.split(key)
    t_phys = random.uniform(k1, (N,), minval=0.0, maxval=1.0)
    x_phys = random.uniform(k2, (N,), minval=0.0, maxval=1.0)
    return (t_phys - MU_T) / SIGMA_T, (x_phys - MU_X) / SIGMA_X

# ============================================================
# Training
# ============================================================
N_ITER = 80001
BATCH_SIZE = 300
KERNEL_SIZE = 300
LR_INIT = 1e-3
LOG_EVERY = 100
NTK_EVERY = 100

def train():
    key = random.PRNGKey(1234)
    key, init_key = random.split(key)
    params = init_params(LAYERS, init_key)

    schedule = optax.exponential_decay(
        init_value=LR_INIT, transition_steps=1000, decay_rate=0.9, staircase=False)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    flat0, _ = ravel_pytree(params)
    n_params = flat0.shape[0]

    lam_u = 1.0
    lam_ut = 1.0
    lam_r = 1.0

    @jit
    def train_step(params, opt_state, t_ic_n, x_ic_n, u_ic,
                   t_bc1_n, x_bc1_n, t_bc2_n, x_bc2_n, t_r_n, x_r_n,
                   lam_u_val, lam_ut_val, lam_r_val):
        (loss_val, (l_bcs, l_ut, l_res)), grads = jax.value_and_grad(
            total_loss_fn, has_aux=True
        )(params, t_ic_n, x_ic_n, u_ic, t_bc1_n, x_bc1_n, t_bc2_n, x_bc2_n,
          t_r_n, x_r_n, lam_u_val, lam_ut_val, lam_r_val)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss_val, l_bcs, l_ut, l_res

    # Test grid (physical coords -> normalized for prediction)
    nn_test = 200
    t_test_1d = np.linspace(0, 1, nn_test)
    x_test_1d = np.linspace(0, 1, nn_test)
    T_grid, X_grid = np.meshgrid(t_test_1d, x_test_1d)
    T_flat = T_grid.flatten()
    X_flat = X_grid.flatten()
    U_exact_flat = u_exact_np(T_flat, X_flat)

    T_flat_norm = normalize_t(T_flat)
    X_flat_norm = normalize_x(X_flat)
    T_flat_jax = jnp.array(T_flat_norm)
    X_flat_jax = jnp.array(X_flat_norm)

    loss_bcs_log = []
    loss_ut_log = []
    loss_res_log = []
    l2_error_log = []
    lam_u_log = []
    lam_ut_log = []
    lam_r_log = []
    K_u_log = []
    K_ut_log = []
    K_r_log = []
    iters_log = []

    print(f"Number of trainable parameters: {n_params}")
    print(f"Network architecture: {LAYERS}")
    print(f"Optimizer: Adam, lr_init={LR_INIT}, exponential decay (0.9 per 1000 steps)")
    print(f"Normalization: z-score (mu_t={MU_T:.4f}, sigma_t={SIGMA_T:.4f}, "
          f"mu_x={MU_X:.4f}, sigma_x={SIGMA_X:.4f})")
    print(f"Iterations: {N_ITER}, batch_size={BATCH_SIZE}")
    print(f"NTK adaptive weights: ON\n")

    start_time = time.time()
    best_l2 = 1.0

    for it in range(N_ITER):
        key, k1, k2, k3, k4 = random.split(key, 5)
        bs = BATCH_SIZE

        t_ic_n, x_ic_n, u_ic = sample_ics(k1, bs // 3)
        t_bc1_n, x_bc1_n = sample_bc1(k2, bs // 3)
        t_bc2_n, x_bc2_n = sample_bc2(k3, bs // 3)
        t_r_n, x_r_n = sample_res(k4, bs)

        params, opt_state, loss_val, l_bcs, l_ut, l_res = train_step(
            params, opt_state, t_ic_n, x_ic_n, u_ic,
            t_bc1_n, x_bc1_n, t_bc2_n, x_bc2_n, t_r_n, x_r_n,
            jnp.float32(lam_u), jnp.float32(lam_ut), jnp.float32(lam_r)
        )

        if it % LOG_EVERY == 0:
            u_pred_flat = np.array(predict_u_batch_jit(params, T_flat_jax, X_flat_jax))
            l2_err = np.linalg.norm(U_exact_flat - u_pred_flat) / np.linalg.norm(U_exact_flat)

            loss_bcs_log.append(float(l_bcs))
            loss_ut_log.append(float(l_ut))
            loss_res_log.append(float(l_res))
            l2_error_log.append(float(l2_err))
            iters_log.append(it)

            if l2_err < best_l2:
                best_l2 = l2_err

            elapsed = time.time() - start_time
            print(f"It: {it:5d}, Loss: {float(loss_val):.3e}, "
                  f"L_res: {float(l_res):.3e}, L_bcs: {float(l_bcs):.3e}, "
                  f"L_ut: {float(l_ut):.3e}, L2: {l2_err:.3e}, Time: {elapsed:.1f}s")
            print(f"  lambda_u: {lam_u:.3e}, lambda_ut: {lam_ut:.3e}, lambda_r: {lam_r:.3e}")

        if it % NTK_EVERY == 0:
            key, k5, k6 = random.split(key, 3)
            t_bc_ntk = jnp.concatenate([t_ic_n, t_bc1_n, t_bc2_n])
            x_bc_ntk = jnp.concatenate([x_ic_n, x_bc1_n, x_bc2_n])

            t_ic_ntk, x_ic_ntk, _ = sample_ics(k5, KERNEL_SIZE)
            t_r_ntk, x_r_ntk = sample_res(k6, KERNEL_SIZE)

            K_u_val, K_ut_val, K_r_val = compute_ntk_diag_blocks(
                params, t_bc_ntk, x_bc_ntk, t_ic_ntk, x_ic_ntk, t_r_ntk, x_r_ntk
            )

            K_u_np = np.array(K_u_val)
            K_ut_np = np.array(K_ut_val)
            K_r_np = np.array(K_r_val)

            K_u_log.append(K_u_np)
            K_ut_log.append(K_ut_np)
            K_r_log.append(K_r_np)

            trace_K_u = np.trace(K_u_np)
            trace_K_ut = np.trace(K_ut_np)
            trace_K_r = np.trace(K_r_np)
            trace_total = trace_K_u + trace_K_ut + trace_K_r

            if trace_K_u > 0 and trace_K_ut > 0 and trace_K_r > 0:
                lam_u = float(trace_total / trace_K_u)
                lam_ut = float(trace_total / trace_K_ut)
                lam_r = float(trace_total / trace_K_r)

            lam_u_log.append(lam_u)
            lam_ut_log.append(lam_ut)
            lam_r_log.append(lam_r)

    total_time = time.time() - start_time
    print(f"\nTraining complete. Total time: {total_time:.1f}s")
    print(f"Best L2 relative error: {best_l2:.3e}")
    print(f"Final L2 relative error: {l2_error_log[-1]:.3e}")

    with open(os.path.join(CKPT_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    n_log = len(iters_log)
    loss_data = np.column_stack([
        iters_log, loss_res_log, loss_bcs_log, loss_ut_log, l2_error_log,
        lam_u_log[:n_log], lam_ut_log[:n_log], lam_r_log[:n_log]
    ])
    np.savetxt(os.path.join(DATA_DIR, 'loss_history.txt'), loss_data,
               header="iteration  loss_res  loss_bcs  loss_ut  l2_error  lambda_u  lambda_ut  lambda_r",
               fmt='%.6e')

    np.savetxt(os.path.join(DATA_DIR, 'lambda_history.txt'),
               np.column_stack([iters_log[:len(lam_u_log)], lam_u_log, lam_ut_log, lam_r_log]),
               header="iteration  lambda_u  lambda_ut  lambda_r", fmt='%.6e')

    u_pred_final = np.array(predict_u_batch_jit(params, T_flat_jax, X_flat_jax))
    pred_data = np.column_stack([T_flat, X_flat, U_exact_flat, u_pred_final,
                                 np.abs(U_exact_flat - u_pred_final)])
    np.savetxt(os.path.join(DATA_DIR, 'prediction.txt'), pred_data,
               header="t  x  u_exact  u_pred  abs_error", fmt='%.6e')

    snapshot_indices = [0, len(K_u_log) // 4, len(K_u_log) // 2, len(K_u_log) - 1]
    for si in snapshot_indices:
        iter_label = iters_log[si] if si < len(iters_log) else si * NTK_EVERY
        eig_u = np.sort(np.real(np.linalg.eigvalsh(K_u_log[si])))[::-1]
        eig_ut = np.sort(np.real(np.linalg.eigvalsh(K_ut_log[si])))[::-1]
        eig_r = np.sort(np.real(np.linalg.eigvalsh(K_r_log[si])))[::-1]
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Ku_iter{iter_label}.txt'), eig_u, fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Kut_iter{iter_label}.txt'), eig_ut, fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Kr_iter{iter_label}.txt'), eig_r, fmt='%.6e')

    plot_results(iters_log, loss_bcs_log, loss_ut_log, loss_res_log, l2_error_log,
                 lam_u_log, lam_ut_log, lam_r_log,
                 T_grid, X_grid, T_flat, X_flat, U_exact_flat, u_pred_final,
                 K_u_log, K_ut_log, K_r_log, snapshot_indices,
                 nn_test, n_params, total_time, best_l2)

    print("\n" + "=" * 60)
    print("SUMMARY — Case 2: Wave 1D (Normalized)")
    print("=" * 60)
    print(f"  Network:          {LAYERS}")
    print(f"  Parameters:       {n_params}")
    print(f"  Optimizer:        Adam (lr_init={LR_INIT}, exp decay)")
    print(f"  Normalization:    z-score (mu_t={MU_T:.4f}, sigma_t={SIGMA_T:.4f}, "
          f"mu_x={MU_X:.4f}, sigma_x={SIGMA_X:.4f})")
    print(f"  Iterations:       {N_ITER}")
    print(f"  Best L2 error:    {best_l2:.3e}")
    print(f"  Final L2 error:   {l2_error_log[-1]:.3e}")
    print(f"  Training time:    {total_time:.1f}s")
    print("=" * 60)


# ============================================================
# Plotting (journal quality)
# ============================================================
def _label_subplot(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')


def plot_results(iters_log, loss_bcs_log, loss_ut_log, loss_res_log, l2_error_log,
                 lam_u_log, lam_ut_log, lam_r_log,
                 T_grid, X_grid, T_flat, X_flat, U_exact_flat, u_pred_final,
                 K_u_log, K_ut_log, K_r_log, snapshot_indices,
                 nn_test, n_params, total_time, best_l2):

    iters_arr = np.array(iters_log)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.semilogy(iters_arr, loss_res_log, lw=2, label=r'$\mathcal{L}_{r}$')
    ax.semilogy(iters_arr, loss_bcs_log, lw=2, label=r'$\mathcal{L}_{u}$')
    ax.semilogy(iters_arr, loss_ut_log, lw=2, label=r'$\mathcal{L}_{u_t}$')
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(a)')
    ax = axes[1]
    ax.semilogy(iters_arr, l2_error_log, lw=2, color='tab:red')
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative $L^2$ error', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(b)')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    U_exact_grid = griddata((T_flat, X_flat), U_exact_flat, (T_grid, X_grid), method='cubic')
    U_pred_grid = griddata((T_flat, X_flat), u_pred_final, (T_grid, X_grid), method='cubic')
    Error_grid = np.abs(U_exact_grid - U_pred_grid)
    vmin_u = min(np.nanmin(U_exact_grid), np.nanmin(U_pred_grid))
    vmax_u = max(np.nanmax(U_exact_grid), np.nanmax(U_pred_grid))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    im = ax.pcolormesh(T_grid, X_grid, U_exact_grid, cmap='jet', shading='auto',
                       vmin=vmin_u, vmax=vmax_u)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('$t$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$x$', fontsize=14, fontweight='bold')
    ax.set_title('Exact $u(t, x)$', fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(a)')
    ax = axes[1]
    im = ax.pcolormesh(T_grid, X_grid, U_pred_grid, cmap='jet', shading='auto',
                       vmin=vmin_u, vmax=vmax_u)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('$t$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$x$', fontsize=14, fontweight='bold')
    ax.set_title('Predicted $u(t, x)$', fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(b)')
    ax = axes[2]
    im = ax.pcolormesh(T_grid, X_grid, Error_grid, cmap='jet', shading='auto')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('$t$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$x$', fontsize=14, fontweight='bold')
    ax.set_title('Absolute error', fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(c)')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [r'Eigenvalues of $K_u$', r'Eigenvalues of $K_{u_t}$', r'Eigenvalues of $K_r$']
    data_lists = [K_u_log, K_ut_log, K_r_log]
    for col, (ax, title, K_list) in enumerate(zip(axes, titles, data_lists)):
        for si in snapshot_indices:
            eig = np.sort(np.real(np.linalg.eigvalsh(K_list[si])))[::-1]
            ax.loglog(np.arange(1, len(eig) + 1), np.clip(eig, 1e-30, None),
                      lw=2, label=f'$n={iters_log[si]}$')
        ax.set_xlabel('Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Eigenvalue', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.tick_params(labelsize=12)
        _label_subplot(ax, f'({"abc"[col]})')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(snapshot_indices),
               fontsize=13, frameon=True, fancybox=False, edgecolor='black',
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, 'fig3_ntk_eigenvalues.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    n_lam = len(lam_u_log)
    iter_lam = iters_arr[:n_lam]
    ax.semilogy(iter_lam, lam_u_log, lw=2, label=r'$\lambda_u$')
    ax.semilogy(iter_lam, lam_ut_log, lw=2, label=r'$\lambda_{u_t}$')
    ax.semilogy(iter_lam, lam_r_log, lw=2, label=r'$\lambda_r$')
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\lambda$', fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(labelsize=12)
    ax.set_title('NTK-adaptive weights', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_adaptive_weights.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll figures saved to {FIG_DIR}")
    print(f"All data saved to {DATA_DIR}")


if __name__ == '__main__':
    train()
