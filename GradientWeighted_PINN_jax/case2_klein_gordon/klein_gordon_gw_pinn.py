"""
Gradient-Weighted PINN for Klein-Gordon Equation — JAX
=======================================================
Reference:
  Wang, Teng & Perdikaris, "Understanding and mitigating gradient flow
  pathologies in physics-informed neural networks",
  SIAM J. Sci. Comput., 43(5), A3055-A3081, 2021.

PDE:  u_tt + alpha*u_xx + beta*u + gamma*u^k = f(t,x)
Parameters: alpha=-1, beta=0, gamma=1, k=3
Domain: (t,x) in [0,1]^2
Exact: u(t,x) = x*cos(5*pi*t) + (t*x)^3

Two modes:
  M1 — Standard PINN (lambda_ics = lambda_bcs = 1, fixed)
  M2 — Gradient-weighted adaptive lambda_ics and lambda_bcs (EMA, beta=0.9)

Network: [2, 50, 50, 50, 50, 50, 1], tanh, Xavier init
Training: Adam + exponential LR decay, 40001 iterations, batch=128

Self-contained single file.  Run:
    python klein_gordon_gw_pinn.py [--mode M1|M2|both] [--niter N] [--quick]
"""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import argparse
import pickle
import time

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import optax
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# Paths
# ============================================================
WORKDIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKDIR, 'data')
FIG_DIR = os.path.join(WORKDIR, 'figures')
CKPT_DIR = os.path.join(WORKDIR, 'checkpoints')
for d in [DATA_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Plot style
# ============================================================
def setup_plot_style():
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 16
    rcParams['axes.labelsize'] = 18
    rcParams['axes.titlesize'] = 18
    rcParams['axes.linewidth'] = 2.0
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['xtick.major.width'] = 1.8
    rcParams['ytick.major.width'] = 1.8
    rcParams['xtick.major.size'] = 6
    rcParams['ytick.major.size'] = 6
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['legend.fontsize'] = 14
    rcParams['legend.framealpha'] = 0.9
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'

# ============================================================
# Configuration
# ============================================================
ALPHA = -1.0
BETA_PDE = 0.0
GAMMA = 1.0
K_EXP = 3

LAYERS = [2, 50, 50, 50, 50, 50, 1]
BATCH_SIZE = 128
SEED = 1234
BETA_EMA = 0.9
LR_INIT = 1e-3
LR_DECAY_RATE = 0.9
LR_DECAY_STEPS = 1000
LOG_EVERY = 100

DOM_COORDS = np.array([[0.0, 0.0], [1.0, 1.0]])
ICS_COORDS = np.array([[0.0, 0.0], [0.0, 1.0]])
BC1_COORDS = np.array([[0.0, 0.0], [1.0, 0.0]])
BC2_COORDS = np.array([[0.0, 1.0], [1.0, 1.0]])

# ============================================================
# Exact solution and forcing
# ============================================================
def u_exact_np(X):
    t, x = X[:, 0:1], X[:, 1:2]
    return x * np.cos(5.0 * np.pi * t) + (t * x) ** 3

def u_t_exact_np(X):
    t, x = X[:, 0:1], X[:, 1:2]
    return -5.0 * np.pi * x * np.sin(5.0 * np.pi * t) + 3.0 * t ** 2 * x ** 3

def u_tt_exact_np(X):
    t, x = X[:, 0:1], X[:, 1:2]
    return -25.0 * np.pi ** 2 * x * np.cos(5.0 * np.pi * t) + 6.0 * t * x ** 3

def u_xx_exact_np(X):
    t, x = X[:, 0:1], X[:, 1:2]
    return 6.0 * x * t ** 3

def f_exact_np(X):
    return (u_tt_exact_np(X) + ALPHA * u_xx_exact_np(X)
            + BETA_PDE * u_exact_np(X) + GAMMA * u_exact_np(X) ** K_EXP)

# ============================================================
# MLP Utilities
# ============================================================
def xavier_init(key, fan_in, fan_out):
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return random.normal(key, (fan_in, fan_out), dtype=jnp.float32) * std

def init_mlp_params(key, layer_sizes):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, wk, bk = random.split(key, 3)
        w = xavier_init(wk, layer_sizes[i], layer_sizes[i + 1])
        b = jnp.zeros(layer_sizes[i + 1], dtype=jnp.float32)
        params.append({'w': w, 'b': b})
    return params

def mlp_forward(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer['w'] + layer['b'])
    last = params[-1]
    return x @ last['w'] + last['b']

def count_params(params):
    return sum(l['w'].size + l['b'].size for l in params)

# ============================================================
# Data I/O
# ============================================================
def save_params(params, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(jax.tree.map(np.array, params), f)

def load_params(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_training_history(history, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header_line = '\t'.join(history.keys())
    data = np.column_stack([np.array(v) for v in history.values()])
    np.savetxt(filepath, data, header=header_line, delimiter='\t', fmt='%.8e')

def load_training_history(filepath):
    with open(filepath, 'r') as f:
        header = f.readline().strip().lstrip('# ').split('\t')
    data = np.loadtxt(filepath, delimiter='\t')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {k: data[:, i] for i, k in enumerate(header)}

def save_predictions(filepath, **arrays):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez(filepath, **{k: np.array(v) for k, v in arrays.items()})

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
        return x.astype(np.float32), y.astype(np.float32)

# ============================================================
# Network scalar functions
# ============================================================
# TF-style: network takes pre-normalized (t_n, x_n).
# Derivatives w.r.t. normalized inputs need /sigma correction for physical derivs.
def net_u_scalar(params, t_n, x_n):
    inp = jnp.array([t_n, x_n])
    return mlp_forward(params, inp)[0]

def _u_t_physical(params, t_n, x_n, sigma_t):
    return grad(net_u_scalar, argnums=1)(params, t_n, x_n) / sigma_t

def residual_single(params, t_n, x_n, f_val, sigma_t, sigma_x):
    u = net_u_scalar(params, t_n, x_n)

    u_t = grad(net_u_scalar, 1)(params, t_n, x_n) / sigma_t
    u_tt = grad(lambda p, tn, xn:
                grad(net_u_scalar, 1)(p, tn, xn) / sigma_t,
                argnums=1)(params, t_n, x_n) / sigma_t
    u_xx = grad(lambda p, tn, xn:
                grad(net_u_scalar, 2)(p, tn, xn) / sigma_x,
                argnums=2)(params, t_n, x_n) / sigma_x

    pde = u_tt + ALPHA * u_xx + BETA_PDE * u + GAMMA * u ** K_EXP
    return pde - f_val

residual_batch = vmap(residual_single, in_axes=(None, 0, 0, 0, None, None))
net_u_batch = vmap(net_u_scalar, in_axes=(None, 0, 0))
u_t_phys_batch = vmap(_u_t_physical, in_axes=(None, 0, 0, None))

# ============================================================
# Loss functions (separate for gradient weighting)
# All inputs are PRE-NORMALIZED (t_n, x_n), sigma passed separately.
# ============================================================
def loss_res_fn(params, t_n, x_n, f_r, sigma_t, sigma_x):
    r = residual_batch(params, t_n, x_n, f_r, sigma_t, sigma_x)
    return jnp.mean(r ** 2)

def loss_ics_fn(params, t_n, x_n, u_ic, sigma_t):
    u_pred = net_u_batch(params, t_n, x_n)
    u_t_pred = u_t_phys_batch(params, t_n, x_n, sigma_t)
    return jnp.mean((u_pred - u_ic) ** 2) + jnp.mean(u_t_pred ** 2)

def loss_bcs_fn(params, t_n, x_n, u_bc):
    u_pred = net_u_batch(params, t_n, x_n)
    n_per_bc = t_n.shape[0] // 2
    l1 = jnp.mean((u_pred[:n_per_bc] - u_bc[:n_per_bc]) ** 2)
    l2 = jnp.mean((u_pred[n_per_bc:] - u_bc[n_per_bc:]) ** 2)
    return l1 + l2

def loss_total_fn(params, t_r, x_r, f_r, t_ic, x_ic, u_ic,
                  t_bc, x_bc, u_bc, sigma_t, sigma_x,
                  lam_ics, lam_bcs):
    l_res = loss_res_fn(params, t_r, x_r, f_r, sigma_t, sigma_x)
    l_ics = loss_ics_fn(params, t_ic, x_ic, u_ic, sigma_t)
    l_bcs = loss_bcs_fn(params, t_bc, x_bc, u_bc)
    total = l_res + lam_ics * l_ics + lam_bcs * l_bcs
    return total, (l_res, l_ics, l_bcs)

# ============================================================
# Adaptive gradient weighting (M2)
# ============================================================
def compute_adaptive_lambdas(params, t_r, x_r, f_r, t_ic, x_ic, u_ic,
                             t_bc, x_bc, u_bc, sigma_t, sigma_x,
                             lam_ics_cur, lam_bcs_cur):
    grads_res = grad(loss_res_fn)(params, t_r, x_r, f_r, sigma_t, sigma_x)

    def weighted_ics(params, t_ic, x_ic, u_ic, sigma_t):
        return lam_ics_cur * loss_ics_fn(params, t_ic, x_ic, u_ic, sigma_t)
    grads_ics = grad(weighted_ics)(params, t_ic, x_ic, u_ic, sigma_t)

    def weighted_bcs(params, t_bc, x_bc, u_bc):
        return lam_bcs_cur * loss_bcs_fn(params, t_bc, x_bc, u_bc)
    grads_bcs = grad(weighted_bcs)(params, t_bc, x_bc, u_bc)

    max_res_list = []
    mean_ics_list = []
    mean_bcs_list = []
    for g_r, g_i, g_b in zip(grads_res, grads_ics, grads_bcs):
        max_res_list.append(jnp.max(jnp.abs(g_r['w'])))
        mean_ics_list.append(jnp.mean(jnp.abs(g_i['w'])))
        mean_bcs_list.append(jnp.mean(jnp.abs(g_b['w'])))

    max_grad_res = jnp.max(jnp.array(max_res_list))
    mean_grad_ics = jnp.mean(jnp.array(mean_ics_list))
    mean_grad_bcs = jnp.mean(jnp.array(mean_bcs_list))

    lam_ics_hat = max_grad_res / (mean_grad_ics + 1e-10)
    lam_bcs_hat = max_grad_res / (mean_grad_bcs + 1e-10)
    return lam_ics_hat, lam_bcs_hat

# ============================================================
# Training
# ============================================================
def train_model(mode, n_iter, ics_sampler, bcs_samplers, res_sampler,
                mu_X, sigma_X, test_X, test_u):
    assert mode in ('M1', 'M2')
    print("=" * 70)
    print(f"Training mode: {mode} | Iterations: {n_iter}")
    print("=" * 70)

    mu_t_np, mu_x_np = mu_X[0], mu_X[1]
    sigma_t_np, sigma_x_np = sigma_X[0], sigma_X[1]
    sigma_t = jnp.float32(sigma_t_np)
    sigma_x = jnp.float32(sigma_x_np)

    key = random.PRNGKey(SEED)
    key, init_key = random.split(key)
    params = init_mlp_params(init_key, LAYERS)
    n_params = count_params(params)
    print(f"Network: {LAYERS}, total params = {n_params}")

    lr_schedule = optax.exponential_decay(
        init_value=LR_INIT,
        transition_steps=LR_DECAY_STEPS,
        decay_rate=LR_DECAY_RATE,
        staircase=False,
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    lam_ics_val = jnp.float32(1.0)
    lam_bcs_val = jnp.float32(1.0)

    def _normalize(X_np):
        return ((X_np - mu_X) / sigma_X).astype(np.float32)

    @jit
    def train_step_m1(params, opt_state, t_r, x_r, f_r,
                      t_ic, x_ic, u_ic, t_bc, x_bc, u_bc):
        lam_i = jnp.float32(1.0)
        lam_b = jnp.float32(1.0)
        (loss, (l_res, l_ics, l_bcs)), grads = jax.value_and_grad(
            loss_total_fn, has_aux=True)(
            params, t_r, x_r, f_r, t_ic, x_ic, u_ic,
            t_bc, x_bc, u_bc, sigma_t, sigma_x, lam_i, lam_b
        )
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, l_res, l_ics, l_bcs

    @jit
    def train_step_m2(params, opt_state, t_r, x_r, f_r,
                      t_ic, x_ic, u_ic, t_bc, x_bc, u_bc,
                      lam_ics, lam_bcs):
        (loss, (l_res, l_ics, l_bcs)), grads = jax.value_and_grad(
            loss_total_fn, has_aux=True)(
            params, t_r, x_r, f_r, t_ic, x_ic, u_ic,
            t_bc, x_bc, u_bc, sigma_t, sigma_x, lam_ics, lam_bcs
        )
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, l_res, l_ics, l_bcs

    @jit
    def compute_lams_jit(params, t_r, x_r, f_r, t_ic, x_ic, u_ic,
                         t_bc, x_bc, u_bc, li, lb):
        return compute_adaptive_lambdas(
            params, t_r, x_r, f_r, t_ic, x_ic, u_ic,
            t_bc, x_bc, u_bc, sigma_t, sigma_x, li, lb)

    @jit
    def predict_u_jit(params, t_n_arr, x_n_arr):
        return net_u_batch(params, t_n_arr, x_n_arr)

    history = {
        'iter': [], 'loss_total': [], 'loss_res': [], 'loss_ics': [],
        'loss_bcs': [], 'l2_u': [], 'lambda_ics': [], 'lambda_bcs': []
    }

    np_rng = np.random.RandomState(SEED)
    test_X_n = _normalize(test_X)
    test_tn_j = jnp.array(test_X_n[:, 0], dtype=jnp.float32)
    test_xn_j = jnp.array(test_X_n[:, 1], dtype=jnp.float32)
    test_u_flat = test_u.flatten()

    t0 = time.time()
    for it in range(n_iter):
        X_ic, u_ic_batch = ics_sampler.sample(BATCH_SIZE, rng=np_rng)
        X_bc1, u_bc1 = bcs_samplers[0].sample(BATCH_SIZE, rng=np_rng)
        X_bc2, u_bc2 = bcs_samplers[1].sample(BATCH_SIZE, rng=np_rng)
        X_res, f_res = res_sampler.sample(BATCH_SIZE, rng=np_rng)

        X_ic_n = _normalize(X_ic)
        X_bc1_n = _normalize(X_bc1)
        X_bc2_n = _normalize(X_bc2)
        X_res_n = _normalize(X_res)

        t_r = jnp.array(X_res_n[:, 0])
        x_r = jnp.array(X_res_n[:, 1])
        f_r = jnp.array(f_res.flatten())

        t_ic = jnp.array(X_ic_n[:, 0])
        x_ic = jnp.array(X_ic_n[:, 1])
        u_ic_j = jnp.array(u_ic_batch.flatten())

        t_bc = jnp.concatenate([jnp.array(X_bc1_n[:, 0]), jnp.array(X_bc2_n[:, 0])])
        x_bc = jnp.concatenate([jnp.array(X_bc1_n[:, 1]), jnp.array(X_bc2_n[:, 1])])
        u_bc = jnp.concatenate([jnp.array(u_bc1.flatten()), jnp.array(u_bc2.flatten())])

        if mode == 'M1':
            params, opt_state, loss_val, l_res, l_ics, l_bcs = train_step_m1(
                params, opt_state, t_r, x_r, f_r, t_ic, x_ic, u_ic_j,
                t_bc, x_bc, u_bc)
        else:
            params, opt_state, loss_val, l_res, l_ics, l_bcs = train_step_m2(
                params, opt_state, t_r, x_r, f_r, t_ic, x_ic, u_ic_j,
                t_bc, x_bc, u_bc, lam_ics_val, lam_bcs_val)
            if it % 10 == 0:
                li_hat, lb_hat = compute_lams_jit(
                    params, t_r, x_r, f_r, t_ic, x_ic, u_ic_j,
                    t_bc, x_bc, u_bc, lam_ics_val, lam_bcs_val)
                lam_ics_val = (1.0 - BETA_EMA) * li_hat + BETA_EMA * lam_ics_val
                lam_bcs_val = (1.0 - BETA_EMA) * lb_hat + BETA_EMA * lam_bcs_val

        if it % LOG_EVERY == 0:
            u_p = predict_u_jit(params, test_tn_j, test_xn_j)
            l2_u = float(jnp.linalg.norm(np.array(u_p) - test_u_flat) /
                         jnp.linalg.norm(test_u_flat))

            li_v = float(lam_ics_val) if mode == 'M2' else 1.0
            lb_v = float(lam_bcs_val) if mode == 'M2' else 1.0
            history['iter'].append(it)
            history['loss_total'].append(float(loss_val))
            history['loss_res'].append(float(l_res))
            history['loss_ics'].append(float(l_ics))
            history['loss_bcs'].append(float(l_bcs))
            history['l2_u'].append(l2_u)
            history['lambda_ics'].append(li_v)
            history['lambda_bcs'].append(lb_v)

            elapsed = time.time() - t0
            print(f"Iter {it:6d}/{n_iter} | Loss: {float(loss_val):.4e} | "
                  f"L_res: {float(l_res):.4e} | L_ics: {float(l_ics):.4e} | "
                  f"L_bcs: {float(l_bcs):.4e} | L2(u): {l2_u:.4e} | "
                  f"lam_ic: {li_v:.3f} lam_bc: {lb_v:.3f} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - t0
    min_l2 = min(history['l2_u']) if history['l2_u'] else 0.0

    print(f"\n{'='*70}")
    print(f"Model: {mode} | Params: {n_params} | Min L2(u): {min_l2:.6e} | "
          f"Total Time: {total_time:.1f}s")
    print(f"{'='*70}\n")

    save_training_history(history, os.path.join(DATA_DIR, f'loss_history_{mode}.txt'))
    save_params(params, os.path.join(CKPT_DIR, f'params_{mode}.pkl'))

    nn = 100
    t_grid = np.linspace(0.0, 1.0, nn)
    x_grid = np.linspace(0.0, 1.0, nn)
    T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)
    TX_star = np.hstack([T_mesh.flatten()[:, None], X_mesh.flatten()[:, None]]).astype(np.float32)
    TX_star_n = _normalize(TX_star)

    u_star = u_exact_np(TX_star).flatten()
    u_pred_flat = np.array(predict_u_jit(params,
                                         jnp.array(TX_star_n[:, 0]),
                                         jnp.array(TX_star_n[:, 1])))

    u_star_grid = u_star.reshape(nn, nn)
    u_pred_grid = u_pred_flat.reshape(nn, nn)

    save_predictions(
        os.path.join(DATA_DIR, f'predictions_{mode}.npz'),
        t_grid=t_grid, x_grid=x_grid,
        u_pred=u_pred_grid, u_exact=u_star_grid,
    )

    final_l2 = np.linalg.norm(u_pred_flat - u_star) / np.linalg.norm(u_star)
    print(f"Final L2 error ({mode}): {final_l2:.6e}")

    return {
        'mode': mode, 'n_params': n_params, 'min_l2': min_l2,
        'final_l2': final_l2, 'total_time': total_time,
    }

# ============================================================
# Plotting
# ============================================================
def plot_solution_field(mode, filepath):
    setup_plot_style()
    pred = np.load(os.path.join(DATA_DIR, f'predictions_{mode}.npz'))
    t_grid, x_grid = pred['t_grid'], pred['x_grid']
    T, X = np.meshgrid(t_grid, x_grid)
    u_pred = pred['u_pred']
    u_exact = pred['u_exact']

    vmin = min(u_exact.min(), u_pred.min())
    vmax = max(u_exact.max(), u_pred.max())
    err = np.abs(u_pred - u_exact)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    titles = ['Exact $u(t,x)$', f'Predicted $u(t,x)$ ({mode})', 'Absolute Error']
    data_list = [u_exact, u_pred, err]
    cmaps = ['jet', 'jet', 'hot_r']
    labels = ['(a)', '(b)', '(c)']

    for idx, (ax, title, data, cmap, label) in enumerate(
            zip(axes, titles, data_list, cmaps, labels)):
        if idx < 2:
            cs = ax.pcolormesh(T, X, data, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='auto')
        else:
            cs = ax.pcolormesh(T, X, data, cmap=cmap, shading='auto')
        cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=12, width=1.5)
        for spine in cb.ax.spines.values():
            spine.set_linewidth(1.5)
        ax.set_xlabel('$t$', fontsize=18, fontweight='bold')
        ax.set_ylabel('$x$', fontsize=18, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_aspect('equal')
        ax.text(0.02, -0.12, label, transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top')

    fig.suptitle(f'Klein-Gordon Solution — {mode}', fontsize=20,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_solution_m1_vs_m2(filepath):
    setup_plot_style()
    p1 = os.path.join(DATA_DIR, 'predictions_M1.npz')
    p2 = os.path.join(DATA_DIR, 'predictions_M2.npz')
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("  Skipping M1-M2 comparison (need both).")
        return

    d1, d2 = np.load(p1), np.load(p2)
    t_grid, x_grid = d1['t_grid'], d1['x_grid']
    T, X = np.meshgrid(t_grid, x_grid)
    u_exact = d1['u_exact']
    u_m1, u_m2 = d1['u_pred'], d2['u_pred']
    err_m1 = np.abs(u_m1 - u_exact)
    err_m2 = np.abs(u_m2 - u_exact)

    vmin_u = min(u_exact.min(), u_m1.min(), u_m2.min())
    vmax_u = max(u_exact.max(), u_m1.max(), u_m2.max())
    vmax_err = max(err_m1.max(), err_m2.max())

    fig, axes = plt.subplots(2, 3, figsize=(20, 10.5))
    row_data = [
        ('M1', u_exact, u_m1, err_m1, ['(a)', '(b)', '(c)']),
        ('M2', u_exact, u_m2, err_m2, ['(d)', '(e)', '(f)']),
    ]
    for r, (name, exact, pred, err, lbls) in enumerate(row_data):
        for c, (data, title, cmap) in enumerate([
            (exact, 'Exact $u(t,x)$', 'jet'),
            (pred, f'Predicted $u(t,x)$ ({name})', 'jet'),
            (err, f'Absolute Error ({name})', 'hot_r'),
        ]):
            ax = axes[r, c]
            if c < 2:
                cs = ax.pcolormesh(T, X, data, cmap=cmap,
                                   vmin=vmin_u, vmax=vmax_u, shading='auto')
            else:
                cs = ax.pcolormesh(T, X, data, cmap=cmap,
                                   vmin=0, vmax=vmax_err, shading='auto')
            cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=11, width=1.3)
            for spine in cb.ax.spines.values():
                spine.set_linewidth(1.3)
            ax.set_xlabel('$t$', fontsize=16, fontweight='bold')
            ax.set_ylabel('$x$', fontsize=16, fontweight='bold')
            ax.set_title(title, fontsize=15, fontweight='bold')
            ax.set_aspect('equal')
            ax.text(0.02, -0.10, lbls[c], transform=ax.transAxes,
                    fontsize=18, fontweight='bold', va='top')

    fig.suptitle('Klein-Gordon — M1 vs M2 Comparison',
                 fontsize=20, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_loss_comparison(filepath):
    setup_plot_style()
    modes_hist = {}
    for m in ['M1', 'M2']:
        p = os.path.join(DATA_DIR, f'loss_history_{m}.txt')
        if os.path.exists(p):
            modes_hist[m] = load_training_history(p)
    if len(modes_hist) < 2:
        print("  Skipping loss comparison (need both M1 and M2).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    ax = axes[0]
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['loss_res'],
                lw=2.5, label='$\\mathcal{L}_r$ (M1)', color='C0', ls='-')
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['loss_res'],
                lw=2.5, label='$\\mathcal{L}_r$ (M2)', color='C0', ls='--')
    ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=18, fontweight='bold')
    ax.set_title('Residual Loss', fontsize=18, fontweight='bold')
    ax.legend(loc='best', frameon=True, edgecolor='black', fontsize=13,
              fancybox=False)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.12, '(a)', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    ax = axes[1]
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['loss_ics'],
                lw=2.5, label='$\\mathcal{L}_{ic}$ (M1)', color='C1', ls='-')
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['loss_ics'],
                lw=2.5, label='$\\mathcal{L}_{ic}$ (M2)', color='C1', ls='--')
    ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=18, fontweight='bold')
    ax.set_title('IC Loss', fontsize=18, fontweight='bold')
    ax.legend(loc='best', frameon=True, edgecolor='black', fontsize=13,
              fancybox=False)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.12, '(b)', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    ax = axes[2]
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['loss_bcs'],
                lw=2.5, label='$\\mathcal{L}_{bc}$ (M1)', color='C2', ls='-')
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['loss_bcs'],
                lw=2.5, label='$\\mathcal{L}_{bc}$ (M2)', color='C2', ls='--')
    ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=18, fontweight='bold')
    ax.set_title('BC Loss', fontsize=18, fontweight='bold')
    ax.legend(loc='best', frameon=True, edgecolor='black', fontsize=13,
              fancybox=False)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.12, '(c)', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_l2_error_comparison(filepath):
    setup_plot_style()
    modes_hist = {}
    for m in ['M1', 'M2']:
        p = os.path.join(DATA_DIR, f'loss_history_{m}.txt')
        if os.path.exists(p):
            modes_hist[m] = load_training_history(p)
    if len(modes_hist) < 2:
        print("  Skipping L2 error comparison.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['l2_u'],
                lw=2.5, label='M1 (Standard)', color='C0')
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['l2_u'],
                lw=2.5, label='M2 (Gradient-Weighted)', color='C3')
    ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
    ax.set_ylabel('$L_2$ Relative Error ($u$)', fontsize=18, fontweight='bold')
    ax.set_title('$L_2$ Error Comparison', fontsize=18, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=14,
              fancybox=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_adaptive_lambda(filepath):
    setup_plot_style()
    p = os.path.join(DATA_DIR, 'loss_history_M2.txt')
    if not os.path.exists(p):
        print("  Skipping adaptive lambda plot.")
        return
    hist = load_training_history(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hist['iter'], hist['lambda_ics'], lw=2.5, color='C1',
            label='$\\hat{\\lambda}_{ic}$')
    ax.plot(hist['iter'], hist['lambda_bcs'], lw=2.5, color='C2',
            label='$\\hat{\\lambda}_{bc}$')
    ax.set_xlabel('Iteration', fontsize=18, fontweight='bold')
    ax.set_ylabel('Adaptive Weight', fontsize=18, fontweight='bold')
    ax.set_title('Adaptive Weights $\\hat{\\lambda}_{ic}$ and $\\hat{\\lambda}_{bc}$ (M2)',
                 fontsize=18, fontweight='bold')
    ax.legend(loc='best', frameon=True, edgecolor='black', fontsize=14,
              fancybox=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def generate_all_plots():
    print("\n" + "=" * 70)
    print("Generating publication-quality figures ...")
    print("=" * 70)

    for m in ['M1', 'M2']:
        pred_path = os.path.join(DATA_DIR, f'predictions_{m}.npz')
        if os.path.exists(pred_path):
            plot_solution_field(m, os.path.join(FIG_DIR, f'fig_solution_{m}.png'))

    plot_solution_m1_vs_m2(os.path.join(FIG_DIR, 'fig_solution_M1_vs_M2.png'))
    plot_loss_comparison(os.path.join(FIG_DIR, 'fig_loss_comparison.png'))
    plot_l2_error_comparison(os.path.join(FIG_DIR, 'fig_l2_error_comparison.png'))
    plot_adaptive_lambda(os.path.join(FIG_DIR, 'fig_adaptive_lambda.png'))
    print("All figures generated.")


def write_comparison_summary():
    n_p = sum(LAYERS[i] * LAYERS[i + 1] + LAYERS[i + 1]
              for i in range(len(LAYERS) - 1))

    lines = []
    lines.append("=" * 78)
    lines.append("Comparison Summary: Klein-Gordon (alpha=-1, beta=0, gamma=1, k=3)")
    lines.append("Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021")
    lines.append("=" * 78)
    lines.append(f"Network: {LAYERS}, Activation: tanh, Total params: {n_p}")
    lines.append(f"Training: Adam + ExpDecay(lr0={LR_INIT}, decay={LR_DECAY_RATE}, "
                 f"steps={LR_DECAY_STEPS}), Batch={BATCH_SIZE}")
    lines.append("-" * 78)
    lines.append(f"{'Model':<8} {'Params':>8} {'Iterations':>11} "
                 f"{'Min L2(u)':>12} {'Final L2(u)':>12} {'Final L_res':>12} "
                 f"{'Final L_ics':>12} {'Final L_bcs':>12}")
    lines.append("-" * 78)

    for m in ['M1', 'M2']:
        hp = os.path.join(DATA_DIR, f'loss_history_{m}.txt')
        pp = os.path.join(DATA_DIR, f'predictions_{m}.npz')
        if os.path.exists(hp) and os.path.exists(pp):
            hist = load_training_history(hp)
            pred = np.load(pp)
            min_l2 = min(hist['l2_u'])
            u_pred = pred['u_pred']
            u_exact = pred['u_exact']
            final_l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
            n_iters = int(hist['iter'][-1]) + 1
            lines.append(f"{m:<8} {n_p:>8d} {n_iters:>11d} "
                         f"{min_l2:>12.6e} {final_l2:>12.6e} "
                         f"{hist['loss_res'][-1]:>12.4e} "
                         f"{hist['loss_ics'][-1]:>12.4e} "
                         f"{hist['loss_bcs'][-1]:>12.4e}")

    lines.append("-" * 78)

    if all(os.path.exists(os.path.join(DATA_DIR, f'loss_history_{m}.txt'))
           for m in ['M1', 'M2']):
        h1 = load_training_history(os.path.join(DATA_DIR, 'loss_history_M1.txt'))
        h2 = load_training_history(os.path.join(DATA_DIR, 'loss_history_M2.txt'))
        min_m1 = min(h1['l2_u'])
        min_m2 = min(h2['l2_u'])
        if min_m1 > 0:
            improvement = (min_m1 - min_m2) / min_m1 * 100
            lines.append(f"M2 improvement over M1: {improvement:.1f}% "
                         f"(Min L2: {min_m1:.4e} -> {min_m2:.4e})")

    lines.append("=" * 78)
    summary = '\n'.join(lines)
    print(summary)

    with open(os.path.join(DATA_DIR, 'comparison_summary.txt'), 'w') as f:
        f.write(summary + '\n')
    print(f"Saved: {os.path.join(DATA_DIR, 'comparison_summary.txt')}")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both',
                        choices=['M1', 'M2', 'both'])
    parser.add_argument('--niter', type=int, default=40001)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--plot_only', action='store_true')
    args = parser.parse_args()

    if args.plot_only:
        generate_all_plots()
        write_comparison_summary()
        return

    n_iter = 100 if args.quick else args.niter
    modes = ['M1', 'M2'] if args.mode == 'both' else [args.mode]

    print("=" * 70)
    print("Gradient-Weighted PINN — Klein-Gordon Equation")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"PDE: u_tt + ({ALPHA})*u_xx + ({BETA_PDE})*u + ({GAMMA})*u^{K_EXP} = f")
    print("=" * 70)

    X_sample = DOM_COORDS[0:1, :] + (DOM_COORDS[1:2, :] - DOM_COORDS[0:1, :]) * \
               np.random.rand(100000, 2)
    mu_X = X_sample.mean(0).astype(np.float32)
    sigma_X = X_sample.std(0).astype(np.float32)
    print(f"Normalization: mu={mu_X}, sigma={sigma_X}")

    ics_sampler = Sampler(2, ICS_COORDS, lambda x: u_exact_np(x), name='IC')
    bc1 = Sampler(2, BC1_COORDS, lambda x: u_exact_np(x), name='BC1 (x=0)')
    bc2 = Sampler(2, BC2_COORDS, lambda x: u_exact_np(x), name='BC2 (x=1)')
    bcs_samplers = [bc1, bc2]
    res_sampler = Sampler(2, DOM_COORDS, lambda x: f_exact_np(x), name='Forcing')

    nn = 100
    t_test = np.linspace(0.0, 1.0, nn)
    x_test = np.linspace(0.0, 1.0, nn)
    T_test, X_test = np.meshgrid(t_test, x_test)
    TX_star = np.hstack([T_test.flatten()[:, None],
                         X_test.flatten()[:, None]]).astype(np.float32)
    u_star = u_exact_np(TX_star)

    for mode in modes:
        train_model(mode, n_iter, ics_sampler, bcs_samplers, res_sampler,
                    mu_X, sigma_X, TX_star, u_star)

    generate_all_plots()
    write_comparison_summary()
    print("\nDone!")


if __name__ == '__main__':
    main()
