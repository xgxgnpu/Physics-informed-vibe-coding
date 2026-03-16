"""
Multi-scale Fourier Feature PINN for 1D Heat Equation (JAX)
============================================================
Reproduces the heat1D case from:
  Wang, Wang & Perdikaris, "On the eigenvector bias of Fourier feature
  networks", CMAME 384, 113938 (2021).

Three model variants compared under identical conditions:
  1) NN   — Plain MLP
  2) FF   — Fourier Feature network
  3) ST_FF — Spatio-Temporal Fourier Feature network

Exact solution: u(t,x) = exp(-a*t) * sin(b*pi*x),  a=1, b=500
PDE:  u_t = k * u_xx,  k = a/(b*pi)^2
Domain: t in [0,1], x in [0,1]

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
from scipy.interpolate import griddata

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
A_PARAM = 1
B_PARAM = 500
K_DIFF = A_PARAM / (B_PARAM * np.pi) ** 2

# ============================================================
# Exact solution and source term
# ============================================================
def u_exact(t, x):
    return np.exp(-A_PARAM * t) * np.sin(B_PARAM * np.pi * x)

def f_source(t, x):
    """f = u_t - k*u_xx  (should be 0 for this problem)."""
    return np.zeros_like(t)

# ============================================================
# Sampler (matches original TF code)
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
# Model 1: Plain MLP (NN)
# ============================================================
def apply_nn(params, tx):
    """tx: (2,) array [t_norm, x_norm] -> scalar u."""
    H = tx.reshape(1, -1)
    for (W, b) in params[:-1]:
        H = jnp.tanh(H @ W + b)
    W, b = params[-1]
    return (H @ W + b)[0, 0]

# ============================================================
# Model 2: Fourier Feature (FF)
# ============================================================
def apply_ff(params, W_ff, tx):
    """tx: (2,) -> scalar u.  W_ff is frozen Fourier matrix."""
    H = tx.reshape(1, -1)
    H = jnp.concatenate([jnp.sin(H @ W_ff), jnp.cos(H @ W_ff)], axis=1)
    for (W, b) in params[:-1]:
        H = jnp.tanh(H @ W + b)
    W, b = params[-1]
    return (H @ W + b)[0, 0]

# ============================================================
# Model 3: Spatio-Temporal Fourier Feature (ST_FF)
# ============================================================
def apply_st_ff(params, W_t, W_x, tx):
    """tx: (2,) -> scalar u.  W_t, W_x are frozen."""
    t_val = tx[0:1].reshape(1, 1)
    x_val = tx[1:2].reshape(1, 1)

    H_t = jnp.concatenate([jnp.sin(t_val @ W_t), jnp.cos(t_val @ W_t)], axis=1)
    H_x = jnp.concatenate([jnp.sin(x_val @ W_x), jnp.cos(x_val @ W_x)], axis=1)

    for (W, b) in params[:-1]:
        H_t = jnp.tanh(H_t @ W + b)
        H_x = jnp.tanh(H_x @ W + b)

    H = H_t * H_x  # point-wise multiplication
    W_last, b_last = params[-1]
    return (H @ W_last + b_last)[0, 0]

# ============================================================
# PDE residual via automatic differentiation
# ============================================================
def make_residual_fn(apply_fn, k, sigma_t, sigma_x):
    """Returns a function: (params, *frozen, t_norm, x_norm) -> residual scalar."""
    def residual_single(params, *frozen_and_tx):
        *frozen, t_norm, x_norm = frozen_and_tx
        tx = jnp.array([t_norm, x_norm])

        def u_of_tx(t_n, x_n):
            return apply_fn(params, *frozen, jnp.array([t_n, x_n]))

        u_t = grad(u_of_tx, argnums=0)(t_norm, x_norm) / sigma_t
        u_x = grad(u_of_tx, argnums=1)(t_norm, x_norm)
        u_xx = grad(lambda xn: grad(u_of_tx, argnums=1)(t_norm, xn))(x_norm) / (sigma_x ** 2)
        return u_t - k * u_xx

    return residual_single

# ============================================================
# Loss functions
# ============================================================
def make_loss_fns(apply_fn, residual_fn, n_frozen):
    """Build loss_ic, loss_bc, loss_res, loss_total for a given model."""

    def u_pred_single(params, *frozen_and_tx):
        *frozen, t_n, x_n = frozen_and_tx
        return apply_fn(params, *frozen, jnp.array([t_n, x_n]))

    u_pred_batch = vmap(u_pred_single, in_axes=(None,) + (None,) * n_frozen + (0, 0))
    res_batch = vmap(residual_fn, in_axes=(None,) + (None,) * n_frozen + (0, 0))

    def loss_ic(params, *frozen, t_ic, x_ic, u_ic):
        u_p = u_pred_batch(params, *frozen, t_ic, x_ic)
        return jnp.mean((u_p - u_ic) ** 2)

    def loss_bc(params, *frozen, t_bc1, x_bc1, t_bc2, x_bc2):
        u_bc1 = u_pred_batch(params, *frozen, t_bc1, x_bc1)
        u_bc2 = u_pred_batch(params, *frozen, t_bc2, x_bc2)
        return jnp.mean(u_bc1 ** 2) + jnp.mean(u_bc2 ** 2)

    def loss_res(params, *frozen, t_r, x_r):
        r = res_batch(params, *frozen, t_r, x_r)
        return jnp.mean(r ** 2)

    def loss_total(params, *frozen, t_ic, x_ic, u_ic, t_bc1, x_bc1, t_bc2, x_bc2, t_r, x_r):
        l_ic = loss_ic(params, *frozen, t_ic=t_ic, x_ic=x_ic, u_ic=u_ic)
        l_bc = loss_bc(params, *frozen, t_bc1=t_bc1, x_bc1=x_bc1, t_bc2=t_bc2, x_bc2=x_bc2)
        l_r = loss_res(params, *frozen, t_r=t_r, x_r=x_r)
        return l_ic + l_bc + l_r, (l_ic, l_bc, l_r)

    return loss_total, u_pred_batch, res_batch

# ============================================================
# NTK computation
# ============================================================
def compute_ntk(apply_fn_single, params, frozen_args, t_pts, x_pts):
    """Compute the NTK matrix K = J @ J^T for u-predictions at given points."""
    flat_params, unravel = ravel_pytree(params)

    def f_flat(fp):
        p = unravel(fp)
        def single(t_n, x_n):
            return apply_fn_single(p, *frozen_args, jnp.array([t_n, x_n]))
        return vmap(single)(t_pts, x_pts)

    J = jacrev(f_flat)(flat_params)
    K = J @ J.T
    return K


def compute_ntk_residual(residual_fn_single, params, frozen_args, t_pts, x_pts):
    """NTK for residual outputs."""
    flat_params, unravel = ravel_pytree(params)

    def f_flat(fp):
        p = unravel(fp)
        def single(t_n, x_n):
            return residual_fn_single(p, *frozen_args, t_n, x_n)
        return vmap(single)(t_pts, x_pts)

    J = jacrev(f_flat)(flat_params)
    K = J @ J.T
    return K

# ============================================================
# Training one model
# ============================================================
def train_model(model_name, apply_fn, params, frozen_args, n_frozen,
                mu_X, sigma_X, ics_sampler, bcs_sampler, res_sampler,
                X_star, u_star, nn_test,
                n_iter=40000, batch_size=128, log_every=100, ntk_every=5000,
                ntk_n_pts=64, seed=42):
    """
    Train a single PINN model and return all logs.
    """
    print(f"\n{'='*70}")
    print(f"  Training model: {model_name}")
    print(f"{'='*70}")

    sigma_t = sigma_X[0]
    sigma_x = sigma_X[1]
    k = K_DIFF

    residual_fn = make_residual_fn(apply_fn, k, float(sigma_t), float(sigma_x))
    loss_total_fn, u_pred_batch_fn, res_batch_fn = make_loss_fns(
        apply_fn, residual_fn, n_frozen
    )

    flat0, _ = ravel_pytree(params)
    n_params = flat0.shape[0]
    print(f"  Trainable parameters: {n_params}")

    lr_schedule = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=1000,
        decay_rate=0.9,
        staircase=False,
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state,
                   t_ic, x_ic, u_ic,
                   t_bc1, x_bc1, t_bc2, x_bc2,
                   t_r, x_r):
        (loss_val, (l_ic, l_bc, l_r)), grads = jax.value_and_grad(
            lambda p: loss_total_fn(p, *frozen_args,
                                    t_ic=t_ic, x_ic=x_ic, u_ic=u_ic,
                                    t_bc1=t_bc1, x_bc1=x_bc1,
                                    t_bc2=t_bc2, x_bc2=x_bc2,
                                    t_r=t_r, x_r=x_r),
            has_aux=True
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, l_ic, l_bc, l_r

    # Prediction helper
    def predict_u(params, X_pts):
        X_norm = (X_pts - mu_X) / sigma_X
        t_n = jnp.array(X_norm[:, 0])
        x_n = jnp.array(X_norm[:, 1])
        return np.array(u_pred_batch_fn(params, *frozen_args, t_n, x_n))

    # Test data
    t_test = np.linspace(0, 1, nn_test)
    x_test = np.linspace(0, 1, nn_test)
    tt, xx = np.meshgrid(t_test, x_test)
    X_test = np.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
    u_test = u_exact(X_test[:, 0], X_test[:, 1])

    # NTK sample points (small for memory)
    rng_ntk = np.random.RandomState(seed)
    idx_ntk = rng_ntk.choice(X_test.shape[0], ntk_n_pts, replace=False)
    X_ntk = X_test[idx_ntk]
    X_ntk_norm = (X_ntk - mu_X) / sigma_X
    t_ntk = jnp.array(X_ntk_norm[:, 0])
    x_ntk = jnp.array(X_ntk_norm[:, 1])

    # Logs
    loss_ic_log, loss_bc_log, loss_res_log, l2_error_log = [], [], [], []
    iters_log = []
    ntk_K_log = []
    ntk_iters_log = []
    best_l2 = 1.0

    rng_train = np.random.RandomState(seed)
    start_time = time.time()

    for it in range(n_iter):
        # Sample mini-batches
        X_ic, u_ic_batch = ics_sampler.sample(batch_size, rng_train)
        X_bc1, _ = bcs_sampler[0].sample(batch_size, rng_train)
        X_bc2, _ = bcs_sampler[1].sample(batch_size, rng_train)
        X_res, _ = res_sampler.sample(batch_size, rng_train)

        # Normalize
        X_ic_n = (X_ic - mu_X) / sigma_X
        X_bc1_n = (X_bc1 - mu_X) / sigma_X
        X_bc2_n = (X_bc2 - mu_X) / sigma_X
        X_res_n = (X_res - mu_X) / sigma_X

        params, opt_state, loss_val, l_ic, l_bc, l_r = train_step(
            params, opt_state,
            jnp.array(X_ic_n[:, 0]), jnp.array(X_ic_n[:, 1]), jnp.array(u_ic_batch.flatten()),
            jnp.array(X_bc1_n[:, 0]), jnp.array(X_bc1_n[:, 1]),
            jnp.array(X_bc2_n[:, 0]), jnp.array(X_bc2_n[:, 1]),
            jnp.array(X_res_n[:, 0]), jnp.array(X_res_n[:, 1]),
        )

        if it % log_every == 0:
            u_pred = predict_u(params, X_test)
            l2_err = np.linalg.norm(u_test - u_pred) / np.linalg.norm(u_test)
            loss_ic_log.append(float(l_ic))
            loss_bc_log.append(float(l_bc))
            loss_res_log.append(float(l_r))
            l2_error_log.append(float(l2_err))
            iters_log.append(it)
            if l2_err < best_l2:
                best_l2 = l2_err

            elapsed = time.time() - start_time
            print(f"  It: {it:5d}, Loss: {float(loss_val):.3e}, "
                  f"L_ic: {float(l_ic):.3e}, L_bc: {float(l_bc):.3e}, "
                  f"L_res: {float(l_r):.3e}, L2: {l2_err:.3e}, Time: {elapsed:.1f}s")

        if it % ntk_every == 0:
            K_uu = compute_ntk(apply_fn, params, frozen_args, t_ntk, x_ntk)
            K_rr = compute_ntk_residual(residual_fn, params, frozen_args, t_ntk, x_ntk)
            ntk_K_log.append({
                'K_uu': np.array(K_uu),
                'K_rr': np.array(K_rr),
            })
            ntk_iters_log.append(it)
            print(f"  [NTK computed at iter {it}]")

    total_time = time.time() - start_time

    # Final prediction
    u_pred_final = predict_u(params, X_test)
    final_l2 = np.linalg.norm(u_test - u_pred_final) / np.linalg.norm(u_test)

    print(f"\n  {model_name} training complete.")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Best L2 error:    {best_l2:.3e}")
    print(f"  Final L2 error:   {final_l2:.3e}")
    print(f"  Parameters:       {n_params}")

    results = {
        'model_name': model_name,
        'n_params': n_params,
        'total_time': total_time,
        'best_l2': best_l2,
        'final_l2': final_l2,
        'loss_ic_log': loss_ic_log,
        'loss_bc_log': loss_bc_log,
        'loss_res_log': loss_res_log,
        'l2_error_log': l2_error_log,
        'iters_log': iters_log,
        'ntk_K_log': ntk_K_log,
        'ntk_iters_log': ntk_iters_log,
        'u_pred_final': u_pred_final,
        'u_test': u_test,
        'X_test': X_test,
        'tt': tt, 'xx': xx,
        'nn_test': nn_test,
        'params': params,
    }
    return results

# ============================================================
# Save data for one model
# ============================================================
def save_model_data(results):
    name = results['model_name']
    iters = np.array(results['iters_log'])

    # Loss history
    loss_data = np.column_stack([
        iters,
        results['loss_ic_log'],
        results['loss_bc_log'],
        results['loss_res_log'],
        results['l2_error_log'],
    ])
    np.savetxt(os.path.join(DATA_DIR, f'loss_history_{name}.txt'), loss_data,
               header='iteration  loss_ic  loss_bc  loss_res  l2_error', fmt='%.6e')

    # Prediction data
    X = results['X_test']
    pred_data = np.column_stack([
        X[:, 0], X[:, 1],
        results['u_test'],
        results['u_pred_final'],
        np.abs(results['u_test'] - results['u_pred_final']),
    ])
    np.savetxt(os.path.join(DATA_DIR, f'prediction_{name}.txt'), pred_data,
               header='t  x  u_exact  u_pred  abs_error', fmt='%.6e')

    # NTK data
    ntk_iters = results['ntk_iters_log']
    ntk_logs = results['ntk_K_log']

    eig_Kuu_all, eig_Krr_all = [], []
    K_full_list = []
    for i, nt in enumerate(ntk_logs):
        K_uu = nt['K_uu']
        K_rr = nt['K_rr']
        eig_uu = np.sort(np.real(np.linalg.eigvalsh(K_uu)))[::-1]
        eig_rr = np.sort(np.real(np.linalg.eigvalsh(K_rr)))[::-1]
        eig_Kuu_all.append(eig_uu)
        eig_Krr_all.append(eig_rr)

        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Kuu_{name}_iter{ntk_iters[i]}.txt'),
                   eig_uu, header=f'eigenvalues_Kuu_iter_{ntk_iters[i]}', fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Krr_{name}_iter{ntk_iters[i]}.txt'),
                   eig_rr, header=f'eigenvalues_Krr_iter_{ntk_iters[i]}', fmt='%.6e')

        K_full_list.append(K_uu)

    # NTK change
    if len(K_full_list) > 0:
        K0 = K_full_list[0]
        K0_norm = np.linalg.norm(K0)
        ntk_change = [np.linalg.norm(K - K0) / max(K0_norm, 1e-30) for K in K_full_list]
        np.savetxt(os.path.join(DATA_DIR, f'ntk_change_{name}.txt'),
                   np.column_stack([ntk_iters, ntk_change]),
                   header='iteration  ntk_relative_change', fmt='%.6e')

    # Model params
    with open(os.path.join(CKPT_DIR, f'params_{name}.pkl'), 'wb') as f:
        pickle.dump(results['params'], f)

    results['eig_Kuu_all'] = eig_Kuu_all
    results['eig_Krr_all'] = eig_Krr_all
    results['ntk_change'] = ntk_change if len(K_full_list) > 0 else []

# ============================================================
# Plotting helpers
# ============================================================
def _label_subplot(ax, label, x=-0.10, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')


def _set_axis_style(ax, xlabel=None, ylabel=None, fs=14):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fs, fontweight='bold')
    ax.tick_params(labelsize=12, width=1.5, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

# ============================================================
# Plot: single model — prediction (Reference / Predicted / Error)
# ============================================================
def plot_prediction_single(results):
    name = results['model_name']
    tt, xx = results['tt'], results['xx']
    nn = results['nn_test']

    u_star = results['u_test'].reshape(nn, nn)
    u_pred = results['u_pred_final'].reshape(nn, nn)
    err = np.abs(u_star - u_pred)

    vmin_u = u_star.min()
    vmax_u = u_star.max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    im0 = axes[0].pcolormesh(tt, xx, u_star, cmap='jet', shading='auto', vmin=vmin_u, vmax=vmax_u)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title('Exact $u(t,x)$', fontsize=16, fontweight='bold')
    _set_axis_style(axes[0], '$t$', '$x$')
    _label_subplot(axes[0], '(a)')

    im1 = axes[1].pcolormesh(tt, xx, u_pred, cmap='jet', shading='auto', vmin=vmin_u, vmax=vmax_u)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f'Predicted ({name})', fontsize=16, fontweight='bold')
    _set_axis_style(axes[1], '$t$', '$x$')
    _label_subplot(axes[1], '(b)')

    im2 = axes[2].pcolormesh(tt, xx, err, cmap='jet', shading='auto')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title('Absolute Error', fontsize=16, fontweight='bold')
    _set_axis_style(axes[2], '$t$', '$x$')
    _label_subplot(axes[2], '(c)')

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
    ax.semilogy(iters, results['loss_bc_log'], lw=2, label=r'$\mathcal{L}_{bc}$')
    ax.semilogy(iters, results['loss_ic_log'], lw=2, label=r'$\mathcal{L}_{ic}$')
    ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black')
    _set_axis_style(ax, 'Iterations', 'Loss')
    ax.set_title(f'{name} — Loss Components', fontsize=16, fontweight='bold')
    _label_subplot(ax, '(a)')

    ax = axes[1]
    ax.semilogy(iters, results['l2_error_log'], lw=2, color='tab:red')
    _set_axis_style(ax, 'Iterations', 'Relative $L^2$ Error')
    ax.set_title(f'{name} — $L^2$ Error', fontsize=16, fontweight='bold')
    _label_subplot(ax, '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Plot: single model — NTK eigenvalues
# ============================================================
def plot_ntk_single(results):
    name = results['model_name']
    ntk_iters = results['ntk_iters_log']
    eig_Kuu = results.get('eig_Kuu_all', [])
    eig_Krr = results.get('eig_Krr_all', [])

    if len(eig_Kuu) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(eig_Kuu)))

    ax = axes[0]
    for i, (eig, it_n) in enumerate(zip(eig_Kuu, ntk_iters)):
        ax.loglog(np.arange(1, len(eig) + 1), np.clip(eig, 1e-30, None),
                  lw=2, color=colors[i], label=f'iter {it_n}')
    ax.set_title(f'{name} — $K_{{uu}}$ eigenvalues', fontsize=16, fontweight='bold')
    _set_axis_style(ax, 'Index', 'Eigenvalue')
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    _label_subplot(ax, '(a)')

    ax = axes[1]
    for i, (eig, it_n) in enumerate(zip(eig_Krr, ntk_iters)):
        ax.loglog(np.arange(1, len(eig) + 1), np.clip(eig, 1e-30, None),
                  lw=2, color=colors[i], label=f'iter {it_n}')
    ax.set_title(f'{name} — $K_{{rr}}$ eigenvalues', fontsize=16, fontweight='bold')
    _set_axis_style(ax, 'Index', 'Eigenvalue')
    ax.legend(fontsize=11, frameon=True, fancybox=False, edgecolor='black')
    _label_subplot(ax, '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_ntk_eigenvalues.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Plot: single model — NTK change
# ============================================================
def plot_ntk_change_single(results):
    name = results['model_name']
    ntk_change = results.get('ntk_change', [])
    ntk_iters = results['ntk_iters_log']
    if len(ntk_change) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ntk_iters, ntk_change, lw=2.5, marker='o', markersize=5, color='tab:blue')
    _set_axis_style(ax, 'Iterations', r'$\|K - K_0\| / \|K_0\|$')
    ax.set_title(f'{name} — NTK Relative Change', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_ntk_change.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison plots — three models
# ============================================================
MODEL_COLORS = {'NN': 'tab:blue', 'FF': 'tab:orange', 'ST_FF': 'tab:green'}
MODEL_LINESTYLES = {'NN': '-', 'FF': '--', 'ST_FF': '-.'}
MODEL_MARKERS = {'NN': 'o', 'FF': 's', 'ST_FF': '^'}

def plot_comparison_prediction(all_results):
    """3x3 subplot matrix: rows=models, cols=exact/predicted/error."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    nn = all_results[0]['nn_test']
    tt, xx = all_results[0]['tt'], all_results[0]['xx']
    u_star = all_results[0]['u_test'].reshape(nn, nn)
    vmin_u, vmax_u = u_star.min(), u_star.max()

    for row, res in enumerate(all_results):
        name = res['model_name']
        u_pred = res['u_pred_final'].reshape(nn, nn)
        err = np.abs(u_star - u_pred)
        err_max = max(err.max(), 1e-10)

        im0 = axes[row, 0].pcolormesh(tt, xx, u_star, cmap='jet', shading='auto',
                                        vmin=vmin_u, vmax=vmax_u)
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)
        _set_axis_style(axes[row, 0], '$t$', '$x$')
        if row == 0:
            axes[row, 0].set_title('Exact', fontsize=16, fontweight='bold')
        axes[row, 0].set_ylabel(f'{name}\n$x$', fontsize=14, fontweight='bold')

        im1 = axes[row, 1].pcolormesh(tt, xx, u_pred, cmap='jet', shading='auto',
                                        vmin=vmin_u, vmax=vmax_u)
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)
        _set_axis_style(axes[row, 1], '$t$')
        if row == 0:
            axes[row, 1].set_title('Predicted', fontsize=16, fontweight='bold')

        im2 = axes[row, 2].pcolormesh(tt, xx, err, cmap='jet', shading='auto')
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        _set_axis_style(axes[row, 2], '$t$')
        if row == 0:
            axes[row, 2].set_title('Absolute Error', fontsize=16, fontweight='bold')

    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    for idx, ax in enumerate(axes.flat):
        _label_subplot(ax, labels[idx])

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_l2(all_results):
    """L2 error convergence — all models overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in all_results:
        name = res['model_name']
        ax.semilogy(res['iters_log'], res['l2_error_log'],
                    lw=2.5, color=MODEL_COLORS[name],
                    linestyle=MODEL_LINESTYLES[name],
                    label=f'{name}  (best L2 = {res["best_l2"]:.2e})')
    ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    _set_axis_style(ax, 'Iterations', 'Relative $L^2$ Error', fs=16)
    ax.set_title('$L^2$ Error Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_l2_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_loss_res(all_results):
    """PDE residual loss comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in all_results:
        name = res['model_name']
        ax.semilogy(res['iters_log'], res['loss_res_log'],
                    lw=2.5, color=MODEL_COLORS[name],
                    linestyle=MODEL_LINESTYLES[name],
                    label=name)
    ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black')
    _set_axis_style(ax, 'Iterations', r'$\mathcal{L}_{r}$ (PDE Residual)', fs=16)
    ax.set_title('PDE Residual Loss Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_loss_res.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_loss_all(all_results):
    """All loss components comparison — 3 subplots (IC / BC / Res)."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    titles = [r'$\mathcal{L}_{ic}$', r'$\mathcal{L}_{bc}$', r'$\mathcal{L}_{r}$']
    keys = ['loss_ic_log', 'loss_bc_log', 'loss_res_log']

    for col, (ax, title, key) in enumerate(zip(axes, titles, keys)):
        for res in all_results:
            name = res['model_name']
            ax.semilogy(res['iters_log'], res[key],
                        lw=2.5, color=MODEL_COLORS[name],
                        linestyle=MODEL_LINESTYLES[name],
                        label=name)
        ax.set_title(title, fontsize=18, fontweight='bold')
        _set_axis_style(ax, 'Iterations', 'Loss')
        _label_subplot(ax, f'({"abc"[col]})')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               fontsize=14, frameon=True, fancybox=False, edgecolor='black',
               bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_loss_all.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_ntk_eigenvalues(all_results):
    """NTK K_rr eigenvalue spectra — initial and final, for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    titles = ['Initial ($n=0$)', 'Final']

    for col, (ax, title) in enumerate(zip(axes, titles)):
        for res in all_results:
            name = res['model_name']
            eig_list = res.get('eig_Krr_all', [])
            if len(eig_list) == 0:
                continue
            idx = 0 if col == 0 else -1
            eig = eig_list[idx]
            it_n = res['ntk_iters_log'][idx]
            ax.loglog(np.arange(1, len(eig) + 1), np.clip(eig, 1e-30, None),
                      lw=2.5, color=MODEL_COLORS[name],
                      linestyle=MODEL_LINESTYLES[name],
                      label=f'{name} (iter {it_n})')
        ax.set_title(f'$K_{{rr}}$ Eigenvalues — {title}', fontsize=16, fontweight='bold')
        _set_axis_style(ax, 'Index', 'Eigenvalue')
        ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor='black')
        _label_subplot(ax, f'({"ab"[col]})')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_ntk_eigenvalues.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_bar_chart(all_results):
    """Bar chart comparing final L2, best L2, and training time."""
    names = [r['model_name'] for r in all_results]
    final_l2 = [r['final_l2'] for r in all_results]
    best_l2 = [r['best_l2'] for r in all_results]
    train_time = [r['total_time'] for r in all_results]
    n_params = [r['n_params'] for r in all_results]

    x = np.arange(len(names))
    width = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # L2 errors
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, final_l2, width, label='Final $L^2$',
                   color=[MODEL_COLORS[n] for n in names], alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, best_l2, width, label='Best $L^2$',
                   color=[MODEL_COLORS[n] for n in names], alpha=1.0, edgecolor='black', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14, fontweight='bold')
    _set_axis_style(ax, ylabel='Relative $L^2$ Error')
    ax.legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    ax.set_title('$L^2$ Error Comparison', fontsize=16, fontweight='bold')
    for bar, val in zip(bars1, final_l2):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.3, f'{val:.1e}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    _label_subplot(ax, '(a)')

    # Training time
    ax = axes[1]
    ax.bar(x, train_time, width * 1.5,
           color=[MODEL_COLORS[n] for n in names], edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14, fontweight='bold')
    _set_axis_style(ax, ylabel='Training Time (s)')
    ax.set_title('Training Time', fontsize=16, fontweight='bold')
    for i, (xi, t) in enumerate(zip(x, train_time)):
        ax.text(xi, t + max(train_time) * 0.02, f'{t:.0f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    _label_subplot(ax, '(b)')

    # Parameters
    ax = axes[2]
    ax.bar(x, n_params, width * 1.5,
           color=[MODEL_COLORS[n] for n in names], edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14, fontweight='bold')
    _set_axis_style(ax, ylabel='Number of Parameters')
    ax.set_title('Trainable Parameters', fontsize=16, fontweight='bold')
    for i, (xi, p) in enumerate(zip(x, n_params)):
        ax.text(xi, p + max(n_params) * 0.02, f'{p}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    _label_subplot(ax, '(c)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_bar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_ntk_change(all_results):
    """NTK relative change over training — all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in all_results:
        name = res['model_name']
        ntk_change = res.get('ntk_change', [])
        ntk_iters = res['ntk_iters_log']
        if len(ntk_change) > 0:
            ax.plot(ntk_iters, ntk_change, lw=2.5,
                    color=MODEL_COLORS[name],
                    linestyle=MODEL_LINESTYLES[name],
                    marker=MODEL_MARKERS[name], markersize=5,
                    label=name)
    ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black')
    _set_axis_style(ax, 'Iterations', r'$\|K - K_0\| / \|K_0\|$', fs=16)
    ax.set_title('NTK Relative Change Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_ntk_change.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_comparison_table(all_results):
    """Save summary comparison table."""
    lines = []
    lines.append("=" * 80)
    lines.append("COMPARISON TABLE — 1D Heat Equation (Multi-scale PINN)")
    lines.append("=" * 80)
    lines.append(f"{'Model':<10} {'Params':>10} {'Best L2':>14} {'Final L2':>14} {'Time (s)':>12}")
    lines.append("-" * 80)
    for res in all_results:
        lines.append(f"{res['model_name']:<10} {res['n_params']:>10d} "
                     f"{res['best_l2']:>14.4e} {res['final_l2']:>14.4e} "
                     f"{res['total_time']:>12.1f}")
    lines.append("=" * 80)
    table_str = "\n".join(lines)
    print("\n" + table_str)

    with open(os.path.join(DATA_DIR, 'comparison_table.txt'), 'w') as f:
        f.write(table_str + "\n")

    # Also save as machine-readable CSV-like
    header = 'model  n_params  best_l2  final_l2  time_s'
    rows = []
    for res in all_results:
        rows.append(f"{res['model_name']}  {res['n_params']}  "
                    f"{res['best_l2']:.6e}  {res['final_l2']:.6e}  {res['total_time']:.1f}")
    with open(os.path.join(DATA_DIR, 'comparison_data.txt'), 'w') as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Multi-scale Fourier Feature PINN — 1D Heat Equation")
    print(f"a={A_PARAM}, b={B_PARAM}, k={K_DIFF:.6e}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print("=" * 70)

    # Check JAX backend
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")

    # --------------------------------------------------------
    # Problem setup (shared by all models)
    # --------------------------------------------------------
    ics_coords = np.array([[0.0, 0.0],
                           [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0],
                           [1.0, 0.0]])
    bc2_coords = np.array([[0.0, 1.0],
                           [1.0, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                           [1.0, 1.0]])

    def u_ic_func(x):
        return u_exact(x[:, 0:1], x[:, 1:2])

    def u_bc_func(x):
        return u_exact(x[:, 0:1], x[:, 1:2])

    def f_func(x):
        return f_source(x[:, 0:1], x[:, 1:2])

    ics_sampler = Sampler(2, ics_coords, u_ic_func, name='IC')
    bc1_sampler = Sampler(2, bc1_coords, u_bc_func, name='BC1')
    bc2_sampler = Sampler(2, bc2_coords, u_bc_func, name='BC2')
    res_sampler = Sampler(2, dom_coords, f_func, name='Residual')

    bcs_sampler = [bc1_sampler, bc2_sampler]

    mu_X, sigma_X = compute_norm_stats(res_sampler)
    print(f"Normalization: mu={mu_X}, sigma={sigma_X}")

    nn_test = 100
    t_test = np.linspace(0, 1, nn_test)
    x_test = np.linspace(0, 1, nn_test)
    tt, xx = np.meshgrid(t_test, x_test)
    X_star = np.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
    u_star = u_exact(X_star[:, 0], X_star[:, 1])

    SIGMA_FF = 500
    N_ITER = 40000
    BATCH_SIZE = 128
    SEED = 1234

    # --------------------------------------------------------
    # Model 1: Plain MLP (NN)
    # --------------------------------------------------------
    key = random.PRNGKey(SEED)
    layers_nn = [2, 100, 100, 100, 1]
    params_nn = init_mlp(layers_nn, key)

    results_nn = train_model(
        model_name='NN',
        apply_fn=apply_nn,
        params=params_nn,
        frozen_args=(),
        n_frozen=0,
        mu_X=mu_X, sigma_X=sigma_X,
        ics_sampler=ics_sampler,
        bcs_sampler=bcs_sampler,
        res_sampler=res_sampler,
        X_star=X_star, u_star=u_star, nn_test=nn_test,
        n_iter=N_ITER, batch_size=BATCH_SIZE,
        seed=SEED,
    )
    save_model_data(results_nn)
    plot_prediction_single(results_nn)
    plot_loss_single(results_nn)
    plot_ntk_single(results_nn)
    plot_ntk_change_single(results_nn)

    # --------------------------------------------------------
    # Model 2: Fourier Feature (FF)
    # --------------------------------------------------------
    key = random.PRNGKey(SEED)
    layers_ff = [100, 100, 100, 1]
    params_ff = init_mlp(layers_ff, key)
    key_ff = random.PRNGKey(SEED + 1)
    W_ff = random.normal(key_ff, (2, layers_ff[0] // 2)) * SIGMA_FF

    results_ff = train_model(
        model_name='FF',
        apply_fn=apply_ff,
        params=params_ff,
        frozen_args=(W_ff,),
        n_frozen=1,
        mu_X=mu_X, sigma_X=sigma_X,
        ics_sampler=ics_sampler,
        bcs_sampler=bcs_sampler,
        res_sampler=res_sampler,
        X_star=X_star, u_star=u_star, nn_test=nn_test,
        n_iter=N_ITER, batch_size=BATCH_SIZE,
        seed=SEED,
    )
    save_model_data(results_ff)
    plot_prediction_single(results_ff)
    plot_loss_single(results_ff)
    plot_ntk_single(results_ff)
    plot_ntk_change_single(results_ff)

    # --------------------------------------------------------
    # Model 3: Spatio-Temporal FF (ST_FF)
    # --------------------------------------------------------
    key = random.PRNGKey(SEED)
    layers_stff = [100, 100, 100, 1]
    params_stff = init_mlp(layers_stff, key)
    key_st = random.PRNGKey(SEED + 2)
    k1, k2 = random.split(key_st)
    W_t = random.normal(k1, (1, layers_stff[0] // 2)) * 1.0
    W_x = random.normal(k2, (1, layers_stff[0] // 2)) * SIGMA_FF

    results_stff = train_model(
        model_name='ST_FF',
        apply_fn=apply_st_ff,
        params=params_stff,
        frozen_args=(W_t, W_x),
        n_frozen=2,
        mu_X=mu_X, sigma_X=sigma_X,
        ics_sampler=ics_sampler,
        bcs_sampler=bcs_sampler,
        res_sampler=res_sampler,
        X_star=X_star, u_star=u_star, nn_test=nn_test,
        n_iter=N_ITER, batch_size=BATCH_SIZE,
        seed=SEED,
    )
    save_model_data(results_stff)
    plot_prediction_single(results_stff)
    plot_loss_single(results_stff)
    plot_ntk_single(results_stff)
    plot_ntk_change_single(results_stff)

    # --------------------------------------------------------
    # Comparison plots and table
    # --------------------------------------------------------
    all_results = [results_nn, results_ff, results_stff]

    plot_comparison_prediction(all_results)
    plot_comparison_l2(all_results)
    plot_comparison_loss_res(all_results)
    plot_comparison_loss_all(all_results)
    plot_comparison_ntk_eigenvalues(all_results)
    plot_comparison_ntk_change(all_results)
    plot_comparison_bar_chart(all_results)
    save_comparison_table(all_results)

    # Save L2 comparison data
    max_len = max(len(r['iters_log']) for r in all_results)
    for res in all_results:
        while len(res['l2_error_log']) < max_len:
            res['l2_error_log'].append(res['l2_error_log'][-1])
        while len(res['iters_log']) < max_len:
            res['iters_log'].append(res['iters_log'][-1])

    l2_data = np.column_stack([
        all_results[0]['iters_log'],
        all_results[0]['l2_error_log'],
        all_results[1]['l2_error_log'],
        all_results[2]['l2_error_log'],
    ])
    np.savetxt(os.path.join(DATA_DIR, 'l2_error_comparison.txt'), l2_data,
               header='iteration  l2_NN  l2_FF  l2_ST_FF', fmt='%.6e')

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
