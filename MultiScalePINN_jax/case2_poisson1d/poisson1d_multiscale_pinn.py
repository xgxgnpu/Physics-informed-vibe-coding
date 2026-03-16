"""
Multi-scale Fourier Feature PINN for 1D Poisson Equation (JAX)
===============================================================
Reproduces the Poisson1D case from:
  Wang, Wang & Perdikaris, "On the eigenvector bias of Fourier feature
  networks", CMAME 384, 113938 (2021).

Three model variants compared under identical conditions:
  1) NN   — Plain MLP
  2) FF   — Fourier Feature network
  3) mFF  — Multi-scale Fourier Feature network (concatenation merge)

Exact solution: u(x) = sin(a*pi*x) + 0.1*sin(b*pi*x),  a=2, b=50
PDE:  u_xx(x) = f(x)
Domain: x in [0, 1],  BC: u(0)=0, u(1)=0

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
A_FREQ = 2
B_FREQ = 50
SIGMA_FF = 10

def u_exact(x):
    return np.sin(np.pi * A_FREQ * x) + 0.1 * np.sin(np.pi * B_FREQ * x)

def u_xx_exact(x):
    return (-(np.pi * A_FREQ) ** 2 * np.sin(np.pi * A_FREQ * x)
            - 0.1 * (np.pi * B_FREQ) ** 2 * np.sin(np.pi * B_FREQ * x))

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
# Normalization
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

def init_mlp_mff(shared_layers, final_in, final_out, key):
    """Initialize MLP for mFF: shared hidden layers + final layer with doubled input."""
    params_shared = []
    for i in range(len(shared_layers) - 1):
        key, subkey = random.split(key)
        W, b = xavier_init(subkey, shared_layers[i], shared_layers[i + 1])
        params_shared.append((W, b))
    key, subkey = random.split(key)
    W_final, b_final = xavier_init(subkey, final_in, final_out)
    return params_shared, (W_final, b_final)

# ============================================================
# Model 1: Plain MLP (NN)
# ============================================================
def apply_nn(params, x_norm):
    """x_norm: scalar -> scalar u."""
    H = x_norm.reshape(1, 1)
    for (W, b) in params[:-1]:
        H = jnp.tanh(H @ W + b)
    W, b = params[-1]
    return (H @ W + b)[0, 0]

# ============================================================
# Model 2: Fourier Feature (FF)
# ============================================================
def apply_ff(params, W_ff, x_norm):
    """x_norm: scalar -> scalar u. W_ff is frozen."""
    H = x_norm.reshape(1, 1)
    H = jnp.concatenate([jnp.sin(H @ W_ff), jnp.cos(H @ W_ff)], axis=1)
    for (W, b) in params[:-1]:
        H = jnp.tanh(H @ W + b)
    W, b = params[-1]
    return (H @ W + b)[0, 0]

# ============================================================
# Model 3: Multi-scale Fourier Feature (mFF) — concatenation
# ============================================================
def apply_mff(params_shared, W_final_tuple, W1, W2, x_norm):
    """x_norm: scalar -> scalar u. W1, W2 frozen. Two branches share weights, concat merge."""
    H_in = x_norm.reshape(1, 1)

    H1 = jnp.concatenate([jnp.sin(H_in @ W1), jnp.cos(H_in @ W1)], axis=1)
    H2 = jnp.concatenate([jnp.sin(H_in @ W2), jnp.cos(H_in @ W2)], axis=1)

    for (W, b) in params_shared:
        H1 = jnp.tanh(H1 @ W + b)
        H2 = jnp.tanh(H2 @ W + b)

    H = jnp.concatenate([H1, H2], axis=1)
    W_f, b_f = W_final_tuple
    return (H @ W_f + b_f)[0, 0]

# ============================================================
# PDE residual: u_xx
# ============================================================
def make_u_xx_fn(apply_fn_single, sigma_x):
    """Returns function: (trainable_args..., frozen_args..., x_norm) -> u_xx in physical space."""
    def u_xx_single(*args_and_x):
        *all_args, x_n = args_and_x
        def u_of_x(xn):
            return apply_fn_single(*all_args, xn)
        du = grad(u_of_x)
        d2u = grad(du)
        return d2u(x_n) / (sigma_x ** 2)
    return u_xx_single

# ============================================================
# Build loss functions for each model type
# ============================================================
def build_loss_fns_nn_ff(apply_fn, sigma_x, frozen_args):
    """For NN and FF models. frozen_args captured in closure."""
    u_xx_fn = make_u_xx_fn(apply_fn, sigma_x)

    def u_pred_single(params, x_n):
        return apply_fn(params, *frozen_args, x_n)

    def u_xx_single(params, x_n):
        return u_xx_fn(params, *frozen_args, x_n)

    u_pred_batch = vmap(u_pred_single, in_axes=(None, 0))
    u_xx_batch = vmap(u_xx_single, in_axes=(None, 0))

    def loss_bc(params, x_bc1, u_bc1, x_bc2, u_bc2):
        pred1 = u_pred_batch(params, x_bc1)
        pred2 = u_pred_batch(params, x_bc2)
        return jnp.mean((pred1 - u_bc1) ** 2) + jnp.mean((pred2 - u_bc2) ** 2)

    def loss_res(params, x_r, f_r):
        uxx = u_xx_batch(params, x_r)
        return jnp.mean((uxx - f_r) ** 2)

    def loss_total(params, x_bc1, u_bc1, x_bc2, u_bc2, x_r, f_r):
        l_bc = loss_bc(params, x_bc1, u_bc1, x_bc2, u_bc2)
        l_r = loss_res(params, x_r, f_r)
        return l_bc + l_r, (l_bc, l_r)

    return loss_total, u_pred_batch, u_xx_batch


def build_loss_fns_mff(sigma_x):
    """For mFF model (params_shared + W_final as trainable, W1/W2 frozen)."""
    def u_pred_single(params_shared, W_final_tuple, W1, W2, x_n):
        return apply_mff(params_shared, W_final_tuple, W1, W2, x_n)

    def u_xx_single(params_shared, W_final_tuple, W1, W2, x_n):
        def u_of_x(xn):
            return apply_mff(params_shared, W_final_tuple, W1, W2, xn)
        return grad(grad(u_of_x))(x_n) / (sigma_x ** 2)

    u_pred_batch = vmap(u_pred_single, in_axes=(None, None, None, None, 0))
    u_xx_batch = vmap(u_xx_single, in_axes=(None, None, None, None, 0))

    def loss_bc(ps, wf, W1, W2, x_bc1, u_bc1, x_bc2, u_bc2):
        pred1 = u_pred_batch(ps, wf, W1, W2, x_bc1)
        pred2 = u_pred_batch(ps, wf, W1, W2, x_bc2)
        return jnp.mean((pred1 - u_bc1) ** 2) + jnp.mean((pred2 - u_bc2) ** 2)

    def loss_res(ps, wf, W1, W2, x_r, f_r):
        uxx = u_xx_batch(ps, wf, W1, W2, x_r)
        return jnp.mean((uxx - f_r) ** 2)

    def loss_total(ps, wf, W1, W2, x_bc1, u_bc1, x_bc2, u_bc2, x_r, f_r):
        l_bc = loss_bc(ps, wf, W1, W2, x_bc1, u_bc1, x_bc2, u_bc2)
        l_r = loss_res(ps, wf, W1, W2, x_r, f_r)
        return l_bc + l_r, (l_bc, l_r)

    return loss_total, u_pred_batch, u_xx_batch

# ============================================================
# NTK computation
# ============================================================
def compute_ntk_kernel(fn_batch, trainable, frozen_args, x_pts):
    """Compute NTK: K = J @ J^T. fn_batch(trainable, *frozen_args, x_pts) -> (N,)."""
    flat_params, unravel = ravel_pytree(trainable)
    def f_flat(fp):
        p = unravel(fp)
        return fn_batch(p, *frozen_args, x_pts)
    J = jacrev(f_flat)(flat_params)
    return J @ J.T

# ============================================================
# Train one model (NN or FF)
# ============================================================
def train_nn_ff(model_name, apply_fn, params, frozen_args, n_frozen,
                mu_X, sigma_X, bcs_samplers, res_sampler,
                n_iter=40000, batch_size=128, log_every=100,
                ntk_every=5000, ntk_n_pts=64, seed=42,
                lr_decay_steps=1000, lr_decay_rate=0.9):

    print(f"\n{'='*70}")
    print(f"  Training model: {model_name}")
    print(f"{'='*70}")

    sigma_x = float(sigma_X[0])
    loss_total_fn, u_pred_batch, u_xx_batch = build_loss_fns_nn_ff(
        apply_fn, sigma_x, frozen_args)

    flat0, _ = ravel_pytree(params)
    n_params = flat0.shape[0]
    print(f"  Trainable parameters: {n_params}")

    lr_schedule = optax.exponential_decay(
        init_value=1e-3, transition_steps=lr_decay_steps,
        decay_rate=lr_decay_rate, staircase=False)
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state, x_bc1, u_bc1, x_bc2, u_bc2, x_r, f_r):
        (loss_val, (l_bc, l_r)), grads = jax.value_and_grad(
            lambda p: loss_total_fn(p, x_bc1, u_bc1, x_bc2, u_bc2, x_r, f_r),
            has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss_val, l_bc, l_r

    def predict_u(params, X_pts):
        X_norm = (X_pts - mu_X) / sigma_X
        return np.array(u_pred_batch(params, jnp.array(X_norm[:, 0])))

    nn_test = 1000
    X_test = np.linspace(0, 1, nn_test)[:, None]
    u_test = u_exact(X_test[:, 0])

    X_ntk_raw = np.linspace(0, 1, ntk_n_pts)[:, None]
    X_ntk_norm = (X_ntk_raw - mu_X) / sigma_X
    x_ntk = jnp.array(X_ntk_norm[:, 0])

    loss_bc_log, loss_res_log, l2_error_log, iters_log = [], [], [], []
    ntk_K_log, ntk_iters_log = [], []
    best_l2 = 1.0

    rng_train = np.random.RandomState(seed)
    start_time = time.time()

    for it in range(n_iter):
        X_bc1, u_bc1 = bcs_samplers[0].sample(batch_size, rng_train)
        X_bc2, u_bc2 = bcs_samplers[1].sample(batch_size, rng_train)
        X_res, f_res = res_sampler.sample(batch_size, rng_train)

        X_bc1_n = (X_bc1 - mu_X) / sigma_X
        X_bc2_n = (X_bc2 - mu_X) / sigma_X
        X_res_n = (X_res - mu_X) / sigma_X

        params, opt_state, loss_val, l_bc, l_r = train_step(
            params, opt_state,
            jnp.array(X_bc1_n[:, 0]), jnp.array(u_bc1.flatten()),
            jnp.array(X_bc2_n[:, 0]), jnp.array(u_bc2.flatten()),
            jnp.array(X_res_n[:, 0]), jnp.array(f_res.flatten()))

        if it % log_every == 0:
            u_pred = predict_u(params, X_test)
            l2_err = np.linalg.norm(u_test - u_pred) / np.linalg.norm(u_test)
            loss_bc_log.append(float(l_bc))
            loss_res_log.append(float(l_r))
            l2_error_log.append(float(l2_err))
            iters_log.append(it)
            if l2_err < best_l2:
                best_l2 = l2_err
            elapsed = time.time() - start_time
            print(f"  It: {it:5d}, Loss: {float(loss_val):.3e}, "
                  f"L_bc: {float(l_bc):.3e}, L_res: {float(l_r):.3e}, "
                  f"L2: {l2_err:.3e}, Time: {elapsed:.1f}s")

        if it % ntk_every == 0:
            K_uu = compute_ntk_kernel(u_pred_batch, params, (), x_ntk)
            K_rr = compute_ntk_kernel(u_xx_batch, params, (), x_ntk)
            ntk_K_log.append({'K_uu': np.array(K_uu), 'K_rr': np.array(K_rr)})
            ntk_iters_log.append(it)
            print(f"  [NTK computed at iter {it}]")

    total_time = time.time() - start_time
    u_pred_final = predict_u(params, X_test)
    final_l2 = np.linalg.norm(u_test - u_pred_final) / np.linalg.norm(u_test)

    print(f"\n  {model_name} training complete.")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Best L2 error:    {best_l2:.3e}")
    print(f"  Final L2 error:   {final_l2:.3e}")
    print(f"  Parameters:       {n_params}")

    return {
        'model_name': model_name, 'n_params': n_params,
        'total_time': total_time, 'best_l2': best_l2, 'final_l2': final_l2,
        'loss_bc_log': loss_bc_log, 'loss_res_log': loss_res_log,
        'l2_error_log': l2_error_log, 'iters_log': iters_log,
        'ntk_K_log': ntk_K_log, 'ntk_iters_log': ntk_iters_log,
        'u_pred_final': u_pred_final, 'u_test': u_test,
        'X_test': X_test[:, 0], 'params': params,
    }

# ============================================================
# Train mFF model (separate trainable structure)
# ============================================================
def train_mff(model_name, params_shared, W_final_tuple, W1, W2,
              mu_X, sigma_X, bcs_samplers, res_sampler,
              n_iter=40000, batch_size=128, log_every=100,
              ntk_every=5000, ntk_n_pts=64, seed=42,
              lr_decay_steps=1000, lr_decay_rate=0.9):

    print(f"\n{'='*70}")
    print(f"  Training model: {model_name}")
    print(f"{'='*70}")

    sigma_x = float(sigma_X[0])
    loss_total_fn, u_pred_batch, u_xx_batch = build_loss_fns_mff(sigma_x)

    trainable = (params_shared, W_final_tuple)
    flat0, _ = ravel_pytree(trainable)
    n_params = flat0.shape[0]
    print(f"  Trainable parameters: {n_params}")

    lr_schedule = optax.exponential_decay(
        init_value=1e-3, transition_steps=lr_decay_steps,
        decay_rate=lr_decay_rate, staircase=False)
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(trainable)

    @jit
    def train_step(trainable, opt_state, x_bc1, u_bc1, x_bc2, u_bc2, x_r, f_r):
        ps, wf = trainable
        (loss_val, (l_bc, l_r)), grads = jax.value_and_grad(
            lambda tr: loss_total_fn(tr[0], tr[1], W1, W2,
                                     x_bc1, u_bc1, x_bc2, u_bc2, x_r, f_r),
            has_aux=True)(trainable)
        updates, new_opt = optimizer.update(grads, opt_state, trainable)
        new_trainable = optax.apply_updates(trainable, updates)
        return new_trainable, new_opt, loss_val, l_bc, l_r

    def predict_u(trainable, X_pts):
        ps, wf = trainable
        X_norm = (X_pts - mu_X) / sigma_X
        return np.array(u_pred_batch(ps, wf, W1, W2, jnp.array(X_norm[:, 0])))

    nn_test = 1000
    X_test = np.linspace(0, 1, nn_test)[:, None]
    u_test = u_exact(X_test[:, 0])

    X_ntk_raw = np.linspace(0, 1, ntk_n_pts)[:, None]
    X_ntk_norm = (X_ntk_raw - mu_X) / sigma_X
    x_ntk = jnp.array(X_ntk_norm[:, 0])

    loss_bc_log, loss_res_log, l2_error_log, iters_log = [], [], [], []
    ntk_K_log, ntk_iters_log = [], []
    best_l2 = 1.0

    rng_train = np.random.RandomState(seed)
    start_time = time.time()

    for it in range(n_iter):
        X_bc1, u_bc1 = bcs_samplers[0].sample(batch_size, rng_train)
        X_bc2, u_bc2 = bcs_samplers[1].sample(batch_size, rng_train)
        X_res, f_res = res_sampler.sample(batch_size, rng_train)

        X_bc1_n = (X_bc1 - mu_X) / sigma_X
        X_bc2_n = (X_bc2 - mu_X) / sigma_X
        X_res_n = (X_res - mu_X) / sigma_X

        trainable, opt_state, loss_val, l_bc, l_r = train_step(
            trainable, opt_state,
            jnp.array(X_bc1_n[:, 0]), jnp.array(u_bc1.flatten()),
            jnp.array(X_bc2_n[:, 0]), jnp.array(u_bc2.flatten()),
            jnp.array(X_res_n[:, 0]), jnp.array(f_res.flatten()))

        if it % log_every == 0:
            u_pred = predict_u(trainable, X_test)
            l2_err = np.linalg.norm(u_test - u_pred) / np.linalg.norm(u_test)
            loss_bc_log.append(float(l_bc))
            loss_res_log.append(float(l_r))
            l2_error_log.append(float(l2_err))
            iters_log.append(it)
            if l2_err < best_l2:
                best_l2 = l2_err
            elapsed = time.time() - start_time
            print(f"  It: {it:5d}, Loss: {float(loss_val):.3e}, "
                  f"L_bc: {float(l_bc):.3e}, L_res: {float(l_r):.3e}, "
                  f"L2: {l2_err:.3e}, Time: {elapsed:.1f}s")

        if it % ntk_every == 0:
            ps_cur, wf_cur = trainable
            def ntk_u_batch(tr, pts):
                return u_pred_batch(tr[0], tr[1], W1, W2, pts)
            def ntk_r_batch(tr, pts):
                return u_xx_batch(tr[0], tr[1], W1, W2, pts)
            K_uu = compute_ntk_kernel(ntk_u_batch, trainable, (), x_ntk)
            K_rr = compute_ntk_kernel(ntk_r_batch, trainable, (), x_ntk)
            ntk_K_log.append({'K_uu': np.array(K_uu), 'K_rr': np.array(K_rr)})
            ntk_iters_log.append(it)
            print(f"  [NTK computed at iter {it}]")

    total_time = time.time() - start_time
    u_pred_final = predict_u(trainable, X_test)
    final_l2 = np.linalg.norm(u_test - u_pred_final) / np.linalg.norm(u_test)

    print(f"\n  {model_name} training complete.")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Best L2 error:    {best_l2:.3e}")
    print(f"  Final L2 error:   {final_l2:.3e}")
    print(f"  Parameters:       {n_params}")

    return {
        'model_name': model_name, 'n_params': n_params,
        'total_time': total_time, 'best_l2': best_l2, 'final_l2': final_l2,
        'loss_bc_log': loss_bc_log, 'loss_res_log': loss_res_log,
        'l2_error_log': l2_error_log, 'iters_log': iters_log,
        'ntk_K_log': ntk_K_log, 'ntk_iters_log': ntk_iters_log,
        'u_pred_final': u_pred_final, 'u_test': u_test,
        'X_test': X_test[:, 0], 'params': trainable,
    }

# ============================================================
# Save data for one model
# ============================================================
def save_model_data(results):
    name = results['model_name']
    iters = np.array(results['iters_log'])

    loss_data = np.column_stack([iters, results['loss_bc_log'],
                                 results['loss_res_log'], results['l2_error_log']])
    np.savetxt(os.path.join(DATA_DIR, f'loss_history_{name}.txt'), loss_data,
               header='iteration  loss_bc  loss_res  l2_error', fmt='%.6e')

    pred_data = np.column_stack([results['X_test'], results['u_test'],
                                 results['u_pred_final'],
                                 np.abs(results['u_test'] - results['u_pred_final'])])
    np.savetxt(os.path.join(DATA_DIR, f'prediction_{name}.txt'), pred_data,
               header='x  u_exact  u_pred  abs_error', fmt='%.6e')

    ntk_iters = results['ntk_iters_log']
    ntk_logs = results['ntk_K_log']
    eig_Kuu_all, eig_Krr_all, K_uu_list = [], [], []
    for i, nt in enumerate(ntk_logs):
        K_uu, K_rr = nt['K_uu'], nt['K_rr']
        eig_uu = np.sort(np.real(np.linalg.eigvalsh(K_uu)))[::-1]
        eig_rr = np.sort(np.real(np.linalg.eigvalsh(K_rr)))[::-1]
        eig_Kuu_all.append(eig_uu)
        eig_Krr_all.append(eig_rr)
        K_uu_list.append(K_uu)
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Kuu_{name}_iter{ntk_iters[i]}.txt'),
                   eig_uu, fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Krr_{name}_iter{ntk_iters[i]}.txt'),
                   eig_rr, fmt='%.6e')

    ntk_change = []
    if len(K_uu_list) > 0:
        K0 = K_uu_list[0]
        K0_norm = max(np.linalg.norm(K0), 1e-30)
        ntk_change = [np.linalg.norm(K - K0) / K0_norm for K in K_uu_list]
        np.savetxt(os.path.join(DATA_DIR, f'ntk_change_{name}.txt'),
                   np.column_stack([ntk_iters, ntk_change]),
                   header='iteration  ntk_relative_change', fmt='%.6e')

    with open(os.path.join(CKPT_DIR, f'params_{name}.pkl'), 'wb') as f:
        pickle.dump(results['params'], f)

    results['eig_Kuu_all'] = eig_Kuu_all
    results['eig_Krr_all'] = eig_Krr_all
    results['ntk_change'] = ntk_change

# ============================================================
# Plotting helpers
# ============================================================
def _label(ax, label, x=-0.10, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left')

def _style(ax, xlabel=None, ylabel=None, fs=14):
    if xlabel: ax.set_xlabel(xlabel, fontsize=fs, fontweight='bold')
    if ylabel: ax.set_ylabel(ylabel, fontsize=fs, fontweight='bold')
    ax.tick_params(labelsize=12, width=1.5, length=4)
    for s in ax.spines.values(): s.set_linewidth(2.0)

MODEL_COLORS = {'NN': 'tab:blue', 'FF': 'tab:orange', 'mFF': 'tab:green'}
MODEL_LS = {'NN': '-', 'FF': '--', 'mFF': '-.'}

# ============================================================
# Single model plots
# ============================================================
def plot_prediction_single(res):
    name = res['model_name']
    x = res['X_test']
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(x, res['u_test'], 'b-', lw=2, label='Exact')
    axes[0].plot(x, res['u_pred_final'], 'r--', lw=2, label='Predicted')
    axes[0].legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    _style(axes[0], '$x$', '$u(x)$')
    axes[0].set_title(f'{name} — Solution', fontsize=16, fontweight='bold')
    _label(axes[0], '(a)')

    axes[1].plot(x, res['u_test'] - res['u_pred_final'], lw=2, color='tab:purple')
    _style(axes[1], '$x$', 'Point-wise Error')
    axes[1].set_title(f'{name} — Error', fontsize=16, fontweight='bold')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    _label(axes[1], '(b)')

    axes[2].semilogy(x, np.abs(res['u_test'] - res['u_pred_final']), lw=2, color='tab:green')
    _style(axes[2], '$x$', 'Absolute Error')
    axes[2].set_title(f'{name} — |Error|', fontsize=16, fontweight='bold')
    _label(axes[2], '(c)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_loss_single(res):
    name = res['model_name']
    iters = np.array(res['iters_log'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(iters, res['loss_res_log'], lw=2, label=r'$\mathcal{L}_{r}$')
    axes[0].semilogy(iters, res['loss_bc_log'], lw=2, label=r'$\mathcal{L}_{b}$')
    axes[0].legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black')
    _style(axes[0], 'Iterations', 'Loss')
    axes[0].set_title(f'{name} — Loss', fontsize=16, fontweight='bold')
    _label(axes[0], '(a)')

    axes[1].semilogy(iters, res['l2_error_log'], lw=2, color='tab:red')
    _style(axes[1], 'Iterations', 'Relative $L^2$ Error')
    axes[1].set_title(f'{name} — $L^2$ Error', fontsize=16, fontweight='bold')
    _label(axes[1], '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_loss.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_ntk_single(res):
    name = res['model_name']
    eig_Kuu = res.get('eig_Kuu_all', [])
    eig_Krr = res.get('eig_Krr_all', [])
    ntk_iters = res['ntk_iters_log']
    if not eig_Kuu: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(eig_Kuu)))
    for i, (eu, er, it_n) in enumerate(zip(eig_Kuu, eig_Krr, ntk_iters)):
        axes[0].loglog(np.arange(1, len(eu)+1), np.clip(eu, 1e-30, None),
                       lw=2, color=colors[i], label=f'iter {it_n}')
        axes[1].loglog(np.arange(1, len(er)+1), np.clip(er, 1e-30, None),
                       lw=2, color=colors[i], label=f'iter {it_n}')

    axes[0].set_title(f'{name} — $K_{{uu}}$', fontsize=16, fontweight='bold')
    axes[1].set_title(f'{name} — $K_{{rr}}$', fontsize=16, fontweight='bold')
    for ax in axes:
        _style(ax, 'Index', 'Eigenvalue')
        ax.legend(fontsize=10, frameon=True, fancybox=False, edgecolor='black')
    _label(axes[0], '(a)'); _label(axes[1], '(b)')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'fig_{name}_ntk_eigenvalues.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Comparison plots
# ============================================================
def plot_comparison_prediction(all_res):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    x = all_res[0]['X_test']

    axes[0].plot(x, all_res[0]['u_test'], 'k-', lw=2.5, label='Exact')
    for r in all_res:
        n = r['model_name']
        axes[0].plot(x, r['u_pred_final'], lw=2, color=MODEL_COLORS[n],
                     linestyle=MODEL_LS[n], label=n)
    axes[0].legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    _style(axes[0], '$x$', '$u(x)$')
    axes[0].set_title('Solution Comparison', fontsize=16, fontweight='bold')
    _label(axes[0], '(a)')

    for r in all_res:
        n = r['model_name']
        axes[1].plot(x, r['u_test'] - r['u_pred_final'], lw=2,
                     color=MODEL_COLORS[n], linestyle=MODEL_LS[n], label=n)
    axes[1].legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    _style(axes[1], '$x$', 'Point-wise Error')
    axes[1].set_title('Error Comparison', fontsize=16, fontweight='bold')
    _label(axes[1], '(b)')

    for r in all_res:
        n = r['model_name']
        axes[2].semilogy(x, np.abs(r['u_test'] - r['u_pred_final']) + 1e-16,
                         lw=2, color=MODEL_COLORS[n], linestyle=MODEL_LS[n], label=n)
    axes[2].legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    _style(axes[2], '$x$', 'Absolute Error')
    axes[2].set_title('|Error| Comparison', fontsize=16, fontweight='bold')
    _label(axes[2], '(c)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_comparison_l2(all_res):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_res:
        n = r['model_name']
        ax.semilogy(r['iters_log'], r['l2_error_log'], lw=2.5,
                    color=MODEL_COLORS[n], linestyle=MODEL_LS[n],
                    label=f'{n}  (best={r["best_l2"]:.2e})')
    ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    _style(ax, 'Iterations', 'Relative $L^2$ Error', fs=16)
    ax.set_title('$L^2$ Error Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_l2_convergence.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_comparison_loss_all(all_res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for r in all_res:
        n = r['model_name']
        axes[0].semilogy(r['iters_log'], r['loss_bc_log'], lw=2.5,
                         color=MODEL_COLORS[n], linestyle=MODEL_LS[n], label=n)
        axes[1].semilogy(r['iters_log'], r['loss_res_log'], lw=2.5,
                         color=MODEL_COLORS[n], linestyle=MODEL_LS[n], label=n)
    axes[0].set_title(r'$\mathcal{L}_{b}$', fontsize=18, fontweight='bold')
    axes[1].set_title(r'$\mathcal{L}_{r}$', fontsize=18, fontweight='bold')
    for i, ax in enumerate(axes):
        _style(ax, 'Iterations', 'Loss')
        _label(ax, f'({"ab"[i]})')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=14,
               frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_loss_all.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_comparison_ntk_eigenvalues(all_res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, (ax, title, idx) in enumerate(zip(axes,
            ['$K_{rr}$ — Initial ($n=0$)', '$K_{rr}$ — Final'], [0, -1])):
        for r in all_res:
            n = r['model_name']
            eig = r.get('eig_Krr_all', [])
            if not eig: continue
            it_n = r['ntk_iters_log'][idx]
            ax.loglog(np.arange(1, len(eig[idx])+1), np.clip(eig[idx], 1e-30, None),
                      lw=2.5, color=MODEL_COLORS[n], linestyle=MODEL_LS[n],
                      label=f'{n} (iter {it_n})')
        ax.set_title(title, fontsize=16, fontweight='bold')
        _style(ax, 'Index', 'Eigenvalue')
        ax.legend(fontsize=12, frameon=True, fancybox=False, edgecolor='black')
        _label(ax, f'({"ab"[col]})')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_ntk_eigenvalues.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_comparison_ntk_change(all_res):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in all_res:
        n = r['model_name']
        nc = r.get('ntk_change', [])
        if nc:
            ax.plot(r['ntk_iters_log'], nc, lw=2.5, color=MODEL_COLORS[n],
                    linestyle=MODEL_LS[n], marker='o', markersize=4, label=n)
    ax.legend(fontsize=14, frameon=True, fancybox=False, edgecolor='black')
    _style(ax, 'Iterations', r'$\|K - K_0\| / \|K_0\|$', fs=16)
    ax.set_title('NTK Relative Change', fontsize=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_ntk_change.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_comparison_bar_chart(all_res):
    names = [r['model_name'] for r in all_res]
    final_l2 = [r['final_l2'] for r in all_res]
    best_l2 = [r['best_l2'] for r in all_res]
    times = [r['total_time'] for r in all_res]
    n_params = [r['n_params'] for r in all_res]
    x = np.arange(len(names)); w = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    bars1 = axes[0].bar(x - w/2, final_l2, w, label='Final $L^2$',
                        color=[MODEL_COLORS[n] for n in names], alpha=0.7,
                        edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + w/2, best_l2, w, label='Best $L^2$',
                        color=[MODEL_COLORS[n] for n in names], alpha=1.0,
                        edgecolor='black', linewidth=1.5)
    axes[0].set_yscale('log')
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=14, fontweight='bold')
    _style(axes[0], ylabel='Relative $L^2$ Error')
    axes[0].legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    axes[0].set_title('$L^2$ Error', fontsize=16, fontweight='bold')
    for bar, val in zip(bars1, final_l2):
        axes[0].text(bar.get_x()+bar.get_width()/2, val*1.5, f'{val:.1e}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    _label(axes[0], '(a)')

    axes[1].bar(x, times, w*1.5, color=[MODEL_COLORS[n] for n in names],
                edgecolor='black', linewidth=1.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=14, fontweight='bold')
    _style(axes[1], ylabel='Training Time (s)')
    axes[1].set_title('Training Time', fontsize=16, fontweight='bold')
    for xi, t in zip(x, times):
        axes[1].text(xi, t+max(times)*0.02, f'{t:.0f}s', ha='center', va='bottom',
                     fontsize=12, fontweight='bold')
    _label(axes[1], '(b)')

    axes[2].bar(x, n_params, w*1.5, color=[MODEL_COLORS[n] for n in names],
                edgecolor='black', linewidth=1.5)
    axes[2].set_xticks(x); axes[2].set_xticklabels(names, fontsize=14, fontweight='bold')
    _style(axes[2], ylabel='Parameters')
    axes[2].set_title('Trainable Parameters', fontsize=16, fontweight='bold')
    for xi, p in zip(x, n_params):
        axes[2].text(xi, p+max(n_params)*0.02, f'{p}', ha='center', va='bottom',
                     fontsize=12, fontweight='bold')
    _label(axes[2], '(c)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_comparison_bar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_comparison_table(all_res):
    lines = ["=" * 80,
             "COMPARISON TABLE — 1D Poisson Equation (Multi-scale PINN)",
             "=" * 80,
             f"{'Model':<10} {'Params':>10} {'Best L2':>14} {'Final L2':>14} {'Time (s)':>12}",
             "-" * 80]
    for r in all_res:
        lines.append(f"{r['model_name']:<10} {r['n_params']:>10d} "
                     f"{r['best_l2']:>14.4e} {r['final_l2']:>14.4e} "
                     f"{r['total_time']:>12.1f}")
    lines.append("=" * 80)
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(DATA_DIR, 'comparison_table.txt'), 'w') as f:
        f.write(table + "\n")

    header = 'model  n_params  best_l2  final_l2  time_s'
    with open(os.path.join(DATA_DIR, 'comparison_data.txt'), 'w') as f:
        f.write(header + "\n")
        for r in all_res:
            f.write(f"{r['model_name']}  {r['n_params']}  "
                    f"{r['best_l2']:.6e}  {r['final_l2']:.6e}  {r['total_time']:.1f}\n")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("Multi-scale Fourier Feature PINN — 1D Poisson Equation")
    print(f"a={A_FREQ}, b={B_FREQ}, sigma={SIGMA_FF}")
    print(f"u(x) = sin({A_FREQ}*pi*x) + 0.1*sin({B_FREQ}*pi*x)")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 70)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    # Problem setup
    bc1_coords = np.array([[0.0], [0.0]])
    bc2_coords = np.array([[1.0], [1.0]])
    dom_coords = np.array([[0.0], [1.0]])

    bc1_sampler = Sampler(1, bc1_coords, lambda x: u_exact(x), name='BC1')
    bc2_sampler = Sampler(1, bc2_coords, lambda x: u_exact(x), name='BC2')
    res_sampler = Sampler(1, dom_coords, lambda x: u_xx_exact(x), name='Residual')
    bcs_samplers = [bc1_sampler, bc2_sampler]

    mu_X, sigma_X = compute_norm_stats(res_sampler)
    print(f"Normalization: mu={mu_X}, sigma={sigma_X}")

    N_ITER = 40000
    BATCH_SIZE = 128
    SEED = 1234

    # --------------------------------------------------------
    # Model 1: NN
    # --------------------------------------------------------
    key = random.PRNGKey(SEED)
    layers_nn = [1, 100, 100, 1]
    params_nn = init_mlp(layers_nn, key)

    results_nn = train_nn_ff(
        'NN', apply_nn, params_nn, (), 0,
        mu_X, sigma_X, bcs_samplers, res_sampler,
        n_iter=N_ITER, batch_size=BATCH_SIZE, seed=SEED,
        lr_decay_steps=1000, lr_decay_rate=0.9)
    save_model_data(results_nn)
    plot_prediction_single(results_nn)
    plot_loss_single(results_nn)
    plot_ntk_single(results_nn)

    # --------------------------------------------------------
    # Model 2: FF
    # --------------------------------------------------------
    key = random.PRNGKey(SEED)
    layers_ff = [100, 100, 1]
    params_ff = init_mlp(layers_ff, key)
    key_ff = random.PRNGKey(SEED + 1)
    W_ff = random.normal(key_ff, (1, layers_ff[0] // 2)) * SIGMA_FF

    results_ff = train_nn_ff(
        'FF', apply_ff, params_ff, (W_ff,), 1,
        mu_X, sigma_X, bcs_samplers, res_sampler,
        n_iter=N_ITER, batch_size=BATCH_SIZE, seed=SEED,
        lr_decay_steps=100, lr_decay_rate=0.99)
    save_model_data(results_ff)
    plot_prediction_single(results_ff)
    plot_loss_single(results_ff)
    plot_ntk_single(results_ff)

    # --------------------------------------------------------
    # Model 3: mFF
    # --------------------------------------------------------
    key = random.PRNGKey(SEED)
    shared_layers = [100, 100]
    params_shared, W_final_tuple = init_mlp_mff(shared_layers, 200, 1, key)
    key_mff = random.PRNGKey(SEED + 2)
    k1, k2 = random.split(key_mff)
    W1 = random.normal(k1, (1, 50)) * 1.0
    W2 = random.normal(k2, (1, 50)) * SIGMA_FF

    results_mff = train_mff(
        'mFF', params_shared, W_final_tuple, W1, W2,
        mu_X, sigma_X, bcs_samplers, res_sampler,
        n_iter=N_ITER, batch_size=BATCH_SIZE, seed=SEED,
        lr_decay_steps=1000, lr_decay_rate=0.9)
    save_model_data(results_mff)
    plot_prediction_single(results_mff)
    plot_loss_single(results_mff)
    plot_ntk_single(results_mff)

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------
    all_results = [results_nn, results_ff, results_mff]
    plot_comparison_prediction(all_results)
    plot_comparison_l2(all_results)
    plot_comparison_loss_all(all_results)
    plot_comparison_ntk_eigenvalues(all_results)
    plot_comparison_ntk_change(all_results)
    plot_comparison_bar_chart(all_results)
    save_comparison_table(all_results)

    max_len = max(len(r['iters_log']) for r in all_results)
    for r in all_results:
        while len(r['l2_error_log']) < max_len:
            r['l2_error_log'].append(r['l2_error_log'][-1])
        while len(r['iters_log']) < max_len:
            r['iters_log'].append(r['iters_log'][-1])

    l2_data = np.column_stack([all_results[0]['iters_log'],
                               all_results[0]['l2_error_log'],
                               all_results[1]['l2_error_log'],
                               all_results[2]['l2_error_log']])
    np.savetxt(os.path.join(DATA_DIR, 'l2_error_comparison.txt'), l2_data,
               header='iteration  l2_NN  l2_FF  l2_mFF', fmt='%.6e')

    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
