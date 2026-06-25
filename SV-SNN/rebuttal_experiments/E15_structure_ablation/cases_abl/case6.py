"""
SV-SNN Acceleration – Case 6: 2D Helmholtz Equation (kappa = 48*pi)
====================================================================
PDE: -Laplacian(u) - kappa^2 * u = f(x,y) on [0,1]^2
BC:  u = 0 on boundary (homogeneous Dirichlet)
Exact: u(x,y) = sin(kappa*x) * sin(kappa*y)
Source: f(x,y) = kappa^2 * sin(kappa*x) * sin(kappa*y)

Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, PINN
"""

import os
import sys
import time
import json
import csv

sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jvp, value_and_grad
from functools import partial
import optax
from pyDOE import lhs

# ============================================================
# Configuration
# ============================================================
KAPPA = 48.0 * np.pi
W_CHAR = KAPPA
FF_DIM = 64
SEED = 42
EPOCHS = 10000
LR = 1e-3
N_PDE = 10000
N_BC = 1024
N_TEST = 256
EVAL_EVERY = 100
NC_SPINN = 100

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# PDE Definitions
# ============================================================
def exact_solution(x, y):
    return jnp.sin(KAPPA * x) * jnp.sin(KAPPA * y)


def source_term(x, y):
    return KAPPA**2 * jnp.sin(KAPPA * x) * jnp.sin(KAPPA * y)


# ============================================================
# Data Generation
# ============================================================
def generate_data(seed=SEED):
    np.random.seed(seed)

    # BC points: 256 per side x 4 sides = 1024
    n_per_side = N_BC // 4
    t = np.linspace(0, 1, n_per_side).reshape(-1, 1)

    x_bc_list = [
        np.zeros((n_per_side, 1)), np.ones((n_per_side, 1)), t, t
    ]
    y_bc_list = [
        t, t, np.zeros((n_per_side, 1)), np.ones((n_per_side, 1))
    ]
    x_bc = np.vstack(x_bc_list)
    y_bc = np.vstack(y_bc_list)
    u_bc = np.zeros((x_bc.shape[0], 1))

    # PDE collocation points via LHS
    pde_pts = lhs(2, samples=N_PDE)
    x_pde = pde_pts[:, 0:1]
    y_pde = pde_pts[:, 1:2]

    # Test grid
    x_test_1d = np.linspace(0, 1, N_TEST)
    y_test_1d = np.linspace(0, 1, N_TEST)
    X_test, Y_test = np.meshgrid(x_test_1d, y_test_1d, indexing='ij')
    u_exact_test = np.sin(KAPPA * X_test) * np.sin(KAPPA * Y_test)

    # Flatten for pointwise methods
    x_test_flat = X_test.reshape(-1, 1)
    y_test_flat = Y_test.reshape(-1, 1)

    data = {
        'x_bc': jnp.array(x_bc, dtype=jnp.float32),
        'y_bc': jnp.array(y_bc, dtype=jnp.float32),
        'u_bc': jnp.array(u_bc, dtype=jnp.float32),
        'x_pde': jnp.array(x_pde, dtype=jnp.float32),
        'y_pde': jnp.array(y_pde, dtype=jnp.float32),
        'X_test': X_test,
        'Y_test': Y_test,
        'u_exact_test': u_exact_test,
        'x_test_flat': jnp.array(x_test_flat, dtype=jnp.float32),
        'y_test_flat': jnp.array(y_test_flat, dtype=jnp.float32),
        'x_test_1d': jnp.array(x_test_1d.reshape(-1, 1), dtype=jnp.float32),
        'y_test_1d': jnp.array(y_test_1d.reshape(-1, 1), dtype=jnp.float32),
    }

    # SPINN grid data
    xc = np.linspace(0, 1, NC_SPINN).reshape(-1, 1)
    yc = np.linspace(0, 1, NC_SPINN).reshape(-1, 1)
    Xc, Yc = np.meshgrid(xc.flatten(), yc.flatten(), indexing='ij')
    f_grid = KAPPA**2 * np.sin(KAPPA * Xc) * np.sin(KAPPA * Yc)

    xb_left = np.zeros((NC_SPINN, 1))
    xb_right = np.ones((NC_SPINN, 1))
    yb_bottom = np.zeros((NC_SPINN, 1))
    yb_top = np.ones((NC_SPINN, 1))

    data['spinn'] = {
        'xc': jnp.array(xc, dtype=jnp.float32),
        'yc': jnp.array(yc, dtype=jnp.float32),
        'f_grid': jnp.array(f_grid, dtype=jnp.float32),
        'xb': [jnp.array(xb_left, dtype=jnp.float32),
                jnp.array(xb_right, dtype=jnp.float32),
                jnp.array(xc, dtype=jnp.float32),
                jnp.array(xc, dtype=jnp.float32)],
        'yb': [jnp.array(yc, dtype=jnp.float32),
                jnp.array(yc, dtype=jnp.float32),
                jnp.array(yb_bottom, dtype=jnp.float32),
                jnp.array(yb_top, dtype=jnp.float32)],
    }

    return data


# ============================================================
# Utility: hvp_fwdfwd (forward-over-forward Hessian-vector product)
# ============================================================
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), (tangents,))[1]
    primals_out, tangents_out = jvp(g, (primals,), (tangents,))
    if return_primals:
        return primals_out, tangents_out
    return tangents_out


# ============================================================
# Utility: parameter counting
# ============================================================
def count_params(params):
    leaves = jax.tree.leaves(params)
    return sum(p.size for p in leaves if hasattr(p, 'size'))


# ============================================================
# Utility: L2 relative error
# ============================================================
def l2_relative_error(u_pred, u_exact):
    return float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))


def _sample_frequencies(key, K, w_char):
    n_low = K // 4
    n_char = K // 2
    n_high = K - n_low - n_char
    k1, k2, k3 = jax.random.split(key, 3)
    freqs_low = jnp.linspace(1.0, w_char, n_low)
    freqs_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 30.0 + w_char)
    freqs_high = jax.random.uniform(k3, (n_high,), minval=w_char * 0.5, maxval=w_char)
    return jnp.sort(jnp.concatenate([freqs_low, freqs_char, freqs_high]))


# ============================================================
# METHOD 1: SV-SNN (Ours) — faithful to original svsnn_jax.py
# ============================================================
def svsnn_forward(params, x, y):
    """Module-level forward for save_results compatibility.
    Params layout: spatial_x/spatial_y lists with per-mode freqs, cos_c, sin_c, bias;
    mode_coeffs vector."""
    NUM_MODES = len(params['mode_coeffs'])
    u = jnp.zeros_like(x)
    for n in range(NUM_MODES):
        sp_x = params['spatial_x'][n]
        wx = sp_x['freqs'][None, :] * x
        X_n = jnp.sum(sp_x['cos_c'] * jnp.cos(wx) + sp_x['sin_c'] * jnp.sin(wx),
                       axis=1, keepdims=True) + sp_x['bias']
        sp_y = params['spatial_y'][n]
        wy = sp_y['freqs'][None, :] * y
        Y_n = jnp.sum(sp_y['cos_c'] * jnp.cos(wy) + sp_y['sin_c'] * jnp.sin(wy),
                       axis=1, keepdims=True) + sp_y['bias']
        u = u + params['mode_coeffs'][n] * X_n * Y_n
    return u


def run_svsnn(data):
    """Self-contained SV-SNN training matching original svsnn_jax.py for Helmholtz.
    Returns: params, history, u_pred, n_params, total_time"""
    print(f"\n{'='*60}")
    print(f"Training SV-SNN")
    print(f"{'='*60}")

    NUM_MODES = 8
    NUM_FREQ = 64

    def _sample_frequencies(key, K, w_char):
        n_low = K // 4
        n_char = K // 2
        n_high = K - n_low - n_char
        k1, k2, k3 = jax.random.split(key, 3)
        freqs_low = jnp.linspace(1.0, w_char, n_low)
        freqs_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 30.0 + w_char)
        freqs_high = jax.random.uniform(k3, (n_high,), minval=w_char * 0.5, maxval=w_char)
        return jnp.sort(jnp.concatenate([freqs_low, freqs_char, freqs_high]))

    def init_params(key):
        keys = jax.random.split(key, NUM_MODES * 6 + 1)
        ki = 0
        spatial_x_params = []
        spatial_y_params = []
        for n in range(NUM_MODES):
            freqs_x = _sample_frequencies(keys[ki], NUM_FREQ, KAPPA)
            ki += 1
            cos_cx = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            sin_cx = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            spatial_x_params.append({
                'freqs': freqs_x, 'cos_c': cos_cx,
                'sin_c': sin_cx, 'bias': jnp.zeros(1),
            })
            freqs_y = _sample_frequencies(keys[ki], NUM_FREQ, KAPPA)
            ki += 1
            cos_cy = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            sin_cy = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            spatial_y_params.append({
                'freqs': freqs_y, 'cos_c': cos_cy,
                'sin_c': sin_cy, 'bias': jnp.zeros(1),
            })
        mode_coeffs = jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1
        return {
            'spatial_x': spatial_x_params,
            'spatial_y': spatial_y_params,
            'mode_coeffs': mode_coeffs,
        }

    def spatial_forward(sp, x):
        wx = sp['freqs'][None, :] * x
        out = jnp.sum(sp['cos_c'] * jnp.cos(wx) + sp['sin_c'] * jnp.sin(wx),
                       axis=1, keepdims=True)
        return out + sp['bias']

    def forward(params, x, y):
        u = jnp.zeros_like(x)
        for n in range(NUM_MODES):
            X_n = spatial_forward(params['spatial_x'][n], x)
            Y_n = spatial_forward(params['spatial_y'][n], y)
            u = u + params['mode_coeffs'][n] * X_n * Y_n
        return u

    def pde_residual_single(params, x_s, y_s):
        def u_fn(x_, y_):
            return forward(params, x_[None, None], y_[None, None]).squeeze()
        u_val = u_fn(x_s, y_s)
        u_xx = jax.grad(jax.grad(u_fn, argnums=0), argnums=0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, argnums=1), argnums=1)(x_s, y_s)
        f_val = KAPPA**2 * jnp.sin(KAPPA * x_s) * jnp.sin(KAPPA * y_s)
        return -(u_xx + u_yy) - KAPPA**2 * u_val - f_val

    pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0))

    key = random.PRNGKey(SEED)
    params = init_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    x_pde_flat = data['x_pde'].squeeze()
    y_pde_flat = data['y_pde'].squeeze()
    x_bc, y_bc, u_bc = data['x_bc'], data['y_bc'], data['u_bc']

    def loss_fn(params, x_pde_1d, y_pde_1d, x_bc_, y_bc_, u_bc_):
        u_bc_pred = forward(params, x_bc_, y_bc_)
        bc_loss = jnp.mean((u_bc_pred - u_bc_)**2)

        residuals = pde_residual_batch(params, x_pde_1d, y_pde_1d)
        pde_loss = jnp.mean(residuals**2)

        total_loss = pde_loss + bc_loss
        return total_loss, (pde_loss, bc_loss)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_pde_flat, y_pde_flat, x_bc, y_bc, u_bc
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    history = {'total_loss': [], 'pde_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params

    for warmup_ep in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

    start_time = time.time()

    for epoch in range(2, EPOCHS):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            u_pred = forward(params, data['x_test_flat'], data['y_test_flat'])
            u_pred_np = np.array(u_pred).reshape(N_TEST, N_TEST)
            l2_err = l2_relative_error(u_pred_np, data['u_exact_test'])

            history['total_loss'].append(float(loss_val))
            history['pde_loss'].append(float(pde_val))
            history['bc_loss'].append(float(bc_val))
            history['l2_error'].append(l2_err)
            history['eval_epochs'].append(epoch)

            if l2_err < best_l2:
                best_l2 = l2_err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss_val):.4e} | "
                      f"PDE: {float(pde_val):.4e} | BC: {float(bc_val):.4e} | "
                      f"L2: {l2_err:.4e}")

    total_time = time.time() - start_time
    effective_epochs = EPOCHS - 2
    ms_per_epoch = (total_time / effective_epochs) * 1000

    u_pred_final = forward(best_params, data['x_test_flat'], data['y_test_flat'])
    u_pred_final_np = np.array(u_pred_final).reshape(N_TEST, N_TEST)

    print(f"  Training time: {total_time:.1f}s ({ms_per_epoch:.2f} ms/epoch)")
    print(f"  Best L2 error: {best_l2:.4e}")
    print(f"  Final L2 error: {history['l2_error'][-1]:.4e}")

    return best_params, history, u_pred_final_np, n_params, total_time


# ============================================================
# METHOD 1b: SV-SNN Accelerated (grid-based analytic derivatives)
# ============================================================
def run_svsnn_accelerated(data):
    """SV-SNN with vectorized grid evaluation and analytic spatial derivatives.
    Returns: params, history, u_pred, n_params, total_time"""
    print(f"\n{'='*60}")
    print(f"Training SV-SNN (Accelerated)")
    print(f"{'='*60}")

    NUM_MODES = 8
    NUM_FREQ = 64

    def _sample_frequencies_local(key, K, w_char):
        import _abl, abl_freqs
        if _abl.STRATEGY != 'default':
            return abl_freqs.strategy_sample(key, K, w_char, w_char, _abl.STRATEGY) * _abl.SCALE
        n_low = K // 4
        n_char = K // 2
        n_high = K - n_low - n_char
        k1, k2, k3 = jax.random.split(key, 3)
        freqs_low = jnp.linspace(1.0, w_char, n_low)
        freqs_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 30.0 + w_char)
        freqs_high = jax.random.uniform(k3, (n_high,), minval=w_char * 0.5, maxval=w_char)
        return jnp.sort(jnp.concatenate([freqs_low, freqs_char, freqs_high])) * _abl.SCALE

    def init_params(key):
        keys = jax.random.split(key, NUM_MODES * 6 + 1)
        ki = 0
        spatial_x_params = []
        spatial_y_params = []
        for n in range(NUM_MODES):
            freqs_x = _sample_frequencies_local(keys[ki], NUM_FREQ, KAPPA)
            ki += 1
            cos_cx = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            sin_cx = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            spatial_x_params.append({
                'freqs': freqs_x, 'cos_c': cos_cx,
                'sin_c': sin_cx, 'bias': jnp.zeros(1),
            })
            freqs_y = _sample_frequencies_local(keys[ki], NUM_FREQ, KAPPA)
            ki += 1
            cos_cy = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            sin_cy = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
            ki += 1
            spatial_y_params.append({
                'freqs': freqs_y, 'cos_c': cos_cy,
                'sin_c': sin_cy, 'bias': jnp.zeros(1),
            })
        mode_coeffs = jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1
        return {
            'spatial_x': spatial_x_params,
            'spatial_y': spatial_y_params,
            'mode_coeffs': mode_coeffs,
        }

    def _stack_spatial(params, axis_key):
        """Stack all freqs/cos_c/sin_c/bias across modes."""
        all_freqs = jnp.stack([jax.lax.stop_gradient(params[axis_key][n]['freqs'])
                               for n in range(NUM_MODES)])
        all_cos_c = jnp.stack([params[axis_key][n]['cos_c'] for n in range(NUM_MODES)])
        all_sin_c = jnp.stack([params[axis_key][n]['sin_c'] for n in range(NUM_MODES)])
        all_bias = jnp.stack([params[axis_key][n]['bias'] for n in range(NUM_MODES)])
        return all_freqs, all_cos_c, all_sin_c, all_bias

    def _compute_basis(coord_flat, all_freqs, all_cos_c, all_sin_c, all_bias):
        """Vectorized basis function evaluation.
        coord_flat: (P, 1), all_freqs: (M, K)
        Returns: vals (P, M), trig_terms (P, M, K)"""
        # (P, 1, 1) * (1, M, K) -> (P, M, K)
        phase = coord_flat[:, :, None] * all_freqs[None, :, :]
        cos_phase = jnp.cos(phase)
        sin_phase = jnp.sin(phase)
        trig_terms = all_cos_c[None, :, :] * cos_phase + all_sin_c[None, :, :] * sin_phase
        vals = jnp.sum(trig_terms, axis=2) + all_bias[None, :, 0]
        return vals, trig_terms

    def _second_deriv(trig_terms, all_freqs):
        """Analytic second derivative: sum_k -w_k^2 * trig_terms_k.
        trig_terms: (P, M, K), all_freqs: (M, K)
        Returns: (P, M)"""
        w_sq = all_freqs[None, :, :] ** 2
        return jnp.sum(-w_sq * trig_terms, axis=2)

    def vectorized_forward(params, x, y):
        """Pointwise forward using stacked spatial. x, y: (P, 1)."""
        all_freqs_x, all_cos_x, all_sin_x, all_bias_x = _stack_spatial(params, 'spatial_x')
        all_freqs_y, all_cos_y, all_sin_y, all_bias_y = _stack_spatial(params, 'spatial_y')

        X_vals, _ = _compute_basis(x, all_freqs_x, all_cos_x, all_sin_x, all_bias_x)
        Y_vals, _ = _compute_basis(y, all_freqs_y, all_cos_y, all_sin_y, all_bias_y)

        mode_c = params['mode_coeffs']
        u = jnp.sum(mode_c[None, :] * X_vals * Y_vals, axis=1, keepdims=True)
        return u

    def vectorized_forward_grid(params, x, y):
        """Grid forward: x (Nx,1), y (Ny,1) -> u (Nx, Ny) via einsum."""
        all_freqs_x, all_cos_x, all_sin_x, all_bias_x = _stack_spatial(params, 'spatial_x')
        all_freqs_y, all_cos_y, all_sin_y, all_bias_y = _stack_spatial(params, 'spatial_y')

        X_vals, _ = _compute_basis(x, all_freqs_x, all_cos_x, all_sin_x, all_bias_x)
        Y_vals, _ = _compute_basis(y, all_freqs_y, all_cos_y, all_sin_y, all_bias_y)

        mode_c = params['mode_coeffs']
        # X_vals: (Nx, M), Y_vals: (Ny, M)
        # cX = mode_c * X_vals -> (Nx, M)
        cX = mode_c[None, :] * X_vals
        # u_grid = cX @ Y_vals.T but per-mode sum: einsum('nm,jm->nj', cX, Y_vals)
        u_grid = jnp.einsum('nm,jm->nj', cX, Y_vals)
        return u_grid

    # Grid data for training
    spinn_data = data['spinn']
    xc = spinn_data['xc']  # (NC, 1)
    yc = spinn_data['yc']  # (NC, 1)
    f_grid = spinn_data['f_grid']  # (NC, NC)

    def vectorized_pde_residual(params):
        """Grid PDE residual using analytic second derivatives."""
        all_freqs_x, all_cos_x, all_sin_x, all_bias_x = _stack_spatial(params, 'spatial_x')
        all_freqs_y, all_cos_y, all_sin_y, all_bias_y = _stack_spatial(params, 'spatial_y')

        X_vals, X_trig = _compute_basis(xc, all_freqs_x, all_cos_x, all_sin_x, all_bias_x)
        Y_vals, Y_trig = _compute_basis(yc, all_freqs_y, all_cos_y, all_sin_y, all_bias_y)

        X_dd = _second_deriv(X_trig, all_freqs_x)  # (Nx, M)
        Y_dd = _second_deriv(Y_trig, all_freqs_y)  # (Ny, M)

        mode_c = params['mode_coeffs']

        # u_val grid: sum_m c_m * X_m(x_i) * Y_m(y_j)
        cX = mode_c[None, :] * X_vals
        u_val = jnp.einsum('nm,jm->nj', cX, Y_vals)

        # u_xx grid: sum_m c_m * X_m''(x_i) * Y_m(y_j)
        cXdd = mode_c[None, :] * X_dd
        u_xx = jnp.einsum('nm,jm->nj', cXdd, Y_vals)

        # u_yy grid: sum_m c_m * X_m(x_i) * Y_m''(y_j)
        u_yy = jnp.einsum('nm,jm->nj', cX, Y_dd)

        residual = -(u_xx + u_yy) - KAPPA**2 * u_val - f_grid
        return residual

    def bc_loss_grid(params):
        """Evaluate u on 4 boundary sides, u=0."""
        xb_list = spinn_data['xb']
        yb_list = spinn_data['yb']
        bc_loss = jnp.float32(0.0)
        # Left: x=0
        u_left = vectorized_forward(params, xb_list[0], yb_list[0])
        bc_loss = bc_loss + jnp.mean(u_left**2)
        # Right: x=1
        u_right = vectorized_forward(params, xb_list[1], yb_list[1])
        bc_loss = bc_loss + jnp.mean(u_right**2)
        # Bottom: y=0
        u_bottom = vectorized_forward(params, xb_list[2], yb_list[2])
        bc_loss = bc_loss + jnp.mean(u_bottom**2)
        # Top: y=1
        u_top = vectorized_forward(params, xb_list[3], yb_list[3])
        bc_loss = bc_loss + jnp.mean(u_top**2)
        return bc_loss

    def loss_fn(params):
        residual = vectorized_pde_residual(params)
        pde_loss = jnp.mean(residual**2)
        bc_loss = bc_loss_grid(params)
        total_loss = pde_loss + bc_loss
        return total_loss, (pde_loss, bc_loss)

    key = random.PRNGKey(SEED)
    params = init_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    history = {'total_loss': [], 'pde_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params

    for warmup_ep in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

    start_time = time.time()

    for epoch in range(2, EPOCHS):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            u_pred_grid = vectorized_forward_grid(params, data['x_test_1d'], data['y_test_1d'])
            u_pred_np = np.array(u_pred_grid)
            l2_err = l2_relative_error(u_pred_np, data['u_exact_test'])

            history['total_loss'].append(float(loss_val))
            history['pde_loss'].append(float(pde_val))
            history['bc_loss'].append(float(bc_val))
            history['l2_error'].append(l2_err)
            history['eval_epochs'].append(epoch)

            if l2_err < best_l2:
                best_l2 = l2_err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss_val):.4e} | "
                      f"PDE: {float(pde_val):.4e} | BC: {float(bc_val):.4e} | "
                      f"L2: {l2_err:.4e}")

    total_time = time.time() - start_time
    effective_epochs = EPOCHS - 2
    ms_per_epoch = (total_time / effective_epochs) * 1000

    u_pred_final = vectorized_forward_grid(best_params, data['x_test_1d'], data['y_test_1d'])
    u_pred_final_np = np.array(u_pred_final)

    print(f"  Training time: {total_time:.1f}s ({ms_per_epoch:.2f} ms/epoch)")
    print(f"  Best L2 error: {best_l2:.4e}")
    print(f"  Final L2 error: {history['l2_error'][-1]:.4e}")

    return best_params, history, u_pred_final_np, n_params, total_time


# ============================================================
# METHOD 2: SPINN
# ============================================================
FF_DIM_SPINN = 64

def init_spinn(key, features=64, n_layers=4, r=64):
    """Modified MLP for SPINN branch network with Fourier embedding."""
    ff_input_dim = 2 * FF_DIM_SPINN

    def init_branch(key, d_in):
        keys = random.split(key, 3 + n_layers + 1)
        scale = 1.0 / jnp.sqrt(jnp.array(d_in, dtype=jnp.float32))
        params = {
            'U_w': random.normal(keys[0], (d_in, features)) * scale,
            'U_b': jnp.zeros((features,)),
            'V_w': random.normal(keys[1], (d_in, features)) * scale,
            'V_b': jnp.zeros((features,)),
            'H_w': random.normal(keys[2], (d_in, features)) * scale,
            'H_b': jnp.zeros((features,)),
            'layers': [],
            'out_w': random.normal(keys[-1], (features, r)) * (1.0 / jnp.sqrt(jnp.array(features, dtype=jnp.float32))),
        }
        for i in range(n_layers):
            w = random.normal(keys[3 + i], (features, features)) * (1.0 / jnp.sqrt(jnp.array(features, dtype=jnp.float32)))
            b = jnp.zeros((features,))
            params['layers'].append({'w': w, 'b': b})
        return params

    k1, k2, k3, k4 = random.split(key, 4)
    return {
        'branch_x': init_branch(k1, ff_input_dim),
        'branch_y': init_branch(k2, ff_input_dim),
        'W_x': _sample_frequencies(k3, FF_DIM_SPINN, W_CHAR).reshape(1, -1),
        'W_y': _sample_frequencies(k4, FF_DIM_SPINN, W_CHAR).reshape(1, -1),
    }


def spinn_fourier_embed(coord, W):
    return jnp.concatenate([jnp.sin(coord @ W), jnp.cos(coord @ W)], axis=-1)


def spinn_branch_forward(params, x):
    """Modified MLP forward pass."""
    U = jnp.tanh(x @ params['U_w'] + params['U_b'])
    V = jnp.tanh(x @ params['V_w'] + params['V_b'])
    H = jnp.tanh(x @ params['H_w'] + params['H_b'])
    for layer in params['layers']:
        Z = jnp.tanh(H @ layer['w'] + layer['b'])
        H = (1.0 - Z) * U + Z * V
    return H @ params['out_w']


def spinn_forward(params, x, y):
    """u(x,y) = branch_x(embed(x)) @ branch_y(embed(y)).T -> (Nx, Ny)"""
    x_emb = spinn_fourier_embed(x, params['W_x'])
    y_emb = spinn_fourier_embed(y, params['W_y'])
    bx = spinn_branch_forward(params['branch_x'], x_emb)
    by = spinn_branch_forward(params['branch_y'], y_emb)
    return bx @ by.T


def spinn_forward_for_deriv(params, x, y):
    """Wrapper that takes (x, y) as separate args for differentiation."""
    return spinn_forward(params, x, y)


def spinn_loss_fn(params, xc, yc, f_grid, xb_list, yb_list):
    # Forward on interior grid
    def u_fn_x(x_in):
        return spinn_forward(params, x_in, yc)

    def u_fn_y(y_in):
        return spinn_forward(params, xc, y_in)

    # Second derivatives via hvp_fwdfwd
    Nx = xc.shape[0]
    Ny = yc.shape[0]
    tangents_x = jnp.ones_like(xc)
    tangents_y = jnp.ones_like(yc)

    u_xx = hvp_fwdfwd(u_fn_x, xc, tangents_x)  # (Nx, Ny)
    u_yy = hvp_fwdfwd(u_fn_y, yc, tangents_y).T  # (Ny, Nx).T = (Nx, Ny)

    u_grid = spinn_forward(params, xc, yc)  # (Nx, Ny)
    laplacian = u_xx + u_yy
    residual = -laplacian - KAPPA**2 * u_grid - f_grid
    pde_loss = jnp.mean(residual**2)

    # BC loss
    bc_loss = jnp.float32(0.0)
    # Left: x=0
    u_left = spinn_forward(params, xb_list[0], yb_list[0])
    bc_loss = bc_loss + jnp.mean(u_left**2)
    # Right: x=1
    u_right = spinn_forward(params, xb_list[1], yb_list[1])
    bc_loss = bc_loss + jnp.mean(u_right**2)
    # Bottom: y=0
    u_bottom = spinn_forward(params, xb_list[2], yb_list[2])
    bc_loss = bc_loss + jnp.mean(u_bottom**2)
    # Top: y=1
    u_top = spinn_forward(params, xb_list[3], yb_list[3])
    bc_loss = bc_loss + jnp.mean(u_top**2)

    total_loss = pde_loss + bc_loss
    return total_loss, (pde_loss, bc_loss)


# ============================================================
# METHOD 3: SIREN
# ============================================================
def init_siren(key, ff_dim=64, hidden=128, n_hidden=4):
    k1, k2, key = random.split(key, 3)
    W_x = _sample_frequencies(k1, ff_dim, W_CHAR).reshape(1, -1)
    W_y = _sample_frequencies(k2, ff_dim, W_CHAR).reshape(1, -1)
    ff_input = 4 * ff_dim
    layers_list = [ff_input] + [hidden] * n_hidden + [1]
    params = {'layers': [], 'W_x': W_x, 'W_y': W_y}
    for i in range(len(layers_list) - 1):
        k, key = random.split(key)
        d_in, d_out = layers_list[i], layers_list[i + 1]
        std = jnp.sqrt(2.0 / d_in)
        w = random.normal(k, (d_in, d_out)) * std
        b = jnp.zeros((d_out,))
        params['layers'].append({'w': w, 'b': b})
    return params


def siren_forward(params, xy):
    Wx, Wy = params['W_x'], params['W_y']
    x, y = xy[:, 0:1], xy[:, 1:2]
    Hx = jnp.concatenate([jnp.sin(x @ Wx), jnp.cos(x @ Wx)], axis=-1)
    Hy = jnp.concatenate([jnp.sin(y @ Wy), jnp.cos(y @ Wy)], axis=-1)
    h = jnp.concatenate([Hx, Hy], axis=-1)
    n_layers = len(params['layers'])
    for i, layer in enumerate(params['layers']):
        h = h @ layer['w'] + layer['b']
        if i < n_layers - 1:
            h = jnp.sin(h)
    return h


def siren_u(params, x, y):
    xy = jnp.concatenate([x, y], axis=-1)
    return siren_forward(params, xy)


def siren_loss_fn(params, x_pde, y_pde, x_bc, y_bc, u_bc):
    # BC loss
    u_bc_pred = siren_u(params, x_bc, y_bc)
    bc_loss = jnp.mean((u_bc_pred - u_bc)**2)

    # PDE residual via hvp_fwdfwd
    xy_pde = jnp.concatenate([x_pde, y_pde], axis=-1)
    N = xy_pde.shape[0]

    def u_fn(xy):
        return siren_forward(params, xy)

    tangents_x = jnp.zeros_like(xy_pde).at[:, 0].set(1.0)
    tangents_y = jnp.zeros_like(xy_pde).at[:, 1].set(1.0)

    u_xx = hvp_fwdfwd(u_fn, xy_pde, tangents_x)
    u_yy = hvp_fwdfwd(u_fn, xy_pde, tangents_y)

    u_pred = siren_forward(params, xy_pde)
    laplacian = u_xx + u_yy
    f = source_term(x_pde, y_pde)
    residual = -laplacian - KAPPA**2 * u_pred - f
    pde_loss = jnp.mean(residual**2)

    total_loss = bc_loss + pde_loss
    return total_loss, (pde_loss, bc_loss)


# ============================================================
# METHOD 4: FourierPINN
# ============================================================
def init_fourier_pinn(key, ff_dim=64, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [128, 128, 128, 1]

    k1, k2, key = random.split(key, 3)
    W_x = _sample_frequencies(k1, ff_dim, W_CHAR).reshape(1, -1)
    W_y = _sample_frequencies(k2, ff_dim, W_CHAR).reshape(1, -1)

    input_dim = 4 * ff_dim  # [sin(x@Wx), cos(x@Wx), sin(y@Wy), cos(y@Wy)]
    params = {'W_x': W_x, 'W_y': W_y, 'mlp_layers': []}

    dims = [input_dim] + hidden_layers
    for i in range(len(dims) - 1):
        k, key = random.split(key)
        d_in = dims[i]
        d_out = dims[i + 1]
        limit = jnp.sqrt(6.0 / (d_in + d_out))
        w = random.uniform(k, (d_in, d_out), minval=-limit, maxval=limit)
        b = jnp.zeros((d_out,))
        params['mlp_layers'].append({'w': w, 'b': b})

    return params


def fourier_pinn_forward(params, xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    W_x = jax.lax.stop_gradient(params['W_x'])
    W_y = jax.lax.stop_gradient(params['W_y'])

    hx = jnp.concatenate([jnp.sin(x @ W_x), jnp.cos(x @ W_x)], axis=-1)
    hy = jnp.concatenate([jnp.sin(y @ W_y), jnp.cos(y @ W_y)], axis=-1)
    h = jnp.concatenate([hx, hy], axis=-1)

    n_layers = len(params['mlp_layers'])
    for i, layer in enumerate(params['mlp_layers']):
        h = h @ layer['w'] + layer['b']
        if i < n_layers - 1:
            h = jnp.tanh(h)
    return h


def fourier_pinn_u(params, x, y):
    xy = jnp.concatenate([x, y], axis=-1)
    return fourier_pinn_forward(params, xy)


def fourier_pinn_loss_fn(params, x_pde, y_pde, x_bc, y_bc, u_bc):
    # BC loss
    u_bc_pred = fourier_pinn_u(params, x_bc, y_bc)
    bc_loss = jnp.mean((u_bc_pred - u_bc)**2)

    # PDE residual
    xy_pde = jnp.concatenate([x_pde, y_pde], axis=-1)

    def u_fn(xy):
        return fourier_pinn_forward(params, xy)

    tangents_x = jnp.zeros_like(xy_pde).at[:, 0].set(1.0)
    tangents_y = jnp.zeros_like(xy_pde).at[:, 1].set(1.0)

    u_xx = hvp_fwdfwd(u_fn, xy_pde, tangents_x)
    u_yy = hvp_fwdfwd(u_fn, xy_pde, tangents_y)

    u_pred = fourier_pinn_forward(params, xy_pde)
    laplacian = u_xx + u_yy
    f = source_term(x_pde, y_pde)
    residual = -laplacian - KAPPA**2 * u_pred - f
    pde_loss = jnp.mean(residual**2)

    total_loss = bc_loss + pde_loss
    return total_loss, (pde_loss, bc_loss)


# ============================================================
# METHOD 5: Vanilla PINN
# ============================================================
def init_pinn(key, layers_list=None):
    if layers_list is None:
        layers_list = [2, 128, 128, 128, 128, 1]

    params = {'layers': []}
    for i in range(len(layers_list) - 1):
        k, key = random.split(key)
        d_in = layers_list[i]
        d_out = layers_list[i + 1]
        limit = jnp.sqrt(6.0 / (d_in + d_out))
        w = random.uniform(k, (d_in, d_out), minval=-limit, maxval=limit)
        b = jnp.zeros((d_out,))
        params['layers'].append({'w': w, 'b': b})
    return params


def pinn_forward(params, xy):
    h = xy
    n_layers = len(params['layers'])
    for i, layer in enumerate(params['layers']):
        h = h @ layer['w'] + layer['b']
        if i < n_layers - 1:
            h = jnp.tanh(h)
    return h


def pinn_u(params, x, y):
    xy = jnp.concatenate([x, y], axis=-1)
    return pinn_forward(params, xy)


def pinn_loss_fn(params, x_pde, y_pde, x_bc, y_bc, u_bc):
    # BC loss
    u_bc_pred = pinn_u(params, x_bc, y_bc)
    bc_loss = jnp.mean((u_bc_pred - u_bc)**2)

    # PDE residual
    xy_pde = jnp.concatenate([x_pde, y_pde], axis=-1)

    def u_fn(xy):
        return pinn_forward(params, xy)

    tangents_x = jnp.zeros_like(xy_pde).at[:, 0].set(1.0)
    tangents_y = jnp.zeros_like(xy_pde).at[:, 1].set(1.0)

    u_xx = hvp_fwdfwd(u_fn, xy_pde, tangents_x)
    u_yy = hvp_fwdfwd(u_fn, xy_pde, tangents_y)

    u_pred = pinn_forward(params, xy_pde)
    laplacian = u_xx + u_yy
    f = source_term(x_pde, y_pde)
    residual = -laplacian - KAPPA**2 * u_pred - f
    pde_loss = jnp.mean(residual**2)

    total_loss = bc_loss + pde_loss
    return total_loss, (pde_loss, bc_loss)


# ============================================================
# Training Framework
# ============================================================
def train_pointwise_method(name, params, loss_fn, predict_fn, data, epochs=EPOCHS):
    """Generic training loop for pointwise methods (SV-SNN, SIREN, FourierPINN, PINN)."""
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")

    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    x_pde, y_pde = data['x_pde'], data['y_pde']
    x_bc, y_bc, u_bc = data['x_bc'], data['y_bc'], data['u_bc']

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, x_pde, y_pde, x_bc, y_bc, u_bc
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    total_loss_hist = []
    pde_loss_hist = []
    bc_loss_hist = []
    l2_error_hist = []
    eval_epochs = []
    best_l2 = float('inf')
    best_params = params

    # Warmup (epoch 0 and 1 for JIT)
    for warmup_ep in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

    start_time = time.time()

    for epoch in range(2, epochs):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == epochs - 1:
            # Evaluate
            u_pred = predict_fn(params, data['x_test_flat'], data['y_test_flat'])
            u_pred_np = np.array(u_pred).reshape(N_TEST, N_TEST)
            l2_err = l2_relative_error(u_pred_np, data['u_exact_test'])

            total_loss_hist.append(float(loss_val))
            pde_loss_hist.append(float(pde_val))
            bc_loss_hist.append(float(bc_val))
            l2_error_hist.append(l2_err)
            eval_epochs.append(epoch)

            if l2_err < best_l2:
                best_l2 = l2_err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss_val):.4e} | "
                      f"PDE: {float(pde_val):.4e} | BC: {float(bc_val):.4e} | "
                      f"L2: {l2_err:.4e}")

    elapsed = time.time() - start_time
    effective_epochs = epochs - 2
    ms_per_epoch = (elapsed / effective_epochs) * 1000

    print(f"  Training time: {elapsed:.1f}s ({ms_per_epoch:.2f} ms/epoch)")
    print(f"  Best L2 error: {best_l2:.4e}")
    print(f"  Final L2 error: {l2_error_hist[-1]:.4e}")

    return {
        'params': best_params,
        'total_loss': np.array(total_loss_hist),
        'pde_loss': np.array(pde_loss_hist),
        'bc_loss': np.array(bc_loss_hist),
        'l2_error': np.array(l2_error_hist),
        'eval_epochs': np.array(eval_epochs),
        'total_time_sec': elapsed,
        'ms_per_epoch': ms_per_epoch,
        'best_l2_error': best_l2,
        'final_l2_error': l2_error_hist[-1],
        'total_params': n_params,
    }


def train_spinn(data, epochs=EPOCHS, params=None):
    """Training loop for SPINN (grid-based)."""
    print(f"\n{'='*60}")
    print(f"Training SPINN")
    print(f"{'='*60}")

    key = random.PRNGKey(SEED)
    if params is None:
        params = init_spinn(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    spinn_data = data['spinn']
    xc, yc = spinn_data['xc'], spinn_data['yc']
    f_grid = spinn_data['f_grid']
    xb_list = spinn_data['xb']
    yb_list = spinn_data['yb']

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = jax.value_and_grad(spinn_loss_fn, has_aux=True)(
            params, xc, yc, f_grid, xb_list, yb_list
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    total_loss_hist = []
    pde_loss_hist = []
    bc_loss_hist = []
    l2_error_hist = []
    eval_epochs = []
    best_l2 = float('inf')
    best_params = params

    # Warmup
    for warmup_ep in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

    start_time = time.time()

    for epoch in range(2, epochs):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == epochs - 1:
            # Evaluate on test grid
            u_pred_grid = spinn_forward(params, data['x_test_1d'], data['y_test_1d'])
            u_pred_np = np.array(u_pred_grid)
            l2_err = l2_relative_error(u_pred_np, data['u_exact_test'])

            total_loss_hist.append(float(loss_val))
            pde_loss_hist.append(float(pde_val))
            bc_loss_hist.append(float(bc_val))
            l2_error_hist.append(l2_err)
            eval_epochs.append(epoch)

            if l2_err < best_l2:
                best_l2 = l2_err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss_val):.4e} | "
                      f"PDE: {float(pde_val):.4e} | BC: {float(bc_val):.4e} | "
                      f"L2: {l2_err:.4e}")

    elapsed = time.time() - start_time
    effective_epochs = epochs - 2
    ms_per_epoch = (elapsed / effective_epochs) * 1000

    print(f"  Training time: {elapsed:.1f}s ({ms_per_epoch:.2f} ms/epoch)")
    print(f"  Best L2 error: {best_l2:.4e}")
    print(f"  Final L2 error: {l2_error_hist[-1]:.4e}")

    return {
        'params': best_params,
        'total_loss': np.array(total_loss_hist),
        'pde_loss': np.array(pde_loss_hist),
        'bc_loss': np.array(bc_loss_hist),
        'l2_error': np.array(l2_error_hist),
        'eval_epochs': np.array(eval_epochs),
        'total_time_sec': elapsed,
        'ms_per_epoch': ms_per_epoch,
        'best_l2_error': best_l2,
        'final_l2_error': l2_error_hist[-1],
        'total_params': n_params,
    }


# ============================================================
# Save Results
# ============================================================
def save_results(name, result, data):
    """Save params, history, prediction, and summary for one method."""
    # Parameters
    np.save(os.path.join(SAVE_DIR, f"{name}_params.npy"),
            jax.tree.map(np.array, result['params']), allow_pickle=True)

    # History
    np.savez(os.path.join(SAVE_DIR, f"{name}_history.npz"),
             total_loss=result['total_loss'],
             pde_loss=result['pde_loss'],
             bc_loss=result['bc_loss'],
             l2_error=result['l2_error'],
             eval_epochs=result['eval_epochs'])

    # Prediction on test grid
    if name in ('SPINN', 'SVSNN_accel'):
        u_pred_np = result.get('u_pred_grid', None)
        if u_pred_np is None:
            if name == 'SPINN':
                u_pred_grid = spinn_forward(result['params'], data['x_test_1d'], data['y_test_1d'])
            u_pred_np = np.array(u_pred_grid)
    else:
        if name in ('SVSNN', 'SVSNN_orig'):
            pred_fn = svsnn_forward
        elif name == 'SIREN':
            pred_fn = siren_u
        elif name == 'FourierPINN':
            pred_fn = fourier_pinn_u
        elif name == 'PINN':
            pred_fn = pinn_u
        u_pred = pred_fn(result['params'], data['x_test_flat'], data['y_test_flat'])
        u_pred_np = np.array(u_pred).reshape(N_TEST, N_TEST)

    np.savez(os.path.join(SAVE_DIR, f"{name}_prediction.npz"),
             u_pred=u_pred_np,
             u_exact=data['u_exact_test'],
             X=data['X_test'],
             Y=data['Y_test'])

    # Summary JSON
    summary = {
        'method': name,
        'total_params': result['total_params'],
        'total_time_sec': result['total_time_sec'],
        'best_l2_error': result['best_l2_error'],
        'final_l2_error': result['final_l2_error'],
        'ms_per_epoch': result['ms_per_epoch'],
    }
    with open(os.path.join(SAVE_DIR, f"{name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved results for {name}")


def save_comparison_table(all_results):
    """Save comparison CSV with all methods side by side."""
    fieldnames = ['method', 'total_params', 'total_time_sec', 'best_l2_error',
                  'final_l2_error', 'ms_per_epoch']
    filepath = os.path.join(SAVE_DIR, "comparison_table.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, result in all_results.items():
            writer.writerow({
                'method': name,
                'total_params': result['total_params'],
                'total_time_sec': f"{result['total_time_sec']:.2f}",
                'best_l2_error': f"{result['best_l2_error']:.6e}",
                'final_l2_error': f"{result['final_l2_error']:.6e}",
                'ms_per_epoch': f"{result['ms_per_epoch']:.2f}",
            })
    print(f"\nComparison table saved to {filepath}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("SV-SNN Acceleration – Case 6: 2D Helmholtz (kappa = 48*pi)")
    print(f"  kappa = {KAPPA:.4f}")
    print(f"  Epochs = {EPOCHS}")
    print(f"  LR = {LR}")
    print(f"  N_PDE = {N_PDE}, N_BC = {N_BC}, N_TEST = {N_TEST}")
    print("=" * 60)

    # Generate data
    print("\nGenerating data...")
    data = generate_data()
    print("  Done.")

    key = random.PRNGKey(SEED)
    all_results = {}

    # ---- Method 1a: SV-SNN (Accelerated) ----
    params_accel, hist_accel, u_pred_accel, npar_accel, time_accel = run_svsnn_accelerated(data)
    result_svsnn_accel = {
        'params': params_accel,
        'total_loss': np.array(hist_accel['total_loss']),
        'pde_loss': np.array(hist_accel['pde_loss']),
        'bc_loss': np.array(hist_accel['bc_loss']),
        'l2_error': np.array(hist_accel['l2_error']),
        'eval_epochs': np.array(hist_accel['eval_epochs']),
        'total_time_sec': time_accel,
        'ms_per_epoch': (time_accel / (EPOCHS - 2)) * 1000,
        'best_l2_error': float(min(hist_accel['l2_error'])),
        'final_l2_error': float(hist_accel['l2_error'][-1]),
        'total_params': npar_accel,
        'u_pred_grid': u_pred_accel,
    }
    all_results['SVSNN_accel'] = result_svsnn_accel
    save_results('SVSNN_accel', result_svsnn_accel, data)

    # ---- Method 1b: SV-SNN (Original) ----
    params_sv, hist_sv, u_pred_sv, npar_sv, time_sv = run_svsnn(data)
    result_svsnn_orig = {
        'params': params_sv,
        'total_loss': np.array(hist_sv['total_loss']),
        'pde_loss': np.array(hist_sv['pde_loss']),
        'bc_loss': np.array(hist_sv['bc_loss']),
        'l2_error': np.array(hist_sv['l2_error']),
        'eval_epochs': np.array(hist_sv['eval_epochs']),
        'total_time_sec': time_sv,
        'ms_per_epoch': (time_sv / (EPOCHS - 2)) * 1000,
        'best_l2_error': float(min(hist_sv['l2_error'])),
        'final_l2_error': float(hist_sv['l2_error'][-1]),
        'total_params': npar_sv,
    }
    all_results['SVSNN_orig'] = result_svsnn_orig
    save_results('SVSNN_orig', result_svsnn_orig, data)

    # ---- Method 2: SPINN ----
    result_spinn = train_spinn(data)
    all_results['SPINN'] = result_spinn
    save_results('SPINN', result_spinn, data)

    # ---- Method 3: SIREN ----
    k3, key = random.split(key)
    siren_params = init_siren(k3)
    result_siren = train_pointwise_method(
        'SIREN', siren_params, siren_loss_fn, siren_u, data
    )
    all_results['SIREN'] = result_siren
    save_results('SIREN', result_siren, data)

    # ---- Method 4: FourierPINN ----
    k4, key = random.split(key)
    fp_params = init_fourier_pinn(k4)
    result_fp = train_pointwise_method(
        'FourierPINN', fp_params, fourier_pinn_loss_fn, fourier_pinn_u, data
    )
    all_results['FourierPINN'] = result_fp
    save_results('FourierPINN', result_fp, data)

    # ---- Method 5: PINN ----
    k5, key = random.split(key)
    pinn_params = init_pinn(k5)
    result_pinn = train_pointwise_method(
        'PINN', pinn_params, pinn_loss_fn, pinn_u, data
    )
    all_results['PINN'] = result_pinn
    save_results('PINN', result_pinn, data)

    # ---- Save comparison table ----
    save_comparison_table(all_results)

    # ---- Print final comparison ----
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Method':<14} {'Params':>10} {'Time(s)':>10} {'Best L2':>12} {'Final L2':>12} {'ms/epoch':>10}")
    print("-" * 68)
    for name, r in all_results.items():
        print(f"{name:<14} {r['total_params']:>10,} {r['total_time_sec']:>10.1f} "
              f"{r['best_l2_error']:>12.4e} {r['final_l2_error']:>12.4e} {r['ms_per_epoch']:>10.2f}")
    print("=" * 60)
    print("\nAll results saved to:", SAVE_DIR)


CASE_INFO = {"id": "case6", "title": "2D Helmholtz kappa=48pi", "family": "elliptic",
             "has_classical": True}


def E11_run(method, budget, seed, epochs=None, target=None, save_pred_path=None):
    import _e11common
    return _e11common.run_modular_elliptic(
        sys.modules[__name__], method, budget, seed, epochs,
        target=target, save_pred_path=save_pred_path)


if __name__ == "__main__":
    main()
