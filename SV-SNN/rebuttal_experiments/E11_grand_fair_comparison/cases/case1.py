"""
SV-SNN Acceleration Experiment: 1D Heat equation (kappa=20*pi).

Compares 6 methods:
  1. SV-SNN (accelerated) -- analytic spatial derivatives + batched jvp + grid
  2. SV-SNN (original)    -- vmap + nested grad (baseline)
  3. SPINN
  4. SIREN
  5. FourierPINN
  6. Vanilla PINN

PDE: du/dt - alpha * d2u/dx2 = 0
alpha = 1/(20*pi)^2, domain: x in [-1,1], t in [0,1]
IC: u(x,0) = sin(20*pi*x), BC: u(-1,t)=u(1,t)=0
Exact: u(x,t) = exp(-t)*sin(20*pi*x)
"""

import os
import sys
import time
import json
import csv

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jvp, value_and_grad
from functools import partial
import optax
from pyDOE import lhs

sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

KAPPA = 20.0 * np.pi
ALPHA = 1.0 / KAPPA**2
W_CHAR = KAPPA
FF_DIM = 64
SEED = 42
EPOCHS = 10000
LR = 1e-3
N_PDE = 10000
N_IC = 256
N_BC = 200
N_TEST_X = 256
N_TEST_T = 100
EVAL_EVERY = 100
NC_SPINN = 100

E11_OVR = {}  # E11 size overrides for matched budget (set by E11_run)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Data Generation
# ============================================================
def generate_training_data(seed=SEED):
    np.random.seed(seed)
    x_ic = -1.0 + 2.0 * lhs(1, N_IC)
    t_ic = np.zeros((N_IC, 1))
    u_ic = np.sin(KAPPA * x_ic)

    t_bc_left = lhs(1, N_BC // 2)
    x_bc_left = -np.ones((N_BC // 2, 1))
    u_bc_left = np.zeros((N_BC // 2, 1))

    t_bc_right = lhs(1, N_BC // 2)
    x_bc_right = np.ones((N_BC // 2, 1))
    u_bc_right = np.zeros((N_BC // 2, 1))

    x_bc = np.vstack([x_bc_left, x_bc_right])
    t_bc = np.vstack([t_bc_left, t_bc_right])
    u_bc = np.vstack([u_bc_left, u_bc_right])

    xt_pde = np.hstack([
        -1.0 + 2.0 * lhs(1, N_PDE),
        lhs(1, N_PDE)
    ])
    x_pde = xt_pde[:, 0:1]
    t_pde = xt_pde[:, 1:2]

    return (
        jnp.array(x_ic, dtype=jnp.float32),
        jnp.array(t_ic, dtype=jnp.float32),
        jnp.array(u_ic, dtype=jnp.float32),
        jnp.array(x_bc, dtype=jnp.float32),
        jnp.array(t_bc, dtype=jnp.float32),
        jnp.array(u_bc, dtype=jnp.float32),
        jnp.array(x_pde, dtype=jnp.float32),
        jnp.array(t_pde, dtype=jnp.float32),
    )


def generate_test_data():
    x = np.linspace(-1, 1, N_TEST_X)
    t = np.linspace(0, 1, N_TEST_T)
    X, T = np.meshgrid(x, t)
    U_exact = np.exp(-T) * np.sin(KAPPA * X)
    return X, T, U_exact


# ============================================================
# Utility Functions
# ============================================================
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    return tangents_out


def count_params(params):
    leaves = jax.tree.leaves(params)
    return sum(p.size for p in leaves)


def l2_relative_error(u_pred, u_exact):
    return np.sqrt(np.sum((u_pred - u_exact)**2) / np.sum(u_exact**2))


def _sample_frequencies(key, K, w_char):
    n_low = K // 4
    n_char = K // 2
    n_high = K - n_low - n_char
    k1, k2, k3 = jax.random.split(key, 3)
    freqs_low = jnp.linspace(1.0, w_char, n_low)
    freqs_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 20.0 + w_char)
    freqs_high = jax.random.uniform(k3, (n_high,), minval=w_char * 0.5, maxval=w_char)
    freqs = jnp.concatenate([freqs_low, freqs_char, freqs_high])
    return jnp.sort(freqs)


# ============================================================
# SV-SNN Common: model architecture (shared by both variants)
# ============================================================
NUM_MODES = 10
NUM_FREQ = 40
TEMPORAL_LAYERS = 4
TEMPORAL_HIDDEN = 10


def init_svsnn_params(key):
    keys = jax.random.split(key, NUM_MODES * 3 + 1)
    ki = 0
    spatial_params = []
    temporal_params = []
    for n in range(NUM_MODES):
        freqs = _sample_frequencies(keys[ki], NUM_FREQ, W_CHAR)
        ki += 1
        cos_c = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
        ki += 1
        sin_c = jax.random.normal(keys[ki], (NUM_FREQ,)) * 0.1
        ki += 1
        bias = jnp.zeros(1)
        spatial_params.append({'freqs': freqs, 'cos_c': cos_c, 'sin_c': sin_c, 'bias': bias})

    key_t = keys[ki]
    for n in range(NUM_MODES):
        key_t, *subkeys = jax.random.split(key_t, TEMPORAL_LAYERS * 2 + 1)
        layers = []
        in_dim = 1
        for l in range(TEMPORAL_LAYERS - 1):
            w = jax.random.normal(subkeys[2*l], (in_dim, TEMPORAL_HIDDEN)) * jnp.sqrt(2.0 / (in_dim + TEMPORAL_HIDDEN))
            b = jnp.zeros(TEMPORAL_HIDDEN)
            layers.append({'w': w, 'b': b})
            in_dim = TEMPORAL_HIDDEN
        w = jax.random.normal(subkeys[2*(TEMPORAL_LAYERS-1)], (in_dim, 1)) * jnp.sqrt(2.0 / (in_dim + 1))
        b = jnp.zeros(1)
        layers.append({'w': w, 'b': b})
        temporal_params.append(layers)

    mode_coeffs = jax.random.normal(jax.random.PRNGKey(999), (NUM_MODES,)) * 0.1
    return {
        'spatial': spatial_params,
        'temporal': temporal_params,
        'mode_coeffs': mode_coeffs,
    }


def spatial_forward(sp, x):
    wx = sp['freqs'][None, :] * x
    out = jnp.sum(sp['cos_c'] * jnp.cos(wx) + sp['sin_c'] * jnp.sin(wx), axis=1, keepdims=True)
    return out + sp['bias']


def spatial_second_deriv(sp, x):
    """X_n''(x) = sum_k [-a_k w_k^2 cos(w_k x) - b_k w_k^2 sin(w_k x)]"""
    wx = sp['freqs'][None, :] * x
    w2 = sp['freqs'] ** 2
    return jnp.sum(-w2 * (sp['cos_c'] * jnp.cos(wx) + sp['sin_c'] * jnp.sin(wx)), axis=1, keepdims=True)


def temporal_forward(layers, t):
    h = t
    for l in layers[:-1]:
        h = jnp.tanh(h @ l['w'] + l['b'])
    h = h @ layers[-1]['w'] + layers[-1]['b']
    return h


def svsnn_forward(params, x, t):
    u = jnp.zeros_like(x)
    for n in range(NUM_MODES):
        sp = params['spatial'][n]
        sp_sg = {**sp, 'freqs': jax.lax.stop_gradient(sp['freqs'])}
        X_n = spatial_forward(sp_sg, x)
        T_n = temporal_forward(params['temporal'][n], t)
        u = u + params['mode_coeffs'][n] * X_n * T_n
    return u


def svsnn_forward_grid(params, x, t):
    """x: (Nx,1), t: (Nt,1) -> (Nx, Nt)"""
    result = jnp.zeros((x.shape[0], t.shape[0]))
    for n in range(NUM_MODES):
        sp = params['spatial'][n]
        sp_sg = {**sp, 'freqs': jax.lax.stop_gradient(sp['freqs'])}
        X_n = spatial_forward(sp_sg, x)
        T_n = temporal_forward(params['temporal'][n], t)
        result = result + params['mode_coeffs'][n] * (X_n @ T_n.T)
    return result


def temporal_forward_with_deriv(layers, t):
    """Compute T_n(t) and T_n'(t) simultaneously via manual chain rule.
    Eliminates jvp from the loss, so value_and_grad only does first-order backprop.
    """
    h = t
    dh_dt = jnp.ones_like(t)

    for l in layers[:-1]:
        pre = h @ l['w'] + l['b']
        h = jnp.tanh(pre)
        dh_dt = (1 - h**2) * (dh_dt @ l['w'])

    T_n = h @ layers[-1]['w'] + layers[-1]['b']
    T_n_dot = dh_dt @ layers[-1]['w']
    return T_n, T_n_dot


def _single_temporal_fwd_deriv(stacked_layer_w, stacked_layer_b, t):
    """Single-mode temporal forward + derivative for vmap.
    stacked_layer_w: list of weight arrays for one mode
    stacked_layer_b: list of bias arrays for one mode
    t: (Nt, 1)
    Returns T_n: (Nt, 1), T_n_dot: (Nt, 1)
    """
    h = t
    dh_dt = jnp.ones_like(t)
    n_layers = stacked_layer_w.shape[0]

    def body(i, carry):
        h, dh_dt = carry
        w = stacked_layer_w[i]
        b = stacked_layer_b[i]
        pre = h @ w + b
        h_new = jnp.tanh(pre)
        dh_dt_new = (1 - h_new**2) * (dh_dt @ w)
        return h_new, dh_dt_new

    h, dh_dt = jax.lax.fori_loop(0, n_layers - 1, body, (h, dh_dt))
    w_last = stacked_layer_w[n_layers - 1]
    b_last = stacked_layer_b[n_layers - 1]
    T_n = h @ w_last + b_last
    T_n_dot = dh_dt @ w_last
    return T_n, T_n_dot


# ============================================================
# Method 1: SV-SNN ACCELERATED
# ============================================================
def run_svsnn_accelerated(x_ic, t_ic, u_ic, x_bc, t_bc, u_bc,
                          X_test, T_test, U_exact):
    print("\n" + "="*60)
    print("Training SV-SNN (ACCELERATED)")
    print("  Vectorized modes, analytic X_n'', manual T_n', separable grid")
    print("  stop_gradient on freqs, zero AD in PDE residual")
    print("="*60)

    key = random.PRNGKey(SEED)
    params = init_svsnn_params(key)

    xc = jnp.linspace(-1, 1, NC_SPINN).reshape(-1, 1)
    tc = jnp.linspace(0, 1, NC_SPINN).reshape(-1, 1)

    def vectorized_pde_residual(params, x_grid, t_grid):
        """Fully vectorized PDE residual.
        Spatial: all modes at once via batched trig (single cos/sin computation).
        Temporal: loop over 10 small MLPs with manual chain rule.
        """
        Nx = x_grid.shape[0]
        Nt = t_grid.shape[0]

        all_freqs, all_cos_c, all_sin_c, all_bias = _stack_spatial(params)
        coeffs = params['mode_coeffs']
        X_all, trig_terms = _compute_X_all(
            x_grid.squeeze(), all_freqs, all_cos_c, all_sin_c, all_bias)

        w2 = all_freqs ** 2
        X_dd_all = jnp.sum(-w2[None, :, :] * trig_terms, axis=-1)

        return u_t_grid - ALPHA * u_xx_grid

    def _stack_spatial(params):
        all_freqs = jnp.stack([jax.lax.stop_gradient(params['spatial'][n]['freqs'])
                               for n in range(NUM_MODES)])
        all_cos_c = jnp.stack([params['spatial'][n]['cos_c'] for n in range(NUM_MODES)])
        all_sin_c = jnp.stack([params['spatial'][n]['sin_c'] for n in range(NUM_MODES)])
        all_bias = jnp.stack([params['spatial'][n]['bias'] for n in range(NUM_MODES)])
        return all_freqs, all_cos_c, all_sin_c, all_bias

    def _stack_temporal(params):
        w_list, b_list = [], []
        for l_idx in range(TEMPORAL_LAYERS):
            w_list.append(jnp.stack([params['temporal'][n][l_idx]['w']
                                     for n in range(NUM_MODES)]))
            b_list.append(jnp.stack([params['temporal'][n][l_idx]['b']
                                     for n in range(NUM_MODES)]))
        return w_list, b_list

    def _compute_X_all(x_flat, all_freqs, all_cos_c, all_sin_c, all_bias):
        """x_flat: (N,) -> X_all: (N, M), trig_terms: (N, M, K)"""
        wx = x_flat[:, None, None] * all_freqs[None, :, :]
        cos_wx = jnp.cos(wx)
        sin_wx = jnp.sin(wx)
        trig_terms = all_cos_c[None, :, :] * cos_wx + all_sin_c[None, :, :] * sin_wx
        X_all = jnp.sum(trig_terms, axis=-1) + all_bias[None, :, 0]
        return X_all, trig_terms

    def _batched_temporal_fwd(w_list, b_list, t):
        """All modes at once via vmap. Returns (M, Nt, 1)."""
        def single_fwd(w0, b0, w1, b1, w2, b2, w3, b3):
            h = t
            for w, b in [(w0, b0), (w1, b1), (w2, b2)]:
                h = jnp.tanh(h @ w + b)
            return h @ w3 + b3
        return jax.vmap(single_fwd)(
            w_list[0], b_list[0], w_list[1], b_list[1],
            w_list[2], b_list[2], w_list[3], b_list[3])

    def _batched_temporal_fwd_deriv(w_list, b_list, t):
        """All modes at once via vmap. Returns T: (M,Nt,1), T': (M,Nt,1)."""
        def single_fwd_deriv(w0, b0, w1, b1, w2, b2, w3, b3):
            h = t
            dh = jnp.ones_like(t)
            for w, b in [(w0, b0), (w1, b1), (w2, b2)]:
                pre = h @ w + b
                h = jnp.tanh(pre)
                dh = (1 - h**2) * (dh @ w)
            return h @ w3 + b3, dh @ w3
        return jax.vmap(single_fwd_deriv)(
            w_list[0], b_list[0], w_list[1], b_list[1],
            w_list[2], b_list[2], w_list[3], b_list[3])

    def vectorized_forward(params, x, t):
        """Fully vectorized pointwise forward for IC/BC."""
        all_freqs, all_cos_c, all_sin_c, all_bias = _stack_spatial(params)
        w_list, b_list = _stack_temporal(params)
        coeffs = params['mode_coeffs']
        X_all, _ = _compute_X_all(x.squeeze(), all_freqs, all_cos_c, all_sin_c, all_bias)
        T_all = _batched_temporal_fwd(w_list, b_list, t)
        u = jnp.sum(coeffs[None, :] * X_all * T_all[:, :, 0].T, axis=-1, keepdims=True)
        return u

    def vectorized_forward_grid(params, x, t):
        """Fully vectorized grid forward: x(Nx,1), t(Nt,1) -> (Nx,Nt)."""
        all_freqs, all_cos_c, all_sin_c, all_bias = _stack_spatial(params)
        w_list, b_list = _stack_temporal(params)
        coeffs = params['mode_coeffs']
        X_all, _ = _compute_X_all(x.squeeze(), all_freqs, all_cos_c, all_sin_c, all_bias)
        T_all = _batched_temporal_fwd(w_list, b_list, t)
        cX = coeffs[None, :] * X_all
        return jnp.einsum('nm,mj->nj', cX, T_all[:, :, 0])

    def vectorized_pde_residual(params, x_grid, t_grid):
        """Fully vectorized PDE residual. No per-mode loop anywhere."""
        all_freqs, all_cos_c, all_sin_c, all_bias = _stack_spatial(params)
        w_list, b_list = _stack_temporal(params)
        coeffs = params['mode_coeffs']

        X_all, trig_terms = _compute_X_all(
            x_grid.squeeze(), all_freqs, all_cos_c, all_sin_c, all_bias)

        w2 = all_freqs ** 2
        X_dd_all = jnp.sum(-w2[None, :, :] * trig_terms, axis=-1)

        T_all, T_dot_all = _batched_temporal_fwd_deriv(w_list, b_list, t_grid)

        cX = coeffs[None, :] * X_all
        cXdd = coeffs[None, :] * X_dd_all
        u_t = jnp.einsum('nm,mj->nj', cX, T_dot_all[:, :, 0])
        u_xx = jnp.einsum('nm,mj->nj', cXdd, T_all[:, :, 0])

        return u_t - ALPHA * u_xx

    def loss_fn(params):
        u_pred_ic = vectorized_forward(params, x_ic, t_ic)
        loss_ic = jnp.mean((u_pred_ic - u_ic)**2)

        u_pred_bc = vectorized_forward(params, x_bc, t_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2)

        residual = vectorized_pde_residual(params, xc, tc)
        loss_pde = jnp.mean(residual**2)

        return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}

    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, (loss_pde, loss_ic, loss_bc) = train_step(
            params, opt_state)

        if epoch % EVAL_EVERY == 0:
            x_test_j = jnp.linspace(-1, 1, N_TEST_X).reshape(-1, 1)
            t_test_j = jnp.linspace(0, 1, N_TEST_T).reshape(-1, 1)
            u_pred_grid = np.array(vectorized_forward_grid(params, x_test_j, t_test_j)).T
            err = l2_relative_error(u_pred_grid, U_exact)
            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(loss_pde))
            history['ic_loss'].append(float(loss_ic))
            history['bc_loss'].append(float(loss_bc))
            history['l2_error'].append(float(err))
            history['eval_epochs'].append(epoch)
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch:5d} | Loss: {loss:.6e} | L2 Err: {err:.6e}", flush=True)

    total_time = time.time() - start_time
    n_params = count_params(params)

    x_test_j = jnp.linspace(-1, 1, N_TEST_X).reshape(-1, 1)
    t_test_j = jnp.linspace(0, 1, N_TEST_T).reshape(-1, 1)
    u_pred_grid = np.array(vectorized_forward_grid(params, x_test_j, t_test_j)).T

    return params, history, u_pred_grid, n_params, total_time


# ============================================================
# Method 2: SV-SNN ORIGINAL (vmap + nested grad)
# ============================================================
def run_svsnn_original(x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
                       X_test, T_test, U_exact):
    print("\n" + "="*60)
    print("Training SV-SNN (ORIGINAL)")
    print("  vmap + nested grad(grad(...)) per-point AD")
    print("="*60)

    key = random.PRNGKey(SEED)
    params = init_svsnn_params(key)

    def u_scalar(params, x_s, t_s):
        x_ = x_s.reshape(1, 1)
        t_ = t_s.reshape(1, 1)
        return svsnn_forward(params, x_, t_).squeeze()

    def pde_residual_single(params, x_s, t_s):
        u_t = jax.grad(u_scalar, argnums=2)(params, x_s, t_s)
        u_x_fn = lambda x_: u_scalar(params, x_, t_s)
        u_xx = jax.grad(jax.grad(u_x_fn))(x_s)
        return u_t - ALPHA * u_xx

    pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0))

    def loss_fn(params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        u_pred_ic = svsnn_forward(params, x_ic, t_ic)
        loss_ic = jnp.mean((u_pred_ic - u_ic)**2)

        u_pred_bc = svsnn_forward(params, x_bc, t_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2)

        res = pde_residual_batch(params, x_pde.squeeze(), t_pde.squeeze())
        loss_pde = jnp.mean(res**2)

        return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(
            params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}

    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, (loss_pde, loss_ic, loss_bc) = train_step(
            params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)

        if epoch % EVAL_EVERY == 0:
            x_test_j = jnp.linspace(-1, 1, N_TEST_X).reshape(-1, 1)
            t_test_j = jnp.linspace(0, 1, N_TEST_T).reshape(-1, 1)
            u_pred_grid = np.array(svsnn_forward_grid(params, x_test_j, t_test_j)).T
            err = l2_relative_error(u_pred_grid, U_exact)
            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(loss_pde))
            history['ic_loss'].append(float(loss_ic))
            history['bc_loss'].append(float(loss_bc))
            history['l2_error'].append(float(err))
            history['eval_epochs'].append(epoch)
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch:5d} | Loss: {loss:.6e} | L2 Err: {err:.6e}", flush=True)

    total_time = time.time() - start_time
    n_params = count_params(params)

    x_test_j = jnp.linspace(-1, 1, N_TEST_X).reshape(-1, 1)
    t_test_j = jnp.linspace(0, 1, N_TEST_T).reshape(-1, 1)
    u_pred_grid = np.array(svsnn_forward_grid(params, x_test_j, t_test_j)).T

    return params, history, u_pred_grid, n_params, total_time


# ============================================================
# Method 3: SPINN
# ============================================================
def run_spinn(X_test, T_test, U_exact):
    print("\n" + "="*60)
    print("Training SPINN")
    print("="*60)

    features = E11_OVR.get('spinn_features', 64)
    n_layers = E11_OVR.get('spinn_n_layers', 4)
    r = E11_OVR.get('spinn_r', 64)
    _ff = E11_OVR.get('spinn_ff', FF_DIM)
    key = random.PRNGKey(SEED)

    ff_input_dim = 2 * _ff

    def init_modified_mlp(key, d_in, features, n_layers, d_out):
        params = {}
        key, k1, k2 = random.split(key, 3)
        params['U_w'] = random.normal(k1, (d_in, features)) * jnp.sqrt(2.0 / d_in)
        params['U_b'] = jnp.zeros(features)
        params['V_w'] = random.normal(k2, (d_in, features)) * jnp.sqrt(2.0 / d_in)
        params['V_b'] = jnp.zeros(features)
        key, k = random.split(key)
        params['H_w'] = random.normal(k, (d_in, features)) * jnp.sqrt(2.0 / d_in)
        params['H_b'] = jnp.zeros(features)
        params['layers'] = []
        for _ in range(n_layers - 1):
            key, k = random.split(key)
            w = random.normal(k, (features, features)) * jnp.sqrt(2.0 / features)
            b = jnp.zeros(features)
            params['layers'].append((w, b))
        key, k = random.split(key)
        params['out_w'] = random.normal(k, (features, d_out)) * jnp.sqrt(2.0 / features)
        return params

    key, k1, k2, k3, k4 = random.split(key, 5)
    params = {
        'branch_x': init_modified_mlp(k1, ff_input_dim, features, n_layers, r),
        'branch_t': init_modified_mlp(k2, ff_input_dim, features, n_layers, r),
        'W_x': _sample_frequencies(k3, _ff, W_CHAR).reshape(1, -1),
        'W_t': random.normal(k4, (1, _ff)),
    }

    def fourier_embed(coord, W):
        return jnp.concatenate([jnp.sin(coord @ W), jnp.cos(coord @ W)], axis=-1)

    def modified_mlp_forward(p, x):
        U = jnp.tanh(x @ p['U_w'] + p['U_b'])
        V = jnp.tanh(x @ p['V_w'] + p['V_b'])
        H = jnp.tanh(x @ p['H_w'] + p['H_b'])
        for (w, b) in p['layers']:
            Z = jnp.tanh(H @ w + b)
            H = (1 - Z) * U + Z * V
        return H @ p['out_w']

    def forward_spinn(params, x, t):
        x_emb = fourier_embed(x, params['W_x'])
        t_emb = fourier_embed(t, params['W_t'])
        bx = modified_mlp_forward(params['branch_x'], x_emb)
        bt = modified_mlp_forward(params['branch_t'], t_emb)
        return bx @ bt.T

    xc = jnp.linspace(-1, 1, NC_SPINN).reshape(-1, 1)
    tc = jnp.linspace(0, 1, NC_SPINN).reshape(-1, 1)
    u_ic_spinn = jnp.sin(KAPPA * xc)

    def loss_fn(params):
        t_zero = jnp.array([[0.0]])
        u_pred_ic = forward_spinn(params, xc, t_zero)
        loss_ic = jnp.mean((u_pred_ic - u_ic_spinn)**2)

        x_left = jnp.array([[-1.0]])
        x_right = jnp.array([[1.0]])
        u_bc_left = forward_spinn(params, x_left, tc)
        u_bc_right = forward_spinn(params, x_right, tc)
        loss_bc = jnp.mean(u_bc_left**2) + jnp.mean(u_bc_right**2)

        def f_t(t):
            return forward_spinn(params, xc, t)
        u_t = jvp(f_t, (tc,), (jnp.ones_like(tc),))[1]

        def f_x(x):
            return forward_spinn(params, x, tc)
        u_xx = hvp_fwdfwd(f_x, (xc,), (jnp.ones_like(xc),))

        residual = u_t - ALPHA * u_xx
        loss_pde = jnp.mean(residual**2)

        return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, (loss_pde, loss_ic, loss_bc) = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0:
            x_test_j = jnp.linspace(-1, 1, N_TEST_X).reshape(-1, 1)
            t_test_j = jnp.linspace(0, 1, N_TEST_T).reshape(-1, 1)
            u_pred_grid = np.array(forward_spinn(params, x_test_j, t_test_j)).T
            err = l2_relative_error(u_pred_grid, U_exact)
            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(loss_pde))
            history['ic_loss'].append(float(loss_ic))
            history['bc_loss'].append(float(loss_bc))
            history['l2_error'].append(float(err))
            history['eval_epochs'].append(epoch)
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch:5d} | Loss: {loss:.6e} | L2 Err: {err:.6e}", flush=True)

    total_time = time.time() - start_time
    n_params = count_params(params)

    x_test_j = jnp.linspace(-1, 1, N_TEST_X).reshape(-1, 1)
    t_test_j = jnp.linspace(0, 1, N_TEST_T).reshape(-1, 1)
    u_pred_grid = np.array(forward_spinn(params, x_test_j, t_test_j)).T

    return params, history, u_pred_grid, n_params, total_time


# ============================================================
# Method 4: SIREN
# ============================================================
def run_siren(x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
              X_test, T_test, U_exact):
    print("\n" + "="*60)
    print("Training SIREN")
    print("="*60)

    key = random.PRNGKey(SEED)
    key, k_wx, k_wt = random.split(key, 3)
    _ff = E11_OVR.get('ff', FF_DIM)
    _hid = E11_OVR.get('hidden', 128)
    _nh = E11_OVR.get('n_hidden', 4)
    siren_W_x = _sample_frequencies(k_wx, _ff, W_CHAR).reshape(1, -1)
    siren_W_t = random.normal(k_wt, (1, _ff))
    ff_input_dim_siren = 4 * _ff
    layers = [ff_input_dim_siren] + [_hid] * _nh + [1]

    def init_siren(key, layers):
        params = []
        for i in range(len(layers) - 1):
            key, k = random.split(key)
            d_in, d_out = layers[i], layers[i + 1]
            std = jnp.sqrt(2.0 / d_in)
            w = random.normal(k, (d_in, d_out)) * std
            b = jnp.zeros(d_out)
            params.append((w, b))
        return params

    siren_mlp = init_siren(key, layers)
    params = {'mlp': siren_mlp, 'W_x': siren_W_x, 'W_t': siren_W_t}

    def forward_siren(params, x, t):
        Hx = jnp.concatenate([jnp.sin(x @ params['W_x']), jnp.cos(x @ params['W_x'])], axis=-1)
        Ht = jnp.concatenate([jnp.sin(t @ params['W_t']), jnp.cos(t @ params['W_t'])], axis=-1)
        h = jnp.concatenate([Hx, Ht], axis=-1)
        for i, (w, b) in enumerate(params['mlp']):
            h = h @ w + b
            if i < len(params['mlp']) - 1:
                h = jnp.sin(h)
        return h

    def loss_fn(params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        u_pred_ic = forward_siren(params, x_ic, t_ic)
        loss_ic = jnp.mean((u_pred_ic - u_ic)**2)

        u_pred_bc = forward_siren(params, x_bc, t_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2)

        u_t = jvp(lambda t: forward_siren(params, x_pde, t), (t_pde,), (jnp.ones_like(t_pde),))[1]
        u_xx = hvp_fwdfwd(lambda x: forward_siren(params, x, t_pde), (x_pde,), (jnp.ones_like(x_pde),))
        residual = u_t - ALPHA * u_xx
        loss_pde = jnp.mean(residual**2)

        return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(
            params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, (loss_pde, loss_ic, loss_bc) = train_step(
            params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)

        if epoch % EVAL_EVERY == 0:
            X_flat = X_test.flatten()[:, None]
            T_flat = T_test.flatten()[:, None]
            u_pred = np.array(forward_siren(params, jnp.array(X_flat), jnp.array(T_flat)))
            u_pred_grid = u_pred.reshape(N_TEST_T, N_TEST_X)
            err = l2_relative_error(u_pred_grid, U_exact)
            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(loss_pde))
            history['ic_loss'].append(float(loss_ic))
            history['bc_loss'].append(float(loss_bc))
            history['l2_error'].append(float(err))
            history['eval_epochs'].append(epoch)
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch:5d} | Loss: {loss:.6e} | L2 Err: {err:.6e}", flush=True)

    total_time = time.time() - start_time
    n_params = count_params(params)

    X_flat = X_test.flatten()[:, None]
    T_flat = T_test.flatten()[:, None]
    u_pred = np.array(forward_siren(params, jnp.array(X_flat), jnp.array(T_flat)))
    u_pred_grid = u_pred.reshape(N_TEST_T, N_TEST_X)

    return params, history, u_pred_grid, n_params, total_time


# ============================================================
# Method 5: FourierPINN
# ============================================================
def run_fourierpinn(x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
                    X_test, T_test, U_exact):
    print("\n" + "="*60)
    print("Training FourierPINN")
    print("="*60)

    _ff = E11_OVR.get('ff', FF_DIM)
    hidden_layers = [E11_OVR.get('hidden', 128)] * E11_OVR.get('n_hidden', 3) + [1]
    key = random.PRNGKey(SEED)

    key, k1, k2 = random.split(key, 3)
    W_x = _sample_frequencies(k1, _ff, W_CHAR).reshape(1, -1)
    W_t = random.normal(k2, (1, _ff))

    input_dim = 4 * _ff
    mlp_layers = [input_dim] + hidden_layers

    def init_mlp(key, layers):
        params = []
        for i in range(len(layers) - 1):
            key, k = random.split(key)
            d_in, d_out = layers[i], layers[i + 1]
            w = random.normal(k, (d_in, d_out)) * jnp.sqrt(2.0 / d_in)
            b = jnp.zeros(d_out)
            params.append((w, b))
        return params

    key, k = random.split(key)
    mlp_params = init_mlp(k, mlp_layers)
    params = {'W_x': W_x, 'W_t': W_t, 'mlp': mlp_params}

    def forward_fpinn(params, x, t):
        Hx = jnp.concatenate([jnp.sin(x @ params['W_x']), jnp.cos(x @ params['W_x'])], axis=-1)
        Ht = jnp.concatenate([jnp.sin(t @ params['W_t']), jnp.cos(t @ params['W_t'])], axis=-1)
        h = jnp.concatenate([Hx, Ht], axis=-1)
        for i, (w, b) in enumerate(params['mlp']):
            h = h @ w + b
            if i < len(params['mlp']) - 1:
                h = jnp.tanh(h)
        return h

    def loss_fn(params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        u_pred_ic = forward_fpinn(params, x_ic, t_ic)
        loss_ic = jnp.mean((u_pred_ic - u_ic)**2)

        u_pred_bc = forward_fpinn(params, x_bc, t_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2)

        u_t = jvp(lambda t: forward_fpinn(params, x_pde, t), (t_pde,), (jnp.ones_like(t_pde),))[1]
        u_xx = hvp_fwdfwd(lambda x: forward_fpinn(params, x, t_pde), (x_pde,), (jnp.ones_like(x_pde),))
        residual = u_t - ALPHA * u_xx
        loss_pde = jnp.mean(residual**2)

        return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(
            params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, (loss_pde, loss_ic, loss_bc) = train_step(
            params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)

        if epoch % EVAL_EVERY == 0:
            X_flat = X_test.flatten()[:, None]
            T_flat = T_test.flatten()[:, None]
            u_pred = np.array(forward_fpinn(params, jnp.array(X_flat), jnp.array(T_flat)))
            u_pred_grid = u_pred.reshape(N_TEST_T, N_TEST_X)
            err = l2_relative_error(u_pred_grid, U_exact)
            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(loss_pde))
            history['ic_loss'].append(float(loss_ic))
            history['bc_loss'].append(float(loss_bc))
            history['l2_error'].append(float(err))
            history['eval_epochs'].append(epoch)
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch:5d} | Loss: {loss:.6e} | L2 Err: {err:.6e}", flush=True)

    total_time = time.time() - start_time
    n_params = count_params(params)

    X_flat = X_test.flatten()[:, None]
    T_flat = T_test.flatten()[:, None]
    u_pred = np.array(forward_fpinn(params, jnp.array(X_flat), jnp.array(T_flat)))
    u_pred_grid = u_pred.reshape(N_TEST_T, N_TEST_X)

    return params, history, u_pred_grid, n_params, total_time


# ============================================================
# Method 6: Vanilla PINN
# ============================================================
def run_pinn(x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde,
             X_test, T_test, U_exact):
    print("\n" + "="*60)
    print("Training Vanilla PINN")
    print("="*60)

    layers = [2] + [E11_OVR.get('hidden', 128)] * E11_OVR.get('n_hidden', 4) + [1]
    key = random.PRNGKey(SEED)

    def init_mlp(key, layers):
        params = []
        for i in range(len(layers) - 1):
            key, k = random.split(key)
            d_in, d_out = layers[i], layers[i + 1]
            w = random.normal(k, (d_in, d_out)) * jnp.sqrt(2.0 / d_in)
            key, k = random.split(key)
            b = jnp.zeros(d_out)
            params.append((w, b))
        return params

    params = init_mlp(key, layers)

    def forward_pinn(params, x, t):
        h = jnp.concatenate([x, t], axis=-1)
        for i, (w, b) in enumerate(params):
            h = h @ w + b
            if i < len(params) - 1:
                h = jnp.tanh(h)
        return h

    def loss_fn(params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        u_pred_ic = forward_pinn(params, x_ic, t_ic)
        loss_ic = jnp.mean((u_pred_ic - u_ic)**2)

        u_pred_bc = forward_pinn(params, x_bc, t_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2)

        u_t = jvp(lambda t: forward_pinn(params, x_pde, t), (t_pde,), (jnp.ones_like(t_pde),))[1]
        u_xx = hvp_fwdfwd(lambda x: forward_pinn(params, x, t_pde), (x_pde,), (jnp.ones_like(x_pde),))
        residual = u_t - ALPHA * u_xx
        loss_pde = jnp.mean(residual**2)

        return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(
            params, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, (loss_pde, loss_ic, loss_bc) = train_step(
            params, opt_state, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde)

        if epoch % EVAL_EVERY == 0:
            X_flat = X_test.flatten()[:, None]
            T_flat = T_test.flatten()[:, None]
            u_pred = np.array(forward_pinn(params, jnp.array(X_flat), jnp.array(T_flat)))
            u_pred_grid = u_pred.reshape(N_TEST_T, N_TEST_X)
            err = l2_relative_error(u_pred_grid, U_exact)
            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(loss_pde))
            history['ic_loss'].append(float(loss_ic))
            history['bc_loss'].append(float(loss_bc))
            history['l2_error'].append(float(err))
            history['eval_epochs'].append(epoch)
            if epoch % 1000 == 0:
                print(f"  Epoch {epoch:5d} | Loss: {loss:.6e} | L2 Err: {err:.6e}", flush=True)

    total_time = time.time() - start_time
    n_params = count_params(params)

    X_flat = X_test.flatten()[:, None]
    T_flat = T_test.flatten()[:, None]
    u_pred = np.array(forward_pinn(params, jnp.array(X_flat), jnp.array(T_flat)))
    u_pred_grid = u_pred.reshape(N_TEST_T, N_TEST_X)

    return params, history, u_pred_grid, n_params, total_time


# ============================================================
# Save Results
# ============================================================
def save_results(name, params, history, u_pred, U_exact, X_test, T_test,
                 n_params, total_time):
    import pickle
    with open(os.path.join(SAVE_DIR, f"{name}_params.pkl"), 'wb') as f:
        pickle.dump(params, f)

    np.savez(os.path.join(SAVE_DIR, f"{name}_history.npz"),
             total_loss=np.array(history['total_loss']),
             pde_loss=np.array(history['pde_loss']),
             ic_loss=np.array(history['ic_loss']),
             bc_loss=np.array(history['bc_loss']),
             l2_error=np.array(history['l2_error']),
             eval_epochs=np.array(history['eval_epochs']))

    np.savez(os.path.join(SAVE_DIR, f"{name}_prediction.npz"),
             u_pred=u_pred, u_exact=U_exact, X=X_test, T=T_test)

    summary = {
        'method': name,
        'total_params': int(n_params),
        'total_time_sec': round(total_time, 2),
        'best_l2_error': float(min(history['l2_error'])),
        'final_l2_error': float(history['l2_error'][-1]),
        'ms_per_epoch': round(total_time / EPOCHS * 1000, 2),
    }
    with open(os.path.join(SAVE_DIR, f"{name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Main
# ============================================================
def method_done(name):
    return os.path.exists(os.path.join(SAVE_DIR, f"{name}_summary.json"))


CASE_INFO = {"id": "case1", "title": "1D Heat kappa=20pi (x,t)", "family": "heat",
             "has_classical": False}

_ARCH_HEAT = dict(in_dim=2, out_dim=1, n_coord=2, spinn_n_branch=2, per_out_weight=False)


def E11_run(method, budget, seed, epochs=None, target=None, save_pred_path=None):
    import numpy as _np
    import _e11common
    g = globals()
    g["SEED"] = seed
    if epochs is not None:
        g["EPOCHS"] = epochs
    g["E11_OVR"] = {}
    EP = g["EPOCHS"]
    matched_within = None
    tgt = None
    if method != "SVSNN" and budget == "matched":
        assert target is not None
        _, matched_within = _e11common.set_matched_ovr(
            sys.modules[__name__], method, target, seed, _ARCH_HEAT)
        tgt = target

    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde = generate_training_data(seed)
    X_test, T_test, U_exact = generate_test_data()

    if method == "SVSNN":
        params, hist, u_pred, n_params, tt = run_svsnn_accelerated(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, X_test, T_test, U_exact)
        n_coll = NC_SPINN * NC_SPINN
    elif method == "SPINN":
        params, hist, u_pred, n_params, tt = run_spinn(X_test, T_test, U_exact)
        n_coll = NC_SPINN * NC_SPINN
    elif method == "SIREN":
        params, hist, u_pred, n_params, tt = run_siren(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, X_test, T_test, U_exact)
        n_coll = int(N_PDE)
    elif method == "FourierPINN":
        params, hist, u_pred, n_params, tt = run_fourierpinn(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, X_test, T_test, U_exact)
        n_coll = int(N_PDE)
    elif method == "PINN":
        params, hist, u_pred, n_params, tt = run_pinn(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, X_test, T_test, U_exact)
        n_coll = int(N_PDE)
    else:
        raise ValueError(method)

    best = float(min(hist["l2_error"]))
    final = float(hist["l2_error"][-1])
    rec = _e11common.harness.normalize_record(
        method, budget, seed, params=int(n_params), best_l2=best, final_l2=final,
        train_time_sec=float(tt), n_epochs=EP, n_collocation=n_coll,
        inference_ms=float("nan"), target_params=tgt, matched_within_tol=matched_within)
    if save_pred_path is not None:
        _np.savez(save_pred_path, u_pred=_np.asarray(u_pred), u_exact=_np.asarray(U_exact),
                  X=_np.asarray(X_test), T=_np.asarray(T_test))
    return rec


if __name__ == "__main__":
    print("="*60)
    print("SV-SNN Acceleration Experiment")
    print(f"1D Heat Equation: KAPPA={KAPPA:.4f}, ALPHA={ALPHA:.6e}")
    print(f"EPOCHS={EPOCHS}, LR={LR}")
    print("="*60)

    x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde = generate_training_data()
    X_test, T_test, U_exact = generate_test_data()

    method_runners = [
        ('SVSNN_accel', lambda: run_svsnn_accelerated(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, X_test, T_test, U_exact)),
        ('SVSNN_original', lambda: run_svsnn_original(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, X_test, T_test, U_exact)),
        ('SPINN', lambda: run_spinn(X_test, T_test, U_exact)),
        ('SIREN', lambda: run_siren(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, X_test, T_test, U_exact)),
        ('FourierPINN', lambda: run_fourierpinn(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, X_test, T_test, U_exact)),
        ('PINN', lambda: run_pinn(
            x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, x_pde, t_pde, X_test, T_test, U_exact)),
    ]

    summaries = []
    for name, runner in method_runners:
        if method_done(name):
            print(f"\n  [SKIP] {name} already completed.")
            with open(os.path.join(SAVE_DIR, f"{name}_summary.json")) as f:
                summaries.append(json.load(f))
            continue
        params, history, u_pred, n_params, total_time = runner()
        summaries.append(save_results(
            name, params, history, u_pred, U_exact, X_test, T_test, n_params, total_time))

    csv_path = os.path.join(SAVE_DIR, "comparison_table.csv")
    fieldnames = ['method', 'total_params', 'total_time_sec', 'best_l2_error',
                  'final_l2_error', 'ms_per_epoch']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Method':<20} {'Params':>8} {'Time(s)':>8} {'Best L2':>12} {'Final L2':>12} {'ms/epoch':>10}")
    print("-"*72)
    for s in summaries:
        print(f"{s['method']:<20} {s['total_params']:>8d} {s['total_time_sec']:>8.1f} "
              f"{s['best_l2_error']:>12.6e} {s['final_l2_error']:>12.6e} {s['ms_per_epoch']:>10.2f}")
    print("="*70)
    print(f"\nAll results saved to: {SAVE_DIR}")
