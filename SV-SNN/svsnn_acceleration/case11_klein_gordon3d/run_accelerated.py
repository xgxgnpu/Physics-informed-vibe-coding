"""
SV-SNN Acceleration - Case 11: Klein-Gordon 3D (2+1D) High Frequency
=====================================================================
PDE (linear massive Klein-Gordon):
  u_tt - u_xx - u_yy + u = f(x, y, t)

Domain: x, y in [-1, 1],  t in [0, 1]
IC:  u(x,y,0) = sin(kx)*sin(ky),  u_t(x,y,0) = 0
BC:  u = 0 on all spatial boundaries (natural zeros of sin)
Exact:  u(x,y,t) = sin(kx)*sin(ky)*cosh(t)
        kappa = 4*pi

The solution represents a growing mode of the massive Klein-Gordon equation
with high spatial frequency (2 oscillation cycles per dimension in [-1,1]).
cosh(t) temporal evolution is smooth and monotone.

Source:
  f = (2 + 2*kappa^2)*sin(kx)*sin(ky)*cosh(t)

Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, PINN
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
import optax
from pyDOE import lhs

sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_command_buffer= "
    "--xla_gpu_enable_cublaslt=false"
)

# ============================================================
# Configuration
# ============================================================
KAPPA = 4.0 * np.pi
PDE_COEFF = 2.0 + 2.0 * KAPPA**2
PDE_NORM = PDE_COEFF
W_CHAR = KAPPA
SEED = 42
EPOCHS = 50000
LR = 1e-3
N_PDE = 10000
N_IC = 5000
N_BC = 1000
N_TEST = 50
EVAL_EVERY = 500
NC_SPINN = 50
W_IC = 100.0
W_BC = 100.0

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Exact Solution and Source Term
# ============================================================
def exact_u(x, y, t):
    return jnp.sin(KAPPA * x) * jnp.sin(KAPPA * y) * jnp.cosh(t)


def exact_u_np(x, y, t):
    return np.sin(KAPPA * x) * np.sin(KAPPA * y) * np.cosh(t)


def source_term(x, y, t):
    return PDE_COEFF * jnp.sin(KAPPA * x) * jnp.sin(KAPPA * y) * jnp.cosh(t)


def source_term_np(x, y, t):
    return PDE_COEFF * np.sin(KAPPA * x) * np.sin(KAPPA * y) * np.cosh(t)


# ============================================================
# Data Generation
# ============================================================
def generate_data(seed=SEED):
    np.random.seed(seed)

    pts_ic = lhs(2, samples=N_IC)
    x_ic = -1.0 + 2.0 * pts_ic[:, 0:1]
    y_ic = -1.0 + 2.0 * pts_ic[:, 1:2]
    t_ic = np.zeros((N_IC, 1))
    u_ic = np.sin(KAPPA * x_ic) * np.sin(KAPPA * y_ic)

    n_per_face = N_BC // 4
    t_bc = np.random.uniform(0, 1, (n_per_face, 1))
    x_bc_left = -np.ones((n_per_face, 1))
    x_bc_right = np.ones((n_per_face, 1))
    y_bc_left = np.random.uniform(-1, 1, (n_per_face, 1))
    y_bc_right = np.random.uniform(-1, 1, (n_per_face, 1))

    t_bc2 = np.random.uniform(0, 1, (n_per_face, 1))
    y_bc_bottom = -np.ones((n_per_face, 1))
    y_bc_top = np.ones((n_per_face, 1))
    x_bc_bottom = np.random.uniform(-1, 1, (n_per_face, 1))
    x_bc_top = np.random.uniform(-1, 1, (n_per_face, 1))

    x_bc_all = np.vstack([x_bc_left, x_bc_right, x_bc_bottom, x_bc_top])
    y_bc_all = np.vstack([y_bc_left, y_bc_right, y_bc_bottom, y_bc_top])
    t_bc_all = np.vstack([t_bc, t_bc, t_bc2, t_bc2])
    u_bc_all = np.zeros_like(x_bc_all)

    pts = lhs(3, samples=N_PDE)
    x_pde = -1.0 + 2.0 * pts[:, 0:1]
    y_pde = -1.0 + 2.0 * pts[:, 1:2]
    t_pde = pts[:, 2:3]

    xt = np.linspace(-1, 1, N_TEST)
    yt = np.linspace(-1, 1, N_TEST)
    tt = np.linspace(0, 1, N_TEST)
    XT, YT, TT = np.meshgrid(xt, yt, tt, indexing='ij')
    u_exact = exact_u_np(XT, YT, TT)

    f32 = jnp.float32
    data = {
        'x_ic': jnp.array(x_ic, dtype=f32), 'y_ic': jnp.array(y_ic, dtype=f32),
        't_ic': jnp.array(t_ic, dtype=f32), 'u_ic': jnp.array(u_ic, dtype=f32),
        'x_bc': jnp.array(x_bc_all, dtype=f32), 'y_bc': jnp.array(y_bc_all, dtype=f32),
        't_bc': jnp.array(t_bc_all, dtype=f32), 'u_bc': jnp.array(u_bc_all, dtype=f32),
        'x_pde': jnp.array(x_pde, dtype=f32), 'y_pde': jnp.array(y_pde, dtype=f32),
        't_pde': jnp.array(t_pde, dtype=f32),
        'X_test': XT, 'Y_test': YT, 'T_test': TT, 'u_exact': u_exact,
        'x_test_flat': jnp.array(XT.reshape(-1, 1), dtype=f32),
        'y_test_flat': jnp.array(YT.reshape(-1, 1), dtype=f32),
        't_test_flat': jnp.array(TT.reshape(-1, 1), dtype=f32),
    }

    xc = np.linspace(-1, 1, NC_SPINN).reshape(-1, 1)
    yc = np.linspace(-1, 1, NC_SPINN).reshape(-1, 1)
    tc = np.linspace(0, 1, NC_SPINN).reshape(-1, 1)
    Xg, Yg = np.meshgrid(xc.ravel(), yc.ravel(), indexing='ij')
    data['spinn'] = {
        'xc': jnp.array(xc, dtype=f32),
        'yc': jnp.array(yc, dtype=f32),
        'tc': jnp.array(tc, dtype=f32),
        'u_ic_grid': jnp.array(np.sin(KAPPA * Xg) * np.sin(KAPPA * Yg), dtype=f32),
    }
    data['x_test_1d'] = jnp.array(xt.reshape(-1, 1), dtype=f32)
    data['y_test_1d'] = jnp.array(yt.reshape(-1, 1), dtype=f32)
    data['t_test_1d'] = jnp.array(tt.reshape(-1, 1), dtype=f32)
    return data


# ============================================================
# Utilities
# ============================================================
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), (tangents,))[1]
    primals_out, tangents_out = jvp(g, (primals,), (tangents,))
    if return_primals:
        return primals_out, tangents_out
    return tangents_out


def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params) if hasattr(p, 'size'))


def l2_relative_error(u_pred, u_exact):
    return float(np.sqrt(np.sum((u_pred - u_exact)**2) / (np.sum(u_exact**2) + 1e-30)))


def _sample_frequencies(key, K):
    n_low = K // 4
    n_char = K // 2
    n_high = K - n_low - n_char
    k1, k2, k3 = jax.random.split(key, 3)
    freqs_low = jnp.linspace(1.0, W_CHAR, n_low)
    freqs_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 2.0 + W_CHAR)
    freqs_high = jax.random.uniform(k3, (n_high,), minval=W_CHAR * 0.5, maxval=W_CHAR * 1.5)
    return jnp.sort(jnp.concatenate([freqs_low, freqs_char, freqs_high]))


# ============================================================
# SV-SNN Common Architecture
# ============================================================
NUM_MODES = 10
NUM_FREQ = 40
TEMPORAL_LAYERS = 4
TEMPORAL_HIDDEN = 20


def init_svsnn_params(key):
    keys = jax.random.split(key, NUM_MODES * 7 + 1)
    ki = 0
    spatial_x, spatial_y = [], []
    for _ in range(NUM_MODES):
        for s_list in [spatial_x, spatial_y]:
            s_list.append({
                'freqs': _sample_frequencies(keys[ki], NUM_FREQ),
                'cos_c': jax.random.normal(keys[ki + 1], (NUM_FREQ,)) * 0.1,
                'sin_c': jax.random.normal(keys[ki + 2], (NUM_FREQ,)) * 0.1,
                'bias': jnp.zeros(1),
            })
            ki += 3
    temporal = []
    key_t = keys[ki]
    for _ in range(NUM_MODES):
        key_t, *subkeys = jax.random.split(key_t, TEMPORAL_LAYERS * 2 + 1)
        layers = []
        d_in = 1
        for l in range(TEMPORAL_LAYERS - 1):
            w = jax.random.normal(subkeys[2 * l], (d_in, TEMPORAL_HIDDEN)) * jnp.sqrt(2.0 / (d_in + TEMPORAL_HIDDEN))
            b = jnp.zeros(TEMPORAL_HIDDEN)
            layers.append({'w': w, 'b': b})
            d_in = TEMPORAL_HIDDEN
        w = jax.random.normal(subkeys[2 * (TEMPORAL_LAYERS - 1)], (d_in, 1)) * jnp.sqrt(2.0 / (d_in + 1))
        b = jnp.zeros(1)
        layers.append({'w': w, 'b': b})
        temporal.append(layers)

    mode_coeffs = jax.random.normal(jax.random.PRNGKey(999), (NUM_MODES,)) * 0.1
    return {
        'spatial_x': spatial_x,
        'spatial_y': spatial_y,
        'temporal': temporal,
        'mode_coeffs': mode_coeffs,
    }


def spatial_forward(sp, x):
    freqs = jax.lax.stop_gradient(sp['freqs'])
    wx = freqs[None, :] * x
    return jnp.sum(sp['cos_c'] * jnp.cos(wx) + sp['sin_c'] * jnp.sin(wx),
                   axis=1, keepdims=True) + sp['bias']


def temporal_forward(layers, t):
    h = t
    for l in layers[:-1]:
        h = jnp.tanh(h @ l['w'] + l['b'])
    return h @ layers[-1]['w'] + layers[-1]['b']


def svsnn_forward(params, x, y, t):
    u = jnp.zeros_like(x)
    for n in range(NUM_MODES):
        Xn = spatial_forward(params['spatial_x'][n], x)
        Yn = spatial_forward(params['spatial_y'][n], y)
        Tn = temporal_forward(params['temporal'][n], t)
        u = u + params['mode_coeffs'][n] * Xn * Yn * Tn
    return u


# ############################################################
# METHOD 1: SV-SNN ACCELERATED
# ############################################################
def run_svsnn_accelerated(data):
    print(f"\n{'=' * 60}")
    print("Training SV-SNN (ACCELERATED)")
    print(f"{'=' * 60}")

    key = random.PRNGKey(SEED)
    params = init_svsnn_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    NC = NC_SPINN
    xc = jnp.linspace(-1, 1, NC).reshape(-1, 1)
    yc = jnp.linspace(-1, 1, NC).reshape(-1, 1)
    tc = jnp.linspace(0, 1, NC).reshape(-1, 1)

    Xg_f, Yg_f, Tg_f = jnp.meshgrid(xc.ravel(), yc.ravel(), tc.ravel(), indexing='ij')
    f_grid = source_term(Xg_f, Yg_f, Tg_f)

    def _stack_spatial(params, axis_key):
        all_freqs = jnp.stack([jax.lax.stop_gradient(params[axis_key][n]['freqs']) for n in range(NUM_MODES)])
        all_cos_c = jnp.stack([params[axis_key][n]['cos_c'] for n in range(NUM_MODES)])
        all_sin_c = jnp.stack([params[axis_key][n]['sin_c'] for n in range(NUM_MODES)])
        all_bias = jnp.stack([params[axis_key][n]['bias'] for n in range(NUM_MODES)])
        return all_freqs, all_cos_c, all_sin_c, all_bias

    def _compute_basis_with_derivs(coord_flat, all_freqs, all_cos_c, all_sin_c, all_bias):
        wz = coord_flat[:, None, None] * all_freqs[None, :, :]
        cw = jnp.cos(wz)
        sw = jnp.sin(wz)
        trig_terms = all_cos_c[None, :, :] * cw + all_sin_c[None, :, :] * sw
        vals = jnp.sum(trig_terms, axis=-1) + all_bias[None, :, 0]
        w2 = all_freqs ** 2
        d2 = jnp.sum(-w2[None, :, :] * trig_terms, axis=-1)
        return vals, d2

    def _stack_temporal(params):
        w_list, b_list = [], []
        for l_idx in range(TEMPORAL_LAYERS):
            w_list.append(jnp.stack([params['temporal'][n][l_idx]['w'] for n in range(NUM_MODES)]))
            b_list.append(jnp.stack([params['temporal'][n][l_idx]['b'] for n in range(NUM_MODES)]))
        return w_list, b_list

    def _batched_temporal_fwd(w_list, b_list, t):
        def single_fwd(w0, b0, w1, b1, w2, b2, w3, b3):
            h = t
            for w, b in [(w0, b0), (w1, b1), (w2, b2)]:
                h = jnp.tanh(h @ w + b)
            return h @ w3 + b3
        return jax.vmap(single_fwd)(
            w_list[0], b_list[0], w_list[1], b_list[1],
            w_list[2], b_list[2], w_list[3], b_list[3])

    def _batched_temporal_fwd_deriv(w_list, b_list, t):
        def single_fwd_d1(w0, b0, w1, b1, w2, b2, w3, b3):
            h = t
            dh = jnp.ones_like(t)
            for w, b in [(w0, b0), (w1, b1), (w2, b2)]:
                pre = h @ w + b
                h = jnp.tanh(pre)
                dh = (1 - h**2) * (dh @ w)
            return h @ w3 + b3, dh @ w3
        return jax.vmap(single_fwd_d1)(
            w_list[0], b_list[0], w_list[1], b_list[1],
            w_list[2], b_list[2], w_list[3], b_list[3])

    def _batched_temporal_fwd_second_deriv(w_list, b_list, t):
        def single_fwd_d2(w0, b0, w1, b1, w2, b2, w3, b3):
            h = t
            dh = jnp.ones_like(t)
            d2h = jnp.zeros_like(t)
            for w, b in [(w0, b0), (w1, b1), (w2, b2)]:
                pre = h @ w + b
                h_new = jnp.tanh(pre)
                dh_w = dh @ w
                d2h_new = -2 * h_new * (1 - h_new**2) * dh_w**2 + (1 - h_new**2) * (d2h @ w)
                dh_new = (1 - h_new**2) * dh_w
                h = h_new
                dh = dh_new
                d2h = d2h_new
            T = h @ w3 + b3
            dT = dh @ w3
            d2T = d2h @ w3
            return T, dT, d2T
        return jax.vmap(single_fwd_d2)(
            w_list[0], b_list[0], w_list[1], b_list[1],
            w_list[2], b_list[2], w_list[3], b_list[3])

    def vectorized_forward(params, x, y, t):
        fx, cx, sx, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy, by = _stack_spatial(params, 'spatial_y')
        X_all, _ = _compute_basis_with_derivs(x.squeeze(), fx, cx, sx, bx)
        Y_all, _ = _compute_basis_with_derivs(y.squeeze(), fy, cy, sy, by)
        wl, bl = _stack_temporal(params)
        T_all = _batched_temporal_fwd(wl, bl, t)
        mode = X_all * Y_all * T_all[:, :, 0].T
        coeffs = params['mode_coeffs']
        return jnp.sum(coeffs[None, :] * mode, axis=-1, keepdims=True)

    def vectorized_forward_grid(params, x, y, t):
        fx, cx, sx, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy, by = _stack_spatial(params, 'spatial_y')
        wl, bl = _stack_temporal(params)
        coeffs = params['mode_coeffs']
        X_all, _ = _compute_basis_with_derivs(x.squeeze(), fx, cx, sx, bx)
        Y_all, _ = _compute_basis_with_derivs(y.squeeze(), fy, cy, sy, by)
        T_all = _batched_temporal_fwd(wl, bl, t)
        Tv = T_all[:, :, 0]
        cX = coeffs[None, :] * X_all
        return jnp.einsum('im,jm,km->ijk', cX, Y_all, Tv.T)

    def vectorized_pde_residual_grid(params):
        fx, cx, sx, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy, by = _stack_spatial(params, 'spatial_y')
        wl, bl = _stack_temporal(params)
        coeffs = params['mode_coeffs']

        X, d2X = _compute_basis_with_derivs(xc.squeeze(), fx, cx, sx, bx)
        Y, d2Y = _compute_basis_with_derivs(yc.squeeze(), fy, cy, sy, by)
        T, _dT, d2T = _batched_temporal_fwd_second_deriv(wl, bl, tc)
        Tv = T[:, :, 0]
        d2Tv = d2T[:, :, 0]

        def field_3d(A, B, C):
            cA = coeffs[None, :] * A
            return jnp.einsum('im,jm,km->ijk', cA, B, C)

        u_val = field_3d(X, Y, Tv.T)
        u_tt = field_3d(X, Y, d2Tv.T)
        u_xx = field_3d(d2X, Y, Tv.T)
        u_yy = field_3d(X, d2Y, Tv.T)

        return u_tt - u_xx - u_yy + u_val - f_grid

    x_ic, y_ic, t_ic_d = data['x_ic'], data['y_ic'], data['t_ic']

    def ic_velocity_loss(params):
        fx, cx, sx, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy, by = _stack_spatial(params, 'spatial_y')
        wl, bl = _stack_temporal(params)
        coeffs = params['mode_coeffs']
        X_ic, _ = _compute_basis_with_derivs(x_ic.squeeze(), fx, cx, sx, bx)
        Y_ic, _ = _compute_basis_with_derivs(y_ic.squeeze(), fy, cy, sy, by)
        t0 = jnp.zeros((1, 1))
        _T0, dT0 = _batched_temporal_fwd_deriv(wl, bl, t0)
        dTv0 = dT0[:, :, 0]
        ut_pred = jnp.sum(coeffs[None, :] * X_ic * Y_ic * dTv0[:, 0][None, :], axis=-1, keepdims=True)
        return jnp.mean(ut_pred**2)

    schedule = optax.cosine_decay_schedule(init_value=LR, decay_steps=EPOCHS, alpha=1e-2)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    def loss_fn(params):
        u_ic_pred = vectorized_forward(params, x_ic, y_ic, t_ic_d)
        loss_ic = jnp.mean((u_ic_pred - data['u_ic'])**2)
        loss_ic_vel = ic_velocity_loss(params)

        u_bc_pred = vectorized_forward(params, data['x_bc'], data['y_bc'], data['t_bc'])
        loss_bc = jnp.mean((u_bc_pred - data['u_bc'])**2)

        residual = vectorized_pde_residual_grid(params)
        loss_pde = jnp.mean((residual / PDE_NORM)**2)

        total = W_IC * (loss_ic + loss_ic_vel) + W_BC * loss_bc + loss_pde
        return total, (loss_pde, loss_ic, loss_bc)

    @jax.jit
    def train_step(params, opt_state):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss, aux[0], aux[1], aux[2]

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params
    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, pde_l, ic_l, bc_l = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred_grid = np.array(vectorized_forward_grid(
                params, data['x_test_1d'], data['y_test_1d'], data['t_test_1d']))
            err = l2_relative_error(u_pred_grid, data['u_exact'])

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(err)
            history['eval_epochs'].append(epoch)

            if err < best_l2:
                best_l2 = err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2: {err:.4e}", flush=True)

    total_time = time.time() - start_time
    u_pred_grid = np.array(vectorized_forward_grid(
        best_params, data['x_test_1d'], data['y_test_1d'], data['t_test_1d']))
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': u_pred_grid,
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# METHOD 2: SV-SNN ORIGINAL (vmap + nested grad)
# ############################################################
def run_svsnn_original(data):
    print(f"\n{'=' * 60}")
    print("Training SV-SNN (ORIGINAL)")
    print(f"{'=' * 60}")

    key = random.PRNGKey(SEED)
    params = init_svsnn_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    def u_scalar(params, x_s, y_s, t_s):
        x_ = x_s.reshape(1, 1)
        y_ = y_s.reshape(1, 1)
        t_ = t_s.reshape(1, 1)
        return svsnn_forward(params, x_, y_, t_).squeeze()

    def pde_residual_single(params, x_s, y_s, t_s):
        u_tt = jax.grad(jax.grad(lambda t: u_scalar(params, x_s, y_s, t)))(t_s)
        u_xx = jax.grad(jax.grad(lambda x: u_scalar(params, x, y_s, t_s)))(x_s)
        u_yy = jax.grad(jax.grad(lambda y: u_scalar(params, x_s, y, t_s)))(y_s)
        u_val = u_scalar(params, x_s, y_s, t_s)
        f_val = source_term(x_s, y_s, t_s)
        return u_tt - u_xx - u_yy + u_val - f_val

    pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0, 0))

    def ut_scalar(params, x_s, y_s, t_s):
        return jax.grad(lambda t: u_scalar(params, x_s, y_s, t))(t_s)

    ut_batch = jax.vmap(ut_scalar, in_axes=(None, 0, 0, 0))

    x_pde, y_pde, t_pde = data['x_pde'], data['y_pde'], data['t_pde']
    x_ic, y_ic, t_ic_d = data['x_ic'], data['y_ic'], data['t_ic']

    def loss_fn(params):
        u_ic_pred = svsnn_forward(params, x_ic, y_ic, t_ic_d)
        loss_ic = jnp.mean((u_ic_pred - data['u_ic'])**2)

        ut_ic_pred = ut_batch(params, x_ic.squeeze(), y_ic.squeeze(), t_ic_d.squeeze())
        loss_ic_vel = jnp.mean(ut_ic_pred**2)

        u_bc_pred = svsnn_forward(params, data['x_bc'], data['y_bc'], data['t_bc'])
        loss_bc = jnp.mean((u_bc_pred - data['u_bc'])**2)

        res = pde_residual_batch(params, x_pde.squeeze(), y_pde.squeeze(), t_pde.squeeze())
        loss_pde = jnp.mean((res / PDE_NORM)**2)

        total = W_IC * (loss_ic + loss_ic_vel) + W_BC * loss_bc + loss_pde
        return total, (loss_pde, loss_ic, loss_bc)

    schedule = optax.cosine_decay_schedule(init_value=LR, decay_steps=EPOCHS, alpha=1e-2)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss, aux[0], aux[1], aux[2]

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params
    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, pde_l, ic_l, bc_l = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            up = svsnn_forward(params, data['x_test_flat'],
                               data['y_test_flat'], data['t_test_flat'])
            u_pred = np.array(up).reshape(N_TEST, N_TEST, N_TEST)
            err = l2_relative_error(u_pred, data['u_exact'])

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(err)
            history['eval_epochs'].append(epoch)

            if err < best_l2:
                best_l2 = err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2: {err:.4e}", flush=True)

    total_time = time.time() - start_time
    up = svsnn_forward(best_params, data['x_test_flat'],
                       data['y_test_flat'], data['t_test_flat'])
    u_pred = np.array(up).reshape(N_TEST, N_TEST, N_TEST)
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': u_pred,
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# METHOD 3: SPINN
# ############################################################
def run_spinn(data):
    print(f"\n{'=' * 60}")
    print("Training SPINN")
    print(f"{'=' * 60}")

    R = 32
    FEATURES = 64
    N_LAYERS = 4
    FF_DIM_S = 64
    ff_in = 2 * FF_DIM_S

    def init_branch(key, d_in):
        keys = random.split(key, N_LAYERS + 4)
        sc = 1.0 / jnp.sqrt(jnp.float32(d_in))
        fsc = 1.0 / jnp.sqrt(jnp.float32(FEATURES))
        bp = {
            'U_w': random.normal(keys[0], (d_in, FEATURES)) * sc,
            'U_b': jnp.zeros(FEATURES),
            'V_w': random.normal(keys[1], (d_in, FEATURES)) * sc,
            'V_b': jnp.zeros(FEATURES),
            'H_w': random.normal(keys[2], (d_in, FEATURES)) * sc,
            'H_b': jnp.zeros(FEATURES),
            'out_w': random.normal(keys[3], (FEATURES, R)) * fsc,
            'layers': [],
        }
        for i in range(N_LAYERS):
            bp['layers'].append({
                'w': random.normal(keys[4 + i], (FEATURES, FEATURES)) * fsc,
                'b': jnp.zeros(FEATURES),
            })
        return bp

    key = random.PRNGKey(SEED)
    keys = random.split(key, 7)
    params = {
        'branch_x': init_branch(keys[0], ff_in),
        'branch_y': init_branch(keys[1], ff_in),
        'branch_t': init_branch(keys[2], ff_in),
        'W_x': _sample_frequencies(keys[3], FF_DIM_S).reshape(1, -1),
        'W_y': _sample_frequencies(keys[4], FF_DIM_S).reshape(1, -1),
        'W_t': random.normal(keys[5], (1, FF_DIM_S)) * 1.0,
    }
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    def fourier_embed(coord, W):
        return jnp.concatenate([jnp.sin(coord @ W), jnp.cos(coord @ W)], axis=-1)

    def branch_fwd(bp, x):
        U = jnp.tanh(x @ bp['U_w'] + bp['U_b'])
        V = jnp.tanh(x @ bp['V_w'] + bp['V_b'])
        H = jnp.tanh(x @ bp['H_w'] + bp['H_b'])
        for layer in bp['layers']:
            Z = jnp.tanh(H @ layer['w'] + layer['b'])
            H = (1.0 - Z) * U + Z * V
        return H @ bp['out_w']

    def forward_spinn(params, xc, yc, tc):
        Vx = branch_fwd(params['branch_x'], fourier_embed(xc, params['W_x']))
        Vy = branch_fwd(params['branch_y'], fourier_embed(yc, params['W_y']))
        Vt = branch_fwd(params['branch_t'], fourier_embed(tc, params['W_t']))
        return jnp.einsum('ir,jr,kr->ijk', Vx, Vy, Vt)

    spd = data['spinn']
    xc, yc, tc = spd['xc'], spd['yc'], spd['tc']
    ones_x = jnp.ones_like(xc)
    ones_y = jnp.ones_like(yc)
    ones_t = jnp.ones_like(tc)

    Xg_f, Yg_f, Tg_f = jnp.meshgrid(xc.ravel(), yc.ravel(), tc.ravel(), indexing='ij')
    f_grid_spinn = source_term(Xg_f, Yg_f, Tg_f)

    def loss_fn(params):
        fu = lambda x_: forward_spinn(params, x_, yc, tc)
        u = fu(xc)

        _, utt = hvp_fwdfwd(lambda t_: forward_spinn(params, xc, yc, t_),
                            tc, ones_t, return_primals=True)
        uxx = hvp_fwdfwd(fu, xc, ones_x)
        uyy = hvp_fwdfwd(lambda y_: forward_spinn(params, xc, y_, tc),
                         yc, ones_y)

        residual = utt - uxx - uyy + u - f_grid_spinn
        loss_pde = jnp.mean((residual / PDE_NORM)**2)

        t_zero = jnp.array([[0.0]])
        u_ic = forward_spinn(params, xc, yc, t_zero)[:, :, 0]
        loss_ic = jnp.mean((u_ic - spd['u_ic_grid'])**2)

        _, ut_ic = jvp(lambda t_: forward_spinn(params, xc, yc, t_),
                       (t_zero,), (jnp.ones_like(t_zero),))
        loss_ic_vel = jnp.mean(ut_ic[:, :, 0]**2)

        xl = jnp.array([[-1.0]])
        xr = jnp.array([[1.0]])
        yb = jnp.array([[-1.0]])
        yt_b = jnp.array([[1.0]])
        loss_bc = (jnp.mean(forward_spinn(params, xl, yc, tc)**2) +
                   jnp.mean(forward_spinn(params, xr, yc, tc)**2) +
                   jnp.mean(forward_spinn(params, xc, yb, tc)**2) +
                   jnp.mean(forward_spinn(params, xc, yt_b, tc)**2))

        total = W_IC * (loss_ic + loss_ic_vel) + W_BC * loss_bc + loss_pde
        return total, (loss_pde, loss_ic, loss_bc)

    schedule = optax.cosine_decay_schedule(init_value=LR, decay_steps=EPOCHS, alpha=1e-2)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        (loss, (pde_l, ic_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss, pde_l, ic_l, bc_l

    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params
    print("  JIT compiling...", flush=True)
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, pde_l, ic_l, bc_l = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred = np.array(forward_spinn(params, data['x_test_1d'],
                                            data['y_test_1d'], data['t_test_1d']))
            err = l2_relative_error(u_pred, data['u_exact'])

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(err)
            history['eval_epochs'].append(epoch)

            if err < best_l2:
                best_l2 = err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2: {err:.4e}", flush=True)

    total_time = time.time() - start_time
    u_pred = np.array(forward_spinn(best_params, data['x_test_1d'],
                                    data['y_test_1d'], data['t_test_1d']))
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': u_pred,
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# Pointwise Klein-Gordon loss (shared by SIREN/FourierPINN/PINN)
# ############################################################
def pointwise_kg_loss(forward_fn, params, data):
    xyt = jnp.concatenate([data['x_pde'], data['y_pde'], data['t_pde']], axis=-1)
    f = lambda xyt_: forward_fn(params, xyt_)

    tx = jnp.zeros_like(xyt).at[:, 0].set(1.0)
    ty = jnp.zeros_like(xyt).at[:, 1].set(1.0)
    tt = jnp.zeros_like(xyt).at[:, 2].set(1.0)

    u = f(xyt)
    uxx = hvp_fwdfwd(f, xyt, tx)
    uyy = hvp_fwdfwd(f, xyt, ty)
    utt = hvp_fwdfwd(f, xyt, tt)

    f_val = source_term(data['x_pde'], data['y_pde'], data['t_pde'])
    residual = utt - uxx - uyy + u - f_val
    loss_pde = jnp.mean((residual / PDE_NORM)**2)

    xyt_ic = jnp.concatenate([data['x_ic'], data['y_ic'], data['t_ic']], axis=-1)
    u_ic_pred = forward_fn(params, xyt_ic)
    loss_ic = jnp.mean((u_ic_pred - data['u_ic'])**2)

    tt_ic = jnp.zeros_like(xyt_ic).at[:, 2].set(1.0)
    _, ut_ic_pred = jvp(f, (xyt_ic,), (tt_ic,))
    loss_ic_vel = jnp.mean(ut_ic_pred**2)

    xyt_bc = jnp.concatenate([data['x_bc'], data['y_bc'], data['t_bc']], axis=-1)
    u_bc_pred = forward_fn(params, xyt_bc)
    loss_bc = jnp.mean((u_bc_pred - data['u_bc'])**2)

    total = W_IC * (loss_ic + loss_ic_vel) + W_BC * loss_bc + loss_pde
    return total, (loss_pde, loss_ic, loss_bc)


def train_pointwise_method(name, forward_fn, params, data, epochs=EPOCHS):
    print(f"\n{'=' * 60}")
    print(f"Training {name}")
    print(f"{'=' * 60}")

    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}", flush=True)

    def loss_fn(params):
        return pointwise_kg_loss(forward_fn, params, data)

    schedule = optax.cosine_decay_schedule(init_value=LR, decay_steps=epochs, alpha=1e-2)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        (loss, (pde_l, ic_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss, pde_l, ic_l, bc_l

    print("  JIT compiling...", flush=True)
    history = {'total_loss': [], 'pde_loss': [], 'ic_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        params, opt_state, loss, pde_l, ic_l, bc_l = train_step(params, opt_state)
        if epoch == 1:
            jit_time = time.time() - start_time
            print(f"  JIT done in {jit_time:.1f}s", flush=True)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            xyt_test = jnp.concatenate([data['x_test_flat'], data['y_test_flat'],
                                        data['t_test_flat']], axis=-1)
            u_pred = np.array(forward_fn(params, xyt_test)).reshape(N_TEST, N_TEST, N_TEST)
            err = l2_relative_error(u_pred, data['u_exact'])

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(err)
            history['eval_epochs'].append(epoch)

            if err < best_l2:
                best_l2 = err
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2: {err:.4e}")

    total_time = time.time() - start_time
    xyt_test = jnp.concatenate([data['x_test_flat'], data['y_test_flat'],
                                data['t_test_flat']], axis=-1)
    u_pred = np.array(forward_fn(best_params, xyt_test)).reshape(N_TEST, N_TEST, N_TEST)
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': u_pred,
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# METHOD 4: SIREN
# ############################################################
def init_siren(key):
    FF_DIM_S = 64
    k1, k2, k3, key = random.split(key, 4)
    W_x = _sample_frequencies(k1, FF_DIM_S).reshape(1, -1)
    W_y = _sample_frequencies(k2, FF_DIM_S).reshape(1, -1)
    W_t = random.normal(k3, (1, FF_DIM_S)) * 1.0
    ff_input = 6 * FF_DIM_S
    layers = [ff_input, 128, 128, 128, 128, 1]
    params = {'layers': [], 'W_x': W_x, 'W_y': W_y, 'W_t': W_t}
    for i in range(len(layers) - 1):
        k, key = random.split(key)
        d_in, d_out = layers[i], layers[i + 1]
        std = jnp.sqrt(2.0 / d_in)
        params['layers'].append({
            'w': random.normal(k, (d_in, d_out)) * std,
            'b': jnp.zeros(d_out),
        })
    return params


def siren_forward(params, xyt):
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    Wx, Wy, Wt = params['W_x'], params['W_y'], params['W_t']
    Hx = jnp.concatenate([jnp.sin(x @ Wx), jnp.cos(x @ Wx)], axis=-1)
    Hy = jnp.concatenate([jnp.sin(y @ Wy), jnp.cos(y @ Wy)], axis=-1)
    Ht = jnp.concatenate([jnp.sin(t @ Wt), jnp.cos(t @ Wt)], axis=-1)
    h = jnp.concatenate([Hx, Hy, Ht], axis=-1)
    for i, layer in enumerate(params['layers']):
        h = h @ layer['w'] + layer['b']
        if i < len(params['layers']) - 1:
            h = jnp.sin(h)
    return h


# ############################################################
# METHOD 5: FourierPINN
# ############################################################
def init_fourier_pinn(key):
    FF_DIM_F = 64
    hidden = [128, 128, 128, 1]
    k1, k2, k3, key = random.split(key, 4)
    params = {
        'W_x': _sample_frequencies(k1, FF_DIM_F).reshape(1, -1),
        'W_y': _sample_frequencies(k2, FF_DIM_F).reshape(1, -1),
        'W_t': random.normal(k3, (1, FF_DIM_F)) * 1.0,
        'mlp': [],
    }
    dims = [6 * FF_DIM_F] + hidden
    for i in range(len(dims) - 1):
        k, key = random.split(key)
        d_in, d_out = dims[i], dims[i + 1]
        params['mlp'].append({
            'w': random.normal(k, (d_in, d_out)) * jnp.sqrt(2.0 / d_in),
            'b': jnp.zeros(d_out),
        })
    return params


def fourier_pinn_forward(params, xyt):
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    Wx = jax.lax.stop_gradient(params['W_x'])
    Wy = jax.lax.stop_gradient(params['W_y'])
    Wt = jax.lax.stop_gradient(params['W_t'])
    h = jnp.concatenate([jnp.sin(x @ Wx), jnp.cos(x @ Wx),
                          jnp.sin(y @ Wy), jnp.cos(y @ Wy),
                          jnp.sin(t @ Wt), jnp.cos(t @ Wt)], axis=-1)
    for i, layer in enumerate(params['mlp']):
        h = h @ layer['w'] + layer['b']
        if i < len(params['mlp']) - 1:
            h = jnp.tanh(h)
    return h


# ############################################################
# METHOD 6: Vanilla PINN
# ############################################################
def init_pinn(key):
    layers = [3, 128, 128, 128, 128, 1]
    params = {'layers': []}
    for i in range(len(layers) - 1):
        k, key = random.split(key)
        d_in, d_out = layers[i], layers[i + 1]
        params['layers'].append({
            'w': random.normal(k, (d_in, d_out)) * jnp.sqrt(2.0 / d_in),
            'b': jnp.zeros(d_out),
        })
    return params


def pinn_forward(params, xyt):
    h = xyt
    for i, layer in enumerate(params['layers']):
        h = h @ layer['w'] + layer['b']
        if i < len(params['layers']) - 1:
            h = jnp.tanh(h)
    return h


# ############################################################
# Save Utilities
# ############################################################
def save_results(name, result, data):
    np.savez(os.path.join(SAVE_DIR, f"{name}_history.npz"),
             total_loss=np.array(result['history']['total_loss']),
             pde_loss=np.array(result['history']['pde_loss']),
             ic_loss=np.array(result['history']['ic_loss']),
             bc_loss=np.array(result['history']['bc_loss']),
             l2_error=np.array(result['history']['l2_error']),
             eval_epochs=np.array(result['history']['eval_epochs']))

    np.savez(os.path.join(SAVE_DIR, f"{name}_prediction.npz"),
             u_pred=result['u_pred'], u_exact=data['u_exact'],
             X=data['X_test'], Y=data['Y_test'], T=data['T_test'])

    summary = {
        'method': name,
        'total_params': int(result['total_params']),
        'total_time_sec': round(result['total_time_sec'], 2),
        'best_l2_error': float(result['best_l2_error']),
        'final_l2_error': float(result['final_l2_error']),
        'ms_per_epoch': round(result['total_time_sec'] / EPOCHS * 1000, 2),
    }
    with open(os.path.join(SAVE_DIR, f"{name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {name} -> {SAVE_DIR}/")


def save_comparison_table(all_results):
    fields = ['method', 'total_params', 'total_time_sec',
              'best_l2_error', 'final_l2_error', 'ms_per_epoch']
    filepath = os.path.join(SAVE_DIR, "comparison_table.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, r in all_results.items():
            writer.writerow({
                'method': name,
                'total_params': r['total_params'],
                'total_time_sec': f"{r['total_time_sec']:.2f}",
                'best_l2_error': f"{r['best_l2_error']:.6e}",
                'final_l2_error': f"{r['final_l2_error']:.6e}",
                'ms_per_epoch': f"{r['total_time_sec'] / EPOCHS * 1000:.2f}",
            })
    print(f"\nComparison table -> {filepath}")


# ############################################################
# Main
# ############################################################
def method_done(name):
    return os.path.exists(os.path.join(SAVE_DIR, f"{name}_summary.json"))


def main():
    print("=" * 60)
    print("SV-SNN Acceleration - Case 11: Klein-Gordon 3D (2+1D)")
    print(f"  Linear Klein-Gordon: u_tt - u_xx - u_yy + u = f")
    print(f"  kappa = {KAPPA:.4f} (4pi)")
    print(f"  u = sin(kx)*sin(ky)*cosh(t)")
    print(f"  PDE coeff = {PDE_COEFF:.2f}")
    print(f"  Device: {jax.devices()}")
    print(f"  Epochs: {EPOCHS} | LR: {LR} (cosine) | N_PDE: {N_PDE}")
    print(f"  IC/BC weight: {W_IC}")
    print("=" * 60)

    data = generate_data()
    print("Data generated.")

    all_results = {}
    key = random.PRNGKey(SEED)

    if not method_done('SVSNN_accel'):
        result = run_svsnn_accelerated(data)
        save_results('SVSNN_accel', result, data)
        all_results['SVSNN_accel'] = result
    else:
        print("\n  [SKIP] SVSNN_accel already completed.")

    if not method_done('SVSNN_orig'):
        result = run_svsnn_original(data)
        save_results('SVSNN_orig', result, data)
        all_results['SVSNN_orig'] = result
    else:
        print("\n  [SKIP] SVSNN_orig already completed.")

    if not method_done('SPINN'):
        result = run_spinn(data)
        save_results('SPINN', result, data)
        all_results['SPINN'] = result
    else:
        print("\n  [SKIP] SPINN already completed.")

    if not method_done('SIREN'):
        k, key = random.split(key)
        params = init_siren(k)
        result = train_pointwise_method('SIREN', siren_forward, params, data)
        save_results('SIREN', result, data)
        all_results['SIREN'] = result
    else:
        print("\n  [SKIP] SIREN already completed.")

    if not method_done('FourierPINN'):
        k, key = random.split(key)
        params = init_fourier_pinn(k)
        result = train_pointwise_method('FourierPINN', fourier_pinn_forward, params, data)
        save_results('FourierPINN', result, data)
        all_results['FourierPINN'] = result
    else:
        print("\n  [SKIP] FourierPINN already completed.")

    if not method_done('PINN'):
        k, key = random.split(key)
        params = init_pinn(k)
        result = train_pointwise_method('PINN', pinn_forward, params, data)
        save_results('PINN', result, data)
        all_results['PINN'] = result
    else:
        print("\n  [SKIP] PINN already completed.")

    if all_results:
        save_comparison_table(all_results)
        print("\n" + "=" * 70)
        print("FINAL COMPARISON")
        print("=" * 70)
        hdr = f"{'Method':<14} {'Params':>10} {'Time(s)':>10} {'Best L2':>12} {'Final L2':>12} {'ms/epoch':>10}"
        print(hdr)
        print("-" * len(hdr))
        for name, r in all_results.items():
            print(f"{name:<14} {r['total_params']:>10,} {r['total_time_sec']:>10.1f} "
                  f"{r['best_l2_error']:>12.4e} {r['final_l2_error']:>12.4e} "
                  f"{r['total_time_sec'] / EPOCHS * 1000:>10.2f}")
        print("=" * 70)
    print(f"\nAll results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
