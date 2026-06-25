"""
SV-SNN Acceleration - Case 8: Taylor-Green Vortex (2D Navier-Stokes)
=====================================================================
PDE: Incompressible Navier-Stokes (2D, time-dependent)
  du/dt + u*du/dx + v*du/dy + dp/dx - (1/Re)*(u_xx + u_yy) = 0
  dv/dt + u*dv/dx + v*dv/dy + dp/dy - (1/Re)*(v_xx + v_yy) = 0
  du/dx + dv/dy = 0
Re = 100, Domain: [-pi, pi]^2 x [0, 1], Periodic BC

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

# ============================================================
# Configuration
# ============================================================
RE = 100.0
W_CHAR = float(np.pi)
FF_DIM = 64
SEED = 42
EPOCHS = 10000
LR = 1e-3
N_PDE = 5000
N_IC = 5000
N_BC = 1000
N_TEST_X = 50
N_TEST_Y = 50
N_TEST_T = 20
EVAL_EVERY = 100
NC_SPINN = 50

E11_OVR = {}  # E11 size overrides for matched budget (set by E11_run)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Exact Solution
# ============================================================
def exact_u(x, y, t):
    return -jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.exp(-2 * jnp.pi**2 * t / RE)


def exact_v(x, y, t):
    return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.exp(-2 * jnp.pi**2 * t / RE)


def exact_p(x, y, t):
    return -0.25 * (jnp.cos(2 * jnp.pi * x) + jnp.cos(2 * jnp.pi * y)) * jnp.exp(-4 * jnp.pi**2 * t / RE)


# ============================================================
# Data Generation
# ============================================================
def generate_data(seed=SEED):
    np.random.seed(seed)
    PI = np.pi

    x1d = np.linspace(-PI, PI, 100)
    y1d = np.linspace(-PI, PI, 100)
    Xic, Yic = np.meshgrid(x1d, y1d, indexing='ij')
    x_ic = Xic.reshape(-1, 1)
    y_ic = Yic.reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    decay0 = np.ones_like(x_ic)
    u_ic = -np.cos(PI * x_ic) * np.sin(PI * y_ic) * decay0
    v_ic = np.sin(PI * x_ic) * np.cos(PI * y_ic) * decay0
    p_ic = -0.25 * (np.cos(2 * PI * x_ic) + np.cos(2 * PI * y_ic)) * decay0

    y_bc_x = np.random.uniform(-PI, PI, (N_BC // 2, 1))
    t_bc_x = np.random.uniform(0, 1, (N_BC // 2, 1))
    x_bc_left = -PI * np.ones((N_BC // 2, 1))
    x_bc_right = PI * np.ones((N_BC // 2, 1))

    x_bc_y = np.random.uniform(-PI, PI, (N_BC // 2, 1))
    t_bc_y = np.random.uniform(0, 1, (N_BC // 2, 1))
    y_bc_bottom = -PI * np.ones((N_BC // 2, 1))
    y_bc_top = PI * np.ones((N_BC // 2, 1))

    pts = lhs(3, samples=N_PDE)  # plain LHS (maximin too slow for repeated runs)
    x_pde = -PI + 2 * PI * pts[:, 0:1]
    y_pde = -PI + 2 * PI * pts[:, 1:2]
    t_pde = pts[:, 2:3]

    xt = np.linspace(-PI, PI, N_TEST_X)
    yt = np.linspace(-PI, PI, N_TEST_Y)
    tt = np.linspace(0, 1, N_TEST_T)
    XT, YT, TT = np.meshgrid(xt, yt, tt, indexing='ij')
    decay_u = np.exp(-2 * PI**2 * TT / RE)
    decay_p = np.exp(-4 * PI**2 * TT / RE)
    u_exact = -np.cos(PI * XT) * np.sin(PI * YT) * decay_u
    v_exact = np.sin(PI * XT) * np.cos(PI * YT) * decay_u
    p_exact = -0.25 * (np.cos(2 * PI * XT) + np.cos(2 * PI * YT)) * decay_p

    f32 = jnp.float32
    data = {
        'x_ic': jnp.array(x_ic, dtype=f32), 'y_ic': jnp.array(y_ic, dtype=f32),
        't_ic': jnp.array(t_ic, dtype=f32),
        'u_ic': jnp.array(u_ic, dtype=f32), 'v_ic': jnp.array(v_ic, dtype=f32),
        'p_ic': jnp.array(p_ic, dtype=f32),
        'x_bc_left': jnp.array(x_bc_left, dtype=f32),
        'x_bc_right': jnp.array(x_bc_right, dtype=f32),
        'y_bc_x': jnp.array(y_bc_x, dtype=f32),
        't_bc_x': jnp.array(t_bc_x, dtype=f32),
        'x_bc_y': jnp.array(x_bc_y, dtype=f32),
        'y_bc_bottom': jnp.array(y_bc_bottom, dtype=f32),
        'y_bc_top': jnp.array(y_bc_top, dtype=f32),
        't_bc_y': jnp.array(t_bc_y, dtype=f32),
        'x_pde': jnp.array(x_pde, dtype=f32),
        'y_pde': jnp.array(y_pde, dtype=f32),
        't_pde': jnp.array(t_pde, dtype=f32),
        'X_test': XT, 'Y_test': YT, 'T_test': TT,
        'u_exact': u_exact, 'v_exact': v_exact, 'p_exact': p_exact,
        'x_test_flat': jnp.array(XT.reshape(-1, 1), dtype=f32),
        'y_test_flat': jnp.array(YT.reshape(-1, 1), dtype=f32),
        't_test_flat': jnp.array(TT.reshape(-1, 1), dtype=f32),
    }

    xc = np.linspace(-PI, PI, NC_SPINN).reshape(-1, 1)
    yc = np.linspace(-PI, PI, NC_SPINN).reshape(-1, 1)
    tc = np.linspace(0, 1, NC_SPINN).reshape(-1, 1)
    Xg, Yg = np.meshgrid(xc.ravel(), yc.ravel(), indexing='ij')
    data['spinn'] = {
        'xc': jnp.array(xc, dtype=f32),
        'yc': jnp.array(yc, dtype=f32),
        'tc': jnp.array(tc, dtype=f32),
        'u_ic_grid': jnp.array(-np.cos(PI * Xg) * np.sin(PI * Yg), dtype=f32),
        'v_ic_grid': jnp.array(np.sin(PI * Xg) * np.cos(PI * Yg), dtype=f32),
        'p_ic_grid': jnp.array(-0.25 * (np.cos(2 * PI * Xg) + np.cos(2 * PI * Yg)), dtype=f32),
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


FREQ_SCALE = 12.0

def _sample_frequencies(key, K):
    n_basic = K * 2 // 5
    n_char = K * 2 // 5
    n_high = K - n_basic - n_char
    k1, k2, k3 = jax.random.split(key, 3)
    f_basic = jnp.linspace(0.5, 5.0, n_basic)
    f_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 0.05 + 2.0)
    f_high = jax.random.uniform(k3, (n_high,), minval=5.0, maxval=FREQ_SCALE)
    return jnp.sort(jnp.concatenate([f_basic, f_char, f_high]))


# ############################################################
# METHOD 1: SV-SNN
# ############################################################
def run_svsnn(data):
    print(f"\n{'='*60}")
    print("Training SV-SNN")
    print(f"{'='*60}")

    NUM_MODES = 6
    NUM_FREQ = 32
    FREQ_SCALE = 12.0
    TEMPORAL_LAYERS = 4
    TEMPORAL_HIDDEN = 10

    def _sample_frequencies(key, K):
        n_basic = K * 2 // 5
        n_char = K * 2 // 5
        n_high = K - n_basic - n_char
        k1, k2, k3 = jax.random.split(key, 3)
        f_basic = jnp.linspace(0.5, 5.0, n_basic)
        f_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 0.05 + 2.0)
        f_high = jax.random.uniform(k3, (n_high,), minval=5.0, maxval=FREQ_SCALE)
        return jnp.sort(jnp.concatenate([f_basic, f_char, f_high]))

    def init_params(key):
        keys = jax.random.split(key, NUM_MODES * 7 + 3)
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
        for _ in range(NUM_MODES):
            key_t = keys[ki]; ki += 1
            layers = []
            d_in = 1
            for _ in range(TEMPORAL_LAYERS - 1):
                key_t, k = jax.random.split(key_t)
                layers.append({
                    'w': jax.random.normal(k, (d_in, TEMPORAL_HIDDEN)) * jnp.sqrt(2.0 / (d_in + TEMPORAL_HIDDEN)),
                    'b': jnp.zeros(TEMPORAL_HIDDEN),
                })
                d_in = TEMPORAL_HIDDEN
            key_t, k = jax.random.split(key_t)
            layers.append({
                'w': jax.random.normal(k, (d_in, 1)) * jnp.sqrt(2.0 / (d_in + 1)),
                'b': jnp.zeros(1),
            })
            temporal.append(layers)
        return {
            'spatial_x': spatial_x, 'spatial_y': spatial_y, 'temporal': temporal,
            'mode_coeffs_u': jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1,
            'mode_coeffs_v': jax.random.normal(keys[ki + 1], (NUM_MODES,)) * 0.1,
            'mode_coeffs_p': jax.random.normal(keys[ki + 2], (NUM_MODES,)) * 0.1,
        }

    def spatial_forward(sp, x):
        freqs = jax.lax.stop_gradient(sp['freqs'])
        wx = freqs[None, :] * x
        return jnp.sum(sp['cos_c'] * jnp.cos(wx) + sp['sin_c'] * jnp.sin(wx),
                       axis=1, keepdims=True) + sp['bias']

    def spatial_forward_with_derivs(sp, x):
        freqs = jax.lax.stop_gradient(sp['freqs'])
        cc, sc = sp['cos_c'], sp['sin_c']
        wx = freqs[None, :] * x
        cw, sw = jnp.cos(wx), jnp.sin(wx)
        val = jnp.sum(cc * cw + sc * sw, axis=1, keepdims=True) + sp['bias']
        d1 = jnp.sum(-cc * freqs * sw + sc * freqs * cw, axis=1, keepdims=True)
        d2 = jnp.sum(-cc * freqs**2 * cw - sc * freqs**2 * sw, axis=1, keepdims=True)
        return val, d1, d2

    def temporal_forward(layers, t):
        h = t
        for l in layers[:-1]:
            h = jnp.tanh(h @ l['w'] + l['b'])
        return h @ layers[-1]['w'] + layers[-1]['b']

    def forward(params, x, y, t):
        u = jnp.zeros_like(x)
        v = jnp.zeros_like(x)
        p = jnp.zeros_like(x)
        for n in range(NUM_MODES):
            Xn = spatial_forward(params['spatial_x'][n], x)
            Yn = spatial_forward(params['spatial_y'][n], y)
            Tn = temporal_forward(params['temporal'][n], t)
            mode = Xn * Yn * Tn
            u += params['mode_coeffs_u'][n] * mode
            v += params['mode_coeffs_v'][n] * mode
            p += params['mode_coeffs_p'][n] * mode
        return u, v, p

    def forward_with_derivs(params, x, y, t):
        zeros = jnp.zeros_like(x)
        u, v, p = zeros, zeros, zeros
        du_dx, du_dy, du_dt = zeros, zeros, zeros
        d2u_dx2, d2u_dy2 = zeros, zeros
        dv_dx, dv_dy, dv_dt = zeros, zeros, zeros
        d2v_dx2, d2v_dy2 = zeros, zeros
        dp_dx, dp_dy = zeros, zeros

        for n in range(NUM_MODES):
            Xn, dXn, d2Xn = spatial_forward_with_derivs(params['spatial_x'][n], x)
            Yn, dYn, d2Yn = spatial_forward_with_derivs(params['spatial_y'][n], y)
            Tn = temporal_forward(params['temporal'][n], t)
            _, dTn = jvp(lambda t_: temporal_forward(params['temporal'][n], t_),
                         (t,), (jnp.ones_like(t),))

            cu = params['mode_coeffs_u'][n]
            cv = params['mode_coeffs_v'][n]
            cp = params['mode_coeffs_p'][n]

            XYT = Xn * Yn * Tn
            u += cu * XYT;  v += cv * XYT;  p += cp * XYT

            du_dx += cu * dXn * Yn * Tn;  du_dy += cu * Xn * dYn * Tn
            du_dt += cu * Xn * Yn * dTn
            d2u_dx2 += cu * d2Xn * Yn * Tn;  d2u_dy2 += cu * Xn * d2Yn * Tn

            dv_dx += cv * dXn * Yn * Tn;  dv_dy += cv * Xn * dYn * Tn
            dv_dt += cv * Xn * Yn * dTn
            d2v_dx2 += cv * d2Xn * Yn * Tn;  d2v_dy2 += cv * Xn * d2Yn * Tn

            dp_dx += cp * dXn * Yn * Tn;  dp_dy += cp * Xn * dYn * Tn

        return (u, v, p, du_dx, du_dy, du_dt, d2u_dx2, d2u_dy2,
                dv_dx, dv_dy, dv_dt, d2v_dx2, d2v_dy2, dp_dx, dp_dy)

    key = random.PRNGKey(SEED)
    params = init_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    x_pde, y_pde, t_pde = data['x_pde'], data['y_pde'], data['t_pde']
    x_ic, y_ic, t_ic_d = data['x_ic'], data['y_ic'], data['t_ic']

    def loss_fn(params):
        u_ic, v_ic, p_ic = forward(params, x_ic, y_ic, t_ic_d)
        loss_ic = (jnp.mean((u_ic - data['u_ic'])**2) +
                   jnp.mean((v_ic - data['v_ic'])**2) +
                   jnp.mean((p_ic - data['p_ic'])**2))

        (u, v, p, du_dx, du_dy, du_dt, d2u_dx2, d2u_dy2,
         dv_dx, dv_dy, dv_dt, d2v_dx2, d2v_dy2, dp_dx, dp_dy) = \
            forward_with_derivs(params, x_pde, y_pde, t_pde)

        inv_re = 1.0 / RE
        r_u = du_dt + u * du_dx + v * du_dy + dp_dx - inv_re * (d2u_dx2 + d2u_dy2)
        r_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - inv_re * (d2v_dx2 + d2v_dy2)
        r_d = du_dx + dv_dy
        loss_pde = jnp.mean(r_u**2) + jnp.mean(r_v**2) + jnp.mean(r_d**2)

        loss_bc = jnp.float32(0.0)
        return loss_ic + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
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
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, pde_l, ic_l, bc_l = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            up, vp, pp = forward(params, data['x_test_flat'],
                                 data['y_test_flat'], data['t_test_flat'])
            shape = (N_TEST_X, N_TEST_Y, N_TEST_T)
            l2_u = l2_relative_error(np.array(up).reshape(shape), data['u_exact'])
            l2_v = l2_relative_error(np.array(vp).reshape(shape), data['v_exact'])
            l2_p = l2_relative_error(np.array(pp).reshape(shape), data['p_exact'])
            l2_avg = (l2_u + l2_v + l2_p) / 3.0

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(l2_avg)
            history['eval_epochs'].append(epoch)

            if l2_avg < best_l2:
                best_l2 = l2_avg
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2avg: {l2_avg:.4e}")

    total_time = time.time() - start_time
    up, vp, pp = forward(best_params, data['x_test_flat'],
                         data['y_test_flat'], data['t_test_flat'])
    shape = (N_TEST_X, N_TEST_Y, N_TEST_T)
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': np.array(up).reshape(shape),
        'v_pred': np.array(vp).reshape(shape),
        'p_pred': np.array(pp).reshape(shape),
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# METHOD 1b: SV-SNN ACCELERATED
# ############################################################
def run_svsnn_accelerated(data):
    print(f"\n{'='*60}")
    print("Training SV-SNN (ACCELERATED)")
    print("  Analytic spatial derivs, manual T_n', vectorized modes, grid eval")
    print(f"{'='*60}")

    NUM_MODES = 6
    NUM_FREQ = 32
    FREQ_SCALE = 12.0
    TEMPORAL_LAYERS = 4
    TEMPORAL_HIDDEN = 10
    NC = NC_SPINN

    def _sample_freqs(key, K):
        n_basic = K * 2 // 5
        n_char = K * 2 // 5
        n_high = K - n_basic - n_char
        k1, k2, k3 = jax.random.split(key, 3)
        f_basic = jnp.linspace(0.5, 5.0, n_basic)
        f_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 0.05 + 2.0)
        f_high = jax.random.uniform(k3, (n_high,), minval=5.0, maxval=FREQ_SCALE)
        return jnp.sort(jnp.concatenate([f_basic, f_char, f_high]))

    def init_params(key):
        keys = jax.random.split(key, NUM_MODES * 7 + 3)
        ki = 0
        spatial_x, spatial_y = [], []
        for _ in range(NUM_MODES):
            for s_list in [spatial_x, spatial_y]:
                s_list.append({
                    'freqs': _sample_freqs(keys[ki], NUM_FREQ),
                    'cos_c': jax.random.normal(keys[ki + 1], (NUM_FREQ,)) * 0.1,
                    'sin_c': jax.random.normal(keys[ki + 2], (NUM_FREQ,)) * 0.1,
                    'bias': jnp.zeros(1),
                })
                ki += 3
        temporal = []
        for _ in range(NUM_MODES):
            key_t = keys[ki]; ki += 1
            layers = []
            d_in = 1
            for _ in range(TEMPORAL_LAYERS - 1):
                key_t, k = jax.random.split(key_t)
                layers.append({
                    'w': jax.random.normal(k, (d_in, TEMPORAL_HIDDEN)) * jnp.sqrt(2.0 / (d_in + TEMPORAL_HIDDEN)),
                    'b': jnp.zeros(TEMPORAL_HIDDEN),
                })
                d_in = TEMPORAL_HIDDEN
            key_t, k = jax.random.split(key_t)
            layers.append({
                'w': jax.random.normal(k, (d_in, 1)) * jnp.sqrt(2.0 / (d_in + 1)),
                'b': jnp.zeros(1),
            })
            temporal.append(layers)
        return {
            'spatial_x': spatial_x, 'spatial_y': spatial_y, 'temporal': temporal,
            'mode_coeffs_u': jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1,
            'mode_coeffs_v': jax.random.normal(keys[ki + 1], (NUM_MODES,)) * 0.1,
            'mode_coeffs_p': jax.random.normal(keys[ki + 2], (NUM_MODES,)) * 0.1,
        }

    key = random.PRNGKey(SEED)
    params = init_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

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
        d1 = jnp.sum(all_freqs[None, :, :] * (-all_cos_c[None, :, :] * sw + all_sin_c[None, :, :] * cw), axis=-1)
        w2 = all_freqs ** 2
        d2 = jnp.sum(-w2[None, :, :] * trig_terms, axis=-1)
        return vals, d1, d2

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

    def vectorized_forward(params, x, y, t):
        fx, cx, sx, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy, by = _stack_spatial(params, 'spatial_y')
        X_all, _, _ = _compute_basis_with_derivs(x.squeeze(), fx, cx, sx, bx)
        Y_all, _, _ = _compute_basis_with_derivs(y.squeeze(), fy, cy, sy, by)
        wl, bl = _stack_temporal(params)
        T_all = _batched_temporal_fwd(wl, bl, t)
        mode = X_all * Y_all * T_all[:, :, 0].T
        cu = params['mode_coeffs_u']
        cv = params['mode_coeffs_v']
        cp = params['mode_coeffs_p']
        u = jnp.sum(cu[None, :] * mode, axis=-1, keepdims=True)
        v = jnp.sum(cv[None, :] * mode, axis=-1, keepdims=True)
        p = jnp.sum(cp[None, :] * mode, axis=-1, keepdims=True)
        return u, v, p

    xc = jnp.linspace(-jnp.pi, jnp.pi, NC).reshape(-1, 1)
    yc = jnp.linspace(-jnp.pi, jnp.pi, NC).reshape(-1, 1)
    tc = jnp.linspace(0, 1, NC).reshape(-1, 1)

    x_ic, y_ic, t_ic_d = data['x_ic'], data['y_ic'], data['t_ic']

    def vectorized_pde_residual_grid(params):
        fx, cx, sx, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy, by = _stack_spatial(params, 'spatial_y')
        wl, bl = _stack_temporal(params)

        X, dX, d2X = _compute_basis_with_derivs(xc.squeeze(), fx, cx, sx, bx)
        Y, dY, d2Y = _compute_basis_with_derivs(yc.squeeze(), fy, cy, sy, by)
        T, dT = _batched_temporal_fwd_deriv(wl, bl, tc)
        Tv = T[:, :, 0]
        dTv = dT[:, :, 0]

        cu = params['mode_coeffs_u']
        cv = params['mode_coeffs_v']
        cp = params['mode_coeffs_p']

        def field_3d(coeffs, A, B, C):
            cA = coeffs[None, :] * A
            return jnp.einsum('im,jm,km->ijk', cA, B, C)

        u_val = field_3d(cu, X, Y, Tv.T)
        v_val = field_3d(cv, X, Y, Tv.T)
        p_val = field_3d(cp, X, Y, Tv.T)

        du_dx = field_3d(cu, dX, Y, Tv.T)
        du_dy = field_3d(cu, X, dY, Tv.T)
        du_dt = field_3d(cu, X, Y, dTv.T)
        d2u_dx2 = field_3d(cu, d2X, Y, Tv.T)
        d2u_dy2 = field_3d(cu, X, d2Y, Tv.T)

        dv_dx = field_3d(cv, dX, Y, Tv.T)
        dv_dy = field_3d(cv, X, dY, Tv.T)
        dv_dt = field_3d(cv, X, Y, dTv.T)
        d2v_dx2 = field_3d(cv, d2X, Y, Tv.T)
        d2v_dy2 = field_3d(cv, X, d2Y, Tv.T)

        dp_dx = field_3d(cp, dX, Y, Tv.T)
        dp_dy = field_3d(cp, X, dY, Tv.T)

        inv_re = 1.0 / RE
        r_u = du_dt + u_val * du_dx + v_val * du_dy + dp_dx - inv_re * (d2u_dx2 + d2u_dy2)
        r_v = dv_dt + u_val * dv_dx + v_val * dv_dy + dp_dy - inv_re * (d2v_dx2 + d2v_dy2)
        r_d = du_dx + dv_dy

        return r_u, r_v, r_d

    def loss_fn(params):
        u_ic_p, v_ic_p, p_ic_p = vectorized_forward(params, x_ic, y_ic, t_ic_d)
        loss_ic = (jnp.mean((u_ic_p - data['u_ic'])**2) +
                   jnp.mean((v_ic_p - data['v_ic'])**2) +
                   jnp.mean((p_ic_p - data['p_ic'])**2))

        r_u, r_v, r_d = vectorized_pde_residual_grid(params)
        loss_pde = jnp.mean(r_u**2) + jnp.mean(r_v**2) + jnp.mean(r_d**2)

        return loss_ic + loss_pde, (loss_pde, loss_ic, jnp.float32(0.0))

    optimizer = optax.adam(LR)
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
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, pde_l, ic_l, bc_l = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            up, vp, pp = vectorized_forward(params, data['x_test_flat'],
                                            data['y_test_flat'], data['t_test_flat'])
            shape = (N_TEST_X, N_TEST_Y, N_TEST_T)
            l2_u = l2_relative_error(np.array(up).reshape(shape), data['u_exact'])
            l2_v = l2_relative_error(np.array(vp).reshape(shape), data['v_exact'])
            l2_p = l2_relative_error(np.array(pp).reshape(shape), data['p_exact'])
            l2_avg = (l2_u + l2_v + l2_p) / 3.0

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(l2_avg)
            history['eval_epochs'].append(epoch)

            if l2_avg < best_l2:
                best_l2 = l2_avg
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2avg: {l2_avg:.4e}")

    total_time = time.time() - start_time
    up, vp, pp = vectorized_forward(best_params, data['x_test_flat'],
                                    data['y_test_flat'], data['t_test_flat'])
    shape = (N_TEST_X, N_TEST_Y, N_TEST_T)
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': np.array(up).reshape(shape),
        'v_pred': np.array(vp).reshape(shape),
        'p_pred': np.array(pp).reshape(shape),
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# METHOD 2: SPINN
# ############################################################
def run_spinn(data):
    print(f"\n{'='*60}")
    print("Training SPINN")
    print(f"{'='*60}")

    R = E11_OVR.get('spinn_r', 64)
    FEATURES = E11_OVR.get('spinn_features', 64)
    N_LAYERS = E11_OVR.get('spinn_n_layers', 4)
    FF_DIM = E11_OVR.get('spinn_ff', 64)
    ff_in = 2 * FF_DIM

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
    keys = random.split(key, 9)
    params = {
        'branch_x': init_branch(keys[0], ff_in),
        'branch_y': init_branch(keys[1], ff_in),
        'branch_t': init_branch(keys[2], ff_in),
        'W_x': _sample_frequencies(keys[3], FF_DIM).reshape(1, -1),
        'W_y': _sample_frequencies(keys[4], FF_DIM).reshape(1, -1),
        'W_t': random.normal(keys[5], (1, FF_DIM)) * 1.0,
        'w_u': random.normal(keys[6], (R,)) * 0.1,
        'w_v': random.normal(keys[7], (R,)) * 0.1,
        'w_p': random.normal(keys[8], (R,)) * 0.1,
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

    def forward_var(params, xc, yc, tc, w):
        Vx = branch_fwd(params['branch_x'], fourier_embed(xc, params['W_x']))
        Vy = branch_fwd(params['branch_y'], fourier_embed(yc, params['W_y']))
        Vt = branch_fwd(params['branch_t'], fourier_embed(tc, params['W_t']))
        return jnp.einsum('ir,jr,kr->ijk', Vx * w[None, :], Vy, Vt)

    spd = data['spinn']
    xc, yc, tc = spd['xc'], spd['yc'], spd['tc']
    ones_x = jnp.ones_like(xc)
    ones_y = jnp.ones_like(yc)
    ones_t = jnp.ones_like(tc)
    PI_j = jnp.pi

    def loss_fn(params):
        fu = lambda x_: forward_var(params, x_, yc, tc, params['w_u'])
        fv = lambda x_: forward_var(params, x_, yc, tc, params['w_v'])

        u = fu(xc)
        v_val = fv(xc)

        du_dx, d2u_dx2 = hvp_fwdfwd(fu, xc, ones_x, return_primals=True)
        gu_y = lambda y_: forward_var(params, xc, y_, tc, params['w_u'])
        du_dy, d2u_dy2 = hvp_fwdfwd(gu_y, yc, ones_y, return_primals=True)
        _, du_dt = jvp(lambda t_: forward_var(params, xc, yc, t_, params['w_u']),
                       (tc,), (ones_t,))

        dv_dx, d2v_dx2 = hvp_fwdfwd(fv, xc, ones_x, return_primals=True)
        gv_y = lambda y_: forward_var(params, xc, y_, tc, params['w_v'])
        dv_dy, d2v_dy2 = hvp_fwdfwd(gv_y, yc, ones_y, return_primals=True)
        _, dv_dt = jvp(lambda t_: forward_var(params, xc, yc, t_, params['w_v']),
                       (tc,), (ones_t,))

        _, dp_dx = jvp(lambda x_: forward_var(params, x_, yc, tc, params['w_p']),
                       (xc,), (ones_x,))
        _, dp_dy = jvp(lambda y_: forward_var(params, xc, y_, tc, params['w_p']),
                       (yc,), (ones_y,))

        inv_re = 1.0 / RE
        r_u = du_dt + u * du_dx + v_val * du_dy + dp_dx - inv_re * (d2u_dx2 + d2u_dy2)
        r_v = dv_dt + u * dv_dx + v_val * dv_dy + dp_dy - inv_re * (d2v_dx2 + d2v_dy2)
        r_d = du_dx + dv_dy
        loss_pde = jnp.mean(r_u**2) + jnp.mean(r_v**2) + jnp.mean(r_d**2)

        t_zero = jnp.array([[0.0]])
        u_ic = forward_var(params, xc, yc, t_zero, params['w_u'])[:, :, 0]
        v_ic = forward_var(params, xc, yc, t_zero, params['w_v'])[:, :, 0]
        p_ic = forward_var(params, xc, yc, t_zero, params['w_p'])[:, :, 0]
        loss_ic = (jnp.mean((u_ic - spd['u_ic_grid'])**2) +
                   jnp.mean((v_ic - spd['v_ic_grid'])**2) +
                   jnp.mean((p_ic - spd['p_ic_grid'])**2))

        xl = jnp.array([[-PI_j]]); xr = jnp.array([[PI_j]])
        yb = jnp.array([[-PI_j]]); yt_b = jnp.array([[PI_j]])

        loss_bc = jnp.float32(0.0)
        for w_key in ['w_u', 'w_v', 'w_p']:
            w = params[w_key]
            loss_bc += jnp.mean((forward_var(params, xl, yc, tc, w) -
                                 forward_var(params, xr, yc, tc, w))**2)
            loss_bc += jnp.mean((forward_var(params, xc, yb, tc, w) -
                                 forward_var(params, xc, yt_b, tc, w))**2)

        return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)

    optimizer = optax.adam(LR)
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
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss, pde_l, ic_l, bc_l = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_t = np.array(forward_var(params, data['x_test_1d'], data['y_test_1d'],
                                       data['t_test_1d'], params['w_u']))
            v_t = np.array(forward_var(params, data['x_test_1d'], data['y_test_1d'],
                                       data['t_test_1d'], params['w_v']))
            p_t = np.array(forward_var(params, data['x_test_1d'], data['y_test_1d'],
                                       data['t_test_1d'], params['w_p']))
            l2_u = l2_relative_error(u_t, data['u_exact'])
            l2_v = l2_relative_error(v_t, data['v_exact'])
            l2_p = l2_relative_error(p_t, data['p_exact'])
            l2_avg = (l2_u + l2_v + l2_p) / 3.0

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(l2_avg)
            history['eval_epochs'].append(epoch)

            if l2_avg < best_l2:
                best_l2 = l2_avg
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2avg: {l2_avg:.4e}")

    total_time = time.time() - start_time
    u_f = np.array(forward_var(best_params, data['x_test_1d'], data['y_test_1d'],
                               data['t_test_1d'], best_params['w_u']))
    v_f = np.array(forward_var(best_params, data['x_test_1d'], data['y_test_1d'],
                               data['t_test_1d'], best_params['w_v']))
    p_f = np.array(forward_var(best_params, data['x_test_1d'], data['y_test_1d'],
                               data['t_test_1d'], best_params['w_p']))
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': u_f, 'v_pred': v_f, 'p_pred': p_f,
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# Pointwise methods: shared infrastructure
# ############################################################
def pointwise_ns_loss(forward_fn, params, data):
    xyt = jnp.concatenate([data['x_pde'], data['y_pde'], data['t_pde']], axis=-1)
    f = lambda xyt_: forward_fn(params, xyt_)

    tx = jnp.zeros_like(xyt).at[:, 0].set(1.0)
    ty = jnp.zeros_like(xyt).at[:, 1].set(1.0)
    tt = jnp.zeros_like(xyt).at[:, 2].set(1.0)

    uvp = f(xyt)
    u, v = uvp[:, 0:1], uvp[:, 1:2]

    uvp_x, uvp_xx = hvp_fwdfwd(f, xyt, tx, return_primals=True)
    uvp_y, uvp_yy = hvp_fwdfwd(f, xyt, ty, return_primals=True)
    _, uvp_t = jvp(f, (xyt,), (tt,))

    du_dx, dv_dx, dp_dx = uvp_x[:, 0:1], uvp_x[:, 1:2], uvp_x[:, 2:3]
    du_dy, dv_dy, dp_dy = uvp_y[:, 0:1], uvp_y[:, 1:2], uvp_y[:, 2:3]
    du_dt, dv_dt = uvp_t[:, 0:1], uvp_t[:, 1:2]
    d2u_dx2, d2v_dx2 = uvp_xx[:, 0:1], uvp_xx[:, 1:2]
    d2u_dy2, d2v_dy2 = uvp_yy[:, 0:1], uvp_yy[:, 1:2]

    inv_re = 1.0 / RE
    r_u = du_dt + u * du_dx + v * du_dy + dp_dx - inv_re * (d2u_dx2 + d2u_dy2)
    r_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - inv_re * (d2v_dx2 + d2v_dy2)
    r_d = du_dx + dv_dy
    loss_pde = jnp.mean(r_u**2) + jnp.mean(r_v**2) + jnp.mean(r_d**2)

    xyt_ic = jnp.concatenate([data['x_ic'], data['y_ic'], data['t_ic']], axis=-1)
    uvp_ic = forward_fn(params, xyt_ic)
    loss_ic = (jnp.mean((uvp_ic[:, 0:1] - data['u_ic'])**2) +
               jnp.mean((uvp_ic[:, 1:2] - data['v_ic'])**2) +
               jnp.mean((uvp_ic[:, 2:3] - data['p_ic'])**2))

    xyt_l = jnp.concatenate([data['x_bc_left'], data['y_bc_x'], data['t_bc_x']], axis=-1)
    xyt_r = jnp.concatenate([data['x_bc_right'], data['y_bc_x'], data['t_bc_x']], axis=-1)
    loss_bc_x = jnp.mean((forward_fn(params, xyt_l) - forward_fn(params, xyt_r))**2)

    xyt_b = jnp.concatenate([data['x_bc_y'], data['y_bc_bottom'], data['t_bc_y']], axis=-1)
    xyt_t = jnp.concatenate([data['x_bc_y'], data['y_bc_top'], data['t_bc_y']], axis=-1)
    loss_bc_y = jnp.mean((forward_fn(params, xyt_b) - forward_fn(params, xyt_t))**2)

    loss_bc = loss_bc_x + loss_bc_y
    return loss_ic + loss_bc + loss_pde, (loss_pde, loss_ic, loss_bc)


def train_pointwise_method(name, forward_fn, params, data, epochs=EPOCHS):
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")

    import sys
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}", flush=True)

    def loss_fn(params):
        return pointwise_ns_loss(forward_fn, params, data)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        (loss, (pde_l, ic_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss, pde_l, ic_l, bc_l

    print(f"  JIT compiling...", flush=True)
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
            uvp = forward_fn(params, xyt_test)
            shape = (N_TEST_X, N_TEST_Y, N_TEST_T)
            l2_u = l2_relative_error(np.array(uvp[:, 0]).reshape(shape), data['u_exact'])
            l2_v = l2_relative_error(np.array(uvp[:, 1]).reshape(shape), data['v_exact'])
            l2_p = l2_relative_error(np.array(uvp[:, 2]).reshape(shape), data['p_exact'])
            l2_avg = (l2_u + l2_v + l2_p) / 3.0

            history['total_loss'].append(float(loss))
            history['pde_loss'].append(float(pde_l))
            history['ic_loss'].append(float(ic_l))
            history['bc_loss'].append(float(bc_l))
            history['l2_error'].append(l2_avg)
            history['eval_epochs'].append(epoch)

            if l2_avg < best_l2:
                best_l2 = l2_avg
                best_params = jax.tree.map(lambda x: x.copy(), params)

            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss: {float(loss):.4e} | "
                      f"PDE: {float(pde_l):.4e} | IC: {float(ic_l):.4e} | "
                      f"L2avg: {l2_avg:.4e}")

    total_time = time.time() - start_time
    xyt_test = jnp.concatenate([data['x_test_flat'], data['y_test_flat'],
                                data['t_test_flat']], axis=-1)
    uvp_f = forward_fn(best_params, xyt_test)
    shape = (N_TEST_X, N_TEST_Y, N_TEST_T)
    print(f"  Time: {total_time:.1f}s | Best L2: {best_l2:.4e}")

    return {
        'params': best_params, 'history': history,
        'u_pred': np.array(uvp_f[:, 0]).reshape(shape),
        'v_pred': np.array(uvp_f[:, 1]).reshape(shape),
        'p_pred': np.array(uvp_f[:, 2]).reshape(shape),
        'total_params': n_params, 'total_time_sec': total_time,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
    }


# ############################################################
# METHOD 3: SIREN
# ############################################################
def init_siren(key, ff=64, hidden=128, n_hidden=4):
    FF_DIM_S = ff
    k1, k2, k3, key = random.split(key, 4)
    W_x = _sample_frequencies(k1, FF_DIM_S).reshape(1, -1)
    W_y = _sample_frequencies(k2, FF_DIM_S).reshape(1, -1)
    W_t = random.normal(k3, (1, FF_DIM_S)) * 1.0
    ff_input = 6 * FF_DIM_S
    layers = [ff_input] + [hidden] * n_hidden + [3]
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
# METHOD 4: FourierPINN
# ############################################################
def init_fourier_pinn(key, ff=64, hidden_w=128, n_hidden=3):
    FF_DIM = ff
    hidden = [hidden_w] * n_hidden + [3]
    k1, k2, k3, key = random.split(key, 4)
    params = {
        'W_x': _sample_frequencies(k1, FF_DIM).reshape(1, -1),
        'W_y': _sample_frequencies(k2, FF_DIM).reshape(1, -1),
        'W_t': random.normal(k3, (1, FF_DIM)) * 1.0,
        'mlp': [],
    }
    dims = [6 * FF_DIM] + hidden
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
# METHOD 5: Vanilla PINN
# ############################################################
def init_pinn(key, hidden=128, n_hidden=4):
    layers = [3] + [hidden] * n_hidden + [3]
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
             u_pred=result['u_pred'], v_pred=result['v_pred'], p_pred=result['p_pred'],
             u_exact=data['u_exact'], v_exact=data['v_exact'], p_exact=data['p_exact'],
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
    print("Ablation Study - Case 8: Taylor-Green Vortex (Re=100)")
    print(f"  Device: {jax.devices()}")
    print(f"  Epochs: {EPOCHS} | LR: {LR} | N_PDE: {N_PDE}")
    print("=" * 60)

    data = generate_data()
    print("Data generated.")

    all_results = {}
    key = random.PRNGKey(SEED)

    # --- SV-SNN ACCELERATED ---
    if not method_done('SVSNN_accel'):
        result = run_svsnn_accelerated(data)
        save_results('SVSNN_accel', result, data)
        all_results['SVSNN_accel'] = result
    else:
        print("\n  [SKIP] SVSNN_accel already completed.")

    # --- SV-SNN ORIGINAL ---
    if not method_done('SVSNN_orig'):
        result = run_svsnn(data)
        save_results('SVSNN_orig', result, data)
        all_results['SVSNN_orig'] = result
    else:
        print("\n  [SKIP] SVSNN_orig already completed.")

    # --- SPINN ---
    if not method_done('SPINN'):
        result = run_spinn(data)
        save_results('SPINN', result, data)
        all_results['SPINN'] = result
    else:
        print("\n  [SKIP] SPINN already completed.")

    # --- SIREN ---
    if not method_done('SIREN'):
        k, key = random.split(key)
        params = init_siren(k)
        result = train_pointwise_method('SIREN', siren_forward, params, data)
        save_results('SIREN', result, data)
        all_results['SIREN'] = result
    else:
        print("\n  [SKIP] SIREN already completed.")

    # --- FourierPINN ---
    if not method_done('FourierPINN'):
        k, key = random.split(key)
        params = init_fourier_pinn(k)
        result = train_pointwise_method('FourierPINN', fourier_pinn_forward, params, data)
        save_results('FourierPINN', result, data)
        all_results['FourierPINN'] = result
    else:
        print("\n  [SKIP] FourierPINN already completed.")

    # --- PINN ---
    if not method_done('PINN'):
        k, key = random.split(key)
        params = init_pinn(k)
        result = train_pointwise_method('PINN', pinn_forward, params, data)
        save_results('PINN', result, data)
        all_results['PINN'] = result
    else:
        print("\n  [SKIP] PINN already completed.")

    # --- Comparison ---
    if all_results:
        save_comparison_table(all_results)
        print("\n" + "=" * 60)
        print("FINAL COMPARISON")
        print("=" * 60)
        hdr = f"{'Method':<14} {'Params':>10} {'Time(s)':>10} {'Best L2':>12} {'Final L2':>12}"
        print(hdr)
        print("-" * len(hdr))
        for name, r in all_results.items():
            print(f"{name:<14} {r['total_params']:>10,} {r['total_time_sec']:>10.1f} "
                  f"{r['best_l2_error']:>12.4e} {r['final_l2_error']:>12.4e}")
        print("=" * 60)
    print(f"\nAll results saved to: {SAVE_DIR}")


CASE_INFO = {"id": "case8", "title": "Taylor-Green vortex 2D NS (Re=100)",
             "family": "ns", "has_classical": False}

_ARCH8 = dict(in_dim=3, out_dim=3, n_coord=3, spinn_n_branch=3, per_out_weight=True)


def E11_run(method, budget, seed, epochs=None, target=None, save_pred_path=None):
    import numpy as _np
    import _e11common
    from jax import random as _random
    g = globals()
    g["SEED"] = seed
    if epochs is not None:
        g["EPOCHS"] = epochs
    g["E11_OVR"] = {}
    EP = g["EPOCHS"]
    tgt = target if (method != "SVSNN" and budget == "matched") else None

    data = generate_data(seed)
    k = _random.PRNGKey(seed)

    if method == "SVSNN":
        res = run_svsnn_accelerated(data); n_coll = NC_SPINN ** 3
    elif method == "SPINN":
        if budget == "matched":
            _e11common.set_matched_ovr(sys.modules[__name__], "SPINN", target, seed, _ARCH8)
        res = run_spinn(data); n_coll = NC_SPINN ** 3
    else:
        if budget == "matched":
            sz, _, _ = _e11common.harness.choose_matched(
                method, target, in_dim=3, out_dim=3, n_coord=3,
                n_hidden_pinn=4, n_hidden_fourier=3, n_hidden_siren=4)
        if method == "SIREN":
            params = (init_siren(k, ff=sz["ff"], hidden=sz["hidden"], n_hidden=sz["n_hidden"])
                      if budget == "matched" else init_siren(k))
            res = train_pointwise_method("SIREN", siren_forward, params, data, epochs=EP)
        elif method == "FourierPINN":
            params = (init_fourier_pinn(k, ff=sz["ff"], hidden_w=sz["hidden"], n_hidden=sz["n_hidden"])
                      if budget == "matched" else init_fourier_pinn(k))
            res = train_pointwise_method("FourierPINN", fourier_pinn_forward, params, data, epochs=EP)
        elif method == "PINN":
            params = (init_pinn(k, hidden=sz["hidden"], n_hidden=sz["n_hidden"])
                      if budget == "matched" else init_pinn(k))
            res = train_pointwise_method("PINN", pinn_forward, params, data, epochs=EP)
        else:
            raise ValueError(method)
        n_coll = int(N_PDE)

    n_params = int(res["total_params"])
    matched_within = (abs(n_params - target) <= 0.10 * target) if tgt is not None else None
    rec = _e11common.harness.normalize_record(
        method, budget, seed, params=n_params, best_l2=float(res["best_l2_error"]),
        final_l2=float(res["final_l2_error"]), train_time_sec=float(res["total_time_sec"]),
        n_epochs=EP, n_collocation=n_coll, inference_ms=float("nan"),
        target_params=tgt, matched_within_tol=matched_within)
    if save_pred_path is not None:
        _np.savez(save_pred_path, u_pred=_np.asarray(res["u_pred"]),
                  u_exact=_np.asarray(data["u_exact"]))
    return rec


if __name__ == "__main__":
    main()
