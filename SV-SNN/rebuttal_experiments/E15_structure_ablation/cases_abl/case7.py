"""
SV-SNN Acceleration – Case 7: 2D Poisson on Complex Multi-Hole Domain
=====================================================================
PDE:  -Laplacian(u) = f(x,y)   on Omega = [-1,1]^2 minus holes
      f(x,y) = 2*MU^2 * sin(MU*x) * sin(MU*y),   MU = 7*pi
BC:   u = u_exact on ALL boundaries (exterior square + hole boundaries)
Exact: u(x,y) = sin(MU*x) * sin(MU*y)

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
from jax import random, jit, value_and_grad, jvp
import optax
from pyDOE import lhs

sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ================================================================
#  Configuration
# ================================================================
MU = 7.0 * np.pi
W_CHAR = MU
FF_DIM = 64
SEED = 42
EPOCHS = 10000
LR = 1e-3
N_PDE = 20000
N_BC_EXTERIOR = 400
N_BC_PER_HOLE = 200
N_TEST = 256
EVAL_EVERY = 100
NC_SPINN = 100

CIRCLES = [
    {'cx': -0.5, 'cy': -0.5, 'r': 0.1},
    {'cx':  0.5, 'cy':  0.5, 'r': 0.2},
    {'cx':  0.5, 'cy': -0.5, 'r': 0.2},
]
ELLIPSE = {'cx': -0.5, 'cy': 0.5, 'a': 0.25, 'b': 0.125}

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)


# ================================================================
#  PDE Definition
# ================================================================
def exact_solution(x, y):
    return jnp.sin(MU * x) * jnp.sin(MU * y)


def source_term(x, y):
    return 2.0 * MU**2 * jnp.sin(MU * x) * jnp.sin(MU * y)


# ================================================================
#  Domain Geometry
# ================================================================
def is_inside_hole_np(x, y):
    """True where (x,y) lies strictly inside any hole (NumPy)."""
    inside = np.zeros(x.shape, dtype=bool)
    for c in CIRCLES:
        inside |= ((x - c['cx'])**2 + (y - c['cy'])**2) < c['r']**2
    e = ELLIPSE
    inside |= (((x - e['cx']) / e['a'])**2 + ((y - e['cy']) / e['b'])**2) < 1.0
    return inside


def domain_mask_np(x, y):
    """1.0 outside all holes, 0.0 inside (NumPy)."""
    return (~is_inside_hole_np(x, y)).astype(np.float64)


# ================================================================
#  Data Generation
# ================================================================
def generate_hole_bc(n_per_hole):
    """Boundary points on all hole surfaces."""
    theta = np.linspace(0, 2 * np.pi, n_per_hole, endpoint=False)
    xs, ys = [], []
    for c in CIRCLES:
        xs.append(c['cx'] + c['r'] * np.cos(theta))
        ys.append(c['cy'] + c['r'] * np.sin(theta))
    e = ELLIPSE
    xs.append(e['cx'] + e['a'] * np.cos(theta))
    ys.append(e['cy'] + e['b'] * np.sin(theta))
    x_h = np.concatenate(xs).reshape(-1, 1)
    y_h = np.concatenate(ys).reshape(-1, 1)
    u_h = np.sin(MU * x_h) * np.sin(MU * y_h)
    return x_h, y_h, u_h


def generate_data(seed=SEED):
    np.random.seed(seed)

    # Exterior BC: 100 per side on [-1,1]^2
    n_per_side = N_BC_EXTERIOR // 4
    t = np.linspace(-1.0, 1.0, n_per_side).reshape(-1, 1)
    x_ext = np.vstack([t, t, -np.ones_like(t), np.ones_like(t)])
    y_ext = np.vstack([-np.ones_like(t), np.ones_like(t), t, t])
    u_ext = np.sin(MU * x_ext) * np.sin(MU * y_ext)

    # Hole BC
    x_hole, y_hole, u_hole = generate_hole_bc(N_BC_PER_HOLE)

    x_bc = np.vstack([x_ext, x_hole])
    y_bc = np.vstack([y_ext, y_hole])
    u_bc = np.vstack([u_ext, u_hole])

    # PDE collocation: LHS in [-1,1]^2, reject inside holes
    n_sample = int(N_PDE * 1.5)
    pts = lhs(2, samples=n_sample) * 2.0 - 1.0
    keep = ~is_inside_hole_np(pts[:, 0], pts[:, 1])
    pts = pts[keep][:N_PDE]
    x_pde = pts[:, 0:1]
    y_pde = pts[:, 1:2]
    print(f"  PDE collocation points after hole rejection: {pts.shape[0]}")

    # Test grid
    x1d = np.linspace(-1.0, 1.0, N_TEST)
    y1d = np.linspace(-1.0, 1.0, N_TEST)
    X_test, Y_test = np.meshgrid(x1d, y1d, indexing='ij')
    u_exact_test = np.sin(MU * X_test) * np.sin(MU * Y_test)
    test_mask = domain_mask_np(X_test, Y_test)

    data = {
        'x_bc': jnp.array(x_bc, dtype=jnp.float32),
        'y_bc': jnp.array(y_bc, dtype=jnp.float32),
        'u_bc': jnp.array(u_bc, dtype=jnp.float32),
        'x_pde': jnp.array(x_pde, dtype=jnp.float32),
        'y_pde': jnp.array(y_pde, dtype=jnp.float32),
        'X_test': X_test,
        'Y_test': Y_test,
        'u_exact_test': u_exact_test,
        'test_mask': test_mask,
        'x_test_flat': jnp.array(X_test.reshape(-1, 1), dtype=jnp.float32),
        'y_test_flat': jnp.array(Y_test.reshape(-1, 1), dtype=jnp.float32),
        'x_test_1d': jnp.array(x1d.reshape(-1, 1), dtype=jnp.float32),
        'y_test_1d': jnp.array(y1d.reshape(-1, 1), dtype=jnp.float32),
    }

    # SPINN-specific grid data
    xc = np.linspace(-1.0, 1.0, NC_SPINN).reshape(-1, 1)
    yc = np.linspace(-1.0, 1.0, NC_SPINN).reshape(-1, 1)
    Xg, Yg = np.meshgrid(xc.ravel(), yc.ravel(), indexing='ij')
    grid_mask = domain_mask_np(Xg, Yg)
    f_grid = 2.0 * MU**2 * np.sin(MU * Xg) * np.sin(MU * Yg)

    data['spinn'] = {
        'xc': jnp.array(xc, dtype=jnp.float32),
        'yc': jnp.array(yc, dtype=jnp.float32),
        'f_grid': jnp.array(f_grid, dtype=jnp.float32),
        'grid_mask': jnp.array(grid_mask, dtype=jnp.float32),
        'x_hole_bc': jnp.array(x_hole, dtype=jnp.float32),
        'y_hole_bc': jnp.array(y_hole, dtype=jnp.float32),
        'u_hole_bc': jnp.array(u_hole, dtype=jnp.float32),
    }
    return data


# ================================================================
#  Utility: hvp_fwdfwd (forward-over-forward second derivative)
# ================================================================
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    """Second-order directional derivative via forward-over-forward JVP.
    f: callable taking a single array argument.
    primals, tangents: plain arrays (not tuples)."""
    g = lambda primals: jvp(f, (primals,), (tangents,))[1]
    primals_out, tangents_out = jvp(g, (primals,), (tangents,))
    if return_primals:
        return primals_out, tangents_out
    return tangents_out


# ================================================================
#  Utility: parameter counting
# ================================================================
def count_params(params):
    leaves = jax.tree.leaves(params)
    return sum(p.size for p in leaves if hasattr(p, 'size'))


# ================================================================
#  Utility: L2 relative error (masked for complex domain)
# ================================================================
def l2_relative_error_masked(u_pred, u_exact, mask):
    diff = (u_pred - u_exact) * mask
    ref = u_exact * mask
    denom = np.sum(ref**2)
    if denom < 1e-30:
        return float(np.sqrt(np.sum(diff**2)))
    return float(np.sqrt(np.sum(diff**2) / denom))


def _sample_frequencies(key, K, w_char):
    freqs = jnp.abs(jax.random.normal(key, (K,)) * 10.0 + w_char)
    return jnp.sort(freqs)


# ################################################################
#  METHOD 1: SV-SNN
# ################################################################
def svsnn_forward(params, x, y):
    """Module-level forward for save_results compatibility."""
    NUM_MODES = len(params['mode_coeffs'])
    u = jnp.zeros_like(x)
    for n in range(NUM_MODES):
        sp_x = params['spatial_x'][n]
        freqs_x = jax.lax.stop_gradient(sp_x['freqs'])
        wx = freqs_x[None, :] * x
        X_n = (jnp.sum(sp_x['cos_c'] * jnp.cos(wx) + sp_x['sin_c'] * jnp.sin(wx),
                        axis=1, keepdims=True) + sp_x['bias'])
        sp_y = params['spatial_y'][n]
        freqs_y = jax.lax.stop_gradient(sp_y['freqs'])
        wy = freqs_y[None, :] * y
        Y_n = (jnp.sum(sp_y['cos_c'] * jnp.cos(wy) + sp_y['sin_c'] * jnp.sin(wy),
                        axis=1, keepdims=True) + sp_y['bias'])
        u = u + params['mode_coeffs'][n] * X_n * Y_n
    return u


def run_svsnn(data):
    print(f"\n{'='*60}")
    print("Training SV-SNN")
    print(f"{'='*60}")

    NUM_MODES = 8
    NUM_FREQ = 40

    def _sample_frequencies(key, K, w_char):
        import _abl, abl_freqs
        if _abl.STRATEGY != 'default':
            return abl_freqs.strategy_sample(key, K, w_char, w_char, _abl.STRATEGY) * _abl.SCALE
        freqs = jnp.abs(jax.random.normal(key, (K,)) * 10.0 + w_char)
        return jnp.sort(freqs) * _abl.SCALE

    def init_params(key):
        keys = jax.random.split(key, NUM_MODES * 6 + 1)
        ki = 0
        spatial_x, spatial_y = [], []
        for _ in range(NUM_MODES):
            spatial_x.append({
                'freqs': _sample_frequencies(keys[ki], NUM_FREQ, W_CHAR),
                'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                'bias': jnp.zeros(1),
            })
            ki += 3
            spatial_y.append({
                'freqs': _sample_frequencies(keys[ki], NUM_FREQ, W_CHAR),
                'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                'bias': jnp.zeros(1),
            })
            ki += 3
        mode_coeffs = jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1
        return {'spatial_x': spatial_x, 'spatial_y': spatial_y,
                'mode_coeffs': mode_coeffs}

    def spatial_forward(sp, x):
        freqs = jax.lax.stop_gradient(sp['freqs'])
        wx = freqs[None, :] * x
        return (jnp.sum(sp['cos_c'] * jnp.cos(wx) + sp['sin_c'] * jnp.sin(wx),
                        axis=1, keepdims=True) + sp['bias'])

    def forward(params, x, y):
        u = jnp.zeros_like(x)
        for n in range(NUM_MODES):
            u = u + (params['mode_coeffs'][n]
                     * spatial_forward(params['spatial_x'][n], x)
                     * spatial_forward(params['spatial_y'][n], y))
        return u

    def pde_residual_single(params, x_s, y_s):
        def u_fn(x_, y_):
            return forward(params, x_[None, None], y_[None, None]).squeeze()
        u_xx = jax.grad(jax.grad(u_fn, argnums=0), argnums=0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, argnums=1), argnums=1)(x_s, y_s)
        f_val = source_term(x_s, y_s)
        return u_xx + u_yy + f_val

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

    def loss_fn(params, xp, yp, xb, yb, ub):
        u_bc_pred = forward(params, xb, yb)
        bc_loss = jnp.mean((u_bc_pred - ub)**2)
        residuals = pde_residual_batch(params, xp, yp)
        pde_loss = jnp.mean(residuals**2)
        return pde_loss + bc_loss, (pde_loss, bc_loss)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(
            params, x_pde_flat, y_pde_flat, x_bc, y_bc, u_bc)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    history = {'total_loss': [], 'pde_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params

    for _ in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

    start_time = time.time()
    for epoch in range(2, EPOCHS):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            u_pred = forward(params, data['x_test_flat'], data['y_test_flat'])
            u_pred_np = np.array(u_pred).reshape(N_TEST, N_TEST)
            l2_err = l2_relative_error_masked(u_pred_np, data['u_exact_test'],
                                              data['test_mask'])
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
    ms_per_epoch = (total_time / (EPOCHS - 2)) * 1000
    u_final = forward(best_params, data['x_test_flat'], data['y_test_flat'])
    u_final_np = np.array(u_final).reshape(N_TEST, N_TEST)

    print(f"  Training time: {total_time:.1f}s ({ms_per_epoch:.2f} ms/epoch)")
    print(f"  Best L2 error: {best_l2:.4e}")
    print(f"  Final L2 error: {history['l2_error'][-1]:.4e}")

    return best_params, history, u_final_np, n_params, total_time


# ################################################################
#  METHOD 1b: SV-SNN ACCELERATED
# ################################################################
def run_svsnn_accelerated(data):
    print(f"\n{'='*60}")
    print("Training SV-SNN (ACCELERATED)")
    print("  Analytic X_n''/Y_n'', vectorized, grid + multi-hole mask")
    print(f"{'='*60}")

    NUM_MODES = 8
    NUM_FREQ = 40
    NC = NC_SPINN

    def _sample_frequencies(key, K, w_char):
        import _abl, abl_freqs
        if _abl.STRATEGY != 'default':
            return abl_freqs.strategy_sample(key, K, w_char, w_char, _abl.STRATEGY) * _abl.SCALE
        freqs = jnp.abs(jax.random.normal(key, (K,)) * 10.0 + w_char)
        return jnp.sort(freqs) * _abl.SCALE

    def init_params(key):
        keys = jax.random.split(key, NUM_MODES * 6 + 1)
        ki = 0
        spatial_x, spatial_y = [], []
        for _ in range(NUM_MODES):
            spatial_x.append({
                'freqs': _sample_frequencies(keys[ki], NUM_FREQ, W_CHAR),
                'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                'bias': jnp.zeros(1),
            })
            ki += 3
            spatial_y.append({
                'freqs': _sample_frequencies(keys[ki], NUM_FREQ, W_CHAR),
                'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                'bias': jnp.zeros(1),
            })
            ki += 3
        mode_coeffs = jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1
        return {'spatial_x': spatial_x, 'spatial_y': spatial_y, 'mode_coeffs': mode_coeffs}

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

    def _compute_basis(coord_flat, all_freqs, all_cos_c, all_sin_c, all_bias):
        wz = coord_flat[:, None, None] * all_freqs[None, :, :]
        cos_wz = jnp.cos(wz)
        sin_wz = jnp.sin(wz)
        trig_terms = all_cos_c[None, :, :] * cos_wz + all_sin_c[None, :, :] * sin_wz
        vals = jnp.sum(trig_terms, axis=-1) + all_bias[None, :, 0]
        return vals, trig_terms

    def _second_deriv(trig_terms, all_freqs):
        w2 = all_freqs ** 2
        return jnp.sum(-w2[None, :, :] * trig_terms, axis=-1)

    xc = jnp.linspace(-1, 1, NC).reshape(-1, 1)
    yc = jnp.linspace(-1, 1, NC).reshape(-1, 1)
    X_mesh, Y_mesh = jnp.meshgrid(xc.squeeze(), yc.squeeze(), indexing='ij')
    f_grid = 2.0 * MU**2 * jnp.sin(MU * X_mesh) * jnp.sin(MU * Y_mesh)

    grid_mask_np = domain_mask_np(np.array(X_mesh), np.array(Y_mesh))
    grid_mask = jnp.array(grid_mask_np, dtype=jnp.float32)
    f_grid = f_grid * grid_mask

    x_bc, y_bc, u_bc = data['x_bc'], data['y_bc'], data['u_bc']
    test_mask = data['test_mask']

    def vectorized_forward(params, x, y):
        freqs_x, cos_x, sin_x, bias_x = _stack_spatial(params, 'spatial_x')
        freqs_y, cos_y, sin_y, bias_y = _stack_spatial(params, 'spatial_y')
        coeffs = params['mode_coeffs']
        X_all, _ = _compute_basis(x.squeeze(), freqs_x, cos_x, sin_x, bias_x)
        Y_all, _ = _compute_basis(y.squeeze(), freqs_y, cos_y, sin_y, bias_y)
        return jnp.sum(coeffs[None, :] * X_all * Y_all, axis=-1, keepdims=True)

    def vectorized_forward_grid(params, x, y):
        freqs_x, cos_x, sin_x, bias_x = _stack_spatial(params, 'spatial_x')
        freqs_y, cos_y, sin_y, bias_y = _stack_spatial(params, 'spatial_y')
        coeffs = params['mode_coeffs']
        X_all, _ = _compute_basis(x.squeeze(), freqs_x, cos_x, sin_x, bias_x)
        Y_all, _ = _compute_basis(y.squeeze(), freqs_y, cos_y, sin_y, bias_y)
        cX = coeffs[None, :] * X_all
        return jnp.einsum('nm,jm->nj', cX, Y_all)

    def vectorized_pde_residual(params):
        freqs_x, cos_x, sin_x, bias_x = _stack_spatial(params, 'spatial_x')
        freqs_y, cos_y, sin_y, bias_y = _stack_spatial(params, 'spatial_y')
        coeffs = params['mode_coeffs']

        X_all, trig_x = _compute_basis(xc.squeeze(), freqs_x, cos_x, sin_x, bias_x)
        Y_all, trig_y = _compute_basis(yc.squeeze(), freqs_y, cos_y, sin_y, bias_y)

        X_dd = _second_deriv(trig_x, freqs_x)
        Y_dd = _second_deriv(trig_y, freqs_y)

        cXdd = coeffs[None, :] * X_dd
        u_xx = jnp.einsum('nm,jm->nj', cXdd, Y_all)

        cX = coeffs[None, :] * X_all
        u_yy = jnp.einsum('nm,jm->nj', cX, Y_dd)

        residual = (u_xx + u_yy + f_grid) * grid_mask
        return residual

    def loss_fn(params):
        residual = vectorized_pde_residual(params)
        n_valid = jnp.sum(grid_mask)
        pde_loss = jnp.sum(residual**2) / jnp.maximum(n_valid, 1.0)

        u_bc_pred = vectorized_forward(params, x_bc, y_bc)
        bc_loss = jnp.mean((u_bc_pred - u_bc)**2)

        return pde_loss + bc_loss, (pde_loss, bc_loss)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    history = {'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params

    for _ in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

    start_time = time.time()
    for epoch in range(2, EPOCHS):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            u_pred_grid = vectorized_forward_grid(params, data['x_test_1d'], data['y_test_1d'])
            u_pred_np = np.array(u_pred_grid)
            l2_err = l2_relative_error_masked(u_pred_np, data['u_exact_test'], test_mask)

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
    ms_per_epoch = (total_time / (EPOCHS - 2)) * 1000
    u_final = np.array(vectorized_forward_grid(best_params, data['x_test_1d'], data['y_test_1d']))

    print(f"  Training time: {total_time:.1f}s ({ms_per_epoch:.2f} ms/epoch)")
    print(f"  Best L2 error: {best_l2:.4e}")

    return best_params, history, u_final, n_params, total_time


# ################################################################
#  METHOD 2: SPINN
# ################################################################
FF_DIM_SPINN = 64


def init_spinn(key, features=64, n_layers=4, r=64):
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
            'out_w': random.normal(keys[-1], (features, r)) * (
                1.0 / jnp.sqrt(jnp.array(features, dtype=jnp.float32))),
        }
        for i in range(n_layers):
            w = random.normal(keys[3 + i], (features, features)) * (
                1.0 / jnp.sqrt(jnp.array(features, dtype=jnp.float32)))
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


def spinn_branch_forward(bp, x):
    U = jnp.tanh(x @ bp['U_w'] + bp['U_b'])
    V = jnp.tanh(x @ bp['V_w'] + bp['V_b'])
    H = jnp.tanh(x @ bp['H_w'] + bp['H_b'])
    for layer in bp['layers']:
        Z = jnp.tanh(H @ layer['w'] + layer['b'])
        H = (1.0 - Z) * U + Z * V
    return H @ bp['out_w']


def spinn_forward(params, x, y):
    """Grid-based: (Nx,1) x (Ny,1) -> (Nx, Ny)."""
    x_emb = spinn_fourier_embed(x, params['W_x'])
    y_emb = spinn_fourier_embed(y, params['W_y'])
    bx = spinn_branch_forward(params['branch_x'], x_emb)
    by = spinn_branch_forward(params['branch_y'], y_emb)
    return bx @ by.T


def spinn_forward_pointwise(params, x, y):
    """Evaluate at N scattered points: (N,1) x (N,1) -> (N,1)."""
    x_emb = spinn_fourier_embed(x, params['W_x'])
    y_emb = spinn_fourier_embed(y, params['W_y'])
    bx = spinn_branch_forward(params['branch_x'], x_emb)
    by = spinn_branch_forward(params['branch_y'], y_emb)
    return jnp.sum(bx * by, axis=1, keepdims=True)


def train_spinn(data, epochs=EPOCHS, params=None):
    print(f"\n{'='*60}")
    print("Training SPINN")
    print(f"{'='*60}")

    key = random.PRNGKey(SEED)
    if params is None:
        params = init_spinn(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params:,}")

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    sd = data['spinn']
    xc, yc = sd['xc'], sd['yc']
    f_grid = sd['f_grid']
    grid_mask = sd['grid_mask']
    x_hbc = sd['x_hole_bc']
    y_hbc = sd['y_hole_bc']
    u_hbc = sd['u_hole_bc']

    # Exterior BC exact values (sin(MU*±1)=0, so u=0 on square boundary)
    xb_left  = jnp.array([[-1.0]])
    xb_right = jnp.array([[ 1.0]])
    yb_bot   = jnp.array([[-1.0]])
    yb_top   = jnp.array([[ 1.0]])

    def loss_fn(params):
        # PDE on grid with hole mask
        u_xx = hvp_fwdfwd(lambda x_in: spinn_forward(params, x_in, yc),
                          xc, jnp.ones_like(xc))
        u_yy = hvp_fwdfwd(lambda y_in: spinn_forward(params, xc, y_in),
                          yc, jnp.ones_like(yc)).T

        residual = u_xx + u_yy + f_grid
        n_valid = jnp.maximum(jnp.sum(grid_mask), 1.0)
        pde_loss = jnp.sum((residual * grid_mask)**2) / n_valid

        # Exterior BC (u=0 on square boundary)
        bc_ext = (jnp.mean(spinn_forward(params, xb_left, yc)**2) +
                  jnp.mean(spinn_forward(params, xb_right, yc)**2) +
                  jnp.mean(spinn_forward(params, xc, yb_bot)**2) +
                  jnp.mean(spinn_forward(params, xc, yb_top)**2))

        # Hole BC (non-homogeneous, pointwise)
        u_hole_pred = spinn_forward_pointwise(params, x_hbc, y_hbc)
        bc_hole = jnp.mean((u_hole_pred - u_hbc)**2)

        bc_loss = bc_ext + bc_hole
        return pde_loss + bc_loss, (pde_loss, bc_loss)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    history = {'total_loss': [], 'pde_loss': [], 'bc_loss': [],
               'l2_error': [], 'eval_epochs': []}
    best_l2 = float('inf')
    best_params = params

    for _ in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

    start_time = time.time()
    for epoch in range(2, epochs):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == epochs - 1:
            u_pred_grid = spinn_forward(params, data['x_test_1d'],
                                        data['y_test_1d'])
            u_pred_np = np.array(u_pred_grid)
            l2_err = l2_relative_error_masked(u_pred_np, data['u_exact_test'],
                                              data['test_mask'])
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

    elapsed = time.time() - start_time
    ms_per_epoch = (elapsed / (epochs - 2)) * 1000

    print(f"  Training time: {elapsed:.1f}s ({ms_per_epoch:.2f} ms/epoch)")
    print(f"  Best L2 error: {best_l2:.4e}")
    print(f"  Final L2 error: {history['l2_error'][-1]:.4e}")

    return {
        'params': best_params,
        'total_loss': np.array(history['total_loss']),
        'pde_loss': np.array(history['pde_loss']),
        'bc_loss': np.array(history['bc_loss']),
        'l2_error': np.array(history['l2_error']),
        'eval_epochs': np.array(history['eval_epochs']),
        'total_time_sec': elapsed,
        'ms_per_epoch': ms_per_epoch,
        'best_l2_error': best_l2,
        'final_l2_error': history['l2_error'][-1],
        'total_params': n_params,
    }


# ################################################################
#  METHOD 3: SIREN
# ################################################################
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
    return siren_forward(params, jnp.concatenate([x, y], axis=-1))


def siren_loss_fn(params, x_pde, y_pde, x_bc, y_bc, u_bc):
    u_bc_pred = siren_u(params, x_bc, y_bc)
    bc_loss = jnp.mean((u_bc_pred - u_bc)**2)

    xy_pde = jnp.concatenate([x_pde, y_pde], axis=-1)

    def u_fn(xy):
        return siren_forward(params, xy)

    tangents_x = jnp.zeros_like(xy_pde).at[:, 0].set(1.0)
    tangents_y = jnp.zeros_like(xy_pde).at[:, 1].set(1.0)
    u_xx = hvp_fwdfwd(u_fn, xy_pde, tangents_x)
    u_yy = hvp_fwdfwd(u_fn, xy_pde, tangents_y)

    f_pde = source_term(x_pde, y_pde)
    residual = u_xx + u_yy + f_pde
    pde_loss = jnp.mean(residual**2)

    return bc_loss + pde_loss, (pde_loss, bc_loss)


# ################################################################
#  METHOD 4: FourierPINN
# ################################################################
def init_fourier_pinn(key, ff_dim=64, hidden_layers=None):
    if hidden_layers is None:
        hidden_layers = [128, 128, 128, 1]

    k1, k2, key = random.split(key, 3)
    W_x = _sample_frequencies(k1, ff_dim, W_CHAR).reshape(1, -1)
    W_y = _sample_frequencies(k2, ff_dim, W_CHAR).reshape(1, -1)

    input_dim = 4 * ff_dim
    params = {'W_x': W_x, 'W_y': W_y, 'mlp_layers': []}
    dims = [input_dim] + hidden_layers
    for i in range(len(dims) - 1):
        k, key = random.split(key)
        d_in, d_out = dims[i], dims[i + 1]
        limit = jnp.sqrt(6.0 / (d_in + d_out))
        w = random.uniform(k, (d_in, d_out), minval=-limit, maxval=limit)
        b = jnp.zeros((d_out,))
        params['mlp_layers'].append({'w': w, 'b': b})
    return params


def fourier_pinn_forward(params, xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
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
    return fourier_pinn_forward(params, jnp.concatenate([x, y], axis=-1))


def fourier_pinn_loss_fn(params, x_pde, y_pde, x_bc, y_bc, u_bc):
    u_bc_pred = fourier_pinn_u(params, x_bc, y_bc)
    bc_loss = jnp.mean((u_bc_pred - u_bc)**2)

    xy_pde = jnp.concatenate([x_pde, y_pde], axis=-1)

    def u_fn(xy):
        return fourier_pinn_forward(params, xy)

    tangents_x = jnp.zeros_like(xy_pde).at[:, 0].set(1.0)
    tangents_y = jnp.zeros_like(xy_pde).at[:, 1].set(1.0)
    u_xx = hvp_fwdfwd(u_fn, xy_pde, tangents_x)
    u_yy = hvp_fwdfwd(u_fn, xy_pde, tangents_y)

    f_pde = source_term(x_pde, y_pde)
    residual = u_xx + u_yy + f_pde
    pde_loss = jnp.mean(residual**2)

    return bc_loss + pde_loss, (pde_loss, bc_loss)


# ################################################################
#  METHOD 5: Vanilla PINN
# ################################################################
def init_pinn(key, layers_list=None):
    if layers_list is None:
        layers_list = [2, 128, 128, 128, 128, 1]
    params = {'layers': []}
    for i in range(len(layers_list) - 1):
        k, key = random.split(key)
        d_in, d_out = layers_list[i], layers_list[i + 1]
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
    return pinn_forward(params, jnp.concatenate([x, y], axis=-1))


def pinn_loss_fn(params, x_pde, y_pde, x_bc, y_bc, u_bc):
    u_bc_pred = pinn_u(params, x_bc, y_bc)
    bc_loss = jnp.mean((u_bc_pred - u_bc)**2)

    xy_pde = jnp.concatenate([x_pde, y_pde], axis=-1)

    def u_fn(xy):
        return pinn_forward(params, xy)

    tangents_x = jnp.zeros_like(xy_pde).at[:, 0].set(1.0)
    tangents_y = jnp.zeros_like(xy_pde).at[:, 1].set(1.0)
    u_xx = hvp_fwdfwd(u_fn, xy_pde, tangents_x)
    u_yy = hvp_fwdfwd(u_fn, xy_pde, tangents_y)

    f_pde = source_term(x_pde, y_pde)
    residual = u_xx + u_yy + f_pde
    pde_loss = jnp.mean(residual**2)

    return bc_loss + pde_loss, (pde_loss, bc_loss)


# ################################################################
#  Generic Training Loop (SIREN / FourierPINN / PINN)
# ################################################################
def train_pointwise_method(name, params, loss_fn, predict_fn, data,
                           epochs=EPOCHS):
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
        (loss, (pde_l, bc_l)), grads = value_and_grad(
            loss_fn, has_aux=True)(
            params, x_pde, y_pde, x_bc, y_bc, u_bc)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    total_loss_hist, pde_loss_hist, bc_loss_hist = [], [], []
    l2_error_hist, eval_epochs = [], []
    best_l2 = float('inf')
    best_params = params

    for _ in range(2):
        params, opt_state, loss_val, pde_val, bc_val = train_step(
            params, opt_state)

    start_time = time.time()
    for epoch in range(2, epochs):
        params, opt_state, loss_val, pde_val, bc_val = train_step(
            params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == epochs - 1:
            u_pred = predict_fn(params, data['x_test_flat'],
                                data['y_test_flat'])
            u_pred_np = np.array(u_pred).reshape(N_TEST, N_TEST)
            l2_err = l2_relative_error_masked(
                u_pred_np, data['u_exact_test'], data['test_mask'])
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
    ms_per_epoch = (elapsed / (epochs - 2)) * 1000

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


# ################################################################
#  Save Results
# ################################################################
def save_results(name, result, data):
    np.save(os.path.join(SAVE_DIR, f"{name}_params.npy"),
            jax.tree.map(np.array, result['params']), allow_pickle=True)

    np.savez(os.path.join(SAVE_DIR, f"{name}_history.npz"),
             total_loss=result['total_loss'],
             pde_loss=result['pde_loss'],
             bc_loss=result['bc_loss'],
             l2_error=result['l2_error'],
             eval_epochs=result['eval_epochs'])

    if name in ('SPINN', 'SVSNN_accel'):
        u_pred_np = result.get('u_pred_grid', None)
        if u_pred_np is None:
            u_pred_grid = spinn_forward(result['params'],
                                        data['x_test_1d'], data['y_test_1d'])
            u_pred_np = np.array(u_pred_grid)
    else:
        pred_map = {'SVSNN': svsnn_forward, 'SVSNN_orig': svsnn_forward,
                    'SIREN': siren_u, 'FourierPINN': fourier_pinn_u, 'PINN': pinn_u}
        pred_fn = pred_map.get(name)
        if pred_fn is None:
            return
        u_pred = pred_fn(result['params'], data['x_test_flat'],
                         data['y_test_flat'])
        u_pred_np = np.array(u_pred).reshape(N_TEST, N_TEST)

    np.savez(os.path.join(SAVE_DIR, f"{name}_prediction.npz"),
             u_pred=u_pred_np,
             u_exact=data['u_exact_test'],
             X=data['X_test'], Y=data['Y_test'],
             mask=data['test_mask'])

    summary = {
        'method': name,
        'total_params': int(result['total_params']),
        'total_time_sec': float(result['total_time_sec']),
        'best_l2_error': float(result['best_l2_error']),
        'final_l2_error': float(result['final_l2_error']),
        'ms_per_epoch': float(result['ms_per_epoch']),
    }
    with open(os.path.join(SAVE_DIR, f"{name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved results for {name}")


def save_comparison_table(all_results):
    fieldnames = ['method', 'total_params', 'total_time_sec',
                  'best_l2_error', 'final_l2_error', 'ms_per_epoch']
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


# ################################################################
#  MAIN
# ################################################################
def main():
    print("=" * 60)
    print("Ablation Study – Case 7: 2D Poisson Complex Multi-Hole Domain")
    print(f"  MU = 7*pi = {MU:.4f}")
    print(f"  Epochs = {EPOCHS}, LR = {LR}")
    print(f"  N_PDE = {N_PDE}, N_BC_ext = {N_BC_EXTERIOR}, "
          f"N_BC/hole = {N_BC_PER_HOLE}")
    print(f"  Device: {jax.devices()}")
    print("=" * 60)

    print("\nGenerating data...")
    data = generate_data()
    print("  Done.")

    key = random.PRNGKey(SEED)
    all_results = {}

    # ---- Method 1a: SV-SNN ACCELERATED ----
    params_accel, hist_accel, u_pred_accel, npar_accel, time_accel = run_svsnn_accelerated(data)
    result_accel = {
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
    all_results['SVSNN_accel'] = result_accel
    save_results('SVSNN_accel', result_accel, data)

    # ---- Method 1b: SV-SNN ORIGINAL ----
    params_sv, hist_sv, u_pred_sv, npar_sv, time_sv = run_svsnn(data)
    result_svsnn = {
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
    all_results['SVSNN_orig'] = result_svsnn
    save_results('SVSNN_orig', result_svsnn, data)

    # ---- Method 2: SPINN ----
    result_spinn = train_spinn(data)
    all_results['SPINN'] = result_spinn
    save_results('SPINN', result_spinn, data)

    # ---- Method 3: SIREN ----
    k3, key = random.split(key)
    siren_params = init_siren(k3)
    result_siren = train_pointwise_method(
        'SIREN', siren_params, siren_loss_fn, siren_u, data)
    all_results['SIREN'] = result_siren
    save_results('SIREN', result_siren, data)

    # ---- Method 4: FourierPINN ----
    k4, key = random.split(key)
    fp_params = init_fourier_pinn(k4)
    result_fp = train_pointwise_method(
        'FourierPINN', fp_params, fourier_pinn_loss_fn, fourier_pinn_u, data)
    all_results['FourierPINN'] = result_fp
    save_results('FourierPINN', result_fp, data)

    # ---- Method 5: PINN ----
    k5, key = random.split(key)
    pinn_params = init_pinn(k5)
    result_pinn = train_pointwise_method(
        'PINN', pinn_params, pinn_loss_fn, pinn_u, data)
    all_results['PINN'] = result_pinn
    save_results('PINN', result_pinn, data)

    # ---- Comparison table ----
    save_comparison_table(all_results)

    # ---- Print final summary ----
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    hdr = (f"{'Method':<14} {'Params':>10} {'Time(s)':>10} "
           f"{'Best L2':>12} {'Final L2':>12} {'ms/epoch':>10}")
    print(hdr)
    print("-" * 68)
    for name, r in all_results.items():
        print(f"{name:<14} {r['total_params']:>10,} "
              f"{r['total_time_sec']:>10.1f} "
              f"{r['best_l2_error']:>12.4e} {r['final_l2_error']:>12.4e} "
              f"{r['ms_per_epoch']:>10.2f}")
    print("=" * 60)
    print("\nAll results saved to:", SAVE_DIR)


CASE_INFO = {"id": "case7", "title": "2D Poisson complex multi-hole mu=7pi",
             "family": "elliptic", "has_classical": False}


def E11_run(method, budget, seed, epochs=None, target=None, save_pred_path=None):
    import _e11common
    return _e11common.run_modular_elliptic(
        sys.modules[__name__], method, budget, seed, epochs,
        target=target, save_pred_path=save_pred_path)


if __name__ == "__main__":
    main()
