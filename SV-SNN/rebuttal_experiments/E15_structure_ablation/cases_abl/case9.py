"""
SV-SNN Acceleration – Case 9: Steady Navier-Stokes with Double Cylinder
========================================================================
PDE: Steady 2D incompressible Navier-Stokes (rho=1, mu=1)
Domain: [-pi,pi]^2 minus two cylinders

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
W_CHAR = 2.0
FF_DIM = 64
SEED = 42
EPOCHS = 15000
LR = 1e-3
N_PDE_LHS = 5000
OVERSAMPLE = 1.3
N_BC_EXTERIOR = 200  # per edge
N_BC_CYLINDER = 100  # per cylinder
N_TEST = 15000
EVAL_EVERY = 100

RHO = 1.0
MU = 1.0
NU = MU / RHO

CYL1_CENTER = (-1.0, 0.5)
CYL1_RADIUS = 0.3
CYL2_CENTER = (1.0, -0.5)
CYL2_RADIUS = 0.3

DOMAIN_X = (-np.pi, np.pi)
DOMAIN_Y = (-np.pi, np.pi)

E11_OVR = {}  # E11 size overrides for matched budget (set by E11_run)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)


# ================================================================
#  PDE Definition
# ================================================================
def exact_u(x, y):
    return jnp.sin(2.0 * x) * jnp.cos(2.0 * y) + 0.5 * jnp.sin(x + y)


def exact_v(x, y):
    return -jnp.cos(2.0 * x) * jnp.sin(2.0 * y) - 0.5 * jnp.sin(x + y)


def exact_p(x, y):
    return jnp.zeros_like(x)


def source_x(x, y):
    return (jnp.sin(4.0 * x) - 0.25 * jnp.sin(x - 3.0 * y)
            + jnp.sin(x + y) + 8.0 * jnp.sin(2.0 * x) * jnp.cos(2.0 * y)
            + 0.75 * jnp.sin(3.0 * x - y))


def source_y(x, y):
    return (jnp.sin(4.0 * y) - 0.75 * jnp.sin(x - 3.0 * y)
            - jnp.sin(x + y) - 8.0 * jnp.cos(2.0 * x) * jnp.sin(2.0 * y)
            + 0.25 * jnp.sin(3.0 * x - y))


# ================================================================
#  Geometry Utilities
# ================================================================
def is_inside_cylinder(x, y):
    d1 = (x - CYL1_CENTER[0])**2 + (y - CYL1_CENTER[1])**2
    d2 = (x - CYL2_CENTER[0])**2 + (y - CYL2_CENTER[1])**2
    return (d1 < CYL1_RADIUS**2) | (d2 < CYL2_RADIUS**2)


def remove_cylinder_points(x, y):
    mask = ~is_inside_cylinder(x.ravel(), y.ravel())
    return x[mask].reshape(-1, 1), y[mask].reshape(-1, 1)


# ================================================================
#  Data Generation
# ================================================================
def generate_bc_data(key):
    """BC on exterior rectangle (200 per edge) + cylinders (100 per cylinder)."""
    n_ext = N_BC_EXTERIOR
    xlo, xhi = DOMAIN_X
    ylo, yhi = DOMAIN_Y

    t = np.linspace(xlo, xhi, n_ext).reshape(-1, 1)
    ty = np.linspace(ylo, yhi, n_ext).reshape(-1, 1)

    x_bottom = t;          y_bottom = np.full_like(t, ylo)
    x_top = t;             y_top = np.full_like(t, yhi)
    x_left = np.full_like(ty, xlo); y_left = ty
    x_right = np.full_like(ty, xhi); y_right = ty

    x_ext = np.concatenate([x_bottom, x_top, x_left, x_right], axis=0)
    y_ext = np.concatenate([y_bottom, y_top, y_left, y_right], axis=0)

    # Cylinder BCs
    theta1 = np.linspace(0, 2 * np.pi, N_BC_CYLINDER, endpoint=False).reshape(-1, 1)
    x_c1 = CYL1_CENTER[0] + CYL1_RADIUS * np.cos(theta1)
    y_c1 = CYL1_CENTER[1] + CYL1_RADIUS * np.sin(theta1)

    theta2 = np.linspace(0, 2 * np.pi, N_BC_CYLINDER, endpoint=False).reshape(-1, 1)
    x_c2 = CYL2_CENTER[0] + CYL2_RADIUS * np.cos(theta2)
    y_c2 = CYL2_CENTER[1] + CYL2_RADIUS * np.sin(theta2)

    x_bc = np.concatenate([x_ext, x_c1, x_c2], axis=0)
    y_bc = np.concatenate([y_ext, y_c1, y_c2], axis=0)

    x_bc = jnp.array(x_bc, dtype=jnp.float32)
    y_bc = jnp.array(y_bc, dtype=jnp.float32)
    u_bc = exact_u(x_bc, y_bc)
    v_bc = exact_v(x_bc, y_bc)
    return x_bc, y_bc, u_bc, v_bc


def generate_pde_data(key):
    """LHS collocation with oversampling, remove cylinder interiors."""
    n_sample = int(N_PDE_LHS * OVERSAMPLE)
    pts = lhs(2, samples=n_sample)  # plain LHS (maximin too slow for repeated runs)
    xlo, xhi = DOMAIN_X
    ylo, yhi = DOMAIN_Y
    x = pts[:, 0:1] * (xhi - xlo) + xlo
    y = pts[:, 1:2] * (yhi - ylo) + ylo
    x, y = remove_cylinder_points(x, y)
    x = jnp.array(x, dtype=jnp.float32)
    y = jnp.array(y, dtype=jnp.float32)
    return x, y


def generate_test_data(key):
    """Random test points outside cylinders."""
    rng = np.random.RandomState(SEED + 999)
    pts = []
    xlo, xhi = DOMAIN_X
    ylo, yhi = DOMAIN_Y
    while len(pts) < N_TEST:
        batch = rng.uniform(size=(N_TEST * 2, 2))
        x_b = batch[:, 0] * (xhi - xlo) + xlo
        y_b = batch[:, 1] * (yhi - ylo) + ylo
        mask = ~is_inside_cylinder(x_b, y_b)
        good = np.stack([x_b[mask], y_b[mask]], axis=1)
        pts.append(good)
    pts = np.concatenate(pts, axis=0)[:N_TEST]
    x_test = jnp.array(pts[:, 0:1], dtype=jnp.float32)
    y_test = jnp.array(pts[:, 1:2], dtype=jnp.float32)
    u_exact = exact_u(x_test, y_test)
    v_exact = exact_v(x_test, y_test)
    return x_test, y_test, u_exact, v_exact


# ================================================================
#  Utilities
# ================================================================
def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params))


def compute_l2_error(pred, exact):
    return float(jnp.sqrt(jnp.sum((pred - exact)**2) / jnp.sum(exact**2)))


def compute_ns_l2(u_pred, v_pred, u_exact, v_exact):
    l2_u = compute_l2_error(u_pred, u_exact)
    l2_v = compute_l2_error(v_pred, v_exact)
    return (l2_u + l2_v) / 2.0


FREQ_SCALE = 8.0

def _sample_frequencies(rng_key, K):
    n_basic = K * 2 // 5
    n_char = K * 2 // 5
    n_high = K - n_basic - n_char
    freqs_basic = jnp.linspace(0.5, 5.0, n_basic)
    freqs_char = jnp.abs(jax.random.normal(rng_key, (n_char,)) * 0.5 + 2.0)
    k1, k2 = jax.random.split(rng_key)
    freqs_high = jax.random.uniform(k2, (n_high,), minval=5.0, maxval=FREQ_SCALE)
    return jnp.concatenate([freqs_basic, freqs_char, freqs_high])


# ################################################################
#  METHOD 1: SV-SNN
# ################################################################
def run_svsnn():
    print("\n" + "=" * 70)
    print("  METHOD 1 / 5 :  SV-SNN")
    print("=" * 70)

    NUM_MODES = 4
    NUM_FREQ = 16
    FREQ_SCALE = 8.0
    key = random.PRNGKey(SEED)

    def _sample_frequencies(rng_key, K):
        import _abl, abl_freqs
        if _abl.STRATEGY != 'default':
            return abl_freqs.strategy_sample(rng_key, K, 2.0, FREQ_SCALE, _abl.STRATEGY) * _abl.SCALE
        n_basic = K * 2 // 5
        n_char = K * 2 // 5
        n_high = K - n_basic - n_char
        freqs_basic = jnp.linspace(0.5, 5.0, n_basic)
        freqs_char = jnp.abs(jax.random.normal(rng_key, (n_char,)) * 0.5 + 2.0)
        k1, k2 = jax.random.split(rng_key)
        freqs_high = jax.random.uniform(k2, (n_high,), minval=5.0, maxval=FREQ_SCALE)
        return jnp.concatenate([freqs_basic, freqs_char, freqs_high]) * _abl.SCALE

    def init_params(rng_key):
        keys = jax.random.split(rng_key, NUM_MODES * 6 + 10)
        ki = 0
        sx, sy = [], []
        for _ in range(NUM_MODES):
            for s_list in [sx, sy]:
                s_list.append({
                    'freqs': _sample_frequencies(keys[ki], NUM_FREQ),
                    'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                    'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                    'bias': jnp.zeros(1),
                })
                ki += 3
        return {
            'spatial_x': sx, 'spatial_y': sy,
            'mode_coeffs_u': jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1,
            'mode_coeffs_v': jax.random.normal(keys[ki+1], (NUM_MODES,)) * 0.1,
            'mode_coeffs_p': jax.random.normal(keys[ki+2], (NUM_MODES,)) * 0.1,
            'global_bias_u': jnp.zeros(1),
            'global_bias_v': jnp.zeros(1),
            'global_bias_p': jnp.zeros(1),
        }

    params = init_params(key)

    def spatial_forward(sp, x):
        freqs = jax.lax.stop_gradient(sp['freqs'])
        wx = freqs[None, :] * x
        return (jnp.sum(sp['cos_c'] * jnp.cos(wx)
                        + sp['sin_c'] * jnp.sin(wx),
                        axis=1, keepdims=True) + sp['bias'])

    def forward_u(params, x, y):
        u = jnp.zeros_like(x)
        for n in range(NUM_MODES):
            u += (params['mode_coeffs_u'][n]
                  * spatial_forward(params['spatial_x'][n], x)
                  * spatial_forward(params['spatial_y'][n], y))
        return u + params['global_bias_u']

    def forward_v(params, x, y):
        v = jnp.zeros_like(x)
        for n in range(NUM_MODES):
            v += (params['mode_coeffs_v'][n]
                  * spatial_forward(params['spatial_x'][n], x)
                  * spatial_forward(params['spatial_y'][n], y))
        return v + params['global_bias_v']

    def forward_p(params, x, y):
        p = jnp.zeros_like(x)
        for n in range(NUM_MODES):
            p += (params['mode_coeffs_p'][n]
                  * spatial_forward(params['spatial_x'][n], x)
                  * spatial_forward(params['spatial_y'][n], y))
        return p + params['global_bias_p']

    def ns_residual_single(params, x_s, y_s):
        def u_fn(xv, yv):
            return forward_u(params, xv[None, None], yv[None, None]).squeeze()
        def v_fn(xv, yv):
            return forward_v(params, xv[None, None], yv[None, None]).squeeze()
        def p_fn(xv, yv):
            return forward_p(params, xv[None, None], yv[None, None]).squeeze()

        u_val = u_fn(x_s, y_s)
        v_val = v_fn(x_s, y_s)

        u_x = jax.grad(u_fn, 0)(x_s, y_s)
        u_y = jax.grad(u_fn, 1)(x_s, y_s)
        u_xx = jax.grad(jax.grad(u_fn, 0), 0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, 1), 1)(x_s, y_s)

        v_x = jax.grad(v_fn, 0)(x_s, y_s)
        v_y = jax.grad(v_fn, 1)(x_s, y_s)
        v_xx = jax.grad(jax.grad(v_fn, 0), 0)(x_s, y_s)
        v_yy = jax.grad(jax.grad(v_fn, 1), 1)(x_s, y_s)

        p_x = jax.grad(p_fn, 0)(x_s, y_s)
        p_y = jax.grad(p_fn, 1)(x_s, y_s)

        sx = source_x(x_s, y_s)
        sy = source_y(x_s, y_s)

        res_u = u_val * u_x + v_val * u_y + p_x / RHO - NU * (u_xx + u_yy) - sx
        res_v = u_val * v_x + v_val * v_y + p_y / RHO - NU * (v_xx + v_yy) - sy
        res_div = u_x + v_y

        return res_u, res_v, res_div

    ns_residual_batch = jax.vmap(ns_residual_single, in_axes=(None, 0, 0))

    # --- Data ---
    key = random.PRNGKey(SEED + 100)
    x_bc, y_bc, u_bc, v_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    key, subkey = random.split(key)
    x_test, y_test, u_exact_test, v_exact_test = generate_test_data(subkey)

    x_pde_flat = x_pde.ravel()
    y_pde_flat = y_pde.ravel()

    def loss_fn(params):
        u_pred_bc = forward_u(params, x_bc, y_bc)
        v_pred_bc = forward_v(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2) + jnp.mean((v_pred_bc - v_bc)**2)

        res_u, res_v, res_div = ns_residual_batch(params, x_pde_flat, y_pde_flat)
        loss_pde = jnp.mean(res_u**2) + jnp.mean(res_v**2) + jnp.mean(res_div**2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred = forward_u(params, x_test, y_test)
            v_pred = forward_v(params, x_test, y_test)
            l2_err = compute_ns_l2(u_pred, v_pred, u_exact_test, v_exact_test)
            history["total_loss"].append(float(loss_val))
            history["pde_loss"].append(float(pde_val))
            history["bc_loss"].append(float(bc_val))
            history["l2_error"].append(l2_err)
            history["eval_epochs"].append(epoch)
            if l2_err < best_l2:
                best_l2 = l2_err
            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss {float(loss_val):.4e} | "
                      f"PDE {float(pde_val):.4e} | BC {float(bc_val):.4e} | "
                      f"L2 {l2_err:.4e}")

    total_time = time.time() - t0
    u_pred_final = np.array(forward_u(params, x_test, y_test))
    v_pred_final = np.array(forward_v(params, x_test, y_test))
    p_pred_final = np.array(forward_p(params, x_test, y_test))
    final_l2 = compute_ns_l2(jnp.array(u_pred_final), jnp.array(v_pred_final),
                             u_exact_test, v_exact_test)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SVSNN", params, history, u_pred_final, v_pred_final, p_pred_final,
            np.array(u_exact_test), np.array(v_exact_test),
            np.array(x_test), np.array(y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 1b: SV-SNN ACCELERATED
# ################################################################
def run_svsnn_accelerated():
    print("\n" + "=" * 70)
    print("  SV-SNN (ACCELERATED)")
    print("  Analytic spatial 1st+2nd derivs, vectorized, grid + mask")
    print("=" * 70)

    NUM_MODES = 4
    NUM_FREQ = 16
    FREQ_SCALE = 8.0
    NC = 80
    key = random.PRNGKey(SEED)

    def _sample_freqs(rng_key, K):
        n_basic = K * 2 // 5
        n_char = K * 2 // 5
        n_high = K - n_basic - n_char
        freqs_basic = jnp.linspace(0.5, 5.0, n_basic)
        freqs_char = jnp.abs(jax.random.normal(rng_key, (n_char,)) * 0.5 + 2.0)
        k1, k2 = jax.random.split(rng_key)
        freqs_high = jax.random.uniform(k2, (n_high,), minval=5.0, maxval=FREQ_SCALE)
        return jnp.concatenate([freqs_basic, freqs_char, freqs_high])

    def init_params(rng_key):
        keys = jax.random.split(rng_key, NUM_MODES * 6 + 10)
        ki = 0
        sx, sy = [], []
        for _ in range(NUM_MODES):
            for s_list in [sx, sy]:
                s_list.append({
                    'freqs': _sample_freqs(keys[ki], NUM_FREQ),
                    'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                    'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                    'bias': jnp.zeros(1),
                })
                ki += 3
        return {
            'spatial_x': sx, 'spatial_y': sy,
            'mode_coeffs_u': jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1,
            'mode_coeffs_v': jax.random.normal(keys[ki+1], (NUM_MODES,)) * 0.1,
            'mode_coeffs_p': jax.random.normal(keys[ki+2], (NUM_MODES,)) * 0.1,
            'global_bias_u': jnp.zeros(1),
            'global_bias_v': jnp.zeros(1),
            'global_bias_p': jnp.zeros(1),
        }

    params = init_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params}")

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
        trig = all_cos_c[None, :, :] * cw + all_sin_c[None, :, :] * sw
        vals = jnp.sum(trig, axis=-1) + all_bias[None, :, 0]
        d1 = jnp.sum(all_freqs[None, :, :] * (-all_cos_c[None, :, :] * sw + all_sin_c[None, :, :] * cw), axis=-1)
        d2 = jnp.sum(-all_freqs[None, :, :] ** 2 * trig, axis=-1)
        return vals, d1, d2

    def _compute_basis(coord_flat, all_freqs, all_cos_c, all_sin_c, all_bias):
        wz = coord_flat[:, None, None] * all_freqs[None, :, :]
        trig = all_cos_c[None, :, :] * jnp.cos(wz) + all_sin_c[None, :, :] * jnp.sin(wz)
        vals = jnp.sum(trig, axis=-1) + all_bias[None, :, 0]
        return vals

    xlo, xhi = DOMAIN_X
    ylo, yhi = DOMAIN_Y
    xc = jnp.linspace(xlo, xhi, NC).reshape(-1, 1)
    yc = jnp.linspace(ylo, yhi, NC).reshape(-1, 1)
    X_mesh, Y_mesh = jnp.meshgrid(xc.squeeze(), yc.squeeze(), indexing='ij')
    cyl_mask = is_inside_cylinder(np.array(X_mesh), np.array(Y_mesh))
    grid_mask = jnp.array((~cyl_mask).astype(np.float32))

    sx_grid = source_x(X_mesh, Y_mesh) * grid_mask
    sy_grid = source_y(X_mesh, Y_mesh) * grid_mask

    key_data = random.PRNGKey(SEED + 100)
    x_bc, y_bc, u_bc, v_bc = generate_bc_data(key_data)
    key_data, subkey = random.split(key_data)
    x_test, y_test, u_exact_test, v_exact_test = generate_test_data(subkey)

    def vectorized_forward_uvp(params, x, y):
        fx, cx, sx_p, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy_p, by = _stack_spatial(params, 'spatial_y')
        X_all = _compute_basis(x.squeeze(), fx, cx, sx_p, bx)
        Y_all = _compute_basis(y.squeeze(), fy, cy, sy_p, by)
        mode = X_all * Y_all
        u = jnp.sum(params['mode_coeffs_u'][None, :] * mode, axis=-1, keepdims=True) + params['global_bias_u']
        v = jnp.sum(params['mode_coeffs_v'][None, :] * mode, axis=-1, keepdims=True) + params['global_bias_v']
        p = jnp.sum(params['mode_coeffs_p'][None, :] * mode, axis=-1, keepdims=True) + params['global_bias_p']
        return u, v, p

    def vectorized_pde_residual(params):
        fx, cx, sx_p, bx = _stack_spatial(params, 'spatial_x')
        fy, cy, sy_p, by = _stack_spatial(params, 'spatial_y')

        X, dX, d2X = _compute_basis_with_derivs(xc.squeeze(), fx, cx, sx_p, bx)
        Y, dY, d2Y = _compute_basis_with_derivs(yc.squeeze(), fy, cy, sy_p, by)

        cu = params['mode_coeffs_u']
        cv = params['mode_coeffs_v']
        cp = params['mode_coeffs_p']

        def field_2d(coeffs, A, B):
            cA = coeffs[None, :] * A
            return jnp.einsum('nm,jm->nj', cA, B)

        u_val = field_2d(cu, X, Y) + params['global_bias_u']
        v_val = field_2d(cv, X, Y) + params['global_bias_v']

        u_x = field_2d(cu, dX, Y)
        u_y = field_2d(cu, X, dY)
        u_xx = field_2d(cu, d2X, Y)
        u_yy = field_2d(cu, X, d2Y)

        v_x = field_2d(cv, dX, Y)
        v_y = field_2d(cv, X, dY)
        v_xx = field_2d(cv, d2X, Y)
        v_yy = field_2d(cv, X, d2Y)

        p_x = field_2d(cp, dX, Y)
        p_y = field_2d(cp, X, dY)

        r_u = (u_val * u_x + v_val * u_y + p_x / RHO - NU * (u_xx + u_yy) - sx_grid) * grid_mask
        r_v = (u_val * v_x + v_val * v_y + p_y / RHO - NU * (v_xx + v_yy) - sy_grid) * grid_mask
        r_d = (u_x + v_y) * grid_mask

        return r_u, r_v, r_d

    def loss_fn(params):
        u_pred, v_pred, _ = vectorized_forward_uvp(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred - u_bc)**2) + jnp.mean((v_pred - v_bc)**2)

        r_u, r_v, r_d = vectorized_pde_residual(params)
        n_valid = jnp.sum(grid_mask)
        loss_pde = (jnp.sum(r_u**2) + jnp.sum(r_v**2) + jnp.sum(r_d**2)) / jnp.maximum(n_valid, 1.0)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state_new, loss, pde_l, bc_l

    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred, v_pred, _ = vectorized_forward_uvp(params, x_test, y_test)
            l2_err = compute_ns_l2(u_pred, v_pred, u_exact_test, v_exact_test)
            history["total_loss"].append(float(loss_val))
            history["pde_loss"].append(float(pde_val))
            history["bc_loss"].append(float(bc_val))
            history["l2_error"].append(l2_err)
            history["eval_epochs"].append(epoch)
            if l2_err < best_l2:
                best_l2 = l2_err
            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss {float(loss_val):.4e} | "
                      f"PDE {float(pde_val):.4e} | BC {float(bc_val):.4e} | "
                      f"L2 {l2_err:.4e}")

    total_time = time.time() - t0
    u_pred_f = np.array(vectorized_forward_uvp(params, x_test, y_test)[0])
    v_pred_f = np.array(vectorized_forward_uvp(params, x_test, y_test)[1])
    p_pred_f = np.array(vectorized_forward_uvp(params, x_test, y_test)[2])
    final_l2 = compute_ns_l2(jnp.array(u_pred_f), jnp.array(v_pred_f),
                             u_exact_test, v_exact_test)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SVSNN_accel", params, history, u_pred_f, v_pred_f, p_pred_f,
            np.array(u_exact_test), np.array(v_exact_test),
            np.array(x_test), np.array(y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 2: SPINN (pointwise fallback for complex domain)
# ################################################################
def run_spinn():
    print("\n" + "=" * 70)
    print("  METHOD 2 / 5 :  SPINN")
    print("=" * 70)

    FEATURES = E11_OVR.get('spinn_features', 64)
    N_LAYERS = E11_OVR.get('spinn_n_layers', 4)
    R = E11_OVR.get('spinn_r', 64)
    FF_DIM = E11_OVR.get('spinn_ff', 64)
    key = random.PRNGKey(SEED)

    ff_input_dim = 2 * FF_DIM

    def init_mlp(key, in_dim, features, n_layers, out_dim):
        params = []
        fan_in = in_dim
        for _ in range(n_layers):
            fan_out = features
            key, k1 = random.split(key)
            W = random.normal(k1, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros(fan_out)
            params.append({"W": W, "b": b})
            fan_in = fan_out
        key, k1 = random.split(key)
        W_out = random.normal(k1, (features, out_dim)) * jnp.sqrt(2.0 / features)
        b_out = jnp.zeros(out_dim)
        params.append({"W": W_out, "b": b_out})
        return params

    key, k1, k2, k3, k4, k5 = random.split(key, 6)
    branch_x_params = init_mlp(k1, ff_input_dim, FEATURES, N_LAYERS, R)
    branch_y_params = init_mlp(k2, ff_input_dim, FEATURES, N_LAYERS, R)
    head_u_params = [{"W": random.normal(k3, (R * 2, 3)) * 0.01, "b": jnp.zeros(3)}]

    params = {
        "branch_x": branch_x_params,
        "branch_y": branch_y_params,
        "head": head_u_params,
        "W_x": _sample_frequencies(k4, FF_DIM).reshape(1, -1),
        "W_y": _sample_frequencies(k5, FF_DIM).reshape(1, -1),
    }

    def fourier_embed(coord, W):
        return jnp.concatenate([jnp.sin(coord @ W), jnp.cos(coord @ W)], axis=-1)

    def mlp_forward(mlp_params, x):
        h = x
        for layer in mlp_params[:-1]:
            h = h @ layer["W"] + layer["b"]
            h = jnp.tanh(h)
        out_layer = mlp_params[-1]
        return h @ out_layer["W"] + out_layer["b"]

    def forward(params, x, y):
        x_emb = fourier_embed(x, params["W_x"])
        y_emb = fourier_embed(y, params["W_y"])
        Vx = mlp_forward(params["branch_x"], x_emb)  # (N, R)
        Vy = mlp_forward(params["branch_y"], y_emb)  # (N, R)
        features = jnp.concatenate([Vx, Vy], axis=-1)  # (N, 2R)
        out = features @ params["head"][0]["W"] + params["head"][0]["b"]  # (N, 3)
        return out

    def forward_uvp(params, x, y):
        out = forward(params, x, y)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def ns_residual_single(params, x_s, y_s):
        def u_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 0]
        def v_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 1]
        def p_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 2]

        u_val = u_fn(x_s, y_s)
        v_val = v_fn(x_s, y_s)

        u_x = jax.grad(u_fn, 0)(x_s, y_s)
        u_y = jax.grad(u_fn, 1)(x_s, y_s)
        u_xx = jax.grad(jax.grad(u_fn, 0), 0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, 1), 1)(x_s, y_s)

        v_x = jax.grad(v_fn, 0)(x_s, y_s)
        v_y = jax.grad(v_fn, 1)(x_s, y_s)
        v_xx = jax.grad(jax.grad(v_fn, 0), 0)(x_s, y_s)
        v_yy = jax.grad(jax.grad(v_fn, 1), 1)(x_s, y_s)

        p_x = jax.grad(p_fn, 0)(x_s, y_s)
        p_y = jax.grad(p_fn, 1)(x_s, y_s)

        sx = source_x(x_s, y_s)
        sy = source_y(x_s, y_s)

        res_u = u_val * u_x + v_val * u_y + p_x / RHO - NU * (u_xx + u_yy) - sx
        res_v = u_val * v_x + v_val * v_y + p_y / RHO - NU * (v_xx + v_yy) - sy
        res_div = u_x + v_y
        return res_u, res_v, res_div

    ns_residual_batch = jax.vmap(ns_residual_single, in_axes=(None, 0, 0))

    # --- Data ---
    key = random.PRNGKey(SEED + 200)
    x_bc, y_bc, u_bc, v_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    key, subkey = random.split(key)
    x_test, y_test, u_exact_test, v_exact_test = generate_test_data(subkey)

    x_pde_flat = x_pde.ravel()
    y_pde_flat = y_pde.ravel()

    def loss_fn(params):
        u_pred_bc, v_pred_bc, _ = forward_uvp(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2) + jnp.mean((v_pred_bc - v_bc)**2)

        res_u, res_v, res_div = ns_residual_batch(params, x_pde_flat, y_pde_flat)
        loss_pde = jnp.mean(res_u**2) + jnp.mean(res_v**2) + jnp.mean(res_div**2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred, v_pred, _ = forward_uvp(params, x_test, y_test)
            l2_err = compute_ns_l2(u_pred, v_pred, u_exact_test, v_exact_test)
            history["total_loss"].append(float(loss_val))
            history["pde_loss"].append(float(pde_val))
            history["bc_loss"].append(float(bc_val))
            history["l2_error"].append(l2_err)
            history["eval_epochs"].append(epoch)
            if l2_err < best_l2:
                best_l2 = l2_err
            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss {float(loss_val):.4e} | "
                      f"PDE {float(pde_val):.4e} | BC {float(bc_val):.4e} | "
                      f"L2 {l2_err:.4e}")

    total_time = time.time() - t0
    u_pred_final, v_pred_final, p_pred_final = forward_uvp(params, x_test, y_test)
    u_pred_final = np.array(u_pred_final)
    v_pred_final = np.array(v_pred_final)
    p_pred_final = np.array(p_pred_final)
    final_l2 = compute_ns_l2(jnp.array(u_pred_final), jnp.array(v_pred_final),
                             u_exact_test, v_exact_test)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SPINN", params, history, u_pred_final, v_pred_final, p_pred_final,
            np.array(u_exact_test), np.array(v_exact_test),
            np.array(x_test), np.array(y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 3: SIREN
# ################################################################
def run_siren():
    print("\n" + "=" * 70)
    print("  METHOD 3 / 5 :  SIREN")
    print("=" * 70)

    key = random.PRNGKey(SEED)
    key, k_wx, k_wy = random.split(key, 3)
    _ff = E11_OVR.get('ff', FF_DIM)
    _hid = E11_OVR.get('hidden', 128)
    _nh = E11_OVR.get('n_hidden', 4)
    siren_W_x = _sample_frequencies(k_wx, _ff).reshape(1, -1)
    siren_W_y = _sample_frequencies(k_wy, _ff).reshape(1, -1)
    ff_input_dim_siren = 4 * _ff
    LAYERS = [ff_input_dim_siren] + [_hid] * _nh + [3]

    def init_siren(key, layers):
        mlp = []
        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i + 1]
            key, k = random.split(key)
            std = jnp.sqrt(2.0 / fan_in)
            W = random.normal(k, (fan_in, fan_out)) * std
            b = jnp.zeros(fan_out)
            mlp.append({"W": W, "b": b})
        return mlp

    siren_mlp = init_siren(key, LAYERS)
    params = {'mlp': siren_mlp, 'W_x': siren_W_x, 'W_y': siren_W_y}

    def forward(params, x, y):
        Hx = jnp.concatenate([jnp.sin(x @ params['W_x']), jnp.cos(x @ params['W_x'])], axis=-1)
        Hy = jnp.concatenate([jnp.sin(y @ params['W_y']), jnp.cos(y @ params['W_y'])], axis=-1)
        h = jnp.concatenate([Hx, Hy], axis=-1)
        for layer in params['mlp'][:-1]:
            h = jnp.sin(h @ layer["W"] + layer["b"])
        last = params['mlp'][-1]
        return h @ last["W"] + last["b"]

    def forward_uvp(params, x, y):
        out = forward(params, x, y)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def ns_residual_single(params, x_s, y_s):
        def u_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 0]
        def v_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 1]
        def p_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 2]

        u_val = u_fn(x_s, y_s)
        v_val = v_fn(x_s, y_s)

        u_x = jax.grad(u_fn, 0)(x_s, y_s)
        u_y = jax.grad(u_fn, 1)(x_s, y_s)
        u_xx = jax.grad(jax.grad(u_fn, 0), 0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, 1), 1)(x_s, y_s)

        v_x = jax.grad(v_fn, 0)(x_s, y_s)
        v_y = jax.grad(v_fn, 1)(x_s, y_s)
        v_xx = jax.grad(jax.grad(v_fn, 0), 0)(x_s, y_s)
        v_yy = jax.grad(jax.grad(v_fn, 1), 1)(x_s, y_s)

        p_x = jax.grad(p_fn, 0)(x_s, y_s)
        p_y = jax.grad(p_fn, 1)(x_s, y_s)

        sx = source_x(x_s, y_s)
        sy = source_y(x_s, y_s)

        res_u = u_val * u_x + v_val * u_y + p_x / RHO - NU * (u_xx + u_yy) - sx
        res_v = u_val * v_x + v_val * v_y + p_y / RHO - NU * (v_xx + v_yy) - sy
        res_div = u_x + v_y
        return res_u, res_v, res_div

    ns_residual_batch = jax.vmap(ns_residual_single, in_axes=(None, 0, 0))

    # --- Data ---
    key = random.PRNGKey(SEED + 300)
    x_bc, y_bc, u_bc, v_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    key, subkey = random.split(key)
    x_test, y_test, u_exact_test, v_exact_test = generate_test_data(subkey)

    x_pde_flat = x_pde.ravel()
    y_pde_flat = y_pde.ravel()

    def loss_fn(params):
        u_pred_bc, v_pred_bc, _ = forward_uvp(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2) + jnp.mean((v_pred_bc - v_bc)**2)

        res_u, res_v, res_div = ns_residual_batch(params, x_pde_flat, y_pde_flat)
        loss_pde = jnp.mean(res_u**2) + jnp.mean(res_v**2) + jnp.mean(res_div**2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred, v_pred, _ = forward_uvp(params, x_test, y_test)
            l2_err = compute_ns_l2(u_pred, v_pred, u_exact_test, v_exact_test)
            history["total_loss"].append(float(loss_val))
            history["pde_loss"].append(float(pde_val))
            history["bc_loss"].append(float(bc_val))
            history["l2_error"].append(l2_err)
            history["eval_epochs"].append(epoch)
            if l2_err < best_l2:
                best_l2 = l2_err
            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss {float(loss_val):.4e} | "
                      f"PDE {float(pde_val):.4e} | BC {float(bc_val):.4e} | "
                      f"L2 {l2_err:.4e}")

    total_time = time.time() - t0
    u_pred_final, v_pred_final, p_pred_final = forward_uvp(params, x_test, y_test)
    u_pred_final = np.array(u_pred_final)
    v_pred_final = np.array(v_pred_final)
    p_pred_final = np.array(p_pred_final)
    final_l2 = compute_ns_l2(jnp.array(u_pred_final), jnp.array(v_pred_final),
                             u_exact_test, v_exact_test)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SIREN", params, history, u_pred_final, v_pred_final, p_pred_final,
            np.array(u_exact_test), np.array(v_exact_test),
            np.array(x_test), np.array(y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 4: FourierPINN
# ################################################################
def run_fourierpinn():
    print("\n" + "=" * 70)
    print("  METHOD 4 / 5 :  FourierPINN")
    print("=" * 70)

    FF_DIM = E11_OVR.get('ff', 64)
    MLP_LAYERS = [E11_OVR.get('hidden', 128)] * E11_OVR.get('n_hidden', 3) + [3]
    key = random.PRNGKey(SEED)

    key, k1, k2 = random.split(key, 3)
    W_x = _sample_frequencies(k1, FF_DIM).reshape(1, -1)
    W_y = _sample_frequencies(k2, FF_DIM).reshape(1, -1)

    def init_mlp(key, in_dim, layer_sizes):
        params = []
        fan_in = in_dim
        for fan_out in layer_sizes:
            key, k = random.split(key)
            W = random.normal(k, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros(fan_out)
            params.append({"W": W, "b": b})
            fan_in = fan_out
        return params

    key, k = random.split(key)
    mlp_in_dim = 4 * FF_DIM
    mlp_params = init_mlp(k, mlp_in_dim, MLP_LAYERS)

    params = {"W_x": W_x, "W_y": W_y, "mlp": mlp_params}

    def forward(params, x, y):
        Hx_cos = jnp.cos(x @ params["W_x"])
        Hx_sin = jnp.sin(x @ params["W_x"])
        Hy_cos = jnp.cos(y @ params["W_y"])
        Hy_sin = jnp.sin(y @ params["W_y"])
        h = jnp.concatenate([Hx_cos, Hx_sin, Hy_cos, Hy_sin], axis=-1)
        for layer in params["mlp"][:-1]:
            h = jnp.tanh(h @ layer["W"] + layer["b"])
        last = params["mlp"][-1]
        return h @ last["W"] + last["b"]

    def forward_uvp(params, x, y):
        out = forward(params, x, y)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def ns_residual_single(params, x_s, y_s):
        def u_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 0]
        def v_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 1]
        def p_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 2]

        u_val = u_fn(x_s, y_s)
        v_val = v_fn(x_s, y_s)

        u_x = jax.grad(u_fn, 0)(x_s, y_s)
        u_y = jax.grad(u_fn, 1)(x_s, y_s)
        u_xx = jax.grad(jax.grad(u_fn, 0), 0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, 1), 1)(x_s, y_s)

        v_x = jax.grad(v_fn, 0)(x_s, y_s)
        v_y = jax.grad(v_fn, 1)(x_s, y_s)
        v_xx = jax.grad(jax.grad(v_fn, 0), 0)(x_s, y_s)
        v_yy = jax.grad(jax.grad(v_fn, 1), 1)(x_s, y_s)

        p_x = jax.grad(p_fn, 0)(x_s, y_s)
        p_y = jax.grad(p_fn, 1)(x_s, y_s)

        sx = source_x(x_s, y_s)
        sy = source_y(x_s, y_s)

        res_u = u_val * u_x + v_val * u_y + p_x / RHO - NU * (u_xx + u_yy) - sx
        res_v = u_val * v_x + v_val * v_y + p_y / RHO - NU * (v_xx + v_yy) - sy
        res_div = u_x + v_y
        return res_u, res_v, res_div

    ns_residual_batch = jax.vmap(ns_residual_single, in_axes=(None, 0, 0))

    # --- Data ---
    key = random.PRNGKey(SEED + 400)
    x_bc, y_bc, u_bc, v_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    key, subkey = random.split(key)
    x_test, y_test, u_exact_test, v_exact_test = generate_test_data(subkey)

    x_pde_flat = x_pde.ravel()
    y_pde_flat = y_pde.ravel()

    def loss_fn(params):
        u_pred_bc, v_pred_bc, _ = forward_uvp(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2) + jnp.mean((v_pred_bc - v_bc)**2)

        res_u, res_v, res_div = ns_residual_batch(params, x_pde_flat, y_pde_flat)
        loss_pde = jnp.mean(res_u**2) + jnp.mean(res_v**2) + jnp.mean(res_div**2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred, v_pred, _ = forward_uvp(params, x_test, y_test)
            l2_err = compute_ns_l2(u_pred, v_pred, u_exact_test, v_exact_test)
            history["total_loss"].append(float(loss_val))
            history["pde_loss"].append(float(pde_val))
            history["bc_loss"].append(float(bc_val))
            history["l2_error"].append(l2_err)
            history["eval_epochs"].append(epoch)
            if l2_err < best_l2:
                best_l2 = l2_err
            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss {float(loss_val):.4e} | "
                      f"PDE {float(pde_val):.4e} | BC {float(bc_val):.4e} | "
                      f"L2 {l2_err:.4e}")

    total_time = time.time() - t0
    u_pred_final, v_pred_final, p_pred_final = forward_uvp(params, x_test, y_test)
    u_pred_final = np.array(u_pred_final)
    v_pred_final = np.array(v_pred_final)
    p_pred_final = np.array(p_pred_final)
    final_l2 = compute_ns_l2(jnp.array(u_pred_final), jnp.array(v_pred_final),
                             u_exact_test, v_exact_test)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("FourierPINN", params, history, u_pred_final, v_pred_final, p_pred_final,
            np.array(u_exact_test), np.array(v_exact_test),
            np.array(x_test), np.array(y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 5: Vanilla PINN
# ################################################################
def run_pinn():
    print("\n" + "=" * 70)
    print("  METHOD 5 / 5 :  Vanilla PINN")
    print("=" * 70)

    LAYERS = [2] + [E11_OVR.get('hidden', 128)] * E11_OVR.get('n_hidden', 4) + [3]
    key = random.PRNGKey(SEED)

    def init_mlp(key, layers):
        params = []
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i + 1]
            key, k = random.split(key)
            W = random.normal(k, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros(fan_out)
            params.append({"W": W, "b": b})
        return params

    params = init_mlp(key, LAYERS)

    def forward(params, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for layer in params[:-1]:
            h = jnp.tanh(h @ layer["W"] + layer["b"])
        last = params[-1]
        return h @ last["W"] + last["b"]

    def forward_uvp(params, x, y):
        out = forward(params, x, y)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

    def ns_residual_single(params, x_s, y_s):
        def u_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 0]
        def v_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 1]
        def p_fn(xv, yv):
            out = forward(params, xv[None, None], yv[None, None])
            return out[0, 2]

        u_val = u_fn(x_s, y_s)
        v_val = v_fn(x_s, y_s)

        u_x = jax.grad(u_fn, 0)(x_s, y_s)
        u_y = jax.grad(u_fn, 1)(x_s, y_s)
        u_xx = jax.grad(jax.grad(u_fn, 0), 0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, 1), 1)(x_s, y_s)

        v_x = jax.grad(v_fn, 0)(x_s, y_s)
        v_y = jax.grad(v_fn, 1)(x_s, y_s)
        v_xx = jax.grad(jax.grad(v_fn, 0), 0)(x_s, y_s)
        v_yy = jax.grad(jax.grad(v_fn, 1), 1)(x_s, y_s)

        p_x = jax.grad(p_fn, 0)(x_s, y_s)
        p_y = jax.grad(p_fn, 1)(x_s, y_s)

        sx = source_x(x_s, y_s)
        sy = source_y(x_s, y_s)

        res_u = u_val * u_x + v_val * u_y + p_x / RHO - NU * (u_xx + u_yy) - sx
        res_v = u_val * v_x + v_val * v_y + p_y / RHO - NU * (v_xx + v_yy) - sy
        res_div = u_x + v_y
        return res_u, res_v, res_div

    ns_residual_batch = jax.vmap(ns_residual_single, in_axes=(None, 0, 0))

    # --- Data ---
    key = random.PRNGKey(SEED + 500)
    x_bc, y_bc, u_bc, v_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    key, subkey = random.split(key)
    x_test, y_test, u_exact_test, v_exact_test = generate_test_data(subkey)

    x_pde_flat = x_pde.ravel()
    y_pde_flat = y_pde.ravel()

    def loss_fn(params):
        u_pred_bc, v_pred_bc, _ = forward_uvp(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc)**2) + jnp.mean((v_pred_bc - v_bc)**2)

        res_u, res_v, res_div = ns_residual_batch(params, x_pde_flat, y_pde_flat)
        loss_pde = jnp.mean(res_u**2) + jnp.mean(res_v**2) + jnp.mean(res_div**2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred, v_pred, _ = forward_uvp(params, x_test, y_test)
            l2_err = compute_ns_l2(u_pred, v_pred, u_exact_test, v_exact_test)
            history["total_loss"].append(float(loss_val))
            history["pde_loss"].append(float(pde_val))
            history["bc_loss"].append(float(bc_val))
            history["l2_error"].append(l2_err)
            history["eval_epochs"].append(epoch)
            if l2_err < best_l2:
                best_l2 = l2_err
            if epoch % (EVAL_EVERY * 10) == 0 or epoch == EVAL_EVERY:
                print(f"  Epoch {epoch:5d} | Loss {float(loss_val):.4e} | "
                      f"PDE {float(pde_val):.4e} | BC {float(bc_val):.4e} | "
                      f"L2 {l2_err:.4e}")

    total_time = time.time() - t0
    u_pred_final, v_pred_final, p_pred_final = forward_uvp(params, x_test, y_test)
    u_pred_final = np.array(u_pred_final)
    v_pred_final = np.array(v_pred_final)
    p_pred_final = np.array(p_pred_final)
    final_l2 = compute_ns_l2(jnp.array(u_pred_final), jnp.array(v_pred_final),
                             u_exact_test, v_exact_test)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("PINN", params, history, u_pred_final, v_pred_final, p_pred_final,
            np.array(u_exact_test), np.array(v_exact_test),
            np.array(x_test), np.array(y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  Saving Utilities
# ################################################################
def save_method_results(result):
    (name, params, history, u_pred, v_pred, p_pred,
     u_exact, v_exact, x_test, y_test,
     n_params, total_time, best_l2, final_l2) = result

    import pickle
    with open(os.path.join(SAVE_DIR, f"{name}_params.pkl"), 'wb') as f:
        pickle.dump(params, f)

    np.savez(os.path.join(SAVE_DIR, f"{name}_history.npz"),
             total_loss=np.array(history["total_loss"]),
             pde_loss=np.array(history["pde_loss"]),
             bc_loss=np.array(history["bc_loss"]),
             l2_error=np.array(history["l2_error"]),
             eval_epochs=np.array(history["eval_epochs"]))

    np.savez(os.path.join(SAVE_DIR, f"{name}_prediction.npz"),
             u_pred=u_pred, v_pred=v_pred, p_pred=p_pred,
             u_exact=u_exact, v_exact=v_exact,
             x_test=x_test, y_test=y_test)

    summary = {
        "method": name,
        "total_params": int(n_params),
        "total_time_sec": round(total_time, 2),
        "best_l2_error": float(best_l2),
        "final_l2_error": float(final_l2),
        "ms_per_epoch": round(total_time / EPOCHS * 1000, 2),
    }
    with open(os.path.join(SAVE_DIR, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved {name} -> {SAVE_DIR}/")


def save_comparison_table(results):
    csv_path = os.path.join(SAVE_DIR, "comparison_table.csv")
    fields = ["method", "total_params", "total_time_sec",
              "best_l2_error", "final_l2_error", "ms_per_epoch"]
    rows = []
    for r in results:
        name = r[0]
        n_params = r[10]
        total_time = r[11]
        best_l2 = r[12]
        final_l2 = r[13]
        rows.append({
            "method": name,
            "total_params": int(n_params),
            "total_time_sec": round(total_time, 2),
            "best_l2_error": float(best_l2),
            "final_l2_error": float(final_l2),
            "ms_per_epoch": round(total_time / EPOCHS * 1000, 2),
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Comparison table -> {csv_path}")


# ################################################################
#  MAIN
# ################################################################
def main():
    print("=" * 70)
    print("  Ablation Study – Case 9: Steady Navier-Stokes, Double Cylinder")
    print("  Domain: [-pi,pi]^2 minus two cylinders")
    print(f"  Device: {jax.devices()}")
    print(f"  Epochs: {EPOCHS} | LR: {LR} | N_PDE: ~{N_PDE_LHS} | "
          f"N_BC: {4*N_BC_EXTERIOR + 2*N_BC_CYLINDER}")
    print("=" * 70)

    runners = [
        ("SVSNN_accel", run_svsnn_accelerated),
        ("SVSNN",       run_svsnn),
        ("SPINN",       run_spinn),
        ("SIREN",       run_siren),
        ("FourierPINN", run_fourierpinn),
        ("PINN",        run_pinn),
    ]

    results = []
    for tag, runner_fn in runners:
        summary_path = os.path.join(SAVE_DIR, f"{tag}_summary.json")
        if os.path.exists(summary_path):
            print(f"\n  [SKIP] {tag} already completed.")
            continue
        result = runner_fn()
        save_method_results(result)
        results.append(result)

    if results:
        save_comparison_table(results)

    # --- Final summary table ---
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    header = (f"{'Method':<14} {'Params':>8} {'Time(s)':>9} "
              f"{'Best L2':>12} {'Final L2':>12} {'ms/epoch':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        name = r[0]
        n_params = r[10]
        total_time = r[11]
        best_l2 = r[12]
        final_l2 = r[13]
        ms_ep = total_time / EPOCHS * 1000
        print(f"{name:<14} {n_params:>8d} {total_time:>9.1f} {best_l2:>12.4e} "
              f"{final_l2:>12.4e} {ms_ep:>10.2f}")
    print("=" * 70)
    print("All results saved to:", SAVE_DIR)


CASE_INFO = {"id": "case9", "title": "Steady double-cylinder 2D NS",
             "family": "ns", "has_classical": False}

_ARCH9 = dict(in_dim=2, out_dim=3, n_coord=2, spinn_n_branch=2, per_out_weight=True)
_RUNNERS9 = {"SVSNN": "run_svsnn_accelerated", "SPINN": "run_spinn",
             "SIREN": "run_siren", "FourierPINN": "run_fourierpinn", "PINN": "run_pinn"}


def E11_run(method, budget, seed, epochs=None, target=None, save_pred_path=None):
    import numpy as _np
    import _e11common
    g = globals()
    g["SEED"] = seed
    if epochs is not None:
        g["EPOCHS"] = epochs
    g["E11_OVR"] = {}
    EP = g["EPOCHS"]
    tgt = target if (method != "SVSNN" and budget == "matched") else None
    if tgt is not None:
        _e11common.set_matched_ovr(sys.modules[__name__], method, target, seed, _ARCH9)

    out = g[_RUNNERS9[method]]()
    u_pred = out[3]
    n_params, total_time, best_l2, final_l2 = out[-4], out[-3], out[-2], out[-1]
    matched_within = (abs(int(n_params) - target) <= 0.10 * target) if tgt is not None else None
    rec = _e11common.harness.normalize_record(
        method, budget, seed, params=int(n_params), best_l2=float(best_l2),
        final_l2=float(final_l2), train_time_sec=float(total_time), n_epochs=EP,
        n_collocation=int(N_PDE_LHS), inference_ms=float("nan"),
        target_params=tgt, matched_within_tol=matched_within)
    if save_pred_path is not None:
        _np.savez(save_pred_path, u_pred=_np.asarray(u_pred))
    return rec


if __name__ == "__main__":
    main()
