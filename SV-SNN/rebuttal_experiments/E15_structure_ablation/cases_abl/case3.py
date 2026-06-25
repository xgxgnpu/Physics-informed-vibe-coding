"""
SV-SNN Acceleration – Case 3: 2D Nonlinear Elliptic Equation
==============================================================
PDE:  Laplacian(u) + u^2 = f(x,y)   on [0,1]^2
BC:   u = u_exact                      on boundary (non-homogeneous Dirichlet)
Exact: u(x,y) = (x+y) * cos(10x) * sin(10y)

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
W_CHAR = 10.0
FF_DIM = 64
SEED = 42
EPOCHS = 10000
LR = 1e-3
N_PDE = 10000
N_BC = 1024
N_TEST = 256
EVAL_EVERY = 100
NC_SPINN = 100

E11_OVR = {}  # E11 size overrides for matched budget (set by E11_run)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================================================
#  PDE Definition
# ================================================================
def exact_solution(x, y):
    return (x + y) * jnp.cos(10.0 * x) * jnp.sin(10.0 * y)


def source_term(x, y):
    return (-200.0 * (x + y) * jnp.cos(10.0 * x) * jnp.sin(10.0 * y)
            - 20.0 * jnp.sin(10.0 * x) * jnp.sin(10.0 * y)
            + 20.0 * jnp.cos(10.0 * x) * jnp.cos(10.0 * y)
            + (x + y)**2 * jnp.cos(10.0 * x)**2 * jnp.sin(10.0 * y)**2)


# ================================================================
#  Data Generation
# ================================================================
def generate_bc_data(key):
    """256 points per side, non-homogeneous Dirichlet BC."""
    n_per_side = N_BC // 4
    t = jnp.linspace(0.0, 1.0, n_per_side).reshape(-1, 1)

    x_bottom = t;           y_bottom = jnp.zeros_like(t)
    x_top    = t;           y_top    = jnp.ones_like(t)
    x_left   = jnp.zeros_like(t); y_left = t
    x_right  = jnp.ones_like(t);  y_right = t

    x_bc = jnp.concatenate([x_bottom, x_top, x_left, x_right], axis=0)
    y_bc = jnp.concatenate([y_bottom, y_top, y_left, y_right], axis=0)
    u_bc = exact_solution(x_bc, y_bc)
    return x_bc, y_bc, u_bc


def generate_pde_data(key):
    """LHS interior collocation points."""
    pts = jnp.array(lhs(2, samples=N_PDE))  # plain LHS (maximin too slow for repeated runs)
    x_pde = pts[:, 0:1]
    y_pde = pts[:, 1:2]
    return x_pde, y_pde


def generate_test_data():
    """256x256 uniform test grid."""
    x1d = jnp.linspace(0.0, 1.0, N_TEST)
    y1d = jnp.linspace(0.0, 1.0, N_TEST)
    X, Y = jnp.meshgrid(x1d, y1d, indexing="ij")
    U_exact = exact_solution(X, Y)
    return X, Y, U_exact


# ================================================================
#  Utility: hvp_fwdfwd (second-order derivative via forward-mode)
# ================================================================
def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: jvp(f, (primals,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    return tangents_out


# ================================================================
#  Parameter counting
# ================================================================
def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params))


# ================================================================
#  L2 relative error on test grid
# ================================================================
def compute_l2_error(u_pred, u_exact):
    return float(jnp.sqrt(jnp.sum((u_pred - u_exact)**2) /
                           jnp.sum(u_exact**2)))


def _sample_frequencies(rng_key, K, w_char):
    freqs = jnp.abs(jax.random.normal(rng_key, (K,)) * 0.1 + w_char)
    return jnp.sort(freqs)


# ################################################################
#  METHOD 1: SV-SNN
# ################################################################
def run_svsnn():
    print("\n" + "=" * 70)
    print("  METHOD 1 / 5 :  SV-SNN")
    print("=" * 70)

    NUM_MODES = 6
    NUM_FREQ = 32
    key = random.PRNGKey(SEED)

    # --- Tight Gaussian frequency sampling: |N(w_char, 0.1)| per branch ---
    def _sample_frequencies(rng_key, K, w_char):
        freqs = jnp.abs(jax.random.normal(rng_key, (K,)) * 0.1 + w_char)
        return jnp.sort(freqs)

    # --- Init params: per-mode, per-direction independent freq sets ---
    def init_params(rng_key):
        keys = jax.random.split(rng_key, NUM_MODES * 6 + 2)
        ki = 0
        sx, sy = [], []
        for _ in range(NUM_MODES):
            for s_list in [sx, sy]:
                s_list.append({
                    'freqs': _sample_frequencies(keys[ki], NUM_FREQ, W_CHAR),
                    'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                    'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                    'bias': jnp.zeros(1),
                })
                ki += 3
        return {'spatial_x': sx, 'spatial_y': sy,
                'mode_coeffs': jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1,
                'global_bias': jnp.zeros(1)}

    params = init_params(key)

    # --- Spatial branch forward (freqs frozen via stop_gradient) ---
    def spatial_forward(sp, x):
        freqs = jax.lax.stop_gradient(sp['freqs'])
        wx = freqs[None, :] * x
        return (jnp.sum(sp['cos_c'] * jnp.cos(wx)
                        + sp['sin_c'] * jnp.sin(wx),
                        axis=1, keepdims=True)
                + sp['bias'])

    # --- Network forward ---
    def forward(params, x, y):
        u = jnp.zeros_like(x)
        for n in range(NUM_MODES):
            u += (params['mode_coeffs'][n]
                  * spatial_forward(params['spatial_x'][n], x)
                  * spatial_forward(params['spatial_y'][n], y))
        u += params['global_bias']
        return u

    # --- PDE residual via vmap(jax.grad(jax.grad(...))) per point ---
    def pde_residual_single(params, x_s, y_s):
        def u_fn(x_, y_):
            return forward(params, x_[None, None], y_[None, None]).squeeze()
        u_val = u_fn(x_s, y_s)
        u_xx = jax.grad(jax.grad(u_fn, 0), 0)(x_s, y_s)
        u_yy = jax.grad(jax.grad(u_fn, 1), 1)(x_s, y_s)
        f_val = source_term(x_s, y_s)
        return u_xx + u_yy + u_val**2 - f_val

    pde_residual_batch = jax.vmap(pde_residual_single, in_axes=(None, 0, 0))

    # --- Data ---
    key = random.PRNGKey(SEED + 100)
    x_bc, y_bc, u_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    X_test, Y_test, U_exact = generate_test_data()

    x_pde_flat = x_pde.ravel()
    y_pde_flat = y_pde.ravel()

    # --- Loss ---
    def loss_fn(params):
        u_pred_bc = forward(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc) ** 2)

        residuals = pde_residual_batch(params, x_pde_flat, y_pde_flat)
        loss_pde = jnp.mean(residuals ** 2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, pde_l, bc_l

    # --- Training loop ---
    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred_test = forward(params,
                                  X_test.reshape(-1, 1),
                                  Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
            l2_err = compute_l2_error(u_pred_test, U_exact)
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
    u_pred_final = forward(params,
                           X_test.reshape(-1, 1),
                           Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
    final_l2 = compute_l2_error(u_pred_final, U_exact)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SVSNN_orig", params, history, np.array(u_pred_final),
            np.array(U_exact), np.array(X_test), np.array(Y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 1b: SV-SNN ACCELERATED
# ################################################################
def run_svsnn_accelerated():
    print("\n" + "=" * 70)
    print("  SV-SNN (ACCELERATED)")
    print("  Analytic X_n''/Y_n'', vectorized modes, separable grid, zero AD")
    print("=" * 70)

    NUM_MODES = 6
    NUM_FREQ = 32
    NC = NC_SPINN
    key = random.PRNGKey(SEED)

    def _sample_freqs(rng_key, K, w_char):
        import _abl, abl_freqs
        if _abl.STRATEGY != 'default':
            return abl_freqs.strategy_sample(rng_key, K, w_char, w_char, _abl.STRATEGY) * _abl.SCALE
        freqs = jnp.abs(jax.random.normal(rng_key, (K,)) * 0.1 + w_char)
        return jnp.sort(freqs) * _abl.SCALE

    def init_params(rng_key):
        keys = jax.random.split(rng_key, NUM_MODES * 6 + 2)
        ki = 0
        sx, sy = [], []
        for _ in range(NUM_MODES):
            for s_list in [sx, sy]:
                s_list.append({
                    'freqs': _sample_freqs(keys[ki], NUM_FREQ, W_CHAR),
                    'cos_c': jax.random.normal(keys[ki+1], (NUM_FREQ,)) * 0.1,
                    'sin_c': jax.random.normal(keys[ki+2], (NUM_FREQ,)) * 0.1,
                    'bias': jnp.zeros(1),
                })
                ki += 3
        return {'spatial_x': sx, 'spatial_y': sy,
                'mode_coeffs': jax.random.normal(keys[ki], (NUM_MODES,)) * 0.1,
                'global_bias': jnp.zeros(1)}

    params = init_params(key)
    n_params = count_params(params)
    print(f"  Parameters: {n_params}")

    def _stack_spatial(params, axis_key):
        all_freqs = jnp.stack([jax.lax.stop_gradient(params[axis_key][n]['freqs'])
                               for n in range(NUM_MODES)])
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

    xc = jnp.linspace(0, 1, NC).reshape(-1, 1)
    yc = jnp.linspace(0, 1, NC).reshape(-1, 1)
    X_mesh, Y_mesh = jnp.meshgrid(xc.squeeze(), yc.squeeze(), indexing='ij')
    f_grid = source_term(X_mesh, Y_mesh)

    key_data = random.PRNGKey(SEED + 100)
    x_bc, y_bc, u_bc = generate_bc_data(key_data)
    X_test, Y_test, U_exact = generate_test_data()

    n_per_side = NC
    xb_left = jnp.zeros((n_per_side, 1));  yb_l = jnp.linspace(0, 1, n_per_side).reshape(-1, 1)
    xb_right = jnp.ones((n_per_side, 1));  yb_r = jnp.linspace(0, 1, n_per_side).reshape(-1, 1)
    xb_bot = jnp.linspace(0, 1, n_per_side).reshape(-1, 1); yb_bot = jnp.zeros((n_per_side, 1))
    xb_top = jnp.linspace(0, 1, n_per_side).reshape(-1, 1); yb_top = jnp.ones((n_per_side, 1))

    u_bc_left = exact_solution(xb_left, yb_l)
    u_bc_right = exact_solution(xb_right, yb_r)
    u_bc_bot = exact_solution(xb_bot, yb_bot)
    u_bc_top = exact_solution(xb_top, yb_top)

    def vectorized_forward(params, x, y):
        freqs_x, cos_x, sin_x, bias_x = _stack_spatial(params, 'spatial_x')
        freqs_y, cos_y, sin_y, bias_y = _stack_spatial(params, 'spatial_y')
        coeffs = params['mode_coeffs']
        X_all, _ = _compute_basis(x.squeeze(), freqs_x, cos_x, sin_x, bias_x)
        Y_all, _ = _compute_basis(y.squeeze(), freqs_y, cos_y, sin_y, bias_y)
        u = jnp.sum(coeffs[None, :] * X_all * Y_all, axis=-1, keepdims=True)
        return u + params['global_bias']

    def vectorized_forward_grid(params, x, y):
        freqs_x, cos_x, sin_x, bias_x = _stack_spatial(params, 'spatial_x')
        freqs_y, cos_y, sin_y, bias_y = _stack_spatial(params, 'spatial_y')
        coeffs = params['mode_coeffs']
        X_all, _ = _compute_basis(x.squeeze(), freqs_x, cos_x, sin_x, bias_x)
        Y_all, _ = _compute_basis(y.squeeze(), freqs_y, cos_y, sin_y, bias_y)
        cX = coeffs[None, :] * X_all
        return jnp.einsum('nm,jm->nj', cX, Y_all) + params['global_bias']

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

        u_val = jnp.einsum('nm,jm->nj', cX, Y_all) + params['global_bias']

        return (u_xx + u_yy) + u_val**2 - f_grid

    def loss_fn(params):
        residual = vectorized_pde_residual(params)
        pde_loss = jnp.mean(residual**2)

        bc_loss = jnp.float32(0.0)
        bc_loss += jnp.mean((vectorized_forward(params, xb_left, yb_l) - u_bc_left)**2)
        bc_loss += jnp.mean((vectorized_forward(params, xb_right, yb_r) - u_bc_right)**2)
        bc_loss += jnp.mean((vectorized_forward(params, xb_bot, yb_bot) - u_bc_bot)**2)
        bc_loss += jnp.mean((vectorized_forward(params, xb_top, yb_top) - u_bc_top)**2)

        return pde_loss + bc_loss, (pde_loss, bc_loss)

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
            u_pred_grid = vectorized_forward_grid(params,
                jnp.linspace(0, 1, N_TEST).reshape(-1, 1),
                jnp.linspace(0, 1, N_TEST).reshape(-1, 1))
            l2_err = compute_l2_error(u_pred_grid, U_exact)
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
    u_pred_final = np.array(vectorized_forward_grid(params,
        jnp.linspace(0, 1, N_TEST).reshape(-1, 1),
        jnp.linspace(0, 1, N_TEST).reshape(-1, 1)))
    final_l2 = compute_l2_error(jnp.array(u_pred_final), U_exact)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SVSNN_accel", params, history, u_pred_final,
            np.array(U_exact), np.array(X_test), np.array(Y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 2: SPINN
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

    def init_modified_mlp(key, in_dim, features, n_layers, r):
        params = []
        key, k = random.split(key)
        fan_in = in_dim
        for i in range(n_layers):
            fan_out = features
            key, k1, k2 = random.split(key, 3)
            W = random.normal(k1, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
            b = jnp.zeros(fan_out)
            params.append({"W": W, "b": b})
            fan_in = fan_out
        key, k1, k2 = random.split(key, 3)
        W_out = random.normal(k1, (features, r)) * jnp.sqrt(2.0 / features)
        b_out = jnp.zeros(r)
        params.append({"W": W_out, "b": b_out})
        return params

    key, k1, k2, k3, k4 = random.split(key, 5)
    params_x = init_modified_mlp(k1, ff_input_dim, FEATURES, N_LAYERS, R)
    params_y = init_modified_mlp(k2, ff_input_dim, FEATURES, N_LAYERS, R)
    params = {
        "branch_x": params_x,
        "branch_y": params_y,
        "W_x": _sample_frequencies(k3, FF_DIM, W_CHAR).reshape(1, -1),
        "W_y": _sample_frequencies(k4, FF_DIM, W_CHAR).reshape(1, -1),
    }

    def fourier_embed(coord, W):
        return jnp.concatenate([jnp.sin(coord @ W), jnp.cos(coord @ W)], axis=-1)

    def mlp_forward(mlp_params, x):
        h = x
        for layer in mlp_params[:-1]:
            h = h @ layer["W"] + layer["b"]
            h = jnp.tanh(h)
        out_layer = mlp_params[-1]
        h = h @ out_layer["W"] + out_layer["b"]
        return h

    def forward(params, xc, yc):
        """(Nx,1) x (Ny,1) -> (Nx, Ny) grid prediction."""
        x_emb = fourier_embed(xc, params["W_x"])
        y_emb = fourier_embed(yc, params["W_y"])
        Vx = mlp_forward(params["branch_x"], x_emb)
        Vy = mlp_forward(params["branch_y"], y_emb)
        return Vx @ Vy.T

    def forward_for_dx(params, xc, yc):
        """Wrapper that flattens output for hvp along x."""
        return forward(params, xc, yc)

    def forward_for_dy(params, xc, yc):
        """Wrapper that flattens output for hvp along y."""
        return forward(params, xc, yc)

    # --- Data ---
    xc = jnp.linspace(0.0, 1.0, NC_SPINN).reshape(-1, 1)
    yc = jnp.linspace(0.0, 1.0, NC_SPINN).reshape(-1, 1)
    Xg, Yg = jnp.meshgrid(xc.ravel(), yc.ravel(), indexing="ij")
    source_grid = source_term(Xg, Yg)  # (NC, NC)

    # BC exact values
    xb_left = jnp.zeros((1, 1));   xb_right = jnp.ones((1, 1))
    yb_bottom = jnp.zeros((1, 1)); yb_top = jnp.ones((1, 1))
    u_bc_left   = exact_solution(xb_left,  yc)       # (NC,1) -> use xc as y
    u_bc_right  = exact_solution(xb_right, yc)
    u_bc_bottom = exact_solution(xc, yb_bottom)
    u_bc_top    = exact_solution(xc, yb_top)

    X_test, Y_test, U_exact = generate_test_data()

    # --- Loss ---
    def loss_fn(params):
        u = forward(params, xc, yc)  # (NC, NC)

        tx = jnp.ones_like(xc)
        ty = jnp.ones_like(yc)
        u_xx = hvp_fwdfwd(lambda xc: forward(params, xc, yc), (xc,), (tx,))
        u_yy = hvp_fwdfwd(lambda yc: forward(params, xc, yc), (yc,), (ty,))

        residual = u_xx + u_yy + u**2 - source_grid
        loss_pde = jnp.mean(residual ** 2)

        # BC losses
        u_l = forward(params, xb_left, yc)     # (1, NC)
        u_r = forward(params, xb_right, yc)    # (1, NC)
        u_b = forward(params, xc, yb_bottom)   # (NC, 1)
        u_t = forward(params, xc, yb_top)      # (NC, 1)

        loss_bc = (jnp.mean((u_l - u_bc_left.T)**2) +
                   jnp.mean((u_r - u_bc_right.T)**2) +
                   jnp.mean((u_b - u_bc_bottom)**2) +
                   jnp.mean((u_t - u_bc_top)**2))

        return loss_pde + loss_bc, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    # --- Training ---
    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            x_test_flat = X_test.reshape(-1, 1)
            y_test_flat = Y_test.reshape(-1, 1)
            # Evaluate in chunks for memory
            x_1d = jnp.linspace(0.0, 1.0, N_TEST).reshape(-1, 1)
            y_1d = jnp.linspace(0.0, 1.0, N_TEST).reshape(-1, 1)
            u_pred_test = forward(params, x_1d, y_1d)  # (N_TEST, N_TEST)
            l2_err = compute_l2_error(u_pred_test, U_exact)
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
    x_1d = jnp.linspace(0.0, 1.0, N_TEST).reshape(-1, 1)
    y_1d = jnp.linspace(0.0, 1.0, N_TEST).reshape(-1, 1)
    u_pred_final = np.array(forward(params, x_1d, y_1d))
    final_l2 = compute_l2_error(jnp.array(u_pred_final), U_exact)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SPINN", params, history, u_pred_final,
            np.array(U_exact), np.array(X_test), np.array(Y_test),
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
    siren_W_x = _sample_frequencies(k_wx, _ff, W_CHAR).reshape(1, -1)
    siren_W_y = _sample_frequencies(k_wy, _ff, W_CHAR).reshape(1, -1)
    ff_input_dim_siren = 4 * _ff
    LAYERS = [ff_input_dim_siren] + [_hid] * _nh + [1]

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
        h = h @ last["W"] + last["b"]
        return h

    # --- Data ---
    key = random.PRNGKey(SEED + 200)
    x_bc, y_bc, u_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    f_pde = source_term(x_pde, y_pde)
    X_test, Y_test, U_exact = generate_test_data()

    # --- Loss ---
    def loss_fn(params):
        u_pred_bc = forward(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc) ** 2)

        u = forward(params, x_pde, y_pde)
        u_xx = hvp_fwdfwd(lambda x: forward(params, x, y_pde),
                          (x_pde,), (jnp.ones_like(x_pde),))
        u_yy = hvp_fwdfwd(lambda y: forward(params, x_pde, y),
                          (y_pde,), (jnp.ones_like(y_pde),))
        residual = u_xx + u_yy + u**2 - f_pde
        loss_pde = jnp.mean(residual ** 2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    # --- Training ---
    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred_test = forward(params,
                                  X_test.reshape(-1, 1),
                                  Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
            l2_err = compute_l2_error(u_pred_test, U_exact)
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
    u_pred_final = forward(params,
                           X_test.reshape(-1, 1),
                           Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
    final_l2 = compute_l2_error(u_pred_final, U_exact)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("SIREN", params, history, np.array(u_pred_final),
            np.array(U_exact), np.array(X_test), np.array(Y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 4: FourierPINN
# ################################################################
def run_fourierpinn():
    print("\n" + "=" * 70)
    print("  METHOD 4 / 5 :  FourierPINN")
    print("=" * 70)

    FF_DIM = E11_OVR.get('ff', 64)
    MLP_LAYERS = [E11_OVR.get('hidden', 128)] * E11_OVR.get('n_hidden', 3) + [1]
    key = random.PRNGKey(SEED)

    # --- Fourier feature init ---
    key, k1, k2 = random.split(key, 3)
    W_x = _sample_frequencies(k1, FF_DIM, W_CHAR).reshape(1, -1)
    W_y = _sample_frequencies(k2, FF_DIM, W_CHAR).reshape(1, -1)

    # --- MLP init ---
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

    # --- Data ---
    key = random.PRNGKey(SEED + 300)
    x_bc, y_bc, u_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    f_pde = source_term(x_pde, y_pde)
    X_test, Y_test, U_exact = generate_test_data()

    # --- Loss ---
    def loss_fn(params):
        u_pred_bc = forward(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc) ** 2)

        u = forward(params, x_pde, y_pde)
        u_xx = hvp_fwdfwd(lambda x: forward(params, x, y_pde),
                          (x_pde,), (jnp.ones_like(x_pde),))
        u_yy = hvp_fwdfwd(lambda y: forward(params, x_pde, y),
                          (y_pde,), (jnp.ones_like(y_pde),))
        residual = u_xx + u_yy + u**2 - f_pde
        loss_pde = jnp.mean(residual ** 2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    # --- Training ---
    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred_test = forward(params,
                                  X_test.reshape(-1, 1),
                                  Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
            l2_err = compute_l2_error(u_pred_test, U_exact)
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
    u_pred_final = forward(params,
                           X_test.reshape(-1, 1),
                           Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
    final_l2 = compute_l2_error(u_pred_final, U_exact)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("FourierPINN", params, history, np.array(u_pred_final),
            np.array(U_exact), np.array(X_test), np.array(Y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  METHOD 5: Vanilla PINN
# ################################################################
def run_pinn():
    print("\n" + "=" * 70)
    print("  METHOD 5 / 5 :  Vanilla PINN")
    print("=" * 70)

    LAYERS = [2] + [E11_OVR.get('hidden', 128)] * E11_OVR.get('n_hidden', 4) + [1]
    key = random.PRNGKey(SEED)

    # --- Xavier init ---
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

    # --- Data ---
    key = random.PRNGKey(SEED + 400)
    x_bc, y_bc, u_bc = generate_bc_data(key)
    key, subkey = random.split(key)
    x_pde, y_pde = generate_pde_data(subkey)
    f_pde = source_term(x_pde, y_pde)
    X_test, Y_test, U_exact = generate_test_data()

    # --- Loss ---
    def loss_fn(params):
        u_pred_bc = forward(params, x_bc, y_bc)
        loss_bc = jnp.mean((u_pred_bc - u_bc) ** 2)

        u = forward(params, x_pde, y_pde)
        u_xx = hvp_fwdfwd(lambda x: forward(params, x, y_pde),
                          (x_pde,), (jnp.ones_like(x_pde),))
        u_yy = hvp_fwdfwd(lambda y: forward(params, x_pde, y),
                          (y_pde,), (jnp.ones_like(y_pde),))
        residual = u_xx + u_yy + u**2 - f_pde
        loss_pde = jnp.mean(residual ** 2)

        return loss_bc + loss_pde, (loss_pde, loss_bc)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jit
    def train_step(params, opt_state):
        (loss, (pde_l, bc_l)), grads = value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss, pde_l, bc_l

    # --- Training ---
    history = {"total_loss": [], "pde_loss": [], "bc_loss": [],
               "l2_error": [], "eval_epochs": []}
    best_l2 = float("inf")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        params, opt_state, loss_val, pde_val, bc_val = train_step(params, opt_state)

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            u_pred_test = forward(params,
                                  X_test.reshape(-1, 1),
                                  Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
            l2_err = compute_l2_error(u_pred_test, U_exact)
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
    u_pred_final = forward(params,
                           X_test.reshape(-1, 1),
                           Y_test.reshape(-1, 1)).reshape(N_TEST, N_TEST)
    final_l2 = compute_l2_error(u_pred_final, U_exact)
    n_params = count_params(params)

    print(f"  Done: {total_time:.1f}s | Params: {n_params} | "
          f"Best L2: {best_l2:.4e} | Final L2: {final_l2:.4e}")

    return ("PINN", params, history, np.array(u_pred_final),
            np.array(U_exact), np.array(X_test), np.array(Y_test),
            n_params, total_time, best_l2, final_l2)


# ################################################################
#  Saving Utilities
# ################################################################
def save_method_results(result):
    name, params, history, u_pred, u_exact, X, Y, n_params, \
        total_time, best_l2, final_l2 = result

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
             u_pred=u_pred, u_exact=u_exact, X=X, Y=Y)

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
        name, _, _, _, _, _, _, n_params, total_time, best_l2, final_l2 = r
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
    print("  Ablation Study – Case 3: 2D Nonlinear Elliptic Equation")
    print("  PDE: Lap(u) + u^2 = f(x,y),  u_exact = (x+y)*cos(10x)*sin(10y)")
    print(f"  Device: {jax.devices()}")
    print(f"  Epochs: {EPOCHS} | LR: {LR} | N_PDE: {N_PDE} | N_BC: {N_BC}")
    print("=" * 70)

    runners = [
        ("SVSNN_accel", run_svsnn_accelerated),
        ("SVSNN_orig",  run_svsnn),
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

    # --- Final summary table ---
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    header = f"{'Method':<14} {'Params':>8} {'Time(s)':>9} {'Best L2':>12} {'Final L2':>12} {'ms/epoch':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        name = r[0]
        n_params = r[7]
        total_time = r[8]
        best_l2 = r[9]
        final_l2 = r[10]
        ms_ep = total_time / EPOCHS * 1000
        print(f"{name:<14} {n_params:>8d} {total_time:>9.1f} {best_l2:>12.4e} "
              f"{final_l2:>12.4e} {ms_ep:>10.2f}")
    print("=" * 70)
    print("All results saved to:", SAVE_DIR)


CASE_INFO = {"id": "case3", "title": "2D nonlinear elliptic", "family": "elliptic_mono",
             "has_classical": False}

_ARCH3 = dict(in_dim=2, out_dim=1, n_coord=2, spinn_n_branch=2, per_out_weight=False)
_RUNNERS3 = {"SVSNN": "run_svsnn_accelerated", "SPINN": "run_spinn",
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
    matched_within = None
    tgt = None
    if method != "SVSNN" and budget == "matched":
        assert target is not None
        _, matched_within = _e11common.set_matched_ovr(
            sys.modules[__name__], method, target, seed, _ARCH3)
        tgt = target

    runner = g[_RUNNERS3[method]]
    out = runner()
    (_name, params, history, u_pred, U_exact, X, Y,
     n_params, total_time, best_l2, final_l2) = out

    n_coll = (NC_SPINN * NC_SPINN) if method in ("SVSNN", "SPINN") else int(N_PDE)
    rec = _e11common.harness.normalize_record(
        method, budget, seed, params=int(n_params), best_l2=float(best_l2),
        final_l2=float(final_l2), train_time_sec=float(total_time), n_epochs=EP,
        n_collocation=n_coll, inference_ms=float("nan"),
        target_params=tgt, matched_within_tol=matched_within)
    if save_pred_path is not None:
        _np.savez(save_pred_path, u_pred=_np.asarray(u_pred),
                  u_exact=_np.asarray(U_exact), X=_np.asarray(X), Y=_np.asarray(Y))
    return rec


if __name__ == "__main__":
    main()
