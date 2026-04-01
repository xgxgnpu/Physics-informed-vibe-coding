"""
Scale-PINN vs Standard PINN: Lid-Driven Cavity Re=7500 — JAX
=============================================================
Fair comparison of Scale-PINN (evolutionary regularization) against
standard PINN on the 2D steady-state incompressible Navier-Stokes:

  continuity:   u_x + v_y = 0
  momentum_x:   u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy) = 0
  momentum_y:   u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy) = 0

Domain: [0,1] x [0,1],  Re = 7500
BC: top lid u=1, v=0; other walls u=v=0; corners excluded

Two modes (fair comparison — identical network, optimizer, data, seed):
  M1 — Standard PINN  (ER=0, pure PDE residual + BC loss)
  M2 — Scale-PINN     (ER=0.095, ER_xx=0.5, evolutionary regularization)

Network: shared trunk [x,y,x-1,y-1] -> Dense(4*64) -> sin(2*pi*.) ->
         2x Dense+SiLU -> three branches (u,v,p) each 3x Dense+SiLU -> scalar
         Total params: 59520

Reference:
  Peng, W., Zhou, W., Zhang, J., & Yao, W. (2026).
  "Scale-PINN: Learning Efficient Physics-Informed Neural Networks
   with Evolutionary Regularization."

Self-contained single file.  Run:
    python ldc_re7500_scalepinn.py [--mode M1|M2|both] [--niter N] [--quick] [--plot_only]
"""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import sys
import argparse
import time
import pickle

import jax
import jax.numpy as jnp
from jax import random, vmap, jacfwd, jit
from jax import flatten_util
import flax.linen as nn
import optax
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
jax.config.update("jax_default_matmul_precision", "highest")

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
# Configuration
# ============================================================
RE = 7500.0
N_NODES = 64
BS_ALL = 1000
BS_BC = 50
SEED = 50
WEIGHT_BC = 10.0
MAX_LR = 5e-4
EXPONENT = 1.2
MAX_TIME = 300.0
LOG_EVERY = 500

ER_DEFAULT = 0.095
ER_XX_DEFAULT = 0.5

Re = RE
x_l = x_u = y_l = y_u = 0.0
x_ref = y_ref = 0.5


# ============================================================
# Plot style — journal quality
# ============================================================
def setup_plot_style():
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 16
    rcParams['axes.labelsize'] = 18
    rcParams['axes.titlesize'] = 18
    rcParams['axes.linewidth'] = 2.0
    rcParams['axes.labelweight'] = 'bold'
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
# Data I/O
# ============================================================
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
# Neural Network Definitions
# ============================================================
class PINN(nn.Module):
    n_nodes: int

    def setup(self):
        kinit = jax.nn.initializers.he_uniform()
        self.feats = nn.Dense(self.n_nodes * 4, kernel_init=kinit)
        self.layers = [
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
        ]
        self.splitu = nn.Dense(self.n_nodes, kernel_init=kinit)
        self.layeru = [
            nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(1, kernel_init=kinit, use_bias=False),
        ]
        self.splitv = nn.Dense(self.n_nodes, kernel_init=kinit)
        self.layerv = [
            nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(1, kernel_init=kinit, use_bias=False),
        ]
        self.splitp = nn.Dense(self.n_nodes, kernel_init=kinit)
        self.layerp = [
            nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(1, kernel_init=kinit, use_bias=False),
        ]

    @nn.compact
    def __call__(self, inputs):
        x, y = inputs[:, 0:1], inputs[:, 1:2]

        def get_uvp(x, y):
            inp = jnp.hstack([x, y, x - 1.0, y - 1.0])
            hidden = self.feats(inp)
            hidden = jnp.sin(2 * jnp.pi * hidden)
            for lyr in self.layers:
                hidden = lyr(hidden)
            u = self.splitu(hidden)
            for lyr in self.layeru:
                u = lyr(u)
            v = self.splitv(hidden)
            for lyr in self.layerv:
                v = lyr(v)
            p = self.splitp(hidden)
            for lyr in self.layerp:
                p = lyr(p)
            return (u, v, p)

        u, v, p = get_uvp(x, y)

        def get_uvp_xy(get_uvp, x, y):
            u_x, v_x, p_x = jacfwd(get_uvp)(x, y)
            u_y, v_y, p_y = jacfwd(get_uvp, argnums=1)(x, y)
            return u_x, u_y, v_x, v_y, p_x, p_y

        f_xy_vmap = vmap(get_uvp_xy, in_axes=(None, 0, 0))
        u_x, u_y, v_x, v_y, p_x, p_y = f_xy_vmap(get_uvp, x, y)
        u_x = u_x[:, :, 0]
        u_y = u_y[:, :, 0]
        v_x = v_x[:, :, 0]
        v_y = v_y[:, :, 0]
        p_x = p_x[:, :, 0]
        p_y = p_y[:, :, 0]

        def get_uvp_xxyy(get_uvp, x, y):
            u_xx, v_xx, p_xx = jacfwd(jacfwd(get_uvp))(x, y)
            u_yy, v_yy, p_yy = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(x, y)
            return u_xx, u_yy, v_xx, v_yy, p_xx, p_yy

        f_xxyy_vmap = vmap(get_uvp_xxyy, in_axes=(None, 0, 0))
        u_xx, u_yy, v_xx, v_yy, p_xx, p_yy = f_xxyy_vmap(get_uvp, x, y)
        u_xx = u_xx[:, :, 0, 0]
        u_yy = u_yy[:, :, 0, 0]
        v_xx = v_xx[:, :, 0, 0]
        v_yy = v_yy[:, :, 0, 0]

        bc = (x == x_l) | (x == x_u) | (y == y_l) | (y == y_u)
        nbc = (~bc)

        residuals_continuity = u_x + v_y
        residuals_momentum_1 = u * u_x + v * u_y + p_x - 1.0 / Re * (u_xx + u_yy)
        residuals_momentum_2 = u * v_x + v * v_y + p_y - 1.0 / Re * (v_xx + v_yy)

        mom1 = -1.0 / Re * (u_xx + u_yy)
        mom2 = -1.0 / Re * (v_xx + v_yy)

        outputs = jnp.hstack([
            u, v, p,
            residuals_continuity, residuals_momentum_1, residuals_momentum_2,
            bc, nbc, mom1, mom2,
        ])
        return outputs


class DNN(nn.Module):
    n_nodes: int

    def setup(self):
        kinit = jax.nn.initializers.he_uniform()
        self.feats = nn.Dense(self.n_nodes * 4, kernel_init=kinit)
        self.layers = [
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
        ]
        self.splitu = nn.Dense(self.n_nodes, kernel_init=kinit)
        self.layeru = [
            nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(1, kernel_init=kinit, use_bias=False),
        ]
        self.splitv = nn.Dense(self.n_nodes, kernel_init=kinit)
        self.layerv = [
            nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(1, kernel_init=kinit, use_bias=False),
        ]
        self.splitp = nn.Dense(self.n_nodes, kernel_init=kinit)
        self.layerp = [
            nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(self.n_nodes, kernel_init=kinit), nn.silu,
            nn.Dense(1, kernel_init=kinit, use_bias=False),
        ]

    @nn.compact
    def __call__(self, inputs):
        x, y = inputs[:, 0:1], inputs[:, 1:2]

        def get_uvp(x, y):
            inp = jnp.hstack([x, y, x - 1.0, y - 1.0])
            hidden = self.feats(inp)
            hidden = jnp.sin(2 * jnp.pi * hidden)
            for lyr in self.layers:
                hidden = lyr(hidden)
            u = self.splitu(hidden)
            for lyr in self.layeru:
                u = lyr(u)
            v = self.splitv(hidden)
            for lyr in self.layerv:
                v = lyr(v)
            p = self.splitp(hidden)
            for lyr in self.layerp:
                p = lyr(p)
            return (u, v, p)

        u, v, p = get_uvp(x, y)

        def get_uvp_xxyy(get_uvp, x, y):
            u_xx, v_xx, _ = jacfwd(jacfwd(get_uvp))(x, y)
            u_yy, v_yy, _ = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(x, y)
            return u_xx, u_yy, v_xx, v_yy

        f_xxyy_vmap = vmap(get_uvp_xxyy, in_axes=(None, 0, 0))
        u_xx, u_yy, v_xx, v_yy = f_xxyy_vmap(get_uvp, x, y)
        u_xx = u_xx[:, :, 0, 0]
        u_yy = u_yy[:, :, 0, 0]
        v_xx = v_xx[:, :, 0, 0]
        v_yy = v_yy[:, :, 0, 0]

        mom1 = -1.0 / Re * (u_xx + u_yy)
        mom2 = -1.0 / Re * (v_xx + v_yy)

        x_pref = jnp.ones_like(x) * x_ref
        y_pref = jnp.ones_like(y) * y_ref
        _, _, pref = get_uvp(x_pref, y_pref)
        pout = p - pref

        outputs = jnp.hstack([u, v, pout, mom1, mom2])
        return outputs


# ============================================================
# Training
# ============================================================
def train_model(mode, max_iters, seed=SEED):
    global Re, x_l, x_u, y_l, y_u, x_ref, y_ref

    assert mode in ('M1', 'M2')
    ER = 0.0 if mode == 'M1' else ER_DEFAULT
    ER_xx = ER_XX_DEFAULT
    weight_bc = WEIGHT_BC

    print("=" * 70)
    print(f"Training mode: {mode} ({'Standard PINN' if mode == 'M1' else 'Scale-PINN'})")
    print(f"  ER={ER}, ER_xx={ER_xx}, weight_bc={weight_bc}")
    print(f"  max_iters={max_iters}, max_lr={MAX_LR}, seed={seed}")
    print("=" * 70)

    Re = RE

    sim = pd.read_csv(os.path.join(DATA_DIR, 'LDC_RE7500_150x150_CELL_sub256x256.csv'))
    data_X = sim[['x', 'y']].values
    data_Y = sim[['u', 'v', 'p']].values

    x_l = float(np.min(data_X[:, 0]))
    x_u = float(np.max(data_X[:, 0]))
    y_l = float(np.min(data_X[:, 1]))
    y_u = float(np.max(data_X[:, 1]))

    corners = (
        ((sim.x == x_l) & (sim.y == y_u)) |
        ((sim.x == x_u) & (sim.y == y_u)) |
        ((sim.x == x_l) & (sim.y == y_l)) |
        ((sim.x == x_u) & (sim.y == y_l))
    )
    data_X = data_X[~corners]
    data_Y = data_Y[~corners]

    x_vals = np.unique(sim.x.values)
    y_vals = np.unique(sim.y.values)
    x_ref = float(x_vals[x_vals.size // 2])
    y_ref = float(y_vals[y_vals.size // 2])

    bc_mask = (
        (data_X[:, 0] == x_l) | (data_X[:, 0] == x_u) |
        (data_X[:, 1] == y_l) | (data_X[:, 1] == y_u)
    )
    data_X_BC = data_X[bc_mask]
    data_Y_BC = data_Y[bc_mask]

    data_X = jnp.array(data_X)
    data_Y = jnp.array(data_Y)
    data_X_BC = jnp.array(data_X_BC)
    data_Y_BC = jnp.array(data_Y_BC)

    key, rng = random.split(random.PRNGKey(seed))
    a = random.normal(key, [1, 2])

    model = PINN(N_NODES)
    model_0 = DNN(N_NODES)
    params_tree = model.init(key, a)
    params, unravel_fn = flatten_util.ravel_pytree(params_tree)
    num_params = params.shape[0]
    params_0 = params

    print(f"Number of parameters: {num_params}")

    n_all = len(data_X)
    n_bc = len(data_X_BC)

    @jit
    def minibatch(key):
        key1, key2 = key
        batch_all = random.choice(key1, n_all, (BS_ALL - BS_BC,))
        batch_bc = random.choice(key2, n_bc, (BS_BC,))
        batch_X = jnp.vstack([data_X[batch_all], data_X_BC[batch_bc]])
        batch_Y = jnp.vstack([data_Y[batch_all], data_Y_BC[batch_bc]])
        return batch_X, batch_Y

    def eval_loss(params, params_0, inputs, labels):
        pred = model.apply(unravel_fn(params), inputs)
        u, v, p, res_cont, res_mom1, res_mom2, bc, nbc, m_1, m_2 = \
            jnp.split(pred, 10, axis=1)
        gt_u, gt_v, gt_p = jnp.split(labels, 3, axis=1)

        pred0 = model_0.apply(unravel_fn(params_0), inputs)
        u_0, v_0, p_0, m0_1, m0_2 = jnp.split(pred0, 5, axis=1)

        if ER > 0:
            res_cont = res_cont + (p - p_0) / ER
            res_mom1 = res_mom1 + (u - u_0) / ER + (m_1 - m0_1) / ER_xx
            res_mom2 = res_mom2 + (v - v_0) / ER + (m_2 - m0_2) / ER_xx

        pde_uvp = (jnp.square(res_cont) +
                   jnp.square(res_mom1) +
                   jnp.square(res_mom2))
        pde_loss = jnp.sum(pde_uvp * nbc) / nbc.sum()

        bc_loss = (jnp.sum(jnp.square(u - gt_u) * bc) / bc.sum() +
                   jnp.sum(jnp.square(v - gt_v) * bc) / bc.sum())

        uv = jnp.hstack([u, v])
        gt_uv = jnp.hstack([gt_u, gt_v])
        mse = jnp.mean(jnp.square(uv - gt_uv))
        rl2 = jnp.linalg.norm(uv - gt_uv) / jnp.linalg.norm(gt_uv)

        loss = pde_loss + weight_bc * bc_loss
        return loss, (mse, rl2, pde_loss, bc_loss)

    loss_grad = jax.jit(jax.value_and_grad(eval_loss, has_aux=True))

    @jit
    def update(params, params_0, opt_state, key):
        batch_X, batch_Y = minibatch(key)
        (loss, (mse, rl2, pde_loss, bc_loss)), grad = \
            loss_grad(params, params_0, batch_X, batch_Y)
        updates, opt_state = optimizer.update(grad, opt_state)
        params_0 = params
        params = optax.apply_updates(params, updates)
        return params, params_0, opt_state, loss, mse, rl2, pde_loss, bc_loss

    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=MAX_LR, peak_value=MAX_LR, warmup_steps=0,
        decay_steps=max_iters, end_value=1e-10, exponent=EXPONENT)
    optimizer = optax.adam(learning_rate=lr_scheduler)
    opt_state = optimizer.init(params)

    runtime = 0.0
    train_iters = 0
    history = {
        'iter': [], 'time': [], 'total_loss': [], 'pde_loss': [],
        'bc_loss': [], 'mse': [], 'rl2': [],
    }
    best_rl2 = 1e10
    best_params = params

    while train_iters <= max_iters and runtime < MAX_TIME:
        start = time.time()
        key1, key2, rng = random.split(rng, 3)
        params, params_0, opt_state, loss, mse, rl2, pde_loss, bc_loss = \
            update(params, params_0, opt_state, (key1, key2))
        end = time.time()
        runtime += (end - start)

        if train_iters % LOG_EVERY == 0:
            loss_v = float(loss)
            mse_v = float(mse)
            rl2_v = float(rl2)
            pde_v = float(pde_loss)
            bc_v = float(bc_loss)

            print(f'  iter={train_iters:05d}  time={runtime:7.1f}s  '
                  f'loss={loss_v:.2e}  pde={pde_v:.2e}  bc={bc_v:.2e}  '
                  f'mse={mse_v:.2e}  rl2={rl2_v:.2e}')

            history['iter'].append(train_iters)
            history['time'].append(runtime)
            history['total_loss'].append(loss_v)
            history['pde_loss'].append(pde_v)
            history['bc_loss'].append(bc_v)
            history['mse'].append(mse_v)
            history['rl2'].append(rl2_v)

            if rl2_v < best_rl2:
                best_rl2 = rl2_v
                best_params = params

        train_iters += 1

    inputs = data_X
    labels = data_Y
    prediction = model_0.apply(unravel_fn(best_params), inputs)
    u_pred, v_pred, p_pred, _, _ = jnp.split(prediction, 5, axis=-1)

    gt_u, gt_v, gt_p = jnp.split(labels, 3, axis=-1)

    uv_final = jnp.hstack([u_pred, v_pred])
    gt_uv = jnp.hstack([gt_u, gt_v])
    mse_final = float(jnp.mean(jnp.square(uv_final - gt_uv)))
    rl2_final = float(jnp.linalg.norm(uv_final - gt_uv) / jnp.linalg.norm(gt_uv))

    print(f'\n  [{mode}] Final eval on all data:  MSE={mse_final:.2e}  RL2={rl2_final:.2e}')

    save_training_history(history, os.path.join(DATA_DIR, f'loss_history_{mode}.txt'))
    save_predictions(
        os.path.join(DATA_DIR, f'predictions_{mode}.npz'),
        data_X=np.array(data_X),
        gt_u=np.array(gt_u), gt_v=np.array(gt_v), gt_p=np.array(gt_p),
        pred_u=np.array(u_pred), pred_v=np.array(v_pred), pred_p=np.array(p_pred),
    )

    with open(os.path.join(CKPT_DIR, f'params_{mode}.pkl'), 'wb') as f:
        pickle.dump(np.array(best_params), f)

    total_time = runtime
    print(f"\n{'=' * 70}")
    print(f"  {mode} DONE | Params={num_params} | Best RL2={best_rl2:.4e} | "
          f"Final RL2={rl2_final:.4e} | Time={total_time:.1f}s")
    print(f"{'=' * 70}\n")

    return {
        'mode': mode,
        'num_params': num_params,
        'best_rl2': best_rl2,
        'final_rl2': rl2_final,
        'final_mse': mse_final,
        'total_time': total_time,
        'max_iters': max_iters,
        'history': history,
    }


# ============================================================
# Plotting functions — publication quality
# ============================================================
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['pde_loss'],
                linewidth=2.5, label='$\\mathcal{L}_{PDE}$ (Standard PINN)',
                color='#2196F3', linestyle='-')
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['pde_loss'],
                linewidth=2.5, label='$\\mathcal{L}_{PDE}$ (Scale-PINN)',
                color='#F44336', linestyle='-')
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['bc_loss'],
                linewidth=2.0, label='$\\mathcal{L}_{BC}$ (Standard PINN)',
                color='#2196F3', linestyle='--', alpha=0.7)
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['bc_loss'],
                linewidth=2.0, label='$\\mathcal{L}_{BC}$ (Scale-PINN)',
                color='#F44336', linestyle='--', alpha=0.7)
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black',
              fancybox=False, framealpha=0.9, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.12, '(a)', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    ax = axes[1]
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['total_loss'],
                linewidth=2.5, label='Total Loss (Standard PINN)',
                color='#2196F3', linestyle='-')
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['total_loss'],
                linewidth=2.5, label='Total Loss (Scale-PINN)',
                color='#F44336', linestyle='-')
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black',
              fancybox=False, framealpha=0.9, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.12, '(b)', transform=ax.transAxes,
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
        print("  Skipping L2 error comparison (need both M1 and M2).")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(modes_hist['M1']['iter'], modes_hist['M1']['rl2'],
                linewidth=2.5, label='Standard PINN (M1)', color='#2196F3')
    ax.semilogy(modes_hist['M2']['iter'], modes_hist['M2']['rl2'],
                linewidth=2.5, label='Scale-PINN (M2)', color='#F44336')
    ax.set_xlabel('Iteration', fontweight='bold')
    ax.set_ylabel('Relative $L_2$ Error (velocity)', fontweight='bold')
    ax.set_title('Convergence Comparison: Standard PINN vs Scale-PINN',
                 fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black',
              fancybox=False, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_velocity_field(mode, filepath):
    setup_plot_style()
    pred = np.load(os.path.join(DATA_DIR, f'predictions_{mode}.npz'))
    data_X = pred['data_X']
    xp, yp = data_X[:, 0], data_X[:, 1]

    ref_u, ref_v = pred['gt_u'].ravel(), pred['gt_v'].ravel()
    pr_u, pr_v = pred['pred_u'].ravel(), pred['pred_v'].ravel()

    gt_mag = np.sqrt(ref_u**2 + ref_v**2)
    pr_mag = np.sqrt(pr_u**2 + pr_v**2)
    err_mag = np.abs(gt_mag - pr_mag)

    method_name = 'Standard PINN' if mode == 'M1' else 'Scale-PINN'

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    titles = [f'Reference $|\\mathbf{{u}}|$',
              f'{method_name} $|\\mathbf{{u}}|$',
              'Absolute Error']
    data_list = [gt_mag, pr_mag, err_mag]
    cmaps = ['RdYlBu_r', 'RdYlBu_r', 'hot_r']
    labels = ['(a)', '(b)', '(c)']

    vmin = min(gt_mag.min(), pr_mag.min())
    vmax = max(gt_mag.max(), pr_mag.max())

    for idx, (ax, title, data, cmap, label) in enumerate(
            zip(axes, titles, data_list, cmaps, labels)):
        if idx < 2:
            im = ax.tricontourf(xp, yp, data, levels=80, cmap=cmap,
                                vmin=vmin, vmax=vmax)
        else:
            im = ax.tricontourf(xp, yp, data, levels=80, cmap=cmap)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=12, width=1.5)
        for spine in cb.ax.spines.values():
            spine.set_linewidth(1.5)
        ax.set_xlabel('$x$', fontweight='bold')
        ax.set_ylabel('$y$', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.text(0.02, -0.12, label, transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top')

    fig.suptitle(f'Velocity Magnitude — {method_name} (Re=7500)',
                 fontsize=20, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_velocity_comparison_m1_m2(filepath):
    setup_plot_style()
    p1 = os.path.join(DATA_DIR, 'predictions_M1.npz')
    p2 = os.path.join(DATA_DIR, 'predictions_M2.npz')
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("  Skipping M1-M2 velocity comparison (need both predictions).")
        return

    d1 = np.load(p1)
    d2 = np.load(p2)
    xp, yp = d1['data_X'][:, 0], d1['data_X'][:, 1]

    gt_u, gt_v = d1['gt_u'].ravel(), d1['gt_v'].ravel()
    vel_ref = np.sqrt(gt_u**2 + gt_v**2)

    vel_m1 = np.sqrt(d1['pred_u'].ravel()**2 + d1['pred_v'].ravel()**2)
    vel_m2 = np.sqrt(d2['pred_u'].ravel()**2 + d2['pred_v'].ravel()**2)
    err_m1 = np.abs(vel_m1 - vel_ref)
    err_m2 = np.abs(vel_m2 - vel_ref)

    vmin_vel = min(vel_ref.min(), vel_m1.min(), vel_m2.min())
    vmax_vel = max(vel_ref.max(), vel_m1.max(), vel_m2.max())
    vmax_err = max(err_m1.max(), err_m2.max())

    fig, axes = plt.subplots(2, 3, figsize=(20, 10.5))
    row_data = [
        ('Standard PINN', vel_ref, vel_m1, err_m1, ['(a)', '(b)', '(c)']),
        ('Scale-PINN', vel_ref, vel_m2, err_m2, ['(d)', '(e)', '(f)']),
    ]

    for r, (name, ref, pred, err, lbls) in enumerate(row_data):
        for c, (data, title, cmap) in enumerate([
            (ref, 'Reference $|\\mathbf{u}|$', 'RdYlBu_r'),
            (pred, f'{name} $|\\mathbf{{u}}|$', 'RdYlBu_r'),
            (err, f'Error ({name})', 'hot_r'),
        ]):
            ax = axes[r, c]
            if c < 2:
                im = ax.tricontourf(xp, yp, data, levels=80, cmap=cmap,
                                    vmin=vmin_vel, vmax=vmax_vel)
            else:
                im = ax.tricontourf(xp, yp, data, levels=80, cmap=cmap,
                                    vmin=0, vmax=vmax_err)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=11, width=1.3)
            for spine in cb.ax.spines.values():
                spine.set_linewidth(1.3)
            ax.set_xlabel('$x$', fontweight='bold')
            ax.set_ylabel('$y$', fontweight='bold')
            ax.set_title(title, fontsize=15, fontweight='bold')
            ax.set_aspect('equal')
            ax.text(0.02, -0.10, lbls[c], transform=ax.transAxes,
                    fontsize=18, fontweight='bold', va='top')

    fig.suptitle('Velocity Magnitude — Standard PINN vs Scale-PINN (Re=7500)',
                 fontsize=20, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_uvp_contours(mode, filepath):
    setup_plot_style()
    pred = np.load(os.path.join(DATA_DIR, f'predictions_{mode}.npz'))
    data_X = pred['data_X']
    xp, yp = data_X[:, 0], data_X[:, 1]

    method_name = 'Standard PINN' if mode == 'M1' else 'Scale-PINN'
    fields = [
        ('u', pred['gt_u'].ravel(), pred['pred_u'].ravel()),
        ('v', pred['gt_v'].ravel(), pred['pred_v'].ravel()),
        ('p', pred['gt_p'].ravel(), pred['pred_p'].ravel()),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for row, (fname, ref, prd) in enumerate(fields):
        err = np.abs(ref - prd)
        vmin = min(ref.min(), prd.min())
        vmax = max(ref.max(), prd.max())
        emax = max(err.max(), 1e-10)

        items = [
            (ref, 'RdYlBu_r', vmin, vmax, f'Reference ${fname}$'),
            (prd, 'RdYlBu_r', vmin, vmax, f'{method_name} ${fname}$'),
            (err, 'hot_r', 0, emax, 'Absolute Error'),
        ]
        for col, (data, cmap, lo, hi, label) in enumerate(items):
            ax = axes[row, col]
            tag = chr(97 + row * 3 + col)
            im = ax.tricontourf(xp, yp, data, levels=80, cmap=cmap,
                                vmin=lo, vmax=hi)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=11, width=1.5)
            for spine in cb.ax.spines.values():
                spine.set_linewidth(1.3)
            ax.set_xlabel('$x$', fontweight='bold')
            ax.set_ylabel('$y$', fontweight='bold')
            ax.set_title(f'({tag}) {label}', fontweight='bold', loc='left')
            ax.set_aspect('equal')

    fig.suptitle(f'Flow Fields — {method_name} (Re=7500)',
                 fontsize=20, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_centerline_comparison(filepath):
    setup_plot_style()
    p1 = os.path.join(DATA_DIR, 'predictions_M1.npz')
    p2 = os.path.join(DATA_DIR, 'predictions_M2.npz')
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print("  Skipping centerline comparison (need both predictions).")
        return

    d1 = np.load(p1)
    d2 = np.load(p2)
    data_X = d1['data_X']
    xp, yp = data_X[:, 0], data_X[:, 1]

    x_vals = np.unique(xp)
    y_vals = np.unique(yp)
    x_mid = x_vals[x_vals.size // 2]
    y_mid = y_vals[y_vals.size // 2]

    tol = 1e-6
    mask_xmid = np.abs(xp - x_mid) < tol
    mask_ymid = np.abs(yp - y_mid) < tol

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    y_line = yp[mask_xmid]
    sort_idx = np.argsort(y_line)
    y_line = y_line[sort_idx]
    gt_u_line = d1['gt_u'].ravel()[mask_xmid][sort_idx]
    m1_u_line = d1['pred_u'].ravel()[mask_xmid][sort_idx]
    m2_u_line = d2['pred_u'].ravel()[mask_xmid][sort_idx]

    ax.plot(gt_u_line, y_line, 'k-', linewidth=2.5, label='Reference')
    ax.plot(m1_u_line, y_line, '--', linewidth=2.0, color='#2196F3',
            label='Standard PINN')
    ax.plot(m2_u_line, y_line, '-.', linewidth=2.0, color='#F44336',
            label='Scale-PINN')
    ax.set_xlabel('$u$', fontweight='bold')
    ax.set_ylabel('$y$', fontweight='bold')
    ax.set_title(f'$u(y)$ at $x = {x_mid:.4f}$', fontweight='bold')
    ax.legend(loc='best', frameon=True, edgecolor='black',
              fancybox=False, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.12, '(a)', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    ax = axes[1]
    x_line = xp[mask_ymid]
    sort_idx = np.argsort(x_line)
    x_line = x_line[sort_idx]
    gt_v_line = d1['gt_v'].ravel()[mask_ymid][sort_idx]
    m1_v_line = d1['pred_v'].ravel()[mask_ymid][sort_idx]
    m2_v_line = d2['pred_v'].ravel()[mask_ymid][sort_idx]

    ax.plot(x_line, gt_v_line, 'k-', linewidth=2.5, label='Reference')
    ax.plot(x_line, m1_v_line, '--', linewidth=2.0, color='#2196F3',
            label='Standard PINN')
    ax.plot(x_line, m2_v_line, '-.', linewidth=2.0, color='#F44336',
            label='Scale-PINN')
    ax.set_xlabel('$x$', fontweight='bold')
    ax.set_ylabel('$v$', fontweight='bold')
    ax.set_title(f'$v(x)$ at $y = {y_mid:.4f}$', fontweight='bold')
    ax.legend(loc='best', frameon=True, edgecolor='black',
              fancybox=False, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, -0.12, '(b)', transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='top')

    fig.suptitle('Centerline Velocity Profiles — Standard PINN vs Scale-PINN (Re=7500)',
                 fontsize=18, fontweight='bold', y=1.02)
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
            plot_velocity_field(
                m, os.path.join(FIG_DIR, f'fig_velocity_{m}.png'))
            plot_uvp_contours(
                m, os.path.join(FIG_DIR, f'fig_uvp_comparison_{m}.png'))

    plot_velocity_comparison_m1_m2(
        os.path.join(FIG_DIR, 'fig_velocity_M1_vs_M2.png'))
    plot_loss_comparison(
        os.path.join(FIG_DIR, 'fig_loss_comparison.png'))
    plot_l2_error_comparison(
        os.path.join(FIG_DIR, 'fig_l2_error_comparison.png'))
    plot_centerline_comparison(
        os.path.join(FIG_DIR, 'fig_centerline_comparison.png'))

    print("All figures generated.")


# ============================================================
# Comparison summary
# ============================================================
def write_comparison_summary():
    lines = []
    lines.append("=" * 82)
    lines.append("Comparison Summary: Lid-Driven Cavity Re=7500 (2D Steady NS)")
    lines.append("Standard PINN (M1) vs Scale-PINN (M2)")
    lines.append("=" * 82)
    lines.append(f"Network: shared trunk + 3 branches, n_nodes={N_NODES}, "
                 f"activation=SiLU, Fourier features")
    lines.append(f"Optimizer: Adam + CosineDecay(lr={MAX_LR}, exponent={EXPONENT})")
    lines.append(f"Batch: BS_ALL={BS_ALL}, BS_BC={BS_BC}, weight_bc={WEIGHT_BC}")
    lines.append(f"Re = {RE}")
    lines.append("-" * 82)
    lines.append(f"{'Model':<16} {'Params':>8} {'Iters':>7} "
                 f"{'Best RL2':>12} {'Final RL2':>12} {'Final MSE':>12} "
                 f"{'Time(s)':>8}")
    lines.append("-" * 82)

    results_data = {}
    for m in ['M1', 'M2']:
        hist_path = os.path.join(DATA_DIR, f'loss_history_{m}.txt')
        pred_path = os.path.join(DATA_DIR, f'predictions_{m}.npz')
        if os.path.exists(hist_path) and os.path.exists(pred_path):
            hist = load_training_history(hist_path)
            pred = np.load(pred_path)

            best_rl2 = min(hist['rl2'])
            n_iters = int(hist['iter'][-1])
            total_time = hist['time'][-1]

            uv_pred = np.hstack([pred['pred_u'], pred['pred_v']])
            uv_gt = np.hstack([pred['gt_u'], pred['gt_v']])
            full_rl2 = float(np.linalg.norm(uv_pred - uv_gt) / np.linalg.norm(uv_gt))
            full_mse = float(np.mean((uv_pred - uv_gt)**2))

            method_name = 'Std PINN (M1)' if m == 'M1' else 'Scale-PINN (M2)'
            lines.append(f"{method_name:<16} {59520:>8d} {n_iters:>7d} "
                         f"{best_rl2:>12.6e} {full_rl2:>12.6e} {full_mse:>12.6e} "
                         f"{total_time:>8.1f}")
            results_data[m] = {
                'best_rl2': best_rl2, 'full_rl2': full_rl2,
                'full_mse': full_mse, 'time': total_time,
            }

    lines.append("-" * 82)

    if 'M1' in results_data and 'M2' in results_data:
        r1 = results_data['M1']
        r2 = results_data['M2']
        impr_best = (r1['best_rl2'] - r2['best_rl2']) / r1['best_rl2'] * 100
        impr_full = (r1['full_rl2'] - r2['full_rl2']) / r1['full_rl2'] * 100
        lines.append(f"Scale-PINN improvement (best RL2): {impr_best:+.1f}%  "
                     f"({r1['best_rl2']:.4e} -> {r2['best_rl2']:.4e})")
        lines.append(f"Scale-PINN improvement (full RL2): {impr_full:+.1f}%  "
                     f"({r1['full_rl2']:.4e} -> {r2['full_rl2']:.4e})")

    lines.append("=" * 82)
    summary = '\n'.join(lines)
    print('\n' + summary)

    with open(os.path.join(DATA_DIR, 'comparison_summary.txt'), 'w') as f:
        f.write(summary + '\n')
    print(f"Saved: {os.path.join(DATA_DIR, 'comparison_summary.txt')}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Scale-PINN vs Standard PINN: LDC Re=7500')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['M1', 'M2', 'both'],
                        help='M1=Standard PINN, M2=Scale-PINN, both=run both')
    parser.add_argument('--niter', type=int, default=50000,
                        help='Max training iterations')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 500 iterations')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only generate plots from saved data')
    args = parser.parse_args()

    if args.plot_only:
        generate_all_plots()
        write_comparison_summary()
        return

    n_iter = 500 if args.quick else args.niter
    modes = ['M1', 'M2'] if args.mode == 'both' else [args.mode]

    print("=" * 70)
    print("Scale-PINN vs Standard PINN — Lid-Driven Cavity (Re=7500)")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Modes: {modes}")
    print(f"Max iterations: {n_iter}")
    print("=" * 70)

    results = {}
    for mode in modes:
        res = train_model(mode, max_iters=n_iter, seed=SEED)
        results[mode] = res

    generate_all_plots()
    write_comparison_summary()

    print("\nDone!")


if __name__ == '__main__':
    main()
