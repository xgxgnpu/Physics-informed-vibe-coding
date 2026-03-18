"""
Case 1: 2D Steady Navier-Stokes (Cylinder Flow) — VS-PINN Adam (JAX)
=====================================================================
PDE: Steady incompressible Navier-Stokes
  Continuity: u_x + v_y = 0
  x-momentum: ρ(u·u_x + v·u_y) + p_x − μ(u_xx + u_yy) = 0
  y-momentum: ρ(u·v_x + v·v_y) + p_y − μ(v_xx + v_yy) = 0
  ρ=1, μ=0.02

Network: MLP 2→40→40→40→40→40→3 (u,v,p), tanh, Xavier init
VS-PINN scaling: N=10
Training: Adam lr=0.001, 80000 iterations, collocation resampled each epoch

Self-contained single file. Run: python ns2d_vspinn_pinn.py
"""

import os
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_command_buffer=')

import pickle
import time

import jax
import jax.numpy as jnp
from jax import random
import optax
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import rcParams

# ==================== Paths ====================
WORKDIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKDIR, 'data')
FIG_DIR = os.path.join(WORKDIR, 'figures')
CKPT_DIR = os.path.join(WORKDIR, 'checkpoints')
for d in [DATA_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== Plotting Style ====================
def setup_plot_style():
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 14
    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 16
    rcParams['axes.linewidth'] = 1.8
    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5
    rcParams['legend.fontsize'] = 12
    rcParams['legend.framealpha'] = 0.9
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'

# ==================== MLP Utilities ====================
def init_mlp_params(key, layer_sizes, init_type='default'):
    params = []
    for i in range(len(layer_sizes) - 1):
        key, wkey, bkey = random.split(key, 3)
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        if init_type == 'xavier':
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            w = random.uniform(wkey, (fan_in, fan_out), minval=-limit, maxval=limit)
        else:
            limit = 1.0 / jnp.sqrt(fan_in)
            w = random.uniform(wkey, (fan_in, fan_out), minval=-limit, maxval=limit)
        b = random.uniform(bkey, (fan_out,), minval=-limit, maxval=limit)
        params.append({'w': w, 'b': b})
    return params


def mlp_forward(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer['w'] + layer['b'])
    last = params[-1]
    return x @ last['w'] + last['b']


def count_params(params):
    total = 0
    for layer in params:
        total += layer['w'].size + layer['b'].size
    return total


# ==================== Data I/O ====================
def save_params(params, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(jax.tree.map(np.array, params), f)


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

# ==================== Configuration ====================
SEED = 1234
N_VS = 10
RHO = 1.0
MU = 0.02
LAYERS = [2, 40, 40, 40, 40, 40, 3]
LR = 0.001
N_EPOCHS = 80000
N_C = 8000
N_R = 600
N_B = 200
N_W = 400
N_S = 200
BC_WEIGHT = 2.0
LOG_EVERY = 100

XMIN = 0.0
XMAX = 1.1 * N_VS
YMIN = 0.0
YMAX = 0.41 * N_VS
XC = 0.2 * N_VS
YC = 0.2 * N_VS
R_CYL = 0.05 * N_VS


def func_u_inlet(y):
    ys = y / N_VS
    return 4.0 * ys * (0.41 - ys) / (0.41 ** 2)


def remove_pts_inside_cylinder(xy, xc, yc, r):
    dist = np.sqrt((xy[:, 0] - xc) ** 2 + (xy[:, 1] - yc) ** 2)
    return xy[dist > r]


# ==================== Load Reference Data ====================
def load_reference_data():
    mat = scipy.io.loadmat(os.path.join(DATA_DIR, 'FluentSol.mat'))
    x = mat['x'].flatten().astype(np.float32)
    y = mat['y'].flatten().astype(np.float32)
    vx = mat['vx'].flatten().astype(np.float32)
    vy = mat['vy'].flatten().astype(np.float32)
    p = mat['p'].flatten().astype(np.float32)
    return x, y, vx, vy, p


# ==================== Generate Boundary Data ====================
def generate_boundary_data():
    inlet_xy = np.linspace([XMIN, YMIN], [XMIN, YMAX], N_B).astype(np.float32)
    inlet_u = func_u_inlet(inlet_xy[:, 1]).reshape(-1).astype(np.float32)
    inlet_v = np.zeros(N_B, dtype=np.float32)

    outlet_xy = np.linspace([XMAX, YMIN], [XMAX, YMAX], N_B).astype(np.float32)
    outlet_p = np.zeros(N_B, dtype=np.float32)

    wallup_xy = np.linspace([XMIN, YMAX], [XMAX, YMAX], N_W).astype(np.float32)
    walldn_xy = np.linspace([XMIN, YMIN], [XMAX, YMIN], N_W).astype(np.float32)
    wallup_uv = np.zeros((N_W, 2), dtype=np.float32)
    walldn_uv = np.zeros((N_W, 2), dtype=np.float32)

    theta = np.linspace(0.0, 2 * np.pi, N_S).astype(np.float32)
    cyld_x = R_CYL * np.cos(theta) + XC
    cyld_y = R_CYL * np.sin(theta) + YC
    cyld_xy = np.stack([cyld_x, cyld_y], axis=1).astype(np.float32)
    cyld_uv = np.zeros((N_S, 2), dtype=np.float32)

    inlet_uv = np.stack([inlet_u, inlet_v], axis=1)
    bnd_xy = np.concatenate([inlet_xy, wallup_xy, walldn_xy, cyld_xy], axis=0)
    bnd_uv = np.concatenate([inlet_uv, wallup_uv, walldn_uv, cyld_uv], axis=0)

    return bnd_xy, bnd_uv, outlet_xy, outlet_p


# ==================== Sample Collocation Points ====================
def sample_collocation_np(rng, n_c=N_C, n_r=N_R, target_size=None):
    if target_size is None:
        target_size = n_c + n_r
    oversample = int(1.1 * n_c)
    x_col = rng.uniform(XMIN, XMAX, (oversample, 1))
    y_col = rng.uniform(YMIN, YMAX, (oversample, 1))
    xy_col = np.concatenate([x_col, y_col], axis=1)

    x_ref = rng.uniform(XC - 2 * R_CYL, XC + 2 * R_CYL, (n_r, 1))
    y_ref = rng.uniform(YC - 2 * R_CYL, YC + 2 * R_CYL, (n_r, 1))
    xy_ref = np.concatenate([x_ref, y_ref], axis=1)

    xy_all = np.concatenate([xy_col, xy_ref], axis=0)
    xy_all = remove_pts_inside_cylinder(xy_all, XC, YC, R_CYL)
    if len(xy_all) >= target_size:
        xy_all = xy_all[:target_size]
    else:
        pad = np.tile(xy_all[-1:], (target_size - len(xy_all), 1))
        xy_all = np.concatenate([xy_all, pad], axis=0)
    return xy_all.astype(np.float32)


# ==================== Network Scalar Functions ====================
def net_u(params, x, y):
    inp = jnp.stack([x, y])
    return mlp_forward(params, inp)[0]


def net_v(params, x, y):
    inp = jnp.stack([x, y])
    return mlp_forward(params, inp)[1]


def net_p(params, x, y):
    inp = jnp.stack([x, y])
    return mlp_forward(params, inp)[2]


def net_uvp_batch(params, x_arr, y_arr):
    def fwd(x, y):
        inp = jnp.stack([x, y])
        return mlp_forward(params, inp)
    out = jax.vmap(fwd)(x_arr, y_arr)
    return out[:, 0], out[:, 1], out[:, 2]


# ==================== PDE Residuals ====================
def ns_residual_single(params, x, y):
    u = net_u(params, x, y)
    v = net_v(params, x, y)

    u_x = jax.grad(net_u, argnums=1)(params, x, y)
    u_y = jax.grad(net_u, argnums=2)(params, x, y)
    v_x = jax.grad(net_v, argnums=1)(params, x, y)
    v_y = jax.grad(net_v, argnums=2)(params, x, y)

    u_xx = jax.grad(lambda p, xx, yy: jax.grad(net_u, 1)(p, xx, yy), 1)(params, x, y)
    u_yy = jax.grad(lambda p, xx, yy: jax.grad(net_u, 2)(p, xx, yy), 2)(params, x, y)
    v_xx = jax.grad(lambda p, xx, yy: jax.grad(net_v, 1)(p, xx, yy), 1)(params, x, y)
    v_yy = jax.grad(lambda p, xx, yy: jax.grad(net_v, 2)(p, xx, yy), 2)(params, x, y)

    p_x = jax.grad(net_p, argnums=1)(params, x, y)
    p_y = jax.grad(net_p, argnums=2)(params, x, y)

    N = N_VS
    r1 = (RHO * (u * N * u_x + v * N * u_y) + N * p_x
          - MU * (N * N * u_xx + N * N * u_yy)) / N
    r2 = (RHO * (u * N * v_x + v * N * v_y) + N * p_y
          - MU * (N * N * v_xx + N * N * v_yy)) / N
    r3 = (N * u_x + N * v_y) / N

    return r1, r2, r3


ns_residual_vmap = jax.vmap(ns_residual_single, in_axes=(None, 0, 0))

CHUNK_SIZE = 2000


def ns_residual_chunked(params, x_arr, y_arr):
    n_chunks = x_arr.shape[0] // CHUNK_SIZE
    x_ch = x_arr.reshape(n_chunks, CHUNK_SIZE)
    y_ch = y_arr.reshape(n_chunks, CHUNK_SIZE)

    def body(carry, xy):
        r = ns_residual_vmap(params, xy[0], xy[1])
        return carry, r
    _, (r1_all, r2_all, r3_all) = jax.lax.scan(body, None, (x_ch, y_ch))
    return r1_all.reshape(-1), r2_all.reshape(-1), r3_all.reshape(-1)


# ==================== Loss Function ====================
def loss_fn(params, xy_col, bnd_xy, bnd_uv, outlet_xy, outlet_p_ref):
    x_col = xy_col[:, 0]
    y_col = xy_col[:, 1]
    r1, r2, r3 = ns_residual_chunked(params, x_col, y_col)
    mse_r1 = jnp.mean(r1 ** 2)
    mse_r2 = jnp.mean(r2 ** 2)
    mse_r3 = jnp.mean(r3 ** 2)

    u_bnd, v_bnd, _ = net_uvp_batch(params, bnd_xy[:, 0], bnd_xy[:, 1])
    mse_bnd_u = jnp.mean((u_bnd - bnd_uv[:, 0]) ** 2)
    mse_bnd_v = jnp.mean((v_bnd - bnd_uv[:, 1]) ** 2)

    _, _, p_out = net_uvp_batch(params, outlet_xy[:, 0], outlet_xy[:, 1])
    mse_outlet = jnp.mean((p_out - outlet_p_ref) ** 2)

    loss_pde = mse_r1 + mse_r2 + mse_r3
    loss_bc = BC_WEIGHT * (mse_bnd_u + mse_bnd_v) + BC_WEIGHT * mse_outlet
    total = loss_pde + loss_bc
    return total, (mse_r1, mse_r2, mse_r3, mse_bnd_u, mse_bnd_v, mse_outlet)


# ==================== Compute L2 Errors ====================
def compute_l2_errors(params, ref_x, ref_y, ref_vx, ref_vy, ref_p):
    xs = ref_x * N_VS
    ys = ref_y * N_VS
    u_pred, v_pred, p_pred = net_uvp_batch(params, jnp.array(xs), jnp.array(ys))
    l2_u = jnp.sqrt(jnp.mean((u_pred - ref_vx) ** 2) / jnp.mean(ref_vx ** 2))
    l2_v = jnp.sqrt(jnp.mean((v_pred - ref_vy) ** 2) / jnp.mean(ref_vy ** 2))
    l2_p = jnp.sqrt(jnp.mean((p_pred - ref_p) ** 2) / jnp.mean(ref_p ** 2))
    return float(l2_u), float(l2_v), float(l2_p)


# ==================== Plotting ====================
def plot_loss_components(history, keys, labels, filepath, title='Loss Components'):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = history.get('epoch', np.arange(len(list(history.values())[0])))
    for key, label in zip(keys, labels):
        if key in history:
            ax.semilogy(epochs, history[key], linewidth=2, label=label)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.legend(loc='best', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_l2_errors_ns(history, filepath):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = history['epoch']
    ax.semilogy(epochs, history['l2_u'], linewidth=2, label='$L_2(u)$', color='C0')
    ax.semilogy(epochs, history['l2_v'], linewidth=2, label='$L_2(v)$', color='C1')
    ax.semilogy(epochs, history['l2_p'], linewidth=2, label='$L_2(p)$', color='C2')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('$L_2$ Relative Error', fontweight='bold')
    ax.set_title('L2 Relative Error History', fontweight='bold', fontsize=16)
    ax.legend(loc='best', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_ns_field(x, y, ref, pred, error, filepath, field_name='u',
                  levels=50, cyl_center=(0.2, 0.2), cyl_radius=0.05):
    setup_plot_style()
    triang = tri.Triangulation(x, y)
    xc, yc = cyl_center
    cx = x[triang.triangles].mean(axis=1)
    cy = y[triang.triangles].mean(axis=1)
    mask = (cx - xc) ** 2 + (cy - yc) ** 2 < cyl_radius ** 2
    triang.set_mask(mask)

    vmin = min(np.min(ref), np.min(pred))
    vmax = max(np.max(ref), np.max(pred))

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    titles = [f'Reference ${field_name}$', f'Predicted ${field_name}$', 'Absolute Error']
    data_list = [ref, pred, error]
    labels = ['(a)', '(b)', '(c)']

    for idx, (ax, title, data, label) in enumerate(zip(axes, titles, data_list, labels)):
        if idx < 2:
            cs = ax.tricontourf(triang, data, levels=levels, cmap='jet',
                                vmin=vmin, vmax=vmax)
        else:
            cs = ax.tricontourf(triang, data, levels=levels, cmap='hot_r')
        cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=11, width=1.2)

        circle = plt.Circle((xc, yc), cyl_radius, color='grey',
                             fill=True, zorder=5, alpha=0.8)
        ax.add_patch(circle)

        ax.set_xlabel('$x$', fontweight='bold')
        ax.set_ylabel('$y$', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_aspect('equal')
        ax.text(0.02, -0.15, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def run_plotting():
    """Generate figures from saved data."""
    history_path = os.path.join(DATA_DIR, 'history.txt')
    pred_path = os.path.join(DATA_DIR, 'predictions.npz')
    if not os.path.exists(history_path) or not os.path.exists(pred_path):
        print("Skipping plotting: history or predictions not found.")
        return

    history = load_training_history(history_path)
    print(f"Loaded history: {len(history['epoch'])} records")

    plot_loss_components(
        history,
        keys=['mse_mom_x', 'mse_mom_y', 'mse_cont', 'mse_bc'],
        labels=['$x$-momentum', '$y$-momentum', 'Continuity', 'BC loss'],
        filepath=os.path.join(FIG_DIR, 'loss_components.png'),
        title='Loss Components (NS Adam)'
    )

    plot_l2_errors_ns(history, os.path.join(FIG_DIR, 'l2_error_history.png'))

    pred = np.load(pred_path)
    x = pred['x']
    y = pred['y']
    print(f"Prediction data: {len(x)} points")

    for field, ref_key, pred_key, name in [
        ('u', 'u_ref', 'u_pred', 'u'),
        ('v', 'v_ref', 'v_pred', 'v'),
        ('p', 'p_ref', 'p_pred', 'p'),
    ]:
        ref_f = pred[ref_key]
        pred_f = pred[pred_key]
        err_f = np.abs(ref_f - pred_f)
        plot_ns_field(
            x, y, ref_f, pred_f, err_f,
            filepath=os.path.join(FIG_DIR, f'field_{field}.png'),
            field_name=name
        )

    print("All plots generated.")


# ==================== Main Training ====================
def main():
    print("=" * 60)
    print("VS-PINN Case 1: 2D Navier-Stokes (Cylinder Flow) — Adam Only")
    print("=" * 60)

    ref_x, ref_y, ref_vx, ref_vy, ref_p = load_reference_data()
    print(f"Reference data: {len(ref_x)} points")
    print(f"Physical domain: x∈[{ref_x.min():.4f}, {ref_x.max():.4f}], "
          f"y∈[{ref_y.min():.4f}, {ref_y.max():.4f}]")
    print(f"Scaled domain (N={N_VS}): x∈[{XMIN}, {XMAX}], y∈[{YMIN}, {YMAX}]")

    bnd_xy, bnd_uv, outlet_xy, outlet_p_ref = generate_boundary_data()
    bnd_xy_j = jnp.array(bnd_xy)
    bnd_uv_j = jnp.array(bnd_uv)
    outlet_xy_j = jnp.array(outlet_xy)
    outlet_p_j = jnp.array(outlet_p_ref)
    print(f"BC points: {len(bnd_xy)} (inlet+walls+cylinder) + {len(outlet_xy)} (outlet)")

    key = random.PRNGKey(SEED)
    key, init_key = random.split(key)
    params = init_mlp_params(init_key, LAYERS, init_type='xavier')
    n_params = count_params(params)
    print(f"Network: {LAYERS}, params={n_params}")

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, xy_col):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, xy_col, bnd_xy_j, bnd_uv_j, outlet_xy_j, outlet_p_j
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    history = {
        'epoch': [], 'total_loss': [],
        'mse_cont': [], 'mse_mom_x': [], 'mse_mom_y': [], 'mse_bc': [],
        'l2_u': [], 'l2_v': [], 'l2_p': []
    }

    col_target = N_C + N_R
    total_col_size = col_target + len(bnd_xy) + len(outlet_xy)
    n_chunks_needed = (total_col_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    padded_total = n_chunks_needed * CHUNK_SIZE
    pad_bnd_out = np.concatenate([bnd_xy, outlet_xy], axis=0).astype(np.float32)

    np_rng = np.random.RandomState(SEED)

    print(f"\nTraining: Adam lr={LR}, {N_EPOCHS} epochs")
    print(f"Collocation: {N_C} interior + {N_R} refine (resampled each epoch)")
    print(f"Fixed collocation array: {padded_total} (padded for chunking)")
    print("-" * 60)

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        xy_col_np = sample_collocation_np(np_rng, target_size=col_target)
        xy_full_np = np.concatenate([xy_col_np, pad_bnd_out], axis=0)
        if len(xy_full_np) < padded_total:
            pad = np.tile(xy_full_np[-1:], (padded_total - len(xy_full_np), 1))
            xy_full_np = np.concatenate([xy_full_np, pad], axis=0)
        xy_col_full = jnp.array(xy_full_np[:padded_total])

        params, opt_state, loss_val, aux = train_step(params, opt_state, xy_col_full)
        mse_r1, mse_r2, mse_r3, mse_bu, mse_bv, mse_out = aux

        if epoch % LOG_EVERY == 0 or epoch == 1:
            l2_u, l2_v, l2_p = compute_l2_errors(
                params, ref_x, ref_y, ref_vx, ref_vy, ref_p
            )
            mse_bc_total = float(BC_WEIGHT * (mse_bu + mse_bv) + BC_WEIGHT * mse_out)

            history['epoch'].append(epoch)
            history['total_loss'].append(float(loss_val))
            history['mse_cont'].append(float(mse_r3))
            history['mse_mom_x'].append(float(mse_r1))
            history['mse_mom_y'].append(float(mse_r2))
            history['mse_bc'].append(mse_bc_total)
            history['l2_u'].append(l2_u)
            history['l2_v'].append(l2_v)
            history['l2_p'].append(l2_p)

            elapsed = time.time() - t0
            print(f"Epoch {epoch:6d}/{N_EPOCHS} | Loss: {float(loss_val):.6e} | "
                  f"PDE: {float(mse_r1+mse_r2+mse_r3):.4e} | BC: {mse_bc_total:.4e} | "
                  f"L2(u): {l2_u:.4e} L2(v): {l2_v:.4e} L2(p): {l2_p:.4e} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")
    print(f"Final L2 errors: u={history['l2_u'][-1]:.6e}, "
          f"v={history['l2_v'][-1]:.6e}, p={history['l2_p'][-1]:.6e}")

    # Save
    save_params(params, os.path.join(CKPT_DIR, 'params.pkl'))
    save_training_history(history, os.path.join(DATA_DIR, 'history.txt'))

    xs_test = jnp.array(ref_x * N_VS)
    ys_test = jnp.array(ref_y * N_VS)
    u_pred, v_pred, p_pred = net_uvp_batch(params, xs_test, ys_test)
    save_predictions(
        os.path.join(DATA_DIR, 'predictions.npz'),
        x=ref_x, y=ref_y,
        u_ref=ref_vx, v_ref=ref_vy, p_ref=ref_p,
        u_pred=np.array(u_pred), v_pred=np.array(v_pred), p_pred=np.array(p_pred)
    )
    print("Results saved.")

    # Plot
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)
    run_plotting()


if __name__ == '__main__':
    main()
