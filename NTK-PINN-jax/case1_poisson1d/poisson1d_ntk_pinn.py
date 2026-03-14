"""
Case 1: 1D Poisson Equation — NTK Analysis with PINN (JAX)
WITH z-score input normalization + gradient chain-rule correction

PDE: u_xx(x) = f(x),  x in [0, 1]
Exact solution: u(x) = sin(4*pi*x)
BC: u(0) = 0, u(1) = 0

Normalization: input z-score (x_norm = (x - mu) / sigma)
               gradient correction (d/dx_phys = (1/sigma) * d/dx_norm)
"""

import os
import time
import pickle

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

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 12
rcParams['axes.linewidth'] = 2.0
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5

WORKDIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKDIR, 'data')
FIG_DIR = os.path.join(WORKDIR, 'figures')
CKPT_DIR = os.path.join(WORKDIR, 'checkpoints')
for d in [DATA_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# Exact solution
# ============================================================
A_FREQ = 4

def u_exact_fn(x):
    return np.sin(np.pi * A_FREQ * x)

def u_xx_exact_fn(x):
    return -(np.pi * A_FREQ) ** 2 * np.sin(np.pi * A_FREQ * x)

# ============================================================
# Normalization statistics (computed from collocation points)
# ============================================================
NN = 100
X_r_raw = np.linspace(0.0, 1.0, NN)
MU_X = float(X_r_raw.mean())
SIGMA_X = float(X_r_raw.std())

def normalize_x(x):
    return (x - MU_X) / SIGMA_X

# ============================================================
# Network
# ============================================================
LAYERS = [1, 512, 1]

def init_params(layers, key):
    params = []
    for i in range(len(layers) - 1):
        k1, k2, key = random.split(key, 3)
        fan_in = layers[i]
        std = 1.0 / np.sqrt(fan_in)
        W = std * random.normal(k1, (layers[i], layers[i + 1]))
        b = random.normal(k2, (layers[i + 1],))
        params.append((W, b))
    return params


def apply_net(params, x):
    h = x
    for (W, b) in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b).squeeze(-1)


def net_u_single(params, x_scalar):
    """u(x_norm) for a single scalar normalized x."""
    x_in = jnp.array([x_scalar]).reshape(1, 1)
    return apply_net(params, x_in)[0]


def net_u_xx_single(params, x_scalar):
    """u_xx in physical space via chain rule on normalized input.
    d/dx_phys = (1/sigma_x) * d/dx_norm, so u_xx_phys = u_xx_norm / sigma_x^2.
    """
    du_dx = grad(net_u_single, argnums=1)
    d2u_dx2 = grad(du_dx, argnums=1)
    u_xx_norm = d2u_dx2(params, x_scalar)
    return u_xx_norm / (SIGMA_X ** 2)

net_u_batch = vmap(net_u_single, in_axes=(None, 0))
net_u_xx_batch = vmap(net_u_xx_single, in_axes=(None, 0))

# ============================================================
# Loss
# ============================================================
def loss_bcs_fn(params, x_bc_norm, u_bc):
    u_pred = net_u_batch(params, x_bc_norm)
    return jnp.mean((u_pred - u_bc) ** 2)


def loss_res_fn(params, x_r_norm, f_r):
    u_xx_pred = net_u_xx_batch(params, x_r_norm)
    return jnp.mean((u_xx_pred - f_r) ** 2)


def loss_fn(params, x_bc_norm, u_bc, x_r_norm, f_r):
    l_bcs = loss_bcs_fn(params, x_bc_norm, u_bc)
    l_res = loss_res_fn(params, x_r_norm, f_r)
    return l_bcs + l_res, (l_bcs, l_res)

# ============================================================
# NTK computation
# ============================================================
def compute_jacobian(fn_batch, params, x_pts):
    flat_params, unravel = ravel_pytree(params)
    def f_flat(fp):
        p = unravel(fp)
        return fn_batch(p, x_pts)
    J = jacrev(f_flat)(flat_params)
    return J


def compute_ntk_matrices(params, x_bc_norm, x_r_norm):
    J_u = compute_jacobian(net_u_batch, params, x_bc_norm)
    J_r = compute_jacobian(net_u_xx_batch, params, x_r_norm)
    K_uu = J_u @ J_u.T
    K_ur = J_u @ J_r.T
    K_rr = J_r @ J_r.T
    return K_uu, K_ur, K_rr

# ============================================================
# Training data (normalized)
# ============================================================
x_bc1 = np.zeros((NN // 2,))
x_bc2 = np.ones((NN // 2,))
X_bc = np.concatenate([x_bc1, x_bc2])
U_bc = u_exact_fn(X_bc)

X_r = np.linspace(0.0, 1.0, NN)
F_r = u_xx_exact_fn(X_r)

X_bc_norm = normalize_x(X_bc)
X_r_norm = normalize_x(X_r)

X_bc_jax = jnp.array(X_bc_norm)
U_bc_jax = jnp.array(U_bc)
X_r_jax = jnp.array(X_r_norm)
F_r_jax = jnp.array(F_r)

# ============================================================
# Training
# ============================================================
N_ITER = 40001
LR = 1e-3
LOG_EVERY = 100
NTK_EVERY = 100

def train():
    key = random.PRNGKey(1234)
    params = init_params(LAYERS, key)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    flat0, _ = ravel_pytree(params)
    flat0_np = np.array(flat0)
    n_params = flat0.shape[0]

    @jit
    def train_step(params, opt_state):
        (loss_val, (l_bcs, l_res)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, X_bc_jax, U_bc_jax, X_r_jax, F_r_jax
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss_val, l_bcs, l_res

    nn_test = 1000
    X_test = np.linspace(0.0, 1.0, nn_test)
    U_test = u_exact_fn(X_test)
    X_test_norm = normalize_x(X_test)
    X_test_jax = jnp.array(X_test_norm)

    loss_bcs_log = []
    loss_res_log = []
    l2_error_log = []
    K_uu_log = []
    K_ur_log = []
    K_rr_log = []
    weights_change_log = []
    iters_log = []

    print(f"Number of trainable parameters: {n_params}")
    print(f"Network architecture: {LAYERS}")
    print(f"Optimizer: Adam, lr={LR}")
    print(f"Normalization: z-score (mu={MU_X:.4f}, sigma={SIGMA_X:.4f})")
    print(f"Iterations: {N_ITER}\n")

    start_time = time.time()
    best_l2 = 1.0

    for it in range(N_ITER):
        params, opt_state, loss_val, l_bcs, l_res = train_step(params, opt_state)

        if it % LOG_EVERY == 0:
            u_pred = net_u_batch(params, X_test_jax)
            u_pred_np = np.array(u_pred)
            l2_err = np.linalg.norm(U_test - u_pred_np) / np.linalg.norm(U_test)

            loss_bcs_log.append(float(l_bcs))
            loss_res_log.append(float(l_res))
            l2_error_log.append(float(l2_err))
            iters_log.append(it)

            if l2_err < best_l2:
                best_l2 = l2_err

            elapsed = time.time() - start_time
            print(f"It: {it:5d}, Loss: {float(loss_val):.3e}, "
                  f"L_bcs: {float(l_bcs):.3e}, L_res: {float(l_res):.3e}, "
                  f"L2: {l2_err:.3e}, Time: {elapsed:.1f}s")

            flat_cur, _ = ravel_pytree(params)
            flat_cur_np = np.array(flat_cur)
            w_change = np.linalg.norm(flat_cur_np - flat0_np) / np.linalg.norm(flat0_np)
            weights_change_log.append(float(w_change))

        if it % NTK_EVERY == 0:
            K_uu, K_ur, K_rr = compute_ntk_matrices(params, X_bc_jax, X_r_jax)
            K_uu_log.append(np.array(K_uu))
            K_ur_log.append(np.array(K_ur))
            K_rr_log.append(np.array(K_rr))

    total_time = time.time() - start_time
    print(f"\nTraining complete. Total time: {total_time:.1f}s")
    print(f"Best L2 relative error: {best_l2:.3e}")
    print(f"Final L2 relative error: {l2_error_log[-1]:.3e}")

    # ============================================================
    # Save model
    # ============================================================
    with open(os.path.join(CKPT_DIR, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # ============================================================
    # Save data
    # ============================================================
    header_loss = "iteration  loss_bcs  loss_res  l2_error"
    loss_data = np.column_stack([iters_log, loss_bcs_log, loss_res_log, l2_error_log])
    np.savetxt(os.path.join(DATA_DIR, 'loss_history.txt'), loss_data, header=header_loss, fmt='%.6e')

    u_pred_final = np.array(net_u_batch(params, X_test_jax))
    pred_data = np.column_stack([X_test, U_test, u_pred_final, np.abs(U_test - u_pred_final)])
    np.savetxt(os.path.join(DATA_DIR, 'prediction.txt'), pred_data,
               header="x  u_exact  u_pred  abs_error", fmt='%.6e')

    lambda_K_log = []
    lambda_K_uu_log = []
    lambda_K_rr_log = []
    K_full_list = []

    for k in range(len(K_uu_log)):
        K_uu = K_uu_log[k]
        K_ur = K_ur_log[k]
        K_rr = K_rr_log[k]
        K_full = np.block([[K_uu, K_ur], [K_ur.T, K_rr]])
        K_full_list.append(K_full)

        eig_K = np.sort(np.real(np.linalg.eigvalsh(K_full)))[::-1]
        eig_K_uu = np.sort(np.real(np.linalg.eigvalsh(K_uu)))[::-1]
        eig_K_rr = np.sort(np.real(np.linalg.eigvalsh(K_rr)))[::-1]

        lambda_K_log.append(eig_K)
        lambda_K_uu_log.append(eig_K_uu)
        lambda_K_rr_log.append(eig_K_rr)

    ntk_change = []
    K0 = K_full_list[0]
    K0_norm = np.linalg.norm(K0)
    for K in K_full_list:
        ntk_change.append(np.linalg.norm(K - K0) / K0_norm)

    np.savetxt(os.path.join(DATA_DIR, 'ntk_change.txt'),
               np.column_stack([iters_log[:len(ntk_change)], ntk_change]),
               header="iteration  ntk_relative_change", fmt='%.6e')

    np.savetxt(os.path.join(DATA_DIR, 'weights_change.txt'),
               np.column_stack([iters_log[:len(weights_change_log)], weights_change_log]),
               header="iteration  weights_relative_change", fmt='%.6e')

    snapshot_iters = [0, len(lambda_K_log) // 4, len(lambda_K_log) // 2, len(lambda_K_log) - 1]
    for si in snapshot_iters:
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_K_iter{iters_log[si]}.txt'),
                   lambda_K_log[si], header=f"eigenvalues_K_at_iter_{iters_log[si]}", fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Kuu_iter{iters_log[si]}.txt'),
                   lambda_K_uu_log[si], header=f"eigenvalues_Kuu_at_iter_{iters_log[si]}", fmt='%.6e')
        np.savetxt(os.path.join(DATA_DIR, f'ntk_eig_Krr_iter{iters_log[si]}.txt'),
                   lambda_K_rr_log[si], header=f"eigenvalues_Krr_at_iter_{iters_log[si]}", fmt='%.6e')

    # ============================================================
    # Plotting
    # ============================================================
    plot_results(iters_log, loss_bcs_log, loss_res_log, l2_error_log,
                 X_test, U_test, u_pred_final,
                 lambda_K_log, lambda_K_uu_log, lambda_K_rr_log,
                 ntk_change, weights_change_log,
                 snapshot_iters, n_params, total_time, best_l2)

    print("\n" + "=" * 60)
    print("SUMMARY — Case 1: Poisson 1D (Normalized)")
    print("=" * 60)
    print(f"  Network:          {LAYERS}")
    print(f"  Parameters:       {n_params}")
    print(f"  Optimizer:        Adam (lr={LR})")
    print(f"  Normalization:    z-score (mu={MU_X:.4f}, sigma={SIGMA_X:.4f})")
    print(f"  Iterations:       {N_ITER}")
    print(f"  Best L2 error:    {best_l2:.3e}")
    print(f"  Final L2 error:   {l2_error_log[-1]:.3e}")
    print(f"  Training time:    {total_time:.1f}s")
    print("=" * 60)


# ============================================================
# Plotting functions (journal-quality)
# ============================================================
def _label_subplot(ax, label, x=-0.12, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='left')


def plot_results(iters_log, loss_bcs_log, loss_res_log, l2_error_log,
                 X_test, U_test, u_pred_final,
                 lambda_K_log, lambda_K_uu_log, lambda_K_rr_log,
                 ntk_change, weights_change_log,
                 snapshot_iters, n_params, total_time, best_l2):

    iters_arr = np.array(iters_log)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.semilogy(iters_arr, loss_res_log, lw=2, label=r'$\mathcal{L}_{r}$')
    ax.semilogy(iters_arr, loss_bcs_log, lw=2, label=r'$\mathcal{L}_{b}$')
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(a)')

    ax = axes[1]
    ax.semilogy(iters_arr, l2_error_log, lw=2, color='tab:red')
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative $L^2$ error', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(X_test, U_test, 'b-', lw=2, label='Exact')
    ax.plot(X_test, u_pred_final, 'r--', lw=2, label='Predicted')
    ax.set_xlabel('$x$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$u(x)$', fontsize=14, fontweight='bold')
    ax.legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(a)')

    ax = axes[1]
    ax.semilogy(X_test, np.abs(U_test - u_pred_final), lw=2, color='tab:green')
    ax.set_xlabel('$x$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Point-wise absolute error', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(b)')

    ax = axes[2]
    ax.plot(X_test, U_test - u_pred_final, lw=2, color='tab:purple')
    ax.set_xlabel('$x$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Point-wise error', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(c)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_prediction.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [r'Eigenvalues of $K$', r'Eigenvalues of $K_{uu}$', r'Eigenvalues of $K_{rr}$']
    data_lists = [lambda_K_log, lambda_K_uu_log, lambda_K_rr_log]

    for col, (ax, title, eig_list) in enumerate(zip(axes, titles, data_lists)):
        for si in snapshot_iters:
            ax.loglog(np.arange(1, len(eig_list[si]) + 1),
                      np.clip(eig_list[si], 1e-30, None),
                      lw=2, marker='', label=f'$n={iters_log[si]}$')
        ax.set_xlabel('Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Eigenvalue', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.tick_params(labelsize=12)
        _label_subplot(ax, f'({"abc"[col]})')

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(snapshot_iters),
               fontsize=13, frameon=True, fancybox=False, edgecolor='black',
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(FIG_DIR, 'fig3_ntk_eigenvalues.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(iters_arr[:len(ntk_change)], ntk_change, lw=2, color='tab:blue')
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\|K - K_0\| / \|K_0\|$', fontsize=14, fontweight='bold')
    ax.set_title('NTK relative change', fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(a)')

    ax = axes[1]
    ax.plot(iters_arr[:len(weights_change_log)], weights_change_log, lw=2, color='tab:orange')
    ax.set_xlabel('Iterations', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\|\theta - \theta_0\| / \|\theta_0\|$', fontsize=14, fontweight='bold')
    ax.set_title('Weights relative change', fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=12)
    _label_subplot(ax, '(b)')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_ntk_weights_change.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nAll figures saved to {FIG_DIR}")
    print(f"All data saved to {DATA_DIR}")


if __name__ == '__main__':
    train()
