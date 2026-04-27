"""
RFM Parameter Study: 2D Stokes Flow on Holed Square — JAX
==========================================================
Random Feature Method (RFM) for the Stokes equations on a perforated
square domain.  This script performs systematic parameter sweeps and
generates publication-quality comparison figures.

PDE:
    -Delta u + dp/dx = f1    in Omega
    -Delta v + dp/dy = f2    in Omega
    du/dx + dv/dy    = 0     in Omega

Domain:
    Omega = (0,1)^2  minus 3 circles at (0.5,0.2), (0.2,0.8), (0.8,0.8)
    with radius r = 0.1

Boundary conditions:
    u = u_exact, v = v_exact  on dOmega
    p(0,0) = -4/3             (pressure pinning)

Exact (polynomial) solution:
    u = x + x^2 - 2xy + x^3 - 3xy^2 + x^2 y
    v = -y - 2xy + y^2 - 3x^2 y + y^3 - x y^2
    p = xy + x + y + x^3 y^2 - 4/3

Parameter sweeps
----------------
  sweep_Q        — Collocation points:  Q in {200,400,600,800,1000}
  sweep_nhidden  — Random features:     n_hidden in {100,200,400,600,800}
  sweep_nsub     — Subdomains:          n_sub in {1,4,9}
  sweep_seed     — Seed stability:      seed in {42,100,200,300,400}

Reference:
    Chen, Y. & Bhatt, A. (2024).  Random Feature Method for Solving PDEs.

Run:
    python stokes_2d_rfm.py                      # full parameter study
    python stokes_2d_rfm.py --quick               # quick test (small grids)
    python stokes_2d_rfm.py --plot_only            # regenerate plots only
    python stokes_2d_rfm.py --sweeps sweep_Q sweep_nhidden  # selected sweeps
"""

import os, sys, time, json, argparse, math
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

# ============================================================
# Paths
# ============================================================
WORKDIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKDIR, 'data')
FIG_DIR = os.path.join(WORKDIR, 'figures')
CKPT_DIR = os.path.join(WORKDIR, 'checkpoints')
for _d in [DATA_DIR, FIG_DIR, CKPT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================
# Exact solution & source terms
# ============================================================
def exact_u(x):
    x0, x1 = x[:, 0:1], x[:, 1:2]
    return x0 + x0**2 - 2*x0*x1 + x0**3 - 3*x0*x1**2 + x0**2*x1

def exact_v(x):
    x0, x1 = x[:, 0:1], x[:, 1:2]
    return -x1 - 2*x0*x1 + x1**2 - 3*x0**2*x1 + x1**3 - x0*x1**2

def exact_p(x):
    x0, x1 = x[:, 0:1], x[:, 1:2]
    return x0*x1 + x0 + x1 + x0**3*x1**2 - 4.0/3.0

def source_f1(x):
    x0, x1 = x[:, 0:1], x[:, 1:2]
    return 3*x0**2*x1**2 - x1 - 1

def source_f2(x):
    x0, x1 = x[:, 0:1], x[:, 1:2]
    return 2*x0**3*x1 + 3*x0 - 1


# ============================================================
# RFM building blocks
# ============================================================
def tanh_forward(x, c, r, W, b):
    return jnp.tanh((x - c) / r @ W + b)

def tanh_d1(x, c, r, W, b, axis):
    t = jnp.tanh((x - c) / r @ W + b)
    return (1 - t**2) * (W[axis, :] / r[0, axis])

def tanh_d2(x, c, r, W, b, a1, a2):
    t = jnp.tanh((x - c) / r @ W + b)
    return -2*t*(1 - t**2) * (W[a1, :] / r[0, a1]) * (W[a2, :] / r[0, a2])


# ============================================================
# Partition of Unity (PoU) — smooth bump functions
# ============================================================
def psi_b_1d(xn):
    return jnp.where(xn < -1.25, 0.0,
           jnp.where(xn < -0.75, 0.5*(1 + jnp.sin(2*jnp.pi*xn)),
           jnp.where(xn <= 0.75, 1.0,
           jnp.where(xn <= 1.25, 0.5*(1 - jnp.sin(2*jnp.pi*xn)), 0.0))))

def psi_b_d1_1d(xn):
    return jnp.where(xn < -1.25, 0.0,
           jnp.where(xn < -0.75, jnp.pi*jnp.cos(2*jnp.pi*xn),
           jnp.where(xn <= 0.75, 0.0,
           jnp.where(xn <= 1.25, -jnp.pi*jnp.cos(2*jnp.pi*xn), 0.0))))

def psi_b_d2_1d(xn):
    return jnp.where(xn < -1.25, 0.0,
           jnp.where(xn < -0.75, -2*jnp.pi**2*jnp.sin(2*jnp.pi*xn),
           jnp.where(xn <= 0.75, 0.0,
           jnp.where(xn <= 1.25, 2*jnp.pi**2*jnp.sin(2*jnp.pi*xn), 0.0))))

def psi_b_2d(x, c, r):
    xn = (x - c) / r
    return psi_b_1d(xn[:, 0:1]) * psi_b_1d(xn[:, 1:2])

def psi_b_2d_d1(x, c, r, ax):
    xn = (x - c) / r
    if ax == 0:
        return psi_b_d1_1d(xn[:, 0:1]) / r[0, 0] * psi_b_1d(xn[:, 1:2])
    return psi_b_1d(xn[:, 0:1]) * psi_b_d1_1d(xn[:, 1:2]) / r[0, 1]

def psi_b_2d_d2(x, c, r, a1, a2):
    xn = (x - c) / r
    if a1 == a2 == 0:
        return psi_b_d2_1d(xn[:, 0:1]) / r[0, 0]**2 * psi_b_1d(xn[:, 1:2])
    if a1 == a2 == 1:
        return psi_b_1d(xn[:, 0:1]) * psi_b_d2_1d(xn[:, 1:2]) / r[0, 1]**2
    return psi_b_d1_1d(xn[:, 0:1]) / r[0, 0] * psi_b_d1_1d(xn[:, 1:2]) / r[0, 1]

def pou_2d(x, centers, radii):
    n = centers.shape[0]
    raw = jnp.concatenate([psi_b_2d(x, centers[i:i+1], radii[i:i+1])
                           for i in range(n)], 1)
    s = jnp.sum(raw, 1, keepdims=True)
    ss = jnp.where(s != 0, s, 1.0)
    return jnp.where(s != 0, raw / ss, 0.0)

def pou_d1_2d(x, centers, radii, axis):
    n = centers.shape[0]
    raw = jnp.concatenate([psi_b_2d(x, centers[i:i+1], radii[i:i+1])
                           for i in range(n)], 1)
    dr = jnp.concatenate([psi_b_2d_d1(x, centers[i:i+1], radii[i:i+1], axis)
                          for i in range(n)], 1)
    s = jnp.sum(raw, 1, keepdims=True)
    ds = jnp.sum(dr, 1, keepdims=True)
    ss = jnp.where(s != 0, s, 1.0)
    return jnp.where(s != 0, (dr - raw * ds / ss) / ss, 0.0)

def pou_d2_2d(x, centers, radii, a1, a2):
    n = centers.shape[0]
    raw = jnp.concatenate([psi_b_2d(x, centers[i:i+1], radii[i:i+1])
                           for i in range(n)], 1)
    dr1 = jnp.concatenate([psi_b_2d_d1(x, centers[i:i+1], radii[i:i+1], a1)
                           for i in range(n)], 1)
    dr2 = jnp.concatenate([psi_b_2d_d1(x, centers[i:i+1], radii[i:i+1], a2)
                           for i in range(n)], 1)
    d2r = jnp.concatenate([psi_b_2d_d2(x, centers[i:i+1], radii[i:i+1], a1, a2)
                           for i in range(n)], 1)
    s = jnp.sum(raw, 1, keepdims=True)
    ds1 = jnp.sum(dr1, 1, keepdims=True)
    ds2 = jnp.sum(dr2, 1, keepdims=True)
    d2s = jnp.sum(d2r, 1, keepdims=True)
    ss = jnp.where(s != 0, s, 1.0)
    i1 = 1/ss; i2 = i1*i1; i3 = i2*i1
    return jnp.where(s != 0,
                     d2r*i1 - 2*dr1*ds2*i2 - raw*d2s*i2 + 2*raw*ds1*ds2*i3,
                     0.0)


# ============================================================
# Feature assembly
# ============================================================
def feat(x, centers, radii, rf_list, pou_val,
         pou_d1d=None, pou_d2v=None, order=0, a1=None, a2=None):
    n = centers.shape[0]
    blocks = []
    for i in range(n):
        c, r = centers[i:i+1], radii[i:i+1]
        W, b = rf_list[i]
        pi = pou_val[:, i:i+1]
        if order == 0:
            blocks.append(tanh_forward(x, c, r, W, b) * pi)
        elif order == 1:
            rf = tanh_forward(x, c, r, W, b)
            rf1 = tanh_d1(x, c, r, W, b, a1)
            pd1 = pou_d1d[a1][:, i:i+1]
            blocks.append(rf * pd1 + rf1 * pi)
        elif order == 2:
            rf = tanh_forward(x, c, r, W, b)
            rf1_a1 = tanh_d1(x, c, r, W, b, a1)
            rf1_a2 = tanh_d1(x, c, r, W, b, a2) if a1 != a2 else rf1_a1
            rf2 = tanh_d2(x, c, r, W, b, a1, a2)
            pd1_1 = pou_d1d[a1][:, i:i+1]
            pd1_2 = pou_d1d[a2][:, i:i+1] if a1 != a2 else pd1_1
            pd2 = pou_d2v[:, i:i+1]
            blocks.append(rf*pd2 + rf2*pi + rf1_a1*pd1_2 + rf1_a2*pd1_1)
    return jnp.concatenate(blocks, 1)

def solve_ls(A, b):
    An = jnp.linalg.norm(A, axis=1, keepdims=True)
    An = jnp.where(An > 0, An, 1.0)
    As, bs = A / An, b / An
    w, _, _, _ = jnp.linalg.lstsq(As, bs, rcond=None)
    res = float(jnp.linalg.norm(As @ w - bs) / jnp.linalg.norm(bs))
    return w, res


# ============================================================
# Geometry: square minus 3 circles
# ============================================================
BBOX = [0.0, 1.0, 0.0, 1.0]
CIRCLE_CENTERS = [(0.5, 0.2), (0.2, 0.8), (0.8, 0.8)]
CIRCLE_RADII = [0.1, 0.1, 0.1]

def in_domain(x):
    x0, x1 = x[:, 0], x[:, 1]
    inside_sq = (x0 >= BBOX[0]) & (x0 <= BBOX[1]) & (x1 >= BBOX[2]) & (x1 <= BBOX[3])
    outside_circles = jnp.ones(x.shape[0], dtype=bool)
    for cc, cr in zip(CIRCLE_CENTERS, CIRCLE_RADII):
        dist = jnp.sqrt((x0 - cc[0])**2 + (x1 - cc[1])**2)
        outside_circles = outside_circles & (dist > cr)
    return inside_sq & outside_circles

def sample_interior(n_target):
    n_per_dim = int(math.ceil(math.sqrt(n_target * 2)))
    xs = jnp.linspace(BBOX[0], BBOX[1], n_per_dim + 2)[1:-1]
    ys = jnp.linspace(BBOX[2], BBOX[3], n_per_dim + 2)[1:-1]
    xx, yy = jnp.meshgrid(xs, ys, indexing='ij')
    pts = jnp.stack([xx.ravel(), yy.ravel()], 1)
    mask = in_domain(pts)
    return pts[mask]

def sample_boundary(n_total):
    n_sq = n_total // 2
    nps = max(n_sq // 4, 5)
    x0, x1, y0, y1 = BBOX
    bd = jnp.stack([jnp.linspace(x0, x1, nps), jnp.full(nps, y0)], 1)
    tp = jnp.stack([jnp.linspace(x0, x1, nps), jnp.full(nps, y1)], 1)
    lt = jnp.stack([jnp.full(nps, x0), jnp.linspace(y0, y1, nps)], 1)
    rt = jnp.stack([jnp.full(nps, x1), jnp.linspace(y0, y1, nps)], 1)
    sq_pts = jnp.concatenate([bd, tp, lt, rt], 0)

    n_circ = n_total - sq_pts.shape[0]
    n_per_circ = max(n_circ // len(CIRCLE_CENTERS), 10)
    circ_pts = []
    for cc, cr in zip(CIRCLE_CENTERS, CIRCLE_RADII):
        theta = jnp.linspace(0, 2*jnp.pi, n_per_circ, endpoint=False)
        pts = jnp.stack([cc[0] + cr*jnp.cos(theta), cc[1] + cr*jnp.sin(theta)], 1)
        circ_pts.append(pts)
    circ_pts = jnp.concatenate(circ_pts, 0)
    return jnp.concatenate([sq_pts, circ_pts], 0)


# ============================================================
# Subdomain layout generator
# ============================================================
def make_subdomain_layout(n_sub):
    """Generate centers and radii for an n_sub-subdomain grid.
    n_sub must be a perfect square (1, 4, 9, ...).
    Returns (centers, radii) as jnp arrays of shape (n_sub, 2).
    Overlap factor 1.5 ensures smooth PoU coverage.
    """
    k = int(math.sqrt(n_sub))
    assert k * k == n_sub, f"n_sub={n_sub} must be a perfect square"
    h = 1.0 / k
    overlap = 1.5
    cx = jnp.linspace(h/2, 1.0 - h/2, k)
    cy = jnp.linspace(h/2, 1.0 - h/2, k)
    cxx, cyy = jnp.meshgrid(cx, cy, indexing='ij')
    centers = jnp.stack([cxx.ravel(), cyy.ravel()], 1)
    radii = jnp.full_like(centers, h/2 * overlap)
    return centers, radii


# ============================================================
# Core solver
# ============================================================
def run_rfm(Q=800, n_hidden=400, n_sub=1, n_boundary=400, seed=100):
    """Solve the 2D Stokes problem with RFM.

    Returns dict with errors, timing, weights, and field data for plotting.
    """
    start = time.time()

    centers, radii = make_subdomain_layout(n_sub)

    key = random.PRNGKey(seed)
    rf_list = []
    for _ in range(n_sub):
        key, k1, k2 = random.split(key, 3)
        W = random.uniform(k1, (2, n_hidden), minval=-1.0, maxval=1.0)
        b = random.uniform(k2, (1, n_hidden), minval=-1.0, maxval=1.0)
        rf_list.append((W, b))

    M_total = n_sub * n_hidden

    x_in = sample_interior(Q)
    x_on = sample_boundary(n_boundary)
    x_corner = jnp.array([[0.0, 0.0]])

    # --- Interior PDE features ---
    pou_in = pou_2d(x_in, centers, radii)
    pd1_in = {0: pou_d1_2d(x_in, centers, radii, 0),
              1: pou_d1_2d(x_in, centers, radii, 1)}
    pd2_xx = pou_d2_2d(x_in, centers, radii, 0, 0)
    pd2_yy = pou_d2_2d(x_in, centers, radii, 1, 1)

    u_xx = feat(x_in, centers, radii, rf_list, pou_in, pd1_in, pd2_xx, 2, 0, 0)
    u_yy = feat(x_in, centers, radii, rf_list, pou_in, pd1_in, pd2_yy, 2, 1, 1)
    u_x  = feat(x_in, centers, radii, rf_list, pou_in, pd1_in, None, 1, 0, None)
    u_y  = feat(x_in, centers, radii, rf_list, pou_in, pd1_in, None, 1, 1, None)

    Z = jnp.zeros_like(u_xx)

    # Stokes block system: [u-block, v-block, p-block]
    A1 = jnp.concatenate([
        jnp.concatenate([-(u_xx + u_yy), Z, u_x], axis=1),
        jnp.concatenate([Z, -(u_xx + u_yy), u_y], axis=1),
        jnp.concatenate([u_x, u_y, Z], axis=1),
    ], axis=0)

    f1 = source_f1(x_in)
    f2 = source_f2(x_in)
    b1 = jnp.concatenate([f1, f2, jnp.zeros_like(f1)], axis=0)

    # --- Boundary features ---
    pou_on = pou_2d(x_on, centers, radii)
    u_on = feat(x_on, centers, radii, rf_list, pou_on, order=0)
    Z_on = jnp.zeros_like(u_on)

    A2 = jnp.concatenate([
        jnp.concatenate([u_on, Z_on, Z_on], axis=1),
        jnp.concatenate([Z_on, u_on, Z_on], axis=1),
    ], axis=0)
    b2 = jnp.concatenate([exact_u(x_on), exact_v(x_on)], axis=0)

    # --- Pressure pinning at corner ---
    pou_c = pou_2d(x_corner, centers, radii)
    u_c = feat(x_corner, centers, radii, rf_list, pou_c, order=0)
    Z_c = jnp.zeros_like(u_c)
    A3 = jnp.concatenate([Z_c, Z_c, u_c], axis=1)
    b3 = jnp.array([[-4.0 / 3.0]])

    A_full = jnp.concatenate([A1, A2, A3], axis=0)
    b_full = jnp.concatenate([b1, b2, b3], axis=0)
    N = A_full.shape[0]

    w, ls_res = solve_ls(A_full, b_full)
    elapsed = time.time() - start

    # --- Evaluate on interior points ---
    w_u = w[:M_total]
    w_v = w[M_total:2*M_total]
    w_p = w[2*M_total:]

    pou_t = pou_2d(x_in, centers, radii)
    A_t = feat(x_in, centers, radii, rf_list, pou_t, order=0)
    u_p = A_t @ w_u; v_p = A_t @ w_v; p_p = A_t @ w_p
    u_e = exact_u(x_in); v_e = exact_v(x_in); p_e = exact_p(x_in)

    err_u = float(jnp.linalg.norm(u_p - u_e) / jnp.linalg.norm(u_e))
    err_v = float(jnp.linalg.norm(v_p - v_e) / jnp.linalg.norm(v_e))
    err_p = float(jnp.linalg.norm(p_p - p_e) / jnp.linalg.norm(p_e))

    # --- Fine grid for plotting ---
    nf = 100
    xf = jnp.linspace(0, 1, nf)
    yf = jnp.linspace(0, 1, nf)
    xxf, yyf = jnp.meshgrid(xf, yf, indexing='ij')
    x_fine = jnp.stack([xxf.ravel(), yyf.ravel()], 1)
    mask = in_domain(x_fine)

    pou_f = pou_2d(x_fine, centers, radii)
    A_f = feat(x_fine, centers, radii, rf_list, pou_f, order=0)

    u_f = np.array(A_f @ w_u).reshape(nf, nf)
    v_f = np.array(A_f @ w_v).reshape(nf, nf)
    p_f = np.array(A_f @ w_p).reshape(nf, nf)

    u_ef = np.array(exact_u(x_fine)).reshape(nf, nf)
    v_ef = np.array(exact_v(x_fine)).reshape(nf, nf)
    p_ef = np.array(exact_p(x_fine)).reshape(nf, nf)

    mask_2d = np.array(mask).reshape(nf, nf)

    return {
        "Q": Q, "n_hidden": n_hidden, "n_sub": n_sub, "seed": seed,
        "N_eq": int(N), "M_total": 3*M_total,
        "n_interior": int(x_in.shape[0]), "n_boundary": int(x_on.shape[0]),
        "err_u": err_u, "err_v": err_v, "err_p": err_p,
        "ls_residual": ls_res, "elapsed": elapsed,
        "w": np.array(w),
        "xf": np.array(xf), "yf": np.array(yf),
        "u_f": u_f, "v_f": v_f, "p_f": p_f,
        "u_ef": u_ef, "v_ef": v_ef, "p_ef": p_ef,
        "mask_2d": mask_2d,
    }


# ============================================================
# Sweep configurations
# ============================================================
SWEEP_CONFIGS = {
    "sweep_Q": {
        "description": "Collocation point density sweep",
        "param": "Q",
        "values": [200, 400, 600, 800, 1000],
        "defaults": {"n_hidden": 400, "n_sub": 1, "seed": 100},
    },
    "sweep_nhidden": {
        "description": "Random feature width sweep",
        "param": "n_hidden",
        "values": [100, 200, 400, 600, 800],
        "defaults": {"Q": 800, "n_sub": 1, "seed": 100},
    },
    "sweep_nsub": {
        "description": "Subdomain count sweep",
        "param": "n_sub",
        "values": [1, 4, 9],
        "defaults": {"Q": 800, "n_hidden": 400, "seed": 100},
    },
    "sweep_seed": {
        "description": "Random seed stability",
        "param": "seed",
        "values": [42, 100, 200, 300, 400],
        "defaults": {"Q": 800, "n_hidden": 400, "n_sub": 1},
    },
}

SWEEP_ORDER = ["sweep_Q", "sweep_nhidden", "sweep_nsub", "sweep_seed"]

QUICK_CONFIGS = {
    "sweep_Q": {
        "description": "Collocation point density sweep (quick)",
        "param": "Q",
        "values": [100, 200, 400],
        "defaults": {"n_hidden": 200, "n_sub": 1, "seed": 100},
    },
    "sweep_nhidden": {
        "description": "Random feature width sweep (quick)",
        "param": "n_hidden",
        "values": [50, 100, 200],
        "defaults": {"Q": 400, "n_sub": 1, "seed": 100},
    },
    "sweep_nsub": {
        "description": "Subdomain count sweep (quick)",
        "param": "n_sub",
        "values": [1, 4],
        "defaults": {"Q": 400, "n_hidden": 200, "seed": 100},
    },
    "sweep_seed": {
        "description": "Random seed stability (quick)",
        "param": "seed",
        "values": [42, 100, 200],
        "defaults": {"Q": 400, "n_hidden": 200, "n_sub": 1},
    },
}


# ============================================================
# Run sweeps
# ============================================================
def run_sweep(sweep_name, cfg, verbose=True):
    """Run a single parameter sweep and return list of result dicts."""
    param = cfg["param"]
    values = cfg["values"]
    defaults = cfg["defaults"]

    results = []
    for val in values:
        kwargs = dict(defaults)
        kwargs[param] = val
        if verbose:
            print(f"  [{sweep_name}] {param}={val}  (defaults: {defaults})")
        r = run_rfm(**kwargs)
        results.append(r)
        if verbose:
            print(f"    err_u={r['err_u']:.4e}  err_v={r['err_v']:.4e}  "
                  f"err_p={r['err_p']:.4e}  time={r['elapsed']:.2f}s")
    return results


def save_sweep_data(sweep_name, results):
    """Save sweep results to data/ directory."""
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items()
                     if not isinstance(v, np.ndarray)})

    with open(os.path.join(DATA_DIR, f'{sweep_name}.json'), 'w') as f:
        json.dump(rows, f, indent=2)

    lines = [f"{'Q':>6} {'n_hid':>6} {'n_sub':>5} {'seed':>5} "
             f"{'err_u':>12} {'err_v':>12} {'err_p':>12} "
             f"{'LS_res':>12} {'time(s)':>8}"]
    lines.append("-" * 90)
    for r in results:
        lines.append(f"{r['Q']:>6d} {r['n_hidden']:>6d} {r['n_sub']:>5d} "
                     f"{r['seed']:>5d} {r['err_u']:>12.4e} {r['err_v']:>12.4e} "
                     f"{r['err_p']:>12.4e} {r['ls_residual']:>12.4e} "
                     f"{r['elapsed']:>8.2f}")
    txt = '\n'.join(lines)
    with open(os.path.join(DATA_DIR, f'{sweep_name}.txt'), 'w') as f:
        f.write(txt + '\n')
    print(f"  Saved: data/{sweep_name}.txt")


def save_best_checkpoint(all_sweep_results):
    """Save weights from the run with lowest err_u across all sweeps."""
    best = None
    for results in all_sweep_results.values():
        for r in results:
            if best is None or r['err_u'] < best['err_u']:
                best = r
    if best is not None:
        np.savez(os.path.join(CKPT_DIR, 'best_weights.npz'), w=best['w'])
        np.savez(os.path.join(CKPT_DIR, 'best_fields.npz'),
                 xf=best['xf'], yf=best['yf'],
                 u_f=best['u_f'], v_f=best['v_f'], p_f=best['p_f'],
                 u_ef=best['u_ef'], v_ef=best['v_ef'], p_ef=best['p_ef'],
                 mask_2d=best['mask_2d'])
        info = {k: v for k, v in best.items() if not isinstance(v, np.ndarray)}
        with open(os.path.join(CKPT_DIR, 'best_config.json'), 'w') as f:
            json.dump(info, f, indent=2)
        print(f"  Best config: Q={best['Q']}, n_hidden={best['n_hidden']}, "
              f"n_sub={best['n_sub']}, err_u={best['err_u']:.4e}")


def write_comparison_summary(all_sweep_results):
    """Write unified comparison summary across all sweeps."""
    lines = ["=" * 95]
    lines.append("RFM STOKES 2D — PARAMETER STUDY SUMMARY")
    lines.append("=" * 95)

    for sname in SWEEP_ORDER:
        if sname not in all_sweep_results:
            continue
        results = all_sweep_results[sname]
        cfg = SWEEP_CONFIGS.get(sname, QUICK_CONFIGS.get(sname, {}))
        lines.append(f"\n--- {cfg.get('description', sname)} ---")
        lines.append(f"  Swept: {cfg.get('param','?')}  |  "
                     f"Defaults: {cfg.get('defaults','?')}")
        lines.append(f"  {'Value':>8} {'err_u':>12} {'err_v':>12} "
                     f"{'err_p':>12} {'time(s)':>8}")
        lines.append("  " + "-" * 60)
        for r in results:
            val = r[cfg['param']]
            lines.append(f"  {val:>8} {r['err_u']:>12.4e} {r['err_v']:>12.4e} "
                         f"{r['err_p']:>12.4e} {r['elapsed']:>8.2f}")

    lines.append("\n" + "=" * 95)
    txt = '\n'.join(lines)
    print('\n' + txt)

    with open(os.path.join(DATA_DIR, 'comparison_summary.txt'), 'w') as f:
        f.write(txt + '\n')
    print(f"  Saved: data/comparison_summary.txt")

    all_rows = []
    for sname in SWEEP_ORDER:
        if sname not in all_sweep_results:
            continue
        for r in all_sweep_results[sname]:
            row = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            row['sweep'] = sname
            all_rows.append(row)
    with open(os.path.join(DATA_DIR, 'comparison_summary.json'), 'w') as f:
        json.dump(all_rows, f, indent=2)


# ============================================================
# Plot style — journal quality (English, Times New Roman)
# ============================================================
def setup_plot_style():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman'] + rcParams.get('font.serif', ['DejaVu Serif'])
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.size'] = 14
    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 16
    rcParams['axes.linewidth'] = 2.0
    rcParams['axes.labelweight'] = 'bold'
    rcParams['xtick.labelsize'] = 13
    rcParams['ytick.labelsize'] = 13
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['legend.fontsize'] = 12
    rcParams['legend.framealpha'] = 0.9
    rcParams['figure.dpi'] = 100
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'


def _add_circles(ax, plt):
    for cc, cr in zip(CIRCLE_CENTERS, CIRCLE_RADII):
        circle = plt.Circle(cc, cr, fill=True, color='white',
                            ec='black', lw=1.5, zorder=5)
        ax.add_patch(circle)


# ============================================================
# Figure 1: Solution fields (3x3 grid)
# ============================================================
def plot_solution_fields(filepath, sweep_results=None):
    setup_plot_style()
    import matplotlib.pyplot as plt

    best = _load_best_fields()
    if best is None and sweep_results:
        best_r = None
        for results in sweep_results.values():
            for r in results:
                if best_r is None or r['err_u'] < best_r['err_u']:
                    best_r = r
        best = best_r
    if best is None:
        print("  Skipping solution fields plot (no data).")
        return

    xf, yf = best['xf'], best['yf']
    mask = best['mask_2d']

    fields = [
        ('u', best['u_ef'], best['u_f']),
        ('v', best['v_ef'], best['v_f']),
        ('p', best['p_ef'], best['p_f']),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))

    for row, (name, ref, pred) in enumerate(fields):
        err = np.abs(ref - pred)
        ref_m = np.where(mask, ref, np.nan)
        pred_m = np.where(mask, pred, np.nan)
        err_m = np.where(mask, err, np.nan)

        vmin = np.nanmin(ref_m)
        vmax = np.nanmax(ref_m)

        for col, (data, title, cmap) in enumerate([
            (ref_m, f'Reference ${name}$', 'RdBu_r'),
            (pred_m, f'RFM ${name}$', 'RdBu_r'),
            (err_m, f'Abs. Error $|\\Delta {name}|$', 'hot_r'),
        ]):
            ax = axes[row, col]
            if col < 2:
                im = ax.pcolormesh(xf, yf, data.T, cmap=cmap,
                                   vmin=vmin, vmax=vmax, shading='auto')
            else:
                im = ax.pcolormesh(xf, yf, data.T, cmap=cmap, shading='auto')
            cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
            cb.ax.tick_params(labelsize=10, width=1.2)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_title(title, fontweight='bold')
            ax.set_aspect('equal')
            label = chr(97 + row * 3 + col)
            ax.text(0.02, 0.95, f'({label})', transform=ax.transAxes,
                    fontsize=15, fontweight='bold', va='top')
            for sp in ax.spines.values():
                sp.set_linewidth(2.0)
            _add_circles(ax, plt)

    fig.suptitle('RFM Solution Fields — 2D Stokes on Holed Square',
                 fontsize=18, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Figure 2: Convergence vs Q
# ============================================================
def plot_convergence_Q(filepath):
    setup_plot_style()
    import matplotlib.pyplot as plt

    data = _load_sweep_json('sweep_Q')
    if data is None or len(data) < 2:
        print("  Skipping Q convergence plot (no data).")
        return

    Qs = [d['Q'] for d in data]
    eu = [d['err_u'] for d in data]
    ev = [d['err_v'] for d in data]
    ep = [d['err_p'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(Qs, eu, 'o-', color='#1f77b4', lw=2.5, ms=8, label='$u$ velocity')
    ax.semilogy(Qs, ev, 's--', color='#ff7f0e', lw=2.5, ms=8, label='$v$ velocity')
    ax.semilogy(Qs, ep, 'D-.', color='#2ca02c', lw=2.5, ms=8, label='$p$ pressure')
    ax.set_xlabel('Number of Interior Points $Q$', fontweight='bold')
    ax.set_ylabel('Relative $L_2$ Error', fontweight='bold')
    ax.set_title('Convergence w.r.t. Collocation Points', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, which='both')
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Figure 3: Convergence vs n_hidden
# ============================================================
def plot_convergence_nhidden(filepath):
    setup_plot_style()
    import matplotlib.pyplot as plt

    data = _load_sweep_json('sweep_nhidden')
    if data is None or len(data) < 2:
        print("  Skipping n_hidden convergence plot (no data).")
        return

    nh = [d['n_hidden'] for d in data]
    eu = [d['err_u'] for d in data]
    ev = [d['err_v'] for d in data]
    ep = [d['err_p'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(nh, eu, 'o-', color='#1f77b4', lw=2.5, ms=8, label='$u$ velocity')
    ax.semilogy(nh, ev, 's--', color='#ff7f0e', lw=2.5, ms=8, label='$v$ velocity')
    ax.semilogy(nh, ep, 'D-.', color='#2ca02c', lw=2.5, ms=8, label='$p$ pressure')
    ax.set_xlabel('Number of Random Features $n_{\\mathrm{hidden}}$', fontweight='bold')
    ax.set_ylabel('Relative $L_2$ Error', fontweight='bold')
    ax.set_title('Convergence w.r.t. Random Feature Width', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, which='both')
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Figure 4: Subdomain comparison (grouped bar)
# ============================================================
def plot_subdomain_comparison(filepath):
    setup_plot_style()
    import matplotlib.pyplot as plt

    data = _load_sweep_json('sweep_nsub')
    if data is None or len(data) < 2:
        print("  Skipping subdomain comparison plot (no data).")
        return

    nsubs = [d['n_sub'] for d in data]
    eu = [d['err_u'] for d in data]
    ev = [d['err_v'] for d in data]
    ep = [d['err_p'] for d in data]

    x_pos = np.arange(len(nsubs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x_pos - width, eu, width, label='$u$ velocity',
                   color='#1f77b4', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x_pos, ev, width, label='$v$ velocity',
                   color='#ff7f0e', edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x_pos + width, ep, width, label='$p$ pressure',
                   color='#2ca02c', edgecolor='black', linewidth=1.2)

    ax.set_xticks(x_pos)
    nsub_labels = {1: '$1\\times1$', 4: '$2\\times2$', 9: '$3\\times3$'}
    ax.set_xticklabels([nsub_labels.get(n, str(n)) for n in nsubs])
    ax.set_xlabel('Subdomain Layout', fontweight='bold')
    ax.set_ylabel('Relative $L_2$ Error', fontweight='bold')
    ax.set_title('Effect of Subdomain Decomposition', fontweight='bold')
    ax.set_yscale('log')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y', which='both')
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h*1.3,
                    f'{h:.1e}', ha='center', va='bottom', fontsize=8, rotation=45)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Figure 5: Seed stability (box + scatter)
# ============================================================
def plot_seed_stability(filepath):
    setup_plot_style()
    import matplotlib.pyplot as plt

    data = _load_sweep_json('sweep_seed')
    if data is None or len(data) < 2:
        print("  Skipping seed stability plot (no data).")
        return

    seeds = [d['seed'] for d in data]
    eu = [d['err_u'] for d in data]
    ev = [d['err_v'] for d in data]
    ep = [d['err_p'] for d in data]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (vals, label, color) in enumerate([
        (eu, '$u$ velocity', '#1f77b4'),
        (ev, '$v$ velocity', '#ff7f0e'),
        (ep, '$p$ pressure', '#2ca02c'),
    ]):
        ax = axes[idx]
        bp = ax.boxplot([vals], patch_artist=True, widths=0.4,
                        boxprops=dict(facecolor=color, alpha=0.3, linewidth=2),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(markersize=8))
        ax.scatter(np.ones(len(vals)), vals, color=color, s=60,
                   zorder=5, edgecolor='black', linewidth=1)
        for i, (s, v) in enumerate(zip(seeds, vals)):
            ax.annotate(f'seed={s}', (1, v), textcoords="offset points",
                        xytext=(15, 0), fontsize=9, va='center')
        ax.set_ylabel('Relative $L_2$ Error', fontweight='bold')
        ax.set_title(label, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y', which='both')
        for sp in ax.spines.values():
            sp.set_linewidth(2.0)
        ax.text(0.02, 0.95, f'({chr(97+idx)})', transform=ax.transAxes,
                fontsize=15, fontweight='bold', va='top')

    fig.suptitle('Sensitivity to Random Feature Initialization',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Figure 6: Timing analysis
# ============================================================
def plot_timing(filepath):
    setup_plot_style()
    import matplotlib.pyplot as plt

    colors = {'sweep_Q': '#1f77b4', 'sweep_nhidden': '#ff7f0e',
              'sweep_nsub': '#2ca02c'}
    labels = {'sweep_Q': '$Q$ sweep', 'sweep_nhidden': '$n_{\\mathrm{hidden}}$ sweep',
              'sweep_nsub': '$n_{\\mathrm{sub}}$ sweep'}
    markers = {'sweep_Q': 'o', 'sweep_nhidden': 's', 'sweep_nsub': 'D'}

    fig, ax = plt.subplots(figsize=(9, 6))
    has_data = False

    for sname in ['sweep_Q', 'sweep_nhidden', 'sweep_nsub']:
        data = _load_sweep_json(sname)
        if data is None or len(data) < 2:
            continue
        has_data = True
        times = [d['elapsed'] for d in data]
        errs = [d['err_u'] for d in data]
        ax.semilogy(times, errs, marker=markers[sname], color=colors[sname],
                    lw=2.5, ms=9, label=labels[sname], linestyle='-')
        cfg_key = SWEEP_CONFIGS.get(sname, QUICK_CONFIGS.get(sname, {}))
        param = cfg_key.get('param', '?')
        for t, e, d in zip(times, errs, data):
            ax.annotate(f'{param}={d[param]}', (t, e),
                        textcoords="offset points", xytext=(5, 8),
                        fontsize=8, color=colors[sname])

    if not has_data:
        print("  Skipping timing plot (no data).")
        plt.close(fig)
        return

    ax.set_xlabel('Wall Time (s)', fontweight='bold')
    ax.set_ylabel('Relative $L_2$ Error ($u$)', fontweight='bold')
    ax.set_title('Accuracy vs. Computational Cost', fontweight='bold')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, which='both')
    for sp in ax.spines.values():
        sp.set_linewidth(2.0)

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Figure 7: Summary table image
# ============================================================
def plot_summary_table(filepath):
    setup_plot_style()
    import matplotlib.pyplot as plt

    all_data = []
    for sname in SWEEP_ORDER:
        data = _load_sweep_json(sname)
        if data is None:
            continue
        cfg = SWEEP_CONFIGS.get(sname, QUICK_CONFIGS.get(sname, {}))
        for d in data:
            all_data.append({
                'Sweep': cfg.get('description', sname)[:25],
                'Q': d['Q'], 'n_hidden': d['n_hidden'],
                'n_sub': d['n_sub'], 'seed': d['seed'],
                'err_u': f"{d['err_u']:.3e}",
                'err_v': f"{d['err_v']:.3e}",
                'err_p': f"{d['err_p']:.3e}",
                'Time(s)': f"{d['elapsed']:.2f}",
            })

    if not all_data:
        print("  Skipping summary table (no data).")
        return

    col_labels = list(all_data[0].keys())
    cell_text = [[str(row[k]) for k in col_labels] for row in all_data]

    fig, ax = plt.subplots(figsize=(18, 1.0 + 0.45 * len(all_data)))
    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#FAFAFA' if row % 2 == 0 else '#ECEFF1')
        cell.set_edgecolor('#B0BEC5')

    ax.set_title('RFM Parameter Study — 2D Stokes (Full Summary)',
                 fontsize=16, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================
# Data loading helpers
# ============================================================
def _load_sweep_json(sweep_name):
    path = os.path.join(DATA_DIR, f'{sweep_name}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def _load_best_fields():
    fp = os.path.join(CKPT_DIR, 'best_fields.npz')
    cp = os.path.join(CKPT_DIR, 'best_config.json')
    if not (os.path.exists(fp) and os.path.exists(cp)):
        return None
    d = np.load(fp)
    with open(cp, 'r') as f:
        cfg = json.load(f)
    result = dict(cfg)
    for k in d.files:
        result[k] = d[k]
    return result


# ============================================================
# Generate all figures
# ============================================================
def generate_all_plots(sweep_results=None):
    print("\n" + "=" * 72)
    print("Generating publication-quality figures ...")
    print("=" * 72)

    plot_solution_fields(os.path.join(FIG_DIR, 'fig_solution_fields.png'),
                         sweep_results)
    plot_convergence_Q(os.path.join(FIG_DIR, 'fig_convergence_Q.png'))
    plot_convergence_nhidden(os.path.join(FIG_DIR, 'fig_convergence_nhidden.png'))
    plot_subdomain_comparison(os.path.join(FIG_DIR, 'fig_subdomain_comparison.png'))
    plot_seed_stability(os.path.join(FIG_DIR, 'fig_seed_stability.png'))
    plot_timing(os.path.join(FIG_DIR, 'fig_timing.png'))
    plot_summary_table(os.path.join(FIG_DIR, 'fig_summary_table.png'))

    print("\nAll figures generated.\n")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='RFM Parameter Study — 2D Stokes on Holed Square')
    parser.add_argument('--sweeps', nargs='+', default=None,
                        choices=SWEEP_ORDER,
                        help='Which sweeps to run (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced parameters')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only regenerate plots from saved data')
    args = parser.parse_args()

    if args.plot_only:
        generate_all_plots()
        return

    sweep_names = args.sweeps or SWEEP_ORDER
    configs = QUICK_CONFIGS if args.quick else SWEEP_CONFIGS

    print("=" * 72)
    print("RFM Parameter Study — 2D Stokes Flow on Holed Square")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices:     {jax.devices()}")
    print(f"Sweeps:      {sweep_names}")
    print(f"Mode:        {'QUICK' if args.quick else 'FULL'}")
    print("=" * 72)

    all_sweep_results = {}
    for sname in sweep_names:
        cfg = configs[sname]
        print(f"\n--- {cfg['description']} ---")
        results = run_sweep(sname, cfg)
        save_sweep_data(sname, results)
        all_sweep_results[sname] = results

    save_best_checkpoint(all_sweep_results)
    write_comparison_summary(all_sweep_results)
    generate_all_plots(all_sweep_results)

    print("\nParameter study complete!")
    print(f"Results: {DATA_DIR}")
    print(f"Figures: {FIG_DIR}")


if __name__ == '__main__':
    main()
