"""Shared fair-comparison engine for the frequency-scenario experiments (E17/E18/E19).

A single 2D Poisson  -Lap u = f  substrate (manufactured exact solution), solved by
five methods under an IDENTICAL frequency prior and a MATCHED parameter budget:

  SVSNN       : separable spectral net, analytic 2nd derivative on a tensor grid
                (forward/residual ported verbatim from E12 run_E12.py svsnn_run,
                 residual switched to Poisson; non-homogeneous Dirichlet BC).
  FourierPINN : frozen multi-level Fourier features + tanh MLP   (case7 form)
  SIREN       : Fourier features + sine-activated MLP             (case7 form)
  SPINN       : separable Fourier-feature tanh branches, low rank (case7 form)
  PINN        : plain tanh MLP, NO frequency prior (floor)        (case7 form)

Baselines are sized to the SV-SNN parameter count (+-10%) via harness.choose_matched.
All frequency-aware methods get the same multi-level sampler around `w_char`.
Residual sign convention: r = u_xx + u_yy + f  (so f = -Lap u_exact), matching case7.

A "problem" is a dict:
  name      : str
  u_exact   : f(x, y) -> np array       (elementwise, numpy)
  source    : f(x, y) -> np array       (= -Lap u_exact, numpy)
  w_char    : float                     (frequency prior center/upper)
  domain    : (lo, hi)  (square; default (0,1))
  components: optional list of (k, amp) for per-component projected error
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from pyDOE import lhs

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import harness

# ---- global config (overridable by importing experiment before run) ----
EPOCHS = 8000
LR = 1e-3
EVAL_EVERY = 100
M = 6              # SV-SNN modes
K = 32             # SV-SNN freqs per axis per mode
NC = 100           # SV-SNN / grid resolution (tensor-grid collocation)
N_PDE = 4000       # scattered collocation for pointwise baselines
N_BC = 400         # boundary supervision points (shared by all methods)
N_TEST = 200       # test grid resolution
FF = 32            # Fourier-feature count per axis for baselines (matched search adjusts width)


def _l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cp(p):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(p) if hasattr(x, "size")))


def sfreqs(key, wc, k):
    """Multi-level frozen frequency sampler (low / characteristic / high)."""
    nl = k // 4; ncc = k // 2; nh = k - nl - ncc
    _, k2, k3 = jax.random.split(key, 3)
    return jnp.sort(jnp.concatenate([
        jnp.linspace(1.0, wc, nl),
        jnp.abs(jax.random.normal(k2, (ncc,)) * float(wc) * 0.3 + wc),
        jax.random.uniform(k3, (nh,), minval=wc * 0.5, maxval=wc)]))


# ============================ data ============================
def build_data(problem, seed):
    lo, hi = problem.get("domain", (0.0, 1.0))
    L = hi - lo
    ue = problem["u_exact"]; src = problem["source"]
    np.random.seed(seed)
    nps = N_BC // 4
    t = (lo + L * np.linspace(0, 1, nps)).reshape(-1, 1)
    xb = np.vstack([np.full((nps, 1), lo), np.full((nps, 1), hi), t, t])
    yb = np.vstack([t, t, np.full((nps, 1), lo), np.full((nps, 1), hi)])
    ub = ue(xb, yb)
    pde = lo + L * lhs(2, samples=N_PDE)
    xp, yp = pde[:, 0:1], pde[:, 1:2]
    fp = src(xp, yp)
    x1 = lo + L * np.linspace(0, 1, N_TEST); y1 = x1.copy()
    X, Y = np.meshgrid(x1, y1, indexing="ij")
    UE = ue(X, Y)
    xc = (lo + L * np.linspace(0, 1, NC)).reshape(-1, 1)
    yc = xc.copy()
    Xc, Yc = np.meshgrid(xc.squeeze(), yc.squeeze(), indexing="ij")
    fg = src(Xc, Yc)
    return {
        "xb": jnp.asarray(xb, jnp.float32), "yb": jnp.asarray(yb, jnp.float32),
        "ub": jnp.asarray(ub, jnp.float32),
        "xp": jnp.asarray(xp, jnp.float32), "yp": jnp.asarray(yp, jnp.float32),
        "fp": jnp.asarray(fp, jnp.float32),
        "xc": jnp.asarray(xc, jnp.float32), "yc": jnp.asarray(yc, jnp.float32),
        "fg": jnp.asarray(fg, jnp.float32),
        "x1d": jnp.asarray(x1.reshape(-1, 1), jnp.float32),
        "y1d": jnp.asarray(y1.reshape(-1, 1), jnp.float32),
        "UE": UE, "X": X, "Y": Y,
    }


def _hvp(f, x, v):
    g = lambda z: jax.jvp(f, (z,), (v,))[1]
    return jax.jvp(g, (x,), (v,))[1]


def _per_component_error(u_pred, problem, data):
    comps = problem.get("components")
    if not comps:
        return None
    X, Y = data["X"], data["Y"]
    errs = {}
    for (k, amp) in comps:
        phi = np.sin(k * X) * np.sin(k * Y)
        c = float(np.sum(u_pred * phi) / np.sum(phi * phi))
        errs[f"k_{k:.2f}"] = {"amp_true": float(amp), "amp_fit": c,
                              "rel_err": float(abs(c - amp) / abs(amp))}
    return errs


# ============================ SV-SNN ============================
def run_svsnn(problem, seed, wc, epochs=None, return_pred=False):
    epochs = epochs or EPOCHS
    data = build_data(problem, seed)

    def init(key):
        keys = jax.random.split(key, M * 6 + 1); ki = 0; sx, sy = [], []
        for _ in range(M):
            sx.append({"freqs": sfreqs(keys[ki], wc, K),
                       "cos_c": jax.random.normal(keys[ki + 1], (K,)) * 0.1,
                       "sin_c": jax.random.normal(keys[ki + 2], (K,)) * 0.1,
                       "bias": jnp.zeros(1)}); ki += 3
            sy.append({"freqs": sfreqs(keys[ki], wc, K),
                       "cos_c": jax.random.normal(keys[ki + 1], (K,)) * 0.1,
                       "sin_c": jax.random.normal(keys[ki + 2], (K,)) * 0.1,
                       "bias": jnp.zeros(1)}); ki += 3
        return {"spatial_x": sx, "spatial_y": sy,
                "mode_coeffs": jax.random.normal(keys[ki], (M,)) * 0.1}

    def st(p, axis):
        f = jnp.stack([jax.lax.stop_gradient(p[axis][n]["freqs"]) for n in range(M)])
        c = jnp.stack([p[axis][n]["cos_c"] for n in range(M)])
        s = jnp.stack([p[axis][n]["sin_c"] for n in range(M)])
        b = jnp.stack([p[axis][n]["bias"] for n in range(M)])
        return f, c, s, b

    def fwd(p, x, y):
        u = jnp.zeros_like(x)
        for n in range(M):
            wx = p["spatial_x"][n]["freqs"][None, :] * x
            Xn = jnp.sum(p["spatial_x"][n]["cos_c"] * jnp.cos(wx) + p["spatial_x"][n]["sin_c"] * jnp.sin(wx), axis=1, keepdims=True) + p["spatial_x"][n]["bias"]
            wy = p["spatial_y"][n]["freqs"][None, :] * y
            Yn = jnp.sum(p["spatial_y"][n]["cos_c"] * jnp.cos(wy) + p["spatial_y"][n]["sin_c"] * jnp.sin(wy), axis=1, keepdims=True) + p["spatial_y"][n]["bias"]
            u = u + p["mode_coeffs"][n] * Xn * Yn
        return u

    def grid(p, x1d, y1d):
        fx, cx, sx, bx = st(p, "spatial_x"); fy, cy, sy, by = st(p, "spatial_y")
        phx = x1d[:, :, None] * fx[None]; Xv = jnp.sum(cx[None] * jnp.cos(phx) + sx[None] * jnp.sin(phx), axis=2) + bx[None, :, 0]
        phy = y1d[:, :, None] * fy[None]; Yv = jnp.sum(cy[None] * jnp.cos(phy) + sy[None] * jnp.sin(phy), axis=2) + by[None, :, 0]
        return jnp.einsum("nm,jm->nj", p["mode_coeffs"][None] * Xv, Yv)

    xc, yc, fg = data["xc"], data["yc"], data["fg"]
    xb, yb, ub = data["xb"], data["yb"], data["ub"]

    def resid(p):
        fx, cx, sx, bx = st(p, "spatial_x"); fy, cy, sy, by = st(p, "spatial_y")
        phx = xc[:, :, None] * fx[None]; Xt = cx[None] * jnp.cos(phx) + sx[None] * jnp.sin(phx)
        Xv = jnp.sum(Xt, axis=2) + bx[None, :, 0]; Xdd = jnp.sum(-(fx[None] ** 2) * Xt, axis=2)
        phy = yc[:, :, None] * fy[None]; Yt = cy[None] * jnp.cos(phy) + sy[None] * jnp.sin(phy)
        Yv = jnp.sum(Yt, axis=2) + by[None, :, 0]; Ydd = jnp.sum(-(fy[None] ** 2) * Yt, axis=2)
        mc = p["mode_coeffs"]; cX = mc[None] * Xv
        uxx = jnp.einsum("nm,jm->nj", mc[None] * Xdd, Yv)
        uyy = jnp.einsum("nm,jm->nj", cX, Ydd)
        return uxx + uyy + fg

    p = init(random.PRNGKey(seed)); opt = optax.adam(LR); state = opt.init(p)
    def loss(p):
        return jnp.mean(resid(p) ** 2) + jnp.mean((fwd(p, xb, yb) - ub) ** 2)

    @jit
    def step(p, s):
        l, g = value_and_grad(loss)(p); u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    for _ in range(2): p, state, _ = step(p, state)
    best = float("inf"); bp = p; t0 = time.time()
    for ep in range(2, epochs):
        p, state, l = step(p, state)
        if ep % EVAL_EVERY == 0 or ep == epochs - 1:
            e = _l2(np.array(grid(p, data["x1d"], data["y1d"])), data["UE"])
            if e < best: best = e; bp = jax.tree.map(lambda z: z.copy(), p)
    up = np.array(grid(bp, data["x1d"], data["y1d"]))
    rec = {"method": "SVSNN", "best_l2": best, "params": _cp(p),
           "time_s": time.time() - t0, "per_comp": _per_component_error(up, problem, data)}
    if return_pred:
        rec["u_pred"] = up; rec["UE"] = data["UE"]; rec["X"] = data["X"]; rec["Y"] = data["Y"]
    return rec


# ============================ pointwise baselines ============================
def _init_fourier(key, wc, hidden, n_hidden, ff, siren):
    k1, k2, key = random.split(key, 3)
    Wx = sfreqs(k1, wc, ff).reshape(1, -1); Wy = sfreqs(k2, wc, ff).reshape(1, -1)
    dims = [4 * ff] + [hidden] * n_hidden + [1]; layers = []
    for i in range(len(dims) - 1):
        k, key = random.split(key); din, dout = dims[i], dims[i + 1]
        if siren:
            w = random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din)
        else:
            lim = jnp.sqrt(6.0 / (din + dout))
            w = random.uniform(k, (din, dout), minval=-lim, maxval=lim)
        layers.append({"w": w, "b": jnp.zeros((dout,))})
    return {"Wx": Wx, "Wy": Wy, "mlp": layers}


def _fwd_fourier(p, xy, siren=False):
    x, y = xy[:, 0:1], xy[:, 1:2]
    Wx = jax.lax.stop_gradient(p["Wx"]); Wy = jax.lax.stop_gradient(p["Wy"])
    h = jnp.concatenate([jnp.sin(x @ Wx), jnp.cos(x @ Wx),
                         jnp.sin(y @ Wy), jnp.cos(y @ Wy)], axis=-1)
    act = jnp.sin if siren else jnp.tanh
    n = len(p["mlp"])
    for i, L in enumerate(p["mlp"]):
        h = h @ L["w"] + L["b"]
        if i < n - 1: h = act(h)
    return h


def _init_pinn(key, hidden, n_hidden):
    dims = [2] + [hidden] * n_hidden + [1]; layers = []; k = key
    for i in range(len(dims) - 1):
        k, sub = random.split(k); din, dout = dims[i], dims[i + 1]
        lim = jnp.sqrt(6.0 / (din + dout))
        layers.append({"w": random.uniform(sub, (din, dout), minval=-lim, maxval=lim),
                       "b": jnp.zeros((dout,))})
    return {"mlp": layers}


def _fwd_pinn(p, xy):
    h = xy; n = len(p["mlp"])
    for i, L in enumerate(p["mlp"]):
        h = h @ L["w"] + L["b"]
        if i < n - 1: h = jnp.tanh(h)
    return h


def _init_spinn(key, wc, features, n_layers, r, ff):
    def init_branch(key, d_in):
        keys = random.split(key, 3 + n_layers + 1)
        scale = 1.0 / jnp.sqrt(jnp.array(d_in, jnp.float32))
        bp = {"U_w": random.normal(keys[0], (d_in, features)) * scale, "U_b": jnp.zeros((features,)),
              "V_w": random.normal(keys[1], (d_in, features)) * scale, "V_b": jnp.zeros((features,)),
              "H_w": random.normal(keys[2], (d_in, features)) * scale, "H_b": jnp.zeros((features,)),
              "layers": [],
              "out_w": random.normal(keys[-1], (features, r)) * (1.0 / jnp.sqrt(jnp.array(features, jnp.float32)))}
        for i in range(n_layers):
            bp["layers"].append({"w": random.normal(keys[3 + i], (features, features)) * (1.0 / jnp.sqrt(jnp.array(features, jnp.float32))),
                                 "b": jnp.zeros((features,))})
        return bp
    k1, k2, k3, k4 = random.split(key, 4)
    return {"branch_x": init_branch(k1, 2 * ff), "branch_y": init_branch(k2, 2 * ff),
            "Wx": sfreqs(k3, wc, ff).reshape(1, -1), "Wy": sfreqs(k4, wc, ff).reshape(1, -1)}


def _spinn_branch(bp, x):
    U = jnp.tanh(x @ bp["U_w"] + bp["U_b"]); V = jnp.tanh(x @ bp["V_w"] + bp["V_b"])
    H = jnp.tanh(x @ bp["H_w"] + bp["H_b"])
    for L in bp["layers"]:
        Z = jnp.tanh(H @ L["w"] + L["b"]); H = (1.0 - Z) * U + Z * V
    return H @ bp["out_w"]


def _fwd_spinn(p, xy):
    x, y = xy[:, 0:1], xy[:, 1:2]
    xe = jnp.concatenate([jnp.sin(x @ jax.lax.stop_gradient(p["Wx"])), jnp.cos(x @ jax.lax.stop_gradient(p["Wx"]))], axis=-1)
    ye = jnp.concatenate([jnp.sin(y @ jax.lax.stop_gradient(p["Wy"])), jnp.cos(y @ jax.lax.stop_gradient(p["Wy"]))], axis=-1)
    bx = _spinn_branch(p["branch_x"], xe); by = _spinn_branch(p["branch_y"], ye)
    return jnp.sum(bx * by, axis=1, keepdims=True)


def _spinn_param_count(features, n_layers, r, ff):
    """Exact param count of the gated (U/V/H) two-branch SPINN used here."""
    branch = 3 * (2 * ff * features + features) + n_layers * (features * features + features) + features * r
    return 2 * branch + 2 * ff


def _match_spinn(target):
    """Pick (features, ff) for the gated SPINN closest to the target param count."""
    best = None
    for ff in (8, 12, 16, 24, 32):
        for feat in range(2, 64):
            c = _spinn_param_count(feat, 2, feat, ff)
            e = abs(c - target)
            if best is None or e < best[0]:
                best = (e, dict(features=feat, n_layers=2, r=feat, ff=ff), c)
            if c > target * 1.4:
                break
    return best[1], best[2]


def run_baseline(method, problem, seed, wc, target, epochs=None, return_pred=False):
    epochs = epochs or EPOCHS
    data = build_data(problem, seed)
    in_dim, out_dim, n_coord = 2, 1, 2
    if method == "SPINN":
        sizes, est = _match_spinn(target)
    else:
        sizes, est, within = harness.choose_matched(
            method, target, in_dim=in_dim, out_dim=out_dim, n_coord=n_coord, ff=FF)

    if method == "FourierPINN":
        p0 = _init_fourier(random.PRNGKey(seed), wc, sizes["hidden"], sizes["n_hidden"], sizes["ff"], False)
        fwd = lambda p, xy: _fwd_fourier(p, xy, False)
    elif method == "SIREN":
        p0 = _init_fourier(random.PRNGKey(seed), wc, sizes["hidden"], sizes["n_hidden"], sizes["ff"], True)
        fwd = lambda p, xy: _fwd_fourier(p, xy, True)
    elif method == "SPINN":
        p0 = _init_spinn(random.PRNGKey(seed), wc, sizes["features"], sizes["n_layers"], sizes["r"], sizes["ff"])
        fwd = _fwd_spinn
    elif method == "PINN":
        p0 = _init_pinn(random.PRNGKey(seed), sizes["hidden"], sizes["n_hidden"])
        fwd = _fwd_pinn
    else:
        raise ValueError(method)

    xp, yp, fp = data["xp"], data["yp"], data["fp"]
    xb, yb, ub = data["xb"], data["yb"], data["ub"]
    xyp = jnp.concatenate([xp, yp], axis=-1)
    tx = jnp.zeros_like(xyp).at[:, 0].set(1.0); ty = jnp.zeros_like(xyp).at[:, 1].set(1.0)

    def loss(p):
        uxx = _hvp(lambda z: fwd(p, z), xyp, tx)
        uyy = _hvp(lambda z: fwd(p, z), xyp, ty)
        r = uxx + uyy + fp
        bc = jnp.mean((fwd(p, jnp.concatenate([xb, yb], axis=-1)) - ub) ** 2)
        return jnp.mean(r ** 2) + bc

    def predict(p):
        X, Y = data["X"], data["Y"]
        xy = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1), jnp.float32)
        return np.array(fwd(p, xy)).reshape(N_TEST, N_TEST)

    p = p0; opt = optax.adam(LR); state = opt.init(p)
    @jit
    def step(p, s):
        l, g = value_and_grad(loss)(p); u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l
    for _ in range(2): p, state, _ = step(p, state)
    best = float("inf"); bp = p; t0 = time.time()
    for ep in range(2, epochs):
        p, state, l = step(p, state)
        if ep % EVAL_EVERY == 0 or ep == epochs - 1:
            e = _l2(predict(p), data["UE"])
            if e < best: best = e; bp = jax.tree.map(lambda z: z.copy(), p)
    up = predict(bp)
    rec = {"method": method, "best_l2": best, "params": _cp(p), "time_s": time.time() - t0,
           "target": int(target), "matched_within_tol": bool(abs(_cp(p) - target) <= 0.10 * target),
           "per_comp": _per_component_error(up, problem, data)}
    if return_pred:
        rec["u_pred"] = up; rec["UE"] = data["UE"]; rec["X"] = data["X"]; rec["Y"] = data["Y"]
    return rec


def svsnn_target(problem, wc):
    """Parameter count of the SV-SNN config = matched target for baselines."""
    r = run_svsnn(problem, 0, wc, epochs=3)
    return r["params"]
