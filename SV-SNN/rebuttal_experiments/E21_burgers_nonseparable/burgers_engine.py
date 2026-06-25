"""Shared fair-comparison engine for the non-separable Burgers shock (E21, Track A).

PDE (the classic Basdevant / Raissi benchmark, genuinely non-separable in (x,t)):

    u_t + u * u_x = nu * u_xx ,   nu = 0.01 / pi
    x in [-1, 1] ,  t in [0, 1]
    IC:  u(x, 0) = -sin(pi x)
    BC:  u(-1, t) = u(+1, t) = 0           (Dirichlet)

Five methods solved as *pure-residual* space-time PINNs under an IDENTICAL
training protocol (same collocation budget, same Adam+LBFGS schedule, same
soft IC/BC weights, float32, 3 shared seeds) and a MATCHED parameter budget
(baselines sized to the SV-SNN parameter count +-10%):

  SVSNN       : separable spectral net  u = sum_m c_m * X_m(x) * T_m(t),
                X_m / T_m are frozen-frequency cos/sin spectral nets.
  FourierPINN : frozen multi-level Fourier features + tanh MLP   (case7 form)
  SIREN       : Fourier features + sine-activated MLP             (case7 form)
  SPINN       : separable Fourier-feature gated branches, low rank
  PINN        : plain tanh MLP, no frequency prior                (floor)

All derivatives (u_t, u_x, u_xx) are obtained by the SAME autodiff path
(forward-over-forward jvp) for every method, so the comparison isolates the
*architecture* only.  This is the honest "SV-SNN as a residual PINN" track;
on this genuinely non-separable shock every pure PINN is expected to be far
from machine precision -- that limitation is the intended, defensible message.
The high-accuracy hybrid (Galerkin-ODE + neural compression, solve_burgers_svsnn_v10.py)
is reported SEPARATELY in Track B with an explicit caveat.
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad, jvp
import optax
import scipy.optimize as sopt
from pyDOE import lhs

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.abspath(os.path.join(HERE, "..", "_fair_freq_common"))
for p in (HERE, COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import harness
import fair_engine as fe   # reuse validated baseline inits/forwards + matching

# ---------------- PDE / domain ----------------
NU = 0.01 / np.pi
X_LO, X_HI = -1.0, 1.0
T0, T1 = 0.0, 1.0

# ---------------- training protocol (uniform across all methods) -----------
EPOCHS_ADAM = 15000
LR = 3e-3
LBFGS_ITERS = 3000
EVAL_EVERY = 250
W_IC = 20.0          # soft initial-condition weight
W_BC = 20.0          # soft boundary-condition weight

# ---------------- collocation / supervision budget --------------------------
N_PDE = 20000        # scattered space-time collocation
N_IC = 512           # initial-condition supervision points
N_BC = 256           # per boundary (x=-1 and x=+1)

# ---------------- SV-SNN architecture ---------------------------------------
# Faithful SV-SNN for a time-dependent PDE: a SEPARABLE rank-M ansatz
#     u(x,t) = sum_m  X_m(x) * T_m(t)
# with X_m a frozen-frequency cos/sin SPECTRAL spatial net (the SV-SNN spatial
# spectral separation) and the temporal factors [T_1..T_M](t) produced by a
# small shared learnable MLP (mirrors the neural temporal coefficients of the
# v-series).  The product structure keeps SV-SNN strictly separable -- this is
# precisely the inductive bias that the non-separable shock stresses.
M = int(os.environ.get("E21_M", "8"))    # separable modes (= rank)
KX = int(os.environ.get("E21_KX", "24"))  # x-frequencies per spatial mode
TH = int(os.environ.get("E21_TH", "32"))  # temporal MLP hidden width
TL = 2               # temporal MLP hidden layers
WC_X = float(os.environ.get("E21_WCX", "15.0"))  # x frequency prior (shared with freq-aware baselines)
FF = 24              # Fourier-feature count per coordinate for baselines


def _l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _cp(p):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(p) if hasattr(x, "size")))


# ============================ data ============================
def load_reference():
    ref = os.path.join(HERE, "saved_data", "burgers_reference_hires.npz")
    d = np.load(ref)
    t_r = np.asarray(d["t"], np.float64)        # (Nt,)
    x_r = np.asarray(d["x"], np.float64)        # (Nx,)
    u_r = np.asarray(d["usol"], np.float64)     # (Nt, Nx)
    return t_r, x_r, u_r


def build_data(seed):
    rng = np.random.RandomState(seed)
    # scattered space-time collocation via LHS
    s = lhs(2, samples=N_PDE)
    xp = X_LO + (X_HI - X_LO) * s[:, 0:1]
    tp = T0 + (T1 - T0) * s[:, 1:2]
    # initial condition  u(x,0) = -sin(pi x)
    xic = np.linspace(X_LO, X_HI, N_IC).reshape(-1, 1)
    tic = np.zeros_like(xic)
    uic = -np.sin(np.pi * xic)
    # Dirichlet boundaries  u(+-1, t) = 0
    tb = np.linspace(T0, T1, N_BC).reshape(-1, 1)
    xb = np.vstack([np.full((N_BC, 1), X_LO), np.full((N_BC, 1), X_HI)])
    tb2 = np.vstack([tb, tb])
    ub = np.zeros_like(xb)
    return {
        "xtp": jnp.asarray(np.concatenate([xp, tp], 1), jnp.float32),
        "xtic": jnp.asarray(np.concatenate([xic, tic], 1), jnp.float32),
        "uic": jnp.asarray(uic, jnp.float32),
        "xtb": jnp.asarray(np.concatenate([xb, tb2], 1), jnp.float32),
        "ub": jnp.asarray(ub, jnp.float32),
    }


def eval_grid(nx=512, nt=201):
    """Down-sampled evaluation grid (interpolated from the hi-res reference)."""
    from scipy.interpolate import RegularGridInterpolator
    t_r, x_r, u_r = load_reference()
    xe = np.linspace(X_LO, X_HI, nx)
    te = np.linspace(T0, T1, nt)
    Te, Xe = np.meshgrid(te, xe, indexing="ij")     # (nt, nx)
    interp = RegularGridInterpolator((t_r, x_r), u_r, method="cubic",
                                     bounds_error=False, fill_value=None)
    Ue = interp((Te, Xe))                            # (nt, nx)
    XT = np.stack([Xe.reshape(-1), Te.reshape(-1)], 1)
    return jnp.asarray(XT, jnp.float32), Ue, xe, te


# ============================ SV-SNN (separable spectral spatial x neural temporal) ===========
def svsnn_init(key, wcx=WC_X):
    keys = random.split(key, M * 3 + TL + 3); ki = 0; sx = []
    for _ in range(M):
        sx.append({"freqs": fe.sfreqs(keys[ki], wcx, KX),
                   "cos_c": random.normal(keys[ki + 1], (KX,)) * 0.1,
                   "sin_c": random.normal(keys[ki + 2], (KX,)) * 0.1}); ki += 3
    # shared temporal MLP: 1 -> [TH]*TL -> M
    dims = [1] + [TH] * TL + [M]; tmlp = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        lim = float(np.sqrt(6.0 / (din + dout)))
        tmlp.append({"w": random.uniform(keys[ki], (din, dout), minval=-lim, maxval=lim),
                     "b": jnp.zeros((dout,))}); ki += 1
    return {"sx": sx, "tmlp": tmlp}


def _svsnn_temporal(p, t):
    h = t; n = len(p["tmlp"])
    for i, L in enumerate(p["tmlp"]):
        h = h @ L["w"] + L["b"]
        if i < n - 1:
            h = jnp.tanh(h)
    return h                          # (N, M)


def svsnn_fwd(p, xt):
    x = xt[:, 0:1]; t = xt[:, 1:2]
    Xs = []
    for m in range(M):
        fx = jax.lax.stop_gradient(p["sx"][m]["freqs"])[None, :]
        wx = fx * x
        Xm = jnp.sum(p["sx"][m]["cos_c"] * jnp.cos(wx) + p["sx"][m]["sin_c"] * jnp.sin(wx),
                     axis=1, keepdims=True)
        Xs.append(Xm)
    X = jnp.concatenate(Xs, axis=1)   # (N, M)
    T = _svsnn_temporal(p, t)         # (N, M)
    return jnp.sum(X * T, axis=1, keepdims=True)


# ============================ uniform Burgers residual loss ============================
def make_loss(fwd, data):
    xtp = data["xtp"]; xtic = data["xtic"]; uic = data["uic"]
    xtb = data["xtb"]; ub = data["ub"]
    tx = jnp.zeros_like(xtp).at[:, 0].set(1.0)   # d/dx direction
    tt = jnp.zeros_like(xtp).at[:, 1].set(1.0)   # d/dt direction

    def loss(p):
        f = lambda z: fwd(p, z)
        u, ux = jvp(f, (xtp,), (tx,))
        _, ut = jvp(f, (xtp,), (tt,))
        uxx = fe._hvp(f, xtp, tx)
        r = ut + u * ux - NU * uxx
        ic = f(xtic) - uic
        bc = f(xtb) - ub
        return jnp.mean(r ** 2) + W_IC * jnp.mean(ic ** 2) + W_BC * jnp.mean(bc ** 2)

    return loss


def _adam_train(p, loss, epochs, eval_fn):
    sched = optax.cosine_decay_schedule(LR, epochs, alpha=1e-3)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    state = opt.init(p)

    @jit
    def step(p, s):
        l, g = value_and_grad(loss)(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    for _ in range(2):
        p, state, _ = step(p, state)
    best = float("inf"); bp = p
    for ep in range(2, epochs):
        p, state, l = step(p, state)
        if ep % EVAL_EVERY == 0 or ep == epochs - 1:
            e = eval_fn(p)
            if e < best:
                best = e; bp = jax.tree_util.tree_map(lambda z: z.copy(), p)
    return bp, best


def _lbfgs_polish(p, loss, eval_fn, iters):
    if iters <= 0:
        return p, eval_fn(p)
    flat0, unravel = jax.flatten_util.ravel_pytree(p)

    @jit
    def lg(fp):
        pp = unravel(fp)
        l, g = value_and_grad(loss)(pp)
        gf, _ = jax.flatten_util.ravel_pytree(g)
        return l, gf

    def fg(x):
        l, g = lg(jnp.asarray(x, jnp.float32))
        return float(l), np.asarray(g, np.float64)

    res = sopt.minimize(fg, np.asarray(flat0, np.float64), method="L-BFGS-B",
                        jac=True, options={"maxiter": iters, "maxfun": iters * 2,
                                           "ftol": 1e-14, "gtol": 1e-12, "maxcor": 50})
    pp = unravel(jnp.asarray(res.x, jnp.float32))
    return pp, eval_fn(pp)


# ============================ public run API ============================
def _eval_fn_factory(fwd, XT, Ue):
    nt, nx = Ue.shape

    def ef(p):
        up = np.asarray(fwd(p, XT)).reshape(nt, nx)
        return _l2(up, Ue)
    return ef


def svsnn_target(seed=0):
    """SV-SNN parameter count = matched target for the baselines."""
    p = svsnn_init(random.PRNGKey(seed))
    return _cp(p)


def run(method, seed, target=None, epochs_adam=None, lbfgs_iters=None,
        return_pred=False, eval_nx=512, eval_nt=201):
    epochs_adam = EPOCHS_ADAM if epochs_adam is None else epochs_adam
    lbfgs_iters = LBFGS_ITERS if lbfgs_iters is None else lbfgs_iters
    data = build_data(seed)
    XT, Ue, xe, te = eval_grid(eval_nx, eval_nt)

    if method == "SVSNN":
        p0 = svsnn_init(random.PRNGKey(seed))
        fwd = svsnn_fwd
        matched = True
        tgt = _cp(p0)
    else:
        if target is None:
            target = svsnn_target(seed)
        if method == "SPINN":
            sizes, est = fe._match_spinn(target)
            p0 = fe._init_spinn(random.PRNGKey(seed), WC_X, sizes["features"],
                                sizes["n_layers"], sizes["r"], sizes["ff"])
            fwd = fe._fwd_spinn
        elif method == "FourierPINN":
            sizes, est, _ = harness.choose_matched("FourierPINN", target, in_dim=2,
                                                   out_dim=1, n_coord=2, ff=FF)
            p0 = fe._init_fourier(random.PRNGKey(seed), WC_X, sizes["hidden"],
                                  sizes["n_hidden"], sizes["ff"], False)
            fwd = lambda p, xy: fe._fwd_fourier(p, xy, False)
        elif method == "SIREN":
            sizes, est, _ = harness.choose_matched("SIREN", target, in_dim=2,
                                                   out_dim=1, n_coord=2, ff=FF)
            p0 = fe._init_fourier(random.PRNGKey(seed), WC_X, sizes["hidden"],
                                  sizes["n_hidden"], sizes["ff"], True)
            fwd = lambda p, xy: fe._fwd_fourier(p, xy, True)
        elif method == "PINN":
            sizes, est, _ = harness.choose_matched("PINN", target, in_dim=2,
                                                   out_dim=1, n_coord=2)
            p0 = fe._init_pinn(random.PRNGKey(seed), sizes["hidden"], sizes["n_hidden"])
            fwd = fe._fwd_pinn
        else:
            raise ValueError(method)
        tgt = target
        matched = bool(abs(_cp(p0) - target) <= 0.10 * target)

    loss = make_loss(fwd, data)
    eval_fn = _eval_fn_factory(fwd, XT, Ue)

    t0 = time.time()
    bp, best_adam = _adam_train(p0, loss, epochs_adam, eval_fn)
    bp, best = _lbfgs_polish(bp, loss, eval_fn, lbfgs_iters)
    best = min(best, best_adam)
    train_t = time.time() - t0

    # final / best prediction & inference timing
    nt, nx = Ue.shape
    fwd_call = lambda: fwd(bp, XT)
    inf_ms = harness.time_inference(fwd_call, n_repeat=10)
    final_l2 = eval_fn(bp)

    rec = harness.normalize_record(
        method, "matched", seed,
        params=_cp(bp), best_l2=best, final_l2=final_l2,
        train_time_sec=train_t, n_epochs=epochs_adam + lbfgs_iters,
        n_collocation=N_PDE, inference_ms=inf_ms,
        target_params=tgt, matched_within_tol=matched,
        extra={"nu": float(NU)})

    if return_pred:
        up = np.asarray(fwd(bp, XT)).reshape(nt, nx)
        rec["u_pred"] = up; rec["UE"] = Ue; rec["xe"] = xe; rec["te"] = te
    return rec


if __name__ == "__main__":
    # quick self-test on CPU with a tiny budget
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="SVSNN")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam", type=int, default=400)
    ap.add_argument("--lbfgs", type=int, default=0)
    a = ap.parse_args()
    print("SV-SNN target params:", svsnn_target())
    r = run(a.method, a.seed, epochs_adam=a.adam, lbfgs_iters=a.lbfgs)
    print(json.dumps({k: v for k, v in r.items()
                      if k not in ("u_pred", "UE", "xe", "te")}, indent=2))
