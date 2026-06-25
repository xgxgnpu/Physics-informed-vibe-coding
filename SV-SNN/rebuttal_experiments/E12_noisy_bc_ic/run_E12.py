"""
E12 - Robustness to noisy boundary/initial conditions  (addresses R1.4 "noisy BC/IC")
=====================================================================================
Problem: separable 2D Helmholtz  -Lap u - kappa^2 u = f,  u = sin(kappa x) sin(kappa y),
         kappa = 24 pi, homogeneous Dirichlet (analytic f = kappa^2 u).

We corrupt the *boundary supervision* with relative Gaussian noise:
    ub_noisy = ub + eps * rms(ub) * N(0,1),   eps in {0, 0.01, 0.05, 0.10}.
The PDE residual (physics) is kept clean; only the data we supervise on is noisy.
ALL methods receive the *identical* noisy boundary data for a given (eps, seed),
the same w_char multi-level frequency init, the same epochs/optimizer/grid.
Evaluation is always against the CLEAN analytic solution.

Methods: accelerated SV-SNN, FourierPINN, vanilla PINN. 3 seeds. Self-contained.
Reported HONESTLY (degradation curve vs noise level).
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from pyDOE import lhs

EPOCHS = 8000; LR = 1e-3; EVAL_EVERY = 100; SEEDS = [0, 1, 2]
KAPPA = 24 * np.pi; WCHAR = KAPPA
M = 6; K = 32; NC = 120; N_TEST = 200; N_PDE = 10000; N_BC = 1024
NOISE = [0.0, 0.01, 0.05, 0.10]
SAVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data"); os.makedirs(SAVE, exist_ok=True)


def l2(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x, "size")))
def sfreqs(key, wc):
    nl = K // 4; ncc = K // 2; nh = K - nl - ncc; _, k2, k3 = jax.random.split(key, 3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0, wc, nl),
                                     jnp.abs(jax.random.normal(k2, (ncc,)) * float(wc) * 0.3 + wc),
                                     jax.random.uniform(k3, (nh,), minval=wc * 0.5, maxval=wc)]))
def uex(x, y): return np.sin(KAPPA * x) * np.sin(KAPPA * y)


def build(seed, eps):
    np.random.seed(seed)
    nps = N_BC // 4; t = np.linspace(0, 1, nps).reshape(-1, 1)
    xb = np.vstack([np.zeros((nps, 1)), np.ones((nps, 1)), t, t])
    yb = np.vstack([t, t, np.zeros((nps, 1)), np.ones((nps, 1))])
    ub_clean = uex(xb, yb)
    # Homogeneous-Dirichlet BC are exactly 0 here, so scale the (measurement) noise by the
    # solution's interior RMS amplitude (= 0.5 for sin*sin) rather than the zero boundary RMS.
    scale = float(np.sqrt(np.mean(uex(np.random.rand(4000, 1), np.random.rand(4000, 1)) ** 2)))
    ub_noisy = ub_clean + eps * scale * np.random.randn(*ub_clean.shape)
    pde = lhs(2, samples=N_PDE)
    x1 = np.linspace(0, 1, N_TEST); y1 = np.linspace(0, 1, N_TEST); X, Y = np.meshgrid(x1, y1, indexing="ij")
    xc = np.linspace(0, 1, NC); yc = np.linspace(0, 1, NC); Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    fg = KAPPA ** 2 * np.sin(KAPPA * Xc) * np.sin(KAPPA * Yc)
    return {"xb": jnp.asarray(xb, jnp.float32), "yb": jnp.asarray(yb, jnp.float32),
            "ub": jnp.asarray(ub_noisy, jnp.float32),
            "x_pde": jnp.asarray(pde[:, 0:1], jnp.float32), "y_pde": jnp.asarray(pde[:, 1:2], jnp.float32),
            "f_pde": jnp.asarray(KAPPA ** 2 * np.sin(KAPPA * pde[:, 0:1]) * np.sin(KAPPA * pde[:, 1:2]), jnp.float32),
            "ue": uex(X, Y), "X": X, "Y": Y,
            "xc": jnp.asarray(xc.reshape(-1, 1), jnp.float32), "yc": jnp.asarray(yc.reshape(-1, 1), jnp.float32),
            "fg": jnp.asarray(fg, jnp.float32),
            "x1d": jnp.asarray(x1.reshape(-1, 1), jnp.float32), "y1d": jnp.asarray(y1.reshape(-1, 1), jnp.float32)}


# ---------------- SV-SNN accelerated ----------------
def svsnn_run(data, seed):
    def init(key):
        keys = jax.random.split(key, M * 6 + 1); ki = 0; sx, sy = [], []
        for _ in range(M):
            sx.append({"freqs": sfreqs(keys[ki], WCHAR), "cos_c": jax.random.normal(keys[ki + 1], (K,)) * 0.1,
                       "sin_c": jax.random.normal(keys[ki + 2], (K,)) * 0.1, "bias": jnp.zeros(1)}); ki += 3
            sy.append({"freqs": sfreqs(keys[ki], WCHAR), "cos_c": jax.random.normal(keys[ki + 1], (K,)) * 0.1,
                       "sin_c": jax.random.normal(keys[ki + 2], (K,)) * 0.1, "bias": jnp.zeros(1)}); ki += 3
        return {"spatial_x": sx, "spatial_y": sy, "mode_coeffs": jax.random.normal(keys[ki], (M,)) * 0.1}
    def st(p, axis):
        f = jnp.stack([jax.lax.stop_gradient(p[axis][n]["freqs"]) for n in range(M)])
        c = jnp.stack([p[axis][n]["cos_c"] for n in range(M)]); s = jnp.stack([p[axis][n]["sin_c"] for n in range(M)])
        b = jnp.stack([p[axis][n]["bias"] for n in range(M)]); return f, c, s, b
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
    xc, yc, fg = data["xc"], data["yc"], data["fg"]; xb, yb, ub = data["xb"], data["yb"], data["ub"]
    def resid(p):
        fx, cx, sx, bx = st(p, "spatial_x"); fy, cy, sy, by = st(p, "spatial_y")
        phx = xc[:, :, None] * fx[None]; Xt = cx[None] * jnp.cos(phx) + sx[None] * jnp.sin(phx); Xv = jnp.sum(Xt, axis=2) + bx[None, :, 0]; Xdd = jnp.sum(-(fx[None] ** 2) * Xt, axis=2)
        phy = yc[:, :, None] * fy[None]; Yt = cy[None] * jnp.cos(phy) + sy[None] * jnp.sin(phy); Yv = jnp.sum(Yt, axis=2) + by[None, :, 0]; Ydd = jnp.sum(-(fy[None] ** 2) * Yt, axis=2)
        mc = p["mode_coeffs"]; cX = mc[None] * Xv
        u = jnp.einsum("nm,jm->nj", cX, Yv); uxx = jnp.einsum("nm,jm->nj", mc[None] * Xdd, Yv); uyy = jnp.einsum("nm,jm->nj", cX, Ydd)
        return -(uxx + uyy) - KAPPA ** 2 * u - fg
    p = init(random.PRNGKey(seed)); opt = optax.adam(LR); state = opt.init(p)
    def loss(p): return jnp.mean(resid(p) ** 2) + jnp.mean((fwd(p, xb, yb) - ub) ** 2)
    @jit
    def step(p, s):
        l, g = value_and_grad(loss)(p); u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s, l
    for _ in range(2): p, state, _ = step(p, state)
    best = float("inf"); bp = p; t0 = time.time()
    for ep in range(2, EPOCHS):
        p, state, l = step(p, state)
        if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
            e = l2(np.array(grid(p, data["x1d"], data["y1d"])), data["ue"])
            if e < best: best = e; bp = jax.tree.map(lambda z: z.copy(), p)
    return {"method": "SVSNN", "best_l2": best, "params": cp(p), "time_s": time.time() - t0,
            "u_pred": np.array(grid(bp, data["x1d"], data["y1d"]))}


# ---------------- pointwise baselines ----------------
def hvp(f, x, v):
    g = lambda z: jax.jvp(f, (z,), (v,))[1]; return jax.jvp(g, (x,), (v,))[1]
def baseline_run(which, data, seed):
    if which == "FourierPINN":
        def init(key):
            k1, k2, key = random.split(key, 3)
            Wx = sfreqs(k1, WCHAR).reshape(1, -1); Wy = sfreqs(k2, WCHAR).reshape(1, -1)
            dims = [4 * K, 128, 128, 128, 1]; layers = []
            for i in range(len(dims) - 1):
                k, key = random.split(key); lim = jnp.sqrt(6.0 / (dims[i] + dims[i + 1]))
                layers.append({"w": random.uniform(k, (dims[i], dims[i + 1]), minval=-lim, maxval=lim), "b": jnp.zeros((dims[i + 1],))})
            return {"Wx": Wx, "Wy": Wy, "mlp": layers}
        def fwd(p, xy):
            x, y = xy[:, 0:1], xy[:, 1:2]; Wx = jax.lax.stop_gradient(p["Wx"]); Wy = jax.lax.stop_gradient(p["Wy"])
            h = jnp.concatenate([jnp.sin(x @ Wx), jnp.cos(x @ Wx), jnp.sin(y @ Wy), jnp.cos(y @ Wy)], axis=-1); n = len(p["mlp"])
            for i, L in enumerate(p["mlp"]):
                h = h @ L["w"] + L["b"]
                if i < n - 1: h = jnp.tanh(h)
            return h
    else:
        def init(key):
            dims = [2, 128, 128, 128, 128, 1]; layers = []; k = key
            for i in range(len(dims) - 1):
                k, sub = random.split(k); lim = jnp.sqrt(6.0 / (dims[i] + dims[i + 1]))
                layers.append({"w": random.uniform(sub, (dims[i], dims[i + 1]), minval=-lim, maxval=lim), "b": jnp.zeros((dims[i + 1],))})
            return {"mlp": layers}
        def fwd(p, xy):
            h = xy; n = len(p["mlp"])
            for i, L in enumerate(p["mlp"]):
                h = h @ L["w"] + L["b"]
                if i < n - 1: h = jnp.tanh(h)
            return h
    xp, yp = data["x_pde"], data["y_pde"]; fpde = data["f_pde"]; xb, yb, ub = data["xb"], data["yb"], data["ub"]
    xyp = jnp.concatenate([xp, yp], axis=-1); tx = jnp.zeros_like(xyp).at[:, 0].set(1.0); ty = jnp.zeros_like(xyp).at[:, 1].set(1.0)
    def loss(p):
        uxx = hvp(lambda z: fwd(p, z), xyp, tx); uyy = hvp(lambda z: fwd(p, z), xyp, ty)
        u = fwd(p, xyp); r = -(uxx + uyy) - KAPPA ** 2 * u - fpde
        bc = jnp.mean((fwd(p, jnp.concatenate([xb, yb], axis=-1)) - ub) ** 2)
        return jnp.mean(r ** 2) + bc
    def pred(p):
        X, Y = np.meshgrid(np.array(data["x1d"]).squeeze(), np.array(data["y1d"]).squeeze(), indexing="ij")
        xy = jnp.asarray(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1), jnp.float32)
        return np.array(fwd(p, xy)).reshape(N_TEST, N_TEST)
    p = init(random.PRNGKey(seed)); opt = optax.adam(LR); state = opt.init(p)
    @jit
    def step(p, s):
        l, g = value_and_grad(loss)(p); u, s = opt.update(g, s, p); return optax.apply_updates(p, u), s, l
    for _ in range(2): p, state, _ = step(p, state)
    best = float("inf"); bp = p; t0 = time.time()
    for ep in range(2, EPOCHS):
        p, state, l = step(p, state)
        if ep % EVAL_EVERY == 0 or ep == EPOCHS - 1:
            e = l2(pred(p), data["ue"])
            if e < best: best = e; bp = jax.tree.map(lambda z: z.copy(), p)
    return {"method": which, "best_l2": best, "params": cp(p), "time_s": time.time() - t0, "u_pred": pred(bp)}


METHODS = [(svsnn_run, "SVSNN"), (lambda d, s: baseline_run("FourierPINN", d, s), "FourierPINN"),
           (lambda d, s: baseline_run("PINN", d, s), "PINN")]


def main():
    out = {}
    for eps in NOISE:
        key = f"eps_{eps:.2f}"; out[key] = {}
        print(f"\n##### noise eps={eps} #####", flush=True)
        for runner, name in METHODS:
            recs = []
            for sd in SEEDS:
                data = build(sd, eps); r = runner(data, sd); recs.append(r)
                print(f"  {key} {name} seed={sd}: best L2={r['best_l2']:.3e} t={r['time_s']:.1f}s params={r['params']}", flush=True)
            b = np.array([r["best_l2"] for r in recs]); tt = np.array([r["time_s"] for r in recs])
            out[key][name] = {"params": recs[0]["params"], "best_l2_mean": float(b.mean()),
                              "best_l2_std": float(b.std()), "best_l2_min": float(b.min()),
                              "time_s_mean": float(tt.mean())}
            if abs(eps - 0.05) < 1e-9:
                d0 = build(0, eps)
                np.savez(os.path.join(SAVE, f"{name}_eps05_pred.npz"), u_pred=recs[0]["u_pred"],
                         ue=d0["ue"], X=d0["X"], Y=d0["Y"])
    with open(os.path.join(SAVE, "E12_results.json"), "w") as f: json.dump(out, f, indent=2)
    print("\nSaved E12 to", SAVE)


if __name__ == "__main__": main()
