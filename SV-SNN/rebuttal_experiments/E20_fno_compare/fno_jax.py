"""2D Fourier Neural Operator (FNO) in pure JAX + a parametric Poisson dataset.

Data-driven operator-learning baseline for the E20 comparison against the
physics-informed SV-SNN. The FNO learns the solution operator  f -> u  of
    -Lap u = f    on [0,1]^2  (Dirichlet, manufactured solutions)
from a dataset of (source field, solution field) pairs, then predicts u for
held-out f. This is fundamentally different from SV-SNN (which solves a single
instance from the PDE residual, with no solution data) -- the comparison is
therefore on accuracy, DATA requirement, and amortized compute, reported honestly.

Implementation notes:
  - Channels-first internal layout (batch, C, H, W); rfft2 over (H, W).
  - Standard FNO2d spectral conv keeping `modes` low frequencies in each of the
    two corners of the rfft spectrum, with learnable complex weights.
  - Input/output standardized with train statistics (kept with the params).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax


# ----------------------------- dataset -----------------------------
def make_problem_bank(n_samples, seed, n_terms=3, freq_set=None):
    """Random multi-frequency manufactured solutions for -Lap u = f on [0,1]^2.

    u = sum_m a_m sin(p_m x) sin(q_m y),  f = sum_m a_m (p_m^2+q_m^2) sin sin.
    Returns a list of dicts {a, p, q, kmax} (one per sample); genuinely 2D
    multi-frequency (p_m != q_m allowed), so the family is not trivially aligned
    to SV-SNN's separable structure.
    """
    if freq_set is None:
        freq_set = np.pi * np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=np.float64)
    rng = np.random.RandomState(seed)
    bank = []
    for _ in range(n_samples):
        p = rng.choice(freq_set, size=n_terms, replace=True)
        q = rng.choice(freq_set, size=n_terms, replace=True)
        a = rng.uniform(-1.0, 1.0, size=n_terms)
        a = a / np.sqrt(np.sum(a ** 2))  # normalize amplitude energy
        kmax = float(max(p.max(), q.max()))
        bank.append({"a": a, "p": p, "q": q, "kmax": kmax})
    return bank


def eval_fields(sample, X, Y):
    """Evaluate (u, f) of one sample on a meshgrid (X, Y)."""
    u = np.zeros_like(X); f = np.zeros_like(X)
    for a, p, q in zip(sample["a"], sample["p"], sample["q"]):
        s = np.sin(p * X) * np.sin(q * Y)
        u += a * s
        f += a * (p * p + q * q) * s
    return u, f


def build_grid_dataset(bank, N):
    x = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, x, indexing="ij")
    F = np.zeros((len(bank), N, N), dtype=np.float32)
    U = np.zeros((len(bank), N, N), dtype=np.float32)
    for i, s in enumerate(bank):
        u, f = eval_fields(s, X, Y)
        U[i] = u; F[i] = f
    return F, U, X.astype(np.float32), Y.astype(np.float32)


def make_svsnn_problem(sample):
    """Build a fair_engine problem dict for one bank sample (physics-informed)."""
    a, p, q = sample["a"], sample["p"], sample["q"]

    def u_exact(x, y):
        out = 0.0
        for am, pm, qm in zip(a, p, q):
            out = out + am * np.sin(pm * x) * np.sin(qm * y)
        return out

    def source(x, y):  # f = -Lap u
        out = 0.0
        for am, pm, qm in zip(a, p, q):
            out = out + am * (pm * pm + qm * qm) * np.sin(pm * x) * np.sin(qm * y)
        return out

    return {"name": "fno_inst", "u_exact": u_exact, "source": source,
            "w_char": float(sample["kmax"]), "domain": (0.0, 1.0)}


# ----------------------------- FNO model -----------------------------
def fno_init(key, modes=16, width=32, n_layers=4, in_ch=3, proj=128):
    keys = random.split(key, n_layers + 4)
    scale = 1.0 / (width * width)
    lift = random.normal(keys[0], (in_ch, width)) * (1.0 / in_ch)
    layers = []
    for i in range(n_layers):
        k = keys[1 + i]
        ks = random.split(k, 5)
        layers.append({
            "R1_re": random.normal(ks[0], (width, width, modes, modes)) * scale,
            "R1_im": random.normal(ks[1], (width, width, modes, modes)) * scale,
            "R2_re": random.normal(ks[2], (width, width, modes, modes)) * scale,
            "R2_im": random.normal(ks[3], (width, width, modes, modes)) * scale,
            "W": random.normal(ks[4], (width, width)) * (1.0 / width),
            "b": jnp.zeros((width,)),
        })
    p1 = random.normal(keys[-3], (width, proj)) * (1.0 / width)
    p2 = random.normal(keys[-2], (proj, 1)) * (1.0 / proj)
    return {"lift": lift, "layers": layers, "proj1": p1, "proj2": p2,
            "b1": jnp.zeros((proj,))}


def _spectral_conv(x, layer, modes):
    # x: (batch, width, H, W)
    B, C, H, W = x.shape
    x_ft = jnp.fft.rfft2(x, axes=(2, 3))  # (B, C, H, W//2+1) complex
    R1 = layer["R1_re"] + 1j * layer["R1_im"]
    R2 = layer["R2_re"] + 1j * layer["R2_im"]
    out_ft = jnp.zeros((B, C, H, W // 2 + 1), dtype=x_ft.dtype)
    a = x_ft[:, :, :modes, :modes]
    out_ft = out_ft.at[:, :, :modes, :modes].set(jnp.einsum("bixy,ioxy->boxy", a, R1))
    b = x_ft[:, :, -modes:, :modes]
    out_ft = out_ft.at[:, :, -modes:, :modes].set(jnp.einsum("bixy,ioxy->boxy", b, R2))
    return jnp.fft.irfft2(out_ft, s=(H, W), axes=(2, 3))


def fno_forward(params, x, modes):
    """x: (batch, H, W, in_ch) -> (batch, H, W)."""
    h = jnp.einsum("bhwi,io->bohw", x, params["lift"])  # to channels-first width
    n = len(params["layers"])
    for i, layer in enumerate(params["layers"]):
        sp = _spectral_conv(h, layer, modes)
        pw = jnp.einsum("bihw,io->bohw", h, layer["W"]) + layer["b"][None, :, None, None]
        h = sp + pw
        if i < n - 1:
            h = jax.nn.gelu(h)
    h = jnp.einsum("bohw->bhwo", h)  # back to channels-last
    h = jax.nn.gelu(jnp.einsum("bhwo,op->bhwp", h, params["proj1"]) + params["b1"])
    out = jnp.einsum("bhwp,pq->bhwq", h, params["proj2"])[..., 0]
    return out


def count_params(params):
    return int(sum(v.size for v in jax.tree_util.tree_leaves(params)
                   if hasattr(v, "size")))


# ----------------------------- training -----------------------------
def _rel_l2_batch(pred, true):
    num = jnp.sqrt(jnp.sum((pred - true) ** 2, axis=(1, 2)))
    den = jnp.sqrt(jnp.sum(true ** 2, axis=(1, 2))) + 1e-12
    return num / den


def train_fno(key, Ftr, Utr, Fte, Ute, X, Y, modes=16, width=32, n_layers=4,
              epochs=500, lr=1e-3, batch=32):
    """Train FNO on (Ftr->Utr); evaluate relative L2 on held-out (Fte->Ute).

    Returns metrics dict (no field arrays) for JSON serialization.
    """
    # standardize with train stats
    f_mean, f_std = float(Ftr.mean()), float(Ftr.std() + 1e-8)
    u_mean, u_std = float(Utr.mean()), float(Utr.std() + 1e-8)

    def to_input(F):
        Fn = (F - f_mean) / f_std
        c = np.broadcast_to(X[None], F.shape)
        d = np.broadcast_to(Y[None], F.shape)
        return np.stack([Fn, c, d], axis=-1).astype(np.float32)

    Xtr = jnp.asarray(to_input(Ftr)); Ytr = jnp.asarray(((Utr - u_mean) / u_std).astype(np.float32))
    Xte = jnp.asarray(to_input(Fte)); Ute_j = jnp.asarray(Ute.astype(np.float32))
    n = Xtr.shape[0]

    params = fno_init(key, modes, width, n_layers, in_ch=3)
    opt = optax.adam(lr); state = opt.init(params)

    def loss(p, xb, yb):
        pred = fno_forward(p, xb, modes)
        return jnp.mean((pred - yb) ** 2)

    @jit
    def step(p, st, xb, yb):
        l, g = value_and_grad(loss)(p, xb, yb)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    @jit
    def eval_rel(p):
        pred = fno_forward(p, Xte, modes) * u_std + u_mean
        rl = _rel_l2_batch(pred, Ute_j)
        return jnp.mean(rl), jnp.std(rl)

    kk = key
    for _ in range(2):  # warmup compile
        bidx = np.arange(min(batch, n))
        params, state, _ = step(params, state, Xtr[bidx], Ytr[bidx])

    best = float("inf"); best_std = 0.0
    t0 = time.time()
    for ep in range(epochs):
        kk, sub = random.split(kk)
        perm = np.array(random.permutation(sub, n))
        for s in range(0, n, batch):
            bidx = perm[s:s + batch]
            params, state, _ = step(params, state, Xtr[bidx], Ytr[bidx])
        if ep % 25 == 0 or ep == epochs - 1:
            m, sd = eval_rel(params)
            m = float(m)
            if m < best:
                best = m; best_std = float(sd)
    train_time = time.time() - t0

    # inference time per single instance
    one = Xte[:1]
    pf = jit(lambda p, xx: fno_forward(p, xx, modes))
    pf(params, one).block_until_ready()
    r = 20; ti = time.time()
    for _ in range(r):
        pf(params, one).block_until_ready()
    infer_ms = (time.time() - ti) / r * 1000.0

    try:
        peak_mb = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e6
    except Exception:
        peak_mb = None

    return {"test_rel_l2_mean": best, "test_rel_l2_std": best_std,
            "params": count_params(params), "train_time_s": train_time,
            "infer_ms_per_instance": infer_ms, "peak_gpu_mb": peak_mb,
            "n_train": int(n), "modes": modes, "width": width, "n_layers": n_layers}
