"""
E1 - Fair & Complete Comparison (Accelerated SV-SNN)
=====================================================
Addresses reviewer themes:
  - R3.1 / R5.2 / R7.3 : baseline fairness (all Fourier-based baselines receive the
                          SAME characteristic-frequency initialization as SV-SNN).
  - R5.7 / R9.5        : complete computational report (params, ms/epoch, wall-clock,
                          peak GPU memory, #collocation points, inference time,
                          whether SVD cost is included).
  - R3.4               : single-run vs averaged vs diagnostic. We run 5 seeds and report
                          best / final / mean +- std to explain Table 2/10/11 differences.
  - R5.3 / R9.1        : a CLASSICAL Fourier spectral (Galerkin) reference solver is added
                          as an accuracy upper bound for the (separable) manufactured solution.

Problem family: 2D Helmholtz  -Laplacian(u) - kappa^2 u = f  on [0,1]^2, u=0 on boundary,
                exact u = sin(kappa x) sin(kappa y),  f = kappa^2 sin(kappa x) sin(kappa y).
Cases: kappa = 24*pi and kappa = 48*pi  (exactly the cases reviewers discuss).

All methods: SV-SNN (accelerated), SPINN, SIREN, FourierPINN, PINN, Classical-Spectral.
Self-contained: no imports from sibling project files.
"""

import os, sys, time, json, csv
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, jvp, value_and_grad
from jax.flatten_util import ravel_pytree
import optax
from pyDOE import lhs

# ------------------------------------------------------------------
# Global config (fair, identical across methods)
# ------------------------------------------------------------------
EPOCHS    = 10000
LR        = 1e-3
N_PDE     = 10000
N_BC      = 1024
N_TEST    = 256
EVAL_EVERY= 100
NC_SPINN  = 100
NC_GRID   = 100          # accelerated SV-SNN separable training grid
FF_DIM    = 64
SEEDS     = [0, 1, 2, 3, 4]
CASES     = {"helmholtz24pi": 24.0*np.pi, "helmholtz48pi": 48.0*np.pi}
# Per-case SV-SNN config matching the best-known accelerated configuration
# (preserves SV-SNN's best accuracy & speed, requirement #4):
#   kappa=24pi -> 6 modes x 32 freqs (params ~1170);  kappa=48pi -> 8 modes x 64 freqs (~3096)
SVSNN_CFG = {"helmholtz24pi": (6, 32), "helmholtz48pi": (8, 64)}

SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")
os.makedirs(SAVE_DIR, exist_ok=True)


def gpu_peak_mb():
    try:
        s = jax.devices()[0].memory_stats()
        return float(s.get("peak_bytes_in_use", 0)) / 1e6
    except Exception:
        return float("nan")


def l2_rel(u_pred, u_exact):
    return float(np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))


def count_params(params):
    return int(sum(p.size for p in jax.tree.leaves(params) if hasattr(p, "size")))


def hvp_fwdfwd(f, primals, tangents):
    g = lambda p: jvp(f, (p,), (tangents,))[1]
    return jvp(g, (primals,), (tangents,))[1]


def sample_frequencies(key, K, w_char):
    """3-level multi-level sampling shared by SV-SNN AND the Fourier baselines (fairness)."""
    n_low = K // 4
    n_char = K // 2
    n_high = K - n_low - n_char
    _, k2, k3 = jax.random.split(key, 3)
    freqs_low = jnp.linspace(1.0, w_char, n_low)
    freqs_char = jnp.abs(jax.random.normal(k2, (n_char,)) * 30.0 + w_char)
    freqs_high = jax.random.uniform(k3, (n_high,), minval=w_char * 0.5, maxval=w_char)
    return jnp.sort(jnp.concatenate([freqs_low, freqs_char, freqs_high]))


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def generate_data(kappa, seed):
    np.random.seed(seed)
    n_per = N_BC // 4
    t = np.linspace(0, 1, n_per).reshape(-1, 1)
    x_bc = np.vstack([np.zeros((n_per,1)), np.ones((n_per,1)), t, t])
    y_bc = np.vstack([t, t, np.zeros((n_per,1)), np.ones((n_per,1))])
    u_bc = np.zeros((x_bc.shape[0], 1))
    pde = lhs(2, samples=N_PDE)
    x_pde, y_pde = pde[:, 0:1], pde[:, 1:2]
    x1 = np.linspace(0, 1, N_TEST); y1 = np.linspace(0, 1, N_TEST)
    X, Y = np.meshgrid(x1, y1, indexing="ij")
    u_ex = np.sin(kappa*X) * np.sin(kappa*Y)
    d = {
        "x_bc": jnp.asarray(x_bc, jnp.float32), "y_bc": jnp.asarray(y_bc, jnp.float32),
        "u_bc": jnp.asarray(u_bc, jnp.float32),
        "x_pde": jnp.asarray(x_pde, jnp.float32), "y_pde": jnp.asarray(y_pde, jnp.float32),
        "X": X, "Y": Y, "u_ex": u_ex,
        "x_flat": jnp.asarray(X.reshape(-1,1), jnp.float32),
        "y_flat": jnp.asarray(Y.reshape(-1,1), jnp.float32),
        "x_1d": jnp.asarray(x1.reshape(-1,1), jnp.float32),
        "y_1d": jnp.asarray(y1.reshape(-1,1), jnp.float32),
    }
    xc = np.linspace(0,1,NC_SPINN).reshape(-1,1); yc = np.linspace(0,1,NC_SPINN).reshape(-1,1)
    Xc, Yc = np.meshgrid(xc.flatten(), yc.flatten(), indexing="ij")
    f_grid = kappa**2 * np.sin(kappa*Xc) * np.sin(kappa*Yc)
    d["spinn"] = {
        "xc": jnp.asarray(xc, jnp.float32), "yc": jnp.asarray(yc, jnp.float32),
        "f_grid": jnp.asarray(f_grid, jnp.float32),
        "xb": [jnp.zeros((NC_SPINN,1),jnp.float32), jnp.ones((NC_SPINN,1),jnp.float32),
                jnp.asarray(xc,jnp.float32), jnp.asarray(xc,jnp.float32)],
        "yb": [jnp.asarray(yc,jnp.float32), jnp.asarray(yc,jnp.float32),
                jnp.zeros((NC_SPINN,1),jnp.float32), jnp.ones((NC_SPINN,1),jnp.float32)],
    }
    return d


# ------------------------------------------------------------------
# Method 1: SV-SNN (accelerated) - analytic spatial 2nd derivatives + separable grid
# ------------------------------------------------------------------
def run_svsnn_accel(data, kappa, seed):
    NUM_MODES, NUM_FREQ = data["_svsnn_cfg"]
    def init(key):
        keys = jax.random.split(key, NUM_MODES*6+1); ki=0
        sx, sy = [], []
        for _ in range(NUM_MODES):
            fx = sample_frequencies(keys[ki], NUM_FREQ, kappa); ki+=1
            cx = jax.random.normal(keys[ki],(NUM_FREQ,))*0.1; ki+=1
            sxx= jax.random.normal(keys[ki],(NUM_FREQ,))*0.1; ki+=1
            sx.append({"freqs":fx,"cos_c":cx,"sin_c":sxx,"bias":jnp.zeros(1)})
            fy = sample_frequencies(keys[ki], NUM_FREQ, kappa); ki+=1
            cy = jax.random.normal(keys[ki],(NUM_FREQ,))*0.1; ki+=1
            syy= jax.random.normal(keys[ki],(NUM_FREQ,))*0.1; ki+=1
            sy.append({"freqs":fy,"cos_c":cy,"sin_c":syy,"bias":jnp.zeros(1)})
        mc = jax.random.normal(keys[ki],(NUM_MODES,))*0.1
        return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":mc}

    def stack(p, k):
        f = jnp.stack([jax.lax.stop_gradient(p[k][n]["freqs"]) for n in range(NUM_MODES)])
        c = jnp.stack([p[k][n]["cos_c"] for n in range(NUM_MODES)])
        s = jnp.stack([p[k][n]["sin_c"] for n in range(NUM_MODES)])
        b = jnp.stack([p[k][n]["bias"] for n in range(NUM_MODES)])
        return f, c, s, b
    def basis(coord, f, c, s, b):
        ph = coord[:,:,None]*f[None,:,:]
        cp, sp = jnp.cos(ph), jnp.sin(ph)
        trig = c[None,:,:]*cp + s[None,:,:]*sp
        return jnp.sum(trig,axis=2)+b[None,:,0], trig
    def d2(trig, f):
        return jnp.sum(-(f[None,:,:]**2)*trig, axis=2)
    def fwd_pts(p, x, y):
        fx,cx,sx,bx = stack(p,"spatial_x"); fy,cy,sy,by = stack(p,"spatial_y")
        Xv,_ = basis(x,fx,cx,sx,bx); Yv,_ = basis(y,fy,cy,sy,by)
        return jnp.sum(p["mode_coeffs"][None,:]*Xv*Yv, axis=1, keepdims=True)
    def fwd_grid(p, x, y):
        fx,cx,sx,bx = stack(p,"spatial_x"); fy,cy,sy,by = stack(p,"spatial_y")
        Xv,_ = basis(x,fx,cx,sx,bx); Yv,_ = basis(y,fy,cy,sy,by)
        cX = p["mode_coeffs"][None,:]*Xv
        return jnp.einsum("nm,jm->nj", cX, Yv)
    sp = data["spinn"]; xc, yc, fg = sp["xc"], sp["yc"], sp["f_grid"]
    def residual(p):
        fx,cx,sx,bx = stack(p,"spatial_x"); fy,cy,sy,by = stack(p,"spatial_y")
        Xv,Xt = basis(xc,fx,cx,sx,bx); Yv,Yt = basis(yc,fy,cy,sy,by)
        Xdd = d2(Xt,fx); Ydd = d2(Yt,fy); mc = p["mode_coeffs"]
        cX = mc[None,:]*Xv
        u   = jnp.einsum("nm,jm->nj", cX, Yv)
        uxx = jnp.einsum("nm,jm->nj", mc[None,:]*Xdd, Yv)
        uyy = jnp.einsum("nm,jm->nj", cX, Ydd)
        return -(uxx+uyy) - kappa**2*u - fg
    def bcloss(p):
        xb, yb = sp["xb"], sp["yb"]; loss = 0.0
        for i in range(4):
            loss = loss + jnp.mean(fwd_pts(p, xb[i], yb[i])**2)
        return loss
    def loss_fn(p):
        r = residual(p); return jnp.mean(r**2)+bcloss(p), (jnp.mean(r**2), bcloss(p))
    return _train_grid(init, loss_fn, fwd_grid, data, seed, "SVSNN_accel")


# ------------------------------------------------------------------
# Method 2: SPINN  (separable modified-MLP + shared freq embedding)
# ------------------------------------------------------------------
def run_spinn(data, kappa, seed):
    feats, nlay, r = 64, 4, 64
    def init_branch(key, din):
        keys = random.split(key, 3+nlay+1)
        sc = 1.0/jnp.sqrt(jnp.float32(din))
        p = {"U_w":random.normal(keys[0],(din,feats))*sc,"U_b":jnp.zeros((feats,)),
             "V_w":random.normal(keys[1],(din,feats))*sc,"V_b":jnp.zeros((feats,)),
             "H_w":random.normal(keys[2],(din,feats))*sc,"H_b":jnp.zeros((feats,)),
             "layers":[],"out_w":random.normal(keys[-1],(feats,r))*(1.0/jnp.sqrt(jnp.float32(feats)))}
        for i in range(nlay):
            p["layers"].append({"w":random.normal(keys[3+i],(feats,feats))*(1.0/jnp.sqrt(jnp.float32(feats))),
                                "b":jnp.zeros((feats,))})
        return p
    def init(key):
        k1,k2,k3,k4 = random.split(key,4)
        return {"branch_x":init_branch(k1,2*FF_DIM),"branch_y":init_branch(k2,2*FF_DIM),
                "W_x":sample_frequencies(k3,FF_DIM,kappa).reshape(1,-1),
                "W_y":sample_frequencies(k4,FF_DIM,kappa).reshape(1,-1)}
    def embed(c,W): return jnp.concatenate([jnp.sin(c@W),jnp.cos(c@W)],axis=-1)
    def branch(p,x):
        U=jnp.tanh(x@p["U_w"]+p["U_b"]); V=jnp.tanh(x@p["V_w"]+p["V_b"]); H=jnp.tanh(x@p["H_w"]+p["H_b"])
        for L in p["layers"]:
            Z=jnp.tanh(H@L["w"]+L["b"]); H=(1.0-Z)*U+Z*V
        return H@p["out_w"]
    def fwd(p,x,y):
        return branch(p["branch_x"],embed(x,p["W_x"]))@branch(p["branch_y"],embed(y,p["W_y"])).T
    sp=data["spinn"]; xc,yc,fg=sp["xc"],sp["yc"],sp["f_grid"]; xb,yb=sp["xb"],sp["yb"]
    def loss_fn(p):
        uxx=hvp_fwdfwd(lambda xx:fwd(p,xx,yc), xc, jnp.ones_like(xc))
        uyy=hvp_fwdfwd(lambda yy:fwd(p,xc,yy), yc, jnp.ones_like(yc)).T
        u=fwd(p,xc,yc); r=-(uxx+uyy)-kappa**2*u-fg; pde=jnp.mean(r**2)
        bc=0.0
        for i in range(4): bc=bc+jnp.mean(fwd(p,xb[i],yb[i])**2)
        return pde+bc,(pde,bc)
    return _train_grid(init, loss_fn, fwd, data, seed, "SPINN")


# ------------------------------------------------------------------
# Methods 3-5: SIREN / FourierPINN / PINN (pointwise, hvp residual)
# ------------------------------------------------------------------
def _pointwise_runner(name, init, fwd_xy, data, kappa, seed):
    """fwd_xy(params, xy[N,2]) -> [N,1]"""
    x_pde,y_pde=data["x_pde"],data["y_pde"]; x_bc,y_bc,u_bc=data["x_bc"],data["y_bc"],data["u_bc"]
    xy_pde=jnp.concatenate([x_pde,y_pde],axis=-1)
    tx=jnp.zeros_like(xy_pde).at[:,0].set(1.0); ty=jnp.zeros_like(xy_pde).at[:,1].set(1.0)
    def loss_fn(p):
        ub=fwd_xy(p,jnp.concatenate([x_bc,y_bc],axis=-1)); bc=jnp.mean((ub-u_bc)**2)
        uxx=hvp_fwdfwd(lambda z:fwd_xy(p,z),xy_pde,tx)
        uyy=hvp_fwdfwd(lambda z:fwd_xy(p,z),xy_pde,ty)
        u=fwd_xy(p,xy_pde); f=kappa**2*jnp.sin(kappa*x_pde)*jnp.sin(kappa*y_pde)
        r=-(uxx+uyy)-kappa**2*u-f; pde=jnp.mean(r**2)
        return pde+bc,(pde,bc)
    def predict_flat(p):
        xy=jnp.concatenate([data["x_flat"],data["y_flat"]],axis=-1)
        return np.array(fwd_xy(p,xy)).reshape(N_TEST,N_TEST)
    return _train_pointwise(init, loss_fn, predict_flat, data, seed, name)

def run_siren(data, kappa, seed):
    def init(key):
        k1,k2,key=random.split(key,3)
        Wx=sample_frequencies(k1,FF_DIM,kappa).reshape(1,-1); Wy=sample_frequencies(k2,FF_DIM,kappa).reshape(1,-1)
        dims=[4*FF_DIM,128,128,128,128,1]; layers=[]
        for i in range(len(dims)-1):
            k,key=random.split(key); std=jnp.sqrt(2.0/dims[i])
            layers.append({"w":random.normal(k,(dims[i],dims[i+1]))*std,"b":jnp.zeros((dims[i+1],))})
        return {"layers":layers,"W_x":Wx,"W_y":Wy}
    def fwd(p,xy):
        x,y=xy[:,0:1],xy[:,1:2]
        Hx=jnp.concatenate([jnp.sin(x@p["W_x"]),jnp.cos(x@p["W_x"])],axis=-1)
        Hy=jnp.concatenate([jnp.sin(y@p["W_y"]),jnp.cos(y@p["W_y"])],axis=-1)
        h=jnp.concatenate([Hx,Hy],axis=-1); n=len(p["layers"])
        for i,L in enumerate(p["layers"]):
            h=h@L["w"]+L["b"]
            if i<n-1: h=jnp.sin(h)
        return h
    return _pointwise_runner("SIREN", init, fwd, data, kappa, seed)

def run_fourierpinn(data, kappa, seed):
    def init(key):
        k1,k2,key=random.split(key,3)
        Wx=sample_frequencies(k1,FF_DIM,kappa).reshape(1,-1); Wy=sample_frequencies(k2,FF_DIM,kappa).reshape(1,-1)
        dims=[4*FF_DIM,128,128,128,1]; layers=[]
        for i in range(len(dims)-1):
            k,key=random.split(key); lim=jnp.sqrt(6.0/(dims[i]+dims[i+1]))
            layers.append({"w":random.uniform(k,(dims[i],dims[i+1]),minval=-lim,maxval=lim),"b":jnp.zeros((dims[i+1],))})
        return {"W_x":Wx,"W_y":Wy,"mlp":layers}
    def fwd(p,xy):
        x,y=xy[:,0:1],xy[:,1:2]; Wx=jax.lax.stop_gradient(p["W_x"]); Wy=jax.lax.stop_gradient(p["W_y"])
        hx=jnp.concatenate([jnp.sin(x@Wx),jnp.cos(x@Wx)],axis=-1)
        hy=jnp.concatenate([jnp.sin(y@Wy),jnp.cos(y@Wy)],axis=-1)
        h=jnp.concatenate([hx,hy],axis=-1); n=len(p["mlp"])
        for i,L in enumerate(p["mlp"]):
            h=h@L["w"]+L["b"]
            if i<n-1: h=jnp.tanh(h)
        return h
    return _pointwise_runner("FourierPINN", init, fwd, data, kappa, seed)

def run_pinn(data, kappa, seed):
    def init(key):
        dims=[2,128,128,128,128,1]; layers=[]; key0=key
        for i in range(len(dims)-1):
            k,key0=random.split(key0); lim=jnp.sqrt(6.0/(dims[i]+dims[i+1]))
            layers.append({"w":random.uniform(k,(dims[i],dims[i+1]),minval=-lim,maxval=lim),"b":jnp.zeros((dims[i+1],))})
        return {"layers":layers}
    def fwd(p,xy):
        h=xy; n=len(p["layers"])
        for i,L in enumerate(p["layers"]):
            h=h@L["w"]+L["b"]
            if i<n-1: h=jnp.tanh(h)
        return h
    return _pointwise_runner("PINN", init, fwd, data, kappa, seed)


# ------------------------------------------------------------------
# Method 6: Classical Fourier spectral (Galerkin) reference  (R5.3 / R9.1)
# ------------------------------------------------------------------
def run_classical_spectral(data, kappa, seed):
    """Sine-basis Galerkin solver: u = sum_{m,n} c_mn sin(m pi x) sin(n pi y).
    For -Delta u - kappa^2 u = f with f given on a grid, project f onto sine modes
    and solve (lambda_mn - kappa^2) c_mn = f_mn analytically. This is the classical
    spectral method and is essentially exact for separable manufactured solutions.
    Reported as an accuracy upper bound, NOT a learning method."""
    M = 64  # number of sine modes per dimension
    X, Y, u_ex = data["X"], data["Y"], data["u_ex"]
    Ng = N_TEST
    x = X[:, 0]; y = Y[0, :]
    t0 = time.time()
    # source on test grid
    f = (kappa**2) * np.sin(kappa*X) * np.sin(kappa*Y)
    ms = np.arange(1, M+1)
    Sx = np.sin(np.outer(x, ms*np.pi))      # (Ng, M)
    Sy = np.sin(np.outer(y, ms*np.pi))      # (Ng, M)
    # mode coefficients of f via discrete inner products (orthogonality of sines on [0,1])
    # f_mn = 4 * <f, sin(m pi x) sin(n pi y)>
    norm = (Sx.T @ Sx) / Ng  # ~0.5 I
    # project: F_mn = (Sx^T f Sy) * (2/Ng)^2 with diagonal normalization
    Fmn = (Sx.T @ f @ Sy) * (2.0/Ng) * (2.0/Ng) / (4*norm[0,0]*norm[0,0])
    lam = (np.add.outer(ms**2, ms**2)) * (np.pi**2)   # (M,M)
    denom = lam - kappa**2
    Cmn = np.where(np.abs(denom) > 1e-8, Fmn/denom, 0.0)
    u_pred = Sx @ Cmn @ Sy.T
    solve_time = time.time() - t0
    err = l2_rel(u_pred, u_ex)
    return {"method":"ClassicalSpectral","total_params":M*M,"total_time_sec":solve_time,
            "ms_per_epoch":0.0,"best_l2_error":err,"final_l2_error":err,
            "peak_mem_mb":gpu_peak_mb(),"inference_time_ms":solve_time*1000,
            "n_collocation":Ng*Ng,"includes_svd_cost":False,"u_pred":u_pred}


# ------------------------------------------------------------------
# Shared training loops
# ------------------------------------------------------------------
def _train_grid(init_fn, loss_fn, fwd_grid, data, seed, name):
    key = random.PRNGKey(seed)
    params = init_fn(key); n_params = count_params(params)
    opt = optax.adam(LR); opt_state = opt.init(params)
    @jit
    def step(p, s):
        (loss,(pde,bc)), g = value_and_grad(loss_fn, has_aux=True)(p)
        upd, s = opt.update(g, s, p); return optax.apply_updates(p, upd), s, loss
    best=float("inf"); best_p=params; l2_hist=[]
    for _ in range(2): params,opt_state,_=step(params,opt_state)
    t0=time.time()
    for ep in range(2,EPOCHS):
        params,opt_state,_=step(params,opt_state)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            up=np.array(fwd_grid(params,data["x_1d"],data["y_1d"]))
            e=l2_rel(up,data["u_ex"]); l2_hist.append(e)
            if e<best: best=e; best_p=jax.tree.map(lambda z:z.copy(),params)
    train_time=time.time()-t0
    up=np.array(fwd_grid(best_p,data["x_1d"],data["y_1d"]))
    # inference time
    fwd_grid(best_p,data["x_1d"],data["y_1d"]).block_until_ready()
    ti=time.time()
    for _ in range(10):
        pr=fwd_grid(best_p,data["x_1d"],data["y_1d"])
    pr.block_until_ready()
    inf_ms=(time.time()-ti)/10*1000
    return {"method":name,"total_params":n_params,"total_time_sec":train_time,
            "ms_per_epoch":train_time/(EPOCHS-2)*1000,"best_l2_error":best,
            "final_l2_error":l2_hist[-1],"peak_mem_mb":gpu_peak_mb(),
            "inference_time_ms":inf_ms,"n_collocation":NC_GRID*NC_GRID,
            "includes_svd_cost":False,"u_pred":up}

def _train_pointwise(init_fn, loss_fn, predict_flat, data, seed, name):
    key=random.PRNGKey(seed); params=init_fn(key); n_params=count_params(params)
    opt=optax.adam(LR); opt_state=opt.init(params)
    @jit
    def step(p,s):
        (loss,(pde,bc)),g=value_and_grad(loss_fn,has_aux=True)(p)
        upd,s=opt.update(g,s,p); return optax.apply_updates(p,upd),s,loss
    best=float("inf"); best_p=params; l2_hist=[]
    for _ in range(2): params,opt_state,_=step(params,opt_state)
    t0=time.time()
    for ep in range(2,EPOCHS):
        params,opt_state,_=step(params,opt_state)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=l2_rel(predict_flat(params),data["u_ex"]); l2_hist.append(e)
            if e<best: best=e; best_p=jax.tree.map(lambda z:z.copy(),params)
    train_time=time.time()-t0
    up=predict_flat(best_p)
    ti=time.time()
    for _ in range(10): _=predict_flat(best_p)
    inf_ms=(time.time()-ti)/10*1000
    return {"method":name,"total_params":n_params,"total_time_sec":train_time,
            "ms_per_epoch":train_time/(EPOCHS-2)*1000,"best_l2_error":best,
            "final_l2_error":l2_hist[-1],"peak_mem_mb":gpu_peak_mb(),
            "inference_time_ms":inf_ms,"n_collocation":N_PDE,
            "includes_svd_cost":False,"u_pred":up}


# ------------------------------------------------------------------
# Jacobian-SVD effective-rank cost probe (R5.7: is SVD cost included?)
# ------------------------------------------------------------------
def svd_cost_probe(data, kappa, seed):
    """Measure the wall-clock cost of ONE Jacobian-SVD effective-rank computation
    for SV-SNN, to report it separately (it is a diagnostic, not part of training)."""
    NUM_MODES, NUM_FREQ = 8, FF_DIM
    key=random.PRNGKey(seed); keys=jax.random.split(key,NUM_MODES*6+1); ki=0
    sx,sy=[],[]
    for _ in range(NUM_MODES):
        sx.append({"freqs":sample_frequencies(keys[ki],NUM_FREQ,kappa),"cos_c":jax.random.normal(keys[ki+1],(NUM_FREQ,))*0.1,
                   "sin_c":jax.random.normal(keys[ki+2],(NUM_FREQ,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        sy.append({"freqs":sample_frequencies(keys[ki],NUM_FREQ,kappa),"cos_c":jax.random.normal(keys[ki+1],(NUM_FREQ,))*0.1,
                   "sin_c":jax.random.normal(keys[ki+2],(NUM_FREQ,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    params={"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(NUM_MODES,))*0.1}
    flat,unravel=ravel_pytree(params)
    xs=jnp.linspace(0,1,40); ys=jnp.linspace(0,1,40)
    XX,YY=jnp.meshgrid(xs,ys,indexing="ij"); xf=XX.reshape(-1,1); yf=YY.reshape(-1,1)
    def fwd(fp,x,y):
        p=unravel(fp); u=jnp.zeros_like(x)
        for n in range(NUM_MODES):
            wx=p["spatial_x"][n]["freqs"][None,:]*x
            Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
            wy=p["spatial_y"][n]["freqs"][None,:]*y
            Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
            u=u+p["mode_coeffs"][n]*Xn*Yn
        return u.squeeze()
    J=jax.jacrev(lambda fp:fwd(fp,xf,yf))(flat)
    J.block_until_ready()
    t0=time.time()
    Jm=np.array(J); s=np.linalg.svd(Jm,compute_uv=False)
    cost=time.time()-t0
    e2=s**2; r_eff=int(np.searchsorted(np.cumsum(e2)/np.sum(e2),0.99)+1)
    return {"svd_cost_sec":cost,"effective_rank":r_eff,"jacobian_shape":list(Jm.shape)}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
METHODS = [("SVSNN_accel",run_svsnn_accel),("SPINN",run_spinn),("SIREN",run_siren),
           ("FourierPINN",run_fourierpinn),("PINN",run_pinn),("ClassicalSpectral",run_classical_spectral)]

def main():
    all_rows=[]; per_seed_records={}
    for cname, kappa in CASES.items():
        print(f"\n########## CASE {cname}  kappa={kappa:.3f} ##########")
        per_seed_records[cname]={}
        for mname, fn in METHODS:
            seeds = [0] if mname=="ClassicalSpectral" else SEEDS
            recs=[]
            for sd in seeds:
                data=generate_data(kappa, sd)
                data["_svsnn_cfg"]=SVSNN_CFG[cname]
                print(f"  [{cname}] {mname} seed={sd} ...", flush=True)
                r=fn(data, kappa, sd)
                recs.append(r)
                print(f"     best L2={r['best_l2_error']:.4e} time={r['total_time_sec']:.1f}s "
                      f"mem={r['peak_mem_mb']:.0f}MB params={r['total_params']}", flush=True)
            per_seed_records[cname][mname]=[{k:v for k,v in r.items() if k!='u_pred'} for r in recs]
            # save a representative prediction (seed 0)
            np.savez(os.path.join(SAVE_DIR,f"{cname}_{mname}_pred.npz"),
                     u_pred=recs[0]["u_pred"], u_exact=data["u_ex"], X=data["X"], Y=data["Y"])
            best=np.array([r["best_l2_error"] for r in recs])
            final=np.array([r["final_l2_error"] for r in recs])
            tt=np.array([r["total_time_sec"] for r in recs])
            mse=np.array([r["ms_per_epoch"] for r in recs])
            mem=np.array([r["peak_mem_mb"] for r in recs])
            inf=np.array([r["inference_time_ms"] for r in recs])
            row={"case":cname,"method":mname,"n_seeds":len(seeds),
                 "params":recs[0]["total_params"],"n_collocation":recs[0]["n_collocation"],
                 "best_l2_mean":float(best.mean()),"best_l2_std":float(best.std()),
                 "best_l2_min":float(best.min()),
                 "final_l2_mean":float(final.mean()),"final_l2_std":float(final.std()),
                 "time_mean_s":float(tt.mean()),"time_std_s":float(tt.std()),
                 "ms_per_epoch_mean":float(mse.mean()),
                 "peak_mem_mb_mean":float(mem.mean()),
                 "inference_ms_mean":float(inf.mean())}
            all_rows.append(row)
        # SVD cost probe per case
        probe=svd_cost_probe(generate_data(kappa,0), kappa, 0)
        per_seed_records[cname]["_svd_probe"]=probe
        print(f"  [SVD probe] cost={probe['svd_cost_sec']*1000:.1f}ms r_eff={probe['effective_rank']} "
              f"J shape={probe['jacobian_shape']}")

    with open(os.path.join(SAVE_DIR,"per_seed_records.json"),"w") as f:
        json.dump(per_seed_records,f,indent=2)
    with open(os.path.join(SAVE_DIR,"summary_meanstd.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0].keys())); w.writeheader()
        for row in all_rows: w.writerow(row)
    print("\nSaved E1 results to",SAVE_DIR)

if __name__=="__main__":
    main()
