"""
E2 - Hybrid analytical differentiation vs pure automatic differentiation
========================================================================
Addresses: R1.2, R7.1, R10.2, R9.4 (differentiation part).

Two PDE-residual back-ends on the SAME SV-SNN architecture & initialization:
  (A) PURE-AD     : spatial 2nd derivatives via nested jax.grad (pointwise, vmap).
  (B) HYBRID/ANALYTIC: spatial 2nd derivatives via closed-form  X'' = -sum w^2 (a cos + b sin),
                       evaluated on a separable grid (the accelerated SV-SNN).

We sweep kappa in {20,40,60,80,100} * pi on 2D Helmholtz and report, per kappa:
  - best/final relative L2 error of each back-end (accuracy parity),
  - training wall-clock and ms/epoch (speed),
  - speedup = t_AD / t_hybrid,
  - a controlled derivative-accuracy probe: |u_xx(AD,float32) - u_xx(analytic,float64)|
    and |u_xx(AD,float64) - u_xx(analytic,float64)| vs frequency (numerical error growth).

Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad
import optax
from pyDOE import lhs

EPOCHS=6000; LR=1e-3; N_PDE=10000; N_BC=1024; N_TEST=200; EVAL_EVERY=100
NC_GRID=100; NUM_MODES=8; NUM_FREQ=64; SEED=0; SEEDS=[0,1,2]
KAPPAS={"20pi":20*np.pi,"40pi":40*np.pi,"60pi":60*np.pi,"80pi":80*np.pi,"100pi":100*np.pi}
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))
def gpu_mem():
    try: return float(jax.devices()[0].memory_stats().get("peak_bytes_in_use",0))/1e6
    except Exception: return float("nan")

def sample_freqs(key,K,wc):
    nl=K//4; ncc=K//2; nh=K-nl-ncc; _,k2,k3=jax.random.split(key,3)
    fl=jnp.linspace(1.0,wc,nl); fc=jnp.abs(jax.random.normal(k2,(ncc,))*30.0+wc)
    fh=jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)
    return jnp.sort(jnp.concatenate([fl,fc,fh]))

def init_svsnn(key,kappa):
    keys=jax.random.split(key,NUM_MODES*6+1); ki=0; sx,sy=[],[]
    for _ in range(NUM_MODES):
        sx.append({"freqs":sample_freqs(keys[ki],NUM_FREQ,kappa),"cos_c":jax.random.normal(keys[ki+1],(NUM_FREQ,))*0.1,
                   "sin_c":jax.random.normal(keys[ki+2],(NUM_FREQ,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        sy.append({"freqs":sample_freqs(keys[ki],NUM_FREQ,kappa),"cos_c":jax.random.normal(keys[ki+1],(NUM_FREQ,))*0.1,
                   "sin_c":jax.random.normal(keys[ki+2],(NUM_FREQ,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(NUM_MODES,))*0.1}

def make_data(kappa,seed):
    np.random.seed(seed)
    nps=N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    pde=lhs(2,samples=N_PDE)
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    ue=np.sin(kappa*X)*np.sin(kappa*Y)
    xc=np.linspace(0,1,NC_GRID).reshape(-1,1); yc=np.linspace(0,1,NC_GRID).reshape(-1,1)
    Xc,Yc=np.meshgrid(xc.flatten(),yc.flatten(),indexing="ij"); fg=kappa**2*np.sin(kappa*Xc)*np.sin(kappa*Yc)
    return {"xb":jnp.asarray(xb,jnp.float32),"yb":jnp.asarray(yb,jnp.float32),
            "x_pde":jnp.asarray(pde[:,0:1],jnp.float32),"y_pde":jnp.asarray(pde[:,1:2],jnp.float32),
            "X":X,"Y":Y,"ue":ue,"x_flat":jnp.asarray(X.reshape(-1,1),jnp.float32),"y_flat":jnp.asarray(Y.reshape(-1,1),jnp.float32),
            "xc":jnp.asarray(xc,jnp.float32),"yc":jnp.asarray(yc,jnp.float32),"fg":jnp.asarray(fg,jnp.float32),
            "x1d":jnp.asarray(x1.reshape(-1,1),jnp.float32),"y1d":jnp.asarray(y1.reshape(-1,1),jnp.float32)}

# ---- shared SV-SNN forward (pointwise) ----
def svsnn_fwd(p,x,y):
    u=jnp.zeros_like(x)
    for n in range(NUM_MODES):
        wx=p["spatial_x"][n]["freqs"][None,:]*x
        Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
        wy=p["spatial_y"][n]["freqs"][None,:]*y
        Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
        u=u+p["mode_coeffs"][n]*Xn*Yn
    return u

def predict_grid(p,x1d,y1d):
    fx=jnp.stack([p["spatial_x"][n]["freqs"] for n in range(NUM_MODES)])
    cx=jnp.stack([p["spatial_x"][n]["cos_c"] for n in range(NUM_MODES)])
    sx=jnp.stack([p["spatial_x"][n]["sin_c"] for n in range(NUM_MODES)])
    bx=jnp.stack([p["spatial_x"][n]["bias"] for n in range(NUM_MODES)])
    fy=jnp.stack([p["spatial_y"][n]["freqs"] for n in range(NUM_MODES)])
    cy=jnp.stack([p["spatial_y"][n]["cos_c"] for n in range(NUM_MODES)])
    sy=jnp.stack([p["spatial_y"][n]["sin_c"] for n in range(NUM_MODES)])
    by=jnp.stack([p["spatial_y"][n]["bias"] for n in range(NUM_MODES)])
    phx=x1d[:,:,None]*fx[None,:,:]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
    phy=y1d[:,:,None]*fy[None,:,:]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
    return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None,:]*Xv,Yv)

# ---- back-end B: hybrid/analytic on grid ----
def train_hybrid(data,kappa,seed):
    p=init_svsnn(random.PRNGKey(seed),kappa); n=cp(p)
    xc,yc,fg=data["xc"],data["yc"],data["fg"]; xb,yb=data["xb"],data["yb"]
    def residual(p):
        fx=jnp.stack([jax.lax.stop_gradient(p["spatial_x"][n]["freqs"]) for n in range(NUM_MODES)])
        cx=jnp.stack([p["spatial_x"][n]["cos_c"] for n in range(NUM_MODES)]); sx=jnp.stack([p["spatial_x"][n]["sin_c"] for n in range(NUM_MODES)]); bx=jnp.stack([p["spatial_x"][n]["bias"] for n in range(NUM_MODES)])
        fy=jnp.stack([jax.lax.stop_gradient(p["spatial_y"][n]["freqs"]) for n in range(NUM_MODES)])
        cy=jnp.stack([p["spatial_y"][n]["cos_c"] for n in range(NUM_MODES)]); sy=jnp.stack([p["spatial_y"][n]["sin_c"] for n in range(NUM_MODES)]); by=jnp.stack([p["spatial_y"][n]["bias"] for n in range(NUM_MODES)])
        phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
        phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
        mc=p["mode_coeffs"]; cX=mc[None]*Xv
        u=jnp.einsum("nm,jm->nj",cX,Yv); uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
        return -(uxx+uyy)-kappa**2*u-fg
    def loss(p):
        r=residual(p); bc=0.0
        for i in range(4): bc=bc+jnp.mean(svsnn_fwd(p,xb[i],yb[i])**2)
        return jnp.mean(r**2)+bc
    return _train(p,loss,data,n,"hybrid")

# ---- back-end A: pure AD pointwise ----
def train_ad(data,kappa,seed):
    p=init_svsnn(random.PRNGKey(seed),kappa); n=cp(p)
    xp=data["x_pde"].squeeze(); yp=data["y_pde"].squeeze(); xb,yb=data["xb"],data["yb"]
    def res_single(p,xs,ys):
        uf=lambda a,b: svsnn_fwd(p,a[None,None],b[None,None]).squeeze()
        uxx=jax.grad(jax.grad(uf,0),0)(xs,ys); uyy=jax.grad(jax.grad(uf,1),1)(xs,ys)
        u=uf(xs,ys); f=kappa**2*jnp.sin(kappa*xs)*jnp.sin(kappa*ys)
        return -(uxx+uyy)-kappa**2*u-f
    res_batch=vmap(res_single,in_axes=(None,0,0))
    def loss(p):
        r=res_batch(p,xp,yp); bc=0.0
        for i in range(4): bc=bc+jnp.mean(svsnn_fwd(p,xb[i],yb[i])**2)
        return jnp.mean(r**2)+bc
    return _train(p,loss,data,n,"pure_ad")

def _train(p,loss,data,nparams,tag):
    opt=optax.adam(LR); st=opt.init(p)
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); gnorm=optax.global_norm(g)
        u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l,gnorm
    for _ in range(2): p,st,_,_=step(p,st)
    best=float("inf"); bp=p; hist=[]; gnorms=[]
    t0=time.time()
    for ep in range(2,EPOCHS):
        p,st,l,gn=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            up=np.array(predict_grid(p,data["x1d"],data["y1d"])); e=l2(up,data["ue"])
            hist.append(e); gnorms.append(float(gn))
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    tt=time.time()-t0
    up=np.array(predict_grid(bp,data["x1d"],data["y1d"]))
    return {"tag":tag,"params":nparams,"time_s":tt,"ms_per_epoch":tt/(EPOCHS-2)*1000,
            "best_l2":best,"final_l2":hist[-1],"l2_hist":hist,"grad_norms":gnorms,
            "peak_mem_mb":gpu_mem(),"u_pred":up}

# ---- derivative-accuracy probe (no training) ----
def deriv_probe(kappa,seed):
    """Compare AD vs analytic 2nd derivative against float64 analytic ground truth."""
    p32=init_svsnn(random.PRNGKey(seed),kappa)
    p64=jax.tree.map(lambda z: z.astype(jnp.float64), p32)
    xs=jnp.linspace(0.01,0.99,256).reshape(-1,1); ys=jnp.full_like(xs,0.5)
    # analytic u_xx (single axis): sum_n c_n * X_n''(x) * Y_n(y)
    def analytic_uxx(p,x,y):
        u_xx=jnp.zeros_like(x)
        for n in range(NUM_MODES):
            wx=p["spatial_x"][n]["freqs"][None,:]*x
            Xdd=jnp.sum(-(p["spatial_x"][n]["freqs"]**2)*(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx)),axis=1,keepdims=True)
            wy=p["spatial_y"][n]["freqs"][None,:]*y
            Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
            u_xx=u_xx+p["mode_coeffs"][n]*Xdd*Yn
        return u_xx
    def ad_uxx(p,x,y):
        uf=lambda a,b: svsnn_fwd(p,a[None,None],b[None,None]).squeeze()
        return vmap(lambda a,b: jax.grad(jax.grad(uf,0),0)(a,b))(x.squeeze(),y.squeeze())
    a64=np.array(analytic_uxx(p64,xs.astype(jnp.float64),ys.astype(jnp.float64))).squeeze()
    a32=np.array(analytic_uxx(p32,xs,ys)).squeeze()
    ad32=np.array(ad_uxx(p32,xs,ys)).squeeze()
    ad64=np.array(ad_uxx(p64,xs.astype(jnp.float64),ys.astype(jnp.float64))).squeeze()
    den=np.linalg.norm(a64)+1e-30
    return {"analytic32_vs_truth":float(np.linalg.norm(a32-a64)/den),
            "ad32_vs_truth":float(np.linalg.norm(ad32-a64)/den),
            "ad64_vs_truth":float(np.linalg.norm(ad64-a64)/den)}

def main():
    jax.config.update("jax_enable_x64", True)  # for the deriv probe only
    results={}
    for name,kappa in KAPPAS.items():
        print(f"\n===== kappa={name} ({kappa:.2f}) =====",flush=True)
        probe=deriv_probe(kappa,SEED)
        rhs=[]; ras=[]
        for sd in SEEDS:
            rh=train_hybrid(make_data(kappa,sd),kappa,sd); ra=train_ad(make_data(kappa,sd),kappa,sd)
            rhs.append(rh); ras.append(ra)
            print(f"  seed={sd} hybrid best={rh['best_l2']:.3e} t={rh['time_s']:.1f}s | AD best={ra['best_l2']:.3e} t={ra['time_s']:.1f}s",flush=True)
        bh=np.array([r["best_l2"] for r in rhs]); ba=np.array([r["best_l2"] for r in ras])
        th=np.array([r["time_s"] for r in rhs]); ta=np.array([r["time_s"] for r in ras])
        speedup=float(ta.mean()/max(th.mean(),1e-9))
        print(f"  -> hybrid {bh.mean():.3e}+-{bh.std():.1e} (min {bh.min():.3e}) | AD {ba.mean():.3e}+-{ba.std():.1e} | speedup={speedup:.2f}x "
              f"| deriv ad32={probe['ad32_vs_truth']:.2e}",flush=True)
        results[name]={"kappa":float(kappa),"speedup":speedup,"deriv_probe":probe,
            "hybrid":{"best_l2":float(bh.mean()),"best_l2_std":float(bh.std()),"best_l2_min":float(bh.min()),
                      "time_s":float(th.mean()),"ms_per_epoch":float(np.mean([r["ms_per_epoch"] for r in rhs])),
                      "peak_mem_mb":float(np.mean([r["peak_mem_mb"] for r in rhs])),"l2_hist":rhs[0]["l2_hist"],"grad_norms":rhs[0]["grad_norms"]},
            "pure_ad":{"best_l2":float(ba.mean()),"best_l2_std":float(ba.std()),"best_l2_min":float(ba.min()),
                      "time_s":float(ta.mean()),"ms_per_epoch":float(np.mean([r["ms_per_epoch"] for r in ras])),
                      "peak_mem_mb":float(np.mean([r["peak_mem_mb"] for r in ras])),"l2_hist":ras[0]["l2_hist"],"grad_norms":ras[0]["grad_norms"]}}
        np.savez(os.path.join(SAVE,f"fields_{name}.npz"),u_hybrid=rhs[0]["u_pred"],u_ad=ras[0]["u_pred"],ue=make_data(kappa,0)["ue"])
    with open(os.path.join(SAVE,"E2_results.json"),"w") as f: json.dump(results,f,indent=2)
    print("\nSaved E2 to",SAVE)

if __name__=="__main__": main()
