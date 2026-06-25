"""
E4 - Frequency sampling strategy ablation
=========================================
Addresses: R7.2 (why three-level over two-level / continuous), R3.5 (why 40/40/20 for flows).

Same separable SV-SNN, only the FREQUENCY INITIALIZATION differs:
  S1 single   : all frequencies = w_char
  S2 two-level: 50% linear[1,w_char] + 50% N(w_char)
  S3 three-25 : 25% low / 50% char-Gaussian / 25% high   (paper default)
  S4 continuous: uniform broadband [1, 2 w_char]
  S5 three-40 : 40% low / 40% char / 20% high            (the "flow" allocation)

Two Poisson problems on [0,1]^2 (-Lap u = f, u=0 BC):
  (P1) pure high-frequency : u = sin(24pi x) sin(24pi y)
  (P2) multi-scale         : u = sin(4pi x) sin(4pi y) + sin(24pi x) sin(24pi y)
The multi-scale case mimics flow problems (significant low-freq energy), explaining
why emphasizing low/characteristic bands (40/40/20) helps there.

3 seeds, report mean+-std. Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax

EPOCHS=8000; LR=1e-3; N_BC=1024; N_TEST=200; EVAL_EVERY=100
NC=100; M=6; K=32; SEEDS=[0,1,2]; W_CHAR=24*np.pi
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))

# ---- frequency strategies ----
def freqs_strategy(key,strat,wc):
    k1,k2,k3=jax.random.split(key,3)
    if strat=="S1_single":
        return jnp.full((K,),wc)
    if strat=="S2_two":
        nl=K//2; return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(K-nl,))*30.0+wc)]))
    if strat=="S3_three25":
        nl=K//4; ncc=K//2; nh=K-nl-ncc
        return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*30.0+wc),jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)]))
    if strat=="S4_continuous":
        return jnp.sort(jax.random.uniform(k1,(K,),minval=1.0,maxval=2.0*wc))
    if strat=="S5_three40":
        nl=int(0.4*K); ncc=int(0.4*K); nh=K-nl-ncc
        return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*30.0+wc),jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)]))
    raise ValueError(strat)

def init_sep(key,strat,wc):
    keys=jax.random.split(key,M*6+1); ki=0; sx,sy=[],[]
    for _ in range(M):
        sx.append({"freqs":freqs_strategy(keys[ki],strat,wc),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        sy.append({"freqs":freqs_strategy(keys[ki],strat,wc),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(M,))*0.1}

def sep_fwd(p,x,y):
    u=jnp.zeros_like(x)
    for n in range(M):
        wx=p["spatial_x"][n]["freqs"][None,:]*x; Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
        wy=p["spatial_y"][n]["freqs"][None,:]*y; Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
        u=u+p["mode_coeffs"][n]*Xn*Yn
    return u
def stacks(p,axis):
    f=jnp.stack([jax.lax.stop_gradient(p[axis][n]["freqs"]) for n in range(M)])
    c=jnp.stack([p[axis][n]["cos_c"] for n in range(M)]); s=jnp.stack([p[axis][n]["sin_c"] for n in range(M)]); b=jnp.stack([p[axis][n]["bias"] for n in range(M)])
    return f,c,s,b
def sep_grid(p,x1d,y1d):
    fx,cx,sx,bx=stacks(p,"spatial_x"); fy,cy,sy,by=stacks(p,"spatial_y")
    phx=x1d[:,:,None]*fx[None]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
    phy=y1d[:,:,None]*fy[None]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
    return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None]*Xv,Yv)
def residual(p,xc,yc,fg):
    fx,cx,sx,bx=stacks(p,"spatial_x"); fy,cy,sy,by=stacks(p,"spatial_y")
    phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
    phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
    mc=p["mode_coeffs"]; cX=mc[None]*Xv
    uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
    return -(uxx+uyy)-fg   # Poisson: -Lap u = f

def make_poisson(kind,seed):
    np.random.seed(seed)
    nps=N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    xc=np.linspace(0,1,NC); yc=np.linspace(0,1,NC); Xc,Yc=np.meshgrid(xc,yc,indexing="ij")
    klo=4*np.pi; khi=24*np.pi
    if kind=="pure_high":
        ue=np.sin(khi*X)*np.sin(khi*Y); fg=2*khi**2*np.sin(khi*Xc)*np.sin(khi*Yc)
    else:
        ue=np.sin(klo*X)*np.sin(klo*Y)+np.sin(khi*X)*np.sin(khi*Y)
        fg=2*klo**2*np.sin(klo*Xc)*np.sin(klo*Yc)+2*khi**2*np.sin(khi*Xc)*np.sin(khi*Yc)
    return {"xb":jnp.asarray(xb,jnp.float32),"yb":jnp.asarray(yb,jnp.float32),
            "ue":ue,"X":X,"Y":Y,"xc":jnp.asarray(xc.reshape(-1,1),jnp.float32),"yc":jnp.asarray(yc.reshape(-1,1),jnp.float32),
            "fg":jnp.asarray(fg,jnp.float32),"x1d":jnp.asarray(x1.reshape(-1,1),jnp.float32),"y1d":jnp.asarray(y1.reshape(-1,1),jnp.float32)}

def train(strat,data,seed):
    p=init_sep(random.PRNGKey(seed),strat,W_CHAR)
    opt=optax.adam(LR); st=opt.init(p); xc,yc,fg=data["xc"],data["yc"],data["fg"]; xb,yb=data["xb"],data["yb"]
    def loss(p):
        r=residual(p,xc,yc,fg); bc=jnp.mean(sep_fwd(p,xb,yb)**2); return jnp.mean(r**2)+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); bp=p; hist=[]; t0=time.time()
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=l2(np.array(sep_grid(p,data["x1d"],data["y1d"])),data["ue"]); hist.append(e)
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    tt=time.time()-t0
    return {"strat":strat,"params":cp(p),"best_l2":best,"final_l2":hist[-1],"time_s":tt,"l2_hist":hist,
            "u_pred":np.array(sep_grid(bp,data["x1d"],data["y1d"]))}

STRATS=["S1_single","S2_two","S3_three25","S4_continuous","S5_three40"]
def main():
    out={}
    for kind in ["pure_high","multi_scale"]:
        print(f"\n######### problem={kind} #########",flush=True); out[kind]={}
        for s in STRATS:
            recs=[]
            for sd in SEEDS:
                r=train(s,make_poisson(kind,sd),sd); recs.append(r)
                print(f"  {kind} {s} seed={sd}: best={r['best_l2']:.3e} t={r['time_s']:.1f}s",flush=True)
            b=np.array([r["best_l2"] for r in recs])
            out[kind][s]={"params":recs[0]["params"],"best_l2_mean":float(b.mean()),"best_l2_std":float(b.std()),
                          "best_l2_min":float(b.min()),"l2_hist":recs[0]["l2_hist"]}
    with open(os.path.join(SAVE,"E4_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E4 to",SAVE)

if __name__=="__main__": main()
