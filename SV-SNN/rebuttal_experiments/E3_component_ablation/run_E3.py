"""
E3 - Component-wise ablation of SV-SNN
======================================
Addresses: R7.5 (which module drives the gain; additive or dominant), R4.2, R10.1
("is it just an initialization strategy?").

On 2D Helmholtz kappa=24pi (separable) AND a weakly non-separable problem,
we toggle ONE design component at a time and measure accuracy / params / time:

  V0 Full        : separable + Fourier spectral basis + 3-level sampling + analytic deriv
  V1 -Separation : coupled (non-separable) 2D Fourier network (analytic deriv)
  V2 -Spectral   : separable + per-axis tanh-MLP basis (no Fourier) + AD deriv
  V3 -Multilevel : separable + Fourier + RANDOM uniform freq init + analytic deriv
  V4 -Analytic   : separable + Fourier + 3-level + PURE-AD deriv (isolates differentiation)

Run over 3 seeds, report mean+-std. Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad
import optax
from pyDOE import lhs

EPOCHS=8000; LR=1e-3; N_PDE=10000; N_BC=1024; N_TEST=200; EVAL_EVERY=100
NC=100; M=6; K=32; SEEDS=[0,1,2]
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))
def sfreqs(key,Kk,wc):
    nl=Kk//4; ncc=Kk//2; nh=Kk-nl-ncc; _,k2,k3=jax.random.split(key,3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*30.0+wc),
                                     jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)]))

# ---- problems ----
def make_problem(kind, kappa, seed):
    np.random.seed(seed)
    nps=N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    xc=np.linspace(0,1,NC); yc=np.linspace(0,1,NC); Xc,Yc=np.meshgrid(xc,yc,indexing="ij")
    if kind=="separable":      # u=sin(kx)sin(ky)
        ue=np.sin(kappa*X)*np.sin(kappa*Y); fg=kappa**2*np.sin(kappa*Xc)*np.sin(kappa*Yc)
        ub=np.zeros((xb.shape[0],1))
        def uex(xx,yy): return np.sin(kappa*xx)*np.sin(kappa*yy)
    else:                       # weakly non-separable: u=sin(k x)sin(k y)+0.5 sin(k(x+y))
        ue=np.sin(kappa*X)*np.sin(kappa*Y)+0.5*np.sin(kappa*(X+Y))
        # -Lap u - k^2 u = f ; Lap[sin(kx)sin(ky)]=-2k^2 sin sin ; Lap[sin(k(x+y))]=-2k^2 sin(k(x+y))
        lap=-2*kappa**2*(np.sin(kappa*Xc)*np.sin(kappa*Yc)+0.5*np.sin(kappa*(Xc+Yc)))
        usep=np.sin(kappa*Xc)*np.sin(kappa*Yc)+0.5*np.sin(kappa*(Xc+Yc))
        fg=-lap-kappa**2*usep
        def uex(xx,yy): return np.sin(kappa*xx)*np.sin(kappa*yy)+0.5*np.sin(kappa*(xx+yy))
        ub=uex(xb,yb)
    return {"xb":jnp.asarray(xb,jnp.float32),"yb":jnp.asarray(yb,jnp.float32),"ub":jnp.asarray(ub,jnp.float32),
            "X":X,"Y":Y,"ue":ue,"xc":jnp.asarray(xc.reshape(-1,1),jnp.float32),"yc":jnp.asarray(yc.reshape(-1,1),jnp.float32),
            "fg":jnp.asarray(fg,jnp.float32),"x1d":jnp.asarray(x1.reshape(-1,1),jnp.float32),"y1d":jnp.asarray(y1.reshape(-1,1),jnp.float32),
            "x_flat":jnp.asarray(X.reshape(-1,1),jnp.float32),"y_flat":jnp.asarray(Y.reshape(-1,1),jnp.float32),
            "kappa":kappa}

# ===== separable spectral core (V0,V3,V4) =====
def init_sep(key,kappa,multilevel=True):
    keys=jax.random.split(key,M*6+1); ki=0; sx,sy=[],[]
    for _ in range(M):
        fx=sfreqs(keys[ki],K,kappa) if multilevel else jax.random.uniform(keys[ki],(K,),minval=1.0,maxval=2.0*kappa)
        sx.append({"freqs":jnp.sort(fx),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        fy=sfreqs(keys[ki],K,kappa) if multilevel else jax.random.uniform(keys[ki],(K,),minval=1.0,maxval=2.0*kappa)
        sy.append({"freqs":jnp.sort(fy),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(M,))*0.1}
def sep_fwd(p,x,y):
    u=jnp.zeros_like(x)
    for n in range(M):
        wx=p["spatial_x"][n]["freqs"][None,:]*x; Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
        wy=p["spatial_y"][n]["freqs"][None,:]*y; Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
        u=u+p["mode_coeffs"][n]*Xn*Yn
    return u
def sep_grid(p,x1d,y1d):
    fx=jnp.stack([p["spatial_x"][n]["freqs"] for n in range(M)]); cx=jnp.stack([p["spatial_x"][n]["cos_c"] for n in range(M)]); sx=jnp.stack([p["spatial_x"][n]["sin_c"] for n in range(M)]); bx=jnp.stack([p["spatial_x"][n]["bias"] for n in range(M)])
    fy=jnp.stack([p["spatial_y"][n]["freqs"] for n in range(M)]); cy=jnp.stack([p["spatial_y"][n]["cos_c"] for n in range(M)]); sy=jnp.stack([p["spatial_y"][n]["sin_c"] for n in range(M)]); by=jnp.stack([p["spatial_y"][n]["bias"] for n in range(M)])
    phx=x1d[:,:,None]*fx[None]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
    phy=y1d[:,:,None]*fy[None]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
    return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None]*Xv,Yv)
def sep_residual_analytic(p,xc,yc,fg,kappa):
    fx=jnp.stack([jax.lax.stop_gradient(p["spatial_x"][n]["freqs"]) for n in range(M)]); cx=jnp.stack([p["spatial_x"][n]["cos_c"] for n in range(M)]); sx=jnp.stack([p["spatial_x"][n]["sin_c"] for n in range(M)]); bx=jnp.stack([p["spatial_x"][n]["bias"] for n in range(M)])
    fy=jnp.stack([jax.lax.stop_gradient(p["spatial_y"][n]["freqs"]) for n in range(M)]); cy=jnp.stack([p["spatial_y"][n]["cos_c"] for n in range(M)]); sy=jnp.stack([p["spatial_y"][n]["sin_c"] for n in range(M)]); by=jnp.stack([p["spatial_y"][n]["bias"] for n in range(M)])
    phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
    phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
    mc=p["mode_coeffs"]; cX=mc[None]*Xv
    u=jnp.einsum("nm,jm->nj",cX,Yv); uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
    return -(uxx+uyy)-kappa**2*u-fg

# ===== V1 coupled 2D Fourier (non-separable) =====
KC=290
def init_coupled(key,kappa):
    k1,k2,k3,k4=random.split(key,4)
    wx=sfreqs(k1,KC,kappa); wy=sfreqs(k2,KC,kappa)
    # random sign mix for directionality
    wy=wy*jnp.where(jax.random.uniform(k4,(KC,))>0.5,1.0,-1.0)
    return {"wx":wx,"wy":wy,"a":jax.random.normal(k3,(KC,))*0.05,"b":jax.random.normal(k1,(KC,))*0.05,"bias":jnp.zeros(1)}
def coupled_fwd(p,x,y):
    ph=x*jax.lax.stop_gradient(p["wx"])[None,:]+y*jax.lax.stop_gradient(p["wy"])[None,:]
    return jnp.sum(p["a"][None,:]*jnp.cos(ph)+p["b"][None,:]*jnp.sin(ph),axis=1,keepdims=True)+p["bias"]
def coupled_residual(p,x,y,kappa,f):
    wx=jax.lax.stop_gradient(p["wx"])[None,:]; wy=jax.lax.stop_gradient(p["wy"])[None,:]
    ph=x*wx+y*wy; cos=jnp.cos(ph); sin=jnp.sin(ph)
    u=jnp.sum(p["a"][None]*cos+p["b"][None]*sin,axis=1,keepdims=True)+p["bias"]
    lap=jnp.sum((-(wx**2)-(wy**2))*(p["a"][None]*cos+p["b"][None]*sin),axis=1,keepdims=True)
    return -lap-kappa**2*u-f

# ===== V2 separable MLP per axis =====
def init_mlp(key,kappa):
    keys=jax.random.split(key,M*2+1); ki=0; sx,sy=[],[]
    H=24
    def mlp(k):
        k1,k2,k3=random.split(k,3)
        return {"w1":random.normal(k1,(1,H))*0.5,"b1":jnp.zeros((H,)),
                "w2":random.normal(k2,(H,H))*(1/np.sqrt(H)),"b2":jnp.zeros((H,)),
                "w3":random.normal(k3,(H,1))*(1/np.sqrt(H)),"b3":jnp.zeros((1,))}
    for _ in range(M):
        sx.append(mlp(keys[ki])); ki+=1; sy.append(mlp(keys[ki])); ki+=1
    return {"sx":sx,"sy":sy,"mc":jax.random.normal(keys[ki],(M,))*0.1}
def mlp1(m,x):
    h=jnp.tanh(x@m["w1"]+m["b1"]); h=jnp.tanh(h@m["w2"]+m["b2"]); return h@m["w3"]+m["b3"]
def mlp_fwd(p,x,y):
    u=jnp.zeros_like(x)
    for n in range(M): u=u+p["mc"][n]*mlp1(p["sx"][n],x)*mlp1(p["sy"][n],y)
    return u

# ===== generic trainers =====
def train_grid(p,residual_fn,fwd_grid,data,tag):
    opt=optax.adam(LR); st=opt.init(p)
    xc,yc,fg=data["xc"],data["yc"],data["fg"]; xb,yb,ub=data["xb"],data["yb"],data["ub"]; kappa=data["kappa"]
    def loss(p):
        r=residual_fn(p,xc,yc,fg,kappa); bc=jnp.mean((sep_fwd(p,xb,yb)-ub)**2)
        return jnp.mean(r**2)+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    return _loop(p,step,fwd_grid,data,tag)

def _loop(p,step,fwd_grid,data,tag):
    opt=optax.adam(LR); st=opt.init(p)
    best=float("inf"); bp=p; hist=[]
    for _ in range(2): p,st,_=step(p,st)  # warmup / JIT compile
    t0=time.time()
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            up=np.array(fwd_grid(p,data["x1d"],data["y1d"])); e=l2(up,data["ue"]); hist.append(e)
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    tt=time.time()-t0
    up=np.array(fwd_grid(bp,data["x1d"],data["y1d"]))
    return {"tag":tag,"params":cp(p),"time_s":tt,"best_l2":best,"final_l2":hist[-1],"u_pred":up}

# specialized trainers (different fwd/residual signatures)
def run_V0(data,seed):
    p=init_sep(random.PRNGKey(seed),data["kappa"],True)
    return train_grid(p,sep_residual_analytic,sep_grid,data,"V0_Full")
def run_V3(data,seed):
    p=init_sep(random.PRNGKey(seed),data["kappa"],False)
    return train_grid(p,sep_residual_analytic,sep_grid,data,"V3_NoMultilevel")
def run_V4(data,seed):
    # separable + spectral + 3-level but PURE AD residual (pointwise)
    p=init_sep(random.PRNGKey(seed),data["kappa"],True); kappa=data["kappa"]
    xc=data["xc"].squeeze(); yc=data["yc"].squeeze()
    Xc,Yc=jnp.meshgrid(xc,yc,indexing="ij"); xf=Xc.reshape(-1); yf=Yc.reshape(-1)
    fg_flat=data["fg"].reshape(-1)
    def res_single(p,xs,ys,fs):
        uf=lambda a,b: sep_fwd(p,a[None,None],b[None,None]).squeeze()
        uxx=jax.grad(jax.grad(uf,0),0)(xs,ys); uyy=jax.grad(jax.grad(uf,1),1)(xs,ys)
        return -(uxx+uyy)-kappa**2*uf(xs,ys)-fs
    rb=vmap(res_single,in_axes=(None,0,0,0))
    opt=optax.adam(LR); st=opt.init(p)
    xb,yb,ub=data["xb"],data["yb"],data["ub"]
    def loss(p):
        r=rb(p,xf,yf,fg_flat); return jnp.mean(r**2)+jnp.mean((sep_fwd(p,xb,yb)-ub)**2)
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    return _loop(p,step,sep_grid,data,"V4_NoAnalytic")
def run_V1(data,seed):
    p=init_coupled(random.PRNGKey(seed),data["kappa"]); kappa=data["kappa"]
    xc=data["xc"].squeeze(); yc=data["yc"].squeeze(); Xc,Yc=jnp.meshgrid(xc,yc,indexing="ij")
    xf=Xc.reshape(-1,1); yf=Yc.reshape(-1,1); fg=data["fg"].reshape(-1,1)
    xb,yb,ub=data["xb"],data["yb"],data["ub"]
    opt=optax.adam(LR); st=opt.init(p)
    def loss(p):
        r=coupled_residual(p,xf,yf,kappa,fg); bc=jnp.mean((coupled_fwd(p,xb,yb)-ub)**2)
        return jnp.mean(r**2)+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    def fwd_grid(p,x1d,y1d):
        X,Y=jnp.meshgrid(x1d.squeeze(),y1d.squeeze(),indexing="ij")
        return coupled_fwd(p,X.reshape(-1,1),Y.reshape(-1,1)).reshape(X.shape)
    return _loop(p,step,fwd_grid,data,"V1_NoSeparation")
def run_V2(data,seed):
    p=init_mlp(random.PRNGKey(seed),data["kappa"]); kappa=data["kappa"]
    xc=data["xc"].squeeze(); yc=data["yc"].squeeze(); Xc,Yc=jnp.meshgrid(xc,yc,indexing="ij")
    xf=Xc.reshape(-1); yf=Yc.reshape(-1); fg=data["fg"].reshape(-1)
    xb,yb,ub=data["xb"],data["yb"],data["ub"]
    opt=optax.adam(LR); st=opt.init(p)
    def res_single(p,xs,ys,fs):
        uf=lambda a,b: mlp_fwd(p,a[None,None],b[None,None]).squeeze()
        uxx=jax.grad(jax.grad(uf,0),0)(xs,ys); uyy=jax.grad(jax.grad(uf,1),1)(xs,ys)
        return -(uxx+uyy)-kappa**2*uf(xs,ys)-fs
    rb=vmap(res_single,in_axes=(None,0,0,0))
    def loss(p):
        r=rb(p,xf,yf,fg); return jnp.mean(r**2)+jnp.mean((mlp_fwd(p,xb,yb)-ub)**2)
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    def fwd_grid(p,x1d,y1d):
        X,Y=jnp.meshgrid(x1d.squeeze(),y1d.squeeze(),indexing="ij")
        return mlp_fwd(p,X.reshape(-1,1),Y.reshape(-1,1)).reshape(X.shape)
    return _loop(p,step,fwd_grid,data,"V2_NoSpectral")

VARIANTS=[("V0_Full",run_V0),("V1_NoSeparation",run_V1),("V2_NoSpectral",run_V2),
          ("V3_NoMultilevel",run_V3),("V4_NoAnalytic",run_V4)]

def main():
    kappa=24*np.pi; out={}
    for kind in ["separable","nonseparable"]:
        print(f"\n######### problem={kind} #########",flush=True)
        out[kind]={}
        for vname,fn in VARIANTS:
            recs=[]
            for sd in SEEDS:
                data=make_problem(kind,kappa,sd)
                r=fn(data,sd); recs.append(r)
                print(f"  {kind} {vname} seed={sd}: best={r['best_l2']:.3e} t={r['time_s']:.1f}s params={r['params']}",flush=True)
            b=np.array([r["best_l2"] for r in recs]); t=np.array([r["time_s"] for r in recs])
            out[kind][vname]={"params":recs[0]["params"],"best_l2_mean":float(b.mean()),"best_l2_std":float(b.std()),
                              "best_l2_min":float(b.min()),"time_mean":float(t.mean())}
            np.savez(os.path.join(SAVE,f"{kind}_{vname}_pred.npz"),u_pred=recs[0]["u_pred"],ue=make_problem(kind,kappa,0)["ue"])
    with open(os.path.join(SAVE,"E3_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E3 to",SAVE)

if __name__=="__main__": main()
