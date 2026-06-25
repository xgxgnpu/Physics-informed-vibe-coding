"""
E5 - Characteristic frequency w_char: sensitivity & automatic estimation
========================================================================
Addresses: R1.5, R3.2, R4.3, R5.1, R9.3.

Sub-experiments (2D, separable SV-SNN unless noted):
  (a) w_char SCAN : true kappa=24pi; set SV-SNN w_char = ratio*kappa for
                    ratio in {0.5,0.75,0.9,1.0,1.1,1.25,1.5,2.0}. -> U-shaped sensitivity.
  (b) MULTI-MODAL : Poisson with u = sin(k1 x)sin(k1 y)+sin(k2 x)sin(k2 y), k1=12pi,k2=24pi.
                    Compare w_char in {k1, k2, mean, max} and a SPLIT allocation covering both.
  (c) SPATIAL CHIRP: Poisson with spatially varying frequency u=sin(a x^2)*sin(a y^2) -> single
                    w_char insufficient; report degradation + learnable-freq recovery.
  (d) FFT AUTO-ESTIMATE : estimate w_char from the source term via 2D FFT, compare to manual.
  (e) FROZEN vs LEARNABLE freqs (robustness when w_char is wrong, ratio=0.6).
  (f) NOISY prior : w_char = kappa*(1+eps), eps in {0,0.05,0.1,0.2,0.3}.

3 seeds where stochastic. Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax

EPOCHS=6000; LR=1e-3; N_BC=1024; N_TEST=200; EVAL_EVERY=100; NC=100; M=6; K=32; SEEDS=[0,1,2]
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))

def freqs3(key,wc,split=(0.25,0.5,0.25)):
    nl=int(split[0]*K); ncc=int(split[1]*K); nh=K-nl-ncc; _,k2,k3=jax.random.split(key,3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,max(nl,1)),
        jnp.abs(jax.random.normal(k2,(max(ncc,1),))*30.0+wc),
        jax.random.uniform(k3,(max(nh,1),),minval=wc*0.5,maxval=wc)]))[:K]

def init_sep(key,wc,freqs_list=None):
    keys=jax.random.split(key,M*6+1); ki=0; sx,sy=[],[]
    for n in range(M):
        fx=freqs_list[n] if freqs_list is not None else freqs3(keys[ki],wc); 
        sx.append({"freqs":fx,"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        fy=freqs_list[n] if freqs_list is not None else freqs3(keys[ki],wc)
        sy.append({"freqs":fy,"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(M,))*0.1}

def sep_fwd(p,x,y):
    u=jnp.zeros_like(x)
    for n in range(M):
        wx=p["spatial_x"][n]["freqs"][None,:]*x; Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
        wy=p["spatial_y"][n]["freqs"][None,:]*y; Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
        u=u+p["mode_coeffs"][n]*Xn*Yn
    return u
def stacks(p,axis,freeze):
    sg=jax.lax.stop_gradient if freeze else (lambda z:z)
    f=jnp.stack([sg(p[axis][n]["freqs"]) for n in range(M)])
    c=jnp.stack([p[axis][n]["cos_c"] for n in range(M)]); s=jnp.stack([p[axis][n]["sin_c"] for n in range(M)]); b=jnp.stack([p[axis][n]["bias"] for n in range(M)])
    return f,c,s,b
def sep_grid(p,x1d,y1d):
    fx,cx,sx,bx=stacks(p,"spatial_x",True); fy,cy,sy,by=stacks(p,"spatial_y",True)
    phx=x1d[:,:,None]*fx[None]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
    phy=y1d[:,:,None]*fy[None]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
    return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None]*Xv,Yv)
def residual(p,xc,yc,fg,helm_kappa,freeze):
    fx,cx,sx,bx=stacks(p,"spatial_x",freeze); fy,cy,sy,by=stacks(p,"spatial_y",freeze)
    phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
    phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
    mc=p["mode_coeffs"]; cX=mc[None]*Xv
    u=jnp.einsum("nm,jm->nj",cX,Yv); uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
    if helm_kappa is None: return -(uxx+uyy)-fg          # Poisson
    return -(uxx+uyy)-helm_kappa**2*u-fg                  # Helmholtz

def grid_pts():
    xc=np.linspace(0,1,NC); yc=np.linspace(0,1,NC); return xc,yc

def bc_data(seed):
    np.random.seed(seed); nps=N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    return jnp.asarray(xb,jnp.float32),jnp.asarray(yb,jnp.float32)

def test_grid():
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    return x1,y1,X,Y

def train(wc,problem,seed,freeze=True,freqs_list=None,helm_kappa=None,split=(0.25,0.5,0.25)):
    if freqs_list is None and split!=(0.25,0.5,0.25):
        key=random.PRNGKey(seed); keys=jax.random.split(key,2*M)
        freqs_list=[freqs3(keys[i],wc,split) for i in range(M)]
    p=init_sep(random.PRNGKey(seed),wc,freqs_list)
    xc,yc=grid_pts(); Xc,Yc=np.meshgrid(xc,yc,indexing="ij")
    fg=jnp.asarray(problem["f"](Xc,Yc),jnp.float32)
    xcj=jnp.asarray(xc.reshape(-1,1),jnp.float32); ycj=jnp.asarray(yc.reshape(-1,1),jnp.float32)
    xb,yb=bc_data(seed); ub=jnp.asarray(problem["u"](np.array(xb),np.array(yb)),jnp.float32)
    x1,y1,X,Y=test_grid(); ue=problem["u"](X,Y); x1d=jnp.asarray(x1.reshape(-1,1),jnp.float32); y1d=jnp.asarray(y1.reshape(-1,1),jnp.float32)
    opt=optax.adam(LR); st=opt.init(p)
    def loss(p):
        r=residual(p,xcj,ycj,fg,helm_kappa,freeze); bc=jnp.mean((sep_fwd(p,xb,yb)-ub)**2); return jnp.mean(r**2)+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); bp=p
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=l2(np.array(sep_grid(p,x1d,y1d)),ue); 
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    return best, np.array(sep_grid(bp,x1d,y1d)), ue

# ---- problem builders ----
def helm(kappa):
    return {"u":lambda x,y:np.sin(kappa*x)*np.sin(kappa*y),
            "f":lambda x,y:kappa**2*np.sin(kappa*x)*np.sin(kappa*y),"kappa":kappa}
def poisson_modes(k1,k2):
    u=lambda x,y:np.sin(k1*x)*np.sin(k1*y)+np.sin(k2*x)*np.sin(k2*y)
    f=lambda x,y:2*k1**2*np.sin(k1*x)*np.sin(k1*y)+2*k2**2*np.sin(k2*x)*np.sin(k2*y)
    return {"u":u,"f":f}
def poisson_chirp(a):
    # u=sin(a x^2) sin(a y^2); u_xx = 2a cos(a x^2) - (2a x)^2 sin(a x^2)
    def u(x,y): return np.sin(a*x**2)*np.sin(a*y**2)
    def f(x,y):
        uxx=(2*a*np.cos(a*x**2)-(2*a*x)**2*np.sin(a*x**2))*np.sin(a*y**2)
        uyy=np.sin(a*x**2)*(2*a*np.cos(a*y**2)-(2*a*y)**2*np.sin(a*y**2))
        return -(uxx+uyy)
    return {"u":u,"f":f}

def main():
    kappa=24*np.pi; out={}

    # (a) w_char scan (Helmholtz)
    print("\n## (a) w_char scan ##",flush=True); out["a_scan"]={}
    for ratio in [0.5,0.75,0.9,1.0,1.1,1.25,1.5,2.0]:
        wc=ratio*kappa; bs=[]
        for sd in SEEDS:
            b,_,_=train(wc,helm(kappa),sd,freeze=True,helm_kappa=kappa); bs.append(b)
        bs=np.array(bs); out["a_scan"][f"{ratio:.2f}"]={"mean":float(bs.mean()),"std":float(bs.std()),"min":float(bs.min())}
        print(f"   ratio={ratio:.2f} wc/kappa best L2 mean={bs.mean():.3e}",flush=True)

    # (b) multi-modal (Poisson)
    print("\n## (b) multi-modal ##",flush=True); out["b_multimodal"]={}
    k1,k2=12*np.pi,24*np.pi; prob=poisson_modes(k1,k2)
    for label,wc in [("k1",k1),("k2",k2),("mean",(k1+k2)/2),("max",k2)]:
        bs=[train(wc,prob,sd,freeze=True,helm_kappa=None)[0] for sd in SEEDS]; bs=np.array(bs)
        out["b_multimodal"][label]={"wc":float(wc),"mean":float(bs.mean()),"std":float(bs.std())}
        print(f"   w_char={label}: best L2 mean={bs.mean():.3e}",flush=True)
    # split allocation covering both: half modes at k1, half at k2
    bs_split=[]
    for sd in SEEDS:
        kk=jax.random.split(random.PRNGKey(sd),2*M)
        fl_sd=[freqs3(kk[i],k1) if i<M//2 else freqs3(kk[i],k2) for i in range(M)]
        bs_split.append(train(k2,prob,sd,freeze=True,freqs_list=fl_sd,helm_kappa=None)[0])
    bs_split=np.array(bs_split)
    out["b_multimodal"]["split_k1k2"]={"mean":float(bs_split.mean()),"std":float(bs_split.std())}
    print(f"   split(k1&k2): best L2 mean={bs_split.mean():.3e}",flush=True)

    # (c) spatial chirp
    print("\n## (c) spatial chirp ##",flush=True)
    a=30.0; prob=poisson_chirp(a)
    wc_guess=np.sqrt(2*a*1.0)  # rough max local freq ~ 2 a x at x=1 -> 2a; use 2a
    b_frozen=np.array([train(2*a,prob,sd,freeze=True,helm_kappa=None)[0] for sd in SEEDS])
    b_learn=np.array([train(2*a,prob,sd,freeze=False,helm_kappa=None)[0] for sd in SEEDS])
    out["c_chirp"]={"frozen_mean":float(b_frozen.mean()),"learnable_mean":float(b_learn.mean()),
                    "frozen_std":float(b_frozen.std()),"learnable_std":float(b_learn.std())}
    print(f"   chirp frozen={b_frozen.mean():.3e} learnable={b_learn.mean():.3e}",flush=True)

    # (d) FFT auto-estimate
    print("\n## (d) FFT auto-estimate ##",flush=True)
    Ng=256; xx=np.linspace(0,1,Ng,endpoint=False); X,Y=np.meshgrid(xx,xx,indexing="ij")
    src=kappa**2*np.sin(kappa*X)*np.sin(kappa*Y)
    F=np.fft.fft2(src); mag=np.abs(F)
    freqs=np.fft.fftfreq(Ng,d=1.0/Ng)  # cycles over domain
    idx=np.unravel_index(np.argmax(mag[:Ng//2,:Ng//2]),(Ng//2,Ng//2))
    fx_cyc=abs(freqs[idx[0]]); est_wchar=2*np.pi*fx_cyc  # angular
    out["d_fft"]={"true_kappa":float(kappa),"est_wchar":float(est_wchar),"rel_err":float(abs(est_wchar-kappa)/kappa)}
    print(f"   true kappa={kappa:.3f}, FFT-estimated w_char={est_wchar:.3f}, rel err={abs(est_wchar-kappa)/kappa:.2%}",flush=True)
    # train with FFT-estimated vs manual
    b_auto=np.array([train(est_wchar,helm(kappa),sd,freeze=True,helm_kappa=kappa)[0] for sd in SEEDS])
    b_manual=np.array([train(kappa,helm(kappa),sd,freeze=True,helm_kappa=kappa)[0] for sd in SEEDS])
    out["d_fft"]["best_l2_auto"]=float(b_auto.mean()); out["d_fft"]["best_l2_manual"]=float(b_manual.mean())
    print(f"   auto best L2={b_auto.mean():.3e}, manual best L2={b_manual.mean():.3e}",flush=True)

    # (e) frozen vs learnable when w_char wrong (ratio 0.6)
    print("\n## (e) frozen vs learnable (wrong w_char=0.6 kappa) ##",flush=True)
    wc=0.6*kappa
    b_fro=np.array([train(wc,helm(kappa),sd,freeze=True,helm_kappa=kappa)[0] for sd in SEEDS])
    b_lea=np.array([train(wc,helm(kappa),sd,freeze=False,helm_kappa=kappa)[0] for sd in SEEDS])
    out["e_freeze"]={"frozen_mean":float(b_fro.mean()),"learnable_mean":float(b_lea.mean()),
                     "frozen_std":float(b_fro.std()),"learnable_std":float(b_lea.std())}
    print(f"   frozen={b_fro.mean():.3e} learnable={b_lea.mean():.3e}",flush=True)

    # (f) noisy prior
    print("\n## (f) noisy prior ##",flush=True); out["f_noise"]={}
    for eps in [0.0,0.05,0.1,0.2,0.3]:
        bs=[]
        for sd in SEEDS:
            rng=np.random.RandomState(sd); wc=kappa*(1+eps*rng.randn())
            bs.append(train(wc,helm(kappa),sd,freeze=True,helm_kappa=kappa)[0])
        bs=np.array(bs); out["f_noise"][f"{eps:.2f}"]={"mean":float(bs.mean()),"std":float(bs.std())}
        print(f"   eps={eps:.2f}: best L2 mean={bs.mean():.3e}",flush=True)

    with open(os.path.join(SAVE,"E5_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E5 to",SAVE)

if __name__=="__main__": main()
