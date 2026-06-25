"""
E4b - Multi-scale frequency COVERAGE test (fair, residual-normalized)
=====================================================================
Motivation: the plain multi-scale Poisson in run_E4.py shows that ALL sampling
strategies plateau at L2~0.707, because the PINN MSE residual is dominated by the
high-wavenumber source term (|f_hi| ~ 36x |f_lo| for k_hi/k_lo=6 in 2D Poisson),
which STARVES the low-frequency solution component. This is a residual-magnitude
imbalance, NOT a frequency-coverage failure (adding more low-frequency basis in
S5_three40 does not help).

To FAIRLY isolate the value of multi-level frequency *coverage*, we normalize the
PDE residual by the local source magnitude (relative residual), removing the
amplitude imbalance, and compare three frequency priors on the SAME multi-scale
solution u = sin(k_lo x)sin(k_lo y) + sin(k_hi x)sin(k_hi y):
   C1 low_only  : all freqs ~ k_lo            (misses high band)
   C2 high_only : all freqs ~ k_hi            (misses low band)
   C3 multilevel: three-level covering [k_lo, k_hi]   (covers both)
Expectation: single-band priors miss one scale (high L2); multilevel captures both.

Reuses run_E4.py machinery. 3 seeds, mean+-std.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
import run_E4 as m   # reuse init_sep, sep_grid, stacks, residual, etc.

EPOCHS=8000; LR=1e-3; EVAL_EVERY=100; SEEDS=[0,1,2]
KLO=4*np.pi; KHI=24*np.pi
SAVE=m.SAVE

def freqs_band(key,center,K):
    # frequencies tightly around a single characteristic band
    _,k2,_=jax.random.split(key,3)
    return jnp.sort(jnp.abs(jax.random.normal(k2,(K,))*0.05*center+center))

def freqs_multi(key,klo,khi,K):
    # explicit coverage of both bands: 40% near low, 40% near high, 20% spread between
    nl=int(0.4*K); nh=int(0.4*K); nm=K-nl-nh
    k1,k2,k3=jax.random.split(key,3)
    lo=jnp.abs(jax.random.normal(k1,(nl,))*0.05*klo+klo)
    hi=jnp.abs(jax.random.normal(k2,(nh,))*0.05*khi+khi)
    mid=jax.random.uniform(k3,(nm,),minval=klo,maxval=khi)
    return jnp.sort(jnp.concatenate([lo,mid,hi]))

def make_freqs(strat,seed):
    key=random.PRNGKey(seed); keys=jax.random.split(key,2*m.M)
    if strat=="C1_low_only":
        return [freqs_band(keys[i],KLO,m.K) for i in range(m.M)]
    if strat=="C2_high_only":
        return [freqs_band(keys[i],KHI,m.K) for i in range(m.M)]
    if strat=="C3_multilevel":
        return [freqs_multi(keys[i],KLO,KHI,m.K) for i in range(m.M)]
    raise ValueError(strat)

def init_with_freqs(key,fl):
    keys=jax.random.split(key,m.M*6+1); ki=0; sx,sy=[],[]
    for n in range(m.M):
        sx.append({"freqs":fl[n],"cos_c":jax.random.normal(keys[ki+1],(m.K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(m.K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        sy.append({"freqs":fl[n],"cos_c":jax.random.normal(keys[ki+1],(m.K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(m.K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(m.M,))*0.1}

def _bc_data(seed):
    np.random.seed(seed); nps=m.N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    return jnp.asarray(xb,jnp.float32),jnp.asarray(yb,jnp.float32)

def train_norm(strat,seed):
    fl=make_freqs(strat,seed)
    p=init_with_freqs(random.PRNGKey(seed),fl)
    xc=np.linspace(0,1,m.NC); yc=np.linspace(0,1,m.NC); Xc,Yc=np.meshgrid(xc,yc,indexing="ij")
    u=lambda x,y:np.sin(KLO*x)*np.sin(KLO*y)+np.sin(KHI*x)*np.sin(KHI*y)
    fgn=2*KLO**2*np.sin(KLO*Xc)*np.sin(KLO*Yc)+2*KHI**2*np.sin(KHI*Xc)*np.sin(KHI*Yc)
    fg=jnp.asarray(fgn,jnp.float32)
    # relative-residual normalizer (per-point source magnitude, floored)
    wnorm=jnp.asarray(np.maximum(np.abs(fgn),0.05*np.abs(fgn).max()),jnp.float32)
    xcj=jnp.asarray(xc.reshape(-1,1),jnp.float32); ycj=jnp.asarray(yc.reshape(-1,1),jnp.float32)
    xb,yb=_bc_data(seed)
    x1=np.linspace(0,1,m.N_TEST); y1=np.linspace(0,1,m.N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    ue=u(X,Y); x1d=jnp.asarray(x1.reshape(-1,1),jnp.float32); y1d=jnp.asarray(y1.reshape(-1,1),jnp.float32)
    opt=optax.adam(LR); st=opt.init(p)
    def loss(p):
        r=m.residual(p,xcj,ycj,fg)/wnorm   # relative residual (Poisson)
        bc=jnp.mean(m.sep_fwd(p,xb,yb)**2)
        return jnp.mean(r**2)+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); upd,st=opt.update(g,st,p); return optax.apply_updates(p,upd),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); bp=p; t0=time.time()
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=m.l2(np.array(m.sep_grid(p,x1d,y1d)),ue)
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    return best,float(time.time()-t0)

def main():
    out={}
    print("\n######### E4b multi-scale coverage (normalized residual) #########",flush=True)
    for strat in ["C1_low_only","C2_high_only","C3_multilevel"]:
        bs=[]
        for sd in SEEDS:
            b,t=train_norm(strat,sd); bs.append(b)
            print(f"  {strat} seed={sd}: best={b:.3e} t={t:.1f}s",flush=True)
        bs=np.array(bs)
        out[strat]={"best_l2_mean":float(bs.mean()),"best_l2_std":float(bs.std()),"best_l2_min":float(bs.min())}
    out["_meta"]={"klo":float(KLO),"khi":float(KHI),"note":"relative residual normalized by source magnitude"}
    with open(os.path.join(SAVE,"E4b_coverage.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E4b to",SAVE)

if __name__=="__main__": main()
