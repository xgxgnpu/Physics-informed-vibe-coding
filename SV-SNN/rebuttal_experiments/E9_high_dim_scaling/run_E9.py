"""
E9 - 3D / higher-dimensional scalability
========================================
Addresses: R8.3 (no d>=3 experiment), R8.5 (how N scales with dimensionality & accuracy),
           R1.4 (3D), R10.3 (3D+).

Part A - TRUE 3D spatial benchmark: Poisson on [0,1]^3,
         u = sin(k x) sin(k y) sin(k z),  -Lap u = 3 k^2 u,  k=8*pi, u=0 on boundary.
         Compare accelerated SV-SNN (separable 3D) vs vanilla PINN vs FourierPINN.

Part B - DIMENSION SCALING: run the SV-SNN on d=1,2,3 (same per-axis frequency content)
         and record parameters / accuracy / training time to demonstrate O(d*K) parameter
         growth (vs the O(K^d) cost of dense grids/coupled bases).

3 seeds. Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad
import optax
from pyDOE import lhs

EPOCHS=6000; LR=1e-3; EVAL_EVERY=100; SEEDS=[0,1,2]
M=6; K=24; KAP=8*np.pi
NC3=40; NC2=80; NC1=400; NTEST3=48; NTEST2=160; NTEST1=2000
N_PDE_PINN=8000; N_BC=2048
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))
def sfreqs(key,wc):
    nl=K//4; ncc=K//2; nh=K-nl-ncc; _,k2,k3=jax.random.split(key,3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*float(wc)*0.3+wc),
                                     jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)]))

# ---------------- SV-SNN d-dimensional separable ----------------
def svsnn_init(key,d):
    keys=jax.random.split(key,M*d*3+1); ki=0; axes=[[] for _ in range(d)]
    for _ in range(M):
        for ax in range(d):
            axes[ax].append({"freqs":sfreqs(keys[ki],KAP),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,
                             "sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"axes":axes,"mode_coeffs":jax.random.normal(keys[ki],(M,))*0.1}

def axis_vals(sp_list,coord):  # coord (N,1) -> (N,M), and second deriv (N,M)
    f=jnp.stack([jax.lax.stop_gradient(s["freqs"]) for s in sp_list]); c=jnp.stack([s["cos_c"] for s in sp_list])
    s_=jnp.stack([s["sin_c"] for s in sp_list]); b=jnp.stack([s["bias"] for s in sp_list])
    ph=coord[:,:,None]*f[None]; trig=c[None]*jnp.cos(ph)+s_[None]*jnp.sin(ph)
    val=jnp.sum(trig,axis=2)+b[None,:,0]; dd=jnp.sum(-(f[None]**2)*trig,axis=2)
    return val,dd

def svsnn_u_grid(p,coords):
    d=len(p["axes"]); vals=[axis_vals(p["axes"][ax],coords[ax])[0] for ax in range(d)]
    mc=p["mode_coeffs"]
    if d==1: return jnp.einsum("im,m->i",vals[0],mc)
    if d==2: return jnp.einsum("im,jm,m->ij",vals[0],vals[1],mc)
    return jnp.einsum("im,jm,km,m->ijk",vals[0],vals[1],vals[2],mc)

def svsnn_u_points(p,pts):
    """u at scattered points pts (N,d) for boundary penalty."""
    d=len(p["axes"]); N=pts.shape[0]; acc=jnp.ones((N,p["mode_coeffs"].shape[0]))
    for ax in range(d):
        v,_=axis_vals(p["axes"][ax],pts[:,ax:ax+1]); acc=acc*v
    return acc@p["mode_coeffs"]

def boundary_points(d,seed,nb=256):
    np.random.seed(1000+seed); faces=[]
    for ax in range(d):
        for val in [0.0,1.0]:
            pts=np.random.rand(nb,d).astype(np.float32); pts[:,ax]=val; faces.append(pts)
    return jnp.asarray(np.vstack(faces),jnp.float32)

def svsnn_resid(p,coords,fg):
    d=len(p["axes"]); va=[axis_vals(p["axes"][ax],coords[ax]) for ax in range(d)]
    vals=[v[0] for v in va]; dds=[v[1] for v in va]; mc=p["mode_coeffs"]
    if d==1:
        u=jnp.einsum("im,m->i",vals[0],mc); lap=jnp.einsum("im,m->i",dds[0],mc)
    elif d==2:
        u=jnp.einsum("im,jm,m->ij",vals[0],vals[1],mc)
        lap=jnp.einsum("im,jm,m->ij",dds[0],vals[1],mc)+jnp.einsum("im,jm,m->ij",vals[0],dds[1],mc)
    else:
        u=jnp.einsum("im,jm,km,m->ijk",vals[0],vals[1],vals[2],mc)
        lap=(jnp.einsum("im,jm,km,m->ijk",dds[0],vals[1],vals[2],mc)
            +jnp.einsum("im,jm,km,m->ijk",vals[0],dds[1],vals[2],mc)
            +jnp.einsum("im,jm,km,m->ijk",vals[0],vals[1],dds[2],mc))
    return -lap-fg, u

def make_grid(d):
    if d==1: nc=NC1; nt=NTEST1
    elif d==2: nc=NC2; nt=NTEST2
    else: nc=NC3; nt=NTEST3
    c=np.linspace(0,1,nc)
    coords=[jnp.asarray(c.reshape(-1,1),jnp.float32) for _ in range(d)]
    tc=np.linspace(0,1,nt); tcoords=[jnp.asarray(tc.reshape(-1,1),jnp.float32) for _ in range(d)]
    grids=np.meshgrid(*([np.linspace(0,1,nc)]*d),indexing="ij")
    fg=3.0*KAP**2*np.prod([np.sin(KAP*g) for g in grids],axis=0) if d==3 else (
        (d*KAP**2)*np.prod([np.sin(KAP*g) for g in grids],axis=0))
    tgrids=np.meshgrid(*([tc]*d),indexing="ij")
    ue=np.prod([np.sin(KAP*g) for g in tgrids],axis=0)
    return coords,jnp.asarray(fg,jnp.float32),tcoords,ue

def svsnn_train(d,seed):
    p=svsnn_init(random.PRNGKey(seed),d); coords,fg,tcoords,ue=make_grid(d)
    bpts=boundary_points(d,seed)  # u=0 on all faces; pins constant/linear null directions
    opt=optax.adam(LR); st=opt.init(p)
    def loss(p):
        r,u=svsnn_resid(p,coords,fg); bc=jnp.mean(svsnn_u_points(p,bpts)**2)
        return jnp.mean(r**2)+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); bp=p; t0=time.time()
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            up=np.array(svsnn_u_grid(p,tcoords)); e=l2(up,ue)
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    return {"d":d,"best_l2":best,"params":cp(p),"time_s":time.time()-t0,
            "u_pred":np.array(svsnn_u_grid(bp,tcoords)),"ue":ue}

# ---------------- 3D baselines ----------------
def hvp(f,x,v):
    g=lambda z: jax.jvp(f,(z,),(v,))[1]; return jax.jvp(g,(x,),(v,))[1]
def baseline3d(which,seed):
    np.random.seed(seed)
    pde=lhs(3,samples=N_PDE_PINN).astype(np.float32)
    # boundary points on 6 faces
    nb=N_BC//6; rb=np.random.rand(nb,2).astype(np.float32); faces=[]
    for ax in range(3):
        for val in [0.0,1.0]:
            pts=np.zeros((nb,3),np.float32); idx=[i for i in range(3) if i!=ax]
            pts[:,idx[0]]=rb[:,0]; pts[:,idx[1]]=rb[:,1]; pts[:,ax]=val; faces.append(pts)
    xb=np.vstack(faces); 
    ub=(np.sin(KAP*xb[:,0])*np.sin(KAP*xb[:,1])*np.sin(KAP*xb[:,2])).reshape(-1,1)
    f_pde=(3*KAP**2*np.sin(KAP*pde[:,0])*np.sin(KAP*pde[:,1])*np.sin(KAP*pde[:,2])).reshape(-1,1)
    pde=jnp.asarray(pde); xb=jnp.asarray(xb); ub=jnp.asarray(ub); f_pde=jnp.asarray(f_pde)
    tc=np.linspace(0,1,NTEST3); TG=np.meshgrid(tc,tc,tc,indexing="ij"); ue=np.sin(KAP*TG[0])*np.sin(KAP*TG[1])*np.sin(KAP*TG[2])
    test_xy=jnp.asarray(np.stack([g.reshape(-1) for g in TG],axis=1),jnp.float32)
    if which=="FourierPINN":
        def init(key):
            ks=random.split(key,4); Ws=[sfreqs(ks[i],KAP).reshape(1,-1) for i in range(3)]
            dims=[6*K,128,128,128,1]; layers=[]; key=ks[3]
            for i in range(len(dims)-1):
                k,key=random.split(key); lim=jnp.sqrt(6.0/(dims[i]+dims[i+1]))
                layers.append({"w":random.uniform(k,(dims[i],dims[i+1]),minval=-lim,maxval=lim),"b":jnp.zeros((dims[i+1],))})
            return {"W":Ws,"mlp":layers}
        def fwd(p,xyz):
            feats=[]
            for ax in range(3):
                c=xyz[:,ax:ax+1]; W=jax.lax.stop_gradient(p["W"][ax]); feats+= [jnp.sin(c@W),jnp.cos(c@W)]
            h=jnp.concatenate(feats,axis=-1); n=len(p["mlp"])
            for i,L in enumerate(p["mlp"]):
                h=h@L["w"]+L["b"]
                if i<n-1: h=jnp.tanh(h)
            return h
    else:
        def init(key):
            dims=[3,128,128,128,128,1]; layers=[]; k=key
            for i in range(len(dims)-1):
                k,sub=random.split(k); lim=jnp.sqrt(6.0/(dims[i]+dims[i+1]))
                layers.append({"w":random.uniform(sub,(dims[i],dims[i+1]),minval=-lim,maxval=lim),"b":jnp.zeros((dims[i+1],))})
            return {"mlp":layers}
        def fwd(p,xyz):
            h=xyz; n=len(p["mlp"])
            for i,L in enumerate(p["mlp"]):
                h=h@L["w"]+L["b"]
                if i<n-1: h=jnp.tanh(h)
            return h
    tx=jnp.zeros_like(pde).at[:,0].set(1.0); ty=jnp.zeros_like(pde).at[:,1].set(1.0); tz=jnp.zeros_like(pde).at[:,2].set(1.0)
    def loss(p):
        uxx=hvp(lambda z:fwd(p,z),pde,tx); uyy=hvp(lambda z:fwd(p,z),pde,ty); uzz=hvp(lambda z:fwd(p,z),pde,tz)
        r=-(uxx+uyy+uzz)-f_pde; bc=jnp.mean((fwd(p,xb)-ub)**2); return jnp.mean(r**2)+bc
    p=init(random.PRNGKey(seed)); opt=optax.adam(LR); st=opt.init(p)
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); t0=time.time()
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            up=np.array(fwd(p,test_xy)).reshape(NTEST3,NTEST3,NTEST3); best=min(best,l2(up,ue))
    return {"method":which,"best_l2":best,"params":cp(p),"time_s":time.time()-t0}

def main():
    out={"partA_3d":{},"partB_scaling":{}}
    # Part A: 3D Poisson, SV-SNN vs baselines
    print("\n##### Part A: 3D Poisson #####",flush=True)
    for sd in SEEDS:
        r=svsnn_train(3,sd)
        out["partA_3d"].setdefault("SVSNN",[]).append({"best_l2":r["best_l2"],"params":r["params"],"time_s":r["time_s"]})
        print(f"  SVSNN 3D seed={sd}: best L2={r['best_l2']:.3e} params={r['params']} t={r['time_s']:.1f}s",flush=True)
        if sd==0: np.savez(os.path.join(SAVE,"svsnn_3d_pred.npz"),u_pred=r["u_pred"],ue=r["ue"])
    for which in ["FourierPINN","PINN"]:
        for sd in SEEDS:
            r=baseline3d(which,sd)
            out["partA_3d"].setdefault(which,[]).append({"best_l2":r["best_l2"],"params":r["params"],"time_s":r["time_s"]})
            print(f"  {which} 3D seed={sd}: best L2={r['best_l2']:.3e} params={r['params']} t={r['time_s']:.1f}s",flush=True)
    # Part B: dimension scaling SV-SNN d=1,2,3
    print("\n##### Part B: SV-SNN dimension scaling #####",flush=True)
    for d in [1,2,3]:
        recs=[svsnn_train(d,sd) for sd in SEEDS]
        b=np.array([r["best_l2"] for r in recs]); t=np.array([r["time_s"] for r in recs])
        out["partB_scaling"][str(d)]={"params":recs[0]["params"],"best_l2_mean":float(b.mean()),
                                      "best_l2_std":float(b.std()),"time_mean":float(t.mean())}
        print(f"  d={d}: params={recs[0]['params']} best L2={b.mean():.3e} t={t.mean():.1f}s",flush=True)
    with open(os.path.join(SAVE,"E9_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E9 to",SAVE)

if __name__=="__main__": main()
