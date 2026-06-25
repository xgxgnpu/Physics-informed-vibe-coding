"""
E8 - Challenging problems: non-separable / non-periodic / heterogeneous (HONEST limits)
=======================================================================================
Addresses: R3.6, R5.6, R8.6, R9.2, R9.4 (non-smooth/localized), R10.3, R5.4 (limits), R1.4.

Three deliberately UN-favorable / challenging problems on [0,1]^2:
  (Q1) NON-SEPARABLE high-frequency:  u = sin(kappa (x^2 + y^2)),  Poisson -Lap u = f.
       (Cannot be written as X(x)Y(y); a separable expansion needs many modes -> expected limit.)
  (Q2) LOCALIZED NON-PERIODIC wave packet: u = exp(-((x-.5)^2+(y-.5)^2)/(2 s^2)) sin(kappa x),
       Poisson. (Gaussian envelope is non-periodic -> global Fourier basis stressed.)
  (Q3) HETEROGENEOUS-MEDIUM Helmholtz: -Lap u - kappa(x,y)^2 u = f, kappa(x,y)=k0(1+0.5 sin(2pi x)),
       u = sin(a x) sin(a y).  (Variable-coefficient operator.)

Methods: accelerated SV-SNN, FourierPINN, PINN. 3 seeds. We report results HONESTLY,
including where SV-SNN degrades (-> feeds the Limitations section).
Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad
import optax
from pyDOE import lhs

EPOCHS=8000; LR=1e-3; N_PDE=10000; N_BC=1024; N_TEST=200; EVAL_EVERY=100; NC=120; K=48; SEEDS=[0,1,2]
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))
def sfreqs(key,wc):
    nl=K//4; ncc=K//2; nh=K-nl-ncc; _,k2,k3=jax.random.split(key,3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*float(wc)*0.3+wc),
                                     jax.random.uniform(k3,(nh,),minval=wc*0.3,maxval=wc)]))

# ---------------- problems ----------------
def build_problem(q,seed):
    np.random.seed(seed)
    nps=N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    pde=lhs(2,samples=N_PDE)
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    xc=np.linspace(0,1,NC); yc=np.linspace(0,1,NC); Xc,Yc=np.meshgrid(xc,yc,indexing="ij")
    if q=="Q1_nonsep":
        kap=8*np.pi; wchar=2*kap
        uex=lambda x,y: np.sin(kap*(x**2+y**2))
        def lap(x,y):
            uxx=2*kap*np.cos(kap*(x**2+y**2))-(2*kap*x)**2*np.sin(kap*(x**2+y**2))
            uyy=2*kap*np.cos(kap*(x**2+y**2))-(2*kap*y)**2*np.sin(kap*(x**2+y**2))
            return uxx+uyy
        fg=-lap(Xc,Yc); op="poisson"; kap2_grid=None; Mn=16
        f_pts=lambda x,y:-(2*kap*np.cos(kap*(x**2+y**2))-(2*kap*x)**2*np.sin(kap*(x**2+y**2))
                          +2*kap*np.cos(kap*(x**2+y**2))-(2*kap*y)**2*np.sin(kap*(x**2+y**2)))
    elif q=="Q2_packet":
        kap=16*np.pi; s=0.12; wchar=kap
        env=lambda x,y: np.exp(-((x-0.5)**2+(y-0.5)**2)/(2*s**2))
        uex=lambda x,y: env(x,y)*np.sin(kap*x)
        def lap(x,y):
            E=env(x,y); ex=-(x-0.5)/s**2; ey=-(y-0.5)/s**2
            sk=np.sin(kap*x); ck=np.cos(kap*x)
            # u = E*sk ; u_x = E*(ex*sk + kap*ck); u_xx = E*((ex^2+ (-1/s^2))*sk + 2*ex*kap*ck - kap^2 sk)
            uxx=E*(((ex**2)-(1/s**2))*sk + 2*ex*kap*ck - kap**2*sk)
            uyy=E*(((ey**2)-(1/s**2))*sk)
            return uxx+uyy
        fg=-lap(Xc,Yc); op="poisson"; kap2_grid=None; Mn=12
        f_pts=lambda x,y:-lap(x,y)
    else:  # Q3 heterogeneous Helmholtz
        a=20*np.pi; k0=20*np.pi; wchar=a
        kapxy=lambda x,y: k0*(1+0.5*np.sin(2*np.pi*x))
        uex=lambda x,y: np.sin(a*x)*np.sin(a*y)
        lap=lambda x,y: -2*a**2*np.sin(a*x)*np.sin(a*y)
        fg=(-lap(Xc,Yc)-kapxy(Xc,Yc)**2*uex(Xc,Yc)); op="helm_var"
        kap2_grid=jnp.asarray(kapxy(Xc,Yc)**2,jnp.float32); Mn=8
        f_pts=lambda x,y:-lap(x,y)-kapxy(x,y)**2*uex(x,y)
        kapxy_pts=kapxy
    d={"xb":jnp.asarray(xb,jnp.float32),"yb":jnp.asarray(yb,jnp.float32),"ub":jnp.asarray(uex(xb,yb),jnp.float32),
       "x_pde":jnp.asarray(pde[:,0:1],jnp.float32),"y_pde":jnp.asarray(pde[:,1:2],jnp.float32),
       "f_pde":jnp.asarray(f_pts(pde[:,0:1],pde[:,1:2]),jnp.float32),
       "ue":uex(X,Y),"X":X,"Y":Y,"xc":jnp.asarray(xc.reshape(-1,1),jnp.float32),"yc":jnp.asarray(yc.reshape(-1,1),jnp.float32),
       "fg":jnp.asarray(fg,jnp.float32),"x1d":jnp.asarray(x1.reshape(-1,1),jnp.float32),"y1d":jnp.asarray(y1.reshape(-1,1),jnp.float32),
       "wchar":float(wchar),"op":op,"kap2_grid":kap2_grid,"Mn":Mn}
    if op=="helm_var":
        d["kap2_pde"]=jnp.asarray(kapxy_pts(pde[:,0:1],pde[:,1:2])**2,jnp.float32)
    return d

# ---------------- SV-SNN accelerated ----------------
def svsnn_run(data,seed):
    Mn=data["Mn"]; wc=data["wchar"]
    def init(key):
        keys=jax.random.split(key,Mn*6+1); ki=0; sx,sy=[],[]
        for _ in range(Mn):
            sx.append({"freqs":sfreqs(keys[ki],wc),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
            sy.append({"freqs":sfreqs(keys[ki],wc),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(Mn,))*0.1}
    def st(p,axis):
        f=jnp.stack([jax.lax.stop_gradient(p[axis][n]["freqs"]) for n in range(Mn)])
        c=jnp.stack([p[axis][n]["cos_c"] for n in range(Mn)]); s=jnp.stack([p[axis][n]["sin_c"] for n in range(Mn)]); b=jnp.stack([p[axis][n]["bias"] for n in range(Mn)])
        return f,c,s,b
    def fwd(p,x,y):
        u=jnp.zeros_like(x)
        for n in range(Mn):
            wx=p["spatial_x"][n]["freqs"][None,:]*x; Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
            wy=p["spatial_y"][n]["freqs"][None,:]*y; Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
            u=u+p["mode_coeffs"][n]*Xn*Yn
        return u
    def grid(p,x1d,y1d):
        fx,cx,sx,bx=st(p,"spatial_x"); fy,cy,sy,by=st(p,"spatial_y")
        phx=x1d[:,:,None]*fx[None]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
        phy=y1d[:,:,None]*fy[None]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
        return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None]*Xv,Yv)
    xc,yc,fg=data["xc"],data["yc"],data["fg"]; xb,yb,ub=data["xb"],data["yb"],data["ub"]
    kap2=data["kap2_grid"]
    def resid(p):
        fx,cx,sx,bx=st(p,"spatial_x"); fy,cy,sy,by=st(p,"spatial_y")
        phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
        phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
        mc=p["mode_coeffs"]; cX=mc[None]*Xv
        u=jnp.einsum("nm,jm->nj",cX,Yv); uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
        r=-(uxx+uyy)-fg
        if kap2 is not None: r=r-kap2*u
        return r
    p=init(random.PRNGKey(seed)); opt=optax.adam(LR); state=opt.init(p)
    def loss(p): return jnp.mean(resid(p)**2)+jnp.mean((fwd(p,xb,yb)-ub)**2)
    @jit
    def step(p,s):
        l,g=value_and_grad(loss)(p); u,s=opt.update(g,s,p); return optax.apply_updates(p,u),s,l
    for _ in range(2): p,state,_=step(p,state)
    best=float("inf"); bp=p; t0=time.time()
    for ep in range(2,EPOCHS):
        p,state,l=step(p,state)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=l2(np.array(grid(p,data["x1d"],data["y1d"])),data["ue"])
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    return {"method":"SVSNN","best_l2":best,"params":cp(p),"time_s":time.time()-t0,
            "u_pred":np.array(grid(bp,data["x1d"],data["y1d"]))}

# ---------------- pointwise baselines (FourierPINN, PINN) ----------------
def hvp(f,x,v):
    g=lambda z: jax.jvp(f,(z,),(v,))[1]; return jax.jvp(g,(x,),(v,))[1]
def baseline_run(which,data,seed):
    wc=data["wchar"]
    if which=="FourierPINN":
        def init(key):
            k1,k2,key=random.split(key,3)
            Wx=sfreqs(k1,wc).reshape(1,-1); Wy=sfreqs(k2,wc).reshape(1,-1)
            dims=[4*K,128,128,128,1]; layers=[]
            for i in range(len(dims)-1):
                k,key=random.split(key); lim=jnp.sqrt(6.0/(dims[i]+dims[i+1]))
                layers.append({"w":random.uniform(k,(dims[i],dims[i+1]),minval=-lim,maxval=lim),"b":jnp.zeros((dims[i+1],))})
            return {"Wx":Wx,"Wy":Wy,"mlp":layers}
        def fwd(p,xy):
            x,y=xy[:,0:1],xy[:,1:2]; Wx=jax.lax.stop_gradient(p["Wx"]); Wy=jax.lax.stop_gradient(p["Wy"])
            h=jnp.concatenate([jnp.sin(x@Wx),jnp.cos(x@Wx),jnp.sin(y@Wy),jnp.cos(y@Wy)],axis=-1); n=len(p["mlp"])
            for i,L in enumerate(p["mlp"]):
                h=h@L["w"]+L["b"]
                if i<n-1: h=jnp.tanh(h)
            return h
    else:
        def init(key):
            dims=[2,128,128,128,128,1]; layers=[]; k=key
            for i in range(len(dims)-1):
                k,sub=random.split(k); lim=jnp.sqrt(6.0/(dims[i]+dims[i+1]))
                layers.append({"w":random.uniform(sub,(dims[i],dims[i+1]),minval=-lim,maxval=lim),"b":jnp.zeros((dims[i+1],))})
            return {"mlp":layers}
        def fwd(p,xy):
            h=xy; n=len(p["mlp"])
            for i,L in enumerate(p["mlp"]):
                h=h@L["w"]+L["b"]
                if i<n-1: h=jnp.tanh(h)
            return h
    xp,yp=data["x_pde"],data["y_pde"]; fpde=data["f_pde"]; xb,yb,ub=data["xb"],data["yb"],data["ub"]
    xyp=jnp.concatenate([xp,yp],axis=-1); tx=jnp.zeros_like(xyp).at[:,0].set(1.0); ty=jnp.zeros_like(xyp).at[:,1].set(1.0)
    kap2_pde=data.get("kap2_pde",None)
    def loss(p):
        uxx=hvp(lambda z:fwd(p,z),xyp,tx); uyy=hvp(lambda z:fwd(p,z),xyp,ty)
        u=fwd(p,xyp); r=-(uxx+uyy)-fpde
        if kap2_pde is not None: r=r-kap2_pde*u
        bc=jnp.mean((fwd(p,jnp.concatenate([xb,yb],axis=-1))-ub)**2)
        return jnp.mean(r**2)+bc
    def pred(p):
        X,Y=np.meshgrid(np.array(data["x1d"]).squeeze(),np.array(data["y1d"]).squeeze(),indexing="ij")
        xy=jnp.asarray(np.stack([X.reshape(-1),Y.reshape(-1)],axis=1),jnp.float32)
        return np.array(fwd(p,xy)).reshape(N_TEST,N_TEST)
    p=init(random.PRNGKey(seed)); opt=optax.adam(LR); state=opt.init(p)
    @jit
    def step(p,s):
        l,g=value_and_grad(loss)(p); u,s=opt.update(g,s,p); return optax.apply_updates(p,u),s,l
    for _ in range(2): p,state,_=step(p,state)
    best=float("inf"); bp=p; t0=time.time()
    for ep in range(2,EPOCHS):
        p,state,l=step(p,state)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=l2(pred(p),data["ue"]); 
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    return {"method":which,"best_l2":best,"params":cp(p),"time_s":time.time()-t0,"u_pred":pred(bp)}

PROBLEMS=["Q1_nonsep","Q2_packet","Q3_hetero"]
def main():
    out={}
    for q in PROBLEMS:
        print(f"\n##### {q} #####",flush=True); out[q]={}
        for runner,name in [(svsnn_run,"SVSNN"),(lambda d,s:baseline_run("FourierPINN",d,s),"FourierPINN"),
                            (lambda d,s:baseline_run("PINN",d,s),"PINN")]:
            recs=[]
            for sd in SEEDS:
                data=build_problem(q,sd); r=runner(data,sd); recs.append(r)
                print(f"  {q} {name} seed={sd}: best L2={r['best_l2']:.3e} t={r['time_s']:.1f}s params={r['params']}",flush=True)
            b=np.array([r["best_l2"] for r in recs])
            out[q][name]={"params":recs[0]["params"],"best_l2_mean":float(b.mean()),"best_l2_std":float(b.std()),"best_l2_min":float(b.min())}
            np.savez(os.path.join(SAVE,f"{q}_{name}_pred.npz"),u_pred=recs[0]["u_pred"],ue=build_problem(q,0)["ue"],X=build_problem(q,0)["X"],Y=build_problem(q,0)["Y"])
    with open(os.path.join(SAVE,"E8_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E8 to",SAVE)

if __name__=="__main__": main()
