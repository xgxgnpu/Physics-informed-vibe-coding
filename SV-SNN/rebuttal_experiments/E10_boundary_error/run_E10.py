"""
E10 - Near-boundary error on complex geometry
=============================================
Addresses: R5.5 (how errors behave near holes/corners/multiply-connected boundaries;
           boundary-near error plots; comparison vs a baseline).

Domain: [0,1]^2 minus a circular hole (center (0.5,0.5), radius 0.2)  -> multiply connected.
PDE: Poisson -Lap u = f, u = sin(mu x) sin(mu y), mu = 6 pi, f = 2 mu^2 u.
BC enforced on outer boundary AND hole boundary (collocation + boundary loss).

Methods: accelerated SV-SNN (separable grid + interior mask) and FourierPINN.
Outputs: error field, and error vs distance-to-nearest-boundary curve.
3 seeds. Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from pyDOE import lhs

EPOCHS=8000; LR=1e-3; EVAL_EVERY=100; SEEDS=[0,1,2]
M=8; K=48; MU=6*np.pi; NC=140; N_TEST=240; N_HOLE_BC=400; N_OUT_BC=400
CX,CY,RAD=0.5,0.5,0.2
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))
def sfreqs(key,wc):
    nl=K//4; ncc=K//2; nh=K-nl-ncc; _,k2,k3=jax.random.split(key,3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*float(wc)*0.3+wc),
                                     jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)]))
def uex(x,y): return np.sin(MU*x)*np.sin(MU*y)

def dist_to_boundary(X,Y):
    d_out=np.minimum.reduce([X,1-X,Y,1-Y])
    d_hole=np.sqrt((X-CX)**2+(Y-CY)**2)-RAD
    return np.minimum(d_out,d_hole)

def build(seed):
    np.random.seed(seed)
    # test grid
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    inside_hole=((X-CX)**2+(Y-CY)**2)<RAD**2
    ue=uex(X,Y); ue_masked=np.where(inside_hole,np.nan,ue)
    # training tensor grid + interior mask (outside hole)
    xc=np.linspace(0,1,NC); yc=np.linspace(0,1,NC); Xc,Yc=np.meshgrid(xc,yc,indexing="ij")
    mask=((Xc-CX)**2+(Yc-CY)**2>=RAD**2).astype(np.float32)
    fg=2*MU**2*np.sin(MU*Xc)*np.sin(MU*Yc)
    # hole boundary points
    th=np.linspace(0,2*np.pi,N_HOLE_BC,endpoint=False)
    xh=CX+RAD*np.cos(th); yh=CY+RAD*np.sin(th)
    # outer boundary points
    t=np.linspace(0,1,N_OUT_BC//4)
    xob=np.concatenate([np.zeros_like(t),np.ones_like(t),t,t]); yob=np.concatenate([t,t,np.zeros_like(t),np.ones_like(t)])
    xb=np.concatenate([xh,xob]).reshape(-1,1); yb=np.concatenate([yh,yob]).reshape(-1,1); ub=uex(xb,yb)
    # LHS interior points (outside hole) for pointwise baseline
    pts=lhs(2,samples=15000); keep=((pts[:,0]-CX)**2+(pts[:,1]-CY)**2)>=RAD**2; pts=pts[keep][:10000]
    return {"X":X,"Y":Y,"ue":ue,"ue_masked":ue_masked,"inside_hole":inside_hole,
            "xc":jnp.asarray(xc.reshape(-1,1),jnp.float32),"yc":jnp.asarray(yc.reshape(-1,1),jnp.float32),
            "mask":jnp.asarray(mask,jnp.float32),"fg":jnp.asarray(fg,jnp.float32),
            "xb":jnp.asarray(xb,jnp.float32),"yb":jnp.asarray(yb,jnp.float32),"ub":jnp.asarray(ub,jnp.float32),
            "x1d":jnp.asarray(x1.reshape(-1,1),jnp.float32),"y1d":jnp.asarray(y1.reshape(-1,1),jnp.float32),
            "pde":jnp.asarray(pts,jnp.float32),"f_pde":jnp.asarray(2*MU**2*np.sin(MU*pts[:,0:1])*np.sin(MU*pts[:,1:2]),jnp.float32)}

# ---- SV-SNN accelerated ----
def svsnn_run(data,seed):
    def init(key):
        keys=jax.random.split(key,M*6+1); ki=0; sx,sy=[],[]
        for _ in range(M):
            sx.append({"freqs":sfreqs(keys[ki],MU),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
            sy.append({"freqs":sfreqs(keys[ki],MU),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(M,))*0.1}
    def stx(p,axis):
        f=jnp.stack([jax.lax.stop_gradient(p[axis][n]["freqs"]) for n in range(M)]); c=jnp.stack([p[axis][n]["cos_c"] for n in range(M)])
        s=jnp.stack([p[axis][n]["sin_c"] for n in range(M)]); b=jnp.stack([p[axis][n]["bias"] for n in range(M)]); return f,c,s,b
    def fwd_pts(p,x,y):
        u=jnp.zeros_like(x)
        for n in range(M):
            wx=p["spatial_x"][n]["freqs"][None,:]*x; Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
            wy=p["spatial_y"][n]["freqs"][None,:]*y; Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
            u=u+p["mode_coeffs"][n]*Xn*Yn
        return u
    def grid(p,x1d,y1d):
        fx,cx,sx,bx=stx(p,"spatial_x"); fy,cy,sy,by=stx(p,"spatial_y")
        phx=x1d[:,:,None]*fx[None]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
        phy=y1d[:,:,None]*fy[None]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
        return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None]*Xv,Yv)
    xc,yc,fg,mask=data["xc"],data["yc"],data["fg"],data["mask"]; xb,yb,ub=data["xb"],data["yb"],data["ub"]
    def resid(p):
        fx,cx,sx,bx=stx(p,"spatial_x"); fy,cy,sy,by=stx(p,"spatial_y")
        phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
        phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
        mc=p["mode_coeffs"]; cX=mc[None]*Xv
        uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
        return (-(uxx+uyy)-fg)*mask
    p=init(random.PRNGKey(seed)); opt=optax.adam(LR); st=opt.init(p)
    def loss(p):
        r=resid(p); pde=jnp.sum(r**2)/jnp.sum(data["mask"]); bc=jnp.mean((fwd_pts(p,xb,yb)-ub)**2)
        return pde+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); bp=p; t0=time.time()
    def masked_l2(p):
        up=np.array(grid(p,data["x1d"],data["y1d"])); m=~data["inside_hole"]
        return l2(up[m],data["ue"][m]), up
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e,_=masked_l2(p)
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    e,up=masked_l2(bp)
    return {"method":"SVSNN","best_l2":best,"params":cp(p),"time_s":time.time()-t0,"u_pred":up}

# ---- FourierPINN baseline ----
def hvp(f,x,v):
    g=lambda z: jax.jvp(f,(z,),(v,))[1]; return jax.jvp(g,(x,),(v,))[1]
def fp_run(data,seed):
    def init(key):
        k1,k2,key=random.split(key,3); Wx=sfreqs(k1,MU).reshape(1,-1); Wy=sfreqs(k2,MU).reshape(1,-1)
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
    pde=data["pde"]; fpde=data["f_pde"]; xb,yb,ub=data["xb"],data["yb"],data["ub"]
    tx=jnp.zeros_like(pde).at[:,0].set(1.0); ty=jnp.zeros_like(pde).at[:,1].set(1.0)
    def loss(p):
        uxx=hvp(lambda z:fwd(p,z),pde,tx); uyy=hvp(lambda z:fwd(p,z),pde,ty)
        r=-(uxx+uyy)-fpde; bc=jnp.mean((fwd(p,jnp.concatenate([xb,yb],axis=-1))-ub)**2)
        return jnp.mean(r**2)+bc
    def pred(p):
        X,Y=np.meshgrid(np.array(data["x1d"]).squeeze(),np.array(data["y1d"]).squeeze(),indexing="ij")
        xy=jnp.asarray(np.stack([X.reshape(-1),Y.reshape(-1)],axis=1),jnp.float32)
        return np.array(fwd(p,xy)).reshape(N_TEST,N_TEST)
    p=init(random.PRNGKey(seed)); opt=optax.adam(LR); st=opt.init(p)
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); bp=p; t0=time.time(); m=~data["inside_hole"]
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            up=pred(p); e=l2(up[m],data["ue"][m])
            if e<best: best=e; bp=jax.tree.map(lambda z:z.copy(),p)
    up=pred(bp)
    return {"method":"FourierPINN","best_l2":best,"params":cp(p),"time_s":time.time()-t0,"u_pred":up}

def error_vs_distance(up,data):
    X,Y=data["X"],data["Y"]; m=~data["inside_hole"]
    dist=dist_to_boundary(X,Y); err=np.abs(up-data["ue"])
    dvals=dist[m]; evals=err[m]
    bins=np.linspace(0,dvals.max(),21); idx=np.digitize(dvals,bins)
    centers=[]; means=[]
    for b in range(1,len(bins)):
        sel=idx==b
        if sel.sum()>5: centers.append(0.5*(bins[b-1]+bins[b])); means.append(float(evals[sel].mean()))
    return centers,means

def main():
    out={}; saved_fields={}
    for runner in [svsnn_run,fp_run]:
        recs=[]; name=None
        for sd in SEEDS:
            data=build(sd); r=runner(data,sd); recs.append(r); name=r["method"]
            print(f"  {r['method']} seed={sd}: masked best L2={r['best_l2']:.3e} params={r['params']} t={r['time_s']:.1f}s",flush=True)
        b=np.array([r["best_l2"] for r in recs])
        c,mn=error_vs_distance(recs[0]["u_pred"],build(0))
        out[name]={"params":recs[0]["params"],"best_l2_mean":float(b.mean()),"best_l2_std":float(b.std()),
                   "best_l2_min":float(b.min()),"dist_centers":c,"err_means":mn}
        np.savez(os.path.join(SAVE,f"{name}_field.npz"),u_pred=recs[0]["u_pred"],ue=build(0)["ue"],
                 X=build(0)["X"],Y=build(0)["Y"],inside_hole=build(0)["inside_hole"])
    with open(os.path.join(SAVE,"E10_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E10 to",SAVE)

if __name__=="__main__": main()
