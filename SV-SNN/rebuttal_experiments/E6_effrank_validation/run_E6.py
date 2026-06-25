"""
E6 - Effective rank validation (predictive, not just diagnostic)
================================================================
Addresses: R1.3, R3.3, R8.2.

We empirically validate the paper's claim that effective rank of the residual Jacobian
governs spectral bias / high-frequency convergence, and show it has PREDICTIVE value.

Procedure (2D Helmholtz, kappa in {20pi,40pi,80pi}):
  - Train SV-SNN (accelerated) and a vanilla PINN.
  - At checkpoints, compute the residual Jacobian J_F = d r_i / d theta_j on a fixed
    sample of collocation points, its singular values, and effective rank r_eff(0.99).
  - Record (early-epoch effective rank) vs (final relative L2) to test predictivity.
  - Also record the singular-value spectra at initialization across kappa to validate
    the conjectured cause: higher frequency -> faster singular-value decay -> lower r_eff.

3 seeds. Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, vmap, value_and_grad
from jax.flatten_util import ravel_pytree
import optax
from pyDOE import lhs

EPOCHS=6000; LR=1e-3; N_PDE=4000; N_BC=512; N_TEST=200; EVAL_EVERY=100
NC=160; M=8; K=64; SEEDS=[0,1,2]; N_JAC=300  # NC=160 -> >=4 pts/wavelength even at kappa=80pi (avoids grid aliasing, cf. E2)
CKPTS=[100,500,1000,2000,4000,5990]
KAPPAS={"20pi":20*np.pi,"40pi":40*np.pi,"80pi":80*np.pi}
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def eff_rank(sv,eta=0.99):
    e=sv**2; c=np.cumsum(e)/np.sum(e); return int(np.searchsorted(c,eta)+1)

def sfreqs(key,Kk,wc):
    nl=Kk//4; ncc=Kk//2; nh=Kk-nl-ncc; _,k2,k3=jax.random.split(key,3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*30.0+wc),
                                     jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)]))

def make_data(kappa,seed):
    np.random.seed(seed); nps=N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    pde=lhs(2,samples=N_PDE)
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij"); ue=np.sin(kappa*X)*np.sin(kappa*Y)
    xc=np.linspace(0,1,NC); yc=np.linspace(0,1,NC); Xc,Yc=np.meshgrid(xc,yc,indexing="ij"); fg=kappa**2*np.sin(kappa*Xc)*np.sin(kappa*Yc)
    jx=pde[:N_JAC,0:1]; jy=pde[:N_JAC,1:2]
    return {"xb":jnp.asarray(xb,jnp.float32),"yb":jnp.asarray(yb,jnp.float32),
            "x_pde":jnp.asarray(pde[:,0:1],jnp.float32),"y_pde":jnp.asarray(pde[:,1:2],jnp.float32),
            "ue":ue,"X":X,"Y":Y,"xc":jnp.asarray(xc.reshape(-1,1),jnp.float32),"yc":jnp.asarray(yc.reshape(-1,1),jnp.float32),
            "fg":jnp.asarray(fg,jnp.float32),"x1d":jnp.asarray(x1.reshape(-1,1),jnp.float32),"y1d":jnp.asarray(y1.reshape(-1,1),jnp.float32),
            "jx":jnp.asarray(jx,jnp.float32),"jy":jnp.asarray(jy,jnp.float32)}

# ---------------- SV-SNN ----------------
def svsnn_init(key,kappa):
    keys=jax.random.split(key,M*6+1); ki=0; sx,sy=[],[]
    for _ in range(M):
        sx.append({"freqs":sfreqs(keys[ki],K,kappa),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        sy.append({"freqs":sfreqs(keys[ki],K,kappa),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(M,))*0.1}
def svsnn_fwd(p,x,y):
    u=jnp.zeros_like(x)
    for n in range(M):
        wx=p["spatial_x"][n]["freqs"][None,:]*x; Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
        wy=p["spatial_y"][n]["freqs"][None,:]*y; Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
        u=u+p["mode_coeffs"][n]*Xn*Yn
    return u
def svsnn_grid(p,x1d,y1d):
    fx=jnp.stack([p["spatial_x"][n]["freqs"] for n in range(M)]); cx=jnp.stack([p["spatial_x"][n]["cos_c"] for n in range(M)]); sx=jnp.stack([p["spatial_x"][n]["sin_c"] for n in range(M)]); bx=jnp.stack([p["spatial_x"][n]["bias"] for n in range(M)])
    fy=jnp.stack([p["spatial_y"][n]["freqs"] for n in range(M)]); cy=jnp.stack([p["spatial_y"][n]["cos_c"] for n in range(M)]); sy=jnp.stack([p["spatial_y"][n]["sin_c"] for n in range(M)]); by=jnp.stack([p["spatial_y"][n]["bias"] for n in range(M)])
    phx=x1d[:,:,None]*fx[None]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
    phy=y1d[:,:,None]*fy[None]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
    return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None]*Xv,Yv)
def svsnn_resid_grid(p,xc,yc,fg,kappa):
    fx=jnp.stack([jax.lax.stop_gradient(p["spatial_x"][n]["freqs"]) for n in range(M)]); cx=jnp.stack([p["spatial_x"][n]["cos_c"] for n in range(M)]); sx=jnp.stack([p["spatial_x"][n]["sin_c"] for n in range(M)]); bx=jnp.stack([p["spatial_x"][n]["bias"] for n in range(M)])
    fy=jnp.stack([jax.lax.stop_gradient(p["spatial_y"][n]["freqs"]) for n in range(M)]); cy=jnp.stack([p["spatial_y"][n]["cos_c"] for n in range(M)]); sy=jnp.stack([p["spatial_y"][n]["sin_c"] for n in range(M)]); by=jnp.stack([p["spatial_y"][n]["bias"] for n in range(M)])
    phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
    phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
    mc=p["mode_coeffs"]; cX=mc[None]*Xv
    u=jnp.einsum("nm,jm->nj",cX,Yv); uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
    return -(uxx+uyy)-kappa**2*u-fg
def svsnn_resid_pts(p,x,y,kappa):
    """residual at scattered points for Jacobian (analytic)."""
    def one(xs,ys):
        u_xx=0.0; u_yy=0.0; u=0.0
        for n in range(M):
            wx=p["spatial_x"][n]["freqs"]*xs
            Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx))+p["spatial_x"][n]["bias"][0]
            Xdd=jnp.sum(-(p["spatial_x"][n]["freqs"]**2)*(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx)))
            wy=p["spatial_y"][n]["freqs"]*ys
            Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy))+p["spatial_y"][n]["bias"][0]
            Ydd=jnp.sum(-(p["spatial_y"][n]["freqs"]**2)*(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy)))
            u=u+p["mode_coeffs"][n]*Xn*Yn; u_xx=u_xx+p["mode_coeffs"][n]*Xdd*Yn; u_yy=u_yy+p["mode_coeffs"][n]*Xn*Ydd
        f=kappa**2*jnp.sin(kappa*xs)*jnp.sin(kappa*ys)
        return -(u_xx+u_yy)-kappa**2*u-f
    return vmap(one)(x.squeeze(),y.squeeze())

# ---------------- PINN ----------------
def pinn_init(key):
    dims=[2,128,128,128,128,1]; layers=[]; k=key
    for i in range(len(dims)-1):
        k,sub=random.split(k); lim=jnp.sqrt(6.0/(dims[i]+dims[i+1]))
        layers.append({"w":random.uniform(sub,(dims[i],dims[i+1]),minval=-lim,maxval=lim),"b":jnp.zeros((dims[i+1],))})
    return {"layers":layers}
def pinn_fwd(p,xy):
    h=xy; n=len(p["layers"])
    for i,L in enumerate(p["layers"]):
        h=h@L["w"]+L["b"]
        if i<n-1: h=jnp.tanh(h)
    return h
def pinn_resid_pts(p,x,y,kappa):
    def one(xs,ys):
        uf=lambda a,b: pinn_fwd(p,jnp.array([[a,b]]))[0,0]
        uxx=jax.grad(jax.grad(uf,0),0)(xs,ys); uyy=jax.grad(jax.grad(uf,1),1)(xs,ys)
        f=kappa**2*jnp.sin(kappa*xs)*jnp.sin(kappa*ys)
        return -(uxx+uyy)-kappa**2*uf(xs,ys)-f
    return vmap(one)(x.squeeze(),y.squeeze())

def jac_effrank(resid_fn_pts,params,data,kappa):
    flat,unravel=ravel_pytree(params)
    def rflat(fp): return resid_fn_pts(unravel(fp),data["jx"],data["jy"],kappa)
    J=np.array(jax.jacrev(rflat)(flat))
    sv=np.linalg.svd(J,compute_uv=False)
    return eff_rank(sv), sv

def train_model(which,data,kappa,seed):
    if which=="svsnn":
        p=svsnn_init(random.PRNGKey(seed),kappa)
        xc,yc,fg=data["xc"],data["yc"],data["fg"]; xb,yb=data["xb"],data["yb"]
        def loss(p):
            r=svsnn_resid_grid(p,xc,yc,fg,kappa)
            bc=jnp.mean(svsnn_fwd(p,xb,yb)**2)
            return jnp.mean(r**2)+bc
        pred=lambda p: np.array(svsnn_grid(p,data["x1d"],data["y1d"]))
        resid_pts=svsnn_resid_pts
    else:
        p=pinn_init(random.PRNGKey(seed))
        xp=data["x_pde"].squeeze(); yp=data["y_pde"].squeeze(); xb,yb=data["xb"],data["yb"]
        ub=jnp.zeros((xb.shape[0],1))
        def res_single(p,xs,ys):
            uf=lambda a,b: pinn_fwd(p,jnp.array([[a,b]]))[0,0]
            uxx=jax.grad(jax.grad(uf,0),0)(xs,ys); uyy=jax.grad(jax.grad(uf,1),1)(xs,ys)
            f=kappa**2*jnp.sin(kappa*xs)*jnp.sin(kappa*ys)
            return -(uxx+uyy)-kappa**2*uf(xs,ys)-f
        rb=vmap(res_single,in_axes=(None,0,0))
        def loss(p):
            r=rb(p,xp,yp); bc=jnp.mean((pinn_fwd(p,jnp.concatenate([xb,yb],axis=-1))-ub)**2); return jnp.mean(r**2)+bc
        def pred(p):
            X,Y=np.meshgrid(np.array(data["x1d"]).squeeze(),np.array(data["y1d"]).squeeze(),indexing="ij")
            xy=jnp.asarray(np.stack([X.reshape(-1),Y.reshape(-1)],axis=1),jnp.float32)
            return np.array(pinn_fwd(p,xy)).reshape(N_TEST,N_TEST)
        resid_pts=pinn_resid_pts
    opt=optax.adam(LR); st=opt.init(p)
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    erank_traj={}; best=float("inf"); 
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if (ep+1) in CKPTS or ep in CKPTS:
            r_eff,_=jac_effrank(resid_pts,p,data,kappa); erank_traj[ep]=r_eff
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=l2(pred(p),data["ue"]); best=min(best,e)
    final=l2(pred(p),data["ue"])
    # init spectrum
    p0 = svsnn_init(random.PRNGKey(seed),kappa) if which=="svsnn" else pinn_init(random.PRNGKey(seed))
    _,sv0=jac_effrank(resid_pts,p0,data,kappa)
    return {"erank_traj":{int(k):int(v) for k,v in erank_traj.items()},"best_l2":best,"final_l2":final,
            "init_sv":sv0[:200].tolist()}

def main():
    out={}
    for name,kappa in KAPPAS.items():
        print(f"\n##### kappa={name} #####",flush=True); out[name]={}
        for which in ["svsnn","pinn"]:
            recs=[]
            for sd in SEEDS:
                r=train_model(which,make_data(kappa,sd),kappa,sd); recs.append(r)
                print(f"  {which} seed={sd}: final L2={r['final_l2']:.3e} erank={r['erank_traj']}",flush=True)
            out[name][which]={"seeds":recs,
                "final_l2_mean":float(np.mean([r["final_l2"] for r in recs])),
                "erank_mean":{str(c):float(np.mean([r["erank_traj"].get(c,r["erank_traj"].get(c+1,np.nan)) for r in recs])) for c in CKPTS}}
    with open(os.path.join(SAVE,"E6_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E6 to",SAVE)

if __name__=="__main__": main()
