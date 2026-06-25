"""
E7 - Mode number N scaling
==========================
Addresses: R5.4 (how N affects accuracy, params, training time; limits for non-separable),
           R8.5 (N vs desired accuracy).

Separable SV-SNN with N in {1,2,4,6,8,12,16} on:
  (P1) separable     : Helmholtz kappa=24pi, u=sin(kx)sin(ky)  (rank-1 -> needs few modes)
  (P2) non-separable : Poisson, u = sum_{i=1}^{4} sin(a_i x) sin(b_i y) with a_i != b_i
                       (genuine rank-4 cross-coupling -> needs more modes)
Report best L2 / params / training time vs N (mean over 3 seeds). Self-contained.
"""
import os, sys, time, json
sys.stdout.reconfigure(line_buffering=True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax

EPOCHS=8000; LR=1e-3; N_BC=1024; N_TEST=200; EVAL_EVERY=100; NC=100; K=32; SEEDS=[0,1,2]
NS=[1,2,4,6,8,12,16]; W_CHAR=24*np.pi
SAVE=os.path.join(os.path.dirname(os.path.abspath(__file__)),"saved_data"); os.makedirs(SAVE,exist_ok=True)

def l2(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
def cp(p): return int(sum(x.size for x in jax.tree.leaves(p) if hasattr(x,"size")))
def sfreqs(key,wc):
    nl=K//4; ncc=K//2; nh=K-nl-ncc; _,k2,k3=jax.random.split(key,3)
    return jnp.sort(jnp.concatenate([jnp.linspace(1.0,wc,nl),jnp.abs(jax.random.normal(k2,(ncc,))*30.0+wc),
                                     jax.random.uniform(k3,(nh,),minval=wc*0.5,maxval=wc)]))

def init_sep(key,Mn,wc):
    keys=jax.random.split(key,Mn*6+1); ki=0; sx,sy=[],[]
    for _ in range(Mn):
        sx.append({"freqs":sfreqs(keys[ki],wc),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
        sy.append({"freqs":sfreqs(keys[ki],wc),"cos_c":jax.random.normal(keys[ki+1],(K,))*0.1,"sin_c":jax.random.normal(keys[ki+2],(K,))*0.1,"bias":jnp.zeros(1)}); ki+=3
    return {"spatial_x":sx,"spatial_y":sy,"mode_coeffs":jax.random.normal(keys[ki],(Mn,))*0.1}
def sep_fwd(p,x,y,Mn):
    u=jnp.zeros_like(x)
    for n in range(Mn):
        wx=p["spatial_x"][n]["freqs"][None,:]*x; Xn=jnp.sum(p["spatial_x"][n]["cos_c"]*jnp.cos(wx)+p["spatial_x"][n]["sin_c"]*jnp.sin(wx),axis=1,keepdims=True)+p["spatial_x"][n]["bias"]
        wy=p["spatial_y"][n]["freqs"][None,:]*y; Yn=jnp.sum(p["spatial_y"][n]["cos_c"]*jnp.cos(wy)+p["spatial_y"][n]["sin_c"]*jnp.sin(wy),axis=1,keepdims=True)+p["spatial_y"][n]["bias"]
        u=u+p["mode_coeffs"][n]*Xn*Yn
    return u
def stacks(p,axis,Mn):
    f=jnp.stack([jax.lax.stop_gradient(p[axis][n]["freqs"]) for n in range(Mn)])
    c=jnp.stack([p[axis][n]["cos_c"] for n in range(Mn)]); s=jnp.stack([p[axis][n]["sin_c"] for n in range(Mn)]); b=jnp.stack([p[axis][n]["bias"] for n in range(Mn)])
    return f,c,s,b
def sep_grid(p,x1d,y1d,Mn):
    fx,cx,sx,bx=stacks(p,"spatial_x",Mn); fy,cy,sy,by=stacks(p,"spatial_y",Mn)
    phx=x1d[:,:,None]*fx[None]; Xv=jnp.sum(cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx),axis=2)+bx[None,:,0]
    phy=y1d[:,:,None]*fy[None]; Yv=jnp.sum(cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy),axis=2)+by[None,:,0]
    return jnp.einsum("nm,jm->nj",p["mode_coeffs"][None]*Xv,Yv)
def residual(p,xc,yc,fg,Mn,helm_kappa):
    fx,cx,sx,bx=stacks(p,"spatial_x",Mn); fy,cy,sy,by=stacks(p,"spatial_y",Mn)
    phx=xc[:,:,None]*fx[None]; Xt=cx[None]*jnp.cos(phx)+sx[None]*jnp.sin(phx); Xv=jnp.sum(Xt,axis=2)+bx[None,:,0]; Xdd=jnp.sum(-(fx[None]**2)*Xt,axis=2)
    phy=yc[:,:,None]*fy[None]; Yt=cy[None]*jnp.cos(phy)+sy[None]*jnp.sin(phy); Yv=jnp.sum(Yt,axis=2)+by[None,:,0]; Ydd=jnp.sum(-(fy[None]**2)*Yt,axis=2)
    mc=p["mode_coeffs"]; cX=mc[None]*Xv
    u=jnp.einsum("nm,jm->nj",cX,Yv); uxx=jnp.einsum("nm,jm->nj",mc[None]*Xdd,Yv); uyy=jnp.einsum("nm,jm->nj",cX,Ydd)
    if helm_kappa is None: return -(uxx+uyy)-fg
    return -(uxx+uyy)-helm_kappa**2*u-fg

def make_problem(kind,seed):
    np.random.seed(seed); nps=N_BC//4; t=np.linspace(0,1,nps).reshape(-1,1)
    xb=np.vstack([np.zeros((nps,1)),np.ones((nps,1)),t,t]); yb=np.vstack([t,t,np.zeros((nps,1)),np.ones((nps,1))])
    x1=np.linspace(0,1,N_TEST); y1=np.linspace(0,1,N_TEST); X,Y=np.meshgrid(x1,y1,indexing="ij")
    xc=np.linspace(0,1,NC); yc=np.linspace(0,1,NC); Xc,Yc=np.meshgrid(xc,yc,indexing="ij")
    if kind=="separable":
        kappa=24*np.pi
        ue=np.sin(kappa*X)*np.sin(kappa*Y); fg=kappa**2*np.sin(kappa*Xc)*np.sin(kappa*Yc)
        ub=np.zeros((xb.shape[0],1)); helm=kappa
        def uex(x,y): return np.sin(kappa*x)*np.sin(kappa*y)
    else:
        # rank-4 non-separable Poisson: sum sin(a_i x) sin(b_i y), a_i!=b_i
        A=[6*np.pi,10*np.pi,16*np.pi,22*np.pi]; B=[22*np.pi,16*np.pi,10*np.pi,6*np.pi]
        def uex(x,y):
            return sum(np.sin(a*x)*np.sin(b*y) for a,b in zip(A,B))
        ue=uex(X,Y)
        fg=sum((a**2+b**2)*np.sin(a*Xc)*np.sin(b*Yc) for a,b in zip(A,B))  # -Lap u
        ub=uex(xb,yb); helm=None
    return {"xb":jnp.asarray(xb,jnp.float32),"yb":jnp.asarray(yb,jnp.float32),"ub":jnp.asarray(ub,jnp.float32),
            "ue":ue,"X":X,"Y":Y,"xc":jnp.asarray(xc.reshape(-1,1),jnp.float32),"yc":jnp.asarray(yc.reshape(-1,1),jnp.float32),
            "fg":jnp.asarray(fg,jnp.float32),"x1d":jnp.asarray(x1.reshape(-1,1),jnp.float32),"y1d":jnp.asarray(y1.reshape(-1,1),jnp.float32),"helm":helm}

def train(Mn,data,seed):
    p=init_sep(random.PRNGKey(seed),Mn,W_CHAR); opt=optax.adam(LR); st=opt.init(p)
    xc,yc,fg=data["xc"],data["yc"],data["fg"]; xb,yb,ub=data["xb"],data["yb"],data["ub"]; helm=data["helm"]
    def loss(p):
        r=residual(p,xc,yc,fg,Mn,helm); bc=jnp.mean((sep_fwd(p,xb,yb,Mn)-ub)**2); return jnp.mean(r**2)+bc
    @jit
    def step(p,st):
        l,g=value_and_grad(loss)(p); u,st=opt.update(g,st,p); return optax.apply_updates(p,u),st,l
    for _ in range(2): p,st,_=step(p,st)
    best=float("inf"); t0=time.time()
    for ep in range(2,EPOCHS):
        p,st,l=step(p,st)
        if ep%EVAL_EVERY==0 or ep==EPOCHS-1:
            e=l2(np.array(sep_grid(p,data["x1d"],data["y1d"],Mn)),data["ue"]); best=min(best,e)
    return best,cp(p),time.time()-t0

def main():
    out={}
    for kind in ["separable","nonseparable"]:
        print(f"\n##### {kind} #####",flush=True); out[kind]={}
        for Mn in NS:
            bs=[]; ps=0; ts=[]
            for sd in SEEDS:
                b,pc,tt=train(Mn,make_problem(kind,sd),sd); bs.append(b); ps=pc; ts.append(tt)
            bs=np.array(bs); ts=np.array(ts)
            out[kind][str(Mn)]={"params":ps,"best_l2_mean":float(bs.mean()),"best_l2_std":float(bs.std()),
                                "best_l2_min":float(bs.min()),"time_mean":float(ts.mean())}
            print(f"  N={Mn:2d}: params={ps:5d} best L2={bs.mean():.3e} t={ts.mean():.1f}s",flush=True)
    with open(os.path.join(SAVE,"E7_results.json"),"w") as f: json.dump(out,f,indent=2)
    print("\nSaved E7 to",SAVE)

if __name__=="__main__": main()
