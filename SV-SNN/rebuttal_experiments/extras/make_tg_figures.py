"""
Taylor-Green vortex prediction & error maps (addresses R7.4).
Reads the EXISTING accelerated-suite data in svsnn_acceleration/case8_taylor_green/saved_data
(this is a non-experiment deliverable: re-plotting already-computed results) and writes
publication-style figures to extras/figures/.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.abspath(os.path.join(HERE,"..","..","svsnn_acceleration","case8_taylor_green","saved_data"))
FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"figure.dpi":150,"savefig.bbox":"tight"})

d=np.load(os.path.join(SRC,"SVSNN_accel_prediction.npz"))
X,Y=d["X"][:,:,0],d["Y"][:,:,0]; tk=d["u_pred"].shape[2]//2
comp={"u":("u_pred","u_exact"),"v":("v_pred","v_exact"),"p":("p_pred","p_exact")}

# prediction vs exact vs error for u, v, p at mid time
fig,axes=plt.subplots(3,3,figsize=(13,11))
for r,(name,(pk,ek)) in enumerate(comp.items()):
    up=d[pk][:,:,tk]; ue=d[ek][:,:,tk]; err=np.abs(up-ue)
    im0=axes[r,0].pcolormesh(X,Y,ue,cmap="RdBu_r",shading="auto"); axes[r,0].set_ylabel(f"${name}$",fontsize=14)
    im1=axes[r,1].pcolormesh(X,Y,up,cmap="RdBu_r",shading="auto")
    im2=axes[r,2].pcolormesh(X,Y,err,cmap="magma",shading="auto")
    if r==0:
        axes[r,0].set_title("Exact"); axes[r,1].set_title("SV-SNN prediction"); axes[r,2].set_title("Absolute error")
    for c,im in zip(range(3),[im0,im1,im2]):
        axes[r,c].set_aspect("equal"); fig.colorbar(im,ax=axes[r,c],fraction=0.046,pad=0.04)
fig.suptitle("Taylor-Green vortex: SV-SNN prediction and error maps (mid-time slice)",y=1.0)
fig.savefig(os.path.join(FIG,"TaylorGreen_fields.png")); plt.close(fig)

# velocity magnitude + error comparison across methods
methods=["SVSNN_accel","SPINN","FourierPINN","SIREN","PINN"]
fig,axes=plt.subplots(1,len(methods),figsize=(4*len(methods),3.8))
for ax,m in zip(axes,methods):
    fp=os.path.join(SRC,f"{m}_prediction.npz")
    if not os.path.exists(fp): ax.axis("off"); continue
    dm=np.load(fp); up=dm["u_pred"][:,:,tk]; vp=dm["v_pred"][:,:,tk]; ue=dm["u_exact"][:,:,tk]; ve=dm["v_exact"][:,:,tk]
    err=np.sqrt((up-ue)**2+(vp-ve)**2)
    im=ax.pcolormesh(X,Y,err,cmap="magma",shading="auto"); ax.set_aspect("equal")
    ax.set_title(f"{'SV-SNN' if m=='SVSNN_accel' else m}"); fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
fig.suptitle("Taylor-Green: velocity-magnitude error |$(u,v)-(u^*,v^*)$| (mid-time)",y=1.03)
fig.savefig(os.path.join(FIG,"TaylorGreen_velocity_error.png")); plt.close(fig)

# convergence histories
fig,ax=plt.subplots(figsize=(7,5))
colors={"SVSNN_accel":"#d62728","SPINN":"#1f77b4","FourierPINN":"#ff7f0e","SIREN":"#2ca02c","PINN":"#9467bd"}
for m in methods:
    hp=os.path.join(SRC,f"{m}_history.npz")
    if not os.path.exists(hp): continue
    h=np.load(hp); ax.semilogy(h["eval_epochs"],h["l2_error"],label="SV-SNN" if m=="SVSNN_accel" else m,color=colors[m])
ax.set_xlabel("epoch"); ax.set_ylabel("relative $L_2$ error"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Taylor-Green vortex: convergence history")
fig.savefig(os.path.join(FIG,"TaylorGreen_convergence.png")); plt.close(fig)
print("Taylor-Green figures written to",FIG)
