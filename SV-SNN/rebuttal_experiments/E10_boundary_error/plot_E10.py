"""Plot E10 near-boundary error on complex geometry."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E10_results.json")))
colors={"SVSNN":"#d62728","FourierPINN":"#ff7f0e"}
# error vs distance
fig,ax=plt.subplots(figsize=(7.5,5))
for m in R:
    c=R[m]["dist_centers"]; e=R[m]["err_means"]
    ax.plot(c,e,marker="o",color=colors.get(m,"gray"),label=f"{'SV-SNN' if m=='SVSNN' else m} (L2={R[m]['best_l2_mean']:.2e})")
ax.set_yscale("log"); ax.set_xlabel("distance to nearest boundary"); ax.set_ylabel("mean abs error")
ax.set_title("E10: Error vs distance-to-boundary (multiply-connected domain)"); ax.legend()
fig.savefig(os.path.join(FIG,"E10_error_vs_distance.png")); plt.close(fig)
# error fields
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
d=np.load(os.path.join(SD,"SVSNN_field.npz")); up,ue,X,Y,hole=d["u_pred"],d["ue"],d["X"],d["Y"],d["inside_hole"]
err=np.abs(up-ue); err_m=np.where(hole,np.nan,err); ue_m=np.where(hole,np.nan,ue); up_m=np.where(hole,np.nan,up)
im0=axes[0].pcolormesh(X,Y,ue_m,cmap="RdBu_r",shading="auto"); axes[0].set_title("Exact $u$ (hole masked)")
im1=axes[1].pcolormesh(X,Y,up_m,cmap="RdBu_r",shading="auto"); axes[1].set_title("SV-SNN prediction")
im2=axes[2].pcolormesh(X,Y,err_m,cmap="magma",shading="auto"); axes[2].set_title("Absolute error")
th=np.linspace(0,2*np.pi,200)
for a,im in zip(axes,[im0,im1,im2]):
    a.plot(0.5+0.2*np.cos(th),0.5+0.2*np.sin(th),"k-",lw=1.2); a.set_aspect("equal"); fig.colorbar(im,ax=a,fraction=0.046,pad=0.04)
fig.suptitle("E10: SV-SNN on [0,1]$^2$ with circular hole (seed 0)",y=1.02)
fig.savefig(os.path.join(FIG,"E10_fields.png")); plt.close(fig)
print("E10 figures written")
