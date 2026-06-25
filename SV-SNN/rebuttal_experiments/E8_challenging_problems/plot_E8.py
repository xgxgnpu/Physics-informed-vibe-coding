"""Plot E8 challenging problems."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E8_results.json")))
probs=list(R.keys()); methods=["SVSNN","FourierPINN","PINN"]
colors={"SVSNN":"#d62728","FourierPINN":"#ff7f0e","PINN":"#9467bd"}
titles={"Q1_nonsep":"Q1 Non-separable\n$\\sin(\\kappa(x^2{+}y^2))$","Q2_packet":"Q2 Localized wave packet\n(non-periodic)","Q3_hetero":"Q3 Heterogeneous Helmholtz\n$\\kappa(x)$ varying"}
# bar comparison
fig,ax=plt.subplots(figsize=(11,5.5))
x=np.arange(len(probs)); w=0.25
for i,m in enumerate(methods):
    means=[R[p][m]["best_l2_mean"] for p in probs]; stds=[R[p][m]["best_l2_std"] for p in probs]
    ax.bar(x+(i-1)*w,means,w,yerr=stds,capsize=3,label=m if m!="SVSNN" else "SV-SNN",color=colors[m],alpha=0.85,edgecolor="black",lw=0.5)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([titles[p] for p in probs],fontsize=9)
ax.set_ylabel("best relative $L_2$ error"); ax.legend(); ax.set_title("E8: Challenging problems (3 seeds mean$\\pm$std)")
fig.savefig(os.path.join(FIG,"E8_bars.png"),bbox_inches="tight"); plt.close(fig)
# fields
fig,axes=plt.subplots(len(probs),3,figsize=(13,4.2*len(probs)))
for r,p in enumerate(probs):
    d=np.load(os.path.join(SD,f"{p}_SVSNN_pred.npz")); up,ue,X,Y=d["u_pred"],d["ue"],d["X"],d["Y"]; err=np.abs(up-ue)
    im0=axes[r,0].pcolormesh(X,Y,ue,cmap="RdBu_r",shading="auto"); axes[r,0].set_title(f"{p}: exact")
    im1=axes[r,1].pcolormesh(X,Y,up,cmap="RdBu_r",shading="auto"); axes[r,1].set_title("SV-SNN pred")
    im2=axes[r,2].pcolormesh(X,Y,err,cmap="magma",shading="auto"); axes[r,2].set_title("abs error")
    for c,im in zip(range(3),[im0,im1,im2]):
        axes[r,c].set_aspect("equal"); fig.colorbar(im,ax=axes[r,c],fraction=0.046,pad=0.04)
fig.suptitle("E8: SV-SNN fields on challenging problems (seed 0)",y=1.005)
fig.savefig(os.path.join(FIG,"E8_fields.png"),bbox_inches="tight"); plt.close(fig)
print("E8 figures written")
