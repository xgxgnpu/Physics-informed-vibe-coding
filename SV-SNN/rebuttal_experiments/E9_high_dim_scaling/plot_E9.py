"""Plot E9 3D / dimension scaling."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E9_results.json")))
A=R["partA_3d"]; B=R["partB_scaling"]
fig,ax=plt.subplots(1,3,figsize=(16,5))
# (a) 3D Poisson method comparison
methods=[m for m in ["SVSNN","FourierPINN","PINN"] if m in A]
colors={"SVSNN":"#d62728","FourierPINN":"#ff7f0e","PINN":"#9467bd"}
means=[np.mean([s["best_l2"] for s in A[m]]) for m in methods]; stds=[np.std([s["best_l2"] for s in A[m]]) for m in methods]
pars=[A[m][0]["params"] for m in methods]
x=np.arange(len(methods)); ax[0].bar(x,means,yerr=stds,capsize=4,color=[colors[m] for m in methods],alpha=0.85,edgecolor="black")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([m if m!="SVSNN" else "SV-SNN" for m in methods])
ax[0].set_ylabel("best rel. $L_2$"); ax[0].set_title("(a) TRUE 3D Poisson $[0,1]^3$")
for xi,mn,pp in zip(x,means,pars): ax[0].text(xi,mn,f"{mn:.2e}\n{pp}p",ha="center",va="bottom",fontsize=7)
# (b) dimension scaling: params & accuracy
ds=sorted(int(k) for k in B); par=[B[str(d)]["params"] for d in ds]; err=[B[str(d)]["best_l2_mean"] for d in ds]; tim=[B[str(d)]["time_mean"] for d in ds]
ax[1].plot(ds,par,marker="s",color="#1f77b4",label="parameters")
ax[1].set_xlabel("dimension $d$"); ax[1].set_ylabel("parameters",color="#1f77b4"); ax[1].set_xticks(ds)
ax2=ax[1].twinx(); ax2.plot(ds,err,marker="o",color="#d62728",label="rel. $L_2$"); ax2.set_yscale("log"); ax2.set_ylabel("best rel. $L_2$",color="#d62728"); ax2.grid(False)
ax[1].set_title("(b) SV-SNN scaling: params (linear) & accuracy")
# (c) time scaling
ax[2].plot(ds,tim,marker="^",color="#2ca02c"); ax[2].set_xlabel("dimension $d$"); ax[2].set_ylabel("training time (s)"); ax[2].set_xticks(ds); ax[2].set_title("(c) SV-SNN training time vs $d$")
fig.suptitle("E9: 3D benchmark and dimensional scalability of SV-SNN",y=1.02)
fig.savefig(os.path.join(FIG,"E9_highdim.png")); plt.close(fig)
# 3D slice
if os.path.exists(os.path.join(SD,"svsnn_3d_pred.npz")):
    d=np.load(os.path.join(SD,"svsnn_3d_pred.npz")); up,ue=d["u_pred"],d["ue"]; k=up.shape[2]//2
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    im0=axes[0].imshow(ue[:,:,k],cmap="RdBu_r",origin="lower"); axes[0].set_title("Exact (z-mid slice)")
    im1=axes[1].imshow(up[:,:,k],cmap="RdBu_r",origin="lower"); axes[1].set_title("SV-SNN (z-mid slice)")
    im2=axes[2].imshow(np.abs(up-ue)[:,:,k],cmap="magma",origin="lower"); axes[2].set_title("abs error")
    for a,im in zip(axes,[im0,im1,im2]): fig.colorbar(im,ax=a,fraction=0.046,pad=0.04)
    fig.suptitle("E9: 3D Poisson mid-plane slice (SV-SNN)",y=1.03); fig.savefig(os.path.join(FIG,"E9_3d_slice.png")); plt.close(fig)
print("E9 figures written")
