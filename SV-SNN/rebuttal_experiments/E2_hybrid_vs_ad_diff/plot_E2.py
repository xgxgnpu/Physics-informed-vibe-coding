"""Plot E2: hybrid vs pure-AD differentiation. English journal style."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures")
os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})

with open(os.path.join(SD,"E2_results.json")) as f: R=json.load(f)
names=list(R.keys()); kap=[R[n]["kappa"] for n in names]
order=np.argsort(kap); names=[names[i] for i in order]; kap=[kap[i] for i in order]

best_h=[R[n]["hybrid"]["best_l2"] for n in names]; best_a=[R[n]["pure_ad"]["best_l2"] for n in names]
t_h=[R[n]["hybrid"]["time_s"] for n in names]; t_a=[R[n]["pure_ad"]["time_s"] for n in names]
spd=[R[n]["speedup"] for n in names]
mem_h=[R[n]["hybrid"]["peak_mem_mb"] for n in names]; mem_a=[R[n]["pure_ad"]["peak_mem_mb"] for n in names]
an32=[R[n]["deriv_probe"]["analytic32_vs_truth"] for n in names]
ad32=[R[n]["deriv_probe"]["ad32_vs_truth"] for n in names]

x=np.arange(len(names)); xl=[f"{int(k/np.pi)}$\\pi$" for k in kap]

fig,ax=plt.subplots(2,2,figsize=(13,9))
# (a) accuracy parity
ax[0,0].plot(x,best_h,"o-",label="Hybrid (analytic)",color="#d62728")
ax[0,0].plot(x,best_a,"s--",label="Pure AD",color="#1f77b4")
ax[0,0].set_yscale("log"); ax[0,0].set_xticks(x); ax[0,0].set_xticklabels(xl)
ax[0,0].set_xlabel("$\\kappa$"); ax[0,0].set_ylabel("Best relative $L_2$ error")
ax[0,0].set_title("(a) Accuracy parity across frequency"); ax[0,0].legend()
# (b) training time + speedup
ax[0,1].plot(x,t_h,"o-",label="Hybrid",color="#d62728"); ax[0,1].plot(x,t_a,"s--",label="Pure AD",color="#1f77b4")
ax[0,1].set_xticks(x); ax[0,1].set_xticklabels(xl); ax[0,1].set_xlabel("$\\kappa$"); ax[0,1].set_ylabel("Training time (s)")
ax2=ax[0,1].twinx(); ax2.plot(x,spd,"^:",color="green",label="Speedup"); ax2.set_ylabel("Speedup (AD/Hybrid)",color="green")
ax2.grid(False)
for xi,s in zip(x,spd): ax2.text(xi,s,f"{s:.2f}x",color="green",fontsize=8,ha="center",va="bottom")
ax[0,1].set_title("(b) Training time & speedup"); ax[0,1].legend(loc="upper left")
# (c) GPU memory
ax[1,0].plot(x,mem_h,"o-",label="Hybrid",color="#d62728"); ax[1,0].plot(x,mem_a,"s--",label="Pure AD",color="#1f77b4")
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels(xl); ax[1,0].set_xlabel("$\\kappa$"); ax[1,0].set_ylabel("Peak GPU memory (MB)")
ax[1,0].set_title("(c) Peak GPU memory"); ax[1,0].legend()
# (d) derivative numerical error vs frequency
ax[1,1].plot(x,an32,"o-",label="Analytic (float32) error",color="#d62728")
ax[1,1].plot(x,ad32,"s--",label="AD (float32) error",color="#1f77b4")
ax[1,1].set_yscale("log"); ax[1,1].set_xticks(x); ax[1,1].set_xticklabels(xl)
ax[1,1].set_xlabel("$\\kappa$"); ax[1,1].set_ylabel("Rel. error of $u_{xx}$ vs float64 truth")
ax[1,1].set_title("(d) 2nd-derivative numerical error growth"); ax[1,1].legend()
fig.suptitle("E2: Hybrid analytic differentiation vs pure automatic differentiation (SV-SNN, 3 seeds)",y=1.01)
fig.savefig(os.path.join(FIG,"E2_hybrid_vs_ad.png")); plt.close(fig)

# ---- grid-resolution study (aliasing) ----
gs_path=os.path.join(SD,"E2_gridstudy.json")
if os.path.exists(gs_path):
    G=json.load(open(gs_path)); ncs=sorted(int(k) for k in G)
    ppw=[G[str(n)]["pts_per_wavelength"] for n in ncs]; em=[G[str(n)]["best_l2_mean"] for n in ncs]
    es=[G[str(n)]["best_l2_std"] for n in ncs]; tm=[G[str(n)]["time_mean"] for n in ncs]
    fig,ax=plt.subplots(1,2,figsize=(13,4.8))
    ax[0].errorbar(ppw,em,yerr=es,marker="o",capsize=4,color="#d62728")
    ax[0].axvline(2.0,ls="--",color="gray"); ax[0].text(2.05,0.3,"Nyquist (2 pts/$\\lambda$)",rotation=90,fontsize=8,color="gray")
    ax[0].set_yscale("log"); ax[0].set_xlabel("grid points per wavelength"); ax[0].set_ylabel("hybrid best rel. $L_2$")
    ax[0].set_title("(a) $\\kappa{=}100\\pi$: accuracy restored by resolving the grid")
    for n,p,e in zip(ncs,ppw,em): ax[0].annotate(f"$N_c$={n}",(p,e),fontsize=7,xytext=(3,3),textcoords="offset points")
    ax[1].plot(ppw,tm,marker="s",color="#2ca02c"); ax[1].set_xlabel("grid points per wavelength"); ax[1].set_ylabel("training time (s)")
    ax[1].set_title("(b) Cost is nearly flat (grid eval is $O(N_c)$)"); ax[1].set_ylim(0,max(tm)*1.4)
    fig.suptitle("E2 supplement: the $\\kappa{=}100\\pi$ dip is grid aliasing, not a differentiation defect",y=1.02)
    fig.savefig(os.path.join(FIG,"E2_grid_study.png")); plt.close(fig)
print("E2 figure written to",FIG)
