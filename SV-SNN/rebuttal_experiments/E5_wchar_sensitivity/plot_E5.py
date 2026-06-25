"""Plot E5 w_char sensitivity & auto-estimation."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E5_results.json")))

fig,ax=plt.subplots(2,3,figsize=(16,9))
# (a) scan
sc=R["a_scan"]; ratios=sorted(float(k) for k in sc); means=[sc[f"{r:.2f}"]["mean"] for r in ratios]; stds=[sc[f"{r:.2f}"]["std"] for r in ratios]
ax[0,0].errorbar(ratios,means,yerr=stds,marker="o",capsize=4,color="#d62728")
ax[0,0].axvline(1.0,ls="--",color="gray"); ax[0,0].set_yscale("log")
ax[0,0].set_xlabel("$w_{char}/\\kappa$"); ax[0,0].set_ylabel("Best rel. $L_2$"); ax[0,0].set_title("(a) Sensitivity to $w_{char}$ (U-shaped)")
# (b) multimodal
mm=R["b_multimodal"]; keys=["k1","k2","mean","max","split_k1k2"]; keys=[k for k in keys if k in mm]
vals=[mm[k]["mean"] for k in keys]
ax[0,1].bar(range(len(keys)),vals,color="#1f77b4",alpha=0.85,edgecolor="black")
ax[0,1].set_xticks(range(len(keys))); ax[0,1].set_xticklabels(keys,rotation=20); ax[0,1].set_yscale("log")
ax[0,1].set_ylabel("Best rel. $L_2$"); ax[0,1].set_title("(b) Multi-modal: choice of $w_{char}$")
for i,v in enumerate(vals): ax[0,1].text(i,v,f"{v:.2e}",ha="center",va="bottom",fontsize=7)
# (c) chirp frozen vs learnable
ch=R["c_chirp"]; ax[0,2].bar([0,1],[ch["frozen_mean"],ch["learnable_mean"]],yerr=[ch["frozen_std"],ch["learnable_std"]],
                              capsize=4,color=["#7f7f7f","#2ca02c"],alpha=0.85,edgecolor="black")
ax[0,2].set_xticks([0,1]); ax[0,2].set_xticklabels(["frozen","learnable"]); ax[0,2].set_yscale("log")
ax[0,2].set_ylabel("Best rel. $L_2$"); ax[0,2].set_title("(c) Spatial chirp: frozen vs learnable freq")
# (d) FFT
fft=R["d_fft"]
ax[1,0].bar([0,1],[fft["best_l2_manual"],fft["best_l2_auto"]],color=["#d62728","#ff7f0e"],alpha=0.85,edgecolor="black")
ax[1,0].set_xticks([0,1]); ax[1,0].set_xticklabels([f"manual\n$w$={fft['true_kappa']:.0f}",f"FFT-auto\n$w$={fft['est_wchar']:.0f}"])
ax[1,0].set_yscale("log"); ax[1,0].set_ylabel("Best rel. $L_2$")
ax[1,0].set_title(f"(d) FFT auto-estimate (rel err {fft['rel_err']*100:.1f}%)")
# (e) frozen vs learnable when wrong
fe=R["e_freeze"]; ax[1,1].bar([0,1],[fe["frozen_mean"],fe["learnable_mean"]],yerr=[fe["frozen_std"],fe["learnable_std"]],
                              capsize=4,color=["#7f7f7f","#2ca02c"],alpha=0.85,edgecolor="black")
ax[1,1].set_xticks([0,1]); ax[1,1].set_xticklabels(["frozen","learnable"]); ax[1,1].set_yscale("log")
ax[1,1].set_ylabel("Best rel. $L_2$"); ax[1,1].set_title("(e) Wrong $w_{char}=0.6\\kappa$: freeze vs learn")
# (f) noise
fn=R["f_noise"]; eps=sorted(float(k) for k in fn); m=[fn[f"{e:.2f}"]["mean"] for e in eps]; s=[fn[f"{e:.2f}"]["std"] for e in eps]
ax[1,2].errorbar(eps,m,yerr=s,marker="s",capsize=4,color="#9467bd"); ax[1,2].set_yscale("log")
ax[1,2].set_xlabel("noise level $\\epsilon$"); ax[1,2].set_ylabel("Best rel. $L_2$"); ax[1,2].set_title("(f) Noisy frequency prior")
fig.suptitle("E5: Characteristic frequency $w_{char}$ — sensitivity & automatic estimation",y=1.01)
fig.savefig(os.path.join(FIG,"E5_wchar.png")); plt.close(fig)
print("E5 figure written")
