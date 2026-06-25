"""Plot E3 component ablation. English journal style."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E3_results.json")))
order=["V0_Full","V1_NoSeparation","V2_NoSpectral","V3_NoMultilevel","V4_NoAnalytic"]
labels={"V0_Full":"Full SV-SNN","V1_NoSeparation":"-Separation\n(coupled 2D Fourier)","V2_NoSpectral":"-Spectral\n(MLP per axis)","V3_NoMultilevel":"-Multilevel\n(random freq)","V4_NoAnalytic":"-Analytic\n(pure AD)"}
colors={"V0_Full":"#d62728","V1_NoSeparation":"#1f77b4","V2_NoSpectral":"#2ca02c","V3_NoMultilevel":"#ff7f0e","V4_NoAnalytic":"#9467bd"}
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,kind in zip(axes,["separable","nonseparable"]):
    ms=[m for m in order if m in R[kind]]
    means=[R[kind][m]["best_l2_mean"] for m in ms]; stds=[R[kind][m]["best_l2_std"] for m in ms]
    x=np.arange(len(ms))
    ax.bar(x,means,yerr=stds,capsize=4,color=[colors[m] for m in ms],alpha=0.85,edgecolor="black",lw=0.6)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in ms],fontsize=8)
    ax.set_ylabel("Best relative $L_2$ error"); ax.set_title(f"{kind} problem")
    for xi,mn in zip(x,means): ax.text(xi,mn,f"{mn:.2e}",ha="center",va="bottom",fontsize=7)
fig.suptitle("E3: Component-wise ablation (toggle one design choice; 3 seeds mean$\\pm$std)",y=1.02)
fig.savefig(os.path.join(FIG,"E3_ablation.png")); plt.close(fig)
print("E3 figure written")
