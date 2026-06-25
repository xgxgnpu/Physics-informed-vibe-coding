"""Plot E4 frequency sampling strategy. English journal style."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E4_results.json")))
order=["S1_single","S2_two","S3_three25","S4_continuous","S5_three40"]
labels={"S1_single":"single","S2_two":"two-level","S3_three25":"three-level\n25/50/25","S4_continuous":"continuous","S5_three40":"three-level\n40/40/20"}
colors={"S1_single":"#7f7f7f","S2_two":"#1f77b4","S3_three25":"#d62728","S4_continuous":"#2ca02c","S5_three40":"#ff7f0e"}
fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,kind in zip(axes,["pure_high","multi_scale"]):
    ms=[m for m in order if m in R[kind]]
    means=[R[kind][m]["best_l2_mean"] for m in ms]; stds=[R[kind][m]["best_l2_std"] for m in ms]
    x=np.arange(len(ms))
    bars=ax.bar(x,means,yerr=stds,capsize=4,color=[colors[m] for m in ms],alpha=0.85,edgecolor="black",lw=0.6)
    best_i=int(np.argmin(means)); bars[best_i].set_edgecolor("black"); bars[best_i].set_linewidth(2.2)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in ms],fontsize=9)
    ax.set_ylabel("Best relative $L_2$ error")
    ax.set_title(f"{'Pure high-frequency' if kind=='pure_high' else 'Multi-scale (low + high)'}")
    for xi,mn in zip(x,means): ax.text(xi,mn,f"{mn:.2e}",ha="center",va="bottom",fontsize=7)
fig.suptitle("E4: Frequency sampling strategy (same SV-SNN; only freq init differs; 3 seeds)",y=1.02)
fig.savefig(os.path.join(FIG,"E4_strategies.png")); plt.close(fig)

# ---- E4b: fair multi-scale COVERAGE test (normalized residual) ----
cov_path=os.path.join(SD,"E4b_coverage.json")
if os.path.exists(cov_path):
    C=json.load(open(cov_path))
    corder=["C1_low_only","C2_high_only","C3_multilevel"]
    clab={"C1_low_only":"low-band only\n(misses high)","C2_high_only":"high-band only\n(misses low)","C3_multilevel":"multi-level\n(covers both)"}
    ccol={"C1_low_only":"#1f77b4","C2_high_only":"#ff7f0e","C3_multilevel":"#d62728"}
    fig2,ax=plt.subplots(figsize=(7.2,5))
    means=[C[m]["best_l2_mean"] for m in corder]; stds=[C[m]["best_l2_std"] for m in corder]
    x=np.arange(len(corder))
    bars=ax.bar(x,means,yerr=stds,capsize=4,color=[ccol[m] for m in corder],alpha=0.88,edgecolor="black",lw=0.7)
    bars[2].set_linewidth(2.4)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([clab[m] for m in corder],fontsize=9)
    ax.set_ylabel("Best relative $L_2$ error")
    ax.set_title("Multi-scale coverage test (normalized residual)\n$u=\\sin(k_{lo}x)\\sin(k_{lo}y)+\\sin(k_{hi}x)\\sin(k_{hi}y)$")
    for xi,mn in zip(x,means): ax.text(xi,mn,f"{mn:.2e}",ha="center",va="bottom",fontsize=8)
    fig2.savefig(os.path.join(FIG,"E4b_coverage.png"),bbox_inches="tight"); plt.close(fig2)
    print("E4b figure written")
print("E4 figure written")
