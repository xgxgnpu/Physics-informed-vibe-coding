"""Plot E7 mode-number scaling."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E7_results.json")))
fig,ax=plt.subplots(1,3,figsize=(16,5))
colors={"separable":"#d62728","nonseparable":"#1f77b4"}
for kind in R:
    Ns=sorted(int(k) for k in R[kind]); 
    err=[R[kind][str(n)]["best_l2_mean"] for n in Ns]; estd=[R[kind][str(n)]["best_l2_std"] for n in Ns]
    par=[R[kind][str(n)]["params"] for n in Ns]; tim=[R[kind][str(n)]["time_mean"] for n in Ns]
    lab="separable (rank-1)" if kind=="separable" else "non-separable (rank-4)"
    ax[0].errorbar(Ns,err,yerr=estd,marker="o",capsize=4,color=colors[kind],label=lab)
    ax[1].plot(Ns,par,marker="s",color=colors[kind],label=lab)
    ax[2].plot(Ns,tim,marker="^",color=colors[kind],label=lab)
ax[0].set_yscale("log"); ax[0].set_xlabel("number of modes $N$"); ax[0].set_ylabel("best rel. $L_2$"); ax[0].set_title("(a) Accuracy vs $N$"); ax[0].legend()
ax[1].set_xlabel("number of modes $N$"); ax[1].set_ylabel("parameters"); ax[1].set_title("(b) Parameters vs $N$ (linear)"); ax[1].legend()
ax[2].set_xlabel("number of modes $N$"); ax[2].set_ylabel("training time (s)"); ax[2].set_title("(c) Training time vs $N$"); ax[2].legend()
fig.suptitle("E7: Mode-number scaling (separable needs few modes; non-separable needs more)",y=1.02)
fig.savefig(os.path.join(FIG,"E7_mode_scaling.png")); plt.close(fig)
print("E7 figure written")
