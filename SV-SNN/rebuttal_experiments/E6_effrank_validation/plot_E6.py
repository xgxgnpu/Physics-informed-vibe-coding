"""Plot E6 effective rank validation."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); SD=os.path.join(HERE,"saved_data"); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150,"savefig.bbox":"tight"})
R=json.load(open(os.path.join(SD,"E6_results.json")))
cases=list(R.keys())
fig,ax=plt.subplots(1,3,figsize=(16,5))
# (a) effective-rank trajectory
colors={"svsnn":"#d62728","pinn":"#1f77b4"}
for case in cases:
    for which in ["svsnn","pinn"]:
        em=R[case][which]["erank_mean"]; ck=sorted(int(k) for k in em); vals=[em[str(c)] for c in ck]
        ls="-" if which=="svsnn" else "--"
        ax[0].plot(ck,vals,ls=ls,marker="o",label=f"{which} {case}",alpha=0.8)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("effective rank $r_{eff}$(0.99)"); ax[0].set_title("(a) Effective-rank trajectory")
ax[0].legend(fontsize=7)
# (b) init singular spectra across kappa (svsnn vs pinn)
for case in cases:
    sv=np.array(R[case]["svsnn"]["seeds"][0]["init_sv"]); ax[1].semilogy(sv/sv[0],label=f"SV-SNN {case}")
for case in cases:
    sv=np.array(R[case]["pinn"]["seeds"][0]["init_sv"]); ax[1].semilogy(sv/sv[0],ls="--",label=f"PINN {case}")
ax[1].set_xlabel("singular value index"); ax[1].set_ylabel("normalized $\\sigma_i/\\sigma_0$")
ax[1].set_title("(b) Jacobian singular spectra at init"); ax[1].legend(fontsize=7); ax[1].set_xlim(0,120)
# (c) early effective rank vs final error (predictivity)
early_c=500
xs=[]; ys=[]; cs=[]; labs=[]
for case in cases:
    for which in ["svsnn","pinn"]:
        for rec in R[case][which]["seeds"]:
            tr=rec["erank_traj"]; key=str(early_c) if str(early_c) in tr else (str(early_c+1) if str(early_c+1) in tr else None)
            if key is None: continue
            xs.append(tr[key]); ys.append(rec["final_l2"]); cs.append(colors[which]); labs.append(which)
ax[2].scatter(xs,ys,c=cs,s=60,edgecolor="black",alpha=0.8)
ax[2].set_yscale("log"); ax[2].set_xlabel(f"effective rank @ epoch~{early_c}"); ax[2].set_ylabel("final rel. $L_2$")
ax[2].set_title("(c) Early $r_{eff}$ predicts final error")
import matplotlib.lines as mlines
ax[2].legend(handles=[mlines.Line2D([],[],marker="o",ls="",color="#d62728",label="SV-SNN"),
                      mlines.Line2D([],[],marker="o",ls="",color="#1f77b4",label="PINN")],fontsize=8)
# correlation
if len(xs)>2:
    r=np.corrcoef(xs,np.log10(ys))[0,1]; ax[2].text(0.05,0.05,f"corr(r_eff, log10 L2)={r:.2f}",transform=ax[2].transAxes,fontsize=9)
fig.suptitle("E6: Effective rank governs and PREDICTS high-frequency convergence",y=1.02)
fig.savefig(os.path.join(FIG,"E6_effrank.png")); plt.close(fig)
print("E6 figure written")
