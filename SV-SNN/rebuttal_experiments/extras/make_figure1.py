"""
Redraw of Figure 1 (architecture schematic) in a concise, professional academic style.
Addresses R9.7. Pure matplotlib (no GPU)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.join(HERE,"figures"); os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.size":11,"figure.dpi":150,"savefig.bbox":"tight"})

fig,ax=plt.subplots(figsize=(12,6.2)); ax.set_xlim(0,12); ax.set_ylim(0,6.2); ax.axis("off")
def box(x,y,w,h,text,fc,ec="#333333",fs=10,tc="black"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=fc,ec=ec,lw=1.3)); ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=tc)
def arrow(x1,y1,x2,y2,color="#444444"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,lw=1.4,color=color))

C_IN="#dbe9f6"; C_SP="#fde2c4"; C_FREQ="#e7f5d8"; C_COMB="#f6d6e0"; C_OUT="#dbe9f6"; C_LOSS="#eee6f7"
# inputs
box(0.2,3.9,1.3,0.8,"$x$",C_IN); box(0.2,1.6,1.3,0.8,"$t$ (or $y$)",C_IN)
# per-axis spectral feature blocks
box(2.2,3.5,2.7,1.6,"Per-axis spectral features\n$\\phi_x=\\{\\cos(\\omega_k x),\\sin(\\omega_k x)\\}$\nadaptive coeff. $a_k,b_k$",C_SP,fs=9)
box(2.2,1.0,2.7,1.6,"Per-axis spectral features\n$\\phi_t=\\{\\cos(\\omega_k t),\\sin(\\omega_k t)\\}$",C_SP,fs=9)
# multi-level frequency init
box(2.2,5.35,2.7,0.7,"Multi-level frequency sampling  (low / characteristic $w_{char}$ / high)",C_FREQ,fs=9)
arrow(3.55,5.35,3.55,5.1); arrow(3.55,5.35,3.55,2.6)
# combination
box(5.5,2.2,2.6,2.0,"Separated-variable\ncombination\n$u=\\sum_{n=1}^{N} c_n\\,X_n(x)\\,T_n(t)$",C_COMB,fs=10)
# output
box(8.7,2.7,1.3,1.0,"$u(x,t)$",C_OUT)
# losses / hybrid diff
box(8.5,4.5,3.2,1.3,"Hybrid differentiation:\nanalytic spatial $\\partial_{xx}$ (closed form)\n+ automatic temporal $\\partial_t$",C_LOSS,fs=9)
box(8.5,0.4,3.2,1.5,"Physics-informed loss\n$\\mathcal{L}=\\|\\mathcal{N}[u]-f\\|^2+\\|u-g\\|_{\\partial\\Omega}^2$\n(PDE residual + BC/IC)",C_LOSS,fs=9)
# arrows
arrow(1.5,4.3,2.2,4.3); arrow(1.5,2.0,2.2,1.8)
arrow(4.9,4.3,5.5,3.6); arrow(4.9,1.8,5.5,2.8)
arrow(8.1,3.2,8.7,3.2)
arrow(10.0,3.2,10.1,4.5); arrow(10.1,4.5,9.6,4.5)
arrow(10.0,2.7,10.1,1.9)
arrow(10.0,0.9,8.1,2.2,color="#b03a2e")  # loss feedback (gradient)
ax.text(7.0,0.65,"backprop / Adam",color="#b03a2e",fontsize=8,style="italic")
ax.set_title("Separated-Variable Spectral Neural Network (SV-SNN): architecture overview",fontsize=12)
fig.savefig(os.path.join(FIG,"Figure1_architecture.png")); plt.close(fig)
print("Figure 1 written to",FIG)
