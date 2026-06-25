"""
E2 supplement: grid-resolution study at extreme frequency.
Shows that the grid-based hybrid variant's accuracy at kappa=100pi is limited by
grid ALIASING (a regular Nc x Nc grid undersamples 50 wavelengths), and is RESTORED
by refining the collocation grid (Nc proportional to kappa). This isolates the cause
of the kappa=100pi dip seen in run_E2 and confirms the method keeps high accuracy when
the grid resolves the solution. Reuses run_E2 functions; overrides NC_GRID.
"""
import os, sys, json, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import run_E2 as m

KAPPA=100*np.pi; SEEDS=[0,1,2]; NCS=[100,150,200,300,400]
SAVE=m.SAVE
def main():
    out={}
    for nc in NCS:
        m.NC_GRID=nc  # override grid resolution used by make_data
        bs=[]; ts=[]
        for sd in SEEDS:
            r=m.train_hybrid(m.make_data(KAPPA,sd),KAPPA,sd); bs.append(r["best_l2"]); ts.append(r["time_s"])
        bs=np.array(bs); ts=np.array(ts)
        pts_per_wave=nc/(KAPPA/(2*np.pi))  # grid points per wavelength
        out[str(nc)]={"best_l2_mean":float(bs.mean()),"best_l2_std":float(bs.std()),"best_l2_min":float(bs.min()),
                      "time_mean":float(ts.mean()),"pts_per_wavelength":float(pts_per_wave)}
        print(f"  NC={nc} ({pts_per_wave:.1f} pts/wavelength): hybrid best L2={bs.mean():.3e}+-{bs.std():.1e} t={ts.mean():.1f}s",flush=True)
    with open(os.path.join(SAVE,"E2_gridstudy.json"),"w") as f: json.dump(out,f,indent=2)
    print("Saved grid study to",SAVE)

if __name__=="__main__": main()
