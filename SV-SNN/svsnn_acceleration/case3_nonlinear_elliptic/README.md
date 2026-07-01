# Case 3 — Nonlinear Elliptic PDE

**PDE.** `Δu + u² = f(x,y)`.
**Domain.** `[0,1]²`, non-homogeneous Dirichlet BC.
**Exact solution.** `u(x,y) = (x+y)·cos(10x)·sin(10y)` (characteristic frequency ω = 10).

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 10,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 1,171 | **1.946e-3** | 2.064e-3 | 2.19 | 21.89 |
| SV-SNN (original) | 1,171 | 2.178e-3 | 4.711e-3 | 2.66 | 26.61 |
| SPINN | 49,920 | 1.106e-2 | 1.853e-2 | 3.35 | 33.47 |
| SIREN | 82,689 | 1.324e-1 | 5.553e-1 | 4.88 | 48.75 |
| FourierPINN | 66,177 | 1.444e-1 | 1.444e-1 | 2.60 | 25.95 |
| PINN | 50,049 | 2.187e-1 | 2.261e-1 | 2.25 | 22.53 |

## Conclusions

- SV-SNN (accel) is most accurate (1.95e-3), ~5.7× better than the next-best baseline (SPINN).
- 1.22× speedup vs the original, and better on both best and final L2.
- Only 1,171 params vs 50k–83k for baselines.

> Note: these are single-configuration standalone runs. The authoritative paper number for this case (C3) is the 3-seed matched-budget mean **2.0e-3** reported in [E11](../../rebuttal_experiments/E11_grand_fair_comparison/README.md) (`tab:comprehensive_comparison`).

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`.
