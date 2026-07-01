# Case 2 — Helmholtz Equation (κ = 24π)

**PDE.** `-Δu - κ²u = f`, with `κ = 24π`.
**Domain.** `[0,1]²`, homogeneous Dirichlet BC.
**Exact solution.** `u(x,y) = sin(κx)·sin(κy)`; source `f = κ²·sin(κx)·sin(κy)`.

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 10,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 1,170 | **4.543e-3** | 4.543e-3 | 1.05 | 10.51 |
| SV-SNN (original) | 1,170 | 9.357e-3 | 1.592e-2 | 1.55 | 15.48 |
| SPINN | 91,136 | 5.171e-1 | 5.171e-1 | 1.37 | 13.69 |
| SIREN | 82,689 | 8.993e-1 | 8.993e-1 | 2.31 | 23.05 |
| FourierPINN | 66,177 | 3.491e-1 | 3.516e-1 | 1.30 | 12.98 |
| PINN | 50,049 | 9.982e-1 | 1.023 | 1.43 | 14.33 |

## Conclusions

- SV-SNN (accel) is best overall: L2 = 4.54e-3, ~77× better than the next-best baseline (FourierPINN).
- 1.47× training speedup vs the original SV-SNN, and more accurate here too.
- 1,170 params vs 50k–91k for baselines (~43–78× compression).

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
