# Case 7 — Poisson on a Perforated Domain (μ = 7π)

**PDE.** `-Δu = f = 2μ²·sin(μx)sin(μy)`, with `μ = 7π`.
**Domain.** `[-1,1]²` with 3 circular holes and 1 elliptical hole. BC `u = u_exact` on all boundaries.
**Exact solution.** `u(x,y) = sin(μx)·sin(μy)`.

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 10,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 1,944 | **2.731e-2** | 2.731e-2 | 1.28 | 12.79 |
| SV-SNN (original) | 1,944 | 3.844e-2 | 3.844e-2 | 1.57 | 15.68 |
| SPINN | 91,136 | 7.215e-1 | 7.215e-1 | 1.36 | 13.56 |
| SIREN | 82,689 | 1.742e-1 | 1.742e-1 | 4.58 | 45.75 |
| FourierPINN | 66,177 | 1.238e-1 | 1.253e-1 | 2.94 | 29.40 |
| PINN | 50,049 | 4.064 | 4.064 | 3.13 | 31.31 |

## Conclusions

- SV-SNN (accel) is best (2.73e-2), beating even the original (3.84e-2) — acceleration helps accuracy here.
- ~4.5× better than FourierPINN; PINN fails catastrophically (L2 = 4.06).
- 1,944 params; 1.23× speedup vs the original.

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
