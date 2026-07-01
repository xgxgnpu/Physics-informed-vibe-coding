# Case 6 — Helmholtz Equation (κ = 48π)

**PDE.** `-Δu - κ²u = f`, with `κ = 48π`.
**Domain.** `[0,1]²`, homogeneous Dirichlet BC.
**Exact solution.** `u(x,y) = sin(κx)·sin(κy)`.

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 10,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 3,096 | 2.092e-2 | 2.092e-2 | 1.23 | 12.34 |
| SV-SNN (original) | 3,096 | **1.650e-2** | 2.133e-2 | 1.92 | 19.16 |
| SPINN | 91,136 | 1.399 | 1.399 | 1.47 | 14.73 |
| SIREN | 82,689 | 9.981e-1 | 9.981e-1 | 2.81 | 28.06 |
| FourierPINN | 66,177 | 8.470e-1 | 8.473e-1 | 1.30 | 13.02 |
| PINN | 50,049 | 1.000 | 1.028 | 2.16 | 21.64 |

## Conclusions

- SV-SNN (original) is best (1.65e-2); the accelerated variant is 1.55× faster at 2.09e-2.
- All non-SV-SNN methods fail (L2 ≥ 0.85); SV-SNN is ~40×+ better.
- Only 3,096 params.

> Note: these are single-configuration standalone runs. The authoritative paper number for this case (C6) is the 3-seed matched-budget mean **9.1e-3 (min 2.9e-3)** reported in [E11](../../rebuttal_experiments/E11_grand_fair_comparison/README.md) (`tab:comprehensive_comparison`); the main text uses this 3-seed mean rather than any single-seed value.

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
