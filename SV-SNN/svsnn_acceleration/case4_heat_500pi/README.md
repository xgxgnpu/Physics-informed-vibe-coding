# Case 4 — Heat Equation (κ = 500π, ultra-high frequency)

**PDE.** `∂u/∂t - α·∂²u/∂x² = 0`, with `α = 1/(500π)²`.
**Domain.** `x ∈ [-1,1]`, `t ∈ [0,1]`. IC `u(x,0) = sin(500πx)`, BC `u(±1,t) = 0`.
**Exact solution.** `u(x,t) = exp(-t)·sin(500πx)`.

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 10,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 1,612 | 3.917e-3 | 3.917e-3 | 2.51 | 25.05 |
| SV-SNN (original) | 1,612 | **2.826e-3** | 2.826e-3 | 3.63 | 36.34 |
| SPINN | 82,816 | 1.599 | 1.599 | 4.17 | 41.65 |
| SIREN | 82,689 | 2.414e-1 | 2.419e-1 | 3.55 | 35.45 |
| FourierPINN | 66,177 | 3.184e-1 | 3.522e-1 | 1.86 | 18.63 |
| PINN | 50,049 | 1.011 | 1.292 | 1.74 | 17.42 |

## Conclusions

- The original SV-SNN is slightly more accurate (2.83e-3); the accelerated variant is 1.45× faster and still at the 1e-3 level.
- Traditional methods collapse at this frequency: SPINN/PINN L2 > 1; SIREN/FourierPINN ~0.24–0.32.
- SV-SNN maintains usable accuracy at κ = 500π with only ~1,612 params.

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
