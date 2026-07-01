# Case 5 — Helmholtz with Cylindrical Obstacle (κ = 24π)

**PDE.** `-Δu - κ²u = f`, with `κ = 24π`.
**Domain.** `[0,1]²` minus a cylinder (center (0.5, 0.5), r = 0.15). Outer BC `u = 0`; cylinder BC `u = sin(κx)sin(κy)`.
**Exact solution.** `u(x,y) = sin(κx)·sin(κy)`.

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 10,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

> Note: per the E5 guideline, the SV-SNN frequency center is set to `1.4κ` for this obstacle-scattering problem (parameter count unchanged).

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 2,322 | 4.469e-2 | 1.677e-1 | 0.94 | 9.35 |
| SV-SNN (original) | 2,322 | **2.751e-2** | 8.411e-2 | 1.44 | 14.39 |
| SPINN | 91,136 | 5.359e-1 | 5.359e-1 | 1.40 | 13.96 |
| SIREN | 82,689 | 8.762e-1 | 8.762e-1 | 2.29 | 22.87 |
| FourierPINN | 66,177 | 3.304e-1 | 3.307e-1 | 1.28 | 12.77 |
| PINN | 50,049 | 1.000 | 1.079 | 1.41 | 14.12 |

## Conclusions

- SV-SNN (original) has the best best-L2 (2.75e-2); the accelerated variant is 1.54× faster, though its final L2 drifts to 1.68e-1.
- SV-SNN is still ~7.4× better than FourierPINN on best L2; all baselines stay above 0.33.
- 2,322 params vs 50k–91k for baselines.

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
