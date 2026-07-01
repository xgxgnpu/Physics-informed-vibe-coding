# Case 9 — Double-Cylinder Flow (Steady Navier-Stokes)

**PDE.** Steady 2D incompressible Navier-Stokes, ρ = 1, μ = 1.
**Domain.** `[-π,π]²` minus two cylinders (centers (-1.0, 0.5) and (1.0, -0.5), r = 0.3 each).

## Unified configuration

15,000 epochs, Adam (lr = 1e-3), 5,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 407 | 3.340e-4 | 4.304e-4 | 1.36 | 20.40 |
| SV-SNN (original) | 407 | **2.246e-4** | 7.034e-4 | 2.29 | 34.35 |
| SPINN | 50,307 | 1.001 | 1.251 | 4.90 | 73.49 |
| SIREN | 82,947 | 7.572e-3 | 9.045e-3 | 8.14 | 122.05 |
| FourierPINN | 66,435 | 7.455e-3 | 7.633e-3 | 6.97 | 104.61 |
| PINN | 50,307 | 9.977e-3 | 1.884e-2 | 5.70 | 85.56 |

## Conclusions

- SV-SNN (original) is most accurate (2.25e-4); the accelerated variant is 1.68× faster at 3.34e-4 — still ~22× better than FourierPINN.
- Extreme parameter efficiency: only **407 params** (~123–204× fewer than baselines).
- Total time 20.4 s vs 73–122 s for the others (3.6–6.0× wall-clock advantage).

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
