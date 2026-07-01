# Case 8 — Taylor-Green Vortex (2D Navier-Stokes)

**PDE.** Incompressible 2D Navier-Stokes (momentum + continuity), Re = 100.
**Domain.** `[-π,π]² × [0,1]`, periodic BC. Outputs `(u, v, p)`.

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 125,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 2,688 | **3.947e-3** | 4.339e-3 | 4.16 | 41.59 |
| SV-SNN (original) | 2,688 | 4.083e-3 | 4.281e-3 | 5.26 | 52.55 |
| SPINN | 136,896 | 2.550e-1 | 6.765e-1 | 12.50 | 125.00 |
| SIREN | 99,395 | 1.536e-1 | 1.540e-1 | 6.60 | 66.03 |
| FourierPINN | 82,883 | 1.681e-1 | 1.832e-1 | 2.45 | 24.54 |
| PINN | 50,435 | 2.407e-1 | 2.429e-1 | 2.47 | 24.66 |

## Conclusions

- SV-SNN (accel) is best at the 1e-3 level (~39× better than SIREN); 1.26× faster than the original.
- All baselines are stuck at L2 > 0.15; SPINN is slowest (125 s) and unstable (final L2 = 0.68).
- 2,688 params vs 50k–137k for baselines.

## How to run

```bash
python run_accelerated.py
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
