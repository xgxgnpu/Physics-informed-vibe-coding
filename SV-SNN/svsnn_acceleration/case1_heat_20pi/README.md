# Case 1 — Heat Equation (κ = 20π)

**PDE.** `∂u/∂t - α·∂²u/∂x² = 0`, with `α = 1/(20π)²`.
**Domain.** `x ∈ [-1,1]`, `t ∈ [0,1]`. IC `u(x,0) = sin(20πx)`, BC `u(±1,t) = 0`.
**Exact solution.** `u(x,t) = exp(-t)·sin(20πx)`.

## Unified configuration

10,000 epochs, Adam (lr = 1e-3), 10,000 collocation points, evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SV-SNN (accel) | 3,730 | 4.957e-4 | 1.155e-3 | 4.04 | 40.41 |
| SV-SNN (original) | 3,730 | **3.450e-4** | 9.066e-4 | 7.14 | 71.45 |
| SPINN | 82,816 | 8.909e-1 | 8.909e-1 | 4.04 | 40.44 |
| SIREN | 82,689 | 1.529e-3 | 4.120e-3 | 3.59 | 35.90 |
| FourierPINN | 66,177 | 1.852e-3 | 1.905e-2 | 1.96 | 19.63 |
| PINN | 50,049 | 4.689e-1 | 4.689e-1 | 1.75 | 17.54 |

## Conclusions

- SV-SNN (original) is most accurate (3.45e-4); the accelerated variant is 1.77× faster (40.4 s vs 71.5 s) at a slightly higher best L2.
- SV-SNN beats every baseline on accuracy with only 3,730 params (~14–22× fewer).
- SPINN and PINN fail (L2 ≈ 0.47–0.89); SIREN/FourierPINN reach ~1e-3 but lag SV-SNN.

## How to run

```bash
python run_accelerated.py
python plot_results.py        # optional
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`, figures `fig1`–`fig4`. See `实验报告_SVSNN加速.md` and `complexity_analysis.md` for the detailed analysis.
