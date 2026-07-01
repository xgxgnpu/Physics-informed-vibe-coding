# Case 11 — Klein-Gordon Equation (high-frequency, second-order in time)

**PDE (as executed in `run_accelerated.py`).** Linear Klein-Gordon `u_tt - u_xx - u_yy + u = f`, with `κ = 4π`.
**Domain.** `x, y ∈ [-1,1]`, `t ∈ [0,1]`. IC `u(x,y,0) = sin(κx)sin(κy)`, `u_t(x,y,0) = 0`; spatial BC `u = 0`.
**Exact solution.** `u(x,y,t) = sin(κx)sin(κy)·cosh(t)`.

> Note: the master report lists Case 11 with a different (nonlinear, κ = ω = 10π) specification marked "to be run". The numbers below come from the actually executed runs (linear setup, κ = 4π, **50,000 epochs** — the higher epoch count reflects the second-order time derivative `u_tt`).

## Unified configuration

50,000 epochs, Adam (lr = 1e-3), evaluation every 100 epochs. Methods: SV-SNN (accelerated), SV-SNN (original), SPINN, SIREN, FourierPINN, vanilla PINN.

## Results

| Method | Params | Best L2 | Final L2 | ms/epoch | Time (s) |
|--------|-------:|--------:|---------:|---------:|---------:|
| SPINN | 130,560 | **2.734e-3** | 2.914e-3 | 3.38 | 168.92 |
| SV-SNN (accel) | 11,440 | 6.716e-3 | 6.716e-3 | 3.79 | 189.47 |
| SV-SNN (original) | 11,440 | 6.909e-3 | 6.909e-3 | 10.84 | 541.80 |
| FourierPINN | 82,625 | 2.473e-2 | 2.473e-2 | 3.20 | 159.97 |
| SIREN | 99,137 | 3.295e-2 | 3.295e-2 | 5.81 | 290.45 |
| PINN | 50,177 | 1.473e-1 | 1.473e-1 | 3.18 | 158.84 |

## Conclusions

- **SPINN attains the best accuracy** here (2.73e-3) — the only case where a baseline beats SV-SNN on best L2, though with ~11× more parameters.
- SV-SNN accelerated is **2.86× faster** than the original (189.5 s vs 541.8 s) at comparable accuracy (~6.7e-3).
- The second-order time derivative `u_tt` is supported; SV-SNN uses 11,440 params vs 50k–131k for baselines.

## How to run

```bash
python run_accelerated.py
python plot_results.py        # optional
```

Outputs: `saved_data/*_summary.json`, `comparison_table.csv`.
