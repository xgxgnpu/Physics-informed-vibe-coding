# GradientWeighted PINN (JAX)

JAX reproduction of **"Understanding and mitigating gradient flow pathologies in
physics-informed neural networks"** (Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021).

## Cases

| Case | Problem | Reference |
|------|---------|-----------|
| case1_lid_driven_cavity | 2D Steady Lid-Driven Cavity (Re=100) | Ghia et al. benchmark |
| case2_klein_gordon | 1D Klein-Gordon (nonlinear) | Wang et al. 2021 |

## Method

- **M1**: Standard PINN (fixed equal weighting for all loss terms)
- **M2**: Gradient-weighted adaptive λ for IC/BC losses (learning rate annealing)
- Exponential moving average (β=0.9) for smooth λ updates
- Input normalization (z-score) with chain-rule derivative correction
- Adam optimizer

## Run

```bash
cd case1_lid_driven_cavity
python lid_driven_cavity_gw_pinn.py
```

```bash
cd case2_klein_gordon
python klein_gordon_gw_pinn.py
```

Results are saved to `data/`, `figures/`, and `checkpoints/` subdirectories.
