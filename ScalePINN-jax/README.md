# Scale-PINN — JAX Implementation

JAX implementation of **Scale-PINN: evolutionary regularization for physics-informed neural networks**, applied to the lid-driven cavity problem at $Re = 7500$.

## Algorithm Overview

Standard PINNs fail to converge on high-Reynolds-number Navier-Stokes equations due to the extreme scale mismatch between convective and diffusive terms. Scale-PINN introduces **evolutionary regularization (ER)** — a pseudo-time-stepping correction that augments the PDE residual with the difference between the current and previous solutions:

$$\tilde{R} = R(\theta) + \frac{u_\theta - u_{\theta_0}}{\text{ER}}$$

This transforms a single intractable optimization into a sequence of well-conditioned sub-problems, each constrained to remain close to the previous iterate.

## Cases

### Case 1: Lid-Driven Cavity (Re = 7500)

- **PDE**: 2D steady incompressible Navier-Stokes
- **Domain**: $[0, 1]^2$, top-lid velocity $u = 1$
- **Network**: shared trunk (Fourier features + 2×Dense+SiLU) + 3 branches (u, v, p), 59,520 parameters
- **Optimizer**: Adam + CosineDecay (lr=5e-4, exponent=1.2), 50,000 iterations
- **Fair comparison**: M1 (Standard PINN, ER=0) vs M2 (Scale-PINN, ER=0.095, ER_xx=0.5)

## Results Summary

| Model | Parameters | Best RL2 | Final RL2 | Final MSE | Time (s) |
|-------|:---:|:---:|:---:|:---:|:---:|
| Standard PINN (M1) | 59,520 | 8.460e-01 | 9.458e-01 | 4.350e-02 | 100.4 |
| **Scale-PINN (M2)** | 59,520 | **2.751e-02** | **2.979e-02** | **4.316e-05** | 110.0 |

Scale-PINN achieves **96.7% improvement** in relative L2 error with only 10% additional training time.

## Features

- Shared trunk + three-branch architecture with Fourier feature embedding
- Evolutionary regularization with separate ER and ER_xx coefficients
- PINN module (10-channel output with AD-computed residuals) + DNN module (5-channel for inference)
- Parameter sharing via `jax.flatten_util.ravel_pytree`
- Journal-quality figures (300 dpi, Times New Roman)

## File Structure

```
case1_ldc_re7500/
├── ldc_re7500_scalepinn.py          # Main code (self-contained)
├── data/
│   ├── LDC_RE7500_*.csv             # Reference CFD data (not tracked)
│   ├── comparison_summary.txt       # M1 vs M2 quantitative comparison
│   ├── loss_history_M1.txt          # M1 training log
│   ├── loss_history_M2.txt          # M2 training log
│   ├── predictions_M1.npz           # M1 predictions on full grid
│   └── predictions_M2.npz           # M2 predictions on full grid
├── figures/
│   ├── fig_loss_comparison.png      # PDE/BC/Total loss curves
│   ├── fig_l2_error_comparison.png  # RL2 convergence curves
│   ├── fig_velocity_M1.png          # M1 velocity magnitude
│   ├── fig_velocity_M2.png          # M2 velocity magnitude
│   ├── fig_velocity_M1_vs_M2.png    # Side-by-side velocity comparison
│   ├── fig_uvp_comparison_M1.png    # M1 u/v/p contours
│   ├── fig_uvp_comparison_M2.png    # M2 u/v/p contours
│   └── fig_centerline_comparison.png # Centerline velocity profiles
└── checkpoints/
    ├── params_M1.pkl                # M1 best parameters
    └── params_M2.pkl                # M2 best parameters
```

## Usage

```bash
# Full training (both M1 and M2, ~210s on RTX 4090)
python ldc_re7500_scalepinn.py

# Train only Scale-PINN
python ldc_re7500_scalepinn.py --mode M2

# Quick test (500 iterations)
python ldc_re7500_scalepinn.py --quick

# Regenerate plots from saved data
python ldc_re7500_scalepinn.py --plot_only
```

## References

1. Chiu, P.-H. et al. (2026). Scale-PINN: Learning Efficient Physics-Informed Neural Networks Through Sequential Correction. *arXiv:2602.19475*.
2. Raissi, M. et al. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
3. Cao, W. & Zhang, W. (2025). An analysis and solution of ill-conditioning in physics-informed neural networks. *Journal of Computational Physics*, 520, 113494.
4. Ghia, U. et al. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.
