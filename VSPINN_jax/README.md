# VS-PINN — JAX Implementation

JAX implementation of **"VS-PINN: A fast and efficient training of physics-informed neural networks using variable-scaling methods for solving PDEs with stiff behavior"** (Ko & Park, JCP 2025).

## Algorithm Overview

Standard PINNs struggle with **stiff PDEs** where different physical terms (e.g., convection vs. diffusion) span several orders of magnitude. VS-PINN addresses this through a simple yet effective **coordinate scaling** strategy:

$$\tilde{x} = Nx, \quad \tilde{y} = Ny$$

The chain rule introduces amplification factors $N^k$ for $k$-th order derivatives, effectively rebalancing the relative magnitudes of PDE residual terms without modifying the network architecture or adding computational overhead.

## Cases

### Case 1: 2D Navier-Stokes (Cylinder Flow)

- **PDE**: 2D steady incompressible Navier-Stokes ($\rho=1$, $\mu=0.02$)
- **Domain**: Channel $[0, 1.1] \times [0, 0.41]$ with cylinder at $(0.2, 0.2)$, $r=0.05$
- **Network**: `[2, 40, 40, 40, 40, 40, 3]`, tanh, Xavier initialization
- **Scaling factor**: $N = 10$
- **Final L2 relative errors**: $L^2(u) = 2.10\%$, $L^2(v) = 5.06\%$, $L^2(p) = 4.45\%$

## Results Summary

| Case | Parameters | Scaling N | L2(u) | L2(v) | L2(p) | Epochs | Optimizer |
|------|-----------|:---------:|:-----:|:-----:|:-----:|:------:|-----------|
| NS 2D Cylinder | 6,803 | 10 | 2.10% | 5.06% | 4.45% | 80,000 | Adam |

## Features

- Variable-scaling coordinate transformation with factor $N$
- Chunked PDE residual computation via `jax.lax.scan` for memory efficiency
- Collocation points resampled every epoch with cylinder-vicinity refinement
- Fluent CFD reference solution for validation
- Journal-quality figures (300 dpi, Times New Roman)

## File Structure

```
case1_ns2d/
├── ns2d_vspinn_pinn.py    # Self-contained script (train + plot)
├── data/
│   ├── FluentSol.mat      # Fluent reference solution
│   ├── history.txt         # Training history (802 records)
│   └── predictions.npz     # Model predictions
├── figures/                 # Result plots (loss, L2 error, fields)
└── checkpoints/
    └── params.pkl           # Saved model parameters
```

## Reference

Ko, S., & Park, S. (2025). VS-PINN: A fast and efficient training of physics-informed neural networks using variable-scaling methods for solving PDEs with stiff behavior. *Journal of Computational Physics*, 529, 113860.
