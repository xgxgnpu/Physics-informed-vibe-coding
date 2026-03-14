# NTK-PINN — JAX Implementation

JAX re-implementation of **"When and why PINNs fail to train: A neural tangent kernel perspective"** (Wang, Yu & Perdikaris, JCP 2022).

## Algorithm Overview

Standard PINNs suffer from imbalanced convergence rates across different loss components (PDE residual, boundary conditions, initial conditions). NTK-PINN leverages the **Neural Tangent Kernel** to compute adaptive weights that balance the training dynamics:

$$\lambda_i = \frac{\mathrm{Tr}(\mathbf{K})}{\mathrm{Tr}(\mathbf{K}_i)}$$

where $\mathbf{K}$ is the full NTK matrix and $\mathbf{K}_i$ is the sub-kernel corresponding to the $i$-th loss component.

## Cases

### Case 1: Poisson 1D

- **PDE**: $u_{xx} = f(x)$, exact solution $u(x) = \sin(4\pi x)$
- **Network**: `[1, 512, 1]`, tanh, NTK initialization
- **Best L2 relative error**: `2.157e-05` (with normalization)

### Case 2: Wave 1D

- **PDE**: $u_{tt} = c^2 u_{xx}$, $c = 2$
- **Exact solution**: $u(x,t) = \sin(\pi x)\cos(2\pi t) + 0.5\sin(4\pi x)\cos(8\pi t)$
- **Network**: `[2, 500, 500, 500, 1]`, tanh, Xavier initialization
- **Best L2 relative error**: `4.806e-03` (with normalization)

## Results Summary

| Case | Parameters | Best L2 Error | Training Time | Optimizer |
|------|-----------|---------------|---------------|-----------|
| Poisson 1D (normalized) | 1,025 | 2.157e-05 | 45.1s | Adam |
| Wave 1D (normalized) | 502,001 | 4.806e-03 | 687.5s | Adam |

## Features

- Z-score input normalization with gradient chain-rule correction
- NTK computation via `jax.jacrev`
- Adaptive loss weighting updated every 1000 steps
- Journal-quality figures (300 dpi, Times New Roman)
- Complete data logging (loss history, NTK eigenvalues, predictions)

## File Structure

```
case1_poisson1d/
├── poisson1d_ntk_pinn.py    # Self-contained script
├── data/                     # Loss history, NTK data, predictions
├── figures/                  # Publication-quality plots
└── checkpoints/              # Saved model parameters

case2_wave1d/
├── wave1d_ntk_pinn.py
├── data/
├── figures/
└── checkpoints/
```

## Reference

Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*, 449, 110768.
