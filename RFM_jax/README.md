# RFM — Random Feature Method for PDEs (JAX)

> JAX-GPU implementation of the **Random Feature Method (RFM)** with systematic parameter studies.

## Algorithm

RFM is a **meshless, non-iterative** solver for PDEs.  Unlike PINNs which
require gradient-based training over thousands of iterations, RFM constructs a
set of random basis functions (tanh features) combined with a Partition of Unity
(PoU) and solves the resulting linear system in a **single least-squares step**.

Key components:
- **Random features**: `tanh((x-c)/r @ W + b)` with randomly sampled `W, b`
- **Partition of Unity**: smooth bump functions for domain decomposition
- **Least-squares solve**: collocation on interior + boundary points

## Cases

| # | Case | PDE | Description |
|---|------|-----|-------------|
| 1 | `case1_stokes_2d/` | 2D Stokes | Stokes flow on holed square with parameter sweeps |

## Quick Start

```bash
# Full parameter study (Q, n_hidden, n_sub, seed sweeps)
cd case1_stokes_2d/
python stokes_2d_rfm.py

# Quick test
python stokes_2d_rfm.py --quick

# Regenerate plots from saved data
python stokes_2d_rfm.py --plot_only

# Selected sweeps only
python stokes_2d_rfm.py --sweeps sweep_Q sweep_nhidden
```

## Reference

```bibtex
@article{chen2024random,
  title={Random Feature Method for Solving Partial Differential Equations},
  author={Chen, Yifan and Bhatt, Anudeep},
  journal={arXiv preprint arXiv:2406.xxxxx},
  year={2024}
}
```

## Dependencies

`jax`, `jaxlib` (CUDA), `numpy`, `matplotlib`
