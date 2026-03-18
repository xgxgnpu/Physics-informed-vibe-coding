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

## References

1. Ko, S., & Park, S. (2025). VS-PINN: A fast and efficient training of physics-informed neural networks using variable-scaling methods for solving PDEs with stiff behavior. *Journal of Computational Physics*, 529, 113860.
2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
3. Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*, 449, 110768.
4. Schäfer, M., & Turek, S. (1996). Benchmark computations of laminar flow around a cylinder. In E. H. Hirschel (Ed.), *Flow Simulation with High-Performance Computers II* (Notes on Numerical Fluid Mechanics, Vol. 52, pp. 547–566). Vieweg+Teubner Verlag.
5. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*.
6. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. In *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 249–256.
7. Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1), 208–228.
8. Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs. http://github.com/jax-ml/jax
9. Xiong, X., Lu, K., Zhang, Z., Zeng, Z., Zhou, S., Hu, R., & Deng, Z. (2025). High-frequency flow field super-resolution via physics-informed hierarchical adaptive Fourier feature networks. *Physics of Fluids*, 37(9).
10. Xiong, X., Lu, K., Zhang, Z., Zeng, Z., Zhou, S., Deng, Z., & Hu, R. (2025). J-PIKAN: A physics-informed KAN network based on Jacobi orthogonal polynomials for solving fluid dynamics. *Communications in Nonlinear Science and Numerical Simulation*, 109414.
