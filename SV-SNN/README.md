# SV-SNN: Separated-Variable Spectral Neural Networks

Source code and experimental data for the paper:

> **Separated-Variable Spectral Neural Networks: A Physics-Informed Learning Approach for Solving High-Frequency Partial Differential Equations**
>
> Xiong Xiong, Zheng Zeng, Zhuo Zhang, Rongchun Hu\*, Chen Gao, Zichen Deng
>
> Northwestern Polytechnical University & National University of Defense Technology & Seoul National University

## Publication

| Item | Detail |
|------|--------|
| **Journal** | Engineering Applications of Artificial Intelligence (EAAI) |
| **Manuscript ID** | EAAI-26-5940 |
| **Status** | Under review (revised manuscript submitted, June 2026) |
| **arXiv** | [https://arxiv.org/abs/2508.00628](https://arxiv.org/abs/2508.00628) |
| **Repository** | [https://github.com/xgxgnpu/Physics-informed-vibe-coding/tree/main/SV-SNN](https://github.com/xgxgnpu/Physics-informed-vibe-coding/tree/main/SV-SNN) |

## Overview

Physics-Informed Neural Networks (PINNs) suffer from spectral bias that severely limits their ability to resolve high-frequency components of oscillatory PDEs. SV-SNN addresses this challenge through three key ideas:

1. **Separated-variable spectral representation.** Multivariate solutions are decomposed as sums of products of univariate adaptive Fourier spectral factors, reducing parameter complexity from O(K^d) to O(dK).

2. **Multilevel frequency initialization.** A structured sampling strategy distributes spectral resources across low-, characteristic-, and high-frequency bands according to the problem structure, providing robust initialization without per-problem tuning.

3. **Jacobian SVD effective rank diagnostic.** An a posteriori metric that quantifies spectral bias severity and guides architecture selection, derived from the singular value distribution of the network Jacobian.

Under matched parameter budgets (three seeds), SV-SNN achieves the lowest error on all nine benchmark cases, outperforming baselines by 1.9x--93x. Training completes in 11--34 seconds per problem on a single GPU.

## Repository Structure

```
SV-SNN/
├── svsnn_acceleration/          # Core benchmark experiments (9 cases)
│   ├── case1_heat_20pi/         # Heat equation, omega = 20pi
│   ├── case2_helmholtz_24pi/    # Helmholtz equation, omega = 24pi
│   ├── case3_nonlinear_elliptic/# Nonlinear elliptic PDE
│   ├── case4_heat_500pi/        # Heat equation, omega = 500pi
│   ├── case5_helmholtz_cylinder/# Helmholtz on cylindrical domain
│   ├── case6_helmholtz_48pi/    # Helmholtz equation, omega = 48pi
│   ├── case7_poisson_complex/   # Poisson on complex geometry
│   ├── case8_taylor_green/      # Taylor-Green vortex (Navier-Stokes)
│   ├── case9_double_cylinder_ns/# Double cylinder flow (Navier-Stokes)
│   └── case11_klein_gordon3d/   # 3D Klein-Gordon equation
│
├── rebuttal_experiments/        # Extended experiments for peer review
│   ├── E1_fair_complete_comparison/   # Fair comparison under matched budgets
│   ├── E2_hybrid_vs_ad_diff/         # Hybrid vs auto-diff training
│   ├── E3_component_ablation/        # Component-wise ablation study
│   ├── E4_freq_sampling_ablation/    # Frequency sampling strategy ablation
│   ├── E5_wchar_sensitivity/         # Characteristic frequency sensitivity
│   ├── E6_effrank_validation/        # Effective rank diagnostic validation
│   ├── E7_mode_scaling/              # Mode count scaling analysis
│   ├── E8_challenging_problems/      # Non-periodic & variable-coefficient tests
│   ├── E9_high_dim_scaling/          # 3D scalability demonstration
│   ├── E10_boundary_error/           # Boundary error analysis
│   ├── E11_grand_fair_comparison/    # Comprehensive 9-case x 6-method comparison
│   ├── E12_noisy_bc_ic/              # Noisy boundary condition robustness
│   ├── E13_boundary_layer/           # Boundary layer problems
│   ├── E14_fbpinn_compare/           # Comparison with FBPINN
│   ├── E15_structure_ablation/       # Structural ablation study
│   ├── E16_layering_ablation/        # Layering strategy ablation (135 runs)
│   ├── E17_multi_frequency/          # Multi-frequency component analysis
│   ├── E18_variable_frequency/       # Variable-frequency (chirp) signals
│   ├── E19_wchar_misestimation/      # Frequency misestimation robustness
│   ├── E20_fno_compare/              # Comparison with FNO
│   ├── E21_burgers_nonseparable/     # Non-separable Burgers equation
│   └── E22_freq_init_innovation/     # Multi-level init ablation (243 runs)
│
└── README.md
```

## Benchmark Cases

| # | Problem | Domain | Wavenumber | Dimension |
|---|---------|--------|------------|-----------|
| C1 | Heat equation | [0,1]^2 | 20pi | 2D |
| C2 | Helmholtz equation | [0,1]^2 | 24pi | 2D |
| C3 | Nonlinear elliptic PDE | [0,1]^2 | -- | 2D |
| C4 | Heat equation | [0,1]^2 | 500pi | 2D |
| C5 | Helmholtz (cylindrical) | annulus | -- | 2D |
| C6 | Helmholtz equation | [0,1]^2 | 48pi | 2D |
| C7 | Poisson (complex domain) | L-shape | -- | 2D |
| C8 | Taylor-Green vortex (NS) | [0,2pi]^2 | -- | 2D |
| C9 | Double cylinder flow (NS) | channel | -- | 2D |

## Baselines

All methods are compared under matched parameter budgets (within ±10%) and identical training configurations (Adam, lr=1e-3, 10000 epochs, 3 seeds):

- **PINN** -- standard physics-informed neural network (MLP + tanh)
- **FourierPINN** -- Fourier feature input embedding
- **SIREN** -- sinusoidal activation functions
- **SPINN** -- separable physics-informed neural network
- **FNO** -- Fourier Neural Operator (data-driven baseline)
- **FBPINN** -- finite-basis PINN

## Dependencies

- JAX + jaxlib (CUDA)
- Optax
- NumPy, SciPy
- Matplotlib

## How to Run

Each case directory contains a self-contained Python script:

```bash
cd svsnn_acceleration/case1_heat_20pi/
python run_accelerated.py

cd rebuttal_experiments/E11_grand_fair_comparison/cases/
python case1.py
```

Results (`.json`, `.npz`, `.csv`, figures) are saved to each case's `saved_data/` and `figures/` directories.

## Citation

```bibtex
@article{xiong2025separated,
  title   = {Separated-Variable Spectral Neural Networks: A Physics-Informed
             Learning Approach for Solving High-Frequency Partial Differential
             Equations},
  author  = {Xiong, Xiong and Zeng, Zheng and Zhang, Zhuo and
             Hu, Rongchun and Gao, Chen and Deng, Zichen},
  journal = {Engineering Applications of Artificial Intelligence},
  note    = {Under review, Manuscript ID: EAAI-26-5940},
  year    = {2025},
  eprint  = {2508.00628},
  archivePrefix = {arXiv}
}
```

## License

MIT
