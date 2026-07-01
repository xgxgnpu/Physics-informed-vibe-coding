# SV-SNN: Separated-Variable Spectral Neural Networks

Source code and experimental data for the paper:

> **Separated-Variable Spectral Neural Networks with Characteristic-Frequency Multi-Level Initialization for High-Frequency Partial Differential Equations**
>
> Xiong Xiong, Zheng Zeng, Zhuo Zhang, Rongchun Hu\*, Chen Gao, Zichen Deng
>
> Northwestern Polytechnical University & National University of Defense Technology & Seoul National University

## Publication

| Item | Detail |
|------|--------|
| **Journal** | Engineering Applications of Artificial Intelligence (EAAI) |
| **Status** | Revised manuscript under review (2026) |
| **arXiv** | [https://arxiv.org/abs/2508.00628](https://arxiv.org/abs/2508.00628) |
| **Repository** | [https://github.com/xgxgnpu/Physics-informed-vibe-coding/tree/main/SV-SNN](https://github.com/xgxgnpu/Physics-informed-vibe-coding/tree/main/SV-SNN) |

## Overview

Physics-Informed Neural Networks (PINNs) suffer from spectral bias that severely limits their ability to resolve high-frequency components of oscillatory PDEs, and conventional Fourier-feature remedies scale exponentially with dimension. Rather than a wholly new class of network, SV-SNN is best understood as a **principled combination of established ingredients tailored to high-frequency PDEs**, built around three core innovations:

1. **Separated-variable spectral representation.** Multivariate solutions are decomposed as sums of products of univariate adaptive Fourier spectral factors, reducing parameter complexity from exponential to linear in dimension (O(K^d) to O(dK)). The spectral factors admit closed-form spatial derivatives, evaluated through a **hybrid differentiation** strategy (analytic in space, automatic differentiation in time) that yields a ~2.5x training speedup; this is an efficiency benefit, not a precision benefit (automatic differentiation is equally accurate for smooth bases).
2. **Characteristic-frequency multi-level initialization.** A structured sampling strategy distributes spectral resources across low-, characteristic-, and high-frequency bands centered on a problem-specific characteristic frequency `w_char`. When `w_char` is not known a priori it is **estimated automatically** from the source term via FFT, providing robust initialization without per-problem tuning.
3. **Jacobian SVD effective rank diagnostic.** A metric derived from the singular value distribution of the residual Jacobian that quantifies spectral-bias severity. Beyond a posteriori diagnosis, on the tested benchmarks it separates success from failure early in training (predictive), guiding architecture selection.

As a special case, when the number of modes reduces to `N = 1` with randomly sampled, frozen frequencies (only the linear output coefficients trained), SV-SNN degenerates to a single-hidden-layer random-feature / extreme learning machine (ELM) representation; the general `N > 1` model with multi-level (optionally trainable) frequencies is a spectrally structured, multi-modal generalization of such random-feature/ELM solvers.

Under matched parameter budgets (three seeds), SV-SNN achieves the lowest error on all nine benchmark cases, outperforming baselines by 1.9x–94x. Training completes in 11–34 seconds per problem on a single NVIDIA RTX 4090 GPU.

---

## How to read this guide (for reviewers)

This README is a single entry point to **all 32 experiments**. Each experiment folder has its own English `README.md` with full setup, a transcribed data table, figure captions, and a run command. The tables below consolidate the headline numbers; click any experiment ID to open its self-contained document.

- **Benchmark cases** (`svsnn_acceleration/caseN_*`): the 9 main paper benchmarks + a 3D Klein-Gordon case, comparing SV-SNN against 5 learned baselines.
- **Rebuttal experiments** (`rebuttal_experiments/EX_*`): 22 extended studies added during peer review — fair comparisons, ablations, robustness, scalability, and cross-paradigm baselines (FBPINN, FNO).
- All numbers are transcribed from the per-experiment Chinese analysis reports (`分析报告_EX.md`) and `saved_data/` (`*summary*.json/.csv`). No values are fabricated.

---

## 1. Main comparison (E11, authoritative)

Best relative L2, mean (min), 3 seeds, **matched parameter budget (±10%)**. Full table incl. the "rich" budget and computational costs in [E11](rebuttal_experiments/E11_grand_fair_comparison/README.md).

| Case | SV-SNN params | **SV-SNN** | SPINN | SIREN | FourierPINN | PINN | SV-SNN lead |
|------|-------------:|-----------|-------|-------|-------------|------|:-----------:|
| C1 Heat 20π | 3730 | **5.5e-4 (3.7e-4)** | 2.1e-1 | 2.9e-3 | 1.9e-3 | 8.3e-1 | 3.5× |
| C2 Helmholtz 24π | 1170 | **9.2e-3 (7.9e-3)** | 1.0e-1 | 5.5e-1 | 8.6e-1 | 9.9e-1 | 11× |
| C3 Nonlinear elliptic | 1171 | **2.0e-3 (1.9e-3)** | 1.1e-2 | 9.4e-1 | 5.0e-1 | 4.6e-1 | 5.3× |
| C4 Heat 500π | 1612 | **3.2e-3 (1.8e-3)** | 3.9e-1 | 6.2e-3 | 9.8e-3 | 1.0e+0 | 1.9× |
| C5 Helmholtz cylinder | 2322 | **1.1e-2 (6.7e-3)** | 3.5e-2 | 2.3e-1 | 8.3e-1 | 9.9e-1 | 3.1× |
| C6 Helmholtz 48π | 3096 | **9.1e-3 (2.9e-3)** | 6.2e-1 | 4.1e-1 | 4.8e-1 | 1.0e+0 | 46× |
| C7 Poisson porous | 1944 | **3.0e-2 (2.5e-2)** | 1.2e-1 | 2.7e-1 | 3.6e-1 | 1.5e+0 | 4.1× |
| C8 Taylor-Green (NS) | 2688 | **9.6e-3 (1.3e-3)** | 2.2e-1 | 3.8e-1 | 3.9e-1 | 6.5e-1 | 23× |
| C9 Dual-cylinder (NS) | 407 | **3.4e-4 (3.1e-4)** | 1.0e+0 | 9.2e-1 | 1.0e+0 | 3.2e-2 | 94× |

SV-SNN is best on **9/9 cases** at the matched budget (1.9×–94×) and remains best at the rich (50k–140k param) budget (2.1×–99×). On the two separable Helmholtz cases (C2, C6) a classical Fourier Galerkin solver reaches machine precision (~1e-13); SV-SNN's value is for harder geometries, variable coefficients, nonlinearity, and flow.

## 2. Computational cost (E11, SV-SNN, 3-seed mean)

Single NVIDIA RTX 4090 GPU. Full cross-method cost table in [E11](rebuttal_experiments/E11_grand_fair_comparison/README.md).

| Case | Params | ms/100 epoch | Peak GPU (MB) | Wall-clock (s) | Collocation |
|------|-------:|-------------:|--------------:|---------------:|------------:|
| C1 | 3730 | 326.8 | 88.4 | 32.7 | 10000 |
| C2 | 1170 | 113.1 | 89.1 | 11.3 | 10000 |
| C6 | 3096 | 134.2 | 118.8 | 13.4 | 10000 |
| C8 | 2688 | 336.1 | 322.3 | 33.6 | 125000 |
| C9 | 407 | 128.6 | 88.7 | 19.3 | 5000 |

At high collocation density SV-SNN is ~2× faster than SPINN (C8) and ~4–5× faster with ~6× less GPU memory than the rich baselines (C9).

## 3. Per-case detailed results (svsnn_acceleration)

Each row links to a self-contained doc with the PDE definition, all 6 methods (incl. original vs accelerated SV-SNN), and per-method params/L2/time.

| Case | Problem | Best L2 (SV-SNN) | Doc |
|------|---------|-----------------:|-----|
| C1 | Heat equation, κ = 20π | 3.45e-4 | [case1_heat_20pi](svsnn_acceleration/case1_heat_20pi/README.md) |
| C2 | Helmholtz, κ = 24π | 4.54e-3 | [case2_helmholtz_24pi](svsnn_acceleration/case2_helmholtz_24pi/README.md) |
| C3 | Nonlinear elliptic | 1.95e-3 | [case3_nonlinear_elliptic](svsnn_acceleration/case3_nonlinear_elliptic/README.md) |
| C4 | Heat equation, κ = 500π | 2.83e-3 | [case4_heat_500pi](svsnn_acceleration/case4_heat_500pi/README.md) |
| C5 | Helmholtz w/ cylinder, κ = 24π | 2.75e-2 | [case5_helmholtz_cylinder](svsnn_acceleration/case5_helmholtz_cylinder/README.md) |
| C6 | Helmholtz, κ = 48π | 1.65e-2 | [case6_helmholtz_48pi](svsnn_acceleration/case6_helmholtz_48pi/README.md) |
| C7 | Poisson, perforated domain, μ = 7π | 2.73e-2 | [case7_poisson_complex](svsnn_acceleration/case7_poisson_complex/README.md) |
| C8 | Taylor-Green vortex (NS) | 3.95e-3 | [case8_taylor_green](svsnn_acceleration/case8_taylor_green/README.md) |
| C9 | Double-cylinder flow (NS) | 2.25e-4 | [case9_double_cylinder_ns](svsnn_acceleration/case9_double_cylinder_ns/README.md) |
| C11 | 3D Klein-Gordon, second-order in time | 6.72e-3 | [case11_klein_gordon3d](svsnn_acceleration/case11_klein_gordon3d/README.md) |

## 4. Extended benchmarks (robustness & scalability)

| ID | Topic | Headline result | Doc |
|----|-------|-----------------|-----|
| E8 | Non-separable / aperiodic / heterogeneous | SV-SNN 0.0098–0.020 vs baselines 0.19–2.8 (13–100× lead) | [E8](rebuttal_experiments/E8_challenging_problems/README.md) |
| E9 | 3D scalability | 3D Poisson L2 = 8.3e-3 @ 1320 params; params scale linearly O(d) | [E9](rebuttal_experiments/E9_high_dim_scaling/README.md) |
| E10 | Complex-geometry near-boundary error | SV-SNN 0.049 vs FourierPINN 0.215 (4.4× lead, 25× fewer params) | [E10](rebuttal_experiments/E10_boundary_error/README.md) |
| E12 | Noisy boundary conditions | No degradation 0→10% noise (0.042→0.033); ~8–24× over baselines | [E12](rebuttal_experiments/E12_noisy_bc_ic/README.md) |
| E13 | Boundary-layer / singular perturbation | Honest limitation: ε ≤ 0.01 hard for all methods | [E13](rebuttal_experiments/E13_boundary_layer/README.md) |
| E17 | Multi-frequency solutions | ~10× lead at 2–3 components; all struggle at 4 | [E17](rebuttal_experiments/E17_multi_frequency/README.md) |
| E18 | Variable-frequency (chirp) | Best at weak chirp; SPINN better at r = 8 (frozen-spectrum limit) | [E18](rebuttal_experiments/E18_variable_frequency/README.md) |
| E21 | Non-separable Burgers shock | Reaches spectral floor (5.6e-6) on shared substrate; pure-residual PINN fails | [E21](rebuttal_experiments/E21_burgers_nonseparable/README.md) |

## 5. Ablations (what makes SV-SNN work)

| ID | Topic | Headline result | Doc |
|----|-------|-----------------|-----|
| E2 | Hybrid analytic diff vs pure AD | ~2.5× speedup, accuracy-neutral; aliasing needs ≥3 pts/wavelength | [E2](rebuttal_experiments/E2_hybrid_vs_ad_diff/README.md) |
| E3 | Component ablation (hard switch) | Separation + spectral basis decisive (10–27× degradation if removed) | [E3](rebuttal_experiments/E3_component_ablation/README.md) |
| E4 | Frequency sampling strategy | Multi-scale failure is residual imbalance, not coverage | [E4](rebuttal_experiments/E4_freq_sampling_ablation/README.md) |
| E5 | `w_char` sensitivity & auto-estimation | U-shaped sensitivity; FFT auto = 0% error vs manual | [E5](rebuttal_experiments/E5_wchar_sensitivity/README.md) |
| E6 | Effective rank validation | PINN rank collapses to ~3 (fails); SV-SNN stays 150–250 | [E6](rebuttal_experiments/E6_effrank_validation/README.md) |
| E7 | Mode count N scaling | Rank-1 saturates N≈4–6; rank-4 needs N≥4; linear scaling | [E7](rebuttal_experiments/E7_mode_scaling/README.md) |
| E15 | Structure ablation (2 axes) | Full SV-SNN optimal 9/9 at matched budget | [E15](rebuttal_experiments/E15_structure_ablation/README.md) |
| E16 | Layering strategy (135 runs) | 3-level: zero catastrophic failures (only strategy) | [E16](rebuttal_experiments/E16_layering_ablation/README.md) |
| E19 | `w_char` misestimation robustness | Underestimation fatal, overestimation safe; FFT-auto recovers | [E19](rebuttal_experiments/E19_wchar_misestimation/README.md) |
| E22 | Multi-level init innovation (243 runs) | 3-level most robust (20.1× vs 90.0× degradation) | [E22](rebuttal_experiments/E22_freq_init_innovation/README.md) |

## 6. Cross-paradigm baselines

| ID | Topic | Headline result | Doc |
|----|-------|-----------------|-----|
| E1 | Fair complete comparison (shared prior) | ~19–110× lead even when baselines share the frequency prior | [E1](rebuttal_experiments/E1_fair_complete_comparison/README.md) |
| E14 | FBPINN / domain decomposition | SV-SNN 5.8–28× better, fewest params, fastest | [E14](rebuttal_experiments/E14_fbpinn_compare/README.md) |
| E20 | FNO (data-driven operator) | Complementary regimes; SV-SNN data-free, ~3590× fewer params | [E20](rebuttal_experiments/E20_fno_compare/README.md) |

## 7. Reviewer concern → experiment → result map

| Reviewer concern | Experiment(s) | Result |
|------------------|---------------|--------|
| Fair comparison under matched budgets | [E1](rebuttal_experiments/E1_fair_complete_comparison/README.md), [E11](rebuttal_experiments/E11_grand_fair_comparison/README.md) | Best on 9/9 even with shared priors / matched params |
| Hybrid differentiation & gradient stability | [E2](rebuttal_experiments/E2_hybrid_vs_ad_diff/README.md) | 2.5× faster, accuracy-neutral, no error accumulation claim |
| Is SV-SNN just an initialization trick? | [E3](rebuttal_experiments/E3_component_ablation/README.md), [E15](rebuttal_experiments/E15_structure_ablation/README.md) | Architecture is decisive; init is a robustness boost |
| Why three-level / 40-40-20 sampling? | [E4](rebuttal_experiments/E4_freq_sampling_ablation/README.md), [E16](rebuttal_experiments/E16_layering_ablation/README.md), [E22](rebuttal_experiments/E22_freq_init_innovation/README.md) | Only strategy with zero catastrophic failures |
| `w_char` requires prior knowledge | [E5](rebuttal_experiments/E5_wchar_sensitivity/README.md), [E19](rebuttal_experiments/E19_wchar_misestimation/README.md) | FFT auto-estimation matches manual (0% error) |
| Effective-rank diagnostic justification | [E6](rebuttal_experiments/E6_effrank_validation/README.md) | Early rank predicts final accuracy |
| Effect of N / non-separable limits | [E7](rebuttal_experiments/E7_mode_scaling/README.md), [E8](rebuttal_experiments/E8_challenging_problems/README.md) | N ≥ separation rank prescription |
| No d ≥ 3 experiments | [E9](rebuttal_experiments/E9_high_dim_scaling/README.md) | True 3D, linear parameter scaling |
| Near-boundary / local-basis methods | [E10](rebuttal_experiments/E10_boundary_error/README.md), [E14](rebuttal_experiments/E14_fbpinn_compare/README.md) | Lower near-boundary error, beats FBPINN |
| Robustness to noisy BC/IC | [E12](rebuttal_experiments/E12_noisy_bc_ic/README.md) | No degradation to 10% noise |
| Boundary layers | [E13](rebuttal_experiments/E13_boundary_layer/README.md) | Honest limitation reported |
| Multi-frequency / variable frequency | [E17](rebuttal_experiments/E17_multi_frequency/README.md), [E18](rebuttal_experiments/E18_variable_frequency/README.md) | Leads except strong chirp |
| FNO baseline | [E20](rebuttal_experiments/E20_fno_compare/README.md) | Complementary; data-free advantage |
| Non-separable / shock problem | [E21](rebuttal_experiments/E21_burgers_nonseparable/README.md) | Reaches spectral floor on shared substrate |

---

## 8. Limitations (honestly reported)

SV-SNN is a well-designed PINN variant for high-frequency PDEs, not a universal solver. The paper and these experiments report the following failure modes:

- **Non-separable / shock-dominated problems.** A pure-residual Burgers formulation fails for all methods including SV-SNN (SV-SNN relative L2 ~0.325); SV-SNN only reaches the spectral floor when placed on a shared hybrid-Galerkin substrate (see [E21](rebuttal_experiments/E21_burgers_nonseparable/README.md)). Problems with high effective separation rank require more modes.
- **Thin boundary layers / singular perturbation.** For epsilon <= 0.01 all global spectral methods break down (relative L2 > 1.0); moderate widths give intermittent success across seeds (see [E13](rebuttal_experiments/E13_boundary_layer/README.md)).
- **Strong spatially varying frequency (chirp).** At r = k_max/k_min = 8, the frozen global basis cannot follow the local wavenumber and SV-SNN (~0.487) loses to SPINN (~0.086); SV-SNN leads only for weak chirp r <= 4 (see [E18](rebuttal_experiments/E18_variable_frequency/README.md)).
- **Classical spectral ceiling on idealized separable problems.** On the separable linear Helmholtz cases (C2, C6), a classical sine-Galerkin solver reaches machine precision (~1e-14); SV-SNN cannot match this on those idealized cases. SV-SNN's value is for complex geometries, variable coefficients, nonlinearity, and flow.
- **Asymmetric `w_char` robustness.** Tolerance to frequency misestimation is *not* symmetric: overestimation is safe (the low band still covers the true frequency), but severe **underestimation** (rho = w_char/kappa <= 0.7) is fatal (see [E19](rebuttal_experiments/E19_wchar_misestimation/README.md), [E22](rebuttal_experiments/E22_freq_init_innovation/README.md)). The FFT auto-estimator recovers rho ~ 1.0 when a source term is available; multi-peak or source-free problems remain a caveat.
- **Many widely separated frequencies at small budget.** With 4 well-separated components the advantage shrinks to ~1.1x and all methods struggle (see [E17](rebuttal_experiments/E17_multi_frequency/README.md)).
- **Dimensionality and noise scope.** Scaling is demonstrated through 3D (linear parameter growth); higher dimensions and noisy time-dependent initial conditions are left for future work (boundary-condition noise robustness is shown in [E12](rebuttal_experiments/E12_noisy_bc_ic/README.md)).

---

## Repository structure

```
SV-SNN/
├── svsnn_acceleration/          # 9 main benchmark cases + 3D Klein-Gordon (each has README.md)
│   ├── case1_heat_20pi/ … case9_double_cylinder_ns/, case11_klein_gordon3d/
│
├── rebuttal_experiments/        # 22 extended experiments E1–E22 (each has README.md)
│   ├── E1_fair_complete_comparison/ … E22_freq_init_innovation/
│
└── README.md                    # this file
```

## Baselines

All learned methods are compared under matched parameter budgets (within ±10%) and identical training configurations (Adam, lr = 1e-3, 10,000 epochs, 3 seeds):

- **PINN** — standard physics-informed neural network (MLP + tanh)
- **FourierPINN** — Fourier feature input embedding
- **SIREN** — sinusoidal activation functions
- **SPINN** — separable physics-informed neural network
- **FNO** — Fourier Neural Operator (data-driven baseline, see E20)
- **FBPINN** — finite-basis PINN (domain decomposition, see E14)
- **Classical Fourier spectral** — sine Galerkin reference (separable cases)

## Dependencies

- JAX + jaxlib (CUDA)
- Optax
- NumPy, SciPy
- Matplotlib

## How to run

Each experiment directory is self-contained. From a benchmark case:

```bash
cd svsnn_acceleration/case1_heat_20pi/
python run_accelerated.py
```

From a rebuttal experiment:

```bash
cd rebuttal_experiments/E11_grand_fair_comparison/
python run_all.py
python plot_E11.py
```

Results (`.json`, `.npz`, `.csv`, figures) are saved to each experiment's `saved_data/` and `figures/` directories.

## Citation

```bibtex
@article{xiong2025separated,
  title   = {Separated-Variable Spectral Neural Networks with
             Characteristic-Frequency Multi-Level Initialization for
             High-Frequency Partial Differential Equations},
  author  = {Xiong, Xiong and Zeng, Zheng and Zhang, Zhuo and
             Hu, Rongchun and Gao, Chen and Deng, Zichen},
  journal = {Engineering Applications of Artificial Intelligence},
  note    = {Revised manuscript under review},
  year    = {2025},
  eprint  = {2508.00628},
  archivePrefix = {arXiv}
}
```

## License

MIT
