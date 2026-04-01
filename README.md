# Physics-Informed Vibe Coding

> **首个采用 Vibe Coding 理念进行 Physics-Informed Neural Networks (PINNs) 相关研究的开源仓库。**
>
> The first open-source repository dedicated to PINN research via the Vibe Coding paradigm.

## Author

**Xiong Xiong (熊雄)**

- Northwestern Polytechnical University (NWPU)
- Research interests: AI4PDE, Physics-Informed Deep Learning, Data-Driven Discovery
- Email: xiongxiongnwpu@mail.nwpu.edu.cn
- Google Scholar: [Xiong Xiong](https://scholar.google.com.hk/citations?user=j1M9tkwAAAAJ)
- ResearchGate: [Xiong Xiong](https://www.researchgate.net/profile/Xiong-Xiong-19)

---

本项目提供一系列**完整、自包含的 JAX-GPU 实现**，覆盖 PINN 领域的前沿算法，并配套详细的中文学术教程。所有代码均由 AI 智能体在人类指导下完成，不手写一行代码。

## Philosophy

> **Vibe Coding & Vibe Researching** — 人类负责设计、指挥和验收把控，AI 智能体负责执行。不用手写一行代码，（尝试）完成完整的复杂科研项目。
>
> Humans design, direct, and validate; AI agents execute. Not a single line of code is written by hand.

Every algorithm in this repository is implemented in **JAX** with GPU acceleration, and each case is fully reproducible with saved data, figures, and model checkpoints.

## Contents

| # | Algorithm | Directory | Tutorial | Status |
|---|-----------|-----------|----------|--------|
| 1 | **NTK-PINN** — Neural Tangent Kernel adaptive weighting | [`NTK-PINN-jax/`](NTK-PINN-jax/) | [NTK-PINN 教程](tutorials/NTK-PINN-tutorial.md) | Done |
| 2 | **MultiScale-PINN** — Multi-scale Fourier feature networks for PDEs | [`MultiScalePINN_jax/`](MultiScalePINN_jax/) | [MultiScale-PINN 教程](tutorials/MultiScale-PINN/MultiScale-PINN-tutorial.md) | Done |
| 3 | **VS-PINN** — Variable-Scaling PINN for Navier-Stokes | [`VSPINN_jax/`](VSPINN_jax/) | 待发布 | Done |
| 4 | **GW-PINN** — Gradient-Weighted adaptive loss balancing | [`GradientWeighted_PINN_jax/`](GradientWeighted_PINN_jax/) | 待发布 | Done |
| 5 | **Scale-PINN** — Evolutionary regularization for high-Re flows | [`ScalePINN-jax/`](ScalePINN-jax/) | [Scale-PINN 教程](tutorials/ScalePINN/ScalePINN-tutorial.md) | Done |

## Dependencies

`jax`, `jaxlib` (CUDA), `optax`, `matplotlib`, `numpy`, `scipy`

## How to Run

Each case directory contains a single self-contained `.py` file:

```bash
cd NTK-PINN-jax/case2_wave1d/
python wave1d_ntk_pinn.py
```

```bash
cd MultiScalePINN_jax/case1_heat1d/
python heat1d_multiscale_pinn.py
```

```bash
cd VSPINN_jax/case1_ns2d/
python ns2d_vspinn_pinn.py
```

```bash
cd GradientWeighted_PINN_jax/case2_klein_gordon/
python klein_gordon_gw_pinn.py
```

```bash
cd ScalePINN-jax/case1_ldc_re7500/
python ldc_re7500_scalepinn.py
```

All results (data `.txt`, figures `.png`, checkpoints `.pkl`) are saved automatically.

## License

MIT

## Citation

If you find this repository useful, please consider citing the following related works:

```bibtex
@article{WANG2022110768,
  title={When and why PINNs fail to train: A neural tangent kernel perspective},
  author={Wang, Sifan and Yu, Xinling and Perdikaris, Paris},
  journal={Journal of Computational Physics},
  volume={449},
  pages={110768},
  year={2022},
  doi={https://doi.org/10.1016/j.jcp.2021.110768},
  publisher={Elsevier}
}

@article{WANG2021113938,
  title={On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks},
  author={Wang, Sifan and Wang, Hanwen and Perdikaris, Paris},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={384},
  pages={113938},
  year={2021},
  doi={https://doi.org/10.1016/j.cma.2021.113938},
  publisher={Elsevier}
}

@article{xiong2025high,
  title={High-frequency flow field super-resolution via physics-informed hierarchical adaptive Fourier feature networks},
  author={Xiong, Xiong and Lu, Kang and Zhang, Zhuo and Zeng, Zheng and Zhou, Sheng and Hu, Rongchun and Deng, Zichen},
  journal={Physics of Fluids},
  volume={37},
  number={9},
  year={2025},
  publisher={AIP Publishing}
}

@article{xiong2025j,
  title={J-PIKAN: A physics-informed KAN network based on Jacobi orthogonal polynomials for solving fluid dynamics},
  author={Xiong, Xiong and Lu, Kang and Zhang, Zhuo and Zeng, Zheng and Zhou, Sheng and Deng, Zichen and Hu, Rongchun},
  journal={Communications in Nonlinear Science and Numerical Simulation},
  pages={109414},
  year={2025},
  publisher={Elsevier}
}

@article{xiong2025separated,
  title={Separated-variable spectral neural networks: a physics-informed learning approach for high-frequency pdes},
  author={Xiong, Xiong and Zhang, Zhuo and Hu, Rongchun and Gao, Chen and Deng, Zichen},
  journal={arXiv preprint arXiv:2508.00628},
  year={2025}
}

@article{zhang2025legend,
  title={Legend-KINN: A Legendre Polynomial-Based Kolmogorov-Arnold-Informed Neural Network for Efficient PDE Solving},
  author={Zhang, Zhuo and Xiong, Xiong and Zhang, Sen and Wang, Wei and Zhong, Yanxu and Yang, Canqun and Yang, Xi},
  journal={Expert Systems with Applications},
  pages={129839},
  year={2025},
  publisher={Elsevier}
}
```
