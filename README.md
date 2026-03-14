# Physics-Informed Vibe Coding

> **首个采用 Vibe Coding 理念进行 Physics-Informed Neural Networks (PINNs) 相关研究的开源仓库。**
>
> The first open-source repository dedicated to PINN research via the Vibe Coding paradigm.

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

## Dependencies

`jax`, `jaxlib` (CUDA), `optax`, `matplotlib`, `numpy`, `scipy`

## How to Run

Each case directory contains a single self-contained `.py` file:

```bash
cd NTK-PINN-jax/case2_wave1d/
python wave1d_ntk_pinn.py
```

All results (data `.txt`, figures `.png`, checkpoints `.pkl`) are saved automatically.

## License

MIT

## Citation

If you find this repository useful, please consider citing our related works:

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
