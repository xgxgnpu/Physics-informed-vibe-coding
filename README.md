<div align="center">

# Physics-Informed Vibe Coding

**首个采用 Vibe Coding 理念进行科学机器学习（PINNs · PIELM · 随机特征方法 · 神经算子）研究的开源仓库**

The first open-source repository for Scientific Machine Learning research — PINNs, PIELM, Random Feature Methods, Neural Operators, and beyond — via the Vibe Coding paradigm

[![GitHub Stars](https://img.shields.io/github/stars/xgxgnpu/Physics-informed-vibe-coding?style=flat-square&logo=github&label=Stars)](https://github.com/xgxgnpu/Physics-informed-vibe-coding)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![JAX](https://img.shields.io/badge/Framework-JAX-red.svg?style=flat-square&logo=google&logoColor=white)](https://github.com/google/jax)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![GPU](https://img.shields.io/badge/GPU-CUDA-76B900.svg?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

<div align="center">

**Xiong Xiong (熊雄)** · Northwestern Polytechnical University (NWPU)

AI4PDE · Physics-Informed Deep Learning · Data-Driven Discovery

[Google Scholar](https://scholar.google.com.hk/citations?user=j1M9tkwAAAAJ) · [ResearchGate](https://www.researchgate.net/profile/Xiong-Xiong-19) · [Email](mailto:xiongxiongnwpu@mail.nwpu.edu.cn)

</div>

---

## 2026 暑期课程 · PINN & 神经算子 & AI 智能体全面加速科研

我们暑期课程【PINN & 神经算子 & AI 智能体全面加速科研】，自 2025 年暑期第 1 期开课以来，累计国内外硕博研究生和高校教师报名学员超 140 人，获一众好评！我们课程主打的就是帮助大家学明白，不仅会用，还要理解底层数学原理，以及代码算法高效准确实现，学会顶刊如何发表。2026 年暑期课程是我们第 2 期课程，课程更新进度第 2 课，暑期每周六晚上 8 点–10 点直播，欢迎大家扫码进群或者加我微信交流学习/课程咨询报名/论文辅导报名。

<div align="center">

![2026 暑期课程海报](最新课程海报.png)

**扫码加入课程交流群**

![PINN&神经算子2026暑期课程交流群](课程群.jpg)

</div>

## Philosophy

> **Vibe Coding & Vibe Researching** — Humans discover and formulate the important scientific problems; AI agents write code, execute experiments, and automatically optimize the computation pipeline; humans verify, validate, and accept the final results. Not a single line of code is written by hand.

We believe the future of computational research lies in a clear division of labor between human intellect and AI capability. **Humans** bring domain expertise to identify meaningful problems, design solution strategies, and make critical judgments on result quality. **AI agents** handle the entire implementation cycle — from writing code and running experiments to tuning hyperparameters and generating visualizations. This closed-loop workflow enables researchers to focus on scientific insight rather than engineering overhead. Every algorithm in this repository is implemented in **JAX** with GPU acceleration, and each case is fully reproducible with saved data, figures, and model checkpoints.

> **Vibe Coding & Vibe Researching** — 人类负责发现并提出重要科学问题，AI 智能体负责编程、执行实验与自动优化计算过程，最终由人类核对、验证并验收结果。全程不手写一行代码，（尝试）完成完整的复杂科研项目。

本项目提供一系列**完整、自包含的 JAX-GPU 实现**，覆盖科学机器学习（Scientific Machine Learning）领域的前沿算法，包括 PINNs、物理信息极限学习机（PIELM）、随机特征方法、神经算子等，并配套详细的中文学术教程。**人类**凭借领域知识发现有价值的科学问题、设计求解策略、把控结果质量；**AI 智能体**承担从代码编写、实验运行到超参调优、可视化生成的全链路工作。这一闭环协作模式使研究者得以专注于科学洞察，而非工程细节。

## Contents

| # | Algorithm | Directory | Tutorial | Status |
|---|-----------|-----------|----------|--------|
| 1 | **NTK-PINN** — Neural Tangent Kernel adaptive weighting | [`NTK-PINN-jax/`](NTK-PINN-jax/) | [NTK-PINN 教程](tutorials/NTK-PINN-tutorial.md) | Done |
| 2 | **MultiScale-PINN** — Multi-scale Fourier feature networks for PDEs | [`MultiScalePINN_jax/`](MultiScalePINN_jax/) | 待发布 | Done |
| 3 | **VS-PINN** — Variable-Scaling PINN for Navier-Stokes | [`VSPINN_jax/`](VSPINN_jax/) | [VS-PINN 教程](tutorials/VSPINN/VSPINN-tutorial.md) | Done |
| 4 | **GW-PINN** — Gradient-Weighted adaptive loss balancing | [`GradientWeighted_PINN_jax/`](GradientWeighted_PINN_jax/) | [GW-PINN 教程](tutorials/GradientWeighted-PINN/GradientWeighted-PINN-tutorial.md) | Done |
| 5 | **Scale-PINN** — Evolutionary regularization for high-Re flows | [`ScalePINN-jax/`](ScalePINN-jax/) | 待发布 | Done |
| 6 | **Maxwell-PINN (No BO)** — Pure PINN for 2D EM scattering (Helmholtz + ABC) | `MaxwellPINN_jax/` | 待发布 | Done |
| 7 | **TINN** — Time-Induced Neural Networks with Levenberg-Marquardt for 1D Burgers | `TINN_jax/` | 待发布 | Done |
| 8 | **RFM** — Random Feature Method for PDEs (non-iterative least-squares solver) | [`RFM_jax/`](RFM_jax/) | 待发布 | Done |

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

```bash
cd RFM_jax/case1_stokes_2d/
python stokes_2d_rfm.py
```

All results (data `.txt`, figures `.png`, checkpoints `.pkl`) are saved automatically.

## License

MIT

## Citation

If you find this repository useful, please consider citing it:

```bibtex
@software{xiong2026vibecoding,
  author       = {Xiong, Xiong},
  title        = {Physics-Informed Vibe Coding: Scientific Machine Learning in JAX via Human-AI Collaboration},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/xgxgnpu/Physics-informed-vibe-coding},
  note         = {PINNs, PIELM, Random Feature Methods, Neural Operators, and beyond}
}
```

You may also cite the following related works:

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

@article{chen2022rfm,
  title={Bridging Traditional and Machine Learning-based Algorithms for Solving PDEs: The Random Feature Method},
  author={Chen, Jingrun and Chi, Xurong and E, Weinan and Yang, Zhouwang},
  journal={Journal of Machine Learning},
  volume={1},
  number={3},
  pages={268--298},
  year={2022},
  doi={10.4208/jml.220726}
}

@article{dwivedi2020pielm,
  title={Physics Informed Extreme Learning Machine (PIELM)--A rapid method for the numerical solution of partial differential equations},
  author={Dwivedi, Vikas and Srinivasan, Balaji},
  journal={Neurocomputing},
  volume={391},
  pages={96--118},
  year={2020},
  doi={10.1016/j.neucom.2019.12.099}
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
