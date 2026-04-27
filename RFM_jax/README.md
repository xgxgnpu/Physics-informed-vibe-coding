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

## References

```bibtex
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

@article{raissi2019pinn,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  doi={10.1016/j.jcp.2018.10.045}
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

@article{huang2006elm,
  title={Extreme learning machine: Theory and applications},
  author={Huang, Guang-Bin and Zhu, Qin-Yu and Siew, Chee-Kheong},
  journal={Neurocomputing},
  volume={70},
  number={1-3},
  pages={489--501},
  year={2006},
  doi={10.1016/j.neucom.2005.12.126}
}

@article{xiong2025separated,
  title={Separated-variable spectral neural networks: A physics-informed learning approach for high-frequency PDEs},
  author={Xiong, Xiong and Zhang, Zhuo and Hu, Rongchun and Gao, Chen and Deng, Zichen},
  journal={arXiv preprint arXiv:2508.00628},
  year={2025}
}
```

## Dependencies

`jax`, `jaxlib` (CUDA), `numpy`, `matplotlib`
