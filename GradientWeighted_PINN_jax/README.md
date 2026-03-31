# GradientWeighted PINN (JAX)

JAX reproduction of **"Understanding and mitigating gradient flow pathologies in
physics-informed neural networks"** (Wang, Teng & Perdikaris, SIAM J. Sci. Comput. 2021).

## Cases

| Case | Problem | Reference |
|------|---------|-----------|
| case1_lid_driven_cavity | 2D Steady Lid-Driven Cavity (Re=100) | Ghia et al. benchmark |
| case2_klein_gordon | 1D Klein-Gordon (nonlinear) | Wang et al. 2021 |

## Method

- **M1**: Standard PINN (fixed equal weighting for all loss terms)
- **M2**: Gradient-weighted adaptive λ for IC/BC losses (learning rate annealing)
- Exponential moving average (β=0.9) for smooth λ updates
- Input normalization (z-score) with chain-rule derivative correction
- Adam optimizer

## Run

```bash
cd case1_lid_driven_cavity
python lid_driven_cavity_gw_pinn.py
```

```bash
cd case2_klein_gordon
python klein_gordon_gw_pinn.py
```

Results are saved to `data/`, `figures/`, and `checkpoints/` subdirectories.

## References

```bibtex
@article{wang2021understanding,
  title={Understanding and mitigating gradient flow pathologies in physics-informed neural networks},
  author={Wang, Sifan and Teng, Yujun and Perdikaris, Paris},
  journal={SIAM Journal on Scientific Computing},
  volume={43},
  number={5},
  pages={A3055--A3081},
  year={2021},
  publisher={SIAM},
  doi={10.1137/20M1318043}
}

@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.jcp.2018.10.045}
}

@article{wang2022and,
  title={When and why PINNs fail to train: A neural tangent kernel perspective},
  author={Wang, Sifan and Yu, Xinling and Perdikaris, Paris},
  journal={Journal of Computational Physics},
  volume={449},
  pages={110768},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.jcp.2021.110768}
}
```
