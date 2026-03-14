# Physics-Informed Vibe Coding

A collection of **complete, self-contained JAX implementations** of state-of-the-art Physics-Informed Neural Network (PINN) algorithms, accompanied by detailed Chinese tutorials.

## Philosophy

> **Vibe Coding & Vibe Researching** — Humans design, direct, and validate; AI agents execute. Not a single line of code is written by hand.

Every algorithm in this repository is implemented in **JAX** with GPU acceleration, and each case is fully reproducible with saved data, figures, and model checkpoints.

## Contents

| # | Algorithm | Directory | Tutorial | Status |
|---|-----------|-----------|----------|--------|
| 1 | **NTK-PINN** — Neural Tangent Kernel adaptive weighting | [`NTK-PINN-jax/`](NTK-PINN-jax/) | [NTK-PINN 教程](tutorials/NTK-PINN-tutorial.md) | Done |

## Environment

```bash
# JAX 0.6.0 + CUDA GPU
source /root/autodl-tmp/pinn_env/bin/activate
```

Key dependencies: `jax`, `jaxlib` (CUDA), `optax`, `matplotlib`, `numpy`, `scipy`

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

If you find this repository useful, please consider citing the original papers referenced in each tutorial.
