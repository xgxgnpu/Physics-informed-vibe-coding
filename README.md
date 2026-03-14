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

If you find this repository useful, please consider citing the original papers referenced in each tutorial.
