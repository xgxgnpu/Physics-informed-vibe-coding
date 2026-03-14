# Physics-Informed Vibe Coding 之 NTK-PINN

> **系列导读**
>
> 本学期，我们将持续推出一系列 **Physics-Informed Vibe Coding** 的代码算法实验与教程，致力于践行 **Vibe Coding & Vibe Researching** 的理念——**人类负责设计、指挥和验收把控，AI 智能体负责执行**，即不用手写一行代码，（尝试）完成完整的复杂科研项目。
>
> 我们的全部代码和教程将开源在 [GitHub: Physics-informed-vibe-coding](https://github.com/xgxgnpu/Physics-informed-vibe-coding) 仓库，所有实现均采用 **JAX** 语言进行 GPU 加速编程。
>
> 本期主题：**NTK-PINN —— 基于神经正切核的自适应权重物理信息神经网络**，以一维波动方程为算例，详解 NTK 理论在 PINN 训练中的应用。

---

## 摘要

物理信息神经网络（Physics-Informed Neural Networks, PINNs）通过将偏微分方程（PDE）残差嵌入损失函数，实现了无网格、无标注数据的 PDE 求解范式。然而，标准 PINN 的多任务损失函数在训练过程中往往面临严重的**收敛速率不平衡**问题——边界/初始条件损失与 PDE 残差损失的梯度尺度可能相差数个量级，导致网络难以同时满足所有约束。

Wang 等人 [1] 从**神经正切核（Neural Tangent Kernel, NTK）**的视角揭示了这一现象的数学本质：不同损失分量对应的 NTK 子矩阵迹（trace）之比决定了各约束的有效学习速率。基于此分析，他们提出了一种**NTK 自适应权重算法**，通过动态调整各损失分量的权重系数来平衡训练动态。

本教程以一维波动方程为例，基于 JAX 框架完整复现了 NTK-PINN 算法，并从以下维度展开详细讨论：

- NTK 理论在 PINN 框架下的完整数学推导
- JAX 实现中的网络结构与维度变换分析
- 核心代码块（PDE 残差计算、NTK Jacobian 计算、自适应权重更新）的数学原理解读
- 输入归一化与梯度链式法则修正的实现细节
- 实验结果的量化对比分析

---

## 1. 引言

### 1.1 PINN 的基本思想

PINN [2] 的核心思想是将 PDE 的物理约束融入神经网络的损失函数。考虑一个一般性的 PDE 系统：

$$\mathcal{N}[u](x) = 0, \quad x \in \Omega$$

$$\mathcal{B}[u](x) = 0, \quad x \in \partial\Omega$$

其中 $\mathcal{N}$ 为微分算子，$\mathcal{B}$ 为边界条件算子。PINN 用一个参数为 $\theta$ 的神经网络 $u_\theta(x)$ 来近似 $u(x)$，通过最小化如下复合损失函数进行训练：

$$\mathcal{L}(\theta) = \lambda_r \mathcal{L}_r(\theta) + \lambda_b \mathcal{L}_b(\theta)$$

其中 $\mathcal{L}_r$ 为 PDE 残差损失，$\mathcal{L}_b$ 为边界/初始条件损失，$\lambda_r, \lambda_b$ 为权重系数。

### 1.2 多任务学习的不平衡问题

标准 PINN 通常设定 $\lambda_r = \lambda_b = 1$，但这一朴素选择在实际中频繁导致训练失败。其根本原因在于：**不同损失分量的梯度尺度和收敛速率可能存在巨大差异**。

具体而言，PDE 残差 $\mathcal{L}_r$ 涉及网络输出的高阶微分（如二阶导数 $u_{xx}$、$u_{tt}$），其梯度信号往往随网络深度呈指数衰减；而边界条件 $\mathcal{L}_b$ 仅约束网络的零阶输出 $u$，梯度信号相对稳定。这种不对称性导致网络在训练初期过分拟合边界条件，而 PDE 残差收敛缓慢，最终产生不物理的解。

### 1.3 NTK 视角的动机

Wang 等人 [1] 指出，可以利用**神经正切核**（NTK）[3] 理论来**定量刻画**上述不平衡现象，并据此设计自适应权重策略。NTK 将无限宽度神经网络的训练动态等价为一个核回归问题，其中核矩阵的特征值谱直接决定了各方向的收敛速率。将这一分析应用于 PINN 的分块损失结构，可以自然地推导出最优权重的解析表达式。

---

## 2. 方法与数学原理

### 2.1 PINN 基本框架

以本教程使用的一维波动方程为例：

$$u_{tt} = c^2 u_{xx}, \quad (t, x) \in [0, 1]^2, \quad c = 2$$

配合初始条件和边界条件：

$$u(0, x) = \sin(\pi x) + 0.5 \sin(4\pi x), \quad x \in [0, 1]$$

$$u_t(0, x) = 0, \quad x \in [0, 1]$$

$$u(t, 0) = u(t, 1) = 0, \quad t \in [0, 1]$$

精确解为：

$$u(t, x) = \sin(\pi x)\cos(2\pi t) + 0.5\sin(4\pi x)\cos(8\pi t)$$

下图展示了 PINN 求解波动方程的整体框架：

![PINN Framework for Wave Equation](https://pic1.imgdb.cn/item/69b4e8f6963e55431f540517.png)

**图 S1**：PINN 框架示意图。神经网络 $u_\theta(t,x)$ 接收时空坐标作为输入，输出预测解 $\hat{u}$。通过自动微分获取高阶导数以构建 PDE 残差，与边界/初始条件损失一起组成加权总损失，反向传播更新网络参数。

PINN 的损失函数分解为四个分量：

| 损失分量 | 数学表达式 | 物理含义 |
|---------|-----------|---------|
| $\mathcal{L}_r$ | $\frac{1}{N_r}\sum_{i=1}^{N_r}\left(u_{tt}^{\theta}(t_i, x_i) - c^2 u_{xx}^{\theta}(t_i, x_i)\right)^2$ | PDE 残差 |
| $\mathcal{L}_u$ | $\frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}}\left(u^\theta(0, x_i) - u_0(x_i)\right)^2$ | 初始位移条件 |
| $\mathcal{L}_{u_t}$ | $\frac{1}{N_{ic}}\sum_{i=1}^{N_{ic}}\left(u_t^\theta(0, x_i)\right)^2$ | 初始速度条件 |
| $\mathcal{L}_{bc}$ | $\frac{1}{N_{bc}}\sum_{i=1}^{N_{bc}}\left(u^\theta(t_i, 0)\right)^2 + \left(u^\theta(t_i, 1)\right)^2$ | 边界条件 |

总损失为加权求和：

$$\mathcal{L}(\theta) = \lambda_u \left(\mathcal{L}_u + \mathcal{L}_{bc}\right) + \lambda_{u_t} \mathcal{L}_{u_t} + \lambda_r \mathcal{L}_r$$

### 2.2 Neural Tangent Kernel 理论基础

#### 2.2.1 NTK 的定义

考虑一个参数为 $\theta \in \mathbb{R}^P$ 的神经网络 $f_\theta: \mathbb{R}^d \to \mathbb{R}$，其在一组训练点 $\{x_i\}_{i=1}^N$ 上的输出向量为 $\mathbf{f}(\theta) = [f_\theta(x_1), \ldots, f_\theta(x_N)]^\top \in \mathbb{R}^N$。

**Neural Tangent Kernel** 定义为：

$$K(\theta) = J(\theta) J(\theta)^\top \in \mathbb{R}^{N \times N}$$

其中 $J(\theta) = \frac{\partial \mathbf{f}(\theta)}{\partial \theta} \in \mathbb{R}^{N \times P}$ 为 Jacobian 矩阵。NTK 矩阵的第 $(i, j)$ 元素为：

$$K_{ij}(\theta) = \left\langle \frac{\partial f_\theta(x_i)}{\partial \theta}, \frac{\partial f_\theta(x_j)}{\partial \theta} \right\rangle = \sum_{p=1}^{P} \frac{\partial f_\theta(x_i)}{\partial \theta_p} \frac{\partial f_\theta(x_j)}{\partial \theta_p}$$

#### 2.2.2 NTK 与梯度下降动态

在连续时间梯度下降（gradient flow）框架下，参数更新遵循：

$$\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta)$$

对于均方误差损失 $\mathcal{L}(\theta) = \frac{1}{2}\|\mathbf{f}(\theta) - \mathbf{y}\|^2$，网络输出的演化方程为：

$$\frac{d\mathbf{f}}{dt} = J(\theta) \frac{d\theta}{dt} = -J(\theta) J(\theta)^\top (\mathbf{f} - \mathbf{y}) = -K(\theta)(\mathbf{f} - \mathbf{y})$$

定义残差 $\mathbf{e}(t) = \mathbf{f}(t) - \mathbf{y}$，则：

$$\frac{d\mathbf{e}}{dt} = -K(\theta(t))\,\mathbf{e}(t)$$

Jacot 等人 [3] 证明，在网络宽度趋于无穷的极限下，NTK 矩阵 $K(\theta(t)) \to K^*$ 趋于常数。此时残差的解为：

$$\mathbf{e}(t) = e^{-K^* t}\,\mathbf{e}(0)$$

将 $K^*$ 进行特征分解 $K^* = Q \Lambda Q^\top$（其中 $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_N)$），则在各特征方向上：

$$e_k(t) = e^{-\lambda_k t} e_k(0)$$

**关键结论**：NTK 的特征值 $\lambda_k$ 直接决定了对应方向的收敛速率。特征值越大，对应残差分量收敛越快。

### 2.3 PINN 的 NTK 推导

#### 2.3.1 分块 NTK 矩阵

在 PINN 框架下，损失函数涉及多种网络输出算子（$u$, $u_t$, $u_{tt} - c^2 u_{xx}$ 等），因此 NTK 矩阵具有**分块结构**。

对于 Wave 方程的 PINN，定义三组 Jacobian 矩阵：

$$J_u = \frac{\partial \mathbf{u}(\theta)}{\partial \theta} \in \mathbb{R}^{N_u \times P}$$

$$J_{u_t} = \frac{\partial \mathbf{u}_t(\theta)}{\partial \theta} \in \mathbb{R}^{N_{ut} \times P}$$

$$J_r = \frac{\partial \mathbf{r}(\theta)}{\partial \theta} \in \mathbb{R}^{N_r \times P}$$

其中 $\mathbf{u}$ 为边界/初始位移预测向量，$\mathbf{u}_t$ 为初始速度预测向量，$\mathbf{r}$ 为 PDE 残差向量。

下图直观展示了 NTK 分块结构及其与自适应权重的关系：

![NTK Block Structure and Adaptive Weights](https://pic1.imgdb.cn/item/69b4e8f7963e55431f540518.png)

**图 S2**：NTK 分块结构与自适应权重示意图。三组 Jacobian 矩阵构成全 NTK 矩阵的分块结构，对角块的迹（trace）决定各损失分量的有效收敛速率。自适应权重 $\lambda_i = \mathrm{Tr}(K)/\mathrm{Tr}(K_{ii})$ 如同一个"天平"，通过增大慢收敛分量的权重来平衡训练动态。

全 NTK 矩阵为：

$$K = \begin{bmatrix} J_u \\ J_{u_t} \\ J_r \end{bmatrix} \begin{bmatrix} J_u \\ J_{u_t} \\ J_r \end{bmatrix}^\top = \begin{bmatrix} K_{uu} & K_{u,u_t} & K_{u,r} \\ K_{u_t,u} & K_{u_tu_t} & K_{u_t,r} \\ K_{r,u} & K_{r,u_t} & K_{rr} \end{bmatrix}$$

其中对角块 $K_{uu} = J_u J_u^\top$, $K_{u_tu_t} = J_{u_t} J_{u_t}^\top$, $K_{rr} = J_r J_r^\top$ 分别控制三个损失分量的收敛动态。

#### 2.3.2 收敛速率差异的数学分析

在梯度流下，三个损失分量的残差演化为：

$$\frac{d}{dt}\begin{bmatrix} \mathbf{e}_u \\ \mathbf{e}_{u_t} \\ \mathbf{e}_r \end{bmatrix} = -\begin{bmatrix} \lambda_u K_{uu} & \lambda_{u_t} K_{u,u_t} & \lambda_r K_{u,r} \\ \lambda_u K_{u_t,u} & \lambda_{u_t} K_{u_tu_t} & \lambda_r K_{u_t,r} \\ \lambda_u K_{r,u} & \lambda_{u_t} K_{r,u_t} & \lambda_r K_{rr} \end{bmatrix} \begin{bmatrix} \mathbf{e}_u \\ \mathbf{e}_{u_t} \\ \mathbf{e}_r \end{bmatrix}$$

忽略非对角块的耦合效应，各分量的有效收敛速率由 $\lambda_i \cdot \mathrm{Tr}(K_{ii})$ 决定。当 $\mathrm{Tr}(K_{uu}) \gg \mathrm{Tr}(K_{rr})$ 时（这在 PINN 中非常常见，因为 PDE 残差涉及高阶微分），边界条件会以远快于 PDE 残差的速率收敛，导致训练不平衡。

### 2.4 NTK 自适应权重算法

#### 2.4.1 算法推导

为使各损失分量以**相同的有效速率**收敛，要求：

$$\lambda_u \cdot \mathrm{Tr}(K_{uu}) \approx \lambda_{u_t} \cdot \mathrm{Tr}(K_{u_tu_t}) \approx \lambda_r \cdot \mathrm{Tr}(K_{rr})$$

令各分量的有效速率均等于某一常数 $\rho$，则：

$$\lambda_i = \frac{\rho}{\mathrm{Tr}(K_{ii})}$$

取 $\rho = \mathrm{Tr}(K)$（全 NTK 矩阵的迹），得到：

$$\boxed{\lambda_i = \frac{\mathrm{Tr}(K)}{\mathrm{Tr}(K_{ii})}, \quad i \in \{u, u_t, r\}}$$

其中 $\mathrm{Tr}(K) = \mathrm{Tr}(K_{uu}) + \mathrm{Tr}(K_{u_tu_t}) + \mathrm{Tr}(K_{rr})$。

#### 2.4.2 算法流程

NTK 自适应权重的完整训练流程如下：

| 步骤 | 操作 | 频率 |
|-----|------|------|
| 1 | 采样一批训练点（IC, BC, 残差点） | 每步 |
| 2 | 计算损失函数，反向传播更新参数 | 每步 |
| 3 | 采样 NTK 计算点 | 每 $M$ 步（本文 $M = 100$） |
| 4 | 计算 Jacobian 矩阵 $J_u, J_{u_t}, J_r$ | 每 $M$ 步 |
| 5 | 计算对角块 $K_{uu}, K_{u_tu_t}, K_{rr}$ 及其迹 | 每 $M$ 步 |
| 6 | 更新权重 $\lambda_i = \mathrm{Tr}(K) / \mathrm{Tr}(K_{ii})$ | 每 $M$ 步 |
| 7 | 重复步骤 1-6 直至收敛 | — |

### 2.5 输入归一化与梯度链式法则修正

#### 2.5.1 Z-score 归一化

对于物理域 $(t, x) \in [0, 1]^2$，定义 z-score 归一化：

$$\tilde{t} = \frac{t - \mu_t}{\sigma_t}, \quad \tilde{x} = \frac{x - \mu_x}{\sigma_x}$$

其中 $\mu_t, \sigma_t, \mu_x, \sigma_x$ 为从均匀采样点计算得到的均值和标准差。对于 $[0, 1]$ 上的均匀分布，理论值约为 $\mu \approx 0.5$, $\sigma \approx 0.289$。

网络接收归一化坐标 $(\tilde{t}, \tilde{x})$ 作为输入，输出预测值 $\hat{u}(\tilde{t}, \tilde{x})$。

#### 2.5.2 链式法则修正

由于 PDE 中的微分是对**物理坐标**定义的，而网络的自动微分是对**归一化坐标**进行的，因此需要进行链式法则修正：

$$\frac{\partial u}{\partial t} = \frac{\partial \hat{u}}{\partial \tilde{t}} \cdot \frac{\partial \tilde{t}}{\partial t} = \frac{1}{\sigma_t} \frac{\partial \hat{u}}{\partial \tilde{t}}$$

$$\frac{\partial^2 u}{\partial t^2} = \frac{1}{\sigma_t^2} \frac{\partial^2 \hat{u}}{\partial \tilde{t}^2}$$

$$\frac{\partial^2 u}{\partial x^2} = \frac{1}{\sigma_x^2} \frac{\partial^2 \hat{u}}{\partial \tilde{x}^2}$$

因此，PDE 残差在归一化坐标下的计算公式为：

$$r = \frac{1}{\sigma_t^2} \frac{\partial^2 \hat{u}}{\partial \tilde{t}^2} - c^2 \cdot \frac{1}{\sigma_x^2} \frac{\partial^2 \hat{u}}{\partial \tilde{x}^2}$$

下表总结了各物理量的归一化修正因子：

| 物理量 | 归一化坐标下的表达式 | 修正因子 |
|-------|-------------------|---------|
| $u$ | $\hat{u}(\tilde{t}, \tilde{x})$ | 1（无需修正） |
| $u_t$ | $\frac{1}{\sigma_t}\frac{\partial \hat{u}}{\partial \tilde{t}}$ | $1/\sigma_t$ |
| $u_{tt}$ | $\frac{1}{\sigma_t^2}\frac{\partial^2 \hat{u}}{\partial \tilde{t}^2}$ | $1/\sigma_t^2$ |
| $u_{xx}$ | $\frac{1}{\sigma_x^2}\frac{\partial^2 \hat{u}}{\partial \tilde{x}^2}$ | $1/\sigma_x^2$ |

---

## 3. JAX 实现详解

### 3.1 网络结构与维度分析

本算例使用一个 4 层全连接网络，结构为 `[2, 500, 500, 500, 1]`，激活函数为 $\tanh$，初始化采用 Xavier 方法。下图展示了完整的网络结构、各层维度变换以及归一化链式法则修正：

![Network Architecture and Dimension Flow](https://pic1.imgdb.cn/item/69b4e8f9963e55431f54051a.png)

**图 S3**：网络结构与维度变换示意图。归一化坐标 $(\tilde{t}, \tilde{x})$ 经过 3 个隐藏层（各 500 神经元，$\tanh$ 激活）到输出层，总参数量 502,001。下方展示了物理空间导数与归一化坐标导数之间的链式法则修正关系。

#### 3.1.1 层级维度变换

下表详细展示了数据在网络中逐层传播时的维度变化过程（设 batch size 为 $N$）：

| 层 | 输入维度 | 权重 $W$ 维度 | 偏置 $b$ 维度 | 输出维度 | 激活函数 | 参数量 |
|----|---------|-------------|-------------|---------|---------|-------|
| 输入层 | $(N, 2)$ | — | — | $(N, 2)$ | — | 0 |
| 隐藏层 1 | $(N, 2)$ | $(2, 500)$ | $(500,)$ | $(N, 500)$ | $\tanh$ | 1,500 |
| 隐藏层 2 | $(N, 500)$ | $(500, 500)$ | $(500,)$ | $(N, 500)$ | $\tanh$ | 250,500 |
| 隐藏层 3 | $(N, 500)$ | $(500, 500)$ | $(500,)$ | $(N, 500)$ | $\tanh$ | 250,500 |
| 输出层 | $(N, 500)$ | $(500, 1)$ | $(1,)$ | $(N, 1)$ | 无 | 501 |
| **合计** | | | | | | **502,001** |

#### 3.1.2 维度变换流程图

```
输入 (N, 2)  ──[W₁: 2×500, b₁: 500]──> (N, 500) ──tanh──> (N, 500)
                                                               │
            ──[W₂: 500×500, b₂: 500]──> (N, 500) ──tanh──> (N, 500)
                                                               │
            ──[W₃: 500×500, b₃: 500]──> (N, 500) ──tanh──> (N, 500)
                                                               │
            ──[W₄: 500×1, b₄: 1]──────> (N, 1)  ──────────> 输出 û
```

其中每层的前向计算为：

$$\mathbf{h}^{(l+1)} = \tanh(\mathbf{h}^{(l)} W^{(l)} + b^{(l)})$$

网络的输入为归一化坐标 $(\tilde{t}, \tilde{x})$，输出为预测值 $\hat{u}$。网络的参数 $\theta$ 包含所有层的权重和偏置，展平后形成一个 $P = 502,001$ 维的参数向量。

### 3.2 核心代码块解读

#### 3.2.1 PDE 残差计算

PDE 残差的计算是 PINN 的核心。以下代码展示了如何利用 JAX 的自动微分（`jax.grad`）计算二阶偏导数，并应用链式法则修正：

```python
def net_residual_single(params, t_norm, x_norm):
    du_dt_norm = grad(net_u_single, argnums=1)
    du_dx_norm = grad(net_u_single, argnums=2)
    d2u_dt2_norm = grad(du_dt_norm, argnums=1)
    d2u_dx2_norm = grad(du_dx_norm, argnums=2)
    u_tt_phys = d2u_dt2_norm(params, t_norm, x_norm) / (SIGMA_T ** 2)
    u_xx_phys = d2u_dx2_norm(params, t_norm, x_norm) / (SIGMA_X ** 2)
    return u_tt_phys - C_PARAM ** 2 * u_xx_phys
```

**数学原理解析**：

1. `grad(net_u_single, argnums=1)` 对归一化时间坐标 $\tilde{t}$ 求导，得到 $\partial \hat{u}/\partial \tilde{t}$
2. 再次对 $\tilde{t}$ 求导，得到 $\partial^2 \hat{u}/\partial \tilde{t}^2$
3. 除以 $\sigma_t^2$ 得到物理空间的 $u_{tt}$；类似地处理 $u_{xx}$
4. 最终返回 PDE 残差 $r = u_{tt} - c^2 u_{xx}$

**维度分析**：此函数处理标量输入 $(t_{\text{norm}}, x_{\text{norm}})$，输出标量残差 $r$。通过 `vmap` 向量化，可以一次处理整个 batch：

```python
net_residual_batch = jit(vmap(net_residual_single, in_axes=(None, 0, 0)))
```

`vmap(... , in_axes=(None, 0, 0))` 表示：`params` 不做向量化（所有样本共享同一组参数），`t_norm` 和 `x_norm` 沿第 0 维（batch 维）向量化。最终 `net_residual_batch` 接收 $(N,)$ 形状的 `t_norm` 和 `x_norm`，输出 $(N,)$ 形状的残差向量。

#### 3.2.2 NTK Jacobian 计算

NTK 的计算需要求解网络输出对所有参数的 Jacobian 矩阵。以下代码展示了这一过程：

```python
def compute_jacobian_r(params, t_pts, x_pts):
    flat_params, unravel = ravel_pytree(params)
    def f_flat(fp):
        return net_residual_batch(unravel(fp), t_pts, x_pts)
    return jacrev(f_flat)(flat_params)
```

**数学原理解析**：

1. `ravel_pytree(params)` 将 PyTree 结构的参数 $\theta = \{(W^{(l)}, b^{(l)})\}_{l=1}^{L}$ 展平为一维向量 $\hat{\theta} \in \mathbb{R}^P$（$P = 502,001$），同时返回逆映射 `unravel`
2. 定义标量函数 `f_flat`：$\hat{\theta} \mapsto \mathbf{r}(\hat{\theta}) \in \mathbb{R}^{N_r}$
3. `jacrev(f_flat)(flat_params)` 使用反向模式自动微分计算 Jacobian $J_r = \frac{\partial \mathbf{r}}{\partial \hat{\theta}} \in \mathbb{R}^{N_r \times P}$

**反向模式 vs 前向模式的选择**：由于 $N_r \ll P$（采样点数远少于参数数），反向模式（`jacrev`）的计算复杂度为 $O(N_r)$ 次反向传播，而前向模式（`jacfwd`）需要 $O(P)$ 次前向传播。因此 `jacrev` 更高效。

**NTK 矩阵的组装**：

```python
def compute_ntk_diag_blocks(params, t_bc_n, x_bc_n, t_ic_n, x_ic_n, t_r_n, x_r_n):
    J_u = compute_jacobian_u(params, t_bc_n, x_bc_n)    # (N_u, P)
    J_ut = compute_jacobian_ut(params, t_ic_n, x_ic_n)  # (N_ut, P)
    J_r = compute_jacobian_r(params, t_r_n, x_r_n)      # (N_r, P)
    K_u = J_u @ J_u.T     # (N_u, N_u)
    K_ut = J_ut @ J_ut.T  # (N_ut, N_ut)
    K_r = J_r @ J_r.T     # (N_r, N_r)
    return K_u, K_ut, K_r
```

**维度分析**：

| 变量 | 维度 | 含义 |
|------|------|------|
| `J_u` | $(N_u, P)$ | 边界/IC 输出对参数的 Jacobian |
| `J_ut` | $(N_{ut}, P)$ | 初始速度输出对参数的 Jacobian |
| `J_r` | $(N_r, P)$ | PDE 残差对参数的 Jacobian |
| `K_u` | $(N_u, N_u)$ | 边界/IC 分量的 NTK 子矩阵 |
| `K_ut` | $(N_{ut}, N_{ut})$ | 初始速度分量的 NTK 子矩阵 |
| `K_r` | $(N_r, N_r)$ | PDE 残差分量的 NTK 子矩阵 |

本算例中取 $N_u = N_{ut} = N_r = 300$，$P = 502,001$。

#### 3.2.3 自适应权重更新

以下代码展示了 NTK 自适应权重的计算和更新过程：

```python
trace_K_u = np.trace(K_u_np)
trace_K_ut = np.trace(K_ut_np)
trace_K_r = np.trace(K_r_np)
trace_total = trace_K_u + trace_K_ut + trace_K_r

if trace_K_u > 0 and trace_K_ut > 0 and trace_K_r > 0:
    lam_u = float(trace_total / trace_K_u)
    lam_ut = float(trace_total / trace_K_ut)
    lam_r = float(trace_total / trace_K_r)
```

**数学对应关系**：

$$\lambda_u = \frac{\mathrm{Tr}(K_{uu}) + \mathrm{Tr}(K_{u_tu_t}) + \mathrm{Tr}(K_{rr})}{\mathrm{Tr}(K_{uu})}, \quad \lambda_{u_t} = \frac{\mathrm{Tr}(K)}{\mathrm{Tr}(K_{u_tu_t})}, \quad \lambda_r = \frac{\mathrm{Tr}(K)}{\mathrm{Tr}(K_{rr})}$$

注意 `trace_K_u > 0` 的检查是数值安全措施，防止除零错误。

### 3.3 训练流程与采样策略

#### 3.3.1 采样策略

每步训练中，分别从四个区域进行随机采样：

| 采样区域 | 采样点数 | 坐标范围 | 采样方式 |
|---------|---------|---------|---------|
| 初始条件（IC） | $N/3 = 100$ | $t = 0, \, x \in [0, 1]$ | 均匀随机 |
| 左边界（BC1） | $N/3 = 100$ | $t \in [0, 1], \, x = 0$ | 均匀随机 |
| 右边界（BC2） | $N/3 = 100$ | $t \in [0, 1], \, x = 1$ | 均匀随机 |
| 残差点 | $N = 300$ | $(t, x) \in [0, 1]^2$ | 均匀随机 |

所有采样点在生成时立即进行 z-score 归一化。

#### 3.3.2 优化器配置

本实现使用 Adam 优化器 [4] 配合指数学习率衰减：

$$\eta(n) = \eta_0 \cdot 0.9^{n/1000}$$

| 超参数 | 值 | 说明 |
|--------|---|------|
| 初始学习率 $\eta_0$ | $10^{-3}$ | Adam 默认初始值 |
| 衰减率 | 0.9 | 每 1000 步衰减 10% |
| $\beta_1, \beta_2$ | 0.9, 0.999 | Adam 默认动量参数 |
| 训练轮次 | 80,001 | 包含第 0 轮 |
| NTK 更新频率 | 每 100 步 | NTK 计算的间隔 |

---

## 4. 实验设置与结果

### 4.1 1D Wave 方程问题设定

本教程选取的 benchmark 问题为一维波动方程，其精确解包含两个不同频率的模式，对网络的表达能力提出了较高要求：

$$u(t, x) = \underbrace{\sin(\pi x)\cos(2\pi t)}_{\text{低频模式}} + \underbrace{0.5\sin(4\pi x)\cos(8\pi t)}_{\text{高频模式}}$$

第二个模式的空间频率是第一个的 4 倍，时间频率也是 4 倍，这使得标准 PINN 难以同时捕捉两个模式。

### 4.2 实验配置表

| 配置项 | 设定值 |
|-------|-------|
| PDE | $u_{tt} = 4 u_{xx}$，$(t, x) \in [0, 1]^2$ |
| 网络结构 | `[2, 500, 500, 500, 1]` |
| 激活函数 | $\tanh$ |
| 初始化 | Xavier |
| 总参数量 | 502,001 |
| 优化器 | Adam（$\eta_0 = 10^{-3}$，指数衰减） |
| 输入归一化 | Z-score（$\mu_t \approx 0.500$, $\sigma_t \approx 0.289$） |
| 训练轮次 | 80,001 |
| Batch size | 300 |
| NTK 更新间隔 | 100 步 |
| NTK 采样点数 | 300 |
| 测试网格 | $200 \times 200 = 40,000$ 点 |

### 4.3 训练结果分析

#### 4.3.1 损失曲线与 L2 误差

下图展示了训练过程中各损失分量和 L2 相对误差的变化：

![Loss curves and L2 error](https://pic1.imgdb.cn/item/69b4e7ca963e55431f540487.png)

**图 1**：(a) 各损失分量的训练曲线。可以观察到在 NTK 自适应权重的作用下，PDE 残差损失 $\mathcal{L}_r$（蓝色）与边界条件损失 $\mathcal{L}_u$（橙色）的量级保持在相近范围，避免了一方主导训练的情况。(b) L2 相对误差曲线。误差在约 20,000 步后进入稳定下降阶段，最终达到 $4.806 \times 10^{-3}$。

#### 4.3.2 预测结果可视化

![Prediction comparison](https://pic1.imgdb.cn/item/69b4e7cd963e55431f54048a.png)

**图 2**：(a) 精确解 $u(t, x)$；(b) NTK-PINN 预测解；(c) 逐点绝对误差。预测解能够较好地捕捉波动方程的双频模式结构，最大误差集中在高频模式的峰值区域附近。

### 4.4 NTK 分析

#### 4.4.1 NTK 特征值演化

![NTK eigenvalues](https://pic1.imgdb.cn/item/69b4e7cf963e55431f54048e.png)

**图 3**：三个 NTK 子矩阵（$K_u$, $K_{u_t}$, $K_r$）的特征值在不同训练阶段的分布。可以观察到：

- $K_u$ 的特征值谱在训练过程中保持相对稳定，衰减较为缓慢
- $K_r$ 的特征值谱衰减最快，前几个特征值与尾部特征值之比可达数个量级
- 这种差异正是导致训练不平衡的根本原因，也证实了 NTK 自适应权重的必要性

#### 4.4.2 自适应权重演化

![Adaptive weights](https://pic1.imgdb.cn/item/69b4e7d2963e55431f540490.png)

**图 4**：三个自适应权重 $\lambda_u, \lambda_{u_t}, \lambda_r$ 的演化曲线。可以看到：

- $\lambda_r$ 始终接近 1，表明 PDE 残差的 NTK 迹在总迹中占主导（符合 $K_r$ 采样点最多的设定）
- $\lambda_u, \lambda_{u_t}$ 在训练初期较大（数十倍），用于补偿边界/初始条件的较小 NTK 迹
- 随着训练推进，权重逐渐趋于平稳

### 4.5 归一化 vs 无归一化对比

为验证输入归一化对训练效果的影响，我们同时进行了无归一化版本的实验。以下表格汇总了两种配置在所有算例上的对比结果：

#### 表 1：Wave 1D 算例对比

| 指标 | 无归一化 | 有归一化（z-score） | 变化 |
|------|---------|-----------------|------|
| Best L2 相对误差 | $1.051 \times 10^{-2}$ | $4.806 \times 10^{-3}$ | **降低 54.3%** |
| Final L2 相对误差 | $1.054 \times 10^{-2}$ | $4.820 \times 10^{-3}$ | **降低 54.3%** |
| 训练时间 | 588.7 s | 687.5 s | 增加 16.8% |
| 网络结构 | `[2,500,500,500,1]` | `[2,500,500,500,1]` | 相同 |
| 参数量 | 502,001 | 502,001 | 相同 |
| 优化器 | Adam | Adam | 相同 |

#### 表 2：Poisson 1D 算例对比

| 指标 | 无归一化 | 有归一化（z-score） | 变化 |
|------|---------|-----------------|------|
| Best L2 相对误差 | $6.660 \times 10^{-5}$ | $2.157 \times 10^{-5}$ | **降低 67.6%** |
| 训练时间 | 41.5 s | 45.1 s | 增加 8.7% |
| 网络结构 | `[1,512,1]` | `[1,512,1]` | 相同 |
| 参数量 | 1,025 | 1,025 | 相同 |

#### 表 3：全算例综合对比

| 算例 | 方法 | Best L2 | Final L2 | 时间 (s) | 参数量 |
|------|------|---------|----------|---------|--------|
| Poisson 1D | 无归一化 | 6.660e-05 | — | 41.5 | 1,025 |
| Poisson 1D | **有归一化** | **2.157e-05** | — | 45.1 | 1,025 |
| Wave 1D | 无归一化 | 1.051e-02 | 1.054e-02 | 588.7 | 502,001 |
| Wave 1D | **有归一化** | **4.806e-03** | 4.820e-03 | 687.5 | 502,001 |

**结论**：输入归一化在两个算例中均带来了显著的精度提升（L2 误差降低 54%–68%），代价仅为略微增加的训练时间（主要来自归一化相关的额外计算）。这验证了输入归一化在 PINN 训练中的重要性。

---

## 5. 总结与展望

### 5.1 主要发现

本教程通过 JAX 框架完整复现了 NTK-PINN 算法，并以一维波动方程为例进行了深入分析。主要发现包括：

1. **NTK 自适应权重的有效性**：通过动态调整各损失分量的权重，NTK-PINN 有效缓解了标准 PINN 中的多任务损失不平衡问题，在 Wave 1D 这一具有多频成分的 benchmark 上取得了较好的结果。

2. **输入归一化的显著作用**：z-score 归一化配合梯度链式法则修正，在不改变网络结构和参数量的前提下，将 L2 误差降低了 54%–68%。这一结果表明，简单的数据预处理技巧对 PINN 性能的影响不可忽视。

3. **NTK 特征值谱的诊断价值**：NTK 子矩阵的特征值分布直观反映了各损失分量的"学习难度"，可以作为训练策略设计的理论指导。

### 5.2 局限性

- NTK 的计算复杂度为 $O(N_r \cdot P)$，当参数量和采样点数增大时，计算开销迅速增长。本算例中 NTK 的计算在总训练时间中占据了相当比例。
- NTK 理论的严格成立依赖于无限宽度极限，有限宽度网络上的 NTK 会随训练发生变化（"lazy training" 假设不严格成立）。
- 当前的权重更新策略是"快照式"的（每 $M$ 步更新一次），未能实现完全的连续自适应。

### 5.3 未来方向

| 方向 | 描述 |
|------|------|
| 高效 NTK 近似 | 使用随机迹估计（Hutchinson 估计器）或低秩近似降低 NTK 计算开销 |
| 其他自适应策略 | 对比 NTK 权重与其他方案（如 GradNorm、Causal weighting） |
| 高维问题 | 将 NTK-PINN 推广到二维/三维 PDE 系统 |
| 非线性 PDE | 在 Burgers 方程、Navier-Stokes 方程等非线性问题上验证 NTK 方法的鲁棒性 |
| 理论分析 | 研究有限宽度网络上 NTK 变化对自适应权重策略的影响 |

---

## 附录：常见问题与解决方案

### Q1：NTK 计算时内存不足（OOM）怎么办？

**原因**：Jacobian 矩阵的维度为 $(N, P)$，当 $N = 300$, $P = 502,001$ 时，单精度浮点数下占用约 572 MB。三组 Jacobian 和对应的 NTK 矩阵会进一步增加内存需求。

**解决方案**：
- 减小 NTK 采样点数（`KERNEL_SIZE`）
- 使用 `jax.checkpoint` 进行梯度重计算以换取内存
- 使用随机迹估计替代显式 NTK 计算

### Q2：训练前期 L2 误差不下降甚至上升

**原因**：NTK 自适应权重需要一定的训练步数才能稳定，初期权重的大幅波动可能导致各损失分量交替主导。

**解决方案**：
- 增加训练总步数，允许充分的"预热期"
- 降低学习率以减缓权重振荡
- 对权重更新施加平滑（如指数移动平均）

### Q3：归一化后梯度计算结果不正确

**原因**：忘记在微分后乘以链式法则修正因子 $1/\sigma$（一阶）或 $1/\sigma^2$（二阶）。

**解决方案**：严格遵循第 2.5 节的公式，对每一阶微分操作都乘以对应的修正因子。建议使用独立的解析解测试用例验证微分实现的正确性。

### Q4：`jax.jacrev` 和 `jax.jacfwd` 如何选择？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 输出维度 $< $ 参数维度 | `jacrev` | 反向模式每次计算一行 Jacobian，$O(N)$ 次传播 |
| 参数维度 $< $ 输出维度 | `jacfwd` | 前向模式每次计算一列 Jacobian，$O(P)$ 次传播 |
| PINN 中的 NTK 计算 | `jacrev` | 通常 $N \ll P$（采样点远少于参数） |

### Q5：为什么选择 Adam 而非原论文中的 SGD？

原论文 [1] 的 Poisson 1D 算例使用了 SGD 优化器。本实现统一使用 Adam [4]，主要基于以下考虑：

- Adam 的自适应学习率机制与 NTK 自适应权重可以互补协作
- Adam 在 PINN 文献中的使用更为普遍，便于与其他方法对比
- 对于 Wave 1D 这种多频问题，Adam 的动量机制有助于避免局部最优

---

## 参考文献

[1] Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*, 449, 110768.

[2] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

[3] Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *Advances in Neural Information Processing Systems*, 31.

[4] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*.

[5] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs. *http://github.com/google/jax*.

[6] Xiong, X., Lu, K., Zhang, Z., Zeng, Z., Zhou, S., Hu, R., & Deng, Z. (2025). High-frequency flow field super-resolution via physics-informed hierarchical adaptive Fourier feature networks. *Physics of Fluids*, 37(9).

[7] Xiong, X., Lu, K., Zhang, Z., Zeng, Z., Zhou, S., Deng, Z., & Hu, R. (2025). J-PIKAN: A physics-informed KAN network based on Jacobi orthogonal polynomials for solving fluid dynamics. *Communications in Nonlinear Science and Numerical Simulation*, 109414.

[8] Xiong, X., Zhang, Z., Hu, R., Gao, C., & Deng, Z. (2025). Separated-variable spectral neural networks: A physics-informed learning approach for high-frequency PDEs. *arXiv preprint arXiv:2508.00628*.

[9] Zhang, Z., Xiong, X., Zhang, S., Wang, W., Zhong, Y., Yang, C., & Yang, X. (2025). Legend-KINN: A Legendre Polynomial-Based Kolmogorov-Arnold-Informed Neural Network for Efficient PDE Solving. *Expert Systems with Applications*, 129839.
