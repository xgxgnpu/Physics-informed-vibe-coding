# Physics-Informed Vibe Coding 之尺度变换-PINN

![VS-PINN 封面](https://pic1.imgdb.cn/item/69ba4709b96fa53fd04be6af.png)

> **系列导读**
>
> 欢迎来到 **Physics-Informed Vibe Coding** 系列的第三期实验教程。回顾前两期，我们分别从 NTK 自适应权重与多尺度 Fourier 特征嵌入的角度，探讨了标准 PINN 在训练动态与频谱表达方面的改进路径。在这个过程中，一个始终贯穿的方法论内核逐渐清晰：**科研的本质在于提出正确的问题和做出关键的判断，而非机械地敲击键盘**。本系列正是对这一信念的实践检验——我们坚持 **Vibe Coding & Vibe Researching** 的协作范式，研究者专注于算法构思、实验方案的制定与结果的严格审查，AI 智能体则全权负责从代码实现到数值调试的全链路执行。至今，我们仍未手动编写过任何一行代码，却已完成了多个完整的 PINN 变体算法实验。
>
> 全部代码与教程文档持续更新并开源于 [GitHub: Physics-informed-vibe-coding](https://github.com/xgxgnpu/Physics-informed-vibe-coding)，所有数值实现一律采用 **JAX** 框架编写，充分利用其即时编译（JIT）与自动微分（AD）能力实现高效 GPU 加速训练。
>
> 本期主题：**VS-PINN —— 基于变量尺度变换的物理信息神经网络**，以二维稳态不可压 Navier-Stokes 方程（圆柱绕流问题，$Re=20$）为算例，系统阐述坐标尺度变换如何改善 PINN 对刚性 PDE 的训练效率与求解精度。

---

## 摘要导读

物理信息神经网络（PINN）在求解偏微分方程时，常因方程的**刚性特征**（stiff behavior）——即不同物理项之间存在数个量级的尺度差异——而遭遇训练困难。具体表现为：对流项与黏性项的量级悬殊导致梯度信号失衡，网络难以同时精确逼近所有物理约束。

Ko 与 Park [1] 提出的**变量尺度 PINN（Variable-Scaling PINN, VS-PINN）**方法，通过对输入坐标施加简单的线性缩放 $\tilde{x} = Nx$，将原始紧凑的计算域拉伸至更大范围。经链式法则推导，该变换等价于在 PDE 残差中为高阶导数项引入一个幂次放大因子 $N^k$，从而有效缩小不同物理项之间的尺度差距，改善训练过程中的梯度平衡性。

本教程以 2D 稳态不可压 Navier-Stokes 方程的圆柱绕流问题为算例（$\rho=1$, $\mu=0.02$, $Re=20$），采用缩放因子 $N=10$，在 80000 个 Adam 优化迭代后，达到如下 $L^2$ 相对误差：

| 物理量 | $u$（$x$-速度） | $v$（$y$-速度） | $p$（压力） |
|--------|:---:|:---:|:---:|
| $L^2$ 相对误差 | 2.10% | 5.06% | 4.45% |

---

## 1. 引言

### 1.1 PINN 基本思想

PINN [2] 的核心理念是将偏微分方程的物理约束直接嵌入神经网络的损失函数。对于一般性的 PDE 边值问题：

$$\mathcal{N}[u](\mathbf{x}) = 0, \quad \mathbf{x} \in \Omega$$

$$\mathcal{B}[u](\mathbf{x}) = 0, \quad \mathbf{x} \in \partial\Omega$$

PINN 使用一个参数为 $\theta$ 的神经网络 $u_\theta(\mathbf{x})$ 逼近真解 $u(\mathbf{x})$，通过最小化复合损失函数进行训练：

$$\mathcal{L}(\theta) = \lambda_r \mathcal{L}_r(\theta) + \lambda_b \mathcal{L}_b(\theta)$$

其中 $\mathcal{L}_r$ 为 PDE 残差的均方误差，$\mathcal{L}_b$ 为边界/初始条件损失。PDE 残差中的各阶导数通过自动微分精确计算，无需有限差分近似，也无需网格剖分。

### 1.2 刚性 PDE 与多尺度问题的挑战

当 PDE 的不同物理项之间存在显著的量级差异时，我们称该方程具有**刚性特征**（stiffness）。以 Navier-Stokes 方程为例，对流项 $\rho(\mathbf{u} \cdot \nabla)\mathbf{u}$ 与黏性扩散项 $\mu \nabla^2 \mathbf{u}$ 的比值由 Reynolds 数 $Re = \rho U L / \mu$ 决定。即使在中等 $Re$ 条件下（如本算例 $Re \approx 20$），两者之间仍存在量级差异，导致：

- 损失函数中不同残差分量的梯度尺度严重不匹配
- 优化器倾向于优先最小化主导项的残差，忽略量级较小的物理约束
- 训练后期出现残差震荡甚至发散现象

此外，当计算域本身尺寸较小（例如本算例的物理域仅为 $[0, 1.1] \times [0, 0.41]$）时，神经网络在输入空间受限的条件下难以充分展开其表征能力，进一步加剧了训练困难。

### 1.3 尺度变换（Variable Scaling）的动机

VS-PINN [1] 的核心洞察是：通过简单的坐标缩放来改变 PDE 残差中各项的相对大小，使得网络在一个"更均衡"的数值环境中进行学习。相比于其他解决刚性问题的方法（如 NTK 自适应权重 [3]、损失函数预处理、学习率调度等），尺度变换具有以下独特优势：

| 特性 | 标准 PINN | NTK 自适应权重 | 损失函数预处理 | **VS-PINN（尺度变换）** |
|------|:---------:|:------------:|:----------:|:------------------:|
| 实现复杂度 | 低 | 高（需计算 NTK 矩阵） | 中 | **极低（仅需乘 N）** |
| 额外计算开销 | 无 | 大（Jacobian 计算） | 中 | **几乎为零** |
| 超参数数量 | 少 | 多（更新频率等） | 中 | **仅 1 个（N）** |
| 对刚性问题的改善 | 差 | 好 | 一般 | **好** |
| 与其他方法的兼容性 | — | — | — | **高（可叠加使用）** |

---

## 2. 方法与数学原理

### 2.1 不可压 Navier-Stokes 方程

本算例求解的是二维稳态不可压 Navier-Stokes 方程，其控制方程由连续性方程和动量守恒方程组成：

**连续性方程**（质量守恒）：

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**$x$-方向动量方程**：

$$\rho\left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}\right) + \frac{\partial p}{\partial x} - \mu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$

**$y$-方向动量方程**：

$$\rho\left(u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y}\right) + \frac{\partial p}{\partial y} - \mu\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0$$

其中 $\rho = 1.0$ 为流体密度，$\mu = 0.02$ 为动力黏度，$u, v$ 分别为 $x, y$ 方向的速度分量，$p$ 为压力。

### 2.2 VS-PINN 尺度变换的数学推导

VS-PINN 的核心操作是定义一个**正整数缩放因子** $N$（本算例取 $N = 10$），将物理坐标 $(x, y)$ 映射到缩放坐标 $(\tilde{x}, \tilde{y})$：

$$\tilde{x} = N \cdot x, \quad \tilde{y} = N \cdot y$$

![VS-PINN 尺度变换原理图](https://pic1.imgdb.cn/item/69ba4709b96fa53fd04be6b0.png)
*图 1：VS-PINN 坐标变换原理——物理域 $(x, y)$ 经缩放因子 $N$ 映射到扩展域 $(\tilde{x}, \tilde{y})$*

神经网络的输入为缩放坐标 $(\tilde{x}, \tilde{y})$，输出为物理量 $(u, v, p)$。由于自动微分计算的是 $u, v, p$ 关于 $\tilde{x}, \tilde{y}$ 的导数，我们需要通过**链式法则**将其转换回物理坐标下的导数。

**一阶导数变换**：

$$\frac{\partial u}{\partial x} = \frac{\partial u}{\partial \tilde{x}} \cdot \frac{\partial \tilde{x}}{\partial x} = N \cdot \frac{\partial u}{\partial \tilde{x}}$$

$$\frac{\partial u}{\partial y} = N \cdot \frac{\partial u}{\partial \tilde{y}}$$

**二阶导数变换**：

$$\frac{\partial^2 u}{\partial x^2} = \frac{\partial}{\partial x}\left(N \cdot \frac{\partial u}{\partial \tilde{x}}\right) = N \cdot \frac{\partial^2 u}{\partial \tilde{x}^2} \cdot \frac{\partial \tilde{x}}{\partial x} = N^2 \cdot \frac{\partial^2 u}{\partial \tilde{x}^2}$$

$$\frac{\partial^2 u}{\partial y^2} = N^2 \cdot \frac{\partial^2 u}{\partial \tilde{y}^2}$$

将上述关系代入原始 NS 方程，得到**缩放坐标下的动量方程**：

$$\rho\left(u \cdot N\frac{\partial u}{\partial \tilde{x}} + v \cdot N\frac{\partial u}{\partial \tilde{y}}\right) + N\frac{\partial p}{\partial \tilde{x}} - \mu\left(N^2\frac{\partial^2 u}{\partial \tilde{x}^2} + N^2\frac{\partial^2 u}{\partial \tilde{y}^2}\right) = 0$$

在代码实现中，对上式除以 $N$ 进行归一化，得到最终的残差形式：

$$r_1 = \rho\left(u \frac{\partial u}{\partial \tilde{x}} + v \frac{\partial u}{\partial \tilde{y}}\right) + \frac{\partial p}{\partial \tilde{x}} - \mu N \left(\frac{\partial^2 u}{\partial \tilde{x}^2} + \frac{\partial^2 u}{\partial \tilde{y}^2}\right) = 0$$

$$r_2 = \rho\left(u \frac{\partial v}{\partial \tilde{x}} + v \frac{\partial v}{\partial \tilde{y}}\right) + \frac{\partial p}{\partial \tilde{y}} - \mu N \left(\frac{\partial^2 v}{\partial \tilde{x}^2} + \frac{\partial^2 v}{\partial \tilde{y}^2}\right) = 0$$

**连续性方程残差**（除以 $N$ 后简化）：

$$r_3 = \frac{\partial u}{\partial \tilde{x}} + \frac{\partial v}{\partial \tilde{y}} = 0$$

**关键观察**：归一化后的残差 $r_1, r_2$ 中，黏性项的系数变为 $\mu N$（原始为 $\mu$），**尺度变换等价于将有效黏性系数从 $\mu = 0.02$ 放大到 $\mu N = 0.2$**，从而大幅缩小对流项与黏性项之间的量级差距。

下表直观展示了这一变化：

| 物理项 | 原始坐标系数 | 缩放坐标系数（除以 $N$ 后） | 放大倍数 |
|--------|:----------:|:---------------------:|:------:|
| 对流项 $\rho(u \cdot u_x + v \cdot u_y)$ | $\rho = 1$ | $\rho = 1$ | $1\times$ |
| 压力梯度 $p_x$ | $1$ | $1$ | $1\times$ |
| 黏性项 $\mu(u_{xx} + u_{yy})$ | $\mu = 0.02$ | $\mu N = 0.2$ | $10\times$ |

### 2.3 边界条件在缩放域中的处理

物理域 $[0, 1.1] \times [0, 0.41]$ 经缩放后变为 $[0, 11] \times [0, 4.1]$。边界条件保持物理量不变，但施加位置按 $N$ 倍缩放：

| 边界类型 | 物理域位置 | 缩放域位置 | 条件 |
|----------|:--------:|:--------:|------|
| 入口（Inlet） | $x = 0$ | $\tilde{x} = 0$ | $u = \frac{4y(0.41 - y)}{0.41^2}$，$v = 0$ |
| 出口（Outlet） | $x = 1.1$ | $\tilde{x} = 11$ | $p = 0$ |
| 上/下壁面（Walls） | $y = 0, 0.41$ | $\tilde{y} = 0, 4.1$ | $u = v = 0$（无滑移） |
| 圆柱表面 | 圆心 $(0.2, 0.2)$, $r = 0.05$ | 圆心 $(2, 2)$, $r = 0.5$ | $u = v = 0$ |

注意入口速度剖面的表达式中，$y$ 需先转换回物理坐标 $y_{\text{phys}} = \tilde{y}/N$ 后代入抛物线公式：

$$u_{\text{inlet}}(\tilde{y}) = \frac{4 \cdot (\tilde{y}/N) \cdot (0.41 - \tilde{y}/N)}{0.41^2}$$

### 2.4 损失函数设计

总损失函数由 PDE 残差项和边界条件项构成：

$$\mathcal{L} = \underbrace{\frac{1}{N_c}\sum_{i=1}^{N_c}\left(r_1^2 + r_2^2 + r_3^2\right)_i}_{\mathcal{L}_{\text{PDE}}} + \underbrace{\lambda_{\text{bc}}\left(\mathcal{L}_{\text{inlet}} + \mathcal{L}_{\text{wall}} + \mathcal{L}_{\text{cylinder}} + \mathcal{L}_{\text{outlet}}\right)}_{\mathcal{L}_{\text{BC}}}$$

其中 $\lambda_{\text{bc}} = 2.0$ 为边界条件权重系数。各损失分量的定义如下：

| 损失分量 | 数学表达式 | 物理含义 |
|----------|----------|---------|
| $\mathcal{L}_{r_1}$ | $\frac{1}{N_c}\sum r_1^2$ | $x$-动量方程残差 |
| $\mathcal{L}_{r_2}$ | $\frac{1}{N_c}\sum r_2^2$ | $y$-动量方程残差 |
| $\mathcal{L}_{r_3}$ | $\frac{1}{N_c}\sum r_3^2$ | 连续性方程残差 |
| $\mathcal{L}_{\text{inlet}}$ | $\lambda_{\text{bc}} \cdot \frac{1}{N_b}\sum[(u - u_{\text{in}})^2 + v^2]$ | 入口速度偏差 |
| $\mathcal{L}_{\text{wall+cyl}}$ | $\lambda_{\text{bc}} \cdot \frac{1}{N_w}\sum(u^2 + v^2)$ | 无滑移条件偏差 |
| $\mathcal{L}_{\text{outlet}}$ | $\lambda_{\text{bc}} \cdot \frac{1}{N_o}\sum p^2$ | 出口压力偏差 |

---

## 3. 网络结构与维度分析

### 3.1 MLP 架构

本算例采用标准全连接前馈网络（MLP），结构为 $[2, 40, 40, 40, 40, 40, 3]$：

![网络结构与维度流](https://pic1.imgdb.cn/item/69ba470ab96fa53fd04be6b2.png)
*图 2：VS-PINN 网络结构与数据维度变换流程*

网络的完整数据流如下：

1. **物理坐标输入** $(x, y) \in \mathbb{R}^{B \times 2}$
2. **坐标缩放** $(\tilde{x}, \tilde{y}) = (Nx, Ny) \in \mathbb{R}^{B \times 2}$
3. **输入层 → 隐藏层 1**：$\mathbf{h}_1 = \tanh(\tilde{\mathbf{x}} \cdot W_1 + b_1)$
4. **逐层传播**：$\mathbf{h}_{l+1} = \tanh(\mathbf{h}_l \cdot W_{l+1} + b_{l+1})$
5. **输出层**：$[u, v, p] = \mathbf{h}_5 \cdot W_6 + b_6$

**层级维度变换表**：

| 层 | 输入维度 | 权重矩阵 $W$ | 偏置 $b$ | 输出维度 | 激活函数 | 参数量 |
|----|:-------:|:----------:|:------:|:-------:|:------:|:-----:|
| Input → Hidden 1 | $(B, 2)$ | $(2, 40)$ | $(40,)$ | $(B, 40)$ | tanh | 120 |
| Hidden 1 → Hidden 2 | $(B, 40)$ | $(40, 40)$ | $(40,)$ | $(B, 40)$ | tanh | 1,640 |
| Hidden 2 → Hidden 3 | $(B, 40)$ | $(40, 40)$ | $(40,)$ | $(B, 40)$ | tanh | 1,640 |
| Hidden 3 → Hidden 4 | $(B, 40)$ | $(40, 40)$ | $(40,)$ | $(B, 40)$ | tanh | 1,640 |
| Hidden 4 → Hidden 5 | $(B, 40)$ | $(40, 40)$ | $(40,)$ | $(B, 40)$ | tanh | 1,640 |
| Hidden 5 → Output | $(B, 40)$ | $(40, 3)$ | $(3,)$ | $(B, 3)$ | — | 123 |
| **总计** | | | | | | **6,803** |

### 3.2 Xavier 初始化分析

网络参数采用 Xavier（Glorot 均匀）初始化策略。对于第 $l$ 层，权重从均匀分布 $\mathcal{U}(-\alpha, \alpha)$ 中采样：

$$\alpha = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$

| 层 | $n_{\text{in}}$ | $n_{\text{out}}$ | 初始化范围 $\alpha$ |
|----|:---:|:---:|:---:|
| Input → Hidden 1 | 2 | 40 | $\sqrt{6/42} \approx 0.378$ |
| Hidden → Hidden | 40 | 40 | $\sqrt{6/80} \approx 0.274$ |
| Hidden 5 → Output | 40 | 3 | $\sqrt{6/43} \approx 0.373$ |

Xavier 初始化确保了信号在前向传播和反向传播过程中的方差保持稳定，避免了深层网络中的梯度消失/爆炸问题。

### 3.3 自动微分中的链式法则与缩放因子

在 JAX 的自动微分框架中，一个需要特别关注的维度关系是：当我们对网络输出 $u(\tilde{x}, \tilde{y})$ 关于输入 $\tilde{x}$ 求导时，`jax.grad` 直接计算的是 $\partial u / \partial \tilde{x}$。要获得物理坐标下的导数，需手动乘以缩放因子。

| 量 | JAX 自动微分直接计算 | 物理含义（物理坐标） | 关系 |
|----|:-----------------:|:----------------:|:----:|
| $u_{\tilde{x}}$ | $\partial u / \partial \tilde{x}$ | $\partial u / \partial x = N \cdot u_{\tilde{x}}$ | $\times N$ |
| $u_{\tilde{x}\tilde{x}}$ | $\partial^2 u / \partial \tilde{x}^2$ | $\partial^2 u / \partial x^2 = N^2 \cdot u_{\tilde{x}\tilde{x}}$ | $\times N^2$ |
| $p_{\tilde{x}}$ | $\partial p / \partial \tilde{x}$ | $\partial p / \partial x = N \cdot p_{\tilde{x}}$ | $\times N$ |

---

## 4. 核心代码解读

### 4.1 NS 残差计算

`ns_residual_single` 是整个 VS-PINN 的数学核心函数，它接收网络参数和单个配点坐标 $(\tilde{x}, \tilde{y})$，返回三个 PDE 残差分量。以下是关键代码段：

```python
def ns_residual_single(params, x, y):
    u = net_u(params, x, y)
    v = net_v(params, x, y)

    u_x = jax.grad(net_u, argnums=1)(params, x, y)
    u_y = jax.grad(net_u, argnums=2)(params, x, y)
    v_x = jax.grad(net_v, argnums=1)(params, x, y)
    v_y = jax.grad(net_v, argnums=2)(params, x, y)

    u_xx = jax.grad(lambda p, xx, yy: jax.grad(net_u, 1)(p, xx, yy), 1)(params, x, y)
    u_yy = jax.grad(lambda p, xx, yy: jax.grad(net_u, 2)(p, xx, yy), 2)(params, x, y)
    v_xx = jax.grad(lambda p, xx, yy: jax.grad(net_v, 1)(p, xx, yy), 1)(params, x, y)
    v_yy = jax.grad(lambda p, xx, yy: jax.grad(net_v, 2)(p, xx, yy), 2)(params, x, y)

    p_x = jax.grad(net_p, argnums=1)(params, x, y)
    p_y = jax.grad(net_p, argnums=2)(params, x, y)

    N = N_VS  # 缩放因子
    r1 = (RHO * (u * N * u_x + v * N * u_y) + N * p_x
          - MU * (N * N * u_xx + N * N * u_yy)) / N
    r2 = (RHO * (u * N * v_x + v * N * v_y) + N * p_y
          - MU * (N * N * v_xx + N * N * v_yy)) / N
    r3 = (N * u_x + N * v_y) / N

    return r1, r2, r3
```

**数学对应关系解析**：

- `u_x` 是 JAX 计算的 $\partial u / \partial \tilde{x}$，物理导数 $\partial u / \partial x = N \cdot \texttt{u\_x}$
- 代码中 `u * N * u_x` 对应 $u \cdot (N \cdot u_{\tilde{x}}) = u \cdot \partial u / \partial x$（对流项）
- 代码中 `N * N * u_xx` 对应 $N^2 \cdot u_{\tilde{x}\tilde{x}} = \partial^2 u / \partial x^2$（黏性项）
- 最后整体除以 $N$ 进行归一化，得到 $r_1 = \rho(u \cdot u_{\tilde{x}} + v \cdot u_{\tilde{y}}) + p_{\tilde{x}} - \mu N(u_{\tilde{x}\tilde{x}} + u_{\tilde{y}\tilde{y}})$

二阶导数通过**嵌套 `jax.grad`** 实现：外层 `jax.grad` 对内层已经求过一次导的函数再次求导，实现了 $\partial^2 u / \partial \tilde{x}^2$ 的精确计算。

### 4.2 分块残差计算策略

由于 NS 方程涉及 3 个输出分量的一阶和二阶导数计算（共需 10 次 `jax.grad` 调用），对大批量配点直接 `vmap` 会导致显著的内存峰值。代码采用 `jax.lax.scan` 进行**分块计算**：

```python
CHUNK_SIZE = 2000

def ns_residual_chunked(params, x_arr, y_arr):
    n_chunks = x_arr.shape[0] // CHUNK_SIZE
    x_ch = x_arr.reshape(n_chunks, CHUNK_SIZE)
    y_ch = y_arr.reshape(n_chunks, CHUNK_SIZE)

    def body(carry, xy):
        r = ns_residual_vmap(params, xy[0], xy[1])
        return carry, r
    _, (r1_all, r2_all, r3_all) = jax.lax.scan(body, None, (x_ch, y_ch))
    return r1_all.reshape(-1), r2_all.reshape(-1), r3_all.reshape(-1)
```

| 策略 | 内存占用 | 计算效率 | JIT 友好性 |
|------|:------:|:------:|:---------:|
| 全量 `vmap` | 高（$O(N_c \times P)$） | 最高 | 好 |
| Python `for` 循环 | 低 | 最低（无法 JIT） | 差 |
| **`jax.lax.scan` 分块** | **可控（$O(\text{chunk} \times P)$）** | **高** | **好** |

`jax.lax.scan` 是 JAX 提供的函数式循环原语，其优势在于：

1. **内存效率**：每次仅处理 `CHUNK_SIZE = 2000` 个配点，峰值内存可控
2. **JIT 兼容**：整个分块循环可被 JIT 编译为高效 XLA 计算图
3. **梯度穿透**：`scan` 的反向传播自动展开，无需手动实现

### 4.3 边界条件采样策略

![配点与边界条件采样](https://pic1.imgdb.cn/item/69ba470bb96fa53fd04be6b5.png)
*图 3：2D 圆柱绕流计算域配点分布与边界条件设置*

边界条件采样的关键设计如下：

```python
def generate_boundary_data():
    # 入口：N_B=200 点，等距分布于 x̃=0 的左边界
    inlet_xy = np.linspace([XMIN, YMIN], [XMIN, YMAX], N_B)
    inlet_u = func_u_inlet(inlet_xy[:, 1])  # 抛物线速度剖面

    # 出口：N_B=200 点，x̃=XMAX 的右边界
    outlet_xy = np.linspace([XMAX, YMIN], [XMAX, YMAX], N_B)

    # 上下壁面：各 N_W=400 点
    wallup_xy = np.linspace([XMIN, YMAX], [XMAX, YMAX], N_W)
    walldn_xy = np.linspace([XMIN, YMIN], [XMAX, YMIN], N_W)

    # 圆柱表面：N_S=200 点，沿圆周均匀分布
    theta = np.linspace(0.0, 2 * np.pi, N_S)
    cyld_x = R_CYL * np.cos(theta) + XC
    cyld_y = R_CYL * np.sin(theta) + YC
    ...
```

各类型采样点的配置汇总：

| 采样区域 | 采样点数 | 空间分布 | 边界条件类型 |
|----------|:------:|---------|------------|
| 入口 | $N_B = 200$ | $\tilde{x} = 0$ 上等距 | Dirichlet：$u = u_{\text{in}}(\tilde{y})$, $v = 0$ |
| 出口 | $N_B = 200$ | $\tilde{x} = 11$ 上等距 | Dirichlet：$p = 0$ |
| 上壁面 | $N_W = 400$ | $\tilde{y} = 4.1$ 上等距 | 无滑移：$u = v = 0$ |
| 下壁面 | $N_W = 400$ | $\tilde{y} = 0$ 上等距 | 无滑移：$u = v = 0$ |
| 圆柱表面 | $N_S = 200$ | 圆周等距 | 无滑移：$u = v = 0$ |
| 内部（均匀） | $N_C = 8{,}000$ | 随机均匀（排除圆柱） | PDE 残差约束 |
| 内部（加密） | $N_R = 600$ | 圆柱周围 $2R$ 范围加密 | PDE 残差约束 |

内部配点在**每个训练迭代**中重新采样（resample），避免网络对固定配点位置产生过拟合。

### 4.4 损失函数与训练循环

```python
def loss_fn(params, xy_col, bnd_xy, bnd_uv, outlet_xy, outlet_p_ref):
    x_col, y_col = xy_col[:, 0], xy_col[:, 1]
    r1, r2, r3 = ns_residual_chunked(params, x_col, y_col)
    mse_r1 = jnp.mean(r1 ** 2)
    mse_r2 = jnp.mean(r2 ** 2)
    mse_r3 = jnp.mean(r3 ** 2)

    u_bnd, v_bnd, _ = net_uvp_batch(params, bnd_xy[:, 0], bnd_xy[:, 1])
    mse_bnd_u = jnp.mean((u_bnd - bnd_uv[:, 0]) ** 2)
    mse_bnd_v = jnp.mean((v_bnd - bnd_uv[:, 1]) ** 2)

    _, _, p_out = net_uvp_batch(params, outlet_xy[:, 0], outlet_xy[:, 1])
    mse_outlet = jnp.mean((p_out - outlet_p_ref) ** 2)

    loss_pde = mse_r1 + mse_r2 + mse_r3
    loss_bc = BC_WEIGHT * (mse_bnd_u + mse_bnd_v) + BC_WEIGHT * mse_outlet
    total = loss_pde + loss_bc
    return total, (mse_r1, mse_r2, mse_r3, mse_bnd_u, mse_bnd_v, mse_outlet)
```

损失函数返回一个元组 `(total_loss, auxiliary_data)`，配合 `jax.value_and_grad(..., has_aux=True)` 实现"一次前向传播同时获取损失值和梯度"的高效计算模式。

---

## 5. 实验设置与结果分析

### 5.1 问题设定

本算例求解经典的 Schäfer-Turek 圆柱绕流基准问题 [3]，具体参数如下：

| 参数 | 符号 | 取值 |
|------|:----:|:----:|
| 流体密度 | $\rho$ | 1.0 |
| 动力黏度 | $\mu$ | 0.02 |
| 通道长度 | $L$ | 1.1 m |
| 通道高度 | $H$ | 0.41 m |
| 圆柱中心 | $(x_c, y_c)$ | $(0.2, 0.2)$ m |
| 圆柱半径 | $R$ | 0.05 m |
| 最大入口速度 | $U_{\max}$ | $4 \times 0.2 \times 0.21 / 0.41^2 \approx 1.0$ m/s |
| 参考 Reynolds 数 | $Re$ | $\rho U_{\max} (2R) / \mu = 1 \times 1 \times 0.1 / 0.02 \approx 5$（基于直径） |
| 参考解来源 | — | Fluent CFD 求解 |

**训练超参数配置**：

| 超参数 | 符号 | 取值 | 说明 |
|--------|:----:|:----:|------|
| 缩放因子 | $N$ | 10 | 坐标放大倍数 |
| 网络结构 | — | $[2, 40, 40, 40, 40, 40, 3]$ | 5 隐藏层，每层 40 神经元 |
| 总参数量 | — | 6,803 | 含权重和偏置 |
| 激活函数 | — | tanh | 全部隐藏层 |
| 初始化 | — | Xavier 均匀 | $\mathcal{U}(-\alpha, \alpha)$ |
| 优化器 | — | Adam [4] | $\beta_1=0.9, \beta_2=0.999$ |
| 学习率 | $\eta$ | $1 \times 10^{-3}$ | 固定 |
| 训练迭代 | $N_{\text{epoch}}$ | 80,000 | — |
| BC 权重 | $\lambda_{\text{bc}}$ | 2.0 | 边界条件损失乘子 |
| 分块大小 | — | 2,000 | `jax.lax.scan` 单块 |
| 配点总数 | — | 8,600 + 边界 | 每迭代重采样 |
| 随机种子 | — | 1234 | 确保可重复性 |

### 5.2 损失曲线分析

![损失分量曲线](https://pic1.imgdb.cn/item/69ba470bb96fa53fd04be6b6.png)
*图 4：VS-PINN 训练过程中各损失分量的变化曲线（对数坐标）*

从损失曲线中可以观察到以下关键现象：

1. **边界条件损失（BC loss）快速下降**：在前 1,000 个迭代内从 $O(1)$ 量级降至 $O(10^{-2})$，表明网络优先学习了边界约束
2. **连续性残差（Continuity）的"平台期"**：约在 10,000—15,000 迭代区间，连续性损失出现一个明显的平台（$\sim 10^{-2}$），随后突破并继续下降至 $O(10^{-5})$
3. **动量方程残差的同步下降**：$x$-动量和 $y$-动量残差在整个训练过程中保持相近的数量级，说明尺度变换有效平衡了两个方向的梯度信号
4. **最终损失层级**：$\mathcal{L}_{\text{BC}} \sim 10^{-4}$，$\mathcal{L}_{\text{cont}} \sim 10^{-5}$，$\mathcal{L}_{\text{mom}} \sim 10^{-5}$

### 5.3 $L^2$ 误差收敛分析

![L2 误差历史](https://pic1.imgdb.cn/item/69ba470cb96fa53fd04be6b7.png)
*图 5：训练过程中 $u, v, p$ 三个物理量的 $L^2$ 相对误差变化*

$L^2$ 相对误差定义为：

$$L^2(q) = \frac{\sqrt{\sum_{i=1}^{N_{\text{ref}}}(q_{\text{pred},i} - q_{\text{ref},i})^2}}{\sqrt{\sum_{i=1}^{N_{\text{ref}}} q_{\text{ref},i}^2}}$$

**训练各阶段 $L^2$ 误差与损失值**：

| 迭代 (Epoch) | 总损失 | 连续性 $\mathcal{L}_{r_3}$ | $x$-动量 $\mathcal{L}_{r_1}$ | $y$-动量 $\mathcal{L}_{r_2}$ | BC 损失 | $L^2(u)$ | $L^2(v)$ | $L^2(p)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 1.55e+0 | 2.30e-2 | 1.08e-2 | 3.85e-3 | 1.51e+0 | 67.4% | 160.8% | 87.6% |
| 1,000 | 2.13e-2 | 1.36e-2 | 8.10e-4 | 3.20e-4 | 6.58e-3 | 94.6% | 90.4% | 103.7% |
| 10,000 | 1.50e-2 | 9.25e-3 | 1.24e-3 | 4.98e-4 | 4.05e-3 | 85.1% | 71.9% | 96.2% |
| 20,000 | 4.75e-4 | 1.08e-4 | 1.31e-4 | 8.30e-5 | 1.53e-4 | 6.4% | 6.5% | 10.9% |
| 40,000 | 3.31e-4 | 2.47e-5 | 3.79e-5 | 2.38e-5 | 2.45e-4 | 3.0% | 4.9% | 6.2% |
| 60,000 | 2.69e-4 | 2.01e-5 | 6.21e-5 | 3.59e-5 | 1.51e-4 | 2.4% | 4.8% | 4.6% |
| 80,000 | 1.71e-4 | 1.32e-5 | 2.42e-5 | 1.72e-5 | 1.16e-4 | **2.1%** | **5.1%** | **4.4%** |

**关键观察**：

1. **"假收敛"阶段**（epoch 1—10,000）：虽然总损失从 1.55 降到 0.015，但 $L^2$ 误差始终保持在 70%—100% 的高水平。这是因为网络在此阶段主要拟合边界条件，内部流场解几乎未被学习。
2. **相变突破**（epoch 10,000—20,000）：$L^2$ 误差在约 10,000 个迭代时间窗口内从 $\sim 90\%$ 骤降至 $\sim 7\%$，对应于 PDE 残差的连续性分量突破平台期。
3. **精细收敛阶段**（epoch 20,000—80,000）：误差缓慢但稳定地下降，最终 $L^2(u) = 2.1\%$。

### 5.4 流场预测结果

以下三组图分别展示了 $x$-速度 $u$、$y$-速度 $v$ 和压力 $p$ 的参考解（Fluent）、VS-PINN 预测以及绝对误差分布。

**$x$-速度场 $u$**：

![u 场对比](https://pic1.imgdb.cn/item/69ba470cb96fa53fd04be6b8.png)
*图 6：$x$-速度 $u$ 的参考解、VS-PINN 预测与绝对误差分布*

**$y$-速度场 $v$**：

![v 场对比](https://pic1.imgdb.cn/item/69ba470cb96fa53fd04be6b9.png)
*图 7：$y$-速度 $v$ 的参考解、VS-PINN 预测与绝对误差分布*

**压力场 $p$**：

![p 场对比](https://pic1.imgdb.cn/item/69ba470db96fa53fd04be6ba.png)
*图 8：压力 $p$ 的参考解、VS-PINN 预测与绝对误差分布*

**各物理量预测精度对比**：

| 物理量 | $L^2$ 相对误差 | 最优 $L^2$ | 达到最优的迭代 | 误差主要集中区域 |
|--------|:---:|:---:|:---:|:---:|
| $u$（$x$-速度） | 2.10% | 1.78% | 68,100 | 圆柱尾流区 |
| $v$（$y$-速度） | 5.06% | 3.75% | 78,300 | 圆柱两侧分离区 |
| $p$（压力） | 4.45% | 3.59% | 77,400 | 圆柱前驻点附近 |

从误差分布图可以观察到：

- $u$ 场的预测精度最高，误差主要集中在圆柱正后方的低速尾流区
- $v$ 场由于其值域较小（$|v| \ll |u|$），相对误差较高，误差峰值出现在圆柱侧面的速度分离点附近
- $p$ 场的误差主要出现在圆柱前方的驻点区域，该区域存在较大的压力梯度

### 5.5 尺度参数 $N$ 的效果分析

缩放因子 $N$ 是 VS-PINN 唯一的核心超参数。其选择需要平衡以下因素：

| $N$ 取值 | 物理域扩展 | 有效黏性 $\mu N$ | 对流-扩散比 | 预期效果 |
|:--------:|:---------:|:---------------:|:-----------:|:--------:|
| 1（无缩放） | $[0, 1.1] \times [0, 0.41]$ | 0.02 | 高 | 刚性严重，收敛困难 |
| 5 | $[0, 5.5] \times [0, 2.05]$ | 0.10 | 中 | 部分改善 |
| **10** | **$[0, 11] \times [0, 4.1]$** | **0.20** | **低** | **较好平衡** |
| 50 | $[0, 55] \times [0, 20.5]$ | 1.00 | 极低 | 可能过度平滑 |

$N$ 过大时，黏性项在残差中占主导，可能导致网络过度关注扩散效应而忽略对流特征；$N$ 过小则无法有效缓解刚性问题。文献 [1] 建议通过少量预实验选择使各残差分量量级接近的 $N$ 值。

---

## 6. 总结与展望

### 6.1 主要发现

本教程通过 2D 稳态不可压 Navier-Stokes 圆柱绕流算例，系统验证了 VS-PINN 尺度变换方法的有效性：

1. **方法简洁高效**：仅通过对输入坐标乘以缩放因子 $N$，无需修改网络结构或增加额外模块，即可显著改善 PINN 对含刚性特征 PDE 的训练效果
2. **数学机理明确**：尺度变换通过链式法则在 PDE 残差中引入 $N^k$ 幂次因子，等价于调整不同微分阶次项的相对权重
3. **训练表现**：在 80,000 个 Adam 迭代后，$u, v, p$ 三个物理量的 $L^2$ 相对误差分别达到 2.10%、5.06% 和 4.45%
4. **计算开销极低**：相比标准 PINN，VS-PINN 的额外计算量仅为坐标缩放的乘法操作和残差归一化的除法操作，可忽略不计

### 6.2 局限性分析

| 局限性 | 具体表现 | 可能原因 |
|--------|---------|---------|
| $v$ 场精度偏低 | $L^2(v) = 5.06\%$，约为 $L^2(u)$ 的 2.4 倍 | $v$ 的值域远小于 $u$，相对误差被放大 |
| 存在"假收敛"阶段 | 前 10,000 次迭代损失下降但 $L^2$ 无改善 | 边界条件先被学习，PDE 约束延后收敛 |
| $N$ 需手动调节 | 缺乏自适应选择策略 | 最优 $N$ 依赖于具体问题的刚性程度 |
| 固定学习率 | 后期收敛变慢 | 未使用学习率调度策略 |

### 6.3 未来方向

| 改进方向 | 具体思路 | 预期提升 |
|----------|---------|---------|
| 自适应 $N$ 选择 | 基于 NTK 特征值谱或残差分量比值自动调节 $N$ | 消除超参数调节负担 |
| 与 NTK 权重结合 | 在尺度变换基础上叠加自适应损失权重 | 进一步平衡训练动态 |
| L-BFGS 二阶优化 | Adam 预训练 + L-BFGS 精细优化 | 提高后期收敛精度 |
| 多尺度 Fourier 嵌入 | 结合 Fourier 特征映射增强高频表达 | 改善尾流区涡结构捕捉 |
| 非稳态扩展 | 推广至含时间维的非稳态 NS 方程 | 拓宽方法适用范围 |
| 三维复杂几何 | 应用于三维圆柱/球体绕流等问题 | 验证方法的可扩展性 |

---

## 附录：常见问题与解决方案

| # | 问题 | 可能原因 | 解决方案 |
|---|------|---------|---------|
| 1 | 损失下降但 $L^2$ 误差很高 | 网络先拟合 BC，PDE 约束延后 | 耐心等待（通常需 15,000+ 迭代突破平台期） |
| 2 | 训练中出现 NaN | 梯度爆炸或学习率过大 | 降低学习率至 $10^{-4}$；检查 $N$ 是否过大 |
| 3 | 圆柱附近误差极大 | 配点密度不足 | 增大 $N_R$（加密区点数）；缩小加密区半径 |
| 4 | 内存不足（OOM） | 配点总数 × 参数量超出显存 | 减小 `CHUNK_SIZE`；减少 $N_C$ |
| 5 | 训练速度慢 | JIT 首次编译耗时 | 第一个迭代的编译开销是正常的；确保 XLA 标志正确设置 |
| 6 | $v$ 场精度始终偏低 | $v$ 值域小，相对误差敏感 | 尝试为 $v$ 的 PDE 残差增加单独的权重 |
| 7 | 入口速度剖面不匹配 | 缩放域中 $y$ 未正确反变换 | 确认 `func_u_inlet` 中 `y/N_VS` 的正确性 |
| 8 | JAX 随机数结果不一致 | PRNGKey 管理不当 | 使用固定 `SEED` 并正确 `split` |
| 9 | 如何选择 $N$ | 无经验公式 | 从 $N=5$ 开始试验，观察各残差分量的量级是否接近 |
| 10 | 可否用于非稳态问题 | 时间维度也需缩放 | 可以，对 $t$ 和空间坐标分别设定缩放因子 |

---

## 参考文献

[1] Ko, S., & Park, S. (2025). VS-PINN: A fast and efficient training of physics-informed neural networks using variable-scaling methods for solving PDEs with stiff behavior. *Journal of Computational Physics*, 529, 113860.

[2] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

[3] Schäfer, M., & Turek, S. (1996). Benchmark computations of laminar flow around a cylinder. In E. H. Hirschel (Ed.), *Flow Simulation with High-Performance Computers II* (Notes on Numerical Fluid Mechanics, Vol. 52, pp. 547–566). Vieweg+Teubner Verlag.

[4] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*.

[5] Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., Necula, G., Paszke, A., VanderPlas, J., Wanderman-Milne, S., & Zhang, Q. (2018). JAX: Composable transformations of Python+NumPy programs. http://github.com/jax-ml/jax
