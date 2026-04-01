# Physics-Informed Vibe Coding 之 Scale-PINN

![Scale-PINN 封面](https://pic1.imgdb.cn/item/69cca1dd0d45b9ceac3b6d0f.png)

> **系列导读**
>
> 这是 **Physics-Informed Vibe Coding** 系列的第五期实验教程。前四期中，我们先后围绕 NTK 自适应权重、多尺度 Fourier 嵌入、坐标尺度变换和梯度加权策略，对标准 PINN 的训练瓶颈展开了层层递进的改进实验。在这一过程中，我们的"零手写代码"科研协作模式已从一种尝试演变为一种持续运转的工作流：**研究者把控方向与质量，AI 智能体负责从算法实现到数值实验的一切技术细节**。这正是 **Vibe Coding & Vibe Researching** 的精髓——将人类的科学判断力与 AI 的执行效率有机融合，在不触碰键盘的前提下推进完整的科研闭环。五期实验下来，我们对这一范式的可行性与局限性都有了更为务实的认知。
>
> 全部源代码与教程文档开源于 [GitHub: Physics-informed-vibe-coding](https://github.com/xgxgnpu/Physics-informed-vibe-coding)，所有数值实现采用 **JAX** 框架编写。
>
> 本期主题：**Scale-PINN —— 基于演化正则化的序列修正物理信息神经网络**，以二维稳态不可压 Navier-Stokes 方程（方腔驱动流问题，$Re = 7500$）为算例，系统阐述序列修正思想如何克服高雷诺数流动中标准 PINN 的训练失败问题。

---

## 摘要导读

在求解高雷诺数流动问题时，标准 PINN 往往因对流项与扩散项之间悬殊的尺度差异而陷入训练停滞。Scale-PINN 借鉴经典数值方法中的**伪时间步进（pseudo-time stepping）**思想，在 PDE 残差中引入**演化正则项（Evolutionary Regularization）**，将当前解与前一步解的差异作为额外约束，从而将高度非线性的稳态问题转化为一系列更易收敛的子问题。

本教程基于 JAX 框架，在完全相同的网络结构、优化器配置和随机种子下，公平对比了 **Standard PINN (M1)** 与 **Scale-PINN (M2)** 的求解性能。核心实验结果如下：

| 指标 | Standard PINN (M1) | Scale-PINN (M2) | 提升幅度 |
|------|:---:|:---:|:---:|
| 最优 Relative $L_2$ Error | $8.460 \times 10^{-1}$ | $2.751 \times 10^{-2}$ | **96.7%** |
| 最终 Relative $L_2$ Error | $9.458 \times 10^{-1}$ | $2.979 \times 10^{-2}$ | **96.8%** |
| 最终 MSE | $4.350 \times 10^{-2}$ | $4.316 \times 10^{-5}$ | **三个数量级** |
| 训练时间 (s) | 100.4 | 110.0 | +9.6% |
| 参数量 | 59,520 | 59,520 | 相同 |
| 迭代次数 | 50,000 | 50,000 | 相同 |

**核心发现**：在几乎相同的计算开销下，Scale-PINN 将方腔驱动流 $Re = 7500$ 问题的速度场相对误差从 $O(1)$（训练完全失败）降至 $O(10^{-2})$，实现了从"不可解"到"可解"的质变。

---

## 1. 引言

### 1.1 PINN 在高雷诺数流动中的挑战

物理信息神经网络（PINN）[2] 通过将偏微分方程残差嵌入损失函数，实现了无网格、无标注数据的 PDE 求解。然而，当方程的非线性程度显著增强——例如高雷诺数不可压 Navier-Stokes 方程——标准 PINN 的表现急剧恶化。

考虑二维稳态不可压 Navier-Stokes 方程：

$$\nabla \cdot \mathbf{u} = 0$$

$$(\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p - \frac{1}{Re} \nabla^2 \mathbf{u} = 0$$

当 $Re = 7500$ 时，对流项 $(\mathbf{u} \cdot \nabla)\mathbf{u}$ 的量级约为扩散项 $\frac{1}{Re}\nabla^2 \mathbf{u}$ 的 $10^3 \sim 10^4$ 倍。这一巨大的尺度差异导致：

- **梯度信号被对流项主导**：扩散项的梯度贡献被淹没，网络无法学习到正确的黏性效应
- **损失景观高度非凸**：大量局部极小值使标准梯度下降陷入停滞
- **PDE 残差无法有效降低**：相对 $L_2$ 误差停留在 $O(1)$ 量级，表明网络输出几乎是随机猜测

### 1.2 数值迭代修正思想的引入动机

在传统计算流体力学中，直接求解高 $Re$ 稳态 NS 方程同样困难。经典的策略是将稳态问题转化为非稳态问题，通过伪时间步进逐步逼近稳态解：

$$\frac{\partial \mathbf{u}}{\partial \tau} + (\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p - \frac{1}{Re}\nabla^2 \mathbf{u} = 0$$

当 $\tau \to \infty$ 时，$\frac{\partial \mathbf{u}}{\partial \tau} \to 0$，解自然趋向稳态。将时间导数离散化：

$$\frac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\Delta \tau} + \text{(PDE residual of } \mathbf{u}^{n+1}\text{)} = 0$$

这一离散化在 PDE 残差上附加了一个**演化修正项** $\frac{\mathbf{u}^{n+1} - \mathbf{u}^n}{\Delta \tau}$，起到正则化作用：当 $\Delta \tau$ 较小时，系统被强约束在前一步解的邻域内，每步的非线性程度大幅降低。

Scale-PINN [1] 和 Cao & Zhang [5] 分别从不同角度将这一经典思想引入 PINN 框架：**每一步训练不再从零开始求解完整的 PDE，而是在前一步解的基础上进行修正**。Cao & Zhang [5] 从 PDE 系统 Jacobian 矩阵的条件数出发，揭示了 PINN 病态优化的根源，并通过构造"受控系统"调节条件数来缓解病态性，成功模拟了 $Re = 5000$ 的三维 M6 机翼绕流问题。Scale-PINN 则通过演化正则项实现类似的序列修正策略。两者的核心洞察一致：通过伪时间步进将困难的稳态问题分解为可控的增量求解过程。

### 1.3 本教程的组织结构

本教程围绕以下内容展开：

- **第 2 节**：方法与数学原理——Navier-Stokes 方程的 PINN 形式化、标准 PINN 损失函数设计、Scale-PINN 演化正则项的数学推导
- **第 3 节**：网络架构与维度分析——共享主干 + 三分支架构的设计逻辑、Fourier 特征层的维度变换、PINN 与 DNN 双模块的功能划分
- **第 4 节**：核心代码解读——关键算法实现的数学原理对照
- **第 5 节**：实验结果与分析——训练收敛曲线、流场对比、中心线剖面等定量分析
- **第 6 节**：总结与展望

---

## 2. 方法与数学原理

### 2.1 不可压 Navier-Stokes 方程（LDC Re=7500）

**方腔驱动流（Lid-Driven Cavity, LDC）**是计算流体力学中经典的基准问题。在单位正方形域 $\Omega = [0, 1]^2$ 上，二维稳态不可压 Navier-Stokes 方程写为：

**连续方程：**

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$$

**$x$-动量方程：**

$$u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \frac{1}{Re}\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0$$

**$y$-动量方程：**

$$u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \frac{1}{Re}\left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0$$

**边界条件**（四角点排除）：

| 边界 | $u$ | $v$ |
|------|-----|-----|
| 上壁面 ($y = 1$) | 1 | 0 |
| 下壁面 ($y = 0$) | 0 | 0 |
| 左壁面 ($x = 0$) | 0 | 0 |
| 右壁面 ($x = 1$) | 0 | 0 |

当 $Re = 7500$ 时，流场呈现复杂的多涡结构，主涡偏心且存在多个次级涡，对求解方法提出了严格的精度要求。

### 2.2 标准 PINN 损失函数

标准 PINN 用一个参数为 $\theta$ 的神经网络来同时逼近速度场 $(u_\theta, v_\theta)$ 和压力场 $p_\theta$。定义三个 PDE 残差：

$$R_\text{cont}(\theta) = \frac{\partial u_\theta}{\partial x} + \frac{\partial v_\theta}{\partial y}$$

$$R_\text{mom,x}(\theta) = u_\theta \frac{\partial u_\theta}{\partial x} + v_\theta \frac{\partial u_\theta}{\partial y} + \frac{\partial p_\theta}{\partial x} - \frac{1}{Re}\left(\frac{\partial^2 u_\theta}{\partial x^2} + \frac{\partial^2 u_\theta}{\partial y^2}\right)$$

$$R_\text{mom,y}(\theta) = u_\theta \frac{\partial v_\theta}{\partial x} + v_\theta \frac{\partial v_\theta}{\partial y} + \frac{\partial p_\theta}{\partial y} - \frac{1}{Re}\left(\frac{\partial^2 v_\theta}{\partial x^2} + \frac{\partial^2 v_\theta}{\partial y^2}\right)$$

PDE 损失和边界条件损失分别为：

$$\mathcal{L}_\text{PDE}(\theta) = \frac{1}{N_\text{int}} \sum_{i \in \mathcal{S}_\text{int}} \left[ R_\text{cont}^2(\mathbf{x}_i; \theta) + R_\text{mom,x}^2(\mathbf{x}_i; \theta) + R_\text{mom,y}^2(\mathbf{x}_i; \theta) \right]$$

$$\mathcal{L}_\text{BC}(\theta) = \frac{1}{N_\text{bc}} \sum_{i \in \mathcal{S}_\text{bc}} \left[ (u_\theta(\mathbf{x}_i) - u_\text{bc})^2 + (v_\theta(\mathbf{x}_i) - v_\text{bc})^2 \right]$$

总损失为：

$$\mathcal{L}_\text{total}(\theta) = \mathcal{L}_\text{PDE}(\theta) + \lambda_\text{bc} \mathcal{L}_\text{BC}(\theta), \quad \lambda_\text{bc} = 10$$

### 2.3 Scale-PINN 的序列修正原理

#### 2.3.1 演化正则项的数学推导

Scale-PINN 的核心思想是：在 PDE 残差中引入一个**与前一步参数 $\theta_0$ 对应的解的差异项**，作为伪时间步进的正则化。

设当前参数为 $\theta$，前一步参数为 $\theta_0$，则修正后的残差为：

**修正连续方程残差：**

$$\tilde{R}_\text{cont}(\theta, \theta_0) = R_\text{cont}(\theta) + \frac{p_\theta(\mathbf{x}) - p_{\theta_0}(\mathbf{x})}{\text{ER}}$$

**修正 $x$-动量方程残差：**

$$\tilde{R}_\text{mom,x}(\theta, \theta_0) = R_\text{mom,x}(\theta) + \frac{u_\theta(\mathbf{x}) - u_{\theta_0}(\mathbf{x})}{\text{ER}} + \frac{m_{1,\theta}(\mathbf{x}) - m_{1,\theta_0}(\mathbf{x})}{\text{ER}_{xx}}$$

**修正 $y$-动量方程残差：**

$$\tilde{R}_\text{mom,y}(\theta, \theta_0) = R_\text{mom,y}(\theta) + \frac{v_\theta(\mathbf{x}) - v_{\theta_0}(\mathbf{x})}{\text{ER}} + \frac{m_{2,\theta}(\mathbf{x}) - m_{2,\theta_0}(\mathbf{x})}{\text{ER}_{xx}}$$

其中 $m_{1,\theta} = -\frac{1}{Re}(u_{xx} + u_{yy})$，$m_{2,\theta} = -\frac{1}{Re}(v_{xx} + v_{yy})$ 为扩散项。

![Scale-PINN 残差修正示意图](https://pic1.imgdb.cn/item/69cca1dc0d45b9ceac3b6d0a.png)

**图 1**：标准 PINN（M1）与 Scale-PINN（M2）的残差构造对比。M1 直接最小化原始 PDE 残差，M2 在残差中加入基于前一步解的演化修正项，通过 $\theta_0$ 提供的参考信息逐步引导收敛。

#### 2.3.2 参数 ER 与 ER_xx 的物理含义

| 参数 | 符号 | 默认值 | 物理含义 |
|------|------|--------|---------|
| 演化正则化系数 | $\text{ER}$ | 0.095 | 对应伪时间步长 $\Delta \tau$；越小则每步修正越保守，约束越强 |
| 扩散修正系数 | $\text{ER}_{xx}$ | 0.5 | 控制扩散项修正的相对强度；独立调节黏性项的收敛速率 |

两个参数的物理直觉：

- **$\text{ER}$ 较小**：每步只允许解发生微小变化，等价于小时间步长的隐式时间推进，收敛更稳定但需要更多迭代
- **$\text{ER}$ 较大**：每步允许解发生大幅变化，收敛速度更快但可能振荡
- **$\text{ER}_{xx}$** 的引入：由于 $Re = 7500$ 时扩散项量级远小于对流项，不单独调节扩散项的修正强度会导致黏性效应被忽略

#### 2.3.3 与经典数值迭代修正的联系

Scale-PINN 的修正策略与经典 CFD 方法存在明确的对应关系：

| 经典数值方法 | Scale-PINN 对应 |
|-------------|----------------|
| 伪时间步长 $\Delta \tau$ | 演化系数 $\text{ER}$ |
| 前一时间步解 $\mathbf{u}^n$ | 前一步参数 $\theta_0$ 对应的解 $(u_{\theta_0}, v_{\theta_0}, p_{\theta_0})$ |
| 时间离散格式（隐式 Euler） | 每次训练步更新 $\theta_0 \leftarrow \theta$ |
| 残差松弛/欠松弛 | $\text{ER}_{xx}$ 对扩散项的独立调节 |

本质上，Scale-PINN 在 PINN 的连续优化过程中**隐式地实现了一种伪时间步进格式**，将困难的稳态问题转化为一系列近似线性化的子问题。

### 2.4 PINN 与 DNN 双模块设计的数学意义

本实现中定义了两个网络模块：

- **PINN 模块**：输出 10 通道 $[u, v, p, R_\text{cont}, R_\text{mom,x}, R_\text{mom,y}, \text{bc}, \text{nbc}, m_1, m_2]$，包含 PDE 残差和边界掩码，用于构造训练损失
- **DNN 模块**：输出 5 通道 $[u, v, p_\text{out}, m_1, m_2]$，仅包含物理量和扩散项，用于推理阶段的场预测和 $\theta_0$ 参考值计算

两者**共享完全相同的网络结构和参数**（通过 `flatten_util.ravel_pytree` 实现参数的统一管理），区别仅在于 DNN 模块省略了 $\nabla p$、$\nabla u$、$\nabla v$ 等一阶导数的计算（这些在推理时不需要），并额外执行了压力参考点归一化 $p_\text{out} = p - p_\text{ref}$。

这种双模块设计在数学上是等价的——给定相同参数，两模块对 $(u, v)$ 的预测完全一致——但在工程上显著提升了计算效率：DNN 模块省去了大量 Jacobian 计算，推理速度远快于 PINN 模块。

### 2.5 损失函数的完整对比表

| 损失组成 | Standard PINN (M1, $\text{ER}=0$) | Scale-PINN (M2, $\text{ER}=0.095$) |
|---------|-----------------------------------|-------------------------------------|
| 连续方程残差 | $R_\text{cont} = u_x + v_y$ | $\tilde{R}_\text{cont} = R_\text{cont} + \frac{p - p_{\theta_0}}{\text{ER}}$ |
| $x$-动量残差 | $R_\text{mom,x} = uu_x + vu_y + p_x - \frac{1}{Re}(u_{xx} + u_{yy})$ | $\tilde{R}_\text{mom,x} = R_\text{mom,x} + \frac{u - u_{\theta_0}}{\text{ER}} + \frac{m_1 - m_{1,\theta_0}}{\text{ER}_{xx}}$ |
| $y$-动量残差 | $R_\text{mom,y} = uv_x + vv_y + p_y - \frac{1}{Re}(v_{xx} + v_{yy})$ | $\tilde{R}_\text{mom,y} = R_\text{mom,y} + \frac{v - v_{\theta_0}}{\text{ER}} + \frac{m_2 - m_{2,\theta_0}}{\text{ER}_{xx}}$ |
| PDE 损失 | $\mathcal{L}_\text{PDE} = \frac{1}{N}\sum_i (R_\text{cont}^2 + R_\text{mom,x}^2 + R_\text{mom,y}^2)$ | $\mathcal{L}_\text{PDE} = \frac{1}{N}\sum_i (\tilde{R}_\text{cont}^2 + \tilde{R}_\text{mom,x}^2 + \tilde{R}_\text{mom,y}^2)$ |
| BC 损失 | $\mathcal{L}_\text{BC} = \frac{1}{N_\text{bc}}\sum_i [(u - u_\text{bc})^2 + (v - v_\text{bc})^2]$ | 相同 |
| 总损失 | $\mathcal{L} = \mathcal{L}_\text{PDE} + 10 \cdot \mathcal{L}_\text{BC}$ | $\mathcal{L} = \mathcal{L}_\text{PDE} + 10 \cdot \mathcal{L}_\text{BC}$ |
| $\theta_0$ 更新 | 无 | 每步 $\theta_0 \leftarrow \theta$ |

---

## 3. 网络架构与维度分析

### 3.1 共享主干 + 三分支架构

本实现采用共享主干网络提取低维特征，再通过三个独立分支分别预测 $u$、$v$、$p$。这一设计基于两层考量：

1. **共享主干**：速度场和压力场在物理上通过 NS 方程紧密耦合，共享底层特征可以隐式编码这种耦合关系
2. **独立分支**：$u$、$v$、$p$ 的数值范围和变化特征不同（例如 $u \in [-0.5, 1]$，$p$ 的尺度取决于参考点选取），独立分支允许各自适应其输出空间

### 3.2 Fourier 特征层维度变换

网络的第一层执行**Fourier 特征嵌入**，将低维坐标映射到高维特征空间：

$$\mathbf{h}_0 = \sin(2\pi \cdot \mathbf{W}_\text{feat} \cdot \mathbf{x}_\text{aug})$$

其中 $\mathbf{x}_\text{aug} = [x, y, x-1, y-1]^\top \in \mathbb{R}^4$ 为增广坐标（引入 $x-1, y-1$ 增强边界附近的表达能力），$\mathbf{W}_\text{feat} \in \mathbb{R}^{256 \times 4}$ 为可学习的投影矩阵。

正弦变换 $\sin(2\pi \cdot)$ 的作用在于：
- 引入**周期性非线性**，使网络天然具备捕获不同频率分量的能力
- 与随机 Fourier 特征理论 [3] 的联系：可学习频率矩阵 $\mathbf{W}_\text{feat}$ 等价于对频谱的自适应采样

### 3.3 PINN 模块 10 通道输出解析

PINN 模块的 `__call__` 方法通过自动微分计算一阶和二阶导数，最终输出 10 个通道：

| 通道索引 | 符号 | 维度 | 含义 |
|---------|------|------|------|
| 0 | $u$ | $(N, 1)$ | $x$-速度 |
| 1 | $v$ | $(N, 1)$ | $y$-速度 |
| 2 | $p$ | $(N, 1)$ | 压力 |
| 3 | $R_\text{cont}$ | $(N, 1)$ | 连续方程残差 |
| 4 | $R_\text{mom,x}$ | $(N, 1)$ | $x$-动量残差 |
| 5 | $R_\text{mom,y}$ | $(N, 1)$ | $y$-动量残差 |
| 6 | bc | $(N, 1)$ | 边界掩码（布尔） |
| 7 | nbc | $(N, 1)$ | 内部掩码（布尔） |
| 8 | $m_1$ | $(N, 1)$ | 扩散项 $-\frac{1}{Re}(u_{xx} + u_{yy})$ |
| 9 | $m_2$ | $(N, 1)$ | 扩散项 $-\frac{1}{Re}(v_{xx} + v_{yy})$ |

### 3.4 DNN 模块 5 通道输出解析

DNN 模块省去一阶导数计算，输出 5 个通道：

| 通道索引 | 符号 | 维度 | 含义 |
|---------|------|------|------|
| 0 | $u$ | $(N, 1)$ | $x$-速度 |
| 1 | $v$ | $(N, 1)$ | $y$-速度 |
| 2 | $p_\text{out}$ | $(N, 1)$ | 参考点归一化后的压力 |
| 3 | $m_1$ | $(N, 1)$ | 扩散项 |
| 4 | $m_2$ | $(N, 1)$ | 扩散项 |

### 3.5 完整维度流表

下表详细列出数据在网络中的维度变换过程（$N$ 为样本数，$n = 64$ 为隐层宽度）：

| 阶段 | 操作 | 输入维度 | 输出维度 | 参数量 |
|------|------|---------|---------|--------|
| 输入增广 | $[x, y] \to [x, y, x{-}1, y{-}1]$ | $(N, 2)$ | $(N, 4)$ | 0 |
| Fourier 特征 | $\text{Dense}(4 \to 256)$ | $(N, 4)$ | $(N, 256)$ | $4 \times 256 + 256 = 1280$ |
| 正弦激活 | $\sin(2\pi \cdot)$ | $(N, 256)$ | $(N, 256)$ | 0 |
| 主干层 1 | $\text{Dense}(256 \to 64) + \text{SiLU}$ | $(N, 256)$ | $(N, 64)$ | $256 \times 64 + 64 = 16448$ |
| 主干层 2 | $\text{Dense}(64 \to 64) + \text{SiLU}$ | $(N, 64)$ | $(N, 64)$ | $64 \times 64 + 64 = 4160$ |
| 分支入口（$\times 3$） | $\text{Dense}(64 \to 64)$ | $(N, 64)$ | $(N, 64)$ | $3 \times (64 \times 64 + 64) = 12480$ |
| 分支隐层 1（$\times 3$） | $\text{SiLU} + \text{Dense}(64 \to 64) + \text{SiLU}$ | $(N, 64)$ | $(N, 64)$ | $3 \times 4160 = 12480$ |
| 分支隐层 2（$\times 3$） | $\text{Dense}(64 \to 64) + \text{SiLU}$ | $(N, 64)$ | $(N, 64)$ | $3 \times 4160 = 12480$ |
| 分支输出（$\times 3$） | $\text{Dense}(64 \to 1)$（无偏置） | $(N, 64)$ | $(N, 1)$ | $3 \times 64 = 192$ |
| **合计** | | | | **59,520** |

![网络架构与维度流转图](https://pic1.imgdb.cn/item/69cca1dc0d45b9ceac3b6d09.png)

**图 2**：Scale-PINN 网络架构与维度流转示意图。输入坐标经增广、Fourier 特征映射和共享主干后，分别进入 $u$、$v$、$p$ 三个独立分支，最终输出标量预测值。圆圈中的数字标注了各阶段的特征维度。

---

## 4. 核心代码解读

### 4.1 Fourier 特征嵌入

网络的第一步是将二维坐标映射到高维 Fourier 特征空间：

```python
def get_uvp(x, y):
    inp = jnp.hstack([x, y, x - 1.0, y - 1.0])
    hidden = self.feats(inp)
    hidden = jnp.sin(2 * jnp.pi * hidden)
    for lyr in self.layers:
        hidden = lyr(hidden)
    # ... branches for u, v, p
```

**数学原理**：输入 $\mathbf{x}_\text{aug} = [x, y, x{-}1, y{-}1]^\top \in \mathbb{R}^4$ 经过线性层 `self.feats` 映射到 $\mathbb{R}^{256}$，再施加 $\sin(2\pi \cdot)$ 非线性。增广项 $x - 1$ 和 $y - 1$ 打破了网络输入关于 $(0.5, 0.5)$ 的对称性，帮助网络更好地区分对称位置上不同的流场行为。这等价于在 Fourier 空间中同时对 $x$ 和 $1 - x$ 方向进行频率采样。

### 4.2 自动微分求导链

PDE 残差的计算依赖于对网络输出的精确微分。JAX 的前向模式自动微分 `jacfwd` 用于计算一阶和二阶空间导数：

```python
def get_uvp_xy(get_uvp, x, y):
    u_x, v_x, p_x = jacfwd(get_uvp)(x, y)
    u_y, v_y, p_y = jacfwd(get_uvp, argnums=1)(x, y)
    return u_x, u_y, v_x, v_y, p_x, p_y

f_xy_vmap = vmap(get_uvp_xy, in_axes=(None, 0, 0))
u_x, u_y, v_x, v_y, p_x, p_y = f_xy_vmap(get_uvp, x, y)
```

**数学原理**：`jacfwd(get_uvp)(x, y)` 计算函数 `get_uvp` 关于第一个参数 $x$ 的 Jacobian，即 $\frac{\partial}{\partial x}(u, v, p)$。通过 `argnums=1` 切换到对 $y$ 求导。`vmap` 将逐点计算向量化为批量操作。

对于二阶导数，使用嵌套的 `jacfwd`：

```python
def get_uvp_xxyy(get_uvp, x, y):
    u_xx, v_xx, p_xx = jacfwd(jacfwd(get_uvp))(x, y)
    u_yy, v_yy, p_yy = jacfwd(jacfwd(get_uvp, argnums=1), argnums=1)(x, y)
    return u_xx, u_yy, v_xx, v_yy, p_xx, p_yy
```

**数学原理**：`jacfwd(jacfwd(get_uvp))(x, y)` 等价于 $\frac{\partial^2}{\partial x^2}(u, v, p)$。这是前向模式自动微分的链式应用：外层 `jacfwd` 对内层 `jacfwd` 的结果再次关于 $x$ 求导。

维度变化分析：
- `get_uvp(x, y)` 返回 $(u, v, p)$，各为 $(1, 1)$
- `jacfwd(get_uvp)(x, y)` 返回 $\frac{\partial (u, v, p)}{\partial x}$，各为 $(1, 1)$
- `jacfwd(jacfwd(get_uvp))(x, y)` 返回 $\frac{\partial^2 (u, v, p)}{\partial x^2}$，各为 $(1, 1, 1)$

因此在代码中需要通过 `u_xx[:, :, 0, 0]` 提取标量值。

### 4.3 演化正则项的实现

`eval_loss` 函数中，Scale-PINN 的核心逻辑通过 `ER > 0` 的条件分支实现：

```python
def eval_loss(params, params_0, inputs, labels):
    pred = model.apply(unravel_fn(params), inputs)
    u, v, p, res_cont, res_mom1, res_mom2, bc, nbc, m_1, m_2 = \
        jnp.split(pred, 10, axis=1)

    pred0 = model_0.apply(unravel_fn(params_0), inputs)
    u_0, v_0, p_0, m0_1, m0_2 = jnp.split(pred0, 5, axis=1)

    if ER > 0:
        res_cont = res_cont + (p - p_0) / ER
        res_mom1 = res_mom1 + (u - u_0) / ER + (m_1 - m0_1) / ER_xx
        res_mom2 = res_mom2 + (v - v_0) / ER + (m_2 - m0_2) / ER_xx

    pde_uvp = (jnp.square(res_cont) +
               jnp.square(res_mom1) +
               jnp.square(res_mom2))
    pde_loss = jnp.sum(pde_uvp * nbc) / nbc.sum()
```

**数学原理**：

1. `model.apply(unravel_fn(params), inputs)` 通过 PINN 模块计算当前参数 $\theta$ 下的 10 通道输出
2. `model_0.apply(unravel_fn(params_0), inputs)` 通过 DNN 模块计算前一步参数 $\theta_0$ 下的 5 通道参考输出
3. 当 $\text{ER} > 0$ 时，原始残差被修正为：
   - $\tilde{R}_\text{cont} = R_\text{cont} + (p - p_{\theta_0}) / \text{ER}$
   - $\tilde{R}_\text{mom,x} = R_\text{mom,x} + (u - u_{\theta_0}) / \text{ER} + (m_1 - m_{1,\theta_0}) / \text{ER}_{xx}$
4. PDE 损失仅在内部点（`nbc`）上计算，通过掩码 `nbc` 排除边界点的贡献

这段代码清晰地体现了 Scale-PINN 与 Standard PINN 的唯一区别：**是否在残差中加入演化修正项**。当 `ER = 0` 时（M1），修正分支不执行，退化为标准 PINN。

### 4.4 参数扁平化与双模块共享

JAX 中通过 `flatten_util.ravel_pytree` 实现参数的统一管理：

```python
model = PINN(N_NODES)
model_0 = DNN(N_NODES)
params_tree = model.init(key, a)
params, unravel_fn = flatten_util.ravel_pytree(params_tree)
params_0 = params
```

**数学原理**：`ravel_pytree` 将嵌套的参数树结构展平为一维向量 $\theta \in \mathbb{R}^{59520}$，`unravel_fn` 负责逆变换。由于 PINN 和 DNN 模块具有完全相同的网络结构（层数、宽度、初始化方式），展平后的参数向量可以无歧义地传递给任一模块的 `apply` 方法。

`params_0 = params` 的初始赋值使得**第一步训练的演化修正项为零**（$u_\theta = u_{\theta_0}$），等价于标准 PINN 的一步。

### 4.5 训练循环中 params_0 更新逻辑

在训练循环的 `update` 函数中，$\theta_0$ 的更新发生在参数更新之前：

```python
@jit
def update(params, params_0, opt_state, key):
    batch_X, batch_Y = minibatch(key)
    (loss, (mse, rl2, pde_loss, bc_loss)), grad = \
        loss_grad(params, params_0, batch_X, batch_Y)
    updates, opt_state = optimizer.update(grad, opt_state)
    params_0 = params          # 保存当前参数作为下一步的参考
    params = optax.apply_updates(params, updates)  # 梯度更新
    return params, params_0, opt_state, loss, mse, rl2, pde_loss, bc_loss
```

**数学原理**：更新顺序为 $\theta_0^{(t+1)} \leftarrow \theta^{(t)}$，$\theta^{(t+1)} \leftarrow \theta^{(t)} - \alpha \nabla_\theta \mathcal{L}(\theta^{(t)}, \theta_0^{(t)})$。即**先保存当前参数为参考，再执行梯度下降**。这意味着在第 $t+1$ 步计算损失时，修正项 $(u_\theta - u_{\theta_0})$ 衡量的是**相邻两步之间的解的变化量**，与伪时间步进的离散格式完全对应。

---

## 5. 实验结果与分析

### 5.1 实验配置对比表

两种方法在完全相同的实验配置下进行训练，唯一差异是 ER 参数：

| 配置项 | Standard PINN (M1) | Scale-PINN (M2) |
|--------|:---:|:---:|
| 网络架构 | 共享主干 + 3 分支 | 相同 |
| 隐层宽度 $n$ | 64 | 相同 |
| 激活函数 | SiLU | 相同 |
| Fourier 特征维度 | 256 | 相同 |
| 参数量 | 59,520 | 相同 |
| 优化器 | Adam | 相同 |
| 学习率调度 | Cosine Decay, $\text{lr}_\text{max} = 5 \times 10^{-4}$, exponent=1.2 | 相同 |
| 总迭代次数 | 50,000 | 相同 |
| 内部点批量 $N_\text{int}$ | 950 | 相同 |
| 边界点批量 $N_\text{bc}$ | 50 | 相同 |
| BC 权重 $\lambda_\text{bc}$ | 10 | 相同 |
| 随机种子 | 50 | 相同 |
| 演化系数 ER | **0** | **0.095** |
| 扩散修正系数 ER_xx | — | **0.5** |

### 5.2 训练收敛曲线分析

下图展示了 M1 和 M2 在训练过程中 PDE 损失、BC 损失以及总损失的变化趋势。

![M1 vs M2 收敛行为对比](https://pic1.imgdb.cn/item/69cca1df0d45b9ceac3b6d19.png)

**图 3**：Standard PINN (M1) 与 Scale-PINN (M2) 的收敛行为对比示意图。M1 在整个训练过程中相对 $L_2$ 误差停留在 $\sim 0.85$，完全未能收敛；M2 经历初始下降、快速收敛和精细调优三个阶段，最终达到 $\sim 0.027$，实现 96.7% 的精度提升。

以下为实际训练过程中记录的损失曲线和 Relative $L_2$ Error 收敛曲线：

![损失曲线对比：PDE Loss 与 BC Loss](https://pic1.imgdb.cn/item/69cca4c50d45b9ceac3b7bac.png)

**图 4**：训练损失曲线对比。(a) PDE 损失与 BC 损失的分项对比；(b) 总损失的对比。Scale-PINN 的 PDE 损失持续下降，而 Standard PINN 的 PDE 损失在约 5000 步后即停止下降。

![Relative L2 Error 收敛曲线](https://pic1.imgdb.cn/item/69cca4c60d45b9ceac3b7bb2.png)

**图 5**：速度场 Relative $L_2$ Error 收敛曲线。Standard PINN (M1) 的误差始终在 $0.8 \sim 1.0$ 之间波动，Scale-PINN (M2) 在约 40k 步后收敛至 $\sim 0.027$。

**关键观察**：

1. **M1 的训练停滞**：标准 PINN 的 PDE 损失在约 5000 步后即停止下降，Relative $L_2$ Error 始终在 $0.8 \sim 1.0$ 之间波动。这表明网络陷入了损失景观中的平坦区域，梯度信号不足以驱动进一步优化。

2. **M2 的三阶段收敛**：
   - **Phase 1（初始下降，0–10k 步）**：演化修正项的约束使网络迅速从随机初始化向合理解域移动
   - **Phase 2（快速收敛，10k–40k 步）**：随着参考解 $\theta_0$ 逐步改善，修正项引导网络持续逼近真实解
   - **Phase 3（精细调优，40k–50k 步）**：学习率衰减与接近收敛的残差使优化进入精细搜索阶段

### 5.3 流场预测对比分析

流场预测结果的可视化直观展示了两种方法的求解质量差异。

**速度场幅值**：M1 的预测场几乎均匀，未能捕获主涡和二次涡结构；M2 的预测与参考解高度吻合，清晰呈现了主涡的偏心特征和角涡的存在。

![Standard PINN vs Scale-PINN 速度场幅值对比](https://pic1.imgdb.cn/item/69cca4c80d45b9ceac3b7bbe.png)

**图 6**：速度场幅值 $|\mathbf{u}|$ 的对比。上排为 Standard PINN (M1)，下排为 Scale-PINN (M2)；左列为参考解，中列为预测解，右列为绝对误差。M1 的误差场几乎覆盖整个计算域，M2 的误差集中在壁面附近的薄层区域。

![Standard PINN (M1) 流场分量 u/v/p](https://pic1.imgdb.cn/item/69cca4cd0d45b9ceac3b7bd6.png)

**图 7**：Standard PINN (M1) 的 $u$、$v$、$p$ 三个分量的参考解、预测解与绝对误差。M1 在三个物理量上均表现出大范围的高误差，表明网络未学到有效的流场结构。

![Scale-PINN (M2) 流场分量 u/v/p](https://pic1.imgdb.cn/item/69cca4cd0d45b9ceac3b7bd8.png)

**图 8**：Scale-PINN (M2) 的 $u$、$v$、$p$ 三个分量的参考解、预测解与绝对误差。M2 在所有分量上均与参考解高度一致，误差量级远低于 M1。

**压力场**：M1 的压力预测缺乏物理意义的空间分布；M2 正确预测了驱动壁面附近的高压区和涡心处的低压区。

### 5.4 中心线速度剖面分析

中心线速度剖面是 LDC 问题中最重要的定量验证手段 [4]：

- **$u(y)$ 剖面**（$x = x_\text{mid}$ 处）：反映主涡的垂直结构
- **$v(x)$ 剖面**（$y = y_\text{mid}$ 处）：反映主涡的水平结构

![中心线速度剖面对比](https://pic1.imgdb.cn/item/69cca4ce0d45b9ceac3b7bdb.png)

**图 9**：中心线速度剖面对比。(a) 垂直中心线 $u(y)$；(b) 水平中心线 $v(x)$。黑色实线为参考解，蓝色虚线为 Standard PINN (M1)，红色点划线为 Scale-PINN (M2)。M2 的剖面与参考解几乎完全重合，而 M1 的剖面偏离严重，进一步确认了 Standard PINN 在高 $Re$ 条件下的失效。

### 5.5 定量指标汇总表

| 指标 | Standard PINN (M1) | Scale-PINN (M2) | M2 相对 M1 的提升 |
|------|:---:|:---:|:---:|
| Best Relative $L_2$ (velocity) | $8.460 \times 10^{-1}$ | $2.751 \times 10^{-2}$ | $\downarrow$ 96.7% |
| Final Relative $L_2$ (velocity) | $9.458 \times 10^{-1}$ | $2.979 \times 10^{-2}$ | $\downarrow$ 96.8% |
| Final MSE (velocity) | $4.350 \times 10^{-2}$ | $4.316 \times 10^{-5}$ | $\downarrow$ 3 个数量级 |
| 训练时间 | 100.4 s | 110.0 s | $\uparrow$ 9.6% |
| 时间-精度效率 | $\text{RL2}/\text{s} = 8.4 \times 10^{-3}$ | $\text{RL2}/\text{s} = 2.5 \times 10^{-4}$ | $\uparrow$ 33.6× |

### 5.6 训练效率分析

| 效率指标 | Standard PINN (M1) | Scale-PINN (M2) |
|---------|:---:|:---:|
| 每步平均耗时 | $\sim 2.0$ ms | $\sim 2.2$ ms |
| 额外计算开销 | — | +10%（DNN 模块前向推理） |
| 内存开销 | 1× | $\sim$ 1.05×（存储 params_0） |
| 达到 RL2 < 0.1 所需迭代 | 未达到 | $\sim 15{,}000$ |
| 达到 RL2 < 0.05 所需迭代 | 未达到 | $\sim 35{,}000$ |

Scale-PINN 的额外计算开销几乎可以忽略——DNN 模块的前向推理仅涉及无梯度计算的正向传播和少量二阶导数求取，与 PINN 模块的完整 Jacobian 计算相比微不足道。

---

## 6. 总结与展望

### 6.1 核心结论

本教程通过方腔驱动流 $Re = 7500$ 基准问题的公平对比实验，得到以下核心结论：

1. **标准 PINN 在高雷诺数条件下完全失效**：相对 $L_2$ 误差停留在 $O(1)$ 量级，网络输出几乎无物理意义
2. **Scale-PINN 通过演化正则化实现了从"不可解"到"可解"的质变**：同等条件下将精度提升近两个数量级
3. **方法的本质是将困难的稳态问题转化为一系列简单的子问题**：通过隐式的伪时间步进格式逐步逼近稳态解
4. **计算代价增量极小**：仅增加约 10% 的训练时间和极少的内存占用

### 6.2 局限性讨论

Scale-PINN 并非万能方案，需要注意以下局限：

- **超参数敏感性**：$\text{ER}$ 和 $\text{ER}_{xx}$ 的选取依赖经验，过大会导致修正不足（退化为标准 PINN），过小会导致收敛速度极慢
- **单一稳态假设**：演化正则化假设系统存在唯一稳态解，对于存在多稳态的系统可能收敛到非物理解
- **时间相关问题**：当前实现针对稳态方程设计，推广到非稳态问题需要额外的时间离散策略
- **精度上限**：最终精度（$\text{RL2} \sim 3\%$）距离高精度数值解仍有差距，可能需要与其他改进策略（如自适应采样、域分解）结合使用

### 6.3 未来方向

基于本教程的实验结果，以下研究方向值得进一步探索：

1. **自适应 ER 策略**：根据训练过程中的残差变化动态调整 $\text{ER}$，初期使用较小值保证稳定性，后期逐步增大以加速收敛
2. **与 NTK 权重的结合**：在 Scale-PINN 的框架上叠加 NTK 自适应权重，同时解决损失不平衡和非线性刚性两个问题
3. **多级修正策略**：将单一的 $\theta_0$ 参考扩展为多步历史参考，借鉴多步法（如 BDF 格式）进一步提升稳定性
4. **三维和非稳态推广**：将 Scale-PINN 扩展到三维湍流问题和时空求解框架中
5. **自动超参数搜索**：利用贝叶斯优化或元学习方法自动确定 $\text{ER}$ 和 $\text{ER}_{xx}$ 的最优值

---

## 附录：常见问题与解决方案

| 问题 | 原因分析 | 解决方案 |
|------|---------|---------|
| M1 的 RL2 始终在 0.8–1.0 | 高 $Re$ 下对流项主导，标准 PINN 无法学习正确的流场结构 | 使用 Scale-PINN（$\text{ER} > 0$）或降低 $Re$ |
| M2 收敛后 RL2 出现回弹 | 学习率衰减不够快或 $\text{ER}$ 设置过大 | 减小 $\text{ER}$、增大余弦衰减指数 `exponent`、或使用学习率重启 |
| 训练初期 loss 出现 NaN | Fourier 特征层输出幅值过大，导致数值溢出 | 降低学习率初值或对 Fourier 层输出施加梯度裁剪 |
| BC 损失远大于 PDE 损失 | $\lambda_\text{bc}$ 设置过大或边界采样不均匀 | 减小 $\lambda_\text{bc}$，检查边界点的采样密度分布 |
| 压力场预测出现常数偏移 | 不可压 NS 方程中压力仅定义到常数差 | 使用参考点归一化 $p_\text{out} = p - p(x_\text{ref}, y_\text{ref})$ |
| 四角点导致 BC 损失异常 | 方腔四角处速度不连续（物理奇点） | 在数据预处理中排除四角点（本代码已实现） |
| DNN 模块与 PINN 模块预测不一致 | 参数传递方式不正确 | 确保两模块使用相同的 `unravel_fn` 反序列化参数 |
| 内存不足 (OOM) | 批量过大或二阶导数计算占用大量内存 | 减小 `BS_ALL`，或使用 `jax.checkpoint` 降低内存峰值 |

---

## 参考文献

[1] Chiu, P.-H., Wong, J. C., Ooi, C. C., Wei, C., Fan, Y., & Ong, Y.-S. (2026). Scale-PINN: Learning Efficient Physics-Informed Neural Networks Through Sequential Correction. *arXiv:2602.19475*.

[2] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.

[3] Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*, 449, 110768.

[4] Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411.

[5] Cao, W., & Zhang, W. (2025). An analysis and solution of ill-conditioning in physics-informed neural networks. *Journal of Computational Physics*, 520, 113494.

[6] Bradbury, J. et al. (2018). JAX: Composable transformations of Python+NumPy programs. [http://github.com/jax-ml/jax](http://github.com/jax-ml/jax).
