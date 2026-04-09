---
title: "SonicMoE"
tags:
  - MoE
  - GPU优化
  - 训练加速
  - 显存优化
  - CUDA内核
  - GEMM加速
---

## 🧠 核心摘要 (TL;DR)
> SonicMoE 针对细粒度 Mixture-of-Experts 训练中算术强度低、激活显存随 granularity 爆炸、Padding 浪费严重三大痛点，提出 IO 感知激活重算 + Tile 感知 GPU Kernel（覆盖 Hopper/Blackwell 架构）+ Token Rounding 三组优化，在 64×H100 上实现 2130 亿 token/天的训练吞吐，激活显存降低 45%，计算吞吐提升 1.86×。

## 🏷️ 元数据
- **论文标题**: SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
- **发表机构/会议**: arXiv（UC Berkeley / Together AI / Tri Dao 团队）
- **年份**: 2025（提交 2025-12-16，修订 2026-03-26）
- **链接**: [[SonicMoE_paper|📄论文]] | 💻 Code: 论文声明已开源，具体仓库地址待确认

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #moe #gpu-kernel #training #memory-efficiency #cuda
- **前置知识 (Prerequisites)**: [[ZipServ]], [[ShiftParallelism]]
- **相关节点 (Related Notes)**: [[MSched]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈
细粒度 MoE（即 expert 中间维度 $d_{ff}$ 较小、expert 数量 $E$ 极多）是 DeepSeek-V3、Kimi K2、Qwen3 等顶级模型的主流架构。但在训练侧，它面临三个严峻挑战：

1. **算术强度（Arithmetic Intensity，AI）随 granularity 升高而急剧下降**。对于 MoE 的 GEMM 来说，每个 expert 处理的 token 数 $T_e = TK/E$（其中 $K$ 为 top-K，$E$ 为 expert 数），随着 $E$ 增大，每个 GEMM 的批尺寸缩小，导致算术强度按 $\propto 1/E$ 下降——计算从 compute-bound 退化为 memory-bound。

2. **激活显存随 granularity 线性爆炸**。在标准 checkpointing 策略下，MoE 层每层需要缓存多个中间激活：$X$（输入）、$H$（gate 后隐层）、$S$（SwiGLU 激活值）、$A$（SwiGLU 后激活）等，总量与 expert 数成正比，在 120B 参数规模下每层可达数 GiB。

3. **Padding 浪费严重**。GEMM tile 通常是 $M_{tile} \times K_{tile}$ 的整倍数，而当 token-choice 路由导致某些 expert 仅收到少量 token 时（尤其在高稀疏场景 $K/E \le 1/32$），每个 expert 的 GEMM 批维度 $M$ 无法对齐 tile，造成大量计算浪费。

### 论文的切入点
SonicMoE 从 **IO 感知（IO-aware）** 和 **Tile 感知（Tile-aware）** 两个维度切入：
- 通过重新推导 MoE 反向传播中激活缓存的必要性，设计内存节约型算法；
- 通过在 Hopper（H100）和 Blackwell（B300）架构上定制 CUDA Kernel，将 Scatter/Gather 操作与 GEMM 流水线融合；
- 通过 Token Rounding 路由算法消除 padding 浪费。

### 相关工作

#### 先驱/奠基工作
| 论文 | 核心贡献 | 备注 |
|------|---------|------|
| Shazeer et al. (ICLR 2017) — *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer* | 提出 top-K Sparse MoE 的原型架构，引入 gating 机制 | MoE 奠基论文 |
| Lepikhin et al. (2021) — *GShard* | 将 MoE 扩展到 600B 参数，引入 expert parallelism | 超大规模 MoE 先驱 |
| Fedus et al. (2022) — *Switch Transformers* | 简化为 top-1 routing，证明稀疏 MoE 可以在不降质量的情况下大幅降低每 token 计算量 | Google 经典工作 |
| Gale et al. (2023) — *MegaBlocks: Efficient Sparse Training with MoE* | 用 block-sparse 矩阵格式（dMoE）将不等长的 expert GEMM 转化为一个大稀疏 GEMM，避免 padding | SonicMoE 直接对比基线 |

#### 直接对比/竞争工作
| 论文 | 核心贡献 | 与 SonicMoE 的差异 |
|------|---------|------------------|
| Liu et al. (2024) — *ScatterMoE* | 用 triton 实现 scatter/gather 融合 GEMM，减少显存搬运 | SonicMoE 后向传播比其快 83%，激活显存低 45% |
| *MoMoE: Memory Optimized MoE* | 针对 MoE 反向传播优化激活存储 | SonicMoE 后向传播比其快 115%，激活显存更低 |
| NVIDIA DeepGEMM/CUTLASS | 提供高性能 FP8 GEMM kernel，作为计算基线 | SonicMoE 前向较 DeepGEMM 基线提升 43% |

#### 近期 MoE 模型（使用 Fine-grained MoE 的目标系统）
| 模型 | 机构 | 规模 | granularity |
|------|------|------|-------------|
| DeepSeek-V3 (2024) | DeepSeek | 671B total / 37B active | 256 experts |
| Kimi K2 (2025) | Moonshot | 1T total / 32B active | 384 experts |
| Qwen3 (2025) | Alibaba | - | 128 experts |
| OLMoE (2024) | AI2 | 7B total / 1B active | 64 experts（开源基准） |


## ⚙️ 核心设计与机制

SonicMoE 的三大技术柱互相独立，可以按需组合使用。

---

### 1. IO 感知激活重算（IO-Aware Activation Recomputation）

#### 痛点分析：激活显存为什么与 granularity 成正比？

标准 MoE 反向传播需要保存前向传播中的中间激活，以便在 backward pass 计算梯度。一个标准的 MoE 前向（含 SwiGLU）的计算图如下：

$$\text{SwiGLU: } A = \text{SiLU}(X W_{gate}) \odot (X W_{up}), \quad Y = A W_{down}$$

要计算所有参数梯度，朴素做法需要缓存：
- $X$：token 特征，尺寸 $T \times d$
- $H_1 = X W_{gate}$：gate 前激活，尺寸 $T_{routed} \times d_{ff}$
- $S = \text{SiLU}(H_1)$：SiLU 激活值，尺寸 $T_{routed} \times d_{ff}$
- $H_2 = X W_{up}$：up projection 激活，尺寸 $T_{routed} \times d_{ff}$
- $A = S \odot H_2$：SwiGLU 输出，尺寸 $T_{routed} \times d_{ff}$

其中 $T_{routed} = T \times K$（top-K token），这些激活总字节数为 $O(T \cdot K \cdot d_{ff})$，随 $d_{ff}$ 和 $K$ 线性增长。

#### SonicMoE 的解法：精选缓存 + 选择性重计算

SonicMoE 指出，**并非所有中间激活都值得缓存**。关键洞察是：

> **只缓存 $X$ 和 $H$（up/gate 的第一个中间值），以及路由元数据，其余在 backward 时重计算。**

具体地：
- **缓存内容**：$X$（$T \times d$）+ $H$（$T \times K \times n$，其中 $n$ 为每 expert 的中间维度）+ 路由索引，总计 $2Td + 4TKn$ 字节
- **不缓存**：$S$（SiLU 输出）、$A$（SwiGLU 乘积输出）

在反向传播时，从 $X$ 和 $H$ 重计算所需激活。由于这些重计算所需的数据比全缓存少得多，激活显存与 granularity **解耦**——即使 expert 数量增加，缓存总量不随 $E$ 增长。

**节省量**（论文 Table 1）：
- 7B 规模（$n=256$）：相比 ScatterMoE（缓存全部激活）节省 **45%**
- 120B 规模：每层节省超过 **3 GiB**

---

### 2. Tile 感知 GPU Kernel（Tile-Aware GPU Kernels）

#### 2.1 Hopper 架构（H100）：Ping-Pong 调度重叠 GEMM 与 IO

H100 的 Tensor Core 采用 **Warpgroup MMA（WGMMA）** 异步指令，允许 producer warpgroup 异步加载数据的同时，consumer warpgroup 执行矩阵乘法。SonicMoE 利用这一特性实现 **Ping-Pong pipeline**：

```
Warpgroup A: [Load Tile N]  [MMA Tile N-1]  [Load Tile N+2]  [MMA Tile N+1]
Warpgroup B: [MMA Tile N-1] [Load Tile N]   [MMA Tile N+1]   [Load Tile N+2]
             ←————————————— 两个 warpgroup 交替执行 IO 与计算 ——————————————→
```

**关键创新——Gather 融合**：传统实现中，Scatter/Gather（将 token 根据路由表重排给对应 expert）是独立的 memory-copy kernel，需要额外的显存带宽。SonicMoE 将 Gather 操作融合进 GMEM→SMEM 的加载阶段（`cp.async` 指令期间），消除独立 kernel 启动，节省一次全局内存访问。

**Epilogue 融合**：SwiGLU 的激活函数与 Scatter/写出操作融合进 kernel epilogue，避免中间结果写回全局内存。

#### 2.2 Blackwell 架构（B300）：TMEM + Relay Warp

B300 引入了全新的 **Tensor Memory（TMEM）** 架构和 **UMMA（Universal Matrix Multiply Accumulate）**：

- **UMMA**：单线程异步矩阵乘法指令，不占用寄存器文件存放累加结果（结果存在 TMEM 中），极大减少寄存器压力
- **并行 epilogue**：由于 MMA 累加器在 TMEM 而不占寄存器，epilogue warps（负责激活函数计算和写出）可以与 MMA warps 真正并发执行

**Relay Warp 机制**（B300 新增）：
- 在 multi-CTA cluster 内，Gather 需要跨 CTA 协调
- SonicMoE 使用专用的 relay warp，接收 `cp.async` 完成信号并通过 cluster 级同步原语（cluster barrier）转发给 CTA 0
- 实现跨 CTA 的 Gather 融合，无额外同步开销

#### 2.3 后向传播 Kernel 优化

后向传播涉及 $dX$（输入梯度）、$dW$（权重梯度）、$dH$（中间激活梯度）和 $dS$（SiLU 梯度）的计算。SonicMoE 发现：

**计算等价选择**：在 down projection 的反向传播中，$dS$ 可以用两种等价公式计算：
- 方式 A：$dS = \langle dO, Y \rangle$（先等待 $dO$ 就绪）
- 方式 B：$dS = \langle dA, A' \rangle$（使用局部缓存值，更早可用）

SonicMoE 选择方式 B，将 $dH$ 和 $dS$ 融合进同一个 kernel 的 epilogue，减少一次额外的全局内存读写。

---

### 3. Token Rounding（TR）路由算法

#### 问题描述

在 top-K token-choice 路由中，每个 expert 收到的 token 数量 $M_e$ 是随机且不均的。GEMM tile 要求 $M$ 维对齐到 $M_{tile}$（通常为 64 或 128），若 $M_e \mod M_{tile} \neq 0$，最后一个 tile 会有无效 padding，浪费计算。

在高稀疏度场景（$K/E \leq 1/32$），每个 expert 平均收到的 token 数 $\leq M_{tile}/2$，padding 浪费比例可超过 50%。

#### 算法设计（Algorithm 4）

Token Rounding 在路由分数层面进行干预，使每个 expert 的 token 数自然对齐到 $M_{tile}$，而不修改模型结构：

1. **执行标准 top-K routing**：得到路由分数矩阵 $R \in \mathbb{R}^{T \times E}$
2. **统计 token 频率**：计算每个 expert 当前接收的 token 数 $c_e$
3. **计算补充需求**：对于每个 expert，计算将其 token 数"凑整"到下一个 $M_{tile}$ 倍数还差几个 token：
$$\delta_e = M_{tile} \cdot \lceil c_e / M_{tile} \rceil - c_e$$
4. **调整 top-K 优先级**：对于需要补充 token 的 expert，从"未入选"的 token 中按 routing 分数排名优先选取 $\delta_e$ 个 token，补充给该 expert

**约束保证**：
- 对每个 expert，与原 token-choice 路由的最大偏差不超过 1 个 tile（$\leq M_{tile}$ 个 token）
- 高优先级的 top-K 分配不会被剥夺

**质量验证**：在 OLMoE 规模（7B，64 experts，top-8）的语言模型预训练中，Token Rounding 在极端稀疏设置下"验证困惑度相似或更低"，表明路由质量不受损。


## 📊 实验与性能评价

### 实验配置

SonicMoE 在以下配置下进行评测：
- **硬件**：64 × H100（Hopper）和 B300（Blackwell，数量未明确说明）
- **模型规模**：OLMoE 规模（7B 参数，64 experts，top-8）
- **精度**：BF16（训练精度）
- **基线系统**：ScatterMoE、MoMoE、MegaBlocks、DeepGEMM（NVIDIA 官方 FP8 GEMM kernel）

---

### 激活显存节省（与 ScatterMoE 对比）

| 模型规模 | Expert 数 | ScatterMoE 激活内存 | SonicMoE 激活内存 | 节省比例 |
|---------|---------|------------------|--------------------|--------|
| 7B      | n=256   | baseline         | -45%               | **45%** |
| 120B    | -       | -                | -                  | >3 GiB/层 |

关键点：SonicMoE 的激活显存大小**不随 granularity（expert 中间维度 $n$）增长**，而 ScatterMoE 和 MoMoE 均随 $n$ 线性增长。

---

### 训练吞吐量（H100，前向/后向 kernel 性能）

| Kernel | SonicMoE | 对比基线 | 提升 |
|--------|----------|---------|-----|
| 前向传播 | - | DeepGEMM | **+43%** |
| 后向传播 | - | ScatterMoE | **+83%** |
| 后向传播 | - | MoMoE | **+115%** |
| 端到端（64×H100） | **213 亿 token/天** | - | - |

注：相对基线的 1.86× 计算吞吐提升来自前后向综合数据（论文摘要数据）。

---

### Token Rounding 效果

| 场景 | Token Rounding vs. Top-K | 说明 |
|------|--------------------------|------|
| 极端高稀疏（$K/E \leq 1/32$） | **+16% kernel TFLOPS** | padding 利用率提升 |
| 一般稀疏度 | 效果递减 | padding 本来不严重 |
| 语言模型质量 | 相似或更低验证困惑度 | 路由质量不受损 |

---

### Blackwell（B300）性能

| 模型规模 | 指标 | SonicMoE | 对比基线（DeepGEMM++） |
|---------|------|---------|----------------------|
| 7B（OLMoE） | 前向 | +25% | DeepGEMM++ |
| 7B（OLMoE） | 后向 | +15% | DeepGEMM++ |
| 绝对计算利用率 | TFLOPS | 1100+ TFLOPS | - |

---

### 局限性（Limitations）

1. **架构绑定**：深度优化针对 Hopper（H100）和 Blackwell（B300）架构，在其他 GPU 平台（A100、AMD MI300X 等）不适用或需要重新实现。
2. **Token Rounding 质量权衡**：在低稀疏度场景（$K/E$ 较大），Token Rounding 带来的计算增益有限，且轻微改变了路由分布，对对模型质量的长期影响需要更大规模实验验证。
3. **细粒度 MoE 专属**：论文的优化专门针对 expert 数量极多、expert 中间维度较小的 fine-grained MoE 架构（如 DeepSeek-V3 的 $E=256$ 方案）。对于粗粒度 MoE（$E$ 小，$d_{ff}$ 大）效果未经验证。
4. **开源代码状态**：论文声明已开源全部 kernel，但截止检索时未在论文正文中明确给出 GitHub 仓库链接，实际可用性待确认。

---

## 💡 启发与我的想法

### 可借鉴之处

1. **激活重算的精细化选择**：SonicMoE 并非简单地"要么全缓存，要么全重算"，而是分析了每种激活的重计算成本与显存收益，选择性地只缓存关键激活（$X$ 和 $H$）。这种**成本-收益分析驱动的精选重算**策略可以迁移到其他网络结构的显存优化中。

2. **Gather 融合思路的推广**：将路由相关的 scatter/gather 操作融合到数据加载阶段（IO 过程中做数据重排），而不是作为独立 kernel 运行，是一种系统性消除"无谓 IO"的优化思路。类似的机会存在于其他需要动态数据重排的场景（如稀疏注意力、KV cache 管理等）。

3. **Token Rounding 的工程意义**：通过在路由分数上做小幅调整来对齐 GEMM tile，而不是修改模型结构或填充假 token，是一种优雅的工程解法。类似的"对齐感知路由"思想值得在 KV Cache 稀疏选择（如按 block 对齐的 token 选择）中借鉴。

4. **Hardware-co-design 的重要性**：SonicMoE 针对 Hopper/Blackwell 的 WGMMA/UMMA 异步特性量身设计 Ping-Pong pipeline 和 TMEM epilogue，体现了当前 AI 系统研究"算法-硬件协同设计"的主流范式。

### 改进空间

1. **跨平台泛化**：当前实现高度依赖 CUDA 特定指令，对 AMD ROCm 或未来 NPU 平台不友好，通用化抽象值得研究。
2. **与 FlashAttention 类方案对比**：FlashAttention 对注意力机制做了类似的 IO 感知优化，但 MoE 的路由不规则性更强，能否设计出类似 FlashMoE 的统一框架？
3. **动态稀疏 + Token Rounding 的结合**：Token Rounding 当前是静态对齐策略；若结合动态稀疏（不同 step 稀疏度变化），如何自适应调整 $M_{tile}$ 目标？
4. **RL 训练场景**：RL 的 batch size 和序列长度分布比预训练更不规则，Token Rounding 在 RL on-policy rollout 阶段的效果尚未研究。

### 下一步 Action
- [ ] 追踪 SonicMoE 开源仓库，运行本地 benchmark 对比 ScatterMoE
- [ ] 研读 FlashAttention-3（arXiv:2407.08608）的 Ping-Pong pipeline 实现，与 SonicMoE 的 Hopper Kernel 对比设计思路
- [ ] 调研 Token Rounding 与 expert capacity factor 的关系：capacity factor 本质上也是在做 token 截断，两者如何统一？
- [ ] 考虑将 "Gather 融合" 思路用于项目中 KV Cache 的稀疏 block 读取优化


