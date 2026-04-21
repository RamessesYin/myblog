---
title: "Helix Parallelism：面向交互式超长上下文 LLM 解码的分片策略重思"
tags:
  - 并行推理
  - KV Cache
  - LLM服务优化
  - 分布式推理
  - 长上下文
---

## 🧠 核心摘要 (TL;DR)

> Helix Parallelism 针对百万 token 上下文下的 LLM 实时解码场景，提出**时序解耦（Temporal Decoupling）**策略：在 Attention 阶段对 KV Cache 跨 GPU 分片（KV Parallelism），在 FFN 阶段将同一批 GPU 重新组织为张量并行/专家并行，从而同时解决 KV Cache 带宽墙与 FFN 权重带宽墙。在 NVIDIA GB200 上测试 DeepSeek-R1，TTL 降低 1.5×，批大小提升 32×。

## 🏷️ 元数据
- **论文标题**: Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM Decoding
- **发表机构/会议**: arXiv 2507.07120（NVIDIA, 2025）
- **年份**: 2025
- **链接**: [[HelixParallelism_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #kv-cache #tensor-parallelism #long-context #llm-inference #moe #attention
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[ShiftParallelism]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[MSched]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

超长上下文（≥1M token）实时解码面临两大独立的带宽瓶颈，并且二者的最优并行策略截然相反：

**瓶颈一：KV Cache 带宽墙**

KV Cache 大小随上下文长度 $S$ 和批大小 $B$ 线性增长：

$$\text{KV Size} = 2 \cdot L \cdot H \cdot S \cdot B \cdot \text{dtype\_bytes}$$

其中 $L$ 为层数，$H$ 为隐层维度。对于 DeepSeek-R1（671B MoE）在 1M token、batch=1 的配置下，KV Cache 已达到单 GPU HBM 容量上限。

传统张量并行（TP）在 $\text{TP} > K$（KV head 数量）时，必须在每个 GPU 上复制完整 KV Cache，无法通过增加 GPU 数量来分摊带宽压力——多卡并行反而造成内存冗余。

**瓶颈二：FFN 权重带宽墙**

在小批大小解码时，每生成一个 token 都需要从 HBM 加载 FFN 权重，这是主要延迟来源。解决这一瓶颈的最优策略恰好是扩大张量并行度（TP 越大，权重分片越细，每 GPU 加载量越小）。

**核心矛盾**：Attention 阶段需要跨 GPU 分片 KV Cache（KV Parallelism），FFN 阶段需要最大化 TP，而传统方案强制两个阶段使用相同并行策略，无法同时优化。

### 相关工作对比

| 方法 | 解决的问题 | 局限 |
|------|-----------|------|
| **Tensor Parallelism** [Megatron-LM, Shoeybi et al. 2020] | FFN 权重分片 | TP > K 时 KV 冗余，无法扩展 |
| **Medha** [Agrawal et al. 2024] | KV Cache 并行分片（KVP） | Attention 与 FFN 阶段未解耦，同一 TP 度受限 |
| **LoongServe** [Wu et al. 2024] | 弹性序列并行（ESP）动态调整并行度 | 跨请求 KV 迁移开销大，专注于整体服务系统而非单请求 TTL |
| **USP** [Fang & Zhao, 2024] | 统一序列并行 | 面向训练，解码场景延迟未优化 |
| **Flash-Decoding** [Dao et al. 2023] | 长上下文解码中 Attention 的序列维度并行 | 限于单机，未考虑 FFN 阶段 |
| **Context Parallelism** [Yang et al. 2025] | 百万 token 推理跨 GPU 上下文分割 | Ring-Attention 通信开销随序列长度线性增长 |
| **PagedAttention/vLLM** [Kwon et al. 2023] | KV Cache 内存管理 | 聚焦内存碎片，不解决带宽墙 |

**先驱工作**：

- **FlashAttention** [Dao et al. 2022, arXiv:2205.14135]：IO 感知的分块注意力算法，减少 HBM 访问次数，为 Helix 的单 GPU 高效 Attention 计算提供基础。
- **FlashAttention-2** [Dao 2023, arXiv:2307.08691]：改进 GPU 并行化与工作分配，增加跨线程块的 query 并行，为 Helix 中 KVP ranks 的本地 FlashAttention 计算奠定基础。
- **FlashAttention-3** [Shah et al. 2024]：针对 Hopper/Blackwell 架构的异步和低精度优化。
- **GQA** [Ainslie et al. 2023, arXiv:2305.13245]：分组查询注意力，将 KV head 数 $K$ 与 Query head 数 $H$ 解耦，是 Helix 中 KVP × TPA ≤ K 约束的前提条件。
- **MQA** [Shazeer 2019]：多查询注意力，将 K 降至 1，极端减少 KV 规模，影响后续所有 KV 优化工作方向。
- **DeepSeek-V2/V3** [DeepSeek-AI 2024/2025]：MLA（多头隐注意力）与 MoE 架构，Helix 的主要测试目标 DeepSeek-R1 的基础。
- **SARATHI** [Agrawal et al. 2023, arXiv:2308.16369]：分块预填充与解码搭便车，提供了将 prefill/decode 拆分为不同并行策略的思路先例。
- **Native Sparse Attention** [Yuan et al. 2025, arXiv:2502.11089]：硬件对齐的稀疏注意力，作为减少长上下文 Attention 计算量的互补方向。


## ⚙️ 核心设计与机制

### 整体架构：时序解耦并行

Helix 的核心洞见是：**同一批 GPU 在不同计算阶段采用不同的并行拓扑**。设总 GPU 数为 $N = \text{KVP} \times \text{TPA}$，其中：

- **KVP**（KV Partition）：沿序列维度分片 KV Cache 的 GPU 数
- **TPA**（Tensor Parallelism for Attention）：Attention 投影层的 TP 度，要求 $\text{TPA} \leq K$（KV head 数）

整个前向传播流程按阶段切换并行策略：

```
┌─────────────────────────────────────────────────────┐
│  每层 Transformer Block                              │
│                                                     │
│  [QKV Projection]  → TP(TPA) 并行                  │
│  [FlashAttention]  → 本地 KV 分片计算               │
│  [All-to-All]      → Query head 维度重组             │
│  [Output Proj]     → All-Reduce over N GPUs         │
│                                                     │
│  [FFN / MoE]       → TP(N) 或 TP×EP 重新分片        │
└─────────────────────────────────────────────────────┘
```

### 关键技术点 1：Attention 阶段的 KV 分片与精确重建

**分片方式**：KV Cache 沿序列（token）维度均匀分配到 KVP 个 GPU 组，每组内 TPA 个 GPU 进一步做 TP。

**All-to-All 通信设计**：

执行完本地 KV 分片的 FlashAttention 后，各 GPU 持有的 Attention 输出是基于局部 KV 的**部分 softmax 结果**。Helix 利用 FlashAttention 的 online softmax 属性，通过一次 **All-to-All（在 query head 轴）** 交换，精确重建完整的 Attention 输出——无需近似，无精度损失。

通信量分析：
$$\text{All-to-All 数据量} = O(B \times H) \quad \text{（与序列长度 } S \text{ 无关）}$$

这是 Helix 相比 Ring-Attention（通信量 $O(B \times S \times H)$）的关键优势：**通信开销不随上下文长度增长**。

### 关键技术点 2：FFN 阶段的 GPU 重利用

Attention 阶段结束后，同一批 $N$ 个 GPU 被重新组织为：

- **Dense LLM（如 Llama-405B）**：维持 $\text{TPF} = N$ 的完整 TP，最大化权重分片收益。
- **MoE LLM（如 DeepSeek-R1）**：分解为 $\text{TPF} \times \text{EP}$ 网格，TPF 内做权重 TP，EP 维度做专家并行路由。

无需额外 GPU 调度或通信重构——硬件拓扑不变，仅改变逻辑分组。

### 关键技术点 3：HOP-B（异步通信隐藏）

Attention 阶段的 All-to-All 通信可与**下一个请求的 Attention 计算**重叠（Pipelined Batching），将通信延迟从关键路径上移除。

HOP-B 收益对比：
- **MoE 模型**：无 HOP-B 时 TTL 劣化约 1%（All-to-All 小，FFN 主导）
- **Dense 模型**：无 HOP-B 时 TTL 劣化约 12%（通信占比更高）

### 关键技术点 4：KV Cache 循环分配

当 KVP > 1 时，新生成 token 的 KV 对需要追加到各分片。为避免某个 rank 积累过多负载，Helix 采用**轮询（Round-Robin）** 策略：第 $t$ 个新 token 的 KV 写入第 $t \bmod \text{KVP}$ 个 rank，保证各 rank 内存增长均匀。

### 核心公式：Attention 计算复杂度

设 $S$ 为序列长度，$B$ 为批大小，$H$ 为隐层维度，$K$ 为 KV head 数：

$$\text{标准 TP Attention 计算量（每 GPU）} = O\left(\frac{B \cdot S^2 \cdot H}{\text{TPA}}\right)$$

$$\text{Helix KVP+TPA（每 GPU）} = O\left(\frac{B \cdot S^2 \cdot H}{\text{KVP} \cdot \text{TPA}}\right) = O\left(\frac{B \cdot S^2 \cdot H}{N}\right)$$

计算量线性扩展到全部 $N$ 个 GPU，而通信开销仅为 $O(B \times H)$，实现了接近理想的线性加速比。

---

## 📊 实验与性能评价

### 实验设置

- **硬件**：NVIDIA GB200（Blackwell 架构），FP4 精度
- **规模**：1～64 GPU，1M token 上下文长度
- **测试模型**：DeepSeek-R1（671B MoE）、Llama-405B（Dense）
- **基线**：纯 TP 方案（TensorRT-LLM 默认配置）

### 主要结果

**DeepSeek-R1（671B MoE，1M token）**：

| 指标 | Helix vs. TP 基线 |
|------|------------------|
| TTL（首 token 时延） | 降低 **1.5×** |
| 等延迟预算下批大小 | 扩大 **32×** |
| Tokens/s/GPU 吞吐 | 提升 **32×** |

**Llama-405B（Dense，1M token）**：

| 指标 | Helix vs. TP 基线 |
|------|------------------|
| 交互性（TTL） | 提升 **1.13×** |
| 吞吐量 | 提升 **4×** |

### HOP-B 效果验证

在 MoE 模型上，启用 HOP-B 后 All-to-All 通信被完全隐藏，TTL 无额外开销；Dense 模型中通信占总延迟比例更高，HOP-B 节省约 12% 的 TTL。

### 扩展性

GPU 数量从 1 到 64 的扩展测试中，Helix 保持**接近线性的 TTL 下降**，而 TP 在超过 KV head 数后扩展收益趋近于零。

### 局限性

1. **仅针对解码（Decode）阶段**：Prefill 阶段的并行策略未在本文中讨论，实际部署需要额外处理 Prefill-Decode 拆分。
2. **依赖 HBM 拓扑**：All-to-All 通信效率依赖 NVLink 带宽，跨节点（InfiniBand）场景的性能未详细报告。
3. **KVP 约束**：$\text{TPA} \leq K$ 的限制使得 GQA/MQA 模型中 KV head 数极少时 TPA 的扩展受限，KVP 必须承担更多 GPU 利用任务。
4. **FP4 精度依赖**：测试基于 Blackwell FP4，在旧代 GPU（Ampere/Hopper）上的性能增益需独立验证。

---

## 💡 启发与我的想法

**可借鉴之处**：

- **时序解耦思路**是一个通用设计原则：当一个 Transformer block 的不同子模块（Attention vs. FFN）有截然不同的最优并行策略时，不必被迫选择一个折中方案，而是可以在同一批硬件上动态切换并行拓扑。这个思路可以扩展到 Prefill 阶段（prefill 中 Attention 计算密集，可用 TPA 缩小；FFN 权重密集，可用大 TP）。

- **All-to-All 的通信量分析**（$O(B \times H)$ vs. Ring-Attention 的 $O(B \times S \times H)$）提供了一个清晰的决策框架：当 $S$ 很大时，基于 query head 轴的 All-to-All 天然优于 Ring 方案。

- **KV 循环分配**在工程实现上是一个低成本的负载均衡技巧，可推广到任何需要对流式写入的 KV Cache 做分片均衡的场景。

**改进空间**：

- **Prefill-Decode 联合优化**：本文只优化了解码阶段，而实际推理系统中 Prefill 的并行策略会影响 KV Cache 的初始分布，两者如何协同是下一步值得研究的问题。

- **跨节点扩展**：当 GPU 数超过单节点 NVLink 域时，All-to-All 会变成跨节点通信，延迟剧增。需要设计层次化的 KVP（节点内 KVP + 节点间不同策略）。

- **与 KV 稀疏化的结合**：Helix 假设所有历史 KV 都需要完整保留。若结合 KV 稀疏（如 KVZap、NSA）丢弃低重要性 KV，不仅减少 KV Cache 总量，还可能允许更大的 KVP，进一步分摊带宽压力。

- **MoE 路由与 EP 交互**：在 MoE 场景下，EP（Expert Parallelism）的路由决策依赖 token 分布，极端不平衡时部分专家 GPU 空转。Helix 目前未报告 EP 负载不均对 TTL 的影响。

**下一步 Action**：
- [ ] 追踪作者是否发布 TensorRT-LLM 的 Helix 实现 PR，尝试在 Hopper 架构上复现关键数据
- [ ] 调研 All-to-All over query-head axis 在不同 KV head 配置（GQA K=8 vs K=32）下的通信量变化
- [ ] 思考 Helix 的时序解耦策略能否应用到本项目的 KV Cache 优化路径中

