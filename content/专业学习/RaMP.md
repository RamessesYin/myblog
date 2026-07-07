---
title: "RaMP: 面向MoE推理的运行时感知超内核多态优化"
tags:
  - MoE推理
  - kernel优化
  - 专家路由
  - GPU调度
  - LLM服务
---

## 🧠 核心摘要 (TL;DR)
> 现有 MoE 推理系统依赖静态批量大小决策内核配置，忽略了运行时实际的专家路由分布，导致 10-70% 的内核吞吐浪费。RaMP 提出一个路由感知的动态分派框架，通过四参数波浪代价模型从运行时专家直方图中选择最优内核配置，在 vLLM 端到端服务中取得 1.30× 加速（相比 Triton），1.41× 相比 DeepGEMM。

## 🏷️ 元数据
- **论文标题**: RaMP: Runtime-Aware Megakernel Polymorphism for Mixture-of-Experts
- **发表机构/会议**: arXiv（Vyom Sharma, Debajyoti Datta）
- **年份**: 2026
- **链接**: [[RaMP_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #moe推理 #kernel优化 #专家路由 #gpu调度 #llm服务
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[ZipServ]]
- **相关节点 (Related Notes)**: [[MSched]], [[ShiftParallelism]]
- **向下延申 (Successors)**: 

---

## 🎯 动机与痛点

### 现有方案的瓶颈

Mixture-of-Experts（MoE）架构已成为当前最先进 LLM 的核心设计范式，从 DeepSeek-V3（671B 参数，每 token 激活 37B）、Qwen3 MoE（235B 参数）到 OLMoE-1B-7B 等，均采用稀疏激活的 MoE FFN 层。MoE 的核心优势在于：以接近稠密模型的计算代价（activated FLOPs）实现更大的模型容量，从而在推理阶段实现更优质的输出。

然而，MoE 推理的 GPU 内核优化面临一个根本矛盾：**静态内核配置 vs. 动态路由分布**。

**传统方案的核心假设与失效**：现有生产系统（Triton、DeepGEMM、FlashInfer CUTLASS）在决定调用哪种内核配置（block tile size、pipeline stage 数量、warp 划分方式等）时，仅依赖**批量大小（batch size）**作为启发信息。这一假设在稠密推理中成立，因为 dense FFN 的计算量仅由批量大小决定。但在 MoE 推理中，**每个专家实际接收的 token 数量**（expert load）取决于路由器对当前批次的分配结果，而非均匀分布。

具体而言，若当前 batch 有 B 个 token，MoE 层有 E 个专家，top-k 路由，则理论上每个专家期望 $\frac{B \cdot k}{E}$ 个 token，但实际分布高度不均——某些专家可能承载 10 倍于平均值的负载，另一些专家则几乎为空。这种**运行时路由偏斜（routing skew）**导致：

1. **内核吞吐损失 10-70%**：为处理大专家负载而选择的大 tile 内核，在处理小负载专家时产生严重的 padding 浪费；反之亦然
2. **固定配置无法覆盖全谱**：GEMM 的最优 tile size 与输入矩阵尺寸高度相关，静态选择必然在部分工况下次优
3. **跨架构泛化差**：针对 H100 手工调优的内核未必在 A100、H200 或 MI300X 上最优

### 论文的切入点

RaMP 的洞察是：**路由分布是可以在运行时低成本观测的**。在 MoE forward pass 中，router 产生的 token-to-expert 映射（专家直方图）早于 GEMM 执行，这创造了一个窗口——在 GEMM 实际执行前动态选择最优内核配置。

RaMP 的核心设计包括三个相互支撑的组件：
1. **性能区域分析**：离线分析各架构上不同内核配置的性能包络
2. **四参数波浪代价模型**：运行时从专家直方图预测各配置代价，以 0.93% 均值遗憾选出最优配置
3. **CuTe DSL 超内核**：提供 134-268 个多态配置，无需重编译即可切换

### 相关工作

#### 先驱/奠基工作

**MegaBlocks (Gale et al., 2022, arXiv:2211.15841)** 是 MoE GPU 内核优化领域的先驱工作。它将 MoE 计算重新表述为块稀疏操作（block-sparse operations），开发了专用的块稀疏 GPU 内核，在不丢弃 token 的前提下高效处理 MoE 的动态性，训练速度相比 Tutel 提升 40%。但 MegaBlocks 面向训练场景，不专注于推理时的路由感知配置选择。

**ScatterMoE (Tan et al., 2024, arXiv:2403.08245)** 提出了 Scattered Mixture-of-Experts 实现，通过 ParallelLinear 组件避免不必要的 padding 和输入复制，改善推理和训练速度及内存占用。ScatterMoE 相比 MegaBlocks 在吞吐和内存效率上均有提升，但仍未解决运行时路由感知的内核动态选择问题。

**vLLM / PagedAttention (Kwon et al., 2023, arXiv:2309.06180, SOSP'23)** 奠定了现代 LLM 高吞吐服务的基础，通过虚拟内存式 KV cache 管理将显存浪费降至接近零，吞吐相比 FasterTransformer 提升 2-4×。RaMP 在 vLLM 框架上实验，将 FFN kernel 替换为路由感知调度，实现端到端加速。

**FlashInfer (Ye et al., 2025, arXiv:2501.01005, MLSys'25)** 提出了一个高度可定制的 LLM 推理注意力引擎，通过 JIT 编译和负载均衡调度算法实现 29-69% 的 inter-token latency 减少。RaMP 在与 FlashInfer CUTLASS 的对比中实现了 1.13× 的内核加速。

#### 直接对比方法

**Triton** 是目前主流 MoE 推理框架（vLLM、SGLang 默认后端）使用的 GPU 内核，基于静态批量大小选择配置。RaMP 在 vLLM 端到端基准中相比 Triton 实现 1.30× 加速。

**DeepGEMM** 是 DeepSeek 团队开发的高性能 GEMM 内核，专为 H800/H100 优化，也采用静态分派策略。RaMP 相比 DeepGEMM 实现 1.41× 加速（仅内核层）。

**Cross-Platform Fused MoE Dispatch in Triton (Mitra, 2026, arXiv:2605.23911)** 提出 TritonMoE，在 Triton 中实现完整 MoE forward pass（包括 router scoring、token permutation、expert GEMMs），达到 CUDA 优化 MegaBlocks 的 89-131% 吞吐，但仍缺乏运行时路由感知的配置选择。

**SERE (Wu et al., 2026, arXiv:2602.07616)** 通过基于相似度的专家重路由实现批量解码中动态减少激活专家数量，以最小质量损失实现最高 2.0× 加速。SERE 关注路由策略本身的改变，而 RaMP 在固定路由策略下优化内核执行。

#### Follow-up 与相关系统工作

**MegaScale-Infer (Zhu et al., 2025, arXiv:2504.02263)** 提出通过分离 Attention 和 FFN 模块的算子分解（disaggregated expert parallelism）实现 MoE 服务扩展，配合 ping-pong pipeline parallelism，实现每 GPU 吞吐 1.90× 提升。该工作与 RaMP 互补——MegaScale-Infer 在系统层面优化通信与调度，RaMP 在内核层面优化 GEMM 执行。

**ReaLB (Wang et al., 2026, arXiv:2604.19503)** 针对多模态 MoE 推理中视觉 token 导致的负载不均衡，通过运行时精度调整实现 1.10-1.32× 端到端加速，展示了运行时感知优化的普适性。

**MoEShard (Balmau et al., 2026, arXiv:2503.08467)** 利用张量分片平衡多 GPU 间的专家计算，在首 token 延迟上取得最高 6.4× 加速，专注于多 GPU 扩展性。

**Multi-Layer Scheduling for MoE LLM (Sun et al., 2026, arXiv:2602.21626)** 提出三层调度框架（请求级、引擎级、专家级），TTFT 降低 17.8%，与 RaMP 关注的内核级优化互补。


## ⚙️ 核心设计与机制

RaMP 的整体架构可概括为：**离线性能分析 + 运行时代价建模 + 多态内核执行**。

### 架构总览

```
          [路由器输出]
               │
    ┌──────────▼──────────┐
    │  专家直方图提取      │  ← 从 top-k 路由结果统计每专家 token 数
    └──────────┬──────────┘
               │  histogram
    ┌──────────▼──────────┐
    │  波浪代价模型查询    │  ← 四参数模型预测各配置执行时间
    └──────────┬──────────┘
               │  最优配置 ID
    ┌──────────▼──────────┐
    │  CuTe 超内核分派    │  ← 134-268 个多态配置，无需重编译
    └─────────────────────┘
```

### 关键技术点 1：性能区域分析（Performance Region Analysis）

在实际选择内核配置之前，RaMP 需要理解不同内核配置在不同硬件上的性能包络。论文对目标架构（H100、A100、H200、MI300X 等共 8 种，含 3 种此前未见架构）进行系统性 profiling：

**离线分析流程**：
1. 对每种硬件，遍历所有内核配置（tile size $M_b \times N_b \times K_b$、流水线 stage 数 $s$、warp 分组方式等）
2. 针对不同专家负载大小（expert load sizes），测量各配置的实际执行时间
3. 识别不同负载区间内的**性能最优配置集合**（Pareto-optimal frontier）

**关键发现**：性能最优配置并不是单调的——随着专家负载从小到大，最优配置在多个"性能区域（performance regions）"之间切换。小负载下，小 tile 配置避免过多 padding；大负载下，大 tile 配置充分利用 tensor core 的 MMA 效率。这使得"一配置通吃所有负载"策略的损失可高达 70%。

**跨架构泛化**：关键设计是只需 **10-24 分钟的 profiling**（相比传统 autotuning 数小时）即可为新架构建立性能区域模型，具有强的工程实用性。对于 3 种未见架构，通过性能区域迁移实现预测准确性，验证了框架的泛化能力。

### 关键技术点 2：四参数波浪代价模型（Wave Cost Model）

这是 RaMP 运行时决策的核心。内核执行可以被建模为若干"波浪（waves）"——每个波浪对应一批并发执行的线程块（CTAs）。总执行时间近似为：

$$T = \lceil \text{num\_CTAs} / \text{SMs} \rceil \times T_{\text{wave}}$$

但这一简单模型不能捕捉 MoE 的关键特征：**不同专家的负载不同，因此在同一个内核调用中，各专家对应的线程块数量不同**，导致最后一个波浪（tail wave）可能严重浪费 SM 资源。

RaMP 提出四参数代价模型，精确捕捉这种 tail wave 效应：

$$\text{Cost}(config, histogram) = f(n_{\text{full}}, n_{\text{tail}}, \lambda_{\text{full}}, \lambda_{\text{tail}})$$

其中：
- $n_{\text{full}}$：完整波浪的数量（所有 SM 都被占用）
- $n_{\text{tail}}$：tail wave 中实际使用的 SM 数
- $\lambda_{\text{full}}$：完整波浪的延迟（从 profiling 数据获得）
- $\lambda_{\text{tail}}$：tail wave 的延迟（通常与完整波浪相同，因为 SM 执行时间不随占用数变化）

**从专家直方图到代价计算**：给定运行时专家直方图 $h = \{h_1, h_2, ..., h_E\}$（$h_i$ 为第 $i$ 个专家接收的 token 数），对于某内核配置 $c$（tile size $(M_b, N_b)$，SM 数 $S$）：

$$\text{CTAs}_i = \lceil h_i / M_b \rceil \times \lceil d_{model} / N_b \rceil$$
$$\text{Total CTAs} = \sum_i \text{CTAs}_i$$
$$\text{Cost}(c, h) = \lceil \text{Total CTAs} / S \rceil \times \lambda_{\text{full}} + \text{tail\_penalty}(h, c)$$

这一模型的优势在于：给定专家直方图后，对每个候选配置的代价计算**复杂度为 $O(E \cdot |C|)$**（$E$ 为专家数，$|C|$ 为候选配置数），在实际中（$E \approx 256$，$|C| \approx 200$）可在毫秒级完成，远低于 GEMM 执行时间（数十毫秒）。

**均值遗憾 0.93%**：在 8 种架构上的评估表明，波浪代价模型选出的配置与 oracle 最优配置之间的性能差距均值仅为 0.93%，即模型几乎每次都能找到最优配置。

### 关键技术点 3：CuTe DSL 超内核设计

为了支持运行时多态配置切换，RaMP 使用 NVIDIA 的 **CuTe DSL（CUDA Template Library）** 设计了一个高度参数化的超内核（megakernel）：

**多态参数空间**：CuTe 超内核通过 C++ 模板实例化为 **134-268 个不同配置**（因模型/硬件而异），覆盖：
- 多种 MMA tile size（$M_b \in \{16, 32, 64, 128\}$，$N_b \in \{32, 64, 128, 256\}$）
- 多种 pipeline stage 数（$s \in \{1, 2, 3, 4\}$）
- 多种 epilogue 方案（fused 量化、bias、activation 等）

**内核不可知性（Kernel-Agnostic）**：RaMP 的代价模型和分派框架并不绑定于特定的内核实现——理论上可以将任意 MoE GEMM 内核（Triton、CUTLASS、自定义内核）包装为多态超内核，只需提供离线 profiling 数据即可。

**与 Alpha-MoE 的集成**：论文展示了将 RaMP 框架应用于 Alpha-MoE（一个专家并行 MoE 系统）而无需修改源代码，取得 1.14× 加速，验证了框架的通用性。

### 核心公式汇总

内核执行代价建模（简化形式）：
$$T_{\text{kernel}}(c, h) = \left\lceil \frac{\sum_{i=1}^{E} \text{CTAs}(h_i, c)}{S} \right\rceil \cdot \lambda(c)$$

最优配置选择：
$$c^* = \arg\min_{c \in \mathcal{C}} T_{\text{kernel}}(c, h_{\text{runtime}})$$

其中 $\mathcal{C}$ 为所有预编译的多态内核配置集合，$h_{\text{runtime}}$ 为路由器产生的实时专家直方图。


## 📊 实验与性能评价

### 对比基线 (Baseline)

RaMP 在以下基准上进行评估：

| 系统 | 类型 | 特点 |
|------|------|------|
| Triton MoE | 主流生产系统 | vLLM/SGLang 默认后端，静态 batch size 调度 |
| DeepGEMM | DeepSeek 团队 | H800/H100 优化，FP8 GEMM，静态分派 |
| FlashInfer CUTLASS | MLSys'25 | 可定制注意力引擎，最先进 GEMM 后端 |
| Alpha-MoE | 专家并行系统 | 基线静态配置 |

### 主要指标与结果

**内核级（Kernel-Level）**：
- **1.22× 优于静态分派**（同一 CuTe 内核，RaMP 动态选择 vs. 静态选择）
- **1.41× 优于 DeepGEMM**
- **1.13× 优于 FlashInfer CUTLASS**

**系统级（End-to-End in vLLM）**：
- **1.30× 优于 Triton MoE**（端到端服务吞吐）
- **1.14× 优于 Alpha-MoE** 基线配置（无需修改源码）

**架构覆盖**：
- 在 **8 种 GPU 架构**上验证（包含 H100、A100、H200、MI300X，以及 3 种此前未见架构）
- 对 3 种新架构，仅需 **10-24 分钟 profiling** 即可部署

**模型覆盖**：
- DeepSeek-V3（671B/37B activated）
- Qwen MoE 系列
- OLMoE-1B-7B

**代价模型精度**：
- 四参数波浪代价模型的**均值遗憾（mean regret）仅 0.93%**，即与 oracle 最优选择的性能差距极小

### 局限性 (Limitations)

1. **Profiling 开销**：虽然 10-24 分钟的离线 profiling 已大幅低于传统 autotuning，但每个新架构仍需一次性投入
2. **仅针对 FFN GEMM**：RaMP 优化的是 MoE FFN 中的 GEMM 部分，不涉及 Attention 计算或 all-to-all 通信开销；在专家并行场景下，通信开销可能成为新瓶颈
3. **极端不均衡场景**：当路由极度偏斜（少数专家承载几乎全部 token），即使多态内核选择最优配置，tail wave 浪费仍难以完全消除
4. **代码未开源**：目前论文未提供代码库，复现和在其他框架集成存在障碍

## 💡 启发与我的想法

### 可借鉴之处

1. **运行时感知 = 低代价感知器 + 预编译多态策略的组合**：RaMP 的设计哲学非常清晰——不尝试在运行时 JIT 编译新内核（代价太高），而是**预先编译所有可能的优质配置，运行时只做选择**。这一思路可推广到其他有动态输入特征的算子（如 variable-length attention）。

2. **波浪代价模型的工程价值**：GPU 执行时间的"波浪效应"在很多算子中都存在，但极少有工作将其系统化为一个轻量级预测模型。这一思路对于任何需要在运行时选择 CUDA 内核配置的场景（如 Flash Attention 的 tile size 选择）都有参考价值。

3. **Kernel-Agnostic 框架**：RaMP 将代价模型与具体内核实现解耦，意味着不同团队的内核实现（Triton、CUTLASS、自定义内核）只需提供 profiling 数据即可受益。这是一个优秀的系统设计原则。

4. **MoE 推理的结构性机会**：本文揭示的核心洞察——路由分布在 GEMM 执行前已知——是一个被长期忽视的优化窗口。类似的"算子执行前可获得的元信息"在 LLM 推理系统中可能还有很多（如前缀缓存命中率影响 Attention 的 tile 策略）。

### 改进空间

1. **与负载均衡路由的协同**：RaMP 假设路由策略固定。如果路由策略可以感知"内核效率"（即倾向于生成更均匀的专家负载，使内核能更好地利用 SM），可能取得更大整体加速。

2. **扩展到专家并行（Expert Parallelism）场景**：当 MoE 跨多 GPU 分布时，专家负载信息的收集涉及 all-to-all 通信，代价模型需要同时考虑通信时延和计算时延的权衡。

3. **自适应 profiling**：当前 profiling 需要枚举所有配置。可以考虑基于高斯过程或贝叶斯优化的自适应 profiling，用更少的测量点覆盖性能曲面。

4. **FP8 / INT8 量化路径**：论文主要针对 BF16/FP16 GEMM，量化路径下的多态配置空间和代价模型可能有所不同，值得深入探索。

### 下一步 Action
- [ ] 跟踪 RaMP 代码开源情况，尝试在本地 H100 环境复现 1.30× vLLM 加速
- [ ] 阅读 MegaBlocks 和 ScatterMoE 原文，理解块稀疏 GEMM 的基础设计
- [ ] 调研 CuTe DSL 的使用方式，探索将 RaMP 类似思路用于 Attention kernel 的可行性
- [ ] 关注 DeepGEMM 的路由感知版本是否已经出现（DeepSeek 团队可能跟进）


