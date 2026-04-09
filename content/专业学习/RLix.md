---
title: "RLix"
tags:
  - 强化学习
  - GPU调度
  - 资源管理
  - LLM训练
  - RL
  - 多任务
---

## 🧠 核心摘要 (TL;DR)
> RLix 是一个 GPU 调度与资源管理系统，针对 RL 实验"多任务并发排队时间长、GPU 利用率低"的痛点，通过**跨任务 GPU 共享**、**多 LoRA 适配器整合**和**弹性 Rollout 伸缩**三大机制，让多个 RL 训练任务共享 GPU 容量，减少等待时间、提升硬件利用率。

## 🏷️ 元数据
- **项目标题**: RLix: GPU Scheduling for Reinforcement Learning Experiments
- **发表机构/会议**: rlops 组织（GitHub 开源项目）
- **年份**: 2025（推断，基于 ROLL 的发展时间线）
- **链接**: [[RLix_blog|📄项目主页]] | [💻 Code](https://github.com/rlops/rlix)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #reinforcement-learning #gpu-scheduling #multi-task #lora #elastic-scaling
- **前置知识 (Prerequisites)**: [[RL中的KV稀疏讨论]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[MSched]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈

在 RL 实验集群中，GPU 资源管理面临一个根本性矛盾：

- **串行等待问题**：多个 RL 训练任务需要独占分配 GPU，当集群繁忙时，"有前景的任务可能要等待太长时间才能启动"，严重拖慢研究迭代速度
- **GPU 利用率不均衡**：RL 训练管线内部各阶段（Rollout 采样 vs 参数更新）对 GPU 的需求差异巨大；Rollout 阶段是 memory-bound 推理，Actor/Critic 训练是 compute-bound 更新，两者不能很好地复用同一套 GPU 配置
- **内存浪费**：多个 LoRA 实验需要各自加载基础模型，造成基础模型权重的重复显存占用
- **资源孤岛效应**：不同 RL 任务之间的空闲 GPU 无法被其他任务利用，利用率在时间维度上严重碎片化

### 论文的切入点

RLix 的核心洞见是：**RL 训练管线内存在天然的优先级层次结构**，可以将低优先级阶段（如 Rollout 采样）设计为可抢占的弹性资源，从而在不影响训练正确性的前提下，让空闲 GPU 供其他任务借用。

这一思路直接受 Alibaba/ROLL 项目的**"Partial Overlapping"调度**概念启发——将原本串行的计算阶段部分重叠，减少 GPU 空转时间。

---

### 相关工作

#### 奠基工作

**1. ROLL（Alibaba）**
RLix 的直接灵感来源。ROLL（Reinforcement Learning Optimization for Large-Scale Learning）是 Alibaba 开源的大规模 LLM 强化学习框架，核心创新是"样本级异步并行 Rollout"（Sample-level Asynchronous Parallel Rollout）与动态采样机制。其 Partial Overlapping 调度思想——让训练与采样阶段部分重叠——被 RLix 拓展到多任务、跨管线维度。ROLL 集成了 Megatron-Core、SGLang、vLLM，支持 PPO、GRPO 等多种 RL 算法，并在 2026 年 3 月新增了 Qwen3.5 和 FSDP2 支持。

**2. Ray（Moritz et al., OSDI 2018, arXiv:1712.05889）**
RLix 底层基础设施。Ray 提供统一的任务并行与 Actor 计算编程接口，其分布式调度器和故障容错状态管理使得跨节点、跨任务的动态资源协调成为可能。RLix 使用 Ray 作为分布式计算框架，实现跨任务 GPU 资源的动态分配。

#### 直接对比/同类系统

**3. HybridFlow/veRL（Sheng et al., arXiv:2409.19256）**
Volcengine 开源的 RLHF 框架，核心创新是混合单控制器/多控制器编程模型和 3D-HybridEngine（在训练与生成之间零冗余 reshape 权重），相比 DeepSpeed-Chat 提升 3.67×，相比 OpenRLHF 提升 3.25×，相比 NeMo-Aligner 提升 12.52×。与 RLix 的区别：HybridFlow 聚焦**单任务 RLHF 管线内部**的阶段间效率优化；RLix 则聚焦**多任务之间**的 GPU 共享。

**4. OpenRLHF（Hu et al., arXiv:2405.11143）**
基于 Ray+vLLM+DeepSpeed+HuggingFace 的 RLHF 框架，设计简洁、易用，每个模型（Actor、Critic、Reward、Reference）部署在独立设备上支持并行执行，但存在 Actor 模型重复副本导致的显存冗余。训练速度相比竞争框架提升 1.22×-1.68×。RLix 中的多管线支持部分借鉴了 OpenRLHF 的分布式模型管理理念。

**5. DeepSpeed-Chat（Yao et al., arXiv:2308.01320）**
Microsoft DeepSpeed 团队推出的 RLHF 系统，将所有模型共置于同一设备、采用 ZeRO-3 训练和张量并行生成，存在权重 reshape 开销和次优生成吞吐量问题。是 veRL 和 RLix 设计上的重要参考对比基线。

**6. NeMo-Aligner（Shen et al., arXiv:2405.01481, COLM 2024）**
NVIDIA 推出的对齐训练工具包，支持 RLHF、DPO、SteerLM 等多种方法，可扩展至千 GPU 规模（处理 Nemotron 4 340B、Llama 3.1 405B 等超大模型）。其均匀 3D 并行策略在生成阶段存在利用率不足的问题。

**7. ReaLHF（Wei et al., arXiv:2406.14088, MLSys 2025）**
提出"参数动态重分配"（Parameter ReaLlocation），针对 RLHF 训练中多 LLM 实例之间的复杂依赖关系，自动搜索最优执行计划并动态重分配 GPU 资源，在 70B 模型 128 GPU 场景下实现 3.58× 加速，长上下文场景比 Megatron-LM 基线提升 81%。

#### 关键技术基础

**8. S-LoRA（Sheng et al., arXiv:2311.03285）**
提出同时服务数千个 LoRA 适配器的系统：将全部适配器存主存、按需加载到 GPU，统一内存分页同时管理不同 Rank 的适配器权重和 KV Cache，自定义 CUDA 核实现异构批次 LoRA 计算，吞吐提升最高 4×。RLix 的多 LoRA 整合机制（多适配器共享基础模型）与 S-LoRA 的思路高度相关，但 RLix 聚焦训练侧而非推理侧。

**9. Kimi k1.5（Moonshot AI, arXiv:2501.12599）**
RL 训练大规模应用的代表性实践，展示了 RL scaling 的核心价值——通过长上下文 scaling 和改进的策略优化，在 AIME 达到 77.5 分，MATH 500 达到 96.2 分，无需 MCTS 或价值函数。说明高效 RL 基础设施（如 RLix 解决的 GPU 利用率问题）对前沿 RL 研究的重要性。



---

## ⚙️ 核心设计与机制

RLix 的架构分为三层，围绕一个核心思想：**将 RL 管线中可抢占的阶段（Rollout）设计为弹性资源池，供集群内其他任务动态借用**。

### 整体架构：三层分离

```
┌─────────────────────────────────────────────────┐
│          共享编排层（Shared Orchestration）         │
│   ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│   │  Orchestrator │  │ Scheduler │  │ Resource Mgr │  │
│   │ 任务生命周期  │  │ 优先级/共享│  │ 集群资源追踪  │  │
│   └──────────┘  └──────────┘  └───────────────┘  │
└─────────────────────────────────────────────────┘
         │                │
    ┌────┴────┐      ┌────┴────┐
    │ Pipeline A│      │ Pipeline B│    ...
    │ Coordinator│    │ Coordinator│
    └─────────┘      └─────────┘
```

- **Orchestrator**：管理所有任务的生命周期（创建、调度、终止）
- **Scheduler**：维护全局 GPU 分配优先级，决定哪些任务可以借用哪些资源
- **Resource Manager**：实时追踪集群中每张 GPU 的可用状态
- **Pipeline Coordinator**：每个 RL 任务独有，负责维护自身的训练逻辑，并遵从编排层的 GPU 分配决策

### 关键机制一：GPU 优先级调度层次

RLix 将 RL 训练管线内的各计算阶段按优先级排序，从高到低：

| 优先级 | 阶段 | 说明 |
|--------|------|------|
| Level 0 | 模型初始化 | 最高优先，确保任务启动不被抢占 |
| Level 1 | Actor 训练（策略网络更新） | 训练核心，保证 GPU 独占 |
| Level 2 | Critic 训练（价值网络更新） | 同上 |
| Level 3 | Log Probability 计算 | 数据准备 |
| Level 4 | Value Estimation | 数据准备 |
| Level 5 | 其他日志概率计算 | 辅助计算 |
| **Level 6** | **Rollout（轨迹采样）** | **可抢占阶段 ✓** |

Rollout 是唯一可被抢占的阶段，原因是：
1. Rollout 是推理任务（memory-bound），可以随时暂停/恢复
2. Rollout 只需维持一定的吞吐量即可保证训练收敛，短暂暂停不影响正确性（通过 staleness bound 约束）
3. Rollout 通常是 RL 管线中最耗时的阶段（占比可达 50%+ 的总迭代时间），也是最大的可共享资源

### 关键机制二：跨任务 GPU 共享（Cross-job GPU Sharing）

当任务 A 的 Actor/Critic 处于训练阶段（占用自己的训练 GPU）时，其 Rollout 工作者可能暂时闲置等待新 prompt；与此同时，任务 B 的全参数微调管线需要额外 GPU 来加速 Rollout。RLix 的调度器可以：

1. 检测任务 A 的 Rollout 工作者当前空闲（或低优先级）
2. 将这些 GPU 临时借给任务 B 使用
3. 当任务 A 的 Rollout 阶段重新开始时，调度器发送抢占信号，任务 B 释放 GPU

这种设计类似 OS 中的时间片轮转，但是面向 **RL 阶段粒度**（而非任务粒度）的资源共享。staleness bound 确保 off-policy 数据不会陈旧到影响训练质量。

### 关键机制三：多 LoRA 适配器整合（RollMultiLoraPipeline）

当多个 LoRA 实验共享同一基础模型时：

```
传统方式：
  实验A: [基础模型权重 7B] + [LoRA-A]  → 独占 GPU 组 #1
  实验B: [基础模型权重 7B] + [LoRA-B]  → 独占 GPU 组 #2
  显存总占用 ≈ 14B 参数 × 2

RLix 方式：
  [基础模型权重 7B, 共享]
       ↓
  LoRA-A + LoRA-B 并发训练在同一 GPU 组
  显存总占用 ≈ 7B + LoRA-A + LoRA-B
```

RollMultiLoraPipeline 将多个 LoRA 适配器同时加载在一个共享基础模型之上并发训练，显著减少显存开销，适合需要搜索 LoRA rank/alpha 超参的实验场景。

### 关键机制四：弹性 Rollout 伸缩（Dynamic Rollout Scaling）

Rollout 工作者实现**自动弹性扩缩容**：
- **扩容时机**：Rollout 采样是系统瓶颈（训练工作者等待数据），调度器将来自其他任务的闲置 GPU 分配给当前 Rollout 工作者，加速数据收集
- **缩容时机**：其他任务的 Level 1-5 阶段需要 GPU，Rollout 工作者收到抢占信号，优雅地停止当前采样批次并释放 GPU

这一机制与 HybridFlow 中的 3D-HybridEngine 理念相似（减少计算资源在训练-生成切换时的浪费），但 RLix 在**跨任务**维度实现了更粗粒度的弹性。

### 支持的管线类型

| 管线 | 适用场景 | 特点 |
|------|---------|------|
| `RollFullFinetunePipeline` | 全参数微调，追求模型质量 | 弹性 GPU 管理，训练精度最高 |
| `RollMultiLoraPipeline` | 显存受限的多实验场景 | 多适配器共享基础模型，显存更省 |

两种管线均支持 on-policy 和 off-policy 训练方式，并通过 staleness bound 机制确保数据新鲜度符合算法要求。

---

## 📊 实验与性能评价

> ⚠️ 注意：RLix 是开源工程项目（GitHub 仓库），未发表配套学术论文，公开 README 中未提供系统性的性能评测数据。以下内容基于 README 的设计描述与架构分析，实验数据须参阅实际测试结果。

### 对比基线（Baseline）

根据 README 描述，RLix 的设计目标是优于以下基线场景：

- **串行排队模式**：多个 RL 任务独占 GPU，等待轮流执行——RLix 通过共享消除等待
- **固定 Rollout 分配**：Rollout 工作者与 GPU 绑定，不能动态扩缩——RLix 通过弹性伸缩提升利用率
- **多 LoRA 独立训练**：每个 LoRA 实验独占一套基础模型——RLix 通过 RollMultiLoraPipeline 节省显存

### 主要设计目标指标

- **等待时间（Wait Time）**：核心优化目标，通过跨任务 GPU 共享减少任务排队时间
- **GPU 利用率（GPU Utilization）**：通过弹性 Rollout 和跨任务共享填充 GPU 空转时间
- **显存效率（Memory Efficiency）**：多 LoRA 适配器整合减少基础模型的重复显存占用

### 局限性（Limitations）

1. **staleness 约束**：off-policy 数据被其他任务抢占 GPU 后，陈旧度（staleness）会增加，需要在算法层面设置合理的 staleness bound，否则可能影响训练收敛
2. **调度开销**：跨任务 GPU 共享需要全局调度器持续追踪集群状态，调度决策本身有延迟开销
3. **工程复杂性**：多任务共享 GPU 增加了系统状态管理的复杂度，调试难度高于单任务训练框架
4. **适用场景**：目前主要面向 RL 实验研究场景（多个并发 RL 实验），尚不清楚是否适合大规模生产部署
5. **AI 辅助开发**：README 明确注明"经过大量 AI 辅助开发，全程有人工指导和监督"，工程成熟度有待验证

---

## 💡 启发与我的想法

### 可借鉴之处

1. **阶段内优先级抽象**：将 RL 管线内的计算阶段按优先级分层（Level 0-6），是一个非常清晰的系统设计原则。训练相关阶段（Actor/Critic 更新）优先级高，数据采集阶段（Rollout）优先级低且可抢占——这与 OS 中将 I/O-bound 任务设为低优先级的思路完全一致。

2. **Rollout 弹性是核心杠杆**：在当前主流 RL 训练框架（PPO、GRPO）中，Rollout 采样往往是最大的时间开销，且是 memory-bound 推理任务（可以随时暂停）。将 Rollout 设计为弹性可抢占阶段，是提升 GPU 利用率的最高性价比抓手。

3. **多 LoRA 合并训练的工程价值**：类似 S-LoRA 在推理侧的洞察，在训练侧将多个 LoRA 适配器整合到同一基础模型上并发训练，对超参搜索实验（大量 LoRA rank/learning rate 探索）的显存节省效果显著。

4. **受 ROLL 启发的跨任务 Partial Overlap**：ROLL 将 Partial Overlapping 应用于单任务管线内部（训练与 Rollout 时间重叠），RLix 将这一思想拓展到多任务维度（任务 A 的 Rollout 空档给任务 B 用）。这是一个自然而有价值的工程推广。

### 改进空间

1. **缺乏公开 Benchmark 数据**：RLix 目前是纯工程项目，没有配套论文或系统性性能评测。如果能在 veRL/OpenRLHF 的标准 benchmark（如 Llama 7B/70B + PPO）上发布对比数据，说服力会大幅提升。

2. **staleness 的理论保证不清晰**：README 提到维持 staleness bound，但没有给出具体算法描述——在 off-policy 比例较高时，如何保证训练收敛性？建议参考 IMPALA 或 V-trace 的 staleness 理论分析。

3. **与 vLLM Speculative Decoding 结合**：Rollout 采样目前使用标准自回归推理，如果集成 vLLM 的推测解码，可以在 Rollout 阶段进一步提升 GPU token/s，减少 Rollout 成为瓶颈的频率。

4. **多任务间公平性调度**：当前 README 描述的优先级层次主要面向单任务内部阶段，多任务之间的调度公平性（某个任务长期借不到 GPU）需要更细致的设计，类似 Tiresias 或 Gandiva 中的公平队列机制。

### 下一步 Action

- [ ] 实际部署 RLix，测试 2+ 个并发 PPO 任务的 GPU 利用率提升效果
- [ ] 对比 RLix 和 veRL 在多任务 RL 实验场景下的调度效率
- [ ] 研究 staleness bound 的具体实现（查看源码 `scheduler.py`）
- [ ] 关注 rlops 组织后续是否发布技术报告或论文

