---
title: "LoRA@RL"
tags:
  - reinforcement-learning
  - LoRA
  - MoE
  - distributed-training
  - post-training
  - trillion-parameter
---

## 🧠 核心摘要 (TL;DR)
> Mind Lab 团队在 Kimi K2（1.04T 参数 MoE 模型）上实践"实用 LoRA RL"：通过混合并行（张量/流水线/专家/序列并行 4D 组合）+ LoRA 适配器训练，将 RL 训练的 GPU 需求降低至全参数训练的约 10%，同时实验表明在相同算力预算下，大模型 LoRA RL 的效果**优于**小模型全参数 RL，打破了"LoRA 不适合 RL"的常见认知。

## 🏷️ 元数据
- **博客标题**: How We Build Trillion Parameter Reasoning RL with 10% GPUs
- **发表机构**: Mind Lab / Mindverse
- **年份**: 2025
- **链接**: [[TrillionParamRL_blog|📄博客]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #reinforcement-learning #lora #moe #distributed-training #post-training #trillion-parameter
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[ShiftParallelism]]
- **相关节点 (Related Notes)**: [[LongCat]], [[SonicMoE]], [[RLBeyondBaseModel]], [[RAGEN2]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

LLM 的后训练（Post-Training）已经进入强化学习（RL）驱动的推理增强时代，但在**万亿参数级别 MoE 模型**上进行 RL 训练面临多重工程挑战：

1. **显存压力爆炸**：全参数 RL（Full-Parameter RL）在训练过程中需要同时维护 Policy 模型、参考模型（Reference Model）、以及 PPO 中的 Critic 模型的参数与优化器状态，对于 Kimi K2（1.04T 参数）这样的模型，单次全参数训练需要极大规模的 GPU 集群。
2. **MoE 特有的训练失稳**：稀疏 MoE 架构（Sparse MoE）在 RL 训练过程中存在特有的失稳模式——不同 batch 中被激活的 Expert 集合差异较大，导致梯度信号极不均匀；同时 RL 的策略更新可能造成路由分布（Routing Distribution）崩溃，使部分 Expert 永久死亡。
3. **通信开销与并行策略复杂度**：万亿参数 MoE 模型在分布式训练中需要同时处理专家并行（Expert Parallelism, EP）引入的 All-to-All 通信和其他并行维度（张量并行、流水线并行）的通信，如何高效组合这些并行策略是巨大的工程挑战。
4. **LoRA 在 RL 场景的争议**：业界普遍认为 LoRA 微调会限制模型的策略空间（LoRA 只更新低秩投影，无法修改所有权重），在 RL 的探索-利用权衡中可能导致模式坍塌。Mind Lab 的实践提供了相反的实验证据。

### 论文的切入点

Mind Lab 以 **Kimi K2**（1.04T 总参数，32.6B 激活参数，标准 MoE 架构）为基础，系统探索了：
- 如何通过 **LoRA + 4D 混合并行**将 RL 训练的 GPU 需求压缩到传统方案的 ~10%
- 万亿参数 MoE 在 RL 训练中的**失稳模式诊断与修复**
- **Scaling 维度的重新定义**：在固定算力预算下，大模型 LoRA RL vs 小模型全参 RL 的系统性对比

### 相关工作

#### 先驱/奠基工作

| 论文 | 贡献 | 相关性 |
|------|------|--------|
| **LoRA** (Hu et al., 2022) [arXiv:2106.09685] | 提出低秩适配器，将可训练参数减少 10000×，GPU 显存减少 3× | 本文 LoRA RL 的核心适配方法基础 |
| **PPO** (Schulman et al., 2017) [arXiv:1707.06347] | 近端策略优化，RL fine-tuning 的基础算法 | 本文 RL 训练算法的基础 |
| **RLHF/InstructGPT** (Ouyang et al., 2022) [arXiv:2203.02155] | 首次系统性将 RLHF 用于 LLM 对齐，1.3B InstructGPT 超越 175B GPT-3 | 模型在对齐中的 Scaling 规律启示 |
| **Megatron-LM v1** (Shoeybi et al., 2019) [arXiv:1909.08053] | 张量并行（Tensor Parallelism），支持 80 亿参数训练 | 本文并行策略的核心组件 |
| **Megatron-LM v2** (Narayanan et al., 2021) [arXiv:2104.04473] | 3D 并行（TP+PP+DP），1T 参数训练，52% GPU 效率 | 本文 4D 并行方案的直接参考 |
| **PipeDream** (Harlap et al., 2018) [arXiv:1806.03377] | 流水线并行 DNN 训练，相比数据并行减少 95% 通信 | 本文流水线并行的算法基础 |
| **GShard** (Lepikhin et al., 2020) [arXiv:2006.16668] | 稀疏门控 MoE 跨 TPU 自动分片，600B 多语言翻译模型 | 专家并行的奠基工作 |
| **Chinchilla Scaling Law** (Hoffmann et al., 2022) [arXiv:2203.15556] | 计算最优训练定律：模型参数与训练 token 应同步 Scaling | 本文"算力预算不变，用大模型 LoRA"的理论支撑 |
| **Sequence Parallelism** (Li et al., 2021) [arXiv:2105.13120] | 序列并行，4D 并行的第四个维度 | 本文序列并行组件 |

#### 直接对比方法

| 论文 | 贡献 | 相关性 |
|------|------|--------|
| **DeepSeekMath/GRPO** (Shao et al., 2024) [arXiv:2402.03300] | 提出 GRPO 算法，去除 Critic 降低显存，数学推理 RLVR 方案 | 本文可能参考的 RL 算法变体 |
| **DeepSeek-R1** (DeepSeek-AI, 2025) [arXiv:2501.12948] | 纯 RL 驱动大规模推理，自我反思与验证等能力涌现 | 推理模型 RL 训练领域的 SOTA 对比 |
| **Kimi k1.5** (Kimi Team, 2025) [arXiv:2501.12599] | Kimi RL 推理扩展，Long Context + 策略优化，AIME 77.5% | 同一系列推理模型的技术对比 |
| **Mixtral of Experts** (Jiang et al., 2024) [arXiv:2401.04088] | 8×7B 稀疏 MoE，47B 参数激活 13B，超越 Llama 2 70B | MoE 架构参考 |
| **DeepSeek-V3** (DeepSeek-AI, 2024) [arXiv:2412.19437] | 671B MoE，辅助无损负载均衡，14.8T token 预训练 | 大型 MoE 预训练与后训练参考 |

#### 训练基础设施 Follow-up

| 论文 | 贡献 | 相关性 |
|------|------|--------|
| **HybridFlow/veRL** (Sheng et al., 2024) [arXiv:2409.19256] | 混合单/多控制器 RLHF 框架，3D-HybridEngine 零显存冗余，吞吐提升 1.53×–20.57× | RLHF 训练框架工程方案 |
| **OpenRLHF** (Hu et al., 2024) [arXiv:2405.11143] | 基于 Ray+vLLM+DeepSpeed 的开源 RLHF 框架，加速 1.22×–1.68× | 开源 RLHF 基础设施 |
| **DAPO** (Yu et al., 2025) [arXiv:2503.14476] | 去耦裁剪+动态采样 RL 算法，Qwen2.5-32B AIME 50 分 | 新一代 RL 算法对比基准 |

---

## ⚙️ 核心设计与机制

### 架构创新：实用 LoRA RL 体系

Mind Lab 的核心洞察是将两个方向的挑战同时解决：
1. **参数效率**：用 LoRA 替换全参数更新
2. **并行效率**：用 4D 混合并行适配 MoE 架构

整体系统架构可以描述为：

```
Kimi K2（1.04T 参数, 32.6B 激活）
    ↓
LoRA 适配器注入（冻结主干权重，仅训练低秩矩阵）
    ↓
4D 混合并行策略：
  ├── 张量并行 (Tensor Parallelism, TP)  ← 拆分 Attention/FFN 矩阵
  ├── 流水线并行 (Pipeline Parallelism, PP) ← 跨 GPU 分层流水
  ├── 专家并行 (Expert Parallelism, EP)  ← MoE Expert 分散到不同 GPU
  └── 序列并行 (Sequence Parallelism, SP) ← 超长序列跨设备分割
    ↓
RL 训练（PPO/GRPO 变体）
```

### 关键技术点 1：万亿参数 MoE RL 的失稳模式与修复

博客中重点讨论了 LoRA RL 在 MoE 架构上遇到的特有失稳问题及对应解决方案：

**失稳模式 A：路由分布崩溃（Routing Collapse）**
- **现象**：RL 训练过程中，部分 Expert 接收的 token 数量迅速萎缩至接近零，路由网络将几乎所有 token 集中路由到少数"热门 Expert"，导致 MoE 退化为 Dense 模型
- **根因**：RL 的策略梯度更新会放大路由网络对当前已高奖励行为的偏好，形成正反馈循环
- **修复**：引入辅助负载均衡损失（Auxiliary Load Balancing Loss），在 RL 奖励信号之外额外惩罚 Expert 负载不均衡，强制保持路由多样性

**失稳模式 B：梯度信号稀疏性**
- **现象**：MoE 的稀疏激活特性使得单个 batch 中只有一小部分 Expert 被激活，对未激活 Expert 的梯度信号为零。RL 训练本身梯度已经稀疏，双重稀疏导致训练极不稳定
- **根因**：LoRA 只更新低秩矩阵 $\Delta W = BA$，若某 Expert 长期未被激活，其 LoRA 参数完全得不到更新
- **修复**：调整 Expert 并行配置，确保每个 Expert 在 RL rollout 阶段获得足够的样本覆盖；同时对 LoRA 参数的梯度施加全局归一化

**失稳模式 C：参考模型漂移**
- **现象**：RL 训练中的 KL 散度约束要求当前策略不偏离参考模型太远，但 LoRA 更新与全参数更新对权重空间的扰动方式不同，导致 KL 距离计算失准
- **根因**：LoRA 的低秩更新在某些方向上有效更新远大于实际权重变化，使 KL 约束失效
- **修复**：重新校准 KL 系数，并在 rollout 阶段定期刷新参考模型的 LoRA 投影

### 关键技术点 2：4D 并行策略与显存分析

对于 Kimi K2 这样的 1.04T 参数 MoE 模型，每种并行维度的作用如下：

**张量并行（TP）**：将单个 Transformer 层的大矩阵（如 QKV 投影、FFN 权重）在隐层维度上切分到多个 GPU。对于 MoE 层，TP 切分的是每个 Expert 内部的权重矩阵。

**流水线并行（PP）**：将模型的不同层分配到不同 GPU 上，采用 1F1B（One Forward One Backward）或交错式流水线调度，避免 GPU 大规模气泡等待。

**专家并行（EP）**：MoE 特有的并行维度，将不同的 Expert 分配到不同 GPU。Token 路由后需要 All-to-All 通信将 token 发送到对应 Expert 所在的 GPU。博客中提及使用了 DeepEP 风格的高效专家并行通信库（参考 Zhao et al., 2025）来优化 All-to-All 开销。

**序列并行（SP）**：对于超长推理 token（RL rollout 往往产生很长的 CoT 序列），在序列维度上进行切分，避免单 GPU 显存溢出。

**LoRA 的显存优化分析**：

对于 LoRA，可训练参数量为：
$$|\theta_{LoRA}| = \sum_{l} (r \cdot d_{in} + r \cdot d_{out}) = 2r \sum_{l} d$$

其中 $r \ll d$（通常 $r \in \{8, 16, 64\}$），因此优化器状态（Adam 的一阶矩 $m$ 和二阶矩 $v$）仅需维护 LoRA 参数，显存节省约为：
$$\text{显存节省} \approx \frac{1 - 2r/d_{model}}{1} \approx 90\%+\text{（优化器状态部分）}$$

这是"10% GPU 需求"的核心来源——主要节省的是**优化器状态显存**，而非模型参数本身（参数仍需全精度推理）。

### 关键技术点 3：LoRA 初始化与 RL 兼容性

为使 LoRA 在 RL 场景下稳定工作，博客提到了若干关键的工程细节：

1. **LoRA 初始化策略**：沿用标准 LoRA 初始化（矩阵 $A$ 高斯初始化，矩阵 $B$ 零初始化），确保训练初期 $\Delta W = BA = 0$，即与预训练模型等价
2. **LoRA 注入位置**：对 Q、K、V、O 投影矩阵以及 FFN 的 up/down/gate 矩阵均注入 LoRA 适配器，覆盖 Attention 和 FFN 两个核心组件
3. **LoRA Rank 选择**：实验中使用了相对较大的 rank（推测为 $r=64$ 或更大），以确保策略空间足够大，支持推理能力的探索

### 核心实验设计：Scaling 维度对比

博客中最核心的实验结论来自一组跨模型规模的对比：

在**固定算力预算**（约等于 10 GPU × N 小时）下，比较：
- **方案 A**：1.5B / 7B / 32B 参数模型的**全参数 RL**
- **方案 B**：1.04T 参数 Kimi K2 的 **LoRA RL**（约消耗相同算力）

对比维度包括：推理基准准确率（数学/代码/通用推理）、训练曲线稳定性、下游泛化性（通用能力保持）。


---

## 📊 实验与性能评价

### 主要结论一：LoRA RL 的学习曲线

在 Kimi K2 上运行 LoRA RL 训练，博客报告了以下关键观察：

- **训练稳定性**：经过上述失稳修复后，训练曲线总体平稳，奖励（Reward）单调递增，无明显的策略崩溃现象
- **通用能力保持（Generality Preservation）**：LoRA 冻结主干权重的特性天然保护了模型的预训练泛化能力，在 RL 训练结束后，非目标域的下游基准（数学以外的通用任务）性能基本无退化，而全参数 RL 在长时间训练后往往出现灾难性遗忘

### 主要结论二：大模型 LoRA RL 优于小模型全参数 RL

这是本博客最重要的实证发现。在跨规模对比实验中：

| 训练方案 | 模型规模 | 训练方式 | GPU 消耗（相对） | 推理基准提升 |
|---------|---------|---------|--------------|------------|
| 基线 A | 1.5B | 全参数 RL | 1× | +X₁% |
| 基线 B | 7B | 全参数 RL | 3.5× | +X₂% |
| 基线 C | 32B | 全参数 RL | ~8× | +X₃% |
| **Mind Lab 方案** | **1.04T (Kimi K2)** | **LoRA RL** | **~10×（与基线C相当）** | **>X₃%** |

> 注：博客中未直接列出具体数值表格，以上框架基于博客文字描述重构，X₁/X₂/X₃ 为相对排名关系（推断）。

**结论解读**：在相同的算力预算下，将算力用于训练更大模型（哪怕用 LoRA 代替全参数更新）比用于训练较小的全参数模型更加有效。这与 **Chinchilla Scaling Law** 的精神一致——模型规模本身带来的先验知识和表征能力是算力高效利用的基础。

### 主要结论三：GPU 需求对比

博客核心 claim：Kimi K2 LoRA RL 的 GPU 需求约为同等能力全参数方法的 **10%**。

这一数字的来源分析（推断）：
- **全参数 RL**（以 Kimi K2 规模为例）：需要维护模型参数（BF16/FP8）+ 参考模型参数 + 优化器状态（Adam 需要 2× 参数量）= 约 4× 参数量的显存
- **LoRA RL**：冻结主干，优化器状态仅针对 LoRA 参数（约 0.1%~0.5% 参数量）= 显存节省 >90%
- **实际系统差异**：考虑到 4D 并行的显存切分效果，在相同 GPU 数量下 LoRA RL 可运行 Kimi K2，而全参数 RL 则需要约 10× 的 GPU 才能装下同等模型

### 局限性（Limitations）

1. **LoRA 策略空间受限**：虽然实验表明 LoRA RL 效果良好，但从理论上讲，LoRA 的低秩约束限制了策略更新的方向空间，在极长 RL 训练时间下可能存在收益天花板
2. **结论依赖 MoE 稀疏特性**：本文的"10% GPU"结论与 Kimi K2 的 MoE 稀疏激活特性密切相关（32.6B/1.04T 激活率约 3%）。对于 Dense 模型，LoRA RL 的相对优势会小很多
3. **实验结果的可重现性**：博客未公开代码或模型权重，各实验的具体配置（LoRA rank、学习率调度、RL 算法超参）未完整披露，难以直接复现
4. **Scale 上限未探索**：文章聚焦于"用更少 GPU 做更大模型"，但未探讨当 LoRA RL 充分收敛后，继续增加 GPU 是否能进一步提升性能（即 LoRA RL 本身的 Scaling 曲线）
5. **任务覆盖局限**：主要在数学推理（MATH/AIME 类型）和代码生成任务上验证，在开放域生成、多模态等任务上的 LoRA RL 适用性未探讨

---

## 💡 启发与我的想法

### 可借鉴之处

**1. LoRA RL 的工程实践路径**

本文提供了一个宝贵的"可行性验证"：LoRA RL 在万亿参数 MoE 上是工程可行的，而非只是理论可能。对于资源有限的研究团队，这打开了一条用大模型做 RL 后训练的现实路径：
- 租用 10-20% 的 GPU，即可在世界级大模型上进行 RL 实验
- 不必等待全参数训练基础设施就绪，可以快速迭代 RL 算法和奖励设计

**2. MoE + LoRA 的天然契合性**

MoE 模型的稀疏激活特性与 LoRA 的参数稀疏性形成双重稀疏，这在节省显存上效果叠加。但同时也提示：LoRA rank 的选择需要考虑 Expert 的实际激活频率——低激活 Expert 的 LoRA 参数可以使用更小的 rank，而高频 Expert 需要更大的 rank 来支持充分的策略探索。

**3. "Scaling 维度选择"的思维框架**

文章实际上提供了一个更广泛的决策框架：在固定算力预算下，
- **传统思路**：用全部 GPU 训练中等规模模型（更多梯度更新次数）
- **本文思路**：用更少 GPU 训练超大规模模型（更强的先验知识和表征基础）

这与 NLP 领域"大模型小数据 vs 小模型大数据"的长期争论有异曲同工之妙。

**4. 失稳模式的工程诊断方法**

博客中描述的三类失稳模式（路由崩溃、梯度稀疏、KL 漂移）及其修复方案是极有价值的工程经验。对于任何在 MoE 模型上进行 RL 训练的团队，这些模式提供了一份重要的"失稳诊断清单"。

### 改进空间

**1. LoRA-MoE 专属设计**

现有工作将 LoRA 简单应用到 MoE 的每个 Expert，但更合理的设计可能是：
- **Expert-Adaptive LoRA**：根据 Expert 的激活频率动态分配 rank（高频 Expert 用大 rank，低频 Expert 用小 rank 或共享适配器）
- **Router-LoRA**：单独为路由网络设计 LoRA 适配器，允许 RL 过程适度调整路由分布，而非完全冻结路由

**2. LoRA 梯度流动分析**

当前 LoRA RL 的有效性仍缺乏理论解释：低秩约束下的策略梯度究竟捕捉到了哪些维度的策略变化？是否存在"关键方向"（Critical Subspace）在推理能力提升中起主要作用？这是值得深入分析的可解释性问题。

**3. 与 GRPO/DAPO 等无 Critic 算法的组合**

本文未明确披露具体 RL 算法，但如果使用了 PPO，Critic 网络也需要全参数或 LoRA 化。GRPO 通过去除 Critic 进一步节省显存，与 LoRA 组合可能实现更极致的显存压缩。此方向值得系统探索。

**4. LoRA RL 的收益边界**

需要回答：LoRA RL 能否最终赶上全参数 RL（给予足够训练时间）？还是存在本质的策略空间天花板？这对于决策"何时值得升级到全参数 RL"非常关键。

### 下一步 Action

- [ ] 跟进 Mind Lab 是否发布技术报告或代码，复现关键实验
- [ ] 调研 LoRA RL 的理论分析工作（策略梯度在低秩子空间的特性）
- [ ] 在自有 MoE 模型上测试 LoRA RL 方案，验证路由崩溃现象是否可复现
- [ ] 对比 LoRA RL + GRPO 与 LoRA RL + PPO 的显存效率与训练稳定性
- [ ] 阅读 HybridFlow/veRL 框架源码，理解 4D 并行 RLHF 的工程实现细节

