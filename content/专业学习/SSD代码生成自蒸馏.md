---
title: "SSD：简单自蒸馏"
tags:
  - 代码生成
  - 自蒸馏
  - LLM后训练
  - 采样策略
---

## 🧠 核心摘要 (TL;DR)
> SSD（Simple Self-Distillation）通过一个令人惊讶的简单方案提升 LLM 代码生成能力：用高温或截断调整后的采样配置从模型自身生成训练数据，然后直接在这批**未经验证的原始输出**上做监督微调——无需验证器、无需教师模型、无需强化学习。Qwen3-30B-Instruct 在 LiveCodeBench v6 上的 pass@1 从 42.4% 提升至 55.3%（+12.9 pp），难题上提升幅度高达 +15.3 pp。

## 🏷️ 元数据
- **论文标题**: Embarrassingly Simple Self-Distillation Improves Code Generation
- **发表机构/会议**: arXiv 2026
- **年份**: 2026
- **作者**: Ruixiang Zhang, Richard He Bai, Huangjie Zheng, Navdeep Jaitly, Ronan Collobert, Yizhe Zhang
- **链接**: [[SSD_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #代码生成 #自蒸馏 #自训练 #采样策略 #llm后训练 #token分布
- **前置知识 (Prerequisites)**: [[KV稀疏调研]]
- **相关节点 (Related Notes)**: [[AttentionRes]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈

大语言模型在代码生成任务上的后训练（post-training）通常依赖以下三类外部信号：

1. **执行反馈（Execution-based verification）**：如 ReST^EM（Singh et al., 2024）、AlphaCode（Li et al., 2022）等，需要能运行代码并判断对错的执行环境和过滤管线；
2. **强化学习（RL-based post-training）**：如 DeepSeek-R1（DeepSeek-AI, 2025）、GRPO（Shao et al., 2024）等，需要定义可验证的奖励信号；
3. **教师模型蒸馏（Teacher-guided distillation）**：如 Hsieh et al. (2023) 的合成数据管线，需要更大/更强的教师模型生成高质量样本。

这些方案的共同代价是：**依赖外部基础设施**（执行沙箱、奖励模型、更大教师），在实际部署中成本高昂，且在无法运行代码的场景（如某些受限环境）下根本无法使用。

更深层的问题是：即便没有任何验证信号，模型自身的输出分布中是否已经包含足够的学习信号？

### 论文的切入点

SSD 给出肯定答案，并提出了极为简洁的方案：**只需改变采样温度和截断参数，就能从模型自身的输出中产生有效学习信号**。这个发现触及了一个深层问题——真正推动自蒸馏收益的机制是什么？作者将其归结为 token 分布的**精确度-探索冲突（Precision-Exploration Conflict）**，并给出了信息论层面的形式化分析。

### 相关工作

#### 先驱/奠基工作

| 论文 | 核心贡献 | 与本文关系 |
|------|----------|-----------|
| Hinton et al., 2015 — *Distilling the Knowledge in a Neural Network* | 知识蒸馏的奠基工作，用软标签从大模型压缩知识到小模型 | SSD 属于自蒸馏，无需教师模型 |
| Kim & Rush, 2016 — *Sequence-Level Knowledge Distillation* (EMNLP 2016) | 序列级蒸馏，将蒸馏思想扩展到生成模型 | SSD 继承序列级自训练思路 |
| Furlanello et al., 2018 — *Born Again Neural Networks* (ICML 2018) | 同参数量的学生模型比教师表现更好，奠定自蒸馏有效性基础 | SSD 是代码生成版的 Born-Again 框架 |
| Zelikman et al., 2022 — *STaR: Bootstrapping Reasoning with Reasoning* (NeurIPS 2022) | 从正确答案反推推理链，迭代自训练推理能力 | STaR 需要正确答案作为过滤信号，SSD 无需任何验证 |
| Holtzman et al., 2020 — *The Curious Case of Neural Text Degeneration* (ICLR 2020) | 提出 Nucleus Sampling（top-p 采样），通过截断低概率 token 改善文本质量 | SSD 的截断操作与 nucleus sampling 思路相近，但 SSD 用于训练而非推理 |
| Hewitt et al., 2022 — *Truncation Sampling as Language Model Desmoothing* (EMNLP Findings 2022) | 将截断采样理解为"去平滑化"，提出 η-sampling | SSD 的理论分析直接建立在 Hewitt 的截断即去平滑框架之上 |

#### 直接对比方法

| 论文 | 方法 | 对比优势 |
|------|------|---------|
| Singh et al., 2023 — *Beyond Human Data: ReST^EM* | EM 式迭代自训练，需要二元执行反馈 | SSD 无需执行环境 |
| DeepSeek-AI, 2025 — *DeepSeek-R1* | 基于 GRPO 的 RL 后训练，需可验证奖励 | SSD 零外部信号 |
| Shao et al., 2024 — *DeepSeekMath* (GRPO) | Group Relative Policy Optimization，RL 提升数学推理 | SSD 用极简监督微调实现类似目标 |
| Chen et al., 2024 — *SPIN (Self-Play Fine-Tuning)* | 自对弈机制，区分模型生成与人类标注数据 | 仍需人类标注作为正例锚点，SSD 完全无监督 |
| Yuan et al., 2024 — *Self-Rewarding Language Models* (ICML 2024) | 模型自评估提供奖励信号，迭代 DPO | 需要 LLM-as-Judge，SSD 不需要 |
| Wei et al., 2022 — *Chain-of-Thought Prompting* | few-shot CoT 提升推理，无需训练 | SSD 是训练时的改变，比 CoT 更持久 |

#### Follow-up 与相关系统

| 论文 | 内容 | 关系 |
|------|------|------|
| Jain et al., 2024 — *LiveCodeBench: Holistic and Contamination Free Evaluation* | 持续更新的代码生成评测基准，从 LeetCode/AtCoder/Codeforces 收集题目 | SSD 的主要评测基准 |
| Li et al., 2022 — *Competition-Level Code Generation with AlphaCode* | 大规模代码生成+执行过滤+聚类，达到 Codeforces top 54% | SSD 无需执行过滤但可在类似场景应用 |
| Kwon et al., 2023 — *vLLM (SOSP 2023)* | PagedAttention 高吞吐推理系统 | SSD 实验使用 vLLM v0.11.0 进行采样 |
| Wang et al., 2023 — *Self-Consistency Improves Chain of Thought* (ICLR 2023) | 多路采样取多数投票，提升推理准确率 | 与 SSD 的 pass@k 测量方法相关 |
| Fan et al., 2018 — *Hierarchical Neural Story Generation* | 引入 top-k 采样概念 | SSD 采样配置的基础技术 |



## ⚙️ 核心设计与机制

### 方法总览：三阶段流程

SSD（Simple Self-Distillation）的操作极为简洁，分三步：

```
1. Data Synthesis（数据合成）
   用温度 T_train 和截断参数 ρ_train 从基础模型采样，
   生成 K 条原始输出（不验证正确性）

2. Training（训练）
   用标准监督学习（cross-entropy）在这批原始输出上微调基础模型

3. Inference（推理）
   用评测温度 T_eval 和截断参数 ρ_eval 部署微调后的模型
```

其中 **T_train ≠ T_eval**，这是方法起效的关键：训练采样温度高于推理温度，使模型学到的分布在正式推理时更集中。

### 精确度-探索冲突（Precision-Exploration Conflict）

作者提出了理解 SSD 核心机制的关键概念框架：

代码中的每个生成位置可分为两类：

**Locks（锁定位）**：分布高度集中，但存在少量"干扰尾巴"（distractor tail）。典型例子是语法约束位置（括号闭合、关键字拼写），这里几乎只有一个正确答案，但分布的尾部可能包含低概率错误候选。**在此类位置，需要压制干扰尾巴，提升精确度。**

**Forks（分叉位）**：分布均匀分散，多种候选都是合理的（真正的语义分叉点，如不同的算法路径选择）。**在此类位置，需要保留多样性，允许模型探索。**

问题在于：**固定的推理温度无法同时满足两种需求**。低温有助于 Locks，却会抑制 Forks 的探索；高温有利于 Forks，却会激活 Locks 的干扰尾巴。

SSD 通过训练时采样的方式**以上下文感知的（context-dependent）方式重塑 token 分布**：

$$\mathcal{L}_{\text{SSD}} = \underbrace{\mathcal{L}_{\text{support compression}}}_{\text{截断：压缩支撑集}} + \underbrace{\mathcal{L}_{\text{within-support reshaping}}}_{\text{温度：重塑支撑集内概率}} + \underbrace{\mathcal{L}_{\text{alignment}}}_{\text{对齐基础模型}}$$

这种分解解释了 SSD 为何能从未验证的输出中产生学习信号——它的作用不是"学习正确答案"，而是**改变 token 概率的形状**。

### 关键参数：训练采样配置

实验揭示了两个关键采样参数对效果的影响：

**训练温度 T_train**：
- 适当高温（如 T_train = 1.5 ~ 2.0）效果最佳，因为可以激活更多 Fork 位的多样性
- 超高温（T_train = 2.0 且无截断）会导致约 62% 的输出不含可提取代码，但模型仍能从中收益（pass@1 提升 +5.7 pp），说明收益来自分布重塑而非学习正确解

**训练截断参数 ρ_train**（nucleus truncation）：
- 截断操作对应 Hewitt et al. (2022) 的"去平滑化"（desmoothing）效果
- 截断 + 高温组合能最大化 Lock 位的干扰压制 和 Fork 位的多样性保留

### 为何温度调整单独不够用

一个关键的消融实验：

仅在推理时调整温度（不进行 SSD 训练），pass@1 在温度范围 [0.2, 2.0] 内只有 1.5~3.0 pp 的波动。而 SSD 在 30B 模型上的提升高达 +12.9 pp，远超最优温度下的推理基线 +11.8 pp。

原因在于：**推理温度只能在原始分布上做线性缩放**，无法打破固定点（fixed-point）问题——原始分布中 Locks 和 Forks 的混合结构不变；SSD 训练则从根本上**重塑**了这种混合结构。

### 无需验证的理论基础

直觉上令人困惑的一点：既然 SSD 训练用的是未经验证的输出（其中包含大量错误代码），为什么还能提升？

作者的解释是：SSD 的学习信号不来自"正确 vs 错误"，而来自**温度和截断配置引起的分布偏移**。这等价于对原始模型进行了一种特殊形式的"分布外压缩"（out-of-distribution compression）——训练样本与评测样本的分布差异促使模型重新校准 token 概率，而非记忆具体答案。

这与经典的自训练（self-training）失败模式形成对比：朴素自训练（直接在模型输出上训练，不改变采样配置）会收敛到固定点，因为模型已经对自己的输出分配最高概率，再训练无法产生梯度信号。



## 📊 实验与性能评价

### 主要实验结果

**评测基准**：LiveCodeBench v6（从 LeetCode、AtCoder、CodeForces 持续更新的代码题，抗污染）

**主要指标**：pass@1（随机采样一条输出，判断是否通过测试）、pass@5（采样 5 条，至少一条通过）

#### 跨模型一致性验证（pass@1 增量）

| 模型 | 基础线 pass@1 | SSD 后 pass@1 | 增量 |
|------|-------------|-------------|------|
| Qwen3-30B-Instruct | 42.4% | 55.3% | **+12.9 pp（+30.4% 相对提升）** |
| Qwen3-4B-Instruct | - | - | +7.5 pp |
| Qwen3-4B-Thinking | - | - | +3.3 pp |
| Llama-3.1-8B | - | - | +3.5 pp |

SSD 在 Qwen 和 Llama 两个系列、4B 到 30B 的不同规模模型上均有效，说明改进是**方法本身带来的，不依赖特定模型架构**。

#### 难易题分层分析（Qwen3-30B-Instruct）

| 题目难度 | 基础线 pass@1 | SSD 后 pass@1 | 增量 |
|---------|-------------|-------------|------|
| Easy（简单） | - | - | +6.5 pp |
| Hard（困难） | - | - | **+15.3 pp** |

**难题上的提升显著更大**，这与 Precision-Exploration Conflict 理论一致：难题中 Fork 位更多，SSD 通过保留多样性探索的收益更明显。

#### Pass@1 vs Pass@5 对比

Pass@5 的提升幅度**系统性地超过** pass@1 的提升，说明 SSD 并未压制模型的多样性——它在提高单次命中率的同时，也保留了多条正确路径的概率，这正是 Fork 位分布保留的体现。

#### 与推理时温度调整的对比（Qwen3-30B-Instruct）

| 方法 | pass@1 |
|------|--------|
| 基础模型（T=1.0） | 42.4% |
| 基础模型最优温度扫描 | ≈43.5~45.4%（+1.5~3.0 pp） |
| SSD（最优训练配置） | 55.3%（+12.9 pp） |
| SSD 超越最优温度基线 | **+11.8 pp** |

单纯调整推理温度远不足以达到 SSD 的效果。

#### 高温无截断的极端案例（Bad Data, Good Results）

| 训练配置 | 输出中无可提取代码比例 | pass@1 | 相比基础线增量 |
|---------|-------------------|--------|-------------|
| T_train=2.0，无截断 | ~62% | 48.1% | **+5.7 pp** |
| T_train=1.5，有截断 | 较低 | 55.3% | +12.9 pp |

即便训练数据中 62% 是"垃圾"（不含可提取代码），模型仍然获得了 +5.7 pp 的提升。这强力支持了 SSD 的学习信号来自**分布重塑而非正确答案**的理论。

### 对比基线（Baseline）

- **朴素自训练（T_train = T_eval，不改变采样配置）**：接近 fixed-point，几乎无提升，验证了"改变采样配置"是 SSD 起效的关键
- **温度推理调优（基础模型 + 温度扫描）**：最优情况下仅 +1.5~3.0 pp，说明推理时分布线性缩放的上限很低
- **ReST^EM（Singh et al., 2024）**：需要执行反馈，SSD 无需外部验证且效果更简洁

### 局限性（Limitations）

1. **评测范围**：论文主要测试 LiveCodeBench，未系统对比 HumanEval、APPS 等其他代码基准
2. **模型规模上限**：最大测试到 30B，超大规模（70B+）模型上的规律未知
3. **任务泛化性**：SSD 对代码任务的机制是否同样适用于数学推理、自然语言任务，论文未完整展开
4. **Thinking 模型收益较小**：Qwen3-4B-Thinking（已是推理增强型）的提升只有 +3.3 pp，说明对已经内化了"探索"能力的模型，SSD 的边际收益下降
5. **超参数敏感性**：T_train 和 ρ_train 的最优组合需要一定调优，不同模型的最优配置可能不同

## 💡 启发与我的想法

### 可借鉴之处

1. **"分布重塑"作为学习信号的通用性**：SSD 揭示了一个深刻洞察——不必依赖"答案是否正确"才能产生有效学习信号；改变生成时的分布形状本身就是一种监督信号。这个视角对设计其他无奖励自训练方法有参考价值。

2. **Precision-Exploration Conflict 框架的普适性**：这个框架不只适用于代码生成，代码中的"Locks vs Forks"类比可以推广到其他结构化生成任务（如 SQL 生成、程序合成、数学推导）。在这些任务中，固定解码温度同样面临同样的矛盾。

3. **极简基线的价值**：SSD 提醒研究者在引入复杂管线（执行反馈、RL、教师模型）之前，先充分挖掘最简单方案的上限。+12.9 pp 的提升来自一行代码的改动（改变采样温度），这本身就是一个值得深思的结果。

4. **Pass@5 > Pass@1 增量的观察**：这可以作为判断一个后训练方法是否"过拟合"（牺牲多样性换取单次准确率）的指标。SSD 展示了多样性可以和准确率同时提升的可能性。

### 改进空间

1. **与执行反馈的结合**：SSD 是无验证的极简版本，与 ReST^EM 风格的执行过滤结合（SSD 初始化 + 执行验证微调）可能进一步提升上限；

2. **自适应采样配置**：当前 T_train 和 ρ_train 是全局固定的，能否根据模型当前在不同 Fork/Lock 位的不确定性动态调整？这与 KV Cache 稀疏化中自适应选择关注位置的思路有异曲同工之处；

3. **多轮迭代**：论文主要报告单轮 SSD 的结果，多轮迭代（用 SSD 微调后的模型再次生成训练数据）的稳定性和收敛性值得研究；

4. **其他模态**：探索 SSD 类思路在多模态代码生成（如图-代码、自然语言规范-代码）中的应用。

### 下一步 Action
- [ ] 复现 SSD 在 Qwen3-4B-Instruct 上的结果，验证 +7.5 pp 的数字
- [ ] 阅读 Hewitt et al. (2022) 截断即去平滑的理论推导，深入理解 SSD 的信息论基础
- [ ] 思考 SSD 中的"Forks"概念与 KV 稀疏调研中注意力分叉位置的潜在联系


