---
title: "Online Experiential Learning (OEL)"
tags:
  - online-learning
  - LLM
  - continual-learning
  - context-distillation
  - deployment
---

## 🧠 核心摘要 (TL;DR)
> 现有大模型改进范式依赖离线训练，导致实际部署期间积累的大量交互经验被白白浪费。OEL（Online Experiential Learning）提出一个**无需奖励模型、无需访问用户端环境**的在线学习框架：先从交互轨迹中提取可转移的体验知识，再通过 on-policy context distillation 将知识内化到模型参数，实现"部署即学习"的良性循环，在 token 效率和任务准确率上均持续提升。

## 🏷️ 元数据
- **论文标题**: Online Experiential Learning for Language Models
- **发表机构/会议**: Microsoft Research（arXiv）
- **年份**: 2026
- **链接**: [[OEL_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #online-learning #context-distillation #continual-learning #LLM #deployment
- **前置知识 (Prerequisites)**: [[KV稀疏调研]]
- **相关节点 (Related Notes)**: [[AttentionRes]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

当前改进大型语言模型的主流范式是**离线训练（Offline Training）**：收集人工标注数据或在模拟环境中生成合成数据，再以监督微调（SFT）或强化学习（RLVR）更新模型参数。这一范式存在三个根本性缺陷：

1. **部署经验白白流失**：模型在实际用户交互中积累的海量轨迹（成功的策略、失败的尝试、用户反馈）被完全丢弃，无法用于持续改进。
2. **奖励模型依赖**：主流 RLVR 方法（如 DeepSeek-R1、DAPO）需要可验证奖励函数，而现实场景中大多数任务（文本写作、对话、策略游戏）无法轻易定义标量奖励。
3. **环境访问假设不现实**：传统 RL 训练需要在训练侧模拟用户环境；但实际部署中，用户环境（游戏状态、工具接口、对话历史）往往只存在于客户端，服务器侧无法直接访问。

Silver & Sutton (2025) 在"Welcome to the Era of Experience"中指出，模型在与真实世界的交互中所获得的体验，将是超越人类数据瓶颈的关键路径，但现有框架尚未充分挖掘这一潜力。

### 论文的切入点

OEL 的核心洞察是：**体验知识（Experiential Knowledge）可以从已完成的轨迹中提取，并以上下文形式传递给模型，再通过 on-policy 蒸馏内化到参数中**——整个过程无需访问用户侧环境，也不依赖奖励模型。

---

### 相关工作

#### 先驱/奠基工作

**上下文蒸馏（Context Distillation）**

- **Learning by Distilling Context** (Snell et al., 2022; arXiv:2209.15189)：最早系统提出上下文蒸馏范式——将放在 context 中的辅助信息（指令、示例）通过训练内化到模型参数，使模型无需 context 也能复现其效果。在 SPIDER Text-to-SQL 数据集上取得 9% 提升。
- **MiniLLM: On-Policy Distillation of Large Language Models** (Gu et al., 2024; arXiv:2306.08543, ICLR 2024)：提出以**反向 KL 散度**替代正向 KL 散度训练蒸馏学生模型，并引入 on-policy 思路，防止学生在教师分布低概率区域产生高置信估计。MiniLLM 为 OEL 的知识整合阶段提供了核心损失函数设计。
- **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** (Agarwal et al., 2024; ICLR)：从自生成的错误样本出发做 on-policy 蒸馏，强调模型与自身分布的一致性在减少遗忘方面的重要性。

**基于体验的 Agent 学习**

- **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., 2023; arXiv:2303.11366, NeurIPS 2023)：提出 Agent 在每轮任务后通过语言文本反思自身错误，并将反思结果保存在工作记忆中用于后续决策。Reflexion 是 OEL 知识提取思路的直接先驱：都是从轨迹中提炼结构化知识，区别在于 Reflexion 是 in-context 的，OEL 将知识进一步内化到参数。
- **ExpeL: LLM Agents are Experiential Learners** (Zhao et al., 2024; AAAI 2024)：Agent 自主从训练任务的经验中提取自然语言知识并存入知识库，推理时调用以辅助决策。ExpeL 与 OEL 同样基于"从经验中提取知识"的理念，但同样不更新模型参数。
- **Agent Learning via Early Experience** (Zhang et al., 2025; arXiv:2510.08558)：提出以 Agent 自身生成的"早期经验"（无需专家数据）作为监督信号，引入隐式世界建模和自反思两种策略，跨 8 个多样环境验证了泛化能力，是 OEL 的重要平行工作。

#### 对比方法：RL 路线

- **DeepSeek-R1** (Guo et al., 2025; arXiv:2501.12948)：用 GRPO 做大规模 RL 训练，在可验证数学推理任务上取得突破性结果，但依赖标量奖励函数和大规模环境模拟，与 OEL 的无奖励设计形成对比。
- **DAPO** (Yu et al., 2025; arXiv:2503.14476)：开源大规模 LLM RL 训练系统，引入 Decoupled Clip 和 Dynamic Sampling 等技术，在 AIME 2024 上达到 50 分，同样依赖可验证奖励。
- **DeepSeekMath / GRPO** (Shao et al., 2024; arXiv:2402.03300)：引入 Group Relative Policy Optimization，在数学推理上接近 GPT-4 水平，是主流 RLVR 路线的代表。
- **Cogito, Ergo Ludo** (Wang et al., 2025; arXiv:2509.25052)：LLM Agent 在文本游戏中通过规则归纳和策略摘要两个模块迭代改进，无需标量奖励，与 OEL 设计理念接近。

#### 相关：遗忘与 On-Policy 的关系

- **RL's Razor** (Shenfeld et al., 2025; arXiv:2509.04259)：理论证明 on-policy RL 隐式偏向 KL 距离最小的解，因此比 SFT 遗忘更少；这为 OEL 选择 on-policy context distillation 而非 SFT 提供了理论支撑。
- **Retaining by Doing** (Chen et al., 2025; arXiv:2510.18874)：在 Llama 和 Qwen 多个模型上系统比较 SFT vs RL 的遗忘程度，确认 RL 在保持基础能力方面优于 SFT，关键因素是 on-policy 数据的使用。
- **On-Policy Context Distillation** (Ye et al., 2026; arXiv:2602.12275)：OEL 的直接前序工作，由同一组作者提出。在数学推理、文本游戏和特定领域任务中，on-policy context distillation 超越了多个基线，OEL 在此基础上增加了在线知识提取环节。

#### 评测工具

- **TextArena** (Guertler et al., 2025; arXiv:2504.11442)：开源文本游戏竞技平台，包含 57+ 游戏环境，支持单人/多人 LLM Agent 评测，OEL 使用其中的 Frozen Lake 等任务作为主要实验场景。
- **IFEval** (Zhou et al., 2023; arXiv:2311.07911)：指令遵循能力评测基准，OEL 用它验证在线学习后模型是否保持分布外（OOD）性能。


## ⚙️ 核心设计与机制

### 整体框架

OEL 构建了一个"部署-学习"闭环，整体流程如下：

```
用户端 (User Side)
  ↓ 真实交互轨迹 τ
知识提取模块 (Knowledge Extraction)
  ↓ 体验知识 e
知识整合模块 (Knowledge Consolidation)
  ↓ 参数更新 Δθ
服务器端模型 (Server Side)
  ↓ 改进后的模型
用户端 → （下一轮）
```

**关键约束**：知识整合过程完全在**服务器侧**完成，不需要访问用户侧环境。用户侧只需上传匿名化的交互轨迹，服务器侧通过"单轮回滚"（single-step rollout）模拟部分轨迹。

---

### 第一阶段：体验知识提取（Experiential Knowledge Extraction）

给定从用户侧收集的若干交互轨迹 $\tau_1, \tau_2, \ldots, \tau_n$，OEL 以**递归累积**的方式提取体验知识：

$$e_i = [e_{i-1};\; e'_i], \quad e'_i = \text{KnowledgeExtract}(\tau_i, e_{i-1})$$

其中 $e_0 = \emptyset$，每步生成的新知识 $e'_i$ 被追加拼接到已有知识库中。知识提取本身由当前模型 $\pi_\theta$ 完成（zero-shot prompt，格式为自然语言摘要或规则总结），无需额外训练模型。

**为什么提取知识而非直接使用轨迹？**论文在消融实验中验证：将"提取的知识"与"原始轨迹"分别作为上下文提供给教师模型，前者的 context distillation 效果显著优于后者。原因是原始轨迹包含大量冗余（无效步骤、重复尝试），而提取的知识是结构化、可泛化的"经验精华"。

---

### 第二阶段：知识整合（Knowledge Consolidation via On-Policy Context Distillation）

**目标**：将体验知识 $e$ 从上下文（context）内化到模型参数，使模型无需 $e$ 也能表现出知识加持后的行为。

**核心损失函数**（反向 KL 散度，token 级）：

$$\mathcal{L}(\theta) = \sum_{t} D_{\text{KL}}\!\bigl(\pi_\theta(\cdot \mid e, x, y_{<t}) \;\big\|\; \pi_\theta(\cdot \mid x, y_{<t})\bigr)$$

其中：
- 教师（Teacher）：$\pi_\theta(\cdot \mid e, x, y_{<t})$ — 以体验知识 $e$ 和输入 $x$ 为条件的输出分布（**参数冻结**，使用知识生成时的快照）
- 学生（Student）：$\pi_\theta(\cdot \mid x, y_{<t})$ — 不使用体验知识的输出分布（**参数更新**）
- 优化方向：最小化学生与教师之间的 KL 散度，即让模型"吸收"体验知识后即使没有 context 也能表现出相同输出

**On-Policy 的关键设计——单轮回滚（Single-Step Rollout）**：

传统 context distillation 使用离线数据（轨迹 $\tau$ 中的 token 序列）直接计算损失，这等价于 off-policy 更新，可能导致**灾难性遗忘**。OEL 的创新在于：

1. 采样一个部分轨迹前缀 $x$（可以是用户问题，也可以是轨迹中的某个中间状态）
2. 让**当前模型** $\pi_\theta$ 从该前缀自主生成响应 $y \sim \pi_\theta(\cdot \mid x)$
3. 以 $(x, y)$ 为基础计算知识蒸馏损失

这样，损失函数中的序列 $y$ 始终来自当前策略分布，保证了 on-policy 特性，使 KL 距离最小化方向与参数更新始终保持一致，从而大幅降低遗忘风险。

**词汇截断优化**：实际计算中仅使用 Top-256 词汇 token 计算 KL 散度，大幅降低内存和计算开销，同时不影响整体训练质量（概率质量集中在高概率 token）。

---

### 两阶段迭代与飞轮效应

OEL 框架设计为多轮迭代：
```
第 k 轮：
  1. 使用当前模型 πθ 部署，收集交互轨迹 τ⁽ᵏ⁾
  2. 提取体验知识 e⁽ᵏ⁾
  3. On-policy context distillation → 更新 θ⁽ᵏ⁺¹⁾
  4. 用 θ⁽ᵏ⁺¹⁾ 部署，收集更多（更优质的）轨迹 τ⁽ᵏ⁺¹⁾
```

随着模型能力提升，其生成的轨迹质量也随之提高（更多成功路径、更高效的策略），进而为下一轮知识提取提供更丰富的素材，形成**"越学越好、越好越有得学"**的飞轮效应。

---

## 📊 实验与性能评价

### 实验设置

- **主要实验任务**：Frozen Lake（TextArena 文本游戏环境），要求 Agent 在格子世界中导航到目标格子而不踩到冰洞，需要空间推理和策略规划
- **评估指标**：Pass Rate（任务成功率）、平均响应长度（衡量 token 效率）、IFEval 准确率（衡量分布外能力保持）
- **模型**：Qwen3-1.7B、Qwen3-4B、Qwen3-8B
- **对比方法**：
  - **Off-Policy Context Distillation**：使用相同知识，但以离线轨迹 token 计算损失
  - **SFT on Trajectories**：直接在原始成功轨迹上做监督微调
  - **Direct Trajectory as Context**：将原始轨迹（而非提取的知识）作为上下文给教师

### 主要结果

**任务准确率（Frozen Lake Pass Rate）**：

| 迭代轮次 | Qwen3-1.7B | Qwen3-4B | Qwen3-8B |
|---------|-----------|---------|---------|
| 第 0 轮（基线） | ~7.5% | 更高 | 更高 |
| 第 1 轮（OEL） | ~18%+ | 持续提升 | 持续提升 |
| 第 2、3 轮 | 继续增长 | 继续增长 | 继续增长 |

三轮迭代均呈现**单调提升**趋势，无论模型大小。

**Token 效率**：经过三轮迭代后，Qwen3-1.7B 的平均响应长度降至初始值的约 70%，说明模型学会了更简洁高效地解决问题，而非靠"长思维链"堆砌token。

**OOD 性能保持（IFEval）**：
- **OEL（on-policy）**：IFEval 准确率接近初始模型，遗忘极少
- **Off-Policy 方法**：IFEval 准确率明显下降，遗忘显著

这验证了 on-policy 设计在保持基础能力方面的核心作用，与 RL's Razor 和 Retaining by Doing 等理论分析一致。

**消融实验**：

| 组件 | 去除后效果 |
|------|----------|
| 知识提取（使用原始轨迹） | 提升幅度明显减小，说明"知识 > 轨迹" |
| On-Policy 设计（改为 Off-Policy） | 遗忘加剧，OOD 性能下降 |
| 递归累积（改为单次提取） | 多轮迭代后提升减缓 |

### 局限性 (Limitations)

1. **知识提取质量依赖初始模型能力**：若基础模型过弱，提取的体验知识可能不准确或不可靠，导致学习效率低下
2. **轨迹来源单一**：当前实验在文本游戏环境中，游戏规则结构清晰；在更复杂的开放域对话场景中，轨迹质量和知识可提取性尚未验证
3. **隐私与安全**：在线学习依赖真实用户轨迹，如何在数据匿名化和学习效率间取得平衡，论文未深入讨论
4. **知识累积爆炸**：随着迭代轮次增加，$e_i$ 的长度线性增长，可能超出模型上下文窗口限制；论文未提出压缩方案

---

## 💡 启发与我的想法

### 可借鉴之处

1. **无奖励在线学习范式**：OEL 证明了不依赖标量奖励也能持续改进——这对无法设计奖励函数的场景（如开放域对话、创意写作）极具参考价值。"以自然语言形式的体验知识代替奖励信号"是一个值得推广的设计哲学。

2. **On-Policy 的重要性**：实验清晰展示了 on-policy 设计对遗忘的抑制效果。在任何需要持续更新模型的场景（continual learning、多任务学习），保持 on-policy 特性应作为首选设计原则。

3. **知识提取 vs 原始轨迹**：OEL 的消融实验证明，"提炼后的知识"显著优于"原始经验"作为上下文。这对 RAG（Retrieval-Augmented Generation）、长文本压缩等任务有重要启示：与其存储原始文档，不如存储对其的结构化摘要。

4. **两阶段分离架构**：知识提取发生在用户侧（或数据收集阶段），知识整合发生在服务器侧，架构分离使隐私保护和部署效率都得到照顾。

### 改进空间

1. **知识库的压缩与索引**：当前知识累积方式（线性拼接）会导致 context 无限增长。可引入知识合并（knowledge merge）机制，检测冗余知识并压缩，或参考 InfiniteICL (arXiv:2504.11442) 的 LSTM 风格压缩方案。

2. **与 RL 的结合**：OEL 目前不依赖奖励，但若任务场景中存在部分可验证奖励（如代码执行结果），可将 OEL 的知识提取与 RLVR 的奖励信号结合，形成"有奖励时强化学习、无奖励时体验学习"的混合框架。

3. **多智能体协同学习**：多个模型实例的交互轨迹可相互共享——一个 Agent 的体验知识可能对另一个 Agent 有帮助，类似分布式经验回放机制。

4. **知识遗忘与选择性更新**：并非所有体验知识都应被永久内化（某些经验可能是偶然成功或环境特有的）。引入知识置信度评估，选择性地进行参数更新，可提高泛化能力。

### 下一步 Action
- [ ] 阅读前序工作 On-Policy Context Distillation (arXiv:2602.12275)，深入理解 OEL 的蒸馏损失设计细节
- [ ] 关注 OEL 在更复杂任务（代码生成、多轮对话）上的 follow-up 工作
- [ ] 思考 OEL 的知识提取模块是否可以与 KV Cache 稀疏化结合（减少长 context 知识蒸馏时的显存占用）

