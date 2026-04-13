---
title: "RAGEN-2"
tags:
  - reinforcement-learning
  - agentic-rl
  - reasoning
  - entropy
  - mutual-information
  - LLM训练
---

## 🧠 核心摘要 (TL;DR)

> RAGEN-2 发现了 Agentic RL 训练中的一种隐蔽失效模式——**模板坍塌（Template Collapse）**：模型输出保持高熵（看似多样），但实际上对不同输入的响应趋于一致，推理失去了输入依赖性。论文通过信息论分解（$H(Z) = I(X;Z) + H(Z|X)$）证明互信息（MI）才是衡量推理质量的真正指标，并提出 **SNR-Aware Filtering** 机制——过滤低奖励方差样本——从根源上抑制模板坍塌，在多任务、多算法、多模型规模下平均提升 **+6.9 分**（PPO/3B 基准）。

## 🏷️ 元数据
- **论文标题**: RAGEN-2: Reasoning Collapse in Agentic RL
- **发表机构/会议**: arXiv 2026（Stanford / UCLA / Microsoft Research 等合作）
- **年份**: 2026
- **链接**: [[RAGEN2_paper|📄论文]] | 💻 Code: 暂未开源（项目页：https://ragen-ai.github.io/v2/）

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #reinforcement-learning #agentic-rl #reasoning #entropy #mutual-information #template-collapse #GRPO #PPO
- **前置知识 (Prerequisites)**: [[RL中的KV稀疏讨论]]
- **相关节点 (Related Notes)**: [[KATCoder]]

---

## 🎯 动机与痛点

### 背景：Agentic RL 的崛起与困境

近年来，以 DeepSeek-R1 为代表的强化学习（RL）训练范式显著提升了大语言模型（LLM）在数学推理、代码生成等封闭域任务中的能力。然而，当 RL 被应用到**多轮交互的智能体场景**（Agentic RL）——如网页导航、数学规划、代码执行——时，研究者发现训练常常不稳定，甚至出现模型"学坏"的现象：

- 训练初期指标好转，但随后性能停滞甚至下滑
- 奖励曲线出现剧烈波动，难以收敛
- 模型输出看起来"多样"但实际毫无意义

RAGEN-2 的出发点正是深入诊断这一现象，找到其信息论根源，并给出可操作的工程解决方案。

### 现有方案的核心盲区：熵≠推理质量

在 RL 训练中，**策略熵**（Policy Entropy）$H(Z)$ 被广泛用作衡量探索多样性的指标。熵越高，意味着模型输出越多样，被认为是训练仍在进行有效探索的信号。

然而，RAGEN-2 通过大量实验揭示了一个反直觉的事实：**熵高并不等于推理有效**。模型完全可以在保持高熵的同时，对所有不同的输入都生成相似的"模板化"响应——这种现象被称为**模板坍塌（Template Collapse）**。

> "熵只衡量同一输入下输出的多样性，但无法判断推理是否真正在响应不同的输入。"

这是现有 Agentic RL 诊断体系中的根本性盲区：所有依赖熵的监控工具都无法检测模板坍塌。

### 论文的切入点：信息论分解

RAGEN-2 的核心洞见是将输出熵做如下**信息论分解**：

$$H(Z) = I(X; Z) + H(Z \mid X)$$

其中：
- $X$：输入提示（Prompt）
- $Z$：模型输出（Response）
- $I(X; Z)$：**互信息（Mutual Information）**，衡量输出对输入的依赖程度，即"推理有多敏感地响应不同输入"
- $H(Z \mid X)$：**条件熵（Conditional Entropy）**，给定输入后输出仍然保留的随机性，即"输入无关的噪声"

**健康的推理**：$I(X; Z)$ 高，模型对不同输入给出有针对性的不同输出。
**模板坍塌**：$I(X; Z)$ 低，$H(Z \mid X)$ 高——模型对所有输入都给出看似随机但实质上与输入无关的模板化响应。

这一分解的精妙之处在于：它用一个统一的理论框架解释了为什么"熵高"的模型可以"推理差"，并给出了正确的诊断指标。

---

### 相关工作

#### 先驱工作（奠基）

**1. PPO (Schulman et al., 2017)**
近端策略优化算法，是 LLM 强化学习训练的基础算法之一。通过裁剪策略比率限制单步更新幅度，保证训练稳定性。RAGEN-2 在 PPO 框架下验证了 SNR-Aware Filtering 的有效性。
- 论文：*Proximal Policy Optimization Algorithms*（arXiv:1707.06347）

**2. InstructGPT / RLHF (Ouyang et al., 2022)**
首次将 RL 与人类反馈相结合训练 LLM，奠定了现代 LLM 对齐范式。RAGEN-2 的工作在此基础上进一步深入到多轮 Agentic RL 场景。
- 论文：*Training language models to follow instructions with human feedback*（arXiv:2203.02155）

**3. DeepSeekMath / GRPO (Shao et al., 2024)**
提出 Group Relative Policy Optimization（GRPO），通过组内相对优势替代价值网络，大幅降低 PPO 的计算开销，成为后续大量工作的基础算法。RAGEN-2 在 GRPO 框架下同样验证了其方法。
- 论文：*DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（arXiv:2402.03300）

**4. DeepSeek-R1 (DeepSeek-AI, 2025)**
展示了纯 RL 训练可以涌现出复杂推理链（CoT），在数学和代码等封闭域任务上超越监督学习，引发了 Agentic RL 研究热潮。RAGEN-2 正是针对 DeepSeek-R1 范式推广到开放 Agentic 任务时出现的新问题进行研究。
- 论文：*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*（arXiv:2501.12948）

**5. RAGEN v1 (Wang et al., 2025)**
RAGEN-2 的前身，提出了 StarPO 多轮轨迹级 RL 框架，构建了 RAGEN 系统用于多轮 Agent 训练与评估。识别了"Echo Trap"（奖励方差不稳定）等现象，为 RAGEN-2 的深入研究奠定基础。
- 论文：*RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning*（arXiv:2504.20073）

#### 对比/竞争工作

**6. DAPO (Yu et al., 2025)**
提出解耦裁剪与动态采样策略优化，开源了大规模 LLM RL 训练系统，在 AIME 2024 上取得 50 分（Qwen2.5-32B）。RAGEN-2 在 DAPO 的算法框架下测试了 SNR-Aware Filtering 的适配性。
- 论文：*DAPO: An Open-Source LLM Reinforcement Learning System at Scale*（arXiv:2503.14476）

**7. REINFORCE++ (Hu et al., 2025)**
提出全局优势归一化替代局部归一化，在无评论家网络的策略优化中取得更稳定的效果。与 RAGEN-2 关注的角度不同（后者关注数据过滤，前者关注梯度归一化），但都针对 RL 训练稳定性。
- 论文：*REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization*（arXiv:2501.03262）

**8. Kimi k1.5 (Kimi Team, 2025)**
通过长上下文扩展和长链式思维实现 RL 推理扩展，展示了 RL 在推理任务上的潜力。RAGEN-2 所研究的 Agentic RL 场景与 k1.5 的封闭域推理场景形成对比，揭示了开放环境中 RL 的额外挑战。
- 论文：*Kimi k1.5: Scaling Reinforcement Learning with LLMs*（arXiv:2501.12599）

#### 环境/评测工作

**9. WebShop (Yao et al., 2022)**
构建了包含 118 万真实产品的电商模拟环境，成为 LLM Agent 网页交互能力的重要基准。RAGEN-2 将 WebShop 作为网页导航任务的评测环境之一。
- 论文：*WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents*（arXiv:2207.01206）

**10. ALFWorld (Shridhar et al., 2021)**
将文本游戏与具身环境对齐，提供了家庭任务规划的测试平台（ICLR 2021）。RAGEN-2 使用 ALFWorld 类型的规划任务作为评估域之一。
- 论文：*ALFWorld: Aligning Text and Embodied Environments for Interactive Learning*（arXiv:2010.03768）

**11. ReAct (Yao et al., 2022)**
提出将推理轨迹与行动生成交错进行的框架，在 HotpotQA、WebShop 等任务上大幅超越基线，成为 Agentic LLM 的经典范式。RAGEN-2 的 Agent 输出风格继承了 ReAct 框架。
- 论文：*ReAct: Synergizing Reasoning and Acting in Language Models*（arXiv:2210.03629）

**12. MINT (Wang et al., 2023)**
系统评估了 LLM 在多轮工具使用和语言反馈下的交互能力，发现单轮性能优秀不保证多轮能力强——这与 RAGEN-2 发现的 Agentic RL 特有问题一脉相承。
- 论文：*MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback*（arXiv:2309.10691）

**13. MetaMathQA (Yu et al., 2023)**
通过多角度问题重写自举生成数学数据集，MetaMath-7B 在 GSM8K 上超越同规模 SOTA 11.5%。作为 RAGEN-2 数学推理评测的数据源之一。
- 论文：*MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models*（arXiv:2309.12284）

**14. Qwen2.5-Math (Alibaba, 2024)**
面向数学推理的专用大模型系列，覆盖 GSM8K、MATH、高考、AIME 等多层次基准。RAGEN-2 使用 Qwen2.5 系列模型（0.5B-7B）作为 backbone 验证方法的扩展性。
- 论文：*Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement*（arXiv:2409.12122）


## ⚙️ 核心设计与机制

### 2.1 模板坍塌（Template Collapse）的精确定义

RAGEN-2 将**模板坍塌**形式化定义为：当模型的条件响应分布 $P(Z \mid X)$ 对不同输入 $X$ 趋于相同时，即

$$\forall x_1, x_2 \in \mathcal{X}: P(Z \mid X = x_1) \approx P(Z \mid X = x_2)$$

此时尽管边缘分布 $P(Z) = \mathbb{E}_X[P(Z \mid X)]$ 的熵 $H(Z)$ 可能仍然很高（不同采样之间有差异），但**每个具体输入对应的推理实际上已经退化为输入无关的随机模板**。

从可观测现象来看，模板坍塌具有以下特征：
1. 输出文本依然多样且流畅，看似"仍在思考"
2. 对完全不同的问题（如数学推导 vs. 网页点击）给出结构几乎相同的响应
3. 奖励均值可能略有波动，但任务成功率持续低迷
4. 标准熵监控无法检测（熵值正常甚至偏高）

### 2.2 信号噪声比（SNR）假说：为什么会发生模板坍塌？

RAGEN-2 提出了一个清晰的机制性解释：**梯度信噪比（Signal-to-Noise Ratio, SNR）失衡**。

在 RL 训练中，参数更新梯度来自两类来源：

| 梯度来源 | 作用 | 特征 |
|---------|------|------|
| **任务梯度**（Task Gradient） | 将策略推向高奖励行为 | 强度∝奖励方差；不同输入对应不同方向 |
| **正则化梯度**（Regularization Gradient） | 熵惩罚（保持探索）+ KL 散度（防止偏离基础模型） | 强度固定；对所有输入相同方向 |

当某个提示的奖励方差极低时（例如所有采样的 rollout 都得到相同奖励），任务梯度趋近于零，而正则化梯度（常数量级）相对主导。此时：

$$\text{SNR} = \frac{\|\nabla_\theta \mathcal{L}_{\text{task}}\|}{\|\nabla_\theta \mathcal{L}_{\text{reg}}\|} \approx 0$$

正则化梯度将策略推向一个"熵最大但输入无关"的解——这正是模板坍塌的形成过程。**RAGEN-2 通过实验验证了任务梯度范数与奖励方差的正相关关系，且这一规律在 PPO 和 GRPO 算法下均成立。**

### 2.3 互信息代理指标（MI Proxy）：在线诊断

精确计算 $I(X; Z)$ 需要对输入-输出联合分布积分，实际不可行。RAGEN-2 提出了一种**在线 MI 代理估计方法**：

**核心思路**：利用一个评分函数衡量"响应 $z_i$ 对输入 $x_j$ 的匹配程度"，构造一个响应-输入匹配矩阵，通过对角占优程度来估计 MI。

具体步骤：
1. 从不同输入 $\{x_1, ..., x_n\}$ 各采样若干响应 $\{z_1, ..., z_n\}$
2. 对每对 $(x_i, z_j)$ 计算检索分数 $s(x_i, z_j)$（如基于奖励模型或语义相似度）
3. 构造 $n \times n$ 分数矩阵，若对角线元素（正确配对）显著高于非对角线元素，说明 MI 高（输出对输入敏感）
4. 对角占优度（**Diagonal Dominance**）作为 MI 的代理指标

RAGEN-2 实验验证：**MI 代理与任务最终性能的相关系数为 +0.39**，而熵与最终性能的相关系数仅为 **−0.14**（甚至负相关！）。这证明 MI 代理是远比熵更可靠的训练健康度指标。

### 2.4 SNR-Aware Filtering：核心工程方案

基于上述分析，RAGEN-2 提出了直接、有效的干预手段：

**算法描述**：

在每次 RL 训练迭代中：
1. 对每个提示 $x_i$ 采样 $k$ 条 rollout，计算奖励 $\{r_{i,1}, ..., r_{i,k}\}$
2. 计算每个提示的**奖励方差** $\text{Var}_i = \text{Var}(\{r_{i,j}\})$
3. 按奖励方差从高到低排序，选取前 $\rho$ 比例的提示组成训练批次
4. 在过滤后的子集上执行标准 PPO/GRPO 更新

**核心公式**（过滤条件）：

$$\mathcal{B}_{\text{filtered}} = \{x_i : \text{Var}(\{r_{i,j}\}_{j=1}^k) \geq \tau_\rho\}$$

其中 $\tau_\rho$ 是使保留比例恰好为 $\rho$ 的方差阈值。

**为什么有效？**
- 低方差提示（所有 rollout 奖励相同）→ 任务梯度为零 → 被过滤掉
- 高方差提示（rollout 奖励有差异）→ 有效任务梯度 → 保留用于训练
- 从根本上提升了 SNR，使正则化梯度不再主导更新方向

**超参数选择**：
- 最优过滤比例 $\rho$ 在 0.80–0.98 范围内表现最好
- 过于激进（$\rho \leq 0.6$）会损失样本覆盖度，导致过拟合
- 不过滤（$\rho = 1.0$）则模板坍塌重新出现
- 在随机性 0–50% 的环境中效果最显著，随机性接近 100% 时效果下降（因为高随机性本身会产生奖励方差，信号被环境噪声淹没）

### 2.5 信息论框架的统一性

RAGEN-2 的贡献之一是将上述各概念统一在同一信息论框架下：

```
H(Z) = I(X; Z) + H(Z|X)
 ↑          ↑           ↑
总熵      有效推理    输入无关噪声
（可观测） （希望最大化） （模板坍塌的根源）

SNR 低 → 正则化梯度主导 → H(Z|X) ↑，I(X;Z) ↓ → 模板坍塌
SNR 高 → 任务梯度主导 → I(X;Z) ↑ → 有效推理
SNR-Aware Filtering → 提升 SNR → 防止模板坍塌
```

这一框架不仅解释了已有现象，还为未来 Agentic RL 的诊断和优化提供了清晰的理论指引。

---

## 📊 实验与性能评价

### 3.1 实验设置

**模型**：Qwen2.5 系列，规模覆盖 0.5B、1.5B、3B、7B
**算法**：PPO、GRPO、DAPO、REINFORCE++（共 4 种主流 RL 算法）
**任务环境**（8 个）：
- 数学推理（GSM8K 类）
- 规划（ALFWorld 类）
- 网页导航（WebShop 类）
- 代码执行

### 3.2 主要结果

#### 诊断能力对比

| 指标 | 与最终任务成功率的相关系数 |
|------|--------------------------|
| 策略熵 $H(Z)$ | −0.14（**负相关**） |
| MI 代理（对角占优度） | **+0.39**（强正相关） |

**结论**：传统熵指标不仅无法预测性能，甚至可能误导训练判断；MI 代理是更可靠的在线诊断工具。

#### SNR-Aware Filtering 效果

- **PPO / 3B 基准**：应用 SNR-Aware Filtering 后平均成绩提升 **+6.9 分**
- 在全部 4 种 RL 算法上均有一致提升
- 在 0.5B 到 7B 的所有模型规模上均有效果
- 在 2 个模型家族（Qwen2.5）、8 个环境中保持一致

#### 梯度分解验证

实验直接测量了任务梯度范数和正则化梯度范数随奖励方差的变化：
- 任务梯度范数与奖励方差**强正相关**
- 正则化梯度范数与奖励方差**近似无关**（常数量级）
- 在 PPO 和 GRPO 下规律相同

这从梯度角度直接验证了 SNR 假说的正确性。

#### 推理长度的普遍收缩

论文还记录了一个独立于模板坍塌的普遍现象：**所有训练配置下，推理输出长度持续下降 5–66%**，无论是否应用 SNR-Aware Filtering，无论模型规模或环境类型。作者认为这是 Agentic RL 内在的压缩动态（模型倾向于找到"最短有效路径"），而非退化现象。

#### 最优过滤率的鲁棒性

| 过滤比例 $\rho$ | 效果 |
|----------------|------|
| $\rho \leq 0.6$ | 样本覆盖不足，过拟合，性能下降 |
| $\rho = 0.8$ | 接近最优 |
| $\rho = 0.9$–$0.98$ | **最优区间** |
| $\rho = 1.0$ | 无过滤，模板坍塌重现 |

### 3.3 局限性

1. **环境随机性敏感**：当环境本身随机性极高（>50% 状态转移噪声）时，SNR-Aware Filtering 的效果减弱，因为奖励方差中包含大量环境噪声而非策略信号。
2. **MI 代理的近似性**：对角占优度是 MI 的代理而非精确估计，在某些分布下可能失效。
3. **样本效率下降**：过滤掉低方差提示意味着部分采集的 rollout 被丢弃，总体样本利用率降低。
4. **对高度随机任务的适应**：论文未充分探讨如何在强随机性环境中区分"信号"和"环境噪声"。

---

## 💡 启发与我的想法

### 可借鉴之处

1. **信息论工具的诊断价值**：将 $H(Z) = I(X;Z) + H(Z|X)$ 分解引入 RL 训练诊断，是一个非常优雅的理论工具。相比直接看奖励曲线，MI 代理能提前发现模板坍塌的早期信号，有潜力作为通用的 Agentic RL 训练健康度监控指标。

2. **数据过滤的正交性**：SNR-Aware Filtering 与具体 RL 算法无关（PPO/GRPO/DAPO 均有效），可以作为"插件式"改进叠加到任何现有的 RL 训练框架上，工程实现简单，效果显著。

3. **对 KV Cache 研究的启示**：模板坍塌的本质是"注意力失去输入依赖性"，这与 KV Cache 稀疏化研究中"哪些 token 携带真正有价值的信息"问题具有深刻的概念相似性——MI 分解框架或许也可用于分析 KV 重要性评估。

4. **奖励方差作为课程信号**：论文用奖励方差作为"难度代理"，自适应地过滤掉对当前策略来说"太简单"或"太随机"的样本。这与课程学习（Curriculum Learning）的思想高度吻合，但实现更为简洁——无需显式的难度估计模型。

### 改进空间

1. **自适应过滤率**：当前 $\rho$ 是固定超参数，可以探索根据 MI 代理动态调整 $\rho$ 的方案（MI 低时过滤更激进，MI 高时放宽过滤）。

2. **环境随机性感知**：在高随机性环境中，可以尝试用"提示级奖励方差的时间移动平均"替代单次方差估计，减少环境噪声的影响。

3. **MI 代理的计算优化**：当前代理方法需要构造 $n \times n$ 匹配矩阵，在大批次时计算开销较大。可以研究更高效的近似方法（如对比学习式的 MI 估计）。

4. **与长度惩罚结合**：推理长度的普遍收缩现象值得进一步研究——是否应该引入长度奖励来平衡"效率"与"推理深度"？这在需要复杂多步推理的任务中可能特别重要。

### 下一步 Action
- [ ] 关注 RAGEN 项目页（https://ragen-ai.github.io/v2/）的代码开源情况
- [ ] 思考 SNR-Aware Filtering 能否应用于 RL 辅助的 KV Cache 稀疏化训练（低奖励方差 = 当前稀疏策略对该 token 无判别力？）
- [ ] 研究 MI 代理监控方法在实际工程中的开销评估

