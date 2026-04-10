---
title: "Sparse but Critical"
tags:
  - RLVR
  - reinforcement-learning
  - token-distribution
  - LLM-reasoning
  - post-training
---

## 🧠 核心摘要 (TL;DR)
> 本文系统分析了 RLVR（Reinforcement Learning with Verifiable Rewards）微调后 LLM 内部 token 级别的分布变化，发现仅有极少数位置（1~17%）的 token 分布发生显著偏移，但正是这些"稀疏但关键"的 token 驱动了全部的推理性能提升；通过交叉采样实验证明替换约 1~10% 的 token 即可复现或抹除 RL 增益，并在此基础上提出了基于散度加权的改进训练算法，在数学推理基准上超越 DAPO 基线约 2.5 分。

## 🏷️ 元数据
- **论文标题**: Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs
- **发表机构/会议**: ICLR 2026 / Alibaba Group (Qwen Pilot Team)
- **年份**: 2026
- **链接**: [[SparseButCritical_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #rlvr #reinforcement-learning #token-distribution #llm-reasoning #post-training #grpo
- **前置知识 (Prerequisites)**: [[RL中的KV稀疏讨论]]
- **相关节点 (Related Notes)**:
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

RLVR 是当前提升 LLM 推理能力最有效的 post-training 手段之一。DeepSeek-R1、DAPO、SimpleRL 等一系列工作展示了 RLVR 在数学、代码、逻辑推理等任务上的显著效果。然而，**RL 微调究竟在"哪里"改变了模型**，始终是一个未被完整回答的问题：

- 以往研究（如 Wang et al. 2025 "Beyond the 80/20 rule"、Cui et al. 2025 "The entropy mechanism of RL"）从 **序列/聚合层面** 分析了 RL 后的熵变化，但缺少 token 级别的精细刻画
- 不清楚分布偏移是广泛分散在所有位置，还是集中在少数关键 token 上
- 不清楚这些偏移是否与推理性能的提升存在因果关系
- 缺乏 RLVR 与 SFT 在分布改变模式上的系统对比

### 论文的切入点

本文提出三个核心研究问题：
1. **哪些 token 的分布发生了变化**（JS 散度分析）？
2. **这些变化是否功能性地驱动了性能提升**（交叉采样实验）？
3. **变化的微观机制是什么**（top-k 重叠、概率质量重分配分析）？

在此基础上，提出了**散度加权优势（Divergence-Weighted Advantage）**改进算法，对高/低 KL 散度位置施加不同的训练权重。

---

### 🔗 相关工作综述

#### 先驱/奠基工作（RLVR 方法）

**1. PPO（Schulman et al., 2017）**
近端策略优化算法，是 RLHF 和 RLVR 的经典基础，通过 clip ratio 约束策略更新幅度，避免策略崩溃。RLVR 中的所有主要算法均是对 PPO 的变体或改进。

**2. InstructGPT / RLHF（Ouyang et al., 2022）**
首次在大规模 LLM 中应用 RL from human feedback，奠定了 LLM post-training 用 RL 的实践范式，证明了 RL 反馈可以有效改变模型行为与偏好。

**3. DeepSeekMath / GRPO（Shao et al., 2024）**
提出 Group Relative Policy Optimization（GRPO），是 RLVR 领域最重要的算法奠基工作之一。GRPO 去除了 value function critic，改用同一问题下多条采样响应的相对奖励作为基线，大幅降低了训练内存开销。DeepSeekMath 7B 在 MATH 上达到 51.7%。

**4. DeepSeek-R1（DeepSeek-AI, 2025）**
展示了纯 RLVR 训练（不依赖人类标注推理轨迹）可以激励 LLM 自发产生推理能力。模型自然涌现出自我反思、回溯、动态策略调整等高级推理行为，是 RLVR 的代表性成果。

**5. Kimi k1.5（Kimi Team, 2025）**
通过长上下文扩展和改进的策略优化，在 AIME 上达到 77.5、MATH 500 达到 96.2，与 OpenAI o1 性能持平。展示了无需 MCTS、value function 的简化 RL 框架的可行性。

#### 直接对比方法

**6. DAPO（Yu et al., 2025）**
Decoupled Clip and Dynamic Sampling Policy Optimization，本文的主要实验对比基线之一。使用 Qwen2.5-32B 在 AIME 2024 上达到 50 分。本文的交叉采样实验和改进算法均在 DAPO 框架下进行。

**7. SimpleRL-Zoo（Zeng et al., 2025）**
在 10 个不同基础模型上系统研究 zero RL 训练，是本文另一个主要实验对比基线。发现调整格式奖励和控制查询难度可有效改善推理精度和响应长度。首次在非 Qwen 系列小模型上观察到"顿悟时刻"。

**8. Dr.GRPO（Liu et al., 2025）**
识别并修正了 GRPO 中的优化偏差（导致 response 长度不合理增长），提出无偏的 Dr.GRPO，在保持推理性能的同时提升 token 效率。在 7B 基础模型上 AIME 2024 达到 43.3%。

**9. GSPO（Zheng et al., 2025）**
Group Sequence Policy Optimization，将 token 级别的 importance ratio 改为序列级别，提出序列级裁剪和优化。相比 GRPO 训练更稳定高效，已应用于 Qwen3 模型。

**10. Tulu 3（Lambert et al., 2024）**
提出完全开放的 post-training 流程，结合 SFT、DPO 和 RLVR（Reinforcement Learning with Verifiable Rewards），是第一批将 RLVR 命名为 RLVR 并系统研究的工作之一。

#### 相关分析工作（Token 级别/分布分析）

**11. Beyond the 80/20 Rule（Wang et al., 2025）**
研究高熵少数 token 在 RL 推理中的作用，发现高熵 token 对 RL 效果有不成比例的重要贡献。与本文的稀疏性发现高度相关，但本文进一步通过交叉采样给出了因果证据。

**12. On the Direction of RLVR Updates（Huang et al., 2026）**
与本文同期工作，研究 RLVR 更新方向（$\Delta \log p$ 符号），证明方向比幅度更关键。提出测试时外推和训练时重权重两种应用方法，与本文的散度加权思路互补。

**13. New Skills or Sharper Primitives（Wang et al., 2026）**
与本文同期工作，从概率视角研究 RLVR 是否为模型带来"新技能"，发现 RLVR 主要通过锐化原子步骤概率来提升复合性能，与本文"概率质量重分配"的发现一致。

**14. SFT Memorizes, RL Generalizes（Chu et al., 2025）**
系统对比 SFT 和 RL 微调在泛化性上的差异，发现 SFT 倾向于记忆训练数据而难以泛化 OOD 场景，RL 则展现更好的泛化能力。与本文发现的"SFT 产生更密集分布偏移"形成呼应。

**15. Cognitive Behaviors for Self-Improving Reasoners（Gandhi et al., 2025）**
识别了 RL 训练中自我改进的四种关键认知行为（验证、回溯、子目标设定、反向链式推理），从行为层面解释了 RLVR 的推理增益机制。

#### Follow-up / 系统应用工作

**16. Speculative Decoding（Leviathan et al., 2023）**
论文中 cross-sampling 框架与 speculative decoding 在"用草稿模型采样、目标模型验证"的结构上有相似之处，但目标和机制完全不同。本文的 cross-sampling 是分析工具而非部署优化。

**17. DeepSeek-V3（DeepSeek-AI, 2024）**
作为强基础模型，DeepSeek-V3-Base 在未经 RL 训练时已展现一定推理能力（"顿悟时刻"），为分析 RL 微调的增量效果提供了基准。


## ⚙️ 核心设计与机制

### 1. 分布偏移度量：Jensen-Shannon 散度

本文选择 **JS 散度（Jensen-Shannon Divergence）** 度量基础模型 $\pi_{\text{base}}$ 与 RL 微调后模型 $\pi_{\text{RL}}$ 在每个位置 $t$ 上的 token 分布差异：

$$\text{JS}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M), \quad M = \frac{P + Q}{2}$$

选择 JS 散度的原因：
- **对称性**：$\text{JS}(P \| Q) = \text{JS}(Q \| P)$，无需考虑方向问题
- **有界性**：值域 $[0, \log 2]$，便于比较
- **良定义性**：即使两个分布支撑集不完全重合也不会发散（相比 KL 散度）

对每个位置的 JS 散度，论文在 **top-$p$ 截断**后的分布上计算（与实际生成采样一致）。

### 2. 稀疏性发现：绝大多数位置几乎不变

**实验设置**：在 AIME 2024/2025、GPQA-Diamond 等数学推理基准上，对 SimpleRL 和 DAPO 两种 RLVR 方法的微调前后模型进行对比分析。

**核心发现**：

| 方法 | 近零散度位置占比 |
|------|---------------|
| SimpleRL | >98% 位置的 JS 散度 ≈ 0 |
| DAPO | >83% 位置的 JS 散度 ≈ 0 |
| SFT（对比） | 显著更密集，全局分散 |

这一发现揭示了 **RLVR 微调的高度稀疏性**：即使是效果显著的 RLVR 方法（如 DAPO），也只改变了模型中极少部分位置的 token 分布。相比之下，SFT 产生的是广泛分散的全局分布变化。

### 3. 交叉采样框架：因果验证

**核心问题**：稀疏偏移位置是否真的"驱动"了性能提升？还是只是统计相关？

论文提出了 **Cross-Sampling（交叉采样）框架**，通过"手术式"地替换特定位置的 token 来验证因果关系：

定义混合策略：
$$\pi_{\text{mix}}^{(\pi_{\text{prim}}, \pi_{\text{int}})} = (1 - \mathcal{S}_t) \pi_{\text{prim}} + \mathcal{S}_t \pi_{\text{int}}$$

其中 $\mathcal{S}_t$ 是切换规则（基于散度阈值 $\epsilon$）：
$$\mathcal{S}_t = \mathbb{1}[\text{JS}(\pi_{\text{RL}}(\cdot | x_{<t}), \pi_{\text{base}}(\cdot | x_{<t})) > \epsilon]$$

**两种干预方向**：

| 干预方向 | Primary 策略 | Intervention 策略 | 目标 |
|---------|------------|-----------------|------|
| 正向（Forward）| $\pi_{\text{base}}$ | $\pi_{\text{RL}}$ | 在基础模型生成中注入 RL token |
| 反向（Reverse）| $\pi_{\text{RL}}$ | $\pi_{\text{base}}$ | 在 RL 生成中替换回基础 token |

**关键实验结果**：

*正向干预*（基础模型生成 + 注入少量 RL token）：

| 数据集 | 方法 | 干预 token 占比 | 性能提升 |
|-------|------|--------------|--------|
| AIME 2024 | SimpleRL | ~3.86% | 8.23% → >25% |
| AIME 2024 | DAPO | ~7.8% | 8.23% → >44% |
| AIME 2025 | SimpleRL | ~1.53% | 5.3% → >14% |
| AIME 2025 | DAPO | ~6.47% | 5% → >33% |

*反向干预*（RL 模型生成 + 替换回少量基础 token）：仅替换相同比例的高散度 token 就能使 RL 性能**全面崩溃**至基础模型水平。

**结论**：约 1~10% 的高散度 token 替换即可**复现**或**抹除**全部 RL 性能增益，证明这些稀疏位置对 RLVR 效果具有**直接因果作用**。

### 4. 微观机制分析

**4.1 Top-k 重叠分析**

在高散度位置，基础模型和 RL 模型的 top-k 候选集高度重叠：
- $k \geq 2$ 时，SimpleRL 的平均重叠率 >80~85%
- 说明 RL 主要进行的是**候选集内的概率重新排名**，而不是引入基础模型从未考虑过的全新 token

**4.2 概率质量重分配机制**

进一步分析发现，RL 微调的核心机制是**概率质量在已有候选集中的重新分配**（Rank Reordering）：
- RL 很少"发明"基础模型认为极不可能的 token（即 RL 不会让之前概率接近 0 的 token 突然变成主要候选）
- 而是在已有高概率候选 token 之间进行概率质量的转移（如把第 2 名候选提升为第 1 名）

**4.3 训练演化轨迹**

通过分析中间检查点，发现性能的提升是随着干预 token 数量的增加**平滑累积**的，而不是突变式的——说明是多个决策点处的累积效应，而非少数"顿悟时刻"。

### 5. 散度加权优势（Divergence-Weighted Advantage）

基于上述分析，论文提出了改进的训练算法，在优势估计阶段对不同散度位置施加不同权重：

$$w_t = 1 + s \cdot \left(\sigma(\alpha \cdot \text{KL}_t) - \frac{1}{2}\right)$$

其中：
- $\text{KL}_t$：当前训练迭代中该位置的 KL 散度（相对于参考模型）
- $\alpha$：控制权重函数的曲率/敏感度
- $s$：控制方向（$s > 0$ 时提升高 KL 位置权重；$s < 0$ 时提升低 KL 位置权重）
- $\sigma(\cdot)$：sigmoid 函数（使权重平滑有界）

**两种配置**：
- **High-KL Boost**（$s > 0$）：放大已显著偏移位置的学习信号，强化已发生变化的方向
- **Low-KL Boost**（$s < 0$）：提升"尚未偏移但可能重要"的低 KL 位置的学习权重，探索更多潜在关键 token

实际集成方式：将 $w_t$ 融入 DAPO 的 token 级 loss 加权：
$$\mathcal{L}_{\text{DWA}} = -\sum_t w_t \cdot \text{PPO-loss}_t$$


## 📊 实验与性能评价

### 对比基线（Baseline）

| 基线方法 | 模型 | 基础分数（AIME 2024） |
|---------|------|---------------------|
| DAPO | Qwen2.5-Math-7B | ~33.61 |
| SimpleRL | Qwen2.5-Math-7B | 较低 |

### 主要实验结果

**散度加权优势（DWA）在 Qwen2.5-Math-7B 上的效果**（3 次运行均值）：

| 配置 | AIME 2024 | AIME 2025 | AMC | 总体 |
|------|-----------|-----------|-----|------|
| DAPO 基线 | 33.61 | 18.75 | 75.08 | 42.48 ± 1.35 |
| Low-KL Boost | **35.90** | **19.90** | **78.97** | **44.92 ± 0.05** |
| High-KL Boost | **36.74** | **20.00** | **78.40** | **45.05 ± 0.79** |

- 两种 DWA 配置相比 DAPO 基线均提升约 **+2.5 分**（总体）
- High-KL Boost 在 AIME 2024 上提升最显著（+3.13 分）
- Low-KL Boost 方差极低（±0.05），训练更稳定

**交叉采样详细结果**（AIME 2024，SimpleRL）：

| 干预 token 比例 | Forward（注入 RL token）| Reverse（替换为 Base token）|
|---------------|----------------------|--------------------------|
| 0% | 基础模型水平 (8.23%) | RL 模型水平 (>25%) |
| ~1.53-3.86% | **恢复大部分 RL 性能** | **RL 性能崩溃至基础水平** |
| 100% | 完整 RL 模型水平 | 基础模型水平 |

### 局限性（Limitations）

1. **实验规模有限**：主要在 Qwen2.5-Math-7B 上验证，更大规模（70B+）模型的稀疏性模式尚未验证
2. **任务单一性**：实验聚焦数学推理（AIME、GPQA），对代码、自然语言推理等其他任务的泛化性未知
3. **JS 散度的局限**：JS 散度衡量的是整体分布距离，无法捕捉"方向性"信息（同期工作 Huang et al. 2026 从符号方向 $\Delta \log p$ 角度补充了这一分析）
4. **机制不完整**：仅分析了 top-k 重叠和概率质量重分配，"哪些语义/句法特征的 token 成为关键位置"尚未深入探讨
5. **DWA 超参数敏感性**：$\alpha$、$s$ 的选取对结果有一定影响，最优超参数的泛化性需进一步验证

---

## 💡 启发与我的想法

### 可借鉴之处

1. **分布稀疏性是普遍规律还是任务特定的？**  
   本文发现在数学推理中 RL 的影响高度稀疏（SimpleRL >98%、DAPO >83% 位置近零偏移）。这一规律在 RL 微调的早期阶段（大部分参数更新未落在"关键"位置上）可能意味着传统的 token 均匀加权是低效的。**设计自适应加权方案**（如 DWA）是一个高价值的工程方向。

2. **Cross-Sampling 框架的可扩展性**  
   论文的交叉采样框架本质上是一种"模型内部手术"工具，不仅适用于分析，还可以用于：
   - **模型融合**：在两个不同策略的生成序列间进行选择性 token 替换
   - **推理加速**：识别不需要"RL 版本"策略介入的位置，用更轻量的 base 模型处理大部分 token（类似 Speculative Decoding 的思路，但基于语义重要性而非速度）

3. **稀疏性 vs SFT 的区别**  
   SFT 产生密集分布偏移，RL 产生稀疏偏移——这为理解"为什么 RL 比 SFT 泛化能力更强"提供了机制层面的解释（稀疏改变 = 只调整了真正需要改变的决策点，未引入不必要的分布扭曲）。

4. **KL 散度的双向价值**  
   Low-KL 位置不一定不重要（可能是"沉默的关键路标"），High-KL 位置也不一定全都有用（可能有噪声）。DWA 的思路启示：**训练信号的空间分布与任务性能的关系是非线性的**，值得更系统地研究。

### 改进空间

1. **动态阈值**：本文的交叉采样使用固定散度阈值 $\epsilon$，可以探索随训练迭代自适应调整阈值的方案
2. **多模型多任务验证**：将稀疏性分析扩展到 Llama、Mistral 等非 Qwen 系列模型，以及代码生成、多步骨物理推理等任务
3. **位置语义解释**：哪些位置的 token 更容易成为"关键偏移位置"？初步猜测是推理链中的"转折词"（"However", "Therefore", "Let me reconsider"）和数学表达式起始 token，值得进一步统计验证
4. **与 MoE 的结合**：在 MoE 模型中，不同 token 激活不同专家，稀疏偏移位置是否与特定专家路由高度相关？

### 下一步 Action
- [ ] 阅读同期工作 "On the Direction of RLVR Updates"（Huang et al., 2026），对比 $\Delta \log p$ 方向视角 vs JS 散度幅度视角的异同
- [ ] 关注 GSPO、Dr.GRPO 在 token-level 加权上的实践，看 DWA 思路是否已有工程落地
- [ ] 思考：稀疏偏移分析 + KV Cache 稀疏化是否有结合点？（RL 后高散度位置对应的 token 是否也是 KV Cache 中注意力集中的位置？）


