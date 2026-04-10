---
title: "ThinkTwice：推理与自我精炼"
tags:
  - LLM推理
  - 强化学习
  - 自我精炼
  - GRPO
  - 数学推理
---

## 🧠 核心摘要 (TL;DR)
> ThinkTwice 提出一个简洁的两阶段 RLVR 框架，仅用二元正确性奖励同时训练模型的初始推理能力和自我精炼能力，无需任何外部 critique 标注；在 Qwen3-4B 上 AIME pass@4 较 GRPO 基线提高 5 个百分点，经一次自我精炼后再额外提升 11.5 个百分点，同时训练效率提升（减少 16% 的 wall-clock 时间）。

## 🏷️ 元数据
- **论文标题**: ThinkTwice: Jointly Optimizing Large Language Models for Reasoning and Self-Refinement
- **发表机构/会议**: arXiv（University of Toronto / 独立研究）
- **年份**: 2026
- **链接**: [[ThinkTwice_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #llm-reasoning #self-refinement #rlvr #grpo #math-reasoning
- **前置知识 (Prerequisites)**: [[KV稀疏调研]]
- **相关节点 (Related Notes)**: [[AttentionRes]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈

LLM 在复杂推理任务（数学竞赛、代码生成）上的性能瓶颈长期存在两类研究路线：

1. **RLVR 路线**（Reinforcement Learning with Verifiable Rewards）：以 DeepSeek-R1、DeepSeekMath、DAPO 为代表，通过 GRPO 等算法利用结果奖励大规模强化训练，显著提升初始推理能力，但**训练完成后模型无法在推理时自主改进答案**；
2. **自我精炼路线**（Self-Refinement / Self-Correction）：以 Self-Refine、Reflexion、SCoRe 为代表，让模型对自己的输出进行反思和修改；但主流方法存在明显缺陷：
   - **训练无关方法**（Self-Refine、Reflexion）：依赖 GPT-4 级别的基础能力，普通模型无法产生有效 feedback，且 Huang et al. (2023) 等工作指出 LLM 无法仅靠自身 feedback 可靠地自我纠错；
   - **需要 critique 标注的训练方法**（Critique-GRPO、PAG）：需要额外的批评标注数据，数据获取成本高且难以扩展；
   - **分离训练**（SCoRe、S2R）：将推理训练和自我精炼训练解耦，先后独立优化，导致初始推理阶段没有为精炼做准备，可能陷入局部最优。

**核心矛盾**：现有方法要么只优化推理，要么只优化精炼，或者虽然联合但需要昂贵的标注；没有一种方法能以**最小化额外开销**同时优化两个能力。

### 论文的切入点

ThinkTwice 的核心洞察是：**推理和自我精炼可以在同一个 GRPO 训练循环中被联合优化，只需在每次训练迭代中随机混入两类样本——Phase 1（初始求解）和 Phase 2（对已有解进行精炼）**，且两个 phase 使用完全相同的二元正确性奖励，无需任何 critique 注释。

此外，论文发现了一个**隐式课程学习**（Implicit Curriculum Learning）现象：随着 Phase 1 训练的推进，模型的初始解质量逐步提升，使得 Phase 2 所"复习"的 base solution 难度也随之动态增加，形成自然的由易到难的训练课程，无需手动设计课程。

### 相关工作

#### 先驱 / 奠基工作

| 论文 | 贡献 | 关联 |
|------|------|------|
| Shao et al. (2024) — **DeepSeekMath** [arXiv:2402.03300] | 提出 GRPO 算法，将 Group Relative Policy Optimization 用于数学推理 RL 训练 | ThinkTwice 以 GRPO 为骨干 |
| Guo et al. (2025) — **DeepSeek-R1** [arXiv:2501.12948] | 证明纯 RL 可激发 LLM 推理能力，无需人工标注推理链 | 奠定 RLVR 范式 |
| Madaan et al. (2023) — **Self-Refine** [arXiv:2303.17651] | 首个系统性自迭代精炼框架：生成→feedback→refine 循环 | ThinkTwice 的直接对比对象 |
| Shinn et al. (2023) — **Reflexion** [arXiv:2303.11366] | 用语言 RL 让 Agent 从失败中反思，存储在 episodic memory 中 | 训练无关精炼代表方法 |

#### 直接对比方法（RL-Based Refinement）

| 论文 | 方法 | 与 ThinkTwice 的区别 |
|------|------|------|
| Kumar et al. (2024) — **SCoRe** [arXiv:2409.12917] | 两阶段 RL 训练：先训练初始解，再训练自我修正；防止 collapse 的正则约束 | 两阶段分离训练，Phase 2 依赖正则防止退化 |
| Ma et al. (2025) — **S2R** [arXiv:2502.12853] | 先训练 reasoning，再训练 verification，最后 self-correction；三步独立 | 三步解耦，数据效率低 |
| Zhang et al. (2025b) — **Critique-GRPO** | GRPO + 自然语言 critique + 数值 feedback | 需要 critique 标注数据 |
| Jiang et al. (2025) — **PAG** | Policy as Generative Verifier，多轮 RL 精炼 | 模型兼任 verifier 和 refiner，结构复杂 |

#### 近期 Follow-up / 相关系统

| 论文 | 贡献 |
|------|------|
| Yu et al. (2025) — **DAPO** [arXiv:2503.14476] | 大规模 LLM RL 系统，Decoupled Clip + Dynamic Sampling，AIME 2024 达 50 分 |
| He et al. (2025b) — **Skywork Open Reasoner 1** | 开源 reasoning 模型技术报告 |
| Hu et al. (2025) — **Open-Reasoner-Zero** | 开源 zero-RL 推理训练方法 |
| Xiong et al. (2025) — **Self-Rewarding Correction** | 数学推理中自我奖励的自我修正方法 |
| Zhao et al. (2025a) — **Spontaneous Self-Correction** | 大模型自发自我修正能力的提升 |
| Welleck et al. (2022) — **Self-Correct via Learning** | 学习自我纠错序列生成 |
| Wang et al. (2022) — **Self-Consistency** | CoT 多路径自洽解码，无需训练 |
| Huang et al. (2023) — **LLMs Cannot Self-Correct** | 实验证明无外部 oracle 时 LLM 不能可靠自我纠错 |


## ⚙️ 核心设计与机制

### 整体框架

ThinkTwice 的核心思想是：**在单一 GRPO 训练循环中，将 Phase 1（初始推理）和 Phase 2（自我精炼）样本随机混合，以相同的二元奖励函数联合优化**。

```
每次训练迭代
  ├── Phase 1 样本（概率 p）：(problem, base_prompt) → 直接生成答案
  └── Phase 2 样本（概率 1-p）：(problem, base_solution, refine_prompt) → 精炼先前答案
```

训练中，同一个模型既作为 Phase 1 的 reasoning model，也作为 Phase 2 的 refinement model。

---

### Phase 1：推理训练

Phase 1 与标准 GRPO 相同：给定问题 $x$，模型从策略 $\pi_\theta$ 采样 $G$ 个候选解 $\{y_i\}_{i=1}^G$，计算组内相对奖励：

$$
r_i = \mathbf{1}[\mathcal{E}(y_i) = a^*]
$$

其中 $\mathcal{E}(\cdot)$ 为答案提取函数，$a^*$ 为真实答案。GRPO 的策略梯度目标为：

$$
\mathcal{J}_\text{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min\!\left(\rho_i \hat{A}_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta \cdot \mathbb{D}_\text{KL}[\pi_\theta \| \pi_\text{ref}]\right]
$$

其中 $\rho_i = \pi_\theta(y_i|x) / \pi_{\text{old}}(y_i|x)$ 为重要性比，$\hat{A}_i$ 为组内标准化优势估计：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}
$$

---

### Phase 2：自我精炼训练

Phase 2 的输入是一个**多轮对话格式的 prompt**，包含：
1. 原始问题 $x$
2. 一个"base solution" $y_\text{base}$（从历史 Phase 1 输出中随机采样）
3. 一条**任务无关的精炼指令**（如 "Review your answer. Is it correct? Refine if necessary."）

关键设计：**精炼指令不包含 $y_\text{base}$ 是否正确的任何提示**，使模型在不知道先前答案对错的情况下完成精炼，从而训练出真正的"双向精炼"能力（对错误答案纠正、对正确答案维持）。

模型对此 multi-turn prompt 生成精炼解 $\{z_i\}_{i=1}^G$，同样用二元正确性奖励评估：

$$
r_i^{(2)} = \mathbf{1}[\mathcal{E}(z_i) = a^*]
$$

**GRPO 目标函数与 Phase 1 完全相同**，只是条件输入从 $(x)$ 变为 $(x, y_\text{base}, \text{refine\_prompt})$。

---

### 联合训练策略

**随机混合（Random Interleaving）**：在每个训练 step，以概率 $p$（论文取 $p=0.5$）决定使用 Phase 1 还是 Phase 2 的数据构建 batch。两个 phase 共享同一组模型参数，梯度在同一个 optimizer step 中累积更新。

**Base Solution 的采样来源**：Phase 2 的 $y_\text{base}$ 从**历史 Phase 1 输出**中随机选取（非当前 step 实时生成），包含正确解和错误解的混合。这样：
- 模型需要在**不知道 $y_\text{base}$ 对错**的情况下判断是否需要修改；
- 随着训练进行，历史 Phase 1 解的质量逐渐提升，形成**隐式课程学习**。

---

### 隐式课程学习（Implicit Curriculum Learning）

这是论文的一个重要发现：**ThinkTwice 自动实现了从易到难的课程学习，无需手动设计**。

机制如下：
- 训练初期：Phase 1 质量较差，大量错误 base solution 进入 Phase 2，精炼任务相对容易（发现错误并修正）；
- 训练中后期：Phase 1 质量提升，base solution 多数正确，Phase 2 需要更细粒度的审查（维持正确答案 or 发现细微错误），难度自然递增。

这与手动设计课程（如先用简单数学题再用竞赛题）的效果类似，但完全由训练动态自发涌现。

---

### 与 SCoRe 的对比

SCoRe（Kumar et al., 2024）也是联合训练推理和自我修正的代表方法，但存在关键差异：

| 方面 | ThinkTwice | SCoRe |
|------|------------|-------|
| 训练方式 | 两阶段**随机混合**，单次训练循环 | 两阶段**顺序训练**，Phase 2 完成后再 Phase 1 |
| 防 collapse 机制 | 无需额外约束（混合训练自然稳定） | 需要 KL 约束防止 Phase 2 退化 Phase 1 |
| Base solution 来源 | 历史 Phase 1 输出（off-policy） | 当前模型实时生成（on-policy） |
| Critique 标注 | 不需要 | 不需要 |
| 额外开销 | 仅 +3% wall-clock time | 较高（两阶段分别训练） |


## 📊 实验与性能评价

### 实验设置

- **模型**：Qwen3-4B、OLMo3-7B（两个不同规模和系列的基座模型）
- **数学推理基准**：
  - AIME（美国数学竞赛，高难度）
  - MATH（Hendrycks et al., 2021，标准数学）
  - Minerva Math（Lewkowycz et al., 2022，量化推理）
  - AMC（美国数学邀请赛）
  - OlympiadBench（He et al., 2024，奥林匹克级别）
- **评估指标**：pass@4（4 次采样中至少 1 次正确的概率）、精炼后 pass@4
- **对比基线**：
  - GRPO（原始算法，无精炼）
  - SCoRe（顺序两阶段 RL 精炼）
  - S2R（三阶段：reasoning + verification + correction）
  - Critique-GRPO（带 critique feedback 的 GRPO）
  - Self-Refine（训练无关迭代精炼）

---

### 主要结果：推理性能（Phase 1 pass@4）

| 方法 | Qwen3-4B AIME | Qwen3-4B 平均 | OLMo3-7B 平均 |
|------|--------------|--------------|--------------|
| GRPO | 39.06% | — | — |
| SCoRe | — | — | — |
| ThinkTwice | **44.11%** | — | **64.22%** |

ThinkTwice 在 AIME 上较 GRPO 提升 **+5.05 个百分点**，且在所有 5 个数学基准和 2 个模型上均超越所有基线（最高平均分）。

---

### 主要结果：精炼性能（Phase 2 pass@4）

| 方法 | Qwen3-4B AIME 精炼后 | Qwen3-4B 平均精炼后 | OLMo3-7B 平均精炼后 |
|------|---------------------|---------------------|---------------------|
| GRPO（直接精炼） | 48.91% | — | — |
| SCoRe | — | — | — |
| ThinkTwice | **60.43%** | **71.88%** | **69.35%** |

精炼后 AIME 较 GRPO 提升 **+11.52 个百分点**，展示了显著更强的自我精炼能力。

---

### 训练效率

- **额外计算开销**：仅 +3%（相对于标准 GRPO）
- **收敛速度**：ThinkTwice 比 GRPO 节省 **16% wall-clock time** 即达到最佳 checkpoint，推测是因为隐式课程学习提供了更丰富的训练信号，加速了策略优化。

---

### 消融实验

**Phase 混合比例（p）**：
- $p=0.5$（Phase 1 和 Phase 2 各占 50%）表现最优
- $p=1.0$（仅 Phase 1，即纯 GRPO）退化为基线
- $p=0.0$（仅 Phase 2）性能最差，模型失去基础推理能力

**Base Solution 选取策略**：
- 随机选取历史解（包含正确+错误混合）最佳
- 仅选取错误解：精炼能力提升但初始推理退化
- 仅选取正确解：模型学会"维持"但难以"纠错"

**精炼指令是否揭示正确性**：
- 不揭示（ThinkTwice 设计）优于揭示，后者反而限制了模型的泛化能力。

---

### 局限性（Limitations）

1. **仅在数学推理领域验证**：未在代码生成、科学推理等其他任务验证
2. **最大精炼轮数**：实验中最多测试了 1 次精炼，多轮精炼的稳定性未深入分析
3. **Base solution 来源**：使用历史离线数据，若模型进化过快，历史数据可能与当前策略差距过大（distribution shift 问题）
4. **模型规模**：仅测试了 4B 和 7B 规模，更大模型（如 70B+）上的效果未验证

---

## 💡 启发与我的想法

### 可借鉴之处

1. **随机混合训练的优雅性**：Phase 1 和 Phase 2 使用完全相同的奖励函数，避免了 reward hacking；随机混合避免了分阶段训练中的 forgetting 问题，是工程上非常简洁的设计。

2. **Off-policy Base Solution 的妙用**：用历史 Phase 1 输出作为 Phase 2 的输入，自然形成课程学习，且避免了 on-policy 方法中推理和精炼竞争计算资源的问题——这类似于经验回放（Experience Replay）在 DQN 中的作用。

3. **"不告诉模型答案对错"的精炼指令**：这是一个很有意思的训练信号设计。如果告诉模型"你的答案是错的"，模型学到的是"如何修复错误"；如果不告诉，模型需要同时学会"识别错误"和"修复错误"，泛化能力更强。

4. **与 KV Cache 优化的潜在联系**：多轮精炼场景中，每次精炼需要重新处理问题 + 历史答案，这恰好是 KV Cache 的强需求场景——保留问题的 KV Cache，只计算精炼过程中新增 token 的注意力，可以显著减少多轮精炼的计算开销。

### 改进空间

1. **多轮自我精炼的稳定训练**：如何在保持初始推理质量的同时训练多轮（3-5轮）精炼能力，需要研究 reward 信号随轮数的衰减问题。

2. **Non-math 领域迁移**：代码修复、科学推理等任务是否同样适用？这些领域的"正确性奖励"获取可能更复杂（如代码需要运行测试）。

3. **精炼质量 vs. 数量权衡**：当前是一次精炼，能否在推理时动态决定是否需要精炼（confidence-based early exit），避免对已经正确答案的无效精炼消耗？

4. **与稀疏注意力/高效推理的结合**：多轮精炼中，Problem context 的 KV Cache 应被最大化复用；结合 KV 稀疏技术（如 KVZap、H2O）可能进一步减少多轮精炼的显存占用。

### 下一步 Action
- [ ] 阅读 SCoRe (Kumar et al., 2024) 论文，深入理解两阶段分离训练的 collapse 问题
- [ ] 关注 ThinkTwice 在代码生成任务（HumanEval/MBPP）上的潜在应用
- [ ] 思考多轮精炼 + KV Cache 复用的系统设计（与 KvZap 笔记对接）


