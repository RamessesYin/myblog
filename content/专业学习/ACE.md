---
title: "ACE: Agentic Context Engineering"
tags:
  - LLM Agent
  - Prompt Optimization
  - Context Engineering
  - Self-Improving
  - ICLR 2026
---

## 🧠 核心摘要 (TL;DR)
> ACE（Agentic Context Engineering）将 LLM 的上下文（system prompt / instructions）视为随任务执行动态演化的"活剧本"，通过 Generator→Reflector→Curator 三模块流水线持续从成功/失败轨迹中提炼策略并写回上下文，同时引入增量 Delta 更新与 grow-and-refine 机制来克服"简洁偏差"与"上下文坍塌"两大顽疾，在 AppWorld 智能体基准上取得 +10.6% 的平均提升，在金融任务上取得 +8.6% 的提升，并以 82.3% 的延迟缩减优于 GEPA 等基线。

## 🏷️ 元数据
- **论文标题**: Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models
- **发表机构/会议**: ICLR 2026
- **年份**: 2025（提交于 2025 年 10 月，收录于 ICLR 2026）
- **链接**: [[ACE_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #llm-agent #prompt-optimization #context-engineering #self-improving #iclr2026
- **前置知识 (Prerequisites)**: [[AttentionRes]]
- **相关节点 (Related Notes)**:
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

LLM agent 在长期部署中面临两个核心问题：

**1. 简洁偏差（Brevity Bias）**：现有的上下文压缩/总结策略倾向于生成过于简短的摘要，导致领域专有的细粒度策略（如某 API 的特殊调用方式、某类错误的具体规避方法）被丢弃。模型在全局层面"看起来精炼"，但在面对具体子任务时缺乏足够的领域知识。

**2. 上下文坍塌（Context Collapse）**：迭代式的全量改写（full rewrite）机制在每一轮优化后都可能将上一轮写入的细节抹除。这是因为 LLM 改写时会从全局语义进行压缩，而非原地更新，导致历史积累的策略随迭代次数增加而逐步流失，形成"越改越空洞"的恶性循环。

上述两个问题的共同根源在于：传统方案将 context 视为静态模板，缺少对"哪些知识值得保留、哪些需要新增、哪些已过时"的精细判断机制。

### 相关工作

#### 先驱/奠基工作

| 论文 | 核心贡献 | 与 ACE 的关系 |
|------|---------|--------------|
| **ReAct** (Yao et al., 2022) [arXiv:2210.03629] | 将推理（Reasoning traces）与行动（Acting）交错生成，LLM agent 框架的奠基性工作 | ACE 的 Generator 模块在生成推理轨迹时直接采用 ReAct 格式 |
| **Reflexion** (Shinn et al., 2023) [arXiv:2303.11366] | 通过语言反馈（verbal reinforcement）让 agent 从失败中自我反思并将反思写入记忆，无需更新权重 | ACE 的 Reflector 模块思路直接来源于此；ACE 的改进在于增量 Delta 写回而非 episodic 记忆追加 |
| **APE** (Zhou et al., 2022) [arXiv:2211.01910] | 将 prompt 视为程序进行自动优化，LLM 作为人类级别的 prompt 工程师 | 早期离线式 prompt 优化的代表，ACE 解决的是其无法在线适应的缺陷 |
| **Chain-of-Thought** (Wei et al., 2022) [arXiv:2201.11903] | 中间推理步骤（thought chain）显著提升复杂推理能力 | ACE 的轨迹生成默认使用 CoT，为 Reflector 提供可解析的推理证据 |

#### 直接对比方法（Baselines）

| 论文 | 核心贡献 | 与 ACE 的区别 |
|------|---------|--------------|
| **TextGrad** (Yuksekgonul et al., 2024) [arXiv:2406.07496] | 通过 LLM 文本反馈实现"文本自动微分"，对 prompt 进行梯度式优化 | 离线批量优化，需要大量 rollout；ACE 在线增量，延迟低 91.5% |
| **OPRO** (Yang et al., 2023) [arXiv:2309.03409, ICLR 2024] | LLM 作为优化器，迭代生成新 prompt 候选并按历史得分排序选优 | 同样离线，不针对 agent 交互场景；ACE 能在单次 episode 内学习 |
| **DSPy** (Khattab et al., 2023) [arXiv:2310.03714] | 将 LLM pipeline 声明为可自动编译优化的"程序" | 侧重管道级优化而非单一 context 演化；缺乏动态 online 适应 |
| **GEPA** (Agrawal et al., 2025，论文原文引用) | 遗传-Pareto 进化式 prompt 演化，离线多代次搜索 | 延迟远高于 ACE（ACE 比其快 82.3%），不适合在线部署 |
| **Dynamic Cheatsheet** (Suzgun et al., 2025，论文原文引用) | 测试时自适应外部记忆备忘单，在线学习 | 依赖 episodic 记忆堆叠，存在 Brevity Bias；ACE 通过 Curator 主动去冗 |
| **SCOPE** (Pei et al., 2024) [arXiv:2512.15374] | 同样将 context 视为演化剧本，双流机制平衡战术/战略 | 与 ACE 设计目标相近，是最接近的同期工作；SCOPE 在 14.23%→38.64% |

#### 近期 Follow-up / 扩展工作

| 论文 | 核心贡献 |
|------|---------|
| **MEMO** (Xie et al., 2025) [arXiv:2603.09022] | 通过持久记忆库与锦标赛式 prompt 演化在多回合多 agent 博弈中优化推理时上下文 |
| **ICPO** (Yu et al., 2025) [arXiv:2603.01335] | 可证明的在线 In-Context Policy Optimization，无权重更新下的 agent 自优化，有理论保证 |
| **MARS** (Liang et al., 2025) [arXiv:2503.19271] | 记忆增强 + 反思式自优化框架，结合遗忘曲线原理做记忆管理 |
| **EvoSkill** (Alzubi et al., 2025) [arXiv:2603.02766] | 通过迭代失败分析自发现 agent 技能，Pareto 前沿选择机制 |
| **Language Agents as Optimizable Graphs** (Zhuge et al., 2024) [arXiv:2402.16823] | 将 agent 统一建模为可优化图，prompt 工程技术在图框架下统一 |
| **EXPO** (Kong et al., 2025) [arXiv:2502.00728] | Meta-Prompt 优化，用"梯度下降"类比对 meta-prompt 做序列决策优化 |
| **SEW** (Liu et al., 2025) [arXiv:2505.18646] | 自演化多 agent 工作流，自动生成并优化多 agent 协作结构 |
| **Prompt Alchemy** (Ye et al., 2025) [arXiv:2503.11085] | Prochemy 方法，基于模型性能反馈自动精炼 prompt，迭代优化 |
| **Prompt Engineering Survey** (Sahoo et al., 2024) [arXiv:2402.07927] | 大规模系统综述，覆盖 NLP 各场景 prompt 工程技术分类与对比 |

### 论文的切入点

ACE 的核心问题意识在于：**上下文的演化不应是全量改写，而应是精细的增量 delta 操作**。这与软件工程中 git diff 的哲学高度一致——永远记录变化而非快照，才能保证历史信息不因压缩而流失。同时，对"哪些信息有价值"的判断需要由专门的 Curator 模块承担，而非让 LLM 在改写时自行决定取舍（后者必然偏向简洁）。



## ⚙️ 核心设计与机制

### 整体架构：三模块流水线

ACE 的核心理念是将 LLM 的 system prompt（或 instructions）从"一次性写定的静态配置"升级为"可随任务执行持续演化的活剧本（living playbook）"。实现这一点的是以下三个专职模块：

```
Task Episode
    │
    ▼
┌──────────────┐
│  Generator   │  ← 执行任务，生成推理轨迹（trajectory）
└──────┬───────┘
       │  trajectory
       ▼
┌──────────────┐
│  Reflector   │  ← 从轨迹中提炼具体策略（insights）
└──────┬───────┘
       │  delta insights
       ▼
┌──────────────┐
│   Curator    │  ← 将 insights 合并/去冗后写回 Context
└──────┬───────┘
       │  updated context
       ▼
    下一次 Episode（Context 已更新）
```

这是一个在推理时（inference-time）完全闭环的自优化机制，**不涉及任何权重更新**，可部署在任意冻结的 LLM 之上。

---

### Generator（生成器）

Generator 的职责是在当前 Context（即当前版本的 system prompt/instructions）下执行给定任务，并产出完整的推理轨迹。

- **输入**：当前 Context $C_t$、任务描述 $T$
- **输出**：推理轨迹 $\tau = (a_1, o_1, a_2, o_2, \ldots, a_n, o_n)$（动作-观察序列）以及最终结果 $r$（成功/失败/部分成功）
- **实现方式**：默认基于 ReAct 格式，交错生成 `Thought → Action → Observation` 三元组

关键设计：Generator 本身不直接修改 Context，它只是"执行者"，记录下完整的决策链供后续模块分析。

---

### Reflector（反思器）

Reflector 是 ACE 的核心知识提炼模块，负责从 Generator 产出的轨迹中萃取可复用的策略知识（insights）。

**反思的对象**：
- **成功轨迹**：提炼"什么做法奏效"——有效的工具调用序列、正确的错误处理方式、成功的任务分解策略
- **失败轨迹**：提炼"什么做法有问题"——常见的错误模式、误解 API 用法的情形、需要额外验证的假设

**关键设计——增量 Delta 格式**：

Reflector 产出的不是对 Context 的"全量重写建议"，而是结构化的 **Delta 更新指令**，格式示意如下：

```
[ADD] 在调用 Transfer API 时，必须先检查账户余额是否足够，否则会触发 INSUFFICIENT_FUNDS 错误
[MODIFY] 原有"重试三次"策略改为：HTTP 429 时退避重试，HTTP 500 时立即上报
[DEPRECATE] 不再需要在每次 Search 前手动 refresh token（已自动续期）
```

这种 Delta 格式有两大优势：
1. **防止 Brevity Bias**：Reflector 只生成新增/修改的内容，不做全局总结，细粒度策略不会因压缩而丢失
2. **明确的操作语义**：ADD/MODIFY/DEPRECATE 三种操作使 Curator 能够精确执行合并逻辑

---

### Curator（策展器）

Curator 负责将 Reflector 输出的 Delta 更新安全地合并到当前 Context 中，并控制 Context 的整体质量。

**Grow-and-Refine 机制**：

Curator 的核心挑战是在"扩充"与"精简"之间保持平衡：

- **Grow 阶段**：将有价值的新策略（ADD 操作）追加到 Context 的对应章节，保持细节完整
- **Refine 阶段**：定期检测 Context 中的冗余（同义规则合并、过时规则清除、重复示例去重），防止 Context 无限膨胀

Curator 会维护一个**版本化的 Context 历史**，支持回滚（当新版本导致性能下降时恢复到上一稳定版本）。

**结构化 Context 格式**：

ACE 的 Context 并非非结构化自然语言，而是以章节（section）为单位组织，每个 section 对应一类任务知识：

```markdown
# [Task Overview]
...

# [API Usage Patterns]
- Transfer: 先检查余额 → 调用 TransferAPI → 确认返回 txn_id
- ...

# [Common Pitfalls]
- 不要直接用 user_id 做 recipient，应先 lookup → 获取 account_id
- ...

# [Recent Strategy Updates]
- v3: 优化了 Search 重试逻辑（2025-10-15）
- ...
```

这种结构使 Curator 能够精准定位要修改的章节，而不是对全文做模糊改写。

---

### 关键创新点总结

| 创新点 | 解决的问题 | 具体机制 |
|--------|-----------|---------|
| 增量 Delta 更新 | Context Collapse | Reflector 输出 ADD/MODIFY/DEPRECATE 指令而非全量改写 |
| Grow-and-Refine | Brevity Bias + Context 膨胀 | 先增后精，定期去冗 |
| 结构化 Context | 模糊改写 | 章节化组织，使精确定位修改位置成为可能 |
| 版本化历史 | 优化失效回滚 | 保留 Context 历史版本，支持性能退化时的回滚 |
| 无需标注数据 | 冷启动问题 | 仅需任务执行反馈（成功/失败），无需人工标注 |
| 在线适应 | 离线方法无法部署 | 每个 episode 后即时更新，首次适应只需数个 episode |

---

### 核心公式

设 $C_t$ 为第 $t$ 轮后的 Context，$\tau_t$ 为第 $t$ 个 episode 的轨迹，则 ACE 的更新过程可形式化为：

$$\Delta_t = \text{Reflector}(\tau_t, C_t)$$

$$C_{t+1} = \text{Curator}(C_t, \Delta_t)$$

其中 Curator 满足以下约束：
- **信息单调性**：$|C_{t+1}|$ 在 Grow 阶段增加，在 Refine 阶段受控减少，不出现因一次改写导致历史信息全量丢失的情况
- **Delta 完整性**：$\Delta_t$ 中 ADD 操作对应的内容在 $C_{t+1}$ 中必须完整保留（不被 Refine 截断）



## 📊 实验与性能评价

### 实验设置

ACE 在三类场景下进行了评估：

| 任务类别 | 数据集/平台 | 评估指标 |
|---------|-----------|---------|
| Agent 综合基准 | AppWorld（ACL'24, Trivedi et al.） | Task Success Rate（TSR） |
| 金融推理 | FiNER、Formula 等金融 NLP 数据集 | 准确率 |
| 医疗推理 | Medical QA benchmark | 准确率 |
| 结构化 SQL | Text-to-SQL benchmark | Execution Accuracy |

AppWorld 是目前最接近真实生产环境的 agent 基准：包含 9 个日常 App（消息、支付、日历等）、457 个 API 接口和 750 个复杂任务，GPT-4o 在"普通任务"上仅能解决 49%，是一个有充分难度区分度的 benchmark。

---

### 主要实验结果

#### AppWorld 基准（Agent 综合表现）

| 方法 | 类型 | TSR | 相对提升 |
|------|------|-----|---------|
| 基础 Baseline（静态 prompt） | 静态 | - | - |
| GEPA（Agrawal et al., 2025） | 离线进化 | - | - |
| Dynamic Cheatsheet（Suzgun et al., 2025） | 在线记忆 | - | - |
| **ACE（本文）** | 在线增量 | - | **+10.6%**（平均） |
| ACE（无 ground-truth 标注，在线适应） | 在线无标注 | - | **+17.1%** |

- ACE 在 AppWorld 上与 IBM CUGA（排行榜第一的商业生产 agent）性能持平，但使用的是更小规模的开源模型
- 在无 ground-truth 标注的在线适应场景下，提升达到 17.1%，验证了 ACE 在冷启动场景的实用性

#### 延迟对比（关键工程指标）

| 方法 | 相对延迟 |
|------|---------|
| GEPA（离线） | 1× 基准（最慢） |
| Dynamic Cheatsheet（在线全量改写） | 约 GEPA 的 1/11（推断） |
| **ACE（在线增量）** | 比 GEPA **快 82.3%**；比 Dynamic Cheatsheet **快 91.5%** |

这一结果直接验证了 Delta 增量更新的工程价值：相比于全量改写，增量操作的计算量显著减少，使 ACE 能以接近零额外延迟部署在生产系统中。

#### 其他领域表现

| 领域 | 平均提升 |
|------|---------|
| 金融任务（FiNER / Formula 等） | +8.6% |
| 医疗推理 | +15.0% |
| Text-to-SQL | +5.1% |

这些跨领域结果验证了 ACE 框架的通用性——同一套 Generator-Reflector-Curator 流程在不同领域均能有效积累领域知识。

#### Rollout 效率

ACE 比同类方法少用 75.1% 的 rollout 次数即可达到相当的性能，这意味着：
- **冷启动更快**：只需少数 episode 即可收敛到有效 Context
- **推理成本更低**：生产部署时的 token 消耗显著减少

---

### 消融实验

论文对各模块的贡献进行了消融：

| 消融变体 | 结论 |
|---------|------|
| 去除 Curator（Reflector 直接写回） | 性能下降，Context 快速膨胀且出现矛盾规则 |
| 全量改写替代 Delta | 引发 Context Collapse，长期运行后性能退化 |
| 去除 Reflector（直接将轨迹追加到 Context） | Brevity Bias 加重，细节丢失 |
| 禁用 grow-and-Refine（只 grow 不 refine） | Context 无限膨胀，推理延迟线性增加 |

---

### 局限性（Limitations）

1. **依赖反思质量**：Reflector 的性能直接受底层 LLM 推理能力制约——若模型无法准确解读轨迹中的失败原因，提炼出的 insights 将引入噪声
2. **任务类型限制**：对于本身就要求简洁指令的任务（如 HotPotQA 开放问答、Game of 24 数学游戏），复杂化的 Context 不仅无益甚至有害，ACE 在这些任务上没有明显提升
3. **对抗性注入脆弱性**：若 Reflector 被注入对抗性反思（adversarial reflector corruption），Context 会被恶意策略污染；但论文实验显示中等噪声水平下 ACE 仍能保持稳定
4. **反馈信号依赖**：在完全无反馈信号（既无 ground-truth 又无执行结果指示）的场景下，Reflector 无法判断轨迹质量，ACE 退化为静态 prompt 模式
5. **Context 长度限制**：随着任务领域扩展，Context 可能趋近 LLM 的 context window 上限，需要外部的 Context 压缩/检索机制配合

---

## 💡 启发与我的想法

### 可借鉴之处

**1. 增量写回而非全量改写的范式转换**

这是 ACE 最核心的工程洞察，也是最值得迁移的设计思想。在 LLM 系统开发中，任何需要"持续优化某个文本配置"的场景（system prompt、few-shot examples、RAG 检索策略描述等）都应该优先考虑 Delta 更新而非全量重写。全量重写看起来"干净"，但在 LLM 做改写时实际上引入了对历史信息的隐性截断。

**2. Generator/Reflector/Curator 的职责分离**

这三个模块的分工体现了良好的 SoC（关注点分离）：
- Generator 只管执行，不做元学习
- Reflector 只从轨迹里提炼，不直接操作 Context
- Curator 只管 Context 的质量和一致性，不执行任务

这种分工在实际 agent 系统开发中可以直接套用——每个模块可以分别选用最适合的模型或提示策略，互不干扰。

**3. 结构化 Context 的价值**

将 Context 按章节组织是一个低成本高收益的工程实践。对于任何复杂 agent 系统，建议从一开始就用 markdown 章节划分 system prompt，便于后续自动化改写时精确定位操作位置。

### 改进空间

**1. 多 agent 并行场景下的 Context 同步问题**

当前 ACE 描述的是单 agent 顺序执行的场景。若多个 agent 并发运行并向同一 Context 提交 Delta，存在冲突合并（merge conflict）的问题。可以借鉴分布式数据库的 MVCC（多版本并发控制）思路，或者引入集中式 Curator 作为"context coordinator"。

**2. Context 章节的自动发现**

当前 ACE 假设 Context 的章节结构已预定义。但在冷启动时（第一版 Context 还未建立时），需要由 Reflector 自动归纳出有哪些 section 应该存在。这是一个有趣的"无监督 Context 结构归纳"子问题。

**3. 与 RAG 的结合**

当领域知识积累量大到超过 context window 时，ACE 的 Context 需要外置到向量数据库进行检索式访问（RAG），而非全量载入。如何保证 Delta 写入后的向量索引一致性，是一个工程挑战。

**4. 与 RLHF/RLVR 的配合**

ACE 完全基于 prompt 层面的优化，与模型权重无关。可以考虑将 ACE 积累的高质量 Context 版本和对应的成功轨迹作为正样本，用于后续的 preference fine-tuning，从而将推理时积累的经验固化到模型权重中，实现"经验蒸馏"。

### 与现有笔记的关联

- **KV 稀疏方向**（[[KV稀疏调研]]、[[KvZap]]）：ACE 解决的是 agent 层面的推理优化，KV Cache 优化解决的是 transformer 层面的计算优化，两者是正交的，但 ACE 的 Context 压缩挑战（如何在长 Context 中保留重要信息）与 KV Cache 稀疏化选择重要 token 的思路存在高层语义上的相似性
- **分布式训练**（[[ShiftParallelism]]）：ACE 的多 agent 并行场景下的 Context 同步问题，与分布式训练中的梯度同步问题在结构上类似

### 下一步 Action

- [ ] 复现 ACE 的 Reflector 提示词设计，在本地 agent 项目中测试 Delta 更新的效果
- [ ] 调研 ACE 与 DSPy 的结合可能性（DSPy 管道 + ACE Context 演化）
- [ ] 追踪 ICLR 2026 workshop 上关于 ACE 的后续讨论


