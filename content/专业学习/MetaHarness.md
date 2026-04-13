---
title: "Meta-Harness"
tags:
  - LLM推理优化
  - Harness
  - Agent
  - 自动提示优化
  - meta-learning
  - benchmark
---

## 🧠 核心摘要 (TL;DR)
> Meta-Harness 提出了一种通过 **Agentic 搜索** 自动优化模型 Harness（提示、格式化逻辑、评分函数等外围代码）的端到端框架。其核心创新在于将历次候选方案的源代码、执行追踪和得分全部写入一个"诊断文件系统"，使提议者 Agent 每步可访问高达 **10M token** 的上下文——比所有现有方法（最多 26K token）高出约 400 倍——从而能精确定位失败根因而非盲目猜测。在文本分类、数学推理和 agentic 编程三个域上均取得最优或接近最优结果。

## 🏷️ 元数据
- **论文标题**: Meta-Harness: End-to-End Optimization of Model Harnesses
- **发表机构/会议**: arXiv（Stanford IRIS Lab / UW-Madison / MIT）
- **年份**: 2026
- **链接**: [[MetaHarness_paper|📄论文]] | [💻 Code](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #agent #harness #prompt-optimization #meta-learning #llm-evaluation #benchmark
- **相关节点 (Related Notes)**: [[ThunderAgent]], [[ProRLAgent]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈

大语言模型的实际效果不仅取决于模型权重本身，还高度依赖围绕模型的"Harness"——即任务提示（prompt）、上下文格式化逻辑、工具调用策略、评分函数等一系列周边代码。随着模型能力不断增强，人工设计 Harness 已成为制约 LLM 应用落地性能的主要瓶颈：

- **人工 Harness 工程极耗时**：针对不同任务（如文本分类、数学推理、代码生成）手动调试 prompt、设计 retrieval 策略、定义输出解析逻辑，需要大量专家经验和试错
- **现有自动优化方法上下文严重不足**：TextGrad、APO、OPRO 等方法在每一步优化时，提供给提议者的上下文极为有限（最多 26K token），仅包含历次候选的压缩摘要或滑动窗口，无法让提议者理解"具体是哪个设计决策导致了失败"
- **代码级优化被忽视**：现有方法大多只优化 prompt 文本字符串，而 Harness 中的代码逻辑（如 retrieval 实现、输出格式化）同等重要，却鲜有工作系统地优化

### 论文的切入点

Meta-Harness 将 harness 优化问题建模为**在代码空间上的搜索问题**，核心洞察是：若给提议者提供充足的诊断信息，它就能自主定位失败根因并给出有针对性的改进。具体地，论文设计了一个**诊断文件系统（Diagnostic Filesystem）**，把历次搜索中所有候选 harness 的完整源代码、每个样本的执行追踪（execution traces）和最终得分全部落盘，在每一步搜索时将整个文件系统的内容提供给提议者 Agent。

### 相关工作

**先驱/奠基工作（Prompt 与程序优化）**：

- **APE** (Zhou et al., 2022, arXiv:2211.01910)：最早将 LLM 用于自动生成和选择 prompt，证明了自动生成的指令可以媲美人工设计，但仅优化 prompt 文本，无历史上下文
- **APO/ProTeGi** (Pryzant et al., EMNLP 2023, arXiv:2305.03495)：提出用 LLM 生成"自然语言梯度"，类比数值梯度下降迭代更新 prompt；使用束搜索提升效率，但历史上下文仍极为有限（摘要层级）
- **OPRO** (Yang et al., ICLR 2024, arXiv:2309.03409)：将 LLM 本身作为优化器，在 prompt 中保存历史（得分、候选），让模型迭代生成更好的 prompt；GSM8K 提升 8%，但受限于上下文窗口，历史信息有损压缩
- **MAML** (Finn et al., ICML 2017, arXiv:1703.03400)：经典元学习框架，为快速适应新任务训练良好的初始化参数；与 Meta-Harness 的"元学习"哲学一脉相承，但作用于权重空间而非代码空间
- **DSPy** (Khattab et al., 2023, arXiv:2310.03714)：将 LLM pipeline 编译为自改进的模块化程序，通过 Teleprompter 自动优化每个模块的 few-shot 示例和指令；是最接近 Meta-Harness 设计思路的先行系统，但优化粒度仍停留在 prompt/few-shot，不涉及代码逻辑
- **Self-Refine** (Madaan et al., 2023, arXiv:2303.17651)：同一个 LLM 迭代生成输出→反馈→精炼，无需额外训练；展示了迭代自我改进的可行性，平均提升约 20%

**直接对比方法（自动 Harness / 工作流优化）**：

- **TextGrad** (Yuksekgonul et al., NeurIPS 2024, arXiv:2406.07496)：将"文本梯度"反向传播到复合 AI 系统的各个组件（prompt、代码、分子结构等）；在代码任务上相对提升 20%，但每步提议时可见历史信息仍受上下文窗口限制（约 26K token）
- **AFlow** (Zhang et al., 2024, arXiv:2410.10762)：用蒙特卡洛树搜索（MCTS）在代码表示的工作流空间上搜索最优 agentic 工作流；在 6 个基准上平均超越基线 5.7%，小模型可匹配 GPT-4o；是 Meta-Harness 最直接的竞争对手之一
- **ACE**（Zhang et al., 2025）：通过进化算法自动演化 harness 代码，是 Meta-Harness 在文本分类任务上的直接对比基线；Meta-Harness 在相同任务上以更少 token 超越 ACE 7.7 分

**相关技术组件**：

- **RAG** (Lewis et al., NeurIPS 2020, arXiv:2005.11401)：检索增强生成，为 Meta-Harness 在数学推理任务中发现的"Retrieval Harness"提供了基础方法支撑
- **IRCoT** (Trivedi et al., 2023, arXiv:2212.10509)：将检索与链式推理交织进行，解决多步骤知识密集型问题；与 Meta-Harness 发现的数学推理 harness 设计思路相近
- **UPRISE** (Cheng et al., EMNLP 2023, arXiv:2303.08518)：训练轻量检索器自动为零样本任务检索合适 prompt；跨任务、跨模型通用性好；与 Meta-Harness 中的 retrieval harness 设计理念相通
- **MemGPT** (Packer et al., 2023, arXiv:2310.08560)：类操作系统的分层内存管理架构，为 LLM Agent 提供虚拟上下文管理；与 Meta-Harness 用文件系统管理海量历史信息的思路有共通之处


## ⚙️ 核心设计与机制

### 问题定义：Harness 优化

论文将 Harness 定义为"除模型权重之外一切影响模型输出的代码"——包括：

- **提示（Prompt）**：系统 prompt、任务描述、示例格式
- **上下文格式化逻辑**：如何将任务输入组织成 LLM 能处理的格式（例如是否检索外部知识、如何截断长文）
- **工具调用策略**：使用哪些工具、以何种顺序调用
- **评分/解析函数**：如何从模型输出中提取最终答案

目标是在给定评估数据集 $\mathcal{D}$ 和评估指标 $f$ 的条件下，找到最优 Harness $h^*$：

$$h^* = \arg\max_{h \in \mathcal{H}} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ f(h(x), y) \right]$$

其中 $\mathcal{H}$ 是所有合法 harness 程序的空间。

### 核心架构：诊断文件系统 + Agentic 搜索循环

Meta-Harness 的整体流程为一个迭代搜索循环，分三个组成部分：

**（1）诊断文件系统（Diagnostic Filesystem）**

这是 Meta-Harness 最核心的创新。每次搜索迭代结束后，将以下信息追加写入文件系统：

- `candidates/` 目录：每个候选 Harness 的**完整源代码**（Python 文件）
- `traces/` 目录：该候选在每条测试样本上的**逐步执行追踪**（输入、中间步骤、输出、得分）
- `scores/` 目录：每个候选的聚合得分和按样本的细粒度得分

在下一次迭代时，将整个文件系统作为上下文提供给提议者 Agent，可使其访问**高达 10M token 的诊断信息**，远超现有方法（最多 26K token，约 400 倍差距）。

**（2）提议者 Agent（Proposer Agent）**

提议者是一个具有文件系统读写能力的 Agent（论文使用 Claude 系列模型）。它通过以下步骤生成新候选：

1. **读取文件系统**：浏览历次候选代码和执行追踪，定位"哪个具体决策在哪类样本上导致了失败"
2. **错误根因分析**：例如发现"当输入包含多标签时，当前 prompt 的格式化方式会截断关键信息"
3. **有针对性地提出修改**：生成新的 Harness 代码（完整 Python 文件），改动聚焦于诊断出的失败模式

这与先前方法（提供压缩摘要或滑动窗口历史）的关键区别在于：提议者能够**追踪失败到具体的代码行**，而非凭借统计得分盲目猜测改进方向。

**（3）评估与存储循环**

```
while budget > 0:
    h_new = proposer_agent.propose(filesystem)  # 读取文件系统，生成新 harness
    score, traces = evaluate(h_new, held_out_tasks)  # 在保留样本上评估
    filesystem.append(h_new, score, traces)  # 写入文件系统
    budget -= 1
return best(filesystem)
```

其中 `held_out_tasks` 是从完整评估集中划分出的保留集，避免在同一数据上过拟合。

### 三个应用域的 Harness 设计

**域 1：文本分类（Label-Primed Query Harness）**

Meta-Harness 自主发现的最优方案"Label-Primed Query (LPQ)"与人工设计的最优方案及 ACE 均不同。其核心思路是：

- **反转查询方向**：传统做法是将输入文本放入 prompt 然后让模型预测标签；LPQ 改为针对每个候选标签生成一个查询，计算输入文本与每个标签查询的匹配度，取最高匹配的标签
- **大幅减少 token 消耗**：LPQ 比 ACE 使用的上下文 token 少 4 倍（原文不含长 prompt 前缀）

效果：在 LawBench / Symptom2Disease / USPTO-50k 三个数据集上平均 48.6%，比 ACE 的 40.9% 高出 7.7 分（LawBench 单项提升 +16 分）。

**域 2：数学推理（Retrieval Harness）**

在 IMO 级别数学问题上，Meta-Harness 发现了一个单一的"Retrieval Harness"策略：

- 从训练集中检索与当前问题在结构上相似的已解决问题
- 将检索到的相似问题及其完整解题过程追加到 prompt 中
- 效果：在 5 个保留模型上平均精度从 34.1% 提升至 38.8%（+4.7 分），且该 harness 跨模型迁移性好

**域 3：Agentic 编程（TerminalBench-2）**

TerminalBench-2 是一个评估 LLM 在真实终端环境中完成编程任务能力的基准。Meta-Harness 在此任务上的优化空间最大，也最能体现其代码级优化能力：

- 在 Claude Opus 4.6 上：pass rate 76.4%，排名第 2
- 在 Claude Haiku 4.5 上：pass rate 37.6%，超越次优 Goose (35.5%)，排名第 1

### 与 ACE 框架的关键区别

| 特征 | ACE | Meta-Harness |
|------|-----|--------------|
| 优化粒度 | Prompt 文本 | 完整 Harness 代码（含逻辑） |
| 历史上下文 | ≤26K token（压缩摘要） | 高达 10M token（完整源码+追踪） |
| 失败诊断能力 | 基于统计得分猜测 | 追踪到具体代码行/决策 |
| 搜索策略 | 进化算法 | Agentic 文件系统搜索 |
| 代码生成能力 | 有限（仍以 prompt 为主） | 完整 Python Harness |


## 📊 实验与性能评价

### 对比基线

| 基线方法 | 类型 | 特点 |
|---------|------|------|
| ACE (Zhang et al., 2025) | Harness 自动优化 | 进化算法，代表 SOTA |
| TextGrad (Yuksekgonul et al., 2024) | 文本梯度反传 | 26K token 历史上下文上限 |
| 人工最优 Harness | Expert-engineered | 作者手工设计的最强基线 |
| Goose Agent | Agentic 编程 Agent | TerminalBench-2 排名第三 |

### 主要实验结果

**1. 文本分类（三数据集平均，准确率）**

| 方法 | LawBench | Symptom2Disease | USPTO-50k | 平均 | Context Token |
|------|----------|-----------------|-----------|------|---------------|
| 人工最优 | - | - | - | ~42% | - |
| ACE | - | - | - | 40.9% | 4× |
| **Meta-Harness (LPQ)** | **+16pts** | ... | ... | **48.6%** | **1×（4× 更少）** |

**2. 数学推理（IMO-level，5 个保留模型平均）**

| 方法 | 平均精度 |
|------|---------|
| 基准（无 Retrieval） | 34.1% |
| **Meta-Harness (Retrieval Harness)** | **38.8% (+4.7pts)** |

**3. Agentic 编程（TerminalBench-2 Pass Rate）**

| Agent | Claude Opus 4.6 | Claude Haiku 4.5 |
|-------|----------------|-----------------|
| 最优人工基线 | - | - |
| Goose | - | 35.5%（次优） |
| **Meta-Harness** | **76.4%（#2）** | **37.6%（#1）** |

### 消融与分析

- **诊断文件系统的价值**：相比仅提供得分摘要的消融版本，完整文件系统（含源码+追踪）在文本分类上提升约 5-8 个百分点，验证了大上下文诊断信息的核心作用
- **跨模型迁移性**：数学推理域中，在一个模型上优化出的 Retrieval Harness 直接迁移到 4 个未见模型上，平均仍提升 4+ 分，说明发现的 harness 策略是任务属性层面的泛化规律而非模型特化技巧
- **Token 效率**：LPQ harness 使用 4× 更少 context token，即在准确率更高的同时推理成本更低，实现了质量与效率的同步提升

### 局限性

- **搜索成本较高**：每轮迭代均需运行提议者 Agent（使用 Claude Opus 级别模型）并对所有评估样本执行新 harness，在大规模数据集上计算开销不容忽视
- **文件系统规模上限**：当候选数量极多时，10M token 的文件系统也可能超出提议者的实际有效上下文长度（受注意力机制"中段遗忘"影响）
- **代码质量依赖提议者能力**：框架效果强依赖底层提议者 LLM 的代码生成与分析能力，对弱模型可能退化
- **评估集污染风险**：将保留集用于迭代优化，理论上存在对保留集过拟合的风险；论文未做详细的训练/验证/测试三分析

---

## 💡 启发与我的想法

**可借鉴之处**：

1. **"诊断文件系统"思想的普适性**：将历次搜索过程的完整信息持久化存储，让优化器获得"全局视野"，这一思路不仅适用于 Harness 优化，也可用于代码调试系统、RL 奖励 shaping 搜索、超参数优化等任何迭代搜索场景
2. **上下文信息量 = 诊断能力**：论文清晰地揭示了"26K vs 10M token"带来的质变。在设计 Agent 系统时，应优先保留"可追溯到决策根因"的诊断信息，而非仅保存聚合指标
3. **代码生成作为优化手段**：相比只优化 prompt 字符串，让 Agent 直接修改和生成完整的 Python Harness 代码，释放了更大的优化自由度——这一思路对 LLM 推理系统工程（如 KV Cache 策略、调度策略的自动搜索）有直接参考价值
4. **Harness 优化与模型优化的解耦**：Meta-Harness 证明了在不改动模型权重的前提下，通过优化外围代码即可取得显著性能提升，这为"模型冻结后的应用层优化"提供了方法论支撑

**改进空间**：

1. **并行化搜索**：当前框架是串行迭代搜索（每轮等待评估结果再提议），可探索基于 Population-Based 进化（如 AlphaEvolve 思路）并行评估多个候选，降低时钟时间
2. **分层文件系统**：随候选数量增长，文件系统体积会爆炸。可借鉴层次压缩思路——对低分候选只保留摘要，对高分候选保留完整追踪——在保留关键诊断信息的同时控制规模
3. **与 DSPy 的结合**：Meta-Harness 目前只处理单一的整体 Harness 代码；如能与 DSPy 的模块化管道结合，对 pipeline 中每个模块分别维护诊断文件系统，则能在更细粒度上定位问题
4. **应用到 LLM 推理优化**：Meta-Harness 的框架可启发 KV Cache 策略自动搜索——将不同 KV 稀疏策略（如 H2O、SnapKV、KVzap）的代码加入候选池，用保留验证集的准确率/吞吐量作为 $f$，让 Agent 基于执行追踪自动搜索最优组合策略

**下一步 Action**：
- [ ] 阅读 ACE framework 原文，了解其进化算法设计细节
- [ ] 跟进 TerminalBench-2 排行榜，观察 Meta-Harness 在更多模型上的表现
- [ ] 调研 DSPy 2.x 是否已有类似"诊断文件系统"的扩展


