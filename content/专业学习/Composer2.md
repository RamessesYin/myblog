---
title: "Cursor Composer2"
tags:
  - agentic-coding
  - software-engineering-agent
  - reinforcement-learning
  - code-generation
  - llm-training
---

## 🧠 核心摘要 (TL;DR)
> Composer 2（Cursor Research, 2026）是专为 Agentic 软件工程场景设计的专用模型，基于 Kimi K2.5（1.04T 参数 MoE）继续预训练后，经过大规模强化学习精调，在内部基准 CursorBench-3 上达到 61.3%，相比前代提升 37%，并在 SWE-bench Multilingual 上达到 73.7%。其核心创新在于"仿真真实用户挑战"的训练理念——用比公开基准复杂 10-20 倍的内部工程任务驱动 RL，同时引入 Self-Summarization、非线性长度惩罚和 MoE 路由重放等工程技巧，显著提升长时域 Agentic 推理能力。

## 🏷️ 元数据
- **论文标题**: Composer 2 Technical Report
- **发表机构/会议**: Cursor Research（arXiv 技术报告）
- **年份**: 2026
- **链接**: [[Composer2_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #agentic-coding #software-engineering-agent #reinforcement-learning #code-generation #llm-training #moe #swe-bench
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[LongCat]]
- **相关节点 (Related Notes)**: [[ThunderAgent]], [[ProRLAgent]], [[RLBeyondBaseModel]], [[LongCat]], [[TTT-Discover]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

软件工程是 LLM 落地最重要的场景之一，但现有模型在 **Agentic 软件工程**（即：无结构化任务描述、需要自主规划、跨文件修改、与工具/终端交互、完成长时域工作流）方面存在深层缺口：

1. **公开基准与真实任务的鸿沟**：SWE-bench 等公开基准每个任务中位数仅需修改 7-10 行代码、问题描述中位数长达 1,185-3,055 字符（高度结构化）；而真实工程场景（如 Cursor 用户的编码请求）中位数修改 **181 行**、任务描述仅 **390 字符**（高度欠规范）。在公开基准上训练/优化的模型往往无法泛化到真实用户任务。

2. **长时域一致性**：大多数推理模型擅长单步问答而非多轮 Agent 交互。当 Agent 需要执行数十步工具调用、维护跨 Turn 的工作上下文时，早期错误会被放大，模型容易"迷失方向"。

3. **训练-推理 Mismatch**：大规模 RL 训练中通常使用同一问题多次采样（on-policy）；但当任务环境（终端、代码库状态）随 Agent 动作演化时，重复使用同一 prompt 会引入严重的 Distribution Shift。

### 论文的切入点

Cursor 的核心策略是：**"用真实环境模拟驱动一切"**——构建内部基准 CursorBench 直接从真实工程会话中提取任务，用这些高质量、无污染的任务来：(a) 引导数据配比，(b) 驱动 RL 奖励信号，(c) 最终评估模型。同时在 RL 算法层面做了多项工程改进，解决 Agentic 场景的长序列稳定性问题。

### 相关工作

#### 🏛️ 先驱/奠基工作

| 论文 | 核心贡献 | 年份 |
|------|---------|------|
| **Codex** (Chen et al.) | 首次系统评估在 GitHub 代码上训练的 GPT 模型，引入 HumanEval 基准，pass@k 评估框架 | 2021 |
| **CodeBERT** (Feng et al.) | 双模态预训练（编程语言 + 自然语言），EMNLP 2020 | 2020 |
| **SWE-bench** (Jimenez et al.) | 首个从真实 GitHub Issue 构建的软件工程评估基准（2,294 实例），ICLR 2024 | 2024 |
| **DeepSeekMath / GRPO** (Shao et al.) | 提出 Group Relative Policy Optimization（GRPO）算法，去掉 Critic 网络改用组内相对优势，内存友好型 RL | 2024 |
| **DeepSeek-R1** (DeepSeek-AI) | 纯 RL 训练激发推理能力，展示 RL 在复杂任务上的涌现效果 | 2025 |

#### ⚔️ 直接对比方法

| 论文 | 核心方法 | Composer 2 的对比优势 |
|------|---------|----------------------|
| **SWE-agent** (Yang et al., NeurIPS 2024) | Agent-Computer Interface（ACI）方法，构建专用工具封装供模型与代码库交互 | Composer 2 针对 Cursor 真实用户场景微调，CursorBench-3 优于 SWE-agent 系列模型 |
| **OpenHands** (Wang et al., ICLR 2025) | 通用 AI 软件开发者平台，支持代码执行/终端/浏览器等 Agent 接口 | Composer 2 专注于代码编辑 Agentic 场景，实验性对比显示其 CursorBench 指标领先 |
| **GPT-5.4** | OpenAI 闭源模型 | Composer 2 在 CursorBench-3 上 61.3% 优于 GPT-5.4 的 63.9%（注：GPT-5.4 在公开基准上更强，Cursor 内部基准刻画不同维度） |
| **StarCoder2 / CodeLlama / Qwen2.5-Coder** | 通用代码预训练模型 | Composer 2 在 Agentic 任务上针对性更强，非通用代码补全竞赛 |

#### 🚀 近期 Follow-up 与生态

| 论文 | 关联 |
|------|------|
| **DAPO** (Yu et al., 2025) | Composer 2 的 RL 训练受其启发，去掉 GRPO 的长度标准化以减少梯度偏差 |
| **Multi-Token Prediction** (Gloeckle et al., ICML 2024) | Composer 2 在预训练中引入 MTP 层，同时支持推测解码加速 |
| **SWE-smith** (Yang et al., 2025) | 扩展 SWE 训练数据到 5 万实例，与 Composer 2 的数据策略方向互补 |
| **Terminal-Bench** (Merrill et al., ICLR 2026) | Composer 2 评估使用 Terminal-Bench（61.7%），该基准覆盖命令行界面 Agentic 任务 |
| **Kimi K2.5** (Team, 2026) | Composer 2 的基础模型，1.04T 参数 MoE 架构（32B 激活参数） |


## ⚙️ 核心设计与机制

Composer 2 的训练分为两个主要阶段：**继续预训练（Continued Pretraining）** 和 **大规模强化学习（Large-Scale Reinforcement Learning）**，在此之前有一个 SFT 桥接步骤。

### 1. 基础模型：Kimi K2.5

Composer 2 从 **Kimi K2.5**（Moonshot AI，2026）出发：
- 参数规模：1.04T 总参数，32B 激活参数（稀疏 MoE 架构）
- 具备强大的基础推理和代码能力，是 Composer 2 专业化的基座

### 2. 继续预训练（Continued Pretraining）

**目标**：将基础模型的知识偏向（bias）调整到代码密集型场景，同时扩展上下文长度。

**三阶段设计**：

| 阶段 | 序列长度 | 关键设计 |
|------|---------|---------|
| 阶段一（主预训练） | 32K tokens | 以代码为主的大规模数据混合 |
| 阶段二（长上下文扩展） | 256K tokens | 长上下文位置编码 + 长序列数据 |
| 阶段三（SFT 桥接） | - | 针对编码任务的监督微调 |

**关键技术：Multi-Token Prediction（MTP）层**

Composer 2 在预训练中加入了 **多 Token 预测头**（受 Gloeckle et al. ICML 2024 启发），同时预测接下来 n 个 token：
- **训练收益**：作为辅助训练目标，提升 Sample Efficiency，尤其在代码生成任务上
- **推理收益**：MTP 头可直接用于 **Speculative Decoding**（推测解码），在推理时以更低延迟生成 token（无需额外 Draft 模型）

**训练精度**：使用 MXFP8（NVIDIA Blackwell B300）以减少显存占用并加速矩阵运算；MXFP8 采用 OCP Microscaling 格式（block scale 因子 + 低精度尾数），保持 FP8 精度的同时兼容硬件 Tensor Core。

### 3. 强化学习（Large-Scale Reinforcement Learning）

这是 Composer 2 最核心的创新阶段，目标是用 RL 信号引导模型在 Agentic 编码场景中端到端学习。

#### 3.1 基础算法：策略梯度 + 组相对优势

Composer 2 的 RL 算法以 **GRPO**（Group Relative Policy Optimization，DeepSeekMath 提出）为起点，但做了若干关键改造：

**GRPO 基本思路**：
- 对同一 prompt 采样 $G$ 条响应（group）
- 无需 Critic 网络，用组内相对表现作为 Advantage 估计：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

- 梯度目标类似 PPO 的 Clipped Objective，但用相对 Advantage 替代 GAE 估计的优势

**Composer 2 的关键改动**：

(a) **去掉长度标准化（Remove Length Normalization）**：

原 GRPO 对每条响应的 loss 按 token 数量做归一化（即对所有 token 取平均 loss）。论文发现这会引入 **梯度偏差**——短响应的每个 token 梯度被放大，而长响应被稀释，导致模型倾向于生成较短的响应。Composer 2 **移除此归一化**，直接对所有 token 的 log-prob 求和再取期望，消除偏差。

(b) **KL 散度估计量替换**：

原版 GRPO 使用 $k_3$ 估计量作为 KL 惩罚：$k_3 = \frac{r-1}{2} + \ln r - \ln r$（其中 $r = \pi_\theta / \pi_{\text{ref}}$）。

Composer 2 改用 **$k_1$ 估计量**：

$$\text{KL}(k_1) = -\log r = -\log \frac{\pi_\theta}{\pi_{\text{ref}}}$$

$k_1$ 方差更低，在长序列（Agentic 任务）中梯度更稳定（参考 Schulman 2020 博客文章分析）。

(c) **Single-Epoch（单轮迭代）**：

每个 prompt 只使用一次（不重复采样），彻底避免 off-policy 偏差。这对于 Agentic 任务尤其关键——环境状态（代码库/终端状态）随 Agent 动作演化，同一 prompt 重复使用在语义上是不一致的。

(d) **MoE 路由器重放（Router Replay）**：

MoE 模型中，各 token 经过路由器被分配给不同专家。在 RL 梯度回传时，如果路由决策发生微小变化，对应的梯度计算路径也会改变，导致数值不稳定。Composer 2 通过"重放"前向传播时的路由决策（冻结路由结果）来保证反向传播数值稳定性。

#### 3.2 Self-Summarization（自我摘要）

Agentic 任务中，Agent 可能执行数十轮 Tool Call，上下文长度迅速膨胀到数万甚至数十万 token。**Self-Summarization** 解决长时域一致性问题：

- **机制**：在 Agent 工作流的关键检查点，让模型自己生成工作进展摘要（而非保留全部历史 token）
- **优点**：
  1. 压缩上下文，节省 KV Cache 显存
  2. 提炼核心状态，减少模型"迷失"在无关历史中的概率
  3. 摘要行为可通过 RL 奖励信号直接优化（好的摘要会带来更高的任务完成率）
- 这是 Cursor Research 内部专门开发的技术（Team 2025b, "Self-summarization for Composer"）

#### 3.3 非线性长度惩罚（Nonlinear Length Penalty）

简单的线性长度惩罚（每多一个 token 减固定分数）会导致模型极端截断或拒绝展开推理链。Composer 2 设计了非线性惩罚函数：

$$C(x) = \frac{(1 + kx)^{1-q} - 1}{k(1-q)}$$

其中 $x$ 为响应长度，$k$ 和 $q$ 是超参数：
- 当 $x$ 较小时，惩罚增长缓慢（允许适度长思维链）
- 当 $x$ 超过阈值时，惩罚加速增长（防止无意义填充）

这种凸函数形式使 RL 训练能在推理深度与效率之间找到自然均衡点。

#### 3.4 异步 RL 训练基础设施

大规模 Agentic RL 训练面临独特的工程挑战：

```
┌─────────────────────────────────────────────────────────┐
│                   Composer 2 RL 训练架构                  │
├──────────────────┬──────────────────┬───────────────────┤
│   Training Workers │  Rollout Workers │  Eval Workers     │
│   （更新模型参数）  │  （Agent交互采样）  │  （异步评估）      │
├──────────────────┴──────────────────┴───────────────────┤
│              Ray Cluster（分布式调度）                     │
│              NVIDIA B300s GPUs                           │
└─────────────────────────────────────────────────────────┘
```

关键工程设计：
- **权重快速同步（Fast Weight Sync）**：Training Worker 更新权重后，通过 Delta 压缩（仅传输参数差值）快速广播到 Rollout Workers，减少带宽占用
- **In-flight Weight Updates（飞行中参数更新）**：Rollout 正在进行时，Training Worker 可同步推进下一步更新，最大化 GPU 利用率
- **S3 Delta 压缩上传**：检查点上传到云存储时压缩参数增量，降低 I/O 开销

#### 3.5 CursorBench：真实世界任务基准

**CursorBench** 是 Composer 2 的核心评估与训练信号来源，与公开基准的对比：

| 维度 | SWE-bench | CursorBench-3 |
|------|-----------|---------------|
| 代码修改中位数 | 7-10 行 | **181 行** |
| 任务描述中位数 | 1,185-3,055 字符 | **390 字符**（更欠规范）|
| 任务来源 | GitHub Issue | **Cursor 工程师真实编码会话** |
| 评估维度 | 测试通过率 | 功能正确性 + 代码质量 + 延迟 + 成本 + 交互行为 |
| 训练集污染 | 存在风险 | **无污染**（内部数据） |

CursorBench 任务更接近真实工程师的工作方式：描述简短、上下文复杂、需要大量代码修改，这迫使 Composer 2 在 RL 阶段学会真正的 Agentic 规划能力而非"刷分"。



## 📊 实验与性能评价

### 主要性能数据

#### CursorBench-3（内部基准，最高可信度）

| 模型 | CursorBench-3 准确率 |
|------|-------------------|
| Composer 1 | 38.0% |
| Composer 1.5 | 44.2% |
| **Composer 2** | **61.3%（↑37% 相对提升）** |
| GPT-5.4（参考） | 63.9% |

Composer 2 相比 Composer 1 **相对提升 37%**（(61.3-44.2)/44.2 ≈ 38.7%），这是系列模型最大的单版本跳跃。

#### 公开基准

| 基准 | Composer 2 成绩 |
|------|---------------|
| SWE-bench Multilingual | **73.7%** |
| Terminal-Bench (ICLR 2026) | **61.7%** |

- **SWE-bench Multilingual**：73.7% 达到公开榜单竞争力水平，覆盖多编程语言软件工程任务
- **Terminal-Bench**：61.7%，该基准测试命令行界面 Agentic 任务，涵盖文件操作、脚本执行、系统调试等

#### RL 训练过程中的观测

论文提到了几项重要的 RL 训练规律：
- **Average vs Best-of-K 双提升**：Composer 2 在 RL 训练中同时提升了平均性能（average）和最优采样性能（best-of-K），表明 RL 不仅在分布中心发力，也拓展了上界
- **自我摘要（Self-Summarization）效果**：引入 Self-Summarization 后，长时域任务完成率显著提升，模型在多轮 Tool Call 后维持任务一致性的能力增强

### 对比基线（Baseline）

- **Composer 1**（38.0% on CursorBench-3）：仅基础预训练 + 较小规模 RL，无专门的 Agentic 设计
- **Composer 1.5**（44.2% on CursorBench-3）：中间版本
- **GPT-5.4**（63.9% on CursorBench-3）：闭源商业模型，Composer 2 目标是在 Cursor 场景上接近或超越
- **SWE-agent、OpenHands**：社区开源 Agent 框架，Composer 2 在 CursorBench 维度上全面超越

### 推理效率

论文中还提到 CursorBench 的评估维度包含**延迟**和**成本**，表明 Composer 2 在追求准确率时也控制了推理开销。MTP 辅助的 Speculative Decoding 是其中的关键优化：
- 直接用预训练中训练的 MTP 头做草稿生成，无需独立 Draft 模型
- 在自回归解码中可并行预测多步，在 B300 硬件上实现加速（具体数字论文未披露）

### 局限性（Limitations）

1. **CursorBench 不公开**：内部基准无法被社区独立验证，可复现性受限
2. **代码不开源**：Composer 2 模型权重及训练代码均未开源（与 DeepSeek 等透明度对比存在差距）
3. **任务分布特定性**：CursorBench 来自 Cursor 工程团队的编码会话，可能偏向特定技术栈（如 TypeScript/Python），泛化性待验证
4. **成本约束**：RL 训练使用 NVIDIA B300 大规模集群，复现门槛极高
5. **评估维度不均**：公开基准仅报告 Pass@1，而 CursorBench 多维度评估（交互行为、代码质量）无法直接对比

---

## 💡 启发与我的想法

### 可借鉴之处

**1. "仿真真实场景"的训练理念**

Composer 2 最大的方法论贡献不是某个算法，而是**用内部真实任务驱动整个训练循环**的工程哲学：

> 训练/评估数据越接近真实部署场景，模型能力的转化率越高

这对于任何想训练 Agentic 模型的团队都是核心启示：公开基准往往是"考点"，而不是"实战场"，最好的 RL 奖励来自真实用户行为数据。

**2. GRPO 改进的细节**

去掉长度标准化 + 切换 KL 估计量（$k_3$ → $k_1$）这两个改动看似微小，但在长序列 Agentic 任务中效果显著。这提示我们：当任务响应长度差异极大时（从几百 token 到几万 token），标准 GRPO 的梯度估计存在系统性偏差，需要针对性校正。

**3. Multi-Token Prediction 的双重价值**

MTP 在这里既作为训练辅助目标（提升 Sample Efficiency），又直接支持推理加速（Speculative Decoding），是一个"训练-推理"协同设计的好例子——在预训练时就规划推理效率。

**4. 非线性长度惩罚的工程直觉**

线性惩罚容易让 RL 陷入局部最优（要么极短要么无约束地长），而凸函数惩罚 $C(x) = \frac{(1+kx)^{1-q}-1}{k(1-q)}$ 提供了更平滑的梯度信号，值得在 RL 工程中推广。

### 改进空间

1. **Self-Summarization 的质量控制**：当前论文未详述如何评估摘要质量（避免关键信息丢失），这是 Agentic 长时域推理的关键瓶颈

2. **CursorBench 的泛化性问题**：若能设计一个介于公开基准（易污染）和私有基准（不可复现）之间的"半公开"评估机制，将更有科研价值

3. **MoE 路由稳定性**：MoE Router Replay 解决了训练时的数值稳定性，但路由策略随 RL 训练的演化是否合理尚未分析

4. **多智能体协作**：当前 Composer 2 是单一 Agent 模型，MetaGPT/ChatDev 等多智能体框架的思路（代码审查、规划-执行分离）或可进一步提升复杂工程任务的成功率

### 下一步 Action
- [ ] 关注 Cursor 是否开放 CursorBench 或发布评估协议，以便独立验证
- [ ] 研究 GRPO 长度标准化对梯度偏差的理论分析（对比 Schulman 2020 KL 估计量分析）
- [ ] 尝试将 Non-linear Length Penalty 的设计思路应用到当前的 RL 代码训练实验
- [ ] 阅读 Kimi K2.5 论文，了解 Composer 2 基础模型的 MoE 架构设计


