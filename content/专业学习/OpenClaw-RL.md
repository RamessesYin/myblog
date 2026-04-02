---
title: "OpenClaw-RL"
tags:
  - 强化学习
  - LLM-Agent
  - 优势估计
  - 在线学习
  - 过程奖励
---

## 🧠 核心摘要 (TL;DR)

> 现有 GRPO/PPO 等方法都在**离线批量模式**下工作，Agent 每次交互产生的"下一状态信号"（用户回复、执行结果）被白白丢弃。OpenClaw-RL 提出从实时交互信号中恢复两类被浪费的学习信号：通过 PRM 法官转化为标量奖励的**评估信号**（Binary RL），以及通过 Hindsight hint 构造 token 级方向性优势的**指导信号**（OPD）——无需 Critic 网络，无需 group 采样，适用于无法构造 group structure 的真实对话场景。==可以在agent场景持续在线学习==

## 🏷️ 元数据
- **论文标题**: OpenClaw-RL: Train Any Agent Simply by Talking
- **发表机构/会议**: arXiv (cs.CL, cs.AI, cs.LG)
- **年份**: 2026（提交于 2026-03-10）
- **链接**: [[OpenClaw-RL_paper|📄论文]] | [💻 Code](https://github.com/Gen-Verse/OpenClaw-RL)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #强化学习 #OpenClaw #RL #LLM-Agent #优势估计 #在线学习 #PRM #过程奖励 #知识蒸馏 #优势函数
- **前置知识 (Prerequisites)**: PPO/GRPO 基础，LLM RLVR 范式
- **相关节点 (Related Notes)**: [[RL中的KV稀疏讨论]]
- **向下延申 (Successors)**: DAPO (arXiv:2503.14476), Dr.GRPO (arXiv:2503.20783), RLAnything

---

## 🎯 动机与痛点

### 核心洞察：两种被浪费的信号

每个部署中的 AI Agent 都在持续交互，但产生的"下一状态"信号被完全丢弃。作者识别出两类浪费：

**Waste 1 — 评估信号（Evaluative Signal）**：下一状态 $s_{t+1}$ 隐式对前序动作 $a_t$ 打分。
- 用户重新提问 → 不满意
- 测试通过 → 成功
- 错误日志 → 失败
这些构成天然的**过程奖励**，无需额外标注。

**Waste 2 — 指导信号（Directive Signal）**：下一状态还携带方向性信息。
- 用户说"你应该先查看文件再编辑它"——不仅说明"错了"，还在 token 级别指出"应该怎么做"。

### 现有 RL 范式的局限

现有方法（GRPO、PPO、DAPO）的共同问题：**在批量离线模式下运行**，数据收集和训练分离，下一状态信号没有被在线利用。更关键的是：

| 方法 | 核心限制 |
|------|---------|
| **PPO** | 需要 Critic 网络，参数量翻倍，训练复杂 |
| **GRPO** | 需要对同一 prompt 采样 $G$ 个回答做组内归一化，对话场景中**无法构造 group structure** |
| **DAPO** | 仍依赖 group 采样，仅改进了 clip 和熵正则 |
| **REINFORCE** | 高方差，序列级粗粒度信号 |

---

## ⚙️ 核心设计与机制

### 问题形式化

将 Agent 交互建模为 MDP $(S, A, T, r)$：
- 状态 $s_t$：截至第 $t$ 轮的完整对话/环境上下文
- 动作 $a_t$：策略 $\pi_\theta$ 生成的 token 序列
- 转移 $\mathcal{T}(s_{t+1} | s_t, a_t)$：确定性（用户回复/执行结果）
- 奖励 $r(a_t, s_{t+1})$：由 PRM 法官从下一状态信号推断

---

### 方法一：Binary RL（评估信号 → 标量 Advantage）

#### PRM 多数投票

$$\text{PRM}(a_t, s_{t+1}) \rightarrow r \in \{+1, -1, 0\}$$

$$r_{\text{final}} = \text{MajorityVote}(r_1, \ldots, r_m)$$

运行 $m$ 次独立查询（实验中 $m=1$ 或 $3$）取多数票，降低单次判断噪声。

#### 优势定义

$$A_t = r_{\text{final}} \in \{+1, -1, 0\}$$

这是**序列级标量优势**：对该轮回答中所有 token 施加相同的推动力。

#### 非对称 PPO Clip 损失

概率比：$\rho_t = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$

$$\mathcal{L}_{\text{pg}} = -\mathbb{E}_t\left[\min\!\left(\rho_t A_t,\ \text{clip}(\rho_t,\ 1-\varepsilon,\ 1+\varepsilon_{\text{high}}) \cdot A_t\right)\right]$$

$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

超参数：$\varepsilon = 0.2$，$\varepsilon_{\text{high}} = 0.28$，$\beta_{\text{KL}} = 0.02$

**为什么上下界不对称？**
标准 PPO 对称 clip（$\varepsilon_{\text{high}} = \varepsilon = 0.2$）限制了正样本的更新空间。本文设 $\varepsilon_{\text{high}} = 0.28 > \varepsilon$，允许正确行为（$A_t > 0$）有更大的策略更新步长，而对负样本保持保守——"奖励正确比惩罚错误更激进"，在对话场景中可防止过度惩罚导致的 collapse。

---

### 方法二：Hindsight-Guided OPD（指导信号 → Token 级 Advantage）

#### 核心洞察

Binary RL 把 $s_{t+1}$ 压缩为一个标量，丢弃了文本中的方向性信息。OPD 的关键思想：

> 把用户的 hint 塞回 prompt，**同一个模型**在"知道 hint"的情况下会生成不同的 token 分布。这个分布与原始分布的差值，精确地告诉每个 token "应该往哪个方向走"。

#### 完整推导（四步）

**Step 1 — Hindsight Hint 提取**

$$\text{Judge}(a_t, s_{t+1}) \rightarrow \{\text{score} \in \{+1,-1\},\ \text{hint} \in \mathcal{T}^*\}$$

Judge 模型不直接使用 $s_{t+1}$（原始状态噪声大、冗长），而是将其**提炼**为简洁可操作的 1-3 句纠错指令。

**Step 2 — 质量过滤**

$$\text{valid} = \{h_i : \text{score}_i = +1 \land |h_i| > 10\}$$

只保留正向且足够具体（>10 字符）的 hint，取最长的（信息最丰富）。若 valid = ∅，**丢弃该样本**。

> OPD 以样本数量换信号质量：只有下一状态携带清晰指令的轮次才进入训练。

**Step 3 — 增强教师语境构造**

$$s_{\text{enhanced}} = s_t \oplus \texttt{"[user's hint]\textbackslash n\{hint\}"}$$

将 hint 追加到最后一条用户消息后，形成"教师视角"的上下文。

**Step 4 — Token 级优势计算**

在 $s_{\text{enhanced}}$ 下对原始回答 $a_t$ 做 teacher forward pass，逐 token 计算对数概率差：

$$\boxed{A_t^{(k)} = \log \pi_{\text{teacher}}\!\left(a_t^{(k)} \mid s_{\text{enhanced}}\right) - \log \pi_\theta\!\left(a_t^{(k)} \mid s_t\right)}$$

**直觉解读：**
- $A^{(k)} > 0$：teacher（知道 hint）对该 token 赋更高概率 → 该 token 正确，应强化
- $A^{(k)} < 0$：teacher 认为该 token 不合适 → 该 token 导致错误，应抑制
- 同一回答中不同 token 得到**不同方向和强度**的梯度，而非 Binary RL 的统一标量

**与 Binary RL 的根本区别：**

```
Binary RL:  [+1] [+1] [+1] [+1] [+1]   ← 全序列同一个值
OPD:       [+0.3] [-0.8] [+1.2] [+0.1] [-0.5]  ← 每 token 独立
```

---

### 两种方法的组合

同时运行 Binary RL 和 OPD，将优势线性组合：

$$A_t = w_{\text{binary}} \cdot r_{\text{final}} + w_{\text{opd}} \cdot \left(\log \pi_{\text{teacher}}(a_t \mid s_{\text{enhanced}}) - \log \pi_\theta(a_t \mid s_t)\right)$$

默认 $w_{\text{binary}} = w_{\text{opd}} = 1$

**组合逻辑：**
- Binary RL：广覆盖，所有有评分的轮次都能提供梯度，适合早期训练
- OPD：高精度，只有携带 hint 的轮次触发，适合后期精细纠正
- 组合：Binary RL 提供早期广泛梯度，OPD 提供精确 token 级方向

---

### 通用 Agent 的步骤级奖励整合

对长时程任务（Terminal/GUI/SWE），PRM 在每一步提供过程奖励，与结果奖励 $o$ 叠加：

$$\text{Total Reward} = o + \sum_{i=1}^{m} \frac{r_i}{m}$$

**关键工程问题**：通用场景没有 group structure，无法用 GRPO 的组归一化。解决方法：按**步骤索引**（step index）分组动作，实验验证有效。

---

### 异步基础设施

四路完全解耦的异步循环，互不等待：

```
策略服务(SGLang) → 环境执行(HTTP/API) → PRM打分(SGLang/API) → 梯度更新(Megatron)
```

> "模型在为下一个用户请求服务时，PRM 在判断上一个回复，训练器在应用梯度更新。"

---

## 📊 实验与性能评价

### 个人 Agent 实验（Table 3，基准分 0.17）

| 方法 | 8 步更新 | 16 步更新 |
|------|---------|----------|
| Binary RL | 0.25 | 0.23 |
| OPD | 0.25 | **0.72** |
| **Combined** | **0.76** | **0.81** |

**关键规律：**
- Binary RL 早期起效（8 步 0.17→0.25），但后期下降（标量信号粗糙，梯度方向不精准，容易震荡）
- OPD 早期慢（样本因质量过滤而稀疏），后期大幅提升（token 级信号精准，收敛稳健）
- 组合方案在 8 步就达到 0.76，是最优方案

### 通用 Agent 实验（Table 4）

| 场景 | 整合奖励（过程+结果） | 仅结果奖励 |
|------|---------------------|----------|
| Tool-call（250步） | **0.30** | 0.17 |
| GUI（120步） | **0.33** | 0.31 |

过程奖励在 Tool-call 场景提升显著（+76%），GUI 场景提升有限（任务步骤短，结果奖励已足够密集）。

### 快速学习效率

- **学生设置**：仅 **36 次**问答交互后，模型学会避免 AI 化措辞，生成自然风格
- **教师设置**：仅 **24 次**批改交互后，反馈质量明显改善

### 局限性

1. **OPD 样本稀疏**：需要 hint 质量足够（>10字符、明确指令），大量轮次被过滤，数据利用率低
2. **Judge 模型依赖**：hint 提取质量高度依赖 Judge 的理解能力，弱 Judge 会引入噪声
3. **无 group 归一化**：Binary RL 缺乏 GRPO 式的组内基线，优势估计方差相对更高
4. **超参数敏感**：$w_{\text{binary}}$ 和 $w_{\text{opd}}$ 的平衡对不同任务可能需要调整

---

## 💡 启发与我的想法

### RL 范式对比的核心维度

读完这篇论文，最有价值的是它隐含地揭示了 RL 训练范式选择的核心权衡：

```
信号粒度  vs  信号覆盖率
  ↑高精度         ↑高覆盖
  OPD            Binary RL / GRPO
  token级        序列级
  稀疏样本        密集样本
```

GRPO 在数学推理任务上成功的前提是：可以采样 $G=8$ 个独立答案做组内归一化，天然地消除了基线方差。但在真实 Agent 对话中，这个前提完全不成立——用户不会对同一句话给出 8 个不同的回应。本文找到了一条在**单流对话**中估计优势的新路径。

### Token 级 Advantage 的设计空间

OPD 的 token 级优势本质上是一种**隐式的 KL 散度利用**：

$$A_t^{(k)} = \log \frac{\pi_{\text{teacher}}(a_t^{(k)} \mid s_{\text{enhanced}})}{\pi_\theta(a_t^{(k)} \mid s_t)}$$

这在形式上和 KL 散度的逐 token 分解非常接近。与 DPO 的比较也很有意思：DPO 用偏好对比隐式计算优势，本文用上下文增强隐式构造教师分布，两者都避免了显式 Critic。

### 与本项目其他方向的关联

- [[RL中的KV稀疏讨论]]：在 RL rollout 阶段，OPD 的 teacher forward pass 需要对同一 prompt 跑两次（原始 + enhanced），KV Cache 复用率存在优化空间——enhanced 前缀与原始 prompt 共享前缀部分，可以用 prefix caching 优化
- [[Continuum]]：OpenClaw-RL 的异步 Agent 训练架构本质上也是多轮 Agent 调度问题，TTL 式的 KV Cache 保留机制可以天然应用于 OPD 的 teacher forward pass 场景

### 下一步 Action
- [ ] 阅读 DAPO（arXiv:2503.14476），理解其对 GRPO 四个问题的系统分析，与本文的非对称 clip 设计对比
- [ ] 阅读 Dr.GRPO（arXiv:2503.20783），其对 token-level 和 sample-level bias 的修正与 OPD 的 token 级优势思路有交叉
- [ ] 思考 OPD 的 hint 提取是否可以用更轻量的方式实现（如直接用模型内部的 attention pattern 判断哪些 token 是关键错误）
- [ ] 验证在 RL rollout 场景下，teacher forward pass 的 KV Cache 与 student rollout 的 KV Cache 共享前缀的比例，估算 prefix caching 的加速潜力
