---
title: "M2RL：多域RLVR"
tags:
  - 强化学习
  - 多任务学习
  - RLVR
  - 模型合并
  - LLM后训练
---

## 🧠 核心摘要 (TL;DR)
> M2RL 系统性地对比了"多任务混合 RLVR 训练"与"单域 RLVR 后模型合并"这两种多域强化学习范式，发现 reasoning 密集型域（数学/代码/科学）之间存在协同增益、干扰极小，混合训练以仅 63.7% 的 GPU 时间代价达到与合并方案相当的综合性能。

## 🏷️ 元数据
- **论文标题**: To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models
- **发表机构/会议**: Samsung Research Beijing / arXiv
- **年份**: 2026
- **链接**: [[M2RL_paper|📄论文]] | 💻 Code: [GitHub M2RL](https://github.com/Mosi-AI/M2RL)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #rlvr #multi-task-rl #model-merging #llm-post-training #grpo
- **前置知识 (Prerequisites)**: [[RL中的KV稀疏讨论]]
- **相关节点 (Related Notes)**: [[KV稀疏调研]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈

LLM 的后训练（Post-Training）正在从单一域走向多域。当我们希望一个模型同时具备数学推理、代码生成、科学问答、指令遵循、Agent 调用等能力时，核心问题在于：**应该把所有数据混在一起同时训练，还是分别训练后再合并？**

现有单域 RLVR（Reinforcement Learning with Verifiable Rewards）已取得显著进展（如 DeepSeek-R1、Qwen3 等），但直接应用于多域时面临以下挑战：

1. **任务干扰未知**：不同 domain 的 gradient 可能互相抵消，导致某些任务性能下降（灾难性遗忘或负迁移）
2. **计算效率与能力保留的取舍**：分别训练后合并（Model Merging）可以保留各域的专门化，但需要 5× 的 GPU 资源；混合训练节省资源但担心互相干扰
3. **多域奖励设计复杂**：不同 domain 的可验证奖励信号形式不同（数学公式验证、代码执行、指令约束检查等），如何在统一框架下路由

### 论文的切入点

M2RL（Mix or Merge for RL）在统一实验框架下，用 Qwen3-4B-Base 作为基模型，在数学、代码、科学、指令遵循（IF）、Agent 五个域上系统对比：
- **RL-Multi**：所有域数据混合，统一 GRPO 训练 1000 步
- **Merging**：每域独立 RLVR 后，用权重合并（Task Arithmetic、TIES、DARE 等）或行为合并（多教师蒸馏）整合

同时从**权重空间几何、梯度干扰、信息约束、预测行为、自我验证**五个角度分析机制。

---

### 📚 相关工作

#### 先驱/奠基工作

**RLVR 与推理能力激发**：
- **DeepSeek-R1** [Guo et al., 2025, arXiv:2501.12948]：首次大规模证明纯 RLVR（无 SFT 冷启动）可显著激发 LLM 的链式推理能力，采用 GRPO 算法；M2RL 的核心训练范式直接继承自该工作
- **DeepSeekMath / GRPO** [Shao et al., 2024, arXiv:2402.03300]：提出 Group Relative Policy Optimization，相比 PPO 无需价值网络，降低显存开销；M2RL 采用 GRPO 作为训练算法
- **OpenAI o1** [Jaech et al., 2024, arXiv:2412.16720]：通过 RL 实现 test-time scaling 的代表性工作，奠定了"RL 激发推理"这一研究方向的商业价值

**多任务强化学习基础**：
- **Gradient Surgery** [Yu et al., 2020, NeurIPS]：提出通过投影梯度来缓解多任务 RL 中的任务间干扰，是 M2RL 分析梯度冲突的重要参照；M2RL 发现 RLVR 场景下梯度干扰远小于传统 MTL 预期
- **DAPO** [Yu et al., 2025, arXiv:2503.14476]：大规模 RLVR 系统，引入 clip-higher 策略防止 token 层级的 entropy 崩溃，提高多样性

#### 直接对比方法（模型合并）

- **Task Arithmetic** [Ilharco et al., 2022, arXiv:2212.04089]：将 fine-tuned 模型与 base 模型之差定义为"task vector"，通过线性叠加实现多任务合并；M2RL 的 Merging 基线之一
- **TIES-Merging** [Yadav et al., 2023, NeurIPS]：通过修剪冗余参数（Trim）、统一符号（Elect）、叠加（Sum）三步解决参数干扰问题，优于 Task Arithmetic；M2RL 实验中的主要合并方案
- **TIES+DARE**：在 TIES 基础上加入随机稀疏化（DARE ratio=0.8），进一步减少合并噪声
- **Model Soups** [Wortsman et al., 2022, ICML]：对多个 fine-tuned checkpoint 进行权重平均，提升泛化性而不增加推理成本；M2RL Average Merging 的参考基础
- **Fisher-weighted Averaging** [Matena & Raffel, 2022, NeurIPS]：用 Fisher 信息矩阵加权合并参数，对重要参数保留更多；M2RL 对比的合并方案之一
- **DARE（Super Mario）** [Yu et al., 2024, ICML]：证明 fine-tuned 模型的大量参数变化可随机置零（DARE 掩码），而不影响性能；有助于减少合并时的干扰
- **Systematic Study of Model Merging** [Hitit et al., 2025, arXiv:2511.21437]：对现有 LLM 合并技术进行系统评估，为 M2RL 的合并基线选择提供参考

#### 多任务 RLVR 相关工作

- **Imbalanced Gradients in RL Post-Training** [Wu et al., 2025, arXiv:2510.19178]：专门研究多任务 LLM RL 后训练中的梯度不平衡问题，与 M2RL 的梯度干扰分析高度相关
- **Does RL Incentivize Reasoning?** [Yue et al., 2025, arXiv:2504.13837]：质疑 RLVR 是否真正在基础能力之外激发了新的推理能力，对 M2RL 的协同效应分析构成理论对话
- **RLVR Incentivizes Correct Reasoning** [Wen et al., 2025, arXiv:2506.14245]：证明 RLVR 隐式激励了正确推理链，为 M2RL 的跨域协同提供机制解释
- **Process Reinforcement (PRIME)** [Cui et al., 2025, arXiv:2502.01456]：通过隐式过程奖励改进 RL 训练，M2RL 的过程验证分析（process-level rigor vs. outcome accuracy）与此密切相关

#### 强基线与关联系统

- **Tulu 3** [Lambert et al., 2024, arXiv:2411.15124]：代表性的多任务 SFT+DPO 后训练 pipeline，展示了混合数据训练在 SFT 阶段的有效性；M2RL 将该结论延伸到 RLVR 阶段
- **Skywork Open Reasoner 1** [He et al., 2025, arXiv:2505.22312]：基于 Qwen 系列的开源推理模型，与 M2RL 使用相同的基础模型，可作为性能对标
- **FuseChat** [Wan et al., 2025, EMNLP 2025]：通过知识融合合并多个 chat 模型，代表 Behavior-based 合并的方向；M2RL 的蒸馏合并（MT-OPD）与其理念相通
- **MiMo-v2-flash** [Xiao et al., 2026, arXiv:2601.02780]：MoE 混合架构 reasoning 模型，代表 2026 年多域 LLM 系统的典型设计
- **GLM-4.5 ARC** [Zeng et al., 2025, arXiv:2508.06471]：Agentic+Reasoning+Coding 三域统一模型，与 M2RL 研究的"是否可以同时提升多个 reasoning 域"问题直接对应
- **On-policy Distillation** [Agarwal et al., 2024, ICLR]：多教师在线蒸馏方案，M2RL 的 MT-OPD（Multi-Teacher On-Policy Distillation）行为合并方案直接参考此工作


## ⚙️ 核心设计与机制

### 整体框架：两种多域 RLVR 范式

```
┌─────────────────────────────────────────────────────────┐
│                      Qwen3-4B-Base                       │
└─────────────────┬──────────────────┬────────────────────┘
                  │                  │
        ┌─────────▼──────┐  ┌────────▼─────────────────┐
        │  混合训练范式   │  │      分别训练+合并范式     │
        │  (RL-Multi)     │  │      (Merging)            │
        │                 │  │                           │
        │ 所有5域数据混合  │  │ RL-Math, RL-Code,         │
        │ GRPO统一训练    │  │ RL-Science, RL-IF,        │
        │ 1000步          │  │ RL-Agent（各400步）       │
        │ domain-routed   │  │                           │
        │ rewards         │  │ ↓ Weight Merging / MT-OPD │
        └─────────────────┘  └───────────────────────────┘
```

### 关键技术点 1：GRPO 训练设置

**算法基础**：Group Relative Policy Optimization（GRPO），无需价值网络，通过组内相对奖励计算优势估计。

**统一训练参数**：
- 基础模型：Qwen3-4B-Base
- 批大小：128；组大小：16
- 最大生成长度：32k tokens
- 采样温度：1.0
- 学习率：$2 \times 10^{-6}$（单域恒定；多任务线性衰减）

**奖励设计（domain-routed rewards）**：

| Domain | 奖励类型 | 验证器 |
|--------|---------|--------|
| 数学 | 二值结果奖励（0/1）+ 格式奖励 | Qwen QwQ 裁判 |
| 代码 | 执行正确性奖励 | SandboxFusion 沙箱 |
| 科学 | 多项选择准确率 | 规则匹配 |
| 指令遵循（IF） | 约束满足度 | IFEval 规则 |
| Agent | 函数调用成功率 | BFCL v3 评测 |

**格式奖励**：通过"Chain-of-Thought 与最终答案分离"的格式要求，鼓励模型在 `<think>` 标签内进行推理，在 `<answer>` 标签内给出结论。

### 关键技术点 2：权重合并方案

**方案 A：Task Arithmetic（TA）**

$$\theta_{merged} = \theta_{base} + \lambda \sum_{i=1}^{K} (\theta_i - \theta_{base})$$

其中 $\theta_i$ 是第 $i$ 个域的 fine-tuned 模型，$\lambda$ 为缩放系数，$\theta_{base}$ 为基础模型。

**方案 B：TIES-Merging**

三步操作解决参数干扰：
1. **Trim**：按幅度阈值修剪 $\tau\%$ 最小的 delta 参数
2. **Elect**：对符号冲突的参数，取多数符号（符号投票）
3. **Sum**：对选中参数按 $(θ_i - θ_{base})$ 加和后加回 base

**方案 C：TIES+DARE**

在 TIES 基础上，对 delta 参数加入 DARE 随机稀疏化（保留率 0.8），进一步降低合并噪声。

**方案 D：Multi-Teacher On-Policy Distillation（MT-OPD）**

行为层面的合并：将 K 个单域专家模型共同作为教师，用在线蒸馏的方式训练一个学生模型。训练 200 步，批大小 256，模拟各教师的输出分布。

### 关键技术点 3：域间干扰与协同分析

**梯度干扰测量**：

$$\text{Interference}(i,j) = \frac{-\nabla_i \cdot \nabla_j}{|\nabla_i||\nabla_j|}$$

其中负余弦相似度越大，表示任务 $i$ 和 $j$ 的梯度方向越冲突。M2RL 发现五个 domain 之间的梯度冲突值 **均接近 0**，与传统多任务学习中的干扰问题形成对比。

**权重更新 Jaccard 重叠**：

对两个单域模型更新的 top-k 参数集合，计算 Jaccard 系数：
$$J(S_i, S_j) = \frac{|S_i \cap S_j|}{|S_i \cup S_j|}$$

结果：数学-代码-科学三域之间的参数更新 Jaccard 重叠率在 **45-48%**，说明这三个域共享大量底层优化路径（"policy neighborhoods" 相邻），是协同效应的几何解释。

**Policy Neighborhood 假说**：
同一 base model 经不同域 RLVR 后，若两个域的 policy distribution 在参数空间中"距离近"（KL 散度小），则它们更容易互相迁移；数学/代码/科学三域符合此条件，而 Agent 域（依赖函数调用格式）与其他域距离较远，因此协同效益不对称。

### 关键技术点 4：过程验证能力的退化

**重要发现**：长时间多任务 RLVR 训练会导致过程验证（process-level verification）能力严重退化。

具体现象：在 IFEval 上，混合训练模型的 **outcome 判断准确率为 80.5%**，但 **process 验证准确率仅 27.5%**——即模型能给出正确答案，但无法在推理链中正确追踪约束是否被满足。

这是因为：outcome-based reward 驱动模型优化"最终结果"，而对"中间约束的满足"没有直接正向信号；多任务训练叠加了更多 outcome 信号，进一步压缩了 process-level 的学习空间。模型合并（Merging）通过保留单域 SFT/RL 能力在一定程度上缓解了这一问题。


## 📊 实验与性能评价

### 对比基线 (Baseline)

| 方案 | 描述 |
|------|------|
| SFT | 监督微调基线（~14.1M 样本） |
| RL-{Math/Coding/Science/IF/Agent} | 各域单独 RLVR（400 步） |
| Average Merging | 5 个单域模型直接平均权重 |
| Task Arithmetic (TA) | task vector 线性叠加 |
| TIES-Merging | 带符号投票的参数合并 |
| TIES+DARE | TIES + 随机稀疏化 |
| MT-OPD | 多教师在线蒸馏（行为合并） |
| **RL-Multi** | 混合多任务 RLVR（1000 步） |

### 主要实验结果

**数学（Math）**

| Benchmark | RL-Math | RL-Multi | Best Merging |
|-----------|---------|----------|--------------|
| AIME'24 | 77.66% | 81.20% | 81.15% (TIES+DARE) |
| AIME'25 | 70.42% | 73.39% | 74.74% (TIES+DARE) |

→ RL-Multi 超越了单域 RL-Math（+3.5% AIME'24），说明代码/科学域数据对数学有协同增益。

**代码（Coding）**

| Benchmark | RL-Coding | RL-Multi | Best Merging |
|-----------|-----------|----------|--------------|
| LCB v5 | 65.00% | 63.21% | 60.84% |
| LCB v6 | 58.86% | 56.57% | 57.71% |

→ 代码是唯一 RL-Multi 略低于单域的 domain（-1.8%），合并方案也未显著超越单域，说明代码对其他域受益最少（给出的多，得到的少）。

**科学（Science）**

| Benchmark | RL-Science | RL-Multi | Best Merging |
|-----------|-----------|----------|--------------|
| HLE | 6.26% | 7.18% | 7.92% (TIES) |
| GPQA-Diamond | 53.79% | 53.62% | 57.58% (TIES) |

→ 合并方案在科学域表现最突出（GPQA +3.8%），说明多个推理域的知识叠加对科学有更大帮助；混合训练相对保守。

**指令遵循（Instruction Following）**

| Benchmark | RL-IF | RL-Multi | Best Merging |
|-----------|-------|----------|--------------|
| IFEval | 90.94% | 93.53% | 92.61% |
| IFBench | 59.52% | 61.22% | 54.76% |

→ RL-Multi 在 IFEval 上超过了单域 RL-IF（+2.6%），说明推理域数据对 IF 有正向迁移。但 IFBench（更难）的合并方案（54.76%）显著低于 RL-Multi（61.22%），说明 merging 在复杂指令遵循任务上有性能损失。

**Agent（函数调用）**

| Benchmark | RL-Agent | RL-Multi | Best Merging |
|-----------|----------|----------|--------------|
| BFCL v3 | 59.14% | 60.74% | 61.73% (TIES) |

→ Agent 域在混合训练和合并后均有小幅提升（受益于推理域），但向其他域的反向贡献极小（不对称）。

### 计算效率对比

| 方案 | 相对 GPU 小时（标准化为 1× = 单域 5× 训练） |
|------|---------------------------------------------|
| 5× 单域 RLVR | **100%** |
| TIES/TA Merging | ~100%（后处理近乎免费） |
| MT-OPD 蒸馏合并 | ~115%（需额外蒸馏训练 200 步） |
| **RL-Multi** | **63.7%** |

→ RL-Multi 以节省 36.3% 计算代价，达到与合并方案相当的综合性能，是"工程效率优先"场景的推荐选择。

### 局限性 (Limitations)

1. **过程验证退化**：长时间多任务 RL 会造成 process-level 推理能力下降（IFEval process 验证：27.5% vs 80.5% outcome），这一问题在应用于需要可解释推理的场景时需要额外处理
2. **代码域受益最少**：代码 domain 在混合训练中略有下降，可能与其奖励信号（执行结果 0/1）的稀疏性和任务难度分布差异有关
3. **实验规模限制**：实验基于 Qwen3-4B，4B 参数量下的协同规律是否在 7B/70B 量级同样成立尚未验证
4. **域覆盖范围**：五个 domain 主要聚焦于 reasoning 类任务；创意写作、多语言、长文档等域的多任务 RLVR 行为未涉及
5. **权重合并的理论基础**：现有合并方案（TIES、DARE）基于经验设计，缺乏对 RLVR 训练后参数分布的理论刻画

---

## 💡 启发与我的想法

### 可借鉴之处

1. **推理域协同是实证基础**：M2RL 通过梯度分析和权重几何严格证明了 math/code/science 在 RLVR 阶段的协同效应。这为"一次训练，多域受益"提供了坚实的实验依据，可以直接指导我们在新项目中选择是否进行多域混合 RLVR。

2. **Policy Neighborhood 视角很有价值**：用 Jaccard 参数重叠率来解释协同性，是一个简洁有效的分析工具。可以推广到其他分析场景——如分析两个 fine-tuning 任务是否"兼容"时，预先检查参数更新的 Jaccard 重叠率或许可以预测合并效果。

3. **过程验证与结果验证的分离**：M2RL 揭示了 outcome reward 训练后 process verification 能力的退化现象（27.5% vs 80.5%）。这在工程上有重要意义：如果下游应用需要模型输出可解释的推理过程（而不仅是最终答案），仅靠 outcome reward 的 RLVR 是不够的，需要引入 Process Reward Model（PRM）或 PRIME 类方法。

4. **63.7% GPU 时间的工程价值**：在资源受限的训练场景（如消费级 GPU 集群），混合训练相比分别训练+合并节省的 36.3% 计算是显著的优势，尤其当合并后性能差距可接受时。

### 改进空间

1. **Domain 动态权重调度**：当前 RL-Multi 使用固定的 domain-routed rewards（各域采样权重固定）。可以借鉴课程学习思想，根据各域的"学习进度"（如 reward 增长曲线斜率）动态调整采样权重，让落后的域获得更多训练信号，避免代码域受益偏少的问题。

2. **过程奖励 + 结果奖励的混合**：针对 process-level 退化问题，可以在 IF/Agent 域引入过程奖励（如 PRIME 的隐式过程奖励），在 multi-task 训练中对这两类域单独使用 process-aware reward，而数学/代码/科学继续使用 outcome reward，构成"异构奖励"的多域训练。

3. **规模律验证**：在 7B 乃至 70B 模型上验证协同规律是否成立，以及合并方案的参数数量对 scaling 的敏感性，是直接的 follow-up 工作。

4. **动态合并权重学习**：TIES/DARE 的合并系数（$\lambda$、DARE ratio）目前通过网格搜索确定；可以引入元学习或贝叶斯优化自动寻找最优合并系数，减少人工调优成本。

### 下一步 Action
- [ ] 阅读 DeepSeek-R1 原文（arXiv:2501.12948），理解 GRPO 算法的完整推导
- [ ] 研究 PRIME（arXiv:2502.01456），了解 process reward 如何与 RLVR 结合
- [ ] 关注 M2RL GitHub 仓库（https://github.com/Mosi-AI/M2RL）的代码更新，复现合并实验
- [ ] 思考：KV Cache 稀疏化与多任务 RLVR 能否结合？在 RL 训练阶段引入 KV 稀疏能否减少显存占用而不损害多域协同效益


