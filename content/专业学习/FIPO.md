---
title: "FIPO：基于未来KL散度"
tags:
  - 强化学习
  - LLM推理
  - 信用分配
  - policy-optimization
  - RLVR
---

## 🧠 核心摘要 (TL;DR)
> FIPO 针对 GRPO 类方法中"全序列统一分配优势值"的粗粒度信用分配缺陷，提出将折扣未来KL散度（Future-KL）加入策略更新，从而构造密集优势表示（Dense Advantage）。该方法无需独立评估网络，在 Qwen2.5-32B 上将 AIME 2024 准确率从 DAPO 的 50.0% 提升至 56.0%（峰值 58.0%），并将平均推理链长度从约 4,000 token 扩展至超过 10,000 token。

## 🏷️ 元数据
- **论文标题**: FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization
- **发表机构/会议**: arXiv（Alibaba Group / Qwen Pilot Team，Dartmouth College，Peking University）
- **年份**: 2026
- **链接**: [[FIPO_paper|📄论文]] | 💻 Code: 暂未提供具体 GitHub URL（基于 verl 框架）

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #稠密信号奖励 #reinforcement-learning #llm-reasoning #credit-assignment #policy-optimization #rlvr #dense-advantage
- **前置知识 (Prerequisites)**: [[KV稀疏调研]]
- **相关节点 (Related Notes)**: [[RL中的KV稀疏讨论]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈

LLM 推理能力的强化学习路线（RLVR）大体可分为两类：
1. **基于结果的奖励（ORM）方法**：以 GRPO、DAPO 为代表，仅依赖最终答案的正确性给出奖励，奖励信号极为稀疏；
2. **基于价值函数的方法**：以 PPO+GAE、VAPO 为代表，通过训练独立的评估网络（Critic）估计每个 token 的价值，信号更密集，但引入了额外参数和训练不稳定性。

**GRPO 的核心问题——粗粒度信用分配**：在 GRPO（DeepSeekMath 提出）中，每次采样 $G$ 条轨迹，计算相对优势后，将**同一个优势值 $A_t$** 广播给轨迹中的每一个 token。这种做法等价于假设每个 token 对最终结果的贡献完全相同，显然与实际推理过程不符——一个关键的"思维转折"token 与后续大量重复验算的 token 不应被同等看待。

**性能天花板问题**：DAPO（GRPO 的改进版本，引入了 Decoupled Clip 和 Dynamic Sampling）在 Qwen2.5-32B 上将 AIME 2024 提到了 50.0%，但推理链长度稳定在约 4,000 token，难以继续延伸，==说明模型无法从更长的推理中获益，本质上是缺乏细粒度的 token 级别信用信号。==

**引入 Critic 的代价**：PPO 使用广义优势估计（GAE，Schulman et al., 2015）可以提供密集信号，但需要一个与策略模型等大的 Critic 网络，显著增加显存和训练复杂度，在 32B+ 规模下尤为昂贵。研究表明（Yuan et al., 2025），PPO 在长 CoT 任务上容易因价值函数初始化偏差和奖励信号衰减而崩溃。

**FIPO 的切入点**：能否在 **不引入额外 Critic 网络** 的前提下，利用策略自身的信息构造dense的 token 级别优势？论文的答案是"用 Future-KL"——即利用当前策略更新对后续 token 概率分布的影响来衡量每个 token 的信用。

---

### 相关工作综览

#### 奠基/先驱工作

| 论文 | 贡献 | 与 FIPO 关系 |
|------|------|-------------|
| **PPO** (Schulman et al., 2017, arXiv:1707.06347) | 提出代理目标函数 + clip 机制，是现代 RLHF 基础 | FIPO 的 clip 设计直接借鉴自 PPO |
| **GAE** (Schulman et al., 2015, arXiv:1506.02438) | 广义优势估计，指数加权 TD(λ) 平衡方差-偏差 | FIPO 的 Future-KL 在设计直觉上类似 GAE，但无需 Critic |
| **GRPO** (DeepSeekMath, Shao et al., 2024, arXiv:2402.03300) | 引入组相对策略优化，为 LLM 推理 RL 的主流基线 | FIPO 在 GRPO 框架上改造优势计算 |
| **DeepSeek-R1** (Guo et al., 2025) | 展示 RLVR 在推理模型中的大规模成功 | 激发本领域一系列工作包括 FIPO |

#### 直接对比方法

| 论文 | 方法特点 | 与 FIPO 的主要差异 |
|------|----------|-------------------|
| **DAPO** (Yu et al., 2025, arXiv:2503.14476) | Decoupled Clip + Dynamic Sampling，AIME 50% | 仍是均匀优势广播，FIPO 的主要对比基线 |
| **VAPO** (Yue et al., 2025, arXiv:2504.05118) | 基于值函数的推理 RL，AIME 60.4 | 需要 Critic 网络，FIPO 无 Critic 但接近其效果 |
| **BAPO** (Xi et al., 2025, arXiv:2510.18927) | 自适应 clip + 平衡正负样本，off-policy 稳定 | 解决熵崩溃问题，角度不同 |
| **Open-Reasoner-Zero** (Hu et al., 2025, arXiv:2503.24290) | 标准 PPO+GAE，无 KL 正则，Qwen2.5-32B | 有 Critic，作为 Critic 方法的参照 |
| **GSPO** (Zheng et al., 2025, arXiv:2507.18071) | 组序列策略优化 | 同在 GRPO 改进路线上 |

#### Follow-up 与周边工作

| 论文 | 贡献 |
|------|------|
| **Kimi k1.5** (Team et al., 2025, arXiv:2501.12599) | 长上下文 RL 训练，无 MCTS/Value Function |
| **Understanding R1-Zero-like Training** (Liu et al., 2025, arXiv:2503.20783) | 揭示 GRPO 长度偏置问题，提出 Dr. GRPO |
| **PPO Collapse** (Yuan et al., 2025, arXiv:2503.01491) | 分析 PPO 在长 CoT 下崩溃原因，提出 VC-PPO |
| **TTRL** (Zuo et al., 2025, arXiv:2504.16084) | 测试时强化学习 |
| **Spurious Rewards** (Shao et al., 2025, arXiv:2506.10947) | 反思 RLVR 中奖励信号的可靠性 |
| **Entropy Minimization** (Agarwal et al., 2025, arXiv:2505.15134) | 熵最小化在 LLM 推理中的意外有效性 |
| **RLVR Direction** (Huang et al., 2025, ICLR 2025) | 分析 RLVR 更新方向的本质 |
| **Sparse but Critical** (Meng et al., 2025, ICLR 2025) | Token 级分布漂移的稀疏性分析，与 FIPO 动机高度相关 |
| **HybridFlow/verl** (Sheng et al., 2025, EuroSys 2025) | RLHF 训练框架，FIPO 基于此实现 |
| **Qwen2.5-Math** (Yang et al., 2024, arXiv:2409.12122) | 评估所用基础模型 |



## ⚙️ 核心设计与机制

FIPO 的核心思想：**用策略在一次更新中对整条后续轨迹的概率分布影响来估计当前 token 的信用**，无需独立 Critic 网络。

### 背景：GRPO 基准优势计算

在 GRPO 框架中，对每个问题 $q$ 采样 $G$ 条输出轨迹 $\{o_1, \ldots, o_G\}$，计算每条轨迹的奖励 $\{r_1, \ldots, r_G\}$ 后，利用**组内相对归一化**得到基线优势：

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

随后将 $\hat{A}_i$ 这一**标量**直接广播给轨迹 $i$ 中每个位置 $t$ 的 token：$A_t = \hat{A}_i$。这是 FIPO 要克服的粗粒度问题。

### Step 1：概率漂移（Probability Shift）

对于轨迹中位置 $t$ 处的 token $o_t$，定义**当前更新步相对于旧策略的对数概率差**：

$$\Delta\log p_t = \log \pi_\theta(o_t \mid q, o_{<t}) - \log \pi_{\theta_\text{old}}(o_t \mid q, o_{<t})$$

这一量度量了本次梯度步对 token $o_t$ 的生成概率做了多大程度的调整（正值 = 增大，负值 = 减小）。

### Step 2：Future-KL 计算

将 $\Delta\log p_t$ 沿时间轴向未来累加，得到 **Future-KL**——当前位置 $t$ 的概率改变对后续轨迹分布的累积影响：

$$\text{FutureKL}_t = \sum_{k=t}^{T} M_k \cdot \gamma^{k-t} \cdot \Delta\log p_k$$

其中包含两个关键设计：

**掩码 $M_k$（Masking）**：过滤掉重要性比率极端的 token，防止训练不稳定：

$$M_k = \mathbb{1}\!\left[\frac{\pi_\theta(o_k)}{\pi_{\theta_\text{old}}(o_k)} \leq c\right]$$

**指数衰减 $\gamma^{k-t}$（Soft Decay Window）**：对距离较远的 token 权重衰减，使近邻 token 的影响更大：

$$\gamma = 2^{-1/\tau}, \quad \tau = 32 \text{（半衰期，超参数）}$$

直觉：$\tau=32$ 意味着距当前 token 32 步之外的影响权重已衰减到一半。

### Step 3：影响权重与密集优势

将 Future-KL 映射为**影响权重** $f_t$，并做非对称 clip 以避免过大扰动：

$$f_t = \text{clip}\!\left(\exp(\text{FutureKL}_t),\ 1 - \varepsilon_\text{flow},\ 1 + \varepsilon_\text{high}\right)$$

其中 $\varepsilon_\text{flow} = 0.0$，$\varepsilon_\text{high} = 0.2$（非对称，仅允许权重正向增大至 1.2）。

**密集优势（Dense Advantage）**：将 GRPO 组内归一化优势 $\hat{A}_t$ 与影响权重相乘：

$$\tilde{A}_t = \hat{A}_t \cdot f_t$$

**信用分配语义**：
- 若位置 $t$ 的 token 在本次更新中增大了概率，且后续轨迹中也有许多 token 概率同步增大（即正向更新具有"传导效应"），则 $\text{FutureKL}_t > 0$，$f_t > 1$，该 token 的优势被放大 → **正确推理步骤得到更强的正向信号**；
- 若后续轨迹概率下降（负向传导），则 $f_t < 1$，优势被压缩 → **与最终结果不一致的推测步骤被削弱**。

### Step 4：最终目标函数

FIPO 延用 DAPO 的 Decoupled Clip 框架，将 $\hat{A}_t$ 替换为 $\tilde{A}_t$：

$$\mathcal{L}_\text{FIPO} = -\mathbb{E}\!\left[\min\!\left(r_t(\theta) \cdot \tilde{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon_l,\ 1+\varepsilon_h) \cdot \tilde{A}_t\right)\right]$$

其中 $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$ 为概率比，clip 范围使用非对称设置 $[\varepsilon_l, \varepsilon_h] = [0.2, 0.28]$。

### 关键超参数（Qwen2.5-32B-Base 实验配置）

| 超参数 | 取值 | 说明 |
|--------|------|------|
| Global Batch Size | 512 | |
| Group Size $G$ | 16 | 每问题采样 16 条轨迹 |
| Learning Rate | $1 \times 10^{-6}$ | |
| Future-KL 半衰期 $\tau$ | 32 | $\gamma = 2^{-1/32}$ |
| Influence Weight Clip | $[1.0, 1.2]$ | 非对称 |
| Max Response Length | 20,480 tokens | |
| Policy Clip Ratio | $[0.2, 0.28]$ | 非对称，来自 DAPO |

### 方法对比总结

| 方法 | 优势计算 | 需要 Critic | 信用粒度 |
|------|----------|------------|---------|
| GRPO | 组内归一化标量广播 | ✗ | 粗粒度（序列级） |
| PPO + GAE | Critic 估计 + 指数加权 TD(λ) | ✓ | 细粒度（token 级） |
| DAPO | GRPO + Decoupled Clip | ✗ | 粗粒度（序列级） |
| **FIPO** | GRPO 优势 × Future-KL 权重 | ✗ | **细粒度（token 级）** |



## 📊 实验与性能评价

### 训练设置

- **基础模型**: Qwen2.5-32B-Base（从预训练基础模型直接开始，不依赖 SFT 或已有推理模型的先验知识）
- **训练框架**: verl（HybridFlow，Sheng et al., EuroSys 2025），基于 DAPO 代码库扩展
- **评估基准**: AIME 2024、AIME 2025
- **采样设置**: 评估时温度 $T=0.6$，Pass@1 计算方式（每题多次采样取平均）

### 主要结果

#### AIME 基准性能对比

| 方法 | 基础模型 | AIME 2024 | AIME 2025 | 备注 |
|------|----------|-----------|-----------|------|
| **DAPO** | Qwen2.5-32B | 50.0% | 38.0% | 主要基线 |
| **DeepSeek-R1-Zero** | Qwen-32B | ~47.0% | - | 开源推理基线 |
| **o1-mini** | - | ~56.0% | - | 商业基线 |
| **FIPO（收敛）** | Qwen2.5-32B | **56.0%** | **43.0%** | ±~2% 标准误 |
| **FIPO（峰值）** | Qwen2.5-32B | **58.0%** | - | 训练过程峰值 |
| VAPO（参照） | Qwen-32B | 60.4 | - | 需要 Critic 网络 |

FIPO 在 **不使用 Critic 网络** 的前提下，将 AIME 2024 从 DAPO 的 50.0% 提升至 56.0%（+6%），接近同等规模需要 Critic 的 VAPO（60.4%）。

#### 推理链长度增长

| 方法 | 平均 CoT 长度 |
|------|-------------|
| DAPO 训练前 | ~4,000 tokens |
| DAPO 训练后 | ~4,000 tokens（停滞） |
| **FIPO 训练后** | **>10,000 tokens**（+150%） |

这一结果表明，FIPO 的密集信用信号能有效引导模型学会进行更深、更长的推理，而非在浅层思考后就输出答案。

### 关键消融分析（定性）

论文的实验设计刻意选择在**干净的基础模型**上训练（不依赖 SFT 热身或已有推理模型权重），以确保性能提升源自算法本身而非继承先验知识。

**Future-KL 的物理意义验证**：在正确轨迹中，高影响权重的 token 往往出现在推理的关键转折点（如"因此我们需要…"、"注意到…"），而非重复计算部分——这与 Meng et al. (ICLR 2025) 关于 RLVR 训练中 token 级分布漂移稀疏性的发现一致。

### 局限性

1. **信用信号自举**：Future-KL 依赖当前策略对自身的信息估算信用，属于自举（bootstrapping）估计，可能在训练早期引入偏差，需要 clip 机制抑制极端情况；
2. **超参数敏感性**：半衰期 $\tau=32$ 和影响权重上界 $1+\varepsilon_\text{high}=1.2$ 需要仔细调优，论文未充分展示对这两个参数的敏感性分析；
3. **评估基准较窄**：主要在 AIME 2024/2025（数学竞赛题）上验证，缺乏代码、科学推理等其他领域的泛化实验；
4. **对比基线不含最新方法**：BAPO、GSPO、VAPO 等在实验时间上存在交叉，论文未完整横向对比；
5. **代码未附具体 URL**：论文声称开源但未在正文给出仓库地址。

---

## 💡 启发与我的想法

### 可借鉴之处

1. **「未来影响力」作为信用估计的通用思路**：FIPO 的 Future-KL 本质上是一种**无模型、无 Critic 的密集信用分配机制**，其核心直觉（"当前 token 如何影响后续轨迹"）可以推广到其他序列决策任务，如代码生成、工具调用等。

2. **非对称 Clip 设计**：将影响权重限制在 $[1.0, 1.2]$（只允许放大，不允许衰减到 1.0 以下），以及不对称策略 clip $[0.2, 0.28]$，体现了"正向强化比负向惩罚更重要"的训练哲学，值得在其他 RLHF 设置中参考。

3. **在基础模型（非 SFT 后的模型）上直接训练**：这一选择排除了"SFT 热身带来的先验"的混淆因素，使算法贡献的测量更干净，是实验设计上的优良实践。

4. **与 KV Cache 稀疏研究的联系**：推理链长度从 4k 扩展到 10k token，意味着 KV Cache 的体积随之增长——这给 KV 稀疏化方法（如 KvZap）提供了更强的应用动机：长推理链中早期 token 的 KV 可能更适合被稀疏压缩。

### 改进空间

1. **自适应半衰期 $\tau$**：固定的 $\tau=32$ 不区分题目难度和推理阶段。直觉上，对于长推理链的早期阶段，应该给予更长的半衰期（关注更远的未来）；对于答案输出附近，应该缩短。可以设计动态调整策略。

2. **与 Process Reward Model（PRM）结合**：Future-KL 提供了无监督的 token 级信用代理，而 PRM 提供了有监督的步骤级奖励——两者结合可能进一步细化信用分配颗粒度。

3. **多维度评估**：Future-KL 的信用分配在代码推理（工具调用成功率）、逻辑推理（GPQA）等领域的有效性有待验证，推理错误的"关键 token"未必与数学推理场景相同。

4. **与 KV Cache 压缩协同优化**：长推理链中大量 token 的影响权重趋近于 1.0（非关键 token），这些 token 在 KV 层的冗余度也更高，是 KV 稀疏化的优先候选——FIPO 的权重信息可以作为 KV 剪枝的指引信号。

### 下一步 Action
- [ ] 追踪 FIPO 官方代码仓库（基于 verl），复现 Qwen2.5-7B 规模的实验
- [ ] 阅读 Meng et al. (ICLR 2025) "Sparse but Critical"，理解 token 级分布漂移的稀疏性理论依据
- [ ] 思考 Future-KL 权重与 KV Cache 重要性评分（如 SnapKV 的注意力分数）的关联性
- [ ] 关注 FIPO 在非数学领域（代码、多步工具调用）的 follow-up 工作


