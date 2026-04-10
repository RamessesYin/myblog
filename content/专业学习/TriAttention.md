---
title: "TriAttention"
tags:
  - KV Cache
  - LLM推理
  - 注意力机制
  - 长文本推理
  - RoPE
---

## 🧠 核心摘要 (TL;DR)
> TriAttention 发现 Transformer 中大多数注意力头的 Q/K 向量在 pre-RoPE 空间中会向固定非零中心聚集（Q/K Concentration），由此推导出一个基于三角函数级数的得分函数，可在**不观察真实注意力分数**的情况下预判哪些位置的 KV 会被关注；在 AIME25 长推理任务上，实现了与全注意力相当的精度，同时带来 **2.5× 吞吐量提升或 10.7× KV 显存缩减**。

## 🏷️ 元数据
- **论文标题**: TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
- **发表机构/会议**: arXiv 2026
- **年份**: 2026
- **链接**: [[TriAttention_paper|📄论文]] | [💻 Code](https://github.com/WeianMao/triattention)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #kv-cache #attention #rope #long-context #reasoning #llm-inference
- **前置知识 (Prerequisites)**: [[KV稀疏调研]]
- **相关节点 (Related Notes)**: [[KvZap]], [[RL中的KV稀疏讨论]], [[AttentionRes]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

大语言模型在长推理任务（如 AIME 数学竞赛、LiveCodeBench 编程题）中需要生成数万乃至数十万 token。自回归解码过程中，每步生成都要访问全量的 KV Cache，随序列长度线性增长的显存占用已成为系统瓶颈：

- **吞吐量瓶颈**：KV Cache 占满 HBM，批量大小被迫降低，GPU 利用率不足
- **延迟瓶颈**：注意力计算的内存带宽消耗随上下文窗口线性增长，解码速度随序列变长而持续劣化

现有 KV Cache 压缩方法可分为两大范式，但各有根本局限：

**范式一：基于历史注意力分数的启发式淘汰**  
H2O（Zhang et al., NeurIPS 2023）发现少数"重型 token"主导了注意力权重，仅保留这些 token 可大幅减少 Cache；Scissorhands（Liu et al., NeurIPS 2023）提出"重要性持久性假设"，认为历史重要 token 在未来步也会重要；StreamingLLM（Xiao et al., ICLR 2024）发现初始 token 充当注意力汇聚点（Attention Sink），必须保留以维持稳定性。这些方法依赖**过去**的注意力分数来预测**未来**重要性，但在长推理场景下，推理的早期阶段尚未出现足够的历史分数，导致决策偏差。

**范式二：基于当前 Query 的动态选择**  
Quest（Tang et al., ICML 2024）跟踪 KV Cache 分页的最小/最大 Key 值，用当前 Query 估算每页重要性，实现 2.23× 自注意力加速；SnapKV（Li et al., NeurIPS 2024）观察每个注意力头对提示末尾窗口的偏好模式，自动选取重要 token。这些方法的核心问题是**需要实际执行注意力操作后才能决定保留哪些 KV**，难以在预填充阶段提前压缩，且 FlashAttention 等内核不暴露完整注意力矩阵，工程实现复杂。

Expected Attention（Devoto et al., 2025）尝试估计未来查询分布，推导预期注意力得分来做压缩决策，思路接近 TriAttention，但未利用 RoPE 本身的代数结构。

### 推理模型的特殊挑战

DeepSeek-R1（DeepSeek-AI, 2025）等思维链推理模型在推理过程中会生成大量中间步骤，单次推理可达 32K+ token。此类场景下：
1. **输出长度不可预知**，难以提前规划 Cache 预算
2. **推理步骤具有内在逻辑相关性**，不同于普通文档的稀疏注意力模式
3. **延迟要求严格**，每步解码的压缩决策需高效执行

R-KV（Cai et al., 2025）专门针对推理模型设计了冗余感知压缩，但未从 RoPE 代数结构中挖掘更深层的理论依据。RaaS（Hu et al., ACL 2025）则提出推理感知注意力稀疏，关注推理步骤之间的注意力模式差异。

### 论文的切入点

TriAttention 的核心洞察是：**不必观察实际注意力分数**，可以从 pre-RoPE 的 Q/K 向量分布特性出发，用三角函数级数在理论层面预测"哪些距离（相对位置）会受到偏好关注"，并以 Key 向量模长作为绝对重要性的辅助估计。这使得 TriAttention 成为**第一个从 RoPE 代数结构推导得到理论保证的 KV 压缩框架**。

### 相关工作分类

**先驱/奠基工作**

| 论文 | 核心思路 | 局限 |
|------|---------|------|
| H2O (Zhang et al., NeurIPS 2023) | 保留重型 token（高累积注意力权重） | 依赖历史分数，长推理早期失准 |
| Scissorhands (Liu et al., NeurIPS 2023) | 重要性持久性假设 | 同上 |
| StreamingLLM (Xiao et al., ICLR 2024) | 保留 Attention Sink + 滑动窗口 | 无法自适应任务需求 |
| RoFormer/RoPE (Su et al., 2024) | 旋转位置编码，绑定相对位置 | 基础理论，TriAttention 利用其代数结构 |

**直接对比方法**

| 论文 | 核心思路 | TriAttention 优势 |
|------|---------|------------------|
| Quest (Tang et al., ICML 2024) | 查询感知分页稀疏 | 无需运行时注意力操作，理论推导更优雅 |
| SnapKV (Li et al., NeurIPS 2024) | 提示偏好感知压缩 | 适用于输出端压缩，SnapKV 主要针对输入提示 |
| R-KV (Cai et al., 2025) | 推理模型冗余感知压缩 | AIME25 上显著超越（32.9% vs 17.5%） |
| DuoAttention (Xiao et al., ICLR 2025) | 检索头/流式头分离 | 聚焦 token 级压缩，而非头级分类 |
| PyramidKV (Cai et al., 2024) | 金字塔式分层 Cache 分配 | TriAttention 基于理论得分，分配更精确 |

**近期 Follow-up 与相关方向**

| 论文 | 方向 |
|------|------|
| EliteKV (Zhou et al., 2025) | RoPE 频率选择 + 联合低秩投影，与 TriAttention 的 RoPE 视角互补 |
| LazyEviction (Zhang et al., 2025) | 滞后式 KV 淘汰，观察注意力模式后再决策 |
| RaaS (Hu et al., ACL 2025) | 推理感知注意力稀疏，面向推理模型定制 |
| Ada-KV (Feng et al., 2025) | 自适应预算分配，跨头优化 Cache 分配 |
| RocketKV (Behnam et al., 2025) | 两阶段 KV 压缩，先粗粒度后细粒度 |
| Expected Attention (Devoto et al., 2025) | 估计未来查询分布，推导预期注意力 |
| FlashAttention-2/3 (Dao 2024; Shah et al. 2024) | 高效注意力内核，TriAttention 在其之上实现 |
| Survey: Efficient LLM Inference (Zhou et al., 2024) | 涵盖数据/模型/系统三层优化综述 |


## ⚙️ 核心设计与机制

### Q/K Concentration 现象

TriAttention 的理论基础是作者对 Transformer 中 Q/K 向量分布的实证发现。在 **pre-RoPE 空间**（即旋转位置编码施加之前）中，查询向量 $\mathbf{q}_t$ 和键向量 $\mathbf{k}_t$ 并不均匀分布于超球面，而是呈现出向固定非零中心聚集的现象：

$$
\mathbf{q}_t \approx \boldsymbol{\mu}_q + \boldsymbol{\epsilon}_t^q, \quad \mathbf{k}_t \approx \boldsymbol{\mu}_k + \boldsymbol{\epsilon}_t^k
$$

其中 $\boldsymbol{\mu}_q, \boldsymbol{\mu}_k$ 是跨位置、跨上下文保持稳定的均值向量，$\boldsymbol{\epsilon}$ 是零均值的噪声扰动。

**定量化：Mean Resultant Length（均值合力长度）**

作者借用定向统计学（Mardia & Jupp, 1999）中的 Mean Resultant Length $R \in [0, 1]$ 来度量向量聚集程度：

$$
R = \left\| \mathbb{E}\left[\frac{\mathbf{x}}{||\mathbf{x}||}\right] \right\|
$$

- $R \to 1$：向量高度聚集于某个方向（完美聚集）
- $R \to 0$：向量均匀分布在超球面

实验表明，在 Qwen3、Qwen2.5、Llama3 等主流模型的大多数注意力头中，pre-RoPE Q/K 向量的 $R$ 值显著大于 0，且该统计量跨位置和上下文保持稳定（图 2 的可视化验证）。

### 从聚集现象到三角函数级数

RoPE 将位置信息编码为旋转变换。设 token $t$ 在第 $f$ 个频率分量上的 pre-RoPE 查询向量为 $\mathbf{q}_f$，经 RoPE 旋转角 $\omega_f \cdot t$ 后变为 $R(\omega_f t)\mathbf{q}_f$（$R$ 表示旋转矩阵）。则位置 $s$（key）对位置 $t$（query）的注意力 logit 为：

$$
\text{logit}(t, s) = \sum_f \langle R(\omega_f t)\mathbf{q}_f, R(\omega_f s)\mathbf{k}_f \rangle = \sum_f \langle R(\omega_f \Delta)\mathbf{q}_f, \mathbf{k}_f \rangle
$$

其中 $\Delta = t - s$ 为相对距离。展开内积得到：

$$
\text{logit}(\Delta) = \sum_f \left[ q_f^{(1)} k_f^{(1)} \cos(\omega_f \Delta) + q_f^{(1)} k_f^{(2)} (-\sin(\omega_f \Delta)) + q_f^{(2)} k_f^{(1)} \sin(\omega_f \Delta) + q_f^{(2)} k_f^{(2)} \cos(\omega_f \Delta) \right]
$$

**当 Q/K 向量高度聚集时**（即 $\mathbf{q}_f \approx \boldsymbol{\mu}_f^q$，$\mathbf{k}_f \approx \boldsymbol{\mu}_f^k$），注意力 logit 近似为关于 $\Delta$ 的**三角函数级数**：

$$
\text{logit}(\Delta) \approx \sum_f \left[ a_f \cos(\omega_f \Delta) + b_f \sin(\omega_f \Delta) \right]
$$

其中系数 $a_f, b_f$ 仅依赖于 Q/K 中心 $\boldsymbol{\mu}_f^q, \boldsymbol{\mu}_f^k$，与当前具体的 token 内容无关。这意味着：

> **高度聚集的注意力头存在一个与内容无关的"位置偏好"函数**，可提前校准，无需在推理时计算真实注意力分数。

### TriAttention 得分函数

实际向量不完全等于均值，需要额外处理非聚集分量。TriAttention 将最终重要性得分分为两部分：

**分量一：三角函数系列得分（处理聚集分量）**

$$
S_{\text{trig}}(k, \Delta) = \sum_f ||\mathbb{E}[\mathbf{q}_f]|| \cdot ||\mathbf{k}_f|| \cdot \cos(\omega_f \Delta + \varphi_f)
$$

其中 $\varphi_f$ 是由 Q 均值向量中心的方向角确定的相位偏移。这部分分数刻画了：
1. 该键向量与 "平均查询" 的方向对齐程度（通过余弦项）
2. 该键向量自身的模长（通过 $||\mathbf{k}_f||$，模长大意味着在注意力中贡献大）

**分量二：模长归一化得分（处理非聚集分量）**

$$
S_{\text{norm}}(k) = \sum_f (1 - R_f) \cdot \mathbb{E}[||\mathbf{q}_f||] \cdot ||\mathbf{k}_f||
$$

当聚集程度低（$R_f \to 0$）时，该分量权重增大；此时无法用均值中心预测方向偏好，退化为纯粹基于模长的重要性估计。

**最终得分**（加权融合）：

$$
S(k, \Delta) = S_{\text{trig}}(k, \Delta) + S_{\text{norm}}(k)
$$

两部分通过聚集度 $R_f$ 自适应加权，在聚集头上三角函数得分主导，在弱聚集头上模长得分兜底。

### 校准阶段：提取 Q/K 统计量

TriAttention 需要在实际推理前进行**一次性校准**，提取每个注意力头的 Q/K 聚集统计量：

1. 在少量**校准数据**（如 128 条代码片段）上前向传播
2. 对每个头的每个频率分量 $f$，计算 pre-RoPE Q/K 向量的均值 $\boldsymbol{\mu}_f^q, \boldsymbol{\mu}_f^k$ 及聚集度 $R_f$
3. 由此确定三角函数得分函数的系数 $a_f, b_f, \varphi_f$ 及模长加权系数

**跨域泛化**：作者实验表明，在编程数据上校准后的统计量，在数学推理任务上同样有效，说明 Q/K 聚集是模型的内在属性，与任务域无关。

### 推理时执行流程

```
校准阶段（离线，仅一次）：
  提取各头 pre-RoPE Q/K 均值 → 计算三角函数系数

推理阶段（在线，每步解码）：
  1. 新 token 进入 → 计算其 pre-RoPE K 向量的模长
  2. 根据当前位置 Δ → 查询预校准的三角函数得分
  3. S(k, Δ) = S_trig(k, Δ) + S_norm(k)
  4. 选取 Top-K 得分最高的 KV 对 → 执行稀疏注意力
  5. 淘汰低得分 KV，释放显存
```

整个推理时决策过程**无需实际计算注意力权重**，计算量仅为逐元素的三角函数运算和模长计算，额外开销可忽略不计。

### 理论验证

作者用皮尔逊相关系数 $r$ 来衡量三角函数得分与真实注意力 logit 的重建精度：
- 跨 Qwen3、Qwen2.5、Llama3 三个架构，各头的平均 $r > 0.5$
- 全局平均 $r = 0.72$（图 3）

这为 TriAttention 的有效性提供了直接的数学依据，而非单纯的实证黑盒验证。


## 📊 实验与性能评价

### 对比基线 (Baseline)

- **Full Attention**：标准全注意力（上界）
- **R-KV**（Cai et al., 2025）：专门为推理模型设计的冗余感知 KV 压缩
- 其他 KV 压缩基线（SnapKV、H2O 等，在部分任务上对比）

所有实验基于 **Qwen3-8B**（支持长上下文推理的模型）；TriAttention 的校准数据为 128 条代码片段，校准与推理完全分离。

### 主要实验结果

**AIME25（美国数学邀请赛 2025，32K token 生成预算）**

| 方法 | 精度 | KV 预算比例 |
|------|------|------------|
| Full Attention | 40.8% | 100% |
| **TriAttention** | **32.9%** | ~9.3%（10.7× 缩减） |
| R-KV | 17.5% | ~9.3% |

在相同的 KV 显存预算下，TriAttention 以接近 Full Attention 一半精度损失的代价超越 R-KV 近一倍。

**效率指标（AIME25，以精度持平 Full Attention 为目标）**

- **吞吐量提升**：2.5× （相比 Full Attention）
- **KV 显存缩减**：10.7×

**MATH 500（1024 token 生成预算）**

| 方法 | 精度 | 吞吐量（tokens/sec） |
|------|------|---------------------|
| Full Attention | 69.6% | 223 |
| **TriAttention** | **68.4%** | **1,405** |

吞吐量提升 **6.3×**，精度损失仅 1.2 个百分点，表现非常接近全注意力。

**LongBench（长文本理解综合基准）**

TriAttention 在多个长文档理解子任务上（如 HotpotQA、2WikiMQA、MuSiQue）保持与 Full Attention 相当的表现，验证了方法在问答和多跳推理场景的泛化性。

**RULER（长上下文能力评测）**

在 RULER 基准的 Needle-in-a-Haystack 等任务上，TriAttention 在各种上下文长度下均维持有竞争力的表现。

### 消融实验

| 消融设置 | AIME25 精度 |
|---------|------------|
| TriAttention（完整） | 32.9% |
| 仅 S_trig（去掉 S_norm） | 下降 ~3-5%（推断）|
| 仅 S_norm（退化为模长排序） | ~R-KV 水平 |
| 随机选 KV | <10% |

实验表明三角函数分量和模长分量的组合缺一不可，两者互补覆盖聚集度强/弱的注意力头。

**校准域泛化消融**：分别在代码数据、数学数据上校准，两组统计量在数学推理任务上的表现差异极小，佐证 Q/K 聚集是模型固有属性。

### 局限性 (Limitations)

1. **弱聚集头的退化**：当某些头的 $R$ 值接近 0（低聚集度），三角函数得分退化为模长估计，理论优势减弱，此时 TriAttention 本质上等同于基于模长的 KV 排序，与其他方法差距缩小
2. **需要一次校准**：虽然开销小，但增加了部署流程的复杂度；对于频繁更新的新模型，每次都需重新校准
3. **目前主要在 Qwen3-8B 上验证**：不同架构的 Q/K 聚集程度差异较大，在低聚集度架构上的效果有待验证
4. **离线 KV 压缩假设**：TriAttention 本质上是基于位置偏好的静态排序，未充分利用运行时的 Query-Key 动态交互（Quest 等方法的优势）
5. **极长上下文下位置感知的衰减**：三角函数得分依赖相对位置 $\Delta$，当 $\Delta$ 极大时三角函数振荡特性可能导致得分预测不稳定

---

## 💡 启发与我的想法

### 可借鉴之处

1. **利用模型内在统计量而非运行时计算**：TriAttention 将"哪些 KV 重要"的问题转化为一个可离线校准的统计模型，这一思路对系统设计有通用价值——与其在推理时做额外计算，不如提前从模型权重/激活分布中提取结构信息。

2. **RoPE 的代数结构是金矿**：RoPE 将位置信息编码为旋转，其内积具有完美的三角展开形式。EliteKV 从频率选择角度挖掘，TriAttention 从 Q/K 聚集角度挖掘，均证明这一结构蕴含大量待开发的优化空间。

3. **弱聚集头的优雅处理**：通过 $R_f$ 加权自适应融合两类得分，避免了二元分类的硬切换，是一种鲁棒的设计。

4. **与 FlashAttention 兼容**：TriAttention 不要求访问完整注意力矩阵，天然兼容 FlashAttention 的内存高效实现，这是系统落地的重要前提。

### 改进空间

1. **与动态 Query 信息的融合**：TriAttention 的得分函数完全依赖 Q 的均值中心，忽略了当前 Query 的具体内容。可考虑在三角函数得分的基础上叠加一个轻量的 Query-Key 交互项（如 Quest 中的页级估计），进一步提升精度。

2. **头级别自适应压缩率**：当前 TriAttention 对所有头使用相同的 KV 预算，而 Ada-KV（Feng et al., 2025）表明跨头自适应分配可以在相同总 Budget 下取得更好效果。两者可组合：对聚集头用三角函数得分选 KV，对弱聚集头增加 Budget 或使用其他策略。

3. **与 MLA/GQA 的适配**：DeepSeek-V3 等模型采用 Multi-head Latent Attention，KV Cache 被投影到低维空间；现有 TriAttention 的公式推导基于标准 MHA，需要重新推导 pre-RoPE 统计量在 MLA 架构下的形式。

4. **跨层 KV 共享**：xKV 发现不同层的 KV Cache 存在低秩对齐，TriAttention 对这一跨层结构尚未利用。可探索将三角函数得分扩展为跨层联合排序。

5. **理论上限分析**：对于给定的 Q/K 聚集程度 $R$，三角函数近似的误差有多大？能否推导出一个关于 $R$ 的精度界，指导不同头的压缩率设置？

### 下一步 Action
- [ ] 复现 TriAttention 在 Qwen3 上的 AIME25 实验，验证 10.7× 显存缩减与 2.5× 吞吐量的可复现性
- [ ] 探索 TriAttention 与 Quest 动态选择的混合策略：对强聚集头用 TriAttention，对弱聚集头用 Quest
- [ ] 调研 TriAttention 的 CUDA 内核实现，看是否可以与 PagedAttention/vLLM 集成
- [ ] 分析 KV 稀疏调研笔记中提到的方法与 TriAttention 的互补性


