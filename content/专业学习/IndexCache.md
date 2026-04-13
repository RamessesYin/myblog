---
title: "IndexCache：跨层稀疏索引复用"
tags:
  - sparse-attention
  - kv-cache
  - cross-layer
  - long-context
  - llm推理
  - deepseek
---

## 🧠 核心摘要 (TL;DR)
> DeepSeek Sparse Attention（DSA）已将核心注意力从 O(L²) 降至 O(Lk)，但其 **Indexer 本身仍以 O(L²) 运行于每一层**，在极长上下文下成为新瓶颈。IndexCache 观察到相邻层 Indexer 的 top-k 选择高度重叠（70–100%），提出将层分为**Full 层**（运行完整 Indexer）和 **Shared 层**（复用最近 Full 层的索引），配合贪心层选择算法（训练无关）与跨层蒸馏损失（训练感知），在 30B 模型 / 200K 上下文下实现 **1.82× Prefill 加速** 和 **1.48× Decode 加速**，精度损失极小。

## 🏷️ 元数据
- **论文标题**: IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse
- **发表机构/会议**: arXiv（清华大学、智谱 AI GLM 团队）
- **年份**: 2026
- **链接**: [[IndexCache_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #sparse-attention #cross-layer #kv-cache #long-context #indexer #deepseek
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[KvZap]]
- **相关节点 (Related Notes)**: [[RL中的KV稀疏讨论]], [[SparseButCritical]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

大语言模型在长上下文（100K–1M token）场景下，标准 Softmax 注意力的时间与显存复杂度均为 O(L²)，成为推理效率的核心瓶颈。现有稀疏注意力方案（如 DeepSeek Sparse Attention，DSA）引入了轻量级 **Indexer** 来预先筛选 top-k 关键 token，将核心注意力计算从 O(L²) 降至 O(Lk)。然而，这套方案存在一个被忽视的隐性瓶颈：

> **Indexer 本身的复杂度仍为 O(L²)**——在每一层独立运行，随着上下文长度增加，Indexer 的开销从计算瓶颈的"次要角色"逐渐变成"主角"。

具体而言，当上下文长度超过 64K 时，Indexer 的耗时开始超越稀疏注意力计算本身，200K 上下文时其占比可达 Prefill 总时长的相当大比例。

### 论文的核心观察与切入点

**关键观察**：Indexer 在相邻层之间的 top-k 选择**高度相关**——实验测量显示相邻层共享 70–100% 的选中 token。这意味着每层都从头运行完整的 O(L²) Indexer，存在大量冗余计算。

**切入点**：将 N 个 Transformer 层划分为两类：
- **Full 层（F 层）**：保留完整 Indexer，生成新鲜的 top-k 索引
- **Shared 层（S 层）**：直接复用最近前驱 F 层生成的 top-k 索引，跳过 O(L²) 的 Indexer 计算

这一修改在实现上只需推理代码中**一个条件分支**，额外开销极小。

### 相关工作

#### 先驱/奠基工作

| 论文 | 机构/年份 | 核心贡献 |
|------|-----------|----------|
| **Attention is All You Need** (Vaswani et al., 2017) | Google Brain | Transformer 架构，定义了 Softmax 注意力的标准形式 |
| **H2O** (Ge et al., ICLR 2024, arXiv:2310.01801) | UIUC 等 | 基于注意力分数自适应地驱逐 KV Cache，首创"模型告知你丢弃什么"范式 |
| **SparQ Attention** (Ribar et al., 2023, arXiv:2312.04985) | Graphcore | 通过选择性地只取少量 query 维度来近似注意力分数，节省带宽 |
| **MInference 1.0** (Jiang et al., NeurIPS 2024, arXiv:2407.02490) | Microsoft | 识别三种注意力 pattern（A-shape / Vertical-Slash / Block-Sparse），在 Prefill 阶段动态稀疏化，最高 10× 加速 |
| **DuoAttention** (Xiao et al., arXiv:2410.10819) | MIT / CMU | 区分 Retrieval 头（保留完整 KV Cache）和 Streaming 头（只保留局部窗口），实现头级别异构稀疏 |

#### 直接对比方法（论文明确对比或引用）

| 论文 | 机构/年份 | 方法要点 |
|------|-----------|----------|
| **DeepSeek-V2 / DSA** (Liu et al., 2024) | DeepSeek AI | 提出 DeepSeek Sparse Attention，以 O(L²) Indexer + O(Lk) 注意力为标准设计，IndexCache 正是在其基础上优化 |
| **DeepSeek-V3.2** (Liu et al., 2025) | DeepSeek AI | DSA 的生产版本，IndexCache 以其为 Baseline 进行对比 |
| **HySparse** (Gao et al., 2026) | 北航等 | Hybrid Sparse Attention，动态在稀疏与稠密注意力之间切换，是 IndexCache 的直接对比基线 |
| **SampleAttention** (Zhu et al., 2024, arXiv:2406.15486) | 商汤等 | 自适应结构化稀疏注意力，TTFT 降低最高 2.42×，代表结构化稀疏路线 |
| **MoBA** (Lu et al., 2025) | 清华等 | Mixture of Block Attention，在 block 级别动态混合不同稀疏 pattern |
| **InfLLM-V2** (Zhao et al., 2025) | 清华等 | 动态在稠密与稀疏注意力间切换，专注超长文本 |
| **FlexPrefill** (Lai et al., 2025) | - | 自适应调整 Prefill 阶段的稀疏比例 |
| **QUEST** (Tang et al., 2024) | - | Query-aware Sparse Attention，动态选取最相关的 KV 块 |
| **SwiftKV** (Qiao et al., 2025) | - | 跨层 KV 复用，与 IndexCache 的"共享索引"思路最为接近，但 SwiftKV 复用的是 KV 向量本身，IndexCache 复用的是索引 |
| **LServe** (Yang et al., 2025, arXiv:2502.14866) | MIT HAN Lab | 统一稀疏注意力的 LLM Serving 系统，包含多种稀疏 pattern |
| **TidalDecode** (Yang et al., 2025) | - | 稀疏注意力 + 推测解码组合方案 |
| **Native Sparse Attention (NSA)** (Yuan et al., 2025, arXiv:2410.02703) | DeepSeek AI | 硬件对齐 + 可原生训练的稀疏注意力，与 IndexCache 的"训练感知"路线呼应 |

#### 相关 Follow-up 方向

| 论文 | 机构/年份 | 关联性 |
|------|-----------|--------|
| **PyramidKV** (Cai et al., 2024, arXiv:2406.02069) | - | 跨层异构 KV Cache 分配（底层多、顶层少），与 IndexCache 的跨层视角同源 |
| **MLKV** (Zuhri et al., 2025) | - | 多层共享 KV 头（Multi-Layer KV Heads），从 KV 向量共享而非索引共享的角度节省计算 |
| **OmniKV** (Hao et al., 2025) | - | 跨层动态 KV，结合注意力分数决定保留策略 |
| **SageAttention2** (Zhang et al., ICML 2025) | 清华大学 | INT4/FP8 量化加速注意力，与稀疏注意力形成正交的加速路线 |
| **Loki** (Singhania et al., 2024, arXiv:2406.02542) | 马里兰大学等 | 在低维空间计算注意力分数以高效选取 top-k token，与 IndexCache 的 Indexer 加速互补 |


## ⚙️ 核心设计与机制

### 整体架构

IndexCache 以 DSA（DeepSeek Sparse Attention）为基础，在其之上引入**跨层索引共享机制**。整个 N 层 Transformer 被划分为 F（Full）层集合和 S（Shared）层集合，两类层的比例由目标压缩率决定。

```
层 0 [F]：运行完整 Indexer → 生成 top-k 索引 I₀
层 1 [S]：复用 I₀，跳过 Indexer 计算
层 2 [S]：复用 I₀，跳过 Indexer 计算
层 3 [F]：运行完整 Indexer → 生成新索引 I₃
层 4 [S]：复用 I₃，跳过 Indexer 计算
...
```

关键的实现细节：**每个 S 层只复用其最近的前驱 F 层的索引**，而非全局共享同一组索引，这样可以平衡索引新鲜度与计算节省。

---

### 方案一：训练无关的 IndexCache（Greedy 层选择）

**动机**：最朴素的做法是均匀保留（每 r 层保留一个 F 层），但论文发现均匀间隔会导致显著的质量下降。原因是**不同层对 Indexer 的敏感度差异极大**——某些早期层和过渡层如果被转为 S 层，会严重损害质量；而另一些层则几乎可以无损地共享索引。

**贪心搜索算法（Greedy Layer Selection）**：

1. 初始化：所有层均为 F 层（全量 Indexer）
2. 迭代 K 步（K = 目标 S 层数量）：
   - 对每个当前 F 层，**试探性地将其转为 S 层**
   - 在小型标定集（calibration set）上评估语言模型困惑度（LM loss）
   - **提交损失最小的那个翻转**（即将最不敏感的 F 层转为 S 层）
3. 重复直到达到目标 S 层数量

**复杂度优化**：原始搜索需要 $\frac{N(N-1)}{2}$ 次前向传播，通过**流水线并行搜索**（按块顺序搜索）可大幅加速。

**实验结果（30B 模型，LongBench v2 基准）**：

| 方案 | Indexer 保留比例 | Long Avg 分数 | 备注 |
|------|----------------|--------------|------|
| DSA Baseline | 1/1 | 50.2 | 所有层均有 Indexer |
| 均匀间隔 | 1/4 | 43.0 | 严重退化 |
| **Greedy 搜索** | **1/2** | **50.3** | 几乎无损 |
| **Greedy 搜索** | **1/4** | **49.9** | 仅降 0.3 分 |

> **结论**：搜索到的非均匀层分配方案在 1/4 Indexer 保留率下，将均匀间隔的 7.2 分退化（43.0 vs 50.2）几乎完全消除，只剩 0.3 分差距。

---

### 方案二：训练感知的 IndexCache（多层蒸馏损失）

**标准 DSA 蒸馏**：DSA 本身通过 KL 散度损失训练 Indexer，让每层的 Indexer 对准**该层自身**的注意力分布：

$$\mathcal{L}_{\text{std}}^I = \sum_t D_{\text{KL}}\!\left(p_t^{(\ell)} \;\|\; q_t^{(\ell)}\right)$$

其中 $p_t^{(\ell)}$ 是第 $\ell$ 层的完整注意力分布，$q_t^{(\ell)}$ 是 Indexer 选择的稀疏分布。

**多层蒸馏损失（Multi-Layer Distillation）**：对于一个 F 层 $\ell$，它同时为 S 层 $\ell+1, \ldots, \ell+m$ 服务。因此，将其 Indexer 的监督信号扩展为对 m+1 层均负责：

$$\mathcal{L}_{\text{multi}}^I = \sum_{j=0}^{m} \frac{1}{m+1} \sum_t D_{\text{KL}}\!\left(p_t^{(\ell+j)} \;\|\; q_t^{(\ell)}\right)$$

**关键理论结论（Proposition 1）**：上述多层损失产生的梯度等价于将 Indexer 蒸馏对准**所有层注意力分布的平均值** $\bar{p}_t = \frac{1}{m+1}\sum_j p_t^{(\ell+j)}$，即：

> 训练感知的 IndexCache Indexer 学到的是"跨越它所服务的所有层的一致性 top-k 选择"，是所有被服务层注意力分布的共识。

**实验结果（30B 模型）**：

| 方案 | 均匀/搜索 | Indexer 保留比例 | Long Avg | vs. Baseline（51.0） |
|------|---------|----------------|----------|---------------------|
| 训练感知均匀 | 均匀 | 1/2 | **51.6** | **+0.6**（超越基线！）|
| 训练感知均匀 | 均匀 | 1/4 | 50.6 | -0.4 |
| 去掉跨层损失 | 均匀 | 1/2 | 49.8 | -1.2 |

> **核心发现**：跨层蒸馏损失不仅是"容忍损失的补偿"，而是**实质性提升了 Indexer 质量**——1/2 保留率时甚至超越了全量 Indexer 基线（51.6 vs 51.0）。

---

### 端到端推理加速（30B 模型，200K 上下文，1/4 Indexer 保留）

| 阶段 | 原始耗时 | IndexCache 耗时 | 加速比 |
|------|---------|----------------|-------|
| Prefill | 19.5s | 10.7s | **1.82×** |
| Decode（单请求 tok/s） | 58 tok/s | 86 tok/s | **1.48×** |
| Decode（Full KV Cache tok/s） | 197 tok/s | 297 tok/s | **1.51×** |

---

### 在 744B GLM-5 上的规模化结果

为验证可扩展性，论文在 744B 参数的 GLM-5 模型上进行了实验：

| 方案 | 保留比例 | 性能 | vs. Baseline（78.4） |
|------|---------|------|---------------------|
| GLM-5 Baseline | 1/1 | 78.4 | — |
| IndexCache 训练感知 | 1/2 | **78.7** | **+0.3** |
| IndexCache 训练感知 | 1/4 | 78.0 | -0.4 |

端到端推理加速至少 **1.3×**（具体数字受限于当前 Prefill/Decode 计算比例）。


## 📊 实验与性能评价

### 基准测试配置

- **基础模型**：GLM-4.5-Air（30B）和 GLM-5（744B），均使用 DSA 稀疏注意力
- **评估集**：
  - LongBench v2：长文档理解与问答（Long Avg 综合指标）
  - RULER：合成长上下文评估基准
  - LiveCodeBench：代码生成
  - GPQA-Diamond：研究生级专业问答
  - AIME 2025：数学竞赛
  - IFBench：指令遵循
  - AA-LCR：长上下文检索

### 主要实验结果总结

**训练无关方案（30B / 200K）**：
- 1/2 Indexer 保留 + 贪心搜索：Long Avg **50.3**（Baseline 50.2），几乎零损失
- 1/4 Indexer 保留 + 贪心搜索：Long Avg **49.9**（vs. 均匀间隔 43.0），差距从 7.2 分缩至 0.3 分
- General 和 Reasoning 子任务均在 1 分以内

**训练感知方案（30B / 200K）**：
- 1/2 均匀保留：Long Avg **51.6**，超越全量 Baseline（51.0）**+0.6 分**
- 1/4 均匀保留：Long Avg **50.6**，仅差 0.4 分
- 跨层蒸馏损失消融：去掉后 1/2 保留方案 Long Avg 从 51.6 降到 49.8（差 1.8 分），证明跨层损失至关重要

**推理加速（30B / 200K，1/4 保留）**：
- Prefill：**1.82×**（19.5s → 10.7s）
- Decode per-request：**1.48×**（58 → 86 tok/s）
- Full KV Cache Decode：**1.51×**（197 → 297 tok/s）

**GLM-5 744B 规模化**：
- 1/2 保留：性能 **78.7**，微超 Baseline（78.4）
- 1/4 保留：性能 **78.0**，差 0.4 分
- 端到端至少 **1.3× 加速**

### 局限性（Limitations）

1. **Indexer 架构依赖**：IndexCache 专为 DSA 类 Indexer 设计，其他稀疏注意力方案（如静态模式稀疏、基于块的稀疏）若无 Indexer 结构则无法直接应用
2. **搜索成本**：训练无关的贪心搜索需要 N(N-1)/2 次前向传播，虽可并行化，但对于超大模型仍有一定开销
3. **上下文长度依赖性**：跨层索引相关性在较短上下文下（< 32K）是否依然稳定，论文未作系统验证
4. **解码阶段加速有限**：相比 Prefill 的 1.82×，Decode 的 1.48× 提升幅度较小，原因是解码阶段受限于 KV Cache 加载带宽，而非 Indexer 计算

---

## 💡 启发与我的想法

### 可借鉴之处

1. **冗余发现优先于方法设计**：论文的出发点是对"相邻层 Indexer 选择 70–100% 重叠"这一现象的精准测量，而不是先拍脑袋设计架构。这种"先定量测量冗余 → 再设计针对性压缩"的思路值得借鉴
2. **贪心搜索解决异质性**：面对层间敏感度不均匀的问题，贪心搜索（逐步将最不敏感的层转为 Shared 模式）是低成本且可解释的解决方案，比 NAS 或强化学习搜索轻量得多
3. **蒸馏损失设计的理论保证**：多层蒸馏损失的梯度等价性（Proposition 1）给了方法一个清晰的理论解释——Indexer 学的是"多层共识"，而非单层精确分布，从而天然支持跨层复用

### 改进空间

1. **动态复用比例**：不同输入的跨层相关性可能差异较大（如精确搜索类任务 vs 摘要类任务），可以引入在线估计机制，根据当前输入动态调整哪些层转为 Shared 模式
2. **非 DSA 场景推广**：IndexCache 当前针对 DSA 定制。对于其他带 Indexer 的稀疏注意力（如 SeerAttention、SampleAttention），类似的跨层复用框架是否适用值得探索
3. **与 SwiftKV 的融合**：SwiftKV 复用 KV 向量，IndexCache 复用索引，两者可能是正交的优化维度，融合后潜在加速比更大
4. **训练阶段应用**：训练时每层 Indexer 的 O(L²) 开销同样是瓶颈，IndexCache 思路是否能应用于训练加速有待验证

### 下一步 Action
- [ ] 阅读 HySparse（Gao et al., 2026）和 SeerAttention 系列，对比不同稀疏 Indexer 设计的跨层相关性
- [ ] 研究 SwiftKV（Qiao et al., 2025）的 KV 向量复用方案，思考与 IndexCache 的组合可能性
- [ ] 关注 GLM-5 / DeepSeek 后续版本中 IndexCache 是否被集成进生产推理栈


