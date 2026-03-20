---
title: "KV稀疏调研综述"
tags:
  - KV-Cache
  - LLM
  - Sparse-Attention
  - 算法优化
  - 推理加速
  - Attention
  - 调研综述
  - KV稀疏
---

# Motivation——为什么需要稀疏性

## 装不下—— KVCache导致显存容量瓶颈

以 LLaMA-3-70B 为例，128K 上下文的 KV Cache 约占 **40GB**，一张 A100-80G 放下模型权重后几乎无余量

## 搬不动——Decode阶段访存带宽瓶颈

Decode 阶段每生成一个 token，需要加载全部 KV Cache 到Shared Mem。这一步是 **memory-bandwidth bound**：GPU 算力远超数据搬运速度。

**减少 **KV** Cache 大小 = 减少每步需要搬运的数据量 = 降低 per-token latency = 提高 decode 吞吐。**

## 算不完——注意力计算瓶颈

注意力**计算复杂度为 O(n²)**。128K token 意味着 128K × 128K 的注意力矩阵。

TTFT（Time To First Token）**随上下文长度二次增长**，直接影响用户体验。

---

为了解决上述三种问题, 有两个流派的稀疏方法:

1. **丢弃KVCache:** 选择直接不存注意力分数较低的KVCache, 或者存全量KVCache, 但是加载的时候只加载最重要的解决问题1和2——对容量和带宽要求变小, 但是无法解决算得慢的问题
2. **丢弃计算:** 注意力分数不高的QKV matmul直接跳过不计算解决问题3——Prefill/Decode计算变快了
需要说明的是, **上面两个方法是正交的, 即可以在丢弃计算的同时保留KVCache **(比如Han Song团队的BLASST)

# Theory —— 为什么稀疏方法行得通

## 1. Softmax 的"赢家通吃"


- 解决问题1和2——对容量和带宽要求变小, 但是无法解决算得慢的问题
- 解决问题3——Prefill/Decode计算变快了
注意力公式中的 Softmax 将线性差距放大为指数差距。若两个原始分数相差 Δ，softmax 后的比值是 $e^\Delta$。

**直觉类比**：想象一场选举，规则是把票数做"指数爆炸"再分配：60票 vs 40票，按 Softmax 分配后，60票占99.9999%，40票方几乎归零。这就是"赢家通吃"。

## 2.自然语言本身就是稀疏的

**信息论视角**：英语文本约 70% 是冗余的（香农熵 1-1.5 bits/字符 vs 编码容量 4-5 bits）。在 LLM 处理的长文本中，这种冗余进一步放大——一篇 10 万字报告的核心论点可能只占 2000 字。

**长尾分布**：FlashPrefill 数据显示，**128K 上下文中前 10% 的注意力块承载了约 90% 的权重**——典型的幂律分布。文本越长，"废话"越多，模型自动跳过更多废话。

## 3.存在多种代理指标可以更低代价估计重要性。

**不需要完整的注意力矩阵**才能判断"谁重要"

- L2 Norm（Key 模长）, Q-Filters（QK 各向异性投影）, KeyDiff（Key 余弦差异）, LagKV（相邻 chunk KV 方差）, RetrievalAttention（向量检索先验）
## 4.KV 矩阵存在低秩/局部相关结构，

因此KVCache可以被压缩而不是被丢弃。

- DeepSeek MLA, Compactor（SVD 杠杆分数）, CurDKV（CUR 分解 K×V 联合杠杆分数)
# Trade-offs——稀疏方法中的设计抉择

## 重要性度量 —— 直接度量 vs 代理度量

知道"谁重要"是所有稀疏方法的前提。但**如何衡量重要性**，决定了方法的精度上限和工程可行性。

**直接使用 Attention Score**：最准确的信号——注意力分数直接反映了模型认为哪些 token 对当前生成最有价值。但 FlashAttention 通过 Tiling + Online Softmax 实现 IO 感知计算，**不显式物化中间 attention score 矩阵**。这就是为什么 H2O、TOVA、SnapKV 等高引方法全部基于 HuggingFace transformers 评测，与 vLLM/TRT-LLM 的生产推理管线结构性不兼容。

- 代表：H2O, TOVA, SnapKV, PyramidKV（全部 FlashAttention不兼容）
**使用代理信号（Proxy Score）**：绕过 attention score，用 Key/Value 的几何属性间接估计重要性。天然兼容 FlashAttention（只需要访问 K/V tensor，不需要注意力矩阵），推理时只需一个轻量外挂模块即可完成评估。但代理信号与真实注意力之间存在系统性偏差：

- 代表： L2 Norm, Q-Filters, KeyDiff, Compactor/CurDKV
## 丢弃的粒度 —— 粗粒度 vs 细粒度

稀疏化的粒度决定了"**每次丢弃操作的最小单位是什么**"。

**细粒度（Token-level）**：逐 token 评估重要性，精度最高，但**产生不规则的稀疏访存模式**。GPU 的 SIMT 架构天生偏好连续内存访问，细粒度的 scatter/gather 操作会**严重降低 FlashAttention kernel 的效率**，同时与 PagedAttention 的 page 粒度管理产生结构性冲突。

- 代表：H2O, TOVA, SnapKV, L2 Norm, LagKV, KVzip, KVzap
**粗粒度（Block/Chunk-level）**：**以 Block（如 128 tokens）为单位整体保留或丢弃，**GPU** 友好**，可直接嵌入 FlashAttention 的 Tiling 流程。但当 **Block 内部语义**异质性**高时（代码 **`if (user_id == 1024)`** 中各 token 功能截然不同），整块丢弃会误杀关键 token**，整块保留又浪费预算。

- 代表：BLASST, FlashPrefill, DistilledSparse
## 丢弃的时机 —— 早丢 vs 晚丢

**早丢（Post-Prefill 一次性驱逐）**：**Prefill 结束后立即压缩 **KV** Cache**，后续 Decode 全程享受压缩后的小 Cache——**省存储 + 省带宽**，收益最大化。但信息一旦丢弃不可恢复，对多轮对话致命：第 1 轮被判为不重要而丢弃的 token，可能在第 5 轮变成关键信息（SCBench 实证的 Distribution Shift）。最适合"长输入短输出"的 Prefill-Only 场景（RAG 问答、摘要）。

- 代表：SnapKV, PyramidKV, KVzip, Compactor, CurDKV, Q-Filters


**晚丢（Pre-Decode 按需检索）**：**全量 **KV** Cache 始终保留**（GPU 显存或 CPU RAM 中），每步 Decode 时只选择性加载最相关的子集参与注意力计算——**不省存储，只省 Decode 加载带宽**。因为全量 KV 始终存在，天然支持多轮对话中注意力重心的动态漂移，不存在信息不可恢复的问题。代价是需要维护检索/评估索引。

- 代表：Quest（全量 KV 在 GPU，per-page min/max 上界估计选 Top-K Page，7.03x 注意力加速）, RetrievalAttention（全量 KV 卸载到 CPU，ANNS 向量检索取回 1-2%）, DistilledSparse（LoRA 蒸馏桥接分布偏移）
**不丢 **KV**，少算注意力（**Sparse** Attention）**：产生并保留全量 KV Cache，但在注意力计算过程中跳过不重要的 block。BLASST 在 FA kernel 内部跳过低分 block 的 Softmax + V 加载 + Matmul（全量 K 仍然参与 QK^T 计算以判断分数），**省的是注意力计算和 V 的加载带宽，不省 KV 存储**。FlashPrefill 在 Prefill 阶段用探针采样先粗估 block 重要性，仅对高分 block 执行精确注意力。

- 代表：BLASST（Prefill + Decode 双阶段，全量 KV），FlashPrefill（仅 Prefill，探针采样）
## 是否需要预计算 —— 看清楚再丢 vs 边算边丢

**要判断"哪些 token 重要"，理论上需要先完整计算一遍注意力矩阵**——但如果你已经算完了，省计算的意义就没了。这就是稀疏方法的核心悖论。

**需要预计算 / Warmup**：先跑一遍（完整或近似的）注意力计算，得到分数后再决定丢弃。看得最清楚，决策最准确，但引入额外 overhead。这类方法只在 Prefill 阶段有意义——因为 Prefill 本来就要算一遍，overhead 相对可控；但对短输入来说，预计算的固定开销可能吃掉全部收益。

- **完整预计算**：KVzip（额外一次前向传播 ≈ 2x Prefill 开销，scoring 本身比推理更贵）, CurDKV（CUR 分解 +40-50% Prefill 开销）
- **轻量预计算**：FlashPrefill（均匀采样探针 Q × 池化 K，开销比全量注意力低两个数量级）, SnapKV（观察窗口内投票）
- **离线预计算**：DuoAttention（离线训练 Head 分类器）, Q-Filters（离线 SVD 提取 Q-Filter 向量，<3 分钟）, KVzap（离线训练代理模型）

**不需要预计算（In-flight 决策）**：在 FlashAttention kernel 的正常计算流程中，利用已有的中间结果实时决定跳过或保留。零额外 overhead，但决策信息不完整——只能看到"局部"而非全局，可能漏掉远距离的关键依赖。

- 代表：BLASST（在 Online Softmax 过程中，用当前块的局部最大值与全局运行最大值之差决定是否跳过——决策发生在计算正在进行的过程中，但代价是 Warp 内 32 线程必须一致投票才能跳过，保守性限制了实际稀疏度利用率）
- 代表：StreamingLLM（固定规则，不需要任何计算）
**使用代理信号（Proxy）**：完全不计算注意力分数，而是用 Key/Value 向量本身的几何属性来估计重要性。开销极低且天然兼容 FA（不需要显式 attention score），但代理信号与真实注意力分数之间的误差是系统性的——在特定模型架构（如 QK-Norm）上可能失效。

- 代表：L2 Norm（Key 模长）, KeyDiff（Key 余弦相似度）, LagKV（相邻 chunk KV 方差）, Compactor（SVD 杠杆分数）
# Pitfalls —— 何时稀疏化会失败

## **不可压缩任务**

- 原因: 每个 token 同等重要，无冗余 - 违法法则2 - 文本“没有重点”, LLM不知道丢弃哪些注意力
- 例子: 海量随机 JSON 键值对提取、精确密码记忆
大海捞针（Needle-in-the-Haystack）基准测试将关键信息（“针”）嵌入到重复的噪声（“干草堆”）中，**由于噪声具有高度可压缩性，这使得压缩方法也能达到合理的准确率。但是在必须记住全局信息，或者任意局部信息同等重要（比如超大的**json** kv-pair）的检索任务中性能糟糕。**

## Block 局部相似性不成立

- 原因: 结构化数据（代码/JSON）中相邻 token 语义截然不同, 粗粒度(非tokenwise)KV压缩直接失效
- 例子: `if (user_id == 1024)` 中各 token 功能完全不同, 不能全部丢弃
## 多轮对话注意力漂移

- 原因: KVCache被一次性驱逐无法恢复后续需要的信息
- 例子: 第1轮被丢弃的 token 在第5轮变成关键信息

随着推理的进行与用户对话轮次的增多，每一次推理，LLM对于同一段文本的注意力重心都会有变化。**这意味着基于****<mark style="background: yellow">单次</mark>****Attention Score的KVCache Compression方法都是先天不足的**：

第一轮对话中重要的 Token，到第五轮时可能已经不再重要；而之前被由于权重低而被忽略的 Token，现在可能变成了新的“离群值”（重要节点）。**原本在初始阶段被认为不重要的信息，在后续生成中可能变得至关重要。**

然而，基于Attention score进行KVCache驱逐的压缩算法，**只能看到一个**时间片**内各个token的注意力得分**，如果以此进行KV丢弃，那么未来生成中就永远丢失了这部分信息，造成不可避免的精度问题

## 同一Context, Query不同

KV压缩算法必须能够感知用户的query并以此做出调整，否则在多轮对话下，随着注意力漂移，query内容改变，压缩算法的精度下降严重


比如上图SnapKV， 19.0/9.7的得分。**19.0是在给定长文本后，明确给出**query**（比如“帮我总结这篇文章”）后进行压缩得到的分数**。而9.7是**完全不给query，只给长文本的情况下**，压缩算法获得的分数

## KVCache关键路径带宽不再紧张

- 原因: KV 压缩/丢弃的"减少访存"收益不再明显, **反而因为压缩引入额外开销**(但是计算级别的稀疏化还是有效的)
- 例子: MLA一条KV仅70KB, TP=4+, 旗舰卡HBM vs GDDR等情况分担了KVCache带宽压力

LLM 解码阶段通常是“访存受限”（Memory-bound）的，即 GPU 计算很快，但把 KV Cache 从显存搬运到计算核心的速度太慢。

当使用Tensor Parallelism时，**每一块 **GPU** 负责的 **KV** Cache 变少了，加载KV的带宽也翻倍(**HBM**芯片多了), 增加了可用的**总带宽，缓解了带宽压力。

**但是，**KV** 压缩通过减少数据量来缓解带宽压力。**

**结论**：**如果 **TP** 已经把带宽压力解决了，那么 **KV** 压缩带来的“减少访存”收益就不再明显**，反而因为压缩算法本身引入的额外计算开销（如量化/反量化、重要性采样计算），**导致整体速度可能还不如不压缩。**

## 过度压缩导致模型变得啰嗦


如果压缩的太狠, 模型会丢失重要的注意力.

在Rethinking Key-Value Cache Compression Techniques这篇论文中, 作者测试超过 20% 的样本在压缩后**回复长度增加了至少 1.5 倍。**此前的吞吐量分析显示，在许多场景下，**压缩方法无法实现超过 1.5 倍的解码吞吐量提升。**这意味着，由于回复长度显著增加，这些样本的**端到端延迟反而会不降反升。**

这种变长的回复是 KV 缓存压缩**隐式补偿准确率损失**的一种方式。

# Explore ——  RL中的稀疏注意力

在强化学习（RL）流程中，经验采样（Rollout）阶段的自回归生成会消耗大量的显存和时间。尤其在长上下文场景下，KV Cache 的存储需求呈线性或二次方增长。引入稀疏注意力，可以大幅压缩推理时的 KV Cache 占用，提高并发吞吐量，从而加速 RL 的数据收集过程。

尽管稀疏化在推理时极大地释放了系统资源，但这可能会与标准的训练（Backward/Update）流程产生严重的结构性冲突。在标准的 RL 训练中，模型需要计算生成动作的对数概率（Log-probs）并进行反向传播。如果推理阶段（采样数据）使用了稀疏注意力，而训练阶段（计算梯度更新参数）依然采用标准的稠密注意力（Dense Attention），就会导致**前向计算图与反向计算图的彻底错位**。

这种“训推不一致”会直接破坏反向传播的数学严谨性：

- **“幽灵梯度”问题**：在稀疏推理时，某些 Token 的 KV 值被主动丢弃或屏蔽，它们对后续 Token 的生成没有产生任何实际影响。但在使用稠密模型进行反向传播时，梯度依然会流向这些理论上被“隔离”的 Token，导致无意义甚至错误的参数更新。（Recomputation可解）
- **策略分布的剧烈偏移**：用于评估优势函数（Advantage）的概率分布，与模型实际生成数据时的概率分布不再匹配，方差爆炸或者全被clip。这会破坏 PPO 等算法中重要性采样（Importance Sampling）的根基，导致算法极难收敛
- **奖励欺骗**： KL 散度惩罚项是用于限制当前训练策略不能偏离初始策略太远，由于每次生成都有偏差，序列长度 T越长， KL 散度惩罚项累加的和就越大，最终彻底淹没模型原本应该得到的真实奖励。于是模型发现最优策略（Optimal Policy）并不是去解答问题，而是在生成的第一个或前几个 Token 就立刻输出 `<EOS>`，从而规避较大的KL 散度惩罚.
  
基于上述问题，如果想引入稀疏注意力通常只能在两条路径中二选一：

## **强制训推严格对齐**

要求训练反向传播时，必须采用与推理完全一致的稀疏策略和丢弃策略。代价是工程实现难度极高。需要开发不仅支持高效前向推理，同时还要支持高效反向传播的定制化稀疏算子（如基于 Triton 或 CUDA 重写 FlashAttention）。此外，强行在训练端引入稀疏，可能会不可逆地损害大模型的长期记忆和泛化能力。

前面介绍的多种稀疏方案中，效果相对来说比较好的有两个：BLASST和KV Zap。如果按照比较保守的要求训推一致对齐的话，需要面


|稀疏方案|原理|效果分析|训推是否一致|
|---|---|---|---|
|BLASST|最开始的注意力得分如果较低，那么softmax以后在最后的贡献就很小。于是在online softmax中动态丢弃实际得分较低的块，最终依然能得到较好的最终结果|需要全量存kv，并且需要算score，只是节省了部分softmax和value的计算 <br> 算是查询感知（Query-Aware）的算法，可以用于超长上下文和多轮对话|理论上可以改成一致，只要把训练模型中的FA也改掉就行 <br> 但实际上受限于kernel实现的方法（比如得分的block，以及warp的同一行为要求），一旦切分方式变了，padding变了等等原因，都会导致稀疏的执行产生差异|
|KV Zap|基于自然语言特征训练一个模型，对每个token判断其包含的信息密度高不高，对于维持整段话的语义完备性重不重要|查询无关（Query-Agnostic）算法，可以压缩自然语言中的冗余。对于部分自然语言无关的任务表现较差。<br>不用存全量kv，效率相对较高。但永久丢弃kv cache后，多轮对话效果存疑。|训推相对一致，kv zap是一个外挂，运行时丢弃kv cache|

### 衍生想法
想法1：BLASST能不能结合KV Zap，用一个阈值来决定是否在online softmax中丢弃，从而可以让训推在不同切分不同padding的情况下都保持一致？

想法2：是否最好不要把稀疏注意力直接引入后训练。预训练是稠密网络，即便某个 Token 当前被认为不那么重要，在反向传播时，它依然能接收到一丝微弱的梯度信号。如果事实证明这个 Token 其实很重要，这丝微弱的梯度就会在后续迭代中把它唤醒，慢慢调高它的权重。如果在 RL 训练时突然切换到全局稀疏，相当于强行用剪断了 90% 的线，只保留最粗的 10%。可能会使得反向传播的梯度变得离散。如果与基座模型的训练方法不一致，可能会破坏在预训练时建立的平滑参数空间。

## **额外工作进行补偿**

接受不一致，引入，允许训推不一致，但在系统和算法层面加入复杂的修正机制。代价是需要引入大量额外开销：

- 在计算梯度前，必须用稠密模型把采样出的轨迹重新进行一次前向计算以对齐 Log-probs；
- 在 RL 算法层引入复杂的 Off-policy 修正项（如动态截断、额外的 KL 惩罚重加权）
- 利用 LoRA** 作为极低成本的载体**，通过**在线蒸馏**在数学分布上接近 Dense 模型的形状
- 以上的目标都是为了迎合各种PO之类的算法，让重要性采样比率不要方差过大被clip，或者kl散度非常大让奖励被扭曲
KV Zap

基于自然语言特征训练一个模型，对每个token判断其包含的信息密度高不高，对于维持整段话的语义完备性重不重要

- 查询无关（Query-Agnostic）算法，可以压缩自然语言中的冗余。对于部分自然语言无关的任务表现较差。
- 不用存全量kv，效率相对较高。但永久丢弃kv cache后，多轮对话效果存疑
- 训推相对一致，kv zap是一个外挂，运行时丢弃kv cache
## 稀疏对强化学习的帮助

稀疏并不仅仅是孤立地加速了rollout时间，可以利用稀疏带来的高性能生成更多的轨迹，专门挑出那些不仅真实奖励分数高，而且其概率分布距离 Dense Policy 非常近的轨迹送去计算梯度。从而可以帮助模型只学习价值更高的轨迹，这也是一种提高算力利用率，加速模型迭代的好办法。

# Papers

## BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

作者: MIT-HAN-LAB 韩松团队

[https://arxiv.org/pdf/2512.12087](https://arxiv.org/pdf/2512.12087)

### Summary

本文引入了**与FlashAttention无缝集成的，几乎零开销的在线动态稀疏注意力计算方法。**作者发现如果注意力分数较低->Softmax归一化后对最终的结果贡献接近于0, 于是在FlashAttention Kernel内部，结合Tilling和Online Softmax策略做了动态的优化：如果当前小分块的注意力分数小于当前注意力分数的最大值，那么直接丢弃这个块。同时避免了对该分块的softmax计算，Value矩阵加载和后续的Matmul操作。作者同时引入了可自动调节的自适应压缩率，可以在任意模型和上下文长度下达到最优的压缩率。**由于压缩率在运行时已知，因此该方法的加速比例是可以精确预测的，与任务类型，上下文无关。**实验结论为**在74.2%的稀疏度下，Prefill阶段的**加速比**可达1.62x。 同时该kernel也可用于后训练，让模型更加适应稀疏方法。**


> 🚀 Insight1. Kernel级别的优化可用于后训练：让模型学会用稀疏的方式思考

稀疏感知训练（Sparsity-aware training）可以进一步拓展模型的准确性-稀疏性边界。

由于 BLASST 是直接在底层前向传播内核中屏蔽了无关块的计算，在微调的后向传播中，这些块自然不会获得梯度更新 。这种训练方式无需修改模型架构或引入辅助损失函数 ，就能**促使模型主动将关键信息集中到高分数的注意力块中，从而适应更高程度的推理**稀疏性 。

> 🚀 Insight2. 稀疏方法在推理任务上可以获得小幅精度提升


**在某些情况下，把不重要的计算删掉，模型反而变得更聪明了。**

在高度冗余的长上下文任务中，动态丢弃低注意力分数的块不仅能减少计算和访存开销，还在一定程度上过滤了不相关的上下文干扰（Distractions） 。

实验表明，在约 50% 的稀疏度下，**模型在某些复杂数学和推理**基准测试**中的表现甚至略微优于全量注意力**，这表明**适度的**稀疏性**可以充当隐式的降噪机制**。

> 🚀 缺陷: 无“预计算”: 运行时流水线空泡, Warp Divergence

在一整个Warp中, 如果只有一部分线程选择跳过计算, 这一部分线程就要“陪跑”——Warp的调度规则是32线程锁步的. 因此即使这些线程跳过了Matmul等计算, 指令流水线和控制流一样需要运行, 造成空泡.

作者也意识到了这一点, 因此设计了VOTE机制, 使用VOTE指令, **只有当一个Warp内的32个线程一致同意跳过计算时, 才跳过. **这种机制从设计上有效规避了运行时昂贵计算部分的 Warp Divergence, 但是代价是**保守性**：为了避免分化并保证正确性，这种机制是“保守”的。即便某个线程本来可以跳过，但如果同 Warp 内的其他线程需要处理该块，该线程也必须陪同执行计算。这限制了稀疏度的利用: **但凡有一个线程需要计算, 另外的31个线程就要陪跑.**

这也解释了为什么 BLASST 在 50% 的稀疏度下，**Prefill 阶段只能获得约 1.24 倍的加速，而不是理想的 2 倍。**这是它为了实现“无需预计算（no pre-computation）”所付出的代价 。

## SPARSE ATTENTION FOR EFFICIENT LLM REINFORCEMENT LEARNING

作者: 陈贝迪 [InfiniAI Lab @ CMU](https://keroro824.github.io/lab-page/)**也是DuoAttention的作者**
[paper](https://openreview.net/pdf?id=vMrhhCAgCA)

### Summary

本文研究的是一个此前几乎无人触及的问题：**将稀疏注意力应用于 **RL** 训练的rollout（轨迹采样）阶段时，会导致训练崩溃。**核心原因是稀疏注意力引入的 likelihood 近似偏差（approximation bias）会随 autoregressive 生成长度累积放大，**导致 actor-policy（稀疏采样策略）和 dense policy（全量注意力训练策略）之间的分布偏移**（distribution mismatch）持续扩大，最终引发训练不稳定甚至崩溃。

作者提出 DISTILL SPARSE 框架：

1. 用轻量级**LoRA**在线蒸馏（on-policy distillation），将 dense policy 的知识蒸馏到 sparse sampler 中，弥合分布差距；
2. 对长生成高稀疏度场景，**过采样（oversampling）多条轨迹并用 reward-aware filtering 筛选高质量样本。**在 NVIDIA H200 上实现 1.72x rollout 加速，下游数学推理任务表现与 dense baseline 持平。
> 🚀 Insight1. 稀疏注意力的偏差会随生成长度累积放大——推理时无害，训练时致命

在推理（inference）时，稀疏注意力的近似误差只影响当前 token的生成质量，每一步的误差是独立的。

但在 RL 训练中，importance ratio  需要对整条轨迹的 token-level probability 求积/求和。**稀疏注意力对每个 token的概率估计都有偏差，这些偏差在序列维度上乘性累积（因为概率是连乘的），导致：**

- sparse rollout 和 dense policy 之间的 KL 散度持续增长并最终爆炸
- 模型生成的回复长度持续缩短直到崩溃——**为什么生成长度会缩短? 和上面讨论的“稀疏使得模型变得啰嗦”有何不同?**


> 1️⃣ 第 1 轮:
>     生成长度 1000 tokens
>     前 100 个 token 的 ratio 偏差较小（误差还没累积）
>     后 900 个 token 的 ratio 偏差较大（误差累积了）
>     → 后半段的梯度信号是噪声
>     → 模型从后半段学到的东西基本是垃圾

> 2️⃣ 第 1 轮:

> 3️⃣ 第 5 轮:
>     生成长度缩短到 500 tokens
>     这是模型的"理性"自适应：短回复 = 更少的误差累积 = 更稳定的训练信号

> 4️⃣ 第 20 轮:
>     生成长度缩短到 50 tokens
>     模型已经无法完成任何需要长推理链的任务
>     → 所有数学题都答错
>     → reward 全部为 0
>     → advantage 全部为 0（组内无差异）
>     → 学不到任何东西
>     → 训练事实上停止

**可以对阅读偷懒，绝不能对思考偷懒** :

- FlashPrefill 等方法是**在Prefill 阶段省计算，每个 token 的误差不累积**；
- 而 DISTILLSPARSE 要解决的是 Decode阶段的稀疏注意力导致的累积误差问题。**稀疏注意力在推理和训练中的影响机制是根本不同的。**

> 🚀 Insight2. LoRA 蒸馏的精妙设计：zero extra trajectory cost

DISTILL SPARSE 的 LoRA 蒸馏并没有额外生成 dense 轨迹（那样会失去稀疏加速的意义）。它的做法是：

1. 用 sparse policy（base model + LoRA）生成 rollout 轨迹
2. 用 dense model 重新计算这些已有轨迹的 log-probabilities（这一步本来就要做，因为 PPO 需要参考策略的
logprobs）

1. 把 dense logprobs 和 sparse logprobs 的差异作为 KL 蒸馏损失，只更新 LoRA 参数
关键洞察是：**dense forward pass 的计算在 **PPO** 流程中本来就是必须的**（用于计算 importance ratio），因此蒸馏几乎是"免费附赠"的。唯一的额外开销来自 sparse policy 的一次额外 forward pass 和 LoRA 的backward pass，但 LoRA 的参数量远小于全模型，开销极低。**整个系统的额外开销仅约增加 dense policy update 的55%**（Fig. 7），换来的是训练稳定性的质变。


> 🚀 Insight3. 过采样+筛选：稀疏注意力的"缺陷"反而成了free lunch

这是本文最反直觉的洞察。由于稀疏注意力降低了每个 token 的计算成本，在相同的 GPU时间预算下，可以生成更多的候选轨迹（比如原来生成 n=8 条，现在可以生成 M=16 或 24 条）。然后用 reward进行筛选，**只保留得分最高的 n 条用于训练。(Oversampling)**

Fig. 4（Middle）展示了一个关键规律：**随着过采样倍数 N 增加，Oracle Top-8 的平均 reward 持续上升，但所有 N**

**条轨迹的平均 reward 基本不变。**

——这意味着高 reward 的轨迹天然更接近 dense policy 的分布——quality 和distribution alignment 是正相关的。


更重要的是，过采样的计算开销几乎为零：因为稀疏 rollout 本身就比 dense rollout便宜，多生成几条轨迹的额外成本被稀疏加速完全覆盖。说人话就是：**稀疏注意力虽然让每条轨迹质量下降了，但它让你能"多试几次"，而且"多试几次然后挑好的"比"只试一次但每次都用全量计算"效果更好。**

> 🚀 Insight4. Rollout 阶段是 RL 训练的绝对瓶颈，稀疏注意力可以释放 long-CoT 的 scaling 潜力

在典型的 RL 训练流程中，**rollout（轨迹生成）占据了超过 90%的端到端训练时间。**而 rollout 的核心瓶颈是 KV-cache 加载，其内存开销随生成长度二次增长（O(L_out²)）。

**稀疏注意力将这个瓶颈从 O(L_out²) 降低到 O(K·L_out)**，其中 K 是稀疏 KV 预算。这意味着在 long chain-of-thought（CoT）推理场景下（比如生成 16K 甚至 32K tokens的数学推导），稀疏注意力的加速效果会更加显著。

作者在 Appendix B 中明确指出：

> sparse attention unlocks superior inference scaling in the usual long CoT regime

如果不使用稀疏注意力，RL 训练的计算成本会随 CoT 长度的增加而快速变得不可承受。

# Scenarios——稀疏性方法的适用场景

| ISL/OSL分类 | Context共享 | 典型Case | 适合稀疏? | 原因 |
|---|---|---|---|---|
| Long Prefill Short Decode | 单轮对话 | 摘要 / 单轮QA / 信息提取 | ✅ | Prefill 阶段 KV 巨大且自然语言高度冗余 |
| Long Prefill Short Decode | 多轮对话 | RAG / 企业知识库 | ✅ | 一次 Prefill 多用户复用，KV 压缩的收益被多次查询摊销。 |
| Long Prefill Long Decode | 单轮对话 | 长文翻译 / 报告改写 | ⚠️ | Prefill 和 Decode 双重瓶颈, 大多数方法只优化 Prefill。需要双阶段方法（BLASST/KVzap） |
| Long Prefill Long Decode | 多轮对话 | 企业共享代码仓库, 多轮提问 | ❌ | Prefill 和 Decode 双重瓶颈, 大多数方法只优化 Prefill。需要双阶段方法（BLASST/KVzap） |
| Short Prefill Long Decode | 单轮对话 | CoT推理 / RL Rollout | ⚠️ | Prefill 阶段 KV 小无需压缩，瓶颈在 Decode 阶段持续增长的 KV. 需要支持Decode 的稀疏方法 |
| Short Prefill Long Decode | 多轮对话 | Chatbot / Agentic Coding | ❌ | 注意力分布跨轮次漂移，一次性驱逐不可恢复。仅全量 KV 保留的方法可用 |
| Short Prefill Short Decode | 单轮对话 | 分类 / 简单问答 | ❌ | KV 足够小，稀疏化的固定 overhead 超过收益。 |
| Short Prefill Short Decode | 多轮对话 | 闲聊 / 指令跟随 | ❌ | KV 足够小，稀疏化的固定 overhead 超过收益。 |

# Key Takeways:

## 稀疏化的天花板远未触及

Softmax 的指数非线性制造"赢家通吃"——128K 上下文中前 10% 的注意力块承载约 90% 的权重，其余 90% 的计算贡献趋近于零。自然语言本身约 70% 是冗余的，KV 矩阵也存在可利用的低秩结构。这意味着：**绝大多数 KV Cache 在统计意义上"可以丢弃而不改变结果"，稀疏化方法不过是将这个已经存在的事实工程化地加以利用。** 问题不在"能不能做"，而在"如何做对"。

## 方法正在从"看注意力分数"走向"看几何结构"

早期方法（H2O、SnapKV）依赖显式 attention score 判断重要性，**但 FlashAttention 不暴露中间分数**，导致这批高引方法与生产推理内核结构性不兼容。2025 年的新方法（L2 Norm、Q-Filters、KeyDiff、LagKV、Compactor、CurDKV）转向 Key/Value 向量的几何属性（模长、投影、相似度、杠杆分数）作为代理信号，天然兼容 FlashAttention。同时，FlashPrefill 和 BLASST 证明了稀疏化可以直接嵌入 Kernel 内部而非作为外挂模块。**从"应用层外挂"到"Kernel 层原生"，从"依赖注意力分数"到"几何代理"，是这个领域的技术演进主线。**

## 可以对"阅读"偷懒，绝不能对"思考"偷懒

SCBench 和 DistilledSparse 共同揭示了一条铁律：**Prefill 阶段（阅读）可以大胆稀疏化，Decode 阶段（推理/生成）必须极其谨慎。** 人类语言的冗余性保证了"跳着读"仍能抓住重点；但 Decode 是精密的逻辑推导，每步误差会以概率连乘的形式累积放大。DistilledSparse 记录了极端案例：RL rollout 中稀疏注意力导致生成长度在 20 轮内从 1000 崩溃到 50 tokens。这解释了为什么 FlashPrefill 在 256K 上下文能做到 27.78x 加速且几乎无损，而 Decode 阶段的稀疏化至今没有令人满意的方案。

## 现有方法存在系统性的评测盲区

**(1) 训推不一体**，仅 2 篇关注训练场景；

**(2) 只关注长上下文**，短上下文没有收益甚至负收益，但没有论文讨论按场景动态分发；

**(3) 只管 Prefill 不管 Decode**，长序列生成（CoT/Rollout）场景被系统性回避；

**(4) 只关注一问一答**，多轮对话和 Distribution Shift 仅 2 篇涉及；

**(5) 只评测小模型**，无一方法在 >100B 模型上验证，而 TP 并行会大幅削弱压缩收益；

**(6) 没有关注batch_size>1的**批处理**情况, **没有在真实服务并发压力下进行测试

**(7) 没有向 vLLM/SGLang 等生产框架适配**，大部分论文使用的框架是 torch后端的transformers，性能较差

这些盲区不是个别论文的疏忽，而是整个领域的结构性问题——学术评测与生产需求之间存在巨大鸿沟。

