---
title: "Attention-FFN Disaggregation"
tags:
  - LLM推理
  - MoE
  - AFD
  - 专家并行
  - 系统优化
  - 异构部署
---

## 论文内容概述

无论是 Dense 模型还是 MoE 的大模型，在长序列推理场景下，往往是 decode 阶段的 GPU 占用时间更高，从而导致 decode 的成本一般会明显高于 Prefill。这里的主要原因在于，传统 Attention 算法如 MHA、GQA 等实现，KV Cache 往往比较大，特别是在长文本的情况下。导致 decode 时过早出现显存容量或显存带宽的瓶颈，无法积攒更大的 batch，从而将过长的 GPU 占用时间转变为实际的 decode 吞吐。

**AFD** 的全称是 **Attention-FFN/Expert Disaggregation**（字节 & 阶跃），指将 Transformer 结构中的 Attention 与 FFN 分离成两个部分进行独立部署，它的主要目标就是优化 Decode 阶段的效率与成本。之所以要分离 A 与 F，是因为：一方面 A 与 F 有不同的计算与存储需求，另一方面不同 GPU 在算力与存储的参数和成本上也有所差异，它们之间可以进行更精准的匹配。

下表分析了当前主要模型在 A/F 两侧的资源需求对比：

| 模型 | hidden\_size | 总参数量 | 激活参数量 | Attention 方案 | A 计算强度 | F 计算强度 | KV Cache / token | MoE | 显存/expert |
|------|-------------|---------|-----------|---------------|-----------|-----------|-----------------|-----|-----------|
| Qwen3-235B | 4096 | 235B | 22B | GQA（Head=64, G=16） | 32 | $2 \times B_{\text{exp}}$ | 188 KB | 8/128 | 1.7 GB |
| DeepSeekV3 | 7168 | 671B | 37B | MLA（Head=128, $D_c$=512） | 512 | — | 34.3 KB | 8/256 | 2.5 GB |
| Step3 | 7168 | 316B | 38B | MFA（Head=64, D=256） | 128 | — | 24.6 KB | 3/48 | 6.2 GB |

针对 A/F 的差异化特征，两篇论文都着重分析了不同 GPU 的适配情况，并论述了自身是如何应用和优化的。

字节测算 H20 的显存成本最低，而 L40S 的计算成本最低，因此异构的 A 部署在 H20，F 部署在 L40S，单位吞吐成本最低。

![字节GPU成本分析](/myblog/assets/AFD/image1.png)

阶跃的测算中同样 H20 显存成本最低，H800 是所列 GPU 中计算成本最低的。

![阶跃GPU成本分析](/myblog/assets/AFD/image2.png)

阶跃进一步细化了 A 和 F 的成本测算，随着序列变长，A 的成本近似线性增加，但 F 成本不变。

![A与F成本随序列长度变化](/myblog/assets/AFD/image3.png)

根据上述表格，选取最优硬件合并后，理论上 AFD 具有 cost 优势。

![AFD成本优势](/myblog/assets/AFD/image4.png)

阶跃的模型 Step3 可以用较少的卡（AFD），在同样 SLA 的要求下实现更高的吞吐，虽然激活参数量更高（38B），但毕竟模型结构不同，Attention 算法不同，并不是完全的同类对比。

![Step3 AFD吞吐对比](/myblog/assets/AFD/image5.png)

由于 MLA 计算强度的问题，DSv3 在许多其他 GPU 上的部署效率不够理想，阶跃定制实现了 MFA 降低计算强度。

![MFA计算强度对比](/myblog/assets/AFD/image6.png)

![MFA详情](/myblog/assets/AFD/image7.png)

---

## 需要解决的工程问题

### 多级流水设计

在传统的流水线并行中，双批次重叠（Two-batch Overlap, TBO）通常足以掩盖部分通信成本。但在 AFD 架构下，TBO 还不够。因为在同等层数与严苛的 SLO 约束假设下，A-role 和 F-role 分布在两批卡上，A 和 F 自身不能重叠。假设 A 和 F 逐层延迟预算是大致相当的（这需要配平，不过相对容易配平）。当一个微批次从 Attention 实例输出后，它必须经历分发调度（dispatch），这个 dispatch 延迟意味着无论如何 A 和 F 之间的时间不能完美重叠，因此至少需要三或四阶段流水线才能掩盖传输延迟。

好处是，可以通过分别 scale A 和 F 做出接近完美的流水线。尤其 F 是完全无状态，可以微服务化，动态扩缩容。

![阶跃三阶段流水线设计](/myblog/assets/AFD/image8.png)

![字节四阶段流水线设计](/myblog/assets/AFD/image9.png)

与之相比，DeepSeek 采取的大 EP 不分离方案中，实际上只需要两个 micro batch 就能实现比较好的掩盖。

![大EP两阶段重叠对比](/myblog/assets/AFD/image10.png)

### 容易放大负载不平衡

在大语言模型的实际生产环境中，工作负载的动态波动是不可避免的物理法则。不同用户请求的上下文长度差异、以及 MoE 模型内部固有的动态专家路由机制，都会在集群内引发持续的负载微小抖动。AFD 架构最大的隐患在于其物理隔离的特性阻止了系统内部处理这些波动的"弹性缓冲池"，从而引发了比传统 EP 严重得多的"不平衡惩罚"。

AFD 应对算力波动的唯一手段是调节 Attention 或者 FFN 节点的物理规模，相较在集群内部平滑吸收，这是一种极其粗放的节点级扩展，效率并不够好。

但是如果 AFD 可以显著降低部署规模（同时保证 SLO 和吞吐的前提下），相比大 EP 更大规模下的不均衡性，其实是有改善的。

---

## 分离的收益和应用场景

### MoE 与 AFD 的关系

MoE 既推动了 AFD，也制约了 AFD。本质上来说，当模型大到某个程度（包括 Dense 模型），单机显存容量或者算力和带宽不足以支撑足够大的 batch 并保住 TPOT SLA 要求（单机推理能打的 batch size 达不到 FFN 进入高 MFU 区），就有需要做 AFD，因为拆开后把 FFN batch size 攒上去把 MFU 打高，就有收益。

一方面，MoE 模型的 F 非常大，放大了这部分的影响。另一方面，MoE 需要做 All-to-All 通信，非常考验通信带宽，A/F 分离错配有可能会加剧通信压力。当触及通信瓶颈时，分离 A/F 也无法提升 FFN 的效率。

### 模型稀疏度影响

给定 SLA，假设采用三阶段流水线，那么对于每个阶段其总时间需求为：

$$t_{\text{stage}} = \frac{\text{TPOT}}{3} = \frac{50\text{ms}}{3} \approx 16.6\text{ms}$$

**FFN 阶段计算强度：**

$$I = \frac{2 \times B_e \times W_e}{W_e} = 2 \times B_e$$

为保证 FFN 效率，算术强度 $I$ 必须大于等于硬件的 roofline：

$$2 \times B_e \geq \frac{\text{FLOPs\_HW}}{\text{Bandwidth\_HW}}$$

换算得到单个专家所需积攒的最小局部 token 数量：

$$B_e \geq \frac{\text{FLOPs\_HW}}{2 \times \text{Bandwidth\_HW}}$$

**稀疏度定义：** $S$ 表示的是每次前向传播中激活的专家数量与模型总专家数量的比例，本质上代表了计算资源的激活率。假设模型在 MoE 层通过路由算法，在总共 $E_{\text{total}}$ 个专家中为每个 Token 激活了 $K$ 个专家，则稀疏度 $S$ 可近似表示为：

$$S = \frac{K}{E_{\text{total}}}$$

在宏观统计平均且路由负载完全均衡的理想状态下，总计 $B_{\text{MoE}}$ 个 Token 进入 MoE 层，每个 Token 产生 $K$ 次专家调用，系统总共产生了 $B_{\text{MoE}} \times K$ 个计算请求。这些请求被均摊给 $E_{\text{total}}$ 个专家，因此每个专家分担到的预期 Token 数量为：

$$B_e = \frac{B_{\text{MoE}} \times K}{E_{\text{total}}} = B_{\text{MoE}} \times S$$

代入算术强度边界方程，我们得到了极其重要的系统总 batch size 下界：

$$B_{\text{MoE}} \geq \frac{\text{FLOPs\_HW}}{2 \times S \times \text{Bandwidth\_HW}}$$

当模型追求极度的稀疏性（即赋予 $S$ 极小的值）以期获得低激活参数量时，由于 $S$ 处于分母位置，系统为了将 GPU 推入计算受限的高效区间，所必须积攒的总批处理大小 $B_{\text{MoE}}$ 将呈现出反比例的增长。而在确立了为了吃满算力所必需的 $B_{\text{MoE}}$ 之后，接下来就会考验网络通信。

假设 dispatch 用 FP8，combine 为了精度需要 BF16，那么通信数据量为：

$$V_{\text{send}} = 1 \times H \times B_{\text{MoE}} \quad \text{(Bytes)}$$

$$V_{\text{return}} = 2 \times H \times B_{\text{MoE}} \quad \text{(Bytes)}$$

在处理单层 MoE 时，每个 Token 带来的绝对跨节点网络流量固定为 $1H + 2H = 3H$ 字节，传输耗时为：

$$t_{\text{comm}} = \frac{3 \times H \times B_{\text{MoE}}}{\text{Net}}$$

结合前文 16.6 ms 的时间预算，对于一个拥有 $L$ 个层的模型：

$$t_{\text{comm}} \leq \frac{16.6\text{ms}}{L}$$

代入前面实际通信的计算表达式可得：

$$\frac{3 \times H \times B_{\text{MoE}}}{\text{Net}} \leq \frac{16.6\text{ms}}{L}$$

为了保证专家计算效率，$B_{\text{MoE}}$ 也有与稀疏度和硬件相关的约束，代入最小下界：

$$\frac{3 \times H}{\text{Net}} \times \frac{\text{FLOPs\_HW}}{2 \times S \times \text{Bandwidth\_HW}} \leq \frac{16.6\text{ms}}{L}$$

可得稀疏度与网络带宽以及硬件 roofline 的约束关系：

$$\boxed{S \geq \frac{H \times \text{FLOPs\_HW} \times L}{\text{Net} \times \text{Bandwidth\_HW} \times 11.1\text{ms}}}$$

这一计算并不完全准确，因为加 batch 的同时可能也会面临显存容量与显存带宽的瓶颈，使得无法满足 SLA 要求或者 Attn 浪费很多。但是它提示了**低稀疏度其实会加剧通信瓶颈**。

![稀疏度约束与通信瓶颈示意](/myblog/assets/AFD/image11.png)

Step3 的 co-design 在于，模型为了能 fit 不同 GPU 的最低稀疏下界，做成了 48 选 3 的 0.063 稀疏度，比 DSv3 的 0.031（256 选 8）要高很多。

DSv3 因为不需要三阶段流水线，只需要两阶段，所以容忍的稀疏度下限会再低一点，比如以 H800 计算应该是 0.038，实际上 DSv3 的稀疏度 0.031 比这个值还要低，有几个可能原因：

- 通信已经 bound 了，或者利用 NVLink 其实有更高的带宽，可以进一步提高稀疏度容忍上限
- batch 数还不够达到 FFN 的 roofline，EP 侧的效率并未达到最优
- 也有可能 A/F 不分离时，batch 数已经在 Attn 侧被 bound 了，无法提升

> **PS**：目前模型都在朝更大更稀疏发展，芯片算力堆的也更高，相对来说也更需要超节点。

### 异构

目前 AFD 最明确的收益，还是在异构部署上。比如字节的论文就用 L40S 来降低 FFN 的成本，后续还使用了成本更低的自研 GPU，这也是字节内部应用 AFD 的核心场景。此时更关注的是"单位成本吞吐量"（Throughput per unit cost / per dollar），而非绝对的吞吐量或者时延。实际上因为使用了 L40S，TPOT 已经增加到了 150ms，可能更适合离线场景，这样低端硬件也能被比较好的利用起来。

如果不做 A/F 分离，L40S 的孱弱存储带宽与通信带宽去部署 Attention 的话，算力利用率也会下降很多，性价比降低，TPOT 也会进一步上升。

![字节异构部署成本对比](/myblog/assets/AFD/image12.png)

另一个策略是可以使用 L20 这类特别的"瘸腿卡"，它的显存带宽是 864 GB/s，但算力只有 239T，正好适合用在计算强度低的 Attn 场景，通过 TP scale 满足需要的时延，而在 FFN 阶段用更强算力的卡，降低 FFN 阶段的时延。H20 也是类似思路。

### 同构

假设不考虑利用低成本异构硬件，在同构场景下是否有必要进行 A/F 分离，则与模型需求和硬件特征强相关。

我们再次回顾 A/F 分离的核心思路：Attention 是 decode 时的主要瓶颈，同时我们想保专家计算利用率。那么除非 MLA 这种 decode attention 计算强度很大的，A/F 本身需求就相对比较平的，其他的 Attn 跟 FFN 很难同时达到 roofline，因为 **scale A 很可能会先遇到显存瓶颈（带宽或者容量）**。

再具体一点，当 context 长的时候受限于容量，会导致 A 的 batch 小；或者受限于带宽与 SLO 要求，也会导致 A 的 batch 小，那也就意味着 global batch 小，从而每个 expert 的 batch 小。

假如 context 很长，长到每卡跑 Attention 都只能跑 batch=1，跑 2 就保不住 50ms TPOT。于是 FFN 的阶段，expert 就一定也只有 batch=1，FFN 效率直接血崩。但是这种极端情况，Attention 是绝对主要开销，FFN 那点效率提升也是蚊子肉了。

#### 大EP

通常解决上述问题的方案是"大 EP"，也即像 DeepSeekV3 那样 EP320 或者 EP144 的部署方案。能够这样做是因为 FFN 可以高效的按专家维度切分，专家之间无需通信，如果是 Dense 模型的话 FFN 是不适合 TP 切太碎的。大 EP 的目的，是为了凑更多的 global batch。

根据 profiling，DSv3 这个尺寸的 FFN 只要 32 卡就能保住所需的 TPOT + 高 MFU，但单独的 EP32（32 卡）是跑不出 TPOT 50ms + 高 MFU 的，因为能跑 FFN 高 MFU 的 batch size，同时跑在 32 卡上的 Attention 是保不住 50ms 的，尤其是 context 长点（DSv3 是 5k avg）。

因此需要大 EP 继续横向 scaling 去扩 global batch 的同时也增加 Attn 的计算资源，确保 Attention 也还能满足 TPOT。不过加 EP 总归是有限的（不能超过 expert 数量的卡数），一旦开始复制一些 expert，这些 expert 的 batch 也要往下掉了。当然当大 EP 到达上限的时候，可以用 TP 来继续扩容。而 TP 对于细粒度专家来说是很不友好的，比如 DS 专家的维度只有 2048，当 TP 太大后就会 GEMM 效率极低。

大 EP 的另一个好处是，专家切得越碎（极端情况下每张卡只管一个专家），单卡的因专家带来的显存压力就越小，留给 Attn 的就越多，这也是一种隔离的尝试，只是没有 A/F 分离来的彻底。以 DSV3 为例，每次激活 9 个 expert，光是这 9 个 expert 参数占用的显存就达到了 22.15 GB，而 1K Sequence Length 下的 KV cache 仅为 34 MB。

那么如果不考虑规模问题，A/F 是否有必要分离的问题可以转换成：**A/F 不分离时，Expert 的计算利用率是否到达上限，或者说是否处于计算瓶颈**。下面会进行推导计算，会用到以下参考数据：

| 模型 | hidden\_size | 总参数量 | 激活参数量 | Attention 方案 | A 计算强度 | F 计算强度 | KV Cache / token | MoE | 显存/expert |
|------|-------------|---------|-----------|---------------|-----------|-----------|-----------------|-----|-----------|
| Qwen3-235B | 4096 | 235B | 22B | GQA（Head=64, G=16） | 32 | $2 \times B_{\text{exp}}$ | 188 KB | 8/128 | 1.7 GB |
| DeepSeekV3 | 7168 | 671B | 37B | MLA（Head=128, $D_c$=512） | 512 | — | 34.3 KB | 8/256 | 2.5 GB |
| Step3 | 7168 | 316B | 38B | MFA（Head=64, D=256） | 128 | — | 24.6 KB | 3/48 | 6.2 GB |

首先需要考虑的就是，达到需要打满 FFN 计算瓶颈的 batch 数以后，对于长序列，Attn 显存是否够用。卡数越多的情况下，需要分摊的权重越少，可以用来存 KV 的空间越大。比如对于 Qwen 来说，32K 上下文就需要至少 25 张卡，这个数字随着 context 增加而增加：

$$B_{\text{MoE}} \times \frac{\text{Mem}_{kv}}{N} > \text{Mem}_{\text{total}} - \text{Mem}_{\text{weight}}$$

如果 A/F 不分离，Attention 使用 DP only 的情况下，Attention 每张卡的输入 batch size $B_{\text{attn}}$ 可以使用如下公式表达（$N$ 为总卡数，$\text{Experts}$ 为专家数）：

$$B_{\text{attn}} = \frac{B_{\text{exp}} \times \max(\text{Experts}, N)}{\text{TopK} \times N}$$

而 $B_{\text{exp}}$ 若要满足 FFN 时的最优计算效率，需要满足：

$$2 \times B_{\text{exp}} \geq \frac{\text{FLOPs\_HW}}{\text{Bandwidth\_HW}}$$

暂未考虑 SLO 的要求，也即必须 scale A 来降低 TPOT，实际上下面列的几种场景保住显存容量的情况下应该都能保住 50ms，但是对于一些显存带宽更低的可能保不住。

**长序列 (32K) & 专家少 (48) & roofline 高 (H800)**

以 Step3 的参数代入，对于一个 32K 上下文输入的场景，Attention 每个 Batch 需要输入约 1GB 的 KV cache。此时会发现 A/E 不分离的话，要么 A 的显存不够，要么 FFN 无法打满。这里 Step3 特意让专家数比较少，导致了无法持续 scale N。

而如果 A/F 分离，Attention 集群的规模和 Expert 集群的规模则可以不一致，用 $N$ 依然表示 Attention 集群规模，用 $M$ 表示 Expert 集群的规模，那么 Attention 每张卡的输入 batch size $B_{\text{attn}}$ 的公式转变成：

$$B_{\text{attn}} = \frac{B_{\text{exp}} \times \max(\text{Experts}, M)}{\text{TopK} \times N}$$

设置 $M = 48$，$N = 96$，$B_{\text{attn}}$ 则为 50，此时 $M$ 和 $N$ 的约束满足了 KV cache 的存储和专家利用率的上限。32K 的上下文 Attention 计算需要访存 1 GB 的 KV cache，使用 H800 需要约 295 µs，每个 Expert 满计算利用率计算需要约 20 µs，乘以 batch size 后也满足 50ms。

**短序列 (8K)**

同样用 Step3 模型，如果序列降低到 8K，保持 $B_{\text{attn}}$ 能被显存容纳就可以，此时 A/F 不分离一样能达到 FFN 算力被打满。

**低 roofline（H20）**

如果 GPU 自身的 roofline 低，算力更容易被打满的话，也可以降低 $B_{\text{attn}}$，从而降低显存需求，可以 A/F 不分离。

**专家数更多（DeepSeek 256）**

DeepSeek 模型的专家数量很多，高达 256，所以其可以扩展的大 EP 规模要远高于 Step3 的 48，所以 DeepSeek 模型所能容忍的 A/E 不分离的序列长度要远超过 Step3。

#### 小结

是否必须分离和以下几个因素相关：

1. **模型结构**：专家数量决定大 EP 的上限，专家数量越多，大 EP 即可满足足够的专家输入 batch
2. **硬件的计算访存比**：硬件的计算访存比越低，专家所需要输入的 batch 就越小，可以不分离
3. **硬件的 HBM 空间**：HBM 的空间越大，Attention 侧所能承受的 batch 越大也能保证足够的专家输入 batch
4. **上下文序列的长度和业务的 SLO**：上下文序列长度越小和业务 SLO 越宽松，对 HBM 资源和集群规模要求也就越小，也可以不分离就满足需求

### 更小规模达成

根据前面的分析，可知很多场景下可能并不必须 A/F 分离——光靠 scale EP 也能保住 FFN 的效率。但阶跃的作者认为，A/F 分离可以用更小的规模达成同样的效果。这是很有收益的，因为多卡资源更难获得，同时可以减少不均衡，并行度越高越容易出现严重 straggler。

![更小规模达成同等效果对比](/myblog/assets/AFD/image13.png)

但是我对这里还是有所疑问的，论文中给出的实验数据看似不错，实际上并不是同类对比：

- Step3 虽然激活参数与 DSV3 相同，但是总体参数量小一倍，需要的显存数量更低，留给 Attn 与 FFN 的空间也更大
- MFA 也让它的 KV cache 需求降低了很多，Attn 计算时间也降低了，对 SLO 的要求下降了
- 48 选 3 的稀疏度很高，对 global batch 的要求也没那么高，从而 A 部分的 DP 也不需要更高
- FFN 部分 DSv3 是 2048 维，Step3 的维度会高很多（只激活 4 个就达到了同样的参数量），计算更友好

再分析阶跃作者的解释：A/F 分离后 F 可以不用跟着 A 一起 scale，所以 F 可以被切分得更少。跑到高效率需要的 global batch 就小，相应的 A 那边需要的也少。实际上 FFN 的计算 pattern 应该是逐 EP 的，虽然切的少的话，F 卡分到的 token 变多了，但是如果总 batch 不高的话，单个专家分到的 token 也并没有变高。所以还是需要更高的 global batch，Step3 就 48 专家，需要的 global batch 要求低。

而且根据前面的计算，低序列长度下，A/F 不分离大约也只需要 24/48 卡就能满足 FFN 的 MFU，与这里给的 A/F 分离的规模是一样的。所以这个优点看起来并不能成立。

而 DSv3 这个尺寸的 FFN 大约需要 4F（32 卡）就能保住所需的 TPOT + 高 MFU，但是这样的 batch size 需要更多卡才能让 Attn 侧不变成瓶颈，导致最终加到了 144 卡。MLA 计算强度很高，所以 A 和 F 是平衡的，并不会因为共卡时 scale A 导致 A 的算力浪费更多，如果是因为显存容量瓶颈，同构时 A/F 分离并不会缓解。理论上 A/F 分离也不能显著降低 DSv3 的最优规模（待验证）。
