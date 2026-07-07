---
title: "LMetric：乘法即调度——KVCache 感知与负载均衡的免调参融合"
tags:
  - LLM推理
  - 请求调度
  - KVCache
  - 负载均衡
  - OSDI
---

## 🧠 核心摘要 (TL;DR)
> LLM 请求调度需要同时兼顾两个相互拉扯的目标——把请求路由到「持有可复用 KVCache 的实例」以跳过 prefill 计算，以及在实例间「保持负载均衡」。现有方案用线性组合（加权和）来融合这两个指标，但权重 λ 是工作负载相关的，需要繁琐的逐场景调参或自建模拟器。本文（OSDI'26）提出 **LMETRIC**：把「KVCache 感知指标（新增 prefill token 数 P-token）」与「负载均衡指标（实例 batch size BS）」**相乘**作为调度分数，取最小积的实例。关键洞察是——乘法在「跨实例比较」时让线性组合的超参自动抵消，因此**完全无需调参**。在真实 chatbot / coding-agent 负载上，相比 vLLM-v1 把 TTFT 降低 92%、TPOT 降低 24%；相比线上生产调度器把 TTFT 降低 39%、TPOT 降低 51%，并已通过 canary 灰度上线。

## 🏷️ 元数据
- **论文标题**: Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling
- **发表机构/会议**: OSDI'26（上海交大 IPADS + 阿里巴巴 Qwen/Bailian 团队）
- **年份**: 2026（arXiv 提交 2026-03-16，v3 2026-06-05）
- **链接**: [[LMetric_paper|📄论文]] | [💻 Code](https://github.com/blitz-serving/blitz-router)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #llm-serving #request-scheduling #kvcache #load-balancing #prefix-caching
- **前置知识 (Prerequisites)**: [[KV稀疏调研]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[MSched]], [[ShiftParallelism]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 核心制约对：KVCache 命中率 vs 负载均衡

在分布式 LLM 推理集群中，请求路由器面对的是一对**结构性矛盾**的优化目标：

1. **KVCache 感知（KV\$-awareness）**：现代 LLM 服务普遍开启 prefix caching，多轮对话、共享 system prompt、coding agent 的长上下文都会在某些实例上留下可复用的 KV 缓存。若把请求路由到「已经缓存了它的共享前缀」的实例，prefill 阶段可以直接跳过这部分计算，TTFT 大幅下降。
2. **负载均衡（Load balancing）**：vLLM 默认采用纯负载均衡策略（类似 JSQ，Join-the-Shortest-Queue），把请求均匀分散到各实例，避免某些实例过载导致排队延迟。

这两个目标**天然冲突**：一味追求 KV\$ 命中，会把请求持续往「持有缓存的少数实例」上堆，造成这些实例过载、其余实例空闲——论文明确指出「纯 KV\$ 感知会把请求偏置到有缓存命中的实例，损害负载均衡」。反过来，纯负载均衡（vLLM 的做法）则完全浪费了 prefix cache 的复用机会。论文的 motivation 实验显示：在纯负载均衡基础上叠加 KV\$ 感知，平均 TTFT 改善 84%、平均 TPOT 改善 17%——可见 KV\$ 感知不可或缺，但又不能压垮均衡。

### 现有方案的瓶颈：四类策略都需要「调参」或「建模拟器」

论文把已有融合策略归纳为四类，逐一指出其工程痛点：

- **线性组合（Linear combination）**：以阿里 Bailian、NVIDIA ai-Dynamo 为代表，用加权和 `Score = λ·(KV指标) + (1-λ)·(负载指标)` 融合。**致命问题是 λ 与工作负载强相关**：ChatBot 最优 λ≈0.7，API 负载最优 λ≈0.55；而且存在「拐点（knee point）」——超过某个 λ 负载失衡会急剧恶化。论文给出量化证据：λ=0.9 时两实例的 prefill 时间分化到 3.57s vs 2.17s（严重失衡），λ=0.7 时才平衡到 3.43s vs 3.40s。由于线上负载是**动态变化**的，任何静态调好的 λ 都是次优解。
- **过滤式（Filter-based）**：以 AIBrix、Preble 为代表，先用阈值过滤掉一批实例再做选择，**阈值本身又需要调**。
- **模拟式（Simulation-based）**：以 llm-d（基于 VIDUR）、PolyServe 为代表，用性能模拟器预测每个实例的延迟。**问题是模拟器需要逐模型、逐硬件地开发和标定**——论文测得：标定良好 vs 未标定的模拟器，TTFT/TPOT 尾延迟相差 75.6% / 79.7%；即便标定良好，仍有约 10% 的请求预测误差 >20%。
- **纯负载均衡**：vLLM 的默认策略，简单但完全忽略 KV\$ 复用。

> 一句话概括痛点：**所有现有方案要么需要调超参，要么需要建模拟器，且都对工作负载漂移敏感**。论文要回答的核心问题是——能否找到一个「免调参、对负载漂移鲁棒」的融合方式？

### 相关工作（按先驱 / 直接对比 / Follow-up 分类）

> 以下论文均来自本文参考文献列表（共 56 篇），经 WebFetch 实际抓取确认，未编造。

**先驱 / 奠基工作（serving 基础设施 + KVCache 复用）**
- **vLLM / PagedAttention** [Kwon et al., SOSP'23]：分页式 KV 显存管理，是现代 LLM serving 的事实基线，也是本文 prefix caching 与负载均衡基线的载体。
- **Mooncake** [Qin et al., FAST'25]：KVCache-centric 架构，「用更多存储换更少计算」，奠定了 KV\$ 复用作为一等公民的调度范式。
- **CachedAttention** [Gao et al., ATC'24]：面向多轮对话的成本高效服务，通过缓存历史 KV 降低重复 prefill。
- **DistServe** [Zhong et al., OSDI'24] / **Splitwise** [Patel et al., ISCA'24]：Prefill-Decode 解耦（PD 分离）的奠基工作，使 prefill 与 decode 的资源/调度可分别优化。
- **Sarathi-Serve** [Agrawal et al., OSDI'24]：Chunked prefill，调和吞吐-延迟权衡，是 prefill 调度的重要先驱。
- **VTC** [Sheng et al., OSDI'24]：LLM 服务中的公平性度量，调度目标多元化的先驱。

**直接对比基线（本文实验中真正较量的调度器）**
- **Bailian** [阿里云 Bailian, 2026]：本文的「生产前任」调度器，采用调好的线性组合，是 canary 对比对象。
- **NVIDIA Dynamo** [github.com/ai-dynamo/dynamo, 2025]：线性组合（P-token + total tokens）。
- **llm-d** [Google, 2025]：基于 VIDUR 模拟器的模拟式调度。
- **Preble** [Srivatsa et al., ICLR'25]：混合策略（过滤 + 线性组合）的高效分布式 prompt 调度。
- **AIBrix** [vllm-project/aibrix, 2025]：过滤式调度，含 Preble plugin。
- **PolyServe** [Zhu et al., 2025]：模拟式、面向多 SLO、用 SLO 梯度做调度。
- **VIDUR** [Agrawal et al., MLSys'24]：大规模 LLM 推理模拟框架，是 llm-d 模拟式调度的底座。

**近期 Follow-up / 平行研究（同期 KV 亲和 + 负载均衡融合）**
- **DualMap** [Yuan et al., 2026]：与本文同期、最贴近的平行工作——同时实现「缓存亲和」与「负载均衡」的分布式 LLM 服务。
- **KVCache in the Wild** [Wang et al., ATC'25]：在大型云厂商刻画并优化 KVCache，是本文 trace 与 motivation 的数据来源（同团队工作）。
- **FairBatching** [Lyu et al., 2025]：公平感知的 batch 形成。
- **SLOs-Serve** [Chen et al., 2025] / **PolyServe**：多 SLO 服务方向。
- **TokenScale** [Lai et al., 2025] / **BlitzScale** [Zhang et al., OSDI'25] / **Aegaeon** [Xiang et al., SOSP'25]：自动扩缩容与 GPU 池化，与调度互补。
- **POD-Attention** [Kamath et al., ASPLOS'25]：prefill-decode 完全 overlap 的算子层优化。
- **ServeGen** [Xiang et al., 2025] / **BurstGPT** [Wang et al., SIGKDD'25]：工作负载刻画与数据集，为调度评测提供真实 trace。

---

## ⚙️ 核心设计与机制

### 层 1：顶层算法决策——为什么是「乘法」而不是「加法」

**冲突点**：线性组合 `λ·A + (1-λ)·B` 的趋势是对的（同时考虑两个指标），但它把「两个指标的相对重要性」硬编码进了一个需要调的权重 λ 里。能不能既保留「同时考虑两指标」的趋势，又让权重消失？

**消解**：LMETRIC 的答案是把加法换成乘法。调度分数定义为两个指标的乘积，路由到**乘积最小**的实例：

$$\text{Score}(i) = \text{P-token}(i) \times \text{BS}(i)$$

$$i^* = \arg\min_i \; \text{P-token}(i) \times \text{BS}(i)$$

两个指标都经过精心选择，且都是 LLM 推理的本征特征量：

- **P-token（KV\$ 感知指标）**：若把请求路由到实例 $i$，在**考虑 KV\$ 命中后**实际需要新计算的 prefill token 数。命中的共享前缀越多，需要新算的 token 越少，P-token 越小。它直接刻画 prefill 的计算量（即 TTFT 的主要成本）。
- **BS（负载均衡指标）**：实例 $i$ 当前的 batch size（运行中 + 排队中的请求数），刻画 decode 阶段的负载压力（即 TPOT 的主要成本）。

**为什么选 P-token 而不是更直觉的「KV\$ 命中率」？** 因为 P-token 不仅含命中信息，还隐含了「排队中的 prefill token」——当一个实例命中很多但已经排了大量待处理 prefill 时，它的 P-token 反而会偏大，从而被自然降权。这让单一指标同时携带了「缓存复用」与「prefill 队列」两重信息（消融实验证明 P-token 比命中率更优，见层 4）。

**为什么选 BS 而不是「总 token 数」？** 因为 decode 时间对 batch size 比对总 token 数更稳定——同样 batch size 下 decode 每步耗时近似恒定，BS 是 decode 负载更干净的代理（消融实验证明 BS 优于 #Tokens）。

### 层 2：数学推导——超参为何抵消 + 失效条件 + 置信度校验

**(a) 超参抵消的直觉。** 设线性组合分数为 $\lambda A + (1-\lambda) B$。论文的关键论证是：乘积 $A \times B$ **保留了与线性组合相似的「随两指标变化的单调趋势」**，但由于调度只关心「跨实例比较 Score 的相对大小」（取 argmin），线性组合里那些刻画 A、B 相对量纲/重要性的系数，在比较过程中被抵消掉了。换言之，乘法把「需要人为标定的相对权重」转化为「两个本征量自带的量纲关系」，于是无需再引入并调节 λ。

**(b) 乘法什么时候会失效？** 这是论文最严谨的部分。乘法的隐含假设是：当某实例 KV\$ 命中多（P-token↓），它的 batch size 增长（BS↑）能够「抵消」P-token 的下降，从而 Score 不会一味偏小、避免持续吸入请求。**当这个抵消不成立时，乘法失效——即出现 KV\$ 热点（hotspot）**：某个前缀被反复访问，却只缓存在少数实例上，BS 的上升赶不上 P-token 的下降。

论文给出形式化推导。将请求按「共享 KV\$ 前缀」划分为类别集合 $\mathcal{C}$；对某类 $c$，把实例划分为持有该前缀的集合 $M$ 与其余 $\bar{M}$。设 $x$ 为到达请求中属于类 $c$ 的比例，$\bar{x}=1-x$，QPS 为总查询率，$BS_0$ 为 $t$ 时刻前的均衡 batch size，则两组实例的 batch size 比值（**式 1**）为：

$$\frac{BS_t}{\bar{BS}_t} = \frac{BS_0 + \dfrac{x\cdot \text{QPS}}{|M|}\, t}{BS_0 + \dfrac{\bar{x}\cdot \text{QPS}}{|\bar{M}|}\, t}$$

由此推出**安全条件（式 2）**：

$$\frac{x}{\bar{x}} \le \frac{|M|}{|\bar{M}|}$$

物理含义直白：**某类前缀的「流行度比例」不得超过它的「缓存覆盖比例」**。满足时 $BS_t \le \bar{BS}_t$，即热点实例不会累积出更大的 batch，乘法安全。论文实测**所有评测 trace 都满足式 2**；唯一违反出现在 Bailian 的一个「thinking 工作负载——突发的、共享同一前缀的长请求」，在第 11 分钟附近可见。

**(c) 置信度校验 / 失效缓解——两阶段热点检测器。** 这正是 paper-reading skill 强调的「近似方法必须自证不失真」的环节。由于式 2 是在「最坏情况：全部请求都打向 $M$」下推导的，它是失效的**必要非充分**条件，因此论文设计两阶段检测：
- **阶段 1（必要条件报警）**：在线监控每类的 $x/\bar{x}$ 与 $|M|/|\bar{M}|$（只跟踪 KV\$ 命中最高的那批请求以降开销），一旦违反式 2 即拉响警报。
- **阶段 2（充分性确认）**：报警后并不立即处理，而是继续观察——只有当**连续 $2|M|$ 个请求**在热点实例 $m\in M$ 上的得分确实低于非热点实例 $m'\in\bar{M}$（即 $\text{P-token}_m \times \text{BS}_m \le \text{P-token}_{m'} \times \text{BS}_{m'}$），才确认热点成立。
- **缓解动作**：确认后，把热点集合 $M$ 从候选中过滤掉，**回退到纯负载均衡（vLLM 策略）**——即放弃这部分缓存复用，优先保住均衡。这是一个「鲁棒几何/静态基线兜底」的经典置信度开关设计。

### 层 3：底层实现与工程落地——blitz-router

**冲突点**：调度策略的多样性（线性、过滤、模拟、乘法……）与「不想为每种策略重写一套路由器」之间的矛盾；以及路由器本身若用 Python 写会成为吞吐瓶颈。

**消解**：
- **语言选择**：blitz-router 用 **Rust** 实现，「保证高效与健壮」，可与任意 LLM 推理引擎配合。论文给出性能数据：AIBrix 的 Go 重写版比 vLLM 的 Python 版快约 **6.2×**（部分因 vLLM 一个已确认的 bug），而 Rust 路由器在同策略下又比 AIBrix 快约 **1.2×**。
- **指标采集——indicator factory + piggyback**：路由器维护一个「指标工厂」自动采集/计算每实例的指标；每个引擎保持一条长连接，指标被 **piggyback 在推理响应的 header 里**返回，避免额外的探测请求。
- **可编程调度模型**：提供一套编程 API，把调度表达为「对符号化的每实例指标的函数」，配套 filter（过滤）与 min/max-selection（最值选择）原语。LMETRIC 就是一行「对 P-token×BS 取 min」的策略，过滤式/线性式也能用同一框架表达——这是工程上的核心抽象。
- **开源**：路由器开源于 `github.com/blitz-serving/blitz-router`；评测用的 trace 开源于 `github.com/alibaba-edu/qwen-bailian-usagetraces-anon`；trace replayer 单独开源。（以上地址均来自论文原文，未拼接。）

## 📊 实验与性能评价

### 层 4：多模型泛化验证与量化实验

**测试床**：16× NVIDIA H20 GPU（每卡 96 GB HBM）；路由器跑在 160 核 Intel Xeon、1 TB DRAM 上；推理引擎为 vLLM-v1。
**模型（跨家族/跨架构验证）**：Qwen2-7B（稠密）、Qwen3-30B（MoE），覆盖 dense 与 MoE 两类架构。
**工作负载（4 类真实 trace）**：ChatBot（Qwen）、Agent/API（Qwen）、Coder（Bailian，2025-11）、ToolAgent（Kimi/Mooncake trace）。trace 被缩放到测试床最大负载的一半。

**对比基线（6 个，含工业生产系统）**：Bailian（调好的线性）、vLLM（纯负载均衡）、Dynamo（线性：P-token + total tokens）、llm-d（模拟式 / VIDUR）、Preble（过滤 + 线性混合）、PolyServe（模拟式、SLO 梯度）。

**核心结果**：

| 对比对象 | 工作负载 | 主要指标改善 |
|---|---|---|
| vLLM-v1 | 综合（摘要口径） | 平均 TTFT **−92%**，平均 TPOT **−24%** |
| 生产调度器(Bailian) | canary | 平均 TTFT **−39%**，平均 TPOT **−51%** |
| llm-d（次优） | ChatBot | P99 TPOT **−13%**，TTFT 相近 |
| Preble | 综合 | 平均 TTFT −56%、平均 TPOT −8%，P99 TTFT −45%、P99 TPOT −16% |
| PolyServe | 各速率 | 平均/P99 的 TTFT 与 TPOT 全面更低 |

- 对 **Preble**：因 Preble 大部分时间会回退到它的线性分支，故 LMETRIC 全面领先。
- 对 **PolyServe**：PolyServe 在实例 0–8 上制造了一个负载梯度，让实例 9–15 闲置；LMETRIC 把负载铺满全部 16 个实例，故延迟更低。
- **ToolAgent 例外**：LMETRIC 的平均 TTFT 比 llm-d 高约 10%，但 TPOT 低约 30%——属于 TTFT/TPOT 的权衡，整体仍占优。

**消融实验（Qwen3-30B，ChatBot）**——验证两个指标的选择：
- **P-token vs (1 − KV\$ 命中率)**：在 KV\$ 命中率相近时，P-token 让 P50 TTFT 低 14.4%、P95 TTFT 低 42.8%。差距来自 P-token 携带的「排队 prefill token」信息带来的更好负载均衡。
- **BS vs 总 token 数（#Tokens）**：BS 更优，因为 decode 时间随 batch size 比随总 token 数更稳定；用 decode batch size 结果相近。

**模拟器敏感性**：标定良好 vs 未标定模拟器，TTFT/TPOT 尾延迟差 75.6% / 79.7%；即便标定良好仍有约 10% 请求预测误差 >20%。在 ToolAgent 上，模拟式 TPOT 尾延迟比最优线性方法慢 71.1%——印证「模拟式调度脆弱、依赖标定」的论点。

**生产部署（canary，2026-05）**：LMETRIC 服务一个 Qwen3.5-27B 集群（数百 GPU 规模）。流量按 1/3（LMETRIC）vs 2/3（原 Bailian 调度器）切分，每 GPU 请求数相同。结果：平均 TTFT −39%、平均 TPOT −51%，意味着**在相同 SLO 下可用更少 GPU 服务同样负载**。

### 局限性 (Limitations)
- **失效条件依赖 trace 特性**：乘法在 KV\$ 热点下失效，虽有两阶段检测器兜底，但兜底动作是「回退纯负载均衡」，意味着热点期间放弃了缓存复用收益。论文承认所有评测 trace 几乎都满足式 2，热点是边缘情形（仅 Bailian thinking 负载第 11 分钟出现），现实中热点更频繁/更持久时的表现仍待观察。
- **两指标的普适性**：P-token 与 BS 的选择基于「prefill 计算量 ∝ TTFT、decode batch ∝ TPOT」的经验规律，在 PD 解耦、推测解码、长 decode 等新范式下是否仍是最佳代理，论文未充分展开。
- **检测器超参隐患**：两阶段检测器引入了 $2|M|$ 这一确认窗口，本质上又是一个（虽然弱）需要设定的参数，与「完全免调参」的主张存在轻微张力。

---

## 💡 启发与我的想法

- **「比较时抵消」是一个被低估的设计哲学**：本文最漂亮的地方不是乘法本身，而是抓住了「调度只需相对序、不需绝对值」这一事实，从而把「需要标定的权重」转化为「指标自带的量纲关系」。这种思路可迁移到其他需要融合多指标且只关心 argmin/argmax 的系统决策（如 KV 驱逐优先级、专家路由、缓存替换）——凡是「只比相对大小」的地方，加法融合都值得反思能否换成乘法/比值。
- **置信度开关 + 鲁棒兜底是近似方法的标准范式**：与 KV 稀疏方法里「用代表性子集估精度、不达标回退稠密」如出一辙（参见 [[KV稀疏调研]]）。本文的两阶段检测器（必要条件报警 → 充分性确认 → 回退 vLLM）是这一范式在调度层的实例，值得作为模板复用。
- **可借鉴之处**：indicator factory + piggyback header 的指标采集方式很轻量，避免了额外探测 RPC，对自研路由器有直接参考价值；把调度策略抽象为「对符号指标的函数 + filter/min-max 原语」的可编程模型，是解耦「策略」与「机制」的好工程实践。
- **改进空间**：① 热点缓解能否不「全退均衡」，而是做「部分复用 + 限流」的折中？② P-token×BS 是否可推广为更一般的「指标乘积族」，针对 PD 解耦场景换用 prefill-queue × decode-batch？③ 检测器的 $2|M|$ 窗口能否进一步自适应化，真正做到零参数。
- **下一步 Action**：
  - [ ] 阅读同期平行工作 DualMap，对比它在「缓存亲和 + 负载均衡」上与乘法范式的异同
  - [ ] 复盘 blitz-router 的 Rust 实现，关注 indicator factory 的指标更新一致性如何保证
  - [ ] 思考乘法调度思想能否迁移到 KV Cache 驱逐 / RL rollout 调度场景


