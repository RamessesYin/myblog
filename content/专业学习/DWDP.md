---
title: "DWDP：分布式权重数据并行"
tags:
  - LLM推理
  - MoE
  - 数据并行
  - 分布式推理
  - NVLink
---

## 🧠 核心摘要 (TL;DR)
> 现有 TP/PP/EP 并行策略在层级边界处需要集体同步（AllReduce/AllGather），当输入序列不均衡时同步等待开销高达 12%。DWDP 提出"保留数据并行语义、跨 GPU 分散 MoE 专家权重"的新范式：各 GPU 独立运行，按需通过 NVLink P2P 拉取缺失专家，并配合双缓冲异步预取与带宽复用优化，在 GB200 NVL72 + DeepSeek-R1 上实现端到端 8.8% TPS/GPU 提升（20-100 TPS/user 范围）。

## 🏷️ 元数据
- **论文标题**: DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72
- **发表机构/会议**: arXiv（NVIDIA 团队）
- **年份**: 2026
- **链接**: [[DWDP_paper|📄论文]] | 💻 Code: TensorRT-LLM PR #12136（集成中，暂无独立开源仓库）

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #moe #expert-parallelism #data-parallelism #llm-inference #nvlink #distributed-inference
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[ShiftParallelism]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[MSched]], [[ShiftParallelism]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

大规模 MoE 模型（如 DeepSeek-R1/V3）推理时，业界主流并行策略存在三类形态：

| 并行策略 | 原理 | 核心缺陷 |
|---------|------|---------|
| **张量并行 (TP)** | 将模型权重按列/行切分到多 GPU | 每层结束需 AllReduce，层数越多同步越频繁 |
| **管道并行 (PP)** | 按层划分 stage，bubble 形式流水 | 层级边界需要显式同步，气泡浪费严重 |
| **专家并行 (EP)** | 将不同专家分配到不同 GPU | 路由不均衡导致负载倾斜，集体 AllGather 是强同步点 |

论文核心观测：当序列长度变异系数（CV）为 20% 时，**集体同步等待占总计算时间约 12%**。这种开销在 GB200 NVL72 这类高带宽互联架构上尤为浪费——NVLink 提供了充裕的 P2P 带宽，却被集体通信原语的同步语义锁住。

**双重不均衡根源**：
1. **请求级不均衡**：不同 rank 处理不同长度的序列、不同 KV Cache 命中率
2. **权重级不均衡（路由偏斜）**：MoE 路由器倾向于将更多 token 导向少数热门专家，部分 rank 计算量远超其他 rank

### 论文的切入点

DWDP 的核心思路是**解耦"执行同步"与"权重分布"**：

- 保持**数据并行 (DP)** 的执行语义 → 每个 GPU 独立处理自己的 batch，无需等待其他 rank
- 将 MoE 专家权重**打散到所有 peer GPU** → 每个 GPU 只本地存储 1/N 的专家
- 需要远端专家时通过 **NVLink P2P（cudaMemcpyAsync）按需拉取** → 点对点传输，无集体同步

与标准 EP（数据并行 + 专家并行，DEP）相比，DWDP 避免了 AllGather 类的障碍同步，将通信从"集体同步墙"转变为"异步点对点流"。

---

### 🗂️ 相关工作

#### 先驱/奠基工作

**① GPipe（Huang et al., NeurIPS 2019）**
管道并行的奠基工作，将神经网络按层切分为多个 micro-stage，用气泡流水掩盖通信延迟。DWDP 在设计上脱离管道并行的层级边界同步，采用更细粒度的异步通信。

**② Megatron-LM（Narayanan et al., SC'21）**
张量并行（TP）的系统性实现，为大规模语言模型训练/推理建立了"切权重 + AllReduce"的范式。DWDP 主要应对的正是 TP 在推理不均衡场景下的同步开销。

**③ DeepSpeed-MoE（Rajbhandari et al., ICML'22）**
首个系统性研究 MoE 推理与训练的工程框架，提出专家并行和 All-to-All 通信策略。DWDP 的专家权重分布思想在此基础上进一步演进，将 All-to-All 集体通信替换为 P2P 异步拉取。

**④ Tutel（Hwang et al., MLSys'23）**
自适应 MoE 规模化方案，动态调整专家数量和并行度。与 DWDP 互补——Tutel 在路由粒度优化，DWDP 在通信模式优化。

**⑤ Roofline Model（Williams et al., CACM'09）**
多核架构性能分析经典模型，DWDP 在分析通信-计算重叠时借用 Roofline 方法论来识别瓶颈转移。

---

#### 直接对比方法

**⑥ DistServe（Zhong et al., OSDI'24）**
将 prefill 与 decode 阶段物理分离到不同 GPU 集群，解决二者资源需求不匹配的问题。与 DWDP 侧重点不同：DistServe 关注阶段分离，DWDP 关注单阶段内的专家负载均衡。两者可以正交组合。

**⑦ Splitwise（Patel et al., ISCA'24）**
通过相变分割（phase splitting）将 prompt 处理与 token 生成解耦，在异构集群上分别优化。类似 DistServe，关注的是 prefill/decode 分离而非 MoE 内部通信。

**⑧ Preble（Srivatsa et al., OSDI'24）**
分布式 prompt 调度框架，通过 KV Cache 感知的负载均衡减少 GPU 空闲。在调度层面与 DWDP 互补——Preble 优化请求路由，DWDP 优化执行并行策略。

**⑨ Mooncake（Qin et al., FAST'25）**
"用存储换计算"的 KV Cache 管理架构，通过 prefix cache 减少重复计算。关注 KV Cache 复用，与 DWDP 关注的 MoE 权重传输不同维度。

**⑩ SGLang v0.4（SGLang Team, 2024）**
提出零开销批调度器和 Cache 感知负载均衡，是当前高性能 LLM 推理的重要基线框架。DWDP 在 TensorRT-LLM 框架内实现，与 SGLang 是竞争关系。

**⑪ TensorRT-LLM ADP 均衡策略（NVIDIA, 2026）**
NVIDIA 官方在 TRT-LLM 中提出的自动数据并行（ADP）负载均衡方案，是 DWDP 的直接前身和集成宿主。DWDP 作为 TRT-LLM PR #12136 集成其中。

**⑫ TensorRT-LLM EP 扩展（NVIDIA, 2026）**
NVIDIA 在 TRT-LLM 中实现的专家并行扩展（scaling expert parallelism），是 DWDP 对比的主要基线 DEP（数据并行+专家并行）方案。

**⑬ Least-Loaded Expert Parallelism（arXiv:2601.17111）**
专门针对 MoE 负载不均衡的专家并行优化方案，与 DWDP 从不同角度（负载感知调度 vs. 消除集体同步）解决同一问题。

**⑭ MixServe（arXiv:2601.08800）**
自动混合并行策略的 MoE 服务系统，在多种并行度之间自动切换。DWDP 则专注于固定数据并行拓扑下的权重分散策略。

**⑮ Capacity-Aware Inference（arXiv:2503.05066）**
针对 MoE 中 straggler 效应（慢专家拖累整体）的容量感知推理优化。从容量调度角度缓解不均衡，与 DWDP 的通信优化路径不同。

**⑯ MoETuner（arXiv:2502.06643）**
通过优化专家放置（expert placement）实现均衡服务的 MoE 推理框架，与 DWDP 的思路有相通之处（都关注专家分布），但 MoETuner 侧重静态放置策略，DWDP 侧重动态按需拉取。

**⑰ FLYING SERVING（arXiv:2602.22593）**
动态 DP-TP 并行度切换方案，在线重配置以适应负载变化。DWDP 采用固定数据并行配置，通过权重分散而非并行度重配来解决不均衡。

---

#### 近期 Follow-up / 相关扩展

**⑱ DeepSeek-R1（DeepSeek-AI, arXiv:2501.12948）**
DWDP 的主要测试模型，671B MoE 架构，通过强化学习激励推理能力。其 MoE 架构的路由偏斜特性是 DWDP 针对的核心场景。

**⑲ DeepSeek-V3（DeepSeek-AI, arXiv:2412.19437）**
671B MoE 架构，DeepSeek-R1 的前一代基础模型。DWDP 的设计目标是同时支持 V3/R1 系列。

**⑳ Scalable Training of MoE with Megatron Core（Yan et al., arXiv:2603.07685）**
将 Megatron-Core 扩展到大规模 MoE 训练，提出训练侧的专家并行改进。与 DWDP 的推理侧优化形成互补——训练与推理均走向异步化专家管理。

**㉑ CRAFT（arXiv:2603.28768）**
成本感知专家副本分配方案，通过逐层估计优化专家副本数量。关注专家副本分配策略，与 DWDP 的按需拉取形成对比。

**㉒ Multi-Layer Scheduling for MoE Reasoning（arXiv:2602.21626）**
MoE 推理的多层调度框架，从调度层面减少专家并行和路由复杂度开销。

**㉓ MoEless（arXiv:2603.06350）**
通过 Serverless 计算解决专家负载不均衡问题，利用弹性扩缩容应对热点专家。与 DWDP 的硬件内核优化路径不同，但目标一致。

**㉔ HarMoEny（arXiv:2506.12417）**
多 GPU 高效 MoE 推理方案，针对多 GPU 场景优化专家并行。

**㉕ Expert Streaming（arXiv:2603.27624）**
通过多 chiplet 架构和动态专家轨迹调度加速低 batch MoE 推理。与 DWDP 的 NVLink 权重流传输有相似的"专家流式获取"思路。

**㉖ Shift Parallelism（arXiv:2509.16495）**
根据请求长度动态在 TP 与序列并行之间切换，降低推理延迟。作为并行策略自适应优化的代表，与 DWDP 的固定 DP 路线形成对比。



---

## ⚙️ 核心设计与机制

### 整体架构

DWDP 的系统架构可分为三个核心层次：

```
┌─────────────────────────────────────────────────────────┐
│                  DWDP 执行框架                            │
│                                                          │
│  GPU-0 (DP rank 0)    GPU-1 (DP rank 1)   ...           │
│  ┌─────────────────┐  ┌─────────────────┐               │
│  │ 本地专家 E0,E4  │  │ 本地专家 E1,E5  │               │
│  │ 远端专家缓冲区  │  │ 远端专家缓冲区  │               │
│  │  prefetch 队列  │  │  prefetch 队列  │               │
│  └────────┬────────┘  └────────┬────────┘               │
│           │ NVLink P2P cudaMemcpyAsync (异步)            │
│           └──────────────────────────────────           │
│  注意力层：各 GPU 独立计算，无需同步                       │
│  MoE 层：本地专家直接计算，远端专家从 peer GPU 拉取        │
└─────────────────────────────────────────────────────────┘
```

**关键设计原则**：
- 数据并行语义完整保留：每个 GPU 维护自己独立的 KV Cache、attention 计算，不与其他 rank 共享中间激活
- 权重跨 GPU 分散（Distributed Weight）：MoE 的 N 个专家均匀分配到 K 个 GPU，每 GPU 存 N/K 个
- P2P 异步拉取替代 AllGather：需要远端专家时发起 `cudaMemcpyAsync`，不阻塞本地流水

### 关键技术点 1：双缓冲异步预取（Double-Buffering Prefetch）

这是 DWDP 覆盖 P2P 通信延迟的核心机制，类似于 DMA 双缓冲：

**执行时序**（以相邻两层为例）：

```
时间轴 →

第 l 层 MoE 计算:    [MoE block-l 计算               ]
第 l+1 层注意力:                      [Attn block-(l+1)]
第 l+1 层 prefetch:  [P2P拉取远端专家 --------→]

覆盖效果：P2P 通信与 MoE 计算 + 注意力计算并行
```

具体流程：
1. 路由器（Router）在第 $l$ 层决策后，立即确定第 $l+1$ 层需要哪些远端专家
2. 通过 NVLink P2P 异步发起 `cudaMemcpyAsync` 拉取第 $l+1$ 层的缺失专家
3. 第 $l$ 层 MoE 计算 + 第 $l+1$ 层注意力计算期间，P2P 传输在后台完成
4. 第 $l+1$ 层 MoE 执行时，远端专家已在本地缓冲区就绪

**关键约束**：P2P 拉取完成必须先于第 $l+1$ 层 MoE 计算开始，否则需要插入同步点（但这种同步是局部的，针对特定专家，而非全局 barrier）。

### 关键技术点 2：消除分割权重合并开销（Eliminating Weight Merge Overhead）

**问题背景**：标准 groupedGEMM（分组矩阵乘法）期望专家权重在连续内存中。但 DWDP 中，本地存储的专家权重和从远端 P2P 拉取的专家权重存放在不同缓冲区，若执行前先合并则引入额外的内存拷贝开销。

**解决方案**：扩展 groupedGEMM 内核，使其能够**直接接受多个离散权重缓冲区的指针数组**，在计算时隐式处理多缓冲区，跳过显式合并步骤。

这一改动将专家权重的离散存放对算子完全透明，消除了 O(Expert_size) 的额外拷贝开销。

### 关键技术点 3：通信-计算竞争缓解（P2P Bandwidth Multiplexing）

**问题**：在 DWDP group 内，多个 GPU 同时发起对不同 peer 的 P2P 传输时，会产生 NVLink 带宽竞争（contention），导致大型权重传输互相阻塞。

**解决方案**：**时间分割多路复用（Time-Division Multiplexing, TDM）**

核心思路：将大型专家权重传输切片（slice）成小块，以**轮询（round-robin）方式**跨 destination rank 调度发送顺序：

```
传统方式（顺序发送）：
[rank-0 全量 → rank-1 全量 → rank-2 全量] —— 串行，存在竞争

TDM 方式（交错切片）：
[rank-0 slice-1] [rank-1 slice-1] [rank-2 slice-1]
[rank-0 slice-2] [rank-1 slice-2] [rank-2 slice-2]
...
—— 各 rank 的传输被均匀交织，NVLink 带宽利用更平滑
```

该策略的本质是**将突发式大块传输转化为细粒度流式传输**，减少单个 P2P 通道的峰值带宽占用，从而降低竞争。

### 核心公式

**同步等待时间建模**：令 $T_{\text{sync}}$ 为一轮集体同步的等待时间，则：

$$T_{\text{sync}} = \max_{i}(T_{\text{compute}}^{(i)}) - \frac{1}{N}\sum_{i=1}^{N} T_{\text{compute}}^{(i)}$$

即等待时间 = 最慢 rank 完成时间 - 所有 rank 的平均完成时间。当序列长度分布的变异系数（CV）为 20% 时，该值约占总计算时间的 12%，这是 DWDP 希望消除的主要开销。

**DWDP 通信量估算**：对于 MoE 模型，设每个专家参数量为 $E_{\text{size}}$，每 GPU 每层需拉取 $k$ 个远端专家（$k \leq N \cdot (1 - 1/K)$，N 为每 token 激活专家数，K 为 DWDP group 大小），则单层 P2P 通信量为：

$$C_{\text{P2P}} = k \cdot E_{\text{size}} \cdot \text{BatchSize}_{remote}$$

通过双缓冲将 $C_{\text{P2P}}$ 与计算时间 $T_{\text{compute}}$ 重叠，理想情况下通信开销可被完全隐藏。

### 灵活专家放置（Flexible Expert Placement）

DWDP 不要求专家总数被 DWDP group 大小整除，支持不均匀的专家分配策略。这允许在实际部署中针对不同层的路由偏斜特性调整各 GPU 的专家持有量，为后续负载感知的静态专家重分配留下接口。

---

## 📊 实验与性能评价

### 实验设置

| 维度 | 配置 |
|------|------|
| 硬件 | GB200 NVL72（72 × Blackwell GPU，900GB/s NVLink 带宽）|
| 模型 | DeepSeek-R1（671B MoE，NVFP4 量化）|
| 框架 | TensorRT-LLM（DWDP 通过 PR #12136 集成）|
| 基线 | DEP（数据并行 + 专家并行，ADP + EP 组合）|
| 请求配置 | 8K 输入 / 1K 输出序列，在线服务模式 |
| 性能指标 | TPS/GPU（吞吐量）、TPS/user（用户体验）、TTFT（首 token 时间）|

### 主要实验结果

#### Prefill（上下文计算）阶段性能

| 场景 | TPS/GPU 加速比 | TTFT 加速比 |
|------|--------------|------------|
| 8K 均匀序列（CV=0） | 1.09× | 1.11× |
| 8K 序列（CV=0.5） | 1.12× | 1.18× |
| 序列长度 std=4096 | 1.15× | 1.27× |

**关键发现**：不均衡程度越高，DWDP 的收益越显著。当序列长度标准差从 0 增至 4096 时，TPS/GPU 加速比从 1.09× 提升至 1.15×，印证了论文关于"消除同步等待"的核心命题。

#### 端到端（Prefill + Decode）性能

| TPS/user 范围 | TPS/user 加速比 | TPS/GPU 加速比 |
|--------------|---------------|---------------|
| 20-30 TPS/user | 1.15× | 1.10× |
| 40-100 TPS/user | 约 1.05-1.08× | 约 1.03-1.05× |
| >170 TPS/user | <1（回退）| <1（回退）|

端到端结果（8K in / 1K out，20-100 TPS/user 综合）：**8.8% TPS/GPU 提升**（论文摘要给出的标题数字）。

#### Decode 阶段分析

Decode 阶段 DWDP 的收益低于 Prefill，主要原因：
- Decode 是内存带宽瓶颈（访问 KV Cache 为主），而非计算瓶颈
- Decode batch 通常较小，MoE 路由较均匀，同步等待本身较少
- 当 TPS/user 超过 170 时，高批量场景下 P2P 带宽竞争大于节省的同步开销，出现性能回退

### 主要干扰源分析

论文通过实验排除了三个潜在干扰因素：

1. **内存带宽竞争**（排除）：P2P 传输占用的 NVLink 带宽与 MoE 权重读取的 HBM 带宽不共享，因此二者不存在竞争
2. **NVLink 通信竞争**（排除）：TDM 调度使 NVLink 利用率平滑，峰值带宽不超限
3. **功率诱导的频率节流**（确认为主要干扰）：GB200 在高负载时 GPU 主频会因功率预算限制而降频，这是实测数字低于理论估算的主要原因

### 对比基线（DEP）详情

DEP = Data Parallelism + Expert Parallelism，是 NVIDIA TRT-LLM 现有的高性能 MoE 推理基线。DEP 在每层 MoE 计算前后执行 AllGather（汇聚所有 rank 的 token-expert 分配信息）。

### 局限性 (Limitations)

1. **高吞吐场景回退**：TPS/user > 170 时 P2P 带宽竞争超过同步节省，性能不如 DEP
2. **仅适用于 NVLink 高带宽平台**：依赖 NVLink 的低延迟 P2P 传输，PCIe 互联环境下效果未验证
3. **功率节流限制**：GB200 的频率动态调节降低了实测收益上限
4. **仅在 Prefill 阶段效果显著**：Decode 阶段收益有限，端到端收益受 Decode 占比稀释
5. **暂无独立开源实现**：代码在 TensorRT-LLM 闭源仓库中集成（PR #12136），社区复现受限



---

## 💡 启发与我的想法

### 可借鉴之处

**1. "保留执行语义，改变通信拓扑"的设计思路**

DWDP 最优雅之处在于它**不改变每个 GPU 的计算模式**（还是标准数据并行），只改变了权重的存放位置和获取方式。这是一种"最小侵入"的优化范式：只要硬件互联带宽足够（NVLink），就可以把显存压力和通信开销分离处理。

这个思路对 KV Cache 管理也有启发——类似于 Mooncake 把 KV Cache 从 GPU 卸载到 DRAM/SSD，DWDP 把专家权重的"持有"与"使用"分离，按需流式获取。

**2. 不均衡越严重、优化空间越大**

论文明确量化了这一规律：CV=0 时加速 1.09×，CV=0.5 时加速 1.12×，std=4096 时加速 1.15×。这给了我们一个重要的工程直觉：**针对工作负载不均衡的优化，其价值与不均衡程度强相关**。如果线上服务的实际序列长度分布较集中（CV 接近 0），DWDP 的收益会很有限；反之在长尾分布场景下收益显著。

**3. 双缓冲异步预取的通用性**

双缓冲（double buffering）是一个在多个层次都适用的经典模式：
- GPU 显存 ↔ HBM（如分层 KV Cache 的 prefetch）
- 远端 GPU 内存 ↔ 本地 GPU 内存（DWDP 的专家预取）
- 磁盘 ↔ DRAM（数据加载 pipeline）

在设计推理系统时，凡是遇到"数据获取-数据使用"的串行依赖，都应优先考虑能否通过预测下一步需求来实现双缓冲覆盖。

**4. 功率节流是真实系统的隐藏变量**

论文诚实地指出功率诱导的频率节流是性能损失的主要干扰。这提醒我们：GPU 的峰值性能（FLOPS、带宽）在持续高负载下通常无法达到，实际系统的性能分析必须在真实功率限制下进行，而非基于理论峰值。

### 改进空间

**1. 动态 DWDP group 配置**

目前 DWDP 采用固定的 group 大小和专家分配。可以考虑：
- 根据请求队列的实时不均衡程度（CV 监控）动态决定是否启用 DWDP 模式
- 在低不均衡场景（CV ≈ 0）回退到 DEP，在高不均衡场景切换到 DWDP
- 类似 Shift Parallelism 的思路：运行时动态切换并行策略

**2. 与 KV Cache 管理的联合优化**

DWDP 和 KV Cache 管理（如 Mooncake、Preble）目前是相互独立的。联合优化方向：
- 若某些序列命中 prefix cache（KV Cache 复用率高），则该序列的 Prefill 开销极低，可能不值得为其牺牲 P2P 带宽做专家预取
- 反之，KV Cache miss 率高的新请求对 Prefill 速度更敏感，应优先保障其专家预取带宽

**3. 专家热度感知的静态预分配**

DWDP 目前是纯动态按需拉取。对于路由偏斜稳定（某些专家长期是热门专家）的情况，可以：
- 分析历史路由日志，识别热门专家
- 将热门专家在所有 GPU 上都存一份（副本策略），冷门专家按 DWDP 分散
- 类似于 CRAFT（arXiv:2603.28768）的成本感知副本分配思路与 DWDP 的结合

**4. 跨节点扩展**

当前 DWDP 主要面向 NVL72 单机内的 NVLink P2P。对于跨节点（InfiniBand）场景，P2P 延迟大幅增加，需要更激进的预取窗口或更大的本地 cache 来补偿延迟。这是 DWDP 向 "超大规模 MoE 推理集群" 扩展需要解决的核心问题。

### 下一步 Action
- [ ] 阅读 DWDP 的上游 PR（TensorRT-LLM #12136），了解具体实现细节
- [ ] 对比 Shift Parallelism（arXiv:2509.16495）的动态并行切换策略与 DWDP 的静态 DP 策略，分析各自适用场景
- [ ] 调研 CRAFT（arXiv:2603.28768）专家副本分配，考虑与 DWDP 按需拉取的混合策略
- [ ] 关注 DWDP 在 PCIe 互联（非 NVLink）环境下的效果评估（若有后续论文）


