---
title: "CloudMatrix384"
tags:
  - 华为
  - Ascend-910
  - LLM推理
  - MoE
  - 系统优化
  - 专家并行
---

## 🧠 核心摘要 (TL;DR)

> 华为 CloudMatrix384 是一个将 384 块 Ascend 910 NPU 通过超高带宽统一总线（UB）全对等互联的超节点平台。本文提出 CloudMatrix-Infer 推理系统，三大创新分别是：**对等 PDC 分离架构**（Prefill/Decode/Cache 独立子集群）、**EP320 大规模专家并行**（解码集群使用 320 个 NPU die 做全量专家并行）、以及针对 Ascend 硬件的深度优化（AIV-Direct 通信、早期 INT8 量化、MLA 算子融合、MTP 流水线）。在 DeepSeek-R1 671B 上，预填充计算效率达 4.45 tokens/s/TFLOPS，解码效率 1.29 tokens/s/TFLOPS，**在 tokens/s/TFLOPS 这一核心效率指标上全面超越 H800 和 H100 基线**。

## 🏷️ 元数据
- **论文标题**: Serving Large Language Models on Huawei CloudMatrix384
- **发表机构/会议**: 华为 / SiliconFlow @ arXiv (cs.DC)
- **年份**: 2025（v3 更新于 2025-06-19）
- **链接**: [[CloudMatrix384_paper|📄论文]]

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #MoE #专家并行 #LLM推理服务 #KV-Cache #Ascend-NPU #PDC分离 #MLA优化 #INT8量化
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[Continuum]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[ShiftParallelism]]
- **向下延申 (Successors)**: 可关注 UB-Mesh 网络架构论文（arXiv:2503.20377）和 Pangu Ultra MoE（arXiv:2505.04519）

---

## 🎯 动机与痛点

### MoE 模型推理的通信墙

DeepSeek-R1/V3（671B，256 路由专家，Top-8 激活）的推理面临与稠密模型截然不同的性能瓶颈。每个 decode 步骤的 MoE 层需要经历三次全量 All-to-All 通信（路由元数据分发 → token 数据 Dispatch → 专家输出 Combine），且由于每个 token 只激活 8/256 个专家，token 必须被分发到不同的 GPU/NPU 上计算后再汇聚回来。在传统 GPU 集群（NVLink/IB 互联）中，这一通信开销会随着专家并行度（EP度）的提升而线性增长，成为制约吞吐量的根本瓶颈。

### 标准推理系统的三个问题

1. **通信与计算无法有效重叠**：H800 上的 DeepEP 实现在 EP256 时 Dispatch 延迟 194 µs、Combine 延迟 360 µs，成为每个 Transformer 层的显著开销
2. **KV Cache 管理割裂**：Prefill 和 Decode 混在同一 GPU 池中竞争资源，缓存数据通过高延迟网络传输
3. **软件算子未针对异构核优化**：Ascend 910 拥有 AIC（AI Cube Core，矩阵计算）和 AIV（AI Vector Core，向量计算）两类异构核心，直接移植 CUDA 算子无法发挥硬件潜力

### 相关工作对比

| 系统 | 代表论文 | 核心方案 | 与本文差异 |
|------|---------|---------|-----------|
| **vLLM** | Kwon et al., SOSP'23 | PagedAttention KV 内存管理 | 无 PDC 分离，MoE 通信无专项优化 |
| **SGLang** | Zheng et al., MLSys'24 (arXiv:2312.07104) | RadixAttention 前缀复用 + EP 调度 | 同平台 H100 对比基线，预填充效率 3.75 tok/s/TFLOPS |
| **DistServe** | Zhong et al., OSDI'24 | Prefill/Decode 分离调度 | 仅分离调度，无统一高带宽内存池 |
| **Mooncake** | Qin et al., FAST'25 (arXiv:2407.00079) | KVCache 中心化分离架构 | 面向 GPU 集群，依赖 DRAM 分层存储 |
| **MegaScale-Infer** | Zhu et al., arXiv:2504.02263 | 分离式专家并行 | GPU 平台，关注跨节点 EP 调度 |
| **Continuum** | Li et al., arXiv:2511.02230 | 多轮 Agent KV Cache TTL | 聚焦 Agent 调度层，不涉及硬件优化 |

---

## ⚙️ 核心设计与机制

### 一、硬件基础：CloudMatrix384 超节点

**物理规格**：384 Ascend 910 NPU + 192 Kunpeng CPU，共 16 个机架（12 计算 + 4 通信），通过三层网络互联：

| 网络平面 | 协议 | 用途 | 关键指标 |
|---------|------|------|---------|
| **UB 平面** | 自研统一总线 | 超节点内全对等互联，TP/EP通信，内存池访问 | 跨节点带宽衰减 <3%，延迟增加 <1 µs |
| **RDMA 平面** | RoCE | 跨超节点 KV Cache 传输 | 标准 RDMA 带宽 |
| **VPC 平面** | 以太网 | 数据中心管理/存储 | 标准网络 |

Ascend 910 每块芯片（die）包含 **24 个 AIC 核**（专做矩阵乘法，对应 NVIDIA 的 Tensor Core）和 **48 个 AIV 核**（向量运算，对应 CUDA Core），两类异构核可并行执行不同任务，是后续流水线优化的硬件基础。

### 二、PDC 对等分离架构

Continuum 仅在调度层做 Prefill/Decode 分离，CloudMatrix-Infer 更进一步，在**物理层面**将三类工作负载部署为独立集群，并通过 UB 平面实现低延迟内存级互联：

```
Prefill 集群（96 NPU）  ←→  Caching 集群（EMS，池化 CPU DRAM）
         ↓ RDMA KV传输
Decode 集群（160 NPU）
```

**部署规格（本文实验配置）**：

| 集群 | NPU 数量 | NPU Die 数量 | 并行策略 | 专家配置/rank |
|------|---------|-------------|---------|-------------|
| Prefill（6实例） | 96（16/实例） | 192（32/实例） | EP32 | 10（1共享+8路由+1冗余） |
| Decode（1实例） | 160 | **320** | **EP320** | 1（32共享副本+256路由+32冗余） |

EP320 是本文最激进的创新——整个解码集群只有 320 个 NPU die，每个 die 上只放 **1 个专家**，将 all-to-all 通信范围扩展到整个集群内。这仅在 UB 平面全对等低延迟互联的条件下才可行，在传统 IB 网络上会因拓扑不对称导致延迟急剧上升。

### 三、LEP 融合通信算子（Large-scale Expert Parallelism）

针对 EP320 下 MoE 通信的三项关键优化：

**① AIV-Direct：绕过 SDMA 的低延迟直写**

传统路径：AIV 核 → SDMA 引擎（启动开销）→ UB 网络 → 目标 NPU
优化路径：AIV 核 **直接写** → UB 网络 → 目标 NPU 内存

消除 SDMA 的启动延迟，对解码阶段（每层延迟约 600–900 µs）影响显著。

**② 早期 INT8 量化（Early Quantization）**

在 Dispatch 前将 BF16 token（7168 维）量化为 INT8，单 token 消息体积：

$$\text{BF16 消息大小} = 7168 \times 2 = 14,336 \text{ B} \approx 14.3 \text{ KB}$$
$$\text{INT8 消息大小} = 7168 \times 1 + 512 \text{（scale对齐）} = 7,680 \text{ B} \approx 7.5 \text{ KB}$$

**通信数据量减少约 47.5%**，直接降低 All-to-All 通信的带宽压力。

**③ 静态预分配缓冲区 + 双缓冲**

预先为 FusedDispatch 和 FusedCombine 分配固定大小缓冲区（容量 = rank数 × max_tokens × 消息大小），消除动态内存分配的 CPU 开销，双缓冲避免数据竞争。

### 四、MLA 算子融合优化

DeepSeek 的多头潜在注意力（MLA）通过低秩压缩大幅减少 KV Cache，但在 Ascend 910 上直接运行存在三个性能问题，本文针对性地提出了解决方案：

| 问题 | 根本原因 | 解决方案 |
|------|---------|---------|
| 细粒度算子启动开销 | RMSNorm/线性投影/RoPE 各自独立 kernel 调用 | **MLAProlog 融合算子**：将注意力前所有操作融合，AIC-AIV 流水化 |
| KV Cache 格式转换 | AIC 最优格式为 NZ，内存存 ND 需转换 | **原生 NZ 格式存储**：写入时在线转换，消除独立转换步骤 |
| MTP 场景负载不均 | BNSD 布局按 N/D 切分 tiling，MTP 时序列长不等 | **BSND 布局 + 动态 tiling**：沿 B/S 轴自适应划分 |

**效果**：MLA 算子在计算密集型场景下达到 **65.4% TFLOPS 利用率**（vs H800 FlashMLA 66.7%），内存密集型场景 **84.1% 带宽利用率**（vs H800 89.6%），两平台基本持平。

### 五、Microbatch 解码流水线（异构核双流）

EP320 下每 NPU die 只有 1 个专家，没有共享专家计算来掩盖通信延迟（与 H800 不同）。本文设计了非对称 AIC/AIV 资源分配的双流流水线：

| Stream | 包含操作 | 资源分配 |
|--------|---------|---------|
| Stream 0（注意力路径） | MLAProlog + FusedAttention + O_PROJ | **16 AIC + 32 AIV** |
| Stream 1（MoE 路径） | Gate + Dispatch + MLP + Combine | 8 AIC + 16 AIV |

两个流各处理一个 microbatch，延迟接近（~600 µs/microbatch），实现完美重叠。**效果**：batch=96 时解码吞吐量提升 **+9.4%**，每层延迟降低约 10%。

### 六、MTP 无 CPU 流水线

DeepSeek-R1 支持多 token 预测（MTP），但 k+1 个计算图的串行调度带来 0.6–0.8 ms/图的 CPU 介入延迟，引入 NPU 空闲气泡。本文通过两项改造消除 CPU 瓶颈：

1. **聚合元数据初始化**：解码步骤开始时一次性预计算所有 k+1 个图的元数据，存入 NPU 内存
2. **片上采样（In-NPU Sampling）**：将 token 采样（排序、累积和、过滤）全部迁移到 NPU 上执行，图与图之间无 CPU 介入

**效果**：batch=96 时吞吐量提升 **+6%–49%**（小 batch 提升更显著）；70% MTP 接受率下，每解码步平均产出 1.7 个 token，净效益为正。

### 七、EMS 分布式内存缓存

弹性内存服务（EMS）通过 UB 平面将超节点内所有 Kunpeng CPU 的 DRAM 池化为统一共享内存池，提供：
- **上下文缓存**：KV Cache 按 128–512 token 的页式块组织，哈希寻址；复用率 50% 时预填充吞吐提升 **1.42×**，90% 时提升 **2.28×**，TTFT 最多降低 **59%**
- **模型缓存**：671B INT8 模型（约 671 GB）从 OBS 冷启动需 2,560s（8 实例并发），通过 EMS 降至 **320s**；热启动（EMS→NPU）仅 **~5s**，且多实例共享单副本，DRAM 占用为本地缓存方案的 **1/8**

---

## 📊 实验与性能评价

### 预填充吞吐量对比（表2，16,384 batch，4,096 input length）

| 系统 | 原始吞吐 (tokens/s/accel) | 计算效率 (tok/s/TFLOPS) |
|------|--------------------------|------------------------|
| DeepSeek on H800 (Blog) | 4,026 | 2.03 |
| SGLang on H100 (Default) | 6,288 | 3.18 |
| **CloudMatrix-Infer (Default)** | **5,655** | **3.76** |
| DeepSeek on H800 (Profile) | 7,839 | 3.96 |
| SGLang on H100 (Perfect EPLB) | 7,417 | 3.75 |
| **CloudMatrix-Infer (Perfect EPLB)** | **6,688** | **4.45** |

### 解码吞吐量对比（表3，TPOT < 50 ms，KV Cache 4,096）

| 系统 | Batch | TPOT (ms) | 吞吐 (tok/s/accel) | 效率 (tok/s/TFLOPS) |
|------|-------|-----------|-------------------|---------------------|
| DeepSeek on H800 (Blog) | N/A | ~50.0 | 1,850 | 0.93 |
| DeepSeek on H800 (Profile) | 128 | ~50.2 | 2,325 | 1.17 |
| SGLang on H100 (Simu. MTP) | 128 | ~55.6 | 2,172 | 1.10 |
| **CloudMatrix-Infer** | **96** | **49.4** | **1,943** | **1.29** |

### 通信算子对比（表6，batch=128/rank）

| 算子 | EP度 | H800 延迟 (µs) | CM384 延迟 (µs) | CM384 带宽 (GB/s) |
|------|------|----------------|-----------------|-------------------|
| Dispatch | 256 | 194 | 152 | 54 |
| Combine | 8 | 318 | **118** | **131** |
| Combine | 256 | 360 | **149** | **103** |

### 局限性

1. **解码原始吞吐略低于高配 H800**：batch=96 时 CM384 解码 1,943 tok/s，而 H800 Profile（batch=128）达 2,325 tok/s；batch 差异使直接比较有失公平，但差距客观存在
2. **大 EP 度下 Dispatch 带宽下降**：EP8 时 Dispatch 带宽 71 GB/s，EP256 时降至 54 GB/s，大规模并行下存在可扩展性瓶颈（论文承认留待未来优化）
3. **实验仅限单超节点（256 NPU）**：跨超节点的 RDMA 互联场景性能未充分验证
4. **INT8 量化部分精度轻微下降**：如 DROP（3-shot F1）：90.42 vs API 91.02，LiveCodeBench：63.80 vs API 63.44，整体可接受但非零损失

---

## 💡 启发与我的想法

### 高效率背后的本质：通信-计算异构化

CloudMatrix-Infer 的最大贡献不是单一技术点，而是将**硬件异构性**（AIC vs AIV）、**网络拓扑优势**（UB 全对等低延迟）和**算法感知**（MoE 工作模式）三者对齐的系统性设计。特别是 AIV-Direct 旁路 SDMA、MLP 与 All-to-All 在异构核上并行这两个设计，是在清楚理解硬件底层执行模型的基础上才能做到的优化，在 CUDA 单核范式下很难有对应物。

### 与本项目其他笔记的关联

- 与 [[KvZap]] 的关系：KvZap 在单次推理中稀疏化 KV，本文 EMS 在跨请求级别缓存 KV——两者正交，可以叠加使用
- 与 [[Continuum]] 的关系：Continuum 通过 TTL 保留 GPU 内 KV Cache 来避免多轮重计算，本文 EMS 提供了更大的 CPU DRAM 池作为 KV 存储层，理论上两者可以结合形成 GPU KV（TTL保留）→ CPU EMS（中期缓存）→ SSD（长期持久化）的三层存储体系
- 与 [[ShiftParallelism]] 的关系：本文 EP320 是迄今见到的最大规模专家并行配置，其可行性完全依赖 UB 平面的全对等特性，若部署在常规 fat-tree IB 网络上，跨机架的 all-to-all 延迟会使 EP320 变得不实际

### 改进空间

1. **EPLB 负载均衡算法**：Default 与 Perfect EPLB 预填充吞吐相差 18%（5,655 vs 6,688 tok/s），提升 EPLB 的在线动态调整能力是最直接的优化空间
2. **多超节点扩展**：当前实验限于单超节点内，跨超节点（RDMA 平面）的 KV Cache 传输和 EP 通信性能需要进一步建模和优化
3. **MTP 接受率感知调度**：当前 MTP 以固定 k=1 运行，若能根据任务类型动态调整预测深度（推理类任务接受率低则关闭 MTP），可进一步降低延迟

### 下一步 Action
- [ ] 阅读 UB-Mesh 架构论文（arXiv:2503.20377），深入理解 CloudMatrix384 UB 全对等互联的硬件实现原理
- [ ] 对比 MegaScale-Infer（arXiv:2504.02263）中分离式专家并行与本文 EP320 统一 EP 的设计取舍
- [ ] 思考 Continuum TTL + EMS 三层 KV 存储的设计可行性，分析瓶颈在哪一层

---

## 📌 附：比 DeepSeek V3/H800 单位吞吐高的原因分析（专项）

> 详见下方正文的深度分析章节

CloudMatrix-Infer 在 **tokens/s/TFLOPS 计算效率**上超越 H800 的核心原因可以从三个维度归因：

### 1. 通信开销更低（Combine 延迟降低 2.4×）

H800 集群的 All-to-All 通过 InfiniBand 实现，在 EP256 规模下 Combine 算子延迟 **360 µs**；CM384 通过 UB 平面 + AIV-Direct 实现，EP256 下 Combine 延迟仅 **149 µs**（降低 59%）。更低的通信延迟意味着每个 Transformer 层中计算单元（AIC）的空闲等待时间更短，有效计算占比更高。

定量估算：假设每层解码总延迟 ~900 µs，其中通信占 Dispatch（152 µs）+ Combine（149 µs）= 301 µs，通信占比约 33%；而 H800 EP256 通信为 194+360 = 554 µs，若总延迟同规模则通信占比接近 60%。

### 2. 计算资源利用率更高（AIC/AIV 异构并行）

H800 上 MoE 通信通过 SM 并发（占用一部分 SM 做通信 DMA），与计算 SM 竞争资源；CM384 上 SDMA 专门负责大块数据传输，AIV 处理轻量元数据计算，**AIC（矩阵乘法核心）全程不受通信任务干扰**，维持高 TFLOPS 利用率（INT8 GEMM 利用率 77.4%–82.7%）。

### 3. INT8 量化的乘数效应

CM384 全程使用 INT8（含 Dispatch 前的早期量化），相比 H800 的 FP8/BF16 混合方案：
- 通信数据量减少 47.5%（每 token 7.5 KB vs 14.3 KB），进一步降低 All-to-All 带宽压力
- INT8 矩阵乘法在 Ascend 910 AIC 上的峰值算力高于 BF16（INT8 = 2× BF16 TFLOPS），分母（TFLOPS）增大而分子（tokens/s）同步提升，效率比并未摊薄

**综合结论**：CloudMatrix-Infer 的效率优势不是来自单一技术，而是**低通信延迟（UB平面）+ 计算-通信资源隔离（AIC/AIV异构）+ INT8 全链路量化**的协同增益，三者在 tokens/s/TFLOPS 这一效率比上形成叠加效应。
