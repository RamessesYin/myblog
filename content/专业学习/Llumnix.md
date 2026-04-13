---
title: "Llumnix：动态请求调度"
tags:
  - LLM推理
  - 请求调度
  - KV Cache
  - 系统优化
  - OSDI
---

## 🧠 核心摘要 (TL;DR)
> Llumnix 将操作系统"进程迁移"的思想引入 LLM 推理服务，通过跨实例**运行时动态重调度**请求（连同其 KV Cache 状态），在不修改模型的前提下统一解决负载均衡、内存碎片、请求隔离与优先级差异化四类问题，尾延迟降低一个数量级，高优先级请求加速最高 1.5×，成本降低 36%。

## 🏷️ 元数据
- **论文标题**: Llumnix: Dynamic Scheduling for Large Language Model Serving
- **发表机构/会议**: OSDI 2024（阿里巴巴）
- **年份**: 2024
- **链接**: [[Llumnix_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #LLM推理 #请求调度 #KV-Cache #系统优化 #负载均衡
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[ZipServ]]
- **相关节点 (Related Notes)**: [[MSched]], [[ShiftParallelism]], [[ZipServ]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

LLM 推理服务通常将多个模型实例（instance）并行部署以扩展吞吐量。然而，每个实例独立处理分配到的请求，导致以下四类核心问题：

1. **负载不均衡（Load Imbalance）**：LLM 请求的输入/输出长度高度异构且不可预测——短请求瞬间完成，长请求长期占用 GPU 内存。一旦某个实例"运气不好"分配到大量长请求，其队列便会显著落后于其他实例，造成整体服务延迟分布极度不均。即便使用最短队列（Shortest-Queue）等动态路由策略，因为实际排队时间无法在分发时预知，负载仍然难以真正平衡。

2. **内存碎片（Memory Fragmentation）**：多个实例同时增长各自的 KV Cache，会引发跨实例的内存碎片。由于每个实例独立管理自身的 KV Cache 池，空间利用率无法全局最优化，导致某些实例提前触发 OOM 抢占（preemption）而另一些实例内存充裕。

3. **请求干扰（Request Interference）**：延迟敏感型请求（如在线对话）与吞吐优先型请求（如批量文档处理）被混合调度到同一实例，互相干扰，无法分别优化 SLO。

4. **优先级无法差异化（Priority Differentiation）**：不同 SLA 等级的用户请求（如付费用户 vs 免费用户）若落到同一实例，高优先级请求无法被加速服务，低优先级请求也无法被降级以让路。

### 根本原因与论文切入点

上述问题的根本原因在于：**一旦请求被分发到某个实例，就无法再被迁移**。请求调度在分发时确定，之后固定不变。面对请求运行时特征的高度动态性，静态分发策略天然存在信息滞后问题。

类比操作系统的思路：现代 OS 在进程运行时会持续监控 CPU/内存负载，并通过进程迁移（process migration）或上下文切换动态重分配算力。Llumnix 将这一思想迁移到 LLM 推理服务场景——将**运行中的请求连同其 KV Cache 状态**跨实例迁移，从而在运行时动态均衡负载。

### 相关工作与 Llumnix 的定位

**先驱工作 — 推理系统基础设施**：

- **vLLM (Kwon et al., SOSP'23)**：提出 PagedAttention，通过 OS 分页机制管理 KV Cache，解决了单实例内的内存碎片问题，实现 2-4× 吞吐提升。Llumnix 以 vLLM 为构建基础，将其扩展到跨实例的动态调度场景。
- **Orca (Yu et al., OSDI'22)**：提出 iteration-level 连续批处理（continuous batching），使推理系统可以动态合并/拆分批次。Llumnix 的迁移机制需要在 iteration 级别进行状态传输，与 Orca 批处理策略深度兼容。
- **FastServe (Wu et al., 2023)**：提出 Skip-Join Multi-Level Feedback Queue 调度，通过抢占（preemption）解决 head-of-line blocking。Llumnix 的优先级调度场景与之互补，但采用的是跨实例迁移而非单实例内抢占。

**直接相关系统 — 多实例调度**：

- **AlpaServe (Li et al., OSDI'23)**：使用模型并行实现跨设备统计复用，针对多模型部署场景。Llumnix 针对单模型多实例场景，机制不同。
- **Infinite-LLM (Lin et al., 2024)**：通过分离 Attention 层与其他层、池化 GPU 内存，实现长上下文服务。与 Llumnix 的跨实例请求迁移思路有交集，但系统目标不同（长上下文 vs. 动态调度）。
- **DistServe (Zhong et al., 2024)**：将 prefill 与 decode 阶段分离到不同硬件，消除两阶段干扰。Llumnix 与 DistServe 正交，可以在 DistServe 架构之上再叠加动态请求迁移。

**近期 Follow-up — 解聚合与缓存优化**：

- **Mooncake (Qin et al., 2024)**：Moonshot AI 将 KV Cache 池化，实现 prefill/decode 解聚合。与 Llumnix 类似地利用 KV Cache 的可迁移性，但重点在缓存复用而非负载均衡。
- **Sarathi-Serve (Agrawal et al., 2024)**：通过 chunked-prefill + stall-free scheduling 解决 prefill 与 decode 的延迟竞争，与 Llumnix 的批调度机制互补。
- **SGLang (Zheng et al., 2023)**：提出 RadixAttention 实现 KV Cache 跨请求复用，提升吞吐 6.4×。在 KV Cache 管理层面与 Llumnix 的迁移机制存在协同空间。
- **LoongServe (Wu et al., 2024)**：弹性序列并行（ESP），动态调整不同请求和阶段的并行度，尾延迟减少。与 Llumnix 的动态调度理念一致，但切入点在并行度而非实例间迁移。
- **MemServe (Hu et al., 2024)**：弹性 MemPool 统一管理分布式 KV Cache，实现上下文缓存复用。进一步扩展了 Llumnix 的内存管理思路。
- **Helix (Mei et al., 2025)**：异构 GPU 集群上用最大流建模 LLM 推理调度，实现吞吐 3.3× 提升。代表了 Llumnix 之后系统调度的进一步演化方向。

**推理优化综述**：

- **Survey on Efficient Inference (Zhou et al., 2024)**：全面梳理数据级、模型级、系统级的 LLM 推理优化方向，将 Llumnix 所属的系统级调度定位在整体优化图谱中。


## ⚙️ 核心设计与机制

### 总体架构

Llumnix 引入一个**全局调度器（Global Scheduler）**，凌驾于各模型实例之上，持续观察所有实例的运行状态，并在必要时触发**请求迁移（Request Migration）**。系统架构由三个核心组件构成：

```
┌─────────────────────────────────────────────────────┐
│                   Llumnix Global Scheduler           │
│  ┌─────────────────┐   ┌──────────────────────────┐ │
│  │  Migration Trigger│   │  Scheduling Policy Engine│ │
│  │  (when to migrate)│   │  (where to migrate)      │ │
│  └────────┬────────┘   └────────────┬─────────────┘ │
└───────────┼────────────────────────┼────────────────┘
            │ monitor                │ command
     ┌──────▼──────┐         ┌───────▼──────┐
     │  Instance A  │ ←migrate→  │  Instance B  │
     │  (vLLM-based)│         │  (vLLM-based)│
     └─────────────┘         └──────────────┘
```

**Llumlet**：每个实例上运行的本地代理（local agent），负责执行迁移指令、上报状态，并对本地的 KV Cache 进行序列化/反序列化。

### 关键技术点 1：高效 KV Cache 迁移机制

请求迁移的核心挑战在于**迁移 KV Cache 的代价**。一个长请求的 KV Cache 可能占用数 GB 显存，若同步传输会导致请求响应停顿（stall），用户感知到明显的延迟尖刺。

Llumnix 提出**流水线式异步迁移（Pipelined Migration）**：

1. **迁移触发**：Scheduler 决定将实例 A 上的请求 $r$ 迁移到实例 B。
2. **KV Cache 流式传输**：发送方（实例 A）将请求当前已生成的 KV Cache 层 $[0, L-1]$ 通过 GPU-to-GPU 传输（NVLink 或 PCIe）**后台异步**发送给接收方（实例 B），与此同时实例 A **继续执行**该请求的下一个 iteration。
3. **增量同步**：在迁移传输过程中，实例 A 新生成的 KV Cache 层被追加传输（delta sync），直到 delta 足够小（通常 1-2 层）时进行最终切换（final handoff）。
4. **最终切换**：实例 A 将最后几层 KV Cache 发送给 B，并通知 Scheduler 迁移完成；B 接管该请求，从断点处继续生成 token。

这一设计的关键性质：**迁移对用户几乎透明**，因为请求在迁移过程中从未停止推进，只在 final handoff 阶段有极短的暂停（约 1-2 个 iteration 时间）。

### 关键技术点 2：调度策略统一框架

Llumnix 的调度策略以**请求迁移**作为统一原语，将四类问题映射为迁移触发条件：

| 问题 | 迁移触发策略 |
|------|------------|
| **负载均衡** | 当某实例队列等待时间显著超过其他实例均值时，将头部请求迁出到最空闲实例 |
| **内存碎片** | 当某实例剩余内存不足以容纳新 KV Cache 增长时，将部分请求迁出以整理内存 |
| **请求隔离** | 将延迟敏感请求（SLO tight）与吞吐优先请求分类，路由到不同实例集群，避免干扰 |
| **优先级差异** | 当高优先级请求入队时，将低优先级请求迁出，为高优先级请求腾出实例资源 |

**调度器状态模型**：Scheduler 维护每个实例的以下状态：
- $Q_i$：当前队列中的请求数（包括等待和执行中）
- $M_i$：已使用 KV Cache 内存量
- $T_i^{wait}$：队首请求的预计等待时间（基于历史统计）
- Priority bitmap：各优先级请求的分布

调度器周期性地（默认每 iteration 一次）计算跨实例的**负载不均衡度**，当不均衡超过阈值时触发迁移。

### 关键技术点 3：迁移感知批处理

在执行迁移的过程中，实例 B 可能正在进行正常的 iteration，需要将即将接收的请求无缝插入当前 batch。Llumnix 对 vLLM 的调度器进行了扩展：

- 接收方在收到 KV Cache 的同时，即将对应请求加入**预接收队列（Pre-admission Queue）**
- 在下一个 iteration 启动时，预接收队列中的请求与本地队列合并，参与正常的 continuous batching
- 迁移过程中的 token 生成不间断，无需等待完整 KV Cache 传输完毕

### 核心公式

迁移收益函数（用于判断是否值得迁移）：

$$
\Delta \text{Benefit}(r, A \to B) = \underbrace{T_A^{wait}(r)}_{\text{迁移前等待时间}} - \underbrace{T_B^{wait}(r)}_{\text{迁移后等待时间}} - \underbrace{C_{migrate}(r)}_{\text{迁移开销}}
$$

其中迁移开销 $C_{migrate}(r)$ 主要由 KV Cache 大小（请求当前生成长度 $\times$ 层数 $\times$ 数据类型字节数）决定：

$$
C_{migrate}(r) = \frac{|KV(r)|}{BW_{GPU \to GPU}}
$$

仅当 $\Delta \text{Benefit} > 0$ 时才触发迁移，防止过度迁移带来的额外开销。


## 📊 实验与性能评价

### 实验设置

- **硬件**：多台配备 A100/A800 GPU 的服务器，通过 NVLink/PCIe 互联
- **基线对比**：
  - vLLM（静态路由，最短队列策略）
  - Orca 风格系统
  - Triton Inference Server（商业系统）
- **模型**：LLaMA-2（7B/13B/70B）、OPT 系列
- **工作负载**：基于 ShareGPT 真实对话数据集（输入/输出长度呈重尾分布，与真实服务场景吻合）

### 主要指标与结果

**1. 负载均衡（Load Balancing）**

在异构请求到达场景（突发长请求混合短请求）下：

| 指标 | vLLM (Shortest Queue) | Llumnix |
|------|----------------------|---------|
| P99 延迟（尾延迟） | 高（一个数量级以上） | 降低约 **10×** |
| P95 延迟 | 较高 | 显著下降 |
| 平均延迟 | 基线 | 相近（不牺牲平均延迟） |

核心发现：静态路由在请求运行时特征不可预知的情况下无法消除负载不均，而 Llumnix 动态迁移可以"事后纠错"，将尾延迟压缩到接近理想值。

**2. 优先级差异化（Priority Differentiation）**

在高/低优先级请求混合提交的场景：

| 场景 | 高优先级请求加速 |
|------|----------------|
| 低优先级队列挤压 | 最高 **1.5×** 响应加速 |
| 低优先级抢占迁移 | 延迟隔离效果显著 |

**3. 内存碎片整理（Defragmentation）**

在 KV Cache 长期占用导致碎片严重时：

- Llumnix 主动迁移高内存占用请求，使内存利用率维持在 85%+ 水位
- 相比 vLLM 的被动 OOM 抢占，系统吞吐下降幅度减小 **36%**

**4. 迁移开销分析**

- 对于 1024 token 的请求（典型长度），KV Cache 大小约 500MB
- GPU-to-GPU（NVLink）传输时延：约 50-100ms
- 迁移引入的额外 TTFT（Time to First Token）影响：< 5%（通过流水线几乎掩盖）

**5. 综合成本节省**

在真实云服务场景（混合优先级、异构请求长度）下：

> 与当时最先进系统相比，**成本降低 36%**（在满足 SLO 约束的前提下，可服务更多请求）

### 局限性 (Limitations)

1. **迁移带宽依赖**：在 PCIe 互联（相比 NVLink 带宽低 5-10×）的场景下，KV Cache 迁移延迟增大，收益可能打折。
2. **短请求效果有限**：对于极短请求（<128 token），KV Cache 体积小，静态路由的负载不均影响本身不大，迁移的边际收益有限。
3. **单模型多实例假设**：当前设计针对同一模型的多个实例。跨模型、跨架构的调度（如同时服务 7B 和 70B）超出当前设计范围。
4. **调度器单点**：全局调度器若成为瓶颈或发生故障，会影响整体系统。论文中对调度器的高可用设计着墨不多。


## 💡 启发与我的想法

### 可借鉴之处

1. **"OS 思维"迁移**：将 OS 进程调度的动态迁移思想搬运到 LLM 推理服务，是一种非常优雅的类比驱动设计。KV Cache 即"进程状态"，实例间迁移即"上下文切换"。这种跨领域类比为 LLM 系统设计提供了丰富的思路来源——OS 领域还有哪些机制（内存换页、CPU 亲和性、NUMA 感知调度等）可以被类似地迁移？

2. **流水线迁移的"透明性"设计**：迁移期间请求不停止推进这一设计，是将系统级优化对用户透明化的典型范式。未来在推理系统设计中，凡是需要状态迁移的操作（如 KV Cache offloading 到 CPU/SSD），都可以借鉴这种"边迁移边服务"的流水线化思路。

3. **统一原语（Unified Primitive）**：用"请求迁移"这一单一原语统一解决四类异构问题，体现了良好的系统抽象能力。类比于 vLLM 用"PagedAttention"统一解决 KV Cache 碎片问题，优秀的系统论文往往能找到这样的"通用支点"。

4. **收益函数量化**：显式建模迁移收益与开销，并只在收益正的情况下迁移，防止了调度器的过激行为（thrashing）。这种保守迁移策略对实际工程实现很有参考价值。

### 改进空间

1. **与 prefill/decode 解聚合的深度融合**：Llumnix 当前将整个请求（含 prefill 和 decode）作为迁移单元。在 DistServe/Mooncake 等 prefill-decode 解聚合架构下，decode 阶段的请求迁移可能需要同时协调 prefill 集群的 KV Cache 分布，设计复杂度更高，是值得探索的方向。

2. **预测式迁移**：当前 Llumnix 是**响应式**迁移（检测到不均衡后才迁移）。若能预测请求的剩余生成长度（如用轻量级输出长度预测模型），可以更早地触发迁移，进一步降低尾延迟。

3. **SSD/CPU 分层迁移**：在 GPU 内存不足时，将低优先级请求的 KV Cache 迁移到 CPU DRAM 或 NVMe SSD（而非另一个 GPU 实例），可以在更低成本下实现隔离，对于混合算力的云环境尤为适用。

4. **多模型多实例调度**：在实际生产中，往往需要同时服务多个版本的模型（7B/13B/70B），且不同模型的实例数可以动态伸缩。Llumnix 的调度框架如何扩展到这种"多模型异构实例"场景，是一个开放问题。

### 下一步 Action
- [ ] 阅读 DistServe（2401.09670）：了解 prefill/decode 解聚合的详细设计，与 Llumnix 的联系点在哪里
- [ ] 阅读 Mooncake（2407.00079）：KV Cache 池化方案与 Llumnix 迁移机制的对比
- [ ] 思考：Llumnix 的迁移触发策略是否可以用强化学习来自动学习最优策略（当前是启发式阈值）



