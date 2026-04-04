---
title: "ThunderAgent：面向智能体工作流的程序感知推理系统"
tags:
  - agentic-inference
  - KV-cache
  - LLM-serving
  - tool-use
  - scheduling
  - reinforcement-learning
---

## 🧠 核心摘要 (TL;DR)
> 现有 LLM 推理引擎（如 vLLM）以「请求」为粒度管理资源，导致跨工具调用的 KV Cache 频繁驱逐、Docker 容器泄漏以及多轮 Agent 工作流吞吐量低下。ThunderAgent 提出「LLM Program」抽象，将整个 Agent 生命周期视为一个调度单元，通过程序感知的 KV Cache 保护、基于钩子的资源回收以及异步环境预热，在服务吞吐量上比 vLLM 提升 1.5–3.6×，在 RL Rollout 场景下提升 1.8–3.9×，并节省最多 4.2× 的磁盘内存。

## 🏷️ 元数据
- **论文标题**: ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System
- **发表机构/会议**: arXiv（Georgia Tech / CMU，2026）
- **年份**: 2026
- **链接**: [[ThunderAgent_paper|📄论文]] | [💻 Code](https://github.com/HaoKang-Timmy/ThunderAgent)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #agentic-inference #kv-cache #llm-serving #tool-use #scheduling #reinforcement-learning
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[ZipServ]], [[MSched]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[MSched]], [[RL中的KV稀疏讨论]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

当前 LLM 推理服务引擎（vLLM、SGLang 等）的设计范式以**单次请求（Request）**为核心调度粒度，面向批量无状态服务优化。然而，Agent 工作流具有根本性的不同特征：

1. **多轮交互与长上下文**：Agent 在一次任务中往往经历数十次 LLM 调用（Inference 阶段）与工具执行（Action 阶段）的交替，上下文长度随对话轮次持续增长。现有系统在工具执行期间（数秒至数十秒的等待时间）无法保留 KV Cache，迫使系统在每次工具返回后重新执行预填充（Prefill），带来 $O(c^2)$ 量级的冗余计算。

2. **工具资源泄漏**：Agent 依赖 Docker 沙箱、网络套接字、代码执行环境等外部工具资源。传统 LLM 服务引擎对工具资源毫无感知，导致「幽灵容器」和磁盘空间无限累积——论文实验显示，使用 vLLM 运行 24 小时后磁盘占用持续线性增长，最终耗尽存储。

3. **多机负载不均**：分布式场景下，不同 DP（数据并行）节点因 Agent 工作流进度不同，其 KV Cache 占用严重失衡，导致部分节点内存溢出而其他节点空闲。现有引擎缺乏跨节点的全局视野，无法动态平衡负载。

4. **RL Rollout 瓶颈**：强化学习训练需要大规模并发 Rollout（Agent 并行与环境交互），工具调用密集且异步性强。每次 Rollout 中 LLM 输出触发工具调用、等待结果、再继续推理，这种频繁的 pause-resume 模式下，现有系统的资源管理效率极低。

### 论文的切入点

ThunderAgent 认为：**现有系统的根本问题在于「请求级孤立性」**——每个 LLM 请求被独立处理，系统无法感知同一 Agent 程序跨请求的连续性与资源依赖。解决方案是引入「LLM Program」这一更高层次的抽象，为系统提供跨请求的统一视野。

### 相关工作

#### 先驱/奠基工作

| 论文 | 核心贡献 | 与本文关系 |
|------|---------|-----------|
| **vLLM / PagedAttention**（Kwon et al., SOSP'23） | KV Cache 分页管理，连续批处理（Continuous Batching），大幅提升 LLM 服务吞吐量 | ThunderAgent 的基础引擎之一，对比基线 |
| **SGLang**（Zheng et al., 2024） | 基于 RadixAttention 的 Prefix KV Cache 复用，支持结构化生成 | ThunderAgent 的基础引擎之一，支持作为后端 |
| **Sarathi-Serve**（Agrawal et al., 2024） | Chunked Prefill，解决 Prefill 阻塞 Decode 的问题 | 优化 Prefill 阶段，ThunderAgent 优化的是跨工具调用的 KV Cache 生命周期 |
| **DistServe / Mooncake**（2024） | Prefill-Decode 解耦服务，降低 TTFT 与吞吐量冲突 | 系统级优化方向相关 |

#### 直接对比方法

| 论文 | 核心贡献 | 与本文差异 |
|------|---------|-----------|
| **Continuum**（Zheng et al., 2025） | 针对 Agent 工作流的 LLM 推理系统，通过工作流感知的调度提升吞吐量 | ThunderAgent 在多数配置下以 1.17–3.31× 超越 Continuum，差异在于程序级抽象与资源管理的完整性 |

#### Agent 框架与基准

| 论文 | 核心贡献 | 与本文关系 |
|------|---------|-----------|
| **SWE-bench**（Jimenez et al., 2024） | 基于真实 GitHub Issues 的软件工程 Agent 评测基准 | 实验中 mini-SWEAgent 使用 SWE-bench 任务 |
| **LiveCodeBench**（Jain et al., 2024） | 代码生成 LLM 实时评测基准 | ThunderAgent 实验评测场景之一 |
| **OSWorld**（Xie et al., 2024） | 多模态桌面任务 Agent 评测基准 | 评估 Agent 与操作系统交互的场景 |
| **Windows Agent Arena**（Bonatti et al., 2024） | Windows 环境下的 Agent 操作基准 | 评估 Windows 平台 Agent 任务 |
| **OpenHands / OpenDevin**（Wang et al., 2024） | 开源代码 Agent 框架，支持 SWE-bench 任务 | 实验基线 Agent，ThunderAgent 相比 vLLM 基线取得 3.92× Rollout 加速 |
| **ReAct**（Yao et al., 2022） | 推理与行动交替的 Agent 范式（Reasoning + Acting） | ThunderAgent 处理的正是 ReAct 式 Inference/Action 交替工作流 |

#### RL 训练与 Rollout

| 论文 | 核心贡献 | 与本文关系 |
|------|---------|-----------|
| **Large-scale Async RL**（Fu et al., 2025） | 大规模异步 RL 训练，LLM 与环境并行交互 | ThunderAgent 的 RL Rollout 场景直接对应此类工作负载 |
| **Stabilizing RL for LLMs**（Zheng et al., 2025） | 稳定 LLM 强化学习训练的技术 | RL 训练生态系统的上下文 |

#### 目标模型

| 模型 | 机构 | 与本文关系 |
|------|------|-----------|
| **GLM-4.6**（5Team, 2025） | 智谱 AI | ThunderAgent 主要实验模型，支持 Agentic Reasoning & Coding |
| **Qwen3**（Yang et al., 2025） | Alibaba | 实验中使用的第二个测试模型 |

---

## ⚙️ 核心设计与机制

### 架构概览

ThunderAgent 定位于 **Agent 客户端与 LLM 推理引擎之间的中间层调度器**，兼容 OpenAI API（仅需在请求头中额外传入 `Program_id` 字段），支持 vLLM 和 SGLang 作为后端。其三大核心组件为：

1. **LLM Program 抽象**：统一的跨请求元数据单元
2. **程序感知调度器（Program-Aware Scheduler）**：KV Cache 生命周期管理
3. **工具资源管理器（Tool Resource Manager，TRM）**：Docker 容器/外部 API 的生命周期管理

```
┌───────────────────────────────────────────────────────────┐
│                  Agent Clients                            │
│        (mini-SWEAgent / OpenHands / Custom)               │
└────────────────────────┬──────────────────────────────────┘
                         │ OpenAI-compatible API + Program_id
┌────────────────────────▼──────────────────────────────────┐
│              ThunderAgent Scheduler                       │
│  ┌─────────────────────┐  ┌──────────────────────────┐    │
│  │ Program-Aware       │  │ Tool Resource Manager    │    │
│  │ KV Cache Scheduler  │  │ (Hook GC + Async Prep)   │    │
│  └─────────────────────┘  └──────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │   Global Program-Aware Waiting Queue (DP Unified)   │  │
│  └──────────────────────────────────────────────────────┘  │
└──────┬────────────────────────────────────────────────────┘
       │
┌──────▼────────────────────────────────────────────────────┐
│         LLM Inference Engines (vLLM / SGLang)             │
│              Multi-GPU / Multi-Node                       │
└───────────────────────────────────────────────────────────┘
```

### 关键技术点 1：LLM Program 抽象

**定义**（论文公式 1）：

$$P = \langle \text{ID}, c, T, L, \tau, s \rangle$$

各分量含义：
- $\text{ID}$：全局唯一标识符，跨请求追踪同一 Agent 任务
- $c$：当前上下文的 token 数量，即 KV Cache 的内存占用代理指标
- $T$：该程序需要的工具环境集合（Docker 沙箱、API 端点等）
- $L$：程序分配的 GPU 节点位置（空间局部性）
- $\tau$：当前执行阶段，$\tau \in \{\text{Inference}, \text{Action}\}$
- $s$：调度状态，$s \in \{\text{Active}, \text{Paused}, \text{Terminated}\}$

**关键洞察**：传统推理引擎仅维护每个请求的 $c$，缺乏 $T$、$L$、$\tau$、$s$ 的感知能力，因而无法：
- 在 Action 阶段主动判断是否保留 KV Cache
- 知晓工具资源已分配（防止泄漏）
- 跨节点调度以平衡内存

### 关键技术点 2：程序感知 KV Cache 调度

#### 成本模型

论文将 Agent 推理的总成本分解（公式 3）：

$$\text{Cost}_\text{total} \approx \underbrace{\text{Cost}_\text{decode} + \text{Cost}_\text{prefill}}_{\text{生产性开销}} + \underbrace{\text{Cost}_\text{recompute} + \text{Cost}_\text{unused} + \text{Cost}_\text{caching}}_{\text{非生产性开销}}$$

ThunderAgent 的目标是**最小化非生产性开销**的三项：
- $\text{Cost}_\text{recompute}$：KV Cache 被驱逐后恢复时的重计算成本，与 $c^2$ 成正比
- $\text{Cost}_\text{unused}$：KV Cache 长时间占用 GPU 内存但未使用的机会成本
- $\text{Cost}_\text{caching}$：KV Cache 换入换出（GPU ↔ CPU）的传输成本

#### 颤振检测（Thrashing Detection）

当内存容量 $C_\text{total}$ 不足以同时容纳所有活跃程序的 KV Cache 时，发生颤振：

$$C_\text{total} < \sum_{p \in L} c_p + \sum_{q \in L, \tau_q = A} c_q \cdot f(t_q)$$

其中 $f(t_q)$ 是**时间衰减函数**，对处于 Action 阶段的程序 $q$，随时间 $t_q$（等待工具结果的时长）对其 KV Cache 的「实际价值」进行折现。

时间衰减函数设计：
- **连续形式**：$f(t) = e^{-\lambda t}$
- **离散形式**（论文实验使用）：$f(k) = 2^{-k}$，$k$ 为等待的调度周期数

**直觉**：工具调用等待越久，其 KV Cache 越不值得保留（因为其他活跃程序需要内存）。

#### 暂停/恢复打分

系统在检测到颤振时执行 Pause/Restore 操作：

- **暂停得分**（选择暂停哪个程序）：
$$S_\text{pause}(P) = \frac{1}{c_P} + \mathbb{1}[\tau_P = \text{Action}]$$
优先暂停上下文短（重计算成本低）且处于 Action 阶段（当前不需要 GPU）的程序。

- **恢复得分**（选择恢复哪个程序）：
$$S_\text{restore}(P) = \frac{1}{c_P} + \mathbb{1}[\tau_P = \text{Inference}]$$
优先恢复上下文短（重计算成本低）且处于 Inference 阶段（即将使用 GPU）的程序。


#### 全局程序感知等待队列

ThunderAgent 引入**统一的跨 DP 节点等待队列**，打破每个 GPU 节点各自维护本地队列的局限。当某节点内存压力过大时，系统可将暂停的程序路由至其他节点执行，实现跨节点负载均衡。

具体约束（未使用成本的上界）：

$$\text{Cost}_\text{unused} < c_\text{min} \cdot \Delta t$$

其中 $c_\text{min}$ 是最小程序的上下文长度，$\Delta t = 5$ 秒为调度检测间隔。该约束限制了 KV Cache 空占 GPU 内存的最大时长。

内存管理水位：
- **高水位** $\lambda_\text{max} = 1$：KV Cache 占满时触发 Pause
- **低水位** $\lambda_\text{min} = 1$：（默认配置下等于高水位，即精确触发）

### 关键技术点 3：工具资源管理器（Tool Resource Manager）

TRM 解决了 LLM 推理引擎对外部工具资源「视而不见」的问题，通过两个机制实现工具资源的全生命周期管理：

#### 基于钩子的垃圾回收（Hook-Based GC）

当程序状态转为 $s = \text{Terminated}$ 时，TRM 触发一条资源回收序列：
1. 销毁关联的 Docker 沙箱
2. 关闭网络套接字
3. 释放计算槽位（CPU、GPU 的工具执行资源）
4. 更新磁盘配额统计

这完全类比于操作系统的进程终止时的资源回收。论文实验（Figure 2b）表明：
- **vLLM 基线**：磁盘占用随时间线性增长，24 小时后耗尽存储
- **ThunderAgent**：磁盘占用维持恒定，节省最多 **4.2×** 磁盘内存

#### 异步环境预热（Async Environment Preparation）

在等待队列中排队的高优先级程序可能即将进入 Action 阶段，TRM 提前在后台初始化其所需的 Docker 容器或远程 API 连接，将环境初始化时间「隐藏」在 Inference 阶段之中。

这与操作系统的**预取（Prefetching）**机制高度类似。论文指出该优化贡献约 **10% 的端到端延迟改善**。

---

## 📊 实验与性能评价

### 实验设置

- **硬件**：2 × H100 节点（分布式场景）
- **测试模型**：GLM-4.6（智谱 AI Agent 专用模型）、Qwen3
- **Agent 框架**：mini-SWEAgent（基于 SWE-bench 任务）、OpenHands
- **对比基线**：vLLM（标准 LLM 服务引擎）、Continuum（专为 Agent 优化的系统）

### 服务吞吐量（Agent Serving Throughput）

**实验配置**：多并发 Agent 实例同时运行代码任务（LiveCodeBench 等场景），测量每秒完成的任务数或 token 吞吐量。

**结果**（Figure 4，对比 vLLM 基线）：
- ThunderAgent vs vLLM：**1.48–3.58×** 加速
- ThunderAgent vs Continuum：**1.17–3.31×** 加速
- 覆盖 GLM-4.6 和 Qwen3 两个模型，多种并发级别

**分析**：加速比在高并发时更显著，因为此时 KV Cache 竞争加剧，程序感知调度带来的收益更大。

### RL Rollout 吞吐量

**实验配置**：使用 2 × H100 节点，运行强化学习训练中的 Rollout 阶段（Agent 与代码执行环境交互），测量每分钟执行的推理步数（steps/min）。

**结果**（Table 2）：

| Agent 框架 | ThunderAgent | vLLM 基线 | 加速比 |
|-----------|-------------|---------|-------|
| mini-SWEAgent | 671.8 steps/min | 375.4 steps/min | **1.79×** |
| OpenHands | 270.8 steps/min | 69.1 steps/min | **3.92×** |

OpenHands 加速比更高，因其工具调用更密集，程序感知调度的 KV Cache 保护效益更明显。

### KV Cache 命中率分析

**结果**（Figure 5）：
- **确定性工具调用**（工具执行时间可预测）：KV Cache 命中率接近 **100%**，因时间衰减函数可以精准判断何时保留 Cache
- **随机工具调用**（执行时间波动大）：系统在 Cache 命中率与 GPU 内存空闲成本之间动态权衡，命中率随衰减参数 $\lambda$ 调整

### 资源效率

- **磁盘内存节省**：**4.2×**（Hook GC 消除容器泄漏）
- **环境准备延迟**：显著降低（Async 预热减少等待）（Figure 2c 定性展示）
- **跨节点均衡**：全局等待队列有效减少某节点 OOM 同时其他节点空闲的情况

### 局限性（Limitations）

1. **API 侵入性**：需要 Agent 在请求中携带 `Program_id`，现有 Agent 需要少量改动（但论文称为「OpenAI 兼容」，改动极小）
2. **时间衰减参数敏感**：$\lambda$（衰减速率）需要根据工具类型调优，不同类型工具的执行时间分布差异大
3. **单一调度器瓶颈**：全局等待队列的调度器本身可能在极高并发下成为瓶颈（论文未讨论此限制）
4. **仅支持 vLLM 和 SGLang**：不支持其他推理引擎（如 TensorRT-LLM）
5. **工具环境异构性**：对于非 Docker 化的工具（如部分云 API），TRM 的 GC 机制需要额外适配

---

## 💡 启发与我的想法

### 可借鉴之处

1. **「程序」抽象的普适性**：将 Agent 工作流抽象为操作系统意义上的「进程」，这一思路与 MemGPT 等工作异曲同工，但 ThunderAgent 专注于推理引擎层面的资源调度而非上下文管理，更加底层。这种**将 OS 概念映射到 LLM 系统**的方法论值得借鉴——Pause/Resume 对应进程挂起/唤醒，Hook GC 对应析构函数，全局队列对应操作系统的进程调度队列。

2. **时间衰减函数设计**：$f(t) = 2^{-t}$ 的离散衰减设计简洁有效，将不确定的工具等待时间转化为连续的 Cache 价值估计。这一思路可以推广到其他需要「价值随时间衰减」的场景，如 Speculative Decoding 中 Draft Token 的过期处理。

3. **跨系统边界的资源感知**：传统推理引擎只管 GPU 侧的 KV Cache，ThunderAgent 打通了 GPU（KV Cache）+ CPU（工具进程）+ 磁盘（Docker 镜像）的全栈视野，这种全栈资源管理将成为 Agent 时代推理系统的标配能力。

4. **异步预热的通用性**：将环境初始化开销「隐藏」在计算期间，这是经典的延迟隐藏（Latency Hiding）技术，与 GPU 核函数异步执行、DMA 异步传输的设计哲学一致。在 KV Cache offloading、模型权重预加载等场景均有相似应用空间。

### 改进空间

1. **自适应衰减参数**：目前 $\lambda$ 需要手动调优，可以引入在线学习机制，根据历史工具执行时间分布自动调整衰减速率。例如，对 Shell 命令（ms 级）和网络爬虫（s 级）使用不同的 $\lambda$，或用指数滑动平均动态估计工具执行时间。

2. **预测性 KV Cache 保护**：当前系统在工具调用发出后才开始「等待」，可以在 Agent 生成工具调用 token 的过程中即预测该工具的执行时间，提前做出是否保留 Cache 的决策（推测性资源管理）。

3. **与 KV Cache 稀疏化结合**：如 KvZap 等工作针对长上下文 KV Cache 的稀疏压缩，与 ThunderAgent 的 KV Cache 生命周期管理正交，可以叠加——先用稀疏化压缩 Cache 体积，再用 ThunderAgent 延长 Cache 存活时间，双重收益。

4. **多租户场景下的公平性**：当前调度以吞吐量为目标，未考虑租户间的 SLA（延迟保证）。在多租户生产环境中，需要在全局吞吐量最大化与个别 Agent 的尾延迟保障之间取得平衡。

### 下一步 Action
- [ ] 阅读 Continuum 原文，理解其与 ThunderAgent 在设计上的具体差异
- [ ] 调研 vLLM 的 KV Cache 驱逐策略，与 ThunderAgent 的暂停/恢复机制对比
- [ ] 思考 ThunderAgent + KvZap 稀疏化的结合点：是在 Pause 时做稀疏压缩，还是在 Restore 时选择性重建？
- [ ] 关注 GLM-4.6 和 Qwen3 在 Agent 场景下的性能差异，思考模型结构是否影响系统优化效果

