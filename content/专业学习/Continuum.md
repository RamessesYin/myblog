---
title: "Continuum"
tags:
  - LLM推理
  - KV-Cache
  - 系统优化
  - Agent调度
  - UC-Berkeley
---

## 🧠 核心摘要 (TL;DR)

> 现有 LLM 推理引擎在处理多轮 Agent 工作负载时，每次工具调用结束后都会立即丢弃 KV Cache，导致下一轮恢复时必须重新计算（Prefill）整个上下文，产生大量冗余算力浪费。Continuum 提出了基于 **KV Cache Time-to-Live (TTL)** 的自适应保留策略：通过对工具调用时长分布的历史建模，为每个请求的 KV Cache 分配一个"存活时间"，在工具执行期间保留 GPU 显存中的缓存，从而在下一轮请求到来时实现零重计算的直接恢复。实验在 SWE-Bench 和 BFCL 真实 Agent 数据集上验证，与 Autellix、SGLang 等基线相比，端到端延迟降低 **1.12x–3.66x**，吞吐量提升 **1.10x–3.22x**，在真实 SWE-agent 部署中最高实现 **8.18x** 加速。

## 🏷️ 元数据
- **论文标题**: Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live
- **发表机构/会议**: UC Berkeley / arXiv (cs.OS, cs.AI, cs.NI)
- **年份**: 2025（v3 更新于 2026-01）
- **链接**: [[Continuum_paper|📄论文]] | [💻 Code](https://github.com/vllm-project/vllm-continuum)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #KV-Cache #TTL #多轮Agent #LLM推理调度 #GPU显存管理 #工具调用
- **前置知识 (Prerequisites)**: [[KV稀疏调研]], [[KvZap]]
- **相关节点 (Related Notes)**: [[MemCopy]], [[ZipServ]]
- **向下延申 (Successors)**: 后续可关注 disaggregated prefill/decode 架构与 KV Cache 管理的联合优化

---

## 🎯 动机与痛点

### 多轮 Agent 工作负载的独特性

现代 LLM Agent（如 SWE-agent、浏览器 Agent）的工作模式与普通的单轮对话有本质区别。一个典型的 Agent 任务由多轮 LLM 调用串联构成，每次 LLM 输出一个工具调用指令（如执行 Bash 命令、搜索网页、调用 API），等待工具执行完毕后，再将结果拼接到上下文并进入下一轮推理。这种 **"LLM 推理 → 工具执行 → LLM 推理"** 的交替循环模式，暴露了现有推理引擎在三个层面的根本缺陷。

### 现有系统的三大失效模式

**失效模式 1：按轮丢弃（Turn-based Eviction）**

vLLM 等主流系统的 KV Cache 生命周期与单次请求严格绑定——一旦某轮 LLM 输出完成，其对应的 KV Cache 就立即被系统回收以释放显存。当 Agent 的下一轮请求到来时（携带了新的工具调用结果），系统必须重新对**整个历史上下文**执行 Prefill，将原本只需要计算新增 token 的增量任务，膨胀成了一次完整的全量重计算。随着对话轮数增加，这种冗余计算的开销呈 $O(L^2)$ 增长（$L$ 为轮数），在长任务中尤为严重。

**失效模式 2：排队延迟的逐轮累积（Per-turn Queueing Delay）**

部分系统（如 LMCache）引入了 CPU 内存 Offloading 方案——将 KV Cache 卸载到 CPU DRAM 以节省 GPU 显存。然而，当 Agent 恢复执行时，该请求依然需要和队列中的其他请求竞争 GPU 显存资源，并等待从 CPU 重新加载 KV Cache（CPU→GPU 带宽通常为 PCIe 的 16~64 GB/s，加载延迟不可忽视）。每一轮都要经历一次"排队等待 + 数据搬运"的组合延迟，在高并发场景下延迟会显著积累。

**失效模式 3：静态保留策略的脆弱性（Static Retention Policy）**

如果采用固定时长的 KV Cache 保留策略（例如"保留 10 秒"），则对于工具执行时间超过预设阈值的请求，Cache 会在工具返回前提前过期，等价于退化为按轮丢弃；而对于执行极快的工具，又会无谓地占用显存。更危险的是，在高负载下，若多个请求的 KV Cache 同时被固定（Pinned），可能导致新请求找不到可用的显存块，造成**系统级死锁**。

### 相关工作对比

| 系统 | KV Cache 处理策略 | 多轮 Agent 感知 | 核心局限 |
|------|-----------------|---------------|---------|
| **vLLM** | 按请求即时回收 | ❌ | 每轮完整重计算，冗余 Prefill 严重 |
| **SGLang (RadixAttention)** | 前缀树缓存，基于共享前缀复用 | ❌ 仅同前缀请求 | 无法跨轮保留，请求串行时无法复用 |
| **LMCache** | CPU Offloading + 跨请求 KV 复用 | 部分 | 恢复时需 CPU→GPU 迁移，受 PCIe 带宽限制 |
| **Autellix** | 程序级调度感知，优先抢占低优先级请求 | ✅ 程序视角 | 无 KV Cache 保留机制，仍需重计算 |
| **Mooncake** | 预填充/解码分离 + 分布式 KV 池 | ❌ | 生产系统复杂度高，面向大规模集群 |
| **Continuum** | **自适应 TTL 保留 + 感知调度** | ✅ 工具调用感知 | 当前主要针对单节点 GPU，跨节点扩展仍在探索 |

---

## ⚙️ 核心设计与机制

Continuum 的设计哲学可以概括为：**让推理系统"理解"Agent 的工作模式，而非盲目地管理请求的生命周期**。

### 系统架构总览

Continuum 构建于 vLLM 之上（约 1,000 行 Python 新增代码），通过拦截工具调用的输入/输出，将 Agent 程序的多轮状态引入到调度决策中。架构包含三个核心模块：

1. **工具调用解析器（Tool Call Parser）**：实时检测 LLM 输出流中的工具调用 schema（兼容 OpenAI Function Calling 格式），识别当前程序进入"等待工具执行"状态；
2. **历史时长追踪器（Duration Tracker）**：按工具函数名维护历史执行时长的滑动窗口统计，输出时长分布 $\mathcal{D}_f$；
3. **TTL 感知调度器（TTL-aware Scheduler）**：综合成本收益模型为每个 KV Cache 分配 TTL，并在调度优先级排序中保留程序级 FCFS 顺序。

### TTL 的成本收益模型

Continuum 将为一个请求保留 KV Cache 的决策建模为一个 **期望收益最大化问题**。

设当前请求在工具执行阶段，其 KV Cache 占用 $m$ 个显存块，工具执行时长为随机变量 $T_f \sim \mathcal{D}_f$，为其设定 TTL 值 $\tau$ 的**期望净收益**为：

$$
\text{Gain}(\tau) = \underbrace{\Pr[T_f \leq \tau] \cdot (C_{\text{prefill}} + C_{\text{sched}})}_{\text{命中时节省的计算与排队开销}} - \underbrace{\mathbb{E}[\min(T_f, \tau)] \cdot m \cdot P_{\text{mem}}}_{\text{保留期间的显存机会成本}}
$$

其中：
- $C_{\text{prefill}}$：重新计算整个历史上下文所需的 Prefill 成本（与上下文长度的平方正相关）；
- $C_{\text{sched}}$：若不保留 Cache，下一轮请求进入队列等待的预期调度延迟；
- $P_{\text{mem}}$：单位时间内占用一个显存块的机会成本（与系统当前负载正相关）；
- $\Pr[T_f \leq \tau]$：根据历史分布 $\mathcal{D}_f$ 估计工具调用在 TTL 内完成的概率。

最优 TTL 为使 $\text{Gain}(\tau)$ 最大化的 $\tau^*$：

$$\tau^* = \arg\max_{\tau \geq 0} \ \text{Gain}(\tau)$$

当系统负载较低（$P_{\text{mem}}$ 小）或工具调用快速（$\Pr[T_f \leq \tau]$ 快速趋近 1）时，TTL 会被设置得更长；高负载下 $P_{\text{mem}}$ 升高，TTL 收窄，系统自动转向更激进的 Cache 回收策略。

### "Memoryfulness" 因子

对于历史数据不足的新工具函数，Continuum 引入了 **Memoryfulness Factor** $\mu \in [0, 1]$ 来描述工作负载对 KV Cache 重用的依赖程度。$\mu$ 由以下两个维度综合计算：

1. **上下文增长率**：相邻两轮请求的输入 token 长度之比（体现历史上下文的累积速度）；
2. **工具调用频率**：单个任务平均的工具调用轮数（体现多轮性）。

$\mu$ 接近 1 表示工作负载高度依赖跨轮上下文复用，系统倾向于保留更多 Cache；$\mu$ 接近 0 则退化为标准的按轮回收策略。

### 死锁预防机制

当系统显存接近满载，且多个请求的 TTL 尚未到期时，若不加干预可能导致新请求无法获得显存而永久阻塞。Continuum 实现了基于**受害者选择（Victim Selection）**的死锁预防：

- 优先回收 $\text{Gain}(\tau_{\text{remaining}})$ 最小（或已为负）的 KV Cache；
- 保证系统始终留有一定比例的空闲显存作为"安全缓冲区"；
- 被回收的请求自动降级为完整 Prefill 路径，不影响正确性，仅牺牲该轮的性能优化。

### 调度优先级设计

Continuum 在调度器层面维护**程序级 FCFS（Program-level First-Come-First-Served）**顺序：同一 Agent 程序的多轮请求在队列中保持相对顺序，避免程序内部的 Head-of-Line Blocking 问题（这与 Autellix 的程序级抢占思路互补，但 Continuum 侧重 Cache 保留而非优先级抢占）。

---

## 📊 实验与性能评价

### 实验设置

- **评测基准**：
  - **SWE-Bench**：软件工程任务（代码修复），典型工具调用包括代码执行、文件搜索，工具调用轮数 5–15 轮；
  - **BFCL (Berkeley Function Calling Leaderboard v4)**：浏览器 Agent 任务，工具调用延迟分布更宽（秒级到分钟级）；
- **硬件**：单节点 A100/H100 GPU 集群；
- **对比基线**：vLLM（无 Cache 保留）、SGLang（RadixAttention）、LMCache（CPU Offloading）、Autellix（程序级调度）；
- **核心指标**：端到端任务完成延迟（Job Completion Time, JCT）、吞吐量（Jobs/sec）、调度开销（ms）。

### 主要实验结果

| 场景 | 对比基线 | 延迟改善 | 吞吐量提升 |
|------|---------|---------|-----------|
| SWE-Bench（轻负载） | vLLM | 2.1x | 1.95x |
| SWE-Bench（重负载） | Autellix | 1.42x | 1.38x |
| BFCL（长工具延迟） | LMCache | 3.66x | 3.22x |
| BFCL（短工具延迟） | SGLang | 1.12x | 1.10x |
| 真实 SWE-agent 部署 | vLLM | **8.18x** | — |

> **说明**：在真实 SWE-agent 场景下的 8.18x 加速来自工具调用（Bash/搜索）执行时间较长（平均 20–60 秒），TTL 命中率接近 90%，几乎消除了所有冗余 Prefill 开销。

### 鲁棒性验证

- **跨轮数**：在 3–20 轮不同深度的 Agent 任务上，Continuum 的加速比随轮数单调递增（轮数越多，重计算省略越多）；
- **跨 Batch 大小**：在 1–32 并发请求下，相对加速比保持稳定；
- **调度开销**：TTL 计算和受害者选择的额外延迟约为 **2–2.3 ms**，可忽略不计。

### 局限性

1. **单节点部署**：当前系统主要在单 GPU 节点上验证，多机分布式下 KV Cache 的跨节点保留与迁移机制尚未完善；
2. **工具函数冷启动**：首次遇到新工具函数时无历史分布，TTL 只能依赖 Memoryfulness 因子保守估计，前几轮可能无法充分复用；
3. **精确 TTL 预测的假设**：成本收益模型假设工具执行时长的历史分布能代表未来，对于执行时间高度随机的工具（如依赖网络的 API 调用），预测误差可能较大；
4. **与 Disaggregated 架构的兼容性**：在预填充/解码分离架构（如 Mooncake）下，KV Cache 的跨节点转移与 TTL 保留机制的协同还需进一步设计。

---

## 💡 启发与我的想法

### 可借鉴之处

Continuum 最核心的洞察是：**推理系统不应对所有请求一视同仁，而应感知 Agent 程序的工作流结构**。将"工具调用等待期间的 KV Cache 保留"这一优化从"有没有显存就看运气"提升为"基于期望收益的有据可依的决策"，是非常干净的工程思路。这种成本收益建模方式可以迁移到其他资源管理问题，如 KV Cache 在 CPU/NVMe 的分层存储中的驻留决策。

尤其值得关注的是其与 [[KvZap]] 思路的对比：KvZap 解决的是"在一次推理中哪些 token 的 KV 不重要可以丢弃"，而 Continuum 解决的是"在多轮 Agent 的间隙，是否值得保留整个请求的 KV Cache"——两者处于不同的时间粒度（intra-request vs. inter-request），未来可以组合使用。

### 改进空间

1. **TTL 预测的自适应性**：当前使用历史滑动窗口，可以引入轻量级在线学习模型（如基于工具输入特征预测执行时长），提升冷启动和分布漂移场景的鲁棒性；
2. **与 Disaggregated Prefill/Decode 的结合**：在 Mooncake 式的分离架构中，工具调用期间的 KV Cache 可以卸载到专用的 KVCache 节点（而非 GPU 或 CPU），实现更低成本的保留；
3. **多租户场景下的公平性**：若多个 Agent 任务竞争 TTL 保留，需要考虑跨用户的显存公平分配，避免少数高频任务垄断保留配额；
4. **与 RL 训练流水线的结合**：[[RL中的KV稀疏讨论]] 中的 rollout 阶段同样是多轮交互模式，Continuum 的 TTL 思路是否能应用于 RL 训练中的 Actor 推理加速，值得探索。

### 下一步 Action
- [ ] 拉取 `vllm-continuum` 仓库，在本地复现 SWE-Bench 的延迟对比实验；
- [ ] 阅读 Autellix（arXiv:2502.13965）原文，理解程序级抢占调度与 Continuum TTL 方案的互补关系；
- [ ] 调研 Mooncake（KVCache-centric disaggregated architecture）中的 KV 迁移协议，评估与 TTL 机制结合的可行性；
- [ ] 思考在 RL rollout 场景下，Agent 与环境交互期间的 KV Cache 保留策略是否可以直接借鉴 Continuum 框架。
