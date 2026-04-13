---
title: "Mirage Persistent Kernel"
tags:
  - GPU编译器
  - LLM推理优化
  - 算子融合
  - 持久化内核
  - 张量程序
---

## 🧠 核心摘要 (TL;DR)
> 现有 LLM 推理系统以"每算子一次 kernel launch"的方式驱动 GPU，launch 开销与跨算子调度的低效严重制约了低延迟场景的性能上限。Mirage Persistent Kernel（MPK）提出将整个模型推理编译为**单个 megakernel**——一次 kernel launch 完成所有计算与通信，并通过 SM 级图表示（ttGraph）、去中心化调度和跨算子软流水线三项核心机制，将端到端 LLM 推理延迟降低 **1.2× 至 6.7×**（单批次场景最高 **1.7×**）。

## 🏷️ 元数据
- **论文标题**: Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs
- **发表机构/会议**: arXiv 预印本（CMU / Apache 2.0 开源项目）
- **年份**: 2025（提交于 2025-12-22）
- **链接**: [[Mirage_blog|📄项目页]] | [💻 Code](https://github.com/mirage-project/mirage)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #gpu-compiler #kernel-fusion #llm-inference #persistent-kernel #tensor-program #megakernel
- **前置知识 (Prerequisites)**: [[ZipServ]], [[MSched]]
- **相关节点 (Related Notes)**: [[ShiftParallelism]], [[KV稀疏调研]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

现代 LLM 推理框架（vLLM、SGLang、FasterTransformer 等）普遍采用**以算子为粒度的 kernel launch** 模式：每执行一个算子（如 LayerNorm、QKV 投影、Attention、FFN）就进行一次 CUDA kernel launch，各算子依次提交至 GPU。这一模式在高吞吐量场景（大批次）下近乎合理，但在**低延迟 / 小批次场景**（如在线推理、实时对话）中暴露出三大结构性缺陷：

1. **Kernel Launch 开销累积**：单次 kernel launch 的 CPU→GPU 调度开销约 5–20 μs，对于 70B 参数的模型一次 forward 需要 launch 数百个 kernel，开销之和可占据端到端延迟的相当比例。

2. **跨算子流水线空洞**：相邻算子之间存在隐式同步——前一个 kernel 必须完成后才能 launch 下一个，GPU SM 在等待期间空转，计算-通信无法重叠。

3. **无法跨算子全局优化**：每个 kernel 独立编译，无法将相邻算子的 pre-loading 与 compute 阶段交叠，也无法跨算子共享寄存器或 shared memory 缓冲区。

NanoFlow（Zhu 等，2024）通过将输入切成 nano-batch 实现设备内并行，在吞吐量上达到理论峰值的 50–72%，但仍依赖多次 kernel launch。FlashDecoding（Dao 等，2023）和 FlashInfer（Ye 等，2024）在单个注意力 kernel 内做了深度优化，但覆盖范围局限于 attention 算子，未触及跨算子融合。

### 论文的切入点

MPK 提出从根本上改变执行模式：将整个推理图**静态编译为一个长时间运行的 megakernel**（persistent kernel），GPU 上的每个 SM 在 megakernel 内部循环执行任务，不再需要从 CPU 侧触发 kernel launch。这一思路将 GPU 推理的控制权从 CPU 的 runtime dispatcher 转移到 GPU 内部的自调度机制，消除了 launch 开销并为跨算子优化打开了空间。

### 相关工作

#### 先驱 / 奠基工作

| 论文 | 机构 | 核心贡献 | 与 MPK 的关系 |
|------|------|---------|--------------|
| TVM: An Automated End-to-End Optimizing Compiler for Deep Learning（Chen 等，OSDI 2018）| UW / OctoML | 首个面向多后端的深度学习算子级编译器，引入调度（schedule）与计算（compute）分离 | MPK 的编译器设计借鉴了 TVM 算子级 IR，但将粒度从算子提升到 SM |
| Ansor: Generating High-Performance Tensor Programs for Deep Learning（Zheng 等，OSDI 2020）| UC Berkeley | 基于分层搜索空间 + 学习成本模型自动搜索高性能张量程序 | MPK 中 ttGraph 的 SM 级任务搜索受 Ansor 搜索策略启发 |
| Mirage: A Multi-Level Superoptimizer for Tensor Programs（Wu 等，OSDI 2025）| CMU | 在 thread-block 粒度进行超优化，验证等价性；MPK 是其运行时扩展 | 直接前身：OSDI 2025 论文提出超优化器框架，MPK 将其扩展到 megakernel 编译与运行时 |
| FlashAttention: Fast and Memory-Efficient Exact Attention（Dao 等，NeurIPS 2022）| Stanford | IO-aware tiling 大幅减少 HBM 访问，实现 BERT-large +15%、GPT-2 3× 加速 | 启发了 MPK 在单 SM 粒度对内存层次的精细管理 |
| FlashAttention-2: Faster Attention with Better Parallelism（Dao，ICLR 2024）| Princeton | 减少非 matmul FLOPs，线程块内部并行，GPU 利用率从 25–40% → 50–73% | MPK megakernel 中 attention 计算的直接优化基础 |
| Megatron-LM: Training Multi-Billion Parameter Language Models（Shoeybi 等，2019）| NVIDIA | 模型并行（tensor/pipeline parallel）支持千亿参数训练 | MPK 多 GPU 模式的通信算子融合参考了 Megatron 的分层并行设计 |

#### 直接对比 / 竞争方法

| 论文 | 机构 | 核心贡献 | 与 MPK 的关系 |
|------|------|---------|--------------|
| Efficient Memory Management for LLM Serving with PagedAttention（Kwon 等，SOSP 2023）| UC Berkeley | PagedAttention + vLLM，通过虚拟内存管理 KV Cache 近零碎片，吞吐 2–4× 于 FasterTransformer | MPK 的对比基线之一；MPK 在小批次延迟上显著优于 vLLM |
| SGLang: Efficient Execution of Structured LM Programs（Zheng 等，NeurIPS 2024）| UC Berkeley | RadixAttention 跨请求复用 KV Cache，结构化输出加速，吞吐最高 6.4× | MPK 的另一主要对比基线；MPK 在端到端延迟上优于 SGLang |
| NanoFlow: Towards Optimal LLM Serving Throughput（Zhu 等，2024）| UW | nano-batch 设备内并行，吞吐达理论峰值 50–72%，比最优系统 +1.91× | 思路与 MPK 相似（减少流水线泡沫），但 NanoFlow 面向吞吐，MPK 面向延迟 |
| FlashInfer: Efficient and Customizable Attention Engine（Ye 等，2024）| UW / NVIDIA | 块稀疏 KV Cache 格式 + JIT 定制化，延迟降低 29–69%，已集成进 SGLang/vLLM | 注意力算子层面与 MPK 互补；MPK 在跨算子粒度上更全面 |

#### Follow-up / 延伸方向

| 论文 | 机构 | 关联方向 |
|------|------|---------|
| FlashDMoE: Fast Distributed Mixture-of-Experts in a Single Kernel（2024）| — | 类似的"单 kernel 完成分布式 MoE"思路，验证 megakernel 在 MoE 场景的可行性 |
| TensorIR: An Abstraction for Automatic Tensorized Program Optimization（2022）| — | IR 层抽象，影响 MPK 的 ttGraph 设计 |
| KORCH: Optimizing Kernel Orchestration for Tensor Programs（2023）| — | 算子编排优化，与 MPK 的 in-kernel 调度思路对应 |


## ⚙️ 核心设计与机制

MPK 的系统架构由三层核心机制叠加构成：**SM 级图表示（ttGraph）→ 去中心化调度运行时 → 跨算子软流水线**。

### 1. SM 级图表示：ttGraph

传统深度学习 IR（如 TVM 的 TIR、Triton 的 JIT IR）以**算子（operator）为节点**建模数据流图，每个算子对应一次 kernel launch。MPK 将粒度下探到**流多处理器（SM）级别**，引入 ttGraph（Tensor-Task Graph）：

- **节点**：每个节点对应一个 SM 上执行的**任务（Task）**，是一个微型计算单元（例如"SM #3 执行 token 5–7 的 RMSNorm"）
- **边（依赖事件）**：节点之间通过共享内存或寄存器缓冲区传递数据，依赖关系以"事件"（event）表达，而非跨 kernel 的全局 barrier
- **任务分解**：每个 Task 进一步拆分为两个阶段：
  - **Pre-loading phase**：从全局内存（HBM）预取下一次计算所需的权重或 KV 数据到 shared memory
  - **Compute phase**：在已预取的数据上执行矩阵乘法或归约操作

通过 ttGraph，编译器可以在**单个 SM 内**调度来自不同算子的任务，从根本上打破算子边界的壁垒。

```
传统模式：
[Kernel A: LayerNorm] ──sync── [Kernel B: Linear] ──sync── [Kernel C: Attention]
     SM全部空转等待下一launch ↑

ttGraph 模式（同一 megakernel 内）：
SM_0: [LN_task0 pre] → [LN_task0 compute | Linear_task0 pre] → [Linear_task0 compute | Attn_task0 pre] → ...
SM_1: [LN_task1 pre] → [LN_task1 compute | Linear_task1 pre] → ...
```

此设计使得**跨算子的 pre-loading 与 compute 可以在时间轴上精确交叠**，消除 HBM 访问的等待气泡。

### 2. 去中心化 SM 调度运行时（Decentralized Scheduling）

如何在 megakernel 运行期间，让数千个 SM 高效地领取任务而不产生全局协调开销？MPK 采用**去中心化调度**策略：

**核心思想**：不设置全局 task queue 和 central dispatcher，每个 SM 配备一个轻量级本地 scheduler，仅依赖**本地状态**决定下一步执行哪个 task。

**实现机制**：
- 每个 SM 维护一个小型的本地 ready queue，存放其可执行的下一批 task
- 当一个 task 完成后，其"事件"（event）会通知依赖该结果的下游 task（通过 shared memory 的 flag 翻转）
- 下游 task 的归属 SM 检测到 flag 后，将该 task 加入自己的 local queue，无需访问全局状态
- 不同 SM 之间**无需锁、无需原子操作跨 SM 通信**，彻底消除全局协调开销

这与传统的集中式 dynamic parallelism 或 thread block cluster 方案形成鲜明对比：后者在 SM 数量增加时，全局 queue 的竞争成为性能瓶颈；MPK 的去中心化方案随 SM 数量线性扩展。

**API 层面**：
```python
from mirage import PersistentKernel

pk = PersistentKernel(
    num_workers=108,        # SM 上的 worker 数量
    num_schedulers=4        # 轻量级本地 scheduler 数量（每 scheduler 管理一组 SM）
)

# 注册 meta tensors（megakernel 全局共享的状态）
step   = pk.new_meta_tensor(shape=(1,), dtype=torch.int32)     # decoding 步骤计数器
tokens = pk.new_meta_tensor(shape=(max_seq_len, hidden), ...)  # token 缓冲区

# 定义计算图（算子间依赖通过 Python API 隐式建立）
x = pk.rmsnorm_linear_layer(tokens, W_norm, W_q, W_k, W_v)
# ... 更多算子
pk.compile()
```

### 3. 跨算子软流水线（Cross-Operator Software Pipelining）

软流水线是 MPK 性能提升的核心来源。其原理是将 task 的 pre-loading 与 compute **解耦**，在时间维度上形成流水：

**单任务内的 double-buffering**：

```
时间轴（单 SM）：
|←── pre-load buffer_A ──→|←── compute buffer_A / pre-load buffer_B ──→|←── compute buffer_B ... ──→|
```

**跨算子的任务流水线**：

假设模型结构为 RMSNorm → Linear → Attention：

```
时间轴（SM_0）：
Tick 1: [RMSNorm pre-load: token 0-3]
Tick 2: [RMSNorm compute: token 0-3] + [Linear pre-load: token 0-3 weights]
Tick 3: [Linear compute: token 0-3] + [Attention pre-load: KV for token 0-3]
Tick 4: [Attention compute: token 0-3] + [RMSNorm pre-load: token 4-7]  ← 进入下一循环
```

关键优化点：
- **权重预取**：Linear 层的大型权重矩阵在前一算子 compute 阶段就开始异步从 HBM 加载，消除内存访问等待
- **KV Cache 预取**：在 decoding 阶段，下一 token 所需的 KV block 可在当前 token 计算时提前加载
- **通信-计算重叠**：在多 GPU 张量并行场景下，AllReduce 等集合通信操作的数据传输阶段与本地 SM 的下一批计算精确交叠，实现通信隐藏

**多 GPU 扩展（Tensor Parallelism）**：

MPK 通过 NVSHMEM 实现多 GPU 间的 in-kernel 数据访问，无需退出 megakernel 进行 NCCL 通信：

```python
# 多 GPU tensor：SM 可通过 NVSHMEM 直接读写远端 GPU 的 tensor
kv_cache = pk.new_nvshmem_tensor(shape=(...), dtype=torch.float16)  # 分布于 8 张 H100
```

在 8 × H100 的张量并行配置下，MPK 实现 **1.1–1.4×** 相对于 SGLang 的延迟改善，通信开销被有效隐藏。

### 核心公式

**端到端延迟分解**（MPK 消除的延迟来源）：

$$T_{\text{total}} = T_{\text{compute}} + \underbrace{T_{\text{launch}} + T_{\text{sync}}}_{\text{MPK 消除}} + \underbrace{T_{\text{idle}}}_{\text{MPK 通过流水线消除}}$$

**软流水线效率**（粗估）：

$$\text{Speedup} \approx \frac{T_{\text{compute}} + T_{\text{memory}}}{T_{\text{compute}} + \max(T_{\text{memory}} - T_{\text{overlap}}, 0)}$$

其中 $T_{\text{overlap}}$ 是跨算子预取所能隐藏的内存访问时间。在权重矩阵大、SM 计算能力饱和的场景下，$T_{\text{overlap}} \to T_{\text{memory}}$，加速比趋近于计算-访存之比（即 arithmetic intensity 决定的理论上限）。


## 📊 实验与性能评价

### 对比基线（Baselines）

MPK 与以下两个主流推理框架进行了系统比较：
- **SGLang**（v0.4，RadixAttention + PagedAttention，NeurIPS 2024）
- **vLLM**（PagedAttention，SOSP 2023）

实验平台：NVIDIA H100 80GB SXM5（单 GPU 及 8-GPU 张量并行），测试模型包括 Qwen3-8B、LLaMA 系列。

### 主要实验结果

#### 1. 单 GPU 单批次延迟（Latency-Sensitive Scenario）

这是 MPK 最具优势的场景——batch size = 1，模拟在线对话的低延迟需求：

| 场景 | MPK vs SGLang | MPK vs vLLM |
|------|--------------|------------|
| 总体范围 | **1.0–1.7×** 延迟改善 | 类似 |
| 跨任务流水线贡献 | **1.2–1.3×** | — |
| 计算-通信重叠贡献 | **1.1×** | — |
| 多 GPU (8×H100) | **1.1–1.4×** | — |

> 注：GitHub README 中标注"reduces LLM inference latency by 1.2× to 6.7×"，此处 6.7× 为特定小模型 / 特定序列长度下的极端情况，论文正文中的平均数据为 1.2–1.7×。

#### 2. 软流水线贡献分解（Ablation Study）

| 优化组件 | 单独加速比 |
|---------|-----------|
| Cross-task software pipelining | 1.2–1.3× |
| Compute-communication overlap (multi-GPU) | ~1.1× |
| 两者叠加 | 最高 1.7× |

#### 3. 硬件利用率

MPK 使端到端延迟接近"硬件限制"（hardware limits），即在给定模型参数量和精度下，理论上 SM 利用率所能达到的最优延迟。具体指标：SM 利用率在小批次场景下较 SGLang 提升显著（具体数据论文中以图表形式呈现）。

### 局限性（Limitations）

1. **Compile-time overhead**：将整个推理图静态编译为 megakernel 需要较长的编译时间，不适合频繁变换模型结构的场景（如动态网络）。
2. **动态控制流受限**：现有 MPK 编译器主要针对 Transformer 解码的固定计算图，对带有条件分支的动态模型（如 Mixture-of-Experts 的专家路由）支持有限（虽然 FlashDMoE 验证了 MoE megakernel 的可行性，但 MPK 当前版本尚未正式支持）。
3. **Small-batch 专属优势**：在大批次（batch size ≫ 1）场景下，传统 launch 模式的 launch overhead 占比极低，MPK 相对收益有限；其核心价值在于延迟敏感的在线服务场景。
4. **硬件依赖**：当前实现深度依赖 NVIDIA H100/A100 的 NVSHMEM 和 Programmatic Dependent Launch（PDL）特性，在较旧 GPU 上的可用性尚未完整验证。
5. **生态成熟度**：项目仍处于早期阶段（GitHub roadmap 显示多项功能在开发中），与 vLLM / SGLang 的集成深度有限。

---

## 💡 启发与我的想法

### 可借鉴之处

1. **"将控制权交给 GPU"的范式转变**：MPK 最深层的贡献不是某个具体的算法优化，而是从根本上改变了 LLM 推理的执行模型——从"CPU 驱动 GPU"转变为"GPU 自驱动"。这与操作系统从协作式多任务到抢占式多任务的演进有哲学上的相似性。

2. **粒度下探的威力**：从算子级 IR（TVM）→ thread-block 级超优化（Mirage OSDI 2025）→ SM 级运行时（MPK），每次粒度下探都解锁了新的优化空间。这提示我们：当系统存在性能瓶颈时，有时需要向下突破抽象层边界，而不是在原有粒度上修修补补。

3. **去中心化调度的可扩展性**：MPK 的 SM 级 decentralized scheduler 是一个优雅的设计——避免全局 coordinator 成为热点，通过本地状态驱动调度，天然扩展到大规模 GPU。这一设计模式在分布式系统（如 work-stealing scheduler）中已有先例，MPK 将其引入 GPU intra-kernel 调度。

4. **软流水线与 double-buffering**：这是 HPC 领域的经典技术（在 CUDA 的 cutlass 库中广泛使用），MPK 的贡献在于**跨算子**地应用了这一技术，而不局限于单个 matmul 内部。

### 改进空间

1. **动态批次大小适应**：当前 MPK 对固定 batch size 编译，对于实际推理服务中请求数量动态变化的场景，可以考虑"多版本 megakernel + 动态选择"策略。

2. **与 KV Cache 管理的深度集成**：MPK 目前主要聚焦于计算内核的融合，但 vLLM 的 PagedAttention 和 SGLang 的 RadixAttention 在 KV Cache 内存管理上有成熟方案。将 MPK 的 megakernel 编译与 KV Cache 分页/复用机制深度集成，可能是下一个重要突破点。

3. **量化感知编译**：将 INT8/INT4 量化（如 GPTQ、AWQ 系列）融入 ttGraph 的任务定义，在 megakernel 内部同时完成 dequantization 和 matmul，避免额外的数据类型转换开销。

4. **与稀疏注意力的协同**：KV Cache 稀疏化（如本笔记站中的 KvZap、RL 中的 KV 稀疏讨论所涉及的方向）可与 MPK 形成正交组合——稀疏化减少需要处理的 KV 数量，MPK 消除处理这些数据的 kernel launch 开销，两者叠加效果值得探索。

### 下一步 Action

- [ ] 在本地环境安装 MPK（`pip install mirage-project`），用 Qwen3-8B 跑 benchmark，复现 1.2–1.7× 延迟改善数据
- [ ] 阅读 Mirage OSDI 2025 论文（Wu 等），理解 thread-block 级超优化器的工作原理，厘清 MPK 与之的关系（MPK 是运行时，OSDI 论文是编译器）
- [ ] 研究 NVSHMEM 的编程模型，理解 MPK 多 GPU 模式的通信实现细节
- [ ] 对比 NanoFlow 的 nano-batch 方法与 MPK 的 megakernel 方法，分析在不同 batch size 和 model size 下的适用场景


