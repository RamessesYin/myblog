---
title: "Intra-Kernel Profiler"
tags:
  - GPU profiling
  - CUDA
  - performance analysis
  - NVBit
  - CUPTI
  - kernel optimization
---

## 🧠 核心摘要 (TL;DR)
> IKP（Intra-Kernel Profiler）是一个面向 CUDA 内核的**区域级性能剖析框架**，突破了传统工具（如 Nsight Compute）只能给出整个 kernel 聚合指标的局限，通过三套互补的后端（Trace Profiler、NVBit 区域分析器、CUPTI 采集器）将指令计数、内存流量、硬件计数器、stall 原因等数十个指标精细归因到用户命名的代码区域（region），并汇聚到一个自包含的交互式 HTML 仪表盘（IKP Explorer）中可视化呈现，是当前最细粒度的 GPU 内核开源剖析工具之一。

## 🏷️ 元数据
- **项目名称**: Intra-Kernel Profiler (IKP)
- **发布机构**: yao-jz (Albert)，GitHub 个人项目
- **年份**: 2026（首次提交 2026-03-25）
- **链接**: [[IntraKernelProfiler_blog|📄项目主页]] | [💻 Code](https://github.com/yao-jz/intra-kernel-profiler)
- **License**: Apache 2.0

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #gpu-profiling #cuda #nvbit #cupti #kernel-optimization #performance-analysis #warp-timing #roofline
- **前置知识 (Prerequisites)**: [[MSched]], [[ZipServ]]
- **相关节点 (Related Notes)**: [[KV稀疏调研]], [[AttentionRes]]

---

## 🎯 动机与痛点

### 现有方案的瓶颈

GPU 内核性能优化面临一个根本性问题：**现有工具给出的是每次 kernel 启动的聚合指标，无法区分同一个 kernel 内不同代码段（数据加载、计算核心、结果写回）各自贡献了多少时间、消耗了多少带宽**。

以 NVIDIA Nsight Compute 为例，其报告一次 kernel 启动的总 FLOPS、总 L2 访问字节、总 stall 周期等，这些数据对于找到"哪段代码出了问题"收效甚微：

- 一个 GEMM kernel 中，load 阶段、矩阵乘阶段、epilogue 写出阶段的内存访问模式完全不同，聚合在一起的 cache miss 率无法定位热点。
- Attention 内核中，QK 乘法段是 compute-bound，softmax 归一化段是 memory-bound，混合在一起的 roofline 分析会给出误导性的算术强度。
- 大模型推理服务（如 FlashAttention、PagedAttention）中，开发者手动插 `clock64()` 来测量子阶段耗时，但这种方式缺乏系统性，无法同时收集硬件计数器信息。

**痛点总结**：

| 传统工具 | 局限性 |
|---------|--------|
| Nsight Compute | 整个 kernel 聚合指标，无法 per-region 归因 |
| nvprof (已废弃) | 同上，且不支持新架构 |
| 手写 clock64() | 只能测时间，无法收集 instruction mix / stall 等 |
| perf / VTune | 不支持 GPU warp 级时间戳 |

### 论文的切入点

IKP 的核心思想是：**让开发者在 CUDA 内核代码中插入"区域标记"（region markers），然后由多套后端各自采集该区域的多维指标，最终在一个统一的仪表盘中联合展示**。

开发者只需在感兴趣的代码段前后分别调用：
```cpp
IKP_TRACE_REC_B(region_id);   // region begin
// ... 核心代码 ...
IKP_TRACE_REC_E(region_id);   // region end
```
或者通过 NVBit 标记：
```cpp
IKP_NVBIT_REGION_BEGIN(region_id);
// ... 核心代码 ...
IKP_NVBIT_REGION_END(region_id);
```

这种侵入式但低开销的设计，使得指标采集能够**精确到用户定义的代码边界**。

---

### 相关工作

#### 先驱/奠基工作

**NVBit（NVIDIA Binary Instrumentation Tool, MICRO 2019）**
NVBit 是 IKP 的核心依赖之一。它是 NVIDIA 开源的 GPU 二进制插桩框架，允许在不重新编译的情况下对预编译 CUDA 程序进行动态 SASS 级别插桩。NVBit 支持创建动态指令计数器、指令追踪器、内存访问追踪器等工具，是 IKP 的 NVBit Region Profiler 后端的基础。IKP 在其之上扩展了"区域栈"（region stack）机制，将 SASS 指令执行与用户标记的代码区域相关联。

**CUPTI（CUDA Profiling Tools Interface）**
CUPTI 是 NVIDIA CUDA Toolkit 随附的官方性能工具接口，提供 PC 采样、SASS 指标、PM（性能监控）采样等能力。IKP 利用 CUPTI 的 injection-based 采集模式（无需修改源码）将硬件计数器精细化到每个 PC 地址，再通过与内核区域的 PC 范围映射，实现 per-region 的硬件指标归因。

**Chrome Tracing / Perfetto**
Chrome Tracing 格式（`.json`）是 IKP Trace Profiler 的输出格式之一，与 Google 的 Perfetto UI 兼容。这一选择使得 GPU warp 时间轴可以与 CPU 侧的事件（如 kernel 启动、内存拷贝）放在同一个时间轴上对比，是 LLM 推理系统（如 vLLM）调试中常用的工具链。

#### 直接对比方法

**NVIDIA Nsight Compute（NCU）**
当前最主流的 CUDA kernel 性能分析工具。提供极为丰富的硬件计数器（Memory throughput、Roofline、Warp Stall、Instruction Statistics 等），但**粒度是整个 kernel**。NCU 不支持用户在 kernel 内部划分分析区域。此外，NCU 需要以特权模式运行（或需 root / Docker），部署门槛较高。

**NVIDIA Nsight Systems（NSYS）**
面向系统级 trace 的工具，可以追踪 GPU kernel 时间线、CPU-GPU 协同、NVLink 流量等，但 NSYS 的 GPU profiling 粒度同样停留在 kernel 级别，不支持 intra-kernel 区域分析。

**手动 clock64() 插桩**
最原始的方案：开发者在 CUDA 核内调用 `__globaltimer()` 或 `clock64()` 手动记录时间戳，然后将时间戳写回 host 内存分析。这种方式灵活但缺乏系统性：无法同时收集硬件计数器、没有自动化可视化、对 warp 级别的采样逻辑需要自行设计。IKP 的 Trace Profiler 正是对这一方案的系统化封装。

**GPUMechDB / MGPUSim**
学术界有一些更底层的 GPU 模拟器（MGPUSim）或微架构数据库（GPUMechDB）用于分析 GPU 内核的微架构行为，但它们运行的是模拟而非真实硬件，无法反映 A100/H100 等实际芯片上的 stall 分布。

#### 应用场景中的 Follow-up

**FlashAttention（arXiv:2205.14135, Dao et al., 2022）**
FlashAttention 通过 IO-Aware 的 tiling 策略显著降低了 Attention kernel 的 HBM 访问，使 BERT-large 推理提速 15%，GPT-2（1K token）提速 3×。FlashAttention 的性能优化正是依赖于对 kernel 内部"从 HBM 加载 Q/K/V"、"SRAM 上的 QK 矩阵乘"、"softmax 归一化"、"写出 O"等阶段的精细分析。IKP 这类 intra-kernel 剖析工具，能够为此类手工优化提供精确的数据支撑。

**FlashAttention-2（arXiv:2307.08691, Tri Dao, 2023）**
在 FlashAttention 基础上进一步优化 warp 间工作分配，减少 non-matmul 开销，在 A100 上达到理论峰值 FLOPs 的 50-73%。这类 warp 级并行优化的验证，恰好是 IKP Trace Profiler 的典型用例——通过 per-warp 时间戳观察 warp 间的负载均衡与 stall 分布。

**vLLM / PagedAttention（arXiv:2309.06180, Kwon et al., 2023）**
vLLM 通过 PagedAttention 将 KV Cache 以块为单位管理，消除内存碎片，对 GPU 显存利用率与 kernel 调度提出了新挑战。在 vLLM 这类系统中，profiling 需要追踪每个 attention kernel 内部的内存访问模式，正是 IKP 所能支持的能力。


---

## ⚙️ 核心设计与机制

IKP 由三套互补的剖析后端和一个统一的可视化前端组成。三套后端各有侧重，分别解决时序精度、指令级归因和硬件计数器采集三个不同维度的问题。

### 架构总览

```
用户 CUDA 内核 (注有 region 标记)
         │
    ┌────┴─────────────────────────┐
    ▼                              ▼
Trace Profiler              NVBit Region Profiler
(lock-free ring buffer)     (SASS-level 插桩)
纳秒级 per-warp 时序           指令类型 / 内存访问
    │                              │
    │           CUPTI Collectors   │
    │           (PC 采样 / SASS    │
    │            指标 / PM 采样)   │
    └──────────────┬───────────────┘
                   ▼
             IKP Explorer
        (自包含 HTML 仪表盘)
    Overview / Line / Regions /
    Execution / Memory / Stalls / Trace
```

### 关键技术点 1：Trace Profiler — 纳秒级 Per-Warp 时序记录

**设计目标**：以极低开销（~1% kernel 减速）记录每个 warp 进出各 region 的精确时间戳。

**实现机制**：
- 在 device 端为每个 warp 分配一个**环形缓冲区（ring buffer）**，每个事件固定占用 16 字节：
  ```
  [timestamp (8B)] [region_id (2B)] [event_type (2B)] [sm_id (2B)] [block_id (2B)]
  ```
- 时间戳来源：`__globaltimer()`（又称 globaltimer，精度约 1 ns，以 GPU 时钟周期计）
- 缓冲区大小计算公式：`capacity = 2 × regions_per_iteration × iterations + 2 × outer_regions`，需要开发者在启动配置时预估。
- 环形缓冲区写入采用无锁（lock-free）设计，避免 warp 间竞争导致的额外 stall。
- Host 侧通过 CUDA 内存拷贝读取所有 warp 的缓冲区数据，再做后处理。

**输出格式**：
1. **Chrome Trace JSON**（与 Perfetto / chrome://tracing 兼容）：每个事件表示为 complete event，携带 SM ID、block index、warp index 元数据，可在同一时间轴上展示多个 warp 的执行时序。
2. **Summary JSON**：每个 region 的统计汇总，包括 count、mean duration、coefficient of variation（CV）、P50/P95/P99 等百分位数、duration 直方图。

**Device API 宏**：
```cuda
// Session 管理（host 侧）
IKP_TRACE_SESSION_BEGIN(config);
IKP_TRACE_SESSION_END(session);

// 内核插桩（device 侧）
IKP_TRACE_REC_B(region_id);   // Begin: 记录进入该 region 的时间戳
IKP_TRACE_REC_E(region_id);   // End: 记录退出该 region 的时间戳  
IKP_TRACE_REC_M(region_id);   // Mark: 记录中间里程碑点
```

**典型 overhead**：约 1% kernel 减速（远低于 NVBit 的 5-50%+ 模式）。

### 关键技术点 2：NVBit Region Profiler — SASS 级指令归因

**设计目标**：将每条被执行的 SASS 指令精确归因到用户定义的 region，从而得到每个 region 的指令混合比（instruction mix）、内存访问模式、分支发散率等。

**实现机制**：
- 依赖 NVBit 框架对已编译的 CUDA kernel 进行**动态二进制插桩**，不需要重新编译，但需要以 `-rdc=true` 编译 kernel（支持 relocatable device code）。
- NVBit 维护一个 **per-warp region 栈**（region stack），当 warp 执行到 `IKP_NVBIT_REGION_BEGIN(id)` 时压栈，执行到 `IKP_NVBIT_REGION_END(id)` 时出栈，栈顶即为当前 region ID。
- 对每条被执行的 SASS 指令，NVBit 注入统计代码，将该指令的类型、是否访存、访存地址等属性记录到对应的 region 计数器中。

**支持模式**：

| 模式 | 采集内容 | 额外 overhead |
|------|---------|--------------|
| `pcmap` | 指令 PC 与 region 的映射关系 | ~5-10% |
| `instmix` | 每 region 的指令类型分布（INT/FP/MEM/CTRL 等） | ~10-20% |
| `memtrace` | 每条 load/store 的地址、大小、warp mask | ~50%+（需降采样） |
| `all` | 以上全部 | 视具体模式 |

**Region 限制**：每个 kernel 最多支持 7 个并发 region（region ID 0-6），超出需分批或嵌套。

**与 Trace Profiler 的互补性**：Trace Profiler 给时间，NVBit 给指令结构，两者结合可以回答"这个 region 慢是因为 instruction count 太多，还是每条指令的 stall 太多"。

### 关键技术点 3：CUPTI Collectors — 硬件计数器精细采集

**设计目标**：在不修改 CUDA 源码的情况下，收集每个 PC 地址的硬件性能计数器，再映射到 region。

**四个 injection-based 采集模块**：

| 模块 | 采集内容 | 适用场景 |
|------|---------|---------|
| PC Sampling | 每个 PC 的采样频率，反映指令执行热点 | 定位最耗时的 SASS 指令 |
| SASS Metrics | 每条 SASS 指令的 warp 级指标（Issue cycles, Stall cycles, Active threads 等） | 分析管线瓶颈 |
| Instruction Execution | 每个 PC 的执行次数（精确计数，非采样） | 配合 instmix 验证 |
| PM Sampling (Hopper+) | 性能监控寄存器采样（需 H100 及以上架构） | L2/SM 级别精细分析 |

**Injection-based 的意义**：CUPTI injection 模式通过 LD_PRELOAD 等机制在 kernel 启动前注入采集逻辑，**不需要修改源码，也不需要以 root 权限运行 NCU**，大幅降低了在生产系统中 profiling 的门槛。

### 关键技术点 4：IKP Explorer — 统一交互式仪表盘

IKP Explorer 是一个**自包含的单页 HTML 文件**，无需服务器，直接在浏览器中打开即可交互。它整合了所有后端的输出，提供以下七个分析视图：

**① Overview（总览）**
- Kernel 基本信息（架构、SM 数量、thread block 配置）
- 自动检测并标注瓶颈类型（compute-bound / memory-bound / latency-bound）
- 各 region 的时间占比饼图

**② Line（源码行级分析）**
- 将指标映射到 CUDA 源码的每一行（需包含调试符号）
- 以热力图方式展示每行的 instruction count / execution time

**③ Regions（区域分析）**
- **核心视图**：每个 region 40+ 个字段，包括：
  - 执行次数、平均时延、P99 时延
  - 指令混合比（INT / FP32 / FP64 / MEM / CTRL）
  - L1/L2 命中率、HBM 带宽利用
  - 分支发散率（divergence %）、SIMT 效率
- 支持多 region 并排对比
- 可视化条形图和雷达图

**④ Execution（执行细节）**
- 指令类型分布（instruction mix）的详细对比图
- Basic block 执行热点可视化
- 与 PTX 和 SASS 的对照（按 region 着色）

**⑤ Memory（内存特性）**
- **Reuse Distance 分析**：绘制访存地址的重用距离分布，判断 cache 友好程度
- **Working Set 曲线**：展示 region 运行过程中累计访问的唯一地址数量
- **Inter-Warp Sharing 分析**：检测相邻 warp 是否访问同一 cache line（影响 bank conflict 和 L1 效率）

**⑥ Stalls（停顿分析）**
- 各 SASS 停顿原因（STALL_MIO_THROTTLE / STALL_LONG_SCOREBOARD / STALL_WAIT 等）的比例
- L2 访问效率与 coalescing 分析（访存合并率）

**⑦ Trace（时序分析）**
- Per-warp duration 直方图（识别长尾 outlier）
- Block-level 和 warp-level 热力图（发现 SM 间负载不均衡）
- 时间百分位表（P50/P90/P99/P99.9）

---

## 📊 实验与性能评价

### 对比基线（与 IKP 设计对比）

| 工具 | region 粒度 | 时序精度 | 指令归因 | 硬件计数 | 可视化 | 无需特权 |
|------|------------|---------|---------|---------|--------|--------|
| NVIDIA Nsight Compute | ❌ kernel 级 | ❌ | ❌ | ✅ | ✅ | ❌ (需 root/特殊权限) |
| NVIDIA Nsight Systems | ❌ kernel 级 | ✅ (ns 级系统时间轴) | ❌ | ❌ | ✅ | ✅ |
| 手写 clock64() | ✅ (用户自定义) | ✅ | ❌ | ❌ | ❌ | ✅ |
| NVBit（裸用） | ✅ (用户自定义) | ❌ | ✅ | ❌ | ❌ | ✅ |
| **IKP（本项目）** | **✅** | **✅ (~1ns)** | **✅** | **✅ (via CUPTI)** | **✅ (Explorer)** | **✅** |

### Trace Profiler Overhead

根据 README 描述：
- Trace Profiler：约 **1% kernel 减速**（仅时序记录，无指令插桩）
- NVBit pcmap 模式：约 **5-10%** 额外开销
- NVBit memtrace 模式（未采样）：约 **50%+**（建议配合采样率降低）
- CUPTI injection 模式：视采样频率，通常 **5-15%**

### 参考示例：Tiled GEMM 分析

项目 `examples/gemm/` 提供了一个完整的 tiled GEMM 示例，演示了 IKP 的典型分析流程：

1. **标记三个核心 region**：数据加载（global→shared memory）、矩阵乘（tensor core 或 CUDA core）、结果写出（shared→global memory）
2. **运行 Trace + NVBit + CUPTI**：收集各 region 的时序、指令比例和硬件计数器
3. **用 IKP Explorer 分析**：通过 Memory 视图发现数据加载阶段的 coalescing 问题，通过 Stalls 视图确认是 STALL_LONG_SCOREBOARD（L2 延迟等待）
4. **优化**：改变 tile size 或调整访存顺序后，重新 profiling 验证优化效果

**Explorer 预生成结果**可不挂 GPU 直接在浏览器打开体验：`examples/gemm/explorer.html`（用 Python HTTP server 本地服务）。

### 局限性（Limitations）

1. **Region 数量上限**：NVBit 模式下每个 kernel 最多 7 个并发 region（region ID 0-6），超出需要分批运行或层次化标记。
2. **NVBit 重编译要求**：使用 NVBit Region Profiler 需要以 `-rdc=true` 重新编译 CUDA kernel（支持 relocatable device code），对已有的第三方 .so 库无效。
3. **PM Sampling 架构限制**：PM Sampling 模块仅支持 Hopper 及以上架构（H100+），对 A100 及更早架构不可用。
4. **Ring Buffer 溢出**：若实际迭代次数超出预估的 ring buffer 容量，最早的事件将被覆盖。需要开发者在启动配置时合理估算。
5. **缺乏自动标注**：IKP 要求开发者手动插入 region 标记，不支持自动检测代码块边界（如编译器 pass 自动化）。
6. **闭源 kernel 不可用**：对于完全闭源的 GPU kernel（不提供源码或 PTX），NVBit 模式可部分工作（SASS 级插桩），但 region 标记无法注入。

---

## 💡 启发与我的想法

### 可借鉴之处

1. **三后端正交互补设计**：IKP 用 Trace（时序） + NVBit（指令结构） + CUPTI（硬件计数器）三条完全独立的数据采集链路，各自有独立的 overhead，可按需开启。这种设计避免了单一采集机制的过重开销，非常值得在系统设计中借鉴。

2. **Lock-free Ring Buffer 时序采集**：用 lock-free 环形缓冲区解决 warp 间并发写入的竞争问题，overhead 仅 ~1%，远优于传统的 atomic + global memory 记录方案。这一模式在需要低干扰的 GPU trace 场景中具有通用价值。

3. **自包含 HTML 仪表盘**：将所有分析逻辑封装到单页 HTML，不依赖后端服务，便于分享（发送一个文件即可）。这一设计思路对构建轻量级 LLM 推理调试工具也有参考价值。

4. **Region 语义层次**：IKP 的 region 不只是"开始-结束"，还维护了 per-warp 的 region 栈，支持嵌套 region 分析（外层 region 包含内层 region）。这对分析 FlashAttention 这类内部有多层循环结构的 kernel 很有用。

### 改进空间

1. **自动 region 标注**：目前 IKP 完全依赖手动插桩。理想情况下，可以结合编译器（Clang CUDA pass 或 NVCC 插件）自动在循环体、函数调用、内存访问前后插入 region 标记，降低使用门槛。

2. **与 LLM 推理框架集成**：IKP 目前是一个独立工具，与 vLLM、TensorRT-LLM 等框架没有集成。若能提供一个 Python 包装层，允许开发者直接从 PyTorch 或 CUDA Python 中调用 IKP session 管理接口，会大幅降低在 LLM 推理场景中的使用摩擦。

3. **跨 kernel 关联分析**：LLM 推理中一次 decode step 可能触发数十个 kernel（attention、LayerNorm、FFN 等），IKP 目前只能单独分析一个 kernel。若能将多个 kernel 的 region 时序串联在同一时间轴上，能更好地支持端到端性能归因。

4. **自动 Roofline 标注**：IKP 的 Regions 视图已有 40+ 字段，但尚未自动在 Roofline 模型坐标系上标注每个 region（compute-bound vs memory-bound 的精确边界）。加入自动 Roofline 绘图会使性能瓶颈诊断更直观。

5. **SM 90a 以外架构的完整支持**：README 明确提到 SM 90a（H100）有最佳支持，更早架构（A100/V100/RTX 系列）的测试覆盖有待完善。

### 下一步 Action

- [ ] 在个人的 FlashAttention 或 GEMM 复现代码中尝试接入 IKP Trace Profiler，观察 warp 时序分布
- [ ] 研究 NVBit pcmap 模式的输出格式，理解如何将 SASS PC 映射回 CUDA 源码行号
- [ ] 阅读 IKP 的 `docs/trace_format.md`，了解 Chrome Trace JSON 格式与 Perfetto 的对接细节
- [ ] 调研 CUPTI PC Sampling 的采样率配置对 overhead 的影响，找到精度与速度的 Pareto 最优点

