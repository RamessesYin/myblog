---
title: "CUDA-L2：强化学习@cuBLAS"
tags:
  - CUDA
  - 强化学习
  - GPU优化
  - 矩阵乘法
  - LLM代码生成
---

## 🧠 核心摘要 (TL;DR)
> CUDA-L2 将大语言模型与多阶段强化学习相结合，自动生成半精度（FP16）矩阵乘法 CUDA 内核，在 A100 上对 1,000 种矩阵规模配置的离线评测中，分别比 torch.matmul 快 22.0%、比 cuBLAS 快 19.2%、比 cuBLASLt-AutoTuning 快 11.4%；服务器模式下增益更高，胜率达 79.3%~95.7%。

## 🏷️ 元数据
- **论文标题**: CUDA-L2: Surpassing cuBLAS Performance for Matrix Multiplication through Reinforcement Learning
- **发表机构/会议**: DeepReinforce Team / arXiv
- **年份**: 2025
- **链接**: [[CUDA-L2_paper|📄论文]] | [💻 Code](https://github.com/deepreinforce-ai/CUDA-L2)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #cuda #reinforcement-learning #gpu-optimization #matmul #llm-code-generation #hgemm
- **前置知识 (Prerequisites)**: [[KV稀疏调研]]
- **相关节点 (Related Notes)**:
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

高性能矩阵乘法（GEMM）是深度学习推理与训练的核心算子，也是 GPU 性能优化最难啃的硬骨头之一。现有的优化路径主要有三类：

1. **手工调优的供应商库**（cuBLAS / cuBLASLt）：由 NVIDIA 工程师针对不同 GPU 架构和矩阵规模精心编写，性能接近硬件极限，但维护成本极高，每代新硬件都需要重新开发。
2. **编译器驱动的自动调优**（TVM / Ansor / AutoTVM）：利用搜索算法或 ML 代价模型在参数化调度空间中寻找最优配置。但搜索空间受限于预设的调度原语，难以探索手工库使用的 PTX 汇编、CUTLASS 模板等底层优化技巧。
3. **规则/模板驱动的代码生成**（Triton、CUTLASS）：提高了开发效率，但仍然需要人类工程师理解并编写正确的优化策略。

这三条路径都面临共同的天花板：**优化策略由人类先验决定，不能自适应地探索远超人类直觉的内核实现空间**。

### 论文的切入点

CUDA-L2 的核心假设是：**用大语言模型（LLM）充当"代码作者"，用强化学习（RL）驱动其探索超过人类工程师能手工覆盖的 CUDA 实现空间**。具体而言：

- LLM 能理解 CUDA C/C++、CuTe 算子、内联 PTX 汇编、CUDA intrinsics、CUTLASS 模板等多种编程范式，具备写出"看起来合理"的内核代码的能力；
- RL 以**实测执行速度**作为奖励信号，迫使 LLM 不断生成更快的实现，而无需人类标注"正确答案"；
- 两者结合可以绕过人类编写优化代码的经验瓶颈，在巨大的配置空间中进行自动化探索。

这篇论文是对其前序工作 **CUDA-L1**（Li et al., 2025，同一团队）的重要延伸。CUDA-L1 使用对比式强化学习（Contrastive RL）在通用 CUDA 内核生成上建立了基础能力，CUDA-L2 在此基础上引入更系统的预训练数据、三阶段渐进 RL、NCU profiling 指标集成与 RAG 上下文，专攻半精度矩阵乘法（HGEMM）这一最重要的算子。

### 相关工作

#### 🏛️ 先驱/奠基工作

| 论文 | 贡献 | 相关性 |
|------|------|--------|
| **TVM** (Chen et al., OSDI 2018) | 端到端深度学习编译器，通过 ML 代价模型引导张量程序搜索 | CUDA-L2 超越的传统自动调优范式的奠基之作 |
| **AutoTVM / Learning to Optimize Tensor Programs** (Chen et al., NeurIPS 2018) | 用学习型代价模型替代人工调度，在多硬件上接近手工库性能 | 直接先驱，定义了"ML 驱动 kernel 优化"的问题框架 |
| **Ansor** (Zheng et al., OSDI 2020) | 层次化采样 + 进化搜索 + 学习代价模型，在 CPU/GPU 上超越 AutoTVM | 搜索效率提升的重要里程碑，但仍受限于调度原语空间 |
| **AlphaCode** (Li et al., Science 2022) | 首次在竞赛级编程中达到人类水平，验证了 LLM 的代码生成能力 | 奠定了"LLM 能写复杂代码"的认知基础 |
| **DeepSeek-R1** (DeepSeek-AI, 2025) | 纯 RL 训练出强推理 LLM，无需人工标注推理轨迹 | CUDA-L2 RL 训练范式的重要参考，验证了 RL 驱动 LLM 自我提升的可行性 |

#### ⚔️ 直接对比方法（同时期并行工作）

| 论文 | 方法 | CUDA-L2 的优势 |
|------|------|---------------|
| **KernelBench** (Ouyang et al., arXiv 2502.10517, 2025) | 构建包含 250 个 PyTorch ML 算子的 GPU 内核编写基准，测试 frontier LLM 的 kernel 生成能力 | KernelBench 发现 frontier 模型在 <20% 情形下才能匹配 PyTorch 基线；CUDA-L2 通过专门 RL 训练将性能推过 cuBLAS |
| **The AI CUDA Engineer** (Lange et al., 2025) | 利用 LLM 自动生成并迭代优化 CUDA 内核 | CUDA-L2 强调多阶段渐进式 RL，以 NCU profiling 为辅助奖励，提供更系统的训练框架 |
| **KEVIN** (Baronio et al., 2025) | 多轮 RL 框架，生成多种类型的 CUDA 内核 | CUDA-L2 专注 HGEMM 的深度优化，在目标算子上取得更强的性能提升 |
| **SwizzlePerf** (Tschand et al., 2025) | 硬件感知 LLM 用于 GPU 性能优化 | CUDA-L2 通过 RL 环境中的直接硬件反馈实现闭环优化，不依赖静态硬件知识 |
| **cuBLASLt-AutoTuning** (NVIDIA) | 对最多 100 个候选 kernel 进行枚举评测，选择最快实现 | CUDA-L2 在 1,000 个配置上的胜率为 79.3%~95.7%，系统性地超越 AutoTuning |

#### 📈 背景技术：RL 与 LLM 的融合

- **DeepSeekMath** (Shao et al., 2024)：将 RL 用于数学推理，验证了奖励驱动的 LLM 自我提升范式；
- **GPT-4 Technical Report** (Achiam et al., 2023)：展示了大规模 LLM 在代码任务上的能力边界；
- **Llama 3** (Dubey et al., 2024)：提供了高质量开源 LLM 作为 CUDA-L2 的基础模型选项；
- **DeepSeek-V3** (Liu et al., 2024)：与 DeepSeek-R1 共同构成 CUDA-L2 实验中使用的 LLM 后端基础。



---

## ⚙️ 核心设计与机制

### 整体架构：LLM × 三阶段 RL

CUDA-L2 的系统架构可以分为两大主体：**代码生成器**（LLM）和**训练驱动器**（RL 环境）。两者通过以下管线闭环交互：

```
┌─────────────────────────────────────────────────────────┐
│  CUDA-L2 训练管线                                        │
│                                                          │
│  ┌──────────────────┐   生成 CUDA C/C++/PTX/CUTLASS      │
│  │  LLM (Base Model │  ──────────────────────────►       │
│  │  + CUDA 预训练)  │                                     │
│  └──────────────────┘  ◄──────────── RL 梯度更新          │
│           ▲                                              │
│           │  奖励信号 r(custom)                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │  RL 评测环境                                        │  │
│  │  1. nvcc 编译 .cu 文件                              │  │
│  │  2. 在 A100 上运行，记录执行时间 t_custom           │  │
│  │  3. 数值正确性校验 (max element-wise diff)          │  │
│  │  4. NCU profiling 辅助指标                          │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 关键技术点 1：三阶段渐进式强化学习训练

CUDA-L2 相比 CUDA-L1 最重要的架构创新是**从通用到专用的三阶段渐进训练策略**：

**阶段一：持续预训练（Continual Pretraining）**
- 在约 1,000 个 CUDA 内核代码示例上进行持续预训练
- 数据来源：PyTorch 算子库、CUTLASS 模板库等高质量 CUDA 代码
- 目标：让 LLM 获得扎实的 CUDA 编程语法与常见优化模式的先验知识，避免 RL 阶段在"写不出可编译代码"上浪费探索预算

**阶段二：通用内核 RL 训练（General Kernel RL）**
- 在多种类型的 CUDA 算子（不限于 GEMM）上进行对比式强化学习
- 使用 CUDA-L1 中定义的对比学习策略，跨多种算子类型建立"更快的内核实现应得到更高奖励"的通用优化能力
- 目标：让模型学会在真实 GPU 执行反馈中迭代改进，建立通用的"CUDA 优化元认知"

**阶段三：HGEMM 专项 RL 训练**
- 专门针对半精度矩阵乘法（Half-precision GEMM，HGEMM）进行深度优化
- 配置空间：M, N, K 各取 {64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384}，共 10³ = 1,000 个配置组合
- 每次 RL 迭代中，从配置空间中采样矩阵规模，LLM 生成对应的内核代码，评测后返回奖励
- 目标：将通用优化能力收敛到 HGEMM 这一最关键算子上，精细化性能提升

这种"宽→窄"的课程式学习（Curriculum Learning）设计使模型先建立广泛的 CUDA 优化经验，再集中攻克 HGEMM，避免了从零开始在特定算子上的低效随机探索。

### 关键技术点 2：奖励函数设计

CUDA-L2 的奖励函数同时考虑**执行速度**、**数值正确性**和**代码简洁性**三个维度：

$$r(\text{custom}) = \frac{1}{N} \sum_{i=1}^{N} \left[ \frac{t_{\text{ref}}}{t_{\text{custom}}} - \alpha \cdot \text{diff} \right] - \beta \cdot L(\text{custom})$$

各项含义：

| 符号 | 含义 |
|------|------|
| $t_{\text{ref}}$ | 参考实现（如 cuBLAS）的执行时间 |
| $t_{\text{custom}}$ | LLM 生成的自定义内核执行时间 |
| $\text{diff}$ | 与参考实现输出的最大逐元素偏差（数值正确性惩罚） |
| $L(\text{custom})$ | 生成代码的长度（冗余代码惩罚） |
| $\alpha, \beta$ | 惩罚项权重超参数 |
| $N$ | 每次评测的矩阵规模采样数 |

**设计意图分析**：
- 速度项 $t_{\text{ref}}/t_{\text{custom}}$ 是比率形式，使奖励在不同矩阵规模下具有可比性（小矩阵基线时间短，大矩阵基线时间长，比率统一量纲）；
- 正确性惩罚 $\alpha \cdot \text{diff}$ 确保模型不会为了速度牺牲数值精度（对深度学习推理而言 FP16 精度至关重要）；
- 代码长度惩罚 $\beta \cdot L$ 引导模型生成简洁高效的实现，避免堆砌无效代码。

### 关键技术点 3：NCU Profiling 指标集成

CUDA-L2 的 RL 环境不仅记录端到端执行时间，还集成了 **NVIDIA Nsight Compute（NCU）的 profiling 指标**，包括：

- **内存带宽利用率**（Memory Bandwidth Utilization）：识别内存瓶颈
- **SM 占用率**（SM Occupancy）：反映 warp 并发度
- **L2 缓存命中率**：识别数据局部性问题
- **Tensor Core 利用率**：衡量是否充分利用 A100 的 FP16 加速单元

这些指标以辅助奖励或状态特征的形式注入 RL 训练，帮助模型理解"为什么一个内核比另一个快"，而非仅仅靠盲目搜索。

### 关键技术点 4：RAG 检索增强上下文

在每次 LLM 代码生成时，CUDA-L2 引入**检索增强生成（RAG）机制**：
- 从已有的高性能 CUDA 内核库中检索与当前矩阵规模最相似的成功案例；
- 将检索到的高性能内核代码片段作为 few-shot 示例注入 prompt，引导 LLM 复用已验证的优化模式（如 warp-level 数据搬运、double buffering、swizzle 地址布局等）；
- 这减少了从头生成低效代码的概率，显著提升 RL 探索效率。

### 关键技术点 5：代码多样性支持

CUDA-L2 生成的内核覆盖多种 CUDA 编程范式，包括：

- **CUDA C/C++**：标准 CUDA 编程接口
- **CuTe**：NVIDIA CUTLASS 中的 tensor layout DSL，用于高效描述张量操作的内存访问模式
- **内联 PTX 汇编**（Inline PTX Assembly）：直接操作 GPU 指令集，绕过高层抽象
- **CUDA intrinsics**：半精度运算、向量化 Load/Store 等底层接口
- **CUTLASS 模板**：工业级高性能 GEMM 模板库

这种多范式支持使模型能够发现不同层次的优化机会，而不受单一编程模型的限制。



---

## 📊 实验与性能评价

### 实验配置

- **硬件**：NVIDIA A100 Ampere GPU（支持 FP16 Tensor Core，理论峰值 FP16 FLOPS 可达 312 TFLOPS）
- **矩阵规模**：M, N, K 分别取 {64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 16384}，覆盖 10³ = **1,000 个配置组合**，代表从小批次推理到大批次训练的全谱场景
- **数据类型**：FP16 半精度（HGEMM）
- **矩阵布局**：NN 和 TN 两种 layout（论文对两种均有评测）

### 基线对比

| 基线 | 说明 |
|------|------|
| **torch.matmul** | PyTorch 默认矩阵乘法，底层调用 cuBLAS |
| **cuBLAS（NN / TN）** | NVIDIA 手工优化 BLAS 库的直接调用 |
| **cuBLASLt-heuristic** | cuBLASLt 启发式选择最优算法 |
| **cuBLASLt-AutoTuning** | 枚举最多 100 个候选 kernel，选最快实现 |

### 主要实验结果

**离线模式**（内核连续运行，无间歇，代表算力压满的训练场景）：

| 基线 | 平均速度提升 | 胜率（Win Rate） |
|------|------------|----------------|
| torch.matmul | **+22.0%** | - |
| cuBLAS-max | **+19.2%** | - |
| cuBLASLt-heuristic | **+16.8%** | - |
| cuBLASLt-AutoTuning | **+11.4%** | **79.3%** |

**服务器模式**（内核随机间隔运行，模拟真实推理服务场景，涉及 GPU 热态恢复效应）：

| 基线 | 平均速度提升 |
|------|------------|
| torch.matmul | **+28.7%** |
| cuBLAS-max | **+26.0%** |
| cuBLASLt-heuristic | **+22.4%** |
| cuBLASLt-AutoTuning | **+15.9%** |

服务器模式的提升幅度**普遍高于离线模式**（约 4-7 个百分点），原因是：推理服务中 GPU 并非持续满负载，存在 thermal/frequency 波动；cuBLAS 等库对某些频率状态下的性能退化更敏感，而 CUDA-L2 生成的内核鲁棒性更好。

### 性能随矩阵规模的变化趋势

论文揭示了一个重要规律：**CUDA-L2 的优势在小矩阵规模下更显著**。

- $\log_2(M \times N \times K) \approx 18$~$20$（小矩阵）：CUDA-L2 内核速度约为基线的 **1.4 倍**
- 随着矩阵规模增大，优势逐渐收窄，趋近基线性能

**原因分析（推断）**：
- 大矩阵场景下，GPU 计算单元几乎全部饱和，内存访问模式对性能影响减小，cuBLAS 等精心设计的大矩阵策略难以被超越；
- 小矩阵场景中，tile 大小选择、shared memory bank conflict 规避、prefetching 策略对性能影响更敏感，LLM 探索出的非传统实现有更多发挥空间。

这一发现对实际部署有重要启示：在 LLM 推理服务中，batch size 通常较小（尤其是延迟敏感场景），恰恰是 CUDA-L2 优势最大的区间。

### 局限性 (Limitations)

1. **仅支持 HGEMM 单一算子**：目前 CUDA-L2 专注于半精度矩阵乘法，尚未覆盖 Flash Attention、LayerNorm、SoftMax 等同样关键的推理算子；
2. **仅在 A100 上评测**：不同 GPU 架构（如 H100 Hopper、RTX 4090 Ada Lovelace）的 SM 架构、Tensor Core 规格、shared memory 层次不同，CUDA-L2 生成的内核是否可迁移、迁移后是否仍优于 cuBLAS，尚未验证；
3. **编译依赖 nvcc**：需要完整 CUDA 工具链，推理服务部署时须考虑编译开销（尽管内核可以预编译离线部署）；
4. **RL 训练成本高**：评测环境需要在真实 GPU 上运行内核并计时，训练成本显著高于仅依赖语言模型推理的方法；
5. **泛化性问题**：1,000 个训练配置虽然覆盖面广，但对于特殊非标准规模（如 M=3000, N=7000, K=300）的适用性需进一步验证。

---

## 💡 启发与我的想法

### 可借鉴之处

1. **RL × LLM 的闭环优化范式**：CUDA-L2 的核心洞察——用真实硬件执行时间作为奖励、LLM 作为策略网络——是一种极其通用的自动优化框架。不局限于 GEMM，理论上可扩展到任何"可测量的性能指标"作为 reward 的优化场景，例如 Flash Attention 内核、通信算子、量化内核。

2. **渐进式课程学习（宽→窄）**：CUDA-L2 的三阶段训练（通用 CUDA 预训练 → 多算子 RL → HGEMM 专项 RL）是一个值得借鉴的范式。先训练通用能力再专项化，比从零开始训练专项任务收敛更快、探索效率更高。这与 DeepSeek-R1 的两阶段冷启动策略有相似之处。

3. **NCU 指标的辅助奖励**：将 GPU profiling 工具的指标（带宽利用率、Tensor Core 利用率等）融入 RL 奖励，是提高训练可解释性和效率的好主意。未来可以探索将 profiling 信息作为 state 特征注入 LLM context，帮助模型理解"为什么当前内核不够快"。

4. **服务器模式 vs 离线模式**：区分两种评测模式是工程上值得学习的严谨性——实际推理场景并非连续满负载，intermittent kernel launch 下的性能表现与 sustained benchmark 有显著差异，这一区分对系统优化研究普遍适用。

### 改进空间

1. **多算子联合优化**：HGEMM 只是 LLM 推理栈中的一个算子。更有价值的方向是在 transformer 的"算子融合"层面做 RL 优化，例如将 QKV projection + Attention + Projection 的多算子融合成一个高效 CUDA 内核，整体 E2E 延迟的优化空间更大。

2. **跨架构迁移**：当前实验仅在 A100 上进行。考虑到 H100 的 Hopper 架构引入了 TMA（Tensor Memory Accelerator）和 wgmma 指令，CUDA-L2 的 RL 框架能否快速迁移到新架构是重要验证点。一种可能的改进是在奖励函数中加入硬件特征的架构感知项。

3. **RL 训练效率**：每次 reward 评测需要真实 GPU 运行时间，训练成本较高。可以探索"轻量代理模型预估性能 → 高置信度时再做真实 GPU 验证"的两阶段采样策略，降低 GPU 时间消耗。

4. **与 LLM 推理服务的端到端集成**：将 CUDA-L2 生成的内核集成到 vLLM、TensorRT-LLM 等推理框架，测量对实际 LLM 服务 QPS / TTFT 的影响，比单算子 microbenchmark 更有说服力。

5. **FP8/INT8 扩展**：随着 FP8 在 H100/H200 上的普及，将 CUDA-L2 扩展到 HGEMM-FP8 优化是自然的下一步，可能带来更大的性能提升空间（因为 FP8 GEMM 的调优生态尚不成熟）。

### 下一步 Action
- [ ] 阅读 CUDA-L1 论文（Li et al., 2025），理解对比 RL 训练框架与 CUDA-L2 的具体差异
- [ ] 尝试运行 github.com/deepreinforce-ai/CUDA-L2 的开源代码，复现在 A100 上超越 cuBLAS 的结果
- [ ] 调研 KernelBench (arXiv:2502.10517) 的 250 个 benchmark 算子，评估当前 SOTA（包括 CUDA-L2）在哪些算子上仍有明显优化空间
- [ ] 思考：能否将 CUDA-L2 的 RL 框架应用到 Flash Attention 内核优化？reward 如何设计？


