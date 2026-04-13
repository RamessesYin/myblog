---
title: "AttentionEngine"
tags:
  - attention
  - kernel-optimization
  - multi-backend
  - compiler
  - LLM-inference
---

## 🧠 核心摘要 (TL;DR)
> 现有注意力优化库（如 FlashAttention）高度硬件专用且只支持固定的注意力变体，难以适应快速扩张的注意力变种与多样硬件平台。AttentionEngine 将注意力机制抽象为「相关性评分」和「聚合」两个核心操作，通过可定制化模板 + 跨后端自动调度，在无需手工 kernel 的情况下，对现有库不支持的配置实现最高 **10.4×** 加速，同时在 NVIDIA H100 和 AMD MI250 上均实现高性能。

## 🏷️ 元数据
- **论文标题**: AttentionEngine: A Versatile Framework for Efficient Attention Mechanisms on Diverse Hardware Platforms
- **发表机构/会议**: ASPLOS 2025（上海交通大学、北京大学、微软研究院）
- **年份**: 2025
- **链接**: [[AttentionEngine_paper|📄论文]] | [💻 Code](https://github.com/microsoft/AttentionEngine)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #attention #kernel-optimization #multi-backend #compiler #llm-inference
- **前置知识 (Prerequisites)**: [[AttentionRes]], [[KV稀疏调研]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[MSched]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

注意力机制是现代大语言模型（LLM）的核心计算模块。随着序列长度增加，注意力的计算占比急剧上升：在 LLAMA-3B 中，序列长度为 2048 时注意力占推理时间的 55%，增至 8192 时则高达 **82%**。然而，高效注意力的实现面临三重困境：

1. **注意力变体爆炸**：从标准 Softmax Attention 到 Sigmoid Attention [Ramapuram et al., 2024]、ReLU Attention [Wortsman et al., 2023]、线性注意力（Mamba2 [Dao & Gu, 2024]、RetNet [Sun et al., 2023]）、差分注意力（Diff-Transformer [Ye et al., 2024]）等，各有不同的计算流程；
2. **硬件平台多样**：NVIDIA A100/H100 与 AMD MI300X 的 tile size、内存层次、流水线策略差异显著。FlashAttention v2 在 A100 上达到 70% 峰值算力，在 H100 上却仅有 30%；
3. **非标准维度支持缺失**：DeepSeek-V2 [2024] 使用 $d_{qk}=192, d_v=128$，Diff-Transformer 使用 $d_{qk}=128, d_v=256$，现有库必须 padding 至相同维度，引入额外开销。

**现有方案的具体局限**：

| 方案 | 局限性 |
|------|--------|
| FlashAttention-2/3 [Dao, 2023; Shah et al., 2024] | 仅支持标准 Softmax Attention，$d_{qk}=d_v$ 要求 |
| FlexAttention [Dong et al., 2024] | 基于模板，难以扩展到线性注意力 |
| FlashInfer [Ye et al., 2025] | 主要聚焦 LLM 推理 serving 场景，不覆盖训练 |
| TVM/Ansor [Chen et al., 2018; Zheng et al., 2020] | 通用编译器不理解注意力语义，无法应用 online softmax 等高级优化 |
| PyTorch Inductor | 性能远低于手工优化库 |

### 论文的切入点

AttentionEngine 的洞察是：**尽管注意力变体的实现千差万别，它们在语义上都可归结为两个基本操作**：
- **相关性评分（Relevance Scoring）**：计算 token 间的成对相似度；
- **聚合（Aggregation）**：使用评分对 Value 加权求和，生成上下文表示。

基于此抽象，论文提出了一套统一的可编程模板 + 自动调度框架，从而兼顾灵活性与性能。

### 相关工作

#### 奠基/先驱工作
- **Attention Is All You Need** [Vaswani et al., 2017]：提出 Transformer Q/K/V 架构，确立注意力机制的计算框架；
- **FlashAttention** [Dao et al., NeurIPS 2022]：首次引入 IO-aware 分块计算 + online softmax，将注意力 FLOPS 利用率提升至极高水平，成为工业界标准；
- **FlashAttention-2** [Dao, 2023]：改进并行策略和 warp 级工作分配，在 A100 上达到 50-73% 峰值 FLOPS；
- **FlashAttention-3** [Shah et al., 2024]：利用 Hopper GPU warp 专化、异步流水线和 FP8 支持，H100 上 FP16 达 740 TFLOPs/s，FP8 近 1.2 PFLOPs/s；
- **TVM** [Chen et al., OSDI 2018]：自动化端到端 DNN 编译器，支持 CPU/GPU/FPGA，是通用 kernel 生成的基础框架；
- **Ansor** [Zheng et al., OSDI 2020]：基于层次化搜索空间的自动化 tensor program 生成，超越手工优化；
- **Longformer** [Beltagy et al., 2020]：局部滑窗 + 全局注意力，将序列注意力复杂度降至线性，奠定稀疏注意力基础；
- **BigBird** [Zaheer et al., NeurIPS 2020]：结构化稀疏注意力，理论证明其与标准 Transformer 等价，支持 8× 更长序列。

#### 直接对比方法
- **FlexAttention** [Dong et al., 2024]：模板化方法，允许用 PyTorch 代码片段定义注意力变体，但难以泛化到线性注意力；
- **FlashInfer** [Ye et al., 2025]：专为 LLM 推理 serving 优化，支持块稀疏格式和 JIT 编译，集成到 vLLM/SGLang；
- **Flash-Linear-Attention (FLA)** [Yang & Zhang, 2024]：基于 Triton 的线性注意力库，覆盖 Mamba2、GLA 等多种线性注意力变体。

#### 注意力变体（实验覆盖的目标）
- **Mamba2 / SSM** [Dao & Gu, ICML 2024]：通过结构化状态空间对偶性（SSD）将 SSM 与 Transformer 统一，线性复杂度；
- **RetNet** [Sun et al., 2023]：保留网络，支持并行/递归/分块三种计算模式；
- **DeepSeek-V2** [2024]：Multi-head Latent Attention（MLA），使用非标准维度 $d_{qk}=192, d_v=128$；
- **Differential Transformer** [Ye et al., ICLR 2025]：差分注意力，两个 Softmax 注意力图相减以降噪；
- **Sigmoid Attention / FlashSigmoid** [Ramapuram et al., 2024]：用 Sigmoid 代替 Softmax，提升训练稳定性；
- **ReLU Attention** [Wortsman et al., 2023]：在 ViT 中用 ReLU 替换 Softmax，经序列长度归一化后可达到可比性能；
- **YOCO** [Sun et al., 2024]：解码器-解码器架构，KV cache 仅缓存一次，大幅降低显存；
- **SeerAttention** [Gao et al., 2024]：从 LLM 中学习内在块级稀疏模式，减少长文本推理延迟；
- **Random Feature Attention** [Peng et al., 2021]：用随机特征近似 Softmax 注意力，将复杂度降至线性。


---

## ⚙️ 核心设计与机制

### 架构总览

AttentionEngine 分为三层：**编程接口层（Programming Interface）→ 调度空间层（Scheduling Space）→ 后端代码生成层（Backend）**。

```
用户定义的注意力函数（Attention Variant API）
         ↓
   计算图构建（Computation Graph）
         ↓
   核心模板选择（Kernel Template）
         ↓
   两层调度（Tile Config + Tile Resource Scheduling）
         ↓
    TileLang / CUTE / Triton 后端代码生成
         ↓
  NVIDIA GPU（A100/H100）或 AMD GPU（MI250）执行
```

### 关键技术点 1：统一注意力抽象与可编程模板

**核心抽象**：所有注意力机制都可以分解为两个固定运算 + 两个可定制化函数：

$$\text{固定部分：} \quad S = \text{RelevanceScore}(Q, K) = Q \cdot K^T, \quad O = \text{Aggregate}(S, V) = S \cdot V$$

$$\text{可定制化函数：} \quad \text{modification}(·), \quad \text{row-wise normalization}(·)$$

AttentionEngine 提供两种**实例化计算模式**：

| 模式 | 适用场景 | 核心公式 |
|------|---------|---------|
| **Parallel Pattern（并行模式）** | 标准 Transformer 类注意力 | $O = \text{RowNorm}(\text{Mod}(Q \cdot K^T)) \cdot V$ |
| **Recurrent Pattern（递归模式）** | 线性注意力（Mamba2、RetNet 等） | 状态迭代：$h_t = \text{Mod}(k_t \cdot v_t) + h_{t-1}$，$o_t = q_t \cdot h_t$ |

**可定制化函数的编程接口**：

1. **modification 函数**：支持元素级变换与 masking，例如：
   - Causal mask：`score_mod = lambda scores: scores + causal_mask`
   - Query 缩放：`q_mod = lambda q: q / sqrt(d_k)`
   - ReLU Attention：`score_mod = lambda scores: max(scores, 0)`

2. **row-wise normalization 函数**：支持行级归约操作（reduceSum/reduceMax），并提供 **online row-norm 接口**，灵感来自 FlashAttention 的 online softmax 技术：
   - `online_prologue()`：初始化状态变量（如 `row_max = -inf, row_sum = 0`）
   - `online_fwd()`：逐块更新状态（数值稳定的增量 softmax）
   - `online_epilogue()`：最终归一化

**Softmax 注意力的 online 实现示例**（对应论文 Figure 7）：

```python
def online_prologue():
    row_max = -inf
    row_sum = 0

def online_fwd(row_max_prev, row_sum_prev):
    row_max_cur = scores.reduceMax()
    row_max = max(row_max_cur, row_max_prev)
    scores = exp(scores - row_max)
    row_sum_cur = scores.reduceSum()
    row_sum = exp(row_max_prev - row_max) * row_sum_prev + row_sum_cur
    return row_max, row_sum

def online_epilogue():
    scores = scores / row_sum
```

这套接口使用户无需了解 CUDA/Triton 细节，即可为新注意力变体定义高性能 kernel。

**典型注意力变体的映射示例**：

| 注意力变体 | modification | row-wise norm |
|-----------|-------------|---------------|
| Softmax Attention（LLAMA 标准） | `q_mod: q/sqrt(dk)` | online softmax |
| Causal Attention | `score_mod: mask + scores` | online softmax |
| ReLU Attention（ViT） | `score_mod: max(scores, 0)` | 无（row-wise sum 归一化） |
| RetNet Parallel | `score_mod: scores × mask` | 自定义数值稳定化 |
| Mamba2 SSM | 递归模式，`k_mod: k × gate` / `h_mod: h × decay × gate` | - |

### 关键技术点 2：调度空间（Scheduling Space）

调度空间由三个核心组件共同定义：

1. **Kernel Template（核心模板）**：固化注意力的计算流程骨架：
   - 并行模式 kernel 模板包含 online 技术，高效处理行级归一化；
   - 递归模式 kernel 模板利用 chunk 并行性，最大化张量核心利用率；
   - 支持多后端：TileLang（类 Triton 风格）和 CUTE（NVIDIA 汇编级）；

2. **IntermediateTensor（中间张量描述）**：描述计算过程中产生的瞬态数据：
   ```cpp
   class IntermediateTensor {
       TileShape tile;          // 分块形状
       MemoryLocation mem;      // 存储层次（寄存器/共享内存/全局内存）
       int pipelineStage;       // 流水线阶段数
   };
   ```
   AttentionEngine 通过计算图自动传播 tiling 方案，推导出 Q/K/V 及所有中间张量的分块形状。

3. **DeviceConfig（设备约束）**：封装硬件约束：
   ```cpp
   class DeviceConfig {
       BaseTileShape basetile;         // 最优 tile 形状（对齐 GEMM 指令）
       List<MemoryCapacity> memoryInfo; // 各层内存容量
   };
   ```

### 关键技术点 3：两层调度策略

AttentionEngine 采用**两层调度策略**最小化延迟：

**第一层：Tile Config 调度**（穷举 tile 配置空间）

```
TileConfigScheduling(G: Graph, D: DeviceConfig):
  tensor_tile_configs = InferPossibleTileConfigs(G, D.basetile)
  for tile_config in tensor_tile_configs:
    plans += TileResourceScheduling(tile_config, G.IntermediateTensors, D)
    best_plan = argmin(Profile(plan) for plan in plans)
  return best_plan
```

由于注意力机制的计算结构约束，tile 配置空间本身是受限的，使得穷举搜索可行。

**第二层：Tile Resource 调度**（贪心分配存储层次）

```
TileResourceScheduling(tile_config, t, D):
  InitMemLocation(t.memLoc, REGISTER)   # 所有中间张量初始化为寄存器
  t = sortByTensorSizeDec(t)            # 按张量大小降序排列
  for tensor_i in t:
    plans = GeneratePlans(t)
    plans = filter(ComputeMemoryConstraint(tile_config, t, plan, D.memoryInfo))
    if plans:
      return plans
    else:
      LowerMemLocation(tensor_i.memLoc)  # 降级到共享内存/全局内存
```

策略核心：**优先将所有中间张量放入寄存器**，减少内存 I/O；当寄存器不足时，按大小降序将张量逐步降级到共享内存或全局内存。

### 关键技术点 4：后端实现

**前端（Python/C++ 7.3K 行）**：
- 接受用户自定义函数，通过计算图追踪（DAG 形式）捕获张量依赖关系；
- 支持自动微分：每个节点包含 `grad` 指针，指向对应梯度节点；
- 计算原语分为元素级（add/sub/tanh 等）和行归约（reduceSum/reduceMax）。

**后端代码生成**：
- **表达式生成**：对计算图拓扑排序，生成内核无关的计算表达式序列；
- **代码生成**：通过字符串匹配将计算表达式映射到对应 kernel 模板的代码；
- **NVIDIA GPU 支持**：TileLang（Triton 风格）和 CUTE（thread-level 数据布局）两条后端；
- **AMD GPU 支持**：通过 TileLang 生成 MI250 优化 kernel，利用其 Matrix Core 和异步拷贝单元。

**关键优化：非标维度原生支持**

现有 FlashAttention 等库要求 $d_{qk} = d_v$，对非标维度模型必须 padding。AttentionEngine 的 kernel 模板原生支持不同 $d_{qk}$ 和 $d_v$，消除了 padding 开销，这是其对 DeepSeek-V2-Lite（$d_{qk}=192, d_v=128$）和 Diff-Transformer（$d_{qk}=128, d_v=256$）获得显著加速的根本原因。


---

## 📊 实验与性能评价

### 实验设置

**硬件平台**：
- NVIDIA H100（CUDA 12.4，Triton 2.3.1）
- AMD MI250（ROCm 6.2.4，Triton 3.1.0）

**测试注意力算子**（完整基准覆盖 10+ 种）：

| 类别 | 算子 | 代表模型 | 维度配置 |
|------|------|---------|---------|
| 并行模式 | Softmax Attention | LLAMA-3.1-8B | head=32, dqk=128, dv=128 |
| 并行模式 | Softmax Attention | DeepSeek-V2-Lite | head=16, dqk=192, dv=128 |
| 并行模式 | Softmax Attention | DiffTransformer-3B | head=12, dqk=128, dv=256 |
| 并行模式 | Sigmoid Attention | LLAMA-3-8B-style | head=32, dqk=128, dv=128 |
| 并行模式 | ReLU Attention | ViT-s/16-style | head=6, dqk=64, dv=64 |
| 并行模式 | Retention Parallel | RetNet-6.7B | head=32, dqk=256, dv=512 |
| 递归模式 | Mamba2 SSM | Mamba2-2.7B | headv=80, dqk=128, dv=64 |
| 递归模式 | Retention Recurrent | RetNet-6.7B | head=32, dqk=256, dv=512 |
| 递归模式 | Gated Retention | YOCO-13B | head=40, dqk=256, dv=256 |

**基线**：FlashAttention-2/3、FlashSigmoid、Mamba2 Triton kernel、Flash-Linear-Attention、FlexAttention、FlashInfer、PyTorch Inductor。

### 主要结果：NVIDIA H100 算子性能

**标准 Softmax Attention（LLAMA-3.1-8B）**：
- AttentionEngine 与 FlashAttention-3（手工优化基准）性能**相当**，验证了自动化框架无需牺牲性能。

**非标准维度 Softmax Attention（DeepSeek-V2-Lite / DiffTransformer-3B）**：
- 相比 FlashAttention，前向平均加速 **1.88×**，后向平均加速 **1.52×**；
- 根本原因：AttentionEngine 原生支持 $d_{qk} \neq d_v$，FlashAttention 必须 padding，引入额外计算。

**定制化注意力（Sigmoid / ReLU / Retention Parallel）**：
- 现有专家优化库不支持这些变体（FlashSigmoid 未针对 H100 优化，ReLU/RetNet 无融合 kernel）；
- AttentionEngine 相对 FlashSigmoid/PyTorch 实现，加速范围 **1.1× ∼ 10.4×**，平均 **3.6×**；
- 其中 SeerAttention 这类「现有库完全不支持」的配置，最高达到 **10.4×** 加速。

**Mamba2 SSM（线性注意力）**：
- 相比官方 Triton kernel，前向加速 **1.99×**，后向加速 **1.65×**。

**Gated Retention / RetNet（线性注意力递归）**：
- 相比 Flash-Linear-Attention，前向平均加速 **1.33×**，后向平均加速 **1.93×**。

### 端到端推理性能（NVIDIA H100）

在 DeepSeek-V2-Lite、Diff-Transformer-3B、Mamba2-2.7B、YOCO-160M 四个模型上：

- 推理平均加速 **1.4×**（FP16）；
- DeepSeek-V2-Lite 注意力占推理时间 85%，通过 AttentionEngine 将注意力算子加速 2.2×，端到端推理提升 **1.85×**。

### 端到端训练性能（NVIDIA H100）

在 DeepSeek-V2-Lite（$d_{qk} \neq d_v$）、ViT-S/16（ReLU Attention）、YOCO-160M 三个模型上：

- 训练平均加速 **1.4×**；
- ViT-S/16（ReLU Attention）因现有库完全缺失支持，实现 **1.7×** 加速。

### AMD MI250 GPU 评测

在 Softmax Attention、ReLU Attention、Mamba2、RetNet Recurrent 上：
- 相比 AMD 现有基线，前向平均加速 **3.3×**，后向平均加速 **2.0×**；
- 验证了 AttentionEngine 跨硬件平台扩展的能力。

### 局限性

1. **搜索代价**：调度策略依赖 profiling，对新硬件或新注意力变体，首次 JIT 编译需要搜索时间；
2. **表达能力边界**：AttentionEngine 的统一模板覆盖了实验中 10+ 种变体，但对更特殊的稀疏注意力模式（如非规则 block sparse）可能需要扩展；
3. **多 GPU/分布式**：论文仅评估单卡场景，跨 GPU 的分布式注意力（如 Ring Attention）未在框架内讨论；
4. **代码规模**：7.3K 行 C++/Python，工程实现有一定学习成本。

---

## 💡 启发与我的想法

### 可借鉴之处

1. **抽象层次设计**：将千变万化的注意力变体归结为「相关性评分 + 聚合」两步，是一个极为简洁且有力的抽象。这一思路与 FlashAttention 的「online softmax = 数值安全的增量更新」异曲同工——本质都是找到了算法的**不变核心**，把可变部分参数化。

2. **受限搜索空间的穷举**：核心调度的可行性建立在「注意力计算结构约束了 tile 配置空间」这一洞察上。当搜索空间足够小，穷举 + profiling 反而比启发式搜索更可靠。这对 KV Cache 稀疏化等其他优化场景的调度设计有参考价值。

3. **中间张量驱动的 tiling 传播**：通过 IntermediateTensor 抽象，将分块形状的推导自动化，避免了手工调优。这种「从计算图结构推导内存访问模式」的方法，是 compiler 化 kernel 生成的关键思路。

4. **向 AMD 生态延伸**：AMD MI250 上 3.3× 的前向加速，证明一套抽象确实可以跨越 GPU 生态。对于 AI 基础设施多元化（不再只依赖 NVIDIA H100）有重要的工程意义。

### 改进空间

1. **与 KV Cache 稀疏化的协同**：SeerAttention [Gao et al., 2024] 等工作通过学习注意力稀疏模式减少计算，AttentionEngine 的框架理论上可以支持这类稀疏注意力的高效 kernel 生成，但论文中未深入讨论；

2. **前向推理解码加速**：论文 Table 2 中提到 DeepSeek-V2-Lite 解码（Seqlen=1）的配置，但正文对解码阶段的优化策略着墨不多。GQA/MQA 解码的优化是 LLM 推理的重要场景；

3. **与量化的结合**：FlashAttention-3 已支持 FP8，AttentionEngine 未来若加入量化感知的 kernel 生成，可进一步降低内存带宽瓶颈；

4. **自动微分的完整性**：论文提到通过计算图编码梯度信息实现自动微分，但对于某些非标准的 backward 路径（如 Gated Retention 的复杂梯度），是否完整支持有待进一步验证。

### 下一步 Action
- [ ] 在 Mamba2/RetNet 等线性注意力模型的实现中，探索 AttentionEngine 与现有框架（vLLM、SGLang）的集成方式
- [ ] 阅读 FlashInfer（2025）的实现，对比 JIT 编译策略与 AttentionEngine 的调度策略的异同
- [ ] 思考：AttentionEngine 的调度框架能否用于加速 KV Cache 稀疏化后的不规则注意力访问？


