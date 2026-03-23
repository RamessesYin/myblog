---
title: "Shift Parallelism"
tags:
  - LLM
  - 推理加速
  - 弹性并行
  - TP-Scale
---

## 🧠 核心摘要 (TL;DR)

> 把训练里的SP用到了prefill里，这个方案的收益其实是在FFN阶段，在FFN阶段序列并行从而免去了这部分原本TP通信。但是这样就要求模型FFN能在单卡放得下，否则还得继续切TP，对大尺寸MoE没啥帮助，或者说MoE里有EP。如果直接DP的话，decode的时候就没法复用kv cache了，所以才在attention阶段整了个SP，本质上是为了后续decode时KV Cache布局不变。

## 🏷️ 元数据 
- **论文标题**: Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
- **发表机构/会议**: [arXiv]
- **年份**: 2026
- **链接**: [[ShiftParallelism_paper|📄论文]]

---

## 🕸️ 知识图谱索引 
- **关键词 (Keywords)**: #Tensor-Parallelism #Data-Parallelism #Sequence-Parallelism #KV-Cache 
<!-- - **前置知识 (Prerequisites)**: [[前置概念文档链接]]
- **相关节点 (Related Notes)**: [[相关项目文档1]], [[相关概念文档2]]
- **向下延申 (Successors)**: [[后续研究或拓展文档]] -->

---


## 🎯 动机与痛点
推理系统（Inference Systems）的效率成为了决定商业可行性的核心变量。长期以来，系统架构师在设计大语言模型（LLM）服务设施时，被迫在**低延迟（Low Latency）**与**高吞吐量（High Throughput）**之间进行零和博弈。传统的并行计算策略——张量并行（Tensor Parallelism, TP）和数据并行（Data Parallelism, DP）——分别占据了性能光谱的两个极端，却无法兼顾。TP 通过聚合多 GPU 算力极大地降低了单次响应的延迟，但其频繁的通信开销限制了系统在高并发下的吞吐上限；DP 虽然实现了计算资源的线性扩展和极高的吞吐量，却因缺乏单请求内的算力聚合而导致首字延迟（TTFT）过高，无法满足实时交互需求。

与训练阶段高度规整、同质化的工作负载不同，推理阶段的流量特征呈现出极端的异构性和不可预测性。企业级 AI 应用通常面临两类截然不同的流量模式，这对底层基础设施提出了截然相反的要求：

|流量类型|典型场景|性能指标 (SLO)|资源瓶颈特征|
|---|---|---|---|
|**在线交互式 (Online/Interactive)**|智能客服、Copilot 代码补全、实时翻译|**TTFT (Time To First Token)**：极低 <br>**TPOT (Time Per Output Token)**：低，且抖动小|**Latency Bound**：受限于内存带宽和通信延迟，GPU 计算单元利用率通常较低。|
|**离线批处理 (Offline/Batch)**|文档摘要、大规模数据清洗、RAG 索引构建|**Throughput (Tokens/sec)**：极大化 <br>**Cost Per Token**：极小化|**Throughput Bound**：受限于 GPU 计算能力 (TFLOPS) 和显存容量 (KV Cache)。|

在现实生产环境中，这两种流量并非静态隔离，而是随时间动态波动。例如，白天工作时间主要是高频的交互式请求，而夜间则可能突发大规模的离线批处理任务。传统的“静态并行”策略要求运维团队为不同负载部署独立的集群（如 TP 集群服务在线，DP 集群服务离线），这不仅导致了硬件资源的碎片化和闲置，也大幅增加了总拥有成本（TCO）和运维复杂度 。

本论文的核心突破在于利用了 **KV Cache 不变性（KV Cache Invariance）**——即张量并行与基于 Ulysses 的序列并行（Sequence Parallelism, SP）在注意力层共享完全相同的内存布局。这一数学特性使得系统能够在运行时根据流量负载的实时变化，无缝、零成本地在“延迟优先（TP）”和“吞吐优先（SP）”模式之间动态切换，从而在单一部署中同时逼近两种策略的理论最优值。

结合 **SwiftKV**（预填充计算优化）和 **Speculative Decoding**（推测解码），基于 Shift Parallelism 构建的 **Arctic Inference** 系统实现了推理性能的全面飞跃：相比最佳吞吐优化方案，请求完成速度提升 3.4 倍；相比最佳延迟优化方案，吞吐量提升 1.75 倍。

## ⚙️ 核心设计与机制 

###  现有并行策略的局限性剖析

为了应对大模型（如 Llama-3-70B/405B）对显存和算力的巨大需求，分布式推理是必选项。然而，现有的并行策略在应对动态负载时均显现出明显的短板。

####  张量并行 (Tensor Parallelism, TP)：延迟低，吞吐高

张量并行（主要基于 Megatron-LM 的实现）是目前降低大模型推理延迟的事实标准。其核心思想是将模型每一层的权重矩阵（Weight Matrices）切分到多个 GPU 上，通过并行计算来加速单次矩阵乘法。

-   **工作机制**：在 Transformer 的每一层，前向传播都需要在所有参与的 GPU 之间进行一次 `All-Reduce` 通信，以同步部分和（Partial Sums）。
    
-   **通信瓶颈**：对于一个 $L$ 层的模型，处理每一个 Token 都需要 $L$ 次同步通信。这种高频的通信虽然在 NVLink 等高带宽互联下尚可接受，但随着 Batch Size 的增加，通信开销占比并不会显著下降，反而TP开大以后 GPU 算力无法被充分利用
    
-   **吞吐量天花板**：当系统试图通过增大 Batch Size 来提高吞吐时，TP 模式下的 KV Cache 被切分到各卡，虽然显存利用均衡，但通信量随 Token 数量线性增长。这导致 TP 在高并发下的吞吐量远低于理论峰值。
    

####  数据并行 (Data Parallelism, DP)：吞吐高，延迟差

数据并行是最简单的并行形式，通过在不同 GPU 上复制完整的模型副本来并行处理不同的请求。

-   **优势**：除了极少量的梯度同步（仅在训练时），DP 在推理时几乎是零通信（Zero Communication）。这意味着 GPU 可以将全部时间用于计算，从而在处理大 Batch 时达到极高的吞吐量。
    
-   **劣势**：DP 无法拆分单个请求的计算。对于 70B 或更大参数的模型，单张 GPU 即使能装下模型权重（通常需要量化），其串行计算的延迟也过高，导致 TTFT 无法满足人类交互的 200ms 黄金法则。此外，模型权重的多重复制浪费了宝贵的 HBM 显存，挤占了 KV Cache 的空间，反而限制了最大支持的 Batch Size。
    

####  序列并行 (Sequence Parallelism, SP)：

序列并行旨在解决长序列（Long Context）带来的显存压力。早期的 Ring Self-Attention 虽然解决了显存墙问题，但其通信模式复杂，难以在推理中获得低延迟。然而，随着 **DeepSpeed Ulysses** 等新一代 SP 技术的出现，通过 `All-to-All` 通信进行注意力头（Head）切分成为可能。在 Shift Parallelism 提出之前，SP 主要被视为训练技术，而非推理加速手段 。  

### Shift Parallelism 
Shift Parallelism并非发明了一种全新的并行方式，而是发现并利用了 TP 和 Ulysses SP 之间某种深刻的**同构性**，从而实现了对两种策略优势的动态融合。

#### 核心洞察：KV Cache 不变性 (The Invariance Principle)

要理解 Shift Parallelism，必须深入到 Transformer 注意力机制的张量布局（Tensor Layout）层面。

在分布式推理中，KV Cache（Key-Value Cache）是显存占用的最大来源，也是状态管理的核心。传统的观点认为，从 TP 切换到 DP/SP 需要对 KV Cache 进行极其昂贵的重排（Resharding），这涉及到跨 GPU 搬运数 GB 的数据，导致切换延迟高达数秒甚至数分钟，使得动态切换在毫秒级的推理服务中不可行。

然而，Snowflake AI Research 的研究人员发现，**张量并行（TP）与 Ulysses 序列并行（SP）在注意力层（Attention Layer）拥有完全一致的 KV Cache 内存布局**。
 

#### 动态调度策略 (The Shift Schedule)

基于 KV Cache 不变性，Shift Parallelism 引入了一个轻量级的运行时调度器，该调度器根据当前的**批次大小（Batch Size）**实时决定采用何种并行策略。

#####  策略阈值机制

调度算法的核心逻辑如下：
$\text{Config} = \begin{cases} 
(\text{TP}=P, \text{SP}=1) & \text{if } \text{Batch Size} < \tau \\
(\text{TP}=1, \text{SP}=P) & \text{if } \text{Batch Size} \ge \tau 
\end{cases}$

其中 $\tau$  是根据硬件特性（计算能力 vs 通信带宽）和模型参数确定的最优切换阈值。

-   **低负载阶段（交互式）**：当系统处于低流量或处理单个用户的实时请求时，Batch Size 较小。此时系统采用 **Full TP** 模式。
    
    -   **优势**：利用所有 GPU 的显存带宽并行读取权重，利用所有 Tensor Core 并行计算，最小化单步延迟（TPOT）。
        
    -   **通信**：高频的 All-Reduce，但在小 Batch 下数据量小，延迟可控。
        
-   **高负载阶段（批处理/高并发）**：当大量请求涌入，Batch Size 超过阈值 时，系统瞬间切换至 **Full SP** 模式。
    
    -   **优势**：SP 模式下，除了 Attention 部分的 All-to-All，前馈网络（FFN）部分是完全数据并行的（在序列维度独立计算）。这意味着占模型计算量约 2/3 的 FFN 层实现了零通信。
        
    -   **通信**：虽然 Attention 需要 All-to-All，但相比 TP 的每层 All-Reduce，SP 的通信频率更低，且在大 Batch 下更容易被计算掩盖（Overlap）。更重要的是，SP 允许更大的 Batch Size，从而极大提升了 GPU 的计算利用率（SM Utilization）。  
        

#####  混合配置支持
- $(SP=1, TP=8)$：极低延迟。
- $(SP=2, TP=4)$：平衡模式。
- $(SP=4, TP=2)$：高吞吐倾向。
- $(SP=8, TP=1)$：极高吞吐 。
这种灵活性使得 Shift Parallelism 能够适应不同的硬件拓扑（如节点内 NVLink vs 节点间 Ethernet）和模型架构。

## 📊 实验与性能评价 

Snowflake AI Research 在 AWS p5en.48xlarge 实例（搭载 8x H200 GPU）上对 Shift Parallelism 进行了评估

###  延迟与吞吐的帕累托改进

最令人瞩目的结果是 Shift Parallelism 在 Latency-Throughput 权衡曲线上的表现。

-   **实验设置**：对比基准包括“仅 TP”（低延迟配置）、“仅 DP”（高吞吐配置）以及 Shift Parallelism。模型为 Llama-3-8B 和 Llama-3-70B。
    
-   **结果分析**：
    
    -   **低负载区**：Shift Parallelism 曲线与 TP 曲线完全重合，保持了最低的延迟。这证明了动态调度在小 Batch 下正确选择了 TP 模式。
        
    -   **高负载区**：随着并发请求数增加，TP 曲线迅速饱和，吞吐量不再增长；而 Shift Parallelism 曲线平滑过渡，吞吐量持续攀升，最终与 DP 曲线趋同。
        
    -   **量化数据**：相比于纯 TP 方案，Shift Parallelism 在高负载下的**吞吐量提升了 50%**；相比于纯 DP 方案，在交互式负载下的**响应速度（TTFT）提升了 1.51 倍** 。  
        
    -   **结论**：Shift Parallelism 实际上扩展了推理性能的帕累托前沿（Pareto Frontier），在单一系统中实现了以往需要两套系统才能覆盖的性能包络。
        

###  真实流量回放 (Trace Replay)

为了验证系统在真实生产环境中的表现，研究团队使用了包含动态到达率（泊松分布）和混合长短请求的 Trace 进行压测。

-   **性能三位一体 (Trifecta)**：单一配置的 Arctic Inference 实现了：
    
    -   **请求完成时间**：比最佳吞吐配置（DP=8）快 **3.4 倍**。
        
    -   **吞吐量**：比最佳延迟配置（TP=8）高 **1.06 倍**。
        
    -   **生成速度**：比最佳延迟配置快 **1.75 倍**（归功于 Speculative Decoding）。  
        
-   **意义**：这一结果表明，企业不再需要根据预测的峰值流量来过度配置（Over-provisioning）硬件资源。Shift Parallelism 使得集群能够像流体一样自适应流量波动，极大提升了资源利用率。
    

###  Embedding 模型的性能飞跃

除了生成任务，Arctic Inference 对 Embedding 模型（如 BERT 架构）也进行了优化。由于 Embedding 任务本质上是纯 Prefill，且 Batch Size 极大，Shift Parallelism 的 SP 模式在此场景下威力倍增。

-   **数据**：单卡 H200 处理 Embedding 的吞吐量达到了惊人的 **160 万 tokens/sec**。
    
    -   **短序列**：比原生 vLLM 快 **16 倍**，比 Text Embeddings Inference (TEI) 快 **2.4 倍**。
        
    -   **长序列**：比原生 vLLM 快 **4.2 倍** 。  
        
-   **技术细节**：这得益于向量化的数据序列化和并行的 Tokenization 优化，消除了 Python 运行时的开销，让 GPU 始终处于满载状态。
    


## 💡 启发与我的想法 
确实prefill和decode有不同的需求，所以现在pd分离以后就不需要搞这么复杂的sp来保证kv cache的不变性，因为本来就是要传输这些kv cache的。这篇论文的重点在于解决不同batch的prefill需求，我设想的RL弹性并行也是类似的思路，从attention DP切回TP，其实本质上就是all gather一次 attention 权重，这样从保吞吐改成保延迟，同时也需要一次all2all来传kv cache。论文中对FFN用DP的方法对于我想优化的RL场景感觉意义不大，因为671B这种模型FFN根本没法塞在一张卡上，EP替代TP其实是类似的思路，但如果没有足够大的batch EP也是一样打不满，可能还不如直接TP。rollout的瓶颈在decode，论文中提到的sp是不能应用的，否则传kv cache是负优化，还是得传q做分布式softmax