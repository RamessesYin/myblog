---
title: "Msched"
tags:
  - kernel launch
  - kernel 调度
  - 陈海波
  - IPADS
  - mlsys
---

## 🧠 核心摘要 (TL;DR)

通过解析launch kernel时的参数信息，建立整个任务的完整时间线，记录每个kernel的内存访问需求，从而在后续的内存调度中获得“天眼”，能够及时调度最需要的数据给到当前任务，减少page fault。

经典的 Belady 算法认为，最优的页面置换策略是驱逐那些在未来最长时间内不会被访问的页面。在通用 OS 中，这是不可实现的，因为无法预知未来。但是在Msched的场景下，这一切变成了可能。黑盒式的资源管理策略（如按需分页）已经失效。通过引入领域知悉（Domain-Aware） 的机制——即利用 GPU 程序的确定性特征——操作系统可以实现数量级的性能飞跃。

但这一做法非常依赖内核访存的可预测性。对于那些访存地址依赖于运行时数据内容（Data-dependent memory access）的不规则负载（如某些图计算算法），模板预测可能会失效。未来的工作需要探索如何结合动态分析或推测执行（Speculative Execution）来覆盖这些长尾场景。

## 🏷️ 元数据 (Metadata)
- **论文标题**: [Towards Fully-fledged GPU Multitasking via Proactive Memory Scheduling]
- **发表机构/会议**: [e.g., SOSP / OSDI / arXiv / 某大厂团队]
- **年份**: 2025
- **原文链接**: [[msched_paper|📄论文]]

---

## 🕸️ 知识图谱索引 (Graph Index)
- **关键词 (Keywords)**: #分布式系统 #大模型 #CUDA #paper/arxiv-1910-02054 #paper/Megatron-LM
- **前置知识 (Prerequisites)**: [[前置概念文档链接]]
- **相关节点 (Related Notes)**: [[相关项目文档1]], [[相关概念文档2]]
- **向下延申 (Successors)**: [[后续研究或拓展文档]]

---


## 🎯 动机与痛点 (Motivation)
- **现有方案的瓶颈**: 现有的通信原语、内存管理方式或并行策略（如 TP/PP/EP）存在什么缺陷？（例如：跨节点通信开销大、显存碎片化严重等）
- **论文的切入点**: 作者为什么觉得这个问题值得解决？

## ⚙️ 核心设计与机制 (Methodology)
*(这里可以记录系统的架构图或核心算法推导)*
- **架构创新**: 它是如何重新设计数据流或计算逻辑的？
- **关键技术点 1**: (例如：一种新的内存池化技术或自定义 Tensor 结构)
- **关键技术点 2**: (例如：如何隐式隐藏通信延迟，避免 SM 竞争)
- **核心公式**: 
  $这里可以使用 LaTeX 记录关键的推导过程，例如目标函数或复杂度计算$

## 📊 实验与性能评价 (Evaluation)
- **对比基线 (Baseline)**: 它对比了哪些主流框架或库（如 vLLM, 传统 NCCL, SGLang 等）？
- **主要指标**:
  - 吞吐量 (Throughput) / 延迟 (Latency) 的提升比例。
  - 显存占用 (Memory Footprint) 的降低幅度。
- **局限性 (Limitations)**: 在什么特定的拓扑结构或 workload 下表现不佳？

## 💡 启发与我的想法 (My Thoughts & Ideas)
- **可借鉴之处**: 论文中的设计理念，是否可以应用到我目前正在编写的底层组件或自定义通信库中？
- **改进空间**: 这个方案还有什么明显的漏洞或可以继续优化的地方？
- **下一步 Action**:
  - [ ] 拉取他们的开源代码跑一下 Profiling (如使用 nsys/ncu)。
  - [ ] 验证该思路在特定模型结构下的实际表现。