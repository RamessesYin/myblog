---
title: 文档标题
date: {{date}}
tags:
  - 标签1
  - 标签2
aliases:
  - 文档别名
draft: false
---

# 你的文档标题

> [!abstract] 核心摘要
> 一句话总结这篇笔记的核心内容、主要观点或研究结论。方便在 Quartz 列表中预览或快速回顾。
> [[paper]]
## 🕸️ 知识图谱索引 (Graph Index)

- **关键词 (Keywords)**: #分布式系统 #大模型 #CUDA #paper/arxiv-1910-02054 #paper/Megatron-LM
- **前置知识 (Prerequisites)**: [[前置概念文档链接]]
- **相关节点 (Related Notes)**: [[相关项目文档1]], [[相关概念文档2]]
- **向下延申 (Successors)**: [[后续研究或拓展文档]]

---

## 1. 文本与基础排版 (Typography)

这里是正文内容的开始。你可以使用 **粗体** 来强调重点，或者使用 *斜体* 表示专有名词。如果你使用 Obsidian，还可以使用 ==高亮文本== 来标记核心结论。对于不需要的内容，可以使用 ~~删除线~~。

包含行内代码：你可以使用 `std::vector<int>` 或 `__global__` 这样的行内代码来标记函数或变量。

## 2. 列表与任务管理 (Lists & Tasks)

### 2.1 无序列表
* 核心特性 A
* 核心特性 B
  * 子特性 B1
  * 子特性 B2

### 2.2 有序列表
1. 收集数据集并预处理
2. 初始化模型权重
3. 启动分布式训练

### 2.3 任务列表 (TODOs)
- [x] 完成架构设计文档
- [ ] 优化算子显存占用
- [ ] 撰写性能分析报告

## 3. 引用与 Callout 标注 (Blockquotes & Callouts)

> 标准的 Markdown 引用格式。通常用于引用外部文章的一段话或他人的发言。

Obsidian 和 Quartz 原生支持非常丰富的 Callout 格式，适合做不同层级的视觉提醒：

> [!info] 补充信息
> 这是一个 info 类型的 callout，适合写一些背景补充或额外说明。

> [!tip] 技巧与提示
> 这是一个 tip 类型的 callout，适合记录某个优化小技巧或最佳实践。

> [!warning] 警告与注意
> 这是一个 warning 类型的 callout，用于记录容易踩坑的地方（例如：注意避免 SM 资源竞争）。

> [!bug] 错误与问题
> 记录遇到的 Bug 现象、报错日志以及最终的修复方案。

## 4. 代码与数学公式 (Code & Math)

### 4.1 多行代码块 (带语法高亮)

```cpp
// 这是一个 CUDA Kernel 示例
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
```

### 4.2 数学公式 (LaTeX)

使用行内公式标记复杂度，例如 $O(N^2)$ 或注意力机制中的 $\sqrt{d_k}$。

如果需要展示推导过程，可以使用独立公式块：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

## 5. 表格与可视化分析 (Tables)

Markdown 原生表格，非常适合做多方案的横向对比：

| 框架/模型 | 并行策略 | 显存优化 | 适用场景 |
| :--- | :---: | :---: | :--- |
| **vLLM** | TP, PP | PagedAttention | 高吞吐推理服务端 |
| **SGLang** | TP | RadixAttention | 复杂 Prompt 调度 |
| **自定义通信** | 自定义 | 预注册内存 | 特定网络拓扑极致优化 |

## 6. 图片、图标与媒体链接 (Media & Links)

### 6.1 外部链接与内部双链
* 外部参考链接：[GitHub 仓库地址](https://github.com/)
* 知识库内部双链：点击跳转到 [[template]]

### 6.2 图片引用
* 外部图片链接：
![架构图](https://dummyimage.com/600x200/cccccc/000000&text=Architecture+Diagram)

* Obsidian 本地图片引用（支持调整宽度）：
![[Pasted image 20260317.png|500]]

## 7. 参考文献 (References)

在正文中，你可以使用 Markdown 脚注语法进行引用标记[^1]。如果有多个来源，可以继续添加[^2]。

---
**参考文献与扩展阅读**：

[^1]: [作者姓名]. "[论文/文章标题]". (年份). *发布期刊或会议*. [URL或DOI链接]
[^2]: [开源项目团队]. "[项目官方文档]". *GitHub/官方网站*. [URL]
[^3]: [arXiv论文] [URL]