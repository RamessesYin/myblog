# 任务一规范：论文/博客精读笔记生成

## 项目基本信息

- **项目路径**: `/mnt/yinhao/note/myblog/`
- **框架**: Quartz v4（静态数字花园）
- **部署**: GitHub Pages（`https://ramessesyin.github.io/myblog/`）

---

## 硬约束与 Red Lines

> **必须先读取** `/mnt/yinhao/note/myblog/.claude/tasks/common_constraints.md`，其中包含文件写入方式、进度汇报、禁止编造等所有公共约束，本文件不再重复。

---

## 目录结构

```
/mnt/yinhao/note/myblog/
├── content/
│   ├── 专业学习/          ← 正式阅读笔记输出目录
│   └── 参考文献/
│       ├── paper/         ← 学术论文参考文献
│       └── blog/          ← 博客参考文献
├── template/
│   └── template.md        ← 笔记模板
└── .claude/logs/
    └── task_log.md        ← 日志
```

---

## 执行流程

### 第一步：解析原文
- 使用 `WebFetch` 抓取目标论文/博客
- 提取：标题、作者、机构、年份、arXiv/DOI 链接、核心贡献、方法设计、实验结果

### 第二步：大范围检索相关工作（目标 20-30 篇，硬下限 10 篇）
- 来源一：`WebFetch` 抓取论文原文的参考文献列表
- 来源二：多轮 `WebSearch`，策略：
  1. `"<Title>" arxiv`
  2. `site:arxiv.org <keyword1> <keyword2> LLM`
  3. `<topic> survey site:arxiv.org`
  4. `<topic> 2024 2025 site:arxiv.org`
  5. `<topic> benchmark SOSP OSDI ASPLOS 2024 2025`
- **禁止编造**：所有引用论文必须来自实际检索结果

### 第三步：生成正式阅读笔记
- **输出路径**: `content/专业学习/<文章名称>.md`
- **格式**：七节结构（见下方模板）
- **链接**: `[[<Name>_paper|📄论文]]`
- 将检索到的相关论文按先驱/对比/Follow-up 分类整合到"相关工作"小节

### 第四步：生成参考文献笔记（仅目标论文，不含检索到的相关论文）
- **学术论文**: `content/参考文献/paper/<Name>_paper.md`
- **博客**: `content/参考文献/blog/<Name>_blog.md`
- 格式含 iframe PDF 嵌入：`<iframe src="https://arxiv.org/pdf/<ID>" width="100%" height="800px"></iframe>`

### 第五步：追加日志
```
## [YYYY-MM-DD HH:MM] <论文标题简称>
- **原文**: <URL>
- **正式笔记**: content/专业学习/<文件名>.md
- **参考文献**: content/参考文献/paper/<文件名>.md
- **关键词**: #tag1 #tag2
- **摘要**: 一句话说明论文核心贡献
```

---

## 笔记模板（七节结构）

```markdown
---
title: "笔记标题"
tags:
  - 标签1
  - 标签2
---

## 🧠 核心摘要 (TL;DR)
> 一两句话概括：解决什么痛点？提出什么核心机制？效果如何？

## 🏷️ 元数据
- **论文标题**: [Full Title]
- **发表机构/会议**: [e.g., SOSP / OSDI / arXiv]
- **年份**: 202X
- **链接**: [[<Name>_paper|📄论文]] | 💻 Code: 暂未开源

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #tag1 #tag2
- **前置知识 (Prerequisites)**: [[前置概念]]
- **相关节点 (Related Notes)**: [[相关笔记]]
- **向下延申 (Successors)**: [[后续研究]]

---

## 🎯 动机与痛点
- **现有方案的瓶颈**: ...
- **论文的切入点**: ...
- **相关工作**（先驱/对比/Follow-up 分类）: ...

## ⚙️ 核心设计与机制
- **架构创新**: ...
- **关键技术点 1**: ...
- **关键技术点 2**: ...
- **核心公式**: $LaTeX$

## 📊 实验与性能评价
- **对比基线 (Baseline)**: ...
- **主要指标**: 吞吐量 / 延迟 / 显存
- **局限性 (Limitations)**: ...

## 💡 启发与我的想法
- **可借鉴之处**: ...
- **改进空间**: ...
- **下一步 Action**: [ ] ...
```

---

## 参考文献笔记模板

```markdown
---
title: "<短标题>"
tags:
  - <机构/会议>
  - <技术领域>
---

# <论文完整标题>

## 🏷️ 元数据
- **作者**: <作者列表>
- **年份**: <年份>

<iframe src="https://arxiv.org/pdf/<ID>" width="100%" height="800px"></iframe>
```

---

## 🚫 Red Lines

> 见 `common_constraints.md`，本文件不重复。

---

## 图片路径规范

> 见 `common_constraints.md`，本文件不重复。
