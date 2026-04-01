# 📚 项目 Memory — 论文解读笔记生成助手

> 每次启动新对话时，请优先读取本文件，以还原完整的工作上下文。

---

## 🗂️ 项目概述

- **项目路径**: `/mnt/yinhao/note/myblog/`
- **框架**: Quartz v4（基于 TypeScript / Preact 的静态数字花园）
- **用途**: 个人专业学习笔记站，聚焦 LLM 推理优化、注意力机制、KV Cache、分布式训练等方向
- **部署**: GitHub Pages

---

## 📁 关键目录结构

```
/mnt/yinhao/note/myblog/
├── content/                         # 所有笔记内容
│   ├── index.md                     # 首页
│   ├── 专业学习/                    # ⭐ 正式阅读笔记（主要输出目录）
│   │   ├── AttentionRes.md
│   │   ├── KV稀疏调研.md
│   │   ├── KvZap.md
│   │   ├── MemCopy.md
│   │   ├── MSched.md
│   │   ├── ShiftParallelism.md
│   │   ├── ZipServ.md
│   │   ├── RL中的KV稀疏讨论.md
│   │   └── LLM数据特征.md
│   └── 参考文献/                    # ⭐ 论文元数据 + PDF 嵌入（索引目录）
│       ├── paper/                   # 学术论文
│       │   ├── AttnRes_paper.md
│       │   ├── MemCopy_paper.md
│       │   ├── KvZap_paper.md
│       │   ├── Msched_paper.md
│       │   ├── ShiftParallelism_paper.md
│       │   └── ZipServ_paper.md
│       └── blog/                    # 博客文章
│           └── TraPO.md
├── template/
│   └── template.md                  # ⭐ 阅读笔记模板
├── .claude/
│   ├── memory.md                    # 本文件（项目 Memory）
│   └── logs/
│       └── task_log.md              # 任务完成日志
└── quartz.config.ts                 # 站点配置
```

---

## 📝 核心任务说明

用户每次发来一篇**论文或博客的链接**，我需要完成以下全套流程：

### 第一步：解析原文
- 使用 `WebFetch` 抓取并解读目标论文/博客
- 提取：标题、作者、机构、年份、arXiv/DOI 链接、核心贡献、方法设计、实验结果

### 第二步：网络检索相关工作
- 根据论文**关键词**和**参考文献**，使用 `WebSearch` 在 Google Scholar 和 arXiv 上检索相关论文
- 重点检索：相似方法的对比工作、该问题的先驱工作、近期 Follow-up 工作
- 搜索策略示例：`site:arxiv.org <keyword>` 或在 Google Scholar 搜索标题/关键词

### 第三步：生成正式阅读笔记
- **输出路径**: `content/专业学习/<文章名称>.md`
- **格式**: 严格参照 `template/template.md` 的七节结构（见下方）
- **链接规范**: 在元数据行使用 `[[<Name>_paper|📄论文]]` 链接到参考文献文件

### 第四步：生成参考文献笔记
- **输出路径**: `content/参考文献/paper/<Name>_paper.md`（学术论文）或 `content/参考文献/blog/<Name>_blog.md`（博客）
- **格式**: 参照 `AttnRes_paper.md` 的格式（见下方）
- **作用**: 作为 Wiki 节点，允许正式笔记通过 `[[<Name>_paper|📄论文]]` 引用

### 第五步：追加日志
- **日志路径**: `.claude/logs/task_log.md`
- 每次任务完成后，追加一条记录（格式见下方）

---

## 📄 模板格式（template/template.md）

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
- **发表机构/会议**: [e.g., SOSP / OSDI / arXiv / 某大厂团队]
- **年份**: 202X
- **链接**: [[<Name>_paper|📄论文]] | [💻 Code](link)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #tag1 #tag2
- **前置知识 (Prerequisites)**: [[前置概念文档链接]]
- **相关节点 (Related Notes)**: [[相关项目文档1]]
- **向下延申 (Successors)**: [[后续研究或拓展文档]]

---

## 🎯 动机与痛点
- **现有方案的瓶颈**: ...
- **论文的切入点**: ...

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

## 📎 参考文献笔记格式（参照 AttnRes_paper.md）

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
- **关联论文**

<iframe src="<PDF直链或arXiv PDF链接>" width="100%" height="800px"></iframe>
```

- arXiv 论文的 PDF 链接格式：`https://arxiv.org/pdf/<ID>`
- 其他 PDF 用直链

---

## 🔗 现有笔记索引（已完成）

| 文件名 | 论文/博客 | 关键词 |
|--------|-----------|--------|
| AttentionRes.md | Attention Residuals (Kimi, 2026) | 残差连接, Softmax聚合, 训练效率 |
| KV稀疏调研.md | KV Cache 稀疏化综合调研 | KV Cache, 稀疏注意力, LLM推理 |
| KvZap.md | KVzap (Nvidia, 2026) | KV Cache剪枝, 自适应稀疏 |
| MemCopy.md | How to Copy Memory (SOSP'25) | 异步内存拷贝, OS服务 |
| MSched.md | MSched | - |
| ShiftParallelism.md | Shift Parallelism | 分布式训练并行 |
| ZipServ.md | ZipServ | LLM服务优化 |
| RL中的KV稀疏讨论.md | RL与KV稀疏 (讨论稿) | RL, KV Cache |

---

## 🛠️ 工具使用规范

| 工具 | 用途 |
|------|------|
| `WebFetch` | 抓取论文/博客页面内容（arXiv abstract 页、PDF 直链等） |
| `WebSearch` | 在 Google Scholar / arXiv 检索相关论文，关键词需精准 |
| `Write` | 创建新的笔记文件 |
| `Edit` | 追加日志内容到已有文件 |
| `Read` | 读取现有文件（编辑前必须先读） |
| `Bash` | 执行文件操作、目录检查等 |

### WebSearch 搜索策略
1. 用论文标题搜索：`"<Title>" arxiv`
2. 用关键词搜索：`site:arxiv.org <keyword1> <keyword2> LLM`
3. 用 Google Scholar 格式：`<keyword> survey site:scholar.google.com`
4. 找相关基线论文：`<topic> benchmark SOSP OSDI ASPLOS 2024 2025`

---

## 📋 日志格式（.claude/logs/task_log.md）

每次完成任务后，在日志文件末尾追加：
```
## [YYYY-MM-DD HH:MM] <论文标题简称>
- **原文**: <URL>
- **正式笔记**: content/专业学习/<文件名>.md
- **参考文献**: content/参考文献/paper/<文件名>.md
- **关键词**: #tag1 #tag2
- **摘要**: 一句话说明论文核心贡献
```

---

## ⚠️ 注意事项

1. **Wiki 链接命名**: `<Name>_paper.md` 中的 `<Name>` 要与正式笔记中的 `[[<Name>_paper|📄论文]]` 完全一致
2. **标签规范**: tags 中用中文或英文均可，但 `#关键词` 在知识图谱中统一小写英文
3. **公式**: 行内用 `$...$`，块级用 `$$...$$`，兼容 KaTeX
4. **相关工作深度**: 至少检索并对比 3 篇相关论文，整合到"动机与痛点"或独立的"相关工作"小节中
5. **文件编码**: 统一 UTF-8，文件名中文路径已被 Quartz 正确处理
6. **不创建 README.md**，除非用户明确要求

---

## 🚫 严格禁止的行为（Red Lines）

> 以下规则由用户明确要求，违反将导致笔记内容不可信，**必须严格遵守**：

### 1. 禁止编造参考文献
- **不得**凭推断或"看起来合理"的方式捏造论文标题、作者、arXiv ID、会议名称
- 相关工作部分引用的每一篇论文，**必须**来自以下来源之一：
  - 原论文的参考文献列表（通过 `WebFetch` 从论文 HTML/PDF 中读取到的）
  - `WebSearch` 实际检索并返回的结果页面中明确存在的论文
- 若无法通过上述方式找到足够的相关论文，**宁可少写，也不编造**

### 2. 禁止编造代码/项目地址
- 论文元数据中的 `[💻 Code](URL)` 链接，**必须**来自论文原文中作者明确提供的地址
- **不得**根据作者名、组织名、项目名自行拼接 GitHub URL（即使看起来非常合理）
- 若论文未提供代码链接，则在笔记中写 `💻 Code: 暂未开源` 或直接省略该字段
- **反面案例**（2026-03-31 教训）：曾将作者个人仓库 `github.com/Hanchenli/vllm-continuum` 错误编造为组织仓库 `github.com/vllm-project/vllm-continuum`，该地址返回 404

### 3. 数据与结论须有原文依据
- 实验数字（如"提升 X 倍"、"降低 Y%"）必须能在原文中找到对应表格或句子
- 若某项数据是从上下文推断而非原文直接陈述，须在笔记中加注"（推断）"或"（约）"

---

*最后更新: 2026-03-31*
