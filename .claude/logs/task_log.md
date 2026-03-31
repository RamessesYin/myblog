# 📋 任务完成日志

> 每次完成论文/博客解读任务后，自动追加一条记录到此文件。

---

## [2026-03-31 初始化]
- **操作**: 初始化项目 Memory 文件与日志系统
- **创建文件**: `.claude/memory.md`、`.claude/logs/task_log.md`
- **说明**: 完成项目结构探索，记录模板格式、目录规范、工具使用策略，为后续论文解读任务建立标准化工作流

---

## [2026-03-31 完成]  Continuum — KV Cache TTL 多轮 Agent 调度
- **原文**: https://arxiv.org/pdf/2511.02230
- **正式笔记**: content/专业学习/Continuum.md
- **参考文献**: content/参考文献/paper/Continuum_paper.md
- **关键词**: #KV-Cache #TTL #多轮Agent #LLM推理调度 #GPU显存管理 #工具调用
- **摘要**: Continuum 为多轮 LLM Agent 推理系统引入基于成本收益建模的自适应 KV Cache TTL 保留机制，在工具调用等待期间智能保留 GPU 显存中的 KV Cache，避免每轮冗余 Prefill，在 SWE-Bench 和 BFCL 实测中延迟降低最高 8.18x。
- **对比检索**: Autellix (arXiv:2502.13965), SGLang/RadixAttention (arXiv:2312.07104), Mooncake (arXiv:2407.00079), vLLM (Kwon et al. SOSP'23), DeepSpeed-FastGen (arXiv:2401.08671)

---
