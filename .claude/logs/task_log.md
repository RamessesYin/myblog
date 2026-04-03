# 📋 任务完成日志

> 每次完成论文/博客解读任务后，自动追加一条记录到此文件。

---

## [2026-03-31 初始化]
- **操作**: 初始化项目 Memory 文件与日志系统
- **创建文件**: `.claude/memory.md`、`.claude/logs/task_log.md`
- **说明**: 完成项目结构探索，记录模板格式、目录规范、工具使用策略，为后续论文解读任务建立标准化工作流

---

## [2026-04-01 完成]  OpenClaw-RL — 从实时 Agent 交互信号在线强化学习
- **原文**: https://arxiv.org/abs/2603.10165
- **正式笔记**: content/专业学习/OpenClaw-RL.md
- **参考文献**: content/参考文献/paper/OpenClaw-RL_paper.md
- **关键词**: #强化学习 #LLM-Agent #优势估计 #在线学习 #PRM #Token级优势
- **摘要**: OpenClaw-RL 从 Agent 交互的"下一状态信号"中恢复两类被浪费的学习信号：PRM 多数投票构造标量 Advantage 的 Binary RL，以及通过 Hindsight hint 构造 token 级方向性优势的 OPD，二者组合在 36 次交互内实现显著个性化改善，无需 Critic 网络或 group 采样。
- **对比检索**: GRPO/DeepSeek-R1 (arXiv:2501.12948), DAPO (arXiv:2503.14476), PPO (Schulman et al. arXiv:1707.06347), HER, STaR, RLAnything

---

## [2026-04-01 完成]  CloudMatrix384 — 华为 Ascend 910 MoE 大规模推理服务
- **原文**: https://arxiv.org/abs/2506.12708
- **正式笔记**: content/专业学习/CloudMatrix384.md
- **参考文献**: content/参考文献/paper/CloudMatrix384_paper.md
- **关键词**: #MoE #专家并行 #LLM推理服务 #Ascend-NPU #PDC分离 #MLA优化 #INT8量化 #KV-Cache
- **摘要**: 华为 CloudMatrix-Infer 在 384 NPU 全对等 UB 互联超节点上，通过 EP320 大规模专家并行、AIV-Direct 通信、早期 INT8 量化、MLA 算子融合等优化，在 DeepSeek-R1 671B 上实现预填充效率 4.45 tok/s/TFLOPS、解码效率 1.29 tok/s/TFLOPS，全面超越 H800 和 H100 基线。
- **对比检索**: DeepSeek H800 (DeepSeek open-infra-index), SGLang on H100 (lmsys.org/blog), DistServe (OSDI'24), Mooncake (FAST'25, arXiv:2407.00079), MegaScale-Infer (arXiv:2504.02263)

---

## [2026-04-03 14:00] 大模型量化 综述
- **专题文件**: todo/量化.md
- **正式笔记**: content/专业学习/大模型量化综述.md
- **参考文献**: N/A（综述任务）
- **关键词**: #量化 #quant #fp4 #fp8 #混合精度量化 #qat #计算加速 #显存降低 #通信量降低
- **摘要**: 系统覆盖 LLM 量化的 10 个核心问题：量化原理与离群值难点、量化粒度（Per-Tensor/Per-Token/Per-Channel 及 Tensor Core 硬件限制）、权重/激活/KV Cache 量化差异、NVIDIA NVFP4 (E2M1 + FP8 block scale) 规格与 Blackwell 硬件支持、vLLM 量化算子融合实现（Marlin/DeepGEMM/Triton）、QAT 方法（LLM-QAT/EfficientQAT/BitNet b1.58/QLoRA）、精度评估体系、离群值抑制四大方案（LLM.int8/SmoothQuant/QuaRot/SpinQuant）、混合精度与精度回退策略，以及伪量化（Fake Quantization）的原理与实现。
- **来源文件**: todo/量化.md（任务三批处理）

---

## [2026-03-31 完成]  Continuum — KV Cache TTL 多轮 Agent 调度
- **原文**: https://arxiv.org/pdf/2511.02230
- **正式笔记**: content/专业学习/Continuum.md
- **参考文献**: content/参考文献/paper/Continuum_paper.md
- **关键词**: #KV-Cache #TTL #多轮Agent #LLM推理调度 #GPU显存管理 #工具调用
- **摘要**: Continuum 为多轮 LLM Agent 推理系统引入基于成本收益建模的自适应 KV Cache TTL 保留机制，在工具调用等待期间智能保留 GPU 显存中的 KV Cache，避免每轮冗余 Prefill，在 SWE-Bench 和 BFCL 实测中延迟降低最高 8.18x。
- **对比检索**: Autellix (arXiv:2502.13965), SGLang/RadixAttention (arXiv:2312.07104), Mooncake (arXiv:2407.00079), vLLM (Kwon et al. SOSP'23), DeepSpeed-FastGen (arXiv:2401.08671)

---
