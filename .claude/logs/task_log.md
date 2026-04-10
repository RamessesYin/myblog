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

## [2026-04-03 完成] MetaAttention — 跨硬件与算法的统一高性能注意力框架
- **原文**: https://dl.acm.org/doi/10.1145/3774934.3786444
- **正式笔记**: content/专业学习/MetaAttention.md
- **参考文献**: content/参考文献/paper/MetaAttention_paper.md
- **关键词**: #attention #gpu #kernel-optimization #llm推理 #代码生成 #dsl
- **摘要**: MetaAttention（PPoPP'26，SJTU-IPADS）提出 OnlineFunc + score_mod + mask_mod 三层 DSL 抽象，将多样化注意力算法（MHA、RetNet、Mamba2、MLA 等）与多种 GPU 硬件（NVIDIA Hopper、AMD MI200）的代码生成统一，实现媲美手写库的性能，同时极大降低新型注意力算子的开发门槛。
- **来源文件**: todo/task7.md（任务三批处理）

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

## [2026-04-03 16:30] 虚假奖励悖论（Spurious Rewards Paradox）
- **原文**: https://arxiv.org/abs/2601.11061
- **正式笔记**: content/专业学习/SpuriousReward.md
- **参考文献**: content/参考文献/paper/SpuriousReward_paper.md
- **关键词**: #rlvr #reinforcement-learning #mechanistic-interpretability #data-contamination #memorization #ppo #grpo #spurious-rewards
- **摘要**: 论文通过 Path Patching + Logit Lens + Neural ODE 等机械可解释性工具，定位 Qwen2.5-Math 中的"锚点-适配器电路"，揭示 RLVR 虚假奖励提升本质上是 GRPO 裁剪偏差激活污染数据记忆检索路径，并提出因果操控方法实现双向去污染。
- **来源文件**: todo/Spurious Reward in RL.md（任务三批处理）

---

## [2026-04-03 18:00] Close Look at DeepSeek-OCR — 视觉能力还是语言拐杖
- **原文**: https://arxiv.org/pdf/2601.03714
- **正式笔记**: content/专业学习/CloseLookDeepSeekOCR.md
- **参考文献**: content/参考文献/paper/CloseLookDeepSeekOCR_paper.md
- **关键词**: #ocr #vlm #language-prior #hallucination #visual-compression #multimodal
- **摘要**: 通过五组语义破坏实验（句级/词级扰动 + 随机字符 + 下游推理 + 长文本崩溃），系统揭示 DeepSeek-OCR 的高 OCR 精度主要依赖语言先验而非真正的视觉感知（去除语言上下文后精度从 ~90% 骤降至 ~20%），并发现所有分辨率模式均在约 8000~10500 字符处发生灾难性崩溃，暴露光学上下文压缩范式的根本性局限。
- **来源文件**: todo/Close Look at DeepSeek-OCR.md（任务三批处理）

---

## [2026-04-03 17:30] TTT-Discover：推理时的自我进化
- **原文**: https://arxiv.org/abs/2601.16175
- **正式笔记**: content/专业学习/TTT-Discover.md
- **参考文献**: content/参考文献/paper/TTT-Discover_paper.md
- **关键词**: #test-time-training #reinforcement-learning #scientific-discovery #llm-optimization #entropic-objective
- **摘要**: TTT-Discover 在推理阶段针对单一问题用强化学习（熵增目标 + PUCT 树搜索）直接更新 LoRA 参数，在数学发现、GPU Kernel 工程、算法竞赛、生物信息四个领域突破 SOTA，代价是每题约 $500 的算力成本与不可逆过拟合，适用于「只需一次性极限突破」的场景。
- **来源文件**: todo/Learning to Discover at Test Time.md（任务三批处理）

---

## [2026-04-03 22:00] RL Beyond the Base Model — RLVR 能力边界研究
- **原文**: https://arxiv.org/abs/2504.13837
- **正式笔记**: content/专业学习/RLBeyondBaseModel.md
- **参考文献**: content/参考文献/paper/RLBeyondBaseModel_paper.md
- **关键词**: #rlvr #reinforcement-learning #llm-reasoning #pass-at-k #base-model
- **摘要**: 通过 pass@k（大 k 值）评估框架，系统证明六种主流 RLVR 算法（PPO/GRPO/DAPO/Reinforce++/RLOO/ReMax）仅提升采样效率而未扩展基模能力边界——大 k 值下基模反超 RL 模型，仅 0~1% 的问题由 RL 独占；困惑度分析进一步证实 RL 模型的推理路径早已存在于基模分布中；蒸馏则能真正引入新路径。NeurIPS 2025 Best Paper Runner-Up。
- **来源文件**: todo/RL Beyond the Base Model.md（任务三批处理）

---

## [2026-04-03 23:30] LongCat-Flash-Thinking-2601 — 美团 Agentic Reasoning 实践
- **原文**: https://arxiv.org/abs/2601.16725
- **正式笔记**: content/专业学习/LongCat.md
- **参考文献**: content/参考文献/paper/LongCat_paper.md
- **关键词**: #agentic-reasoning #mixture-of-experts #reinforcement-learning #tool-use #long-context
- **摘要**: 美团 LongCat 团队提出 560B MoE 开源推理模型，通过 DORA 全异步 RL 系统、三路 Agentic 数据合成管线、噪声鲁棒训练、上下文管理模块及 Heavy Thinking 测试时扩展，在 BrowseComp（73.1%）、τ²-Bench（88.2%）、SWE-bench Verified（70.0%）等 Agentic 基准上达到开源 SOTA。
- **来源文件**: todo/美团LongCat-Flash-Thinking.md（任务三批处理）

---

## [2026-04-03 23:45] 任务三批处理汇总
- **处理文件数**: 5 个（全部成功）
- **生成笔记列表**:
  - content/专业学习/SpuriousReward.md（来自 todo/Spurious Reward in RL.md）
  - content/专业学习/TTT-Discover.md（来自 todo/Learning to Discover at Test Time.md）
  - content/专业学习/CloseLookDeepSeekOCR.md（来自 todo/Close Look at DeepSeek-OCR.md）
  - content/专业学习/RLBeyondBaseModel.md（来自 todo/RL Beyond the Base Model.md）
  - content/专业学习/LongCat.md（来自 todo/美团LongCat-Flash-Thinking.md）
- **移入 done 目录**: 全部 5 个文件已移入 todo/done/
- **跳过/失败**: 无

---

## [2026-04-03 23:55] SonicMoE — 细粒度 MoE IO 与 Tile 感知训练加速
- **原文**: https://arxiv.org/abs/2512.14080
- **正式笔记**: content/专业学习/SonicMoE.md
- **参考文献**: content/参考文献/paper/SonicMoE_paper.md
- **关键词**: #moe #gpu-kernel #training #memory-efficiency #cuda #hopper #blackwell
- **摘要**: SonicMoE 针对细粒度 MoE 训练中的算术强度低、激活显存随 granularity 爆炸、Padding 浪费三大痛点，提出 IO 感知激活精选缓存 + Tile 感知 CUDA Kernel（Hopper Ping-Pong + Blackwell TMEM）+ Token Rounding 路由算法，在 64×H100 上实现 2130 亿 token/天，激活显存降低 45%，计算吞吐相比 ScatterMoE/MoMoE 提升最高 115%。
- **来源文件**: todo/task5.md（任务三批处理）

---

## [2026-04-03] ThunderAgent — 程序感知智能体推理系统
- **原文**: https://arxiv.org/abs/2602.13692
- **正式笔记**: content/专业学习/ThunderAgent.md
- **参考文献**: content/参考文献/paper/ThunderAgent_paper.md
- **关键词**: #agentic-inference #kv-cache #llm-serving #tool-use #scheduling #reinforcement-learning
- **摘要**: ThunderAgent 提出 LLM Program 抽象，将 Agent 生命周期视为统一调度单元，通过程序感知 KV Cache 保护（时间衰减打分）、基于钩子的工具资源 GC 以及异步环境预热，在服务吞吐量上比 vLLM 提升 1.48–3.58×，RL Rollout 提升最高 3.92×，节省磁盘内存最多 4.2×。
- **来源文件**: todo/task3.md（任务三批处理）

---

## [2026-04-03] ProRL Agent — Rollout-as-a-Service RL 训练基础设施
- **原文**: https://arxiv.org/abs/2603.18815
- **正式笔记**: content/专业学习/ProRLAgent.md
- **参考文献**: content/参考文献/paper/ProRLAgent_paper.md
- **关键词**: #rl-training #llm-agent #rollout-service #multi-turn #infrastructure #nvidia #nemo-gym
- **摘要**: ProRL Agent 将 RL 训练的 rollout 生命周期封装为独立 HTTP 服务（三阶段异步流水线 INIT→RUN→EVAL），通过 Token-in/Token-out 消除 Re-tokenization 漂移、SingularityRuntime 支持无特权 HPC 容器、可插拔沙箱抽象覆盖多种任务类型，Qwen3-14B 经训练后在 SWE-Bench Verified 上 pass rate 从 15.4% 提升至 23.6%。
- **来源文件**: todo/task9.md（任务三批处理）

---

## [2026-04-03] RLix — 多任务 RL 实验 GPU 调度与资源共享框架
- **原文**: https://github.com/rlops/rlix
- **正式笔记**: content/专业学习/RLix.md
- **参考文献**: content/参考文献/blog/RLix_blog.md
- **关键词**: #reinforcement-learning #gpu-scheduling #multi-task #lora #elastic-scaling
- **摘要**: RLix 是 rlops 组织的开源 GPU 调度系统，通过六级优先级抢占（Rollout 为可抢占阶段）、跨任务 GPU 借用、多 LoRA 适配器共享基础模型等机制，减少多并发 RL 实验的 GPU 等待时间并提升利用率，灵感来源于 Alibaba/ROLL 的 Partial Overlapping 调度思想。
- **来源文件**: todo/task8.md（任务三批处理）

---

## [2026-04-03] Medusa — 无服务器 LLM 推理 CUDA 图物化加速
- **原文**: https://dl.acm.org/doi/10.1145/3669940.3707285
- **正式笔记**: content/专业学习/Medusa_Serverless.md
- **参考文献**: content/参考文献/paper/Medusa_Serverless_paper.md
- **关键词**: #serverless #llm-inference #cuda-graph #cold-start #kv-cache #materialization
- **摘要**: Medusa 将 LLM 无服务器推理的三阶段冷启动（模型加载 + KV Cache 初始化 + CUDA 图构建）中的后两个阶段通过"离线物化 + 语义标签重绑定 + 触发内核"机制预计算并持久化，在线恢复跳过重建，使模型加载延迟降低 42.5%，TTFT 尾延迟降低 53.0%（ASPLOS 2025）。
- **来源文件**: todo/task10.md（任务三批处理）

---

## [2026-04-03] MagiAttention — 面向超长上下文与异构 Mask 的分布式注意力
- **原文**: https://github.com/SandAI-org/MagiAttention
- **正式笔记**: content/专业学习/MagiAttention.md
- **参考文献**: content/参考文献/paper/MagiAttention_paper.md
- **关键词**: #context-parallelism #distributed-attention #long-context #heterogeneous-mask #flash-attention
- **摘要**: MagiAttention 针对超长上下文（4M token 以上）异构 Mask 训练提出四层协同优化（AttnSlice 内核抽象 + Dispatch Solver 负载均衡 + GroupCast/GroupReduce 零冗余通信 + 自适应多阶段重叠调度），在 H100/B200 上实现接近线性扩展，是 MAGI-1（24B 参数视频生成模型）的核心训练基础设施。
- **来源文件**: todo/task6.md（任务三批处理）

---

## [2026-03-31 完成]  Continuum — KV Cache TTL 多轮 Agent 调度
- **原文**: https://arxiv.org/pdf/2511.02230
- **正式笔记**: content/专业学习/Continuum.md
- **参考文献**: content/参考文献/paper/Continuum_paper.md
- **关键词**: #KV-Cache #TTL #多轮Agent #LLM推理调度 #GPU显存管理 #工具调用
- **摘要**: Continuum 为多轮 LLM Agent 推理系统引入基于成本收益建模的自适应 KV Cache TTL 保留机制，在工具调用等待期间智能保留 GPU 显存中的 KV Cache，避免每轮冗余 Prefill，在 SWE-Bench 和 BFCL 实测中延迟降低最高 8.18x。
- **对比检索**: Autellix (arXiv:2502.13965), SGLang/RadixAttention (arXiv:2312.07104), Mooncake (arXiv:2407.00079), vLLM (Kwon et al. SOSP'23), DeepSpeed-FastGen (arXiv:2401.08671)

---

## [2026-04-03] 任务三批处理汇总（task1-10）
- **处理文件数**: 10 个（7 个成功，3 个失败）
- **生成笔记列表**:
  - content/专业学习/ThunderAgent.md（来自 todo/task3.md）
  - content/专业学习/SonicMoE.md（来自 todo/task5.md）
  - content/专业学习/ProRLAgent.md（来自 todo/task9.md）
  - content/专业学习/RLix.md（来自 todo/task8.md）
  - content/专业学习/MetaAttention.md（来自 todo/task7.md）
  - content/专业学习/Medusa_Serverless.md（来自 todo/task10.md）
  - content/专业学习/MagiAttention.md（来自 todo/task6.md）
- **移入 done 目录**: done/task3.md, done/task5.md, done/task6.md, done/task7.md, done/task8.md, done/task9.md, done/task10.md
- **跳过/失败**:
  - task1.md：微信公众号链接无法抓取 + WebSearch RPM 限速，未处理
  - task2.md：circuit breaker 错误（API 熔断），未处理
  - task4.md：微信公众号链接无法抓取 + WebSearch RPM 限速，未处理

---

## [2026-04-09] 任务三批处理汇总（task1-15，本次新增）
- **处理文件数**: 15 个
- **生成笔记列表**:
  - content/专业学习/TurboQuant.md（来自 todo/task1.md）
  - content/专业学习/ClaudeCode与AutoMemory.md（来自 todo/task2.md）
  - content/专业学习/M2RL.md（来自 todo/task3.md）
  - content/专业学习/DWDP.md（来自 todo/task4.md）
  - content/专业学习/SSD代码生成自蒸馏.md（来自 todo/task5.md）
  - content/专业学习/OEL.md（来自 todo/task6.md）
  - content/专业学习/ACE.md（来自 todo/task7.md）
  - content/专业学习/Composer2.md（来自 todo/task8.md）
  - content/专业学习/SparseButCritical.md（来自 todo/task9.md）
  - content/专业学习/FIPO.md（来自 todo/task10.md）
  - content/专业学习/Composer2.md（来自 todo/task11.md，已存在，跳过重建）
  - content/专业学习/FIPO.md（来自 todo/task12.md，已存在，跳过重建）
  - content/专业学习/SparseCritical.md（来自 todo/task13.md）
  - content/专业学习/TriAttention.md（来自 todo/task14.md）
  - content/专业学习/ThinkTwice.md（来自 todo/task15.md）
- **移入 done 目录**: 全部 15 个文件已移入 done/
- **跳过/失败**: task11/12 与 task8/10 为重复论文，sub-agent 正确识别已有笔记并跳过重建

---
