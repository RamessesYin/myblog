---
title: "ProRL Agent：面向多轮 LLM Agent 的 RL 训练 Rollout 即服务基础设施"
tags:
  - NVIDIA
  - RL训练基础设施
  - LLM Agent
  - 多轮交互
  - Rollout服务化
---

## 🧠 核心摘要 (TL;DR)
> ProRL Agent 提出"Rollout-as-a-Service"范式，将 RL 训练的 rollout 生成与策略训练解耦为独立 HTTP 服务，解决了多轮 LLM Agent 训练中 rollout 编排复杂、HPC 无特权沙箱难以部署、Re-tokenization 漂移等工程痛点；在 SWE-Bench Verified 上，Qwen3-14B 经过训练后 pass rate 从 15.4% 提升到 23.6%。

## 🏷️ 元数据
- **论文标题**: ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM Agents
- **发表机构/会议**: NVIDIA / arXiv 预印本
- **年份**: 2026
- **链接**: [[ProRLAgent_paper|📄论文]] | 💻 Code: [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)

---

## 🕸️ 知识图谱索引
- **关键词 (Keywords)**: #rl-training #llm-agent #rollout-service #multi-turn #infrastructure
- **前置知识 (Prerequisites)**: [[RL中的KV稀疏讨论]]
- **相关节点 (Related Notes)**: [[ZipServ]], [[MSched]]
- **向下延申 (Successors)**:

---

## 🎯 动机与痛点

### 现有方案的瓶颈

当前主流的 LLM RL 训练框架（如 VERL、HybridFlow）在处理**多轮 Agent 交互**时面临根本性的架构挑战：

1. **紧耦合问题**：传统框架将 rollout 生成（模型推理 + 环境交互）与策略梯度更新捆绑在同一进程中，两者互相争夺 GPU 显存与计算资源。单轮对话场景尚可接受，但多轮 Agent 任务中，一次 rollout 可能涉及数十次工具调用、代码执行、环境观测，流程极为复杂，难以用简单的批处理范式驾驭。

2. **HPC 无特权部署困境**：学术界和工业界的 HPC 集群通常不允许 root 权限，而 Agent 任务（如软件工程任务需要运行容器沙箱）往往依赖 Docker 等特权容器技术。现有框架缺乏对无特权环境的原生支持，导致研究人员在配置环境上耗费大量精力。

3. **Re-tokenization 漂移（Retokenization Drift）**：在 Agent 多轮对话中，工具调用返回的中间结果（如代码执行输出）通常以字符串形式传递给推理引擎，再被重新 tokenize。然而 tokenization 不具有完美的幂等性——相同的字符串在不同上下文、不同 BOS/EOS 拼接方式下可能产生不同的 token ID 序列，导致训练时记录的 log-prob 与实际推理时的 log-prob 存在微小但系统性的偏差，进而影响 PPO/GRPO 等 on-policy 算法的梯度质量。

4. **任务多样性管理**：不同 Agent 任务（代码、数学、STEM、Web 交互）的环境初始化、动作空间、奖励计算逻辑差异悬殊，现有框架难以以统一接口管理，导致任务适配成本高昂。

### 论文的切入点

ProRL Agent 的核心思路是：**将整个 rollout 生命周期封装为一个可独立扩展的 HTTP 服务**，训练框架只需通过标准 REST API 与 rollout 服务交互，不感知内部的沙箱管理、多轮交互编排等细节。这一设计理念类似于"推理即服务"（Inference-as-a-Service）在 LLM 服务领域的成功实践，ProRL Agent 将其延伸到了 RL 训练的 rollout 阶段。

### 相关工作

#### 🏛️ 先驱/奠基工作

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| **ReAct** (Yao et al., ICLR 2023) | 2022 | 将"推理"与"行动"交替执行的 Agent 框架，是多轮 LLM Agent 的基础范式 |
| **DeepSeek-R1** (Guo et al., 2025, arXiv:2501.12948) | 2025 | 通过 RL（GRPO）激励大模型推理能力，展示了 RL 在 LLM 训练中的巨大潜力 |
| **HybridFlow/VERL** (Sheng et al., EuroSys 2025) | 2025 | 提出灵活的 RLHF 框架，支持 actor、critic、rollout 的灵活部署拓扑 |
| **DAPO** (Yu et al., 2025, arXiv:2503.14476) | 2025 | 开源 LLM RL 系统，提出动态采样策略优化，针对长推理链条优化训练稳定性 |
| **vLLM** (Kwon, 2025, PhD Thesis, UC Berkeley) | 2025 | 高效 LLM 推理引擎，PagedAttention + continuous batching，是 rollout 服务的推理后端 |
| **POMDP** (Kaelbling et al., 1998) | 1998 | 部分可观测马尔可夫决策过程，多轮 Agent 任务的理论基础框架 |

#### ⚔️ 直接对比方法（同类 RL 训练框架）

| 论文 | 年份 | 方法特点 | 与 ProRL Agent 的差异 |
|------|------|---------|---------------------|
| **SkyRL-Agent** (Cao et al., 2025b, arXiv:2511.16108) | 2025 | 多轮 LLM Agent 高效 RL 训练框架 | 无独立 rollout 服务；不支持 rootless HPC |
| **VeRL-Tool** (Jiang et al., 2025, arXiv:2509.01055) | 2025 | 工具集成 RL 系统，整体性 agentic RL | 工具集成与训练耦合，扩展性受限 |
| **Agent Lightning / AGL** (Luo et al., 2025c, arXiv:2508.03680) | 2025 | 任意 AI Agent RL 训练框架；提出"不再需要 re-tokenization"的思路 | 博客形式提出 token-id 传输理念，ProRL Agent 在工程层面系统实现 |
| **ToRL** (Li et al., 2025, arXiv:2503.23383) | 2025 | 工具集成 RL 的规模化 | 聚焦工具调用策略，非基础设施层 |
| **rLLM** (Tan et al., 2025) | 2025 | 语言 Agent 后训练框架 | 架构设计不同，未专门处理多轮 HPC 部署 |
| **VAGEN** (Wang et al., NeurIPS 2025) | 2025 | 多轮 VLM Agent 世界模型强化 | 聚焦视觉语言模型，非通用基础设施 |

#### 📈 Follow-up / 扩展应用

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| **DeepSWE** (Luo et al., 2025a) | 2025 | 通过规模化 RL 训练出 SOTA 编程 Agent，使用 ProRL 类框架 |
| **GEM** (Liu et al., 2025b, arXiv:2510.01051) | 2025 | Agentic LLM 的 Gym 环境 |
| **R2E-Gym** (Jain et al., 2025, arXiv:2504.07164) | 2025 | 过程化环境 + 混合验证器，专用于 SWE Agent 规模化 |
| **Beyond Ten Turns** (Gao et al., 2025, arXiv:2508.07976) | 2025 | 大规模异步 RL 解锁长视野 agentic search（10轮以上） |
| **Search-R1** (Jin et al., 2025, arXiv:2503.09516) | 2025 | 通过 RL 训练 LLM 使用搜索引擎推理 |
| **Nemotron-Tool-N1** (Zhang et al., ICLR 2026) | 2026 | 工具使用语言模型的强化推理，NVIDIA 后续工作 |
| **RAGen-v2** (Wang et al., 2026) | 2026 | 分析多轮 Agent RL 中的推理崩溃问题 |
| **AgentGym-RL** (Xi et al., ICLR 2026) | 2026 | 开源框架，训练 LLM Agent 完成长视野决策任务 |

#### 📊 评测基准

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| **SWE-bench** (Jimenez et al., ICLR 2024, arXiv:2310.06770) | 2023/2024 | 评测语言模型是否能解决真实 GitHub issue 的基准 |
| **WebArena** (Zhou et al., 2023, arXiv:2307.13854) | 2023 | 真实 Web 环境下构建自主 Agent 的评测平台 |
| **OSWorld** (Xie et al., NeurIPS 2024) | 2024 | 真实计算机环境中多模态 Agent 开放任务基准 |
| **BFCL** (Patil et al., ICML 2025) | 2025 | Berkeley 函数调用排行榜，评测工具使用到 agentic 评估 |
| **OpenHands** (Wang et al., ICLR 2025) | 2024 | 开放平台，使 AI 作为通用软件开发 Agent |


## ⚙️ 核心设计与机制

### 架构创新：三层解耦架构

ProRL Agent 的整体架构分为三个独立层次，通过 HTTP 接口连接：

```
┌─────────────────────────────────────────────────────────┐
│                    RL Trainer（训练侧）                   │
│      (VERL / NeMo / 任意支持 HTTP 回调的训练框架)         │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP REST API
┌─────────────────────▼───────────────────────────────────┐
│              ProRL Agent Server（Rollout 服务层）         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │  INIT 池  │→│  RUN 池  │→│      EVAL 池          │   │
│  │ (初始化)  │  │ (执行)   │  │  (奖励计算/验证)      │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
│          Min-Heap 负载均衡 ←── 推理请求路由               │
└─────────────────────┬───────────────────────────────────┘
                      │ 沙箱 API / 推理 API
┌─────────────────────▼───────────────────────────────────┐
│              环境与推理层（可插拔）                        │
│  ┌─────────────────┐    ┌──────────────────────────┐    │
│  │  Sandbox Envs   │    │  Inference Backend        │    │
│  │  (SingularityRT)│    │  (vLLM / SGLang)          │    │
│  └─────────────────┘    └──────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### 关键技术点 1：三阶段异步流水线

ProRL Agent Server 将每条 rollout 轨迹的生成分为三个独立阶段，分别由独立 worker 池处理：

**INIT 阶段（初始化）**：
- 为每个任务实例化沙箱环境（如 Git repo clone、Docker/Singularity 容器启动）
- 为每条轨迹准备初始观测（initial observation）
- 与推理后端（vLLM/SGLang）建立连接

**RUN 阶段（执行）**：
- 执行多轮 Agent 循环：调用推理引擎获取动作 → 执行动作 → 获取环境反馈 → 循环
- 每一步的中间状态（token IDs、动作、观测）全部记录到轨迹缓冲区
- **关键设计**：通过 OpenAI 兼容 API 的扩展字段直接传递 token IDs（而非文本字符串），避免 re-tokenization

**EVAL 阶段（评估）**：
- 轨迹完成后，由独立 worker 运行奖励函数/验证器
- 计算每步或最终的奖励信号
- 将完整轨迹（tokens + rewards）返回给训练框架

**三阶段独立的好处**：不同阶段的计算瓶颈不同（INIT 是 IO 密集，RUN 是推理密集，EVAL 是计算密集），独立 worker 池允许各阶段独立扩缩容，避免互相阻塞。

### 关键技术点 2：Token-in/Token-out（消除 Re-tokenization 漂移）

这是 ProRL Agent 在工程细节上最具创意的设计之一。

**问题根源**：
在多轮 Agent 对话中，每轮的上下文包含：之前所有轮次的 tokens + 当前轮环境观测文本。传统做法是将环境观测以字符串形式返回给推理引擎，推理引擎将其 tokenize 后与历史 tokens 拼接。但 tokenizer 的行为受上下文影响（如某些 BPE 合并规则依赖前缀），同一段文本在不同位置可能被切分为不同 token 序列。

**后果**：训练阶段记录的 log-probabilities 是基于某种 tokenization 的，但推理引擎实际使用的 tokenization 可能不同，导致：
$$\log \pi_\theta(a_t | s_t)_{\text{recorded}} \neq \log \pi_\theta(a_t | s_t)_{\text{actual}}$$

这种偏差在 PPO/GRPO 的 importance sampling ratio $\frac{\pi_\theta}{\pi_{\text{old}}}$ 计算中会被放大，影响训练稳定性。

**解决方案**：ProRL Agent 要求推理后端（vLLM）通过 API 直接返回生成的 **token IDs**（而非仅返回文本），并在下一轮调用时直接将 token IDs 作为输入传入（而非重新 tokenize 文本）。这保证了整个多轮对话的 tokenization 完全一致，彻底消除漂移。

论文将这一理念称为 **"Token-in/Token-out"**，并指出这需要推理引擎在 OpenAI 兼容 API 中暴露 token ID 的输入输出接口（vLLM 已支持此功能）。

### 关键技术点 3：SingularityRuntime（无特权 HPC 容器支持）

HPC 集群（如 SLURM 管理的科研集群）普遍不允许 root 权限，这使得 Docker 等需要特权的容器技术无法使用。但软件工程类 Agent 任务（如 SWE-bench）需要在隔离环境中运行任意代码。

ProRL Agent 的解决方案是集成 **Singularity/Apptainer** 运行时：
- Singularity 是专为 HPC 设计的容器运行时，支持无 root 权限运行容器镜像
- 通过 `SingularityRuntime` 模块将沙箱环境封装为 Singularity 镜像
- Agent 执行的每个代码操作都在独立的 Singularity 容器实例中运行，保证隔离性

这使得 ProRL Agent 能在学术界的 HPC 集群上直接部署，无需申请特殊权限。

### 关键技术点 4：可插拔沙箱抽象（Pluggable Task Abstraction）

ProRL Agent 定义了一套标准的沙箱接口，每种任务类型只需实现三个方法：

```python
class SandboxEnv:
    def initialize(self, task_config) -> Observation:
        """初始化环境，返回初始观测"""
        ...
    
    def step(self, action: str) -> Observation:
        """执行一个 Agent 动作，返回新观测"""
        ...
    
    def evaluate(self, trajectory) -> float:
        """计算轨迹的最终奖励"""
        ...
```

目前已实现的沙箱类型包括：
- **SWE 沙箱**：基于 SWE-bench 的 GitHub 仓库 clone + 代码执行环境
- **数学沙箱**：AMC/AIME 竞赛数学题，使用符号计算验证答案
- **STEM 沙箱**：理工科综合题目，混合验证器
- **代码沙箱**：Codeforces 算法竞赛题，OJ 风格的测试用例验证

这种解耦设计使得新增任务类型只需实现沙箱接口，无需修改 ProRL Agent Server 的核心逻辑。

### 关键技术点 5：Min-Heap 负载均衡

在多个 GPU 节点运行推理后端时（如多台机器各跑 vLLM 实例），ProRL Agent 使用**最小堆（Min-Heap）**数据结构进行推理请求的负载均衡：

- 堆中每个节点记录一个推理实例的当前在途请求数（assignment count）
- 每次新 rollout 发送推理请求时，从堆中取出负载最小的推理实例
- 请求完成后更新对应节点的计数

相比轮询（Round-Robin）或随机分配，Min-Heap 能更好地处理推理延迟不均匀（如不同序列长度导致的推理时间差异）的情况，减少 head-of-line blocking。



## 📊 实验与性能评价

### 实验设置

ProRL Agent 在四个领域的 Agent 任务上进行了评测，均使用 Qwen3 系列模型（4B / 8B / 14B）作为策略网络，通过 GRPO 算法进行 RL 训练。

### 对比基线（Baseline）

实验对比的基线是**相同模型的 zero-shot 推理结果**（即使用相同的 Qwen3 预训练模型，不经过 RL 训练，直接在测试集上推理）。

### 主要实验结果

#### SWE-Bench Verified（软件工程代码修复）

**任务描述**：给定真实 GitHub 仓库和 issue 描述，Agent 需要通过多轮交互（文件读取、代码编辑、测试运行）修复 bug，输出通过测试的 patch。

| 模型 | Baseline（无训练） | ProRL Agent 训练后 | 提升 |
|------|-------------------|--------------------|------|
| Qwen3-4B | 14.8% | 21.2% | +6.4pp |
| Qwen3-8B | 9.6% | 18.0% | +8.4pp |
| Qwen3-14B | 15.4% | 23.6% | +8.2pp |

> 注：指标为 pass rate（通过测试的任务比例）。数据来自论文 Table 1。

#### STEM Agent（理工科综合任务）

通过 RL 训练后，STEM Agent 的 episode reward 从 **0.2 提升至 0.65**，相对提升约 225%。

#### 数学 Agent（AMC 竞赛数学）

Pass@1 指标从 **0.4 提升至 0.9**，Pass@1 翻倍以上。体现 RL 训练在结构化推理任务上的显著增益。

#### 代码 Agent（Codeforces 算法竞赛）

Solve rate 从 **0.23 提升至 0.42**，约提升 83%。

### 系统工程指标

论文并未提供详细的系统吞吐量或延迟数字（如 tokens/s、rollout/h），主要聚焦在**下游任务性能提升**上，将系统的扩展性和可靠性作为定性优势展示。

### 局限性（Limitations）

1. **通信开销**：HTTP 服务化引入了额外的网络通信开销，在单机场景下可能不如紧耦合框架效率高；对于超大规模训练，HTTP 带宽可能成为瓶颈。

2. **评测覆盖有限**：实验只使用了 Qwen3 系列模型，没有与其他 RL 训练框架（如 SkyRL-Agent、VeRL-Tool）在相同模型和任务上做直接对比，缺乏 apples-to-apples 比较。

3. **任务类型仍然有限**：目前开源的沙箱类型（SWE、Math、STEM、Code）覆盖范围相对有限，Web 交互（WebArena）、操作系统任务（OSWorld）等更复杂的多模态 Agent 场景尚未验证。

4. **训练成本未披露**：论文未报告训练所需的 GPU 计算量（GPU-hours）和数据量，难以评估实际落地成本。

---

## 💡 启发与我的想法

### 可借鉴之处

**1. "服务化"范式的可迁移性**

ProRL Agent 将 rollout 从训练循环中抽离为独立服务，这与工业界 ML 系统工程的趋势高度吻合：计算图解耦 → 各部分独立扩缩容 → 标准接口连接。这一思路可迁移到其他 RL 训练瓶颈，例如将**奖励模型（Reward Model）服务化**，使其成为独立的可扩缩计算单元。

**2. Token-in/Token-out 的实践价值**

Re-tokenization 漂移是一个容易被忽视但实际影响训练稳定性的 bug。这一发现提醒我们：在多轮对话 RL 训练中，必须严格控制 tokenization 的一致性。在自行设计类似系统时，推理引擎的接口设计需要支持 token ID 直传，而不只是文本接口。

**3. 沙箱接口抽象的设计模式**

三方法（initialize / step / evaluate）的沙箱抽象非常简洁，完整覆盖了 Agent 任务的生命周期。这种接口设计可以作为构建自定义 RL 训练任务的参考模板。

**4. HPC 适配的工程意识**

在算法论文中专门讨论 HPC 无特权部署问题，体现了工程落地意识。对于需要在学术集群上做 RL 训练实验的研究者，Singularity 替代 Docker 的思路具有直接实用价值。

### 改进空间

**1. 缺乏与同类系统的系统性对比**

论文最大的缺憾是没有与 SkyRL-Agent、VeRL-Tool 等同期工作在相同条件下进行对比实验，无法判断 ProRL Agent 的架构设计带来的实际训练效率提升究竟有多少来自"服务化"本身，有多少来自 token-in/token-out 优化。

**2. 异步流水线的稳定性**

INIT→RUN→EVAL 三阶段异步流水线在实际训练中可能面临状态管理复杂性（如沙箱崩溃、推理超时、部分轨迹丢失）。论文对故障恢复机制的描述较为简略。

**3. 长轨迹训练的显存压力**

多轮 Agent 任务（如 20+ 轮交互）会产生非常长的 token 序列（数万 tokens），对推理显存和训练时的 KV Cache 压力极大。ProRL Agent 未讨论如何与 KV Cache 稀疏化、量化等技术结合以降低长序列训练成本。

### 下一步 Action

- [ ] 阅读 HybridFlow/VERL 论文，理解 ProRL Agent 所解决的架构问题在 VERL 中的原始形态
- [ ] 调研 SkyRL-Agent（arXiv:2511.16108）的架构设计，与 ProRL Agent 做对比
- [ ] 思考 KV Cache 稀疏化技术（参见 KvZap 笔记）与长轨迹 RL 训练的结合点
- [ ] 关注 NVIDIA NeMo Gym 开源仓库（https://github.com/NVIDIA-NeMo/Gym）的后续更新


