---
title: "SGLang 代码架构分析"
tags:
  - 推理引擎
  - SGLang
---

# SGLang 代码架构深度分析

> 本文档基于 SGLang `release/v0.5.9` ，重点覆盖启动流程、请求分发、并行方式、推理调度与 KV Cache。

---

## 目录

1. [主进程启动流程](#1-主进程启动流程)
2. [请求服务框架](#2-请求服务框架)
3. [并行方式的判断、启动与初始化](#3-并行方式的判断启动与初始化)
4. [推理请求处理流程](#4-推理请求处理流程)
5. [弹性并行切换相关要点](#5-弹性并行切换相关要点)

---

## 1. 主进程启动流程

### 1.1 两种入口路径

SGLang 提供两种使用方式，对应不同的入口文件，但**核心启动逻辑完全共享**：

```
┌─────────────────────────────────────────────────────────┐
│                    用户调用方式                           │
├───────────────────────┬─────────────────────────────────┤
│  HTTP Server 模式     │  Python Engine 模式              │
│  python -m sglang...  │  engine = Engine(model_path=..) │
│  http_server.py       │  engine.py                       │
│  :launch_server()     │  :Engine.__init__()              │
├───────────────────────┴─────────────────────────────────┤
│           共同调用 _launch_subprocesses()                 │
│           (engine.py:1013)                               │
└─────────────────────────────────────────────────────────┘
```

| 入口 | 文件 | 函数 | 说明 |
|------|------|------|------|
| HTTP Server | `entrypoints/http_server.py` | `launch_server()` | FastAPI + Uvicorn，提供 REST API |
| Python Engine | `entrypoints/engine.py:118` | `Engine.__init__()` | 纯 Python API，无 HTTP 开销 |

### 1.2 核心启动函数 `_launch_subprocesses()`

> 文件：`entrypoints/engine.py:1013`

这是**整个系统最关键的启动入口**，负责拉起所有子进程并建立 IPC 通道。

```python
def _launch_subprocesses(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable,
    run_scheduler_process_func: Callable,
    run_detokenizer_process_func: Callable,
    port_args: Optional[PortArgs] = None,
) -> Tuple[TokenizerManager, TemplateManager, Tuple[Dict], PortArgs]:
```

**完整启动时序图：**

```
_launch_subprocesses() [主进程]
│
├─ 1. configure_logger() + _set_envs_and_config()      # 全局配置
│     设置 NCCL, CUDA, OMP 等环境变量
│
├─ 2. PortArgs.init_new(server_args)                    # 分配 IPC 端口
│     生成 ZMQ socket 地址 (ipc:// 或 tcp://)
│
├─ 3. _launch_scheduler_processes()  ──────────────┐    # 启动调度器
│     [engine.py:911]                               │
│                                                   │
│     if dp_size == 1:                              │
│       ├─ 遍历 pp_rank × tp_rank                   │
│       ├─ 计算每个 rank 的 gpu_id                   │
│       └─ mp.Process(run_scheduler_process)        │
│     else (dp_size > 1):                           │
│       └─ mp.Process(run_data_parallel_controller) │
│                                                   │
│     各 Scheduler 子进程并行初始化: ◄──────────────┘
│       Scheduler.__init__() [scheduler.py:268]
│         ├─ init_model_config()
│         ├─ init_ipc_channels()      # ZMQ PULL/PUSH
│         ├─ init_tokenizer()
│         ├─ init_model_worker()      # → TpModelWorker → ModelRunner
│         │    └─ 加载模型权重 (最耗时步骤)
│         ├─ init_cache_with_memory_pool()
│         │    ├─ ReqToTokenPool      # 请求→token位置映射
│         │    ├─ TokenToKVPool       # token→KV Cache显存
│         │    └─ RadixCache          # 前缀缓存树
│         ├─ init_schedule_policy()
│         └─ pipe_writer.send({"status": "ready", ...})
│
├─ 4. mp.Process(run_detokenizer_process)               # 启动 Detokenizer
│     [engine.py:1064]
│     DetokenizerManager.__init__() [detokenizer_manager.py:77]
│       ├─ init_ipc_channels()
│       ├─ init_tokenizer()
│       └─ init_running_status()
│
├─ 5. init_tokenizer_manager()                          # 初始化 Tokenizer (主进程)
│     [engine.py:97]
│     TokenizerManager.__init__() [tokenizer_manager.py:191]
│       ├─ init_model_config()
│       ├─ init_tokenizer_and_processor()  # 加载 tokenizer
│       ├─ init_ipc_channels()             # ZMQ PUSH/PULL
│       ├─ init_running_status()
│       └─ init_disaggregation()
│
├─ 6. _wait_for_scheduler_ready()                       # 阻塞等待调度器就绪
│     通过 mp.Pipe 读取 scheduler 发来的 "ready" 信号
│
└─ 7. 返回 (tokenizer_manager, template_manager, scheduler_infos, port_args)
```

### 1.3 进程/线程架构总览

```
┌──────────────────────────────────────────────────────────────┐
│                        主进程 (node_rank=0)                   │
│                                                              │
│  ┌────────────────┐  ┌───────────────┐  ┌────────────────┐  │
│  │ HTTP Server    │  │ Tokenizer     │  │ Template       │  │
│  │ (FastAPI/      │──│ Manager       │  │ Manager        │  │
│  │  Uvicorn)      │  │               │  │                │  │
│  └────────────────┘  └──────┬────────┘  └────────────────┘  │
│                             │ ZMQ PUSH                       │
└─────────────────────────────┼────────────────────────────────┘
                              │
              ┌───────────────┼───────────────────┐
              │               │                   │
              ▼               ▼                   ▼
┌─────────────────┐ ┌─────────────────┐  ┌─────────────────┐
│ 子进程:          │ │ 子进程:          │  │ 子进程:          │
│ Scheduler[0]    │ │ Scheduler[1]    │  │ Detokenizer     │
│ (TP rank 0)     │ │ (TP rank 1)     │  │ Manager         │
│                 │ │                 │  │                 │
│ ┌─────────────┐ │ │ ┌─────────────┐ │  │ recv_from_      │
│ │TpModelWorker│ │ │ │TpModelWorker│ │  │ scheduler       │
│ │ ┌─────────┐ │ │ │ │ ┌─────────┐ │ │  │ (ZMQ PULL)      │
│ │ │ModelRun-│ │ │ │ │ │ModelRun-│ │ │  │                 │
│ │ │  ner    │ │ │ │ │ │  ner    │ │ │  │ send_to_        │
│ │ └─────────┘ │ │ │ │ └─────────┘ │ │  │ tokenizer       │
│ └─────────────┘ │ │ └─────────────┘ │  │ (ZMQ PUSH)      │
└─────────────────┘ └─────────────────┘  └─────────────────┘
        GPU 0               GPU 1

注: 只有 TP rank 0 的 Scheduler 拥有 ZMQ 收发通道,
    其他 TP rank 通过 NCCL broadcast 接收请求。
```

### 1.4 关键组件职责

| 组件 | 文件 | 进程 | 核心职责 |
|------|------|------|---------|
| **TokenizerManager** | `managers/tokenizer_manager.py:188` | 主进程 | 接收请求、tokenize、发送给 Scheduler、收集返回结果 |
| **Scheduler** | `managers/scheduler.py:253` | 子进程 (每 TP group 一个) | 调度 prefill/decode 批次、管理 KV Cache、驱动推理 |
| **TpModelWorker** | `managers/tp_worker.py:206` | Scheduler 内部 | 管理 ModelRunner，执行 forward pass |
| **ModelRunner** | `model_executor/model_runner.py` | TpModelWorker 内部 | 加载模型、执行前向推理、CUDA Graph |
| **DetokenizerManager** | `managers/detokenizer_manager.py:74` | 子进程 | 将 token id 转为文本，支持增量解码 |
| **DataParallelController** | `managers/data_parallel_controller.py:116` | 子进程 (dp>1时) | 在 DP 模式下分发请求到多个 Scheduler |

### 1.5 IPC 通信架构

所有进程间通信基于 **ZMQ (ØMQ)** 库，使用如下 socket 模式：

```
TokenizerManager ──PUSH──► scheduler_input_ipc_name ──PULL──► Scheduler[0]
                                                                   │
Scheduler[0] ──PUSH──► detokenizer_ipc_name ──PULL──► DetokenizerManager
                                                                   │
DetokenizerManager ──PUSH──► tokenizer_ipc_name ──PULL──► TokenizerManager
```

**IPC 地址命名** (由 `PortArgs.init_new()` 在 `server_args.py:5606` 生成):
- 单机模式: `ipc:///tmp/sglang_{random_hex}_{component}`
- 多机 DP-Attention 模式: `tcp://{host}:{port}`

**同步机制**：
- **Scheduler 就绪**: 通过 `mp.Pipe` 发送 `{"status": "ready"}` 字典
- **NCCL 组初始化**: 通过 `nccl_port` 在 TP group 内部同步
- **DP Attention 端口广播**: 通过 ZMQ REQ/REP 在节点间同步

---

## 2. 请求服务框架

### 2.1 HTTP API 端点

> 文件：`entrypoints/http_server.py`

SGLang 通过 FastAPI 注册了**35+ 个 API 端点**，在 `lifespan()` 函数 (line 260) 中初始化各 Serving 处理器：

| 端点 | 方法 | Serving 类 | 说明 |
|------|------|-----------|------|
| `/generate` | POST/PUT | 直接调用 TokenizerManager | 原生 SGLang 生成接口 |
| `/v1/chat/completions` | POST | `OpenAIServingChat` | OpenAI 兼容聊天 |
| `/v1/completions` | POST | `OpenAIServingCompletion` | OpenAI 兼容补全 |
| `/v1/embeddings` | POST | `OpenAIServingEmbedding` | Embedding 接口 |
| `/v1/responses` | POST | `OpenAIServingResponses` | OpenAI Responses API |
| `/encode` | POST/PUT | 直接调用 TokenizerManager | 编码接口 |
| `/v1/models` | GET | - | 模型列表 |
| `/get_model_info` | GET | - | 模型信息 |
| `/flush_cache` | GET/POST | - | 刷新 RadixCache |
| `/control/standby` | POST | StandbyManager | **弹性并行：进入待机** |
| `/control/activate` | POST | StandbyManager | **弹性并行：激活** |
| `/control/status` | GET | - | **弹性并行：状态查询** |

### 2.2 请求完整生命周期

以 `/generate` 端点为例，一个请求的完整流转如下：

```
                        ┌─────────────────────────┐
                        │   HTTP Client (curl等)   │
                        └───────────┬─────────────┘
                                    │ POST /generate
                                    ▼
                     ┌──────────────────────────────┐
                     │  FastAPI Handler              │
          ┌──────────│  generate_request()           │
          │          │  [http_server.py:798]          │
          │          └──────────────────────────────┘
          │                     │
          │                     ▼
          │          ┌──────────────────────────────┐
          │          │  TokenizerManager             │
Step 1    │          │  .generate_request()          │
Tokenize  │          │  [tokenizer_manager.py:490]   │
          │          │                              │
          │          │  1. normalize_batch_and_args()│
          │          │  2. _tokenize_one_request()   │
          │          │  3. _send_one_request()       │
          │          │     → ZMQ PUSH to Scheduler   │
          │          │  4. _wait_one_response()      │
          │          │     ← asyncio.Event 等待      │
          │          └──────────────────────────────┘
          │                     │ ZMQ PUSH
          │                     ▼
          │          ┌──────────────────────────────┐
          │          │  Scheduler 事件循环            │
Step 2    │          │  event_loop_normal()          │
Schedule  │          │  [scheduler.py:1108]          │
& Infer   │          │                              │
          │          │  1. recv_requests()           │
          │          │  2. process_input_requests()  │
          │          │  3. get_next_batch_to_run()   │
          │          │  4. run_batch(batch)           │
          │          │  5. process_batch_result()    │
          │          │     → 发送 output tokens       │
          │          └──────────────────────────────┘
          │                     │ ZMQ PUSH
          │                     ▼
          │          ┌──────────────────────────────┐
Step 3    │          │  DetokenizerManager           │
Detokenize│          │  event_loop()                 │
          │          │  [detokenizer_manager.py:144] │
          │          │                              │
          │          │  1. recv_from_scheduler       │
          │          │  2. 增量 detokenize            │
          │          │  3. send_to_tokenizer         │
          │          └──────────────────────────────┘
          │                     │ ZMQ PUSH
          │                     ▼
          │          ┌──────────────────────────────┐
Step 4    │          │  TokenizerManager             │
Return    │          │  handle_loop() 收到结果       │
          │          │                              │
          │          │  1. 根据 rid 找到 ReqState     │
          │          │  2. 设置 asyncio.Event         │
          │          │  3. _wait_one_response() 恢复  │
          └──────────│  4. yield 结果给 HTTP handler  │
                     └──────────────────────────────┘
                                    │
                                    ▼
                        ┌─────────────────────────┐
                        │   HTTP Response / SSE    │
                        └─────────────────────────┘
```

### 2.3 TokenizerManager 详解

> 文件：`managers/tokenizer_manager.py:188`

TokenizerManager 是请求进入系统的**第一个处理环节**（在主进程中运行）：

```python
class TokenizerManager(TokenizerCommunicatorMixin, TokenizerManagerMultiItemMixin):
    def __init__(self, server_args, port_args):
        self.init_model_config()                    # 读取模型配置
        self.init_tokenizer_and_processor()          # 加载 tokenizer (HF)
        self.init_ipc_channels(port_args)            # 创建 ZMQ sockets
        self.init_running_status()                   # rid_to_state 字典等
        self.init_request_logging_and_dumping()
        self.init_weight_update()
        self.init_lora()
        self.init_disaggregation()
        self.init_metric_collector_watchdog()
        self.init_request_dispatcher()
```

**核心方法 `generate_request()`** (line 490):

```python
async def generate_request(self, obj, request):
    obj.normalize_batch_and_arguments()     # 标准化输入

    if obj.is_single:
        tokenized_obj = await self._tokenize_one_request(obj)
        state = self._send_one_request(obj, tokenized_obj, created_time)
        async for response in self._wait_one_response(obj, state, request):
            yield response
    else:
        # 批量请求：并行 tokenize + 发送
        async for response in self._handle_batch_request(obj, request):
            yield response
```

**请求状态跟踪**：每个请求通过 `ReqState` 对象管理，以 `rid` (request ID) 为键存储在 `rid_to_state` 字典中。当 DetokenizerManager 返回结果时，通过 `asyncio.Event` 唤醒等待中的协程。

### 2.4 DP 模式下的请求分发

> 文件：`managers/data_parallel_controller.py:116`

当 `dp_size > 1` 时，系统插入一个 **DataParallelController** 进程来分发请求：

```
TokenizerManager ──ZMQ PUSH──► DataParallelController ──ZMQ PUSH──► Scheduler[dp_rank=0]
                                       │                          ► Scheduler[dp_rank=1]
                                       │                          ► Scheduler[dp_rank=2]
                                       │
                                  调度策略选择
```

**4 种负载均衡策略** (`LoadBalanceMethod` 枚举，line 70):

| 策略 | 枚举值 | 算法 | 适用场景 |
|------|--------|------|---------|
| `ROUND_ROBIN` | 轮询 | 按序分发到下一个 worker | 请求大致均匀 |
| `TOTAL_REQUESTS` | 最少请求 | 选择当前 `num_reqs` 最少的 worker | 通用场景 |
| `TOTAL_TOKENS` | 最少 Token | 选择 `num_tokens` 最少的 worker (以 requests 为 tiebreaker) | 长短请求混合 |
| `FOLLOW_BOOTSTRAP_ROOM` | 粘性路由 | 根据 `bootstrap_room` 路由到固定 worker | 会话保持 |

**请求分发流程** (简化)：

```python
class DataParallelController:
    def __init__(self, ...):
        # 接收请求的 ZMQ PULL socket
        self.recv_from_tokenizer = get_zmq_socket(ctx, zmq.PULL, ...)

        # 每个 DP worker 一个 ZMQ PUSH socket
        self.workers: List[zmq.Socket] = [None] * dp_size

    # TypeBasedDispatcher 根据消息类型路由：
    #   TokenizedGenerateReqInput → dispatching() (选一个 worker)
    #   BlockReqInput             → send_to_all_workers() (广播)
    #   WatchLoadUpdateReq        → handle_load_update_req() (更新负载信息)
    #   其他控制消息              → send_control_message() (发给第一个)
```

**负载信息更新**: 每个 Scheduler 定期通过 `WatchLoadUpdateReq` 上报自身负载 (`num_reqs`, `num_tokens`)，DP Controller 使用 `DPBudget` (line 87) 类跟踪并据此分发。

### 2.5 DetokenizerManager 详解

> 文件：`managers/detokenizer_manager.py:74`

Detokenizer 运行在独立子进程中，事件循环极简：

```python
def event_loop(self):   # line 144
    while True:
        recv_obj = self.recv_from_scheduler.recv_pyobj()  # ZMQ PULL
        output = self._request_dispatcher(recv_obj)       # 处理
        if output is not None:
            self.send_to_tokenizer.send_pyobj(output)     # ZMQ PUSH
```

**增量解码机制**：使用 `DecodeStatus` 对象跟踪每个请求的解码状态，支持增量 detokenize（避免每次从头解码），通过 `LimitedCapacityDict` (默认容量 65536) 管理状态。

---

## 3. 并行方式的判断、启动与初始化

### 3.1 并行参数定义

> 文件：`server_args.py`

| 参数 | 行号 | 默认值 | 说明 |
|------|------|--------|------|
| `--tp-size` | ~350 | 1 | Tensor Parallelism，模型权重切分到多个 GPU |
| `--dp-size` | ~419 | 1 | Data Parallelism，多个完整模型副本 |
| `--pp-size` | ~351 | 1 | Pipeline Parallelism，模型按层切分 |
| `--ep-size` | ~494 | 1 | Expert Parallelism，MoE 专家并行 |
| `--enable-dp-attention` | - | False | DP Attention 模式(共享 KV Cache) |
| `--moe-dp-size` | - | 1 | MoE 数据并行度 |
| `--attn-cp-size` | - | 1 | Attention Context Parallelism |
| `--moe-dense-tp-size` | - | - | Dense 层的 TP 大小 |

### 3.2 并行层次结构

SGLang 定义了一个**嵌套的并行层次** (在 `engine.py:951` 注释中):

```
并行层次 (从外到内):
  Attention 路径: Global(TP) → DP → ATTN_CP → ATTN_TP (最内层)
  MoE 路径:      Global(TP) → MOE_DP → EP → MOE_TP (最内层)
```

### 3.3 模式判断与分支逻辑

启动时的**核心分支点**在 `_launch_scheduler_processes()` (engine.py:911):

```python
def _launch_scheduler_processes(server_args, port_args, ...):
    if server_args.dp_size == 1:
        # ────────── TP/PP 模式（无 DP） ──────────
        # 遍历 pp_rank_range × tp_rank_range
        # 为每个 (pp_rank, tp_rank) 组合启动一个 Scheduler 进程
        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                gpu_id = base_gpu_id + pp_rank * tp_size_per_node + tp_rank * gpu_id_step
                mp.Process(target=run_scheduler_process, args=(..., gpu_id, tp_rank, ...))
    else:
        # ────────── DP 模式 ──────────
        # 启动一个 DataParallelController 进程
        # 它内部再为每个 dp_rank 启动各自的 TP group
        mp.Process(target=run_data_parallel_controller_process, ...)
```

### 3.4 各并行模式对比

#### 3.4.1 TP (Tensor Parallelism)

```
┌─────────────────────────────────────┐
│         TP Group (tp_size=4)         │
│                                     │
│  GPU0      GPU1      GPU2      GPU3 │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐│
│  │1/4  │  │1/4  │  │1/4  │  │1/4  ││
│  │模型 │  │模型 │  │模型 │  │模型 ││
│  │权重 │  │权重 │  │权重 │  │权重 ││
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘│
│     └────────┴────────┴────────┘    │
│           NCCL AllReduce            │
└─────────────────────────────────────┘

进程数: tp_size 个 Scheduler 进程 (每个对应一个 GPU)
通信: NCCL (AllReduce, AllGather)
特点: 只有 tp_rank=0 的 Scheduler 有 ZMQ 通道
      其他 rank 通过 NCCL broadcast 接收请求
```

**TP 初始化关键路径**:
1. `run_scheduler_process()` (scheduler.py:3068) → `Scheduler.__init__()`
2. `init_model_worker()` → `TpModelWorker.__init__()` (tp_worker.py:206)
3. TpModelWorker 内部创建 `ModelRunner`
4. `ModelRunner.__init__()` (model_runner.py) 调用 `init_distributed_environment()` → 创建 NCCL 通信组

**NCCL 通信组创建** (`distributed/parallel_state.py:1661`):
```python
# TP group: 连续 rank 构成一组
# 例如 8 GPU, tp=2: [0,1], [2,3], [4,5], [6,7]
for i in range(0, world_size, tp_size):
    ranks = list(range(i, i + tp_size))
    tp_group = init_process_group(ranks)
```

**通信后端选择** (parallel_state.py:216-627):

SGLang 实现了**多层通信后端**，按 tensor 大小自动选择最优方案：

| 后端 | 文件 | 适用场景 |
|------|------|---------|
| PyNccl | `device_communicators/pynccl.py` | 大 tensor all-reduce |
| CustomAllReduce | `device_communicators/custom_all_reduce.py` | 小 tensor (IPC + 共享内存) |
| PyMscclpp | `device_communicators/pymscclpp.py` | MSCCL++ 后端 |
| TorchSymmMem | `device_communicators/torch_symm_mem.py` | 对称内存通信 |
| FlashInfer Fusion | `layers/flashinfer_comm_fusion.py` | Attention 融合通信 |

#### 3.4.2 DP (Data Parallelism)

```
┌─────────────────────────────────────────────────────┐
│                DataParallelController                 │
│            [data_parallel_controller.py:116]          │
│                                                     │
│  ┌──────────┐  负载均衡  ┌──────────┐              │
│  │ 请求分发  │──────────►│ DPBudget │              │
│  └─────┬────┘           └──────────┘              │
│        │                                           │
│   ┌────┴────┬────────────┬────────────┐           │
│   ▼         ▼            ▼            ▼           │
│ DP Rank 0  DP Rank 1  DP Rank 2  DP Rank 3      │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│ │完整模型│ │完整模型│ │完整模型│ │完整模型│     │
│ │GPU 0   │ │GPU 1   │ │GPU 2   │ │GPU 3   │     │
│ └────────┘ └────────┘ └────────┘ └────────┘     │
└─────────────────────────────────────────────────────┘

进程数: 1 (Controller) + dp_size 个 Scheduler 进程
通信: ZMQ (Controller↔Scheduler), 无 NCCL 跨 DP rank
特点: 每个 DP rank 是完整的独立推理实例
```

**DP 启动流程** (`launch_dp_schedulers()`, line 222):

```python
def launch_dp_schedulers(self, server_args, port_args):
    for dp_rank in range(server_args.dp_size):
        tmp_port_args = PortArgs.init_new(server_args)  # 每个 DP rank 独立端口
        tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name  # 共享 tokenizer 端口
        tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name  # 共享 detokenizer 端口

        # 在独立线程中启动 TP group
        thread = Thread(target=self.launch_tensor_parallel_group_thread,
                        args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ...))
        base_gpu_id += tp_size * pp_size * gpu_id_step  # GPU 偏移
```

#### 3.4.3 DP + TP 混合

```
┌─────────────────────────────────────────────────┐
│              DataParallelController               │
│                                                 │
│  DP Rank 0 (tp_size=2)    DP Rank 1 (tp_size=2)│
│  ┌──────┐  ┌──────┐      ┌──────┐  ┌──────┐   │
│  │1/2模型│  │1/2模型│      │1/2模型│  │1/2模型│   │
│  │GPU 0  │  │GPU 1  │      │GPU 2  │  │GPU 3  │   │
│  └──┬───┘  └──┬───┘      └──┬───┘  └──┬───┘   │
│     └──NCCL──┘              └──NCCL──┘         │
└─────────────────────────────────────────────────┘

GPU 总数 = dp_size × tp_size × pp_size
```

#### 3.4.4 EP (Expert Parallelism)

EP 专门用于 **Mixture of Experts (MoE)** 模型，在 TP 的基础上进一步切分专家：

```
┌────────────────────────────────────────────┐
│          EP Group (ep_size=4)               │
│                                            │
│  GPU0         GPU1         GPU2         GPU3│
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐│
│  │Expert  │  │Expert  │  │Expert  │  │Expert  ││
│  │0,1     │  │2,3     │  │4,5     │  │6,7     ││
│  └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘│
│       └──────────┴──────────┴──────────┘   │
│              All-to-All Token Dispatch      │
└────────────────────────────────────────────┘

关键文件: layers/moe/ep_moe/layer.py
通信: All-to-All (每个 token 路由到对应专家所在 GPU)
后端: DeepEP, Mooncake, FlashInfer, Mori
```

**EP 与 TP 的关系**：EP 是 TP 的一种特化形式。`ep_size` 必须能整除 `tp_size`。在 EP 模式下，Dense 层 (Attention) 仍使用 TP 方式 AllReduce，而 MoE 层使用 All-to-All 路由。

**EP rank 计算** (engine.py:960):
```python
moe_ep_rank = (
    tp_rank
    % (server_args.tp_size // server_args.moe_dp_size)
    // (server_args.tp_size // server_args.moe_dp_size // server_args.ep_size)
)
```

#### 3.4.5 PP (Pipeline Parallelism)

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ PP Stage 0│───►│ PP Stage 1│───►│ PP Stage 2│
│ Layer 0-10│    │ Layer 11-21│   │ Layer 22-31│
│ GPU 0     │    │ GPU 1     │    │ GPU 2     │
└──────────┘    └──────────┘    └──────────┘
                P2P 通信

关键文件: distributed/parallel_state.py:1850
约束: 不能与 context parallelism 或 moe_dp_size > 1 同时使用
```

### 3.5 参数约束与校验

> `server_args.py:check_server_args()` (~line 2085)

| 约束 | 说明 |
|------|------|
| `ep_size` 必须整除 `tp_size` | EP 在 TP 内部 |
| `dp_size == 1` 或 `tp_size % dp_size == 0` | DP 需要均匀分配 |
| PP 不能与 CP 同时使用 | 实现限制 |
| `moe_dp_size > 1` 不能与 PP 同时使用 | 实现限制 |
| `GPU 总数 = dp_size × tp_size × pp_size` | 必须有足够 GPU |

---

## 4. 推理请求处理流程

### 4.1 Scheduler 事件循环

> 文件：`managers/scheduler.py`

Scheduler 有两种事件循环模式：

| 模式 | 函数 | 行号 | 特点 |
|------|------|------|------|
| Normal | `event_loop_normal()` | 1108 | 串行：接收→调度→推理→处理结果 |
| Overlap | `event_loop_overlap()` | 1134 | **CPU/GPU 重叠**：当前批次 GPU 执行时，CPU 处理上一批次结果 |

**Normal 模式核心循环** (简化):

```python
def event_loop_normal(self):
    while True:
        # 1. 接收新请求 (ZMQ PULL)
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)

        if self._engine_paused:
            continue

        # 2. 调度决策：选择下一个要执行的 batch
        batch = self.get_next_batch_to_run()

        # 3. 执行推理
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.self_check_during_idle()

        self.last_batch = batch
```

**Overlap 模式** 的核心优化是：在 GPU 执行当前 batch 的 forward 时，CPU 同时处理上一个 batch 的 result（采样、更新状态等）。吞吐量提升 1.5~3×。

```
时间轴 (Overlap 模式):
───────────────────────────────────────────────────
GPU:  [  forward(batch_N)  ][  forward(batch_N+1)  ]
CPU:  [process(batch_N-1)] [  process(batch_N)    ]
      ↑ 重叠区域           ↑ 重叠区域
───────────────────────────────────────────────────
```

### 4.2 请求接收与分发

**`recv_requests()`** (line 1217):

```python
def recv_requests(self):
    """在 tp_rank=0 接收请求，并 broadcast 给所有其他 TP rank。"""
    if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
        # 只有 rank 0 有 ZMQ socket
        recv_reqs = []
        while self.recv_from_tokenizer.poll(timeout=...):
            recv_req = self.recv_from_tokenizer.recv_pyobj()
            recv_reqs.append(recv_req)
            if self.recv_limit_reached(len(recv_reqs)):
                break

        if self.tp_size > 1:
            # Broadcast 给其他 TP rank
            broadcast_recv_input(recv_reqs, self.tp_rank, self.tp_cpu_group)
    else:
        recv_reqs = broadcast_recv_input(None, self.tp_rank, self.tp_cpu_group)

    return recv_reqs
```

### 4.3 Prefill vs Decode 调度策略

> **核心函数**: `get_next_batch_to_run()` (line 1875)

SGLang 使用 **Prefill 优先的连续批处理 (Continuous Batching)** 策略：

```python
def get_next_batch_to_run(self):
    # 1. 将上一轮 prefill 完成的请求合入 running_batch (decode 队列)
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.running_batch.merge_batch(self.last_batch)

    # 2. 尝试构建新的 prefill batch
    new_batch = self.get_new_batch_prefill()

    # 3. Prefill 优先！
    if new_batch is not None:
        return new_batch        # 有新请求要 prefill → 先做 prefill
    else:
        # 没有新 prefill → 执行 decode
        if not self.running_batch.is_empty():
            self.running_batch = self.update_running_batch(self.running_batch)
            return self.running_batch
        return None             # 系统空闲
```

**调度决策流程图：**

```
                    get_next_batch_to_run()
                           │
                           ▼
                ┌─ 合并上轮 prefill 到 running_batch ─┐
                │  (完成 prefill 的请求转入 decode)      │
                └──────────────────────────────────────┘
                           │
                           ▼
                 get_new_batch_prefill()
                       ╱         ╲
                有新请求?         无新请求
                  ╱                   ╲
                 ▼                     ▼
          ┌───────────┐       running_batch 非空?
          │ 返回      │          ╱         ╲
          │ prefill   │        是            否
          │ batch     │        ╱              ╲
          └───────────┘       ▼                ▼
                     ┌──────────────┐   ┌──────────┐
                     │ 返回 decode  │   │ 返回 None│
                     │ batch       │   │ (空闲)    │
                     └──────────────┘   └──────────┘
```

### 4.4 Prefill Batch 构建详解

> 函数：`get_new_batch_prefill()` → `_get_new_batch_prefill_raw()` (line 1977)

```python
def _get_new_batch_prefill_raw(self):
    # 1. 检查 grammar 就绪队列
    # 2. 创建 PrefillAdder (管理内存预算)
    adder = PrefillAdder(
        page_size, tree_cache, token_to_kv_pool_allocator,
        running_batch, new_token_ratio, max_prefill_tokens,
        chunked_prefill_size, ...
    )

    # 3. 处理分块 prefill 中断的请求
    if self.chunked_req is not None:
        self.chunked_req = adder.add_chunked_req(self.chunked_req)

    # 4. 从 waiting_queue 中取请求
    for req in self.waiting_queue:
        # 检查 running_batch 是否已满
        if self.running_batch.batch_is_full:
            break

        # 初始化请求的下一轮输入 (查找 prefix cache)
        req.init_next_round_input(self.tree_cache)

        # 尝试添加到 batch (检查 token 预算)
        res = adder.add_one_req(req, ...)
        if res != AddReqResult.CONTINUE:
            break

    # 5. 创建 ScheduleBatch
    can_run_list = adder.can_run_list
    new_batch = ScheduleBatch.init_new(can_run_list, ...)
    new_batch.prepare_for_extend()      # 准备 extend 模式的 attention metadata

    # 6. Mixed Chunked Prefill (可选)
    #    将 decode 请求混入 prefill batch 以提高 GPU 利用率
    if self.is_mixed_chunk and not self.running_batch.is_empty():
        new_batch.mix_with_running(self.running_batch)

    return new_batch
```

**分块 Prefill (Chunked Prefill)**：当请求的 prompt 很长时，分多次 prefill 处理，每次处理 `chunked_prefill_size` 个 token，避免单次 prefill 占用过多 GPU 时间影响 decode 延迟。

**Mixed Chunked Prefill**：在执行 prefill 时，将正在 decode 的请求也混入同一个 batch（decode 请求只需计算 1 个 token），最大化 GPU 利用率。

### 4.5 Forward 模式

SGLang 定义了多种 Forward 模式：

| 模式 | 含义 | 对应 ModelRunner 函数 |
|------|------|----------------------|
| `EXTEND` | Prefill（处理新 prompt） | `forward_extend()` (line 2307) |
| `DECODE` | 自回归解码（生成下一个 token） | `forward_decode()` (line 2284) |
| `IDLE` | 空闲（DP Attention 同步用） | `forward_idle()` (line 2347) |
| `SPLIT_PREFILL` | 分块 prefill | `forward_extend()` |

### 4.6 Forward Pass 执行

> 函数：`run_batch()` (scheduler.py:2278)

```python
def run_batch(self, batch):
    self.forward_ct += 1

    if self.is_generation:
        worker_batch = batch.get_model_worker_batch()

        if self.enable_overlap:
            # Overlap 模式：在独立 CUDA stream 上异步执行
            with self.forward_stream_ctx:
                batch_result = self.model_worker.forward_batch_generation(worker_batch)
                batch_result.copy_to_cpu(...)  # 异步拷贝结果到 CPU
        else:
            batch_result = self.model_worker.forward_batch_generation(worker_batch)

    return batch_result
```

**Batch 数据层次** (三层转换):

```
ScheduleBatch                     # 调度层：包含 Req 列表、调度元数据
    │
    ▼  batch.get_model_worker_batch()
ModelWorkerBatch                  # Worker 层：提取推理所需的张量
    │
    ▼  model_runner 内部转换
ForwardBatch                      # 推理层：GPU 张量，直接喂入模型
    ├─ input_ids: torch.Tensor
    ├─ seq_lens: torch.Tensor
    ├─ req_pool_indices: torch.Tensor
    ├─ attn_backend: AttentionBackend
    └─ ...
```

### 4.7 CUDA Graph 加速

> 文件：`model_executor/cuda_graph_runner.py:238`

对于 **decode 阶段** (batch size 固定，序列长度每步+1)，SGLang 使用 CUDA Graph 录制一次 forward pass，后续直接 replay，避免 Python/CUDA 启动开销。

```python
class CudaGraphRunner:
    def __init__(self, model_runner):
        self.graphs = {}           # bs → CUDAGraph
        self.output_buffers = {}   # bs → 预分配输出缓冲

    def capture(self, batch_size):
        # 录制 CUDA Graph (warmup 阶段)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model_runner.model.forward(...)
        self.graphs[batch_size] = graph

    def replay(self, batch_size):
        # Replay (推理阶段，极快)
        self.graphs[batch_size].replay()
        return self.output_buffers[batch_size]
```

**CUDA Graph 约束**：
- 仅用于 decode 模式（batch size 可预测）
- Prefill 由于序列长度变化大，不使用 CUDA Graph
- 通过 `torch_memory_saver` 可以 pause/resume CUDA Graph 的显存

### 4.8 请求生命周期状态机

一个请求从到达到完成，经历以下状态：

```
  ┌──────────┐
  │ WAITING  │◄──────────── 新请求到达，加入 waiting_queue
  └────┬─────┘
       │  被 PrefillAdder 选中
       ▼
  ┌──────────────┐
  │ PREFILL      │◄──────── get_new_batch_prefill() 构建 prefill batch
  │ (EXTEND)     │          forward_extend() 执行
  └────┬─────────┘
       │  prefill 完成，合入 running_batch
       ▼
  ┌──────────────┐
  │ DECODE       │◄──────── 循环执行 forward_decode()
  │ (自回归循环)  │          每步生成一个 token
  └────┬─────────┘
       │  触发终止条件 (EOS / max_tokens / stop_str)
       ▼
  ┌──────────────┐
  │ FINISHED     │◄──────── process_batch_result() 检测到完成
  └────┬─────────┘
       │  从 running_batch 中移除
       │  发送结果到 DetokenizerManager
       ▼
  ┌──────────────┐
  │ DETOKENIZE   │◄──────── DetokenizerManager 增量解码
  └────┬─────────┘
       │  发回 TokenizerManager
       ▼
  ┌──────────────┐
  │ RETURNED     │◄──────── 通过 asyncio.Event 唤醒 HTTP handler
  └──────────────┘          返回给客户端
```

**特殊情况**：
- **Retract (显存不足回退)**：decode 过程中显存不足时，低优先级请求会被 retract 回 waiting_queue
- **Chunked Prefill 中断**：长 prompt 分多次处理，中间状态保存在 `self.chunked_req`
- **Streaming**：decode 每步都会将部分结果发送到 Detokenizer，实现流式输出

### 4.9 KV Cache 管理

#### 4.9.1 三层存储结构

```
┌────────────────────────────────────────────────────────┐
│                    RadixCache (前缀树)                   │
│                    [mem_cache/radix_cache.py:261]        │
│                                                        │
│    root                                                │
│    ├── "The cat" ──► Node (ref_count=2)                │
│    │   ├── " sat on" ──► Node                          │
│    │   └── " is"     ──► Node                          │
│    └── "Hello" ──► Node (ref_count=1)                  │
│        └── " world" ──► Node                           │
│                                                        │
│    每个 Node 存储: token_ids[], kv_indices[], ref_count  │
│    kv_indices 指向 TokenToKVPool 中的物理位置            │
└────────────────────────────────────────────────────────┘
         │
         │ kv_indices
         ▼
┌────────────────────────────────────────────────────────┐
│              TokenToKVPool (物理 KV 显存)                │
│              [mem_cache/memory_pool.py:697]              │
│                                                        │
│  k_buffer: List[torch.Tensor]  # 每层一个              │
│      shape: [pool_size, head_num, head_dim]             │
│  v_buffer: List[torch.Tensor]  # 每层一个              │
│      shape: [pool_size, head_num, head_dim]             │
│                                                        │
│  Token 粒度的 KV Cache 存储 (支持 page_size > 1)       │
└────────────────────────────────────────────────────────┘
         ▲
         │ req_to_token 映射
┌────────────────────────────────────────────────────────┐
│              ReqToTokenPool (请求→token映射)             │
│              [mem_cache/memory_pool.py:126]              │
│                                                        │
│  req_to_token: torch.Tensor                            │
│      shape: [max_requests, max_context_len]             │
│      dtype: int32                                      │
│                                                        │
│  req_to_token[req_pool_idx][seq_pos] = kv_pool_index   │
│  实现了请求到物理 KV Cache 位置的间接映射               │
└────────────────────────────────────────────────────────┘
```

**三层关系**:
```
请求 req_i → req_pool_idx → req_to_token[req_pool_idx][pos] → kv_pool_index
                                                                    │
                                                                    ▼
                                              k_buffer[layer][kv_pool_index]
                                              v_buffer[layer][kv_pool_index]
```

这种**间接寻址**设计的好处：
1. 允许 KV Cache 碎片化分配（不需要连续显存）
2. 支持前缀共享（多个请求指向同一 KV Cache 位置）
3. 支持 page_size > 1 的分页管理

#### 4.9.2 RadixCache 前缀匹配

> 函数：`match_prefix()` (radix_cache.py:352)

当新请求到达时，Scheduler 通过 RadixCache 查找已有的 KV Cache 前缀：

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """查找 key (token序列) 在 radix tree 中的最长缓存前缀。
    
    返回:
      - device_indices: 匹配到的 KV cache 索引
      - last_node: 匹配终止的树节点
    """
```

**前缀匹配流程**:
```
输入 token 序列: [A, B, C, D, E, F]

RadixCache 树:
    root → [A, B, C] → Node1 (kv_indices=[0,1,2])
                └─► [D, E] → Node2 (kv_indices=[3,4])

匹配结果:
    matched_prefix = [A, B, C, D, E]  (5 tokens命中)
    device_indices = [0, 1, 2, 3, 4]   (可直接复用的 KV Cache)
    need_prefill = [F]                  (仅需 prefill 1 个 token)
```

**前缀缓存的巨大价值**：对于多轮对话、系统 prompt 相同的请求，可以大幅减少重复计算。

#### 4.9.3 缓存淘汰策略

RadixCache 支持 **6 种淘汰策略** (radix_cache.py):

| 策略 | 说明 |
|------|------|
| `LRU` (默认) | 最近最少使用 |
| `LFU` | 最不频繁使用 |
| `FIFO` | 先进先出 |
| `MRU` | 最近最多使用 |
| `FILO` | 后进先出 |
| `Priority` | 优先级驱动 |

#### 4.9.4 KV Cache 分配流程

```
请求到达 Scheduler
    │
    ▼
req.init_next_round_input(tree_cache)
    │
    ├─ match_prefix() → 找到已缓存的前缀
    │   返回 prefix_indices (已有 KV cache 的位置)
    │
    ├─ extend_prefix_len = total_len - prefix_len
    │   (需要新计算的 token 数)
    │
    ▼
PrefillAdder.add_one_req(req)
    │
    ├─ 检查 token_to_kv_pool_allocator 是否有足够空间
    │   available = allocator.available_size()
    │   needed = extend_prefix_len × (1 + new_token_ratio)
    │                                 ↑ 预留 decode 空间
    │
    ├─ 如果空间不够 → evict_to_free(num_needed)
    │   从 RadixCache 淘汰叶子节点，回收 KV Cache 空间
    │
    └─ 分配新的 kv_pool_indices
        allocator.alloc(num_tokens) → 返回物理位置索引
```

### 4.10 SchedulePolicy (调度策略)

`init_schedule_policy()` 中根据 `--schedule-policy` 参数选择策略：

| 策略 | 算法 | 适用场景 |
|------|------|---------|
| `lpm` (默认) | Longest Prefix Match，优先调度 prefix cache 命中率高的请求 | 通用 |
| `fcfs` | First Come First Served，FIFO | 公平性要求高 |
| `random` | 随机选择 | 测试/基准 |
| `lof` | Longest Output First | 特殊优化 |

---

## 5. 弹性并行切换相关要点

本节总结架构分析中与弹性并行切换项目直接相关的关键点：

### 5.1 切换涉及的进程拓扑变化

从 TP4 切换到 DP4 意味着：

| | TP4 | DP4 |
|---|---|---|
| Scheduler 进程 | 4 个 (同一 TP group) | 1 Controller + 4 个独立 |
| NCCL 通信组 | 1 组 (4 ranks) | 无跨进程通信 |
| 模型副本 | 1/4 × 4 | 完整 × 4 |
| 请求分发 | 直接到 TP rank 0 | Controller 负载均衡 |

### 5.2 需要 Pause/Resume 的组件

| 组件 | 持有的 GPU 资源 | Pause 方式 | 对应设计文档 |
|------|----------------|-----------|-------------|
| ModelRunner 权重 | 模型参数 tensor | `torch_memory_saver.pause()` | 模块2 |
| CUDA Graph | 录制的 graph + 输出缓冲 | `torch_memory_saver.pause()` (需 `SGLANG_MEMORY_SAVER_CUDA_GRAPH=True`) | 模块3 |
| KV Cache Pool | `k_buffer`, `v_buffer` | `torch_memory_saver.pause()` | 模块6 |
| NCCL 通信组 | symmetric memory, buffers | `ncclCommSuspend()` | 模块5 |
| 辅助 GPU 内存 | 各种 buffer (CustomAllReduce 等) | VMM 改造 / 销毁重建 | 模块7 |

### 5.3 关键代码锚点

| 功能 | 文件:行号 | 备注 |
|------|----------|------|
| Standby 进入点 | `http_server.py` `/control/standby` | 当前实现在 HTTP 层 |
| Scheduler 事件循环 | `scheduler.py:1108` | Phase 2 需将 standby 逻辑下沉到此处 |
| `_engine_paused` 标志 | `scheduler.py:1114` | 已有 pause 机制，可复用 |
| torch_memory_saver 适配器 | `utils/torch_memory_saver_adapter.py` | pause/resume 权重 |
| NCCL 通信组 | `distributed/parallel_state.py` | 需要 suspend/resume |
| RadixCache | `mem_cache/radix_cache.py:261` | 切换时可能需要迁移/丢弃 |
| ReqToTokenPool / TokenToKVPool | `mem_cache/memory_pool.py` | 显存大头 |

### 5.4 请求排空 (Drain) 路径

切换前必须确保所有在途请求处理完毕：

```
1. pause_generation(retract=True)     # 停止接收新 prefill，retract 正在 decode 的请求
2. 轮询 /get_load → 等待 num_reqs == 0  # 所有请求处理完毕
3. enter_standby()                     # 进入待机
```

---

## 附录 A: 关键文件索引

| 文件路径 | 行数 | 核心内容 |
|----------|------|---------|
| `entrypoints/engine.py` | ~1090 | Engine 类、`_launch_subprocesses()`、`_launch_scheduler_processes()` |
| `entrypoints/http_server.py` | ~2100 | FastAPI 路由、`launch_server()`、Standby 接口 |
| `managers/tokenizer_manager.py` | ~1200 | TokenizerManager、`generate_request()`、请求 tokenize |
| `managers/scheduler.py` | ~3200 | Scheduler、事件循环、batch 调度、prefill/decode 逻辑 |
| `managers/schedule_batch.py` | ~2400 | ScheduleBatch、Req、ForwardMode、ModelWorkerBatch |
| `managers/detokenizer_manager.py` | ~500 | DetokenizerManager、增量 detokenize |
| `managers/data_parallel_controller.py` | ~500 | DP 控制器、负载均衡策略 |
| `managers/tp_worker.py` | ~500 | TpModelWorker、forward_batch_generation |
| `model_executor/model_runner.py` | ~2700 | ModelRunner、forward_extend/decode、模型加载 |
| `model_executor/cuda_graph_runner.py` | ~700 | CUDA Graph 录制与 replay |
| `mem_cache/radix_cache.py` | ~1000 | RadixCache 前缀树、匹配、淘汰 |
| `mem_cache/memory_pool.py` | ~2000 | ReqToTokenPool、MHATokenToKVPool、显存管理 |
| `distributed/parallel_state.py` | ~1900 | NCCL 组创建、通信后端初始化 |
| `server_args.py` | ~5700 | 所有命令行参数、PortArgs |

---

## 附录 B: 数据结构关系图

```
GenerateReqInput (HTTP 请求体)
    │
    ▼ tokenize
TokenizedGenerateReqInput (token 化后)
    │
    ▼ Scheduler 接收
Req (调度层请求对象, schedule_batch.py)
    ├── rid: str                    # 请求 ID
    ├── input_ids: List[int]        # 输入 token
    ├── prefix_indices: torch.Tensor # 已缓存前缀的 KV 索引
    ├── req_pool_idx: int           # ReqToTokenPool 中的位置
    ├── output_ids: List[int]       # 已生成的 token
    └── sampling_params: SamplingParams
        │
        ▼ 组装批次
ScheduleBatch (调度批次)
    ├── reqs: List[Req]
    ├── forward_mode: ForwardMode (EXTEND / DECODE / IDLE)
    └── req_to_token_pool, token_to_kv_pool_allocator
        │
        ▼ get_model_worker_batch()
ModelWorkerBatch (worker 层数据)
    ├── input_ids: torch.Tensor     # GPU tensor
    ├── seq_lens: torch.Tensor
    └── sampling_info: SamplingBatchInfo
        │
        ▼ ModelRunner 内部转换
ForwardBatch (推理层数据)
    ├── input_ids, seq_lens, ...    # 最终 GPU tensor
    └── attn_backend: AttentionBackend
        │
        ▼ model.forward()
LogitsProcessorOutput
    ├── next_token_logits: torch.Tensor
    └── next_token_ids: torch.Tensor (采样后)
```

<!-- END OF DOCUMENT -->
