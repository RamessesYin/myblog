---
title: "SGLang 关键组件与机制详解"
tags:
  - 推理引擎
  - SGLang
---

> 本文档是 `sglang_architecture.md` 的补充深入文档，针对三个主题进行源码级剖析：
> 1. TokenizerManager 的作用与进程间关系
> 2. Scheduler 子进程的作用与进程间关系
> 3. ZMQ 通信机制详解
> 4. `attn_cp_size` (Attention Context Parallelism) 完整分析

---

## 目录

- [1. TokenizerManager 详解](#1-tokenizermanager-详解)
  - [1.1 定位与职责](#11-定位与职责)
  - [1.2 初始化流程](#12-初始化流程)
  - [1.3 核心方法剖析](#13-核心方法剖析)
  - [1.4 与其他组件的关系](#14-与其他组件的关系)
- [2. Scheduler 子进程详解](#2-scheduler-子进程详解)
  - [2.1 定位与职责](#21-定位与职责)
  - [2.2 初始化流程](#22-初始化流程)
  - [2.3 事件循环剖析](#23-事件循环剖析)
  - [2.4 与其他组件的关系](#24-与其他组件的关系)
- [3. ZMQ 通信机制详解](#3-zmq-通信机制详解)
  - [3.1 ZMQ 是什么](#31-zmq-是什么)
  - [3.2 SGLang 使用的 Socket 类型](#32-sglang-使用的-socket-类型)
  - [3.3 三条核心数据管道](#33-三条核心数据管道)
  - [3.4 IPC 地址命名规则](#34-ipc-地址命名规则)
  - [3.5 各进程 ZMQ 初始化代码](#35-各进程-zmq-初始化代码)
  - [3.6 发送与接收 API](#36-发送与接收-api)
  - [3.7 Bind vs Connect 规则](#37-bind-vs-connect-规则)
  - [3.8 DP 模式下的额外 ZMQ 通道](#38-dp-模式下的额外-zmq-通道)
- [4. attn_cp_size (Context Parallelism) 详解](#4-attn_cp_size-context-parallelism-详解)
  - [4.1 概念：什么是 Context Parallelism](#41-概念什么是-context-parallelism)
  - [4.2 参数定义与约束](#42-参数定义与约束)
  - [4.3 并行层次与 Rank 计算](#43-并行层次与-rank-计算)
  - [4.4 NCCL 通信组创建](#44-nccl-通信组创建)
  - [4.5 请求接收与三层广播](#45-请求接收与三层广播)
  - [4.6 序列切分方式](#46-序列切分方式)
  - [4.7 前向计算中的 CP 通信](#47-前向计算中的-cp-通信)
  - [4.8 Prefill 完整流程示例](#48-prefill-完整流程示例)
  - [4.9 Decode 阶段的行为](#49-decode-阶段的行为)
  - [4.10 当前使用限制](#410-当前使用限制)
  - [4.11 通信组全局拓扑](#411-通信组全局拓扑)
  - [4.12 与弹性并行切换的关系](#412-与弹性并行切换的关系)

---

## 1. TokenizerManager 详解

### 1.1 定位与职责

TokenizerManager (`managers/tokenizer_manager.py:188`) 运行在**主进程**中，是整个系统的**"前台总管"**——所有用户请求的第一站和最后一站。

```python
class TokenizerManager(TokenizerCommunicatorMixin, TokenizerManagerMultiItemMixin):
    """TokenizerManager is a process that tokenizes the text."""
```

**四大核心职责**：

| 职责 | 说明 | 关键方法 |
|------|------|---------|
| ① 请求入口 | 接收 HTTP 请求，tokenize 文本为 token ids，通过 ZMQ 发给 Scheduler | `generate_request()`, `_tokenize_one_request()`, `_send_one_request()` |
| ② 结果回收 | 从 Detokenizer 接收处理结果，唤醒等待中的 HTTP 协程 | `handle_loop()`, `_handle_batch_output()` |
| ③ 状态管理 | 管理所有在途请求的状态（`rid_to_state` 字典） | `ReqState` 数据类 |
| ④ 辅助功能 | LoRA 验证、多模态预处理、权重更新协调、指标收集 | 各 init_* 方法 |

### 1.2 初始化流程

> `TokenizerManager.__init__()` — line 191

```python
def __init__(self, server_args, port_args):
    self.init_model_config()                    # 读取模型配置、context_len 等
    self.init_tokenizer_and_processor()          # 加载 HuggingFace tokenizer
    self.init_ipc_channels(port_args)            # ★ 创建 ZMQ sockets
    self.init_running_status()                   # rid_to_state 字典、负载计数器
    self.init_request_logging_and_dumping()       # 请求日志
    self.init_weight_update()                    # 权重更新锁
    self.init_lora()                             # LoRA 适配器状态
    self.init_disaggregation()                   # PD 分离模式
    self.init_metric_collector_watchdog()         # 指标 + 看门狗
    self.init_request_dispatcher()               # 消息类型路由表
```

### 1.3 核心方法剖析

#### `generate_request()` — 请求入口 (line 490)

```python
async def generate_request(self, obj, request):
    obj.normalize_batch_and_arguments()     # 标准化输入格式

    async with self.is_pause_cond:
        await self.is_pause_cond.wait_for(lambda: not self.is_pause)  # 等待非暂停状态

    async with self.model_update_lock.reader_lock:
        if obj.is_single:
            # 单请求路径
            tokenized_obj = await self._tokenize_one_request(obj)      # Tokenize
            state = self._send_one_request(obj, tokenized_obj, time)   # ZMQ 发送
            async for response in self._wait_one_response(obj, state, request):
                yield response                                         # 等待并返回
        else:
            # 批量请求路径
            async for response in self._handle_batch_request(obj, request):
                yield response
```

#### `_send_one_request()` — ZMQ 发送 (line 1057)

```python
def _send_one_request(self, obj, tokenized_obj, created_time):
    self.send_to_scheduler.send_pyobj(tokenized_obj)     # ★ ZMQ PUSH 发出
    state = ReqState([], False, asyncio.Event(), obj)     # 创建等待状态
    self.rid_to_state[obj.rid] = state                    # 按 rid 存储
    return state
```

#### `_wait_one_response()` — 等待结果 (line 1100)

```python
async def _wait_one_response(self, obj, state, request):
    while True:
        await asyncio.wait_for(state.event.wait(), timeout=TIMEOUT)  # ★ 阻塞等待 Event
        out = state.out_list[-1]
        if state.finished:
            del self.rid_to_state[obj.rid]    # 清理
            yield out
            break
        else:
            yield out    # streaming: 中间结果
            state.event.clear()
```

#### `handle_loop()` — 结果接收循环 (line 1467)

```python
async def handle_loop(self):
    """从 Detokenizer 接收结果的异步事件循环"""
    while True:
        recv_obj = await self.recv_from_detokenizer.recv_pyobj()  # ★ ZMQ PULL
        self._result_dispatcher(recv_obj)    # 分发到 _handle_batch_output()
```

#### `_handle_batch_output()` — 结果分发 (line 1476)

```python
def _handle_batch_output(self, recv_obj):
    for i, rid in enumerate(recv_obj.rids):
        state = self.rid_to_state.get(rid)
        if state is None:
            continue

        # 构建返回值
        meta_info = {"id": rid, "finish_reason": recv_obj.finished_reasons[i], ...}
        out = {"text": recv_obj.output_strs[i], "meta_info": meta_info, ...}

        state.out_list.append(out)
        state.finished = recv_obj.finished_reasons[i] is not None
        state.event.set()    # ★ 唤醒 _wait_one_response() 中的协程
```

#### `ReqState` 数据类 (line 129)

```python
@dataclass
class ReqState:
    out_list: List[Dict]           # 输出缓冲区
    finished: bool                 # 是否完成
    event: asyncio.Event           # ★ 用于唤醒等待协程的信号
    obj: GenerateReqInput          # 原始请求引用

    # 性能指标
    created_time: float
    first_token_time: float = 0.0
    request_sent_to_scheduler_ts: float = 0.0
    response_sent_to_client_ts: float = 0.0

    # Streaming 增量状态
    text: str = ""
    output_ids: List[int] = field(default_factory=list)
    last_output_offset: int = 0
```

### 1.4 与其他组件的关系

```
                    ┌─ HTTP Server (FastAPI) ─┐
                    │  调用 generate_request() │
                    └───────────┬─────────────┘
                                │ 同一进程，直接函数调用
                                ▼
┌───────────────────────────────────────────────────────┐
│             TokenizerManager (主进程)                   │
│                                                       │
│  发送侧:                          接收侧:             │
│  send_to_scheduler               recv_from_detokenizer│
│  (ZMQ PUSH)                      (ZMQ asyncio PULL)   │
│  ↓                                ↑                   │
│  连接到 scheduler_input_ipc       绑定 tokenizer_ipc   │
│                                                       │
│  rid_to_state = {                                     │
│    "req-001": ReqState(event=Event(), ...),           │
│    "req-002": ReqState(event=Event(), ...),           │
│  }                                                    │
└───────────────────────────────────────────────────────┘
        │ ZMQ PUSH                       ▲ ZMQ PUSH
        ▼                                │
  Scheduler 子进程  ──ZMQ PUSH──►  Detokenizer 子进程
```

**关键点**：
- TokenizerManager 与 HTTP Server **在同一进程**，直接函数调用
- 与 Scheduler 和 Detokenizer 是**跨进程**的，通过 ZMQ
- 它**不直接接触模型和 GPU**——所有推理由 Scheduler 完成
- 使用 `zmq.asyncio.Context` (异步版本)，与 FastAPI 的 asyncio 事件循环兼容

---

## 2. Scheduler 子进程详解

### 2.1 定位与职责

Scheduler (`managers/scheduler.py:253`) 是运行在**独立子进程**中的推理调度核心，是整个系统**最复杂**的组件。每个 TP rank 一个 Scheduler 进程，但只有 `(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)` 的那个拥有 ZMQ 通道。

```python
class Scheduler(
    SchedulerOutputProcessorMixin,    # 结果处理
    SchedulerUpdateWeightsMixin,      # 权重更新
    SchedulerProfilerMixin,           # 性能分析
    SchedulerMetricsMixin,            # 指标收集
    SchedulerDisaggregationDecodeMixin,  # PD分离-Decode
    SchedulerDisaggregationPrefillMixin, # PD分离-Prefill
    SchedulerMultiplexMixin,          # PD复用
    SchedulerRuntimeCheckerMixin,     # 运行时检查
    SchedulerPPMixin,                 # Pipeline 并行
    SchedulerDPAttnMixin,             # DP Attention
    SchedulerDllmMixin,               # 扩散式 LLM
):
    """A scheduler that manages a tensor parallel GPU worker."""
```

**六大核心职责**：

| 职责 | 说明 | 关键函数 |
|------|------|---------|
| ① 事件循环 | 无限循环：收请求→调度→推理→处理结果 | `event_loop_normal()`, `event_loop_overlap()` |
| ② 调度决策 | Prefill 优先的连续批处理 | `get_next_batch_to_run()`, `get_new_batch_prefill()` |
| ③ 驱动推理 | 调用 TpModelWorker 执行 forward pass | `run_batch()` |
| ④ KV Cache 管理 | RadixCache + 内存池 + 前缀匹配 | `init_cache_with_memory_pool()` |
| ⑤ 结果处理 | 检查完成、发送输出、retract | `process_batch_result()` |
| ⑥ 请求广播 | rank 0 收到请求后 NCCL broadcast 给其他 TP rank | `recv_requests()` |

### 2.2 初始化流程

> `Scheduler.__init__()` — line 268

```python
def __init__(self, server_args, port_args, gpu_id, tp_rank, moe_ep_rank,
             pp_rank, attn_cp_rank, moe_dp_rank, dp_rank):

    # ─── 基础配置 ───
    self.init_model_config()                      # 模型配置
    self.init_metrics(tp_rank, pp_rank, dp_rank)  # 指标
    self.init_ipc_channels(port_args)             # ★ ZMQ sockets

    # ─── 模型与推理 ───
    self.init_tokenizer()                         # 加载 tokenizer (检测 stop token)
    self.init_moe_gemm_config()                   # MoE/FP8 配置
    self.init_model_worker()                      # ★★ 创建 TpModelWorker → ModelRunner → 加载模型权重 (最耗时)

    # ─── 缓存与内存 ───
    self.init_cache_with_memory_pool()            # ★★ ReqToTokenPool + TokenToKVPool + RadixCache

    # ─── 调度策略 ───
    self.init_running_status()                    # waiting_queue, running_batch 等
    self.init_chunked_prefill()                   # 分块 prefill 配置
    self.init_schedule_policy()                   # LPM/FCFS/Random 调度策略

    # ─── 高级特性 ───
    self.init_watch_dog_memory_saver()            # 看门狗 + torch_memory_saver
    self.init_disaggregation()                    # PD 分离
    self.init_overlap()                           # CPU/GPU 重叠调度
    self.init_request_dispatcher()                # 消息类型路由表
    self.grammar_manager = GrammarManager(self)   # 约束生成语法
```

### 2.3 事件循环剖析

#### Normal 模式 (line 1108)

```python
def event_loop_normal(self):
    while True:
        # 1. 接收新请求 (ZMQ PULL + NCCL broadcast)
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)    # 放入 waiting_queue

        if self._engine_paused:
            continue    # standby 模式：不调度

        # 2. 调度决策
        batch = self.get_next_batch_to_run()       # ★ prefill 优先
        self.cur_batch = batch

        # 3. 执行推理 + 处理结果
        if batch:
            result = self.run_batch(batch)           # → TpModelWorker.forward_batch_generation()
            self.process_batch_result(batch, result)  # 检查完成、发 token 给 detokenizer
        else:
            self.self_check_during_idle()            # 空闲自检

        self.last_batch = batch
```

#### Overlap 模式 (line 1134)

Overlap 模式的核心优化：GPU 执行当前 batch 的 forward 时，CPU 同时处理上一 batch 的结果。

```
时间轴:
GPU: ──[forward(batch_N)]──[forward(batch_N+1)]──[forward(batch_N+2)]──
CPU: ──[process(N-1)]──────[process(N)]──────────[process(N+1)]────────
       ↑ 重叠              ↑ 重叠                ↑ 重叠
```

吞吐量提升 1.5~3×。

#### 调度决策: `get_next_batch_to_run()` (line 1875)

```python
def get_next_batch_to_run(self):
    # 1. 合并上一轮 prefill 完成的请求到 running_batch (decode 队列)
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.running_batch.merge_batch(self.last_batch)

    # 2. 尝试构建新的 prefill batch
    new_batch = self.get_new_batch_prefill()

    # 3. ★ Prefill 优先！
    if new_batch is not None:
        return new_batch          # 有新请求要 prefill → 先做 prefill
    else:
        # 没有新 prefill → 执行 decode
        if not self.running_batch.is_empty():
            return self.running_batch
        return None               # 系统空闲
```

#### 请求接收: `recv_requests()` (line 1217)

```python
def recv_requests(self):
    if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
        # ★ 只有这个 rank 有 ZMQ socket
        recv_reqs = []
        while True:
            try:
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                recv_reqs.append(recv_req)
            except zmq.ZMQError:
                break
    else:
        recv_reqs = None    # 其他 rank 没有 ZMQ

    # 通过 NCCL broadcast 分发给所有 TP rank
    if self.tp_size != 1:
        recv_reqs = broadcast_pyobj(recv_reqs, ..., self.tp_cpu_group, ...)
    return recv_reqs
```

### 2.4 与其他组件的关系

```
              TokenizerManager (主进程)
                     │
                     │ ZMQ PUSH (TokenizedGenerateReqInput)
                     ▼
┌────────────────────────────────────────────────────────┐
│          Scheduler[tp_rank=0] (子进程)                   │
│                                                        │
│  recv_from_tokenizer  ← ZMQ PULL   ← 只有 rank 0 有   │
│  send_to_detokenizer  → ZMQ PUSH   → 发结果给 Detoken  │
│  send_to_tokenizer    → ZMQ PUSH   → 发控制信号        │
│                                                        │
│  内部对象 (非独立进程):                                  │
│  ┌──────────────────────────────────────┐              │
│  │ TpModelWorker                        │              │
│  │  └─ ModelRunner                      │              │
│  │      ├─ 模型权重 (GPU)                │              │
│  │      └─ CudaGraphRunner (decode加速) │              │
│  └──────────────────────────────────────┘              │
│                                                        │
│  缓存与内存 (GPU):                                      │
│  ├─ RadixCache (前缀树)                                 │
│  ├─ TokenToKVPool (KV Cache 显存)                       │
│  └─ ReqToTokenPool (请求→token 位置映射)                │
│                                                        │
│  调度状态:                                               │
│  ├─ waiting_queue: [Req, Req, ...]                     │
│  └─ running_batch: ScheduleBatch                       │
└──────┬─────────────────────────────────────────────────┘
       │ NCCL broadcast (请求) + NCCL AllReduce (推理)
       ▼
┌────────────────────────────────────────────────────────┐
│          Scheduler[tp_rank=1] (子进程)                   │
│                                                        │
│  recv_from_tokenizer = None  ← 没有 ZMQ！              │
│  send_to_detokenizer = None                            │
│                                                        │
│  TpModelWorker → ModelRunner  ← 持有 1/tp_size 模型权重 │
│  RadixCache + MemoryPool      ← 自己的 KV Cache        │
└────────────────────────────────────────────────────────┘
```

**关键点**:
- Scheduler 是**子进程**（`mp.Process` 创建），有自己独立的 GPU 上下文
- 使用同步 `zmq.Context` (非 asyncio)，因为 Scheduler 有自己的事件循环
- TpModelWorker 和 ModelRunner 是 Scheduler **内部的对象**，不是独立进程
- 只有 `(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)` 拥有 ZMQ 通道
- 其他 TP rank 通过 **NCCL broadcast** 被动接收请求

---

## 3. ZMQ 通信机制详解

### 3.1 ZMQ 是什么

**ZMQ (ZeroMQ / ØMQ)** 是一个高性能的**异步消息传递库**，可以理解为"超级 socket"。它不是消息中间件（不需要 broker 服务器），而是直接在进程之间建立点对点连接。

**为什么 SGLang 选择 ZMQ**:

| 方案 | 对比 |
|------|------|
| `multiprocessing.Queue` | 太慢，有 GIL 问题，不支持异步 |
| gRPC | 太重，序列化开销大 |
| 共享内存 | 需要手动同步，容易出 bug |
| **ZMQ** | ✅ 零拷贝、异步、支持多模式、极低延迟、跨进程/跨机器统一接口 |

### 3.2 SGLang 使用的 Socket 类型

**① PUSH/PULL (管道模式)** — 单向数据流
```
发送方 (PUSH) ─────────► 接收方 (PULL)
  只能发送                 只能接收
  非阻塞发送               阻塞/非阻塞接收
  多个 PULL 时自动负载均衡
```
用于：所有数据流通道（请求分发、结果回传）

**② DEALER (双向异步)** — 请求/响应
```
Engine (DEALER) ◄─────────► Scheduler (DEALER)
  发送+接收                   发送+接收
```
用于：Engine 的 RPC 通道（直接控制 Scheduler）

### 3.3 三条核心数据管道

整个系统有**三条核心 ZMQ 管道**，形成环路：

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  管道1: TokenizerManager → Scheduler                         │
│    endpoint: scheduler_input_ipc_name                        │
│    模式: PUSH → PULL                                         │
│    数据: TokenizedGenerateReqInput (tokenized 后的请求)       │
│                                                              │
│  管道2: Scheduler → DetokenizerManager                       │
│    endpoint: detokenizer_ipc_name                            │
│    模式: PUSH → PULL                                         │
│    数据: BatchTokenIDOutput (输出的 token ids)                │
│                                                              │
│  管道3: DetokenizerManager → TokenizerManager                │
│    endpoint: tokenizer_ipc_name                              │
│    模式: PUSH → PULL                                         │
│    数据: BatchStrOutput (解码后的文本)                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘

图示:
TokenizerManager ──PUSH──► scheduler_input_ipc ──PULL──► Scheduler
       ▲                                                     │
       │                                                     │
  tokenizer_ipc ◄──PULL── DetokenizerManager ◄──PULL── detokenizer_ipc
                             PUSH                       PUSH
```

### 3.4 IPC 地址命名规则

地址由 `PortArgs.init_new()` (`server_args.py:5606`) 生成：

```python
# 单机模式 — Unix Domain Socket (快，仅本机)
scheduler_input_ipc_name = "ipc:///tmp/sglang_{random_hex}_scheduler_input"
tokenizer_ipc_name       = "ipc:///tmp/sglang_{random_hex}_tokenizer"
detokenizer_ipc_name     = "ipc:///tmp/sglang_{random_hex}_detokenizer"

# 多机模式 — TCP (跨网络)
scheduler_input_ipc_name = "tcp://{host}:{port}"
```

### 3.5 各进程 ZMQ 初始化代码

**创建工具函数** (`utils/common.py:1416`):
```python
def get_zmq_socket(context, socket_type, endpoint, bind):
    """
    参数:
      context:     zmq.Context — ZMQ 上下文
      socket_type: zmq.PUSH / zmq.PULL / zmq.DEALER
      endpoint:    "ipc:///tmp/xxx" 或 "tcp://host:port"
      bind:        True=绑定(服务端), False=连接(客户端)

    规则: 一端 bind，另一端 connect，两者连接到同一个 endpoint
    """
    socket = context.socket(socket_type)
    config_socket(socket, socket_type)    # 设置缓冲区 (大内存机器 0.5GB)
    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket
```

**TokenizerManager** (line 321-339):
```python
def init_ipc_channels(self, port_args):
    context = zmq.asyncio.Context(2)     # ★ 异步版本 (配合 FastAPI asyncio)

    # 从 Detokenizer 接收结果 (服务端)
    self.recv_from_detokenizer = get_zmq_socket(
        context, zmq.PULL, port_args.tokenizer_ipc_name, bind=True
    )
    # 向 Scheduler 发送请求 (服务端)
    self.send_to_scheduler = get_zmq_socket(
        context, zmq.PUSH, port_args.scheduler_input_ipc_name, bind=True
    )
```

**Scheduler** (line 420-460):
```python
def init_ipc_channels(self, port_args):
    context = zmq.Context(2)             # ★ 同步版本 (Scheduler 自己的事件循环)

    if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
        # ★ 只有这一个 rank 有 ZMQ 通道
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, bind=False  # 客户端
        )
        send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_ipc_name, bind=False      # 客户端
        )
        send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, bind=False        # 客户端
        )
    else:
        self.recv_from_tokenizer = None     # 其他 rank 没有 ZMQ
        self.send_to_detokenizer = SenderWrapper(None)
```

**DetokenizerManager** (line 101-108):
```python
def init_ipc_channels(self, port_args):
    context = zmq.Context(2)             # 同步版本

    # 从 Scheduler 接收 token ids (服务端)
    self.recv_from_scheduler = get_zmq_socket(
        context, zmq.PULL, port_args.detokenizer_ipc_name, bind=True
    )
    # 向 Tokenizer 发送解码后的文本 (客户端)
    self.send_to_tokenizer = get_zmq_socket(
        context, zmq.PUSH, port_args.tokenizer_ipc_name, bind=False
    )
```

### 3.6 发送与接收 API

SGLang 使用 ZMQ 的 Python pickle 序列化接口：

```python
# ─── 发送 (任何 Python 对象，自动 pickle 序列化) ───
socket.send_pyobj(tokenized_request)            # 同步版本
await socket.send_pyobj(tokenized_request)      # 异步版本 (TokenizerManager 用)

# ─── 接收 ───
obj = socket.recv_pyobj()                       # 同步阻塞
obj = await socket.recv_pyobj()                 # 异步 (TokenizerManager 用)

# ─── 非阻塞轮询 (Scheduler 事件循环中用) ───
recv_req = socket.recv_pyobj(zmq.NOBLOCK)       # 无消息时抛 zmq.ZMQError
if socket.poll(timeout=0):                      # 检查是否有消息
    obj = socket.recv_pyobj()
```

### 3.7 Bind vs Connect 规则

| 组件 | endpoint | 角色 | 原因 |
|------|----------|------|------|
| **TokenizerManager** → scheduler_input_ipc | **bind** (服务端) | 先启动 |
| **Scheduler** ← scheduler_input_ipc | **connect** (客户端) | 后启动，连接已存在的地址 |
| **DetokenizerManager** → detokenizer_ipc | **bind** (服务端) | 先启动 |
| **Scheduler** → detokenizer_ipc | **connect** (客户端) | 后启动 |
| **TokenizerManager** ← tokenizer_ipc | **bind** (服务端) | 接收结果 |
| **DetokenizerManager** → tokenizer_ipc | **connect** (客户端) | 发结果 |

**规则**：谁先启动谁 bind，后启动的 connect。一个 endpoint 上只能有一个 bind，可以有多个 connect。

### 3.8 DP 模式下的额外 ZMQ 通道

当 `dp_size > 1` 时，TokenizerManager 和 Scheduler 之间插入 DataParallelController：

```
TokenizerManager ──PUSH──► scheduler_input_ipc ──PULL──► DataParallelController
                                                              │
                          ┌──PUSH──► dp_0_scheduler_input ──PULL──► Scheduler[dp=0]
                          ├──PUSH──► dp_1_scheduler_input ──PULL──► Scheduler[dp=1]
                          └──PUSH──► dp_2_scheduler_input ──PULL──► Scheduler[dp=2]
```

每个 DP rank 的 Scheduler 有**独立的 `scheduler_input_ipc_name`**，但共享相同的 `tokenizer_ipc_name` 和 `detokenizer_ipc_name`（因为只有一个 TokenizerManager 和一个 DetokenizerManager）。

---

## 4. attn_cp_size (Context Parallelism) 详解

### 4.1 概念：什么是 Context Parallelism

**Context Parallelism (CP)** 是一种**将长序列切分到多个 GPU 上并行计算 Attention** 的策略。解决的核心问题：

> 当单个请求的 prompt 非常长（几万甚至几十万 token）时，一个 GPU 的显存放不下完整的 KV Cache，或 Attention 计算时间过长。

**与其他并行方式的区别**:

| 并行方式 | 切什么 | 解决什么问题 |
|---------|--------|-------------|
| **TP** (Tensor) | 切**模型权重** (按 head/hidden dim) | 单卡放不下模型 |
| **DP** (Data) | 切**请求** (不同请求到不同副本) | 提高吞吐量 |
| **PP** (Pipeline) | 切**层** (前 N 层/后 N 层) | 单卡放不下模型 |
| **EP** (Expert) | 切**专家** (MoE 专家到不同卡) | 专家太多 |
| **CP** (Context) | 切**序列** (同一请求的 token 分到不同卡) | **单个序列太长** |

### 4.2 参数定义与约束

> 文件：`server_args.py`

```python
# line 422
attn_cp_size: int = 1      # 默认 1，即不启用

# line 3311 — CLI 参数
"--attn-cp-size", type=int, help="The attention context parallelism size."
# 别名: "--attention-context-parallel-size"
```

**约束检查** (line 2085-2093):
```python
if self.attn_cp_size > 1:
    assert self.tp_size % self.attn_cp_size == 0          # tp_size 必须能被 cp_size 整除
    assert self.tp_size % (self.dp_size * self.attn_cp_size) == 0  # 综合整除
    assert self.pp_size == 1   # ★ 不能与 Pipeline Parallelism 同时使用
```

### 4.3 并行层次与 Rank 计算

SGLang 的嵌套并行层次 (`engine.py:951`):
```
Attention 路径: Global(TP) → DP → ATTN_CP → ATTN_TP (最内层)
MoE 路径:      Global(TP) → MOE_DP → EP → MOE_TP (最内层)
```

**维度关系**:
```
tp_size (全局 world_size) = attn_dp_size × attn_cp_size × attn_tp_size
```

**Rank 计算公式** (`engine.py:954-957`, `dp_attention.py:230-244`):
```python
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)
```

**具体例子: 8 GPU, attn_cp_size=2, dp_size=1**:
```
attn_tp_size = 8 / 1 / 2 = 4

GPU:           g0    g1    g2    g3  │  g4    g5    g6    g7
tp_rank:        0     1     2     3  │   4     5     6     7
attn_tp_rank:   0     1     2     3  │   0     1     2     3
attn_cp_rank:   0     0     0     0  │   1     1     1     1
                                     │
          ┌── CP Rank 0 ─────────────┘── CP Rank 1 ──────────┐
          │ 处理序列的一半 token          处理另一半 token      │
          │ 4路 TP 切模型权重            4路 TP 切模型权重      │
          └──────────────────────────────────────────────────┘
```

### 4.4 NCCL 通信组创建

> 文件：`distributed/parallel_state.py:1707-1738`

```python
attn_cp_size = attention_context_model_parallel_size
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

# CP 组：同一 attn_tp_rank、同一 dp_idx 的 GPU 组成一个 CP group
for tp_group_idx in range(num_tp_groups):
    for dp_idx in range(attn_dp_size):
        for attn_tp_idx in range(attn_tp_size):
            st = tp_group_idx * tp_size + dp_idx * attn_tp_size * attn_cp_size + attn_tp_idx
            en = st + attn_tp_size * attn_cp_size
            ranks = list(range(st, en, attn_tp_size))    # stride = attn_tp_size
            group_ranks.append(ranks)
```

**8 GPU 例子的通信组**:
```
TP group (全局):    [g0, g1, g2, g3, g4, g5, g6, g7]
ATTN_CP groups:     [g0, g4]  [g1, g5]  [g2, g6]  [g3, g7]
ATTN_TP groups:     [g0, g1, g2, g3]  [g4, g5, g6, g7]
```

### 4.5 请求接收与三层广播

> 文件：`scheduler.py:1217-1298`

**规则**：只有 `(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0)` 拥有 ZMQ socket (line 424)。

```python
if self.pp_rank == 0 and self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
    self.recv_from_tokenizer = get_zmq_socket(...)    # 有 ZMQ
else:
    self.recv_from_tokenizer = None                    # 没有！
```

**请求广播的三层传播** (line 1269-1298):

```python
# enable_dp_attention 模式下 (CP 需要):
if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
    work_reqs = [从 ZMQ 收到的请求]
else:
    work_reqs = None

# Step 1: TP 组内广播 (attn_tp_rank 0 → 1,2,3)
if self.attn_tp_size != 1:
    work_reqs = broadcast_pyobj(work_reqs, ..., self.attn_tp_cpu_group, ...)

# Step 2: CP 组间广播 (attn_cp_rank 0 → 1)
if self.attn_cp_size != 1:
    work_reqs = broadcast_pyobj(work_reqs, ..., self.attn_cp_cpu_group, ...)

# Step 3: 控制消息在全 TP 组广播
if self.tp_size != 1:
    control_reqs = broadcast_pyobj(control_reqs, ..., self.tp_cpu_group, ...)
```

**广播流程图**:
```
TokenizerManager ──ZMQ PUSH──► g0 (tp_rank=0, cp_rank=0)
                                │
                    TP broadcast│ (attn_tp_cpu_group)
                    ┌───────────┼───────────┬───────────┐
                    ▼           ▼           ▼           ▼
                   g0          g1          g2          g3      (cp_rank=0)
                    │           │           │           │
                    │     CP broadcast (attn_cp_cpu_group)
                    ▼           ▼           ▼           ▼
                   g4          g5          g6          g7      (cp_rank=1)

结果: 8 个 GPU 全部收到相同请求
```

### 4.6 序列切分方式

> 文件：`layers/attention/nsa/utils.py`

SGLang 支持两种 CP 切分模式:

#### Round-Robin 切分 (默认，line 60-88)

```python
def nsa_cp_round_robin_split_data(input_):
    """按 token_idx % cp_size 交错分配"""
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
    indices = torch.arange(cp_rank, len(input_), cp_size, device=input_.device)
    return input_[indices]
```

```
原始: token0, token1, token2, token3, token4, token5, token6, token7
         ↓       ↓       ↓       ↓       ↓       ↓       ↓       ↓
cp=0: token0,         token2,         token4,         token6
cp=1:         token1,         token3,         token5,         token7
```

#### In-Seq-Split 切分 (line 335-398)

```
按连续块切分 (causal mask 优化):
  原始: [block0, block1, block2, block3, block4, block5, block6, block7]
  cp=4:
    cp_rank=0: [block0, block7]    ← 首+尾配对
    cp_rank=1: [block1, block6]
    cp_rank=2: [block2, block5]
    cp_rank=3: [block3, block4]
```

### 4.7 前向计算中的 CP 通信

> 文件：`layers/communicator_nsa_cp.py`

每层 Transformer 的计算变为：

```
输入 hidden_states (完整序列)
    │
    ▼ reduce_scatter (切分到各 CP rank)
    每个 CP rank 持有 1/cp_size 的 token
    │
    ▼ Attention:
    │  Q: 各 CP rank 只算自己那部分
    │  KV: all_gather 收集所有 CP rank 的 KV
    │  然后用自己的 Q 和完整 KV 做 Attention
    │
    ▼ MLP/MoE: 各 CP rank 独立计算自己的 token
    │
    ▼ all_gather (收集所有 CP rank 的结果)
    输出完整 hidden_states
```

**关键 NCCL 通信原语** (`dp_attention.py:560-573`):

```python
# Attention 前: all_gather 收集所有 CP rank 的数据
def attn_cp_all_gather_into_tensor(output, input):
    return get_attention_cp_group().all_gather_into_tensor(output, input)

# MLP 后: reduce_scatter 将结果分回各 CP rank
def attn_cp_reduce_scatter_tensor(output, input):
    return get_attention_cp_group().reduce_scatter_tensor(output, input)
```

**在 `NSACPCommunicator` 中的使用** (communicator_nsa_cp.py:150-210):

```python
# Attention 输出后: attn_tp scattered → full
if nsa_use_prefill_cp(forward_batch):
    attn_cp_all_gather_into_tensor(hidden_states, local_hidden_states)  # line 160

# MLP 输入前: full → attn_tp scattered
if nsa_use_prefill_cp(forward_batch):
    attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)   # line 209
```

### 4.8 Prefill 完整流程示例

以 8 GPU, `attn_cp_size=2`, `attn_tp_size=4` 为例，处理 8192 token 的 prompt：

```
Step 1: 序列切分 (Round-Robin)
┌─────────────────────────────────────────────┐
│ 原始: 8192 tokens                            │
│ CP rank 0 (g0-g3): token 0,2,4,... → 4096个 │
│ CP rank 1 (g4-g7): token 1,3,5,... → 4096个 │
└─────────────────────────────────────────────┘

Step 2: 每层 Transformer Block
┌──────────────────────────────────────────────────────┐
│ Layer N:                                              │
│                                                      │
│ ① Input LayerNorm (各 CP rank 独立)                   │
│                                                      │
│ ② QKV 投影 + Attention                                │
│    Q: 每个 CP rank 只算自己 4096 token 的 Q            │
│    KV: all_gather 到完整 8192 tokens                  │
│         ┌─ CP rank 0 的 K,V (4096) ─┐                │
│    KV = │                           │ ← all_gather   │
│         └─ CP rank 1 的 K,V (4096) ─┘                │
│    各 CP rank 用自己的 Q 和完整 KV 做 Attention        │
│    (内部还有 4路 TP 的 head 切分)                      │
│                                                      │
│ ③ MLP/MoE                                            │
│    reduce_scatter → 每个 CP rank 处理自己的 tokens     │
│    MoE: 走 EP all-to-all (独立于 CP)                  │
│    all_gather → 完整 hidden_states                    │
└──────────────────────────────────────────────────────┘

Step 3: 最终输出
  all_gather → 完整 logits → 采样
```

### 4.9 Decode 阶段的行为

**Decode 时 CP 不起作用**。因为每步只生成 1 个 token，切分无意义。

代码证据 (`forward_batch_info.py:116-125`):
```python
def is_context_parallel_extend(self, ...):
    return (
        self == ForwardMode.EXTEND   # 只有 prefill
        or self == ForwardMode.MIXED
        ...
    )
    # ForwardMode.DECODE 不在这里 → decode 不走 CP 路径
```

```python
# nsa/utils.py:52-57
def can_nsa_prefill_cp_round_robin_split(forward_batch):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False    # Decode 直接返回 False
```

**所以 CP 是"prefill 专用"优化，decode 阶段各 CP rank 退化为普通 TP 行为。**

### 4.10 当前使用限制

> 文件：`server_args.py:1217-1243`

CP 当前主要为 **DeepSeek V3.2 的 NSA (Native Sparse Attention)** 设计：

```python
if self.enable_nsa_prefill_context_parallel:
    logger.warning("Context parallel feature is still under experiment.")

    if self.nsa_prefill_cp_mode == "in-seq-split":
        self.enable_dp_attention = True
        self.moe_dense_tp_size = 1
        self.moe_a2a_backend = "deepep"
        self.ep_size = self.tp_size
        self.kv_cache_dtype = "bf16"
    else:  # round-robin-split
        self.enable_dp_attention = True
        self.moe_dense_tp_size = 1
        assert self.dp_size == 1

    assert self.tp_size == 8  # 当前仅支持单机 8 卡
```

**约束总结**:

| 约束 | 原因 |
|------|------|
| `pp_size` 必须为 1 | PP 与 CP 的序列切分冲突 |
| `tp_size` 必须被 `attn_cp_size` 整除 | Rank 分组需要均匀 |
| 当前仅 `tp_size=8` (单机) | 跨机精度问题未解决 |
| 需要 `enable_dp_attention=True` | CP 复用 DP Attention 通信基础设施 |
| 仅 DeepSeek V3.2 NSA 验证 | 实验性功能 |

### 4.11 通信组全局拓扑

以 8 GPU, `attn_cp_size=2`, `attn_tp_size=4` 为例：

```
┌───────────────────────────────────────────────────────────┐
│                                                           │
│ TP group (全局 world_size=8):                              │
│   [g0, g1, g2, g3, g4, g5, g6, g7]                       │
│   用于: 权重加载分片、控制消息广播                           │
│                                                           │
│ ATTN_CP groups (cp_size=2, 共4组):                         │
│   [g0, g4]  [g1, g5]  [g2, g6]  [g3, g7]                 │
│   用于: all_gather KV, reduce_scatter hidden_states        │
│                                                           │
│ ATTN_TP groups (attn_tp_size=4, 共2组):                    │
│   [g0, g1, g2, g3]  [g4, g5, g6, g7]                     │
│   用于: Attention head 并行 AllReduce                       │
│                                                           │
│ ZMQ 通道: 只有 g0 (attn_tp_rank=0, attn_cp_rank=0)       │
│   g0 → TP broadcast → g1,g2,g3                           │
│   g0 → CP broadcast → g4                                 │
│   g4 → TP broadcast → g5,g6,g7                           │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 4.12 与弹性并行切换的关系

对于弹性并行切换项目，CP 带来的额外复杂性：

| 影响 | 说明 |
|------|------|
| **NCCL 组增多** | 除 TP group 外还有 ATTN_CP group、ATTN_TP group，切换时都需 suspend/resume |
| **显存布局** | CP 模式下 KV Cache 分片存储（每个 CP rank 只存 1/cp_size 的 KV），切换时需重新分配或迁移 |
| **Rank 映射** | 同一 GPU 可能从 `attn_cp_rank=0` 变为另一角色 |
| **仅影响 prefill** | Decode 阶段 CP 退化为普通 TP，切换复杂度在 decode 阶段不增加 |

---

## 附录: 关键文件与行号速查

| 主题 | 文件 | 行号 | 内容 |
|------|------|------|------|
| TokenizerManager 类 | `managers/tokenizer_manager.py` | 188 | 类定义 |
| generate_request | `managers/tokenizer_manager.py` | 490 | 请求入口 |
| _send_one_request | `managers/tokenizer_manager.py` | 1057 | ZMQ 发送 |
| _wait_one_response | `managers/tokenizer_manager.py` | 1100 | 等待结果 |
| handle_loop | `managers/tokenizer_manager.py` | 1467 | 结果接收循环 |
| _handle_batch_output | `managers/tokenizer_manager.py` | 1476 | 结果分发 |
| ReqState | `managers/tokenizer_manager.py` | 129 | 请求状态类 |
| TM init_ipc_channels | `managers/tokenizer_manager.py` | 321 | ZMQ 初始化 |
| Scheduler 类 | `managers/scheduler.py` | 253 | 类定义 |
| event_loop_normal | `managers/scheduler.py` | 1108 | 主事件循环 |
| event_loop_overlap | `managers/scheduler.py` | 1134 | 重叠事件循环 |
| recv_requests | `managers/scheduler.py` | 1217 | 请求接收+广播 |
| get_next_batch_to_run | `managers/scheduler.py` | 1875 | 调度决策 |
| get_new_batch_prefill | `managers/scheduler.py` | 1960 | Prefill batch 构建 |
| run_batch | `managers/scheduler.py` | 2278 | 执行推理 |
| process_batch_result | `managers/scheduler.py` | 2447 | 处理结果 |
| Sched init_ipc_channels | `managers/scheduler.py` | 420 | ZMQ 初始化 |
| run_scheduler_process | `managers/scheduler.py` | 3068 | 子进程入口 |
| DetokenizerManager | `managers/detokenizer_manager.py` | 74 | 类定义 |
| DM event_loop | `managers/detokenizer_manager.py` | 144 | 事件循环 |
| DM init_ipc_channels | `managers/detokenizer_manager.py` | 101 | ZMQ 初始化 |
| get_zmq_socket | `utils/common.py` | 1416 | ZMQ 工具函数 |
| config_socket | `utils/common.py` | 1479 | 缓冲区配置 |
| PortArgs.init_new | `server_args.py` | 5606 | IPC 地址生成 |
| attn_cp_size 定义 | `server_args.py` | 422 | 参数定义 |
| attn_cp_size 约束 | `server_args.py` | 2085-2093 | 参数校验 |
| CP 启用限制 | `server_args.py` | 1217-1243 | NSA CP 配置 |
| compute_dp_attention_world_info | `layers/dp_attention.py` | 230 | Rank 计算 |
| attn_cp_all_gather | `layers/dp_attention.py` | 572 | CP all_gather |
| attn_cp_reduce_scatter | `layers/dp_attention.py` | 560 | CP reduce_scatter |
| CP NCCL 组创建 | `distributed/parallel_state.py` | 1707-1738 | 通信组 |
| round-robin split | `layers/attention/nsa/utils.py` | 60-88 | 序列切分 |
| cp_all_gather_rerange | `layers/attention/nsa/utils.py` | 335-398 | 结果重排 |
| NSACPLayerCommunicator | `layers/communicator_nsa_cp.py` | 49 | CP 通信器 |
| attn_cp_rank 计算 | `entrypoints/engine.py` | 957 | Rank 分配 |
| 并行层次注释 | `entrypoints/engine.py` | 951-953 | 层次说明 |

<!-- END OF DOCUMENT -->
