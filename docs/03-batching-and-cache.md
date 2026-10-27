# 03：多请求调度与 KV Cache 管理

[目录](README.md) · [上一篇：模型前向计算与 token 采样](02-one-model-step.md) · [下一篇：推理优化与张量并行](04-faster-inference.md)

每轮先选请求，再打包输入、执行模型，最后追加输出并释放结束请求的 KV 块。

## 1. 案例与流程

```python
prompt_a = [42] * 512 + [43] * 88  # 600 个 token
prompt_b = [42] * 512 + [44] * 88  # 前 512 个与 A 相同
prompt_c = [21, 22, 23]           # 3 个 token
```

单卡，块大小 256，KV 池四块 `[0,1,2,3]`，初始无缓存。每轮 Prefill 预算为 256 个新 token。A 生成 3 个 token，B、C 各生成 2 个，均设 `ignore_eos=True`。

入队时机由本例指定：A 最先入队，**第 3 步结束后加入 B，第 5 步结束后加入 C**。

| 步骤 | 执行 | 本轮输入 | 输出与结束后的变化 |
| --- | --- | --- | --- |
| 1 | A Prefill | 位置 0—255 | 缓存 256 个位置，尚无输出 |
| 2 | A Prefill | 位置 256—511 | 缓存进度增至 512，尚无输出 |
| 3 | A Prefill | 位置 512—599 | 得到 Y1；随后 B 入队 |
| 4 | B Prefill | 命中前 512，只输入位置 512—599 | 得到 Z1 |
| 5 | A、B Decode | Y1、Z1 | 得到 Y2、Z2；B 结束，随后 C 入队 |
| 6 | C Prefill | `[21,22,23]` | 得到 W1 |
| 7 | A、C Decode | Y2、W1 | 得到 Y3、W2；A、C 都结束 |

调度器优先执行能接纳的 Prefill，因此第 4、6 步暂停 A 的 Decode。分块 Prefill 完成整个 prompt 后才追加第一个输出。

## 2. KV 块如何分配、共享和释放

A 首次分配就取得覆盖全部 600 个位置的三块；分块 Prefill 只限制每轮计算量。B 命中 A 的前两个完整块，只需另分配一个尾块：

```text
A.block_table = [0,1,2]
B.block_table = [0,1,3]
引用计数       = [2,2,1,1]
空闲块         = []

B 结束后：引用计数变为 [1,1,1,0]，块 3 归还空闲队列
C 入队后：取得块 3，C.block_table = [3]
全部结束：空闲队列为 [2,1,0,3]
```

完整块计算完成后，登记包含前缀的链式哈希，并在复用时比对 token 列表；遇到首个未命中块就停止。最后一个逻辑块保留前向，用于取得最终隐藏状态和 logits。

共享块增加引用计数，计数降到 0 才归还。空闲块仍可保留有效缓存和哈希；改存新内容时重置元数据、覆盖数据，Attention 只读取有效长度。

源码：[BlockManager](../nanovllm/engine/block_manager.py)。

<a id="attention-inputs"></a>

## 3. 实际送进模型的内容

第 4 步 B 只输入 88 个新 token，产生 88 行新 Q/K/V；Attention 读取历史 512 个位置和新增 88 个位置，共 600 个：

```text
input_ids      = [44 × 88]        # 重复 88 次
positions      = [512,...,599]    # 逻辑位置，用于 RoPE
cu_seqlens_q   = [0,88]
cu_seqlens_k   = [0,600]
block_tables   = [[0,1,3]]        # 读取 KV 的物理块
slot_mapping   = [768,...,855]    # 写入新增 KV 的物理槽位
```

位置 p 对应的写入槽位：

```python
slot = seq.block_table[p // block_size] * block_size + p % block_size
```

例如 B 的位置 512 写入 `3×256+0=768`。本轮第 0 行查询可见位置 0—512，最后一行可见位置 0—599；LM Head 取本轮隐藏状态的第 87 行，得到 Z1。

Decode 时，每个请求只输入最新一个 token。`context_lens` 包含本轮写入的新位置：

| 字段 | 第 5 步：A、B | 第 7 步：A、C |
| --- | --- | --- |
| `input_ids` | `[Y1,Z1]` | `[Y2,W1]` |
| `positions` | `[600,600]` | `[601,3]` |
| `context_lens` | `[601,601]` | `[602,4]` |
| `slot_mapping` | `[600,856]` | `[601,771]` |
| `block_tables` | `[[0,1,2],[0,1,3]]` | `[[0,1,2],[3,-1,-1]]` |

块表补 -1 对齐行长，实际读取范围由有效长度决定。模型显式接收 `input_ids,positions`，其余元数据通过 [Context](../nanovllm/utils/context.py) 传递；各层先写新 KV，再执行 Attention。

源码：[ModelRunner](../nanovllm/engine/model_runner.py)、[Attention](../nanovllm/layers/attention.py)。

## 4. 入队变化与抢占

- **A、B 同轮 Prefill：** 预算改为 1200、池至少六块。分配 B 时 A 的前缀尚未登记，因此两条请求各计算完整 600 个 token。
- **C 提前和 B 入队：** 预算 256、池至少五块。第 4 步一起处理 B 的 88 个和 C 的 3 个 token，下一轮 A、B、C 一起 Decode。下一篇沿用此变体。
- **C 改为 200 个输入：** B 用掉 88，剩余预算 168，C 等下一轮。源码只允许本轮第一条 Prefill 请求分块。

Decode 扩块时若无空闲块，调度器从 running 队尾抢占请求：保留 token 历史，释放块引用，重置缓存进度，放回 waiting，之后重新 Prefill 恢复。

例如将 A、B 的输出都改为 170：生成 169 个输出后，总长为 769，最新位置 768 需要新块。抢占 B 后，A 取得块 3，生成 Y170 并结束；B 恢复时命中前 512 个位置，将余下 257 个位置分成 256 和 1 两轮重算，再生成 Z170。此前的 169 个输出保留。

源码：[Scheduler](../nanovllm/engine/scheduler.py)。

<a id="capacity"></a>

## 5. KV 池与显存容量

GPU KV 池在初始化时一次分配，归还块只改变使用权。CPU `Block` 保存块号、引用计数、哈希和 token 列表；每层 K/V 是池的视图。

```text
KV 池形状：[2,层数,物理块数,256,本地 KV 头数,head_dim]
```

以 32 层、8 个 KV 头、每头 128 维、FP16 为例：

```text
单卡每块 = 2 × 32 × 256 × 8 × 128 × 2 bytes = 32 MiB
TP=2 时每块每卡 = 16 MiB

假设每卡预算 20 GiB、设备已用 10 GiB、PyTorch 峰值 9 GiB、当前分配 7 GiB：
KV 可用量 = 20 - 10 - (9-7) = 8 GiB
每卡块数 = 8 GiB / 16 MiB = 512
逻辑 token 槽位数 = 512 × 256 = 131072
```

`峰值-当前分配` 用于预留已释放的临时峰值。源码见 [allocate_kv_cache()](../nanovllm/engine/model_runner.py)。
