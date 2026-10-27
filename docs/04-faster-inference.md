# 04：推理优化与张量并行

[目录](README.md) · [上一篇：多请求调度与 KV Cache 管理](03-batching-and-cache.md) · [配置、源码索引与实现边界](reference.md)

本篇沿用第三篇中 **C 与 B 同时入队、KV 池至少五块**的变体。A、B、C 都已得到第一个输出，接下来一起 Decode，观察 CUDA Graph 和双卡 TP 如何执行这一轮。

## 1. 本轮输入与优化方式

A、B、C 的块表分别为 `[0,1,2]`、`[0,1,3]`、`[4]`，本轮输入为：

```text
input_ids      = [Y1,Z1,W1]
positions      = [600,600,3]
context_lens   = [601,601,4]
slot_mapping   = [600,856,1027]
block_tables   = [[0,1,2],[0,1,3],[4,-1,-1]]
```

模型为三个新 token 写入 KV，经过各层得到 `[3,H]` 隐藏状态，再由 LM Head 得到 `[3,V]` logits，采样出 Y2、Z2、W2。H 是隐藏维度，V 是词表大小。

| 优化 | 本仓库怎样使用 | 主要减少什么 |
| --- | --- | --- |
| 合并投影 | Q/K/V 合成一次 Linear，gate/up 合成一次 Linear | 独立矩阵乘调用 |
| 局部 `torch.compile` | 编译 RMSNorm、RoPE、SiLU 门控和采样 | 通过算子融合等方式减少内核启动和中间读写 |
| FlashAttention | Prefill 用 `flash_attn_varlen_func`，Decode 用 `flash_attn_with_kvcache` | Attention 中间分数、概率矩阵的显存读写 |
| CUDA Graph | 捕获模型前向，后续重放 | CPU 反复调用和提交 GPU 操作的开销 |
| 张量并行（TP） | 每卡保存、计算部分权重 | 每卡的权重存储和计算量，同时增加通信 |

这些局部函数各自使用 `@torch.compile`，执行器没有对整个模型调用 `torch.compile(model)`。`enforce_eager=True` 只关闭执行器的 CUDA Graph；局部编译、FlashAttention 和写 KV 的 Triton 内核仍会使用。

FlashAttention 仍读取全部有效上下文：本轮 A、B、C 分别查询 601、601、4 个位置。每层先由 Triton 写入新 KV，再执行 Attention。

源码：[ModelRunner](../nanovllm/engine/model_runner.py)、[Attention](../nanovllm/layers/attention.py)。

## 2. CUDA Graph：三条请求使用四行的图

本例设置 `enforce_eager=False`、`max_num_seqs=16`、`max_model_len=1024`。初始化时 `capture_cudagraph()` 为 batch 大小 `1,2,4,8,16` 分别捕获图；运行时选择能容纳真实 batch 的最小档位，因此三条请求使用 batch=4：

```text
input_ids      = [Y1,Z1,W1,旧值]
positions      = [600,600,3,旧值]
context_lens   = [601,601,4,0]
slot_mapping   = [600,856,1027,-1]
```

每轮先更新固定地址缓冲区里的 token、位置、块表等数据，再调用 `graph.replay()`。GPU 操作沿用捕获时的地址，缓冲区内容随请求进度变化；KV 池也是初始化时分配好的。

第四行是填充，仍会做投影等计算。它的 `slot_mapping=-1`，写缓存内核跳过这一行；有效长度为 0，最终也舍弃这一行的输出。token 和 position 的旧值无需每轮清零。

```text
复制本轮数据 → 重放图，得到 [4,H] → 取前三行 [3,H]
             → 图外 LM Head [3,V] → 图外采样 [Y2,Z2,W2]
```

捕获范围是 `self.model()`：Embedding、各层前向、KV 写入和最终 Norm。LM Head、采样、将结果转回 CPU 的 `.tolist()`、调度和后处理都在图外。

Prefill、`enforce_eager=True` 或 Decode batch 大于 512 时走普通执行。其余 batch 必须有可用图档位；例如 `max_num_seqs=12` 只生成 `1,2,4,8`，batch=9 会找不到图，当前源码没有自动回退。

## 3. 双卡 TP：分片计算与结果合并

设置 `tensor_parallel_size=2` 后，两卡共同处理整批 A、B、C，每卡负责各层的一部分权重和特征。

分片形状继续用第二篇的小模型说明：隐藏维度 4，Q/KV 各两头，每头两维，MLP 中间维度 4，词表大小 6。下面的数值仍是第二篇“爱”的计算结果。

| 权重（二维矩阵按 `[out,in]` 记） | 完整形状 | 每卡保存的形状 | 切分方式 |
| --- | --- | --- | --- |
| Q 投影权重 Wq | `[4,4]` | `[2,4]` | 按输出维，分配不同的头 |
| K 投影权重 Wk | `[4,4]` | `[2,4]` | 按输出维，分配不同的头 |
| V 投影权重 Wv | `[4,4]` | `[2,4]` | 按输出维，分配不同的头 |
| 合并投影权重 Wqkv | `[12,4]` | `[6,4]` | 分别切分 Wq、Wk、Wv，再拼接 |
| Attention 输出投影权重 Wo | `[4,4]` | `[4,2]` | 按输入维 |
| gate 投影权重 | `[4,4]` | `[2,4]` | 按中间维度，取前半或后半 |
| up 投影权重 | `[4,4]` | `[2,4]` | 与 gate 取相同的中间维度范围 |
| 合并 gate/up 投影权重 | `[8,4]` | `[4,4]` | 分别切分 gate、up 权重，再拼接 |
| down 投影权重 | `[4,4]` | `[4,2]` | 按输入维 |
| Embedding 权重 | `[6,4]` | `[3,4]` | 按词表范围 |
| LM Head 权重 | `[6,4]` | `[3,4]` | 按词表范围 |
| 隐藏维 Norm 权重 | `[4]` | `[4]` | 每卡复制一份 |

按输出维切分称为列并行（`ColumnParallelLinear`），各卡计算不同的输出特征；按输入维切分称为行并行（`RowParallelLinear`），各卡计算完整输出的一部分贡献，再求和。

**Attention 路径：** QKV 列并行 → Q/K 做 RoPE → 本地 Attention → Wo 行并行 → AllReduce。

Wo 接收本地 Attention 的头输出，将它投影到完整隐藏维度；**AllReduce 对各卡的投影结果求和，并把总和交给每张卡**。例如第二篇的 Wo 是单位矩阵：

```text
rank 0 的头输出：[0.7150,1.1314] → Wo 部分结果 [0.7150,1.1314,0,0]
rank 1 的头输出：[0.8458,0.9198] → Wo 部分结果 [0,0,0.8458,0.9198]
AllReduce 求和：                            [0.7150,1.1314,0.8458,0.9198]
```

**MLP 路径：** gate/up 列并行 → 本地 SiLU 门控 → down 行并行 → AllReduce。down 接收本地门控结果，投影后通过 AllReduce 合并。每层有 Wo、down 两次 AllReduce，32 层就是 64 次。

**Embedding 与 LM Head。** rank 0 保存词表 ID 0—2，rank 1 保存 ID 3—5。输入 ID 4 时，rank 0 查表结果置零，rank 1 查到 `E[4]=[1,1,0,0]`；AllReduce 后两卡都得到这个完整词向量。

LM Head 各卡计算自己的词表分数，由 Gather 收集到 rank 0，再按词表顺序拼接：

```text
rank 0 的 logits：[0.7410,1.2468,0.7871]
rank 1 的 logits：[1.1300,1.9878,-0.7410]
rank 0 拼接结果： [0.7410,1.2468,0.7871,1.1300,1.9878,-0.7410]
```

所有卡都参与 LM Head 和 Gather，只有 rank 0 采样。使用 CUDA Graph 时，模型内部的 AllReduce 也在图中，LM Head 和 Gather 仍在图外。

TP 减少每卡计算，但增加通信，小模型或小 batch 未必更快。

源码：[Linear](../nanovllm/layers/linear.py)、[Embedding 与 LM Head](../nanovllm/layers/embed_head.py)。

## 4. 两个进程怎样配合

`LLMEngine` 在主进程运行 rank 0，通过 spawn 创建 rank 1 worker，每个进程使用一张 GPU。rank 是通信组内的进程编号，rank 0 管理请求队列和完整状态。

每步 rank 0 把执行命令和序列数据写入 CPU 共享内存，再用 Event 唤醒 worker。Decode 的 token 内容只传最新一个 token，另带长度、KV 进度和块表；Prefill 则传完整 token 列表，由各进程截取本轮要处理的部分。

各卡按相同的块号和槽位编号，在自己的显存池中保存各自负责的 KV 头。模型计算中的 AllReduce、Gather 通过 NCCL 完成。最后 rank 0 采样、追加输出、判断结束并释放块引用，进入下一轮。

多卡容量估算、非默认块大小等实现限制见[参考页](reference.md)。源码：[LLMEngine](../nanovllm/engine/llm_engine.py)、[ModelRunner](../nanovllm/engine/model_runner.py)、[Sequence](../nanovllm/engine/sequence.py)。

<a id="metrics"></a>

## 5. 延迟与吞吐怎样计算

以下是示意时间线，计时从引擎初始化完成后开始，按引擎追加输出的时刻记录：

| 时间 | 发生的事 |
| --- | --- |
| 0 ms | A 入队 |
| 30 ms | A 完成三块 Prefill，得到 Y1；B、C 入队 |
| 40 ms | B、C 完成 Prefill，得到 Z1、W1 |
| 45 ms | 一起 Decode，得到 Y2、Z2、W2；B、C 结束 |
| 50 ms | A 再 Decode，得到 Y3，全部结束 |

| 指标 | 定义与结果 |
| --- | --- |
| 首 token 延迟（TTFT） | A 入队到 Y1，`30−0=30 ms` |
| token 间隔 | A 的 Y1→Y2 为 15 ms，Y2→Y3 为 5 ms |
| 平均每个后续 token 用时（TPOT） | A 首个输出之后的平均间隔，`(50−30)/(3−1)=10 ms` |
| 端到端延迟 | A 入队到最后一个输出，`50−0=50 ms` |
| 输出吞吐 | A/B/C 共输出 7 个 token，`7/0.05=140 token/s` |

A 的第一个 token 间隔包含等待 B、C Prefill 的时间。`generate()` 在整批结束后才返回；测量 TTFT 和 token 间隔需要在引擎内部记录各次输出的时刻。

仓库 [bench.py](../bench.py) 在初始化和一次短请求预热后计时，用输出 token 总数除以整次 `generate()` 耗时，包含 Prefill 和 Decode。比较优化前后时，应保持硬件、模型、输入输出长度、并发数和缓存状态一致；上表数字仅用于说明口径。
