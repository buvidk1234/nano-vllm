# 配置、源码索引与实现边界

[目录](README.md) · [请求 A](01-one-request.md) · [模型算术](02-one-model-step.md) · [并发与缓存](03-batching-and-cache.md) · [图与多卡](04-faster-inference.md)

这页汇集 A、B、C 案例涉及的配置、源码入口和实现边界。每个参数或问题都对应正文中的一次状态变化，完整推导见各篇教程。

## 1. 配置参数及其对案例的影响

引擎参数来自 [`Config`](../nanovllm/config.py)。`LLM(model, **kwargs)` 只接收其中的已知字段，未知名称会被过滤而不报错；例如拼错 `max_num_batched_tokens` 后，A 可能仍按默认预算一次处理完。

| 参数 | 源码默认值 | 在案例中怎样使用 |
| --- | --- | --- |
| `model` | 必填 | 指向本地 Qwen3 模型目录，含配置、tokenizer、safetensors |
| `max_num_batched_tokens` | 16384 | 每轮 Prefill 的新输入 token 总量上限；01 取 1024，A 一次处理 600；03 取 256，A 分成 256+256+88；Decode 分支不检查此预算 |
| `max_num_seqs` | 512 | 每轮最多选取的请求数，教程取 16；不是整个等待队列或全局活跃请求数的上限 |
| `max_model_len` | 4096 | 单条序列的长度配置，按输入与生成内容的总长度理解；实际取它与模型 `max_position_embeddings` 的较小值，入口未完整校验生成总长 |
| `gpu_memory_utilization` | 0.9 | 影响可分配 KV 块数，计算见[四块池的容量算例](03-batching-and-cache.md#capacity) |
| `tensor_parallel_size` | 1 | 04 改为 2；源码允许 1—8，还要满足设备与模型分片约束 |
| `enforce_eager` | False | 前三篇取 True；04 的图重放变体取 False，局部编译始终存在 |
| `kvcache_block_size` | 256 | 全部系统案例保持 256；源码只检查能被 256 整除，未检查正值 |
| `num_kvcache_blocks` | -1 | 初始化时会重新估算；不能靠传 4 强制复现四块教学池 |
| `hf_config` | None | 用 Hugging Face `AutoConfig` 从模型目录读取，包含层数、隐藏维度、`max_position_embeddings` 等；位置上限的含义见[初始化说明](01-one-request.md#model-position-limit) |
| `eos` | -1 | 引擎加载 tokenizer 后更新，教程用 `ignore_eos=True` 控制输出数 |

每请求参数来自 [`SamplingParams`](../nanovllm/sampling_params.py)，不能当作引擎配置传入：

| 参数 | 默认值 | 例子与边界 |
| --- | --- | --- |
| `temperature` | 1.0 | 请求案例取 0.6；第二篇用 1 和 0.5 比较概率，必须大于 `1e-10` |
| `max_tokens` | 64 | A 取 3，B/C 取 2；源码未完整校验正整数 |
| `ignore_eos` | False | 案例取 True，保证 A 生成三个输出；真实使用可按 EOS 停止 |

例如 A 的 `max_tokens=128` 表示需要 128 个输出，原始 600-token prompt 的 Prefill 长度仍为 600。`max_num_batched_tokens=256` 则把这些输入分成 256、256、88 三块，两者分别控制输出目标和每轮输入预算。

## 2. 案例中容易混淆的现象

| 具体问题 | 原因与条件 | 源码入口 / 推导 |
| --- | --- | --- |
| 同样长的 A，为什么有时一轮 Prefill 就完成 | 预算 1024 能容纳全部 600 个输入，预算 256 则分成三轮 | `Scheduler.schedule()`；[第三篇](03-batching-and-cache.md) |
| A 在 RUNNING，却仍执行 Prefill | 状态在最后一块 Prefill 前就改变；批阶段与传输标志另有用途 | `schedule()`、`Sequence.is_prefill`；[第一篇](01-one-request.md) |
| A 总长 601，cached 只有 600 | Y1 刚生成，还没有输入模型计算 KV | `postprocess()`；[第一篇](01-one-request.md) |
| A 还没入队，为什么 KV 池已经占用显存 | 初始化已创建整个 GPU 池，入队后才分配池内块引用 | `allocate_kv_cache()`；[池的分配时机](01-one-request.md#kv-pool) |
| 预热输入为什么长 1024，而 A 只有 600 | 预热由预算和长度配置计算规模，使用临时 `[0] * seq_len`，与 A 的 prompt 无关 | `warmup_model()`；[预热长度计算](01-one-request.md#warmup) |
| 写 KV、登记块、更新 cached 是否同一操作 | 分别写 GPU 浮点张量、建立 CPU 哈希索引、更新请求的 CPU 进度计数 | `run()`、`hash_blocks()`、`postprocess()`；[登记示例](01-one-request.md#cache-registration) |
| A 只算 256，却占了三个块 | 分块限制计算量，首次分配仍覆盖全部已知 token | `BlockManager.allocate()`；[第三篇](03-batching-and-cache.md) |
| B 与 A 前缀相同却没有命中 | B 接纳时 A 是否已前向并登记完整块；是否只剩被排除的尾块 | `can_allocate()`、`hash_blocks()`；[第三篇](03-batching-and-cache.md) |
| B 已命中 512，Attention 却读 600 | 88 个新 Q 仍要查询旧、新 K；命中省去历史投影与各层重计算 | `prepare_prefill()`；[B 的输入](03-batching-and-cache.md#attention-inputs) |
| A 的输出被 B、C 打断 | 有 Prefill 被选中时，本步就不执行 Decode | `Scheduler.schedule()`；[七步时间线](03-batching-and-cache.md) |
| C 没有分到块 3 | GPU 池更多时，free 队列前面可能还有从未使用的块 | `free_block_ids`；[第三篇](03-batching-and-cache.md) |
| B 结束后显存没有明显下降 | 只是归还块引用，GPU 预分配池仍存在 | `deallocate()`、`allocate_kv_cache()`；[第三篇](03-batching-and-cache.md) |
| 抢占后 B 又 Prefill，但保留 169 个输出 | 重算的是已有 token 的 KV；原始 prompt 边界仍为 600 | `preempt()`、`num_completion_tokens`；[压力变体](03-batching-and-cache.md) |
| A 在前三步前两个分块没有输出 | 分块未完成时，采样值被后处理丢弃 | `Scheduler.postprocess()`；[第三篇](03-batching-and-cache.md) |
| 输出没选概率最大的“编程” | `argmax` 比较的是 `p/E`；E 是独立随机变量 | `Sampler.forward()`；[第二篇](02-one-model-step.md) |
| `max_num_seqs=12` 时 batch 9 出错 | 图档位只有 1/2/4/8，没有能容纳 9 的图 | `capture_cudagraph()`、`run_model()`；[第四篇](04-faster-inference.md) |
| 开 eager 后仍出现编译或自定义内核 | 该开关只关闭执行器的 CUDA Graph | `@torch.compile`、`Attention.forward()`；[第四篇](04-faster-inference.md) |
| 两卡结果需要相加还是拼接 | Attention/down 的输出贡献相加；词表 logits 拼接 | `RowParallelLinear`、`ParallelLMHead`；[第四篇](04-faster-inference.md) |

例如 A 等 B Prefill 的时间属于 A 的 token 间隔；整个 `generate()` 完成时间包含 Prefill。它们与纯 Decode 内核耗时表示不同范围，具体计算见[第四篇的指标表](04-faster-inference.md#metrics)。

## 3. 请求执行流程与源码索引

下表把案例中的动作映射到对应文件和函数。

| 案例中的动作 | 文件与符号 |
| --- | --- |
| 创建引擎、提交 A、循环生成、收集结果 | [`llm_engine.py`](../nanovllm/engine/llm_engine.py)：`LLMEngine.__init__`、`add_request`、`step`、`generate`、`exit`；公开类见 [`llm.py`](../nanovllm/llm.py) |
| 保存 A 的历史和 KV 进度，精简 worker 数据 | [`sequence.py`](../nanovllm/engine/sequence.py)：`Sequence`、`__getstate__`、`__setstate__` |
| 选择 A/B/C，更新输出、抢占恢复 | [`scheduler.py`](../nanovllm/engine/scheduler.py)：`schedule`、`postprocess`、`preempt` |
| B 引用前两个块、释放尾块、登记哈希 | [`block_manager.py`](../nanovllm/engine/block_manager.py)：`can_allocate`、`allocate`、`hash_blocks`、`may_append`、`deallocate` |
| 加载模型前后的初始化和 KV 容量估算 | [`model_runner.py`](../nanovllm/engine/model_runner.py)：`__init__`、`warmup_model`、`allocate_kv_cache` |
| 将 B 的 88 个 token 或三个 Decode token 打包 | [`model_runner.py`](../nanovllm/engine/model_runner.py)：`prepare_prefill`、`prepare_decode`、`prepare_block_tables`、`prepare_sample` |
| 选择普通调用或图重放，将采样结果带回 CPU | [`model_runner.py`](../nanovllm/engine/model_runner.py)：`run`、`run_model`、`capture_cudagraph` |
| 把当前命令发给 rank 1 | [`model_runner.py`](../nanovllm/engine/model_runner.py)：`call`、`write_shm`、`read_shm`、`loop` |
| 每层读取本轮槽位、长度和块表 | [`context.py`](../nanovllm/utils/context.py)：`Context`、`set_context`、`get_context`、`reset_context` |
| 把“爱”逐层变成 h，再得到 logits | [`qwen3.py`](../nanovllm/models/qwen3.py)：`Qwen3Attention`、`Qwen3MLP`、`Qwen3DecoderLayer`、`Qwen3Model`、`Qwen3ForCausalLM.compute_logits` |
| 先写当前 KV，再查询有效历史 | [`attention.py`](../nanovllm/layers/attention.py)：`store_kvcache_kernel`、`store_kvcache`、`Attention.forward` |
| 合并 QKV/gate-up，分片并归约 | [`linear.py`](../nanovllm/layers/linear.py)：`QKVParallelLinear`、`MergedColumnParallelLinear`、`ColumnParallelLinear`、`RowParallelLinear` |
| 查词向量、选 Prefill 末行、收集词表分数 | [`embed_head.py`](../nanovllm/layers/embed_head.py)：`VocabParallelEmbedding`、`ParallelLMHead` |
| r1/r2 的 Norm、位置 1 的旋转、MLP 门控 | [`layernorm.py`](../nanovllm/layers/layernorm.py)：`RMSNorm`；[`rotary_embedding.py`](../nanovllm/layers/rotary_embedding.py)：`RotaryEmbedding`；[`activation.py`](../nanovllm/layers/activation.py)：`SiluAndMul` |
| 用温度和指数随机数选输出 | [`sampler.py`](../nanovllm/layers/sampler.py)：`Sampler.forward` |
| 将文件中的 K 权重放进合并参数的 K 区间 | [`loader.py`](../nanovllm/utils/loader.py)：`load_model`；`Qwen3ForCausalLM.packed_modules_mapping`；各参数的 `weight_loader` |

## 4. 输入限制与资源约束

除了前面给出的正常路径，源码还存在以下输入限制和资源约束。

| 条件变化 | 当前源码行为 |
| --- | --- |
| 把 A 换成空 ID 列表 | `Sequence` 直接访问 `token_ids[-1]`；输入必须非空 |
| 向 `generate()` 传 A/B 和一个参数列表 | 使用 `zip()`，未严格检查等长；较长列表的剩余元素不会被处理 |
| 把 A 输出改为 0 或很长 | 未完整校验正输出数和 prompt+output 总长；例如 600 个输入加 500 个输出会超过案例设定的 1024 上限 |
| 给 A 开 greedy、top-p 或重复惩罚 | 没有这些采样分支；temperature=0 也不合法 |
| 希望每次 `step()` 都返回 Y1 等增量 | 只返回本步完成请求的结果；`generate()` 返回整批字典列表，无内置 HTTP/流式服务 |
| 依据 `generate()` 的 `list[str]` 注解理解返回值 | 实际返回 `text`、`token_ids` 字段组成的字典列表；其中 decode 未显式跳过特殊 token，文本可能包含特殊标记 |
| 把 A 换成任意聊天字符串 | 引擎只 encode，不自动套聊天模板；用法参考 [`example.py`](../example.py) |
| 降低池容量，期待大请求自动等待到可运行 | 接纳按队首检查，失败就停止扫描；若 Decode 最终也选不出请求会断言，超过池总容量的请求无法靠等待解决 |
| 把 max_num_seqs 当作全局并发上限或公平保证 | 只约束本步批量；已有 running 数量可更大，Decode 前部请求会继续排在前部 |
| 任意改变图 batch 上限或让序列超长 | 档位未覆盖任意上限；静态块表宽度由 max_model_len 决定，没有通用动态扩容 |
| 两卡的可用显存差别很大 | 每卡独立估算 KV 块数，未对全局容量取最小值；rank 0 的块号可能超出 worker 容量 |
| TP 时把块大小改为 512 | spawn worker 的 `Sequence.block_size` 仍为 256，序列序列化不携带此配置；两侧可能算出不同的尾块偏移 |
| 在同一主机同时启动多个引擎进程组 | NCCL 地址固定为 `tcp://localhost:2333`，多卡共享内存名固定为 `nanovllm`，独立实例之间可能发生资源冲突 |
| 在单个 CPU 命令内传非常大的输入列表 | 共享内存固定 1 MiB，前 4 字节存长度；没有通用的大载荷分片协议 |
| 在同一进程并发调用多个执行步骤 | `Context` 是模块全局变量，当前路径没有并发隔离 |
| 加载其他架构或任意 checkpoint | Runner 固定构造本地 Qwen3；loader 遍历顶层 safetensors，没有通用模型适配和完整缺失参数检查 |
| 显式调用 `llm.exit()` 后进程再退出 | 清理函数不是幂等的，而 atexit 已注册同一函数，可能重复访问已删除的 runner |

例如 TP=2 只通过了 `Config` 的范围检查，还不代表模型能运行：Q/KV 头数、词表和相关投影维度都要满足分片要求。代码没有“KV 头不足时自动复制到更多卡”的通用路径。

## 5. 相关原理与延伸资料

| 从本案例产生的问题 | 阅读入口 |
| --- | --- |
| A/B 的逻辑块如何减少重复存储与碎片 | [PagedAttention / vLLM 论文](https://arxiv.org/abs/2309.06180) |
| B 的 88 个 Q 和 600 个 K 如何做因果 Attention | [FlashAttention 接口与说明](https://github.com/Dao-AILab/flash-attention) |
| 为什么头分片后用 Row Parallel 求和 | [Megatron-LM 论文](https://arxiv.org/abs/1909.08053) |
| 图为什么能换 Y1，却要求固定缓冲区地址 | [PyTorch CUDA Graph 说明](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) |
| 更完整的服务如何取消 B、流式返回 A、混合调度 | [vLLM 仓库](https://github.com/vllm-project/vllm)，先固定版本再沿一个请求追踪 |

这些资料解释原理和底层接口；本教程对 Nano-vLLM 的判断以本仓库源码为准，不把其他项目的能力直接归到这里。
