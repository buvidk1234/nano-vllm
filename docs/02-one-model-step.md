# 02：模型前向计算与 token 采样

[目录](README.md) · [上一篇：单个请求的完整推理流程](01-one-request.md) · [下一篇：多请求调度与 KV Cache 管理](03-batching-and-cache.md)

输入“我爱”，计算下一个 token。下面按前向顺序给出公式、输入和输出；除特别说明外，数值跟踪“爱”这一行。

运算顺序对应 [qwen3.py](../nanovllm/models/qwen3.py)。

## 1. 示例设置

| token | `<pad>` | 我 | 爱 | 你 | 编程 | `<eos>` |
| --- | --- | --- | --- | --- | --- | --- |
| ID | 0 | 1 | 2 | 3 | 4 | 5 |

```text
input_ids = [1,2]
positions = [0,1]
```

| 参数 | 本例值 |
| --- | --- |
| Decoder 层数 | 1 |
| 隐藏维度 H | 4 |
| Q 头数 | 2 |
| KV 头数 | 2 |
| 每头维度 d | 2 |
| MLP 中间维度 | 4 |
| 词表大小 | 6 |

本例所有 Linear 都没有 bias，Norm 权重都为 1，手算忽略 epsilon，结果展示到四位小数。Embedding 与 LM Head 不共享权重。分词和权重均为教学设定。

Linear 统一按下面的公式计算，权重 W 的形状为 `[输出维度,输入维度]`：

$$
Y=XW^\mathsf{T}.
$$

## 2. Embedding 与输入 RMSNorm

Embedding 按 token ID 查表；RMSNorm 对每个 token 的向量分别归一化：

$$
\begin{aligned}
X&=E[\text{input\_ids}],\\
U_{t,i}&=\operatorname{RMSNorm}(X_t)_i
=\frac{X_{t,i}}{\sqrt{\frac{1}{H}\sum_{j=0}^{H-1}X_{t,j}^2+\epsilon}}w_i.
\end{aligned}
$$

t 是 token 行号，H 是每行的特征数；i、j 从 0 到 H−1。w 是 Norm 权重，epsilon 是防止分母为零的小量。

```text
输入 ID = [1,2]

查表得到 X = [[1,0,1,0],
              [1,2,1,2]]                       # 形状 [2,4]

归一化后 U ≈ [[1.4142,0,     1.4142,0     ],
              [0.6325,1.2649,0.6325,1.2649]]   # 形状 [2,4]
```

U 进入 Attention，原始 X 留作后面的残差相加。源码见 [Embedding](../nanovllm/layers/embed_head.py) 和 [RMSNorm](../nanovllm/layers/layernorm.py)。

## 3. 一次矩阵乘法得到 Q、K、V

将 Q、K、V 的权重按行拼成 Wqkv，一次 Linear 把每个 token 的 4 个数映射成 12 个数：

$$
P=[\,Q\quad K\quad V\,]
=UW_{qkv}^\mathsf{T}
=U\begin{bmatrix}W_q\\W_k\\W_v\end{bmatrix}^{\mathsf{T}}.
$$

```text
输入 U 的形状：[2,4]
权重 Wqkv 的形状：[12,4]
输出 P 的形状：[2,12]
```

本例权重将 `[a,b,c,d]` 映射成 Q=`[a,b,d,c]`、K=`[a,b,a,b]`、V=`[a,b,a,b]`。“爱”的输入输出为：

```text
输入：[0.6325,1.2649,0.6325,1.2649]

输出的前 4 个数（Q）：[0.6325,1.2649,1.2649,0.6325]
输出的中 4 个数（K）：[0.6325,1.2649,0.6325,1.2649]
输出的后 4 个数（V）：[0.6325,1.2649,0.6325,1.2649]
```

每组 4 个数再分成两个头，每头 2 个数。拆头只调整形状，不增加数的总量：

```text
Q 的形状：[2,2,2]
K 的形状：[2,2,2]
V 的形状：[2,2,2]
```

三个维度依次表示 2 个 token、2 个头、每头 2 个数。

本例 `qkv_bias=False`，源码接着对每个 Q/K 头做 RMSNorm。这里每头的平方均值恰好为 1，按本例约定，数值不变。

## 4. RoPE：给 Q、K 加入位置

下面 x、y 是单个头的输入、输出列向量，维度 d 为偶数；p 是位置，B 是配置 `rope_theta`。完整的旋转公式为：

$$
\begin{gathered}
y=\operatorname{RoPE}(x,p)=
\begin{bmatrix}
\cos\theta_0&\cdots&0&-\sin\theta_0&\cdots&0\\
\vdots&\ddots&\vdots&\vdots&\ddots&\vdots\\
0&\cdots&\cos\theta_{m-1}&0&\cdots&-\sin\theta_{m-1}\\
\sin\theta_0&\cdots&0&\cos\theta_0&\cdots&0\\
\vdots&\ddots&\vdots&\vdots&\ddots&\vdots\\
0&\cdots&\sin\theta_{m-1}&0&\cdots&\cos\theta_{m-1}
\end{bmatrix}x,\\[4pt]
\theta_j=\frac{p}{B^{2j/d}},\qquad
j=0,1,\ldots,m-1,\qquad m=\frac d2.
\end{gathered}
$$

这个 d×d 矩阵将前后两半的对应分量配对：`(j,j+d/2)`。例如 d=4 时，配对是 `(0,2)`、`(1,3)`，每对使用自己的角度；输出仍是 d 维。

若每头有 3 个数，无法全部两两配对；可以另行规定只旋转其中 2 维，剩余 1 维保持不变，但本仓库当前实现没有这个分支。

本例 d=2，只有 j=0，所以旋转角度为 p。“爱”的位置 p=1，输入输出为：

| 头 | 旋转前 | 旋转后 |
| --- | --- | --- |
| Q0 | `[0.6325,1.2649]` | `[-0.7227,1.2156]` |
| Q1 | `[1.2649,0.6325]` | `[0.1512,1.4061]` |
| K0 | `[0.6325,1.2649]` | `[-0.7227,1.2156]` |
| K1 | `[0.6325,1.2649]` | `[-0.7227,1.2156]` |

V 不旋转。“我”的位置 p=0，旋转不改变数值。Attention 将用到的 K/V 为：

| token | 每个 K 头 | 每个 V 头 |
| --- | --- | --- |
| 我 | `[1.4142,0]` | `[1.4142,0]` |
| 爱 | `[-0.7227,1.2156]` | `[0.6325,1.2649]` |

源码见 [RotaryEmbedding](../nanovllm/layers/rotary_embedding.py)。KV Cache 保存旋转后的 K 和未旋转的 V。

## 5. Attention：打分、softmax、加权求和

每个头先做 Attention，再拼接并做输出投影，完整公式为：

$$
\begin{aligned}
O_h&=\operatorname{softmax}\!\left(\frac{Q_hK_h^{\mathsf{T}}}{\sqrt d}+M\right)V_h,
\qquad h=0,1,\ldots,n-1,\\
A&=\operatorname{Concat}(O_0,\ldots,O_{n-1})W_o^{\mathsf{T}},\\
M_{t,s}&=\begin{cases}0,&s\le t,\\-\infty,&s>t.\end{cases}
\end{aligned}
$$

n 是头数，`Q_h`、`K_h`、`V_h` 是第 h 个头在所有 token 上的矩阵，softmax 沿每行计算。M 是因果掩码，t 是查询位置，s 是被查询的位置。

输入是上一节的 Q、K、V，本例 n=2、d=2、Wo 为单位矩阵。“爱”能查询“我”和“爱”，两个头的计算结果为：

| 查询头 | 对“我”“爱”的分数 | softmax 权重 | 输出 |
| --- | --- | --- | --- |
| Q0 | `[-0.7227,1.4142]` | `[0.1056,0.8944]` | `[0.7150,1.1314]` |
| Q1 | `[0.1512,1.1314]` | `[0.2729,0.7271]` | `[0.8458,0.9198]` |

“我”只能查询自己，未来的“爱”被因果掩码屏蔽。两个 token 都参与本轮主干计算。

```text
输入：头 0 = [0.7150,1.1314]，头 1 = [0.8458,0.9198]
输出：A ≈ [0.7150,1.1314,0.8458,0.9198]
```

源码由 [FlashAttention](../nanovllm/layers/attention.py) 完成打分、softmax 和加权求和。

## 6. 残差与 MLP

从 Attention 输出到最终隐藏向量的完整计算为：

$$
\begin{aligned}
r_1&=x+A,\\
a&=\operatorname{RMSNorm}(r_1),\\
m&=\Big[\operatorname{SiLU}(aW_{gate}^{\mathsf{T}})
\odot(aW_{up}^{\mathsf{T}})\Big]W_{down}^{\mathsf{T}},\\
r_2&=r_1+m,\\
h&=\operatorname{RMSNorm}(r_2),\\[2pt]
\operatorname{SiLU}(v)_i&=\frac{v_i}{1+e^{-v_i}}.
\end{aligned}
$$

`⊙` 表示对应位置相乘。下面的 `gate`、`up` 分别是门控投影和上投影的结果，`z = SiLU(gate) ⊙ up`。

```text
输入 x = [1,2,1,2]
输入 A ≈ [0.7150,1.1314,0.8458,0.9198]

输出 r1 ≈ [1.7150,3.1314,1.8458,2.9198]
输出 a  ≈ [0.6904,1.2607,0.7431,1.1755]
gate    ≈ [0.6904,1.2607,0.7431,0]
up      ≈ [1.2607,0.7431,1.1755,0]
z       ≈ [0.5798,0.7299,0.5920,0]
输出 m  ≈ [0.5798,0.7299,0.5920,0.5798]
输出 r2 ≈ [2.2947,3.8613,2.4377,3.4995]
输出 h  ≈ [0.7410,1.2468,0.7871,1.1300]
```

源码将 gate/up 合成一次 Linear，再由 [SiluAndMul](../nanovllm/layers/activation.py) 拆开计算。Decoder 层返回 `(m,r1)`，把残差相加延后到下一次 Norm；本例由最终 Norm 完成。

## 7. LM Head：得到词表分数

完整 Prefill 后，LM Head 取每条请求最后一个输入 token 的隐藏向量。本例取“爱”的 h，计算下一个 token 的 logits：

$$
\mathrm{logits}=hW_{lm}^\mathsf{T}.
$$

```text
输入 h：[0.7410,1.2468,0.7871,1.1300]       # 4 个数
权重 Wlm：形状 [6,4]
输出 logits：[0.7410,1.2468,0.7871,1.1300,1.9878,-0.7410]
```

这 6 个分数依次对应 `<pad>`、我、爱、你、编程、`<eos>`，下一步才转换成概率。源码见 [ParallelLMHead](../nanovllm/layers/embed_head.py)。

## 8. 温度与随机采样

温度缩放、softmax 和随机采样的完整公式为：

$$
\begin{aligned}
p_i&=\frac{\exp(\mathrm{logits}_i/\tau)}{\sum_{j=0}^{N-1}\exp(\mathrm{logits}_j/\tau)},\\
E_i&\sim\operatorname{Exp}(1),\qquad i=0,1,\ldots,N-1,\\
\mathrm{tokenId}&=\operatorname{argmax}_{0\le i<N}\frac{p_i}{E_i}.
\end{aligned}
$$

τ 是温度，N 是词表大小，E 是各候选独立生成的指数随机数，tokenId 是采样得到的 token 编号。输入是上一节的 6 个 logits，输出概率为：

| token | τ=1 | τ=0.5 |
| --- | ---: | ---: |
| `<pad>` | 11.25% | 5.21% |
| 我 | 18.66% | 14.34% |
| 爱 | 11.78% | 5.72% |
| 你 | 16.60% | 11.35% |
| 编程 | 39.15% | 63.11% |
| `<eos>` | 2.56% | 0.27% |

温度越低，概率越集中到高分候选。再按上式用 p/E 取最大值，得到采样 ID：

```text
输入 p ≈ [0.1125,0.1866,0.1178,0.1660,0.3915,0.0256]  # τ=1

若 E = [1,1,1,1,1,1]，输出 ID 4（编程）
若 E = [1,0.1,1,1,1,1]，输出 ID 1（我）
```

上面是两组示例随机数。实际 E 每次随机生成，所以输出可以不是概率最大的 token。源码见 [Sampler](../nanovllm/layers/sampler.py)。

## 9. 下一轮 Decode

假设本轮输出“编程”，下一轮的输入输出为：

```text
输入：input_ids=[4]，positions=[2]，以及每层已有的“我”“爱”的 KV
计算：只为新 token 计算 Q/K/V，用新 Q 查询历史及当前位置的 K/V
输出：再下一个 token 的 logits 和采样 ID，并留下位置 2 的 KV
```

旧位置的 K/V 直接复用。每层保存自己的 KV，不能跨层替代。下一篇继续说明多个请求的输入和缓存如何组织。

## 10. 权重加载与源码布局

前向前，加载器把文件中分开的权重写入本地合并矩阵的对应行：

| 文件中的权重 | 本地参数区间 |
| --- | --- |
| `q_proj.weight`，形状 `[4,4]` | `qkv_proj.weight[0:4]` |
| `k_proj.weight`，形状 `[4,4]` | `qkv_proj.weight[4:8]` |
| `v_proj.weight`，形状 `[4,4]` | `qkv_proj.weight[8:12]` |
| `gate_proj.weight`，形状 `[4,4]` | `gate_up_proj.weight[0:4]` |
| `up_proj.weight`，形状 `[4,4]` | `gate_up_proj.weight[4:8]` |

[load_model()](../nanovllm/utils/loader.py) 通过 `packed_modules_mapping` 和参数的 `weight_loader` 完成映射与复制。这些操作发生在初始化时；前向直接使用合并权重。第四篇继续用这套数值说明两卡分片。
