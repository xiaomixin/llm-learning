# Week 6 深度知识伴读 — Multi-Head Attention、Positional Encoding 与 Encoder Block

> 本周把 Week 5 的 `SingleHeadSelfAttention` 升级成 `MultiHeadAttention`，补上 Positional Encoding 和 Position-wise FFN，用 LayerNorm + 残差缝成标准 Encoder Block。学完这一周，你已经能从零写出一个 "长得和原论文差不多" 的 Transformer —— 剩下 Week 7 的事情只是堆层、接分类头、跑 warmup。

---

## 1. 本周要回答的核心问题

1. **Multi-head 相比 single-head 到底多了什么？** "把 $d$ 切成 $h$ 份分别算 attention 再拼" 为什么比 "同样参数规模的单头" 更好？
2. **为什么 Sinusoidal PE 能让模型学到 "相对位置"？** 能不能证明 $PE_{p+k} = M_k \cdot PE_p$？Learnable PE 什么时候胜、什么时候败？
3. **Pre-LN 和 Post-LN 到底差在哪？** 在残差 + LayerNorm 的组合里，LN 的位置如何影响梯度通路？
4. **FFN 为什么要 $4 \times$ 扩展？** 这个数字有理论依据还是纯经验？
5. **一个标准 Encoder Block 的参数量怎么估算？** 为什么实际工程里常说 "参数主要在 FFN"？

---

## 2. 理论骨架

### 2.1 Multi-Head Attention 的数学与直觉

单头 attention 把所有的 "注意力能力" 压在一个 $(d, d)$ 的投影矩阵里，学出来的 attention 权重只能反映 **一种** 相似度。多头做法：

1. 把 $d_{model}$ 切成 $h$ 份，每份 $d_{head} = d_{model} / h$。
2. 每个 head 有自己的 $W_q^{(i)}, W_k^{(i)}, W_v^{(i)} \in \mathbb{R}^{d_{model} \times d_{head}}$，独立算 attention。
3. 把 $h$ 个 head 的输出拼接回 $d_{model}$ 维，过统一的 $W_o \in \mathbb{R}^{d_{model} \times d_{model}}$。

$$
\mathrm{MultiHead}(X) = [\,\mathrm{head}_1 ;\, \mathrm{head}_2 ;\, \cdots ;\, \mathrm{head}_h\,]\, W_o
$$

其中 $\mathrm{head}_i = \mathrm{softmax}(Q_i K_i^\top / \sqrt{d_{head}})\, V_i$，$Q_i = X W_q^{(i)}$ 等。

**总参数量对比**：
- Single-head with $d_{model}$: $3 d_{model}^2 + d_{model}^2 = 4 d_{model}^2$（Q, K, V, O）。
- $h$-head with $d_{head} = d_{model}/h$: 每 head $3 d_{model} d_{head}$，共 $h$ head，合计 $3 d_{model}^2$；加 $W_o$ 的 $d_{model}^2$；总 $4 d_{model}^2$。

**完全相同！** multi-head **不增加参数**，只是把同样的参数分块使用。

那为什么有效？三条理由：

1. **子空间多样性**：每个 head 在 $d_{head}$ 维的子空间里算 softmax，可能学到不同的注意模式 —— 有的关注近邻、有的关注极值、有的关注句首。多模式聚合比单一模式鲁棒。
2. **梯度互相抵消的风险降低**：所有 head 共享一套参数时，不同 "目标模式" 的梯度会在同一组 $W_q, W_k$ 上相互冲突；拆成多 head 后，每个头自由优化自己的子空间，梯度互相独立。
3. **low-rank bias 变缓**：softmax 后每个 attention 矩阵本质是 rank-L 的（最多 L 个独立行），但经过 $(d_{head} \times d_{model})$ 的 Value 投影压到 $d_{head}$ 维，单头的 "信息带宽" 被限制在 $d_{head}$。多头并行后每个头贡献 $d_{head}$ 维，总带宽回到 $d_{model}$，表达力更饱满。

**实现关键**（notebook cell 3）：不写 for 循环，用 `reshape + transpose` 把 `(B, L, d_model)` 变成 `(B, h, L, d_head)`，一次 batched matmul 算完：

```python
def _split_heads(self, x):
    B, L, _ = x.shape
    return x.view(B, L, self.h, self.d_head).transpose(1, 2)   # (B, h, L, d_head)
```

Q/K/V 的 score 就成了 `(B, h, L, L)`，每个 head 独立 softmax，最后合并：

```python
out = torch.matmul(attn, V)              # (B, h, L, d_head)
out = self._merge_heads(out)             # (B, L, d_model)
```

这种 "一次投影到 $d_{model}$，再 reshape 切分" 的技巧在现代 Transformer 里是标准做法 —— 从算力角度看，三次大 matmul 比 $3h$ 次小 matmul 更友好。

### 2.2 Multi-Head 为什么比同参数单头好 —— 深入

这是一个常被问到的面试题。回答框架：

**假设对照**：把单头的 $d_{model}$ 等比缩小到 $d_{model} / h$，让参数量和 multi-head 相等。此时单头和单个 head 参数量一样，但 multi-head 多了 "并行 h 个独立视角"。

**表达能力分析**：
- 单 head (dim = $d_{model}/h$) 的 attention 是一个 $(L, L)$ 矩阵，所选 value 都在 $d_{head}$ 维子空间里。如果任务需要捕获多种关系，一个子空间容不下。
- $h$-head 允许 $h$ 个独立的 $(L, L)$ attention 矩阵共同决定输出。输出是 $h$ 个 $d_{head}$ 向量拼接，总带宽回到 $d_{model}$。

**优化角度**：$h$ 个独立 head 的梯度流是分离的（直到 $W_o$ 才融合），相当于一种 "mini-ensemble"，减少了单一头陷入局部最优的风险。Voita et al. 2019 用 head pruning 实验证明：训练好的 BERT 里只有少数 head 承担主要功能，其余可以 prune，但训练过程中所有 head 都在探索，才能找到这些关键模式。

**直觉对比 CNN 多通道**：单头 attention 像 "单通道" CNN，多头像 "多通道" CNN。每个通道学不同滤波器；每个 head 学不同注意模式。这是深度学习里反复出现的 "多视图特征" 思路。

### 2.3 Sinusoidal Positional Encoding 的数学

定义：

$$
PE_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad PE_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right)
$$

其中 $p$ 是位置，$i \in \{0, \dots, d/2 - 1\}$ 是维度对的索引。最低频率维度（$i=0$）波长 $2\pi$，最高频率维度（$i = d/2 - 1$）波长 $2\pi \cdot 10000$。几何级数的频率覆盖让每个位置都能被唯一编码。

**关键性质：相对位置可以由线性变换得到**

设 $\omega_i = 1 / 10000^{2i/d}$，位置 $p$ 处的第 $i$ 对分量是 $(\sin p\omega_i, \cos p\omega_i)$。利用三角和公式：

$$
\begin{aligned}
\sin((p+k)\omega_i) &= \sin(p\omega_i)\cos(k\omega_i) + \cos(p\omega_i)\sin(k\omega_i) \\
\cos((p+k)\omega_i) &= \cos(p\omega_i)\cos(k\omega_i) - \sin(p\omega_i)\sin(k\omega_i)
\end{aligned}
$$

写成矩阵形式：

$$
\begin{bmatrix} \sin((p+k)\omega_i) \\ \cos((p+k)\omega_i) \end{bmatrix}
=
\begin{bmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{bmatrix}
\begin{bmatrix} \sin(p\omega_i) \\ \cos(p\omega_i) \end{bmatrix}
$$

**每一对维度上，从位置 $p$ 到位置 $p+k$ 是一个只依赖 $k$ 的旋转矩阵。** 把所有 $i$ 拼起来，整体 $PE_{p+k} = M_k \cdot PE_p$，其中 $M_k$ 是一个块对角矩阵（$d/2$ 个 $2 \times 2$ 旋转块），只与偏移 $k$ 有关、与起点 $p$ 无关。

**意义**：模型只要学一个线性变换就能从 "绝对位置 p" 得到 "绝对位置 p+k"，相对位置 $k$ 的信息天然可线性编码。点积 $PE_p \cdot PE_{p+k}$ 也只依赖 $k$（不依赖 $p$），这给了 attention 一个纯粹的 "相对位置分数"。

**外推能力**：训练时只见过位置 $[0, 512]$，推理时遇到位置 $1000$，sinusoidal PE 仍然有定义（只是三角函数在更大输入上取值），模型有一定泛化能力。相比之下 learnable PE 的 embedding table 只有 512 个槽，位置 1000 直接越界。但外推也不是万能：超过训练分布后，模型学到的 "如何利用 PE" 的模式可能失效，实际表现不一定好。Week 11 会看到 RoPE/ALiBi 等改进正是在外推性上做文章。

### 2.4 Learnable PE 的利弊

Learnable PE 就是一个 `nn.Embedding(max_len, d_model)`，每个位置对应一个可学习的向量。

**优点**：
- 简单；参数量 $L \cdot d$，对常见 $L=512, d=768$ 只有 0.4M 参数。
- 表达力强：没有 sinusoidal 的 "必须周期" 约束，可以学到任何模式。
- 在足够数据下，学出来的 PE 经常和 sinusoidal 很像（看 BERT 早期可视化），说明 sinusoidal 是一个合理的先验。

**缺点**：
- 无法外推。position > max_len 时查表越界（必须显式预留足够长度）。
- 在低资源任务上，可能在 PE 上过拟合位置分布，而 sinusoidal 是泛化的先验，数据少时更稳。

**选择建议**：
- NLP（BERT/GPT-2）用 learnable；
- 翻译、机器推理等需要外推的任务倾向 sinusoidal；
- 金融时序异常检测数据通常充足且长度固定 ($L=32$ 或 $64$)，两者都可，learnable 稍方便一点。

现代前沿改进（RoPE / ALiBi）是本周不展开但值得知道的：
- **RoPE**：把 sinusoidal 不是加到 embedding，而是乘到 Q/K 上（旋转 Q/K 向量），使得 attention score 天然包含相对位置。LLaMA、ChatGLM 等模型在用。
- **ALiBi**：直接在 attention score 上加一个 $-|i-j|$ 的线性惩罚，完全不需要 PE 向量。推理时外推到任意长度都不退化。

### 2.5 Pre-LN vs Post-LN —— 残差路径的梯度分析

两种 Transformer block 的写法：

**Post-LN（原论文 Vaswani et al. 2017）**：

$$
y = \mathrm{LN}(x + \mathrm{Sublayer}(x))
$$

**Pre-LN（GPT-2 之后成为主流）**：

$$
y = x + \mathrm{Sublayer}(\mathrm{LN}(x))
$$

看起来只是 LN 的位置换了一下，梯度影响却截然不同。

**Post-LN 的梯度路径**：

$$
\frac{\partial y}{\partial x} = \frac{\partial \mathrm{LN}}{\partial (x + \mathrm{Sublayer}(x))} \cdot \left(I + \frac{\partial \mathrm{Sublayer}}{\partial x}\right)
$$

LayerNorm 对输入的 Jacobian 是 **去均值、除标准差、乘 gamma**，这是一个 **缩减方差** 的算子。深层堆叠 $N$ 个 Post-LN 块，梯度每经过一层都会被 LN 的 Jacobian 作用一次，残差支路信号随深度衰减。

Xiong et al. 2020 "On Layer Normalization in the Transformer Architecture" 用严格的谱分析证明：Post-LN 在初始化时，输出对输入的期望梯度范数 $\|\partial y / \partial x\| = O(1/\sqrt{N})$，$N$ 层堆起来梯度衰减显著，**训练初期几乎无法启动**，所以原论文必须用 warmup（慢慢把 lr 从 0 提到目标值），给 post-LN 一段时间让中间层 "暖起来"。

**Pre-LN 的梯度路径**：

$$
y = x + \mathrm{Sublayer}(\mathrm{LN}(x)) \implies \frac{\partial y}{\partial x} = I + \frac{\partial \mathrm{Sublayer}}{\partial x}
$$

**残差路径是严格的恒等，LN 只作用在 sublayer 的输入上，不截断主梯度流**。深层堆叠时梯度 $\partial y / \partial x \approx I$，信号无损流过；sublayer 的贡献是小扰动。结果就是可以用更大的 lr、更少的 warmup，训练更稳。

**实验证据**（notebook cell 20）：`lr=5e-3` 下 Pre-LN 正常收敛，Post-LN 震荡剧烈（在深层模型上甚至 NaN）。`lr=1e-3` 时两者接近 —— 说明 Post-LN 不是不能用，只是对超参敏感、需要精细 warmup。

**现代选择**：GPT-2/3, LLaMA, PaLM 全部 Pre-LN。原论文 Post-LN 现在只在论文 reproduce 时见到。

**一点 nuance**：Pre-LN 也有它的毛病 —— 深层 Pre-LN 的输出方差会随层数累积（因为残差路径线性叠加），所以最后还要加一个 "final LN"（notebook cell 11 的 `self.ln_f`），把输出归一化。Post-LN 不需要 final LN，因为每层输出已经被 LN 过了。

### 2.6 FFN 与 4× 扩展比

Position-wise FFN：

$$
\mathrm{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2
$$

其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$\sigma$ 是激活函数（原论文 ReLU，现代多用 GELU / SwiGLU）。"position-wise" 指的是：在 $(B, L, d)$ 张量上，FFN 对每个 position 独立作用，不跨位置交互。这和 attention 的 "跨位置交互" 形成完美互补 —— attention 聚合上下文，FFN 做非线性变换。

**为什么 $d_{ff} = 4 d_{model}$？**
- **经验值起源**：原论文用 $d_{ff}=2048, d_{model}=512$（4 倍）。后续 BERT、GPT、T5 基本沿用。
- **理论支持**：FFN 被证明是 Transformer 里最大的 "记忆 / knowledge 存储"（Geva et al. 2021 "Transformer Feed-Forward Layers Are Key-Value Memories"），扩展比越大，容量越高。
- **参数权衡**：$d_{ff} = 4 d$ 时，每层 FFN 参数量 $\approx 8 d^2$，占整层参数的 2/3；再扩大收益边际递减。
- **现代变体**：GLU family（SwiGLU, GeGLU）通常用 $d_{ff} \approx 8/3 \cdot d$ 补偿门控机制多出来的矩阵，最终参数量和 $4d$ 版本接近。LLaMA 用的就是 SwiGLU。

**FFN 为什么不可省**：只有 attention 的模型（纯线性聚合）表达能力严重受限 —— 多层 attention 叠加仍然是线性变换（modulo softmax 非线性）。FFN 的 GELU 提供关键的非线性，让每层都能做 "per-position 的非线性特征变换"。Jelassi et al. 2022 证明去掉 FFN 的 Transformer 无法学会某些算术任务。

### 2.7 参数量估算公式

一个标准 Encoder Block（单层）的参数：

| 组件 | 参数量 |
|------|--------|
| Q, K, V, O 投影（MHA） | $4 d^2$ |
| FFN 第一层 $W_1$ + bias | $4d \cdot d + 4d = 4d^2 + 4d$ |
| FFN 第二层 $W_2$ + bias | $4d \cdot d + d = 4d^2 + d$ |
| LayerNorm × 2（gamma + beta 各 d） | $4d$ |
| **合计** | $\approx 12 d^2 + 9d \approx 12 d^2$ |

$N$ 层模型总参数 $\approx 12 N d^2$，加 embedding 的 $V d$（V 是 vocab size）和 PE 的 $L d$。

常见规模：
- `d=512, N=6, V=30000, L=512`：$12 \cdot 6 \cdot 512^2 \approx 19\text{M}$ + embedding 15M + PE 0.26M ≈ 34M 参数。BERT-base 实际是 110M，差在多头输出投影、biases、和一些细节。
- `d=768, N=12`（BERT-base）：$12 \cdot 12 \cdot 768^2 \approx 85\text{M}$ + embedding 23M ≈ 108M，和实际相符。
- 本周 notebook 的 tiny 模型：`d=32, heads=4, n_layers=2`，每层 block 约 $12 \cdot 32^2 = 12\text{K}$，两层 24K，加 vocab 16 × 32 = 0.5K embedding，总 ~25K 参数。

**工程启示**：FFN 占参数的 2/3，是显存大头。做模型压缩 / LoRA 优先在 FFN 动手，attention 权重更难削（裁剪会影响注意模式质量）。

---

## 3. 代码对照 —— 06_mha_pe.ipynb

### 3.1 `MultiHeadAttention` 的 reshape 技巧（cell 3）

```python
def _split_heads(self, x):
    B, L, _ = x.shape
    return x.view(B, L, self.h, self.d_head).transpose(1, 2)

def forward(self, x, padding_mask=None):
    Q = self._split_heads(self.W_q(x))      # (B, h, L, d_head)
    K = self._split_heads(self.W_k(x))
    V = self._split_heads(self.W_v(x))
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
```

两个要点：
1. **先投影到 $d_{model}$，再 reshape 切分**。等价于 $h$ 个 $d_{head}$ 投影的拼接，但一次 matmul 比 $h$ 次 matmul 对 GPU 更友好。
2. **`transpose(1, 2)` 把 head 维度提前**，这样后续的 matmul 在 `(B, h, L, d_head) × (B, h, d_head, L)` 形状下对每个 head 独立计算，恰好是 batched matrix multiplication 的 PyTorch 语义。

padding mask 的扩展：

```python
m = padding_mask[:, None, None, :]     # (B, 1, 1, L)
scores = scores.masked_fill(m == 0, float('-inf'))
```

`(B, L)` → `(B, 1, 1, L)` 通过 broadcast 作用在 scores 的 `(B, h, L, L)` 上，广播到所有 head 和所有 query 位置，只屏蔽 key 轴。Week 5 的单头只有 `(B, 1, L)`，多头多了一个 head 维度要对齐。

### 3.2 Sinusoidal PE 的实现（cell 5）

```python
pe = torch.zeros(max_len, d_model)
pos = torch.arange(max_len).unsqueeze(1).float()
div = torch.exp(torch.arange(0, d_model, 2).float() *
                (-(torch.log(torch.tensor(10000.0))) / d_model))
pe[:, 0::2] = torch.sin(pos * div)
pe[:, 1::2] = torch.cos(pos * div)
self.register_buffer('pe', pe)
```

用 `exp(-log(10000) * 2i / d)` 等价于 $1 / 10000^{2i/d}$，在数值上比直接做指数更稳。`register_buffer` 把 PE 注册为 **不可学习但随模型移动设备** 的张量 —— 和 `nn.Parameter` 的区别是不参与梯度更新。

**一个细节**：原论文把 PE 和 embedding 相加（`return x + pe`）之前，embedding 会先乘 $\sqrt{d_{model}}$，让 embedding 的方差和 PE 的方差匹配。notebook 这里没做这一步，但当 `d=32` 较小影响不大。推广到大模型（`d=512+`）要记得加回来。

### 3.3 Pre-LN EncoderBlock（cell 9）

```python
def forward(self, x, padding_mask=None):
    if self.pre_ln:
        x = x + self.drop(self.mha(self.ln1(x), padding_mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
    else:
        x = self.ln1(x + self.drop(self.mha(x, padding_mask)))
        x = self.ln2(x + self.drop(self.ffn(x)))
    return x
```

对照 2.5 节的公式：
- Pre-LN：sublayer 的输入先 LN，输出直接加到残差。残差路径恒等，梯度无损。
- Post-LN：sublayer 输出加到残差，整体再 LN。残差路径被 LN 作用，梯度衰减。

FFN 里用 **GELU** 而不是 ReLU（cell 9 的 FeedForward）：

```python
nn.Sequential(nn.Linear(d_model, mult * d_model), nn.GELU(), ...)
```

GELU ≈ $x \cdot \Phi(x)$（标准正态 CDF），在 0 附近比 ReLU 平滑，BERT / GPT-2 起流行。现代 LLaMA 系用 SwiGLU，本 notebook 用 GELU 简化实现、效果接近。

### 3.4 PE 消融（cell 18）

```python
no_pe = TransformerClassifier(n_layers=2, use_pe=False)
no_pe_losses, no_pe_acc, _ = train_clf(no_pe, ...)
```

**预期**：toy 任务 "前缀和跨阈值" 对顺序敏感（至少理论上如此），不加 PE 应该学不下来。

**实测可能的惊喜**：如 Week 5 第 2.4 节讨论的，这个 toy 任务其实对 non-negative token 的顺序不太敏感（cumsum max 等于全和），即使不加 PE 也能学到 ~80% 准确率。如果你在实验里看到 no-PE 依然工作正常，那不是 PE 不重要，而是任务降级了。想让消融更戏剧化，可以把任务改成 "最后一位 token > 前面所有 token" 或 "首位与末位差 > 5"，这类强顺序依赖任务下 no-PE 的 acc 会停留在 50% (random)。

### 3.5 Pre-LN vs Post-LN 消融（cell 20）

```python
HIGH_LR = 5e-3
pre = TransformerClassifier(n_layers=2, pre_ln=True)
post = TransformerClassifier(n_layers=2, pre_ln=False)
```

故意用一个 **高 lr** 来暴露 Post-LN 的不稳定性。低 lr 下两者都能跑，这恰好印证 2.5 节的理论：Post-LN 本身能工作，只是对 lr 敏感，需要 warmup 和精细调参。

**观察要点**：
- Pre-LN 的 loss 曲线相对平滑下降。
- Post-LN 的 loss 可能初期高频震荡、甚至短暂上升（梯度方向飘）。
- 2 层的模型差异还不明显，4 层以上差异会急剧扩大（Xiong 的分析基于深层）。

想看更戏剧化的效果，把 `n_layers` 调到 6，你会看到 Post-LN 直接 NaN 的典型现象。

---

## 4. 常见坑位与调试思维

1. **Multi-head reshape 维度弄错**：`view(B, L, h, d_head)` 和 `view(B, h, L, d_head)` 是不同的内存布局，前者后面要配 `transpose(1, 2)`，后者必须在 view 之前已经 transpose。常见症状：attention 权重看起来有意义但 loss 不降。调试办法：打印 `Q[0, 0]` 和 `Q[0, :, 0]` 看哪边是 head 维度。
2. **PE 加错位置**：必须加在 `embedding * scale + PE`，不能加在 block 内部。重复加多次会导致 "位置信号被稀释"。Karpathy 的 nanoGPT 在所有 block 之前一次性加 PE。
3. **Pre-LN 最后忘了加 final LN**：Pre-LN 的每层输出没有被 LN，最后直接进 classification head 可能方差过大。需要在所有 block 之后加一个 `self.ln_f = nn.LayerNorm(d)`（notebook cell 11 的 `self.ln_f` 就是这个作用）。
4. **Post-LN + 无 warmup + 高 lr = NaN**：这是一个可以背下来的组合。如果你非要复现原论文的 Post-LN，**必须** 加 Noam warmup：lr 从 0 线性升到目标 lr over 4000 steps，再按 $\mathrm{step}^{-0.5}$ 衰减。
5. **`register_buffer` vs `nn.Parameter`**：sinusoidal PE 用 buffer（不可学），learnable PE 用 `nn.Embedding` 或 `nn.Parameter`。混淆了会导致：(a) 冻结了 learnable PE，模型学不到；或 (b) 意外训练了 sinusoidal PE，破坏了它的外推性质。
6. **GELU 和 ReLU 的差别**：小 d 模型用 ReLU 也没事，但大模型 ReLU 会让 dead neuron 问题更明显。工程上默认 GELU，性能接近 SwiGLU 但实现简单。
7. **d_model 必须整除 num_heads**：`assert d_model % num_heads == 0`，否则 reshape 报错。规划模型时记得：`d_model=512, h=8, d_head=64`；`d_model=768, h=12, d_head=64`（BERT-base）。习惯 $d_{head}=64$ 是因为它在 A100 GPU 的 tensor core 上对齐最好。
8. **PE 外推失败不是 sinusoidal 的锅**：理论上 sinusoidal 能外推到任意长度，但 **模型在训练时学到的 attention 权重分布** 是依赖于训练长度范围的。训练 $L=512$ 的模型去做 $L=4096$ 的推理，不论 PE 是什么，attention 分布都会漂（softmax 被稀释）。现代改进如 ALiBi 在 attention score 级别引入长度正则化，才能真正外推。

---

## 5. 与未来几周的连接

- **Week 7** 直接堆 4 层本周的 EncoderBlock，加分类头（`[CLS]` token 或 mean pool），配 Noam warmup，跑 MVP v0.3 第一版。本周所有模块都会被复用，不加任何新数学，只做 "深度" 和 "真实数据" 两件事。
- **Week 8** 读 nanoGPT，你会看到 Karpathy 的 `Block` 几乎就是本周 `EncoderBlock`，唯一区别是 MHA 里加了 causal mask（下三角 -∞），让每个位置只能看过去。理解这一个改动，GPT 和 BERT 的差别就清楚了。
- **Week 9 Anomaly Transformer** 用 Gaussian kernel 的 "prior attention" 和学习的 "series attention" 做差异作为异常分数。这需要你对本周 MHA 的权重矩阵非常熟悉 —— 它直接在那个 `(B, h, L, L)` 张量上做文章。
- **Week 11 的 PatchTST / Informer** 是对本周 $O(L^2)$ attention 的减法（patching 降 L，ProbSparse 选 top-u 个 query）；RoPE / ALiBi 是对本周 PE 的替换。理解本周的基线，才能判断这些变体在哪里更好。
- **Week 12** 做推理优化，FFN 的 $4d$ 扩展会是量化 / LoRA 的主攻目标，对应 2.7 节 "FFN 占参数 2/3" 的结论。

---

## 6. 自测题

<details>
<summary>Q1. Multi-head 和 single-head（同参数规模）的差异，用子空间语言描述一下。</summary>

Single-head 在 $d_{model}$ 维整体空间里做一次 softmax，$W_v$ 投影到 $d_{head}$ 维（或保持 $d_{model}$），学一种注意模式。

Multi-head 把 $d_{model}$ 切成 $h$ 个 $d_{head}$ 子空间，每个子空间独立算 attention 和 value。$h$ 个独立的 attention 权重矩阵允许捕获多种关系（近邻、极值、关键位置等），最后拼接融合。总参数量相同，但表达力更丰富、梯度路径相互独立，训练更稳。
</details>

<details>
<summary>Q2. 证明 sinusoidal PE 满足 $PE_{p+k} = M_k \cdot PE_p$，其中 $M_k$ 不依赖 $p$。</summary>

每一对维度 $(2i, 2i+1)$ 上，$(\sin p\omega_i, \cos p\omega_i)$ 转到 $(p+k)$ 时用三角和公式：

$\begin{bmatrix}\sin((p+k)\omega_i) \\ \cos((p+k)\omega_i)\end{bmatrix} = \begin{bmatrix}\cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i)\end{bmatrix}\begin{bmatrix}\sin p\omega_i \\ \cos p\omega_i\end{bmatrix}$

这是一个只依赖 $k\omega_i$ 的旋转矩阵。把所有 $i$ 的旋转块拼成块对角矩阵 $M_k = \mathrm{blockdiag}(R_{k\omega_0}, \dots)$，整体有 $PE_{p+k} = M_k \cdot PE_p$，与 $p$ 无关。
</details>

<details>
<summary>Q3. 为什么 Pre-LN 比 Post-LN 在深层模型上更稳？</summary>

Pre-LN: $y = x + \mathrm{Sub}(\mathrm{LN}(x))$，残差路径恒等（$\partial y / \partial x = I + \partial \mathrm{Sub}$），梯度直通。

Post-LN: $y = \mathrm{LN}(x + \mathrm{Sub}(x))$，残差后还要过 LN，LN 的 Jacobian 是方差缩减算子。$N$ 层堆起来，梯度被 LN 的 Jacobian 作用 $N$ 次，衰减严重，初期梯度小到模型难启动。需要 warmup 补偿。
</details>

<details>
<summary>Q4. FFN 的 $4\times$ 扩展比有什么理论 / 经验依据？</summary>

经验起源于原论文 ($d_{ff}=2048, d=512$)。理论上：FFN 是 Transformer 的主要 "knowledge 存储" 容器（Geva 2021），扩展比越大容量越大。工程上 $4d$ 时 FFN 参数约 $8d^2$，占每层 2/3 参数；再扩大边际递减且显存炸。现代 GLU 变体 (SwiGLU) 用 $d_{ff} \approx 8d/3$ 补偿门控多出的矩阵，总参数等价于 $4d$。
</details>

<details>
<summary>Q5. 一个 $d=512, N=6$ 的 Encoder（不算 embedding）大概多少参数？按 $12 N d^2$ 估。</summary>

$12 \times 6 \times 512^2 \approx 12 \times 6 \times 262144 \approx 18.9 \text{M}$。加 embedding 和 PE 大约 $34\text{M}$，和 BERT-base 一半深度的规模接近。
</details>

<details>
<summary>Q6. Learnable PE 无法外推到训练时没见过的长度，为什么？Sinusoidal 能吗？</summary>

Learnable PE 是一个 `Embedding(max_len, d)`，超出 max_len 没有对应槽（数组越界）。Sinusoidal PE 是 $\sin / \cos$ 的解析公式，任何 position 都有定义 → 形式上可外推。但外推效果不保证：模型在训练分布外可能不知如何使用超长位置。ALiBi / RoPE 对外推做了专门优化，是更强的替代方案。
</details>

<details>
<summary>Q7. `_split_heads` 为什么要 transpose？不 transpose 能不能算？</summary>

目标是让 batched matmul 的广播维度是 `(B, h)`，独立维度是 `(L, d_head)`。

没 transpose 的形状是 `(B, L, h, d_head)`，此时矩阵乘变成在 `(L, h)` 这对维度上广播，语义混乱、性能糟。transpose 后变成 `(B, h, L, d_head)`，`torch.matmul` 把最后两维当矩阵、前面都当 batch，一次算完所有 B×h 个 head，对 GPU tensor core 最友好。
</details>

<details>
<summary>Q8. Pre-LN 模型最后为什么还要加一个 final LN？</summary>

Pre-LN 的每层输出 `x + Sub(LN(x))` 没有被 LN，残差累积会让输出方差随层数增大。最后进 classification head 或 next-token 前加一个 `LayerNorm(d)` 把最终表征归一化，避免 head 输入方差失控。Post-LN 不需要因为每层末尾已 LN。这是现代实现（GPT-2/3, LLaMA）的标准结构。
</details>

---

## 7. 延伸阅读

1. **Vaswani et al., "Attention Is All You Need" — Section 3.3-3.5 (2017)** — <https://arxiv.org/abs/1706.03762>
   现在读：上周读了 Section 3.2，本周把 3.3（FFN）、3.4（Embeddings & Softmax）、3.5（Positional Encoding）补完。对照 2.3 和 2.6 节。原论文用的是 Post-LN，能帮你理解 "为什么当年要 warmup"。

2. **Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML 2020)** — <https://arxiv.org/abs/2002.04745>
   现在读：严格证明 Pre-LN 在初始化下梯度稳定、Post-LN 需要 warmup 的数学依据。2.5 节的梯度分析直接来源于此。篇幅不长，值得精读。

3. **Voita et al., "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned" (ACL 2019)** — <https://arxiv.org/abs/1905.09418>
   现在读：用 head pruning 实验证明 "多头训练出了功能特化"。给 2.2 节的 "多头有效性" 提供实证证据。读完你会觉得 "啊原来训练后真的有头专门管 coreference / 句法……"。

4. **Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)** — <https://arxiv.org/abs/2104.09864>
   现在读：RoPE 的原始论文。对照本周 2.3 节的旋转矩阵分析 —— RoPE 其实就是把这个旋转从 "加到 embedding" 改成 "乘到 Q/K"。读了 sinusoidal 的推导再读 RoPE，会觉得 "这不是一个自然的改进吗"。Week 11 会回来用。

5. **Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (EMNLP 2021)** — <https://arxiv.org/abs/2012.14913>
   现在读：把 FFN 解释为一个隐式的 key-value 记忆表（第一个 $W_1$ 是 keys，第二个 $W_2$ 是 values）。解释了 2.6 节 "FFN 是知识存储" 的说法，让你在 Week 12 做模型压缩 / LoRA 时有个好的心理模型 —— "我在压缩一本字典"。
