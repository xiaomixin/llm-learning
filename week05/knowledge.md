# Week 5 深度知识伴读 — Self-Attention 的数学与直觉

> 本周你第一次亲手写出 `softmax(QK^T/√d_k)V`。这份文档帮你把 "会抄公式" 变成 "理解为什么是这个公式、每个符号为什么出现"。配合 `05_self_attention.ipynb` 的 toy 任务（前缀和跨阈值判定）和 √d_k 消融实验一起食用。

---

## 1. 本周要回答的核心问题

1. **Q/K/V 是一次软字典查询，这个类比到底准不准？** 能不能用 Nadaraya-Watson 核回归把它严格起来？
2. **`softmax(QK^T/√d_k)V` 的每个算符是什么意思？** 为什么是这个顺序、不是别的？
3. **为什么 **必须** 除以 $\sqrt{d_k}$？** 能不能从方差角度推出这个 $\sqrt{d_k}$ 是唯一合理的缩放？
4. **Self-attention 对置换等变（permutation-equivariant）意味着什么？** 为什么这既是它的优势（归纳偏置弱）也是它的负担（需要 PE）？
5. **$O(L^2 \cdot d)$ 的复杂度什么时候是瓶颈、什么时候反而便宜？** 和 RNN 的 $O(L \cdot d^2)$ 在什么 L 下交叉？

---

## 2. 理论骨架

### 2.1 从加权平均到 Nadaraya-Watson，再到 Attention

给一个有监督的 "查表" 问题：训练集 $\{(k_i, v_i)\}_{i=1}^n$，来了一个 query $q$，要输出一个预测值。

- **硬查询（lookup table）**：找 key 完全匹配的那一项，返回它的 value。问题：如果 $q$ 不在表里就 gg，也没法微分。
- **Nadaraya-Watson 核回归**（1964）：

$$
\hat f(q) = \sum_{i=1}^{n} \frac{K(q, k_i)}{\sum_j K(q, k_j)}\, v_i
$$

其中 $K$ 是核函数（RBF、cosine 都行），分数部分就是一组 **归一化权重**。这是统计学里最朴素的 "相似度加权平均"。

- **Self-Attention** = 把核函数换成 **可学习的点积 + softmax 归一化**，并且 query / key / value 都由 **同一份输入经过不同投影** 得到：

$$
\mathrm{Attn}(q, K, V) = \sum_{i=1}^n \frac{\exp(q \cdot k_i / \sqrt{d_k})}{\sum_j \exp(q \cdot k_j / \sqrt{d_k})} v_i = \mathrm{softmax}\!\left(\frac{q K^\top}{\sqrt{d_k}}\right) V
$$

批量化到所有 query：

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

**关键认知**：attention 不是什么神秘的东西，它是 Nadaraya-Watson 核平滑器的一个可微分、可学习的现代版本。核函数从固定的 RBF 变成了 "先把 query 和 key 都投影一下再点积"；权重的 "带宽" 通过学到的投影矩阵隐式控制。

### 2.2 Q / K / V 的三角色与 "软字典" 类比

三个投影 $W_q, W_k, W_v$ 各自强调的信息：

- $Q = X W_q$：**我想找什么？** 当前位置想从其他位置拉取的信息类型。
- $K = X W_k$：**我是什么？** 每个位置对外 "广告" 的标签。
- $V = X W_v$：**我实际提供什么？** 真正被加权汇总的内容。

把 K 和 V 分开的意义 —— **检索 key 可以和返回 value 不同**。比如 K 可能编码 "这个位置是一笔金额异常交易的时间戳"，而 V 编码 "这笔交易的详细特征"。Q 按 "异常时间" 匹配到 K，但真正拿回的是 V 里的详细特征。如果 K 和 V 被绑在一起（像经典记忆网络某些版本），就丧失了这种解耦。

一句话：**Q 问问题，K 自我声明，V 给答案，softmax 决定问题和声明的匹配权重。**

### 2.3 为什么除以 $\sqrt{d_k}$ —— 方差推导

先把 $Q, K$ 的每个元素建模成独立同分布、均值 0、方差 1 的随机变量（做完 Xavier 初始化的线性投影近似满足）。

考虑两个向量 $q, k \in \mathbb{R}^{d_k}$ 的点积：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

每一项的均值 $E[q_i k_i] = E[q_i]E[k_i] = 0$，方差：

$$
\mathrm{Var}(q_i k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2 = E[q_i^2]\, E[k_i^2] - 0 = 1 \cdot 1 = 1
$$

由独立性，和的方差等于各项方差之和：

$$
\mathrm{Var}(q \cdot k) = \sum_{i=1}^{d_k} \mathrm{Var}(q_i k_i) = d_k
$$

因此标准差为 $\sqrt{d_k}$。

**后果**：当 $d_k = 64$ 或 $128$ 时，点积幅度的典型值在 $\pm 8$ 到 $\pm 12$ 之间。把这个值送进 softmax：

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

softmax 对输入的 **差值** 敏感。如果最大值比其他值大 10，$e^{10} \approx 22000$，softmax 几乎输出 one-hot。这时对 $z$ 的梯度：

$$
\frac{\partial \mathrm{softmax}(z)_i}{\partial z_j} = \mathrm{softmax}(z)_i (\delta_{ij} - \mathrm{softmax}(z)_j)
$$

如果 softmax 已经是 one-hot，$p(1-p) \approx 0$，梯度几乎全灭 —— **softmax 饱和导致的梯度消失**。

除以 $\sqrt{d_k}$ 后，$\mathrm{Var}(q \cdot k / \sqrt{d_k}) = 1$，点积幅度回到 $\pm 2 \sim \pm 3$，softmax 输出分布平滑，梯度保持健康。

**为什么是 $\sqrt{d_k}$ 不是 $d_k$**：目标是让方差回到 1，不是让期望回到 $1/d_k$。除以标准差是方差归一，除以 $d_k$ 会让 softmax 太平（几乎均匀），相当于 "永远不聚焦"。这是一个有严格理论依据的 "恰好" 的值。

notebook 的 cell 17-18 做了消融：`scaled=False` 下初期 loss 下降更慢、更振荡，和上面的分析完全一致。

### 2.4 Permutation-Equivariance —— 弱归纳偏置的代价

定义 **置换矩阵** $P$ 作用在序列上：$X' = PX$（把行重排）。self-attention 满足：

$$
\mathrm{Attn}(PX) = P \cdot \mathrm{Attn}(X)
$$

证明不难：$Q' = PXW_q = PQ$，$K' = PK$，$V' = PV$，于是 $(Q')(K')^\top = PQK^\top P^\top$，softmax 逐行不变，$V'$ 也被 $P$ 作用，最终输出只是对应行重排。

**这个性质的两面**：
- 优势：模型对 "哪个位置算第一个" 没有先验，可以处理集合、图等无序数据；也不会像 CNN 那样被 locality 先验 "锁死"。
- 代价：对于有顺序的序列（语言、时间序列、交易流），模型会 **完全看不见顺序**。你把 "早上 100 块、下午 5 万块" 和 "下午 5 万块、早上 100 块" 喂给 attention，输出完全一样。

解决方法：**Positional Encoding**（Week 6 细讲）。本周的 toy 任务是 "前缀和跨阈值判定"，原本极度依赖顺序 —— 但 notebook 取巧了：`mean pool` 之后做分类，而且标签是 "是否存在一个前缀超过 50"，其实等价于 "所有 token 之和是否超过 50"（因为 token 非负），实际上对顺序不敏感。你会看到单头 attention 不加 PE 也能学到 0.80+ acc —— 不是 attention 看到了顺序，而是任务本身被降级成了 bag-of-tokens 的求和问题。这是一个 "有意放水让学生先把 attention 跑起来" 的教学选择，Week 6 的消融会换上真正需要顺序的版本。

### 2.5 复杂度分析 —— $O(L^2 d)$ vs $O(L d^2)$

Self-attention 的主要计算：

| 步骤 | 形状 | FLOPs |
|------|------|-------|
| $Q = XW_q$（K, V 同理） | $(L, d) \times (d, d) \to (L, d)$ | $O(L d^2)$，共 3 次 |
| $QK^\top$ | $(L, d) \times (d, L) \to (L, L)$ | $O(L^2 d)$ |
| softmax | $(L, L)$ | $O(L^2)$ |
| $\mathrm{attn} \cdot V$ | $(L, L) \times (L, d) \to (L, d)$ | $O(L^2 d)$ |
| $W_o$ 输出投影 | $(L, d) \times (d, d) \to (L, d)$ | $O(L d^2)$ |

**总计**：$O(L^2 d + L d^2)$。当 $L \gg d$，$L^2 d$ 主导；当 $L \ll d$，$L d^2$ 主导。

对比 RNN：每一步 $O(d^2)$ 的矩阵乘，一共 L 步，总 $O(L d^2)$。并行度差到地下（每一步依赖前一步），但绝对 FLOPs 可能更低。

**关键交叉点**：Self-attention 和 RNN 在计算量上大致相当当 $L \approx d$。金融异常检测常见 $L=32, d=128$，显然是 attention 更便宜；但 LLM 时代 $L=8192, d=4096$，attention 的 $L^2 d$ 已经变成瓶颈，于是有 Flash Attention、稀疏 attention、PatchTST 等各种优化（Week 11 再讨论）。

此外，attention 的 **并行度** 和 RNN 根本不是一个量级：所有 L 个位置的 Q/K/V 投影和 attention 计算都能一次 batched matmul 完成，GPU 满血运行。RNN 是时间轴强串行，GPU 利用率往往只有 30-50%。

### 2.6 Attention 作为 "可微分检索" 与 kernel smoother 的联系

更深一层的视角：

- **Hard attention**：argmax 选一个位置，不可微。
- **Soft attention**（即 softmax attention）：所有位置加权平均，可微。
- 当温度 $\tau \to 0$（即点积放大），soft attention 退化为 hard；当 $\tau \to \infty$，退化为均匀平均（即 mean pool）。
- 因此 attention 是一个 **可微的检索器**，在 hard lookup 和 uniform aggregation 之间连续插值。

和 kernel smoother 的关系前面已讲。再提一笔：如果把点积换成 RBF，$\exp(-\|q-k\|^2 / 2)$，attention 就是 Nadaraya-Watson 了。论文 "Rethinking Attention with Performers" (Choromanski et al., 2021) 正是用 random features 把 exp 核改写成线性形式，得到 $O(L)$ 复杂度的 attention 变体。理解 "attention = 可微核平滑器" 是理解这类线性 attention 改造的前提。

---

## 3. 代码对照 —— 05_self_attention.ipynb

### 3.1 `ScaledDotProductAttention`（cell 8）

```python
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1))   # (B, L, L)
        if self.scaled:
            scores = scores / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn
```

逐行对应 2.2-2.3 节：
- `K.transpose(-2, -1)`：把 key 从 $(L, d)$ 转成 $(d, L)$，再和 $Q$ 做矩阵乘，得到 $(L, L)$ 的 score 矩阵。第 $(i, j)$ 个元素就是第 i 个 query 和第 j 个 key 的点积。
- `d_k ** 0.5`：这是 2.3 节公式 $\sqrt{d_k}$ 的直接实现。`self.scaled=False` 就是消融实验。
- `masked_fill(mask == 0, float('-inf'))`：把需要屏蔽的位置（padding 或未来 token）的 score 设为 $-\infty$，softmax 后权重是 0。**注意不能直接把 attn 乘 mask**，那会在 softmax 之后破坏归一化；必须在 softmax 之前加 -∞。这是一个经典坑。
- `dim=-1`：softmax 在 key 轴（每个 query 对所有 key 归一化），不是在 batch 轴。

返回 `attn` 矩阵是为了后面可视化 —— self-attention 的一大卖点就是 **权重可解释**。

### 3.2 `SingleHeadSelfAttention`（cell 10）

```python
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, d_model, scaled=True):
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
```

三处设计：
1. **`bias=False`**：attention 投影通常不加 bias。原因是 LayerNorm / softmax 都会消掉 bias 的效果（softmax 对常数平移不变），加了也没用还占参数。
2. **Q/K/V 维度相同 `d_model → d_model`**：单头场景下为了实现简单。多头会拆成 h 份，每份 $d_{head} = d_{model}/h$。
3. **保留 `W_o` 输出投影**：单头其实不需要 $W_o$ —— 它只是一个多余的线性层。这里保留是为了 **和 Week 6 的 multi-head 接口对齐**，方便直接升级。

### 3.3 Toy 任务：前缀和跨阈值判定（cell 4）

```python
VOCAB, SEQ_LEN, THRESH = 16, 8, 50
def make_dataset(n):
    x = torch.randint(0, VOCAB, (n, SEQ_LEN))
    prefix = x.cumsum(dim=1)
    y = (prefix.max(dim=1).values > THRESH).long()
    return x, y
```

任务设计意图：
- 期望和（$16/2 \times 8 = 64$）略高于阈值 50，所以 pos rate 约 50%，label 平衡。
- 需要 "全局信息" —— 单看某一个 token 无法判断。
- MLP baseline 因为 flatten 后参数对位置敏感，能学到一部分；attention 的预期优势是 "直接对所有位置求和"。

如 2.4 节所述，这个任务其实对顺序不敏感（token 非负的 cumsum max 等价于全和），所以 self-attention 不加 PE 也能学。这是教学编排 —— Week 6 的消融会给 attention 一个真正需要顺序的任务，让 PE 的必要性浮出水面。

### 3.4 消融：去掉 $\sqrt{d_k}$（cell 18）

```python
model_noscale = ToyAttnClassifier(scaled=False)
noscale_losses, noscale_acc = train_clf(model_noscale, ...)
```

预期结果和观察到的现象一致：loss 下降更慢、更振荡，最终 acc 低一截。

这里还隐藏了一个有意思的细节：`d=32` 其实没那么大，softmax 饱和程度有限，消融效果不那么戏剧化。如果你把 `d=32` 改成 `d=128` 重跑，会看到 `scaled=False` 几乎学不下来 —— 更大的 d，softmax 饱和更严重，这是你可以自己做的扩展实验。

### 3.5 Attention 权重可视化（cell 16）

画出 `(L, L)` 的权重矩阵，x 轴是 key 位置，y 轴是 query 位置。一个训练好的模型应该表现出 "高值 token（接近 15）的列权重更大"，说明模型学到了 "对判断跨阈值贡献大的是大数字"。如果权重是均匀的（每行接近 1/L），说明模型没学到 —— 它只是在做平均池化。

这种 "用可视化验证模型到底学到了什么" 的手法，是 Transformer 时代特有的福利 —— RNN 的 hidden state 没这种直接的可解释性。后续 Week 6 multi-head 的每个头可能学到不同模式，可视化会更丰富。

---

## 4. 常见坑位与调试思维

1. **softmax 之后才加 mask**：把 `attn = attn * mask` 放在 softmax 之后，看上去 mask 位置是 0，但其他位置的权重因为除了包含 mask 的归一化项而变形。正确做法永远是 **softmax 之前把 score 设为 $-\infty$**。
2. **mask 的 shape 错位**：attention mask 的 shape 约定很容易错。一个稳妥写法：显式构造 `(B, 1, L)` 的 padding mask，让它 broadcast 到 `(B, L, L)` 的 scores 上，查询轴永远不屏蔽，只屏蔽 key 轴。Week 6 的 multi-head 会扩展为 `(B, 1, 1, L)`。
3. **忘记除 $\sqrt{d_k}$**：症状是大 d 情况下 loss 在第一步就卡住不动，或者 attention 权重长期是 one-hot（画出来看一眼就明白）。如果你手写 attention，第一件事是在 sanity check 里打印 attn 的 entropy，接近 $\log L$ 说明均匀、接近 0 说明饱和。
4. **softmax 的 dim 选错**：新手常见 `F.softmax(scores, dim=1)` 在 `(B, L, L)` 张量上 —— 这是在 query 轴上归一化，含义完全错了。永远是 **最后一维**（key 轴）：`dim=-1`。
5. **W_q / W_k / W_v 用同一个矩阵**：有人为了省参数把三个投影共享，变成 $Q = K = V = XW$。此时 $QK^\top = XWW^\top X^\top$ 是对称的，导致 "我关注谁就等于谁关注我"，表达能力大幅受限。尤其 self-attention 中 "A 注意 B" 和 "B 注意 A" 是两件不同的事，必须用独立投影。
6. **数据类型**：fp16 训练时 softmax 容易数值下溢（$e^{-\text{large}} \to 0$）。标准 fix 是 softmax 前先减去 row max（PyTorch 的 `F.softmax` 自动做了），再就是把 softmax 前后切回 fp32（Flash Attention 的默认行为）。
7. **"attention 权重 ≠ 特征重要性"**：千万别把 attention 矩阵当成 explainability 证据写论文 —— Jain & Wallace "Attention is not Explanation" (NAACL 2019) 实证说明同一模型多组 attention 权重可以给出相同预测。可视化用于 debug（检查模型是否学到预期模式）是 ok 的，作为正式解释需要更严格的工具（如 integrated gradients）。

---

## 5. 与未来几周的连接

- **Week 6** 直接把本周的 `SingleHeadSelfAttention` 升级为 `MultiHeadAttention`，保留 `W_o` 的设计一下子就成了多头合并后的输出投影；同时补上 PE，弥补 2.4 节讲的置换等变缺陷；再加 FFN + LayerNorm + 残差，组装成 `EncoderBlock`。
- **Week 7** 把这个 block 堆到 4 层，加分类头，跑完整 Transformer 的 MVP v0.3。本周的 scaled attention 会在 Week 7 的每一层里被复用。
- **Week 8** 读 nanoGPT 源码，你会发现 Karpathy 的 `CausalSelfAttention` 本质上就是本周代码加一个 causal mask（下三角的 -∞）。其他几乎一模一样。本周写顺了，Week 8 读代码就是 "啊原来是这样"。
- **Week 11** 的 PatchTST 和 Informer 都是对本周的 $O(L^2 d)$ 做减法。理解 2.5 节的复杂度分析，才知道它们在省哪一步。

---

## 6. 自测题

<details>
<summary>Q1. 推导 softmax 的梯度，并据此解释为什么 softmax 饱和 = 梯度消失。</summary>

$p_i = \mathrm{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$，求偏导：

$\dfrac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$

当 softmax 输出接近 one-hot，设 $p_i \approx 1$ 某一项，其余 $p_j \approx 0$：$p_i(1-p_i) \approx 0$，$p_i \cdot p_j \approx 0$。所以 $\partial p / \partial z$ 矩阵元素全小，对上游梯度起的作用是乘一个近零矩阵，梯度被 "挤没了"。
</details>

<details>
<summary>Q2. 从方差角度推出 $\sqrt{d_k}$，并说明为什么不是 $d_k$。</summary>

$q, k \in \mathbb{R}^{d_k}$，各分量独立、均值 0、方差 1。$q \cdot k = \sum q_i k_i$，$E = 0$，$\mathrm{Var} = \sum \mathrm{Var}(q_i k_i) = d_k$。标准差 $\sqrt{d_k}$。

目标是让 score 的方差回到 1，使 softmax 输入处在 "有区分度但不饱和" 的区间。除以 $\sqrt{d_k}$ 正好把方差归一；除以 $d_k$ 会让方差变成 $1/d_k$，softmax 太平（所有 score 都很接近，输出接近均匀），模型学不到尖锐的注意力。
</details>

<details>
<summary>Q3. 为什么 W_q、W_k 不能共享参数？</summary>

如果 $W_q = W_k$，则 $QK^\top = X W W^\top X^\top$ 是 **对称矩阵**，意味着 attention 权重 (i, j) 和 (j, i) 必须相等 —— "我注意谁就等于谁注意我"。但 self-attention 中非对称关系很常见（比如 "动词 attend 到主语" 通常不对称），共享参数会严重限制表达能力。
</details>

<details>
<summary>Q4. Self-attention 的复杂度是 $O(L^2 d + L d^2)$。什么时候哪一项主导？和 RNN 的 $O(L d^2)$ 比如何？</summary>

当 $L \gg d$（长序列、小 hidden），$L^2 d$ 主导；当 $L \ll d$（短序列、大 hidden），$L d^2$ 主导。交叉点大约 $L \approx d$。

相比 RNN 的 $O(L d^2)$，attention 在 $L < d$ 时更便宜（FLOPs 更少），$L > d$ 时更贵但可以并行。金融异常检测 $L=32, d=128$ 属于前者，attention 优势明显；LLM $L=8192, d=4096$ 属于后者，于是有了 Flash/Sparse attention。
</details>

<details>
<summary>Q5. 证明 self-attention 是 permutation-equivariant。</summary>

置换矩阵 $P$：$X' = PX$。$Q' = PXW_q = PQ$，同理 $K' = PK, V' = PV$。

Scores：$Q'K'^\top = PQK^\top P^\top$。softmax 逐行做：$P \cdot \mathrm{softmax}(QK^\top) \cdot P^\top$（P^T 是列重排，配合 softmax 不变性，其实 softmax 对 query 轴不变；对 key 轴重排后权重也重排）。最终 $\mathrm{softmax}(Q'K'^\top/\sqrt{d})V' = P \cdot \mathrm{softmax}(QK^\top/\sqrt{d})V = P \cdot \mathrm{Attn}(X)$。即 $\mathrm{Attn}(PX) = P \cdot \mathrm{Attn}(X)$。
</details>

<details>
<summary>Q6. Attention 和 Nadaraya-Watson 核回归的关系？</summary>

Nadaraya-Watson：$\hat f(q) = \sum_i \dfrac{K(q, k_i)}{\sum_j K(q, k_j)} v_i$，是固定核 $K$ 的相似度加权平均。

Attention = (1) 把核换成 $\exp(q \cdot k / \sqrt{d})$ 的学习核；(2) $q, k, v$ 都由输入线性投影得到（可学习）；(3) 批量化。本质就是 "可微分、可学习的 NW 核平滑器"。
</details>

<details>
<summary>Q7. 为什么不能 "softmax 之后再乘 mask"？</summary>

假设某位置要屏蔽。softmax 在归一化时已经把它的权重和其他位置一起分配；softmax 之后再乘 0，该位置权重确实变 0，但**其他位置的权重总和会小于 1**，相当于把那部分权重 "废了"，而不是重新分给有效位置。正确做法是 softmax 之前把 score 设为 $-\infty$，$e^{-\infty}=0$，其他位置自动获得全部归一化权重。
</details>

<details>
<summary>Q8. 本周 toy 任务为什么不加 PE 也能学？这是个 bug 还是教学选择？</summary>

教学选择。任务 "是否存在前缀和 > 50" 对非负 token 等价于 "总和是否 > 50"（因为 cumsum 是单调的），于是对顺序不敏感，mean-pool + attention 的求和能力足以完成。Week 6 会给一个真正需要顺序的版本（或者直接做 "最后一位是否 > 0" 这种强顺序依赖任务），让 PE 消融展现出戏剧化差别。
</details>

---

## 7. 延伸阅读

1. **Vaswani et al., "Attention Is All You Need" — Section 3.2 (2017)** — <https://arxiv.org/abs/1706.03762>
   现在读：精读一次，对照本周代码把 "Scaled Dot-Product Attention" 和 "Multi-Head Attention" 两节啃透。Multi-Head 先不用理解，Week 6 会细讲。

2. **Jay Alammar, "The Illustrated Transformer"** — <https://jalammar.github.io/illustrated-transformer/>
   现在读：看到 "Self-Attention in Detail" 一节就停，配本周可视化一起看。他画的 Q/K/V 动图能把你的直觉锚定到矩阵形状上。

3. **d2l Ch 10.2-10.3 "注意力机制 / Nadaraya-Watson"** — <https://zh.d2l.ai/chapter_attention-mechanisms/>
   现在读：李沐从 NW 核回归讲起引出 attention，对应本周 2.1 节的思路。他会先做一个 "固定核" 版本再做 "学习核" 版本，非常适合建立直觉连续性。

4. **Tsai et al., "Transformer Dissection: An Unified Understanding for Transformer's Attention via the Lens of Kernel" (EMNLP 2019)** — <https://arxiv.org/abs/1908.11775>
   现在读：把 attention 严格地写成 kernel smoother 家族的一员，给 2.6 节的类比做数学背书。如果你读论文有点吃力可以先跳过，Week 11 再回来。

5. **Jain & Wallace, "Attention is not Explanation" (NAACL 2019)** — <https://arxiv.org/abs/1902.10186>
   现在读：**在你开始 "用 attention 权重讲故事" 之前**，先读这篇泼冷水。他们证明多组不同的 attention 权重可以给出相同预测，所以权重作为 "解释" 需要极度谨慎。本周的可视化用于 debug 没问题，但别过度解读。
