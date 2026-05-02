# Week 4 深度知识伴读 — RNN / LSTM / GRU 与梯度消失

> 本周的定位：Transformer 之前，最后一次严肃地对 **循环计算** 做数学审视。所有 "Attention 为什么更好" 的论据，都需要你先理解 RNN 的痛点是 **如何产生的**、以及 LSTM 的门控是 **为什么有效**。这份文档是对 `04_lstm_baseline.ipynb` 的理论陪读。

---

## 1. 本周要回答的核心问题

1. **为什么 vanilla RNN 在长序列上几乎一定训练失败？** 能不能从 BPTT 的 Jacobian 连乘里写出 "指数衰减/爆炸" 的数学形式？
2. **LSTM 的 4 个门 (i, f, o, g) 各自干什么？** cell state 的 **加性更新** 相比 RNN 的 **乘性递推**，在梯度层面到底改了什么？
3. **GRU 是如何把 LSTM "压缩" 的？** 合并哪些门、为什么在很多任务上效果相当？
4. **`pack_padded_sequence` 到底在解决什么问题？** 本任务定长为什么不需要，Week 7+ 为什么一定会用到？
5. **取 `out[:, -1, :]` 作为分类表征是把信息 "压缩"，Attention 是把信息 "加权保留" —— 这两种聚合方式的本质差异在哪？**

---

## 2. 理论骨架

### 2.1 Vanilla RNN 与 BPTT 的 Jacobian 连乘

一个最朴素的 RNN 单元：

$$
h_t = \tanh(W_{hh}\, h_{t-1} + W_{xh}\, x_t + b), \qquad y_t = W_{hy}\, h_t
$$

其中 $h_t \in \mathbb{R}^d$ 是隐状态，$W_{hh}$ 是循环权重矩阵。训练时我们把时间展开，对最终损失 $L = \sum_t \ell_t$ 做反向传播 (BPTT)。关心 **某一远处 loss 对更早 hidden 的梯度**：

$$
\frac{\partial L}{\partial h_k} = \sum_{t\ge k} \frac{\partial \ell_t}{\partial h_t}\, \prod_{s=k+1}^{t} \frac{\partial h_s}{\partial h_{s-1}}
$$

而逐步 Jacobian 为：

$$
\frac{\partial h_s}{\partial h_{s-1}} = \mathrm{diag}\!\left(\tanh'(\cdot)\right)\, W_{hh}
$$

把这个 Jacobian 记作 $J_s$，则跨越 $T$ 步的梯度范数被 $\prod_{s} J_s$ 控制。对这个矩阵做谱分解，设 $\lambda_{\max}$ 是 $W_{hh}$ 的最大奇异值：

- 如果 $\lambda_{\max} \cdot \max|\tanh'| < 1$（注意 $\max|\tanh'|=1$ 且在零点取到），乘积 **指数衰减** → 梯度消失，模型看不到远距离信息。
- 如果 $\lambda_{\max} > 1$，乘积可能 **指数放大** → 梯度爆炸，loss 振荡甚至 NaN。

这就是为什么 RNN 在长序列上要么 "看不见过去"，要么 "炸给你看"。梯度裁剪（`clip_grad_norm_`）只治爆炸，不治消失；治消失要靠结构改造。

**直觉一句话**：RNN 的信息通路是 **乘法链**，只要参与连乘的矩阵不是 "接近正交且激活不饱和"，就会雪崩。

### 2.2 LSTM 四门的数学 —— 加性记忆通路

LSTM 把 hidden state 拆成两条路径：**cell state $c_t$** 作为 "长期记忆"，**hidden state $h_t$** 作为 "对外输出"。所有输入先算 4 个门（忘记、输入、候选、输出），再组合：

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f)  &&\text{forget gate，决定 } c_{t-1} \text{ 留多少} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i)  &&\text{input gate，决定新候选写多少} \\
g_t &= \tanh(W_g [h_{t-1}, x_t] + b_g)   &&\text{候选 cell（新内容）} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o)  &&\text{output gate，决定 } c_t \text{ 怎么暴露为 } h_t \\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t &&\text{加性更新！} \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

**关键句：cell state 的更新是 `f·c + i·g`，是加法，不是矩阵乘法。**

这条通路的 Jacobian：

$$
\frac{\partial c_t}{\partial c_{t-1}} = \mathrm{diag}(f_t)
$$

如果 forget gate 接近 1（实践中一般给 $b_f$ 正初始化偏置 1 或 2 来促成），那么 $\prod_s \mathrm{diag}(f_s) \approx I$，梯度沿 cell state 通道 **几乎恒等传播**。vanilla RNN 的 $W_{hh}$ 连乘问题被绕开了。

这就是 LSTM "缓解" 梯度消失的数学本质：**用一条加性、受门控保护的通路替换矩阵连乘通路**。注意是 "缓解" 不是 "消除"，门如果长期取 0，信息一样会丢。

### 2.3 GRU：把 4 门压成 2 门

GRU 做了两处合并：
1. **合并 cell 与 hidden**：不再维护独立的 $c_t$，直接在 $h_t$ 上做加性更新。
2. **合并 input gate 与 forget gate**：用一个 "update gate" $z_t$ 决定新旧信息的比例（1-z 留旧，z 留新），天然互补。

公式：

$$
\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}, x_t])  &&\text{update gate} \\
r_t &= \sigma(W_r [h_{t-1}, x_t])  &&\text{reset gate，控制候选 } \tilde h \text{ 看多少旧 } h \\
\tilde h_t &= \tanh(W [r_t \odot h_{t-1}, x_t]) \\
h_t &= (1-z_t) \odot h_{t-1} + z_t \odot \tilde h_t
\end{aligned}
$$

GRU 参数量是 LSTM 的约 3/4，训练速度更快。经验上两者在多数任务上差异小于 1 个百分点；但在超长依赖 (seq > 500) 或需要精细 "擦除/写入" 控制的任务（如语言建模大模型时代的部分 ablation）上，LSTM 略稳一些。这就是为什么 Transformer 之前的工业界长期留着 LSTM 做主力。

### 2.4 变长序列与 pack_padded_sequence

在异常检测里，"用户最近 L 笔交易" 经常是变长的（新用户可能只有 5 笔）。做法一般是 pad 到固定长度 $L=32$，但 pad 的 0 向量会 **进入 LSTM 的计算**，污染隐状态 —— 特别是最后时刻，`out[:, -1, :]` 可能是对 pad 的响应而不是对真实末笔的响应。

PyTorch 的 `pack_padded_sequence` 把 padding 剔除掉，LSTM 内部逐 step 只对 "当前还存活" 的样本做矩阵乘：

```python
packed = nn.utils.rnn.pack_padded_sequence(z, lengths.cpu(),
                                           batch_first=True, enforce_sorted=False)
out, (h, c) = self.lstm(packed)
out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
```

配合 pack 的时候，**最后时刻的 hidden** 应该从 `h[-1]`（最后一层的 final hidden）取，而不是 `out[:, -1, :]`（因为 out 在 pad 位置是 0）。本周的 notebook 选择了定长 $L=32$，因此不需要 pack；但代码里保留了注释模板，Week 7+ 处理真实变长交易序列会派上用场。

**为什么这事不能用 mask 凑合**：mask 只能在最后聚合时乘 0，但 LSTM 中间每一步的隐状态已经被 pad 污染；等算完再乘 mask 已经来不及。Attention 场景下 mask 能在 softmax 时彻底屏蔽（score 置 -∞），所以 Week 5+ 不再需要 pack。

### 2.5 Bidirectional 的利弊 —— 因果性陷阱

双向 LSTM 把序列正着过一遍、反着过一遍，把两个方向的 hidden 拼起来作为表征。在 NER、情感分类这种 "看完整句子再决策" 的任务上非常好用。

**但对异常检测场景要格外警惕**：如果你的任务是 "判断第 t 笔是否欺诈"，而你用双向模型喂它 $x_1, \dots, x_L$（其中 t < L），反向路径会把 **第 t 笔之后的真实交易信息** 回传到 t 的表征里。训练集里这没事，**线上决策时你拿不到未来**，于是出现了典型的 "训练指标漂亮、上线崩盘" 的数据泄露。

本周的 notebook 做的是 "整个窗口是否含异常"，标签粒度是窗口级，不是时刻级，所以用单向 LSTM 没有因果性问题；但一旦任务改成 "当前最后一笔分类"，bidirectional 必须换成单向或者 causal mask 的架构。

### 2.6 从 RNN 到 Attention 的信息论视角

两种聚合信息的极端：

| 维度 | RNN / LSTM | Self-Attention |
|------|-----------|----------------|
| 任意两位置路径长度 | $O(L)$（信息要顺着隐状态爬过去） | $O(1)$ |
| 并行度 | 时间轴强串行 | 时间轴完全并行 |
| 每步计算量 | $O(d^2)$ | $O(L \cdot d)$ |
| 总计算 | $O(L \cdot d^2)$ | $O(L^2 \cdot d)$ |
| 归纳偏置 | 强（近因、马尔可夫） | 极弱（需 PE 补） |

两个观察：
- 当 $L \ll d$（例如金融序列 $L=32, d=128$），attention 的 $L^2$ 根本不是瓶颈，RNN 的 $d^2$ 反而更贵。
- RNN 的 "信息传递路径长" 意味着：就算 LSTM 缓解了梯度消失，**远距离依赖的信号要穿过 32 个 tanh 非线性**，仍然会被稀释。这是 Transformer 在 "第 3 笔和第 30 笔之间的关系" 这类任务上压倒性胜出的根本原因。

---

## 3. 代码对照 —— 04_lstm_baseline.ipynb

### 3.1 模型结构（cell 8）

```python
class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden=128, proj=64, num_layers=2, p_drop=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj)
        self.lstm = nn.LSTM(input_size=proj, hidden_size=hidden,
                            num_layers=num_layers, dropout=0.2, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(hidden, 1))
```

三处设计选择值得细讲：

1. **`Linear(F → 64)` 输入投影**：合成数据有 7 维（`amount_log, hour_sin, hour_cos, lat, lng, cat, dev`），Kaggle 有 29 维。先投影到同一维度 64，使 LSTM hidden=128 对两个数据集等价，方便对比实验。这是一种典型的 "特征归一" 思路，在 MVP 里非常高性价比。
2. **`dropout=0.2` 只作用于 LSTM 层间，不作用于最后一层**：PyTorch 的 `nn.LSTM(dropout=p)` 是只在 **层与层之间**（num_layers > 1 时）应用的，最后一层没有 dropout。所以 head 前再加一个 `Dropout(0.3)` 是必要的，否则分类头前完全没有正则。
3. **取 `out[:, -1, :]`**：如第 2.4 节强调，这只在定长 + 无 pad 场景下是 "最后一笔的表征"。拿这个丢到 `head` 做二分类，是最朴素的 sequence-to-label。

### 3.2 训练循环（cell 10）

```python
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
...
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

- **`pos_weight = #neg / #pos`**：Kaggle 欺诈比例 ≈ 0.17%，正负样本比约 1:580。`BCEWithLogitsLoss` 的 `pos_weight` 只在正样本的 loss 项上乘一个系数 w，等价于 "一个正样本顶 w 个负样本"。直接加权比 SMOTE / undersample 更稳，没有合成噪声也没有丢信息。
- **`clip_grad_norm_=1.0`**：LSTM 比纯前馈更容易梯度爆炸（原因就是 2.1 节的 Jacobian 连乘，即使 LSTM 在 cell state 上缓解了，input gate、output gate 的路径依然有矩阵连乘）。裁剪到全局范数 1.0 是工业界经验值。
- **Early stopping by val AUC-PR**：监控的是 `val_ap` 而不是 loss。欺诈场景下，loss 继续下降但 AUC-PR 停滞是常态（过拟合负样本），用下游指标做 early stop 更稳。

### 3.3 合成 vs Kaggle 的对比哲学（cell 18 的复盘）

> 合成异常模式明显（金额 × 10~30），LSTM 轻松学到 → 反映 **模型上限**。
> Kaggle 是真实 PCA 特征 + 伪 user_id，难度接近上限 → 反映 **问题本身** 的困难。

这句话本身就是一条重要的 MLOps 经验：**合成数据是用来验证 "模型能力" 的，真实数据是用来度量 "问题可学性" 的，两者的 gap 就是特征工程和数据质量还能榨出的空间。** 不要指望单一模型同时在两者上都达到 95%，那通常意味着你评估有 bug。

---

## 4. 常见坑位与调试思维

1. **数据泄露**：Kaggle 信用卡数据按 `Time` 切分训练/验证，不能随机 shuffle。Week 3 的序列化里 "按 user_id + 时间排序 + 滑窗" 是第一道防线；训练/验证切分必须按时间（或按 user_id 整组），绝不能让同一用户的未来交易出现在训练集、过去交易出现在验证集。
2. **pad token 污染最后时刻**：当你把 notebook 扩展到变长序列却忘记用 `pack_padded_sequence`，`out[:, -1, :]` 可能是对 0 向量的响应。症状：val loss 看上去还行，但预测概率集中在某个中间值。修复：要么用 pack，要么取 `h[-1]`（final hidden），要么显式按 `lengths` 索引 `out[i, lengths[i]-1, :]`。
3. **`pos_weight` 设得过大**：把 w 调到 1000+ 会出现 "模型把所有样本都预测成正" 的平凡解，val_ap 反而崩掉。实践中 `#neg / #pos` 通常是安全的上限，再大就要配 focal loss 或 undersample。
4. **LSTM 初始化**：PyTorch 默认的 `nn.LSTM` 初始化对 forget gate 的 bias 不做特殊处理，默认是 0。可以手动加 `for name, p in lstm.named_parameters(): if 'bias' in name: n = p.size(0); p.data[n//4:n//2].fill_(1.)` 把 forget gate bias 初始化为 1 —— 这是 Jozefowicz et al. 2015 的经典经验，明显加速早期收敛。
5. **dropout 放错位置**：在 LSTM **内部时间轴** 上做 dropout（每个 time step 都 drop 不同维度）会破坏门的学习，PyTorch 的 `nn.LSTM(dropout=p)` 故意只在层间做。如果你自己实现 LSTMCell 循环，记得 dropout 用 **same mask across time**（variational dropout, Gal & Ghahramani 2016），或者干脆不要在循环内部做。
6. **`BCEWithLogitsLoss` vs `BCELoss + sigmoid`**：前者数值稳定（log-sum-exp trick），后者容易在极端 logit 时产生 NaN。notebook 里评估时也是 `torch.sigmoid(model(x))` 先算概率再做指标，训练时直接传 logit 给 loss，不要早早做 sigmoid。
7. **AUC-PR vs AUC-ROC**：在正样本 0.17% 的场景，AUC-ROC 对模型差异不敏感（因为 FPR 分母太大）。PR 曲线关注的是 "你预测的 positive 里有多少是真的"，更贴近业务。线上 alert 量受限时尤其看 `Recall@FPR=0.001`。

---

## 5. 与未来几周的连接

- **Week 5** 会直接对照 LSTM：同样的 toy 任务，single-head attention 会比 MLP/RNN 类模型更快收敛，并且 attention 权重可视化取代 "隐状态黑盒"。注意 Week 5 的 toy 是 "前缀和是否跨阈值"，是一个 **全局聚合** 任务，恰好是 RNN 的短板、Attention 的长板。
- **Week 6** 组装 EncoderBlock，你会看到 `out[:, -1, :]` 这种 "最后时刻 pooling" 被 `mean pool`（Week 6）或 `[CLS] token pool`（Week 7+）替换。设计选择的分歧都会在那里展开。
- **Week 9 的 Anomaly Transformer** 反过来又借鉴了 RNN 时代的 "时序先验" 思路：prior-association 用高斯衰减建模 "相邻位置更相关"，某种意义上把 RNN 的归纳偏置显式注回了 attention。理解 RNN 的归纳偏置是什么（强近因），才能理解为什么 attention 需要 "注回" 一些。
- **Week 11 的 PatchTST** 用 patch 化 + channel independence 再次降低 attention 的 $L^2$ 成本。你会发现：当 L 特别大时，vanilla attention 的优势（O(1) 路径）开始被计算成本拉回 RNN 的领域，于是有了各种稀疏/分块 attention 的回归。

---

## 6. 自测题

<details>
<summary>Q1. 为什么 vanilla RNN 在长序列上训练失败？请写出 BPTT 的梯度表达式，并指出衰减/爆炸的充分条件。</summary>

$\dfrac{\partial h_t}{\partial h_k} = \prod_{s=k+1}^{t} \mathrm{diag}(\tanh'(\cdot))\, W_{hh}$。

设 $W_{hh}$ 的最大奇异值为 $\sigma_{\max}$，$\max|\tanh'|=1$。若 $\sigma_{\max} < 1$，连乘的谱范数按 $\sigma_{\max}^{t-k}$ 指数衰减 → 梯度消失；若 $\sigma_{\max} > 1$ 且 $\tanh$ 不饱和，连乘可能指数放大 → 梯度爆炸。裁剪可治爆炸不治消失。
</details>

<details>
<summary>Q2. LSTM 的 cell state 更新为什么被称为 "加性"？这对梯度流动意味着什么？</summary>

$c_t = f_t \odot c_{t-1} + i_t \odot g_t$ 是加法而非矩阵乘法。求导 $\partial c_t / \partial c_{t-1} = \mathrm{diag}(f_t)$，只要 $f_t \approx 1$，就近似恒等。因此 cell state 路径上 **不存在矩阵连乘**，梯度能无损传递；hidden state 路径仍有矩阵乘，但至少有 cell state 这条 "高速公路" 兜底。
</details>

<details>
<summary>Q3. GRU 合并了 LSTM 的哪些门？为什么效果相当？</summary>

合并了 (a) cell 与 hidden（只留一条 state），(b) input gate 与 forget gate（合并为 update gate $z_t$，用 $(1-z, z)$ 互补）。参数量降到约 3/4，训练更快。效果相当的原因：LSTM 的 i 和 f 在实际训练中经常学成 "互补关系"（写多就留少），GRU 把这个强约束显式化了，相当于一种有益的归纳偏置。
</details>

<details>
<summary>Q4. 本周的 LSTM 为什么不用 pack_padded_sequence？什么时候必须用？</summary>

本周所有窗口定长 $L=32$，每个位置都是真实交易，无 padding，所以不需要。必须用的场景：序列长度不等、pad 到统一长度时。否则 LSTM 会在 pad 位置继续更新隐状态，污染后续输出；`out[:, -1, :]` 会是 "对 0 向量的响应" 而不是 "对最后真实 token 的响应"。
</details>

<details>
<summary>Q5. Bidirectional LSTM 在哪种异常检测任务下是禁区？</summary>

"根据前 L 笔判断当前（最后一笔）是否异常" 这种 **实时预测** 任务。双向 LSTM 的反向路径会把未来信息回传到当前位置，线上无法获取未来，训练和线上分布不一致，典型数据泄露。

如果任务是 "整段窗口是否含异常"（离线判定、滑后一段再看），bidirectional 没问题。
</details>

<details>
<summary>Q6. LSTM 最后时刻 pooling `out[:, -1, :]` 和 Attention pooling 的本质区别是什么？</summary>

`out[:, -1, :]` 是把所有历史信息 **压缩** 进一个长度固定的隐状态，信息损失不可逆，尤其远距离信号衰减严重。

Attention pooling 是对所有位置的 value **加权平均**，权重由 query-key 相似度决定；原始信息都保留在 value 里，每次分类可以 "按需取用" 不同位置。信息是 **保留 + 选择性聚焦**，不是压缩。
</details>

<details>
<summary>Q7. pos_weight 的直觉是什么？和 focal loss 有何互补？</summary>

pos_weight 让正样本的 loss 加权 w 倍，等价于 "一个正样本顶 w 个"。对类别不平衡有效，但对 "难例 vs 易例" 不区分 —— 一个大量的 "容易分对的负样本" 仍然会主导梯度。

focal loss 额外乘上 $(1-p_t)^\gamma$，对已经预测得很准的样本降权，把注意力转向难例。两者可叠加：`pos_weight` 处理不平衡，`focal` 处理难易分布。
</details>

<details>
<summary>Q8. 如果合成数据 AUC-PR = 0.99 但 Kaggle 只有 0.72，是好消息还是坏消息？</summary>

好消息 —— 说明 LSTM 的 **表达能力** 够用（合成上限已达），Kaggle 分数低反映的是 **问题本身** 的困难（PCA 特征脱敏、伪 user_id 引入噪声、标签稀疏）。下一步应该把精力投入特征工程、窗口标签设计、数据清洗，而不是换更大的模型。

反面例子：如果合成上也只有 0.80，那是模型结构/训练出了问题，换大模型才有意义。
</details>

---

## 7. 延伸阅读

1. **colah, "Understanding LSTM Networks"** — <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
   现在读：配合第 2.2 节的公式，把 colah 的门控示意图和数学严格对应起来；他的 "cell state 是一条 conveyor belt" 比喻是本周的理解锚点。

2. **Pascanu, Mikolov & Bengio, "On the difficulty of training recurrent neural networks" (2013)** — <https://arxiv.org/abs/1211.5063>
   现在读：第 3 章严格证明梯度爆炸/消失与 Jacobian 谱半径的关系。本周 2.1 节的数学就是从这里来的，看完原文你会对 "裁剪" 和 "正交初始化" 的来龙去脉彻底清楚。

3. **Jozefowicz, Zaremba & Sutskever, "An Empirical Exploration of RNN Architectures" (2015)** — ICML
   现在读：第 3.1 节解释为什么 forget gate bias 初始化为 1 会显著加速。本周 notebook 没做这个初始化，但你在扩展时可以试，观察收敛曲线差异。

4. **Gal & Ghahramani, "A Theoretically Grounded Application of Dropout in RNNs" (2016)** — NeurIPS
   现在读：讲清楚为什么 RNN 内部的 dropout 要 "时间共享 mask"，这是你自己手写 LSTM 循环时极易踩的坑。PyTorch 的 `nn.LSTM(dropout=)` 没实现 variational dropout，如果需要要自己写。

5. **Vaswani et al., "Attention Is All You Need" (2017) — Section 1 & 2** — <https://arxiv.org/abs/1706.03762>
   现在读：不看全文，只读 Introduction 和 Background。他们在这里明确对比了 RNN 的串行 + 长路径瓶颈，正好是本周 2.6 节的思想来源。Week 5 再回来精读 Section 3。
