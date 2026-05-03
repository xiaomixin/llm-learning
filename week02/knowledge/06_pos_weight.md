# 06 — `pos_weight` 处理类别不平衡

> **一句话口诀**:`pos_weight = w` 让正样本对 loss 的贡献(以及回传的梯度)**放大 w 倍**,相当于把"少数类的声音"调大。

---

## 1. 先跑起来(`../snippets/06_pos_weight.py`)

脚本合成 1:100 不平衡数据,比较:
- 不加 `pos_weight`
- `pos_weight = neg/pos`(推荐)
- `pos_weight = 10 × neg/pos`(太大)

再手动验证梯度确实放大了。

```bash
python snippets/06_pos_weight.py
```

期望看到 AUC-PR:**no_pw < too_big_pw < suggested_pw**,并且梯度乘以正好等于 `pos_weight`。

---

## 2. 发生了什么(白话)

极不平衡数据(99.83% 负 vs 0.17% 正)下,**"全部预测 0" 的准确率就是 99.83%**。loss 函数不加干预的话,模型会发现"只要押 0,loss 就很低"——它根本懒得去学正样本。

两种干预思路:
1. **改数据**:下采样/上采样/SMOTE,让两边变 1:1。
2. **改 loss**:正样本那一项在 loss 里"加倍计分",梯度回传时也跟着放大——这就是 `pos_weight`。

`pos_weight` 的好处:**不改一行数据,只改一个 loss 函数的参数**。最轻量、最干净,应当优先尝试。

---

## 3. 多角度理解

### 视角 A:loss 公式的变化

普通 BCE:
$$\mathcal{L}(z, y) = -y \log \sigma(z) - (1-y) \log(1-\sigma(z))$$

加权 BCE:
$$\mathcal{L}_w(z, y) = -\underbrace{w}_{\text{正样本才乘}} \cdot y \log \sigma(z) - (1-y) \log(1-\sigma(z))$$

正样本的 loss 被放大 w 倍,负样本不变。

### 视角 B:梯度上的效果

对 logit $z$ 求导:
$$\frac{\partial \mathcal{L}_w}{\partial z} = \begin{cases} w \cdot (\sigma(z) - 1) & y = 1 \\ \sigma(z) & y = 0 \end{cases}$$

**口诀**:`pos_weight=w` 把正样本的梯度**精确放大 w 倍**,负样本梯度不动。训练出的效果相当于"每看到 1 个正样本,等于看到 w 个正样本"。

### 视角 C:和其他方法比

| 方法 | 改什么 | 优点 | 风险 |
|------|-------|------|------|
| `pos_weight` | loss 函数 | 干净、不改数据、1 个超参 | 极度不平衡时 w 太大会让模型过度矫正 |
| 下采样 | 训练数据 | 训练快 | 丢信息、方差大 |
| SMOTE 上采样 | 训练数据 | 样本增多 | PCA 空间合成点可能"物理不合理" |
| Focal Loss | loss 函数 | 还能区分难易样本 | 多 1 个超参 γ,可能过拟合噪声 |

**Week 2 优先级**:`pos_weight` > Focal > SMOTE。先上最简单的。

### 视角 D:w 选多大合适?

经典选法:$w = N_{\text{neg}} / N_{\text{pos}}$,让**正负样本对 loss 的总贡献相等**。

Week 2 credit card 数据:$w \approx 577$。这个值**不需要调**——是一个数据本身的"比例统计量"。真要微调的话:在 `{0.5w, w, 2w}` 三个点用 val AUC-PR 对比即可。太大(比如 `10w`)会让模型对负样本几乎不更新,precision 崩盘。

---

## 4. 公式慢推

**正样本梯度放大 w 倍的推导**(y = 1):

$$\mathcal{L}_w = -w \log \sigma(z)$$

对 $z$ 求导(用 $\sigma'(z) = \sigma(z)(1 - \sigma(z))$):

$$\frac{\partial \mathcal{L}_w}{\partial z} = -w \cdot \frac{\sigma'(z)}{\sigma(z)} = -w \cdot (1 - \sigma(z)) = w \cdot (\sigma(z) - 1)$$

对比普通 BCE 下 y = 1 的梯度是 $\sigma(z) - 1$,确实**刚好放大 w 倍**。

**为什么选 $w = N_{\text{neg}}/N_{\text{pos}}$**:一个 batch 里期望梯度贡献:
- 正样本方:$N_{\text{pos}} \cdot w \cdot \mathbb{E}[(\sigma(z_{\text{pos}}) - 1)]$
- 负样本方:$N_{\text{neg}} \cdot \mathbb{E}[\sigma(z_{\text{neg}})]$

取 $w = N_{\text{neg}}/N_{\text{pos}}$ 让两边的**样本数权重**相等——就像"对半 balance"的 loss,但不用改数据。

---

## 5. 一个坑

**类型 / shape / device 三件套**:

```python
pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
```

- **必须是 tensor**,不能是 Python float(内部要广播乘法)。
- **shape = (1,)** 而不是标量——文档要求"match output shape",1 维输出对应 `(1,)`。
- **device 要和 logits 一致**——不一致 forward 会报 `RuntimeError`。

**三者任一搞错,运行时会报错或结果偷偷不对**。cell 12 三件都处理了。

---

## 6. 与 notebook 的连接

- `02_mlp_baseline.ipynb` cell 12 就是这个逻辑。算出 `pos_weight ≈ 577`。
- cell 17 第一轮训练会用这个 loss;没它你的 val AUC-PR 大概率卡在 0.2 以下。

**下一站**:[07_focal_loss.md](07_focal_loss.md) —— `pos_weight` 对所有正样本一视同仁。能不能让"已经分对了的容易样本"权重更低?
