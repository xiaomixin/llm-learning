# 04 — BCE 损失 + 数值稳定

> **一句话口诀**:`BCEWithLogitsLoss` = BCE 的**不会炸版本**。极端 logit 下 `Sigmoid + BCELoss` 会给你 `inf` 或 `nan`,这个不会。

---

## 1. 先跑起来(`../snippets/04_bce.py`)

```bash
python snippets/04_bce.py
```

期望输出(节选):
```
logit z | y | naive BCE           | stable BCE
--------+---+---------------------+-----------
 -50.0  | 1 | ERROR: math domain  | 50.000000
 +50.0  | 0 | ERROR: math domain  | 50.000000
```

朴素公式在极端值下直接抛 `log(0)`;稳定版给你合理的"大 loss"。

---

## 2. 发生了什么(白话)

二元交叉熵的定义很直观——**猜对了 loss ≈ 0,猜反了 loss 很大**:

$$\mathcal{L}_{\text{BCE}}(p, y) = -y \log p - (1-y) \log(1-p)$$

其中 $p = \sigma(z)$ 是模型输出的概率。

问题出在**怎么算这个公式**。天真实现:
1. 先算 $p = \sigma(z) = \frac{1}{1+e^{-z}}$
2. 再算 $\log p$ 或 $\log(1-p)$

当 $z = 50$ 时 $p \approx 1 - 10^{-22}$,float32 直接截断成 `1.0`,然后 $\log 1 = 0$——对 y=0 的样本你想拿到的是"巨大 loss 惩罚这个错得离谱的预测",结果拿到 `-log(1-1) = -log(0) = inf`。反向更是 nan。

**`BCEWithLogitsLoss` 的做法**:直接从 logit $z$ 算,绕开 $\sigma(z)$ 的舍入,用 log-sum-exp 技巧让中间值永远在安全区间。

---

## 3. 多角度理解

### 视角 A:信息论

$-\log p$ 是"这件事发生了,但模型只给了 $p$ 的概率——我要这么多 bits 去编码这个意外"。越意外 loss 越高。完美预测 $p = 1$ → $-\log 1 = 0$,没损失。

### 视角 B:惩罚曲线

- y = 1 时,loss = $-\log p$。$p \to 1$ → loss → 0;$p \to 0$ → loss → $+\infty$。
- y = 0 时,loss = $-\log(1-p)$。对称的。

这就是"猜反了惩罚急剧增大"的数学来源。

### 视角 C:稳定公式从哪儿来

把两种情况合并:
$$\mathcal{L} = -[y \log \sigma(z) + (1-y) \log(1-\sigma(z))]$$

把 $\log \sigma(z) = -\log(1+e^{-z})$ 代入,合并整理(见 §4 推导),得到:

$$\boxed{\mathcal{L}_{\text{BCE}}(z, y) = \max(z, 0) - z \cdot y + \log\!\big(1 + e^{-|z|}\big)}$$

里头再也没有 $\sigma(z)$ 了。三项都是稳定的:
- $\max(z, 0) - z y$:简单加减,不溢出。
- $\log(1 + e^{-|z|})$:指数参数 $-|z| \le 0$,$e^{-|z|} \in (0, 1]$,结果在 $[0, \log 2]$。

**口诀**:`max(z,0) - z·y + log(1 + e^{-|z|})`——三段拼成一个永远不爆的 BCE。

---

## 4. 公式慢推(想看细节再读)

设 y = 1:
$$-\log \sigma(z) = -\log \frac{1}{1+e^{-z}} = \log(1 + e^{-z})$$

当 $z < 0$ 时 $e^{-z}$ 可能巨大,rewrite:
$$\log(1 + e^{-z}) = \log\!\big(e^{-z}(e^z + 1)\big) = -z + \log(1 + e^z)$$

合并成对称形式:
$$\log(1 + e^{-z}) = \max(z, 0) + \log(1 + e^{-|z|}) - z \cdot \mathbb{1}[z < 0]$$

同样处理 y = 0 的情况,再统一公式,就得到 §3 视角 C 的 boxed 等式。这正是 PyTorch 源码里的实现。

**你不需要记推导,记结论就够**:`BCEWithLogitsLoss = 接 logit,内部稳`。

---

## 5. 一个坑

**把概率传进 `BCEWithLogitsLoss`**:

```python
probs = torch.sigmoid(logits)
loss = nn.BCEWithLogitsLoss()(probs, targets)   # ✗ 错了!probs 还会被 sigmoid 一次
```

这个不会报错——因为 `probs` 也是合法的 tensor,数值上被"二次 sigmoid",loss 偏低、训练扭曲。

**正确做法**:模型最后**不要**接 `nn.Sigmoid`,直接输出 logit;损失函数用 `BCEWithLogitsLoss`(内部会算 sigmoid + log-sum-exp);**要概率时**(预测/画 PR 曲线)再手动 `torch.sigmoid(logits)`。

---

## 6. 与 notebook 的连接

- cell 10 的 `MLP.forward` **不接 `nn.Sigmoid`** —— 返回的是 logit。
- cell 12 用 `nn.BCEWithLogitsLoss(pos_weight=...)` 作为损失函数。
- cell 14 的 `predict_scores` 里才手动 `torch.sigmoid(logits)`,把 logit 转成 0–1 概率用于 PR 曲线。

**下一站**:[05_training_loop.md](05_training_loop.md) —— 有模型、有 loss 了,怎么把参数更新起来?5 步训练循环上场。
