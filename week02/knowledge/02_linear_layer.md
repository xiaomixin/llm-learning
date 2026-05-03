# 02 — 线性层(`nn.Linear`)

> **一句话口诀**:`Linear(in, out)` 就是 $y = Wx + b$,一次矩阵乘加偏置,没别的。

---

## 1. 先跑起来(`../snippets/02_linear.py`)

```python
import torch, torch.nn as nn

layer = nn.Linear(3, 2)           # 输入 3 维,输出 2 维
x = torch.tensor([1.0, 2.0, 3.0])
y = layer(x)                      # shape (2,)

# 和手写一致
y_manual = layer.weight @ x + layer.bias
assert torch.allclose(y, y_manual)
```

`W.shape = (out, in) = (2, 3)`,`b.shape = (out,) = (2,)`。

---

## 2. 发生了什么(白话)

`Linear(in=3, out=2)` 干三件事:

1. 造一个 `W` 矩阵,形状 `(2, 3)`——"输出 2 个,每个看 3 个输入"。
2. 造一个 `b` 向量,形状 `(2,)`——每个输出各一个偏置。
3. 前向时算 $y = Wx + b$。

`W` 和 `b` 都是 `nn.Parameter`,挂在 `layer.parameters()` 下,**会被 autograd 自动追踪**(参考 01)。训练时就是在更新这两个东西。

---

## 3. 多角度理解

### 视角 A:矩阵乘

一个样本 `(3,) → (2,)`:
$$y = Wx + b,\quad y_j = \sum_{i=1}^{3} W_{ji} x_i + b_j$$

一批样本 $X \in \mathbb{R}^{B \times 3}$ → $Y \in \mathbb{R}^{B \times 2}$:
$$Y = X W^{\top} + b$$

(注意是 `W^T`——PyTorch 约定 `W.shape = (out, in)`,批量做 `X @ W^T`。)

### 视角 B:学到的"特征投影"

把 3 维输入当"原始特征",2 维输出当"新特征"。`W` 的每一行是一个"滤波器":
- `W[0]` 看哪些输入、怎么加权,给你第 1 个新特征
- `W[1]` 看哪些输入、怎么加权,给你第 2 个新特征

训练的目的就是找到让 loss 最小的那组 `(W, b)`。

### 视角 C:人话版

"输出的每个数 = 把所有输入加权求和,再加一个偏移"。
初中听得懂的那种加权平均,只是权重是学出来的。

---

## 4. 公式慢推

单个输入 $x \in \mathbb{R}^n$,`Linear(n, m)` 输出 $y \in \mathbb{R}^m$:

$$y_j = \sum_{i=1}^{n} W_{ji} \cdot x_i + b_j,\quad j = 1, 2, \dots, m$$

每个符号是谁:

| 符号 | 含义 | 形状 |
|------|------|------|
| $x_i$ | 输入第 $i$ 维的值 | 标量 |
| $W_{ji}$ | "第 $j$ 个输出"对"第 $i$ 个输入"的权重 | 标量 |
| $b_j$ | "第 $j$ 个输出"的偏置 | 标量 |
| $y_j$ | 第 $j$ 个输出 | 标量 |

**参数总量** = $m \times n + m$ 个浮点数。
对 Week 2 的 MLP: `Linear(30, 64)` → $30 \times 64 + 64 = 1984$ 个参数。

---

## 5. 一个坑

**shape 反了**:用户经常写 `W.shape = (in, out)` 然后疑惑结果不对。PyTorch 是 `(out, in)`,批量形式是 `Y = X @ W.T`,少了那个转置就炸:

```python
X = torch.randn(5, 3)          # (batch, in)
W = torch.randn(2, 3)          # (out, in)
Y = X @ W                      # ✗ shape 不合: (5,3) @ (2,3) 根本乘不了
Y = X @ W.T                    # ✓ (5,3) @ (3,2) = (5,2)
```

用 `nn.Linear` 的话 PyTorch 帮你处理转置,你只要传 `(B, in)` 输入就能拿 `(B, out)` 输出。**自己手算对照时才需要想起这层转置**。

---

## 6. 与 notebook 的连接

- `02_mlp_baseline.ipynb` cell 10 的 `nn.Linear(in_dim, hidden)` 就是本文讲的东西。
- MLP 里多个 Linear 串起来,中间插 ReLU 和 Dropout,就是下一节的内容。

**下一站**:[03_mlp_build.md](03_mlp_build.md) —— Linear + Linear = 还是 Linear,为啥必须塞激活函数。
