# 03 — 搭一个 MLP

> **一句话口诀**:Linear + Linear 还是 Linear,中间必须塞一个**非线性激活**(比如 ReLU)才能学复杂函数。

---

## 1. 先跑起来(`../snippets/03_mlp.py`)

脚本会:
1. 证明两个 `Linear` 不加激活 = 一个等价 `Linear`;
2. 展示 ReLU 的"折线效应";
3. 建好 Week 2 用的 MLP 并数参数。

```bash
python snippets/03_mlp.py
```

期望看到:
```
two Linears w/o activation == one equivalent Linear?  True
  -> stacking without a nonlinearity gains you NOTHING.

Week-2 MLP
  input : torch.Size([8, 30])
  output: torch.Size([8])   (logits, before sigmoid)
  params: 6209
```

---

## 2. 发生了什么(白话)

- 一个 `Linear` 能表达的函数就是**仿射**(直线/平面)。
- 两个 `Linear` 叠一起,数学上能折叠成**另一个**仿射——等于白叠。
- 塞个 ReLU 进去(`ReLU(x) = max(x, 0)`),就有"**折线**"了——负数变 0,正数保留。折线够多,就能拼成任意复杂的形状(万能逼近定理的直觉版)。

所以"深度学习"的深度要有意义,**中间必须有非线性**。

---

## 3. 多角度理解

### 视角 A:代数证明

两个 Linear 堆叠:
$$y = W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2) = W' x + b'$$

**就是一个 Linear**。要打破这个坍缩,只能在 $W_1 x + b_1$ 之后加一个**不是矩阵乘法**的函数——激活函数。

### 视角 B:几何

- 一个 Linear 把空间做一次"线性变换 + 平移"——**保持直线还是直线**。
- ReLU 把空间"沿超平面折一下"——负半边被压到零。
- 反复 "变换 → 折 → 变换 → 折",就能把输入空间揉成任意形状的决策边界。

### 视角 C:工程视角

写 `nn.Sequential(Linear, ReLU, Linear)` 是"先做一次线性投影,再把负的信号滤掉,再做一次线性投影"。每一次"滤"都加一点非线性能力。

### 视角 D:Dropout 的位置

Week 2 MLP:
```
Linear(30→64) → ReLU → Dropout(0.3) → Linear(64→64) → ReLU → Dropout(0.3) → Linear(64→1)
```

Dropout 放在 ReLU 后、下一层 Linear 前,意思是"训练时随机遮盖 30% 的激活值,让网络不能依赖单一神经元"。`model.eval()` 切换到 eval 模式后 Dropout 自动关。

---

## 4. 公式慢推

Week 2 的 MLP:

$$h_1 = \text{ReLU}(W_1 x + b_1) \in \mathbb{R}^{64}$$
$$h_2 = \text{ReLU}(W_2 h_1 + b_2) \in \mathbb{R}^{64}$$
$$\hat{z} = W_3 h_2 + b_3 \in \mathbb{R}^1$$

每层维度:
- $x \in \mathbb{R}^{30}$ (V1–V28 + Time + Amount)
- $W_1 \in \mathbb{R}^{64 \times 30}$, $b_1 \in \mathbb{R}^{64}$ → 参数 $30 \cdot 64 + 64 = 1984$
- $W_2 \in \mathbb{R}^{64 \times 64}$, $b_2 \in \mathbb{R}^{64}$ → 参数 $64 \cdot 64 + 64 = 4160$
- $W_3 \in \mathbb{R}^{1 \times 64}$, $b_3 \in \mathbb{R}^{1}$ → 参数 $64 + 1 = 65$
- **总计 6209**(和脚本输出吻合)

输出 $\hat{z}$ 是 **logit**——不过 sigmoid。loss 函数会在内部做 sigmoid(参见 [04](04_bce_loss.md))。

---

## 5. 一个坑

**`.squeeze(-1)` vs `.squeeze()`**:

```python
return self.net(x).squeeze(-1)
```

最后 `Linear(64, 1)` 输出 `(B, 1)`,想要 `(B,)`。用 `.squeeze(-1)` 只压最后一维。

**为什么不用 `.squeeze()`**:当 batch size 恰好为 1,输出是 `(1, 1)`,无参数的 `.squeeze()` 会压成标量 `()`。loss 函数对齐 target `(1,)` 时就炸了。带参数的 `.squeeze(-1)` 永远只压最后一维,安全。

---

## 6. 与 notebook 的连接

- 这正是 cell 10 的 `class MLP`。
- 6209 参数也能在 cell 10 末尾的 `sum(p.numel() for p in model.parameters())` 里对上。

**下一站**:[04_bce_loss.md](04_bce_loss.md) —— 模型输出 logit 了,下一步怎么算 loss?为啥不直接 `Sigmoid + BCELoss`?
