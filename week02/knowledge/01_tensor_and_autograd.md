# 01 — 张量与 autograd

> **一句话口诀**:tensor 是"会记得自己从哪儿来的 numpy 数组";`.backward()` 自动沿路回溯,把 `dy/dx` 填进每个叶子的 `.grad`。

---

## 1. 先跑起来(`../snippets/01_tensor.py`)

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()     # y = 1 + 4 + 9 = 14
y.backward()           # 自动算 dy/dx
print(x.grad)          # tensor([2., 4., 6.])
```

命令行:
```bash
python snippets/01_tensor.py
```

---

## 2. 发生了什么(白话)

- 你手写的是 $y = x_1^2 + x_2^2 + x_3^2$。
- 数学上 $\frac{\partial y}{\partial x_i} = 2x_i$,所以 `x=[1,2,3]` 时梯度应该是 `[2,4,6]`。
- **你没写这个导数**。PyTorch 自己推出来了。它是怎么做到的?

答:每次你对 tensor 做运算(`**`、`sum`、`+`…),PyTorch 在结果上挂一个 **`.grad_fn`**——"我是谁生的"。一串运算下来就是一条链。`.backward()` 沿着这条链,一段一段用**链式法则**反推,把每个叶子节点的梯度填到 `.grad` 里。

这套机制叫 **autograd**(automatic differentiation)。

---

## 3. 多角度理解

### 视角 A:numpy + 身世

```python
# numpy 只有数据
a = np.array([1.0, 2.0, 3.0])
# torch 多记一份身世
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x**2).sum()
print(y.grad_fn)  # <SumBackward0>  <- 我是 sum 生的
```

`grad_fn` 是"家谱的最后一环"。反向时 PyTorch 沿着家谱一层一层爬回去。

### 视角 B:计算图(Computational Graph)

```
x ── (x**2) ── (sum) ── y
```

每个节点知道"给我下游的梯度,我能算出给上游的梯度"。反向是信息沿箭头反向流动。

### 视角 C:为什么 `.grad` 是**累加**而不是**覆盖**

早期设计(RNN/BPTT)需要同一个参数被用于多个时间步,梯度要**累加**。
代价:每次 step 前你必须手动 `zero_grad()`,否则上一次的梯度会叠进来。

```python
y = (x**2).sum(); y.backward()  # x.grad = [2,4,6]
y = (x**2).sum(); y.backward()  # x.grad = [4,8,12]  <- 没清零就翻倍
```

---

## 4. 公式慢推

对 $y = \sum_i x_i^2$:

$$\frac{\partial y}{\partial x_j} = \frac{\partial}{\partial x_j}\Big(x_1^2 + x_2^2 + x_3^2\Big) = 2x_j$$

链式法则一般形式——如果 $y$ 经过中间变量 $u$:

$$y = f(u),\quad u = g(x) \quad\Rightarrow\quad \frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$$

**口诀**:上游给下游的梯度,乘上下游对自己输入的偏导,就是再往上传的梯度。
**autograd 做的就是这件事,一路做到叶子**。

---

## 5. 一个坑

**叶子 vs 非叶子**:只有 `requires_grad=True` 的、**用户直接创建**的 tensor 是"叶子",`.grad` 会被填。中间结果(比如 `y = x**2`)默认不保留 `.grad`。

```python
x = torch.tensor([1.0], requires_grad=True)
u = x * 2
y = u ** 2
y.backward()
print(x.grad)   # tensor([8.])  ← 叶子,有
print(u.grad)   # None         ← 中间结果,默认无
```

想看中间的 `.grad` 要显式 `u.retain_grad()`。训练时你只关心参数(叶子),所以这事儿通常不碰。

---

## 6. 与 notebook 的连接

- `02_mlp_baseline.ipynb` cell 16 里每行 `loss.backward()` 干的就是这套。
- `optimizer.zero_grad()` = 前面 §3 视角 C 说的"清累加"。
- 你训练时看到的 `.grad` 全是自动算出来的——这正是深度学习的"底层黑科技",搞懂了以后所有训练循环都不神秘。

**下一站**:[02_linear_layer.md](02_linear_layer.md) —— 有了 autograd,`nn.Linear(30, 64)` 这样的模块就很容易理解:它就是一堆参数,所有梯度都由 autograd 填。
