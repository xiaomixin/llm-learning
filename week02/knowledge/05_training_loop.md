# 05 — 训练循环的 5 步

> **一句话口诀**:**清零 → 前向 → 算 loss → 反向 → 更新**。少一步,训练就出问题。

---

## 1. 先跑起来(`../snippets/05_train_loop.py`)

脚本用 1 参数 `Linear` 拟合 $y = 2x + 1$,20 个 epoch 就能收敛到 W≈2,b≈1。然后演示**不 `zero_grad()`** 会发生什么。

```bash
python snippets/05_train_loop.py
```

---

## 2. 发生了什么(白话)

一个训练 step 的数学定义:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(f_\theta(x), y)$$

用人话:**"看看现在参数给出的预测跟真值差多少,顺着差距修正参数一点点"**。

怎么在代码里实现这件事?PyTorch 把它拆成 5 步:

```python
optimizer.zero_grad()       # 1. 清上一步的梯度
logits = model(x)           # 2. 前向:算预测
loss = loss_fn(logits, y)   # 3. 算 loss(标量)
loss.backward()             # 4. 反向:autograd 填每个参数的 .grad
optimizer.step()            # 5. 按 .grad 更新参数
```

---

## 3. 多角度理解

### 视角 A:每一步干什么、读写谁

| 步骤 | 读 | 写 | 失败症状 |
|------|-----|-----|----------|
| `zero_grad()` | — | `param.grad = 0` | 漏 → 梯度越积越大,loss 震荡/发散 |
| `model(x)` | `param.data`, `x` | 构建计算图 | 漏 → 没 logits,loss 算不了 |
| `loss_fn(logits, y)` | logits, y | 标量 loss | 漏 → `backward()` 没东西可算 |
| `loss.backward()` | 计算图, `loss` | `param.grad` | 漏 → `.grad` 是 None,step 报错 |
| `optimizer.step()` | `param.grad` | `param.data` | 漏 → 参数永远不变,loss 不降 |

### 视角 B:为什么这个顺序不能换

- `backward` 要读 `loss` → 必须在 `loss_fn` 之后。
- `step` 要读 `.grad` → 必须在 `backward` 之后。
- `zero_grad` 清的是**上一步留下来的**梯度,不是这一步的,所以放在最前或 step 之后都行(约定放最前)。
- `clip_grad_norm_` 要改 `.grad`,**必须在 backward 之后、step 之前**——顺序反了等于白裁剪。

### 视角 C:后续扩展都是往这 5 步里插东西

- **学习率 warmup/schedule**:每 step 改 `optimizer.lr`——插在 step 之后。
- **梯度累积**(模拟大 batch):每 N 个小 batch 才 `step + zero_grad`——把"清零"从"每步"改成"每 N 步"。
- **Mixed precision**(AMP):forward/backward 在 fp16,step 时 unscale—包一层 `GradScaler`。
- **Transformer warmup 阶段**:clip → step 之间没改,只加外壳。

所以把这 5 步真正吃透,**后面所有训练调参都是它的 variation**。

---

## 4. 公式慢推

SGD 一步:
$$\theta_{t+1} = \theta_t - \eta \cdot g_t,\quad g_t = \nabla_\theta \mathcal{L}(f_\theta(x_t), y_t)$$

- $\theta_t$ = 参数当前值。在 PyTorch 里就是 `param.data`。
- $g_t$ = 当前 batch 下 loss 对参数的梯度。在 PyTorch 里就是 `param.grad`。
- $\eta$ = 学习率(lr)。常见值 `1e-3` 到 `1e-4`。

**Adam 一步**(Week 2 notebook 用的):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(梯度一阶动量)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(梯度二阶动量)}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t),\ \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$$

**口诀**:Adam = "用梯度的移动平均代替原始梯度,再按梯度方差自适应 scale"。
对本 MLP 来说,Adam 默认参数($\beta_1=0.9, \beta_2=0.999, \varepsilon=10^{-8}$)够用,不需要你调。

---

## 5. 一个坑

**`model.train()` 和 `model.eval()` 的切换**:

```python
for epoch in ...:
    model.train()           # 打开 Dropout、BatchNorm 训练模式
    for batch in train_loader:
        ...
    model.eval()            # 关闭 Dropout、BatchNorm 用累积统计
    val = evaluate(model, val_loader)
```

**常见 bug**:evaluate 里 `model.eval()` 被设上了,下一个 epoch 训练开始忘了 `model.train()`,结果训练时 dropout 失效、BN 不更新统计量。
**修复**:每个 epoch 最外层先 `model.train()`,evaluate 里第一行 `model.eval()`,两边都写死。

`@torch.no_grad()` 再加一层保险——评估时不建计算图,省显存、快 2x。

---

## 6. 与 notebook 的连接

这一节基本覆盖了 cell 16 的**每一行**。对照着看:

```python
for epoch in range(1, max_epochs + 1):
    model.train()                                 # <- 视角 C 坑位
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()                     # 1
        logits = model(x)                         # 2
        loss = loss_fn(logits, y)                 # 3
        loss.backward()                           # 4
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 4.5
        optimizer.step()                          # 5
        total += loss.item() * x.size(0)
    train_loss = total / len(train_loader.dataset)
    val = evaluate(model, val_loader)             # 里面做 model.eval()
    ...
```

**下一站**:[06_pos_weight.md](06_pos_weight.md) —— 基础训练能跑了,但欺诈数据正样本只有 0.17%——怎么让模型别偷懒全部预测 0?
