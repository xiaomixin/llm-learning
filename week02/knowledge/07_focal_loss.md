# 07 — Focal Loss

> **一句话口诀**:Focal = 加权 BCE 再乘一个**"难样本放大器"** $(1 - p_t)^\gamma$。越容易分对的样本,$p_t$ 越大,权重越接近 0——等于把容易样本"静音",让梯度集中在难样本上。

---

## 1. 先跑起来(`../snippets/07_focal.py`)

```bash
python snippets/07_focal.py
```

脚本打印不同 $p_t$ 和 $\gamma$ 下的调制因子 $(1-p_t)^\gamma$,让你直观看到"**预测越自信越正确,权重越小**"。

---

## 2. 发生了什么(白话)

`pos_weight` 解决的是"**类别层面**"的失衡:正样本少,整体声音小。但还有一个更隐蔽的问题:**类内**也有难易之分。

想象 492 个欺诈样本里:
- 300 个是"教科书欺诈"——模型几轮就学会了,每次前向 $p_t \approx 0.95$。
- 100 个是"灰色地带"——$p_t \approx 0.5$,模型还没拿定主意。
- 92 个是"伪装得很好的"——$p_t \approx 0.1$,模型还分错。

加权 BCE 对三种样本一视同仁(都乘 w)。训练进行到后期,loss 的大部分来自那 300 个已经分对的"教科书"——梯度在浪费。**Focal Loss 的想法**:让已经分对的样本"自觉闭嘴",把梯度留给难样本。

---

## 3. 多角度理解

### 视角 A:$p_t$ 的定义——"正确类的预测概率"

```
y = 1 时  p_t = p = sigmoid(z)
y = 0 时  p_t = 1 - p
```

无论正负,`p_t` 都代表"**模型对真实标签有多自信**"。
- `p_t = 1.0` → 完美猜对
- `p_t = 0.5` → 完全没谱
- `p_t = 0.0` → 完全猜反

### 视角 B:调制因子 $(1-p_t)^\gamma$——易样本静音器

| $p_t$ | $\gamma=0$ | $\gamma=1$ | $\gamma=2$ | $\gamma=5$ |
|-------|-----------|-----------|-----------|-----------|
| 0.99  | 1.00      | 0.01      | 0.0001    | ~0        |
| 0.90  | 1.00      | 0.10      | 0.01      | ~0        |
| 0.50  | 1.00      | 0.50      | 0.25      | 0.03      |
| 0.10  | 1.00      | 0.90      | 0.81      | 0.59      |
| 0.01  | 1.00      | 0.99      | 0.98      | 0.95      |

读这张表的口诀:
- **$\gamma = 0$**:因子永远 = 1,Focal 退化为**加权 BCE**(只靠 $\alpha$)。
- **$\gamma = 2$**(原论文推荐):易样本(`p_t=0.9`)权重剩 1%,难样本(`p_t=0.5`)权重剩 25%。**25/1 = 25 倍"难样本优先"**。
- **$\gamma$ 太大**:过度聚焦极难样本——**和标签噪声合谋**,反而过拟合噪声。

### 视角 C:$\alpha_t$——还是类别权重

Focal 里的 $\alpha$ 仍然是"正样本给多大权重"的调节,和 `pos_weight` 作用一样。对称写法:

```python
alpha_t = alpha * y + (1 - alpha) * (1 - y)
```

原论文做 dense object detection,**正样本(有物体)本来就少**,但作者选了 $\alpha = 0.25$ 反而**降低**正样本权重——因为 $\gamma$ 已经在单边把难负样本托了起来,再给正样本 0.75 就过。

**信用卡场景的陷阱**:直接抄 $\alpha = 0.25$ 等于在**已经是少数的正样本上再减权**。合理值应该是 $\alpha = 0.75$(或更激进的 0.9),然后搜 $\gamma \in \{1, 2, 3\}$。

---

## 4. 公式慢推

**标准 CE**:$\mathcal{L}_{CE}(p_t) = -\log p_t$。

**Focal Loss**(Lin et al. 2017):
$$\boxed{\mathcal{L}_{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log p_t}$$

三部分:
- $-\log p_t$:原始的 CE 惩罚。
- $(1 - p_t)^\gamma$:**调制因子**,易样本小、难样本大。
- $\alpha_t$:**类别权重**,类似 `pos_weight`。

**梯度形状**(以 y = 1 推导):
$$\frac{\partial \mathcal{L}_{FL}}{\partial z} = -\alpha (1-p)^\gamma \cdot \big[\gamma p \log p + (p - 1)\big]$$

(详细推导见 `knowledge_legacy.md` §2.5——这里只需要知道:**$\gamma$ 越大,$p \to 1$ 时这个梯度衰减得越快**。)

---

## 5. 实现上的一个工程技巧

```python
def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()
```

**亮点**:不手写 $-\log p_t$,直接复用 `binary_cross_entropy_with_logits`(它内部用 log-sum-exp)算出 `ce`,再乘调制因子。这样**数值稳定性复用**了 PyTorch 的实现,自己只加"形状变换"。这是一种好习惯:**组合已经稳定的算子,而不是手写底层**。

---

## 6. 一个坑

**$\alpha$ 正着想还是反着想**:

原论文 $\alpha = 0.25$ 作用在**正样本**(y=1)。信用卡场景里正样本是少数,你想**提高**正样本权重——所以应该 $\alpha = 0.75$,不是 0.25。

**怎么判断没用反**:训练几轮后看:
- 正样本 loss 均值 / 负样本 loss 均值 应该接近 1(两类贡献差不多)。
- 如果正样本 loss 均值 ≪ 负样本 loss 均值,说明正样本被**进一步压低**,$\alpha$ 取反了。

---

## 7. 什么时候 Focal 打得过 `pos_weight`

**打得过**:类内难易分布宽 + 类别失衡(IEEE-CIS、图像检测、语义分割)。
**打不过**:类内样本都差不多难(Kaggle credit card 的 V1–V28 已经是 PCA 主成分,都挺"干净")。

实际 Kaggle credit card 上,Focal 比加权 BCE 大约 ±0.01 AUC-PR——**收益很小,不值得复杂化**。但一定要亲手切换跑一次 cell 23,知道**差距不是理论,是数据决定**。

---

## 8. 与 notebook 的连接

- cell 23 的 `FocalLoss` 就是本节的实现。
- `USE_FOCAL = True` 会用它重训一次并和加权 BCE 对比。
- 对比时记得把 `alpha=0.25` 改成 `alpha=0.75` 才公平。

**下一站**:[08_data_split_scaler.md](08_data_split_scaler.md) —— 训练/损失都 OK 了,但数据切分和标准化上有一类**静默泄露**会让 val 指标虚高,必须管住。
