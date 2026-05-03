# 08 — 数据切分 + Scaler 只 fit 在 train

> **一句话口诀**:**stratify 保比例**,**scaler 只在 train 上 fit**。
> 前者让每个 split 的欺诈比例一致,后者防止 val/test 的统计量泄露进训练。

---

## 1. 先跑起来(`../snippets/08_split_scaler.py`)

需要 creditcard.csv(见 `snippets/_data.py` 的查找规则)。

```bash
python snippets/08_split_scaler.py
```

期望看到:
- 三个 split 的 fraud ratio 几乎一样(stratify 成功);
- 不 stratify 时 val 正样本数能差十几个;
- `fit on all` 和 `fit on train` 的 scaler 均值存在差异——**这就是泄露的量**。

---

## 2. 发生了什么(白话)

**两个独立的坑**,常被合并讲,其实是两件事:

### 坑 A:随机切分时正样本数量抖动

欺诈只占 0.17%。`train_test_split(..., test_size=0.15)` 不加 stratify 时,15% 的 test 里可能随机抽到 60~80 个正样本,波动很大。val 同样。
**指标方差爆炸**:一次训练的 val AUC-PR 是 0.72,再 split 一次可能 0.68——你分不清是模型变差还是数据抖动。
**解**:`stratify=y` 让采样器在正负类里分别按比例抽,压住波动。

### 坑 B:scaler 在全集上 fit = 用未来数据

`StandardScaler` 会算 $\mu, \sigma$ 然后做 $x' = (x - \mu)/\sigma$。如果 $\mu, \sigma$ 用全集算,就包含了 val/test 的信息。训练时模型看到的 train 特征实际上"知道未来的分布"——val 指标偏高,上线后分布一变就打脸。

**解**:`scaler.fit(X_train)` 之后 `scaler.transform(X_val)`、`scaler.transform(X_test)`。

---

## 3. 多角度理解

### 视角 A:stratify 的两次切分魔数

Week 2 要 train : val : test = 70 : 15 : 15。sklearn 只能二分,所以:

```python
# 1) 切出 test
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=SEED)

# 2) 从剩余里切 val
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tv, y_tv, test_size=0.1765,   # 0.15 / 0.85 = 0.1765
    stratify=y_tv, random_state=SEED)
```

`0.1765 = 0.15 / 0.85` 的算术是为了让最终比例精确 70:15:15,不是魔数。

### 视角 B:泄露的"严重程度"

| 预处理类型 | 是否可在全集上做 | 原因 |
|-----------|-----------------|------|
| `StandardScaler` | ✗ 只能 train | 用到 $\mu, \sigma$ 统计量 |
| PCA | ✗ 只能 train | 学习主轴 |
| IQR 去异常 | ✗ 只能 train | 分位数统计量 |
| 类别频率编码 | ✗ 只能 train | 频率统计量 |
| `log1p(x)` | ✓ 全集可做 | 固定数学公式,不依赖数据 |
| 独热编码 | ✓ 全集可做 | 结构变换(**但编码时要锁 categories**) |
| 特征间乘除比例 | ✓ 全集可做 | 公式变换 |

**原则**:**"用了数据统计量就要严格只 fit 在 train"**;公式变换随意。

### 视角 C:Week 3 的衍生规则

Week 3 切序列不能随机 stratify,要**按时间切**。同样的"train 只能看 train 范围"原则,但更严格:**`train.max(ts) ≤ val.min(ts)`**——防时间泄露。Week 3 的时候回来看这一节的逻辑,会发现只是一个升级版。

---

## 4. 一个看得见的泄露例子

看脚本输出的 `fit on all` vs `fit on train` 的均值差:

```
Scaler mean (first 3 features):
  fit on all  : [0.0000, -0.0001, +0.0001]
  fit on train: [0.0003, -0.0008, +0.0005]
  max |delta| across all features: 0.0023
```

看起来很小?但这些差值会**乘以 14 亿次前向**(batch=512, 30 epoch, ~40K 步),微小偏差累积成"**val 指标盲目高了 0.01~0.02**"——恰好是你以为的"模型改进"。

**在不平衡检测里更严重**:val 的正样本只有几十个,$\mu$ 小小偏移就可能把"难正样本 z-score 从 2.5 降到 2.0"——模型更容易捕捉,但生产环境没这层 buff。

---

## 5. 一个坑

**fit 一次,transform 多次**:

```python
scaler = StandardScaler().fit(X_train)     # ← 只 fit 一次
X_train = scaler.transform(X_train)         # reuse
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
```

常见错误:每次都 `.fit_transform()`——val/test 用的是自己的 $\mu, \sigma$,分布彻底错位。`fit_transform` 只能用一次,用在 `X_train`。

**上线时**:把 `scaler.mean_` 和 `scaler.scale_` 保存到 checkpoint(cell 19 就这么做),生产环境用同一组参数 transform 新数据。如果线上分布漂移明显,**周期性重训 + 重 fit scaler**。

---

## 6. 与 notebook 的连接

- cell 6 完整实现了这套逻辑。
- cell 19 保存 scaler 参数到 checkpoint,这是"模型 + 预处理"一起打包的必要步骤——少保一个,上线就错。

**下一站**:[09_early_stopping.md](09_early_stopping.md) —— 训练多久合适?过早停欠拟合,过晚停过拟合,怎么自动决定?
