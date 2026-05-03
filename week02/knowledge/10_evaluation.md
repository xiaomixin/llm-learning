# 10 — 评估指标(AUC-PR / AUC-ROC / Recall@FPR)

> **一句话口诀**:极不平衡下 **ROC 过度乐观**,**AUC-PR 更诚实**;实际上线关心 **Recall @ 给定 FPR**——"误伤多少换来多少抓获"。

---

## 1. 先跑起来(`../snippets/10_evaluation.py`)

```bash
python snippets/10_evaluation.py
```

脚本合成 1:500 的不平衡分数,分别算 AUC-ROC、AUC-PR、Recall@FPR=0.001,并找最大 F1 阈值打印混淆矩阵。

---

## 2. 发生了什么(白话)

三个指标看的是同一件事的不同侧面:

- **AUC-ROC**:把所有样本按分数排序,正样本平均排在负样本之**前**的概率。
- **AUC-PR**:PR 曲线下面积,相当于"**拦截到的正样本中,精确率的平均**"。
- **Recall @ FPR=0.001**:一个具体的**业务工作点**——只允许 0.1% 的正常被误拦,这时能抓到多少欺诈?

---

## 3. 多角度理解

### 视角 A:为什么 ROC 在不平衡下"过度乐观"

ROC 的横轴是 FPR = FP / N_negative。N_negative 巨大(20 万+),即使 FP 有几千,FPR 也才 0.02——看起来曲线靠左,AUC 很高。

PR 的横轴是 Recall = TP / N_positive,纵轴是 Precision = TP / (TP + FP)。**FP 哪怕几百,也能把 Precision 拉很低**,曲线直观反映"误报被稀释"。

**举例**:模型把 400 个真欺诈排在前 1000 名,另外 600 名是误报。
- 前 1000 命中 400 真欺诈:Recall = 400/492 ≈ 0.81,但 Precision = 400/1000 = 0.4。
- 从 ROC 视角看,前 1000 里只有 600 个负样本,FPR = 600/283k ≈ 0.002——看起来非常好。
- 从 PR 视角看,Precision 只有 0.4——**业务上 60% 的警报是假的**,运营没法用。

**口诀**:**分母大的指标骗人,分母小的指标才诚实**。ROC 的分母是所有负样本,PR 的分母是预测为正的样本。

### 视角 B:Recall @ FPR = 0.001 的业务意义

- "每 1000 个正常交易,最多误拦 1 个"是风控团队常用的运营红线。
- 在这个工作点上,模型能抓到多少真欺诈?——就是 `Recall @ FPR=0.001`。
- 这个数直接和风控 KPI 挂钩,比 AUC 更有"我能不能上线"的决策价值。

### 视角 C:阈值从哪里来

模型输出是连续分数 $\in [0, 1]$,业务要的是 0/1 决定。选阈值的思路:
1. **最大 F1**:学习阶段的"综合感受"阈值,兼顾 P 和 R。
2. **固定 Recall @ 0.8,找最大 Precision**:风控运营关心"抓够了再看误报"。
3. **固定 FPR @ 0.001,找最大 TPR**:风控运营关心"先保住体验再看抓获率"。

Week 2 cell 21 用"最大 F1"作为默认,但上线前应该和业务沟通到底选哪个。

---

## 4. 公式慢推

**精确率 & 召回率**:

$$\text{Precision} = \frac{TP}{TP + FP}\quad\text{("我报的警有多少是真的")}$$
$$\text{Recall} = \frac{TP}{TP + FN}\quad\text{("真欺诈里我抓了多少")}$$

**F1**:

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}\quad\text{(P 和 R 的调和平均)}$$

**AUC-PR** = PR 曲线下面积(average precision)≈ 把所有正样本的 Recall@thr 求平均。

**AUC-ROC** = ROC 曲线下面积 = "随便抽一个正样本和一个负样本,模型把正排前面的概率"。

---

## 5. 实现上的两个坑

### 坑 A:`precision_recall_curve` 多一个端点

```python
prec, rec, thr = precision_recall_curve(y_true, s)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.nanargmax(f1[:-1])   # <- 注意 [:-1]
```

`prec` 和 `rec` 的长度比 `thr` 多 1——最后一个点对应 `threshold = +∞`(precision=1, recall=0),那里 F1 = 0,不是有意义的阈值。用 `[:-1]` 去掉这个端点。

### 坑 B:`searchsorted` 边界

```python
fpr, tpr, _ = roc_curve(y_true, s)
idx = np.searchsorted(fpr, 0.001)
rec_at_fpr = tpr[min(idx, len(tpr) - 1)]
```

- `fpr` 数组长度不一定等于样本数,`idx` 可能超界,需要 `min(idx, len-1)`。
- 如果 `fpr` 数组根本没达到 `0.001`(模型太差),返回的 TPR 是最接近的那个点。
- 严谨时应该加 assert 检查 `fpr[idx] ≤ 0.001 + 1e-6`。

---

## 6. 和 notebook 的连接

- cell 14 定义 `predict_scores` + `evaluate`,包括 AUC-PR、AUC-ROC、Recall@FPR。
- cell 21 做最大 F1 阈值选择 + 混淆矩阵 + PR 曲线绘图。

---

## 7. Week 2 评估验收三件套

跑完 notebook,你应该能看到:
- **val AUC-PR > 0.70**(接近 0.75 更好)
- **val AUC-ROC > 0.95**(不难,几乎所有 baseline 都能到——正因为如此不能光看它)
- **test Recall@FPR=0.001 > 0.5**(业务红线)

**验收不过?**
- AUC-PR 太低:先检查 `pos_weight` 是不是 577(概念 06),再检查 scaler 有没有漏 fit(概念 08),再调 lr/epoch。
- AUC-ROC 很高但 AUC-PR 低:典型的"ROC 乐观"。不是模型坏,是 MLP 容量在序列信号面前不够——Week 4 上 LSTM 会改善。
- Recall@FPR 太低:模型分数分布太宽,难样本混在负样本里。可以试 focal loss(概念 07),或等 Week 4。

---

## 8. 回顾

走完 10 个概念,你应该能做到:

1. 跑 notebook 看 val_ap = 0.75,知道这个数是**怎么来的**(AUC-PR 定义、为什么在不平衡下用它)。
2. 指着 cell 16 任一行说清**为什么这么写**(训练循环 5 步 + zero_grad 坑 + deepcopy 坑)。
3. 被问"为什么不用 Sigmoid+BCELoss?",讲得出 log-sum-exp 稳定性(概念 04)。
4. 被问"为什么不用 SMOTE?",讲得出 `pos_weight` 更轻量且不制造假数据(概念 06)。
5. 被问"scaler 为啥不在全集上 fit?",讲得出 val/test 统计量泄露(概念 08)。

然后就可以进 Week 3 了。
