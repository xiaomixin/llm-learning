# Week 1 — 知识伴读：问题定义、极不平衡分类与 EDA 思维

> 配套 notebook：`week01/env_check.ipynb`、`week01/01_eda.ipynb`
> 前置阅读：`transformer-12week-plan.md` §1（目标 & MVP）、§5 Week 1、§7.2（异常检测自问）
> 本周目标（能力层）：能在任何"欺诈/异常"场景里，独立做出"为什么是分类？用什么指标？EDA 先看什么？"这三个判断。

---

## 1. 本周要回答的核心问题

1. **为什么信用卡欺诈 = 极度不平衡二分类？**"异常"的业务定义如何被翻译成 ML 任务？
2. **AUC-PR 与 AUC-ROC 的几何直觉差在哪里？**为什么 0.17% 正样本场景下 AUC-ROC 会"说谎"？
3. **Recall@FPR=ε 这种业务指标背后的成本模型是什么？**ε 怎么定？
4. **Kaggle 这套 PCA 脱敏特征（V1–V28）给建模带来了哪些便利和代价？**
5. **EDA 应该看什么？**类别分布、时间、金额这三张图背后的 builder-mindset 是什么？

这五个问题是 Week 1 剩下两节的骨架；后续两周会把"判断"落成"代码"。

---

## 2. 理论骨架

### 2.1 欺诈检测的 ML 建模框架

欺诈检测问题的原生业务目标是"**损失最小化**"：

$$\text{Expected Loss} = C_{FN} \cdot P(\text{miss}) \cdot L_{\text{fraud}} + C_{FP} \cdot P(\text{false alarm}) \cdot L_{\text{friction}}$$

其中 $C_{FN}$ 是漏报一笔欺诈造成的赔付 / 商誉损失（通常 $10^2$ 到 $10^4$ USD/次），$C_{FP}$ 是误拦一笔正常交易造成的用户流失与运营成本（通常 $1$ USD/次量级）。$C_{FN} / C_{FP}$ 一般在 $10^2$ – $10^4$ 量级，这种非对称代价正是不平衡问题的"业务版本"。

把这个目标映射到 ML，通常只有两种可行的二分类形式：

- **有监督分类**：$y \in \{0, 1\}$，学 $\hat{p}(y=1 \mid x)$，按阈值 $\tau$ 决策。Kaggle 信用卡数据集属于这类。
- **无监督异常检测**：$\hat{p}(x)$ 建模正常分布，异常 = 低密度区。W9 的重构模型是这条路。

Week 1 只考察**有监督 + 极度不平衡**的形式。关键的建模选择是：

$$\text{Decision}(x) = \mathbb{1}[\hat{p}(y=1 \mid x) > \tau], \quad \tau \text{ 由业务成本决定}$$

这个决策框架决定了我们关心"模型在不同 $\tau$ 下的行为"，而不是单一阈值上的 accuracy。这就是为什么下面要讲 PR 曲线。

### 2.2 为什么 Accuracy 不能用

Kaggle 数据：$N = 284807$，$N_{pos} = 492$，$P(y=1) = 0.00173$。一个恒输出 0 的"空模型"可达到：

$$\text{Accuracy}_0 = 1 - P(y=1) = 99.827\%$$

这个数字没有任何信息量——"模型压根没学"这件事被 accuracy 给掩盖了。信息论视角更清晰：若以 $y$ 的熵 $H(y)$ 作为基线，$y \in \{0,1\}$ 且 $p=0.00173$ 时

$$H(y) = -p \log p - (1-p)\log(1-p) \approx 0.0166 \text{ bits}$$

整个二元标签只携带了 **0.017 bit** 的信息。accuracy 这种"等权"的指标对 0.017 bit 的信号完全不敏感。

### 2.3 PR 曲线 vs ROC 曲线：几何与数学

对任意阈值 $\tau$，定义

$$\text{TPR}(\tau) = \frac{TP}{TP+FN} = \text{Recall}, \quad \text{FPR}(\tau) = \frac{FP}{FP+TN}, \quad \text{Precision}(\tau) = \frac{TP}{TP+FP}$$

**ROC 曲线**：$(FPR, TPR)$ 在 $\tau$ 变化下的轨迹。AUC-ROC = 随机抽一正一负，正样本得分高于负样本的概率。

**PR 曲线**：$(Recall, Precision)$ 轨迹。AUC-PR（又叫 Average Precision）是 Recall 上加权的平均 Precision。

**关键差异**：ROC 的 FPR 分母是所有负样本 $N_{neg}$，PR 的 Precision 分母是所有被预测为正的样本 $TP + FP$。当 $N_{pos} \ll N_{neg}$ 时：

- 一个模型把 TPR 从 0.5 提到 0.9，可能只需从 1000 个负样本里多抓 50 个 FP——FPR 从 0.01 升到 0.015，ROC 曲线几乎平坦。
- 但对 Precision：若只有 500 个真正正样本，多来 50 个 FP 会让 Precision 从 0.91 掉到 0.82——PR 曲线明显下滑。

数学上，**FPR 与 Precision 的关系**在偏斜数据下严重非对称。设 $\pi = P(y=1)$，给定 TPR 和 FPR：

$$\text{Precision} = \frac{\pi \cdot \text{TPR}}{\pi \cdot \text{TPR} + (1-\pi) \cdot \text{FPR}}$$

代入 $\pi = 0.00173$：即便 TPR = 0.9, FPR = 0.01，Precision = $\frac{0.00173 \cdot 0.9}{0.00173 \cdot 0.9 + 0.998 \cdot 0.01} \approx 0.135$。也就是说"抓 10 个有 1.35 个是真欺诈"——ROC 看起来漂亮（FPR=0.01 很低），但业务上仍是灾难。

**经验法则**（ICML 2006, Davis & Goadrich）：极不平衡场景 **ROC 平滑骗人，PR 诚实暴露问题**。所以本项目把 AUC-PR 作为首要指标，AUC-ROC 只作辅助。

### 2.4 Recall@FPR=ε：业务语义的指标

AUC-PR 是整条曲线的面积，方便比较模型但不直接回答"我应该把阈值设在哪"。**Recall@FPR=0.001** 的含义是：

> "我能接受每 1000 笔正常交易最多误拦 1 笔（运营成本上限 = 0.1% 用户摩擦），在此约束下能抓回多少比例的欺诈？"

这是典型的 **Neyman–Pearson 决策**：固定 $\alpha = \text{FPR}$，最大化 $\text{TPR}$。它等价于把非对称代价写进约束。$\varepsilon$ 的选取依赖业务：

- 银行高风险账户：$\varepsilon = 10^{-3}$ 到 $10^{-4}$（严格，用户体验优先）
- 反洗钱案例排查队列：$\varepsilon = 10^{-2}$（宽松，人工复核接得住）
- 实时风控拦截：$\varepsilon = 10^{-4}$ 到 $10^{-5}$（极严格）

Week 2 的代码用 $\varepsilon = 10^{-3}$，是"既够严、又留足调试空间"的妥协值。

### 2.5 PCA 脱敏数据（V1–V28）的影响

Kaggle 信用卡数据的 30 列特征里，V1–V28 是原始特征（未知语义：可能是 MCC、持卡人年龄段、设备指纹等）在 PCA 下的 28 个主成分。Time 和 Amount 保持原样。

PCA 之后数据有以下性质：

1. **零均值、近似独立**：$\mathbb{E}[V_i] = 0$、$\text{Cov}(V_i, V_j) \approx 0$，这让线性模型天然就该好用。
2. **尺度被放大后的方差排序**：$\text{Var}(V_1) > \text{Var}(V_2) > \cdots$，但 sklearn 的 PCA 默认做方差白化与否不同数据不同；从 `df.describe()` 可见 V1–V28 的 std 在 1–2 量级波动。
3. **语义丢失**：看不到"是不是跨境、是不是夜间"，只能从分布差异反推。

**对建模的影响**：

- **优点**：特征本身近正态、去相关，MLP 这种"各向等权"的模型容易收敛；不需要再手动 one-hot/scaling 业务类别；避免了"从原始数据做特征工程"可能带来的泄露。
- **缺点**：不能做可解释性归因（"模型为什么把这笔判异常"无法说清）；也不能做特征工程（类别组合、时间桶等）。这是 Kaggle 公开数据集的"教学代价"——真实产线会回到原始特征。
- **建模推论**：V1–V28 已经是"压缩过"的表示，所以**加深的 MLP 收益递减**；Week 2 的 2 层 hidden=64 就能逼近 AUC-PR ≈ 0.75，再深容易过拟合。Transformer 在 W7+ 也主要靠"序列依赖"拿额外收益，不是靠更复杂的单笔特征提取。

### 2.6 EDA 应该看什么

EDA 不是"画漂亮图"，是**建模前提的三重确认**：

| 问题 | 图 | 想看什么 |
|------|----|---------|
| 任务是什么？ | 类别分布 | 正样本比例 → 决定损失函数、评估指标、采样策略 |
| 特征有信号吗？ | V1–V5 条件分布 | 正负样本分布是否可分 → 决定能不能线性/浅层模型 |
| 有没有偏斜需要处理？ | Amount 长尾 + time 分布 | 是否需要 $\log(1+\text{Amount})$、时间切分策略 |

这三张图不是可选，是**建模决策的前置证据**。比如：

- 如果 V4 在正负样本上完全重合，说明单个 PCA 主成分没信息，得靠组合——此时浅层 MLP 可能不如 GBDT（树能做特征交叉）。
- Amount 在 log 尺度下正常集中在 1–100，欺诈集中在 0–20（小额刷卡测试卡有效性）——这提示"金额偏低"其实是欺诈信号之一。
- 小时分布如果差别不大，说明 Time 本身弱信号；这也解释了为什么 Kaggle 顶级方案里很多人直接扔掉 Time。

---

## 3. 代码对照

### 3.1 Bootstrap：可复现的环境约定（`01_eda.ipynb` cell 0）

```python
import os, sys, pathlib, random
IN_COLAB = 'google.colab' in sys.modules
...
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

为什么四行 seed？三个独立 RNG：`random`（Python shuffle、choice），`numpy`（大多数科学计算），`torch`（张量运算和 DataLoader 的 worker）。`torch.cuda.manual_seed_all` 覆盖所有 GPU（多卡场景），单卡 T4 也是安全写法。

**注意一个 subtle 坑**：`torch.manual_seed` 只控制 CPU 上 tensor 初始化；DataLoader 的 `num_workers > 0` 时每个 worker 进程有独立的 seed 逻辑（通过 `worker_init_fn` 控制）。Week 1 不涉及 DataLoader，但 Week 2+ 要记住这个点。

### 3.2 `env_check.ipynb` cell 11：GPU smoke test

```python
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
torch.cuda.synchronize() if device.type == 'cuda' else None
t0 = time.time()
c = a @ b
torch.cuda.synchronize() if device.type == 'cuda' else None
```

`torch.cuda.synchronize()` 这对包裹是**必做的**。CUDA 默认异步执行：`c = a @ b` 返回时 GPU 上的 kernel 可能还没跑完。不做 sync 直接 `time.time()` 会把 0.1 秒的工作量误记成 0.001 秒，完全没法衡量性能。4096² FP32 矩阵乘 = $2 \cdot 4096^3 / 10^9 \approx 137$ GFLOPs，T4 峰值 8.1 TFLOPs FP32 → 理论下界 17 ms，实测通常 30–80 ms。

### 3.3 类别分布（`01_eda.ipynb` cell 8）

```python
counts = df['Class'].value_counts()
pct = df['Class'].value_counts(normalize=True) * 100
...
sns.countplot(x='Class', data=df, ax=ax[0])
ax[0].set_yscale('log')
```

`set_yscale('log')` 是必须的——线性尺度下欺诈那根柱子只有一个像素高，看不出来。对应理论 §2.2：这张图确认"任务是极度不平衡"，触发后续"pos_weight / focal / 特殊指标"的所有决策。

### 3.4 条件分布（`01_eda.ipynb` cell 10）

```python
for ax, col in zip(axes.flat, ['V1', 'V2', 'V3', 'V4', 'V5']):
    sns.kdeplot(df[df.Class == 0][col], label='Normal', ax=ax, fill=True)
    sns.kdeplot(df[df.Class == 1][col], label='Fraud',  ax=ax, fill=True)
```

**用 KDE 而不是 histogram 的原因**：欺诈只有 492 条样本，直方图在稀疏区会极度不稳定（宽 bin 丢细节、窄 bin 到处是空）。KDE 用核密度平滑，视觉上更好判断"两条曲线是否真的可分"。对应理论 §2.6 的第二个问题："特征有信号吗"。

建模判断：如果某几个 V 列的 Fraud 分布明显左移/右移或多峰，说明这些主成分有强信号。工程上对应 Week 2 MLP 能收敛的前置条件。

### 3.5 Amount 长尾（`01_eda.ipynb` cell 12）

```python
for cls, color, label in [(0, 'C0', 'Normal'), (1, 'C3', 'Fraud')]:
    ax[0].hist(df[df.Class == cls]['Amount'], bins=80, alpha=0.6, label=label, color=color, log=True)
```

`log=True` 只是把 y 轴取对数，x 轴仍是原始金额。**更好的做法**是 `np.log1p(amount)` 然后画线性直方图（Week 3 的合成数据那部分就是这么做的）。长尾 → 标准化前一定要 log；这是 Week 2 `StandardScaler` 为什么能 work 的前提（Amount 若不取 log，PCA 后的 V 列在同一 scaler 下会被 Amount 的巨大方差"压死"）。

### 3.6 时间分布（`01_eda.ipynb` cell 14）

```python
df['Hour'] = (df['Time'] // 3600).astype(int) % 24
...
counts = df[df.Class == cls].groupby('Hour').size()
counts = counts / counts.sum()   # 归一化为分布
```

`% 24` 的处理把原始 0–48h 折叠成"一天中的小时"。归一化很重要——正常交易约 28 万条，欺诈 492 条，不归一化两条曲线不在一个量级，看不出"欺诈是否有时间偏好"。

对应建模判断：如果欺诈在凌晨密集，说明 Hour 是强特征；如果分布和正常几乎重合，说明时间信息被 PCA 吃掉了（V 列已包含时间相关信号）。这个结论会直接影响 Week 3 如何构造时间特征。

---

## 4. 常见坑位与调试思维

### 4.1 分层切分忘了做

一旦直接 `train_test_split(X, y, test_size=0.15)`（默认不 stratify），val 集里的欺诈数量是服从 Binomial(42721, 0.00173) ≈ 74 ± 8 的随机变量。seed 换一下 AUC-PR 能抖动 0.02+，你以为模型好了其实只是 split 好了。**任何时候**欺诈比例 < 1% 都必须 `stratify=y`。

### 4.2 Scaler 用全量 fit

在 Week 2 会踩到：`scaler = StandardScaler().fit(X)` + `X_train = scaler.transform(X_train)`。从 AUC 数字看可能只差 0.001，但这是真正的 test set information leakage——val 和 test 的均值方差偷偷参与了训练集的归一化参数。正确做法是 `fit(X_train)` 然后 transform 三份。

### 4.3 用 Accuracy 画 learning curve

不止是评估指标选错的问题——Accuracy 在训练过程中变化极小（99.83% → 99.91%），**你看不出模型是变好还是过拟合**。一定要用 AUC-PR 做训练日志。

### 4.4 "感觉有信号"不等于"有信号"

KDE 图肉眼看起来分开 ≠ 模型能学到。严谨做法：把每个特征做单变量 logistic，看单特征 AUC。Kaggle 数据里 V14、V17 单特征 AUC-PR 就能到 0.3+，是多特征模型的主要贡献者。EDA 阶段心里要对此有初步排序，Week 2 结果对不上时好 debug。

### 4.5 Colab Drive 路径切换

`os.chdir(PROJECT_ROOT)` 是必须的。Colab 默认工作目录是 `/content`，所有相对路径（`data/creditcard.csv`）都会跑去 `/content/data` 找数据，第二次开 runtime 全丢。Week 1 bootstrap cell 的 `os.chdir` 不是装饰，是**唯一能让 notebook 二次打开直接 Run All 的机制**。

### 4.6 Kaggle API 凭证管理

Cell 3 的 `~/.kaggle/kaggle.json` 方式在 Colab 里要上传文件，重启 runtime 丢。Cell 0 用的 `Colab Secrets + os.environ['KAGGLE_USERNAME']` 方式更优——secrets 跟随账号持久，不入代码仓库。如果看到两种方式并存，首选 secrets，老的 `!cp kaggle.json` 逻辑在 cell 3 只是向后兼容的 fallback。

### 4.7 debug 思维：怎么回答"这张图对吗"

三条准则：

- **一致性**：累计数量对得上 `df.shape[0]` 吗？
- **对称性**：如果把 Class 标签交换（0↔1），这张图应该怎么变？实际变了吗？
- **极端值**：分布尾巴上最大/最小值做 reality check（例如 Amount 最大 25691 USD 合理吗？最小 0 USD 合理吗——是的，很多免费试用授权）。

---

## 5. 与未来几周的连接

```mermaid
flowchart LR
    W1["Week 1 问题定义 EDA"] --> W2["Week 2 MLP baseline pos_weight"]
    W1 --> W3["Week 3 序列构造 时间切分"]
    W2 --> W3
    W2 --> W4["Week 4 LSTM"]
    W3 --> W4
    W4 --> W7["Week 7 Transformer"]
```

- **Week 2** 会把"AUC-PR 指标"和"pos_weight"直接实现成代码；Week 1 的类别分布图就是 `pos_weight = neg/pos` 这一行的数据依据。
- **Week 3** 会把"Time 是秒、有昼夜周期"扩展到合成数据的 hour_sin/hour_cos 特征；Week 1 的小时分布图是这个设计的直觉来源。
- **Week 9** 无监督异常检测会回到理论 §2.1 的第二类：$\hat{p}(x)$ 建模。届时 AUC-PR 仍是主指标，但 Recall@FPR 的 $\varepsilon$ 可能放宽（无监督上限本就低）。
- **Week 10** 切到 IEEE-CIS 数据集，特征不再 PCA 脱敏——届时 Week 1 §2.5 讲的"可解释性 vs 原始特征"取舍会反转回来，需要重新做一次深度 EDA。
- **Week 12 落地思考**：产线上阈值 $\tau$ 会按业务成本动态调——Week 1 §2.4 的 Neyman–Pearson 框架就是那时候的工具。

---

## 6. 自测题

**Q1**. 一个模型在 Kaggle 信用卡数据上 AUC-ROC = 0.98, AUC-PR = 0.30。能上线吗？

<details><summary>答案</summary>
不能直接下结论好或坏——AUC-ROC 0.98 在这种偏斜数据里是"及格起步"，AUC-PR 0.30 说明在给定 recall 下 precision 平均只有 30%，业务成本要算清楚。需要看 Recall@FPR=0.001 和业务成本比 $C_{FN}/C_{FP}$。如果 $C_{FN}$ 极高（比如跨境大额欺诈），0.30 AUC-PR 可能也值得上；如果是小额交易，摩擦成本占优，就不能上。
</details>

**Q2**. 为什么不直接用 $F_1$ 做模型选择指标？

<details><summary>答案</summary>
$F_1$ 依赖一个具体阈值 $\tau$，而最优 $\tau$ 在训练中会变。训练曲线用 AUC-PR（阈值无关）更稳。最终部署时才按业务约束挑 $\tau$ 再报 $F_1$。
</details>

**Q3**. 如果 PCA 脱敏的 V 列换成原始特征（MCC、时区、设备、持卡历史），EDA 要怎么改？

<details><summary>答案</summary>
加几件事：类别特征的基数统计（唯一值数量、top 类别占比）、缺失模式、类别和标签的互信息（$I(X_j; y)$ 或 WoE）、数值特征的偏度/峰度。另外得做**泄露检查**（比如"是否有一个特征是欺诈后才产生的反查 flag"）。
</details>

**Q4**. 假设把正样本比例从 0.17% 改成 50%（人工下采样负样本），AUC-ROC 会怎么变？AUC-PR 呢？

<details><summary>答案</summary>
AUC-ROC 理论上**不变**（它是排序指标，跟先验无关）；AUC-PR 会**显著变高**，因为 Precision 分母里的负样本变少。这也解释了为什么 AUC-PR 数字不能跨数据集比较——必须同一 $\pi$ 下比。
</details>

**Q5**. `df['Time']` 是秒，总长度 172792 秒 ≈ 48h。为什么数据集设计者只给 48h？

<details><summary>答案</summary>
短时间窗可以让特征之间的"业务周期"影响减弱，使得 PCA 出来的主成分更接近 iid，便于公开脱敏。48h 足够覆盖两个完整昼夜周期，能让时间特征仍可观察但不至于暴露月度/季节规律。
</details>

**Q6**. 同一个数据集上，AUC-PR 0.75 的模型 A 和 0.70 的模型 B，能不能直接断言 A 更好？

<details><summary>答案</summary>
不能。要看：1) val/test 是否从同一时间段切出；2) 阈值区间是否关心——A 可能在高 recall 区间好、低 recall 区间差，而我们只关心低 recall；3) 置信区间——bootstrap 几次看看 AUC-PR 的 std，偏斜数据上 0.05 差异可能在噪声里。
</details>

**Q7**. 如果 EDA 发现 Fraud 集中在深夜且 Amount 偏小，但 Normal 时间均匀分布，建模时能直接加规则"22:00–05:00 && Amount < 20 → 标异常"吗？

<details><summary>答案</summary>
作为特征可以（加 hour + amount_bucket），作为硬规则不行：1) 合规——黑名单式规则会被合规/产品打回；2) 过拟合——48h 数据样本量极小，规则精度会在产线上显著下降；3) 攻防——欺诈者会马上换时段/换金额。EDA 的规律应当**启发模型设计**而非直接变成规则。
</details>

**Q8**. 本周复盘要回答"AUC-PR 和 AUC-ROC 差别"，最凝练的一句话是什么？

<details><summary>答案</summary>
"ROC 在极端不平衡下因 FPR 分母过大而过度乐观；PR 把模型真实的 Precision 代价摆在前面，所以欺诈场景选 PR。"
</details>

---

## 7. 延伸阅读

1. **Davis & Goadrich, "The Relationship Between Precision-Recall and ROC Curves" (ICML 2006)** — 5 页短文，证明了"一条曲线 dominates 另一条"在 PR 和 ROC 下不等价。为什么现在读：这是"为什么选 PR"的原始论据，本周所有直觉都出自这里。
2. **d2l 第 3.4 节 Softmax 回归 + 第 4.1 节 MLP** — 为什么现在读：下周要实现 MLP，需要知道 logistic / softmax 在二分类下的等价性，以及为什么输出层不加 softmax（loss 内部合并更稳定）。
3. **Janio Bachmann, "Credit Fraud: Dealing with Imbalanced Datasets" (Kaggle)** — 为什么现在读：社区公开 EDA 做得最系统的 notebook 之一，能直接对照自己画的图看"还漏了什么"（尤其是特征相关性热力图）。
4. **He & Garcia, "Learning from Imbalanced Data" (IEEE TKDE 2009)** — 综述级别，重点读第 3 节（采样方法）和第 5 节（cost-sensitive learning）。为什么现在读：Week 2 讨论 pos_weight / SMOTE 时需要的全景。
5. **scikit-learn 文档 `average_precision_score`** — 为什么现在读：确认"AP = $\sum_n (R_n - R_{n-1}) P_n$"这个离散估计 vs "PR 曲线下的积分"在 sklearn 里的具体实现细节，避免后面跟论文的 mAP 定义对不上。
