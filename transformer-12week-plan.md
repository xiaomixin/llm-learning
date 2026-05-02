# Transformer 12 周学习计划（场景驱动：交易流 → Transformer → 异常检测）

> **学习者画像**：深度学习有基础但生疏；Python 熟练；工具栈 Google Colab；场景聚焦交易/时序异常检测。
> **总时长**：12 周 × 10 小时 = 120 小时
> **核心目标**：12 周后，能独立用 Transformer 在（脱敏/合成）交易流数据上做异常检测，并理解时序 Transformer 变体的优劣。

---

## 目录

1. [学习目标与 MVP 定义](#1-学习目标与-mvp-定义)
2. [主教材与参考资料](#2-主教材与参考资料)
3. [每周 10 小时时间分配模板](#3-每周-10-小时时间分配模板)
4. [MVP 迭代路线图](#4-mvp-迭代路线图)
5. [12 周详细计划](#5-12-周详细计划)
6. [Colab 工程模板](#6-colab-工程模板)
7. [验收标准与自检清单](#7-验收标准与自检清单)
8. [常见坑与学习建议](#8-常见坑与学习建议)

---

## 1. 学习目标与 MVP 定义

### 1.1 12 周后的具体能力

- **理论**：能从零推导 Scaled Dot-Product Attention、Multi-Head Attention、Positional Encoding；能解释 Transformer 为什么比 RNN 更适合并行 + 长依赖；能理解 Informer/PatchTST/Anomaly Transformer 三种变体的核心改动。
- **工程**：能用 PyTorch 从零实现一个 Transformer Encoder（不依赖 `nn.Transformer`）；能用 HuggingFace / 官方实现组合出完整 Pipeline；能在 Colab 免费 GPU (T4) 上训练并评估。
- **业务**：能把"交易流"这种结构化序列数据映射成 Transformer 的输入格式（token 化、embedding、mask）；能设计合理的异常检测损失函数（reconstruction / classification / contrastive）；能回答"这个模型在产线能用吗？推理延迟？特征漂移怎么办？"

### 1.2 MVP 定义

**场景一句话描述**：给定一个用户的近 N 笔交易序列（金额、时间、对手方、设备、地理等），判断最后一笔是否异常。

```
输入:  x ∈ ℝ^(B × L × F)        # B=batch, L=序列长度(如64), F=特征维度
        ↓
     [Embedding + Positional Encoding]
        ↓
     [Transformer Encoder × N 层]
        ↓
     [分类头 or 重构头]
        ↓
输出:  y ∈ ℝ^B (异常分数) 或 ℝ^(B×L×F) (重构值)
```

### 1.3 MVP 5 次迭代

| 版本 | 完成时机 | 模型 | 数据 | 目标 |
|------|---------|------|------|------|
| v0.1 | W2 末 | MLP | Kaggle 信用卡欺诈（单笔） | 建立 baseline 和评估框架 |
| v0.2 | W4 末 | LSTM | 按用户聚合后的序列 | 序列建模 baseline |
| v0.3 | W8 末 | 手写 Transformer Encoder | 合成交易序列 | 验证 Transformer 学会了模式 |
| v0.4 | W10 末 | Transformer + 双头（分类+重构） | 真实 Kaggle 数据 | 跑通异常检测 MVP |
| v0.5 | W12 末 | PatchTST / Informer 改造 | 同上 + 长序列 | 理解时序 Transformer 并做优化 |

---

## 2. 主教材与参考资料

### 2.1 主干教材（必读）

**《动手学深度学习》PyTorch 版** — 李沐
- 网址：<https://zh.d2l.ai/>
- 代码：<https://github.com/d2l-ai/d2l-zh>
- 重点章节：
  - Ch 3-5：线性/MLP（W1-W2 复习）
  - Ch 9：循环神经网络（W3-W4）
  - Ch 10：注意力机制（W5-W6）
  - Ch 11：Transformer（W7）
  - Ch 15：自然语言处理（选读，W8）

### 2.2 手写实战（必看）

**Andrej Karpathy《Neural Networks: Zero to Hero》** — YouTube 免费
- 播放列表：<https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ>
- 重点视频：
  - `makemore Part 4: Becoming a Backprop Ninja`（W5 复习反向传播）
  - `makemore Part 5: Building a WaveNet`（W5 序列卷积对比）
  - **`Let's build GPT: from scratch, in code, spelled out`**（W7-W8 核心！逐行讲解 nanoGPT）
- 仓库：<https://github.com/karpathy/nanoGPT>

### 2.3 工程化（W9+）

**HuggingFace NLP Course** — 免费中英文
- 网址：<https://huggingface.co/learn/nlp-course>
- 重点章节：
  - Chapter 1：Transformer 是怎么工作的
  - Chapter 2：使用 🤗 Transformers
  - Chapter 3：微调预训练模型
  - Chapter 6：Tokenizer 库
- 用途：了解工业级 Transformer 生态，即使我们做的是结构化数据也能借鉴训练流程。

### 2.4 核心论文（每周读 1 篇）

| 论文 | 年份 | 用途 | 阅读时机 |
|------|------|------|---------|
| Attention Is All You Need | 2017 | Transformer 开山之作 | W5-W7，跟着 d2l 精读 |
| BERT | 2018 | Encoder-only 范式 | W8 粗读 |
| Anomaly Transformer | ICLR 2022 | 时序异常检测 SOTA 思路 | W9 精读 |
| Informer | AAAI 2021 Best | 稀疏注意力长序列预测 | W11 精读 |
| PatchTST | ICLR 2023 | Patch 化时序，简单 SOTA | W11 精读 |
| Are Transformers Effective for Time Series Forecasting? (DLinear) | AAAI 2023 | 反思：Transformer 真的必要吗？ | W12 选读 |

论文工具：用 [alphaXiv](https://www.alphaxiv.org/) 或 [Papers with Code](https://paperswithcode.com/) 看代码实现。

### 2.5 时序/异常检测专项

- **Time-Series Library (tslib)**：<https://github.com/thuml/Time-Series-Library>
  - 包含 Informer / Autoformer / PatchTST / Anomaly Transformer 等统一实现
- **PyOD**：<https://github.com/yzhao062/pyod>
  - 经典异常检测算法（Isolation Forest, LOF, AutoEncoder）的 baseline 库

### 2.6 数据集

| 数据集 | 用途 | 链接 |
|--------|------|------|
| Credit Card Fraud Detection | W1-W2 入门，单笔交易二分类 | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| IEEE-CIS Fraud Detection | W10 进阶，更真实的特征 | [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection) |
| PaySim（合成支付数据） | W3-W4 序列建模，有 userId | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| 自己生成的合成交易流 | W7-W8 验证 Transformer | 见 W3 任务 |

---

## 3. 每周 10 小时时间分配模板

**固定时段**：周一到周五，3 个早上 + 2 个晚上（周六周日休息）

| 时段 | 时间 | 时长 | 类型 | 主要任务 |
|------|------|------|------|---------|
| 🌅 周一早 | 09:00–11:00 | 2h | 理论 | 教材章节阅读（d2l / Karpathy 视频 Part 1） |
| 🌙 周二晚 | 20:30–22:30 | 2h | 编码 | Colab 实验（实现当周核心模块） |
| 🌅 周三早 | 09:00–11:00 | 2h | 理论 + 论文 | 理论剩余 1h + 精读论文 1h |
| 🌙 周四晚 | 20:30–22:30 | 2h | 编码 | Colab 实验（跑通训练 + 调试） |
| 🌅 周五早 | 09:00–11:00 | 2h | 编码 + 复盘 | 编码收尾 1h + 周复盘笔记 1h |

**本周合计**：早上 6h + 晚上 4h = **10h**

**类型分配小计**（与原定比例一致）：

| 块 | 时长 | 分布 |
|----|------|------|
| 📖 理论阅读 | 3h | 周一 2h + 周三 1h |
| 💻 动手编码 | 5h | 周二晚 2h + 周四晚 2h + 周五早 1h |
| 📄 论文/思考 | 1h | 周三早 1h |
| 📝 复盘笔记 | 1h | 周五早 1h |

**排程逻辑**：
- **早上 9-11am（头脑清醒）**：留给需要思考的活——读书、读论文、写笔记。
- **晚上 8:30-10:30pm（容易进入心流）**：留给动手编码，一次 2h 能完整实现一个模块。
- **周二/周四两晚编码**：前一天早上学理论，当晚就动手实现，吸收最快。
- **周五早上复盘**：周末前收尾，把本周产物整理成可复用的 notebook。

**如果某天临时有事**：优先顺延到当周其他空档（不要挪到周末）；若整周进度掉队 20% 以上，考虑当前章节再用一周，别硬赶。

---

## 4. MVP 迭代路线图

贯穿 12 周的 MVP 拆解（每次迭代的产出可直接复制到新 notebook 继续演进）：

### v0.1 (W2) — MLP Baseline

- **输入形状**：`(B, F)`，F ≈ 30（Kaggle 信用卡数据集原始特征）
- **模型**：`Linear(F, 64) → ReLU → Dropout → Linear(64, 1)`
- **损失**：`BCEWithLogitsLoss`（注意正负样本极不平衡，用 `pos_weight`）
- **评估**：AUC-PR（不是 AUC-ROC！欺诈场景 PR 更重要）、Recall@FPR=0.001
- **预期效果**：AUC-PR ≈ 0.70~0.75
- **踩坑点**：不做 scaling 会不收敛；`V1-V28` 是 PCA 主成分，已脱敏

### v0.2 (W4) — LSTM Baseline

- **输入形状**：`(B, L, F)`，L=32（每个用户最近 32 笔）
- **模型**：`Embedding(或线性映射) → LSTM(hidden=128, 2层) → Linear(128, 1)`
- **数据处理**：按 userId groupby，按时间排序，滑动窗口
- **损失/评估**：同 v0.1
- **预期效果**：AUC-PR ≈ 0.80（序列信息帮助识别异常模式）
- **踩坑点**：LSTM 训练慢；padding/mask 处理；数据泄露（未来信息不能进训练集）

### v0.3 (W7-W8) — 手写 Transformer Encoder

- **输入形状**：`(B, L, F)` + positional encoding
- **模型**：从零实现 4 层 Encoder，d_model=64, heads=4, d_ff=128
- **数据**：先用合成数据（自己生成有明显异常规律的序列），验证模型能学到；再上 Kaggle
- **损失**：BCE（分类头接在 `[CLS]` token 或最后一个 token 上）
- **预期效果**：合成数据应 >95% 召回，Kaggle AUC-PR ≈ 0.82+
- **踩坑点**：Attention mask 错了模型会直接挂；学习率要用 warmup；初始化很重要（Xavier）

### v0.4 (W9-W10) — 双头 Transformer 异常检测

- **模型**：共享 Encoder + 两个头
  - Classification Head：BCE 损失（有监督）
  - Reconstruction Head：MSE 损失（无监督重构）
  - Loss = α·BCE + β·MSE
- **数据**：IEEE-CIS Fraud（更真实、更难）
- **技巧**：重构损失大 → 可能是异常（即使标签没标）
- **预期效果**：AUC-PR 接近 Kaggle 公开 top solution 的 70-80%
- **踩坑点**：类别不平衡需用 focal loss / undersampling / SMOTE；特征类型混合需要 categorical embedding

### v0.5 (W11-W12) — 时序 Transformer 变体

- **选型**：PatchTST（推荐，简单有效）或 Anomaly Transformer（更贴近异常检测场景）
- **PatchTST 思路**：把序列切 patch，每个 patch 看成一个 token（降低 attention 复杂度 + 更好的局部归纳偏置）
- **Anomaly Transformer 思路**：用 "Association Discrepancy"（先验关联 vs 序列关联）作为异常信号
- **目标**：理解论文 + 复现核心模块 + 与 v0.4 对比

---

## 5. 12 周详细计划

### Week 1 — 环境与问题定义（复习周 1）

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 1-2（序言 + 预备知识）；d2l Ch 3.1-3.4（线性回归） |
| 💻 编码 (5h) | ① Colab 环境：挂载 Drive、验证 GPU（`!nvidia-smi`）；② 下载 Kaggle 信用卡欺诈数据集；③ 用 pandas + seaborn 做 EDA：类别分布、特征相关性、时间分布；④ 写一个空壳项目：`notebooks/01_eda.ipynb`、`src/data.py` |
| 📄 论文 (1h) | 阅读 [Kaggle Credit Card Fraud 公开 notebook top 1](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)，学 EDA 和不平衡处理思路 |
| 📝 复盘 (1h) | 笔记："欺诈检测为什么是不平衡问题？AUC-PR 和 AUC-ROC 差别是什么？" |
| ✅ 验收 | 能在 Colab 跑通数据加载、画出 EDA 图；能解释为什么欺诈样本只占 0.17% |

### Week 2 — DL 基础复习 + v0.1 MLP Baseline（复习周 2）

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 4（MLP）、Ch 5.1-5.3（深度学习计算） |
| 💻 编码 (5h) | ① 写一个完整训练循环（train/val/test split、early stopping、保存 best model）；② 实现 MLP 模型，处理不平衡（`pos_weight` + 分层采样）；③ 实现 AUC-PR、Recall@FPR 评估；④ 产出 **MVP v0.1**：`notebooks/02_mlp_baseline.ipynb` |
| 📄 论文 (1h) | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) 摘要 + 结论 |
| 📝 复盘 (1h) | 笔记："训练循环的 5 个关键组件"、"Focal Loss 和加权 BCE 的区别" |
| ✅ 验收 | MLP 在验证集上 AUC-PR > 0.70；能解释每一行 PyTorch 代码 |

### Week 3 — 序列建模入门 + 合成交易数据生成

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 8.1-8.4（序列模型、文本预处理、语言模型）；Ch 9.1（RNN 思想） |
| 💻 编码 (5h) | ① 写合成交易数据生成器：模拟正常用户行为（时间周期、金额分布、商户偏好），注入 3 类异常（金额飙升、异地刷卡、高频小额）；② 把 Kaggle 数据改造成序列格式（按 userId + time 排序 + 滑窗 L=32）；③ 写一个 `src/dataset.py`，封装成 `torch.utils.data.Dataset` |
| 📄 论文 (1h) | [PaySim 论文](https://www.researchgate.net/publication/313138956_PaySim_A_financial_mobile_money_simulator_for_fraud_detection) 方法部分 |
| 📝 复盘 (1h) | 笔记："为什么交易序列比单笔更有信息？"、"滑动窗口 vs 完整序列的权衡" |
| ✅ 验收 | 合成数据 notebook 能画出正常/异常的差异图；序列化后的 Kaggle 数据形状 `(N, 32, F)` 正确；无数据泄露 |

### Week 4 — RNN/LSTM + v0.2 序列 Baseline

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 9.2-9.7（RNN/GRU/LSTM/深层 RNN/双向 RNN）；重点理解 **梯度消失** 和 **LSTM 门控如何缓解** |
| 💻 编码 (5h) | ① 实现 LSTM 分类器（PyTorch `nn.LSTM`）；② 处理变长序列（`pack_padded_sequence`）；③ 在合成数据 + Kaggle 上各训一版；④ 产出 **MVP v0.2**：`notebooks/03_lstm_baseline.ipynb` |
| 📄 论文 (1h) | 读 [Understanding LSTM Networks (colah's blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| 📝 复盘 (1h) | 笔记："画出 LSTM 的 4 个门，用自己的话描述它们的作用" |
| ✅ 验收 | LSTM 在合成数据 >95% AUC-PR；Kaggle 数据 AUC-PR > MLP baseline |

### Week 5 — Attention 直觉 + 手写 Self-Attention

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 10.1-10.5（注意力提示、Nadaraya-Watson、注意力打分函数、Bahdanau Attention、多头注意力） |
| 💻 编码 (5h) | ① 跟着 Karpathy **`Let's build GPT`** 视频 0:00–1:00 （self-attention 动机）；② 手写单头 Self-Attention：`Q = xW_q, K = xW_k, V = xW_v, Attention = softmax(QK^T/√d)V`；③ 不用 `nn.MultiheadAttention`，自己实现并跑通一个 toy 任务 |
| 📄 论文 (1h) | Attention Is All You Need 的 **Section 3.2（Attention）** 精读 |
| 📝 复盘 (1h) | 笔记："Self-Attention 相比 RNN 的 3 个优势"、"为什么要除以 √d_k" |
| ✅ 验收 | 能在白板上推导 scaled dot-product attention；手写的 attention 在合成数据上能过拟合 |

### Week 6 — Multi-Head Attention + Positional Encoding

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 10.6-10.7（自注意力和位置编码、Transformer 概述）；复习正弦/余弦 Positional Encoding 的数学 |
| 💻 编码 (5h) | ① 把 W5 的单头扩展为 Multi-Head（4 heads）；② 实现 Sinusoidal PE + Learnable PE 两版；③ 把 PE + MHA + FFN + LayerNorm + Residual 组装成一个 **Encoder Block**；④ 在合成数据上验证 |
| 📄 论文 (1h) | Attention Is All You Need 的 **Section 3.3-3.5（FFN/PE/Embeddings）** |
| 📝 复盘 (1h) | 笔记："为什么需要 PE？"、"LayerNorm 放 pre 还是 post，有什么影响？（Pre-LN 更稳定）" |
| ✅ 验收 | 能从零实现一个 Encoder Block，能跑通前向和反向 |

### Week 7 — 完整 Transformer Encoder + v0.3 初版

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 11.7（Transformer 完整实现）；对比自己的实现 |
| 💻 编码 (5h) | ① 堆 4 层 Encoder，加分类头（`[CLS]` token 或 mean pooling）；② 加 learning rate warmup（Noam schedule）；③ 在合成数据上验证能 >95% 召回异常；④ 产出 **MVP v0.3 第一版**：`notebooks/04_transformer_v1.ipynb` |
| 📄 论文 (1h) | [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) |
| 📝 复盘 (1h) | 笔记："我的实现和 d2l / nanoGPT 有哪些差异？为什么？" |
| ✅ 验收 | 合成数据上 AUC-PR > 0.95；能解释每个超参的作用 |

### Week 8 — nanoGPT 深读 + v0.3 完整版

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | 看 Karpathy **`Let's build GPT`** 完整视频（1h50m），重点 1:00–1:50（完整 GPT 构造） |
| 💻 编码 (5h) | ① clone nanoGPT，读 `model.py` 每一行；② 把 nanoGPT 改造成"交易序列分类器"：去掉 decoder causal mask（或保留作 autoregressive 异常检测）、换分类头；③ 在 Kaggle 数据上跑，产出 **MVP v0.3 完整版**：`notebooks/05_transformer_v2_kaggle.ipynb` |
| 📄 论文 (1h) | BERT 论文摘要 + Section 3（理解 Encoder-only 范式） |
| 📝 复盘 (1h) | 笔记："GPT (causal) vs BERT (bidirectional) 架构差异在哪几行代码？"、"哪种更适合我们的异常检测？" |
| ✅ 验收 | Kaggle AUC-PR > LSTM baseline；能改 nanoGPT 的任意一个模块并解释影响 |

### Week 9 — 无监督异常检测 MVP（v0.4 上）

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | 精读 Anomaly Transformer 论文（ICLR 2022）；理解 prior-association vs series-association |
| 💻 编码 (5h) | ① 把 Encoder 改成"重构模型"：输入序列 → 输出序列，MSE 损失；② 用重构误差作为异常分数；③ 在 Kaggle 和合成数据上评估；④ 对比有监督 vs 无监督效果 |
| 📄 论文 (1h) | [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/abs/1901.03407) 的 Transformer 部分 |
| 📝 复盘 (1h) | 笔记："无监督异常检测的 3 种范式（重构、预测、对比）"、"什么情况下选哪种" |
| ✅ 验收 | 无监督模型在无标签的合成数据上 AUC-PR > 0.80 |

### Week 10 — 双头异常检测 + 真实数据（v0.4 下）

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | d2l Ch 15.6-15.8（BERT 微调，借鉴结构化数据微调思路）；HuggingFace Course Ch 3 |
| 💻 编码 (5h) | ① 共享 Encoder + 双头（分类 + 重构）；② 切换到 IEEE-CIS Fraud 数据集（更真实）；③ 做 baseline 对比：IsolationForest、LSTM、MLP、Transformer（分类头）、Transformer（双头）；④ 产出完整对比表格 |
| 📄 论文 (1h) | IEEE-CIS 比赛 public solutions（看 feature engineering 思路） |
| 📝 复盘 (1h) | 笔记："我的 Transformer 双头 vs IsolationForest 差距在哪？为什么？" |
| ✅ 验收 | 完整对比表；Transformer 双头不差于 LSTM；能回答"特征工程 vs 模型，哪个更关键？" |

### Week 11 — 时序 Transformer 变体

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | 精读 Informer（稀疏注意力） + PatchTST（patching）两篇；理解 vanilla attention 在长序列上的瓶颈 |
| 💻 编码 (5h) | ① clone Time-Series-Library 仓库；② 跑通 Informer 和 PatchTST 的官方脚本；③ 选一个（推荐 PatchTST，更简单）替换 v0.4 的 backbone；④ 在长序列（L=256）版本的 Kaggle 数据上对比 |
| 📄 论文 (1h) | PatchTST 或 DLinear 反思论文（看时序 SOTA 的争议） |
| 📝 复盘 (1h) | 笔记："PatchTST 为什么有效？它的 2 个核心设计是什么？"（channel-independence + patching） |
| 📊 对比 | 长序列下 vanilla Transformer vs PatchTST 的参数量、训练时间、效果 |
| ✅ 验收 | 能说出 Informer/PatchTST 至少 3 处改动及动机 |

### Week 12 — 落地与收尾

| 块 | 任务 |
|----|------|
| 📖 理论 (3h) | 推理优化：[torch.compile 入门](https://pytorch.org/docs/stable/torch.compiler.html)、ONNX 导出、量化基础 |
| 💻 编码 (5h) | ① 把最佳模型导出 ONNX，对比 PyTorch vs ONNX Runtime 延迟；② 写一个简单的推理服务（FastAPI，在 Colab 用 `pyngrok` 暴露）；③ 做一张 **项目总结图**（Mermaid），梳理 12 周走过的路 |
| 📄 思考 (1h) | 思考："这个模型在真实产线会遇到什么问题？"（特征一致性、实时性、模型漂移、AB 测试、解释性） |
| 📝 复盘 (2h) | 写一篇 ~3000 字的 **12 周总结博客**，发 GitHub/个人博客；列出下一步学习方向（对比学习？图 Transformer？大模型微调？） |
| ✅ 验收 | 项目总结博客 + 完整代码仓库（GitHub private 即可）+ 清晰的"下一步"清单 |

---

## 6. Colab 工程模板

### 6.1 推荐目录结构

```
transformer-anomaly/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_mlp_baseline.ipynb
│   ├── 03_lstm_baseline.ipynb
│   ├── 04_transformer_v1.ipynb
│   ├── 05_transformer_v2_kaggle.ipynb
│   ├── 06_anomaly_unsupervised.ipynb
│   ├── 07_anomaly_dualhead.ipynb
│   ├── 08_patchtst.ipynb
│   └── 09_deployment.ipynb
├── src/
│   ├── data.py          # 数据加载 + 预处理
│   ├── dataset.py       # PyTorch Dataset
│   ├── models/
│   │   ├── mlp.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   ├── train.py         # 训练循环
│   ├── eval.py          # 评估指标
│   └── utils.py
├── configs/
│   └── base.yaml
├── data/                # Drive 软链
└── requirements.txt
```

### 6.2 Colab 必做（每次 notebook 开头）

```python
# 1. 挂载 Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 跳到项目目录
%cd /content/drive/MyDrive/transformer-anomaly

# 3. 装依赖（pin 版本，保证复现）
!pip install -q -r requirements.txt

# 4. 固定 seed
import random, numpy as np, torch
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 5. 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### 6.3 训练循环模板（可直接复用）

```python
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler=None, clip=1.0):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:           # 混合精度
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)
```

### 6.4 复现性清单

- [ ] `requirements.txt` pin 版本：`torch==2.2.0`、`numpy==1.26.*`、`pandas==2.1.*`
- [ ] 固定 seed（random / numpy / torch / cuda）
- [ ] 保存 best model：`torch.save({'state_dict': model.state_dict(), 'config': cfg}, path)`
- [ ] 记录实验：用 WandB（免费） 或简单的 CSV 记录每次运行

---

## 7. 验收标准与自检清单

### 7.1 每周末 3 问

1. **本周核心概念用一句话说清楚是什么？**
2. **本周 Colab 产出的 notebook 能独立跑通吗？** （明天打开能直接 Run all 吗？）
3. **如果下周别人问我"为什么这么做"，我能回答吗？**

### 7.2 季度末硬核问题（12 周结束时自问，能流畅回答即毕业）

**架构层面**
- Transformer 相比 RNN/CNN 的 3 个核心优势？（并行、长依赖、归纳偏置弱）
- Self-Attention 的计算复杂度是多少？为什么是 O(n²·d)？
- Pre-LN 和 Post-LN 的区别？为什么现代 Transformer 大多用 Pre-LN？
- Multi-Head 为什么比 Single-Head 好？（不同子空间捕获不同关系）
- 为什么需要 Positional Encoding？Sinusoidal 和 Learnable 的优劣？

**训练层面**
- Learning rate warmup 为什么重要？（Transformer 初期梯度不稳定）
- 为什么 Transformer 对 learning rate 敏感？
- Dropout 通常放在哪几个位置？
- Gradient clipping 的典型值是多少？

**异常检测层面**
- 分类（有监督）vs 重构（无监督）vs 对比学习，三种范式的适用场景？
- Anomaly Transformer 的 Association Discrepancy 是什么？
- 为什么欺诈检测用 AUC-PR 而不是 AUC-ROC？
- 类别不平衡的 5 种处理方法？

**时序 Transformer 层面**
- Informer 的 ProbSparse Attention 是怎么做的？
- PatchTST 为什么有效？它的 2 个核心设计？
- Transformer 对时序预测一定比 DLinear 好吗？（不一定，场景依赖）

**工程层面**
- 怎么在 Colab T4 上训练 1 亿参数模型？（梯度累积 + fp16 + 减 batch）
- Transformer 推理延迟怎么优化？（KV cache、flash attention、量化、ONNX）
- 模型上线后遇到特征漂移怎么办？

---

## 8. 常见坑与学习建议

### 8.1 典型坑位

- **数据泄露**：按 userId 切分训练/验证，不能随机切！时间也要前后切。
- **类别不平衡**：别一上来就 SMOTE，先试 `pos_weight` 或 focal loss，看 baseline 能到多少。
- **Attention mask**：padding mask 和 causal mask 别混！结构化数据一般只要 padding mask。
- **LayerNorm 位置**：Pre-LN（x + Attn(LN(x))）比 Post-LN（LN(x + Attn(x))）训练更稳定。
- **Learning rate**：Transformer 需要 warmup，常见 base_lr=1e-4，warmup 1000 steps。
- **评估指标**：欺诈场景绝对不要只看 Accuracy。

### 8.2 学习建议

- **写代码 > 看视频 > 看书**：5h 编码是铁律，不要挤压。
- **每周 push 一次 GitHub**：即使代码不完美，保持记录。
- **笔记用自己的话**：不要复制粘贴；画图解释。
- **遇到卡壳**：问自己"最小可运行例子是什么？" 一步步排查。
- **不要过度优化**：MVP 跑通比优雅重要。

### 8.3 社区与求助渠道

- PyTorch 论坛：<https://discuss.pytorch.org/>
- Papers with Code：<https://paperswithcode.com/task/anomaly-detection>
- Karpathy 的 Twitter/X：持续分享神经网络洞察
- 中文：李沐 B 站课程配套讨论区

---

## 附录 A：Week 1 立刻开始清单

如果今天就要启动，按这个顺序做：

1. ☐ 注册 Kaggle 账号，下载 [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 数据集
2. ☐ 在 Google Drive 建 `transformer-anomaly/` 文件夹，传数据集
3. ☐ 打开 Colab，新建 notebook `01_eda.ipynb`，跑通 [6.2 Colab 必做](#62-colab-必做每次-notebook-开头) 模板
4. ☐ 第一个实验：加载数据 + 画类别分布 + 画前 5 个特征的分布
5. ☐ 注册 WandB（可选但推荐），拿到 API key
6. ☐ 开一个 Notion/Obsidian 学习日志，每周记录

---

## 附录 B：每周时间粗算

| 周次 | 累计小时 | 累计进度 |
|------|---------|---------|
| W4 末 | 40h | LSTM baseline 完成（MVP v0.2） |
| W8 末 | 80h | 完整 Transformer + nanoGPT 改造（MVP v0.3） |
| W10 末 | 100h | 双头异常检测 + 真实数据对比（MVP v0.4） |
| W12 末 | 120h | 时序 Transformer + 落地总结（MVP v0.5） |

---

**最后**：学习 Transformer 的本质不是背架构，而是理解"为什么注意力机制如此强大"。12 周走下来，你会获得的不是"会用 Transformer"这一个技能，而是**用 Transformer 的视角看序列数据**的能力——这会让你在异常检测、推荐、时序预测任何场景都能快速落地。

祝学习顺利 🚀
