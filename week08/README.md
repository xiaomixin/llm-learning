# Week 8 — nanoGPT 深读 + Kaggle 欺诈 MVP v0.3 v2

> 目标：用 nanoGPT 风格重构上周的 Encoder（CausalSelfAttention + Block + Transformer，< 150 行），在 Kaggle 信用卡欺诈真实数据上打赢 MLP baseline，并能讲清 GPT causal 与 BERT bidirectional 的 3 行关键差异。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | 看 Karpathy *Let's build GPT* 视频 1h50m，重点 1:00–1:50 |
| 周二 晚 20:30–22:30 | 编码 | 读 nanoGPT `model.py` 每一行；仿写 CausalSelfAttention / Block / Transformer |
| 周三 早 09:00–11:00 | 编码 | Kaggle CSV → 时间序列窗口；时间切分 70/15/15（不泄露） |
| 周四 晚 20:30–22:30 | 编码 | 训练 + MLP baseline + 早停；评估 AUC-PR |
| 周五 早 09:00–11:00 | 编码+复盘 | causal=True vs False 对比实验 + 注意力可视化 + 保存 checkpoint |

## 起步资源

**主教材**：
- [nanoGPT GitHub](https://github.com/karpathy/nanoGPT) — 重点读 `model.py`
- [Let's build GPT (Karpathy, YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [BERT 论文 Section 3 (Encoder-only pretraining)](https://arxiv.org/abs/1810.04805)

## 文件结构

```
week08/
├── README.md                        ← 本文件
├── 08_transformer_kaggle.ipynb      ← 主 notebook（nanoGPT refactor + Kaggle）
└── requirements.txt                 ← 依赖 pin 版本
```

## 本周关键设计点

| 设计点 | 本 notebook 的选择 | 原因 |
|--------|-----------|------|
| `causal: bool` | **False**（默认） | 异常检测要看整窗上下文，非自回归任务 |
| 序列构造 | 按 `Time` 排序 + pseudo userId 分组 + 滑窗 L=32 | Kaggle 没 user 列，用 `Time bucket + Amount hash` 近似 |
| Split 策略 | **时间切分 70/15/15** | 防数据泄露（未来信息不能进训练） |
| Baseline | 同输入重训 MLP（序列 flatten） | 公平对比，展示 Transformer 优势来自结构而非数据 |
| Pooling | `mean` + `[CLS]` 两种 | 承接 W7 结论，继续对比 |
| Checkpoint | `PROJECT_ROOT/models/transformer_v2.pt` | 供 W9 无监督版本加载 |

## 本周验收

- [ ] 跑通 `08_transformer_kaggle.ipynb`（Colab T4 端到端 ≤ 25 min）
- [ ] **Kaggle 验证集 AUC-PR > MLP baseline** 并把差距写在 notebook 里
- [ ] `causal=True` 对照实验已跑，结果比 `causal=False` 低——解释"为什么双向适合异常检测"
- [ ] 对 1 条欺诈窗口绘制 attention heatmap，能指出模型关注哪些历史交易
- [ ] 复盘 cell 写下 GPT vs BERT 在代码上的差异（具体到哪几行）
- [ ] Checkpoint 保存在 `PROJECT_ROOT/models/transformer_v2.pt`

## 周五复盘三问

1. nanoGPT 里 `tril` 下三角 mask 那一行去掉后，整个模型的"语义"变了什么？
2. 时间切分 vs 随机切分，Kaggle 上你观察到 AUC-PR 差多少？为什么？
3. 如果下周要做"无监督重构"，本周的 Encoder 需要改哪几行？（提前想）
