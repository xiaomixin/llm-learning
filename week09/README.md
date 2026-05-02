# Week 9 — 无监督异常检测 MVP (v0.4 上)

> 目标：抛开标签，只用"正常"样本训练一个重构 Transformer，用重构误差作为异常分数；理解无监督异常检测三大范式（重构 / 预测 / 对比），精读 Anomaly Transformer 论文并实现简化版 Association Discrepancy。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | Anomaly Transformer (ICLR 2022) 论文 Section 1-3；画出 prior-assoc vs series-assoc 的示意图 |
| 周二 晚 20:30–22:30 | 编码 | 合成数据生成器 + 重构 Encoder 搭建（d=64、4 层），跑通 shape |
| 周三 早 09:00–11:00 | 编码 | 训练 20 epochs，画重构误差分布（正常 vs 3 种异常），目标 AUC-PR > 0.80 |
| 周四 晚 20:30–22:30 | 编码 | Kaggle creditcardfraud 序列重构，对比 W7/W8 有监督 baseline |
| 周五 早 09:00–11:00 | 编码+复盘 | 实现 ~40 行的 Association Discrepancy 消融；整理笔记 |

## 起步资源

**论文**：
- [Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy](https://arxiv.org/abs/2110.02642) — ICLR 2022
- [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/abs/1901.03407) — 读 Transformer 章节

**数据集**：
- 合成数据：notebook 里内联生成（AR(1) + 正弦 + 噪声 + 3 类异常注入）
- [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)（按 Time 切分，与 Week 8 对齐）

**参考仓库**：
- [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer)（只读官方实现 `model/AnomalyTransformer.py`）

## 文件结构

```
week09/
├── README.md                   ← 本文件
├── requirements.txt            ← 依赖
└── 09_recon_anomaly.ipynb      ← 本周主 notebook
```

## 本周验收

- [ ] 能用一句话说清无监督异常检测 3 种范式各自的假设
- [ ] 合成数据上：纯重构 AUC-PR ≥ 0.80
- [ ] Kaggle 数据上：跑出无监督 vs W7/W8 有监督的对比数字（预期无监督会低一些，知道差距有多大）
- [ ] Association Discrepancy 简化版能跑通；写下它相比纯重构多/少学到了什么
- [ ] 笔记回答：生产环境里什么时候会选无监督？（label scarcity / concept drift / novel fraud types 至少举 2 个）

## 周五复盘三问

1. 我的重构模型对"点异常 / 上下文异常 / 集体异常"哪种最有效？为什么？
2. Anomaly Transformer 的"prior-association"和 kernel smoothing 有什么异同？
3. 如果让我在生产上线一套无监督方案，我最担心的 3 个坑是什么？
