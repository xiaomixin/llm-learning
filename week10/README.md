# Week 10 — 双头 Transformer + IEEE-CIS Fraud (v0.4 下)

> 目标：共享 Encoder + 两个头（分类 + 重构）合训，在 IEEE-CIS Fraud 数据集上跟 IsolationForest / MLP / LSTM / 分类头 Transformer 做完整 benchmark，产出可交付的对比表。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 15.6-15.8 (BERT 微调思路) + HuggingFace Course Ch 3 |
| 周二 晚 20:30–22:30 | 编码 | IEEE-CIS 数据下载、数值/类别特征分离、Embedding 层搭建 |
| 周三 早 09:00–11:00 | 编码 | 双头 Transformer 搭建；小规模 sanity check；α / β 扫参 |
| 周四 晚 20:30–22:30 | 编码 | 跑完 5 个 baseline (IsolationForest, MLP, LSTM, TF-cls, TF-dual-head)，收集指标 |
| 周五 早 09:00–11:00 | 编码+复盘 | 特征消融（numerical / categorical / temporal）；写对比表与反思 |

## 起步资源

**主要数据**：
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)（竞赛数据，需加入竞赛并接受条款）
- Fallback：如果竞赛下载失败，notebook 会自动切到 `mlg-ulb/creditcardfraud` 并在输出里明显警告。

**参考阅读**：
- IEEE-CIS 公开 solutions：先读 Konstantin Yakovlev 1st place 简述，重点学 feature engineering 思路（不复现 magic features）。
- [HuggingFace Course Ch 3 — Fine-tuning](https://huggingface.co/course/chapter3)

## 文件结构

```
week10/
├── README.md                ← 本文件
├── requirements.txt         ← 依赖
└── 10_dual_head.ipynb       ← 本周主 notebook
```

## 本周验收

完整对比表必须包含下列列，所有 baseline 在**同一 train/val/test**上评估：

| Model | AUC-PR | AUC-ROC | Recall@FPR=0.001 | #Params | Training Time |
|-------|--------|---------|------------------|---------|---------------|
| IsolationForest          |   |   |   |   |   |
| MLP                      |   |   |   |   |   |
| LSTM                     |   |   |   |   |   |
| Transformer (cls-only)   |   |   |   |   |   |
| **Transformer (dual)**   |   |   |   |   |   |

- [ ] 表格完整、数字真实可复现
- [ ] 双头 Transformer 的 AUC-PR ≥ LSTM baseline
- [ ] 特征消融图：numerical / categorical / temporal 各自被 mask 后 AUC-PR 下降幅度
- [ ] 笔记回答："在这份数据上，特征工程 vs 模型架构谁更重要？数字说话。"

## 周五复盘三问

1. 如果我只能选一个提升路径：调优 Transformer 还是造 5 个新特征，会怎么选？数据支持？
2. 双头 (α·cls + β·recon) 带来的增益来自哪里？把 β=0 ablation 跑一下看。
3. 生产上我最想把这套扔掉的一刻会是什么？（速度？解释性？drift？）
