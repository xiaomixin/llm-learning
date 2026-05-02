# Week 2 — DL 基础复习 + MLP Baseline (MVP v0.1)

> 目标：在 Kaggle 信用卡欺诈数据上跑通完整训练/评估管线，产出 **MVP v0.1**，验证集 AUC-PR > 0.70，能解释每一行 PyTorch 代码。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 4（MLP）：多层感知机、激活函数、前向/反向、过拟合/欠拟合 |
| 周二 晚 20:30–22:30 | 编码 | `02_mlp_baseline.ipynb` 的 1–5 节：数据切分 + Dataset + MLP + 加权 BCE |
| 周三 早 09:00–11:00 | 理论+论文 | d2l Ch 5.1-5.3（层与块、参数管理、延后初始化）+ 精读 [Focal Loss](https://arxiv.org/abs/1708.02002) 摘要/结论 |
| 周四 晚 20:30–22:30 | 编码 | 训练循环 + early stopping + 完整评估（AUC-PR / AUC-ROC / Recall@FPR / 混淆矩阵） |
| 周五 早 09:00–11:00 | 编码+复盘 | Focal Loss 对比 cell + 写本周笔记 + GitHub push |

## 起步资源

**主教材**：[动手学深度学习](https://zh.d2l.ai/) Ch 4-5

**关键论文**：
- [Focal Loss for Dense Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)
- 参考 [Kaggle 公开 baseline notebooks](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)（但不要抄结论，自己算一遍）

## 文件结构

```
week02/
├── README.md                 ← 本文件
├── 02_mlp_baseline.ipynb     ← MVP v0.1 主 notebook（训练 + 评估 + Focal Loss 对比）
└── requirements.txt          ← 依赖 pin（与 Week 1 对齐 + sklearn）
```

## 本周验收

- [ ] Notebook 一键 Run All 通过
- [ ] 验证集 **AUC-PR > 0.70**（预期 0.70–0.78）
- [ ] 训练日志显示 early stopping 真的触发过
- [ ] best model checkpoint 保存到 `PROJECT_ROOT/checkpoints/mlp_baseline.pt`
- [ ] Test 集 Recall@FPR=0.001 ≥ 0.5（业务可接受的"误拦 0.1%"阈值）
- [ ] 能指着代码讲出 5 个关键组件：`zero_grad / forward / loss / backward / step`
- [ ] 把 Focal Loss cell 切换跑一次，记录与加权 BCE 的差距
- [ ] 笔记回答：
  - 为什么用 `BCEWithLogitsLoss` 而不是 `Sigmoid + BCELoss`？（数值稳定）
  - `pos_weight` 的数学含义？（正样本梯度 × pos_weight）
  - Focal Loss 的 α 和 γ 各自调节什么？
  - 为什么 Scaler 只在训练集上 fit？（防止信息泄露）

## 周五复盘三问

1. MVP v0.1 的 AUC-PR 达标了吗？如果没到 0.70，瓶颈是特征、模型、还是损失？
2. 今天打开 notebook 能一键 Run All 吗？
3. 下周要生成合成数据 + 构造序列，我最担心的环节是什么？
