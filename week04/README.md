# Week 4 — RNN/LSTM + 序列 Baseline (MVP v0.2)

> 目标：用 LSTM 处理 `(B, 32, F)` 的交易序列。合成数据 AUC-PR > 0.95，Kaggle 序列 AUC-PR 超过 Week 2 的 MLP baseline。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 9.2-9.7（RNN / GRU / LSTM / 深层 & 双向 RNN） |
| 周二 晚 20:30–22:30 | 编码 | `04_lstm_baseline.ipynb` 的 1–4 节：数据加载 + LSTM 模型 + 训练循环 |
| 周三 早 09:00–11:00 | 理论+论文 | [colah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| 周四 晚 20:30–22:30 | 编码 | 在合成 + Kaggle 上各训一版 → 对比表 + 混淆矩阵 |
| 周五 早 09:00–11:00 | 编码+复盘 | 写本周笔记 + 画 LSTM 4 门图 + push GitHub |

## 本周验收

- [ ] Notebook 一键 Run All 通过（`data/processed/*.pt` 缺失时 fallback 自动重建）
- [ ] **合成数据**：val AUC-PR > 0.95
- [ ] **Kaggle 序列**：test AUC-PR > Week 2 MLP（≈ 0.70）
- [ ] 对比表输出到 stdout，包含 `val_ap / test_ap / test_roc / test_rec@fpr=0.001`
- [ ] 两个 checkpoint 保存到 `PROJECT_ROOT/checkpoints/lstm_{synth,kaggle}.pt`
- [ ] 笔记画出 LSTM 4 个门并用自己的话描述作用
- [ ] 能解释 `pack_padded_sequence` 的使用场景（本任务定长不需要，但知道何时需要）

## 文件结构

```
week04/
├── README.md
├── 04_lstm_baseline.ipynb   ← MVP v0.2 主 notebook（synth + kaggle 各一版）
└── requirements.txt
```

## 关键技术点

- **最后时刻 pooling**：`out[:, -1, :]` 作为分类表征；Week 5+ 会换成 attention pooling 对比。
- **Dropout 位置**：LSTM 层间（`dropout=0.2` 只作用在非最后一层），分类头前 `Dropout(0.3)`。
- **梯度裁剪**：`clip_grad_norm_=1.0`，LSTM 容易梯度爆炸，必加。
- **`pack_padded_sequence` 模板**：代码里保留注释，Week 7+ 处理变长序列直接抄。

## 周五复盘三问

1. 合成达标了吗？Kaggle 赢 MLP 多少？差值合理吗？
2. 训练日志里 loss 下降稳定还是反复？如果反复，是 lr 还是 pos_weight 问题？
3. 下周进 Attention，最想替换掉 LSTM 哪个部分？（提示：取最后时刻是**压缩**，attention 是**加权保留**）
