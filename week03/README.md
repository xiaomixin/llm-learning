# Week 3 — 序列建模入门 + 合成交易数据生成

> 目标：产出合成交易数据集 + 把 Kaggle / 合成数据都切成 `(N, 32, F)` 的序列张量，为 Week 4 LSTM baseline 铺路。**严禁时间泄露**。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 8.1-8.4（序列模型、文本预处理、语言模型） |
| 周二 晚 20:30–22:30 | 编码 | `03_synth_data.ipynb`：200 用户 × 30 天 + 注入 3 类异常 + parquet |
| 周三 早 09:00–11:00 | 理论+论文 | d2l Ch 9.1（RNN 思想）+ 精读 [PaySim 论文](https://www.researchgate.net/publication/313138956_PaySim) 的方法部分 |
| 周四 晚 20:30–22:30 | 编码 | `03b_sequence_builder.ipynb`：时间切分 + 滑窗 L=32 + Dataset 封装 + 泄露自检 |
| 周五 早 09:00–11:00 | 编码+复盘 | 把 parquet / .pt 的产出 push GitHub + 写笔记 |

## 本周验收

- [ ] `03_synth_data.ipynb` 能跑出 `data/synth/transactions.parquet`，异常占比 2–5%
- [ ] 可视化图能清晰区分 `amount_spike / geo_jump / burst / normal`
- [ ] `03b_sequence_builder.ipynb` 产出形状检查：
  - 合成: `(N_s, 32, 7)`
  - Kaggle: `(N_k, 32, 29)`
- [ ] **时间泄露自检通过**：`train.max(ts) ≤ val.min(ts) ≤ test.min(ts)`
- [ ] 写入 `data/processed/{synth,kaggle}_{train,val,test}.pt`
- [ ] 笔记回答：
  - 为什么 `(user_id + ts)` 排序而不是随机？
  - 滑动窗口 vs 完整序列的权衡（训练样本量 vs 窗口外信息）
  - 窗口标签取"最后一笔"vs"窗口内任一异常"，业务语义差在哪？

## 文件结构

```
week03/
├── README.md
├── 03_synth_data.ipynb         ← 合成数据生成器（产出 parquet）
└── 03b_sequence_builder.ipynb  ← 序列构造（合成 + Kaggle，产出 .pt）
```

## 关键技术点

- **时间分位切分**：`q70 / q85` 作为 train/val/test 边界，保证"未来信息不进训练"。
- **Kaggle 没有 userId**：用 `(time_bucket × 97 + round(amount) × 53) mod 500` 生成 500 个伪用户——不完美但足够跑通管线；Week 4 对比时知道这是上限偏低的原因。
- **L=32**：按"最近 32 笔"建模，比"全量历史"样本多、batch 更均匀；后续 Week 8 我们会把 L 拉到 64/128 做对比。

## 周五复盘三问

1. 合成数据里 3 类异常，哪一类对 LSTM 最难？（猜测 + 下周验证）
2. Kaggle 的伪 userId 会不会过拟合某种虚假模式？能怎么验证？
3. 下周 LSTM 如果打不过 MLP，第一步 debug 在哪儿？
