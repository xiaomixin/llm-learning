# Week 7 — 完整 Transformer Encoder + MVP v0.3 v1

> 目标：从零搭建一个 4 层 Transformer Encoder，加上分类头 + Noam warmup，在合成交易数据上达到 AUC-PR ≥ 0.95，并能逐条解释每个超参的作用。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 11.7（Transformer 完整实现）— 逐行对照自己 W6 的 Encoder Block |
| 周二 晚 20:30–22:30 | 编码 | 堆 4 层 Encoder + Sinusoidal PE + 分类头（mean pool 版） |
| 周三 早 09:00–11:00 | 编码 | 加 `[CLS]` token 版本 + Noam warmup schedule + 绘制 LR 曲线 |
| 周四 晚 20:30–22:30 | 编码 | 合成数据生成（小号版 W3）+ 训练循环 + 早停 |
| 周五 早 09:00–11:00 | 编码+复盘 | 评估 AUC-PR / Recall@FPR + 注意力可视化 + 写差异复盘 |

## 起步资源

**主教材**：
- [动手学深度学习 PyTorch 版 Ch 11.7 — Transformer](https://zh.d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

**论文**：《Attention Is All You Need》重读 Section 5.3（Optimizer + Noam schedule）

## 文件结构

```
week07/
├── README.md                    ← 本文件
├── 07_transformer_v1.ipynb      ← 全周主 notebook（合成数据 + 完整 Encoder）
└── requirements.txt             ← 依赖 pin 版本
```

## 本周关键超参（务必能解释每一项）

| 超参 | 取值 | 作用一句话 |
|------|------|-----------|
| `d_model` | 64 | token 向量维度；所有子层输入输出都是它 |
| `num_heads` | 4 | 每头维度 = 64/4 = 16；多头让模型关注不同子空间 |
| `num_layers` | 4 | 堆越深越能组合特征，但 T4 上 4 层已够验证 |
| `d_ff` | 128 | FFN 扩展维度，一般 2-4× d_model |
| `dropout` | 0.1 | Attention + FFN 后都加，防过拟合 |
| `warmup` | 1000 step | Noam schedule：前 1000 步线性升 lr，之后按 step^-0.5 衰减 |
| `pos_weight` | neg/pos | BCE 里给正样本加权，对抗类别不平衡 |
| `seq_len L` | 32 | 每条用户交易序列长度（含当前交易） |

## 本周验收

- [ ] 能跑通 `07_transformer_v1.ipynb` 全部 cells（Colab T4 端到端 ≤ 15 min）
- [ ] 合成数据验证集 **AUC-PR ≥ 0.95**
- [ ] `mean pool` vs `[CLS] token` 两个变体都跑过，结果对比写在 notebook 里
- [ ] Noam LR 曲线图已绘制（前 1000 步上升，之后 step^-0.5 衰减）
- [ ] 对 1 条异常 + 1 条正常序列画了 layer-4 head-0 的 attention heatmap
- [ ] 复盘 cell 写下至少 3 处"我的实现 vs d2l 11.7"差异

## 周五复盘三问

1. Pre-LN 和 Post-LN 放置位置不同，为什么 Pre-LN 更稳定？（从梯度路径角度说）
2. Noam schedule 为什么不直接用 cosine？warmup 解决了什么实际问题？
3. 分类任务里 mean pool 和 `[CLS]` token 各自优劣？你这次的数据选哪个？为什么？
