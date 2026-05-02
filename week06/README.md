# Week 6 — Multi-Head Attention + Positional Encoding + Encoder Block

> 目标：本周结束时，能从零实现多头注意力 + 位置编码 + 完整的 Encoder Block，并在 Week 5 的 toy 任务上跑赢单头模型。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 10.6（自注意力和位置编码）+ 10.7（Transformer 概述） |
| 周二 晚 20:30–22:30 | 编码 | `06_mha_pe.ipynb` 前半：MultiHeadAttention + mask + Positional Encoding |
| 周三 早 09:00–11:00 | 编码 | `06_mha_pe.ipynb` 中段：EncoderBlock (Pre-LN) + 形状/参数验证 |
| 周四 晚 20:30–22:30 | 编码 | 训练 + 与 Week 5 单头对比 + PE/Post-LN 消融 |
| 周五 早 09:00–11:00 | 论文+复盘 | 精读 Attention Is All You Need §3.3-3.5 + 写三问笔记 |

## 起步资源

**主教材**：[动手学深度学习 PyTorch 版](https://zh.d2l.ai/) Ch 10.6-10.7

**论文**：Vaswani et al. 2017 《Attention Is All You Need》§3.3 Position-wise FFN / §3.4 Embeddings and Softmax / §3.5 Positional Encoding

**参考**：[The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

## 文件结构

```
week06/
├── README.md              ← 本文件
├── 06_mha_pe.ipynb        ← 本周主 notebook
└── requirements.txt
```

## 本周验收

- [ ] 能从零实现 MultiHeadAttention（`reshape` + `transpose` 拆头，不调 `nn.MultiheadAttention`）
- [ ] 能实现并可视化 Sinusoidal PE 和 Learnable PE 的矩阵
- [ ] 能解释 Pre-LN 和 Post-LN 的区别，以及为什么 Pre-LN 训练更稳定
- [ ] EncoderBlock 输入 `(B=4, L=8, d=32)` → 输出 `(4, 8, 32)`，反向传播可执行
- [ ] 2 层 EncoderBlock 在 Week 5 toy 任务上准确率 ≥ 单头结果，收敛更快
- [ ] PE 消融：去掉 PE 后 loss 几乎不下降（因为自注意力对位置置换不变）
- [ ] Post-LN 消融：在较高学习率下 Post-LN 明显不稳定 / 发散

## 周五复盘三问

1. 为什么 Transformer 需要 positional encoding？（不加会怎样？）
2. Pre-LN 和 Post-LN 的区别是什么？为什么 Pre-LN 更稳定？
3. 多头为什么比单头好？每个头学到的东西一样吗？
