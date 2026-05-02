# Week 5 — Attention 直觉 + 手写单头 Self-Attention

> 目标：本周结束时，能在白板上推导 scaled dot-product attention，并用手写的 self-attention 在一个玩具任务上过拟合。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 10.1-10.3（注意力提示、Nadaraya-Watson、注意力打分函数） |
| 周二 晚 20:30–22:30 | 理论+视频 | d2l Ch 10.4-10.5 + Karpathy `Let's build GPT` 0:00–1:00（self-attention 动机） |
| 周三 早 09:00–11:00 | 编码 | `05_self_attention.ipynb` 前半：ScaledDotProductAttention + SingleHeadSelfAttention |
| 周四 晚 20:30–22:30 | 编码 | `05_self_attention.ipynb` 后半：训练 toy 任务 + 画 attention 热力图 + sqrt 缩放消融 |
| 周五 早 09:00–11:00 | 论文+复盘 | 精读 Attention Is All You Need §3.2 + 写三问笔记 |

## 起步资源

**主教材**：[动手学深度学习 PyTorch 版](https://zh.d2l.ai/) Ch 10.1-10.5

**必看视频**：[Karpathy — Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)（0:00–1:00 self-attention 动机部分）

**论文**：Vaswani et al. 2017 《Attention Is All You Need》§3.2 Scaled Dot-Product Attention

## 文件结构

```
week05/
├── README.md                  ← 本文件
├── 05_self_attention.ipynb    ← 本周主 notebook
└── requirements.txt           ← 与 week01 相同的核心 pin
```

## 本周验收

- [ ] 能在白板上从头推导 `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`
- [ ] 能用 1 分钟口头解释为什么除以 `sqrt(d_k)`（方差控制 → softmax 不饱和）
- [ ] 能列举 Self-Attention 相对 RNN 的 3 个优势（并行 / O(1) 路径长度 / 软字典查询）
- [ ] 手写 SingleHeadSelfAttention 在 toy 任务上能过拟合到 >98% 训练准确率
- [ ] sqrt(d_k) 消融实验：移除缩放后能肉眼看到 loss 收敛变慢 / 不稳定
- [ ] Attention 热力图能看出模型在"关注"相关位置（非均匀分布）

## 周五复盘三问

1. Self-Attention 相比 RNN 的 3 个优势？
2. 为什么要除以 `sqrt(d_k)` 而不是 `d_k` 或不除？
3. 去掉 softmax 会发生什么？换成 sigmoid 呢？
