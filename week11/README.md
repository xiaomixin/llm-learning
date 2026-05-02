# Week 11 — 时序 Transformer 变体 (MVP v0.5)：PatchTST

> 目标：本周结束时，能说出 Informer / PatchTST 在 vanilla attention 上做了至少 3 处改动并给出动机；在 L=256 的长序列合成数据上，用 PatchTST 替换 v0.4 的 backbone，产出「参数量 / 显存 / 训练时长 / AUC-PR」四维对比表。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | 精读 Informer (AAAI 2021 Best Paper) § 3：ProbSparse attention + self-attention distilling |
| 周二 晚 20:30–22:30 | 理论 | 精读 PatchTST (ICLR 2023) § 3：channel-independence + patching；读 DLinear 反思论文 |
| 周三 早 09:00–11:00 | 编码 | `11_patchtst.ipynb` §1-3：长序列合成数据 + patching backbone（约 120 行） |
| 周四 晚 20:30–22:30 | 编码 | `11_patchtst.ipynb` §4-6：ProbSparse 小型 ablation + vanilla baseline + 三方对比表 |
| 周五 早 09:00–11:00 | 编码+复盘 | `11_patchtst.ipynb` §7-8：训练曲线 + 反思笔记 |

## 起步资源

**论文**：
- [Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021 Best)](https://arxiv.org/abs/2012.07436)
- [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST, ICLR 2023)](https://arxiv.org/abs/2211.14730)
- [Are Transformers Effective for Time Series Forecasting? (DLinear, AAAI 2023)](https://arxiv.org/abs/2205.13504)

**参考代码库**（仅读，不必 clone）：
- [Time-Series-Library (THUML)](https://github.com/thuml/Time-Series-Library)
- [PatchTST 官方实现](https://github.com/yuqinie98/PatchTST)

## 文件结构

```
week11/
├── README.md               ← 本文件
├── 11_patchtst.ipynb       ← 周三-周五：PatchTST + ProbSparse ablation + 对比
└── requirements.txt        ← 依赖 pin 版本（与 W07/W08 相同基底）
```

## 本周验收

- [ ] 能说出 Informer 至少 3 处改动及动机（ProbSparse attention / self-attention distilling / 生成式 decoder 一次出全长度）
- [ ] 能说出 PatchTST 两个核心设计：channel-independence + patching，并解释为什么它能把 attention 的 `O(L²·d)` 降到 `O((L/S)²·d)`
- [ ] 在 L=256 长序列合成数据上完成 vanilla Transformer / ProbSparse / PatchTST 三方训练
- [ ] 产出并填写如下对比表（notebook 内）

| Backbone | 参数量 | Peak GPU Mem | 每 epoch 壁钟时长 | Final AUC-PR |
|----------|--------|--------------|-------------------|--------------|
| Vanilla Transformer | _____ | _____ MB | _____ s | _____ |
| ProbSparse (Informer-lite) | _____ | _____ MB | _____ s | _____ |
| PatchTST | _____ | _____ MB | _____ s | _____ |

- [ ] 笔记回答："PatchTST 为什么有效？" + "什么情况下该选 PatchTST 而不是 vanilla Transformer？"

## 周五复盘三问

1. Vanilla attention 在 L=256 下的 `L²` 瓶颈有多明显？你看到的显存/时长增长是否匹配理论？
2. PatchTST 的 channel-independence 在欺诈检测场景（多模态异构特征）是优势还是劣势？为什么？
3. 下周要做推理优化，PatchTST 和 vanilla 哪一个更适合 ONNX 导出？为什么？
