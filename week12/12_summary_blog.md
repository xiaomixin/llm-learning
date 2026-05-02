# 从 MLP 到 PatchTST：12 周自学 Transformer 做交易欺诈检测

> 这是我按照一份 [12 周 Transformer 自学计划](../transformer-12week-plan.md) 走下来的完整记录。全程每周 10 小时（早 6h + 晚 4h），产出了 9 个 notebook、5 个 MVP 版本，并最终把模型导成 ONNX 用 FastAPI 服务化。
>
> 这篇文章不是纯技术讲解，而是「走一遍真实坑位」的流水账——哪一步 AUC-PR 跳了多少、哪一步踩了什么坑、回头看哪一步最值得做、哪一步其实可以跳过。

---

## 1. 动机：为什么用 12 周来学一个「老」模型？

2024 年往后，新手学 DL 有两个明显困惑：

1. **LLM 把所有注意力都抢走了**，以至于「Transformer」几乎变成「LLM」的代名词。但 Transformer 是一个更基础的架构，在风控、时序预测、推荐、语音这些领域照样是主力。
2. **教程两极分化**：要么是「15 分钟调用 HuggingFace」的浅层复制粘贴；要么是「从零手写 GPT-2」的硬核 2000 行。中间那一段——**按业务场景从 baseline 一步步迭代到 Transformer**——很少有人完整讲。

我想填这一段。选的业务场景是**交易欺诈检测**：数据天然不平衡、天然带时序、既可以做监督又可以做无监督，正好能把 MLP / LSTM / Transformer / PatchTST 串一条线。

目标是 12 周后能独立回答这三个问题：

- Transformer 为什么能打败 LSTM？具体打败在哪些 case 上？
- 长序列下 vanilla Transformer 的瓶颈在哪？PatchTST / Informer 怎么解？
- 训完之后，这个模型如何真正被「用起来」？

---

## 2. 数据集总览

| 数据集 | 用法 | 样本量 | 正例比 |
|--------|------|--------|--------|
| Credit Card Fraud (Kaggle) | W1-W6 入门 + baseline | 284,807 单笔交易 | 0.173% |
| 自造合成交易序列 | W3-W8 验证模型能学到时序 pattern | 2k-10k 序列 | 5-10% |
| IEEE-CIS Fraud (Kaggle) | W10 真实进阶 | 590k | 3.5% |
| 自造长序列 (L=256) | W11 PatchTST 对比 | 2k | 5% |

**关键取舍**：前 6 周全部只看 Kaggle 小数据 + 合成数据，是为了把时间花在「理解模型」上而不是「理解 pandas」上。W10 才切到 IEEE-CIS 这种真实粗糙数据集。

---

## 3. v0.1 — MLP Baseline（Week 2-3）

最朴素一版：把每一笔交易的 30 维特征直接过 `Linear(30, 64) → ReLU → Dropout → Linear(64, 1)`，`BCEWithLogitsLoss` + `pos_weight` 处理不平衡。

- **训练时长**：<!-- TODO: your result numbers --> s/epoch on T4
- **验证 AUC-PR**：<!-- TODO: your result numbers -->
- **Recall @ FPR=0.001**：<!-- TODO: your result numbers -->

**学到**：

- 不做 scaling 直接挂（V1-V28 虽然是 PCA 脱敏后的，但 Amount 没脱敏）。
- `pos_weight = (1 - π) / π`（π 是正例比）基本就够了，不一定要上 Focal Loss。
- 评估指标这里定型：**AUC-PR 主，AUC-ROC 参考**，理由：极度不平衡时 ROC 会过度乐观。

**心得**：花一整周在 MLP 上值得。很多后面 Transformer 的坑（学习率、正则、scaling）都是在这里第一次遇到的。

---

## 4. v0.2 — LSTM Baseline（Week 4）

把相同特征按用户/按时间排序，滑动窗口切成 `(B, L=32, F)`，LSTM 两层 hidden=128。

- **验证 AUC-PR**：<!-- TODO: your result numbers -->
- **相比 v0.1 的 Δ**：<!-- TODO: your result numbers -->

**学到**：

- 序列信息确实帮了忙（AUC-PR 往上跳了一档），但不是所有欺诈都有序列 pattern——单点型欺诈 LSTM 反而不如 MLP。
- Padding/mask 坑真不少：前 pad 还是后 pad、mask 放在 loss 还是放在 forward，直接决定模型能不能收敛。
- **时间泄露**：最容易犯也最致命——训练集的滑动窗口不能看到未来交易，val/test 切分要按**时间**切而不是按**用户**切。

---

## 5. v0.3 — 手写 Transformer Encoder（Week 5-8）

这是整个 12 周最重要的四周。

**W5-W6 理论阶段**：跟着 d2l 第 10 章从 scaled dot-product attention 推到 multi-head，再到 positional encoding。过程中不断回头看 "Attention Is All You Need" 原文。

**W7 合成数据阶段**：自己生成 `(B, L=64, F=8)` 的合成交易序列，故意注入明显的异常 pattern（随机位置的 spike + 通道偏置），验证手写 Transformer 能把这类规律学到 95%+ 召回。这一步是**信心支柱**——等真跑 Kaggle 效果不好时，你能确定「是数据难，不是模型写错了」。

**W8 真实数据阶段**：切到 Kaggle，d_model=64, n_heads=4, n_layers=4, 加 warmup。

- **验证 AUC-PR**：<!-- TODO: your result numbers -->
- **训练时长 / epoch**：<!-- TODO: your result numbers -->

**踩的坑**：

- Attention mask 搞反（`0` 应该是 attend，`-inf` 应该是屏蔽，PyTorch 和 HF 的 convention 不一样）。
- 不加 warmup 直接挂——AdamW + 1k step 的 linear warmup 是救命稻草。
- Xavier vs Kaiming 初始化差一个量级的梯度稳定性。

**心得**：手写一遍 Encoder 是值得的。`nn.TransformerEncoderLayer` 能用，但你必须先知道它里面是什么，才能在调试时不瞎蒙。

---

## 6. v0.4 — 双头 Transformer（Week 9-10）

W9：把 Encoder 改成重构模型（输入序列 → 输出序列，MSE 损失），用重构误差当异常分数。这是无监督路线的典型做法。

W10：共享 Encoder + 两个头——分类头（BCE）+ 重构头（MSE），loss = α·BCE + β·MSE。数据切到 **IEEE-CIS Fraud**（更真实、带大量 categorical 特征）。

加入了对比 baseline：IsolationForest / MLP / LSTM / Transformer（单分类头）/ Transformer（双头）。

| Baseline | AUC-PR |
|----------|--------|
| IsolationForest | <!-- TODO: your result numbers --> |
| MLP | <!-- TODO: your result numbers --> |
| LSTM | <!-- TODO: your result numbers --> |
| Transformer 单头 | <!-- TODO: your result numbers --> |
| Transformer 双头 | <!-- TODO: your result numbers --> |

**结论**：

- 双头 > 单头，但是差距不如想象中大。α/β 权重调起来很烦。
- 特征工程的重要性在 IEEE-CIS 上第一次被放大——**好特征 + MLP > 烂特征 + Transformer**，这个真实世界的朴素规律必须记住。
- IsolationForest 作为无监督 baseline 的性价比之高，也是我低估的。生产上先用 IF 做 guardrail、Transformer 做提升，是很务实的组合。

---

## 7. v0.5 — PatchTST（Week 11）

到 W11 把序列长度拉到 `L=256`，vanilla Transformer 开始明显吃紧——显存和 epoch 时间都按 `L²` 涨。

读 Informer 和 PatchTST 两篇论文。主要结论两句话：

- **Informer**：只有一小部分 query 真正活跃，用 `M(q_i, K) = max - mean` 稀疏性指标挑 top-u 个 query 算 softmax，其它 query 用 V 的均值填回。复杂度从 `O(L²)` 降到 `O(L·lnL)`。
- **PatchTST**：把 `L=256` 的序列切成长度 16、stride 8 的 patch，得到 31 个 token 后再过 Transformer。附带两个杀手锏：channel-independence（每个变量独立走 encoder）+ instance norm。

W11 我手写了 PatchTST（~120 行）和 ProbSparse attention mini-ablation（~50 行），和 vanilla 对比：

| Backbone | Params | Peak GPU Mem | Epoch 时长 | AUC-PR |
|----------|--------|--------------|-----------|--------|
| Vanilla Transformer | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> MB | <!-- TODO: your result numbers --> s | <!-- TODO: your result numbers --> |
| Informer-lite (ProbSparse) | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> MB | <!-- TODO: your result numbers --> s | <!-- TODO: your result numbers --> |
| PatchTST | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> MB | <!-- TODO: your result numbers --> s | <!-- TODO: your result numbers --> |

**学到**：

- Patching 真的管用。token 数从 256 降到 31，显存几乎线性下降。
- Channel-independence 在欺诈场景有争议——跨变量 attention 也许本来就带信息，独立跑会把这部分扔掉。**什么时候选 PatchTST**：序列长、变量间量纲/语义差异大、显存吃紧。**什么时候选 vanilla**：短序列、变量耦合强、任务偏重构。
- DLinear 的反思论文提醒我：很多时候简单的线性层 + 归一化就能打败花哨的 Transformer，上模型之前先跑 DLinear 做 sanity check。

---

## 8. Deployment（Week 12）

最后一周是「把模型变成服务」的一周。

**torch.compile**：默认 mode，Warmup 后 100 次 forward 基准，得到 speedup <!-- TODO: your result numbers --> x。

**ONNX 导出**：`torch.onnx.export` + `dynamic_axes={'x': {0: 'batch'}}`，opset_version=17。用 32 个随机 batch 验证 PyTorch 和 ORT 的输出 max abs diff < 1e-4。

**三方延迟对比（p50 / p95 / p99，500 次）**：

| 后端 | Device | p50 (ms) | p95 (ms) | p99 (ms) |
|------|--------|----------|----------|----------|
| PyTorch eager | CPU | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> |
| PyTorch eager | GPU | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> |
| torch.compile | GPU | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> |
| ONNX Runtime | CPU | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> |
| ONNX Runtime | GPU | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> | <!-- TODO: your result numbers --> |

**INT8 动态量化**：`quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)`。CPU p50 从 <!-- TODO: your result numbers --> ms 降到 <!-- TODO: your result numbers --> ms，AUC-PR 从 <!-- TODO: your result numbers --> 降到 <!-- TODO: your result numbers -->（可接受/不可接受，业务决定）。

**服务化**：FastAPI `POST /score`，Pydantic 校验 `(B, L, F)` 输入，ONNX Runtime 推理。Colab 用 `pyngrok` 暴露出 demo URL，`curl` 一条就能打通。

**Production readiness**：真正上线还缺的 8 件事，清单在 notebook §7。最需要补的前三名是：**特征一致性（训练 scaler 持久化 + 线上一致加载）**、**Schema + 范围校验**、**模型漂移监控（PSI + 预测分数直方图）**。

---

## 9. 下一步：我接下来想学什么

这 12 周基本把 Transformer 的**有监督 + 无监督 + 长序列 + 部署**都走了一遍，但还有几个方向没碰：

1. **对比学习预训练 → 微调**：SimCLR / TS2Vec 风格的无监督预训练，再接分类头做 few-shot 欺诈检测。欺诈标签少而金贵，这条线值得深挖。
2. **图 Transformer**：把用户-商户-设备-IP 建成异构图，用 Graph Transformer (GPS / NodeFormer) 端到端建模。交易欺诈本质是图上的异常模式检测。
3. **LLM-augmented fraud detection**：给 LLM 喂一条交易 + 一些规则，让它生成「为什么可疑」的推理链，辅助人工审核。核心价值不是 LLM 分类更准，而是**提供可解释的推理链让合规/审核提效**。
4. **时序基础模型 (TimeGPT / Chronos / Moirai)**：看看 zero-shot 做交易异常能到什么水平。

---

## 10. 12 周回头看：如果重来一次

**会继续做的**：
- MLP / LSTM / 手写 Transformer 全部手搓一遍——每搓一次都多懂一层。
- 前 6 周只用 Kaggle 小数据 + 合成数据。
- 每周写复盘笔记。这是最省事也最增值的动作。

**会改的**：
- W5-W6 读 Attention Is All You Need 的时候，应该同步看 [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)，不然纯论文太抽象。
- W10 在 IEEE-CIS 上应该花更多时间做特征工程，而不是急着调模型——**真实世界里特征 > 模型**。
- W12 应该早一周开始做 ONNX 导出。很多 PyTorch 模型在导出时会卡算子兼容性，留够时间能救命。

---

## 11. 资源（跟着这份计划走的话会用到）

| 论文 | 用途 |
|------|------|
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Transformer 开山；W5-W7 精读 |
| [BERT (2018)](https://arxiv.org/abs/1810.04805) | Encoder-only 范式；W8 粗读 |
| [Anomaly Transformer (ICLR 2022)](https://arxiv.org/abs/2110.02642) | 时序异常检测 SOTA 思路；W9 精读 |
| [Informer (AAAI 2021 Best)](https://arxiv.org/abs/2012.07436) | ProbSparse attention + 生成式 decoder；W11 精读 |
| [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730) | Patching + channel-independence；W11 精读 |
| [DLinear (AAAI 2023)](https://arxiv.org/abs/2205.13504) | Transformer 反思 + 线性基线；W12 选读 |

**教材**：
- [动手学深度学习 (d2l PyTorch 版)](https://zh.d2l.ai/)
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [HuggingFace Course](https://huggingface.co/learn/nlp-course)

**数据集**：
- [Credit Card Fraud (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [IEEE-CIS Fraud (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection)
- [PaySim (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1)

**工具**：
- [Time-Series-Library (thuml)](https://github.com/thuml/Time-Series-Library) — Informer / Autoformer / PatchTST / Anomaly Transformer 的官方统一实现
- [PyOD](https://github.com/yzhao062/pyod) — 经典异常检测 baseline 库
- [PatchTST 官方](https://github.com/yuqinie98/PatchTST)

---

> 全部 notebook 在 GitHub：`<!-- TODO: your repo link -->`。
>
> 如果你要按同样的路线走一遍，建议就从 `learning/week01/` 开始，一周一个 notebook，不要跳。跳过的每一步都会在三周之后找上门来。
