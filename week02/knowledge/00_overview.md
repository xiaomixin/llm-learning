# Week 2 知识树 — 阅读顺序

> 旧的 `knowledge_legacy.md` 太厚，一口吞下不消化。这份重写把 Week 2 拆成 **10 个独立概念**,每个概念一个 `.md`,配一个 **自包含可运行的 Python 脚本**。
> 每个概念读完大约 5–8 分钟,跑完对应脚本 1 分钟。

---

## 学习路径(自下而上,每步只多学一件事)

| # | 概念 | 一句话 | 文档 | 脚本 |
|---|------|-------|------|------|
| 01 | 张量与 autograd | tensor 是"会记得自己从哪儿来的 numpy 数组",`.backward()` 自动填好 `.grad` | [01_tensor_and_autograd.md](01_tensor_and_autograd.md) | `../snippets/01_tensor.py` |
| 02 | 线性层 | `nn.Linear` 就是 `y = Wx + b`,一次矩阵乘 | [02_linear_layer.md](02_linear_layer.md) | `../snippets/02_linear.py` |
| 03 | MLP 组装 | 线性叠线性还是线性,中间必须塞 ReLU | [03_mlp_build.md](03_mlp_build.md) | `../snippets/03_mlp.py` |
| 04 | BCE 损失 + 数值稳定 | `BCEWithLogitsLoss` 用 log-sum-exp,避免 `log(0)` 和 `log(1)` | [04_bce_loss.md](04_bce_loss.md) | `../snippets/04_bce.py` |
| 05 | 训练循环 | 5 步循环:清零 → 前向 → 算 loss → 反向 → 更新 | [05_training_loop.md](05_training_loop.md) | `../snippets/05_train_loop.py` |
| 06 | `pos_weight` | 类别不平衡时,把正样本梯度放大 `neg/pos` 倍 | [06_pos_weight.md](06_pos_weight.md) | `../snippets/06_pos_weight.py` |
| 07 | Focal Loss | 进一步压低"已经分对了的容易样本"的权重 | [07_focal_loss.md](07_focal_loss.md) | `../snippets/07_focal.py` |
| 08 | 数据切分 + Scaler | stratify 保比例;scaler 只 fit 在 train 防泄露 | [08_data_split_scaler.md](08_data_split_scaler.md) | `../snippets/08_split_scaler.py` |
| 09 | Early stopping | 监控 val 指标,记录最佳权重,耐心耗尽就停 | [09_early_stopping.md](09_early_stopping.md) | `../snippets/09_early_stop.py` |
| 10 | 评估指标 | 极不平衡下 AUC-PR 比 AUC-ROC 更诚实 | [10_evaluation.md](10_evaluation.md) | `../snippets/10_evaluation.py` |

---

## 怎么用这份材料

**三遍法**:

1. **第一遍快读**:只看每个 md 顶部的"一句话 + Python 演示"。跑脚本,看输出。
2. **第二遍慢读**:看"公式怎么来的"和"多个视角"部分。不懂的停下来,回到脚本改参数试。
3. **第三遍对照 notebook**:打开 `02_mlp_baseline.ipynb`,每个 cell 找到对应的 md 概念,连起来看。

**每个 md 的固定结构**:

```
1. 一句话口诀
2. Python 演示(先跑起来)
3. 发生了什么 —— 白话解释
4. 多角度理解(至少 2 个视角)
5. 公式慢推(每个符号都点名)
6. 一个坑
7. 和 notebook 哪个 cell 对应
```

---

## 脚本怎么跑

所有脚本都在 `../snippets/` 下,每个都能独立 `python xx.py` 跑起来。

**依赖**(和 `requirements.txt` 一致):
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

**需要 creditcard.csv 的脚本**(08–10):把数据放在以下任一位置,或用 `CREDITCARD_CSV` 环境变量指定:
- `~/Code/binance/risk/openspec-projects/learning/week02/data/creditcard.csv`
- `./data/creditcard.csv`(相对当前目录)
- `$PROJECT_ROOT/data/creditcard.csv`

脚本里有个 `_data.py`(`../snippets/_data.py`)专门处理这个查找。

**概念型脚本**(01–07):用合成数据或微型真实片段,不依赖 creditcard.csv,`python` 直接跑。

---

## 和 Week 2 总目标的连接

- 跑通 notebook 达成 val AUC-PR > 0.70 是**能力验收**
- 读完这 10 个 md 是**理解验收**——能对着 notebook 任何一行说"这行做了什么、为什么"
- 两者都达成 = Week 2 真的过了

---

## 什么时候回看

| 后续场景 | 回看哪个概念 |
|---------|------------|
| Week 3 写 LSTM 训练循环 | 05(循环原样复用) |
| Week 3 按时间切分 | 08(原理同 stratify-only-on-train) |
| Week 4 序列模型加 pos_weight | 06(per-token 版本) |
| Week 7 Transformer warmup | 05(warmup 是往 5 步里插逻辑) |
| Week 9 无监督 MSE 重构 | 04(换损失函数,其他不变) |
| Week 10 多分类 Focal | 07(多类版 Focal) |

---

## 需要深究再读

[knowledge_legacy.md](../knowledge_legacy.md) —— 旧版高密度手册,把 10 个概念混编在一起。当你把 10 个概念都消化后,再读它作为"总复习"是合适的。
