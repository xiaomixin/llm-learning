# 09 — Early Stopping + Best-Model Checkpoint

> **一句话口诀**:监控 val 指标,**有进步就深拷贝权重**,**耐心用完就停**。两件事必须配合:过早停拿的权重对不上,记对了也停不了。

---

## 1. 先跑起来(`../snippets/09_early_stop.py`)

```bash
python snippets/09_early_stop.py
```

脚本故意造了"小 train + 大 val"的过拟合场景,展示:
- 正确版(用 `copy.deepcopy`):回滚后 AP 和记录的 best AP 一致。
- 错误版(不 deepcopy):回滚后可能拿到最新的(更差的)权重。

---

## 2. 发生了什么(白话)

**过拟合的典型曲线**:
- epoch 1–10: train loss ↓, val AP ↑(模型在学通用规律)
- epoch 10–20: train loss ↓↓, val AP 开始震荡持平
- epoch 20+: train loss 继续 ↓,val AP 开始 ↓(开始记噪声)

**Early stopping 的策略**:
1. 每个 epoch 算一次 val 指标。
2. 如果比历史最好还好,**拷贝一份权重备份**,耐心值归零。
3. 如果比历史最好差,耐心值 +1。
4. 耐心值达到 `patience`(通常 5),**停止训练,把权重恢复到备份**。

结果:你拿到的是**整个训练过程中 val 指标最好的那一轮**,而不是最后一轮。

---

## 3. 多角度理解

### 视角 A:early stop vs best checkpoint 不是一回事

| 机制 | 干什么 | 怎么没它 |
|-----|-------|---------|
| Early stopping | 提前终止训练 | 跑满 max_epochs,浪费时间 |
| Best checkpoint | 回滚到最佳权重 | 拿到最后一个(过拟合)epoch 的权重 |

通常**一起用**:early stop 节省算力,best checkpoint 保证拿到好权重。cell 16 就是两者合体。

### 视角 B:监控哪个指标

**差的做法**:监控 val loss。
- `pos_weight` 改变了 loss 的 scale,loss 下降 ≠ 精度提升。
- Focal Loss 更是 loss 的绝对值失去可比性。

**好的做法**:监控**和业务目标对齐的指标**——Week 2 用 val AUC-PR。它直接反映"在不平衡下的排序质量",和评估脚本对齐。

### 视角 C:patience 的经验取值

- `patience=1-2`:噪声太敏感,容易误停。
- `patience=5`:Week 2 的经验起点,对大多数任务够用。
- `patience=10+`:val 指标本身噪声大(小 val 集)时才需要。

调 patience 的方法:先跑满 max_epochs,看 history 里 val 指标的**波动幅度**和**真正拐点位置**。如果波动 ±0.01、拐点很清晰,patience=3 就够;波动 ±0.05,patience=8。

---

## 4. `copy.deepcopy` 的坑(最常见的 early stop bug)

```python
best_state = model.state_dict()            # ✗ 浅拷贝
best_state = copy.deepcopy(model.state_dict())   # ✓ 深拷贝
```

**`state_dict()` 返回一个 OrderedDict**——字典本身是新的,但**里面的 tensor 是原模型参数的引用**。下一个 epoch `optimizer.step()` 一更新,`best_state` 里的 tensor 也跟着变。

**症状**:训练显示 val_ap 在 epoch 5 达到 0.75 后下降,epoch 10 early stop 退出。`model.load_state_dict(best_state)` 之后评估发现 AP 是 0.68——因为你加载的其实是 epoch 10 的权重。

**深拷贝**:`copy.deepcopy` 递归复制 dict 和里面每个 tensor 的 `.data`,和原模型彻底分家。

**备选**:直接保存到磁盘 `torch.save(model.state_dict(), tmp_path)`,later `model.load_state_dict(torch.load(tmp_path))`——更重,但绝对隔离。

---

## 5. 一个完整的 early stop 模板

```python
import copy

best_ap, best_state, bad = -1.0, None, 0
patience = 5

for epoch in range(1, max_epochs + 1):
    train_one_epoch(model, train_loader, optimizer, loss_fn)
    val = evaluate(model, val_loader)

    if val['ap'] > best_ap:
        best_ap = val['ap']
        best_state = copy.deepcopy(model.state_dict())
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print(f'early stop @ epoch {epoch}')
            break

model.load_state_dict(best_state)   # 回滚到最佳
```

记住:**深拷贝、监控业务指标、耐心合理、最后 load 回去**。四件事齐活,早停就不会有坑。

---

## 6. 与 notebook 的连接

- cell 16 的训练函数 `train_model` 完整实现了本节逻辑。
- 如果你看到 `best val_ap = 0.75` 但 test 评估时指标明显低,八成是 deepcopy 忘了——去 cell 16 检查。

**下一站**:[10_evaluation.md](10_evaluation.md) —— 怎么把训出来的模型诚实地评一遍?AUC-PR、AUC-ROC、Recall@FPR 三个指标有何区别?
