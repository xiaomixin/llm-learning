# Week 1 — 环境准备与问题定义

> 目标：本周结束时，能在 Colab 加载 Kaggle 信用卡欺诈数据集，画出类别分布与前 5 个特征的分布图。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | d2l Ch 1-2（序言 + 预备知识）+ Ch 3.1-3.4（线性回归） |
| 周二 晚 20:30–22:30 | 编码 | Colab 环境搭建 + Drive 挂载 + `env_check.ipynb` 跑通 |
| 周三 早 09:00–11:00 | 理论+论文 | d2l Ch 3.5-3.7 + 读 Kaggle 公开 notebook top 1 |
| 周四 晚 20:30–22:30 | 编码 | 下载 Kaggle 数据 + `01_eda.ipynb` 完成 EDA |
| 周五 早 09:00–11:00 | 编码+复盘 | 整理 notebook + 写本周笔记 |

## 起步资源

**主教材**：[动手学深度学习 PyTorch 版](https://zh.d2l.ai/)

**数据集**：
- [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 参考 [Top 1 notebook (Janio Bachmann)](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)

**工具注册**：
- [ ] Kaggle 账号 + 生成 API token（头像 → Settings → Create New Token，下载 `kaggle.json`）
- [ ] Google Drive 新建 `transformer-anomaly/` 文件夹
- [ ] Colab Pro 可选（免费 T4 够用）
- [ ] WandB 账号（可选，用于实验跟踪）

## 文件结构

```
week01/
├── README.md               ← 本文件
├── env_check.ipynb         ← 周二：环境自检
├── 01_eda.ipynb            ← 周四：EDA 主 notebook
├── requirements.txt        ← 依赖 pin 版本
└── ../scripts/
    ├── download_kaggle.py  ← Kaggle 数据一键下载
    └── env_self_check.py   ← Python 端环境自检
```

## 本周验收

- [ ] Colab 能正常 mount Drive + 识别 GPU
- [ ] `env_check.ipynb` 全部 cells 通过
- [ ] 数据加载后 `df.shape == (284807, 31)`
- [ ] 画出类别分布图（欺诈占 0.173%）
- [ ] 至少画出 V1-V5 的分布（正常 vs 欺诈对比）
- [ ] 笔记回答：欺诈检测为什么是不平衡问题？AUC-PR vs AUC-ROC 差别？

## 周五复盘三问

1. 本周学到的最核心概念用一句话说清楚？
2. 今天打开 `01_eda.ipynb` 能一键 Run All 吗？
3. 下周要进 MLP baseline，我还有什么障碍？
