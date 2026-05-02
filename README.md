# llm-learning

个人 LLM / Transformer 学习仓库。当前主项目：**交易序列 → Transformer → 异常检测**（12 周计划，每周 10 小时）。

> **数据声明**：本项目使用 **公开 Kaggle 数据集**（Credit Card Fraud Detection、IEEE-CIS Fraud）与 **自生成合成交易数据**。仓库中不包含任何真实业务数据。

## 快速开始

```bash
# 1. Clone
git clone git@github.com:xiaomixin/llm-learning.git
cd llm-learning

# 2. 启用 pre-commit 安全扫描（防止误提交密钥）
git config core.hooksPath .githooks

# 3. 看完整 12 周计划
open transformer-12week-plan.md
```

## 目录

```
.
├── transformer-12week-plan.md   # 主计划（12 周 × 10h）
├── week01/                      # 每周 README + notebooks
│   ├── README.md
│   ├── env_check.ipynb          # Colab 环境自检
│   ├── 01_eda.ipynb             # Kaggle Credit Card Fraud EDA
│   └── requirements.txt
├── scripts/
│   ├── learning_schedule.json   # 12 周 × 5 天的完整 schedule
│   ├── security_check.sh        # 提交前扫密钥/凭证
│   ├── env_self_check.py        # Colab 环境自检
│   ├── download_kaggle.py       # Kaggle 数据一键下载
│   └── sync_to_drive.sh         # 同步 notebook 到 Google Drive
└── .githooks/pre-commit         # 调用 security_check.sh
```

## Colab 工作流

VS Code 本地编辑 → Google Drive 同步 → Colab 运行：

1. 在 VS Code 本地编辑 + `git commit`
2. 运行 `scripts/sync_to_drive.sh` 把 notebooks 拷贝到 Google Drive（本地 Drive 客户端会自动上云）
3. Colab 里 `drive.mount()` 然后从 `MyDrive/<你的文件夹>/` 打开 notebook 运行
4. Colab 端修改的 notebook 保存回 Drive，本地 Drive 客户端再同步回本地
5. 本地 `git add + commit` 保留历史

配置 Drive 路径：在 `scripts/` 目录创建 `sync.config.local`（已 gitignore），填入：

```bash
DRIVE_DIR="$HOME/Library/CloudStorage/GoogleDrive-<your-email>/My Drive/<your-folder>"
```

## 安全检查

每次 `git commit` 自动触发 `scripts/security_check.sh`，扫描常见密钥模式：

- GitHub / OpenAI / Anthropic / Slack / AWS / Google API keys
- JWT、Bearer token、PEM 私钥
- Kaggle 凭证、硬编码密码

命中则阻止提交。若为误报，`git commit --no-verify` 临时绕过（请务必确认后再用）。

## 主要教材

- 《动手学深度学习》PyTorch 版（d2l.ai）第 9/10/11 章
- Andrej Karpathy《Neural Networks: Zero to Hero》YouTube 系列（nanoGPT / makemore）
- HuggingFace NLP Course
- 核心论文：Attention Is All You Need / Informer / PatchTST / Anomaly Transformer
