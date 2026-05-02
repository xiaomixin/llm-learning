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
├── templates/
│   └── notebook_template.ipynb  # 新 notebook 从这里复制（含 bootstrap cell）
├── scripts/
│   ├── learning_schedule.json   # 12 周 × 5 天的完整 schedule
│   ├── security_check.sh        # 提交前扫密钥/凭证
│   ├── env_self_check.py        # Colab 环境自检
│   ├── download_kaggle.py       # Kaggle 数据一键下载
│   └── sync_to_drive.sh         # 备用：同步 notebook 到 Google Drive
└── .githooks/pre-commit         # 调用 security_check.sh
```

## Colab 工作流（双向同步）

VS Code 本地 ↔ GitHub ↔ Colab，走 Colab 原生 GitHub 集成，免 Drive、免 PAT。

**本地 → Colab**：
```bash
git add . && git commit -m "..." && git push
```
Colab 菜单 `File → Open notebook → GitHub → xiaomixin/llm-learning`，或直接：
```
https://colab.research.google.com/github/xiaomixin/llm-learning/blob/main/week01/env_check.ipynb
```

**Colab → 本地**：Colab 菜单 `File → Save a copy in GitHub`（首次走 OAuth 授权），然后本地 `git pull`。

### 新建 notebook

从 `templates/notebook_template.ipynb` 复制。第一个 cell 是 bootstrap —— 自动检测 Colab/local、挂 Drive、读 Kaggle Secrets、设种子。

### Colab Secrets（一次性配置）

左侧 🔑 图标添加：
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

**永远不要** 把 `kaggle.json` 上传到 Colab 文件系统或提交到仓库。

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
