# Week 12 — 落地与收尾：ONNX + FastAPI + 12 周总结

> 目标：本周结束时，把 W10 或 W11 的最佳 checkpoint 导出成 ONNX，用 ONNX Runtime 跑出推理延迟对比；能在 Colab 起一个 FastAPI `/score` 端点并通过 `pyngrok` 对外暴露；提交一篇 ~3000 字的 12 周总结博客初稿。

## 本周 10 小时任务清单

| 日 | 时段 | 2h 任务 |
|----|------|--------|
| 周一 早 09:00–11:00 | 理论 | 读 `torch.compile` 入门、ONNX 导出最佳实践、INT8 动态量化原理 |
| 周二 晚 20:30–22:30 | 编码 | `12_deployment.ipynb` §1-3：加载 checkpoint、`torch.compile` 基准、ONNX 导出 + 数值验证 |
| 周三 早 09:00–11:00 | 编码 | `12_deployment.ipynb` §4-5：三方延迟表（eager / compile / ORT）+ 动态 INT8 量化 |
| 周四 晚 20:30–22:30 | 编码 | `12_deployment.ipynb` §6：FastAPI + `pyngrok` + `curl` 演示 |
| 周五 早 09:00–11:00 | 复盘 | `12_summary_blog.md`：填充各周 v0.1→v0.5 的实际数字，写下一步 |

## 起步资源

**推理优化**：
- [torch.compile 入门](https://pytorch.org/docs/stable/torch.compiler.html)
- [ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [PyTorch Dynamic Quantization](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html)

**服务化**：
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [pyngrok + Colab](https://pyngrok.readthedocs.io/en/latest/integrations.html#google-colaboratory)

## 文件结构

```
week12/
├── README.md                ← 本文件
├── 12_deployment.ipynb      ← 周二-周四：推理优化 + 服务化
├── 12_summary_blog.md       ← 周五：3000 字总结博客模板
└── requirements.txt         ← 依赖 pin 版本（加上 onnx / onnxruntime / fastapi / pyngrok）
```

> Notebook 在 PROJECT_ROOT 下会生成一个独立 `server.py`（FastAPI app）。

## 本周验收

- [ ] `torch.compile` 100 次 forward 基准跑通，打印 speedup
- [ ] ONNX 导出成功，ONNX Runtime 与 PyTorch 数值差 `max|Δ| < 1e-4`（32 个随机 batch）
- [ ] 延迟基准表填完（eager / compile / ORT × CPU / GPU，p50/p95/p99 over 500 runs）
- [ ] INT8 动态量化 demo 跑通，记录「延迟 ↓ 多少 / AUC-PR ↓ 多少」
- [ ] FastAPI `/score` 能通过 `pyngrok` 对外访问，`curl` 示例返回 JSON
- [ ] 产线 readiness 清单逐条过一遍
- [ ] Mermaid 图渲染出 12 周全景
- [ ] `12_summary_blog.md` 草稿提交（`<!-- TODO: your result numbers -->` 占位全部填上）

## 周五复盘三问

1. 同一个模型在 PyTorch eager / compile / ONNX Runtime 下的 p95 差几倍？瓶颈在哪一层？
2. 动态 INT8 量化在 CPU 上加速多少？AUC-PR 下降是否在业务可接受范围内？
3. 如果明天要上 prod，我还需要搞定哪 3 件事？（对着 readiness 清单挑）
