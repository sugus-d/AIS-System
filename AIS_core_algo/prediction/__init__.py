"""AIS 预测服务包 — 核心 + 双入口（CLI 与 HTTP API 共用同一核心）。

- 核心：`prediction.predict`（特征提取 + 模型预测 + 报告生成）
- HTTP API 服务：`uvicorn prediction.api:app`
- 脚本 CLI：`python -m prediction.cli`（三模式：landmarks / predict / auto）

注：本包不预导出 `app`，避免 `import prediction` 触发
`prediction.api → prediction.predict` 链路导致 `python -m prediction.cli`
的 runpy 预加载警告。
"""
