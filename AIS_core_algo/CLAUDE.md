# AIS Project Configuration

Working language: **Chinese**.  所有交流用中文。

---

## Core Rules

- **Think first** — assumptions/pitfalls before coding. STOP if unsure.
- **Surgical changes** — 200 lines max per task, no new deps, no dead code left behind.
- **Reuse before write** — 先 `search_graph("keyword")` 查知识图谱，找不到再用 `rg` 做文本搜索。
- **Naming** — no `p/q/r/t/tmp/temp/data/res/ret`, no abbreviations in function names. Public before private in file order.
- **No temp dirs** — all code goes to formal directories. No `/tmp/` or `.tmp/`.
- **Evidence before claims** — attach command output for every assertion.
- **🔴 红线：禁止主动建议结束/停止/收工/提交** — 除非用户明确说出"提交"、"结束"、"停下"、"commit"，否则不得主动提议结束当前工作。被用户批评后继续 debug/实现，不得辩驳或解释。
- **🔴 红线：禁止在会话中用 Read 工具直接读图片** — 图片（png/jpg 等）一律通过 `claude-vision-skill` 查看，避免二进制内容撑爆上下文。违反即视为红线违规。
- **重构必删旧代码** — 重构将代码移到新位置后，旧文件必须直接删除，**不得**留 deprecation wrapper 或转发 stub。删除前必须 grep 所有引用并逐一修改，然后运行验证确认无遗漏。

## 产物目录规则（2026-08-16 定）

1. **data/ 只读**：`data/` 是输入，代码禁止向 data/ 写入。
2. **预测产物 → prediction/outputs/**：预测服务运行期产物（每次 API/CLI 预测的 roi.ply / landmarks.json / prediction.json / report/）统一在 `prediction/outputs/<subject_id>/`，刻意不归入 results/。
3. **离线产物 → results/**：训练/评估/导出/特征工程等离线产物统一在 `results/`，按 `阶段/子阶段/subject|方案|版本/文件` 分层。逐 subject 用 `<sid>/`；按方案用方案名；训练运行用 `<时间戳>_<模型名>`。
4. **路径常量单一来源**：新代码必须从 `utils/paths.py` 导入路径常量，禁止硬编码 `"results/..."` / `"prediction/outputs/..."` 字符串。
5. **废弃产物进 results/archive/**：一次性/历史产物统一归 `results/archive/`（results/ 不跟 git，删除即永久，归档优先于删除）。
6. **日志 → logs/**：运行日志在 `logs/`（`utils/logger.py` 管理，RotatingFileHandler）。
7. **可再生数据保留 7 天**：可确定性重新生成的产物（cache / archive / landmarks / parameterization / roi / 算法预标 GT 等）超过 7 天即清理；不可再生数据保留（labeling/ 人工编辑 ROI、data/ground_truth 人工标注、modeling/ 模型与训练结果、eval/tables/ 论文表）。

## Memory Architecture

```
.claude/MEMORY.md          ← index only (add entry on new memory)
.claude/memory/             ← persistent knowledge, feedback, lessons (.md only)
.claude/memory/feedback.md ← pitfalls log
```

Write to `.claude/memory/` for persistent knowledge, `.claude/goals/` for session goals (via `/goal-prompt` / `/goal`). Docs → `.claude/docs/`. Skills → `.claude/skills/<name>/`.

## Development Framework

```bash
# Pipeline CLI (三步骤：ROI → 特征工程 → 训练)
python ais-cli.py                    # 默认配置
python ais-cli.py --list-steps       # 查看可用步骤/算法
python ais-cli.py --step roi --roi-algo bfs --step train --model HistGBRT

# Training
Workflow({scriptPath: ".claude/workflows/dev.workflow.js"})     # Planner → Coder → Reviewer → Tester
Workflow({scriptPath: "~/.claude/workflows/refactor.workflow.js"}) # 全局重构 workflow（等价/非等价 + golden 闸门，流程见 /refactor-workflow skill）
```

## Project Docs

| Link | Description |
|------|-------------|
| [MEMORY.md](.claude/MEMORY.md) | Shared memory index |
| [PROJECT.md](.claude/PROJECT.md) | Parameters, algorithms, thresholds |
| [refactoring-summary.md](.claude/docs/refactoring-summary.md) | Code design decisions |
| [references.md](.claude/memory/references.md) | Quick-reference cheatsheet |
| [landmark-standards.md](.claude/rules/landmark-standards.md) | Numerical standards |
| [asymmetric-index-background.md](.claude/docs/asymmetric-index-background.md) | Background |
| [feedback.md](.claude/memory/feedback.md) | Pitfalls log |

## Tools

| Tool | Purpose |
|------|---------|
| **codebase-memory-mcp** `search_graph` / `trace_path` | **代码搜索优先用此工具**，替代 grep。在以下场景必须调用：搜索函数/类定义（`search_graph("keyword")`）、追踪调用链（`trace_path("fn", mode="calls")`）、获取函数源码（`get_code_snippet("pkg.mod.fn")`）、查看架构概览（`get_architecture()`）。需要对代码做结构性理解时**先**用 `search_graph`，找不到再用 grep。 |
| **claude-vision-skill** | 识图 — 用户分享/粘贴/引用图片（本地路径或 URL）需要描述/分析/识别时使用（`/claude-vision-skill`），运行 vision.js 把图片转成文字，避免把图片 base64 塞进会话上下文。🔴 红线：会话中禁止用 Read 工具直接读图片，一律走本 skill |
| **Playwright** | `playwright screenshot --wait 2000 <url> <out.png>` / `page.inner_text('body')` for DOM text |
| **agent-browser** | `agent_browser_open(url)` → `agent_browser_screenshot()` → `agent_browser_snapshot()` |
| **AnySearch** | `search(query="...")` / `batch_search(queries=[...])` for web search |
