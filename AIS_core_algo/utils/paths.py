"""全部产物/数据路径的单一来源。

规则（2026-08-16 定）：
- 所有产物必须落在 ``RESULTS_DIR``（results/）下，禁止写入 data/。
- 所有路径基于 ``_PROJECT_ROOT``（core/）绝对解析，不依赖 cwd。
- 新代码一律从这里导入路径常量，禁止硬编码 ``"results/..."`` 字符串。
- 层级约定：``results/<阶段>/<子阶段>/<subject|方案|版本>/<文件>``。
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # core/

# === 产物根（单一收口） ===
RESULTS_DIR = Path(os.environ.get("AIS_RESULTS_ROOT", _PROJECT_ROOT / "results"))

# === ROI / 地标 ===
ROI_DIR = RESULTS_DIR / "roi"
LANDMARKS_DIR = RESULTS_DIR / "landmarks"
GROUND_TRUTH_OUTPUT_DIR = RESULTS_DIR / "ground-truth"  # 算法生成 GT（original.ply + ground_truth.json）

# === 参数化 ===
PARAM_DIR = RESULTS_DIR / "parameterization"
PARAM_SELECTED_DIR = RESULTS_DIR / "parameterization_selected"

# === 特征工程 ===
EXTRACTION_DIR = RESULTS_DIR / "extraction"
CURVATURE_DIR = EXTRACTION_DIR / "curvature"
FEATURES_DIR = EXTRACTION_DIR / "features"
FEATURES_EXTRACTION_DIR = EXTRACTION_DIR / "features_extraction"
FEATURES_SELECTION_DIR = EXTRACTION_DIR / "features_selection"

# === 建模 ===
MODELING_DIR = RESULTS_DIR / "modeling"
MODELS_DIR = MODELING_DIR / "models"
MODELING_PREDICTION_DIR = MODELING_DIR / "prediction"  # 训练运行结果
FEATURE_IMPORTANCE_DIR = MODELING_DIR / "feature_importance"
COMPOSITE_DIR = MODELING_DIR / "composite"

# === 预测推理产物（PREDICT_ROOT，逐 subject 目录；刻意不归入 results/） ===
PREDICTION_OUTPUTS_DIR = Path(os.environ.get("AIS_RESULTS_ROOT", _PROJECT_ROOT / "prediction" / "outputs")) / "prediction-outputs"

# === 评估 ===
EVAL_DIR = RESULTS_DIR / "eval"
EVAL_TABLES_DIR = EVAL_DIR / "tables"
EVAL_CUT_DIR = EVAL_DIR / "cut_eval"
EVAL_EVALUATION_DIR = EVAL_DIR / "evaluation"

# === 导出 / 论文 ===
FORMULA_DIR = RESULTS_DIR / "formulas"
FIGURES_DIR = RESULTS_DIR / "figures"
# 导出产物目标：环境变量 AIS_EXPORT_DIR 优先（如指向外部 docs/manuscript）；
# 未设置时退化到 results/export（core 独立使用时）
_explicit_export = os.environ.get("AIS_EXPORT_DIR")
EXPORT_DIR = Path(_explicit_export) if _explicit_export else RESULTS_DIR / "export"
EXPORT_FIGURES_DIR = EXPORT_DIR / "分析图片"
EXPORT_SHAP_DIR = EXPORT_DIR / "分析图片"

# === 日志 / 缓存 / 归档 ===
LOGS_DIR = _PROJECT_ROOT / "logs"
CACHE_DIR = RESULTS_DIR / "cache"
ARCHIVE_DIR = RESULTS_DIR / "archive"

# === 输入数据（只读，禁止写入） ===
DATA_DIR = Path(os.environ.get("AIS_DATA_ROOT", _PROJECT_ROOT / "data"))
MESH_DIR = DATA_DIR / "mesh"
GROUND_TRUTH_INPUT_DIR = DATA_DIR / "ground_truth"  # 人工标注输入
CLINICAL_DATA = DATA_DIR / "form" / "clinical_data.json"

# === 导出管线派生常量（predictions 源） ===
FEATURE_FILE = FEATURES_DIR / "features_2700d.csv"
ENSEMBLE_PRED_PATH = (
    MODELING_PREDICTION_DIR / "ensemble_composite_v7_ai60" / "Ensemble" / "predictions.csv"
)
PRED_CSV = ENSEMBLE_PRED_PATH
MANUAL_PRED_PATH = (
    MODELING_PREDICTION_DIR
    / "v1.0.0"
    / "ensemble-ai_refit_ridge_boundary"
    / "Ensemble"
    / "predictions.csv"
)
MANUAL_REGION_CSV = FEATURES_EXTRACTION_DIR / "v1.0.0" / "region_asymmetry.csv"
MANUAL_TABLES_DIR = EXPORT_DIR / "v1.0.0" / "tables"
MANUAL_FIGURES_DIR = EXPORT_DIR / "v1.0.0" / "figures"
WATERFALL_DIR = RESULTS_DIR / "analysis" / "feature_contributions"
