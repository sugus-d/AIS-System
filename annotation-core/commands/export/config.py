"""导出管线共享路径与常量。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

RESULTS_DIR = PROJECT_DIR / "results"  # 注意：此为 results 根目录，与 modeling/contracts.py 的 RESULTS_DIR（训练子目录）不同，两者勿混
TABLES_DIR = RESULTS_DIR / "eval" / "tables"
FORMULA_DIR = RESULTS_DIR / "formulas"
FEATURE_DIR = RESULTS_DIR / "extraction" / "features"
FIGURES_DIR = RESULTS_DIR / "figures"
# 导出产物目标：环境变量 AIS_EXPORT_DIR 优先（如指向外部 docs/manuscript）；
# 未设置时退化到 results/export（core 独立使用时）
_explicit_export = os.environ.get("AIS_EXPORT_DIR")
EXPORT_DIR = Path(_explicit_export) if _explicit_export else RESULTS_DIR / "export"
EXPORT_FIGURES_DIR = EXPORT_DIR / "分析图片"
EXPORT_SHAP_DIR = EXPORT_DIR / "分析图片"
FEATURE_IMPORTANCE_DIR = RESULTS_DIR / "modeling" / "feature_importance"
PREDICTION_DIR = RESULTS_DIR / "modeling" / "prediction"
PARAM_SELECTED_DIR = RESULTS_DIR / "parameterization_selected"

# 旧版输出目录（保留兼容）
WATERFALL_DIR = RESULTS_DIR / "analysis" / "feature_contributions"

DATA_DIR = PROJECT_DIR / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
CLINICAL_FILE = DATA_DIR / "form" / "clinical_data.json"

FEATURE_FILE = FEATURE_DIR / "features_2700d.csv"

# 预测源
ENSEMBLE_PRED_PATH = PREDICTION_DIR / "ensemble_composite_v7_ai60" / "Ensemble" / "predictions.csv"
PRED_CSV = ENSEMBLE_PRED_PATH
