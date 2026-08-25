"""modeling 包 — 机器学习模型训练与评估的统一入口。

用法:
    from modeling import get_model, list_models
    from modeling.metrics import compute_metrics
"""

from modeling.metrics import compute_metrics  # noqa: F401
from modeling.models import get_model, list_models  # noqa: F401
