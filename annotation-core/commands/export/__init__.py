from commands.export.analyze import main as analyze_main  # noqa: F401  # 导入触发 warnings 过滤副作用
from commands.export.charts_feature_importance import (
    main as charts_fi_main,  # noqa: F401  # 导入触发 matplotlib rcParams 初始化
)
from commands.export.charts_waterfall import main as charts_wf_main  # noqa: F401  # 导入触发 matplotlib rcParams 初始化
