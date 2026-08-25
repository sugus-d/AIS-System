"""项目共享常量 — 无依赖（utils 是最底层，所有包均可依赖）。

cobb 严重度分级与校准钳制范围在此单源定义，modeling/prediction/features 三方共用，
避免同值散落多处（曾有三份 SEVERITY_BINS 各自维护）。
"""

# cobb 角分级阈值（Normal <10 / Mild <20 / Moderate <40 / Severe ≥40）
SEVERITY_BINS = [0.0, 10.0, 20.0, 40.0, float("inf")]
SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
# 每类预测值的钳制范围（校准应用偏差后）
CLASS_RANGES = {0: (0.0, 10.0), 1: (10.0, 20.0), 2: (20.0, 40.0), 3: (40.0, 200.0)}


def classify_cobb(cobb: float) -> str:
    """按临床阈值分级 cobb 角（Normal/Mild/Moderate/Severe）。

    全库唯一实现：prediction（预测分级）、commands/export（论文表）、
    modeling（训练评估）共用，避免 _COBB_* 阈值与分级逻辑散落多处。
    """
    for i in range(len(SEVERITY_LABELS)):
        if SEVERITY_BINS[i] <= cobb < SEVERITY_BINS[i + 1]:
            return SEVERITY_LABELS[i]
    return "Severe"
