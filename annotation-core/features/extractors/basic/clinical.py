"""基础临床特征提取。"""

import numpy as np


def extract_basic(clinical_data: dict) -> dict[str, dict[str, float | str]]:
    """从临床数据字典中提取基础特征。

    对每个受试者提取：身高(cm)、体重(kg)、BMI、性别、身高×体重。
    缺失值用 np.nan 填充。

    Args:
        clinical_data: 以 subject_id 为键的临床数据字典，
                       每个条目包含 height_cm, weight_kg, gender 等字段。

    Returns:
        以 subject_id 为键的字典，每个值包含：
            Height: float
            Weight: float
            BMI: float
            Gender: str
            Height_x_Weight: float
    """
    result: dict[str, dict[str, float | str]] = {}
    for subject_id, data in clinical_data.items():
        height = data.get("height_cm")
        weight = data.get("weight_kg")
        gender = data.get("gender")

        if height is None or (isinstance(height, float) and np.isnan(height)):
            height = np.nan
        if weight is None or (isinstance(weight, float) and np.isnan(weight)):
            weight = np.nan
        if gender is None:
            gender = np.nan

        if np.isfinite(height) and np.isfinite(weight) and height > 0:
            bmi = weight / (height / 100.0) ** 2
            height_x_weight = height * weight
        else:
            bmi = np.nan
            height_x_weight = np.nan

        result[subject_id] = {
            "Height": height,
            "Weight": weight,
            "BMI": bmi,
            "Gender": gender,
            "Height_x_Weight": height_x_weight,
        }

    return result
