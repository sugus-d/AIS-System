"""基本临床特征提取。"""


def extract_basic(clinical_data: dict) -> dict:
    """从临床数据字典中提取基本人口学信息。"""
    return {
        "age": clinical_data.get("age"),
        "sex": clinical_data.get("sex", ""),
        "bmi": clinical_data.get("bmi"),
    }
