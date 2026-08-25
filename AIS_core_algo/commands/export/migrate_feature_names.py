"""一次性迁移：spine_P* 特征列名 → 段语义名（CSV + 模型包 feature_names）。

morphology 特征列名从 P 索引命名（spine_P0_P1_length）改为段语义缩写
（spine_neck_scapular_length）。HistGBRT 按 feature_names 顺序喂数组，
列名仅标签——模型不必重训，只要训练 CSV 列名 + 模型包 feature_names 同步改名。

覆盖：
1. results/extraction/features_extraction/v0.1.0/morphology.csv（训练特征 CSV）
2. prediction/models/*.joblib（模型包 feature_names）

幂等：无 spine_P 列则跳过。回滚：git checkout 代码 + 重跑（CSV 已迁移则手动改回）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from landmarks.constants import SPINE_SEG_SEMANTIC
from utils.logger import logger

# 训练特征 CSV（只读列结构 + 改名；数据值不变）
_TRAIN_MORPH_CSV = Path("results/extraction/features_extraction/v0.1.0/morphology.csv")
# 模型包目录
_MODELS_DIR = Path("prediction/models")


def _rename_feature(name: str) -> str:
    """spine_P0_P1_length → spine_neck_scapular_length；非 spine_P 名原样返回。"""
    if not name.startswith("spine_") or "_P" not in name:
        return name
    result = name
    for p_token, semantic in SPINE_SEG_SEMANTIC.items():
        if p_token in result:
            result = result.replace(p_token, semantic)
            break
    return result


def migrate_csv(csv_path: Path) -> int:
    """迁移单个特征 CSV 的 spine_P 列名；返回改名列数。幂等。"""
    df = pd.read_csv(csv_path)
    renamed = 0
    mapping = {}
    for col in df.columns:
        new_col = _rename_feature(col)
        if new_col != col:
            mapping[col] = new_col
            renamed += 1
    if renamed:
        df = df.rename(columns=mapping)
        df.to_csv(csv_path, index=False)
        logger.info(f"{csv_path.name}: {renamed} 列改名 → {sorted(mapping.values())}")
    else:
        logger.info(f"{csv_path.name}: 无 spine_P 列，跳过")
    return renamed


def migrate_model_package(model_path: Path) -> int:
    """迁移模型包 feature_names；返回改名数。幂等。"""
    if not model_path.exists():
        logger.warning(f"模型包不存在: {model_path}，跳过")
        return 0
    pkg = joblib.load(model_path)
    feature_names = pkg.get("feature_names")
    if not isinstance(feature_names, list):
        return 0
    new_names = [_rename_feature(n) for n in feature_names]
    renamed = sum(1 for old, new in zip(feature_names, new_names, strict=True) if old != new)
    if renamed:
        pkg["feature_names"] = new_names
        joblib.dump(pkg, model_path)
        logger.info(f"{model_path.name}: feature_names {renamed} 个改名")
    else:
        logger.info(f"{model_path.name}: feature_names 无 spine_P，跳过")
    return renamed


def main() -> None:
    """迁移训练 CSV + 模型包特征列名。"""
    parser = argparse.ArgumentParser(description="spine_P 特征列名 → 段语义名迁移")
    parser.add_argument("--check", action="store_true", help="只检查不写")
    args = parser.parse_args()

    total = 0
    if _TRAIN_MORPH_CSV.exists():
        if args.check:
            df = pd.read_csv(_TRAIN_MORPH_CSV)
            sp = [c for c in df.columns if _rename_feature(c) != c]
            logger.info(f"CSV 待改名列: {len(sp)}")
            total += len(sp)
        else:
            total += migrate_csv(_TRAIN_MORPH_CSV)
    else:
        logger.warning(f"训练 CSV 不存在: {_TRAIN_MORPH_CSV}，跳过")

    for model_path in sorted(_MODELS_DIR.glob("*.joblib")):
        if args.check:
            pkg = joblib.load(model_path)
            sp = [n for n in pkg.get("feature_names", []) if _rename_feature(n) != n]
            logger.info(f"{model_path.name}: 待改名 feature_names {len(sp)} 个")
            total += len(sp)
        else:
            total += migrate_model_package(model_path)

    logger.info(f"迁移完成：共 {total} 个列名" + ("（--check 只读）" if args.check else ""))


if __name__ == "__main__":
    main()
