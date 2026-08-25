"""一次性迁移：results/ground-truth/*/ground_truth.json 嵌套格式 → 扁平 18 键。

嵌套格式（标注平台旧产物）：
    {"neck_root": {"L": [x,y,z], "R": [x,y,z]}, ..., "spine_points": [[..]×6], "_features": {...}}
扁平 18 键（全链路统一契约，FLAT_KEYS 单源）：
    {"neck_root_L": [..], "neck_root_R": [..], ..., "thoracic_spine_point": [..], "_features": {...}}

幂等：已含 ``neck_root_L`` 键的视为已扁平，跳过。``--check`` 只校验不写。
安全约定：只读/写 results/ground-truth/，不触碰训练缓存与数据副本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from landmarks.constants import BILATERAL_KEYS, FLAT_SPINE_KEYS
from utils.logger import logger

# GT 目录（本地数据，与标注平台共享 AIS_RESULTS_ROOT）
_GT_ROOT = Path("results/ground-truth")

# 扁平格式判据：含任意 bilateral _L 键（neck_root_L 必存在）
_FLAT_MARKER = "neck_root_L"


def _to_flat(gt: dict) -> dict:
    """嵌套 dict → 扁平 18 键 dict（保留 _features 元数据）。"""
    flat: dict = {}
    for key, value in gt.items():
        if key.startswith("_"):
            flat[key] = value
            continue
        if key in BILATERAL_KEYS and isinstance(value, dict):
            for side in ("L", "R"):
                if value.get(side) is not None:
                    flat[f"{key}_{side}"] = list(value[side])
        elif key == "spine_points" and isinstance(value, list):
            for idx, semantic_key in enumerate(FLAT_SPINE_KEYS):
                if idx < len(value) and value[idx] is not None:
                    flat[semantic_key] = list(value[idx])
        # 未知顶层键（非 bilateral/spine/_）直接保留，避免丢数据
        elif not key.startswith("spine_P"):
            flat[key] = value
    return flat


def migrate_one(gt_file: Path) -> bool:
    """迁移单个 GT 文件嵌套→扁平；返回是否发生写入。幂等（已扁平跳过）。"""
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    if _FLAT_MARKER in gt:
        return False
    flat = _to_flat(gt)
    gt_file.write_text(json.dumps(flat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> None:
    """遍历 GT 目录迁移全部 subject（--check 只校验不写）。"""
    parser = argparse.ArgumentParser(description="ground_truth.json 嵌套→扁平 18 键迁移")
    parser.add_argument("--check", action="store_true", help="只检查是否全部已扁平，不写文件")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个 subject（冒烟用）")
    args = parser.parse_args()

    files = sorted(_GT_ROOT.glob("*/ground_truth.json"))
    if args.limit:
        files = files[: args.limit]
    logger.info(f"扫描到 {len(files)} 个 GT 文件（--check={args.check}）")

    migrated = unchanged = 0
    for gt_file in files:
        if args.check:
            gt = json.loads(gt_file.read_text(encoding="utf-8"))
            if _FLAT_MARKER not in gt:
                logger.error(f"{gt_file.parent.name}: 仍是嵌套格式")
            else:
                unchanged += 1
            continue
        if migrate_one(gt_file):
            migrated += 1
            logger.info(f"{gt_file.parent.name}: 已迁移为扁平")
        else:
            unchanged += 1

    if args.check:
        logger.info(f"校验完成：{unchanged}/{len(files)} 已扁平" + ("" if unchanged == len(files) else " ⚠ 有嵌套残留"))
    else:
        logger.info(f"迁移完成：{migrated} 个转换，{unchanged} 个已扁平，共 {len(files)}")


if __name__ == "__main__":
    main()
