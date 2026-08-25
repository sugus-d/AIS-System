"""组装 results/export/ 目录并打包 ZIP。

将各模块的输出统一复制到 results/export/ 下按类型分类的子目录，
然后创建 AIS_数据导出_YYYYMMDD.zip。

注意: 图片由 commands/export/figures.py 和 commands/plot_waterfall.py 直接输出到 export/，
     本脚本只负责数据文件、Excel、标准化结果和 ZIP 打包。

用法:
    uv run python -m commands.export.pack
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

from commands.export.config import (
    ENSEMBLE_PRED_PATH,
    EXPORT_DIR,
    FEATURE_IMPORTANCE_DIR,
    PARAM_SELECTED_DIR,
    RESULTS_DIR,
    TABLES_DIR,
)
from utils.logger import logger


def _copy(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _cp_dir(src: Path, dst: Path) -> None:
    """Copy entire directory tree."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 原始数据 — CSV from results/eval/tables/
    logger.info("\n[1] 原始数据...")
    raw_dir = EXPORT_DIR / "原始数据"
    raw_dir.mkdir(exist_ok=True)
    raw_files = [
        ("table1_raw.csv", "表1原始数据_人口学.csv"),
        ("table2_raw.csv", "表2原始数据_体征参数.csv"),
        ("table3_raw.csv", "表3-4原始数据_复合指标.csv"),
        ("table5_raw.csv", "表5原始数据_预测结果.csv"),
        ("table6_raw.csv", "表6原始数据_分类结果.csv"),
        ("data_inventory.csv", "数据清单.csv"),
    ]
    for src_name, dst_name in raw_files:
        _copy(TABLES_DIR / src_name, raw_dir / dst_name)

    # 地标3D坐标 — 从 results/ground-truth/{sid}/ground_truth.json 合并（仅含 122 分析集）
    from parameterization.landmark_io import parse_landmarks_json

    pred_ids = set(pd.read_csv(ENSEMBLE_PRED_PATH)["subject_id"])
    landmark_rows = []
    for sid_dir in sorted(Path("results/ground-truth").iterdir()):
        if not sid_dir.is_dir() or sid_dir.name not in pred_ids:
            continue
        gt_file = sid_dir / "ground_truth.json"
        if gt_file.exists():
            lm = parse_landmarks_json(str(gt_file))
            row: dict = {"subject_id": sid_dir.name}
            for short, vec in lm.items():
                row[short] = f"({vec[0]},{vec[1]},{vec[2]})"
            landmark_rows.append(row)
    if landmark_rows:
        df_all = pd.DataFrame(landmark_rows)
        df_all.to_csv(raw_dir / "地标3D坐标.csv", index=False)
        logger.info(f"  地标3D坐标: {len(df_all)} × {len(df_all.columns)}")
    else:
        logger.info("  (地标3D坐标数据不可用，跳过)")
    n_raw = sum(1 for _ in raw_dir.iterdir() if _.is_file())
    logger.info(f"  {n_raw} 个文件")
    # 2. 数据表 — Excel from results/eval/tables/
    logger.info("\n[2] 数据表...")
    xl_dir = EXPORT_DIR / "数据表"
    xl_dir.mkdir(exist_ok=True)
    xl_map = {
        "table1.xlsx": "表1_人口学与临床特征.xlsx",
        "table2.xlsx": "表2_体征参数.xlsx",
        "table3.xlsx": "表3_复合指标按严重度分布.xlsx",
        "table4.xlsx": "表4_复合指标与Cobb角相关性.xlsx",
        "table5.xlsx": "表5_Cobb角预测性能.xlsx",
        "table6.xlsx": "表6_分类性能.xlsx",
    }
    for src_name, dst_name in xl_map.items():
        _copy(TABLES_DIR / src_name, xl_dir / dst_name)
    logger.info(f"  {len(xl_map)} 个文件")

    # 3. 特征重要性分析 — CSV from results/modeling/feature_importance/ → 原始数据/
    logger.info("\n[3] 特征重要性分析...")
    fi_dir = EXPORT_DIR / "原始数据"
    fi_map = {
        "feature_importance_decomposed.csv": "特征重要性_95特征排序.csv",
        "importance_by_group.csv": "特征重要性_按测量类型分组.csv",
        "importance_by_horizontal_band.csv": "特征重要性_按UV水平带分组.csv",
    }
    for src_name, dst_name in fi_map.items():
        _copy(FEATURE_IMPORTANCE_DIR / src_name, fi_dir / dst_name)

    # 4. 标准化结果 — from results/parameterization_selected/
    logger.info("\n[4] 标准化结果...")
    param_dir = EXPORT_DIR / "标准化结果"
    if PARAM_SELECTED_DIR.exists():
        for sev_dir in sorted(PARAM_SELECTED_DIR.iterdir()):
            if not sev_dir.is_dir():
                continue
            _cp_dir(sev_dir, param_dir / sev_dir.name)
        n_files = sum(1 for _ in param_dir.rglob("*") if _.is_file())
        logger.info(f"  {n_files} 个文件")
    else:
        logger.info(f"  (目录不存在: {PARAM_SELECTED_DIR})")

    # 5. 说明文档（PDF）
    logger.info("\n[5] 说明文档...")
    guide_dir = Path("/home/nnb/projects/AIS/docs/manuscript/export")
    for fname in ("README.pdf", "technical.pdf"):
        src = guide_dir / fname
        if src.exists():
            shutil.copy2(src, EXPORT_DIR / fname)
            logger.info(f"  {fname}")

    # 6. 打包 ZIP（只打包 PNG 图片，保留 PDF 说明文档）
    logger.info("\n[6] 打包 ZIP...")
    today = date.today().strftime("%Y%m%d")
    zip_path = RESULTS_DIR / f"AIS_数据导出_{today}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in EXPORT_DIR.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(EXPORT_DIR)
            if f.suffix == ".pdf" and "README.pdf" not in str(rel) and "technical.pdf" not in str(rel):
                continue
            zf.write(f, rel)
    size_mb = zip_path.stat().st_size / 1024 / 1024
    logger.info(f"  {zip_path.name} ({size_mb:.0f} MB)")

    logger.info(f"\n导出完成: {EXPORT_DIR}/")
    logger.info(f"压缩包:   {zip_path}")


if __name__ == "__main__":
    main()
