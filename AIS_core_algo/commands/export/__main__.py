#!/usr/bin/env python3
"""从零导出全流程：清理 → 计算 → 图片 → 表格 → 打包。

用法:
    python -m commands.export
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from commands.export import excel, pack, raw_tables, tables  # noqa: E402  # 需在 sys.path 注入后导入
from utils.logger import logger  # noqa: E402
from utils.paths import (  # noqa: E402
    EVAL_TABLES_DIR,
    EXPORT_DIR,
    FEATURE_IMPORTANCE_DIR,
    MANUAL_FIGURES_DIR,
    MANUAL_PRED_PATH,
    MANUAL_REGION_CSV,
    MANUAL_TABLES_DIR,
    RESULTS_DIR,
)


def _clean():
    logger.info("=" * 60)
    logger.info("清理旧数据...")
    for d in [EVAL_TABLES_DIR, EXPORT_DIR, FEATURE_IMPORTANCE_DIR]:
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"  删除: {d}")
    for z in RESULTS_DIR.glob("AIS_数据导出_*.zip"):
        z.unlink()
        logger.info(f"  删除: {z}")
    # 清理分析中间结果
    for d in [
        RESULTS_DIR / "analysis" / "feature_contributions",
    ]:
        if d.exists():
            shutil.rmtree(d)
            logger.info(f"  删除: {d}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AIS 论文导出管线")
    parser.add_argument(
        "--scheme", choices=["v0.1.0", "v1.0.0", "beta", "production"], default="v0.1.0",
        help="特征重要性分析方案：v0.1.0/beta（算法 ROI，默认）或 v1.0.0/production（人工 ROI，CI 分解同口径）",
    )
    args = parser.parse_args()
    from features.selectors import get_selector

    is_production = get_selector(args.scheme).version == "v1.0.0"

    _clean()

    # 1. 特征重要性分析（scheme 切换，非替代：beta 现场训练 / production 模型包内嵌模型）
    logger.info(f"\n[1/8] 特征重要性分析（scheme={args.scheme}）...")
    from commands.export.analyze import main as fi_main
    fi_main(args.scheme)

    # 2. 特征重要性可视化
    logger.info("\n[2/8] 特征重要性可视化...")
    from commands.export.charts_feature_importance import main as pi_main
    if is_production:
        from utils.paths import EXPORT_DIR
        pi_main(EXPORT_DIR / "v1.0.0" / "feature_importance")
    else:
        pi_main()

    # 3. 论文关键图（散点图、Bland-Altman、混淆矩阵）
    logger.info(f"\n[3/8] 论文关键图（scheme={args.scheme}）...")
    from commands.export.figures import main as fig_main
    if is_production:
        fig_main(MANUAL_PRED_PATH, MANUAL_FIGURES_DIR)
    else:
        fig_main()

    # 4. 瀑布图 + 树结构 + 残差收敛
    logger.info("\n[4/8] 特征贡献分析（瀑布图/树结构）...")
    from commands.export.charts_waterfall import main as wf_main
    wf_main()

    # 5. 数据表 + Excel
    logger.info(f"\n[5/8] 数据表（scheme={args.scheme}）...")
    if is_production:
        tables.main(MANUAL_PRED_PATH, MANUAL_REGION_CSV, MANUAL_TABLES_DIR)
        raw_tables.main(MANUAL_PRED_PATH, MANUAL_REGION_CSV, MANUAL_TABLES_DIR)
        excel.main(MANUAL_TABLES_DIR)
    else:
        tables.main()
        raw_tables.main()
        excel.main()

    # 6. 数据清单
    logger.info("\n[6/8] 数据清单...")
    from commands.export import inventory
    inventory.main()

    # 7. 说明文档 PDF
    logger.info("\n[7/8] 说明文档 PDF...")
    import subprocess
    guide_dir = Path("/home/nnb/projects/AIS/docs/manuscript/export")
    for md in ["README.md", "technical.md"]:
        pdf_path = guide_dir / md.replace(".md", ".pdf")
        result = subprocess.run(
            [str(Path(__file__).resolve().parent / "html2pdf.sh"),
             str(guide_dir / md), str(pdf_path)],
            capture_output=True, text=True)
        if result.returncode != 0:
            logger.info(f"  WARNING: PDF generation failed for {md}")
            logger.info(f"    stderr: {result.stderr.strip()[-200:]}")
        else:
            logger.info(f"  {pdf_path.name}")

    # 8. 打包
    logger.info("\n[8/8] 打包 ZIP...")
    pack.main()

    # 9. v1.0.0 方案批量导出（--scheme v1.0.0/production 时启用；特征重要性已在步骤 1 输出）
    if is_production:
        logger.info("\n[9/9] v1.0.0 批量导出（indices + 瀑布图）...")
        from commands.export.v1_0_0_export import main as v100_main
        v100_main(run_analysis=False)

    logger.info(f"\n{'=' * 60}")
    logger.info("导出完成！")
    zips = list(RESULTS_DIR.glob("AIS_数据导出_*.zip"))
    if zips:
        logger.info(f"压缩包: {zips[-1]}")


if __name__ == "__main__":
    main()
