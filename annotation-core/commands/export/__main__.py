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
from commands.export.config import EXPORT_DIR, FEATURE_IMPORTANCE_DIR, RESULTS_DIR, TABLES_DIR  # noqa: E402
from utils.logger import logger  # noqa: E402


def _clean():
    logger.info("=" * 60)
    logger.info("清理旧数据...")
    for d in [TABLES_DIR, EXPORT_DIR, FEATURE_IMPORTANCE_DIR]:
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
    _clean()

    # 1. 特征重要性分析（morph_region_ci_40d + Permutation Importance + CI 反解）
    logger.info("\n[1/8] 特征重要性分析...")
    from commands.export.analyze import main as fi_main
    fi_main()

    # 2. 特征重要性可视化
    logger.info("\n[2/8] 特征重要性可视化...")
    from commands.export.charts_feature_importance import main as pi_main
    pi_main()

    # 3. 论文关键图（散点图、Bland-Altman、混淆矩阵）
    logger.info("\n[3/8] 论文关键图...")
    from commands.export.figures import main as fig_main
    fig_main()

    # 4. 瀑布图 + 树结构 + 残差收敛
    logger.info("\n[4/8] 特征贡献分析（瀑布图/树结构）...")
    from commands.export.charts_waterfall import main as wf_main
    wf_main()

    # 5. 数据表 + Excel
    logger.info("\n[5/8] 数据表...")
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

    logger.info(f"\n{'=' * 60}")
    logger.info("导出完成！")
    zips = list(RESULTS_DIR.glob("AIS_数据导出_*.zip"))
    if zips:
        logger.info(f"压缩包: {zips[-1]}")


if __name__ == "__main__":
    main()
