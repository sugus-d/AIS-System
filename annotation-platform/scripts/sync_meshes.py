"""部署导出 mesh → 标注平台 3D 显示 & 刷新 2D 曲率 + 下游缓存。

操作逻辑（幂等）：
  1. 扫描 ./data/ground_truth/{id}/roi.ply
  2. 有 exported_data 的 subject：
      复制 roi.ply → labeling/cache/{id}/extract_roi/roi_edited_{now}.ply
      覆盖原有 roi_edited（各版本只保留最新）
  3. 无 exported_data 但 cache 中有 roi_edited 的 subject：
      删除全部 roi_edited_*.ply → 系统回退到 ./results/roi/{id}/roi.ply
  4. 如果前两者都没有 → 系统已自动回退到原始含衣 mesh（不操作）
  5. 刷新 2D 曲率缓存：
      清除 results/cache/curvature_images/{id}/
      清除 results/extraction/curvature/{id}_*.png
      清除 results/cache/{id}/curvature/
  6. 刷新下游 pipeline 缓存（landmarks 及以前保留）：
      清除 results/cache/{id}/parameterization/
      清除 results/cache/{id}/height/
      清除 results/cache/{id}/segmentation/
      清除 results/cache/{id}/indices/
      清除 results/cache/{id}/region_indices/
      清除 results/cache/{id}/features/
  7. landmarks 不变
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Generator
from pathlib import Path


def _env_path(key: str, default: Path) -> Path:
    """从环境变量读路径，未设置时用默认值（与 backend.constants 一致）。"""
    raw = os.environ.get(key)
    return Path(raw) if raw else default


PLATFORM_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = _env_path("AIS_DATA_ROOT", PLATFORM_DIR / "data")
RESULTS_ROOT = _env_path("AIS_RESULTS_ROOT", PLATFORM_DIR / "results")
EXPORTED_DIR = DATA_ROOT / "ground_truth"
CACHE_DIR = RESULTS_ROOT / "labeling" / "cache"
ROI_DIR = RESULTS_ROOT / "roi"
RESULTS_CACHE = RESULTS_ROOT / "cache"
RESULTS_CURVATURE = RESULTS_ROOT / "extraction" / "curvature"

# pipeline 下游步骤（landmarks 及之前保留）
DOWNSTREAM_STEPS = [
    "parameterization",
    "height",
    "segmentation",
    "indices",
    "region_indices",
    "features",
]


def exported_subjects() -> set[str]:
    """返回所有有 roi.ply 的 subject_id。"""
    if not EXPORTED_DIR.exists():
        return set()
    return {d.name for d in sorted(EXPORTED_DIR.iterdir()) if d.is_dir() and (d / "roi.ply").exists()}


def subjects_with_cache() -> Generator[Path, None, None]:
    """遍历 labeling cache 下所有有 roi_edited PLY 的 subject。"""
    for d in sorted(CACHE_DIR.iterdir()):
        if not d.is_dir():
            continue
        roi_dir = d / "extract_roi"
        if not roi_dir.exists():
            continue
        edited = sorted(roi_dir.glob("roi_edited_*.ply"))
        if edited:
            yield d


def deploy_exported(subject_id: str) -> bool:
    """复制 roi.ply → roi_edited_{now}.ply，清理旧版本。"""
    src = EXPORTED_DIR / subject_id / "roi.ply"
    if not src.exists():
        return False
    dst_dir = CACHE_DIR / subject_id / "extract_roi"
    dst_dir.mkdir(parents=True, exist_ok=True)
    # 清理旧 roi_edited 文件
    for old in dst_dir.glob("roi_edited_*.ply"):
        old.unlink()
    ts = int(time.time() * 1000)
    dst = dst_dir / f"roi_edited_{ts}.ply"
    shutil.copy2(str(src), str(dst))
    print(f"  ✓ {subject_id}: {src.name} → {dst.name}")
    return True


def cleanup_cache(subject_path: Path) -> bool:
    """删除无 exported_data 的 subject 的全部 roi_edited 文件。"""
    subj_id = subject_path.name
    roi_dir = subject_path / "extract_roi"
    removed = 0
    for f in sorted(roi_dir.glob("roi_edited_*.ply")):
        f.unlink()
        removed += 1
    if removed:
        print(f"  ✗ {subj_id}: 删除 {removed} 个 roi_edited PLY → 回退 roi.ply")
    # 如果 extract_roi 空了则清理目录
    if not list(roi_dir.iterdir()):
        roi_dir.rmdir()
    return True


def clear_curvature_cache(subject_id: str):
    """清除 subject 的曲率相关缓存。"""
    # results/cache/{id}/curvature/
    p = RESULTS_CACHE / subject_id / "curvature"
    if p.exists():
        shutil.rmtree(p)
    # results/cache/curvature_images/{id}/
    p = RESULTS_CACHE / "curvature_images" / subject_id
    if p.exists():
        shutil.rmtree(p)
    # results/extraction/curvature/{id}_mean.png / {id}_gauss.png
    for suffix in ("_mean.png", "_gauss.png"):
        f = RESULTS_CURVATURE / f"{subject_id}{suffix}"
        if f.exists():
            f.unlink()


def clear_downstream_cache(subject_id: str):
    """清除 landmark 之后的 pipeline 下游步骤缓存。"""
    for step in DOWNSTREAM_STEPS:
        p = RESULTS_CACHE / subject_id / step
        if p.exists():
            shutil.rmtree(p)


def refresh_caches(subject_id: str):
    """刷新所有依赖 mesh 的缓存（landmarks 不变）。"""
    clear_curvature_cache(subject_id)
    clear_downstream_cache(subject_id)


def main():
    print("=" * 60)
    print("【AIS】部署导出 mesh → 标注平台 3D 显示 & 刷新缓存")
    print("=" * 60)
    print()

    exported = exported_subjects()
    print(f"[发现] exported_data 含 roi.ply: {len(exported)} subject")
    print()

    # ── Phase 1: 部署 exported → cache ──
    print("── Phase 1: 部署 ground_truth → labeling cache ──")
    deployed = 0
    for subj_id in sorted(exported):
        if deploy_exported(subj_id):
            deployed += 1
    print(f"结果: {deployed}/{len(exported)} subject 部署成功")
    print()

    # ── Phase 2: 清理无 exported 的 cache ──
    print("── Phase 2: 清理无 exported_data 的 roi_edited ──")
    cleaned = 0
    skipped = 0
    for subject_path in subjects_with_cache():
        sid = subject_path.name
        if sid in exported:
            # 已经在 Phase 1 处理过了，跳过
            skipped += 1
            continue
        cleanup_cache(subject_path)
        cleaned += 1
    print(f"结果: 清理 {cleaned} subject, 跳过(已部署) {skipped} subject")
    print()

    # ── Phase 3: 刷新缓存 ──
    print("── Phase 3: 刷新 2D 曲率 + downstream pipeline 缓存 ──")
    # 对所有有 cache 或 exported 的 subject 刷新
    all_cache_subjects: set[str] = set()
    if CACHE_DIR.exists():
        for d in CACHE_DIR.iterdir():
            if d.is_dir():
                all_cache_subjects.add(d.name)
    all_subjects_to_refresh = exported | all_cache_subjects
    refreshed = 0
    for subj_id in sorted(all_subjects_to_refresh):
        refresh_caches(subj_id)
        refreshed += 1
    print(f"结果: 刷新 {refreshed}/{len(all_subjects_to_refresh)} subject 的缓存")
    print()

    # ── Summary ──
    print("=" * 60)
    print("完成。")
    print(f"  - 部署: {deployed} subject → roi_edited")
    print(f"  - 清理: {cleaned} subject → 回退 roi.ply")
    print(f"  - 缓存刷新: {refreshed} subject")
    print()
    print("标注平台 3D mesh 读取优先级: roi_edited > roi.ply > 原始含衣 PLY")
    print("2D 曲率热力图 & downstream pipeline 缓存已清除，下次访问/运行时自动重建。")
    print("=" * 60)


if __name__ == "__main__":
    main()
