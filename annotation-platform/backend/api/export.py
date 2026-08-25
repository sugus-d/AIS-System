"""异步文件导出：原始 mesh + 去衣 mesh（Landmark 以 JSON 为唯一格式，不再导出 CSV）。"""

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter

from ..constants import (
    BILATERAL_LANDMARKS,
    CACHE_DIR,
    DATA_ROOT,
    MESH_DIR,
    SPINE_POINT_COUNT,
)
from ..services.lifter import _get_latest_edited, load_landmarks
from ..services.subject_loader import compute_labeling_status, discover_subjects

router = APIRouter(prefix="/api/export", tags=["export"])

EXPORT_DIR: Path = DATA_ROOT / "ground_truth"

# 导出任务进度存储
_export_tasks: dict[str, dict] = {}

_SIDE_COUNT: int = 2  # 左右双侧数


def _copy_ply_if_changed(src: Path, dst: Path) -> None:
    """如果目标文件存在且大小一致则跳过，否则复制。"""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    shutil.copy2(str(src), str(dst))


def _run_export(task_id: str) -> None:
    """执行异步导出：原始 mesh + 去衣 mesh（Landmark 以 JSON 为唯一格式）。"""
    task = _export_tasks[task_id]
    subjects = discover_subjects()
    # 只导出已标（labeled）的 subject
    labeled = [s for s in subjects if compute_labeling_status(s.id) == "labeled"]
    task["total"] = len(labeled)
    task["status"] = "running"

    for i, s in enumerate(labeled):
        try:
            out_dir = EXPORT_DIR / s.id
            out_dir.mkdir(parents=True, exist_ok=True)

            # 0. 校验 landmark 是否完整（ground_truth.json 为准）
            lm = load_landmarks(s.id)
            completed = 0
            total_expected = SPINE_POINT_COUNT + len(BILATERAL_LANDMARKS) * _SIDE_COUNT
            for name in BILATERAL_LANDMARKS:
                pts = lm.get(name, [])
                if pts and pts[0] is not None:
                    completed += 1
                if len(pts) >= _SIDE_COUNT and pts[1] is not None:
                    completed += 1
            sp = lm.get("spine_points", [])
            completed += sum(1 for pt in sp if pt is not None)
            if completed < total_expected:
                task["error"] = f"{s.id}: 标注不完整 ({completed}/{total_expected})"
                task["done"] = i + 1
                continue

            # 1. 原始 mesh（有衣服）— 优先 STD_fuse_mesh → STD*_fuse_mesh
            mesh_dir = MESH_DIR / s.id
            orig_ply = (sorted(mesh_dir.glob("STD_fuse_mesh_*.ply"))
                       or sorted(mesh_dir.glob("STD*_fuse_mesh_*.ply")))
            if orig_ply:
                _copy_ply_if_changed(orig_ply[0], out_dir / "original.ply")

            # 2. 最新去衣 mesh — 优先 edited → declothed cache → 原始 fallback
            edited = _get_latest_edited(s.id)
            if edited:
                _copy_ply_if_changed(Path(edited), out_dir / "roi.ply")
            else:
                de_ply = CACHE_DIR / s.id / "extract_roi" / "output.ply"
                if de_ply.exists():
                    _copy_ply_if_changed(de_ply, out_dir / "roi.ply")
                elif orig_ply:
                    _copy_ply_if_changed(orig_ply[0], out_dir / "roi.ply")

        except Exception as e:  # noqa: BLE001 — 单 subject 失败不中断整体导出
            task["error"] = f"{s.id}: {e}"
        task["done"] = i + 1

    task["status"] = "done"


@router.post("/data-export")
def start_data_export() -> dict:
    """启动异步文件导出任务（原始 mesh + 去衣 mesh）。"""
    task_id = uuid.uuid4().hex[:12]
    _export_tasks[task_id] = {"done": 0, "total": 0, "status": "pending", "error": None}
    t = threading.Thread(target=_run_export, args=(task_id,), daemon=True)
    t.start()
    return {"task_id": task_id}


@router.get("/data-export/{task_id}")
def get_export_progress(task_id: str) -> dict:
    """查询异步导出任务的进度。"""
    task = _export_tasks.get(task_id)
    if not task:
        return {"error": "task not found"}
    return {
        "done": task["done"],
        "total": task["total"],
        "status": task["status"],
        "error": task.get("error"),
    }
