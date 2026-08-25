"""ROI 评估数据加载 — 从预计算结果 JSON 读取。"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

EVAL_PATH = Path("results/eval/cut_eval/eval_pipeline.json")
REGION_PATH = Path("results/eval/cut_eval/eval_region_deltri.json")
ROI_DIR = Path("results/roi")
EXPORT_DIR = Path("data/ground_truth")


@st.cache_data(ttl=60)
def load_eval() -> dict:
    """加载 87 例评估结果，按 subject 索引。"""
    if not EVAL_PATH.exists():
        return {}
    return {r["subject"]: r for r in json.loads(EVAL_PATH.read_text())}


@st.cache_data(ttl=60)
def load_regions() -> dict:
    """加载区域 Δtri 结果。"""
    if not REGION_PATH.exists():
        return {}
    return {r["subject"]: r for r in json.loads(REGION_PATH.read_text())}


def clear_cache() -> None:
    st.cache_data.clear()


def roi_path(sid: str) -> Path:
    return ROI_DIR / sid / "roi.ply"


def gt_path(sid: str) -> Path:
    return EXPORT_DIR / sid / "roi.ply"


def subject_list() -> list[dict]:
    """扫描所有 subject 的基本信息（快速，只读文件头）。"""
    subjects = []
    for sd in sorted(EXPORT_DIR.iterdir()):
        if not sd.is_dir():
            continue
        sid = sd.name
        oc = _ply_header(sd / "original.ply")
        rc = _ply_header(roi_path(sid))
        if oc is None:
            continue
        subjects.append({
            "subject": sid,
            "orig_vertices": oc[0],
            "roi_vertices": rc[0] if rc else 0,
            "ratio": round(rc[0] / max(oc[0], 1), 4) if rc else 0,
            "has_gt": gt_path(sid).exists(),
        })
    return subjects


def _ply_header(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as f:
            head = f.read(4096).decode("latin-1")
        v, t = 0, 0
        for line in head.splitlines():
            line = line.strip()
            if line.startswith("element vertex"):
                v = int(line.split()[-1])
            elif line.startswith("element face"):
                t = int(line.split()[-1])
            elif line == "end_header":
                break
        return (v, t) if v > 0 else None
    except Exception:
        return None
