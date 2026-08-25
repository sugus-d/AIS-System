"""Subject 发现与元数据加载。"""

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache

from ..constants import BILATERAL_LANDMARKS, CACHE_DIR, CONFIG_PATH, DATA_ROOT, GT_DIR, MESH_DIR, SPINE_POINT_COUNT


@dataclass
class SubjectInfo:
    """Subject 基本信息（列表用，轻量）。"""

    id: str
    has_gt: bool
    has_cache: bool


def discover_subjects() -> list[SubjectInfo]:
    """扫描 data/mesh/ 下全部 subject 目录，返回 SubjectInfo 列表。

    仅做目录扫描和文件存在性检查，不做 glob / 文件读取 / 临床数据加载。
    """
    if not MESH_DIR.exists():
        return []
    subjects = []
    for d in sorted(MESH_DIR.iterdir()):
        if not d.is_dir():
            continue
        sid = d.name
        gt_file = GT_DIR / sid / "ground_truth.json"
        has_gt = gt_file.exists()
        lm_file = CACHE_DIR / sid / "landmarks" / "landmarks.pkl"
        curv_file = CACHE_DIR / sid / "curvature" / "mean_curvature.npy"
        has_cache = lm_file.exists() or curv_file.exists()
        subjects.append(SubjectInfo(id=sid, has_gt=has_gt, has_cache=has_cache))
    return subjects


def _load_clinical(subject_id: str) -> tuple[str | None, str | None, str | None]:
    """从 config.yaml 或 features.pkl 读取临床信息（age/sex/bmi）。"""
    # 优先从 config.yaml
    if CONFIG_PATH.exists():
        import yaml

        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        for s in cfg.get("subjects", []):
            if s["id"] == subject_id:
                c = s.get("clinical", {})
                return (str(c.get("age", "?")), str(c.get("sex", "?")), str(c.get("bmi", "?")))
    # 其次从 features.pkl
    feat_path = CACHE_DIR / subject_id / "features" / "features.pkl"
    if feat_path.exists():
        feat = pickle.loads(feat_path.read_bytes())
        if hasattr(feat, "columns"):
            row = feat.iloc[0]
            return (str(row.get("age", "?")), str(row.get("sex", "?")), str(row.get("bmi", "?")))
    return ("?", "?", "?")


def _find_mesh_file(subject_id: str) -> str | None:
    """找到 subject 的 mesh PLY 文件路径（延迟加载，仅在需要时调用）。"""
    d = MESH_DIR / subject_id
    if not d.exists():
        return None
    mesh_files = sorted(d.glob("*mesh*.ply")) + sorted(d.glob("*.ply"))
    if mesh_files:
        return str(mesh_files[0].relative_to(DATA_ROOT))
    return None


@lru_cache(maxsize=512)
def get_gt_features(subject_id: str) -> dict[str, object]:
    """读取 GT 的 _features 字段（手臂状态、不对称标志）。"""
    gt_file = GT_DIR / subject_id / "ground_truth.json"
    if gt_file.exists():
        gt = json.loads(gt_file.read_text())
        return gt.get("_features", {})
    return {}


# ── 标注状态计算（18 个 landmark 完整度） ──────────────────

BILATERAL_PAIRS: list[str] = BILATERAL_LANDMARKS
PAIR_SIDES = 2  # bilateral 成对 landmark 的 L/R 数量
LANDMARK_TOTAL: int = len(BILATERAL_PAIRS) * 2 + SPINE_POINT_COUNT  # 18


def compute_labeling_status(subject_id: str) -> str:
    """根据 landmark 数据计算标注状态: unlabeled / prelabeled / labeled。

    优先读取 GT 中的手动覆盖（_features.labeling_status），
    否则从 landmark 完整度自动计算：
      - 18 个非 None → labeled
      - 无 GT → unlabeled
      - GT 不完整 → prelabeled
    """
    gt_file = GT_DIR / subject_id / "ground_truth.json"
    if gt_file.exists():
        gt = json.loads(gt_file.read_text())
        # 从实际 landmark 数据实时计算，不依赖 _features 中的陈旧状态
        completed = 0
        for name in BILATERAL_PAIRS:
            pair = gt.get(name, {})
            if not isinstance(pair, dict):
                continue
            if pair.get("L") is not None:
                completed += 1
            if pair.get("R") is not None:
                completed += 1
        spine = gt.get("spine_points", [])
        completed += sum(1 for pt in spine if pt is not None)
        if completed >= LANDMARK_TOTAL:
            return "labeled"
        return "prelabeled" if completed > 0 else "unlabeled"

    # 无 GT → 未标（无论有无算法缓存）
    return "unlabeled"


def set_manual_labeling_status(subject_id: str, status: str) -> None:
    """手动覆盖标注状态，写入 GT 的 _features.labeling_status。"""
    gt_dir = GT_DIR / subject_id
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_file = gt_dir / "ground_truth.json"
    data = json.loads(gt_file.read_text()) if gt_file.exists() else {}
    features = data.setdefault("_features", {})
    features["labeling_status"] = status
    gt_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def get_algorithm_features(subject_id: str) -> dict:
    """从 axilla_debug 推断手臂状态，从 neck_root 推断左右不对称。"""
    lm_file = CACHE_DIR / subject_id / "landmarks" / "landmarks.pkl"
    if not lm_file.exists():
        return {}
    data = pickle.loads(lm_file.read_bytes())
    ad = data.get("axilla_debug", {})
    arms = []
    for side in ["left", "right"]:
        sd = ad.get(side, {})
        if isinstance(sd, dict) and sd.get("has_arm", False):
            arms.append(side)
    features: dict = {"arms": ",".join(arms) if arms else "none"}
    nr = data.get("neck_root", [])
    if isinstance(nr, list) and len(nr) >= PAIR_SIDES:
        try:
            dy = abs(float(nr[0][1]) - float(nr[1][1]))
        except (TypeError, IndexError):
            dy = 0
        features["neck_root_y_asym"] = round(dy, 1)
    return features
