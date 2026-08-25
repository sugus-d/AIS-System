"""特征方案注册表 — 定义从上游提取/筛选哪些特征。

EXTRACTION_REGISTRY: 定义从 mesh+landmarks 提取哪些原始特征。
SELECTION_REGISTRY:  定义从提取结果中筛选出哪些特征用于训练。

方案分级（2026-08 收敛，依据 122-subject 可比 MF1）:
  - morph_region_ci_40d:  默认方案（pipeline 默认 + modeling.train 默认），文档最优 🏆 MF1 0.669
  - archived/*:          历史保留方案（可加载复现，不再推荐），key 含 archived/ 前缀
  - 已删除（2026-08-01）: anova_29d(0.601) / r_ranked_41d(0.585) /
                          normal_enhanced_40d(0.583) / dual_ci_ai_41d(明确未提升)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from features.selectors._loaders import (  # noqa: F401 — load_fn 直接引用
    _compute_ci_features,
    _load_canonical_44d,
    _load_canonical_union_64d,
    _load_dual_ci,
    _load_extraction,
    _load_morph_region_ci_27d,
    _load_selection,
)
from utils.logger import logger


@dataclass
class FeatureScheme:
    """特征方案元数据 + 加载函数。

    每个实例表示一个固定的特征选择方案，含：
    - 描述性元数据（选择算法、来源、维度）
    - load() 返回 cross_validate 兼容的数据 dict
    - 向前兼容：新增字段不影响已有调用方。
    """

    # ---- 标识 ----
    name: str
    label: str
    description: str

    # ---- 构成 ----
    n_features: int
    components: dict[str, int]  # e.g. {"basic": 5, "ci": 3, "morph": 14, "region": 25}

    # ---- 选择算法 ----
    selection_method: str  # "per-fold" | "global-static" | "global-hybrid"
    selection_pipeline: str  # 人类可读的流程描述
    selection_ref: list[str]  # 参考代码/文档路径
    feature_names: list[str] | None  # None = per-fold 不固定

    # ---- 来源 ----
    source_files: list[str]  # CSV/Parquet 文件名

    # ---- 加载函数 ----
    load_fn: callable = field(repr=False, compare=False)

    def load(self) -> dict:
        """加载数据，返回 cross_validate 兼容 dict。"""
        logger.info(f"加载方案: {self.label} ({self.n_features}D)")
        return self.load_fn()

    def summary(self) -> str:
        return (
            f"[{self.name:20s}] {self.label:14s}  "
            f"{self.n_features:2d}D  {self.selection_method:15s}  "
            f"组件: {self.components}"
        )


# ── Extraction Registry ─────────────────────────────────────
EXTRACTION_REGISTRY: dict[str, FeatureScheme] = {
    "extract_back_v1": FeatureScheme(
        name="extract_back_v1",
        label="背部表面提取 v1",
        description="特征提取: basic(5) + morph(31) + region_asym(2700)",
        n_features=2736,
        components={"basic": 5, "morph": 31, "region": 2700},
        selection_method="fixed-extraction",
        selection_pipeline="extract_morphology()→31D + region_asymmetry from UV mesh",
        selection_ref=["features/morphology/__init__.py",
                       "features/asymmetry/"],
        feature_names=None,
        source_files=["results/extraction/features_extraction/back_v1/basic.csv",
                       "results/extraction/features_extraction/back_v1/morphology.csv",
                       "results/extraction/features_extraction/back_v1/region_asymmetry.csv"],
        load_fn=lambda: _load_extraction("back_v1"),
    ),
}


# ── Selection Registry ───────────────────────────────────────
SELECTION_REGISTRY: dict[str, FeatureScheme] = {
    "archived/morph_region_ci_37d": FeatureScheme(
        name="archived/morph_region_ci_37d",
        label="Morph+Region+CI 37D（archived）",
        description="basic(5) + morph(10) + region(18) + CI(4)；曾为 pipeline 默认",
        n_features=37,
        components={"basic": 5, "morph": 10, "region": 18, "ci": 4},
        selection_method="dual-condition+lasso",
        selection_pipeline=(
            "Morph: |r|>0.15 + ANOVA p<0.05 → dedup|r|>0.85 → top 10 | "
            "Region: |r|>0.2 + ANOVA p<0.01 → dedup → LassoCV → top 18 | "
            "CI: 12 groups → dedup → Lasso → pw/dm → top 4"
        ),
        selection_ref=["features/selectors/_loaders.py:_load_selection"],
        feature_names=None,
        source_files=["results/extraction/features_selection/back_v1/morphology.csv",
                       "results/extraction/features_selection/back_v1/region_asymmetry.csv",
                       "results/extraction/features_selection/back_v1/ci.csv"],
        load_fn=lambda: _load_selection("back_v1"),
    ),
    "archived/canonical_union_64d": FeatureScheme(
        name="archived/canonical_union_64d",
        label="Canonical 并集 64D（archived）",
        description="morph_region_ci_37d(37) ∪ canonical_44d(44) 去重 = 64D（2 个 canonical CI 不存在于提取）",
        n_features=64,
        components={"basic": 5, "morph": 16, "region": 39, "ci": 4},
        selection_method="screened ∪ canonical",
        selection_pipeline="morph_region_ci_37d 与 canonical_44d 的完全并集",
        selection_ref=["features/selectors/schemes.py:_load_canonical_union_64d"],
        feature_names=None,
        source_files=["results/extraction/features_extraction/back_v1/morphology.csv",
                       "results/extraction/features_extraction/back_v1/region_asymmetry.csv",
                       "results/extraction/features_selection/back_v1/ci.csv"],
        load_fn=lambda: _load_canonical_union_64d("back_v1"),
    ),
}


# ── 文档对齐 36D — 按构成语义重建 ─────────────────────
SELECTION_REGISTRY["archived/morph_region_ci_36d"] = FeatureScheme(
    name="archived/morph_region_ci_36d",
    label="Morph+Region+CI 36D（archived）",
    description="按文档重建: Morph 10 + Region 17 + CI 4, 纯 |r| 排序, CI 自建",
    n_features=36,
    components={"basic": 5, "morph": 10, "region": 17, "ci": 4},
    selection_method="docs-aligned",
    selection_pipeline="OR双条件→|r|降序去高相关→LassoCV→CI自建(Lasso+线性组合)",
    selection_ref=["features/selectors/schemes.py:_load_selection", "features/selectors/scheme_morph_region_ci_35d.py:load_36d"],
    feature_names=None,
    source_files=[
        "results/extraction/features_extraction/back_v1/basic.csv",
        "results/extraction/features_extraction/back_v1/morphology.csv",
        "results/extraction/features_extraction/back_v1/region_asymmetry.csv",
    ],
    load_fn=lambda: __import__("features.selectors.scheme_morph_region_ci_35d", fromlist=["load_36d"]).load_36d(),
)

# ── 单条件 35D ────────────────────────────────────
SELECTION_REGISTRY["archived/morph_region_ci_35d"] = FeatureScheme(
    name="archived/morph_region_ci_35d",
    label="Morph+Region+CI 35D（archived）",
    description="Basic(5)+Morph|r|>0.15→dedup→top10 + Region|r|>0.2→dedup→LassoCV→top + CI(6=自建12组各measure择优)",
    n_features=35,
    components={"basic": 5, "morph": 10, "region": 14, "ci": 6},
    selection_method="single-condition |r|",
    selection_pipeline="纯 |Pearson r| 排序 + 去高相关 + LassoCV, 无 ANOVA",
    selection_ref=["features/selectors/scheme_morph_region_ci_35d.py"],
    feature_names=None,
    source_files=[
        "results/extraction/features_extraction/back_v1/basic.csv",
        "results/extraction/features_extraction/back_v1/morphology.csv",
        "results/extraction/features_extraction/back_v1/region_asymmetry.csv",
    ],
    load_fn=lambda: __import__("features.selectors.scheme_morph_region_ci_35d", fromlist=["load"]).load(),
)


# ── 最终版：Dual CI 40D ─────────────────────────────
SELECTION_REGISTRY["archived/morph_region_ci_27d"] = FeatureScheme(
    name="archived/morph_region_ci_27d",
    label="Morph+Region+CI 42D（archived）",
    description="CI 占用剔除→剩余池双条件→LassoCV 非零全量",
    n_features=42,
    components={"basic": 5, "morph": 10, "region": 21, "ci": 4, "ci10": 1, "ci20": 1},
    selection_method="CI-first + dedup-region",
    selection_pipeline="CI构建→标记占用region→剔除→剩余region双条件→LassoCV非零全量",
    selection_ref=["features/selectors/schemes.py:_load_morph_region_ci_27d"],
    feature_names=None,
    source_files=[
        "results/extraction/features_extraction/back_v1/basic.csv",
        "results/extraction/features_extraction/back_v1/morphology.csv",
        "results/extraction/features_extraction/back_v1/region_asymmetry.csv",
    ],
    load_fn=_load_morph_region_ci_27d,
)

SELECTION_REGISTRY["morph_region_ci_40d"] = FeatureScheme(
    name="morph_region_ci_40d",
    label="Morph+Region+CI 40D",
    description="Region放松(r>0.15)+LassoCV→4CI+10°CI+20°CI, 0.669；默认方案",
    n_features=40,
    components={"basic": 5, "morph": 10, "region": 19, "ci": 4, "ci10": 1, "ci20": 1},
    selection_method="docs-aligned + dual-CI",
    selection_pipeline="OR双条件→|r|降序去高相关→LassoCV→CI自建+10°CI+20°CI",
    selection_ref=["features/selectors/schemes.py:_load_dual_ci"],
    feature_names=None,
    source_files=[
        "results/extraction/features_extraction/back_v1/basic.csv",
        "results/extraction/features_extraction/back_v1/morphology.csv",
        "results/extraction/features_extraction/back_v1/region_asymmetry.csv",
    ],
    load_fn=_load_dual_ci,
)

# canonical_44d: canonical 固定参考集（从 back_v1 提取加载）
SELECTION_REGISTRY["archived/canonical_44d"] = FeatureScheme(
    name="archived/canonical_44d",
    label="Canonical 44D（archived）",
    description="canonical 固定集 45 维中去掉 back_v1 不存在维 = 44D",
    n_features=44,
    components={"basic": 5, "morph": 11, "region": 24, "ci": 4},
    selection_method="global-static",
    selection_pipeline="canonical_47d 固定集，从 back_v1 提取数据加载（原数据仅 N=60）",
    selection_ref=["features/selectors/schemes.py:_load_canonical_44d"],
    feature_names=None,
    source_files=[
        "results/extraction/features_extraction/back_v1/basic.csv",
        "results/extraction/features_extraction/back_v1/morphology.csv",
        "results/extraction/features_extraction/back_v1/region_asymmetry.csv",
        "results/extraction/features_selection/back_v1/ci.csv",
    ],
    load_fn=lambda: _load_canonical_44d("back_v1"),
)
