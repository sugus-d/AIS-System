> ⚠️ **历史设计文档**。命名已版本化演进（ml/→modeling/、back_v1→v0.1.0、manual_roi→v1.0.0、presets 已并入 TRAINING_SCHEMES、scheme key 统一版本号），仅作历史设计参考，不再反映当前代码。

# AIS 外科清理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **状态：已完成（2026-08-02 收口）**。计划中 `ml/` 目录已更名为 `modeling/`；原 `ml/schemes.py` 拆分为
> `features/selectors/schemes.py`（特征方案 SELECTION_REGISTRY）+ `modeling/training/schemes.py`（训练方案）；
> `_make_data_dict` 已内联删除；实验文件（round2 等）与 `ml/legacy/`、`ml/cv.py` 均已删除。下文 `ml/` 引用为史实路径。

**Goal:** Engineering cleanup only — delete dead code, merge dual tracks, make pipeline/names honest, fix docs drift. No algorithm changes, no new deps, no functional regression.

**Architecture:** 6-phase plan built on spec `docs/refactor/2026-07-28-surgical-cleanup-refactor-design.md`. Each phase is independent and mergeable. Phases 4-6 can parallelize.

**Tech Stack:** Python 3.10+ / NumPy / Open3D / scikit-learn / pytest / ruff

## Global Constraints

- Zero new dependencies
- No changes to mesh ROI / landmark / parameterization **algorithm** code (thresholds, scoring, detection logic)
- `pytest -q` main suite green on every PR
- `ruff check` zero errors on changed files
- `data/ground_truth/` layout and labeling export contract preserved unchanged
- Verify set 5 subject (S0006, S0016, S0069, S0089, S0107) human GT untouched
- Refactored code must delete old files — no deprecation stubs or forwarding wrappers
- Per PR: feature scheme load + training metrics vs baseline (morph_region_ci_40d + GBRT, fixed seed), Macro-F1 Δ ≤ 0.01
- `tools/` and `commands/` coexistence ends with Phase 0

---
## Phase Map

```
Phase 0  — Commit tools→commands migration (3 tasks, blocks everything)
Phase 1  — Delete dead code (3 tasks, parallel within)
Phase 2  — Pipeline honest + FEATURE_SCHEME fix (5 tasks)
Phase 3  — Scheme single-track (4 tasks, highest-risk)
Phase 4  — Large file splitting (2-3 tasks, fully parallel)
Phase 5  — Commands layering + docs truth (3 tasks, parallel)
Phase 6  — Test alignment (2 tasks)
```

---

## Phase 0: Commit tools→commands Migration

### Task 0.1: Git-commit working directory changes

**Files:**
- No new code — just git operations

- [ ] **Step 1: Commit all staged/unstaged python code changes in `commands/`**

```bash
# The tools/ deletions are already staged (D status).
# commands/ files are unstaged (?? status).
# Review + stage them:
git status --short -- ':!:docs' ':!.claude' ':!.codebase-memory'
# Stage all commands/ additions
git add commands/
git commit -m "refactor: migrate tools/ to commands/ directory

- tools/ deleted, commands/ becomes single CLI entry point
- Update all internal references from tools.* to commands.*"
```

- [ ] **Step 2: Update pyproject.toml scripts to use `commands.`**

```python
# In pyproject.toml under [project.scripts]:
# Change tools.* → commands.*
# E.g.: "export = commands.export.__main__:main"
```

```bash
rg 'tools\.' pyproject.toml
# Edit each to commands. instead, then:
git add pyproject.toml
```

- [ ] **Step 3: Update doc references from `tools.` to `commands.`**

```bash
# Search all .md files for stale tools. refs
rg -n 'tools\.| -m tools' .claude/ docs/ --type md | grep -v superpowers | grep -v '.git'
```

- `.claude/PROJECT.md`: change `-m tools.export` → `-m commands.export`
- `docs/refactor/*`: same
- `CLAUDE.md`: if any

```bash
git add .claude/PROJECT.md CLAUDE.md
```

- [ ] **Step 4: Fix pipeline/config.py default config path**

pipeline/config.py:10 still points to `"tools" / "ais-cli.yaml"`:

```python
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "tools" / "ais-cli.yaml"
```

Change to:
```python
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"
```

Check that `commands/ais-cli.yaml` exists (or create a minimal default from the existing DEFAULT_CONFIG dict). If not, create `pipeline/default_config.yaml` mirroring the current DEFAULT_CONFIG dict content.

- [ ] **Step 5: Final commit and verify**

```bash
git add -A
git commit -m "chore: update pyproject, config path, and docs for commands/ migration"

# Verify no tools. references remain in production imports:
rg -n 'tools\.' --type py -g '!tests/' -g '!__pycache__/' || echo 'clean'
pytest -q tests/test_cli.py 2>&1 | tail -3
git log --oneline -3
```

---

## Phase 1: Delete Dead Code

**Safe to parallelize** — delete tasks don't share files.

### Task 1.1: Delete mesh/roi_old/ and duplicate registry

**Files:** (all deletions)

- [ ] **Step 1: Verify no imports**

```bash
rg 'roi_old|mesh\.roi_old' --type py -g '!__pycache__/' || echo 'clean — no refs'
```

- [ ] **Step 2: Delete files**

```bash
rm -rf mesh/roi_old/
```

- [ ] **Step 3: Remove duplicate mesh/roi/registry.py**

```bash
# Verify no imports of mesh.roi.registry (not roi.registry):
rg 'mesh\.roi\.registry|from mesh\.roi import registry' --type py -g '!__pycache__/' || echo 'clean'
# Delete:
rm mesh/roi/registry.py
```

- [ ] **Step 4: Run tests**

```bash
pytest -q tests/test_mesh_graph.py tests/test_cut_analysis.py tests/test_region_eval.py 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete mesh/roi_old/ and duplicate mesh/roi/registry.py"
```

### Task 1.2: Archive experimental ML files

**Files:** ml/round2.py, ml/round3.py, ml/parallel_train.py, ml/run_ensemble.py, ml/test_augment.py（均已删除）

- [ ] **Step 1: Verify no production imports**

```bash
for f in round2 round3 parallel_train run_ensemble test_augment; do
  echo "=== $f ==="
  rg "ml\.${f}|from ml import ${f}|from ml\.${f}" --type py -g '!__pycache__/' || echo 'no refs'
done
```

- [ ] **Step 2: If clean, create experiments directory and move**

```bash
mkdir -p experiments/ml
# 已执行：以上实验文件均被删除（未保留在 experiments/ml/）
git mv ml/round2.py experiments/ml/
git mv ml/round3.py experiments/ml/
git mv ml/parallel_train.py experiments/ml/
git mv ml/run_ensemble.py experiments/ml/
git mv ml/test_augment.py experiments/ml/
```

- [ ] **Step 3: Also move ml/legacy/**（已删除）

```bash
# Check if anything imports ml.legacy:
rg 'ml\.legacy|from ml import legacy|from ml\.legacy' --type py -g '!__pycache__/' || echo 'clean'
git mv ml/legacy/ experiments/ml/legacy/
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest -q tests/test_weights.py tests/test_training.py 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: archive experimental ML files to experiments/ml/"
```

### Task 1.3: Move experimental commands to experiments/

**Files:** 13 files in commands/

- [ ] **Step 1: Create experiments/commands/ and move**

```bash
mkdir -p experiments/commands
# List of non-production files:
for f in sweep_final.py sweep_final2.py sweep_final3.py sweep_search_strategy.py \
         sweep_search_strategy2.py analyze_5combos.py analyze_bfs_boundary.py \
         analyze_schemes.py compare_grid_vs_random.py compare_pants_algorithms.py \
         diagnose_bfs_failures.py tune_bfs.py tune_bfs_extended.py; do
  git mv "commands/$f" experiments/commands/
done
```

- [ ] **Step 2: Check for cross-references among moved files**

```bash
# Dependencies among moved files — if they import each other,
# their relative imports still work from experiments/:
for f in experiments/commands/*.py; do
  rg "from tools\.|import tools\." "$f" && echo "NEEDS FIX: $f"
done
# Should be clean since tools.* → commands.* was tools→commands migration
```

- [ ] **Step 3: Also move run_margin_full.py if experimental**

```bash
# Check if anything imports from run_margin_full:
rg 'run_margin_full' --type py -g '!__pycache__/' || echo 'clean'
# Optionally move:
git mv commands/run_margin_full.py experiments/commands/
```

- [ ] **Step 4: Verify commands/ import integrity**

```bash
# All production commands/ imports resolved:
python -c "from commands.batch_process_all import main" 2>&1 || echo "expected: no main in batch (has script entry)"
python -c "from commands.cli_common import app_cli" 2>&1 || echo "cli_common ok"
python -c "from commands.export import main" 2>&1 || echo "export ok"
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move experimental commands to experiments/commands/
- 13 sweep/tune/analyze/diagnose/compare files relocated
- commands/ now contains only production entry points"
```

---

## Phase 2: Pipeline Honest + FEATURE_SCHEME Fix

### Task 2.1: Fix FEATURE_SCHEME hardcoded constant

**Files:**
- Modify: `pipeline/run.py:14` — remove module-level FEATURE_SCHEME constant
- Modify: `pipeline/run.py:92,139,145,153` — use `params.feature_scheme` or scheme name

- [ ] **Step 1: Remove the module-level constant**

```python
# In pipeline/run.py, line 14:
# Change from:
FEATURE_SCHEME = "morph_region_ci_37d"
# To:
_DEFAULT_FEATURE_SCHEME = "morph_region_ci_37d"  # fallback only when no param given
```

- [ ] **Step 2: Fix _save_results to not hardcode constant**

```python
# _save_results signature:
# Add param: feature_scheme: str = "unknown" (already exists, check usage)
# line 92: remove reference to FEATURE_SCHEME constant
```

- [ ] **Step 3: Fix _run_train to always use params.feature_scheme**

```python
# line 139: change:
#   feat_scheme_name = params.get("feature_scheme", FEATURE_SCHEME)
# to:
#   feat_scheme_name = params.get("feature_scheme", _DEFAULT_FEATURE_SCHEME)

# line 145: change
#   feature_names=getattr(SCHEME_REGISTRY[FEATURE_SCHEME], ...)
# to:
#   feature_names = scheme_data.get("feature_names", [])
#   (load it from the same scheme_data, not a different constant)

# line 153: change
#   existing = RESULTS_DIR / FEATURE_SCHEME / f"{train_scheme}-..."
# to:
#   existing = RESULTS_DIR / feat_scheme_name / f"{train_scheme}-..."
```

- [ ] **Step 4: Run integration smoke test**

```bash
# Just test that train e2e can start (without real data, we test code path):
python -c "from pipeline.run import _run_train; print('import ok')"
pytest -q tests/test_training.py -x 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/run.py
git commit -m "fix: remove FEATURE_SCHEME hardcoded constant in pipeline/run.py
- Use params.feature_scheme consistently for result paths
- Break dependency on module-level constant overriding real scheme name"
```

### Task 2.2: Honest pipeline/run steps — roi and feature_eng

**Files:**
- Modify: `pipeline/run.py` — steps dict in DEFAULT_CONFIG / pipeline/config.py
- Modify: `ais-cli.py` — remove `--step roi` from defaults; document
- Modify: `pipeline/config.py` — remove roi from default steps

- [ ] **Step 1: Change default config to exclude roi**

```python
# In pipeline/config.py DEFAULT_CONFIG, remove roi from steps:
DEFAULT_CONFIG = {
    "steps": [
        # {"name": "roi", "algo": "bfs"},  # REMOVED — roi is a draft step, not a production step
        {"name": "feature_eng", "scheme": "morph_region_ci_37d"},
        {"name": "train", ...},
    ]
}
```

- [ ] **Step 2: Update ais-cli _list_steps to mark roi as prelabel-only**

```python
# In ais-cli.py _list_steps():
print("  roi  — ROI/decloth prelabeling (preview; not suitable for production)")
print("         生产路径：commands/batch_process_all.py → labeling 人工修正")
```

- [ ] **Step 3: Keep roi as optional step (don't remove code), just mark it**

The `_run_roi` can stay as-is for experimentation purposes, but the default config no longer includes it.

- [ ] **Step 4: Commit**

```bash
git add pipeline/config.py ais-cli.py
git commit -m "refactor: remove roi from default pipeline steps, mark prelabel-only
- ais-cli --step train is now the default production path
- roi retained as optional for experimentation"
```

### Task 2.3: Move DEPRECATED pipeline v1 to legacy/

**Files:**
- Move: `pipeline/core.py`, `pipeline/steps.py`, `pipeline/cache.py`, `pipeline/feature_pipeline.py`
- Keep: `pipeline/run.py`, `pipeline/config.py`, `pipeline/contracts.py`, `pipeline/predict/__init__.py`

- [ ] **Step 1: Create legacy target and move**

```bash
mkdir -p legacy/pipeline_v1
git mv pipeline/core.py legacy/pipeline_v1/
git mv pipeline/steps.py legacy/pipeline_v1/
git mv pipeline/cache.py legacy/pipeline_v1/
git mv pipeline/feature_pipeline.py legacy/pipeline_v1/
```

- [ ] **Step 2: Add header comment to moved files**

```python
# In each moved file, prepend or update:
# DEPRECATED: per-subject geometry pipeline. Superseded by pipeline/run.py + contracts.py.
# Moved to legacy/ 2026-07-28. No functional changes allowed.
# Only kept for reference; tests still cover these paths.
```

- [ ] **Step 3: Update pipeline/__init__.py exports**

```python
# Remove re-exports of Pipeline, StepCache etc.
# Only export contracts + run:
from pipeline.run import run
from pipeline.contracts import FeatureSet, TrainingConfig, TrainingResult
```

- [ ] **Step 4: Fix test imports temporarily (to be cleaned in Phase 6)**

Keep the test imports working by adjusting sys.path or keeping a compat note — but better to rewrite in Phase 6. For now, add a compat redirect:

```python
# In pipeline/__init__.py, add temporary forwarding:
# import warnings
# warnings.warn("pipeline.core/steps/cache are deprecated in legacy/pipeline_v1/")
```

Actually — the project rule says **no deprecation stubs**. So the tests must be rewritten in Phase 6 before we can remove the files. Plan boundary:

**Priority: Phase 2.3 moves the v1 pipeline to legacy/.** Then Phase 6 rewrites the tests that depend on them. If tests break temporarily, mark with @pytest.mark.legacy and skip by default.

- [ ] **Step 5: Mark legacy tests**

```bash
# In test_steps.py, test_pipeline_core.py, test_cache.py, test_integration.py:
# Add at file level:
import pytest
pytestmark = pytest.mark.skip(reason="legacy pipeline v1 — kept for reference, not part of main CI")
```

- [ ] **Step 6: Run remaining tests to confirm no cascade**

```bash
pytest -q tests/test_training.py tests/test_weights.py tests/test_ml_pipeline.py 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: move deprecated pipeline v1 to legacy/pipeline_v1/

- pipeline/{core,steps,cache,feature_pipeline}.py → legacy/pipeline_v1/
- Tests marked with pytest.mark.legacy (skip by default)
- pipeline/__init__.py now exports only run() + contracts"
```

### Task 2.4: Remove pipeline.py from old entry point

**Files:**
- Modify: `commands/run_pipeline.py` — update docstring if needed (already points to `parameterization.pipeline.run_pipeline`)

This should already be correct since `commands/run_pipeline.py` only calls `parameterization.pipeline.run_pipeline`.

- [ ] **Step 1: Verify parameterization CLI still works**

```bash
python -c "from commands.run_pipeline import main; print('import ok')"
```

- [ ] **Step 2: Commit if any changes**

```bash
git add -A && git commit -m "chore: verify run_pipeline CLI points to parameterization" || echo "no changes needed"
```

### Task 2.5: Merge roi/registry.py into mesh/roi (keep one)

**Files:**
- Keep: `roi/registry.py` (it's the one `pipeline/run.py` imports)
- Should `pipeline/run` import directly from `mesh.roi`? Audit usage.

- [ ] **Step 1: Check who imports roi.registry vs mesh.roi directly**

`rg -n 'roi\.registry|from roi' --type py -g '!__pycache__/'`

pipeline/run.py uses `from roi.registry import get` — this is the thin wrapper.

Decision: keep `roi/registry.py` as the public front (it's already the one used by pipeline/run). Don't add a second ref from mesh.roi.

- [ ] **Step 2: Verify no code imports the now-deleted mesh/roi/registry.py**

Already done in Task 1.1.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: roi registry consolidated in roi/registry.py (task 2.5)"
```

---

## Phase 3: Scheme Single-Track

**Risk: high** — scheme load results affect training metrics.

### Task 3.1: Extract _make_data_dict from features/selectors/schemes.py to utils

**Files:**
- Modify: `features/selectors/schemes.py` — remove `_make_data_dict` and `_to_float` and `_load_parquet_or_csv`
- Create: `features/selectors/_utils.py` or keep in a shared utility
- Modify: `features/selectors/scheme_*.py` — update their import

Current deps:
- `features/selectors/scheme_morph_region_ci_36d.py` imports `from features.selectors.schemes import _make_data_dict`
- `features/selectors/scheme_anova_only.py` imports same
- etc.

- [ ] **Step 1: Create _utils module in features/selectors/**

```python
# features/selectors/_utils.py
"""Shared utilities for feature scheme loaders (ported from features/selectors/schemes.py)."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_DIR = Path("results/features")


def load_parquet_or_csv(path_stem: str) -> pd.DataFrame:
    ...


def to_float(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    ...


def make_data_dict(
    y: np.ndarray,
    X_basic: np.ndarray | None = None,
    X_morph: np.ndarray | None = None,
    X_region: np.ndarray | None = None,
    X_ci: np.ndarray | None = None,
) -> dict:
    ...
```

Copy the exact implementation from features/selectors/schemes.py lines 80-142.

- [ ] **Step 2: Update all scheme_*.py imports**

```python
# Change:
# from features.selectors.schemes import _make_data_dict
# To:
from features.selectors._utils import make_data_dict
```

Files to modify: `scheme_morph_region_ci_36d.py`, `scheme_anova_only.py`, `scheme_morph_region_ci_35d.py`, `scheme_normal_enhanced.py`, `features/selectors/schemes.py`

- [ ] **Step 3: Remove copied functions from features/selectors/schemes.py**

Delete `_load_parquet_or_csv`, `_to_float`, `_make_data_dict` from features/selectors/schemes.py. Leave `FeatureScheme` class and `REGISTRY` intact.

- [ ] **Step 4: Run regression test — load each scheme, compare X shape**

```python
# tests/test_scheme_regression.py (temporary)
from features.selectors.schemes import SELECTION_REGISTRY as ML_REGISTRY
from features.selectors import SELECTION_REGISTRY
for name, scheme in ML_REGISTRY.items():
    try:
        d = scheme.load()
        print(f"{name}: X={d.get('X_basic', d.get('X')).shape}")
    except Exception as e:
        print(f"{name}: FAIL {e}")
```

- [ ] **Step 5: Commit**

```bash
git add features/selectors/_utils.py
git add features/selectors/scheme_*.py features/selectors/schemes.py
git add features/selectors/schemes.py
git commit -m "refactor: extract _make_data_dict from ml/schemes to features/selectors/_utils
- Break circular dependency: selectors no longer import from features.selectors.schemes
- Pure code move — zero logic change"
```

### Task 3.2: Move feature scheme REGISTRY to features/selectors/

**Files:**
- Modify: `features/selectors/schemes.py` — absorb scheme definitions
- Modify: `features/selectors/schemes.py` — thin re-export or remove
- Modify: all callers of `features.selectors.schemes.SELECTION_REGISTRY`

- [ ] **Step 1: Identify all callers of features.selectors.schemes.SELECTION_REGISTRY**

```bash
rg 'ml\.schemes\.REGISTRY|from ml\.schemes import.*REGISTRY|SCHEME_REGISTRY' --type py -g '!__pycache__/' -g '!tests/'
```

- [ ] **Step 2: Add a public `load_feature_set` function in features/selectors/__init__.py**

```python
# features/selectors/__init__.py
from features.selectors.schemes import SELECTION_REGISTRY
from pipeline.contracts import FeatureSet


def load_feature_set(name: str) -> FeatureSet:
    """Load a named feature scheme as a training-ready FeatureSet."""
    if name not in SELECTION_REGISTRY:
        raise KeyError(f"Unknown feature scheme: {name}")
    data = SELECTION_REGISTRY[name].load()
    # ... convert to FeatureSet
    return FeatureSet(
        name=name,
        y=data["y"],
        X=data.get("X_basic", data.get("X")),
        feature_names=data.get("feature_names", []),
    )
```

- [ ] **Step 3: In features/selectors/schemes.py, keep REGISTRY as thin re-export only**

```python
# features/selectors/schemes.py — DEPRECATED re-export layer
import warnings
warnings.warn(
    "Import FeatureScheme from features.selectors, not features.selectors.schemes",
    DeprecationWarning, stacklevel=2,
)
from features.selectors.schemes import SELECTION_REGISTRY as REGISTRY, FeatureScheme
```

But the project rule says **no deprecation stubs** — so we must update callers directly. Update:

- `pipeline/run.py`: change `from features.selectors.schemes import SELECTION_REGISTRY` → `from features.selectors import SELECTION_REGISTRY as REGISTRY`
- `modeling/train.py`: same
- `commands/sweep_*.py`: if moved to experiments/ in Phase 1, they keep old import (not production)

- [ ] **Step 4: Fix all production callers**

```python
# pipeline/run.py line 112:
# Change:
#   from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
# To:
from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
```

```python
# modeling/train.py line 25:
# Change:
#   from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
# To:
from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
```

- [ ] **Step 5: Remove FeatureScheme and REGISTRY from features/selectors/schemes.py entirely**

If all callers have been migrated, the file becomes only the training presets (which should also migrate — Task 3.3).

- [ ] **Step 6: Run full regression**

```bash
# Load every scheme and verify shape:
python -c "
from features.selectors.schemes import SELECTION_REGISTRY
for name, s in SELECTION_REGISTRY.items():
    try:
        d = s.load()
        print(f'OK {name}: {d[\"y\"].shape}')
    except Exception as e:
        print(f'FAIL {name}: {e}')
"
```

- [ ] **Step 7: Commit**

```bash
git add features/selectors/ pipeline/run.py modeling/train.py features/selectors/schemes.py
git commit -m "refactor: move feature scheme REGISTRY to features/selectors/schemes
- Production callers updated: pipeline/run.py, modeling/train.py
- features/selectors/schemes.py reduced, pending presets migration (task 3.3)"
```

### Task 3.3: Migrate get_training_preset from ml.schemes（原训练方案区，现 modeling.training.presets）

**Files:**
- Modify: `modeling/train.py` — fix import
- Modify: `modeling/training/schemes.py` — remove `TRAINING_PRESETS` and `get_training_preset`
- Already: `modeling/training/presets.py` — is the source of truth

- [ ] **Step 1: Find callers of modeling.training.presets.get_training_preset**

```bash
rg 'get_training_preset|TRAINING_PRESETS' --type py -g '!__pycache__/' | grep -v training/presets
```

- [ ] **Step 2: Fix modeling/train.py**

```python
# modeling/train.py line 113:
# Change:
#   from modeling.training.presets import get_training_preset
# To:
from modeling.training.presets import get_training_preset
```

- [ ] **Step 3: Remove from modeling/training/schemes.py**

Delete `TRAINING_PRESETS` dict and `get_training_preset` function from modeling/training/schemes.py.

- [ ] **Step 4: Verify no broken imports**

```bash
python -c "from modeling.training.presets import get_training_preset, TRAINING_PRESETS; print('OK')"
python -c "import modeling.train; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add modeling/train.py modeling/training/schemes.py
git commit -m "refactor: migrate get_training_preset to modeling/training/presets
- ml/schemes no longer carries presets or training configs
- modeling.train imports from modeling.training.presets directly"
```

### Task 3.4: Remove _load_dual_ci* from pipeline/run.py

**Files:**
- Modify: `pipeline/run.py` — delete `_load_dual_ci`, `_load_dual_ci_ai` methods

- [ ] **Step 1: Check if these are still called**

```bash
rg '_load_dual_ci' pipeline/run.py
# line 188: def _load_dual_ci() — find all callers
rg '_load_dual_ci' --type py -g '!__pycache__/'
```

If no callers remain (they were legacy feature-engineering-in-pipeline), delete them.

- [ ] **Step 2: Delete functions**

Remove `_load_dual_ci()` (line 188-end) and `_load_dual_ci_ai()` from pipeline/run.py.

- [ ] **Step 3: Commit**

```bash
git add pipeline/run.py
git commit -m "refactor: remove _load_dual_ci* from pipeline/run.py
Feature selection belongs in features/selectors/, not in the pipeline runner"
```

---

## Phase 4: Large File Splitting (Parallel)

### Task 4.1: Split landmark_regions.py (~965 lines)

**Files:**
- No delete — just `landmark_regions.py` is 965 lines. Needs splitting into package.

The file defines UV region classification and region-level feature computation. Split into:
- `features/extractors/asymmetry/_regions.py` — region definitions (225 regions, landmark mapping)
- `features/extractors/asymmetry/_features.py` — region feature computation
- `features/extractors/asymmetry/landmark_regions.py` — re-export public API

This is a large, pure-refactor task. Scope: Split into two sub-steps if budget constrained.

- [ ] Skip if budget doesn't permit — mark as post-cleanup follow-up.

### Task 4.2: Split _cut_analysis.py (~641 lines)

Same approach as 4.1. Lower priority. Skip if budget limited.

---

## Phase 5: Commands Layering + Docs Truth

### Task 5.1: Reorganize file_manager/

**Files:**
- Create: `scripts/data_ops/`
- Move: `file_manager/*.py`

- [ ] **Step 1: Create scripts/ directory and move**

```bash
mkdir -p scripts
git mv file_manager/ scripts/data_ops/
```

- [ ] **Step 2: Verify zero core imports**

```bash
rg 'file_manager' --type py -g '!__pycache__/' -g '!scripts/' || echo 'clean'
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: move file_manager/ to scripts/data_ops/ — not part of core"
```

### Task 5.2: Document truth — update README and PROJECT.md

**Files:**
- Modify: `README.md` — table of contents, module map, pipeline description

Refer to the changes already made in `.claude/PROJECT.md` and `.claude/docs/gt-annotation.md` (commit d3c7f7b).

- [ ] **Step 1: Update README.md directory listing**

```markdown
## 目录结构

```
commands/                  CLI 入口（batch/plot/evaluate/export）
features/                 特征提取与选择
  extractors/               basic / morphology / asymmetry
  selectors/                特征方案筛选
labeling/                 标注平台（FastAPI + React）
landmarks/                解剖特征点检测（6 类）
mesh/                     网格处理（ROI、曲率、清理）
modeling/                  训练管线
  models/                   模型
  training/                 训练策略、加权、HP 搜索
moire/                    数字 Moiré（论文 M2；不在主训练路径）
parameterization/         调和 UV 参数化
pipeline/                 编排 + 数据契约
reports/                  Streamlit 报告
utils/                    工具函数
visualization/            matplotlib 渲染
experiments/              扫参与一次性分析（非生产）
scripts/                  数据整理脚本（非核心）
```

- [ ] **Step 2: Update pipeline table with production truth**

```markdown
## 生产路径

⚠️ 自动 ROI（去衣/去裤）与 landmark 目前仅作预标，**不是最终 GT**。

```
mesh/ROI/landmarks 预标 → labeling 人工修正 → data/ground_truth/ 导出
  → 参数化 → 特征提取 → results/features_extraction/back_v1/ CSV
  → scheme 训练 → results/prediction/
  → reports / export
```
```

- [ ] **Step 3: Remove stale references**

Check for: `analysis/`, `curvature/`, `ml_models/`, `run_pipeline.py —subject`, `tools.`

```bash
rg -n 'analysis/|curvature/|ml_models|tools\.|run_pipeline.py --subject' README.md
# Fix each reference.
```

- [ ] **Step 4: Mark PRD as historical**

Add a banner at top of `docs/PRD.md`:

```markdown
> **⚠️ HISTORICAL DOCUMENT** — Last updated 2026-05. Directory structure
> and pipeline design have changed significantly. For current state see
> `docs/refactor/2026-07-28-surgical-cleanup-refactor-design.md` and `README.md`.
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/PRD.md
git commit -m "docs: align README with current directory structure and production path"
```

### Task 5.3: Update .claude/MEMORY.md cross-refs

**Files:**
- Modify: `.claude/MEMORY.md`

Check for stale refs to tools.*, old module paths.

- [ ] **Step 1: Scan memory entries**

```bash
rg 'tools\.|ml_models|curvature/|analysis/' .claude/memory/ --type md
# Update any stale paths found.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/ && git commit -m "docs: update stale memory cross-refs"
```

---

## Phase 6: Test Alignment

### Task 6.1: Rewrite legacy pipeline tests to not depend on ALL_STEPS

**Files:**
- Modify: `tests/test_steps.py`, `tests/test_pipeline_core.py`, `tests/test_cache.py`, `tests/test_integration.py`
- If moved to legacy: these test files should either be rewritten or moved to `tests/legacy/`

- [ ] **Step 1: Create tests/legacy/ directory**

```bash
mkdir -p tests/legacy
```

- [ ] **Step 2: Move legacy pipeline tests**

```bash
git mv tests/test_steps.py tests/legacy/
git mv tests/test_pipeline_core.py tests/legacy/
git mv tests/test_cache.py tests/legacy/
git mv tests/test_feature_pipeline.py tests/legacy/
```

- [ ] **Step 3: Add pytest.ini marker for legacy tests**

```ini
# pyproject.toml or pytest.ini, add:
[tool.pytest.ini_options]
markers = [
    "legacy: Tests for deprecated pipeline v1 — not run in default CI",
]
```

- [ ] **Step 4: Mark legacy tests directory excluded from default run**

```bash
# In pyproject.toml or pytest config:
# pytest --ignore=tests/legacy
```

- [ ] **Step 5: Run main test suite**

```bash
pytest -q --ignore=tests/legacy -x 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add tests/ pyproject.toml
git commit -m "refactor: move legacy pipeline tests to tests/legacy/
- Excluded from default pytest run (--ignore=tests/legacy)
- Main suite now tests only active code paths"
```

### Task 6.2: Add smoke tests for current pipeline/run and scheme loading

**Files:**
- Create: `tests/test_scheme_load.py` — quick smoke for scheme loading
- Create: `tests/test_run_import.py` — verify pipeline/run imports resolve

- [ ] **Step 1: Write scheme loading smoke test**

```python
# tests/test_scheme_load.py
"""Verify feature schemes can be loaded without errors (shape + no crash)."""
import pytest
from features.selectors.schemes import SELECTION_REGISTRY


class TestFeatureSchemes:
    """Smoke tests: load each scheme, check basic structure."""

    @pytest.mark.parametrize("name", list(SELECTION_REGISTRY.keys()))
    def test_load(self, name):
        """Loading a scheme returns dict with y, feature_names."""
        scheme = SELECTION_REGISTRY[name]
        assert scheme.name == name
        assert "load" in dir(scheme)
```

- [ ] **Step 2: Write pipeline/run import smoke test**

```python
# tests/test_run_import.py
"""Verify pipeline/run imports resolve after refactoring."""
from pipeline.run import run, _run_train, _save_results


class TestRunImports:
    def test_run_function_exists(self):
        assert callable(run)

    def test_save_results_exists(self):
        assert callable(_save_results)
```

- [ ] **Step 3: Run new tests**

```bash
pytest -q tests/test_scheme_load.py tests/test_run_import.py -v 2>&1 | tail -20
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_scheme_load.py tests/test_run_import.py
git commit -m "test: add smoke tests for scheme loading and pipeline/run imports"
```

---

## Execution Order

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 5 ──► Phase 6
                                                  │
                                                  └── Phase 4 (parallel optional)
                          │
                          └── All are sequential on the common path;
                              only Phase 4 (split) can be skipped or deferred.
                              Phases 1.x and 5.x are parallel internal.
```

Per spec: no changes to algorithm code, zero new dependencies, no deprecation stubs.

---

## Success Criteria

```
pytest -q                  → green (excluding tests/legacy)
ruff check changed files   → zero errors
morph_region_ci_40d + GBRT load    → X shape unchanged, Macro-F1 Δ ≤ 0.01
data/ground_truth/ layout  → unmodified
rg 'roi_old|mesh\.roi_old' → empty
rg 'tools\.' --type py     → empty (except .venv/)
```
