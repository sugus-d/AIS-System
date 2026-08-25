"""Ground Truth 全量校验。对指定 subject 的 GT 坐标跑全部 landmark 的检查项。

Usage:
    python landmarks/gt_validate.py --subject S0006
    python landmarks/gt_validate.py --subject S0006 --landmark neck_root
"""

import argparse
from collections.abc import Callable

import numpy as np

from landmarks._validate_utils import load_gt, load_mesh
from landmarks.axilla.validate import validate as validate_axilla
from landmarks.lateral_profile import extract_split_contours
from landmarks.neck_root.validate import validate as validate_neck_root
from landmarks.scapular_peak.validate import validate as validate_scapular_peak
from landmarks.shoulder_transition.validate import validate as validate_shoulder_transition
from landmarks.waist.validate import validate as validate_waist

# validator 签名: (gt, features, vertices, left_c, right_c, relaxed) -> list[dict]
_ValidatorFn = Callable[..., list[dict]]

_VALIDATORS: dict[str, _ValidatorFn] = {
    "neck_root": validate_neck_root,
    "shoulder_transition": validate_shoulder_transition,
    "axilla": validate_axilla,
    "waist": validate_waist,
    "scapular_peaks": validate_scapular_peak,
}


def validate_all(subject: str) -> dict[str, list[dict]]:
    """运行全部校验项，返回 {landmark_name: [issues]}。"""
    gt: dict = load_gt(subject)
    if not gt:
        return {}
    features: dict = dict(gt.get("_features", {}))
    features["_subject"] = subject
    relaxed: bool = bool(features.get("arms") == "none" or features.get("body_asymmetry", False))

    vertices: np.ndarray = load_mesh(subject)
    left_c: np.ndarray
    right_c: np.ndarray
    left_c, right_c = extract_split_contours(vertices)

    results: dict[str, list[dict]] = {}
    for lmk_name, validator in _VALIDATORS.items():
        if lmk_name not in gt:
            results[lmk_name] = []
            continue
        results[lmk_name] = validator(gt, features, vertices, left_c, right_c, relaxed)

    return results


def validate(subject: str, landmark: str | None = None) -> None:
    """打印指定 subject 的校验报告。"""
    print(f"\n{'=' * 65}")
    print(f"GT Validation — {subject}")
    gt: dict = load_gt(subject)
    features: dict = gt.get("_features", {})
    print(f"  arms={features.get('arms', '?')}  asym={features.get('body_asymmetry', False)}")
    print(f"{'=' * 65}")

    all_results: dict[str, list[dict]] = validate_all(subject)
    total_issues: int = 0
    for lmk_name, issues in all_results.items():
        if landmark and lmk_name != landmark:
            continue
        n: int = len(issues)
        total_issues += n
        status: str = "PASS" if n == 0 else f"{n} ISSUE(S)"
        print(f"\n  {lmk_name}: {status}")
        for iss in issues:
            print(f"    [{iss['tag']}] {iss['detail']}")

    print(f"\n  {'─' * 40}")
    print(f"  Total: {total_issues} issue(s)")


def main() -> None:
    """CLI 入口：解析 --subject 和可选的 --landmark 参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--landmark", default=None)
    args = parser.parse_args()
    validate(args.subject, args.landmark)


if __name__ == "__main__":
    main()
