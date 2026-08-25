"""Tests for run_pipeline.py CLI argument parsing.

本测试验证 CLI 参数解析（--list-steps、--help、--subject 等）行为是否符合预期。
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# 原 run_pipeline.py 已由根目录 ais-cli.py 取代（--list-steps/--help/--subject）。
# 文件名含连字符无法直接 import，通过文件路径加载。
# 注意：测试已按领域分组，从 tests/misc/ 上溯两级到项目根。
_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("ais_cli", _ROOT / "ais-cli.py")
_AIS_CLI = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_AIS_CLI)
main = _AIS_CLI.main


def test_list_steps():
    """--list-steps should print all step names without error."""
    test_args = ["run_pipeline.py", "--list-steps"]
    with patch.object(sys, "argv", test_args):
        try:
            main()
        except SystemExit as e:
            pytest.fail(f"main() raised SystemExit: {e}")


def test_help():
    """--help should exit with code 0 after printing."""
    test_args = ["run_pipeline.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0


def test_invalid_subject_shows_error():
    """Running with --subject that doesn't exist should raise ValueError."""
    test_args = ["run_pipeline.py", "--subject", "nonexistent"]
    with patch.object(sys, "argv", test_args), pytest.raises((ValueError, SystemExit)):
        main()
