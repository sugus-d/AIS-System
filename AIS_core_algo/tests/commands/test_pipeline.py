"""Verify pipeline/run imports resolve after refactoring."""
from commands.pipeline import _run_train, run
from modeling.training.result_paths import find_existing_metrics, save_results


class TestRunImports:
    def test_run_function_exists(self):
        assert callable(run)

    def test_save_results_shared_in_result_paths(self):
        # _save_results 已迁至 modeling.training.result_paths（两训练入口共享）
        assert callable(save_results)
        assert callable(find_existing_metrics)

    def test_run_train_importable(self):
        assert callable(_run_train)
