"""logger 单元测试 — 行为测试，不测内部实现。"""

from __future__ import annotations

import logging

from utils.logger import add_file_handler, AISLogger, logger


class TestLoggerInstance:
    def test_logger_is_ais_logger(self) -> None:
        assert isinstance(logger, AISLogger)

    def test_info_calls_without_error(self) -> None:
        logger.info("info test message")

    def test_warning_calls_without_error(self) -> None:
        logger.warning("warning test message")

    def test_error_calls_without_error(self) -> None:
        logger.error("error test message")


class TestAddFileHandler:
    def test_returns_handler(self) -> None:
        handler = add_file_handler("test_logger_unit")
        assert isinstance(handler, logging.Handler)
        logger.remove(handler)

    def test_repeated_register_returns_same(self) -> None:
        h1 = add_file_handler("test_logger_repeat")
        h2 = add_file_handler("test_logger_repeat")
        assert h1 is h2
        logger.remove(h1)


class TestRemove:
    def test_remove_none_clears_non_console(self) -> None:
        """remove(None) 应清除非 console handler。"""
        ais_logger = logging.getLogger("ais")
        # 先注册一个文件 handler
        h = add_file_handler("test_logger_remove")
        assert h in ais_logger.handlers
        logger.remove(None)
        assert h not in ais_logger.handlers

    def test_remove_none_closes_file_handlers(self) -> None:
        """remove(None) 应关闭文件 handler（消除 ResourceWarning）。"""
        h = add_file_handler("test_logger_remove_close")
        logger.remove(None)
        # RotatingFileHandler.close() 会置 stream=None
        assert h.stream is None

    def test_remove_handler_closes_single(self) -> None:
        """remove(handler) 应关闭指定文件 handler。"""
        h = add_file_handler("test_logger_remove_single")
        logger.remove(h)
        assert h.stream is None
        assert h not in logging.getLogger("ais").handlers


class TestThreadSafety:
    """多线程行为验证 — 每线程独立 _tls.caller，per-module 文件过滤不丢日志。

    `AISLogger._log` 每次调用都会经 `_ensure_file_handler` 重设当前线程的
    `_tls.caller`，因此 worker 线程的日志能正确路由到自身模块的文件 handler。
    """

    def test_worker_thread_logs_without_error(self) -> None:
        import threading

        errors: list[BaseException] = []

        def worker() -> None:
            try:
                logger.info("worker thread log message")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive()
        assert errors == []

    def test_module_filter_respects_current_thread_caller(self) -> None:
        """同一线程内 filter 只放行本模块记录，其他模块名被拒。"""
        from utils.logger import _ensure_file_handler, _ModuleFilter, _tls

        _ensure_file_handler()
        current = _tls.caller
        assert _ModuleFilter(current).filter(logging.LogRecord("x", logging.INFO, "", 1, "m", (), None))
        assert not _ModuleFilter("some_other_module").filter(
            logging.LogRecord("x", logging.INFO, "", 1, "m", (), None)
        )
