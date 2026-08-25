"""统一日志 — logging + RichHandler，loguru 风格的格式。

用法:
    from utils.logger import logger
    logger.info("消息")
    logger.warning("警告")
    logger.error("错误")
    logger.success("完成")

特点:
    - 控制台: RichHandler 彩色输出
    - 文件:   RotatingFileHandler 自动轮转 (10MB)，logs/<module>.log
    - 自动:   每个模块自动创建自己的文件 handler，不收其他模块的日志
"""

from __future__ import annotations

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_FILE_HANDLERS: dict[str, logging.Handler] = {}
_FILE_HANDLER_LOCK = threading.Lock()

# ── 日志格式（loguru 兼容） ──
FILE_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
FILE_DATE = "%Y-%m-%d %H:%M:%S"

# ── 线程局部变量：记住当前调用栈是谁 ──
_tls = threading.local()
_tls.caller = "unknown"


class _ModuleFilter(logging.Filter):
    """按模块名过滤日志记录：只收来自指定模块的消息。

    设计说明（2026-08-06 验证）：
    ``_tls.caller`` 在每次 ``AISLogger._log`` 调用前由 ``_ensure_file_handler``
    重设（非仅首次），且 ``_tls`` 为 thread-local——每个线程持有独立副本，
    因此多线程下各线程日志仍能正确路由到自身模块的文件 handler
    （见 tests/utils/test_logger.py::TestThreadSafety）。
    """

    def __init__(self, module_name: str) -> None:
        super().__init__()
        self.module_name = module_name

    def filter(self, _record: logging.LogRecord) -> bool:
        return getattr(_tls, "caller", None) == self.module_name


class _FileFormatter(logging.Formatter):
    """文件专用 — 去掉 Rich markup 标记。"""

    def format(self, record: logging.LogRecord) -> str:
        import re
        msg = record.getMessage()
        record.msg = re.sub(r"\[/?\w+\]", "", msg)
        return super().format(record)


# ── 根 Logger（仅控制台） ──
_logger = logging.getLogger("ais")
_logger.setLevel(logging.DEBUG)
_logger.handlers.clear()

_console = RichHandler(
    console=Console(stderr=True),
    rich_tracebacks=True,
    tracebacks_show_locals=True,
    show_time=True,
    show_path=True,
    show_level=True,
    omit_repeated_times=False,
    markup=True,
)
_console.setLevel(logging.DEBUG)
_logger.addHandler(_console)


def _caller_stem() -> str:
    """获取调用者文件名的 stem。"""
    import inspect
    frame = inspect.currentframe()
    try:
        f = frame
        while f is not None:
            name = Path(f.f_code.co_filename).stem
            if name not in ("logger", "threading"):
                return name
            f = f.f_back
    finally:
        del frame
    return "unknown"


def _ensure_file_handler() -> None:
    """为当前调用线程的模块注册文件 handler（仅首次 + 不重复注册）。"""
    caller = _caller_stem()
    _tls.caller = caller
    if caller == "unknown":
        return
    with _FILE_HANDLER_LOCK:
        if caller in _FILE_HANDLERS:
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"{caller}.log")
        fh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_FileFormatter(FILE_FORMAT, datefmt=FILE_DATE))
        fh.addFilter(_ModuleFilter(caller))
        _logger.addHandler(fh)
        _FILE_HANDLERS[caller] = fh


# ── Logger 接口 ──
class AISLogger:
    """日志接口，兼容 loguru 风格。"""

    def _log(self, level: int, msg: str) -> None:
        """底层日志写入。同时写入控制台、文件 handler、会话日志。"""
        _ensure_file_handler()
        _logger.log(level, msg)
        if hasattr(self, "_session_fh") and self._session_fh:
            self._write_session(msg)

    def debug(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._log(logging.DEBUG, str(msg))

    def info(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._log(logging.INFO, str(msg))

    def warning(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._log(logging.WARNING, str(msg))

    def error(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._log(logging.ERROR, str(msg))

    def success(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._log(logging.INFO, f"✓ {msg}")

    def add(self, *args: object, **kwargs: object) -> logging.Handler | None:
        sink = args[0] if args else kwargs.get("sink")
        if isinstance(sink, str):
            fh = RotatingFileHandler(sink, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(_FileFormatter(FILE_FORMAT, datefmt=FILE_DATE))
            _logger.addHandler(fh)
            return fh
        return None

    def remove(self, handler: logging.Handler | None = None) -> None:
        if handler is None:
            for h in list(_logger.handlers):
                if h is not _console:
                    _logger.removeHandler(h)
            _FILE_HANDLERS.clear()
        else:
            _logger.removeHandler(handler)

    def begin_session(self, name: str) -> str:
        """开始训练会话日志。直接写文件，绕过 logging handler 的复杂路由。"""
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = f"{ts}_{name}"
        log_dir = os.path.join(LOG_DIR, "train")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{session}.log")
        # 句柄需在会话期间保持打开，由 end_session() 关闭
        self._session_fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
        self._session_path = log_path
        self._write_session(f"=== 训练会话开始: {name} ===")
        return log_path

    def _write_session(self, msg: str) -> None:
        """写入会话日志行。"""
        if hasattr(self, "_session_fh") and self._session_fh:
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
            import re
            clean = re.sub(r"\[/?\w+\]", "", msg)
            line = f"{ts} | SESSION | {clean}\n"
            self._session_fh.write(line)
            self._session_fh.flush()

    def end_session(self) -> None:
        """结束训练会话日志。"""
        if hasattr(self, "_session_fh") and self._session_fh:
            self._write_session("=== 训练会话结束 ===")
            self._session_fh.close()
            self._session_fh = None


logger: AISLogger = AISLogger()


def add_file_handler(name: str | None = None) -> logging.Handler:
    """显式注册文件 handler（兼容旧 API）。"""
    handler_key = name or _caller_stem()
    with _FILE_HANDLER_LOCK:
        if handler_key in _FILE_HANDLERS:
            return _FILE_HANDLERS[handler_key]
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = os.path.join(LOG_DIR, f"{handler_key}.log")
        fh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_FileFormatter(FILE_FORMAT, datefmt=FILE_DATE))
        fh.addFilter(_ModuleFilter(handler_key))
        _logger.addHandler(fh)
        _FILE_HANDLERS[handler_key] = fh
        return fh
