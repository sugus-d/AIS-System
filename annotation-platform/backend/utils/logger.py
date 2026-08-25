"""精简日志 — 标准 logging，标注平台自包含，无第三方依赖。

从核心仓库 utils.logger 抽取，保留标注平台实际使用的方法（warning/info/error/debug）。
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
    "%(name)s:%(funcName)s:%(lineno)d - %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _PlatformLogger:
    """日志接口，兼容核心仓库 utils.logger 的常用方法。"""

    def __init__(self, name: str = "annotation-platform") -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def debug(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._logger.debug(str(msg))

    def info(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._logger.info(str(msg))

    def warning(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._logger.warning(str(msg))

    def error(self, msg: object, *_args: object, **_kwargs: object) -> None:
        self._logger.error(str(msg))


logger = _PlatformLogger()
