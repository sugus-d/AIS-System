#!/usr/bin/env python3
"""Streamlit 服务器管理脚本 — start/stop/restart/status。

用法:
    uv run python -m commands.streamlit_server status    # 查看状态
    uv run python -m commands.streamlit_server start     # 启动（如已运行则报错）
    uv run python -m commands.streamlit_server restart   # 重启（如未运行则启动）
    uv run python -m commands.streamlit_server stop      # 停止
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8500
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(PROJECT_ROOT / "reports" / "app.py")
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python3")
PID_FILE = Path("/tmp/streamlit_ais.pid")
LOG_FILE = Path("/tmp/streamlit_ais.log")

# main() 需要的最小参数个数（脚本名 + 一个 action）
_MIN_ARG_COUNT = 2


def _find_process() -> int | None:
    """返回已有进程的 PID，或 None。"""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # 探活
            return pid
        except (ValueError, ProcessLookupError, OSError):
            PID_FILE.unlink(missing_ok=True)
    # 兜底：ps 搜索
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "streamlit run.*reports/app.py"],
            text=True,
        ).strip()
        if out:
            return int(out.split("\n")[0])
    except (subprocess.CalledProcessError, ValueError):
        pass
    return None


def cmd_status() -> None:
    pid = _find_process()
    if pid is not None:
        print(f"✅ Streamlit 运行中 (PID={pid})")
        print(f"   http://{HOST}:{PORT}")
        # 检查日志行数
        if LOG_FILE.exists():
            age = time.time() - LOG_FILE.stat().st_mtime
            print(f"   日志: {LOG_FILE} ({(age / 60):.0f} 分钟前更新)")
    else:
        print("❌ Streamlit 未运行")
        print("   启动: uv run python -m commands.streamlit_server start")


def cmd_start() -> None:
    if _find_process() is not None:
        print("❌ Streamlit 已在运行中。如需重启: uv run python -m commands.streamlit_server restart")
        sys.exit(1)

    env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / ".venv")}
    cmd = [
        VENV_PYTHON, "-m", "streamlit", "run",
        APP_PATH,
        "--server.headless=true",
        f"--server.port={PORT}",
        "--server.address=0.0.0.0",
    ]
    with open(LOG_FILE, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=log, start_new_session=True, env=env)

    PID_FILE.write_text(str(proc.pid))
    print(f"✅ Streamlit 已启动 (PID={proc.pid})")
    print(f"   http://{HOST}:{PORT}")
    print(f"   日志: {LOG_FILE}")


def cmd_stop() -> None:
    pid = _find_process()
    if pid is None:
        print("❌ Streamlit 未运行")
        return

    os.kill(pid, signal.SIGTERM)
    # 等进程退出
    for _ in range(5):
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except ProcessLookupError:
            break
    else:
        # SIGTERM 没杀死（uv 包装进程）, 用 SIGKILL
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.3)
        except ProcessLookupError:
            pass

    # 清理残留子进程（无匹配进程时 pkill 返回非零，忽略即可）
    with contextlib.suppress(Exception):
        subprocess.run(["pkill", "-f", "streamlit run.*reports/app.py", "-9"],
                       capture_output=True)

    PID_FILE.unlink(missing_ok=True)
    print(f"✅ Streamlit 已停止 (PID={pid})")


def cmd_restart() -> None:
    cmd_stop()
    time.sleep(1)
    cmd_start()


def main() -> None:
    if len(sys.argv) < _MIN_ARG_COUNT or sys.argv[1] not in ("start", "stop", "restart", "status"):
        print(__doc__.strip())
        sys.exit(1)

    action = sys.argv[1]
    {"start": cmd_start, "stop": cmd_stop, "restart": cmd_restart, "status": cmd_status}[action]()


if __name__ == "__main__":
    main()
