from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".dev_pids.json"
LOG_DIR = ROOT / "data" / "webapp" / "logs"


def _python_has_module(python_bin: str, module_name: str) -> bool:
    try:
        result = subprocess.run(
            [python_bin, "-c", f"import {module_name};print('ok')"],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
        return result.returncode == 0
    except Exception:
        return False


def _python_has_cv2(python_bin: str) -> bool:
    return _python_has_module(python_bin, "cv2")


def _python_has_zai(python_bin: str) -> bool:
    return _python_has_module(python_bin, "zai")


def _select_backend_python() -> str:
    candidates = []
    env_python = (os.environ.get("RUNNING_PYTHON") or "").strip()
    if env_python:
        candidates.append(env_python)
    candidates.extend(
        [
            sys.executable,
            r"D:\anaconda3\python.exe",
            r"C:\Users\lin\anaconda3\python.exe",
            r"C:\ProgramData\anaconda3\python.exe",
        ]
    )

    checked = set()
    cv2_only_candidates = []
    for candidate in candidates:
        if not candidate:
            continue
        c = str(Path(candidate))
        if c in checked:
            continue
        checked.add(c)
        if not Path(c).exists():
            continue
        has_cv2 = _python_has_cv2(c)
        has_zai = _python_has_zai(c)
        if has_cv2 and has_zai:
            return c
        if has_cv2:
            cv2_only_candidates.append(c)
    if cv2_only_candidates:
        return cv2_only_candidates[0]
    return sys.executable


def _find_pids(process_name: str, cmdline_regex: str) -> list[int]:
    if __import__("os").name != "nt":
        return []
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.Name -eq '{process_name}' -and $_.CommandLine -match '{cmdline_regex}' }} | "
            "Select-Object -ExpandProperty ProcessId"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return sorted(set(pids))


def _kill(pid: int) -> bool:
    if pid <= 0:
        return False
    if __import__("os").name == "nt":
        result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    return False


def _find_port_pids(port: int) -> list[int]:
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        (
            f"netstat -ano | findstr LISTENING | findstr :{port} | "
            "ForEach-Object { ($_ -split '\\s+')[-1] }"
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    pids: list[int] = []
    for line in result.stdout.splitlines():
        v = line.strip()
        if v.isdigit():
            pids.append(int(v))
    return sorted(set(pids))


def _first_listen_pid(port: int) -> int:
    pids = _find_port_pids(port)
    return int(pids[0]) if pids else 0


def _wait_http(url: str, timeout_sec: int = 25) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as resp:  # nosec B310
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    # Idempotent startup: clean stale backend/frontend dev processes first.
    for pid in _find_pids("python.exe", r"backend/app\.py"):
        _kill(pid)
    for pid in _find_port_pids(5000):
        _kill(pid)
    for pid in _find_port_pids(5173):
        _kill(pid)

    backend_python = _select_backend_python()
    backend_has_zai = _python_has_zai(backend_python)
    backend_cmd = [backend_python, "backend/app.py"]
    frontend_cmd = ["npm.cmd", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173", "--strictPort"]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    backend_log_path = LOG_DIR / "backend_dev.log"
    frontend_log_path = LOG_DIR / "frontend_dev.log"

    backend_log = backend_log_path.open("a", encoding="utf-8", errors="ignore")
    frontend_log = frontend_log_path.open("a", encoding="utf-8", errors="ignore")

    backend = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT),
        stdout=backend_log,
        stderr=backend_log,
    )
    frontend = subprocess.Popen(
        frontend_cmd,
        cwd=str(ROOT / "frontend"),
        stdout=frontend_log,
        stderr=frontend_log,
    )

    backend_ok = _wait_http("http://127.0.0.1:5000/health", timeout_sec=35)
    frontend_ok = _wait_http("http://127.0.0.1:5173", timeout_sec=35)
    backend_listen_pid = _first_listen_pid(5000) if backend_ok else 0
    frontend_listen_pid = _first_listen_pid(5173) if frontend_ok else 0

    pids = {
        "backend_pid": backend_listen_pid or backend.pid,
        "frontend_pid": frontend_listen_pid or frontend.pid,
        "backend_launcher_pid": backend.pid,
        "frontend_launcher_pid": frontend.pid,
        "started_at": int(time.time()),
        "backend_log": str(backend_log_path),
        "frontend_log": str(frontend_log_path),
    }
    PID_FILE.write_text(json.dumps(pids, ensure_ascii=False, indent=2), encoding="utf-8")

    backend_log.close()
    frontend_log.close()

    print("backend_pid:", pids["backend_pid"], "ok:", backend_ok)
    print("backend_python:", backend_python)
    print("backend_ai_zai:", backend_has_zai)
    print("frontend_pid:", pids["frontend_pid"], "ok:", frontend_ok)
    print("backend_url: http://127.0.0.1:5000")
    print("frontend_url: http://127.0.0.1:5173")
    print("pid_file:", PID_FILE)

    if not backend_ok or not frontend_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
