from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".dev_pids.json"


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
    try:
        import os

        os.kill(pid, signal.SIGTERM)
        return True
    except Exception:
        return False


def main() -> int:
    backend_pid = 0
    frontend_pid = 0
    backend_launcher_pid = 0
    frontend_launcher_pid = 0
    if PID_FILE.exists():
        data = json.loads(PID_FILE.read_text(encoding="utf-8"))
        backend_pid = int(data.get("backend_pid") or 0)
        frontend_pid = int(data.get("frontend_pid") or 0)
        backend_launcher_pid = int(data.get("backend_launcher_pid") or 0)
        frontend_launcher_pid = int(data.get("frontend_launcher_pid") or 0)
    else:
        print("pid_file not found:", PID_FILE)

    kill_targets = []
    for pid in (backend_pid, backend_launcher_pid, frontend_pid, frontend_launcher_pid):
        if pid > 0 and pid not in kill_targets:
            kill_targets.append(pid)
    for pid in kill_targets:
        tag = "backend" if pid in {backend_pid, backend_launcher_pid} else "frontend"
        print(f"stop {tag}:", pid, _kill(pid))

    # Defensive cleanup: only target listening ports used by this project.
    for pid in _find_port_pids(5000):
        if pid not in set(kill_targets):
            print("cleanup port5000:", pid, _kill(pid))
    for pid in _find_port_pids(5173):
        if pid not in set(kill_targets):
            print("cleanup port5173:", pid, _kill(pid))

    try:
        PID_FILE.unlink()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
