from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def safe_json_dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def safe_json_loads(text: str | None, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""

