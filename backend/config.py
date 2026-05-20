from __future__ import annotations

import os
from pathlib import Path


class BackendConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    APP_DATA_DIR = PROJECT_ROOT / "data" / "webapp"
    DB_PATH = APP_DATA_DIR / "running_web.db"
    UPLOAD_DIR = APP_DATA_DIR / "uploads"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "tasks"

    SECRET_KEY = os.getenv("RUNNING_SECRET_KEY", "running-thesis-secret-key")
    DEFAULT_ADMIN_USERNAME = os.getenv("RUNNING_ADMIN_USERNAME", "admin")
    DEFAULT_ADMIN_PASSWORD = os.getenv("RUNNING_ADMIN_PASSWORD", "admin123456")
    DEFAULT_ADMIN_EMAIL = os.getenv("RUNNING_ADMIN_EMAIL", "admin@running.local")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 120
    REFRESH_TOKEN_EXPIRE_DAYS = 14

    AUTO_AI_ANALYSIS = False
    ENABLE_MODEL_CHECKSUM = True

    MAX_CONTENT_LENGTH = 800 * 1024 * 1024
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
