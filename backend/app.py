from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import BackendConfig
from backend.db import ensure_default_admin, get_db, init_db
from backend.routes.admin import admin_bp
from backend.routes.analysis import analysis_bp
from backend.routes.auth import auth_bp
from backend.services.data_cleanup import cleanup_stale_queued_tasks


def _stabilize_stdio() -> None:
    # In detached IDE launches on Windows, stdio handles can become invalid.
    # Rebind to os.devnull to prevent engine prints from crashing worker threads.
    try:
        sys.stdout.write("")
    except Exception:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="ignore")
    try:
        sys.stderr.write("")
    except Exception:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="ignore")


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    BackendConfig.ensure_dirs()
    app.config.from_mapping(
        PROJECT_ROOT=str(BackendConfig.PROJECT_ROOT),
        SECRET_KEY=BackendConfig.SECRET_KEY,
        DEFAULT_ADMIN_USERNAME=BackendConfig.DEFAULT_ADMIN_USERNAME,
        DEFAULT_ADMIN_PASSWORD=BackendConfig.DEFAULT_ADMIN_PASSWORD,
        DEFAULT_ADMIN_EMAIL=BackendConfig.DEFAULT_ADMIN_EMAIL,
        ACCESS_TOKEN_EXPIRE_MINUTES=BackendConfig.ACCESS_TOKEN_EXPIRE_MINUTES,
        REFRESH_TOKEN_EXPIRE_DAYS=BackendConfig.REFRESH_TOKEN_EXPIRE_DAYS,
        JWT_ALGORITHM=BackendConfig.JWT_ALGORITHM,
        AUTO_AI_ANALYSIS=BackendConfig.AUTO_AI_ANALYSIS,
        ENABLE_MODEL_CHECKSUM=BackendConfig.ENABLE_MODEL_CHECKSUM,
        UPLOAD_DIR=str(BackendConfig.UPLOAD_DIR),
        OUTPUT_DIR=str(BackendConfig.OUTPUT_DIR),
        DB_PATH=str(BackendConfig.DB_PATH),
        MAX_CONTENT_LENGTH=BackendConfig.MAX_CONTENT_LENGTH,
        ALLOWED_VIDEO_EXTENSIONS=BackendConfig.ALLOWED_VIDEO_EXTENSIONS,
    )
    if test_config:
        app.config.update(test_config)

    db_path = Path(app.config["DB_PATH"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    ensure_default_admin(
        username=str(app.config.get("DEFAULT_ADMIN_USERNAME") or "").strip(),
        password=str(app.config.get("DEFAULT_ADMIN_PASSWORD") or "").strip(),
        email=str(app.config.get("DEFAULT_ADMIN_EMAIL") or "").strip(),
    )

    # Queue is in-memory only; tasks stuck in queued status across restarts become zombies.
    # Auto-clean stale queued tasks (older than 10 min) to keep DB/media consistent.
    db = get_db()
    try:
        summary = cleanup_stale_queued_tasks(
            db,
            Path(app.config["OUTPUT_DIR"]),
            [Path(app.config["OUTPUT_DIR"]), Path(app.config["UPLOAD_DIR"])],
            min_age_minutes=10,
        )
        db.commit()
        if int(summary.get("stale_queued_deleted") or 0) > 0:
            print(f"[startup] cleaned stale queued tasks: {summary}")
    finally:
        db.close()

    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(analysis_bp, url_prefix="/api")
    app.register_blueprint(admin_bp, url_prefix="/api")

    @app.get("/")
    def index():
        return jsonify({"message": "Running Analysis Backend", "status": "ok"})

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    return app


app = create_app()


if __name__ == "__main__":
    _stabilize_stdio()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
