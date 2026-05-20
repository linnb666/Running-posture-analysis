from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import current_app, g, jsonify, request


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_token(user_id: int, token_type: str, expire_delta: timedelta) -> str:
    now = _utc_now()
    payload = {
        "sub": str(user_id),
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expire_delta).timestamp()),
    }
    return jwt.encode(
        payload,
        current_app.config["SECRET_KEY"],
        algorithm=current_app.config["JWT_ALGORITHM"],
    )


def create_access_token(user_id: int) -> str:
    minutes = int(current_app.config["ACCESS_TOKEN_EXPIRE_MINUTES"])
    return _build_token(user_id, "access", timedelta(minutes=minutes))


def create_refresh_token(user_id: int) -> str:
    days = int(current_app.config["REFRESH_TOKEN_EXPIRE_DAYS"])
    return _build_token(user_id, "refresh", timedelta(days=days))


def decode_token(token: str):
    try:
        return jwt.decode(
            token,
            current_app.config["SECRET_KEY"],
            algorithms=[current_app.config["JWT_ALGORITHM"]],
        )
    except Exception:
        return None


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
        else:
            # Fallback for media tags (<video>/<img>) that cannot attach custom headers.
            token = (request.args.get("access_token") or "").strip()
            if not token:
                return jsonify({"success": False, "error": "missing bearer token"}), 401
        payload = decode_token(token)
        if not payload or payload.get("type") != "access":
            return jsonify({"success": False, "error": "invalid token"}), 401
        try:
            g.user_id = int(payload["sub"])
        except Exception:
            return jsonify({"success": False, "error": "invalid token subject"}), 401

        from backend.db import get_db
        from backend.db_models import User

        db = get_db()
        try:
            user = db.get(User, g.user_id)
            if user is None:
                return jsonify({"success": False, "error": "user not found"}), 401
            if not int(user.is_active or 0):
                return jsonify({"success": False, "error": "account disabled"}), 403
            g.is_admin = bool(int(user.is_admin or 0))
        finally:
            db.close()
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn):
    @wraps(fn)
    @auth_required
    def wrapper(*args, **kwargs):
        if not bool(getattr(g, "is_admin", False)):
            return jsonify({"success": False, "error": "admin required"}), 403
        return fn(*args, **kwargs)

    return wrapper
