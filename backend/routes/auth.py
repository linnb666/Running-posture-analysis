from __future__ import annotations

from flask import Blueprint, g, jsonify, request
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth import auth_required, create_access_token, create_refresh_token, decode_token
from backend.db import get_db
from backend.db_models import User


auth_bp = Blueprint("auth", __name__)


def _user_payload(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": bool(int(user.is_admin or 0)),
        "is_active": bool(int(user.is_active or 0)),
    }


@auth_bp.post("/auth/register")
def register():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    email = (payload.get("email") or "").strip() or None

    if not username or not password:
        return jsonify({"success": False, "error": "username and password are required"}), 400
    if len(username) < 3 or len(password) < 6:
        return jsonify({"success": False, "error": "username>=3 and password>=6"}), 400

    db = get_db()
    try:
        if db.query(User).filter(User.username == username).first():
            return jsonify({"success": False, "error": "username already exists"}), 409
        if email and db.query(User).filter(User.email == email).first():
            return jsonify({"success": False, "error": "email already exists"}), 409

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            is_admin=0,
            is_active=1,
        )
        db.add(user)
        db.commit()
        return jsonify({"success": True, "message": "registered"}), 201
    finally:
        db.close()


@auth_bp.post("/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if not username:
        return jsonify({"success": False, "error": "请输入用户名"}), 400
    if not password:
        return jsonify({"success": False, "error": "请输入密码"}), 400
    db = get_db()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            return jsonify({"success": False, "error": "用户不存在"}), 404
        if not check_password_hash(user.password_hash, password):
            return jsonify({"success": False, "error": "密码错误"}), 401
        if not int(user.is_active or 0):
            return jsonify({"success": False, "error": "account disabled"}), 403
        return jsonify(
            {
                "success": True,
                "access_token": create_access_token(user.id),
                "refresh_token": create_refresh_token(user.id),
                "user": _user_payload(user),
            }
        )
    finally:
        db.close()


@auth_bp.post("/auth/refresh")
def refresh():
    payload = request.get_json(silent=True) or {}
    refresh_token = payload.get("refresh_token") or ""
    token_payload = decode_token(refresh_token)
    if not token_payload or token_payload.get("type") != "refresh":
        return jsonify({"success": False, "error": "invalid refresh token"}), 401
    user_id = int(token_payload["sub"])
    db = get_db()
    try:
        user = db.get(User, user_id)
        if user is None or not int(user.is_active or 0):
            return jsonify({"success": False, "error": "account disabled"}), 403
        return jsonify({"success": True, "access_token": create_access_token(user_id)})
    finally:
        db.close()


@auth_bp.post("/auth/logout")
def logout():
    return jsonify({"success": True, "message": "logout on client side"})


@auth_bp.get("/auth/me")
@auth_required
def me():
    db = get_db()
    try:
        user = db.get(User, g.user_id)
        if user is None:
            return jsonify({"success": False, "error": "user not found"}), 404
        return jsonify(
            {
                "success": True,
                "user": _user_payload(user),
            }
        )
    finally:
        db.close()
