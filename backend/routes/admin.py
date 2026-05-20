from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import func
from werkzeug.security import generate_password_hash

from backend.auth import admin_required
from backend.db import get_db
from backend.db_models import AdminAuditLog, AnalysisRecord, AnalysisTask, User
from backend.services.data_cleanup import (
    cleanup_dangling_tasks,
    cleanup_orphan_task_dirs,
    cleanup_stale_queued_tasks,
    hard_delete_user_data,
    purge_record_with_related_tasks,
)
from backend.utils import safe_json_dumps, safe_json_loads


admin_bp = Blueprint("admin", __name__)


def _cleanup_roots() -> list[Path]:
    roots = []
    for key in ("OUTPUT_DIR", "UPLOAD_DIR"):
        value = current_app.config.get(key)
        if value:
            roots.append(Path(value))
    return roots


def _output_root() -> Path:
    return Path(current_app.config["OUTPUT_DIR"])


def _to_user_payload(user: User, task_count: int | None = None, record_count: int | None = None) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": bool(int(user.is_admin or 0)),
        "is_active": bool(int(user.is_active or 0)),
        "task_count": int(task_count or 0),
        "record_count": int(record_count or 0),
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
    }


def _append_audit(
    db,
    action: str,
    target_user_id: int | None = None,
    target_record_id: int | None = None,
    details: dict | None = None,
) -> None:
    db.add(
        AdminAuditLog(
            admin_user_id=g.user_id,
            action=action,
            target_user_id=target_user_id,
            target_record_id=target_record_id,
            details_json=safe_json_dumps(details or {}),
        )
    )


def _action_label(action: str) -> str:
    mapping = {
        "update_user": "更新用户信息",
        "reset_password": "重置用户密码",
        "hard_delete_record": "硬删除单条记录",
        "hard_delete_user": "硬删除单个用户",
        "batch_hard_delete_records": "批量硬删除记录",
        "batch_hard_delete_users": "批量硬删除用户",
        "cleanup_orphan_task_dirs": "清理孤儿任务目录",
        "cleanup_dangling_tasks": "修复悬挂任务索引",
        "cleanup_stale_queued_tasks": "清理卡住队列任务",
    }
    return mapping.get(action, "管理员操作")


def _action_description(action: str, details: dict, target_user_name: str, target_record_id: int | None) -> str:
    if action == "update_user":
        role_text = "管理员" if bool(details.get("is_admin")) else "普通用户"
        active_text = "启用" if bool(details.get("is_active")) else "禁用"
        return f"将用户 {target_user_name} 更新为 {role_text}，状态设为{active_text}。"
    if action == "reset_password":
        return f"已重置用户 {target_user_name} 的登录密码。"
    if action == "hard_delete_record":
        file_count = len(details.get("deleted_paths") or [])
        return f"已硬删除记录 #{target_record_id or '-'}，并清理 {file_count} 个文件。"
    if action == "hard_delete_user":
        return (
            f"已硬删除用户 {target_user_name}，删除记录 {int(details.get('records_deleted') or 0)} 条，"
            f"任务 {int(details.get('tasks_deleted') or 0)} 个，文件 {int(details.get('files_deleted') or 0)} 个。"
        )
    if action == "batch_hard_delete_records":
        return (
            f"批量硬删除记录成功 {int(details.get('deleted_count') or 0)} 条，"
            f"跳过 {int(details.get('skipped_count') or 0)} 条。"
        )
    if action == "batch_hard_delete_users":
        return (
            f"批量硬删除用户成功 {int(details.get('deleted_count') or 0)} 个，"
            f"跳过 {int(details.get('skipped_count') or 0)} 个。"
        )
    if action == "cleanup_orphan_task_dirs":
        return f"已清理孤儿任务目录 {int(details.get('orphan_dirs_deleted') or 0)} 个。"
    if action == "cleanup_dangling_tasks":
        return f"已修复并清理悬挂任务 {int(details.get('dangling_tasks_deleted') or 0)} 个。"
    if action == "cleanup_stale_queued_tasks":
        return f"已清理卡住队列任务 {int(details.get('stale_queued_deleted') or 0)} 个。"
    return "执行了一次管理员操作。"


@admin_bp.get("/admin/overview")
@admin_required
def overview():
    db = get_db()
    try:
        total_users = int(db.query(func.count(User.id)).scalar() or 0)
        active_users = int(
            db.query(func.count(User.id)).filter(User.is_active == 1).scalar() or 0
        )
        total_records = int(db.query(func.count(AnalysisRecord.id)).scalar() or 0)
        total_tasks = int(db.query(func.count(AnalysisTask.id)).scalar() or 0)
        admins = int(db.query(func.count(User.id)).filter(User.is_admin == 1).scalar() or 0)
        return jsonify(
            {
                "success": True,
                "overview": {
                    "total_users": total_users,
                    "active_users": active_users,
                    "admin_users": admins,
                    "total_records": total_records,
                    "total_tasks": total_tasks,
                },
            }
        )
    finally:
        db.close()


@admin_bp.get("/admin/users")
@admin_required
def list_users():
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, min(50, int(request.args.get("per_page", 20))))
    keyword = (request.args.get("keyword") or "").strip()

    db = get_db()
    try:
        query = db.query(User)
        if keyword:
            pattern = f"%{keyword}%"
            query = query.filter((User.username.like(pattern)) | (User.email.like(pattern)))

        total = int(query.count() or 0)
        users = (
            query.order_by(User.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        user_ids = [item.id for item in users]

        task_counts: dict[int, int] = {}
        record_counts: dict[int, int] = {}
        if user_ids:
            task_rows = (
                db.query(AnalysisTask.user_id, func.count(AnalysisTask.id))
                .filter(AnalysisTask.user_id.in_(user_ids))
                .group_by(AnalysisTask.user_id)
                .all()
            )
            record_rows = (
                db.query(AnalysisRecord.user_id, func.count(AnalysisRecord.id))
                .filter(AnalysisRecord.user_id.in_(user_ids))
                .group_by(AnalysisRecord.user_id)
                .all()
            )
            task_counts = {int(uid): int(cnt) for uid, cnt in task_rows}
            record_counts = {int(uid): int(cnt) for uid, cnt in record_rows}

        payload = [
            _to_user_payload(
                user=item,
                task_count=task_counts.get(item.id, 0),
                record_count=record_counts.get(item.id, 0),
            )
            for item in users
        ]
        return jsonify(
            {
                "success": True,
                "users": payload,
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        )
    finally:
        db.close()


@admin_bp.get("/admin/users/<int:user_id>/records")
@admin_required
def list_user_records(user_id: int):
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, min(50, int(request.args.get("per_page", 10))))
    db = get_db()
    try:
        user = db.get(User, user_id)
        if user is None:
            return jsonify({"success": False, "error": "用户不存在"}), 404

        query = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == user_id)
        total = int(query.count() or 0)
        rows = (
            query.order_by(AnalysisRecord.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        records = [
            {
                "id": row.id,
                "video_filename": row.video_filename,
                "view_angle": row.view_angle,
                "total_score": row.total_score,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
        return jsonify(
            {
                "success": True,
                "records": records,
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        )
    finally:
        db.close()


@admin_bp.patch("/admin/users/<int:user_id>")
@admin_required
def update_user(user_id: int):
    payload = request.get_json(silent=True) or {}
    db = get_db()
    try:
        user = db.get(User, user_id)
        if user is None:
            return jsonify({"success": False, "error": "用户不存在"}), 404

        if "is_admin" in payload:
            to_admin = 1 if bool(payload.get("is_admin")) else 0
            if user.id == g.user_id and to_admin == 0:
                return jsonify({"success": False, "error": "不能取消自己的管理员权限"}), 400
            user.is_admin = to_admin

        if "is_active" in payload:
            to_active = 1 if bool(payload.get("is_active")) else 0
            if user.id == g.user_id and to_active == 0:
                return jsonify({"success": False, "error": "不能禁用当前登录账号"}), 400
            user.is_active = to_active

        if "email" in payload:
            email = (payload.get("email") or "").strip() or None
            if email:
                dup = db.query(User).filter(User.email == email, User.id != user.id).first()
                if dup:
                    return jsonify({"success": False, "error": "邮箱已被占用"}), 409
            user.email = email

        _append_audit(
            db,
            action="update_user",
            target_user_id=user.id,
            details={
                "is_admin": bool(int(user.is_admin or 0)),
                "is_active": bool(int(user.is_active or 0)),
                "email": user.email,
            },
        )
        db.commit()
        return jsonify({"success": True, "user": _to_user_payload(user)})
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.post("/admin/users/<int:user_id>/reset-password")
@admin_required
def reset_user_password(user_id: int):
    payload = request.get_json(silent=True) or {}
    new_password = payload.get("new_password") or ""
    if len(new_password) < 6:
        return jsonify({"success": False, "error": "密码至少6位"}), 400

    db = get_db()
    try:
        user = db.get(User, user_id)
        if user is None:
            return jsonify({"success": False, "error": "用户不存在"}), 404
        user.password_hash = generate_password_hash(new_password)
        _append_audit(
            db,
            action="reset_password",
            target_user_id=user.id,
            details={"username": user.username},
        )
        db.commit()
        return jsonify({"success": True, "message": "密码已重置"})
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.delete("/admin/records/<int:record_id>/hard-delete")
@admin_required
def hard_delete_record(record_id: int):
    db = get_db()
    try:
        record = db.get(AnalysisRecord, record_id)
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404

        summary = purge_record_with_related_tasks(
            db,
            record,
            _cleanup_roots(),
            output_root=_output_root(),
            restrict_user_id=record.user_id,
        )
        target_user_id = record.user_id
        _append_audit(
            db,
            action="hard_delete_record",
            target_user_id=target_user_id,
            target_record_id=record_id,
            details=summary,
        )
        db.commit()
        return jsonify(
            {
                "success": True,
                "message": "记录硬删除完成",
                "deleted_paths": summary["deleted_paths"],
                "deleted_task_ids": summary["deleted_task_ids"],
            }
        )
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.post("/admin/records/hard-delete-batch")
@admin_required
def hard_delete_records_batch():
    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get("record_ids") or []
    if not isinstance(raw_ids, list):
        return jsonify({"success": False, "error": "record_ids 必须是数组"}), 400

    record_ids: list[int] = []
    for item in raw_ids:
        try:
            rid = int(item)
        except Exception:
            continue
        if rid > 0 and rid not in record_ids:
            record_ids.append(rid)
    if not record_ids:
        return jsonify({"success": False, "error": "请选择至少1条记录"}), 400

    db = get_db()
    try:
        records = db.query(AnalysisRecord).filter(AnalysisRecord.id.in_(record_ids)).all()
        existing = {int(row.id): row for row in records}
        deleted_paths: list[str] = []
        deleted_task_ids: list[str] = []
        deleted_ids: list[int] = []
        skipped_ids: list[int] = []

        for rid in record_ids:
            row = existing.get(rid)
            if row is None:
                skipped_ids.append(rid)
                continue
            summary = purge_record_with_related_tasks(
                db,
                row,
                _cleanup_roots(),
                output_root=_output_root(),
                restrict_user_id=row.user_id,
            )
            deleted_paths.extend(summary["deleted_paths"])
            deleted_task_ids.extend(summary["deleted_task_ids"])
            deleted_ids.append(rid)

        _append_audit(
            db,
            action="batch_hard_delete_records",
            details={
                "requested_ids": record_ids,
                "deleted_ids": deleted_ids,
                "skipped_ids": skipped_ids,
                "deleted_count": len(deleted_ids),
                "skipped_count": len(skipped_ids),
                "deleted_paths_count": len(deleted_paths),
                "deleted_tasks_count": len(set(deleted_task_ids)),
            },
        )
        db.commit()
        return jsonify(
            {
                "success": True,
                "deleted_ids": deleted_ids,
                "skipped_ids": skipped_ids,
                "deleted_paths_count": len(deleted_paths),
                "deleted_task_ids": sorted(set(deleted_task_ids)),
            }
        )
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.delete("/admin/users/<int:user_id>/hard-delete")
@admin_required
def hard_delete_user(user_id: int):
    db = get_db()
    try:
        user = db.get(User, user_id)
        if user is None:
            return jsonify({"success": False, "error": "用户不存在"}), 404
        if user.id == g.user_id:
            return jsonify({"success": False, "error": "不能硬删除当前管理员账号"}), 400
        if int(user.is_admin or 0):
            return jsonify({"success": False, "error": "请先取消管理员角色后再删除"}), 400

        summary = hard_delete_user_data(
            db,
            user,
            _cleanup_roots(),
            output_root=_output_root(),
        )
        _append_audit(
            db,
            action="hard_delete_user",
            target_user_id=user_id,
            details=summary,
        )
        db.commit()
        return jsonify({"success": True, "message": "用户数据硬删除完成", "summary": summary})
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.post("/admin/users/hard-delete-batch")
@admin_required
def hard_delete_users_batch():
    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get("user_ids") or []
    if not isinstance(raw_ids, list):
        return jsonify({"success": False, "error": "user_ids 必须是数组"}), 400

    user_ids: list[int] = []
    for item in raw_ids:
        try:
            uid = int(item)
        except Exception:
            continue
        if uid > 0 and uid not in user_ids:
            user_ids.append(uid)
    if not user_ids:
        return jsonify({"success": False, "error": "请选择至少1个用户"}), 400

    db = get_db()
    try:
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        user_map = {int(row.id): row for row in users}
        deleted_user_ids: list[int] = []
        skipped: list[dict] = []
        total_records = 0
        total_tasks = 0
        total_files = 0

        for uid in user_ids:
            user = user_map.get(uid)
            if user is None:
                skipped.append({"id": uid, "reason": "用户不存在"})
                continue
            if uid == g.user_id:
                skipped.append({"id": uid, "reason": "不能删除当前登录管理员"})
                continue
            if int(user.is_admin or 0):
                skipped.append({"id": uid, "reason": "管理员账号不能批量删除"})
                continue

            summary = hard_delete_user_data(
                db,
                user,
                _cleanup_roots(),
                output_root=_output_root(),
            )
            deleted_user_ids.append(uid)
            total_records += int(summary.get("records_deleted") or 0)
            total_tasks += int(summary.get("tasks_deleted") or 0)
            total_files += int(summary.get("files_deleted") or 0)

        _append_audit(
            db,
            action="batch_hard_delete_users",
            details={
                "requested_ids": user_ids,
                "deleted_user_ids": deleted_user_ids,
                "skipped": skipped,
                "deleted_count": len(deleted_user_ids),
                "skipped_count": len(skipped),
                "records_deleted": total_records,
                "tasks_deleted": total_tasks,
                "files_deleted": total_files,
            },
        )
        db.commit()
        return jsonify(
            {
                "success": True,
                "deleted_user_ids": deleted_user_ids,
                "skipped": skipped,
                "summary": {
                    "deleted_count": len(deleted_user_ids),
                    "skipped_count": len(skipped),
                    "records_deleted": total_records,
                    "tasks_deleted": total_tasks,
                    "files_deleted": total_files,
                },
            }
        )
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.post("/admin/storage/cleanup-orphans")
@admin_required
def cleanup_orphan_storage():
    db = get_db()
    try:
        summary = cleanup_orphan_task_dirs(db, _output_root(), _cleanup_roots())
        _append_audit(
            db,
            action="cleanup_orphan_task_dirs",
            details=summary,
        )
        db.commit()
        return jsonify({"success": True, "summary": summary})
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.post("/admin/storage/cleanup-dangling-tasks")
@admin_required
def cleanup_dangling_tasks_api():
    db = get_db()
    try:
        summary = cleanup_dangling_tasks(db, _output_root(), _cleanup_roots())
        _append_audit(
            db,
            action="cleanup_dangling_tasks",
            details=summary,
        )
        db.commit()
        return jsonify({"success": True, "summary": summary})
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.post("/admin/storage/cleanup-stale-queued")
@admin_required
def cleanup_stale_queued_tasks_api():
    payload = request.get_json(silent=True) or {}
    try:
        min_age_minutes = int(payload.get("min_age_minutes", 10))
    except Exception:
        min_age_minutes = 10
    if min_age_minutes < 0:
        min_age_minutes = 0
    if min_age_minutes > 24 * 60:
        min_age_minutes = 24 * 60

    db = get_db()
    try:
        summary = cleanup_stale_queued_tasks(
            db,
            _output_root(),
            _cleanup_roots(),
            min_age_minutes=min_age_minutes,
        )
        _append_audit(
            db,
            action="cleanup_stale_queued_tasks",
            details=summary,
        )
        db.commit()
        return jsonify({"success": True, "summary": summary})
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@admin_bp.get("/admin/audit-logs")
@admin_required
def audit_logs():
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, min(50, int(request.args.get("per_page", 20))))
    db = get_db()
    try:
        query = db.query(AdminAuditLog)
        total = int(query.count() or 0)
        rows = (
            query.order_by(AdminAuditLog.created_at.desc(), AdminAuditLog.id.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        user_ids: set[int] = set()
        for row in rows:
            if row.admin_user_id:
                user_ids.add(int(row.admin_user_id))
            if row.target_user_id:
                user_ids.add(int(row.target_user_id))
        user_map = {}
        if user_ids:
            user_rows = db.query(User.id, User.username).filter(User.id.in_(list(user_ids))).all()
            user_map = {int(uid): name for uid, name in user_rows}

        logs = []
        for row in rows:
            details = safe_json_loads(row.details_json, {})
            if row.target_user_id:
                target_name = user_map.get(int(row.target_user_id), f"ID:{row.target_user_id}")
            else:
                target_name = "-"
            logs.append(
                {
                    "id": row.id,
                    "admin_user_id": row.admin_user_id,
                    "admin_username": user_map.get(int(row.admin_user_id or 0), f"ID:{row.admin_user_id}"),
                    "action": row.action,
                    "action_cn": _action_label(row.action),
                    "description_cn": _action_description(
                        row.action,
                        details,
                        target_name,
                        row.target_record_id,
                    ),
                    "target_user_id": row.target_user_id,
                    "target_username": target_name,
                    "target_record_id": row.target_record_id,
                    "details_json": row.details_json or "{}",
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            )
        return jsonify(
            {
                "success": True,
                "logs": logs,
                "total": total,
                "page": page,
                "per_page": per_page,
            }
        )
    finally:
        db.close()
