from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from backend.db_models import AnalysisRecord, AnalysisTask, User


def _resolve_under_roots(raw_path: str | None, allowed_roots: list[Path]) -> Path | None:
    if not raw_path:
        return None
    try:
        path = Path(raw_path).resolve()
    except Exception:
        return None
    for root in allowed_roots:
        try:
            root_resolved = root.resolve()
            if path == root_resolved or root_resolved in path.parents:
                return path
        except Exception:
            continue
    return None


def _remove_path(path: Path) -> bool:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=False)
            return True
        if path.is_file():
            path.unlink(missing_ok=True)
            return True
    except Exception:
        return False
    return False


def _compact_paths(paths: list[Path]) -> list[Path]:
    unique = []
    seen: set[str] = set()
    for path in sorted(paths, key=lambda x: len(str(x))):
        key = str(path)
        if key in seen:
            continue
        if any(existing == path or existing in path.parents for existing in unique):
            continue
        unique.append(path)
        seen.add(key)
    return sorted(unique, key=lambda x: len(str(x)), reverse=True)


def purge_record_assets(record: AnalysisRecord, allowed_roots: list[Path]) -> list[str]:
    candidates: list[Path] = []
    output_dir = _resolve_under_roots(record.output_dir, allowed_roots)
    if output_dir is not None:
        candidates.append(output_dir)
    original_file = _resolve_under_roots(record.original_video_path, allowed_roots)
    if original_file is not None:
        candidates.append(original_file)

    deleted: list[str] = []
    for path in _compact_paths(candidates):
        if _remove_path(path):
            deleted.append(str(path))
    return deleted


def purge_task_assets(task: AnalysisTask, allowed_roots: list[Path], output_root: Path | None = None) -> list[str]:
    candidates: list[Path] = []
    if output_root is not None:
        candidates.append(output_root / str(task.id))
    input_file = _resolve_under_roots(task.input_video_path, allowed_roots)
    if input_file is not None:
        candidates.append(input_file)

    deleted: list[str] = []
    for path in _compact_paths(candidates):
        resolved = _resolve_under_roots(str(path), allowed_roots)
        if resolved is None:
            continue
        if _remove_path(resolved):
            deleted.append(str(resolved))
    return deleted


def purge_record_with_related_tasks(
    db,
    record: AnalysisRecord,
    allowed_roots: list[Path],
    output_root: Path | None = None,
    restrict_user_id: int | None = None,
) -> dict:
    tasks_to_delete: dict[str, AnalysisTask] = {}

    if record.task_id:
        query = db.query(AnalysisTask).filter(AnalysisTask.id == record.task_id)
        if restrict_user_id is not None:
            query = query.filter(AnalysisTask.user_id == restrict_user_id)
        task = query.first()
        if task is not None:
            tasks_to_delete[str(task.id)] = task

    query = db.query(AnalysisTask).filter(AnalysisTask.result_record_id == record.id)
    if restrict_user_id is not None:
        query = query.filter(AnalysisTask.user_id == restrict_user_id)
    for task in query.all():
        tasks_to_delete[str(task.id)] = task

    deleted_paths = purge_record_assets(record, allowed_roots)
    for task in tasks_to_delete.values():
        deleted_paths.extend(purge_task_assets(task, allowed_roots, output_root=output_root))

    db.delete(record)
    db.flush()
    for task in tasks_to_delete.values():
        db.delete(task)

    return {
        "deleted_task_ids": sorted(tasks_to_delete.keys()),
        "deleted_tasks_count": len(tasks_to_delete),
        "deleted_paths": sorted(set(deleted_paths)),
    }


def hard_delete_user_data(db, user: User, allowed_roots: list[Path], output_root: Path | None = None) -> dict:
    records = list(db.query(AnalysisRecord).filter(AnalysisRecord.user_id == user.id).all())
    tasks = list(db.query(AnalysisTask).filter(AnalysisTask.user_id == user.id).all())

    deleted_files: list[str] = []
    for record in records:
        deleted_files.extend(purge_record_assets(record, allowed_roots))
        db.delete(record)
    for task in tasks:
        deleted_files.extend(purge_task_assets(task, allowed_roots, output_root=output_root))
        db.delete(task)
    db.delete(user)

    return {
        "records_deleted": len(records),
        "tasks_deleted": len(tasks),
        "files_deleted": len(deleted_files),
        "deleted_paths": sorted(set(deleted_files)),
    }


def cleanup_dangling_tasks(db, output_root: Path, allowed_roots: list[Path]) -> dict:
    rows = (
        db.query(AnalysisTask)
        .outerjoin(AnalysisRecord, AnalysisTask.result_record_id == AnalysisRecord.id)
        .filter(AnalysisTask.result_record_id.isnot(None), AnalysisRecord.id.is_(None))
        .all()
    )
    deleted_paths: list[str] = []
    deleted_task_ids: list[str] = []
    for task in rows:
        deleted_paths.extend(purge_task_assets(task, allowed_roots, output_root=output_root))
        deleted_task_ids.append(str(task.id))
        db.delete(task)
    return {
        "dangling_tasks_deleted": len(deleted_task_ids),
        "deleted_task_ids": sorted(set(deleted_task_ids)),
        "deleted_paths": sorted(set(deleted_paths)),
    }


def cleanup_stale_queued_tasks(
    db,
    output_root: Path,
    allowed_roots: list[Path],
    min_age_minutes: int = 10,
) -> dict:
    age = max(0, int(min_age_minutes))
    cutoff = datetime.utcnow() - timedelta(minutes=age)
    candidates = (
        db.query(AnalysisTask)
        .filter(AnalysisTask.status == "queued", AnalysisTask.created_at <= cutoff)
        .all()
    )

    deleted_paths: list[str] = []
    deleted_task_ids: list[str] = []
    skipped_linked_tasks: list[str] = []

    for task in candidates:
        linked_by_task = (
            db.query(AnalysisRecord.id).filter(AnalysisRecord.task_id == task.id).first()
        )
        linked_by_result_id = None
        if task.result_record_id is not None:
            linked_by_result_id = db.get(AnalysisRecord, task.result_record_id)
        if linked_by_task is not None or linked_by_result_id is not None:
            skipped_linked_tasks.append(str(task.id))
            continue
        deleted_paths.extend(purge_task_assets(task, allowed_roots, output_root=output_root))
        deleted_task_ids.append(str(task.id))
        db.delete(task)

    return {
        "stale_queued_deleted": len(deleted_task_ids),
        "deleted_task_ids": sorted(set(deleted_task_ids)),
        "skipped_linked_tasks": sorted(set(skipped_linked_tasks)),
        "deleted_paths": sorted(set(deleted_paths)),
        "cutoff_utc": cutoff.isoformat(),
        "min_age_minutes": age,
    }


def cleanup_orphan_task_dirs(db, output_root: Path, allowed_roots: list[Path]) -> dict:
    root = _resolve_under_roots(str(output_root), allowed_roots)
    if root is None or not root.exists():
        return {"orphan_dirs_deleted": 0, "deleted_paths": []}

    record_dirs = {
        str(Path(row[0]).resolve())
        for row in db.query(AnalysisRecord.output_dir).filter(AnalysisRecord.output_dir.isnot(None)).all()
        if row[0]
    }
    task_dirs = {str((root / str(row[0])).resolve()) for row in db.query(AnalysisTask.id).all()}
    linked_dirs = record_dirs | task_dirs

    deleted: list[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        c = str(child.resolve())
        if c in linked_dirs:
            continue
        resolved = _resolve_under_roots(c, allowed_roots)
        if resolved is None:
            continue
        if _remove_path(resolved):
            deleted.append(c)

    return {
        "orphan_dirs_deleted": len(deleted),
        "deleted_paths": sorted(set(deleted)),
    }
