from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker


Base = declarative_base()
_engine = None
_session_factory = None
SessionLocal = None


def _table_exists(conn, table_name: str) -> bool:
    row = conn.exec_driver_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _table_columns(conn, table_name: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _run_migrations() -> None:
    if _engine is None:
        return
    with _engine.begin() as conn:
        if _table_exists(conn, "users"):
            cols = _table_columns(conn, "users")
            if "is_admin" not in cols:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
            if "is_active" not in cols:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")

        if _table_exists(conn, "analysis_records"):
            cols = _table_columns(conn, "analysis_records")
            if "manual_notes" not in cols:
                conn.exec_driver_sql("ALTER TABLE analysis_records ADD COLUMN manual_notes TEXT")
            if "manual_notes_updated_at" not in cols:
                conn.exec_driver_sql("ALTER TABLE analysis_records ADD COLUMN manual_notes_updated_at DATETIME")

        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS admin_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_user_id INTEGER NOT NULL,
                action VARCHAR(64) NOT NULL,
                target_user_id INTEGER,
                target_record_id INTEGER,
                details_json TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_admin_audit_created_at ON admin_audit_logs(created_at DESC)"
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS idx_admin_audit_admin_user ON admin_audit_logs(admin_user_id)"
        )


def init_db(db_path: Path) -> None:
    global _engine, _session_factory, SessionLocal
    db_url = f"sqlite:///{db_path.as_posix()}"
    _engine = create_engine(
        db_url,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
        },
    )

    @event.listens_for(_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=6000;")
        cursor.close()

    _session_factory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    SessionLocal = scoped_session(_session_factory)
    Base.metadata.create_all(bind=_engine)
    _run_migrations()


def get_db():
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db first.")
    return SessionLocal()


def close_db() -> None:
    global SessionLocal
    if SessionLocal is not None:
        SessionLocal.remove()


def shutdown_db() -> None:
    global _engine, _session_factory, SessionLocal
    if SessionLocal is not None:
        SessionLocal.remove()
    SessionLocal = None
    _session_factory = None
    if _engine is not None:
        _engine.dispose()
    _engine = None


def ensure_default_admin(username: str, password: str, email: str | None = None) -> None:
    user_name = (username or "").strip()
    pwd = (password or "").strip()
    if not user_name or not pwd:
        return

    from werkzeug.security import generate_password_hash

    from backend.db_models import User

    db = get_db()
    try:
        admin = db.query(User).filter(User.username == user_name).first()
        if admin is None:
            admin = User(
                username=user_name,
                email=(email or "").strip() or None,
                password_hash=generate_password_hash(pwd),
                is_admin=1,
                is_active=1,
            )
            db.add(admin)
        else:
            changed = False
            if not int(admin.is_admin or 0):
                admin.is_admin = 1
                changed = True
            if not int(admin.is_active or 0):
                admin.is_active = 1
                changed = True
            if changed:
                db.add(admin)
        db.commit()
    finally:
        db.close()
