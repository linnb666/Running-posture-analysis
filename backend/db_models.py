from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from backend.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=True, index=True)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Integer, nullable=False, default=0, index=True)
    is_active = Column(Integer, nullable=False, default=1, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    tasks = relationship("AnalysisTask", back_populates="user", cascade="all, delete-orphan")
    records = relationship("AnalysisRecord", back_populates="user", cascade="all, delete-orphan")
    admin_logs = relationship(
        "AdminAuditLog",
        back_populates="admin_user",
        foreign_keys="AdminAuditLog.admin_user_id",
    )


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    status = Column(String(32), nullable=False, index=True, default="queued")
    progress = Column(Integer, default=0, nullable=False)
    stage = Column(String(255), nullable=True)
    error_message = Column(Text, nullable=True)

    view_angle = Column(String(16), nullable=False, default="side")
    enable_3d = Column(Integer, nullable=False, default=1)
    input_video_path = Column(Text, nullable=True)
    result_record_id = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="tasks")
    record = relationship("AnalysisRecord", back_populates="task", uselist=False)


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    task_id = Column(String(64), ForeignKey("analysis_tasks.id"), nullable=True, index=True)

    video_filename = Column(String(255), nullable=False)
    video_hash = Column(String(128), nullable=True, index=True)
    video_info_json = Column(Text, nullable=True)

    original_video_path = Column(Text, nullable=True)
    output_dir = Column(Text, nullable=True)
    pose_video_filename = Column(String(255), nullable=True)
    keyframes_json = Column(Text, nullable=True)

    view_angle = Column(String(16), nullable=False)
    enable_3d = Column(Integer, default=1, nullable=False)
    model_version = Column(String(64), nullable=True)
    model_checksum = Column(String(128), nullable=True)
    config_json = Column(Text, nullable=True)
    git_commit = Column(String(64), nullable=True)

    total_score = Column(Float, nullable=True)
    rating = Column(String(32), nullable=True)
    dimension_scores_json = Column(Text, nullable=True)
    strengths_json = Column(Text, nullable=True)
    weaknesses_json = Column(Text, nullable=True)
    suggestions_json = Column(Text, nullable=True)

    kinematic_json = Column(Text, nullable=True)
    temporal_json = Column(Text, nullable=True)
    quality_json = Column(Text, nullable=True)
    ai_analysis = Column(Text, nullable=True)
    manual_notes = Column(Text, nullable=True)
    manual_notes_updated_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="records")
    task = relationship("AnalysisTask", back_populates="record")


class AdminAuditLog(Base):
    __tablename__ = "admin_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    admin_user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(64), nullable=False, index=True)
    target_user_id = Column(Integer, nullable=True, index=True)
    target_record_id = Column(Integer, nullable=True, index=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    admin_user = relationship("User", foreign_keys=[admin_user_id], back_populates="admin_logs")
