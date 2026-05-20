from __future__ import annotations

import io
import sys
import time
import traceback
import builtins
from pathlib import Path

import cv2
from sqlalchemy.exc import OperationalError

from config.config import MOTIONBERT_CONFIG, POSE_CONFIG
from modules.ai_analyzer import AIAnalyzer
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.pose_estimator import create_pose_estimator, create_pose_estimator_3d
from modules.quality_evaluator import QualityEvaluator
from modules.video_processor import VideoProcessor

from backend.config import BackendConfig
from backend.db import get_db
from backend.db_models import AnalysisRecord, AnalysisTask
from backend.utils import get_git_commit, safe_json_dumps, sha256_file


_MODEL_CHECKSUM_CACHE: str | None = None


def _disabled_temporal_result() -> dict:
    """时序模块停用占位结果（保留兼容字段）。"""
    return {
        "disabled": True,
        "status": "deprecated",
        "model_type": "disabled",
        "note": "时序模型已停用，不参与当前分析与评分流程",
        "phase_distribution": {},
    }


class _SafeTextStream(io.TextIOBase):
    def __init__(self, fallback):
        super().__init__()
        self.fallback = fallback

    def write(self, s):
        text = "" if s is None else str(s)
        try:
            if self.fallback is not None:
                self.fallback.write(text)
        except Exception:
            # Ignore invalid console handles / encoding failures in detached mode.
            pass
        return len(text)

    def flush(self):
        try:
            if self.fallback is not None:
                self.fallback.flush()
        except Exception:
            pass


def _model_checksum() -> str:
    global _MODEL_CHECKSUM_CACHE
    if not bool(BackendConfig.ENABLE_MODEL_CHECKSUM):
        return ""
    if _MODEL_CHECKSUM_CACHE is not None:
        return _MODEL_CHECKSUM_CACHE
    ckpt = MOTIONBERT_CONFIG.get("checkpoint_path")
    if not ckpt:
        _MODEL_CHECKSUM_CACHE = ""
        return _MODEL_CHECKSUM_CACHE
    path = Path(ckpt)
    if not path.exists():
        _MODEL_CHECKSUM_CACHE = ""
        return _MODEL_CHECKSUM_CACHE
    _MODEL_CHECKSUM_CACHE = sha256_file(path)
    return _MODEL_CHECKSUM_CACHE


def _update_task_state(task_id: str, retries: int = 5, **fields) -> bool:
    for attempt in range(retries):
        db = get_db()
        try:
            task = db.get(AnalysisTask, task_id)
            if task is None:
                return False
            for key, value in fields.items():
                if value is None:
                    continue
                if key == "progress":
                    current = int(task.progress or 0)
                    task.progress = max(current, int(value))
                else:
                    setattr(task, key, value)
            db.commit()
            return True
        except OperationalError:
            try:
                db.rollback()
            except Exception:
                pass
            time.sleep(0.2 * (attempt + 1))
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
            return False
        finally:
            db.close()
    return False


def _create_pose_video(frames, keypoints_sequence, fps: float, estimator, output_dir: Path) -> str | None:
    if not frames or not keypoints_sequence:
        return None
    h, w = frames[0].shape[:2]
    fps_int = max(1, int(round(fps or 24)))
    codecs = [("avc1", ".mp4"), ("mp4v", ".mp4"), ("XVID", ".avi")]
    writer = None
    output_path = None

    for codec, ext in codecs:
        test_path = output_dir / f"pose_overlay{ext}"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        maybe_writer = cv2.VideoWriter(str(test_path), fourcc, fps_int, (w, h))
        if maybe_writer.isOpened():
            writer = maybe_writer
            output_path = test_path
            break
        maybe_writer.release()

    web_writer = None
    web_output_path = output_dir / "pose_overlay_web.mp4"
    web_fps = max(1, min(30, fps_int))

    def _even(v: int) -> int:
        return max(2, v if v % 2 == 0 else v - 1)

    max_side = 960
    scale = min(1.0, float(max_side) / float(max(w, h)))
    web_w = _even(int(round(w * scale)))
    web_h = _even(int(round(h * scale)))
    if web_w <= 0 or web_h <= 0:
        web_w, web_h = _even(w), _even(h)

    for codec in ("mp4v", "avc1", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        maybe_writer = cv2.VideoWriter(str(web_output_path), fourcc, web_fps, (web_w, web_h))
        if maybe_writer.isOpened():
            web_writer = maybe_writer
            break
        maybe_writer.release()

    if writer is None and web_writer is None:
        return None

    for frame, kp in zip(frames, keypoints_sequence):
        if isinstance(kp, dict) and kp.get("detected", False):
            vis = estimator.visualize_pose(frame, kp)
        else:
            vis = frame.copy()
        if writer is not None:
            writer.write(vis)
        if web_writer is not None:
            if vis.shape[1] != web_w or vis.shape[0] != web_h:
                resized = cv2.resize(vis, (web_w, web_h), interpolation=cv2.INTER_AREA)
            else:
                resized = vis
            web_writer.write(resized)

    if writer is not None:
        writer.release()
    if web_writer is not None:
        web_writer.release()

    if output_path is not None and output_path.exists():
        return output_path.name
    if web_output_path.exists():
        return web_output_path.name
    return None


def _extract_keyframes_with_pose(frames, keypoints_sequence, fps: float, estimator, output_dir: Path, num_keyframes: int = 6):
    if not frames or not keypoints_sequence:
        return []
    keyframes_dir = output_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    total = len(frames)
    if total <= num_keyframes:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (num_keyframes - 1)) for i in range(num_keyframes)]

    saved = []
    now = int(time.time())
    for i, idx in enumerate(indices):
        frame = frames[idx]
        kp = keypoints_sequence[idx]
        if isinstance(kp, dict) and kp.get("detected", False):
            vis_frame = estimator.visualize_pose(frame.copy(), kp)
        else:
            vis_frame = frame.copy()
            cv2.putText(
                vis_frame,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        time_sec = idx / max(1.0, float(fps or 1.0))
        cv2.putText(
            vis_frame,
            f"Time: {time_sec:.2f}s",
            (10, vis_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        name = f"keyframe_{now}_{i:02d}.jpg"
        out_file = keyframes_dir / name
        cv2.imwrite(str(out_file), vis_frame)
        saved.append(
            {
                "name": name,
                "relpath": f"keyframes/{name}",
                "frame_idx": idx,
                "time_sec": time_sec,
                "detected": bool(isinstance(kp, dict) and kp.get("detected", False)),
            }
        )
    return saved


def run_single_task(
    project_root: Path,
    task_id: str,
    user_id: int,
    video_path: Path,
    view_angle: str,
    enable_3d: bool,
    output_root: Path,
) -> None:
    processor = None
    estimator = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_print = builtins.print
    patched_module_prints = []
    try:
        def _safe_print(*args, **kwargs):
            try:
                return original_print(*args, **kwargs)
            except Exception:
                return None

        builtins.print = _safe_print
        # Some engine modules resolve `print` via module globals.
        # Explicitly patch these module-level symbols to avoid detached-console failures.
        try:
            import modules.kinematic_analyzer as _ka_mod

            prev = getattr(_ka_mod, "print", None)
            setattr(_ka_mod, "print", _safe_print)
            patched_module_prints.append((_ka_mod, prev))
        except Exception:
            pass
        # Detached Windows processes may have invalid stdio handles; swallow all prints from engine code.
        sys.stdout = _SafeTextStream(original_stdout)
        sys.stderr = _SafeTextStream(original_stderr)

        _update_task_state(
            task_id,
            status="running",
            progress=5,
            stage="准备分析资源",
            error_message=None,
        )

        output_dir = output_root / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        _update_task_state(task_id, progress=12, stage="1/7 视频预处理中")
        processor = VideoProcessor(str(video_path))
        video_info = processor.get_video_info()

        video_duration = float(video_info.get("duration", 0) or 0)
        video_fps = float(video_info.get("fps", 30.0) or 30.0)
        target_duration = min(10.0, video_duration) if video_duration > 0 else 0
        if video_duration > 10.0:
            start_time = (video_duration - target_duration) / 2
            start_frame = int(start_time * video_fps)
            max_frames = int(target_duration * video_fps)
        else:
            start_frame = 0
            max_frames = int(video_duration * video_fps) if video_duration > 0 else 0

        frames, fps = processor.extract_frames_from_position(
            start_frame=start_frame,
            target_fps=video_fps,
            max_frames=max_frames,
        )
        if not frames:
            raise RuntimeError("视频帧提取失败，无法继续分析")

        _update_task_state(task_id, progress=24, stage="2/7 姿态估计中")
        keypoints_sequence = None
        keypoints_3d = None
        poses_3d = None
        has_3d = False

        if enable_3d:
            try:
                estimator = create_pose_estimator_3d(
                    backend_2d=POSE_CONFIG["backend"],
                    enable_3d=True,
                    device="auto",
                )
                pose_result = estimator.process_frames(
                    frames,
                    lift_to_3d=True,
                    view_angle=view_angle,
                )
                keypoints_sequence = pose_result["keypoints_2d"]
                keypoints_3d = pose_result.get("keypoints_3d")
                poses_3d = pose_result.get("poses_3d")
                has_3d = bool(pose_result.get("has_3d", False))
            except Exception:
                estimator = create_pose_estimator(POSE_CONFIG["backend"], POSE_CONFIG)
                keypoints_sequence = estimator.process_frames(frames)
        else:
            estimator = create_pose_estimator(POSE_CONFIG["backend"], POSE_CONFIG)
            keypoints_sequence = estimator.process_frames(frames)

        detected_count = sum(1 for kp in keypoints_sequence if isinstance(kp, dict) and kp.get("detected"))
        if detected_count == 0:
            raise RuntimeError("未检测到有效人体姿态关键点")

        _update_task_state(task_id, progress=34, stage="2/7 生成姿态可视化")
        pose_video_filename = _create_pose_video(frames, keypoints_sequence, fps, estimator, output_dir)
        keyframes = _extract_keyframes_with_pose(frames, keypoints_sequence, fps, estimator, output_dir, num_keyframes=6)

        _update_task_state(task_id, progress=42, stage="3/7 视角设置中")
        detected_view = view_angle

        _update_task_state(task_id, progress=58, stage="4/7 运动学分析中")
        kinematic = KinematicAnalyzer().analyze_sequence(
            keypoints_sequence,
            fps,
            view_angle=detected_view,
            poses_3d=poses_3d,
            keypoints_3d=keypoints_3d,
        )

        _update_task_state(task_id, progress=72, stage="5/7 时序模块已停用")
        temporal = _disabled_temporal_result()

        _update_task_state(task_id, progress=84, stage="6/7 质量评价中")
        quality = QualityEvaluator().evaluate(
            kinematic,
            temporal,
            view_angle=detected_view,
        )

        ai_text = ""
        if BackendConfig.AUTO_AI_ANALYSIS:
            _update_task_state(task_id, progress=92, stage="7/7 AI报告处理中")
            ai_payload = {
                "quality_evaluation": quality,
                "kinematic_analysis": kinematic,
                "temporal_analysis": temporal,
                "view_angle": detected_view,
            }
            ai_text = AIAnalyzer().generate_analysis_report(ai_payload)
        else:
            _update_task_state(task_id, progress=92, stage="7/7 AI报告待手动触发")

        dimension_scores = quality.get("dimension_scores", {}) or {}
        if detected_view == "front" and quality.get("frontal_dimension_scores"):
            dimension_scores = quality.get("frontal_dimension_scores", dimension_scores)

        _update_task_state(task_id, status="persisting", progress=96, stage="写入分析结果")
        db = get_db()
        try:
            record = AnalysisRecord(
                user_id=user_id,
                task_id=task_id,
                video_filename=video_path.name,
                video_hash=sha256_file(video_path),
                video_info_json=safe_json_dumps(video_info),
                original_video_path=str(video_path),
                output_dir=str(output_dir),
                pose_video_filename=pose_video_filename,
                keyframes_json=safe_json_dumps(keyframes),
                view_angle=detected_view,
                enable_3d=1 if enable_3d else 0,
                model_version="thesis-v2",
                model_checksum=_model_checksum(),
                config_json=safe_json_dumps({"view_angle": detected_view, "enable_3d": enable_3d}),
                git_commit=get_git_commit(project_root),
                total_score=quality.get("total_score"),
                rating=quality.get("rating"),
                dimension_scores_json=safe_json_dumps(dimension_scores),
                strengths_json=safe_json_dumps(quality.get("strengths", [])),
                weaknesses_json=safe_json_dumps(quality.get("weaknesses", [])),
                suggestions_json=safe_json_dumps(quality.get("suggestions", [])),
                kinematic_json=safe_json_dumps(kinematic),
                temporal_json=safe_json_dumps(temporal),
                quality_json=safe_json_dumps(quality),
                ai_analysis=ai_text,
            )
            db.add(record)
            db.flush()

            task = db.get(AnalysisTask, task_id)
            if task is not None:
                task.status = "succeeded"
                task.progress = 100
                task.stage = "完成"
                task.error_message = None
                task.result_record_id = record.id
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        err = f"{exc}\n{traceback.format_exc()}"
        _update_task_state(
            task_id,
            status="failed",
            stage="失败",
            error_message=err,
        )
    finally:
        for mod, prev in patched_module_prints:
            try:
                if prev is None:
                    delattr(mod, "print")
                else:
                    setattr(mod, "print", prev)
            except Exception:
                pass
        try:
            builtins.print = original_print
        except Exception:
            pass
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        except Exception:
            pass
        try:
            if processor is not None:
                processor.release()
        except Exception:
            pass
        try:
            if estimator is not None:
                estimator.close()
        except Exception:
            pass
