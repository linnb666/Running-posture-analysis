from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from io import BytesIO
import os
from html import escape
from pathlib import Path
from urllib.parse import quote
import uuid

from flask import Blueprint, current_app, g, jsonify, make_response, request, send_file, url_for
from werkzeug.utils import secure_filename

from backend.auth import auth_required
from backend.db import get_db
from backend.db_models import AnalysisRecord, AnalysisTask
from backend.services.data_cleanup import purge_record_with_related_tasks
from backend.services.report_pdf import build_report_pdf
from backend.services.task_queue import TaskQueue
from backend.utils import safe_json_loads
from modules.ai_analyzer import AIAnalyzer


analysis_bp = Blueprint("analysis", __name__)
task_queue = TaskQueue(max_workers=1)
AI_REQUEST_TIMEOUT_SECONDS = max(15, int(float(os.getenv("AI_REQUEST_TIMEOUT_SECONDS", "60"))))
AI_FALLBACK_TO_LOCAL = os.getenv("AI_FALLBACK_TO_LOCAL", "0").strip().lower() in {"1", "true", "yes", "on"}


def _grade_from_rating(rating: str | None) -> str | None:
    if not rating:
        return None
    return str(rating).lower()


def _even(value: int) -> int:
    return max(2, value if value % 2 == 0 else value - 1)


def _serialize_datetime(dt) -> str | None:
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _format_local_datetime(dt) -> str:
    if dt is None:
        return "--"
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _build_browser_pose_video(output_dir: Path, source: Path) -> Path | None:
    target = output_dir / "pose_overlay_web.mp4"
    if target.exists() and target.stat().st_size > 0:
        return target
    if not source.exists():
        return None

    try:
        import cv2  # lazy import to avoid heavy import cost on every request
    except Exception:
        return None

    temp_target = output_dir / "pose_overlay_web.tmp.mp4"
    try:
        temp_target.unlink(missing_ok=True)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None

    try:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        src_fps = int(round(cap.get(cv2.CAP_PROP_FPS) or 24))
        if src_w <= 0 or src_h <= 0:
            return None

        max_side = 960
        scale = min(1.0, float(max_side) / float(max(src_w, src_h)))
        out_w = _even(int(round(src_w * scale)))
        out_h = _even(int(round(src_h * scale)))
        out_fps = max(1, min(30, src_fps))

        writer = None
        for codec in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*codec)
            maybe = cv2.VideoWriter(str(temp_target), fourcc, out_fps, (out_w, out_h))
            if maybe.isOpened():
                writer = maybe
                break
            maybe.release()
        if writer is None:
            return None

        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
        writer.release()

        if written <= 0 or not temp_target.exists() or temp_target.stat().st_size <= 0:
            temp_target.unlink(missing_ok=True)
            return None
        temp_target.replace(target)
        return target
    except Exception:
        try:
            temp_target.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    finally:
        cap.release()


def _media_payload(record: AnalysisRecord):
    payload = {
        "original_video_url": None,
        "pose_video_url": None,
        "keyframes": [],
    }
    original_path = Path(record.original_video_path or "")
    output_dir = Path(record.output_dir or "")
    if original_path.exists():
        payload["original_video_url"] = url_for("analysis.media_original", record_id=record.id)

    pose_file = None
    raw_pose_candidate = None
    pose_name = record.pose_video_filename or ""
    if pose_name and output_dir.exists():
        candidate = output_dir / pose_name
        if candidate.exists():
            raw_pose_candidate = candidate
    if raw_pose_candidate is None and output_dir.exists():
        for pattern in ("pose_overlay.mp4", "pose_overlay.avi", "skeleton_3d.mp4", "pose_video.mp4", "skeleton.mp4"):
            candidates = sorted(output_dir.glob(pattern))
            if candidates:
                raw_pose_candidate = candidates[0]
                break

    if output_dir.exists():
        web_candidate = output_dir / "pose_overlay_web.mp4"
        if web_candidate.exists():
            pose_file = web_candidate
        elif raw_pose_candidate is not None:
            generated = _build_browser_pose_video(output_dir, raw_pose_candidate)
            if generated is not None and generated.exists():
                pose_file = generated
    if pose_file is None and raw_pose_candidate is not None and raw_pose_candidate.exists():
        pose_file = raw_pose_candidate

    if pose_file is not None and pose_file.exists():
        payload["pose_video_url"] = url_for(
            "analysis.media_output",
            record_id=record.id,
            filename=pose_file.name,
        )

    keyframes = safe_json_loads(record.keyframes_json, [])
    if not keyframes and output_dir.exists():
        keyframe_dir = output_dir / "keyframes"
        if keyframe_dir.exists():
            for kf in sorted(keyframe_dir.glob("*.jpg")):
                keyframes.append(
                    {
                        "name": kf.name,
                        "relpath": f"keyframes/{kf.name}",
                        "time_sec": None,
                        "detected": True,
                    }
                )
    for item in keyframes:
        relpath = item.get("relpath") or item.get("name")
        if not relpath:
            continue
        payload["keyframes"].append(
            {
                "name": item.get("name") or Path(relpath).name,
                "url": url_for("analysis.media_output", record_id=record.id, filename=relpath),
                "time_sec": item.get("time_sec"),
                "detected": item.get("detected"),
            }
        )
    return payload


def _record_to_public(record: AnalysisRecord):
    quality = safe_json_loads(record.quality_json, {})
    dimension_scores = safe_json_loads(record.dimension_scores_json, {})
    if not dimension_scores:
        dimension_scores = quality.get("dimension_scores", {})
    if record.view_angle == "front" and quality.get("frontal_dimension_scores"):
        dimension_scores = quality.get("frontal_dimension_scores", dimension_scores)

    kinematic = safe_json_loads(record.kinematic_json, {})
    temporal = safe_json_loads(record.temporal_json, {})
    strengths = safe_json_loads(record.strengths_json, [])
    weaknesses = safe_json_loads(record.weaknesses_json, [])
    suggestions = safe_json_loads(record.suggestions_json, [])
    grade = _grade_from_rating(record.rating)

    view_value = record.view_angle
    created_at = _serialize_datetime(record.created_at)
    updated_at = _serialize_datetime(record.updated_at)
    manual_notes_updated_at = _serialize_datetime(record.manual_notes_updated_at)

    payload = {
        "id": record.id,
        "task_id": record.task_id,
        "video_filename": record.video_filename,
        "view_angle": view_value,
        "created_at": created_at,
        "updated_at": updated_at,
        "manual_notes_updated_at": manual_notes_updated_at,
        "model_version": record.model_version,
        "total_score": record.total_score,
        "rating": record.rating,
        "grade": grade,
        "dimension_scores": dimension_scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "suggestions": suggestions,
        "kinematic_results": kinematic,
        "temporal_results": temporal,
        "quality_results": quality,
        "ai_analysis": record.ai_analysis or "",
        "manual_notes": record.manual_notes or "",
        "media": _media_payload(record),
        # compatibility aliases for old pages
        "view_type": view_value,
        "completed_at": created_at,
        "kinematic": kinematic,
        "temporal": temporal,
        "ai_report": record.ai_analysis or "",
        "status": "completed",
    }
    return payload


def _append_access_token(url: str | None, token: str) -> str:
    if not url:
        return ""
    token = (token or "").strip()
    if not token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}access_token={quote(token)}"


def _report_text_to_html(text: str | None) -> str:
    if not text or not str(text).strip():
        return '<p class="report-empty">暂无分析文本</p>'
    rendered = escape(str(text).strip()).replace("\n", "<br>")
    return f'<div class="report-pre">{rendered}</div>'


def _dimension_label(key: str) -> str:
    return {
        "stability": "稳定性",
        "efficiency": "效率",
        "form": "跑姿",
        "lower_limb_alignment": "下肢力线",
        "lateral_stability": "横向稳定性",
    }.get(key, key)


def _collect_highlight_metrics(record: AnalysisRecord, kinematic: dict) -> list[tuple[str, str]]:
    metrics: list[tuple[str, str]] = []
    if record.view_angle == "front":
        lower = kinematic.get("lower_limb_alignment", {})
        shoulder = kinematic.get("shoulder_analysis", {})
        cadence = kinematic.get("cadence", {})
        metrics.extend(
            [
                ("左膝偏移", f"{float(lower.get('left_leg', {}).get('mean', 0) or 0):.1f}°"),
                ("右膝偏移", f"{float(lower.get('right_leg', {}).get('mean', 0) or 0):.1f}°"),
                ("髋部下沉", f"{abs(float(lower.get('hip_drop', {}).get('mean', lower.get('hip_drop', {}).get('drop_mean', 0)) or 0)):.1f}°"),
                ("肩部倾斜", f"{abs(float(shoulder.get('tilt_mean', 0) or 0)):.1f}°"),
                ("步频", f"{float(cadence.get('cadence', 0) or 0):.0f} 步/分"),
            ]
        )
    else:
        cadence = kinematic.get("cadence", {})
        vertical = kinematic.get("vertical_motion", {})
        gait = kinematic.get("gait_cycle", {})
        stability = kinematic.get("stability", {})
        body_lean = kinematic.get("body_lean", {})
        metrics.extend(
            [
                ("步频", f"{float(cadence.get('cadence', 0) or 0):.0f} 步/分"),
                ("步数", str(int(cadence.get('step_count', 0) or 0))),
                ("垂直振幅", f"{float(vertical.get('amplitude_normalized', 0) or 0):.1f}%"),
                ("触地时间", f"{float(gait.get('phase_duration_ms', {}).get('ground_contact', 0) or 0):.0f} ms"),
                ("躯干前倾", f"{float(body_lean.get('forward_lean', body_lean.get('mean', 0)) or 0):.1f}°"),
                ("综合稳定性", f"{float(stability.get('overall', 0) or 0):.0f}/100"),
            ]
        )
    return [(label, value) for label, value in metrics if value not in {"0.0°", "0 步/分", "0 ms", "0.0%", "0/100", "0"}]


def _collect_keyframe_files(record: AnalysisRecord) -> list[dict]:
    items = []
    output_dir = Path(record.output_dir or "")
    keyframes = safe_json_loads(record.keyframes_json, [])
    if not keyframes and output_dir.exists():
        keyframe_dir = output_dir / "keyframes"
        if keyframe_dir.exists():
            for file_path in sorted(keyframe_dir.glob("*.jpg"))[:6]:
                keyframes.append({"name": file_path.name, "relpath": f"keyframes/{file_path.name}", "time_sec": None})

    for item in keyframes[:6]:
        relpath = item.get("relpath") or item.get("name")
        if not relpath:
            continue
        file_path = output_dir / relpath
        if not file_path.exists():
            continue
        time_value = item.get("time_sec")
        label = f"{float(time_value):.2f}s" if time_value is not None else (item.get("name") or Path(relpath).name)
        items.append({"path": str(file_path), "label": label})
    return items


def _build_report_context(record: AnalysisRecord) -> dict:
    payload = _record_to_public(record)
    quality = safe_json_loads(record.quality_json, {})
    kinematic = safe_json_loads(record.kinematic_json, {})
    temporal = safe_json_loads(record.temporal_json, {})
    local_report = AIAnalyzer().local_engine.generate_analysis_report(
        {
            "quality_evaluation": quality,
            "kinematic_analysis": kinematic,
            "temporal_analysis": temporal,
            "view_angle": record.view_angle,
        }
    )
    analysis_text = (record.ai_analysis or "").strip() or local_report
    analysis_source = "AI 智能分析" if (record.ai_analysis or "").strip() else "本地规则报告"
    return {
        "video_filename": record.video_filename,
        "task_id": record.task_id,
        "view_label": "正面视角" if record.view_angle == "front" else "侧面视角",
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "manual_notes_updated_at": payload.get("manual_notes_updated_at"),
        "model_version": record.model_version,
        "total_score": payload.get("total_score") or 0,
        "rating": payload.get("rating") or "--",
        "dimension_scores": {
            _dimension_label(str(key)): float(value or 0)
            for key, value in (payload.get("dimension_scores") or {}).items()
        },
        "metrics": _collect_highlight_metrics(record, kinematic),
        "strengths": payload.get("strengths") or [],
        "weaknesses": payload.get("weaknesses") or [],
        "suggestions": payload.get("suggestions") or [],
        "analysis_source": analysis_source,
        "analysis_text": analysis_text,
        "manual_notes": record.manual_notes or "",
        "keyframes": _collect_keyframe_files(record),
    }


def _build_report_preview_html(record: AnalysisRecord, access_token: str = "") -> str:
    payload = _record_to_public(record)
    quality = safe_json_loads(record.quality_json, {})
    kinematic = safe_json_loads(record.kinematic_json, {})
    temporal = safe_json_loads(record.temporal_json, {})
    dimension_scores = payload.get("dimension_scores", {}) or {}
    keyframes = payload.get("media", {}).get("keyframes", []) or []
    metrics = _collect_highlight_metrics(record, kinematic)
    local_report = AIAnalyzer().local_engine.generate_analysis_report(
        {
            "quality_evaluation": quality,
            "kinematic_analysis": kinematic,
            "temporal_analysis": temporal,
            "view_angle": record.view_angle,
        }
    )
    analysis_text = (record.ai_analysis or "").strip() or local_report
    analysis_source = "AI 智能分析" if (record.ai_analysis or "").strip() else "本地规则报告"

    dimension_cards = "".join(
        f"<div class='dim-card'><span>{escape(_dimension_label(str(key)))}</span><strong>{float(value or 0):.1f}</strong></div>"
        for key, value in dimension_scores.items()
    ) or "<div class='empty-box'>暂无维度评分</div>"

    metrics_html = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
        for label, value in metrics
    ) or "<tr><td colspan='2'>暂无核心指标</td></tr>"

    keyframe_parts = []
    for item in keyframes[:6]:
        time_label = f"{float(item.get('time_sec') or 0):.2f}s"
        keyframe_parts.append(
            f"<figure class='frame-card'><img src='{escape(_append_access_token(item.get('url'), access_token))}' alt='关键帧'><figcaption>{escape(time_label)}</figcaption></figure>"
        )
    keyframes_html = "".join(keyframe_parts) or "<div class='empty-box'>暂无关键帧图像</div>"

    notes_html = ""
    if (record.manual_notes or "").strip():
        notes_html = f"""
        <section class='notes-section'>
          <h2>人工备注</h2>
          <div class='notes-card'>{_report_text_to_html(record.manual_notes)}</div>
        </section>
        """

    created_text = _format_local_datetime(record.created_at)
    total_score = float(payload.get("total_score") or 0)
    rating = payload.get("rating") or "--"
    view_label = "正面视角" if record.view_angle == "front" else "侧面视角"
    strengths_text = '; '.join(payload.get('strengths') or []) or "暂无"
    weaknesses_text = '; '.join(payload.get('weaknesses') or []) or "暂无"
    suggestions_text = '; '.join(payload.get('suggestions') or []) or "暂无"

    return f"""
<!doctype html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>跑步动作分析报告 - {escape(record.video_filename)}</title>
  <style>
    @page {{ size: A4; margin: 14mm; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Microsoft YaHei', 'PingFang SC', sans-serif; color: #1f2937; background: #f3f0ea; }}
    .page {{ max-width: 794px; margin: 0 auto; background: #fffdf9; min-height: 100vh; padding: 28px 30px 36px; }}
    .hero {{ display: flex; justify-content: space-between; gap: 18px; padding: 22px 24px; border-radius: 22px; background: linear-gradient(135deg, #fff6e8 0%, #f7efe4 55%, #efe4d5 100%); border: 1px solid rgba(191, 145, 80, 0.18); }}
    .hero h1 {{ margin: 0 0 10px; font-size: 28px; }}
    .hero p {{ margin: 4px 0; color: #6b7280; font-size: 13px; }}
    .score-box {{ min-width: 176px; border-radius: 18px; background: rgba(255,255,255,0.82); padding: 18px; text-align: center; border: 1px solid rgba(191,145,80,0.18); }}
    .score-box strong {{ display: block; font-size: 40px; line-height: 1; margin-bottom: 8px; color: #9a5d1a; }}
    .chip {{ display: inline-block; padding: 5px 10px; border-radius: 999px; background: #fff; border: 1px solid rgba(154,93,26,0.14); font-size: 12px; color: #8a5b26; }}
    section {{ margin-top: 22px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .grid {{ display: grid; gap: 12px; }}
    .dim-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .dim-card, .panel, .notes-card {{ border: 1px solid #ebe3d7; border-radius: 16px; background: #fff; }}
    .dim-card {{ padding: 14px 16px; }}
    .dim-card span {{ display: block; font-size: 12px; color: #8b8f97; margin-bottom: 6px; }}
    .dim-card strong {{ font-size: 24px; color: #2f2f2f; }}
    .panel {{ padding: 16px 18px; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .metrics-table td {{ padding: 10px 8px; border-bottom: 1px solid #f0e7d8; }}
    .metrics-table td:first-child {{ width: 36%; color: #6b7280; }}
    .frames-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .frame-card {{ margin: 0; border-radius: 14px; overflow: hidden; background: #0f172a; border: 1px solid #e7dcc9; }}
    .frame-card img {{ display: block; width: 100%; aspect-ratio: 16/9; object-fit: contain; background: #000; }}
    .frame-card figcaption {{ padding: 8px 10px; font-size: 12px; color: #f5f5f5; text-align: center; }}
    .report-pre {{ white-space: normal; font-size: 13px; line-height: 1.85; color: #374151; }}
    .report-empty, .empty-box {{ color: #9ca3af; font-size: 13px; }}
    .empty-box {{ padding: 14px 0; }}
    .split {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 14px; }}
    .toolbar {{ position: sticky; top: 0; background: rgba(243,240,234,0.92); backdrop-filter: blur(8px); padding: 14px 0 8px; margin-bottom: 10px; }}
    .toolbar button {{ border: 0; border-radius: 999px; padding: 10px 16px; background: #9a5d1a; color: #fff; font-weight: 600; cursor: pointer; }}
    .toolbar span {{ margin-left: 10px; font-size: 12px; color: #6b7280; }}
    .report-meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 18px; margin-top: 8px; }}
    .report-meta div {{ font-size: 13px; color: #5b616b; }}
    .report-meta strong {{ color: #111827; margin-right: 6px; }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ padding: 0; max-width: none; background: #fff; }}
      .toolbar {{ display: none; }}
      .hero, .dim-card, .panel, .notes-card, .frame-card {{ break-inside: avoid; box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <div class='page'>
    <div class='toolbar'>
      <button type='button' onclick='window.print()'>打印 / 另存为 PDF</button>
      <span>建议在打印目标中选择“另存为 PDF”</span>
    </div>
    <section class='hero'>
      <div>
        <div class='chip'>跑步动作分析系统报告</div>
        <h1>{escape(record.video_filename)}</h1>
        <div class='report-meta'>
          <div><strong>任务ID</strong>{escape(record.task_id or '--')}</div>
          <div><strong>分析视角</strong>{escape(view_label)}</div>
          <div><strong>完成时间</strong>{escape(created_text)}</div>
          <div><strong>报告来源</strong>{escape(analysis_source)}</div>
        </div>
      </div>
      <div class='score-box'>
        <strong>{total_score:.1f}</strong>
        <div>{escape(str(rating))}</div>
      </div>
    </section>

    <section>
      <h2>维度评分</h2>
      <div class='grid dim-grid'>{dimension_cards}</div>
    </section>

    <section class='split'>
      <div class='panel'>
        <h2>关键指标</h2>
        <table class='metrics-table'>{metrics_html}</table>
      </div>
      <div class='panel'>
        <h2>分析结论</h2>
        <p><strong>优势：</strong>{escape(strengths_text)}</p>
        <p><strong>待改进：</strong>{escape(weaknesses_text)}</p>
        <p><strong>建议：</strong>{escape(suggestions_text)}</p>
      </div>
    </section>

    <section>
      <h2>关键帧骨架</h2>
      <div class='frames-grid'>{keyframes_html}</div>
    </section>

    <section>
      <h2>{escape(analysis_source)}</h2>
      <div class='panel'>{_report_text_to_html(analysis_text)}</div>
    </section>

    {notes_html}
  </div>
  <script>
    const params = new URLSearchParams(window.location.search);
    if (params.get('autoprint') === '1') {{
      window.addEventListener('load', () => setTimeout(() => window.print(), 500));
    }}
  </script>
</body>
</html>
"""


def _task_to_public(task: AnalysisTask):
    return {
        "task_id": task.id,
        "status": task.status,
        "progress": int(task.progress or 0),
        "stage": task.stage or "",
        "error_message": task.error_message,
        "view_angle": task.view_angle,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
    }


def _cleanup_roots() -> list[Path]:
    roots = []
    for key in ("OUTPUT_DIR", "UPLOAD_DIR"):
        value = current_app.config.get(key)
        if value:
            roots.append(Path(value))
    return roots


def _output_root() -> Path:
    return Path(current_app.config["OUTPUT_DIR"])


@analysis_bp.post("/upload")
@auth_required
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "未上传视频文件"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "文件名为空"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in current_app.config["ALLOWED_VIDEO_EXTENSIONS"]:
        return jsonify({"success": False, "error": "不支持的视频格式"}), 400

    view_angle = (request.form.get("view_angle") or "side").strip().lower()
    if view_angle not in {"side", "front"}:
        return jsonify({"success": False, "error": "view_angle 必须是 side/front"}), 400

    enable_3d = str(request.form.get("enable_3d", "true")).lower() != "false"
    task_id = uuid.uuid4().hex
    task_output_dir = Path(current_app.config["OUTPUT_DIR"]) / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = secure_filename(Path(file.filename).name) or f"video{suffix}"
    input_path = task_output_dir / f"input_{safe_name}"
    file.save(input_path)

    db = get_db()
    try:
        task = AnalysisTask(
            id=task_id,
            user_id=g.user_id,
            status="queued",
            progress=0,
            stage="已接收",
            view_angle=view_angle,
            enable_3d=1 if enable_3d else 0,
            input_video_path=str(input_path),
        )
        db.add(task)
        db.commit()
    finally:
        db.close()

    try:
        task_queue.submit(
            Path(current_app.config["PROJECT_ROOT"]),
            task_id,
            g.user_id,
            input_path,
            view_angle,
            enable_3d,
            Path(current_app.config["OUTPUT_DIR"]),
        )
    except ModuleNotFoundError as exc:
        db = get_db()
        try:
            task = db.get(AnalysisTask, task_id)
            if task is not None:
                task.status = "failed"
                task.error_message = f"运行环境缺少依赖: {exc}"
                task.stage = "失败"
                db.commit()
        finally:
            db.close()
        return jsonify({"success": False, "error": f"后端依赖缺失: {exc}"}), 500
    except Exception as exc:
        db = get_db()
        try:
            task = db.get(AnalysisTask, task_id)
            if task is not None:
                task.status = "failed"
                task.error_message = str(exc)
                task.stage = "失败"
                db.commit()
        finally:
            db.close()
        return jsonify({"success": False, "error": f"任务提交失败: {exc}"}), 500

    return jsonify(
        {
            "success": True,
            "task_id": task_id,
            "status": "queued",
            "message": "视频已上传，开始分析",
        }
    )


@analysis_bp.get("/task/<task_id>")
@auth_required
def task_status(task_id: str):
    db = get_db()
    try:
        task = (
            db.query(AnalysisTask)
            .filter(AnalysisTask.id == task_id, AnalysisTask.user_id == g.user_id)
            .first()
        )
        if task is None:
            return jsonify({"success": False, "error": "任务不存在"}), 404

        payload = {"success": True, **_task_to_public(task)}
        if task.status == "succeeded":
            record_id = task.result_record_id
            if not record_id:
                record = (
                    db.query(AnalysisRecord)
                    .filter(AnalysisRecord.task_id == task.id, AnalysisRecord.user_id == g.user_id)
                    .first()
                )
                if record:
                    record_id = record.id
            if record_id:
                payload["result"] = {"record_id": record_id}
        return jsonify(payload)
    finally:
        db.close()


@analysis_bp.get("/result/<int:record_id>")
@auth_required
def result(record_id: int):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404
        return jsonify({"success": True, "record": _record_to_public(record)})
    finally:
        db.close()


@analysis_bp.post("/result/<int:record_id>/notes")
@auth_required
def save_manual_notes(record_id: int):
    payload = request.get_json(silent=True) or {}
    notes = str(payload.get("manual_notes", payload.get("notes", "")) or "").strip()
    if len(notes) > 5000:
        return jsonify({"success": False, "error": "备注长度不能超过 5000 个字符"}), 400

    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404
        record.manual_notes = notes
        record.manual_notes_updated_at = datetime.utcnow()
        db.commit()
        return jsonify({
            "success": True,
            "manual_notes": record.manual_notes or "",
            "updated_at": _serialize_datetime(record.updated_at),
            "manual_notes_updated_at": _serialize_datetime(record.manual_notes_updated_at),
        })
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@analysis_bp.get("/result/<int:record_id>/pdf-preview")
@auth_required
def pdf_preview(record_id: int):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404
        html = _build_report_preview_html(record, request.args.get("access_token", ""))
        response = make_response(html)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        response.headers["Cache-Control"] = "no-store"
        return response
    finally:
        db.close()


@analysis_bp.get("/result/<int:record_id>/pdf")
@auth_required
def download_pdf(record_id: int):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404
        report_bytes = build_report_pdf(_build_report_context(record))
        stem = Path(record.video_filename or f"record_{record.id}").stem or f"record_{record.id}"
        download_name = f"{secure_filename(stem) or f'record_{record.id}'}_analysis_report.pdf"
        return send_file(
            BytesIO(report_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=download_name,
            max_age=0,
        )
    finally:
        db.close()


@analysis_bp.post("/result/<int:record_id>/ai")
@auth_required
def generate_ai(record_id: int):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404

        quality = safe_json_loads(record.quality_json, {})
        kinematic = safe_json_loads(record.kinematic_json, {})
        temporal = safe_json_loads(record.temporal_json, {})
        payload = {
            "quality_evaluation": quality,
            "kinematic_analysis": kinematic,
            "temporal_analysis": temporal,
            "view_angle": record.view_angle,
        }
        analyzer = AIAnalyzer()
        if analyzer.provider_name == "local" and not AI_FALLBACK_TO_LOCAL:
            return jsonify({"success": False, "error": "AI服务未配置或不可用，请检查API密钥"}), 503

        # Avoid hanging UI when external AI provider is slow/unreachable.
        ex = ThreadPoolExecutor(max_workers=1)
        source = "ai"
        try:
            future = ex.submit(
                analyzer.generate_analysis_report,
                payload,
                AI_FALLBACK_TO_LOCAL,
            )
            ai_text = future.result(timeout=AI_REQUEST_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            try:
                future.cancel()
            except Exception:
                pass
            if AI_FALLBACK_TO_LOCAL:
                ai_text = analyzer.local_engine.generate_analysis_report(payload)
                source = "local_fallback_timeout"
            else:
                return jsonify(
                    {
                        "success": False,
                        "error": f"AI分析超时（>{AI_REQUEST_TIMEOUT_SECONDS}s），请稍后重试",
                    }
                ), 504
        except Exception:
            if AI_FALLBACK_TO_LOCAL:
                ai_text = analyzer.local_engine.generate_analysis_report(payload)
                source = "local_fallback_error"
            else:
                return jsonify({"success": False, "error": "AI服务调用失败，请稍后重试"}), 502
        finally:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        if not ai_text or not str(ai_text).strip():
            if AI_FALLBACK_TO_LOCAL:
                ai_text = analyzer.local_engine.generate_analysis_report(payload)
                source = "local_fallback_empty"
            else:
                return jsonify({"success": False, "error": "AI分析未返回有效内容"}), 502
        record.ai_analysis = ai_text
        db.commit()
        return jsonify({"success": True, "ai_analysis": ai_text, "source": source})
    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@analysis_bp.post("/result/<int:record_id>/local-report")
@auth_required
def generate_local_report(record_id: int):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404

        quality = safe_json_loads(record.quality_json, {})
        kinematic = safe_json_loads(record.kinematic_json, {})
        temporal = safe_json_loads(record.temporal_json, {})
        payload = {
            "quality_evaluation": quality,
            "kinematic_analysis": kinematic,
            "temporal_analysis": temporal,
            "view_angle": record.view_angle,
        }
        analyzer = AIAnalyzer()
        local_text = analyzer.local_engine.generate_analysis_report(payload)
        if not local_text or not str(local_text).strip():
            return jsonify({"success": False, "error": "未生成有效本地报告"}), 500

        return jsonify({"success": True, "local_analysis": local_text})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        db.close()


@analysis_bp.get("/history")
@auth_required
def history():
    page = max(1, int(request.args.get("page", 1)))
    per_page = max(1, min(50, int(request.args.get("per_page", request.args.get("limit", 20)))))
    view_type = (request.args.get("view_type") or request.args.get("view_angle") or "").strip().lower()

    db = get_db()
    try:
        query = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == g.user_id)
        if view_type in {"side", "front"}:
            query = query.filter(AnalysisRecord.view_angle == view_type)
        total = query.count()
        items = (
            query.order_by(AnalysisRecord.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        records = []
        for item in items:
            records.append(
                {
                    "id": item.id,
                    "task_id": item.task_id,
                    "video_filename": item.video_filename,
                    "view_angle": item.view_angle,
                    "view_type": item.view_angle,
                    "total_score": item.total_score,
                    "rating": item.rating,
                    "grade": _grade_from_rating(item.rating),
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                    "status": "completed",
                }
            )
        return jsonify(
            {
                "success": True,
                "records": records,
                "total": total,
                "page": page,
                "per_page": per_page,
                "limit": per_page,
            }
        )
    finally:
        db.close()


@analysis_bp.delete("/result/<int:record_id>")
@auth_required
def delete_result(record_id: int):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404

        summary = purge_record_with_related_tasks(
            db,
            record,
            _cleanup_roots(),
            output_root=_output_root(),
            restrict_user_id=g.user_id,
        )
        db.commit()
        return jsonify(
            {
                "success": True,
                "message": "记录已删除",
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


@analysis_bp.post("/result/<int:record_id>/rename")
@auth_required
def rename_result(record_id: int):
    payload = request.get_json(silent=True) or {}
    new_name = (payload.get("video_filename") or payload.get("name") or "").strip()
    if not new_name:
        return jsonify({"success": False, "error": "新名称不能为空"}), 400
    if len(new_name) > 255:
        return jsonify({"success": False, "error": "名称长度不能超过255"}), 400

    invalid_chars = set('\\/:*?"<>|')
    sanitized = "".join(ch for ch in new_name if ch not in invalid_chars).strip()
    if not sanitized:
        return jsonify({"success": False, "error": "名称包含非法字符"}), 400

    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404

        record.video_filename = sanitized
        db.commit()
        return jsonify(
            {
                "success": True,
                "record": {
                    "id": record.id,
                    "video_filename": record.video_filename,
                    "updated_at": record.updated_at.isoformat() if record.updated_at else None,
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


@analysis_bp.get("/statistics")
@auth_required
def statistics():
    db = get_db()
    try:
        rows = db.query(AnalysisRecord).filter(AnalysisRecord.user_id == g.user_id).all()
        total = len(rows)
        avg = round(sum((x.total_score or 0) for x in rows) / total, 2) if total else 0
        score_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        view_counts = {"side": 0, "front": 0}
        for row in rows:
            grade = _grade_from_rating(row.rating)
            if grade in score_distribution:
                score_distribution[grade] += 1
            if row.view_angle in view_counts:
                view_counts[row.view_angle] += 1
        return jsonify(
            {
                "success": True,
                "statistics": {
                    "total_analyses": total,
                    "completed_analyses": total,
                    "average_score": avg,
                    "score_distribution": score_distribution,
                    "view_type_counts": view_counts,
                    "view_distribution": view_counts,
                    "recent_trend": [],
                },
            }
        )
    finally:
        db.close()


@analysis_bp.get("/media/original/<int:record_id>")
@auth_required
def media_original(record_id: int):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404
        file_path = Path(record.original_video_path or "")
        if not file_path.exists():
            return jsonify({"success": False, "error": "原视频不存在"}), 404
        return send_file(file_path)
    finally:
        db.close()


@analysis_bp.get("/media/output/<int:record_id>/<path:filename>")
@auth_required
def media_output(record_id: int, filename: str):
    db = get_db()
    try:
        record = (
            db.query(AnalysisRecord)
            .filter(AnalysisRecord.id == record_id, AnalysisRecord.user_id == g.user_id)
            .first()
        )
        if record is None:
            return jsonify({"success": False, "error": "记录不存在"}), 404
        output_dir = Path(record.output_dir or "")
        if not output_dir.exists():
            return jsonify({"success": False, "error": "输出目录不存在"}), 404
        clean = filename.replace("\\", "/")
        file_path = (output_dir / clean).resolve()
        if output_dir.resolve() not in file_path.parents and file_path != output_dir.resolve():
            return jsonify({"success": False, "error": "非法路径"}), 400
        if not file_path.exists():
            return jsonify({"success": False, "error": "文件不存在"}), 404
        return send_file(file_path)
    finally:
        db.close()

