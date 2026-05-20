import streamlit as st
import sys
from pathlib import Path
import cv2
import tempfile
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import STREAMLIT_CONFIG, POSE_CONFIG, VIEW_DETECTION_CONFIG, MOTIONBERT_CONFIG
from modules.video_processor import VideoProcessor
from modules.pose_estimator import create_pose_estimator, create_pose_estimator_3d
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer
from modules.database import DatabaseManager
# view_detector 已移除，视角通过手动选择

# 可视化图表工具
from utils.visualization_charts import (
    create_radar_chart,
    create_phase_distribution_pie,
    create_knee_angle_chart,
    create_metrics_comparison_chart,
    create_gait_timeline,
    create_score_gauge,
    create_lower_limb_alignment_chart,
    create_hip_drop_stats_chart
)

# 页面配置
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout']
)


def get_disabled_temporal_result() -> dict:
    """时序模块停用占位结果（保留兼容字段）。"""
    return {
        'disabled': True,
        'status': 'deprecated',
        'model_type': 'disabled',
        'note': '时序模型已停用，不参与当前分析与评分流程',
        'phase_distribution': {}
    }


# 初始化组件
@st.cache_resource
def init_components():
    """初始化系统组件"""
    return {
        'db': DatabaseManager(),
        'ai': AIAnalyzer()
    }


components = init_components()


def convert_video_for_browser(input_path: str, output_path: str = None) -> str:
    """
    将视频转换为浏览器兼容的H.264编码格式，同时处理视频旋转

    Args:
        input_path: 输入视频路径
        output_path: 输出路径（可选，默认创建临时文件）

    Returns:
        转换后的视频路径
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix='_web.mp4')

    # 使用VideoProcessor来正确处理旋转
    try:
        processor = VideoProcessor(input_path)
        video_info = processor.get_video_info()
        rotation = video_info.get('rotation', 0)

        # 获取旋转后的正确尺寸
        width = video_info['width']
        height = video_info['height']
        fps = video_info['fps']

        # 使用H.264编码（最兼容浏览器）
        codecs_to_try = ['avc1', 'H264', 'X264', 'mp4v']
        writer = None

        for codec in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                break
            writer = None

        if writer is None:
            processor.release()
            return input_path  # 无法创建写入器，返回原路径

        # 提取所有帧（已自动应用旋转）
        frames, _ = processor.extract_frames()

        # 写入帧
        for frame in frames:
            writer.write(frame)

        processor.release()
        writer.release()

        # 验证输出文件
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            return output_path
        return input_path

    except Exception as e:
        print(f"⚠️ 视频转换失败: {e}，使用原始视频")
        return input_path


def main():
    """主界面"""
    st.title("🏃 跑步动作分析系统")
    st.markdown("*基于深度学习的跑步动作视频解析与技术质量评价*")
    st.markdown("---")

    # 侧边栏
    with st.sidebar:
        st.header("📋 导航")
        page = st.radio(
            "选择功能",
            ["视频分析", "历史记录", "系统统计", "系统设置"]
        )

        st.markdown("---")
        st.info("💡 上传跑步视频，获取专业技术分析")

        # 3D姿态分析开关
        st.markdown("---")
        st.subheader("🎯 3D分析设置")
        enable_3d = st.toggle(
            "启用3D姿态提升",
            value=MOTIONBERT_CONFIG.get('enabled', True),
            help="使用MotionBERT将2D姿态提升为3D，提高膝关节角度等指标的精度"
        )
        st.session_state['enable_3d'] = enable_3d

        if enable_3d:
            st.success("✓ 3D分析已启用")
        else:
            st.warning("2D模式（精度较低）")

        # 显示系统信息
        st.markdown("---")
        st.caption("系统信息")
        st.caption(f"姿态估计: {POSE_CONFIG['backend'].upper()}")
        if enable_3d:
            st.caption("3D提升: MotionBERT")

    # 主内容区
    if page == "视频分析":
        video_analysis_page()
    elif page == "历史记录":
        history_page()
    elif page == "系统统计":
        statistics_page()
    elif page == "系统设置":
        settings_page()


def video_analysis_page():
    """视频分析页面"""
    st.header("📹 视频分析")

    # 检查是否有已完成的分析结果（用于保持页面状态）
    if st.session_state.get('analysis_complete', False):
        # 显示已保存的结果
        _display_saved_results()
        return

    # 视角选择（必须手动选择）
    st.info("📐 请根据您的视频拍摄角度选择正确的视角")
    view_angle = st.radio(
        "选择视频拍摄视角",
        ["侧面视角", "正面视角"],
        horizontal=True,
        help="侧面视角：从跑者侧面拍摄，适合分析膝关节角度、垂直振幅、躯干前倾。\n正面视角：从跑者正前方拍摄，适合分析下肢力线。"
    )

    selected_view = "side" if view_angle == "侧面视角" else "front"

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传跑步视频",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="支持常见视频格式，建议使用侧面或正面拍摄的视频"
    )

    if uploaded_file is not None:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # 转换视频为浏览器兼容格式并显示
        with st.spinner("正在处理视频..."):
            web_video_path = convert_video_for_browser(video_path)
            # 保存转换后的路径供后续使用
            st.session_state['web_video_path'] = web_video_path

        # 显示转换后的视频
        if Path(web_video_path).exists():
            st.video(web_video_path)
        else:
            st.warning("⚠️ 视频预览不可用，但分析可以继续")

        # 分析按钮
        if st.button("🔍 开始分析", type="primary"):
            analyze_video(video_path, selected_view)


def _display_saved_results():
    """显示已保存的分析结果"""
    saved = st.session_state.get('saved_analysis_results', {})
    if not saved:
        st.session_state['analysis_complete'] = False
        st.rerun()
        return

    # 新分析按钮
    if st.button("📹 分析新视频", type="secondary"):
        # 清除所有分析相关的session状态
        for key in ['analysis_complete', 'saved_analysis_results', 'ai_analysis_data',
                    'show_ai_dialog', 'ai_analysis_result', 'ai_analysis_success']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
        return

    # 显示保存的视频信息
    video_info = saved.get('video_info', {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("分辨率", f"{video_info.get('width', 0)}x{video_info.get('height', 0)}")
    col2.metric("帧率", f"{video_info.get('fps', 0):.1f} FPS")
    col3.metric("时长", f"{video_info.get('duration', 0):.1f} 秒")
    col4.metric("分析帧数", f"{saved.get('frame_count', 0)}")

    # 显示视角和3D状态信息
    detected_view = saved.get('detected_view', 'side')
    enable_3d = saved.get('enable_3d', False)
    pose_3d_info = saved.get('pose_3d_info', {})

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"📐 使用视角: {get_view_name(detected_view)} - {get_strategy_name(detected_view)}")
    with col_info2:
        if enable_3d and pose_3d_info and pose_3d_info.get('success'):
            st.success(f"🎯 3D分析已启用 (有效帧: {pose_3d_info.get('valid_frames_ratio', 0)*100:.0f}%)")
        elif enable_3d:
            st.warning("⚠ 3D分析受限，部分使用2D数据")
        else:
            st.warning("📊 2D分析模式")

    # 显示原始视频（如果有）
    # 优先使用转换后的浏览器兼容视频
    web_video_path = st.session_state.get('web_video_path')
    original_video_path = saved.get('original_video_path')

    video_to_show = None
    if web_video_path and Path(web_video_path).exists():
        video_to_show = web_video_path
    elif original_video_path and Path(original_video_path).exists():
        # 尝试转换原始视频
        video_to_show = convert_video_for_browser(original_video_path)
        st.session_state['web_video_path'] = video_to_show

    if video_to_show and Path(video_to_show).exists():
        st.subheader("📹 原始视频")
        st.video(video_to_show)

    # 显示姿态识别视频（如果有）
    pose_video_path = saved.get('pose_video_path')
    if pose_video_path and Path(pose_video_path).exists():
        st.subheader("🦴 姿态识别视频")
        st.video(pose_video_path)

    # 显示关键帧（如果有）
    keyframe_data = saved.get('keyframe_data', [])
    if keyframe_data:
        st.subheader("🖼️ 关键帧姿态分析")
        for row_start in range(0, len(keyframe_data), 3):
            cols = st.columns(3)
            for i, kf in enumerate(keyframe_data[row_start:row_start+3]):
                with cols[i]:
                    if Path(kf['path']).exists():
                        st.image(kf['path'], caption=f"时间: {kf['time_sec']:.2f}s",
                                 width='stretch')

    # 显示分析结果
    st.markdown("---")
    display_results(
        saved['quality_results'],
        saved['kinematic_results'],
        saved['temporal_results'],
        saved['local_report'],
        saved['detected_view'],
        saved['results_for_ai']
    )


def analyze_video(video_path: str, selected_view: str = 'side'):
    """执行视频分析"""
    try:
        # 获取3D开关状态
        enable_3d = st.session_state.get('enable_3d', True)

        # 进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. 视频预处理
        status_text.text("1️⃣ 视频预处理中...")
        progress_bar.progress(5)
        processor = VideoProcessor(video_path)
        video_info = processor.get_video_info()

        # 计算中间10秒的帧提取参数
        video_duration = video_info['duration']
        video_fps = video_info['fps']
        target_duration = min(10.0, video_duration)  # 最多10秒，不足10秒则取全部

        if video_duration > 10.0:
            # 计算中间段的起始时间
            start_time = (video_duration - target_duration) / 2
            # 计算需要跳过的帧数和提取的帧数
            start_frame = int(start_time * video_fps)
            max_frames = int(target_duration * video_fps)  # 按实际帧率计算
            st.info(f"📍 视频时长 {video_duration:.1f}s，提取中间 {target_duration:.0f} 秒进行分析 (从 {start_time:.1f}s 开始)")
        else:
            start_frame = 0
            max_frames = int(video_duration * video_fps)  # 按实际帧率计算
            st.info(f"📍 视频时长 {video_duration:.1f}s，分析完整视频")

        # 提取帧（从指定位置开始）
        frames, fps = processor.extract_frames_from_position(
            start_frame=start_frame,
            target_fps=video_fps,  # 使用实际帧率，不降采样
            max_frames=max_frames
        )

        # 显示视频信息
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("分辨率", f"{video_info['width']}x{video_info['height']}")
        col2.metric("帧率", f"{video_info['fps']:.1f} FPS")
        col3.metric("时长", f"{video_info['duration']:.1f} 秒")
        col4.metric("提取帧数", f"{len(frames)}")

        # 2. 姿态估计（2D或3D）
        keypoints_3d = None
        poses_3d = None  # 3D姿态数组
        pose_3d_info = None

        if enable_3d:
            status_text.text("2️⃣ 2D姿态估计 + 3D提升中...")
            progress_bar.progress(15)
            try:
                estimator = create_pose_estimator_3d(
                    backend_2d=POSE_CONFIG['backend'],
                    enable_3d=True,
                    device='auto'
                )
                pose_result = estimator.process_frames(
                    frames, lift_to_3d=True, view_angle=selected_view
                )
                keypoints_sequence = pose_result['keypoints_2d']
                keypoints_3d = pose_result.get('keypoints_3d')
                poses_3d = pose_result.get('poses_3d')  # 3D姿态数组
                pose_3d_info = pose_result.get('lift_info', {})

                # 显示3D提升状态
                if pose_3d_info.get('success'):
                    valid_ratio = pose_3d_info.get('valid_frames_ratio', 0)
                    st.success(f"✓ 3D姿态提升成功 (有效帧比例: {valid_ratio*100:.1f}%)")
                else:
                    st.warning(f"⚠ 3D提升受限: {pose_3d_info.get('error', '未知原因')}，使用2D数据")
                    keypoints_3d = None

            except Exception as e:
                st.warning(f"⚠ 3D估计器初始化失败: {e}，回退到2D模式")
                estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
                keypoints_sequence = estimator.process_frames(frames)
        else:
            status_text.text("2️⃣ 2D姿态估计中...")
            progress_bar.progress(20)
            estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
            keypoints_sequence = estimator.process_frames(frames)

        detected_count = sum(1 for kp in keypoints_sequence if kp['detected'])
        st.info(f"✓ 姿态检测成功: {detected_count}/{len(keypoints_sequence)} 帧 ({detected_count/len(keypoints_sequence)*100:.1f}%)")

        # 3. 使用用户选择的视角
        status_text.text("3️⃣ 确认分析视角...")
        progress_bar.progress(30)
        detected_view = selected_view
        st.info(f"📐 使用视角: {get_view_name(detected_view)} - {get_strategy_name(detected_view)}")

        # 生成姿态识别内容
        status_text.text("3️⃣ 生成姿态识别视频与关键帧...")
        progress_bar.progress(40)

        # 尝试生成视频
        pose_video_path = None
        try:
            pose_video_path = generate_pose_video(frames, keypoints_sequence, fps, estimator)
        except Exception as video_err:
            st.warning(f"视频生成失败: {video_err}，将显示关键帧图像")

        # 提取关键帧（不在此处显示，在_display_saved_results中显示）
        keyframe_data = extract_keyframes_with_poses(frames, keypoints_sequence, fps, estimator, num_keyframes=6)

        # 4. 运动学分析（直接使用 KinematicAnalyzer，支持3D）
        status_text.text("4️⃣ 运动学分析中...")
        progress_bar.progress(55)

        kinematic_analyzer = KinematicAnalyzer()
        kinematic_results = kinematic_analyzer.analyze_sequence(
            keypoints_sequence, fps,
            view_angle=detected_view,
            keypoints_3d=keypoints_3d,  # 传入3D关键点
            poses_3d=poses_3d  # 传入3D姿态数组
        )
        kinematic_results['view_angle'] = detected_view  # 添加视角信息

        # 5. 深度学习分析（已停用，保留兼容结构）
        status_text.text("5️⃣ 时序模块已停用...")
        progress_bar.progress(70)
        temporal_results = get_disabled_temporal_result()

        # 6. 质量评价
        status_text.text("6️⃣ 技术质量评价中...")
        progress_bar.progress(85)
        quality_evaluator = QualityEvaluator()
        quality_results = quality_evaluator.evaluate(
            kinematic_results, temporal_results,
            view_angle=detected_view
        )

        # 7. 生成本地规则引擎报告（始终生成）
        status_text.text("7️⃣ 生成本地分析报告...")
        progress_bar.progress(95)
        results_for_report = {
            'quality_evaluation': quality_results,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results,
            'view_angle': detected_view
        }
        # 使用本地规则引擎生成报告
        local_report = components['ai'].local_engine.generate_analysis_report(results_for_report)

        # 完成
        progress_bar.progress(100)
        status_text.text("✅ 分析完成!")

        # 保存到数据库
        complete_results = {
            'video_info': video_info,
            'kinematic_analysis': kinematic_results,
            'temporal_analysis': temporal_results,
            'quality_evaluation': quality_results,
            'ai_analysis': local_report,
            'view_angle': detected_view
        }
        record_id = components['db'].save_analysis(complete_results)

        # 清理资源
        processor.release()
        estimator.close()

        # 保存分析结果到session_state（用于AI分析按钮）
        st.session_state['saved_analysis_results'] = {
            'video_info': video_info,
            'frame_count': len(frames),
            'quality_results': quality_results,
            'kinematic_results': kinematic_results,
            'temporal_results': temporal_results,
            'local_report': local_report,
            'detected_view': detected_view,
            'results_for_ai': results_for_report,
            'keyframe_data': keyframe_data if keyframe_data else [],
            'record_id': record_id,
            'pose_video_path': pose_video_path,  # 保存姿态识别视频路径
            'original_video_path': video_path,    # 保存原始视频路径
            'enable_3d': enable_3d,               # 是否启用3D
            'pose_3d_info': pose_3d_info,         # 3D提升信息
        }
        st.session_state['analysis_complete'] = True

        # 重新运行以显示结果（使用保存的状态）
        st.rerun()

    except Exception as e:
        st.error(f"分析过程出错: {e}")
        import traceback
        st.code(traceback.format_exc())


def get_view_name(view: str) -> str:
    """获取视角中文名称"""
    names = {
        'side': '侧面视角',
        'front': '正面视角'
    }
    return names.get(view, view)


def get_strategy_name(view: str) -> str:
    """获取分析策略名称"""
    strategies = {
        'side': '膝角+振幅+躯干前倾',
        'front': '下肢力线+横向稳定+肩部晃动'
    }
    return strategies.get(view, '标准分析')


def generate_pose_video(frames, keypoints_sequence, fps, estimator):
    """将姿态骨架绘制到每一帧并生成视频"""
    import tempfile
    import os

    # 使用临时文件避免路径问题
    output_dir = Path("output/videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用唯一的文件名
    import time
    timestamp = int(time.time())
    output_path = output_dir / f"pose_visualization_{timestamp}.mp4"

    h, w = frames[0].shape[:2]
    fps_int = max(1, int(round(fps)))

    # 尝试多种编码格式
    codecs = [
        ('avc1', '.mp4'),  # H.264 - 最兼容
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # XVID
    ]

    writer = None
    final_path = None

    for codec, ext in codecs:
        test_path = output_dir / f"pose_visualization_{timestamp}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(test_path),
            fourcc,
            fps_int,
            (w, h)
        )
        if writer.isOpened():
            final_path = test_path
            break
        writer.release()

    if not writer or not writer.isOpened():
        raise RuntimeError("❌ VideoWriter 打开失败，尝试了多种编码格式")

    for frame, kp in zip(frames, keypoints_sequence):
        if kp.get("detected", False):
            vis_frame = estimator.visualize_pose(frame, kp)
        else:
            vis_frame = frame.copy()

        writer.write(vis_frame)

    writer.release()

    return str(final_path)


def extract_keyframes_with_poses(frames, keypoints_sequence, fps, estimator, num_keyframes=6):
    """提取关键帧并绘制姿态骨架"""
    import time

    output_dir = Path("output/keyframes")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(frames)
    if total_frames == 0:
        return []

    # 计算关键帧索引（均匀分布）
    if total_frames <= num_keyframes:
        indices = list(range(total_frames))
    else:
        indices = [int(i * (total_frames - 1) / (num_keyframes - 1)) for i in range(num_keyframes)]

    keyframe_paths = []
    timestamp = int(time.time())

    for i, idx in enumerate(indices):
        frame = frames[idx]
        kp = keypoints_sequence[idx]

        if kp.get("detected", False):
            vis_frame = estimator.visualize_pose(frame.copy(), kp)
        else:
            vis_frame = frame.copy()
            # 在未检测到姿态的帧上添加提示
            cv2.putText(vis_frame, "No pose detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加时间戳
        time_sec = idx / fps
        cv2.putText(vis_frame, f"Time: {time_sec:.2f}s", (10, vis_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 保存关键帧
        keyframe_path = output_dir / f"keyframe_{timestamp}_{i}.jpg"
        cv2.imwrite(str(keyframe_path), vis_frame)
        keyframe_paths.append({
            'path': str(keyframe_path),
            'frame_idx': idx,
            'time_sec': time_sec,
            'detected': kp.get("detected", False)
        })

    return keyframe_paths


def display_results(quality, kinematic, temporal, local_report, view_angle='side', results_for_ai=None):
    """
    显示分析结果 - 重构版本

    特点：
    1. 技术质量评分基于纯运动学规则（生物力学标准）
    2. 深度学习分析结果仅作为参考信息展示
    3. 丰富的数据可视化图表
    """
    st.header("📊 技术质量分析报告")

    # ========== 第一部分：总体评分概览 ==========
    st.subheader("🎯 总体评价")

    # 使用两列布局：左侧仪表盘，右侧信息
    score_col, info_col = st.columns([1, 1])

    with score_col:
        # 创建评分仪表盘
        score = quality['total_score']
        gauge_fig = create_score_gauge(score, "技术质量评分")
        st.plotly_chart(gauge_fig, width='stretch', key="score_gauge")

    with info_col:
        # 评级和基本信息
        st.markdown(f"### {quality['rating']}")
        st.metric("分析视角", get_view_name(view_angle))

        # 优势和薄弱项
        if quality.get('strengths'):
            st.markdown("**✅ 技术优势:**")
            for s in quality['strengths'][:3]:
                st.markdown(f"- {s}")
        if quality.get('weaknesses'):
            st.markdown("**⚠️ 待改进:**")
            for w in quality['weaknesses'][:3]:
                st.markdown(f"- {w}")

    st.markdown("---")

    # ========== 第二部分：各维度表现（雷达图 + 数值）==========
    st.subheader("📈 各维度表现")

    is_frontal = view_angle in ['front', 'back']

    # 根据视角选择显示不同的维度
    if is_frontal and 'frontal_dimension_scores' in quality:
        # 正面视角：显示独立的评价维度（3维度结构）
        frontal_dims = quality['frontal_dimension_scores']
        frontal_names = quality.get('frontal_dimension_names', {
            'lower_limb_alignment': '下肢力线',
            'lateral_stability': '横向稳定性',
            'efficiency': '效率'
        })

        # 雷达图和数值并排
        radar_col, dims_col = st.columns([1, 1])

        with radar_col:
            # 正面视角专用雷达图（3维度）
            radar_data = {
                frontal_names['lower_limb_alignment']: frontal_dims.get('lower_limb_alignment', 0),
                frontal_names['lateral_stability']: frontal_dims.get('lateral_stability', 0),
                frontal_names['efficiency']: frontal_dims.get('efficiency', 0),
            }
            radar_fig = create_radar_chart(radar_data)
            st.plotly_chart(radar_fig, width='stretch', key="radar_chart_frontal")

        with dims_col:
            # 正面视角维度评分详情
            st.markdown("#### 正面视角评分维度")

            # 下肢力线 (35%)
            limb_score = frontal_dims.get('lower_limb_alignment', 0)
            st.markdown(f"**{frontal_names['lower_limb_alignment']}** - `{limb_score:.1f}` 分 *(权重35%)*")
            st.progress(limb_score / 100)
            st.caption("膝外翻(50%) + 髋部下沉(30%) + 整体评级(20%)")

            # 横向稳定性 (35%) - 含对称性
            lateral_score = frontal_dims.get('lateral_stability', 0)
            st.markdown(f"**{frontal_names['lateral_stability']}** - `{lateral_score:.1f}` 分 *(权重35%)*")
            st.progress(lateral_score / 100)
            st.caption("髋部横摆(40%) + 肩部倾斜(30%) + 对称性(30%)")

            # 效率 (30%) - 步频 + 垂直振幅
            eff_score = frontal_dims.get('efficiency', 0)
            st.markdown(f"**{frontal_names['efficiency']}** - `{eff_score:.1f}` 分 *(权重30%)*")
            st.progress(eff_score / 100)
            st.caption("步频(60%) + 垂直振幅(40%)")

        # 正面视角数据可靠性提示
        st.info("📐 **正面视角说明**: 下肢力线和横向稳定性数据可靠度高，效率维度（步频/垂直振幅）为估算值。如需完整分析，建议同时使用侧面视角拍摄。")

    else:
        # 侧面视角：显示原有的三维度
        dimensions = quality.get('dimension_scores', {})

        # 雷达图和数值并排
        radar_col, dims_col = st.columns([1, 1])

        with radar_col:
            # 创建雷达图
            radar_fig = create_radar_chart(dimensions)
            st.plotly_chart(radar_fig, width='stretch', key="radar_chart")

        with dims_col:
            # 各维度数值展示
            st.markdown("#### 维度评分详情")

            # 稳定性
            stability_score = dimensions.get('stability', 0)
            st.markdown(f"**动作稳定性** - `{stability_score:.1f}` 分")
            st.progress(stability_score / 100)
            st.caption("躯干稳定、膝关节角度变异度、垂直运动稳定性")

            # 效率
            efficiency_score = dimensions.get('efficiency', 0)
            st.markdown(f"**跑步效率** - `{efficiency_score:.1f}` 分")
            st.progress(efficiency_score / 100)
            st.caption("垂直振幅控制、步频合理性、触地时间")

            # 跑姿
            form_score = dimensions.get('form', 0)
            st.markdown(f"**跑姿标准度** - `{form_score:.1f}` 分")
            st.progress(form_score / 100)
            st.caption("膝关节角度、躯干前倾、着地方式")

    # 数据可靠性指示（如果有）
    data_reliability = quality.get('data_reliability', {})
    if data_reliability:
        with st.expander("📊 数据可靠性说明", expanded=False):
            overall = data_reliability.get('overall', 'unknown')
            reliability_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴', 'unknown': '⚪'}
            reliability_names = {'high': '高（3D数据）', 'medium': '中等', 'low': '低（2D投影）', 'unknown': '未知'}

            st.markdown(f"**整体数据可靠性:** {reliability_colors.get(overall, '⚪')} {reliability_names.get(overall, overall)}")

            details = data_reliability.get('details', {})
            if details:
                st.markdown("**各指标可靠性:**")
                for metric, info in details.items():
                    level = info.get('level', 'unknown')
                    reason = info.get('reason', '')
                    metric_names = {
                        'knee_angle': '膝关节角度',
                        'vertical_oscillation': '垂直振幅',
                        'trunk_lean': '躯干前倾',
                        'cadence': '步频'
                    }
                    display_name = metric_names.get(metric, metric)
                    st.caption(f"- {display_name}: {reliability_colors.get(level, '⚪')} {reliability_names.get(level, level)} - {reason}")

    st.markdown("---")

    # ========== 第三部分：数据可视化图表区 ==========
    st.subheader("📊 数据可视化")

    # 准备可视化数据
    phase_dist = temporal.get('phase_distribution', {})
    gait_cycle = kinematic.get('gait_cycle', {})
    angles = kinematic.get('angles', {})
    cadence_data = kinematic.get('cadence', {})
    vertical_motion = kinematic.get('vertical_motion', {})

    # 第一行：步态阶段分布 + 膝关节角度对比
    viz_row1_col1, viz_row1_col2 = st.columns(2)

    with viz_row1_col1:
        # 步态阶段分布饼图
        if phase_dist:
            pie_fig = create_phase_distribution_pie(phase_dist)
            st.plotly_chart(pie_fig, width='stretch', key="phase_pie")
        else:
            if temporal.get('disabled'):
                st.info("时序模块已停用，阶段分布图不可用")
            else:
                st.info("暂无步态阶段数据")

    with viz_row1_col2:
        # 膝关节角度对比（侧面视角）
        if view_angle == 'side' and 'phase_analysis' in angles:
            phase_analysis = angles['phase_analysis']
            # 安全提取数据，处理None和空值情况
            gc_data = phase_analysis.get('ground_contact') or {}
            fl_data = phase_analysis.get('flight') or {}

            gc_mean = gc_data.get('mean', 0) if isinstance(gc_data, dict) else 0
            fl_mean = fl_data.get('mean', 0) if isinstance(fl_data, dict) else 0

            # 只有在有有效数据时才显示图表
            if gc_mean > 0 or fl_mean > 0:
                knee_data = {
                    'ground_contact': gc_mean,
                    'flight': fl_mean
                }
                knee_fig = create_knee_angle_chart(knee_data)
                st.plotly_chart(knee_fig, width='stretch', key="knee_chart")
            else:
                st.info("膝关节角度数据不足")
        else:
            st.info("膝关节角度图表仅适用于侧面视角")

    # 第二行：关键指标对比 + 步态时间线
    viz_row2_col1, viz_row2_col2 = st.columns(2)

    with viz_row2_col1:
        # 关键指标与参考值对比
        metrics_data = {}

        # 步频
        cadence_val = cadence_data.get('cadence', 0) or 0
        if cadence_val > 0:
            metrics_data['步频(步/分)'] = {
                'value': cadence_val,
                'ref_min': 170,
                'ref_max': 190
            }

        # 垂直振幅
        amp_val = vertical_motion.get('amplitude_normalized', 0) or 0
        if amp_val > 0:
            metrics_data['垂直振幅(%)'] = {
                'value': amp_val,
                'ref_min': 3,
                'ref_max': 8
            }

        # 触地时间
        phase_dur = gait_cycle.get('phase_duration_ms') or {}
        gc_time_val = phase_dur.get('ground_contact', 0) or 0
        if gc_time_val > 0:
            metrics_data['触地时间(ms)'] = {
                'value': gc_time_val,
                'ref_min': 180,
                'ref_max': 270
            }

        if metrics_data:
            metrics_fig = create_metrics_comparison_chart(metrics_data)
            st.plotly_chart(metrics_fig, width='stretch', key="metrics_chart")
        else:
            st.info("关键指标数据不足")

    with viz_row2_col2:
        # 步态周期时间线
        phase_duration = gait_cycle.get('phase_duration_ms') or {}
        gc_ms = phase_duration.get('ground_contact', 0) or 0
        flight_ms = phase_duration.get('flight', 0) or 0

        if gc_ms > 0 or flight_ms > 0:
            timeline_fig = create_gait_timeline(phase_duration)
            st.plotly_chart(timeline_fig, width='stretch', key="gait_timeline")
        else:
            st.info("步态周期数据不足")

    st.markdown("---")

    # ========== 第四部分：运动学详细指标 ==========
    st.subheader("🔬 运动学详细指标")

    # 基础指标（根据视角调整显示）
    col1, col2, col3 = st.columns(3)

    # 步频显示
    cadence_val = cadence_data.get('cadence', 0)
    cadence_confidence = cadence_data.get('confidence', 1.0)
    is_cadence_estimate = cadence_data.get('is_estimate', False) or is_frontal

    if is_cadence_estimate and cadence_val > 0:
        # 正面视角步频为估算值
        col1.metric("步频 ⚠️", f"{cadence_val:.1f} 步/分",
                    delta="估算值",
                    help=f"正面视角基于左右脚X坐标交替检测，置信度: {cadence_confidence*100:.0f}%")
    else:
        col1.metric("步频", f"{cadence_val:.1f} 步/分",
                    delta=cadence_data.get('rating', {}).get('description', ''))

    col2.metric("检测步数", f"{cadence_data.get('step_count', 0)} 步",
                help=f"视频时长 {cadence_data.get('duration', 0):.1f} 秒")

    # 垂直振幅 - 使用归一化值
    is_amp_estimate = vertical_motion.get('is_estimate', False) or is_frontal

    if 'amplitude_normalized' in vertical_motion:
        amplitude_pct = vertical_motion['amplitude_normalized']
        rating_info = vertical_motion.get('amplitude_rating', {})

        if is_amp_estimate and amplitude_pct > 0:
            # 正面视角垂直振幅为估算值
            col3.metric("垂直振幅 ⚠️", f"{amplitude_pct:.1f}% 躯干",
                        delta="估算值",
                        help="正面视角基于髋部/肩部Y坐标估算，仅供参考")
        else:
            col3.metric("垂直振幅", f"{amplitude_pct:.1f}% 躯干",
                        delta=rating_info.get('description', ''),
                        help="相对于躯干长度的垂直振幅百分比")
    elif vertical_motion.get('amplitude', 0) > 0:
        col3.metric("垂直振幅", f"{vertical_motion['amplitude']:.4f}",
                    help="归一化坐标下的振幅")
    else:
        col3.metric("垂直振幅", "数据不足")

    # 触地时间显示
    phase_duration = gait_cycle.get('phase_duration_ms', {})
    gait_rating = gait_cycle.get('gait_rating', {})
    is_gait_unavailable = gait_rating.get('level') == 'not_available'

    if is_frontal and is_gait_unavailable:
        # 正面视角触地时间不可用
        st.markdown("#### ⏱️ 步态时间")
        st.warning("⚠️ **正面视角无法准确检测触地/离地时刻**，触地时间数据不可用。如需此数据，请使用侧面视角拍摄。")
    elif phase_duration:
        st.markdown("#### ⏱️ 步态时间")
        time_cols = st.columns(3)
        ground_contact_ms = phase_duration.get('ground_contact', 0)
        flight_ms = phase_duration.get('flight', 0)

        # 触地时间评级
        if ground_contact_ms > 0:
            if ground_contact_ms < 210:
                gc_rating = "精英"
            elif ground_contact_ms < 240:
                gc_rating = "优秀"
            elif ground_contact_ms < 270:
                gc_rating = "良好"
            elif ground_contact_ms < 300:
                gc_rating = "一般"
            else:
                gc_rating = "较差"
            time_cols[0].metric("触地时间", f"{ground_contact_ms:.1f} ms", delta=gc_rating)
        else:
            time_cols[0].metric("触地时间", "数据不足")

        time_cols[1].metric("腾空时间", f"{flight_ms:.1f} ms" if flight_ms > 0 else "数据不足")

        cycle_ms = gait_cycle.get('avg_cycle_duration_ms', 0)
        time_cols[2].metric("步态周期", f"{cycle_ms:.1f} ms" if cycle_ms > 0 else "数据不足")

    # 膝关节角度分析（侧面视角重点）
    if view_angle == 'side':
        if 'phase_analysis' in angles:
            st.markdown("#### 🦵 膝关节角度分析")
            phase_analysis = angles['phase_analysis']
            gc = phase_analysis.get('ground_contact', {})

            # 落地膝角（重点指标）
            landing_mean = gc.get('landing_angle_mean', gc.get('mean', 0))
            landing_std = gc.get('landing_angle_std', gc.get('std', 0))
            landing_count = gc.get('landing_count', gc.get('count', 0))

            # 落地膝角评级
            # 膝角评级（基于马拉松运动员研究数据）
            # 理想范围：150-165°（约15-30°屈曲）
            if 150 <= landing_mean <= 165:
                landing_rating = "优秀"
            elif 140 <= landing_mean < 150 or 165 < landing_mean <= 170:
                landing_rating = "良好"
            elif 135 <= landing_mean < 140 or 170 < landing_mean <= 175:
                landing_rating = "一般"
            else:
                landing_rating = "需改进"

            # 主要指标显示
            st.markdown("**落地时膝关节角度（最大稳定伸展角）**")
            main_cols = st.columns([2, 1, 1])
            with main_cols[0]:
                st.metric(
                    "平均角度",
                    f"{landing_mean:.1f}°",
                    delta=landing_rating
                )
            with main_cols[1]:
                st.metric("标准差", f"±{landing_std:.1f}°")
            with main_cols[2]:
                st.metric("检测步数", f"{landing_count}")

            st.caption("理想范围：150-165°（中长跑标准，约15-30°屈曲以缓冲冲击）")

            # 显示有效/拒绝统计
            valid_count = gc.get('valid_count', landing_count)
            rejected_count = gc.get('rejected_count', 0)
            if rejected_count > 0:
                st.info(f"🔍 检测质量：{valid_count}次通过生物力学约束 / {valid_count + rejected_count}次候选落地")

            # 有效落地详细统计（可展开）
            per_step_stats = gc.get('per_step_stats', [])
            if per_step_stats:
                with st.expander(f"✅ 有效落地数据（共{len(per_step_stats)}步）", expanded=False):
                    import pandas as pd

                    # 安全的数值转换函数
                    def safe_float(val, default=0.0):
                        if val is None:
                            return default
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return default

                    step_data = []
                    for i, step in enumerate(per_step_stats):
                        angle = safe_float(step.get('landing_angle', step.get('max_stable_angle', 0)))
                        confidence = safe_float(step.get('confidence', 0))
                        duration = safe_float(step.get('duration_ms', 0))
                        step_data.append({
                            '序号': i + 1,
                            '脚': '左' if step.get('foot') == 'left' else '右',
                            '膝角(°)': f"{angle:.1f}",
                            '置信度': f"{confidence:.2f}",
                            '触地时长(ms)': f"{duration:.0f}"
                        })
                    df = pd.DataFrame(step_data)
                    st.dataframe(df, width='stretch', hide_index=True)

            # 被拒绝的落地（显示原因）
            rejected_steps = gc.get('rejected_steps', [])
            if rejected_steps:
                with st.expander(f"❌ 被拒绝的落地（共{len(rejected_steps)}次）", expanded=False):
                    import pandas as pd
                    reason_map = {
                        'angle_too_low': '膝角过小（摆动期）',
                        'angle_too_high': '膝角过大',
                        'flexion_trend': '屈膝趋势（非伸展）',
                        'no_valid_frames': '无有效帧',
                        'window_too_small': '窗口过小'
                    }

                    # 安全的数值转换函数
                    def safe_float_reject(val, default=0.0):
                        if val is None:
                            return default
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return default

                    reject_data = []
                    for step in rejected_steps:
                        actual_angle = safe_float_reject(step.get('actual_angle', 0))
                        actual_rate = safe_float_reject(step.get('actual_rate', 0))
                        reject_data.append({
                            '脚': '左' if step.get('foot') == 'left' else '右',
                            '拒绝原因': reason_map.get(step.get('rejection_reason', ''), step.get('rejection_reason', '')),
                            '实际角度': f"{actual_angle:.1f}°",
                            '变化率': f"{actual_rate:.2f}"
                        })
                    df = pd.DataFrame(reject_data)
                    st.dataframe(df, width='stretch', hide_index=True)
                    st.caption("💡 被拒绝的原因：膝角不在140-170°范围内，或处于屈膝趋势（非落地相位）")

            # 其他阶段角度（折叠显示）
            with st.expander("其他阶段角度统计", expanded=False):
                phase_cols = st.columns(3)

                with phase_cols[0]:
                    st.markdown("**触地阶段整体**")
                    st.caption(f"平均: {gc.get('mean', 0):.1f}°")
                    st.caption(f"范围: {gc.get('min', 0):.1f}° - {gc.get('max', 0):.1f}°")

                fl = phase_analysis.get('flight', {})
                with phase_cols[1]:
                    st.markdown("**腾空阶段**")
                    st.caption(f"平均: {fl.get('mean', 0):.1f}°")
                    st.caption(f"范围: {fl.get('min', 0):.1f}° - {fl.get('max', 0):.1f}°")

                tr = phase_analysis.get('transition', {})
                with phase_cols[2]:
                    st.markdown("**过渡阶段**")
                    st.caption(f"平均: {tr.get('mean', 0):.1f}°")
                    st.caption(f"范围: {tr.get('min', 0):.1f}° - {tr.get('max', 0):.1f}°")

    # 正面视角分析（下肢力线、肩部倾斜、横向稳定性）
    if view_angle in ['front', 'back']:
        st.markdown("---")
        st.subheader("🎯 正面视角核心指标")

        # 下肢力线分析
        lower_limb = kinematic.get('lower_limb_alignment', {})
        if lower_limb:
            st.markdown("#### 🦿 下肢力线分析")

            # ========== 详细统计数据 ==========
            left_leg = lower_limb.get('left_leg', {})
            right_leg = lower_limb.get('right_leg', {})
            hip_drop = lower_limb.get('hip_drop', {})
            asymmetry = lower_limb.get('asymmetry', 0)

            # 第一行：左右腿膝关节偏移统计
            st.markdown("**膝关节偏移角度统计**")
            col1, col2, col3, col4 = st.columns(4)

            # 左腿数据
            left_mean = left_leg.get('mean', 0)
            left_issue = left_leg.get('issue', 'unknown')
            issue_names = {'valgus': '膝外翻', 'varus': '膝内扣', 'normal': '正常', 'unstable': '不稳定', 'unknown': '未知'}
            col1.metric("左腿平均偏移", f"{left_mean:.1f}°",
                       delta=issue_names.get(left_issue, ''),
                       delta_color="off" if left_issue == 'normal' else "inverse")
            col2.metric("左腿范围", f"{left_leg.get('min', 0):.1f}° ~ {left_leg.get('max', 0):.1f}°",
                       help=f"标准差: ±{left_leg.get('std', 0):.1f}°")

            # 右腿数据
            right_mean = right_leg.get('mean', 0)
            right_issue = right_leg.get('issue', 'unknown')
            col3.metric("右腿平均偏移", f"{right_mean:.1f}°",
                       delta=issue_names.get(right_issue, ''),
                       delta_color="off" if right_issue == 'normal' else "inverse")
            col4.metric("右腿范围", f"{right_leg.get('min', 0):.1f}° ~ {right_leg.get('max', 0):.1f}°",
                       help=f"标准差: ±{right_leg.get('std', 0):.1f}°")

            # 第二行：髋部下沉和左右差异
            col5, col6, col7 = st.columns(3)

            if hip_drop:
                hip_mean = hip_drop.get('mean', 0)
                hip_max = hip_drop.get('max', 0)

                # 髋部下沉评级
                if abs(hip_mean) <= 3:
                    hip_rating = "优秀"
                elif abs(hip_mean) <= 5:
                    hip_rating = "良好"
                elif abs(hip_mean) <= 8:
                    hip_rating = "一般"
                else:
                    hip_rating = "需改进"

                col5.metric("髋部平均下沉", f"{hip_mean:.1f}°", delta=hip_rating,
                           delta_color="off" if abs(hip_mean) <= 5 else "inverse")
                col6.metric("髋部最大下沉", f"{hip_max:.1f}°",
                           help="理想范围：<5°，>8°表示核心力量不足")

            col7.metric("左右不对称度", f"{asymmetry:.1f}°",
                       help="左右腿平均偏移角度差，越小越对称")

            # ========== 时序曲线图 ==========
            chart_data = lower_limb.get('chart_data', {})
            if chart_data and chart_data.get('left') and chart_data.get('right'):
                st.markdown("**膝关节偏移时序曲线**")
                alignment_chart = create_lower_limb_alignment_chart(chart_data)
                st.plotly_chart(alignment_chart, width='stretch', key="alignment_chart")

                # 曲线说明
                st.caption("💡 曲线显示膝关节相对于髋-踝连线的横向偏移。正值表示膝外翻（膝盖向外偏离），负值表示膝内扣。绿色区域(-2°~+2°)为正常范围。")

            # ========== 整体评级 ==========
            overall_rating = lower_limb.get('overall_rating', {})
            if overall_rating:
                score = overall_rating.get('score', 0)
                level = overall_rating.get('level', 'unknown')
                desc = overall_rating.get('description', '')

                # 使用颜色标识
                if level == 'excellent':
                    st.success(f"✅ **下肢力线综合评分**: {score:.0f} 分 - {desc}")
                elif level == 'good':
                    st.info(f"👍 **下肢力线综合评分**: {score:.0f} 分 - {desc}")
                elif level in ['fair', 'moderate']:
                    st.warning(f"⚠️ **下肢力线综合评分**: {score:.0f} 分 - {desc}")
                else:
                    st.error(f"❗ **下肢力线综合评分**: {score:.0f} 分 - {desc}")

        # 肩部倾斜分析（新增正面视角指标）
        shoulder_analysis = kinematic.get('shoulder_analysis', {})
        if shoulder_analysis:
            st.markdown("#### 💪 肩部倾斜分析")
            shoulder_cols = st.columns(4)

            tilt_mean = shoulder_analysis.get('tilt_mean', 0)
            tilt_std = shoulder_analysis.get('tilt_std', 0)
            tilt_max = shoulder_analysis.get('tilt_max', 0)
            rating = shoulder_analysis.get('rating', {})

            shoulder_cols[0].metric("平均倾斜", f"{tilt_mean:.1f}°")
            shoulder_cols[1].metric("倾斜变化", f"±{tilt_std:.1f}°")
            shoulder_cols[2].metric("最大倾斜", f"{tilt_max:.1f}°")
            shoulder_cols[3].metric("评级", rating.get('level', 'N/A'),
                                    help="理想范围：0-3°，>10°影响跑步效率")

        # 横向稳定性
        lateral = kinematic.get('lateral_stability', {})
        if lateral:
            st.markdown("#### ↔️ 横向稳定性")
            lat_cols = st.columns(3)

            hip_sway = lateral.get('hip_sway', lateral.get('hip_sway_normalized', 0) * 100)
            shoulder_sway = lateral.get('shoulder_sway', 0)
            stability_score = lateral.get('stability_score', 0)

            lat_cols[0].metric("髋部横摆", f"{hip_sway:.2f}%",
                              help="髋部侧向移动幅度（相对髋宽百分比），<3%优秀")
            lat_cols[1].metric("肩部横摆", f"{shoulder_sway:.2f}%",
                              help="肩部侧向移动幅度（相对肩宽百分比）")
            lat_cols[2].metric("稳定评分", f"{stability_score:.1f}",
                              help="综合横向稳定性评分")

    st.markdown("---")

    # ========== 第五部分：时序模块状态 ==========
    st.subheader("🤖 时序深度学习模块状态")
    st.info("时序模型已停用（不参与当前分析流程与评分计算）。")
    st.caption("保留temporal字段仅用于历史兼容与路由稳定。")

    st.markdown("---")

    # ========== 第六部分：改进建议 ==========
    if quality.get('suggestions'):
        st.subheader("💡 改进建议")
        for i, suggestion in enumerate(quality['suggestions'], 1):
            st.markdown(f"**{i}.** {suggestion}")
        st.markdown("---")

    # ========== 第七部分：本地分析报告 ==========
    st.subheader("📝 详细分析报告")
    with st.expander("查看完整分析报告", expanded=True):
        st.markdown(local_report)

    # ========== 第八部分：评分调试模块 ==========
    display_debug_scoring_details(quality, kinematic, view_angle)

    # ========== 第九部分：AI智能分析（可选）==========
    st.markdown("---")
    st.subheader("🧠 AI智能分析（可选）")
    st.info("💡 点击下方按钮使用智谱AI大模型对数据进行深度分析和个性化建议。")

    # 保存数据到session_state供弹窗使用
    if results_for_ai:
        st.session_state['ai_analysis_data'] = results_for_ai

    # AI分析按钮
    if st.button("🚀 启动AI智能分析", type="secondary", key="ai_analysis_btn"):
        show_ai_analysis_dialog()


@st.dialog("🧠 AI深度分析", width="large")
def show_ai_analysis_dialog():
    """AI分析结果弹窗"""
    results_for_ai = st.session_state.get('ai_analysis_data', None)

    if not results_for_ai:
        st.error("没有可用的分析数据")
        return

    # 每次打开弹窗时进行AI分析
    with st.spinner("正在调用智谱AI进行深度分析..."):
        try:
            ai_response = components['ai'].generate_analysis_report(results_for_ai)
            st.markdown("### 📊 AI分析报告")
            st.markdown(ai_response)
            st.markdown("---")
            st.caption("由智谱AI (GLM-4) 生成")
        except Exception as e:
            st.error(f"AI分析失败: {str(e)}")
            st.info("您可以关闭此窗口后重试，或检查网络连接。")


def history_page():
    """历史记录页面"""
    st.header("📜 历史记录")

    # 管理选项
    with st.expander("🛠️ 管理选项", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空所有记录", type="secondary"):
                count = components['db'].delete_all_analyses()
                st.success(f"已删除 {count} 条记录")
                st.rerun()

        with col2:
            if st.button("🧹 清理临时文件", type="secondary"):
                cleanup_temp_files()
                st.success("临时文件清理完成")

    # 获取记录
    records = components['db'].get_recent_analyses(50)

    if not records:
        st.info("暂无历史记录")
        return

    st.markdown(f"共 **{len(records)}** 条记录")

    for record in records:
        record_id = record.get('id', 0)
        with st.expander(f"📹 {record['video_filename']} - {record['analysis_date']} (ID: {record_id})"):
            # 基本信息
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("评分", f"{record['total_score']:.1f}")
            col2.metric("评级", record['rating'])
            col3.metric("时长", f"{record['video_duration']:.1f}秒")
            col4.metric("步频", f"{record['cadence']:.1f}")

            # 各维度得分
            st.markdown("**各维度得分:**")
            dim_cols = st.columns(3)
            dim_cols[0].metric("稳定性", f"{record.get('stability_score', 0):.1f}")
            dim_cols[1].metric("效率", f"{record.get('efficiency_score', 0):.1f}")
            dim_cols[2].metric("跑姿", f"{record.get('form_score', 0):.1f}")

            # 深度学习结果
            st.markdown("**深度学习分析:**")
            dl_cols = st.columns(2)
            dl_cols[0].metric("AI质量评分", f"{record.get('dl_quality_score', 0):.1f}")
            dl_cols[1].metric("AI稳定性", f"{record.get('dl_stability_score', 0):.1f}")

            # AI分析文本
            ai_text = record.get('ai_analysis_text', '')
            if ai_text:
                with st.container():
                    st.markdown("**AI分析报告:**")
                    st.markdown(ai_text)

            # 操作按钮
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button(f"📊 查看完整数据", key=f"view_{record_id}"):
                    full_results = components['db'].get_full_results(record_id)
                    if full_results:
                        st.json(full_results)
                    else:
                        st.warning("完整数据不可用")

            with btn_col2:
                if st.button(f"🗑️ 删除记录", key=f"delete_{record_id}"):
                    if components['db'].delete_analysis(record_id):
                        st.success("记录已删除")
                        st.rerun()
                    else:
                        st.error("删除失败")


def cleanup_temp_files():
    """清理临时文件"""
    import shutil
    from pathlib import Path

    cleanup_dirs = [
        Path("output/videos"),
        Path("output/keyframes"),
        Path("output/visualizations")
    ]

    total_cleaned = 0
    for dir_path in cleanup_dirs:
        if dir_path.exists():
            for file in dir_path.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                        total_cleaned += 1
                except Exception:
                    pass

    return total_cleaned


def statistics_page():
    """统计页面"""
    st.header("📊 系统统计")

    stats = components['db'].get_statistics()

    col1, col2 = st.columns(2)
    col1.metric("总分析次数", stats['total_analyses'])
    col2.metric("平均评分", f"{stats['average_score']:.1f}")

    st.subheader("评级分布")
    for rating, count in stats['rating_distribution'].items():
        st.markdown(f"**{rating}:** {count} 次")


def settings_page():
    """系统设置页面"""
    st.header("⚙️ 系统设置")

    st.subheader("姿态估计设置")
    st.info(f"当前后端: {POSE_CONFIG['backend'].upper()}")
    st.caption("如需切换姿态估计后端，请修改配置文件 config/config.py")

    # 3D姿态提升设置
    st.subheader("🎯 3D姿态提升设置")
    if MOTIONBERT_CONFIG.get('enabled'):
        st.success("MotionBERT 3D提升已启用")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**模型配置:**")
            st.caption(f"- 特征维度: {MOTIONBERT_CONFIG['model']['dim_feat']}")
            st.caption(f"- Transformer深度: {MOTIONBERT_CONFIG['model']['depth']}")
            st.caption(f"- 注意力头数: {MOTIONBERT_CONFIG['model']['num_heads']}")
        with col2:
            st.markdown("**显存优化:**")
            st.caption(f"- 每批最大帧数: {MOTIONBERT_CONFIG['memory_optimization']['max_batch_frames']}")
            st.caption(f"- 设备: {MOTIONBERT_CONFIG['device']}")

        # 检查模型文件
        from pathlib import Path
        checkpoint_path = Path(MOTIONBERT_CONFIG['checkpoint_path'])
        if checkpoint_path.exists():
            st.success(f"✓ 模型文件已找到: {checkpoint_path.name}")
        else:
            st.error(f"✗ 模型文件未找到: {checkpoint_path}")
            st.caption("请将 best_epoch.bin 放入 data/checkpoints/ 目录")
    else:
        st.warning("3D姿态提升未启用")
        st.caption("如需启用，请在 config/config.py 中设置 MOTIONBERT_CONFIG['enabled'] = True")

    st.subheader("视角设置")
    st.markdown("""
    **支持的视角（使用独立评分系统）:**

    | 视角 | 核心指标 | 辅助指标 | 不可用指标 |
    |------|----------|----------|------------|
    | **侧面视角** | 膝关节角度、垂直振幅、躯干前倾、触地时间 | 手臂摆动 | - |
    | **正面视角** | 下肢力线(40%)、横向稳定性(35%)、对称性(25%) | 步频(估算)、垂直振幅(估算) | 触地时间 |

    **正面视角特点:**
    - 下肢力线：膝外翻/内翻角度、髋部下沉
    - 横向稳定性：髋部横摆、肩部倾斜
    - 对称性：左右动作对称性

    **建议：** 如需全面分析，建议同时拍摄侧面和正面视角视频
    """)

    st.subheader("AI分析设置")
    st.markdown("**使用智谱AI (glm-4.6模型)**")
    st.caption("需要安装zai库：pip install zai")

    # 显示智谱AI状态
    import os
    from config.config import AI_CONFIG
    api_key = AI_CONFIG.get('api_key', '')

    if api_key:
        st.success("智谱AI已配置")
        st.caption(f"API Key: {api_key[:10]}...{api_key[-4:]}")
    else:
        st.warning("智谱AI未配置，将使用本地规则引擎")
        st.caption("请在 config/config.py 中配置 ZHIPU_API_KEY")

    st.subheader("评价维度")
    st.markdown("""
    **侧面视角评分维度:**
    | 维度 | 权重 | 说明 |
    |------|------|------|
    | 动作稳定性 | 30% | 躯干稳定、头部稳定、膝角稳定性 |
    | 跑步效率 | 40% | 垂直振幅、步频、触地时间 |
    | 跑姿标准度 | 30% | 膝关节角度、躯干前倾 |

    **正面视角评分维度（独立系统）:**
    | 维度 | 权重 | 说明 |
    |------|------|------|
    | 下肢力线 | 35% | 膝外翻角度、髋部下沉、着地位置 |
    | 横向稳定性 | 35% | 髋部摆动、肩部倾斜、躯干侧向 |
    | 效率 | 30% | 步频、垂直振幅 |
    """)


# ========== 评分调试模块 ==========

def display_debug_scoring_details(quality_results: dict, kinematic_results: dict, view_angle: str):
    """
    显示评分调试详情

    展示详细的评分构成，帮助定位评分问题和数据准确性问题

    Args:
        quality_results: 质量评估结果
        kinematic_results: 运动学分析结果
        view_angle: 视角类型 ('side' 或 'front')
    """
    import pandas as pd

    st.markdown("---")
    st.subheader("🔧 评分详情（调试模式）")
    st.caption("展示详细的评分构成，用于定位评分问题和数据准确性问题")

    # 参考范围定义
    REFERENCE_RANGES = {
        'vertical_amplitude': '≤8%优秀, 8-12%良好, 12-16%一般, >16%待改进',
        'cadence': '≥185精英, 175-185优秀, 165-175良好, 155-165一般, <155较差',
        'trunk_lean': '3-8°最优, 1-12°良好, 0-15°可接受',
        'knee_angle_landing': '150-165°最优, 140-170°可接受',
        'ground_contact_time': '<210ms精英, 210-240ms优秀, 240-270ms良好, 270-300ms一般',
        'knee_valgus': '<5°优秀, 5-10°良好, 10-15°一般, >15°较差',
        'hip_drop': '<3°优秀, 3-5°良好, 5-8°一般, >8°较差',
        'shoulder_tilt': '<3°优秀, 3-6°良好, 6-10°一般, >10°较差',
        'lateral_sway': '<3%髋宽优秀, 3-6%良好, 6-10%一般, >10%较差',
    }

    is_frontal = view_angle in ['front', 'back']

    if is_frontal:
        _display_frontal_debug(quality_results, kinematic_results, REFERENCE_RANGES)
    else:
        _display_side_debug(quality_results, kinematic_results, REFERENCE_RANGES)


def _display_side_debug(quality_results: dict, kinematic_results: dict, refs: dict):
    """侧面视角调试详情 - 展示维度子指标构成"""
    import pandas as pd

    with st.expander("🔍 侧面视角评分明细", expanded=False):
        # === 总分概览 ===
        total_score = quality_results.get('total_score', 0)
        rating = quality_results.get('rating', '未知')

        col1, col2 = st.columns(2)
        col1.metric("总分", f"{total_score:.1f}")
        col2.metric("评级", rating)

        st.markdown("---")

        # ========== 【稳定性】维度 (30%) ==========
        st.markdown("### 【稳定性】维度")
        dimensions = quality_results.get('dimension_scores', {})

        # 先计算子指标得分，再显示维度得分（确保一致性）
        stability_data = []

        # 1. 躯干稳定性 (40%)
        stability = kinematic_results.get('stability', {})
        trunk_stability = stability.get('overall', 0) if isinstance(stability, dict) else 0
        stability_data.append({
            '子指标': '躯干稳定性',
            '权重': '40%',
            '原始值': f"{trunk_stability:.1f}",
            '得分': f"{trunk_stability:.0f}",
            '说明': '来自 stability.overall'
        })

        # 2. 膝角稳定性 (35%) - 优先使用触地期std
        angles = kinematic_results.get('angles', {})
        phase_analysis = angles.get('phase_analysis', {})
        gc = phase_analysis.get('ground_contact', {})
        gc_std = gc.get('std', 0)

        if gc_std > 0:
            avg_knee_std = gc_std
            std_source = '触地期'
        else:
            knee_std_l = angles.get('knee_left_std', 0)
            knee_std_r = angles.get('knee_right_std', 0)
            avg_knee_std = (knee_std_l + knee_std_r) / 2 if (knee_std_l + knee_std_r) > 0 else 0
            std_source = '全序列'

        if avg_knee_std <= 5:
            knee_stab_score = 95
        elif avg_knee_std <= 8:
            knee_stab_score = 80
        elif avg_knee_std <= 12:
            knee_stab_score = 65
        else:
            knee_stab_score = max(40, 100 - avg_knee_std * 2)
        stability_data.append({
            '子指标': '膝角稳定性',
            '权重': '35%',
            '原始值': f"{std_source}std={avg_knee_std:.1f}°",
            '得分': f"{knee_stab_score:.0f}",
            '说明': '≤5°=95, ≤8°=80, ≤12°=65'
        })

        # 3. 垂直稳定性 (25%) — 使用 std_position（归一化坐标下的位置标准差）
        vm = kinematic_results.get('vertical_motion', {})
        pos_std = vm.get('std_position', 0)
        if pos_std < 0.008:
            vert_stab_score = 95
        elif pos_std < 0.015:
            vert_stab_score = 80
        elif pos_std < 0.025:
            vert_stab_score = 65
        else:
            vert_stab_score = 50
        stability_data.append({
            '子指标': '垂直稳定性',
            '权重': '25%',
            '原始值': f"std={pos_std:.4f}",
            '得分': f"{vert_stab_score:.0f}",
            '说明': '<0.008=95, <0.015=80, <0.025=65'
        })

        # 使用本地重算值作为维度得分（避免与 evaluator 缓存不同步）
        stability_score = trunk_stability * 0.4 + knee_stab_score * 0.35 + vert_stab_score * 0.25
        st.markdown(f"**维度得分: {stability_score:.1f}** (权重30%, 加权贡献: {stability_score*0.30:.1f})")

        st.dataframe(pd.DataFrame(stability_data), hide_index=True, width='stretch')
        st.caption(f"计算: {trunk_stability:.0f}×0.4 + {knee_stab_score:.0f}×0.35 + {vert_stab_score:.0f}×0.25 = {stability_score:.1f}")

        st.markdown("---")

        # ========== 【效率】维度 (40%) ==========
        st.markdown("### 【效率】维度")

        # 先计算子指标得分，再显示维度得分（确保一致性）
        efficiency_data = []
        eff_scores = []

        # 1. 垂直振幅 (等权)
        amp_norm = vm.get('amplitude_normalized', 0)
        amp_rating = vm.get('amplitude_rating', {})
        amp_score = amp_rating.get('score', 0)
        if amp_score == 0 and amp_norm > 0:
            if amp_norm <= 8: amp_score = 100
            elif amp_norm <= 12: amp_score = 80
            elif amp_norm <= 16: amp_score = 60
            else: amp_score = 40
        if amp_norm > 0:
            eff_scores.append(amp_score)
        efficiency_data.append({
            '子指标': '垂直振幅',
            '权重': '等权',
            '原始值': f"{amp_norm:.2f}%",
            '等级': amp_rating.get('level', '-'),
            '得分': f"{amp_score:.0f}" if amp_norm > 0 else '-',
            '参考': refs['vertical_amplitude']
        })

        # 2. 步频 (等权)
        cadence_data = kinematic_results.get('cadence', {})
        cadence_val = cadence_data.get('cadence', 0)
        cadence_rating = cadence_data.get('rating', {})
        cadence_score = cadence_rating.get('score', 0)
        if cadence_score == 0 and cadence_val > 0:
            if cadence_val >= 185: cadence_score = 100
            elif cadence_val >= 175: cadence_score = 90
            elif cadence_val >= 165: cadence_score = 75
            elif cadence_val >= 155: cadence_score = 60
            else: cadence_score = 45
        if cadence_val > 0:
            eff_scores.append(cadence_score)
        efficiency_data.append({
            '子指标': '步频',
            '权重': '等权',
            '原始值': f"{cadence_val:.1f} 步/分",
            '等级': cadence_rating.get('level', '-'),
            '得分': f"{cadence_score:.0f}" if cadence_val > 0 else '-',
            '参考': refs['cadence']
        })

        # 3. 触地时间 (等权)
        gait = kinematic_results.get('gait_cycle', {})
        gait_rating = gait.get('gait_rating', {})
        gait_score = gait_rating.get('score', 0)
        gc_time_ms = gait.get('phase_duration_ms', {}).get('ground_contact', 0)
        if gait_score > 0:
            eff_scores.append(gait_score)
            efficiency_data.append({
                '子指标': '触地时间',
                '权重': '等权',
                '原始值': f"{gc_time_ms:.0f} ms",
                '等级': gait_rating.get('level', '-'),
                '得分': f"{gait_score:.0f}",
                '参考': '<210ms精英, 210-240优秀, 240-270良好, 270-300一般'
            })

        # 使用本地重算值作为维度得分（避免与 evaluator 缓存不同步）
        if eff_scores:
            efficiency_score = sum(eff_scores) / len(eff_scores)
        else:
            efficiency_score = dimensions.get('efficiency', 0)
        st.markdown(f"**维度得分: {efficiency_score:.1f}** (权重40%, 加权贡献: {efficiency_score*0.40:.1f})")

        st.dataframe(pd.DataFrame(efficiency_data), hide_index=True, width='stretch')
        if eff_scores:
            st.caption(f"计算: ({' + '.join([f'{s:.0f}' for s in eff_scores])}) / {len(eff_scores)} = {efficiency_score:.1f}")

        st.markdown("---")

        # ========== 【跑姿】维度 (30%) ==========
        st.markdown("### 【跑姿】维度")

        # 先计算子指标得分，再显示维度得分（确保一致性）
        form_data = []
        form_scores = []

        phase_analysis = angles.get('phase_analysis', {})

        # 1. 触地期膝角
        gc = phase_analysis.get('ground_contact', {})
        gc_mean = gc.get('mean', 0)
        gc_rating = gc.get('rating', {})
        gc_score = gc_rating.get('score', 0)
        if gc_score == 0 and gc_mean > 0:
            if 150 <= gc_mean <= 165: gc_score = 100
            elif 140 <= gc_mean <= 170: gc_score = 75
            else: gc_score = 50
        if gc_mean > 0:
            form_scores.append(gc_score)
            form_data.append({
                '子指标': '触地期膝角',
                '权重': '等权',
                '原始值': f"{gc_mean:.1f}°",
                '等级': gc_rating.get('level', '-'),
                '得分': f"{gc_score:.0f}",
                '参考': '150-165°理想, 140-170°可接受'
            })

        # 2. 最大弯曲膝角 - 添加数据有效性检查
        max_flex = phase_analysis.get('max_flexion', 0)
        if max_flex >= 80:
            if 90 <= max_flex <= 130: flex_score = 100
            elif 80 <= max_flex <= 140: flex_score = 75
            else: flex_score = 50
            form_scores.append(flex_score)
            form_data.append({
                '子指标': '最大弯曲膝角',
                '权重': '等权',
                '原始值': f"{max_flex:.1f}°",
                '等级': '-',
                '得分': f"{flex_score:.0f}",
                '参考': '90-130°理想, 80-140°可接受'
            })
        elif max_flex > 0:
            form_data.append({
                '子指标': '最大弯曲膝角',
                '权重': '-',
                '原始值': f"{max_flex:.1f}°",
                '等级': '数据异常',
                '得分': '未计入',
                '参考': '需要更多步态周期数据'
            })

        # 3. 活动范围 - 添加数据有效性检查
        rom = phase_analysis.get('range_of_motion', 0)
        if 20 <= rom <= 100:
            if 40 <= rom <= 70: rom_score = 100
            elif 30 <= rom <= 80: rom_score = 75
            else: rom_score = 50
            form_scores.append(rom_score)
            form_data.append({
                '子指标': '活动范围',
                '权重': '等权',
                '原始值': f"{rom:.1f}°",
                '等级': '-',
                '得分': f"{rom_score:.0f}",
                '参考': '40-70°理想, 30-80°可接受'
            })
        elif rom > 0:
            form_data.append({
                '子指标': '活动范围',
                '权重': '-',
                '原始值': f"{rom:.1f}°",
                '等级': '数据异常',
                '得分': '未计入',
                '参考': '需要更多步态周期数据'
            })

        # 4. 身体前倾
        body_lean = kinematic_results.get('body_lean', {})
        lean_val = body_lean.get('forward_lean', 0)
        lean_rating = body_lean.get('rating', {})
        lean_score = lean_rating.get('score', 0)
        if lean_score > 0:
            form_scores.append(lean_score)
            form_data.append({
                '子指标': '身体前倾',
                '权重': '等权',
                '原始值': f"{lean_val:.1f}°",
                '等级': lean_rating.get('level', '-'),
                '得分': f"{lean_score:.0f}",
                '参考': refs['trunk_lean']
            })

        # 使用本地重算值作为维度得分（避免与 evaluator 缓存不同步）
        if form_scores:
            form_score = sum(form_scores) / len(form_scores)
        else:
            form_score = dimensions.get('form', 0)
        st.markdown(f"**维度得分: {form_score:.1f}** (权重30%, 加权贡献: {form_score*0.30:.1f})")

        if form_data:
            st.dataframe(pd.DataFrame(form_data), hide_index=True, width='stretch')
        if form_scores:
            st.caption(f"计算: ({' + '.join([f'{s:.0f}' for s in form_scores])}) / {len(form_scores)} = {form_score:.1f}")

        st.markdown("---")

        # === 膝角详细数据 ===
        if phase_analysis:
            st.markdown("### 膝关节角度各阶段详情")

            phase_names = {
                'ground_contact': '触地期',
                'flight': '腾空期',
                'transition': '过渡期'
            }

            phase_data = []
            for phase_key, phase_info in phase_analysis.items():
                if isinstance(phase_info, dict) and phase_key in phase_names:
                    rating_info = phase_info.get('rating', {})
                    phase_data.append({
                        '阶段': phase_names.get(phase_key, phase_key),
                        '平均值': f"{phase_info.get('mean', 0):.1f}°",
                        '标准差': f"±{phase_info.get('std', 0):.1f}°",
                        '样本数': str(phase_info.get('count', '-')),
                        '等级': rating_info.get('level', '-'),
                        '得分': str(rating_info.get('score', '-')),
                        '评价': rating_info.get('description', '-')
                    })

            if phase_data:
                df_phase = pd.DataFrame(phase_data)
                st.dataframe(df_phase, hide_index=True, width='stretch')


def _display_frontal_debug(quality_results: dict, kinematic_results: dict, refs: dict):
    """正面视角调试详情（3维度重构版）"""
    import pandas as pd

    with st.expander("🔍 正面视角评分明细", expanded=False):
        # === 总分概览 ===
        total_score = quality_results.get('total_score', 0)
        rating = quality_results.get('rating', '未知')

        col1, col2 = st.columns(2)
        col1.metric("总分", f"{total_score:.1f}")
        col2.metric("评级", rating)

        st.markdown("---")

        # 获取维度得分
        frontal_dims = quality_results.get('frontal_dimension_scores', {})

        # ========== 【下肢力线】维度 (35%) ==========
        st.markdown("### 【下肢力线】维度")
        limb_score = frontal_dims.get('lower_limb_alignment', 0)
        st.markdown(f"**维度得分: {limb_score:.1f}** (权重35%, 加权贡献: {limb_score*0.35:.1f})")

        limb_data = []
        limb_scores = []
        limb_weights = []

        # 1. 膝外翻（取左右平均）(50%)
        lower_limb = kinematic_results.get('lower_limb_alignment', {})
        knee_valgus = lower_limb.get('knee_valgus', {})
        left_mean = abs(knee_valgus.get('left_mean', 0))
        right_mean = abs(knee_valgus.get('right_mean', 0))
        valgus_avg = (left_mean + right_mean) / 2 if (left_mean > 0 or right_mean > 0) else 0

        if valgus_avg > 0:
            # 基于阈值评分：0-5优秀(95), 5-10良好(80), 10-15一般(65), >15较差(45)
            if valgus_avg <= 5:
                valgus_score = 95
            elif valgus_avg <= 10:
                valgus_score = 80
            elif valgus_avg <= 15:
                valgus_score = 65
            else:
                valgus_score = 45
            limb_scores.append(valgus_score)
            limb_weights.append(0.5)
        else:
            valgus_score = 0
        limb_data.append({
            '子指标': '膝外翻',
            '权重': '50%',
            '原始值': f"左{left_mean:.1f}° / 右{right_mean:.1f}° (均{valgus_avg:.1f}°)" if valgus_avg > 0 else '无数据',
            '等级': '优秀' if valgus_avg <= 5 else ('良好' if valgus_avg <= 10 else ('一般' if valgus_avg <= 15 else '较差')) if valgus_avg > 0 else '-',
            '得分': f"{valgus_score:.0f}" if valgus_avg > 0 else '-',
            '参考': refs['knee_valgus']
        })

        # 2. 髋部下沉 (30%)
        hip_drop = lower_limb.get('hip_drop', {})
        drop_mean = abs(hip_drop.get('mean', hip_drop.get('drop_mean', 0)))
        if drop_mean > 0:
            # 基于阈值评分：0-3优秀(95), 3-5良好(80), 5-8一般(65), >8较差(45)
            if drop_mean <= 3:
                drop_score = 95
            elif drop_mean <= 5:
                drop_score = 80
            elif drop_mean <= 8:
                drop_score = 65
            else:
                drop_score = 45
            limb_scores.append(drop_score)
            limb_weights.append(0.3)
        else:
            drop_score = 0
        limb_data.append({
            '子指标': '髋部下沉',
            '权重': '30%',
            '原始值': f"{drop_mean:.1f}°" if drop_mean > 0 else '无数据',
            '等级': '优秀' if drop_mean <= 3 else ('良好' if drop_mean <= 5 else ('一般' if drop_mean <= 8 else '较差')) if drop_mean > 0 else '-',
            '得分': f"{drop_score:.0f}" if drop_mean > 0 else '-',
            '参考': refs['hip_drop']
        })

        # 3. 整体评级 (20%)
        overall_rating = lower_limb.get('overall_rating', {})
        overall_score = overall_rating.get('score', 0)
        if overall_score > 0:
            limb_scores.append(overall_score)
            limb_weights.append(0.2)
        limb_data.append({
            '子指标': '整体评级',
            '权重': '20%',
            '原始值': overall_rating.get('description', '-'),
            '等级': overall_rating.get('level', '-'),
            '得分': f"{overall_score:.0f}" if overall_score > 0 else '-',
            '参考': '综合评估'
        })

        st.dataframe(pd.DataFrame(limb_data), hide_index=True, width='stretch')
        if limb_scores and limb_weights:
            total_w = sum(limb_weights)
            calc_limb = sum(s * w for s, w in zip(limb_scores, limb_weights)) / total_w if total_w > 0 else 0
            weight_str = ' + '.join([f'{s:.0f}×{w:.0%}' for s, w in zip(limb_scores, limb_weights)])
            st.caption(f"计算: ({weight_str}) = {calc_limb:.1f}")

        st.markdown("---")

        # ========== 【横向稳定性】维度 (35%) ==========
        st.markdown("### 【横向稳定性】维度")
        lateral_dim_score = frontal_dims.get('lateral_stability', 0)
        st.markdown(f"**维度得分: {lateral_dim_score:.1f}** (权重35%, 加权贡献: {lateral_dim_score*0.35:.1f})")

        lateral_data = []
        lateral_scores = []
        lateral_weights = []

        # 1. 髋部横摆 (40%) - 相对髋宽的百分比
        lateral = kinematic_results.get('lateral_stability', {})
        hip_sway = lateral.get('hip_sway', lateral.get('hip_sway_normalized', 0) * 100)
        if hip_sway > 0:
            # 基于阈值评分（髋宽百分比）：<3%优秀(95), 3-6%良好(80), 6-10%一般(65), >10%较差(45)
            if hip_sway < 3:
                sway_score = 95
            elif hip_sway < 6:
                sway_score = 80
            elif hip_sway < 10:
                sway_score = 65
            else:
                sway_score = 45
            lateral_scores.append(sway_score)
            lateral_weights.append(0.4)
        else:
            sway_score = 0
        lateral_data.append({
            '子指标': '髋部横摆',
            '权重': '40%',
            '原始值': f"{hip_sway:.2f}%" if hip_sway > 0 else '无数据',
            '等级': '优秀' if hip_sway < 3 else ('良好' if hip_sway < 6 else ('一般' if hip_sway < 10 else '较差')) if hip_sway > 0 else '-',
            '得分': f"{sway_score:.0f}" if hip_sway > 0 else '-',
            '参考': refs['lateral_sway']
        })

        # 2. 肩部倾斜 (30%) - 角度（度）
        shoulder = kinematic_results.get('shoulder_analysis', {})
        tilt_mean = abs(shoulder.get('tilt_mean', 0))
        if tilt_mean > 0:
            # 基于阈值评分（度）：0-3°优秀(95), 3-6°良好(80), 6-10°一般(65), >10°较差(45)
            if tilt_mean <= 3:
                tilt_score = 95
            elif tilt_mean <= 6:
                tilt_score = 80
            elif tilt_mean <= 10:
                tilt_score = 65
            else:
                tilt_score = 45
            lateral_scores.append(tilt_score)
            lateral_weights.append(0.3)
        else:
            tilt_score = 0
        lateral_data.append({
            '子指标': '肩部倾斜',
            '权重': '30%',
            '原始值': f"{tilt_mean:.1f}°" if tilt_mean > 0 else '无数据',
            '等级': '优秀' if tilt_mean <= 3 else ('良好' if tilt_mean <= 6 else ('一般' if tilt_mean <= 10 else '较差')) if tilt_mean > 0 else '-',
            '得分': f"{tilt_score:.0f}" if tilt_mean > 0 else '-',
            '参考': refs['shoulder_tilt']
        })

        # 3. 对称性 (30%)
        # 膝外翻对称性
        sym_scores = []
        if left_mean > 0 or right_mean > 0:
            max_val = max(left_mean, right_mean) if max(left_mean, right_mean) > 0 else 1
            valgus_asym = abs(left_mean - right_mean) / max_val * 100
            if valgus_asym < 20:
                sym_scores.append(95)
            elif valgus_asym < 40:
                sym_scores.append(80)
            elif valgus_asym < 60:
                sym_scores.append(65)
            else:
                sym_scores.append(50)
        # 步态对称性
        sym_analysis = kinematic_results.get('symmetry_analysis', {})
        if 'overall_score' in sym_analysis:
            sym_scores.append(sym_analysis['overall_score'])

        if sym_scores:
            sym_score = sum(sym_scores) / len(sym_scores)
            lateral_scores.append(sym_score)
            lateral_weights.append(0.3)
        else:
            sym_score = 0
        lateral_data.append({
            '子指标': '对称性',
            '权重': '30%',
            '原始值': f"膝外翻差异{valgus_asym:.0f}%" if (left_mean > 0 or right_mean > 0) else '无数据',
            '等级': '-',
            '得分': f"{sym_score:.0f}" if sym_scores else '-',
            '参考': '左右差异<20%为优秀'
        })

        st.dataframe(pd.DataFrame(lateral_data), hide_index=True, width='stretch')
        if lateral_scores and lateral_weights:
            total_w = sum(lateral_weights)
            calc_lateral = sum(s * w for s, w in zip(lateral_scores, lateral_weights)) / total_w if total_w > 0 else 0
            weight_str = ' + '.join([f'{s:.0f}×{w:.0%}' for s, w in zip(lateral_scores, lateral_weights)])
            st.caption(f"计算: ({weight_str}) = {calc_lateral:.1f}")

        st.markdown("---")

        # ========== 【效率】维度 (30%) ==========
        st.markdown("### 【效率】维度")
        eff_dim_score = frontal_dims.get('efficiency', 0)
        st.markdown(f"**维度得分: {eff_dim_score:.1f}** (权重30%, 加权贡献: {eff_dim_score*0.30:.1f})")

        eff_data = []
        eff_scores = []
        eff_weights = []

        # 1. 步频 (60%)
        cadence_data = kinematic_results.get('cadence', {})
        cadence_val = cadence_data.get('cadence', 0)
        if cadence_val > 0:
            # 基于阈值评分：≥185精英(100), 175-185优秀(90), 165-175良好(75), 155-165一般(60), <155较差(45)
            if cadence_val >= 185:
                cadence_score = 100
            elif cadence_val >= 175:
                cadence_score = 90
            elif cadence_val >= 165:
                cadence_score = 75
            elif cadence_val >= 155:
                cadence_score = 60
            else:
                cadence_score = 45
            eff_scores.append(cadence_score)
            eff_weights.append(0.6)
        else:
            cadence_score = 0
        eff_data.append({
            '子指标': '步频',
            '权重': '60%',
            '原始值': f"{cadence_val:.0f} 步/分" if cadence_val > 0 else '无数据',
            '等级': '精英' if cadence_val >= 185 else ('优秀' if cadence_val >= 175 else ('良好' if cadence_val >= 165 else ('一般' if cadence_val >= 155 else '较差'))) if cadence_val > 0 else '-',
            '得分': f"{cadence_score:.0f}" if cadence_val > 0 else '-',
            '参考': refs['cadence']
        })

        # 2. 垂直振幅 (40%)
        vm = kinematic_results.get('vertical_motion', {})
        amp_norm = vm.get('normalized_amplitude', vm.get('amplitude_normalized', 0))
        if amp_norm > 0:
            # 基于阈值评分：≤8%优秀(100), 8-12%良好(80), 12-16%一般(60), >16%较差(40)
            if amp_norm <= 8:
                amp_score = 100
            elif amp_norm <= 12:
                amp_score = 80
            elif amp_norm <= 16:
                amp_score = 60
            else:
                amp_score = 40
            eff_scores.append(amp_score)
            eff_weights.append(0.4)
        else:
            amp_score = 0
        eff_data.append({
            '子指标': '垂直振幅',
            '权重': '40%',
            '原始值': f"{amp_norm:.2f}%" if amp_norm > 0 else '无数据',
            '等级': '优秀' if amp_norm <= 8 else ('良好' if amp_norm <= 12 else ('一般' if amp_norm <= 16 else '较差')) if amp_norm > 0 else '-',
            '得分': f"{amp_score:.0f}" if amp_norm > 0 else '-',
            '参考': refs['vertical_amplitude']
        })

        st.dataframe(pd.DataFrame(eff_data), hide_index=True, width='stretch')
        if eff_scores and eff_weights:
            total_w = sum(eff_weights)
            calc_eff = sum(s * w for s, w in zip(eff_scores, eff_weights)) / total_w if total_w > 0 else 0
            weight_str = ' + '.join([f'{s:.0f}×{w:.0%}' for s, w in zip(eff_scores, eff_weights)])
            st.caption(f"计算: ({weight_str}) = {calc_eff:.1f}")

        st.markdown("---")

        # === 下肢力线综合评估 ===
        if overall_rating:
            st.markdown("### 下肢力线综合评估")
            rating_col1, rating_col2, rating_col3 = st.columns(3)
            rating_col1.metric("综合得分", f"{overall_rating.get('score', 0):.0f}")
            rating_col2.metric("等级", overall_rating.get('level', '-'))
            rating_col3.metric("描述", overall_rating.get('description', '-'))


if __name__ == '__main__':
    main()
