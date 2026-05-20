import sys
import argparse
from pathlib import Path

from config.config import OUTPUT_DIR, POSE_CONFIG, MOTIONBERT_CONFIG
from modules.video_processor import VideoProcessor
from modules.pose_estimator import create_pose_estimator, create_pose_estimator_3d
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer
from modules.database import DatabaseManager
# ViewAngleDetector removed - using manual view selection only
from utils.visualization import create_comparison_video, plot_angle_curves, create_3d_skeleton_video


def _disabled_temporal_result() -> dict:
    """时序模块停用占位结果（保留兼容字段）。"""
    return {
        'disabled': True,
        'status': 'deprecated',
        'model_type': 'disabled',
        'note': '时序模型已停用，不参与当前分析与评分流程',
        'phase_distribution': {}
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='跑步动作分析系统')
    parser.add_argument('video_path', type=str, help='视频文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化结果')
    parser.add_argument('--save-db', action='store_true', help='保存到数据库')
    parser.add_argument('--view', type=str, choices=['side', 'front', 'back'],
                        default='side', help='视频视角 (side=侧面, front=正面, back=背面)')
    parser.add_argument('--enable-3d', action='store_true', default=True,
                        help='启用MotionBERT 3D姿态提升 (默认启用)')
    parser.add_argument('--no-3d', action='store_true',
                        help='禁用3D姿态提升 (仅使用2D)')

    args = parser.parse_args()

    # 处理3D开关
    enable_3d = args.enable_3d and not args.no_3d

    # 验证视频文件
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"错误: 视频文件不存在 - {video_path}")
        sys.exit(1)

    # 设置输出目录
    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("基于深度学习的跑步动作视频解析与技术质量评价系统")
    print("=" * 80)
    print(f"视频文件: {video_path.name}")
    print(f"姿态估计后端: {POSE_CONFIG['backend'].upper()}")
    print(f"视角模式: {args.view}")
    print(f"3D姿态提升: {'✅ 启用 (MotionBERT)' if enable_3d else '❌ 禁用'}")
    print("=" * 80)

    try:
        # 执行分析
        results = run_analysis_pipeline(
            str(video_path), output_dir, args.visualize,
            view_mode=args.view,
            enable_3d=enable_3d
        )

        # 打印结果
        print_results(results)

        # 保存到数据库
        if args.save_db:
            db = DatabaseManager()
            record_id = db.save_analysis(results)
            print(f"\n💾 分析结果已保存到数据库 (ID: {record_id})")

        print("\n" + "=" * 80)
        print("✅ 分析完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_analysis_pipeline(video_path: str, output_dir: Path, visualize: bool = False,
                          view_mode: str = 'auto', enable_3d: bool = True):
    """
    运行完整分析流程

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        visualize: 是否生成可视化结果
        view_mode: 视角模式 ('auto', 'side', 'front', 'back')
        enable_3d: 是否启用3D姿态提升
    """

    # 1. 视频预处理
    print("\n1️⃣ 视频输入与预处理...")
    processor = VideoProcessor(video_path)
    video_info = processor.get_video_info()
    print(f"   分辨率: {video_info['width']}x{video_info['height']}")
    print(f"   帧率: {video_info['fps']:.2f} FPS")
    print(f"   时长: {video_info['duration']:.2f} 秒")

    # 帧提取策略：超过10秒取中间10秒，否则全部提取
    video_duration = video_info['duration']
    video_fps = video_info['fps']
    target_duration = min(10.0, video_duration)

    if video_duration > 10.0:
        start_time = (video_duration - target_duration) / 2
        start_frame = int(start_time * video_fps)
        max_frames = int(target_duration * 30)
        print(f"   📍 视频时长 {video_duration:.1f}s，提取中间 {target_duration:.0f} 秒 (从 {start_time:.1f}s 开始)")
    else:
        start_frame = 0
        max_frames = int(video_duration * 30)
        print(f"   📍 视频时长 {video_duration:.1f}s，分析完整视频")

    frames, fps = processor.extract_frames_from_position(
        start_frame=start_frame,
        target_fps=30,
        max_frames=max_frames
    )
    print(f"   提取帧数: {len(frames)}")

    # 2. 姿态估计（支持2D和3D）
    print("\n2️⃣ 人体姿态估计...")

    # 初始化变量
    keypoints_sequence = None
    keypoints_3d = None
    poses_3d = None
    has_3d = False

    if enable_3d:
        # 使用3D姿态估计器
        print("   使用3D姿态估计器 (MediaPipe + MotionBERT)...")
        try:
            estimator = create_pose_estimator_3d(
                backend_2d=POSE_CONFIG['backend'],
                enable_3d=True,
                device='auto'
            )
            pose_result = estimator.process_frames(
                frames,
                lift_to_3d=True,
                view_angle=view_mode
            )
            keypoints_sequence = pose_result['keypoints_2d']
            keypoints_3d = pose_result.get('keypoints_3d')
            poses_3d = pose_result.get('poses_3d')
            has_3d = pose_result.get('has_3d', False)

            if has_3d:
                print("   ✅ 3D姿态提升成功")
            else:
                print("   ⚠️ 3D姿态提升失败，使用2D数据")
        except Exception as e:
            print(f"   ⚠️ 3D估计器初始化失败: {e}")
            print("   回退到2D姿态估计...")
            estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
            keypoints_sequence = estimator.process_frames(frames)
    else:
        # 仅使用2D姿态估计器
        estimator = create_pose_estimator(POSE_CONFIG['backend'], POSE_CONFIG)
        keypoints_sequence = estimator.process_frames(frames)

    detected_count = sum(1 for kp in keypoints_sequence if kp['detected'])
    print(f"   检测成功: {detected_count}/{len(keypoints_sequence)} 帧 ({detected_count/len(keypoints_sequence)*100:.1f}%)")

    # 可视化姿态
    if visualize and detected_count > 0:
        print("   生成姿态可视化...")
        pose_frames = []
        for i, kp in enumerate(keypoints_sequence[:10]):  # 仅前10帧
            pose_frame = estimator.visualize_pose(frames[i], kp)
            pose_frames.append(pose_frame)

        # 保存第一帧
        import cv2
        cv2.imwrite(str(output_dir / 'pose_sample.jpg'), pose_frames[0])

    # 生成3D骨架视频（独立于visualize参数，只要有3D数据就生成）
    if has_3d and keypoints_3d:
        print("   生成3D骨架视频...")
        skeleton_video_path = str(output_dir / 'skeleton_3d.mp4')
        try:
            success = create_3d_skeleton_video(
                frames, keypoints_3d, skeleton_video_path,
                fps=fps, show_original=True
            )
            if not success:
                print("   ⚠️ 3D骨架视频生成失败")
        except Exception as e:
            print(f"   ⚠️ 3D骨架视频生成错误: {e}")

    # 3. 视角设置
    print("\n3️⃣ 视角设置...")
    detected_view = view_mode
    view_confidence = 1.0
    print(f"   使用视角: {get_view_name(detected_view)}")
    print(f"   分析策略: {get_strategy_description(detected_view)}")

    # 4. 运动学特征解析（支持3D数据）
    print("\n4️⃣ 运动学特征解析...")

    # 使用KinematicAnalyzer直接分析，传递3D数据
    kinematic_analyzer = KinematicAnalyzer()
    kinematic_results = kinematic_analyzer.analyze_sequence(
        keypoints_sequence, fps,
        view_angle=detected_view,
        poses_3d=poses_3d,
        keypoints_3d=keypoints_3d
    )

    # 显示数据可靠性信息
    if 'angles' in kinematic_results:
        reliability = kinematic_results['angles'].get('data_reliability', {})
        if reliability.get('is_3d', False):
            print("   📊 数据来源: 3D姿态 (高可靠性)")
        else:
            print("   ⚠️ 数据来源: 2D投影 (可能不准确)")

    # 基础指标输出
    cadence_data = kinematic_results['cadence']
    print(f"   步频: {cadence_data['cadence']:.1f} 步/分")
    print(f"   检测步数: {cadence_data['step_count']} 步 (视频时长 {cadence_data['duration']:.1f} 秒)")
    if cadence_data.get('confidence', 0) > 0:
        print(f"   步频置信度: {cadence_data['confidence']*100:.1f}%")

    # 垂直振幅（归一化）
    vertical_motion = kinematic_results.get('vertical_motion', {})
    # 优先使用归一化振幅（amplitude_normalized 是相对躯干长度的百分比）
    if 'amplitude_normalized' in vertical_motion:
        amplitude_pct = vertical_motion['amplitude_normalized']
        data_source = vertical_motion.get('data_source', '2D')
        source_label = '3D髋部' if data_source == '3D' else '2D髋部'
        print(f"   垂直振幅: {amplitude_pct:.2f}% (躯干长度) [{source_label}]")
        rating = vertical_motion.get('amplitude_rating', {})
        if rating:
            print(f"   振幅评级: {rating.get('description', '')}")
    elif vertical_motion.get('amplitude', 0) > 0:
        print(f"   垂直振幅: {vertical_motion['amplitude']:.4f} (归一化坐标)")
    else:
        print(f"   垂直振幅: 数据不足")

    # 膝关节角度分析（侧面视角）
    if detected_view in ['side', 'mixed']:
        angles = kinematic_results.get('angles', {})
        knee_angles = angles.get('knee', {})
        if 'phase_analysis' in knee_angles:
            print("   膝关节角度（分阶段）:")
            phase_analysis = knee_angles['phase_analysis']
            # 只处理阶段字典，跳过标量值（max_flexion等）
            phase_keys = ['ground_contact', 'flight', 'transition']
            for phase_name in phase_keys:
                if phase_name in phase_analysis:
                    phase_data = phase_analysis[phase_name]
                    phase_cn = {'ground_contact': '触地', 'flight': '腾空', 'transition': '过渡'}.get(phase_name, phase_name)
                    if isinstance(phase_data, dict) and phase_data.get('count', 0) > 0:
                        print(f"      {phase_cn}: {phase_data['mean']:.1f}° (范围: {phase_data['min']:.1f}°-{phase_data['max']:.1f}°)")
            # 显示关键指标
            if 'max_flexion' in phase_analysis:
                print(f"      最大屈曲: {phase_analysis['max_flexion']:.1f}°")
            if 'range_of_motion' in phase_analysis:
                print(f"      活动范围: {phase_analysis['range_of_motion']:.1f}°")

    # 可视化角度曲线
    if visualize and 'angles' in kinematic_results:
        print("   生成角度曲线图...")
        try:
            plot_angle_curves(kinematic_results['angles'],
                              str(output_dir / 'angle_curves.png'))
        except Exception as e:
            print(f"   警告: 无法生成角度曲线图 - {e}")

    # 5. 时序深度学习模块（已停用）
    print("\n5️⃣ 时序深度学习模块已停用...")
    temporal_results = _disabled_temporal_result()
    print("   （注：时序模型已停用，评分完全基于运动学规则）")

    # 6. 跑步技术质量评价
    print("\n6️⃣ 跑步技术质量评价...")
    quality_evaluator = QualityEvaluator()
    quality_results = quality_evaluator.evaluate(
        kinematic_results, temporal_results,
        view_angle=detected_view
    )

    print(f"   总体评分: {quality_results['total_score']:.2f}/100")
    print(f"   评级: {quality_results['rating']}")

    # 7. AI文本分析
    print("\n7️⃣ AI文本分析与报告生成...")
    ai_analyzer = AIAnalyzer()
    results_for_ai = {
        'quality_evaluation': quality_results,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results,
        'view_angle': detected_view
    }
    ai_text = ai_analyzer.generate_analysis_report(results_for_ai)

    # 保存AI报告
    with open(output_dir / 'ai_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(ai_text)
    print(f"   AI报告已保存: {output_dir / 'ai_analysis_report.txt'}")

    # 整合结果
    complete_results = {
        'video_info': video_info,
        'view_angle': detected_view,
        'view_confidence': view_confidence,
        'kinematic_analysis': kinematic_results,
        'temporal_analysis': temporal_results,
        'quality_evaluation': quality_results,
        'ai_analysis': ai_text,
        # 3D数据信息
        'pose_3d_info': {
            'enabled': enable_3d,
            'has_3d_data': has_3d,
            'description': '使用MotionBERT进行2D→3D姿态提升' if has_3d else '仅使用2D姿态数据'
        }
    }

    # 清理资源
    processor.release()
    estimator.close()

    return complete_results


def get_view_name(view: str) -> str:
    """获取视角中文名称"""
    names = {
        'side': '侧面视角',
        'front': '正面视角',
        'back': '背面视角',
        'mixed': '混合视角'
    }
    return names.get(view, view)


def get_strategy_description(view: str) -> str:
    """获取分析策略描述"""
    strategies = {
        'side': '膝关节角度 + 垂直振幅 + 躯干前倾',
        'front': '身体对称性 + 髋部稳定性 + 膝外翻检测',
        'back': '身体对称性 + 髋部稳定性 + 足跟外翻检测',
        'mixed': '综合分析（侧面+正面指标）'
    }
    return strategies.get(view, '标准分析')


def print_results(results: dict):
    """打印分析结果"""
    quality = results['quality_evaluation']
    kinematic = results.get('kinematic_analysis', {})
    temporal = results.get('temporal_analysis', {})
    view_angle = results.get('view_angle', 'unknown')

    print("\n" + "=" * 80)
    print("📊 分析结果汇总")
    print("=" * 80)

    # ==================== 数据来源信息 ====================
    print(f"\n📐 数据来源")
    print(f"   分析视角: {get_view_name(view_angle)}")

    pose_3d_info = results.get('pose_3d_info', {})
    if pose_3d_info.get('has_3d_data', False):
        print(f"   姿态数据: ✅ 3D姿态 (MotionBERT深度学习模型)")
    else:
        print(f"   姿态数据: 2D投影 (MediaPipe深度学习模型)")

    data_reliability = quality.get('data_reliability', {})
    if data_reliability:
        overall_reliability = data_reliability.get('overall', 'unknown')
        print(f"   数据可靠性: {'🟢 高' if overall_reliability == 'high' else '🟡 中等'}")

    # ==================== 技术质量评价（基于运动学规则）====================
    print(f"\n{'='*40}")
    print(f"🎯 技术质量评价（基于运动生物力学规则）")
    print(f"{'='*40}")

    print(f"\n   总体评分: {quality['total_score']:.1f}/100  【{quality['rating']}】")

    print(f"\n   各维度得分:")
    dims = quality.get('dimension_scores', {})
    dim_bars = {
        '稳定性': dims.get('stability', 0),
        '效率': dims.get('efficiency', 0),
        '跑姿': dims.get('form', 0)
    }
    for name, score in dim_bars.items():
        bar_len = int(score / 5)  # 20格满分
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"   {name}: [{bar}] {score:.1f}")

    if quality.get('strengths') and quality['strengths'][0] != '暂无突出优势':
        print(f"\n   ✅ 优势: {', '.join(quality['strengths'])}")

    if quality.get('weaknesses') and quality['weaknesses'][0] != '无明显薄弱项':
        print(f"   ⚠️  薄弱项: {', '.join(quality['weaknesses'])}")

    # ==================== 关键运动学指标 ====================
    print(f"\n{'='*40}")
    print(f"📏 关键运动学指标")
    print(f"{'='*40}")

    # 步频
    cadence_data = kinematic.get('cadence', {})
    if cadence_data:
        print(f"\n   步频: {cadence_data.get('cadence', 0):.0f} 步/分")

    # 垂直振幅
    vm = kinematic.get('vertical_motion', {})
    if 'amplitude_normalized' in vm:
        print(f"   垂直振幅: {vm['amplitude_normalized']:.1f}% (躯干长度)")

    # 触地时间
    gait = kinematic.get('gait_cycle', {})
    phase_duration = gait.get('phase_duration_ms', {})
    if phase_duration:
        gc_time = phase_duration.get('ground_contact', 0)
        if gc_time > 0:
            print(f"   触地时间: {gc_time:.0f} ms")

    # 膝关节角度
    angles = kinematic.get('angles', {})
    if 'phase_analysis' in angles:
        phase = angles['phase_analysis']
        gc_angle = phase.get('ground_contact', {}).get('mean', 0)
        flight_angle = phase.get('flight', {}).get('mean', 0)
        if gc_angle > 0:
            print(f"   触地期膝角: {gc_angle:.1f}°")
        if flight_angle > 0:
            print(f"   腾空期膝角: {flight_angle:.1f}°")

    # ==================== 深度学习分析（参考信息）====================
    print(f"\n{'='*40}")
    print(f"🤖 深度学习分析（参考信息）")
    print(f"{'='*40}")

    phase_dist = temporal.get('phase_distribution', {})
    if phase_dist:
        gc_pct = phase_dist.get('ground_contact', 0) * 100
        fl_pct = phase_dist.get('flight', 0) * 100
        tr_pct = phase_dist.get('transition', 0) * 100
        print(f"\n   步态阶段分布:")
        print(f"   触地期: {'█' * int(gc_pct/5)}{'░' * (20-int(gc_pct/5))} {gc_pct:.1f}%")
        print(f"   腾空期: {'█' * int(fl_pct/5)}{'░' * (20-int(fl_pct/5))} {fl_pct:.1f}%")
        print(f"   过渡期: {'█' * int(tr_pct/5)}{'░' * (20-int(tr_pct/5))} {tr_pct:.1f}%")

    print(f"\n   💡 说明: 阶段分布由LSTM/Transformer模型预测，仅供参考")

    # ==================== 改进建议 ====================
    if quality.get('suggestions'):
        print(f"\n{'='*40}")
        print(f"💡 改进建议")
        print(f"{'='*40}")
        for i, suggestion in enumerate(quality['suggestions'], 1):
            print(f"   {i}. {suggestion}")


if __name__ == '__main__':
    main()
