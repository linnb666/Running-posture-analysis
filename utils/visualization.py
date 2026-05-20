import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


# H36M骨架连接定义（17个关键点）
H36M_SKELETON = [
    # 躯干
    (0, 7),   # hip -> spine
    (7, 8),   # spine -> thorax
    (8, 9),   # thorax -> neck
    (9, 10),  # neck -> head

    # 左腿
    (0, 4),   # hip -> left_hip
    (4, 5),   # left_hip -> left_knee
    (5, 6),   # left_knee -> left_ankle

    # 右腿
    (0, 1),   # hip -> right_hip
    (1, 2),   # right_hip -> right_knee
    (2, 3),   # right_knee -> right_ankle

    # 左臂
    (8, 11),  # thorax -> left_shoulder
    (11, 12), # left_shoulder -> left_elbow
    (12, 13), # left_elbow -> left_wrist

    # 右臂
    (8, 14),  # thorax -> right_shoulder
    (14, 15), # right_shoulder -> right_elbow
    (15, 16), # right_elbow -> right_wrist
]

# 关节点颜色（按部位分组）
H36M_JOINT_COLORS = {
    'hip': (255, 255, 255),      # 白色 - 髋部中心
    'spine': (200, 200, 200),    # 灰色 - 脊柱
    'thorax': (200, 200, 200),
    'neck': (200, 200, 200),
    'head': (255, 200, 100),     # 橙色 - 头部

    'left_hip': (100, 255, 100),    # 绿色 - 左腿
    'left_knee': (100, 255, 100),
    'left_ankle': (100, 255, 100),

    'right_hip': (100, 100, 255),   # 蓝色 - 右腿
    'right_knee': (100, 100, 255),
    'right_ankle': (100, 100, 255),

    'left_shoulder': (255, 255, 100),  # 黄色 - 左臂
    'left_elbow': (255, 255, 100),
    'left_wrist': (255, 255, 100),

    'right_shoulder': (255, 100, 255), # 紫色 - 右臂
    'right_elbow': (255, 100, 255),
    'right_wrist': (255, 100, 255),
}

# 骨骼连接颜色
H36M_BONE_COLORS = [
    (200, 200, 200),  # 躯干 - 灰色
    (200, 200, 200),
    (200, 200, 200),
    (200, 200, 200),

    (100, 255, 100),  # 左腿 - 绿色
    (100, 255, 100),
    (100, 255, 100),

    (100, 100, 255),  # 右腿 - 蓝色
    (100, 100, 255),
    (100, 100, 255),

    (255, 255, 100),  # 左臂 - 黄色
    (255, 255, 100),
    (255, 255, 100),

    (255, 100, 255),  # 右臂 - 紫色
    (255, 100, 255),
    (255, 100, 255),
]

H36M_JOINT_NAMES = [
    'hip', 'right_hip', 'right_knee', 'right_ankle',
    'left_hip', 'left_knee', 'left_ankle',
    'spine', 'thorax', 'neck', 'head',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_shoulder', 'right_elbow', 'right_wrist'
]


def create_3d_skeleton_video(
    frames: List[np.ndarray],
    keypoints_3d: List[Dict],
    output_path: str,
    fps: float = 30,
    show_original: bool = True
) -> bool:
    """
    创建3D骨架可视化视频

    Args:
        frames: 原始视频帧列表
        keypoints_3d: 3D关键点序列（来自Pose3DEstimator）
        output_path: 输出视频路径
        fps: 帧率
        show_original: 是否同时显示原始视频

    Returns:
        是否成功生成视频
    """
    if not frames or not keypoints_3d:
        print("⚠️ 无法生成3D骨架视频：缺少帧数据或3D关键点")
        return False

    # 检查是否有有效的3D数据
    valid_3d_count = sum(1 for kp in keypoints_3d if kp.get('has_3d', False))
    if valid_3d_count == 0:
        print("⚠️ 无法生成3D骨架视频：没有有效的3D关键点数据")
        return False

    print(f"   生成3D骨架视频: {valid_3d_count}/{len(keypoints_3d)} 帧有3D数据")

    h, w = frames[0].shape[:2]

    # 输出视频尺寸
    if show_original:
        output_w = w * 2
    else:
        output_w = w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, h))

    for i, (frame, kp_3d) in enumerate(zip(frames, keypoints_3d)):
        # 创建骨架画布（黑色背景）
        skeleton_frame = np.zeros_like(frame)

        if kp_3d.get('has_3d', False):
            # 获取3D姿态数据
            pose_3d = kp_3d.get('pose_3d', {})
            if pose_3d:
                # 将3D坐标投影到2D并绘制
                skeleton_frame = draw_3d_skeleton_on_frame(
                    skeleton_frame, pose_3d, w, h
                )

        # 合并原始帧和骨架帧
        if show_original:
            combined = np.hstack([frame, skeleton_frame])
        else:
            combined = skeleton_frame

        out.write(combined)

    out.release()
    print(f"   ✅ 3D骨架视频已保存: {output_path}")
    return True


def draw_3d_skeleton_on_frame(
    frame: np.ndarray,
    pose_3d: Dict,
    width: int,
    height: int,
    view_angle: str = 'side'
) -> np.ndarray:
    """
    在帧上绘制3D骨架（改进版：使用Z轴深度信息）

    Args:
        frame: 目标帧
        pose_3d: 3D姿态字典（短名称格式，如 'l_hip', 'r_knee'）
        width: 画布宽度
        height: 画布高度
        view_angle: 视角类型 ('side' 或 'frontal')

    Returns:
        绘制了骨架的帧

    改进说明：
    1. 根据视角选择投影平面（侧面: XY平面, 正面: ZY平面）
    2. 使用线条粗细和点大小表示深度（远处更细/更小）
    3. 使用颜色强度表示深度（远处更暗）
    """
    # 长名称到短名称的映射
    long_to_short = {
        'hip': 'hip', 'right_hip': 'r_hip', 'right_knee': 'r_knee',
        'right_ankle': 'r_ankle', 'left_hip': 'l_hip', 'left_knee': 'l_knee',
        'left_ankle': 'l_ankle', 'spine': 'spine', 'thorax': 'thorax',
        'neck': 'neck', 'head': 'head', 'left_shoulder': 'l_shoulder',
        'left_elbow': 'l_elbow', 'left_wrist': 'l_wrist',
        'right_shoulder': 'r_shoulder', 'right_elbow': 'r_elbow',
        'right_wrist': 'r_wrist'
    }

    # 首先收集所有有效的3D点
    valid_points_3d = []
    for joint_name in H36M_JOINT_NAMES:
        short_name = long_to_short.get(joint_name, joint_name)
        if short_name in pose_3d:
            point = pose_3d[short_name]
            if isinstance(point, np.ndarray) and len(point) >= 3:
                valid_points_3d.append(point[:3])  # 取完整XYZ

    if len(valid_points_3d) < 3:
        return frame  # 点太少，无法绘制

    valid_points_3d = np.array(valid_points_3d)

    # 根据视角选择投影平面
    if view_angle == 'side':
        # 侧面视角：X-Y平面（Z为深度）
        proj_x_idx, proj_y_idx, depth_idx = 0, 1, 2
    else:
        # 正面视角：Z-Y平面（X为深度）
        proj_x_idx, proj_y_idx, depth_idx = 2, 1, 0

    # 计算投影边界框
    proj_points = valid_points_3d[:, [proj_x_idx, proj_y_idx]]
    x_min, y_min = proj_points.min(axis=0)
    x_max, y_max = proj_points.max(axis=0)

    # 添加边距（增加到1.7，给下肢留出更多空间）
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    x_center = (x_min + x_max) / 2

    # 【修复】使用髋部(hip)作为垂直中心，而不是所有关节的平均
    # 髋部是身体重心，相对稳定，可以避免踝关节超出画布底边
    hip_point = pose_3d.get('hip')
    if hip_point is not None and isinstance(hip_point, np.ndarray) and len(hip_point) >= 3:
        y_center = hip_point[proj_y_idx]
    else:
        y_center = (y_min + y_max) / 2  # 回退到原方法

    scale = max(x_range, y_range) * 1.8  # 从1.3增加到1.8

    # 计算深度范围（用于归一化）
    depth_values = valid_points_3d[:, depth_idx]
    depth_min, depth_max = depth_values.min(), depth_values.max()
    depth_range = max(depth_max - depth_min, 0.01)

    # 提取所有3D点并转换为2D，同时记录深度和visibility
    points_2d = []
    valid_mask = []
    depth_normalized = []  # 0=最近, 1=最远
    visibility_values = []  # 原始2D visibility

    for joint_name in H36M_JOINT_NAMES:
        short_name = long_to_short.get(joint_name, joint_name)
        vis_key = f'{short_name}_vis'

        if short_name in pose_3d:
            point = pose_3d[short_name]
            # 获取原始2D visibility（如果有）
            vis = pose_3d.get(vis_key, 1.0)

            if isinstance(point, np.ndarray) and len(point) >= 3:
                # 投影坐标
                proj_x = point[proj_x_idx]
                proj_y = point[proj_y_idx]
                depth = point[depth_idx]

                # 归一化到[-0.5, 0.5]范围
                norm_x = (proj_x - x_center) / scale
                norm_y = (proj_y - y_center) / scale

                # 映射到画布坐标（添加垂直偏移，给下肢留更多空间）
                y_offset = 0.10  # 整体上移10%
                px = int((norm_x + 0.5) * width)
                py = int((norm_y + 0.5 - y_offset) * height)

                # 确保在画布范围内（clip作为安全保护，正常情况不应触发）
                px = np.clip(px, 0, width - 1)
                py = np.clip(py, 0, height - 1)

                # 计算归一化深度 (0=近, 1=远)
                d_norm = (depth - depth_min) / depth_range

                points_2d.append((px, py))
                valid_mask.append(True)
                depth_normalized.append(d_norm)
                visibility_values.append(vis)
            else:
                points_2d.append((0, 0))
                valid_mask.append(False)
                depth_normalized.append(0.5)
                visibility_values.append(0.0)
        else:
            points_2d.append((0, 0))
            valid_mask.append(False)
            depth_normalized.append(0.5)
            visibility_values.append(0.0)

    # 绘制骨骼连接（根据深度和visibility调整粗细和颜色）
    for bone_idx, (start, end) in enumerate(H36M_SKELETON):
        if valid_mask[start] and valid_mask[end]:
            base_color = H36M_BONE_COLORS[bone_idx]
            pt1 = points_2d[start]
            pt2 = points_2d[end]

            # 计算骨骼平均深度
            avg_depth = (depth_normalized[start] + depth_normalized[end]) / 2

            # 计算骨骼平均visibility（低visibility时降低渲染强度）
            avg_vis = (visibility_values[start] + visibility_values[end]) / 2

            # 根据深度调整线条粗细（近处粗，远处细）
            thickness = max(1, int(3 * (1 - 0.5 * avg_depth)))

            # 根据深度和visibility调整颜色强度
            # visibility低时颜色更暗，表示数据可靠性低
            depth_factor = 1 - 0.4 * avg_depth
            vis_factor = 0.3 + 0.7 * avg_vis  # 最低保留30%亮度
            intensity_factor = depth_factor * vis_factor
            color = tuple(int(c * intensity_factor) for c in base_color)

            cv2.line(frame, pt1, pt2, color, thickness)

    # 绘制关节点（根据深度和visibility调整大小和颜色）
    for idx, (point, is_valid) in enumerate(zip(points_2d, valid_mask)):
        if is_valid:
            joint_name = H36M_JOINT_NAMES[idx]
            base_color = H36M_JOINT_COLORS.get(joint_name, (255, 255, 255))
            d = depth_normalized[idx]
            vis = visibility_values[idx]

            # 根据深度调整点大小（近处大，远处小）
            point_size = max(2, int(5 * (1 - 0.4 * d)))

            # 根据深度和visibility调整颜色
            depth_factor = 1 - 0.4 * d
            vis_factor = 0.3 + 0.7 * vis  # 最低保留30%亮度
            intensity_factor = depth_factor * vis_factor
            color = tuple(int(c * intensity_factor) for c in base_color)

            cv2.circle(frame, point, point_size, color, -1)
            # 高visibility关节用白色边框，低visibility用灰色
            border_color = (255, 255, 255) if vis > 0.5 else (100, 100, 100)
            cv2.circle(frame, point, point_size + 1, border_color, 1)

    # 添加3D标识和视角信息
    view_label = "Side View" if view_angle == 'side' else "Frontal View"
    cv2.putText(frame, f"3D Skeleton ({view_label})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def create_comparison_video(original_frames: List[np.ndarray],
                            pose_frames: List[np.ndarray],
                            output_path: str,
                            fps: float = 30):
    """创建对比视频"""
    if not original_frames or not pose_frames:
        return

    h, w = original_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    for orig, pose in zip(original_frames, pose_frames):
        combined = np.hstack([orig, pose])
        out.write(combined)

    out.release()


def plot_angle_curves(angles: Dict, output_path: str):
    """绘制角度变化曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(angles['knee_left'])
    axes[0, 0].set_title('Left Knee Angle')
    axes[0, 0].set_ylabel('Degrees')

    axes[0, 1].plot(angles['knee_right'])
    axes[0, 1].set_title('Right Knee Angle')

    axes[1, 0].plot(angles['hip_left'])
    axes[1, 0].set_title('Left Hip Angle')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Degrees')

    axes[1, 1].plot(angles['hip_right'])
    axes[1, 1].set_title('Right Hip Angle')
    axes[1, 1].set_xlabel('Frame')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()