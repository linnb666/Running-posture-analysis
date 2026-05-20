# modules/kinematic_utils.py
"""
运动学分析工具函数

包含 3D 角度计算、坐标投影等通用工具函数。
从 kinematic_analyzer.py 中提取，供各分析模块复用。
"""
import numpy as np
from typing import Dict, Optional


def calculate_3d_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算3D空间中三点形成的关节角度

    Args:
        p1, p2, p3: 三个3D点坐标，角度在p2处

    Returns:
        角度（度），若计算失败返回np.nan
    """
    try:
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return np.nan

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)
    except:
        return np.nan


def project_point_to_plane(point: np.ndarray, view_angle: str) -> np.ndarray:
    """
    将3D点投影到指定视角的平面

    MotionBERT坐标系：
    - X轴：左右方向（跑步方向）
    - Y轴：上下方向（负=上，正=下）
    - Z轴：深度方向（前后）

    Args:
        point: 3D坐标 [x, y, z]
        view_angle: 'side'=侧面, 'front'/'back'=正面/背面

    Returns:
        投影后的2D坐标（第三维设为0）
    """
    if view_angle == 'side':
        # 侧面视角：投影到矢状面（XY平面），忽略Z轴深度
        # 这样得到的是"纯侧面"视角的角度
        return np.array([point[0], point[1], 0.0])
    elif view_angle in ['front', 'back']:
        # 正面/背面视角：投影到冠状面（YZ平面），忽略X轴
        return np.array([0.0, point[1], point[2]])
    else:
        # 默认返回原始3D点
        return point


def calculate_projected_joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                                     view_angle: str = None) -> float:
    """
    计算投影到指定平面后的关节角度

    对于侧面视角，投影到XY平面消除相机角度偏差的影响
    对于正面/背面视角，投影到YZ平面

    Args:
        p1, p2, p3: 三个3D点坐标，角度在p2处
        view_angle: 视角类型 ('side', 'front', 'back', None=使用完整3D)

    Returns:
        角度（度），若计算失败返回np.nan
    """
    try:
        # 如果指定了视角，先投影
        if view_angle:
            p1 = project_point_to_plane(p1, view_angle)
            p2 = project_point_to_plane(p2, view_angle)
            p3 = project_point_to_plane(p3, view_angle)

        v1 = p1 - p2
        v2 = p3 - p2

        # 投影后只用非零维度计算角度
        if view_angle == 'side':
            # XY平面，只用x和y
            v1 = v1[:2]
            v2 = v2[:2]
        elif view_angle in ['front', 'back']:
            # YZ平面，只用y和z
            v1 = v1[1:]
            v2 = v2[1:]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return np.nan

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)
    except:
        return np.nan


def get_3d_point_from_pose(pose_3d: Dict, joint_name: str) -> Optional[np.ndarray]:
    """
    从3D姿态中提取指定关节点

    Args:
        pose_3d: H36M格式的3D姿态字典
        joint_name: 关节名称 ('l_hip', 'r_knee', etc.)

    Returns:
        3D坐标数组，或None
    """
    if pose_3d is None or joint_name not in pose_3d:
        return None
    point = pose_3d[joint_name]
    if isinstance(point, np.ndarray):
        return point
    return np.array(point)


def smooth_signal(signal: np.ndarray, window_length: int = 15,
                  polyorder: int = 3) -> np.ndarray:
    """
    使用 Savitzky-Golay 滤波器平滑信号

    Args:
        signal: 输入信号
        window_length: 窗口长度（必须为奇数）
        polyorder: 多项式阶数

    Returns:
        平滑后的信号
    """
    from scipy.signal import savgol_filter

    if len(signal) < window_length:
        window_length = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        if window_length < 3:
            return signal

    return savgol_filter(signal, window_length, polyorder)


def filter_outliers_iqr(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    使用 IQR 方法过滤异常值

    Args:
        data: 输入数据
        k: IQR 乘数（默认1.5）

    Returns:
        过滤后的数据（异常值替换为 nan）
    """
    data = np.array(data, dtype=float)
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    result = data.copy()
    result[(data < lower_bound) | (data > upper_bound)] = np.nan

    return result


def get_rating(value: float, thresholds: Dict[str, float],
               higher_is_better: bool = True) -> str:
    """
    根据阈值获取评级

    Args:
        value: 数值
        thresholds: 阈值字典，包含 'excellent', 'good', 'fair' 等键
        higher_is_better: 是否数值越高越好

    Returns:
        评级字符串
    """
    if np.isnan(value):
        return "unknown"

    if higher_is_better:
        if value >= thresholds.get('excellent', float('inf')):
            return "excellent"
        elif value >= thresholds.get('good', float('inf')):
            return "good"
        elif value >= thresholds.get('fair', float('inf')):
            return "fair"
        else:
            return "poor"
    else:
        if value <= thresholds.get('excellent', float('-inf')):
            return "excellent"
        elif value <= thresholds.get('good', float('-inf')):
            return "good"
        elif value <= thresholds.get('fair', float('-inf')):
            return "fair"
        else:
            return "poor"
