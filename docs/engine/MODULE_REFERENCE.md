# 模块 API 参考文档

本文档提供各核心模块的功能说明和使用示例。

---

## 1. video_processor.py

### 功能
视频文件处理，包括帧抽取、分辨率统一、旋转校正。

### 主要类

```python
class VideoProcessor:
    def __init__(self, video_path: str)

    def extract_frames(self, max_frames: int = None) -> Tuple[List[np.ndarray], float, dict]
        """
        提取视频帧

        Returns:
            frames: 帧图像列表
            fps: 视频帧率
            video_info: 视频信息字典 {width, height, fps, frame_count, duration}
        """

    def get_video_info(self) -> dict
        """获取视频基本信息"""
```

### 使用示例

```python
from modules.video_processor import VideoProcessor

processor = VideoProcessor("running.mp4")
frames, fps, info = processor.extract_frames()
print(f"帧率: {fps}, 总帧数: {len(frames)}")
```

---

## 2. pose_estimator.py

### 功能
基于 MediaPipe Pose 的 2D 人体姿态估计，检测 33 个关键点。

### 主要类

```python
class PoseEstimator:
    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5)

    def process_frames(self, frames: List[np.ndarray]) -> List[Dict]
        """
        处理帧序列，返回关键点序列

        Returns:
            keypoints: List[{
                'frame_idx': int,
                'landmarks': List[{x, y, z, x_norm, y_norm, visibility}],
                'timestamp': float
            }]
        """

    def process_single_frame(self, frame: np.ndarray) -> Dict
        """处理单帧"""
```

### 关键点索引 (MediaPipe 33 点)

| 索引 | 关键点 | 索引 | 关键点 |
|------|--------|------|--------|
| 0 | nose | 11/12 | left/right shoulder |
| 23/24 | left/right hip | 25/26 | left/right knee |
| 27/28 | left/right ankle | 31/32 | left/right foot |

---

## 3. pose_lifter.py

### 功能
使用 DSTformer (MotionBERT 架构) 将 2D 姿态提升为 3D 姿态。

### 主要类

```python
class PoseLifter:
    def __init__(self, checkpoint_path: str = None,
                 device: str = 'auto')

    def lift_to_3d(self, keypoints_2d: List[Dict], fps: float) -> Tuple[np.ndarray, List[Dict]]
        """
        2D -> 3D 姿态提升

        Args:
            keypoints_2d: MediaPipe 2D 关键点序列
            fps: 视频帧率

        Returns:
            poses_3d: np.ndarray, shape (T, 17, 3), H36M 格式
            keypoints_3d: List[{has_3d, pose_3d: {joint_name: [x,y,z]}}]
        """

class KeypointMapper:
    """MediaPipe 33 点 -> H36M 17 点映射"""

    @staticmethod
    def mediapipe_to_h36m(landmarks: List[Dict]) -> np.ndarray
```

### H36M 17 关节名称

```python
H36M_JOINTS = [
    'pelvis', 'r_hip', 'r_knee', 'r_ankle',
    'l_hip', 'l_knee', 'l_ankle',
    'spine', 'neck', 'head', 'site',
    'l_shoulder', 'l_elbow', 'l_wrist',
    'r_shoulder', 'r_elbow', 'r_wrist'
]
```

---

## 4. kinematic_analyzer.py

### 功能
运动学特征分析，支持正面和侧面双视角。

### 主要类

```python
class KinematicAnalyzer:
    def __init__(self)

    def analyze_sequence(self,
                         keypoints_sequence: List[Dict],
                         fps: float,
                         view_angle: str = 'side',
                         poses_3d: np.ndarray = None,
                         keypoints_3d: List[Dict] = None) -> Dict
        """
        分析完整关键点序列

        Args:
            keypoints_sequence: 2D 关键点序列
            fps: 帧率
            view_angle: 'side' 或 'front'
            poses_3d: 可选 3D 姿态数组
            keypoints_3d: 可选 3D 关键点序列

        Returns:
            侧面视角返回:
                angles, vertical_motion, cadence, stride_info,
                stability, body_lean, arm_swing, gait_cycle

            正面视角返回:
                lower_limb_alignment, lateral_stability, cadence,
                shoulder_analysis, gait_symmetry, hip_sway
        """
```

### 侧面视角输出结构

```python
{
    'angles': {
        'knee_left': List[float],
        'knee_right': List[float],
        'phase_analysis': {
            'ground_contact': {'mean': float, 'std': float, 'rating': dict},
            'flight': {'mean': float, 'std': float}
        }
    },
    'vertical_motion': {
        'amplitude_normalized': float,  # 去趋势后按步态周期统计的躯干长度百分比
        'amplitude_rating': str
    },
    'cadence': {
        'cadence': float,  # 步/分
        'rating': str
    },
    'gait_cycle': {
        'phase_duration_ms': {
            'ground_contact': float,
            'flight': float
        }
    },
    'stability': {
        'overall': float,
        'trunk': float,
        'head': float
    },
    'body_lean': {
        'mean_lean': float,
        'std_lean': float,
        'forward_lean': float,
        'rating': {'level': str, 'score': float, 'description': str},
        'data_source': str
    }
}
```

### 正面视角输出结构

```python
{
    'lower_limb_alignment': {
        'left_leg': {'mean': float, 'issue': str},
        'right_leg': {'mean': float, 'issue': str},
        'hip_drop': {'mean': float},
        'asymmetry': float
    },
    'lateral_stability': {
        'hip_sway': {'relative_range': float},
        'shoulder_tilt': float
    },
    'cadence': {
        'cadence': float,
        'rating': str
    }
}
```

---

## 5. temporal_model.py

### 功能
时序深度学习分析（历史模块，当前默认停用）。

### 主要类

```python
class TemporalModelAnalyzer:
    def __init__(self, device: str = 'auto')

    def analyze(self, keypoints_sequence: List[Dict],
                poses_3d: np.ndarray = None) -> Dict
        """
        时序分析

        Returns:
            {
                'phases': List[int],  # 每帧阶段
                'quality_score': float,
                'phase_distribution': Dict[str, float],
                'stability_metrics': Dict
            }
        """
```

> 当前主流程不再调用该模块；保留此文件用于历史记录兼容与论文对照。

---

## 6. quality_evaluator.py

### 功能
技术质量评价，计算综合评分和各维度得分。

### 主要类

```python
class QualityEvaluator:
    def __init__(self)

    def evaluate(self, kinematic_results: Dict,
                 temporal_results: Dict = None,
                 view_angle: str = 'side') -> Dict
        """
        综合质量评价

        Returns:
            {
                'total_score': float,  # 0-100
                'grade': str,  # 'excellent'/'good'/'fair'/'poor'
                'dimension_scores': {
                    # 侧面: stability, efficiency, form
                    # 正面: lower_limb_alignment, lateral_stability, efficiency
                },
                'strengths': List[str],
                'weaknesses': List[str],
                'suggestions': List[str]
            }
        """
```

### 侧面评分权重

| 维度 | 权重 | 核心指标 |
|------|------|----------|
| stability | 30% | 躯干稳定性、头部稳定性 |
| efficiency | 40% | 步频、垂直振幅、触地时间 |
| form | 30% | 膝角、躯干前倾 |

### 正面评分权重

| 维度 | 权重 | 核心指标 |
|------|------|----------|
| lower_limb_alignment | 35% | 膝外翻、髋部下沉 |
| lateral_stability | 35% | 髋部横摆、肩部倾斜、对称性 |
| efficiency | 30% | 步频、垂直振幅 |

---

## 7. ai_analyzer.py

### 功能
AI 智能分析报告生成，支持智谱 AI 和本地规则引擎。

### 主要类

```python
class AIAnalyzer:
    def __init__(self, api_key: str = None)

    def generate_report(self,
                        kinematic_results: Dict,
                        quality_results: Dict,
                        view_angle: str = 'side') -> str
        """
        生成分析报告

        Args:
            kinematic_results: 运动学分析结果
            quality_results: 质量评价结果
            view_angle: 视角

        Returns:
            Markdown 格式的分析报告
        """

class LocalRuleEngine:
    """本地规则引擎，无需 API"""

    def generate_report(self, kinematic_results: Dict,
                        quality_results: Dict,
                        view_angle: str) -> str
```

---

## 8. database.py

### 功能
SQLite 数据持久化，存储分析记录。

### 主要类

```python
class AnalysisDatabase:
    def __init__(self, db_path: str = 'data/database.db')  # 原引擎兼容数据库；当前 Web 平台使用 data/webapp/running_web.db

    def save_analysis(self, video_path: str,
                      kinematic_results: Dict,
                      quality_results: Dict,
                      view_angle: str) -> int
        """保存分析记录，返回记录 ID"""

    def get_analysis(self, analysis_id: int) -> Dict
        """获取分析记录"""

    def get_history(self, limit: int = 50) -> List[Dict]
        """获取历史记录列表"""
```

---

## 完整分析流程示例

```python
from modules.video_processor import VideoProcessor
from modules.pose_estimator import PoseEstimator
from modules.pose_lifter import PoseLifter
from modules.kinematic_analyzer import KinematicAnalyzer
from modules.quality_evaluator import QualityEvaluator
from modules.ai_analyzer import AIAnalyzer

# 1. 视频处理
processor = VideoProcessor("running_side.mp4")
frames, fps, video_info = processor.extract_frames()

# 2. 2D 姿态估计
estimator = PoseEstimator()
keypoints_2d = estimator.process_frames(frames)

# 3. 3D 姿态提升
lifter = PoseLifter()
poses_3d, keypoints_3d = lifter.lift_to_3d(keypoints_2d, fps)

# 4. 运动学分析
analyzer = KinematicAnalyzer()
kinematic = analyzer.analyze_sequence(
    keypoints_2d, fps,
    view_angle='side',
    poses_3d=poses_3d,
    keypoints_3d=keypoints_3d
)

# 5. 质量评价
evaluator = QualityEvaluator()
quality = evaluator.evaluate(kinematic, view_angle='side')

# 6. AI 报告
ai_analyzer = AIAnalyzer()
report = ai_analyzer.generate_report(kinematic, quality, view_angle='side')

print(f"总分: {quality['total_score']:.1f}")
print(f"评级: {quality['grade']}")
print(report)
```
