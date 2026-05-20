"""
时序深度学习分析模块（升级版）

整合新的Transformer和CNN模型，支持：
1. 视角感知分析
2. Transformer阶段分类（可选CRF）
3. 多尺度TCN质量评估
4. 联合模型推理
5. 【新增】3D关键点输入支持（MotionBERT）

适用于毕业设计：基于深度学习的跑步动作视频解析与技术质量评价系统
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings

from config.config import (
    BODY_LEAN_THRESHOLDS,
    CHECKPOINT_DIR,
    MODEL_CONFIG,
    VERTICAL_AMPLITUDE_THRESHOLDS,
)


class TemporalModelAnalyzer:
    """
    时序深度学习分析器（升级版）

    支持多种模型后端：
    - legacy: 原有LSTM+CNN模型
    - transformer: 新的Transformer阶段分类模型
    - joint: 联合阶段分类和质量评估模型

    策略C优化：运动学特征注入 + 动态评分调整
    """

    # 视角ID映射
    VIEW_TO_ID = {'side': 0, 'front': 1, 'back': 2, 'mixed': 3}

    def __init__(self, model_type: str = 'joint', device: str = 'cpu'):
        """
        初始化模型分析器

        Args:
            model_type: 模型类型
                - 'legacy': 使用原有LSTM+CNN
                - 'transformer': 使用Transformer阶段分类
                - 'joint': 使用联合模型（推荐）
            device: 设备 ('cpu' 或 'cuda')
        """
        self.model_type = model_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 运动学特征存储（用于特征注入）
        self.kinematic_features = None

        # 加载模型
        self._load_models()

    def _load_models(self):
        """加载模型"""
        if self.model_type == 'legacy':
            self._load_legacy_models()
        elif self.model_type == 'transformer':
            self._load_transformer_models()
        elif self.model_type == 'joint':
            self._load_joint_model()
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

    def _load_legacy_models(self):
        """加载原有模型"""
        try:
            from models.lstm_model import RunningPhaseLSTM
            from models.cnn_model import RunningQualityCNN

            self.phase_model = RunningPhaseLSTM().to(self.device)
            self.quality_model = RunningQualityCNN().to(self.device)

            # 加载权重
            self._load_checkpoint(self.phase_model, 'phase_model.pth')
            self._load_checkpoint(self.quality_model, 'quality_model.pth')

            self.phase_model.eval()
            self.quality_model.eval()

            print(f"✅ 加载Legacy模型成功 (LSTM + CNN)")

        except Exception as e:
            warnings.warn(f"加载Legacy模型失败: {e}，使用随机初始化")
            self._init_random_legacy()

    def _load_transformer_models(self):
        """加载Transformer模型"""
        try:
            from models.transformer_model import RunningPhaseTransformer
            from models.quality_model import RunningQualityModel

            self.phase_model = RunningPhaseTransformer(
                d_model=128,
                num_heads=8,
                num_layers=4
            ).to(self.device)

            self.quality_model = RunningQualityModel(
                hidden_dim=128,
                num_levels=4
            ).to(self.device)

            # 加载权重
            self._load_checkpoint(self.phase_model, 'transformer_phase_model.pth')
            self._load_checkpoint(self.quality_model, 'quality_tcn_model.pth')

            self.phase_model.eval()
            self.quality_model.eval()

            print(f"✅ 加载Transformer模型成功")

        except Exception as e:
            warnings.warn(f"加载Transformer模型失败: {e}，回退到Legacy模型")
            self.model_type = 'legacy'
            self._load_legacy_models()

    def _load_joint_model(self):
        """加载联合模型"""
        try:
            from models.quality_model import JointPhaseQualityModel

            self.joint_model = JointPhaseQualityModel(
                hidden_dim=128,
                num_levels=4
            ).to(self.device)

            # 加载权重
            self._load_checkpoint(self.joint_model, 'joint_model.pth')
            self.joint_model.eval()

            print(f"✅ 加载联合模型成功")

        except Exception as e:
            warnings.warn(f"加载联合模型失败: {e}，回退到Transformer模型")
            self.model_type = 'transformer'
            self._load_transformer_models()

    def _load_checkpoint(self, model, filename: str) -> bool:
        """加载模型权重

        注意：如果权重文件不存在，模型将使用随机初始化，
        这会导致预测结果不可靠。请确保权重文件存在。
        """
        checkpoint_path = CHECKPOINT_DIR / filename

        if checkpoint_path.exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"   加载权重: {filename}")
                return True
            except Exception as e:
                warnings.warn(
                    f"加载权重 {filename} 失败: {e}。模型使用随机初始化，预测结果不可靠！",
                    RuntimeWarning
                )
                return False
        else:
            warnings.warn(
                f"权重文件 {filename} 不存在于 {CHECKPOINT_DIR}。"
                f"模型使用随机初始化，时序分析结果仅供参考！",
                RuntimeWarning
            )
            return False

    def _init_random_legacy(self):
        """初始化随机Legacy模型"""
        from models.lstm_model import RunningPhaseLSTM
        from models.cnn_model import RunningQualityCNN

        self.phase_model = RunningPhaseLSTM().to(self.device)
        self.quality_model = RunningQualityCNN().to(self.device)
        self.phase_model.eval()
        self.quality_model.eval()

    def set_kinematic_features(self, kinematic_results: Dict):
        """
        设置运动学特征（用于特征注入）

        Args:
            kinematic_results: 来自KinematicAnalyzer的分析结果
        """
        self.kinematic_features = kinematic_results

    def analyze(self, keypoints_sequence: List[Dict],
                view_angle: str = 'side',
                kinematic_results: Dict = None,
                keypoints_3d: List[Dict] = None) -> Dict:
        """
        分析关键点序列

        Args:
            keypoints_sequence: 关键点时间序列（2D）
            view_angle: 视角类型 ('side', 'front', 'back', 'mixed')
            kinematic_results: 可选的运动学分析结果（用于特征注入）
            keypoints_3d: 可选的3D关键点序列

        Returns:
            分析结果字典
        """
        # 保存运动学特征
        if kinematic_results:
            self.kinematic_features = kinematic_results

        # 准备输入数据（支持3D）
        input_tensor = self._prepare_input(keypoints_sequence, keypoints_3d)

        if input_tensor is None:
            print("⚠️ 输入数据不足，返回默认结果")
            return self._get_empty_results()

        # 视角ID
        view_id = torch.tensor([self.VIEW_TO_ID.get(view_angle, 0)], device=self.device)

        # 根据模型类型进行推理
        with torch.no_grad():
            if self.model_type == 'joint':
                results = self._analyze_joint(input_tensor, view_id)
            elif self.model_type == 'transformer':
                results = self._analyze_transformer(input_tensor, view_id)
            else:  # legacy
                results = self._analyze_legacy(input_tensor)

        # 策略C：运动学特征注入 - 动态调整评分
        results = self._apply_kinematic_adjustment(results)

        return results

    def _apply_kinematic_adjustment(self, results: Dict) -> Dict:
        """
        应用运动学特征注入，动态调整深度学习评分

        策略：DL原始评分 * 0.4 + 运动学评分 * 0.6
        这样既保留了深度学习的特征提取能力，又引入了可靠的规则修正
        """
        if not self.kinematic_features:
            return results

        # 提取运动学指标
        kinematic = self.kinematic_features

        # 计算运动学评分组件
        kinematic_scores = self._calculate_kinematic_scores(kinematic)

        # 融合权重
        dl_weight = 0.4  # 深度学习权重
        kinematic_weight = 0.6  # 运动学权重

        # 调整各维度评分
        original_quality = results.get('quality_score', 50)
        original_stability = results.get('quality_stability', 50)
        original_efficiency = results.get('quality_efficiency', 50)
        original_form = results.get('quality_form', 50)
        original_rhythm = results.get('quality_rhythm', 50)

        # 融合评分
        results['quality_score'] = float(
            original_quality * dl_weight +
            kinematic_scores['overall'] * kinematic_weight
        )
        results['quality_stability'] = float(
            original_stability * dl_weight +
            kinematic_scores['stability'] * kinematic_weight
        )
        results['quality_efficiency'] = float(
            original_efficiency * dl_weight +
            kinematic_scores['efficiency'] * kinematic_weight
        )
        results['quality_form'] = float(
            original_form * dl_weight +
            kinematic_scores['form'] * kinematic_weight
        )
        results['quality_rhythm'] = float(
            original_rhythm * dl_weight +
            kinematic_scores['rhythm'] * kinematic_weight
        )

        # 添加调整标记
        results['kinematic_adjusted'] = True
        results['kinematic_scores'] = kinematic_scores

        return results

    def _calculate_kinematic_scores(self, kinematic: Dict) -> Dict:
        """
        根据运动学特征计算评分

        评分规则基于专业跑步标准
        """
        scores = {
            'stability': 50.0,
            'efficiency': 50.0,
            'form': 50.0,
            'rhythm': 50.0,
            'overall': 50.0
        }

        # 1. 稳定性评分
        stability = kinematic.get('stability', {})
        if isinstance(stability, dict):
            stability_overall = stability.get('overall', 50)
            scores['stability'] = float(np.clip(stability_overall, 30, 100))

        # 2. 效率评分（基于步频和垂直振幅）
        efficiency_score = 50.0

        # 步频评分
        cadence_data = kinematic.get('cadence', {})
        if isinstance(cadence_data, dict):
            cadence = cadence_data.get('cadence', 0)
            if cadence > 0:
                # 精英：185+, 优秀：175-185, 良好：165-175, 一般：155-165, 较差：<155
                if cadence >= 185:
                    cadence_score = 95
                elif cadence >= 175:
                    cadence_score = 85
                elif cadence >= 165:
                    cadence_score = 72
                elif cadence >= 155:
                    cadence_score = 58
                else:
                    cadence_score = 40
                efficiency_score = cadence_score * 0.5

        # 垂直振幅评分
        vertical = kinematic.get('vertical_motion', {})
        if isinstance(vertical, dict):
            amplitude = vertical.get('amplitude_normalized', 0)
            if amplitude > 0:
                excellent_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['excellent_max'])
                good_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['good_max'])
                fair_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['fair_max'])

                # 统一使用稳健版阈值：≤11%优秀, 11-17%良好, 17-25%一般, >25%较差
                if amplitude <= excellent_max:
                    amp_score = 95
                elif amplitude <= good_max:
                    amp_score = 78
                elif amplitude <= fair_max:
                    amp_score = 55
                else:
                    amp_score = 35
                efficiency_score += amp_score * 0.5

        scores['efficiency'] = efficiency_score

        # 3. 跑姿评分（基于膝关节角度和躯干前倾）
        form_score = 50.0

        # 膝关节角度
        angles = kinematic.get('angles', {})
        if isinstance(angles, dict):
            phase_analysis = angles.get('phase_analysis', {})
            if phase_analysis:
                gc_angle = phase_analysis.get('ground_contact', {}).get('mean', 0)
                if gc_angle > 0:
                    # 理想触地角度：155-170度
                    if 155 <= gc_angle <= 170:
                        knee_score = 95
                    elif 145 <= gc_angle < 155 or 170 < gc_angle <= 180:
                        knee_score = 75
                    else:
                        knee_score = 50
                    form_score = knee_score * 0.5

        # 躯干前倾（中长跑/马拉松稳健阈值）
        body_lean = kinematic.get('body_lean', {})
        if isinstance(body_lean, dict):
            forward_lean = body_lean.get('forward_lean', 0)
            if forward_lean > 0:
                optimal_min = float(BODY_LEAN_THRESHOLDS['optimal_min'])
                optimal_max = float(BODY_LEAN_THRESHOLDS['optimal_max'])
                good_min = float(BODY_LEAN_THRESHOLDS['good_min'])
                good_max = float(BODY_LEAN_THRESHOLDS['good_max'])
                fair_min = float(BODY_LEAN_THRESHOLDS['fair_min'])
                fair_max = float(BODY_LEAN_THRESHOLDS['fair_max'])
                if optimal_min <= forward_lean <= optimal_max:
                    lean_score = 100  # 最优
                elif good_min <= forward_lean <= good_max:
                    lean_score = 85   # 良好
                elif fair_min <= forward_lean <= fair_max:
                    lean_score = 70   # 可接受
                else:
                    lean_score = 55   # 过于直立或过度前倾
                form_score += lean_score * 0.5

        scores['form'] = form_score

        # 4. 节奏评分（基于触地时间和相位分布）
        rhythm_score = 50.0

        gait_cycle = kinematic.get('gait_cycle', {})
        if isinstance(gait_cycle, dict):
            phase_duration = gait_cycle.get('phase_duration_ms', {})
            gc_time = phase_duration.get('ground_contact', 0)
            if gc_time > 0:
                # 精英：<210ms, 优秀：210-240ms, 良好：240-270ms, 一般：270-300ms
                if gc_time < 210:
                    gc_score = 98
                elif gc_time < 240:
                    gc_score = 85
                elif gc_time < 270:
                    gc_score = 70
                elif gc_time < 300:
                    gc_score = 55
                else:
                    gc_score = 40
                rhythm_score = gc_score

        scores['rhythm'] = rhythm_score

        # 5. 综合评分
        scores['overall'] = (
            scores['stability'] * 0.25 +
            scores['efficiency'] * 0.30 +
            scores['form'] * 0.25 +
            scores['rhythm'] * 0.20
        )

        return scores

    def _analyze_joint(self, input_tensor: torch.Tensor,
                       view_id: torch.Tensor) -> Dict:
        """使用联合模型分析"""
        outputs = self.joint_model(input_tensor, view_id)

        # 阶段预测
        phase_probs = torch.softmax(outputs['phase_logits'], dim=-1)
        phase_labels = torch.argmax(phase_probs, dim=-1)

        # 质量评分
        quality_scores = outputs['quality_scores'].cpu().numpy()[0]

        # 注意力权重
        attention_weights = outputs['attention_weights'].cpu().numpy()[0]

        # 计算阶段分布
        phase_distribution = self._calculate_phase_distribution(phase_labels[0])

        # 计算稳定性
        stability_score = self._calculate_stability_from_phases(phase_labels[0])

        return {
            'phase_sequence': phase_labels.cpu().numpy().tolist()[0],
            'phase_distribution': phase_distribution,
            'quality_score': float(quality_scores[0]),
            'quality_stability': float(quality_scores[1]),
            'quality_efficiency': float(quality_scores[2]),
            'quality_form': float(quality_scores[3]),
            'quality_rhythm': float(quality_scores[4]),
            'stability_score': stability_score,
            'attention_weights': attention_weights.tolist(),
            'model_type': 'joint'
        }

    def _analyze_transformer(self, input_tensor: torch.Tensor,
                             view_id: torch.Tensor) -> Dict:
        """使用Transformer模型分析"""
        # 阶段分类
        predictions, probs = self.phase_model.predict(input_tensor, view_id)

        # 质量评估
        quality_output = self.quality_model(input_tensor, view_id, return_attention=True)
        quality_scores = quality_output['scores'].cpu().numpy()[0]
        attention_weights = quality_output.get('attention_weights', None)

        # 计算阶段分布
        phase_distribution = self._calculate_phase_distribution(predictions[0])

        # 计算稳定性
        stability_score = self._calculate_stability_from_phases(predictions[0])

        results = {
            'phase_sequence': predictions.cpu().numpy().tolist()[0],
            'phase_distribution': phase_distribution,
            'quality_score': float(quality_scores[0]),
            'quality_stability': float(quality_scores[1]),
            'quality_efficiency': float(quality_scores[2]),
            'quality_form': float(quality_scores[3]),
            'quality_rhythm': float(quality_scores[4]),
            'stability_score': stability_score,
            'model_type': 'transformer'
        }

        if attention_weights is not None:
            results['attention_weights'] = attention_weights.cpu().numpy().tolist()[0]

        return results

    def _analyze_legacy(self, input_tensor: torch.Tensor) -> Dict:
        """使用Legacy模型分析"""
        # 阶段分类
        phase_output = self.phase_model(input_tensor)
        phase_probs = torch.softmax(phase_output, dim=-1)
        phase_labels = torch.argmax(phase_probs, dim=-1)

        # 质量评分
        quality_scores = self.quality_model(input_tensor).cpu().numpy()[0]

        # 计算阶段分布
        phase_distribution = self._calculate_phase_distribution(phase_labels[0])

        # 计算稳定性
        stability_score = self._calculate_stability_from_phases(phase_labels[0])

        return {
            'phase_sequence': phase_labels.cpu().numpy().tolist()[0],
            'phase_distribution': phase_distribution,
            'quality_score': float(quality_scores[0]),
            'quality_stability': float(quality_scores[1]),
            'quality_efficiency': float(quality_scores[2]),
            'quality_form': float(quality_scores[3]),
            'quality_rhythm': float(quality_scores[4]),
            'stability_score': stability_score,
            'model_type': 'legacy'
        }

    def _prepare_input(self, keypoints_sequence: List[Dict],
                        keypoints_3d: List[Dict] = None) -> Optional[torch.Tensor]:
        """
        准备模型输入（优化版：支持2D和3D数据）

        Args:
            keypoints_sequence: 2D关键点序列
            keypoints_3d: 可选的3D关键点序列

        Returns:
            模型输入张量
        """
        valid_frames = [kp for kp in keypoints_sequence if kp.get('detected', False)]

        if len(valid_frames) < MODEL_CONFIG['sequence_length']:
            print(f"⚠️ 有效帧数 {len(valid_frames)} < 最小序列长度 {MODEL_CONFIG['sequence_length']}")
            return None

        # 检查是否有3D数据
        use_3d = (keypoints_3d is not None and len(keypoints_3d) > 0 and
                  keypoints_3d[0].get('has_3d', False))

        # 使用更多帧进行分析（如果可用）
        use_frames = min(len(valid_frames), MODEL_CONFIG['sequence_length'] * 2)

        # 提取关键点坐标（归一化）+ 添加速度特征
        features = []
        prev_frame = None

        for i, kp in enumerate(valid_frames[:use_frames]):
            frame_features = []

            # 【3D模式】使用3D坐标（如果可用）
            if use_3d and i < len(keypoints_3d) and keypoints_3d[i].get('has_3d', False):
                landmarks_3d = keypoints_3d[i].get('landmarks_3d', {})
                for j, landmark in enumerate(kp['landmarks']):
                    # 优先使用3D坐标，否则回退到2D
                    if j in landmarks_3d:
                        lm_3d = landmarks_3d[j]
                        frame_features.extend([lm_3d['x_3d'], lm_3d['y_3d']])
                    else:
                        frame_features.extend([landmark['x_norm'], landmark['y_norm']])
            else:
                # 【2D模式】使用2D坐标
                for j, landmark in enumerate(kp['landmarks']):
                    frame_features.extend([landmark['x_norm'], landmark['y_norm']])

            # 添加速度特征（如果有前一帧）
            if prev_frame is not None and len(frame_features) == len(prev_frame):
                velocity_features = []
                for k in range(0, len(frame_features), 2):
                    if k + 1 < len(frame_features) and k + 1 < len(prev_frame):
                        dx = frame_features[k] - prev_frame[k]
                        dy = frame_features[k + 1] - prev_frame[k + 1]
                        velocity_features.extend([dx * 10, dy * 10])

            # 只在前30帧使用完整特征
            if i < MODEL_CONFIG['sequence_length']:
                features.append(frame_features[:MODEL_CONFIG['input_dim']])

            prev_frame = frame_features

        # 确保有足够的帧
        while len(features) < MODEL_CONFIG['sequence_length']:
            features.append(features[-1] if features else [0] * MODEL_CONFIG['input_dim'])

        # 转换为tensor
        input_tensor = torch.FloatTensor(features[:MODEL_CONFIG['sequence_length']]).unsqueeze(0).to(self.device)

        # 数据归一化（使用更稳健的方法）
        mean = input_tensor.mean(dim=1, keepdim=True)
        std = input_tensor.std(dim=1, keepdim=True) + 1e-6
        input_tensor = (input_tensor - mean) / std

        return input_tensor

    def _calculate_phase_distribution(self, phase_sequence: torch.Tensor) -> Dict[str, float]:
        """计算阶段分布"""
        total = len(phase_sequence)
        if total == 0:
            return {'ground_contact': 0, 'flight': 0, 'transition': 0}

        ground_contact = float(torch.sum(phase_sequence == 0).item() / total)
        flight = float(torch.sum(phase_sequence == 1).item() / total)
        transition = float(torch.sum(phase_sequence == 2).item() / total)

        return {
            'ground_contact': ground_contact,
            'flight': flight,
            'transition': transition
        }

    def _calculate_stability_from_phases(self, phase_sequence: torch.Tensor) -> float:
        """从阶段序列计算稳定性（优化版：更灵活的计算）"""
        if len(phase_sequence) < 2:
            return 50.0  # 默认中等稳定性

        # 计算阶段转换次数
        transitions = torch.sum(phase_sequence[:-1] != phase_sequence[1:]).item()

        # 计算阶段分布的均匀性
        phase_counts = torch.bincount(phase_sequence, minlength=3).float()
        phase_ratios = phase_counts / len(phase_sequence)

        # 理想的步态应该有合理的触地/腾空比例
        # 触地约40-50%，腾空约30-40%，过渡约10-20%
        ideal_ratios = torch.tensor([0.45, 0.35, 0.20])
        ratio_diff = torch.abs(phase_ratios - ideal_ratios).sum().item()

        # 计算节奏规律性（相邻阶段持续时间的一致性）
        segment_lengths = []
        current_phase = phase_sequence[0].item()
        segment_start = 0

        for i in range(1, len(phase_sequence)):
            if phase_sequence[i].item() != current_phase:
                segment_lengths.append(i - segment_start)
                current_phase = phase_sequence[i].item()
                segment_start = i

        if len(segment_lengths) > 1:
            rhythm_consistency = 100 - min(np.std(segment_lengths) * 5, 50)
        else:
            rhythm_consistency = 50

        # 综合评分
        transition_score = max(0, 100 - transitions * 1.5)  # 降低惩罚系数
        ratio_score = max(0, 100 - ratio_diff * 100)
        rhythm_score = rhythm_consistency

        # 加权平均
        stability = (transition_score * 0.4 + ratio_score * 0.3 + rhythm_score * 0.3)

        return float(max(0, min(100, stability)))

    def _get_empty_results(self) -> Dict:
        """返回空结果"""
        return {
            'phase_sequence': [],
            'phase_distribution': {'ground_contact': 0, 'flight': 0, 'transition': 0},
            'quality_score': 0.0,
            'quality_stability': 0.0,
            'quality_efficiency': 0.0,
            'quality_form': 0.0,
            'quality_rhythm': 0.0,
            'stability_score': 0.0,
            'model_type': self.model_type
        }


class TemporalModelEnsemble:
    """
    模型集成分析器

    结合多个模型的预测，提高准确性和鲁棒性
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.analyzers = []

        # 尝试加载多个模型
        for model_type in ['joint', 'transformer', 'legacy']:
            try:
                analyzer = TemporalModelAnalyzer(model_type=model_type, device=device)
                self.analyzers.append(analyzer)
            except Exception as e:
                print(f"⚠️ 无法加载 {model_type} 模型: {e}")

        if not self.analyzers:
            raise RuntimeError("没有可用的模型")

        print(f"✅ 集成分析器初始化完成，共 {len(self.analyzers)} 个模型")

    def analyze(self, keypoints_sequence: List[Dict],
                view_angle: str = 'side') -> Dict:
        """
        集成分析

        Args:
            keypoints_sequence: 关键点序列
            view_angle: 视角

        Returns:
            集成结果
        """
        results_list = []

        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(keypoints_sequence, view_angle)
                results_list.append(result)
            except Exception as e:
                print(f"⚠️ {analyzer.model_type} 模型分析失败: {e}")

        if not results_list:
            return self.analyzers[0]._get_empty_results()

        # 集成结果
        return self._ensemble_results(results_list)

    def _ensemble_results(self, results_list: List[Dict]) -> Dict:
        """集成多个模型的结果"""
        n = len(results_list)

        # 质量评分取平均
        quality_score = np.mean([r['quality_score'] for r in results_list])
        quality_stability = np.mean([r['quality_stability'] for r in results_list])
        quality_efficiency = np.mean([r['quality_efficiency'] for r in results_list])
        quality_form = np.mean([r['quality_form'] for r in results_list])
        quality_rhythm = np.mean([r['quality_rhythm'] for r in results_list])
        stability_score = np.mean([r['stability_score'] for r in results_list])

        # 阶段分布取平均
        phase_distribution = {}
        for key in ['ground_contact', 'flight', 'transition']:
            phase_distribution[key] = np.mean([r['phase_distribution'][key] for r in results_list])

        # 使用第一个模型的阶段序列（通常是最好的模型）
        phase_sequence = results_list[0]['phase_sequence']

        return {
            'phase_sequence': phase_sequence,
            'phase_distribution': phase_distribution,
            'quality_score': float(quality_score),
            'quality_stability': float(quality_stability),
            'quality_efficiency': float(quality_efficiency),
            'quality_form': float(quality_form),
            'quality_rhythm': float(quality_rhythm),
            'stability_score': float(stability_score),
            'model_type': 'ensemble',
            'num_models': n
        }


# 为了向后兼容，保留原有接口
def create_analyzer(model_type: str = 'joint', device: str = 'cpu') -> TemporalModelAnalyzer:
    """
    创建分析器的工厂函数

    Args:
        model_type: 模型类型 ('legacy', 'transformer', 'joint', 'ensemble')
        device: 设备

    Returns:
        分析器实例
    """
    if model_type == 'ensemble':
        return TemporalModelEnsemble(device)
    else:
        return TemporalModelAnalyzer(model_type, device)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("=" * 70)
    print("测试时序深度学习分析模块（升级版）")
    print("=" * 70)

    # 创建模拟关键点数据
    print("\n生成模拟关键点数据...")
    mock_keypoints = []
    for i in range(35):
        kp = {
            'detected': True,
            'landmarks': []
        }
        for j in range(33):
            kp['landmarks'].append({
                'x_norm': np.random.rand(),
                'y_norm': np.random.rand(),
                'visibility': 0.9
            })
        mock_keypoints.append(kp)

    # 测试不同模型类型
    for model_type in ['legacy', 'joint']:
        print(f"\n{'='*50}")
        print(f"测试 {model_type} 模型:")
        print('='*50)

        try:
            analyzer = TemporalModelAnalyzer(model_type=model_type)

            # 测试不同视角
            for view in ['side', 'front', 'mixed']:
                results = analyzer.analyze(mock_keypoints, view_angle=view)
                print(f"\n视角: {view}")
                print(f"  质量评分: {results['quality_score']:.2f}")
                print(f"  稳定性: {results['quality_stability']:.2f}")
                print(f"  效率: {results['quality_efficiency']:.2f}")
                print(f"  阶段分布: 触地{results['phase_distribution']['ground_contact']*100:.1f}% | "
                      f"腾空{results['phase_distribution']['flight']*100:.1f}% | "
                      f"过渡{results['phase_distribution']['transition']*100:.1f}%")

        except Exception as e:
            print(f"  ❌ 测试失败: {e}")

    print("\n" + "=" * 70)
    print("✅ 测试完成!")
    print("=" * 70)
