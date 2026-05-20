# modules/quality_evaluator.py
"""
技术质量评价模块（纯运动学规则版）

核心设计：
1. 评分完全基于运动生物力学规则，确保专业性和可解释性
2. 深度学习模型用于特征提取（MediaPipe/MotionBERT），评分使用规则引擎
3. 适配新的归一化振幅指标
4. 支持分阶段膝关节角度评估
5. 针对不同视角的评价策略

评价维度（侧面视角）：
- 稳定性 (30%): 躯干稳定、膝关节角度稳定性、垂直运动稳定性
- 效率 (40%): 步频、垂直振幅、触地时间
- 跑姿 (30%): 膝关节角度（分阶段）、躯干前倾

评价维度（正面视角 - 独立系统，3维度）：
- 下肢力线 (35%): 膝外翻/内翻(50%)、髋部下沉(30%)、整体评级(20%)
- 横向稳定性 (35%): 髋部横摆(40%)、肩部倾斜(30%)、对称性(30%)
- 效率 (30%): 步频(60%)、垂直振幅(40%)
"""
import numpy as np
from typing import Dict, List, Optional
from config.config import (
    BODY_LEAN_THRESHOLDS,
    QUALITY_WEIGHTS,
    QUALITY_THRESHOLDS,
    VERTICAL_AMPLITUDE_THRESHOLDS,
)


class QualityEvaluator:
    """重构版技术质量评价器"""

    def __init__(self):
        """初始化评价器"""
        self.weights = QUALITY_WEIGHTS
        self.thresholds = QUALITY_THRESHOLDS

        # 正面视角专用权重（独立评分系统）
        # 【重构】3维度结构：下肢力线 + 横向稳定性（含对称性）+ 效率（步频+垂直振幅）
        self.frontal_weights = {
            'lower_limb_alignment': 0.35,  # 下肢力线（正面核心指标）
            'lateral_stability': 0.35,     # 横向稳定性（含对称性子指标）
            'efficiency': 0.30,            # 效率（步频60% + 垂直振幅40%）
        }

        # 专业评价标准（基于运动科学研究 - 新标准）
        amp_excellent = float(VERTICAL_AMPLITUDE_THRESHOLDS['excellent_max'])
        amp_good = float(VERTICAL_AMPLITUDE_THRESHOLDS['good_max'])
        amp_fair = float(VERTICAL_AMPLITUDE_THRESHOLDS['fair_max'])
        lean_optimal_min = float(BODY_LEAN_THRESHOLDS['optimal_min'])
        lean_optimal_max = float(BODY_LEAN_THRESHOLDS['optimal_max'])
        lean_good_min = float(BODY_LEAN_THRESHOLDS['good_min'])
        lean_good_max = float(BODY_LEAN_THRESHOLDS['good_max'])
        lean_fair_min = float(BODY_LEAN_THRESHOLDS['fair_min'])
        lean_fair_max = float(BODY_LEAN_THRESHOLDS['fair_max'])
        self.standards = {
            # 垂直振幅标准（相对躯干长度的百分比，4档稳健标准，参考高驰换算）
            'vertical_amplitude': {
                'excellent': (0, amp_excellent),       # ≤11%（优秀）
                'good': (amp_excellent, amp_good),     # 11-17%（良好）
                'fair': (amp_good, amp_fair),          # 17-25%（一般）
                'poor': (amp_fair, 100),               # >25%（待改进）
            },
            # 步频标准（新标准：5个等级）
            'cadence': {
                'elite': (185, 300),     # 185以上 精英
                'excellent': (175, 185), # 175-185 优秀
                'good': (165, 175),      # 165-175 良好
                'fair': (155, 165),      # 155-165 一般
                'poor': (0, 155),        # <155 较差
            },
            # 触地时间标准（新标准：5个等级，毫秒）
            'ground_contact_time': {
                'elite': (0, 210),       # <210ms 精英
                'excellent': (210, 240), # 210-240ms 优秀
                'good': (240, 270),      # 240-270ms 良好
                'fair': (270, 300),      # 270-300ms 一般
                'poor': (300, 1000),     # >300ms 较差
            },
            # 膝关节角度标准（基于马拉松运动员研究数据调整）
            # 参考：SimpliFaster研究 - 初始接触时约20°屈曲（~160°膝角）
            'knee_angle': {
                # 触地期：膝微屈，150-165度（精英跑者典型值）
                'ground_contact': {'ideal': (150, 165), 'acceptable': (140, 170)},
                # 最大弯曲：摆动期，90-130度
                'max_flexion': {'ideal': (90, 130), 'acceptable': (80, 140)},
                # 活动范围：应有足够的运动幅度
                'rom': {'ideal': (40, 70), 'acceptable': (30, 80)},
            },
            # 身体前倾标准（度）- 中长跑/马拉松稳健阈值
            'body_lean': {
                'optimal': (lean_optimal_min, lean_optimal_max),
                'good': (lean_good_min, lean_good_max),
                'acceptable': (lean_fair_min, lean_fair_max),
            },
            # 稳定性标准
            'stability': {
                'excellent': 85,
                'good': 70,
                'fair': 55,
            }
        }

        # 正面视角专用评价标准
        self.frontal_standards = {
            # 膝外翻角度标准（Q角）
            # 正常范围：0-8度，女性可能略高
            'knee_valgus': {
                'excellent': (0, 5),     # 优秀：0-5度
                'good': (5, 10),         # 良好：5-10度
                'fair': (10, 15),        # 一般：10-15度
                'poor': (15, 90),        # 较差：>15度
            },
            # 髋部下沉标准（骨盆倾斜度）
            # 正常跑步时对侧髋部下沉<5度
            'hip_drop': {
                'excellent': (0, 3),     # 优秀：0-3度
                'good': (3, 5),          # 良好：3-5度
                'fair': (5, 8),          # 一般：5-8度
                'poor': (8, 90),         # 较差：>8度
            },
            # 肩部倾斜标准
            'shoulder_tilt': {
                'excellent': (0, 3),     # 优秀：0-3度
                'good': (3, 6),          # 良好：3-6度
                'fair': (6, 10),         # 一般：6-10度
                'poor': (10, 90),        # 较差：>10度
            },
            # 横向稳定性标准（髋部侧向移动幅度，相对髋宽）
            'lateral_sway': {
                'excellent': (0, 3),     # 优秀：<3%髋宽
                'good': (3, 6),          # 良好：3-6%
                'fair': (6, 10),         # 一般：6-10%
                'poor': (10, 100),       # 较差：>10%
            },
        }

    def evaluate(self, kinematic_results: Dict, temporal_results: Dict,
                 view_angle: str = 'side') -> Dict:
        """
        综合评价跑步技术质量

        Args:
            kinematic_results: 运动学分析结果
            temporal_results: 时序模型分析结果
            view_angle: 视频视角

        Returns:
            评价结果
        """
        # 根据视角选择评估策略
        is_frontal = view_angle in ['front', 'back']

        if view_angle == 'side':
            scores = self._evaluate_side_view(kinematic_results, temporal_results)
        elif is_frontal:
            scores = self._evaluate_frontal_view(kinematic_results, temporal_results)
        else:
            scores = self._evaluate_side_view(kinematic_results, temporal_results)

        # 提取各维度得分
        stability_score = scores['stability']
        efficiency_score = scores['efficiency']
        form_score = scores['form']

        # 计算总分
        if is_frontal and 'frontal_total' in scores:
            # 正面视角使用独立评分系统的总分
            total_score = scores['frontal_total']
        else:
            # 侧面视角使用加权平均
            total_score = (
                stability_score * self.weights['stability'] +
                efficiency_score * self.weights['efficiency'] +
                form_score * self.weights['form']
            )

        # 生成评级
        rating = self._get_rating(total_score)

        # 生成详细分析
        detailed_analysis = self._generate_detailed_analysis(
            kinematic_results, scores, view_angle
        )

        # 生成建议
        suggestions = self._generate_suggestions(
            kinematic_results, scores, view_angle
        )

        # 提取数据可靠性信息
        data_reliability = self._extract_data_reliability(kinematic_results)
        weaknesses = self._identify_weaknesses(
            scores, detailed_analysis, is_frontal
        )

        # 构建返回结果
        result = {
            'total_score': round(total_score, 2),
            'rating': rating,
            'dimension_scores': {
                'stability': round(stability_score, 2),
                'efficiency': round(efficiency_score, 2),
                'form': round(form_score, 2),
            },
            'detailed_analysis': detailed_analysis,
            'suggestions': suggestions,
            'strengths': self._identify_strengths(scores, is_frontal),
            'weaknesses': weaknesses,
            'view_angle': view_angle,
            'data_reliability': data_reliability,
        }

        # 正面视角额外输出独立维度得分
        # 【重构】正面视角3维度输出
        if is_frontal and 'frontal_dimensions' in scores:
            result['frontal_dimension_scores'] = {
                'lower_limb_alignment': round(scores['frontal_dimensions']['lower_limb_alignment'], 2),
                'lateral_stability': round(scores['frontal_dimensions']['lateral_stability'], 2),
                'efficiency': round(scores['frontal_dimensions']['efficiency'], 2),
            }

            result['frontal_dimension_names'] = {
                'lower_limb_alignment': '下肢力线',
                'lateral_stability': '横向稳定性',
                'efficiency': '效率',
            }

            # 输出实际使用的权重（便于调试和UI显示）
            result['frontal_dimension_weights'] = {
                'lower_limb_alignment': 0.35,
                'lateral_stability': 0.35,
                'efficiency': 0.30,
            }

        return result

    def _extract_data_reliability(self, kinematic_results: Dict) -> Dict:
        """
        提取数据可靠性信息

        Args:
            kinematic_results: 运动学分析结果

        Returns:
            数据可靠性字典
        """
        # 默认可靠性
        default_reliability = {
            'overall': 'low',
            'is_3d': False,
            'angle_data': {
                'reliability': 'low',
                'description': '使用2D投影计算，数据可能不准确'
            },
            'warnings': ['建议从正侧面拍摄以获得更准确的角度数据']
        }

        # 检查角度数据的可靠性
        if 'angles' in kinematic_results:
            angles_data = kinematic_results['angles']
            if 'data_reliability' in angles_data:
                reliability_info = angles_data['data_reliability']
                is_3d = reliability_info.get('is_3d', False)

                return {
                    'overall': 'high' if is_3d else 'low',
                    'is_3d': is_3d,
                    'angle_data': {
                        'reliability': reliability_info.get('reliability', 'low'),
                        'description': reliability_info.get('description', '')
                    },
                    'warnings': [] if is_3d else [reliability_info.get('recommendation', '')]
                }

        return default_reliability

    def _evaluate_side_view(self, kinematic: Dict, temporal: Dict) -> Dict:
        """侧面视角评估"""
        scores = {}

        # 1. 稳定性评估
        scores['stability'] = self._evaluate_stability(kinematic, temporal)

        # 2. 效率评估（垂直振幅 + 步频）
        scores['efficiency'] = self._evaluate_efficiency_improved(kinematic)

        # 3. 跑姿评估（分阶段膝关节角度 + 前倾）
        scores['form'] = self._evaluate_form_improved(kinematic)

        return scores

    def _evaluate_frontal_view(self, kinematic: Dict, temporal: Dict) -> Dict:
        """
        正面/后方视角独立评估系统

        【重构】正面视角评价维度（3维度）：
        1. 下肢力线 (35%): 膝外翻(50%) + 髋部下沉(30%) + 整体评级(20%)
        2. 横向稳定性 (35%): 髋部横摆(40%) + 肩部倾斜(30%) + 对称性(30%)
        3. 效率 (30%): 步频(60%) + 垂直振幅(40%)
        """
        frontal_scores = {}
        active_weights = {}  # 记录实际使用的权重

        # 1. 下肢力线评估 (35%)
        frontal_scores['lower_limb_alignment'] = self._evaluate_lower_limb_alignment(kinematic)
        active_weights['lower_limb_alignment'] = self.frontal_weights['lower_limb_alignment']

        # 2. 横向稳定性评估 (35%) - 已合并对称性
        frontal_scores['lateral_stability'] = self._evaluate_lateral_stability(kinematic)
        active_weights['lateral_stability'] = self.frontal_weights['lateral_stability']

        # 3. 效率评估 (30%) - 步频 + 垂直振幅
        frontal_scores['efficiency'] = self._evaluate_efficiency(kinematic)
        active_weights['efficiency'] = self.frontal_weights['efficiency']

        # 计算加权总分（固定3维度权重）
        total_weight = sum(active_weights.values())
        total = sum(
            frontal_scores.get(dim, 0) * weight
            for dim, weight in active_weights.items()
        ) / total_weight if total_weight > 0 else 60.0

        # 转换为侧面视角兼容的三维度格式（用于统一显示）
        # 正面视角的维度映射：
        # - stability -> lateral_stability
        # - efficiency -> efficiency（正面视角自己的效率维度）
        # - form -> lower_limb_alignment
        scores = {
            'stability': frontal_scores['lateral_stability'],
            'efficiency': frontal_scores['efficiency'],
            'form': frontal_scores['lower_limb_alignment'],
            # 保留正面视角原始维度
            'frontal_dimensions': frontal_scores,
            'frontal_total': total,
            # 记录实际使用的权重（便于调试）
            'active_weights': active_weights,
        }

        return scores

    def _evaluate_lower_limb_alignment(self, kinematic: Dict) -> float:
        """
        评估下肢力线（正面视角核心指标）

        评估内容：
        1. 膝外翻/内翻角度（Q角）
        2. 髋部下沉（骨盆倾斜）
        3. 着地位置相对髋部的偏移
        """
        scores = []
        weights = []

        if 'lower_limb_alignment' in kinematic:
            alignment = kinematic['lower_limb_alignment']

            # 1. 膝外翻角度评估 (权重50%)
            if 'knee_valgus' in alignment:
                valgus_data = alignment['knee_valgus']
                # 取左右平均
                valgus_left = valgus_data.get('left_mean', 0)
                valgus_right = valgus_data.get('right_mean', 0)
                valgus_avg = (abs(valgus_left) + abs(valgus_right)) / 2

                valgus_score = self._score_by_range(
                    valgus_avg, self.frontal_standards['knee_valgus']
                )
                scores.append(valgus_score)
                weights.append(0.5)

            # 2. 髋部下沉评估 (权重30%)
            if 'hip_drop' in alignment:
                hip_drop = alignment['hip_drop']
                drop_mean = hip_drop.get('mean', hip_drop.get('drop_mean', 0))

                drop_score = self._score_by_range(
                    abs(drop_mean), self.frontal_standards['hip_drop']
                )
                scores.append(drop_score)
                weights.append(0.3)

            # 3. 整体评级 (权重20%)
            if 'overall_rating' in alignment:
                rating = alignment['overall_rating']
                overall_score = rating.get('score', 60)
                scores.append(overall_score)
                weights.append(0.2)

        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return weighted_score

        return 65.0  # 默认分数

    def _evaluate_lateral_stability(self, kinematic: Dict) -> float:
        """
        评估横向稳定性（正面视角优势指标，含对称性）

        【重构】合并原对称性维度为子指标

        评估内容：
        1. 髋部横向摆动幅度 (40%)
        2. 肩部倾斜变化 (30%)
        3. 对称性 (30%): 膝外翻对称性 + 步态对称性
        """
        scores = []
        weights = []

        # 1. 髋部横摆 (40%)
        if 'lateral_stability' in kinematic:
            lateral = kinematic['lateral_stability']

            if 'hip_sway' in lateral:
                # 【优化】直接使用髋宽百分比（kinematic_analyzer已归一化）
                sway = lateral['hip_sway']  # 已经是髋宽的百分比
                sway_score = self._score_by_range(
                    sway, self.frontal_standards['lateral_sway']
                )
                scores.append(sway_score)
                weights.append(0.4)
            elif 'hip_sway_normalized' in lateral:
                # 兼容旧数据（旧格式是小数）
                sway = lateral['hip_sway_normalized'] * 100
                sway_score = self._score_by_range(
                    sway, self.frontal_standards['lateral_sway']
                )
                scores.append(sway_score)
                weights.append(0.4)
            elif 'stability_score' in lateral:
                # 兼容更早版本
                scores.append(lateral['stability_score'])
                weights.append(0.4)

        # 2. 肩部倾斜 (30%)
        if 'shoulder_analysis' in kinematic:
            shoulder = kinematic['shoulder_analysis']
            if 'tilt_mean' in shoulder:
                tilt = abs(shoulder['tilt_mean'])
                tilt_score = self._score_by_range(
                    tilt, self.frontal_standards['shoulder_tilt']
                )
                scores.append(tilt_score)
                weights.append(0.3)
            elif 'rating' in shoulder:
                # 兼容旧数据
                shoulder_score = shoulder['rating'].get('score', 60)
                scores.append(shoulder_score)
                weights.append(0.3)

        # 3. 对称性子评估 (30%)
        sym_score = self._evaluate_symmetry_sub(kinematic)
        if sym_score is not None:
            scores.append(sym_score)
            weights.append(0.3)

        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return weighted_score

        return 65.0  # 默认分数

    def _evaluate_symmetry(self, kinematic: Dict) -> float:
        """
        评估对称性（正面视角独特指标）

        评估内容：
        1. 左右腿动作对称性（膝外翻差异）
        2. 步态对称性（脚踝/膝关节运动相关性）

        数据源优先级：
        1. symmetry_analysis（新增独立字段，最可靠）
        2. lateral_stability['symmetry'] / stability['symmetry']（兼容旧结构）
        """
        scores = []
        weights = []

        # 1. 下肢力线对称性（膝外翻差异）
        if 'lower_limb_alignment' in kinematic:
            alignment = kinematic['lower_limb_alignment']

            # 左右膝外翻角度差异
            if 'knee_valgus' in alignment:
                valgus = alignment['knee_valgus']
                left = abs(valgus.get('left_mean', 0))
                right = abs(valgus.get('right_mean', 0))
                if left > 0 or right > 0:
                    diff = abs(left - right)
                    # 【校准】改用max分母，避免小角度时百分比过大
                    max_val = max(left, right) if max(left, right) > 0 else 1
                    asymmetry = (diff / max_val) * 100 if max_val > 0 else 0
                    # 【校准】放宽阈值约50%，适应2D检测精度
                    if asymmetry < 30:      # 从20改为30
                        sym_score = 95
                    elif asymmetry < 50:    # 从40改为50
                        sym_score = 80
                    elif asymmetry < 70:    # 从60改为70
                        sym_score = 65
                    else:
                        sym_score = 50
                    scores.append(sym_score)
                    weights.append(0.4)

        # 2. 【优先】从独立的 symmetry_analysis 字段读取步态对称性
        if 'symmetry_analysis' in kinematic:
            sym_data = kinematic['symmetry_analysis']
            if 'overall_score' in sym_data:
                scores.append(sym_data['overall_score'])
                weights.append(0.4)  # 步态对称性权重提高到0.4
        else:
            # 2b. 兼容：从 lateral_stability['symmetry'] 读取
            if 'lateral_stability' in kinematic:
                lateral = kinematic['lateral_stability']
                if 'symmetry' in lateral:
                    scores.append(lateral['symmetry'])
                    weights.append(0.3)

            # 2c. 兼容：从 stability['symmetry'] 读取
            if 'stability' in kinematic:
                stability = kinematic['stability']
                if isinstance(stability, dict) and 'symmetry' in stability:
                    scores.append(stability['symmetry'])
                    weights.append(0.3)

        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return weighted_score

        return 70.0  # 默认分数（对称性通常较好）

    def _evaluate_efficiency(self, kinematic: Dict) -> float:
        """
        评估跑步效率（正面视角新维度）

        子指标：
        1. 步频 (60%): 基于FFT检测的步频评估
        2. 垂直振幅 (40%): 重心垂直运动幅度

        Returns:
            效率得分 (0-100)
        """
        scores = []
        weights = []

        # 1. 步频评估 (60%)
        if 'cadence' in kinematic:
            cadence_data = kinematic['cadence']
            cadence = cadence_data.get('cadence', 0)
            if cadence > 0:
                cadence_score = self._score_cadence(cadence)
                scores.append(cadence_score)
                weights.append(0.6)

        # 2. 垂直振幅评估 (40%)
        if 'vertical_motion' in kinematic:
            vert = kinematic['vertical_motion']
            # 优先使用 normalized_amplitude，其次使用 amplitude_normalized
            amplitude = vert.get('normalized_amplitude',
                        vert.get('amplitude_normalized', 0))
            if amplitude > 0:
                vert_score = self._score_vertical_amplitude(amplitude)
                scores.append(vert_score)
                weights.append(0.4)

        if scores and weights:
            total_weight = sum(weights)
            return sum(s * w for s, w in zip(scores, weights)) / total_weight

        return 70.0  # 默认分数

    def _evaluate_symmetry_sub(self, kinematic: Dict) -> Optional[float]:
        """
        对称性子评估（用于横向稳定性维度的子指标）

        评估内容：
        1. 膝外翻对称性（左右差异）
        2. 步态对称性分析

        Returns:
            对称性得分 或 None（无数据时）
        """
        scores = []

        # 1. 膝外翻对称性
        if 'lower_limb_alignment' in kinematic:
            alignment = kinematic['lower_limb_alignment']
            if 'knee_valgus' in alignment:
                valgus = alignment['knee_valgus']
                left = abs(valgus.get('left_mean', 0))
                right = abs(valgus.get('right_mean', 0))
                if left > 0 or right > 0:
                    diff = abs(left - right)
                    # 使用较大值作为分母（更公平的对比）
                    max_val = max(left, right) if max(left, right) > 0 else 1
                    asymmetry = (diff / max_val) * 100
                    # 差异越小越好
                    if asymmetry < 20:
                        scores.append(95)
                    elif asymmetry < 40:
                        scores.append(80)
                    elif asymmetry < 60:
                        scores.append(65)
                    else:
                        scores.append(50)

        # 2. 步态对称性分析
        if 'symmetry_analysis' in kinematic:
            sym_data = kinematic['symmetry_analysis']
            if 'overall_score' in sym_data:
                scores.append(sym_data['overall_score'])

        return np.mean(scores) if scores else None

    def _assess_vertical_amplitude_detail(self, amplitude_normalized: float,
                                          is_estimate: bool = False) -> Dict:
        """统一垂直振幅详情分析的等级和文案。"""
        if amplitude_normalized <= 0:
            return {}

        excellent_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['excellent_max'])
        good_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['good_max'])
        fair_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['fair_max'])

        if amplitude_normalized <= excellent_max:
            assessment = '垂直振幅控制优秀，能量利用效率高'
            level = 'excellent'
        elif amplitude_normalized <= good_max:
            assessment = '垂直振幅良好，整体在可接受范围'
            level = 'good'
        elif amplitude_normalized <= fair_max:
            assessment = '垂直振幅偏大，存在能量损耗，有优化空间'
            level = 'fair'
        else:
            assessment = '垂直振幅过大，可能导致明显能量浪费'
            level = 'poor'

        if is_estimate:
            assessment += '（基于髋部Y坐标估算，仅供参考）'

        return {
            'value': f'{amplitude_normalized:.1f}%',
            'assessment': assessment,
            'level': level,
            'is_estimate': is_estimate,
        }

    def _score_by_range(self, value: float, standards: Dict) -> float:
        """根据范围标准评分"""
        if standards['excellent'][0] <= value <= standards['excellent'][1]:
            return 95
        elif standards['good'][0] < value <= standards['good'][1]:
            return 80
        elif standards['fair'][0] < value <= standards['fair'][1]:
            return 65
        else:
            return 45

    def _evaluate_stability(self, kinematic: Dict, temporal: Dict) -> float:
        """
        评估动作稳定性（纯运动学规则）

        评估依据：
        1. 躯干稳定性（来自运动学分析）
        2. 膝关节角度稳定性（标准差越小越稳定）
        3. 髋部稳定性（垂直振幅的稳定性）

        注：temporal参数保留用于兼容性，但不用于评分计算
        """
        scores = []
        weights = []

        # 1. 躯干整体稳定性（权重40%）
        if 'stability' in kinematic:
            stability_data = kinematic['stability']
            if isinstance(stability_data, dict) and 'overall' in stability_data:
                scores.append(stability_data['overall'])
                weights.append(0.4)

        # 2. 膝关节角度稳定性（权重35%）
        # 优先使用触地期标准差（更能反映实际稳定性），否则回退到全序列标准差
        if 'angles' in kinematic:
            avg_knee_std = 0

            # 优先使用触地期标准差
            if 'phase_analysis' in kinematic['angles']:
                gc = kinematic['angles']['phase_analysis'].get('ground_contact', {})
                gc_std = gc.get('std', 0)
                if gc_std > 0:
                    avg_knee_std = gc_std

            # 如果触地期数据不可用，回退到全序列标准差
            if avg_knee_std == 0:
                knee_std_left = kinematic['angles'].get('knee_left_std', 0)
                knee_std_right = kinematic['angles'].get('knee_right_std', 0)
                avg_knee_std = (knee_std_left + knee_std_right) / 2

            if avg_knee_std > 0:
                # 标准差越小越稳定
                # 触地期std阈值调整：≤5°优秀, ≤8°良好, ≤12°一般
                if avg_knee_std <= 5:
                    knee_stability = 95
                elif avg_knee_std <= 8:
                    knee_stability = 80
                elif avg_knee_std <= 12:
                    knee_stability = 65
                else:
                    knee_stability = max(40, 100 - avg_knee_std * 2)
                scores.append(knee_stability)
                weights.append(0.35)

        # 3. 垂直运动稳定性（权重25%）
        if 'vertical_motion' in kinematic:
            vm = kinematic['vertical_motion']
            pos_std = vm.get('std_position', 0)
            if pos_std >= 0:
                # 位置标准差越小越稳定（归一化坐标，典型值 0.005-0.03）
                if pos_std < 0.008:
                    vertical_stability = 95
                elif pos_std < 0.015:
                    vertical_stability = 80
                elif pos_std < 0.025:
                    vertical_stability = 65
                else:
                    vertical_stability = 50
                scores.append(vertical_stability)
                weights.append(0.25)

        # 加权平均
        if scores and weights:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return weighted_score

        return 60.0  # 默认中等稳定性

    def _evaluate_efficiency_improved(self, kinematic: Dict) -> float:
        """改进的效率评估"""
        scores = []

        # 1. 垂直振幅评估
        # 优先使用 kinematic_analyzer 预计算的评分（与调试显示一致的单一数据源）
        if 'vertical_motion' in kinematic:
            vm = kinematic['vertical_motion']

            if 'amplitude_rating' in vm and vm['amplitude_rating'].get('score', 0) > 0:
                amp_score = vm['amplitude_rating']['score']
            elif 'amplitude_normalized' in vm:
                amp_norm = vm['amplitude_normalized']
                amp_score = self._score_vertical_amplitude(amp_norm)
            else:
                amp = vm.get('amplitude', 0)
                amp_norm = amp * 100 / 0.25
                amp_score = self._score_vertical_amplitude(amp_norm)

            scores.append(amp_score)

        # 2. 步频评估
        # 优先使用 kinematic_analyzer 预计算的评分
        if 'cadence' in kinematic:
            cadence_data = kinematic['cadence']
            cadence_rating = cadence_data.get('rating', {})
            if cadence_rating.get('score', 0) > 0:
                cadence_score = cadence_rating['score']
            else:
                cadence = cadence_data.get('cadence', 0)
                cadence_score = self._score_cadence(cadence)
            scores.append(cadence_score)

        # 3. 触地时间评估（等权参与平均）
        if 'gait_cycle' in kinematic:
            gait = kinematic['gait_cycle']
            if 'gait_rating' in gait:
                gait_score = gait['gait_rating'].get('score', 70)
                scores.append(gait_score)

        return np.mean(scores) if scores else 60.0

    def _score_vertical_amplitude(self, amplitude_normalized: float) -> float:
        """
        评分垂直振幅
        基于归一化振幅（相对躯干长度的百分比）
        """
        stds = self.standards['vertical_amplitude']

        if stds['excellent'][0] <= amplitude_normalized <= stds['excellent'][1]:
            return 100  # ≤11%
        elif stds['good'][0] < amplitude_normalized <= stds['good'][1]:
            return 80   # 11-17%
        elif stds['fair'][0] < amplitude_normalized <= stds['fair'][1]:
            return 60   # 17-25%
        else:
            return 40   # >25% 振幅过大

    def _score_cadence(self, cadence: float) -> float:
        """评分步频（新标准：5个等级）"""
        if cadence >= 185:
            return 100  # 精英
        elif cadence >= 175:
            return 90   # 优秀
        elif cadence >= 165:
            return 75   # 良好
        elif cadence >= 155:
            return 60   # 一般
        else:
            return 45   # 较差

    def _evaluate_form_improved(self, kinematic: Dict) -> float:
        """改进的跑姿评估"""
        scores = []

        # 1. 分阶段膝关节角度评估
        if 'angles' in kinematic and 'phase_analysis' in kinematic['angles']:
            phase = kinematic['angles']['phase_analysis']

            # 触地期角度
            gc_angle = phase['ground_contact'].get('mean', 0)
            if gc_angle > 0:
                gc_score = self._score_knee_angle(gc_angle, 'ground_contact')
                scores.append(gc_score)

            # 最大弯曲角度 - 添加有效性检查
            max_flex = phase.get('max_flexion', 0)
            if max_flex >= 80:  # 只有在合理范围内才评分（正常跑步应为80-140°）
                flex_score = self._score_knee_angle(max_flex, 'max_flexion')
                scores.append(flex_score)

            # 关节活动范围 - 添加有效性检查
            rom = phase.get('range_of_motion', 0)
            if 20 <= rom <= 100:  # 只有在合理范围内才评分（正常应为30-80°）
                rom_score = self._score_knee_angle(rom, 'rom')
                scores.append(rom_score)

        # 兼容旧版：使用平均膝关节角度
        elif 'angles' in kinematic:
            knee_mean_left = kinematic['angles'].get('knee_left_mean', 0)
            knee_mean_right = kinematic['angles'].get('knee_right_mean', 0)

            for knee_angle in [knee_mean_left, knee_mean_right]:
                if knee_angle > 0:
                    # 使用宽松的判断标准
                    if 130 <= knee_angle <= 170:
                        scores.append(80)
                    elif 120 <= knee_angle < 130 or 170 < knee_angle <= 175:
                        scores.append(70)
                    else:
                        scores.append(55)

        # 2. 身体前倾评估
        if 'body_lean' in kinematic:
            lean = kinematic['body_lean']
            if 'rating' in lean:
                lean_score = lean['rating'].get('score', 70)
                scores.append(lean_score)

        return np.mean(scores) if scores else 65.0

    def _score_knee_angle(self, angle: float, phase_type: str) -> float:
        """评分膝关节角度"""
        stds = self.standards['knee_angle'][phase_type]

        ideal = stds['ideal']
        acceptable = stds['acceptable']

        if ideal[0] <= angle <= ideal[1]:
            return 100
        elif acceptable[0] <= angle <= acceptable[1]:
            return 75
        else:
            return 50

    def _get_rating(self, score: float) -> str:
        """根据分数获取评级"""
        if score >= self.thresholds['excellent']:
            return '优秀'
        elif score >= self.thresholds['good']:
            return '良好'
        elif score >= self.thresholds['fair']:
            return '一般'
        else:
            return '待改进'

    def _generate_detailed_analysis(self, kinematic: Dict, scores: Dict,
                                     view_angle: str) -> Dict:
        """生成详细分析"""
        analysis = {}
        is_frontal = view_angle in ['front', 'back']

        if is_frontal:
            # 正面视角分析
            analysis = self._generate_frontal_detailed_analysis(kinematic, scores)
        else:
            # 侧面视角分析
            analysis = self._generate_side_detailed_analysis(kinematic, scores)

        return analysis

    def _generate_side_detailed_analysis(self, kinematic: Dict, scores: Dict) -> Dict:
        """生成侧面视角详细分析"""
        analysis = {}

        # 垂直振幅分析
        if 'vertical_motion' in kinematic:
            vm = kinematic['vertical_motion']
            amp_norm = vm.get('amplitude_normalized', 0)

            if amp_norm > 0:
                analysis['vertical_amplitude'] = self._assess_vertical_amplitude_detail(amp_norm)

        # 膝关节角度分析
        if 'angles' in kinematic and 'phase_analysis' in kinematic['angles']:
            phase = kinematic['angles']['phase_analysis']
            gc_mean = phase['ground_contact'].get('mean', 0)
            max_flex = phase.get('max_flexion', 0)

            analysis['knee_angles'] = {
                'ground_contact': f'{gc_mean:.1f}°' if gc_mean > 0 else 'N/A',
                'max_flexion': f'{max_flex:.1f}°' if max_flex > 0 else 'N/A',
                'assessment': self._assess_knee_angles(gc_mean, max_flex),
                'level': self._assess_knee_angles_level(gc_mean, max_flex),
            }

        # 步频分析
        if 'cadence' in kinematic:
            cadence = kinematic['cadence'].get('cadence', 0)
            analysis['cadence'] = {
                'value': f'{cadence:.0f} 步/分',
                'assessment': self._assess_cadence(cadence),
                'level': self._assess_cadence_level(cadence),
            }

        if 'body_lean' in kinematic:
            body_lean = kinematic['body_lean']
            lean_val = body_lean.get('forward_lean', 0)
            rating = body_lean.get('rating', {}) if isinstance(body_lean, dict) else {}
            if lean_val > 0:
                analysis['body_lean'] = {
                    'value': f'{lean_val:.1f}°',
                    'assessment': rating.get('description', '前倾角度有优化空间'),
                    'level': rating.get('level', 'acceptable'),
                }

        return analysis

    def _generate_frontal_detailed_analysis(self, kinematic: Dict, scores: Dict) -> Dict:
        """生成正面视角详细分析"""
        analysis = {}

        # 1. 下肢力线分析（正面核心指标）
        if 'lower_limb_alignment' in kinematic:
            alignment = kinematic['lower_limb_alignment']

            # 膝外翻分析
            if 'knee_valgus' in alignment:
                valgus = alignment['knee_valgus']
                left_val = valgus.get('left_mean', 0)
                right_val = valgus.get('right_mean', 0)
                avg_val = (abs(left_val) + abs(right_val)) / 2

                if avg_val <= 5:
                    assessment = '膝关节力线良好，跑姿稳定'
                    level = 'excellent'
                elif avg_val <= 10:
                    assessment = '膝关节有轻微外翻，建议加强臀中肌训练'
                    level = 'good'
                elif avg_val <= 15:
                    assessment = '膝关节外翻明显，需加强髋外展肌群力量'
                    level = 'fair'
                else:
                    assessment = '膝关节外翻严重，有受伤风险，建议咨询专业人士'
                    level = 'needs_improvement'

                analysis['knee_valgus'] = {
                    'value': f'左{left_val:.1f}° / 右{right_val:.1f}°',
                    'assessment': assessment,
                    'level': level
                }

            # 髋部下沉分析
            if 'hip_drop' in alignment:
                hip_drop = alignment['hip_drop']
                drop_mean = hip_drop.get('mean', hip_drop.get('drop_mean', 0))

                if abs(drop_mean) <= 3:
                    assessment = '骨盆稳定性好，核心控制力强'
                    level = 'excellent'
                elif abs(drop_mean) <= 5:
                    assessment = '骨盆稳定性一般，可加强核心训练'
                    level = 'good'
                elif abs(drop_mean) <= 8:
                    assessment = '骨盆下沉明显，需加强臀中肌和核心力量'
                    level = 'fair'
                else:
                    assessment = '骨盆下沉严重，建议针对性康复训练'
                    level = 'needs_improvement'

                analysis['hip_drop'] = {
                    'value': f'{drop_mean:.1f}°',
                    'assessment': assessment,
                    'level': level
                }

        # 2. 横向稳定性分析
        if 'lateral_stability' in kinematic:
            lateral = kinematic['lateral_stability']
            stability_score = lateral.get('stability_score', 0)

            if stability_score >= 85:
                assessment = '横向稳定性优秀，核心控制力强'
                level = 'excellent'
            elif stability_score >= 70:
                assessment = '横向稳定性良好，有优化空间'
                level = 'good'
            elif stability_score >= 55:
                assessment = '横向稳定性一般，建议加强核心训练'
                level = 'fair'
            else:
                assessment = '横向稳定性较差，需重点改善'
                level = 'needs_improvement'

            analysis['lateral_stability'] = {
                'value': f'{stability_score:.0f}分',
                'assessment': assessment,
                'level': level
            }

        # 3. 肩部倾斜分析
        if 'shoulder_analysis' in kinematic:
            shoulder = kinematic['shoulder_analysis']
            tilt_mean = shoulder.get('tilt_mean', 0)

            if abs(tilt_mean) <= 3:
                assessment = '肩部保持水平，上身稳定'
                level = 'excellent'
            elif abs(tilt_mean) <= 6:
                assessment = '肩部轻微摆动，属正常范围'
                level = 'good'
            elif abs(tilt_mean) <= 10:
                assessment = '肩部摆动较大，可能影响效率'
                level = 'fair'
            else:
                assessment = '肩部摆动过大，建议调整上身姿势'
                level = 'needs_improvement'

            analysis['shoulder_tilt'] = {
                'value': f'{tilt_mean:.1f}°',
                'assessment': assessment,
                'level': level
            }

        # 4. 辅助指标（标注为估算值）
        # 步频（正面视角估算）
        if 'cadence' in kinematic:
            cadence_data = kinematic['cadence']
            cadence = cadence_data.get('cadence', 0)
            confidence = cadence_data.get('confidence', 0)

            if cadence > 0:
                assessment = self._assess_cadence(cadence)
                # 标注可信度
                if confidence < 0.7:
                    assessment += '（正面视角估算值，仅供参考）'

                analysis['cadence'] = {
                    'value': f'{cadence:.0f} 步/分',
                    'assessment': assessment,
                    'level': self._assess_cadence_level(cadence),
                    'confidence': f'{confidence*100:.0f}%' if confidence > 0 else '低',
                    'is_estimate': True
                }

        # 垂直振幅（正面视角估算）
        if 'vertical_motion' in kinematic:
            vm = kinematic['vertical_motion']
            amp_norm = vm.get('amplitude_normalized', 0)
            is_estimate = vm.get('is_estimate', False)

            if amp_norm > 0:
                analysis['vertical_amplitude'] = self._assess_vertical_amplitude_detail(
                    amp_norm, is_estimate=is_estimate
                )

        # 5. 不可用指标说明
        if 'gait_cycle' in kinematic:
            gait = kinematic['gait_cycle']
            if gait.get('gait_rating', {}).get('level') == 'not_available':
                analysis['ground_contact_time'] = {
                    'value': '不可用',
                    'assessment': '正面视角无法准确检测触地时间，建议使用侧面视角拍摄',
                    'level': 'not_available',
                    'is_not_available': True
                }

        return analysis

    def _assess_knee_angles(self, gc_angle: float, max_flex: float) -> str:
        """
        评估膝关节角度（基于马拉松运动员研究数据）

        参考标准：
        - 触地期理想范围：150-165°（约15-30°屈曲）
        - 摆动期理想范围：90-130°
        """
        assessments = []

        if gc_angle > 0:
            if 150 <= gc_angle <= 165:
                assessments.append('触地时膝关节角度理想，缓冲良好')
            elif 140 <= gc_angle < 150:
                assessments.append('触地时膝关节缓冲充分')
            elif 165 < gc_angle <= 170:
                assessments.append('触地时膝关节略直，可适当增加缓冲')
            elif gc_angle > 170:
                assessments.append('触地时膝关节伸展过直，冲击力较大')
            else:
                assessments.append('触地时膝关节弯曲过多')

        if max_flex > 0:
            if 90 <= max_flex <= 130:
                assessments.append('摆动期膝关节弯曲充分')
            elif max_flex < 90:
                assessments.append('摆动期膝关节弯曲不足')
            else:
                assessments.append('摆动期膝关节弯曲偏少')

        return '；'.join(assessments) if assessments else '数据不足'

    def _assess_cadence(self, cadence: float) -> str:
        """评估步频"""
        if 180 <= cadence <= 200:
            return '步频处于最佳范围，跑步经济性好'
        elif 170 <= cadence < 180:
            return '步频略低于最佳范围，可适当提高'
        elif 200 < cadence <= 210:
            return '步频略高，注意避免过度紧张'
        elif cadence < 160:
            return '步频偏低，可能步幅过大，建议调整'
        elif cadence > 220:
            return '步频过高，可能影响跑步效率'
        else:
            return '步频有优化空间'

    def _assess_knee_angles_level(self, gc_angle: float, max_flex: float) -> str:
        """膝角总体等级，用于提取待改进项。"""
        score = 0
        count = 0
        if gc_angle > 0:
            count += 1
            if 150 <= gc_angle <= 165:
                score += 3
            elif 140 <= gc_angle < 150 or 165 < gc_angle <= 170:
                score += 2
            else:
                score += 1
        if max_flex > 0:
            count += 1
            if 90 <= max_flex <= 130:
                score += 3
            elif 80 <= max_flex < 90 or 130 < max_flex <= 140:
                score += 2
            else:
                score += 1
        if count == 0:
            return 'not_available'
        avg = score / count
        if avg >= 2.8:
            return 'excellent'
        if avg >= 2.0:
            return 'good'
        if avg >= 1.4:
            return 'fair'
        return 'poor'

    def _assess_cadence_level(self, cadence: float) -> str:
        """步频评级，用于提取待改进指标。"""
        if 180 <= cadence <= 200:
            return 'excellent'
        if 170 <= cadence < 180 or 200 < cadence <= 210:
            return 'good'
        if 160 <= cadence < 170 or 210 < cadence <= 220:
            return 'fair'
        return 'poor'

    def _generate_suggestions(self, kinematic: Dict, scores: Dict,
                               view_angle: str) -> List[str]:
        """生成改进建议"""
        is_frontal = view_angle in ['front', 'back']

        if is_frontal:
            return self._generate_frontal_suggestions(kinematic, scores)
        else:
            return self._generate_side_suggestions(kinematic, scores)

    def _generate_side_suggestions(self, kinematic: Dict, scores: Dict) -> List[str]:
        """生成侧面视角改进建议"""
        suggestions = []

        # 基于各维度得分生成建议
        if scores['stability'] < 70:
            suggestions.append("加强核心力量训练，提高躯干稳定性")

        if scores['efficiency'] < 70:
            # 具体分析效率问题来源
            if 'vertical_motion' in kinematic:
                amp_norm = kinematic['vertical_motion'].get('amplitude_normalized', 0)
                if amp_norm > 10:
                    suggestions.append("减少垂直振幅，想象'贴地飞行'的感觉")
                elif amp_norm < 3:
                    suggestions.append("适当增加步幅，避免过于保守的跑姿")

            if 'cadence' in kinematic:
                cadence = kinematic['cadence'].get('cadence', 0)
                if cadence < 170:
                    suggestions.append("尝试提高步频至170-180步/分，可借助节拍器练习")

        if scores['form'] < 70:
            if 'angles' in kinematic and 'phase_analysis' in kinematic['angles']:
                phase = kinematic['angles']['phase_analysis']
                gc_angle = phase['ground_contact'].get('mean', 0)
                if gc_angle > 170:
                    suggestions.append("落地时保持膝关节微屈（约15-25°），减少冲击力")
                elif gc_angle < 140:
                    suggestions.append("落地时不要过度屈膝，保持自然伸展姿态")

            if 'body_lean' in kinematic:
                lean = kinematic['body_lean'].get('forward_lean', 0)
                if lean < self.standards['body_lean']['acceptable'][0]:
                    suggestions.append("躯干偏直立，支撑期可适度从踝部前移而不是坐在髋后")
                elif lean < self.standards['body_lean']['optimal'][0]:
                    suggestions.append("前倾略小，可轻微增加身体整体前移以改善推进感")
                elif lean > self.standards['body_lean']['acceptable'][1]:
                    suggestions.append("避免过度前倾，保持躯干自然稳定并减少额外负担")

        # 如果没有明显问题
        if not suggestions:
            suggestions.append("继续保持良好状态，可适当增加训练量挑战自己")

        return suggestions

    def _generate_frontal_suggestions(self, kinematic: Dict, scores: Dict) -> List[str]:
        """生成正面视角改进建议"""
        suggestions = []
        frontal_dims = scores.get('frontal_dimensions', {})

        # 1. 下肢力线问题建议
        if frontal_dims.get('lower_limb_alignment', 100) < 70:
            if 'lower_limb_alignment' in kinematic:
                alignment = kinematic['lower_limb_alignment']

                # 膝外翻建议
                if 'knee_valgus' in alignment:
                    valgus = alignment['knee_valgus']
                    avg_val = (abs(valgus.get('left_mean', 0)) + abs(valgus.get('right_mean', 0))) / 2
                    if avg_val > 10:
                        suggestions.append("加强臀中肌训练（蚌式开合、侧向弹力带行走），改善膝关节力线")
                        suggestions.append("练习单腿站立和下蹲，提高膝关节稳定性")

                # 髋部下沉建议
                if 'hip_drop' in alignment:
                    drop = abs(alignment['hip_drop'].get('mean', alignment['hip_drop'].get('drop_mean', 0)))
                    if drop > 5:
                        suggestions.append("强化核心和臀部肌群，推荐平板支撑变式和单腿臀桥")
                        suggestions.append("跑步时注意保持骨盆水平，可通过腰间系绳提示")

        # 2. 横向稳定性问题建议
        if frontal_dims.get('lateral_stability', 100) < 70:
            suggestions.append("增加核心稳定性训练，如死虫式、鸟狗式")
            suggestions.append("练习单腿平衡和跳跃落地，提高动态稳定性")

            # 肩部摆动建议
            if 'shoulder_analysis' in kinematic:
                tilt = abs(kinematic['shoulder_analysis'].get('tilt_mean', 0))
                if tilt > 8:
                    suggestions.append("跑步时保持肩部放松下沉，双臂自然前后摆动而非左右")

        # 3. 对称性问题建议
        if frontal_dims.get('symmetry', 100) < 70:
            suggestions.append("检查是否存在左右腿力量不平衡，可进行单侧力量测试")
            suggestions.append("加入单侧训练（单腿硬拉、保加利亚深蹲），纠正不对称")

        # 4. 正面视角局限性提示
        suggestions.append("💡 建议：正面视角适合分析下肢力线和稳定性，如需分析步频、触地时间等指标，请使用侧面视角拍摄")

        # 如果没有明显问题
        if len(suggestions) == 1:  # 只有局限性提示
            suggestions.insert(0, "正面视角分析显示跑姿良好！继续保持")

        return suggestions

    def _identify_strengths(self, scores: Dict, is_frontal: bool = False) -> List[str]:
        """识别优势项"""
        strengths = []

        if is_frontal and 'frontal_dimensions' in scores:
            # 正面视角使用独立维度名称
            frontal_names = {
                'lower_limb_alignment': '下肢力线',
                'lateral_stability': '横向稳定性',
                'symmetry': '对称性'
            }
            frontal_dims = scores['frontal_dimensions']
            for key, name in frontal_names.items():
                if frontal_dims.get(key, 0) >= 80:
                    strengths.append(name)
        else:
            # 侧面视角
            score_names = {
                'stability': '动作稳定性',
                'efficiency': '跑步效率',
                'form': '跑姿标准度'
            }
            for key, name in score_names.items():
                if scores.get(key, 0) >= 80:
                    strengths.append(name)

        return strengths if strengths else ['暂无突出优势']

    def _identify_weaknesses(self, scores: Dict, detailed_analysis: Dict,
                             is_frontal: bool = False) -> List[str]:
        """识别薄弱项"""
        metric_name_map = {
            'vertical_amplitude': '垂直振幅',
            'knee_angles': '膝关节角度',
            'cadence': '步频',
            'body_lean': '躯干前倾',
            'knee_valgus': '膝关节力线',
            'hip_drop': '髋部下沉',
            'lateral_stability': '横向稳定性',
            'shoulder_tilt': '肩部倾斜',
            'ground_contact_time': '触地时间',
        }
        weak_levels = {'fair', 'poor', 'acceptable', 'upright', 'excessive', 'needs_improvement'}
        weaknesses = []

        for key, item in (detailed_analysis or {}).items():
            if not isinstance(item, dict):
                continue
            level = str(item.get('level', '')).lower()
            if level in weak_levels:
                mapped = metric_name_map.get(key)
                if mapped and mapped not in weaknesses:
                    weaknesses.append(mapped)

        if weaknesses:
            return weaknesses

        if is_frontal and 'frontal_dimensions' in scores:
            # 正面视角使用独立维度名称
            frontal_names = {
                'lower_limb_alignment': '下肢力线',
                'lateral_stability': '横向稳定性',
                'symmetry': '对称性'
            }
            frontal_dims = scores['frontal_dimensions']
            for key, name in frontal_names.items():
                if frontal_dims.get(key, 0) < 65:
                    weaknesses.append(name)
        else:
            # 侧面视角
            score_names = {
                'stability': '动作稳定性',
                'efficiency': '跑步效率',
                'form': '跑姿标准度'
            }
            for key, name in score_names.items():
                if scores.get(key, 0) < 65:
                    weaknesses.append(name)

        return weaknesses if weaknesses else ['无明显薄弱项']


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试重构版质量评估模块")
    print("=" * 60)

    # 模拟运动学分析结果
    mock_kinematic = {
        'vertical_motion': {
            'amplitude': 0.02,
            'amplitude_normalized': 7.5,
            'amplitude_rating': {'level': 'good', 'score': 80}
        },
        'cadence': {
            'cadence': 175,
            'confidence': 0.85
        },
        'angles': {
            'knee_left_mean': 155,
            'knee_right_mean': 158,
            'knee_left_std': 8,
            'knee_right_std': 7,
            'phase_analysis': {
                'ground_contact': {'mean': 165, 'std': 5},
                'flight': {'mean': 110, 'std': 8},
                'max_flexion': 105,
                'range_of_motion': 55
            }
        },
        'stability': {
            'overall': 78,
            'trunk': 82,
            'head': 75,
            'symmetry': 80
        },
        'body_lean': {
            'forward_lean': 10,
            'rating': {'score': 100}
        },
        'gait_cycle': {
            'phase_distribution': {
                'ground_contact': 0.45,
                'flight': 0.35,
                'transition': 0.20
            },
            'gait_rating': {'score': 80}
        }
    }

    mock_temporal = {
        'quality_score': 75,
        'stability_score': 72
    }

    evaluator = QualityEvaluator()
    results = evaluator.evaluate(mock_kinematic, mock_temporal, view_angle='side')

    print(f"\n总分: {results['total_score']}")
    print(f"评级: {results['rating']}")
    print(f"\n各维度得分:")
    for dim, score in results['dimension_scores'].items():
        print(f"  {dim}: {score}")

    print(f"\n优势: {results['strengths']}")
    print(f"薄弱项: {results['weaknesses']}")

    print(f"\n改进建议:")
    for i, sug in enumerate(results['suggestions'], 1):
        print(f"  {i}. {sug}")

    print("\n✅ 模块测试完成!")
