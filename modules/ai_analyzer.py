# modules/ai_analyzer.py
"""
AI分析模块 - 精简版
仅使用智谱AI (zai库 + glm-4.5模型)
"""
import os
import base64
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
from pathlib import Path
from config.config import AI_CONFIG, BODY_LEAN_THRESHOLDS, VERTICAL_AMPLITUDE_THRESHOLDS


def _normalize_report_output(text: str, title: str = '跑步动作分析报告') -> str:
    cleaned = str(text or '').replace('\r\n', '\n').strip()
    if not cleaned:
        return ''
    lines = [line.rstrip() for line in cleaned.split('\n')]
    compact = []
    blank_count = 0
    for line in lines:
        if line.strip():
            compact.append(line)
            blank_count = 0
        else:
            blank_count += 1
            if blank_count <= 1:
                compact.append('')
    cleaned = '\n'.join(compact).strip()
    if not cleaned.startswith('#'):
        cleaned = f"# {title}\n\n## 核心结论\n{cleaned}"
    if '## 说明' not in cleaned:
        cleaned += '\n\n## 说明\n- 本报告用于动作技术解读与训练提示。\n- 结果应结合拍摄质量、视角条件与实际训练感受综合判断。'
    return cleaned


def _vertical_amplitude_reference_text() -> str:
    excellent_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['excellent_max'])
    good_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['good_max'])
    fair_max = float(VERTICAL_AMPLITUDE_THRESHOLDS['fair_max'])
    return f"≤{excellent_max:.0f}%优秀, ≤{good_max:.0f}%良好, ≤{fair_max:.0f}%一般"


def _body_lean_reference_text() -> str:
    optimal_min = float(BODY_LEAN_THRESHOLDS['optimal_min'])
    optimal_max = float(BODY_LEAN_THRESHOLDS['optimal_max'])
    fair_min = float(BODY_LEAN_THRESHOLDS['fair_min'])
    fair_max = float(BODY_LEAN_THRESHOLDS['fair_max'])
    return f"最优{optimal_min:.0f}-{optimal_max:.0f}°，一般不宜低于{fair_min:.0f}°或高于{fair_max:.0f}°"


def _body_lean_eval_text(lean_mean: float) -> str:
    optimal_min = float(BODY_LEAN_THRESHOLDS['optimal_min'])
    optimal_max = float(BODY_LEAN_THRESHOLDS['optimal_max'])
    fair_min = float(BODY_LEAN_THRESHOLDS['fair_min'])
    fair_max = float(BODY_LEAN_THRESHOLDS['fair_max'])
    if optimal_min <= lean_mean <= optimal_max:
        return '适中'
    if lean_mean < fair_min:
        return '偏小'
    if lean_mean > fair_max:
        return '偏大'
    return '一般'



class BaseAIProvider(ABC):
    """AI提供商基类"""

    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """生成文本"""
        pass

    @abstractmethod
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """分析图像（多模态）"""
        pass


class ZhipuProvider(BaseAIProvider):
    """智谱AI提供商 - 使用zai库"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化zai客户端"""
        try:
            from zai import ZhipuAiClient
            self.client = ZhipuAiClient(api_key=self.api_key)
            print("智谱AI客户端初始化成功")
        except ImportError:
            print("警告: zai库未安装，请运行 pip install zai")
            self.client = None
        except Exception as e:
            print(f"智谱AI客户端初始化失败: {e}")
            self.client = None

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """使用glm-4.5生成文本"""
        if not self.client:
            return "智谱AI客户端未初始化，请检查API密钥和zai库安装"

        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})

            response = self.client.chat.completions.create(
                model="glm-4.5-air",
                messages=messages,
                temperature=0.6,
                max_tokens=4000  # 增加token限制，避免文本截断
            )

            # 提取响应内容
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif isinstance(response, dict):
                return response.get('choices', [{}])[0].get('message', {}).get('content', 'API返回异常')
            else:
                return str(response)

        except Exception as e:
            return f"智谱AI请求错误: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """使用glm-4v分析图像"""
        if not self.client:
            return "智谱AI客户端未初始化"

        try:
            # 读取并编码图像
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_data}'}}
                ]
            }]

            response = self.client.chat.completions.create(
                model="glm-4v",
                messages=messages,
                temperature=0.6,
                max_tokens=1500
            )

            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
            elif isinstance(response, dict):
                return response.get('choices', [{}])[0].get('message', {}).get('content', 'API返回异常')
            else:
                return str(response)

        except Exception as e:
            return f"图像分析错误: {str(e)}"

    def analyze_video_frames(self, frame_paths: List[str], prompt: str) -> str:
        """分析多个视频帧"""
        if not frame_paths:
            return "无法分析：未提供帧"

        # 逐帧分析并汇总
        frame_analyses = []
        for i, frame_path in enumerate(frame_paths):
            frame_prompt = f"""请分析这张跑步姿态图片（第{i+1}帧）：
1. 描述跑者当前的姿态
2. 识别可能存在的技术问题
3. 评估动作质量（好/一般/需改进）

请用简洁的语言回答。"""

            result = self.analyze_image(frame_path, frame_prompt)
            if not result.startswith("智谱AI") and not result.startswith("图像分析错误"):
                frame_analyses.append(f"**帧 {i+1}**: {result}")

        if not frame_analyses:
            return "多模态分析失败：无法获取有效的帧分析结果"

        # 汇总分析
        summary_prompt = f"""基于以下各帧的分析结果，请生成一份综合的跑步技术问题报告：

{chr(10).join(frame_analyses)}

请按照以下格式输出：
1. 整体技术评估
2. 各时间段发现的问题
3. 需要重点关注的技术细节
4. 改进建议"""

        return self.generate_text(summary_prompt)

    def analyze_time_segments(self, keyframe_data: List[Dict], kinematic_results: Dict) -> str:
        """时间段问题分析"""
        if not keyframe_data:
            return "无法进行时间段分析：未提供关键帧数据"

        # 构建运动学数据摘要
        cadence = kinematic_results.get('cadence', {}).get('cadence', 0)
        vertical_amp = kinematic_results.get('vertical_motion', {}).get('amplitude_normalized', 0)
        stability = kinematic_results.get('stability', {}).get('overall', 0)

        # 获取相位时间信息
        gait_cycle = kinematic_results.get('gait_cycle', {})
        phase_duration = gait_cycle.get('phase_duration_ms', {})

        context = f"""
运动学数据参考：
- 步频: {cadence:.1f} 步/分
- 垂直振幅: {vertical_amp:.2f}% 躯干长度
- 稳定性评分: {stability:.1f}/100
- 触地时间: {phase_duration.get('ground_contact', 0):.1f}ms
- 腾空时间: {phase_duration.get('flight', 0):.1f}ms
"""

        # 分析每个关键帧
        segment_analyses = []
        for i, kf in enumerate(keyframe_data):
            if not kf.get('detected', False):
                segment_analyses.append(f"时间 {kf['time_sec']:.2f}s: 未检测到姿态")
                continue

            frame_prompt = f"""你是一位专业的跑步教练。请分析这张跑步姿态图片。

当前时间点: {kf['time_sec']:.2f}秒

{context}

请重点分析：
1. 此时刻的身体姿态是否正确
2. 膝关节、髋关节角度是否合理
3. 躯干前倾程度
4. 是否存在明显的技术问题

请用2-3句话简要描述你观察到的问题或亮点。"""

            try:
                result = self.analyze_image(kf['path'], frame_prompt)
                if not result.startswith("智谱AI") and not result.startswith("图像分析错误"):
                    segment_analyses.append(f"**{kf['time_sec']:.2f}秒**: {result}")
            except Exception as e:
                segment_analyses.append(f"时间 {kf['time_sec']:.2f}s: 分析失败 - {str(e)}")

        if not segment_analyses:
            return "时间段分析失败：无法获取有效的分析结果"

        # 生成综合报告
        final_prompt = f"""请基于以下各时间点的跑步姿态分析，生成一份专业的时间段问题分析报告：

{chr(10).join(segment_analyses)}

{context}

请按照以下格式输出报告：

## 时间段问题分析

### 问题时间段识别
（列出存在明显问题的时间段，说明问题类型）

### 技术问题汇总
（总结视频中反复出现的技术问题）

### 改进优先级
（按重要程度排序建议改进的方面）

### 训练建议
（提供具体可操作的训练方法）"""

        return self.generate_text(final_prompt)


class LocalRuleEngine(BaseAIProvider):
    """本地规则引擎（无需API）"""

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """基于规则生成文本"""
        return "请使用 generate_analysis_report 方法获取分析报告"

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """本地无法分析图像"""
        return "本地模式不支持图像分析，请配置智谱AI API"

    def generate_analysis_report(self, results: Dict) -> str:
        """基于规则生成分析报告（区分正面/侧面视角）"""
        quality = results.get('quality_evaluation', {})
        kinematic = results.get('kinematic_analysis', {})
        view_angle = results.get('view_angle', 'side')

        is_frontal = view_angle in ['front', 'back']
        score = quality.get('total_score', 0)

        # 构建报告头部
        report = f"""## 跑步技术分析报告

### 一、总体评价

| 指标 | 结果 |
|------|------|
| **总体评分** | **{score:.1f}/100** |
| **技术评级** | {quality.get('rating', '待评估')} |
| **分析视角** | {self._get_view_name(view_angle)} |

"""
        # 评分解读
        if score >= 85:
            report += "> 您的跑步技术处于**优秀**水平，动作协调高效！\n\n"
        elif score >= 70:
            report += "> 您的跑步技术**良好**，有一定基础，存在提升空间。\n\n"
        elif score >= 55:
            report += "> 您的跑步技术处于**一般**水平，建议针对性改进。\n\n"
        else:
            report += "> 您的跑步技术有较大**提升空间**，建议系统训练。\n\n"

        # ========== 根据视角生成不同的报告内容 ==========
        if is_frontal:
            report += self._generate_frontal_report_sections(quality, kinematic)
        else:
            report += self._generate_side_report_sections(quality, kinematic)

        # 通用部分：优势、薄弱项、建议
        report += self._generate_common_sections(quality, score)

        return report

    def _generate_frontal_report_sections(self, quality: Dict, kinematic: Dict) -> str:
        """生成正面视角专用报告内容"""
        report = ""

        # === 各维度表现 ===
        report += "### 二、各维度表现（正面视角）\n\n"
        frontal_dims = quality.get('frontal_dimension_scores', {})
        if frontal_dims:
            report += "| 维度 | 得分 | 等级 | 权重 |\n"
            report += "|------|------|------|------|\n"
            for key, name, weight in [
                ('lower_limb_alignment', '下肢力线', '35%'),
                ('lateral_stability', '横向稳定性', '35%'),
                ('efficiency', '效率', '30%'),
            ]:
                s = frontal_dims.get(key, 0)
                lv = '优秀' if s >= 85 else '良好' if s >= 70 else '一般' if s >= 55 else '待改进'
                report += f"| {name} | {s:.1f} | {lv} | {weight} |\n"
            report += "\n"

        # === 下肢力线分析 ===
        report += "### 三、下肢力线分析\n\n"
        lower_limb = kinematic.get('lower_limb_alignment', {})
        if lower_limb:
            left_leg = lower_limb.get('left_leg', {})
            right_leg = lower_limb.get('right_leg', {})
            hip_drop = lower_limb.get('hip_drop', {})
            issue_names = {'valgus': '膝外翻', 'varus': '膝内扣', 'normal': '正常', 'unstable': '不稳定'}

            report += "| 指标 | 左侧 | 右侧 | 状态 | 参考 |\n"
            report += "|------|------|------|------|------|\n"

            left_mean = left_leg.get('mean', 0)
            right_mean = right_leg.get('mean', 0)
            left_issue = issue_names.get(left_leg.get('issue', ''), '-')
            right_issue = issue_names.get(right_leg.get('issue', ''), '-')
            report += f"| 膝关节偏移 | {left_mean:.1f}° | {right_mean:.1f}° | 左{left_issue}/右{right_issue} | <5°正常 |\n"

            drop_mean = abs(hip_drop.get('mean', hip_drop.get('drop_mean', 0)))
            drop_max = abs(hip_drop.get('max', 0))
            report += f"| 髋部下沉 | {drop_mean:.1f}°均值 | {drop_max:.1f}°最大 | - | <3°优秀, <5°良好 |\n"

            asymmetry = lower_limb.get('asymmetry', 0)
            asym_status = '对称' if asymmetry <= 2 else '基本对称' if asymmetry <= 4 else '不对称'
            report += f"| 膝偏移差异 | {asymmetry:.1f}° | - | {asym_status} | <2°对称 |\n"

            report += "\n"

        # === 横向稳定性 ===
        report += "### 四、横向稳定性\n\n"
        report += "| 指标 | 数值 | 参考 |\n"
        report += "|------|------|------|\n"

        lateral_stab = kinematic.get('lateral_stability', {})
        sway_val = lateral_stab.get('hip_sway', 0) if isinstance(lateral_stab, dict) else 0
        if sway_val > 0:
            report += f"| 髋部横摆 | {sway_val:.2f}% | <3%稳定, <5%一般 |\n"

        shoulder = kinematic.get('shoulder_analysis', {})
        tilt_mean = abs(shoulder.get('tilt_mean', 0))
        tilt_max = abs(shoulder.get('tilt_max', 0))
        if shoulder:
            report += f"| 肩部倾斜(均值) | {tilt_mean:.1f}° | <3°优秀, <6°良好 |\n"
            report += f"| 肩部倾斜(最大) | {tilt_max:.1f}° | 参考 |\n"

        report += "\n"

        # === 效率指标 ===
        report += "### 五、效率指标\n\n"
        cadence_data = kinematic.get('cadence', {})
        cadence_val = cadence_data.get('cadence', 0)
        cadence_rating = cadence_data.get('rating', {})
        if cadence_val > 0:
            report += "| 指标 | 数值 | 评级 | 参考 |\n"
            report += "|------|------|------|------|\n"
            report += f"| 步频 | {cadence_val:.0f} 步/分 | {cadence_rating.get('level', '-')} | ≥185精英, ≥175优秀 |\n"
            report += "\n"

        # 正面视角说明
        report += "> **正面视角说明**: 本视角擅长分析下肢力线和横向稳定性。垂直振幅和触地时间在此视角下为估算值，未纳入分析。如需完整分析，建议配合侧面视角拍摄。\n\n"

        return report

    def _generate_side_report_sections(self, quality: Dict, kinematic: Dict) -> str:
        """生成侧面视角专用报告内容"""
        report = ""

        # === 各维度表现 ===
        report += "### 二、各维度表现\n\n"
        dims = quality.get('dimension_scores', {})
        report += "| 维度 | 得分 | 等级 | 权重 |\n"
        report += "|------|------|------|------|\n"
        for key, name, weight in [('stability', '稳定性', '30%'), ('efficiency', '效率', '40%'), ('form', '跑姿', '30%')]:
            s = dims.get(key, 0)
            lv = '优秀' if s >= 85 else '良好' if s >= 70 else '一般' if s >= 55 else '待改进'
            report += f"| {name} | {s:.1f} | {lv} | {weight} |\n"
        report += "\n"

        # === 效率指标 ===
        report += "### 三、效率指标\n\n"
        report += "| 指标 | 数值 | 评级 | 参考范围 |\n"
        report += "|------|------|------|----------|\n"

        # 步频
        cadence_data = kinematic.get('cadence', {})
        cadence_val = cadence_data.get('cadence', 0)
        cadence_rating = cadence_data.get('rating', {})
        if cadence_val > 0:
            report += f"| 步频 | {cadence_val:.0f} 步/分 | {cadence_rating.get('level', '-')} | ≥185精英, ≥175优秀, ≥165良好 |\n"

        # 垂直振幅
        vm = kinematic.get('vertical_motion', {})
        amp_norm = vm.get('amplitude_normalized', 0)
        amp_rating = vm.get('amplitude_rating', {})
        if amp_norm > 0:
            report += (
                f"| 垂直振幅 | {amp_norm:.1f}% | {amp_rating.get('level', '-')} | "
                f"{_vertical_amplitude_reference_text()} |\n"
            )

        # 触地时间
        gait = kinematic.get('gait_cycle', {})
        phase_ms = gait.get('phase_duration_ms', {})
        gc_ms = phase_ms.get('ground_contact', 0)
        gait_rating = gait.get('gait_rating', {})
        if gc_ms > 0:
            report += f"| 触地时间 | {gc_ms:.0f} ms | {gait_rating.get('level', '-')} | <210精英, <240优秀, <270良好 |\n"

        # 腾空时间
        fl_ms = phase_ms.get('flight', 0)
        if fl_ms > 0:
            report += f"| 腾空时间 | {fl_ms:.0f} ms | - | 仅供参考 |\n"

        report += "\n"

        # === 跑姿分析 ===
        report += "### 四、跑姿分析\n\n"
        angles = kinematic.get('angles', {})
        phase = angles.get('phase_analysis', {})
        gc_phase = phase.get('ground_contact', {})
        fl_phase = phase.get('flight', {})

        if gc_phase.get('count', 0) > 0 or fl_phase.get('count', 0) > 0:
            report += "| 阶段 | 膝角均值 | 范围 | 参考值 |\n"
            report += "|------|----------|------|--------|\n"
            if gc_phase.get('count', 0) > 0:
                report += f"| 触地期 | {gc_phase.get('mean', 0):.1f}° | {gc_phase.get('min', 0):.1f}°~{gc_phase.get('max', 0):.1f}° | 155-170° |\n"
            if fl_phase.get('count', 0) > 0:
                report += f"| 腾空期 | {fl_phase.get('mean', 0):.1f}° | {fl_phase.get('min', 0):.1f}°~{fl_phase.get('max', 0):.1f}° | 90-130° |\n"
            report += "\n"

        # 躯干前倾
        body_lean = kinematic.get('body_lean', {})
        lean_mean = body_lean.get('forward_lean', 0) if isinstance(body_lean, dict) else 0
        if lean_mean > 0:
            lean_eval = _body_lean_eval_text(lean_mean)
            report += f"- 躯干前倾: **{lean_mean:.1f}°**（{lean_eval}，参考 {_body_lean_reference_text()}）\n\n"

        # === 稳定性分析 ===
        report += "### 五、稳定性分析\n\n"
        stability = kinematic.get('stability', {})
        if isinstance(stability, dict) and stability:
            report += "| 指标 | 得分 | 说明 |\n"
            report += "|------|------|------|\n"
            trunk_val = stability.get('trunk', 0)
            head_val = stability.get('head', 0)
            overall_val = stability.get('overall', 0)
            report += f"| 综合稳定性 | {overall_val:.0f} | 躯干×0.6 + 头部×0.4 |\n"
            report += f"| 躯干稳定 | {trunk_val:.0f} | 躯干倾斜角度变异 |\n"
            report += f"| 头部稳定 | {head_val:.0f} | 头部相对肩部变异 |\n"
            report += "\n"

        return report

    def _generate_common_sections(self, quality: Dict, score: float) -> str:
        """生成通用报告部分（优势、薄弱项、建议、总结）"""
        report = ""

        # 优势
        strengths = quality.get('strengths', [])
        if strengths and strengths != ['暂无突出优势']:
            report += "### 技术优势\n\n"
            for s in strengths:
                report += f"- {s}\n"
            report += "\n"

        # 薄弱项
        weaknesses = quality.get('weaknesses', [])
        if weaknesses and weaknesses != ['无明显薄弱项']:
            report += "### 待改进项\n\n"
            for w in weaknesses:
                report += f"- {w}\n"
            report += "\n"

        # 改进建议
        suggestions = quality.get('suggestions', [])
        if suggestions:
            report += "### 改进建议\n\n"
            for i, sug in enumerate(suggestions, 1):
                report += f"{i}. {sug}\n"
            report += "\n"

        # 总结 — 基于分数给出简洁评语
        rating = quality.get('rating', '')
        report += f"### 总结（{score:.1f}分）\n\n"
        if score >= 85:
            report += "跑步技术出色，动作协调高效。保持当前训练节奏即可。\n"
        elif score >= 70:
            report += "整体表现良好，有较好基础。针对薄弱项进行专项训练可进一步提升。\n"
        elif score >= 55:
            report += "基础动作已掌握，部分环节可优化。建议优先改善核心稳定性，逐步调整跑姿。\n"
        else:
            report += "建议从基础开始，重点关注身体姿态控制和步频节奏的建立。\n"

        report += "\n---\n*本报告由跑步动作分析系统自动生成*\n"

        return report

    def _get_view_name(self, view: str) -> str:
        """获取视角中文名称"""
        names = {
            'side': '侧面视角',
            'front': '正面视角'
        }
        return names.get(view, view)


class AIAnalyzer:
    """AI分析器主类"""

    def __init__(self, api_key: str = None):
        """
        初始化AI分析器

        Args:
            api_key: 智谱AI API密钥
        """
        self.enabled = AI_CONFIG.get('enabled', False)
        api_key = api_key or AI_CONFIG.get('api_key', '')

        # 初始化智谱AI提供商
        if api_key:
            self.provider = ZhipuProvider(api_key)
            self.provider_name = 'zhipu'
        else:
            self.provider = LocalRuleEngine()
            self.provider_name = 'local'

        # 本地规则引擎实例（始终保留作为后备）
        self.local_engine = LocalRuleEngine()

    def generate_analysis_report(self, analysis_results: Dict, allow_local_fallback: bool = True) -> str:
        """
        生成 AI 分析报告。

        Args:
            analysis_results: 结构化分析结果。

        Returns:
            AI 分析报告文本。
        """
        if self.provider_name == 'local':
            if allow_local_fallback:
                return _normalize_report_output(self.local_engine.generate_analysis_report(analysis_results))
            return ""

        try:
            prompt = self._build_analysis_prompt(analysis_results)
            system_prompt = self._get_system_prompt()
            response = self.provider.generate_text(prompt, system_prompt)

            if response and not response.startswith('智谱AI') and not response.startswith('分析失败'):
                return _normalize_report_output(response, title='AI 分析报告文本。')

            print(f"智谱AI返回异常，回退到本地报告: {response}")
            if allow_local_fallback:
                return _normalize_report_output(self.local_engine.generate_analysis_report(analysis_results))
            return ""

        except Exception as e:
            print(f"AI分析失败: {e}")
            if allow_local_fallback:
                return _normalize_report_output(self.local_engine.generate_analysis_report(analysis_results))
            return ""

    def analyze_pose_image(self, image_path: str) -> str:
        """分析姿态图像"""
        if self.provider_name == 'local':
            return "本地模式不支持图像分析，请配置智谱AI API"

        prompt = """请分析这张跑步姿态图像：
1. 描述跑者的整体姿态
2. 分析膝关节、髋关节的角度是否合理
3. 评估躯干前倾程度
4. 判断手臂摆动是否协调
5. 指出可能的技术问题
6. 给出改进建议"""

        try:
            return self.provider.analyze_image(image_path, prompt)
        except Exception as e:
            return f"图像分析失败: {str(e)}"

    def analyze_video_sequence(self, frame_paths: List[str]) -> str:
        """分析视频帧序列"""
        if self.provider_name == 'local':
            return "本地模式不支持视频分析，请配置智谱AI API"

        prompt = """请分析这些跑步视频关键帧：
1. 描述跑者的整体技术水平
2. 分析步态周期的完整性
3. 评估动作的稳定性和一致性
4. 指出技术优势和不足
5. 提供专业的训练建议"""

        try:
            return self.provider.analyze_video_frames(frame_paths, prompt)
        except Exception as e:
            return f"视频分析失败: {str(e)}"

    def analyze_time_segments(self, keyframe_data: List[Dict], kinematic_results: Dict) -> str:
        """多模态时间段问题分析"""
        if self.provider_name == 'local':
            return self._local_time_segment_analysis(keyframe_data, kinematic_results)

        try:
            return self.provider.analyze_time_segments(keyframe_data, kinematic_results)
        except Exception as e:
            print(f"多模态时间段分析失败: {e}")
            return self._local_time_segment_analysis(keyframe_data, kinematic_results)

    def _local_time_segment_analysis(self, keyframe_data: List[Dict], kinematic_results: Dict) -> str:
        """本地时间段分析（基于规则）"""
        if not keyframe_data:
            return "无法进行时间段分析：未提供关键帧数据"

        cadence = kinematic_results.get('cadence', {}).get('cadence', 0)
        vertical_amp = kinematic_results.get('vertical_motion', {}).get('amplitude_normalized', 0)
        stability = kinematic_results.get('stability', {}).get('overall', 0)

        # 识别问题时间段
        problem_segments = []
        angles = kinematic_results.get('angles', {})
        phase_analysis = angles.get('phase_analysis', {})

        if cadence < 160:
            problem_segments.append("全程: 步频偏低，建议提高节奏")
        elif cadence > 210:
            problem_segments.append("全程: 步频过高，注意控制")

        if vertical_amp > 10:
            problem_segments.append("全程: 垂直振幅偏大，能量损耗较多")

        if stability < 60:
            problem_segments.append("全程: 动作稳定性不足，需加强核心训练")

        gc = phase_analysis.get('ground_contact', {})
        if gc.get('mean', 180) < 145:
            problem_segments.append("触地阶段: 膝关节弯曲过大")

        fl = phase_analysis.get('flight', {})
        if fl.get('mean', 90) > 140:
            problem_segments.append("腾空阶段: 腿部后摆不足")

        report = "## 时间段问题分析\n\n"

        if problem_segments:
            report += "### 识别到的问题\n\n"
            for i, problem in enumerate(problem_segments, 1):
                report += f"{i}. {problem}\n"
            report += "\n"
        else:
            report += "### 分析结果\n\n未发现明显技术问题，整体表现良好。\n\n"

        report += "### 关键帧时间点\n\n"
        for kf in keyframe_data:
            status = "姿态正常" if kf.get('detected', False) else "未检测到姿态"
            report += f"- {kf['time_sec']:.2f}s: {status}\n"

        report += "\n*如需更精确的多模态分析，请启用智谱AI*\n"

        return report

    def _build_analysis_prompt(self, results: Dict) -> str:
        """构建分析提示词（视角感知）"""
        view_angle = results.get('view_angle', 'side')
        is_frontal = view_angle in ['front', 'back']

        if is_frontal:
            return self._build_frontal_prompt(results)
        else:
            return self._build_side_prompt(results)

    def _build_side_prompt(self, results: Dict) -> str:
        """构建侧面视角AI提示词"""
        quality = results.get('quality_evaluation', {})
        kinematic = results.get('kinematic_analysis', {})
        dims = quality.get('dimension_scores', {})

        cadence_data = kinematic.get('cadence', {})
        cadence_val = cadence_data.get('cadence', 0)
        cadence_level = cadence_data.get('rating', {}).get('level', '-')

        vm = kinematic.get('vertical_motion', {})
        amp_norm = vm.get('amplitude_normalized', 0)
        amp_level = vm.get('amplitude_rating', {}).get('level', '-')

        gait = kinematic.get('gait_cycle', {})
        gc_ms = gait.get('phase_duration_ms', {}).get('ground_contact', 0)
        gc_level = gait.get('gait_rating', {}).get('level', '-')

        angles = kinematic.get('angles', {})
        phase = angles.get('phase_analysis', {})
        gc_angle = phase.get('ground_contact', {}).get('mean', 0)
        fl_angle = phase.get('flight', {}).get('mean', 0)

        stability = kinematic.get('stability', {})
        trunk_stab = stability.get('trunk', 0) if isinstance(stability, dict) else 0
        head_stab = stability.get('head', 0) if isinstance(stability, dict) else 0

        body_lean = kinematic.get('body_lean', {})
        lean_mean = body_lean.get('forward_lean', 0) if isinstance(body_lean, dict) else 0
        strengths_text = ', '.join(quality.get('strengths', [])) or '无'
        weaknesses_text = ', '.join(quality.get('weaknesses', [])) or '无'
        rating_label = quality.get('rating') or '未知'

        prompt = f"""你是一位专业跑步教练。以下是一次侧面跑步分析的结构化结果，请直接据此生成结论。

?总分?{quality.get('total_score', 0):.1f}/100（评级：{rating_label}）

?维度得分?
- 稳定性: {dims.get('stability', 0):.1f} (权重30%)
- 效率: {dims.get('efficiency', 0):.1f} (权重40%)
- 跑姿: {dims.get('form', 0):.1f} (权重30%)

?关键数据?
- 步频: {cadence_val:.0f} 步/分（评级：{cadence_level}）
- 垂直振幅: {amp_norm:.1f}%（评级：{amp_level}）
- 触地时间: {gc_ms:.0f} ms（评级：{gc_level}）
- 膝角度(触地期): {gc_angle:.1f}° (参考155-170°)
- 膝角度(腾空期): {fl_angle:.1f}° (参考90-130°)
- 躯干前倾: {lean_mean:.1f}° (参考{_body_lean_reference_text()})
- 躯干稳定性: {trunk_stab:.0f}/100
- 头部稳定性: {head_stab:.0f}/100

?已识别优势?{strengths_text}
?已识别薄弱项?{weaknesses_text}

请严格按以下 Markdown 结构输出：
# AI 跑步动作分析报告

## 总体评价
- 用2-3句话概括本次跑姿表现，并直接引用关键数值。

## 关键数据解读
- 分别说明步频、垂直振幅、触地时间、膝角、躯干前倾、稳定性。
- 必须指出至少2项表现较好的指标和2项需要优先改进的指标。

## 优先改进建议
1. 给出3-5条可执行建议。
2. 建议要和上面的具体数据一一对应。

## 一句话总结
- 用一句专业但简洁的话总结。

要求：
- 保持专业、克制、易读。
- 不要编造不存在的数据。
- 不解释算法原理，不输出泛泛而谈的鼓励话术。"""

        return prompt

    def _build_frontal_prompt(self, results: Dict) -> str:
        """构建正面视角AI提示词"""
        quality = results.get('quality_evaluation', {})
        kinematic = results.get('kinematic_analysis', {})
        frontal_dims = quality.get('frontal_dimension_scores', {})

        lower_limb = kinematic.get('lower_limb_alignment', {})
        left_leg = lower_limb.get('left_leg', {})
        right_leg = lower_limb.get('right_leg', {})
        hip_drop = lower_limb.get('hip_drop', {})

        issue_names = {'valgus': '膝外翻', 'varus': '膝内扣', 'normal': '正常', 'unstable': '不稳定'}
        left_issue = issue_names.get(left_leg.get('issue', ''), '-')
        right_issue = issue_names.get(right_leg.get('issue', ''), '-')

        shoulder = kinematic.get('shoulder_analysis', {})
        lateral_stab = kinematic.get('lateral_stability', {})
        hip_sway_val = lateral_stab.get('hip_sway', 0) if isinstance(lateral_stab, dict) else 0

        cadence_data = kinematic.get('cadence', {})
        cadence_val = cadence_data.get('cadence', 0)
        strengths_text = ', '.join(quality.get('strengths', [])) or '无'
        weaknesses_text = ', '.join(quality.get('weaknesses', [])) or '无'
        rating_label = quality.get('rating') or '未知'

        prompt = f"""你是一位专业跑步教练。以下是一次正面跑步分析的结构化结果，请直接据此生成结论。

?总分?{quality.get('total_score', 0):.1f}/100（评级：{rating_label}）

?维度得分?
- 下肢力线: {frontal_dims.get('lower_limb_alignment', 0):.1f} (权重35%)
- 横向稳定性: {frontal_dims.get('lateral_stability', 0):.1f} (权重35%)
- 效率: {frontal_dims.get('efficiency', 0):.1f} (权重30%)

?关键数据?
- 左膝偏移: {left_leg.get('mean', 0):.1f}°（状态：{left_issue}）
- 右膝偏移: {right_leg.get('mean', 0):.1f}°（状态：{right_issue}）
- 髋部下沉: {abs(hip_drop.get('mean', hip_drop.get('drop_mean', 0))):.1f}°（<3°优秀, <5°良好）
- 膝偏移差异: {lower_limb.get('asymmetry', 0):.1f}°（<2°对称）
- 肩部倾斜: {abs(shoulder.get('tilt_mean', 0)):.1f}°（<3°优秀）
- 髋部横摆: {hip_sway_val:.2f}%（<3%稳定）
- 步频: {cadence_val:.0f} 步/分

?已识别优势?{strengths_text}
?已识别薄弱项?{weaknesses_text}

请严格按以下 Markdown 结构输出：
# AI 跑步动作分析报告

## 总体评价
- 用2-3句话概括本次正面跑姿表现，并引用关键数值。

## 关键数据解读
- 重点分析下肢力线、横向稳定性、髋部下沉、肩部倾斜、左右差异和步频。
- 必须指出至少2项表现较好的指标和2项需要优先改进的指标。

## 优先改进建议
1. 给出3-5条可执行建议。
2. 建议要和上面的具体数据一一对应。

## 一句话总结
- 用一句专业但简洁的话总结。

要求：
- 保持专业、克制、易读。
- 不要编造不存在的数据。
- 正面视角重点关注下肢力线与横向稳定性，不要过度解读侧面专属指标。"""

        return prompt

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return "你是一位严谨的跑步技术教练和论文项目顾问。你只基于已给出的结构化结果进行解读，输出专业、简洁、可复盘的 Markdown 报告。必须直接引用数值，避免空泛措辞，不渲染营销感，不编造任何未提供的信息。"


def create_ai_analyzer(api_key: str = None) -> AIAnalyzer:
    """
    创建AI分析器

    Args:
        api_key: 智谱AI API密钥

    Returns:
        AIAnalyzer实例
    """
    return AIAnalyzer(api_key)


# 模块测试
if __name__ == "__main__":
    print("=" * 60)
    print("测试AI分析模块（智谱AI版）")
    print("=" * 60)

    mock_results = {
        'quality_evaluation': {
            'total_score': 75.5,
            'rating': '良好',
            'dimension_scores': {
                'stability': 78,
                'efficiency': 72,
                'form': 76
            },
            'strengths': ['动作稳定性'],
            'weaknesses': [],
            'suggestions': ['可适当提高步频', '保持当前良好状态']
        },
        'kinematic_analysis': {
            'cadence': {'cadence': 175, 'step_count': 15, 'duration': 5.0},
            'vertical_motion': {'amplitude_normalized': 7.5},
            'stability': {'overall': 78, 'trunk': 82, 'head': 75},
            'gait_cycle': {
                'phase_duration_ms': {
                    'ground_contact': 180.5,
                    'flight': 120.3,
                    'transition': 45.2
                },
                'avg_cycle_duration_ms': 345.0
            }
        }
    }

    print("\n测试本地规则引擎...")
    analyzer = AIAnalyzer()
    report = analyzer.local_engine.generate_analysis_report(mock_results)
    print(report)

    print("\n智谱AI分析模块测试完成!")
