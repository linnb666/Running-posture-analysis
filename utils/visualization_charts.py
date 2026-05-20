# utils/visualization_charts.py
"""
数据可视化图表工具模块

为Streamlit界面提供专业的数据可视化图表，包括：
1. 评分雷达图 - 展示各维度得分
2. 阶段分布饼图 - 展示步态阶段分布
3. 膝关节角度时序图 - 展示角度变化曲线
4. 步态时间轴 - 展示触地/腾空周期
5. 指标对比条形图 - 与参考值对比
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_radar_chart(dimension_scores: Dict[str, float],
                       title: str = "技术质量评分雷达图") -> go.Figure:
    """
    创建评分雷达图

    Args:
        dimension_scores: 各维度得分
            - 侧面视角: {'stability': 80, 'efficiency': 75, 'form': 70}
            - 正面视角: {'下肢力线': 80, '横向稳定性': 75, '对称性': 70}
            - 或任意自定义维度名称和分数
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    # 检查是否是侧面视角标准键名
    standard_keys = {'stability', 'efficiency', 'form'}
    if set(dimension_scores.keys()) <= standard_keys or \
       all(k in standard_keys for k in dimension_scores.keys()):
        # 侧面视角：使用标准中文名称
        categories = ['稳定性', '效率', '跑姿']
        values = [
            dimension_scores.get('stability', 0),
            dimension_scores.get('efficiency', 0),
            dimension_scores.get('form', 0)
        ]
    else:
        # 正面视角或自定义维度：直接使用传入的键名和值
        categories = list(dimension_scores.keys())
        values = list(dimension_scores.values())

    # 闭合雷达图
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()

    # 添加实际得分
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='实际得分',
        line_color='rgb(31, 119, 180)',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    # 添加参考线（优秀标准：80分）
    ref_values = [80] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=ref_values,
        theta=categories,
        name='优秀标准 (80)',
        line=dict(color='rgba(44, 160, 44, 0.5)', dash='dash'),
        fill=None
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20', '40', '60', '80', '100']
            )
        ),
        showlegend=True,
        title=dict(text=title, x=0.5),
        height=400
    )

    return fig


def create_phase_distribution_pie(phase_distribution: Dict[str, float],
                                   title: str = "步态阶段分布") -> go.Figure:
    """
    创建步态阶段分布饼图

    Args:
        phase_distribution: 阶段分布 {'ground_contact': 0.45, 'flight': 0.35, 'transition': 0.20}
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    labels = ['触地期', '腾空期', '过渡期']
    values = [
        phase_distribution.get('ground_contact', 0) * 100,
        phase_distribution.get('flight', 0) * 100,
        phase_distribution.get('transition', 0) * 100
    ]
    colors = ['#2ecc71', '#3498db', '#f39c12']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='outside',
        pull=[0.02, 0.02, 0.02]
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5),
        annotations=[dict(text='步态<br>分布', x=0.5, y=0.5, font_size=14, showarrow=False)],
        height=350,
        showlegend=False
    )

    return fig


def create_knee_angle_chart(phase_analysis: Dict,
                            title: str = "膝关节角度分析") -> go.Figure:
    """
    创建膝关节角度对比条形图

    Args:
        phase_analysis: 阶段角度分析数据
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    phases = ['触地期', '腾空期']
    actual_values = []
    ref_min = []
    ref_max = []

    # 触地期（基于马拉松运动员研究数据调整）
    gc = phase_analysis.get('ground_contact', {})
    gc_mean = gc.get('mean', 0) if isinstance(gc, dict) else gc if isinstance(gc, (int, float)) else 0
    actual_values.append(gc_mean)
    ref_min.append(150)  # 理想下限
    ref_max.append(165)  # 理想上限

    # 腾空期
    fl = phase_analysis.get('flight', {})
    fl_mean = fl.get('mean', 0) if isinstance(fl, dict) else fl if isinstance(fl, (int, float)) else 0
    actual_values.append(fl_mean)
    ref_min.append(90)
    ref_max.append(130)

    fig = go.Figure()

    # 参考范围（作为背景）
    fig.add_trace(go.Bar(
        name='参考上限',
        x=phases,
        y=ref_max,
        marker_color='rgba(200, 200, 200, 0.3)',
        width=0.6
    ))

    # 实际值
    colors = []
    for i, val in enumerate(actual_values):
        if val == 0:
            colors.append('gray')
        elif ref_min[i] <= val <= ref_max[i]:
            colors.append('#2ecc71')  # 绿色 - 在范围内
        else:
            colors.append('#e74c3c')  # 红色 - 超出范围

    fig.add_trace(go.Bar(
        name='实际值',
        x=phases,
        y=actual_values,
        marker_color=colors,
        text=[f'{v:.1f}°' if v > 0 else 'N/A' for v in actual_values],
        textposition='outside',
        width=0.4
    ))

    # 添加参考线
    for i, phase in enumerate(phases):
        fig.add_shape(
            type='line',
            x0=i-0.3, x1=i+0.3,
            y0=ref_min[i], y1=ref_min[i],
            line=dict(color='green', dash='dash', width=2)
        )
        fig.add_shape(
            type='line',
            x0=i-0.3, x1=i+0.3,
            y0=ref_max[i], y1=ref_max[i],
            line=dict(color='green', dash='dash', width=2)
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis_title='角度 (°)',
        barmode='overlay',
        height=350,
        showlegend=False,
        yaxis=dict(range=[0, 200])
    )

    # 添加注释
    fig.add_annotation(
        x=0.5, y=-0.15,
        xref='paper', yref='paper',
        text='绿色虚线表示理想范围',
        showarrow=False,
        font=dict(size=10, color='gray')
    )

    return fig


def create_metrics_comparison_chart(metrics_data: Dict,
                                     title: str = "关键指标与参考值对比") -> go.Figure:
    """
    创建关键指标与参考值对比图

    Args:
        metrics_data: 指标数据字典，格式为：
            {
                '指标名称': {
                    'value': 实际值,
                    'ref_min': 参考最小值,
                    'ref_max': 参考最大值
                },
                ...
            }
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    if not metrics_data:
        fig = go.Figure()
        fig.add_annotation(text="数据不足", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=350)
        return fig

    metrics = list(metrics_data.keys())
    actual = []
    ref_optimal = []

    for metric_name, data in metrics_data.items():
        value = data.get('value', 0)
        ref_min = data.get('ref_min', 0)
        ref_max = data.get('ref_max', 0)
        actual.append(value)
        # 使用参考范围的中值作为理想值
        ref_optimal.append((ref_min + ref_max) / 2)

    # 归一化用于比较（百分比差异）
    normalized_actual = []
    for a, r in zip(actual, ref_optimal):
        if r > 0:
            diff_pct = ((a - r) / r) * 100
            normalized_actual.append(diff_pct)
        else:
            normalized_actual.append(0)

    # 创建条形图
    colors = ['#2ecc71' if abs(n) <= 10 else '#f39c12' if abs(n) <= 20 else '#e74c3c'
              for n in normalized_actual]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=metrics,
        y=normalized_actual,
        marker_color=colors,
        text=[f'{a:.1f}' for a in actual],
        textposition='outside'
    ))

    # 添加参考线（0表示等于理想值）
    fig.add_hline(y=0, line_dash='dash', line_color='green',
                  annotation_text='理想值', annotation_position='right')
    fig.add_hline(y=10, line_dash='dot', line_color='orange', opacity=0.5)
    fig.add_hline(y=-10, line_dash='dot', line_color='orange', opacity=0.5)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        yaxis_title='与理想值偏差 (%)',
        height=350,
        yaxis=dict(range=[-50, 50])
    )

    return fig


def create_gait_timeline(phase_duration: Dict, fps: float = 30,
                         title: str = "步态周期时间轴") -> go.Figure:
    """
    创建步态周期时间轴图

    Args:
        phase_duration: 阶段时长数据 {'ground_contact': ms, 'flight': ms}
        fps: 帧率
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    # 兼容两种输入格式
    if 'phase_duration_ms' in phase_duration:
        phase_duration = phase_duration.get('phase_duration_ms', {})

    gc_time = phase_duration.get('ground_contact', 0)
    flight_time = phase_duration.get('flight', 0)

    if gc_time == 0 and flight_time == 0:
        fig = go.Figure()
        fig.add_annotation(text="步态数据不足", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=250)
        return fig

    cycle_time = gc_time + flight_time

    # 创建时间轴
    fig = go.Figure()

    # 模拟多个步态周期
    num_cycles = 3
    for i in range(num_cycles):
        start = i * cycle_time

        # 触地期
        fig.add_trace(go.Bar(
            x=[gc_time],
            y=[f'周期 {i+1}'],
            orientation='h',
            base=start,
            marker_color='#2ecc71',
            name='触地期' if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=f'触地期: {gc_time:.0f}ms<extra></extra>'
        ))

        # 腾空期
        fig.add_trace(go.Bar(
            x=[flight_time],
            y=[f'周期 {i+1}'],
            orientation='h',
            base=start + gc_time,
            marker_color='#3498db',
            name='腾空期' if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=f'腾空期: {flight_time:.0f}ms<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='时间 (ms)',
        barmode='stack',
        height=250,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    # 添加时间标注
    fig.add_annotation(
        x=cycle_time / 2, y=-0.3,
        xref='x', yref='paper',
        text=f'单周期: {cycle_time:.0f}ms',
        showarrow=False,
        font=dict(size=12)
    )

    return fig


def create_score_gauge(score: float, title: str = "总体评分") -> go.Figure:
    """
    创建评分仪表盘

    Args:
        score: 评分 (0-100)
        title: 标题

    Returns:
        Plotly图表对象
    """
    # 确定颜色
    if score >= 85:
        color = '#2ecc71'  # 绿色 - 优秀
    elif score >= 70:
        color = '#3498db'  # 蓝色 - 良好
    elif score >= 55:
        color = '#f39c12'  # 橙色 - 一般
    else:
        color = '#e74c3c'  # 红色 - 待改进

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': 70, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 55], 'color': 'rgba(231, 76, 60, 0.2)'},
                {'range': [55, 70], 'color': 'rgba(243, 156, 18, 0.2)'},
                {'range': [70, 85], 'color': 'rgba(52, 152, 219, 0.2)'},
                {'range': [85, 100], 'color': 'rgba(46, 204, 113, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))

    fig.update_layout(height=280)

    return fig


def create_angle_time_series(angles_data: Dict,
                              title: str = "膝关节角度变化曲线") -> go.Figure:
    """
    创建膝关节角度时序图（如果有原始数据）

    Args:
        angles_data: 角度数据（需要包含时间序列）
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    fig = go.Figure()

    # 如果有时间序列数据
    knee_left = angles_data.get('knee_left_sequence', [])
    knee_right = angles_data.get('knee_right_sequence', [])

    if knee_left:
        x = list(range(len(knee_left)))
        fig.add_trace(go.Scatter(
            x=x, y=knee_left,
            mode='lines',
            name='左膝',
            line=dict(color='#3498db', width=2)
        ))

    if knee_right:
        x = list(range(len(knee_right)))
        fig.add_trace(go.Scatter(
            x=x, y=knee_right,
            mode='lines',
            name='右膝',
            line=dict(color='#e74c3c', width=2)
        ))

    # 添加参考区域
    if knee_left or knee_right:
        max_x = max(len(knee_left), len(knee_right))
        fig.add_hrect(y0=155, y1=170,
                      fillcolor="green", opacity=0.1,
                      annotation_text="触地理想范围", annotation_position="right")
        fig.add_hrect(y0=90, y1=130,
                      fillcolor="blue", opacity=0.1,
                      annotation_text="腾空理想范围", annotation_position="right")

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='帧序号',
        yaxis_title='角度 (°)',
        height=350,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )

    if not knee_left and not knee_right:
        fig.add_annotation(text="无时间序列数据", x=0.5, y=0.5, showarrow=False)

    return fig


def create_summary_dashboard(quality: Dict, kinematic: Dict, temporal: Dict) -> go.Figure:
    """
    创建综合仪表板（多图组合）

    Args:
        quality: 质量评价结果
        kinematic: 运动学分析结果
        temporal: 时序分析结果

    Returns:
        Plotly图表对象
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        subplot_titles=("总体评分", "步态阶段分布", "维度得分", "关键指标"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. 总体评分指示器
    score = quality.get('total_score', 0)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#3498db'},
                'steps': [
                    {'range': [0, 55], 'color': 'rgba(231, 76, 60, 0.3)'},
                    {'range': [55, 70], 'color': 'rgba(243, 156, 18, 0.3)'},
                    {'range': [70, 85], 'color': 'rgba(52, 152, 219, 0.3)'},
                    {'range': [85, 100], 'color': 'rgba(46, 204, 113, 0.3)'}
                ]
            }
        ),
        row=1, col=1
    )

    # 2. 阶段分布饼图
    phase_dist = temporal.get('phase_distribution', {})
    fig.add_trace(
        go.Pie(
            labels=['触地期', '腾空期', '过渡期'],
            values=[
                phase_dist.get('ground_contact', 0.33),
                phase_dist.get('flight', 0.33),
                phase_dist.get('transition', 0.34)
            ],
            hole=0.4,
            marker_colors=['#2ecc71', '#3498db', '#f39c12']
        ),
        row=1, col=2
    )

    # 3. 维度得分条形图
    dims = quality.get('dimension_scores', {})
    fig.add_trace(
        go.Bar(
            x=['稳定性', '效率', '跑姿'],
            y=[dims.get('stability', 0), dims.get('efficiency', 0), dims.get('form', 0)],
            marker_color=['#2ecc71', '#3498db', '#9b59b6'],
            text=[f"{dims.get('stability', 0):.0f}", f"{dims.get('efficiency', 0):.0f}", f"{dims.get('form', 0):.0f}"],
            textposition='outside'
        ),
        row=2, col=1
    )

    # 4. 关键指标
    cadence = kinematic.get('cadence', {}).get('cadence', 0)
    amp = kinematic.get('vertical_motion', {}).get('amplitude_normalized', 0)
    gc_time = kinematic.get('gait_cycle', {}).get('phase_duration_ms', {}).get('ground_contact', 0)

    fig.add_trace(
        go.Bar(
            x=['步频', '振幅', '触地时间'],
            y=[cadence, amp * 10, gc_time / 10],  # 归一化显示
            marker_color=['#1abc9c', '#e74c3c', '#f39c12'],
            text=[f'{cadence:.0f}', f'{amp:.1f}%', f'{gc_time:.0f}ms'],
            textposition='outside'
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="跑步技术分析仪表板"
    )

    return fig


def create_lower_limb_alignment_chart(chart_data: Dict,
                                       title: str = "下肢力线时序变化") -> go.Figure:
    """
    创建下肢力线时序图表

    Args:
        chart_data: 图表数据 {
            'left': [角度列表],
            'right': [角度列表],
            'time_pct': [时间百分比列表]
        }
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    left = chart_data.get('left', [])
    right = chart_data.get('right', [])
    time_pct = chart_data.get('time_pct', [])

    if not left or not right or not time_pct:
        # 返回空图
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="数据不足",
            showarrow=False,
            xref="paper", yref="paper"
        )
        fig.update_layout(height=250)
        return fig

    fig = go.Figure()

    # 左腿曲线
    fig.add_trace(go.Scatter(
        x=time_pct,
        y=left,
        mode='lines',
        name='左腿',
        line=dict(color='#3498db', width=2),
        hovertemplate='时间: %{x:.0f}%<br>左腿: %{y:.1f}°<extra></extra>'
    ))

    # 右腿曲线
    fig.add_trace(go.Scatter(
        x=time_pct,
        y=right,
        mode='lines',
        name='右腿',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='时间: %{x:.0f}%<br>右腿: %{y:.1f}°<extra></extra>'
    ))

    # 添加参考区域
    # 正常范围: -2° 到 +2°
    fig.add_hrect(y0=-2, y1=2, fillcolor="green", opacity=0.1,
                  line_width=0, annotation_text="正常范围",
                  annotation_position="right")

    # 轻度异常范围: ±2° 到 ±5°
    fig.add_hrect(y0=2, y1=5, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=-5, y1=-2, fillcolor="yellow", opacity=0.1, line_width=0)

    # 零线
    fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title='视频进度 (%)',
        yaxis_title='膝关节偏移角度 (°)',
        height=280,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        yaxis=dict(zeroline=True, zerolinecolor='gray'),
        hovermode='x unified'
    )

    # 添加注释
    fig.add_annotation(
        x=1.0, y=0,
        xref='paper', yref='y',
        text='正值=膝外翻<br>负值=膝内扣',
        showarrow=False,
        font=dict(size=10, color='gray'),
        align='left'
    )

    return fig


def create_hip_drop_stats_chart(left_stats: Dict, right_stats: Dict,
                                 hip_drop: Dict,
                                 title: str = "下肢力线统计数据") -> go.Figure:
    """
    创建下肢力线统计数据条形图

    Args:
        left_stats: 左腿统计 {mean, max, min, std}
        right_stats: 右腿统计
        hip_drop: 髋部下沉统计
        title: 图表标题

    Returns:
        Plotly图表对象
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['膝关节偏移角度', '髋部下沉角度'],
        horizontal_spacing=0.15
    )

    # 左右腿膝关节数据
    categories = ['平均值', '最大值', '标准差']
    left_values = [
        left_stats.get('mean', 0),
        left_stats.get('max', 0),
        left_stats.get('std', 0)
    ]
    right_values = [
        right_stats.get('mean', 0),
        right_stats.get('max', 0),
        right_stats.get('std', 0)
    ]

    fig.add_trace(
        go.Bar(
            name='左腿',
            x=categories,
            y=left_values,
            marker_color='#3498db',
            text=[f'{v:.1f}°' for v in left_values],
            textposition='outside'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            name='右腿',
            x=categories,
            y=right_values,
            marker_color='#e74c3c',
            text=[f'{v:.1f}°' for v in right_values],
            textposition='outside'
        ),
        row=1, col=1
    )

    # 髋部下沉数据
    if hip_drop:
        hip_cats = ['平均下沉', '最大下沉']
        hip_values = [
            hip_drop.get('mean', 0),
            hip_drop.get('max', 0)
        ]
        fig.add_trace(
            go.Bar(
                name='髋部',
                x=hip_cats,
                y=hip_values,
                marker_color='#9b59b6',
                text=[f'{v:.1f}°' for v in hip_values],
                textposition='outside'
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=280,
        barmode='group',
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5),
        title=dict(text=title, x=0.5)
    )

    return fig
