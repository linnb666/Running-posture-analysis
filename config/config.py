import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'

# 创建必要目录
for dir_path in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR,
                 OUTPUT_DIR / 'videos', OUTPUT_DIR / 'visualizations']:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据库配置
DATABASE_PATH = DATA_DIR / 'database.db'

# 视频处理配置
VIDEO_CONFIG = {
    'target_width': 640,
    'target_height': 480,
    'fps': 30,
    'supported_formats': ['.mp4', '.avi', '.mov', '.mkv']
}

# ================== 姿态估计配置 ==================

# MediaPipe Pose配置
POSE_CONFIG = {
    'backend': 'mediapipe',  # 'mediapipe' 或 'mmpose'
    'model_complexity': 1,  # 0, 1, 2 (复杂度递增)
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'static_image_mode': False
}

# MMPose配置（预留）
MMPOSE_CONFIG = {
    'det_model': 'rtmdet',
    'det_checkpoint': '',  # 检测模型权重路径
    'pose_model': 'rtmpose',
    'pose_checkpoint': '',  # 姿态模型权重路径
    'device': 'cuda:0'  # 'cuda:0' 或 'cpu'
}

# ================== 视角检测配置 ==================

VIEW_DETECTION_CONFIG = {
    # 视角判断阈值
    'side_view_threshold': 0.4,      # 肩宽/髋宽比值阈值，低于此值判定为侧面
    'frontal_view_threshold': 0.7,   # 高于此值判定为正面
    'ear_visibility_threshold': 0.3,  # 耳朵可见性阈值
    'nose_offset_threshold': 0.15,    # 鼻子偏移阈值

    # 混合视角判断
    'mixed_view_ratio': 0.3,  # 如果侧面帧占比超过此值但不到0.7，判定为混合视角

    # 置信度阈值
    'min_confidence': 0.5,  # 关键点最低置信度

    # 分析策略
    'analysis_strategies': {
        'side': ['knee_angle', 'vertical_oscillation', 'trunk_lean', 'arm_swing'],
        'front': ['shoulder_symmetry', 'hip_alignment', 'knee_valgus', 'foot_strike'],
        'back': ['shoulder_symmetry', 'hip_alignment', 'heel_whip'],
        'mixed': ['knee_angle', 'vertical_oscillation', 'shoulder_symmetry']
    }
}

# ================== 运动学分析配置 ==================

KINEMATIC_CONFIG = {
    'smooth_window': 5,        # 平滑窗口大小
    'min_step_duration': 0.2,  # 最小步态周期(秒)
    'max_step_duration': 1.5,  # 最大步态周期(秒)

    # 躯干归一化配置
    'trunk_normalization': {
        'enabled': True,
        'fallback_ratio': 0.3,  # 当无法计算躯干长度时，使用图像高度的比例作为参考
        'min_trunk_length': 0.1,  # 最小躯干长度（归一化坐标）
        'smoothing_window': 3     # 躯干长度平滑窗口
    },

    # 相位检测配置
    'phase_detection': {
        'enabled': True,
        'ground_contact_threshold': 0.02,  # 触地判断的Y坐标变化阈值
        'flight_threshold': 0.05,          # 腾空判断的Y坐标阈值
        'min_phase_frames': 3              # 最小相位持续帧数
    },

    # 垂直振幅配置（基于躯干长度归一化，4档标准，参考高驰换算）
    'vertical_amplitude': {
        'excellent_max': 0.08,   # 优秀：≤8%躯干长度
        'good_max': 0.12,        # 良好：≤12%躯干长度
        'fair_max': 0.16,        # 一般：≤16%躯干长度
        'poor_min': 0.16         # 待改进：>16%躯干长度
    },

    # 膝关节角度配置（分阶段）
    'knee_angle': {
        'ground_contact': {
            'optimal_range': (155, 175),  # 触地阶段最优范围
            'acceptable_range': (145, 180)
        },
        'flight': {
            'optimal_range': (90, 120),   # 腾空阶段最优范围
            'acceptable_range': (80, 140)
        },
        'transition': {
            'optimal_range': (120, 155),  # 过渡阶段最优范围
            'acceptable_range': (100, 165)
        }
    }
}

# ================== 深度学习模型配置 ==================

MODEL_CONFIG = {
    'input_dim': 33 * 2,  # MediaPipe 33个关键点 * 2D坐标
    'hidden_dim': 64,
    'num_layers': 2,
    'output_dim': 3,  # 触地/腾空/过渡
    'dropout': 0.3,
    'sequence_length': 30,  # 时间序列长度
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50
}

# ================== MotionBERT 3D姿态提升配置 ==================

# 原始H36M权重（备份）: best_epoch.bin, maxlen=243
# 当前使用: AP3D微调权重（跑步动作优化）

MOTIONBERT_CONFIG = {
    'enabled': True,  # 是否启用3D姿态提升
    'checkpoint_path': str(CHECKPOINT_DIR / 'ap3d_rm_v2_best.pth'),  # AP3D V2权重（含2D一致性约束）
    'device': 'auto',  # 'auto', 'cuda', 'cpu'

    # 模型架构配置（AP3D微调版本）
    'model': {
        'dim_in': 3,        # 输入维度（2D坐标 + confidence）
        'dim_out': 3,       # 输出维度（3D坐标）
        'dim_feat': 512,    # 特征维度
        'dim_rep': 512,     # 表示维度
        'depth': 5,         # Transformer深度
        'num_heads': 8,     # 注意力头数
        'mlp_ratio': 2.0,   # MLP扩展比例
        'num_joints': 17,   # 关键点数量（H36M格式）
        'maxlen': 81,       # 序列长度（AP3D使用81帧）
        'qkv_bias': True,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'att_fuse': True
    },

    # 显存优化配置（针对4GB显存）
    'memory_optimization': {
        'max_batch_frames': 81,     # 每批最大帧数（与maxlen一致）
        'use_fp16': False,          # 是否使用半精度（暂不启用，兼容性考虑）
        'gradient_checkpointing': False
    },

    # 数据可靠性配置
    'reliability': {
        '3d_confidence_threshold': 0.5,  # 3D数据置信度阈值
        'min_valid_frames_ratio': 0.7,   # 最小有效帧比例
    }
}

# ================== AthletePose3D 微调配置 ==================

AP3D_CONFIG = {
    # 数据路径
    'data_dir': str(BASE_DIR / 'pose_3d_v3'),
    'train_pkl': str(BASE_DIR / 'pose_3d_v3' / 'train.pkl'),
    'valid_pkl': str(BASE_DIR / 'pose_3d_v3' / 'valid.pkl'),
    'frame81_dir': str(BASE_DIR / 'pose_3d_v3' / 'frame_81'),

    # 序列长度（AP3D 使用 81 帧）
    'maxlen': 81,

    # 筛选的运动类型
    'actions': ['rm'],  # rm = Running Motion (跑步)

    # 是否使用 MediaPipe 映射
    'use_mediapipe_mapping': True,

    # 训练配置（针对 RTX2050 4GB 显存优化）
    'training': {
        'batch_size': 8,          # 批次大小
        'epochs': 6,              # 训练轮数
        'lr': 1e-4,               # 学习率（微调时用 1e-5）
        'weight_decay': 1e-4,     # 权重衰减
        'warmup_epochs': 1,       # 预热轮数
        'lr_decay': 'cosine',     # 学习率衰减策略

        # 冻结策略
        'freeze_layers': 3,       # 冻结前 N 层 encoder (blocks_st/ts)
        'unfreeze_head': True,    # 始终解冻输出层

        # 优化器
        'optimizer': 'AdamW',

        # 数据增强
        'augment': True,
        'augment_config': {
            'coord_noise_std': 0.005,   # 坐标噪声
            'dropout_prob': 0.02,       # 随机丢点
            'conf_noise_std': 0.05,     # 置信度噪声
            'flip_prob': 0.3,           # 水平翻转
        },

        # 验证
        'val_ratio': 0.2,         # 验证集比例
        'eval_interval': 1,       # 每 N 轮评估一次

        # 保存
        'save_interval': 1,       # 每 N 轮保存一次
        'save_best': True,        # 保存最佳模型
    },

    # 模型配置（覆盖 MOTIONBERT_CONFIG 的部分参数）
    'model': {
        'maxlen': 81,  # AP3D 使用 81 帧序列
        # 其他参数继承自 MOTIONBERT_CONFIG
    },

    # 输出路径
    'checkpoint_dir': str(CHECKPOINT_DIR),
    'checkpoint_prefix': 'ap3d_rm_finetune',
}

# ================== 技术质量评价配置 ==================

# 评价维度权重（三维评价体系）
QUALITY_WEIGHTS = {
    'stability': 0.30,     # 动作稳定性（降低）
    'efficiency': 0.40,    # 动作效率（提高）
    'form': 0.30,          # 跑姿标准度
}

# 评价等级阈值
QUALITY_THRESHOLDS = {
    'excellent': 85,
    'good': 70,
    'fair': 55,
    'poor': 0
}

# 步频标准（与quality_evaluator.py保持一致）
# 五级评价标准：精英(185+) > 优秀(175-185) > 良好(165-175) > 一般(155-165) > 较差(<155)
CADENCE_THRESHOLDS = {
    'elite_min': 185,      # 精英级别下限
    'excellent_min': 175,  # 优秀级别下限
    'good_min': 165,       # 良好级别下限
    'fair_min': 155,       # 一般级别下限
    'acceptable_min': 140, # 可接受下限（低于此值为不健康）
    'acceptable_max': 220  # 可接受上限（高于此值可能为过快）
}

# 垂直振幅标准（稳健版，单位：相对躯干长度百分比）
# 依据高驰 5/8/12/15 cm 分档，并考虑躯干长度浮动后的稳健换算
VERTICAL_AMPLITUDE_THRESHOLDS = {
    'excellent_max': 11.0,
    'good_max': 17.0,
    'fair_max': 25.0,
}

# 躯干前倾标准（稳健版，单位：度）
# 面向中长跑/马拉松经济性场景：中等前倾最优，过直立和过度前倾都应扣分
BODY_LEAN_THRESHOLDS = {
    'optimal_min': 4.0,
    'optimal_max': 8.0,
    'good_min': 3.0,
    'good_max': 10.0,
    'fair_min': 2.0,
    'fair_max': 12.0,
}

# ================== AI分析配置 ==================

# 智谱AI API密钥（填入自己的密钥，或通过环境变量 ZHIPU_API_KEY 提供）
ZHIPU_API_KEY = os.getenv('ZHIPU_API_KEY', '')

AI_CONFIG = {
    'enabled': True,  # 启用AI分析
    'provider': 'zhipu',  # 默认使用智谱AI
    'api_key': ZHIPU_API_KEY,

    # 提供商配置
    'providers': {
        'openai': {
            'enabled': False,
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'api_base': os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
            'model': 'gpt-4-turbo-preview',
            'vision_model': 'gpt-4-vision-preview',
            'max_tokens': 1000,
            'temperature': 0.7
        },
        'anthropic': {
            'enabled': False,
            'api_key': os.getenv('ANTHROPIC_API_KEY', ''),
            'model': 'claude-3-sonnet-20240229',
            'vision_model': 'claude-3-sonnet-20240229',
            'max_tokens': 1000,
            'temperature': 0.7
        },
        'qwen': {
            'enabled': False,
            'api_key': os.getenv('DASHSCOPE_API_KEY', ''),
            'model': 'qwen-turbo',
            'vision_model': 'qwen-vl-plus',
            'max_tokens': 1000,
            'temperature': 0.7
        },
        'zhipu': {
            'enabled': True,
            'api_key': ZHIPU_API_KEY,
            'api_base': 'https://open.bigmodel.cn/api/paas/v4',
            'model': 'glm-4.5-air',
            'vision_model': 'glm-4v',
            'max_tokens': 4000,  # 增加token限制，避免文本截断
            'temperature': 0.7
        },
        'local': {
            'enabled': True,  # 本地规则引擎始终可用作后备
            'model': 'rule_engine'
        }
    }
}

# ================== Flask API配置 ==================

API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True
}

# ================== Streamlit配置 ==================

STREAMLIT_CONFIG = {
    'page_title': '跑步动作分析系统',
    'page_icon': '🏃',
    'layout': 'wide'
}

# ================== 日志配置 ==================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(OUTPUT_DIR / 'analysis.log')
}
