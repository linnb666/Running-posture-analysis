import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from config.config import DATABASE_PATH


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: Path = DATABASE_PATH):
        """初始化数据库"""
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建分析记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_filename TEXT NOT NULL,
                video_duration REAL,
                fps REAL,
                frame_count INTEGER,
                analysis_date TEXT NOT NULL,

                -- 技术质量评分
                total_score REAL,
                rating TEXT,
                stability_score REAL,
                efficiency_score REAL,
                form_score REAL,
                rhythm_score REAL,

                -- 运动学指标
                cadence REAL,
                step_count INTEGER,
                vertical_amplitude REAL,
                knee_angle_left REAL,
                knee_angle_right REAL,

                -- 深度学习结果
                dl_quality_score REAL,
                dl_stability_score REAL,
                phase_distribution TEXT,

                -- AI分析文本
                ai_analysis_text TEXT,

                -- 完整结果JSON
                full_results TEXT,

                -- 元数据
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_analysis_date 
            ON analysis_records(analysis_date)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_video_filename 
            ON analysis_records(video_filename)
        ''')

        conn.commit()
        conn.close()

    def save_analysis(self, results: Dict) -> int:
        """
        保存分析结果
        Args:
            results: 完整分析结果
        Returns:
            记录ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        video_info = results.get('video_info', {})
        kinematic = results.get('kinematic_analysis', {})
        temporal = results.get('temporal_analysis', {})
        quality = results.get('quality_evaluation', {})
        ai_text = results.get('ai_analysis', '')

        cursor.execute('''
            INSERT INTO analysis_records (
                video_filename, video_duration, fps, frame_count, analysis_date,
                total_score, rating, stability_score, efficiency_score, 
                form_score, rhythm_score,
                cadence, step_count, vertical_amplitude,
                knee_angle_left, knee_angle_right,
                dl_quality_score, dl_stability_score, phase_distribution,
                ai_analysis_text, full_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            video_info.get('filename', ''),
            video_info.get('duration', 0),
            video_info.get('fps', 0),
            video_info.get('frame_count', 0),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

            quality.get('total_score', 0),
            quality.get('rating', ''),
            quality.get('dimension_scores', {}).get('stability', 0),
            quality.get('dimension_scores', {}).get('efficiency', 0),
            quality.get('dimension_scores', {}).get('form', 0),
            quality.get('dimension_scores', {}).get('rhythm', 0),

            kinematic.get('cadence', {}).get('cadence', 0),
            kinematic.get('cadence', {}).get('step_count', 0),
            kinematic.get('vertical_motion', {}).get('amplitude', 0),
            kinematic.get('angles', {}).get('knee_left_mean', 0),
            kinematic.get('angles', {}).get('knee_right_mean', 0),

            temporal.get('quality_score', 0),
            temporal.get('stability_score', 0),
            json.dumps(temporal.get('phase_distribution', {})),

            ai_text,
            json.dumps(results, ensure_ascii=False)
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return record_id

    def get_analysis_by_id(self, record_id: int) -> Optional[Dict]:
        """根据ID获取分析记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM analysis_records WHERE id = ?
        ''', (record_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_dict(cursor, row)
        return None

    def get_recent_analyses(self, limit: int = 10) -> List[Dict]:
        """获取最近的分析记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM analysis_records 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(cursor, row) for row in rows]

    def search_analyses(self,
                        filename: Optional[str] = None,
                        min_score: Optional[float] = None,
                        rating: Optional[str] = None) -> List[Dict]:
        """搜索分析记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = 'SELECT * FROM analysis_records WHERE 1=1'
        params = []

        if filename:
            query += ' AND video_filename LIKE ?'
            params.append(f'%{filename}%')

        if min_score is not None:
            query += ' AND total_score >= ?'
            params.append(min_score)

        if rating:
            query += ' AND rating = ?'
            params.append(rating)

        query += ' ORDER BY created_at DESC'

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_dict(cursor, row) for row in rows]

    def _row_to_dict(self, cursor, row) -> Dict:
        """将数据库行转换为字典"""
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, row))

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM analysis_records')
        total_count = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(total_score) FROM analysis_records')
        avg_score = cursor.fetchone()[0] or 0

        cursor.execute('''
            SELECT rating, COUNT(*) as count
            FROM analysis_records
            GROUP BY rating
        ''')
        rating_distribution = dict(cursor.fetchall())

        conn.close()

        return {
            'total_analyses': total_count,
            'average_score': round(avg_score, 2),
            'rating_distribution': rating_distribution
        }

    def delete_analysis(self, record_id: int) -> bool:
        """
        删除分析记录
        Args:
            record_id: 记录ID
        Returns:
            是否成功删除
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM analysis_records WHERE id = ?', (record_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
            return deleted
        except Exception as e:
            conn.close()
            return False

    def delete_all_analyses(self) -> int:
        """
        删除所有分析记录
        Returns:
            删除的记录数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM analysis_records')
        count = cursor.fetchone()[0]

        cursor.execute('DELETE FROM analysis_records')
        conn.commit()
        conn.close()

        return count

    def get_full_results(self, record_id: int) -> Optional[Dict]:
        """
        获取完整的分析结果JSON
        Args:
            record_id: 记录ID
        Returns:
            完整结果字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT full_results FROM analysis_records WHERE id = ?', (record_id,))
        row = cursor.fetchone()
        conn.close()

        if row and row[0]:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return None
        return None