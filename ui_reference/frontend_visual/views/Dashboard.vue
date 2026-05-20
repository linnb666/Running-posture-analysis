<template>
  <div class="dashboard">
    <!-- 统计卡片 -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon blue">
          <el-icon><DataAnalysis /></el-icon>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ statistics.total_analyses || 0 }}</span>
          <span class="stat-label">总分析次数</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon green">
          <el-icon><CircleCheck /></el-icon>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ statistics.completed_analyses || 0 }}</span>
          <span class="stat-label">完成分析</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon orange">
          <el-icon><TrendCharts /></el-icon>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ statistics.average_score || '--' }}</span>
          <span class="stat-label">平均得分</span>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon purple">
          <el-icon><View /></el-icon>
        </div>
        <div class="stat-content">
          <span class="stat-value">{{ statistics.view_type_counts?.side || 0 }}</span>
          <span class="stat-label">侧面分析</span>
        </div>
      </div>
    </div>

    <!-- 快速操作 -->
    <div class="section">
      <h2 class="section-title">快速开始</h2>
      <div class="quick-actions">
        <div class="action-card" @click="$router.push('/analyze')">
          <div class="action-icon">
            <el-icon><Upload /></el-icon>
          </div>
          <div class="action-content">
            <h3>上传视频</h3>
            <p>上传跑步视频进行动作分析</p>
          </div>
          <el-icon class="action-arrow"><ArrowRight /></el-icon>
        </div>

        <div class="action-card" @click="$router.push('/history')">
          <div class="action-icon">
            <el-icon><Clock /></el-icon>
          </div>
          <div class="action-content">
            <h3>历史记录</h3>
            <p>查看和管理分析历史</p>
          </div>
          <el-icon class="action-arrow"><ArrowRight /></el-icon>
        </div>

        <div class="action-card" @click="$router.push('/compare')">
          <div class="action-icon">
            <el-icon><DataLine /></el-icon>
          </div>
          <div class="action-content">
            <h3>对比分析</h3>
            <p>多次分析结果对比</p>
          </div>
          <el-icon class="action-arrow"><ArrowRight /></el-icon>
        </div>
      </div>
    </div>

    <!-- 最近分析 -->
    <div class="section">
      <div class="section-header">
        <h2 class="section-title">最近分析</h2>
        <el-button type="primary" link @click="$router.push('/history')">
          查看全部 <el-icon><ArrowRight /></el-icon>
        </el-button>
      </div>

      <div v-if="loading" class="loading-placeholder">
        <el-skeleton :rows="3" animated />
      </div>

      <div v-else-if="recentAnalyses.length === 0" class="empty-state">
        <el-empty description="暂无分析记录">
          <el-button type="primary" @click="$router.push('/analyze')">
            开始第一次分析
          </el-button>
        </el-empty>
      </div>

      <div v-else class="recent-list">
        <div
          v-for="record in recentAnalyses"
          :key="record.id"
          class="recent-item"
          @click="goToResult(record)"
        >
          <div class="recent-info">
            <span class="recent-name">{{ record.video_filename }}</span>
                <span class="recent-meta">
              {{ viewLabel(record.view_type) }}
              · {{ formatDate(record.created_at) }}
            </span>
          </div>
          <div class="recent-score" :class="getGradeClass(record.grade)">
            <span class="score-value">{{ record.total_score?.toFixed(1) || '--' }}</span>
            <span class="score-label">{{ getGradeLabel(record.grade) }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { historyApi } from '@/api/history'
import { gradeLabel, viewLabel } from '@/utils/analysis'
import {
  DataAnalysis,
  CircleCheck,
  TrendCharts,
  View,
  Upload,
  Clock,
  DataLine,
  ArrowRight
} from '@element-plus/icons-vue'

const router = useRouter()

const loading = ref(true)
const statistics = ref({})
const recentAnalyses = ref([])

onMounted(async () => {
  try {
    const stats = await historyApi.getStatistics()
    statistics.value = stats
    const history = await historyApi.getHistory({ page: 1, per_page: 5 })
    recentAnalyses.value = history.records || []
  } catch (error) {
    console.error('Failed to load statistics:', error)
  } finally {
    loading.value = false
  }
})

function goToResult(record) {
  router.push(`/result/${record.id}`)
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function getGradeClass(grade) {
  const map = {
    excellent: 'grade-excellent',
    good: 'grade-good',
    fair: 'grade-fair',
    poor: 'grade-poor'
  }
  return map[grade] || ''
}

function getGradeLabel(grade) {
  return gradeLabel(grade)
}
</script>

<style lang="scss" scoped>
.dashboard {
  max-width: 1200px;
  margin: 0 auto;
}

// 统计卡片
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 32px;
}

.stat-card {
  background: #fff;
  border-radius: 12px;
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  }
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;

  .el-icon {
    font-size: 24px;
    color: #fff;
  }

  &.blue { background: linear-gradient(135deg, #409eff, #337ecc); }
  &.green { background: linear-gradient(135deg, #67c23a, #529b2e); }
  &.orange { background: linear-gradient(135deg, #e67e22, #d35400); }
  &.purple { background: linear-gradient(135deg, #9b59b6, #8e44ad); }
}

.stat-content {
  display: flex;
  flex-direction: column;
}

.stat-value {
  font-size: 28px;
  font-weight: 600;
  color: #2c3e50;
  line-height: 1.2;
}

.stat-label {
  font-size: 13px;
  color: #909399;
  margin-top: 4px;
}

// 区块
.section {
  background: #fff;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.section-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 20px;

  .section-header & {
    margin: 0;
  }
}

// 快速操作
.quick-actions {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
}

.action-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid transparent;

  &:hover {
    background: #fff;
    border-color: #e67e22;
    transform: translateX(4px);

    .action-arrow {
      opacity: 1;
      transform: translateX(0);
    }
  }
}

.action-icon {
  width: 44px;
  height: 44px;
  background: linear-gradient(135deg, #e67e22, #d35400);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;

  .el-icon {
    font-size: 22px;
    color: #fff;
  }
}

.action-content {
  flex: 1;

  h3 {
    font-size: 15px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 4px;
  }

  p {
    font-size: 13px;
    color: #909399;
    margin: 0;
  }
}

.action-arrow {
  font-size: 18px;
  color: #e67e22;
  opacity: 0;
  transform: translateX(-8px);
  transition: all 0.2s ease;
}

// 最近分析
.recent-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.recent-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  background: #f8f9fa;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: #f0f2f5;
    transform: translateX(4px);
  }
}

.recent-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.recent-name {
  font-size: 14px;
  font-weight: 500;
  color: #2c3e50;
}

.recent-meta {
  font-size: 12px;
  color: #909399;
}

.recent-score {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}

.score-value {
  font-size: 20px;
  font-weight: 600;
}

.score-label {
  font-size: 11px;
}

.grade-excellent { .score-value, .score-label { color: #67c23a; } }
.grade-good { .score-value, .score-label { color: #409eff; } }
.grade-fair { .score-value, .score-label { color: #e6a23c; } }
.grade-poor { .score-value, .score-label { color: #f56c6c; } }

// 空状态
.empty-state {
  padding: 40px 0;
}

.loading-placeholder {
  padding: 20px 0;
}

// 响应式
@media (max-width: 1024px) {
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .quick-actions {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 600px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
}
</style>
