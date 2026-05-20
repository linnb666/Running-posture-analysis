<template>
  <div class="result-page">
    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="10" animated />
    </div>

    <template v-else-if="result">
      <!-- 顶部概览 -->
      <div class="overview-section">
        <div class="score-card">
          <div class="score-ring" :class="gradeClass">
            <svg viewBox="0 0 120 120">
              <circle class="ring-bg" cx="60" cy="60" r="54" />
              <circle
                class="ring-progress"
                cx="60" cy="60" r="54"
                :stroke-dasharray="339.292"
                :stroke-dashoffset="339.292 * (1 - (result.total_score || 0) / 100)"
              />
            </svg>
            <div class="score-content">
              <span class="score-value">{{ result.total_score?.toFixed(1) || '--' }}</span>
              <span class="score-label">{{ gradeLabel }}</span>
            </div>
          </div>
        </div>

        <div class="info-card">
          <h2>{{ result.video_filename }}</h2>
          <div class="info-grid">
            <div class="info-item">
              <span class="info-label">视角类型</span>
              <span class="info-value">{{ viewLabel(result.view_type) }}</span>
            </div>
            <div class="info-item">
              <span class="info-label">分析时间</span>
              <span class="info-value">{{ formatDate(result.completed_at) }}</span>
            </div>
            <div class="info-item">
              <span class="info-label">模型版本</span>
              <span class="info-value">{{ result.model_version || 'N/A' }}</span>
            </div>
          </div>
        </div>
      </div>

      <div v-if="originalVideoSrc || poseVideoSrc || keyframeItems.length" class="media-section">
        <div v-if="originalVideoSrc" class="media-card">
          <h3 class="card-title">原始视频</h3>
          <video :src="originalVideoSrc" controls preload="metadata" />
        </div>

        <div v-if="poseVideoSrc" class="media-card">
          <h3 class="card-title">姿态识别视频</h3>
          <video :src="poseVideoSrc" controls preload="metadata" />
        </div>

        <div v-if="keyframeItems.length" class="media-card keyframes-card">
          <h3 class="card-title">关键帧分析</h3>
          <div class="keyframes-grid">
            <div v-for="item in keyframeItems" :key="item.name" class="keyframe-item">
              <img :src="item.src" :alt="item.name" />
              <span>{{ item.name }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 维度分析 -->
      <div class="dimension-section">
        <div class="radar-card">
          <h3 class="card-title">维度评分</h3>
          <div class="radar-chart" ref="radarChartRef"></div>
        </div>

        <div class="dimensions-card">
          <h3 class="card-title">详细分数</h3>
          <div class="dimension-list">
            <div
              v-for="(score, key) in dimensionScores"
              :key="key"
              class="dimension-item"
            >
              <div class="dimension-header">
                <span class="dimension-name">{{ getDimensionName(key) }}</span>
                <span class="dimension-score">{{ score?.toFixed(1) }}</span>
              </div>
              <el-progress
                :percentage="score || 0"
                :stroke-width="8"
                :show-text="false"
                :color="getScoreColor(score)"
              />
              <el-button class="detail-btn" link type="primary" @click="openDimensionDetail(key)">
                查看详细指标
              </el-button>
            </div>
          </div>
        </div>
      </div>

      <!-- 运动学曲线 -->
      <div v-if="!isFront" class="charts-section">
        <div class="chart-card" v-if="hasKneeAngles">
          <h3 class="card-title">膝关节角度变化</h3>
          <div class="chart-container" ref="kneeChartRef"></div>
        </div>

        <div class="chart-card" v-if="hasCadenceData">
          <h3 class="card-title">步频分析</h3>
          <div class="cadence-stats">
            <div class="stat-item">
              <span class="stat-value">{{ result.kinematic?.cadence?.cadence?.toFixed(0) || '--' }}</span>
              <span class="stat-label">步/分钟</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">{{ result.kinematic?.cadence?.step_count || '--' }}</span>
              <span class="stat-label">步数</span>
            </div>
            <div class="stat-item">
              <span class="stat-value" :class="getCadenceRatingClass">
                {{ getCadenceRating }}
              </span>
              <span class="stat-label">评级</span>
            </div>
          </div>
        </div>
      </div>

      <div v-else class="charts-section">
        <div class="chart-card">
          <h3 class="card-title">膝角可视化（正面）</h3>
          <div class="chart-container" ref="kneeChartRef"></div>
        </div>
        <div class="chart-card">
          <h3 class="card-title">下肢力线与肩部稳定</h3>
          <div class="chart-container" ref="frontMetricChartRef"></div>
        </div>
      </div>

      <!-- 分析结论 -->
      <div class="conclusion-section">
        <div class="conclusion-card strengths">
          <h3 class="card-title">
            <el-icon><CircleCheckFilled /></el-icon>
            优点
          </h3>
          <ul class="conclusion-list">
            <li v-for="(item, index) in result.strengths" :key="index">{{ item }}</li>
            <li v-if="!result.strengths?.length" class="empty">暂无数据</li>
          </ul>
        </div>

        <div class="conclusion-card weaknesses">
          <h3 class="card-title">
            <el-icon><WarningFilled /></el-icon>
            待改进
          </h3>
          <ul class="conclusion-list">
            <li v-for="(item, index) in result.weaknesses" :key="index">{{ item }}</li>
            <li v-if="!result.weaknesses?.length" class="empty">暂无数据</li>
          </ul>
        </div>

        <div class="conclusion-card suggestions">
          <h3 class="card-title">
            <el-icon><Opportunity /></el-icon>
            建议
          </h3>
          <ul class="conclusion-list">
            <li v-for="(item, index) in result.suggestions" :key="index">{{ item }}</li>
            <li v-if="!result.suggestions?.length" class="empty">暂无数据</li>
          </ul>
        </div>
      </div>

      <div class="ai-actions-section">
        <el-button type="primary" :loading="aiLoading" @click="generateAIReport">
          启动AI智能分析
        </el-button>
        <el-button v-if="result.ai_report" @click="toggleAIReport">
          {{ aiVisible ? '收起AI分析' : '查看AI分析' }}
        </el-button>
      </div>

      <!-- AI 报告 -->
      <div v-if="aiVisible && result.ai_report" class="ai-report-section">
        <div class="report-card">
          <h3 class="card-title">
            <el-icon><ChatDotRound /></el-icon>
            AI 分析报告
          </h3>
          <div class="report-content" v-html="formatReport(result.ai_report)"></div>
        </div>
      </div>

      <el-dialog v-model="dimensionDialogVisible" :title="dimensionDialogTitle" width="560px">
        <div class="dimension-detail-content">
          <div v-if="dimensionDetailRows.length === 0" class="empty-detail">暂无详细指标</div>
          <div v-else class="detail-list">
            <div v-for="row in dimensionDetailRows" :key="row.label" class="detail-row">
              <span class="label">{{ row.label }}</span>
              <span class="value">{{ row.value }}</span>
            </div>
          </div>
        </div>
      </el-dialog>
    </template>

    <div v-else class="error-state">
      <el-empty description="无法加载分析结果" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import * as echarts from 'echarts'
import { analysisApi } from '@/api/analysis'
import { ElMessage } from 'element-plus'
import { gradeLabel as toGradeLabel, viewLabel } from '@/utils/analysis'
import {
  CircleCheckFilled,
  WarningFilled,
  Opportunity,
  ChatDotRound
} from '@element-plus/icons-vue'

const props = defineProps({
  recordId: String
})

const route = useRoute()

const loading = ref(true)
const result = ref(null)
const aiLoading = ref(false)
const aiVisible = ref(false)
const dimensionDialogVisible = ref(false)
const currentDimensionKey = ref('')
const originalVideoSrc = ref('')
const poseVideoSrc = ref('')
const keyframeItems = ref([])
const radarChartRef = ref(null)
const kneeChartRef = ref(null)
const frontMetricChartRef = ref(null)

let radarChart = null
let kneeChart = null
let frontMetricChart = null

const gradeClass = computed(() => {
  const grade = result.value?.grade
  return {
    'grade-excellent': grade === 'excellent',
    'grade-good': grade === 'good',
    'grade-fair': grade === 'fair',
    'grade-poor': grade === 'poor'
  }
})

const gradeLabel = computed(() => {
  return toGradeLabel(result.value?.grade)
})

const isFront = computed(() => result.value?.view_type === 'front')

const dimensionScores = computed(() => {
  return result.value?.dimension_scores || {}
})

const dimensionDialogTitle = computed(() => `详细指标 - ${getDimensionName(currentDimensionKey.value || '')}`)

const dimensionDetailRows = computed(() => {
  const q = result.value?.quality_results || {}
  const d = q?.detailed_analysis || {}
  const push = (arr, label, value) => {
    if (value === undefined || value === null || value === '') return
    arr.push({ label, value: String(value) })
  }
  const rows = []

  if (!currentDimensionKey.value) return rows

  if (isFront.value) {
    if (currentDimensionKey.value === 'lower_limb_alignment') {
      const knee = d.knee_valgus || {}
      const hip = d.hip_drop || {}
      push(rows, '膝外翻角', knee.value)
      push(rows, '膝外翻评估', knee.assessment)
      push(rows, '髋部下沉角', hip.value)
      push(rows, '髋部下沉评估', hip.assessment)
    } else if (currentDimensionKey.value === 'lateral_stability') {
      const lat = d.lateral_stability || {}
      const shoulder = d.shoulder_tilt || {}
      push(rows, '横向稳定性', lat.value)
      push(rows, '横向稳定评估', lat.assessment)
      push(rows, '肩部倾斜', shoulder.value)
      push(rows, '肩部倾斜评估', shoulder.assessment)
    } else if (currentDimensionKey.value === 'efficiency') {
      const cadence = d.cadence || {}
      const amp = d.vertical_amplitude || {}
      const gct = d.ground_contact_time || {}
      push(rows, '步频', cadence.value)
      push(rows, '步频评估', cadence.assessment)
      push(rows, '垂直振幅', amp.value)
      push(rows, '垂直振幅评估', amp.assessment)
      push(rows, '触地时间', gct.value)
      push(rows, '触地时间评估', gct.assessment)
    }
  } else {
    if (currentDimensionKey.value === 'stability') {
      const knee = d.knee_angles || {}
      push(rows, '触地期膝角', knee.ground_contact)
      push(rows, '最大屈曲角', knee.max_flexion)
      push(rows, '膝角评估', knee.assessment)
    } else if (currentDimensionKey.value === 'efficiency') {
      const cadence = d.cadence || {}
      const amp = d.vertical_amplitude || {}
      push(rows, '步频', cadence.value)
      push(rows, '步频评估', cadence.assessment)
      push(rows, '垂直振幅', amp.value)
      push(rows, '垂直振幅评估', amp.assessment)
    } else if (currentDimensionKey.value === 'form') {
      const knee = d.knee_angles || {}
      push(rows, '触地期膝角', knee.ground_contact)
      push(rows, '最大屈曲角', knee.max_flexion)
      push(rows, '跑姿评估', knee.assessment)
    }
  }

  return rows
})

const hasKneeAngles = computed(() => {
  const angles = result.value?.kinematic?.angles
  return angles?.knee_left?.length > 0 || angles?.knee_right?.length > 0
})

const hasCadenceData = computed(() => {
  return result.value?.kinematic?.cadence?.cadence != null
})

const getCadenceRating = computed(() => {
  const raw = result.value?.kinematic?.cadence?.rating
  if (!raw) return '--'
  if (typeof raw === 'string') {
    const map = { excellent: '优秀', good: '良好', fair: '一般', poor: '较慢' }
    return map[raw] || raw
  }
  if (typeof raw === 'object') {
    return raw.description || raw.level || '--'
  }
  return '--'
})

const getCadenceRatingClass = computed(() => {
  const raw = result.value?.kinematic?.cadence?.rating
  const level = typeof raw === 'object' ? raw.level : raw
  return `rating-${level || 'unknown'}`
})

onMounted(async () => {
  const recordId = props.recordId || route.params.recordId
  if (!recordId) return

  try {
    const response = await analysisApi.getResult(recordId)
    result.value = response.result
    aiVisible.value = false
    await loadMediaAssets()

    await nextTick()
    initCharts()
  } catch (error) {
    console.error('Failed to load result:', error)
  } finally {
    loading.value = false
  }
})

onUnmounted(() => {
  cleanupMediaUrls()
  radarChart?.dispose()
  kneeChart?.dispose()
  frontMetricChart?.dispose()
})

function initCharts() {
  initRadarChart()
  initKneeChart()
  initFrontMetricChart()
}

function initRadarChart() {
  if (!radarChartRef.value || !Object.keys(dimensionScores.value || {}).length) return

  radarChart = echarts.init(radarChartRef.value)

  const dimensions = dimensionScores.value
  const indicator = Object.keys(dimensions).map(key => ({
    name: getDimensionName(key),
    max: 100
  }))
  const values = Object.values(dimensions)

  radarChart.setOption({
    radar: {
      indicator,
      radius: '65%',
      axisName: {
        color: '#606266',
        fontSize: 12
      },
      splitArea: {
        areaStyle: {
          color: ['rgba(230, 126, 34, 0.02)', 'rgba(230, 126, 34, 0.04)']
        }
      },
      axisLine: {
        lineStyle: { color: 'rgba(0, 0, 0, 0.1)' }
      },
      splitLine: {
        lineStyle: { color: 'rgba(0, 0, 0, 0.1)' }
      }
    },
    series: [{
      type: 'radar',
      data: [{
        value: values,
        name: '评分',
        areaStyle: {
          color: 'rgba(230, 126, 34, 0.3)'
        },
        lineStyle: {
          color: '#e67e22',
          width: 2
        },
        itemStyle: {
          color: '#e67e22'
        }
      }]
    }]
  })
}

function initKneeChart() {
  if (!kneeChartRef.value || !hasKneeAngles.value) return

  kneeChart = echarts.init(kneeChartRef.value)

  const angles = result.value.kinematic.angles
  const leftKnee = angles.knee_left || []
  const rightKnee = angles.knee_right || []
  const xData = leftKnee.map((_, i) => i)

  kneeChart.setOption({
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['左膝', '右膝'],
      bottom: 0
    },
    grid: {
      left: 50,
      right: 20,
      top: 20,
      bottom: 40
    },
    xAxis: {
      type: 'category',
      data: xData,
      name: '帧',
      axisLine: { lineStyle: { color: '#ddd' } },
      axisLabel: { color: '#909399' }
    },
    yAxis: {
      type: 'value',
      name: '角度 (°)',
      axisLine: { lineStyle: { color: '#ddd' } },
      axisLabel: { color: '#909399' },
      splitLine: { lineStyle: { color: '#f0f0f0' } }
    },
    series: [
      {
        name: '左膝',
        type: 'line',
        data: leftKnee,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#e67e22', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(230, 126, 34, 0.3)' },
            { offset: 1, color: 'rgba(230, 126, 34, 0.05)' }
          ])
        }
      },
      {
        name: '右膝',
        type: 'line',
        data: rightKnee,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#409eff', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(64, 158, 255, 0.3)' },
            { offset: 1, color: 'rgba(64, 158, 255, 0.05)' }
          ])
        }
      }
    ]
  })
}

function initFrontMetricChart() {
  if (!isFront.value || !frontMetricChartRef.value) return
  const lower = result.value?.kinematic?.lower_limb_alignment || {}
  const shoulder = result.value?.kinematic?.shoulder_analysis || {}
  const left = lower?.left_leg?.mean ?? 0
  const right = lower?.right_leg?.mean ?? 0
  const hipDrop = Math.abs(lower?.hip_drop?.mean ?? lower?.hip_drop?.drop_mean ?? 0)
  const shoulderTilt = Math.abs(shoulder?.tilt_mean ?? 0)

  frontMetricChart = echarts.init(frontMetricChartRef.value)
  frontMetricChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, top: 20, bottom: 30 },
    xAxis: {
      type: 'category',
      data: ['左膝偏移', '右膝偏移', '髋部下沉', '肩部倾斜'],
      axisLabel: { color: '#909399' }
    },
    yAxis: {
      type: 'value',
      name: '角度 (°)',
      axisLabel: { color: '#909399' },
      splitLine: { lineStyle: { color: '#f0f0f0' } }
    },
    series: [{
      type: 'bar',
      barWidth: 36,
      data: [
        { value: left, itemStyle: { color: '#e67e22' } },
        { value: right, itemStyle: { color: '#409eff' } },
        { value: hipDrop, itemStyle: { color: '#67c23a' } },
        { value: shoulderTilt, itemStyle: { color: '#9b59b6' } }
      ]
    }]
  })
}

function getDimensionName(key) {
  const map = {
    stability: '稳定性',
    efficiency: '效率',
    form: '跑姿',
    lower_limb_alignment: '下肢力线',
    lateral_stability: '横向稳定性'
  }
  return map[key] || key
}

function getScoreColor(score) {
  if (score >= 80) return '#67c23a'
  if (score >= 60) return '#409eff'
  if (score >= 40) return '#e6a23c'
  return '#f56c6c'
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleString('zh-CN')
}

function formatReport(text) {
  if (!text) return ''
  return text.replace(/\n/g, '<br>')
}

function openDimensionDetail(key) {
  currentDimensionKey.value = key
  dimensionDialogVisible.value = true
}

function cleanupMediaUrls() {
  for (const url of [originalVideoSrc.value, poseVideoSrc.value]) {
    if (url) URL.revokeObjectURL(url)
  }
  for (const item of keyframeItems.value) {
    if (item?.src) URL.revokeObjectURL(item.src)
  }
  originalVideoSrc.value = ''
  poseVideoSrc.value = ''
  keyframeItems.value = []
}

async function loadMediaAssets() {
  cleanupMediaUrls()
  const media = result.value?.media || {}
  if (!media) return

  if (media.original_video_url) {
    try {
      const blob = await analysisApi.getMediaBlob(media.original_video_url)
      if (blob instanceof Blob) {
        originalVideoSrc.value = URL.createObjectURL(blob)
      }
    } catch (_) {
      // ignore media fetch failure, keep main result visible
    }
  }
  if (media.pose_video_url) {
    try {
      const blob = await analysisApi.getMediaBlob(media.pose_video_url)
      if (blob instanceof Blob) {
        poseVideoSrc.value = URL.createObjectURL(blob)
      }
    } catch (_) {
      // ignore
    }
  }
  const frames = media.keyframes || []
  for (const item of frames) {
    try {
      const blob = await analysisApi.getMediaBlob(item.url)
      if (blob instanceof Blob) {
        keyframeItems.value.push({
          name: item.name,
          src: URL.createObjectURL(blob)
        })
      }
    } catch (_) {
      // ignore
    }
  }
}

function toggleAIReport() {
  aiVisible.value = !aiVisible.value
}

async function generateAIReport() {
  if (!result.value?.id || aiLoading.value) return
  aiLoading.value = true
  try {
    const resp = await analysisApi.generateAIReport(result.value.id)
    result.value.ai_report = resp?.ai_analysis || ''
    aiVisible.value = !!result.value.ai_report
    if (aiVisible.value) {
      ElMessage.success('AI分析已更新')
    } else {
      ElMessage.warning('未生成有效AI分析')
    }
  } catch (error) {
    ElMessage.error(error?.response?.data?.error || 'AI分析生成失败')
  } finally {
    aiLoading.value = false
  }
}
</script>

<style lang="scss" scoped>
.result-page {
  max-width: 1200px;
  margin: 0 auto;
}

// 概览区域
.overview-section {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.score-card {
  background: #fff;
  border-radius: 16px;
  padding: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.score-ring {
  position: relative;
  width: 140px;
  height: 140px;

  svg {
    transform: rotate(-90deg);
    width: 100%;
    height: 100%;
  }

  .ring-bg {
    fill: none;
    stroke: #f0f0f0;
    stroke-width: 8;
  }

  .ring-progress {
    fill: none;
    stroke: currentColor;
    stroke-width: 8;
    stroke-linecap: round;
    transition: stroke-dashoffset 1s ease;
  }

  &.grade-excellent { color: #67c23a; }
  &.grade-good { color: #409eff; }
  &.grade-fair { color: #e6a23c; }
  &.grade-poor { color: #f56c6c; }
}

.score-content {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.score-value {
  font-size: 36px;
  font-weight: 700;
  color: #2c3e50;
}

.score-label {
  font-size: 14px;
  color: #909399;
  margin-top: 4px;
}

.info-card {
  background: #fff;
  border-radius: 16px;
  padding: 32px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);

  h2 {
    font-size: 18px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 20px;
  }
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.info-label {
  font-size: 12px;
  color: #909399;
}

.info-value {
  font-size: 15px;
  font-weight: 500;
  color: #2c3e50;
}

.media-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.media-card {
  background: #fff;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);

  video {
    width: 100%;
    max-height: 320px;
    border-radius: 10px;
    background: #000;
  }
}

.keyframes-card {
  grid-column: 1 / -1;
}

.keyframes-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.keyframe-item {
  background: #f8f9fa;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #ebeef5;

  img {
    width: 100%;
    height: 170px;
    object-fit: cover;
    display: block;
  }

  span {
    display: block;
    padding: 8px 10px;
    font-size: 12px;
    color: #606266;
  }
}

// 维度分析
.dimension-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.radar-card,
.dimensions-card {
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 16px;
  display: flex;
  align-items: center;
  gap: 8px;

  .el-icon {
    font-size: 18px;
  }
}

.radar-chart {
  height: 280px;
}

.dimension-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.dimension-item {
  .dimension-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .dimension-name {
    font-size: 14px;
    color: #606266;
  }

  .dimension-score {
    font-size: 14px;
    font-weight: 600;
    color: #2c3e50;
  }
}

.detail-btn {
  margin-top: 6px;
  padding: 0;
}

.dimension-detail-content {
  .empty-detail {
    color: #909399;
    text-align: center;
    padding: 24px 0;
  }
}

.detail-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 12px;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  background: #fafbfc;

  .label {
    color: #606266;
    font-size: 13px;
  }

  .value {
    color: #2c3e50;
    font-size: 13px;
    font-weight: 500;
    text-align: right;
  }
}

// 图表区域
.charts-section {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.chart-card {
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.chart-container {
  height: 280px;
}

.cadence-stats {
  display: flex;
  justify-content: space-around;
  padding: 40px 0;
}

.stat-item {
  text-align: center;

  .stat-value {
    display: block;
    font-size: 32px;
    font-weight: 700;
    color: #2c3e50;

    &.rating-excellent { color: #67c23a; }
    &.rating-good { color: #409eff; }
    &.rating-fair { color: #e6a23c; }
    &.rating-poor { color: #f56c6c; }
  }

  .stat-label {
    font-size: 13px;
    color: #909399;
    margin-top: 4px;
  }
}

// 结论区域
.conclusion-section {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 24px;
  margin-bottom: 24px;
}

.ai-actions-section {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.conclusion-card {
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);

  &.strengths .card-title .el-icon { color: #67c23a; }
  &.weaknesses .card-title .el-icon { color: #e6a23c; }
  &.suggestions .card-title .el-icon { color: #409eff; }
}

.conclusion-list {
  list-style: none;
  padding: 0;
  margin: 0;

  li {
    padding: 10px 0;
    font-size: 14px;
    color: #606266;
    border-bottom: 1px solid #f0f0f0;

    &:last-child {
      border-bottom: none;
    }

    &.empty {
      color: #c0c4cc;
      font-style: italic;
    }
  }
}

// AI 报告
.ai-report-section {
  margin-bottom: 24px;
}

.report-card {
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);

  .card-title .el-icon {
    color: #9b59b6;
  }
}

.report-content {
  font-size: 14px;
  line-height: 1.8;
  color: #606266;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
}

// 状态
.loading-state,
.error-state {
  padding: 60px 0;
}

// 响应式
@media (max-width: 1024px) {
  .overview-section {
    grid-template-columns: 1fr;
  }

  .dimension-section,
  .charts-section {
    grid-template-columns: 1fr;
  }

  .media-section {
    grid-template-columns: 1fr;
  }

  .keyframes-grid {
    grid-template-columns: 1fr;
  }

  .conclusion-section {
    grid-template-columns: 1fr;
  }
}
</style>
