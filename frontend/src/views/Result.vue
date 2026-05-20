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
          <div class="info-card-head">
            <h2>{{ result.video_filename }}</h2>
            <el-button class="pdf-export-btn" type="primary" plain :loading="pdfExporting" @click="exportPdfReport">
              下载 PDF 报告
            </el-button>
          </div>
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
          <video :key="originalVideoSrc" :src="originalVideoSrc" controls preload="metadata" />
        </div>

        <div v-if="poseVideoSrc" class="media-card">
          <h3 class="card-title">姿态识别视频</h3>
          <video :key="poseVideoSrc" :src="poseVideoSrc" controls preload="metadata" @error="handlePoseVideoError" />
        </div>

        <div v-if="keyframeItems.length" class="media-card keyframes-card">
          <h3 class="card-title">关键帧分析</h3>
          <div class="keyframes-grid">
            <div v-for="item in keyframeItems" :key="item.name" class="keyframe-item">
              <img :src="item.src" :alt="item.name" />
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
          <div class="dimension-summary">
            <span>综合总分（加权）</span>
            <strong>{{ weightedTotalScore }}</strong>
          </div>
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
          <h3 class="card-title">膝关节偏移角度可视化（正面）</h3>
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
        <el-button :loading="localLoading" @click="toggleLocalReport">
          {{ localVisible ? '收起本地分析报告' : (localReportText ? '展开本地分析报告' : '生成并展开本地分析报告') }}
        </el-button>
        <el-button type="primary" :loading="aiLoading" @click="generateAIReport">
          启动AI智能分析
        </el-button>
        <el-button v-if="result.ai_report || aiErrorText" @click="openAIReportDialog">
          查看AI分析窗口
        </el-button>
      </div>

      <div class="floating-notes-widget" :class="{ open: notesPanelOpen, dirty: notesChanged, filled: hasManualNotes }">
        <button type="button" class="notes-fab" @click="notesPanelOpen = !notesPanelOpen">
          <span class="notes-fab-dot"></span>
          <span>添加备注</span>
        </button>
        <transition name="notes-pop">
          <div v-if="notesPanelOpen" class="notes-panel">
            <div class="notes-panel-head">
              <div>
                <h3>添加备注</h3>
                <p>记录教练反馈、人工观察或本次复盘，下载 PDF 报告时会同步写入。</p>
              </div>
              <button type="button" class="notes-close" @click="notesPanelOpen = false">收起</button>
            </div>
            <span class="notes-meta">{{ notesMetaText }}</span>
            <el-input
              v-model="manualNotesDraft"
              class="notes-input"
              type="textarea"
              :rows="6"
              maxlength="5000"
              show-word-limit
              resize="vertical"
              placeholder="补充感受或教练反馈..."
            />
            <div class="notes-actions">
              <span>{{ notesChanged ? '你有未保存的修改' : '保存后会同步写入当前分析记录，并用于下载 PDF 报告。' }}</span>
              <el-button type="primary" plain :loading="notesSaving" :disabled="!canSaveNotes" @click="saveManualNotes">
                保存备注
              </el-button>
            </div>
          </div>
        </transition>
      </div>

      <div v-if="localVisible && localReportText" class="local-report-section">
        <div class="report-card">
          <h3 class="card-title">
            <el-icon><Document /></el-icon>
            本地分析报告（规则引擎）
          </h3>
          <div class="report-content prose-report" v-html="formatReport(localReportText)"></div>
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

      <el-dialog v-model="aiDialogVisible" class="ai-dialog-shell" title="AI 智能分析" width="820px">
        <div class="ai-dialog-content">
          <div class="ai-dialog-hero">
            <div>
              <p class="eyebrow">智能解读窗口</p>
              <h3>{{ aiDialogHeading }}</h3>
              <p>{{ aiDialogSubtext }}</p>
            </div>
            <span class="status-pill" :class="aiStatusClass">{{ aiStatusLabel }}</span>
          </div>

          <div v-if="aiLoading" class="ai-progress-block">
            <div class="progress-shell">
              <div class="progress-top">
                <span>{{ aiStatusText }}</span>
                <strong>{{ Math.round(aiProgress) }}%</strong>
              </div>
              <el-progress :percentage="Math.round(aiProgress)" :stroke-width="12" :show-text="false" />
            </div>
            <div class="progress-steps">
              <div class="step-item" :class="{ active: aiProgress >= 8 }">解析指标</div>
              <div class="step-item" :class="{ active: aiProgress >= 38 }">组织上下文</div>
              <div class="step-item" :class="{ active: aiProgress >= 68 }">生成建议</div>
              <div class="step-item" :class="{ active: aiProgress >= 92 }">整理报告</div>
            </div>
            <p class="ai-progress-sub">窗口会保留历史生成结果，后续可再次打开查看。</p>
          </div>

          <div v-else-if="aiErrorText" class="ai-error-block">
            <el-alert :title="aiErrorText" type="error" :closable="false" show-icon />
          </div>

          <div v-else-if="result.ai_report" class="ai-dialog-report">
            <div class="report-toolbar">
              <span>报告来源：AI 智能分析</span>
              <span v-if="aiGeneratedAt">最近更新：{{ aiGeneratedAt }}</span>
            </div>
            <div class="report-content prose-report" v-html="formatReport(result.ai_report)"></div>
          </div>

          <el-empty v-else description="暂无AI报告，请点击“启动AI智能分析”生成" />
        </div>
        <template #footer>
          <el-button @click="aiDialogVisible = false">关闭</el-button>
          <el-button type="primary" :loading="aiLoading" @click="generateAIReport">
            重新生成
          </el-button>
        </template>
      </el-dialog>
    </template>

    <div v-else class="error-state">
      <el-empty description="无法加载分析结果" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import * as echarts from 'echarts'
import { analysisApi } from '@/api/analysis'
import { useUserStore } from '@/stores/user'
import { ElMessage } from 'element-plus'
import { gradeLabel as toGradeLabel, viewLabel } from '@/utils/analysis'
import {
  CircleCheckFilled,
  WarningFilled,
  Opportunity,
  Document,
  EditPen
} from '@element-plus/icons-vue'

const props = defineProps({
  recordId: String
})

const route = useRoute()
const userStore = useUserStore()

const loading = ref(true)
const result = ref(null)
const aiLoading = ref(false)
const aiDialogVisible = ref(false)
const aiProgress = ref(0)
const aiStatusText = ref('')
const aiErrorText = ref('')
const aiGeneratedAt = ref('')
const localVisible = ref(false)
const localLoading = ref(false)
const localReportText = ref('')
const dimensionDialogVisible = ref(false)
const currentDimensionKey = ref('')
const originalVideoSrc = ref('')
const poseVideoSrc = ref('')
const keyframeItems = ref([])
const poseVideoFallbackTried = ref(false)
const manualNotesDraft = ref('')
const lastSavedNotes = ref('')
const notesSaving = ref(false)
const notesSavedAt = ref('')
const notesPanelOpen = ref(false)
const pdfExporting = ref(false)
const radarChartRef = ref(null)
const kneeChartRef = ref(null)
const frontMetricChartRef = ref(null)

let radarChart = null
let kneeChart = null
let frontMetricChart = null
let aiProgressTimer = null

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

const weaknessItems = computed(() => {
  const current = result.value?.weaknesses
  if (Array.isArray(current) && current.length && !current.every((item) => ['?????', '????', '?????', '????', '?????', '???', '??????'].includes(item))) {
    return current.filter((item) => item && item !== '??????')
  }

  const detailed = result.value?.quality_results?.detailed_analysis || {}
  const metricMap = {
    vertical_amplitude: '????',
    knee_angles: '?????',
    cadence: '??',
    body_lean: '????',
    knee_valgus: '?????',
    hip_drop: '????',
    lateral_stability: '?????',
    shoulder_tilt: '????',
    ground_contact_time: '????'
  }
  const weakLevels = new Set(['fair', 'poor', 'acceptable', 'upright', 'excessive', 'needs_improvement'])
  const items = Object.entries(detailed)
    .filter(([, value]) => value && typeof value === 'object' && weakLevels.has(String(value.level || '').toLowerCase()))
    .map(([key]) => metricMap[key])
    .filter(Boolean)
  return [...new Set(items)]
})

const suggestionItems = computed(() => {
  const items = Array.isArray(result.value?.suggestions) ? result.value.suggestions : []
  return items.filter((item) => item && item !== '?????????????????????' && !String(item).startsWith('?? ???'))
})

const weightedTotalScore = computed(() => {
  const total = Number(result.value?.total_score)
  if (!Number.isFinite(total)) return '--'
  return total.toFixed(1)
})

const dimensionDialogTitle = computed(() => `详细指标 - ${getDimensionName(currentDimensionKey.value || '')}`)

const hasManualNotes = computed(() => !!(lastSavedNotes.value || '').trim())

const notesChanged = computed(() => manualNotesDraft.value !== lastSavedNotes.value)

const canSaveNotes = computed(() => !notesSaving.value && notesChanged.value)

const notesMetaText = computed(() => {
  if (notesSaving.value) return '备注保存中...'
  if (notesChanged.value) return '你有未保存的修改'
  if (notesSavedAt.value) return `最近保存：${notesSavedAt.value}`
  return hasManualNotes.value
    ? '已存在人工备注'
    : '尚未添加人工备注'
})

const aiStatusLabel = computed(() => {
  if (aiLoading.value) return '生成中'
  if (aiErrorText.value) return '失败'
  if (result.value?.ai_report) return '已完成'
  return '未生成'
})

const aiStatusClass = computed(() => ({
  loading: aiLoading.value,
  error: !!aiErrorText.value,
  success: !aiLoading.value && !aiErrorText.value && !!result.value?.ai_report
}))

const aiDialogHeading = computed(() => {
  if (aiLoading.value) return 'AI 正在整理本次跑姿解读'
  if (aiErrorText.value) return '本次 AI 分析未成功返回'
  if (result.value?.ai_report) return 'AI 跑步动作分析报告'
  return '尚未生成 AI 报告'
})

const aiDialogSubtext = computed(() => {
  if (aiLoading.value) return '当前使用结构化指标生成文字解读，请稍候。'
  if (aiErrorText.value) return '你可以稍后重试，失败不会影响本地分析结果。'
  if (result.value?.ai_report) return '该窗口会保留最新一次 AI 分析结果，便于后续复查。'
  return '点击下方按钮后，系统会基于当前分析记录生成专业文字报告。'
})
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
      const st = result.value?.kinematic?.stability || {}
      const lean = result.value?.kinematic?.body_lean || {}
      push(rows, '整体稳定性', st.overall)
      push(rows, '躯干稳定性', st.trunk)
      push(rows, '头部稳定性', st.head)
      push(rows, '稳定性评级', st?.rating?.description || st?.rating?.level)
      push(rows, '躯干前倾均值', lean.mean_lean ?? lean.mean)
      push(rows, '躯干前倾波动', lean.std_lean ?? lean.std)
    } else if (currentDimensionKey.value === 'efficiency') {
      const cadence = d.cadence || {}
      const amp = d.vertical_amplitude || {}
      const gct = d.ground_contact_time || {}
      const gaitCycle = result.value?.kinematic?.gait_cycle || {}
      const phaseDuration = gaitCycle?.phase_duration_ms || {}
      const phaseDistribution = gaitCycle?.phase_distribution || {}
      const gctFromKinematicMs = phaseDuration?.ground_contact
      const gctRatio = phaseDistribution?.ground_contact
      let gctValue = gct.value
      if (!gctValue && gctFromKinematicMs != null) {
        gctValue = `${Number(gctFromKinematicMs).toFixed(1)} ms`
      }
      if (!gctValue && gctRatio != null) {
        gctValue = `${(Number(gctRatio) * 100).toFixed(1)}%`
      }
      push(rows, '步频', cadence.value)
      push(rows, '步频评估', cadence.assessment)
      push(rows, '垂直振幅', amp.value)
      push(rows, '垂直振幅评估', amp.assessment)
      push(rows, '触地时间', gctValue)
      push(rows, '触地时间评估', gct.assessment)
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

async function loadResultById(recordId) {
  if (!recordId) return
  try {
    loading.value = true
    cleanupMediaUrls()
    const response = await analysisApi.getResult(recordId)
    result.value = response.record || response.result || null
    manualNotesDraft.value = result.value?.manual_notes || ''
    lastSavedNotes.value = manualNotesDraft.value
    notesSavedAt.value = result.value?.manual_notes_updated_at ? formatDate(result.value.manual_notes_updated_at) : ''
    notesPanelOpen.value = false
    aiDialogVisible.value = false
    aiErrorText.value = ''
    aiProgress.value = 0
    aiStatusText.value = ''
    localVisible.value = false
    localReportText.value = ''
    loading.value = false
    await nextTick()
    initCharts()
    void loadMediaAssets()
  } catch (error) {
    console.error('Failed to load result:', error)
    loading.value = false
  }
}

onMounted(async () => {
  await loadResultById(props.recordId || route.params.recordId)
})

watch(
  () => props.recordId || route.params.recordId,
  async (nextId, prevId) => {
    if (!nextId || nextId === prevId) return
    await loadResultById(nextId)
  }
)

onUnmounted(() => {
  cleanupMediaUrls()
  radarChart?.dispose()
  kneeChart?.dispose()
  frontMetricChart?.dispose()
  stopAIProgress()
})

function initCharts() {
  try {
    initRadarChart()
  } catch (_) {}
  try {
    initKneeChart()
  } catch (_) {}
  try {
    initFrontMetricChart()
  } catch (_) {}
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
  if (!kneeChartRef.value) return

  const angles = result.value?.kinematic?.angles || {}
  const frontChart = result.value?.kinematic?.lower_limb_alignment?.chart_data || {}
  const leftKnee = isFront.value
    ? (frontChart.left || angles.knee_left || [])
    : (angles.knee_left || [])
  const rightKnee = isFront.value
    ? (frontChart.right || angles.knee_right || [])
    : (angles.knee_right || [])
  if (!leftKnee.length && !rightKnee.length) return

  kneeChart = echarts.init(kneeChartRef.value)
  const hasTimePct = isFront.value &&
    Array.isArray(frontChart.time_pct) &&
    frontChart.time_pct.length === leftKnee.length
  const xData = hasTimePct
    ? frontChart.time_pct.map((x) => Number(x).toFixed(1))
    : leftKnee.map((_, i) => i + 1)
  const leftName = isFront.value ? '左膝偏移角' : '左膝'
  const rightName = isFront.value ? '右膝偏移角' : '右膝'
  const yName = isFront.value ? '偏移角 (°)' : '角度 (°)'

  kneeChart.setOption({
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: [leftName, rightName],
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
      name: hasTimePct ? '步态周期(%)' : '帧',
      axisLine: { lineStyle: { color: '#ddd' } },
      axisLabel: { color: '#909399' }
    },
    yAxis: {
      type: 'value',
      name: yName,
      axisLine: { lineStyle: { color: '#ddd' } },
      axisLabel: { color: '#909399' },
      splitLine: { lineStyle: { color: '#f0f0f0' } }
    },
    series: [
      {
        name: leftName,
        type: 'line',
        data: leftKnee,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#e67e22', width: 2 },
        markLine: isFront.value
          ? { symbol: 'none', lineStyle: { color: '#c0c4cc', type: 'dashed' }, data: [{ yAxis: 0 }] }
          : undefined,
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(230, 126, 34, 0.3)' },
            { offset: 1, color: 'rgba(230, 126, 34, 0.05)' }
          ])
        }
      },
      {
        name: rightName,
        type: 'line',
        data: rightKnee,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#409eff', width: 2 },
        markLine: isFront.value
          ? { symbol: 'none', lineStyle: { color: '#c0c4cc', type: 'dashed' }, data: [{ yAxis: 0 }] }
          : undefined,
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

function parseUtcDate(dateStr) {
  if (!dateStr) return null
  if (dateStr instanceof Date) return dateStr
  const raw = String(dateStr).trim()
  if (!raw) return null
  const normalized = raw.includes('T') ? raw : raw.replace(' ', 'T')
  const hasTimezone = /([zZ]|[+-]\d{2}:\d{2})$/.test(normalized)
  const parsed = new Date(hasTimezone ? normalized : `${normalized}Z`)
  if (!Number.isNaN(parsed.getTime())) return parsed
  const fallback = new Date(raw)
  return Number.isNaN(fallback.getTime()) ? null : fallback
}

function formatDate(dateStr) {
  const date = parseUtcDate(dateStr)
  if (!date) return ''
  return new Intl.DateTimeFormat('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  }).format(date).replace(/\//g, '-')
}

function escapeHtml(text = '') {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function formatInline(text = '') {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
}

function renderTable(lines) {
  const rows = lines
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => line.replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim()))
    .filter(cells => cells.length)
  if (rows.length < 2) return `<p>${formatInline(lines.join(' '))}</p>`
  const header = rows[0]
  const bodyRows = rows.slice(1).filter(cells => !cells.every(cell => /^:?-{2,}:?$/.test(cell.replace(/\s/g, ''))))
  const thead = `<thead><tr>${header.map(cell => `<th>${formatInline(cell)}</th>`).join('')}</tr></thead>`
  const tbody = `<tbody>${bodyRows.map(row => `<tr>${row.map(cell => `<td>${formatInline(cell)}</td>`).join('')}</tr>`).join('')}</tbody>`
  return `<div class="report-table-wrap"><table class="report-table">${thead}${tbody}</table></div>`
}

function formatReport(text) {
  if (!text) return ''
  const normalized = String(text).replace(/\r\n/g, '\n').trim()
  if (!normalized) return ''
  const blocks = normalized.split(/\n\s*\n/)
  return blocks.map((block) => {
    const lines = block.split('\n').filter(line => line.trim().length > 0)
    if (!lines.length) return ''
    if (lines.every(line => line.trim().startsWith('|'))) {
      return renderTable(lines)
    }
    if (lines.length === 1 && /^---+$/.test(lines[0].trim())) {
      return '<hr>'
    }
    if (lines.every(line => /^-\s+/.test(line.trim()))) {
      return `<ul>${lines.map(line => `<li>${formatInline(line.trim().replace(/^-\s+/, ''))}</li>`).join('')}</ul>`
    }
    if (lines.every(line => /^\d+\.\s+/.test(line.trim()))) {
      return `<ol>${lines.map(line => `<li>${formatInline(line.trim().replace(/^\d+\.\s+/, ''))}</li>`).join('')}</ol>`
    }
    if (lines.every(line => /^>\s?/.test(line.trim()))) {
      return `<blockquote>${lines.map(line => formatInline(line.trim().replace(/^>\s?/, ''))).join('<br>')}</blockquote>`
    }
    const first = lines[0].trim()
    if (/^###\s+/.test(first)) {
      return `<h4>${formatInline(first.replace(/^###\s+/, ''))}</h4>${lines.slice(1).length ? `<p>${lines.slice(1).map(formatInline).join('<br>')}</p>` : ''}`
    }
    if (/^##\s+/.test(first)) {
      return `<h3>${formatInline(first.replace(/^##\s+/, ''))}</h3>${lines.slice(1).length ? `<p>${lines.slice(1).map(formatInline).join('<br>')}</p>` : ''}`
    }
    if (/^#\s+/.test(first)) {
      return `<h2>${formatInline(first.replace(/^#\s+/, ''))}</h2>${lines.slice(1).length ? `<p>${lines.slice(1).map(formatInline).join('<br>')}</p>` : ''}`
    }
    return `<p>${lines.map(formatInline).join('<br>')}</p>`
  }).join('')
}

function openDimensionDetail(key) {
  currentDimensionKey.value = key
  dimensionDialogVisible.value = true
}

function cleanupMediaUrls() {
  for (const url of [originalVideoSrc.value, poseVideoSrc.value]) {
    if (url && url.startsWith('blob:')) URL.revokeObjectURL(url)
  }
  for (const item of keyframeItems.value) {
    if (item?.src && item.src.startsWith('blob:')) URL.revokeObjectURL(item.src)
  }
  originalVideoSrc.value = ''
  poseVideoSrc.value = ''
  keyframeItems.value = []
}

function withAccessToken(url, bustKey = '') {
  if (!url) return ''
  let nextUrl = url
  const token = userStore.token
  if (token) {
    const sep = nextUrl.includes('?') ? '&' : '?'
    nextUrl = `${nextUrl}${sep}access_token=${encodeURIComponent(token)}`
  }
  if (bustKey) {
    const sep = nextUrl.includes('?') ? '&' : '?'
    nextUrl = `${nextUrl}${sep}_m=${encodeURIComponent(bustKey)}`
  }
  return nextUrl
}

function loadMediaAssets() {
  cleanupMediaUrls()
  poseVideoFallbackTried.value = false
  const media = result.value?.media || {}
  if (!media) return
  const mediaBust = `${result.value?.id || 'rid'}-${Date.now()}`

  if (media.original_video_url) {
    originalVideoSrc.value = withAccessToken(media.original_video_url, mediaBust)
  }
  if (media.pose_video_url) {
    poseVideoSrc.value = withAccessToken(media.pose_video_url, mediaBust)
  }

  keyframeItems.value = (media.keyframes || [])
    .map((item) => ({
      name: item.name,
      src: withAccessToken(item.url, mediaBust)
    }))
    .filter((item) => !!item.src)
}

function handlePoseVideoError() {
  if (poseVideoFallbackTried.value) return
  const media = result.value?.media || {}
  const basePoseUrl = media.pose_video_url || ''
  if (!basePoseUrl) return

  poseVideoFallbackTried.value = true
  // Fallback for legacy records / codec mismatch: switch web overlay to original overlay.
  if (basePoseUrl.endsWith('/pose_overlay_web.mp4')) {
    poseVideoSrc.value = withAccessToken(
      basePoseUrl.replace('/pose_overlay_web.mp4', '/pose_overlay.mp4'),
      `${result.value?.id || 'rid'}-${Date.now()}`
    )
  }
}

function openAIReportDialog() {
  aiDialogVisible.value = true
  if (result.value?.ai_report && !aiLoading.value && !aiErrorText.value) {
    aiProgress.value = 100
    aiStatusText.value = 'AI分析完成'
  }
}

function stopAIProgress() {
  if (aiProgressTimer) {
    clearInterval(aiProgressTimer)
    aiProgressTimer = null
  }
}

function startAIProgress() {
  stopAIProgress()
  aiProgressTimer = setInterval(() => {
    if (!aiLoading.value) return
    const p = aiProgress.value
    const inc = p < 45
      ? (Math.random() * 5 + 2)
      : p < 75
        ? (Math.random() * 2.6 + 0.8)
        : (Math.random() * 1.2 + 0.2)
    aiProgress.value = Math.min(93, p + inc)
    if (aiProgress.value < 20) aiStatusText.value = '正在解析动作指标...'
    else if (aiProgress.value < 45) aiStatusText.value = '正在构建分析上下文...'
    else if (aiProgress.value < 70) aiStatusText.value = '正在生成专业建议...'
    else aiStatusText.value = '正在整理报告文本...'
  }, 650)
}

async function toggleLocalReport() {
  if (localVisible.value) {
    localVisible.value = false
    return
  }
  if (localReportText.value) {
    localVisible.value = true
    return
  }
  if (!result.value?.id || localLoading.value) return

  localLoading.value = true
  try {
    const resp = await analysisApi.generateLocalReport(result.value.id)
    const text = (resp?.local_analysis || '').trim()
    if (!text) {
      ElMessage.warning('未生成有效本地报告')
      return
    }
    localReportText.value = text
    localVisible.value = true
    ElMessage.success('本地分析报告已生成')
  } catch (error) {
    ElMessage.error(error?.response?.data?.error || '本地报告生成失败')
  } finally {
    localLoading.value = false
  }
}

async function generateAIReport() {
  if (!result.value?.id || aiLoading.value) return
  aiDialogVisible.value = true
  aiErrorText.value = ''
  aiProgress.value = 4
  aiStatusText.value = '正在提交分析请求...'
  aiLoading.value = true
  startAIProgress()
  try {
    const resp = await analysisApi.generateAIReport(result.value.id)
    let aiText = (resp?.ai_analysis || '').trim()
    if (!aiText) {
      const latest = await analysisApi.getResult(result.value.id)
      aiText = (latest?.record?.ai_report || latest?.record?.ai_analysis || '').trim()
    }
    if (!aiText) {
      throw new Error('AI分析未返回有效内容')
    }
    result.value.ai_report = aiText
    result.value.ai_analysis = aiText
    aiGeneratedAt.value = formatDate(new Date().toISOString())
    aiProgress.value = 100
    aiStatusText.value = 'AI分析完成'
    ElMessage.success('AI分析已更新')
  } catch (error) {
    const msg = error?.response?.data?.error || error?.message || 'AI分析生成失败'
    aiErrorText.value = msg
    aiStatusText.value = 'AI分析失败'
    aiProgress.value = 0
    ElMessage.error(msg)
  } finally {
    aiLoading.value = false
    stopAIProgress()
  }
}



async function saveManualNotes() {
  if (!result.value?.id || notesSaving.value || !notesChanged.value) return
  notesSaving.value = true
  try {
    const resp = await analysisApi.saveManualNotes(result.value.id, manualNotesDraft.value)
    result.value.manual_notes = resp?.manual_notes || ''
    result.value.manual_notes_updated_at = resp?.manual_notes_updated_at || resp?.updated_at || null
    lastSavedNotes.value = result.value.manual_notes
    notesSavedAt.value = formatDate(result.value.manual_notes_updated_at || new Date().toISOString())
    notesPanelOpen.value = false
    ElMessage.success('人工备注已保存')
  } catch (error) {
    ElMessage.error(error?.response?.data?.error || '备注保存失败')
  } finally {
    notesSaving.value = false
  }
}

async function exportPdfReport() {
  if (!result.value?.id || pdfExporting.value) return
  pdfExporting.value = true
  try {
    const { blob, filename } = await analysisApi.downloadPdfReport(result.value.id)
    const downloadUrl = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = downloadUrl
    link.download = filename || `record_${result.value.id}_analysis_report.pdf`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000)
    ElMessage.success('PDF 报告已开始下载')
  } catch (error) {
    ElMessage.error(error?.response?.data?.error || 'PDF 导出失败')
  } finally {
    pdfExporting.value = false
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
    margin: 0;
  }
}

.info-card-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 20px;
}

.pdf-export-btn {
  min-width: 132px;
  border-color: rgba(154, 93, 26, 0.18);
  color: #9a5d1a;
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
  background: #111;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  padding: 8px;
  display: flex;
  align-items: center;
  justify-content: center;

  img {
    width: 100%;
    aspect-ratio: 16 / 9;
    height: auto;
    object-fit: contain;
    display: block;
    background: #000;
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

.dimension-summary {
  margin: -2px 0 14px;
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  color: #909399;
  font-size: 13px;

  strong {
    color: #2c3e50;
    font-size: 20px;
    line-height: 1;
  }
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
  margin-top: 8px;
  margin-bottom: 24px;
}

.ai-actions-section {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 8px;
  margin-bottom: 24px;
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

.local-report-section {
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

.local-report-section .report-card .card-title .el-icon {
  color: #16a085;
}

.report-content {
  font-size: 14px;
  line-height: 1.8;
  color: #606266;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
  max-height: 56vh;
  overflow: auto;
}

.ai-dialog-content {
  min-height: 180px;
}

.ai-progress-block {
  padding: 8px 4px;
}

.ai-progress-text {
  margin: 14px 0 4px;
  font-size: 14px;
  color: #2c3e50;
  font-weight: 500;
}

.ai-progress-sub {
  margin: 0;
  font-size: 13px;
  color: #909399;
}

.ai-error-block {
  padding: 8px 0;
}

.ai-dialog-report .report-content {
  max-height: 52vh;
}

// 状态
.floating-notes-widget {
  position: fixed;
  right: 26px;
  bottom: 26px;
  z-index: 30;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 12px;
}

.notes-fab {
  border: 0;
  border-radius: 999px;
  padding: 12px 18px;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  background: linear-gradient(135deg, rgba(154, 93, 26, 0.96), rgba(191, 122, 44, 0.96));
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  box-shadow: 0 18px 35px rgba(83, 47, 18, 0.22);
}

.notes-fab-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 0 0 6px rgba(255, 255, 255, 0.16);
}

.floating-notes-widget.filled .notes-fab {
  background: linear-gradient(135deg, rgba(131, 87, 34, 0.96), rgba(166, 111, 46, 0.96));
}

.floating-notes-widget.dirty .notes-fab {
  background: linear-gradient(135deg, rgba(194, 103, 61, 0.98), rgba(223, 132, 72, 0.96));
}

.notes-panel {
  width: min(360px, calc(100vw - 32px));
  border-radius: 24px;
  background: rgba(255, 252, 246, 0.98);
  border: 1px solid rgba(181, 145, 104, 0.24);
  box-shadow: 0 24px 50px rgba(54, 38, 20, 0.18);
  padding: 18px 18px 16px;
  backdrop-filter: blur(14px);
}

.notes-panel-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;

  h3 {
    margin: 0;
    font-size: 16px;
    color: #2c3e50;
  }

  p {
    margin: 8px 0 0;
    color: #8b8f97;
    font-size: 12px;
    line-height: 1.7;
  }
}

.notes-close {
  border: 0;
  background: rgba(154, 93, 26, 0.08);
  color: #9a5d1a;
  border-radius: 999px;
  padding: 8px 12px;
  font-size: 12px;
  cursor: pointer;
}

.notes-meta {
  display: inline-flex;
  margin: 12px 0 10px;
  font-size: 12px;
  color: #9a5d1a;
}

.notes-input {
  :deep(.el-textarea__inner) {
    min-height: 148px !important;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.88);
    border-color: rgba(191, 145, 80, 0.18);
  }
}

.notes-actions {
  margin-top: 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;

  span {
    color: #8b8f97;
    font-size: 12px;
    line-height: 1.6;
  }
}

.notes-pop-enter-active,
.notes-pop-leave-active {
  transition: opacity 0.22s ease, transform 0.22s ease;
}

.notes-pop-enter-from,
.notes-pop-leave-to {
  opacity: 0;
  transform: translateY(8px) scale(0.98);
}

.prose-report {
  :deep(h2),
  :deep(h3),
  :deep(h4) {
    margin: 0 0 12px;
    color: #243043;
  }

  :deep(h2) {
    font-size: 22px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(154, 93, 26, 0.14);
  }

  :deep(h3) {
    font-size: 17px;
    margin-top: 18px;
  }

  :deep(h4) {
    font-size: 15px;
    margin-top: 12px;
  }

  :deep(p) {
    margin: 0 0 12px;
  }

  :deep(ul),
  :deep(ol) {
    margin: 0 0 12px 18px;
    padding: 0;
  }

  :deep(li) {
    margin-bottom: 8px;
  }

  :deep(blockquote) {
    margin: 0 0 12px;
    padding: 10px 14px;
    background: rgba(154, 93, 26, 0.06);
    border-left: 3px solid rgba(154, 93, 26, 0.35);
    border-radius: 0 10px 10px 0;
  }

  :deep(code) {
    padding: 2px 6px;
    background: #eef1f5;
    border-radius: 6px;
    font-family: Consolas, 'Courier New', monospace;
    font-size: 12px;
  }

  :deep(hr) {
    border: 0;
    border-top: 1px solid #e5e7eb;
    margin: 16px 0;
  }
}

.report-table-wrap {
  overflow-x: auto;
  margin-bottom: 14px;
}

.report-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;

  th,
  td {
    border: 1px solid #ebe3d7;
    padding: 10px 12px;
    text-align: left;
    vertical-align: top;
  }

  th {
    background: #f6f1e9;
    color: #5d6470;
  }
}

.ai-dialog-hero {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 20px 22px;
  border-radius: 20px;
  background: linear-gradient(135deg, #fff5e8 0%, #f5eee5 55%, #f1e2d2 100%);
  margin-bottom: 18px;

  .eyebrow {
    margin: 0 0 6px;
    font-size: 12px;
    letter-spacing: 0.08em;
    color: #9a5d1a;
    text-transform: uppercase;
  }

  h3 {
    margin: 0 0 8px;
    font-size: 22px;
    color: #263238;
  }

  p {
    margin: 0;
    color: #64707d;
    font-size: 13px;
    line-height: 1.7;
  }
}

.status-pill {
  padding: 8px 14px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.86);
  color: #6b7280;

  &.loading { color: #9a5d1a; }
  &.error { color: #d33a2c; }
  &.success { color: #1f8f5f; }
}

.progress-shell {
  padding: 18px;
  border-radius: 18px;
  background: #faf7f2;
  border: 1px solid rgba(191, 173, 148, 0.2);
}

.progress-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
  color: #2c3e50;
  font-size: 14px;

  strong {
    font-size: 16px;
  }
}

.progress-steps {
  margin-top: 14px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.step-item {
  padding: 10px 12px;
  border-radius: 12px;
  background: #f5f5f5;
  color: #909399;
  font-size: 12px;
  text-align: center;
  transition: all 0.25s ease;

  &.active {
    background: linear-gradient(135deg, rgba(230, 126, 34, 0.16), rgba(154, 93, 26, 0.18));
    color: #9a5d1a;
    font-weight: 600;
  }
}

.report-toolbar {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
  color: #8b8f97;
  font-size: 12px;
}

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

  .info-card-head {
    flex-direction: column;
  }
}

@media (max-width: 768px) {
  .floating-notes-widget {
    right: 12px;
    left: 12px;
    bottom: 14px;
    align-items: stretch;
  }

  .notes-fab,
  .notes-panel {
    width: 100%;
  }

  .notes-actions {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
