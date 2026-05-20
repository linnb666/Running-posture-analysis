<template>
  <div class="compare-page">
    <div v-if="loading" class="loading-state">
      <el-skeleton :rows="10" animated />
    </div>

    <template v-else-if="records.length >= 2">
      <!-- 对比概览 -->
      <div class="compare-header">
        <h2>分析结果对比</h2>
        <p>对比 {{ records.length }} 次分析的结果差异</p>
      </div>
      <el-alert
        v-if="hasMixedViews"
        type="error"
        show-icon
        :closable="false"
        class="compare-alert"
        title="当前选择包含不同视角（正面/侧面），详细维度不可直接对比，请改为同视角记录"
      />

      <!-- 分数对比 -->
      <div class="score-compare">
        <div
          v-for="record in records"
          :key="record.id"
          class="score-item"
          :class="getGradeClass(record.grade)"
        >
          <div class="score-ring">
            <svg viewBox="0 0 80 80">
              <circle class="ring-bg" cx="40" cy="40" r="36" />
              <circle
                class="ring-progress"
                cx="40" cy="40" r="36"
                :stroke-dasharray="226.195"
                :stroke-dashoffset="226.195 * (1 - (record.total_score || 0) / 100)"
              />
            </svg>
            <span class="score-value">{{ record.total_score?.toFixed(1) }}</span>
          </div>
          <div class="score-info">
            <span class="score-name">{{ record.video_filename }}</span>
            <span class="score-date">{{ formatDate(record.created_at) }}</span>
          </div>
        </div>
      </div>

      <!-- 维度对比图表 -->
      <div v-if="canCompareDetailed" class="chart-section">
        <div class="chart-card">
          <h3>维度评分对比（{{ activeDimensionLabels.join(' / ') }}）</h3>
          <div class="chart-container" ref="radarChartRef"></div>
        </div>

        <div class="chart-card">
          <h3>分数差异</h3>
          <div class="chart-container" ref="barChartRef"></div>
        </div>
      </div>

      <!-- 详细数据对比表格 -->
      <div v-if="canCompareDetailed" class="table-section">
        <h3>详细指标对比</h3>
        <el-table :data="comparisonData" border stripe>
          <el-table-column prop="metric" label="指标" width="180" fixed />
          <el-table-column
            v-for="(record, index) in records"
            :key="record.id"
            :label="getShortName(record.video_filename)"
            align="center"
          >
            <template #default="{ row }">
              <span :class="getCellClass(row, index)">
                {{ row.values[index] }}
              </span>
            </template>
          </el-table-column>
          <el-table-column label="差异" align="center" width="120">
            <template #default="{ row }">
              <span :class="getDiffClass(row.diff)">
                {{ row.diff }}
              </span>
            </template>
          </el-table-column>
        </el-table>
      </div>

      <div v-else class="blocked-section">
        <el-empty description="不同视角不可进行详细维度对比，请返回历史记录并选择同视角数据。">
          <el-button type="primary" @click="$router.push('/history')">
            返回历史记录
          </el-button>
        </el-empty>
      </div>
    </template>

    <div v-else class="empty-state">
      <el-empty description="请至少选择2条记录进行对比">
        <el-button type="primary" @click="$router.push('/history')">
          返回历史记录
        </el-button>
      </el-empty>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute } from 'vue-router'
import * as echarts from 'echarts'
import { historyApi } from '@/api/history'

const route = useRoute()

const loading = ref(true)
const records = ref([])
const radarChartRef = ref(null)
const barChartRef = ref(null)

let radarChart = null
let barChart = null

const preferredDimensionOrder = [
  'stability',
  'efficiency',
  'form',
  'lower_limb_alignment',
  'lateral_stability'
]

const dimensionNameMap = {
  stability: '稳定性',
  efficiency: '效率',
  form: '跑姿',
  lower_limb_alignment: '下肢力线',
  lateral_stability: '横向稳定性'
}

const hasMixedViews = computed(() => {
  const views = new Set(
    records.value
      .map(r => r.view_type || r.view_angle)
      .filter(Boolean)
  )
  return views.size > 1
})

const canCompareDetailed = computed(() => !hasMixedViews.value)

const activeDimensionKeys = computed(() => {
  if (!records.value.length || !canCompareDetailed.value) return []
  const sets = records.value.map(r => new Set(Object.keys(r.dimension_scores || {})))
  let keys = [...sets[0]].filter(k => sets.every(s => s.has(k)))
  if (keys.length === 0) {
    const union = new Set()
    sets.forEach(s => s.forEach(k => union.add(k)))
    keys = [...union]
  }
  keys.sort((a, b) => {
    const ai = preferredDimensionOrder.indexOf(a)
    const bi = preferredDimensionOrder.indexOf(b)
    const aRank = ai === -1 ? 999 : ai
    const bRank = bi === -1 ? 999 : bi
    return aRank - bRank
  })
  return keys
})

const activeDimensionLabels = computed(() => {
  return activeDimensionKeys.value.map(k => dimensionNameMap[k] || k)
})

function toNumberOrNull(value) {
  const num = Number(value)
  return Number.isFinite(num) ? num : null
}

function dimensionValue(record, key) {
  return toNumberOrNull(record.dimension_scores?.[key])
}

function calcDiff(values) {
  const nums = values.map(v => parseFloat(v)).filter(v => !Number.isNaN(v))
  if (nums.length < 2) return '--'
  return (Math.max(...nums) - Math.min(...nums)).toFixed(1)
}

const comparisonData = computed(() => {
  if (records.value.length < 2 || !canCompareDetailed.value) return []

  const rows = []
  rows.push({
    metric: '总分',
    values: records.value.map(r => {
      const val = toNumberOrNull(r.total_score)
      return val == null ? '--' : val.toFixed(1)
    })
  })

  for (const key of activeDimensionKeys.value) {
    rows.push({
      metric: dimensionNameMap[key] || key,
      values: records.value.map(r => {
        const val = dimensionValue(r, key)
        return val == null ? '--' : val.toFixed(1)
      })
    })
  }

  const hasCadence = records.value.some(r => toNumberOrNull(r.kinematic?.cadence?.cadence) != null)
  if (hasCadence) {
    rows.push({
      metric: '步频 (步/分)',
      values: records.value.map(r => {
        const val = toNumberOrNull(r.kinematic?.cadence?.cadence)
        return val == null ? '--' : val.toFixed(0)
      })
    })
  }

  return rows.map(row => {
    return {
      ...row,
      diff: calcDiff(row.values)
    }
  })
})

onMounted(async () => {
  const rawIds = Array.isArray(route.query.ids)
    ? route.query.ids.join(',')
    : (route.query.ids || '')
  const ids = String(rawIds).split(',').map(x => x.trim()).filter(Boolean)
  if (!ids || ids.length < 2) {
    loading.value = false
    return
  }

  try {
    const promises = ids.map(id => historyApi.getHistoryDetail(id))
    const responses = await Promise.all(promises)
    records.value = responses.map(r => r.record)

    loading.value = false
    await nextTick()
    initCharts()
  } catch (error) {
    console.error('Failed to load records:', error)
    loading.value = false
  } finally {
    if (loading.value) loading.value = false
  }
})

onUnmounted(() => {
  radarChart?.dispose()
  barChart?.dispose()
})

function initCharts() {
  if (!canCompareDetailed.value) {
    radarChart?.dispose()
    barChart?.dispose()
    radarChart = null
    barChart = null
    return
  }
  initRadarChart()
  initBarChart()
}

function initRadarChart() {
  if (!radarChartRef.value || records.value.length < 2) return

  radarChart?.dispose()
  radarChart = echarts.init(radarChartRef.value)

  const dimensions = activeDimensionKeys.value
  if (!dimensions.length) return
  const colors = ['#e67e22', '#409eff', '#67c23a', '#9b59b6', '#16a085']

  const indicator = dimensions.map(d => ({
    name: dimensionNameMap[d] || d,
    max: 100
  }))

  const series = records.value.map((record, index) => ({
    value: dimensions.map(d => dimensionValue(record, d) ?? 0),
    name: getShortName(record.video_filename),
    areaStyle: { color: `${colors[index % colors.length]}33` },
    lineStyle: { color: colors[index % colors.length], width: 2 },
    itemStyle: { color: colors[index % colors.length] }
  }))

  radarChart.setOption({
    legend: {
      data: records.value.map(r => getShortName(r.video_filename)),
      bottom: 0
    },
    radar: {
      indicator,
      radius: '60%'
    },
    series: [{
      type: 'radar',
      data: series
    }]
  })
}

function initBarChart() {
  if (!barChartRef.value || records.value.length < 2) return

  barChart?.dispose()
  barChart = echarts.init(barChartRef.value)

  const dimensions = activeDimensionKeys.value
  if (!dimensions.length) return
  const colors = ['#e67e22', '#409eff', '#67c23a', '#9b59b6', '#16a085']

  const series = records.value.map((record, index) => ({
    name: getShortName(record.video_filename),
    type: 'bar',
    data: dimensions.map(d => dimensionValue(record, d) ?? 0),
    itemStyle: { color: colors[index % colors.length] }
  }))

  barChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: {
      data: records.value.map(r => getShortName(r.video_filename)),
      bottom: 0
    },
    grid: { left: 50, right: 20, top: 20, bottom: 60 },
    xAxis: {
      type: 'category',
      data: dimensions.map(d => dimensionNameMap[d] || d)
    },
    yAxis: {
      type: 'value',
      max: 100
    },
    series
  })
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleDateString('zh-CN')
}

function getShortName(filename) {
  if (!filename) return ''
  return filename.length > 15 ? filename.slice(0, 12) + '...' : filename
}

function getGradeClass(grade) {
  return `grade-${grade || 'unknown'}`
}

function getCellClass(row, index) {
  const values = row.values.map(v => parseFloat(v)).filter(v => !isNaN(v))
  if (values.length < 2) return ''

  const current = parseFloat(row.values[index])
  if (isNaN(current)) return ''

  if (current === Math.max(...values)) return 'cell-best'
  if (current === Math.min(...values)) return 'cell-worst'
  return ''
}

function getDiffClass(diff) {
  const num = parseFloat(diff)
  if (isNaN(num)) return ''
  if (num > 10) return 'diff-large'
  if (num > 5) return 'diff-medium'
  return 'diff-small'
}
</script>

<style lang="scss" scoped>
.compare-page {
  max-width: 1200px;
  margin: 0 auto;
}

.compare-header {
  text-align: center;
  margin-bottom: 32px;

  h2 {
    font-size: 24px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 8px;
  }

  p {
    font-size: 14px;
    color: #909399;
    margin: 0;
  }
}

.compare-alert {
  margin-bottom: 20px;
}

// 分数对比
.score-compare {
  display: flex;
  justify-content: center;
  gap: 48px;
  margin-bottom: 32px;
}

.score-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 24px;
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
  min-width: 160px;

  &.grade-excellent { .ring-progress { stroke: #67c23a; } }
  &.grade-good { .ring-progress { stroke: #409eff; } }
  &.grade-fair { .ring-progress { stroke: #e6a23c; } }
  &.grade-poor { .ring-progress { stroke: #f56c6c; } }
}

.score-ring {
  position: relative;
  width: 80px;
  height: 80px;

  svg {
    transform: rotate(-90deg);
  }

  .ring-bg {
    fill: none;
    stroke: #f0f0f0;
    stroke-width: 6;
  }

  .ring-progress {
    fill: none;
    stroke-width: 6;
    stroke-linecap: round;
    transition: stroke-dashoffset 1s ease;
  }

  .score-value {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 700;
    color: #2c3e50;
  }
}

.score-info {
  text-align: center;

  .score-name {
    display: block;
    font-size: 14px;
    font-weight: 500;
    color: #2c3e50;
    max-width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .score-date {
    font-size: 12px;
    color: #909399;
  }
}

// 图表
.chart-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 32px;
}

.chart-card {
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);

  h3 {
    font-size: 15px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 16px;
  }
}

.chart-container {
  height: 300px;
}

// 表格
.table-section {
  background: #fff;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);

  h3 {
    font-size: 15px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 16px;
  }
}

.blocked-section {
  background: #fff;
  border-radius: 16px;
  padding: 28px 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.cell-best {
  color: #67c23a;
  font-weight: 600;
}

.cell-worst {
  color: #f56c6c;
}

.diff-large {
  color: #f56c6c;
  font-weight: 600;
}

.diff-medium {
  color: #e6a23c;
}

.diff-small {
  color: #67c23a;
}

// 状态
.loading-state,
.empty-state {
  padding: 60px 0;
}

@media (max-width: 768px) {
  .score-compare {
    flex-direction: column;
    align-items: center;
  }

  .chart-section {
    grid-template-columns: 1fr;
  }
}
</style>
