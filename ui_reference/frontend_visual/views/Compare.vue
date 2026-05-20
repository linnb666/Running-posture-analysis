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
      <div class="chart-section">
        <div class="chart-card">
          <h3>维度评分对比</h3>
          <div class="chart-container" ref="radarChartRef"></div>
        </div>

        <div class="chart-card">
          <h3>分数差异</h3>
          <div class="chart-container" ref="barChartRef"></div>
        </div>
      </div>

      <!-- 详细数据对比表格 -->
      <div class="table-section">
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
import { ref, computed, onMounted, nextTick } from 'vue'
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

const comparisonData = computed(() => {
  if (records.value.length < 2) return []

  const metrics = [
    { key: 'total_score', label: '总分', getter: r => r.total_score?.toFixed(1) },
    { key: 'stability', label: '稳定性', getter: r => r.dimension_scores?.stability?.toFixed(1) },
    { key: 'efficiency', label: '效率', getter: r => r.dimension_scores?.efficiency?.toFixed(1) },
    { key: 'form', label: '跑姿', getter: r => r.dimension_scores?.form?.toFixed(1) },
    { key: 'cadence', label: '步频 (步/分)', getter: r => r.kinematic?.cadence?.cadence?.toFixed(0) || '--' }
  ]

  return metrics.map(m => {
    const values = records.value.map(r => m.getter(r) || '--')
    const numValues = values.map(v => parseFloat(v)).filter(v => !isNaN(v))
    const diff = numValues.length >= 2
      ? (Math.max(...numValues) - Math.min(...numValues)).toFixed(1)
      : '--'

    return {
      metric: m.label,
      values,
      diff
    }
  })
})

onMounted(async () => {
  const ids = route.query.ids?.split(',').filter(Boolean)
  if (!ids || ids.length < 2) {
    loading.value = false
    return
  }

  try {
    const promises = ids.map(id => historyApi.getHistoryDetail(id))
    const responses = await Promise.all(promises)
    records.value = responses.map(r => r.record)

    await nextTick()
    initCharts()
  } catch (error) {
    console.error('Failed to load records:', error)
  } finally {
    loading.value = false
  }
})

function initCharts() {
  initRadarChart()
  initBarChart()
}

function initRadarChart() {
  if (!radarChartRef.value || records.value.length < 2) return

  radarChart = echarts.init(radarChartRef.value)

  const dimensions = ['stability', 'efficiency', 'form']
  const dimensionNames = { stability: '稳定性', efficiency: '效率', form: '跑姿' }
  const colors = ['#e67e22', '#409eff', '#67c23a']

  const indicator = dimensions.map(d => ({
    name: dimensionNames[d],
    max: 100
  }))

  const series = records.value.map((record, index) => ({
    value: dimensions.map(d => record.dimension_scores?.[d] || 0),
    name: getShortName(record.video_filename),
    areaStyle: { color: `${colors[index]}33` },
    lineStyle: { color: colors[index], width: 2 },
    itemStyle: { color: colors[index] }
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

  barChart = echarts.init(barChartRef.value)

  const dimensions = ['stability', 'efficiency', 'form']
  const dimensionNames = { stability: '稳定性', efficiency: '效率', form: '跑姿' }
  const colors = ['#e67e22', '#409eff', '#67c23a']

  const series = records.value.map((record, index) => ({
    name: getShortName(record.video_filename),
    type: 'bar',
    data: dimensions.map(d => record.dimension_scores?.[d] || 0),
    itemStyle: { color: colors[index] }
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
      data: dimensions.map(d => dimensionNames[d])
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
