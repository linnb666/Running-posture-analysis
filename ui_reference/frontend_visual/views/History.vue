<template>
  <div class="history-page">
    <!-- 筛选栏 -->
    <div class="filter-bar">
      <div class="filter-left">
        <el-select v-model="filters.status" placeholder="状态" clearable style="width: 120px">
          <el-option label="已完成" value="completed" />
          <el-option label="分析中" value="running" />
          <el-option label="失败" value="failed" />
        </el-select>

        <el-select v-model="filters.view_type" placeholder="视角" clearable style="width: 120px">
          <el-option label="侧面" value="side" />
          <el-option label="正面" value="front" />
        </el-select>

        <el-input
          v-model="filters.min_score"
          placeholder="最低分"
          type="number"
          style="width: 100px"
          clearable
        />
        <span class="filter-separator">-</span>
        <el-input
          v-model="filters.max_score"
          placeholder="最高分"
          type="number"
          style="width: 100px"
          clearable
        />

        <el-button @click="loadHistory">
          <el-icon><Search /></el-icon>
          筛选
        </el-button>
      </div>

      <div class="filter-right">
        <el-button
          type="primary"
          :disabled="selectedIds.length < 2"
          @click="goCompare"
        >
          <el-icon><DataLine /></el-icon>
          对比选中 ({{ selectedIds.length }})
        </el-button>
      </div>
    </div>

    <!-- 记录列表 -->
    <div class="records-container">
      <div v-if="loading" class="loading-state">
        <el-skeleton :rows="5" animated />
      </div>

      <div v-else-if="records.length === 0" class="empty-state">
        <el-empty description="暂无分析记录">
          <el-button type="primary" @click="$router.push('/analyze')">
            开始第一次分析
          </el-button>
        </el-empty>
      </div>

      <div v-else class="records-list">
        <div
          v-for="record in records"
          :key="record.id"
          class="record-card"
          :class="{ selected: selectedIds.includes(record.id) }"
        >
          <div class="record-checkbox">
            <el-checkbox
              :model-value="selectedIds.includes(record.id)"
              :disabled="record.status !== 'completed'"
              @change="toggleSelect(record.id)"
            />
          </div>

          <div class="record-main" @click="goToResult(record)">
            <div class="record-info">
              <h3 class="record-name">{{ record.video_filename }}</h3>
              <div class="record-meta">
                <el-tag size="small" :type="getStatusType(record.status)">
                  {{ getStatusLabel(record.status) }}
                </el-tag>
                <span class="meta-item">
                  <el-icon><View /></el-icon>
                  {{ viewLabel(record.view_type) }}
                </span>
                <span class="meta-item">
                  <el-icon><Clock /></el-icon>
                  {{ formatDate(record.created_at) }}
                </span>
              </div>
            </div>

            <div v-if="record.status === 'completed'" class="record-score" :class="getGradeClass(record.grade)">
              <span class="score-value">{{ record.total_score?.toFixed(1) }}</span>
              <span class="score-grade">{{ getGradeLabel(record.grade) }}</span>
            </div>

            <div v-else-if="record.status === 'running'" class="record-progress">
              <el-progress
                type="circle"
                :percentage="record.progress || 0"
                :width="50"
                :stroke-width="4"
              />
            </div>
          </div>

          <div class="record-actions">
            <el-button
              v-if="record.status === 'completed'"
              type="primary"
              link
              @click.stop="goToResult(record)"
            >
              查看
            </el-button>
            <el-popconfirm
              title="确定删除这条记录吗？"
              @confirm="deleteRecord(record.id)"
            >
              <template #reference>
                <el-button type="danger" link @click.stop>删除</el-button>
              </template>
            </el-popconfirm>
          </div>
        </div>
      </div>

      <!-- 分页 -->
      <div v-if="pagination.total > pagination.per_page" class="pagination-wrapper">
        <el-pagination
          v-model:current-page="pagination.page"
          :page-size="pagination.per_page"
          :total="pagination.total"
          layout="prev, pager, next"
          @current-change="loadHistory"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { historyApi } from '@/api/history'
import { Search, DataLine, View, Clock } from '@element-plus/icons-vue'
import { gradeLabel, viewLabel } from '@/utils/analysis'

const router = useRouter()

const loading = ref(true)
const records = ref([])
const selectedIds = ref([])

const filters = reactive({
  status: '',
  view_type: '',
  min_score: '',
  max_score: ''
})

const pagination = reactive({
  page: 1,
  per_page: 20,
  total: 0
})

onMounted(() => {
  loadHistory()
})

async function loadHistory() {
  loading.value = true

  try {
    const params = {
      page: pagination.page,
      per_page: pagination.per_page
    }

    if (filters.status) params.status = filters.status
    if (filters.view_type) params.view_type = filters.view_type
    if (filters.min_score) params.min_score = parseFloat(filters.min_score)
    if (filters.max_score) params.max_score = parseFloat(filters.max_score)

    const response = await historyApi.getHistory(params)
    let nextRecords = response.records || []
    if (filters.status) {
      nextRecords = nextRecords.filter(item => item.status === filters.status)
    }
    if (filters.min_score) {
      nextRecords = nextRecords.filter(item => (item.total_score || 0) >= parseFloat(filters.min_score))
    }
    if (filters.max_score) {
      nextRecords = nextRecords.filter(item => (item.total_score || 0) <= parseFloat(filters.max_score))
    }
    records.value = nextRecords
    pagination.total = response.total || nextRecords.length
    pagination.per_page = response.limit || pagination.per_page
  } catch (error) {
    ElMessage.error('加载失败')
  } finally {
    loading.value = false
  }
}

function toggleSelect(id) {
  const index = selectedIds.value.indexOf(id)
  if (index > -1) {
    selectedIds.value.splice(index, 1)
  } else {
    if (selectedIds.value.length >= 3) {
      ElMessage.warning('最多选择3条记录进行对比')
      return
    }
    selectedIds.value.push(id)
  }
}

function goToResult(record) {
  if (record.status === 'completed') {
    router.push(`/result/${record.id}`)
  }
}

function goCompare() {
  if (selectedIds.value.length >= 2) {
    router.push({
      path: '/compare',
      query: { ids: selectedIds.value.join(',') }
    })
  }
}

async function deleteRecord(id) {
  try {
    await historyApi.deleteHistory(id)
    ElMessage.success('删除成功')
    loadHistory()
  } catch (error) {
    ElMessage.error('删除失败')
  }
}

function formatDate(dateStr) {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function getStatusType(status) {
  const map = { completed: 'success', running: 'warning', failed: 'danger', pending: 'info' }
  return map[status] || 'info'
}

function getStatusLabel(status) {
  const map = { completed: '已完成', running: '分析中', failed: '失败', pending: '等待中' }
  return map[status] || status
}

function getGradeClass(grade) {
  return `grade-${grade || 'unknown'}`
}

function getGradeLabel(grade) {
  return gradeLabel(grade)
}
</script>

<style lang="scss" scoped>
.history-page {
  max-width: 1000px;
  margin: 0 auto;
}

// 筛选栏
.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #fff;
  padding: 16px 20px;
  border-radius: 12px;
  margin-bottom: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.filter-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.filter-separator {
  color: #c0c4cc;
}

// 记录列表
.records-container {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.records-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.record-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 20px;
  background: #f8f9fa;
  border-radius: 10px;
  border: 2px solid transparent;
  transition: all 0.2s ease;

  &:hover {
    background: #f0f2f5;
  }

  &.selected {
    border-color: #e67e22;
    background: rgba(230, 126, 34, 0.05);
  }
}

.record-checkbox {
  flex-shrink: 0;
}

.record-main {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 20px;
  cursor: pointer;
}

.record-info {
  flex: 1;
}

.record-name {
  font-size: 15px;
  font-weight: 500;
  color: #2c3e50;
  margin: 0 0 8px;
}

.record-meta {
  display: flex;
  align-items: center;
  gap: 16px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: #909399;

  .el-icon {
    font-size: 14px;
  }
}

.record-score {
  text-align: right;

  .score-value {
    display: block;
    font-size: 24px;
    font-weight: 700;
  }

  .score-grade {
    font-size: 12px;
  }

  &.grade-excellent { color: #67c23a; }
  &.grade-good { color: #409eff; }
  &.grade-fair { color: #e6a23c; }
  &.grade-poor { color: #f56c6c; }
}

.record-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

// 分页
.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}

// 状态
.loading-state,
.empty-state {
  padding: 60px 0;
}
</style>
