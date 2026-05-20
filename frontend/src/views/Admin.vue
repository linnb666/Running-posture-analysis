<template>
  <div class="admin-page">
    <div class="page-banner">
      <div class="banner-title">管理员控制台</div>
      <div class="banner-subtitle">用户、记录与文件统一管理，支持高风险操作审计追踪。</div>
    </div>

    <div class="overview-grid">
      <div class="overview-card">
        <div class="label">用户总数</div>
        <div class="value">{{ overview.total_users ?? 0 }}</div>
      </div>
      <div class="overview-card">
        <div class="label">活跃用户</div>
        <div class="value">{{ overview.active_users ?? 0 }}</div>
      </div>
      <div class="overview-card">
        <div class="label">管理员</div>
        <div class="value">{{ overview.admin_users ?? 0 }}</div>
      </div>
      <div class="overview-card">
        <div class="label">分析记录</div>
        <div class="value">{{ overview.total_records ?? 0 }}</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="title-wrap">
          <h2>用户管理</h2>
          <p>可直接修改状态、重置密码、查看记录和硬删除用户全量数据。</p>
        </div>
        <div class="toolbar">
          <el-input
            v-model="keyword"
            placeholder="按用户名/邮箱搜索"
            clearable
            style="width: 260px"
            @keyup.enter="reloadUsers"
            @clear="reloadUsers"
          />
          <el-button type="primary" @click="reloadUsers">搜索</el-button>
          <el-button @click="refreshAll">刷新</el-button>
          <el-button type="warning" plain @click="cleanupStaleQueuedTasks">清理卡住队列</el-button>
          <el-button type="danger" plain @click="cleanupDanglingTasks">修复悬挂任务</el-button>
          <el-button type="warning" plain @click="cleanupOrphanStorage">清理孤儿媒体</el-button>
        </div>
      </div>

      <div v-if="selectedUsers.length" class="bulk-action">
        <span>已选择 {{ selectedUsers.length }} 个用户</span>
        <el-button type="danger" plain size="small" @click="batchHardDeleteUsers">
          批量硬删除
        </el-button>
      </div>

      <el-table
        v-loading="usersLoading"
        :data="users"
        border
        stripe
        row-key="id"
        @selection-change="onUserSelectionChange"
      >
        <el-table-column type="selection" width="50" :selectable="userSelectable" />
        <el-table-column prop="id" label="ID" width="72" />
        <el-table-column prop="username" label="用户名" min-width="130" />
        <el-table-column prop="email" label="邮箱" min-width="180" />
        <el-table-column label="角色" width="110">
          <template #default="{ row }">
            <el-tag :type="row.is_admin ? 'warning' : 'info'">
              {{ row.is_admin ? '管理员' : '普通用户' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="row.is_active ? 'success' : 'danger'">
              {{ row.is_active ? '可用' : '禁用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="记录/任务" width="130">
          <template #default="{ row }">
            <span>{{ row.record_count || 0 }} / {{ row.task_count || 0 }}</span>
          </template>
        </el-table-column>
        <el-table-column label="创建时间" min-width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" min-width="360" fixed="right">
          <template #default="{ row }">
            <div class="actions">
              <el-button link type="primary" @click="openRecords(row)">查看记录</el-button>
              <el-button link type="primary" @click="openResetPassword(row)">重置密码</el-button>
              <el-button link @click="toggleUserAdmin(row)">
                {{ row.is_admin ? '取消管理员' : '设为管理员' }}
              </el-button>
              <el-button link @click="toggleUserActive(row)">
                {{ row.is_active ? '禁用用户' : '启用用户' }}
              </el-button>
              <el-button
                link
                type="danger"
                :disabled="row.is_admin || row.id === userStore.user?.id"
                @click="hardDeleteUser(row)"
              >
                硬删除
              </el-button>
            </div>
          </template>
        </el-table-column>
      </el-table>

      <div class="pager">
        <el-pagination
          background
          layout="total, prev, pager, next"
          :total="total"
          :current-page="page"
          :page-size="perPage"
          @current-change="handlePageChange"
        />
      </div>
    </div>

    <div class="panel">
      <div class="panel-header">
        <div class="title-wrap">
          <h2>管理员审计日志</h2>
          <p>记录用户状态修改、密码重置、批量删除等关键操作（中文解释）。</p>
        </div>
      </div>
      <el-table v-loading="auditLoading" :data="auditLogs" border>
        <el-table-column prop="id" label="日志ID" width="88" />
        <el-table-column prop="action_cn" label="操作类型" width="150" />
        <el-table-column prop="admin_username" label="管理员" width="120" />
        <el-table-column prop="target_username" label="目标用户" width="130" />
        <el-table-column prop="target_record_id" label="目标记录" width="98" />
        <el-table-column label="时间" min-width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="中文说明" min-width="300">
          <template #default="{ row }">
            <span class="detail-text">{{ row.description_cn }}</span>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <el-drawer
      v-model="recordsDrawerVisible"
      size="55%"
      :title="`用户记录 · ${activeUser?.username || ''}`"
      destroy-on-close
    >
      <div v-if="selectedRecords.length" class="bulk-action">
        <span>已选择 {{ selectedRecords.length }} 条记录</span>
        <el-button type="danger" plain size="small" @click="batchHardDeleteRecords">
          批量硬删除
        </el-button>
      </div>
      <el-table
        v-loading="recordsLoading"
        :data="records"
        border
        row-key="id"
        @selection-change="onRecordSelectionChange"
      >
        <el-table-column type="selection" width="50" />
        <el-table-column prop="id" label="记录ID" width="88" />
        <el-table-column prop="video_filename" label="文件名" min-width="180" />
        <el-table-column prop="view_angle" label="视角" width="90" />
        <el-table-column label="得分" width="90">
          <template #default="{ row }">
            {{ typeof row.total_score === 'number' ? row.total_score.toFixed(1) : '--' }}
          </template>
        </el-table-column>
        <el-table-column label="时间" min-width="170">
          <template #default="{ row }">
            {{ formatDateTime(row.created_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="120" fixed="right">
          <template #default="{ row }">
            <el-button link type="danger" @click="hardDeleteRecord(row)">硬删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <div class="pager">
        <el-pagination
          background
          layout="total, prev, pager, next"
          :total="recordTotal"
          :current-page="recordPage"
          :page-size="recordPerPage"
          @current-change="handleRecordPageChange"
        />
      </div>
    </el-drawer>

    <el-dialog v-model="resetDialogVisible" width="420px" title="重置用户密码">
      <el-form label-position="top">
        <el-form-item label="用户名">
          <el-input :model-value="resetForm.username" disabled />
        </el-form-item>
        <el-form-item label="新密码（至少 6 位）">
          <el-input
            v-model="resetForm.password"
            type="password"
            show-password
            placeholder="请输入新密码"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="resetDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="resetLoading" @click="submitResetPassword">
          确认重置
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import { adminApi } from '@/api/admin'
import { useUserStore } from '@/stores/user'

const userStore = useUserStore()

const overview = ref({})
const users = ref([])
const usersLoading = ref(false)
const auditLoading = ref(false)
const recordsLoading = ref(false)
const resetLoading = ref(false)

const keyword = ref('')
const page = ref(1)
const perPage = ref(20)
const total = ref(0)

const auditLogs = ref([])

const recordsDrawerVisible = ref(false)
const activeUser = ref(null)
const records = ref([])
const selectedRecords = ref([])
const recordPage = ref(1)
const recordPerPage = ref(10)
const recordTotal = ref(0)
const selectedUsers = ref([])

const resetDialogVisible = ref(false)
const resetForm = reactive({
  userId: null,
  username: '',
  password: ''
})

onMounted(async () => {
  await refreshAll()
})

async function refreshAll() {
  try {
    await Promise.all([loadOverview(), loadUsers(), loadAuditLogs()])
  } catch (_err) {
    ElMessage.error('管理数据加载失败，请稍后重试')
  }
}

async function loadOverview() {
  overview.value = await adminApi.getOverview()
}

async function loadUsers() {
  usersLoading.value = true
  try {
    const resp = await adminApi.getUsers({
      page: page.value,
      per_page: perPage.value,
      keyword: keyword.value
    })
    users.value = resp.users || []
    total.value = resp.total || 0
    selectedUsers.value = []
  } finally {
    usersLoading.value = false
  }
}

async function reloadUsers() {
  page.value = 1
  await loadUsers()
}

async function handlePageChange(nextPage) {
  page.value = nextPage
  await loadUsers()
}

async function loadAuditLogs() {
  auditLoading.value = true
  try {
    const resp = await adminApi.getAuditLogs({ page: 1, per_page: 12 })
    auditLogs.value = resp.logs || []
  } finally {
    auditLoading.value = false
  }
}

function openResetPassword(user) {
  resetForm.userId = user.id
  resetForm.username = user.username
  resetForm.password = ''
  resetDialogVisible.value = true
}

async function submitResetPassword() {
  if (!resetForm.userId) return
  if (!resetForm.password || resetForm.password.length < 6) {
    ElMessage.warning('密码至少 6 位')
    return
  }
  resetLoading.value = true
  try {
    await adminApi.resetPassword(resetForm.userId, resetForm.password)
    ElMessage.success('密码重置成功')
    resetDialogVisible.value = false
    await loadAuditLogs()
  } finally {
    resetLoading.value = false
  }
}

async function toggleUserAdmin(user) {
  const actionText = user.is_admin ? '取消管理员权限' : '设置为管理员'
  try {
    await ElMessageBox.confirm(
      `确认要${actionText}：${user.username}？`,
      '权限变更',
      { type: 'warning', confirmButtonText: '确认', cancelButtonText: '取消' }
    )
    await adminApi.updateUser(user.id, { is_admin: !user.is_admin })
    ElMessage.success('权限更新成功')
    await Promise.all([loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('权限更新失败')
    }
  }
}

async function toggleUserActive(user) {
  const actionText = user.is_active ? '禁用' : '启用'
  try {
    await ElMessageBox.confirm(
      `确认要${actionText}用户：${user.username}？`,
      '状态变更',
      { type: user.is_active ? 'warning' : 'info', confirmButtonText: '确认', cancelButtonText: '取消' }
    )
    await adminApi.updateUser(user.id, { is_active: !user.is_active })
    ElMessage.success('状态更新成功')
    await Promise.all([loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('状态更新失败')
    }
  }
}

async function openRecords(user) {
  activeUser.value = user
  recordPage.value = 1
  recordsDrawerVisible.value = true
  await loadRecords()
}

async function handleRecordPageChange(nextPage) {
  recordPage.value = nextPage
  await loadRecords()
}

async function loadRecords() {
  if (!activeUser.value) return
  recordsLoading.value = true
  try {
    const resp = await adminApi.getUserRecords(activeUser.value.id, {
      page: recordPage.value,
      per_page: recordPerPage.value
    })
    records.value = resp.records || []
    recordTotal.value = resp.total || 0
    selectedRecords.value = []
  } finally {
    recordsLoading.value = false
  }
}

function onUserSelectionChange(rows) {
  selectedUsers.value = rows || []
}

function userSelectable(row) {
  return !(row.is_admin || row.id === userStore.user?.id)
}

function onRecordSelectionChange(rows) {
  selectedRecords.value = rows || []
}

async function hardDeleteRecord(record) {
  try {
    await ElMessageBox.confirm(
      `确认硬删除记录 #${record.id}？该操作会清除数据库与媒体文件。`,
      '高风险操作',
      { type: 'error', confirmButtonText: '确认删除', cancelButtonText: '取消' }
    )
    await adminApi.hardDeleteRecord(record.id)
    ElMessage.success('记录已硬删除')
    await Promise.all([loadRecords(), loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('记录删除失败')
    }
  }
}

async function batchHardDeleteRecords() {
  const ids = selectedRecords.value.map((item) => item.id)
  if (!ids.length) return
  try {
    await ElMessageBox.confirm(
      `确认批量硬删除 ${ids.length} 条记录？该操作不可恢复。`,
      '高风险操作',
      { type: 'error', confirmButtonText: '确认删除', cancelButtonText: '取消' }
    )
    const resp = await adminApi.hardDeleteRecordsBatch(ids)
    ElMessage.success(`已删除 ${resp.deleted_ids?.length || 0} 条记录`)
    await Promise.all([loadRecords(), loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('批量删除记录失败')
    }
  }
}

async function hardDeleteUser(user) {
  try {
    const prompt = await ElMessageBox.prompt(
      `输入用户名 ${user.username} 以确认硬删除。`,
      '高风险操作',
      {
        type: 'error',
        confirmButtonText: '确认删除',
        cancelButtonText: '取消',
        inputPlaceholder: '请输入用户名',
        inputPattern: /^.+$/,
        inputErrorMessage: '请输入用户名'
      }
    )
    if ((prompt.value || '').trim() !== user.username) {
      ElMessage.warning('用户名校验失败，已取消删除')
      return
    }
    await adminApi.hardDeleteUser(user.id)
    ElMessage.success('用户及其全量数据已硬删除')
    await Promise.all([loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('用户删除失败')
    }
  }
}

async function batchHardDeleteUsers() {
  const ids = selectedUsers.value.map((item) => item.id)
  if (!ids.length) return
  try {
    await ElMessageBox.confirm(
      `确认批量硬删除 ${ids.length} 个用户？会删除其数据库与媒体文件。`,
      '高风险操作',
      { type: 'error', confirmButtonText: '确认删除', cancelButtonText: '取消' }
    )
    const resp = await adminApi.hardDeleteUsersBatch(ids)
    const deletedCount = resp.summary?.deleted_count || 0
    const skippedCount = resp.summary?.skipped_count || 0
    ElMessage.success(`批量删除完成：成功 ${deletedCount}，跳过 ${skippedCount}`)
    await Promise.all([loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('批量删除用户失败')
    }
  }
}

async function cleanupOrphanStorage() {
  try {
    await ElMessageBox.confirm(
      '将清理 output/tasks 下未关联数据库记录的孤儿目录，是否继续？',
      '清理确认',
      { type: 'warning', confirmButtonText: '继续清理', cancelButtonText: '取消' }
    )
    const resp = await adminApi.cleanupOrphanStorage()
    const count = resp.summary?.orphan_dirs_deleted || 0
    ElMessage.success(`清理完成，删除孤儿目录 ${count} 个`)
    await Promise.all([loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('孤儿媒体清理失败')
    }
  }
}

async function cleanupDanglingTasks() {
  try {
    await ElMessageBox.confirm(
      '将清理 result_record_id 指向不存在记录的悬挂任务，并同步删除其任务媒体目录，是否继续？',
      '修复确认',
      { type: 'warning', confirmButtonText: '继续修复', cancelButtonText: '取消' }
    )
    const resp = await adminApi.cleanupDanglingTasks()
    const count = resp.summary?.dangling_tasks_deleted || 0
    ElMessage.success(`修复完成，清理悬挂任务 ${count} 个`)
    await Promise.all([loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('悬挂任务修复失败')
    }
  }
}

async function cleanupStaleQueuedTasks() {
  try {
    await ElMessageBox.confirm(
      '将删除创建时间超过 10 分钟且无结果记录的 queued 任务，并清理其目录，是否继续？',
      '清理确认',
      { type: 'warning', confirmButtonText: '继续清理', cancelButtonText: '取消' }
    )
    const resp = await adminApi.cleanupStaleQueuedTasks(10)
    const count = resp.summary?.stale_queued_deleted || 0
    ElMessage.success(`清理完成，删除卡住队列任务 ${count} 个`)
    await Promise.all([loadUsers(), loadOverview(), loadAuditLogs()])
  } catch (err) {
    if (err !== 'cancel' && err !== 'close') {
      ElMessage.error('卡住队列清理失败')
    }
  }
}

function formatDateTime(dateStr) {
  if (!dateStr) return '-'
  const date = new Date(dateStr)
  if (Number.isNaN(date.getTime())) return '-'
  return date.toLocaleString('zh-CN', { hour12: false })
}

</script>

<style lang="scss" scoped>
.admin-page {
  max-width: 1320px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-banner {
  border-radius: 14px;
  padding: 18px 22px;
  background:
    radial-gradient(circle at 88% 18%, rgba(230, 126, 34, 0.2), transparent 48%),
    linear-gradient(135deg, #f7fafc 0%, #eef3f8 100%);
  border: 1px solid #e7edf4;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

.banner-title {
  font-size: 18px;
  font-weight: 600;
  color: #1f2d3d;
}

.banner-subtitle {
  margin-top: 6px;
  color: #708198;
  font-size: 13px;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}

.overview-card {
  background: linear-gradient(140deg, #ffffff 0%, #f7fafc 100%);
  border: 1px solid #ebeef5;
  border-radius: 12px;
  padding: 16px 18px;
  box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);

  .label {
    color: #7a8699;
    font-size: 13px;
    margin-bottom: 6px;
  }

  .value {
    color: #1f2d3d;
    font-size: 28px;
    font-weight: 600;
    line-height: 1.1;
  }
}

.panel {
  background: #fff;
  border-radius: 14px;
  border: 1px solid #ebeef5;
  padding: 18px;
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
}

.panel-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.title-wrap {
  h2 {
    margin: 0;
    font-size: 18px;
    color: #1f2d3d;
  }

  p {
    margin: 6px 0 0;
    color: #7a8699;
    font-size: 13px;
  }
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
}

.bulk-action {
  margin-bottom: 12px;
  padding: 10px 12px;
  border-radius: 10px;
  background: #fff7f0;
  border: 1px solid #f2d2b4;
  display: flex;
  align-items: center;
  justify-content: space-between;
  color: #9a5a2d;
  font-size: 13px;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 8px;
}

.pager {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.detail-text {
  font-size: 12px;
  color: #5b6472;
  line-height: 1.5;
}

@media (max-width: 1024px) {
  .overview-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .panel-header {
    flex-direction: column;
    align-items: stretch;
  }

  .toolbar {
    flex-wrap: wrap;
  }
}

@media (max-width: 640px) {
  .overview-grid {
    grid-template-columns: 1fr;
  }
}
</style>
