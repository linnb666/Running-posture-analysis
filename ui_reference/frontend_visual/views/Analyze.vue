<template>
  <div class="analyze-page">
    <!-- 上传区域 -->
    <div v-if="!isAnalyzing && !isCompleted" class="upload-section">
      <div class="upload-card">
        <el-upload
          ref="uploadRef"
          class="video-uploader"
          drag
          :auto-upload="false"
          :show-file-list="false"
          :on-change="handleFileChange"
          accept="video/*"
        >
          <div v-if="!selectedFile" class="upload-placeholder">
            <div class="upload-icon">
              <el-icon><VideoCamera /></el-icon>
            </div>
            <div class="upload-text">
              <p class="primary">拖拽视频文件到此处</p>
              <p class="secondary">或点击选择文件</p>
            </div>
            <div class="upload-hint">
              支持 MP4、AVI、MOV 格式，最大 500MB
            </div>
          </div>

          <div v-else class="file-preview">
            <div class="file-icon">
              <el-icon><VideoPlay /></el-icon>
            </div>
            <div class="file-info">
              <span class="file-name">{{ selectedFile.name }}</span>
              <span class="file-size">{{ formatFileSize(selectedFile.size) }}</span>
            </div>
            <el-button type="danger" link @click.stop="clearFile">
              <el-icon><Close /></el-icon>
            </el-button>
          </div>
        </el-upload>

        <div v-if="selectedVideoUrl" class="video-preview">
          <video :src="selectedVideoUrl" controls preload="metadata" />
        </div>
      </div>

      <!-- 配置选项 -->
      <div class="config-section">
        <h3 class="config-title">分析配置</h3>

        <div class="config-grid">
          <div class="config-item">
            <label>视角类型</label>
            <el-radio-group v-model="viewAngle" class="view-radio">
              <el-radio-button value="side">
                <el-icon><Right /></el-icon>
                侧面视角
              </el-radio-button>
              <el-radio-button value="front">
                <el-icon><User /></el-icon>
                正面视角
              </el-radio-button>
            </el-radio-group>
          </div>

          <div class="config-item view-hint">
            <div v-if="viewAngle === 'side'" class="hint-content">
              <el-icon><InfoFilled /></el-icon>
              <span>侧面视角分析：膝关节角度、垂直振幅、躯干稳定性、步频等</span>
            </div>
            <div v-else class="hint-content">
              <el-icon><InfoFilled /></el-icon>
              <span>正面视角分析：下肢力线、膝外翻角度、横向稳定性、对称性等</span>
            </div>
          </div>
        </div>

        <div class="action-buttons">
          <el-button
            type="primary"
            size="large"
            :disabled="!selectedFile"
            :loading="uploading"
            @click="startAnalysis"
          >
            <el-icon><VideoPlay /></el-icon>
            开始分析
          </el-button>
        </div>
      </div>
    </div>

    <!-- 分析进度 -->
    <div v-if="isAnalyzing" class="progress-section">
      <div class="progress-card">
        <div class="progress-header">
          <div class="progress-icon">
            <div class="spinner"></div>
          </div>
          <h2>正在分析中</h2>
          <p>{{ analysisStore.taskMessage || '请稍候，分析可能需要几分钟...' }}</p>
        </div>

        <div class="progress-bar-wrapper">
          <el-progress
            :percentage="displayProgress"
            :stroke-width="8"
            :show-text="false"
            color="#e67e22"
          />
          <span class="progress-text">{{ displayProgress }}%</span>
        </div>

        <div class="progress-stages">
          <div
            v-for="(stage, index) in stages"
            :key="index"
            class="stage-item"
            :class="{ active: isStageActive(index), completed: isStageCompleted(index) }"
          >
            <div class="stage-dot">
              <el-icon v-if="isStageCompleted(index)"><Check /></el-icon>
              <span v-else>{{ index + 1 }}</span>
            </div>
            <span class="stage-label">{{ stage.label }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 分析完成 -->
    <div v-if="isCompleted" class="complete-section">
      <div class="complete-card">
        <div class="complete-icon">
          <el-icon><CircleCheckFilled /></el-icon>
        </div>
        <h2>分析完成</h2>
        <p>您的跑步动作分析已完成，点击查看详细结果</p>
        <div class="complete-actions">
          <el-button type="primary" size="large" @click="viewResult">
            查看分析结果
          </el-button>
          <el-button size="large" @click="resetAnalysis">
            继续分析
          </el-button>
        </div>
      </div>
    </div>

    <!-- 分析失败 -->
    <div v-if="isFailed" class="error-section">
      <div class="error-card">
        <div class="error-icon">
          <el-icon><CircleCloseFilled /></el-icon>
        </div>
        <h2>分析失败</h2>
        <p>{{ analysisStore.taskMessage || '分析过程中发生错误，请重试' }}</p>
        <el-button type="primary" size="large" @click="resetAnalysis">
          重新开始
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAnalysisStore } from '@/stores/analysis'
import { useWebSocket } from '@/composables/useWebSocket'
import {
  VideoCamera,
  VideoPlay,
  Close,
  Right,
  User,
  InfoFilled,
  Check,
  CircleCheckFilled,
  CircleCloseFilled
} from '@element-plus/icons-vue'

const router = useRouter()
const analysisStore = useAnalysisStore()
const { connect } = useWebSocket()

const uploadRef = ref(null)
const selectedFile = ref(null)
const selectedVideoUrl = ref('')
const viewAngle = ref('side')
const uploading = ref(false)
let pollTimer = null

const stages = [
  { label: '视频预处理', key: 's1' },
  { label: '姿态估计', key: 's2' },
  { label: '视角设置', key: 's3' },
  { label: '运动学分析', key: 's4' },
  { label: '时序分析', key: 's5' },
  { label: '质量评价', key: 's6' },
  { label: 'AI报告', key: 's7' }
]

const isAnalyzing = computed(() =>
  analysisStore.taskStatus === 'pending' || analysisStore.taskStatus === 'running'
)

const isCompleted = computed(() => analysisStore.taskStatus === 'completed')
const isFailed = computed(() => analysisStore.taskStatus === 'failed')

const currentStageIndex = computed(() => {
  if (analysisStore.taskStatus === 'completed') return stages.length - 1
  const msg = analysisStore.taskMessage || ''
  const matched = msg.match(/([1-7])\/7/)
  if (matched) return Math.max(0, Math.min(stages.length - 1, parseInt(matched[1], 10) - 1))
  if (msg.includes('保存')) return stages.length - 1
  if (msg.includes('准备')) return 0
  return analysisStore.taskStatus === 'pending' ? 0 : 1
})

const displayProgress = computed(() => {
  if (analysisStore.taskStatus === 'completed') return 100
  const base = analysisStore.taskProgress || 0
  if (analysisStore.taskStatus === 'failed') return Math.max(base, 10)
  return Math.max(base, 5)
})

onMounted(() => {
  connect()
})

onUnmounted(() => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
  if (selectedVideoUrl.value) {
    URL.revokeObjectURL(selectedVideoUrl.value)
    selectedVideoUrl.value = ''
  }
})

function handleFileChange(file) {
  const rawFile = file.raw
  const maxSize = 500 * 1024 * 1024 // 500MB
  const allowedExt = ['mp4', 'avi', 'mov', 'mkv']
  const ext = (rawFile.name?.split('.').pop() || '').toLowerCase()

  if (rawFile.size > maxSize) {
    ElMessage.error('文件大小不能超过 500MB')
    return
  }
  if (!allowedExt.includes(ext)) {
    ElMessage.error('仅支持 MP4、AVI、MOV、MKV 格式')
    return
  }

  if (selectedVideoUrl.value) {
    URL.revokeObjectURL(selectedVideoUrl.value)
  }
  selectedFile.value = rawFile
  selectedVideoUrl.value = URL.createObjectURL(rawFile)
}

function clearFile() {
  if (selectedVideoUrl.value) {
    URL.revokeObjectURL(selectedVideoUrl.value)
    selectedVideoUrl.value = ''
  }
  selectedFile.value = null
  uploadRef.value?.clearFiles()
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

async function startAnalysis() {
  if (!selectedFile.value) return

  uploading.value = true

  try {
    await analysisStore.uploadVideo(selectedFile.value, viewAngle.value)
    ElMessage.success('上传成功，开始分析')

    // 开始轮询状态
    pollStatus()
  } catch (error) {
    ElMessage.error('上传失败: ' + (error.response?.data?.error || error.message))
  } finally {
    uploading.value = false
  }
}

async function pollStatus() {
  if (!analysisStore.currentTaskId) return

  if (pollTimer) clearInterval(pollTimer)
  pollTimer = setInterval(async () => {
    try {
      await analysisStore.fetchTaskStatus()

      if (analysisStore.taskStatus === 'completed' || analysisStore.taskStatus === 'failed') {
        clearInterval(pollTimer)
        pollTimer = null
      }
    } catch (error) {
      console.error('Poll error:', error)
    }
  }, 2000)
}

function isStageActive(index) {
  return currentStageIndex.value === index
}

function isStageCompleted(index) {
  return currentStageIndex.value >= index
}

function viewResult() {
  if (analysisStore.resultRecordId) {
    router.push(`/result/${analysisStore.resultRecordId}`)
  } else {
    ElMessage.warning('结果记录尚未生成，请稍后再试')
  }
}

function resetAnalysis() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
  analysisStore.reset()
  clearFile()
}
</script>

<style lang="scss" scoped>
.analyze-page {
  max-width: 800px;
  margin: 0 auto;
}

// 上传区域
.upload-section {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.upload-card {
  background: #fff;
  border-radius: 12px;
  padding: 32px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.video-preview {
  margin-top: 16px;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  background: #000;

  video {
    width: 100%;
    max-height: 320px;
    display: block;
  }
}

.video-uploader {
  :deep(.el-upload-dragger) {
    border: 2px dashed #dcdfe6;
    border-radius: 12px;
    padding: 48px 32px;
    transition: all 0.3s ease;

    &:hover {
      border-color: #e67e22;
      background: rgba(230, 126, 34, 0.02);
    }
  }
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
}

.upload-icon {
  width: 72px;
  height: 72px;
  background: linear-gradient(135deg, rgba(230, 126, 34, 0.1), rgba(230, 126, 34, 0.05));
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;

  .el-icon {
    font-size: 32px;
    color: #e67e22;
  }
}

.upload-text {
  text-align: center;

  .primary {
    font-size: 16px;
    font-weight: 500;
    color: #2c3e50;
    margin: 0 0 4px;
  }

  .secondary {
    font-size: 14px;
    color: #909399;
    margin: 0;
  }
}

.upload-hint {
  font-size: 12px;
  color: #c0c4cc;
}

.file-preview {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 24px;
  background: #f8f9fa;
  border-radius: 10px;
}

.file-icon {
  width: 48px;
  height: 48px;
  background: linear-gradient(135deg, #e67e22, #d35400);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;

  .el-icon {
    font-size: 24px;
    color: #fff;
  }
}

.file-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.file-name {
  font-size: 14px;
  font-weight: 500;
  color: #2c3e50;
}

.file-size {
  font-size: 12px;
  color: #909399;
}

// 配置区域
.config-section {
  background: #fff;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
}

.config-title {
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 20px;
}

.config-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 24px;
}

.config-item {
  label {
    display: block;
    font-size: 14px;
    font-weight: 500;
    color: #606266;
    margin-bottom: 8px;
  }
}

.view-radio {
  :deep(.el-radio-button__inner) {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 12px 24px;
  }
}

.view-hint {
  .hint-content {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: #f0f9ff;
    border-radius: 8px;
    font-size: 13px;
    color: #606266;

    .el-icon {
      color: #409eff;
      flex-shrink: 0;
    }
  }
}

.action-buttons {
  display: flex;
  justify-content: center;

  .el-button {
    min-width: 160px;
    height: 48px;
    font-size: 15px;
    font-weight: 500;
  }
}

// 进度区域
.progress-section {
  display: flex;
  justify-content: center;
  padding: 40px 0;
}

.progress-card {
  background: #fff;
  border-radius: 16px;
  padding: 48px;
  width: 100%;
  max-width: 500px;
  text-align: center;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);
}

.progress-header {
  margin-bottom: 32px;

  h2 {
    font-size: 20px;
    font-weight: 600;
    color: #2c3e50;
    margin: 16px 0 8px;
  }

  p {
    font-size: 14px;
    color: #909399;
    margin: 0;
  }
}

.progress-icon {
  display: inline-flex;
}

.spinner {
  width: 56px;
  height: 56px;
  border: 3px solid #f0f0f0;
  border-top-color: #e67e22;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.progress-bar-wrapper {
  position: relative;
  margin-bottom: 32px;
}

.progress-text {
  position: absolute;
  right: 0;
  top: -24px;
  font-size: 14px;
  font-weight: 600;
  color: #e67e22;
}

.progress-stages {
  display: flex;
  justify-content: space-between;
}

.stage-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  flex: 1;

  &.active .stage-dot {
    background: #e67e22;
    color: #fff;
    transform: scale(1.1);
  }

  &.completed .stage-dot {
    background: #67c23a;
    color: #fff;
  }
}

.stage-dot {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #f0f0f0;
  color: #909399;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.3s ease;

  .el-icon {
    font-size: 14px;
  }
}

.stage-label {
  font-size: 11px;
  color: #909399;
  white-space: nowrap;
}

// 完成/失败区域
.complete-section,
.error-section {
  display: flex;
  justify-content: center;
  padding: 40px 0;
}

.complete-card,
.error-card {
  background: #fff;
  border-radius: 16px;
  padding: 48px;
  width: 100%;
  max-width: 400px;
  text-align: center;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08);

  h2 {
    font-size: 20px;
    font-weight: 600;
    color: #2c3e50;
    margin: 16px 0 8px;
  }

  p {
    font-size: 14px;
    color: #909399;
    margin: 0 0 24px;
  }
}

.complete-icon {
  .el-icon {
    font-size: 64px;
    color: #67c23a;
  }
}

.error-icon {
  .el-icon {
    font-size: 64px;
    color: #f56c6c;
  }
}

.complete-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
}
</style>
