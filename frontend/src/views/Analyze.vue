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
          <p>{{ displayStageMessage }}</p>
        </div>

        <div class="progress-bar-wrapper">
          <el-progress
            class="analysis-progress"
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
            :class="{ active: isStageActive(index), completed: isStagePassed(index) }"
          >
            <div class="stage-dot">
              <el-icon v-if="isStagePassed(index)"><Check /></el-icon>
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
import { ref, computed, watch, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAnalysisStore } from '@/stores/analysis'
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

const uploadRef = ref(null)
const selectedFile = ref(null)
const selectedVideoUrl = ref('')
const viewAngle = ref('side')
const uploading = ref(false)
const animatedProgress = ref(5)
const holdAnalyzingAfterDone = ref(false)
const completionVirtualActive = ref(false)
let pollTimer = null
let progressTimer = null
let completionHoldTimer = null

const stages = [
  { label: '视频准备', key: 's1' },
  { label: '姿态识别', key: 's2' },
  { label: '动作分析', key: 's3' },
  { label: '结果生成', key: 's4' }
]

const isAnalyzing = computed(() =>
  analysisStore.taskStatus === 'pending' ||
  analysisStore.taskStatus === 'running' ||
  (analysisStore.taskStatus === 'completed' && holdAnalyzingAfterDone.value)
)

const isCompleted = computed(() => analysisStore.taskStatus === 'completed' && !holdAnalyzingAfterDone.value)
const isFailed = computed(() => analysisStore.taskStatus === 'failed')

function mapSevenToFour(raw) {
  const n = Number(raw || 0)
  if (n <= 1) return 1
  if (n <= 2) return 2
  if (n <= 6) return 3
  return 4
}

function stageIndexByProgress(progress) {
  const p = Number(progress || 0)
  if (p >= 92) return 3
  if (p >= 58) return 2
  if (p >= 24) return 1
  return 0
}

const progressTarget = computed(() => {
  if (analysisStore.taskStatus === 'completed') {
    if (completionVirtualActive.value) {
      return Math.max(animatedProgress.value, 92)
    }
    return 100
  }
  const base = Number(analysisStore.taskProgress || 0)
  if (analysisStore.taskStatus === 'failed') return Math.max(base, 10)
  return Math.max(base, 5)
})

const currentStageIndex = computed(() => {
  if (analysisStore.taskStatus === 'completed') return stages.length - 1
  const fromProgress = stageIndexByProgress(animatedProgress.value)
  const msg = analysisStore.taskMessage || ''
  const matched = msg.match(/([1-7])\/7/)
  if (matched) {
    const raw = parseInt(matched[1], 10)
    const fromStageText = mapSevenToFour(raw) - 1
    return Math.max(fromProgress, fromStageText)
  }
  if (msg.includes('完成') || msg.includes('保存') || msg.includes('写入')) {
    return Math.max(fromProgress, stages.length - 1)
  }
  return fromProgress
})

const displayProgress = computed(() => {
  return Math.max(0, Math.min(100, Math.round(animatedProgress.value)))
})

const displayStageMessage = computed(() => {
  if (analysisStore.taskStatus === 'failed') {
    return analysisStore.taskMessage || '分析失败'
  }
  if (analysisStore.taskStatus === 'completed' && holdAnalyzingAfterDone.value) {
    return '4/4 正在生成最终结果...'
  }

  const msg = analysisStore.taskMessage || ''
  if (!msg) return '1/4 正在准备分析...'
  const matched = msg.match(/([1-7])\/7/)
  if (!matched) return msg
  const mapped = mapSevenToFour(parseInt(matched[1], 10))
  return msg.replace(/([1-7])\/7/, `${mapped}/4`)
})

watch(progressTarget, (target) => {
  if (target + 6 < animatedProgress.value) {
    animatedProgress.value = target
  }
}, { immediate: true })

watch(() => analysisStore.taskStatus, (status, prev) => {
  if (status === 'completed' && prev !== 'completed') {
    holdAnalyzingAfterDone.value = true
    completionVirtualActive.value = true
    animatedProgress.value = Math.max(animatedProgress.value, 92)
    return
  }
  if (status !== 'completed') {
    holdAnalyzingAfterDone.value = false
    completionVirtualActive.value = false
  }
})

function ensureProgressTimer() {
  if (progressTimer) return
  progressTimer = setInterval(() => {
    if (completionVirtualActive.value && analysisStore.taskStatus === 'completed') {
      const current = animatedProgress.value
      const step = current < 96
        ? (Math.random() * 1.8 + 0.7)
        : (Math.random() * 0.6 + 0.15)
      animatedProgress.value = Math.min(100, current + step)
      if (animatedProgress.value >= 100) {
        completionVirtualActive.value = false
        if (completionHoldTimer) clearTimeout(completionHoldTimer)
        completionHoldTimer = setTimeout(() => {
          holdAnalyzingAfterDone.value = false
        }, 260)
      }
      return
    }

    const target = progressTarget.value
    const current = animatedProgress.value
    const diff = target - current
    if (Math.abs(diff) < 0.2) {
      animatedProgress.value = target
      return
    }
    const step = Math.min(3.2, Math.max(0.5, Math.abs(diff) * 0.22))
    animatedProgress.value = current + (diff > 0 ? step : -step)
  }, 90)
}

ensureProgressTimer()

onUnmounted(() => {
  if (progressTimer) {
    clearInterval(progressTimer)
    progressTimer = null
  }
  if (completionHoldTimer) {
    clearTimeout(completionHoldTimer)
    completionHoldTimer = null
  }
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
  animatedProgress.value = 5
  holdAnalyzingAfterDone.value = false
  completionVirtualActive.value = false

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
  }, 1200)
}

function isStageActive(index) {
  return currentStageIndex.value === index && analysisStore.taskStatus !== 'completed'
}

function isStagePassed(index) {
  if (analysisStore.taskStatus === 'completed') {
    return currentStageIndex.value >= index
  }
  return currentStageIndex.value > index
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

.progress-bar-wrapper :deep(.analysis-progress .el-progress-bar__outer) {
  overflow: hidden;
  background: #f7ece1;
}

.progress-bar-wrapper :deep(.analysis-progress .el-progress-bar__inner) {
  background: linear-gradient(110deg, #d35400 0%, #e67e22 35%, #f39c12 55%, #e67e22 80%, #d35400 100%) !important;
  background-size: 220% 100% !important;
  animation: progressFlow 1.8s linear infinite;
  box-shadow: 0 0 10px rgba(230, 126, 34, 0.28);
  transition: width 0.4s ease-out;
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
  gap: 8px;
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

.stage-item.active .stage-label {
  color: #d35400;
  font-weight: 600;
}

@keyframes progressFlow {
  0% {
    background-position: 0% 0;
  }
  100% {
    background-position: 220% 0;
  }
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
