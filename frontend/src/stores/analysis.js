import { defineStore } from 'pinia'
import { ref } from 'vue'
import { analysisApi } from '@/api/analysis'

export const useAnalysisStore = defineStore('analysis', () => {
  const currentTaskId = ref(null)
  const taskStatus = ref(null)
  const taskProgress = ref(0)
  const taskMessage = ref('')
  const result = ref(null)
  const resultRecordId = ref(null)

  async function uploadVideo(file, viewAngle, config = {}) {
    const response = await analysisApi.uploadVideo(file, viewAngle, config)
    currentTaskId.value = response.task_id
    taskStatus.value = 'pending'
    taskProgress.value = 0
    taskMessage.value = response.message || '已提交'
    return response
  }

  async function fetchTaskStatus() {
    if (!currentTaskId.value) return null
    const response = await analysisApi.getTaskStatus(currentTaskId.value)
    const backendStatus = response.status
    taskStatus.value = backendStatus === 'succeeded' ? 'completed' : backendStatus
    taskProgress.value = response.progress || 0
    taskMessage.value = response.stage || response.error_message || ''
    if (response.result?.record_id) {
      resultRecordId.value = response.result.record_id
    }
    return response
  }

  async function fetchResult(recordId = null) {
    const id = recordId || resultRecordId.value
    if (!id) return null
    const response = await analysisApi.getResult(id)
    result.value = response.record
    return result.value
  }

  function reset() {
    currentTaskId.value = null
    taskStatus.value = null
    taskProgress.value = 0
    taskMessage.value = ''
    resultRecordId.value = null
    result.value = null
  }

  return {
    currentTaskId,
    taskStatus,
    taskProgress,
    taskMessage,
    resultRecordId,
    result,
    uploadVideo,
    fetchTaskStatus,
    fetchResult,
    reset
  }
})

