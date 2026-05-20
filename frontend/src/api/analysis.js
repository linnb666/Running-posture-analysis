import api from './index'
import { normalizeGrade } from '@/utils/analysis'

function normalizeRecord(record = {}) {
  return {
    ...record,
    grade: normalizeGrade(record.grade || record.rating),
    view_type: record.view_type || record.view_angle,
    completed_at: record.completed_at || record.created_at,
    ai_report: record.ai_report || record.ai_analysis,
    kinematic: record.kinematic || record.kinematic_results || {},
    temporal: record.temporal || record.temporal_results || {},
    status: record.status || 'completed'
  }
}

export const analysisApi = {
  uploadVideo(file, viewAngle = 'side', config = {}) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('view_angle', viewAngle)
    formData.append('enable_3d', String(config.enable_3d ?? true))
    return api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 120000
    })
  },

  getTaskStatus(taskId) {
    return api.get(`/task/${taskId}`)
  },

  getResult(recordId) {
    return api.get(`/result/${recordId}`).then((resp) => ({
      ...resp,
      record: normalizeRecord(resp.record)
    }))
  },

  generateAIReport(recordId) {
    return api.post(`/result/${recordId}/ai`, null, { timeout: 300000 })
  },

  generateLocalReport(recordId) {
    return api.post(`/result/${recordId}/local-report`, null, { timeout: 120000 })
  },

  saveManualNotes(recordId, manualNotes) {
    return api.post(`/result/${recordId}/notes`, { manual_notes: manualNotes }, { timeout: 30000 })
  },

  downloadPdfReport(recordId) {
    return api.get(`/result/${recordId}/pdf`, {
      responseType: 'blob',
      timeout: 120000,
      _rawResponse: true
    }).then((response) => {
      const disposition = response.headers?.['content-disposition'] || ''
      const match = disposition.match(/filename\*?=(?:UTF-8''|")?([^";]+)/i)
      const filename = match ? decodeURIComponent(match[1].replace(/"/g, '')) : `record_${recordId}_analysis_report.pdf`
      return { blob: response.data, filename }
    })
  },

  getPdfPreviewUrl(recordId, options = {}) {
    const params = new URLSearchParams()
    if (options.accessToken) params.set('access_token', options.accessToken)
    if (options.autoprint) params.set('autoprint', '1')
    const query = params.toString()
    return `/api/result/${recordId}/pdf-preview${query ? `?${query}` : ''}`
  },

  getMediaBlob(mediaUrl, options = {}) {
    if (!mediaUrl) return Promise.resolve(null)
    const path = mediaUrl.startsWith('/api') ? mediaUrl.slice(4) : mediaUrl
    return api.get(path, {
      responseType: 'blob',
      timeout: options.timeout ?? 300000
    })
  }
}
