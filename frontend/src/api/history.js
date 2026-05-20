import api from './index'
import { normalizeGrade } from '@/utils/analysis'

function normalizeRecord(record = {}) {
  return {
    ...record,
    view_type: record.view_type || record.view_angle,
    grade: normalizeGrade(record.grade || record.rating),
    status: record.status || 'completed'
  }
}

export const historyApi = {
  getHistory(params = {}) {
    return api.get('/history', { params }).then((resp) => ({
      ...resp,
      records: (resp.records || []).map(normalizeRecord)
    }))
  },

  getHistoryDetail(id) {
    return api.get(`/result/${id}`).then((resp) => ({
      ...resp,
      record: normalizeRecord(resp.record)
    }))
  },

  deleteHistory(id) {
    return api.delete(`/result/${id}`)
  },

  renameHistory(id, videoFilename) {
    return api.post(`/result/${id}/rename`, { video_filename: videoFilename }).then((resp) => ({
      ...resp,
      record: normalizeRecord(resp.record || {})
    }))
  },

  getStatistics() {
    return api.get('/statistics').then((resp) => resp.statistics || {})
  }
}
