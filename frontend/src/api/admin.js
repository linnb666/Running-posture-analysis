import api from './index'

export const adminApi = {
  getOverview() {
    return api.get('/admin/overview').then((resp) => resp.overview || {})
  },

  getUsers(params = {}) {
    return api.get('/admin/users', { params })
  },

  updateUser(userId, payload) {
    return api.patch(`/admin/users/${userId}`, payload)
  },

  resetPassword(userId, newPassword) {
    return api.post(`/admin/users/${userId}/reset-password`, { new_password: newPassword })
  },

  getUserRecords(userId, params = {}) {
    return api.get(`/admin/users/${userId}/records`, { params })
  },

  hardDeleteRecord(recordId) {
    return api.delete(`/admin/records/${recordId}/hard-delete`)
  },

  hardDeleteRecordsBatch(recordIds) {
    return api.post('/admin/records/hard-delete-batch', { record_ids: recordIds })
  },

  hardDeleteUser(userId) {
    return api.delete(`/admin/users/${userId}/hard-delete`)
  },

  hardDeleteUsersBatch(userIds) {
    return api.post('/admin/users/hard-delete-batch', { user_ids: userIds })
  },

  cleanupOrphanStorage() {
    return api.post('/admin/storage/cleanup-orphans')
  },

  cleanupDanglingTasks() {
    return api.post('/admin/storage/cleanup-dangling-tasks')
  },

  cleanupStaleQueuedTasks(minAgeMinutes = 10) {
    return api.post('/admin/storage/cleanup-stale-queued', { min_age_minutes: minAgeMinutes })
  },

  getAuditLogs(params = {}) {
    return api.get('/admin/audit-logs', { params })
  }
}
