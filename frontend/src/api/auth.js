import api from './index'

const localErrorConfig = { _skipGlobalError: true }

export const authApi = {
  register(username, password, email = '') {
    return api.post('/auth/register', { username, password, email }, localErrorConfig)
  },

  login(username, password) {
    return api.post('/auth/login', { username, password }, localErrorConfig)
  },

  refresh(refreshToken) {
    return api.post('/auth/refresh', { refresh_token: refreshToken })
  },

  me() {
    return api.get('/auth/me')
  },

  logout() {
    return api.post('/auth/logout')
  }
}
