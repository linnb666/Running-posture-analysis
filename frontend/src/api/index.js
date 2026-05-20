import axios from 'axios'
import { ElMessage } from 'element-plus'
import router from '@/router'
import { useUserStore } from '@/stores/user'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

api.interceptors.request.use(
  (config) => {
    const userStore = useUserStore()
    if (userStore.token) {
      config.headers.Authorization = `Bearer ${userStore.token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

api.interceptors.response.use(
  (response) => (response.config?._rawResponse ? response : response.data),
  async (error) => {
    const response = error.response
    const reqConfig = error.config || response?.config || {}
    if (reqConfig._skipGlobalError) {
      return Promise.reject(error)
    }
    if (!response) {
      ElMessage.error('网络连接失败')
      return Promise.reject(error)
    }

    if (response.status === 401) {
      const userStore = useUserStore()
      userStore.logout()
      if (router.currentRoute.value.name !== 'Login') {
        router.push({ name: 'Login' })
      }
      ElMessage.error(response.data?.error || '登录已过期，请重新登录')
    } else if (response.status === 403) {
      ElMessage.error('没有权限访问')
    } else if (response.status === 404) {
      ElMessage.error(response.data?.error || '资源不存在')
    } else if (response.status >= 500) {
      ElMessage.error(response.data?.error || '服务器错误')
    } else {
      ElMessage.error(response.data?.error || '请求失败')
    }
    return Promise.reject(error)
  }
)

export default api
