import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { authApi } from '@/api/auth'

export const useUserStore = defineStore('user', () => {
  const user = ref(null)
  const token = ref(localStorage.getItem('token') || null)
  const refreshToken = ref(localStorage.getItem('refreshToken') || null)

  const isLoggedIn = computed(() => !!token.value)
  const username = computed(() => user.value?.username || '')
  const isAdmin = computed(() => !!user.value?.is_admin)
  const isActive = computed(() => user.value?.is_active !== false)

  async function login(usernameValue, password) {
    const response = await authApi.login(usernameValue, password)
    setAuth(response)
    return response
  }

  async function register(usernameValue, password, email = '') {
    return authApi.register(usernameValue, password, email)
  }

  function setAuth(payload) {
    user.value = payload.user
    token.value = payload.access_token
    refreshToken.value = payload.refresh_token
    localStorage.setItem('token', payload.access_token)
    localStorage.setItem('refreshToken', payload.refresh_token)
  }

  async function fetchUser() {
    if (!token.value) return null
    try {
      const resp = await authApi.me()
      user.value = resp.user
      return user.value
    } catch (_err) {
      logout()
      return null
    }
  }

  function logout() {
    user.value = null
    token.value = null
    refreshToken.value = null
    localStorage.removeItem('token')
    localStorage.removeItem('refreshToken')
  }

  return {
    user,
    token,
    refreshToken,
    isLoggedIn,
    username,
    isAdmin,
    isActive,
    login,
    register,
    fetchUser,
    logout
  }
})
