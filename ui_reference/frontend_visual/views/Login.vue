<template>
  <div class="auth-container">
    <!-- 左侧介绍区域 -->
    <div class="intro-section">
      <div class="intro-background">
        <svg class="motion-lines" viewBox="0 0 400 600" preserveAspectRatio="none">
          <path class="line line-1" d="M0,100 Q100,150 200,100 T400,100" />
          <path class="line line-2" d="M0,200 Q150,250 300,200 T400,180" />
          <path class="line line-3" d="M0,350 Q120,400 250,350 T400,320" />
          <path class="line line-4" d="M0,480 Q180,530 350,480 T400,450" />
        </svg>
      </div>

      <div class="intro-content">
        <div class="logo-section">
          <div class="logo-mark">
            <svg viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="12" r="6" stroke="currentColor" stroke-width="2"/>
              <path d="M24 18 L24 28 M24 28 L16 42 M24 28 L32 42 M18 24 L30 24"
                    stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            </svg>
          </div>
          <h1 class="system-title">跑步动作分析系统</h1>
          <p class="system-subtitle">Running Gait Analysis System</p>
        </div>

        <div class="features">
          <div class="feature-item">
            <div class="feature-icon">
              <el-icon><VideoCamera /></el-icon>
            </div>
            <div class="feature-text">
              <h3>视频姿态估计</h3>
              <p>基于 MediaPipe 的高精度 2D 关键点检测</p>
            </div>
          </div>

          <div class="feature-item">
            <div class="feature-icon">
              <el-icon><TrendCharts /></el-icon>
            </div>
            <div class="feature-text">
              <h3>运动学分析</h3>
              <p>步频、膝关节角度、垂直振幅等核心指标</p>
            </div>
          </div>

          <div class="feature-item">
            <div class="feature-icon">
              <el-icon><DataAnalysis /></el-icon>
            </div>
            <div class="feature-text">
              <h3>质量评价体系</h3>
              <p>稳定性、效率、跑姿三维度综合评分</p>
            </div>
          </div>

          <div class="feature-item">
            <div class="feature-icon">
              <el-icon><Document /></el-icon>
            </div>
            <div class="feature-text">
              <h3>智能分析报告</h3>
              <p>AI 辅助生成专业改进建议</p>
            </div>
          </div>
        </div>

        <div class="tech-badge">
          <span>MotionBERT</span>
          <span>·</span>
          <span>3D Pose Lifting</span>
          <span>·</span>
          <span>Deep Learning</span>
        </div>
      </div>
    </div>

    <!-- 右侧登录表单 -->
    <div class="form-section">
      <div class="form-wrapper">
        <div class="form-header">
          <h2>{{ isLogin ? '登录' : '注册' }}</h2>
          <p>{{ isLogin ? '登录以访问您的分析记录' : '创建账户开始使用系统' }}</p>
        </div>

        <el-form
          ref="formRef"
          :model="formData"
          :rules="rules"
          class="auth-form"
          @submit.prevent="handleSubmit"
        >
          <el-form-item prop="username">
            <el-input
              v-model="formData.username"
              placeholder="用户名"
              size="large"
              :prefix-icon="User"
            />
          </el-form-item>

          <el-form-item prop="password">
            <el-input
              v-model="formData.password"
              type="password"
              placeholder="密码"
              size="large"
              :prefix-icon="Lock"
              show-password
            />
          </el-form-item>

          <el-form-item v-if="!isLogin" prop="confirmPassword">
            <el-input
              v-model="formData.confirmPassword"
              type="password"
              placeholder="确认密码"
              size="large"
              :prefix-icon="Lock"
              show-password
            />
          </el-form-item>

          <el-form-item>
            <el-button
              type="primary"
              size="large"
              class="submit-btn"
              :loading="loading"
              @click="handleSubmit"
            >
              {{ isLogin ? '登 录' : '注 册' }}
            </el-button>
          </el-form-item>
        </el-form>

        <div class="form-footer">
          <span>{{ isLogin ? '还没有账户？' : '已有账户？' }}</span>
          <el-button type="primary" link @click="toggleMode">
            {{ isLogin ? '立即注册' : '返回登录' }}
          </el-button>
        </div>
      </div>

      <div class="form-decoration">
        <div class="decoration-line"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock, VideoCamera, TrendCharts, DataAnalysis, Document } from '@element-plus/icons-vue'
import { useUserStore } from '@/stores/user'

const router = useRouter()
const route = useRoute()
const userStore = useUserStore()

const formRef = ref(null)
const isLogin = ref(true)
const loading = ref(false)

const formData = reactive({
  username: '',
  password: '',
  confirmPassword: ''
})

const validateConfirmPassword = (rule, value, callback) => {
  if (!isLogin.value && value !== formData.password) {
    callback(new Error('两次输入的密码不一致'))
  } else {
    callback()
  }
}

const rules = computed(() => ({
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, message: '用户名至少3个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码至少6个字符', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: !isLogin.value, message: '请确认密码', trigger: 'blur' },
    { validator: validateConfirmPassword, trigger: 'blur' }
  ]
}))

function toggleMode() {
  isLogin.value = !isLogin.value
  formData.confirmPassword = ''
  formRef.value?.clearValidate()
}

async function handleSubmit() {
  const valid = await formRef.value?.validate().catch(() => false)
  if (!valid) return

  loading.value = true

  try {
    if (isLogin.value) {
      await userStore.login(formData.username, formData.password)
      ElMessage.success('登录成功')
      const redirect = route.query.redirect || '/'
      router.push(redirect)
    } else {
      await userStore.register(formData.username, formData.password)
      ElMessage.success('注册成功，请登录')
      isLogin.value = true
      formData.password = ''
      formData.confirmPassword = ''
      return
    }
  } catch (error) {
    ElMessage.error(error.response?.data?.error || '操作失败')
  } finally {
    loading.value = false
  }
}
</script>

<style lang="scss" scoped>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;600;700&family=Noto+Sans+SC:wght@300;400;500&display=swap');

.auth-container {
  display: flex;
  min-height: 100vh;
  background: #faf9f7;
}

// 左侧介绍区域
.intro-section {
  flex: 1;
  position: relative;
  background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
  padding: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.intro-background {
  position: absolute;
  inset: 0;
  opacity: 0.15;
}

.motion-lines {
  width: 100%;
  height: 100%;

  .line {
    fill: none;
    stroke: #fff;
    stroke-width: 1.5;
    stroke-linecap: round;
  }

  .line-1 {
    animation: wave 8s ease-in-out infinite;
  }
  .line-2 {
    animation: wave 10s ease-in-out infinite 0.5s;
  }
  .line-3 {
    animation: wave 9s ease-in-out infinite 1s;
  }
  .line-4 {
    animation: wave 11s ease-in-out infinite 1.5s;
  }
}

@keyframes wave {
  0%, 100% {
    d: path("M0,100 Q100,150 200,100 T400,100");
  }
  50% {
    d: path("M0,100 Q100,50 200,100 T400,120");
  }
}

.intro-content {
  position: relative;
  z-index: 1;
  max-width: 480px;
  color: #fff;
}

.logo-section {
  margin-bottom: 48px;
}

.logo-mark {
  width: 64px;
  height: 64px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);

  svg {
    width: 36px;
    height: 36px;
    color: #e67e22;
  }
}

.system-title {
  font-family: 'Noto Serif SC', serif;
  font-size: 32px;
  font-weight: 700;
  margin: 0 0 8px;
  letter-spacing: 2px;
}

.system-subtitle {
  font-family: 'Noto Sans SC', sans-serif;
  font-size: 14px;
  font-weight: 300;
  color: rgba(255, 255, 255, 0.6);
  letter-spacing: 1px;
  text-transform: uppercase;
}

.features {
  display: flex;
  flex-direction: column;
  gap: 24px;
  margin-bottom: 48px;
}

.feature-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  transition: all 0.3s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateX(4px);
  }
}

.feature-icon {
  width: 40px;
  height: 40px;
  background: rgba(230, 126, 34, 0.2);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;

  .el-icon {
    font-size: 20px;
    color: #e67e22;
  }
}

.feature-text {
  h3 {
    font-family: 'Noto Sans SC', sans-serif;
    font-size: 15px;
    font-weight: 500;
    margin: 0 0 4px;
  }

  p {
    font-family: 'Noto Sans SC', sans-serif;
    font-size: 13px;
    font-weight: 300;
    color: rgba(255, 255, 255, 0.6);
    margin: 0;
    line-height: 1.5;
  }
}

.tech-badge {
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: 'Noto Sans SC', sans-serif;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.4);
  letter-spacing: 0.5px;
}

// 右侧表单区域
.form-section {
  width: 480px;
  background: #fff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px;
  position: relative;
}

.form-wrapper {
  width: 100%;
  max-width: 320px;
}

.form-header {
  margin-bottom: 40px;
  text-align: center;

  h2 {
    font-family: 'Noto Serif SC', serif;
    font-size: 28px;
    font-weight: 600;
    color: #2c3e50;
    margin: 0 0 8px;
  }

  p {
    font-family: 'Noto Sans SC', sans-serif;
    font-size: 14px;
    color: #7f8c8d;
    margin: 0;
  }
}

.auth-form {
  :deep(.el-form-item) {
    margin-bottom: 20px;
  }

  :deep(.el-input) {
    --el-input-border-radius: 8px;

    .el-input__wrapper {
      padding: 4px 16px;
      box-shadow: 0 0 0 1px #e0e0e0;
      transition: all 0.3s ease;

      &:hover {
        box-shadow: 0 0 0 1px #bdc3c7;
      }

      &.is-focus {
        box-shadow: 0 0 0 2px rgba(230, 126, 34, 0.3);
      }
    }

    .el-input__inner {
      font-family: 'Noto Sans SC', sans-serif;
      font-size: 14px;
      height: 44px;
    }

    .el-input__prefix {
      color: #95a5a6;
    }
  }
}

.submit-btn {
  width: 100%;
  height: 48px;
  font-family: 'Noto Sans SC', sans-serif;
  font-size: 15px;
  font-weight: 500;
  letter-spacing: 4px;
  border-radius: 8px;
  background: linear-gradient(135deg, #e67e22, #d35400);
  border: none;
  transition: all 0.3s ease;

  &:hover {
    background: linear-gradient(135deg, #d35400, #c0392b);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(230, 126, 34, 0.3);
  }

  &:active {
    transform: translateY(0);
  }
}

.form-footer {
  margin-top: 32px;
  text-align: center;
  font-family: 'Noto Sans SC', sans-serif;
  font-size: 14px;
  color: #7f8c8d;

  .el-button {
    font-family: 'Noto Sans SC', sans-serif;
    font-weight: 500;
    color: #e67e22;

    &:hover {
      color: #d35400;
    }
  }
}

.form-decoration {
  position: absolute;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%);
}

.decoration-line {
  width: 60px;
  height: 3px;
  background: linear-gradient(90deg, transparent, #e67e22, transparent);
  border-radius: 2px;
}

// 响应式
@media (max-width: 960px) {
  .auth-container {
    flex-direction: column;
  }

  .intro-section {
    padding: 40px 24px;
    min-height: auto;
  }

  .intro-content {
    max-width: 100%;
  }

  .features {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }

  .form-section {
    width: 100%;
    padding: 40px 24px;
  }
}

@media (max-width: 600px) {
  .features {
    grid-template-columns: 1fr;
  }

  .system-title {
    font-size: 24px;
  }
}
</style>
