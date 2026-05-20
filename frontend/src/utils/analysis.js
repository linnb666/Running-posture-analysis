export function normalizeGrade(value) {
  const v = String(value || '').toLowerCase()
  if (['excellent', 'good', 'fair', 'poor'].includes(v)) return v
  const map = {
    优秀: 'excellent',
    良好: 'good',
    中等: 'fair',
    一般: 'fair',
    较差: 'poor'
  }
  return map[v] || 'fair'
}

export function gradeLabel(grade) {
  const v = normalizeGrade(grade)
  const map = {
    excellent: '优秀',
    good: '良好',
    fair: '一般',
    poor: '待提升'
  }
  return map[v] || '一般'
}

export function viewLabel(view) {
  const v = String(view || '').toLowerCase()
  if (v === 'front') return '正面'
  if (v === 'side') return '侧面'
  return view || '--'
}

