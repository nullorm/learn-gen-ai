// tools/progress-engine.ts
import { RANKS, XP, PARTS, PART_BADGES, MODULE_NAMES, type ProgressState } from './progress-types.js'

export function getRank(xp: number): string {
  let rank = RANKS[0].name
  for (const r of RANKS) {
    if (xp >= r.xp) rank = r.name
  }
  return rank
}

export function nextRank(xp: number): { name: string; xpNeeded: number } | null {
  for (const r of RANKS) {
    if (xp < r.xp) return { name: r.name, xpNeeded: r.xp - xp }
  }
  return null
}

export function updateStreak(state: ProgressState, today: string): void {
  if (state.last_active === today) return
  if (state.last_active) {
    const last = new Date(state.last_active)
    const now = new Date(today)
    const diffMs = now.getTime() - last.getTime()
    const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24))
    if (diffDays === 1) {
      state.streak_days += 1
    } else {
      state.streak_days = 1
    }
  } else {
    state.streak_days = 1
  }
  state.last_active = today
}

function addBadge(state: ProgressState, badge: string): boolean {
  if (state.badges.includes(badge)) return false
  state.badges.push(badge)
  return true
}

export function checkBadges(state: ProgressState): string[] {
  const newBadges: string[] = []

  // First Call — complete module 1
  if (state.modules['1']?.status === 'completed') {
    if (addBadge(state, 'first_call')) newBadges.push('first_call')
  }

  // Prompt Whisperer — perfect score on module 2
  if (
    state.modules['2']?.quiz_score === state.modules['2']?.quiz_total &&
    state.modules['2']?.quiz_score !== undefined
  ) {
    if (addBadge(state, 'prompt_whisperer')) newBadges.push('prompt_whisperer')
  }

  // Type Safe — complete module 3
  if (state.modules['3']?.status === 'completed') {
    if (addBadge(state, 'type_safe')) newBadges.push('type_safe')
  }

  // Vector Space — complete module 8
  if (state.modules['8']?.status === 'completed') {
    if (addBadge(state, 'vector_space')) newBadges.push('vector_space')
  }

  // Perfect Score — any quiz with perfect score
  for (const mod of Object.values(state.modules)) {
    if (mod.quiz_score !== undefined && mod.quiz_score === mod.quiz_total) {
      if (addBadge(state, 'perfect_score')) newBadges.push('perfect_score')
      break
    }
  }

  // Full Stack LLM — all 24 modules
  const totalModules = Object.keys(MODULE_NAMES).length
  const completed = Object.values(state.modules).filter(m => m.status === 'completed').length
  if (completed === totalModules) {
    if (addBadge(state, 'full_stack_llm')) newBadges.push('full_stack_llm')
  }

  // Streak badges
  if (state.streak_days >= 7) {
    if (addBadge(state, 'streak_7')) newBadges.push('streak_7')
  }
  if (state.streak_days >= 30) {
    if (addBadge(state, 'streak_30')) newBadges.push('streak_30')
  }

  // Part completion badges
  for (const [part, { modules }] of Object.entries(PARTS)) {
    if (modules.every(m => state.modules[String(m)]?.status === 'completed')) {
      const badgeName = PART_BADGES[part].toLowerCase().replace(/ /g, '_')
      if (addBadge(state, badgeName)) newBadges.push(badgeName)
    }
  }

  return newBadges
}

export function startModule(state: ProgressState, moduleNum: number): void {
  const key = String(moduleNum)
  if (!state.modules[key]) {
    state.modules[key] = { status: 'in_progress', exercises: [] }
  } else if (state.modules[key].status === 'not_started') {
    state.modules[key].status = 'in_progress'
  }
}

export function recordQuiz(state: ProgressState, moduleNum: number, score: number, total: number): void {
  const key = String(moduleNum)
  if (!state.modules[key]) {
    state.modules[key] = { status: 'in_progress', exercises: [] }
  }
  state.modules[key].quiz_score = score
  state.modules[key].quiz_total = total
  state.xp += score * XP.CORRECT_ANSWER
  if (score === total) {
    state.xp += XP.PERFECT_QUIZ_BONUS
  }
}

export function recordExercise(state: ProgressState, moduleNum: number, exerciseNum: number): void {
  const key = String(moduleNum)
  if (!state.modules[key]) {
    state.modules[key] = { status: 'in_progress', exercises: [] }
  }
  if (!state.modules[key].exercises.includes(exerciseNum)) {
    state.modules[key].exercises.push(exerciseNum)
    state.xp += XP.EXERCISE
  }
}

export function completeModule(state: ProgressState, moduleNum: number): void {
  const key = String(moduleNum)
  if (!state.modules[key]) {
    state.modules[key] = { status: 'completed', exercises: [] }
  } else {
    state.modules[key].status = 'completed'
  }
  state.xp += XP.MODULE_COMPLETE
}
