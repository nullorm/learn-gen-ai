import { describe, test, expect } from 'bun:test'
import {
  getRank,
  nextRank,
  updateStreak,
  checkBadges,
  startModule,
  recordQuiz,
  recordExercise,
  completeModule,
} from '../../tools/progress-engine.js'
import { INITIAL_STATE, type ProgressState } from '../../tools/progress-types.js'

function freshState(): ProgressState {
  return structuredClone(INITIAL_STATE)
}

describe('getRank', () => {
  test('returns Token at 0 XP', () => {
    expect(getRank(0)).toBe('Token')
  })
  test('returns Prompter at 100 XP', () => {
    expect(getRank(100)).toBe('Prompter')
  })
  test('returns LLM Architect at 3000+ XP', () => {
    expect(getRank(5000)).toBe('LLM Architect')
  })
})

describe('nextRank', () => {
  test('returns Prompter as next from 0 XP', () => {
    const next = nextRank(0)
    expect(next?.name).toBe('Prompter')
    expect(next?.xpNeeded).toBe(100)
  })
  test('returns null at max rank', () => {
    expect(nextRank(3000)).toBeNull()
  })
})

describe('updateStreak', () => {
  test('starts streak at 1 on first activity', () => {
    const state = freshState()
    updateStreak(state, '2026-03-10')
    expect(state.streak_days).toBe(1)
    expect(state.last_active).toBe('2026-03-10')
  })
  test('increments streak on consecutive day', () => {
    const state = freshState()
    state.last_active = '2026-03-09'
    state.streak_days = 3
    updateStreak(state, '2026-03-10')
    expect(state.streak_days).toBe(4)
  })
  test('resets streak on gap', () => {
    const state = freshState()
    state.last_active = '2026-03-07'
    state.streak_days = 5
    updateStreak(state, '2026-03-10')
    expect(state.streak_days).toBe(1)
  })
  test('no-ops on same day', () => {
    const state = freshState()
    state.last_active = '2026-03-10'
    state.streak_days = 3
    updateStreak(state, '2026-03-10')
    expect(state.streak_days).toBe(3)
  })
})

describe('recordQuiz', () => {
  test('awards XP for correct answers', () => {
    const state = freshState()
    recordQuiz(state, 1, 4, 5)
    expect(state.xp).toBe(80)
    expect(state.modules['1'].quiz_score).toBe(4)
  })
  test('awards bonus for perfect quiz', () => {
    const state = freshState()
    recordQuiz(state, 1, 5, 5)
    expect(state.xp).toBe(150)
  })
})

describe('recordExercise', () => {
  test('awards XP and tracks exercise', () => {
    const state = freshState()
    recordExercise(state, 1, 1)
    expect(state.xp).toBe(30)
    expect(state.modules['1'].exercises).toEqual([1])
  })
  test('does not double-count same exercise', () => {
    const state = freshState()
    recordExercise(state, 1, 1)
    recordExercise(state, 1, 1)
    expect(state.xp).toBe(30)
  })
})

describe('completeModule', () => {
  test('marks module completed and awards XP', () => {
    const state = freshState()
    completeModule(state, 1)
    expect(state.modules['1'].status).toBe('completed')
    expect(state.xp).toBe(100)
  })
})

describe('checkBadges', () => {
  test('awards first_call badge on module 1 completion', () => {
    const state = freshState()
    state.modules['1'] = { status: 'completed', exercises: [] }
    const newBadges = checkBadges(state)
    expect(newBadges).toContain('first_call')
  })
  test('awards prompt_whisperer on perfect module 2 quiz', () => {
    const state = freshState()
    state.modules['2'] = { status: 'in_progress', quiz_score: 5, quiz_total: 5, exercises: [] }
    const newBadges = checkBadges(state)
    expect(newBadges).toContain('prompt_whisperer')
  })
  test('awards perfect_score on any perfect quiz', () => {
    const state = freshState()
    state.modules['5'] = { status: 'in_progress', quiz_score: 5, quiz_total: 5, exercises: [] }
    const newBadges = checkBadges(state)
    expect(newBadges).toContain('perfect_score')
  })
  test('awards streak_7 at 7 day streak', () => {
    const state = freshState()
    state.streak_days = 7
    const newBadges = checkBadges(state)
    expect(newBadges).toContain('streak_7')
  })
})
