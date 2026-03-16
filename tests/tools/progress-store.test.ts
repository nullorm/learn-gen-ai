import { describe, test, expect, afterEach } from 'bun:test'
import { existsSync, unlinkSync } from 'node:fs'
import { loadProgress, saveProgress } from '../../tools/progress-store.js'
import { INITIAL_STATE } from '../../tools/progress-types.js'

const TEST_PATH = '/tmp/l1-test-progress.json'

afterEach(() => {
  if (existsSync(TEST_PATH)) unlinkSync(TEST_PATH)
})

describe('progress-store', () => {
  test('returns initial state when file missing', () => {
    const state = loadProgress(TEST_PATH)
    expect(state).toEqual(INITIAL_STATE)
  })

  test('round-trips save and load', () => {
    const state = structuredClone(INITIAL_STATE)
    state.xp = 150
    state.badges = ['first_call']
    saveProgress(state, TEST_PATH)
    const loaded = loadProgress(TEST_PATH)
    expect(loaded.xp).toBe(150)
    expect(loaded.badges).toEqual(['first_call'])
  })
})
