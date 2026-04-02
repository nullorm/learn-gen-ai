import { describe, test, expect } from 'bun:test'
import { writeFileSync } from 'node:fs'
import { loadPreferences, type Preferences } from '../src/config.js'

describe('loadPreferences', () => {
  test('returns defaults when file does not exist', () => {
    const prefs = loadPreferences('/nonexistent/path.toml')
    expect(prefs.background.level).toBe('intermediate')
    expect(prefs.provider.default).toBe('anthropic')
    expect(prefs.provider.ollama_model).toBe('qwen3.5')
    expect(prefs.data.vector_store).toBe('lancedb')
  })

  test('loads and merges with defaults', () => {
    const tmp = '/tmp/l1-test-prefs.toml'
    writeFileSync(tmp, '[background]\nlevel = "advanced"\n')
    const prefs = loadPreferences(tmp)
    expect(prefs.background.level).toBe('advanced')
    expect(prefs.provider.default).toBe('anthropic')
  })
})
