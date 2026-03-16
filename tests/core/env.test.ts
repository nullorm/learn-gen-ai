import { describe, it, expect } from 'bun:test'
import { requireEnv, optionalEnv } from '../../src/core/env.js'

describe('requireEnv', () => {
  it('should return the value when the env var exists', () => {
    process.env.TEST_KEY = 'test-value'
    expect(requireEnv('TEST_KEY')).toBe('test-value')
    delete process.env.TEST_KEY
  })

  it('should throw when the env var is missing', () => {
    delete process.env.MISSING_KEY
    expect(() => requireEnv('MISSING_KEY')).toThrow('Missing required environment variable: MISSING_KEY')
  })
})

describe('optionalEnv', () => {
  it('should return the value when the env var exists', () => {
    process.env.OPT_KEY = 'opt-value'
    expect(optionalEnv('OPT_KEY')).toBe('opt-value')
    delete process.env.OPT_KEY
  })

  it('should return the fallback when the env var is missing', () => {
    delete process.env.OPT_MISSING
    expect(optionalEnv('OPT_MISSING', 'default')).toBe('default')
  })

  it('should return empty string when no fallback is given', () => {
    delete process.env.OPT_MISSING
    expect(optionalEnv('OPT_MISSING')).toBe('')
  })
})
