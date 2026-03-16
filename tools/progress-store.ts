import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { INITIAL_STATE, type ProgressState } from './progress-types.js'

const DEFAULT_PATH = resolve('progress.json')

export function loadProgress(path?: string): ProgressState {
  const filePath = path ?? DEFAULT_PATH
  if (!existsSync(filePath)) {
    return structuredClone(INITIAL_STATE)
  }
  return JSON.parse(readFileSync(filePath, 'utf-8')) as ProgressState
}

export function saveProgress(state: ProgressState, path?: string): void {
  const filePath = path ?? DEFAULT_PATH
  writeFileSync(filePath, JSON.stringify(state, null, 2) + '\n')
}
