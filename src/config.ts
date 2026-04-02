import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { parse } from 'smol-toml'

export interface Preferences {
  background: {
    level: 'beginner' | 'intermediate' | 'advanced'
  }
  provider: {
    default: 'anthropic' | 'openai' | 'ollama'
    ollama_model: string
  }
  data: {
    vector_store: 'lancedb' | 'pgvector' | 'qdrant'
  }
}

const DEFAULTS: Preferences = {
  background: {
    level: 'intermediate',
  },
  provider: {
    default: 'anthropic',
    ollama_model: 'qwen3.5',
  },
  data: {
    vector_store: 'lancedb',
  },
}

export function loadPreferences(path?: string): Preferences {
  const filePath = path ?? resolve('course', 'preferences.toml')

  if (!existsSync(filePath)) {
    return structuredClone(DEFAULTS)
  }

  const raw = parse(readFileSync(filePath, 'utf-8')) as Record<string, unknown>

  return {
    background: { ...DEFAULTS.background, ...(raw.background as Record<string, unknown>) },
    provider: { ...DEFAULTS.provider, ...(raw.provider as Record<string, unknown>) },
    data: { ...DEFAULTS.data, ...(raw.data as Record<string, unknown>) },
  }
}
