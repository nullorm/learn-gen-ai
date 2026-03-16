// tools/progress-types.ts

export const MODULE_NAMES: Record<number, string> = {
  1: 'Setup & First LLM Calls',
  2: 'Prompt Engineering',
  3: 'Structured Output',
  4: 'Conversations & Memory',
  5: 'Long Context & Caching',
  6: 'Streaming & Real-time',
  7: 'Tool Use',
  8: 'Embeddings & Similarity',
  9: 'RAG Fundamentals',
  10: 'Advanced RAG',
  11: 'Document Processing',
  12: 'Knowledge Graphs',
  13: 'Multi-modal',
  14: 'Agent Fundamentals',
  15: 'Multi-Agent Systems',
  16: 'Workflows & Chains',
  17: 'Code Generation',
  18: 'Human-in-the-Loop',
  19: 'Evals & Testing',
  20: 'Fine-tuning',
  21: 'Safety & Guardrails',
  22: 'Cost Optimization',
  23: 'Observability',
  24: 'Deployment',
}

export const PARTS: Record<string, { name: string; modules: number[] }> = {
  I: { name: 'First Contact', modules: [1, 2, 3] },
  II: { name: 'Core Patterns', modules: [4, 5, 6, 7, 8, 9] },
  III: { name: 'Advanced Retrieval', modules: [10, 11, 12, 13] },
  IV: { name: 'Agents & Orchestration', modules: [14, 15, 16, 17, 18] },
  V: { name: 'Quality & Safety', modules: [19, 20, 21, 22] },
  VI: { name: 'Production', modules: [23, 24] },
}

export const PART_BADGES: Record<string, string> = {
  I: 'First Contact',
  II: 'Core Patterns',
  III: 'RAG Builder',
  IV: 'Agent Deployer',
  V: 'Quality Gate',
  VI: 'Production Ready',
}

export const RANKS = [
  { name: 'Token', xp: 0 },
  { name: 'Prompter', xp: 100 },
  { name: 'Embedder', xp: 300 },
  { name: 'Retriever', xp: 600 },
  { name: 'Tool Smith', xp: 1_000 },
  { name: 'Agent Builder', xp: 1_500 },
  { name: 'Eval Master', xp: 2_200 },
  { name: 'LLM Architect', xp: 3_000 },
] as const

export const XP = {
  CORRECT_ANSWER: 20,
  EXERCISE: 30,
  MODULE_COMPLETE: 100,
  PERFECT_QUIZ_BONUS: 50,
} as const

export interface ModuleProgress {
  status: 'not_started' | 'in_progress' | 'completed'
  quiz_score?: number
  quiz_total?: number
  exercises: number[]
}

export interface ProgressState {
  modules: Record<string, ModuleProgress>
  xp: number
  streak_days: number
  last_active: string
  badges: string[]
}

export const INITIAL_STATE: ProgressState = {
  modules: {},
  xp: 0,
  streak_days: 0,
  last_active: '',
  badges: [],
}
