// tools/progress.ts
import { loadProgress, saveProgress } from './progress-store.js'
import {
  getRank,
  nextRank,
  updateStreak,
  checkBadges,
  startModule,
  recordQuiz,
  recordExercise,
  completeModule,
} from './progress-engine.js'
import { MODULE_NAMES, PARTS } from './progress-types.js'

const BADGE_DISPLAY: Record<string, string> = {
  first_call: 'First Call',
  prompt_whisperer: 'Prompt Whisperer',
  type_safe: 'Type Safe',
  vector_space: 'Vector Space',
  perfect_score: 'Perfect Score',
  full_stack_llm: 'Full Stack LLM',
  streak_7: '7-Day Streak',
  streak_30: '30-Day Streak',
  first_contact: 'First Contact',
  core_patterns: 'Core Patterns',
  rag_builder: 'RAG Builder',
  agent_deployer: 'Agent Deployer',
  quality_gate: 'Quality Gate',
  production_ready: 'Production Ready',
}

function today(): string {
  return new Date().toISOString().slice(0, 10)
}

function showProgress(): void {
  const state = loadProgress()
  const rank = getRank(state.xp)
  const next = nextRank(state.xp)
  const total = Object.keys(MODULE_NAMES).length
  const completed = Object.values(state.modules).filter(m => m.status === 'completed').length

  console.log('\n========================================')
  console.log('   Applied LLM Engineering')
  console.log('========================================\n')
  console.log(`  Rank: ${rank}    XP: ${state.xp}`)
  if (next) console.log(`  Next: ${next.name} (${next.xpNeeded} XP needed)`)
  console.log(`  Streak: ${state.streak_days} day(s)`)
  console.log(`  Progress: ${completed}/${total} modules\n`)

  for (const [part, { name, modules }] of Object.entries(PARTS)) {
    console.log(`  Part ${part}: ${name}`)
    for (const m of modules) {
      const mod = state.modules[String(m)]
      const status = mod?.status ?? 'not_started'
      const icon = status === 'completed' ? '[x]' : status === 'in_progress' ? '[>]' : '[ ]'
      const quizInfo = mod?.quiz_score !== undefined ? ` (quiz: ${mod.quiz_score}/${mod.quiz_total})` : ''
      console.log(`    ${icon} ${m}. ${MODULE_NAMES[m]}${quizInfo}`)
    }
    console.log()
  }

  if (state.badges.length > 0) {
    console.log('  Badges:')
    for (const b of state.badges) {
      console.log(`    * ${BADGE_DISPLAY[b] ?? b}`)
    }
    console.log()
  }
}

function main(): void {
  const args = process.argv.slice(2)
  const command = args[0]

  if (!command || command === 'show') {
    showProgress()
    return
  }

  const state = loadProgress()
  updateStreak(state, today())

  switch (command) {
    case 'start': {
      const moduleNum = parseInt(args[1], 10)
      if (!MODULE_NAMES[moduleNum]) {
        console.error(`Unknown module: ${moduleNum}`)
        process.exit(1)
      }
      startModule(state, moduleNum)
      console.log(`Started Module ${moduleNum}: ${MODULE_NAMES[moduleNum]}`)
      break
    }
    case 'quiz': {
      const moduleNum = parseInt(args[1], 10)
      const score = parseInt(args[2], 10)
      const total = parseInt(args[3], 10)
      if (!MODULE_NAMES[moduleNum] || isNaN(score) || isNaN(total)) {
        console.error('Usage: quiz <module> <score> <total>')
        process.exit(1)
      }
      recordQuiz(state, moduleNum, score, total)
      console.log(`Quiz recorded: ${score}/${total} for Module ${moduleNum}`)
      if (score === total) console.log('  Perfect score! +50 bonus XP')
      break
    }
    case 'exercise': {
      const moduleNum = parseInt(args[1], 10)
      const exerciseNum = parseInt(args[2], 10)
      if (!MODULE_NAMES[moduleNum] || isNaN(exerciseNum)) {
        console.error('Usage: exercise <module> <exercise_num>')
        process.exit(1)
      }
      recordExercise(state, moduleNum, exerciseNum)
      console.log(`Exercise ${exerciseNum} recorded for Module ${moduleNum}`)
      break
    }
    case 'complete': {
      const moduleNum = parseInt(args[1], 10)
      if (!MODULE_NAMES[moduleNum]) {
        console.error(`Unknown module: ${moduleNum}`)
        process.exit(1)
      }
      completeModule(state, moduleNum)
      console.log(`Completed Module ${moduleNum}: ${MODULE_NAMES[moduleNum]}`)
      break
    }
    default:
      console.error(`Unknown command: ${command}`)
      console.error('Usage: progress [show|start|quiz|exercise|complete]')
      process.exit(1)
  }

  const newBadges = checkBadges(state)
  if (newBadges.length > 0) {
    console.log('\n  New badges earned:')
    for (const b of newBadges) {
      console.log(`    * ${BADGE_DISPLAY[b] ?? b}`)
    }
  }

  console.log(`  XP: ${state.xp} | Rank: ${getRank(state.xp)}`)
  saveProgress(state)
}

main()
