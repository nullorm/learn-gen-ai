// tools/lint-course.ts
// Structural linter for course modules. Enforces the canonical skeleton
// (opening triad, gap-free sections, optional Going Further coda, closing
// Summary/Quiz/Exercises) and the approved callout vocabulary. See course/STYLE.md.
import { readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'

export const APPROVED_CALLOUTS = new Set([
  'Beginner Note',
  'Advanced Note',
  'Production Patterns',
  'Provider Tip',
  'Local Alternative',
  'Try it',
  'Gotcha',
  'Before / After',
  'Decision',
])

export interface Heading {
  level: number
  text: string
  line: number
}

// A callout label is approved if it equals an approved label, or starts with
// one followed by a boundary (":" or " ") — so "Provider Tip: Native Citations"
// and "Local Alternative (Ollama)" pass, while "Note" / "Important" do not.
export function isApprovedCallout(label: string): boolean {
  for (const a of APPROVED_CALLOUTS) {
    if (label === a) return true
    if (label.startsWith(a)) {
      const next = label.charAt(a.length)
      if (next === ':' || next === ' ') return true
    }
  }
  return false
}

export function parseHeadings(content: string): Heading[] {
  const headings: Heading[] = []
  const lines = content.split('\n')
  let inFence = false
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? ''
    if (line.startsWith('```')) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    const m = line.match(/^(#{1,6})\s+(.*)$/)
    if (m) headings.push({ level: m[1]!.length, text: m[2]!.trim(), line: i + 1 })
  }
  return headings
}

export function lintModule(name: string, content: string): string[] {
  const errors: string[] = []
  const headings = parseHeadings(content)
  const h2 = headings.filter(h => h.level === 2)
  const h2text = h2.map(h => h.text)

  // 1. Opening triad present and ordered
  const triad = ['Learning Objectives', 'Why Should I Care?', 'Connection to Other Modules']
  const idx = triad.map(t => h2text.indexOf(t))
  triad.forEach((t, k) => {
    if (idx[k] === -1) errors.push(`missing "## ${t}"`)
  })
  if (idx.every(i => i >= 0) && !(idx[0]! < idx[1]! && idx[1]! < idx[2]!)) {
    errors.push('opening triad out of order (Learning Objectives -> Why Should I Care? -> Connection to Other Modules)')
  }

  // 2. Closing trio = the LAST three H2s, ordered Summary -> Quiz -> Exercises
  const closing = ['Summary', 'Quiz', 'Exercises']
  closing.forEach(c => {
    const count = h2text.filter(t => t === c).length
    if (count === 0) errors.push(`missing "## ${c}"`)
    else if (count > 1) errors.push(`duplicate "## ${c}" (${count}x)`)
  })
  if (closing.every(c => h2text.includes(c))) {
    const lastThree = h2text.slice(-3).join(' | ')
    if (lastThree !== 'Summary | Quiz | Exercises') {
      errors.push(`Summary/Quiz/Exercises must be the last three H2s in that order (found last three: ${lastThree})`)
    }
  }

  // 3. Section numbering gap-free from 1
  const sectionNums: number[] = []
  for (const t of h2text) {
    const sm = t.match(/^Section (\d+):/)
    if (sm) sectionNums.push(Number(sm[1]!))
  }
  sectionNums.forEach((n, k) => {
    if (n !== k + 1) errors.push(`section numbering gap/dupe: expected Section ${k + 1}, found Section ${n}`)
  })

  // 4. Going Further (if present): after last Section, before Summary
  const gfIdx = h2.findIndex(h => /^Going Further/.test(h.text))
  if (gfIdx >= 0) {
    const isSection = h2.map(h => /^Section \d+:/.test(h.text))
    const lastSectionIdx = isSection.lastIndexOf(true)
    const summaryIdx = h2text.indexOf('Summary')
    if (lastSectionIdx >= 0 && gfIdx < lastSectionIdx) {
      errors.push('"## Going Further" must come after the last "## Section"')
    }
    if (summaryIdx >= 0 && gfIdx > summaryIdx) {
      errors.push('"## Going Further" must come before "## Summary"')
    }
  }

  // 5. Approved callout labels only
  const calloutRe = /^>\s*\*\*([^*]+?):?\*\*/
  const lines = content.split('\n')
  let inFence = false
  lines.forEach((line, i) => {
    if (line.startsWith('```')) {
      inFence = !inFence
      return
    }
    if (inFence) return
    const cm = line.match(calloutRe)
    if (cm) {
      const label = cm[1]!.replace(/:$/, '').trim()
      if (!isApprovedCallout(label)) {
        errors.push(`line ${i + 1}: unapproved callout label "${label}"`)
      }
    }
  })

  return errors.map(e => `${name}: ${e}`)
}

export function lintAll(courseDir: string): string[] {
  const files = readdirSync(courseDir)
    .filter(f => /^module_\d+_.*\.md$/.test(f))
    .sort()
  return files.flatMap(f => lintModule(f, readFileSync(resolve(courseDir, f), 'utf-8')))
}

if (import.meta.main) {
  const args = process.argv.slice(2)
  const errors =
    args.length > 0
      ? args.flatMap(p => lintModule(p.split('/').pop()!, readFileSync(resolve(p), 'utf-8')))
      : lintAll(resolve('course'))
  if (errors.length === 0) {
    console.log('course lint: OK')
  } else {
    console.error(`course lint: ${errors.length} issue(s)`)
    for (const e of errors) console.error('  ' + e)
    process.exit(1)
  }
}
