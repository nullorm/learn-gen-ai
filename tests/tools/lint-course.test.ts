import { describe, test, expect } from 'bun:test'
import { lintModule } from '../../tools/lint-course.js'

const wellFormed = `# Module 99: Test

## Learning Objectives
- a

## Why Should I Care?
text

## Connection to Other Modules
text

## Section 1: First
body

## Section 2: Second
body

## Going Further: Extras
### A thing
body

## Summary
text

## Quiz
q

## Exercises
e
`

describe('lintModule', () => {
  test('well-formed module yields no errors', () => {
    expect(lintModule('m99.md', wellFormed)).toEqual([])
  })

  test('flags a missing triad member', () => {
    const c = wellFormed.replace('## Why Should I Care?\ntext\n\n', '')
    expect(lintModule('m99.md', c).some(e => e.includes('Why Should I Care?'))).toBe(true)
  })

  test('flags closing trio not last / out of order', () => {
    const c = `# M\n\n## Learning Objectives\nx\n\n## Why Should I Care?\nx\n\n## Connection to Other Modules\nx\n\n## Section 1: A\nx\n\n## Quiz\nx\n\n## Exercises\nx\n\n## Summary\nx\n`
    expect(lintModule('m.md', c).some(e => e.includes('last three'))).toBe(true)
  })

  test('flags a section-numbering gap', () => {
    const c = wellFormed.replace('## Section 2: Second', '## Section 3: Second')
    expect(lintModule('m99.md', c).some(e => e.includes('expected Section 2'))).toBe(true)
  })

  test('flags an unapproved callout label', () => {
    const c = wellFormed.replace('## Section 1: First\nbody', '## Section 1: First\n\n> **Pro Tip:** nope')
    expect(lintModule('m99.md', c).some(e => e.includes('unapproved callout label "Pro Tip"'))).toBe(true)
  })

  test('accepts approved callout labels', () => {
    const c = wellFormed.replace('## Section 1: First\nbody', '## Section 1: First\n\n> **Try it:** predict the output\n\n> **Gotcha:** watch out')
    expect(lintModule('m99.md', c)).toEqual([])
  })

  test('ignores headings inside code fences', () => {
    const c = wellFormed.replace('## Section 1: First\nbody', '## Section 1: First\n\n```md\n## Section 99: fake\n```')
    expect(lintModule('m99.md', c)).toEqual([])
  })

  test('flags Going Further placed after Summary', () => {
    const c = `# M\n\n## Learning Objectives\nx\n\n## Why Should I Care?\nx\n\n## Connection to Other Modules\nx\n\n## Section 1: A\nx\n\n## Summary\nx\n\n## Going Further: extras\nx\n\n## Quiz\nx\n\n## Exercises\nx\n`
    expect(lintModule('m.md', c).some(e => e.includes('Going Further'))).toBe(true)
  })
})
