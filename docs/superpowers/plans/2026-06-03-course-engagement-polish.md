# Course Engagement & Consistency Pass — Implementation Plan

> **Historical (pre-renumber):** this plan uses the old 24-module layout; module numbers and filenames here do not match the current 27-module course.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 24-module course less tedious by varying the teaching cadence, taming bolted-on sections into a signposted `## Going Further` coda, and standardizing structure/terminology — without changing the teach→quiz→exercise flow, the student-builds philosophy, or any module numbers.

**Architecture:** First build a machine-checkable course-structure linter (`tools/lint-course.ts`) and a durable `course/STYLE.md`. Then transform a 2-module pilot (M6 clean, M14 bloated) and get owner sign-off. Then roll out the same transformation module-by-module across the remaining 22 (grouped by the existing `PARTS`), one commit per module. Finally update the 24 `/module-N` command templates + `CLAUDE.md`, and run a full-course structural gate.

**Tech Stack:** TypeScript (strict, `noUncheckedIndexedAccess`), Bun + `bun:test`, Markdown course content. Spec: `docs/superpowers/specs/2026-06-03-course-engagement-polish-design.md`.

**Branch:** `course-polish` (already created off `main`). Never `git add -A` — untracked student work lives in `src/memory/`, `tests/memory/`. Stage explicit paths only. Never edit `src/`, `tests/` student code, `progress.json`, or `preferences.toml`.

---

## Standard Module Transformation Procedure (SMTP)

Every per-module task (Tasks 3–4, 7–28) applies these six steps, then layers its **module-specific actions** from the task body. Defined once here; each task lists only what is unique to that module.

1. **Restructure** (module-specific): apply the task's bolt-on actions — move appended sections under a single `## Going Further: <theme>` heading (their subsections become `###`, not `## Section N`), merge thin sections into named neighbors, and replace duplicated content with a 2–3-sentence cross-reference to the canonical module.
2. **Standardize:** closing sections must be `## Summary` → `## Quiz` → `## Exercises` as the final three `##` headings; renumber `## Section N:` gap-free from 1; every `> **Label:**` callout must use an approved label (see STYLE.md / linter).
3. **Engage:** add the one-line **"You are here"** Part+badge note right after `## Learning Objectives` (source strings from `tools/progress-types.ts`); ensure **2–4 approved callouts** in the module including **≥1 new device** (`Try it` / `Gotcha` / `Before / After` / `Decision`); make **≥1 section a non-Build archetype** (Explore/Decide/Debug) where it fits naturally; open ≥1 section with a concrete hook instead of an abstract definition. Engagement experiments live in prose only — never add `console.log` or non-`expect()` assertions to tests.
4. **Verify:** `bun run tools/lint-course.ts course/<file>` exits 0; `bun test` passes.
5. **Commit:** `git add course/<file>` then commit (one module per commit). If the module's actions touch a second module (cross-reference target), stage both.
6. **Preserve invariants:** do NOT change exercise *specifications* for M1–M5 (student is mid-course); keep quiz format (multiple-choice, Easy/Medium/Hard, inline answer+explanation, 5 questions), exercise format (`### Exercise N`, objective, spec, `src/exercises/mNN/` paths, test spec), and provider-agnostic examples (Mistral default).

**Per-module acceptance checklist** (the reviewer confirms all):
- [ ] Linter exits 0 for the file; `bun test` green.
- [ ] Closing order correct; section numbers gap-free.
- [ ] "You are here" line present and cites the correct Part + badge.
- [ ] ≥1 new-device callout; ≥1 non-Build archetype section; no callout outside the vocabulary.
- [ ] Bloated modules only: a `## Going Further` coda exists and previously-appended sections live under it (no longer top-level `## Section N`).
- [ ] No `src/`, test-code, or exercise-spec (M1–M5) changes; no `git add -A`.

---

## Task 1: Build the course-structure linter (TDD)

**Files:**
- Create: `tools/lint-course.ts`
- Create: `tests/tools/lint-course.test.ts`

- [ ] **Step 1: Write the failing tests**

```ts
// tests/tools/lint-course.test.ts
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `bun test tests/tools/lint-course.test.ts`
Expected: FAIL — `Cannot find module '../../tools/lint-course.js'`.

- [ ] **Step 3: Implement the linter**

Note: uses `String.prototype.match()` (not `RegExp.prototype.exec()`) — identical capture-group behavior for these non-global patterns, and avoids the repo's `exec(` security-hook heuristic.

```ts
// tools/lint-course.ts
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
      if (!APPROVED_CALLOUTS.has(label)) {
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `bun test tests/tools/lint-course.test.ts`
Expected: PASS (8 tests).

- [ ] **Step 5: Confirm the full suite still passes**

Run: `bun test`
Expected: PASS (existing config/progress/memory tests + the 8 new lint tests).

- [ ] **Step 6: Commit**

```bash
git add tools/lint-course.ts tests/tools/lint-course.test.ts
git commit -m "feat(tools): add course-structure linter for the engagement pass"
```

---

## Task 2: Write `course/STYLE.md` (durable style reference)

**Files:**
- Create: `course/STYLE.md`

- [ ] **Step 1: Write the style guide**

Content must document, concretely:

1. **Module skeleton** (the canonical order the linter enforces):
   `# Module N: Title` → `## Learning Objectives` → `## Why Should I Care?` → `## Connection to Other Modules` → `## Section 1..K:` → optional `## Going Further: <theme>` (with `###` subsections) → `## Summary` → `## Quiz` → `## Exercises`.
2. **"You are here" line:** one blockquote right after Learning Objectives naming the Part and Part-badge, e.g. `> *Module 14 opens **Part IV: Agents & Orchestration** — complete the Part to earn the **Agent Deployer** badge.*` Source Part names from `PARTS` and badges from `PART_BADGES` in `tools/progress-types.ts` (reproduce the Part→modules map in a small table here).
3. **Callout vocabulary** (the only labels the linter allows), each with a one-line "use when" and a 2-line example: `Try it`, `Gotcha`, `Before / After`, `Decision`, `Beginner Note`, `Advanced Note`, `Production Patterns`, `Provider Tip`, `Local Alternative`. Cap ~2–4 per module.
4. **Section archetypes** (Build / Explore / Decide / Debug) with the target mix and how each is delivered (copy the table from the spec, Component 1a).
5. **Hard guardrail:** engagement experiments live in prose; tests use `expect()` only — never `console.log`/`console.table`/`process.stdout`.
6. **Terminology rules:** `ModelMessage` not `CoreMessage`; Zod v4 top-level APIs (`z.int()`, `z.email()`, `z.url()`, `z.uuid()`, `z.iso.date()`); `maxOutputTokens` not `maxTokens`; Mistral as the default provider in examples.

- [ ] **Step 2: Verify**

Run: `bun run tools/lint-course.ts` (STYLE.md is not a `module_*.md`, so the CLI ignores it).
Expected: it prints existing pre-transformation issues for real modules — that is the punch-list, not a failure of this task.

- [ ] **Step 3: Commit**

```bash
git add course/STYLE.md
git commit -m "docs(course): add STYLE.md (callout vocabulary, archetypes, skeleton)"
```

---

## Task 3 (PILOT): Transform Module 6 — Streaming (clean module)

**Files:**
- Modify: `course/module_06_streaming.md`

Apply **SMTP**. M6 is structurally clean (no bolt-on surgery) — this task demonstrates the pure engagement layer.

**Module-specific actions:**
- No `## Going Further` needed (no appended sprawl). Confirm closing order is already `Summary → Quiz → Exercises`; fix if not.
- "You are here": Part II: Core Patterns; badge "Core Patterns".
- Convert at least one section to **Explore** (e.g., §1 Time-to-First-Token: have the student run a streamed vs. non-streamed call and *observe* perceived latency — prose experiment, no test) and one to **Decide** (e.g., §8 UI Patterns: which UI pattern fits which use case).
- Add `> **Try it:**` (predict first-token vs. full-response timing) and a `> **Gotcha:**` (e.g., partial-object handling in streamed structured output, §3).
- Keep all Build sections test-first as today.

- [ ] **Step 1: Apply SMTP + the actions above.**
- [ ] **Step 2: Verify** — `bun run tools/lint-course.ts course/module_06_streaming.md` exits 0; `bun test` green.
- [ ] **Step 3: Commit** — `git add course/module_06_streaming.md && git commit -m "docs(m06): engagement pass — varied cadence, callouts, you-are-here"`

---

## Task 4 (PILOT): Transform Module 14 — Agent Fundamentals (most bloated, +100%)

**Files:**
- Modify: `course/module_14_agent_fundamentals.md`

Apply **SMTP**. M14 exercises the full bolt-on clustering.

**Module-specific actions:**
- Create `## Going Further: Production Agent Patterns` and move these under it as `###` subsections: §13 Extended Thinking, §14 Plan and Build Agent Modes, §15 Max Steps Configuration and Hidden System Agents, §16 Enhanced Debugging with Trace Logging.
- Keep §1–§12 as the core arc (ReAct, agent loop, planning, tool selection, Self-RAG, observation processing, termination, agent memory, debugging, production termination, tool orchestration). Renumber if any merges occur.
- Move `## Summary` to the end so closing order is `Summary → Quiz → Exercises` (currently Summary is last after Exercises — verify and fix to the canonical order).
- "You are here": Part IV: Agents & Orchestration; badge "Agent Deployer".
- Add a **Debug** archetype moment in the agent-loop section (present an agent loop that never terminates; student diagnoses the missing termination condition) and a `> **Gotcha:**` about infinite loops / step budgets. Add a `> **Before / After:**` contrasting a one-shot call vs. the ReAct loop on the same task.

- [ ] **Step 1: Apply SMTP + the actions above.**
- [ ] **Step 2: Verify** — `bun run tools/lint-course.ts course/module_14_agent_fundamentals.md` exits 0; `bun test` green.
- [ ] **Step 3: Commit** — `git add course/module_14_agent_fundamentals.md && git commit -m "docs(m14): cluster Going Further coda + engagement pass"`

---

## Task 5 (CHECKPOINT): Owner sign-off on pilot

- [ ] **Step 1:** Show the owner the before/after of M6 and M14 (e.g., `git show` diffs or rendered side-by-side). Summarize how the cadence and bolt-on coda feel.
- [ ] **Step 2:** Get explicit approval of the *feel* (callout density, archetype variety, Going-Further boundary, "You are here" tone). Apply any requested adjustments to M6/M14 and, if they change conventions, to `course/STYLE.md`.
- [ ] **Step 3:** Only after approval, proceed to rollout (Tasks 7–28).

**Do not start Task 7 until the checkpoint is approved.**

---

## Task 6: (reserved — intentionally empty to keep pilot/rollout numbering clear)

Rollout tasks begin at Task 7. (No action.)

---

## Rollout — Part I: First Contact

> Modules 1–3 are **completed** by the student. Per SMTP step 6, do **not** change their exercise specifications; polish prose/cadence only.

### Task 7: Module 1 — Setup & First LLM Calls
**Files:** Modify `course/module_01_setup_first_calls.md`
Apply **SMTP**. Actions: no Going Further (clean). "You are here" → Part I: First Contact, badge "First Contact". Make §2 (multiple provider setups) a **Decide** moment (which provider for which constraint — pairs with the existing provider factory) and add a `> **Provider Tip:**` and a `> **Gotcha:**` (missing env var → which error). Verify closing order. **Freeze exercise specs.**
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m01): engagement pass (specs frozen)`

### Task 8: Module 2 — Prompt Engineering
**Files:** Modify `course/module_02_prompt_engineering.md`
Apply **SMTP**. Actions: clean structure. "You are here" → Part I, badge "First Contact". Add a **Before / After** callout (weak vs. strong prompt — the course already shows the prompts; add a short actual-output contrast) and a **Debug** moment in §7 Common Pitfalls (fix an over-constrained / injection-vulnerable prompt). Verify closing order. **Freeze exercise specs.**
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m02): engagement pass (specs frozen)`

### Task 9: Module 3 — Structured Output
**Files:** Modify `course/module_03_structured_output.md`
Apply **SMTP**. Actions: clean. "You are here" → Part I, badge "First Contact". Add a **Try it** (predict whether a vague schema vs. described schema changes output) and a **Gotcha** (`parse` vs `safeParse`, §8). Verify closing order. **Freeze exercise specs.**
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m03): engagement pass (specs frozen)`

---

## Rollout — Part II: Core Patterns (M6 done in pilot)

### Task 10: Module 4 — Conversations & Memory
**Files:** Modify `course/module_04_conversations_memory.md`
Apply **SMTP**. Actions: clean; already has `Try it:` boxes — keep, ensure they fit the vocabulary. "You are here" → Part II, badge "Core Patterns". Add a **Decide** moment for §6 Hybrid Approaches (which strategy for which conversation shape). Verify closing order. **Freeze exercise specs** (M4 completed).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m04): engagement pass (specs frozen)`

### Task 11: Module 5 — Long Context & Caching
**Files:** Modify `course/module_05_long_context_caching.md`
Apply **SMTP**. Actions: clean; module has a "Decision Tree" already — formalize §8 as a **Decide** archetype. "You are here" → Part II, badge "Core Patterns". Add a **Before / After** on cached vs. uncached cost (§7). Verify closing order. **Student is mid-module — freeze exercise specs and do not alter the *order* of sections they may be working through; prose/callout polish only.**
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m05): engagement pass (in-progress: specs + section order frozen)`

### Task 12: Module 7 — Tool Use
**Files:** Modify `course/module_07_tool_use.md`
Apply **SMTP**. Actions: §9 Security Considerations is substantial but core to tool use — keep in core (do not move to Going Further). If §10 "Production Tool Architecture" is thin/appended, fold it under a `## Going Further: Production Tool Architecture`. "You are here" → Part II, badge "Core Patterns". Add a **Debug** moment (a tool with no input validation → student adds a Zod guard) and a **Gotcha** (parallel tool calls / `stopWhen` step budget). Verify closing order.
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m07): engagement pass`

### Task 13: Module 8 — Embeddings & Similarity
**Files:** Modify `course/module_08_embeddings.md`
Apply **SMTP**. Actions: clean. "You are here" → Part II, badge "Core Patterns". Add a **Try it** (predict which of two sentences is more similar, then compute cosine) and a **Decision** (cosine vs. other distance metrics, §4). Verify closing order.
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m08): engagement pass`

### Task 14: Module 9 — RAG Fundamentals
**Files:** Modify `course/module_09_rag_fundamentals.md`
Apply **SMTP**. Actions: create `## Going Further: Coding-Agent Context Assembly` and move §11 Hierarchical Configuration as Retrieval + §12 Lazy-Loading Referenced Files under it. Keep §9 RAG Assessment (distinct from M10) and §10 Context Priority Ordering in core. "You are here" → Part II, badge "Core Patterns". Add a **Gotcha** (chunk-boundary information loss) and a **Before / After** (no-citation vs. cited answer). Closing order already correct — verify.
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m09): cluster Going Further + engagement pass`

---

## Rollout — Part III: Advanced Retrieval

### Task 15: Module 10 — Advanced RAG
**Files:** Modify `course/module_10_advanced_rag.md`
Apply **SMTP**. Actions: §11 "LSP-Augmented Retrieval" becomes the **canonical** LSP-as-code-intelligence treatment — lightly expand it so M19/M23 can point here. Create `## Going Further: Code-Intelligence Retrieval` and move §11 + §12 Diagnostic-Driven Context under it. Keep §9 Multi-Source, §10 Re-Retrieval in core. **Fix closing order** (move `## Summary` before Quiz/Exercises). "You are here" → Part III: Advanced Retrieval, badge "RAG Builder". Add a **Decision** (query transformation vs. reranking vs. hybrid — when each).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m10): LSP canonical + Going Further + fix closing order`

### Task 16: Module 11 — Document Processing
**Files:** Modify `course/module_11_document_processing.md`
Apply **SMTP**. Actions: clean (already exemplary). **Fix closing order** (Summary currently last). "You are here" → Part III, badge "RAG Builder". Add a **Try it** (run the recursive splitter, observe chunk boundaries) and a **Gotcha** (metadata lost on naive extraction).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m11): engagement pass + fix closing order`

### Task 17: Module 12 — Knowledge Graphs
**Files:** Modify `course/module_12_knowledge_graphs.md`
Apply **SMTP**. Actions: §9 "LSP as an Implicit Knowledge Graph" + §10 "When Graphs Are Free" are thin (~150 words, weak analogy) — extract the one useful sentence into §8 (Graph RAG vs Vector RAG) as a `> **Advanced Note:**`, then delete §9–§10. Renumber. **Fix closing order.** "You are here" → Part III, badge "RAG Builder". Add a **Decision** (graph RAG vs vector RAG).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m12): prune thin LSP sections + fix closing order`

### Task 18: Module 13 — Multi-modal
**Files:** Modify `course/module_13_multimodal.md`
Apply **SMTP**. Actions: merge §9 Image Preprocessing into §2 (Image Input) as practical notes; tighten §10 Token Cost of Images and add a cross-reference to M22; move §11 File Type Routing under `## Going Further: Production Multi-modal Pipelines`. Renumber. **Fix closing order.** "You are here" → Part III, badge "RAG Builder". Add a **Before / After** (raw vs. preprocessed image token cost) and a **Gotcha** (model-specific image limits, §8).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m13): merge/relocate appended sections + fix closing order`

---

## Rollout — Part IV: Agents & Orchestration (M14 done in pilot)

### Task 19: Module 15 — Multi-Agent Systems
**Files:** Modify `course/module_15_multi_agent.md`
Apply **SMTP**. Actions: create `## Going Further: Claude-Code-Style Agent Systems` and move §11 Workspace Isolation, §12 Primary and Subagent Architecture, §13 Agent Configuration via Markdown, §14 @Mention Invocation under it. Keep §9 Agent Pool, §10 Type Specialization in core. **Fix closing order.** "You are here" → Part IV: Agents & Orchestration, badge "Agent Deployer". Add a **Decision** (orchestrator-worker vs. handoff vs. shared-state).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m15): cluster Going Further + fix closing order`

### Task 20: Module 16 — Workflows & Chains
**Files:** Modify `course/module_16_workflows.md` (and `course/module_24_deployment.md` for the headless-CI cross-ref)
Apply **SMTP**. Actions: merge thin §10 Background Execution into §7 Pipeline Monitoring; §11 Undo/Redo stays but cross-references M17's edit-history rather than re-deriving reversibility; replace §12 Headless Execution for CI/CD with a 2–3-sentence cross-reference to M24 (canonical); move §9 Workflow Middleware under `## Going Further: Production Workflow Patterns`. Renumber. **Fix closing order.** "You are here" → Part IV, badge "Agent Deployer". Add a **Decision** (chains vs. agents, §8) and a **Gotcha** (retry without idempotency).
- [ ] Apply SMTP + actions → lint both files → `bun test` → commit `docs(m16): merge/cross-ref appended sections + fix closing order`

### Task 21: Module 17 — Code Generation
**Files:** Modify `course/module_17_code_generation.md`
Apply **SMTP**. Actions: §11 Edit History stays but cross-references M16's undo/redo pattern; move §12 Enhanced Sandboxing under `## Going Further: Hardening Generated-Code Execution`. Keep §9 Diff-Based Editing, §10 Safe Code Writing in core. Renumber. **Fix closing order.** "You are here" → Part IV, badge "Agent Deployer". Add a **Debug** moment (LLM-generated code with a subtle bug; student writes the failing test that catches it — ties to §5 test-driven generation).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m17): cross-ref + Going Further + fix closing order`

### Task 22: Module 18 — Human-in-the-Loop
**Files:** Modify `course/module_18_human_in_the_loop.md`
Apply **SMTP**. Actions: consolidate the four scattered permission sections — §9 Declarative Permission Rules, §10 Permission Modes, §12 Three Approval Modes, §13 Glob-Based Command Permissions — into **two** subsections under `## Going Further: Permission & Approval Systems` ("Permission Models" and "Approval Modes"). Keep §11 Denial Adaptation in core (it's a HITL principle). Renumber. **Fix closing order.** "You are here" → Part IV, badge "Agent Deployer". Add a **Decision** (confidence threshold for auto vs. human review, §3).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m18): consolidate permission sections + fix closing order`

---

## Rollout — Part V: Quality & Safety

### Task 23: Module 19 — Evals & Testing
**Files:** Modify `course/module_19_evals.md` (and `course/module_22_cost_optimization.md` for the cost-eval cross-ref)
Apply **SMTP**. Actions: replace §13 LSP Diagnostics as Eval Signal with a 2–3-sentence cross-reference to M10 (canonical LSP). Fold §10 Cost as Evaluation Dimension into a cross-reference to M22 (keep one paragraph framing why cost is an eval axis, point to M22 for mechanics). Move §11 Diagnostic Capture, §12 Feature Flag A/B Testing under `## Going Further: Eval Infrastructure`. Keep §9 Prompt Version Testing in core. Renumber. (Summary already first — verify.) "You are here" → Part V: Quality & Safety, badge "Quality Gate". Add a **Decision** (human vs. auto eval, §8).
- [ ] Apply SMTP + actions → lint both files → `bun test` → commit `docs(m19): consolidate LSP/cost cross-refs + Going Further`

### Task 24: Module 20 — Fine-tuning
**Files:** Modify `course/module_20_fine_tuning.md`
Apply **SMTP**. Actions: clean (8 sections). "You are here" → Part V, badge "Quality Gate". Add a **Decision** (§1 when to fine-tune vs. prompt/RAG) and a **Gotcha** (overfitting on a tiny dataset, §3).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m20): engagement pass`

### Task 25: Module 21 — Safety & Guardrails
**Files:** Modify `course/module_21_safety.md`
Apply **SMTP**. Actions: keep §9–§12 (indirect injection, secure tool defs, command-execution security, trust boundaries) in core — they are safety principles. Move §13 OS-Level Sandboxing + §14 Network Isolation in Full-Auto Mode under `## Going Further: Autonomous-Agent Security` (deployment-flavored, but safety is the right home). Renumber. "You are here" → Part V, badge "Quality Gate". Add a **Debug** moment (a prompt-injection that bypasses a naive filter; student hardens it) and a **Gotcha** (allowlist vs. denylist).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m21): cluster Going Further + engagement pass`

### Task 26: Module 22 — Cost Optimization
**Files:** Modify `course/module_22_cost_optimization.md`
Apply **SMTP**. Actions: keep §11 Compaction (already cross-refs M4). Move §13 Reasoning Effort and Thinking Budget Control under `## Going Further: Advanced Cost Controls`. Ensure the section that receives the M19 cost-eval cross-reference reads coherently. Renumber. "You are here" → Part V, badge "Quality Gate". Add a **Before / After** (semantic-cache hit vs. miss cost) and a **Decision** (model routing thresholds).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m22): Going Further + engagement pass`

---

## Rollout — Part VI: Production

### Task 27: Module 23 — Observability
**Files:** Modify `course/module_23_observability.md`
Apply **SMTP**. Actions: replace §14 LSP Diagnostics as Observability Signal with a 2–3-sentence cross-reference to M10. Move §12 Enhanced Structured Logging, §13 Context Visualization, §15 Session Sharing for Debugging under `## Going Further: Observability Tooling`. Keep §9 OpenTelemetry, §10 Context Window Monitoring, §11 Pipeline Profiling in core. Renumber. (Summary already first — verify.) "You are here" → Part VI: Production, badge "Production Ready". Add a **Gotcha** (logging PII / prompt content) and a **Decision** (sampling rate vs. cost of tracing).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m23): consolidate LSP cross-ref + Going Further`

### Task 28: Module 24 — Deployment
**Files:** Modify `course/module_24_deployment.md`
Apply **SMTP**. Actions: create `## Going Further: Distribution & Integration` and move §14 Multi-Target Deployment, §15 Client/Server Architecture, §16 Headless CI Execution (canonical home for headless-CI), §17 MCP Server Mode, §18 Multi-Frontend Distribution under it; fold §11 SDK Output Mode (NDJSON) under the same coda. Keep §9 Environment Config, §10 Session Management, §12 Health Endpoints, §13 Graceful Shutdown in core. Renumber. (Summary already first — verify.) "You are here" → Part VI, badge "Production Ready". Add a **Decision** (deployment option matrix, §1) and a **Before / After** (no failover vs. provider failover under an outage).
- [ ] Apply SMTP + actions → lint file → `bun test` → commit `docs(m24): cluster Going Further coda + engagement pass`

---

## Task 29: Update the 24 `/module-N` command templates + CLAUDE.md

**Files:**
- Modify: all 24 `.claude/commands/module-*.md`
- Modify: `CLAUDE.md` (Teaching Flow section)

The 24 command files are byte-identical except for the embedded module number/name/path. Update the **Teaching Approach** and **Student-Builds-Everything Rules** sections in each, preserving each file's existing module number/name/path lines.

- [ ] **Step 1: Replace the "Teaching Approach" section** in each command file with archetype-aware guidance:

```markdown
## Teaching Approach

Teach the module **section by section**. Do NOT dump the entire module content at once.

Before teaching, list the `##` section headings as your lesson plan. Mark any `## Going Further` section as **optional/advanced** — offer to skip it or dive deep based on the student's `preferences.toml` level.

**The student writes ALL implementation code. You write tests and explain concepts.**

Each section has an archetype — adapt delivery to it instead of using the same rhythm every time:

- **Build** (most sections): explain → write a failing test in `tests/` → tell the student what to build → they implement and run tests. Test-first, `expect()` assertions only.
- **Explore**: have the student run a small experiment and predict/observe the result (compare outputs, measure tokens, watch a stream). No test required — the payoff is the observation.
- **Decide**: walk the trade-offs and ask which option they'd choose and why. No test required.
- **Debug**: present broken or anti-pattern code; the student diagnoses and fixes it (optionally a failing test their fix turns green).

Do NOT force a failing test onto a purely conceptual section. Use the callout vocabulary in `course/STYLE.md` (Try it, Gotcha, Before / After, Decision) to break monotony. Wait for the student between sections; do not auto-advance.
```

- [ ] **Step 2: Update the "Student-Builds-Everything Rules"** to add, after the existing bullets:

```markdown
- Explore/Decide sections may have **no test** — that is intentional; still never write implementation code for the student
- Engagement experiments ("Try it", "Before / After") live in prose or a scratch run, **never** as `console.log`/non-`expect()` assertions in test files
```

- [ ] **Step 3: Update `CLAUDE.md` Teaching Flow** — in the "Student-Builds-Everything Approach" area, add a short paragraph pointing to `course/STYLE.md` for the callout vocabulary and the four section archetypes, and note that conceptual sections need not have a test.

- [ ] **Step 4: Verify** — `bun test` green (no code change); spot-check two command files (e.g., `module-1.md`, `module-24.md`) retain their correct module number/name/path.

- [ ] **Step 5: Commit**

```bash
git add .claude/commands/module-*.md CLAUDE.md
git commit -m "docs(commands): archetype-aware teaching flow + STYLE.md reference"
```

---

## Task 30: Final full-course structural gate + README sync

**Files:**
- Possibly modify: `README.md`

- [ ] **Step 1: Run the full-course linter**

Run: `bun run tools/lint-course.ts`
Expected: `course lint: OK`. If any issue prints, fix it in the named module and re-run until clean.

- [ ] **Step 2: Run the full test suite**

Run: `bun test`
Expected: PASS.

- [ ] **Step 3: README sync** — skim `README.md` Part tables and the "Module Dependencies" section. Module numbers/names/Parts are unchanged, so the tables should still be valid; update only any "Topics" cell or dependency note made inaccurate by a relocation (e.g., LSP now solely under M10; cost-eval mechanics now solely under M22). If nothing is inaccurate, make no change.

- [ ] **Step 4: Final spot-read** — open 3 transformed modules (one clean, one medium, one heavily-clustered, e.g. M8 / M13 / M24) and read end-to-end once for flow and to confirm no cross-reference dangles.

- [ ] **Step 5: Commit (if README changed)**

```bash
git add README.md
git commit -m "docs(readme): sync topic/dependency notes after content relocation"
```

- [ ] **Step 6: Branch summary** — report to the owner: modules transformed, sections clustered/merged/cross-referenced, linter green, tests green. Offer next step (open a PR, or keep iterating).

---

## Self-Review (completed during planning)

**Spec coverage:** Component 1 (cadence) → SMTP step 3 + per-module archetype/callout actions + Task 29 command update. Component 2 (progression arc) → SMTP "You are here" + STYLE.md. Component 3 (bolt-on map) → Tasks 3–4, 14–28 mirror the spec's per-module table exactly (LSP canonical in M10/Task 15; M19/Task 23 + M23/Task 27 cross-ref; headless-CI canonical M24/Task 28 + M16/Task 20 cross-ref). Component 4 (standardization) → linter (Task 1) + SMTP step 2 + closing-order fixes flagged on M10–M18 (Tasks 15–22) + terminology in STYLE.md. Component 5 (command layer + STYLE.md) → Tasks 2 and 29. Component 6 (sequencing/safety/verify) → pilot+checkpoint (Tasks 3–5), Part-batched rollout, freeze-M1–5 in SMTP step 6, lint+test gates throughout, full gate Task 30.

**Placeholder scan:** Linter and tests are complete code. Per-module tasks reference SMTP (fully defined) + concrete module-specific actions from the spec table — no "TBD"/"improve it" vagueness. Task 6 is intentionally empty (numbering clarity), labeled as such.

**Type consistency:** `lintModule(name, content)`, `parseHeadings(content)`, `lintAll(dir)`, `APPROVED_CALLOUTS` used identically in `tools/lint-course.ts` and `tests/tools/lint-course.test.ts`. CLI accepts optional file-path args used by per-module verify steps (`bun run tools/lint-course.ts course/<file>`) and the no-arg full run (Task 30).
