# Renumber to Layout A — Implementation Plan (Plan 2 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Renumber the existing 24 modules into their final Layout A positions (workflows-first, then agents), reserving gaps at 15/18/19 for the new modules Plan 3 will author — so the risky cross-reference renumber happens exactly **once**.

**Architecture:** A one-off bun script does the mechanical bulk work (file renames + single-pass `Module N` remap + command-file self-number remap). Structured edits (progress-types tables, README, the module-16→14 reframe, 2 plural range refs) are done by hand. `lint-course.ts` does **not** enforce cross-module contiguity, so gaps at 15/18/19 are lint-safe in the interim.

**Tech Stack:** Bun, TypeScript, markdown. Prereq: **Plan 1 (v7 migration) is committed** on `feat/workflow-eve-v7-migration`.

> ⚠️ **Reversibility:** The working tree is clean (Plan 1 committed). If the script misfires, `git checkout -- . && git clean -fd course .claude` restores it. Each task ends committed; the whole renumber lands as one reviewable commit at Task 8.

---

## The renumber map (old → new)

Modules 1–13 unchanged. **15, 18, 19 are reserved gaps** (Plan 3 fills them).

| Old | New | Module | | Old | New | Module |
|---|---|---|---|---|---|---|
| 16 | **14** | Workflows & Chains *(reframe)* | | 20 | **23** | Fine-tuning |
| 14 | **16** | Agent Fundamentals | | 21 | **24** | Safety & Guardrails |
| 15 | **17** | Multi-Agent Systems | | 22 | **25** | Cost Optimization |
| 17 | **20** | Code Generation | | 23 | **26** | Observability |
| 18 | **21** | Human-in-the-Loop | | 24 | **27** | Deployment |
| 19 | **22** | Evals & Testing | | | | |

Final Part IV order: 14 Chains · **15 Durable Workflows (new)** · 16 Agent Fundamentals · 17 Multi-Agent · **18 Eve Fundamentals (new)** · **19 Eve in Production (new)** · 20 Code Generation · 21 Human-in-the-Loop.

## Surfaces touched
- `course/module_NN_*.md` — rename (slugs kept) + ~260 singular `Module N` refs + 2 plural range refs.
- `.claude/commands/module-N.md` — rename (these ARE the `/module-N` skills) + self-number refs (`Module N:`, `module_NN_` path, `start/quiz/exercise/complete N`).
- `tools/progress-types.ts` — `MODULE_NAMES` + `PARTS` (existing 24 at new numbers; gaps reserved).
- `tools/progress-engine.ts` — cosmetic `all 24 modules` comment.
- `README.md` — curriculum tables + Estimated Time table + header count.
- `course/module_14_workflows.md` — reframe (chains now precede agents).
- `progress.json` (gitignored) — optional local key remap.

---

## Task 1: Write the renumber script

**Files:**
- Create: `tools/renumber-layout-a.ts`

- [ ] **Step 1: Write the script**

```typescript
// tools/renumber-layout-a.ts — one-off Layout A renumber. Delete after use.
import { readFileSync, writeFileSync, renameSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'

const MAP: Record<number, number> = {
  14: 16, 15: 17, 16: 14, 17: 20, 18: 21, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27,
}
const REV: Record<number, number> = Object.fromEntries(Object.entries(MAP).map(([o, n]) => [n, Number(o)]))
const mapNum = (n: number) => MAP[n] ?? n
const pad = (n: number) => String(n).padStart(2, '0')

// 1. Rename course files, two-phase (avoids collisions during the 14↔16 swap)
const courseDir = resolve('course')
for (const f of readdirSync(courseDir).filter(f => /^module_\d+_.*\.md$/.test(f))) {
  const m = f.match(/^module_(\d+)_(.*)\.md$/)!
  renameSync(resolve(courseDir, f), resolve(courseDir, `TMP__module_${pad(mapNum(Number(m[1])))}_${m[2]}.md`))
}
for (const f of readdirSync(courseDir).filter(f => f.startsWith('TMP__'))) {
  renameSync(resolve(courseDir, f), resolve(courseDir, f.slice(5)))
}

// 2. Remap singular "Module N" refs in every course file (single pass — handles the swap)
for (const f of readdirSync(courseDir).filter(f => /^module_\d+_.*\.md$/.test(f))) {
  const p = resolve(courseDir, f)
  writeFileSync(p, readFileSync(p, 'utf-8').replace(/\bModule (\d+)\b/g, (_, d) => `Module ${mapNum(Number(d))}`))
}

// 3. Rename command files, two-phase
const cmdDir = resolve('.claude/commands')
for (const f of readdirSync(cmdDir).filter(f => /^module-\d+\.md$/.test(f))) {
  const oldN = Number(f.match(/^module-(\d+)\.md$/)![1])
  renameSync(resolve(cmdDir, f), resolve(cmdDir, `TMP__module-${mapNum(oldN)}.md`))
}
for (const f of readdirSync(cmdDir).filter(f => f.startsWith('TMP__'))) {
  renameSync(resolve(cmdDir, f), resolve(cmdDir, f.slice(5)))
}

// 4. Remap each command file's self-number (targeted — never touches "quiz N <score> 5" etc.)
for (const f of readdirSync(cmdDir).filter(f => /^module-\d+\.md$/.test(f))) {
  const newN = Number(f.match(/^module-(\d+)\.md$/)![1])
  const oldN = REV[newN] ?? newN
  if (oldN === newN) continue
  const p = resolve(cmdDir, f)
  let c = readFileSync(p, 'utf-8')
  c = c.replace(new RegExp(`Module ${oldN}:`, 'g'), `Module ${newN}:`)
  c = c.replace(new RegExp(`module_${pad(oldN)}_`, 'g'), `module_${pad(newN)}_`)
  c = c.replace(new RegExp(`\\b(start|quiz|exercise|complete) ${oldN}\\b`, 'g'), `$1 ${newN}`)
  writeFileSync(p, c)
}

console.log('renumber-layout-a: renamed course + command files, remapped refs')
```

- [ ] **Step 2: Type-check the script**

Run: `bunx tsc --noEmit tools/renumber-layout-a.ts 2>&1 | grep -v "env.js" | tail -5`
Expected: no errors from this file (the pre-existing `env.js` error may still print; ignore it).

- [ ] **Step 3: Commit the script**

```bash
git add tools/renumber-layout-a.ts
git commit -m "chore(renumber): add one-off Layout A renumber script"
```

---

## Task 2: Run the renumber + verify mechanical correctness

- [ ] **Step 1: Run it**

Run: `bun run tools/renumber-layout-a.ts`
Expected: `renumber-layout-a: renamed course + command files, remapped refs`

- [ ] **Step 2: Verify files landed at the right numbers (slugs preserved)**

Run: `ls course/module_1[4-9]_*.md course/module_2*.md && echo "---" && ls .claude/commands/module-1[4-9].md .claude/commands/module-2*.md`
Expected: `module_14_workflows.md`, `module_16_agent_fundamentals.md`, `module_17_multi_agent.md`, `module_20_code_generation.md`, `module_21_human_in_the_loop.md`, `module_22_evals.md`, `module_23_fine_tuning.md`, `module_24_safety.md`, `module_25_cost_optimization.md`, `module_26_observability.md`, `module_27_deployment.md`; **no** `module_15/18/19` (reserved). Command files match numbers.

- [ ] **Step 3: Verify git sees renames (not delete+add) and self-number consistency**

Run:
```bash
git status --short | grep -E "^R|renamed" | head -30
# each command file's title number must equal its filename number:
for f in .claude/commands/module-*.md; do n=$(echo "$f"|grep -oE '[0-9]+'); grep -q "Module $n:" "$f" || echo "MISMATCH $f"; done
echo "mismatch check done"
```
Expected: renames listed; "mismatch check done" with no MISMATCH lines.

- [ ] **Step 4: Verify each command's `course/module_NN_` path points at an existing file**

Run:
```bash
for f in .claude/commands/module-*.md; do p=$(grep -oE "course/module_[0-9]+_[a-z_]+\.md" "$f"|head -1); [ -f "$p" ] || echo "BAD PATH in $f -> $p"; done; echo "path check done"
```
Expected: "path check done" with no BAD PATH lines.

- [ ] **Step 5: Commit**

```bash
git add -A course/ .claude/commands/
git commit -m "refactor(course): renumber existing modules to Layout A (gaps reserved: 15/18/19)"
```

---

## Task 3: Fix the 2 plural range refs the singular regex can't reach

**Files:** `course/module_10_advanced_rag.md`, `course/module_07_tool_use.md`

> Note: Tool Use (module **7**) and Advanced RAG (module **10**) keep their numbers — only their *content's* plural range refs need fixing (the singular-`Module N` script skips `Modules …`).

- [ ] **Step 1: Fix `Modules 19 and 23` → `Modules 22 and 26`** in `course/module_10_advanced_rag.md`

Find: `Modules 19 and 23` → replace with `Modules 22 and 26`.

- [ ] **Step 2: Fix `Modules 14-15` → `Modules 16-17`** in `course/module_07_tool_use.md`

Find: `Modules 14-15` → replace with `Modules 16-17`.

- [ ] **Step 3: Verify no stale plural refs remain**

Run: `grep -rnoE "Modules [0-9]+[ -]" course/*.md`
Expected: only `Modules 1 and 9` (module_04, unchanged), `Modules 22 and 26`, `Modules 16-17`.

- [ ] **Step 4: Commit**

```bash
git add course/
git commit -m "refactor(course): fix plural module range references for Layout A"
```

---

## Task 4: Rewrite `progress-types.ts` tables

**Files:** Modify `tools/progress-types.ts`

- [ ] **Step 1: Replace `MODULE_NAMES`** with the 24 existing modules at their new numbers (gaps 15/18/19 reserved):

```typescript
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
  14: 'Workflows & Chains',
  16: 'Agent Fundamentals',
  17: 'Multi-Agent Systems',
  20: 'Code Generation',
  21: 'Human-in-the-Loop',
  22: 'Evals & Testing',
  23: 'Fine-tuning',
  24: 'Safety & Guardrails',
  25: 'Cost Optimization',
  26: 'Observability',
  27: 'Deployment',
  // Reserved for Plan 3: 15 Durable Workflows, 18 Eve Fundamentals, 19 Eve in Production
}
```

- [ ] **Step 2: Replace `PARTS`** (Part IV keeps its gaps until Plan 3):

```typescript
export const PARTS: Record<string, { name: string; modules: number[] }> = {
  I: { name: 'First Contact', modules: [1, 2, 3] },
  II: { name: 'Core Patterns', modules: [4, 5, 6, 7, 8, 9] },
  III: { name: 'Advanced Retrieval', modules: [10, 11, 12, 13] },
  IV: { name: 'Agents & Orchestration', modules: [14, 16, 17, 20, 21] }, // 15,18,19 added in Plan 3
  V: { name: 'Quality & Safety', modules: [22, 23, 24, 25] },
  VI: { name: 'Production', modules: [26, 27] },
}
```

Leave `PART_BADGES`, `RANKS`, `XP`, and the interfaces unchanged.

- [ ] **Step 3: Verify tooling still compiles + tests pass**

Run:
```bash
bunx tsc --noEmit 2>&1 | grep -v "env.js" | tail -3
bun test 2>&1 | tail -4
```
Expected: no new tsc errors (only the pre-existing `env.js`); test signature = 32 pass, only `env.test.ts` failing.

- [ ] **Step 4: Commit**

```bash
git add tools/progress-types.ts
git commit -m "refactor(progress): renumber MODULE_NAMES + PARTS to Layout A"
```

---

## Task 5: Reframe module 14 (Workflows & Chains now precedes agents)

**Files:** Modify `course/module_14_workflows.md`

The script remapped the numbers, but the *framing* still assumes agents were taught first. Fix the tense/direction.

- [ ] **Step 1: Reframe the "Connection to Other Modules" bullets.** Open the file, find the `## Connection to Other Modules` section, and replace the agent bullets with forward-looking versions:

```markdown
- **Module 16 (Agent Fundamentals)** — up next in Part IV — introduces the autonomous approach. This module starts Part IV with the deterministic counterpart, so you have a concrete baseline to contrast agents against.
- **Module 17 (Multi-Agent Systems)** — later in Part IV — uses an agent for each step. Here we use plain LLM calls with no agent loop.
- **Module 20 (Code Generation)** can use chains for generate-test-fix pipelines.
- **Module 21 (Human-in-the-Loop)** adds approval gates within chain steps.
```

- [ ] **Step 2: Soften the "Why Should I Care?" opener** if it assumes agent familiarity. Find the sentence beginning "Agents are powerful but unpredictable" and prepend a bridge: "You'll build agents next, in Module 16 — but first: " so the reader isn't assumed to know agents yet. Keep the rest.

- [ ] **Step 3: Verify the module still lints (section numbering, triad, callouts unaffected)**

Run: `bun run tools/lint-course.ts course/module_14_workflows.md`
Expected: `course lint: OK`

- [ ] **Step 4: Commit**

```bash
git add course/module_14_workflows.md
git commit -m "docs(course): reframe Workflows & Chains as Part IV opener (precedes agents)"
```

---

## Task 6: Update README curriculum + time tables

**Files:** Modify `README.md`

- [ ] **Step 1: Rewrite the `### Part IV: Agents & Orchestration` table** to the Layout A order, including placeholder rows for the three new modules:

```markdown
### Part IV: Agents & Orchestration

| #   | Module                    | Topics                                                            | Hours |
| --- | ------------------------- | ---------------------------------------------------------------- | ----- |
| 14  | Workflows & Chains        | Sequential/parallel pipelines, branching, composable chains       | 7-9   |
| 15  | Durable Workflows         | 🚧 Workflow SDK: `use step`, suspend/resume, hooks, WorkflowAgent  | TBD   |
| 16  | Agent Fundamentals        | ReAct pattern, planning loops, tool selection, observation cycles | 8-10  |
| 17  | Multi-Agent Systems       | Orchestrator-worker, delegation, shared state, communication      | 7-9   |
| 18  | Eve Fundamentals          | 🚧 Filesystem-first agents: tools, skills, instructions, evals     | TBD   |
| 19  | Eve in Production         | 🚧 Channels, schedules, subagents, connections, deploy            | TBD   |
| 20  | Code Generation           | LLM-generated code, sandboxed execution, iterative refinement     | 8-10  |
| 21  | Human-in-the-Loop         | Approval flows, feedback integration, active learning             | 7-9   |

> 🚧 Modules 15, 18, 19 are being authored (Workflow SDK + Eve integration). See `docs/superpowers/plans/`.
```

- [ ] **Step 2: Renumber the `### Part V` (→ 22-25) and `### Part VI` (→ 26-27) tables** — same rows, new numbers: 22 Evals & Testing, 23 Fine-tuning, 24 Safety & Guardrails, 25 Cost Optimization; 26 Observability, 27 Deployment.

- [ ] **Step 3: Update the Estimated Time table + the header count.** Change the header line "**24 modules** across 6 parts" → "**27 modules** across 6 parts (24 live + 3 in progress)". In the Estimated Time table set ranges: IV `14-21`, V `22-25`, VI `26-27`, Total `1-27`.

- [ ] **Step 4: Verify no stale module numbers remain in README**

Run: `grep -nE "1[4-9]|2[0-7]" README.md | grep -iE "module|part|\|" | head -20`
Expected: numbers reflect the new layout; no `14-18`/`19-22`/`23-24` old ranges.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs(readme): renumber curriculum to Layout A + placeholder rows for 15/18/19"
```

---

## Task 7: Housekeeping — engine comment + optional progress.json remap

**Files:** Modify `tools/progress-engine.ts`; optionally `progress.json` (gitignored)

- [ ] **Step 1: Future-proof the comment** at `tools/progress-engine.ts:77`: change `// Full Stack LLM — all 24 modules` → `// Full Stack LLM — all modules (count from MODULE_NAMES)`.

- [ ] **Step 2 (optional, ask user first): remap local `progress.json` keys.** Only if the user has meaningful local progress. The map is the same old→new as Task 1. Present this command and run **only on request**:

```bash
bun -e 'const m={14:16,15:17,16:14,17:20,18:21,19:22,20:23,21:24,22:25,23:26,24:27};const s=require("./progress.json");const o={};for(const[k,v]of Object.entries(s.modules))o[String(m[+k]??+k)]=v;s.modules=o;require("fs").writeFileSync("progress.json",JSON.stringify(s,null,2))'
```

- [ ] **Step 3: Commit the engine comment**

```bash
git add tools/progress-engine.ts
git commit -m "chore(progress): drop hardcoded module count from comment"
```

---

## Task 8: Verification gate + cleanup

- [ ] **Step 1: Full lint (per-file skeleton — should be unaffected by renumber)**

Run: `bun run tools/lint-course.ts`
Expected: `course lint: OK`

- [ ] **Step 2: Tooling gate**

Run:
```bash
bunx tsc --noEmit 2>&1 | grep -v "env.js" | tail -3
bun test 2>&1 | tail -4
```
Expected: no new tsc errors; test signature 32 pass, only `env.test.ts` failing.

- [ ] **Step 3: Cross-reference sanity — no dangling references to a module number that no longer holds that topic.** Spot-check the highest-traffic renamed pair (Agent Fundamentals 14→16):

```bash
grep -rn "Module 14" course/*.md | grep -iE "agent fundamental" | head   # expect 0 (Agent Fund is now 16)
grep -rn "Module 16" course/*.md | grep -iE "agent fundamental" | head   # expect the references
grep -rn "Module 16" course/*.md | grep -iE "workflow|chain" | head       # expect 0 (Workflows is now 14)
```
Expected: agent-fundamentals references say "Module 16"; no "Module 16" describes workflows/chains.

- [ ] **Step 4: Delete the one-off script**

```bash
git rm tools/renumber-layout-a.ts
git commit -m "chore(renumber): remove one-off renumber script"
```

- [ ] **Step 5: Confirm every module 1–27 slot is either present or an intended gap**

Run: `for n in $(seq -w 1 27); do ls course/module_${n}_*.md >/dev/null 2>&1 && echo "$n ✓" || echo "$n — (gap)"; done`
Expected: all present except `15`, `18`, `19` marked `(gap)`.

---

## Self-review checklist

- [ ] Renumber map is a bijection on 14–24; new numbers {14,16,17,20-27} distinct; gaps {15,18,19} reserved. ✅
- [ ] Command self-number remap is targeted (`Module N:`, `module_NN_`, `start/quiz/exercise/complete N`) — never touches `quiz N <score> 5` / `5 questions` / `80%`. ✅
- [ ] Singular `Module N` remap is single-pass (handles the 14↔16 swap without collision); 2 plural range refs handled manually (Task 3). ✅
- [ ] `progress-types` gaps (15/18/19) are lint-safe (lint is per-file) and badge-safe (badges use `MODULE_NAMES.length` + `PARTS` arrays). ✅
- [ ] Reframe (Task 5) fixes the chains-before-agents tense; README shows the 27-module target with 15/18/19 flagged. ✅
- [ ] No `src/`/`tests/` touched → `bun test` signature unchanged. ✅

## Carry-forward to Plan 3
- Add modules 15/18/19: `course/module_15_*.md` etc., `.claude/commands/module-{15,18,19}.md`, `MODULE_NAMES` + `PARTS[IV]` entries, README rows (replace 🚧 placeholders), and the `apps/` scaffolds.
- After Plan 3, update `PARTS.IV` to `[14,15,16,17,18,19,20,21]` and CLAUDE.md "24-module" → "27-module".

---

## Execution Record (2026-07-04) — COMPLETE

Executed inline on `feat/workflow-eve-v7-migration`. Course lint OK; tooling signature preserved (32 pass; only `env.test.ts`). Harness re-registered the commands at new numbers (`/module-14` Workflows … `/module-27` Deployment).

**Landed as planned:** renumber script (T1) → renames + 260 singular `Module N` remaps + command self-numbers (T2) → 2 plural range refs (T3) → `progress-types` tables (T4) → module-14 reframe (T5) → README + placeholder rows (T6) → engine comment (T7) → verification + script deleted (T8).

**Notes:**
- **Bonus v7 fix folded into T3:** `course/module_07_tool_use.md` L96 prose still labelled the tool field `` `parameters` `` — corrected to `inputSchema` (a Plan 1 miss surfaced by this pass).
- `git status` showed the renames as delete+add (not `R`) because content changed in the same commit — benign; `git log --follow` still traces history.
- T8.3's "2 × Module 16 near workflow/chain" were **false positives** — the reframed forward-references inside `module_14_workflows.md` itself.
- Interim state (correct, by design): modules 15/18/19 are gaps; `PARTS.IV = [14,16,17,20,21]`; README shows them as 🚧; CLAUDE.md still says "24-module". Plan 3 fills all three.
- Optional, not run: local `progress.json` key remap (Task 7 Step 2) — offer only if the user has meaningful saved progress.
