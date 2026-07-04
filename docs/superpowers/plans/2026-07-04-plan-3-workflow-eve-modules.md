# New Framework Modules (Workflow SDK + Eve) — Implementation Plan (Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author the three reserved modules — **15 Durable Workflows (Workflow SDK)**, **18 Eve Fundamentals**, **19 Eve in Production** — as runnable teaching modules with isolated `apps/` scaffolds, native-harness tests, and full command/progress/README wiring.

**Architecture:** Foundation-first. Phase A stands up an `apps/` bun workspace and **installs + smoke-tests the frameworks before any content is written** — if `workflow`/`eve` can't run here, we learn it in Phase A, not after authoring 3 modules. Phases B–D author one module each (scaffold app → author sections per the spec outline → native-harness tests → wire command/progress/README). Phase E finalizes cross-cutting wiring and flips the 🚧 placeholders.

**Tech Stack:** Bun (workspace + package mgr), Node ≥24 (Eve runtime), `workflow@~4.5` + `@ai-sdk/workflow` + `@workflow/vitest` + `@workflow/swc-plugin`, `eve@~0.19`, Vitest, AI SDK v7, Mistral (direct provider). Prereq: **Plans 1 & 2 committed** on `feat/workflow-eve-v7-migration`.

> Authoring rule (per user preference + spec §5.1): the assistant writes the scaffold/config/harness/plumbing directly; the **student-builds rule applies to the teaching sections** — the assistant writes the failing `@workflow/vitest` / `eve eval` test and describes what to build, the student writes the workflow/step/hook/tool/skill logic.

> ⚠️ **Known risks this plan must surface at runtime, not hide:** (1) `eve init` may be interactive (boots a dev server / offers to open a coding agent) — run non-interactively or detached. (2) The Workflow SWC toolchain + `eve` pull native deps (bun may block postinstalls — `bun pm trust` as needed). (3) `@ai-sdk/workflow` must be v7-compatible (Plan 1 T3.3.4 carry-forward). (4) Eve wants `engines.node >=24` (satisfied: Node 26). If any smoke-test in Phase A fails, STOP and report — do not proceed to author content against a broken toolchain.

---

## Module → slot recap (from Plans 1–2)

Reserved gaps: **15** (after 14 Workflows & Chains), **18**/**19** (after 17 Multi-Agent). Final `PARTS.IV = [14,15,16,17,18,19,20,21]` after this plan.

---

## Phase A — Workspace + framework installs + smoke tests

### Task A1: Make the repo a bun workspace

**Files:** Modify `package.json`

- [ ] **Step 1: Add the workspace glob.** Insert `"workspaces": ["apps/*"],` into `package.json` (top level, after `"type": "module"`).

- [ ] **Step 2: Confirm the core course is unaffected**

Run: `bun test 2>&1 | grep -E "pass|fail" | head -2`
Expected: baseline signature — 32 pass, 1 fail (`env.test.ts`). (`bunfig.toml` roots tests at `./tests`, so `apps/*` are excluded.)

- [ ] **Step 3: Commit**

```bash
git add package.json && git commit -m "chore(workspace): enable apps/* bun workspace for framework modules"
```

### Task A2: Scaffold + smoke-test `apps/workflow-lab` (Workflow SDK)

**Files:** Create `apps/workflow-lab/package.json`, `apps/workflow-lab/vitest.config.ts`, `apps/workflow-lab/src/smoke.ts`, `apps/workflow-lab/tests/smoke.test.ts`

- [ ] **Step 1: Create `apps/workflow-lab/package.json`**

```json
{
  "name": "workflow-lab",
  "private": true,
  "type": "module",
  "scripts": { "test": "vitest run" },
  "dependencies": {
    "workflow": "~4.5",
    "@ai-sdk/workflow": "latest",
    "ai": "^7",
    "@ai-sdk/mistral": "^4",
    "zod": "^4"
  },
  "devDependencies": {
    "@workflow/vitest": "latest",
    "@workflow/swc-plugin": "latest",
    "vitest": "latest"
  }
}
```

- [ ] **Step 2: Install + verify exact versions**

Run: `cd apps/workflow-lab && bun install 2>&1 | tail -8 && bun pm ls 2>&1 | grep -E "workflow|vitest|@ai-sdk/workflow"`
Expected: `workflow@4.5.x`, `@workflow/vitest`, `@workflow/swc-plugin`, `vitest` resolved. If `@ai-sdk/workflow` conflicts with `ai@7`, note the resolved versions and (if incompatible) fall back to the "raw AI SDK inside a `use step`" approach for Module 15's agent section.

- [ ] **Step 3: Write the vitest config**

`apps/workflow-lab/vitest.config.ts`:
```typescript
import { defineConfig } from 'vitest/config'
import { workflow } from '@workflow/vitest'

export default defineConfig({ plugins: [workflow()] })
```

- [ ] **Step 4: Write a smoke workflow + step**

`apps/workflow-lab/src/smoke.ts`:
```typescript
export async function addWorkflow(a: number, b: number) {
  'use workflow'
  return await addStep(a, b)
}

async function addStep(a: number, b: number) {
  'use step'
  return a + b
}
```

- [ ] **Step 5: Write the smoke test (proves the SWC transform + Local World run)**

`apps/workflow-lab/tests/smoke.test.ts`:
```typescript
import { expect, test } from 'vitest'
import { start } from 'workflow/api'
import { addWorkflow } from '../src/smoke'

test('durable workflow runs a step and returns its result', async () => {
  const run = await start(addWorkflow, [2, 3])
  expect(await run.returnValue).toBe(5)
})
```

- [ ] **Step 6: Run the smoke test**

Run: `cd apps/workflow-lab && bun run test 2>&1 | tail -15`
Expected: 1 passing test. **If the directives don't transform (test hangs / `returnValue` undefined), STOP** — the SWC toolchain isn't wired; report exact error before continuing. (Confirm the exact `start`/`returnValue` API against the installed `workflow/api` types if it differs.)

- [ ] **Step 7: Commit**

```bash
cd /home/kick/workspace/es/learn/gen-ai
git add apps/workflow-lab bun.lock && git commit -m "feat(apps): scaffold + smoke-test workflow-lab (Workflow SDK toolchain)"
```

### Task A3: Scaffold + smoke-test `apps/eve-agent` (Eve)

**Files:** Create `apps/eve-agent/` (via `eve init`), then a minimal agent + eval.

- [ ] **Step 1: Scaffold non-interactively**

Run: `cd /home/kick/workspace/es/learn/gen-ai && npx eve@latest init apps/eve-agent 2>&1 | tail -20`
If it blocks on a dev-server / coding-agent prompt: re-run detached or with the non-interactive flag the CLI reports (`eve init --help`), or `Ctrl-C` after scaffold and verify `apps/eve-agent/agent/` exists. **If it cannot scaffold non-interactively, STOP and report** the prompt so we choose a workaround.

- [ ] **Step 2: Pin `eve` + set the Mistral provider default**

In `apps/eve-agent/package.json`: pin `"eve": "~0.19"` (exact minor). Ensure `engines.node >= 24`. Create/confirm `apps/eve-agent/agent/agent.ts`:
```typescript
import { defineAgent } from 'eve'
import { mistral } from '@ai-sdk/mistral'

export default defineAgent({ model: mistral('mistral-small-latest') })
```
(Add `@ai-sdk/mistral` if `eve init` didn't. `MISTRAL_API_KEY` comes from the repo `.env`.)

- [ ] **Step 3: Minimal agent + one tool + one eval (smoke)**

`apps/eve-agent/agent/instructions.md`:
```markdown
You are a concise weather assistant. Use tools when they are available.
```
`apps/eve-agent/agent/tools/get_weather.ts`:
```typescript
import { defineTool } from 'eve/tools'
import { z } from 'zod'

export default defineTool({
  description: 'Get the current weather for a city.',
  inputSchema: z.object({ city: z.string().min(1) }),
  async execute({ city }) {
    return { city, condition: 'Sunny', temperatureF: 72 }
  },
})
```

- [ ] **Step 4: Smoke via `eve info` (offline, no model call)**

Run: `cd apps/eve-agent && npx eve info 2>&1 | tail -20`
Expected: lists the discovered tool `get_weather` (and no discovery errors). This proves the filesystem discovery + config load work without needing a live model. **If `eve info` errors, STOP and report.**

- [ ] **Step 5 (optional, needs network + key): one `eve eval` with `mockModel`.** Defer the full eval harness to Module 18; here just confirm `eve info` is green.

- [ ] **Step 6: Commit**

```bash
cd /home/kick/workspace/es/learn/gen-ai
git add apps/eve-agent bun.lock && git commit -m "feat(apps): scaffold + smoke-test eve-agent (Eve toolchain)"
```

### Task A4: Gate — foundation is real

- [ ] **Step 1: Confirm both harnesses + core course** — `apps/workflow-lab` vitest smoke passes, `eve info` green, core `bun test` still baseline. Only proceed to Phase B–D if all three hold. Record any version substitutions (e.g. `@ai-sdk/workflow` resolved version) for use in Module 15/18/19 examples.

---

## Phase B — Module 15: Durable Workflows (Workflow SDK)

**Files:** Create `course/module_15_durable_workflows.md`, `.claude/commands/module-15.md`; extend `apps/workflow-lab/` with per-section stubs + tests.

Section outline (from spec §5.2; archetypes in parens). Each Build section = concept → failing `@workflow/vitest` test I author → student implements the workflow/step/hook → run green.

1. Why durability? *(Decide, no test)*
2. Directives & steps *(Build)*
3. Determinism rules *(Debug)*
4. Starting & inspecting runs *(Build)*
5. Durable sleep *(Build)*
6. Hooks: suspend & resume *(Build)*
7. Retries & errors *(Build)*
8. Worlds & persistence *(Decide, no test)*
9. Durable AI agents — `WorkflowAgent` + `needsApproval` HITL *(Build)*
10. Going Further *(optional)* — `WorkflowChatTransport`, Bun server setup

- [ ] **Task B1: Skeleton that passes lint.** Create `course/module_15_durable_workflows.md` with the opening triad (`## Learning Objectives`, `## Why Should I Care?`, `## Connection to Other Modules` → links Modules 14/16/21), gap-free `## Section 1..9`, optional `## Going Further`, closing `## Summary`/`## Quiz`/`## Exercises`. Verify: `bun run tools/lint-course.ts course/module_15_durable_workflows.md` → OK. Commit.

- [ ] **Task B2..B10: author one section at a time.** For each section: write the concept prose (STYLE.md callouts), and for Build sections add the failing test under `apps/workflow-lab/tests/` + a stub under `apps/workflow-lab/src/` with a TODO header, and describe what the student builds. Acceptance per section: lint OK; the section's test file exists and fails only because the student stub is empty (asserted, not `console.log`). Commit per section.

- [ ] **Task B11: Quiz (5 Q, 2 easy/2 med/1 hard) + Exercises.** Author both sections. Verify lint. Commit.

- [ ] **Task B12: Command + wiring.** Create `.claude/commands/module-15.md` (copy the module-14 command's teaching-flow shape; set number 15, path `course/module_15_durable_workflows.md`, harness note: this module uses `@workflow/vitest` in `apps/workflow-lab`, not `bun test`). Verify `/module-15` registers. Commit.

---

## Phase C — Module 18: Eve Fundamentals

**Files:** Create `course/module_18_eve_fundamentals.md`, `.claude/commands/module-18.md`; extend `apps/eve-agent/`.

Section outline (spec §5.3): 1. Filesystem-first *(Explore)* · 2. `defineAgent` + `instructions.md` *(Build)* · 3. Tools *(Build)* · 4. Tool approval / HITL *(Build)* · 5. Skills *(Build)* · 6. Model config & providers *(Decide)* · 7. Evals with `eve eval` + `mockModel` *(Build)* · 8. Going Further — subagents preview *(optional)*.

- [ ] **Task C1: Skeleton + lint.** `course/module_18_eve_fundamentals.md` triad (Connection → Modules 7/16/17/15) + gap-free sections + closing trio. Lint OK. Commit.
- [ ] **Task C2..C8: author section-by-section.** Build sections use `eve eval` (`defineEval` + matchers, `mockModel` for offline determinism) as the harness; scaffold stubs under `apps/eve-agent/agent/…` with TODO headers; describe what the student builds. Acceptance: lint OK; eval defined and asserts (not logs). Commit per section.
- [ ] **Task C9: Quiz + Exercises.** Lint. Commit.
- [ ] **Task C10: Command + wiring.** `.claude/commands/module-18.md` (harness note: `eve eval`, Node ≥24). Verify registration. Commit.

---

## Phase D — Module 19: Eve in Production

**Files:** Create `course/module_19_eve_in_production.md`, `.claude/commands/module-19.md`; extend `apps/eve-agent/`.

Section outline (spec §5.4): 1. Channels *(Build/Explore)* · 2. Schedules *(Build)* · 3. Subagents *(Build)* · 4. Connections — MCP/OpenAPI *(Decide)* · 5. Execution model & durability — ties to Module 15 *(Explore)* · 6. Deploy anywhere *(Decide/Build)* · 7. Going Further — sandbox, Connect/Slack, OTel → Module 26 *(optional)*.

- [ ] **Task D1: Skeleton + lint.** Triad (Connection → Modules 15/18/21/26). Commit.
- [ ] **Task D2..D7: author section-by-section.** Runtime-heavy sections (channels/deploy) are Explore/Decide with no test; testable ones use `eve eval`. Mark Vercel-only features (Connect/Slack creds, dashboard, Vercel Sandbox) conceptual/optional. Commit per section.
- [ ] **Task D8: Quiz + Exercises.** Commit.
- [ ] **Task D9: Command + wiring.** `.claude/commands/module-19.md`. Verify. Commit.

---

## Phase E — Final cross-cutting wiring

- [ ] **Task E1: `progress-types.ts`.** Add to `MODULE_NAMES`: `15: 'Durable Workflows'`, `18: 'Eve Fundamentals'`, `19: 'Eve in Production'`. Set `PARTS.IV = [14, 15, 16, 17, 18, 19, 20, 21]`. Verify tsc + `bun test` baseline. Commit.

- [ ] **Task E2: README.** Replace the 🚧 rows (15/18/19) with real topics + hour estimates; remove the "being authored" note; drop the `+ TBD` from the Estimated Time table (set real totals). Commit.

- [ ] **Task E3: CLAUDE.md + STYLE.md.** CLAUDE.md: "24-module" → "27-module"; add the **Framework modules (15, 18, 19)** subsection (apps/ layout, `@workflow/vitest` / `eve eval` harness exception, Node ≥24 for Eve, pinned framework versions, Mistral-direct provider). STYLE.md: one line noting framework modules use the native harness but keep assertion-based tests + the same archetype vocabulary. Commit.

- [ ] **Task E4: Full verification gate.**
```bash
bun run tools/lint-course.ts                    # OK (all 27 modules)
bunx tsc --noEmit 2>&1 | grep -v env.js | tail -2
bun test 2>&1 | grep -E "pass|fail" | head -2   # baseline
cd apps/workflow-lab && bun run test 2>&1 | tail -3   # workflow tests green
cd ../eve-agent && npx eve info 2>&1 | tail -3        # eve discovery green
for n in $(seq -w 1 27); do ls course/module_${n}_*.md >/dev/null 2>&1 || echo "MISSING $n"; done; echo "slot check done"
```
Expected: lint OK, tsc clean, core baseline, framework harnesses green, no MISSING slots (all 27 present).

---

## Self-review checklist
- [ ] Phase A gates installs BEFORE authoring — a broken toolchain stops the plan, not 3 modules in. ✅
- [ ] Every module: lint-passing skeleton (triad + gap-free sections + closing trio) → section-by-section authoring → native-harness tests (assertions, never `console.log`) → command + wiring. ✅
- [ ] Provider = Mistral direct object; Vercel-only Eve features flagged conceptual. ✅
- [ ] Final state: 27 contiguous modules, `PARTS.IV` complete, README de-🚧'd, CLAUDE.md 27-module + framework-harness exception. ✅

## Notes
- Content prose is authored at execution (that's the creative work); this plan fixes structure, scaffolds, tests, and acceptance criteria per section.
- If Phase A reveals a hard blocker (e.g. Workflow SWC can't run under this Bun, or `eve init` needs a Vercel login), fall back to a **conceptual + read-along** treatment for the blocked framework and record the decision — do not fake runnable code.

---

## Phase A Execution Record (2026-07-04) — COMPLETE ✅

Both frameworks install AND run in this environment — content modules can be real/runnable, no conceptual fallback needed.

- **workflow-lab:** `workflow@4.5.0` + `@workflow/vitest@4.0.11` + `@workflow/swc-plugin@4.1.1` + `@ai-sdk/workflow@1.0.15`; smoke `addWorkflow(2,3)` → `5` passes under vitest (SWC directive transform works).
- **eve-agent:** `eve@0.19.0` scaffolded via `npx eve init eve-agent` (run from `apps/` — it rejects a nested path as a project *name*); `eve info` = 0 errors/0 warnings. Confirmed Eve runs on the Workflow SDK internally.
- **Gotchas handled:** (1) bun didn't hoist member deps → they live in `apps/*/node_modules` (fine). (2) `@workflow/vitest` + eve emit runtime/build dirs (`.workflow-data`, `.workflow-vitest`, `.eve`, `.output`) — now gitignored. (3) `eve init` wrote `engines.node:"24.x"` + `overrides.ai` to the **root** package.json — reverted engines (kept `apps/eve-agent` `engines.node:">=24"`), kept the harmless `ai` override.
- **Deferred to Phase C:** switch `apps/eve-agent/agent/agent.ts` from the scaffold default `anthropic/claude-sonnet-5` to `mistral('mistral-small-latest')` + add `@ai-sdk/mistral` (that's a Module 18 teaching step, and `eve info` doesn't call the model).

## Phases B–E Execution Record (2026-07-04) — COMPLETE ✅

- **Module 15 (Durable Workflows)** — full module + `.claude/commands/module-15.md` + runnable `apps/workflow-lab` scaffold (TODO stubs + authored `@workflow/vitest` tests). Every API empirically verified (`start`/`returnValue`, `sleep`, `defineHook`/`resumeHook`, `waitForSleep`/`waitForHook`) before authoring. Finding: the Workflow compiler *permits* `Math.random()` in a workflow body (replay bug, not build error) → made the Debug section real.
- **Module 18 (Eve Fundamentals)** — full module + command + `apps/eve-agent` scaffold with an **offline, deterministic `eve eval`** (mockModel + tool + `t.succeeded`/`t.calledTool`/`t.check`, 3/3 gates, no key). Finding: `mockModel` must borrow a *known* model identity so auto-compaction can resolve a context window; `compaction: false` is not a valid shape.
- **Module 19 (Eve in Production)** — full module + command (channels/schedules/subagents/connections/durability/deploy; conceptual Decide/Explore + testable subagents/schedules). APIs confirmed from bundled version-matched eve docs.
- **Phase E** — added 15/18/19 to `MODULE_NAMES` + `PARTS.IV = [14..21]`; flipped README 🚧 rows to real topics/hours (total → ~198–259h, 27 modules); CLAUDE.md → 27-module + a "Framework Modules (15,18,19)" subsection. (STYLE.md left as-is — archetypes unchanged; harness documented in the command files + CLAUDE.md.)
- **Final gate:** lint OK (27), `bun test` baseline (32 pass), `apps/workflow-lab` toolchain runs (stubs fail intentionally), `eve eval` passes offline, all 27 module + command slots present.

**Deviation from plan:** Phase A deferred the eve-agent Mistral swap to Module 18 — but the offline-eval requirement made `mockModel` the right *default* for `apps/eve-agent` (Section 6 teaches the one-line swap to `mistral(...)`). Modules were authored more consolidated than the per-section task list (content is creative work), but each shipped lint-passing + with a verified runnable harness.
