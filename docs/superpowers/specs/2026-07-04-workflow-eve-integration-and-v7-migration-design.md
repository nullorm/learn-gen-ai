# Design: AI SDK v7 Migration + Workflow SDK & Eve Integration

**Date:** 2026-07-04
**Status:** Draft (awaiting review)
**Branch:** `feat/workflow-eve-v7-migration`

## 1. Overview

Two goals, one coordinated change to the Applied LLM Engineering course:

1. **Migrate the whole course from Vercel AI SDK v6 → v7** (package `ai` 6→7, `@ai-sdk/*` providers 3→4, `ai-sdk-ollama` 3→4) plus routine minor bumps.
2. **Teach two Vercel frameworks to full familiarity** — the **Workflow SDK** (`workflow`, durable execution) and **Eve** (`eve`, filesystem-first agent framework) — by adding **3 new modules** and **renumbering** Part IV so workflow modules are contiguous and agent modules are contiguous.

The course grows from **24 → 27 modules**.

### The abstraction ladder (the pedagogical spine)

```
AI SDK primitives  →  Workflow SDK          →  Eve
(generateText,        (durability: 'use step',  (filesystem-first agent
 tools — already        suspend/resume, hooks)    framework, built ON Workflow)
 taught)
```

Eve is *literally built on* the Workflow SDK, so **Workflow must be taught before Eve.** This dependency is the reason Part IV is reordered "workflows first" (see §4).

## 2. Scope

### In scope
- Full v6→v7 migration of all `course/module_*.md` code examples, `src/config.ts`, `tools/*.ts`, and CLAUDE.md guidance.
- Renumbering Part IV to **Layout A** (§4) and shifting Parts V/VI to modules 22–27, including all cross-references, command files, skills, `progress-types.ts`, README, and `lint-course.ts` expectations.
- Three new modules (§5): **15 Durable Workflows (Workflow SDK)**, **18 Eve Fundamentals**, **19 Eve in Production**.
- New on-disk layout for framework code (`apps/` bun workspace, §5.1).
- Dependency bumps + new framework dependencies (§7).
- Convention/doc updates: CLAUDE.md, STYLE.md note, README (§6).

### Out of scope
- Rewriting non-code prose in existing modules beyond what the migration/renumber requires.
- Teaching the **managed** Vercel platform (Vercel Workflows hosting, Agent Runs dashboard, Vercel Connect) beyond a conceptual mention. We teach the **open-source** SDKs that run locally.
- Migrating student `progress.json` state automatically beyond an offered one-time key remap (§4.4).
- Splitting Part IV into two Parts (kept as one; see §6).

## 3. Workstream 1 — AI SDK v6 → v7 migration

`src/` contains only `config.ts` (student-builds-everything), so **almost all edits are in `course/module_*.md` teaching examples.** The migration is well-scoped and mostly mechanical.

> **Module numbers in §3 are pre-renumber (current 1–24).** The migration (§10 step 1) runs *before* the renumber (§10 step 2), so these refer to today's numbering.

### 3.1 What actually breaks (verified against the `ai@7.0.0` changelog + official migration guide)

| Change | Kind | Modules affected |
| --- | --- | --- |
| `stepCountIs` → `isStepCount` | rename | 7, 14, 15, 16 (old numbers) |
| `system` → `instructions`; system-role messages now rejected by default | rename + behavior | ~32 usages course-wide |
| `onFinish` → `onEnd`, `onStepFinish` → `onStepEnd` | rename (aliased) | 1, 3, 6 |
| `fullStream` → `stream` | rename (aliased) | 24 |
| `toUIMessageStreamResponse()` → stateless helper | deprecation | 6 |
| `result.usage` now **accumulates across steps** (`finalStep.usage` = last step) | **semantic** | 7, 14, 15, 16 |
| image content part `{type:'image'}` → `{type:'file',mediaType,data}` | shape | 13 |
| Anthropic `cacheCreationInputTokens` removed from `providerMetadata`; usage token-detail fields relocated | shape | 5 |
| Node 22+, ESM-only | environment | already satisfied (Node 26, `"type":"module"`) |

**Already v7-correct — no change:** embeddings (`embed`/`embedMany`/`cosineSimilarity`), structured output (course already uses `output:` / `result.output` / `partialOutputStream`), `ModelMessage` (and `CoreMessage` still correctly forbidden).

### 3.2 Pre-existing bug to fix (not a v7 issue)
Module 7 (Tool Use) uses `tool({ parameters: ... })`. `parameters` was renamed to `inputSchema` in **v5** — already broken on v6. Fix to `inputSchema` as part of this pass.

### 3.3 Must-verify-at-runtime items (the migration guide is silent or the source is secondary)
These get a verification step in the implementation plan rather than being trusted blind:
1. **Module 5** exact usage field names (`usage.inputTokenDetails.cacheReadTokens` / `usage.outputTokenDetails.reasoningTokens`) — confirm against a real v7 response.
2. **Module 24** `fullStream`/`stream` part-type tags (`text-delta`, `tool-call`, …) — the guide documents no rename of high-level tags, but verify against the v7 `TextStreamPart` type.
3. **`ai-sdk-ollama` v4** `think: false` constructor behavior (CLAUDE.md documents it) — verify against the community provider's own v4 README; update CLAUDE.md if it changed.
4. **`@ai-sdk/workflow`** compatibility with `ai@7` (needed by the Workflow module's `WorkflowAgent`) — confirm the installed version targets v7.

### 3.4 Tooling
- `npx @ai-sdk/codemod v7` exists but operates on `.ts/.js`, **not markdown** — so the module examples are edited by hand/script and reviewed. The codemod may be applied to `src/`/`tools/` where relevant.
- Guardrails after migration: `bun test` (core course green), `bunx tsc --noEmit` typecheck, and `bun run tools/lint-course.ts`.

## 4. Workstream 2 — Renumbering (Layout A)

### 4.1 Target numbering

Part IV reorders to **workflows-first, then the agent arc**; Parts V/VI shift by +3.

| New # | Module | Origin |
| --- | --- | --- |
| **Part IV — Agents & Orchestration** | | |
| 14 | Workflows & Chains | was 16 — **reframe** (see §4.2) |
| 15 | Durable Workflows (Workflow SDK) | **NEW** |
| 16 | Agent Fundamentals | was 14 |
| 17 | Multi-Agent Systems | was 15 |
| 18 | Eve Fundamentals | **NEW** |
| 19 | Eve in Production | **NEW** |
| 20 | Code Generation | was 17 |
| 21 | Human-in-the-Loop | was 18 |
| **Part V — Quality & Safety** | | |
| 22 | Evals & Testing | was 19 |
| 23 | Fine-tuning | was 20 |
| 24 | Safety & Guardrails | was 21 |
| 25 | Cost Optimization | was 22 |
| **Part VI — Production** | | |
| 26 | Observability | was 23 |
| 27 | Deployment | was 24 |

Modules 1–13 are unchanged.

### 4.2 Content reframe required by Layout A
Module 16 "Workflows & Chains" → becomes module **14**, now taught **before** agents. Its current intro contrasts chains against agents ("Module 14 covers the autonomous approach; this is the deterministic counterpart"). Reframe so chains stand on their own as deterministic multi-step pipelines, with agents referenced as "coming later" (forward reference to new 16/17). Cross-links in the module updated accordingly.

### 4.3 Surfaces to update (the renumber blast radius)
- **`course/module_NN_*.md`** — `git mv` the 8 shifting files (old 14–24 → new 14–27, per the map, minding that 14→16, 15→17, 16→14 swap within Part IV). Update **260** `"Module N"` cross-references across 21 files (scripted find/replace with a review pass; watch for prose like "24 modules").
- **`.claude/commands/module-N.md`** — rename to new numbers; update the `course/module_NN_*.md` path, `progress.ts start/quiz/exercise/complete N` calls, and any cross-module mentions inside.
- **`module-N` skills** — the skill identifiers (`/module-16` etc.) are user-facing; renumber to match. Confirm how skills are registered (command-derived vs manifest) and update that source.
- **`tools/progress-types.ts`** — rewrite `MODULE_NAMES` (now 1–27) and `PARTS` (IV = 14–21, V = 22–25, VI = 26–27). `PART_BADGES`, `RANKS`, `XP` unchanged. Badge rules in `progress-engine.ts` reference only modules **1, 2, 3, 8** (below the insertion point) + `Object.keys(MODULE_NAMES).length` — so they need **no** change beyond the tables.
- **`README.md`** — regenerate the curriculum tables, the estimated-time table, and part ranges (IV now 14–21, etc.).
- **`tools/lint-course.ts`** — enforces gap-free sequential numbering and `module_\d+_*.md` naming; the renumber must satisfy it. Run it as a guardrail.
- **CLAUDE.md** — "24-module" → "27-module".

### 4.4 Student progress migration
`progress.json` is gitignored local state keyed by module number as strings. Modules ≥17 change meaning under the renumber. Offer the user a one-time remap script (old→new keys) for their local `progress.json`; do not auto-run destructively. Low priority (local, single-user).

### 4.5 Execution guardrails
Renumber via a script + review, then verify: `lint-course.ts` passes, no stale `Module <old#>` references remain for the shifted set (targeted grep), every command/skill resolves, `bun test` green.

## 5. Workstream 3 — Three new framework modules

### 5.1 On-disk layout, harness, provider (cross-cutting)

**Layout.** Neither framework fits `src/` + `bun test`. Root becomes a **bun workspace** (`"workspaces": ["apps/*"]`); framework code lives in isolated mini-apps:
- `apps/workflow-lab/` — Workflow SDK app (a framework plugin — Hono is already a course dep — or standalone `Bun.serve()` + `@workflow/swc-plugin`; its own `package.json`, `vitest.config.ts`).
- `apps/eve-agent/` — an `eve init` project (`agent/` + `evals/` trees, its own `package.json` pinning `eve`, `engines.node >=24`).

The core `src/` + `tests/` + `bun test` flow for all non-framework modules is **untouched**.

**Harness (the documented exception).**
- Workflow modules → **`@workflow/vitest`** (the SWC directive transform only runs under a build step; `bun test` would silently no-op `'use workflow'`/`'use step'`). Tests use `waitForHook` / `waitForSleep` for deterministic control of suspends.
- Eve module → **`eve eval`** (`defineEval` + matchers), with **`mockModel`** for offline/deterministic assertions.
- These are still **assertion-based** (honoring the course's "no console.log tests" rule) — just a different runner.

**Provider.** Keep the Mistral free-tier default via **direct AI SDK provider objects** — `import { mistral } from '@ai-sdk/mistral'` passed to Eve's `defineAgent({ model: mistral(...) })` and Workflow's `WorkflowAgent({ model: mistral(...) })`. No new key beyond `MISTRAL_API_KEY`. The **Vercel AI Gateway** (dotted model strings, `AI_GATEWAY_API_KEY`) is documented as the alternative. Vercel-only features (Connect-brokered Slack creds, the Agent Runs dashboard, Vercel Sandbox) are taught **conceptually** and marked optional.

**Authoring approach.** Per the user's standing preference, the assistant **authors the scaffold / config / plumbing / harness directly** (this is integration code, not a teaching exercise). The **student-builds rule applies to the conceptual teaching sections**: the assistant writes the failing `@workflow/vitest` / `eve eval` test and describes what to build; the student writes the workflow/step/hook/tool/skill logic.

**Pinned versions.** `workflow@~4.5` (v5 is in beta — pin 4.x; prefer the `WorkflowAgent` + `defineHook` idioms, avoid deprecated `DurableAgent`), `eve@~0.19` (beta, pre-1.0 — pin exact minor; warn API may drift). See §7.

### 5.2 Module 15 — Durable Workflows with the Workflow SDK

**Objective:** understand durable execution and build durable, resumable, human-in-the-loop AI workflows.

Draft sections (archetypes per STYLE.md):
1. **Why durability?** *(Decide/Explore, no test)* — plain async loses state on crash/deploy/long-wait; the replay-from-event-log model; steps as the durability boundary.
2. **Directives & steps** *(Build)* — a workflow with one step; observe replay via `@workflow/vitest`.
3. **Determinism rules** *(Debug)* — fix a workflow whose body uses `Date.now()`/`Math.random()` (must live in a step, not the replayed body).
4. **Starting & inspecting runs** *(Build)* — `start` / `getRun` / `returnValue` from `workflow/api`.
5. **Durable sleep** *(Build/Explore)* — `sleep('7 days')`, zero-compute pause.
6. **Hooks: suspend & resume** *(Build)* — `defineHook`, resume from a route/test; iterate multiple events.
7. **Retries & errors** *(Build)* — `maxRetries`, `FatalError` / `RetryableError`, idempotency.
8. **Worlds & persistence** *(Decide, no test)* — Local vs Postgres vs Vercel.
9. **Durable AI agents** *(Build)* — `WorkflowAgent` (`@ai-sdk/workflow`) with a tool `execute` marked `'use step'` + `needsApproval` for HITL. Ties back to the agent modules (16/17) and forward to HITL (21).
10. **Going Further** *(optional)* — `WorkflowChatTransport` resumable streams; Bun runtime setup notes.

### 5.3 Module 18 — Eve Fundamentals

**Objective:** build a filesystem-first agent with tools, skills, model config, and evals.

Draft sections:
1. **Filesystem-first** *(Explore, no test)* — `npx eve init`; inspect the `agent/` tree; `eve info`. Identity comes from path (no `name` fields).
2. **`defineAgent` + `instructions.md`** *(Build)* — minimal agent; run via `eve dev`; hit the HTTP session API.
3. **Tools** *(Build)* — `defineTool` (filename = tool name, snake_case); `eve eval` asserts `calledTool`.
4. **Tool approval / HITL** *(Build)* — `approval: once()/always()`; the session parks at `waiting`.
5. **Skills** *(Build)* — `SKILL.md` progressive disclosure (load-on-demand); assert behavior.
6. **Model config & providers** *(Decide/Build)* — direct Mistral object vs Gateway string; `reasoning` effort.
7. **Evals** *(Build)* — `defineEval`, `mockModel` (offline determinism), matchers; contrast with `bun test`.
8. **Going Further** *(optional)* — subagents preview (bridge to 19).

### 5.4 Module 19 — Eve in Production

**Objective:** take an Eve agent to production surfaces — channels, schedules, delegation, external connections, deployment.

Draft sections:
1. **Channels** *(Build/Explore)* — the always-on HTTP channel; override auth via `eveChannel`; Slack conceptual.
2. **Schedules** *(Build)* — cron `defineSchedule`; the dev-only trigger route; the UTC + "dev never auto-fires" gotchas.
3. **Subagents** *(Build)* — `subagents/<id>/agent.ts` + the delegation tool.
4. **Connections** *(Decide/Explore, no test)* — MCP / OpenAPI; credentials via Connect (conceptual, feature-flagged).
5. **Execution model & durability** *(Explore, no test)* — Eve sessions *are* Workflow runs; explicit callback to module 15.
6. **Deploy anywhere** *(Decide/Build)* — `eve build` + `eve start` (Nitro/Node); reverse-proxy must forward `/eve/` **and** `/.well-known/workflow/`; Vercel path conceptual.
7. **Going Further** *(optional)* — sandbox, hooks, full Connect/Slack setup, observability spans (bridge to module 26).

### 5.5 New-module wiring (each of 15, 18, 19)
Every new module needs: `course/module_NN_*.md` (following STYLE.md archetypes + section-numbering lint rules), `.claude/commands/module-N.md` (teaching-flow command, harness-aware), the `module-N` skill registration, an entry in `MODULE_NAMES` + `PARTS[IV]`, README rows, and the `apps/*` scaffold.

## 6. Workstream 4 — Conventions & docs

- **CLAUDE.md:** 24→27 modules; add a **"Framework modules (15, 18, 19)"** subsection documenting the `apps/` workspace layout, the `@workflow/vitest` / `eve eval` harness exception, the Node ≥24 requirement for Eve, pinned framework versions, and the direct-provider (Mistral) default. Update the AI-SDK-patterns bullet to v7 (`isStepCount`, `instructions`, `onEnd`, `stream`; note `output`/`ModelMessage` already correct). Verify/adjust the `ai-sdk-ollama` `think:false` note.
- **STYLE.md:** short note that framework modules use the native harness but keep assertion-based tests and the same archetype vocabulary.
- **README.md:** curriculum + time tables regenerated (§4.3); update the top-line "24 modules" and add the two frameworks to the pitch.
- **Part IV kept as one part** (8 modules, "Agent Deployer" badge). Optional future split noted, not done now.

## 7. Dependencies & versions

**Bump (root, v7 migration):**

| Package | From | To |
| --- | --- | --- |
| `ai` | 6.0.195 | 7.x |
| `@ai-sdk/anthropic` | 3.0.81 | 4.x |
| `@ai-sdk/openai` | 3.0.67 | 4.x |
| `@ai-sdk/groq` | 3.0.39 | 4.x |
| `@ai-sdk/mistral` | 3.0.37 | 4.x |
| `ai-sdk-ollama` | 3.8.4 | 4.x |
| `@lancedb/lancedb` | 0.30.0 | 0.31.0 |
| `hono` | 4.12.23 | 4.12.27 |
| `sharp` | 0.34.5 | 0.35.3 |
| `smol-toml` | 1.6.1 | 1.7.0 |

**Add (framework mini-apps, pinned):** `workflow@~4.5`, `@ai-sdk/workflow` (version targeting `ai@7` — verify), `@workflow/vitest`, `@workflow/swc-plugin`, `vitest`, a Workflow framework plugin if used (e.g. Hono); `eve@~0.19` (+ its `ai`/`zod` peers). These live in `apps/*/package.json`, isolated from root.

## 8. Testing strategy

- **Migrated core (modules 1–14, 16, 17, 20–27):** `bun test` green, `bunx tsc --noEmit` clean, `bun run tools/lint-course.ts` clean.
- **Workflow module (15):** `@workflow/vitest` suite green (deterministic via `waitForHook`/`waitForSleep`).
- **Eve modules (18, 19):** `eve eval` green using `mockModel` for offline determinism where assertions must not hit the network.
- **Renumber:** lint sequential-numbering pass + no stale cross-refs + all commands/skills resolve.

## 9. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| **Eve is beta (v0.19), prefers breaking changes** | Pin exact minor; teach stable primitives; warn students; keep Eve code isolated in `apps/`. |
| **Workflow v5 in beta** | Pin `workflow@4.x`; use current idioms (`WorkflowAgent`, `defineHook`); avoid deprecated `DurableAgent`. |
| **`@ai-sdk/workflow` ↔ `ai@7` compat** | Verify at install (§3.3.4); fall back to raw-AI-SDK-in-a-step approach if incompatible. |
| **v7 silent semantic change** (`usage` now sums steps) | Explicitly covered in migrated modules 7, 14, 16, 17 (new numbering); verify multi-step examples. |
| **Renumber breaks a cross-ref/command/skill** | Script + `lint-course.ts` + grep guardrails + `bun test`; do it as one reviewable commit. |
| **`ai-sdk-ollama` think:false drift** | Verify vs v4 README; update CLAUDE.md. |
| **Node 24 requirement for Eve** | Satisfied locally (Node 26); document as a prereq in the Eve modules + CLAUDE.md. |
| **Provider mismatch** (Eve/Workflow docs default to Anthropic/Gateway) | Standardize on direct `mistral(...)` objects; document Gateway alternative. |

## 10. Suggested execution order (for the implementation plan)

1. **v7 migration** of the existing 24 modules (+ runtime-verify items) — get to green `bun test` / typecheck / lint on the current numbering first.
2. **Renumber** to Layout A as one reviewable change (files, cross-refs, commands, skills, `progress-types`, README, lint).
3. **Workspace + `apps/` scaffolding** and framework dependencies.
4. **Module 15** (Workflow SDK), then **18** and **19** (Eve) — content + command + skill + wiring + native-harness tests.
5. **Docs/conventions** (CLAUDE.md, STYLE.md, README) final pass.

Doing (1) before (2) means the migration runs against stable numbering; doing (2) before (3–4) means new modules are authored at their final numbers.

## 11. Open questions
- None blocking. Three defaults in §5.1 (workspace layout, Mistral-direct provider, single Part IV) are open to veto at review.
