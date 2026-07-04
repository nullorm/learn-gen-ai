# AI SDK v7 Migration — Implementation Plan (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the entire course from Vercel AI SDK v6 → v7 (and bump minor deps) so every `course/module_*.md` example uses current v7 idioms, verified against the installed type declarations.

**Architecture:** Bump dependencies first, then apply the verified break-list as a sequence of small, reviewable edits — mechanical token renames done repo-wide, judgment edits done per-module with real before/after. This is Plan 1 of 3 (migration → renumber → new modules); it operates on the **current 1–24 numbering**.

**Tech Stack:** TypeScript, Bun, Vercel AI SDK v7 (`ai@7`, `@ai-sdk/*@4`, `ai-sdk-ollama@4`), Zod v4. Content lives in markdown.

> ⚠️ **No compile-time safety net.** No `.ts` in `src/`/`tools/`/`tests/` imports `ai`, so `bun test`/`tsc` stay green regardless of the markdown edits. Verification is therefore: (a) grep occurrence-counts (old → 0 / intentional, new present), (b) inspecting the installed `.d.ts` types in `node_modules` for the semantic changes, and (c) `bun test` + `bunx tsc --noEmit` + `bun run tools/lint-course.ts` to prove the dep bump didn't break the tooling. Every task ends with a commit.

---

## Verified edit surface (from grep across `course/`)

| Pattern | Sites | Modules (current #) | Edit |
| --- | --- | --- | --- |
| `stepCountIs` | 56 | 07, 14, 15, 16 | rename → `isStepCount` (mechanical) |
| `role: 'system'` messages | 23 | 01,02,04,05,06,07,09,20,21 | move to `instructions` / `allowSystemInMessages` (judgment) |
| tool `parameters:` | 8 real | 07 (×7), 15 (×1) | rename → `inputSchema` (07/15 only; **not** 17/20) |
| `onFinish` | 5 | 01, 03, 06 | rename → `onEnd` (code + prose) |
| `fullStream` | 1 | 24 | rename → `stream` |
| `toUIMessageStreamResponse` | 2 | 06 | modernize to stateless helper |
| `type: 'image'` parts | 3 | 13 | → `{type:'file',mediaType,data}` (verify shape) |
| cache usage fields | 2 | 05 | correct v7 field names (verify) |
| multi-step `.usage` | — | 07,14,15,16 | prose review (now accumulates) |

No `experimental_*`, no `totalUsage`, no `CoreMessage` in code, no SDK-option `maxTokens`/`maxSteps` (all custom) — confirmed absent, nothing to do.

---

## Task 1: Bump dependencies + establish baseline

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Record the pre-bump baseline (must be green before we start)**

Run:
```bash
bun test 2>&1 | tail -5
bunx tsc --noEmit 2>&1 | tail -5
bun run tools/lint-course.ts 2>&1 | tail -5
```
**Recorded baseline signature (2026-07-04):** `bun test` → **32 pass, 1 fail + 1 error, both from `tests/core/env.test.ts`** (imports `src/core/env.js`, which the student builds in Module 1 — intentionally absent; do NOT create it). `tsc` → the single matching `TS2307` for that same missing file. `lint-course` → OK. This is the "clean" fresh-course state. The gate for the rest of this plan is **"no NEW failures beyond this signature,"** not all-green.

- [ ] **Step 2: Bump to latest majors + minors**

Run:
```bash
bun add ai@latest @ai-sdk/anthropic@latest @ai-sdk/openai@latest @ai-sdk/groq@latest @ai-sdk/mistral@latest ai-sdk-ollama@latest @lancedb/lancedb@latest hono@latest sharp@latest smol-toml@latest
```

- [ ] **Step 3: Verify the resolved versions**

Run: `bun pm ls 2>&1 | grep -E "^\s*(ai|@ai-sdk|ai-sdk-ollama|@lancedb|hono|sharp|smol-toml)"`
Expected: `ai@7.x`, `@ai-sdk/*@4.x`, `ai-sdk-ollama@4.x`, `@lancedb/lancedb@0.31.x`, `hono@4.12.27+`, `sharp@0.35.x`, `smol-toml@1.7.x`.

- [ ] **Step 4: Confirm tooling still green (the dep bump can't break markdown, but proves config/tools survive)**

Run:
```bash
bun test 2>&1 | tail -5
bunx tsc --noEmit 2>&1 | tail -5
bun run tools/lint-course.ts 2>&1 | tail -5
```
Expected: same green as Step 1.

- [ ] **Step 5: Commit**

```bash
git add package.json bun.lock
git commit -m "chore(deps): bump Vercel AI SDK v6→v7 + minor deps to latest"
```

---

## Task 2: `stepCountIs` → `isStepCount`

**Files:**
- Modify: `course/module_07_tool_use.md`, `course/module_14_agent_fundamentals.md`, `course/module_15_multi_agent.md`, `course/module_16_workflows.md`

Representative transform (uniform across all 56 sites, including the import line):
```diff
- import { generateText, stepCountIs } from 'ai'
- await generateText({ model, prompt, tools, stopWhen: stepCountIs(5) })
+ import { generateText, isStepCount } from 'ai'
+ await generateText({ model, prompt, tools, stopWhen: isStepCount(5) })
```

- [ ] **Step 1: Apply the rename repo-wide in `course/`**

Run:
```bash
cd course && sed -i 's/stepCountIs/isStepCount/g' module_07_tool_use.md module_14_agent_fundamentals.md module_15_multi_agent.md module_16_workflows.md && cd ..
```

- [ ] **Step 2: Verify zero stale occurrences and the new name is present**

Run:
```bash
grep -rn "stepCountIs" course/ | wc -l   # expect 0
grep -rn "isStepCount" course/ | wc -l    # expect 56
```
Expected: `0` then `56`.

- [ ] **Step 3: Commit**

```bash
git add course/
git commit -m "migrate(v7): stepCountIs → isStepCount"
```

---

## Task 3: System prompts → `instructions` (the real v7 break)

v7 rejects `role:'system'` inside `messages` by default. Two canonical fixes — pick per site:

**Pattern A — static system prompt on a single call → hoist to `instructions:`** (module_01 style):
```diff
  await generateText({
    model,
-   messages: [
-     { role: 'system', content: 'Your persona instructions here...' },
-     { role: 'user', content: 'The user question here...' },
-   ],
+   instructions: 'Your persona instructions here...',
+   messages: [{ role: 'user', content: 'The user question here...' }],
  })
```

**Pattern B — managed message array where the array structure *is* the lesson** (module_04 conversation-memory style): keep the array, opt in explicitly:
```diff
  await generateText({
    model,
+   allowSystemInMessages: true,
    messages, // history array that includes a leading { role: 'system', ... }
  })
```
Prefer **A**; use **B** only when the module's point is managing a message array that carries the system turn (module_04, and module_05 caching where the system block is the cached prefix).

- [ ] **Step 1: List every site with context to classify A vs B**

Run:
```bash
grep -rn "role: *['\"]system['\"]" course/
```
Expected: 23 sites across module_01, 02, 04, 05, 06, 07, 09, 20, 21. For each, read ~6 surrounding lines and decide A or B.

- [ ] **Step 2: Apply Pattern A / B per site**

Edit each site by hand using the diffs above. Also update option-form `system:` on `generateText`/`streamText` calls to `instructions:` where you touch a call (recommended v7 idiom; `system:` still works so this is opportunistic, not mandatory). Do **not** touch `system:`-looking keys that aren't SDK call options (TOML, prose, unrelated objects).

- [ ] **Step 3: Verify only intentional system-in-messages remain**

Run:
```bash
grep -rn "role: *['\"]system['\"]" course/
```
Expected: every remaining hit sits in a call that also has `allowSystemInMessages: true` (Pattern B), or is inside prose explaining the change. No bare `role:'system'` in a live `messages` array without the opt-in.

- [ ] **Step 4: Commit**

```bash
git add course/
git commit -m "migrate(v7): system-role messages → instructions / allowSystemInMessages"
```

---

## Task 4: Tool `parameters:` → `inputSchema:`

Only the 8 real tool-definition sites (module_07 ×7, module_15 ×1). The `parameters:` in module_17 (a code-gen data struct) and module_20 (`hyperparameters`) must stay.

```diff
  tool({
    description: '...',
-   parameters: z.object({ email: z.email() }),
+   inputSchema: z.object({ email: z.email() }),
    execute: async ({ email }) => { /* ... */ },
  })
```

- [ ] **Step 1: Apply to the two tool-def modules only**

Run:
```bash
cd course && sed -i 's/^\(\s*\)parameters: z\.object/\1inputSchema: z.object/' module_07_tool_use.md module_15_multi_agent.md && cd ..
```
(The anchor `parameters: z.object` only matches Zod tool schemas, not `hyperparameters:` or `parameters: Array<...>`.)

- [ ] **Step 2: Verify the 8 tool sites changed and the false positives are untouched**

Run:
```bash
grep -rn "parameters: z.object" course/            # expect 0
grep -rn "inputSchema: z.object" course/ | wc -l    # expect ≥ 8
grep -rn "hyperparameters:\|parameters: Array" course/ | wc -l  # expect unchanged (module_17/20 intact)
```
Expected: `0`, then `≥8`, then the false-positive count unchanged.

- [ ] **Step 3: Commit**

```bash
git add course/
git commit -m "migrate(v7): tool parameters → inputSchema (fixes pre-existing v5 break)"
```

---

## Task 5: `onFinish` / `onStepFinish` → `onEnd` / `onStepEnd`

Sites: code at module_01:668, module_06:150; instructional prose at module_01:659, module_03:732-734. All are `onFinish` (no `onStepFinish` present). Update code and the prose that tells students to use it.

```diff
  streamText({
    model, prompt,
-   onFinish({ text, usage, finishReason }) { /* ... */ },
+   onEnd({ text, usage, finishReason }) { /* ... */ },
  })
```

- [ ] **Step 1: Apply the rename in the three modules**

Run:
```bash
cd course && sed -i 's/onStepFinish/onStepEnd/g; s/onFinish/onEnd/g' module_01_setup_first_calls.md module_03_structured_output.md module_06_streaming.md && cd ..
```

- [ ] **Step 2: Verify**

Run:
```bash
grep -rn "onFinish\|onStepFinish" course/ | wc -l   # expect 0
grep -rn "onEnd" course/ | wc -l                     # expect 5
```
Expected: `0` then `5`.

- [ ] **Step 3: Commit**

```bash
git add course/
git commit -m "migrate(v7): onFinish/onStepFinish → onEnd/onStepEnd"
```

---

## Task 6: `fullStream` → `stream`

Single site in module_24.

```diff
- for await (const part of result.fullStream) { /* ... */ }
+ for await (const part of result.stream) { /* ... */ }
```

- [ ] **Step 1: Apply**

Run: `cd course && sed -i 's/\.fullStream/.stream/g; s/fullStream/stream/g' module_24_deployment.md && cd ..`

- [ ] **Step 2: Verify**

Run: `grep -rn "fullStream" course/ | wc -l`
Expected: `0`.

- [ ] **Step 3: Commit**

```bash
git add course/
git commit -m "migrate(v7): fullStream → stream"
```

---

## Task 7: Modernize `toUIMessageStreamResponse()` (module_06)

Still works in v7 (deprecated). Modernize to the stateless helper so the course teaches current idiom.

```diff
- return result.toUIMessageStreamResponse()
+ import { createUIMessageStreamResponse, toUIMessageStream } from 'ai'
+ return createUIMessageStreamResponse({ stream: toUIMessageStream({ stream: result.stream }) })
```

- [ ] **Step 1: Confirm the exact helper names in the installed types before editing**

Run: `grep -rn "createUIMessageStreamResponse\|toUIMessageStream" node_modules/ai/dist/index.d.ts | head`
Expected: both symbols exported. If a name differs, use the exact exported name.

- [ ] **Step 2: Read module_06 around both sites and apply the diff** (add the import near the top of the code block; replace each of the 2 call sites).

- [ ] **Step 3: Verify**

Run: `grep -rn "toUIMessageStreamResponse" course/ | wc -l`
Expected: `0` (both converted).

- [ ] **Step 4: Commit**

```bash
git add course/
git commit -m "migrate(v7): modernize UI message stream response helper"
```

---

## Task 8: Multimodal image → file content parts (module_13)

v7 removes the `{ type:'image' }` message part; use a file part with an explicit `mediaType`.

- [ ] **Step 1: Confirm the exact v7 file-part shape from installed types**

Run:
```bash
grep -rn "FilePart\|mediaType" node_modules/ai/dist/index.d.ts | head
```
Expected: a `FilePart`-like type with `type: 'file'`, `data`, and `mediaType`. Use the exact field names it shows.

- [ ] **Step 2: Apply the shape change (3 sites + prose)**

```diff
  content: [
-   { type: 'image', image: imageData },
+   { type: 'file', mediaType: 'image/png', data: imageData },
    { type: 'text', text: prompt },
  ]
```
Also update the prose that says "spread them as separate `{ type: 'image' }` content parts" → `{ type: 'file' }`, and the `analyzeImageFile`/`compareImages` instructions that reference the `image` field → `data` + `mediaType`. Note in the prose that `mediaType` must match the real image (e.g. `image/jpeg`).

- [ ] **Step 3: Verify**

Run: `grep -rn "type: 'image'\|type: \"image\"" course/ | wc -l`
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add course/
git commit -m "migrate(v7): multimodal image parts → file parts with mediaType"
```

---

## Task 9: Caching usage fields (module_05)

Current prose tells students to read `usage.cacheReadInputTokens` / `usage.cacheCreationInputTokens`. v7 relocates these. (The `QueryResult` struct's own `cacheReadTokens`/`cacheWriteTokens` fields are the student's — leave them.)

- [ ] **Step 1: Confirm the exact v7 usage shape from installed types**

Run:
```bash
grep -rn "cacheReadTokens\|cachedInputTokens\|inputTokenDetails\|cacheCreation" node_modules/ai/dist/index.d.ts node_modules/@ai-sdk/anthropic/dist/index.d.ts | head -20
```
Expected: reveals the real fields (spec predicts `usage.inputTokenDetails.cacheReadTokens` on core, and that Anthropic's `cacheCreationInputTokens` moved out of `providerMetadata`). Record the exact paths.

- [ ] **Step 2: Update the prose in module_05** to name the confirmed fields (both the "check `usage` for …" sentence and any dynamic-question note). If cache-creation is only in `usage` now (not `providerMetadata`), say so.

- [ ] **Step 3: Verify**

Run: `grep -rn "cacheReadInputTokens\|cacheCreationInputTokens" course/module_05_long_context_caching.md`
Expected: no stale v6 field names in the SDK-read prose (struct field names may still appear — that's fine).

- [ ] **Step 4: Commit**

```bash
git add course/
git commit -m "migrate(v7): correct cache usage field names (module 5)"
```

---

## Task 10: Multi-step `.usage` semantics review (modules 07, 14, 15, 16)

v7: `result.usage` now **sums all steps** (was final-step); `result.finalStep.usage` is the last step. No rename, but examples that read `.usage` after a multi-step loop now report totals — usually *more* correct, occasionally needs a prose tweak.

- [ ] **Step 1: Find multi-step `.usage` reads**

Run: `grep -rn "\.usage" course/module_07_tool_use.md course/module_14_agent_fundamentals.md course/module_15_multi_agent.md course/module_16_workflows.md`

- [ ] **Step 2: For each, confirm the surrounding prose still reads true.** If a passage says "usage of the final step," either switch the code to `result.finalStep.usage` or update the prose to "total usage across all steps." Add a one-line **Gotcha** callout (STYLE.md vocab) noting the v7 accumulation change where a module teaches multi-step loops.

- [ ] **Step 3: Commit**

```bash
git add course/
git commit -m "migrate(v7): clarify multi-step usage accumulation semantics"
```

---

## Task 11: Runtime-verify stream tags + Ollama `think:false`

- [ ] **Step 1: Confirm high-level stream part tags didn't rename**

Run:
```bash
grep -rn "type: 'text-delta'\|type: 'tool-call'\|type: 'tool-result'\|TextStreamPart" node_modules/ai/dist/index.d.ts | head
```
Expected: `TextStreamPart` union still uses `text-delta` / `tool-call` / `tool-result` tags (only the low-level `ModelCallStreamPart`→`LanguageModelStreamPart` renamed). If a high-level tag changed, update module_24's stream `switch` accordingly and note the site.

- [ ] **Step 2: Confirm `ai-sdk-ollama` v4 `think:false` still works**

Run:
```bash
grep -rn "think" node_modules/ai-sdk-ollama/dist/*.d.ts 2>/dev/null | head
cat node_modules/ai-sdk-ollama/package.json | grep '"version"'
```
Expected: the `think` option still exists on the model constructor. If the API changed, note the corrected form (used in Task 12 for the CLAUDE.md update).

- [ ] **Step 3: Commit any resulting edits** (only if Step 1/2 required a change)

```bash
git add course/ && git commit -m "migrate(v7): verify + fix stream tags / ollama think option" || echo "no changes needed"
```

---

## Task 12: Update CLAUDE.md + STYLE.md to v7

**Files:**
- Modify: `CLAUDE.md`, `course/STYLE.md`

- [ ] **Step 1: Update the "Vercel AI SDK patterns" guidance in CLAUDE.md** — call out the v7 idioms this course now uses: `isStepCount` (not `stepCountIs`), `instructions` (not `system`), `onEnd`/`onStepEnd` (not `onFinish`/`onStepFinish`), `result.stream` (not `fullStream`); note that `Output.object()` / `result.output` and `ModelMessage` are already v7-correct. If Task 11 found an Ollama change, update the `ollama('qwen3.5', { think: false })` note to the confirmed form.

- [ ] **Step 2: Update STYLE.md** if it lists any of the renamed APIs (line 118 already correctly states `ModelMessage`/`CoreMessage`; check for `stepCountIs`/`system`/`onFinish` mentions and fix).

- [ ] **Step 3: Verify no stale API names remain in guidance docs**

Run: `grep -rn "stepCountIs\|onFinish\|fullStream" CLAUDE.md course/STYLE.md`
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md course/STYLE.md
git commit -m "docs(v7): update AI SDK pattern guidance to v7 idioms"
```

---

## Task 13: Final sweep + verification gate

- [ ] **Step 1: Comprehensive stale-pattern sweep across the whole repo**

Run:
```bash
grep -rn "stepCountIs\|onFinish\|onStepFinish\|\.fullStream\|toUIMessageStreamResponse\|parameters: z.object\|type: 'image'\|cacheCreationInputTokens\|cacheReadInputTokens\|CoreMessage\|experimental_" course/ CLAUDE.md README.md
```
Expected: no hits except intentional prose (e.g. a "renamed from … in v7" explanation, or the STYLE.md `CoreMessage` warning).

- [ ] **Step 2: Tooling gate**

Run:
```bash
bun test 2>&1 | tail -5
bunx tsc --noEmit 2>&1 | tail -5
bun run tools/lint-course.ts 2>&1 | tail -5
```
Expected: **the recorded baseline signature from Task 1 Step 1, unchanged** — 32 pass, only `tests/core/env.test.ts` failing/erroring, lint OK (proves the dep bump left the tooling intact and introduced no new breakage).

- [ ] **Step 3: Update README's SDK version mention if present**

Run: `grep -n "v6\|AI SDK\|version" README.md | head`
Then update any concrete v6 reference to v7. (No module numbers change in Plan 1.)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "migrate(v7): final sweep — no stale v6 API references"
```

---

## Self-review checklist (run before handing off)

- [ ] Every spec §3 break has a task: stepCountIs (T2), system (T3), tool params (T4), onFinish (T5), fullStream (T6), UI helper (T7), image parts (T8), cache fields (T9), usage semantics (T10), stream tags + ollama (T11). ✅
- [ ] Verify-at-runtime items (§3.3): cache fields (T9), stream tags (T11), ollama (T11), @ai-sdk/workflow compat → **deferred to Plan 3** (not needed for migration). ✅
- [ ] No module renumbering here (that's Plan 2). ✅
- [ ] Every task ends in a commit; every edit task has a grep/type verification. ✅

## Out of scope (noted, not built)
- A code-block typecheck harness (extract ```ts blocks from modules and compile them against v7) would give the examples a real safety net. Valuable follow-up, but new tooling beyond this migration — recommend as a separate task.

---

## Execution Record (2026-07-04) — COMPLETE

Executed inline on `feat/workflow-eve-v7-migration`. All commits landed; tooling signature preserved (32 pass; only `tests/core/env.test.ts` fails, pre-existing).

**Changed as planned:** T1 deps → v7 (see note below); T2 `stepCountIs→isStepCount` (56); T3 system-role messages (hybrid: modules 01/09 → `instructions`; 04/05 → `allowSystemInMessages` + notes; type-defs/data/prose left); T4 `parameters→inputSchema` (8); T5 `onFinish→onEnd` (5); T6 `fullStream→stream` (1); T9 cache fields → `usage.inputTokenDetails.cacheReadTokens`/`cacheWriteTokens` (mod 05); T10 `result.usage` accumulation note (mod 07); T12 CLAUDE.md v7 guidance.

**Verification turned these into NO-OPs (research/guide were wrong; installed `.d.ts` types are ground truth):**
- **T7** — `result.toUIMessageStreamResponse()` is still a valid method in v7 (`ai/dist` L2931). Kept as-is (module 6).
- **T8** — `ImagePart` (`{ type: 'image', image }`) is still a valid v7 content type (`@ai-sdk/provider-utils` L110). The "image part removed" claim was false. Module 13 unchanged.
- **T11** — high-level stream tags (`text-delta`/`tool-call`/`tool-result`/`finish`) unchanged; `ai-sdk-ollama@4` still exposes `think` (`OllamaChatSettings … Pick<…, 'think'>`). No edits; CLAUDE.md `think:false` note stays.

**Notes / carry-forward:**
- `bun@1.3.14-canary` quirk: `bun add pkg@latest` kept the installed old versions; **installed exact versions** instead (ai@7.0.15, `@ai-sdk/*`@4.0.x, ai-sdk-ollama@4.0.0, lancedb@0.31.0, hono@4.12.27, sharp@0.35.3, smol-toml@1.7.0). Result: those 10 deps are pinned **exact** (not caret) in `package.json` — intentional (course reproducibility); normalize to caret if desired.
- Blocked native postinstalls (`onnxruntime-node`, `protobufjs`, via `@lancedb/lancedb`): run `bun pm trust onnxruntime-node protobufjs` to enable the embeddings/RAG modules locally.
- CLAUDE.md still says "24-module" — correct until Plan 3 adds modules 15/18/19.
