# Module 15: Durable Workflows

## Learning Objectives

- Explain what "durable execution" means and why a plain `async` function cannot survive a crash, deploy, or a week-long wait
- Use the `'use workflow'` and `'use step'` directives to split orchestration from side effects
- Understand the replay model: why a workflow body runs many times and a step runs once
- Follow the determinism rule and recognize code that breaks it
- Start, inspect, and await runs with `start` / `getRun` from `workflow/api`
- Pause a workflow for days with `sleep` — at zero compute cost
- Suspend a workflow on a `hook` and resume it later from an external event (webhook, human approval)
- Control retries and failures with `maxRetries`, `RetryableError`, and `FatalError`
- Choose a persistence "World" (Local, Vercel, Postgres) for a given deployment
- Make an AI agent's tool loop durable with `WorkflowAgent`, including human-in-the-loop approval

> *Module 15 is part of **Part IV: Agents & Orchestration**, building toward the **Agent Deployer** badge. Module 14 taught in-memory chains; this module makes them survive anything.*

---

## Why Should I Care?

In Module 14 you built chains: deterministic pipelines of LLM calls. They work beautifully — until the process dies. A chain lives entirely in one process's memory. If the server crashes on step 4 of 6, redeploys mid-run, or needs to wait three days for a human to approve a draft, the entire call stack is gone. You start over, re-paying for every LLM call you already made.

Real production systems can't work that way. An order-fulfillment pipeline that charges a card, waits for a warehouse, then emails a receipt might span hours. An agent that drafts a contract and waits for a lawyer's sign-off might span days. A subscription flow that sends a check-in email seven days after signup must survive every deploy in between.

The **Workflow SDK** (`workflow` on npm — the open-source library behind Vercel Workflows) solves this with *durable execution*: your async function's progress is persisted to an event log, so it can crash, redeploy, or pause indefinitely and resume exactly where it left off. It's also the foundation Eve (Modules 18–19) builds on.

One logistical difference from the rest of the course: this module does **not** use `bun test`. The `'use workflow'`/`'use step'` directives are inert string literals until a build-time compiler transforms them, and the supported harness is `@workflow/vitest`. All code and tests for this module live in `apps/workflow-lab/`; you run them with `bun run test` from that directory.

---

## Connection to Other Modules

- **Module 14 (Workflows & Chains)** built in-memory chains. This module makes each step durable and retryable, and adds suspend/resume.
- **Module 16 (Agent Fundamentals)** builds an autonomous agent loop. Section 9 here wraps that loop in durability with `WorkflowAgent`.
- **Modules 18–19 (Eve)** are built *on top of* the Workflow SDK — every Eve session is a durable workflow run. What you learn here explains how Eve survives restarts.
- **Module 21 (Human-in-the-Loop)** covers approval UX. Hooks (Section 6) are the durable mechanism that makes "wait for a human" free.

---

## Section 1: From Fragile Async to Durable Execution

### The Spectrum of Durability

A plain `async` function is fragile: its state is the call stack, and the call stack is RAM. Three things routinely destroy it in production — a crash, a deploy, and a long wait. Durable execution trades a little ceremony (splitting your code into a workflow and steps) for immunity to all three.

| Property | Plain `async fn` | Durable workflow |
| --- | --- | --- |
| Survives a crash mid-run | No — starts over | Yes — resumes from the last completed step |
| Survives a redeploy | No | Yes |
| Can wait 7 days | Only by holding a process open (expensive, fragile) | Yes — suspended, zero compute |
| Retries a failed side effect | You hand-roll it | Built in, per step |
| Cost of a pause | A blocked process | A row in a store |

### The Core Idea: Replay

A durable workflow works by **re-running its code from the top and replaying completed steps from an event log**. When the workflow calls a step:

- If that step already ran, its result is read instantly from the log (the step is *not* re-executed).
- If it hasn't run yet, the workflow suspends, the step is enqueued and executed, and the workflow resumes with the result recorded.

So the workflow body executes *many times* over a run's life — each replay races through already-done work until it hits the first not-yet-done step. This one idea explains everything else in this module: why steps are the unit of durability, why the workflow body must be deterministic, and why a suspended workflow costs nothing.

> **Decision:** When should you reach for a durable workflow instead of a Module 14 chain? Consider a pipeline that (a) makes irreversible side effects (charges, emails), (b) runs long enough to cross a deploy, or (c) must wait for an external event. If any is true, durability earns its keep. If it's a fast, in-memory transform that you can safely re-run whole, a plain chain is simpler. Which of your Module 14 chains would you make durable, and why?

*(No test — this is a conceptual framing. The next section is where you start building.)*

---

## Section 2: Directives — Workflows and Steps

The Workflow SDK uses **directives**: string literals at the top of a function body, exactly like React's `'use client'`. A build step (the SWC plugin, wired up for you in `apps/workflow-lab`) rewrites them.

- `'use workflow'` marks the **orchestrator**. It is replayed, sandboxed, and must be deterministic. It contains no raw side effects — only calls to steps.
- `'use step'` marks a **step**: a normal async function with the full Node runtime, automatic retries, and a result that is persisted once.

```typescript
// shape only — you'll write the bodies
export async function greetWorkflow(name: string) {
  'use workflow'
  const greeting = await buildGreeting(name) // a step call
  return greeting
}

async function buildGreeting(name: string) {
  'use step'
  return `Hello, ${name}!`
}
```

**Build it.** In `apps/workflow-lab/src/greeting.ts`, create and export `greetWorkflow(name: string)` (a `'use workflow'` function) that calls a `'use step'` helper `buildGreeting(name)` returning `` `Hello, ${name}!` ``. The failing test in `apps/workflow-lab/tests/greeting.test.ts` starts the workflow with `start(greetWorkflow, ['Ada'])` and expects `run.returnValue` to be `'Hello, Ada!'`. Run `bun run test` from `apps/workflow-lab`.

> **Gotcha:** Step arguments and return values are **serialized** — passed by value, not by reference. Mutating an object inside a step is invisible to the workflow. Always *return* new values from steps.

---

## Section 3: The Determinism Rule

Because the workflow body is replayed, it must make the **same decisions every time** given the same step results. Non-deterministic APIs in the workflow body are a bug: `Date.now()`, `Math.random()`, `crypto`, and direct I/O would each produce different values on each replay and corrupt the run. The compiler rejects some of these outright; others you must avoid by discipline.

The fix is always the same: **push non-determinism into a step**. A step runs exactly once and its result is cached, so `Math.random()` is perfectly fine *inside* a step — it just gets recorded and replayed as a constant.

```typescript
// WRONG — random in the workflow body changes on every replay
export async function pickWorkflow() {
  'use workflow'
  const r = Math.random() // ❌ different each replay
  return await handle(r)
}
```

**Debug it.** `apps/workflow-lab/src/lottery.ts` contains a broken workflow that calls `Date.now()` and `Math.random()` directly in the `'use workflow'` body. Its test asserts the run completes and returns a stable result. Diagnose why replay makes it flaky, then fix it by moving the non-deterministic calls into a `'use step'` function (e.g. `drawTicket()`), so the values are drawn once and replayed. Make the test green.

---

## Section 4: Starting and Inspecting Runs

You don't call a workflow like a function — you `start` it. The control-plane API lives in `workflow/api`.

```typescript
import { start, getRun } from 'workflow/api'

const run = await start(greetWorkflow, ['Ada']) // enqueues, returns a handle immediately
run.runId                    // stable id you can persist
await run.status             // 'running' | 'completed' | 'failed'  (async)
await run.returnValue        // awaits completion, returns the result (async)

const again = getRun(run.runId) // re-attach later, from anywhere
await again.returnValue
```

`start` returns *immediately* with a handle — the work runs in the background (in `@workflow/vitest`, in-process). `returnValue` is an async getter that resolves when the run finishes.

**Build it.** In `apps/workflow-lab/src/orders.ts`, build `processOrder(orderId: string)` — a workflow with two steps: `reserveStock(orderId)` returning `{ reserved: true }`, and `chargeCard(orderId)` returning `{ chargeId: 'ch_' + orderId }`. Return `{ orderId, chargeId, reserved: true }`. I'll hand you a test that starts it and asserts `run.runId` is a non-empty string and `run.returnValue` has the expected shape.

---

## Section 5: Durable Sleep

A workflow can pause for any duration at **zero compute cost** — while it sleeps it's just a row in a store, not a held-open process.

```typescript
import { sleep } from 'workflow'

export async function onboarding(email: string) {
  'use workflow'
  await createUser(email)        // step
  await sleep('7 days')          // durable pause — survives deploys
  await sendCheckInEmail(email)  // step, runs a week later
}
```

`sleep` accepts a duration string (`'7 days'`, `'1m'`), a number of milliseconds, or a `Date`. It's replay-safe: on resume, the SDK knows the sleep is already satisfied and races past it.

**Build it.** In `apps/workflow-lab/src/reminder.ts`, build `reminderWorkflow()` that records `'created'` (a step), sleeps `'1 hour'`, then records `'reminded'` and returns `'done'`. The test drives the sleep forward deterministically:

```typescript
import { start, getRun } from 'workflow/api'
import { waitForSleep } from '@workflow/vitest'

const run = await start(reminderWorkflow, [])
const sleepId = await waitForSleep(run)
await getRun(run.runId).wakeUp({ correlationIds: [sleepId] })
expect(await run.returnValue).toBe('done')
```

> **Try it:** After the test passes, remove the `waitForSleep`/`wakeUp` lines and just `await run.returnValue`. In a real World the workflow would wait a real hour; in the test harness the sleep never auto-fires, so the assertion hangs. That hang is the whole point — sleep is *durable*, not a busy-wait.

---

## Section 6: Hooks — Suspend and Resume

Sleeping waits for *time*. A **hook** waits for an *external event* — a webhook, a Slack button, a human clicking "approve". The workflow suspends on the hook and resumes when someone resumes it by token, from a totally separate request.

```typescript
import { defineHook } from 'workflow'

export const approvalHook = defineHook<{ decision: 'approve' | 'revise'; notes?: string }>()

export async function reviewWorkflow(docId: string) {
  'use workflow'
  const draft = await generateDraft(docId) // step
  const events = approvalHook.create({ token: docId }) // suspends here
  for await (const event of events) {
    if (event.decision === 'approve') { await publish(draft); return { status: 'published' } }
    return { status: 'revising' }
  }
  return { status: 'abandoned' }
}
```

Elsewhere — an API route, a Slack handler, a test — you resume it:

```typescript
import { resumeHook } from 'workflow/api'
await resumeHook(docId, { decision: 'approve' })
```

The workflow can suspend for **seconds or weeks** between `create` and the matching `resume`, surviving any number of restarts.

**Build it.** In `apps/workflow-lab/src/approval.ts`, build `approvalWorkflow(token: string)` using a module-level `defineHook<{ approved: boolean }>()`. The workflow suspends on `hook.create({ token })` and returns `'approved'` or `'rejected'` based on the first event. The test:

```typescript
import { start, resumeHook } from 'workflow/api'
import { waitForHook } from '@workflow/vitest'

const run = await start(approvalWorkflow, ['tok-1'])
const hook = await waitForHook(run, { token: 'tok-1' })
await resumeHook(hook.token, { approved: true })
expect(await run.returnValue).toBe('approved')
```

> **Production Patterns:** This is exactly how durable human-in-the-loop works. The "wait for approval" costs nothing while it waits, and the resuming request (a webhook from your review UI) can arrive hours later against a freshly deployed server. Module 21 builds the approval UX on top of this primitive.

---

## Section 7: Retries and Errors

Steps are the retry boundary. By default a step retries a few times on failure. You tune it by assigning `maxRetries` to the step function, and you control retry-vs-give-up with two error classes.

```typescript
import { FatalError, RetryableError } from 'workflow'

async function callFlakyApi(id: string) {
  'use step'
  const res = await fetch(`https://api.example/${id}`)
  if (res.status === 404) throw new FatalError('Not found — do not retry')
  if (res.status === 429) throw new RetryableError('Rate limited', { retryAfter: '1m' })
  return res.json()
}
callFlakyApi.maxRetries = 5 // property on the step fn
```

- `FatalError` — stop immediately, no more retries.
- `RetryableError` — retry, optionally after a delay (`retryAfter`).
- Anything else — retried up to `maxRetries`.

Because a failed step re-runs, side effects must be **idempotent**: dedupe on a stable key (an order id, an idempotency token) so a retry doesn't double-charge.

**Build it.** In `apps/workflow-lab/src/retry.ts`, build a step `chargeOnce(order)` that throws `FatalError` when `order.amount <= 0` and otherwise returns a charge. Build `paymentWorkflow(order)` that calls it. Write no retry loop — that's the SDK's job. The test you'll get for this section asserts a valid order succeeds and a zero-amount order fails the run with a fatal (non-retried) error.

> **Gotcha:** Setting `maxRetries` higher does not help a `FatalError` — that's the point. Reserve `FatalError` for "retrying can never succeed" (bad input, 404); reserve `RetryableError` for transient failures (rate limits, timeouts).

---

## Section 8: Worlds — Where Durable State Lives

The event log (runs, steps, sleeps, hooks) is stored by a pluggable adapter called a **World**, selected with the `WORKFLOW_TARGET_WORLD` env var.

| World | When | Storage | Notes |
| --- | --- | --- | --- |
| **Local** (default in dev) | development, tests | JSON files in `.workflow-data/` + in-memory queue | zero config; single instance; queue does not survive a restart |
| **Vercel** | deploying to Vercel | managed DB + Vercel Queues | zero config on Vercel; serverless-native |
| **Postgres** (`@workflow/world-postgres`) | self-hosting in prod | Postgres + a long-lived worker | needs a persistent worker process (not serverless); Docker/VM/Railway/Fly |

The Local World is what `@workflow/vitest` runs in-process — great for teaching and CI, wrong for production (its queue is ephemeral and there's no auth).

> **Decision:** You're shipping a durable agent. If it's on Vercel, the Vercel World is zero-config. If it's on your own Kubernetes cluster, you need the Postgres World and a `graphile-worker` process polling the DB — which means you *cannot* run it on pure serverless. How does your target infrastructure decide your World, and what does that imply for the "worker" that executes steps?

*(No test — this is an operational trade-off. Pick the World that matches where you deploy.)*

---

## Section 9: Durable AI Agents

Now the payoff. Module 16 builds an agent as an LLM tool-loop in memory. Wrap that loop in a workflow and every tool call becomes a durable, retryable step — and you can suspend for human approval mid-loop. The current API is `WorkflowAgent` from `@ai-sdk/workflow` (it gives you the same agent loop as the AI SDK, plus persistence and approval).

```typescript
import { WorkflowAgent } from '@ai-sdk/workflow'
import { getWritable } from 'workflow'
import { tool, convertToModelMessages, type UIMessage, type ModelCallStreamPart } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

async function bookFlightStep(input: { flightId: string }) {
  'use step'                       // <-- makes THIS tool durable + retried
  return { confirmation: 'BK-' + input.flightId }
}

export async function agentChat(messages: UIMessage[]) {
  'use workflow'
  const agent = new WorkflowAgent({
    model: mistral('mistral-small-latest'),
    instructions: 'You are a flight booking assistant.',
    tools: {
      bookFlight: tool({
        description: 'Book a flight',
        inputSchema: z.object({ flightId: z.string() }),
        needsApproval: true,       // <-- suspends the workflow for a human
        execute: bookFlightStep,
      }),
    },
  })
  const result = await agent.stream({
    messages: await convertToModelMessages(messages),
    writable: getWritable<ModelCallStreamPart>(),
  })
  return { messages: result.messages }
}
```

Two ideas make this durable: each tool's `execute` is a `'use step'`, so a crash after a tool ran doesn't re-run it; and `needsApproval: true` emits an approval request and **suspends the workflow** — the human can respond seconds or hours later, on a different server, and the loop resumes.

**Build it.** In `apps/workflow-lab/src/durable-agent.ts`, build a `WorkflowAgent`-based `bookingAgent(messages)` with one durable tool whose `execute` is a `'use step'`. Because this calls a real model, the test I'll hand you for this section uses the module's shared Mistral provider and is marked slow; it asserts the returned `messages` array is non-empty. (If no provider key is available, read this section as a pattern and skip running the test — the durability wiring is the lesson, not the model output.)

> **Advanced Note:** In the simpler "one whole turn in a single `'use step'`" approach, individual tool calls are *not* separately durable — a failure re-runs the whole turn. `WorkflowAgent` gives you *per-tool* durability, which is why it's the recommended path in production.

---

## Going Further

Two directions once the core clicks:

- **Resumable streaming to a UI.** `WorkflowChatTransport` (from `@ai-sdk/workflow`) makes a `useChat` stream resumable across refreshes and timeouts — the browser reconnects to the same durable run by id. This is how you build a chat UI that survives a dropped connection.
- **Running it as a real server on Bun.** The `@workflow/vitest` plugin hides the compiler wiring. To run a workflow as an actual HTTP service under Bun, you preload the SWC plugin via `bunfig.toml` and expose the generated handler with `Bun.serve()`. Sketch what the three generated endpoints (`flow`, `step`, `webhook`) are for, and why a workflow always needs an HTTP surface plus a compiler.

Offer to skip this section if you just want the core; go deep if you're heading toward deployment (Module 27).

---

## Summary

Durable execution turns a fragile in-memory chain into something that survives crashes, deploys, and week-long waits. The mechanism is **replay**: the `'use workflow'` body re-runs from the top and replays completed `'use step'` results from an event log, so steps run once and the body must be deterministic. You `start` runs and await `returnValue` via `workflow/api`; you pause for time with `sleep` and for external events with `defineHook` + `resumeHook`, both at zero compute cost. Steps are the retry boundary, tuned with `maxRetries` and the `RetryableError`/`FatalError` classes, and side effects must be idempotent because retries re-run them. Where the event log lives is a **World** choice (Local for dev, Vercel or Postgres for prod). Finally, `WorkflowAgent` makes an AI tool-loop durable — each tool a step, `needsApproval` a durable human pause — which is exactly how Eve (next) runs every agent session.

---

## Quiz

1. Why does a `'use workflow'` body run many times over a single run's lifetime, while a `'use step'` runs only once? *(easy)*

   **Answer:** Durable execution works by replay: after every suspension the workflow body re-runs from the top, reading each completed step's result from the event log instead of re-executing it. Steps are the unit of durability — each executes once, and its recorded result is replayed on every subsequent pass.

2. Your workflow body calls `Date.now()` to timestamp a record and the run behaves erratically after a retry. What's the rule being violated, and what's the one-line fix? *(easy)*

   **Answer:** The determinism rule — the replayed body must make the same decisions every time, and `Date.now()` yields a different value on each replay. Move the call into a `'use step'` function: the step runs once, the timestamp is recorded, and every replay sees the same constant.

3. A workflow needs to wait for a customer to click a confirmation link that may arrive in 10 minutes or 10 days. Which primitive do you use — `sleep` or a `hook` — and why does the wait cost nothing either way? *(medium)*

   **Answer:** A hook — `sleep` waits for a known duration, while a hook suspends until an external event resumes it by token (`resumeHook`), whenever that happens to arrive. Either way the suspended run is just persisted state in the store, not a held-open process, so the wait consumes zero compute.

4. A payment step throws on a 404 (resource gone) and on a 429 (rate limited). Which error class should each throw, and what happens differently for each? *(medium)*

   **Answer:** The 404 should throw `FatalError` — the run fails immediately with no retries, because retrying can never succeed. The 429 should throw `RetryableError` (optionally with `retryAfter`) — the step re-runs after the delay, up to `maxRetries`.

5. You wrap an agent's tool loop in a single `'use step'` instead of using `WorkflowAgent` with per-tool steps. A crash occurs after the third of five tools has run and committed a side effect. What goes wrong on resume, and what property must that tool have had to make it safe? *(hard)*

   **Answer:** The whole turn is one step, so on resume the entire loop re-runs — including the third tool, whose side effect executes a second time. It's only safe if that tool is idempotent (dedupes on a stable key); `WorkflowAgent` avoids the problem by making each tool's `execute` its own `'use step'`, individually durable.

---

## Exercises

1. **Cancellable reminder.** In `apps/workflow-lab/src/`, build a `cancellableReminder(userId)` workflow that sleeps `'1 day'` then runs a `sendReminder` step — unless a cancellation arrives first. Race the sleep against a `defineHook<{ reason: string }>()` created with `{ token: userId }` (`Promise.race`); return `'sent'` if the sleep wins, `'cancelled'` (skipping the send step) if the hook does. Write a test for both paths — `waitForSleep` + `wakeUp` for the sent path, `resumeHook` before waking for the cancelled path — and run it with `bun run test`.
2. **Approval with revision.** Extend the Section 6 `approvalWorkflow` so that on a `{ decision: 'revise', notes }` event it regenerates the draft (a step) and suspends on the hook *again*, looping until approved. Test both the approve-first and revise-then-approve paths with two `resumeHook` calls.
3. **Idempotent charge.** Build a `chargeCard` step that records charged order ids in a module-level `Set` and returns the same `chargeId` for a repeat call instead of "charging" twice. Force a retry (throw a `RetryableError` on the first attempt only) and assert the card is charged exactly once — proving why idempotency matters when steps re-run.
