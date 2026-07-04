# Module 18: Eve Fundamentals

## Learning Objectives

- Explain Eve's "filesystem-first" model: how a directory layout *is* the agent's configuration
- Scaffold an agent with `eve init` and inspect it with `eve info`
- Configure the agent in `agent.ts` with `defineAgent`, and write its always-on prompt in `instructions.md`
- Add a typed tool with `defineTool` where the **filename is the tool name**
- Gate a dangerous tool behind human approval with an `approval` policy
- Add a load-on-demand **skill** and confirm the model loads it
- Choose a model: a deterministic `mockModel` fixture, a direct provider, or the AI Gateway
- Write assertion-based, offline, deterministic evals with `eve eval` and `mockModel`

> *Module 18 is part of **Part IV: Agents & Orchestration**, building toward the **Agent Deployer** badge. In Module 16 you hand-built an agent loop; Eve is the batteries-included framework that runs one for you — on the durable Workflow SDK from Module 15.*

---

## Why Should I Care?

In Module 16 you built an agent loop by hand: call the model, run a tool, feed the result back, repeat. That teaches you how agents work. But a production agent needs far more — tools, approvals, skills, memory compaction, channels, scheduling, evals, deployment — and hand-wiring all of that is a second job.

**Eve** (`eve` on npm, from Vercel) is a filesystem-first framework for building durable agents. Its thesis: instead of one giant config object, **each capability is a file in a conventional place**. A tool is a file in `agent/tools/`. A skill is a file in `agent/skills/`. Eve discovers them, compiles a manifest, and runs the agent loop for you — every session a durable Workflow run (Module 15), so it survives crashes and long pauses for free.

This module builds a small but real agent and, crucially, learns to **test it deterministically and offline** with Eve's own eval harness — the same "assertions, not vibes" discipline the rest of this course insists on, applied to a full agent.

Eve does not use `bun test` or the course's `src/` layout. It's a separate app at **`apps/eve-agent/`** with its own `agent/` tree, needs **Node ≥24**, and its tests are **`eve eval`** files under `evals/`. Everything in this module happens in `apps/eve-agent/`.

---

## Connection to Other Modules

- **Module 15 (Durable Workflows)** is what Eve runs on — every Eve session is a durable Workflow run. That's why an Eve agent survives restarts.
- **Module 16 (Agent Fundamentals)** hand-built the loop Eve now provides. Compare the two: Eve trades control for batteries.
- **Module 7 (Tool Use)** defined tools with the AI SDK's `tool()`. Eve's `defineTool` is the same idea, one tool per file.
- **Module 19 (Eve in Production)** takes this agent to channels, schedules, subagents, and deployment.

---

## Section 1: Filesystem-First

Eve's core rule: **identity comes from the path.** You never write a `name` or `id` field — a file's location determines what it is.

| Path | Becomes |
| --- | --- |
| `agent/agent.ts` | the agent's runtime config |
| `agent/instructions.md` | the always-on system prompt |
| `agent/tools/get_weather.ts` | tool `get_weather` |
| `agent/skills/forecast.md` | skill `forecast` |
| `agent/channels/slack.ts` | a Slack channel (Module 19) |
| `evals/weather/forecast.eval.ts` | eval `weather/forecast` |

Add a capability by adding a file; rename it and its identity moves with it. There's no registry to keep in sync.

The project already exists at `apps/eve-agent/` (scaffolded with `npx eve init eve-agent`). Explore it:

```bash
cd apps/eve-agent
bun x eve info    # prints discovered tools/skills/channels + diagnostics
```

> **Try it:** Run `eve info` and read the output. Note the "Agent Root", the discovered tool `get_weather`, and "Diagnostics 0 errors". Then add an empty file `agent/tools/echo.ts` and run `eve info` again — watch discovery report a new (broken) tool. Delete it. This is filesystem-first: the tree *is* the config. *(No test — the payoff is seeing discovery react to the filesystem.)*

---

## Section 2: Configuring the Agent

`agent/agent.ts` calls `defineAgent` (from `eve`) to set runtime config. The only required field is `model`. `agent/instructions.md` is the always-on system prompt — identity and standing rules only.

```typescript
// agent/agent.ts — shape only
import { defineAgent } from 'eve'
export default defineAgent({ model: /* a model — see Section 6 */ })
```

```markdown
<!-- agent/instructions.md -->
You are a concise weather assistant. Use the get_weather tool, then state the city,
condition, and temperature.
```

The current `apps/eve-agent` ships with a deterministic **`mockModel`** so evals run offline (Section 6 explains it and the swap to a real provider). For now, focus on the prompt.

**Build it.** Edit `apps/eve-agent/agent/instructions.md` so the agent always answers in one sentence and always names the city. Keep it to standing behavior — no task logic (tasks come from the user turn). Run `bun x eve info` and confirm "Instructions instructions.md" with 0 diagnostics. *(This is a Build section, but the "test" is the next section's eval — instructions shape behavior, which you'll assert on in Section 7.)*

---

## Section 3: Tools

A tool is a typed function the model can call. In Eve, **the filename is the tool name** (must be snake_case), and you export a `defineTool` (from `eve/tools`) with a Zod `inputSchema` and an `execute`.

```typescript
// agent/tools/get_weather.ts  → the model sees a tool named `get_weather`
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

Tools run in your app runtime with full `process.env` — not in a sandbox. The `description` is what the model reads to decide *when* to call the tool, so write it for the model, not for humans.

**Build it.** Add a second tool `agent/tools/get_forecast.ts` — tool name `get_forecast` — taking `{ city: string, days: z.int().min(1).max(7) }` and returning an array of `{ day, condition }` objects. You'll assert the model can call it in Section 7. Run `eve info` and confirm both tools are discovered.

---

## Section 4: Tool Approval — Human-in-the-Loop

Some tools are dangerous (refunds, deletes, sends). Eve gates them with an `approval` policy from `eve/tools/approval`. A gated call **parks the durable session** until a human approves — the same durable suspend you saw with Workflow hooks in Module 15.

```typescript
import { defineTool } from 'eve/tools'
import { always } from 'eve/tools/approval'
import { z } from 'zod'

export default defineTool({
  description: 'Issue a refund for an order.',
  inputSchema: z.object({ orderId: z.string(), amount: z.number() }),
  approval: always(), // or once() / never() (the default)
  async execute(input) {
    return { refunded: input.amount }
  },
})
```

Policies: `never()` (default — no approval), `once()` (approve once per session), `always()` (every call). When a call is gated, the session waits; a channel (Module 19) renders it as an approve/deny button.

**Build it.** Add `agent/tools/issue_refund.ts` gated with `always()`. In Section 7 you'll write an eval that sends "refund order 42 for $10" and asserts the run **parks** (`t.parked()`) with a pending `issue_refund` call, rather than completing. This is durable HITL: the agent stops and waits for a human.

---

## Section 5: Skills

A **skill** is a load-on-demand procedure — progressive disclosure. Eve advertises each skill's one-line `description` and a built-in `load_skill` tool; the model loads the full body only when relevant.

The simplest skill is a markdown file (name from path):

```markdown
<!-- agent/skills/forecast.md  → skill `forecast` -->
When the user asks about multiple days, call get_forecast (not get_weather) and
summarize the trend in one sentence.
```

**Build it.** Add `agent/skills/forecast.md` as above. In Section 7, write an eval that asks "what's the weather this week in Denver?" and asserts the model **loaded the skill** with `t.loadedSkill('forecast')` (sugar for a `load_skill` tool call) and then called `get_forecast`. Because our agent uses a scripted `mockModel`, you'll script that path in the fixture — the point is the *assertion vocabulary*, which is identical against a real model.

> **Gotcha:** A skill changes what the model *knows to do*, not what it *can* do. Executable capability always lives in a tool; a skill just tells the model when and how to use one.

---

## Section 6: Choosing a Model — Fixture, Provider, or Gateway

`defineAgent({ model })` accepts three shapes. This is the decision that makes your agent testable *and* real.

1. **A deterministic fixture** — `mockModel` (from `eve/evals`). Scripts replies with zero network, so `eve eval` runs offline and reproducibly. This course's agent ships with one:

```typescript
import { mockModel } from 'eve/evals'
model: mockModel({
  provider: 'anthropic', modelId: 'claude-sonnet-5', // borrowed identity (see Gotcha)
  respond: ({ toolResults }) =>
    toolResults.length === 0
      ? { toolCalls: [{ name: 'get_weather', input: { city: 'Brooklyn' } }] }
      : `Weather in Brooklyn: ${JSON.stringify(toolResults[0]?.output)}`,
}),
```

2. **A direct provider** — the course default, needs `MISTRAL_API_KEY`:

```typescript
import { mistral } from '@ai-sdk/mistral'
model: mistral('mistral-small-latest')
```

3. **The AI Gateway** — a dotted string routed through Vercel's gateway, needs `AI_GATEWAY_API_KEY`: `model: 'anthropic/claude-opus-4.8'`.

> **Gotcha:** Eve's auto-compaction needs the model's context-window size, which it looks up by `provider/modelId`. A fixture's *made-up* identity has no known size and fails to compile. The fix is to give `mockModel` a **known** identity (like `anthropic/claude-sonnet-5`) purely for that lookup — the responses stay 100% scripted. (This is a v0.19 beta wrinkle; expect it to smooth out.)

> **Decision:** For CI and this course, the fixture wins — offline, free, byte-identical every run. For a demo you actually talk to, swap in `mistral(...)`. Which would you use for a nightly regression suite, and which for a live product demo, and why does the eval code stay *the same* across that swap?

---

## Section 7: Evaluating with `eve eval`

Eve's test harness is `eve eval`, not `bun test`. Each `evals/*.eval.ts` file is one graded case: an `async test(t)` that drives the agent with `t.send(...)` and asserts inline. With the `mockModel` fixture, it's fully deterministic and offline.

Config is one file:

```typescript
// evals/evals.config.ts
import { defineEvalConfig } from 'eve/evals'
export default defineEvalConfig({})
```

A case sends a turn and asserts on behavior and content:

```typescript
// evals/weather/forecast.eval.ts
import { defineEval } from 'eve/evals'
import { includes } from 'eve/evals/expect'

export default defineEval({
  description: 'Uses get_weather and reports the condition',
  async test(t) {
    await t.send('What is the weather in Brooklyn?')
    t.succeeded()                        // the run finished cleanly
    t.calledTool('get_weather')          // it used the tool
    t.check(t.reply, includes('Sunny'))  // the reply names the condition
  },
})
```

Run it: `bun x eve eval` (from `apps/eve-agent`). The assertion vocabulary is rich — `t.succeeded()`, `t.parked()`, `t.calledTool(name, { input, count })`, `t.notCalledTool()`, `t.loadedSkill()`, `t.toolOrder([...])`, and value checks `t.check(value, includes|equals|matches(...))` (matchers from `eve/evals/expect`). Every one is an `expect()`-style gate — no eyeballing output.

**Build it.** Write three evals under `apps/eve-agent/evals/`:

1. `greetings/no-tools.eval.ts` — send `"Hi!"`, assert `t.succeeded()` and `t.notCalledTool('get_weather')` (don't call tools for a greeting). Script the fixture to reply without a tool call for non-weather input.
2. `refund/parks.eval.ts` — send `"refund order 42 for $10"`, assert the run `t.parked()` with a pending `issue_refund` call (Section 4's durable HITL).
3. `forecast/uses-skill.eval.ts` — send a multi-day question, assert `t.loadedSkill('forecast')` then `t.calledTool('get_forecast')` (Section 5).

Run `bun x eve eval` and make all gates green.

---

## Going Further

**Subagents** are the bridge to Module 19. A subagent is a specialist child agent at `agent/subagents/<id>/agent.ts` with its own instructions and tools; the parent delegates to it via a built-in `agent` tool, and you assert delegation with `t.calledSubagent('researcher')`. Sketch a `researcher` subagent that the weather agent could delegate deep questions to — then we build it for real, along with channels, schedules, and deployment, in Module 19.

Offer to skip this if you just want the fundamentals; go deep if you're heading toward the production module next.

---

## Summary

Eve is filesystem-first: a file's **path is its identity**, so you build an agent by adding files under `agent/` — `agent.ts` for config, `instructions.md` for the standing prompt, `tools/<name>.ts` for typed tools (filename = tool name), `skills/<name>.md` for load-on-demand procedures. Dangerous tools are gated with an `approval` policy that **parks the durable session** for a human, reusing the suspend/resume you learned in Module 15. The model is one of three shapes — a deterministic `mockModel` fixture (offline, for tests), a direct provider like `mistral(...)`, or an AI Gateway string — and the same eval code works across all three. Testing is `eve eval`: `defineEval` cases whose `test(t)` drives the agent and asserts with `t.succeeded()`/`t.calledTool()`/`t.parked()`/`t.check(...)`. With the fixture model, those evals run offline and byte-identical every time — assertions, not vibes, for a whole agent. Next, Module 19 takes this agent to channels, schedules, subagents, and deployment.

---

## Quiz

1. In Eve, where does a tool's *name* come from, and what do you change to rename a tool? *(easy)*

   **Answer:** From the filename — `agent/tools/get_forecast.ts` *is* the tool `get_forecast`; there is no `name:` field. Rename the file to rename the tool: in Eve, identity comes from the path.

2. What is the difference between a **tool** and a **skill** — what does each add to the agent? *(easy)*

   **Answer:** A tool adds executable capability — a typed function (Zod `inputSchema` + `execute`) the model can call. A skill adds load-on-demand *instructions*: it changes what the model knows to do, not what it can do, so any actual execution still happens through a tool.

3. You gate a `delete_account` tool with `always()` approval and send a request to use it. Describe what happens to the session, and which Module 15 primitive makes that "wait for a human" cost nothing. *(medium)*

   **Answer:** The gated call **parks** the session — it suspends until a human approves or denies, then resumes. The Module 15 primitive is the durable hook: a parked session holds no process or compute while it waits and survives restarts, so the wait is free no matter how long the human takes.

4. Your `eve eval` suite must run in CI with no provider API key and give identical results every run. Which of the three model shapes do you use, and what one-line change turns the agent "live" for a demo? *(medium)*

   **Answer:** The deterministic `mockModel` fixture — offline, free, and byte-identical on every run. To go live, change the single `model:` line in `defineAgent` to a direct provider such as `mistral('mistral-small-latest')`; the eval code stays the same across the swap.

5. You configure `mockModel` with `provider: 'my-fixtures', modelId: 'weather-bot'` and `eve info` fails to compile with a context-window error. Explain the root cause and the fix — and why the fix doesn't make any real model call. *(hard)*

   **Answer:** Eve's auto-compaction needs the model's context-window size, which it looks up by `provider/modelId` — a made-up identity has no known size, so compilation fails. The fix is to give the fixture a **known** identity (e.g. `provider: 'anthropic', modelId: 'claude-sonnet-5'`). That identity is used only for the size lookup; every response still comes from your script, so nothing ever goes over the network.

---

## Exercises

1. **Two-tool agent.** Ensure `get_weather` and `get_forecast` both exist and are discovered by `eve info`. Write one eval that asserts a single-day question calls `get_weather` (not `get_forecast`) and a multi-day question calls `get_forecast` — using `t.calledTool` and `t.notCalledTool` to pin the routing.
2. **Durable approval.** Add `issue_refund` gated with `once()`. Write an eval that sends two refund requests in one session and asserts the *first* parks for approval and, after `t.respondAll`-ing an approval, the *second* runs without parking. (Explore the `t.respond`/`t.respondAll` HITL drive API from the assertions docs.)
3. **Skill routing.** Add a `units` skill instructing the agent to answer in Celsius when the user mentions a metric country. Write an eval asserting the skill is loaded (`t.loadedSkill('units')`) for a "weather in Paris?" turn and *not* loaded for "weather in Dallas?".
