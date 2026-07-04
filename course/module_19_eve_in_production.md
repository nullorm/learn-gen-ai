# Module 19: Eve in Production

## Learning Objectives

- Understand channels: how the default HTTP channel exposes your agent, and when to author a custom one
- Start an agent on a cadence with a cron `defineSchedule`, and trigger it in dev
- Delegate work to subagents — the built-in `agent` tool and declared specialists — and assert delegation
- Connect an agent to external services with MCP and OpenAPI connections
- Explain why every Eve session is a durable Workflow run, and what that buys you in production
- Deploy an Eve agent anywhere Node runs with `eve build` + `eve start`, and know the one reverse-proxy rule that matters

> *Module 19 is part of **Part IV: Agents & Orchestration**, completing the **Agent Deployer** badge. Module 18 built the agent; this module ships it.*

---

## Why Should I Care?

A fundamentals agent answers when you talk to it in the terminal. A *production* agent listens on Slack, wakes itself up every morning to post a digest, delegates hard subtasks to specialists, calls your internal APIs, and keeps running across deploys and week-long waits. Eve gives you each of those as a conventional file — a channel, a schedule, a subagent, a connection — and runs the whole thing on the durable Workflow SDK you learned in Module 15.

This module is the operational half of Eve. Several sections are **conceptual by design** — deploying to Slack or Vercel isn't something you unit-test — so we lean on the *Decide* and *Explore* archetypes: understand the trade-off, know the one gotcha that bites everyone, and build the parts that *are* testable (schedules via `eve info` + a manual dispatch; subagents with `eve eval`).

Same as Module 18 — everything is in `apps/eve-agent/` with the `eve` toolchain and Node ≥24, not `bun test`. The Vercel-specific bits (Slack via Connect, the Agent Runs dashboard, Vercel Sandbox) are shown as concepts and are optional; the open-source runtime is what we actually run.

---

## Connection to Other Modules

- **Module 15 (Durable Workflows)** is the engine under every Eve session — Section 5 makes that concrete.
- **Module 18 (Eve Fundamentals)** built the `apps/eve-agent` this module extends.
- **Module 21 (Human-in-the-Loop)** pairs with channels: an approval parks the session, and a channel renders the approve/deny button.
- **Module 26 (Observability)** consumes the OpenTelemetry spans Eve emits — the Going Further section is the bridge.

---

## Section 1: Channels

A **channel** is the edge between a platform and your agent: it turns an inbound message into a session and delivers the reply. Channel files live in `agent/channels/`, the file stem is the channel id, and the default export is the channel.

The **eve HTTP channel is on by default** — even with no file — serving the session API the terminal UI and `curl` use:

```bash
curl -X POST http://127.0.0.1:3000/eve/v1/session \
  -H 'content-type: application/json' -d '{"message":"weather in Brooklyn?"}'
```

You add `agent/channels/eve.ts` only to *override* defaults (usually auth). For a platform Eve doesn't ship, you author a custom channel with `defineChannel` from `eve/channels` (route handlers `GET/POST/WS`, an `events` map, and a `send` call to start/resume a session). Slack, Discord, Teams, and others are separate built-in factories.

> **Try it:** With the dev server running (`bun x eve dev --no-ui`), hit the default session route with the `curl` above, then attach to `GET /eve/v1/session/:id/stream` and watch the NDJSON lifecycle events. You authored no channel file, yet the HTTP API is fully live — that's the default channel. *(No test — the payoff is seeing the always-on API respond.)*

> **Provider Tip:** Slack credentials are brokered by Vercel Connect, so your channel code never holds a `SLACK_BOT_TOKEN`. That's a Vercel-platform feature; treat the Slack channel as conceptual unless you're deploying there.

---

## Section 2: Schedules

A **schedule** starts the agent on a cron cadence instead of waiting for a message — daily digests, syncs, heartbeats. Each is one file under `agent/schedules/` with a `cron` and exactly one of `markdown` (a fire-and-forget prompt) or `run` (a handler). Schedules are **root-only** (not allowed in subagents).

```typescript
// agent/schedules/heartbeat.ts — task mode
import { defineSchedule } from 'eve/schedules'

export default defineSchedule({
  cron: '*/5 * * * *', // standard 5-field cron, evaluated in UTC
  markdown: 'Check for new critical alerts and summarize any you find.',
})
```

**Build it.** Add `agent/schedules/daily-digest.ts` firing at `'0 9 * * 1-5'` (09:00 UTC weekdays) with a `markdown` prompt telling the agent to post a weather digest. Run `bun x eve info` and confirm the schedule is discovered.

> **Gotcha:** `eve dev` **never fires schedules on their cron cadence** — you'd wait all day. Trigger one manually via the dev-only dispatch route (`POST /eve/v1/dev/schedules/<name>`). Only a built app served with `eve start` runs schedules on the real clock. And cron is **UTC**, not your local time — `'0 9 * * *'` is 9am UTC, which may be the middle of your night.

---

## Section 3: Subagents

A **subagent** is a child agent you delegate a focused subtask to — to parallelize, to narrow a child's tools, or to give a specialist its own identity. Two kinds:

**The built-in `agent` tool** — every agent has it. The model delegates to a *copy of itself* with `{ message, outputSchema? }`. The copy shares the sandbox and tools but starts with fresh history. Emit several `agent` calls in one turn to fan out independent subtasks concurrently.

**Declared subagents** live at `agent/subagents/<id>/agent.ts` and use the same `defineAgent`, but `description` is **required** — omitting it fails to compile (the parent reads it to decide whether to delegate):

```typescript
// agent/subagents/researcher/agent.ts  → subagent `researcher`
import { defineAgent } from 'eve'

export default defineAgent({
  description: 'Investigate ambiguous questions before the parent responds.',
  model: 'anthropic/claude-sonnet-5', // or a mockModel fixture for offline evals
})
```

A declared subagent is its own isolation boundary — it sees only the tools/skills/instructions authored under its own directory, not the root's.

**Build it.** Add a `researcher` subagent under `apps/eve-agent/agent/subagents/researcher/` with a `description` and (for offline evals) a `mockModel`. Write an eval that sends a question the agent should delegate and asserts `t.calledSubagent('researcher')`. Confirm `eve info` lists the subagent.

---

## Section 4: Connections

A **connection** gives the agent tools backed by an external service, without you hand-writing each tool. Two kinds, both in `agent/connections/`:

- **MCP** — `defineMcpClientConnection` (from `eve/connections`) points at a Model Context Protocol server; its tools appear to the model as `linear__list_issues`-style names.
- **OpenAPI** — `defineOpenAPIConnection` turns an OpenAPI spec into typed tools.

Credentials are brokered by Vercel Connect, so the model sees tool names, never URLs or tokens.

Suppose you need your agent to file Linear issues. You could (a) hand-write a `create_issue` tool that calls Linear's REST API with a token in `process.env`, or (b) add an MCP or OpenAPI connection and let Eve generate the tools with Connect-managed credentials. What do you trade in each direction — control and simplicity vs. breadth and credential hygiene? When is a single hand-written tool the right call anyway? *(No test — this is an integration-design decision, and Connect needs the Vercel platform.)*

---

## Section 5: The Execution Model — Why Eve Is Durable

Here's the payoff for having done Module 15 first: **every Eve session is a durable Workflow run.** Run `eve info` in `apps/eve-agent` and find the "Workflow Build" line — the compiler is right there in the output. That's not a metaphor; Eve compiles your agent into a workflow whose every model call and tool call is a step.

That single fact explains Eve's production properties:

- A crash or redeploy mid-turn resumes the session from the last completed step — you don't re-run the whole conversation.
- A tool gated with `approval` (Module 18) **parks** the session — a durable hook (Module 15, Section 6) — so "wait for a human" costs nothing and survives deploys.
- A schedule or channel that starts a session gets the same durability for free.

> **Try it:** Re-read Module 15's replay model, then re-read Module 18's approval section. Predict: when an Eve tool call is gated for approval and the human takes two days to click, what is holding that session open, and how much compute does it consume while it waits? Check your answer against the durability guarantees of a Workflow hook. *(No test — the insight is connecting the two frameworks into one mental model.)*

---

## Section 6: Deploy Anywhere

Eve's package promise is "agents that run anywhere." The HTTP host is Nitro; Workflow and Sandbox are pluggable adapters, not hidden Vercel dependencies.

**Anywhere Node runs:**

```bash
eve build                          # compiles .eve artifacts + a standard Node output
PORT=3000 eve start --host 0.0.0.0
```

Self-hosting checklist: use a provider key (or `AI_GATEWAY_API_KEY`) instead of Vercel OIDC; keep the Workflow World's `.workflow-data` on persistent storage; pick a non-Vercel sandbox backend; replace `vercelOidc()` auth in your channel.

**On Vercel:** `eve deploy` maps Workflow → Vercel Workflows, sandbox → Vercel Sandbox, schedules → Vercel Cron, plus the Agent Runs dashboard.

> **Gotcha:** The one deployment rule that bites everyone: a reverse proxy in front of a self-hosted Eve must forward **both** `/eve/` **and** `/.well-known/workflow/`. Restrict it to `/eve/` and sessions will start but then **silently stall forever** — the workflow callbacks never arrive. If your agent accepts a message and then goes quiet in prod, this is the first thing to check.

> **Decision:** You're deploying this weather agent. On Vercel it's `eve deploy` and you're done; off Vercel it's `eve build` + `eve start`. Which fits your constraints — the zero-config managed path, or the portable self-hosted one — and what's the operational cost of each? *(No test — deployment is an infrastructure decision.)*

---

## Going Further

Three production surfaces to explore once the core is solid:

- **Sandbox** — `defineSandbox` gives the agent an isolated compute environment for running code it generates (ties to Module 20).
- **Connect + Slack, end to end** — wire the Slack channel with Vercel Connect-managed credentials and author against `defineChannel`.
- **Observability** — Eve emits OpenTelemetry spans for sessions, tool calls, and steps. Point them at the tracer you build in **Module 26** to see a durable agent's full execution in production.

Offer to skip this if you've got what you need; go deep if you're continuing to the Production part (Modules 26–27).

---

## Summary

Eve's production surfaces are all conventional files. **Channels** connect the agent to the world — the eve HTTP channel is on by default, and custom ones use `defineChannel`. **Schedules** (`defineSchedule` with a UTC cron and a `markdown` prompt or `run` handler) start the agent on a cadence — but `eve dev` never auto-fires them, so you dispatch manually while iterating. **Subagents** delegate focused work, either the built-in `agent` tool (a copy of the agent) or declared specialists under `subagents/<id>/` whose `description` is required. **Connections** (MCP, OpenAPI) generate tools from external services with Connect-managed credentials. Under all of it, **every session is a durable Workflow run**, which is why crashes, redeploys, and human-approval pauses are survivable for free. You deploy **anywhere Node runs** with `eve build` + `eve start` (mind the `/eve/` *and* `/.well-known/workflow/` proxy rule), or to Vercel with `eve deploy`. That completes the Eve arc — from a hand-rolled loop (Module 16), to durable orchestration (Module 15), to a production agent framework (Modules 18–19). Modules 20–21 finish Part IV and the Agent Deployer badge.

---

## Quiz

1. You author no `agent/channels/` file at all. Can an external client still send your agent a message over HTTP? Explain. *(easy)*

   **Answer:** Yes. The eve HTTP channel is on by default even with no channel file — it serves the session API (`POST /eve/v1/session` plus the stream route) that the terminal UI and `curl` use. You add `agent/channels/eve.ts` only to *override* its defaults, usually auth.

2. You add a schedule with `cron: '0 8 * * *'` and it never fires while you run `eve dev`. Give both reasons this is expected and how you'd trigger it during development. *(easy)*

   **Answer:** First, `eve dev` never fires schedules on their cron cadence — only a built app served with `eve start` runs the real clock. Second, cron is evaluated in **UTC**, so `'0 8 * * *'` isn't 8am local time anyway. In development you trigger it manually via the dev-only dispatch route: `POST /eve/v1/dev/schedules/<name>`.

3. What is the one required field on a *declared* subagent's `agent.ts`, and why does the compiler reject a subagent without it? *(medium)*

   **Answer:** `description`. It is the subagent's whole interface to its parent — the parent reads it to decide whether to delegate — so a subagent without one gives the parent no basis for that decision and fails to compile.

4. A teammate deploys the agent behind an nginx proxy that only forwards `/eve/`. Sessions start but replies never arrive. What's the cause and the fix? *(medium)*

   **Answer:** The proxy drops `/.well-known/workflow/`, so the Workflow callbacks that drive every step never reach the app — sessions accept a message and then silently stall forever. The fix is to forward **both** prefixes, `/eve/` *and* `/.well-known/workflow/`.

5. Your self-hosted Eve deploy keeps `.workflow-data` on ephemeral disk. What breaks, and when? *(hard)*

   **Answer:** Durability breaks. `.workflow-data` is the Workflow World's record of completed steps — Section 6's checklist says to keep it on persistent storage. Nothing seems wrong until a restart or redeploy: at that moment every in-flight session (a parked approval, a waiting schedule, a half-finished multi-step run) loses its recorded step results, so it can't resume from the last completed step — it stalls dead or re-executes steps whose side effects already happened.

---

## Exercises

1. **Digest schedule.** Add `agent/schedules/daily-digest.ts` (`'0 9 * * 1-5'`, a `markdown` prompt). Confirm `eve info` discovers it, then trigger it via the dev dispatch route and observe the session it starts. Write a short note on why a `run` handler (vs `markdown`) would be needed to deliver the digest to a specific channel.
2. **Delegating agent.** Build the `researcher` declared subagent (with a `mockModel` fixture). Write an eval that asserts the root agent delegates a research-flavored question with `t.calledSubagent('researcher')` and does *not* delegate a simple weather question.
3. **Deploy dry-run.** Run `eve build` in `apps/eve-agent` and inspect the generated `.output`/`.eve` artifacts (gitignored). Write down the self-hosting checklist for this specific agent: which env vars it needs, where `.workflow-data` must live, and the exact two path prefixes your proxy must forward.
