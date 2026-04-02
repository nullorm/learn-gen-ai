# Module 14: Agent Fundamentals

## Learning Objectives

- Understand what an agent is and how it differs from a single LLM call
- Implement the ReAct (Reasoning + Acting) pattern with the Vercel AI SDK
- Build an agent loop that iterates through think, act, and observe phases
- Distinguish between planning-first and reactive agent strategies
- Implement tool selection logic and observation processing
- Add termination conditions including max steps, goal detection, and stuck detection
- Manage within-session agent memory for multi-step tasks
- Debug agents by tracing reasoning steps and diagnosing failures

---

## Why Should I Care?

A single `generateText` call can answer a question, but it cannot research a topic across multiple sources, verify its own claims, or adapt its approach when a first attempt fails. Agents can. An agent is what you get when you give an LLM the ability to use tools and let it run in a loop until it accomplishes a goal. This is where LLM applications stop being fancy autocomplete and start being genuinely useful software.

Every serious LLM product — coding assistants, research tools, customer support bots, data analysis pipelines — is built on agent patterns. Understanding the fundamentals in this module will let you build agents that are reliable, debuggable, and controllable rather than unpredictable black boxes.

If you skip this module, multi-agent systems (Module 15) and workflows (Module 16) will feel like magic you cannot debug. The patterns here are the foundation for everything that follows.

---

## Connection to Other Modules

- **Module 7 (Tool Use)** introduced tool definitions and single-step tool calls. This module extends that into multi-step loops.
- **Module 15 (Multi-Agent Systems)** builds directly on the agent loop pattern, running multiple agents that coordinate.
- **Module 16 (Workflows & Chains)** contrasts the autonomous agent approach with deterministic pipelines.
- **Module 17 (Code Generation)** applies agent patterns to iterative code writing and debugging.
- **Module 18 (Human-in-the-Loop)** adds approval gates and feedback into the agent loop.

---

## Section 1: What is an Agent?

### The Three Components

An agent is built from three elements working together:

1. **An LLM** — the reasoning engine that decides what to do
2. **Tools** — functions the LLM can call to interact with the world
3. **A loop** — the control flow that keeps the LLM running until the task is done

A single `generateText` call with tools is not an agent. It is a one-shot tool call. An agent runs that call repeatedly, feeding results back in, until it decides the task is complete.

Consider this single call with a tool:

```typescript
const singleCall = await generateText({
  model: mistral('mistral-small-latest'),
  tools: {
    searchWeb: {
      /* ... */
    },
  },
  prompt: 'What is the population of Tokyo?',
})
// singleCall.text might be empty if the model wants to use a tool
// but there is no loop to process the tool result
```

Without a loop, the model calls the tool once but never sees the result. The conversation ends.

> **Beginner Note:** Think of the difference between asking someone a question (single call) versus hiring someone to complete a project (agent). The project requires multiple steps, checking intermediate results, and adjusting the approach.

### From Single Call to Agent

The key insight is that `generateText` with `stopWhen: stepCountIs()` already provides an agent loop. The Vercel AI SDK will automatically re-call the model with tool results until the model either stops calling tools or hits the step limit:

```typescript
import { generateText, stepCountIs } from 'ai'

const agentResult = await generateText({
  model: mistral('mistral-small-latest'),
  stopWhen: stepCountIs(10), // <-- this creates the loop
  tools: {
    /* ... */
  },
  prompt: 'What is the population of Tokyo? Use the search tool to find out.',
})
```

After the call completes, `agentResult.steps` contains an array of every step the agent took. Each step has `toolCalls` (what the model decided to do) and `text` (any reasoning output). You can inspect them:

```typescript
for (const step of agentResult.steps) {
  console.log({ toolCalls: step.toolCalls.map(tc => tc.toolName), hasText: !!step.text })
}
```

Your task: create a file that defines a simple search tool (simulated — just return a hardcoded string for any query) and call `generateText` with `stopWhen: stepCountIs(10)`. Run it and inspect `steps.length` and `result.text`. How many steps does the model take before it answers?

> **Advanced Note:** The `stopWhen: stepCountIs()` parameter is the simplest form of agent loop control in the Vercel AI SDK. For more complex agents that need custom logic between steps, you will build your own loop as shown in Section 3.

### What Makes a Good Agent

Not every task needs an agent. Use an agent when:

- The task requires multiple steps that depend on each other
- The LLM needs to gather information before answering
- The approach might need to change based on intermediate results
- The task involves verification or self-correction

Do not use an agent when:

- A single `generateText` call (with or without `Output.object()`) suffices
- The steps are known in advance and do not depend on LLM decisions (use a workflow instead)
- Latency is critical and you cannot afford multiple LLM calls

---

## Section 2: The ReAct Pattern

### Reasoning + Acting

The ReAct pattern (Yao et al., 2022) is the most widely used agent pattern. It interleaves reasoning (thinking about what to do) with acting (using tools) in a structured cycle:

1. **Thought** — the model reasons about the current state and what to do next
2. **Action** — the model calls a tool
3. **Observation** — the tool result is fed back to the model

This cycle repeats until the model has enough information to give a final answer.

The key to implementing ReAct with the Vercel AI SDK is a good system prompt that encourages explicit reasoning. The `stopWhen: stepCountIs()` parameter provides the loop. The system prompt provides the structure:

```typescript
system: `You are a research agent. For each step:
1. THINK: Reason about what you know and what you still need to find out.
2. ACT: Use a tool to gather information.
3. OBSERVE: Analyze the result and decide if you need more information.

When you have enough information, provide a comprehensive final answer.
Always explain your reasoning before using a tool.`
```

Your task: create a ReAct agent with two tools — a simulated `searchWeb` tool (use a `Record<string, string>` as a lookup table of pre-defined results) and a `calculator` tool (validate the expression against `/^[\d\s+\-*/().]+$/` before evaluating). Give the agent a research task that requires using both tools.

After the agent completes, print a reasoning trace by iterating over `agentResult.steps` and logging each step's `text` (thought), `toolCalls` (actions), and `toolResults` (observations).

### Why ReAct Works

The power of ReAct comes from making the model's reasoning explicit. When the model must articulate its thought process before acting, it makes better decisions about which tools to use and how to interpret results.

Without explicit reasoning, an agent might:

- Call the same tool repeatedly with the same query
- Miss important information in tool results
- Fail to connect information across multiple tool calls

With reasoning, the agent can:

- Identify gaps in its knowledge
- Formulate targeted queries
- Synthesize information from multiple sources
- Recognize when it has enough information to answer

> **Beginner Note:** You do not need to implement ReAct from scratch. The combination of a good system prompt and the Vercel AI SDK's `stopWhen: stepCountIs()` gives you ReAct behavior. The system prompt encourages the model to reason explicitly, and `stopWhen` provides the loop.

> **Advanced Note:** Some models handle ReAct-style reasoning better than others. Claude models naturally tend to reason before acting. For models that rush to tool calls without thinking, you can use a "scratchpad" tool that the model calls to write down its thoughts before using action tools.

---

## Section 3: Agent Loop Implementation

### The Basic Loop

While `stopWhen: stepCountIs()` handles simple cases, building your own agent loop gives you full control over the process — you can insert logging, approval gates, memory management, and other custom logic between each agent step.

The fundamental pattern is a `for` loop that calls `generateText` with `stopWhen: stepCountIs(1)` (single step) on each iteration, appends `response.response.messages` to the conversation, and checks whether the model finished (no tool calls in the last step) or hit the max.

Create a `runAgent` function with this signature:

```typescript
interface AgentConfig {
  model: LanguageModel
  system: string
  tools: Record<string, any>
  maxSteps: number
}

interface AgentResult {
  answer: string
  steps: Array<{
    thought: string
    toolCalls: Array<{ name: string; args: Record<string, unknown> }>
    observations: string[]
  }>
  totalSteps: number
  finished: boolean
}

async function runAgent(config: AgentConfig, task: string): Promise<AgentResult>
```

Inside the function:

1. Initialize `messages: ModelMessage[]` with the user's task
2. Loop up to `config.maxSteps` times
3. Each iteration: call `generateText` with `stopWhen: stepCountIs(1)`, the config's model/system/tools, and current messages
4. Record the step's thought (`response.text`), tool calls (from `response.steps`), and observations (from `response.steps[].toolResults`)
5. Append `response.response.messages` to the conversation
6. Check: if the last step has zero tool calls, the agent is done — return with `finished: true`
7. If the loop ends without finishing, return with `finished: false`

How should you handle the case where `response.text` is empty but the model made tool calls? What about the case where the model neither produces text nor calls tools?

### Adding Step Callbacks

Extend the pattern with a callback that fires after each step. Define:

```typescript
interface StepEvent {
  stepNumber: number
  thought: string
  toolCalls: Array<{ name: string; args: Record<string, unknown> }>
  observations: string[]
  durationMs: number
}

type StepCallback = (event: StepEvent) => void | Promise<void>
```

Create a `runAgentWithCallbacks` function that accepts a task, tools, an `onStep` callback, and a max steps limit. Wrap each `generateText` call with `Date.now()` timing. After processing each step, construct a `StepEvent` and call `await onStep(stepEvent)`.

Test it by passing a logging callback that prints each step's number, duration, thought preview, actions, and observations.

> **Beginner Note:** The custom loop pattern gives you a place to insert logging, approval gates, memory management, and other custom logic between each agent step. The `stopWhen: stepCountIs()` approach in the Vercel AI SDK is simpler but less flexible.

---

## Section 4: Planning vs Reacting

### Two Agent Strategies

Agents can approach tasks in two fundamentally different ways:

1. **Reactive (step-by-step)** — Decide what to do next based only on the current state. This is the basic ReAct pattern.
2. **Planning-first** — Create a plan before executing, then follow the plan (possibly revising it).

Neither is universally better. The right choice depends on the task.

### Reactive Agents

Reactive agents are simpler and work well when:

- The task is exploratory (you do not know what you will find)
- Each step's output heavily influences the next step
- The task is short (fewer than 5 steps)

A reactive agent is just the ReAct pattern from Section 2 — a system prompt that says "decide what to do next based on what you know so far" and `stopWhen: stepCountIs()`.

### Planning Agents

Planning agents create an explicit plan before executing. They work well when:

- The task has multiple independent sub-tasks
- You want the user to review the plan before execution begins
- The task is complex (more than 5 steps)
- Efficiency matters — a plan avoids redundant tool calls

A planning agent has three phases: **plan**, **execute**, **synthesize**.

**Phase 1 — Plan:** Use `Output.object` with a plan schema to generate a structured plan before any tools run:

```typescript
const planSchema = z.object({
  goal: z.string().describe('The overall goal'),
  steps: z
    .array(
      z.object({
        id: z.number(),
        description: z.string(),
        tool: z.string().describe('Which tool to use'),
        query: z.string().describe('What to search/look up'),
        dependsOn: z.array(z.number()).describe('IDs of steps this depends on'),
      })
    )
    .describe('Ordered list of steps to execute'),
})
```

Implement `createPlan(task: string): Promise<Plan>` that calls `generateText` with `Output.object({ schema: planSchema })` and a prompt listing the available tools.

**Phase 2 — Execute:** Implement `executePlan(plan: Plan): Promise<Map<number, string>>` that iterates over plan steps in order. For each step, check that its dependencies are met (all `dependsOn` IDs have results). Gather context from dependency results. Call `generateText` with tools and `stopWhen: stepCountIs(2)` to execute the step. Store the result keyed by step ID.

What should happen when a step's dependencies are not met? Should you skip it, error, or try to reorder?

**Phase 3 — Synthesize:** Implement `synthesize(task: string, plan: Plan, results: Map<number, string>): Promise<string>` that formats all step results and asks the model to synthesize a comprehensive answer.

Wire these together in a `planningAgent(task: string): Promise<string>` function.

> **Advanced Note:** Hybrid approaches work well in practice. Start with a lightweight plan, execute reactively within each step, and revise the plan if new information invalidates it. This balances efficiency with flexibility.

---

## Section 5: Tool Selection

### How Agents Choose Tools

When an agent has multiple tools available, the LLM must decide which one to use. This decision is driven by:

1. **Tool descriptions** — clear, specific descriptions help the model match tools to needs
2. **Parameter schemas** — well-typed parameters with descriptions guide correct usage
3. **System prompt guidance** — explicit instructions about when to use each tool
4. **Context** — the current conversation state and what information is still needed

### Writing Good Tool Descriptions

Tool descriptions are prompts. They should be specific and include examples of when to use (and when not to use) the tool. Compare:

```typescript
// BAD: Vague — "search for stuff" tells the model nothing about scope
description: 'Search for stuff'

// GOOD: Specific scope, clear boundaries, usage guidance
description: 'Search the web for current information, news, or facts. Use this for questions about recent events, statistics, or when you need up-to-date data. Returns a list of relevant snippets. Do NOT use this for calculations or code execution.'
```

Parameter descriptions matter too. Instead of `z.object({ q: z.string() })`, use:

```typescript
z.object({
  query: z.string().describe('A specific search query. Be precise — use keywords rather than full sentences.'),
  maxResults: z.number().optional().default(5).describe('Maximum number of results to return (1-10)'),
})
```

Your task: define a tool set with three tools — `webSearch`, `calculator`, and `readFile`. For each, write a description that explains what the tool does, when to use it, what it returns, and when NOT to use it. Add descriptive parameter schemas with `.describe()` on every field.

### Tool Selection via System Prompt

For complex tool sets, add explicit routing guidance in the system prompt. Create an agent that has `webSearch`, `database`, and `calculator` tools, with a system prompt containing a TOOL SELECTION GUIDE that maps question types to tools. Include rules like "try the most specific tool first" and "if a tool returns no results, try rephrasing before switching tools."

Test it with a question like "How many orders did user 12345 place last month, and what was the average order value?" — does the agent correctly choose `database` over `webSearch` for internal data?

> **Beginner Note:** The model reads tool descriptions like a developer reads API documentation. The better your descriptions, the better the model's tool choices. Spend time on descriptions — they are the most important part of tool definitions.

---

## Section 6: Self-RAG — Adaptive Retrieval

### The Idea: Let the Agent Decide

Standard RAG always retrieves, regardless of whether retrieval is needed. If the user asks "What is 2 + 2?", the pipeline dutifully searches the knowledge base, finds irrelevant chunks, and injects them into the context — wasting tokens and potentially confusing the model.

Self-RAG gives the agent control over the retrieval process. The agent decides:

1. **Whether** to retrieve (is external knowledge needed for this query?)
2. **What** to retrieve (which knowledge source, what search query?)
3. **Whether the results help** (are the retrieved chunks actually useful?)

This is fundamentally an agent behavior, not a retrieval technique — the agent uses retrieval as a tool and reasons about when to invoke it.

### Implementing Self-RAG

Build a `shouldRetrieve` function that uses structured output to decide whether retrieval is needed:

```typescript
const RetrievalDecisionSchema = z.object({
  needsRetrieval: z.boolean().describe('Whether external knowledge is needed'),
  reason: z.string().describe('Why retrieval is or is not needed'),
  searchQuery: z.string().optional().describe('Optimized search query if retrieval is needed'),
})

async function shouldRetrieve(
  query: string,
  conversationContext: string
): Promise<z.infer<typeof RetrievalDecisionSchema>>
```

Use `Output.object({ schema: RetrievalDecisionSchema })` with a system prompt that distinguishes between queries that need retrieval (domain-specific facts, policies, specialized information) and those that do not (simple math, common knowledge, greetings, questions answerable from conversation context).

### Self-RAG in the Agent Loop

In practice, Self-RAG is a tool the agent can choose to invoke — the `search` tool. When the agent has retrieval as one of several available tools, it naturally learns when retrieval is useful through the ReAct loop:

```typescript
// The agent has multiple tools including search
const tools = {
  search: { description: 'Search the knowledge base for relevant information' /* ... */ },
  calculate: { description: 'Perform mathematical calculations' /* ... */ },
  lookup_user: { description: 'Look up user account details' /* ... */ },
}

// The ReAct loop handles tool selection — the agent decides whether
// to search, calculate, or look up a user based on the query.
// No special Self-RAG logic needed — it emerges from good tool descriptions.
```

> **Beginner Note:** The simplest form of Self-RAG is just making retrieval a tool that the agent can choose to call or not. You do not need a separate "retrieval decision" step if your agent loop already handles tool selection well.

> **Advanced Note:** For more sophisticated Self-RAG, the agent can also assess retrieval results before using them. After retrieving chunks, it can decide "these chunks don't actually help — I'll answer from my own knowledge" or "I need to search again with a different query."

---

## Section 7: Observation Processing

### Feeding Results Back

After a tool executes, its result becomes an "observation" that the agent uses for its next decision. The quality of observations directly affects agent performance.

### Structured Observations

Tools should return structured JSON observations, not raw data dumps. A good observation includes the data itself, a count of results, and a human-readable note that helps the agent interpret what it received:

```typescript
return JSON.stringify({
  query,
  totalResults: filtered.length,
  results: filtered,
  note:
    filtered.length === 0
      ? 'No products found. Try broadening your search.'
      : `Found ${filtered.length} products matching "${query}".`,
})
```

Build a `searchProducts` tool that accepts a query, optional category, and optional maxPrice. Filter a hardcoded product array and return a structured JSON observation with the fields shown above. Wire it into an agent with a system prompt that instructs the model to analyze results for relevance, note out-of-stock items, consider price-to-rating ratio, and suggest alternative searches if nothing is found.

### Observation Summarization

When tool results are large, the agent may struggle with long context. Summarize observations before feeding them back.

Build a `fetchWebPage` tool whose `execute` function simulates fetching a large page, then calls `generateText` internally to summarize the content into 3-5 bullet points before returning the summary to the agent. Include the original content length and a note that full content was summarized.

What is the trade-off here? When would you lose important details by summarizing?

> **Advanced Note:** Observation summarization introduces a trade-off: you use fewer tokens in the agent's context window, but you may lose details the agent needs later. A good strategy is to keep full observations for the most recent 2-3 steps and summarize older ones.

### Error Observations

Tools fail. When they do, return useful error observations that help the agent recover. Instead of throwing an exception, catch errors inside the tool's `execute` function and return a JSON object with `success: false`, the error message, a `suggestion` field with recovery guidance, and the original parameters. This lets the agent decide whether to retry, use a different tool, or report the failure.

Build an `apiCall` tool that demonstrates this pattern — simulate a failure when the endpoint contains the word "broken" and return a structured error observation.

---

## Section 8: Termination Conditions

### When Should an Agent Stop?

An agent without proper termination conditions can run forever, wasting tokens and money. There are several reasons an agent should stop:

1. **Goal achieved** — the agent has enough information to answer
2. **Max steps reached** — safety limit to prevent runaway agents
3. **Stuck detection** — the agent is repeating actions without progress
4. **Error threshold** — too many consecutive failures
5. **Budget exhaustion** — token or cost limit reached

### Implementing Termination Conditions

Build a `shouldTerminate` function and an agent loop that uses it.

Define the state and config types:

```typescript
interface TerminationConfig {
  maxSteps: number
  maxConsecutiveErrors: number
  maxRepeatedActions: number
  maxTokens: number
}

interface AgentState {
  step: number
  consecutiveErrors: number
  actionHistory: string[]
  totalTokens: number
  messages: ModelMessage[]
}

function shouldTerminate(state: AgentState, config: TerminationConfig): { terminate: boolean; reason: string }
```

Implement four checks inside `shouldTerminate`:

1. **Max steps** — if `state.step >= config.maxSteps`, terminate
2. **Consecutive errors** — if `state.consecutiveErrors >= config.maxConsecutiveErrors`, terminate
3. **Stuck detection** — take the last `config.maxRepeatedActions` entries from `state.actionHistory`. If they are all identical, the agent is stuck
4. **Token budget** — if `state.totalTokens >= config.maxTokens`, terminate

Return `{ terminate: false, reason: '' }` if none triggered.

Then build `runAgentWithTermination` that uses this function. It should be a `while (true)` loop that:

1. Calls `shouldTerminate` before each step
2. Calls `generateText` with `stopWhen: stepCountIs(1)`
3. Tracks tokens via `response.usage.inputTokens + response.usage.outputTokens`
4. Tracks actions by stringifying tool calls (name + args) and pushing to `actionHistory`
5. Tracks errors by checking if any tool result contains `'"success":false'`
6. Resets `consecutiveErrors` to 0 on successful steps
7. Checks if the agent finished naturally (last step has no tool calls)

How would you detect that an agent is alternating between two different tool calls without making progress, rather than repeating the exact same call?

> **Beginner Note:** Always set a step limit via `stopWhen: stepCountIs()`, even during development. An agent without a step limit can run up a significant API bill very quickly. Start with 5-10 steps and increase only if needed.

> **Advanced Note:** Stuck detection is more nuanced than checking for identical actions. A sophisticated agent might call the same tool with different parameters (which is fine) or alternate between two tools without making progress (which is not). Consider tracking the information gained at each step, not just the tools called.

---

## Section 9: Agent Memory

### Within-Session Context

An agent's "memory" within a session is the conversation history — the messages array that accumulates as the agent runs. Managing this memory well is critical for agent performance.

### Context Window Management

As an agent takes more steps, the conversation history grows. Eventually it can exceed the model's context window. Here are strategies to manage this:

```typescript
import { generateText, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface MemoryManager {
  messages: ModelMessage[]
  maxMessages: number
}

function createMemoryManager(maxMessages: number = 50): MemoryManager {
  /* ... */
}

async function addToMemory(memory: MemoryManager, newMessages: ModelMessage[]): Promise<void> {
  /* ... */
}

async function compactMemory(memory: MemoryManager): Promise<void> {
  /* ... */
}

function getMessages(memory: MemoryManager): ModelMessage[] {
  /* ... */
}
```

Build these four functions:

- `createMemoryManager` — returns a new `MemoryManager` with an empty messages array and the given `maxMessages` limit.
- `addToMemory` — pushes new messages onto the array and calls `compactMemory` if the length exceeds `maxMessages`.
- `compactMemory` — keeps the first message (original task) and the most recent half of messages. Summarizes the older messages using `generateText` with a prompt that focuses on key findings, tool results, and decisions. Replaces the old messages with a single summary message.
- `getMessages` — returns a copy of the messages array.

Why keep the first message (original task) during compaction? What would happen if you summarized it away?

### Working Memory: Key-Value Store

For agents that need to track specific facts across many steps, use a structured working memory:

```typescript
import { generateText, stepCountIs } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface WorkingMemory {
  facts: Map<string, string>
  scratchpad: string[]
  currentGoal: string
  subGoals: string[]
  completedGoals: string[]
}

function createWorkingMemory(goal: string): WorkingMemory {
  /* ... */
}

function memoryToPrompt(memory: WorkingMemory): string {
  /* ... */
}

async function agentWithMemory(task: string): Promise<string> {
  /* ... */
}
```

Build these three functions:

- `createWorkingMemory` — initializes the `WorkingMemory` with empty maps/arrays and the given goal.
- `memoryToPrompt` — converts the memory state into a formatted markdown string showing the current goal, known facts (from the `Map`), scratchpad notes, completed sub-goals, and remaining sub-goals. Use checkbox formatting (`[x]` and `[ ]`) for goals.
- `agentWithMemory` — creates working memory, defines four tools (`search`, `addFact`, `addNote`, `completeGoal`) that read and modify the memory state, then calls `generateText` with `stopWhen: stepCountIs(10)`. The system prompt should include the formatted working memory via `memoryToPrompt`. Each memory tool's `execute` function should mutate the memory and return a confirmation string.

Why does injecting the working memory into the system prompt (rather than just keeping it in tool results) help the agent stay organized?

> **Beginner Note:** Working memory tools (addFact, addNote) are a way to help the agent organize information across many steps. Without them, the agent must rely on the full conversation history which can be noisy and hard to parse.

---

## Section 10: Debugging Agents

### Why Agent Debugging is Hard

Agents are non-deterministic and multi-step. A bug might be:

- The model choosing the wrong tool
- A tool returning unexpected data
- The model misinterpreting a tool result
- The model getting stuck in a loop
- The conversation history growing too long and confusing the model

### Building a Debug Tracer

```typescript
import { generateText, stepCountIs, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface TraceEntry {
  timestamp: number
  stepNumber: number
  type: 'thought' | 'action' | 'observation' | 'error' | 'termination'
  content: string
  metadata?: Record<string, unknown>
}

class AgentTracer {
  private traces: TraceEntry[] = []
  private startTime: number = Date.now()

  log(stepNumber: number, type: TraceEntry['type'], content: string, metadata?: Record<string, unknown>): void {
    /* ... */
  }

  getTraces(): TraceEntry[] {
    /* ... */
  }

  printSummary(): void {
    /* ... */
  }
}

async function tracedAgent(task: string): Promise<string> {
  /* ... */
}
```

Build the `AgentTracer` class and `tracedAgent` function:

- `AgentTracer.log` — creates a `TraceEntry` with a relative timestamp (`Date.now() - startTime`), pushes it to the traces array, and prints a real-time console line with a type-specific prefix (`[THINK]`, `[ACT]`, `[OBS]`, `[ERR]`, `[END]`).
- `AgentTracer.getTraces` — returns a copy of the traces array.
- `AgentTracer.printSummary` — prints a summary showing total steps (unique step numbers), total elapsed time, action count, error count, and a tool usage breakdown (count per tool name from action entries' metadata).
- `tracedAgent` — creates a tracer, runs a manual agent loop (up to N steps), calling `generateText` with `stopWhen: stepCountIs(1)` per iteration. After each response, log thoughts (from `response.text`), actions (from `toolCalls`), and observations (from `toolResults`). Append response messages to the conversation. If the last step has no tool calls, the agent is done. Handle errors with `try/catch` and log them.

What information does the trace summary give you that raw logs do not? How would you use tool usage counts to diagnose a stuck agent?

### Common Agent Failures and Fixes

| Symptom                                         | Likely Cause                                 | Fix                                                                                    |
| ----------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| Agent calls same tool repeatedly with same args | Observation not being processed              | Check that tool results are added to messages correctly                                |
| Agent never uses tools                          | Tool descriptions do not match the task      | Rewrite descriptions to match the agent's goal                                         |
| Agent uses wrong tool                           | Descriptions are ambiguous                   | Add "Use this when..." and "Do NOT use this for..." to descriptions                    |
| Agent runs all steps without answering          | No clear termination signal in system prompt | Add "When you have enough information, respond with your answer without calling tools" |
| Agent gives shallow answers                     | Not enough steps allowed                     | Increase step count in `stepCountIs()` and encourage thorough research                 |
| Agent hallucinates despite having tools         | Model not using tools for factual claims     | Add "Always verify claims using tools" to system prompt                                |

> **Advanced Note:** For production agents, ship traces to an observability platform (Langfuse, LangSmith, or a custom solution). Console logging is fine for development but you need persistent, searchable traces for production debugging.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 11: Production Termination Conditions

### Beyond Max Steps

Section 8 introduced basic termination conditions: max steps and goal detection. Production agents need additional safeguards to handle real-world failure modes.

A production `shouldTerminate` function checks multiple conditions on every iteration:

1. **Token budget exhaustion** — track cumulative input + output tokens across steps. When the running total approaches the model's context window (or your cost budget), stop gracefully
2. **Abort signal** — respect an `AbortController` signal so users (or parent systems) can cancel a running agent
3. **Error threshold** — if the agent encounters N consecutive errors (tool failures, parse errors, API errors), stop rather than burning through remaining steps
4. **Stuck detection** — if the agent makes the same tool call with the same arguments K times in a row, it is looping. Force termination

```typescript
interface TerminationState {
  step: number
  maxSteps: number
  totalTokens: number
  tokenBudget: number
  consecutiveErrors: number
  errorThreshold: number
  recentToolCalls: string[] // JSON-stringified tool calls for duplicate detection
  abortSignal?: AbortSignal
}

function shouldTerminate(state: TerminationState): { terminate: boolean; reason: string } {
  // Check each condition and return the first that triggers
}
```

The function returns both a boolean and a reason string so the agent can include the termination reason in its final response. "I stopped because I used 90% of the token budget" is more useful than silently cutting off.

> **Beginner Note:** Start with max steps and stuck detection — these catch the most common runaway scenarios. Add budget and abort signal when you move to production.

> **Advanced Note:** Different termination conditions warrant different behaviors. Budget exhaustion should trigger a summary of progress so far. Abort signals should clean up immediately. Error thresholds should log diagnostics. Stuck detection should try a different approach before giving up.

---

## Section 12: Tool Orchestration

### Parallel and Sequential Execution

When the model returns multiple tool calls in a single response, the orchestrator decides how to execute them. The two strategies are:

- **Sequential** — execute one at a time, in order. Simpler, easier to debug, and necessary when tools have side effects that depend on each other
- **Parallel** — execute all at once with `Promise.all`. Faster when tools are independent (e.g., searching three different sources simultaneously)

The model decides _which_ tools to call. The orchestrator decides _how_ to run them.

```typescript
// Parallel execution with per-tool error handling
const results = await Promise.allSettled(toolCalls.map(call => executeTool(call.name, call.args)))
```

Per-tool error handling is critical. If the agent calls three tools and one fails, you do not want to abort the entire turn. Use `Promise.allSettled` instead of `Promise.all` — it returns results for all calls, marking failures individually. Feed both successes and failures back to the model so it can decide how to proceed.

Format tool results consistently. Each result should include the tool name, whether it succeeded, and either the result or the error message. This gives the model enough information to retry a failed tool or work around it.

> **Advanced Note:** Some production systems implement a dependency graph for tool calls. If tool B depends on tool A's output, run A first, then B. Independent tools run in parallel. This requires analyzing the tool call arguments to detect dependencies — a useful optimization for agents that make many tool calls per turn.

---

## Section 13: Extended Thinking

### Making the Think Phase Explicit

The ReAct pattern's "think" phase is typically implicit — the model reasons in its output text before deciding on an action. Modern models support **extended thinking**, where the model explicitly allocates tokens to internal reasoning before producing its response.

Extended thinking is the literal implementation of ReAct's think phase. The model spends thinking tokens on:

- Analyzing the current state and available information
- Evaluating which tool to call and why
- Planning multi-step strategies
- Reconsidering previous assumptions

```typescript
const result = await generateText({
  model: anthropic('claude-sonnet-4'),
  stopWhen: stepCountIs(10),
  tools: agentTools,
  providerOptions: {
    anthropic: { thinking: { type: 'enabled', budgetTokens: 5000 } },
  },
  prompt: taskPrompt,
})
```

Thinking tokens are separate from output tokens and are not visible in the final response (unless you stream them for transparency). The trade-off is cost: thinking tokens count toward your bill. Enable extended thinking for complex tasks where decision quality matters (multi-step planning, ambiguous tool selection) and disable it for simple tasks where the overhead is not justified.

> **Beginner Note:** Not all providers support extended thinking. Anthropic's Claude models support it via `providerOptions`. For Ollama models like Qwen3, thinking mode is the default — disable it via the model constructor: `ollama('qwen3.5', { think: false })` when you want faster, cheaper responses.

---

## Section 14: Plan and Build Agent Modes

### Same Agent, Different Constraints

Production agents often support multiple behavioral modes using the same underlying architecture. The two most common modes are:

- **Plan mode** — read-only. The agent can read files, search code, and analyze, but cannot modify anything. Used for exploration, architecture decisions, and impact analysis
- **Build mode** — full access. The agent can create, edit, and delete files. Used for implementation

The implementation is straightforward: same model, same conversation history, but the available tool set changes based on the current mode.

```typescript
const planTools = { read: readFileTool, search: searchTool, glob: globTool }
const buildTools = { ...planTools, write: writeFileTool, edit: editFileTool, delete: deleteTool }

const tools = mode === 'plan' ? planTools : buildTools
```

This demonstrates an important principle: **behavioral constraints come from tool selection, not model changes**. The agent's "personality" changes because it literally cannot perform write operations in plan mode. No prompt engineering needed — the constraint is structural.

The user toggles between modes explicitly. The system prompt can also vary by mode — plan mode might emphasize thorough analysis while build mode emphasizes incremental, testable changes.

---

## Section 15: Max Steps Configuration and Hidden System Agents

### Per-Agent-Type Step Limits

Different agent types need different step limits based on expected task complexity. A primary agent handling an open-ended task might need 200 steps, while a focused subagent performing a single search should finish in 20.

```typescript
const AGENT_STEP_LIMITS: Record<string, number> = {
  primary: 200,
  researcher: 50,
  reviewer: 20,
  compactor: 10,
}
```

Configure step limits per agent type rather than using a single global value. This prevents lightweight agents from running too long and expensive agents from being cut short.

### Hidden System Agents

Not all agents serve the user directly. Some agents run invisibly to maintain the system itself:

- **Compaction agent** — triggers when the context window is 80% full, summarizes the conversation to free space
- **Titling agent** — generates a descriptive title for the conversation after the first few exchanges
- **Summarization agent** — extracts key decisions and action items periodically

These agents run with their own context and tools, and their results are silently integrated. The user never sees them, but the system relies on them. The key architectural insight: agent infrastructure includes agents that maintain the infrastructure.

> **Advanced Note:** Hidden agents are a form of "system housekeeping." They should be lightweight (low max steps, small models), run asynchronously when possible, and fail silently — a titling agent failure should never block the main conversation.

---

## Section 16: Enhanced Debugging with Trace Logging

### Production Trace Logging

Production agents log every step of the agent loop as structured trace events. A trace logger captures the full reasoning chain — tool calls, arguments, results, decisions, timing, and token usage — in a format that can be searched, filtered, and replayed.

```typescript
interface TraceEvent {
  step: number
  timestamp: number
  type: 'tool_call' | 'tool_result' | 'thinking' | 'response' | 'error' | 'termination'
  data: Record<string, unknown>
  durationMs: number
  tokenUsage?: { input: number; output: number }
}
```

The trace logger wraps the agent loop. Before each tool call, it records the call details. After each result, it records the outcome and duration. On errors, it captures the full error context. On termination, it records the reason.

The trace output enables post-hoc debugging: "The agent failed at step 7 because the search tool returned an empty result, and the agent did not retry with a different query." Without traces, you would only see the final failure with no insight into why.

Store traces alongside the conversation. In development, print them to console. In production, ship them to an observability platform where you can query across conversations: "Show me all agent runs where stuck detection triggered in the last 24 hours."

---

## Quiz

### Question 1 (Easy)

What are the three core components of an LLM agent?

- A) Prompt, response, and database
- B) LLM, tools, and a loop
- C) Input, processing, and output
- D) Model, embeddings, and vector store

**Answer: B** — An agent consists of an LLM (the reasoning engine), tools (functions it can call to interact with the world), and a loop (control flow that keeps the LLM running until the task is done). Without the loop, you have a single tool call, not an agent.

---

### Question 2 (Medium)

In the ReAct pattern, what is the correct order of phases?

- A) Act, Think, Observe
- B) Observe, Think, Act
- C) Think, Act, Observe
- D) Plan, Execute, Verify

**Answer: C** — ReAct stands for Reasoning + Acting. The cycle is: Think (reason about what to do), Act (call a tool), Observe (process the tool result). This cycle repeats until the agent has enough information. Option D describes a planning agent, not the ReAct pattern.

---

### Question 3 (Medium)

An agent keeps calling the `searchWeb` tool with the same query "TypeScript history" repeatedly. What is the most likely cause?

- A) The model's temperature is too high
- B) The tool results are not being correctly added back to the conversation messages
- C) The step limit is too low
- D) The model does not support tool use

**Answer: B** — When tool results are not fed back into the conversation, the model has no memory of having already called the tool. It sees the same context every time and makes the same decision. This is the most common agent bug. The fix is to ensure `response.response.messages` are appended to the messages array after each step.

---

### Question 4 (Hard)

When should you prefer a planning agent over a reactive (step-by-step) agent?

- A) When the task is simple and requires fewer than 3 steps
- B) When you need the lowest possible latency
- C) When the task has multiple independent sub-tasks and you want to avoid redundant tool calls
- D) When the tools are unreliable and might fail

**Answer: C** — Planning agents excel when the task can be decomposed into independent sub-tasks because the plan can identify all needed information upfront and avoid redundant searches. For simple tasks (A), a reactive agent is simpler. Planning adds latency (B) because it requires an extra LLM call. Unreliable tools (D) actually favor reactive agents that can adapt on the fly.

---

### Question 5 (Hard)

What is the primary risk of using conversation history as the agent's only form of memory?

- A) The conversation history is not persistent across sessions
- B) The history can grow to exceed the model's context window, causing the agent to lose earlier information
- C) Other agents cannot access the conversation history
- D) The conversation history uses too much disk space

**Answer: B** — As an agent takes more steps, the conversation history grows. Once it exceeds the context window, earlier messages are truncated, and the agent loses access to information from early steps. This is why memory management strategies like summarization and working memory are important for long-running agents. Cross-session persistence (A) is a separate concern. Multi-agent access (C) is addressed in Module 15.

---

### Question 6 (Medium)

An agent has a token budget of 100,000 tokens. After 5 steps, it has consumed 85,000 tokens. The `shouldTerminate` function detects this. Why is it better to terminate gracefully with a progress summary than to let the agent continue until it hits an API error?

a) API errors are always unrecoverable
b) Graceful termination lets the agent summarize what it accomplished so far, giving the user actionable partial results instead of a cryptic error message with no context about progress made
c) Token budgets are always exactly correct
d) The agent cannot make any more tool calls after 85,000 tokens

**Answer: B**

**Explanation:** When an agent hits the context window limit mid-step, the API returns an error and the user gets no useful output — just a failure. Graceful termination at 85% budget usage gives the agent one final step to summarize its findings, report what is still incomplete, and suggest next steps. The user gets partial but useful results instead of nothing. This is especially important for long-running research or analysis tasks where significant work has already been done.

---

### Question 7 (Hard)

A production agent system uses plan mode (read-only tools) and build mode (full tool access). Why is restricting capabilities through tool selection more reliable than restricting through prompt instructions alone?

a) Prompts are ignored by all models
b) Tool selection is a structural constraint — the agent literally cannot call a tool that is not in its tool set, regardless of what the prompt says. Prompt-based restrictions depend on the model following instructions, which is probabilistic and can fail under adversarial or edge-case inputs
c) Tool selection uses fewer tokens than prompt instructions
d) Plan mode agents do not need system prompts

**Answer: B**

**Explanation:** A prompt saying "do not modify files" is a suggestion the model usually follows but occasionally ignores, especially under complex reasoning chains or adversarial inputs. Removing the write tool from the tool set is a hard constraint — the model cannot write files because the tool does not exist in its context. Structural constraints are deterministic and cannot be bypassed. This is a general principle: use structural constraints for safety-critical behavior and prompt instructions for behavioral preferences.

---

## Exercises

### Exercise 1: ReAct Agent for Web Search Research

**Objective:** Build a ReAct agent that can research questions by searching the web, gathering multiple pieces of information, and synthesizing a comprehensive answer.

**Specification:**

1. Create a file `src/exercises/m14/ex01-react-research-agent.ts`
2. Export an async function `researchAgent(question: string, options?: AgentOptions): Promise<ResearchResult>`
3. Define the types:

```typescript
interface AgentOptions {
  maxSteps?: number // default: 8
  verbose?: boolean // default: false — print trace to console
}

interface ResearchStep {
  stepNumber: number
  thought: string
  action?: {
    tool: string
    args: Record<string, unknown>
  }
  observation?: string
  durationMs: number
}

interface ResearchResult {
  answer: string
  steps: ResearchStep[]
  totalSteps: number
  totalDurationMs: number
  toolCallCount: number
  finished: boolean // true if agent finished naturally, false if hit maxSteps
}
```

4. Implement the following tools:
   - `webSearch` — takes a `query` string, returns simulated search results (you may use a Map of pre-defined results or call a real API)
   - `readPage` — takes a `url` string, returns simulated page content
   - `notepad` — takes a `note` string, stores it in an array and returns confirmation (this is the agent's working memory)

5. The agent must:
   - Use a system prompt that encourages the ReAct pattern (think before acting)
   - Track each step's thought, action, and observation
   - Stop when the model provides a final answer without calling tools
   - Stop if `maxSteps` is reached
   - Detect if the agent is stuck (same tool call 3 times in a row) and force termination

6. If `verbose` is true, print each step to the console as it happens

**Example usage:**

```typescript
const result = await researchAgent('What are the main differences between React and Vue.js in 2025?', {
  maxSteps: 8,
  verbose: true,
})

console.log(`Answer: ${result.answer}`)
console.log(`Steps: ${result.totalSteps}, Tool calls: ${result.toolCallCount}`)
```

**Expected output format (verbose mode):**

```
[Step 1] Thought: I need to search for current comparisons between React and Vue.js...
[Step 1] Action: webSearch({ query: "React vs Vue.js 2025 comparison" })
[Step 1] Observation: Found 5 results comparing React and Vue.js...
[Step 1] Duration: 1,204ms

[Step 2] Thought: Let me get more specific data on performance differences...
[Step 2] Action: webSearch({ query: "React vs Vue.js performance benchmarks 2025" })
[Step 2] Observation: Performance benchmarks show Vue 3 is slightly faster in...
[Step 2] Duration: 983ms

...

[Step 5] Thought: I now have enough information to provide a comprehensive answer.
[Step 5] Duration: 1,102ms

=== Research Complete ===
Steps: 5, Tool calls: 4, Duration: 5,892ms
```

**Test specification:**

```typescript
// tests/exercises/m14/ex01-react-research-agent.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: ReAct Research Agent', () => {
  it('should return a non-empty answer', async () => {
    const result = await researchAgent('What is TypeScript?')
    expect(result.answer).toBeTruthy()
    expect(result.answer.length).toBeGreaterThan(50)
  })

  it('should track steps correctly', async () => {
    const result = await researchAgent('Explain REST APIs', { maxSteps: 5 })
    expect(result.steps.length).toBeGreaterThan(0)
    expect(result.steps.length).toBeLessThanOrEqual(5)
    expect(result.totalSteps).toBe(result.steps.length)
  })

  it('should count tool calls', async () => {
    const result = await researchAgent('What is Node.js?')
    expect(result.toolCallCount).toBeGreaterThan(0)
    expect(result.toolCallCount).toBeLessThanOrEqual(result.totalSteps)
  })

  it('should respect maxSteps', async () => {
    const result = await researchAgent('Explain quantum computing in detail', { maxSteps: 3 })
    expect(result.totalSteps).toBeLessThanOrEqual(3)
  })

  it('should detect stuck agent', async () => {
    // This test would require a mock that always returns the same result
    // forcing the agent to repeat the same search
    const result = await researchAgent('Obscure topic with no results', {
      maxSteps: 10,
    })
    expect(result.totalSteps).toBeLessThan(10)
  })
})
```

---

### Exercise 2: Agent with Working Memory

**Objective:** Extend the research agent with a structured working memory that persists facts, tracks sub-goals, and helps the agent stay organized across many steps.

**Specification:**

1. Create a file `src/exercises/m14/ex02-memory-agent.ts`
2. Export an async function `memoryAgent(task: string, options?: MemoryAgentOptions): Promise<MemoryAgentResult>`
3. Define the types:

```typescript
interface MemoryAgentOptions {
  maxSteps?: number // default: 12
  verbose?: boolean // default: false
}

interface WorkingMemory {
  facts: Record<string, string>
  notes: string[]
  subGoals: string[]
  completedGoals: string[]
}

interface MemoryAgentResult {
  answer: string
  memory: WorkingMemory // Final state of working memory
  totalSteps: number
  finished: boolean
}
```

4. In addition to `webSearch` and `readPage`, implement these memory tools:
   - `storeFact` — takes `key` and `value`, stores in working memory
   - `addNote` — takes `note`, adds to notes array
   - `setSubGoals` — takes `goals` (string array), sets the sub-goals list
   - `completeGoal` — takes `goal`, moves it from subGoals to completedGoals

5. Inject the current state of working memory into the system prompt at each step

6. The agent should use working memory to organize a multi-part research task

**Test specification:**

```typescript
// tests/exercises/m14/ex02-memory-agent.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Memory Agent', () => {
  it('should store facts in working memory', async () => {
    const result = await memoryAgent('Research the top 3 JavaScript frameworks')
    expect(Object.keys(result.memory.facts).length).toBeGreaterThan(0)
  })

  it('should track completed goals', async () => {
    const result = await memoryAgent('Compare Python and JavaScript: syntax, performance, and ecosystem')
    expect(result.memory.completedGoals.length).toBeGreaterThan(0)
  })

  it('should produce a final answer using stored facts', async () => {
    const result = await memoryAgent('What are the SOLID principles?')
    expect(result.answer).toBeTruthy()
    expect(result.finished).toBe(true)
  })
})
```

---

### Exercise 3: Production Termination Conditions

**Objective:** Build a `shouldTerminate` function that checks multiple termination conditions and a wrapper that integrates it into an agent loop.

**Specification:**

1. Create a file `src/exercises/m14/ex03-termination.ts`
2. Export a function `shouldTerminate(state: TerminationState): TerminationResult`
3. Define the types:

```typescript
interface TerminationState {
  step: number
  maxSteps: number
  totalTokens: number
  tokenBudget: number
  consecutiveErrors: number
  errorThreshold: number
  recentToolCalls: string[] // JSON-stringified recent tool calls for duplicate detection
  stuckThreshold: number // How many identical calls in a row means "stuck"
  abortSignal?: AbortSignal
}

interface TerminationResult {
  terminate: boolean
  reason: string // e.g., "max_steps", "token_budget", "error_threshold", "stuck", "aborted", "none"
}
```

4. Implement checks in this priority order:
   - **Abort signal** — if `abortSignal?.aborted` is true, terminate immediately
   - **Max steps** — if `step >= maxSteps`
   - **Token budget** — if `totalTokens >= tokenBudget * 0.9` (90% threshold to leave room for a final response)
   - **Error threshold** — if `consecutiveErrors >= errorThreshold`
   - **Stuck detection** — if the last `stuckThreshold` entries in `recentToolCalls` are identical

5. Export an `agentWithTermination` function that runs an agent loop using `shouldTerminate` at each step

**Test specification:**

```typescript
// tests/exercises/m14/ex03-termination.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Production Termination', () => {
  it('should not terminate when all conditions are healthy', () => {
    const result = shouldTerminate({
      step: 3,
      maxSteps: 10,
      totalTokens: 1000,
      tokenBudget: 50000,
      consecutiveErrors: 0,
      errorThreshold: 3,
      recentToolCalls: [],
      stuckThreshold: 3,
    })
    expect(result.terminate).toBe(false)
  })

  it('should terminate on max steps', () => {
    const result = shouldTerminate({
      step: 10,
      maxSteps: 10,
      totalTokens: 1000,
      tokenBudget: 50000,
      consecutiveErrors: 0,
      errorThreshold: 3,
      recentToolCalls: [],
      stuckThreshold: 3,
    })
    expect(result.terminate).toBe(true)
    expect(result.reason).toBe('max_steps')
  })

  it('should terminate on token budget exhaustion', () => {
    const result = shouldTerminate({
      step: 3,
      maxSteps: 10,
      totalTokens: 46000,
      tokenBudget: 50000,
      consecutiveErrors: 0,
      errorThreshold: 3,
      recentToolCalls: [],
      stuckThreshold: 3,
    })
    expect(result.terminate).toBe(true)
    expect(result.reason).toBe('token_budget')
  })

  it('should terminate on consecutive errors', () => {
    const result = shouldTerminate({
      step: 3,
      maxSteps: 10,
      totalTokens: 1000,
      tokenBudget: 50000,
      consecutiveErrors: 3,
      errorThreshold: 3,
      recentToolCalls: [],
      stuckThreshold: 3,
    })
    expect(result.terminate).toBe(true)
    expect(result.reason).toBe('error_threshold')
  })

  it('should detect stuck agent', () => {
    const sameCall = JSON.stringify({ tool: 'search', args: { query: 'test' } })
    const result = shouldTerminate({
      step: 5,
      maxSteps: 10,
      totalTokens: 1000,
      tokenBudget: 50000,
      consecutiveErrors: 0,
      errorThreshold: 3,
      recentToolCalls: [sameCall, sameCall, sameCall],
      stuckThreshold: 3,
    })
    expect(result.terminate).toBe(true)
    expect(result.reason).toBe('stuck')
  })

  it('should terminate on abort signal', () => {
    const controller = new AbortController()
    controller.abort()
    const result = shouldTerminate({
      step: 1,
      maxSteps: 10,
      totalTokens: 100,
      tokenBudget: 50000,
      consecutiveErrors: 0,
      errorThreshold: 3,
      recentToolCalls: [],
      stuckThreshold: 3,
      abortSignal: controller.signal,
    })
    expect(result.terminate).toBe(true)
    expect(result.reason).toBe('aborted')
  })
})
```

---

### Exercise 4: Tool Orchestration

**Objective:** Build a tool executor that handles multiple tool calls per turn with configurable execution strategy and per-tool error handling.

**Specification:**

1. Create a file `src/exercises/m14/ex04-tool-orchestration.ts`
2. Export an async function `executeToolCalls(calls: ToolCall[], options?: ExecutionOptions): Promise<ToolResults>`
3. Define the types:

```typescript
interface ToolCall {
  id: string
  name: string
  args: Record<string, unknown>
}

interface ToolResult {
  id: string
  name: string
  success: boolean
  result?: unknown
  error?: string
  durationMs: number
}

interface ExecutionOptions {
  strategy: 'sequential' | 'parallel'
  continueOnError: boolean // default: true — do not abort all calls if one fails
  timeoutMs?: number // per-tool timeout
}

type ToolResults = ToolResult[]
```

4. Implement two execution strategies:
   - **Sequential** — execute tools one at a time, in order. If `continueOnError` is false, stop on first failure
   - **Parallel** — execute all tools with `Promise.allSettled`. Always returns results for all tools

5. Each tool call should be wrapped with timing and error handling. Failed tools return `{ success: false, error: "message" }` rather than throwing

6. Register tools via a `registerTool(name: string, handler: Function)` pattern

**Test specification:**

```typescript
// tests/exercises/m14/ex04-tool-orchestration.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Tool Orchestration', () => {
  it('should execute tools sequentially', async () => {
    const results = await executeToolCalls(
      [
        { id: '1', name: 'search', args: { query: 'test' } },
        { id: '2', name: 'read', args: { url: 'https://example.com' } },
      ],
      { strategy: 'sequential', continueOnError: true }
    )
    expect(results).toHaveLength(2)
    expect(results.every(r => r.durationMs > 0)).toBe(true)
  })

  it('should execute tools in parallel', async () => {
    const results = await executeToolCalls(
      [
        { id: '1', name: 'search', args: { query: 'a' } },
        { id: '2', name: 'search', args: { query: 'b' } },
      ],
      { strategy: 'parallel', continueOnError: true }
    )
    expect(results).toHaveLength(2)
  })

  it('should handle per-tool errors without aborting', async () => {
    const results = await executeToolCalls(
      [
        { id: '1', name: 'failing_tool', args: {} },
        { id: '2', name: 'search', args: { query: 'test' } },
      ],
      { strategy: 'sequential', continueOnError: true }
    )
    expect(results[0].success).toBe(false)
    expect(results[0].error).toBeTruthy()
    expect(results[1].success).toBe(true)
  })

  it('should stop on first error when continueOnError is false', async () => {
    const results = await executeToolCalls(
      [
        { id: '1', name: 'failing_tool', args: {} },
        { id: '2', name: 'search', args: { query: 'test' } },
      ],
      { strategy: 'sequential', continueOnError: false }
    )
    expect(results).toHaveLength(1)
    expect(results[0].success).toBe(false)
  })
})
```

> **Local Alternative (Ollama):** ReAct agents work with `ollama('qwen3.5')`, which supports tool calling. The agent loop, observation-action cycles, and `stopWhen: stepCountIs()` are provider-agnostic. Local agents are slower but fully private. For complex reasoning tasks, consider `ollama('qwen3.5:cloud')` or `ollama('deepseek-r1')` for better planning capabilities.

---

## Summary

In this module, you learned:

1. **What an agent is:** An LLM plus tools plus a loop. The Vercel AI SDK's `stopWhen: stepCountIs()` provides the simplest agent loop, but custom loops give you more control.
2. **The ReAct pattern:** Think, act, observe — the fundamental cycle that makes agents effective. System prompts encourage explicit reasoning before tool use.
3. **Agent loop implementation:** How to build a custom loop with step tracking, message management, and callbacks for observability.
4. **Planning vs reacting:** Reactive agents work step by step; planning agents create a plan first. Choose based on task complexity and structure.
5. **Tool selection:** Good tool descriptions, typed parameters, and system prompt guidance help agents choose the right tool.
6. **Observation processing:** Structured tool results, summarization of large observations, and helpful error messages improve agent accuracy.
7. **Termination conditions:** Max steps, stuck detection, error thresholds, and token budgets prevent runaway agents.
8. **Agent memory:** Conversation history management, context window compaction, and structured working memory keep agents effective across many steps.
9. **Debugging agents:** Trace logging, step-by-step inspection, and common failure patterns help you diagnose and fix agent issues.
10. **Production termination:** Beyond max steps, production agents check token budgets, abort signals, error thresholds, and stuck detection (repeated identical tool calls) on every iteration.
11. **Tool orchestration:** When the model returns multiple tool calls, the orchestrator decides whether to run them sequentially or in parallel, using `Promise.allSettled` for per-tool error handling.
12. **Extended thinking:** Modern models support explicit thinking tokens that improve decision quality for complex tasks, at the cost of additional token usage.
13. **Plan and build modes:** Behavioral constraints come from tool selection, not prompts — a plan-mode agent literally cannot write files because the write tool is not available.
14. **Per-agent step limits:** Different agent types need different max step values based on expected task complexity, from 10 steps for lightweight subagents to 200 for primary agents.
15. **Hidden system agents:** Compaction, titling, and summarization agents run invisibly to maintain the system, using their own context and tools.
16. **Production trace logging:** Structured trace events (tool calls, results, timing, token usage) enable post-hoc debugging and observability across agent runs.

In Module 15, you will extend these patterns to build systems with multiple agents that coordinate, delegate, and communicate to solve complex tasks.
