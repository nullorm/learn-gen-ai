# Module 15: Multi-Agent Systems

## Learning Objectives

- Understand why multiple agents outperform a single agent for complex tasks
- Implement the orchestrator-worker pattern for agent coordination
- Pass context between agents while maintaining separation of concerns
- Manage shared state and private state across agent boundaries
- Design delegation strategies based on capability and topic
- Run agents in parallel for independent sub-tasks
- Implement agent handoff for transferring conversations between specialists
- Handle sub-agent failures with retries, fallbacks, and graceful degradation

---

## Why Should I Care?

A single agent can research, write, and review — but it cannot do all three well at the same time. Just as software teams have frontend developers, backend developers, and QA engineers, multi-agent systems assign different roles to different agents. Each agent has a focused system prompt, a curated tool set, and a clear responsibility.

Multi-agent systems are how production LLM applications handle complex workflows: a customer support system with a triage agent, a billing specialist, and a technical support specialist; a content pipeline with research, writing, and editing agents; a coding assistant with a planner, implementer, and reviewer.

This module builds directly on the agent fundamentals from Module 14. If you can build one agent, you can build a system of agents that work together. The challenge is coordination — and that is what this module teaches.

---

## Connection to Other Modules

- **Module 14 (Agent Fundamentals)** provides the single-agent patterns that this module composes into multi-agent systems.
- **Module 16 (Workflows & Chains)** offers an alternative to multi-agent systems for deterministic pipelines.
- **Module 17 (Code Generation)** can use multi-agent patterns for plan-implement-review cycles.
- **Module 18 (Human-in-the-Loop)** adds human oversight to multi-agent coordination.

---

## Section 1: Why Multiple Agents?

### The Limits of a Single Agent

A single agent with many tools and a broad system prompt faces several problems:

1. **Prompt dilution** — a system prompt that covers research, writing, review, and formatting gives weak guidance on each
2. **Tool overload** — too many tools make tool selection harder and increase the chance of wrong picks
3. **Context pollution** — intermediate reasoning for one sub-task pollutes the context for another
4. **No separation of concerns** — everything happens in one conversation, making debugging harder

### The Multi-Agent Solution

Split a complex task into roles, give each role to a dedicated agent. Each agent gets its own `generateText` call with a focused system prompt and a curated set of tools.

The contrast: a single do-everything agent with 10+ tools and a sprawling system prompt, versus three focused agents -- a researcher (with search tools, prompt focused on gathering facts), a writer (no tools, prompt focused on readability), and a reviewer (no tools, prompt focused on finding issues).

Each agent is an async function that takes an input string and returns a string. The pipeline runs them sequentially -- the researcher's output feeds the writer, and the writer's output feeds the reviewer.

```typescript
import { generateText, stepCountIs } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
```

Your task: build three agent functions and a pipeline that connects them.

- `researchAgent(topic: string): Promise<string>` -- uses `generateText` with `stopWhen: stepCountIs(5)`, a search tool, and a system prompt that limits the agent to research only
- `writerAgent(researchBrief: string): Promise<string>` -- uses `generateText` with no tools, system prompt focused on writing from a provided brief
- `reviewerAgent(article: string): Promise<string>` -- uses `generateText` with no tools, system prompt focused on reviewing for accuracy, clarity, structure, and style

Think about:

- What makes a good system prompt boundary? How do you tell an agent what NOT to do?
- Why does the researcher need `stopWhen` but the writer and reviewer do not?
- What happens if you pass the research brief as part of the writer's `prompt` rather than the `system`?

> **Beginner Note:** Think of multi-agent systems like a team of people. You would not ask one person to simultaneously research, write, edit, and fact-check. You assign roles because specialists do better work than generalists trying to do everything at once.

> **Advanced Note:** Multi-agent systems are not always better. They add latency (more LLM calls), complexity (coordination logic), and cost (more tokens). Use them when the task genuinely benefits from separation of concerns. For simple tasks, a single well-prompted agent is faster, cheaper, and easier to debug.

---

## Section 2: Orchestrator-Worker Pattern

### The Core Pattern

The orchestrator-worker pattern has one "orchestrator" agent that understands the overall task and delegates sub-tasks to specialized "worker" agents. The orchestrator:

1. Analyzes the task
2. Breaks it into sub-tasks
3. Assigns each sub-task to the right worker
4. Collects and synthesizes results

You need worker agents and an orchestrator function that coordinates them. Start with the types:

```typescript
import { generateText, Output, stepCountIs } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface SubTask {
  id: number
  type: 'research' | 'analysis' | 'writing'
  description: string
  input: string
  dependsOn: number[]
}
```

Build three worker functions:

- `researchWorker(query: string): Promise<string>` -- uses `stopWhen: stepCountIs(3)` and a search tool
- `analysisWorker(data: string, question: string): Promise<string>` -- no tools, system prompt focused on data analysis
- `writingWorker(content: string, format: string): Promise<string>` -- no tools, system prompt focused on transforming content

Then build the orchestrator: `async function orchestrator(task: string): Promise<string>`

The orchestrator has three phases:

1. **Plan** -- use `Output.object` with a schema of subtasks (each with id, type, description, input, dependsOn) to have the LLM decompose the task
2. **Execute** -- iterate through subtasks in dependency order. For each subtask, check that dependencies are complete, build the input (prepending dependency results if any), then dispatch to the right worker via a switch on `subtask.type`
3. **Synthesize** -- collect all results and use a final `generateText` call to combine them into a coherent answer

Think about:

- What happens if a subtask's dependencies are not ready? Should you skip it or throw?
- How do you pass dependency results as context to a subsequent subtask?
- What if the LLM's plan has circular dependencies?

> **Beginner Note:** The orchestrator is like a project manager -- it does not do the actual work, but it knows who should do what and in what order. Worker agents are the specialists who execute specific tasks.

### Dynamic Worker Selection

Instead of hardcoding worker types, let the orchestrator discover available workers at runtime through a registry pattern.

```typescript
interface WorkerDefinition {
  name: string
  description: string
  capabilities: string[]
  execute: (input: string) => Promise<string>
}
```

Build an `AgentRegistry` class with these methods:

- `register(worker: WorkerDefinition): void` -- stores workers in a Map by name
- `getWorker(name: string): WorkerDefinition | undefined`
- `listWorkers(): string` -- returns a formatted string describing all workers and their capabilities, suitable for inclusion in a system prompt
- `dispatch(workerName: string, input: string): Promise<string>` -- looks up and executes a worker, throwing if not found

Then build a `dynamicOrchestrator(task: string, registry: AgentRegistry): Promise<string>` that uses a `delegate` tool:

```typescript
tools: {
  delegate: {
    description: 'Delegate a sub-task to a specific worker agent',
    parameters: z.object({
      worker: z.string().describe('Name of the worker to delegate to'),
      task: z.string().describe('The sub-task description'),
    }),
    execute: async ({ worker, task }) => {
      // dispatch to registry, return JSON with worker, success, and result (or error)
    },
  },
},
```

The orchestrator's system prompt should include `registry.listWorkers()` so the LLM knows what workers are available.

How would you handle the case where the LLM requests a worker that does not exist? Should the delegate tool throw, or return an error message the LLM can react to?

---

## Section 3: Agent Communication

### Passing Context Between Agents

Agents communicate by passing data -- the output of one agent becomes the input of another. The key design decision is what to pass and in what format.

Define a structured handoff schema using Zod:

```typescript
const handoffSchema = z.object({
  summary: z.string().describe('Brief summary of what was accomplished'),
  findings: z
    .array(
      z.object({
        topic: z.string(),
        content: z.string(),
        confidence: z.enum(['high', 'medium', 'low']),
      })
    )
    .describe('Key findings from the research'),
  openQuestions: z.array(z.string()).describe('Questions that still need answers'),
  recommendations: z.array(z.string()).describe('Suggested next steps'),
})

type Handoff = z.infer<typeof handoffSchema>
```

Build two functions:

- `researchWithHandoff(topic: string): Promise<Handoff>` -- runs a research agent with search tools, then uses a second `generateText` call with `Output.object({ schema: handoffSchema })` to structure the raw research into the handoff format
- `writeFromHandoff(handoff: Handoff, format: string): Promise<string>` -- takes the structured handoff and produces polished content. The system prompt should instruct the writer to use high-confidence findings directly, include medium-confidence findings with caveats, and skip low-confidence findings

Think about:

- Why use two LLM calls in `researchWithHandoff` (one for research, one for structuring) rather than one?
- How does the confidence field help the downstream writer make better decisions?
- What would happen if you passed the raw research text instead of the structured handoff?

### Message-Passing Protocol

For more complex systems, define a formal message protocol:

```typescript
interface AgentMessage {
  from: string
  to: string
  type: 'request' | 'response' | 'error' | 'status'
  correlationId: string
  payload: unknown
  timestamp: number
}
```

Build a `MessageBus` class with:

- `register(agentName: string, handler: (msg: AgentMessage) => Promise<void>): void` -- registers a handler for messages sent to a given agent
- `send(message: AgentMessage): Promise<void>` -- logs the message, looks up the handler for `message.to`, and invokes it. Throws if no handler is registered
- `getLog(): AgentMessage[]` -- returns a copy of all messages sent

The `correlationId` ties requests to responses. When a handler receives a request, it should send a response back with the same `correlationId`.

> **Advanced Note:** Message buses add infrastructure complexity. Use direct function calls (agent A calls agent B directly) for simple systems. Introduce a message bus only when you need features like logging, replay, or decoupled agents.

---

## Section 4: Shared State

### What to Share, What Stays Private

In a multi-agent system, some information should be shared across agents and some should stay private:

| Shared State                | Private State                |
| --------------------------- | ---------------------------- |
| Task description and goals  | Internal reasoning traces    |
| Key findings and facts      | Intermediate drafts          |
| Current progress and status | Tool-specific parameters     |
| Error reports and blockers  | Working scratchpad           |
| Final outputs               | Model-specific prompt tricks |

Start with the shared state interface and factory:

```typescript
interface SharedState {
  task: string
  findings: Map<string, string>
  status: Map<string, 'pending' | 'in_progress' | 'completed' | 'failed'>
  errors: string[]
  finalOutputs: Map<string, string>
}

function createSharedState(task: string): SharedState {
  /* ... */
}
```

Build an `AgentContext` class that mediates access between an agent and the shared/private state:

```typescript
class AgentContext {
  constructor(
    private agentName: string,
    private shared: SharedState,
    private privateState: Map<string, unknown> = new Map()
  ) {}

  // Shared state methods
  addFinding(key: string, value: string): void
  getAllFindings(): Map<string, string>
  setStatus(status: 'pending' | 'in_progress' | 'completed' | 'failed'): void
  reportError(error: string): void
  setOutput(key: string, value: string): void
  getTask(): string

  // Private state methods
  setPrivate(key: string, value: unknown): void
  getPrivate<T>(key: string): T | undefined
}
```

Key implementation details:

- `addFinding` should namespace the key with the agent name (e.g., `${this.agentName}:${key}`) to avoid collisions
- `getAllFindings` should return a copy of the findings map, not a reference
- `reportError` should prefix the error with the agent name in brackets

Then build `researchAgentWithState(ctx: AgentContext): Promise<void>` -- an agent that sets its status to `in_progress`, runs a search-based research loop, stores search queries in private state, adds findings and output to shared state, and sets status to `completed` or `failed`.

Think about:

- Why return a copy from `getAllFindings` instead of the original map?
- What happens if two agents call `addFinding` with the same key but different agent names?
- When should an agent read shared findings from other agents versus receiving them as direct input?

> **Beginner Note:** Shared state is like a shared whiteboard that all team members can write on and read from. Private state is like each team member's personal notebook. Keep the shared whiteboard focused on what everyone needs to know.

---

## Section 5: Delegation Strategies

### By Capability

Route sub-tasks to agents based on what they can do. Define agent capabilities as structured data:

```typescript
interface AgentCapability {
  name: string
  skills: string[]
  model: string
  execute: (task: string) => Promise<string>
}
```

Create an array of agents with different skill profiles (e.g., a data-analyst with statistics/SQL skills, a code-writer with TypeScript/debugging skills, a content-creator with writing/editing skills). Each agent's `execute` function wraps a `generateText` call with a role-specific system prompt.

Build `routeByCapability(task: string): Promise<string>`:

1. Use `Output.object` to have the LLM analyze the task against the available agents' skills and select the best match (output schema: `selectedAgent`, `reasoning`, `requiredSkills`)
2. Format the agent list into the prompt so the LLM can see each agent's name and skills
3. Look up the selected agent by name and call its `execute` function
4. Throw if the LLM selects an agent that does not exist

What happens if the task requires skills from multiple agents? How would you modify this pattern to support multi-agent delegation?

### By Topic

Route based on the domain or topic of the request. Build a `TopicRouter`:

```typescript
interface TopicRouter {
  topics: Map<string, (input: string) => Promise<string>>
}
```

Create a `createTopicRouter()` factory that registers handlers for topics like `'billing'`, `'technical'`, and `'general'`. Each handler wraps a `generateText` call with a domain-specific system prompt and tools (e.g., the technical handler gets a `lookupDocs` tool).

Build `routeByTopic(router: TopicRouter, userMessage: string): Promise<{ topic: string; response: string }>`:

1. Use `Output.object` to classify the message into a topic with confidence and reasoning
2. Look up the handler from the router's topics map
3. Execute the handler and return the result

Think about: what should happen when the classifier's confidence is below a threshold? Should you route to a general handler, ask for clarification, or try multiple handlers?

---

## Section 6: Parallel Agent Execution

### Running Independent Agents Concurrently

When sub-tasks are independent, run them in parallel to reduce total latency. Define the types:

```typescript
interface ParallelTask {
  name: string
  execute: () => Promise<string>
}

interface ParallelResult {
  name: string
  result: string
  durationMs: number
  success: boolean
  error?: string
}
```

Build `runParallel(tasks: ParallelTask[], maxConcurrency: number = 5): Promise<ParallelResult[]>`:

- Split tasks into chunks of size `maxConcurrency`
- Process each chunk with `Promise.allSettled` (not `Promise.all` -- why?)
- For each task, record the name, result, duration, and success/failure status
- Log the total wall-clock time versus the sum of individual durations to show the parallel speedup

The key pattern: chunk the tasks array, iterate through chunks sequentially, but within each chunk use `Promise.allSettled` for concurrent execution. This creates a sliding window of concurrent tasks.

Think about:

- Why use `Promise.allSettled` instead of `Promise.all`?
- What happens to timing if one task in a chunk takes 10x longer than the others?
- How would you implement a true sliding window (start next task as each completes) instead of chunk-based batching?

### Map-Reduce with Agents

A common parallel pattern: map a task across multiple agents, then reduce the results.

Build `mapReduceAgents(items, mapPrompt, reducePrompt, concurrency): Promise<string>`:

- **Map phase**: process each item in parallel (respecting concurrency) using the `mapPrompt` function to generate the prompt for each
- **Reduce phase**: feed all map results into a single `generateText` call using the `reducePrompt` function

```typescript
async function mapReduceAgents(
  items: string[],
  mapPrompt: (item: string) => string,
  reducePrompt: (results: string[]) => string,
  concurrency: number = 3
): Promise<string>
```

The map phase chunks items and uses `Promise.all` within each chunk. The reduce phase takes all results and synthesizes them. This pattern works for any task that can be decomposed into independent parts and then combined: analyzing multiple companies, reviewing multiple documents, processing multiple data sources.

> **Advanced Note:** Be mindful of API rate limits when running agents in parallel. Most providers have tokens-per-minute and requests-per-minute limits. Set `maxConcurrency` based on your provider's rate limits. Consider adding exponential backoff for 429 (rate limit) errors.

---

## Section 7: Agent Handoff

### Transferring Conversations Between Agents

Sometimes an agent needs to transfer a conversation to a different specialist. This is common in customer support: a general agent detects a billing question and hands off to the billing specialist.

Define the types for the handoff system:

```typescript
import { type ModelMessage } from 'ai'

interface HandoffRequest {
  targetAgent: string
  reason: string
  conversationSummary: string
  userIntent: string
  fullHistory: ModelMessage[]
}

interface AgentSpec {
  name: string
  system: string
  tools: Record<string, any>
  canHandoff: string[] // which agents this one can hand off to
}
```

Build a `ConversationRouter` class with:

- `register(spec: AgentSpec): void` -- stores agent specs by name
- `runWithHandoff(startAgent, userMessage, maxHandoffs = 3): Promise<{ finalAgent: string; response: string; handoffs: string[] }>`

The `runWithHandoff` method is the core logic. It loops up to `maxHandoffs` times:

1. Look up the current agent's spec
2. If the agent can hand off, inject a `handoff` tool into its tools. The tool's parameters are `targetAgent` (an enum of the agent's `canHandoff` list), `reason`, and `summary`
3. Call `generateText` with the agent's system prompt, current messages, and tools
4. Check if any step's tool calls include a `handoff` call. If so, extract the target agent, update the messages array to include the handoff context, switch to the new agent, and continue the loop
5. If no handoff occurred, return the result

Key design decisions:

- The `handoff` tool should NOT have an `execute` function -- it is a signal to the router, not an action the agent performs
- When constructing messages for the receiving agent, include a summary from the handing-off agent plus the original user message
- The `canHandoff` list restricts which agents each specialist can transfer to

Think about:

- Why limit `maxHandoffs`? What happens without a limit?
- Should the receiving agent see the full conversation history or just a summary?
- How do you prevent ping-pong handoffs (A hands to B, B hands back to A)?

> **Beginner Note:** Agent handoff is like being transferred to a different department when you call customer support. The key is passing enough context so you do not have to repeat yourself -- the receiving agent should know what you already discussed.

---

## Section 8: Error Handling

### Sub-Agent Failures

In a multi-agent system, individual agents can fail. The system must handle these failures gracefully. Define the result type:

```typescript
interface AgentResult {
  agentName: string
  success: boolean
  result?: string
  error?: string
  retries: number
}
```

Build `runWithRetry(agentName, execute, maxRetries = 2, fallback?): Promise<AgentResult>`:

- Loop from 0 to `maxRetries`, calling `execute()` each time
- On failure, log the error and wait with exponential backoff (`Math.pow(2, attempt) * 1000` ms)
- After all retries are exhausted, try the optional `fallback` function if provided
- Return an `AgentResult` with the agent name, success status, result or error, and retry count

Then build `resilientOrchestrator(task: string): Promise<string>`:

- Define an array of subtask objects, each with a name, execute function, and optional fallback
- Run each subtask through `runWithRetry`
- If ALL agents fail, return an error message
- If some agents fail, log warnings but continue with available results
- Use a final `generateText` call to synthesize successful results, noting which agents failed

Think about:

- Should the orchestrator run subtasks sequentially or in parallel? What are the trade-offs?
- When is a fallback better than more retries?
- How would you decide which errors are retryable (transient network issues) versus permanent (bad input)?

### Circuit Breaker Pattern

Prevent cascading failures by stopping calls to a consistently failing agent.

Build a `CircuitBreaker` class with three states: `closed` (normal), `open` (rejecting calls), and `half-open` (testing recovery):

```typescript
class CircuitBreaker {
  private failures = 0
  private lastFailureTime = 0
  private state: 'closed' | 'open' | 'half-open' = 'closed'

  constructor(
    private threshold: number = 3,
    private resetTimeMs: number = 30000
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T>
  getState(): string
}
```

The `execute` method:

- If `open` and enough time has passed since the last failure, transition to `half-open` and try the call
- If `open` and not enough time has passed, throw immediately without calling `fn`
- On success, reset failures to 0 and set state to `closed`
- On failure, increment failures. If failures reach the threshold, set state to `open`

How does this pattern save tokens and reduce latency when a downstream service is down?

> **Advanced Note:** In production multi-agent systems, combine retries, circuit breakers, and fallbacks. The circuit breaker prevents wasting tokens on a consistently failing agent, retries handle transient errors, and fallbacks provide degraded but functional responses when an agent is down.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Agent Pool Coordinator

### Managing Worker Pools

The orchestrator-worker pattern from Section 2 dispatches agents one at a time. A coordinator goes further — it manages a **pool** of worker agents that execute in parallel, with concurrency limits, load balancing, and result aggregation.

The coordinator's job:

1. Receive a complex task
2. Decompose it into independent sub-tasks
3. Dispatch sub-tasks to available workers (up to a concurrency limit)
4. Collect results as workers complete
5. Handle worker failures with retry or fallback
6. Aggregate all results into a final output

```typescript
interface CoordinatorConfig {
  maxConcurrency: number
  workerTimeout: number
  retryLimit: number
}

interface SubTask {
  id: string
  description: string
  toolSubset: string[]
  priority: number
}
```

The concurrency limit is critical. Running too many agents in parallel hits API rate limits. A coordinator with `maxConcurrency: 3` dispatches three workers initially, then dispatches the next worker as each one completes — a sliding window pattern.

Result aggregation depends on the task. For research tasks, concatenate findings. For review tasks, merge feedback. For classification tasks, use majority voting. The coordinator's aggregation logic is what turns individual worker outputs into a coherent result.

> **Beginner Note:** Start with sequential dispatch (concurrency 1) to get the pattern working, then increase concurrency once error handling is solid.

---

## Section 10: Agent Type Specialization

### Defining Agent Roles

Production systems define specialized agent types, each with a focused system prompt, a curated tool set, and behavioral constraints. Common specializations:

| Agent Type          | Mode  | Tools              | Purpose                       |
| ------------------- | ----- | ------------------ | ----------------------------- |
| **General-purpose** | Build | All tools          | Full implementation tasks     |
| **Explorer**        | Plan  | Read, search, glob | Fast codebase navigation      |
| **Planner**         | Plan  | Read, search       | Architecture and design       |
| **Reviewer**        | Plan  | Read, search, glob | Code review, no modifications |

The specialization is structural, not just prompt-based. An explorer agent literally cannot write files — it does not have the write tool. This prevents accidental modifications during read-only tasks and makes the agent's capabilities explicit.

```typescript
interface AgentType {
  name: string
  systemPrompt: string
  tools: Record<string, Tool>
  maxSteps: number
  model: string
}

function createAgent(type: AgentType, task: string) {
  /* ... */
}
```

When building a multi-agent system, define your agent types first. The types become the building blocks that the coordinator dispatches. A well-defined set of agent types makes the system predictable — you know exactly what each agent can and cannot do.

---

## Section 11: Workspace Isolation

### Preventing Conflicts in Parallel Work

When multiple agents work on related tasks simultaneously, they can conflict — two agents editing the same file, or one agent's changes breaking another's assumptions. Workspace isolation prevents this by giving each agent its own working directory.

The simplest approach: create a temporary directory for each agent, copy the relevant files, let the agent work in isolation, then merge results back.

```typescript
import { mkdtemp, cp } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'

async function createAgentWorkspace(sourceDir: string): Promise<string> {
  /* ... */
}
```

Each agent receives its workspace path and operates only within that directory. When the agent completes, the coordinator collects the results and reconciles any conflicts. For file-based tasks, this means diffing the agent's workspace against the original to see what changed.

The trade-off is overhead: copying files takes time, and merging results adds complexity. Use workspace isolation when agents might conflict (parallel code changes, concurrent file processing). Skip it when agents work on clearly independent resources (different APIs, different data sources).

> **Advanced Note:** In production, git worktrees provide a more robust isolation mechanism than temporary directories. Each agent works in a separate worktree of the same repository — they share the git history but have independent working trees. Changes merge through standard git operations with conflict detection built in.

---

## Section 12: Primary and Subagent Architecture

### Agent Lifecycle and Scope

Production multi-agent systems distinguish between two categories of agents:

- **Primary agents** — persistent, user-facing. They own the conversation, have full (or near-full) tool access, and run across the entire session. The user interacts with a primary agent directly
- **Subagents** — task-scoped, invoked on demand. A primary agent (or the user) spawns a subagent for a specific task. The subagent runs its own conversation with its own tools, returns results to the caller, and terminates

The distinction is about scope and lifecycle, not capability. A subagent can be as powerful as a primary agent — the difference is that it is invoked for a specific purpose and its results feed back into the parent context.

```typescript
async function invokeSubagent(agentType: AgentType, task: string, parentContext?: string): Promise<string> {
  /* ... */
}
```

**Context isolation** is critical. Subagents start with fresh context — they do not inherit the parent's full conversation history. The parent passes only the relevant context (a summary or the specific task description). This prevents context pollution: the subagent's token budget is spent on its task, not on irrelevant parent history.

> **Beginner Note:** Think of subagents like function calls in regular programming. The parent agent "calls" a subagent with arguments (the task and context), waits for the return value (the result), and continues. The subagent's internal state is not visible to the parent.

---

## Section 13: Agent Configuration via Markdown

### Declarative Agent Definitions

Instead of hardcoding agent types in source code, define them declaratively in markdown files with YAML frontmatter. The filename becomes the agent ID, the frontmatter specifies configuration, and the body becomes the system prompt.

```markdown
---
description: 'Code review specialist'
mode: plan
model: default
tools: [read, grep, glob]
temperature: 0.2
maxSteps: 50
---

You are a code reviewer. Analyze code for bugs, style issues, and performance problems.
Focus on correctness first, then readability, then performance.
```

An agent loader reads these files, parses the YAML frontmatter, and instantiates agents with the specified configuration. This is configuration-as-code for agents — definitions can be versioned in git, shared across projects, and customized per-project.

```typescript
interface AgentConfig {
  description: string
  mode: 'plan' | 'build'
  model: string
  tools: string[]
  temperature?: number
  maxSteps?: number
}

// Load agent from markdown file → AgentConfig + systemPrompt
```

The advantages are composability and transparency. A new agent type is just a new markdown file. Non-developers can understand and modify agent behavior by editing the system prompt. The configuration is visible and auditable, not buried in application code.

---

## Section 14: @Mention Invocation

### Explicit Agent Targeting

Production systems let users (or parent agents) invoke a specific subagent with `@agent_name` syntax. The mentioned agent runs with its own context, tools, and system prompt, and results return inline to the parent conversation.

```typescript
function parseMention(message: string): { agentName: string; task: string } | null {
  /* ... */
}
```

The routing flow:

1. User (or parent agent) sends a message like `@reviewer check this function for edge cases`
2. The system parses the @mention and looks up the `reviewer` agent configuration
3. The reviewer agent runs with its own tools and system prompt
4. Results return to the caller

This is delegation with explicit targeting. Instead of the orchestrator deciding which agent to use, the caller names the specialist directly. Both patterns are useful — automatic routing for end users who do not know the available agents, explicit mentions for power users and parent agents that know exactly what they need.

> **Advanced Note:** @mention invocation composes naturally with agent configuration via markdown. The agent name in the @mention maps to the markdown filename. Adding a new specialist is: create a markdown file, restart, and `@new_agent` is available.

---

## Quiz

### Question 1 (Easy)

What is the main advantage of splitting a complex task across multiple agents instead of using one agent?

- A) Multiple agents use fewer tokens overall
- B) Each agent has a focused prompt and toolset, leading to better performance on its specific sub-task
- C) Multiple agents are always faster than a single agent
- D) Multiple agents do not need error handling

**Answer: B** — Each agent gets a focused system prompt and a curated set of tools for its specific role. This avoids prompt dilution (one prompt trying to cover everything) and tool overload (too many tools confusing the model). Multi-agent systems often use more tokens (A is wrong), are not always faster (C is wrong), and definitely need error handling (D is wrong).

---

### Question 2 (Medium)

In the orchestrator-worker pattern, what is the orchestrator's primary responsibility?

- A) Executing all the actual work itself
- B) Breaking the task into sub-tasks, delegating to workers, and synthesizing results
- C) Providing tools to the worker agents
- D) Storing the conversation history for all agents

**Answer: B** — The orchestrator is like a project manager: it analyzes the task, determines which workers should handle which parts, dispatches the work, collects results, and synthesizes them into a final answer. The workers execute the actual work (A). Tools belong to the workers (C). History management can be shared or handled separately (D).

---

### Question 3 (Medium)

When running multiple agents in parallel, what is the primary constraint you must consider?

- A) Parallel agents cannot share any state
- B) API rate limits from the model provider
- C) Parallel agents must all use the same model
- D) Results from parallel agents cannot be combined

**Answer: B** — Most API providers have tokens-per-minute and requests-per-minute limits. Running too many agents in parallel can trigger rate limiting (HTTP 429 errors). Parallel agents can share state through shared data structures (A is wrong), can use different models (C is wrong), and their results are typically combined by the orchestrator (D is wrong).

---

### Question 4 (Hard)

An agent handoff system transfers a user from a triage agent to a billing agent, but the billing agent asks the user to repeat their problem. What is the most likely cause?

- A) The billing agent has a different model than the triage agent
- B) The handoff did not include a conversation summary, so the billing agent has no context
- C) The billing agent's system prompt is too short
- D) The triage agent should not have tools

**Answer: B** — When transferring between agents, the receiving agent needs context about what was already discussed. Without a conversation summary or the original message, the billing agent starts fresh and has to ask the user to explain again. The fix is to pass a summary of the conversation and the user's original request as context to the receiving agent.

---

### Question 5 (Hard)

In a multi-agent system, why is a circuit breaker pattern useful?

- A) It limits the total number of agents that can run simultaneously
- B) It prevents repeatedly calling a consistently failing agent, saving tokens and reducing latency
- C) It ensures all agents complete before the orchestrator synthesizes results
- D) It encrypts communication between agents

**Answer: B** — A circuit breaker tracks consecutive failures for an agent. After a threshold is reached, it "opens" and immediately rejects further calls to that agent for a cooldown period. This prevents wasting tokens and time on an agent that is consistently failing (e.g., due to a downstream API outage). After the cooldown, it tries again ("half-open") to see if the issue resolved.

---

### Question 6 (Medium)

Why should subagents start with fresh context rather than inheriting the parent agent's full conversation history?

a) Subagents cannot process conversation history
b) The parent's conversation history consumes tokens that the subagent should spend on its specific task — passing only relevant context (a summary or task description) prevents context pollution and keeps the subagent focused
c) Fresh context makes subagents run faster
d) Conversation history is not serializable

**Answer: B**

**Explanation:** A parent agent with 50 steps of conversation history might have 80,000 tokens of context. If a subagent inherits all of that, most of its token budget is consumed by irrelevant history. Instead, the parent passes only the specific task and any relevant context (a few hundred tokens). This lets the subagent spend its full budget on its task, just like a function call in regular programming passes arguments rather than the entire program state.

---

### Question 7 (Hard)

Your multi-agent system defines agent types in markdown files with YAML frontmatter. A reviewer agent is configured with `mode: plan` and `tools: [read, grep, glob]`. A developer accidentally changes the config to `tools: [read, grep, glob, write]` without changing the mode. What architectural principle does this violate, and why is it dangerous?

a) It violates the DRY principle because tools are duplicated
b) It violates the principle that behavioral constraints should be structural — a plan-mode agent with write tools can modify files despite being designated as read-only, because the tool set is the actual constraint and the mode label is just metadata
c) It violates the single responsibility principle
d) It violates the principle of least privilege only if the agent actually writes files

**Answer: B**

**Explanation:** In a well-designed system, the `mode: plan` label should determine the tool set (plan = read-only tools). When the tool set is specified independently, the mode becomes a misleading label — the agent is called "plan" but has write capabilities. The system should either derive tools from mode or validate that the tool set matches the declared mode. This is a real risk in declarative configurations: labels and behavior can drift apart, and the behavior (available tools) is what actually matters.

---

## Exercises

### Exercise 1: Orchestrator with Research, Writing, and Review Agents

**Objective:** Build a multi-agent system where an orchestrator delegates to three specialized agents: a researcher, a writer, and a reviewer. The system produces a polished article on a given topic.

**Specification:**

1. Create a file `src/exercises/m15/ex01-multi-agent-pipeline.ts`
2. Export an async function `multiAgentArticle(topic: string, options?: PipelineOptions): Promise<PipelineResult>`
3. Define the types:

```typescript
interface PipelineOptions {
  maxResearchSteps?: number // default: 5
  reviewRounds?: number // default: 1
  verbose?: boolean // default: false
}

interface AgentOutput {
  agentName: string
  output: string
  durationMs: number
  success: boolean
  error?: string
}

interface PipelineResult {
  finalArticle: string
  agentOutputs: AgentOutput[]
  totalDurationMs: number
  reviewFeedback: string[]
}
```

4. Implement three worker agents:
   - **Research Agent**: Uses search tools to gather information, returns a structured research brief
   - **Writing Agent**: Takes the research brief and produces a well-structured article
   - **Review Agent**: Reviews the article, provides feedback, and scores quality (1-10)

5. The orchestrator must:
   - Run the research agent first
   - Pass research results to the writing agent
   - Pass the draft to the review agent
   - If `reviewRounds > 1`, send the review feedback back to the writer for revision
   - Handle agent failures with retries (max 2 retries per agent)

6. If any agent fails after retries, the orchestrator should continue with available results and note the failure

**Example usage:**

```typescript
const result = await multiAgentArticle('The future of quantum computing', {
  maxResearchSteps: 5,
  reviewRounds: 2,
  verbose: true,
})

console.log(result.finalArticle)
console.log(`Agents used: ${result.agentOutputs.map(a => a.agentName).join(', ')}`)
console.log(`Review feedback: ${result.reviewFeedback.join('; ')}`)
```

**Test specification:**

```typescript
// tests/exercises/m15/ex01-multi-agent-pipeline.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 15: Multi-Agent Article Pipeline', () => {
  it('should produce a final article', async () => {
    const result = await multiAgentArticle('TypeScript best practices')
    expect(result.finalArticle).toBeTruthy()
    expect(result.finalArticle.length).toBeGreaterThan(200)
  })

  it('should record outputs from all three agents', async () => {
    const result = await multiAgentArticle('Cloud computing trends')
    const agentNames = result.agentOutputs.map(a => a.agentName)
    expect(agentNames).toContain('researcher')
    expect(agentNames).toContain('writer')
    expect(agentNames).toContain('reviewer')
  })

  it('should include review feedback', async () => {
    const result = await multiAgentArticle('Machine learning basics', {
      reviewRounds: 1,
    })
    expect(result.reviewFeedback.length).toBeGreaterThan(0)
  })

  it('should handle multiple review rounds', async () => {
    const result = await multiAgentArticle('Web development', {
      reviewRounds: 2,
    })
    expect(result.reviewFeedback.length).toBe(2)
  })

  it('should complete even if one agent fails', async () => {
    // With a mock that makes the review agent fail
    const result = await multiAgentArticle('Testing error handling')
    expect(result.finalArticle).toBeTruthy()
  })
})
```

---

### Exercise 2: Customer Support Routing System

**Objective:** Build a customer support system with agent handoff. A triage agent classifies incoming messages and routes them to the appropriate specialist.

**Specification:**

1. Create a file `src/exercises/m15/ex02-support-router.ts`
2. Export an async function `handleSupportRequest(message: string): Promise<SupportResult>`
3. Define the types:

```typescript
interface SupportResult {
  finalAgent: string
  response: string
  handoffChain: string[] // e.g., ["triage", "billing"]
  classification: string
  totalDurationMs: number
}
```

4. Implement at least three agents:
   - **Triage Agent**: Classifies the request and routes to the right specialist
   - **Billing Agent**: Handles billing, payment, and subscription questions
   - **Technical Agent**: Handles API issues, bugs, and configuration problems

5. Each specialist should have at least one tool relevant to their domain

6. The system must support handoffs between agents (a specialist can re-route if they realize the issue belongs to another specialist)

**Test specification:**

```typescript
// tests/exercises/m15/ex02-support-router.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 15: Support Router', () => {
  it('should route billing questions to the billing agent', async () => {
    const result = await handleSupportRequest('I was double-charged on my last invoice')
    expect(result.finalAgent).toBe('billing')
  })

  it('should route technical questions to the technical agent', async () => {
    const result = await handleSupportRequest('My API calls are returning 500 errors')
    expect(result.finalAgent).toBe('technical')
  })

  it('should track the handoff chain', async () => {
    const result = await handleSupportRequest('I need help with my account')
    expect(result.handoffChain.length).toBeGreaterThan(0)
    expect(result.handoffChain[0]).toBe('triage')
  })

  it('should always produce a response', async () => {
    const result = await handleSupportRequest('Hello, can you help me?')
    expect(result.response).toBeTruthy()
  })
})
```

---

### Exercise 3: Agent Pool Coordinator

**Objective:** Build a coordinator that decomposes tasks into sub-tasks, dispatches worker agents with appropriate tool subsets, and aggregates results — with concurrency control and failure handling.

**Specification:**

1. Create a file `src/exercises/m15/ex03-coordinator.ts`
2. Export an async function `coordinateTask(task: string, options?: CoordinatorOptions): Promise<CoordinatorResult>`
3. Define the types:

```typescript
interface CoordinatorOptions {
  maxConcurrency?: number // default: 2
  workerTimeout?: number // default: 30000
  retryLimit?: number // default: 1
  verbose?: boolean // default: false
}

interface SubTask {
  id: string
  description: string
  agentType: string // maps to a registered agent type
  status: 'pending' | 'running' | 'completed' | 'failed'
}

interface WorkerResult {
  subTaskId: string
  agentType: string
  output: string
  success: boolean
  error?: string
  durationMs: number
}

interface CoordinatorResult {
  finalOutput: string
  subTasks: SubTask[]
  workerResults: WorkerResult[]
  totalDurationMs: number
  failedTasks: number
}
```

4. Implement the coordinator:
   - Use an LLM call to decompose the input task into 2-4 independent sub-tasks
   - Dispatch sub-tasks to worker agents, respecting the concurrency limit (sliding window: start new workers as others complete)
   - Each worker gets its own tool subset and system prompt based on its agent type
   - Collect results, retry failed tasks up to `retryLimit`
   - Use a final LLM call to aggregate worker results into a coherent output

5. Register at least two worker agent types (e.g., researcher and analyzer) with different tool sets

**Test specification:**

```typescript
// tests/exercises/m15/ex03-coordinator.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 15: Agent Pool Coordinator', () => {
  it('should decompose task into sub-tasks', async () => {
    const result = await coordinateTask('Compare TypeScript and Rust for backend development')
    expect(result.subTasks.length).toBeGreaterThanOrEqual(2)
  })

  it('should respect concurrency limit', async () => {
    const result = await coordinateTask('Research three programming languages', {
      maxConcurrency: 1,
    })
    expect(result.workerResults.length).toBeGreaterThan(0)
  })

  it('should aggregate results into final output', async () => {
    const result = await coordinateTask('Summarize pros and cons of microservices')
    expect(result.finalOutput).toBeTruthy()
    expect(result.finalOutput.length).toBeGreaterThan(100)
  })

  it('should handle worker failures gracefully', async () => {
    const result = await coordinateTask('Research an obscure topic with limited data')
    expect(result.finalOutput).toBeTruthy()
  })
})
```

---

### Exercise 4: Specialized Agent Types

**Objective:** Create a set of specialized agent types with different tool sets, system prompts, and behavioral constraints. Demonstrate that the same task produces different results when handled by different agent types.

**Specification:**

1. Create a file `src/exercises/m15/ex04-specialized-agents.ts`
2. Export a function `createAgentType(config: AgentTypeConfig): AgentType` and a function `runSpecializedAgent(agentType: AgentType, task: string): Promise<AgentOutput>`
3. Define the types:

```typescript
interface AgentTypeConfig {
  name: string
  mode: 'plan' | 'build'
  systemPrompt: string
  toolNames: string[] // References to registered tools
  maxSteps: number
}

interface AgentType {
  name: string
  mode: 'plan' | 'build'
  systemPrompt: string
  tools: Record<string, Tool>
  maxSteps: number
}

interface AgentOutput {
  agentName: string
  mode: string
  response: string
  toolsUsed: string[]
  stepsCompleted: number
  durationMs: number
}
```

4. Create three specialized agents:
   - **Researcher** — plan mode, has search and read tools, system prompt focuses on gathering facts
   - **Implementer** — build mode, has write and edit tools, system prompt focuses on building code
   - **Reviewer** — plan mode, has read and search tools, system prompt focuses on finding issues and suggesting improvements

5. Run the same task through each agent and compare the outputs

**Test specification:**

```typescript
// tests/exercises/m15/ex04-specialized-agents.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 15: Specialized Agents', () => {
  it('should create agents with different tool sets', () => {
    const researcher = createAgentType({
      name: 'researcher',
      mode: 'plan',
      systemPrompt: 'You research topics.',
      toolNames: ['search', 'read'],
      maxSteps: 10,
    })
    const implementer = createAgentType({
      name: 'implementer',
      mode: 'build',
      systemPrompt: 'You write code.',
      toolNames: ['search', 'read', 'write'],
      maxSteps: 15,
    })
    expect(Object.keys(researcher.tools)).not.toContain('write')
    expect(Object.keys(implementer.tools)).toContain('write')
  })

  it('should produce different outputs for different agent types', async () => {
    const researcher = createAgentType({
      name: 'researcher',
      mode: 'plan',
      systemPrompt: 'You research and report facts.',
      toolNames: ['search'],
      maxSteps: 5,
    })
    const reviewer = createAgentType({
      name: 'reviewer',
      mode: 'plan',
      systemPrompt: 'You review and critique.',
      toolNames: ['search'],
      maxSteps: 5,
    })
    const researchOutput = await runSpecializedAgent(researcher, 'Evaluate TypeScript generics')
    const reviewOutput = await runSpecializedAgent(reviewer, 'Evaluate TypeScript generics')
    expect(researchOutput.agentName).toBe('researcher')
    expect(reviewOutput.agentName).toBe('reviewer')
  })
})
```

---

### Exercise 5: Workspace Isolation

**Objective:** Implement workspace isolation for parallel agents so multiple agents can work on file-based tasks without conflicts.

**Specification:**

1. Create a file `src/exercises/m15/ex05-workspace-isolation.ts`
2. Export an async function `runIsolatedAgents(tasks: IsolatedTask[], sourceDir: string): Promise<IsolationResult>`
3. Define the types:

```typescript
interface IsolatedTask {
  id: string
  description: string
  agentType: string
}

interface WorkspaceResult {
  taskId: string
  workspace: string // Path to the agent's temp directory
  output: string
  changedFiles: string[] // Files that differ from the source
  success: boolean
  durationMs: number
}

interface IsolationResult {
  results: WorkspaceResult[]
  conflicts: string[] // Files modified by more than one agent
  totalDurationMs: number
}
```

4. For each task:
   - Create a temporary directory and copy `sourceDir` into it
   - Run the assigned agent within that workspace
   - After completion, diff the workspace against the source to find changed files

5. After all agents complete, detect conflicts (files modified by multiple agents)

6. Clean up temporary directories after collecting results

**Test specification:**

```typescript
// tests/exercises/m15/ex05-workspace-isolation.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 15: Workspace Isolation', () => {
  it('should create separate workspaces for each agent', async () => {
    const result = await runIsolatedAgents(
      [
        { id: '1', description: 'Add a README', agentType: 'writer' },
        { id: '2', description: 'Add a LICENSE', agentType: 'writer' },
      ],
      '/tmp/test-project'
    )
    expect(result.results).toHaveLength(2)
    expect(result.results[0].workspace).not.toBe(result.results[1].workspace)
  })

  it('should detect conflicts when agents modify the same file', async () => {
    const result = await runIsolatedAgents(
      [
        { id: '1', description: 'Edit index.ts to add feature A', agentType: 'implementer' },
        { id: '2', description: 'Edit index.ts to add feature B', agentType: 'implementer' },
      ],
      '/tmp/test-project'
    )
    // If both agents modified index.ts, it should appear in conflicts
    if (result.results.every(r => r.changedFiles.includes('index.ts'))) {
      expect(result.conflicts).toContain('index.ts')
    }
  })

  it('should clean up temporary workspaces', async () => {
    const result = await runIsolatedAgents(
      [{ id: '1', description: 'Simple task', agentType: 'researcher' }],
      '/tmp/test-project'
    )
    expect(result.results[0].workspace).toBeTruthy()
    // Workspace should be cleaned up after results are collected
  })
})
```

> **Looking Ahead: Agent SDKs** — This module teaches multi-agent orchestration from scratch, which is valuable for understanding the patterns. In production, consider the official Agent SDKs: Anthropic's Claude Agent SDK, OpenAI's Agents SDK, and Mistral's Agents API all provide built-in primitives for structured handoffs between agents, built-in guardrails, tracing, and orchestration — handling many of the patterns you've implemented manually here. Mistral's Agents API additionally offers built-in connectors (web search, code execution, image generation), persistent memory across conversations, and native multi-agent orchestration.

> **Local Alternative (Ollama):** Multi-agent orchestration works with `ollama('qwen3.5')`. The orchestrator-worker pattern, delegation, and shared state are all code-level patterns independent of the model provider. You can even mix providers — use a capable API model as the orchestrator and local models as cheaper workers.

---

## Summary

In this module, you learned:

1. **Why multiple agents:** Single agents suffer from prompt dilution and tool overload. Multi-agent systems assign focused roles with curated tools and prompts.
2. **Orchestrator-worker pattern:** An orchestrator breaks tasks into sub-tasks, delegates to specialized workers, and synthesizes results. Workers can be registered dynamically.
3. **Agent communication:** Structured handoff documents and message-passing protocols enable agents to share context without coupling.
4. **Shared state:** Use shared state for task-level information (goals, findings, status) and private state for agent-internal details (reasoning traces, scratchpads).
5. **Delegation strategies:** Route by capability (matching skills to tasks) or by topic (matching domains to specialists).
6. **Parallel execution:** Run independent agents concurrently with concurrency limits and map-reduce patterns for throughput.
7. **Agent handoff:** Transfer conversations between specialists with context summaries so users do not repeat themselves.
8. **Error handling:** Retries with exponential backoff, fallback agents, and circuit breakers create resilient multi-agent systems.
9. **Agent pool coordinator:** A coordinator manages a pool of workers with concurrency limits, dispatching sub-tasks in a sliding window pattern and aggregating results.
10. **Agent type specialization:** Production systems define agent types with focused prompts, curated tool sets, and per-type step limits — making capabilities explicit and structural.
11. **Workspace isolation:** Parallel agents that might conflict work in separate directories (or git worktrees), with results merged back by the coordinator.
12. **Primary and subagent architecture:** Primary agents are persistent and user-facing; subagents are task-scoped, invoked on demand with fresh context, and return results to the caller.
13. **Agent configuration via markdown:** Declarative agent definitions in markdown files with YAML frontmatter make agent types versionable, shareable, and editable by non-developers.
14. **@Mention invocation:** Users or parent agents invoke specific subagents with `@agent_name` syntax, enabling explicit delegation alongside automatic routing.

In Module 16, you will learn about workflows and chains — a more deterministic alternative to autonomous multi-agent systems for tasks with well-defined steps.
