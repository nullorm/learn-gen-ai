# Module 14: Workflows & Chains

## Learning Objectives

- Distinguish between chains (deterministic pipelines) and agents (autonomous decision-makers)
- Build sequential chains where each step's output feeds the next step
- Run independent steps in parallel for better throughput
- Implement branching logic that routes based on LLM output
- Add retry and fallback strategies to handle step failures
- Create composable chain functions as reusable pipeline building blocks
- Monitor pipelines with logging, timing, and token usage tracking
- Choose between chains and agents for a given problem

> *Module 14 opens **Part IV: Agents & Orchestration** — begin the eight-module arc that earns the **Agent Deployer** badge.*

---

## Why Should I Care?

You'll build autonomous **agents** later in this Part, in Module 16 — systems that decide their own path through a task. Agents are powerful but unpredictable: you gain flexibility but lose predictability. For many production applications, though, you know the steps in advance: extract data, transform it, validate it, generate a report. In these cases, a deterministic **chain** — the subject of this module — is faster to build, easier to debug, cheaper to run, and more reliable than an agent.

Chains are the workhorses of production LLM applications. Content pipelines, data processing, document generation, classification workflows — all benefit from the predictability of chains. You decide the steps. The LLM executes them. No loop, no tool selection, no stuck detection. Just a pipeline that does what you designed it to do.

This module teaches you when to reach for a chain instead of an agent, and how to build chains that are composable, resilient, and observable.

---

## Connection to Other Modules

- **Module 15 (Durable Workflows)** — next — makes these chains survive crashes and week-long waits.
- **Module 16 (Agent Fundamentals)** — later in Part IV — introduces the autonomous approach. This module opens Part IV with the deterministic counterpart, giving you a concrete baseline to contrast agents against.
- **Module 17 (Multi-Agent Systems)** — later in Part IV — uses an agent for each step. Here we use plain LLM calls with no agent loop.
- **Module 20 (Code Generation)** can use chains for generate-test-fix pipelines.
- **Module 21 (Human-in-the-Loop)** adds approval gates within chain steps.

---

## Section 1: Chains vs Agents

### The Spectrum of Control

LLM applications exist on a spectrum between full determinism and full autonomy:

| Aspect         | Chain                              | Agent                                           |
| -------------- | ---------------------------------- | ----------------------------------------------- |
| Steps          | Predefined by developer            | Decided by LLM                                  |
| Tool selection | Fixed per step                     | LLM chooses                                     |
| Flow control   | Developer's code                   | LLM's reasoning                                 |
| Debugging      | Predictable: step N failed         | Non-deterministic: why did it choose that tool? |
| Cost           | Predictable: N calls               | Variable: 1 to step limit calls                 |
| Flexibility    | Low -- handles only designed paths | High -- adapts to unexpected situations         |

The short version: reach for a **chain** when you can write the steps down in advance, and an **agent** when the path depends on what the model finds along the way — Section 8 builds the full decision framework (including the hybrid middle ground) once you have built both.

**File:** `src/agents/chains-vs-agents.ts`

Build two functions that take a topic and produce an article, using different approaches:

**Chain approach** -- `chainApproach(topic: string): Promise<string>`:

1. Use `Output.object` to generate an outline (title + sections with headings and key points) -- this always happens
2. Loop through each section, calling `generateText` to write 2-3 paragraphs per section -- this always happens, in order
3. Combine title and sections into a formatted string -- this always happens

**Agent approach** -- `agentApproach(topic: string): Promise<string>`:

- Single `generateText` call with `stopWhen: isStepCount(10)`, a system prompt saying "research and write an article, stop when you have enough," and a search tool
- The LLM decides how many searches to do and when to stop

```typescript
import { generateText, Output, isStepCount } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
```

Think about: which approach gives you more control over the output structure? Which handles unexpected topics better? Which is easier to debug when the output is wrong?

> **Beginner Note:** Think of a chain like a recipe -- follow the steps in order and you get a predictable result. An agent is like telling a chef "make something delicious" -- the result might be amazing, but you do not know exactly what you will get or how long it will take.

---

## Section 2: Sequential Chains

### Output Feeds Input

**File:** `src/agents/sequential.ts`

The fundamental chain pattern: each step produces output that becomes input for the next step. Start with the types:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface ChainStep<TInput, TOutput> {
  name: string
  execute: (input: TInput) => Promise<TOutput>
}
```

Build a chain runner: `runChain<T>(initialInput: T, steps: ChainStep<any, any>[]): Promise<{ finalOutput: any; stepResults: Array<{ name: string; output: any; durationMs: number }> }>`

The runner should:

- Iterate through steps sequentially, passing each step's output as the next step's input
- Record the name, output, and duration (using `Date.now()`) for each step
- Log progress as each step starts and completes
- Return the final output and the array of step results

Then define three chain steps and run them as a pipeline:

1. `extractKeyPoints: ChainStep<string, string[]>` -- uses `Output.object` to extract key points from text
2. `generateSummary: ChainStep<string[], string>` -- takes key points and writes a summary
3. `translateToFrench: ChainStep<string, string>` -- translates English text to French

Each step's `execute` function wraps a single `generateText` call with the appropriate prompt.

Think about: what happens if you swap the order of steps 2 and 3? Would the types still work? What does TypeScript tell you?

### Typed Sequential Chain

For type safety across chain steps, use a builder pattern with generics.

Build a `TypedChain<TInput, TOutput>` class with:

- `static start<T>(): TypedChain<T, T>` -- creates a new empty chain
- `then<TNext>(name: string, fn: (input: TOutput) => Promise<TNext>): TypedChain<TInput, TNext>` -- appends a step, returning a new chain with updated output type
- `async execute(input: TInput): Promise<TOutput>` -- runs all steps sequentially

```typescript
class TypedChain<TInput, TOutput> {
  private constructor(private steps: Array<{ name: string; fn: (input: any) => Promise<any> }>) {}

  static start<T>(): TypedChain<T, T>
  then<TNext>(name: string, fn: (input: TOutput) => Promise<TNext>): TypedChain<TInput, TNext>
  async execute(input: TInput): Promise<TOutput>
}
```

Define the intermediate types for an article pipeline:

```typescript
interface ArticleRequest {
  topic: string
  audience: string
  wordCount: number
}

interface ArticleOutline {
  title: string
  sections: Array<{ heading: string; points: string[] }>
  audience: string
}

interface ArticleDraft {
  title: string
  content: string
  wordCount: number
}

interface ArticleFinal {
  title: string
  content: string
  wordCount: number
  readabilityScore: number
}
```

Build a pipeline: `TypedChain.start<ArticleRequest>().then<ArticleOutline>('outline', ...).then<ArticleDraft>('write', ...).then<ArticleFinal>('review', ...)`

Each `.then()` callback receives the output of the previous step with full type safety. The outline step uses `Output.object` to produce structured output. The write step iterates through sections. The review step rates readability and returns the enriched draft.

What happens at compile time if you try to chain a step that expects `ArticleOutline` after a step that returns `string`?

> **Advanced Note:** The typed chain builder pattern catches type mismatches at compile time. If step 2 expects `ArticleOutline` but step 1 returns `string`, TypeScript will flag the error. This is much safer than passing untyped data between steps.

---

## Section 3: Parallel Chains

### Running Independent Steps Concurrently

When steps do not depend on each other, run them in parallel to reduce total latency.

**File:** `src/agents/parallel.ts`

Build a parallel step runner with this signature:

```typescript
async function runParallelSteps<T extends Record<string, () => Promise<any>>>(
  steps: T
): Promise<{ [K in keyof T]: Awaited<ReturnType<T[K]>> }>
```

The function should:

- Take an object where each value is an async function (a step)
- Run all steps concurrently using `Promise.all`
- Log start and completion time for each step, plus total wall-clock time
- Return an object with the same keys, each mapped to the step's result

Then build `generateBlogPost(topic: string): Promise<BlogPost>`:

```typescript
interface BlogPost {
  title: string
  introduction: string
  body: string
  conclusion: string
  tags: string[]
}
```

The blog post generator has two phases:

1. **Sequential** -- generate a plan (title + points for intro/body/conclusion) using `Output.object`, because the parallel steps need this plan
2. **Parallel** -- generate introduction, body, conclusion, and tags concurrently using `runParallelSteps`, since they all depend on the plan but not on each other

Think about:

- Why must the plan step be sequential but the section steps can be parallel?
- How much faster is this than generating all sections sequentially?
- What happens to the total time if one section takes 5x longer than the others?

> **Beginner Note:** Parallel execution is like having multiple people work on different sections of a report at the same time, then combining their work at the end. The total time is roughly the time of the slowest section, not the sum of all sections.

---

## Section 4: Branching

### Conditional Routing Based on LLM Output

Sometimes a chain needs to take different paths based on what the LLM produces. This is branching -- a hybrid of chain determinism and agent flexibility.

**File:** `src/agents/branching.ts`

Build `branchByClassification(text: string): Promise<{ classification: string; result: string }>`:

1. **Classify** -- use `Output.object` with a schema containing `category` (enum: question, complaint, feedback, request), `confidence`, and `reasoning`
2. **Route** -- switch on the category, calling `generateText` with a different system prompt for each branch (knowledgeable agent for questions, empathetic agent for complaints, customer success agent for feedback, feature request agent for requests)

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
```

Each branch is a simple `generateText` call with a role-specific system prompt. The classification step makes a lightweight LLM call, and only the selected branch makes the full generation call.

Think about: what should happen when classification confidence is low? Should you route to a generic handler, or ask for clarification?

### Multi-Level Branching

For complex routing, chain multiple classification steps using a recursive tree structure:

```typescript
interface BranchNode {
  classify: (input: string) => Promise<string>
  branches: Record<string, BranchNode | ((input: string) => Promise<string>)>
  default?: (input: string) => Promise<string>
}
```

Build `traverseBranch(node: BranchNode, input: string, path?: string[]): Promise<{ result: string; path: string[] }>`:

- Call `node.classify(input)` to get the category
- Push the category onto the path array
- Look up the handler in `node.branches`
- If not found, use `node.default` (or return an error message)
- If the handler is a function, call it and return the result
- If the handler is another `BranchNode`, recurse

Build a routing tree: top level classifies as sales/support/other; the support branch classifies further as billing/technical/account. Each leaf is a simple async function returning a response string.

How deep should the tree go? What are the latency implications of each classification level?

> **Advanced Note:** Multi-level branching gives you the benefits of both chains (predictable structure) and agents (adaptive routing). Each classification is a lightweight LLM call, keeping costs low while still allowing dynamic routing.

---

## Section 5: Retry and Fallback

### Handling Step Failures

Individual chain steps can fail due to API errors, malformed output, or timeouts. Robust chains include retry logic and fallbacks.

> **Gotcha:** Retry only what's *idempotent*. Retrying a step that already charged a card, sent an email, or appended to a file replays the side effect — now you've double-charged. Make side-effecting steps idempotent (dedupe keys, upserts) before wrapping them in retry, or you've built an automatic way to do the wrong thing twice.

**File:** `src/agents/retry-fallback.ts`

Build `withRetry<T>(name: string, fn: () => Promise<T>, config: RetryConfig): Promise<T>`:

```typescript
interface RetryConfig {
  maxRetries: number
  initialDelayMs: number
  maxDelayMs: number
  retryableErrors?: string[]
}
```

The function should:

- Loop from 0 to `maxRetries`, calling `fn()` each time
- On failure, compute delay with exponential backoff: `Math.min(initialDelayMs * Math.pow(2, attempt - 1), maxDelayMs)`
- If `retryableErrors` is provided, check whether the error message contains any of the patterns. If not, throw immediately (non-retryable)
- After all retries are exhausted, throw the last error

Build `withFallback<T>(primary, fallback, name): Promise<{ result: T; usedFallback: boolean }>`:

- Try the primary function
- If it throws, log a warning and try the fallback
- Return the result along with a flag indicating whether the fallback was used

Then compose them into `robustSummarize(text: string): Promise<string>`:

- **Primary**: use `withRetry` wrapping a structured output call (`Output.object` with a schema requiring `summary` and `keyTakeaways`), with a post-validation check that the summary meets a minimum length
- **Fallback**: a simple unstructured `generateText` call

Think about: when is it better to retry (transient error) versus fall back (systematic error)? How do you classify errors into these categories?

### Validation Between Steps

Add validation gates between chain steps to catch bad data early.

```typescript
interface ValidationResult {
  valid: boolean
  errors: string[]
}

type Validator<T> = (data: T) => ValidationResult
```

Build `chainWithValidation<T>(input: T, steps: Array<{ name: string; execute: (input: any) => Promise<any>; validate?: Validator<any> }>): Promise<{ output: any; validationLog: Array<{ step: string; result: ValidationResult }> }>`:

- Execute each step sequentially
- After each step, if a `validate` function is provided, call it on the output
- If validation fails, throw with the step name and error messages
- Return the final output and the full validation log

Build a two-step example: `generate-outline` (validate that sections count is between 3 and 5) followed by `write-content` (validate that content is at least 200 characters).

Why is it valuable to catch a bad outline before spending tokens on writing the full content?

---

## Section 6: Composable Chain Functions

### Building Reusable Pipeline Units

**File:** `src/agents/composable.ts`

Create small, focused chain functions that can be composed into larger pipelines. Start with the composition utilities:

```typescript
type StepFn<TIn, TOut> = (input: TIn) => Promise<TOut>

function compose<A, B, C>(first: StepFn<A, B>, second: StepFn<B, C>): StepFn<A, C> {
  return async (input: A) => second(await first(input))
}

function pipe<T>(...fns: StepFn<any, any>[]): StepFn<T, any> {
  return async (input: T) => {
    let result: any = input
    for (const fn of fns) result = await fn(result)
    return result
  }
}
```

Build a library of reusable step factories. Each factory returns a `StepFn`:

- `summarize(maxLength?: number): StepFn<string, string>` -- wraps `generateText` with a prompt to summarize in N words
- `translate(targetLanguage: string): StepFn<string, string>` -- wraps `generateText` with a translation prompt
- `extractEntities(): StepFn<string, Array<{ name: string; type: string }>>` -- uses `Output.object` with a schema for entity extraction
- `classifySentiment(): StepFn<string, { text: string; sentiment: string; score: number }>` -- uses `Output.object` for sentiment analysis
- `formatAsMarkdown(title: string): StepFn<string, string>` -- wraps `generateText` to add headers and formatting

Each factory captures its configuration in a closure and returns a function that takes input and returns a promise.

Compose pipelines from the library:

```typescript
const summarizeAndTranslate = pipe<string>(summarize(100), translate('Spanish'))
const processContent = pipe<string>(summarize(200), formatAsMarkdown('Content Summary'))
```

Think about: how does `pipe` handle type safety (or not) across steps? Could you build a type-safe version?

### Factory Pattern for Chains

Build `createContentPipeline(config: ContentPipelineConfig)` that returns an object with a `process(rawContent: string)` method:

```typescript
interface ContentPipelineConfig {
  model: string
  summarizeLength?: number
  targetLanguage?: string
  outputFormat?: 'markdown' | 'html' | 'plain'
}
```

The `process` method runs a pipeline: summarize, optionally translate (if `targetLanguage` is set), optionally format (if `outputFormat` is not `'plain'`), and extract metadata (word count, topics, reading time via `Output.object`). It returns `{ summary, formatted, metadata }`.

This pattern lets you create multiple pipeline configurations (`englishPipeline`, `spanishPipeline`) from the same factory, each with different settings.

> **Beginner Note:** Composable chain functions are like LEGO bricks -- small pieces that snap together to build bigger structures. Once you build a `summarize` function, you can use it in any pipeline without rewriting it.

---

## Section 7: Pipeline Monitoring

### Logging, Timing, and Token Usage

**File:** `src/agents/monitoring.ts`

Production chains need observability. Track what happens at each step. Define the metrics types:

```typescript
interface StepMetrics {
  stepName: string
  startTime: number
  endTime: number
  durationMs: number
  inputTokens: number
  outputTokens: number
  totalTokens: number
  success: boolean
  error?: string
}

interface PipelineMetrics {
  pipelineName: string
  startTime: number
  endTime: number
  totalDurationMs: number
  steps: StepMetrics[]
  totalInputTokens: number
  totalOutputTokens: number
  totalTokens: number
  estimatedCost: number
  success: boolean
}
```

Build a `MonitoredPipeline` class:

```typescript
class MonitoredPipeline {
  private steps: Array<{ name: string; fn: (input: any) => Promise<any> }> = []

  constructor(private name: string) {}

  addStep(name: string, fn: (input: any) => Promise<any>): this
  async execute(input: any): Promise<{ output: any; metrics: PipelineMetrics }>
}
```

The `execute` method should:

- Time each step individually and the pipeline overall
- Capture token usage from `generateText` results (check for a `usage` property on the result object with `inputTokens` and `outputTokens`)
- If a step returns a `generateText` result, extract `.text` or `.output` as the value to pass forward
- If a step fails, record the error, mark the pipeline as failed, and stop
- Compute estimated cost from token counts (use per-token pricing constants)
- Return the final output alongside the full metrics

Also build a `printMetrics(metrics: PipelineMetrics): void` helper that formats the metrics into a readable summary showing pipeline name, status, total duration, total tokens (input/output breakdown), estimated cost, and per-step breakdown.

Think about: what metric would you alert on in production? Duration spikes? Token count anomalies? Cost per execution?

> **Advanced Note:** In production, send pipeline metrics to a monitoring system (Datadog, Prometheus, or a custom dashboard). Track metrics over time to detect degradation -- a step that usually takes 500ms but starts taking 3000ms indicates a problem. Also track token usage for cost forecasting.

---

## Section 8: When to Use Chains vs Agents

> **Decision:** Chain or agent? If you can write the steps down in advance, use a **chain** — it's cheaper, debuggable, and predictable. If the path depends on what intermediate results say, use an **agent**. If most of it is fixed but one stage must adapt, use a **hybrid**: a chain with one agent-powered step. Default to the most constrained option that does the job.

### Decision Framework

**File:** `src/agents/decision.ts`

Build `recommendApproach(taskDescription: string): Promise<{ recommendation: 'chain' | 'agent' | 'hybrid'; reasoning: string; factors: Record<string, string> }>`:

Use `Output.object` to have the LLM analyze a task description and recommend an approach. The schema should include:

- `recommendation`: enum of chain/agent/hybrid
- `reasoning`: explanation
- `factors`: an object with keys like `stepsKnown` (yes/mostly/no), `needsAdaptation` (rarely/sometimes/often), `costSensitivity` (high/medium/low), `debuggability` (critical/important/nice_to_have), `latencyRequirement` (strict/moderate/flexible)

The prompt should ask the LLM to consider whether steps are known in advance, whether the approach needs to adapt based on intermediate results, and how important cost predictability, debugging, and latency are.

Test it with a few examples to see how the recommendations differ:

- "Translate a document from English to French, then generate a summary"
- "Research a topic and write an article, adjusting the research based on what you find"
- "Process customer support tickets: classify, route, and generate responses"
- "Explore a codebase to find and fix a bug described by a user"

### Practical Guidelines

| Scenario                      | Approach | Why                                                |
| ----------------------------- | -------- | -------------------------------------------------- |
| Document translation pipeline | Chain    | Steps are fixed: extract, translate, format        |
| Customer support chatbot      | Agent    | Needs to adapt based on user responses             |
| Content moderation            | Chain    | Classify, flag, escalate -- deterministic flow     |
| Research assistant            | Agent    | Must decide what to search based on findings       |
| Data ETL with LLM             | Chain    | Extract, transform, load -- fixed pipeline         |
| Code debugging                | Agent    | Must explore, hypothesize, test -- adaptive        |
| Email drafting from template  | Chain    | Fill template, review, format -- fixed steps       |
| Multi-source fact checking    | Hybrid   | Chain structure with agent-like search flexibility |

> **Advanced Note:** In production, the hybrid from the Decision above usually looks like one agent-powered stage boxed inside an otherwise fixed chain: a content pipeline might use an agent for the "research" step (where the path depends on findings) but plain chain steps for "format" and "publish". The agent's unpredictability stays contained in a single stage while everything around it remains deterministic and debuggable.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Going Further: Production Workflow Patterns

Sections 1–8 are everything you need to design and run a workflow. These last four are the production layer — middleware, background tasks, undo/redo, headless execution — that you add when workflows graduate to infrastructure.

### Workflow Middleware and Hooks

#### Intercepting Workflow Execution

A hook system lets you insert logic at lifecycle points in a workflow without modifying the steps themselves. Hooks run before and after each step, enabling cross-cutting concerns like logging, validation, timing, and authorization.

Common hook points:

- **PreStep** — runs before a step executes. Can validate input, log the start, or block execution
- **PostStep** — runs after a step completes. Can validate output, log results, or transform the output
- **OnError** — runs when a step fails. Can log diagnostics, trigger alerts, or decide whether to retry
- **OnComplete** — runs when the entire workflow finishes

```typescript
interface WorkflowHook {
  name: string
  point: 'preStep' | 'postStep' | 'onError' | 'onComplete'
  handler: (context: HookContext) => Promise<void | 'skip' | 'abort'>
}

interface HookContext {
  stepName: string
  input: unknown
  output?: unknown
  error?: Error
  timing: { startMs: number; durationMs?: number }
}
```

A `preStep` hook returning `'skip'` skips the step. Returning `'abort'` stops the workflow. This gives hooks control over execution flow without the steps knowing about it.

The middleware pattern is the same principle applied as a wrapper. Each middleware wraps the step function, adding behavior before and after:

```typescript
type Middleware = (step: StepFn) => StepFn
const withTiming: Middleware = step => async input => {
  const start = Date.now()
  const result = await step(input)
  console.log(`Step took ${Date.now() - start}ms`)
  return result
}
```

Hooks and middleware keep your step functions clean — each step does one thing, and cross-cutting concerns live in the hook system.

---

### Background Execution

#### Parallel Background Tasks

Some workflow branches do not need to complete before the main flow continues. A background task runs independently — the main workflow proceeds immediately while the background task works in parallel.

Common use cases:

- **Analytics logging** — log detailed metrics about each step without blocking the pipeline
- **Cache warming** — pre-compute and cache results for likely follow-up queries
- **Notification dispatch** — send notifications about workflow progress without waiting for delivery confirmation

```typescript
function runInBackground(task: () => Promise<void>): void {
  task().catch(error => console.error('Background task failed:', error))
}

// In a workflow step:
runInBackground(async () => {
  await logAnalytics(stepResult)
  await warmCache(stepResult.topic)
})
// Main flow continues immediately
```

Background tasks must be fire-and-forget or tracked separately. They should never modify state that the main flow depends on. If a background task fails, it fails silently (with logging) — it must not crash the main workflow.

For tasks that the workflow eventually needs, use a future/promise pattern: start the task early, continue with other steps, and `await` the result only when needed. This is a hybrid between sequential and background execution.

> **Beginner Note:** The simplest background task is just a `Promise` you do not `await`. Be careful — unhandled promise rejections can crash your process. Always add `.catch()` to fire-and-forget promises.

---

### Undo/Redo for Workflow Steps

*Related: Module 20 applies this same reversibility idea to code edits.*

#### Reversible Workflows

Workflows that modify state — editing files, updating databases, calling external APIs — benefit from reversibility. An undo/redo system tracks what each step changed and can roll back or reapply those changes.

The pattern:

1. Before each step executes, record a snapshot or diff of the state it will modify
2. After execution, store the change record in a changelog
3. **Undo** rolls back the most recent step's change by applying the inverse operation
4. **Redo** reapplies a rolled-back change from the changelog

```typescript
interface ChangeRecord {
  stepName: string
  timestamp: number
  changes: Array<{
    type: 'file_edit' | 'file_create' | 'file_delete'
    path: string
    before: string | null // null for creates
    after: string | null // null for deletes
  }>
}
```

The key insight: only side effects are reversed. The step log — which steps ran, with what inputs, outputs, and validation results — is preserved. Rolling a step back reverts its outcome without erasing the execution record, so when the pipeline re-runs after a rollback it still has the previous run's log to compare against (and can skip steps whose outputs were already validated).

For file-based pipelines, undo means restoring the previous file content. For database pipelines, undo means running a compensating transaction. For API calls, undo may not be possible — flag irreversible steps so the operator knows before executing.

> **Advanced Note:** Implement a change stack with a cursor. The cursor points to the current position. Undo moves the cursor back, redo moves it forward. New changes after an undo discard the redo history (just like text editor undo). This gives you a navigable history of workflow execution.

---

### Headless Execution for CI/CD

*Module 27 (Deployment) owns the full headless/CI story; here the focus is the workflow logic itself.*

#### Non-Interactive Workflows

Any workflow system that only works interactively is limited to human-in-the-loop use cases. Headless execution lets the same workflow logic run non-interactively — accepting input via function arguments or stdin, executing all steps without prompts, and returning structured results.

This unlocks automation:

- **CI/CD integration** — run code review workflows on every pull request
- **Scheduled tasks** — run analysis pipelines on a cron schedule
- **Batch processing** — process hundreds of documents through the same workflow

```typescript
interface WorkflowMode {
  interactive: boolean
  confirmBeforeStep?: boolean // Ask user before each step (interactive only)
  outputFormat: 'text' | 'json'
}

async function runWorkflow(input: string, mode: WorkflowMode): Promise<WorkflowResult> {
  for (const step of steps) {
    if (mode.interactive && mode.confirmBeforeStep) {
      const proceed = await promptUser(`Run "${step.name}"?`)
      if (!proceed) continue
    }
    // Execute step regardless of mode
    await step.execute(input)
  }
}
```

The same workflow function supports both modes. In interactive mode, it prompts for confirmation between steps and displays progress. In headless mode, it executes all steps automatically and returns structured JSON. The workflow logic does not change — only the I/O layer differs.

Design workflows for headless execution from the start. Avoid hardcoded `console.log` or `readline` calls in step functions. Instead, emit events that the I/O layer can handle differently based on the mode.

---

## Summary

In this module, you learned:

1. **Chains vs agents:** Chains have predefined steps for predictable, debuggable pipelines. Agents decide their own steps for flexible, adaptive behavior. Choose based on how well-defined the task is.
2. **Sequential chains:** Each step's output feeds the next step. Type-safe chain builders catch errors at compile time.
3. **Parallel chains:** Independent steps run concurrently to reduce total latency. Use `Promise.all`.
4. **Branching:** Conditional routing based on LLM classification creates adaptive chains without full agent autonomy. Multi-level branching handles complex routing trees.
5. **Retry and fallback:** Retry with exponential backoff handles transient errors. Fallbacks provide degraded but functional results when primary steps fail. Validation between steps catches bad data early.
6. **Composable chain functions:** Small, focused step functions compose into larger pipelines. Factory patterns create configurable pipeline variants.
7. **Pipeline monitoring:** Track timing, token usage, and costs at each step. Production pipelines need persistent metrics for debugging and cost forecasting.
8. **When to use which:** Start with chains when steps are known. Move to agents when flexibility is needed. Use hybrids when you need both structure and adaptability.
9. **Workflow middleware and hooks:** PreStep, PostStep, OnError, and OnComplete hooks inject cross-cutting concerns (logging, validation, timing) without modifying step functions.
10. **Background execution:** Fire-and-forget tasks (analytics, cache warming, notifications) run in parallel without blocking the main workflow, using `.catch()` to prevent unhandled rejections.
11. **Undo/redo for workflow steps:** Recording state snapshots before each side-effecting step enables reversible workflows — undo reverts outcomes while preserving decision history.
12. **Headless execution:** The same workflow logic supports interactive and non-interactive modes, enabling CI/CD integration, scheduled tasks, and batch processing without code changes.

Next, **Module 15 (Durable Workflows)** takes the chains you built here and makes them durable — able to survive crashes, redeploys, and week-long waits between steps. Later in the course, Module 20 applies these chain patterns to code generation, a domain where iterative refinement and test-driven approaches produce the best results.

---

## Quiz

### Question 1 (Easy)

What is the fundamental difference between a chain and an agent?

- A) Chains use fewer tokens than agents
- B) Chains have predefined steps while agents decide their own steps
- C) Chains cannot use LLMs, only agents can
- D) Agents are always faster than chains

**Answer: B** — In a chain, the developer defines the steps in advance and the LLM executes each step. In an agent, the LLM decides which steps to take based on its reasoning and tool results. Chains may or may not use fewer tokens (A) — it depends on the task. Both use LLMs (C). Speed depends on the specific implementation (D).

---

### Question 2 (Easy)

When running parallel chain steps, what must be true about the steps?

- A) They must all use the same LLM model
- B) They must all produce the same output type
- C) They must be independent — no step depends on another's output
- D) They must all complete within the same time limit

**Answer: C** — Parallel steps run concurrently, so none of them can depend on the output of another parallel step. If step B needs step A's output, they must run sequentially. The steps can use different models (A), produce different types (B), and take different amounts of time (D).

---

### Question 3 (Medium)

What is the purpose of validation between chain steps?

- A) To speed up the pipeline by skipping unnecessary steps
- B) To catch bad data early before it propagates to later steps and produces poor final output
- C) To reduce the number of tokens used
- D) To convert data between different formats

**Answer: B** — Validation between steps catches problems early. If step 1 produces a summary that is too short or missing key information, validation can detect this before step 2 uses that bad summary. Without validation, errors compound through the pipeline and the final output is poor with no clear indication of where things went wrong.

---

### Question 4 (Hard)

A content pipeline processes 1000 documents daily. Each document goes through: classify (200ms), summarize (800ms), translate (600ms), and format (100ms). Classify and summarize are independent. Translate depends on summarize. Format depends on translate. What is the theoretical minimum processing time per document?

- A) 1700ms (all sequential)
- B) 900ms (classify parallel with summarize, then translate, then format)
- C) 1500ms (summarize, translate, format — classify is free)
- D) 800ms (only summarize matters)

**Answer: C** — Since classify (200ms) and summarize (800ms) are independent, they can run in parallel. The parallel phase takes max(200, 800) = 800ms. Then translate (600ms) must wait for summarize, and format (100ms) must wait for translate. Total: 800 + 600 + 100 = 1500ms. Classify is effectively "free" because it completes within the time summarize takes. The full sequential time would be 1700ms (A), so parallelizing classify and summarize saves 200ms.

---

### Question 5 (Medium)

What is the main advantage of composable chain functions over monolithic pipelines?

- A) Composable functions use less memory
- B) Composable functions are always faster
- C) Individual steps can be tested, reused, and recombined into different pipelines
- D) Composable functions do not need error handling

**Answer: C** — Composable chain functions are like building blocks. A `summarize()` function can be tested in isolation, reused across multiple pipelines, and combined with different steps to create new pipelines. A monolithic pipeline that does everything in one function cannot be partially reused and is harder to test. Memory (A) and speed (B) are generally not affected by composability. Error handling (D) is still needed.

---

## Exercises

### Exercise 1: Content Pipeline

**Objective:** Build a complete content pipeline that takes a topic through five stages: research, outline, draft, review, and final revision.

**Specification:**

1. Create a file `src/exercises/m14/ex01-content-pipeline.ts`
2. Export an async function `contentPipeline(topic: string, options?: PipelineOptions): Promise<PipelineResult>`
3. Define the types:

```typescript
interface PipelineOptions {
  wordCount?: number // default: 800
  targetAudience?: string // default: "general readers"
  verbose?: boolean // default: false
}

interface StepOutput {
  stepName: string
  output: string
  durationMs: number
  tokenCount: number
}

interface PipelineResult {
  finalContent: string
  steps: StepOutput[]
  totalDurationMs: number
  totalTokens: number
  reviewScore: number // 1-10 from the review step
}
```

4. Implement five sequential steps:
   - **Research**: Generate key facts and data points about the topic (use `generateText` with `Output.object` and a structured schema)
   - **Outline**: Create a structured outline from the research (sections with bullet points)
   - **Draft**: Write the full content based on the outline
   - **Review**: Score the draft (1-10) and provide specific feedback
   - **Final**: Revise the draft based on review feedback

5. Track timing and token usage for each step

6. If `verbose` is true, print progress after each step

**Example usage:**

```typescript
const result = await contentPipeline('The Rise of Rust Programming Language', {
  wordCount: 600,
  targetAudience: 'software developers',
  verbose: true,
})

console.log(result.finalContent)
console.log(`Total time: ${result.totalDurationMs}ms`)
console.log(`Total tokens: ${result.totalTokens}`)
console.log(`Review score: ${result.reviewScore}/10`)
```

**Test specification:**

```typescript
// tests/exercises/m14/ex01-content-pipeline.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Content Pipeline', () => {
  it('should produce final content', async () => {
    const result = await contentPipeline('TypeScript generics')
    expect(result.finalContent).toBeTruthy()
    expect(result.finalContent.length).toBeGreaterThan(200)
  })

  it('should complete all 5 steps', async () => {
    const result = await contentPipeline('Functional programming')
    expect(result.steps.length).toBe(5)
    expect(result.steps.map(s => s.stepName)).toEqual(['research', 'outline', 'draft', 'review', 'final'])
  })

  it('should track timing for each step', async () => {
    const result = await contentPipeline('Cloud computing')
    for (const step of result.steps) {
      expect(step.durationMs).toBeGreaterThan(0)
    }
    expect(result.totalDurationMs).toBeGreaterThan(0)
  })

  it('should produce a review score between 1 and 10', async () => {
    const result = await contentPipeline('Machine learning basics')
    expect(result.reviewScore).toBeGreaterThanOrEqual(1)
    expect(result.reviewScore).toBeLessThanOrEqual(10)
  })

  it('should respect word count option', async () => {
    const result = await contentPipeline('APIs', { wordCount: 300 })
    const wordCount = result.finalContent.split(/\s+/).length
    // Allow some variance — LLMs do not hit exact word counts
    expect(wordCount).toBeGreaterThan(150)
    expect(wordCount).toBeLessThan(600)
  })
})
```

---

### Exercise 2: Composable Pipeline Library

**Objective:** You already built the composition utilities and step factories in Section 6 (`src/agents/composable.ts`). Reuse that library — do not rebuild it — to assemble pipelines you have not built before, and take a shot at the type-safe `pipe` that Section 6's closing question teased.

**Specification:**

1. Create a file `src/exercises/m14/ex02-composable-chains.ts`
2. Port your Section 6 factories into the exercise library, adding an optional trailing `model` parameter (defaulting to your provider's model) so a pipeline can run against a different model — or a mock — without editing the factory:

```typescript
// Each factory returns a step function: (input: string) => Promise<string>

export function summarize(maxWords?: number, model?: LanguageModel): (text: string) => Promise<string>
export function translate(targetLanguage: string, model?: LanguageModel): (text: string) => Promise<string>
export function formatAs(format: 'markdown' | 'html' | 'bullet-points', model?: LanguageModel): (text: string) => Promise<string>
export function reviewAndImprove(model?: LanguageModel): (text: string) => Promise<string>
export function extractKeyPoints(count?: number, model?: LanguageModel): (text: string) => Promise<string[]>
```

3. Export a `pipeline` function that composes steps (your Section 6 `pipe`, under its exercise name):

```typescript
export function pipeline<T>(...steps: Array<(input: any) => Promise<any>>): (input: T) => Promise<any>
```

4. Assemble at least three pipelines you did **not** build in Section 6 — for example: summarize then translate('French'); summarize then reviewAndImprove then formatAs('markdown'); extractKeyPoints then a joining step then formatAs('bullet-points')
5. Each step function should use the Vercel AI SDK with Mistral by default
6. **Bonus:** build the type-safe `pipe` teased by Section 6's "could you build a type-safe version?" question — model it on Section 2's `TypedChain` builder so that composing a step returning `string[]` into a step expecting `string` fails at compile time

**Test specification:**

```typescript
// tests/exercises/m14/ex02-composable-chains.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Composable Chains', () => {
  it('should summarize text', async () => {
    const step = summarize(50)
    const result = await step('A very long text about programming...')
    expect(result.split(/\s+/).length).toBeLessThan(100)
  })

  it('should compose summarize and translate', async () => {
    const process = pipeline<string>(summarize(50), translate('French'))
    const result = await process('TypeScript is a great language...')
    expect(result).toBeTruthy()
  })

  it('should compose multiple steps', async () => {
    const process = pipeline<string>(summarize(100), reviewAndImprove(), formatAs('markdown'))
    const result = await process('A long article about web development...')
    expect(result).toContain('#') // Should have markdown headers
  })
})
```

---

### Exercise 3: Multi-Step Command Chain

**Objective:** Extend your Section 2 `runChain` (`src/agents/sequential.ts`) into a production-grade executor. Keep the same output-feeds-input threading, and layer on Section 5's validation gates plus error handling, lifecycle hooks, `stopOnFailure`, skip support, and step-level reporting. This is an upgrade of code you already have, not a from-scratch rebuild.

**Specification:**

1. Create a file `src/exercises/m14/ex03-command-chain.ts`
2. Export an async function `runCommandChain(input: string, steps: ChainStep[], options?: ChainOptions): Promise<ChainResult>`
3. Define the types:

```typescript
interface ChainStep {
  name: string
  execute: (input: string) => Promise<string>
  validate?: (output: string) => boolean // Optional validation — return false to abort
}

interface ChainOptions {
  stopOnFailure?: boolean // default: true
  hooks?: WorkflowHook[] // the WorkflowHook type from Going Further ("Workflow Middleware and Hooks")
  verbose?: boolean // default: false
}

interface StepReport {
  name: string
  input: string
  output: string
  durationMs: number
  success: boolean
  error?: string
  skipped: boolean
}

interface ChainResult {
  finalOutput: string
  steps: StepReport[]
  totalDurationMs: number
  completedSteps: number
  abortedAtStep?: string // Name of the step that caused early termination
}
```

4. Implement the chain executor:
   - Start from your Section 2 `runChain` loop: execute steps in sequence, passing each step's output as the next step's input
   - If a step throws, catch the error, record it in the step report, and either abort (if `stopOnFailure`) or continue with the previous step's output
   - If a step's `validate` function returns false, abort the chain at that step
   - Run any registered hooks at the appropriate lifecycle points

5. Build a demo chain: `validate → transform → enrich → format` — where validate checks input length, transform uses an LLM to rephrase, enrich adds context, and format structures the output

**Test specification:**

```typescript
// tests/exercises/m14/ex03-command-chain.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Multi-Step Command Chain', () => {
  it('should execute all steps in sequence', async () => {
    const steps: ChainStep[] = [
      { name: 'uppercase', execute: async s => s.toUpperCase() },
      { name: 'trim', execute: async s => s.trim() },
      { name: 'prefix', execute: async s => `Result: ${s}` },
    ]
    const result = await runCommandChain('  hello world  ', steps)
    expect(result.finalOutput).toBe('Result: HELLO WORLD')
    expect(result.completedSteps).toBe(3)
  })

  it('should stop on failure when configured', async () => {
    const steps: ChainStep[] = [
      { name: 'step1', execute: async s => s },
      {
        name: 'failing',
        execute: async () => {
          throw new Error('boom')
        },
      },
      { name: 'step3', execute: async s => s },
    ]
    const result = await runCommandChain('input', steps, { stopOnFailure: true })
    expect(result.completedSteps).toBe(1)
    expect(result.abortedAtStep).toBe('failing')
  })

  it('should abort on validation failure', async () => {
    const steps: ChainStep[] = [
      {
        name: 'validate',
        execute: async s => s,
        validate: output => output.length > 10,
      },
      { name: 'process', execute: async s => s.toUpperCase() },
    ]
    const result = await runCommandChain('short', steps)
    expect(result.abortedAtStep).toBe('validate')
  })

  it('should track timing for each step', async () => {
    const steps: ChainStep[] = [
      { name: 'step1', execute: async s => s },
      { name: 'step2', execute: async s => s },
    ]
    const result = await runCommandChain('input', steps)
    for (const step of result.steps) {
      expect(step.durationMs).toBeGreaterThanOrEqual(0)
    }
  })
})
```

---

### Exercise 4: Model Fallback Chain

**Objective:** Build a model fallback chain that tries a cheap model first and falls back to a more expensive model if the result is unsatisfactory.

**Specification:**

1. Create a file `src/exercises/m14/ex04-model-fallback.ts`
2. Export an async function `generateWithFallback(prompt: string, options?: FallbackOptions): Promise<FallbackResult>`
3. Define the types:

```typescript
interface ModelConfig {
  name: string
  modelId: string
  provider: string
  costTier: 'cheap' | 'standard' | 'expensive'
}

interface FallbackOptions {
  models?: ModelConfig[] // Ordered from cheapest to most expensive
  qualityThreshold?: number // 0-1, minimum quality score to accept (default: 0.7)
  maxAttempts?: number // default: models.length
}

interface AttemptRecord {
  model: string
  costTier: string
  response: string
  qualityScore: number
  accepted: boolean
  durationMs: number
}

interface FallbackResult {
  finalResponse: string
  selectedModel: string
  attempts: AttemptRecord[]
  totalDurationMs: number
  fellBack: boolean // true if the first model was not accepted
}
```

4. Implement the fallback chain:
   - Try models in order from cheapest to most expensive
   - After each attempt, evaluate the quality of the response (use an LLM-as-judge call or a heuristic like response length and structure)
   - If quality meets the threshold, accept the response and stop
   - If quality is below threshold, try the next model
   - If all models are exhausted, return the best attempt

5. Include at least two model tiers in the default configuration

**Test specification:**

```typescript
// tests/exercises/m14/ex04-model-fallback.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Model Fallback Chain', () => {
  it('should return a response', async () => {
    const result = await generateWithFallback('Explain closures in JavaScript')
    expect(result.finalResponse).toBeTruthy()
    expect(result.selectedModel).toBeTruthy()
  })

  it('should record all attempts', async () => {
    const result = await generateWithFallback('Write a haiku about TypeScript')
    expect(result.attempts.length).toBeGreaterThan(0)
    expect(result.attempts[0].durationMs).toBeGreaterThan(0)
  })

  it('should accept first model if quality is sufficient', async () => {
    const result = await generateWithFallback('What is 2 + 2?', {
      qualityThreshold: 0.3, // Low threshold — cheap model should pass
    })
    expect(result.attempts).toHaveLength(1)
    expect(result.fellBack).toBe(false)
  })

  it('should fall back when quality is too low', async () => {
    const result = await generateWithFallback(
      'Write a detailed technical analysis of WebAssembly memory management',
      { qualityThreshold: 0.95 } // Very high threshold — likely to trigger fallback
    )
    if (result.attempts.length > 1) {
      expect(result.fellBack).toBe(true)
    }
  })
})
```

> **Local Alternative (Ollama):** Workflows and chains are code-level orchestration — sequential steps, parallel execution, branching, and retries work identically with `ollama('qwen3.5', { think: false })`. Workflows are especially well-suited to local models because each step is a focused, bounded LLM call rather than a complex open-ended generation.

---
