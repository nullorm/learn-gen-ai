# Module 23: Observability

## Learning Objectives

- Understand why observability is critical for non-deterministic LLM applications in production
- Implement structured logging that captures request/response pairs, token usage, and metadata
- Build distributed tracing through RAG pipelines, agent loops, and multi-step workflows
- Define and collect key metrics: latency, token usage, error rates, and cost per request
- Diagnose common LLM failures including hallucinations, wrong tool choices, and context window issues
- Design dashboards that surface actionable insights about LLM application health
- Configure alerting for cost spikes, error rate increases, and latency degradation
- Handle privacy considerations when logging prompts and completions that may contain PII

---

## Why Should I Care?

Traditional software is deterministic. When a function fails, you check the stack trace, find the bug, fix it, and move on. LLM applications are fundamentally different. The same input can produce different outputs. A prompt that worked yesterday may fail today after a model update. A RAG pipeline that answers correctly 95% of the time silently hallucinates on the other 5%. An agent that usually picks the right tool occasionally picks the wrong one and produces a confident but incorrect answer.

Without observability, you are blind. You cannot tell if your application is working well or failing silently. You cannot tell if a prompt change improved quality or degraded it. You cannot tell if costs are spiraling because one user is making expensive requests or because your caching layer is broken. You cannot tell if latency increased because the model is slower or because your retrieval pipeline is hitting a cold cache.

Observability for LLM applications goes beyond traditional APM. You need to track not just "did the request succeed" but "was the answer correct, was the right model used, did retrieval find relevant documents, did the agent take a reasonable path, and how much did it cost." This module teaches you to build comprehensive observability from scratch using TypeScript and the Vercel AI SDK.

---

## Connection to Other Modules

- **Module 6 (Streaming)** produces incremental outputs that need special logging approaches (you cannot log the full response until streaming completes).
- **Module 7 (Tool Use)** creates tool calls that need tracing to understand which tools were selected and why.
- **Module 9-10 (RAG)** builds retrieval pipelines where observability is critical to understanding retrieval quality.
- **Module 14-15 (Agents)** create multi-step loops where tracing is essential to debug reasoning paths.
- **Module 19 (Evals)** provides the quality measurement framework that observability data feeds into.
- **Module 22 (Cost Optimization)** depends on cost tracking data that the observability layer collects.

---

## Section 1: Why Observability for LLM Apps?

### The Non-Determinism Challenge

Traditional application monitoring assumes deterministic behavior. A database query either returns the right data or throws an error. An HTTP request either succeeds or fails. LLM calls exist in a gray zone -- they almost always "succeed" (return a 200 status code with generated text) but the content may be wrong, incomplete, hallucinated, or harmful.

Consider what happens when you call `generateText` with a factual question. The API returns HTTP 200, reports token counts and latency -- all green from the perspective of traditional monitoring. But was the answer actually correct? Did the model confuse one historical treaty with another? Traditional monitoring has no idea.

```typescript
// The fundamental observability challenge: this call "succeeds" even when the
// answer is wrong. Traditional error monitoring will not catch it.
interface LLMCallOutcome {
  technicalSuccess: boolean // HTTP 200, no errors
  contentCorrect: boolean // Answer is actually right
  safeOutput: boolean // No harmful content
  costReasonable: boolean // Did not use excessive tokens
  latencyAcceptable: boolean // Response time within SLA
}

// In traditional software, technicalSuccess === everything is fine.
// In LLM applications, all five dimensions matter.
```

What question should you ask yourself after every LLM call in production? Not just "did it return 200?" but something much broader.

### What Makes LLM Observability Different

LLM applications require observability across multiple dimensions that traditional applications do not have:

```typescript
// The five pillars of LLM observability
interface LLMObservabilityPillars {
  // 1. Request/Response Quality
  // Did the model produce a correct, relevant, safe answer?
  quality: {
    responseRelevance: number // 0-1 score
    factualAccuracy: boolean
    safetyCompliance: boolean
    formatCorrectness: boolean
  }

  // 2. Performance
  // How fast, how many tokens, which model?
  performance: {
    latencyMs: number
    timeToFirstTokenMs: number
    tokensPerSecond: number
    modelUsed: string
  }

  // 3. Cost
  // How much did this request cost? Is it within budget?
  cost: {
    inputTokens: number
    outputTokens: number
    totalCostUsd: number
    cacheHit: boolean
    modelTier: string
  }

  // 4. Pipeline Health
  // For RAG: did retrieval work? For agents: did tool selection work?
  pipeline: {
    retrievalRelevance: number
    toolCallsCorrect: boolean
    agentStepCount: number
    fallbacksTriggered: number
  }

  // 5. User Experience
  // Did the user get what they needed?
  userExperience: {
    streamingLatency: number
    conversationTurnCount: number
    userFeedback: 'positive' | 'negative' | 'none'
    followUpRequired: boolean
  }
}
```

> **Beginner Note:** If you are new to observability, start with just logging requests and responses. Even basic logs that capture what went in and what came out are enormously valuable for debugging. You can add metrics, tracing, and dashboards incrementally as your application grows.

> **Advanced Note:** Production LLM observability often involves sampling strategies. Logging every request at full fidelity can be expensive and create privacy risks. Consider logging full request/response data for a sample (e.g., 10%) and logging only metadata (latency, tokens, cost, model) for all requests. Use higher sampling rates for errors and anomalies.

---

## Section 2: Structured Logging

### Building an LLM Logger

The foundation of observability is structured logging. Every LLM call should produce a structured log entry that captures the complete context of the interaction.

Start with the data model. What information do you need to capture for every LLM interaction?

```typescript
import { generateText, Output, streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Structured log entry for every LLM interaction
interface LLMLogEntry {
  id: string
  timestamp: string
  level: 'debug' | 'info' | 'warn' | 'error'

  // Request details
  request: {
    model: string
    provider: string
    promptHash: string // Hash of the prompt for grouping without storing PII
    systemPromptVersion: string
    temperature: number
    maxOutputTokens?: number
    toolsAvailable: string[]
  }

  // Response details
  response: {
    outputTokens: number
    inputTokens: number
    totalTokens: number
    finishReason: string
    toolCallsMade: string[]
    responseTimeMs: number
    timeToFirstTokenMs?: number
  }

  // Cost tracking
  cost: {
    inputCostUsd: number
    outputCostUsd: number
    totalCostUsd: number
    cacheHit: boolean
    cachedTokens: number
  }

  // Context
  context: {
    userId: string
    sessionId: string
    requestPath: string
    feature: string
    environment: string
    traceId: string
    spanId: string
    parentSpanId?: string
  }

  // Optional content (controlled by privacy settings)
  content?: {
    promptPreview: string // First N characters only
    responsePreview: string
    fullPrompt?: string // Only in debug mode
    fullResponse?: string // Only in debug mode
  }

  // Error info if applicable
  error?: {
    code: string
    message: string
    retryCount: number
    retryable: boolean
  }
}
```

You will also need a cost calculation helper. Model pricing is per-million tokens, so the formula is straightforward:

```typescript
// (inputTokens / 1_000_000) * pricing.inputPerMillion
```

And a simple ID generator for log entries and trace correlation.

### The LLM Logger Class

The `LLMLogger` class ties everything together. It accepts configuration that controls privacy behavior, stores log entries, and emits them to a destination.

```typescript
// Privacy levels control what content is logged
type PrivacyLevel = 'full' | 'preview' | 'metadata-only'

interface LoggerConfig {
  privacyLevel: PrivacyLevel
  previewLength: number // Characters to include in preview
  environment: string
  serviceName: string
  logDestination: 'console' | 'file' | 'remote'
}
```

Build an `LLMLogger` class with this constructor and a `createEntry` method. The class should:

- Accept a `LoggerConfig` in its constructor and store log entries in an array
- `createEntry(params)` -- takes the raw LLM call data (model, prompt, response, usage, latency, context IDs, optional error) and produces a complete `LLMLogEntry`. It should calculate cost from token counts, determine the log level (`'error'` if an error is present, `'info'` otherwise), hash the prompt for fingerprinting, and add content fields based on the privacy level
- `getLogs()` -- returns a copy of all stored log entries
- `emit(entry)` -- serializes the entry as JSON and writes to the configured destination
- Private helpers: `hashString(str)` for prompt fingerprinting, `extractProvider(model)` to derive the provider name from the model string, `isRetryable(errorCode)` to determine if an error can be retried

Think about the three privacy levels. What should each one include in the `content` field? When `privacyLevel` is `'metadata-only'`, should the `content` field exist at all?

### Using the Logger with Vercel AI SDK

Next, build a `loggedGenerateText` wrapper function. This function wraps `generateText` from the AI SDK and automatically creates a log entry for every call.

```typescript
async function loggedGenerateText(params: {
  model: ReturnType<typeof mistral>
  prompt: string
  system?: string
  maxOutputTokens?: number
  logger: LLMLogger
  userId: string
  sessionId: string
  feature: string
  traceId: string
  spanId: string
  parentSpanId?: string
}): Promise<{ text: string; logEntry: LLMLogEntry }>
```

The wrapper should capture timing (`Date.now()` before and after the call), log successful results with all metadata, and log errors in the catch block before re-throwing. What fields would you populate differently in the error case versus the success case?

### Token Usage Tracking

Build a `TokenUsageTracker` class that aggregates usage data across multiple log entries.

```typescript
interface TokenUsageSummary {
  totalInputTokens: number
  totalOutputTokens: number
  totalCostUsd: number
  requestCount: number
  averageInputTokens: number
  averageOutputTokens: number
  averageCostUsd: number
  byModel: Record<string, { inputTokens: number; outputTokens: number; costUsd: number; requestCount: number }>
  byFeature: Record<string, { inputTokens: number; outputTokens: number; costUsd: number; requestCount: number }>
  byUser: Record<string, { inputTokens: number; outputTokens: number; costUsd: number; requestCount: number }>
}
```

The tracker needs three methods:

- `record(entry)` -- stores a log entry for aggregation
- `getSummary(since?)` -- computes the `TokenUsageSummary` by iterating over entries, aggregating totals, and building the per-model, per-feature, and per-user breakdowns. If `since` is provided, filter entries by timestamp first
- `getTopConsumers(dimension, limit)` -- returns the top N consumers sorted by cost for a given dimension (`'model'`, `'feature'`, or `'user'`)

How would you handle the case where `getSummary` is called with no entries recorded yet?

> **Beginner Note:** Structured logging means logging in a machine-readable format like JSON rather than plain text strings. This allows you to search, filter, and aggregate logs programmatically. Instead of `console.log("Request took 1.2s")`, you log `{ "latencyMs": 1200, "model": "claude-sonnet", "userId": "abc" }`. This is essential for any production system.

---

## Section 3: Tracing

### Distributed Tracing for LLM Pipelines

A single user request to an LLM application often involves multiple steps: retrieval, reranking, generation, tool calls, validation. Tracing connects all these steps into a single trace so you can see the full journey of a request.

The core data model for tracing is the **span** -- one unit of work within a trace:

```typescript
// A span represents one unit of work in a trace
interface Span {
  traceId: string
  spanId: string
  parentSpanId?: string
  operationName: string
  serviceName: string
  startTime: number
  endTime?: number
  durationMs?: number
  status: 'ok' | 'error' | 'timeout'
  attributes: Record<string, string | number | boolean>
  events: Array<{
    name: string
    timestamp: number
    attributes?: Record<string, string | number | boolean>
  }>
}
```

Build a `Tracer` class that manages span creation and lifecycle. It needs:

- `startTrace(operationName)` -- creates a root span with a new trace ID and span ID. This is the top-level operation (e.g., `'rag_query'` or `'agent_loop'`)
- `startSpan(operationName, parentSpan)` -- creates a child span that shares the parent's `traceId` but has its own `spanId`, with `parentSpanId` pointing to the parent
- `endSpan(span, status?)` -- sets `endTime`, calculates `durationMs`, and updates `status`
- `addEvent(span, name, attributes?)` -- records a point-in-time event within a span (e.g., `'cache_hit'`, `'vector_search_complete'`)
- `setAttributes(span, attributes)` -- attaches key-value metadata to a span
- `getTrace(traceId)` -- returns all spans for a trace, sorted by start time
- `printTrace(traceId)` -- prints a visual timeline showing span hierarchy with indentation, status icons, timing offsets, and attributes

What distinguishes `startTrace` from `startSpan`? Think about what makes a root span different from a child span.

### Tracing a RAG Pipeline

Once you have the `Tracer`, use it to instrument a RAG pipeline. Build a `tracedRAGQuery` function with this signature:

```typescript
async function tracedRAGQuery(params: {
  query: string
  userId: string
  tracer: Tracer
}): Promise<{ answer: string; sources: { text: string; score: number }[]; traceId: string }>
```

The function should create a root span for the entire query, then child spans for each stage: `embed_query`, `retrieve_documents`, `rerank_documents`, and `generate_answer`. Each span should record relevant attributes -- for example, the retrieval span should record `documentsRetrieved`, `topScore`, and `bottomScore`.

The visual output should look like a nested timeline:

```
Trace: 1709812345-abc123
================================================================================
[OK] rag_query [+0ms, 2340ms] {"userId":"user-123","queryLength":23,...}
  [OK] embed_query [+2ms, 120ms] {"model":"text-embedding-3-small",...}
  [OK] retrieve_documents [+125ms, 450ms] {"documentsRetrieved":3,...}
    > vector_search_complete [+560ms] {"candidates":100,"returned":3}
  [OK] rerank_documents [+580ms, 30ms] {"inputDocs":3,"outputDocs":3,...}
  [OK] generate_answer [+615ms, 1720ms] {"model":"mistral-small-latest",...}
================================================================================
```

What should happen to the root span if one of the child spans throws an error? How would you record both the success flag and the error message?

### Tracing Agent Loops

Agent loops are especially important to trace because they involve multiple iterations of reasoning and tool calls. Build a `tracedAgentLoop` function that creates a root span for the entire loop, then per-step spans, each containing child spans for the `reason` phase and any `tool_call` phase.

```typescript
async function tracedAgentLoop(params: {
  query: string
  maxSteps: number
  tracer: Tracer
}): Promise<{ result: string; traceId: string; steps: number }>
```

Key attributes to record on the root span: `totalSteps`, `completed` (did the agent finish naturally?), and `hitMaxSteps` (did it exhaust the step limit?). On each reasoning span, record `inputTokens`, `outputTokens`, and `decision` (whether the agent chose to use a tool or give a final answer).

What patterns in the trace would indicate a malfunctioning agent? Think about what the step count, tool call patterns, and decision attributes would look like.

> **Advanced Note:** In production, you would integrate with OpenTelemetry rather than building a custom tracer. The Vercel AI SDK has built-in OpenTelemetry support via the `telemetry` option on `generateText` and `streamText`. The concepts shown here (spans, traces, attributes, events) map directly to OpenTelemetry primitives. The custom implementation helps you understand what OpenTelemetry does under the hood.

---

## Section 4: Metrics

### Defining Key Metrics

Metrics are aggregated numerical measurements that tell you how your application is performing over time. Unlike logs (which capture individual events) and traces (which capture request flows), metrics show trends and distributions.

Build a `MetricsCollector` class that supports three metric types:

```typescript
interface MetricPoint {
  name: string
  value: number
  timestamp: number
  tags: Record<string, string>
  type: 'counter' | 'gauge' | 'histogram'
}
```

The collector needs these methods:

- `increment(name, value?, tags?)` -- records a counter point (total requests, total errors)
- `gauge(name, value, tags?)` -- records a gauge point (active connections, queue depth)
- `histogram(name, value, tags?)` -- records a histogram observation (latency, token count). Store histogram values in a separate bucket array keyed by name so you can compute statistics later
- `percentile(name, p)` -- computes the p-th percentile from stored histogram values. Sort the values, then find the value at index `ceil((p/100) * length) - 1`
- `average(name)` -- computes the average of stored histogram values
- `total(name, tags?)` -- sums all counter values matching the name and optional tags filter
- `latestGauge(name)` -- returns the most recently recorded gauge value for the given name

How would you implement tag-based filtering in the `total` method? Think about how to check that every key-value pair in the filter tags matches the metric point's tags.

### Recording LLM-Specific Metrics

Build an `LLMMetrics` class that wraps `MetricsCollector` with domain-specific recording methods. It should have three recording methods:

**`recordCall(params)`** -- records all metrics from a completed LLM call. Use these metric names:

- `llm.requests.total` (counter) -- increment for every call
- `llm.requests.errors` (counter) -- increment when `success` is false
- `llm.cache.hits` / `llm.cache.misses` (counters)
- `llm.latency.ms`, `llm.time_to_first_token.ms` (histograms)
- `llm.tokens.input`, `llm.tokens.output`, `llm.tokens.total` (histograms)
- `llm.cost.usd` (counter, tagged by model and feature)
- `llm.cost.usd.per_user` (counter, tagged by userId)
- `llm.finish_reason` (counter, tagged by reason)

**`recordRetrieval(params)`** -- records RAG retrieval metrics: `rag.retrieval.latency.ms`, `rag.retrieval.documents`, `rag.retrieval.top_score`, and optionally `rag.rerank.latency.ms`.

**`recordAgentLoop(params)`** -- records agent metrics: `agent.steps.total`, `agent.tool_calls.total`, `agent.latency.total.ms`, `agent.max_steps_reached`, and `agent.completed`.

Finally, add a `getReport()` method that compiles a summary object with sections for requests (total, errors, error rate), latency (p50, p90, p99, average), tokens (averages), cost (total USD), and cache (hits, misses, hit rate).

Why is it important to use `Math.max(denominator, 1)` when computing rates like error rate and cache hit rate?

> **Beginner Note:** The three types of metrics serve different purposes. Counters always go up (total requests, total errors). Gauges go up and down (active connections, queue depth). Histograms capture distributions (latency, token counts). For LLM applications, you will use all three: counters for request/error counts, gauges for concurrent connections, and histograms for latency and token distributions.

---

## Section 5: Debugging LLM Failures

### Common Failure Modes

LLM failures are subtle. The application does not crash -- it returns a plausible-sounding but incorrect response. Here are the most common failure modes and how to diagnose them with observability data.

```typescript
// Failure diagnosis framework
interface FailureDiagnosis {
  failureType: string
  symptoms: string[]
  observabilitySignals: string[]
  rootCause: string
  resolution: string
}
```

The four key failure modes to understand:

**Hallucination** -- The model generates plausible but factually incorrect information, cites nonexistent sources, or invents statistics. The observability signals are low retrieval relevance scores in RAG traces, no retrieval step (model answering from parametric knowledge), high token output without corresponding retrieval context, and quality scores dropping on factual accuracy benchmarks. The root cause is insufficient or irrelevant context, and the resolution is improving retrieval quality and adding grounding instructions.

**Wrong Tool Selection** -- The agent uses the wrong tool, calls the same tool repeatedly without progress, or skips available tools entirely. Look for unexpected `tool_call` spans in the agent trace, step counts approaching the limit, and tool call patterns that do not match the query type. The root cause is usually ambiguous tool descriptions.

**Context Window Overflow** -- The model ignores information from early in the conversation and responses become less coherent. The signals are input token counts approaching the model context limit, monotonically increasing input tokens over turns, and `finish_reason` changing from `'stop'` to `'length'`. The fix is conversation summarization or sliding window history.

**Latency Degradation** -- Response times increasing over time, users experiencing timeouts. Look for p99 latency increasing in metrics, growing token counts, provider rate limiting (429 responses), and retrieval latency spikes.

### Building a Failure Detector

Build a `FailureDetector` class that analyzes recent logs to identify failure patterns automatically.

```typescript
class FailureDetector {
  constructor(
    private logger: LLMLogger,
    private metrics: LLMMetrics
  ) {}

  analyzeRecentFailures(): Array<{
    type: string
    confidence: number
    evidence: string[]
    recommendation: string
  }>
}
```

The `analyzeRecentFailures` method should filter logs to the last hour, then run four checks:

1. **High error rate** -- Count error-level logs and compute the rate. If it exceeds 5%, add a finding with confidence proportional to the rate (capped at 1.0). Collect the unique error codes as evidence.

2. **Latency degradation** -- Compute average response time from recent logs. If it exceeds 5000ms, add a finding. Include the average and max latency as evidence.

3. **Token count growth** -- Split recent token counts into first half and second half. If the second half average exceeds the first half by 50% or more, this suggests context window pressure. What percentage increase would you consider alarming?

4. **Cost anomaly** -- Compute average cost per request. If it exceeds $0.05, flag it with evidence showing both per-request and total costs.

Each check produces a finding with a type name, confidence score, evidence array, and recommendation string. How would you handle the edge case where there are zero recent logs?

### Hallucination Diagnosis with Traces

Build a `diagnoseHallucination` function that uses trace data and an LLM-as-judge call to assess whether a response is grounded in the retrieved context.

```typescript
async function diagnoseHallucination(params: {
  query: string
  response: string
  retrievedDocuments: { text: string; score: number }[]
  tracer: Tracer
}): Promise<{
  isLikelyHallucination: boolean
  confidence: number
  evidence: string[]
}>
```

The function should run three checks, each contributing to a suspicion score:

1. Were any documents retrieved? If not, the model answered from parametric knowledge only (add 0.4 to suspicion).
2. What is the average relevance of retrieved documents? If below 0.7, add 0.3 to suspicion.
3. Use `generateText` with `Output.object()` to ask an LLM judge whether the response is grounded in the context. The schema should include `isGrounded` (boolean), `ungroundedClaims` (string array), and `confidenceScore` (0-1 number).

If the total suspicion score reaches 0.5 or above, classify it as a likely hallucination. Each check should create a child span under a root `hallucination_diagnosis` span, recording attributes like `isGrounded`, `ungroundedClaimCount`, and `confidenceScore`.

What would you do if the groundedness check itself hallucinates? How could you mitigate this risk?

---

## Section 6: Dashboard Design

### Key Metrics to Monitor

A well-designed dashboard surfaces the metrics that matter most for LLM application health. Here is the data structure that captures the essential views:

```typescript
// Dashboard data structure for LLM observability
interface DashboardData {
  overview: {
    totalRequests: number
    errorRate: number
    avgLatencyMs: number
    totalCostUsd: number
    activeSessions: number
    periodStart: string
    periodEnd: string
  }

  latencyPanel: {
    p50Ms: number
    p90Ms: number
    p99Ms: number
    timeToFirstTokenP50Ms: number
    timeToFirstTokenP90Ms: number
    timeSeries: Array<{ timestamp: string; p50: number; p90: number }>
  }

  tokenPanel: {
    totalInputTokens: number
    totalOutputTokens: number
    avgInputPerRequest: number
    avgOutputPerRequest: number
    tokensByModel: Record<string, { input: number; output: number }>
  }

  costPanel: {
    totalCostUsd: number
    costByModel: Record<string, number>
    costByFeature: Record<string, number>
    topUsersByCost: Array<{ userId: string; costUsd: number }>
    costTrend: Array<{ date: string; costUsd: number }>
    projectedDailyCost: number
    budgetRemainingUsd: number
  }

  qualityPanel: {
    cacheHitRate: number
    retrievalRelevanceAvg: number
    agentCompletionRate: number
    avgAgentSteps: number
    finishReasonBreakdown: Record<string, number>
  }

  errorPanel: {
    errorRate: number
    errorsByType: Record<string, number>
    recentErrors: Array<{
      timestamp: string
      errorCode: string
      message: string
      traceId: string
    }>
  }
}
```

### Building a Dashboard Generator

Build a `DashboardGenerator` class that takes the `LLMLogger`, `LLMMetrics`, and `MetricsCollector` as dependencies and produces a `DashboardData` object.

```typescript
class DashboardGenerator {
  constructor(
    private logger: LLMLogger,
    private metrics: LLMMetrics,
    private collector: MetricsCollector
  ) {}

  generate(periodHours: number = 24): DashboardData
}
```

The `generate` method should:

1. Filter logs to the specified time period using `periodStart = now - periodHours * 60 * 60 * 1000`.
2. Iterate over filtered logs once, accumulating breakdowns by model, feature, and user for cost and token panels. Also count errors by type and finish reasons.
3. Compute the overview panel from log aggregates -- total requests, error rate (using `Math.max(logs.length, 1)` to avoid division by zero), average latency, total cost, and unique session count.
4. Populate the latency panel using the `MetricsCollector`'s `percentile` method for p50, p90, and p99.
5. Populate the cost panel, sorting top users by cost descending and limiting to 10. Calculate `projectedDailyCost` by extrapolating from the period.
6. Populate the quality panel using the `MetricsCollector`'s `average` and `total` methods for retrieval relevance, agent completion rate, and agent steps.
7. Populate the error panel with the 10 most recent error logs.

Think about how you would populate the `timeSeries` and `costTrend` arrays in production. What time bucketing approach would you use?

> **Beginner Note:** You do not need to build a custom dashboard from scratch. Tools like Grafana, Datadog, and New Relic can visualize the metrics you collect. The key is emitting the right data -- the visualization tool is secondary. Start by logging structured JSON and viewing it in your log aggregator. Add dedicated dashboards as your application grows.

> **Advanced Note:** Consider implementing a "golden signals" dashboard based on Google's SRE practices: latency, traffic, errors, and saturation. For LLM apps, saturation maps to token budget utilization and rate limit proximity. Add LLM-specific panels for cost, retrieval quality, and agent completion rate.

---

## Section 7: Alerting

### Alert Configuration

Alerts notify your team when metrics cross thresholds that indicate problems. For LLM applications, you need alerts across all five observability pillars.

```typescript
// Alert definition and evaluation
interface AlertRule {
  id: string
  name: string
  description: string
  metric: string
  condition: 'above' | 'below' | 'rate_increase'
  threshold: number
  windowMinutes: number
  severity: 'info' | 'warning' | 'critical'
  notifyChannels: string[]
  cooldownMinutes: number // Minimum time between alerts
}

interface AlertEvent {
  ruleId: string
  ruleName: string
  severity: 'info' | 'warning' | 'critical'
  timestamp: string
  currentValue: number
  threshold: number
  message: string
  traceIds: string[] // Related traces for investigation
}
```

Build an `AlertManager` class with the following behavior:

```typescript
class AlertManager {
  private rules: AlertRule[]
  private activeAlerts: Map<string, AlertEvent>
  private lastAlertTime: Map<string, number>
  private alertHistory: AlertEvent[]

  constructor()
  evaluate(dashboardData: DashboardData): AlertEvent[]
  getActiveAlerts(): AlertEvent[]
  getAlertHistory(): AlertEvent[]
}
```

The constructor should initialize default alert rules for LLM applications. Consider including rules for:

- **High error rate** (`error_rate` above 0.05, critical severity, 15-minute window, 30-minute cooldown)
- **P99 latency degradation** (`latency_p99_ms` above 10000, warning severity)
- **Hourly cost spike** (`hourly_cost_usd` above a dynamic threshold, warning severity)
- **Daily budget warning** at 70% (`daily_cost_usd`, warning) and **critical** at 90%
- **Low cache hit rate** (`cache_hit_rate` below 0.2, info severity)
- **Low retrieval quality** (`retrieval_relevance_avg` below 0.6, warning severity)
- **Agent loop failures** (`agent_max_steps_rate` above 0.1, warning severity)

The `evaluate` method should iterate over all rules, check cooldown (skip if the last alert was within `cooldownMinutes`), extract the current metric value from the `DashboardData`, compare against the threshold based on the condition (`'above'`, `'below'`, or `'rate_increase'`), and create an `AlertEvent` if triggered. Resolved alerts (condition no longer met) should be removed from `activeAlerts`.

You will need a private `getMetricValue(metric, data)` method that maps metric names to fields in `DashboardData`. For example, `'error_rate'` maps to `data.overview.errorRate` and `'agent_max_steps_rate'` maps to `1 - data.qualityPanel.agentCompletionRate`.

Also add a private `notify(alert, channels)` method -- for now, log the alert to console with severity, channel, and message. In production, this would send to Slack, PagerDuty, or email.

Why is cooldown important? What would happen without it during a sustained outage?

---

## Section 8: Privacy Considerations

### What to Log and What Not to Log

Logging LLM interactions creates a tension between debuggability and privacy. Full prompt/response logs are invaluable for debugging but may contain PII, sensitive business data, or information subject to regulatory constraints.

### PII Detection Patterns

Start by defining regex patterns for common PII types. Each pattern needs a `name`, a `pattern` (RegExp with the `g` flag), and a `replacement` string:

```typescript
const piiPatterns: Array<{
  name: string
  pattern: RegExp
  replacement: string
}>
```

Cover at minimum: email addresses (`[EMAIL_REDACTED]`), phone numbers (`[PHONE_REDACTED]`), SSNs (`[SSN_REDACTED]`), credit card numbers (`[CC_REDACTED]`), IP addresses (`[IP_REDACTED]`), and API keys (`[API_KEY_REDACTED]`). How would you write a regex that catches phone numbers with various separators (dots, dashes, spaces) and optional country code?

### Building a Privacy Filter

Build a `PrivacyFilter` class with three methods:

```typescript
class PrivacyFilter {
  constructor(customPatterns?: typeof piiPatterns)

  redact(text: string): { redacted: string; detections: string[] }
  containsPII(text: string): boolean
  getStats(): Record<string, number>
}
```

The `redact` method should apply each pattern in sequence, replacing matches with the corresponding replacement string. Track how many instances of each PII type were found and return both the redacted text and a detections array (e.g., `['email: 2 instance(s)', 'phone: 1 instance(s)']`). Maintain a running count across calls in a `detectionCounts` map.

The `containsPII` method should return `true` if any pattern matches, without modifying the text. Watch out for the regex `g` flag -- `RegExp.test()` advances `lastIndex`, so you need to reset it after each test. Why does this matter?

### Privacy Policies by Environment

Define a `PrivacyPolicy` interface with per-environment settings:

```typescript
interface PrivacyPolicy {
  development: {
    logFullPrompts: boolean
    logFullResponses: boolean
    redactPII: boolean
  }
  staging: {
    logFullPrompts: boolean
    logFullResponses: boolean
    redactPII: boolean
  }
  production: {
    logFullPrompts: boolean
    logFullResponses: boolean
    redactPII: boolean
    maxPreviewLength: number
    retentionDays: number
  }
}
```

The recommended defaults are: development logs everything without redaction (developers need real data), staging logs everything with PII redaction, and production logs only metadata with redacted previews truncated to `maxPreviewLength` characters and a 30-day retention policy.

### Building a Privacy-Aware Logger

Build a `PrivacyAwareLogger` class that combines the `PrivacyFilter` with the `PrivacyPolicy`:

```typescript
class PrivacyAwareLogger {
  constructor(environment: 'development' | 'staging' | 'production', policy?: PrivacyPolicy)

  prepareForLogging(text: string): {
    loggableText: string
    piiDetected: boolean
    detections: string[]
  }

  logPIIWarning(context: string, detections: string[]): void
}
```

The `prepareForLogging` method should: (1) check for PII, (2) apply redaction if the environment policy requires it, (3) truncate the result to `maxPreviewLength` if in production with `logFullPrompts` set to false, appending `... [truncated, N total chars]`. The `logPIIWarning` method should emit a structured JSON warning when detections are present, including the context, detections, and timestamp.

For testing, pass in a string like `'My email is john@example.com and my phone is 555-123-4567'` and verify the production output is truncated and redacted.

> **Beginner Note:** The simplest privacy-safe approach is to never log the full prompt or response in production. Log only metadata: model, token counts, latency, cost, user ID, and a hash of the prompt for grouping. If you need to debug a specific request, use the trace ID to look it up in a separate, access-controlled debug log with a short retention period.

> **Advanced Note:** Consider implementing data residency controls for multi-region deployments. Logs containing user data may need to stay in the same region as the user (GDPR, data sovereignty laws). Use log routing to direct logs to region-specific storage. Also consider implementing a "right to be forgotten" mechanism that can purge all logs associated with a specific user ID on request.

> **Local Alternative (Ollama):** All observability patterns (structured logging, tracing, metrics) work identically with `ollama('qwen3.5')`. Logging and tracing are application-level concerns independent of the model provider. In fact, observability is more important with local models -- you need to monitor inference speed, GPU utilization, and memory usage in addition to the standard LLM metrics.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: OpenTelemetry for LLM Apps

OpenTelemetry (OTel) is the industry standard for distributed tracing. For LLM applications, OTel provides structured traces that span the full pipeline: user input, LLM call, tool execution, and response generation.

Each operation becomes a **span** with attributes:

- `llm.model` — which model handled the request
- `llm.tokens.input` / `llm.tokens.output` — token counts
- `llm.duration_ms` — how long the call took
- `tool.name` — which tool was invoked (for tool call spans)

Spans nest naturally: an agent turn span contains an LLM call span, which contains tool execution spans. This hierarchy lets you see exactly where time and tokens are spent.

**Pattern:** Create a root span for each user message. Within it, create child spans for each LLM call and tool execution. Attach token counts and latency as span attributes. Export to a tracing backend (or log as JSON for local development):

```ts
const span = tracer.startSpan('llm.generate', { attributes: { 'llm.model': model.modelId } })
const result = await generateText({ model, prompt })
span.setAttribute('llm.tokens.input', result.usage.inputTokens)
span.end()
```

## Section 10: Context Window Monitoring

The context window is a finite resource that deserves the same monitoring as memory or CPU. A context monitor tracks:

- **Current usage** — how many tokens are consumed by system prompt, conversation history, tool definitions, and tool results
- **Usage over time** — how the window fills up as the conversation progresses
- **Compaction events** — when compaction was triggered and how much space it freed
- **Threshold alerts** — warnings when usage exceeds 60%, 80%, or 95% of the window

**Key insight for LLM apps:** Memory monitoring is not just about heap — it is about context window usage too. Both are finite resources that need monitoring. A context window filling up silently causes degraded responses (the model loses important context) long before it causes an error.

**Pattern:** After each LLM call, calculate the token count of each message category and log it as a structured metric. Trigger compaction proactively when usage crosses a threshold rather than waiting for the window to overflow.

## Section 11: Pipeline Profiling

LLM pipelines have multiple stages, and the bottleneck is not always the LLM call. A pipeline profiler instruments each step with timing:

1. **Embedding generation** — How long does it take to embed the query?
2. **Vector search** — How long does retrieval take? How many documents are returned?
3. **Context assembly** — How long to format retrieved documents into a prompt?
4. **LLM generation** — Time to first token (TTFT) and total generation time.
5. **Post-processing** — Output parsing, validation, guardrail checks.

Profile your pipeline with real queries and identify the bottleneck. Often it is not the LLM call — slow embedding models or unindexed vector searches can dominate total latency.

**Pattern:** Wrap each pipeline stage in a timed span (using OTel or a simple timer). Log the duration of each stage and compute the percentage of total time each stage consumes. The stage taking the most time is where optimization effort should focus.

## Section 12: Enhanced Structured Logging

Upgrade from basic logging to production-grade structured logs. Every log entry should be a JSON object with:

- **Trace ID** — Links the log entry to a distributed trace, enabling correlation across services.
- **Conversation ID** — Which conversation this entry belongs to.
- **Turn number** — Which turn within the conversation.
- **Token counts** — Input and output tokens for the associated LLM call.
- **Severity** — `debug`, `info`, `warn`, `error` levels.
- **Timestamp** — ISO 8601 format for consistent parsing.

```ts
logger.info({
  traceId,
  conversationId,
  turn: 5,
  model: 'mistral-large-latest',
  tokens: { input: 4200, output: 380 },
  latencyMs: 1250,
  event: 'llm.response',
})
```

Structured JSON logs are machine-parseable, enabling automated alerting, dashboarding, and anomaly detection. Plain text logs require regex parsing and break when the format changes.

## Section 13: Context Visualization

Make the abstract "200K token window" tangible by visualizing what occupies it. A context visualizer shows the composition of the context window as a simple bar or table:

```
Context Window Usage (42,000 / 200,000 tokens - 21%)
[████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 21%
  System prompt:     18,200 tokens (43%)
  Tool definitions:   4,100 tokens (10%)
  Conversation:      12,500 tokens (30%)
  Tool results:       7,200 tokens (17%)
```

This visualization connects to every earlier module: you can see your system prompt size (Module 2), conversation history (Module 4), tool definitions (Module 6), and RAG context (Module 10) all competing for space.

## Section 14: LSP Diagnostics as Observability Signal

(See Module 10 Section 11 for LSP background.)

Language Server Protocol (LSP) diagnostics provide a continuous, zero-cost quality signal for generated code. After any code generation or modification, TypeScript compiler diagnostics reveal type errors, missing imports, and unused variables immediately — without running a test suite or paying for an LLM-as-judge call.

**Pattern:** Run `tsc --noEmit` (or use a TypeScript language server) after each code modification and collect the diagnostics. Track error and warning counts over time alongside your other observability metrics (token usage, latency, cost). A spike in diagnostics after a code generation step indicates the model produced low-quality code.

This is an always-on quality signal: unlike tests (which must be run explicitly) or LLM-as-judge (which costs tokens), LSP diagnostics are free and immediate after initial setup.

## Section 15: Session Sharing for Debugging

Agent conversations are shareable debug artifacts. When an agent produces a bad result, the team can review the exact sequence of decisions — prompts, tool calls, results, and responses — rather than trying to reproduce the issue.

**Pattern:** Implement a session export that serializes the full interaction into a reviewable format:

- All messages (system, user, assistant, tool calls, tool results)
- Token counts and latency for each LLM call
- Tool invocations with parameters and return values
- Timestamps for each event

The exported session is a JSON file that teammates can inspect. This transforms agent interactions from ephemeral to auditable. The conversation history IS the debug log.

---

## Summary

In this module, you learned:

1. **Why LLM observability is different:** Non-deterministic outputs, silent failures like hallucination, and multi-step pipelines make traditional monitoring insufficient for LLM applications.
2. **Structured logging:** How to build an LLM logger that captures request/response pairs, token usage, latency, and metadata in a queryable format.
3. **Distributed tracing:** How to trace requests through RAG pipelines, agent loops, and multi-step workflows using spans and trace IDs to understand where time and tokens are spent.
4. **Key metrics:** Defining and collecting latency (TTFT, total), token usage, error rates, cost per request, and quality scores as the foundation for production monitoring.
5. **Debugging LLM failures:** Diagnosing hallucinations, wrong tool choices, context window issues, and quality degradation using traces and logged prompt/response pairs.
6. **Dashboard design:** Building dashboards that surface actionable insights about application health, cost trends, quality distributions, and user patterns.
7. **Alerting:** Configuring alerts for cost spikes, error rate increases, latency degradation, and quality drops that require immediate attention.
8. **Privacy considerations:** Handling PII in logs, implementing field-level redaction, and designing logging policies that balance debugging needs with data protection requirements.
9. **OpenTelemetry for LLM apps:** Using OTel spans with LLM-specific attributes (model, tokens, latency) to build hierarchical traces across agent turns, LLM calls, and tool executions.
10. **Context window monitoring:** Tracking token usage across message categories (system prompt, history, tool results) and triggering proactive compaction at usage thresholds.
11. **Pipeline profiling:** Instrumenting each pipeline stage (embedding, retrieval, generation, post-processing) to identify the actual bottleneck, which is often not the LLM call.
12. **Enhanced structured logging:** Upgrading to JSON log entries with trace ID, conversation ID, turn number, token counts, and severity for machine-parseable alerting and dashboarding.
13. **Context visualization:** Rendering context window composition as a bar or table so developers can see how system prompt, tools, history, and results compete for space.
14. **LSP diagnostics as observability signal:** Using compiler diagnostics as a free, always-on quality signal for generated code alongside traditional metrics.
15. **Session sharing for debugging:** Exporting full agent interactions (messages, tool calls, token counts, timestamps) as shareable JSON artifacts for team review.

In Module 24, you will learn how to deploy LLM applications to production with authentication, rate limiting, streaming endpoints, and provider failover.

---

## Quiz

**Question 1:** Why is traditional error monitoring (HTTP status codes, exception tracking) insufficient for LLM applications?

A) LLM APIs never return errors
B) LLM calls can return HTTP 200 with incorrect, hallucinated, or harmful content
C) Error monitoring is too expensive for LLM applications
D) LLM applications do not use HTTP

**Answer: B** -- LLM API calls almost always "succeed" technically (return 200 OK with generated text), even when the content is factually wrong, hallucinated, or harmful. Traditional monitoring sees every request as successful because there is no error code. LLM observability must go beyond status codes to assess response quality, groundedness, safety, and cost.

---

**Question 2:** What is the primary purpose of distributed tracing in a RAG pipeline?

A) To make the pipeline run faster
B) To reduce token costs
C) To connect retrieval, reranking, and generation steps into a single request flow for debugging
D) To cache intermediate results

**Answer: C** -- Distributed tracing connects all the steps of a RAG pipeline (embedding, retrieval, reranking, generation) into a single trace with parent-child span relationships. This allows you to see the full journey of a request, identify which step is slow or failing, and understand how retrieval quality affects generation quality. Without tracing, each step is logged independently and you cannot correlate them.

---

**Question 3:** Which metric type is most appropriate for tracking LLM response latency?

A) Counter -- it always goes up
B) Gauge -- it represents a current value
C) Histogram -- it captures the distribution of values
D) Timer -- it measures elapsed time

**Answer: C** -- Histograms capture the full distribution of latency values, allowing you to compute percentiles (p50, p90, p99) that are essential for understanding latency characteristics. A counter would only tell you total time (not per-request). A gauge would only show the latest value (not the distribution). Percentiles from histograms reveal whether most requests are fast with a few slow outliers (long tail) or whether latency is uniformly high.

---

**Question 4:** When diagnosing a suspected hallucination, which observability signal is most informative?

A) High latency on the generation step
B) Low retrieval relevance scores combined with high confidence in the response
C) High token count in the output
D) Cache miss on the request

**Answer: B** -- Hallucinations are most likely when the retrieval step returns low-relevance documents (or no documents at all) but the model still generates a confident, detailed response. This combination indicates the model is answering from its parametric knowledge rather than the retrieved context. High latency, high token count, and cache misses are not strong indicators of hallucination.

---

**Question 5:** What is the recommended approach for logging prompts and responses in production?

A) Log everything in full -- debugging is more important than privacy
B) Never log any content -- privacy is absolute
C) Log metadata (tokens, latency, cost, hashes) for all requests, with PII-redacted previews and access-controlled full logs for a sample
D) Only log error responses in full

**Answer: C** -- The balanced approach logs metadata for every request (essential for metrics and alerting) while providing redacted content previews for debugging. Full prompt/response content is logged only for a sample of requests, with PII redacted, in an access-controlled store with a retention policy. This gives you enough data to debug issues without creating a liability from storing sensitive user data at scale.

---

**Question 6 (Medium):** A pipeline profiler shows that embedding generation takes 200ms, vector search takes 800ms, context assembly takes 10ms, LLM generation takes 500ms, and post-processing takes 15ms. Where should optimization effort focus?

A) LLM generation, because it is the most expensive operation in terms of tokens
B) Vector search, because it is the bottleneck at 800ms — more than the LLM call itself
C) Context assembly, because it is the fastest step and could be doing more work
D) Post-processing, because it runs last and delays the final response

**Answer: B** -- Pipeline profiling reveals that the bottleneck is not always the LLM call. In this case, vector search at 800ms dominates total latency. Optimizing it (adding indexes, reducing the search space, caching frequent queries) would have more impact than optimizing the LLM call. Without profiling, most developers would assume the LLM is the bottleneck and miss the real optimization opportunity.

---

**Question 7 (Hard):** A context window monitor shows: system prompt 43%, tool definitions 10%, conversation history 30%, tool results 17%. The total is 42,000 out of 200,000 tokens. A user reports degraded response quality on long conversations. What is the most likely cause and how does context visualization help diagnose it?

A) The model is too slow — context visualization cannot diagnose quality issues
B) As the conversation grows, history will crowd out tool results and the model will lose access to retrieved information needed for accurate responses
C) The system prompt is too large and should be shortened immediately
D) The tool definitions are taking too much space

**Answer: B** -- Context visualization makes the window allocation concrete. At 42K tokens the system has room, but as conversation history grows, it will consume an increasing percentage. Eventually, tool results (retrieved documents, file contents) must be truncated or dropped to fit within the window. When the model loses access to this retrieved context, response quality degrades because the model falls back to parametric knowledge instead of grounded facts. The visualization helps identify exactly when this crowding occurs and which category to compact first.

---

## Exercises

### Exercise 1: Add Observability to a RAG Pipeline

Build a fully observable RAG pipeline that produces structured logs, traces, and metrics for every query.

**Specification:**

1. Implement the `LLMLogger`, `Tracer`, and `LLMMetrics` classes as shown in this module.

2. Build a RAG pipeline with these traced steps:
   - Query embedding (track embedding model, dimensions, latency)
   - Document retrieval (track documents returned, relevance scores, latency)
   - Reranking (track rerank model, input/output doc counts, latency)
   - Answer generation (track model, tokens, latency, cost)
   - Optional: Hallucination check (track groundedness score)

3. Run 10 sample queries through the pipeline, each producing:
   - A structured log entry with full metadata
   - A trace showing all spans with timing
   - Metrics recorded to the collector

4. After all queries, generate:
   - A metrics summary report (avg latency, total cost, cache hit rate)
   - A printed trace for the slowest query
   - A failure analysis identifying any anomalies

**Expected output:** Console output showing structured logs for each request, a visual trace of the slowest query, and a metrics summary report with per-model and per-feature breakdowns.

### Exercise 2: Build a Simple Metrics Dashboard

Build a metrics dashboard generator that produces a JSON report suitable for rendering in a monitoring UI.

**Specification:**

1. Implement the `DashboardGenerator` class from Section 6.

2. Simulate 50 requests with realistic variation:
   - Mix of models (haiku, sonnet, opus)
   - Mix of features (question-answering, summarization, code-gen)
   - Mix of users (5 different user IDs)
   - Include 3-5 error requests
   - Include some cached responses

3. Implement the `AlertManager` from Section 7 with all default rules.

4. Generate the dashboard data and run all alert rules against it.

5. Print:
   - The full dashboard JSON
   - Any triggered alerts with severity, message, and related trace IDs
   - Top 3 cost consumers by user, model, and feature

**Expected output:** A complete dashboard JSON showing all panels (overview, latency, tokens, cost, quality, errors) plus any active alerts with actionable messages.

### Exercise 3: OpenTelemetry Instrumentation

Instrument an LLM pipeline with OpenTelemetry-style spans to produce structured traces.

**Specification:**

1. Create a simple `Tracer` class that supports:
   - `startSpan(name, attributes?)` — returns a `Span` object with `setAttribute`, `addEvent`, and `end` methods
   - Nested spans (child spans reference their parent via `parentSpanId`)
   - Automatic duration calculation (start time to end time)
   - Export as a JSON array of completed spans

2. Instrument a pipeline that performs: user message processing, LLM call (via `generateText`), tool execution (simulate with a delay), and response formatting. Each step should be a span with relevant attributes (`llm.model`, `llm.tokens.input`, `llm.tokens.output`, `tool.name`, `duration_ms`).

3. Run 5 queries through the instrumented pipeline. For each query, export the trace and verify:
   - The root span covers the full request duration
   - Child spans are properly nested
   - Token counts and model name are recorded as attributes
   - The sum of child span durations approximately equals the parent span duration

4. Print a visual trace for the slowest query showing the span hierarchy with indentation and timing.

**Create:** `src/observability/exercises/otel-instrumentation.ts`

**Expected output:** A visual trace tree for each query showing span names, durations, and key attributes, plus a summary of total tokens and cost across all queries.

### Exercise 4: Context Window Monitor

Build a real-time context window monitor that tracks usage across message categories and alerts on thresholds.

**Specification:**

1. Create a `ContextMonitor` class that tracks token usage by category:
   - System prompt tokens
   - Tool definition tokens
   - Conversation history tokens
   - Tool result tokens
   - Remaining capacity

2. Implement `update(messages)` that takes the current message array, estimates tokens per category, and records a usage snapshot with a timestamp.

3. Implement threshold alerts:
   - `warning` at 60% total usage
   - `critical` at 80% total usage
   - `overflow` at 95% total usage
   - Per-category alerts when tool results exceed 30% of the window or conversation history exceeds 50%

4. Implement `visualize()` that prints an ASCII bar chart showing context window composition (similar to the example in Section 13).

5. Simulate a 20-turn conversation where each turn adds messages and tool results. Show how the context window fills over time. Trigger at least one compaction event when usage exceeds 80%.

**Create:** `src/observability/exercises/context-monitor.ts`

**Expected output:** An ASCII visualization after each turn showing context usage, alerts when thresholds are crossed, and a compaction event that reduces usage.

### Exercise 5: Pipeline Profiler

Build a profiler that instruments each stage of a RAG pipeline and identifies the bottleneck.

**Specification:**

1. Create a `PipelineProfiler` class with:
   - `stage(name, fn)` — wraps an async function in timing instrumentation, returns the function's result
   - `getReport()` — returns timing data for all stages: name, duration, percentage of total time
   - `reset()` — clears all recorded data

2. Instrument a simulated RAG pipeline with these stages:
   - `embed_query` — Simulate embedding generation (50-200ms)
   - `vector_search` — Simulate vector DB lookup (100-500ms)
   - `rerank` — Simulate reranking (30-100ms)
   - `assemble_context` — Simulate prompt construction (5-20ms)
   - `llm_generate` — Make an actual `generateText` call

3. Run the pipeline 5 times and collect profiling data for each run.

4. Generate a profiling report showing:
   - Average duration per stage across all runs
   - Percentage of total time per stage
   - The identified bottleneck (stage with highest average percentage)
   - A recommendation (e.g., "vector_search takes 45% of pipeline time — consider adding an index")

**Create:** `src/observability/exercises/pipeline-profiler.ts`

**Expected output:** A profiling report table showing stage timings, percentages, and a bottleneck recommendation.
