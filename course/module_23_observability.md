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

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

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
async function demonstrateObservabilityGap(): Promise<void> {
  const startTime = Date.now()

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: 'What year was the Treaty of Tordesillas signed?',
  })

  const latencyMs = Date.now() - startTime

  // Traditional monitoring sees: 200 OK, 147 tokens, 1.2s latency
  // All green. Ship it.
  console.log('Status: Success')
  console.log(`Latency: ${latencyMs}ms`)
  console.log(`Tokens: ${result.usage?.inputTokens} in, ${result.usage?.outputTokens} out`)

  // But was the answer actually correct? Was it the Treaty of Tordesillas (1494)?
  // Did the model confuse it with another treaty?
  // Traditional monitoring has no idea.
  console.log(`Response: ${result.text}`)
}
```

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

// Pricing lookup for cost calculation
const modelPricing: Record<string, { inputPerMillion: number; outputPerMillion: number }> = {
  'mistral-small-latest': { inputPerMillion: 3.0, outputPerMillion: 15.0 },
  'mistral-small-latest': { inputPerMillion: 0.8, outputPerMillion: 4.0 },
  'claude-opus-4-20250514': { inputPerMillion: 15.0, outputPerMillion: 75.0 },
}

function calculateCost(
  model: string,
  inputTokens: number,
  outputTokens: number
): { inputCost: number; outputCost: number; totalCost: number } {
  const pricing = modelPricing[model] || {
    inputPerMillion: 3.0,
    outputPerMillion: 15.0,
  }
  const inputCost = (inputTokens / 1_000_000) * pricing.inputPerMillion
  const outputCost = (outputTokens / 1_000_000) * pricing.outputPerMillion
  return { inputCost, outputCost, totalCost: inputCost + outputCost }
}

// Generate unique IDs
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 10)}`
}
```

### The LLM Logger Class

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

class LLMLogger {
  private config: LoggerConfig
  private logs: LLMLogEntry[] = []

  constructor(config: LoggerConfig) {
    this.config = config
  }

  // Create a log entry from an LLM call result
  createEntry(params: {
    model: string
    prompt: string
    systemPrompt?: string
    response: string
    usage: { inputTokens: number; outputTokens: number }
    finishReason: string
    latencyMs: number
    timeToFirstTokenMs?: number
    toolCalls?: string[]
    cacheHit?: boolean
    cachedTokens?: number
    userId: string
    sessionId: string
    feature: string
    traceId: string
    spanId: string
    parentSpanId?: string
    error?: { code: string; message: string; retryCount: number }
  }): LLMLogEntry {
    const cost = calculateCost(params.model, params.usage.inputTokens, params.usage.outputTokens)

    const entry: LLMLogEntry = {
      id: generateId(),
      timestamp: new Date().toISOString(),
      level: params.error ? 'error' : 'info',
      request: {
        model: params.model,
        provider: this.extractProvider(params.model),
        promptHash: this.hashString(params.prompt),
        systemPromptVersion: params.systemPrompt ? this.hashString(params.systemPrompt).substring(0, 8) : 'none',
        temperature: 0,
        maxOutputTokens: undefined,
        toolsAvailable: params.toolCalls || [],
      },
      response: {
        outputTokens: params.usage.outputTokens,
        inputTokens: params.usage.inputTokens,
        totalTokens: params.usage.inputTokens + params.usage.outputTokens,
        finishReason: params.finishReason,
        toolCallsMade: params.toolCalls || [],
        responseTimeMs: params.latencyMs,
        timeToFirstTokenMs: params.timeToFirstTokenMs,
      },
      cost: {
        inputCostUsd: cost.inputCost,
        outputCostUsd: cost.outputCost,
        totalCostUsd: cost.totalCost,
        cacheHit: params.cacheHit || false,
        cachedTokens: params.cachedTokens || 0,
      },
      context: {
        userId: params.userId,
        sessionId: params.sessionId,
        requestPath: '',
        feature: params.feature,
        environment: this.config.environment,
        traceId: params.traceId,
        spanId: params.spanId,
        parentSpanId: params.parentSpanId,
      },
    }

    // Add content based on privacy level
    if (this.config.privacyLevel === 'full') {
      entry.content = {
        promptPreview: params.prompt.substring(0, this.config.previewLength),
        responsePreview: params.response.substring(0, this.config.previewLength),
        fullPrompt: params.prompt,
        fullResponse: params.response,
      }
    } else if (this.config.privacyLevel === 'preview') {
      entry.content = {
        promptPreview: params.prompt.substring(0, this.config.previewLength),
        responsePreview: params.response.substring(0, this.config.previewLength),
      }
    }
    // metadata-only: no content field at all

    if (params.error) {
      entry.error = {
        code: params.error.code,
        message: params.error.message,
        retryCount: params.error.retryCount,
        retryable: this.isRetryable(params.error.code),
      }
    }

    this.logs.push(entry)
    this.emit(entry)
    return entry
  }

  // Emit log to configured destination
  private emit(entry: LLMLogEntry): void {
    const serialized = JSON.stringify(entry)
    if (this.config.logDestination === 'console') {
      console.log(serialized)
    }
    // In production: send to your log aggregation service
    // e.g., Datadog, Elasticsearch, CloudWatch, etc.
  }

  // Get all logs (for testing/local development)
  getLogs(): LLMLogEntry[] {
    return [...this.logs]
  }

  // Simple hash for prompt fingerprinting
  private hashString(str: string): string {
    let hash = 0
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i)
      hash = (hash << 5) - hash + char
      hash = hash & hash // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36)
  }

  private extractProvider(model: string): string {
    if (model.includes('mistral')) return 'mistral'
    if (model.includes('claude')) return 'anthropic'
    if (model.includes('gpt')) return 'openai'
    if (model.includes('gemini')) return 'google'
    return 'unknown'
  }

  private isRetryable(errorCode: string): boolean {
    const retryableCodes = ['rate_limit', 'overloaded', 'timeout', 'internal_error']
    return retryableCodes.includes(errorCode)
  }
}
```

### Using the Logger with Vercel AI SDK

```typescript
// Wrapper that adds automatic logging to any generateText call
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
}): Promise<{ text: string; logEntry: LLMLogEntry }> {
  const startTime = Date.now()

  try {
    const result = await generateText({
      model: params.model,
      prompt: params.prompt,
      system: params.system,
      maxOutputTokens: params.maxOutputTokens,
    })

    const latencyMs = Date.now() - startTime

    const logEntry = params.logger.createEntry({
      model: 'mistral-small-latest',
      prompt: params.prompt,
      systemPrompt: params.system,
      response: result.text,
      usage: {
        inputTokens: result.usage?.inputTokens || 0,
        outputTokens: result.usage?.outputTokens || 0,
      },
      finishReason: result.finishReason || 'stop',
      latencyMs,
      userId: params.userId,
      sessionId: params.sessionId,
      feature: params.feature,
      traceId: params.traceId,
      spanId: params.spanId,
      parentSpanId: params.parentSpanId,
    })

    return { text: result.text, logEntry }
  } catch (error) {
    const latencyMs = Date.now() - startTime

    const logEntry = params.logger.createEntry({
      model: 'mistral-small-latest',
      prompt: params.prompt,
      systemPrompt: params.system,
      response: '',
      usage: { inputTokens: 0, outputTokens: 0 },
      finishReason: 'error',
      latencyMs,
      userId: params.userId,
      sessionId: params.sessionId,
      feature: params.feature,
      traceId: params.traceId,
      spanId: params.spanId,
      parentSpanId: params.parentSpanId,
      error: {
        code: (error as any).code || 'unknown',
        message: (error as Error).message,
        retryCount: 0,
      },
    })

    throw error
  }
}

// Usage example
const logger = new LLMLogger({
  privacyLevel: 'preview',
  previewLength: 200,
  environment: 'production',
  serviceName: 'research-assistant',
  logDestination: 'console',
})

// Every call is automatically logged with full context
const { text, logEntry } = await loggedGenerateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Explain the CAP theorem in distributed systems.',
  system: 'You are a technical instructor. Be concise and precise.',
  logger,
  userId: 'user-123',
  sessionId: 'sess-abc',
  feature: 'question-answering',
  traceId: 'trace-xyz',
  spanId: 'span-001',
})

console.log(`Logged: ${logEntry.id}`)
console.log(`Cost: $${logEntry.cost.totalCostUsd.toFixed(6)}`)
console.log(`Latency: ${logEntry.response.responseTimeMs}ms`)
```

### Token Usage Tracking

```typescript
// Token usage tracker that aggregates across requests
interface TokenUsageSummary {
  totalInputTokens: number
  totalOutputTokens: number
  totalCostUsd: number
  requestCount: number
  averageInputTokens: number
  averageOutputTokens: number
  averageCostUsd: number
  byModel: Record<
    string,
    {
      inputTokens: number
      outputTokens: number
      costUsd: number
      requestCount: number
    }
  >
  byFeature: Record<
    string,
    {
      inputTokens: number
      outputTokens: number
      costUsd: number
      requestCount: number
    }
  >
  byUser: Record<
    string,
    {
      inputTokens: number
      outputTokens: number
      costUsd: number
      requestCount: number
    }
  >
}

class TokenUsageTracker {
  private entries: LLMLogEntry[] = []

  record(entry: LLMLogEntry): void {
    this.entries.push(entry)
  }

  getSummary(since?: Date): TokenUsageSummary {
    const filtered = since ? this.entries.filter(e => new Date(e.timestamp) >= since) : this.entries

    const summary: TokenUsageSummary = {
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalCostUsd: 0,
      requestCount: filtered.length,
      averageInputTokens: 0,
      averageOutputTokens: 0,
      averageCostUsd: 0,
      byModel: {},
      byFeature: {},
      byUser: {},
    }

    for (const entry of filtered) {
      summary.totalInputTokens += entry.response.inputTokens
      summary.totalOutputTokens += entry.response.outputTokens
      summary.totalCostUsd += entry.cost.totalCostUsd

      // Aggregate by model
      const model = entry.request.model
      if (!summary.byModel[model]) {
        summary.byModel[model] = {
          inputTokens: 0,
          outputTokens: 0,
          costUsd: 0,
          requestCount: 0,
        }
      }
      summary.byModel[model].inputTokens += entry.response.inputTokens
      summary.byModel[model].outputTokens += entry.response.outputTokens
      summary.byModel[model].costUsd += entry.cost.totalCostUsd
      summary.byModel[model].requestCount++

      // Aggregate by feature
      const feature = entry.context.feature
      if (!summary.byFeature[feature]) {
        summary.byFeature[feature] = {
          inputTokens: 0,
          outputTokens: 0,
          costUsd: 0,
          requestCount: 0,
        }
      }
      summary.byFeature[feature].inputTokens += entry.response.inputTokens
      summary.byFeature[feature].outputTokens += entry.response.outputTokens
      summary.byFeature[feature].costUsd += entry.cost.totalCostUsd
      summary.byFeature[feature].requestCount++

      // Aggregate by user
      const userId = entry.context.userId
      if (!summary.byUser[userId]) {
        summary.byUser[userId] = {
          inputTokens: 0,
          outputTokens: 0,
          costUsd: 0,
          requestCount: 0,
        }
      }
      summary.byUser[userId].inputTokens += entry.response.inputTokens
      summary.byUser[userId].outputTokens += entry.response.outputTokens
      summary.byUser[userId].costUsd += entry.cost.totalCostUsd
      summary.byUser[userId].requestCount++
    }

    if (filtered.length > 0) {
      summary.averageInputTokens = summary.totalInputTokens / filtered.length
      summary.averageOutputTokens = summary.totalOutputTokens / filtered.length
      summary.averageCostUsd = summary.totalCostUsd / filtered.length
    }

    return summary
  }

  // Get top consumers for cost management
  getTopConsumers(
    dimension: 'model' | 'feature' | 'user',
    limit: number = 5
  ): Array<{ key: string; costUsd: number; requestCount: number }> {
    const summary = this.getSummary()
    const data = dimension === 'model' ? summary.byModel : dimension === 'feature' ? summary.byFeature : summary.byUser

    return Object.entries(data)
      .map(([key, value]) => ({
        key,
        costUsd: value.costUsd,
        requestCount: value.requestCount,
      }))
      .sort((a, b) => b.costUsd - a.costUsd)
      .slice(0, limit)
  }
}
```

> **Beginner Note:** Structured logging means logging in a machine-readable format like JSON rather than plain text strings. This allows you to search, filter, and aggregate logs programmatically. Instead of `console.log("Request took 1.2s")`, you log `{ "latencyMs": 1200, "model": "claude-sonnet", "userId": "abc" }`. This is essential for any production system.

---

## Section 3: Tracing

### Distributed Tracing for LLM Pipelines

A single user request to an LLM application often involves multiple steps: retrieval, reranking, generation, tool calls, validation. Tracing connects all these steps into a single trace so you can see the full journey of a request.

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

class Tracer {
  private spans: Map<string, Span> = new Map()
  private serviceName: string

  constructor(serviceName: string) {
    this.serviceName = serviceName
  }

  // Start a new trace (top-level span)
  startTrace(operationName: string): Span {
    const traceId = generateId()
    const spanId = generateId()
    const span: Span = {
      traceId,
      spanId,
      operationName,
      serviceName: this.serviceName,
      startTime: Date.now(),
      status: 'ok',
      attributes: {},
      events: [],
    }
    this.spans.set(spanId, span)
    return span
  }

  // Start a child span within an existing trace
  startSpan(operationName: string, parentSpan: Span): Span {
    const spanId = generateId()
    const span: Span = {
      traceId: parentSpan.traceId,
      spanId,
      parentSpanId: parentSpan.spanId,
      operationName,
      serviceName: this.serviceName,
      startTime: Date.now(),
      status: 'ok',
      attributes: {},
      events: [],
    }
    this.spans.set(spanId, span)
    return span
  }

  // End a span and calculate duration
  endSpan(span: Span, status: 'ok' | 'error' | 'timeout' = 'ok'): void {
    span.endTime = Date.now()
    span.durationMs = span.endTime - span.startTime
    span.status = status
  }

  // Add an event to a span (e.g., "cache_hit", "retrieval_complete")
  addEvent(span: Span, name: string, attributes?: Record<string, string | number | boolean>): void {
    span.events.push({ name, timestamp: Date.now(), attributes })
  }

  // Set attributes on a span
  setAttributes(span: Span, attributes: Record<string, string | number | boolean>): void {
    Object.assign(span.attributes, attributes)
  }

  // Get all spans for a trace
  getTrace(traceId: string): Span[] {
    return Array.from(this.spans.values())
      .filter(s => s.traceId === traceId)
      .sort((a, b) => a.startTime - b.startTime)
  }

  // Print a trace as a visual timeline
  printTrace(traceId: string): void {
    const spans = this.getTrace(traceId)
    if (spans.length === 0) return

    const traceStart = spans[0].startTime
    console.log(`\nTrace: ${traceId}`)
    console.log('='.repeat(80))

    for (const span of spans) {
      const offset = span.startTime - traceStart
      const duration = span.durationMs || 0
      const indent = span.parentSpanId ? '  ' : ''
      const statusIcon = span.status === 'ok' ? '[OK]' : span.status === 'error' ? '[ERR]' : '[TIMEOUT]'

      console.log(
        `${indent}${statusIcon} ${span.operationName} ` +
          `[+${offset}ms, ${duration}ms] ` +
          `${JSON.stringify(span.attributes)}`
      )

      for (const event of span.events) {
        const eventOffset = event.timestamp - traceStart
        console.log(
          `${indent}  > ${event.name} [+${eventOffset}ms] ` +
            `${event.attributes ? JSON.stringify(event.attributes) : ''}`
        )
      }
    }
    console.log('='.repeat(80))
  }
}
```

### Tracing a RAG Pipeline

Here is how tracing works through a complete RAG pipeline, connecting retrieval, reranking, and generation into a single trace:

```typescript
import { generateText, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { openai } from '@ai-sdk/openai'

// Simulated RAG components for demonstration
interface Document {
  id: string
  content: string
  metadata: { source: string; relevanceScore?: number }
}

async function tracedRAGQuery(params: {
  query: string
  userId: string
  tracer: Tracer
}): Promise<{ answer: string; sources: Document[]; traceId: string }> {
  const { query, userId, tracer } = params

  // Start the top-level trace
  const rootSpan = tracer.startTrace('rag_query')
  tracer.setAttributes(rootSpan, {
    userId,
    queryLength: query.length,
  })

  try {
    // Step 1: Embed the query
    const embedSpan = tracer.startSpan('embed_query', rootSpan)
    tracer.setAttributes(embedSpan, { model: 'text-embedding-3-small' })

    const { embedding } = await embed({
      model: openai.embedding('text-embedding-3-small'),
      value: query,
    })

    tracer.setAttributes(embedSpan, {
      embeddingDimensions: embedding.length,
    })
    tracer.endSpan(embedSpan)

    // Step 2: Retrieve documents
    const retrieveSpan = tracer.startSpan('retrieve_documents', rootSpan)

    // Simulate retrieval (in production, this queries your vector store)
    const retrievedDocs: Document[] = [
      {
        id: 'doc-1',
        content: 'The CAP theorem states that...',
        metadata: { source: 'distributed-systems.pdf', relevanceScore: 0.92 },
      },
      {
        id: 'doc-2',
        content: 'In distributed computing, consistency...',
        metadata: { source: 'cs-textbook.pdf', relevanceScore: 0.87 },
      },
      {
        id: 'doc-3',
        content: 'Network partitions in practice...',
        metadata: { source: 'blog-post.md', relevanceScore: 0.78 },
      },
    ]

    tracer.setAttributes(retrieveSpan, {
      documentsRetrieved: retrievedDocs.length,
      topScore: retrievedDocs[0]?.metadata.relevanceScore || 0,
      bottomScore: retrievedDocs[retrievedDocs.length - 1]?.metadata.relevanceScore || 0,
    })
    tracer.addEvent(retrieveSpan, 'vector_search_complete', {
      candidates: 100,
      returned: retrievedDocs.length,
    })
    tracer.endSpan(retrieveSpan)

    // Step 3: Rerank documents
    const rerankSpan = tracer.startSpan('rerank_documents', rootSpan)

    // Simulate reranking
    const rerankedDocs = retrievedDocs.sort(
      (a, b) => (b.metadata.relevanceScore || 0) - (a.metadata.relevanceScore || 0)
    )

    tracer.setAttributes(rerankSpan, {
      inputDocs: retrievedDocs.length,
      outputDocs: rerankedDocs.length,
      rerankModel: 'cross-encoder',
    })
    tracer.endSpan(rerankSpan)

    // Step 4: Generate answer
    const generateSpan = tracer.startSpan('generate_answer', rootSpan)

    const context = rerankedDocs.map(d => d.content).join('\n\n')

    const result = await generateText({
      model: mistral('mistral-small-latest'),
      system: 'Answer the question using only the provided context. Cite sources.',
      prompt: `Context:\n${context}\n\nQuestion: ${query}`,
    })

    tracer.setAttributes(generateSpan, {
      model: 'mistral-small-latest',
      inputTokens: result.usage?.inputTokens || 0,
      outputTokens: result.usage?.outputTokens || 0,
      finishReason: result.finishReason || 'stop',
    })
    tracer.endSpan(generateSpan)

    // End the root trace
    tracer.setAttributes(rootSpan, {
      success: true,
      documentsUsed: rerankedDocs.length,
      totalTokens: (result.usage?.inputTokens || 0) + (result.usage?.outputTokens || 0),
    })
    tracer.endSpan(rootSpan)

    // Print the full trace for debugging
    tracer.printTrace(rootSpan.traceId)

    return {
      answer: result.text,
      sources: rerankedDocs,
      traceId: rootSpan.traceId,
    }
  } catch (error) {
    tracer.setAttributes(rootSpan, {
      success: false,
      errorMessage: (error as Error).message,
    })
    tracer.endSpan(rootSpan, 'error')
    throw error
  }
}

// Usage
const tracer = new Tracer('research-assistant')
const result = await tracedRAGQuery({
  query: 'Explain the CAP theorem',
  userId: 'user-123',
  tracer,
})

// Output:
// Trace: 1709812345-abc123
// ================================================================================
// [OK] rag_query [+0ms, 2340ms] {"userId":"user-123","queryLength":23,...}
//   [OK] embed_query [+2ms, 120ms] {"model":"text-embedding-3-small",...}
//   [OK] retrieve_documents [+125ms, 450ms] {"documentsRetrieved":3,...}
//     > vector_search_complete [+560ms] {"candidates":100,"returned":3}
//   [OK] rerank_documents [+580ms, 30ms] {"inputDocs":3,"outputDocs":3,...}
//   [OK] generate_answer [+615ms, 1720ms] {"model":"mistral-small-latest",...}
// ================================================================================
```

### Tracing Agent Loops

Agent loops are especially important to trace because they involve multiple iterations of reasoning and tool calls:

```typescript
// Trace an agent loop with tool calls
async function tracedAgentLoop(params: {
  query: string
  maxSteps: number
  tracer: Tracer
}): Promise<{ result: string; traceId: string; steps: number }> {
  const { query, maxSteps, tracer } = params

  const rootSpan = tracer.startTrace('agent_loop')
  tracer.setAttributes(rootSpan, {
    query: query.substring(0, 100),
    maxSteps,
  })

  let currentStep = 0
  let agentDone = false
  let finalResult = ''

  while (!agentDone && currentStep < maxSteps) {
    currentStep++
    const stepSpan = tracer.startSpan(`step_${currentStep}`, rootSpan)

    // Reasoning step
    const reasonSpan = tracer.startSpan('reason', stepSpan)

    const reasonResult = await generateText({
      model: mistral('mistral-small-latest'),
      prompt: `Step ${currentStep}: Given the query "${query}", decide what to do next.`,
      system: 'You are a research agent. Decide: use a tool or give final answer.',
    })

    tracer.setAttributes(reasonSpan, {
      inputTokens: reasonResult.usage?.inputTokens || 0,
      outputTokens: reasonResult.usage?.outputTokens || 0,
      decision: reasonResult.text.includes('FINAL') ? 'final_answer' : 'use_tool',
    })
    tracer.endSpan(reasonSpan)

    // Check if the agent decided to give a final answer
    if (reasonResult.text.includes('FINAL')) {
      tracer.addEvent(stepSpan, 'agent_complete', { step: currentStep })
      finalResult = reasonResult.text
      agentDone = true
    } else {
      // Tool call step
      const toolSpan = tracer.startSpan('tool_call', stepSpan)
      tracer.setAttributes(toolSpan, {
        toolName: 'web_search',
        toolInput: query.substring(0, 50),
      })

      // Simulate tool execution
      const toolResult = `Search results for: ${query}`

      tracer.setAttributes(toolSpan, {
        toolOutputLength: toolResult.length,
        toolSuccess: true,
      })
      tracer.endSpan(toolSpan)
    }

    tracer.endSpan(stepSpan)
  }

  tracer.setAttributes(rootSpan, {
    totalSteps: currentStep,
    completed: agentDone,
    hitMaxSteps: !agentDone,
  })
  tracer.endSpan(rootSpan)

  tracer.printTrace(rootSpan.traceId)

  return {
    result: finalResult,
    traceId: rootSpan.traceId,
    steps: currentStep,
  }
}
```

> **Advanced Note:** In production, you would integrate with OpenTelemetry rather than building a custom tracer. The Vercel AI SDK has built-in OpenTelemetry support via the `telemetry` option on `generateText` and `streamText`. The concepts shown here (spans, traces, attributes, events) map directly to OpenTelemetry primitives. The custom implementation helps you understand what OpenTelemetry does under the hood.

---

## Section 4: Metrics

### Defining Key Metrics

Metrics are aggregated numerical measurements that tell you how your application is performing over time. Unlike logs (which capture individual events) and traces (which capture request flows), metrics show trends and distributions.

```typescript
// Core metrics for LLM applications
interface MetricPoint {
  name: string
  value: number
  timestamp: number
  tags: Record<string, string>
  type: 'counter' | 'gauge' | 'histogram'
}

class MetricsCollector {
  private points: MetricPoint[] = []
  private histogramBuckets: Record<string, number[]> = {}

  // Increment a counter (e.g., total requests, total errors)
  increment(name: string, value: number = 1, tags: Record<string, string> = {}): void {
    this.points.push({
      name,
      value,
      timestamp: Date.now(),
      tags,
      type: 'counter',
    })
  }

  // Set a gauge value (e.g., active connections, queue depth)
  gauge(name: string, value: number, tags: Record<string, string> = {}): void {
    this.points.push({
      name,
      value,
      timestamp: Date.now(),
      tags,
      type: 'gauge',
    })
  }

  // Record a histogram observation (e.g., latency, token count)
  histogram(name: string, value: number, tags: Record<string, string> = {}): void {
    this.points.push({
      name,
      value,
      timestamp: Date.now(),
      tags,
      type: 'histogram',
    })

    if (!this.histogramBuckets[name]) {
      this.histogramBuckets[name] = []
    }
    this.histogramBuckets[name].push(value)
  }

  // Calculate percentiles for a histogram metric
  percentile(name: string, p: number): number {
    const values = this.histogramBuckets[name]
    if (!values || values.length === 0) return 0

    const sorted = [...values].sort((a, b) => a - b)
    const index = Math.ceil((p / 100) * sorted.length) - 1
    return sorted[Math.max(0, index)]
  }

  // Get average for a histogram metric
  average(name: string): number {
    const values = this.histogramBuckets[name]
    if (!values || values.length === 0) return 0
    return values.reduce((sum, v) => sum + v, 0) / values.length
  }

  // Get total for a counter metric
  total(name: string, tags?: Record<string, string>): number {
    return this.points
      .filter(p => p.name === name && (!tags || Object.entries(tags).every(([k, v]) => p.tags[k] === v)))
      .reduce((sum, p) => sum + p.value, 0)
  }

  // Get the latest gauge value
  latestGauge(name: string): number {
    const gauges = this.points.filter(p => p.name === name && p.type === 'gauge')
    return gauges.length > 0 ? gauges[gauges.length - 1].value : 0
  }
}
```

### Recording LLM-Specific Metrics

```typescript
// LLM metrics recorder that wraps the generic collector
class LLMMetrics {
  private collector: MetricsCollector

  constructor(collector: MetricsCollector) {
    this.collector = collector
  }

  // Record metrics from a completed LLM call
  recordCall(params: {
    model: string
    feature: string
    userId: string
    latencyMs: number
    timeToFirstTokenMs?: number
    inputTokens: number
    outputTokens: number
    costUsd: number
    success: boolean
    cached: boolean
    finishReason: string
  }): void {
    const tags = {
      model: params.model,
      feature: params.feature,
    }

    // Request counter
    this.collector.increment('llm.requests.total', 1, tags)

    if (!params.success) {
      this.collector.increment('llm.requests.errors', 1, tags)
    }

    if (params.cached) {
      this.collector.increment('llm.cache.hits', 1, tags)
    } else {
      this.collector.increment('llm.cache.misses', 1, tags)
    }

    // Latency histogram
    this.collector.histogram('llm.latency.ms', params.latencyMs, tags)

    if (params.timeToFirstTokenMs !== undefined) {
      this.collector.histogram('llm.time_to_first_token.ms', params.timeToFirstTokenMs, tags)
    }

    // Token usage histograms
    this.collector.histogram('llm.tokens.input', params.inputTokens, tags)
    this.collector.histogram('llm.tokens.output', params.outputTokens, tags)
    this.collector.histogram('llm.tokens.total', params.inputTokens + params.outputTokens, tags)

    // Cost counter
    this.collector.increment('llm.cost.usd', params.costUsd, tags)

    // Per-user cost
    this.collector.increment('llm.cost.usd.per_user', params.costUsd, {
      userId: params.userId,
    })

    // Finish reason counter
    this.collector.increment('llm.finish_reason', 1, {
      ...tags,
      reason: params.finishReason,
    })
  }

  // Record retrieval metrics for RAG pipelines
  recordRetrieval(params: {
    feature: string
    documentsRetrieved: number
    topRelevanceScore: number
    retrievalLatencyMs: number
    rerankLatencyMs?: number
  }): void {
    const tags = { feature: params.feature }

    this.collector.histogram('rag.retrieval.latency.ms', params.retrievalLatencyMs, tags)
    this.collector.histogram('rag.retrieval.documents', params.documentsRetrieved, tags)
    this.collector.histogram('rag.retrieval.top_score', params.topRelevanceScore, tags)

    if (params.rerankLatencyMs !== undefined) {
      this.collector.histogram('rag.rerank.latency.ms', params.rerankLatencyMs, tags)
    }
  }

  // Record agent loop metrics
  recordAgentLoop(params: {
    feature: string
    totalSteps: number
    toolCallCount: number
    totalLatencyMs: number
    completed: boolean
    hitMaxSteps: boolean
  }): void {
    const tags = { feature: params.feature }

    this.collector.histogram('agent.steps.total', params.totalSteps, tags)
    this.collector.histogram('agent.tool_calls.total', params.toolCallCount, tags)
    this.collector.histogram('agent.latency.total.ms', params.totalLatencyMs, tags)

    if (params.hitMaxSteps) {
      this.collector.increment('agent.max_steps_reached', 1, tags)
    }

    this.collector.increment('agent.completed', params.completed ? 1 : 0, tags)
  }

  // Generate a metrics summary report
  getReport(): Record<string, unknown> {
    return {
      requests: {
        total: this.collector.total('llm.requests.total'),
        errors: this.collector.total('llm.requests.errors'),
        errorRate:
          this.collector.total('llm.requests.errors') / Math.max(this.collector.total('llm.requests.total'), 1),
      },
      latency: {
        p50: this.collector.percentile('llm.latency.ms', 50),
        p90: this.collector.percentile('llm.latency.ms', 90),
        p99: this.collector.percentile('llm.latency.ms', 99),
        average: this.collector.average('llm.latency.ms'),
      },
      tokens: {
        averageInput: this.collector.average('llm.tokens.input'),
        averageOutput: this.collector.average('llm.tokens.output'),
        averageTotal: this.collector.average('llm.tokens.total'),
      },
      cost: {
        totalUsd: this.collector.total('llm.cost.usd'),
      },
      cache: {
        hits: this.collector.total('llm.cache.hits'),
        misses: this.collector.total('llm.cache.misses'),
        hitRate:
          this.collector.total('llm.cache.hits') /
          Math.max(this.collector.total('llm.cache.hits') + this.collector.total('llm.cache.misses'), 1),
      },
    }
  }
}

// Usage
const collector = new MetricsCollector()
const metrics = new LLMMetrics(collector)

// Record metrics from LLM calls
metrics.recordCall({
  model: 'mistral-small-latest',
  feature: 'question-answering',
  userId: 'user-123',
  latencyMs: 1200,
  timeToFirstTokenMs: 350,
  inputTokens: 500,
  outputTokens: 200,
  costUsd: 0.0045,
  success: true,
  cached: false,
  finishReason: 'stop',
})

// Get the report
const report = metrics.getReport()
console.log('Metrics Report:', JSON.stringify(report, null, 2))
```

> **Beginner Note:** The three types of metrics serve different purposes. Counters always go up (total requests, total errors). Gauges go up and down (active connections, queue depth). Histograms capture distributions (latency, token counts). For LLM applications, you will use all three: counters for request/error counts, gauges for concurrent connections, and histograms for latency and token distributions.

---

## Section 5: Debugging LLM Failures

### Common Failure Modes

LLM failures are subtle. The application does not crash -- it returns a plausible-sounding but incorrect response. Here are the most common failure modes and how to diagnose them with observability data.

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Failure diagnosis framework
interface FailureDiagnosis {
  failureType: string
  symptoms: string[]
  observabilitySignals: string[]
  rootCause: string
  resolution: string
}

const commonFailures: FailureDiagnosis[] = [
  {
    failureType: 'Hallucination',
    symptoms: [
      'Model generates plausible but factually incorrect information',
      'Model cites sources that do not exist',
      'Model invents statistics or dates',
    ],
    observabilitySignals: [
      'Low retrieval relevance scores in RAG traces',
      'No retrieval step in trace (model answering from parametric knowledge)',
      'High token output without corresponding retrieval context',
      'Quality scores dropping on factual accuracy benchmarks',
    ],
    rootCause:
      'Insufficient or irrelevant context provided, or model relying on training data instead of retrieved documents.',
    resolution: 'Improve retrieval quality, add explicit grounding instructions, implement fact-checking on output.',
  },
  {
    failureType: 'Wrong Tool Selection',
    symptoms: [
      'Agent uses calculator when it should use search',
      'Agent calls the same tool repeatedly without progress',
      'Agent skips available tools and tries to answer from memory',
    ],
    observabilitySignals: [
      'Agent trace shows unexpected tool_call spans',
      'Agent step count approaching maxSteps',
      'Tool call patterns not matching query type',
      'High retry count in agent loop metrics',
    ],
    rootCause:
      'Tool descriptions are ambiguous, or the model cannot distinguish which tool is appropriate for the query.',
    resolution:
      'Improve tool descriptions with explicit use cases and examples. Add few-shot examples of correct tool selection.',
  },
  {
    failureType: 'Context Window Overflow',
    symptoms: [
      'Model ignores information from early in the conversation',
      'Responses become less coherent in long conversations',
      'Truncation warnings in logs',
    ],
    observabilitySignals: [
      'Input token counts approaching model context limit',
      'Monotonically increasing input tokens over conversation turns',
      "finish_reason changing to 'length' instead of 'stop'",
    ],
    rootCause: "Conversation history or retrieved context exceeds the model's effective attention window.",
    resolution:
      'Implement conversation summarization, limit retrieved documents, use sliding window over conversation history.',
  },
  {
    failureType: 'Latency Degradation',
    symptoms: [
      'Response times increasing over time',
      'Users experiencing timeouts',
      'Streaming time-to-first-token increasing',
    ],
    observabilitySignals: [
      'p99 latency increasing in metrics',
      'Token counts growing (longer prompts = slower responses)',
      'Provider rate limiting (429 responses in logs)',
      'Retrieval latency spikes in traces',
    ],
    rootCause: 'Growing prompt sizes, provider rate limiting, cold vector store caches, or increased traffic.',
    resolution:
      'Optimize prompt length, implement request queuing, add caching layers, consider model tier downgrade for simple queries.',
  },
]

// Automated failure detector that analyzes logs and traces
class FailureDetector {
  private logger: LLMLogger
  private metrics: LLMMetrics

  constructor(logger: LLMLogger, metrics: LLMMetrics) {
    this.logger = logger
    this.metrics = metrics
  }

  // Analyze recent logs for failure patterns
  analyzeRecentFailures(): Array<{
    type: string
    confidence: number
    evidence: string[]
    recommendation: string
  }> {
    const logs = this.logger.getLogs()
    const recentLogs = logs.filter(
      l => Date.now() - new Date(l.timestamp).getTime() < 60 * 60 * 1000 // Last hour
    )

    const findings: Array<{
      type: string
      confidence: number
      evidence: string[]
      recommendation: string
    }> = []

    // Check for high error rate
    const errorCount = recentLogs.filter(l => l.level === 'error').length
    const errorRate = errorCount / Math.max(recentLogs.length, 1)
    if (errorRate > 0.05) {
      findings.push({
        type: 'High Error Rate',
        confidence: Math.min(errorRate * 10, 1),
        evidence: [
          `${errorCount} errors in ${recentLogs.length} requests (${(errorRate * 100).toFixed(1)}%)`,
          `Error codes: ${[...new Set(recentLogs.filter(l => l.error).map(l => l.error?.code))].join(', ')}`,
        ],
        recommendation:
          'Check provider status page. If rate limiting, implement backoff. If auth errors, verify API keys.',
      })
    }

    // Check for latency increase
    const latencies = recentLogs.map(l => l.response.responseTimeMs)
    const avgLatency = latencies.reduce((sum, l) => sum + l, 0) / Math.max(latencies.length, 1)
    if (avgLatency > 5000) {
      findings.push({
        type: 'Latency Degradation',
        confidence: 0.8,
        evidence: [
          `Average latency: ${avgLatency.toFixed(0)}ms (threshold: 5000ms)`,
          `Max latency: ${Math.max(...latencies)}ms`,
        ],
        recommendation:
          'Check token counts (may be growing). Check provider status. Consider model routing to faster models.',
      })
    }

    // Check for token count growth (possible context overflow)
    const tokenCounts = recentLogs.map(l => l.response.inputTokens)
    if (tokenCounts.length > 10) {
      const firstHalf = tokenCounts.slice(0, Math.floor(tokenCounts.length / 2))
      const secondHalf = tokenCounts.slice(Math.floor(tokenCounts.length / 2))
      const firstAvg = firstHalf.reduce((s, t) => s + t, 0) / firstHalf.length
      const secondAvg = secondHalf.reduce((s, t) => s + t, 0) / secondHalf.length

      if (secondAvg > firstAvg * 1.5) {
        findings.push({
          type: 'Token Count Growth',
          confidence: 0.7,
          evidence: [
            `Average tokens grew from ${firstAvg.toFixed(0)} to ${secondAvg.toFixed(0)} (${((secondAvg / firstAvg - 1) * 100).toFixed(0)}% increase)`,
            'Possible context window pressure',
          ],
          recommendation:
            'Implement conversation summarization. Check if RAG retrieval is returning too many documents.',
        })
      }
    }

    // Check for cost anomaly
    const totalCost = recentLogs.reduce((sum, l) => sum + l.cost.totalCostUsd, 0)
    const costPerRequest = totalCost / Math.max(recentLogs.length, 1)
    if (costPerRequest > 0.05) {
      findings.push({
        type: 'Cost Anomaly',
        confidence: 0.6,
        evidence: [
          `Average cost per request: $${costPerRequest.toFixed(4)}`,
          `Total cost in last hour: $${totalCost.toFixed(4)}`,
        ],
        recommendation:
          'Check if expensive models are being used unnecessarily. Verify caching is working. Check for runaway agent loops.',
      })
    }

    return findings
  }
}
```

### Hallucination Diagnosis with Traces

```typescript
// Specific hallucination detector using trace data
async function diagnoseHallucination(params: {
  query: string
  response: string
  retrievedDocuments: Document[]
  tracer: Tracer
}): Promise<{
  isLikelyHallucination: boolean
  confidence: number
  evidence: string[]
}> {
  const { query, response, retrievedDocuments, tracer } = params

  const diagSpan = tracer.startTrace('hallucination_diagnosis')
  const evidence: string[] = []
  let suspicionScore = 0

  // Check 1: Were any documents retrieved?
  if (retrievedDocuments.length === 0) {
    evidence.push('No documents were retrieved -- model answered from parametric knowledge only')
    suspicionScore += 0.4
  }

  // Check 2: Are retrieved documents relevant?
  const avgRelevance =
    retrievedDocuments.reduce((sum, d) => sum + (d.metadata.relevanceScore || 0), 0) /
    Math.max(retrievedDocuments.length, 1)
  if (avgRelevance < 0.7) {
    evidence.push(`Low average retrieval relevance: ${avgRelevance.toFixed(2)} (threshold: 0.70)`)
    suspicionScore += 0.3
  }

  // Check 3: Use LLM-as-judge to verify groundedness
  const verifySpan = tracer.startSpan('groundedness_check', diagSpan)

  const verificationResult = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        isGrounded: z.boolean().describe('Whether the response is fully supported by the context'),
        ungroundedClaims: z.array(z.string()).describe('Claims in the response not supported by context'),
        confidenceScore: z.number().min(0).max(1).describe('Confidence that the response is grounded'),
      }),
    }),
    prompt: `Verify if this response is grounded in the provided context.

Context documents:
${retrievedDocuments.map(d => d.content).join('\n---\n')}

Response to verify:
${response}

Check each claim in the response against the context. List any claims not supported by the context.`,
  })

  tracer.setAttributes(verifySpan, {
    isGrounded: verificationResult.output!.isGrounded,
    ungroundedClaimCount: verificationResult.output!.ungroundedClaims.length,
    confidenceScore: verificationResult.output!.confidenceScore,
  })
  tracer.endSpan(verifySpan)

  if (!verificationResult.output!.isGrounded) {
    evidence.push(`LLM judge found ${verificationResult.output!.ungroundedClaims.length} ungrounded claims`)
    for (const claim of verificationResult.output!.ungroundedClaims) {
      evidence.push(`  - Ungrounded: "${claim}"`)
    }
    suspicionScore += 0.3
  }

  tracer.setAttributes(diagSpan, {
    suspicionScore,
    isLikelyHallucination: suspicionScore >= 0.5,
    evidenceCount: evidence.length,
  })
  tracer.endSpan(diagSpan)

  return {
    isLikelyHallucination: suspicionScore >= 0.5,
    confidence: Math.min(suspicionScore, 1),
    evidence,
  }
}
```

---

## Section 6: Dashboard Design

### Key Metrics to Monitor

A well-designed dashboard surfaces the metrics that matter most for LLM application health. Here is a dashboard data structure that captures the essential views:

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

// Dashboard data generator
class DashboardGenerator {
  private logger: LLMLogger
  private metrics: LLMMetrics
  private collector: MetricsCollector

  constructor(logger: LLMLogger, metrics: LLMMetrics, collector: MetricsCollector) {
    this.logger = logger
    this.metrics = metrics
    this.collector = collector
  }

  generate(periodHours: number = 24): DashboardData {
    const now = Date.now()
    const periodStart = now - periodHours * 60 * 60 * 1000
    const logs = this.logger.getLogs().filter(l => new Date(l.timestamp).getTime() >= periodStart)

    const report = this.metrics.getReport() as any

    // Calculate cost breakdown by model
    const costByModel: Record<string, number> = {}
    const costByFeature: Record<string, number> = {}
    const userCosts: Record<string, number> = {}
    const tokensByModel: Record<string, { input: number; output: number }> = {}
    const errorsByType: Record<string, number> = {}
    const finishReasons: Record<string, number> = {}

    for (const log of logs) {
      // Cost by model
      const model = log.request.model
      costByModel[model] = (costByModel[model] || 0) + log.cost.totalCostUsd

      // Cost by feature
      const feature = log.context.feature
      costByFeature[feature] = (costByFeature[feature] || 0) + log.cost.totalCostUsd

      // User costs
      const userId = log.context.userId
      userCosts[userId] = (userCosts[userId] || 0) + log.cost.totalCostUsd

      // Tokens by model
      if (!tokensByModel[model]) {
        tokensByModel[model] = { input: 0, output: 0 }
      }
      tokensByModel[model].input += log.response.inputTokens
      tokensByModel[model].output += log.response.outputTokens

      // Errors
      if (log.error) {
        const code = log.error.code
        errorsByType[code] = (errorsByType[code] || 0) + 1
      }

      // Finish reasons
      const reason = log.response.finishReason
      finishReasons[reason] = (finishReasons[reason] || 0) + 1
    }

    const totalCost = logs.reduce((sum, l) => sum + l.cost.totalCostUsd, 0)
    const latencies = logs.map(l => l.response.responseTimeMs)
    const errorLogs = logs.filter(l => l.level === 'error')

    return {
      overview: {
        totalRequests: logs.length,
        errorRate: errorLogs.length / Math.max(logs.length, 1),
        avgLatencyMs: latencies.reduce((s, l) => s + l, 0) / Math.max(latencies.length, 1),
        totalCostUsd: totalCost,
        activeSessions: new Set(logs.map(l => l.context.sessionId)).size,
        periodStart: new Date(periodStart).toISOString(),
        periodEnd: new Date(now).toISOString(),
      },
      latencyPanel: {
        p50Ms: this.collector.percentile('llm.latency.ms', 50),
        p90Ms: this.collector.percentile('llm.latency.ms', 90),
        p99Ms: this.collector.percentile('llm.latency.ms', 99),
        timeToFirstTokenP50Ms: this.collector.percentile('llm.time_to_first_token.ms', 50),
        timeToFirstTokenP90Ms: this.collector.percentile('llm.time_to_first_token.ms', 90),
        timeSeries: [], // Populated from time-bucketed data in production
      },
      tokenPanel: {
        totalInputTokens: logs.reduce((sum, l) => sum + l.response.inputTokens, 0),
        totalOutputTokens: logs.reduce((sum, l) => sum + l.response.outputTokens, 0),
        avgInputPerRequest: this.collector.average('llm.tokens.input'),
        avgOutputPerRequest: this.collector.average('llm.tokens.output'),
        tokensByModel,
      },
      costPanel: {
        totalCostUsd: totalCost,
        costByModel,
        costByFeature,
        topUsersByCost: Object.entries(userCosts)
          .map(([userId, costUsd]) => ({ userId, costUsd }))
          .sort((a, b) => b.costUsd - a.costUsd)
          .slice(0, 10),
        costTrend: [], // Populated from daily aggregates in production
        projectedDailyCost: (totalCost / Math.max(periodHours, 1)) * 24,
        budgetRemainingUsd: 100 - totalCost, // Example: $100/day budget
      },
      qualityPanel: {
        cacheHitRate: report.cache?.hitRate || 0,
        retrievalRelevanceAvg: this.collector.average('rag.retrieval.top_score'),
        agentCompletionRate:
          this.collector.total('agent.completed') /
          Math.max(this.collector.total('agent.completed') + this.collector.total('agent.max_steps_reached'), 1),
        avgAgentSteps: this.collector.average('agent.steps.total'),
        finishReasonBreakdown: finishReasons,
      },
      errorPanel: {
        errorRate: errorLogs.length / Math.max(logs.length, 1),
        errorsByType,
        recentErrors: errorLogs.slice(-10).map(l => ({
          timestamp: l.timestamp,
          errorCode: l.error?.code || 'unknown',
          message: l.error?.message || 'Unknown error',
          traceId: l.context.traceId,
        })),
      },
    }
  }
}
```

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

class AlertManager {
  private rules: AlertRule[] = []
  private activeAlerts: Map<string, AlertEvent> = new Map()
  private lastAlertTime: Map<string, number> = new Map()
  private alertHistory: AlertEvent[] = []

  constructor() {
    // Default alert rules for LLM applications
    this.rules = [
      {
        id: 'error-rate-high',
        name: 'High Error Rate',
        description: 'Error rate exceeds 5%',
        metric: 'error_rate',
        condition: 'above',
        threshold: 0.05,
        windowMinutes: 15,
        severity: 'critical',
        notifyChannels: ['slack', 'pagerduty'],
        cooldownMinutes: 30,
      },
      {
        id: 'latency-p99-high',
        name: 'P99 Latency Degradation',
        description: 'P99 latency exceeds 10 seconds',
        metric: 'latency_p99_ms',
        condition: 'above',
        threshold: 10000,
        windowMinutes: 10,
        severity: 'warning',
        notifyChannels: ['slack'],
        cooldownMinutes: 15,
      },
      {
        id: 'cost-spike',
        name: 'Hourly Cost Spike',
        description: 'Hourly cost exceeds 3x the daily average',
        metric: 'hourly_cost_usd',
        condition: 'above',
        threshold: 0, // Dynamically calculated
        windowMinutes: 60,
        severity: 'warning',
        notifyChannels: ['slack', 'email'],
        cooldownMinutes: 60,
      },
      {
        id: 'daily-budget-warning',
        name: 'Daily Budget Warning',
        description: 'Daily spend exceeds 70% of budget',
        metric: 'daily_cost_usd',
        condition: 'above',
        threshold: 70, // 70% of daily budget
        windowMinutes: 1440, // 24 hours
        severity: 'warning',
        notifyChannels: ['slack'],
        cooldownMinutes: 240,
      },
      {
        id: 'daily-budget-critical',
        name: 'Daily Budget Critical',
        description: 'Daily spend exceeds 90% of budget',
        metric: 'daily_cost_usd',
        condition: 'above',
        threshold: 90, // 90% of daily budget
        windowMinutes: 1440,
        severity: 'critical',
        notifyChannels: ['slack', 'pagerduty', 'email'],
        cooldownMinutes: 120,
      },
      {
        id: 'cache-hit-rate-low',
        name: 'Low Cache Hit Rate',
        description: 'Cache hit rate drops below 20%',
        metric: 'cache_hit_rate',
        condition: 'below',
        threshold: 0.2,
        windowMinutes: 30,
        severity: 'info',
        notifyChannels: ['slack'],
        cooldownMinutes: 60,
      },
      {
        id: 'retrieval-quality-low',
        name: 'Low Retrieval Quality',
        description: 'Average retrieval relevance drops below 0.6',
        metric: 'retrieval_relevance_avg',
        condition: 'below',
        threshold: 0.6,
        windowMinutes: 30,
        severity: 'warning',
        notifyChannels: ['slack'],
        cooldownMinutes: 60,
      },
      {
        id: 'agent-max-steps',
        name: 'Agent Loop Failures',
        description: 'More than 10% of agent loops hit max steps',
        metric: 'agent_max_steps_rate',
        condition: 'above',
        threshold: 0.1,
        windowMinutes: 30,
        severity: 'warning',
        notifyChannels: ['slack'],
        cooldownMinutes: 60,
      },
    ]
  }

  // Evaluate all rules against current metrics
  evaluate(dashboardData: DashboardData): AlertEvent[] {
    const newAlerts: AlertEvent[] = []
    const now = Date.now()

    for (const rule of this.rules) {
      // Check cooldown
      const lastAlert = this.lastAlertTime.get(rule.id)
      if (lastAlert && now - lastAlert < rule.cooldownMinutes * 60 * 1000) {
        continue
      }

      const currentValue = this.getMetricValue(rule.metric, dashboardData)
      let triggered = false

      switch (rule.condition) {
        case 'above':
          triggered = currentValue > rule.threshold
          break
        case 'below':
          triggered = currentValue < rule.threshold
          break
        case 'rate_increase':
          // Would compare against historical baseline
          triggered = false
          break
      }

      if (triggered) {
        const alert: AlertEvent = {
          ruleId: rule.id,
          ruleName: rule.name,
          severity: rule.severity,
          timestamp: new Date().toISOString(),
          currentValue,
          threshold: rule.threshold,
          message: `${rule.name}: current value ${currentValue.toFixed(4)} ${rule.condition} threshold ${rule.threshold}`,
          traceIds: dashboardData.errorPanel.recentErrors.slice(0, 3).map(e => e.traceId),
        }

        newAlerts.push(alert)
        this.activeAlerts.set(rule.id, alert)
        this.alertHistory.push(alert)
        this.lastAlertTime.set(rule.id, now)

        // In production: send to notification channels
        this.notify(alert, rule.notifyChannels)
      } else {
        // Clear resolved alerts
        if (this.activeAlerts.has(rule.id)) {
          this.activeAlerts.delete(rule.id)
        }
      }
    }

    return newAlerts
  }

  private getMetricValue(metric: string, data: DashboardData): number {
    switch (metric) {
      case 'error_rate':
        return data.overview.errorRate
      case 'latency_p99_ms':
        return data.latencyPanel.p99Ms
      case 'hourly_cost_usd':
        return data.costPanel.totalCostUsd
      case 'daily_cost_usd':
        return (
          (data.costPanel.totalCostUsd / Math.max(data.costPanel.budgetRemainingUsd + data.costPanel.totalCostUsd, 1)) *
          100
        )
      case 'cache_hit_rate':
        return data.qualityPanel.cacheHitRate
      case 'retrieval_relevance_avg':
        return data.qualityPanel.retrievalRelevanceAvg
      case 'agent_max_steps_rate':
        return 1 - data.qualityPanel.agentCompletionRate
      default:
        return 0
    }
  }

  private notify(alert: AlertEvent, channels: string[]): void {
    for (const channel of channels) {
      console.log(`[ALERT][${alert.severity.toUpperCase()}][${channel}] ${alert.message}`)
    }
  }

  getActiveAlerts(): AlertEvent[] {
    return Array.from(this.activeAlerts.values())
  }

  getAlertHistory(): AlertEvent[] {
    return [...this.alertHistory]
  }
}
```

---

## Section 8: Privacy Considerations

### What to Log and What Not to Log

Logging LLM interactions creates a tension between debuggability and privacy. Full prompt/response logs are invaluable for debugging but may contain PII, sensitive business data, or information subject to regulatory constraints.

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// PII detection patterns
const piiPatterns: Array<{
  name: string
  pattern: RegExp
  replacement: string
}> = [
  {
    name: 'email',
    pattern: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
    replacement: '[EMAIL_REDACTED]',
  },
  {
    name: 'phone',
    pattern: /(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g,
    replacement: '[PHONE_REDACTED]',
  },
  {
    name: 'ssn',
    pattern: /\b\d{3}-\d{2}-\d{4}\b/g,
    replacement: '[SSN_REDACTED]',
  },
  {
    name: 'credit_card',
    pattern: /\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/g,
    replacement: '[CC_REDACTED]',
  },
  {
    name: 'ip_address',
    pattern: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g,
    replacement: '[IP_REDACTED]',
  },
  {
    name: 'api_key',
    pattern: /\b(sk-|pk-|api[_-]?key[=:]\s*)[a-zA-Z0-9_-]{20,}\b/gi,
    replacement: '[API_KEY_REDACTED]',
  },
]

// Privacy-aware logging middleware
class PrivacyFilter {
  private patterns: typeof piiPatterns
  private detectionCounts: Record<string, number> = {}

  constructor(customPatterns?: typeof piiPatterns) {
    this.patterns = customPatterns || piiPatterns
  }

  // Redact PII from text
  redact(text: string): {
    redacted: string
    detections: string[]
  } {
    let result = text
    const detections: string[] = []

    for (const { name, pattern, replacement } of this.patterns) {
      const matches = result.match(pattern)
      if (matches && matches.length > 0) {
        detections.push(`${name}: ${matches.length} instance(s)`)
        this.detectionCounts[name] = (this.detectionCounts[name] || 0) + matches.length
        result = result.replace(pattern, replacement)
      }
    }

    return { redacted: result, detections }
  }

  // Check if text contains PII (without redacting)
  containsPII(text: string): boolean {
    for (const { pattern } of this.patterns) {
      if (pattern.test(text)) {
        // Reset regex lastIndex since we use the 'g' flag
        pattern.lastIndex = 0
        return true
      }
    }
    return false
  }

  // Get PII detection statistics
  getStats(): Record<string, number> {
    return { ...this.detectionCounts }
  }
}

// Privacy-aware logger configuration
interface PrivacyPolicy {
  // What to log in each environment
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

const defaultPrivacyPolicy: PrivacyPolicy = {
  development: {
    logFullPrompts: true,
    logFullResponses: true,
    redactPII: false, // Developers need to see real data for debugging
  },
  staging: {
    logFullPrompts: true,
    logFullResponses: true,
    redactPII: true, // Redact PII but log full content
  },
  production: {
    logFullPrompts: false, // Only log hashes and previews
    logFullResponses: false,
    redactPII: true,
    maxPreviewLength: 100,
    retentionDays: 30, // Delete logs after 30 days
  },
}

// Privacy-aware logging wrapper
class PrivacyAwareLogger {
  private filter: PrivacyFilter
  private policy: PrivacyPolicy
  private environment: 'development' | 'staging' | 'production'

  constructor(environment: 'development' | 'staging' | 'production', policy?: PrivacyPolicy) {
    this.filter = new PrivacyFilter()
    this.policy = policy || defaultPrivacyPolicy
    this.environment = environment
  }

  // Prepare text for logging according to privacy policy
  prepareForLogging(text: string): {
    loggableText: string
    piiDetected: boolean
    detections: string[]
  } {
    const envPolicy = this.policy[this.environment]
    let result = text
    let detections: string[] = []

    // Check for PII
    const piiDetected = this.filter.containsPII(text)

    // Apply redaction if policy requires it
    if ('redactPII' in envPolicy && envPolicy.redactPII) {
      const redactResult = this.filter.redact(text)
      result = redactResult.redacted
      detections = redactResult.detections
    }

    // Truncate if in production
    if (this.environment === 'production' && !this.policy.production.logFullPrompts) {
      const maxLen = this.policy.production.maxPreviewLength
      if (result.length > maxLen) {
        result = result.substring(0, maxLen) + `... [truncated, ${text.length} total chars]`
      }
    }

    return { loggableText: result, piiDetected, detections }
  }

  // Log a warning if PII is detected (useful for alerting)
  logPIIWarning(context: string, detections: string[]): void {
    if (detections.length > 0) {
      console.warn(
        JSON.stringify({
          level: 'warn',
          event: 'pii_detected',
          context,
          detections,
          timestamp: new Date().toISOString(),
        })
      )
    }
  }
}

// Usage example
const privacyLogger = new PrivacyAwareLogger('production')

const userInput = 'My email is john@example.com and my phone is 555-123-4567'
const prepared = privacyLogger.prepareForLogging(userInput)

console.log('Loggable text:', prepared.loggableText)
// Output: "My email is [EMAIL_REDACTED] and my phone is [PHONE_REDAC...
//         [truncated, 62 total chars]"
console.log('PII detected:', prepared.piiDetected)
// Output: true
console.log('Detections:', prepared.detections)
// Output: ["email: 1 instance(s)", "phone: 1 instance(s)"]
```

> **Beginner Note:** The simplest privacy-safe approach is to never log the full prompt or response in production. Log only metadata: model, token counts, latency, cost, user ID, and a hash of the prompt for grouping. If you need to debug a specific request, use the trace ID to look it up in a separate, access-controlled debug log with a short retention period.

> **Advanced Note:** Consider implementing data residency controls for multi-region deployments. Logs containing user data may need to stay in the same region as the user (GDPR, data sovereignty laws). Use log routing to direct logs to region-specific storage. Also consider implementing a "right to be forgotten" mechanism that can purge all logs associated with a specific user ID on request.

> **Local Alternative (Ollama):** All observability patterns (structured logging, tracing, metrics) work identically with `ollama('qwen3.5')`. Logging and tracing are application-level concerns independent of the model provider. In fact, observability is more important with local models — you need to monitor inference speed, GPU utilization, and memory usage in addition to the standard LLM metrics.

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
