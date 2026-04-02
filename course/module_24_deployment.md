# Module 24: Deployment

## Learning Objectives

- Evaluate deployment options for LLM applications: serverless, edge, and long-running servers
- Build a production API server using Hono with TypeScript
- Implement streaming endpoints that deliver LLM responses via Server-Sent Events
- Add authentication and API key management middleware to protect endpoints
- Design multi-tenant architectures with per-user data isolation
- Implement rate limiting strategies: per-user, per-endpoint, and token-based
- Build provider failover systems that automatically switch when a primary provider is down
- Handle scaling considerations including concurrency, queue-based processing, and backpressure
- Manage environment configuration and secrets securely

---

## Why Should I Care?

Building an LLM application that works on your laptop is the easy part. Deploying it so that hundreds or thousands of users can use it reliably, securely, and affordably is where the real engineering happens. A RAG pipeline that runs perfectly in a script needs an HTTP layer, authentication, rate limiting, error handling, streaming support, and monitoring before it can serve real users.

LLM applications have unique deployment challenges. Every request is expensive -- you cannot simply throw more servers at the problem like you can with traditional web applications. Each request can take 2-30 seconds, which breaks the assumptions of many web frameworks and load balancers designed for sub-100ms responses. The responses are often streamed token-by-token, requiring Server-Sent Events or WebSocket connections. And because you depend on external API providers, you need failover strategies for when they go down.

This module teaches you to build production-ready API servers for LLM applications using Hono, a lightweight TypeScript web framework that runs on Node.js, Deno, Bun, Cloudflare Workers, and other runtimes. You will add authentication, rate limiting, streaming, multi-tenancy, and provider failover -- everything you need to go from prototype to production.

---

## Connection to Other Modules

- **Module 6 (Streaming)** introduces the streaming concepts you will expose via SSE endpoints.
- **Module 7 (Tool Use)** creates tool-calling capabilities that need API endpoints.
- **Module 9-10 (RAG)** builds the retrieval pipelines you will serve through the API.
- **Module 14-15 (Agents)** creates long-running agent loops that need queue-based processing.
- **Module 21 (Safety)** provides guardrails that become middleware in the deployment stack.
- **Module 22 (Cost Optimization)** connects through rate limiting, caching, and model routing.
- **Module 23 (Observability)** provides the logging, tracing, and metrics infrastructure.

---

## Section 1: Deployment Options

### Choosing the Right Deployment Model

LLM applications have different runtime characteristics than traditional web applications. Understanding these characteristics helps you choose the right deployment model.

```typescript
// Deployment model comparison
interface DeploymentOption {
  name: string
  runtime: string
  maxRequestDuration: string
  coldStartMs: string
  bestFor: string[]
  limitations: string[]
  costModel: string
  examples: string[]
}
```

There are three main deployment models to consider:

**Serverless Functions** (AWS Lambda, Vercel Functions, Google Cloud Functions) -- Pay-per-invocation with 10-300 second duration limits and 100-1000ms cold starts. Best for simple LLM calls, low-traffic apps, and event-driven processing. The key limitations are duration limits that can kill long agent loops, no persistent connections (WebSockets are difficult), and stateless execution (no in-memory caching between requests).

**Edge Functions** (Cloudflare Workers, Vercel Edge, Deno Deploy) -- Pay-per-request with near-zero cold starts (< 10ms) but strict execution limits (30-120 seconds). Best for low-latency routing and simple transformations. Limited by restricted runtime APIs, tight memory constraints (128MB typical), and limited library compatibility.

**Long-Running Servers** (Node.js/Bun/Deno on VMs or containers) -- Always-on with unlimited request duration and no cold starts. Best for complex agent loops, WebSocket connections, in-memory caching, and high-traffic applications. The trade-offs are managing scaling yourself and paying for idle time.

Build a `recommendDeployment` function that takes application requirements and returns a recommendation:

```typescript
function recommendDeployment(params: {
  maxRequestDurationSeconds: number
  requiresWebSockets: boolean
  needsInMemoryCache: boolean
  trafficPattern: 'bursty' | 'steady' | 'low'
  hasAgentLoops: boolean
}): string
```

Think about the decision tree: which parameter should you check first? If agent loops can run for minutes, which deployment model is immediately ruled out? What about bursty traffic with short requests -- which model handles that best without paying for idle time?

> **Beginner Note:** If you are unsure which deployment model to choose, start with a long-running server. It has the fewest limitations and gives you the most flexibility. You can always move to serverless later if cost or scaling becomes a concern. A long-running Node.js server on a platform like Railway or Fly.io costs $5-20/month and handles most use cases.

---

## Section 2: Building an API with Hono

### Why Hono?

Hono is a lightweight, fast web framework for TypeScript that runs on any JavaScript runtime -- Node.js, Bun, Deno, Cloudflare Workers, and more. It has Express-like ergonomics with better TypeScript support and a smaller footprint.

Build a Hono application with these components:

1. **Global middleware** -- Apply CORS and request logging to all routes using `app.use('*', cors())` and `app.use('*', logger())`.

2. **Health check endpoint** (`GET /health`) -- Return a JSON response with `status`, `timestamp`, and `version`. This is always unauthenticated.

3. **Text generation endpoint** (`POST /api/generate`) -- Accept a JSON body with `prompt`, optional `system`, and optional `maxOutputTokens`. Validate that `prompt` is present (return 400 if missing). Call `generateText` and return the text, usage, and finish reason. Wrap in try/catch and return 500 on error.

4. **Structured output endpoint** (`POST /api/analyze`) -- Accept `text` and optional `analysisType`. Use `Output.object()` with a Zod schema for sentiment analysis (summary, sentiment enum, key points array, confidence score). Return the parsed output and usage.

```typescript
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { logger } from 'hono/logger'
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const app = new Hono()
```

How would you export the app for different runtimes? Hono supports `export default app` for Bun/Cloudflare and `serve({ fetch: app.fetch, port: 3000 })` from `@hono/node-server` for Node.js.

### Request Validation Middleware

Build a reusable `validateBody` middleware factory that accepts a Zod schema and validates the request body before the handler runs:

```typescript
function validateBody<T>(schema: ZodSchema<T>): MiddlewareHandler
```

The middleware should parse the request body against the schema. On success, store the validated body via `c.set('validatedBody', validated)` and call `next()`. On `ZodError`, return a 400 response with structured error details (field path and message for each issue). On other errors, return a generic 400.

Define a `generateRequestSchema` using Zod with these fields: `prompt` (string, min 1, max 10000), `system` (optional string, max 5000), `maxOutputTokens` (optional int, default 1024), `temperature` (optional number 0-2, default 0), and `model` (optional enum of supported model IDs, default `'mistral-small-latest'`).

Then build a `createModel` factory function that routes to the correct provider based on the model ID prefix. Model IDs starting with `'claude'` go to the Anthropic provider; everything else goes to Mistral.

> **Advanced Note:** Hono supports type-safe routing with `hono/validator` for even tighter TypeScript integration. You can also use `hono/zod-validator` for automatic Zod schema validation that integrates with Hono's type system, giving you end-to-end type safety from the request schema to the handler function.

---

## Section 3: Streaming Endpoints

### Server-Sent Events (SSE)

LLM responses are naturally streamed -- the model generates tokens one at a time. Exposing this via SSE gives users a responsive experience where they see the response build incrementally.

Build two streaming endpoints:

**Basic SSE endpoint** (`POST /api/stream`) -- Set the three required SSE headers (`Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`). Use Hono's `stream` helper from `hono/streaming` to create the response. Inside the stream callback, call `streamText` from the AI SDK and iterate over `result.textStream`, writing each chunk as an SSE event in the format `data: ${JSON.stringify({ type: 'text', content: chunk })}\n\n`. After the stream completes, await `result.usage` and send a final `done` event with the usage data. Wrap everything in try/catch and send an `error` event if something fails.

```typescript
import { stream } from 'hono/streaming'

// SSE event format: data: <json>\n\n
await stream.write(`data: ${JSON.stringify({ type: 'text', content: chunk })}\n\n`)
```

**Streaming with tool calls** (`POST /api/stream-with-tools`) -- Use `result.fullStream` instead of `result.textStream` to get typed events. Handle four event types in a switch statement: `text-delta` (forward the text), `tool-call` (forward tool name and args), `tool-result` (forward tool name and result), and `finish` (forward finish reason and usage). What other event types does `fullStream` emit that you might want to handle?

### Client-Side SSE Consumption

Build a `consumeStream` function that reads SSE events from the server:

```typescript
async function consumeStream(
  prompt: string,
  onChunk: (text: string) => void,
  onDone: (usage: { inputTokens: number; outputTokens: number }) => void,
  onError: (error: string) => void
): Promise<void>
```

The function should POST to the stream endpoint, get a `ReadableStream` reader from the response body, and read chunks in a loop. Use a `TextDecoder` with `{ stream: true }` to handle multi-byte characters split across chunks. Maintain a buffer and split on `\n` -- the last element (possibly incomplete) stays in the buffer for the next iteration. Parse lines that start with `data: ` and dispatch to the appropriate callback.

Why is the buffer approach necessary? What happens if a JSON payload is split across two network chunks?

> **Beginner Note:** Server-Sent Events (SSE) are a simple, HTTP-based protocol for one-way streaming from server to client. Unlike WebSockets, SSE works over regular HTTP, is automatically reconnected by browsers, and passes through proxies and load balancers without special configuration. For LLM applications where the server streams tokens to the client, SSE is the ideal protocol.

---

## Section 4: Authentication & API Keys

### API Key Authentication

Build an `APIKeyStore` class and an `authMiddleware` function to protect your API routes.

The `APIKey` interface captures the key metadata:

```typescript
interface APIKey {
  key: string
  userId: string
  name: string
  tier: 'free' | 'pro' | 'enterprise'
  createdAt: string
  lastUsedAt?: string
  rateLimit: number // Requests per minute
  dailyBudgetUsd: number
  active: boolean
}
```

The `APIKeyStore` class needs:

- A `Map<string, APIKey>` for storage
- `addKey(key)` -- stores a key in the map
- `validate(key)` -- looks up the key, returns `null` if missing or inactive, otherwise updates `lastUsedAt` and returns the key
- `getUserKeys(userId)` -- returns all keys for a user
- `revokeKey(key)` -- sets `active` to `false`

Seed the store with three test keys at different tiers (free with 10 RPM / $1 budget, pro with 60 RPM / $25 budget, enterprise with 300 RPM / $500 budget).

Build `authMiddleware()` that extracts the API key from the `Authorization` header (supporting both `Bearer <key>` and raw key formats), validates it against the store, and attaches `apiKey`, `userId`, and `tier` to the request context via `c.set()`. Return 401 if the header is missing or the key is invalid.

For web applications, also build a `jwtMiddleware(secret)` that decodes a JWT from the Bearer token, checks expiration, and attaches user context. Use `Buffer.from(parts[1], 'base64url').toString()` to decode the payload. In production, you would use a proper JWT library like `jose`.

Wire it up: public routes (like `/health`) have no auth, protected routes under `/api` use the auth middleware.

How would you implement tier-based model selection? Think about mapping tiers to model IDs so free users get a cheaper model and enterprise users get a premium one.

---

## Section 5: Multi-Tenancy

### Per-User Data Isolation

Multi-tenant LLM applications must isolate user data so that one user's documents, conversations, and embeddings are never visible to another user.

Build a `TenantStore` class with per-user data isolation:

```typescript
interface TenantData {
  userId: string
  documents: Array<{
    id: string
    content: string
    embedding?: number[]
    metadata: Record<string, string>
  }>
  conversations: Array<{
    id: string
    messages: Array<{ role: 'user' | 'assistant'; content: string }>
    createdAt: string
    updatedAt: string
  }>
  settings: {
    defaultModel: string
    systemPrompt: string
    maxTokensPerRequest: number
  }
}
```

The `TenantStore` needs these methods:

- `getTenant(userId)` -- returns the tenant data, creating it with default settings if it does not exist
- `addDocument(userId, doc)` -- adds a document to a specific tenant's collection
- `searchDocuments(userId, queryEmbedding, limit?)` -- searches ONLY this tenant's documents using cosine similarity (from the AI SDK's `cosineSimilarity` function), returning results sorted by score descending
- `getConversation(userId, conversationId)` -- finds a conversation by ID within a tenant
- `upsertConversation(userId, conversationId, messages)` -- creates or updates a conversation

Then build two API routes that use the tenant store:

**Upload document** (`POST /api/documents`) -- Get the `userId` from the authenticated context (never from the request body). Generate an embedding for the document content, create a unique document ID, and store everything in the tenant store.

**Query with RAG** (`POST /api/query`) -- Embed the question, search only this tenant's documents, assemble context from relevant documents, include conversation history if a `conversationId` is provided, call `generateText`, and save the updated conversation.

What is the critical security principle here? Why must the `userId` always come from the authenticated session, never from the request body?

> **Beginner Note:** Multi-tenancy means serving multiple users (tenants) from the same application while keeping their data separate. The simplest approach is to add a `userId` filter to every database query. Never assume the user ID from the request body -- always derive it from the authenticated session or API key. A common security bug is allowing users to specify another user's ID in the request.

---

## Section 6: Rate Limiting

### Multi-Layer Rate Limiting

LLM applications need rate limiting at multiple levels to prevent abuse and control costs.

Build a `RateLimiter` class with sliding window rate limiting:

```typescript
interface RateLimitConfig {
  windowMs: number // Time window in milliseconds
  maxRequests: number // Max requests per window
  maxTokensPerWindow?: number // Max tokens per window
  keyPrefix: string
}

class RateLimiter {
  constructor(private config: RateLimitConfig)

  check(key: string, estimatedTokens?: number): {
    allowed: boolean
    remaining: number
    resetAt: number
    retryAfterMs?: number
  }

  recordTokens(key: string, actualTokens: number): void
}
```

The `check` method should maintain a `Map` of time windows keyed by `${keyPrefix}:${key}`. Each window tracks `count`, `tokens`, and `resetAt`. If the current time exceeds `resetAt`, create a new window. Check the request count against `maxRequests` first, then the token count against `maxTokensPerWindow` (if configured). If either limit is exceeded, return `allowed: false` with `retryAfterMs`. Otherwise, increment the counts and return the remaining quota.

The `recordTokens` method is called after the request completes to record actual token usage (since you only had an estimate at check time).

Define per-tier configurations: free (10 requests/min, 10K tokens/min), pro (60 requests/min, 100K tokens/min), enterprise (300 requests/min, 1M tokens/min). Also create a global rate limiter (1000 requests/min across all users).

Build a `rateLimitMiddleware()` that checks both the global and per-user limits. Set standard rate limit headers on the response: `Retry-After`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset`. Return 429 with a descriptive error message when either limit is exceeded.

Why do you need both global and per-user rate limits? What happens if you only have per-user limits and 1000 free-tier users all make requests simultaneously?

> **Advanced Note:** In production, use Redis or a similar distributed store for rate limiting state instead of in-memory maps. In-memory rate limiters reset when the server restarts and do not work across multiple server instances. Redis-based rate limiters (e.g., using the sliding window algorithm) provide consistent rate limiting across all instances and survive restarts.

---

## Section 7: Provider Failover

### Automatic Provider Switching

LLM providers occasionally experience outages, rate limiting, or degraded performance. A failover system automatically routes requests to backup providers when the primary is unavailable.

Build a `ProviderFailover` class with circuit breaker support:

```typescript
interface ProviderHealth {
  provider: string
  healthy: boolean
  lastError?: string
  lastErrorAt?: number
  consecutiveErrors: number
  totalErrors: number
  totalRequests: number
  avgLatencyMs: number
  lastSuccessAt?: number
  circuitBreakerOpen: boolean
  circuitBreakerResetAt?: number
}

class ProviderFailover {
  constructor()

  async generateWithFailover(params: {
    tier: 'economy' | 'standard' | 'premium'
    prompt: string
    system?: string
    maxOutputTokens?: number
  }): Promise<{
    text: string
    provider: string
    model: string
    usage: any
    failoverAttempts: number
  }>

  getHealthStatus(): ProviderHealth[]
}
```

The class should maintain a provider chain -- an ordered list of providers, each with a `name`, `createModel` factory function, `modelMapping` (mapping tier names to model IDs), and `priority`. Initialize health tracking for each provider.

Key behaviors to implement:

**`getAvailableProviders()`** -- Filter out providers whose circuit breaker is open, unless the reset time has passed (in which case, reset the breaker and allow one request through).

**`recordSuccess(providerName, latencyMs)`** -- Reset consecutive errors, update average latency using a running average formula, and mark as healthy.

**`recordFailure(providerName, error)`** -- Increment consecutive errors. If the count reaches the threshold (e.g., 5), open the circuit breaker with a cooldown (e.g., 60 seconds). Log a warning.

**`generateWithFailover(params)`** -- Iterate over available providers in priority order. For each, look up the model ID from the tier mapping, call `generateText`, and record success/failure. On failure, log a warning and try the next provider. If all providers fail, throw an error with the last error message.

What should `failoverAttempts` in the return value communicate to the caller? How could the API route use this information?

> **Beginner Note:** A circuit breaker is a pattern from electrical engineering applied to software. When a provider fails repeatedly, the circuit breaker "opens" and stops sending requests to it for a cooldown period. This prevents wasting time on a known-bad provider and gives it time to recover. After the cooldown, it lets one request through to test if the provider has recovered.

---

## Section 8: Scaling Considerations

### Concurrency and Queue-Based Processing

LLM requests are slow (2-30 seconds) and expensive. You cannot handle them the same way you handle traditional web requests. Queue-based processing allows you to control concurrency, implement backpressure, and handle bursts.

Build a `RequestQueue` class that manages concurrent LLM requests:

```typescript
interface QueuedRequest {
  id: string
  userId: string
  tier: string
  prompt: string
  system?: string
  maxOutputTokens?: number
  resolve: (value: any) => void
  reject: (error: Error) => void
  queuedAt: number
  priority: number // Lower = higher priority
  timeoutMs: number
}

class RequestQueue {
  constructor(maxConcurrent: number = 10)

  enqueue(params: {
    userId: string
    tier: string
    prompt: string
    system?: string
    maxOutputTokens?: number
    timeoutMs?: number
  }): Promise<any>

  getStatus(): {
    queueLength: number
    activeRequests: number
    maxConcurrent: number
  }
}
```

The `enqueue` method returns a Promise that resolves when the request completes. Internally, it creates a `QueuedRequest` with a priority based on tier (enterprise=1, pro=2, free=3), adds it to the queue, sorts by priority, and triggers processing.

The private `processQueue` method should loop while there are queued requests AND active requests are below `maxConcurrent`. For each request, check if it has timed out while waiting in the queue -- if so, reject its promise and continue to the next one. Otherwise, increment `activeRequests` and process the request without awaiting (let it run concurrently). In the `.finally()` callback, decrement `activeRequests` and call `processQueue` again to keep the pipeline full.

Wire it into the API route with backpressure: if `queueLength` exceeds a threshold (e.g., 100), return 503 immediately. This prevents the queue from growing unboundedly during traffic spikes.

How does the priority queue ensure enterprise users get faster service? What happens to a free-tier request that sits in the queue behind 50 enterprise requests?

### Production Vector Storage

Throughout this course, we used LanceDB (embedded, no server) for vector storage. In production, choose based on scale:

| Scale | Recommendation | Why |
|-------|---------------|-----|
| <50M vectors | **pgvector** (PostgreSQL extension) | Zero incremental cost if you have Postgres. 471 QPS at 99% recall on 50M vectors. Single deployment, SQL + vectors in one DB. |
| >50M vectors, hybrid search | **Elasticsearch/OpenSearch** | Native distributed sharding, BM25 + vector fusion in one query, scales horizontally. |
| Billion-scale | **Qdrant, Milvus** | Purpose-built for sub-millisecond latency at extreme scale. |
| Edge/embedded | **LanceDB** | In-process, no server, SQLite-like for vectors. |

Most applications start with pgvector and never outgrow it.

---

## Section 9: Environment Configuration

### Secrets Management

LLM applications need multiple API keys (one per provider), database credentials, and other secrets. Managing these securely is critical.

Build an environment validation layer using Zod:

```typescript
import { z } from 'zod'

const envSchema = z.object({
  PORT: z.string().default('3000'),
  NODE_ENV: z.enum(['development', 'staging', 'production']).default('development'),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  MISTRAL_API_KEY: z.string().min(1, 'MISTRAL_API_KEY is required'),
  GROQ_API_KEY: z.string().optional(),
  ANTHROPIC_API_KEY: z.string().optional(),
  OPENAI_API_KEY: z.string().optional(),
  // ... database, auth, rate limiting, cost control, observability keys
})

type Env = z.infer<typeof envSchema>
```

Build two functions:

**`loadEnvironment()`** -- Parses `process.env` against the schema. On success, log which optional keys are present (as boolean flags, never the actual values). On `ZodError`, print each validation issue with the field path and message, then `process.exit(1)`. This ensures the application fails fast with a clear message rather than crashing at runtime when a missing key is first used.

**`createConfig(env)`** -- Transforms the flat environment variables into a structured configuration object with sections for `server` (port, environment, log level), `providers` (each with apiKey and enabled flag), `rateLimits`, `costControls`, and `observability`. Parse numeric strings to integers using `parseInt`.

### Putting It All Together

Build a complete server entry point that wires all the pieces from Sections 2-9 together:

1. Call `loadEnvironment()` and `createConfig()` at startup.
2. Create the Hono app with CORS middleware.
3. Add a public health check endpoint.
4. Create protected API routes with `authMiddleware()` and `rateLimitMiddleware()`.
5. Add generate, stream, and provider health endpoints.
6. Use `serve()` from `@hono/node-server` to start the server.

What order should the middleware be applied in? Think about what happens if rate limiting runs before authentication -- how would you identify the user for per-user limits?

> **Beginner Note:** Never commit API keys or secrets to version control. Use `.env` files for local development (added to `.gitignore`) and your deployment platform's secrets manager for production (e.g., Vercel Environment Variables, Railway Variables, AWS Secrets Manager). The Zod validation approach shown here ensures your application fails fast with a clear error message if a required secret is missing, rather than failing mysteriously at runtime.

> **Advanced Note:** For production deployments, consider using a secrets rotation strategy where API keys are automatically rotated on a schedule. This limits the blast radius of a leaked key. Also implement key scoping -- create separate API keys per environment (dev, staging, production) with appropriate permissions and spending limits.

> **Local Alternative (Ollama):** The Hono server, authentication, rate limiting, and API patterns work with any model provider. For deployment, you can run Ollama alongside your Hono server on the same machine or a private server. This gives you a fully self-hosted LLM application with no external API dependencies -- ideal for privacy-sensitive or air-gapped deployments.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 10: Session Management

Production LLM applications run for hours. Users expect to close the terminal, come back later, and pick up exactly where they left off. Session management provides this:

- **Session ID** — Each conversation gets a unique identifier.
- **Persist to disk** — Serialize the full conversation state (messages, tool results, metadata) to a JSON file on every turn.
- **Resume** — Reload a previous session by ID, restoring the full message history so the model has complete context.
- **Session teleport** — Move a session between machines by copying the session file.

**Pattern:** Save session state after every assistant response. Use the session ID as the filename. On resume, load the messages array and pass it to the next `generateText` call as the conversation history:

```ts
const sessionPath = `data/sessions/${sessionId}.json`
await Bun.write(sessionPath, JSON.stringify({ id: sessionId, messages, metadata }))
```

## Section 11: SDK Output Mode (NDJSON)

When your LLM service runs as an SDK (consumed by other programs, not humans), it needs machine-readable output. NDJSON (Newline-Delimited JSON) is the standard format:

- Each event is a self-contained JSON object on one line.
- Events are typed: `{ type: 'text', content: '...' }`, `{ type: 'tool_call', name: '...', args: {...} }`, `{ type: 'error', code: 'RATE_LIMITED', message: '...' }`.
- Clients parse one line at a time — no need to wait for the full response.
- Error codes (not exceptions) make programmatic error handling straightforward.

**Pattern:** Support both human-readable (interactive terminal) and machine-readable (NDJSON) output modes from the same core logic. A flag or content-type header selects the mode:

```ts
if (outputMode === 'ndjson') {
  stream.write(JSON.stringify({ type: 'text', content: chunk }) + '\n')
}
```

## Section 12: Health Endpoints

Production deployments need health checks that load balancers and uptime monitors can query:

- **Liveness probe** (`GET /health`) — Returns `200 OK` immediately. Confirms the process is running. Include server version, uptime, and timestamp.
- **Readiness probe** (`GET /health/ready`) — Runs dependency checks in parallel: Is the LLM API reachable? Is the vector store connected? Are we within rate limits? Returns `200` if all critical dependencies pass, `503` if any critical dependency is down.

Mark each dependency as `critical` or `optional`. A failed optional dependency (e.g., a fallback provider) results in a `degraded` status but not a `503`. This prevents unnecessary restarts when non-essential services are temporarily unavailable.

## Section 13: Graceful Shutdown

Production systems handle shutdown cleanly rather than killing in-progress work:

1. **Listen for signals** — Handle `SIGINT` (Ctrl+C) and `SIGTERM` (container stop) with process signal listeners.
2. **Save session state** — Persist the current conversation so the user can resume later.
3. **Complete or cancel in-progress operations** — If an LLM call is in flight, either wait for it to finish or abort the request cleanly.
4. **Free resources** — Close database connections, flush log buffers, clean up temp files.
5. **Exit with appropriate code** — `0` for clean shutdown, non-zero for errors.

```ts
process.on('SIGINT', async () => {
  await saveSession(currentSession)
  await flushLogs()
  server.close(() => process.exit(0))
})
```

Without graceful shutdown, users lose their conversation state and in-progress work when the process is interrupted.

## Section 14: Multi-Target Deployment

A single codebase can serve multiple deployment targets by separating the core logic from the I/O layer:

- **Core module** — LLM calls, tool execution, state management. No I/O assumptions.
- **HTTP entry point** — Hono server with SSE streaming (existing from earlier sections).
- **CLI entry point** — Accepts prompts via args or stdin, outputs structured JSON.
- **SDK entry point** — Exports the core module as a library for programmatic use.

**Pattern:** Refactor your server so the core LLM logic lives in a shared module. Entry points are thin shells that handle I/O and call into the core. This is the same pattern web applications use — API server + CLI tool + library, all sharing one implementation.

## Section 15: Client/Server Architecture

Production coding agents decouple the backend (LLM processing, tool execution, state management) from the frontend (terminal UI, desktop app, IDE extension). The terminal is just one client connecting to a backend.

This architecture enables:

- **Remote driving** — Control the agent from a mobile device while it executes on a development machine.
- **Multiple simultaneous clients** — IDE extension and terminal both connected to the same session.
- **IDE integration as thin client** — The VS Code extension does not embed the agent; it connects to the running backend.

**Pattern:** The LLM service is a server. Everything else is a client. This is the same separation that web applications use (API server + multiple frontends), applied to AI tooling. The interface boundary is an HTTP/WebSocket API that any client can consume.

## Section 16: Headless CI Execution

Production systems support non-interactive execution for automation:

- Accept prompts via command-line arguments or stdin: `tool exec "refactor this function"`
- Output structured results (NDJSON) instead of interactive formatting
- Support quiet mode that suppresses progress indicators and colors
- Return meaningful exit codes (0 = success, 1 = error, 2 = safety block)

This enables CI/CD integration (run code review as a pipeline step), scripted workflows (chain agent invocations in shell scripts), and batch processing (process multiple files programmatically). The headless mode shares the same core module as the interactive mode — only the I/O layer differs.

## Section 17: MCP Server Mode

Production coding agents can act as both an MCP client (consuming external tools) AND an MCP server (exposing their own capabilities). This bidirectional MCP support enables tool composition — one agent can use another agent as a tool.

Being an MCP server means the agent's capabilities (code search, file editing, RAG retrieval) are available to any MCP client, not just the agent's own UI. This is the microservices pattern applied to AI tools — each agent exposes a well-defined interface that other agents can consume.

## Section 18: Multi-Frontend Distribution

Production coding agents ship as multiple form factors from a single codebase:

- **CLI** — Terminal interface (what you have been building)
- **Desktop app** — Native application with richer UI (Electron, Tauri)
- **VS Code extension** — Embedded in the IDE as a thin client
- **Web interface** — Browser-based access via HTTP API

The core logic (LLM calls, tool execution, state management) is shared. Each distribution target is a thin shell around the same core library. Think of your LLM service as a core library with multiple frontends, not as a monolithic application tied to one interface.

---

## Summary

In this module, you learned:

1. **Deployment options:** Serverless, edge, and long-running server architectures each have trade-offs for LLM applications — long-running servers are often preferred due to streaming support and connection reuse.
2. **API server with Hono:** How to build a production HTTP server with typed routes, request validation middleware, and structured error handling.
3. **Streaming endpoints:** How to implement SSE endpoints that deliver token-by-token LLM responses to clients with proper headers and connection management.
4. **Authentication:** How to add API key authentication middleware that protects endpoints and identifies users for rate limiting and usage tracking.
5. **Multi-tenancy:** How to isolate per-user data, conversation history, and usage quotas in a shared deployment without cross-tenant data leakage.
6. **Rate limiting:** Implementing multi-layer rate limiting — per-user, per-endpoint, and token-based — to prevent abuse and control costs.
7. **Provider failover:** Building automatic provider switching that detects failures and routes requests to backup providers with health checks and circuit breakers.
8. **Scaling and configuration:** Handling concurrency with queue-based processing, managing backpressure, and securely managing environment variables and API keys across environments.
9. **Environment configuration:** Validating all environment variables at startup with Zod schemas so missing or malformed config fails fast with clear messages instead of causing cryptic runtime errors.
10. **Session management:** Persisting conversation state to disk on every turn so users can resume sessions, move them between machines, and survive process restarts.
11. **SDK output mode:** Supporting NDJSON output for machine-readable consumption alongside interactive terminal output, enabling programmatic integration and CI/CD pipelines.
12. **Health endpoints:** Implementing liveness and readiness probes with dependency checks that distinguish critical from optional services to prevent unnecessary restarts.
13. **Graceful shutdown:** Handling SIGINT/SIGTERM to save session state, complete in-flight operations, flush logs, and free resources before exiting cleanly.
14. **Multi-target deployment:** Separating core LLM logic from I/O so the same codebase serves HTTP, CLI, and SDK entry points.
15. **Client/server architecture:** Decoupling the backend from the frontend so multiple clients (terminal, IDE, mobile) can connect to the same agent session.
16. **Headless CI execution:** Supporting non-interactive mode with structured output and meaningful exit codes for automation and batch processing.
17. **MCP server mode:** Exposing agent capabilities as an MCP server so other agents and tools can consume them programmatically.
18. **Multi-frontend distribution:** Shipping CLI, desktop, VS Code extension, and web interfaces from a single shared core library.

You now have the complete toolkit to build, evaluate, secure, optimize, observe, and deploy production LLM applications.

---

## Quiz

**Question 1:** Why are long-running servers often preferred over serverless functions for LLM applications?

A) Long-running servers are cheaper
B) Serverless functions have request duration limits that can kill long agent loops, and they lack persistent connections for streaming
C) Serverless functions do not support TypeScript
D) Long-running servers have better security

**Answer: B** -- LLM applications often involve long-running agent loops (potentially minutes), streaming responses via SSE or WebSockets, and in-memory caching (semantic cache, embeddings). Serverless functions have duration limits (typically 10-300 seconds), no persistent connections, and reset state between invocations. Long-running servers provide unlimited duration, persistent connections, and in-memory state.

---

**Question 2:** What is the purpose of the circuit breaker pattern in provider failover?

A) To make requests faster
B) To reduce API costs
C) To stop sending requests to a provider that is consistently failing, giving it time to recover
D) To encrypt requests in transit

**Answer: C** -- The circuit breaker "opens" after a provider fails multiple times consecutively, preventing further requests to that provider for a cooldown period. This avoids wasting time and resources on requests that are very likely to fail. After the cooldown, the circuit breaker allows one request through to test if the provider has recovered. This pattern prevents cascading failures and improves overall system reliability.

---

**Question 3:** Why should rate limiting in LLM applications include token-based limits, not just request count limits?

A) Token-based limits are easier to implement
B) A single request with a 100K token prompt is far more expensive and resource-intensive than 100 requests with 100 token prompts
C) Token-based limits improve response quality
D) Request count limits are not supported by LLM APIs

**Answer: B** -- LLM costs and processing time are proportional to token count, not request count. A single request with a very long prompt and high maxOutputTokens can consume as many resources as hundreds of small requests. Token-based rate limiting ensures that no single user can monopolize resources or run up costs, regardless of how they structure their requests. Effective rate limiting needs both request count limits (to prevent API abuse) and token limits (to prevent cost abuse).

---

**Question 4:** What is the primary benefit of queue-based processing for LLM requests?

A) It makes responses faster
B) It reduces token costs
C) It controls concurrency, prevents overloading the LLM provider, and enables priority-based processing
D) It improves response quality

**Answer: C** -- Queue-based processing gives you control over how many requests are sent to the LLM provider simultaneously. Without it, a traffic spike could send hundreds of concurrent requests, causing rate limiting, timeouts, and degraded performance. The queue controls concurrency (max N requests at a time), implements priority (enterprise users before free users), provides backpressure (reject requests when queue is too deep), and handles timeouts gracefully.

---

**Question 5:** Why should environment configuration use schema validation (like Zod) instead of just reading process.env directly?

A) Zod is faster than process.env
B) Schema validation catches missing or malformed environment variables at startup rather than causing cryptic runtime errors
C) process.env does not work in production
D) Zod is required by the Vercel AI SDK

**Answer: B** -- Without validation, a missing API key causes a cryptic error only when the first LLM request is made, potentially minutes or hours after deployment. With Zod validation at startup, the application fails immediately with a clear message like "MISTRAL_API_KEY is required." This is especially important for LLM applications with multiple provider keys, database URLs, and configuration values. Schema validation also provides type safety, default values, and documentation of all required configuration.

---

**Question 6 (Medium):** A production LLM application supports both an interactive terminal and a CI/CD pipeline. What architectural pattern enables both use cases from the same codebase?

A) Maintain two separate codebases — one for interactive use and one for CI
B) Separate the core LLM logic from the I/O layer so thin entry points (CLI, HTTP, SDK) share one implementation
C) Use environment variables to conditionally compile different code paths
D) Run the interactive version in CI and parse the terminal output

**Answer: B** -- The multi-target deployment pattern separates core logic (LLM calls, tool execution, state management) from I/O (terminal formatting, HTTP routing, NDJSON output). Each deployment target is a thin shell that handles I/O and calls into the shared core. The CI entry point uses headless mode with structured NDJSON output and meaningful exit codes, while the terminal entry point uses interactive formatting. Both share identical business logic.

---

**Question 7 (Hard):** A deployment's readiness probe checks three dependencies: LLM API (critical), vector store (critical), and a fallback provider (optional). The fallback provider is temporarily down. What should the readiness endpoint return, and why?

A) HTTP 503 — any dependency failure means the service is not ready
B) HTTP 200 with a "degraded" status — the service can operate without the optional dependency, and returning 503 would cause unnecessary restarts
C) HTTP 200 with no indication of the failure — the fallback is not important
D) HTTP 500 — this is a server error

**Answer: B** -- Marking dependencies as critical or optional prevents unnecessary restarts. The service can function without the fallback provider (it just lacks a backup), so returning 503 would cause the load balancer to take the instance out of rotation or the orchestrator to restart it — both harmful. Returning "degraded" status communicates the issue to monitoring systems without triggering corrective actions designed for critical failures. This distinction is essential for production stability.

---

## Exercises

### Exercise 1: Deploy a RAG Pipeline as a Hono API

Build a complete Hono API server that exposes a RAG pipeline with authentication, rate limiting, and provider failover.

**Specification:**

1. Create a Hono server with these endpoints:
   - `GET /health` -- Health check (no auth)
   - `POST /api/documents` -- Upload documents (authenticated)
   - `POST /api/query` -- Query with RAG (authenticated, rate-limited)
   - `POST /api/stream` -- Streaming query (authenticated, rate-limited)
   - `GET /api/providers/health` -- Provider status (authenticated)

2. Implement authentication middleware with 3 API keys at different tiers (free, pro, enterprise).

3. Implement rate limiting:
   - Free: 10 requests/minute, 10K tokens/minute
   - Pro: 60 requests/minute, 100K tokens/minute
   - Enterprise: 300 requests/minute, 1M tokens/minute

4. Implement provider failover with at least 2 providers (Mistral primary, Groq or OpenAI fallback).

5. Add request validation with Zod schemas.

6. Add environment configuration with Zod validation.

7. Test all endpoints:
   - Upload 3 documents
   - Query with RAG (verify tenant isolation)
   - Stream a response
   - Verify rate limiting (exceed the free tier limit)
   - Simulate provider failure (verify failover)

**Expected output:** A running Hono server on localhost:3000 with all endpoints functional, rate limiting enforced, and provider failover working.

### Exercise 2: Health Check Endpoint with Dependency Status

Build a comprehensive health check system that reports the status of every external dependency your LLM application relies on, suitable for use by load balancers, uptime monitors, and on-call dashboards.

**Specification:**

1. Implement a `DependencyChecker` class that can register named dependencies (e.g., `anthropic`, `openai`, `redis`, `vector-store`) each with a health check function that returns `{ healthy: boolean; latencyMs: number; message?: string }`.

2. Implement two health endpoints on a Hono server:
   - `GET /health` -- A shallow health check that returns `200 OK` immediately (for load balancer liveness probes). It should include the server version, uptime in seconds, and current timestamp.
   - `GET /health/ready` -- A deep readiness check that runs all registered dependency checks in parallel with a per-check timeout of 5 seconds. Return `200` if all critical dependencies are healthy, or `503` if any critical dependency is down.

3. Each dependency should be registered as either `critical` or `optional`. A failed optional dependency (e.g., a fallback provider) should not cause the readiness endpoint to return `503`, but its degraded status should still appear in the response body.

4. Simulate the following dependency checks (they do not need to make real network calls):
   - **Anthropic API** (critical) -- Simulate a successful check with 120ms latency.
   - **OpenAI API** (optional) -- Simulate a failed check that times out.
   - **Redis** (critical) -- Simulate a successful check with 2ms latency.
   - **Vector Store** (critical) -- Simulate a successful check with 45ms latency.

5. The readiness response JSON should include:
   - Overall status (`healthy` or `degraded` or `unhealthy`)
   - Total check duration in milliseconds
   - An array of dependency results, each with: name, status, latencyMs, critical flag, and optional error message

6. Add a startup check that runs all critical dependency checks before the server begins accepting requests. If any critical dependency is unreachable at startup, log an error and exit with a non-zero status code.

**Expected output:** A Hono server where `GET /health` returns an instant `200` with uptime info, and `GET /health/ready` returns a detailed JSON report showing Mistral, Redis, and Vector Store as healthy, OpenAI as degraded (timed out), and an overall status of `degraded` (since only the optional dependency failed). The server should start successfully because all critical dependencies pass their checks.

### Exercise 3: Session Save and Resume

Build a session management system that persists conversation state to disk and supports resuming previous sessions.

**Specification:**

1. Create a `SessionManager` class with:
   - `create()` — Creates a new session with a unique ID, empty message history, and metadata (created timestamp, model name)
   - `save(session)` — Serializes the full session (messages, metadata, token usage) to a JSON file in `data/sessions/`
   - `resume(sessionId)` — Loads a session from disk and returns it ready for use with `generateText`
   - `list()` — Returns all saved sessions with their IDs, creation times, message counts, and last activity timestamps

2. Integrate with `generateText`: after each assistant response, automatically save the updated session. On resume, pass the loaded messages as conversation history.

3. Test the full lifecycle:
   - Create a session and have a 3-turn conversation
   - Save the session
   - Create a new `SessionManager` instance (simulating a restart)
   - Resume the session by ID
   - Continue the conversation for 2 more turns (verify the model has context from the earlier turns)
   - List all sessions and verify metadata

**Create:** `src/deployment/exercises/session-manager.ts`

**Expected output:** Console output showing the initial conversation, a simulated restart, the resumed conversation with context preserved, and a session listing.

### Exercise 4: SDK Output Mode

Add a NDJSON output mode to your Hono server alongside the existing SSE streaming mode.

**Specification:**

1. Create an NDJSON streaming endpoint (`POST /api/query/ndjson`) that:
   - Accepts the same request body as your existing query endpoint
   - Streams responses as NDJSON (one JSON object per line)
   - Emits typed events: `{ type: 'text_delta', content: '...' }`, `{ type: 'tool_call', name: '...', args: {...} }`, `{ type: 'usage', inputTokens: N, outputTokens: N }`, `{ type: 'done' }`
   - On error, emits `{ type: 'error', code: 'RATE_LIMITED', message: '...' }` instead of throwing

2. Create a simple NDJSON client function that:
   - Sends a request to the NDJSON endpoint
   - Reads the response line by line
   - Parses each line as JSON and handles each event type
   - Returns the assembled result (full text, tool calls, usage)

3. Test by sending the same query to both the SSE endpoint and the NDJSON endpoint. Verify that both produce equivalent final results.

**Create:** `src/deployment/exercises/ndjson-output.ts`

**Expected output:** Side-by-side output from SSE and NDJSON modes for the same query, showing that both produce the same final text and usage statistics.

### Exercise 5: Graceful Shutdown

Implement a graceful shutdown handler that saves state and cleans up resources on process termination.

**Specification:**

1. Create a `ShutdownManager` class with:
   - `register(name, cleanupFn)` — Registers a named cleanup function to run on shutdown
   - `shutdown()` — Runs all registered cleanup functions in order, with a per-function timeout
   - Signal handling — Listens for `SIGINT` and `SIGTERM` and triggers shutdown

2. Register these cleanup tasks:
   - Save the current session to disk (from Exercise 3)
   - Flush pending log entries to a file
   - Close the HTTP server (stop accepting new requests, wait for in-flight requests to complete)
   - Report final statistics (total requests served, total tokens used)

3. Implement a timeout: if all cleanup does not complete within 10 seconds, force exit with code 1.

4. Test by:
   - Starting a Hono server with the shutdown manager
   - Sending a request (verify it completes)
   - Triggering shutdown programmatically (simulate SIGINT)
   - Verifying that the session was saved, logs were flushed, and the server stopped cleanly

**Create:** `src/deployment/exercises/graceful-shutdown.ts`

**Expected output:** Console output showing each cleanup step completing in order, the saved session file, flushed logs, and a clean exit with code 0.

### Exercise 6: Remote Execution Bridge

Build a simple remote execution bridge where a local CLI sends prompts to a remote HTTP server and streams results back.

**Specification:**

1. Create a server component (`remote-server.ts`) that:
   - Accepts `POST /api/execute` with `{ prompt: string, sessionId?: string }`
   - Runs the prompt through `generateText` (or `streamText` for streaming)
   - Returns the result as NDJSON events
   - Supports session persistence (optional session ID to resume context)

2. Create a client component (`remote-client.ts`) that:
   - Accepts a prompt via command-line args or stdin
   - Sends it to the remote server URL (configurable via env var)
   - Streams NDJSON events back and prints them as they arrive
   - Supports `--session` flag to resume a previous session

3. Test the bridge:
   - Start the server on localhost:4000
   - Use the client to send a prompt and receive a streamed response
   - Send a follow-up prompt with the same session ID and verify context is preserved
   - Test with stdin: `echo "What is TypeScript?" | bun run remote-client.ts`

**Create:** `src/deployment/exercises/remote-server.ts` and `src/deployment/exercises/remote-client.ts`

**Expected output:** The client sending a prompt, the server processing it, and the streamed response appearing in the client's terminal, with session continuity demonstrated across multiple calls.
