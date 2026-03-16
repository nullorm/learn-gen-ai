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

const deploymentOptions: DeploymentOption[] = [
  {
    name: 'Serverless Functions',
    runtime: 'AWS Lambda, Vercel Functions, Google Cloud Functions',
    maxRequestDuration: '10-300 seconds (varies by provider)',
    coldStartMs: '100-1000ms',
    bestFor: [
      'Simple LLM calls with short response times',
      'Low-traffic applications',
      'Event-driven processing (webhooks, queue consumers)',
    ],
    limitations: [
      'Request duration limits can kill long agent loops',
      'Cold starts add latency to first request',
      'No persistent connections (WebSockets difficult)',
      'Stateless -- no in-memory caching between requests',
    ],
    costModel: 'Pay per invocation + duration',
    examples: ['Vercel Functions', 'AWS Lambda', 'Cloudflare Workers'],
  },
  {
    name: 'Edge Functions',
    runtime: 'Cloudflare Workers, Vercel Edge, Deno Deploy',
    maxRequestDuration: '30-120 seconds',
    coldStartMs: '< 10ms',
    bestFor: [
      'Low-latency responses close to users',
      'Simple request/response patterns',
      'Content transformation and routing',
    ],
    limitations: [
      'Limited runtime APIs (no Node.js full API)',
      'Strict memory limits (128MB typical)',
      'Short execution time limits',
      'Limited library compatibility',
    ],
    costModel: 'Pay per request + CPU time',
    examples: ['Cloudflare Workers', 'Vercel Edge Runtime'],
  },
  {
    name: 'Long-Running Server',
    runtime: 'Node.js, Bun, Deno on VMs or containers',
    maxRequestDuration: 'Unlimited',
    coldStartMs: '0ms (already running)',
    bestFor: [
      'Complex agent loops that run for minutes',
      'WebSocket connections for real-time streaming',
      'In-memory caching (semantic cache, embeddings)',
      'High-traffic applications with predictable load',
    ],
    limitations: [
      'Must manage scaling yourself (or use container orchestration)',
      'Pay for idle time',
      'Need health checks, graceful shutdown, etc.',
    ],
    costModel: 'Pay for compute time (always on)',
    examples: ['Docker on ECS/GKE/Fly.io', 'Railway', 'Render'],
  },
]

// Decision matrix: which option to choose
function recommendDeployment(params: {
  maxRequestDurationSeconds: number
  requiresWebSockets: boolean
  needsInMemoryCache: boolean
  trafficPattern: 'bursty' | 'steady' | 'low'
  hasAgentLoops: boolean
}): string {
  if (params.maxRequestDurationSeconds > 120 || params.hasAgentLoops) {
    return 'Long-Running Server -- agent loops and long requests need unlimited duration'
  }
  if (params.requiresWebSockets || params.needsInMemoryCache) {
    return 'Long-Running Server -- persistent connections and in-memory state need a server'
  }
  if (params.trafficPattern === 'bursty') {
    return 'Serverless Functions -- scales to zero, handles bursts automatically'
  }
  if (params.trafficPattern === 'low') {
    return 'Serverless Functions -- do not pay for idle time'
  }
  return 'Long-Running Server -- steady traffic benefits from always-on compute'
}
```

> **Beginner Note:** If you are unsure which deployment model to choose, start with a long-running server. It has the fewest limitations and gives you the most flexibility. You can always move to serverless later if cost or scaling becomes a concern. A long-running Node.js server on a platform like Railway or Fly.io costs $5-20/month and handles most use cases.

---

## Section 2: Building an API with Hono

### Why Hono?

Hono is a lightweight, fast web framework for TypeScript that runs on any JavaScript runtime -- Node.js, Bun, Deno, Cloudflare Workers, and more. It has Express-like ergonomics with better TypeScript support and a smaller footprint.

```typescript
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { logger } from 'hono/logger'
import { generateText, Output, streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Create the Hono application
const app = new Hono()

// Global middleware
app.use('*', cors())
app.use('*', logger())

// Health check endpoint
app.get('/health', c => {
  return c.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
  })
})

// Simple text generation endpoint
app.post('/api/generate', async c => {
  const body = await c.req.json()
  const { prompt, system, maxOutputTokens } = body

  if (!prompt) {
    return c.json({ error: 'prompt is required' }, 400)
  }

  try {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      prompt,
      system: system || 'You are a helpful assistant.',
      maxOutputTokens: maxOutputTokens || 1024,
    })

    return c.json({
      text: result.text,
      usage: {
        inputTokens: result.usage?.inputTokens,
        outputTokens: result.usage?.outputTokens,
      },
      finishReason: result.finishReason,
    })
  } catch (error) {
    console.error('Generation error:', error)
    return c.json({ error: 'Generation failed', message: (error as Error).message }, 500)
  }
})

// Structured output endpoint
app.post('/api/analyze', async c => {
  const body = await c.req.json()
  const { text, analysisType } = body

  if (!text) {
    return c.json({ error: 'text is required' }, 400)
  }

  try {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      output: Output.object({
        schema: z.object({
          summary: z.string().describe('Brief summary of the text'),
          sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']).describe('Overall sentiment'),
          keyPoints: z.array(z.string()).describe('Key points extracted from the text'),
          confidence: z.number().min(0).max(1).describe('Confidence in the analysis'),
        }),
      }),
      prompt: `Analyze this text: ${text}`,
      system: `Perform a ${analysisType || 'general'} analysis.`,
    })

    return c.json({
      analysis: result.output,
      usage: result.usage,
    })
  } catch (error) {
    console.error('Analysis error:', error)
    return c.json({ error: 'Analysis failed', message: (error as Error).message }, 500)
  }
})

// Export for the runtime
export default app

// For Node.js, start the server
// import { serve } from "@hono/node-server";
// serve({ fetch: app.fetch, port: 3000 });
```

### Request Validation Middleware

```typescript
import { Hono } from 'hono'
import { z, ZodSchema } from 'zod'

// Request validation middleware factory
function validateBody<T>(schema: ZodSchema<T>) {
  return async (c: any, next: () => Promise<void>): Promise<void | Response> => {
    try {
      const body = await c.req.json()
      const validated = schema.parse(body)
      c.set('validatedBody', validated)
      await next()
    } catch (error) {
      if (error instanceof z.ZodError) {
        return c.json(
          {
            error: 'Validation failed',
            details: error.errors.map(e => ({
              field: e.path.join('.'),
              message: e.message,
            })),
          },
          400
        )
      }
      return c.json({ error: 'Invalid request body' }, 400)
    }
  }
}

// Define request schemas
const generateRequestSchema = z.object({
  prompt: z.string().min(1).max(10000),
  system: z.string().max(5000).optional(),
  maxOutputTokens: z.int().min(1).max(4096).optional().default(1024),
  temperature: z.number().min(0).max(2).optional().default(0),
  model: z
    .enum(['mistral-small-latest', 'mistral-small-latest', 'claude-opus-4-20250514'])
    .optional()
    .default('mistral-small-latest'),
})

type GenerateRequest = z.infer<typeof generateRequestSchema>

const app = new Hono()

// Use validation middleware
app.post('/api/generate', validateBody(generateRequestSchema), async c => {
  const body = c.get('validatedBody') as GenerateRequest

  const result = await generateText({
    model: mistral(body.model),
    prompt: body.prompt,
    system: body.system,
    maxOutputTokens: body.maxOutputTokens,
    temperature: body.temperature,
  })

  return c.json({
    text: result.text,
    model: body.model,
    usage: result.usage,
  })
})
```

> **Advanced Note:** Hono supports type-safe routing with `hono/validator` for even tighter TypeScript integration. You can also use `hono/zod-validator` for automatic Zod schema validation that integrates with Hono's type system, giving you end-to-end type safety from the request schema to the handler function.

---

## Section 3: Streaming Endpoints

### Server-Sent Events (SSE)

LLM responses are naturally streamed -- the model generates tokens one at a time. Exposing this via SSE gives users a responsive experience where they see the response build incrementally.

```typescript
import { Hono } from 'hono'
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { stream } from 'hono/streaming'

const app = new Hono()

// SSE streaming endpoint
app.post('/api/stream', async c => {
  const body = await c.req.json()
  const { prompt, system, maxOutputTokens } = body

  if (!prompt) {
    return c.json({ error: 'prompt is required' }, 400)
  }

  // Set SSE headers
  c.header('Content-Type', 'text/event-stream')
  c.header('Cache-Control', 'no-cache')
  c.header('Connection', 'keep-alive')

  return stream(c, async stream => {
    try {
      const result = streamText({
        model: mistral('mistral-small-latest'),
        prompt,
        system: system || 'You are a helpful assistant.',
        maxOutputTokens: maxOutputTokens || 1024,
      })

      // Stream each text chunk as an SSE event
      for await (const chunk of result.textStream) {
        await stream.write(`data: ${JSON.stringify({ type: 'text', content: chunk })}\n\n`)
      }

      // Send final usage data
      const usage = await result.usage
      await stream.write(
        `data: ${JSON.stringify({
          type: 'done',
          usage: {
            inputTokens: usage?.inputTokens,
            outputTokens: usage?.outputTokens,
          },
        })}\n\n`
      )
    } catch (error) {
      await stream.write(
        `data: ${JSON.stringify({
          type: 'error',
          message: (error as Error).message,
        })}\n\n`
      )
    }
  })
})

// Streaming with tool calls
app.post('/api/stream-with-tools', async c => {
  const body = await c.req.json()
  const { prompt, system } = body

  c.header('Content-Type', 'text/event-stream')
  c.header('Cache-Control', 'no-cache')
  c.header('Connection', 'keep-alive')

  return stream(c, async stream => {
    try {
      const result = streamText({
        model: mistral('mistral-small-latest'),
        prompt,
        system: system || 'You are a helpful assistant with tools.',
        maxOutputTokens: 2048,
        tools: {
          calculator: {
            description: 'Perform mathematical calculations',
            parameters: z.object({
              expression: z.string().describe('Math expression to evaluate'),
            }),
            execute: async ({ expression }) => {
              // Safe math evaluation
              try {
                const sanitized = expression.replace(/[^0-9+\-*/().% ]/g, '')
                const result = Function(`"use strict"; return (${sanitized})`)()
                return { result: Number(result) }
              } catch {
                return { error: 'Invalid expression' }
              }
            },
          },
        },
      })

      // Stream text and tool call events
      for await (const part of result.fullStream) {
        switch (part.type) {
          case 'text-delta':
            await stream.write(
              `data: ${JSON.stringify({
                type: 'text',
                content: part.textDelta,
              })}\n\n`
            )
            break
          case 'tool-call':
            await stream.write(
              `data: ${JSON.stringify({
                type: 'tool_call',
                toolName: part.toolName,
                args: part.args,
              })}\n\n`
            )
            break
          case 'tool-result':
            await stream.write(
              `data: ${JSON.stringify({
                type: 'tool_result',
                toolName: part.toolName,
                result: part.result,
              })}\n\n`
            )
            break
          case 'finish':
            await stream.write(
              `data: ${JSON.stringify({
                type: 'done',
                finishReason: part.finishReason,
                usage: part.usage,
              })}\n\n`
            )
            break
        }
      }
    } catch (error) {
      await stream.write(
        `data: ${JSON.stringify({
          type: 'error',
          message: (error as Error).message,
        })}\n\n`
      )
    }
  })
})
```

### Client-Side SSE Consumption

```typescript
// Client-side code to consume SSE stream
async function consumeStream(
  prompt: string,
  onChunk: (text: string) => void,
  onDone: (usage: { inputTokens: number; outputTokens: number }) => void,
  onError: (error: string) => void
): Promise<void> {
  const response = await fetch('http://localhost:3000/api/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: 'Bearer your-api-key',
    },
    body: JSON.stringify({ prompt }),
  })

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Parse SSE events from the buffer
    const lines = buffer.split('\n')
    buffer = lines.pop() || '' // Keep incomplete line in buffer

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6))

        switch (data.type) {
          case 'text':
            onChunk(data.content)
            break
          case 'done':
            onDone(data.usage)
            break
          case 'error':
            onError(data.message)
            break
        }
      }
    }
  }
}

// Usage
let fullResponse = ''
await consumeStream(
  'Explain quantum computing in simple terms',
  chunk => {
    fullResponse += chunk
    process.stdout.write(chunk) // Print incrementally
  },
  usage => {
    console.log('\n\nStream complete:', usage)
  },
  error => {
    console.error('Stream error:', error)
  }
)
```

> **Beginner Note:** Server-Sent Events (SSE) are a simple, HTTP-based protocol for one-way streaming from server to client. Unlike WebSockets, SSE works over regular HTTP, is automatically reconnected by browsers, and passes through proxies and load balancers without special configuration. For LLM applications where the server streams tokens to the client, SSE is the ideal protocol.

---

## Section 4: Authentication & API Keys

### API Key Authentication

```typescript
import { Hono } from 'hono'
import { bearerAuth } from 'hono/bearer-auth'

// API key store (in production, use a database or secrets manager)
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

class APIKeyStore {
  private keys: Map<string, APIKey> = new Map()

  constructor() {
    // Seed with example keys (in production, these come from a database)
    this.addKey({
      key: 'sk-test-free-user',
      userId: 'user-free-1',
      name: 'Free Tier Key',
      tier: 'free',
      createdAt: new Date().toISOString(),
      rateLimit: 10,
      dailyBudgetUsd: 1.0,
      active: true,
    })
    this.addKey({
      key: 'sk-test-pro-user',
      userId: 'user-pro-1',
      name: 'Pro Tier Key',
      tier: 'pro',
      createdAt: new Date().toISOString(),
      rateLimit: 60,
      dailyBudgetUsd: 25.0,
      active: true,
    })
    this.addKey({
      key: 'sk-test-enterprise',
      userId: 'user-ent-1',
      name: 'Enterprise Key',
      tier: 'enterprise',
      createdAt: new Date().toISOString(),
      rateLimit: 300,
      dailyBudgetUsd: 500.0,
      active: true,
    })
  }

  addKey(key: APIKey): void {
    this.keys.set(key.key, key)
  }

  validate(key: string): APIKey | null {
    const apiKey = this.keys.get(key)
    if (!apiKey || !apiKey.active) return null

    // Update last used timestamp
    apiKey.lastUsedAt = new Date().toISOString()
    return apiKey
  }

  getUserKeys(userId: string): APIKey[] {
    return Array.from(this.keys.values()).filter(k => k.userId === userId)
  }

  revokeKey(key: string): boolean {
    const apiKey = this.keys.get(key)
    if (!apiKey) return false
    apiKey.active = false
    return true
  }
}

// Authentication middleware
const keyStore = new APIKeyStore()

function authMiddleware() {
  return async (c: any, next: () => Promise<void>): Promise<void | Response> => {
    const authHeader = c.req.header('Authorization')

    if (!authHeader) {
      return c.json({ error: 'Missing Authorization header' }, 401)
    }

    // Support both "Bearer <key>" and raw key
    const key = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : authHeader

    const apiKey = keyStore.validate(key)
    if (!apiKey) {
      return c.json({ error: 'Invalid or revoked API key' }, 401)
    }

    // Attach user context to request
    c.set('apiKey', apiKey)
    c.set('userId', apiKey.userId)
    c.set('tier', apiKey.tier)

    await next()
  }
}

// JWT authentication for web applications
interface JWTPayload {
  userId: string
  email: string
  tier: string
  exp: number
  iat: number
}

function jwtMiddleware(secret: string) {
  return async (c: any, next: () => Promise<void>): Promise<void | Response> => {
    const authHeader = c.req.header('Authorization')

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return c.json({ error: 'Missing or invalid Authorization header' }, 401)
    }

    const token = authHeader.slice(7)

    try {
      // In production, use a proper JWT library like jose
      // This is a simplified example
      const parts = token.split('.')
      if (parts.length !== 3) {
        return c.json({ error: 'Invalid token format' }, 401)
      }

      const payload = JSON.parse(Buffer.from(parts[1], 'base64url').toString()) as JWTPayload

      // Check expiration
      if (payload.exp < Date.now() / 1000) {
        return c.json({ error: 'Token expired' }, 401)
      }

      c.set('userId', payload.userId)
      c.set('email', payload.email)
      c.set('tier', payload.tier)

      await next()
    } catch {
      return c.json({ error: 'Invalid token' }, 401)
    }
  }
}

// Apply auth to API routes
const app = new Hono()

// Public routes (no auth)
app.get('/health', c => c.json({ status: 'ok' }))

// Protected routes
const api = new Hono()
api.use('*', authMiddleware())

api.post('/generate', async c => {
  const apiKey = c.get('apiKey') as APIKey
  const body = await c.req.json()

  // Use tier-based model selection
  const modelForTier: Record<string, string> = {
    free: 'mistral-small-latest',
    pro: 'mistral-small-latest',
    enterprise: 'claude-opus-4-20250514',
  }

  const model = modelForTier[apiKey.tier] || 'mistral-small-latest'

  const result = await generateText({
    model: mistral(model),
    prompt: body.prompt,
    maxOutputTokens: body.maxOutputTokens || 1024,
  })

  return c.json({
    text: result.text,
    model,
    tier: apiKey.tier,
    usage: result.usage,
  })
})

app.route('/api', api)
```

---

## Section 5: Multi-Tenancy

### Per-User Data Isolation

Multi-tenant LLM applications must isolate user data so that one user's documents, conversations, and embeddings are never visible to another user.

```typescript
import { Hono } from 'hono'
import { generateText, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Tenant-isolated data store
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

class TenantStore {
  private tenants: Map<string, TenantData> = new Map()

  // Get or create tenant data
  getTenant(userId: string): TenantData {
    if (!this.tenants.has(userId)) {
      this.tenants.set(userId, {
        userId,
        documents: [],
        conversations: [],
        settings: {
          defaultModel: 'mistral-small-latest',
          systemPrompt: 'You are a helpful assistant.',
          maxTokensPerRequest: 1024,
        },
      })
    }
    return this.tenants.get(userId)!
  }

  // Add a document for a specific tenant
  addDocument(userId: string, doc: { id: string; content: string; metadata: Record<string, string> }): void {
    const tenant = this.getTenant(userId)
    tenant.documents.push({ ...doc })
  }

  // Search documents for a specific tenant only
  searchDocuments(
    userId: string,
    queryEmbedding: number[],
    limit: number = 5
  ): Array<{
    id: string
    content: string
    score: number
    metadata: Record<string, string>
  }> {
    const tenant = this.getTenant(userId)

    return tenant.documents
      .filter(doc => doc.embedding)
      .map(doc => ({
        id: doc.id,
        content: doc.content,
        score: cosineSimilarity(queryEmbedding, doc.embedding!),
        metadata: doc.metadata,
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
  }

  // Get conversation history for a specific tenant
  getConversation(userId: string, conversationId: string): TenantData['conversations'][0] | undefined {
    const tenant = this.getTenant(userId)
    return tenant.conversations.find(c => c.id === conversationId)
  }

  // Create or update conversation
  upsertConversation(
    userId: string,
    conversationId: string,
    messages: Array<{ role: 'user' | 'assistant'; content: string }>
  ): void {
    const tenant = this.getTenant(userId)
    const existing = tenant.conversations.find(c => c.id === conversationId)

    if (existing) {
      existing.messages = messages
      existing.updatedAt = new Date().toISOString()
    } else {
      tenant.conversations.push({
        id: conversationId,
        messages,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      })
    }
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}

// Multi-tenant API routes
const tenantStore = new TenantStore()

const app = new Hono()
app.use('*', authMiddleware()) // Auth middleware from Section 4

// Upload document (tenant-isolated)
app.post('/api/documents', async c => {
  const userId = c.get('userId') as string
  const body = await c.req.json()
  const { content, metadata } = body

  // Generate embedding for the document
  const { embedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: content,
  })

  const docId = `doc-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`

  tenantStore.addDocument(userId, {
    id: docId,
    content,
    metadata: metadata || {},
  })

  // Store embedding separately
  const tenant = tenantStore.getTenant(userId)
  const doc = tenant.documents.find(d => d.id === docId)
  if (doc) doc.embedding = embedding

  return c.json({ id: docId, message: 'Document uploaded' })
})

// Query with RAG (tenant-isolated)
app.post('/api/query', async c => {
  const userId = c.get('userId') as string
  const body = await c.req.json()
  const { question, conversationId } = body

  // Embed the question
  const { embedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: question,
  })

  // Search ONLY this tenant's documents
  const relevantDocs = tenantStore.searchDocuments(userId, embedding, 3)

  const tenant = tenantStore.getTenant(userId)
  const context = relevantDocs.map(d => d.content).join('\n\n---\n\n')

  // Get conversation history if continuing a conversation
  const conversation = conversationId ? tenantStore.getConversation(userId, conversationId) : undefined

  const result = await generateText({
    model: mistral(tenant.settings.defaultModel),
    system: tenant.settings.systemPrompt,
    messages: [
      ...(conversation?.messages || []),
      {
        role: 'user' as const,
        content: `Context:\n${context}\n\nQuestion: ${question}`,
      },
    ],
    maxOutputTokens: tenant.settings.maxTokensPerRequest,
  })

  // Save to conversation history
  const convId = conversationId || `conv-${Date.now()}`
  const messages = [
    ...(conversation?.messages || []),
    { role: 'user' as const, content: question },
    { role: 'assistant' as const, content: result.text },
  ]
  tenantStore.upsertConversation(userId, convId, messages)

  return c.json({
    answer: result.text,
    conversationId: convId,
    sources: relevantDocs.map(d => ({
      id: d.id,
      score: d.score,
      preview: d.content.substring(0, 200),
    })),
    usage: result.usage,
  })
})
```

> **Beginner Note:** Multi-tenancy means serving multiple users (tenants) from the same application while keeping their data separate. The simplest approach is to add a `userId` filter to every database query. Never assume the user ID from the request body -- always derive it from the authenticated session or API key. A common security bug is allowing users to specify another user's ID in the request.

---

## Section 6: Rate Limiting

### Multi-Layer Rate Limiting

LLM applications need rate limiting at multiple levels to prevent abuse and control costs.

```typescript
import { Hono } from 'hono'

// Rate limiter with multiple strategies
interface RateLimitConfig {
  windowMs: number // Time window in milliseconds
  maxRequests: number // Max requests per window
  maxTokensPerWindow?: number // Max tokens per window
  keyPrefix: string
}

class RateLimiter {
  private windows: Map<string, { count: number; tokens: number; resetAt: number }> = new Map()

  constructor(private config: RateLimitConfig) {}

  // Check if a request is allowed
  check(
    key: string,
    estimatedTokens: number = 0
  ): {
    allowed: boolean
    remaining: number
    resetAt: number
    retryAfterMs?: number
  } {
    const windowKey = `${this.config.keyPrefix}:${key}`
    const now = Date.now()

    let window = this.windows.get(windowKey)

    // Create new window if expired or missing
    if (!window || now >= window.resetAt) {
      window = {
        count: 0,
        tokens: 0,
        resetAt: now + this.config.windowMs,
      }
      this.windows.set(windowKey, window)
    }

    // Check request count limit
    if (window.count >= this.config.maxRequests) {
      return {
        allowed: false,
        remaining: 0,
        resetAt: window.resetAt,
        retryAfterMs: window.resetAt - now,
      }
    }

    // Check token limit if configured
    if (this.config.maxTokensPerWindow && window.tokens + estimatedTokens > this.config.maxTokensPerWindow) {
      return {
        allowed: false,
        remaining: 0,
        resetAt: window.resetAt,
        retryAfterMs: window.resetAt - now,
      }
    }

    // Allow the request
    window.count++
    window.tokens += estimatedTokens

    return {
      allowed: true,
      remaining: this.config.maxRequests - window.count,
      resetAt: window.resetAt,
    }
  }

  // Record actual token usage after the request completes
  recordTokens(key: string, actualTokens: number): void {
    const windowKey = `${this.config.keyPrefix}:${key}`
    const window = this.windows.get(windowKey)
    if (window) {
      window.tokens += actualTokens
    }
  }
}

// Per-tier rate limit configurations
const rateLimitsByTier: Record<string, RateLimitConfig> = {
  free: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 10,
    maxTokensPerWindow: 10_000,
    keyPrefix: 'free',
  },
  pro: {
    windowMs: 60 * 1000,
    maxRequests: 60,
    maxTokensPerWindow: 100_000,
    keyPrefix: 'pro',
  },
  enterprise: {
    windowMs: 60 * 1000,
    maxRequests: 300,
    maxTokensPerWindow: 1_000_000,
    keyPrefix: 'enterprise',
  },
}

// Global rate limiter (across all users)
const globalLimiter = new RateLimiter({
  windowMs: 60 * 1000,
  maxRequests: 1000,
  keyPrefix: 'global',
})

// Per-user rate limiters
const userLimiters: Map<string, RateLimiter> = new Map()

function getUserLimiter(tier: string): RateLimiter {
  if (!userLimiters.has(tier)) {
    userLimiters.set(tier, new RateLimiter(rateLimitsByTier[tier] || rateLimitsByTier.free))
  }
  return userLimiters.get(tier)!
}

// Rate limiting middleware
function rateLimitMiddleware() {
  return async (c: any, next: () => Promise<void>): Promise<void | Response> => {
    const userId = c.get('userId') as string
    const tier = (c.get('tier') as string) || 'free'

    // Check global rate limit
    const globalCheck = globalLimiter.check('all')
    if (!globalCheck.allowed) {
      c.header('Retry-After', String(Math.ceil((globalCheck.retryAfterMs || 0) / 1000)))
      return c.json({ error: 'Service rate limit exceeded. Please try again later.' }, 429)
    }

    // Check per-user rate limit
    const userLimiter = getUserLimiter(tier)
    const userCheck = userLimiter.check(userId)
    if (!userCheck.allowed) {
      c.header('Retry-After', String(Math.ceil((userCheck.retryAfterMs || 0) / 1000)))
      c.header('X-RateLimit-Remaining', String(userCheck.remaining))
      c.header('X-RateLimit-Reset', new Date(userCheck.resetAt).toISOString())
      return c.json(
        {
          error: 'Rate limit exceeded',
          tier,
          retryAfterMs: userCheck.retryAfterMs,
          limit: rateLimitsByTier[tier]?.maxRequests || 10,
        },
        429
      )
    }

    // Add rate limit headers to response
    c.header('X-RateLimit-Remaining', String(userCheck.remaining))
    c.header('X-RateLimit-Reset', new Date(userCheck.resetAt).toISOString())

    await next()
  }
}

// Apply to API
const app = new Hono()
const api = new Hono()
api.use('*', authMiddleware())
api.use('*', rateLimitMiddleware())

api.post('/generate', async c => {
  // Handler implementation...
  const body = await c.req.json()
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: body.prompt,
  })

  return c.json({ text: result.text, usage: result.usage })
})

app.route('/api', api)
```

> **Advanced Note:** In production, use Redis or a similar distributed store for rate limiting state instead of in-memory maps. In-memory rate limiters reset when the server restarts and do not work across multiple server instances. Redis-based rate limiters (e.g., using the sliding window algorithm) provide consistent rate limiting across all instances and survive restarts.

---

## Section 7: Provider Failover

### Automatic Provider Switching

LLM providers occasionally experience outages, rate limiting, or degraded performance. A failover system automatically routes requests to backup providers when the primary is unavailable.

```typescript
import { generateText, streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { openai } from '@ai-sdk/openai'
import { google } from '@ai-sdk/google'

// Provider health tracking
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
  private healthMap: Map<string, ProviderHealth> = new Map()
  private circuitBreakerThreshold: number = 5 // Consecutive errors to open circuit
  private circuitBreakerResetMs: number = 60_000 // 1 minute cooldown

  // Define provider chain with model mappings
  private providerChain: Array<{
    name: string
    createModel: (modelId: string) => any
    modelMapping: Record<string, string>
    priority: number
  }> = [
    {
      name: 'mistral',
      createModel: (id: string) => mistral(id),
      modelMapping: {
        standard: 'mistral-small-latest',
        premium: 'mistral-large-latest',
        economy: 'mistral-small-latest',
      },
      priority: 1,
    },
    {
      name: 'openai',
      createModel: (id: string) => openai(id),
      modelMapping: {
        standard: 'gpt-5.4',
        premium: 'gpt-5.4',
        economy: 'gpt-5-mini',
      },
      priority: 2,
    },
    {
      name: 'google',
      createModel: (id: string) => google(id),
      modelMapping: {
        standard: 'gemini-2.0-flash',
        premium: 'gemini-2.5-pro',
        economy: 'gemini-2.0-flash',
      },
      priority: 3,
    },
  ]

  constructor() {
    // Initialize health for all providers
    for (const provider of this.providerChain) {
      this.healthMap.set(provider.name, {
        provider: provider.name,
        healthy: true,
        consecutiveErrors: 0,
        totalErrors: 0,
        totalRequests: 0,
        avgLatencyMs: 0,
        circuitBreakerOpen: false,
      })
    }
  }

  // Get an ordered list of available providers
  private getAvailableProviders(): typeof this.providerChain {
    const now = Date.now()

    return this.providerChain.filter(provider => {
      const health = this.healthMap.get(provider.name)!

      // Check circuit breaker
      if (health.circuitBreakerOpen) {
        if (health.circuitBreakerResetAt && now >= health.circuitBreakerResetAt) {
          // Reset circuit breaker -- allow one request through
          health.circuitBreakerOpen = false
          health.consecutiveErrors = 0
          return true
        }
        return false
      }

      return true
    })
  }

  // Record a successful request
  private recordSuccess(providerName: string, latencyMs: number): void {
    const health = this.healthMap.get(providerName)!
    health.healthy = true
    health.consecutiveErrors = 0
    health.totalRequests++
    health.lastSuccessAt = Date.now()
    health.avgLatencyMs = (health.avgLatencyMs * (health.totalRequests - 1) + latencyMs) / health.totalRequests
  }

  // Record a failed request
  private recordFailure(providerName: string, error: string): void {
    const health = this.healthMap.get(providerName)!
    health.consecutiveErrors++
    health.totalErrors++
    health.totalRequests++
    health.lastError = error
    health.lastErrorAt = Date.now()

    // Open circuit breaker if threshold exceeded
    if (health.consecutiveErrors >= this.circuitBreakerThreshold) {
      health.circuitBreakerOpen = true
      health.circuitBreakerResetAt = Date.now() + this.circuitBreakerResetMs
      health.healthy = false
      console.warn(
        `Circuit breaker OPEN for ${providerName}: ` +
          `${health.consecutiveErrors} consecutive errors. ` +
          `Reset at ${new Date(health.circuitBreakerResetAt).toISOString()}`
      )
    }
  }

  // Generate text with automatic failover
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
  }> {
    const availableProviders = this.getAvailableProviders()

    if (availableProviders.length === 0) {
      throw new Error('All providers are currently unavailable. Please try again later.')
    }

    let lastError: Error | null = null
    let attempts = 0

    for (const provider of availableProviders) {
      attempts++
      const modelId = provider.modelMapping[params.tier]
      const startTime = Date.now()

      try {
        const result = await generateText({
          model: provider.createModel(modelId),
          prompt: params.prompt,
          system: params.system,
          maxOutputTokens: params.maxOutputTokens,
        })

        const latencyMs = Date.now() - startTime
        this.recordSuccess(provider.name, latencyMs)

        return {
          text: result.text,
          provider: provider.name,
          model: modelId,
          usage: result.usage,
          failoverAttempts: attempts,
        }
      } catch (error) {
        lastError = error as Error
        this.recordFailure(provider.name, lastError.message)
        console.warn(`Provider ${provider.name} failed: ${lastError.message}. ` + `Trying next provider...`)
      }
    }

    throw new Error(`All providers failed. Last error: ${lastError?.message}`)
  }

  // Get health status for all providers
  getHealthStatus(): ProviderHealth[] {
    return Array.from(this.healthMap.values())
  }
}

// Use in API routes
const failover = new ProviderFailover()

const app = new Hono()

app.post('/api/generate', async c => {
  const body = await c.req.json()
  const tier = (c.get('tier') as string) || 'standard'

  const result = await failover.generateWithFailover({
    tier: tier as 'economy' | 'standard' | 'premium',
    prompt: body.prompt,
    system: body.system,
    maxOutputTokens: body.maxOutputTokens,
  })

  return c.json({
    text: result.text,
    provider: result.provider,
    model: result.model,
    usage: result.usage,
    failoverAttempts: result.failoverAttempts,
  })
})

// Provider health endpoint
app.get('/api/providers/health', c => {
  return c.json({ providers: failover.getHealthStatus() })
})
```

> **Beginner Note:** A circuit breaker is a pattern from electrical engineering applied to software. When a provider fails repeatedly, the circuit breaker "opens" and stops sending requests to it for a cooldown period. This prevents wasting time on a known-bad provider and gives it time to recover. After the cooldown, it lets one request through to test if the provider has recovered.

---

## Section 8: Scaling Considerations

### Concurrency and Queue-Based Processing

LLM requests are slow (2-30 seconds) and expensive. You cannot handle them the same way you handle traditional web requests. Queue-based processing allows you to control concurrency, implement backpressure, and handle bursts.

```typescript
// Request queue for controlled concurrency
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
  private queue: QueuedRequest[] = []
  private activeRequests: number = 0
  private maxConcurrent: number
  private processingInterval: ReturnType<typeof setInterval> | null = null

  constructor(maxConcurrent: number = 10) {
    this.maxConcurrent = maxConcurrent
  }

  // Add a request to the queue
  enqueue(params: {
    userId: string
    tier: string
    prompt: string
    system?: string
    maxOutputTokens?: number
    timeoutMs?: number
  }): Promise<any> {
    return new Promise((resolve, reject) => {
      // Priority based on tier
      const priorityMap: Record<string, number> = {
        enterprise: 1,
        pro: 2,
        free: 3,
      }

      const request: QueuedRequest = {
        id: `req-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`,
        userId: params.userId,
        tier: params.tier,
        prompt: params.prompt,
        system: params.system,
        maxOutputTokens: params.maxOutputTokens,
        resolve,
        reject,
        queuedAt: Date.now(),
        priority: priorityMap[params.tier] || 3,
        timeoutMs: params.timeoutMs || 30_000,
      }

      this.queue.push(request)

      // Sort by priority (lower = higher priority)
      this.queue.sort((a, b) => a.priority - b.priority)

      // Start processing if not already running
      this.processQueue()
    })
  }

  // Process queued requests up to concurrency limit
  private async processQueue(): Promise<void> {
    while (this.queue.length > 0 && this.activeRequests < this.maxConcurrent) {
      const request = this.queue.shift()
      if (!request) break

      // Check if request has timed out while waiting in queue
      if (Date.now() - request.queuedAt > request.timeoutMs) {
        request.reject(new Error(`Request timed out after ${request.timeoutMs}ms in queue`))
        continue
      }

      this.activeRequests++

      // Process without awaiting -- let it run concurrently
      this.processRequest(request).finally(() => {
        this.activeRequests--
        // Continue processing queue
        this.processQueue()
      })
    }
  }

  private async processRequest(request: QueuedRequest): Promise<void> {
    try {
      const result = await generateText({
        model: mistral('mistral-small-latest'),
        prompt: request.prompt,
        system: request.system,
        maxOutputTokens: request.maxOutputTokens,
      })

      request.resolve({
        text: result.text,
        usage: result.usage,
        queueTimeMs: Date.now() - request.queuedAt,
      })
    } catch (error) {
      request.reject(error as Error)
    }
  }

  // Get queue status
  getStatus(): {
    queueLength: number
    activeRequests: number
    maxConcurrent: number
  } {
    return {
      queueLength: this.queue.length,
      activeRequests: this.activeRequests,
      maxConcurrent: this.maxConcurrent,
    }
  }
}

// Use queue in the API
const requestQueue = new RequestQueue(10) // Max 10 concurrent LLM requests

const app = new Hono()

app.post('/api/generate', async c => {
  const body = await c.req.json()
  const userId = c.get('userId') as string
  const tier = (c.get('tier') as string) || 'free'

  // Check queue depth for backpressure
  const status = requestQueue.getStatus()
  if (status.queueLength > 100) {
    return c.json(
      {
        error: 'Service is busy. Please try again later.',
        queueDepth: status.queueLength,
      },
      503
    )
  }

  try {
    const result = await requestQueue.enqueue({
      userId,
      tier,
      prompt: body.prompt,
      system: body.system,
      maxOutputTokens: body.maxOutputTokens,
      timeoutMs: 30_000,
    })

    return c.json(result)
  } catch (error) {
    return c.json({ error: (error as Error).message }, (error as Error).message.includes('timed out') ? 504 : 500)
  }
})

// Queue status endpoint (admin only)
app.get('/api/admin/queue', c => {
  return c.json(requestQueue.getStatus())
})
```

---

## Section 9: Environment Configuration

### Secrets Management

LLM applications need multiple API keys (one per provider), database credentials, and other secrets. Managing these securely is critical.

```typescript
import { z } from 'zod'

// Environment variable schema with validation
const envSchema = z.object({
  // Server config
  PORT: z.string().default('3000'),
  NODE_ENV: z.enum(['development', 'staging', 'production']).default('development'),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // LLM Provider API Keys
  MISTRAL_API_KEY: z.string().min(1, 'MISTRAL_API_KEY is required'),
  GROQ_API_KEY: z.string().optional(), // Optional -- for failover
  ANTHROPIC_API_KEY: z.string().optional(), // Optional -- for failover
  OPENAI_API_KEY: z.string().optional(), // Optional -- for failover

  // Database
  DATABASE_URL: z.string().optional(),
  REDIS_URL: z.string().optional(),

  // Vector store
  PINECONE_API_KEY: z.string().optional(),
  PINECONE_INDEX: z.string().optional(),

  // Auth
  JWT_SECRET: z.string().min(32).optional(),
  API_KEY_SALT: z.string().min(16).optional(),

  // Rate limiting
  RATE_LIMIT_REQUESTS_PER_MINUTE: z.string().default('60'),
  RATE_LIMIT_TOKENS_PER_MINUTE: z.string().default('100000'),

  // Cost controls
  DAILY_BUDGET_USD: z.string().default('100'),
  MAX_TOKENS_PER_REQUEST: z.string().default('4096'),

  // Observability
  OTEL_EXPORTER_ENDPOINT: z.string().optional(),
  LOG_DESTINATION: z.enum(['console', 'file', 'remote']).default('console'),
})

type Env = z.infer<typeof envSchema>

// Validate and load environment
function loadEnvironment(): Env {
  try {
    const env = envSchema.parse(process.env)

    // Warn about missing optional keys
    if (!env.OPENAI_API_KEY) {
      console.warn('OPENAI_API_KEY not set -- provider failover to OpenAI will not work')
    }
    if (!env.GOOGLE_API_KEY) {
      console.warn('GOOGLE_API_KEY not set -- provider failover to Google will not work')
    }

    // Log configuration (without secrets)
    console.log('Environment loaded:', {
      NODE_ENV: env.NODE_ENV,
      PORT: env.PORT,
      LOG_LEVEL: env.LOG_LEVEL,
      hasMistralKey: !!env.MISTRAL_API_KEY,
      hasGroqKey: !!env.GROQ_API_KEY,
      hasAnthropicKey: !!env.ANTHROPIC_API_KEY,
      hasOpenAIKey: !!env.OPENAI_API_KEY,
      hasDatabase: !!env.DATABASE_URL,
      hasRedis: !!env.REDIS_URL,
      dailyBudget: env.DAILY_BUDGET_USD,
    })

    return env
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('Environment validation failed:')
      for (const issue of error.errors) {
        console.error(`  ${issue.path.join('.')}: ${issue.message}`)
      }
      process.exit(1)
    }
    throw error
  }
}

// Configuration object derived from environment
function createConfig(env: Env) {
  return {
    server: {
      port: parseInt(env.PORT),
      environment: env.NODE_ENV,
      logLevel: env.LOG_LEVEL,
    },
    providers: {
      mistral: {
        apiKey: env.MISTRAL_API_KEY,
        enabled: true,
      },
      groq: {
        apiKey: env.GROQ_API_KEY,
        enabled: !!env.GROQ_API_KEY,
      },
      anthropic: {
        apiKey: env.ANTHROPIC_API_KEY,
        enabled: !!env.ANTHROPIC_API_KEY,
      },
      openai: {
        apiKey: env.OPENAI_API_KEY,
        enabled: !!env.OPENAI_API_KEY,
      },
      google: {
        apiKey: env.GOOGLE_API_KEY,
        enabled: !!env.GOOGLE_API_KEY,
      },
    },
    rateLimits: {
      requestsPerMinute: parseInt(env.RATE_LIMIT_REQUESTS_PER_MINUTE),
      tokensPerMinute: parseInt(env.RATE_LIMIT_TOKENS_PER_MINUTE),
    },
    costControls: {
      dailyBudgetUsd: parseFloat(env.DAILY_BUDGET_USD),
      maxTokensPerRequest: parseInt(env.MAX_TOKENS_PER_REQUEST),
    },
    observability: {
      otelEndpoint: env.OTEL_EXPORTER_ENDPOINT,
      logDestination: env.LOG_DESTINATION,
    },
  }
}
```

### Putting It All Together: Complete Server

```typescript
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serve } from '@hono/node-server'
import { generateText, streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { stream } from 'hono/streaming'

// Load environment and create config
const env = loadEnvironment()
const config = createConfig(env)

// Initialize components
const app = new Hono()

// Global middleware
app.use('*', cors())

// Health check
app.get('/health', c => {
  return c.json({
    status: 'healthy',
    version: '1.0.0',
    environment: config.server.environment,
    timestamp: new Date().toISOString(),
  })
})

// Protected API routes
const api = new Hono()
api.use('*', authMiddleware())
api.use('*', rateLimitMiddleware())

// Text generation
api.post('/generate', async c => {
  const body = await c.req.json()
  const tier = c.get('tier') as string

  const result = await failover.generateWithFailover({
    tier: (tier as 'economy' | 'standard' | 'premium') || 'standard',
    prompt: body.prompt,
    system: body.system,
    maxOutputTokens: Math.min(body.maxOutputTokens || 1024, config.costControls.maxTokensPerRequest),
  })

  return c.json(result)
})

// Streaming generation
api.post('/stream', async c => {
  const body = await c.req.json()

  c.header('Content-Type', 'text/event-stream')
  c.header('Cache-Control', 'no-cache')
  c.header('Connection', 'keep-alive')

  return stream(c, async s => {
    const result = streamText({
      model: mistral('mistral-small-latest'),
      prompt: body.prompt,
      system: body.system,
      maxOutputTokens: body.maxOutputTokens || 1024,
    })

    for await (const chunk of result.textStream) {
      await s.write(`data: ${JSON.stringify({ type: 'text', content: chunk })}\n\n`)
    }

    const usage = await result.usage
    await s.write(`data: ${JSON.stringify({ type: 'done', usage })}\n\n`)
  })
})

// Provider health
api.get('/providers/health', c => {
  return c.json({ providers: failover.getHealthStatus() })
})

// Mount API routes
app.route('/api', api)

// Start server
serve({ fetch: app.fetch, port: config.server.port }, info => {
  console.log(`Server running at http://localhost:${info.port} [${config.server.environment}]`)
})
```

> **Beginner Note:** Never commit API keys or secrets to version control. Use `.env` files for local development (added to `.gitignore`) and your deployment platform's secrets manager for production (e.g., Vercel Environment Variables, Railway Variables, AWS Secrets Manager). The Zod validation approach shown here ensures your application fails fast with a clear error message if a required secret is missing, rather than failing mysteriously at runtime.

> **Advanced Note:** For production deployments, consider using a secrets rotation strategy where API keys are automatically rotated on a schedule. This limits the blast radius of a leaked key. Also implement key scoping -- create separate API keys per environment (dev, staging, production) with appropriate permissions and spending limits.

> **Local Alternative (Ollama):** The Hono server, authentication, rate limiting, and API patterns work with any model provider. For deployment, you can run Ollama alongside your Hono server on the same machine or a private server. This gives you a fully self-hosted LLM application with no external API dependencies — ideal for privacy-sensitive or air-gapped deployments.

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
