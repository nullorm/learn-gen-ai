# Module 22: Cost Optimization

## Learning Objectives

- Understand the complete cost structure of LLM applications including input tokens, output tokens, and per-model pricing
- Implement semantic caching to avoid redundant LLM calls for similar queries
- Build model routing systems that match query complexity to the appropriate model tier
- Enforce token budgets per request, per user, and per day
- Batch multiple queries into single requests for efficiency
- Design fallback chains that try cheaper models first and escalate only when needed
- Optimize prompts for cost without sacrificing quality
- Build monitoring and alerting systems that track spend and enforce budgets

---

## Why Should I Care?

LLM API costs can escalate from manageable to alarming overnight. A prototype that costs $5 per day during development can cost $5,000 per day in production. A single poorly designed prompt that wastes 1,000 tokens per call costs nothing when you make 10 calls during testing, but costs real money when you make 100,000 calls per day.

The economics of LLM applications are fundamentally different from traditional software. In traditional software, the marginal cost of serving an additional request is near zero -- the server is already running. In LLM applications, every request has a direct cost proportional to the tokens processed. More users, more tokens, more money.

Cost optimization is not about being cheap. It is about building a sustainable business. The goal is to deliver the same quality at lower cost, or to deliver higher quality within the same budget. Every dollar saved on unnecessary tokens is a dollar you can spend on more test cases, better monitoring, or additional features.

This module teaches practical techniques that can reduce LLM costs by 50-90% without degrading user experience.

---

## Connection to Other Modules

- **Module 1 (Setup)** introduces the provider and model selection that drives pricing.
- **Module 2 (Prompt Engineering)** creates the prompts you will optimize for cost.
- **Module 9-10 (RAG)** benefits from caching and smart retrieval to reduce redundant calls.
- **Module 19 (Evals)** provides the measurement framework to ensure cost optimization does not degrade quality.
- **Module 20 (Fine-tuning)** is a cost optimization strategy itself -- shorter prompts via fine-tuned models.
- **Module 21 (Safety)** connects through rate limiting and abuse prevention.

---

## Section 1: Understanding LLM Costs

### Token Pricing Models

LLM APIs charge based on tokens processed. Understanding the pricing structure is essential for optimization.

```typescript
import { generateText, Output, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Model pricing database (prices per million tokens, as of 2025)
interface ModelPricing {
  provider: string
  model: string
  inputPricePerMillion: number
  outputPricePerMillion: number
  contextWindow: number
  tier: 'economy' | 'standard' | 'premium'
}

const modelPricing: ModelPricing[] = [
  {
    provider: 'mistral',
    model: 'mistral-small-latest',
    inputPricePerMillion: 0.1,
    outputPricePerMillion: 0.3,
    contextWindow: 128_000,
    tier: 'standard',
  },
  {
    provider: 'mistral',
    model: 'mistral-large-latest',
    inputPricePerMillion: 2.0,
    outputPricePerMillion: 6.0,
    contextWindow: 128_000,
    tier: 'premium',
  },
  {
    provider: 'mistral',
    model: 'mistral-small-latest',
    inputPricePerMillion: 0.1,
    outputPricePerMillion: 0.3,
    contextWindow: 128_000,
    tier: 'economy',
  },
  {
    provider: 'openai',
    model: 'gpt-5.4',
    inputPricePerMillion: 2.5,
    outputPricePerMillion: 10.0,
    contextWindow: 128_000,
    tier: 'standard',
  },
  {
    provider: 'openai',
    model: 'gpt-5-mini',
    inputPricePerMillion: 0.15,
    outputPricePerMillion: 0.6,
    contextWindow: 128_000,
    tier: 'economy',
  },
]

// Calculate cost for a single request
function calculateRequestCost(
  model: ModelPricing,
  inputTokens: number,
  outputTokens: number
): {
  inputCost: number
  outputCost: number
  totalCost: number
} {
  const inputCost = (inputTokens / 1_000_000) * model.inputPricePerMillion
  const outputCost = (outputTokens / 1_000_000) * model.outputPricePerMillion

  return {
    inputCost,
    outputCost,
    totalCost: inputCost + outputCost,
  }
}

// Project monthly costs
function projectMonthlyCost(
  model: ModelPricing,
  avgInputTokens: number,
  avgOutputTokens: number,
  requestsPerDay: number
): {
  dailyCost: number
  monthlyCost: number
  annualCost: number
  costPerRequest: number
  breakdown: { input: number; output: number }
} {
  const perRequest = calculateRequestCost(model, avgInputTokens, avgOutputTokens)

  const dailyCost = perRequest.totalCost * requestsPerDay
  const monthlyCost = dailyCost * 30
  const annualCost = dailyCost * 365

  return {
    dailyCost,
    monthlyCost,
    annualCost,
    costPerRequest: perRequest.totalCost,
    breakdown: {
      input: perRequest.inputCost * requestsPerDay * 30,
      output: perRequest.outputCost * requestsPerDay * 30,
    },
  }
}

// Compare costs across models
function compareModelCosts(avgInputTokens: number, avgOutputTokens: number, requestsPerDay: number): void {
  console.log('\n=== Model Cost Comparison ===')
  console.log(
    `Scenario: ${avgInputTokens} input tokens, ${avgOutputTokens} output tokens, ${requestsPerDay} requests/day\n`
  )

  for (const model of modelPricing) {
    const projection = projectMonthlyCost(model, avgInputTokens, avgOutputTokens, requestsPerDay)

    console.log(`${model.provider}/${model.model} (${model.tier}):`)
    console.log(`  Per request: $${projection.costPerRequest.toFixed(6)}`)
    console.log(`  Monthly:     $${projection.monthlyCost.toFixed(2)}`)
    console.log(`  Annual:      $${projection.annualCost.toFixed(2)}`)
    console.log()
  }
}

// Example: compare costs for a typical customer support scenario
compareModelCosts(1500, 300, 10_000)
```

> **Beginner Note:** LLM pricing has two components: input tokens (what you send to the model, including the system prompt) and output tokens (what the model generates). Output tokens are typically 3-5x more expensive than input tokens. This means controlling output length is often more impactful than reducing input length.

### Cost Tracking

Build instrumentation to track actual costs in real time.

```typescript
interface CostEvent {
  timestamp: string
  model: string
  inputTokens: number
  outputTokens: number
  cost: number
  userId?: string
  feature?: string
  cached: boolean
}

class CostTracker {
  private events: CostEvent[] = []
  private pricing: Map<string, ModelPricing> = new Map()

  constructor(models: ModelPricing[]) {
    for (const model of models) {
      this.pricing.set(model.model, model)
    }
  }

  record(event: Omit<CostEvent, 'cost' | 'timestamp'>): CostEvent {
    const pricing = this.pricing.get(event.model)

    let totalCost = 0
    if (pricing && !event.cached) {
      const result = calculateRequestCost(pricing, event.inputTokens, event.outputTokens)
      totalCost = result.totalCost
    }

    const fullEvent: CostEvent = {
      ...event,
      cost: totalCost,
      timestamp: new Date().toISOString(),
    }

    this.events.push(fullEvent)
    return fullEvent
  }

  getCostSummary(windowMs?: number): {
    totalCost: number
    totalRequests: number
    totalTokens: number
    cachedRequests: number
    cacheSavings: number
    byModel: Record<string, { cost: number; requests: number }>
    byFeature: Record<string, { cost: number; requests: number }>
    byUser: Record<string, { cost: number; requests: number }>
  } {
    const now = Date.now()
    const filtered = windowMs ? this.events.filter(e => now - new Date(e.timestamp).getTime() < windowMs) : this.events

    const byModel: Record<string, { cost: number; requests: number }> = {}
    const byFeature: Record<string, { cost: number; requests: number }> = {}
    const byUser: Record<string, { cost: number; requests: number }> = {}

    let totalCost = 0
    let totalTokens = 0
    let cachedRequests = 0
    let cacheSavings = 0

    for (const event of filtered) {
      totalCost += event.cost
      totalTokens += event.inputTokens + event.outputTokens

      if (event.cached) {
        cachedRequests++
        const p = this.pricing.get(event.model)
        if (p) {
          cacheSavings += calculateRequestCost(p, event.inputTokens, event.outputTokens).totalCost
        }
      }

      // Aggregate by model
      if (!byModel[event.model]) {
        byModel[event.model] = { cost: 0, requests: 0 }
      }
      byModel[event.model].cost += event.cost
      byModel[event.model].requests++

      // Aggregate by feature
      const feature = event.feature ?? 'unknown'
      if (!byFeature[feature]) {
        byFeature[feature] = { cost: 0, requests: 0 }
      }
      byFeature[feature].cost += event.cost
      byFeature[feature].requests++

      // Aggregate by user
      const user = event.userId ?? 'anonymous'
      if (!byUser[user]) {
        byUser[user] = { cost: 0, requests: 0 }
      }
      byUser[user].cost += event.cost
      byUser[user].requests++
    }

    return {
      totalCost,
      totalRequests: filtered.length,
      totalTokens,
      cachedRequests,
      cacheSavings,
      byModel,
      byFeature,
      byUser,
    }
  }
}

const costTracker = new CostTracker(modelPricing)
```

> **Advanced Note:** In production, persist cost events to a database and build dashboards. Cost spikes are often the first indicator of abuse, bugs, or runaway loops. Real-time cost monitoring is as important as performance monitoring.

---

## Section 2: Semantic Caching

### Why Semantic Caching?

Traditional caching requires exact key matches. LLM queries are rarely identical -- "What is the weather?" and "What's the weather like?" should return the same cached response. Semantic caching uses embeddings to find similar queries.

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

interface CacheEntry {
  query: string
  queryEmbedding: number[]
  response: string
  model: string
  timestamp: number
  hitCount: number
  inputTokens: number
  outputTokens: number
}

class SemanticCache {
  private entries: CacheEntry[] = []
  private maxEntries: number
  private similarityThreshold: number
  private ttlMs: number

  constructor(
    options: {
      maxEntries?: number
      similarityThreshold?: number
      ttlMs?: number
    } = {}
  ) {
    this.maxEntries = options.maxEntries ?? 1000
    this.similarityThreshold = options.similarityThreshold ?? 0.92
    this.ttlMs = options.ttlMs ?? 3600_000 // 1 hour default
  }

  async get(query: string): Promise<{ hit: boolean; response?: string; similarity?: number }> {
    // Generate embedding for the query
    const queryEmbedding = await this.getEmbedding(query)

    // Clean expired entries
    this.cleanExpired()

    // Find the most similar cached query
    let bestMatch: CacheEntry | null = null
    let bestSimilarity = 0

    for (const entry of this.entries) {
      const similarity = this.cosineSimilarity(queryEmbedding, entry.queryEmbedding)

      if (similarity > bestSimilarity && similarity >= this.similarityThreshold) {
        bestSimilarity = similarity
        bestMatch = entry
      }
    }

    if (bestMatch) {
      bestMatch.hitCount++
      return {
        hit: true,
        response: bestMatch.response,
        similarity: bestSimilarity,
      }
    }

    return { hit: false }
  }

  async set(query: string, response: string, model: string, inputTokens: number, outputTokens: number): Promise<void> {
    const queryEmbedding = await this.getEmbedding(query)

    // Evict if at capacity
    if (this.entries.length >= this.maxEntries) {
      this.evict()
    }

    this.entries.push({
      query,
      queryEmbedding,
      response,
      model,
      timestamp: Date.now(),
      hitCount: 0,
      inputTokens,
      outputTokens,
    })
  }

  getStats(): {
    entries: number
    totalHits: number
    estimatedSavings: number
  } {
    const totalHits = this.entries.reduce((sum, e) => sum + e.hitCount, 0)

    // Estimate savings based on cache hits and average request cost
    const avgCost =
      this.entries.length > 0
        ? this.entries.reduce((sum, e) => {
            const pricing = modelPricing.find(m => m.model === e.model)
            if (!pricing) return sum
            return sum + calculateRequestCost(pricing, e.inputTokens, e.outputTokens).totalCost
          }, 0) / this.entries.length
        : 0

    return {
      entries: this.entries.length,
      totalHits,
      estimatedSavings: totalHits * avgCost,
    }
  }

  private async getEmbedding(text: string): Promise<number[]> {
    const result = await embed({
      model: openai.embedding('text-embedding-3-small'),
      value: text,
    })
    return result.embedding
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0
    let normA = 0
    let normB = 0

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i]
      normA += a[i] * a[i]
      normB += b[i] * b[i]
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB)
    return denominator === 0 ? 0 : dotProduct / denominator
  }

  private cleanExpired(): void {
    const now = Date.now()
    this.entries = this.entries.filter(e => now - e.timestamp < this.ttlMs)
  }

  private evict(): void {
    // Evict least recently used entry with lowest hit count
    if (this.entries.length === 0) return

    let evictIndex = 0
    let lowestScore = Infinity

    for (let i = 0; i < this.entries.length; i++) {
      // Score = hitCount / age_in_hours (higher is more valuable)
      const ageHours = (Date.now() - this.entries[i].timestamp) / 3600_000
      const score = this.entries[i].hitCount / Math.max(ageHours, 0.1)

      if (score < lowestScore) {
        lowestScore = score
        evictIndex = i
      }
    }

    this.entries.splice(evictIndex, 1)
  }
}
```

### Using the Semantic Cache

Wrap your LLM calls with cache checks.

```typescript
const cache = new SemanticCache({
  maxEntries: 500,
  similarityThreshold: 0.93,
  ttlMs: 3600_000, // 1 hour
})

async function cachedGenerate(
  systemPrompt: string,
  userQuery: string,
  model: string = 'mistral-small-latest'
): Promise<{
  text: string
  cached: boolean
  similarity?: number
  cost: number
}> {
  // Check cache first
  const cacheKey = `${systemPrompt.substring(0, 50)}:${userQuery}`
  const cacheResult = await cache.get(cacheKey)

  if (cacheResult.hit && cacheResult.response) {
    // Record the cache hit for cost tracking
    costTracker.record({
      model,
      inputTokens: 0,
      outputTokens: 0,
      cached: true,
      feature: 'cached_response',
    })

    return {
      text: cacheResult.response,
      cached: true,
      similarity: cacheResult.similarity,
      cost: 0,
    }
  }

  // Cache miss -- make the actual LLM call
  const result = await generateText({
    model: mistral(model as any),
    system: systemPrompt,
    prompt: userQuery,
  })

  const inputTokens = result.usage?.inputTokens ?? 0
  const outputTokens = result.usage?.outputTokens ?? 0

  // Store in cache
  await cache.set(cacheKey, result.text, model, inputTokens, outputTokens)

  // Record cost
  const pricing = modelPricing.find(m => m.model === model)
  const cost = pricing ? calculateRequestCost(pricing, inputTokens, outputTokens).totalCost : 0

  costTracker.record({
    model,
    inputTokens,
    outputTokens,
    cached: false,
    feature: 'llm_call',
  })

  return {
    text: result.text,
    cached: false,
    cost,
  }
}

// Example usage showing cache in action
const query1 = await cachedGenerate('Answer concisely.', 'What is the capital of France?')
console.log(query1.cached) // false (first call)

const query2 = await cachedGenerate('Answer concisely.', "What's the capital city of France?")
console.log(query2.cached) // true (semantically similar)
console.log(query2.similarity) // ~0.95
```

> **Beginner Note:** Semantic caching trades embedding API cost for LLM API cost. An embedding call costs roughly 0.01% of an LLM call. If your cache hit rate is above 5%, you are saving money. Most applications with returning users see cache hit rates of 20-60%.

---

## Section 3: Model Routing

### Complexity-Based Routing

Not every query needs the most powerful (and expensive) model. Route simple queries to cheap models and complex queries to premium models.

```typescript
interface RoutingDecision {
  selectedModel: string
  tier: 'economy' | 'standard' | 'premium'
  reasoning: string
  estimatedCost: number
}

interface RoutingRule {
  name: string
  condition: (query: string, context?: Record<string, unknown>) => boolean
  targetTier: 'economy' | 'standard' | 'premium'
  priority: number
}

class ModelRouter {
  private rules: RoutingRule[] = []
  private models: Map<string, ModelPricing> = new Map()
  private defaultTier: 'economy' | 'standard' | 'premium' = 'standard'

  constructor(models: ModelPricing[]) {
    for (const model of models) {
      this.models.set(model.model, model)
    }
  }

  addRule(rule: RoutingRule): this {
    this.rules.push(rule)
    // Sort by priority (higher first)
    this.rules.sort((a, b) => b.priority - a.priority)
    return this
  }

  route(query: string, context?: Record<string, unknown>): RoutingDecision {
    // Check rules in priority order
    for (const rule of this.rules) {
      if (rule.condition(query, context)) {
        const model = this.getModelForTier(rule.targetTier)

        return {
          selectedModel: model.model,
          tier: rule.targetTier,
          reasoning: rule.name,
          estimatedCost: calculateRequestCost(model, estimateTokens(query), 200).totalCost,
        }
      }
    }

    // Default routing
    const defaultModel = this.getModelForTier(this.defaultTier)
    return {
      selectedModel: defaultModel.model,
      tier: this.defaultTier,
      reasoning: 'default',
      estimatedCost: calculateRequestCost(defaultModel, estimateTokens(query), 200).totalCost,
    }
  }

  private getModelForTier(tier: 'economy' | 'standard' | 'premium'): ModelPricing {
    const candidates = Array.from(this.models.values()).filter(m => m.tier === tier)

    if (candidates.length === 0) {
      // Fallback to the closest tier
      const allModels = Array.from(this.models.values())
      return allModels[0]
    }

    // Return the cheapest model in the tier
    return candidates.sort((a, b) => a.inputPricePerMillion - b.inputPricePerMillion)[0]
  }
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

// Configure routing rules
const router = new ModelRouter(modelPricing)

router
  .addRule({
    name: 'simple_greeting',
    condition: query => {
      const greetings = [/^(hi|hello|hey|good morning|good afternoon)/i, /^(thanks|thank you|bye|goodbye)/i]
      return greetings.some(p => p.test(query.trim()))
    },
    targetTier: 'economy',
    priority: 100,
  })
  .addRule({
    name: 'simple_factual',
    condition: query => {
      // Short factual questions
      return query.length < 100 && /^(what|who|when|where) (is|are|was|were) /i.test(query)
    },
    targetTier: 'economy',
    priority: 90,
  })
  .addRule({
    name: 'complex_analysis',
    condition: query => {
      // Long queries requesting analysis or comparison
      const complexIndicators = [
        /analyze/i,
        /compare and contrast/i,
        /pros and cons/i,
        /in-depth/i,
        /comprehensive/i,
        /evaluate/i,
      ]
      return query.length > 200 && complexIndicators.some(p => p.test(query))
    },
    targetTier: 'premium',
    priority: 80,
  })
  .addRule({
    name: 'code_generation',
    condition: query => {
      const codeIndicators = [
        /write (a |the |some )?code/i,
        /implement/i,
        /create (a |the )?(function|class|component)/i,
        /debug/i,
        /refactor/i,
      ]
      return codeIndicators.some(p => p.test(query))
    },
    targetTier: 'standard',
    priority: 70,
  })
  .addRule({
    name: 'high_stakes',
    condition: (_, context) => {
      // Use premium for high-value customers or critical features
      return (context?.customerTier as string) === 'enterprise' || (context?.feature as string) === 'legal_review'
    },
    targetTier: 'premium',
    priority: 95,
  })

// Use the router
const decision = router.route('What is the capital of France?', { customerTier: 'free' })
console.log(`Model: ${decision.selectedModel} (${decision.tier})`)
console.log(`Reasoning: ${decision.reasoning}`)
console.log(`Estimated cost: $${decision.estimatedCost.toFixed(6)}`)
```

### LLM-Based Routing

For more sophisticated routing, use a cheap LLM to classify query complexity.

```typescript
async function llmBasedRoute(query: string): Promise<RoutingDecision> {
  // Use the cheapest model to classify complexity
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        complexity: z.enum(['simple', 'moderate', 'complex']),
        reasoning: z.string(),
        requiresReasoning: z.boolean(),
        requiresCreativity: z.boolean(),
        domainExpertise: z.boolean(),
      }),
    }),
    system: `Classify the complexity of this query.
- simple: Factual recall, greetings, simple Q&A, short translations
- moderate: Summarization, standard writing, basic analysis, explanations
- complex: Multi-step reasoning, code generation, creative writing, in-depth analysis, domain expertise needed

Be concise in your reasoning.`,
    prompt: query,
  })

  const tierMap: Record<string, 'economy' | 'standard' | 'premium'> = {
    simple: 'economy',
    moderate: 'standard',
    complex: 'premium',
  }

  const tier = tierMap[output!.complexity]
  const model = modelPricing.find(m => m.tier === tier)

  if (!model) {
    const fallback = modelPricing.find(m => m.tier === 'standard')!
    return {
      selectedModel: fallback.model,
      tier: 'standard',
      reasoning: `LLM classified as ${output!.complexity}: ${output!.reasoning}`,
      estimatedCost: calculateRequestCost(fallback, estimateTokens(query), 200).totalCost,
    }
  }

  return {
    selectedModel: model.model,
    tier,
    reasoning: `LLM classified as ${output!.complexity}: ${output!.reasoning}`,
    estimatedCost: calculateRequestCost(model, estimateTokens(query), 200).totalCost,
  }
}
```

> **Advanced Note:** LLM-based routing adds latency and cost (the classification call itself). It pays off when the classification saves enough money by routing queries to cheaper models. Calculate your break-even point: if the classifier costs $0.0001 per call and routes 40% of queries from a $0.01 model to a $0.001 model, you save roughly $0.0035 per routed call. At 10,000 calls/day, that is $35/day saved for $1/day in classification cost.

---

## Section 4: Token Budgets

### Per-Request Token Control

Limit the tokens consumed by individual requests to prevent cost blowouts.

```typescript
interface TokenBudget {
  maxInputTokens: number
  maxOutputTokens: number
  warningThreshold: number // Fraction of max (e.g., 0.8)
}

interface TokenBudgetResult {
  withinBudget: boolean
  inputTokens: number
  maxInputTokens: number
  estimatedOutputTokens: number
  maxOutputTokens: number
  warnings: string[]
}

function checkTokenBudget(input: string, systemPrompt: string, budget: TokenBudget): TokenBudgetResult {
  const inputTokens = estimateTokens(input) + estimateTokens(systemPrompt)
  const warnings: string[] = []
  let withinBudget = true

  if (inputTokens > budget.maxInputTokens) {
    withinBudget = false
    warnings.push(`Input tokens (${inputTokens}) exceed budget (${budget.maxInputTokens})`)
  } else if (inputTokens > budget.maxInputTokens * budget.warningThreshold) {
    warnings.push(`Input tokens (${inputTokens}) approaching budget limit (${budget.maxInputTokens})`)
  }

  return {
    withinBudget,
    inputTokens,
    maxInputTokens: budget.maxInputTokens,
    estimatedOutputTokens: budget.maxOutputTokens,
    maxOutputTokens: budget.maxOutputTokens,
    warnings,
  }
}

// Apply token budget to an LLM call
async function budgetedGenerate(
  systemPrompt: string,
  userInput: string,
  budget: TokenBudget,
  model: string = 'mistral-small-latest'
): Promise<{
  text: string
  tokenUsage: { input: number; output: number }
  withinBudget: boolean
}> {
  const budgetCheck = checkTokenBudget(userInput, systemPrompt, budget)

  if (!budgetCheck.withinBudget) {
    throw new Error(`Token budget exceeded: ${budgetCheck.warnings.join('; ')}`)
  }

  const result = await generateText({
    model: mistral(model as any),
    system: systemPrompt,
    prompt: userInput,
    maxOutputTokens: budget.maxOutputTokens, // Enforce output limit
  })

  return {
    text: result.text,
    tokenUsage: {
      input: result.usage?.inputTokens ?? 0,
      output: result.usage?.outputTokens ?? 0,
    },
    withinBudget: true,
  }
}

// Usage with budget constraints
const response = await budgetedGenerate('Answer concisely.', 'Explain quantum computing.', {
  maxInputTokens: 2000,
  maxOutputTokens: 500,
  warningThreshold: 0.8,
})
```

### Per-User Daily Budgets

Track and enforce spending limits per user.

```typescript
interface UserBudgetConfig {
  dailyBudgetDollars: number
  maxRequestsPerDay: number
  maxTokensPerRequest: number
}

interface UserBudgetTier {
  tier: string
  config: UserBudgetConfig
}

const budgetTiers: UserBudgetTier[] = [
  {
    tier: 'free',
    config: {
      dailyBudgetDollars: 0.5,
      maxRequestsPerDay: 50,
      maxTokensPerRequest: 1000,
    },
  },
  {
    tier: 'pro',
    config: {
      dailyBudgetDollars: 5.0,
      maxRequestsPerDay: 500,
      maxTokensPerRequest: 4000,
    },
  },
  {
    tier: 'enterprise',
    config: {
      dailyBudgetDollars: 50.0,
      maxRequestsPerDay: 5000,
      maxTokensPerRequest: 8000,
    },
  },
]

class UserBudgetManager {
  private usage: Map<
    string,
    {
      date: string
      totalCost: number
      requestCount: number
    }
  > = new Map()

  private getUserKey(userId: string): string {
    const today = new Date().toISOString().split('T')[0]
    return `${userId}:${today}`
  }

  checkBudget(
    userId: string,
    tier: string,
    estimatedCost: number
  ): {
    allowed: boolean
    reason?: string
    remainingBudget: number
    remainingRequests: number
  } {
    const tierConfig = budgetTiers.find(t => t.tier === tier)
    if (!tierConfig) {
      return {
        allowed: false,
        reason: `Unknown tier: ${tier}`,
        remainingBudget: 0,
        remainingRequests: 0,
      }
    }

    const key = this.getUserKey(userId)
    const currentUsage = this.usage.get(key) ?? {
      date: new Date().toISOString().split('T')[0],
      totalCost: 0,
      requestCount: 0,
    }

    const remainingBudget = tierConfig.config.dailyBudgetDollars - currentUsage.totalCost
    const remainingRequests = tierConfig.config.maxRequestsPerDay - currentUsage.requestCount

    if (remainingRequests <= 0) {
      return {
        allowed: false,
        reason: 'Daily request limit reached',
        remainingBudget,
        remainingRequests: 0,
      }
    }

    if (estimatedCost > remainingBudget) {
      return {
        allowed: false,
        reason: `Estimated cost ($${estimatedCost.toFixed(4)}) exceeds remaining daily budget ($${remainingBudget.toFixed(4)})`,
        remainingBudget,
        remainingRequests,
      }
    }

    return {
      allowed: true,
      remainingBudget: remainingBudget - estimatedCost,
      remainingRequests: remainingRequests - 1,
    }
  }

  recordUsage(userId: string, cost: number): void {
    const key = this.getUserKey(userId)
    const current = this.usage.get(key) ?? {
      date: new Date().toISOString().split('T')[0],
      totalCost: 0,
      requestCount: 0,
    }

    current.totalCost += cost
    current.requestCount++
    this.usage.set(key, current)
  }
}
```

> **Beginner Note:** Token budgets serve two purposes: preventing accidental cost blowouts (a bug that sends massive prompts) and protecting against abuse (users trying to exhaust your API budget). Setting `maxOutputTokens` on the API call is your most direct tool -- it hard-limits the output regardless of what the model wants to generate.

---

## Section 5: Batching Requests

### Combining Related Queries

When you have multiple independent queries to process, batching them reduces overhead.

```typescript
interface BatchItem {
  id: string
  query: string
  systemPrompt?: string
  priority: 'high' | 'normal' | 'low'
}

interface BatchResult {
  id: string
  response: string
  tokens: { input: number; output: number }
  latencyMs: number
}

class RequestBatcher {
  private queue: BatchItem[] = []
  private batchSize: number
  private flushIntervalMs: number
  private timer: ReturnType<typeof setTimeout> | null = null
  private pendingResults: Map<
    string,
    {
      resolve: (result: BatchResult) => void
      reject: (error: Error) => void
    }
  > = new Map()

  constructor(batchSize: number = 5, flushIntervalMs: number = 1000) {
    this.batchSize = batchSize
    this.flushIntervalMs = flushIntervalMs
  }

  async add(item: BatchItem): Promise<BatchResult> {
    return new Promise((resolve, reject) => {
      this.pendingResults.set(item.id, { resolve, reject })
      this.queue.push(item)

      if (this.queue.length >= this.batchSize) {
        this.flush()
      } else if (!this.timer) {
        this.timer = setTimeout(() => this.flush(), this.flushIntervalMs)
      }
    })
  }

  private async flush(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer)
      this.timer = null
    }

    if (this.queue.length === 0) return

    const batch = this.queue.splice(0, this.batchSize)

    // Process batch concurrently
    const results = await Promise.allSettled(
      batch.map(async (item): Promise<BatchResult> => {
        const start = Date.now()

        const result = await generateText({
          model: mistral('mistral-small-latest'),
          system: item.systemPrompt ?? 'Answer concisely.',
          prompt: item.query,
        })

        return {
          id: item.id,
          response: result.text,
          tokens: {
            input: result.usage?.inputTokens ?? 0,
            output: result.usage?.outputTokens ?? 0,
          },
          latencyMs: Date.now() - start,
        }
      })
    )

    // Resolve promises
    for (let i = 0; i < batch.length; i++) {
      const pending = this.pendingResults.get(batch[i].id)
      if (!pending) continue

      const result = results[i]
      if (result.status === 'fulfilled') {
        pending.resolve(result.value)
      } else {
        pending.reject(new Error(result.reason instanceof Error ? result.reason.message : String(result.reason)))
      }

      this.pendingResults.delete(batch[i].id)
    }
  }
}

// Multi-query consolidation: combine related queries into one prompt
async function consolidateQueries(queries: string[], systemPrompt: string): Promise<Map<string, string>> {
  const numberedQueries = queries.map((q, i) => `[Q${i + 1}] ${q}`).join('\n')

  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    system: `${systemPrompt}

You will receive multiple numbered questions. Answer each one separately.
Format your response as:
[A1] Answer to question 1
[A2] Answer to question 2
...and so on.

Keep each answer concise.`,
    prompt: numberedQueries,
  })

  // Parse responses
  const results = new Map<string, string>()
  const answerPattern = /\[A(\d+)\]\s*([\s\S]*?)(?=\[A\d+\]|$)/g
  let match: RegExpExecArray | null

  while ((match = answerPattern.exec(text)) !== null) {
    const index = parseInt(match[1]) - 1
    if (index < queries.length) {
      results.set(queries[index], match[2].trim())
    }
  }

  return results
}

// Example: consolidating 5 questions into 1 API call
const questions = [
  'What is TypeScript?',
  'What is the Vercel AI SDK?',
  'What is RAG?',
  'What is prompt engineering?',
  'What is an LLM agent?',
]

const answers = await consolidateQueries(questions, 'You are a helpful programming teacher.')

// This used 1 API call instead of 5
for (const [q, a] of answers) {
  console.log(`Q: ${q}`)
  console.log(`A: ${a}\n`)
}
```

> **Advanced Note:** Query consolidation works well for independent factual questions but poorly for complex or context-dependent queries. The combined prompt can also exceed token limits. Use consolidation when you have many simple, independent queries -- not for complex, multi-turn, or context-heavy scenarios.

---

## Section 6: Fallback Chains

### Try Cheap, Escalate to Expensive

Build a chain of models that tries the cheapest option first and only escalates to more expensive models when the cheap one fails to produce adequate results.

```typescript
interface FallbackConfig {
  models: {
    id: string
    pricing: ModelPricing
    qualityThreshold: number // Minimum quality score to accept
  }[]
  qualityChecker: (query: string, response: string) => Promise<number> // 0-1 quality score
  maxAttempts: number
}

interface FallbackResult {
  response: string
  modelUsed: string
  attemptsMade: number
  totalCost: number
  qualityScore: number
  escalationPath: string[]
}

async function fallbackGenerate(config: FallbackConfig, systemPrompt: string, query: string): Promise<FallbackResult> {
  const escalationPath: string[] = []
  let totalCost = 0

  for (let i = 0; i < Math.min(config.models.length, config.maxAttempts); i++) {
    const modelConfig = config.models[i]
    escalationPath.push(modelConfig.id)

    try {
      const result = await generateText({
        model: mistral(modelConfig.id as any),
        system: systemPrompt,
        prompt: query,
      })

      const inputTokens = result.usage?.inputTokens ?? 0
      const outputTokens = result.usage?.outputTokens ?? 0
      const cost = calculateRequestCost(modelConfig.pricing, inputTokens, outputTokens).totalCost

      totalCost += cost

      // Check quality
      const qualityScore = await config.qualityChecker(query, result.text)

      if (qualityScore >= modelConfig.qualityThreshold) {
        return {
          response: result.text,
          modelUsed: modelConfig.id,
          attemptsMade: i + 1,
          totalCost,
          qualityScore,
          escalationPath,
        }
      }

      // Quality too low -- escalate to next model
      console.log(
        `Model ${modelConfig.id} scored ${qualityScore.toFixed(2)} ` +
          `(threshold: ${modelConfig.qualityThreshold}). Escalating...`
      )
    } catch (error) {
      console.error(`Model ${modelConfig.id} failed:`, error)
      // Continue to next model
    }
  }

  // All models exhausted -- use the last model without quality check
  const lastModel = config.models[config.models.length - 1]

  const result = await generateText({
    model: mistral(lastModel.id as any),
    system: systemPrompt,
    prompt: query,
  })

  return {
    response: result.text,
    modelUsed: lastModel.id,
    attemptsMade: config.models.length,
    totalCost,
    qualityScore: 0,
    escalationPath,
  }
}

// Simple quality checker using keyword heuristics
async function simpleQualityCheck(query: string, response: string): Promise<number> {
  let score = 0.5 // Base score

  // Check response length relative to query complexity
  const queryWords = query.split(/\s+/).length
  const responseWords = response.split(/\s+/).length

  if (responseWords > queryWords) score += 0.1
  if (responseWords > queryWords * 2) score += 0.1

  // Check for refusal patterns
  const refusalPatterns = [
    /I (cannot|can't|am unable to)/i,
    /I don't have (enough |the )?(information|context)/i,
    /I'm not sure/i,
  ]
  if (refusalPatterns.some(p => p.test(response))) {
    score -= 0.3
  }

  // Check for relevant content
  const queryKeywords = query
    .toLowerCase()
    .split(/\s+/)
    .filter(w => w.length > 3)
  const keywordOverlap = queryKeywords.filter(kw => response.toLowerCase().includes(kw)).length
  score += (keywordOverlap / Math.max(queryKeywords.length, 1)) * 0.2

  return Math.max(0, Math.min(1, score))
}

// Configure the fallback chain
const fallbackConfig: FallbackConfig = {
  models: [
    {
      id: 'mistral-small-latest',
      pricing: modelPricing.find(m => m.model === 'mistral-small-latest')!,
      qualityThreshold: 0.7,
    },
    {
      id: 'mistral-small-latest',
      pricing: modelPricing.find(m => m.model === 'mistral-small-latest')!,
      qualityThreshold: 0.6,
    },
    {
      id: 'claude-opus-4-20250514',
      pricing: modelPricing.find(m => m.model === 'claude-opus-4-20250514')!,
      qualityThreshold: 0.5,
    },
  ],
  qualityChecker: simpleQualityCheck,
  maxAttempts: 3,
}

// Use the fallback chain
const fallbackResult = await fallbackGenerate(
  fallbackConfig,
  'Answer the question helpfully and accurately.',
  'What are the key differences between TCP and UDP?'
)

console.log(`Model used: ${fallbackResult.modelUsed}`)
console.log(`Attempts: ${fallbackResult.attemptsMade}`)
console.log(`Total cost: $${fallbackResult.totalCost.toFixed(6)}`)
console.log(`Quality: ${fallbackResult.qualityScore.toFixed(2)}`)
console.log(`Path: ${fallbackResult.escalationPath.join(' -> ')}`)
```

> **Beginner Note:** Fallback chains save money by handling easy queries cheaply. If 70% of your queries can be answered adequately by a cheap model, you save roughly 70% of what you would spend using the premium model for everything. The quality check adds a small cost, but the savings from cheaper model usage far outweigh it.

---

## Section 7: Prompt Optimization

### Reducing Prompt Length

Every token in your system prompt is charged on every request. Shortening your prompt by 500 tokens saves 500 tokens multiplied by your request volume.

```typescript
interface PromptOptimizationResult {
  original: string
  optimized: string
  originalTokens: number
  optimizedTokens: number
  tokensSaved: number
  savingsPercent: number
  monthlySavings: number
}

async function optimizePrompt(
  prompt: string,
  requestsPerDay: number,
  model: ModelPricing
): Promise<PromptOptimizationResult> {
  const { text: optimized } = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a prompt optimization expert. Rewrite the given system prompt to be shorter while preserving ALL functionality.

Rules:
1. Remove redundant instructions
2. Combine overlapping rules
3. Use concise language (no filler words)
4. Preserve all behavioral requirements
5. Keep critical safety instructions intact
6. Remove examples if the instruction is clear without them
7. Use abbreviations only if meaning is unambiguous

Output ONLY the optimized prompt, nothing else.`,
    prompt: `Optimize this system prompt:\n\n${prompt}`,
  })

  const originalTokens = estimateTokens(prompt)
  const optimizedTokens = estimateTokens(optimized)
  const tokensSaved = originalTokens - optimizedTokens

  // Calculate monthly savings
  const costPerToken = model.inputPricePerMillion / 1_000_000
  const monthlySavings = tokensSaved * requestsPerDay * 30 * costPerToken

  return {
    original: prompt,
    optimized,
    originalTokens,
    optimizedTokens,
    tokensSaved,
    savingsPercent: originalTokens > 0 ? (tokensSaved / originalTokens) * 100 : 0,
    monthlySavings,
  }
}

// Example: optimizing a verbose prompt
const verbosePrompt = `You are a helpful and friendly customer support assistant for TechCorp, a technology company that sells laptops, phones, and accessories. Your job is to help customers with their questions and problems.

When a customer asks a question, you should:
1. First, carefully read and understand their question
2. Then, think about the best way to help them
3. Provide a clear, concise, and helpful answer
4. If you don't know the answer, say so honestly
5. If the customer seems frustrated, acknowledge their feelings
6. Always be polite and professional
7. Never make promises you can't keep
8. Never share internal company information
9. If the customer needs to talk to a human, provide the phone number: 1-800-TECH-HELP
10. Always end your response by asking if there's anything else you can help with

Remember:
- You represent TechCorp and should always maintain a professional image
- You should never discuss competitor products
- You should never share pricing that isn't publicly available
- You should never make up information about products
- Always recommend the customer visit our website for the latest information`

const optimizationResult = await optimizePrompt(
  verbosePrompt,
  10_000, // requests per day
  modelPricing.find(m => m.model === 'mistral-small-latest')!
)

console.log(`Original: ${optimizationResult.originalTokens} tokens`)
console.log(`Optimized: ${optimizationResult.optimizedTokens} tokens`)
console.log(`Saved: ${optimizationResult.tokensSaved} tokens (${optimizationResult.savingsPercent.toFixed(1)}%)`)
console.log(`Monthly savings: $${optimizationResult.monthlySavings.toFixed(2)}`)
```

### Context Window Management

When using RAG, carefully manage how much context you include to balance quality and cost.

```typescript
interface ContextBudget {
  maxContextTokens: number
  maxChunksToInclude: number
  minRelevanceScore: number
}

interface RankedChunk {
  content: string
  tokens: number
  relevanceScore: number
  source: string
}

function selectContextChunks(
  chunks: RankedChunk[],
  budget: ContextBudget
): {
  selected: RankedChunk[]
  totalTokens: number
  droppedCount: number
} {
  // Sort by relevance (highest first)
  const sorted = [...chunks].sort((a, b) => b.relevanceScore - a.relevanceScore)

  const selected: RankedChunk[] = []
  let totalTokens = 0

  for (const chunk of sorted) {
    // Check minimum relevance threshold
    if (chunk.relevanceScore < budget.minRelevanceScore) break

    // Check chunk count limit
    if (selected.length >= budget.maxChunksToInclude) break

    // Check token budget
    if (totalTokens + chunk.tokens > budget.maxContextTokens) {
      // Try to fit a truncated version
      const remainingBudget = budget.maxContextTokens - totalTokens
      if (remainingBudget > 100) {
        // Truncate the chunk to fit
        const truncatedContent = chunk.content.substring(
          0,
          remainingBudget * 4 // rough char-to-token conversion
        )
        selected.push({
          ...chunk,
          content: truncatedContent + '...',
          tokens: remainingBudget,
        })
        totalTokens += remainingBudget
      }
      break
    }

    selected.push(chunk)
    totalTokens += chunk.tokens
  }

  return {
    selected,
    totalTokens,
    droppedCount: chunks.length - selected.length,
  }
}

// Example: select context within budget
const retrievedChunks: RankedChunk[] = [
  {
    content: 'TechCorp was founded in 2010...',
    tokens: 200,
    relevanceScore: 0.95,
    source: 'about.md',
  },
  {
    content: 'Our return policy allows...',
    tokens: 150,
    relevanceScore: 0.88,
    source: 'returns.md',
  },
  {
    content: 'Shipping takes 3-5 business days...',
    tokens: 100,
    relevanceScore: 0.72,
    source: 'shipping.md',
  },
  {
    content: 'Employee handbook section...',
    tokens: 500,
    relevanceScore: 0.3,
    source: 'internal.md',
  },
]

const contextResult = selectContextChunks(retrievedChunks, {
  maxContextTokens: 400,
  maxChunksToInclude: 3,
  minRelevanceScore: 0.5,
})

console.log(`Selected ${contextResult.selected.length} chunks (${contextResult.totalTokens} tokens)`)
console.log(`Dropped ${contextResult.droppedCount} chunks`)
```

> **Advanced Note:** A common mistake is including all retrieved chunks regardless of relevance. In our example, including the low-relevance employee handbook (500 tokens, score 0.3) would cost more and could actually degrade answer quality by adding irrelevant noise. Be selective with context -- less can be more.

---

## Section 8: Monitoring and Alerting

### Real-Time Cost Dashboard

Build a monitoring system that tracks costs and alerts on anomalies.

```typescript
interface CostAlert {
  id: string
  type: 'budget_threshold' | 'spike' | 'anomaly' | 'per_user_limit'
  severity: 'info' | 'warning' | 'critical'
  message: string
  currentValue: number
  threshold: number
  timestamp: string
}

interface MonitoringConfig {
  dailyBudget: number
  warningThreshold: number // Fraction (e.g., 0.7 = 70%)
  criticalThreshold: number // Fraction (e.g., 0.9 = 90%)
  spikeMultiplier: number // Alert if cost is N times the average
  perUserDailyLimit: number
  checkIntervalMs: number
}

class CostMonitor {
  private costTrackerRef: CostTracker
  private config: MonitoringConfig
  private alerts: CostAlert[] = []
  private alertCallbacks: ((alert: CostAlert) => void)[] = []
  private hourlyHistory: { hour: string; cost: number }[] = []

  constructor(tracker: CostTracker, config: MonitoringConfig) {
    this.costTrackerRef = tracker
    this.config = config
  }

  onAlert(callback: (alert: CostAlert) => void): void {
    this.alertCallbacks.push(callback)
  }

  check(): CostAlert[] {
    const newAlerts: CostAlert[] = []
    const now = new Date()

    // Check 1: Daily budget threshold
    const dailySummary = this.costTrackerRef.getCostSummary(86_400_000)

    if (dailySummary.totalCost >= this.config.dailyBudget * this.config.criticalThreshold) {
      newAlerts.push({
        id: `budget-critical-${now.toISOString()}`,
        type: 'budget_threshold',
        severity: 'critical',
        message: `Daily spend at ${((dailySummary.totalCost / this.config.dailyBudget) * 100).toFixed(1)}% of budget`,
        currentValue: dailySummary.totalCost,
        threshold: this.config.dailyBudget,
        timestamp: now.toISOString(),
      })
    } else if (dailySummary.totalCost >= this.config.dailyBudget * this.config.warningThreshold) {
      newAlerts.push({
        id: `budget-warning-${now.toISOString()}`,
        type: 'budget_threshold',
        severity: 'warning',
        message: `Daily spend at ${((dailySummary.totalCost / this.config.dailyBudget) * 100).toFixed(1)}% of budget`,
        currentValue: dailySummary.totalCost,
        threshold: this.config.dailyBudget,
        timestamp: now.toISOString(),
      })
    }

    // Check 2: Cost spike detection
    const currentHourCost = this.costTrackerRef.getCostSummary(3_600_000).totalCost
    const avgHourlyCost =
      this.hourlyHistory.length > 0
        ? this.hourlyHistory.reduce((sum, h) => sum + h.cost, 0) / this.hourlyHistory.length
        : currentHourCost

    if (avgHourlyCost > 0 && currentHourCost > avgHourlyCost * this.config.spikeMultiplier) {
      newAlerts.push({
        id: `spike-${now.toISOString()}`,
        type: 'spike',
        severity: 'warning',
        message: `Hourly cost spike: $${currentHourCost.toFixed(4)} vs average $${avgHourlyCost.toFixed(4)} (${(currentHourCost / avgHourlyCost).toFixed(1)}x)`,
        currentValue: currentHourCost,
        threshold: avgHourlyCost * this.config.spikeMultiplier,
        timestamp: now.toISOString(),
      })
    }

    // Track hourly costs
    const currentHour = now.toISOString().substring(0, 13)
    const existingHour = this.hourlyHistory.find(h => h.hour === currentHour)
    if (existingHour) {
      existingHour.cost = currentHourCost
    } else {
      this.hourlyHistory.push({ hour: currentHour, cost: currentHourCost })
      // Keep only last 168 hours (1 week)
      if (this.hourlyHistory.length > 168) {
        this.hourlyHistory.shift()
      }
    }

    // Check 3: Per-user anomalies
    const userSummary = dailySummary.byUser
    for (const [userId, usage] of Object.entries(userSummary)) {
      if (usage.cost > this.config.perUserDailyLimit) {
        newAlerts.push({
          id: `user-limit-${userId}-${now.toISOString()}`,
          type: 'per_user_limit',
          severity: 'warning',
          message: `User ${userId} exceeded daily limit: $${usage.cost.toFixed(4)} (limit: $${this.config.perUserDailyLimit})`,
          currentValue: usage.cost,
          threshold: this.config.perUserDailyLimit,
          timestamp: now.toISOString(),
        })
      }
    }

    // Store and notify
    for (const alert of newAlerts) {
      this.alerts.push(alert)
      for (const callback of this.alertCallbacks) {
        callback(alert)
      }
    }

    return newAlerts
  }

  getAlertHistory(limit: number = 50): CostAlert[] {
    return this.alerts.slice(-limit)
  }

  getDashboardData(): {
    currentDailyCost: number
    dailyBudget: number
    budgetUsedPercent: number
    currentHourlyCost: number
    avgHourlyCost: number
    topModels: { model: string; cost: number }[]
    topFeatures: { feature: string; cost: number }[]
    cacheSavings: number
    activeAlerts: CostAlert[]
  } {
    const daily = this.costTrackerRef.getCostSummary(86_400_000)
    const hourly = this.costTrackerRef.getCostSummary(3_600_000)

    const avgHourlyCost =
      this.hourlyHistory.length > 0
        ? this.hourlyHistory.reduce((sum, h) => sum + h.cost, 0) / this.hourlyHistory.length
        : 0

    const topModels = Object.entries(daily.byModel)
      .map(([model, data]) => ({ model, cost: data.cost }))
      .sort((a, b) => b.cost - a.cost)

    const topFeatures = Object.entries(daily.byFeature)
      .map(([feature, data]) => ({ feature, cost: data.cost }))
      .sort((a, b) => b.cost - a.cost)

    // Active alerts from the last hour
    const oneHourAgo = new Date(Date.now() - 3_600_000).toISOString()
    const activeAlerts = this.alerts.filter(a => a.timestamp > oneHourAgo)

    return {
      currentDailyCost: daily.totalCost,
      dailyBudget: this.config.dailyBudget,
      budgetUsedPercent: (daily.totalCost / this.config.dailyBudget) * 100,
      currentHourlyCost: hourly.totalCost,
      avgHourlyCost,
      topModels,
      topFeatures,
      cacheSavings: daily.cacheSavings,
      activeAlerts,
    }
  }
}

// Set up monitoring
const costMonitor = new CostMonitor(costTracker, {
  dailyBudget: 100, // $100/day
  warningThreshold: 0.7, // Alert at 70%
  criticalThreshold: 0.9, // Critical at 90%
  spikeMultiplier: 3, // Alert if 3x average
  perUserDailyLimit: 5, // $5/user/day
  checkIntervalMs: 60_000, // Check every minute
})

costMonitor.onAlert(alert => {
  console.log(`[${alert.severity.toUpperCase()}] ${alert.type}: ${alert.message}`)

  // In production, send to Slack, PagerDuty, etc.
})
```

### Putting It All Together

A complete cost-optimized request handler that combines all techniques.

```typescript
async function optimizedRequest(
  userId: string,
  userTier: string,
  query: string,
  systemPrompt: string,
  feature: string
): Promise<{
  response: string
  cost: number
  optimizations: string[]
}> {
  const optimizations: string[] = []

  // Step 1: Check user budget
  const budgetManager = new UserBudgetManager()
  const estimatedCost = 0.001 // rough estimate
  const budgetCheck = budgetManager.checkBudget(userId, userTier, estimatedCost)

  if (!budgetCheck.allowed) {
    return {
      response: `Daily usage limit reached. ${budgetCheck.reason}`,
      cost: 0,
      optimizations: ['blocked_by_budget'],
    }
  }

  // Step 2: Check semantic cache
  const cacheResult = await cache.get(query)
  if (cacheResult.hit && cacheResult.response) {
    optimizations.push(`cache_hit (similarity: ${cacheResult.similarity?.toFixed(3)})`)

    costTracker.record({
      model: 'cached',
      inputTokens: 0,
      outputTokens: 0,
      cached: true,
      userId,
      feature,
    })

    return {
      response: cacheResult.response,
      cost: 0,
      optimizations,
    }
  }

  // Step 3: Route to appropriate model
  const routingDecision = router.route(query, {
    customerTier: userTier,
    feature,
  })
  optimizations.push(`routed_to: ${routingDecision.selectedModel} (${routingDecision.reasoning})`)

  // Step 4: Generate with token budget
  const budget: TokenBudget = {
    maxInputTokens: 4000,
    maxOutputTokens: 1000,
    warningThreshold: 0.8,
  }

  const result = await budgetedGenerate(systemPrompt, query, budget, routingDecision.selectedModel)

  // Step 5: Calculate actual cost
  const pricing = modelPricing.find(m => m.model === routingDecision.selectedModel)
  const actualCost = pricing
    ? calculateRequestCost(pricing, result.tokenUsage.input, result.tokenUsage.output).totalCost
    : 0

  // Step 6: Cache the response
  await cache.set(query, result.text, routingDecision.selectedModel, result.tokenUsage.input, result.tokenUsage.output)
  optimizations.push('cached_for_future')

  // Step 7: Track costs
  costTracker.record({
    model: routingDecision.selectedModel,
    inputTokens: result.tokenUsage.input,
    outputTokens: result.tokenUsage.output,
    cached: false,
    userId,
    feature,
  })

  budgetManager.recordUsage(userId, actualCost)

  // Step 8: Check for cost alerts
  costMonitor.check()

  return {
    response: result.text,
    cost: actualCost,
    optimizations,
  }
}

// Example usage
const result = await optimizedRequest(
  'user-789',
  'pro',
  'What are the benefits of TypeScript?',
  'Answer concisely and helpfully.',
  'qa'
)

console.log(`Response: ${result.response.substring(0, 100)}...`)
console.log(`Cost: $${result.cost.toFixed(6)}`)
console.log(`Optimizations: ${result.optimizations.join(', ')}`)
```

> **Beginner Note:** Each optimization technique compounds with the others. Caching alone might save 30%. Model routing might save another 40% on remaining requests. Prompt optimization saves 20% of token costs on every call. Combined, you can realistically achieve 60-80% cost reduction compared to naively using the most expensive model for everything.

> **Advanced Note:** Be careful not to over-optimize. Every optimization adds complexity and potential failure modes. Start with the highest-impact, lowest-complexity optimizations (prompt shortening, maxOutputTokens limits) before adding sophisticated systems like semantic caching and model routing. Measure the actual impact of each optimization before adding the next one.

> **Production Tip: Batch APIs** — For workloads that don't need real-time responses (eval suites, bulk classification, synthetic data generation), both Anthropic and OpenAI offer Batch APIs at 50% cost reduction. You submit a batch of requests and receive results within 24 hours. This is ideal for the eval pipelines from Module 19 and the fine-tuning data preparation from Module 20. Check each provider's documentation for current batch API endpoints and limits.

> **Local Alternative (Ollama):** Running models locally via Ollama eliminates per-token API costs entirely — your only cost is electricity and hardware. The optimization patterns here (semantic caching, model routing, prompt optimization) still apply: caching saves inference time, routing between model sizes saves GPU memory, and shorter prompts mean faster generation. Cost optimization with local models becomes performance optimization.

---

## Summary

In this module, you learned:

1. **LLM cost structure:** Input and output tokens are priced differently across models, and understanding per-model pricing is essential for cost-effective architecture decisions.
2. **Semantic caching:** Embedding queries and checking for similar previous requests avoids redundant LLM calls, dramatically reducing costs for repeated or near-duplicate queries.
3. **Model routing:** Matching query complexity to the appropriate model tier — using cheap models for simple tasks and expensive models only when needed — optimizes cost without sacrificing quality.
4. **Token budgets:** Enforcing per-request, per-user, and per-day token limits with `maxOutputTokens` and budget tracking to prevent cost overruns.
5. **Batching requests:** Combining related queries into single requests reduces per-call overhead and enables more efficient prompt construction.
6. **Fallback chains:** Trying cheaper models first and escalating to expensive models only when quality is insufficient provides automatic cost optimization with quality guarantees.
7. **Prompt optimization:** Shortening prompts, removing redundant instructions, and managing context windows to reduce token usage on every request.
8. **Monitoring and alerting:** Building real-time cost dashboards and budget alerts that track spending and notify you before costs spiral out of control.

In Module 23, you will learn observability techniques to monitor, trace, and debug LLM applications in production.

---

## Quiz

**Question 1:** Why are output tokens typically more expensive than input tokens?

A) Output tokens require more storage
B) Generating output tokens requires more computation (autoregressive decoding) than processing input tokens
C) Output tokens are longer
D) It is an arbitrary pricing decision

**Answer: B** -- Input tokens are processed in parallel during the forward pass, which is computationally efficient. Output tokens are generated one at a time (autoregressively), with each token requiring a full forward pass through the model. This makes output generation roughly 3-5x more computationally expensive per token, which is reflected in the pricing.

---

**Question 2:** What is the key advantage of semantic caching over exact-match caching for LLM applications?

A) Semantic caching is faster
B) Semantic caching uses less memory
C) Semantic caching can match queries with the same meaning but different wording
D) Semantic caching never returns stale results

**Answer: C** -- Users rarely ask the exact same question twice with identical wording. "What is the capital of France?" and "What's France's capital city?" have the same meaning but different text. Semantic caching uses embeddings to measure meaning similarity, allowing it to return cached responses for semantically equivalent queries. This dramatically increases cache hit rates compared to exact-match caching.

---

**Question 3:** When does model routing save the most money?

A) When all queries are equally complex
B) When query complexity varies widely, with many simple queries and some complex ones
C) When you only have one model available
D) When all queries require the most powerful model

**Answer: B** -- Model routing saves money when there is a distribution of query complexities. If most queries are simple (factual Q&A, greetings, basic requests) and only some require deep reasoning, you can route the majority to a cheap model (saving 80-90% per query) and only use the expensive model when needed. If all queries are equally complex, routing provides no benefit.

---

**Question 4:** What is the primary risk of aggressive prompt shortening?

A) It makes the code harder to read
B) The model may lose important behavioral instructions, degrading output quality
C) Shorter prompts are slower to process
D) It violates API terms of service

**Answer: B** -- Every word in a system prompt is there for a reason (or should be). Aggressive shortening can remove critical instructions that guide the model's behavior -- safety guardrails, formatting requirements, tone guidelines, or domain-specific rules. Always evaluate prompt changes with your eval framework (Module 19) to ensure quality is maintained.

---

**Question 5:** Why should cost monitoring include per-user tracking, not just overall spend?

A) To comply with data privacy regulations
B) To detect abuse, identify high-cost users, and enforce fair usage across tiers
C) To make billing calculations easier
D) Per-user tracking is cheaper than aggregate tracking

**Answer: B** -- Per-user cost tracking serves multiple purposes: detecting abuse (a single user consuming disproportionate resources), enforcing tier-based budgets (free users vs paid users), identifying power users who might benefit from optimization, and providing data for usage-based billing. Aggregate-only monitoring can miss situations where one user is responsible for a cost spike.

---

## Exercises

### Exercise 1: Build a Cost-Optimized Pipeline

Build a complete cost-optimized request handling pipeline that combines semantic caching, model routing, and token budgets.

**Specification:**

1. Implement a `SemanticCache` with:
   - Configurable similarity threshold (try values between 0.85 and 0.95)
   - TTL-based expiration
   - Usage statistics (hit rate, estimated savings)

2. Implement a `ModelRouter` with at least 5 routing rules:
   - Simple factual questions -> economy model
   - Code-related queries -> standard model
   - Complex analysis -> premium model
   - High-value customer context -> premium model
   - Default -> standard model

3. Combine them into an `optimizedRequest` function that:
   - Checks the cache first
   - Routes to the appropriate model on cache miss
   - Enforces token budgets
   - Tracks costs per request

4. Test with 20 diverse queries (mix of simple, moderate, and complex):
   - Include some semantically similar queries to test caching
   - Include queries at different complexity levels to test routing
   - Track the total cost with and without optimization

**Expected output:** A cost comparison report showing the savings from each optimization technique and the overall reduction.

### Exercise 2: Cost Monitoring and Alerting

Build a cost monitoring system that tracks spend and generates alerts.

**Specification:**

1. Implement a `CostTracker` that records every LLM call with:
   - Model used, input/output tokens, cost
   - User ID and feature name
   - Whether the response was cached

2. Implement a `CostMonitor` with alerts for:
   - Daily budget threshold (warning at 70%, critical at 90%)
   - Hourly cost spikes (greater than 3x the rolling average)
   - Per-user daily limits exceeded
   - Cache hit rate dropping below threshold

3. Build a dashboard data generator that produces:
   - Current daily spend vs budget
   - Cost breakdown by model, feature, and user
   - Cache savings estimation
   - Active alerts

4. Simulate a day of traffic:
   - Generate 100 requests with varying models and costs
   - Include a simulated abuse scenario (one user making many expensive requests)
   - Include a simulated cost spike scenario

**Expected output:** Dashboard data JSON showing spend breakdown, active alerts, and optimization recommendations.
