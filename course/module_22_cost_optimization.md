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
```

Your task is to build three things:

1. A `modelPricing` array of `ModelPricing` entries covering at least 4 models across different tiers. Include at least one economy model, one standard model, and one premium model. Use realistic pricing -- check your providers' current pricing pages if needed.

2. A `calculateRequestCost` function:

```typescript
function calculateRequestCost(
  model: ModelPricing,
  inputTokens: number,
  outputTokens: number
): { inputCost: number; outputCost: number; totalCost: number }
```

The function converts token counts to dollar costs using the model's per-million pricing. How do you convert "price per million tokens" to "price per token"?

3. A `projectMonthlyCost` function:

```typescript
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
}
```

This function should use `calculateRequestCost` internally and project costs forward. The `breakdown` should show input vs output cost contribution over a month.

Finally, write a `compareModelCosts` function that iterates over all models and logs per-request, monthly, and annual costs for a given scenario. Call it with a typical customer support scenario: 1,500 input tokens, 300 output tokens, 10,000 requests per day.

Think about: Which component (input or output) dominates the cost? What happens to annual cost when you switch from a premium to an economy model?

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
```

Build a `CostTracker` class that accepts a `ModelPricing[]` in its constructor and provides two methods:

1. `record(event)` -- Takes an event without `cost` or `timestamp` (use `Omit<CostEvent, 'cost' | 'timestamp'>`), calculates the cost from the pricing table (zero cost if `cached` is true), adds the timestamp, and stores the full event. Returns the completed `CostEvent`.

2. `getCostSummary(windowMs?)` -- Returns aggregate statistics over an optional time window. The return type should include: `totalCost`, `totalRequests`, `totalTokens`, `cachedRequests`, `cacheSavings`, and breakdowns `byModel`, `byFeature`, and `byUser` (each as `Record<string, { cost: number; requests: number }>`).

For the summary, filter events to the time window first, then iterate once to compute all aggregates. For `cacheSavings`, calculate what cached requests _would have_ cost if they had not been cached. How would you handle events where the `feature` or `userId` is undefined? What default key would you use?

> **Advanced Note:** In production, persist cost events to a database and build dashboards. Cost spikes are often the first indicator of abuse, bugs, or runaway loops. Real-time cost monitoring is as important as performance monitoring.

---

## Section 2: Semantic Caching

> **Note:** This section uses OpenAI embeddings for the semantic cache. Substitute your preferred embedding provider.

### Why Semantic Caching?

Traditional caching requires exact key matches. LLM queries are rarely identical -- "What is the weather?" and "What's the weather like?" should return the same cached response. Semantic caching uses embeddings to find similar queries.

```typescript
import { embed, cosineSimilarity } from 'ai'
import { openai } from '@ai-sdk/openai'

// AI SDK provides cosineSimilarity — see Module 8 for the math

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
```

Build a `SemanticCache` class with configurable `maxEntries`, `similarityThreshold` (default 0.92), and `ttlMs` (default 1 hour). The constructor accepts an options object with these three optional fields.

The class needs four public methods:

1. `async get(query)` -- Generates an embedding for the query (use `embed` from the AI SDK), cleans expired entries, then scans all entries to find the one with the highest cosine similarity above the threshold. If found, increment its `hitCount` and return `{ hit: true, response, similarity }`. Otherwise return `{ hit: false }`.

2. `async set(query, response, model, inputTokens, outputTokens)` -- Generates an embedding, evicts if at capacity, then stores the new entry with `timestamp: Date.now()` and `hitCount: 0`.

3. `getStats()` -- Returns `{ entries, totalHits, estimatedSavings }`. Estimated savings = total hits multiplied by the average cost of a cached entry (computed from the stored token counts and your pricing table).

And three private helpers:

- `getEmbedding(text)` -- Wraps `embed()` with your embedding model.
- `cleanExpired()` -- Filters out entries older than `ttlMs`.
- `evict()` -- Removes the least valuable entry. How would you score entries to decide which one to evict? Consider combining hit count and age -- a high-hit, recent entry is more valuable than a zero-hit, old entry.

### Using the Semantic Cache

Build a `cachedGenerate` function that wraps `generateText` with your cache:

```typescript
async function cachedGenerate(
  systemPrompt: string,
  userQuery: string,
  model?: string
): Promise<{ text: string; cached: boolean; similarity?: number; cost: number }>
```

The function should:

1. Build a cache key from the system prompt prefix and user query.
2. Check the cache. On a hit, record it with your `CostTracker` (zero tokens, `cached: true`) and return immediately.
3. On a miss, call `generateText`, store the result in the cache, record the cost, and return.

Test with two semantically similar queries like "What is the capital of France?" and "What's the capital city of France?" -- the second should be a cache hit.

Think about: What similarity threshold is too aggressive (too many false cache hits)? What threshold is too conservative (too few cache hits)? How would you tune this for your application?

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
```

Build a `ModelRouter` class that accepts `ModelPricing[]` in its constructor and stores them in a Map. It needs:

1. `addRule(rule)` -- Adds a routing rule and re-sorts the rules array by priority (highest first). Returns `this` for method chaining.

2. `route(query, context?)` -- Checks rules in priority order. The first rule whose `condition` returns `true` determines the tier. Look up the cheapest model in that tier and return a `RoutingDecision` with the estimated cost. If no rule matches, fall back to the default tier (`'standard'`).

3. A private `getModelForTier(tier)` helper that filters models by tier and returns the cheapest one (by input price). If no model exists for that tier, fall back to any available model.

You will also need a simple `estimateTokens` helper -- a rough approximation of `Math.ceil(text.length / 4)` works for planning purposes.

Configure the router with at least 5 rules:

- **Simple greetings** (priority 100) -- Regex for "hi," "hello," "thanks," etc. Route to economy.
- **Simple factual** (priority 90) -- Short queries starting with "what is," "who is," etc. Route to economy.
- **High stakes** (priority 95) -- Use context flags like `customerTier === 'enterprise'`. Route to premium.
- **Complex analysis** (priority 80) -- Long queries with keywords like "analyze," "compare and contrast." Route to premium.
- **Code generation** (priority 70) -- Keywords like "implement," "debug," "refactor." Route to standard.

Test the router with queries of varying complexity and verify each routes to the expected tier.

Think about: Why does priority ordering matter? What happens if the "simple greeting" rule has lower priority than "high stakes" -- could an enterprise user's "hi" be routed to premium?

### LLM-Based Routing

For more sophisticated routing, use a cheap LLM to classify query complexity. Build an `llmBasedRoute` function:

```typescript
async function llmBasedRoute(query: string): Promise<RoutingDecision>
```

Use `generateText` with `Output.object` and the cheapest available model. Define a Zod schema for the classification output with fields: `complexity` (enum: simple/moderate/complex), `reasoning` (string), `requiresReasoning`, `requiresCreativity`, `domainExpertise` (all booleans). Map complexity levels to tiers: simple -> economy, moderate -> standard, complex -> premium.

The system prompt should describe what each complexity level means. Look up the model for the chosen tier and compute the estimated cost.

When does LLM-based routing pay for itself versus simple regex rules? How would you calculate the break-even point?

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
```

Build two functions:

1. `checkTokenBudget(input, systemPrompt, budget)` -- Estimates the total input tokens (user input + system prompt) using your `estimateTokens` helper. If the total exceeds `maxInputTokens`, set `withinBudget` to `false` and push a warning. If the total exceeds the `warningThreshold` fraction but is still within budget, push a warning but keep `withinBudget` true. Return a `TokenBudgetResult`.

2. `budgetedGenerate(systemPrompt, userInput, budget, model?)` -- Checks the token budget first. If over budget, throw an error with the warning details. Otherwise, call `generateText` with `maxOutputTokens` set to `budget.maxOutputTokens` -- this is the key: the AI SDK's `maxOutputTokens` parameter hard-limits the model's output at the API level.

```typescript
async function budgetedGenerate(
  systemPrompt: string,
  userInput: string,
  budget: TokenBudget,
  model?: string
): Promise<{ text: string; tokenUsage: { input: number; output: number }; withinBudget: boolean }>
```

Test with a query and a budget of 2,000 max input tokens and 500 max output tokens. What happens when you send a system prompt that alone exceeds the input budget?

### Per-User Daily Budgets

Track and enforce spending limits per user. Define budget tiers:

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
```

Create a `budgetTiers` array with at least three tiers: free ($0.50/day, 50 requests), pro ($5/day, 500 requests), and enterprise ($50/day, 5,000 requests).

Build a `UserBudgetManager` class that tracks daily usage per user in a Map. The Map key should combine the user ID and today's date (e.g., `userId:2025-06-15`) so usage resets daily. The class needs:

1. `checkBudget(userId, tier, estimatedCost)` -- Looks up the tier config, gets the user's current daily usage, and checks whether the request is allowed. Deny if: daily request limit is reached, or the estimated cost exceeds the remaining budget. Return `{ allowed, reason?, remainingBudget, remainingRequests }`.

2. `recordUsage(userId, cost)` -- Increments the user's request count and adds the cost to their daily total.

Think about: How would you handle a user whose tier changes mid-day? Should you reset their usage or carry it forward?

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
```

Build a `RequestBatcher` class with configurable `batchSize` (default 5) and `flushIntervalMs` (default 1000). It needs:

- A queue of pending `BatchItem`s.
- A `pendingResults` Map that stores `{ resolve, reject }` promise callbacks keyed by item ID.
- An `add(item)` method that returns a `Promise<BatchResult>`. It stores the resolve/reject callbacks, pushes the item to the queue, and triggers a flush if the queue reaches `batchSize`. If the queue is not full, set a timer to flush after `flushIntervalMs`.
- A private `flush()` method that takes a batch from the queue, processes all items concurrently with `Promise.allSettled`, and resolves or rejects each pending promise based on the result.

The key pattern here is that `add` returns a Promise, but the actual LLM call happens later when the batch flushes. How does wrapping `generateText` in a `new Promise` and storing the resolver let you decouple the caller from the execution?

### Multi-Query Consolidation

Build a `consolidateQueries` function that combines multiple questions into a single LLM call:

```typescript
async function consolidateQueries(queries: string[], systemPrompt: string): Promise<Map<string, string>>
```

The function should:

1. Number the queries as `[Q1] ...`, `[Q2] ...`, etc. and join them into a single prompt.
2. Instruct the model to answer each separately, formatted as `[A1] ...`, `[A2] ...`.
3. Parse the response with a regex to extract each numbered answer. The pattern `\[A(\d+)\]\s*([\s\S]*?)(?=\[A\d+\]|$)` matches each answer block.
4. Return a Map from original query to its answer.

Test by consolidating 5 simple questions into 1 API call. This uses 1 call instead of 5.

Think about: What happens if the model skips an answer or merges two? How would you detect and handle incomplete responses? When does consolidation _not_ work well?

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
```

Build a `fallbackGenerate` function:

```typescript
async function fallbackGenerate(config: FallbackConfig, systemPrompt: string, query: string): Promise<FallbackResult>
```

The function should iterate through models in order (cheapest first). For each model:

1. Call `generateText` and track the cost (accumulate `totalCost` across attempts).
2. Run the `qualityChecker` on the response. If the score meets the model's `qualityThreshold`, return immediately with the successful result.
3. If the quality is too low, log the score and continue to the next model.
4. If the call throws an error, catch it and continue to the next model.
5. If all models are exhausted, make one final call with the last model and return it regardless of quality.

Track the `escalationPath` -- the list of model IDs attempted in order.

Then build a `simpleQualityCheck` function:

```typescript
async function simpleQualityCheck(query: string, response: string): Promise<number>
```

This is a heuristic quality scorer (0 to 1). Start at a base score of 0.5, then adjust based on:

- Response length relative to query length (longer responses to complex queries score higher).
- Refusal patterns (phrases like "I cannot" or "I'm not sure" should decrease the score).
- Keyword overlap between query and response (if the response contains query-relevant words, it is more likely on-topic).

Clamp the final score between 0 and 1. What other signals could indicate response quality without calling another LLM?

Configure a fallback chain with at least 3 models ordered from cheapest to most expensive, with decreasing quality thresholds (e.g., 0.7, 0.6, 0.5). Test with a knowledge question and observe which model handles it.

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
```

Build an `optimizePrompt` function:

```typescript
async function optimizePrompt(
  prompt: string,
  requestsPerDay: number,
  model: ModelPricing
): Promise<PromptOptimizationResult>
```

Use `generateText` with a system prompt that instructs the model to rewrite the given prompt to be shorter while preserving all functionality. Your optimization instructions should include rules like: remove redundant instructions, combine overlapping rules, use concise language, preserve behavioral requirements, keep safety instructions intact, remove examples if the instruction is clear without them. Tell the model to output only the optimized prompt.

After receiving the optimized version, calculate: `originalTokens`, `optimizedTokens`, `tokensSaved`, `savingsPercent`, and `monthlySavings` (tokens saved per request x requests per day x 30 days x cost per token).

Test with a verbose customer support prompt (15-20 lines of instructions). How much can the model compress it? What is the monthly dollar savings at 10,000 requests per day?

Think about: How do you verify the optimized prompt still produces the same behavior? What if the model strips a critical safety instruction during optimization?

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
```

Build a `selectContextChunks` function:

```typescript
function selectContextChunks(
  chunks: RankedChunk[],
  budget: ContextBudget
): { selected: RankedChunk[]; totalTokens: number; droppedCount: number }
```

The function should:

1. Sort chunks by relevance (highest first).
2. Iterate through sorted chunks, adding each if it passes three checks: relevance above `minRelevanceScore`, chunk count below `maxChunksToInclude`, and token total below `maxContextTokens`.
3. If a chunk would exceed the token budget but there are at least 100 tokens remaining, truncate the chunk to fit and append `'...'`.
4. Return the selected chunks, total tokens used, and the count of dropped chunks.

Test with 4 chunks of varying relevance and sizes against a budget of 400 tokens and 3 max chunks. Include one low-relevance chunk (score 0.3) with a large token count -- verify it gets dropped. Why would including irrelevant context actually _degrade_ answer quality, not just waste tokens?

> **Advanced Note:** A common mistake is including all retrieved chunks regardless of relevance. Including a low-relevance chunk (500 tokens, score 0.3) costs more and can actually degrade answer quality by adding irrelevant noise. Be selective with context -- less can be more.

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
```

Build a `CostMonitor` class that takes a `CostTracker` reference and a `MonitoringConfig` in its constructor. It needs:

1. `onAlert(callback)` -- Registers a callback function that is called whenever a new alert is generated.

2. `check()` -- The core monitoring method. It runs three checks and returns any new alerts:
   - **Daily budget threshold** -- Get the daily cost summary (last 86,400,000 ms). If total cost exceeds the `criticalThreshold` fraction of `dailyBudget`, emit a critical alert. If it exceeds `warningThreshold`, emit a warning.
   - **Cost spike detection** -- Compare the current hour's cost to the rolling average of hourly costs. If the current hour exceeds the average by more than `spikeMultiplier`, emit a spike alert. Maintain an hourly history array (keep the last 168 hours -- one week).
   - **Per-user anomalies** -- Iterate the daily summary's `byUser` breakdown. If any user exceeds `perUserDailyLimit`, emit a per-user alert.

   For each alert generated, call all registered callbacks.

3. `getAlertHistory(limit?)` -- Returns the most recent alerts (default 50).

4. `getDashboardData()` -- Returns a snapshot for display: current daily cost, budget used percentage, current and average hourly cost, top models and features by cost (sorted descending), cache savings, and active alerts from the last hour.

How would you compute the hourly history? Consider using a string key like `now.toISOString().substring(0, 13)` to bucket costs by hour.

### Putting It All Together

Build an `optimizedRequest` function that combines all the techniques from this module into a single request handler:

```typescript
async function optimizedRequest(
  userId: string,
  userTier: string,
  query: string,
  systemPrompt: string,
  feature: string
): Promise<{ response: string; cost: number; optimizations: string[] }>
```

The function should execute these steps in order, tracking which optimizations were applied:

1. **Check user budget** -- Use your `UserBudgetManager`. If denied, return early with the reason.
2. **Check semantic cache** -- On a hit, record it and return the cached response (cost = 0).
3. **Route to appropriate model** -- Use your `ModelRouter` with the user's tier as context.
4. **Generate with token budget** -- Use `budgetedGenerate` with the routed model.
5. **Calculate actual cost** -- Look up the model's pricing and compute the cost.
6. **Cache the response** -- Store it for future cache hits.
7. **Track costs** -- Record with your `CostTracker`.
8. **Check for alerts** -- Run `costMonitor.check()`.

Test with a sample query and log the response, cost, and the list of optimizations that were applied. What order of savings do you see?

> **Beginner Note:** Each optimization technique compounds with the others. Caching alone might save 30%. Model routing might save another 40% on remaining requests. Prompt optimization saves 20% of token costs on every call. Combined, you can realistically achieve 60-80% cost reduction compared to naively using the most expensive model for everything.

> **Advanced Note:** Be careful not to over-optimize. Every optimization adds complexity and potential failure modes. Start with the highest-impact, lowest-complexity optimizations (prompt shortening, maxOutputTokens limits) before adding sophisticated systems like semantic caching and model routing. Measure the actual impact of each optimization before adding the next one.

> **Production Tip: Batch APIs** — For workloads that don't need real-time responses (eval suites, bulk classification, synthetic data generation), both Anthropic and OpenAI offer Batch APIs at 50% cost reduction. You submit a batch of requests and receive results within 24 hours. This is ideal for the eval pipelines from Module 19 and the fine-tuning data preparation from Module 20. Check each provider's documentation for current batch API endpoints and limits.

> **Local Alternative (Ollama):** Running models locally via Ollama eliminates per-token API costs entirely — your only cost is electricity and hardware. The optimization patterns here (semantic caching, model routing, prompt optimization) still apply: caching saves inference time, routing between model sizes saves GPU memory, and shorter prompts mean faster generation. Cost optimization with local models becomes performance optimization.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Real-Time Cost Tracking

Production systems track cost per request in real-time, not as an afterthought. A cost tracking middleware wraps every `generateText` or `streamText` call and records:

- Input tokens consumed
- Output tokens generated
- Thinking/reasoning tokens (tracked separately — these can dominate cost on reasoning models)
- Cost calculated per model (each has different pricing)
- Running session total

The user sees what they are spending. This transparency changes behavior — developers write shorter prompts and use cheaper models when they see the cost of each request.

**Pattern:** Wrap your LLM calls in a cost tracker that reads token counts from the response's `usage` field and multiplies by the model's per-token price:

```ts
const cost = (usage.inputTokens * inputPrice + usage.outputTokens * outputPrice) / 1_000_000
```

Display cumulative cost at the end of each response so the user can make informed decisions about model selection and prompt length.

## Section 10: Token Budget Allocation

The context window is a finite budget that must be allocated across competing needs. Production systems treat it like a resource with priorities:

| Component            | Typical Allocation | Priority |
| -------------------- | ------------------ | -------- |
| System prompt        | ~20K tokens        | Fixed    |
| Tool definitions     | ~5K tokens         | Fixed    |
| Conversation history | Dynamic            | Medium   |
| Tool results         | Capped per result  | Low      |
| Output reserved      | ~4K tokens         | High     |

When the total exceeds the window, something must be cut. A budget allocator enforces these limits: tool results are truncated first, then conversation history is compacted, then older messages are dropped. The system prompt and output reservation are never reduced.

**Key insight:** Without explicit allocation, tool results can silently consume most of your context window. A single large file read (10K tokens) might crowd out conversation history that the model needs for coherent responses.

## Section 11: Compaction as Cost Optimization

Module 4 teaches compaction for context window management. Here, view it as a cost technique.

**The math:** A 100K-token conversation costs roughly $0.30 per API call at typical input pricing ($3/M tokens). After compacting to 20K tokens, each call costs $0.06. Over 50 subsequent calls in the same session, you save: (50 x $0.30) - (50 x $0.06) = **$12.00 per session**.

For an application handling 1,000 sessions per day, compaction saves $12,000/day in input token costs alone. The one-time cost of the compaction call (generating a summary) pays for itself within 2-3 subsequent messages.

**When to compact:** Trigger compaction when context usage exceeds a threshold (e.g., 60% of the window). Do not wait until the window is full — by then you have already paid the high-token cost for several calls.

## Section 12: Tool Result Budgeting

Tool results are one of the largest and least controlled sources of token usage. A file read might return 10K tokens. A search might return 50 results. Without budgets, tool results dominate the context window and inflate costs.

Production systems enforce per-result budgets:

- Cap each tool result at N tokens (e.g., 3,000)
- Truncate intelligently — preserve the beginning and end, remove the middle (the "head + tail" strategy)
- Track how many tokens were truncated and report savings

```ts
function truncateResult(text: string, maxTokens: number): string {
  const tokens = estimateTokens(text)
  if (tokens <= maxTokens) return text
  const headChars = Math.floor(maxTokens * 0.6) * 4 // Convert tokens to approx chars
  const tailChars = (maxTokens - Math.floor(maxTokens * 0.6)) * 4
  return text.slice(0, headChars) + '\n...[truncated]...\n' + text.slice(-tailChars)
}
```

## Section 13: Reasoning Effort and Thinking Budget Control

Not every prompt deserves the same reasoning budget. Production systems expose per-provider controls for model reasoning intensity:

- **Anthropic:** Thinking budgets (e.g., `budgetTokens: 5000`) that control how much internal reasoning the model performs.
- **OpenAI:** Reasoning effort levels (`low`, `medium`, `high`) that trade cost for quality.

Simple tasks (renaming a variable, fixing a typo) use minimal reasoning. Complex tasks (architectural refactoring, debugging a subtle race condition) get full reasoning. A cost-optimized system adapts effort to complexity, potentially saving 50-80% on simple tasks.

**Pattern:** Use a simple classifier — message length, presence of code blocks, keywords like "refactor" or "debug" — to route to different effort levels. Pass the effort level via `providerOptions`:

```ts
const result = await generateText({
  model,
  prompt,
  providerOptions: { anthropic: { thinking: { type: 'enabled', budgetTokens: effortBudget } } },
})
```

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
9. **Real-time cost tracking:** Wrapping every LLM call in a cost tracker that reads token counts from the response and displays cumulative cost per session for transparency.
10. **Token budget allocation:** Treating the context window as a finite resource with priorities — fixed allocations for system prompt and output, dynamic allocation for history and tool results.
11. **Compaction as cost optimization:** Summarizing long conversations to reduce per-call input costs, with savings that compound over subsequent messages in a session.
12. **Tool result budgeting:** Capping per-result token counts with head-plus-tail truncation to prevent tool results from dominating the context window and inflating costs.
13. **Reasoning effort control:** Adapting per-request reasoning budgets (Anthropic thinking tokens, OpenAI reasoning effort) to task complexity, saving 50-80% on simple tasks.

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

**Question 6 (Medium):** A 100K-token conversation costs $0.30 per LLM call. After compacting to 20K tokens, each subsequent call costs $0.06. When should compaction be triggered?

A) Only when the context window is completely full
B) When context usage exceeds a threshold (e.g., 60%) so you avoid paying the high-token cost for multiple calls before compaction
C) After every single message to keep costs minimal
D) Only at the start of each new session

**Answer: B** -- Triggering compaction at a usage threshold (like 60%) is optimal because it avoids paying the inflated per-call cost for several messages while the window fills up. Waiting until the window is full (A) means you have already overpaid for multiple calls. Compacting after every message (C) wastes the cost of the compaction call itself, which only pays for itself when amortized over several subsequent messages.

---

**Question 7 (Hard):** A system uses per-provider reasoning effort controls: Anthropic thinking budgets and OpenAI reasoning effort levels. A request to "fix this typo" is routed with minimal reasoning, while "refactor this module's architecture" gets full reasoning. What is the primary benefit of this approach?

A) It improves response quality for all tasks equally
B) It saves 50-80% on simple tasks by allocating reasoning budget proportional to task complexity, without degrading complex task quality
C) It reduces network latency by sending smaller requests
D) It prevents the model from overthinking and producing worse results

**Answer: B** -- Reasoning effort control adapts cost to complexity. Simple tasks (typo fixes, variable renames) do not need extensive internal reasoning — minimal effort produces equally good results at a fraction of the cost. Complex tasks (architecture refactoring, subtle bug diagnosis) benefit from full reasoning. By classifying tasks and routing to appropriate effort levels, the system saves significantly on the high volume of simple requests while preserving quality where it matters.

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

### Exercise 3: Real-Time Cost Tracker

Build a cost tracking middleware that wraps LLM calls and reports per-call and cumulative costs.

**Specification:**

1. Create a `CostTrackerMiddleware` that wraps `generateText` calls and records:
   - Model name
   - Input tokens, output tokens, and thinking tokens (if applicable)
   - Cost per call (calculated from a configurable pricing table)
   - Cumulative session cost
   - Timestamp

2. Implement a pricing table with at least 3 models at different price points (e.g., Mistral small, Mistral large, Claude Sonnet).

3. Provide a `getSessionReport` method that returns:
   - Total calls, total tokens (in/out), total cost
   - Average cost per call
   - Most expensive call details
   - Cost breakdown by model

4. Test with a sequence of 10 LLM calls across different models. Verify that cumulative cost matches the sum of individual call costs.

**Create:** `src/cost/exercises/cost-tracker.ts`

**Expected output:** A session cost report showing per-call costs, cumulative totals, and a breakdown by model.

### Exercise 4: Token Budget Allocator

Build a budget allocator that distributes a context window across competing components and warns on overflows.

**Specification:**

1. Create a `BudgetAllocator` class that takes a total context window size and allocates portions:
   - System prompt: fixed allocation (configurable)
   - Tool definitions: fixed allocation (configurable)
   - Conversation history: dynamic (fills remaining space)
   - Tool results: capped per result (configurable)
   - Output reserved: fixed allocation (configurable)

2. Implement a `checkBudget` method that takes the current sizes of each component and returns:
   - Whether each component is within budget
   - How much space remains for conversation history
   - Warnings for any component exceeding its allocation
   - A recommendation if total usage exceeds the window (e.g., "compact conversation" or "truncate tool results")

3. Implement an `allocate` method that, given current component sizes, returns a plan: which components to truncate and by how much to fit within the window.

4. Test with scenarios: normal usage (everything fits), tool result overflow (one large result), conversation overflow (long history), and combined pressure (multiple components over budget).

**Create:** `src/cost/exercises/budget-allocator.ts`

**Expected output:** Budget reports for each scenario showing allocations, warnings, and truncation recommendations.

### Exercise 5: Tool Result Truncation

Build an intelligent tool result truncator that reduces token usage while preserving the most useful content.

**Specification:**

1. Create a `ResultTruncator` with configurable max tokens per result.

2. Implement three truncation strategies:
   - `headOnly` — keep the first N tokens
   - `headTail` — keep 60% from the start and 40% from the end, insert a `[truncated]` marker
   - `smart` — for code files, preserve import statements and the function closest to a target line number; for text, preserve the first paragraph and a keyword-relevant section

3. Track truncation statistics: how many results were truncated, total tokens saved, average reduction percentage.

4. Test with 5 tool results of varying sizes (1K, 5K, 10K, 20K, 50K tokens). Compare the three strategies on usefulness (does the truncated result still contain the key information?) and savings.

**Create:** `src/cost/exercises/result-truncation.ts`

**Expected output:** A comparison table showing each strategy's token savings and a qualitative assessment of preserved information for each test input.

### Exercise 6: Reasoning Effort Selector

Build a reasoning effort controller that adjusts model parameters based on task complexity to reduce costs on simple tasks.

**Specification:**

1. Create a `classifyComplexity` function that takes a user message and returns an effort level (`low`, `medium`, `high`) based on heuristics:
   - Message length (short messages are likely simple)
   - Presence of code blocks (suggests higher complexity)
   - Keywords like "refactor," "debug," "architect," "design" (high complexity)
   - Keywords like "rename," "typo," "format," "fix import" (low complexity)

2. Create a `withReasoningEffort` wrapper that takes an effort level and returns the appropriate `providerOptions` for the active provider (Anthropic thinking budget, OpenAI reasoning effort, or a no-op for providers without reasoning control).

3. Test with 10 prompts spanning the complexity range:
   - 3 simple tasks (rename variable, fix typo, add import)
   - 4 medium tasks (write a function, explain a concept, add error handling)
   - 3 complex tasks (refactor architecture, debug race condition, design a system)

4. Verify that simple tasks are classified as low effort and complex tasks as high effort. Compare the token usage (especially thinking tokens) across effort levels for the same prompt.

**Create:** `src/cost/exercises/reasoning-effort.ts`

**Expected output:** A table showing each prompt, its classified effort level, the provider options applied, and token usage comparison demonstrating cost savings on simple tasks.
