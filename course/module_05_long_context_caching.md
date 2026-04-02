# Module 5: Long Context & Caching

## Learning Objectives

- Understand the evolution of context windows and what large context enables
- Evaluate when to use full-context approaches vs retrieval-augmented generation
- Implement prompt caching with Groq's automatic caching and understand Anthropic and OpenAI equivalents
- Understand the KV cache mechanism inside transformer models
- Apply context compression techniques to reduce token usage and cost
- Build a decision framework for caching vs RAG in production systems

---

## Why Should I Care?

The cost equation of LLM applications changed fundamentally when context windows grew from 4K tokens to 200K and beyond. Tasks that previously required complex retrieval pipelines — searching, chunking, re-ranking — can now be solved by dropping the entire document into the context window and asking a question.

But "can" does not mean "should." Large context windows are expensive. Sending 100K tokens costs 25x more than sending 4K tokens, and latency scales with input size. Prompt caching changes this equation dramatically: a cached 100K-token prefix might cost 90% less than an uncached one. Understanding caching is the difference between a $500/month application and a $5,000/month one.

This module teaches you to think about context length as a strategic decision with cost, latency, and quality trade-offs. You will implement prompt caching, measure its impact, and build a framework for deciding when to cache, when to use RAG, and when to just use the full context.

---

## Connection to Other Modules

- **Module 4 (Conversations & Memory)** introduced the problem of managing context within limits. This module explores what happens when those limits are very large.
- **Module 6 (Streaming)** shows how to handle the latency of large-context calls through streaming.
- **Module 8 (Embeddings)** and **Module 9 (RAG)** provide the alternative approach: retrieval instead of full context.
- **Module 2 (Prompt Engineering)** taught prompt design. Here you learn how prompt structure affects cacheability.

---

## Section 1: Context Window Evolution

### The Timeline

The history of context windows is a story of exponential growth:

| Year | Model          | Context Window | Equivalent      |
| ---- | -------------- | -------------- | --------------- |
| 2022 | GPT-3          | 4K tokens      | ~3 pages        |
| 2023 | GPT-4          | 8K-32K tokens  | ~6-24 pages     |
| 2023 | Claude 2       | 100K tokens    | ~75 pages       |
| 2024 | Claude 3       | 200K tokens    | ~150 pages      |
| 2024 | Gemini 1.5 Pro | 1M-2M tokens   | ~750-1500 pages |
| 2025 | Claude 3.5     | 200K tokens    | ~150 pages      |
| 2025 | GPT-4o         | 128K tokens    | ~96 pages       |

### What Large Context Enables

With 200K tokens, you can fit:

- An entire novel (typical novel: 70K-100K words, ~90K-130K tokens)
- A full codebase of a medium project (~500 files of typical size)
- Hundreds of pages of documentation
- Hours of meeting transcripts
- Entire databases of product descriptions

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { readFile } from 'node:fs/promises'

async function analyzeDocument(filePath: string, question: string): Promise<string>
```

Build a function that reads a file from disk, estimates its token count (divide character length by 4), and sends the entire document to `generateText` in a user message wrapped in `<document>` tags. Include a system message instructing the model to cite specific sections. Log the estimated and actual token usage from the `usage` property on the result. What should you do if `usage` is undefined?

> **Beginner Note:** Just because you can fit an entire book in the context window does not mean the model processes it all equally well. Research shows that models pay most attention to the beginning and end of the context (the "lost in the middle" effect). Place the most important information at the start or end of your input.

### The "Lost in the Middle" Problem

Studies have shown that LLMs struggle with information placed in the middle of very long contexts. This affects how you structure large inputs:

```typescript
async function structuredLongContext(document: string, question: string): Promise<string>
```

Build a function that places the **question before the document** in the user message, so the model reads the question first and knows what to look for. After the document, repeat the question as a reminder. The system message should instruct the model to focus on relevant sections and quote source text. Why does placing the question at both the start and end of the user message help with long contexts?

> **Advanced Note:** The "lost in the middle" effect varies by model and has been improving with each generation. Claude 3.5 models handle long contexts significantly better than earlier versions. However, placing key information at the extremes of your context is still good practice as a reliability measure.

---

## Section 2: Long Document Strategies

### Strategy 1: Full Context

Send the entire document to the model. Simple, accurate, but expensive.

```typescript
interface DocumentQAResult {
  answer: string
  tokensUsed: number
  estimatedCost: number
}

async function fullContextQA(document: string, question: string): Promise<DocumentQAResult>
```

Build a function that sends the full document to `generateText`, then computes cost from usage data. Extract `inputTokens` and `outputTokens` from `usage` (default to 0 if undefined). To estimate cost, use per-million-token pricing: `(inputTokens / 1_000_000) * inputRate + (outputTokens / 1_000_000) * outputRate`. Return the answer text, total tokens, and estimated cost.

### Strategy 2: Chunked Retrieval (RAG)

Split the document, embed chunks, retrieve relevant ones, send only those. Complex but cheap for many queries.

```typescript
// Conceptual overview — full implementation in Module 9
interface Chunk {
  text: string
  index: number
  tokens: number
}

function chunkDocument(document: string, chunkSize: number = 500): Chunk[]
```

Build a function that splits a document into chunks of approximately `chunkSize` tokens. Split on paragraph boundaries (`\n\n`). Accumulate paragraphs into a current chunk until adding the next paragraph would exceed the token limit, then start a new chunk. Estimate tokens as `Math.ceil(text.length / 4)`. What happens if a single paragraph exceeds the chunk size?

### Decision Matrix: Full Context vs RAG

| Factor                   | Full Context                    | RAG                                |
| ------------------------ | ------------------------------- | ---------------------------------- |
| Setup complexity         | Low (just send it)              | High (embed, store, retrieve)      |
| Per-query cost           | High (process everything)       | Low (process only relevant chunks) |
| Accuracy                 | High (model sees everything)    | Variable (depends on retrieval)    |
| Latency                  | Higher (more tokens)            | Lower (fewer tokens)               |
| Multi-query amortization | Poor (pay full cost each time)  | Good (embed once, query many)      |
| Document updates         | Trivial (just send updated doc) | Requires re-embedding              |

> **Beginner Note:** As a general rule, if you are asking fewer than 5 questions about a document, full context is usually cheaper and simpler. If you are asking dozens or hundreds of questions, RAG becomes more economical.

---

## Section 3: Prompt Caching

### What is Prompt Caching?

Prompt caching allows the model provider to store the processed representation of your prompt prefix. When you send the same prefix again, the provider reuses the cached computation instead of reprocessing it from scratch.

This is different from application-level caching (storing responses). Prompt caching happens at the infrastructure level — the model provider caches the internal state (KV cache) computed from your prompt prefix.

### Groq Automatic Prompt Caching

Groq supports automatic prompt caching on GPT-OSS models — no code changes or explicit markers required. The platform detects repeated prompt prefixes and caches the computed KV state automatically:

```typescript
import { generateText } from 'ai'
import { groq } from '@ai-sdk/groq'

async function queryWithCaching(question: string): Promise<string>
```

Build a function that takes a question and sends it to `generateText` using `groq('openai/gpt-oss-20b')`. The key pattern: place the large static reference document in the message **before** the dynamic question. Groq automatically detects repeated prompt prefixes and caches them.

After the call, check the `usage` object. Groq reports cached tokens in `providerMetadata` — cast usage and look for a `cachedTokens` property. Log total input tokens, output tokens, and cached tokens.

Test by calling the function multiple times with different questions against the same document. The first call computes and caches the prefix. Subsequent calls should show a non-zero `cachedTokens` value, indicating a cache hit at 50% discount.

> **Beginner Note:** Groq's caching is fully automatic. The platform detects when your prompt starts with the same prefix as a recent request and reuses the cached computation. Cached tokens cost 50% less and do not count toward your rate limits. The cache expires after 2 hours without use.

### Anthropic Explicit Caching

Anthropic takes a different approach: you explicitly mark cache breakpoints using `cache_control` in `providerOptions`. This gives you precise control over what gets cached:

Anthropic requires explicit cache markers. Instead of a single string for the user message content, you pass an **array of content blocks**. The block containing the static document gets a `providerOptions` property:

```typescript
{
  type: 'text',
  text: `Reference documentation:\n${referenceDocument}`,
  providerOptions: {
    anthropic: { cacheControl: { type: 'ephemeral' } },
  },
}
```

The dynamic question goes in a separate content block without cache markers. After the call, check `usage` for `cacheReadInputTokens` and `cacheCreationInputTokens` to see cache behavior.

> **Intermediate Note:** The `cache_control: { type: 'ephemeral' }` marker tells Anthropic "cache everything up to this point." The cache lives about 5 minutes (refreshed each time you use it). You only pay the cache creation cost on the first call. Anthropic's cache offers up to 90% savings on cached tokens — a deeper discount than Groq's 50%, but requires explicit markers and has a shorter TTL.

### OpenAI Automatic Caching

OpenAI also provides automatic caching, similar to Groq. Prompts of 1024+ tokens are automatically cached with a 50% discount on cached input tokens. No code changes needed — just structure your prompts with the static prefix first.

### Cache Placement Strategy

The key to effective caching is identifying the **static prefix** — the part of your prompt that does not change between calls. With automatic caching (Groq, OpenAI), the cache matches from the beginning of the prompt up to the first difference. With explicit caching (Anthropic), you mark the boundary yourself.

```typescript
async function cachedDocumentQA(
  staticContext: string,
  dynamicQuery: string
): Promise<{ answer: string; cacheHit: boolean }>
```

Build a function that separates static and dynamic content across messages. Place the static context inside the **system message** (wrapped in `<context>` tags) — this becomes the cached prefix. The dynamic query goes in a separate **user message**. After the call, check `usage` for `cachedTokens` to determine if a cache hit occurred (any value > 0 means a hit). Return both the answer and a boolean indicating whether the cache was hit.

Test by loading a document from disk and running multiple queries against it in a loop, logging whether each query got a cache hit.

### Multi-turn Caching

Caching is particularly effective for multi-turn conversations where the system prompt and early context remain constant. With automatic caching, the static system prompt prefix is cached across turns without any special configuration:

```typescript
class CachedConversation {
  private messages: Array<{ role: 'user' | 'assistant'; content: string }> = []
  private systemContent: string
  private model: string

  constructor(config: { systemPrompt: string; staticContext: string; model?: string })
  async send(userMessage: string): Promise<{ text: string; cached: boolean }>
}
```

Build a class that maintains a conversation with a cached prefix. The constructor combines the system prompt and static context into a single `systemContent` string (with the context wrapped in `<context>` tags) — this becomes the stable prefix that Groq caches automatically.

The `send` method pushes the user message onto the internal messages array, calls `generateText` with the system content followed by all accumulated messages, pushes the assistant response, and returns both the text and a cache-hit boolean. How does the growing conversation history after the cached prefix affect cache behavior? (Hint: only the prefix is cached — the conversation turns after it are always computed fresh.)

> **Intermediate Note:** With Anthropic's explicit caching, you would add `providerOptions: { anthropic: { cacheControl: { type: 'ephemeral' } } }` to the system message content block. Anthropic also has minimum size requirements (1024 tokens for Claude Sonnet 4, 2048 tokens for Claude Haiku 4.5) — content smaller than this threshold will not be cached. Groq's minimum is lower (128–1024 tokens depending on model) and requires no explicit markers.

### Comparing Caching Approaches

| Feature                   | Groq (automatic)       | Anthropic (explicit)          | OpenAI (automatic)  |
| ------------------------- | ---------------------- | ----------------------------- | ------------------- |
| Setup required            | None                   | `cache_control` markers       | None                |
| Discount on cached tokens | 50%                    | 90%                           | 50%                 |
| Cache TTL                 | 2 hours                | ~5 minutes (refreshed on use) | ~5–10 minutes       |
| Min cacheable tokens      | 128–1024               | 1024–2048                     | 1024                |
| Rate limit relief         | Cached tokens excluded | No                            | No                  |
| Control over boundaries   | None (prefix-based)    | Explicit breakpoints          | None (prefix-based) |

Anthropic's explicit approach gives you precise control and deeper discounts, making it the best choice when you need fine-grained cache boundary management or when you are optimizing heavily for cost. Groq's automatic approach is simpler — just structure your prompts correctly and caching happens for free.

---

## Section 4: KV Cache Concepts

### What Happens Inside the Model

When a transformer model processes your prompt, it computes two matrices for each layer: **Keys (K)** and **Values (V)**. Together, these form the "KV cache." Understanding this mechanism helps you reason about caching behavior.

### The Attention Mechanism (Simplified)

For each token the model generates:

1. It computes a **Query (Q)** from the current token
2. It compares Q against all **Keys (K)** from previous tokens to determine relevance
3. It uses those relevance scores to weight the **Values (V)** and produce the output

```
Input tokens:  [The] [cat] [sat] [on] [the] [mat]
                 ↓     ↓     ↓    ↓    ↓     ↓
Compute:        K,V   K,V   K,V  K,V  K,V   K,V
                 ↓     ↓     ↓    ↓    ↓     ↓
KV Cache:      [K1,V1][K2,V2][K3,V3][K4,V4][K5,V5][K6,V6]

When generating next token:
  - Compute Q for [the next token]
  - Compare Q against all cached Keys → attention scores
  - Weight cached Values by attention scores → output
```

### Why Prompt Caching Works

Computing the KV cache for your prompt is the expensive part. If the prompt prefix is the same as a previous request, the provider can skip this computation and reuse the cached KV pairs. The model only needs to compute KV pairs for the new (uncached) portion of the input.

```
Request 1: [System prompt + Document] + [Question 1]
            ├── 50K tokens ──────────┤  ├─ 50 tokens ─┤
            Compute KV (expensive)       Compute KV (cheap)
            Cache this prefix            Not cached

Request 2: [System prompt + Document] + [Question 2]
            ├── 50K tokens ──────────┤  ├─ 50 tokens ─┤
            Read from cache (fast!)      Compute KV (cheap)
```

The savings come from two places:

1. **Compute**: The provider does not need to run the forward pass through all transformer layers for the cached prefix
2. **Cost**: Providers pass these savings to you through reduced per-token pricing for cached tokens

### KV Cache Size

The KV cache grows linearly with context length and model size:

```typescript
interface ModelConfig {
  layers: number
  heads: number
  headDim: number
  bytesPerParam: number // 2 for float16, 1 for int8
}

function estimateKVCacheSize(config: ModelConfig, sequenceLength: number): { bytes: number; megabytes: number }
```

Build a function that estimates the GPU memory required for the KV cache. The formula is: `bytesPerToken = 2 (K and V) * layers * heads * headDim * bytesPerParam`. Multiply by `sequenceLength` for total bytes, then convert to megabytes. Try it with a Claude-like config (80 layers, 64 heads, 128 headDim, 2 bytes) at 200K tokens — what does the memory requirement look like? This is per-request GPU memory, which is why long-context requests are expensive at scale.

> **Advanced Note:** The KV cache is why longer contexts cost more, even beyond the linear token pricing. Providers need GPU memory to store the cache for each concurrent request. This is a fundamental hardware constraint that affects pricing and availability for long-context requests.

---

## Section 5: Context Compression Techniques

### Why Compress?

Even with large context windows, compression helps:

- Reduce cost (fewer tokens = lower price)
- Reduce latency (fewer tokens = faster processing)
- Fit more information into fixed windows

### Technique 1: Document Preprocessing

Remove irrelevant content before sending to the model:

```typescript
function preprocessDocument(text: string): string

function compressionStats(
  original: string,
  compressed: string
): { originalTokens: number; compressedTokens: number; savings: string }
```

Build two functions. `preprocessDocument` should strip unnecessary content before sending to the model. Use regex replacements to: collapse multiple blank lines into one (`/\n{3,}/g`), collapse excessive whitespace, remove common boilerplate (copyright lines, page numbers), strip HTML tags, and trim each line. Return the cleaned text.

`compressionStats` compares original and compressed text by estimating tokens (`Math.ceil(length / 4)`) and computing savings as a percentage. What kinds of documents benefit most from this preprocessing?

### Technique 2: Selective Inclusion

Only include relevant sections of a document:

```typescript
interface DocumentSection {
  title: string
  content: string
  tokens: number
}

function parseDocumentSections(document: string): DocumentSection[]

async function selectRelevantSections(
  sections: DocumentSection[],
  question: string,
  maxTokens: number
): Promise<DocumentSection[]>
```

Build two functions. `parseDocumentSections` splits a markdown document on `## ` headers, extracting each section's title (first line), content (remaining lines), and estimated token count.

`selectRelevantSections` uses the LLM itself to pick which sections are relevant. Send a numbered list of section titles to `generateText` with a system prompt that says "return the indices of relevant sections as comma-separated numbers." Parse the response into indices, filter out invalid ones, then select sections greedily until the `maxTokens` budget is exhausted. What happens if the LLM returns indices that are out of range? How do you handle that?

### Technique 3: LLM-Based Compression

Use a smaller, cheaper model to compress content before sending to the main model:

```typescript
async function compressContext(text: string, targetRatio: number = 0.3): Promise<string>
```

Build a function that uses a cheaper model to compress content. Calculate the target token count from the original text length and `targetRatio`. Send the text to `generateText` with a system prompt that instructs the model to compress to approximately that many tokens, preserving key facts, names, numbers, and relationships while removing redundancy and examples. Log the compression ratio achieved. Use the compressed output in place of the original document in subsequent prompts.

> **Advanced Note:** LLM-based compression risks losing important details. Always validate that key facts survive compression, especially numerical data, proper nouns, and causal relationships. Consider using structured extraction (Module 3) to pull out key facts before compression.

---

## Section 6: Chunked Prefill Patterns

### What is Chunked Prefill?

When processing very long contexts, the model processes the input in chunks rather than all at once. Understanding this pattern helps you structure your inputs for optimal performance.

### Structuring Input for Efficient Processing

```typescript
async function structuredLongDocument(
  sections: Array<{ title: string; content: string }>,
  question: string
): Promise<string>
```

Build a function that structures long input for efficient model processing. First, build a table of contents from the section titles (e.g., `[Section 1] Introduction`). Then format each section with XML-like delimiters: `<section id="1" title="...">content</section>`. Send the TOC followed by the sections, then the question. The system message should instruct the model to use the TOC for navigation and cite section numbers. Why does a table of contents help the model even though it adds tokens?

### Hierarchical Context Organization

For extremely long inputs, organize content hierarchically:

```typescript
interface HierarchicalDocument {
  summary: string // ~500 tokens
  sectionSummaries: Array<{ title: string; summary: string }> // ~50 tokens each
  sections: Array<{ title: string; content: string }> // Full content
}

async function buildHierarchicalDocument(document: string): Promise<HierarchicalDocument>

async function hierarchicalQA(doc: HierarchicalDocument, question: string): Promise<string>
```

Build two functions that implement a hierarchical approach to long documents.

`buildHierarchicalDocument` parses a document into sections (split on `# ` headers), then makes two LLM calls: one to generate 1-2 sentence summaries for each section (return as JSON), and another to generate an overall summary from those section summaries. The result is a three-level structure: full document summary, per-section summaries, and full section content.

`hierarchicalQA` uses this structure in two steps. First, send the document summary and section summaries to the LLM and ask which sections are relevant to the question (return comma-separated indices). Second, fetch only those sections' full content and send them with the question to get the answer. How does this two-step approach reduce cost compared to always sending the full document?

> **Beginner Note:** Hierarchical organization adds complexity but dramatically reduces cost for repeated queries against the same document. Build the hierarchy once and query many times.

---

## Section 7: Cost Implications

### Cached vs Uncached Pricing

Prompt caching can dramatically reduce costs. Here is a comparison using Groq's GPT-OSS pricing:

```typescript
interface PricingTier {
  inputPer1M: number
  outputPer1M: number
  cachedInputPer1M: number // Price for cached input tokens
}

function calculateGroqCost(
  pricing: PricingTier,
  usage: { inputTokens: number; outputTokens: number; cachedTokens: number }
): { total: number; breakdown: Record<string, number> }

function compareCachingScenarios(): void
```

Build a cost calculator and a comparison function. Reference pricing (verify against current rates):

- **Groq GPT-OSS 20B:** $0.075/1M input, $0.30/1M output, $0.0375/1M cached input (50% discount)
- **Anthropic Sonnet:** $3.0/1M input, $15.0/1M output, $3.75/1M cache write, $0.30/1M cache read

`calculateGroqCost` separates input tokens into uncached (`inputTokens - cachedTokens`) and cached portions, applies different rates to each, adds output cost, and returns a total with a breakdown.

`compareCachingScenarios` models a scenario: 50K-token document, 10 questions, 200-token responses. Compare the cost with zero cached tokens vs the cost where 9 out of 10 queries hit the cache (first query pays full price). Print both totals and the percentage savings. How significant are the savings at this scale?

### Break-Even Analysis

With automatic caching (Groq, OpenAI), there is no cache write surcharge — you simply pay 50% less for cached tokens. This means caching is beneficial from the very first cache hit (the second query):

```typescript
function compareCachingSavings(documentTokens: number, numQueries: number): void
```

Build a function that compares caching savings across providers. For Groq: first query at full price, subsequent queries at 50% discount, no write surcharge. For Anthropic: first query at 125% (cache write surcharge), subsequent queries at 10% (90% discount on reads). Compute the savings percentage for each provider. Test with 2, 10, and 100 queries against a 50K-token document. At what query count does Anthropic's deeper discount overtake Groq's simpler pricing?

> **Advanced Note:** Cache lifetime matters for both approaches. Groq's cache lasts about 2 hours without use — generous for most interactive and batch workloads. Anthropic's ephemeral cache lasts about 5 minutes but is refreshed with each use. OpenAI's cache lasts 5–10 minutes. For high-traffic applications, all caches stay warm naturally. For infrequent queries, Groq's longer TTL is a significant advantage.

---

## Section 8: When to Cache vs When to RAG

### The Decision Framework

Choosing between prompt caching and RAG depends on several factors:

```typescript
interface UseCase {
  documentSize: 'small' | 'medium' | 'large' | 'huge'
  queryFrequency: 'one-off' | 'periodic' | 'high-volume'
  documentUpdateFrequency: 'static' | 'daily' | 'real-time'
  accuracyRequirements: 'best-effort' | 'high' | 'critical'
  latencyRequirements: 'relaxed' | 'moderate' | 'strict'
}

function recommendStrategy(useCase: UseCase): {
  strategy: 'full-context' | 'cached-context' | 'rag' | 'hybrid'
  reasoning: string
}
```

Build a decision function that recommends a strategy based on the use case properties. Work through the logic in this order:

1. **Small documents** — if query frequency is high-volume, recommend `cached-context`; otherwise `full-context` (no need for complexity).
2. **Huge documents** — always `rag` (they exceed the context window).
3. **High-volume + static + critical accuracy** — `cached-context` (full context is most accurate, caching amortizes cost).
4. **High-volume + static + non-critical** — `rag` (cost efficiency wins when accuracy is not paramount).
5. **Real-time updates** — `rag` (cache invalidation makes caching impractical).
6. **Default** — `cached-context` for moderate use cases.

Each return should include a `reasoning` string explaining the choice. Test with several scenarios to verify the logic.

### Decision Tree (Summary)

```
Is the document < context window?
├── No → RAG (no choice)
└── Yes
    ├── Single query? → Full context (simplest)
    └── Multiple queries?
        ├── Document changes frequently? → RAG
        └── Document is static?
            ├── High accuracy needed? → Cached context
            └── Cost sensitive? → RAG
```

### Hybrid: Cache + RAG

For some use cases, combining both strategies works best:

```typescript
async function hybridDocumentQA(config: {
  documentSummary: string // Cached automatically (static prefix)
  retrievedChunks: string[] // From vector search (dynamic suffix)
  question: string
}): Promise<string>
```

Build a function that combines caching and RAG. Place the document summary in the **system message** (this becomes the cached prefix). Place the retrieved chunks and question in the **user message** (this is the dynamic suffix). The system prompt should instruct the model to use the overview for general context and retrieved sections for specific details. Why is this hybrid approach sometimes better than either strategy alone?

> **Beginner Note:** Start with the simplest approach that works. If your document fits in the context window and you are making a few queries, use full context. Add caching when you notice cost issues. Add RAG when documents are too large. Do not over-engineer from the start.

> **Local Alternative (Ollama):** Prompt caching through the API is a cloud provider feature (Groq, Anthropic, OpenAI). Ollama does not offer explicit prompt caching through its API — but llama.cpp (which powers Ollama) automatically caches the KV state for repeated prefixes, giving you similar benefits transparently. The long context strategies (chunking, summarization, map-reduce) work with any provider. Note that most Ollama models have smaller context windows (8K-32K), so the context management techniques here are even more critical.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Auto-Compact Systems

### Token Monitoring and Automatic Compaction

Production LLM applications that run for extended sessions (coding assistants, chat agents, long research tasks) must continuously monitor token usage and compact the conversation before hitting the context limit. This is the runtime version of the compression techniques from Section 5 — but triggered automatically rather than manually.

The pattern works in three stages:

1. **Monitor** — After every API response, check `usage.totalTokens` against the model's context window
2. **Warn** — At 80% capacity, log a warning and consider deferring non-essential context
3. **Compact** — At 90% capacity, summarize older messages and replace them with a condensed version

```typescript
const WARN_THRESHOLD = 0.8
const COMPACT_THRESHOLD = 0.9

// After each generateText call, check usage against the model's max context
const usageRatio = usage.totalTokens / modelMaxTokens
if (usageRatio >= COMPACT_THRESHOLD) {
  messages = await compactMessages(messages)
}
```

The compaction step itself is a separate `generateText` call (often to a smaller, cheaper model) that summarizes the conversation so far into a condensed system message. The key design decision is what to preserve: recent messages stay verbatim, older messages get summarized, and the system prompt remains untouched.

### Configurable Thresholds Per Model

Different models have different context windows, so thresholds must be model-aware. A 200K-token model has more room before compaction is needed than a 32K-token model. Production systems map model IDs to their context sizes and compute thresholds dynamically.

---

## Section 10: Cache-Friendly Message Ordering

### Structuring Messages for Maximum Cache Hits

Prompt caching works on prefixes — the provider caches the KV state computed from the beginning of the prompt. Any change in early messages invalidates the cache for everything after it. This has a direct implication for how you structure API calls:

1. **Static content first** — System prompt, tool definitions, reference documents, and any fixed instructions go at the beginning. These rarely change between calls.
2. **Semi-static content next** — Retrieved context, memory summaries, and session metadata that change infrequently.
3. **Dynamic content last** — The user's latest message and recent conversation turns go at the end.

```typescript
const messages = [
  { role: 'system', content: systemPrompt }, // cached — rarely changes
  { role: 'system', content: toolDefinitions }, // cached — rarely changes
  { role: 'user', content: retrievedContext }, // semi-stable
  ...recentConversation, // dynamic — changes every turn
  { role: 'user', content: currentUserMessage }, // always new
]
```

If you put the user's message before the reference document, every new query invalidates the cache for the (potentially large) document. By keeping static content at the front, the provider reuses the cached prefix and only processes the new dynamic suffix. This is the difference between caching 90% of your tokens and caching 0%.

---

## Section 11: Memory Management for Long-Running LLM Apps

### Applicable Memory Patterns

Long-running LLM applications — agents that operate for hours, chat sessions with hundreds of turns — face memory management challenges beyond just token counts. Several patterns from systems programming apply directly:

**LRU Caches** — Keep the N most recently used items (embeddings, tool results, retrieved documents) and evict the oldest when the cache is full. This prevents unbounded memory growth while keeping frequently accessed data fast to retrieve.

**Circular Buffers** — Store the last N messages in a fixed-size buffer that overwrites the oldest entry when full. This is ideal for "recent context" windows where you always want the last K turns regardless of total conversation length.

**Token Monitoring** — Track cumulative token usage across the session and trigger cleanup actions (compaction, cache eviction, summary generation) when thresholds are exceeded. This is the auto-compact system from Section 9 generalized to all resource types.

**WeakRef for Cleanup** — Use `WeakRef` for references to large objects (parsed documents, embedding vectors) that should be garbage collected when no longer actively used. This prevents memory leaks in long sessions without requiring explicit cleanup code.

These patterns complement each other. A production system might use an LRU cache for embeddings, a circular buffer for recent messages, and token monitoring to trigger compaction — all running simultaneously.

---

## Section 12: Model Variants and Thinking Budgets

### Thinking Tokens as a Context Budget Decision

Some models support configurable reasoning effort — the model spends more tokens "thinking" internally before producing output. This directly trades context capacity for answer quality:

- **Anthropic:** `thinking` budget levels via `providerOptions` (e.g., `{ anthropic: { thinking: { type: 'enabled', budgetTokens: 8192 } } }`)
- **OpenAI:** `reasoningEffort` levels — `'low'`, `'medium'`, `'high'`
- **Google:** thinking budget as a token count

More thinking tokens means fewer tokens available for conversation history. A 200K context window with 10K thinking budget has 190K usable for messages. Production systems expose this as a user-facing "variant" selector — a quick question uses a fast variant with minimal thinking, while a complex task uses the max-thinking variant.

```typescript
// Fast variant — minimal thinking, preserves context space
providerOptions: { anthropic: { thinking: { type: 'enabled', budgetTokens: 1024 } } }

// Deep variant — extensive reasoning, consumes more context
providerOptions: { anthropic: { thinking: { type: 'enabled', budgetTokens: 8192 } } }
```

When building context budgets (Section 8's decision framework), account for thinking tokens as a reserved allocation alongside system prompt, tools, and conversation history.

---

## Summary

In this module, you learned:

1. **Context window evolution:** Context windows have grown from 4K to 200K+ tokens, enabling new approaches but introducing cost and latency trade-offs.
2. **Full context vs RAG:** Dropping entire documents into the context window is simple and effective for smaller corpora, while RAG is better for large or frequently changing data.
3. **Prompt caching:** Groq automatically caches repeated prompt prefixes on GPT-OSS models (50% discount, 2-hour TTL). Anthropic offers explicit `cache_control` markers for up to 90% savings. Both approaches reduce cost and latency on repeated calls.
4. **Cache placement strategy:** Placing static content (system prompts, reference documents) at the beginning of your prompt and dynamic content (user queries) at the end maximizes cache hit rates for both automatic and explicit caching.
5. **KV cache internals:** Understanding how transformers store key-value pairs explains why prompt caching works and why prefix stability matters.
6. **Context compression:** Preprocessing documents, selectively including content, and using LLM-based summarization reduce token usage while preserving the information the model needs.
7. **Decision framework:** Choosing between full context, caching, and RAG depends on document size, query frequency, freshness requirements, and budget constraints.
8. **Auto-compact systems:** Production applications monitor token usage continuously and trigger compaction automatically at configurable thresholds.
9. **Cache-friendly ordering:** Placing static content first and dynamic content last in messages maximizes prompt cache hit rates.
10. **Memory management:** LRU caches, circular buffers, token monitoring, and WeakRef are complementary patterns for managing resources in long-running LLM applications.
11. **Thinking budgets:** Configurable reasoning effort trades context capacity for answer quality — a direct context budget decision.

In Module 6, you will dive deep into streaming — delivering LLM responses to users in real time for better perceived performance.

---

## Quiz

### Question 1 (Easy)

What is the primary benefit of prompt caching?

A) It makes the model more accurate
B) It reduces cost and latency for repeated prompt prefixes
C) It increases the context window size
D) It stores conversation history between sessions

**Answer: B**

Prompt caching stores the internal KV cache computed from a prompt prefix. When the same prefix is sent again, the provider reuses the cached computation, reducing both cost (50% on Groq/OpenAI, up to 90% on Anthropic) and latency (no need to reprocess the prefix).

---

### Question 2 (Easy)

A 200K-token context window can hold approximately how many pages of English text?

A) 20 pages
B) 50 pages
C) 150 pages
D) 500 pages

**Answer: C**

A typical page of English text contains roughly 250-300 words, which translates to about 350-400 tokens. At 200,000 tokens, that is approximately 500-570 pages. However, the standard approximation used is ~150K words for 200K tokens, which at ~1000 words per page gives ~150 pages. The exact number depends on content density and formatting.

---

### Question 3 (Medium)

The "lost in the middle" problem refers to:

A) Models losing track of the conversation topic
B) Models paying less attention to information in the middle of long contexts
C) Tokens being dropped when the context window is exceeded
D) Cache entries expiring before they are used

**Answer: B**

Research has shown that LLMs tend to focus more attention on content at the beginning and end of the context window, with reduced attention to content in the middle. This means important information should be placed at the beginning or end of your input for best results.

---

### Question 4 (Medium)

When does prompt caching break even compared to uncached requests?

A) After approximately 100 queries
B) After approximately 2-3 queries (for most document sizes)
C) It never breaks even because cache writes are more expensive
D) Only after the cache has been alive for 24 hours

**Answer: B**

With automatic caching (Groq, OpenAI), there is no write surcharge — cached tokens simply cost 50% less, so you save from the very first cache hit (2nd query). With Anthropic's explicit caching, cache writes cost 25% more than regular input, but cache reads cost 90% less — breaking even after 2-3 queries. Either way, caching almost always saves money for repeated queries against the same context.

---

### Question 5 (Hard)

When should you prefer RAG over full-context with caching?

A) When the document is small and static
B) When you need the highest possible accuracy
C) When the document exceeds the context window or changes frequently
D) When you are making only one query

**Answer: C**

RAG is the better choice when documents are too large to fit in the context window (making full context impossible) or when documents change frequently (which would invalidate the cache). For small, static documents with high accuracy requirements, full context with caching is usually superior.

---

### Question 6 (Medium)

Why should static content (system prompt, tool definitions) be placed at the beginning of the message array rather than after dynamic content?

- A) The model pays more attention to content at the beginning
- B) Prompt caching works on prefixes — placing stable content first maximizes cache hit rates since dynamic content at the end does not invalidate the cached prefix
- C) The API requires system messages to come first
- D) Static content is always shorter than dynamic content

**Answer: B** — Prompt caching stores the KV state computed from the beginning of the prompt. Any change in early messages invalidates the cache for everything after it. By placing static content first, the provider reuses the cached prefix and only processes the new dynamic suffix. Putting dynamic content before static content would invalidate the cache on every request.

---

### Question 7 (Hard)

A production auto-compact system triggers compaction at 90% context capacity. The compaction itself is a `generateText` call to a smaller model. What design risk does this introduce, and how is it mitigated?

- A) The compaction model might hallucinate — mitigated by using structured output with a strict schema
- B) The compaction call consumes tokens from the already-full context — mitigated by using a separate model call with its own context window, preserving recent messages verbatim and only summarizing older ones
- C) The compaction happens too frequently — mitigated by caching the compaction result
- D) The smaller model cannot handle the conversation length — mitigated by switching to a larger model

**Answer: B** — The compaction call is a separate `generateText` invocation with its own context budget, not an addition to the already-full conversation. It takes the older messages, summarizes them into a condensed system message, and the main conversation replaces those messages with the summary. Recent messages stay verbatim to preserve fidelity, and the system prompt remains untouched.

---

## Exercises

### Exercise 1: Document Q&A with Prompt Caching

Build a document Q&A system that uses prompt caching and measures cost savings.

**Requirements:**

1. Load a text document (at least 10,000 words) from disk
2. Implement two modes: `cached` and `uncached`
3. Run 10 different questions against the document in each mode
4. Track token usage for each query (input, output, cache write, cache read)
5. Calculate and display total cost for each mode
6. Show the percentage savings from caching

**Starter code:**

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface QueryResult {
  question: string
  answer: string
  inputTokens: number
  outputTokens: number
  cacheWriteTokens: number
  cacheReadTokens: number
  latencyMs: number
}

async function runCachedQueries(document: string, questions: string[]): Promise<QueryResult[]> {
  // TODO: Implement with cache_control on the document
  throw new Error('Not implemented')
}

async function runUncachedQueries(document: string, questions: string[]): Promise<QueryResult[]> {
  // TODO: Implement without caching
  throw new Error('Not implemented')
}

function printComparison(cached: QueryResult[], uncached: QueryResult[]): void {
  // TODO: Calculate and display cost comparison
  // Show per-query and total costs
  // Show latency comparison
  // Show percentage savings
}

// Load document and run comparison
const document = await Bun.file('./data/sample-document.txt').text()
const questions = [
  'What is the main topic of this document?',
  'Who are the key people mentioned?',
  'What dates are referenced?',
  'Summarize section 3.',
  'What conclusions are drawn?',
  'Are there any numerical claims? List them.',
  'What methodology was used?',
  'What are the limitations mentioned?',
  'How does this relate to previous work?',
  'What are the recommendations?',
]

const cachedResults = await runCachedQueries(document, questions)
const uncachedResults = await runUncachedQueries(document, questions)
printComparison(cachedResults, uncachedResults)
```

### Exercise 2: Context Compression Pipeline

Build a pipeline that compresses a long document through multiple stages and measures quality vs token trade-offs.

**Requirements:**

1. Take a document of at least 20,000 tokens
2. Implement three compression levels: light (50%), medium (25%), heavy (10%)
3. Ask the same 5 factual questions at each compression level
4. Compare answer accuracy across compression levels
5. Track tokens and cost at each level

**Evaluation criteria:**

- Compression ratios hit their targets within 10%
- Answer accuracy degrades gracefully with compression level
- Cost savings are correctly calculated and displayed
- The pipeline handles edge cases (very short documents, documents with tables)

### Exercise 3: Auto-Compact Monitor

Build a middleware that wraps `generateText` calls, tracks cumulative token usage across a conversation, and triggers a compaction callback when usage exceeds configurable thresholds.

**What to build:** Create `src/memory/auto-compact.ts`

**Requirements:**

1. Define an `AutoCompactMonitor` that accepts a model's max context size and threshold percentages (warn at 80%, compact at 90%)
2. Expose a `trackUsage(usage: LanguageModelUsage)` method that accumulates token counts after each API call
3. When the warn threshold is hit, call an `onWarn` callback with current usage stats
4. When the compact threshold is hit, call an `onCompact` callback that receives the current messages array and returns a compacted version
5. Expose a `getUsageRatio(): number` method that returns current usage as a fraction of the context window
6. Support `reset()` to clear usage counters after compaction

**Expected behavior:**

- After 5 API calls each using ~5K tokens against a 32K context model, the monitor should trigger the warn callback around the 4th-5th call and the compact callback shortly after
- The `onCompact` callback receives messages and returns a shorter array — the monitor should reset its counters after compaction
- Different model context sizes produce different trigger points for the same token usage

**File:** `src/memory/exercises/auto-compact.ts`

### Exercise 4: Context Window Budget Allocator

Build a budget allocator that divides a model's context window into named segments and tracks consumption against each budget.

**What to build:** Create `src/memory/exercises/context-budget.ts`

**Requirements:**

1. Accept a model's total context window size and a budget configuration mapping segment names to token allocations (e.g., `{ systemPrompt: 20000, tools: 10000, memories: 10000, conversation: 150000, output: 10000 }`)
2. Validate that all segments sum to no more than the total context window
3. Expose `allocate(segment: string, tokens: number): boolean` that returns `false` if the segment would exceed its budget
4. Expose `remaining(segment: string): number` that returns how many tokens are left in that segment
5. Expose `summary(): Record<string, { used: number; budget: number; ratio: number }>` for a full overview
6. Support `release(segment: string, tokens: number)` to free tokens when messages are removed

**Expected behavior:**

- Allocating 15K tokens to a segment with a 10K budget returns `false` and does not change state
- After allocating 8K to a 10K segment, `remaining` returns 2K
- The summary shows all segments with their usage ratios
- Releasing tokens increases the remaining budget for that segment

**File:** `src/memory/exercises/context-budget.ts`
