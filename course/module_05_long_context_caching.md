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

// Load an entire book into context
async function analyzeDocument(filePath: string, question: string): Promise<string> {
  const document = await readFile(filePath, 'utf-8')

  const estimatedTokens = Math.ceil(document.length / 4)
  console.log(`Document size: ${document.length} characters (~${estimatedTokens} tokens)`)

  const { text, usage } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content:
          'You are a document analyst. Answer questions based on the provided document. Always cite specific sections.',
      },
      {
        role: 'user',
        content: `<document>\n${document}\n</document>\n\nQuestion: ${question}`,
      },
    ],
  })

  if (usage) {
    console.log(`Actual tokens - Input: ${usage.inputTokens}, Output: ${usage.outputTokens}`)
  }

  return text
}

// Usage
const answer = await analyzeDocument('./data/annual-report-2025.txt', 'What were the key revenue drivers in Q3?')
console.log(answer)
```

> **Beginner Note:** Just because you can fit an entire book in the context window does not mean the model processes it all equally well. Research shows that models pay most attention to the beginning and end of the context (the "lost in the middle" effect). Place the most important information at the start or end of your input.

### The "Lost in the Middle" Problem

Studies have shown that LLMs struggle with information placed in the middle of very long contexts. This affects how you structure large inputs:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Better: Important context at the beginning and end
async function structuredLongContext(document: string, question: string): Promise<string> {
  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content: `You are a precise document analyst. The user will provide a document and a question.
Focus your analysis on the specific sections relevant to the question.
Always quote the source text when answering.`,
      },
      {
        role: 'user',
        content: `Question (read this first): ${question}

<document>
${document}
</document>

Now answer the question above based on the document. Quote relevant passages.`,
      },
    ],
  })

  return text
}
```

> **Advanced Note:** The "lost in the middle" effect varies by model and has been improving with each generation. Claude 3.5 models handle long contexts significantly better than earlier versions. However, placing key information at the extremes of your context is still good practice as a reliability measure.

---

## Section 2: Long Document Strategies

### Strategy 1: Full Context

Send the entire document to the model. Simple, accurate, but expensive.

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface DocumentQAResult {
  answer: string
  tokensUsed: number
  estimatedCost: number
}

async function fullContextQA(document: string, question: string): Promise<DocumentQAResult> {
  const { text, usage } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content: 'Answer the question based on the provided document. Be precise and cite sections.',
      },
      {
        role: 'user',
        content: `<document>\n${document}\n</document>\n\nQuestion: ${question}`,
      },
    ],
  })

  const inputTokens = usage?.inputTokens ?? 0
  const outputTokens = usage?.outputTokens ?? 0

  // Claude Sonnet pricing (example — check current rates)
  const cost = (inputTokens / 1_000_000) * 3.0 + (outputTokens / 1_000_000) * 15.0

  return {
    answer: text,
    tokensUsed: inputTokens + outputTokens,
    estimatedCost: cost,
  }
}
```

### Strategy 2: Chunked Retrieval (RAG)

Split the document, embed chunks, retrieve relevant ones, send only those. Complex but cheap for many queries.

```typescript
// Conceptual overview — full implementation in Module 9
interface Chunk {
  text: string
  index: number
  tokens: number
}

function chunkDocument(document: string, chunkSize: number = 500): Chunk[] {
  const paragraphs = document.split('\n\n')
  const chunks: Chunk[] = []
  let currentChunk = ''
  let chunkIndex = 0

  for (const paragraph of paragraphs) {
    const estimatedTokens = Math.ceil((currentChunk + paragraph).length / 4)

    if (estimatedTokens > chunkSize && currentChunk) {
      chunks.push({
        text: currentChunk.trim(),
        index: chunkIndex++,
        tokens: Math.ceil(currentChunk.length / 4),
      })
      currentChunk = paragraph
    } else {
      currentChunk += '\n\n' + paragraph
    }
  }

  if (currentChunk.trim()) {
    chunks.push({
      text: currentChunk.trim(),
      index: chunkIndex,
      tokens: Math.ceil(currentChunk.length / 4),
    })
  }

  return chunks
}
```

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

// Large reference document that stays the same across queries
const referenceDocument = `... (imagine 50,000 tokens of documentation) ...`

async function queryWithCaching(question: string): Promise<string> {
  const { text, usage } = await generateText({
    model: groq('openai/gpt-oss-20b'),
    messages: [
      {
        role: 'system',
        content: 'You are a documentation expert. Answer questions based on the reference material.',
      },
      {
        // Static content FIRST — this is what gets cached
        role: 'user',
        content: `Reference documentation:\n${referenceDocument}\n\nQuestion: ${question}`,
      },
    ],
  })

  // Check cache performance via usage metadata
  if (usage) {
    console.log(`Total input tokens: ${usage.inputTokens}`)
    console.log(`Output tokens: ${usage.outputTokens}`)
    // Groq reports cached tokens in providerMetadata
    const providerUsage = usage as any
    console.log(`Cached tokens: ${providerUsage.cachedTokens ?? 0}`)
  }

  return text
}

// First call: computes and caches the prefix
const answer1 = await queryWithCaching('What is the authentication flow?')

// Second call: cache hit on the document prefix (50% cheaper)
const answer2 = await queryWithCaching('How do I configure rate limiting?')

// Third call: also hits cache
const answer3 = await queryWithCaching('What are the API endpoints?')
```

> **Beginner Note:** Groq's caching is fully automatic. The platform detects when your prompt starts with the same prefix as a recent request and reuses the cached computation. Cached tokens cost 50% less and do not count toward your rate limits. The cache expires after 2 hours without use.

### Anthropic Explicit Caching

Anthropic takes a different approach: you explicitly mark cache breakpoints using `cache_control` in `providerOptions`. This gives you precise control over what gets cached:

```typescript
import { generateText } from 'ai'
import { anthropic } from '@ai-sdk/anthropic'

async function queryWithExplicitCaching(question: string): Promise<string> {
  const { text, usage } = await generateText({
    model: anthropic('claude-sonnet-4-20250514'),
    messages: [
      {
        role: 'system',
        content: 'You are a documentation expert. Answer questions based on the reference material.',
      },
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `Reference documentation:\n${referenceDocument}`,
            providerOptions: {
              anthropic: { cacheControl: { type: 'ephemeral' } },
            },
          },
          {
            type: 'text',
            text: `\n\nQuestion: ${question}`,
          },
        ],
      },
    ],
  })

  if (usage) {
    const providerUsage = usage as any
    console.log(`Cache read tokens: ${providerUsage.cacheReadInputTokens ?? 0}`)
    console.log(`Cache creation tokens: ${providerUsage.cacheCreationInputTokens ?? 0}`)
  }

  return text
}
```

> **Intermediate Note:** The `cache_control: { type: 'ephemeral' }` marker tells Anthropic "cache everything up to this point." The cache lives about 5 minutes (refreshed each time you use it). You only pay the cache creation cost on the first call. Anthropic's cache offers up to 90% savings on cached tokens — a deeper discount than Groq's 50%, but requires explicit markers and has a shorter TTL.

### OpenAI Automatic Caching

OpenAI also provides automatic caching, similar to Groq. Prompts of 1024+ tokens are automatically cached with a 50% discount on cached input tokens. No code changes needed — just structure your prompts with the static prefix first.

### Cache Placement Strategy

The key to effective caching is identifying the **static prefix** — the part of your prompt that does not change between calls. With automatic caching (Groq, OpenAI), the cache matches from the beginning of the prompt up to the first difference. With explicit caching (Anthropic), you mark the boundary yourself.

```typescript
import { generateText } from 'ai'
import { groq } from '@ai-sdk/groq'

// Pattern: System prompt + static context (prefix) + dynamic query (suffix)
// Groq automatically caches the matching prefix across requests
async function cachedDocumentQA(
  staticContext: string,
  dynamicQuery: string
): Promise<{ answer: string; cacheHit: boolean }> {
  const { text, usage } = await generateText({
    model: groq('openai/gpt-oss-20b'),
    messages: [
      {
        role: 'system',
        content: `You are an expert analyst. Use the provided context to answer questions accurately.

<context>
${staticContext}
</context>`,
      },
      {
        role: 'user',
        content: dynamicQuery,
      },
    ],
  })

  const providerUsage = usage as any
  const cacheHit = (providerUsage.cachedTokens ?? 0) > 0

  return { answer: text, cacheHit }
}

// Multiple queries against the same context
const context = await Bun.file('./data/product-specs.txt').text()

const queries = [
  'What are the hardware requirements?',
  'What operating systems are supported?',
  'What is the maximum concurrent user count?',
  'Describe the backup procedures.',
]

for (const query of queries) {
  const { answer, cacheHit } = await cachedDocumentQA(context, query)
  console.log(`Query: ${query}`)
  console.log(`Cache hit: ${cacheHit}`)
  console.log(`Answer: ${answer.slice(0, 100)}...\n`)
}
```

### Multi-turn Caching

Caching is particularly effective for multi-turn conversations where the system prompt and early context remain constant. With automatic caching, the static system prompt prefix is cached across turns without any special configuration:

```typescript
import { generateText } from 'ai'
import { groq } from '@ai-sdk/groq'

class CachedConversation {
  private messages: Array<{ role: 'user' | 'assistant'; content: string }> = []
  private systemContent: string
  private model: string

  constructor(config: { systemPrompt: string; staticContext: string; model?: string }) {
    this.model = config.model ?? 'openai/gpt-oss-20b'

    // Combine system prompt and static context into a cacheable prefix
    // This stays constant across turns, so Groq caches it automatically
    this.systemContent = `${config.systemPrompt}\n\n<context>\n${config.staticContext}\n</context>`
  }

  async send(userMessage: string): Promise<{ text: string; cached: boolean }> {
    this.messages.push({ role: 'user', content: userMessage })

    const { text, usage } = await generateText({
      model: groq(this.model),
      messages: [
        { role: 'system', content: this.systemContent },
        ...this.messages.map(m => ({
          role: m.role as 'user' | 'assistant',
          content: m.content,
        })),
      ],
    })

    this.messages.push({ role: 'assistant', content: text })

    const providerUsage = usage as any
    const cached = (providerUsage.cachedTokens ?? 0) > 0

    return { text, cached }
  }
}
```

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
// KV cache size estimation (conceptual)
interface ModelConfig {
  layers: number
  heads: number
  headDim: number
  bytesPerParam: number // 2 for float16, 1 for int8
}

function estimateKVCacheSize(config: ModelConfig, sequenceLength: number): { bytes: number; megabytes: number } {
  // Per token: 2 (K and V) × layers × heads × headDim × bytesPerParam
  const bytesPerToken = 2 * config.layers * config.heads * config.headDim * config.bytesPerParam
  const totalBytes = bytesPerToken * sequenceLength

  return {
    bytes: totalBytes,
    megabytes: totalBytes / (1024 * 1024),
  }
}

// Example: Claude-like model (approximate)
const claudeConfig: ModelConfig = {
  layers: 80,
  heads: 64,
  headDim: 128,
  bytesPerParam: 2,
}

const cache200K = estimateKVCacheSize(claudeConfig, 200_000)
console.log(`KV cache for 200K tokens: ${cache200K.megabytes.toFixed(0)} MB`)
// This is per-request GPU memory — expensive at scale
```

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
/**
 * Strip unnecessary whitespace, boilerplate, and formatting
 * from documents before including them in context.
 */
function preprocessDocument(text: string): string {
  let processed = text

  // Collapse multiple blank lines into one
  processed = processed.replace(/\n{3,}/g, '\n\n')

  // Remove excessive whitespace
  processed = processed.replace(/[ \t]+/g, ' ')

  // Remove common boilerplate patterns
  processed = processed.replace(/^Copyright.*$/gm, '')
  processed = processed.replace(/^All rights reserved.*$/gm, '')
  processed = processed.replace(/^Page \d+ of \d+$/gm, '')

  // Remove HTML tags if present
  processed = processed.replace(/<[^>]+>/g, '')

  // Trim each line
  processed = processed
    .split('\n')
    .map(line => line.trim())
    .join('\n')

  return processed.trim()
}

// Measure compression
function compressionStats(
  original: string,
  compressed: string
): { originalTokens: number; compressedTokens: number; savings: string } {
  const origTokens = Math.ceil(original.length / 4)
  const compTokens = Math.ceil(compressed.length / 4)
  const savings = ((1 - compTokens / origTokens) * 100).toFixed(1)

  return {
    originalTokens: origTokens,
    compressedTokens: compTokens,
    savings: `${savings}%`,
  }
}
```

### Technique 2: Selective Inclusion

Only include relevant sections of a document:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface DocumentSection {
  title: string
  content: string
  tokens: number
}

function parseDocumentSections(document: string): DocumentSection[] {
  const sections: DocumentSection[] = []
  const parts = document.split(/^## /gm)

  for (const part of parts) {
    if (!part.trim()) continue

    const lines = part.split('\n')
    const title = lines[0].trim()
    const content = lines.slice(1).join('\n').trim()

    sections.push({
      title,
      content,
      tokens: Math.ceil(content.length / 4),
    })
  }

  return sections
}

async function selectRelevantSections(
  sections: DocumentSection[],
  question: string,
  maxTokens: number
): Promise<DocumentSection[]> {
  // Use the LLM to identify relevant section titles
  const sectionList = sections.map((s, i) => `${i}. ${s.title}`).join('\n')

  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content:
          'You are a document navigator. Given a list of section titles and a question, return the indices of the most relevant sections as comma-separated numbers. Return only the numbers.',
      },
      {
        role: 'user',
        content: `Sections:\n${sectionList}\n\nQuestion: ${question}`,
      },
    ],
    maxTokens: 100,
  })

  const indices = text
    .split(',')
    .map(s => parseInt(s.trim()))
    .filter(n => !isNaN(n) && n >= 0 && n < sections.length)

  // Select sections within token budget
  const selected: DocumentSection[] = []
  let totalTokens = 0

  for (const idx of indices) {
    if (totalTokens + sections[idx].tokens <= maxTokens) {
      selected.push(sections[idx])
      totalTokens += sections[idx].tokens
    }
  }

  return selected
}
```

### Technique 3: LLM-Based Compression

Use a smaller, cheaper model to compress content before sending to the main model:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function compressContext(text: string, targetRatio: number = 0.3): Promise<string> {
  const originalTokens = Math.ceil(text.length / 4)
  const targetTokens = Math.floor(originalTokens * targetRatio)

  const { text: compressed } = await generateText({
    model: mistral('mistral-small-latest'), // Use cheaper model for compression
    messages: [
      {
        role: 'system',
        content: `Compress the following text to approximately ${targetTokens} tokens.
Preserve all key facts, names, numbers, and relationships.
Remove redundancy, examples, and verbose explanations.
Output only the compressed text, no commentary.`,
      },
      {
        role: 'user',
        content: text,
      },
    ],
  })

  const compressedTokens = Math.ceil(compressed.length / 4)
  console.log(
    `Compressed: ${originalTokens} → ${compressedTokens} tokens (${((compressedTokens / originalTokens) * 100).toFixed(1)}%)`
  )

  return compressed
}

// Usage
const longDocument = `... (imagine a 10,000 word document) ...`
const compressed = await compressContext(longDocument, 0.25)
// Now use 'compressed' in your main prompt instead of 'longDocument'
```

> **Advanced Note:** LLM-based compression risks losing important details. Always validate that key facts survive compression, especially numerical data, proper nouns, and causal relationships. Consider using structured extraction (Module 3) to pull out key facts before compression.

---

## Section 6: Chunked Prefill Patterns

### What is Chunked Prefill?

When processing very long contexts, the model processes the input in chunks rather than all at once. Understanding this pattern helps you structure your inputs for optimal performance.

### Structuring Input for Efficient Processing

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * For very long documents, structure the input with clear markers
 * so the model can efficiently navigate the content.
 */
async function structuredLongDocument(
  sections: Array<{ title: string; content: string }>,
  question: string
): Promise<string> {
  // Build a table of contents first
  const toc = sections.map((s, i) => `[Section ${i + 1}] ${s.title}`).join('\n')

  // Format sections with clear delimiters
  const formattedSections = sections
    .map((s, i) => `<section id="${i + 1}" title="${s.title}">\n${s.content}\n</section>`)
    .join('\n\n')

  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content: `You are a document analyst. The user provides a structured document with a table of contents followed by sections. Use the table of contents to navigate efficiently. Always cite section numbers in your answers.`,
      },
      {
        role: 'user',
        content: `Table of Contents:\n${toc}\n\n${formattedSections}\n\nQuestion: ${question}`,
      },
    ],
  })

  return text
}
```

### Hierarchical Context Organization

For extremely long inputs, organize content hierarchically:

```typescript
interface HierarchicalDocument {
  summary: string // ~500 tokens
  sectionSummaries: Array<{ title: string; summary: string }> // ~50 tokens each
  sections: Array<{ title: string; content: string }> // Full content
}

async function buildHierarchicalDocument(document: string): Promise<HierarchicalDocument> {
  // Parse into sections (simplified)
  const rawSections = document.split(/^# /gm).filter(Boolean)
  const sections = rawSections.map(s => {
    const lines = s.split('\n')
    return { title: lines[0].trim(), content: lines.slice(1).join('\n').trim() }
  })

  // Generate section summaries
  const { text: summariesJson } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content: 'For each section, write a 1-2 sentence summary. Return as JSON array of {title, summary}.',
      },
      {
        role: 'user',
        content: sections.map(s => `## ${s.title}\n${s.content.slice(0, 500)}`).join('\n\n'),
      },
    ],
  })

  const sectionSummaries = JSON.parse(summariesJson)

  // Generate overall summary
  const { text: overallSummary } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content: 'Write a concise summary of this document in 2-3 paragraphs.',
      },
      {
        role: 'user',
        content: sectionSummaries.map((s: any) => `${s.title}: ${s.summary}`).join('\n'),
      },
    ],
  })

  return {
    summary: overallSummary,
    sectionSummaries,
    sections,
  }
}

// Use the hierarchical document: start with summary, drill into sections as needed
async function hierarchicalQA(doc: HierarchicalDocument, question: string): Promise<string> {
  // Step 1: Use summaries to identify relevant sections
  const { text: relevantIndices } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: `Document summary: ${doc.summary}\n\nSections:\n${doc.sectionSummaries
          .map((s, i) => `${i}. ${s.title}: ${s.summary}`)
          .join('\n')}\n\nWhich sections are relevant to: ${question}\nReturn comma-separated indices only.`,
      },
    ],
  })

  const indices = relevantIndices
    .split(',')
    .map(s => parseInt(s.trim()))
    .filter(n => !isNaN(n))

  // Step 2: Query with only the relevant sections
  const relevantContent = indices
    .map(i => doc.sections[i])
    .filter(Boolean)
    .map(s => `## ${s.title}\n${s.content}`)
    .join('\n\n')

  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: `Context:\n${relevantContent}\n\nQuestion: ${question}`,
      },
    ],
  })

  return text
}
```

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

// NOTE: Pricing below may be outdated. Verify against:
// Groq: https://groq.com/pricing
// Anthropic: https://www.anthropic.com/pricing
const groqGptOss20bPricing: PricingTier = {
  inputPer1M: 0.075, // $0.075 per 1M input tokens
  outputPer1M: 0.3, // $0.30 per 1M output tokens
  cachedInputPer1M: 0.0375, // $0.0375 per 1M tokens (50% discount)
}

// For comparison: Anthropic has separate cache write (125% of input) and cache read (10% of input)
const anthropicSonnetPricing = {
  inputPer1M: 3.0,
  outputPer1M: 15.0,
  cacheWritePer1M: 3.75, // 25% surcharge on first write
  cacheReadPer1M: 0.3, // 90% discount on subsequent reads
}

function calculateGroqCost(
  pricing: PricingTier,
  usage: {
    inputTokens: number
    outputTokens: number
    cachedTokens: number
  }
): { total: number; breakdown: Record<string, number> } {
  const uncachedInputTokens = usage.inputTokens - usage.cachedTokens
  const inputCost = (uncachedInputTokens / 1_000_000) * pricing.inputPer1M
  const cachedCost = (usage.cachedTokens / 1_000_000) * pricing.cachedInputPer1M
  const outputCost = (usage.outputTokens / 1_000_000) * pricing.outputPer1M

  return {
    total: inputCost + cachedCost + outputCost,
    breakdown: {
      input: inputCost,
      cachedInput: cachedCost,
      output: outputCost,
    },
  }
}

// Scenario: 50K token document, 10 questions, 200 token average response
function compareCachingScenarios(): void {
  const documentTokens = 50_000
  const questionTokens = 50
  const responseTokens = 200
  const numQueries = 10

  // Without caching: pay full input cost each time
  const uncachedCost = calculateGroqCost(groqGptOss20bPricing, {
    inputTokens: (documentTokens + questionTokens) * numQueries,
    outputTokens: responseTokens * numQueries,
    cachedTokens: 0,
  })

  // With caching: first query is full price, subsequent queries cache the document prefix
  const cachedCost = calculateGroqCost(groqGptOss20bPricing, {
    inputTokens: (documentTokens + questionTokens) * numQueries,
    outputTokens: responseTokens * numQueries,
    cachedTokens: documentTokens * (numQueries - 1), // 9 out of 10 queries hit cache
  })

  console.log('=== Groq GPT-OSS 20B Cost Comparison ===')
  console.log(`Without caching: $${uncachedCost.total.toFixed(4)}`)
  console.log(`With caching:    $${cachedCost.total.toFixed(4)}`)
  console.log(
    `Savings:         $${(uncachedCost.total - cachedCost.total).toFixed(4)} (${(((uncachedCost.total - cachedCost.total) / uncachedCost.total) * 100).toFixed(1)}%)`
  )

  console.log('\n--- Uncached Breakdown ---')
  for (const [key, value] of Object.entries(uncachedCost.breakdown)) {
    console.log(`  ${key}: $${value.toFixed(4)}`)
  }

  console.log('\n--- Cached Breakdown ---')
  for (const [key, value] of Object.entries(cachedCost.breakdown)) {
    console.log(`  ${key}: $${value.toFixed(4)}`)
  }
}

compareCachingScenarios()
```

### Break-Even Analysis

With automatic caching (Groq, OpenAI), there is no cache write surcharge — you simply pay 50% less for cached tokens. This means caching is beneficial from the very first cache hit (the second query):

```typescript
/**
 * Compare savings across providers for a given number of queries.
 * Groq/OpenAI: no write surcharge, 50% discount on cached reads.
 * Anthropic: 25% write surcharge, but 90% discount on cached reads.
 */
function compareCachingSavings(documentTokens: number, numQueries: number): void {
  // Groq: 50% discount, no write surcharge, caching starts on 2nd query
  const groqUncached = (documentTokens / 1_000_000) * 0.075 * numQueries
  const groqCached =
    (documentTokens / 1_000_000) * 0.075 + // First query: full price
    (documentTokens / 1_000_000) * 0.0375 * (numQueries - 1) // Subsequent: 50% off
  const groqSavings = ((groqUncached - groqCached) / groqUncached) * 100

  // Anthropic: 25% write surcharge, 90% read discount
  const anthropicUncached = (documentTokens / 1_000_000) * 3.0 * numQueries
  const anthropicCached =
    (documentTokens / 1_000_000) * 3.75 + // First query: cache write (125% of input)
    (documentTokens / 1_000_000) * 0.3 * (numQueries - 1) // Subsequent: cache read (10% of input)
  const anthropicSavings = ((anthropicUncached - anthropicCached) / anthropicUncached) * 100

  console.log(`=== ${numQueries} queries against ${documentTokens} token document ===`)
  console.log(`Groq savings:      ${groqSavings.toFixed(1)}%`)
  console.log(`Anthropic savings:  ${anthropicSavings.toFixed(1)}%`)
}

compareCachingSavings(50_000, 2) // Even 2 queries saves money
compareCachingSavings(50_000, 10) // 10 queries: significant savings on both
compareCachingSavings(50_000, 100) // High volume: Anthropic's 90% discount dominates
```

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
} {
  // Small documents: just use full context
  if (useCase.documentSize === 'small') {
    if (useCase.queryFrequency === 'high-volume') {
      return {
        strategy: 'cached-context',
        reasoning: 'Small document fits easily in context. Caching saves cost at high volume.',
      }
    }
    return {
      strategy: 'full-context',
      reasoning: 'Small document is cheap to include in full. No need for complexity.',
    }
  }

  // Huge documents that exceed context window
  if (useCase.documentSize === 'huge') {
    return {
      strategy: 'rag',
      reasoning: 'Document exceeds context window. RAG is the only viable approach.',
    }
  }

  // Medium/large documents with high query volume
  if (useCase.queryFrequency === 'high-volume' && useCase.documentUpdateFrequency === 'static') {
    if (useCase.accuracyRequirements === 'critical') {
      return {
        strategy: 'cached-context',
        reasoning: 'High accuracy needs full context. Caching amortizes cost across queries.',
      }
    }
    return {
      strategy: 'rag',
      reasoning: 'High volume + non-critical accuracy = RAG for cost efficiency.',
    }
  }

  // Frequently updating documents
  if (useCase.documentUpdateFrequency === 'real-time') {
    return {
      strategy: 'rag',
      reasoning: 'Real-time updates invalidate caches. RAG handles updates by re-embedding changed chunks.',
    }
  }

  // Default for medium/large, moderate use
  return {
    strategy: 'cached-context',
    reasoning: 'Moderate use case benefits from caching simplicity and accuracy.',
  }
}

// Examples
console.log(
  recommendStrategy({
    documentSize: 'medium',
    queryFrequency: 'high-volume',
    documentUpdateFrequency: 'static',
    accuracyRequirements: 'critical',
    latencyRequirements: 'moderate',
  })
)
// { strategy: 'cached-context', reasoning: 'High accuracy needs full context...' }

console.log(
  recommendStrategy({
    documentSize: 'huge',
    queryFrequency: 'periodic',
    documentUpdateFrequency: 'daily',
    accuracyRequirements: 'high',
    latencyRequirements: 'strict',
  })
)
// { strategy: 'rag', reasoning: 'Document exceeds context window...' }
```

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
import { generateText, embed } from 'ai'
import { groq } from '@ai-sdk/groq'

/**
 * Hybrid strategy:
 * 1. Cache a summary/overview of the full document (static prefix)
 * 2. Use RAG to retrieve specific details when needed
 * 3. Combine cached overview + retrieved details in context
 *
 * The system prompt with the document overview stays constant across queries,
 * so Groq automatically caches it.
 */
async function hybridDocumentQA(config: {
  documentSummary: string // Cached automatically (static prefix)
  retrievedChunks: string[] // From vector search (dynamic suffix)
  question: string
}): Promise<string> {
  const { text } = await generateText({
    model: groq('openai/gpt-oss-20b'),
    messages: [
      {
        role: 'system',
        content: `You are a document expert. You have access to a document overview and specific retrieved sections.
Use the overview for general context and the retrieved sections for specific details.

Document Overview:
${config.documentSummary}`,
      },
      {
        role: 'user',
        content: `Retrieved sections:\n${config.retrievedChunks.join('\n---\n')}\n\nQuestion: ${config.question}`,
      },
    ],
  })

  return text
}
```

> **Beginner Note:** Start with the simplest approach that works. If your document fits in the context window and you are making a few queries, use full context. Add caching when you notice cost issues. Add RAG when documents are too large. Do not over-engineer from the start.

> **Local Alternative (Ollama):** Prompt caching through the API is a cloud provider feature (Groq, Anthropic, OpenAI). Ollama does not offer explicit prompt caching through its API — but llama.cpp (which powers Ollama) automatically caches the KV state for repeated prefixes, giving you similar benefits transparently. The long context strategies (chunking, summarization, map-reduce) work with any provider. Note that most Ollama models have smaller context windows (8K-32K), so the context management techniques here are even more critical.

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
