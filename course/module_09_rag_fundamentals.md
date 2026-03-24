# Module 9: RAG Fundamentals

## Learning Objectives

- Understand what Retrieval-Augmented Generation (RAG) is and when to use it
- Implement the complete RAG pipeline: ingest, chunk, embed, store, retrieve, generate
- Apply multiple chunking strategies and understand their trade-offs
- Build retrieval systems with top-k similarity search and metadata filtering
- Inject retrieved context into LLM prompts with proper citation attribution
- Assess RAG pipeline quality using faithfulness, relevance, and correctness metrics

---

## Why Should I Care?

LLMs know a lot, but they do not know everything — and what they know has a cutoff date. They cannot answer questions about your company's internal documentation, yesterday's meeting notes, or the latest version of your API. They also hallucinate: when they do not know the answer, they make one up with alarming confidence.

RAG solves both problems. Instead of relying solely on the model's training data, you retrieve relevant information from your own data sources and feed it to the model alongside the question. The model generates its answer grounded in the provided context, dramatically reducing hallucination and enabling it to work with information it was never trained on.

RAG is the most widely deployed pattern in production LLM applications. Customer support systems retrieve relevant knowledge base articles. Coding assistants retrieve relevant source files. Legal tools retrieve relevant case law. Documentation chatbots retrieve relevant pages. Understanding RAG is not optional — it is a core competency for anyone building LLM applications.

This module teaches you to build a complete RAG pipeline from scratch, from ingesting documents to generating cited answers. You will implement multiple chunking strategies, build retrieval with similarity search, and assess the quality of your pipeline. By the end, you will have a reusable RAG system that you can adapt to any document collection.

---

## Connection to Other Modules

- **Module 5 (Long Context)** explored the alternative to RAG: putting everything in the context window. This module provides the retrieval approach.
- **Module 8 (Embeddings)** provided the building blocks: embedding functions, cosine similarity, and vector stores. RAG assembles them into a pipeline.
- **Module 3 (Structured Output)** taught `generateText` with `Output.object`. You will use structured output for citation extraction.
- **Module 2 (Prompt Engineering)** taught prompt design. The RAG prompt template is one of the most important prompts you will write.
- **Module 7 (Tool Use)** showed how LLMs can call functions. RAG retrieval can be exposed as a tool for agent-based systems.

---

## Section 1: What is RAG?

### The Problem

LLMs have three fundamental limitations that RAG addresses:

1. **Knowledge cutoff**: Training data has a date. The model does not know about events or updates after that date.
2. **No access to private data**: The model was not trained on your company's internal docs, database, or proprietary information.
3. **Hallucination**: When the model lacks knowledge, it generates plausible-sounding but incorrect answers.

### The RAG Solution

RAG adds a retrieval step before generation:

```
Traditional:  Question -> LLM -> Answer (may hallucinate)

RAG:          Question -> Retrieve relevant docs -> LLM + docs -> Grounded answer
```

The model receives both the question and the relevant context, and is instructed to answer based on the provided information.

### When to Use RAG

| Scenario                                | RAG?                   | Why                                       |
| --------------------------------------- | ---------------------- | ----------------------------------------- |
| Questions about your company docs       | Yes                    | Model was never trained on this           |
| General knowledge questions             | No                     | Model already knows this                  |
| Real-time data (stock prices, weather)  | No (use tools)         | RAG is for static or slowly-changing data |
| Large document collection, many queries | Yes                    | Amortize embedding cost across queries    |
| Single document, few questions          | No (use full context)  | Simpler and more accurate                 |
| Frequently updated data                 | Yes (with re-indexing) | RAG handles updates via re-embedding      |

> **Beginner Note:** RAG is not the answer to every question. For general knowledge, the model's training data is usually better than any RAG pipeline you could build. RAG shines specifically for domain-specific, private, or frequently-updated information.

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Without RAG: model may not know the answer or may hallucinate
const { text: withoutRag } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is the refund policy for Acme Corp?',
})
console.log('Without RAG:', withoutRag)
// Likely: "I don't have information about Acme Corp's refund policy."
// Or worse: A hallucinated policy that sounds real but is made up.

// With RAG: model answers based on retrieved context
const retrievedContext = `
Acme Corp Refund Policy (Updated January 2026):
- Full refund within 30 days of purchase
- 50% refund between 30-60 days
- No refunds after 60 days
- Digital products are non-refundable after download
- Contact support@acme.com to initiate a refund
`

const { text: withRag } = await generateText({
  model: mistral('mistral-small-latest'),
  messages: [
    {
      role: 'system',
      content: `Answer the user's question based on the provided context. If the context does not contain the answer, say so. Always cite the source.`,
    },
    {
      role: 'user',
      content: `Context:\n${retrievedContext}\n\nQuestion: What is the refund policy for Acme Corp?`,
    },
  ],
})
console.log('With RAG:', withRag)
// Accurate answer based on the actual policy document
```

---

## Section 2: The RAG Pipeline

### Overview

The RAG pipeline has two phases:

**Indexing phase (offline, done once):**

1. **Ingest**: Load documents from files, databases, APIs
2. **Chunk**: Split documents into manageable pieces
3. **Embed**: Convert chunks to vectors
4. **Store**: Save vectors and metadata in a vector database

**Query phase (online, per request):** 5. **Retrieve**: Find the most relevant chunks for a query 6. **Generate**: Send retrieved chunks + query to the LLM

```
 Indexing Phase:
  Documents -> Chunk -> Embed -> Store in Vector DB

 Query Phase:
  Query -> Embed -> Search Vector DB -> Top-K Chunks
                                          |
                                          v
                              Chunks + Query -> LLM -> Answer
```

### Complete Pipeline Implementation

Here is a minimal but complete RAG pipeline:

```typescript
import { embed, embedMany, generateText } from 'ai'
import { openai } from '@ai-sdk/openai'
import { mistral } from '@ai-sdk/mistral'

interface Chunk {
  id: string
  text: string
  source: string
  index: number
}

interface IndexedChunk extends Chunk {
  embedding: number[]
}

interface RAGResult {
  answer: string
  sources: Array<{ text: string; source: string; score: number }>
}

class SimpleRAG {
  private chunks: IndexedChunk[] = []
  private embeddingModel = openai.embedding('text-embedding-3-small')
  private chatModel = mistral('mistral-small-latest')

  /** Phase 1: Ingest and index documents */
  async ingest(documents: Array<{ name: string; content: string }>): Promise<number> {
    const allChunks: Chunk[] = []

    // Step 1: Chunk documents
    for (const doc of documents) {
      const chunks = this.chunkDocument(doc.content, doc.name)
      allChunks.push(...chunks)
    }

    console.log(`Created ${allChunks.length} chunks from ${documents.length} documents`)

    // Step 2: Embed all chunks
    const { embeddings } = await embedMany({
      model: this.embeddingModel,
      values: allChunks.map(c => c.text),
    })

    // Step 3: Store with embeddings
    this.chunks = allChunks.map((chunk, i) => ({
      ...chunk,
      embedding: embeddings[i] as number[],
    }))

    console.log(`Indexed ${this.chunks.length} chunks`)
    return this.chunks.length
  }

  /** Phase 2: Query */
  async query(question: string, topK: number = 3): Promise<RAGResult> {
    // Step 1: Embed the question
    const { embedding: queryEmbedding } = await embed({
      model: this.embeddingModel,
      value: question,
    })

    // Step 2: Find most similar chunks
    const scored = this.chunks.map(chunk => ({
      chunk,
      score: this.cosineSimilarity(queryEmbedding, chunk.embedding),
    }))

    scored.sort((a, b) => b.score - a.score)
    const topChunks = scored.slice(0, topK)

    // Step 3: Generate answer with context
    const context = topChunks
      .map((item, i) => `[Source ${i + 1}: ${item.chunk.source}]\n${item.chunk.text}`)
      .join('\n\n---\n\n')

    const { text } = await generateText({
      model: this.chatModel,
      messages: [
        {
          role: 'system',
          content: `You are a helpful assistant that answers questions based on the provided sources.
Rules:
- Only use information from the provided sources
- Cite sources using [Source N] notation
- If the sources don't contain the answer, say "I don't have enough information to answer this question."
- Be concise and accurate`,
        },
        {
          role: 'user',
          content: `Sources:\n${context}\n\nQuestion: ${question}`,
        },
      ],
    })

    return {
      answer: text,
      sources: topChunks.map(item => ({
        text: item.chunk.text.slice(0, 200),
        source: item.chunk.source,
        score: item.score,
      })),
    }
  }

  /** Split a document into chunks */
  private chunkDocument(content: string, source: string, chunkSize: number = 500, overlap: number = 50): Chunk[] {
    const words = content.split(/\s+/)
    const chunks: Chunk[] = []
    let index = 0

    for (let i = 0; i < words.length; i += chunkSize - overlap) {
      const chunkWords = words.slice(i, i + chunkSize)
      if (chunkWords.length < 20) continue // Skip tiny chunks

      chunks.push({
        id: `${source}-${index}`,
        text: chunkWords.join(' '),
        source,
        index: index++,
      })
    }

    return chunks
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0,
      magA = 0,
      magB = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      magA += a[i] * a[i]
      magB += b[i] * b[i]
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB))
  }
}

// Usage
const rag = new SimpleRAG()

await rag.ingest([
  {
    name: 'typescript-guide.md',
    content: `TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.
It adds optional static types, classes, interfaces, and modules. TypeScript helps catch
errors early in the development process through type checking. The TypeScript compiler,
tsc, converts TypeScript code into JavaScript that can run in any browser or runtime.
TypeScript supports generics, which allow you to write reusable components that work
with multiple types while maintaining type safety. Union types let a variable hold
values of multiple types, and intersection types combine multiple types into one.`,
  },
  {
    name: 'react-guide.md',
    content: `React is a JavaScript library for building user interfaces. It uses a
component-based architecture where UIs are composed of reusable components. React
introduces JSX, a syntax extension that lets you write HTML-like code in JavaScript.
The virtual DOM in React optimizes rendering by minimizing direct DOM manipulation.
React hooks like useState and useEffect enable state management and side effects in
functional components. The Context API provides a way to pass data through the
component tree without prop drilling.`,
  },
])

const result = await rag.query('How does TypeScript help with error prevention?')
console.log('Answer:', result.answer)
console.log('\nSources:')
for (const s of result.sources) {
  console.log(`  [${s.score.toFixed(3)}] ${s.source}: ${s.text.slice(0, 80)}...`)
}
```

---

## Section 3: Chunking Strategies

### Why Chunking Matters

Chunking is the most impactful decision in a RAG pipeline. Too large and you waste tokens on irrelevant content. Too small and you lose context. The right strategy depends on your document structure and query patterns.

### Strategy 1: Fixed-Size Chunking

Split by word count or character count:

```typescript
interface Chunk {
  text: string
  index: number
  startChar: number
  endChar: number
}

function fixedSizeChunk(
  text: string,
  chunkSize: number = 1000, // characters
  overlap: number = 200 // character overlap
): Chunk[] {
  const chunks: Chunk[] = []
  let start = 0
  let index = 0

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length)
    let chunkText = text.slice(start, end)

    // Try to break at a sentence boundary
    if (end < text.length) {
      const lastPeriod = chunkText.lastIndexOf('. ')
      if (lastPeriod > chunkSize * 0.5) {
        chunkText = chunkText.slice(0, lastPeriod + 1)
      }
    }

    chunks.push({
      text: chunkText.trim(),
      index: index++,
      startChar: start,
      endChar: start + chunkText.length,
    })

    start += chunkText.length - overlap
    if (start >= text.length) break
  }

  return chunks
}

// Usage
const text = 'Lorem ipsum dolor sit amet... '.repeat(100)
const chunks = fixedSizeChunk(text, 500, 100)
console.log(`Document: ${text.length} chars -> ${chunks.length} chunks`)
```

### Strategy 2: Sentence-Based Chunking

Split by sentences, then group into chunks:

```typescript
function sentenceChunk(text: string, sentencesPerChunk: number = 5, overlapSentences: number = 1): Chunk[] {
  // Split into sentences
  const sentences = text.match(/[^.!?]+[.!?]+/g) ?? [text]

  const chunks: Chunk[] = []
  let index = 0

  for (let i = 0; i < sentences.length; i += sentencesPerChunk - overlapSentences) {
    const chunkSentences = sentences.slice(i, i + sentencesPerChunk)
    if (chunkSentences.length === 0) continue

    chunks.push({
      text: chunkSentences.join(' ').trim(),
      index: index++,
      startChar: 0,
      endChar: 0,
    })
  }

  return chunks
}
```

### Strategy 3: Paragraph-Based Chunking

Respect natural document boundaries:

```typescript
function paragraphChunk(
  text: string,
  maxChunkSize: number = 1500 // max characters per chunk
): Chunk[] {
  const paragraphs = text.split(/\n\n+/).filter(p => p.trim())
  const chunks: Chunk[] = []
  let currentChunk = ''
  let index = 0

  for (const paragraph of paragraphs) {
    // If adding this paragraph would exceed the limit, save current chunk
    if (currentChunk && currentChunk.length + paragraph.length > maxChunkSize) {
      chunks.push({
        text: currentChunk.trim(),
        index: index++,
        startChar: 0,
        endChar: 0,
      })
      currentChunk = ''
    }

    currentChunk += (currentChunk ? '\n\n' : '') + paragraph
  }

  // Don't forget the last chunk
  if (currentChunk.trim()) {
    chunks.push({
      text: currentChunk.trim(),
      index: index++,
      startChar: 0,
      endChar: 0,
    })
  }

  return chunks
}
```

### Strategy 4: Recursive Chunking (Best for Markdown)

Split hierarchically: first by heading, then by paragraph, then by sentence:

```typescript
interface RecursiveChunk {
  text: string
  heading: string
  level: number
  index: number
}

function recursiveMarkdownChunk(markdown: string, maxChunkSize: number = 1000): RecursiveChunk[] {
  const chunks: RecursiveChunk[] = []
  let index = 0

  // Split by headings
  const sections = markdown.split(/^(#{1,6}\s.+)$/gm)

  let currentHeading = 'Introduction'
  let currentLevel = 0

  for (let i = 0; i < sections.length; i++) {
    const section = sections[i].trim()
    if (!section) continue

    // Check if this is a heading
    const headingMatch = section.match(/^(#{1,6})\s(.+)$/)
    if (headingMatch) {
      currentHeading = headingMatch[2]
      currentLevel = headingMatch[1].length
      continue
    }

    // Split content if it's too large
    if (section.length <= maxChunkSize) {
      chunks.push({
        text: section,
        heading: currentHeading,
        level: currentLevel,
        index: index++,
      })
    } else {
      // Split by paragraphs
      const paragraphs = section.split(/\n\n+/)
      let buffer = ''

      for (const para of paragraphs) {
        if (buffer && buffer.length + para.length > maxChunkSize) {
          chunks.push({
            text: buffer.trim(),
            heading: currentHeading,
            level: currentLevel,
            index: index++,
          })
          buffer = ''
        }
        buffer += (buffer ? '\n\n' : '') + para
      }

      if (buffer.trim()) {
        chunks.push({
          text: buffer.trim(),
          heading: currentHeading,
          level: currentLevel,
          index: index++,
        })
      }
    }
  }

  return chunks
}

// Usage
const markdown = `# Getting Started

This is the introduction to our application.

## Installation

Install the dependencies using npm or bun:

\`\`\`bash
bun add ai @ai-sdk/mistral
\`\`\`

Make sure you have Node.js 18+ or Bun installed.

## Configuration

Create a \`.env\` file with your API keys.

The configuration supports multiple providers.

## Usage

Import the functions you need and start making calls.
`

const chunks = recursiveMarkdownChunk(markdown, 200)
for (const chunk of chunks) {
  console.log(`[${chunk.heading}] (${chunk.text.length} chars) ${chunk.text.slice(0, 60)}...`)
}
```

### Chunking Strategy Comparison

| Strategy   | Pros                | Cons                      | Best For             |
| ---------- | ------------------- | ------------------------- | -------------------- |
| Fixed-size | Simple, predictable | May split mid-sentence    | Uniform text         |
| Sentence   | Preserves meaning   | Variable chunk sizes      | Prose, articles      |
| Paragraph  | Natural boundaries  | Some paragraphs are huge  | Well-structured docs |
| Recursive  | Preserves hierarchy | More complex to implement | Markdown, code docs  |

> **Beginner Note:** Start with recursive chunking for markdown documents and paragraph-based chunking for plain text. These strategies preserve the most meaning with the least effort. You can always switch strategies later if retrieval quality is not satisfactory.

---

## Section 4: Chunk Overlap

### Why Overlap?

Without overlap, information at chunk boundaries gets split:

```
Document: "...The refund period is 30 days. | Contact support@acme.com for details..."
                                        ^
                                  Chunk boundary

Chunk 1: "...The refund period is 30 days."
Chunk 2: "Contact support@acme.com for details..."

Query: "How do I get a refund?"
  Chunk 1 matches (mentions refund) but misses the contact info
  Chunk 2 has the contact info but might not match the query
```

With overlap, both chunks contain the boundary information:

```
Chunk 1: "...The refund period is 30 days. Contact support@acme.com for details..."
Chunk 2: "...is 30 days. Contact support@acme.com for details. Please include your..."
```

### Implementing Overlap

```typescript
function chunkWithOverlap(
  text: string,
  chunkSize: number,
  overlapPercent: number = 0.15 // 15% overlap
): string[] {
  const overlap = Math.floor(chunkSize * overlapPercent)
  const step = chunkSize - overlap
  const words = text.split(/\s+/)
  const chunks: string[] = []

  for (let i = 0; i < words.length; i += step) {
    const chunk = words.slice(i, i + chunkSize).join(' ')
    if (chunk.trim().length > 0) {
      chunks.push(chunk)
    }
  }

  return chunks
}

// Visualize overlap
const sampleText = Array.from({ length: 100 }, (_, i) => `word${i}`).join(' ')
const chunks = chunkWithOverlap(sampleText, 20, 0.2) // 20 words, 20% overlap

for (const [i, chunk] of chunks.entries()) {
  const words = chunk.split(' ')
  console.log(`Chunk ${i}: words ${words[0]} ... ${words[words.length - 1]} (${words.length} words)`)
}
```

### How Much Overlap?

| Overlap | Trade-off                                                  |
| ------- | ---------------------------------------------------------- |
| 0%      | No redundancy, but boundary information is lost            |
| 10-15%  | Good default — minimal redundancy, catches most boundaries |
| 20-30%  | Better retrieval at boundaries, but more storage and cost  |
| 50%+    | Excessive — too much redundancy, wastes embedding cost     |

> **Advanced Note:** Overlap increases both storage cost (more chunks to store) and embedding cost (more chunks to embed). For a document with N chunks at 0% overlap, 20% overlap creates approximately N/0.8 = 1.25N chunks — a 25% increase. This is usually a worthwhile trade-off for better retrieval quality.

---

## Section 5: Retrieval

### Top-K Similarity Search

The most common retrieval method: embed the query, find the K most similar chunks:

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

interface IndexedChunk {
  id: string
  text: string
  embedding: number[]
  metadata: Record<string, string>
}

interface RetrievalResult {
  chunk: IndexedChunk
  score: number
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0,
    magA = 0,
    magB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    magA += a[i] * a[i]
    magB += b[i] * b[i]
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB))
}

async function retrieve(
  query: string,
  index: IndexedChunk[],
  options: {
    topK?: number
    minScore?: number
    filter?: (chunk: IndexedChunk) => boolean
  } = {}
): Promise<RetrievalResult[]> {
  const { topK = 5, minScore = 0.3, filter } = options

  // Embed the query
  const { embedding: queryEmbedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: query,
  })

  // Score all chunks
  let candidates = index
  if (filter) {
    candidates = candidates.filter(filter)
  }

  const scored: RetrievalResult[] = candidates.map(chunk => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }))

  // Sort and filter
  return scored
    .filter(r => r.score >= minScore)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
}

// Usage with metadata filtering
// const results = await retrieve('How to deploy?', index, {
//   topK: 3,
//   minScore: 0.5,
//   filter: (chunk) => chunk.metadata.category === 'devops',
// });
```

### Retrieval with Reranking

Two-stage retrieval: first retrieve broadly, then rerank with a more expensive method:

```typescript
import { embed, generateText } from 'ai'
import { openai } from '@ai-sdk/openai'
import { mistral } from '@ai-sdk/mistral'

async function retrieveWithReranking(
  query: string,
  index: IndexedChunk[],
  options: { initialK?: number; finalK?: number } = {}
): Promise<RetrievalResult[]> {
  const { initialK = 10, finalK = 3 } = options

  // Stage 1: Broad retrieval
  const initial = await retrieve(query, index, { topK: initialK, minScore: 0.2 })

  if (initial.length <= finalK) {
    return initial // Not enough results to rerank
  }

  // Stage 2: LLM-based reranking
  const { text: rankings } = await generateText({
    model: mistral('mistral-small-latest'), // Use fast model for reranking
    messages: [
      {
        role: 'system',
        content: `You are a relevance ranker. Given a query and a list of text passages, rank them by relevance.
Return only the indices of the top ${finalK} most relevant passages, comma-separated.
Example output: 2,5,1`,
      },
      {
        role: 'user',
        content: `Query: ${query}\n\nPassages:\n${initial
          .map((r, i) => `[${i}] ${r.chunk.text.slice(0, 200)}`)
          .join('\n\n')}`,
      },
    ],
  })

  // Parse the rankings
  const rankedIndices = rankings
    .split(',')
    .map(s => parseInt(s.trim()))
    .filter(n => !isNaN(n) && n >= 0 && n < initial.length)

  return rankedIndices.map(i => initial[i])
}
```

### Hybrid Retrieval: Keyword + Semantic

Combine keyword matching with semantic search for better results:

```typescript
interface HybridResult {
  chunk: IndexedChunk
  semanticScore: number
  keywordScore: number
  combinedScore: number
}

function keywordScore(query: string, text: string): number {
  const queryWords = query.toLowerCase().split(/\s+/)
  const textLower = text.toLowerCase()

  let matches = 0
  for (const word of queryWords) {
    if (word.length > 2 && textLower.includes(word)) {
      matches++
    }
  }

  return matches / queryWords.length
}

async function hybridRetrieve(
  query: string,
  index: IndexedChunk[],
  topK: number = 5,
  semanticWeight: number = 0.7 // 70% semantic, 30% keyword
): Promise<HybridResult[]> {
  const { embedding: queryEmbedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: query,
  })

  const results: HybridResult[] = index.map(chunk => {
    const sem = cosineSimilarity(queryEmbedding, chunk.embedding)
    const kw = keywordScore(query, chunk.text)

    return {
      chunk,
      semanticScore: sem,
      keywordScore: kw,
      combinedScore: sem * semanticWeight + kw * (1 - semanticWeight),
    }
  })

  return results.sort((a, b) => b.combinedScore - a.combinedScore).slice(0, topK)
}
```

> **Advanced Note:** Hybrid retrieval often outperforms pure semantic search because it catches cases where exact keyword matches matter (proper nouns, technical terms, error codes) while still handling paraphrases and synonyms through semantic search. The BM25 algorithm is the industry-standard keyword retrieval method — consider using a library like `minisearch` for production keyword search.

---

## Section 6: Context Injection

### The RAG Prompt Template

How you inject retrieved context into the prompt significantly affects answer quality:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface RetrievedChunk {
  text: string
  source: string
  score: number
}

async function generateRAGAnswer(question: string, chunks: RetrievedChunk[]): Promise<string> {
  // Format the context with clear source attribution
  const context = chunks
    .map(
      (chunk, i) =>
        `<source id="${i + 1}" name="${chunk.source}" relevance="${chunk.score.toFixed(2)}">\n${chunk.text}\n</source>`
    )
    .join('\n\n')

  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'system',
        content: `You are a knowledgeable assistant that answers questions based on provided sources.

Guidelines:
1. Only use information from the provided sources
2. Cite sources inline using [Source N] notation
3. If multiple sources support a claim, cite all of them
4. If the sources contradict each other, note the discrepancy
5. If the sources do not contain enough information, clearly state what is missing
6. Be concise but thorough
7. Do not add information beyond what the sources provide`,
      },
      {
        role: 'user',
        content: `<context>\n${context}\n</context>\n\nQuestion: ${question}`,
      },
    ],
  })

  return text
}
```

### Context Window Management

When you have many retrieved chunks, you may need to fit them within the context window:

```typescript
function selectChunksWithinBudget(
  chunks: RetrievedChunk[],
  maxTokens: number,
  reserveForPrompt: number = 2000 // Reserve tokens for system prompt + question + response
): RetrievedChunk[] {
  const available = maxTokens - reserveForPrompt
  const selected: RetrievedChunk[] = []
  let usedTokens = 0

  // Chunks are assumed to be sorted by relevance (best first)
  for (const chunk of chunks) {
    const chunkTokens = Math.ceil(chunk.text.length / 4) // Rough estimate

    if (usedTokens + chunkTokens > available) {
      break
    }

    selected.push(chunk)
    usedTokens += chunkTokens
  }

  console.log(`Selected ${selected.length}/${chunks.length} chunks (${usedTokens} tokens)`)

  return selected
}
```

### Source Ordering Strategies

The order in which you present sources matters:

```typescript
// Strategy 1: Most relevant first (default)
function orderByRelevance(chunks: RetrievedChunk[]): RetrievedChunk[] {
  return [...chunks].sort((a, b) => b.score - a.score)
}

// Strategy 2: Document order (preserve original narrative flow)
function orderByDocument(chunks: RetrievedChunk[]): RetrievedChunk[] {
  return [...chunks].sort((a, b) => a.source.localeCompare(b.source))
}

// Strategy 3: Sandwich — most relevant at start and end
// Mitigates the "lost in the middle" effect for long context
function orderSandwich(chunks: RetrievedChunk[]): RetrievedChunk[] {
  const sorted = [...chunks].sort((a, b) => b.score - a.score)
  const result: RetrievedChunk[] = []

  for (let i = 0; i < sorted.length; i++) {
    if (i % 2 === 0) {
      result.push(sorted[i]) // Even indices at start
    } else {
      result.splice(Math.floor(result.length / 2), 0, sorted[i]) // Odd indices in middle
    }
  }

  return result
}
```

> **Beginner Note:** Start with "most relevant first" ordering. This works well in most cases. Only switch to attention-optimized ordering if you notice the model ignoring important middle context in long retrieval sets.

> **Advanced Note (Contextual Compression):** Retrieved chunks often contain irrelevant text alongside the useful parts. Before injecting chunks, you can use an LLM to extract only the query-relevant portions — this is the same summarization technique from Module 4 applied to retrieval results. Send each chunk with the query and ask the model to extract only the relevant parts. This reduces token usage and improves signal-to-noise, but adds an LLM call per chunk. Consider it when chunks are large and retrieval returns many results.

---

## Section 7: Citation and Attribution

### Structured Citation Extraction

Use `generateText` with `Output.object` to extract citations in a structured format:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const CitedAnswerSchema = z.object({
  answer: z.string().describe('The answer to the question'),
  citations: z.array(
    z.object({
      sourceIndex: z.number().describe('1-based index of the source'),
      quote: z.string().describe('Exact quote from the source that supports the claim'),
      claim: z.string().describe('The specific claim this citation supports'),
    })
  ),
  confidence: z.enum(['high', 'medium', 'low']).describe('Confidence based on source quality and coverage'),
  missingInfo: z.string().optional().describe('Information needed but not found in sources'),
})

type CitedAnswer = z.infer<typeof CitedAnswerSchema>

async function answerWithCitations(
  question: string,
  sources: Array<{ text: string; name: string }>
): Promise<CitedAnswer> {
  const context = sources.map((s, i) => `[Source ${i + 1}: ${s.name}]\n${s.text}`).join('\n\n---\n\n')

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: CitedAnswerSchema }),
    messages: [
      {
        role: 'system',
        content: `Answer the question using only the provided sources. For each claim, provide an exact quote from the source.`,
      },
      {
        role: 'user',
        content: `Sources:\n${context}\n\nQuestion: ${question}`,
      },
    ],
  })

  return output
}

// Usage
const result = await answerWithCitations('What are the system requirements?', [
  {
    name: 'docs/install.md',
    text: 'System requires Node.js 18+ or Bun 1.0+. Minimum 4GB RAM recommended. Works on macOS, Linux, and Windows.',
  },
  {
    name: 'docs/faq.md',
    text: 'We recommend at least 8GB RAM for development. SSD storage improves build times significantly.',
  },
])

console.log('Answer:', result.answer)
console.log('\nCitations:')
for (const citation of result.citations) {
  console.log(`  [Source ${citation.sourceIndex}] "${citation.quote}"`)
  console.log(`  Supports: ${citation.claim}\n`)
}
console.log('Confidence:', result.confidence)
if (result.missingInfo) {
  console.log('Missing:', result.missingInfo)
}
```

### Verifying Citations

```typescript
/**
 * Verify that citations actually reference text that exists in the sources.
 * LLMs sometimes fabricate quotes even when instructed to cite.
 */
function verifyCitations(
  answer: CitedAnswer,
  sources: Array<{ text: string; name: string }>
): {
  verified: boolean
  details: Array<{
    citation: CitedAnswer['citations'][0]
    found: boolean
    similarity?: number
  }>
} {
  const details = answer.citations.map(citation => {
    const sourceIndex = citation.sourceIndex - 1 // Convert to 0-based
    if (sourceIndex < 0 || sourceIndex >= sources.length) {
      return { citation, found: false }
    }

    const sourceText = sources[sourceIndex].text.toLowerCase()
    const quote = citation.quote.toLowerCase()

    // Exact match
    if (sourceText.includes(quote)) {
      return { citation, found: true, similarity: 1.0 }
    }

    // Fuzzy match: check if most words are present
    const quoteWords = quote.split(/\s+/).filter(w => w.length > 3)
    const matchedWords = quoteWords.filter(w => sourceText.includes(w))
    const similarity = quoteWords.length > 0 ? matchedWords.length / quoteWords.length : 0

    return {
      citation,
      found: similarity > 0.7, // Consider it found if 70%+ words match
      similarity,
    }
  })

  return {
    verified: details.every(d => d.found),
    details,
  }
}
```

---

## Section 8: Basic RAG Pipeline (End-to-End)

### Production-Ready RAG

Here is a complete, production-oriented RAG pipeline that brings together all the concepts:

```typescript
import { embed, embedMany, generateText } from 'ai'
import { openai } from '@ai-sdk/openai'
import { mistral } from '@ai-sdk/mistral'
import { readdir, readFile } from 'node:fs/promises'
import { join, extname } from 'node:path'

// --- Types ---
interface Document {
  id: string
  filePath: string
  content: string
}

interface DocChunk {
  id: string
  documentId: string
  text: string
  heading: string
  chunkIndex: number
  filePath: string
}

interface IndexedDocChunk extends DocChunk {
  embedding: number[]
}

interface QueryResult {
  answer: string
  sources: Array<{
    filePath: string
    heading: string
    text: string
    score: number
  }>
  tokenUsage: {
    embeddingTokens: number
    generationPromptTokens: number
    generationCompletionTokens: number
  }
}

// --- Chunking ---
function chunkMarkdown(content: string, docId: string, filePath: string, maxChunkSize: number = 800): DocChunk[] {
  const chunks: DocChunk[] = []
  let index = 0

  // Split by headings
  const lines = content.split('\n')
  let currentHeading = 'Document Start'
  let currentContent = ''

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.+)/)

    if (headingMatch) {
      // Save current chunk if non-empty
      if (currentContent.trim()) {
        const subChunks = splitLargeChunk(currentContent.trim(), maxChunkSize)
        for (const sub of subChunks) {
          chunks.push({
            id: `${docId}-${index}`,
            documentId: docId,
            text: sub,
            heading: currentHeading,
            chunkIndex: index++,
            filePath,
          })
        }
      }
      currentHeading = headingMatch[2]
      currentContent = ''
    } else {
      currentContent += line + '\n'
    }
  }

  // Last chunk
  if (currentContent.trim()) {
    const subChunks = splitLargeChunk(currentContent.trim(), maxChunkSize)
    for (const sub of subChunks) {
      chunks.push({
        id: `${docId}-${index}`,
        documentId: docId,
        text: sub,
        heading: currentHeading,
        chunkIndex: index++,
        filePath,
      })
    }
  }

  return chunks
}

function splitLargeChunk(text: string, maxSize: number): string[] {
  if (text.length <= maxSize) return [text]

  const paragraphs = text.split(/\n\n+/)
  const result: string[] = []
  let current = ''

  for (const para of paragraphs) {
    if (current && current.length + para.length > maxSize) {
      result.push(current.trim())
      current = ''
    }
    current += (current ? '\n\n' : '') + para
  }

  if (current.trim()) result.push(current.trim())
  return result
}

// --- RAG Pipeline ---
class RAGPipeline {
  private index: IndexedDocChunk[] = []
  private embeddingModel = openai.embedding('text-embedding-3-small')
  private chatModel = mistral('mistral-small-latest')

  /** Ingest all markdown files from a directory */
  async ingestDirectory(dirPath: string): Promise<{
    documentsProcessed: number
    chunksCreated: number
    tokensEmbedded: number
  }> {
    const files = await readdir(dirPath)
    const mdFiles = files.filter(f => extname(f) === '.md')

    const allChunks: DocChunk[] = []

    for (const file of mdFiles) {
      const filePath = join(dirPath, file)
      const content = await readFile(filePath, 'utf-8')
      const docId = file.replace('.md', '')

      const chunks = chunkMarkdown(content, docId, filePath)
      allChunks.push(...chunks)

      console.log(`  ${file}: ${chunks.length} chunks`)
    }

    // Embed all chunks in batches
    const batchSize = 100
    let totalTokens = 0

    for (let i = 0; i < allChunks.length; i += batchSize) {
      const batch = allChunks.slice(i, i + batchSize)
      const { embeddings, usage } = await embedMany({
        model: this.embeddingModel,
        values: batch.map(c => `${c.heading}\n${c.text}`),
      })

      for (let j = 0; j < batch.length; j++) {
        this.index.push({
          ...batch[j],
          embedding: embeddings[j] as number[],
        })
      }

      totalTokens += usage?.tokens ?? 0
    }

    return {
      documentsProcessed: mdFiles.length,
      chunksCreated: allChunks.length,
      tokensEmbedded: totalTokens,
    }
  }

  /** Query the pipeline */
  async query(question: string, options: { topK?: number; minScore?: number } = {}): Promise<QueryResult> {
    const { topK = 5, minScore = 0.3 } = options

    // Embed the question
    const { embedding: queryEmbedding, usage: embedUsage } = await embed({
      model: this.embeddingModel,
      value: question,
    })

    // Retrieve
    const scored = this.index.map(chunk => ({
      chunk,
      score: this.cosineSimilarity(queryEmbedding, chunk.embedding),
    }))

    const topChunks = scored
      .filter(r => r.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)

    // Generate answer
    const context = topChunks
      .map(
        (item, i) =>
          `<source id="${i + 1}" file="${item.chunk.filePath}" section="${item.chunk.heading}">\n${item.chunk.text}\n</source>`
      )
      .join('\n\n')

    const { text, usage: genUsage } = await generateText({
      model: this.chatModel,
      messages: [
        {
          role: 'system',
          content: `You are a documentation assistant. Answer questions based on the provided sources.

Rules:
- Only use information from the provided sources
- Cite sources using [Source N] notation, where N matches the source id
- If sources don't contain the answer, say so explicitly
- Be accurate and concise`,
        },
        {
          role: 'user',
          content: `${context}\n\nQuestion: ${question}`,
        },
      ],
    })

    return {
      answer: text,
      sources: topChunks.map(item => ({
        filePath: item.chunk.filePath,
        heading: item.chunk.heading,
        text: item.chunk.text.slice(0, 200),
        score: item.score,
      })),
      tokenUsage: {
        embeddingTokens: embedUsage?.tokens ?? 0,
        generationInputTokens: genUsage?.inputTokens ?? 0,
        generationOutputTokens: genUsage?.outputTokens ?? 0,
      },
    }
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0,
      magA = 0,
      magB = 0
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i]
      magA += a[i] * a[i]
      magB += b[i] * b[i]
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB))
  }

  /** Get index statistics */
  getStats(): { totalChunks: number; uniqueDocuments: number; avgChunkLength: number } {
    const uniqueDocs = new Set(this.index.map(c => c.documentId))
    const avgLength =
      this.index.length > 0 ? this.index.reduce((sum, c) => sum + c.text.length, 0) / this.index.length : 0

    return {
      totalChunks: this.index.length,
      uniqueDocuments: uniqueDocs.size,
      avgChunkLength: Math.round(avgLength),
    }
  }
}

// --- Usage ---
const pipeline = new RAGPipeline()

// Ingest documents
console.log('Ingesting documents...')
const ingestResult = await pipeline.ingestDirectory('./docs')
console.log(`Processed: ${ingestResult.documentsProcessed} documents`)
console.log(`Created: ${ingestResult.chunksCreated} chunks`)
console.log(`Embedded: ${ingestResult.tokensEmbedded} tokens`)

// Query
const queryResult = await pipeline.query('How do I set up the project?')
console.log('\nAnswer:', queryResult.answer)
console.log('\nSources:')
for (const s of queryResult.sources) {
  console.log(`  [${s.score.toFixed(3)}] ${s.filePath} > ${s.heading}`)
}
console.log('\nToken usage:', queryResult.tokenUsage)
```

---

## Section 9: RAG Assessment

### Why Assess?

A RAG pipeline can fail in multiple ways:

1. **Retrieval failure**: The right chunks are not retrieved
2. **Context failure**: Retrieved chunks do not contain the answer
3. **Generation failure**: The model ignores or misinterprets the context
4. **Hallucination**: The model adds information not in the context
5. **Citation failure**: Citations are incorrect or missing

### Key Metrics

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface RAGAssessmentResult {
  faithfulness: number // Is the answer supported by the sources? (0-1)
  relevance: number // Are the retrieved sources relevant to the question? (0-1)
  correctness: number // Is the answer factually correct? (0-1)
  completeness: number // Does the answer cover all aspects of the question? (0-1)
}

// Automated assessment using LLM-as-judge
const AssessmentSchema = z.object({
  faithfulness: z.object({
    score: z.number().min(0).max(1),
    reasoning: z.string(),
    unsupportedClaims: z.array(z.string()),
  }),
  relevance: z.object({
    score: z.number().min(0).max(1),
    reasoning: z.string(),
    irrelevantSources: z.array(z.number()),
  }),
  completeness: z.object({
    score: z.number().min(0).max(1),
    reasoning: z.string(),
    missingAspects: z.array(z.string()),
  }),
})

async function assessRAGResponse(
  question: string,
  answer: string,
  sources: string[],
  groundTruth?: string
): Promise<z.infer<typeof AssessmentSchema>> {
  const sourcesFormatted = sources.map((s, i) => `[Source ${i + 1}]: ${s}`).join('\n\n')

  const groundTruthSection = groundTruth ? `\nGround truth answer: ${groundTruth}` : ''

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: AssessmentSchema }),
    messages: [
      {
        role: 'system',
        content: `You are a RAG quality assessor. Assess the quality of a RAG-generated answer.

Assess these dimensions:
1. Faithfulness: Is every claim in the answer supported by the sources? Score 0-1.
2. Relevance: Are the retrieved sources relevant to the question? Score 0-1.
3. Completeness: Does the answer fully address the question? Score 0-1.

Be strict and objective.`,
      },
      {
        role: 'user',
        content: `Question: ${question}

Sources:
${sourcesFormatted}

Generated answer: ${answer}
${groundTruthSection}`,
      },
    ],
  })

  return output
}

// Usage
const assessment = await assessRAGResponse(
  'What are the system requirements?',
  'The system requires Node.js 18+ and at least 4GB of RAM [Source 1]. An SSD is recommended for better performance [Source 2].',
  [
    'System requires Node.js 18+ or Bun 1.0+. Minimum 4GB RAM recommended.',
    'SSD storage improves build times significantly.',
  ]
)

console.log('Assessment:')
console.log(`  Faithfulness: ${assessment.faithfulness.score}`)
console.log(`  Relevance: ${assessment.relevance.score}`)
console.log(`  Completeness: ${assessment.completeness.score}`)

if (assessment.faithfulness.unsupportedClaims.length > 0) {
  console.log('  Unsupported claims:', assessment.faithfulness.unsupportedClaims)
}
```

### Building a Test Dataset

```typescript
interface TestCase {
  question: string
  expectedAnswer: string
  expectedSources: string[] // File paths that should be retrieved
  category: string
}

const testDataset: TestCase[] = [
  {
    question: 'How do I install the project?',
    expectedAnswer: 'Run bun install in the project directory.',
    expectedSources: ['docs/getting-started.md'],
    category: 'setup',
  },
  {
    question: 'What database does the project use?',
    expectedAnswer: 'The project uses PostgreSQL as the primary database.',
    expectedSources: ['docs/architecture.md', 'docs/database.md'],
    category: 'architecture',
  },
  // Add more test cases...
]

async function runTestSuite(
  pipeline: RAGPipeline,
  dataset: TestCase[]
): Promise<{
  averageScores: { faithfulness: number; relevance: number; completeness: number }
  results: Array<{ question: string; scores: z.infer<typeof AssessmentSchema> }>
}> {
  const results = []
  let totalFaithfulness = 0
  let totalRelevance = 0
  let totalCompleteness = 0

  for (const testCase of dataset) {
    console.log(`Testing: "${testCase.question}"`)

    const response = await pipeline.query(testCase.question)
    const scores = await assessRAGResponse(
      testCase.question,
      response.answer,
      response.sources.map(s => s.text),
      testCase.expectedAnswer
    )

    results.push({ question: testCase.question, scores })

    totalFaithfulness += scores.faithfulness.score
    totalRelevance += scores.relevance.score
    totalCompleteness += scores.completeness.score
  }

  const n = dataset.length
  return {
    averageScores: {
      faithfulness: totalFaithfulness / n,
      relevance: totalRelevance / n,
      completeness: totalCompleteness / n,
    },
    results,
  }
}
```

### Retrieval Quality Metrics

```typescript
/**
 * Measure retrieval quality independent of generation quality.
 */
function measureRetrievalQuality(
  retrievedSources: string[], // File paths retrieved
  expectedSources: string[] // File paths that should have been retrieved
): {
  precision: number // What fraction of retrieved sources were relevant?
  recall: number // What fraction of relevant sources were retrieved?
  f1: number // Harmonic mean of precision and recall
} {
  const retrievedSet = new Set(retrievedSources)
  const expectedSet = new Set(expectedSources)

  let truePositives = 0
  for (const source of retrievedSet) {
    if (expectedSet.has(source)) truePositives++
  }

  const precision = retrievedSet.size > 0 ? truePositives / retrievedSet.size : 0
  const recall = expectedSet.size > 0 ? truePositives / expectedSet.size : 0
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0

  return { precision, recall, f1 }
}
```

> **Beginner Note:** Start by manually reviewing 20-30 queries before investing in automated assessment. Manual review builds intuition about failure modes and helps you design better automated metrics.

> **Advanced Note:** LLM-as-judge assessment has its own biases and failure modes. The judge model might miss subtle hallucinations or be overly generous with scores. Consider using multiple judge models or combining automated metrics with periodic human review. The RAGAS framework (ragas.io) provides a more comprehensive assessment toolkit.

> **Provider Tip: Native Citations** — This module teaches citation attribution using structured output and manual source tracking. Several providers offer native citation features that can simplify this: **Mistral** supports citations via tool calls — you pass documents as tool responses and the model returns `ReferenceChunk` objects linking claims to specific sources. **Anthropic** offers a native Citations feature that returns precise character-range citations grounded to exact spans in provided documents. Both approaches can replace the manual citation extraction pattern taught here.

> **Local Alternative (Ollama):** RAG pipelines work with any model. Use `ollama('qwen3.5')` for generation and `ollama.embedding('qwen3-embedding:0.6b')` for embeddings (see Module 8). The retrieval and chunking logic is entirely model-agnostic. Local RAG is especially appealing for privacy-sensitive documents that you don't want to send to external APIs.

---

## Summary

In this module, you learned:

1. **What RAG is:** Retrieval-Augmented Generation combines document retrieval with LLM generation, grounding model responses in your data to reduce hallucination and enable access to private or recent information.
2. **The RAG pipeline:** The complete flow from ingestion (load, chunk, embed, store) through retrieval (query, search, rank) to generation (inject context, generate, cite).
3. **Chunking strategies:** Fixed-size, sentence-based, paragraph-based, and recursive chunking each offer different trade-offs between context preservation and retrieval precision.
4. **Chunk overlap:** Overlapping chunks by 10-20% prevents information loss at chunk boundaries and improves retrieval of concepts that span splits.
5. **Retrieval techniques:** Top-K similarity search, reranking retrieved results, and hybrid keyword-plus-semantic retrieval improve the quality of retrieved context.
6. **Context injection:** How to structure RAG prompts with retrieved chunks, source attribution, and instructions that guide the model to answer based on provided context.
7. **RAG assessment:** Using faithfulness, relevance, and correctness metrics — including LLM-as-judge evaluation — to systematically measure pipeline quality.

In Module 10, you will tackle advanced RAG techniques including query transformation, HyDE, hybrid search, and self-RAG to handle the failure modes of naive retrieval.

---

## Quiz

### Question 1 (Easy)

What does RAG stand for, and what problem does it solve?

A) Retrieval-Augmented Generation; it makes models generate faster
B) Retrieval-Augmented Generation; it grounds model responses in retrieved data to reduce hallucination
C) Random Access Generation; it generates random variations of answers
D) Rapid Answer Generator; it speeds up query processing

**Answer: B**

RAG stands for Retrieval-Augmented Generation. It solves the problem of LLMs lacking access to specific, private, or up-to-date information by retrieving relevant documents from a knowledge base and including them in the prompt. This grounds the model's response in actual data, dramatically reducing hallucination.

---

### Question 2 (Medium)

Why is chunk overlap important in RAG?

A) It makes the embeddings more accurate
B) It prevents information at chunk boundaries from being lost
C) It reduces the total number of chunks
D) It speeds up the embedding process

**Answer: B**

Without overlap, information that spans a chunk boundary gets split between two chunks. Neither chunk contains the complete information, and a query about that specific detail may not retrieve either chunk. Overlap ensures that boundary information appears in both adjacent chunks, improving retrieval reliability.

---

### Question 3 (Medium)

In a hybrid retrieval approach, what are the two search methods combined?

A) Embedding search and database search
B) Semantic (embedding) search and keyword (lexical) search
C) Full-text search and regex search
D) Vector search and graph search

**Answer: B**

Hybrid retrieval combines semantic search (using embedding similarity) with keyword/lexical search (using methods like BM25 or simple keyword matching). This captures both semantic similarity (paraphrases, synonyms) and exact keyword matches (proper nouns, technical terms, error codes).

---

### Question 4 (Hard)

What is "faithfulness" in RAG assessment?

A) Whether the user trusts the system
B) Whether the generated answer is supported by the retrieved sources (no hallucination)
C) Whether the system retrieves the correct documents
D) Whether the system cites sources correctly

**Answer: B**

Faithfulness measures whether every claim in the generated answer is supported by the retrieved context. A faithful answer only states information that appears in the sources. An unfaithful answer contains claims that the model added from its training data or fabricated entirely, which is the core problem RAG is designed to solve.

---

### Question 5 (Easy)

When should you prefer RAG over putting the full document in the context window?

A) Always — RAG is always better
B) When the document is small and you have many questions
C) When the document collection is large, queries are frequent, or data changes often
D) When you need the highest possible accuracy

**Answer: C**

RAG is preferred when: (1) the document collection exceeds the context window, (2) you have many queries (amortize embedding cost), or (3) data changes frequently (re-embed changed chunks vs. re-send everything). For small, static documents with few queries, full context is usually simpler and more accurate.

---

## Exercises

### Exercise 1: Complete RAG Pipeline over Markdown Files with Citations

Build a RAG pipeline that indexes a directory of markdown files and answers questions with citations.

**Requirements:**

1. Recursively ingest all `.md` files from a directory
2. Implement recursive markdown chunking (split by heading, then paragraph)
3. Use 15% chunk overlap
4. Embed with `embedMany` using OpenAI `text-embedding-3-small`
5. Store the index in memory (optionally persist to a JSON file)
6. Implement retrieval with top-K similarity search (default K=5)
7. Generate answers with source citations using `[Source N]` notation
8. After each answer, verify that citations reference actual source text
9. Track and report token usage and estimated cost for each query

**Starter code:**

```typescript
import { embed, embedMany, generateText } from 'ai'
import { openai } from '@ai-sdk/openai'
import { mistral } from '@ai-sdk/mistral'
import { readdir, readFile } from 'node:fs/promises'
import { join, extname } from 'node:path'

interface Chunk {
  id: string
  text: string
  filePath: string
  heading: string
  index: number
}

interface IndexedChunk extends Chunk {
  embedding: number[]
}

interface RAGAnswer {
  answer: string
  citations: Array<{
    sourceNumber: number
    filePath: string
    heading: string
    quote: string
    verified: boolean
  }>
  retrievalScores: number[]
  cost: {
    embeddingTokens: number
    inputTokens: number
    outputTokens: number
    estimatedCostUSD: number
  }
}

class MarkdownRAG {
  private index: IndexedChunk[] = []

  async ingest(dirPath: string): Promise<void> {
    // TODO: Read all .md files
    // TODO: Chunk with recursive markdown chunking + overlap
    // TODO: Embed all chunks
    // TODO: Store in this.index
  }

  async query(question: string): Promise<RAGAnswer> {
    // TODO: Embed question
    // TODO: Retrieve top-K chunks
    // TODO: Generate answer with citations
    // TODO: Verify citations
    // TODO: Calculate cost
    throw new Error('Not implemented')
  }
}

// Usage
const rag = new MarkdownRAG()
await rag.ingest('./docs')

const answer = await rag.query('How do I get started with the project?')
console.log(answer.answer)
console.log('\nCitations:')
for (const c of answer.citations) {
  const status = c.verified ? 'VERIFIED' : 'UNVERIFIED'
  console.log(`  [${status}] Source ${c.sourceNumber}: ${c.filePath} > ${c.heading}`)
  console.log(`  Quote: "${c.quote.slice(0, 80)}..."`)
}
console.log('\nCost:', answer.cost)
```

### Exercise 2: RAG Pipeline Quality Assessment

Build an assessment harness that measures your RAG pipeline's quality.

**Requirements:**

1. Create a test dataset with 10+ question-answer pairs
2. For each pair, specify which source files should be retrieved
3. Run each question through the pipeline
4. Measure: retrieval precision/recall, faithfulness, and completeness
5. Generate a summary report with per-question and aggregate scores
6. Identify the weakest queries and suggest improvements

```typescript
interface AssessmentResult {
  question: string
  retrievalPrecision: number
  retrievalRecall: number
  faithfulness: number
  completeness: number
  passed: boolean // All scores above threshold
}

async function assessPipeline(
  pipeline: MarkdownRAG,
  dataset: TestCase[],
  threshold: number = 0.7
): Promise<{
  summary: { avgPrecision: number; avgRecall: number; avgFaithfulness: number; passRate: number }
  results: AssessmentResult[]
  recommendations: string[]
}> {
  // TODO: Implement assessment
  throw new Error('Not implemented')
}
```

**Assessment criteria:**

- Pipeline handles edge cases (empty documents, very long documents, no relevant documents)
- Citations are verifiable (quotes exist in the source text)
- Quality metrics are reasonable and well-calibrated
- The report clearly identifies areas for improvement
