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

Here is an important insight before we dive in: the simplest form of RAG is just reading the right files and injecting them into the prompt. Production coding agents do this constantly — they read project configuration files from disk and inject those instructions into every conversation. That is file-based RAG: retrieve a file, augment the prompt, generate. Not everything needs vector search. This module starts with that intuition and builds toward the full pipeline.

This module teaches you to build a complete RAG pipeline from scratch, from ingesting documents to generating cited answers. You will implement multiple chunking strategies, build retrieval with similarity search, and assess the quality of your pipeline. By the end, you will have a reusable RAG system that you can adapt to any document collection.

---

## Connection to Other Modules

- **Module 5 (Long Context)** explored the alternative to RAG: putting everything in the context window. This module provides the retrieval approach.
- **Module 8 (Embeddings)** provided the building blocks: embedding functions, cosine similarity, and vector stores. RAG assembles them into a pipeline.
- **Module 3 (Structured Output)** taught `generateText` with `Output.object`. You will use structured output for citation extraction.
- **Module 2 (Prompt Engineering)** taught prompt design. The RAG prompt template is one of the most important prompts you will write.
- **Module 7 (Tool Use)** showed how LLMs can call functions. RAG retrieval can be exposed as a tool for agent-based systems.

> **Building on Module 8:** You already built semantic search that finds relevant documents by embedding similarity. RAG adds the generation step — injecting retrieved documents into an LLM prompt to produce grounded answers.

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

### What to Build

Build a script at `src/rag/demo-rag-motivation.ts` that demonstrates the difference between asking an LLM a question about private data with and without retrieved context.

The script should:

1. Call `generateText` with a question about a fictional company's refund policy (no context provided) and log the result
2. Define a `retrievedContext` string containing a fake refund policy document
3. Call `generateText` again, this time injecting the context into a user message alongside the question, with a system message instructing the model to answer based on the provided context and cite the source
4. Log both results and compare

Think about: what system prompt instructions prevent the model from going beyond the provided sources? How does the `messages` array differ from a simple `prompt` call when you need to separate instructions from context?

The key API pattern for injecting context:

```typescript
const { text } = await generateText({
  model: mistral('mistral-small-latest'),
  messages: [
    { role: 'system', content: `Answer based on provided context...` },
    { role: 'user', content: `Context:\n${context}\n\nQuestion: ${question}` },
  ],
})
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

### What to Build

Build a `SimpleRAG` class at `src/rag/simple-rag.ts` that implements both phases of the pipeline.

Here are the types you will need:

```typescript
import { embed, embedMany, generateText, cosineSimilarity } from 'ai'
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
```

Your class needs two public methods:

```typescript
class SimpleRAG {
  private chunks: IndexedChunk[] = []
  private embeddingModel = openai.embedding('text-embedding-3-small')
  private chatModel = mistral('mistral-small-latest')

  async ingest(documents: Array<{ name: string; content: string }>): Promise<number>
  async query(question: string, topK?: number): Promise<RAGResult>
  private chunkDocument(content: string, source: string, chunkSize?: number, overlap?: number): Chunk[]
}
```

For `ingest`: iterate over documents, chunk each one using `chunkDocument`, embed all chunk texts with `embedMany`, and store the results in `this.chunks`. Return the total number of chunks indexed.

For `query`: embed the question with `embed`, compute `cosineSimilarity` against every stored chunk, take the top-K, format them as numbered sources in a prompt, and call `generateText` with a system message that instructs the model to cite sources using `[Source N]` notation. Return the answer and source metadata.

For `chunkDocument`: split content by whitespace into words, create chunks of `chunkSize` words with `overlap` word overlap. Skip tiny chunks (fewer than 20 words). Assign each chunk an ID of `${source}-${index}`.

Questions to consider: Why do you embed the question with the same model as the chunks? What happens if `chunkSize` is too small? Too large? What should the system prompt say about information not found in the sources?

---

## Section 3: Chunking Strategies

### Why Chunking Matters

Chunking is the most impactful decision in a RAG pipeline. Too large and you waste tokens on irrelevant content. Too small and you lose context. The right strategy depends on your document structure and query patterns.

### What to Build

Create `src/rag/chunking.ts` with four chunking functions. Each returns an array of chunks with `text`, `index`, and position metadata.

**Strategy 1: Fixed-Size Chunking** -- `fixedSizeChunk(text, chunkSize?, overlap?): Chunk[]`

```typescript
interface Chunk {
  text: string
  index: number
  startChar: number
  endChar: number
}
```

Slice the text by character count (`chunkSize` defaults to 1000, `overlap` defaults to 200). Before finalizing each chunk, look for the last sentence boundary (`. `) in the second half of the chunk and break there if possible. Track `startChar`/`endChar` positions. Advance by `chunkText.length - overlap` each iteration.

**Strategy 2: Sentence-Based Chunking** -- `sentenceChunk(text, sentencesPerChunk?, overlapSentences?): Chunk[]`

Split text into sentences using a regex like `/[^.!?]+[.!?]+/g`. Group `sentencesPerChunk` sentences (default 5) into each chunk, with `overlapSentences` (default 1) sentence overlap between consecutive chunks.

**Strategy 3: Paragraph-Based Chunking** -- `paragraphChunk(text, maxChunkSize?): Chunk[]`

Split on double newlines. Accumulate paragraphs into a buffer until adding the next paragraph would exceed `maxChunkSize` (default 1500 chars), then flush the buffer as a chunk. Do not forget to emit the last chunk.

**Strategy 4: Recursive Markdown Chunking** -- `recursiveMarkdownChunk(markdown, maxChunkSize?): RecursiveChunk[]`

```typescript
interface RecursiveChunk {
  text: string
  heading: string
  level: number
  index: number
}
```

Split by heading regex (`/^(#{1,6}\s.+)$/gm`). Track the current heading and its level. For each non-heading section, if it fits within `maxChunkSize` (default 1000), emit it as one chunk. If it exceeds the limit, split further by paragraph and buffer until the limit.

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

### What to Build

Add a `chunkWithOverlap(text, chunkSize, overlapPercent?): string[]` function to your `src/rag/chunking.ts` file.

The function should:

1. Calculate the overlap in words from `overlapPercent` (default 0.15, meaning 15%)
2. Compute the step size as `chunkSize - overlap`
3. Split text into words, then iterate with step-sized jumps, slicing `chunkSize` words each time
4. Return the array of chunk strings

After implementing, write a quick visualization that creates chunks from a 100-word sequence (`word0 word1 ... word99`) with `chunkSize = 20` and `overlapPercent = 0.2`. Log the first and last word of each chunk to verify the overlap is working correctly.

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

The most common retrieval method: embed the query, find the K most similar chunks.

### What to Build

Create `src/rag/retrieval.ts` with three retrieval functions. You will need these types:

```typescript
import { embed, cosineSimilarity } from 'ai'
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
```

**Function 1: `retrieve(query, index, options?): Promise<RetrievalResult[]>`**

Options should support `topK` (default 5), `minScore` (default 0.3), and an optional `filter` function `(chunk: IndexedChunk) => boolean`. The function should embed the query with `openai.embedding('text-embedding-3-small')`, compute cosine similarity against all chunks (applying the filter first if provided), filter out results below `minScore`, sort descending by score, and return the top K.

**Function 2: `retrieveWithReranking(query, index, options?): Promise<RetrievalResult[]>`**

This implements two-stage retrieval. First, call `retrieve` with a larger `initialK` (default 10) and a lower `minScore` (0.2) to cast a wide net. Then use `generateText` with a system prompt that instructs the model to rank passages by relevance and return only the indices of the top `finalK` (default 3) most relevant passages as a comma-separated list. Parse the LLM's response to extract the indices and return the corresponding results.

Think about: what happens if the LLM returns invalid indices? How do you handle that gracefully?

**Function 3: `hybridRetrieve(query, index, topK?, semanticWeight?): Promise<HybridResult[]>`**

```typescript
interface HybridResult {
  chunk: IndexedChunk
  semanticScore: number
  keywordScore: number
  combinedScore: number
}
```

Combine semantic search with a simple keyword scorer. The keyword scorer splits the query into words, checks which query words (length > 2) appear in the chunk text (case-insensitive), and returns `matches / totalQueryWords`. The combined score is `semanticScore * semanticWeight + keywordScore * (1 - semanticWeight)`, with `semanticWeight` defaulting to 0.7.

> **Advanced Note:** Hybrid retrieval often outperforms pure semantic search because it catches cases where exact keyword matches matter (proper nouns, technical terms, error codes) while still handling paraphrases and synonyms through semantic search. The BM25 algorithm is the industry-standard keyword retrieval method — consider using a library like `minisearch` for production keyword search.

---

## Section 6: Context Injection

### The RAG Prompt Template

How you inject retrieved context into the prompt significantly affects answer quality. The key decisions are: how to format sources, what instructions to give the model, and how to order chunks when many are retrieved.

### What to Build

Create `src/rag/context-injection.ts` with three functions.

**Function 1: `generateRAGAnswer(question, chunks): Promise<string>`**

```typescript
interface RetrievedChunk {
  text: string
  source: string
  score: number
}
```

Format each chunk as an XML-style source tag:

```typescript
;`<source id="${i + 1}" name="${chunk.source}" relevance="${chunk.score.toFixed(2)}">\n${chunk.text}\n</source>`
```

Call `generateText` with a system message containing guidelines: only use provided sources, cite inline with `[Source N]`, cite multiple sources when they agree, note contradictions, state what is missing if sources are insufficient, do not add information beyond sources. The user message wraps the formatted context in `<context>` tags followed by the question.

**Function 2: `selectChunksWithinBudget(chunks, maxTokens, reserveForPrompt?): RetrievedChunk[]`**

Given chunks sorted by relevance (best first), greedily select chunks that fit within `maxTokens - reserveForPrompt` (default reserve: 2000). Estimate token count as `Math.ceil(chunk.text.length / 4)`. Stop adding chunks when the next one would exceed the budget.

**Function 3: Source ordering strategies**

Implement three ordering functions that each take a `RetrievedChunk[]` and return a reordered copy:

- `orderByRelevance` -- sort descending by score (the default)
- `orderByDocument` -- sort by source name alphabetically (preserves narrative flow)
- `orderSandwich` -- place the most relevant chunks at the start and end, with lower-relevance chunks in the middle. This mitigates the "lost in the middle" effect where models pay less attention to middle context.

For the sandwich strategy: sort by relevance, then alternate placing items at the start (even indices) and in the middle (odd indices).

> **Beginner Note:** Start with "most relevant first" ordering. This works well in most cases. Only switch to attention-optimized ordering if you notice the model ignoring important middle context in long retrieval sets.

> **Advanced Note (Contextual Compression):** Retrieved chunks often contain irrelevant text alongside the useful parts. Before injecting chunks, you can use an LLM to extract only the query-relevant portions — this is the same summarization technique from Module 4 applied to retrieval results. Send each chunk with the query and ask the model to extract only the relevant parts. This reduces token usage and improves signal-to-noise, but adds an LLM call per chunk. Consider it when chunks are large and retrieval returns many results.

---

## Section 7: Citation and Attribution

### Structured Citation Extraction

Use `generateText` with `Output.object` to extract citations in a structured format. This gives you machine-readable citations you can verify programmatically.

### What to Build

Create `src/rag/citations.ts` with two functions.

**Function 1: `answerWithCitations(question, sources): Promise<CitedAnswer>`**

Define a Zod schema for structured citation output:

```typescript
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
```

The function takes a `question` and an array of `{ text: string; name: string }` sources. Format sources as `[Source N: name]\ntext` separated by `---`. Use `Output.object({ schema: CitedAnswerSchema })` in the `generateText` call. The system prompt should instruct the model to answer using only the provided sources and provide exact quotes for each claim.

**Function 2: `verifyCitations(answer, sources): { verified: boolean; details: ... }`**

This is a pure function (no LLM calls) that checks whether each citation's quote actually exists in the referenced source. For each citation:

1. Look up the source by `sourceIndex` (convert from 1-based to 0-based)
2. Check for an exact substring match (case-insensitive)
3. If no exact match, do a fuzzy match: split the quote into words longer than 3 characters, count how many appear in the source text, and compute `matchedWords / totalWords`
4. Consider the citation verified if the similarity exceeds 0.7

Return `{ verified: boolean, details: [...] }` where `verified` is true only if all citations passed.

Why is verification important? LLMs sometimes fabricate quotes even when explicitly instructed to cite. Programmatic verification catches these hallucinated citations before they reach the user.

---

## Section 8: Basic RAG Pipeline (End-to-End)

### Production-Ready RAG

Now bring all the pieces together into a complete, production-oriented RAG pipeline that ingests a directory of markdown files and answers questions with citations.

### What to Build

Create `src/rag/pipeline.ts` with a `RAGPipeline` class. This class composes the chunking, embedding, retrieval, and generation steps you built in earlier sections.

Here are the types:

```typescript
import { embed, embedMany, generateText, cosineSimilarity } from 'ai'
import { openai } from '@ai-sdk/openai'
import { mistral } from '@ai-sdk/mistral'
import { readdir, readFile } from 'node:fs/promises'
import { join, extname } from 'node:path'

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
    generationInputTokens: number
    generationOutputTokens: number
  }
}
```

The class needs:

```typescript
class RAGPipeline {
  private index: IndexedDocChunk[] = []
  private embeddingModel = openai.embedding('text-embedding-3-small')
  private chatModel = mistral('mistral-small-latest')

  async ingestDirectory(
    dirPath: string
  ): Promise<{ documentsProcessed: number; chunksCreated: number; tokensEmbedded: number }>
  async query(question: string, options?: { topK?: number; minScore?: number }): Promise<QueryResult>
  getStats(): { totalChunks: number; uniqueDocuments: number; avgChunkLength: number }
}
```

**`ingestDirectory`** should:

1. Read all `.md` files from the directory using `readdir` and filter by `extname`
2. For each file, chunk using a markdown-aware chunker that splits by headings, then by paragraphs for oversized sections (max 800 chars per chunk)
3. Embed all chunks in batches of 100 using `embedMany`, prepending the heading to the text for better embeddings (`${heading}\n${text}`)
4. Store in `this.index` and track total token usage

**`query`** should:

1. Embed the question
2. Score all indexed chunks by cosine similarity, filter by `minScore` (default 0.3), take top-K (default 5)
3. Format chunks as XML `<source>` tags with file path and section heading
4. Call `generateText` with a system prompt that requires source-only answers with `[Source N]` citations
5. Return the answer, source metadata (truncated to 200 chars), and token usage from both embedding and generation

**`getStats`** should return the total chunk count, unique document count (by `documentId`), and average chunk length.

Questions to consider: Why prepend the heading when embedding? What batch size is appropriate for embeddings? How do you handle the `usage` object when it might be undefined?

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
  // Your implementation here
}
```

Build this function using `generateText` with `Output.object({ schema: AssessmentSchema })`. Format the sources with numbered labels (e.g., `[Source 1]: ...`). The system message should instruct the judge model to assess faithfulness, relevance, and completeness, each scored 0-1. The user message should include the question, formatted sources, generated answer, and optional ground truth. Return the parsed output.

What makes a good system prompt for an LLM-as-judge? Why is it important to tell the judge to "be strict and objective"?

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
  // Your implementation here
}
```

Build this function. Iterate over each test case, run the pipeline's `query` method, then call `assessRAGResponse` with the question, answer, source texts, and expected answer. Accumulate the scores for each dimension and collect per-question results. At the end, compute averages by dividing totals by the dataset length.

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
  // Your implementation here
}
```

Build this function using set intersection. Convert both arrays to `Set`s, count true positives (sources in both sets), then compute precision (true positives / retrieved count), recall (true positives / expected count), and F1 (harmonic mean of precision and recall). Handle the edge case where either set is empty.

Why is F1 more informative than precision or recall alone? If you retrieve 100 sources and only 1 is relevant, what do precision and recall look like?

> **Beginner Note:** Start by manually reviewing 20-30 queries before investing in automated assessment. Manual review builds intuition about failure modes and helps you design better automated metrics.

> **Advanced Note:** LLM-as-judge assessment has its own biases and failure modes. The judge model might miss subtle hallucinations or be overly generous with scores. Consider using multiple judge models or combining automated metrics with periodic human review. The RAGAS framework (ragas.io) provides a more comprehensive assessment toolkit.

> **Provider Tip: Native Citations** — This module teaches citation attribution using structured output and manual source tracking. Several providers offer native citation features that can simplify this: **Mistral** supports citations via tool calls — you pass documents as tool responses and the model returns `ReferenceChunk` objects linking claims to specific sources. **Anthropic** offers a native Citations feature that returns precise character-range citations grounded to exact spans in provided documents. Both approaches can replace the manual citation extraction pattern taught here.

> **Local Alternative (Ollama):** RAG pipelines work with any model. Use `ollama('qwen3.5')` for generation and `ollama.embedding('qwen3-embedding:0.6b')` for embeddings (see Module 8). The retrieval and chunking logic is entirely model-agnostic. Local RAG is especially appealing for privacy-sensitive documents that you don't want to send to external APIs.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 10: Context Priority Ordering

When you retrieve more context than fits in the available window, you need a priority strategy. Production systems that assemble context from multiple sources use a consistent ordering:

1. **System instructions** — always included, highest priority
2. **Project-specific configuration** — usually included (e.g., coding standards, project rules)
3. **Recent conversation** — sliding window of recent messages
4. **Memory entries** — relevant facts from previous sessions
5. **Retrieved chunks** — from vector search or other retrieval
6. **Tool results** — truncated if needed, lowest priority

This priority ordering applies directly to RAG. When your retrieval returns 20 relevant chunks but only 5 fit in the context window, you need to decide which chunks to include. The same principle applies: closer to the user's immediate question means higher priority.

```typescript
interface ContextSource {
  content: string
  priority: number // Lower number = higher priority
  tokenCount: number
}

function assembleContext(sources: ContextSource[], maxTokens: number): string[] {
  /* ... */
}
```

The key design decision is where retrieved chunks sit in the priority order. In most RAG applications, system instructions and recent conversation take precedence, and retrieved chunks fill the remaining space. If your context budget is tight, you may need to truncate or summarize retrieved chunks rather than dropping conversation history.

---

## Section 11: Hierarchical Configuration as Retrieval

Production coding agents use an interesting retrieval pattern: they search for project instructions by walking up the directory tree. Starting from the current file, the system checks for configuration files at each directory level up to the repository root, then to the user's global configuration. Each level can extend or override the previous one.

This is file-based retrieval with hierarchical precedence. The "query" is "what instructions apply here?" and the retrieval strategy is directory traversal with merge semantics. Closer configuration takes priority over farther configuration, creating a specificity gradient — global rules apply everywhere, repo-level rules apply to the project, and directory-level rules apply to the current context.

```typescript
async function collectInstructions(filePath: string, rootDir: string): Promise<string[]> {
  /* ... */
}
```

Build this function. Starting from `dirname(filePath)`, walk up the directory tree while still within `rootDir`. At each level, check for an `INSTRUCTIONS.md` file. If found, prepend it to the array (using `unshift`) so the most specific instructions end up last (highest priority). Stop when you reach the root.

This mirrors how retrieval systems should rank results by relevance. The "closer" a configuration file is to the current context, the more relevant it is — just as a chunk that closely matches the query should rank higher than a loosely related one.

---

## Section 12: Lazy-Loading Referenced Files

Production systems often reference external files in their project instructions — "see coding-standards.md for style rules" or "refer to api-spec.yaml for endpoint details" — but they do not load those files preemptively. They only retrieve and inject the referenced content when the current task makes it relevant.

This is lazy retrieval: deferring the cost of loading until the benefit is clear. If the user is asking about database queries, there is no reason to load the coding standards file. If they ask about code style, then you load it.

The pattern is straightforward: instructions contain pointers (file paths or references), not content. When the user's request touches a relevant area, the system resolves the pointer and injects the content. This avoids consuming context window space with information that may never be needed.

Eager retrieval — loading everything up front — wastes context. In a RAG pipeline, this translates to a practical design principle: do not retrieve all potentially relevant chunks at query time. Instead, retrieve a focused set, generate, and if the answer is insufficient, retrieve additional context in a follow-up step. This is analogous to the "iterative retrieval" pattern where the system retrieves, generates, evaluates, and retrieves again if needed.

> **Key Insight:** Treat references as pointers, not as content to preload. Retrieve late, not early. This applies both to file-based configuration injection and to vector-based RAG pipeline design.

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
8. **Context priority ordering:** When retrieved chunks exceed available context, a priority strategy determines what gets included — system instructions first, then project config, conversation history, and finally retrieved chunks.
9. **Hierarchical retrieval:** Walking up a directory tree to collect and merge configuration files demonstrates retrieval with precedence — a pattern used extensively in production coding agents.
10. **Lazy retrieval:** Deferring the loading of referenced documents until they are actually needed saves context window space and improves relevance.

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

### Question 6 (Medium)

In a context priority ordering strategy, why do system instructions and recent conversation typically rank higher than retrieved chunks?

a) System instructions and conversation are cheaper to process
b) They define the task constraints and immediate context — retrieved chunks are supplementary, and dropping them degrades quality less than losing the user's recent messages or behavioral instructions
c) Retrieved chunks are always less accurate
d) The model cannot process more than two context types at once

**Answer: B**

**Explanation:** System instructions define how the model should behave, and recent conversation captures the user's immediate intent. If you drop system instructions, the model may ignore constraints. If you drop recent conversation, the model loses track of what the user just said. Retrieved chunks supplement the answer with evidence but are lower priority because the model can still produce a useful response without them — it just may lack supporting detail. This priority ordering ensures the most critical context survives when the window is tight.

---

### Question 7 (Hard)

You have a RAG system where project instructions reference five external files (coding standards, API spec, etc.), but loading all of them consumes 40% of the context window. How does lazy-loading referenced files improve this?

a) It compresses the files to use fewer tokens
b) It loads referenced files only when the current task is relevant to their content, avoiding context waste on information that may never be needed
c) It caches all files in memory so they load faster
d) It splits each file into chunks and embeds them

**Answer: B**

**Explanation:** Lazy-loading treats file references as pointers, not preloaded content. If the user asks about database queries, there is no reason to load the coding standards file. Only when the task touches a referenced area does the system resolve the pointer and inject the content. This keeps context available for retrieval results and conversation history instead of consuming it with potentially irrelevant configuration files.

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
import { embed, embedMany, generateText, cosineSimilarity } from 'ai'
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

---

### Exercise 3: Hierarchical Config Retrieval

Build a configuration loader that walks up a directory tree, collecting and merging instructions with precedence rules — a retrieval pattern used by production coding agents.

**Requirements:**

1. Create `src/exercises/m09/ex03-hierarchical-retrieval.ts`
2. Implement a `collectInstructions(filePath, rootDir)` function that:
   - Starts at the directory containing `filePath`
   - Walks up to `rootDir`, checking each directory for an `INSTRUCTIONS.md` file
   - Collects all found instruction files
   - Returns them ordered by specificity (closest directory = highest priority)
3. Implement a `mergeInstructions(instructions)` function that:
   - Concatenates instructions with clear section separators
   - Gives higher priority to more specific (closer) instructions
   - Respects a token budget — if total instructions exceed the budget, drops the least specific (farthest) ones first
4. Create a test directory structure with instructions at three levels (root, project, subdirectory) and demonstrate the merge behavior

```typescript
interface InstructionFile {
  path: string
  content: string
  depth: number // 0 = closest to file, higher = farther
}

async function collectInstructions(filePath: string, rootDir: string): Promise<InstructionFile[]> {
  // TODO: Walk up from filePath to rootDir, collecting INSTRUCTIONS.md files
  throw new Error('Not implemented')
}

function mergeInstructions(instructions: InstructionFile[], maxTokens: number): string {
  // TODO: Merge with priority ordering, respecting token budget
  throw new Error('Not implemented')
}
```

**Expected behavior:**

- Given a file at `project/src/utils/helper.ts` and root at `project/`, the loader checks `project/src/utils/`, `project/src/`, and `project/` for instruction files
- Instructions from `project/src/utils/` override those from `project/` when they conflict
- If total tokens exceed the budget, instructions from `project/` are dropped first

**Test specification:**

```typescript
// tests/exercises/m09/ex03-hierarchical-retrieval.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 9.3: Hierarchical Config Retrieval', () => {
  it('should collect instructions from multiple directory levels', async () => {
    const instructions = await collectInstructions('test-project/src/utils/helper.ts', 'test-project')
    expect(instructions.length).toBeGreaterThanOrEqual(1)
  })

  it('should order instructions by specificity (closest first)', async () => {
    const instructions = await collectInstructions('test-project/src/utils/helper.ts', 'test-project')
    expect(instructions[0].depth).toBe(0) // Closest directory
  })

  it('should respect token budget by dropping least specific first', () => {
    const merged = mergeInstructions(testInstructions, 500)
    // Most specific instructions should be preserved
    expect(merged).toContain('utils-level instruction')
  })
})
```
