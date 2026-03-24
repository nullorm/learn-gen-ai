# Module 10: Advanced RAG

## Learning Objectives

- Identify the failure modes of naive RAG and understand why simple vector search is often insufficient
- Implement query transformation techniques including rewriting, expansion, and decomposition
- Build a HyDE (Hypothetical Document Embeddings) pipeline that generates hypothetical answers before retrieval
- Combine semantic search with keyword/BM25 search using hybrid retrieval strategies
- Apply reranking with cross-encoders, Cohere Rerank, and LLM-based reranking to improve precision
- Understand structure-aware retrieval and build tree indexes for navigating long, structured documents
- Implement LLM-navigated tree search as an alternative to vector similarity for structured document retrieval
- Build a systematic RAG assessment framework with precision, recall, faithfulness, and relevance metrics

---

## Why Should I Care?

If you completed Module 9, you have a working RAG pipeline. You can embed documents, store them in a vector database, retrieve relevant chunks, and inject them into a prompt. That pipeline works well for simple cases — but it fails in predictable, frustrating ways on real-world data.

A user asks "What were our Q3 revenue numbers?" and the naive pipeline retrieves chunks about Q2 revenue because the embedding vectors are similar. A user asks a compound question — "Compare our hiring strategy in 2023 vs 2024" — and the pipeline retrieves chunks about one year but not the other. A user asks a question that requires reasoning across multiple documents, and the pipeline retrieves fragments that individually seem relevant but together miss the point.

These are not edge cases. They are the normal experience of deploying RAG in production. The techniques in this module — query transformation, HyDE, hybrid search, reranking, and structure-aware tree indexing — are how you move from a demo that works on cherry-picked examples to a system that works reliably on arbitrary user queries.

Advanced RAG is not one technique. It is a toolkit of composable strategies. You will rarely use all of them at once. The skill is knowing which ones to apply for your specific failure modes, and this module gives you that diagnostic ability along with the implementation patterns.

---

## Connection to Other Modules

This module builds directly on **Module 9 (RAG Fundamentals)**, extending its retrieval pipeline with more sophisticated strategies. You will need the embedding and vector store concepts from **Module 8 (Embeddings & Similarity)**.

- **Module 11 (Document Processing)** extends the ingestion side of the pipeline — better chunking, metadata extraction, and format handling.
- **Module 12 (Knowledge Graphs)** provides an alternative retrieval strategy (graph traversal) that complements the vector-based approaches here.
- **Module 19 (Evals & Testing)** uses the assessment framework introduced in Section 8 as a foundation for broader LLM testing.

Think of Module 9 as building the engine and this module as tuning it for performance.

---

## Section 1: Limitations of Naive RAG

### What Goes Wrong

Naive RAG — embed the query, find the top-K nearest chunks, inject them into a prompt — has three fundamental failure modes.

**Wrong chunks retrieved.** Embedding similarity does not equal semantic relevance. The query "What is our refund policy?" might retrieve chunks containing the word "refund" in unrelated contexts (e.g., "We do not offer refunds on custom orders" from a blog post, rather than the actual refund policy page). Embedding models capture surface-level semantic similarity, but they cannot always distinguish between mentions and definitions.

**Missed context.** A single query might need information scattered across multiple documents or sections. The naive approach retrieves the top-K most similar chunks, but those chunks might all come from the same document, leaving gaps. Compound questions ("How did our approach to X change between 2023 and 2024?") are especially vulnerable — the pipeline finds chunks about one time period but not the other.

**Hallucination despite retrieval.** Even when the right chunks are retrieved, the model might hallucinate details not present in the context. This happens when the context is partially relevant — the model "fills in the gaps" with plausible-sounding but fabricated information. The user sees cited sources and assumes the answer is grounded, but it is not.

```typescript
// src/advanced-rag/naive-failures.ts

import { generateText, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Simulating the three failure modes

// Failure Mode 1: Wrong chunks retrieved
// The query and a misleading chunk have high cosine similarity
// but the chunk does not actually answer the question.
const query = 'What is the company refund policy?'

const misleadingChunk = `
  Our blog post from March: "Many customers ask about refunds.
  In the tech industry, refund rates vary between 2-8%.
  Our customer satisfaction scores remain high."
`

const correctChunk = `
  Refund Policy (effective Jan 2024): Full refunds are available
  within 30 days of purchase. After 30 days, store credit is
  issued. Contact support@example.com to initiate a refund.
`

// Both chunks will have high similarity to the query because they
// contain "refund" — but only the second one answers the question.

// Failure Mode 2: Missed context with compound queries
const compoundQuery = 'Compare our hiring in 2023 vs 2024'

// Top-K retrieval might return 5 chunks, all from the 2024
// hiring report, because those chunks are slightly more similar
// to the query embedding. The 2023 data is missed entirely.

// Failure Mode 3: Hallucination with partial context
const partialContext = `
  Q3 2024 Revenue Report:
  - Total revenue: $45.2M
  - Growth: 23% YoY
  [Note: Regional breakdown available in Appendix B]
`

// If the user asks "What was Q3 revenue by region?", the model
// might fabricate regional numbers because it sees "regional
// breakdown" mentioned but does not have Appendix B.

async function demonstrateHallucination(): Promise<void> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `Answer based ONLY on the provided context. If the context
does not contain enough information, say "I don't have enough
information to answer that."`,
    messages: [
      {
        role: 'user',
        content: `Context: ${partialContext}

Question: What was Q3 2024 revenue broken down by region?`,
      },
    ],
  })

  console.log('Response:', result.text)
  // A well-prompted model SHOULD say it lacks the information.
  // But weaker prompts or less capable models will fabricate numbers.
}

demonstrateHallucination().catch(console.error)
```

### The Advanced RAG Toolkit

Each failure mode has corresponding solutions:

| Failure Mode           | Solutions                                            | Module Section |
| ---------------------- | ---------------------------------------------------- | -------------- |
| Wrong chunks retrieved | Query transformation, HyDE, hybrid search            | Sections 2-3   |
| Missed context         | Query decomposition, hybrid search, tree indexing     | Sections 2, 3, 5-7 |
| Hallucination          | Reranking, structure-aware retrieval                  | Sections 4-7   |
| Unknown quality        | Assessment framework                                 | Section 8      |

> **Beginner Note:** You do not need to implement all these techniques at once. Start with the naive pipeline from Module 9, measure its failures, then add techniques one at a time. Each technique addresses specific failure modes — diagnose first, then prescribe.

> **Advanced Note:** In production systems, the combination of techniques matters. A common high-performing stack is: query rewriting + hybrid search + reranking. HyDE and tree indexing add latency and cost, so they are reserved for high-stakes use cases where accuracy justifies the overhead.

---

## Section 2: Query Transformation

### Why Transform Queries?

Users write queries like humans — ambiguous, incomplete, and informal. "Tell me about that pricing thing we discussed" is a perfectly natural question to a colleague, but a terrible search query. Query transformation rewrites the user's raw input into one or more queries optimized for retrieval.

There are three main strategies:

1. **Query Rewriting** — Rephrase the query for better retrieval
2. **Query Expansion** — Generate multiple related queries to broaden recall
3. **Query Decomposition** — Break a complex question into simpler sub-questions

### Query Rewriting

The simplest transformation: use an LLM to rewrite the user's query into a better search query.

```typescript
// src/advanced-rag/query-rewriting.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function rewriteQuery(originalQuery: string, conversationContext?: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a search query optimizer. Rewrite the user's question
into a clear, specific search query that will retrieve the most relevant
documents from a knowledge base.

Rules:
- Remove filler words and ambiguity
- Expand abbreviations and acronyms
- Include relevant synonyms in parentheses if helpful
- If conversation context is provided, resolve pronouns and references
- Return ONLY the rewritten query, nothing else`,
    messages: [
      {
        role: 'user',
        content: conversationContext
          ? `Conversation context: ${conversationContext}\n\nUser query: ${originalQuery}`
          : `User query: ${originalQuery}`,
      },
    ],
    temperature: 0,
    maxOutputTokens: 200,
  })

  return result.text.trim()
}

// Examples
async function main(): Promise<void> {
  // Ambiguous query
  const rewritten1 = await rewriteQuery('Tell me about that pricing thing')
  console.log('Rewritten:', rewritten1)
  // Expected: "product pricing plans and pricing structure"

  // Query with pronouns needing context resolution
  const rewritten2 = await rewriteQuery(
    'What are its side effects?',
    'The user previously asked about ibuprofen dosage.'
  )
  console.log('Rewritten:', rewritten2)
  // Expected: "ibuprofen side effects and adverse reactions"

  // Informal query
  const rewritten3 = await rewriteQuery('how do I fix the thing where it crashes on startup')
  console.log('Rewritten:', rewritten3)
  // Expected: "troubleshoot application crash on startup"
}

main().catch(console.error)
```

### Query Expansion

Instead of one rewritten query, generate multiple queries that cover different aspects of the original question. This dramatically improves recall — the chance of finding all relevant documents.

```typescript
// src/advanced-rag/query-expansion.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const ExpandedQueriesSchema = z.object({
  original: z.string().describe('The original user query'),
  expanded: z
    .array(
      z.object({
        query: z.string().describe('An expanded search query'),
        rationale: z.string().describe('Why this expansion helps retrieval'),
      })
    )
    .describe('3-5 expanded queries covering different angles'),
})

type ExpandedQueries = z.infer<typeof ExpandedQueriesSchema>

async function expandQuery(originalQuery: string): Promise<ExpandedQueries> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ExpandedQueriesSchema }),
    system: `You are a search query expansion expert. Given a user query,
generate 3-5 alternative queries that cover different aspects, synonyms,
and phrasings of the same information need.

Each expanded query should:
- Target a slightly different angle of the same question
- Use different keywords to maximize recall
- Be specific enough to retrieve relevant documents`,
    messages: [{ role: 'user', content: originalQuery }],
    temperature: 0.3,
  })

  return output
}

async function retrieveWithExpansion(
  originalQuery: string,
  searchFn: (query: string) => Promise<string[]>
): Promise<string[]> {
  const expanded = await expandQuery(originalQuery)

  console.log('Original:', expanded.original)
  console.log('Expanded queries:')
  for (const eq of expanded.expanded) {
    console.log(`  - ${eq.query} (${eq.rationale})`)
  }

  // Retrieve for each expanded query
  const allQueries = [originalQuery, ...expanded.expanded.map(e => e.query)]
  const allResults: string[] = []
  const seen = new Set<string>()

  for (const query of allQueries) {
    const results = await searchFn(query)
    for (const result of results) {
      if (!seen.has(result)) {
        seen.add(result)
        allResults.push(result)
      }
    }
  }

  return allResults
}

// Example usage
async function main(): Promise<void> {
  const expanded = await expandQuery('How do I handle authentication in our API?')
  console.log(JSON.stringify(expanded, null, 2))
}

main().catch(console.error)
```

### Query Decomposition

For complex, multi-part questions, break the query into independent sub-questions. Each sub-question can be retrieved and answered independently, then the results are synthesized.

```typescript
// src/advanced-rag/query-decomposition.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const DecomposedQuerySchema = z.object({
  isComplex: z.boolean().describe('Whether the query needs decomposition'),
  subQuestions: z
    .array(
      z.object({
        question: z.string().describe('A self-contained sub-question'),
        dependsOn: z.array(z.number()).describe('Indices of sub-questions this depends on (empty if independent)'),
      })
    )
    .describe('Sub-questions in execution order'),
})

type DecomposedQuery = z.infer<typeof DecomposedQuerySchema>

async function decomposeQuery(query: string): Promise<DecomposedQuery> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: DecomposedQuerySchema }),
    system: `You are a query decomposition expert. Analyze whether a question
is complex (requires multiple pieces of information) and if so, break it
into simpler sub-questions.

A question should be decomposed when it:
- Asks about multiple entities or time periods
- Requires comparison between things
- Has implicit sub-questions that must be answered first
- Involves multiple steps of reasoning

Each sub-question should be self-contained and answerable independently.
Mark dependencies when one sub-question's answer is needed for another.`,
    messages: [{ role: 'user', content: query }],
    temperature: 0,
  })

  return output
}

interface SubAnswer {
  question: string
  answer: string
  sources: string[]
}

async function answerWithDecomposition(
  query: string,
  retrieveAndAnswer: (question: string) => Promise<{ answer: string; sources: string[] }>
): Promise<string> {
  const decomposed = await decomposeQuery(query)

  if (!decomposed.isComplex) {
    // Simple query — just answer directly
    const result = await retrieveAndAnswer(query)
    return result.answer
  }

  console.log(`Decomposed into ${decomposed.subQuestions.length} sub-questions:`)

  // Answer sub-questions in dependency order
  const subAnswers: SubAnswer[] = []

  for (let i = 0; i < decomposed.subQuestions.length; i++) {
    const subQ = decomposed.subQuestions[i]
    console.log(`  [${i + 1}] ${subQ.question}`)

    // Build context from dependencies
    let contextFromDeps = ''
    for (const depIdx of subQ.dependsOn) {
      if (subAnswers[depIdx]) {
        contextFromDeps += `\nPrevious finding: ${subAnswers[depIdx].answer}\n`
      }
    }

    const augmentedQuestion = contextFromDeps
      ? `${subQ.question}\n\nContext from previous findings: ${contextFromDeps}`
      : subQ.question

    const result = await retrieveAndAnswer(augmentedQuestion)
    subAnswers.push({
      question: subQ.question,
      answer: result.answer,
      sources: result.sources,
    })
  }

  // Synthesize final answer from sub-answers
  const synthesis = await generateText({
    model: mistral('mistral-small-latest'),
    system: `Synthesize a comprehensive answer from the sub-question answers
below. Maintain all specific details and cite sources where available.`,
    messages: [
      {
        role: 'user',
        content: `Original question: ${query}

Sub-question answers:
${subAnswers
  .map((sa, i) => `${i + 1}. Q: ${sa.question}\n   A: ${sa.answer}\n   Sources: ${sa.sources.join(', ')}`)
  .join('\n\n')}

Provide a unified answer that addresses the original question completely.`,
      },
    ],
  })

  return synthesis.text
}

// Example
async function main(): Promise<void> {
  const decomposed = await decomposeQuery(
    'Compare our engineering team size and hiring velocity in Q1 2023 vs Q1 2024, and explain what drove the changes'
  )
  console.log(JSON.stringify(decomposed, null, 2))
}

main().catch(console.error)
```

> **Beginner Note:** Query transformation adds an LLM call before retrieval, which means more latency and cost. For simple, well-formed queries, it is often unnecessary. Start without it and add it when you see retrieval failures caused by query quality.

> **Advanced Note:** You can make query transformation conditional — use a lightweight classifier to decide whether the query needs rewriting, expansion, or decomposition. This avoids the overhead for simple queries while still handling complex ones correctly.

### HyDE — Hypothetical Document Embeddings

#### The Core Insight

Standard retrieval embeds the _query_ and searches for similar _documents_. But queries and documents are fundamentally different kinds of text. A query is short and interrogative ("What is our refund policy?"). A document is long and declarative ("Refund Policy: Full refunds are available within 30 days..."). Their embedding vectors may not be close in the vector space even when they are semantically related.

HyDE (Hypothetical Document Embeddings) flips the approach: instead of embedding the query, you ask an LLM to _generate a hypothetical answer_ to the query, then embed _that hypothetical answer_. The hypothetical answer is a declarative passage — it looks like the documents in your corpus — so its embedding is closer to the real relevant documents.

The hypothetical answer does not need to be correct. It just needs to _look like_ the kind of document that would answer the question. A hallucinated answer with the right vocabulary and structure will have a better embedding for retrieval than the raw query.

#### Implementing HyDE

```typescript
// src/advanced-rag/hyde.ts

import { generateText, embed, cosineSimilarity } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface HyDEResult {
  originalQuery: string
  hypotheticalDocument: string
  queryEmbedding: number[]
  hydeEmbedding: number[]
}

async function generateHypotheticalDocument(query: string, domainContext?: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a technical writer. Given a question, write a short
passage (3-5 sentences) that would be the ideal answer found in a document.

Write as if you are writing the actual documentation or knowledge base article
that answers this question. Be specific, use declarative statements, and
include the kind of terminology that would appear in a real document.

${domainContext ? `Domain context: ${domainContext}` : ''}

Do NOT say "I think" or "The answer is". Write the passage as if it IS the
source document.`,
    messages: [{ role: 'user', content: query }],
    temperature: 0.3,
    maxOutputTokens: 300,
  })

  return result.text
}

async function hydeRetrieve(
  query: string,
  embeddingModel: Parameters<typeof embed>[0]['model'],
  searchIndex: (embedding: number[], topK: number) => Promise<string[]>,
  options: { topK?: number; domainContext?: string } = {}
): Promise<{ results: string[]; hypotheticalDoc: string }> {
  const { topK = 5, domainContext } = options

  // Step 1: Generate hypothetical document
  const hypotheticalDoc = await generateHypotheticalDocument(query, domainContext)
  console.log('Hypothetical document:', hypotheticalDoc)

  // Step 2: Embed the hypothetical document (not the query)
  const { embedding } = await embed({
    model: embeddingModel,
    value: hypotheticalDoc,
  })

  // Step 3: Search using the hypothetical document embedding
  const results = await searchIndex(embedding, topK)

  return { results, hypotheticalDoc }
}

// Full HyDE pipeline with comparison to standard retrieval
async function compareStandardVsHyDE(
  query: string,
  embeddingModel: Parameters<typeof embed>[0]['model'],
  searchIndex: (embedding: number[], topK: number) => Promise<string[]>
): Promise<void> {
  // Standard: embed the query directly
  const { embedding: queryEmbedding } = await embed({
    model: embeddingModel,
    value: query,
  })
  const standardResults = await searchIndex(queryEmbedding, 5)

  // HyDE: embed a hypothetical answer
  const { results: hydeResults, hypotheticalDoc } = await hydeRetrieve(query, embeddingModel, searchIndex)

  console.log('\n=== Standard Retrieval ===')
  standardResults.forEach((r, i) => console.log(`  ${i + 1}. ${r.slice(0, 100)}...`))

  console.log('\n=== HyDE Retrieval ===')
  console.log(`  Hypothetical: ${hypotheticalDoc.slice(0, 100)}...`)
  hydeResults.forEach((r, i) => console.log(`  ${i + 1}. ${r.slice(0, 100)}...`))
}

export { generateHypotheticalDocument, hydeRetrieve, compareStandardVsHyDE }
```

#### Multi-HyDE: Multiple Hypothetical Documents

A single hypothetical document might capture only one angle of the query. Generate multiple hypothetical documents and use their embeddings for broader recall.

```typescript
// src/advanced-rag/multi-hyde.ts

import { generateText, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface MultiHyDEOptions {
  numHypothetical: number
  temperature: number
  domainContext?: string
}

async function multiHyDE(
  query: string,
  embeddingModel: Parameters<typeof embed>[0]['model'],
  searchIndex: (embedding: number[], topK: number) => Promise<string[]>,
  options: MultiHyDEOptions = { numHypothetical: 3, temperature: 0.5 }
): Promise<string[]> {
  const { numHypothetical, temperature, domainContext } = options

  // Generate multiple hypothetical documents with temperature variation
  const hypotheticals: string[] = []

  for (let i = 0; i < numHypothetical; i++) {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      system: `Write a short passage (3-5 sentences) that would be found in
a document answering this question. Write it as the source document, not
as a response. Use a different angle or emphasis than other passages.
${domainContext ? `Domain: ${domainContext}` : ''}`,
      messages: [{ role: 'user', content: query }],
      temperature: temperature + i * 0.1, // Slight variation
      maxOutputTokens: 300,
    })
    hypotheticals.push(result.text)
  }

  // Embed all hypothetical documents
  const embeddings = await Promise.all(
    hypotheticals.map(doc => embed({ model: embeddingModel, value: doc }).then(r => r.embedding))
  )

  // Option A: Average the embeddings (centroid approach)
  const dimensions = embeddings[0].length
  const centroid = new Array(dimensions).fill(0) as number[]

  for (const emb of embeddings) {
    for (let d = 0; d < dimensions; d++) {
      centroid[d] += emb[d] / embeddings.length
    }
  }

  // Search with the centroid embedding
  const centroidResults = await searchIndex(centroid, 5)

  // Option B: Search with each embedding and merge (union approach)
  const allResults: string[] = []
  const seen = new Set<string>()

  for (const emb of embeddings) {
    const results = await searchIndex(emb, 3)
    for (const r of results) {
      if (!seen.has(r)) {
        seen.add(r)
        allResults.push(r)
      }
    }
  }

  // In practice, you would choose one approach.
  // The union approach typically has better recall.
  console.log(`Generated ${numHypothetical} hypothetical documents`)
  console.log(`Centroid retrieval: ${centroidResults.length} results`)
  console.log(`Union retrieval: ${allResults.length} unique results`)

  return allResults
}

export { multiHyDE }
```

> **Beginner Note:** HyDE adds one LLM call (generating the hypothetical document) before retrieval. This typically adds 1-3 seconds of latency. For interactive applications, consider whether this tradeoff is worth it. For batch processing or high-stakes queries, it almost always is.

> **Advanced Note:** HyDE works best when your corpus contains documents that are structurally similar to each other (e.g., technical documentation, policies, reports). It works less well for heterogeneous corpora where the "shape" of a relevant document is unpredictable. You can mitigate this by using domain-specific prompts for the hypothetical document generation.

---

## Section 3: Hybrid Search

### Why Combine Semantic and Keyword Search?

Semantic search (embedding similarity) is good at understanding meaning — it knows that "automobile" and "car" are similar. But it is bad at exact matches — searching for error code "ERR_0x4F2A" with embeddings is unreliable because the embedding model may not preserve the exact string.

Keyword search (BM25/TF-IDF) is the opposite: it excels at exact matches and rare terms but cannot handle synonyms or paraphrasing. "How to fix a car" will not find a document about "automobile repair."

Hybrid search combines both: run a semantic search AND a keyword search, then merge the results. This gives you the meaning-understanding of embeddings with the precision of keyword matching.

### BM25 Implementation

BM25 (Best Matching 25) is the industry-standard keyword relevance algorithm. It is essentially a sophisticated version of TF-IDF that accounts for document length and term saturation.

```typescript
// src/advanced-rag/bm25.ts

interface BM25Options {
  k1: number // Term frequency saturation (default 1.2)
  b: number // Length normalization (default 0.75)
}

interface BM25Index {
  documents: string[]
  tokenizedDocs: string[][]
  docLengths: number[]
  avgDocLength: number
  termDocFrequency: Map<string, number> // How many docs contain each term
  totalDocs: number
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(token => token.length > 1)
}

function buildBM25Index(documents: string[]): BM25Index {
  const tokenizedDocs = documents.map(tokenize)
  const docLengths = tokenizedDocs.map(tokens => tokens.length)
  const avgDocLength = docLengths.reduce((sum, len) => sum + len, 0) / documents.length

  const termDocFrequency = new Map<string, number>()
  for (const tokens of tokenizedDocs) {
    const uniqueTerms = new Set(tokens)
    for (const term of uniqueTerms) {
      termDocFrequency.set(term, (termDocFrequency.get(term) ?? 0) + 1)
    }
  }

  return {
    documents,
    tokenizedDocs,
    docLengths,
    avgDocLength,
    termDocFrequency,
    totalDocs: documents.length,
  }
}

function bm25Score(
  query: string,
  index: BM25Index,
  options: BM25Options = { k1: 1.2, b: 0.75 }
): Array<{ document: string; score: number; index: number }> {
  const queryTerms = tokenize(query)
  const { k1, b } = options
  const scores: Array<{
    document: string
    score: number
    index: number
  }> = []

  for (let i = 0; i < index.totalDocs; i++) {
    let score = 0
    const docTokens = index.tokenizedDocs[i]
    const docLength = index.docLengths[i]

    // Count term frequencies in this document
    const termFreqs = new Map<string, number>()
    for (const token of docTokens) {
      termFreqs.set(token, (termFreqs.get(token) ?? 0) + 1)
    }

    for (const term of queryTerms) {
      const tf = termFreqs.get(term) ?? 0
      if (tf === 0) continue

      const df = index.termDocFrequency.get(term) ?? 0

      // IDF component
      const idf = Math.log((index.totalDocs - df + 0.5) / (df + 0.5) + 1)

      // TF component with length normalization
      const tfNormalized = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLength / index.avgDocLength)))

      score += idf * tfNormalized
    }

    scores.push({ document: index.documents[i], score, index: i })
  }

  return scores.sort((a, b) => b.score - a.score)
}

// Example usage
const documents = [
  'The refund policy allows returns within 30 days of purchase.',
  'Error code ERR_0x4F2A indicates a database connection timeout.',
  'Our automobile repair service covers engine and transmission work.',
  'Customer satisfaction surveys show 92% approval rating.',
  'The API rate limit is 100 requests per minute per API key.',
]

const index = buildBM25Index(documents)

// Keyword search excels at exact matches
const results = bm25Score('ERR_0x4F2A', index)
console.log("BM25 results for 'ERR_0x4F2A':")
results.slice(0, 3).forEach(r => {
  console.log(`  Score: ${r.score.toFixed(3)} — ${r.document.slice(0, 80)}`)
})

export { buildBM25Index, bm25Score, tokenize, type BM25Index }
```

### Combining Semantic and Keyword Search

The key challenge in hybrid search is merging results from two different scoring systems. Semantic search returns cosine similarity scores (0-1), while BM25 returns unbounded relevance scores. You need to normalize and combine them.

```typescript
// src/advanced-rag/hybrid-search.ts

import { embed } from 'ai'
import { buildBM25Index, bm25Score, type BM25Index } from './bm25.js'

interface HybridResult {
  document: string
  semanticScore: number
  keywordScore: number
  combinedScore: number
  index: number
}

interface HybridSearchOptions {
  semanticWeight: number // 0-1, weight for semantic search
  topK: number
}

function normalizeScores(scores: Array<{ score: number; index: number }>): Map<number, number> {
  if (scores.length === 0) return new Map()

  const maxScore = Math.max(...scores.map(s => s.score))
  const minScore = Math.min(...scores.map(s => s.score))
  const range = maxScore - minScore || 1

  const normalized = new Map<number, number>()
  for (const { score, index } of scores) {
    normalized.set(index, (score - minScore) / range)
  }
  return normalized
}

async function hybridSearch(
  query: string,
  documents: string[],
  embeddingModel: Parameters<typeof embed>[0]['model'],
  options: HybridSearchOptions = { semanticWeight: 0.7, topK: 5 }
): Promise<HybridResult[]> {
  const { semanticWeight, topK } = options
  const keywordWeight = 1 - semanticWeight

  // 1. Semantic search
  const queryEmbedding = await embed({
    model: embeddingModel,
    value: query,
  })

  const documentEmbeddings = await Promise.all(
    documents.map(doc => embed({ model: embeddingModel, value: doc }).then(r => r.embedding))
  )

  const semanticScores = documentEmbeddings.map((docEmb, i) => {
    // Note: The Vercel AI SDK exports cosineSimilarity from 'ai'.
    // We implement it manually here for learning purposes.
    // In production code, use: import { cosineSimilarity } from 'ai'
    let dotProduct = 0
    let normA = 0
    let normB = 0
    for (let d = 0; d < queryEmbedding.embedding.length; d++) {
      dotProduct += queryEmbedding.embedding[d] * docEmb[d]
      normA += queryEmbedding.embedding[d] ** 2
      normB += docEmb[d] ** 2
    }
    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
    return { score: similarity, index: i }
  })

  // 2. Keyword search (BM25)
  const bm25Index = buildBM25Index(documents)
  const keywordResults = bm25Score(query, bm25Index)
  const keywordScores = keywordResults.map(r => ({
    score: r.score,
    index: r.index,
  }))

  // 3. Normalize both score sets to 0-1 range
  const normalizedSemantic = normalizeScores(semanticScores)
  const normalizedKeyword = normalizeScores(keywordScores)

  // 4. Combine with weighted sum
  const combined: HybridResult[] = documents.map((doc, i) => ({
    document: doc,
    semanticScore: normalizedSemantic.get(i) ?? 0,
    keywordScore: normalizedKeyword.get(i) ?? 0,
    combinedScore: semanticWeight * (normalizedSemantic.get(i) ?? 0) + keywordWeight * (normalizedKeyword.get(i) ?? 0),
    index: i,
  }))

  // Sort by combined score and return top-K
  return combined.sort((a, b) => b.combinedScore - a.combinedScore).slice(0, topK)
}

export { hybridSearch, type HybridResult }
```

### Reciprocal Rank Fusion (RRF)

An alternative to weighted normalization is Reciprocal Rank Fusion (RRF), which combines rankings rather than scores. RRF is simpler and often more robust because it does not require score normalization.

```typescript
// src/advanced-rag/rrf.ts

interface RRFResult {
  document: string
  rrfScore: number
  ranks: { semantic: number; keyword: number }
}

function reciprocalRankFusion(
  rankings: Array<Array<{ document: string; index: number }>>,
  k: number = 60 // RRF constant, default 60
): RRFResult[] {
  const scoreMap = new Map<number, { rrfScore: number; ranks: number[] }>()

  for (let r = 0; r < rankings.length; r++) {
    const ranking = rankings[r]
    for (let rank = 0; rank < ranking.length; rank++) {
      const { index } = ranking[rank]
      const existing = scoreMap.get(index) ?? {
        rrfScore: 0,
        ranks: new Array(rankings.length).fill(-1),
      }
      existing.rrfScore += 1 / (k + rank + 1)
      existing.ranks[r] = rank + 1
      scoreMap.set(index, existing)
    }
  }

  const results: RRFResult[] = []
  for (const [index, { rrfScore, ranks }] of scoreMap) {
    results.push({
      document: rankings[0].find(r => r.index === index)?.document ?? '',
      rrfScore,
      ranks: { semantic: ranks[0], keyword: ranks[1] },
    })
  }

  return results.sort((a, b) => b.rrfScore - a.rrfScore)
}

export { reciprocalRankFusion, type RRFResult }
```

> **Beginner Note:** Start with a 70/30 semantic/keyword split. If your corpus has many exact identifiers (error codes, product SKUs, file paths), increase the keyword weight. If your queries are mostly natural language, increase the semantic weight.

> **Advanced Note:** Many vector databases (Pinecone, Weaviate, Qdrant) support hybrid search natively, combining dense (embedding) and sparse (BM25/SPLADE) vectors in a single query. Use native support when available — it is faster and handles normalization internally.

---

## Section 4: Reranking

### Why Rerank?

Retrieval (whether semantic, keyword, or hybrid) casts a wide net. It finds documents that are _probably_ relevant based on surface-level features — embedding proximity or keyword overlap. But "probably relevant" is not "actually relevant."

Reranking is a second pass that applies a more expensive, more accurate model to the retrieved candidates. The retriever finds 20-50 candidates quickly; the reranker scores each one carefully and returns the best 3-5. This two-stage approach gives you both speed (fast retrieval over millions of documents) and accuracy (careful reranking of a small set).

### Cross-Encoder Reranking

A cross-encoder takes the query and a document _together_ as input and outputs a relevance score. Unlike bi-encoders (which embed query and document separately), cross-encoders can model the fine-grained interaction between query and document tokens.

```typescript
// src/advanced-rag/reranking.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Approach 1: LLM-based reranking
// Use a language model to score relevance of each retrieved chunk

const RelevanceScoreSchema = z.object({
  score: z.number().min(0).max(10).describe('Relevance score from 0-10'),
  reasoning: z.string().describe('Brief explanation of the score'),
})

interface RankedDocument {
  document: string
  score: number
  reasoning: string
  originalRank: number
}

async function llmRerank(query: string, documents: string[], topK: number = 3): Promise<RankedDocument[]> {
  // Score each document independently
  const scorePromises = documents.map(async (doc, index) => {
    const { output } = await generateText({
      model: mistral('mistral-small-latest'),
      output: Output.object({ schema: RelevanceScoreSchema }),
      system: `You are a relevance judge. Score how relevant the document is
to answering the query. Consider:
- Does the document directly answer the query? (high score)
- Does it contain related but not directly useful information? (medium)
- Is it about a completely different topic? (low score)

Be precise: a score of 8-10 means it directly answers the query.
A score of 4-7 means it is somewhat related. Below 4 means not useful.`,
      messages: [
        {
          role: 'user',
          content: `Query: ${query}\n\nDocument: ${doc}`,
        },
      ],
      temperature: 0,
    })

    return {
      document: doc,
      score: output.score,
      reasoning: output.reasoning,
      originalRank: index + 1,
    }
  })

  const results = await Promise.all(scorePromises)
  return results.sort((a, b) => b.score - a.score).slice(0, topK)
}

// Approach 2: Listwise LLM reranking
// Present all documents at once and ask the model to rank them
// More efficient (1 LLM call vs N calls) but limited by context window

const ListwiseRankingSchema = z.object({
  rankings: z
    .array(
      z.object({
        documentIndex: z.number().describe('0-based index of the document'),
        relevanceScore: z.number().min(0).max(10),
        reasoning: z.string(),
      })
    )
    .describe('Documents ranked from most to least relevant'),
})

async function listwiseRerank(query: string, documents: string[], topK: number = 3): Promise<RankedDocument[]> {
  const docsFormatted = documents.map((doc, i) => `[Document ${i}]: ${doc}`).join('\n\n')

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ListwiseRankingSchema }),
    system: `You are a relevance ranking expert. Given a query and a list of
documents, rank ALL documents from most to least relevant to the query.
Assign a relevance score (0-10) to each document.`,
    messages: [
      {
        role: 'user',
        content: `Query: ${query}\n\nDocuments:\n${docsFormatted}`,
      },
    ],
    temperature: 0,
  })

  return output.rankings
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, topK)
    .map(r => ({
      document: documents[r.documentIndex],
      score: r.relevanceScore,
      reasoning: r.reasoning,
      originalRank: r.documentIndex + 1,
    }))
}

export { llmRerank, listwiseRerank, type RankedDocument }
```

### Cohere Rerank Integration

Cohere provides a purpose-built reranking API that is faster and cheaper than using a general LLM. The Vercel AI SDK does not wrap reranking APIs directly, but you can call Cohere's API alongside your pipeline.

```typescript
// src/advanced-rag/cohere-rerank.ts

// Note: Cohere reranking is not part of Vercel AI SDK directly.
// This shows how to integrate it alongside your AI SDK pipeline.

interface CohereRerankResult {
  index: number
  relevanceScore: number
  document: string
}

async function cohereRerank(query: string, documents: string[], topK: number = 3): Promise<CohereRerankResult[]> {
  const response = await fetch('https://api.cohere.com/v2/rerank', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.COHERE_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'rerank-v3.5',
      query,
      documents: documents.map(doc => ({ text: doc })),
      top_n: topK,
      return_documents: true,
    }),
  })

  if (!response.ok) {
    throw new Error(`Cohere rerank failed: ${response.statusText}`)
  }

  const data = (await response.json()) as {
    results: Array<{
      index: number
      relevance_score: number
      document: { text: string }
    }>
  }

  return data.results.map(r => ({
    index: r.index,
    relevanceScore: r.relevance_score,
    document: r.document.text,
  }))
}

// Full pipeline: retrieve -> rerank -> generate
import { generateText, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function retrieveRerankGenerate(
  query: string,
  vectorSearch: (query: string, topK: number) => Promise<string[]>,
  options: { retrieveK: number; rerankK: number } = {
    retrieveK: 20,
    rerankK: 3,
  }
): Promise<string> {
  // Step 1: Broad retrieval (cast a wide net)
  const candidates = await vectorSearch(query, options.retrieveK)
  console.log(`Retrieved ${candidates.length} candidates`)

  // Step 2: Rerank (narrow to the best)
  const reranked = await cohereRerank(query, candidates, options.rerankK)
  console.log(
    `Reranked to ${reranked.length} documents:`,
    reranked.map(r => `[${r.relevanceScore.toFixed(3)}]`).join(', ')
  )

  // Step 3: Generate answer from top reranked documents
  const context = reranked.map(r => r.document).join('\n\n---\n\n')

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `Answer the question based on the provided context. If the context
does not contain sufficient information, say so clearly.`,
    messages: [
      {
        role: 'user',
        content: `Context:\n${context}\n\nQuestion: ${query}`,
      },
    ],
  })

  return result.text
}

export { cohereRerank, retrieveRerankGenerate }
```

> **Beginner Note:** Reranking is one of the highest-impact improvements you can make to a RAG pipeline. If you implement only one technique from this module, make it reranking. Retrieve 20 candidates, rerank to 3-5, and generate from those.

> **Advanced Note:** The optimal retrieve-K depends on your reranker's quality and the size of your corpus. With a strong reranker (Cohere rerank-v3.5 or a capable LLM), retrieving 50-100 candidates and reranking to 3-5 is effective. With a weaker reranker, keep retrieve-K lower (10-20) to avoid overwhelming it with irrelevant documents.

---

## Section 5: Structure-Aware Retrieval

### Why Similarity ≠ Relevance

Vector similarity search finds chunks that are semantically close to the query. But "close" is not always "relevant." Consider a financial analyst asking "What drove Q3 revenue growth?":

- A chunk about Q2 revenue might be very similar (same vocabulary: revenue, growth, quarterly) but is the wrong quarter
- A chunk about a new product launch in Q3 might be less similar (different vocabulary) but is the actual answer
- A chunk titled "Management Discussion: Factors Affecting Performance" might have low similarity to "revenue growth" but contains the exact analysis needed

The fundamental issue: embedding similarity measures lexical and semantic proximity, not logical relevance. For complex documents with structure (sections, chapters, hierarchies), the structure itself carries information that embeddings discard.

### The Tree Indexing Approach

Instead of chunking a document and embedding the chunks, tree indexing preserves the document's natural structure as a navigable hierarchy — like an LLM-optimized table of contents.

```
Document
├── Section 1: Company Overview
│   ├── 1.1 History
│   └── 1.2 Mission
├── Section 2: Financial Performance
│   ├── 2.1 Q1 Results
│   ├── 2.2 Q2 Results
│   ├── 2.3 Q3 Results ← LLM navigates here
│   │   ├── Revenue Drivers  ← and finds this
│   │   └── Cost Analysis
│   └── 2.4 Q4 Outlook
└── Section 3: Risk Factors
```

The retrieval process:

1. **Index phase:** Parse the document into a tree of sections, each with a title, summary, and page range
2. **Query phase:** An LLM navigates the tree top-down, reading summaries to decide which branches to explore — like a human expert scanning a table of contents

No embeddings. No vector database. No chunking. The LLM reasons about document structure to find relevant sections.

### When Tree Indexing Wins

| Scenario | Vector RAG | Tree Indexing |
|----------|-----------|--------------|
| Short FAQ documents | Better (simple, fast) | Overkill |
| Long structured reports (financial, legal, technical) | Chunks lose structure | Better — preserves hierarchy |
| Queries requiring cross-section reasoning | Misses connections | Follows document logic |
| Heterogeneous document sets | Better — embeddings generalize | Harder — needs consistent structure |
| Exact fact lookup ("what was Q3 revenue?") | Depends on chunk boundaries | Navigates directly to section |
| Semantic/conceptual queries | Better — similarity works well | May over-navigate |

> **Beginner Note:** Tree indexing requires documents with clear structure (headings, sections, numbered parts). For unstructured text like chat logs or social media posts, vector RAG is still the better approach.

> **Advanced Note:** Projects like PageIndex (github.com/VectifyAI/PageIndex) implement this approach and have achieved 98.7% accuracy on FinanceBench, outperforming vector-based RAG systems on structured financial documents. The trade-off is that tree indexing requires an LLM call during retrieval (to navigate the tree), making it more expensive per query than a vector lookup. For high-value queries on complex documents, this is worthwhile.

---

## Section 6: Building a Tree Index

### Index Structure

A tree index node contains:

```typescript
interface TreeNode {
  id: string
  title: string
  summary: string
  level: number // 0 = root, 1 = chapter, 2 = section, etc.
  pageStart: number
  pageEnd: number
  children: TreeNode[]
  content?: string // leaf nodes may store the actual text
}

interface TreeIndex {
  root: TreeNode
  documentTitle: string
  totalPages: number
}
```

Each non-leaf node has a summary generated by the LLM. The summary is what the retrieval LLM reads when deciding which branch to explore — it needs to be informative enough to guide navigation without being so long that it defeats the purpose of the tree.

### Building the Index

The index construction process:

1. **Parse structure:** Extract headings, sections, and page boundaries from the document
2. **Build hierarchy:** Organize sections into a tree based on heading levels (H1 → H2 → H3)
3. **Generate summaries:** For each node, use the LLM to summarize what that section covers
4. **Assign metadata:** Page ranges, section IDs, content length

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function summarizeSection(title: string, content: string): Promise<string> {
  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    system: 'Summarize what this document section covers in 1-2 sentences. Focus on the topics and key information a reader would find here. Be specific.',
    prompt: `Section: ${title}\n\n${content}`,
    maxOutputTokens: 100,
  })
  return text
}

function parseMarkdownToTree(markdown: string): TreeNode {
  const lines = markdown.split('\n')
  const root: TreeNode = {
    id: 'root',
    title: 'Document',
    summary: '',
    level: 0,
    pageStart: 0,
    pageEnd: 0,
    children: [],
  }

  // Implementation: parse heading levels, group content under headings,
  // build parent-child relationships based on heading depth
  // This is document-format specific — Markdown, PDF, HTML each need different parsers

  return root
}
```

### Summary Quality Matters

The quality of node summaries directly determines retrieval quality. A vague summary ("This section discusses various topics") forces the LLM to explore that branch speculatively. A specific summary ("This section analyzes Q3 2024 revenue by product line, showing 23% growth driven by the Enterprise tier") lets the LLM make confident navigation decisions.

```typescript
// Bad: too vague — LLM can't decide if this branch is relevant
{ summary: "Discusses company performance and metrics." }

// Good: specific — LLM knows exactly what's here
{ summary: "Q3 2024 revenue breakdown: $12M total, Enterprise tier grew 23%, SMB flat. Key driver was the DataFlow product launch in August." }
```

> **Beginner Note:** Building a tree index is an upfront cost — you run LLM calls during indexing, not just during querying. For a 100-page document, you might make 20-50 summarization calls. This is a one-time cost that amortizes over many queries.

> **Advanced Note:** The tree index can be persisted as JSON and reloaded without re-indexing. When the source document changes, you can do incremental updates — only re-summarize the sections that changed. This is analogous to re-embedding changed chunks in vector RAG, but cheaper since summarization is faster than embedding + indexing.

---

## Section 7: LLM-Navigated Tree Search

### The Retrieval Algorithm

Given a query and a tree index, the LLM navigates top-down:

1. Start at the root — read summaries of all top-level children
2. Ask the LLM: "Which of these sections is most likely to contain the answer?"
3. Descend into the chosen branch — read summaries of its children
4. Repeat until reaching a leaf node (actual content)
5. Return the leaf content as the retrieved context

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const NavigationSchema = z.object({
  selectedNodeId: z.string().describe('ID of the most relevant child node'),
  confidence: z.number().min(0).max(1).describe('How confident the selection is'),
  reasoning: z.string().describe('Why this node was selected'),
})

async function navigateTree(
  query: string,
  node: TreeNode,
  model = mistral('mistral-small-latest')
): Promise<TreeNode[]> {
  // Base case: leaf node — return it
  if (node.children.length === 0) {
    return [node]
  }

  // Present children summaries to the LLM
  const childDescriptions = node.children
    .map(c => `[${c.id}] ${c.title}: ${c.summary}`)
    .join('\n')

  const { output } = await generateText({
    model,
    output: Output.object({ schema: NavigationSchema }),
    system: 'You are a document navigator. Given a query and a list of document sections with summaries, select the section most likely to contain the answer.',
    prompt: `Query: ${query}\n\nAvailable sections:\n${childDescriptions}`,
  })

  const selected = node.children.find(c => c.id === output.selectedNodeId)
  if (!selected) return []

  // Recurse into the selected branch
  return navigateTree(query, selected, model)
}
```

### Multi-Path Navigation

Sometimes the answer spans multiple sections. Instead of picking one branch, explore the top-N most promising:

```typescript
const MultiNavigationSchema = z.object({
  selectedNodes: z.array(z.object({
    nodeId: z.string(),
    relevanceScore: z.number().min(0).max(1),
  })).describe('Top nodes ranked by relevance to the query'),
})

async function multiPathNavigate(
  query: string,
  node: TreeNode,
  maxPaths: number = 2,
  model = mistral('mistral-small-latest')
): Promise<TreeNode[]> {
  if (node.children.length === 0) return [node]

  const childDescriptions = node.children
    .map(c => `[${c.id}] ${c.title}: ${c.summary}`)
    .join('\n')

  const { output } = await generateText({
    model,
    output: Output.object({ schema: MultiNavigationSchema }),
    system: `You are a document navigator. Select up to ${maxPaths} sections most relevant to the query. Rank by relevance.`,
    prompt: `Query: ${query}\n\nAvailable sections:\n${childDescriptions}`,
  })

  const results: TreeNode[] = []
  for (const { nodeId } of output.selectedNodes.slice(0, maxPaths)) {
    const child = node.children.find(c => c.id === nodeId)
    if (child) {
      const leaves = await multiPathNavigate(query, child, maxPaths, model)
      results.push(...leaves)
    }
  }

  return results
}
```

### Comparing Retrieval Approaches

The three retrieval paradigms each have different strengths:

| Approach | How it finds relevant content | Strengths | Weaknesses |
|----------|------------------------------|-----------|------------|
| **Vector RAG** | Embed query → cosine similarity → top-K chunks | Fast, works on unstructured text, no LLM during retrieval | Similarity ≠ relevance, loses document structure |
| **Tree Indexing** | LLM navigates document hierarchy top-down | Preserves structure, high precision on structured docs | Requires structured documents, LLM cost during retrieval |
| **Graph RAG** | Traverse entity/relationship graph from query entities | Captures relationships, multi-hop reasoning | Complex setup, entity extraction quality critical |

In practice, these can be combined. Use vector RAG for broad recall, tree indexing for structured document deep-dives, and graph RAG for relationship-heavy queries.

> **Beginner Note:** Start with vector RAG (Module 9). Add tree indexing when you work with long structured documents where vector search misses context. Add graph RAG (Module 12) when relationships between entities matter more than the text itself.

> **Advanced Note:** A production system might route queries to different retrieval backends: simple factual queries go to vector search, analytical queries about structured reports go to tree search, and relationship queries go to graph search. The routing itself can be done by an LLM (see Module 14: Agent Fundamentals).

---

## Section 8: RAG Assessment Framework

### Why You Need Systematic Measurement

The techniques in this module — HyDE, hybrid search, reranking, tree indexing — each add complexity and cost. How do you know they actually improve your pipeline? How do you decide which combination to use?

You need a systematic assessment framework that measures multiple dimensions of RAG quality. Without it, you are tuning by vibes and cherry-picked examples.

### Assessment Dimensions

RAG quality measurement covers four key dimensions:

| Dimension             | Question                                              | What It Catches     |
| --------------------- | ----------------------------------------------------- | ------------------- |
| **Context Relevance** | Are the retrieved chunks relevant to the query?       | Bad retrieval       |
| **Context Recall**    | Do the retrieved chunks cover all needed information? | Missing context     |
| **Faithfulness**      | Is the answer grounded in the retrieved context?      | Hallucination       |
| **Answer Relevance**  | Does the answer actually address the query?           | Off-topic responses |

### Building an Assessment Suite

```typescript
// src/advanced-rag/rag-assessment.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Test case definition
interface RAGTestCase {
  query: string
  expectedAnswer: string // Ground truth or reference answer
  expectedSources?: string[] // Expected source documents
}

// Result for a single test case
interface RAGAssessmentResult {
  query: string
  answer: string
  retrievedChunks: string[]
  metrics: {
    contextRelevance: number // 0-1
    faithfulness: number // 0-1
    answerRelevance: number // 0-1
    answerCorrectness: number // 0-1
  }
  details: {
    contextRelevanceReasoning: string
    faithfulnessReasoning: string
    answerRelevanceReasoning: string
    correctnessReasoning: string
  }
}

// Metric 1: Context Relevance
const ContextRelevanceSchema = z.object({
  relevantChunks: z.number().describe('Number of chunks relevant to the query'),
  totalChunks: z.number().describe('Total number of retrieved chunks'),
  score: z.number().min(0).max(1).describe('Fraction of chunks that are relevant'),
  reasoning: z.string(),
})

async function measureContextRelevance(query: string, chunks: string[]): Promise<{ score: number; reasoning: string }> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ContextRelevanceSchema }),
    system: `Determine how many of the retrieved chunks are relevant to the
query. A chunk is relevant if it contains information that helps answer
the query. Score = relevant chunks / total chunks.`,
    messages: [
      {
        role: 'user',
        content: `Query: ${query}\n\nRetrieved chunks:\n${chunks.map((c, i) => `[${i}]: ${c}`).join('\n\n')}`,
      },
    ],
    temperature: 0,
  })

  return { score: output.score, reasoning: output.reasoning }
}

// Metric 2: Faithfulness
const FaithfulnessSchema = z.object({
  claims: z.array(
    z.object({
      claim: z.string().describe('A factual claim from the answer'),
      supportedByContext: z.boolean(),
      evidence: z.string().optional().describe('Context excerpt supporting this claim'),
    })
  ),
  score: z.number().min(0).max(1).describe('Fraction of claims supported by context'),
  reasoning: z.string(),
})

async function measureFaithfulness(answer: string, context: string): Promise<{ score: number; reasoning: string }> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: FaithfulnessSchema }),
    system: `Extract all factual claims from the answer and check if each
one is supported by the provided context. A claim is "supported" if the
context contains evidence for it. Score = supported claims / total claims.

Be thorough: extract every specific fact, number, date, and assertion.`,
    messages: [
      {
        role: 'user',
        content: `Context: ${context}\n\nAnswer: ${answer}`,
      },
    ],
    temperature: 0,
  })

  return { score: output.score, reasoning: output.reasoning }
}

// Metric 3: Answer Relevance
const AnswerRelevanceSchema = z.object({
  score: z.number().min(0).max(1).describe('How well the answer addresses the query (0=irrelevant, 1=perfect)'),
  reasoning: z.string(),
})

async function measureAnswerRelevance(query: string, answer: string): Promise<{ score: number; reasoning: string }> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: AnswerRelevanceSchema }),
    system: `Rate how well the answer addresses the query.
1.0 = perfectly and completely answers the question
0.7-0.9 = mostly answers it with minor gaps
0.4-0.6 = partially answers it
0.1-0.3 = barely related
0.0 = completely irrelevant`,
    messages: [
      {
        role: 'user',
        content: `Query: ${query}\n\nAnswer: ${answer}`,
      },
    ],
    temperature: 0,
  })

  return { score: output.score, reasoning: output.reasoning }
}

// Metric 4: Answer Correctness (requires ground truth)
const CorrectnessSchema = z.object({
  score: z.number().min(0).max(1).describe('How correct the answer is compared to the reference'),
  reasoning: z.string(),
})

async function measureCorrectness(
  answer: string,
  referenceAnswer: string
): Promise<{ score: number; reasoning: string }> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: CorrectnessSchema }),
    system: `Compare the generated answer to the reference answer.
Score based on factual accuracy, not wording:
1.0 = all facts match the reference
0.7-0.9 = mostly correct with minor discrepancies
0.4-0.6 = partially correct
0.1-0.3 = mostly incorrect
0.0 = completely wrong or contradicts the reference`,
    messages: [
      {
        role: 'user',
        content: `Reference answer: ${referenceAnswer}\n\nGenerated answer: ${answer}`,
      },
    ],
    temperature: 0,
  })

  return { score: output.score, reasoning: output.reasoning }
}

// Full assessment runner
async function assessRAGPipeline(
  testCases: RAGTestCase[],
  ragPipeline: (query: string) => Promise<{ answer: string; chunks: string[] }>
): Promise<{
  results: RAGAssessmentResult[]
  aggregate: {
    avgContextRelevance: number
    avgFaithfulness: number
    avgAnswerRelevance: number
    avgCorrectness: number
  }
}> {
  const results: RAGAssessmentResult[] = []

  for (const testCase of testCases) {
    console.log(`\nAssessing: "${testCase.query}"`)

    // Run the pipeline
    const { answer, chunks } = await ragPipeline(testCase.query)
    const context = chunks.join('\n\n')

    // Measure all dimensions
    const [contextRel, faithfulness, answerRel, correctness] = await Promise.all([
      measureContextRelevance(testCase.query, chunks),
      measureFaithfulness(answer, context),
      measureAnswerRelevance(testCase.query, answer),
      measureCorrectness(answer, testCase.expectedAnswer),
    ])

    results.push({
      query: testCase.query,
      answer,
      retrievedChunks: chunks,
      metrics: {
        contextRelevance: contextRel.score,
        faithfulness: faithfulness.score,
        answerRelevance: answerRel.score,
        answerCorrectness: correctness.score,
      },
      details: {
        contextRelevanceReasoning: contextRel.reasoning,
        faithfulnessReasoning: faithfulness.reasoning,
        answerRelevanceReasoning: answerRel.reasoning,
        correctnessReasoning: correctness.reasoning,
      },
    })

    console.log(
      `  Context relevance: ${contextRel.score.toFixed(2)} | ` +
        `Faithfulness: ${faithfulness.score.toFixed(2)} | ` +
        `Answer relevance: ${answerRel.score.toFixed(2)} | ` +
        `Correctness: ${correctness.score.toFixed(2)}`
    )
  }

  // Aggregate metrics
  const avg = (arr: number[]) => arr.reduce((s, v) => s + v, 0) / arr.length

  const aggregate = {
    avgContextRelevance: avg(results.map(r => r.metrics.contextRelevance)),
    avgFaithfulness: avg(results.map(r => r.metrics.faithfulness)),
    avgAnswerRelevance: avg(results.map(r => r.metrics.answerRelevance)),
    avgCorrectness: avg(results.map(r => r.metrics.answerCorrectness)),
  }

  console.log('\n=== Aggregate Metrics ===')
  console.log(`Context Relevance: ${aggregate.avgContextRelevance.toFixed(3)}`)
  console.log(`Faithfulness:      ${aggregate.avgFaithfulness.toFixed(3)}`)
  console.log(`Answer Relevance:  ${aggregate.avgAnswerRelevance.toFixed(3)}`)
  console.log(`Correctness:       ${aggregate.avgCorrectness.toFixed(3)}`)

  return { results, aggregate }
}

export {
  measureContextRelevance,
  measureFaithfulness,
  measureAnswerRelevance,
  measureCorrectness,
  assessRAGPipeline,
  type RAGTestCase,
  type RAGAssessmentResult,
}
```

### Building a Test Suite

```typescript
// src/advanced-rag/assessment-test-suite.ts

import { assessRAGPipeline, type RAGTestCase } from './rag-assessment.js'

// Define your test cases — these encode your ground truth
const testSuite: RAGTestCase[] = [
  {
    query: "What is the company's refund policy?",
    expectedAnswer: 'Full refunds within 30 days of purchase. After 30 days, store credit is issued.',
    expectedSources: ['refund-policy.md'],
  },
  {
    query: 'What are the API rate limits?',
    expectedAnswer: '100 requests per minute per API key. Enterprise plans have higher limits.',
    expectedSources: ['api-docs/rate-limits.md'],
  },
  {
    query: 'How do I reset my password?',
    expectedAnswer:
      "Go to Settings > Security > Change Password. You can also use the 'Forgot Password' link on the login page.",
    expectedSources: ['help/account-security.md'],
  },
  {
    query: 'Compare Q1 and Q2 revenue',
    expectedAnswer: 'Q1 revenue was $12.3M, Q2 revenue was $15.1M, representing 22.8% growth.',
    expectedSources: ['reports/q1-2024.md', 'reports/q2-2024.md'],
  },
  {
    query: 'What programming languages does the API support?',
    expectedAnswer:
      'Official SDKs are available for Python, TypeScript, Go, and Ruby. Community libraries exist for Java and C#.',
    expectedSources: ['api-docs/sdks.md'],
  },
]

// Compare two pipeline configurations
async function comparePipelines(): Promise<void> {
  // Pipeline A: naive retrieval
  const naiveResults = await assessRAGPipeline(testSuite, async query => {
    // Your naive RAG pipeline from Module 9
    const chunks = await naiveSearch(query, 5)
    const answer = await generateAnswer(query, chunks)
    return { answer, chunks }
  })

  // Pipeline B: advanced retrieval with reranking
  const advancedResults = await assessRAGPipeline(testSuite, async query => {
    // Advanced pipeline with hybrid search + reranking
    const candidates = await runHybridSearch(query, 20)
    const reranked = await rerankDocs(query, candidates, 5)
    const answer = await generateAnswer(query, reranked)
    return { answer, chunks: reranked }
  })

  // Compare
  console.log('\n=== Pipeline Comparison ===')
  console.log('Metric               | Naive  | Advanced | Delta')
  console.log('---------------------|--------|----------|------')

  const metricKeys = ['avgContextRelevance', 'avgFaithfulness', 'avgAnswerRelevance', 'avgCorrectness'] as const

  for (const metric of metricKeys) {
    const naive = naiveResults.aggregate[metric]
    const advanced = advancedResults.aggregate[metric]
    const delta = advanced - naive
    const arrow = delta > 0 ? '+' : ''
    console.log(`${metric.padEnd(21)}| ${naive.toFixed(3)}  | ${advanced.toFixed(3)}    | ${arrow}${delta.toFixed(3)}`)
  }
}

// Placeholder functions — replace with your actual implementations
async function naiveSearch(query: string, topK: number): Promise<string[]> {
  return [] // Your Module 9 search
}
async function runHybridSearch(query: string, topK: number): Promise<string[]> {
  return [] // Your hybrid search
}
async function rerankDocs(query: string, docs: string[], topK: number): Promise<string[]> {
  return [] // Your reranker
}
async function generateAnswer(query: string, chunks: string[]): Promise<string> {
  return '' // Your generation step
}

export { testSuite, comparePipelines }
```

> **Beginner Note:** Start with 10-20 manually curated test cases that cover your most important query types. You do not need hundreds of test cases to get useful signal. Focus on queries you know your pipeline struggles with.

> **Advanced Note:** LLM-as-judge measurement (what we use above) is itself noisy. For high-confidence results, use multiple judge calls per test case and average the scores. Also consider using a different model as judge than the one generating answers — this reduces bias. Module 19 covers testing in much more depth.

---

## Quiz

### Question 1 (Easy)

What is the primary advantage of HyDE over standard query embedding?

- A) HyDE uses a more powerful embedding model
- B) HyDE generates a hypothetical document that is structurally similar to corpus documents, producing a better embedding for retrieval
- C) HyDE reduces the number of LLM calls needed
- D) HyDE eliminates the need for a vector database

**Answer: B** — HyDE works because queries and documents have different structures. A query is short and interrogative; a document is long and declarative. By generating a hypothetical answer (a declarative passage), HyDE produces an embedding that is closer in vector space to the actual relevant documents. The hypothetical answer does not need to be factually correct — it just needs to "look like" the right kind of document.

---

### Question 2 (Medium)

When implementing hybrid search, why is score normalization important before combining semantic and keyword scores?

- A) Normalization makes the search faster
- B) Semantic search returns cosine similarity (0-1) while BM25 returns unbounded scores — without normalization, one signal would dominate the combined score
- C) Normalization reduces the number of results returned
- D) Normalization is only needed for Reciprocal Rank Fusion

**Answer: B** — Cosine similarity scores range from 0 to 1, while BM25 scores are unbounded and depend on corpus statistics. If you add them directly, the BM25 scores (which can be much larger) would dominate the combined score, effectively ignoring the semantic signal. Min-max normalization or z-score normalization brings both to a comparable range. Reciprocal Rank Fusion (RRF) avoids this problem entirely by working with ranks instead of scores.

---

### Question 3 (Medium)

In a retrieve-then-rerank pipeline, why do you retrieve 20-50 candidates but only rerank to 3-5?

- A) Reranking is expensive (LLM calls or cross-encoder inference per candidate), so you want to minimize the number of candidates scored
- B) More than 5 documents always cause hallucination
- C) Vector databases can only return 50 results maximum
- D) The LLM context window can only fit 3-5 documents

**Answer: A** — The two-stage design exists because retrieval (approximate nearest neighbor search) is fast and cheap, while reranking (cross-encoder or LLM scoring) is slow and expensive. You retrieve broadly to maximize recall (making sure the relevant documents are in the candidate set), then rerank precisely to maximize precision (keeping only the truly relevant ones). The specific numbers depend on your reranker quality and cost budget.

---

### Question 4 (Easy)

What does Self-RAG's retrieval decision step prevent that standard RAG does not?

- A) It prevents the model from hallucinating
- B) It prevents unnecessary retrieval for queries that do not need external knowledge, avoiding irrelevant context injection
- C) It prevents the model from generating long responses
- D) It prevents the vector database from being overloaded

**Answer: B** — Standard RAG always retrieves, even for simple questions like "What is 2+2?" where retrieval adds noise. Self-RAG's first step is a classification: does this query need external knowledge? If not, the model answers directly from its parametric knowledge, avoiding the cost and noise of irrelevant retrieval. This is especially valuable when the knowledge base does not cover the query topic — standard RAG would inject random, misleading chunks.

---

### Question 5 (Hard)

You run an assessment suite and find: context relevance = 0.9, faithfulness = 0.4, answer relevance = 0.8. What is the most likely issue with your RAG pipeline?

- A) The retrieval stage is returning wrong documents
- B) The LLM is hallucinating — generating claims not supported by the retrieved context, despite the context being relevant
- C) The query transformation is too aggressive
- D) The chunking strategy is too coarse

**Answer: B** — High context relevance (0.9) means retrieval is working well — the right chunks are being found. High answer relevance (0.8) means the answer addresses the query. But low faithfulness (0.4) means the model is making claims that are not supported by the context — it is filling in gaps with hallucinated information. The fix is better prompting (stricter instructions to only use context), contextual compression (focus the context), or switching to a model that is more instruction-following.

---

## Exercises

### Exercise 1: Upgrade Module 9 RAG with HyDE + Reranking

**Objective:** Take the naive RAG pipeline from Module 9 and add HyDE retrieval and LLM-based reranking. Measure the improvement with the assessment framework.

**Specification:**

1. Create `src/exercises/m10/ex01-advanced-rag.ts`
2. Implement three pipeline variants:
   - **Naive:** Direct query embedding + top-5 retrieval (from Module 9)
   - **HyDE:** Generate hypothetical document, embed it, top-10 retrieval
   - **HyDE + Rerank:** Generate hypothetical document, embed it, top-20 retrieval, LLM rerank to top-5
3. Create a test suite of at least 5 query/expected-answer pairs relevant to your document corpus
4. Run the assessment framework against all three pipeline variants
5. Print a comparison table showing all four metrics for each variant

**Expected output format:**

```
=== RAG Pipeline Comparison ===

Pipeline          | CtxRel | Faith | AnsRel | Correct
------------------|--------|-------|--------|---------
Naive             | 0.620  | 0.540 | 0.700  | 0.580
HyDE              | 0.780  | 0.600 | 0.760  | 0.680
HyDE + Rerank     | 0.880  | 0.820 | 0.860  | 0.800

Improvement (Naive -> HyDE+Rerank):
  Context Relevance: +0.260 (+41.9%)
  Faithfulness:      +0.280 (+51.9%)
  Answer Relevance:  +0.160 (+22.9%)
  Correctness:       +0.220 (+37.9%)
```

**Test specification:**

```typescript
// tests/exercises/m10/ex01-advanced-rag.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 10: Advanced RAG', () => {
  it('should implement all three pipeline variants', async () => {
    const naive = await runNaivePipeline(testQuery)
    const hyde = await runHyDEPipeline(testQuery)
    const hydeRerank = await runHyDERerankPipeline(testQuery)

    expect(naive.answer).toBeTruthy()
    expect(hyde.answer).toBeTruthy()
    expect(hydeRerank.answer).toBeTruthy()
  })

  it('should run assessment on all variants', async () => {
    const results = await compareAllPipelines(testSuite)
    expect(results).toHaveLength(3)

    for (const result of results) {
      expect(result.aggregate.avgContextRelevance).toBeGreaterThanOrEqual(0)
      expect(result.aggregate.avgFaithfulness).toBeGreaterThanOrEqual(0)
    }
  })

  it('HyDE+Rerank should outperform Naive on average', async () => {
    const results = await compareAllPipelines(testSuite)
    const naive = results.find(r => r.pipeline === 'naive')!
    const advanced = results.find(r => r.pipeline === 'hyde+rerank')!

    // Advanced should generally score higher
    expect(advanced.aggregate.avgContextRelevance).toBeGreaterThanOrEqual(
      naive.aggregate.avgContextRelevance * 0.9 // Allow some variance
    )
  })
})
```

---

### Exercise 2: Hybrid Search with Assessment

**Objective:** Implement hybrid search combining semantic embeddings with BM25 keyword search, and measure when each approach excels.

**Specification:**

1. Create `src/exercises/m10/ex02-hybrid-search.ts`
2. Implement the BM25 index and scoring from this module
3. Implement hybrid search with configurable semantic/keyword weighting
4. Create a test suite with two categories of queries:
   - **Semantic queries** (e.g., "How do I fix a broken car?") where meaning matters
   - **Keyword queries** (e.g., "ERR_0x4F2A troubleshooting") where exact matches matter
5. Run the assessment with three configurations:
   - Semantic only (weight: 1.0/0.0)
   - Keyword only (weight: 0.0/1.0)
   - Hybrid (weight: 0.7/0.3)
6. Show that hybrid outperforms either individual approach across both query types

**Test specification:**

```typescript
// tests/exercises/m10/ex02-hybrid-search.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 10: Hybrid Search', () => {
  it('should find exact-match documents with keyword search', async () => {
    const results = await keywordSearch('ERR_0x4F2A', testDocuments)
    expect(results[0].document).toContain('ERR_0x4F2A')
  })

  it('should find semantically similar documents with embedding search', async () => {
    const results = await semanticSearch('fix a broken car', testDocuments)
    expect(results[0].document).toContain('automobile repair')
  })

  it('hybrid search should perform well on both query types', async () => {
    const hybridResults = await assessHybridSearch(mixedTestSuite, {
      semanticWeight: 0.7,
      topK: 5,
    })
    expect(hybridResults.avgContextRelevance).toBeGreaterThan(0.6)
  })
})
```

> **Advanced Technique: Contextual Retrieval** — A technique (originally published by Anthropic) that dramatically improves retrieval quality: before embedding each chunk, prepend a short context summary explaining where the chunk sits in the original document. For example, a chunk about "Q3 revenue" gets prefixed with "This chunk is from Acme Corp's 2025 Annual Report, specifically the Financial Results section." This gives the embedding model crucial context that's lost during chunking. You can generate these context prefixes with a cheap, fast model (`mistral('mistral-small-latest')` or `groq('openai/gpt-oss-20b')`) at ingestion time. Combined with the hybrid search from this module, contextual retrieval can reduce retrieval failures by up to 67%.

> **Local Alternative (Ollama):** Advanced RAG techniques (HyDE, query decomposition, self-RAG) work with `ollama('qwen3.5')` — they're prompt-based strategies, not provider features. LLM-based reranking also works locally, though it will be slower than API reranking services. For hybrid search, the BM25 + semantic combination is fully local.

---

## Summary

In this module, you learned:

1. **Naive RAG failures:** Wrong chunks, missed context, and hallucination are predictable failure modes with specific solutions.
2. **Query transformation and HyDE:** Rewriting, expansion, decomposition, and hypothetical document embeddings improve retrieval by transforming user queries into better search queries — or into hypothetical answers that are structurally similar to corpus documents.
3. **Hybrid search:** Combining semantic (embedding) and keyword (BM25) search gives you meaning-understanding with exact-match precision.
4. **Reranking:** A second pass with a cross-encoder or LLM dramatically improves precision by scoring each candidate carefully.
5. **Structure-aware retrieval:** Vector similarity does not equal relevance — tree indexing preserves document structure as a navigable hierarchy for more precise retrieval on structured documents.
6. **Building a tree index:** Parsing documents into tree nodes with LLM-generated summaries creates a structure that supports top-down navigation without embeddings or vector databases.
7. **LLM-navigated tree search:** An LLM can navigate a tree index top-down, reading summaries to find the most relevant sections — an alternative to vector search for structured documents.
8. **Assessment framework:** Systematic measurement of context relevance, faithfulness, answer relevance, and correctness lets you compare pipeline configurations objectively.

In Module 11, you will tackle the other side of the RAG pipeline — document processing. Better ingestion, chunking, and metadata extraction feed directly into the retrieval quality improvements you built here.
