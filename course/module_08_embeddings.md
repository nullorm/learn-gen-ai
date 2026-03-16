# Module 8: Embeddings & Similarity

## Learning Objectives

- Understand what embeddings are and how they capture semantic meaning as vectors
- Use the Vercel AI SDK's `embed` and `embedMany` functions across different providers
- Implement cosine similarity from scratch and understand why it works
- Set up ChromaDB as a vector store for semantic search
- Build semantic search systems that find relevant documents by meaning, not keywords
- Evaluate embedding dimensions, trade-offs, and similarity thresholds

---

## Why Should I Care?

Keywords fail. A user searching for "how to fix a broken pipe" might mean a plumbing emergency or a Unix signal handling problem. Traditional search matches the words but misses the meaning. Embeddings solve this by converting text into numerical vectors that capture semantic meaning — texts with similar meanings have similar vectors, regardless of the specific words used.

Embeddings are the foundation of retrieval-augmented generation (RAG), recommendation systems, clustering, classification, and anomaly detection. Every production LLM application that works with a knowledge base — customer support bots, documentation search, code assistants — relies on embeddings to find relevant information before sending it to the model.

Understanding embeddings gives you a mental model for how LLMs "understand" text. It also gives you practical tools for building search systems that work with meaning rather than pattern matching. This module teaches both the theory and the implementation.

---

## Connection to Other Modules

- **Module 1 (Setup)** configured providers. The same providers serve both chat and embedding models.
- **Module 5 (Long Context)** discussed when to use RAG vs full context. Embeddings are half of the RAG equation.
- **Module 9 (RAG)** builds directly on this module, using embeddings to retrieve context for generation.
- **Module 3 (Structured Output)** showed Zod schemas. Embedding metadata often uses structured formats.

---

## Section 1: What are Embeddings?

### Text to Vector

An embedding is a function that converts text into a fixed-length array of numbers (a vector). The key property: texts with similar meanings produce similar vectors.

```
"I love programming"      → [0.12, -0.34, 0.56, ..., 0.78]  (1536 dimensions)
"Coding is my passion"    → [0.11, -0.32, 0.55, ..., 0.77]  (similar!)
"The weather is nice"     → [0.89, 0.23, -0.45, ..., -0.12] (different!)
```

### Why Vectors?

Vectors enable mathematical operations on meaning:

- **Similarity**: How close are two texts in meaning? (cosine similarity)
- **Search**: Find the documents most similar to a query
- **Clustering**: Group similar texts together
- **Classification**: Determine which category a text belongs to
- **Anomaly detection**: Find texts that are unlike the rest

### Embedding Models vs Chat Models

Embedding models are separate from chat models. They are smaller, faster, and cheaper because they only need to encode text, not generate it:

| Feature | Chat Model             | Embedding Model          |
| ------- | ---------------------- | ------------------------ |
| Purpose | Generate text          | Encode text as vectors   |
| Output  | Text (variable length) | Vector (fixed length)    |
| Cost    | $3-15 per 1M tokens    | $0.02-0.13 per 1M tokens |
| Speed   | 50-100 tokens/sec      | 1000+ tokens/sec         |
| Size    | 70B-400B parameters    | 100M-1B parameters       |

> **Beginner Note:** Think of embeddings as a "fingerprint" for text. Just as fingerprints let you identify people without seeing them, embeddings let you identify meaning without reading the full text. Similar meanings produce similar fingerprints.

---

## Section 2: Embedding Models

### Available Models

Different providers offer different embedding models:

| Provider  | Model                  | Dimensions | Max Tokens | Cost per 1M      |
| --------- | ---------------------- | ---------- | ---------- | ---------------- |
| Mistral   | mistral-embed          | 1024       | 8192       | Free (free tier) |
| OpenAI    | text-embedding-3-small | 1536       | 8191       | $0.02            |
| OpenAI    | text-embedding-3-large | 3072       | 8191       | $0.13            |
| Anthropic | voyage-3 (via Voyage)  | 1024       | 32000      | $0.06            |
| Ollama    | qwen3-embedding:0.6b   | up to 4096 | 32768      | Free (local)     |
| Ollama    | qwen3-embedding:8b     | up to 4096 | 40960      | Free (local)     |

> **Note:** Groq does not offer embedding models. For embeddings, use Mistral (free API), OpenAI (paid), or Ollama (free local).

### Provider Setup

```typescript
// Mistral embeddings (free tier, recommended default)
import { mistral } from '@ai-sdk/mistral'

const embeddingModel = mistral.textEmbeddingModel('mistral-embed')

// OpenAI embeddings
import { openai } from '@ai-sdk/openai'

const openaiEmbeddingModel = openai.embedding('text-embedding-3-small')

// Ollama embeddings (free, local)
import { ollama } from 'ai-sdk-ollama'

const localEmbeddingModel = ollama.embedding('qwen3-embedding:0.6b')
```

> **Beginner Note:** For this course, we primarily use OpenAI's embedding models because they offer the best balance of quality, speed, and cost. Anthropic does not offer a native embedding model through the AI SDK, but you can use OpenAI embeddings alongside Anthropic chat models — they are independent services.

> **Advanced Note:** Embedding model choice matters. Different models capture different aspects of meaning and perform differently on various tasks. For production systems, benchmark your specific use case with multiple models. The difference between a good and bad embedding model can be a 20-30% accuracy gap in retrieval tasks.

---

## Section 3: Vercel AI SDK embed/embedMany

### Single Embedding

The `embed` function converts a single text string into a vector:

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

const { embedding, usage } = await embed({
  model: openai.embedding('text-embedding-3-small'),
  value: 'TypeScript is a typed superset of JavaScript.',
})

console.log(`Vector dimensions: ${embedding.length}`)
// 1536 for text-embedding-3-small

console.log(`First 5 values: ${embedding.slice(0, 5)}`)
// [-0.023, 0.041, -0.019, 0.055, -0.032]

console.log(`Tokens used: ${usage?.tokens}`)
```

### Understanding the Output

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

const { embedding } = await embed({
  model: openai.embedding('text-embedding-3-small'),
  value: 'Hello, world!',
})

// The embedding is a Float64Array or number[]
console.log(`Type: ${typeof embedding}`)
console.log(`Length: ${embedding.length}`) // 1536

// Values are typically between -1 and 1
const min = Math.min(...embedding)
const max = Math.max(...embedding)
const avg = embedding.reduce((a, b) => a + b, 0) / embedding.length

console.log(`Min: ${min.toFixed(4)}, Max: ${max.toFixed(4)}, Avg: ${avg.toFixed(4)}`)

// The vector is normalized (unit length)
const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0))
console.log(`Magnitude: ${magnitude.toFixed(4)}`) // Close to 1.0
```

### Embedding Different Content Types

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

// Embeddings work for any text content
const examples = [
  'What is machine learning?', // Question
  'Machine learning is a branch of AI.', // Statement
  'function add(a: number, b: number) { return a + b; }', // Code
  'ERROR: Connection refused on port 3000', // Log message
  '{ "name": "Alice", "age": 30 }', // JSON
]

for (const text of examples) {
  const { embedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: text,
  })
  console.log(`"${text.slice(0, 40)}..." → [${embedding.slice(0, 3).map(v => v.toFixed(3))}...]`)
}
```

---

## Section 4: Cosine Similarity

### The Intuition

Cosine similarity measures the angle between two vectors. If two vectors point in the same direction, their cosine similarity is 1 (identical meaning). If they are perpendicular, it is 0 (unrelated). If they point in opposite directions, it is -1 (opposite meaning).

```
Similarity = cos(θ) = (A · B) / (|A| × |B|)

Where:
  A · B = sum of (a_i × b_i) for each dimension
  |A|   = sqrt(sum of a_i²)
  |B|   = sqrt(sum of b_i²)
```

### Implementation

```typescript
/**
 * Calculate cosine similarity between two vectors.
 * Returns a value between -1 (opposite) and 1 (identical).
 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`)
  }

  let dotProduct = 0
  let magnitudeA = 0
  let magnitudeB = 0

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    magnitudeA += a[i] * a[i]
    magnitudeB += b[i] * b[i]
  }

  magnitudeA = Math.sqrt(magnitudeA)
  magnitudeB = Math.sqrt(magnitudeB)

  if (magnitudeA === 0 || magnitudeB === 0) {
    return 0 // Avoid division by zero
  }

  return dotProduct / (magnitudeA * magnitudeB)
}

// Note: The Vercel AI SDK exports cosineSimilarity from 'ai'.
// We implement it manually here for learning purposes.
// In production code, use: import { cosineSimilarity } from 'ai'

// Test with simple vectors
const v1 = [1, 0, 0]
const v2 = [1, 0, 0] // Same direction
const v3 = [0, 1, 0] // Perpendicular
const v4 = [-1, 0, 0] // Opposite

console.log(`Same direction: ${cosineSimilarity(v1, v2)}`) // 1.0
console.log(`Perpendicular: ${cosineSimilarity(v1, v3)}`) // 0.0
console.log(`Opposite: ${cosineSimilarity(v1, v4)}`) // -1.0
```

### Comparing Real Embeddings

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0
  let magnitudeA = 0
  let magnitudeB = 0

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    magnitudeA += a[i] * a[i]
    magnitudeB += b[i] * b[i]
  }

  return dotProduct / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB))
}

async function getEmbedding(text: string): Promise<number[]> {
  const { embedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: text,
  })
  return embedding
}

// Compare semantic similarity
const texts = [
  'How do I sort an array in JavaScript?',
  'What is the best way to order elements in a JS array?',
  'JavaScript array sorting methods',
  'How to cook pasta carbonara',
  'What is the weather like today?',
]

const embeddings = await Promise.all(texts.map(getEmbedding))

// Compare the first text against all others
const query = texts[0]
console.log(`\nQuery: "${query}"\n`)

for (let i = 1; i < texts.length; i++) {
  const similarity = cosineSimilarity(embeddings[0], embeddings[i])
  console.log(`  "${texts[i]}"`)
  console.log(`  Similarity: ${similarity.toFixed(4)}\n`)
}

// Expected output:
// "What is the best way to order elements..." → ~0.85 (very similar)
// "JavaScript array sorting methods"          → ~0.80 (similar)
// "How to cook pasta carbonara"               → ~0.15 (unrelated)
// "What is the weather like today?"            → ~0.20 (unrelated)
```

> **Beginner Note:** Cosine similarity values for real embeddings typically range from 0.1 (unrelated) to 0.95+ (near identical). Values below 0.3 usually indicate no meaningful relationship. Values above 0.7 indicate strong semantic similarity. These thresholds vary by model.

### Other Distance Metrics

```typescript
/**
 * Euclidean distance between two vectors.
 * Lower values = more similar. Range: [0, ∞)
 */
function euclideanDistance(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2
  }
  return Math.sqrt(sum)
}

/**
 * Dot product of two vectors.
 * For normalized vectors, equivalent to cosine similarity.
 */
function dotProduct(a: number[], b: number[]): number {
  let sum = 0
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i]
  }
  return sum
}

// For normalized embeddings (like OpenAI's), cosine similarity
// and dot product give the same ranking. Cosine similarity is
// preferred because it is bounded between -1 and 1.
```

---

## Section 5: Vector Stores (ChromaDB)

### What is a Vector Store?

A vector store is a database optimized for storing and querying vectors. Instead of SQL's `WHERE` clause, you query by similarity: "find the 5 vectors most similar to this query vector."

### ChromaDB Setup

ChromaDB is an open-source embedding database that runs locally:

```bash
# Install ChromaDB client
bun add chromadb
```

```typescript
import { ChromaClient } from 'chromadb'

// Connect to ChromaDB (run `chroma run` in another terminal first)
const chroma = new ChromaClient()

// Create a collection (like a table)
const collection = await chroma.getOrCreateCollection({
  name: 'documents',
  metadata: { 'hnsw:space': 'cosine' }, // Use cosine similarity
})

console.log(`Collection: ${collection.name}`)
console.log(`Count: ${await collection.count()}`)
```

### Inserting Documents

```typescript
import { embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'
import { ChromaClient } from 'chromadb'

const chroma = new ChromaClient()
const collection = await chroma.getOrCreateCollection({ name: 'knowledge-base' })

// Documents to index
const documents = [
  {
    id: 'doc-1',
    text: 'TypeScript adds static typing to JavaScript, catching errors at compile time.',
    metadata: { source: 'typescript-docs', category: 'basics' },
  },
  {
    id: 'doc-2',
    text: 'React is a library for building user interfaces with a component-based architecture.',
    metadata: { source: 'react-docs', category: 'frontend' },
  },
  {
    id: 'doc-3',
    text: 'PostgreSQL is a powerful relational database known for reliability and data integrity.',
    metadata: { source: 'postgres-docs', category: 'database' },
  },
  {
    id: 'doc-4',
    text: 'Docker containers package applications with their dependencies for consistent deployment.',
    metadata: { source: 'docker-docs', category: 'devops' },
  },
  {
    id: 'doc-5',
    text: 'REST APIs use HTTP methods like GET, POST, PUT, DELETE to interact with resources.',
    metadata: { source: 'api-docs', category: 'backend' },
  },
]

// Generate embeddings for all documents
const { embeddings } = await embedMany({
  model: openai.embedding('text-embedding-3-small'),
  values: documents.map(d => d.text),
})

// Insert into ChromaDB
await collection.add({
  ids: documents.map(d => d.id),
  embeddings: embeddings as number[][],
  documents: documents.map(d => d.text),
  metadatas: documents.map(d => d.metadata),
})

console.log(`Inserted ${documents.length} documents`)
console.log(`Collection count: ${await collection.count()}`)
```

### Querying by Similarity

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'
import { ChromaClient } from 'chromadb'

const chroma = new ChromaClient()
const collection = await chroma.getCollection({ name: 'knowledge-base' })

async function semanticSearch(
  query: string,
  topK: number = 3
): Promise<Array<{ text: string; score: number; metadata: Record<string, unknown> }>> {
  // Embed the query
  const { embedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: query,
  })

  // Query ChromaDB
  const results = await collection.query({
    queryEmbeddings: [embedding as number[]],
    nResults: topK,
    include: ['documents', 'distances', 'metadatas'],
  })

  // Format results
  const formatted = []
  for (let i = 0; i < (results.documents?.[0]?.length ?? 0); i++) {
    formatted.push({
      text: results.documents?.[0]?.[i] ?? '',
      score: 1 - (results.distances?.[0]?.[i] ?? 0), // Convert distance to similarity
      metadata: results.metadatas?.[0]?.[i] ?? {},
    })
  }

  return formatted
}

// Search!
const queries = ['How do I add types to my code?', 'What database should I use?', 'How do I deploy my application?']

for (const query of queries) {
  console.log(`\nQuery: "${query}"`)
  const results = await semanticSearch(query, 3)

  for (const result of results) {
    console.log(`  [${result.score.toFixed(3)}] ${result.text.slice(0, 80)}...`)
    console.log(`         Source: ${result.metadata.source}`)
  }
}
```

> **Beginner Note:** ChromaDB stores vectors and performs similarity searches using an algorithm called HNSW (Hierarchical Navigable Small World). You do not need to understand HNSW — just know that it finds similar vectors much faster than comparing every pair, making it practical for millions of documents.

---

## Section 6: Semantic Search

### Building a Complete Semantic Search System

```typescript
import { embed, embedMany, generateText } from 'ai'
import { openai } from '@ai-sdk/openai'
import { mistral } from '@ai-sdk/mistral'

interface Document {
  id: string
  title: string
  content: string
  metadata: Record<string, string>
}

interface SearchResult {
  document: Document
  score: number
}

class SemanticSearchEngine {
  private documents: Document[] = []
  private embeddings: number[][] = []
  private embeddingModel = openai.embedding('text-embedding-3-small')

  /** Index a batch of documents */
  async index(documents: Document[]): Promise<void> {
    console.log(`Indexing ${documents.length} documents...`)

    const { embeddings } = await embedMany({
      model: this.embeddingModel,
      values: documents.map(d => `${d.title}\n${d.content}`),
    })

    this.documents.push(...documents)
    this.embeddings.push(...(embeddings as number[][]))

    console.log(`Index now contains ${this.documents.length} documents`)
  }

  /** Search for documents similar to a query */
  async search(query: string, topK: number = 5): Promise<SearchResult[]> {
    const { embedding: queryEmbedding } = await embed({
      model: this.embeddingModel,
      value: query,
    })

    // Calculate similarity against all documents
    const scored = this.documents.map((doc, i) => ({
      document: doc,
      score: this.cosineSimilarity(queryEmbedding, this.embeddings[i]),
    }))

    // Sort by score (highest first) and return top K
    return scored.sort((a, b) => b.score - a.score).slice(0, topK)
  }

  /** Search and generate an answer using the retrieved context */
  async searchAndAnswer(
    query: string,
    topK: number = 3
  ): Promise<{
    answer: string
    sources: SearchResult[]
  }> {
    const results = await this.search(query, topK)

    // Build context from retrieved documents
    const context = results
      .map((r, i) => `[Source ${i + 1}: ${r.document.title}]\n${r.document.content}`)
      .join('\n\n---\n\n')

    const { text } = await generateText({
      model: mistral('mistral-small-latest'),
      messages: [
        {
          role: 'system',
          content: `Answer the user's question based on the provided sources. Always cite sources using [Source N] notation. If the sources do not contain relevant information, say so.`,
        },
        {
          role: 'user',
          content: `Sources:\n${context}\n\nQuestion: ${query}`,
        },
      ],
    })

    return { answer: text, sources: results }
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0
    let magnitudeA = 0
    let magnitudeB = 0

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i]
      magnitudeA += a[i] * a[i]
      magnitudeB += b[i] * b[i]
    }

    return dotProduct / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB))
  }
}

// Usage
const engine = new SemanticSearchEngine()

await engine.index([
  {
    id: '1',
    title: 'Getting Started with Bun',
    content:
      'Bun is an all-in-one JavaScript runtime that includes a bundler, test runner, and package manager. Install with curl -fsSL https://bun.sh/install | bash.',
    metadata: { category: 'runtime' },
  },
  {
    id: '2',
    title: 'TypeScript Configuration',
    content:
      'TypeScript uses tsconfig.json for configuration. Key options include strict mode, target ES version, and module resolution strategy.',
    metadata: { category: 'typescript' },
  },
  {
    id: '3',
    title: 'Environment Variables',
    content:
      'Store sensitive configuration in .env files. Use dotenv or Bun built-in .env support to load them. Never commit .env to version control.',
    metadata: { category: 'configuration' },
  },
])

// Simple search
const results = await engine.search('How do I configure TypeScript?')
for (const r of results) {
  console.log(`[${r.score.toFixed(3)}] ${r.document.title}`)
}

// Search and answer
const qa = await engine.searchAndAnswer('How do I install the JavaScript runtime?')
console.log('\nAnswer:', qa.answer)
console.log('\nSources used:')
for (const s of qa.sources) {
  console.log(`  [${s.score.toFixed(3)}] ${s.document.title}`)
}
```

---

## Section 7: Embedding Dimensions and Trade-offs

### What Do Dimensions Mean?

Each dimension in an embedding captures some aspect of meaning. More dimensions means more nuance:

- **768 dimensions**: Captures broad semantic categories
- **1536 dimensions**: Good balance of detail and efficiency
- **3072 dimensions**: Fine-grained distinctions

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

// OpenAI's text-embedding-3 models support dimension reduction
// You can request fewer dimensions for smaller storage and faster search

const { embedding: full } = await embed({
  model: openai.embedding('text-embedding-3-large'),
  value: 'TypeScript generics enable reusable components.',
})
console.log(`Full dimensions: ${full.length}`) // 3072

// For the small model, you get 1536 dimensions
const { embedding: small } = await embed({
  model: openai.embedding('text-embedding-3-small'),
  value: 'TypeScript generics enable reusable components.',
})
console.log(`Small dimensions: ${small.length}`) // 1536
```

### Storage and Performance Trade-offs

```typescript
interface DimensionAnalysis {
  dimensions: number
  bytesPerVector: number
  vectorsPerGB: number
  searchTimeRelative: number
}

function analyzeDimensions(dims: number): DimensionAnalysis {
  // Each dimension is a 32-bit float (4 bytes) in most vector stores
  const bytesPerVector = dims * 4
  const vectorsPerGB = Math.floor((1024 * 1024 * 1024) / bytesPerVector)

  return {
    dimensions: dims,
    bytesPerVector,
    vectorsPerGB,
    searchTimeRelative: dims / 768, // Relative to 768-dim baseline
  }
}

const analyses = [768, 1024, 1536, 3072].map(analyzeDimensions)

console.log('Dimension Analysis:')
console.log('Dims  | Bytes/Vec | Vecs/GB    | Relative Speed')
console.log('------|-----------|------------|---------------')
for (const a of analyses) {
  console.log(
    `${a.dimensions.toString().padEnd(5)} | ${a.bytesPerVector.toString().padEnd(9)} | ${a.vectorsPerGB.toLocaleString().padEnd(10)} | ${a.searchTimeRelative.toFixed(2)}x`
  )
}

// Output:
// 768   | 3072      | 349,525    | 1.00x
// 1024  | 4096      | 262,144    | 1.33x
// 1536  | 6144      | 174,762    | 2.00x
// 3072  | 12288     | 87,381     | 4.00x
```

> **Advanced Note:** For most applications, 1536 dimensions (text-embedding-3-small) provides excellent quality at reasonable cost. Use 3072 dimensions only when you need to distinguish very similar texts (e.g., near-duplicate detection). Use 768 dimensions when storage or speed is critical and some quality loss is acceptable.

### Matryoshka Embeddings

Modern embedding models like OpenAI's `text-embedding-3-*` family are trained using **Matryoshka Representation Learning** (MRL). Named after Russian nesting dolls, the key idea is: **the first N dimensions of the vector are themselves a valid N-dimensional embedding**.

This means you don't need separate models for different dimension sizes. A single 3072-dimensional embedding contains valid embeddings at 256, 512, 1024, or any prefix length — each capturing progressively more semantic nuance.

```typescript
import { embed } from 'ai'
import { openai } from '@ai-sdk/openai'

// Generate a full 3072-dimensional embedding
const { embedding: full } = await embed({
  model: openai.embedding('text-embedding-3-large'),
  value: 'Matryoshka embeddings nest smaller representations inside larger ones.',
})
console.log(`Full: ${full.length} dimensions`) // 3072

// Request only 256 dimensions from the same model
// The API truncates server-side — same quality as taking first 256 of the full vector
const { embedding: compact } = await embed({
  model: openai.embedding('text-embedding-3-large'),
  value: 'Matryoshka embeddings nest smaller representations inside larger ones.',
  providerOptions: {
    openai: { dimensions: 256 },
  },
})
console.log(`Compact: ${compact.length} dimensions`) // 256
```

Why does this matter? Look at the storage and performance impact:

```typescript
interface MatryoshkaComparison {
  dimensions: number
  bytesPerVector: number
  vectorsPer1GB: number
  qualityRetention: string // approximate vs full dimensions
}

const comparisons: MatryoshkaComparison[] = [
  { dimensions: 3072, bytesPerVector: 12288, vectorsPer1GB: 87_381, qualityRetention: '100%' },
  { dimensions: 1536, bytesPerVector: 6144, vectorsPer1GB: 174_762, qualityRetention: '~99.5%' },
  { dimensions: 1024, bytesPerVector: 4096, vectorsPer1GB: 262_144, qualityRetention: '~99%' },
  { dimensions: 512, bytesPerVector: 2048, vectorsPer1GB: 524_288, qualityRetention: '~98%' },
  { dimensions: 256, bytesPerVector: 1024, vectorsPer1GB: 1_048_576, qualityRetention: '~95%' },
]

console.log('Matryoshka Dimension Trade-offs (text-embedding-3-large):')
console.log('Dims  | Storage | Vectors/GB   | Quality vs Full')
console.log('------|---------|--------------|----------------')
for (const c of comparisons) {
  console.log(
    `${c.dimensions.toString().padEnd(5)} | ${(c.bytesPerVector / 1024).toFixed(1)}KB`.padEnd(20) +
      `| ${c.vectorsPer1GB.toLocaleString().padEnd(12)} | ${c.qualityRetention}`
  )
}
```

At 256 dimensions you store **12x more vectors per GB** than at 3072 — with roughly 95% of the retrieval quality. For a million-document corpus, that's the difference between 12GB and 1GB of vector storage.

### Two-Stage Retrieval with Matryoshka

The most powerful pattern is **two-stage retrieval**: use compact embeddings for fast initial search, then re-score with full embeddings for precision.

```typescript
import { embed, cosineSimilarity } from 'ai'
import { openai } from '@ai-sdk/openai'

interface StoredDocument {
  id: string
  content: string
  embedding256: number[] // compact — for fast initial retrieval
  embedding1536: number[] // full — for precise re-scoring
}

// At ingestion time: store both compact and full embeddings
async function ingestDocument(content: string): Promise<StoredDocument> {
  // One API call gives us the full embedding
  const { embedding: full } = await embed({
    model: openai.embedding('text-embedding-3-large'),
    value: content,
    providerOptions: { openai: { dimensions: 1536 } },
  })

  // Truncate client-side for the compact version (valid because of MRL training)
  const compact = full.slice(0, 256)

  return {
    id: crypto.randomUUID(),
    content,
    embedding256: compact,
    embedding1536: full,
  }
}

// At query time: two-stage search
async function twoStageSearch(query: string, documents: StoredDocument[], topK: number = 5): Promise<StoredDocument[]> {
  const { embedding: queryFull } = await embed({
    model: openai.embedding('text-embedding-3-large'),
    value: query,
    providerOptions: { openai: { dimensions: 1536 } },
  })
  const queryCompact = queryFull.slice(0, 256)

  // Stage 1: Fast scan with 256-dim embeddings → top 50 candidates
  const candidates = documents
    .map(doc => ({
      doc,
      score: cosineSimilarity(queryCompact, doc.embedding256),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 50)

  // Stage 2: Re-score candidates with full 1536-dim embeddings
  const reScored = candidates
    .map(({ doc }) => ({
      doc,
      score: cosineSimilarity(queryFull, doc.embedding1536),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)

  return reScored.map(r => r.doc)
}
```

Why two stages instead of just using 1536 everywhere? When you have millions of vectors, Stage 1 (256-dim) is **6x faster** because cosine similarity scales linearly with dimensions. You scan 1M vectors cheaply to find 50 candidates, then do the expensive full-precision comparison on only 50.

### When to Use Each Dimension

| Dimensions | Use Case                                  | Example                               |
| ---------- | ----------------------------------------- | ------------------------------------- |
| 256        | Fast filtering, high-volume deduplication | "Is this a duplicate support ticket?" |
| 512        | Good balance for most retrieval tasks     | Standard RAG pipeline                 |
| 1024       | High-quality retrieval                    | Technical documentation search        |
| 1536       | Near-full quality, reasonable storage     | Default recommendation                |
| 3072       | Maximum precision                         | Legal/medical document matching       |

> **Beginner Note:** Not all embedding models support Matryoshka dimensions. OpenAI's `text-embedding-3-*` models do. Voyage and Cohere models have their own dimension options. Always check your model's documentation. If your model doesn't support it, you cannot just truncate vectors — the results will be meaningless. MRL training is what makes truncation work.

> **Advanced Note:** When using Matryoshka embeddings with a vector store, make sure your index is configured for the dimension size you're actually storing. A ChromaDB collection created with 1536-dimensional vectors will reject 256-dimensional vectors. Create separate collections for different dimension sizes, or pick one and stick with it.

---

## Section 8: Batch Embedding (embedMany)

### Why Batch?

Embedding one document at a time is inefficient. Network overhead, cold starts, and API rate limits make individual calls slow. `embedMany` batches multiple texts into a single API call:

```typescript
import { embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'

// Single embedding approach (slow for many documents)
async function embedOneByOne(texts: string[]): Promise<number[][]> {
  const { embed } = await import('ai')
  const embeddings: number[][] = []

  for (const text of texts) {
    const { embedding } = await embed({
      model: openai.embedding('text-embedding-3-small'),
      value: text,
    })
    embeddings.push(embedding)
  }

  return embeddings
}

// Batch embedding approach (fast)
async function embedBatch(texts: string[]): Promise<number[][]> {
  const { embeddings } = await embedMany({
    model: openai.embedding('text-embedding-3-small'),
    values: texts,
  })

  return embeddings as number[][]
}

// Compare performance
const texts = Array.from(
  { length: 100 },
  (_, i) => `Document number ${i}: This is sample text for benchmarking embeddings.`
)

console.log('Embedding 100 documents...')

const start1 = performance.now()
const result1 = await embedBatch(texts)
const batchTime = performance.now() - start1

console.log(`Batch: ${batchTime.toFixed(0)}ms for ${result1.length} embeddings`)
console.log(`Average: ${(batchTime / result1.length).toFixed(1)}ms per embedding`)
```

### Handling Large Batches

Most API providers have limits on batch size. Handle this by chunking:

```typescript
import { embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'

async function embedLargeBatch(texts: string[], batchSize: number = 100): Promise<number[][]> {
  const allEmbeddings: number[][] = []

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize)
    console.log(
      `Embedding batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(texts.length / batchSize)} (${batch.length} texts)`
    )

    const { embeddings } = await embedMany({
      model: openai.embedding('text-embedding-3-small'),
      values: batch,
    })

    allEmbeddings.push(...(embeddings as number[][]))

    // Small delay to avoid rate limits
    if (i + batchSize < texts.length) {
      await new Promise(resolve => setTimeout(resolve, 100))
    }
  }

  return allEmbeddings
}

// Usage: embed 1000 documents
const documents = Array.from({ length: 1000 }, (_, i) => `Document ${i}: Content goes here.`)

const embeddings = await embedLargeBatch(documents, 100)
console.log(`Generated ${embeddings.length} embeddings`)
```

### Embedding with Metadata

```typescript
import { embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'

interface EmbeddedDocument {
  id: string
  text: string
  embedding: number[]
  metadata: Record<string, string>
  embeddedAt: string
}

async function embedDocuments(
  documents: Array<{ id: string; text: string; metadata: Record<string, string> }>
): Promise<EmbeddedDocument[]> {
  const { embeddings } = await embedMany({
    model: openai.embedding('text-embedding-3-small'),
    values: documents.map(d => d.text),
  })

  return documents.map((doc, i) => ({
    id: doc.id,
    text: doc.text,
    embedding: embeddings[i] as number[],
    metadata: doc.metadata,
    embeddedAt: new Date().toISOString(),
  }))
}

// Usage
const docs = [
  { id: '1', text: 'React hooks simplify state management.', metadata: { source: 'blog', author: 'Alice' } },
  { id: '2', text: 'PostgreSQL supports JSON columns.', metadata: { source: 'docs', author: 'Bob' } },
  {
    id: '3',
    text: 'Docker compose orchestrates multi-container apps.',
    metadata: { source: 'tutorial', author: 'Charlie' },
  },
]

const embedded = await embedDocuments(docs)
for (const doc of embedded) {
  console.log(`${doc.id}: ${doc.text.slice(0, 40)}... (${doc.embedding.length} dims)`)
}
```

---

## Section 9: Similarity Thresholds

### Why Thresholds Matter

Not every result from a similarity search is actually relevant. Without a threshold, a search for "quantum physics" against a database of cooking recipes will still return the "most similar" recipes — they will just all be irrelevant. Thresholds filter out these false positives.

### Choosing Thresholds

The right similarity threshold depends on your use case:

```typescript
interface ThresholdConfig {
  highRelevance: number // Only return very relevant results
  mediumRelevance: number // Balance between precision and recall
  lowRelevance: number // Cast a wide net
}

// Common thresholds for OpenAI text-embedding-3-small
const defaultThresholds: ThresholdConfig = {
  highRelevance: 0.8,
  mediumRelevance: 0.6,
  lowRelevance: 0.4,
}

function categorizeResults(
  results: Array<{ text: string; score: number }>,
  thresholds: ThresholdConfig
): {
  high: Array<{ text: string; score: number }>
  medium: Array<{ text: string; score: number }>
  low: Array<{ text: string; score: number }>
  irrelevant: Array<{ text: string; score: number }>
} {
  return {
    high: results.filter(r => r.score >= thresholds.highRelevance),
    medium: results.filter(r => r.score >= thresholds.mediumRelevance && r.score < thresholds.highRelevance),
    low: results.filter(r => r.score >= thresholds.lowRelevance && r.score < thresholds.mediumRelevance),
    irrelevant: results.filter(r => r.score < thresholds.lowRelevance),
  }
}
```

### Visualizing Similarity Distributions

Understanding how similarity scores distribute across your dataset helps you choose better thresholds:

```typescript
import { embed, embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'

/**
 * Analyze the distribution of similarity scores for a query against a corpus.
 * This helps you understand what "similar" means in your specific dataset.
 */
async function analyzeScoreDistribution(query: string, documents: string[]): Promise<void> {
  const { embedding: queryEmb } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: query,
  })

  const { embeddings: docEmbs } = await embedMany({
    model: openai.embedding('text-embedding-3-small'),
    values: documents,
  })

  function cosSim(a: number[], b: number[]): number {
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

  const scores = documents.map((doc, i) => ({
    document: doc.slice(0, 60),
    score: cosSim(queryEmb, docEmbs[i] as number[]),
  }))

  scores.sort((a, b) => b.score - a.score)

  // Calculate distribution statistics
  const allScores = scores.map(s => s.score)
  const mean = allScores.reduce((a, b) => a + b, 0) / allScores.length
  const stdDev = Math.sqrt(allScores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / allScores.length)
  const median = allScores[Math.floor(allScores.length / 2)]

  console.log(`Query: "${query}"`)
  console.log(`\nDistribution: mean=${mean.toFixed(3)}, median=${median.toFixed(3)}, stdDev=${stdDev.toFixed(3)}`)
  console.log(`Suggested threshold: ${(mean + stdDev).toFixed(3)} (mean + 1 std dev)`)
  console.log(`\nTop results:`)
  for (const s of scores.slice(0, 5)) {
    const bar = '#'.repeat(Math.round(s.score * 40))
    console.log(`  ${s.score.toFixed(3)} ${bar} ${s.document}...`)
  }
  console.log(`\nBottom results:`)
  for (const s of scores.slice(-3)) {
    const bar = '#'.repeat(Math.round(s.score * 40))
    console.log(`  ${s.score.toFixed(3)} ${bar} ${s.document}...`)
  }
}

// Usage
await analyzeScoreDistribution('How do I handle errors in TypeScript?', [
  'TypeScript provides try-catch blocks for error handling.',
  'Use custom error classes that extend the Error base class.',
  'React components can use error boundaries to catch rendering errors.',
  'CSS flexbox provides flexible layout options.',
  'The weather in Tokyo is typically mild in spring.',
  'PostgreSQL supports ACID transactions for data integrity.',
])
```

### Adaptive Thresholds

```typescript
import { embed, embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'

/**
 * Instead of a fixed threshold, use the distribution of scores
 * to adaptively determine what counts as "relevant."
 */
async function adaptiveSearch(
  query: string,
  documents: string[],
  docEmbeddings: number[][]
): Promise<Array<{ text: string; score: number; relevant: boolean }>> {
  const { embedding: queryEmbedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: query,
  })

  // Calculate all similarities
  const scored = documents.map((text, i) => ({
    text,
    score: cosineSimilarity(queryEmbedding, docEmbeddings[i]),
    relevant: false,
  }))

  // Sort by score
  scored.sort((a, b) => b.score - a.score)

  // Calculate mean and standard deviation
  const scores = scored.map(s => s.score)
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length
  const stdDev = Math.sqrt(scores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / scores.length)

  // Mark as relevant if score is more than 1 standard deviation above mean
  const threshold = mean + stdDev
  for (const item of scored) {
    item.relevant = item.score >= threshold
  }

  console.log(`Adaptive threshold: ${threshold.toFixed(3)} (mean: ${mean.toFixed(3)}, stdDev: ${stdDev.toFixed(3)})`)

  return scored
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0
  let magA = 0
  let magB = 0
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    magA += a[i] * a[i]
    magB += b[i] * b[i]
  }
  return dotProduct / (Math.sqrt(magA) * Math.sqrt(magB))
}
```

### Threshold Selection Guide

| Use Case            | Threshold | Reasoning                                     |
| ------------------- | --------- | --------------------------------------------- |
| Factual Q&A         | 0.75+     | Need high precision, wrong answers are costly |
| Recommendation      | 0.50+     | Want variety, some irrelevance is OK          |
| Duplicate detection | 0.90+     | Must be near-identical                        |
| Topic clustering    | 0.60+     | Moderate similarity within topics             |
| Anomaly detection   | < 0.30    | Looking for texts unlike the corpus           |

> **Beginner Note:** Start with a threshold of 0.6 and adjust based on your results. If you are getting too many irrelevant results, raise it. If you are missing relevant documents, lower it. There is no universal correct threshold — it depends on your model, your data, and your quality requirements.

> **Advanced Note:** Thresholds are not portable across embedding models. A similarity of 0.7 with OpenAI text-embedding-3-small might correspond to 0.6 with a different model. Always calibrate thresholds with representative queries from your specific use case.

---

## Summary

In this module, you learned:

1. **What embeddings are:** Numerical vector representations of text that capture semantic meaning — similar texts produce similar vectors regardless of word choice.
2. **Embedding models:** How to use the Vercel AI SDK's `embed` and `embedMany` functions with providers like OpenAI and understand model selection trade-offs.
3. **Cosine similarity:** How to measure the angle between vectors to determine semantic similarity, and why cosine similarity is preferred over Euclidean distance for text.
4. **Vector stores with ChromaDB:** How to set up ChromaDB, insert documents with embeddings and metadata, and perform similarity-based queries.
5. **Semantic search:** How to build a complete search system that finds documents by meaning rather than keyword matching, with configurable similarity thresholds.
6. **Embedding dimensions:** How dimension size affects quality, storage, and performance, and how to choose the right model for your use case.
7. **Batch embedding:** How to use `embedMany` efficiently and manage rate limits when embedding large document collections.

In Module 9, you will combine embeddings with LLM generation to build complete RAG pipelines that ground model responses in your own data.

---

## Quiz

### Question 1 (Easy)

What is an embedding?

A) A compressed version of a text document
B) A fixed-length numerical vector that captures the semantic meaning of text
C) A hash function for text deduplication
D) A tokenization method for LLMs

**Answer: B**

An embedding is a function that converts text into a fixed-length vector of numbers (typically 768-3072 dimensions). The key property is that semantically similar texts produce similar vectors, enabling mathematical operations on meaning like similarity search and clustering.

---

### Question 2 (Easy)

What range of values does cosine similarity produce?

A) 0 to 1
B) -1 to 1
C) 0 to infinity
D) -infinity to infinity

**Answer: B**

Cosine similarity measures the cosine of the angle between two vectors, producing values from -1 (opposite directions) to 1 (same direction). In practice, embedding similarity scores for real text typically range from about 0.1 to 0.95, as truly opposite meanings are rare.

---

### Question 3 (Medium)

Why use `embedMany` instead of calling `embed` in a loop?

A) `embedMany` produces more accurate embeddings
B) `embedMany` batches API calls, reducing network overhead and latency
C) `embed` is deprecated in favor of `embedMany`
D) `embedMany` uses fewer tokens per text

**Answer: B**

`embedMany` sends multiple texts in a single API request, dramatically reducing the overhead of individual network round trips. For 100 documents, `embedMany` might make 1 API call versus 100 individual calls, resulting in 10-100x faster total embedding time.

---

### Question 4 (Medium)

What type of database is ChromaDB?

A) A relational database optimized for SQL queries
B) A document database like MongoDB
C) A vector database optimized for similarity search over embeddings
D) A key-value store like Redis

**Answer: C**

ChromaDB is a vector database specifically designed to store embedding vectors and perform fast similarity searches. It uses algorithms like HNSW to find the most similar vectors to a query without comparing every pair, making it practical for collections of millions of documents.

---

### Question 5 (Hard)

When should you use a higher similarity threshold (e.g., 0.85+)?

A) When you want to find as many related documents as possible
B) When you need high precision and wrong results are costly
C) When your document collection is very small
D) When using a local embedding model

**Answer: B**

A high similarity threshold means only very similar results are returned, which increases precision (fewer false positives) at the cost of recall (may miss some relevant documents). This is appropriate for factual Q&A, near-duplicate detection, or any scenario where returning an irrelevant result would be worse than missing a relevant one.

---

## Exercises

### Exercise 1: Semantic Search Engine over Documents

Build a semantic search engine that indexes a directory of markdown files and supports natural language queries.

**Requirements:**

1. Recursively read all `.md` files from a specified directory
2. Split each file into chunks (by heading or paragraph)
3. Embed all chunks using `embedMany`
4. Store embeddings in ChromaDB with metadata (file path, heading, chunk index)
5. Support natural language queries with configurable top-K results
6. Display results with scores, file paths, and relevant text snippets
7. Support metadata filtering (e.g., only search within a specific file)

**Starter code:**

```typescript
import { embed, embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'
import { ChromaClient } from 'chromadb'
import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

interface Chunk {
  id: string
  text: string
  filePath: string
  heading: string
  chunkIndex: number
}

async function indexDirectory(dirPath: string): Promise<number> {
  // TODO: Read all .md files recursively
  // TODO: Split into chunks
  // TODO: Embed all chunks
  // TODO: Store in ChromaDB
  throw new Error('Not implemented')
}

async function search(
  query: string,
  options?: { topK?: number; filePath?: string }
): Promise<Array<{ chunk: Chunk; score: number }>> {
  // TODO: Embed query
  // TODO: Search ChromaDB
  // TODO: Return formatted results
  throw new Error('Not implemented')
}

// Index and search
const indexed = await indexDirectory('./docs')
console.log(`Indexed ${indexed} chunks`)

const results = await search('How do I handle errors?')
for (const r of results) {
  console.log(`[${r.score.toFixed(3)}] ${r.chunk.filePath} — ${r.chunk.heading}`)
  console.log(`  ${r.chunk.text.slice(0, 100)}...\n`)
}
```

**Evaluation criteria:**

- Successfully indexes all markdown files
- Chunks preserve meaningful boundaries (headings, paragraphs)
- Search returns semantically relevant results (not just keyword matches)
- Metadata filtering works correctly
- Handles edge cases: empty files, very large files, non-UTF8 content

### Exercise 2: Embedding Comparison and Threshold Tuning

Build a tool that helps you find the optimal similarity threshold for your dataset.

**Requirements:**

1. Create a test set of 20 query-document pairs where you manually label relevance (relevant/not relevant)
2. Embed all queries and documents
3. For each threshold from 0.3 to 0.9 (step 0.05), calculate precision and recall
4. Plot (or print) a precision-recall curve
5. Recommend the optimal threshold based on F1 score
6. Compare results across two embedding models (e.g., `text-embedding-3-small` vs `text-embedding-3-large`)

**Starter code:**

```typescript
import { embed, embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'

interface LabeledPair {
  query: string
  document: string
  relevant: boolean // Human-labeled ground truth
}

interface ThresholdMetrics {
  threshold: number
  precision: number
  recall: number
  f1: number
  truePositives: number
  falsePositives: number
  falseNegatives: number
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

async function findOptimalThreshold(
  pairs: LabeledPair[],
  modelId: string
): Promise<{
  optimalThreshold: number
  metrics: ThresholdMetrics[]
}> {
  // TODO: Embed all queries and documents
  // TODO: Calculate similarity for each pair
  // TODO: For each threshold, calculate precision/recall/F1
  // TODO: Return the threshold with the highest F1
  throw new Error('Not implemented')
}

// Test dataset
const testPairs: LabeledPair[] = [
  { query: 'How do I sort an array?', document: 'Array.prototype.sort() sorts elements in place.', relevant: true },
  { query: 'How do I sort an array?', document: 'CSS grid layout provides a two-dimensional system.', relevant: false },
  { query: 'What is TypeScript?', document: 'TypeScript adds static types to JavaScript.', relevant: true },
  { query: 'What is TypeScript?', document: 'Python is an interpreted programming language.', relevant: false },
  // TODO: Add 16 more labeled pairs
]

// Run analysis
const result = await findOptimalThreshold(testPairs, 'text-embedding-3-small')
console.log(`Optimal threshold: ${result.optimalThreshold}`)
console.log('\nThreshold | Precision | Recall | F1')
console.log('----------|-----------|--------|----')
for (const m of result.metrics) {
  console.log(
    `${m.threshold.toFixed(2)}      | ${m.precision.toFixed(3)}     | ${m.recall.toFixed(3)}  | ${m.f1.toFixed(3)}`
  )
}
```

**Evaluation criteria:**

- Precision-recall trade-off is correctly calculated at each threshold
- Optimal threshold maximizes F1 score
- Comparison between models shows meaningful differences
- Results are presented in a clear, tabular format
