# Module 8: Embeddings & Similarity

## Learning Objectives

- Understand what embeddings are and how they capture semantic meaning as vectors
- Use the Vercel AI SDK's `embed` and `embedMany` functions across different providers
- Implement cosine similarity from scratch and understand why it works
- Use LanceDB as an embedded vector store for semantic search
- Build semantic search systems that find relevant documents by meaning, not keywords
- Evaluate embedding dimensions, trade-offs, and similarity thresholds

---

## Why Should I Care?

Keywords fail. A user searching for "how to fix a broken pipe" might mean a plumbing emergency or a Unix signal handling problem. Traditional search matches the words but misses the meaning. Embeddings solve this by converting text into numerical vectors that capture semantic meaning — texts with similar meanings have similar vectors, regardless of the specific words used.

Embeddings are the foundation of retrieval-augmented generation (RAG), recommendation systems, clustering, classification, and anomaly detection. Every production LLM application that works with a knowledge base — customer support bots, documentation search, code assistants — relies on embeddings to find relevant information before sending it to the model.

Understanding embeddings gives you a mental model for how LLMs "understand" text. It also gives you practical tools for building search systems that work with meaning rather than pattern matching. This module teaches both the theory and the implementation.

> **Provider Note:** This module uses OpenAI embeddings for examples. You will need an `OPENAI_API_KEY`. Mistral embeddings (`mistral.embedding('mistral-embed')`) are available as a free alternative — see the provider table in Section 1.

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
```

The result contains `embedding` (a `number[]`) and `usage` (with token count). The embedding length matches the model's dimensions (1536 for `text-embedding-3-small`).

### Understanding the Output

Build a small script that embeds a text string and then analyzes the resulting vector. Log the following:

- The vector's length (number of dimensions)
- The min, max, and average values across all dimensions
- The magnitude (L2 norm): `Math.sqrt(sum of val^2)` — for normalized embeddings like OpenAI's, this should be close to 1.0

What does it mean that the vector is normalized? How does normalization affect cosine similarity calculations?

### Embedding Different Content Types

Embeddings work for any text content — questions, statements, code, log messages, JSON. Build a script that embeds at least 5 different content types and logs a preview of each embedding (e.g., the first 3 values). Do code snippets and natural language questions produce vectors that look different? You cannot tell from the raw numbers alone — you will need cosine similarity (Section 4) to compare them meaningfully.

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

Implement `cosineSimilarity` from scratch:

```typescript
function cosineSimilarity(a: number[], b: number[]): number
```

The function should:

1. Throw an error if the vectors have different lengths
2. Compute the dot product, and the magnitudes of both vectors, in a single loop over the dimensions
3. Return 0 if either magnitude is zero (avoid division by zero)
4. Return the dot product divided by the product of magnitudes

Test it with simple 3D vectors: `[1,0,0]` vs `[1,0,0]` (same direction, expect 1.0), vs `[0,1,0]` (perpendicular, expect 0.0), vs `[-1,0,0]` (opposite, expect -1.0). We implement it manually here to understand the math.

> **SDK Shortcut:** The AI SDK exports `cosineSimilarity` from `'ai'` — use that in production. We implement it here to understand the math.

### Comparing Real Embeddings

Now use your `cosineSimilarity` (or the SDK's) with real embeddings. Build a helper function `getEmbedding(text: string): Promise<number[]>` that wraps the `embed` call.

Embed a set of texts that include semantically similar pairs and unrelated ones, for example:

- "How do I sort an array in JavaScript?"
- "What is the best way to order elements in a JS array?"
- "JavaScript array sorting methods"
- "How to cook pasta carbonara"
- "What is the weather like today?"

Compare the first text against all others using cosine similarity. The semantically similar texts should score around 0.7-0.9, while unrelated ones should score around 0.1-0.3. Do your results match these expectations?

> **Beginner Note:** Cosine similarity values for real embeddings typically range from 0.1 (unrelated) to 0.95+ (near identical). Values below 0.3 usually indicate no meaningful relationship. Values above 0.7 indicate strong semantic similarity. These thresholds vary by model.

### Other Distance Metrics

Implement two additional distance metrics:

1. **`euclideanDistance(a, b)`** — the L2 distance between two vectors. Lower values mean more similar. Range: `[0, infinity)`. Compute `sqrt(sum of (a_i - b_i)^2)`.

2. **`dotProduct(a, b)`** — the raw dot product. For normalized vectors (like OpenAI's), the dot product gives the same ranking as cosine similarity. Why is cosine similarity still preferred? Because it is bounded between -1 and 1, making thresholds portable.

---

## Section 5: Vector Stores (LanceDB)

### What is a Vector Store?

A vector store is a database optimized for storing and querying vectors. Instead of SQL's `WHERE` clause, you query by similarity: "find the 5 vectors most similar to this query vector."

### Why LanceDB?

LanceDB is an embedded vector database — think SQLite, but for vectors. There is no server to install or run. You call `connect('data/vectors')` and it creates a directory of `.lance` files on disk. Your data persists between runs automatically.

This makes it ideal for learning and prototyping: zero infrastructure, instant setup, real persistence. But unlike toy in-memory stores, LanceDB scales to millions of vectors and supports filtering, full-text search, and multiple distance metrics — the same capabilities you would use in production.

Install it with `bun add @lancedb/lancedb` (already installed as v0.27.2).

### Key API Patterns

Connect to a local database (creates the directory if it does not exist):

```typescript
import { connect } from '@lancedb/lancedb'
const db = await connect('data/vectors')
```

Create a table from an array of objects — each object must include a `vector` column:

```typescript
const table = await db.createTable('documents', data)
```

Vector search returns nearest neighbors with a `_distance` field:

```typescript
const results = await table.query().nearestTo(queryVector).limit(5).toArray()
```

Filter with SQL-like expressions:

```typescript
const filtered = await table.query().nearestTo(queryVector).where("category = 'api'").limit(10).toArray()
```

> **Note:** LanceDB data persists to `data/vectors/`. This directory is already in `.gitignore`.

### The VectorStore Interface

Build a `VectorStore` class that wraps LanceDB and provides these methods:

```typescript
interface VectorStoreDoc {
  id: string
  text: string
  vector: number[]
  metadata: string // JSON-serialized metadata string for LanceDB filtering
}

class VectorStore {
  static async create(dbPath: string, tableName: string): Promise<VectorStore>
  async index(docs: Array<{ id: string; text: string; embedding: number[]; metadata: Record<string, string> }>): Promise<void>
  async search(queryEmbedding: number[], topK?: number): Promise<Array<{ id: string; text: string; score: number; metadata: Record<string, string> }>>
  async searchWithFilter(queryEmbedding: number[], filter: string, topK?: number): Promise<Array<{ id: string; text: string; score: number; metadata: Record<string, string> }>>
}
```

**`create`** — a static factory that calls `connect(dbPath)` and opens or creates the table. Use a static factory instead of a constructor because `connect` is async. Store the `db` and `table` references as private fields.

**`index`** — takes an array of documents with pre-computed embeddings and inserts them into the LanceDB table via `table.add()`. Map each document to the `VectorStoreDoc` shape: use the `embedding` as the `vector` column, and `JSON.stringify` the metadata. If the table already exists with data, `add` appends to it.

**`search`** — calls `table.query().nearestTo(queryEmbedding).distanceType('cosine').limit(topK).toArray()`. LanceDB returns a `_distance` field (cosine distance, range 0-2). Convert to a similarity score: `1 - distance`. Parse the `metadata` string back to an object.

**`searchWithFilter`** — same as `search`, but chains `.where(filter)` before `.toArray()`. The filter is a SQL expression like `"metadata LIKE '%category%api%'"`.

### Inserting Documents

Build an indexing function that takes an array of documents (each with `id`, `text`, and `metadata`) and stores them in your `VectorStore`. The steps are:

1. Use `embedMany` to generate embeddings for all document texts in one batch call
2. Pass the documents and their embeddings to `store.index()`

Create at least 5 documents covering different topics (TypeScript, React, databases, deployment, APIs) so you can test semantic search across varied content.

### Querying by Similarity

Build a `semanticSearch` function:

```typescript
async function semanticSearch(
  store: VectorStore,
  query: string,
  topK?: number
): Promise<Array<{ text: string; score: number; metadata: Record<string, string> }>>
```

The function should:

1. Embed the query string using `embed`
2. Call `store.search()` with the query embedding and `topK`
3. Return the results

Test with several queries: "How do I add types to my code?", "What database should I use?", "How do I deploy my application?" Do the results match the documents you would expect?

### Filtering

LanceDB supports SQL-like `WHERE` clauses on any column. Since metadata is stored as a JSON string, you can use `LIKE` for simple matching or store filterable fields as top-level columns for exact matching. Use your `searchWithFilter` method to test filtering by metadata values.

> **Beginner Note:** LanceDB uses exact nearest-neighbor search by default (brute-force scan). This works perfectly for thousands of documents. For millions of vectors, you can create an IVF-PQ index on the table for approximate nearest-neighbor search — but you do not need that for learning.

> **Production Note:** LanceDB works great up to millions of vectors. For larger scale, use pgvector (PostgreSQL) or Elasticsearch. See Module 24.

---

## Section 6: Semantic Search

### Building a Complete Semantic Search System

Build a `SemanticSearchEngine` class that stores documents with their embeddings and supports similarity search using your `VectorStore` from Section 5.

```typescript
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

  async index(documents: Document[]): Promise<void>
  async search(query: string, topK?: number): Promise<SearchResult[]>
}
```

The `index` method should:

- Use `embedMany` to embed all documents in one batch. Combine `title` and `content` (e.g., `${d.title}\n${d.content}`) as the text to embed.
- Store the documents and their embeddings in parallel arrays.

The `search` method should:

- Embed the query using `embed`
- Compute cosine similarity between the query embedding and every stored embedding
- Sort by score (highest first) and return the top K results

Index at least 3 documents covering different topics (e.g., Bun runtime, TypeScript config, environment variables). Then test with queries like "How do I configure TypeScript?" and "How do I install the JavaScript runtime?" — does the engine rank the most relevant document first?

This is a pure retrieval system — no generation step. In Module 9, you will combine this with LLM generation to build a complete RAG pipeline.

> **Looking Ahead:** In Module 9, you will combine this semantic search with LLM generation to build a complete RAG pipeline. For now, we focus on retrieval only — finding the most relevant documents for a query.

---

## Section 7: Embedding Dimensions and Trade-offs

### What Do Dimensions Mean?

Each dimension in an embedding captures some aspect of meaning. More dimensions means more nuance:

- **768 dimensions**: Captures broad semantic categories
- **1536 dimensions**: Good balance of detail and efficiency
- **3072 dimensions**: Fine-grained distinctions

OpenAI's models produce different dimension counts: `text-embedding-3-small` gives 1536, `text-embedding-3-large` gives 3072. Try embedding the same text with both models and compare the vector lengths.

### Storage and Performance Trade-offs

Build an `analyzeDimensions` function:

```typescript
interface DimensionAnalysis {
  dimensions: number
  bytesPerVector: number
  vectorsPerGB: number
  searchTimeRelative: number
}

function analyzeDimensions(dims: number): DimensionAnalysis
```

Each dimension is a 32-bit float (4 bytes) in most vector stores. Calculate bytes per vector, vectors per GB (`floor(1GB / bytesPerVector)`), and relative search time (`dims / 768` as a baseline). Run it for 768, 1024, 1536, and 3072 dimensions and print a comparison table.

The key insight: 3072 dimensions stores **4x fewer vectors per GB** than 768 and searches **4x slower**. Is the quality improvement worth it for your use case?

> **Advanced Note:** For most applications, 1536 dimensions (text-embedding-3-small) provides excellent quality at reasonable cost. Use 3072 dimensions only when you need to distinguish very similar texts (e.g., near-duplicate detection). Use 768 dimensions when storage or speed is critical and some quality loss is acceptable.

### Matryoshka Embeddings

Modern embedding models like OpenAI's `text-embedding-3-*` family are trained using **Matryoshka Representation Learning** (MRL). Named after Russian nesting dolls, the key idea is: **the first N dimensions of the vector are themselves a valid N-dimensional embedding**.

This means you don't need separate models for different dimension sizes. A single 3072-dimensional embedding contains valid embeddings at 256, 512, 1024, or any prefix length — each capturing progressively more semantic nuance.

You can request a specific dimension count via provider options:

```typescript
const { embedding } = await embed({
  model: openai.embedding('text-embedding-3-large'),
  value: 'Some text',
  providerOptions: { openai: { dimensions: 256 } },
})
```

Or truncate client-side: `full.slice(0, 256)` — valid because of MRL training.

At 256 dimensions you store **12x more vectors per GB** than at 3072 — with roughly 95% of the retrieval quality. For a million-document corpus, that is the difference between 12GB and 1GB of vector storage.

### Two-Stage Retrieval with Matryoshka

The most powerful pattern is **two-stage retrieval**: use compact embeddings for fast initial search, then re-score with full embeddings for precision.

Build two functions:

```typescript
interface StoredDocument {
  id: string
  content: string
  embedding256: number[] // compact — for fast initial retrieval
  embedding1536: number[] // full — for precise re-scoring
}

async function ingestDocument(content: string): Promise<StoredDocument>
async function twoStageSearch(query: string, documents: StoredDocument[], topK?: number): Promise<StoredDocument[]>
```

**`ingestDocument`** should make one API call with 1536 dimensions, then truncate to 256 client-side for the compact version. Store both.

**`twoStageSearch`** should:

1. Stage 1: Score all documents using 256-dim compact embeddings, take the top 50 candidates
2. Stage 2: Re-score only those 50 candidates using the full 1536-dim embeddings, return top K

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

> **Advanced Note:** When using Matryoshka embeddings with a vector store, make sure all vectors in the same store use the same dimension size. Mixing 1536-dimensional vectors with 256-dimensional vectors in the same store will produce meaningless similarity scores (or runtime errors). Create separate stores for different dimension sizes, or pick one and stick with it.

---

## Section 8: Batch Embedding (embedMany)

### Why Batch?

Embedding one document at a time is inefficient. Network overhead, cold starts, and API rate limits make individual calls slow. `embedMany` batches multiple texts into a single API call:

```typescript
import { embedMany } from 'ai'

const { embeddings } = await embedMany({
  model: embeddingModel,
  values: ['text one', 'text two', 'text three'],
})
```

Build two functions — `embedOneByOne(texts)` using `embed` in a loop, and `embedBatch(texts)` using `embedMany`. Time both with `performance.now()` on a set of 100 sample texts. How much faster is the batch approach? For 100 documents, `embedMany` typically makes 1 API call versus 100 individual calls.

### Handling Large Batches

Most API providers have limits on batch size (often 2048 tokens per text or 100 texts per call). Build an `embedLargeBatch` function:

```typescript
async function embedLargeBatch(texts: string[], batchSize?: number): Promise<number[][]>
```

The function should:

1. Split the texts array into chunks of `batchSize` (default 100)
2. Call `embedMany` on each chunk sequentially
3. Add a small delay between batches to avoid rate limits (e.g., 100ms)
4. Log progress: "Embedding batch 1/10 (100 texts)"
5. Concatenate all embeddings and return them

How would you handle a batch where one text exceeds the model's token limit? What should happen to the other texts in that batch?

### Embedding with Metadata

Build an `embedDocuments` function that takes documents with `id`, `text`, and `metadata`, embeds all texts in one batch, and returns enriched objects:

```typescript
interface EmbeddedDocument {
  id: string
  text: string
  embedding: number[]
  metadata: Record<string, string>
  embeddedAt: string
}

async function embedDocuments(
  documents: Array<{ id: string; text: string; metadata: Record<string, string> }>
): Promise<EmbeddedDocument[]>
```

The function should use `embedMany` to embed all texts at once, then zip the embeddings back with the original documents, adding an `embeddedAt` ISO timestamp. Why is it important that `embedMany` preserves the order of inputs to outputs?

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
  // Your implementation here
}
```

Build this function by filtering the `results` array four times using the threshold boundaries. A result is "high" if its score is at or above `highRelevance`, "medium" if between `mediumRelevance` and `highRelevance`, "low" if between `lowRelevance` and `mediumRelevance`, and "irrelevant" if below `lowRelevance`. Return an object with the four categorized arrays.

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
  // Your implementation here
}
```

Build this function. Use `embed` for the query and `embedMany` for the documents (both with `text-embedding-3-small`). Compute cosine similarity between the query embedding and each document embedding (reuse your `cosineSimilarity` function from earlier sections). Sort scores descending, then calculate distribution statistics: mean, standard deviation, and median. Log a suggested threshold of `mean + stdDev`, the top 5 results, and the bottom 3 results. Include a simple ASCII bar visualization for each score.

Why is `mean + 1 standard deviation` a reasonable starting threshold? In what situations might this heuristic fail?

### Adaptive Thresholds

```typescript
import { embed, embedMany, cosineSimilarity } from 'ai'
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
  // Your implementation here
}
```

Build this function. Embed the query, compute cosine similarity against the pre-computed `docEmbeddings`, sort by score descending, then calculate a dynamic threshold as `mean + stdDev` of all scores. Mark each result as `relevant: true` if its score meets or exceeds the threshold. Log the computed threshold and return the scored array.

How does adaptive thresholding handle a corpus where all documents are somewhat relevant vs. one where most are irrelevant? What happens when the standard deviation is very small?

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
4. **LanceDB vector store:** How to use LanceDB as an embedded vector database — connecting to a local database, inserting documents with embeddings, performing similarity-based queries with cosine distance, and filtering with SQL-like expressions.
5. **Semantic search:** How to build a complete search system that finds documents by meaning rather than keyword matching, with configurable similarity thresholds.
6. **Embedding dimensions:** How dimension size affects quality, storage, and performance, and how to choose the right model for your use case.
7. **Batch embedding:** How to use `embedMany` efficiently and manage rate limits when embedding large document collections.
8. **Similarity thresholds:** How to choose, calibrate, and adapt similarity thresholds per use case, and why thresholds are not portable across embedding models.

> **Production Note:** Embedding API calls are relatively cheap per call but add up fast when re-embedding unchanged content. Production systems cache aggressively — keying embeddings by content hash so identical text is never embedded twice. An LRU cache with a reasonable size cap (e.g., 10,000 entries) prevents unbounded memory growth while keeping hot embeddings in memory. This caching discipline applies broadly to any embedding-heavy workflow and connects to the cost management patterns in Module 22.

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

What is the key advantage of an embedded vector database like LanceDB compared to a server-based vector database like Pinecone or Weaviate?

A) Embedded databases support more distance metrics
B) Embedded databases produce higher quality search results
C) Embedded databases require no separate server process — they run in-process and persist data to local files, like SQLite for vectors
D) Embedded databases support larger vector dimensions

**Answer: C**

Embedded vector databases like LanceDB run directly in your application process and store data as local files on disk (`.lance` files in a directory). There is no server to install, configure, or maintain. This makes them ideal for development, prototyping, and applications with up to millions of vectors. Server-based databases like Pinecone add operational complexity but offer managed scaling, replication, and multi-tenant access for larger deployments.

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

### Question 6 (Medium)

An adaptive threshold uses `mean + 1 standard deviation` of similarity scores to determine relevance. What advantage does this have over a fixed threshold like 0.7?

- A) It is faster to compute
- B) It adjusts automatically to the score distribution of each query, handling datasets where absolute similarity scores vary widely
- C) It always returns more results than a fixed threshold
- D) It eliminates the need for embedding models

**Answer: B** — Different queries produce different score distributions depending on the corpus. A fixed threshold of 0.7 might return too many results for one query and zero for another. An adaptive threshold based on the actual distribution (mean + 1 std dev) automatically adjusts, classifying results as relevant relative to the specific query's score spread.

---

### Question 7 (Hard)

You calibrate a similarity threshold of 0.75 using OpenAI's `text-embedding-3-small` model. You later switch to a different embedding model. Why can you NOT reuse the same threshold?

- A) Different models produce vectors of different lengths, making comparison impossible
- B) Each model maps text to a different vector space with different similarity score distributions — a score of 0.75 in one model may correspond to a different semantic similarity level in another
- C) Only OpenAI models support cosine similarity
- D) The threshold is stored in the model's configuration and cannot be transferred

**Answer: B** — Similarity thresholds are not portable across embedding models because each model learns a different mapping from text to vector space. A cosine similarity of 0.75 with one model may represent "highly relevant" while the same score with a different model may represent "moderately relevant." You must always recalibrate thresholds with representative queries when changing embedding models.

---

## Exercises

### Exercise 1: Semantic Search Engine over Documents

Build a semantic search engine that indexes a directory of markdown files and supports natural language queries.

**Requirements:**

1. Recursively read all `.md` files from a specified directory
2. Split each file into chunks (by heading or paragraph)
3. Embed all chunks using `embedMany`
4. Store embeddings in your LanceDB-backed `VectorStore` with metadata (file path, heading, chunk index)
5. Support natural language queries with configurable top-K results
6. Display results with scores, file paths, and relevant text snippets
7. Support metadata filtering (e.g., only search within a specific file)

**Starter code:**

```typescript
import { embed, embedMany, cosineSimilarity } from 'ai'
import { openai } from '@ai-sdk/openai'
import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

interface Chunk {
  id: string
  text: string
  filePath: string
  heading: string
  chunkIndex: number
}

// Use your VectorStore class from Section 5
// import { VectorStore } from '../../embeddings/vector-store.js'

async function indexDirectory(dirPath: string): Promise<number> {
  // TODO: Read all .md files recursively
  // TODO: Split into chunks
  // TODO: Embed all chunks
  // TODO: Store in VectorStore with metadata
  throw new Error('Not implemented')
}

async function search(
  query: string,
  options?: { topK?: number; filePath?: string }
): Promise<Array<{ chunk: Chunk; score: number }>> {
  // TODO: Embed query
  // TODO: Search VectorStore (filter by filePath metadata if provided)
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

### Exercise 2: Document Indexing Pipeline with LanceDB

Build an indexing pipeline that embeds documents and stores them in LanceDB, then supports semantic search with filtering.

**What to build:** Create `src/embeddings/exercises/indexing-pipeline.ts`

**Requirements:**

1. Create at least 10 documents with `id`, `text`, `category`, and `source` fields covering 3+ topics (e.g., TypeScript, databases, deployment)
2. Embed all documents using `embedMany`
3. Store them in LanceDB via your `VectorStore` class from Section 5
4. Implement a `searchPipeline` function that:
   - Embeds the query
   - Searches with configurable `topK` (default 5)
   - Optionally filters by category using `searchWithFilter`
   - Returns results with scores and metadata
5. Run at least 3 test queries and print results in a formatted table
6. Demonstrate filtered search (e.g., only "database" category documents)

**Starter code:**

```typescript
import { embed, embedMany } from 'ai'
import { openai } from '@ai-sdk/openai'
// import { VectorStore } from '../vector-store.js'

interface Document {
  id: string
  text: string
  category: string
  source: string
}

const documents: Document[] = [
  { id: 'ts-1', text: 'TypeScript adds static types to JavaScript.', category: 'typescript', source: 'docs' },
  { id: 'db-1', text: 'PostgreSQL supports JSONB columns for flexible schemas.', category: 'database', source: 'blog' },
  // TODO: Add 8+ more documents across different categories
]

async function buildIndex(docs: Document[]): Promise<void> {
  // TODO: Embed all document texts using embedMany
  // TODO: Store in LanceDB via VectorStore.create('data/vectors', 'documents')
  // TODO: Call store.index() with the embedded documents
  throw new Error('Not implemented')
}

async function searchPipeline(
  query: string,
  options?: { topK?: number; category?: string }
): Promise<Array<{ id: string; text: string; score: number; category: string }>> {
  // TODO: Embed the query
  // TODO: If category filter provided, use searchWithFilter
  // TODO: Otherwise, use search
  // TODO: Return formatted results
  throw new Error('Not implemented')
}

// Build index and run searches
await buildIndex(documents)

const results = await searchPipeline('How do I add types to my code?')
console.log('--- All results ---')
for (const r of results) {
  console.log(`[${r.score.toFixed(3)}] (${r.category}) ${r.text}`)
}

const dbResults = await searchPipeline('Which database should I use?', { category: 'database' })
console.log('\n--- Filtered: database only ---')
for (const r of dbResults) {
  console.log(`[${r.score.toFixed(3)}] (${r.category}) ${r.text}`)
}
```

**Evaluation criteria:**

- Documents are embedded in a single batch call (not one by one)
- LanceDB table is created and populated correctly
- Unfiltered search returns semantically relevant results ranked by score
- Filtered search narrows results to the specified category
- Results persist across runs (re-running the search without re-indexing should work)

### Exercise 3: Embedding Cache with LRU Eviction

Build an embedding cache that avoids redundant API calls by caching embeddings keyed by content hash, with LRU eviction to bound memory usage.

**What to build:** Create `src/embeddings/exercises/embedding-cache.ts`

**Requirements:**

1. Implement an `EmbeddingCache` class that wraps the Vercel AI SDK's `embed` function
2. Key cache entries by a hash of the input text (use a simple hash like SHA-256 via `Bun.hash` or `crypto.createHash`)
3. On cache hit, return the cached embedding without calling the API
4. On cache miss, call `embed`, store the result, and return it
5. Implement LRU eviction: when the cache exceeds `maxSize` entries, evict the least recently used entry
6. Track and expose cache statistics: `hits`, `misses`, and `evictions`
7. Support `clear()` to reset the cache and statistics

**Expected behavior:**

- Embedding the same text twice results in 1 API call and 1 cache hit
- Embedding `maxSize + 1` unique texts evicts the first entry; re-embedding the first text results in a cache miss
- Embedding a previously cached text moves it to the most-recently-used position, protecting it from eviction
- Statistics accurately reflect the number of hits, misses, and evictions

**File:** `src/embeddings/exercises/embedding-cache.ts`
