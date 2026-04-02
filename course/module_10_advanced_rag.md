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

Naive RAG -- embed the query, find the top-K nearest chunks, inject them into a prompt -- has three fundamental failure modes.

**Wrong chunks retrieved.** Embedding similarity does not equal semantic relevance. The query "What is our refund policy?" might retrieve chunks containing the word "refund" in unrelated contexts (e.g., "We do not offer refunds on custom orders" from a blog post, rather than the actual refund policy page). Embedding models capture surface-level semantic similarity, but they cannot always distinguish between mentions and definitions.

**Missed context.** A single query might need information scattered across multiple documents or sections. The naive approach retrieves the top-K most similar chunks, but those chunks might all come from the same document, leaving gaps. Compound questions ("How did our approach to X change between 2023 and 2024?") are especially vulnerable -- the pipeline finds chunks about one time period but not the other.

**Hallucination despite retrieval.** Even when the right chunks are retrieved, the model might hallucinate details not present in the context. This happens when the context is partially relevant -- the model "fills in the gaps" with plausible-sounding but fabricated information. The user sees cited sources and assumes the answer is grounded, but it is not.

### What to Build

Create `src/advanced-rag/naive-failures.ts` that demonstrates these three failure modes.

Define three illustrative variables:

1. A `query` string about a company refund policy
2. A `misleadingChunk` containing a blog post that mentions refunds in passing (high similarity but does not answer the question)
3. A `correctChunk` containing the actual refund policy document

Then write a `demonstrateHallucination` async function that calls `generateText` with a `partialContext` about Q3 revenue that mentions "regional breakdown available in Appendix B" but does not include the appendix. Ask for Q3 revenue by region. The system prompt should instruct the model to answer based ONLY on the provided context and say "I don't have enough information" when the context is insufficient.

The point of this script is to observe: does the model refuse to answer, or does it fabricate regional numbers? This demonstrates failure mode 3.

Think about: what makes a system prompt effective at preventing hallucination? What happens when the context is "close enough" that the model tries to fill in gaps?

### The Advanced RAG Toolkit

Each failure mode has corresponding solutions:

| Failure Mode           | Solutions                                         | Module Section     |
| ---------------------- | ------------------------------------------------- | ------------------ |
| Wrong chunks retrieved | Query transformation, HyDE, hybrid search         | Sections 2-3       |
| Missed context         | Query decomposition, hybrid search, tree indexing | Sections 2, 3, 5-7 |
| Hallucination          | Reranking, structure-aware retrieval              | Sections 4-7       |
| Unknown quality        | Assessment framework                              | Section 8          |

> **Beginner Note:** You do not need to implement all these techniques at once. Start with the naive pipeline from Module 9, measure its failures, then add techniques one at a time. Each technique addresses specific failure modes -- diagnose first, then prescribe.

> **Advanced Note:** In production systems, the combination of techniques matters. A common high-performing stack is: query rewriting + hybrid search + reranking. HyDE and tree indexing add latency and cost, so they are reserved for high-stakes use cases where accuracy justifies the overhead.

---

## Section 2: Query Transformation

### Why Transform Queries?

Users write queries like humans -- ambiguous, incomplete, and informal. "Tell me about that pricing thing we discussed" is a perfectly natural question to a colleague, but a terrible search query. Query transformation rewrites the user's raw input into one or more queries optimized for retrieval.

There are three main strategies:

1. **Query Rewriting** -- Rephrase the query for better retrieval
2. **Query Expansion** -- Generate multiple related queries to broaden recall
3. **Query Decomposition** -- Break a complex question into simpler sub-questions

### What to Build

Create three files under `src/advanced-rag/`: `query-rewriting.ts`, `query-expansion.ts`, and `query-decomposition.ts`.

**Query Rewriting** -- `rewriteQuery(originalQuery, conversationContext?): Promise<string>`

Call `generateText` with a system prompt that instructs the model to act as a search query optimizer. The rules: remove filler words and ambiguity, expand abbreviations, include relevant synonyms in parentheses, resolve pronouns using conversation context if provided. Use `temperature: 0` and `maxOutputTokens: 200`. Return only the rewritten query text.

Test it with three cases: an ambiguous query ("Tell me about that pricing thing"), a query with pronouns needing context resolution ("What are its side effects?" with ibuprofen context), and an informal query ("how do I fix the thing where it crashes on startup").

**Query Expansion** -- `expandQuery(originalQuery): Promise<ExpandedQueries>`

Use `Output.object` with a Zod schema:

```typescript
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
```

Also build `retrieveWithExpansion(originalQuery, searchFn)` that expands the query, runs each expanded query through a search function, and deduplicates results using a `Set`.

**Query Decomposition** -- `decomposeQuery(query): Promise<DecomposedQuery>`

Use `Output.object` with a schema that outputs `{ isComplex, subQuestions: [{ question, dependsOn: number[] }] }`. The system prompt should detect when a question asks about multiple entities, time periods, or requires comparison. Each sub-question should be self-contained.

Also build `answerWithDecomposition(query, retrieveAndAnswer)` that: checks if the query is complex, answers sub-questions in dependency order (using previous answers as context for dependent questions), and synthesizes a final answer from all sub-answers using a second `generateText` call.

Questions to consider: When does the overhead of decomposition pay off versus just searching with the original query? How do you handle dependencies between sub-questions?

> **Beginner Note:** Query transformation adds an LLM call before retrieval, which means more latency and cost. For simple, well-formed queries, it is often unnecessary. Start without it and add it when you see retrieval failures caused by query quality.

> **Advanced Note:** You can make query transformation conditional -- use a lightweight classifier to decide whether the query needs rewriting, expansion, or decomposition. This avoids the overhead for simple queries while still handling complex ones correctly.

### HyDE -- Hypothetical Document Embeddings

#### The Core Insight

Standard retrieval embeds the _query_ and searches for similar _documents_. But queries and documents are fundamentally different kinds of text. A query is short and interrogative ("What is our refund policy?"). A document is long and declarative ("Refund Policy: Full refunds are available within 30 days..."). Their embedding vectors may not be close in the vector space even when they are semantically related.

HyDE (Hypothetical Document Embeddings) flips the approach: instead of embedding the query, you ask an LLM to _generate a hypothetical answer_ to the query, then embed _that hypothetical answer_. The hypothetical answer is a declarative passage -- it looks like the documents in your corpus -- so its embedding is closer to the real relevant documents.

The hypothetical answer does not need to be correct. It just needs to _look like_ the kind of document that would answer the question. A hallucinated answer with the right vocabulary and structure will have a better embedding for retrieval than the raw query.

#### What to Build

Create `src/advanced-rag/hyde.ts` with three exported functions.

**`generateHypotheticalDocument(query, domainContext?): Promise<string>`**

Call `generateText` with a system prompt instructing the model to act as a technical writer. Given a question, it should write a 3-5 sentence passage as if it IS the source document (no "I think" or "The answer is"). Use `temperature: 0.3` and `maxOutputTokens: 300`. If `domainContext` is provided, include it in the system prompt.

**`hydeRetrieve(query, embeddingModel, searchIndex, options?): Promise<{ results, hypotheticalDoc }>`**

Three steps: (1) generate a hypothetical document, (2) embed it (not the query), (3) search the index using that embedding. The `searchIndex` parameter is a function `(embedding: number[], topK: number) => Promise<string[]>`.

**`compareStandardVsHyDE(query, embeddingModel, searchIndex): Promise<void>`**

Run standard retrieval (embed the query directly) and HyDE retrieval side by side. Log both result sets so you can compare which approach found better documents.

Also create `src/advanced-rag/multi-hyde.ts` with a `multiHyDE` function that generates multiple hypothetical documents (using slight temperature variation for each) and combines their embeddings. Implement two merging strategies: (A) average the embeddings into a centroid and search once, or (B) search with each embedding individually and take the union of results.

> **Beginner Note:** HyDE adds one LLM call (generating the hypothetical document) before retrieval. This typically adds 1-3 seconds of latency. For interactive applications, consider whether this tradeoff is worth it. For batch processing or high-stakes queries, it almost always is.

> **Advanced Note:** HyDE works best when your corpus contains documents that are structurally similar to each other (e.g., technical documentation, policies, reports). It works less well for heterogeneous corpora where the "shape" of a relevant document is unpredictable. You can mitigate this by using domain-specific prompts for the hypothetical document generation.

---

## Section 3: Hybrid Search

> **Building on Module 9:** You built a simple hybrid retrieval in Module 9 Section 5. This section upgrades it with a full BM25 implementation.

### Why Combine Semantic and Keyword Search?

Semantic search (embedding similarity) is good at understanding meaning -- it knows that "automobile" and "car" are similar. But it is bad at exact matches -- searching for error code "ERR_0x4F2A" with embeddings is unreliable because the embedding model may not preserve the exact string.

Keyword search (BM25/TF-IDF) is the opposite: it excels at exact matches and rare terms but cannot handle synonyms or paraphrasing. "How to fix a car" will not find a document about "automobile repair."

Hybrid search combines both: run a semantic search AND a keyword search, then merge the results. This gives you the meaning-understanding of embeddings with the precision of keyword matching.

### What to Build

Create three files: `src/advanced-rag/bm25.ts`, `src/advanced-rag/hybrid-search.ts`, and `src/advanced-rag/rrf.ts`.

**BM25 (`bm25.ts`)**

BM25 (Best Matching 25) is the industry-standard keyword relevance algorithm. It accounts for term frequency, document length, and term rarity.

You need these types and three exports:

```typescript
interface BM25Options {
  k1: number // Term frequency saturation (default 1.2)
  b: number // Length normalization (default 0.75)
}

interface BM25Index {
  documents: string[]
  tokenizedDocs: string[][]
  docLengths: number[]
  avgDocLength: number
  termDocFrequency: Map<string, number>
  totalDocs: number
}
```

`tokenize(text): string[]` -- lowercase, replace non-word characters with spaces, split on whitespace, filter tokens shorter than 2 characters.

`buildBM25Index(documents): BM25Index` -- tokenize each document, compute document lengths and average length, build a `termDocFrequency` map counting how many documents contain each unique term.

`bm25Score(query, index, options?): Array<{ document, score, index }>` -- for each document, compute the BM25 score by iterating over query terms. The score formula for each term has two components:

- IDF: `Math.log((totalDocs - df + 0.5) / (df + 0.5) + 1)`
- Normalized TF: `(tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLength / avgDocLength)))`

Multiply IDF by TF for each query term and sum. Sort results descending by score.

Test with a small corpus that includes an error code document to verify BM25 finds exact string matches that embeddings would miss.

**Hybrid Search (`hybrid-search.ts`)**

`normalizeScores(scores): Map<number, number>` -- min-max normalize an array of scores to the 0-1 range.

`hybridSearch(query, documents, embeddingModel, options?): Promise<HybridResult[]>` -- run both semantic search (embed query and each document, compute cosine similarity) and BM25 search, normalize both score sets, combine with a weighted sum (`semanticWeight` defaults to 0.7), and return the top-K results sorted by combined score.

Think about: why is normalization essential when combining BM25 and cosine similarity scores? What would happen if you skipped it?

**Reciprocal Rank Fusion (`rrf.ts`)**

`reciprocalRankFusion(rankings, k?): RRFResult[]` -- an alternative to weighted normalization. For each document that appears in any ranking, compute `sum(1 / (k + rank + 1))` across all rankings. The constant `k` (default 60) controls how much rank position matters. Return results sorted by RRF score.

```typescript
interface RRFResult {
  document: string
  rrfScore: number
  ranks: { semantic: number; keyword: number }
}
```

> **Beginner Note:** Start with a 70/30 semantic/keyword split. If your corpus has many exact identifiers (error codes, product SKUs, file paths), increase the keyword weight. If your queries are mostly natural language, increase the semantic weight.

> **Advanced Note:** Many vector databases (Pinecone, Weaviate, Qdrant) support hybrid search natively, combining dense (embedding) and sparse (BM25/SPLADE) vectors in a single query. Use native support when available -- it is faster and handles normalization internally.

---

## Section 4: Reranking

### Why Rerank?

Retrieval (whether semantic, keyword, or hybrid) casts a wide net. It finds documents that are _probably_ relevant based on surface-level features -- embedding proximity or keyword overlap. But "probably relevant" is not "actually relevant."

Reranking is a second pass that applies a more expensive, more accurate model to the retrieved candidates. The retriever finds 20-50 candidates quickly; the reranker scores each one carefully and returns the best 3-5. This two-stage approach gives you both speed (fast retrieval over millions of documents) and accuracy (careful reranking of a small set).

### What to Build

Create two files: `src/advanced-rag/reranking.ts` and `src/advanced-rag/cohere-rerank.ts`.

**LLM Reranking (`reranking.ts`)**

Implement two approaches, both exporting `RankedDocument`:

```typescript
interface RankedDocument {
  document: string
  score: number
  reasoning: string
  originalRank: number
}
```

`llmRerank(query, documents, topK?): Promise<RankedDocument[]>` -- **Pointwise approach.** Score each document independently with a `generateText` call using `Output.object` and this schema:

```typescript
const RelevanceScoreSchema = z.object({
  score: z.number().min(0).max(10).describe('Relevance score from 0-10'),
  reasoning: z.string().describe('Brief explanation of the score'),
})
```

The system prompt should instruct the model to judge relevance: 8-10 means directly answers the query, 4-7 is somewhat related, below 4 is not useful. Use `temperature: 0`. Score all documents in parallel with `Promise.all`, sort by score, return the top K.

`listwiseRerank(query, documents, topK?): Promise<RankedDocument[]>` -- **Listwise approach.** Present all documents at once in a single `generateText` call. The schema should return an array of `{ documentIndex, relevanceScore, reasoning }`. More efficient (1 LLM call vs N), but limited by context window.

Think about: when would you choose pointwise over listwise? What are the trade-offs in cost, latency, and accuracy?

**Cohere Rerank (`cohere-rerank.ts`)**

`cohereRerank(query, documents, topK?): Promise<CohereRerankResult[]>` -- call the Cohere rerank API (`POST https://api.cohere.com/v2/rerank`) with model `rerank-v3.5`. Parse the response to extract `{ index, relevanceScore, document }` for each result.

Also build `retrieveRerankGenerate(query, vectorSearch, options?)` that chains the full pipeline: broad retrieval (default 20 candidates), Cohere rerank (default top 3), then `generateText` with the reranked documents as context.

> **Beginner Note:** Reranking is one of the highest-impact improvements you can make to a RAG pipeline. If you implement only one technique from this module, make it reranking. Retrieve 20 candidates, rerank to 3-5, and generate from those.

> **Advanced Note:** The optimal retrieve-K depends on your reranker's quality and the size of your corpus. With a strong reranker (Cohere rerank-v3.5 or a capable LLM), retrieving 50-100 candidates and reranking to 3-5 is effective. With a weaker reranker, keep retrieve-K lower (10-20) to avoid overwhelming it with irrelevant documents.

---

## Section 5: Structure-Aware Retrieval

### Why Similarity does not equal Relevance

Vector similarity search finds chunks that are semantically close to the query. But "close" is not always "relevant." Consider a financial analyst asking "What drove Q3 revenue growth?":

- A chunk about Q2 revenue might be very similar (same vocabulary: revenue, growth, quarterly) but is the wrong quarter
- A chunk about a new product launch in Q3 might be less similar (different vocabulary) but is the actual answer
- A chunk titled "Management Discussion: Factors Affecting Performance" might have low similarity to "revenue growth" but contains the exact analysis needed

The fundamental issue: embedding similarity measures lexical and semantic proximity, not logical relevance. For complex documents with structure (sections, chapters, hierarchies), the structure itself carries information that embeddings discard.

### The Tree Indexing Approach

Instead of chunking a document and embedding the chunks, tree indexing preserves the document's natural structure as a navigable hierarchy -- like an LLM-optimized table of contents.

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
2. **Query phase:** An LLM navigates the tree top-down, reading summaries to decide which branches to explore -- like a human expert scanning a table of contents

No embeddings. No vector database. No chunking. The LLM reasons about document structure to find relevant sections.

### When Tree Indexing Wins

| Scenario                                              | Vector RAG                      | Tree Indexing                        |
| ----------------------------------------------------- | ------------------------------- | ------------------------------------ |
| Short FAQ documents                                   | Better (simple, fast)           | Overkill                             |
| Long structured reports (financial, legal, technical) | Chunks lose structure           | Better -- preserves hierarchy        |
| Queries requiring cross-section reasoning             | Misses connections              | Follows document logic               |
| Heterogeneous document sets                           | Better -- embeddings generalize | Harder -- needs consistent structure |
| Exact fact lookup ("what was Q3 revenue?")            | Depends on chunk boundaries     | Navigates directly to section        |
| Semantic/conceptual queries                           | Better -- similarity works well | May over-navigate                    |

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

Each non-leaf node has a summary generated by the LLM. The summary is what the retrieval LLM reads when deciding which branch to explore -- it needs to be informative enough to guide navigation without being so long that it defeats the purpose of the tree.

### What to Build

Create `src/advanced-rag/tree-index.ts` with two functions.

**`summarizeSection(title, content): Promise<string>`**

Call `generateText` with a system prompt: "Summarize what this document section covers in 1-2 sentences. Focus on the topics and key information a reader would find here. Be specific." Use `maxOutputTokens: 100`.

**`parseMarkdownToTree(markdown): TreeNode`**

Parse a markdown document into a tree structure based on heading levels. Split into lines, track heading levels via `#{1,6}` matches, group content under headings, and build parent-child relationships based on heading depth. Return a root node with level 0.

The parsing logic: iterate through lines. When you encounter a heading, determine its level from the number of `#` characters. Group all content until the next heading as that node's content. A heading at a deeper level becomes a child of the most recent shallower heading.

### Summary Quality Matters

The quality of node summaries directly determines retrieval quality. A vague summary ("This section discusses various topics") forces the LLM to explore that branch speculatively. A specific summary ("This section analyzes Q3 2024 revenue by product line, showing 23% growth driven by the Enterprise tier") lets the LLM make confident navigation decisions.

```typescript
// Bad: too vague -- LLM can't decide if this branch is relevant
{
  summary: 'Discusses company performance and metrics.'
}

// Good: specific -- LLM knows exactly what's here
{
  summary: 'Q3 2024 revenue breakdown: $12M total, Enterprise tier grew 23%, SMB flat. Key driver was the DataFlow product launch in August.'
}
```

> **Beginner Note:** Building a tree index is an upfront cost -- you run LLM calls during indexing, not just during querying. For a 100-page document, you might make 20-50 summarization calls. This is a one-time cost that amortizes over many queries.

> **Advanced Note:** The tree index can be persisted as JSON and reloaded without re-indexing. When the source document changes, you can do incremental updates -- only re-summarize the sections that changed. This is analogous to re-embedding changed chunks in vector RAG, but cheaper since summarization is faster than embedding + indexing.

---

## Section 7: LLM-Navigated Tree Search

### The Retrieval Algorithm

Given a query and a tree index, the LLM navigates top-down:

1. Start at the root -- read summaries of all top-level children
2. Ask the LLM: "Which of these sections is most likely to contain the answer?"
3. Descend into the chosen branch -- read summaries of its children
4. Repeat until reaching a leaf node (actual content)
5. Return the leaf content as the retrieved context

### What to Build

Create `src/advanced-rag/tree-search.ts` with two navigation functions.

**`navigateTree(query, node, model?): Promise<TreeNode[]>`**

Use `Output.object` with this schema:

```typescript
const NavigationSchema = z.object({
  selectedNodeId: z.string().describe('ID of the most relevant child node'),
  confidence: z.number().min(0).max(1).describe('How confident the selection is'),
  reasoning: z.string().describe('Why this node was selected'),
})
```

Base case: if the node has no children, return `[node]`. Otherwise, format each child as `[id] title: summary` and present them to the LLM with a system prompt: "You are a document navigator. Given a query and a list of document sections with summaries, select the section most likely to contain the answer." Find the selected child and recurse.

**`multiPathNavigate(query, node, maxPaths?, model?): Promise<TreeNode[]>`**

Same idea, but instead of selecting one node, use a schema that returns an array of `{ nodeId, relevanceScore }`. Explore the top `maxPaths` (default 2) branches by recursing into each. Collect and return all leaf nodes found across all paths.

Think about: what if the LLM returns a `selectedNodeId` that does not match any child? How should you handle that? What happens if the tree is very deep -- is there a risk of excessive LLM calls?

### Comparing Retrieval Approaches

The three retrieval paradigms each have different strengths:

| Approach          | How it finds relevant content                          | Strengths                                                 | Weaknesses                                                    |
| ----------------- | ------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------- |
| **Vector RAG**    | Embed query, cosine similarity, top-K chunks           | Fast, works on unstructured text, no LLM during retrieval | Similarity does not equal relevance, loses document structure |
| **Tree Indexing** | LLM navigates document hierarchy top-down              | Preserves structure, high precision on structured docs    | Requires structured documents, LLM cost during retrieval      |
| **Graph RAG**     | Traverse entity/relationship graph from query entities | Captures relationships, multi-hop reasoning               | Complex setup, entity extraction quality critical             |

In practice, these can be combined. Use vector RAG for broad recall, tree indexing for structured document deep-dives, and graph RAG for relationship-heavy queries.

> **Beginner Note:** Start with vector RAG (Module 9). Add tree indexing when you work with long structured documents where vector search misses context. Add graph RAG (Module 12) when relationships between entities matter more than the text itself.

> **Advanced Note:** A production system might route queries to different retrieval backends: simple factual queries go to vector search, analytical queries about structured reports go to tree search, and relationship queries go to graph search. The routing itself can be done by an LLM (see Module 14: Agent Fundamentals).

---

## Section 8: RAG Assessment Framework

### Why You Need Systematic Measurement

The techniques in this module -- HyDE, hybrid search, reranking, tree indexing -- each add complexity and cost. How do you know they actually improve your pipeline? How do you decide which combination to use?

You need a systematic assessment framework that measures multiple dimensions of RAG quality. Without it, you are tuning by vibes and cherry-picked examples.

### Assessment Dimensions

RAG quality measurement covers four key dimensions:

| Dimension             | Question                                              | What It Catches     |
| --------------------- | ----------------------------------------------------- | ------------------- |
| **Context Relevance** | Are the retrieved chunks relevant to the query?       | Bad retrieval       |
| **Context Recall**    | Do the retrieved chunks cover all needed information? | Missing context     |
| **Faithfulness**      | Is the answer grounded in the retrieved context?      | Hallucination       |
| **Answer Relevance**  | Does the answer actually address the query?           | Off-topic responses |

### What to Build

Create `src/advanced-rag/rag-assessment.ts` with four measurement functions and one orchestrator.

You will need these types:

```typescript
interface RAGTestCase {
  query: string
  expectedAnswer: string
  expectedSources?: string[]
}

interface RAGAssessmentResult {
  query: string
  answer: string
  retrievedChunks: string[]
  metrics: {
    contextRelevance: number
    faithfulness: number
    answerRelevance: number
    answerCorrectness: number
  }
  details: {
    contextRelevanceReasoning: string
    faithfulnessReasoning: string
    answerRelevanceReasoning: string
    correctnessReasoning: string
  }
}
```

**Metric 1: `measureContextRelevance(query, chunks): Promise<{ score, reasoning }>`**

Use `Output.object` with a schema that returns `{ relevantChunks, totalChunks, score, reasoning }`. The system prompt asks the LLM to count how many chunks contain information that helps answer the query. Score = relevant / total.

**Metric 2: `measureFaithfulness(answer, context): Promise<{ score, reasoning }>`**

Schema returns `{ claims: [{ claim, supportedByContext, evidence? }], score, reasoning }`. The system prompt extracts every factual claim from the answer and checks whether the context provides evidence for it. Score = supported / total claims.

**Metric 3: `measureAnswerRelevance(query, answer): Promise<{ score, reasoning }>`**

Schema returns `{ score, reasoning }`. Scoring guide: 1.0 = perfectly answers, 0.7-0.9 = mostly answers, 0.4-0.6 = partially, 0.1-0.3 = barely related, 0.0 = irrelevant.

**Metric 4: `measureCorrectness(answer, referenceAnswer): Promise<{ score, reasoning }>`**

Compares the generated answer against a ground truth reference. Scores based on factual accuracy, not wording.

**Orchestrator: `assessRAGPipeline(testCases, ragPipeline): Promise<{ results, aggregate }>`**

For each test case: run the pipeline, measure all four metrics in parallel with `Promise.all`, collect results. Compute aggregate averages. Log per-test and aggregate scores.

All measurement functions should use `temperature: 0` for reproducibility.

Also create `src/advanced-rag/assessment-test-suite.ts` that defines a test suite of 5+ query/answer pairs and a `comparePipelines` function that runs two pipeline variants against the same test suite and prints a comparison table.

> **Beginner Note:** Start with 10-20 manually curated test cases that cover your most important query types. You do not need hundreds of test cases to get useful signal. Focus on queries you know your pipeline struggles with.

> **Advanced Note:** LLM-as-judge measurement (what we use above) is itself noisy. For high-confidence results, use multiple judge calls per test case and average the scores. Also consider using a different model as judge than the one generating answers -- this reduces bias. Module 19 covers testing in much more depth.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Multi-Source Retrieval

Real-world RAG systems rarely retrieve from a single source. Production context-building systems assemble context from multiple sources, each with a different retrieval strategy:

- **Environment state** — retrieved via shell commands (e.g., `git status`, runtime info)
- **Project configuration** — retrieved via file reads (project rules, coding standards)
- **Session memory** — retrieved from a memory store (conversation history, extracted facts)
- **Tool definitions** — retrieved from static config (what the system can do)
- **Document chunks** — retrieved via vector search (the classic RAG source)

Each source uses a different retrieval method — file I/O, shell execution, memory lookup, embedding similarity — but they all contribute to the same context window. This is hybrid retrieval in the broadest sense: combining fundamentally different retrieval strategies for different types of information.

```typescript
interface ContextSource {
  name: string
  retrieve: () => Promise<string>
  priority: number
  maxTokens: number
}

async function assembleMultiSourceContext(sources: ContextSource[]): Promise<string> {
  /* ... */
}
```

The design challenge is not the retrieval itself but the orchestration: how to prioritize sources, allocate token budgets, and handle failures gracefully when one source is slow or unavailable.

> **Note: Rule-Based Contextual Selection** — Not all retrieval is query-driven. Production systems use rule-based selection to inject relevant context based on what the user is doing. If the user is editing test files, inject testing guidelines. If they are debugging, inject common error patterns. This is a form of RAG where the "query" is the user's current intent, and the retrieval is a rule-based lookup rather than vector similarity. Sometimes you know exactly what to retrieve — no embeddings needed.

---

## Section 10: Re-Retrieval After Context Loss

Long-running conversations eventually hit context limits. When the system compacts or truncates the conversation, important context can be lost. Production systems handle this by re-retrieving relevant information from their session memory store after compaction.

This is analogous to the query transformation pattern from Section 2, but applied to the system itself rather than the user's query. The "query" is "what was important in the compacted conversation?" and the retrieval source is the memory store where key facts were extracted before compaction.

The pattern works in three steps:

1. **Before compaction:** Extract and store key facts, decisions, and context from the conversation
2. **During compaction:** Summarize or truncate the conversation to fit the window
3. **After compaction:** Re-retrieve relevant memories and inject them into the new context

```typescript
async function rehydrateAfterCompaction(
  compactedHistory: Message[],
  memoryStore: MemoryEntry[],
  currentQuery: string
): Promise<Message[]> {
  /* ... */
}
```

This pattern is critical for any long-running RAG application. Without re-retrieval, the system "forgets" important context after compaction, leading to repeated questions and inconsistent behavior.

---

## Section 11: LSP-Augmented Retrieval

> **What is LSP?** Language Server Protocol is a standard for code intelligence — it lets editors understand code structure (go-to-definition, find-references, type information) by communicating with a language-specific server. Think of it as structured code understanding that complements text-based search.

When building RAG over codebases, Language Server Protocol (LSP) provides a retrieval source that vector search cannot match. LSP gives structured code intelligence — go-to-definition, find-all-references, call hierarchies, type hierarchies — with perfect precision.

Consider the query "find all callers of this function." With vector search, you embed the function and find semantically similar code — which may include functions with similar names, similar logic, or similar comments, but not necessarily actual callers. With LSP, you get the exact list of call sites. No false positives.

The power comes from combining all three retrieval types:

- **Vector search** for fuzzy semantic queries ("functions related to authentication")
- **BM25/keyword search** for exact text matches ("ERR_AUTH_FAILED")
- **LSP** for structural queries ("all implementations of this interface", "call hierarchy of processPayment")

```typescript
interface CodeRetrievalResult {
  source: 'vector' | 'keyword' | 'lsp'
  content: string
  confidence: number
}

async function hybridCodeRetrieval(query: string, context: CodeContext): Promise<CodeRetrievalResult[]> {
  /* ... */
}
```

LSP-augmented retrieval is specific to code, but the principle is general: when a domain has a structured query system (SQL for databases, SPARQL for RDF stores, GraphQL for APIs), use it alongside vector search rather than relying on embeddings alone.

---

## Section 12: Diagnostic-Driven Context

Not all retrieval is triggered by the user's query. Some of the most valuable context comes from automated analysis that surfaces problems the user may not have articulated.

Production coding agents feed LSP diagnostics — type errors, lint warnings, unused imports, deprecated API usage — to the LLM as context. Before the model even looks at the code, it knows what is wrong. This is proactive retrieval: the system retrieves problem indicators from static analysis rather than from the user's question.

```typescript
interface DiagnosticContext {
  file: string
  line: number
  severity: 'error' | 'warning' | 'info'
  message: string
  source: string // 'typescript' | 'eslint' | 'test-runner'
}

function buildDiagnosticContext(diagnostics: DiagnosticContext[], maxItems: number): string {
  /* ... */
}
```

The pattern generalizes beyond code. Any domain where automated analysis can surface relevant signals — linting, validation, schema checks, test results — benefits from diagnostic-driven context. The RAG pipeline retrieves not just from a document store but from analysis tools that proactively identify what matters.

> **Key Insight:** Retrieval does not have to be reactive. Proactive retrieval from automated analysis (type checkers, linters, test runners) often surfaces the most actionable context — problems the user needs to fix but has not yet asked about.

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

What advantage does LLM-navigated tree search have over standard vector similarity search for structured documents?

- A) It uses less memory than vector indexes
- B) It follows the document's hierarchical structure, navigating from broad summaries to specific sections, so it can find information based on meaning and position rather than just embedding similarity
- C) It is always faster than vector search
- D) It does not require an LLM

**Answer: B** — Standard vector search treats all chunks as a flat collection and ranks by embedding similarity, which can miss context that lives in a specific section of a structured document. Tree search builds an index that mirrors the document's hierarchy (chapters, sections, subsections) and uses the LLM to navigate from root summaries down to the most relevant leaf nodes. This preserves structural context and works especially well for long documents where the same terms appear in multiple sections with different meanings.

---

### Question 5 (Hard)

You run an assessment suite and find: context relevance = 0.9, faithfulness = 0.4, answer relevance = 0.8. What is the most likely issue with your RAG pipeline?

- A) The retrieval stage is returning wrong documents
- B) The LLM is hallucinating — generating claims not supported by the retrieved context, despite the context being relevant
- C) The query transformation is too aggressive
- D) The chunking strategy is too coarse

**Answer: B** — High context relevance (0.9) means retrieval is working well — the right chunks are being found. High answer relevance (0.8) means the answer addresses the query. But low faithfulness (0.4) means the model is making claims that are not supported by the context — it is filling in gaps with hallucinated information. The fix is better prompting (stricter instructions to only use context), contextual compression (focus the context), or switching to a model that is more instruction-following.

### Question 6 (Medium)

In a multi-source retrieval system, why is rule-based contextual selection sometimes preferable to query-driven vector search?

a) Rule-based selection is always more accurate than vector search
b) When you know what context is relevant based on what the user is doing (e.g., editing tests triggers testing guidelines), a rule-based lookup is deterministic, free, and instant — no embeddings needed
c) Rule-based selection works with more file types
d) Vector search cannot retrieve configuration files

**Answer: B**

**Explanation:** Not all retrieval needs to be query-driven. If the user is editing test files, the system can deterministically inject testing guidelines without computing any embeddings or running similarity search. This is a form of RAG where the "query" is the user's current activity, and the retrieval is a simple rule-based lookup. It is faster, cheaper, and more predictable than vector search for cases where the relevance mapping is known in advance.

---

### Question 7 (Hard)

After a long conversation is compacted to fit the context window, the agent starts asking the user to repeat information that was discussed earlier. What re-retrieval pattern fixes this, and why does it work?

a) Re-embed the compacted conversation and search for missing topics
b) Before compaction, extract key facts and decisions into a memory store. After compaction, re-retrieve relevant memories and inject them into the new context — restoring continuity without restoring the full conversation
c) Increase the context window size to avoid compaction
d) Disable conversation compaction entirely

**Answer: B**

**Explanation:** The re-retrieval pattern works in three steps: extract key facts before compaction, compact the conversation, then re-retrieve relevant memories from the store based on the current query. This restores important context without restoring the full conversation, keeping the context window manageable while preventing the "amnesia" problem. Simply increasing the window (C) or disabling compaction (D) are not sustainable solutions for long-running sessions.

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

> **Local Alternative (Ollama):** Advanced RAG techniques (HyDE, query decomposition, tree indexing) work with `ollama('qwen3.5')` — they're prompt-based strategies, not provider features. LLM-based reranking also works locally, though it will be slower than API reranking services. For hybrid search, the BM25 + semantic combination is fully local.

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
9. **Multi-source retrieval:** Production RAG systems combine multiple retrieval sources (files, shell commands, memory, vector search) with different strategies, orchestrated by priority and token budgets.
10. **Re-retrieval after context loss:** When conversation compaction drops important context, re-retrieving relevant memories from a session store restores continuity.
11. **LSP-augmented retrieval:** Structured code intelligence (definitions, references, call hierarchies) provides precise retrieval that complements fuzzy vector search for code-specific RAG.
12. **Diagnostic-driven context:** Proactive retrieval from automated analysis (type errors, lint warnings) surfaces actionable context without waiting for the user to ask.

In Module 11, you will tackle the other side of the RAG pipeline — document processing. Better ingestion, chunking, and metadata extraction feed directly into the retrieval quality improvements you built here.
