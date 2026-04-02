# Module 12: Knowledge Graphs

## Learning Objectives

- Understand why knowledge graphs capture relationships that vector search alone cannot
- Extract entities from documents using LLMs with structured output
- Extract relationships as subject-predicate-object triples from text
- Build a simple in-memory graph using an adjacency list
- Implement graph traversal algorithms for multi-hop retrieval
- Combine graph-based and vector-based retrieval in the Graph RAG pattern
- Handle graph consistency challenges including deduplication and entity resolution
- Know when to use Graph RAG vs Vector RAG for different use cases

---

## Why Should I Care?

Vector search finds documents that are _about_ the same topic. But many questions require understanding _relationships between_ entities — and relationships are exactly what vector search is bad at.

Consider a simple question: "Which team members have worked on both Project Alpha and Project Beta?" Vector search will find documents that mention Project Alpha and documents that mention Project Beta. But it cannot _join_ that information. It cannot find the intersection of people. The answer requires traversing relationships: Project Alpha -> team members -> check if they also appear in -> Project Beta -> team members.

Knowledge graphs store exactly this kind of structured, relational information. Entities (people, projects, documents, concepts) are nodes. Relationships (works-on, reports-to, depends-on, mentions) are edges. Queries become graph traversals: start at a node, follow edges, find connected nodes.

In a RAG pipeline, knowledge graphs complement vector search. Vector search finds relevant passages; graph search finds related entities and their connections. Combining both gives you answers that are grounded in both text content and structural relationships. This is the Graph RAG pattern, and it is one of the most powerful retrieval strategies available.

This module teaches you to build knowledge graphs from documents using LLM-powered entity and relationship extraction, implement graph traversal for retrieval, and combine graph and vector search. The graph does not need to be stored in a specialized graph database — an in-memory adjacency list works for many use cases.

---

## Connection to Other Modules

This module builds on the entity extraction concepts from **Module 11 (Document Processing)** and the structured output patterns from **Module 3 (Structured Output)**.

- **Module 8 (Embeddings & Similarity)** provides vector search for the hybrid Graph RAG approach.
- **Module 9 (RAG Fundamentals)** and **Module 10 (Advanced RAG)** provide the retrieval pipeline that graph search augments.
- **Module 14 (Agent Fundamentals)** can use graph traversal as a tool, letting agents navigate knowledge graphs autonomously.

Think of vector search as finding the right library books and graph search as finding the connections between people, ideas, and events in those books.

---

## Section 1: Why Knowledge Graphs?

### What Vector Search Cannot Do

Vector search excels at finding semantically similar text. But there are entire classes of questions it fails on:

**Multi-hop reasoning.** "Who is the manager of the person who wrote the refund policy?" requires following two relationships: document -> author, author -> manager. Vector search cannot chain relationships.

**Aggregation.** "How many projects use TypeScript?" requires counting entities with a specific property. Vector search returns similar documents, not counts.

**Path finding.** "How are Alice and Bob connected in the organization?" requires finding a path through relationships. Vector search has no concept of paths.

**Inverse relationships.** "Which documents reference this API endpoint?" requires following inbound edges. Vector search only goes from query to document, not document to referencing documents.

```typescript
// src/knowledge-graphs/why-graphs.ts

// Examples of questions that require graph-based reasoning

const graphQuestions = [
  {
    question: 'Who manages the author of the security policy?',
    type: 'multi-hop',
    graphTraversal: 'security_policy -> authored_by -> person -> managed_by -> manager',
    vectorSearchFails: 'Returns docs about security OR managers, not the specific relationship',
  },
  {
    question: 'Which projects depend on the authentication service?',
    type: 'inverse-relationship',
    graphTraversal: 'auth_service <- depends_on <- project (find all inbound edges)',
    vectorSearchFails: 'Might find auth service docs but not the projects that depend on it',
  },
  {
    question: 'What is the shortest connection between Alice and Dave?',
    type: 'path-finding',
    graphTraversal: 'BFS from Alice to Dave through any relationship',
    vectorSearchFails: 'No concept of paths or connections between entities',
  },
  {
    question: 'How many teams have more than 5 members?',
    type: 'aggregation',
    graphTraversal: 'For each team node, count member_of edges, filter > 5',
    vectorSearchFails: 'Returns documents mentioning team sizes but cannot aggregate',
  },
]

// When to use graphs vs vectors
const decisionMatrix = {
  useVectorSearch: [
    'Find documents about a topic',
    'Semantic similarity search',
    'Answer questions from a single passage',
    'Find similar items',
  ],
  useGraphSearch: [
    'Multi-hop relationship queries',
    'Find connections between entities',
    'Aggregate across entities',
    'Traverse organizational/dependency structures',
  ],
  useBoth: [
    'Complex questions requiring both content and relationships',
    'Questions that need context from related entities',
    'Verification of claims across multiple sources',
    'Research questions spanning multiple documents',
  ],
}
```

> **Beginner Note:** You do not need a specialized graph database (Neo4j, Amazon Neptune) to start. An in-memory adjacency list or even a JSON file works for knowledge graphs with up to ~100,000 nodes. Graduate to a graph database only when you need persistence at scale, complex graph algorithms, or concurrent access.

> **Advanced Note:** The boundary between "graph question" and "vector question" is blurry. A well-structured RAG pipeline with good chunking and metadata can handle some relationship queries through careful prompt engineering. The decision to add a knowledge graph should be driven by observed failure patterns -- if your users consistently ask relationship questions that your RAG pipeline cannot answer, a graph will help.

### A Tangible Example: Codebase Dependency Graphs

If you are building LLM applications, you already have a knowledge graph in front of you -- your codebase. Tools depend on services, commands depend on tools, and agents depend on tools and other agents:

```
BashTool -> ShellExecutionService
CommitCommand -> GitTools -> BashTool
AgentTool -> spawns sub-agents -> [tool subsets]
```

These are graph-structured relationships. "What happens if the shell service goes down?" requires traversing dependencies to find all affected tools and commands. "Which tools does the commit workflow use?" requires following the dependency chain.

You encounter these dependency graphs in any non-trivial system: microservice architectures (service A calls service B), npm packages (dependency trees), database schemas (foreign key relationships), and API endpoints (route -> handler -> service -> database). Recognizing when your data is naturally graph-shaped is the first step toward deciding whether to build an explicit knowledge graph or leverage an implicit one.

---

## Section 2: Entity Extraction with LLMs

> **Building on Module 11:** You already built entity extraction with `Output.object` in Module 11 Section 5. Here we extend it to extract relationships between entities.

### Defining Entities

An entity is a distinct, identifiable thing: a person, organization, project, technology, concept, or event. Entity extraction is the first step in building a knowledge graph -- you cannot create relationships until you know what the entities are.

Define the schemas that will drive your extraction:

```typescript
// src/knowledge-graphs/entity-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const EntityTypeEnum = z.enum([
  'person',
  'organization',
  'project',
  'technology',
  'concept',
  'document',
  'event',
  'location',
  'product',
  'service',
])

type EntityType = z.infer<typeof EntityTypeEnum>

const EntitySchema = z.object({
  name: z.string().describe('Canonical name of the entity'),
  type: EntityTypeEnum,
  aliases: z.array(z.string()).describe('Alternative names or abbreviations'),
  description: z.string().describe('Brief description (1-2 sentences)'),
  attributes: z.record(z.string()).describe('Key-value attributes like role, version, date'),
})

type Entity = z.infer<typeof EntitySchema>

const EntityExtractionSchema = z.object({
  entities: z.array(EntitySchema).describe('All distinct entities found in the text'),
})

async function extractEntities(text: string, domainContext?: string): Promise<Entity[]>
```

Build `extractEntities` to call `generateText` with `Output.object({ schema: EntityExtractionSchema })`. The system prompt should instruct the model to:

- Use the most complete, canonical form of each name
- List all aliases (abbreviations, nicknames, alternative spellings)
- Classify by type from the enum
- Add relevant attributes as key-value pairs
- Be thorough: extract even implicitly referenced entities
- Deduplicate: if the same entity appears multiple ways, merge them

If `domainContext` is provided, include it in the system prompt to guide extraction.

### Batch Extraction Across Chunks

When processing multiple document chunks, you need to deduplicate entities across chunks. Build `extractEntitiesFromChunks`:

```typescript
async function extractEntitiesFromChunks(
  chunks: string[],
  domainContext?: string
): Promise<{
  entities: Entity[]
  chunkEntityMap: Map<number, string[]> // chunk index -> entity names
}>
```

This function should:

1. Iterate through chunks, calling `extractEntities` on each.
2. Maintain a `seenEntities` map (normalized name -> Entity) for deduplication.
3. When a duplicate is found, merge aliases (using `Set` for uniqueness) and merge attributes (later values overwrite earlier ones).
4. Track which entity names appeared in which chunk via the `chunkEntityMap`.

What normalization would you apply to entity names for deduplication? Why is `.toLowerCase().trim()` a reasonable starting point but not sufficient for all cases?

### Domain-Specific Entity Extraction

For specialized domains, provide explicit entity type definitions and examples to improve extraction accuracy. Here is an example schema for a software engineering domain:

```typescript
// src/knowledge-graphs/domain-entities.ts

const SoftwareEntitySchema = z.object({
  entities: z.array(
    z.object({
      name: z.string(),
      type: z.enum([
        'developer',
        'team',
        'service',
        'api_endpoint',
        'database',
        'library',
        'repository',
        'deployment',
        'incident',
        'feature',
      ]),
      aliases: z.array(z.string()),
      attributes: z.record(z.string()),
    })
  ),
})

async function extractSoftwareEntities(text: string): Promise<z.infer<typeof SoftwareEntitySchema>['entities']>
```

Build `extractSoftwareEntities` with a detailed system prompt that describes each entity type and what attributes to look for. For example: "developer" should include `role` ("frontend", "backend"), "service" should include `language` and `framework`, "database" should include `type` ("postgresql", "redis").

Why does a domain-specific schema with detailed type descriptions produce better extraction than a generic one? What is the trade-off in terms of reusability?

> **Beginner Note:** Start with a small number of broad entity types (person, organization, concept). You can always refine your taxonomy later. Over-specifying entity types (e.g., distinguishing "senior_engineer" from "junior_engineer") makes extraction harder without proportionally improving retrieval.

> **Advanced Note:** Entity extraction quality improves dramatically with few-shot examples specific to your domain. Include 2-3 example texts with their expected entity outputs in the system prompt. This grounds the model in your specific conventions (e.g., whether "Auth Service" and "authentication-service" should be merged).

---

## Section 3: Relationship Extraction

### Subject-Predicate-Object Triples

Relationships are expressed as triples: (subject, predicate, object). "Alice manages Bob" becomes (Alice, manages, Bob). "Project Alpha uses TypeScript" becomes (Project Alpha, uses, TypeScript).

Define the schemas:

```typescript
// src/knowledge-graphs/relationship-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const RelationshipSchema = z.object({
  subject: z.string().describe('The source entity name'),
  predicate: z.string().describe('The relationship type (verb or phrase)'),
  object: z.string().describe('The target entity name'),
  confidence: z.number().min(0).max(1).describe('How confident the extraction is'),
  sourceText: z.string().describe('The text snippet this relationship was extracted from'),
  bidirectional: z.boolean().describe('Whether the relationship goes both ways'),
})

type Relationship = z.infer<typeof RelationshipSchema>

const RelationshipExtractionSchema = z.object({
  relationships: z.array(RelationshipSchema),
})

async function extractRelationships(text: string, knownEntities?: string[]): Promise<Relationship[]>
```

Build `extractRelationships` with a system prompt that enforces:

- Consistent predicate names (always "manages" not sometimes "manages" and sometimes "is manager of")
- Active voice predicates ("manages" not "is managed by")
- Both explicit and implicit relationships
- Confidence scoring based on how directly stated the relationship is
- Identification of bidirectional relationships (e.g., "collaborates_with")

If `knownEntities` is provided, include them in the prompt so the model knows what entities to look for.

What common relationship types would you include in the prompt as guidance? Think about hierarchy (manages, reports_to), collaboration (works_on), dependency (depends_on, uses), and authorship (authored, created).

### Joint Entity-Relationship Extraction

Extracting entities and relationships together in a single call ensures consistency -- the entity names in the relationships will match the entity list exactly.

```typescript
const EntityRelationshipSchema = z.object({
  entities: z.array(
    z.object({
      name: z.string(),
      type: z.string(),
      description: z.string(),
    })
  ),
  relationships: z.array(
    z.object({
      subject: z.string(),
      predicate: z.string(),
      object: z.string(),
      confidence: z.number().min(0).max(1),
    })
  ),
})

async function extractEntitiesAndRelationships(text: string): Promise<z.infer<typeof EntityRelationshipSchema>>
```

Build this with a system prompt that emphasizes: every entity mentioned in a relationship must also appear in the entities list, and names must match exactly between the two lists.

### Predicate Normalization

Without normalization, you end up with "manages," "is the manager of," "leads," and "oversees" all representing the same relationship. Build a two-tier normalization system:

```typescript
// src/knowledge-graphs/predicate-normalization.ts

const PREDICATE_MAP: Record<string, string> = {
  manages: 'manages',
  'is the manager of': 'manages',
  leads: 'manages',
  oversees: 'manages',
  supervises: 'manages',
  'reports to': 'reports_to',
  'reports into': 'reports_to',
  'works under': 'reports_to',
  // ... add more mappings
}

function normalizePredicateStatic(predicate: string): string
async function normalizePredicateLLM(predicate: string, context: string): Promise<string>
```

Build `normalizePredicateStatic` to look up the lowercased, trimmed predicate in `PREDICATE_MAP`. If not found, replace spaces with underscores as a fallback.

Build `normalizePredicateLLM` to try static normalization first, and only call the LLM when the predicate is not in the static map. Use a Zod schema with `normalized` (the canonical form) and `category` (hierarchy, collaboration, dependency, authorship, location, temporal, other).

When would you extend the static map vs rely on the LLM? What is the cost trade-off?

> **Beginner Note:** Start by extracting entities and relationships together in a single LLM call (the `extractEntitiesAndRelationships` function). This ensures consistency -- the model will use the same entity names in both the entity list and the relationships. Split extraction into separate steps only when your texts are too long for a single call.

> **Advanced Note:** Relationship extraction accuracy varies significantly by domain. For well-structured text (technical documentation, organizational charts), accuracy is high. For narrative text (meeting notes, emails), relationships are often implicit and extraction is noisier. Always review extracted relationships before adding them to your graph, especially for high-stakes applications.

---

## Section 4: Building a Simple Graph

### In-Memory Adjacency List

A knowledge graph can be represented as a simple adjacency list: a map from node IDs to their edges. No specialized graph database needed.

```typescript
// src/knowledge-graphs/graph.ts

interface GraphNode {
  id: string
  name: string
  type: string
  attributes: Record<string, string>
  aliases: string[]
}

interface GraphEdge {
  source: string // Node ID
  target: string // Node ID
  predicate: string
  weight: number // Confidence or frequency
  metadata: Record<string, string>
}
```

Build a `KnowledgeGraph` class with these private fields:

- `nodes: Map<string, GraphNode>` -- all nodes by ID
- `adjacencyList: Map<string, GraphEdge[]>` -- outgoing edges per node
- `reverseAdjacencyList: Map<string, GraphEdge[]>` -- incoming edges per node
- `aliasIndex: Map<string, string>` -- maps lowercased name/alias to node ID

Implement these methods:

```typescript
class KnowledgeGraph {
  addNode(node: GraphNode): void
  addEdge(edge: GraphEdge): void
  findNode(nameOrAlias: string): GraphNode | undefined
  getOutgoingEdges(nodeId: string): GraphEdge[]
  getIncomingEdges(nodeId: string): GraphEdge[]
  getNeighbors(nodeId: string): GraphNode[]
  getNodesByType(type: string): GraphNode[]
  getStats(): {
    nodeCount: number
    edgeCount: number
    nodeTypes: Record<string, number>
    predicateTypes: Record<string, number>
  }
  toJSON(): { nodes: GraphNode[]; edges: GraphEdge[] }
  static fromJSON(data: { nodes: GraphNode[]; edges: GraphEdge[] }): KnowledgeGraph
}
```

Key implementation details to think about:

- `addNode` should initialize empty edge lists in both adjacency maps and index the node's name and all aliases in `aliasIndex`.
- `addEdge` should validate that both source and target nodes exist. Check for duplicate edges (same source, target, and predicate) and merge by keeping the higher weight rather than adding a duplicate.
- `findNode` should normalize the input to lowercase and look up in `aliasIndex`.
- `getNeighbors` should follow outgoing edges and return the target nodes.
- `toJSON`/`fromJSON` enable persistence to disk.

Why does the graph need a reverse adjacency list? What queries does it enable that forward-only adjacency cannot answer?

### Building a Graph from Extracted Data

```typescript
// src/knowledge-graphs/build-graph.ts

import { KnowledgeGraph, type GraphNode } from './graph.js'
import { extractEntitiesAndRelationships } from './relationship-extraction.js'
import { normalizePredicateStatic } from './predicate-normalization.js'

function generateNodeId(name: string, type: string): string
async function buildGraphFromText(texts: string[]): Promise<KnowledgeGraph>
```

Build `generateNodeId` to produce IDs like `"person:alice_smith"` -- the type prefix avoids collisions between entities of different types that share names.

Build `buildGraphFromText` to:

1. Create an empty `KnowledgeGraph`.
2. For each text, call `extractEntitiesAndRelationships`.
3. Add extracted entities as nodes using `generateNodeId`.
4. For each relationship, find source and target nodes via `findNode`. If a node referenced in a relationship does not exist, create it with type `'unknown'`.
5. Add edges with normalized predicates and confidence as weight.
6. Log graph statistics at the end.

What should you do when a relationship references an entity that was not in the entity list? Is creating an "unknown" type node the best approach, or can you think of alternatives?

> **Beginner Note:** The node ID convention `type:normalized_name` (e.g., "person:alice_smith") makes it easy to look up nodes and avoid collisions between entities of different types that share names (e.g., a person named "Phoenix" and a city named "Phoenix").

---

## Section 5: Graph Traversal for Retrieval

### Traversal Algorithms

Once you have a graph, you can traverse it to answer relationship questions. The three most useful traversal patterns are:

1. **Neighborhood retrieval:** Get all nodes within N hops of a starting node
2. **Path finding:** Find how two nodes are connected
3. **Subgraph extraction:** Get all nodes and edges related to a query topic

```typescript
// src/knowledge-graphs/traversal.ts

import { KnowledgeGraph, type GraphNode, type GraphEdge } from './graph.js'

interface TraversalResult {
  nodes: GraphNode[]
  edges: GraphEdge[]
  paths: string[][] // Each path is a list of node IDs
}

function getNeighborhood(
  graph: KnowledgeGraph,
  startNodeId: string,
  maxHops?: number, // default 2
  predicateFilter?: string[]
): TraversalResult

function findShortestPath(
  graph: KnowledgeGraph,
  startId: string,
  endId: string,
  maxDepth?: number // default 5
): string[] | null

function extractSubgraph(
  graph: KnowledgeGraph,
  entityNames: string[],
  hops?: number // default 1
): TraversalResult

function traversalToText(result: TraversalResult): string
```

Build `getNeighborhood` using BFS:

1. Initialize a queue with the start node at depth 0 and a visited set.
2. For each dequeued node, collect it and its edges. If depth < maxHops, enqueue unvisited neighbors.
3. Traverse both outgoing and incoming edges for bidirectional discovery.
4. If `predicateFilter` is provided, skip edges whose predicate is not in the filter.

Build `findShortestPath` using BFS:

1. Queue entries are `{ nodeId, path }` where path is the list of node IDs visited so far.
2. When the target is dequeued, return the path. If path length exceeds `maxDepth`, skip.
3. Check both outgoing and incoming edges.
4. Return `null` if no path is found.

Build `extractSubgraph` to find each entity name in the graph, get its neighborhood, and merge the results (deduplicating nodes by ID and edges by a `source-predicate-target` key).

Build `traversalToText` to format the result for LLM consumption:

- List entities with their types and attributes
- List relationships in `"Alice --[manages]--> Bob"` format

Why is traversing both incoming and outgoing edges important for path finding? What kind of relationships would you miss with forward-only traversal?

> **Beginner Note:** Start with neighborhood retrieval (1-2 hops). Most relationship questions can be answered by looking at the immediate neighborhood of the mentioned entities. Path finding is useful for "how are X and Y connected?" questions but can be expensive on large graphs.

> **Advanced Note:** For large graphs (100K+ nodes), BFS traversal without bounds can be slow. Always set a `maxDepth` limit and consider using a priority queue (Dijkstra-like) that weights edges by confidence to find the most relevant paths first.

---

## Section 6: Graph RAG Pattern

### Combining Graph and Vector Retrieval

The Graph RAG pattern runs two retrieval strategies in parallel and merges the results:

1. **Vector retrieval:** Find text chunks relevant to the query (standard RAG)
2. **Graph retrieval:** Extract entities from the query, traverse the graph, get related context

The combined context gives the LLM both relevant text passages and structured relationship information.

First, build a function to extract entities from the user's query:

```typescript
// src/knowledge-graphs/graph-rag.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { KnowledgeGraph } from './graph.js'
import { extractSubgraph, traversalToText } from './traversal.js'

const QueryEntitiesSchema = z.object({
  entities: z.array(z.string()).describe('Entity names mentioned or implied in the query'),
  relationshipType: z.string().optional().describe('The type of relationship being asked about, if any'),
})

async function extractQueryEntities(query: string): Promise<z.infer<typeof QueryEntitiesSchema>>
```

Build this with a system prompt that includes examples:

- "Who manages Alice?" -> entities: ["Alice"], relationshipType: "manages"
- "What tech does Project Alpha use?" -> entities: ["Project Alpha"], relationshipType: "uses"

Then build the full Graph RAG pipeline:

```typescript
interface GraphRAGResult {
  answer: string
  vectorContext: string[]
  graphContext: string
  entitiesFound: string[]
}

async function graphRAG(
  query: string,
  graph: KnowledgeGraph,
  vectorSearch: (query: string, topK: number) => Promise<string[]>,
  options?: { vectorTopK?: number; graphHops?: number }
): Promise<GraphRAGResult>
```

Build `graphRAG` to:

1. Run vector search and entity extraction in parallel using `Promise.all`.
2. Use extracted entity names to call `extractSubgraph` on the graph.
3. Convert the subgraph to text with `traversalToText`.
4. Combine both contexts into a single prompt with clearly labeled sections ("=== Knowledge Graph Context ===" and "=== Document Context ===").
5. Call `generateText` with a system prompt instructing the model to use both contexts, noting any discrepancies between graph and document information.

Why run vector and graph retrieval in parallel rather than sequentially? What does each retrieval strategy contribute that the other cannot?

### Example Usage

To test your Graph RAG pipeline, create a set of sample documents about a fictional software organization:

```typescript
const documents = [
  `Alice Smith is the Engineering Director. She manages the Platform
team and the Product team. Alice reports to CEO Bob Johnson.`,

  `The Platform team is led by Charlie Davis. The team maintains
the authentication service (built with TypeScript and Hono) and
the data pipeline (built with Python). Charlie reports to Alice Smith.`,

  // ... more documents with cross-references
]
```

Build the graph with `buildGraphFromText(documents)`, create a simulated vector search that returns the raw documents, and test with multi-hop queries like "Who does Charlie Davis report to, and who does that person report to?"

> **Beginner Note:** Start by building the graph from a small set of documents (10-20) and testing queries manually. Inspect the extracted entities and relationships to verify quality before scaling up. Bad entity extraction leads to a bad graph, which leads to bad retrieval.

> **Advanced Note:** For production Graph RAG, consider pre-computing and caching common subgraphs. Entity neighborhoods for frequently queried entities can be materialized, avoiding repeated traversals. Also consider embedding the graph context itself and including it in the vector index for unified retrieval.

---

## Section 7: Maintaining Graph Consistency

### The Deduplication Problem

Entity extraction from multiple documents produces duplicates. "Alice Smith," "Alice," "A. Smith," and "Engineering Director Alice" might all refer to the same person. Without deduplication, your graph has four separate nodes for one entity, and relationships are fragmented.

You will build two strategies: string similarity (fast, free) and LLM-based resolution (slower, semantic).

### Strategy 1: String Similarity

```typescript
// src/knowledge-graphs/entity-resolution.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import type { GraphNode } from './graph.js'

function stringSimilarity(a: string, b: string): number
function levenshteinDistance(a: string, b: string): number
function findPotentialDuplicates(nodes: GraphNode[], threshold?: number): Array<[GraphNode, GraphNode, number]>
```

Build `stringSimilarity` with three tiers:

- Exact match (case-insensitive): return 1.0
- One string contains the other: return 0.8
- Otherwise: compute Levenshtein distance and return `1 - distance / maxLength`

Build `levenshteinDistance` using the classic dynamic programming matrix approach: create a 2D matrix of size `(b.length + 1) x (a.length + 1)`, initialize the first row and column with sequential indices, then fill in the rest using the min of insertion, deletion, and substitution costs.

Build `findPotentialDuplicates` to compare all pairs of nodes of the same type. Check both the node name and its aliases against the other node's name. Only compare nodes of the same type -- a person and a city should never be considered duplicates even if their names match.

Why is it important to only compare nodes of the same type? Can you think of a case where two nodes of different types with the same name should remain separate?

### Strategy 2: LLM-Based Entity Resolution

For ambiguous cases, use an LLM:

```typescript
const EntityResolutionSchema = z.object({
  areSameEntity: z.boolean(),
  confidence: z.number().min(0).max(1),
  canonicalName: z.string().describe('The best name to use if they are the same entity'),
  reasoning: z.string(),
})

async function llmEntityResolution(
  entity1: GraphNode,
  entity2: GraphNode,
  context?: string
): Promise<z.infer<typeof EntityResolutionSchema>>
```

Build this to present both entities (name, type, aliases, attributes) to the LLM and ask whether they refer to the same real-world thing.

### Graph Deduplication

Build a `deduplicateGraph` function that combines both strategies:

```typescript
async function deduplicateGraph(
  graph: KnowledgeGraph,
  useLLM?: boolean
): Promise<{
  merged: number
  mergeLog: Array<{ kept: string; removed: string; confidence: number }>
}>
```

This function should:

1. Collect all nodes from the graph.
2. Find potential duplicates with `findPotentialDuplicates` at threshold 0.75.
3. Auto-merge pairs with similarity >= 0.9.
4. For uncertain pairs (0.75-0.9), use LLM resolution if `useLLM` is true.
5. Only merge when confidence >= 0.8.
6. Log each merge decision.

In a full implementation, merging means: reassign all of node2's edges to node1, add node2's aliases to node1, and remove node2. What edge cases could arise during edge reassignment? (Hint: what if node2 has an edge to itself, or an edge to node1?)

> **Beginner Note:** String-based deduplication (exact match and substring match) catches most duplicates and costs nothing. Use it as a first pass. Reserve LLM-based entity resolution for cases where string matching is ambiguous (e.g., "JS" could be "JavaScript" or a person's initials).

> **Advanced Note:** Entity resolution is an active research area. For production systems, consider using embedding similarity between entity names and descriptions in addition to string matching. You can also use transitivity: if A matches B and B matches C, then A likely matches C (but verify). Graph databases like Neo4j have built-in entity resolution capabilities.

---

## Section 8: When to Use Graph RAG vs Vector RAG

### Decision Framework

Not every application needs a knowledge graph. Vector RAG is simpler, cheaper, and sufficient for many use cases. Graph RAG adds value in specific scenarios.

```typescript
// src/knowledge-graphs/decision-framework.ts

interface UseCase {
  description: string
  queryExamples: string[]
  recommendedApproach: 'vector_rag' | 'graph_rag' | 'hybrid'
  reasoning: string
}

const useCases: UseCase[] = [
  {
    description: 'Customer support knowledge base',
    queryExamples: ['How do I reset my password?', 'What is the refund policy?', 'How to configure SSO?'],
    recommendedApproach: 'vector_rag',
    reasoning: 'Questions are self-contained and answered by single passages. No relationship reasoning needed.',
  },
  {
    description: 'Internal wiki with organizational data',
    queryExamples: [
      'Who owns the billing service?',
      'What teams depend on the auth service?',
      'Who approved the Q3 budget?',
    ],
    recommendedApproach: 'graph_rag',
    reasoning: 'Questions involve ownership, dependencies, and approval chains -- all relationship queries.',
  },
  {
    description: 'Technical documentation for a microservices architecture',
    queryExamples: [
      'How does the payment flow work?',
      'What services will be affected if the user service goes down?',
      'What are the API endpoints for authentication?',
    ],
    recommendedApproach: 'hybrid',
    reasoning: 'Mix of content questions (how does it work?) and relationship questions (what depends on what?).',
  },
  {
    description: 'Legal document analysis',
    queryExamples: [
      'What clauses mention indemnification?',
      'Which parties are bound by the NDA?',
      'What obligations does Party A have?',
    ],
    recommendedApproach: 'hybrid',
    reasoning:
      'Need both text retrieval (find relevant clauses) and entity-relationship understanding (who is obligated to whom).',
  },
  {
    description: 'Research paper Q&A',
    queryExamples: [
      'What methods did they use?',
      'What were the key findings?',
      'How does this compare to prior work?',
    ],
    recommendedApproach: 'vector_rag',
    reasoning:
      'Questions are answered by text passages. Citation tracking could benefit from a graph, but is not required for basic Q&A.',
  },
]

// Cost-benefit analysis
const costBenefitMatrix = {
  vector_rag: {
    setupComplexity: 'Low',
    maintenanceCost: 'Low',
    queryLatency: 'Low (single retrieval)',
    bestFor: 'Content questions, similarity search',
    limitations: 'Cannot answer relationship or multi-hop questions',
  },
  graph_rag: {
    setupComplexity: 'High (entity extraction, graph building)',
    maintenanceCost: 'Medium (graph updates, deduplication)',
    queryLatency: 'Medium (entity extraction + traversal + generation)',
    bestFor: 'Relationship questions, multi-hop reasoning, dependency tracking',
    limitations: 'Expensive to build and maintain, extraction quality varies',
  },
  hybrid: {
    setupComplexity: 'High',
    maintenanceCost: 'High',
    queryLatency: 'Higher (parallel retrieval + merge)',
    bestFor: 'Complex applications with mixed query types',
    limitations: 'Most expensive, requires careful tuning of merge strategy',
  },
}

export { useCases, costBenefitMatrix }
```

### Implementation Checklist

Before building Graph RAG, verify these prerequisites:

1. **Your queries actually need relationships.** If all queries are answered by single passages, vector RAG is sufficient. Audit your actual user queries before deciding.

2. **Your documents contain extractable entities.** Free-form creative writing is hard to extract entities from. Structured documents (technical docs, organizational docs, reports with named entities) are ideal.

3. **Entity extraction quality is sufficient.** Run entity extraction on 10-20 sample documents and manually verify. If extraction accuracy is below 80%, the graph will be noisy and unreliable.

4. **You can afford the maintenance cost.** Graphs need updating when documents change. Entity resolution needs periodic review. This is ongoing work, not a one-time setup.

> **Beginner Note:** A strong signal that you need Graph RAG is when users repeatedly ask relationship questions and your vector RAG pipeline returns relevant documents but cannot synthesize the answer. If users ask "Who owns the billing service?" and the pipeline returns the billing service documentation (which mentions the owner) but cannot extract the answer reliably, a graph will help.

> **Advanced Note:** An alternative to building an explicit knowledge graph is to use structured metadata in your vector store. Tag each chunk with its entities and relationships as metadata, then use metadata filtering to simulate graph queries. This "implicit graph" approach is simpler than a full graph but supports basic relationship queries. It falls short for multi-hop queries and path finding.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: LSP as an Implicit Knowledge Graph

(See Module 10 Section 11 for LSP background.)

Language Server Protocol provides an implicit knowledge graph over code that you never have to build or maintain. The graph is always current because it is computed directly from the source code:

- **Entities:** Functions, classes, interfaces, variables, types, modules
- **Relationships:** Calls, imports, extends, implements, references, exports
- **Traversal:** Go-to-definition, find-all-references, call hierarchy, type hierarchy
- **Maintenance:** Zero — the graph is recomputed from code on every query

Unlike the knowledge graphs taught in this module, which must be built through entity extraction, relationship mapping, and ongoing maintenance, the LSP graph is derived from the source of truth (the code itself). When code changes, the graph updates automatically. There is no extraction pipeline, no entity resolution, no stale data.

```typescript
// An LSP provides graph queries natively
const lspQueries = {
  // Entity lookup
  getDefinition: (symbol: string) => '→ exact location of the definition',
  // Relationship traversal (1-hop)
  findReferences: (symbol: string) => '→ all files and lines that reference this symbol',
  // Multi-hop traversal
  getCallHierarchy: (fn: string) => '→ who calls this function, and who calls those callers',
  // Type relationships
  getTypeHierarchy: (type: string) => '→ what extends/implements this type',
}
```

This is an important design lesson: when possible, derive graphs from authoritative sources rather than maintaining them separately. An LSP graph over code is always correct. A manually built knowledge graph over the same code would be stale by the next commit.

The trade-off is flexibility. LSP graphs only know about code structure — they cannot capture business rules, architectural decisions, or domain concepts that are not expressed in the code. For those, you need explicit knowledge graphs built from documentation and human input.

---

## Section 10: When Graphs Are Free

Some domains provide graph structure inherently — you do not have to build it. Recognizing these "free graphs" saves you the cost and complexity of explicit graph construction:

| Domain             | Free Graph Source          | Entities                       | Relationships                    |
| ------------------ | -------------------------- | ------------------------------ | -------------------------------- |
| **Code**           | LSP / AST                  | Functions, classes, modules    | Calls, imports, extends          |
| **Databases**      | Schema / catalog           | Tables, columns, views         | Foreign keys, joins, constraints |
| **APIs**           | OpenAPI / GraphQL specs    | Endpoints, schemas, parameters | Request/response, dependencies   |
| **Packages**       | Package manager            | Libraries, versions            | Dependencies, peer dependencies  |
| **Infrastructure** | Terraform / CloudFormation | Services, resources            | Connections, permissions         |

For these domains, the graph already exists in a structured, machine-readable format. You can query it directly rather than building a knowledge graph from unstructured text. The entity extraction and relationship mapping from this module are needed when the graph does not already exist — when you are working with documents, reports, emails, or other unstructured content.

The practical decision framework is:

1. **Check for an implicit graph first.** Does your domain have a structured schema, protocol, or specification that encodes entities and relationships?
2. **If yes, use it directly.** Write query adapters that translate graph questions into the native query language (SQL for databases, LSP for code, API calls for service catalogs).
3. **If no, build an explicit graph.** Use the entity extraction and relationship mapping techniques from this module to construct one from unstructured data.

> **Key Insight:** The most expensive part of a knowledge graph is not the storage or traversal — it is the construction and maintenance. When you can derive the graph from a structured source, you eliminate the hardest part of the problem.

---

## Quiz

### Question 1 (Easy)

What is the main limitation of vector search that knowledge graphs address?

- A) Vector search is too slow
- B) Vector search cannot traverse relationships between entities — it finds similar text but cannot answer questions about how entities are connected
- C) Vector search requires too much storage
- D) Vector search only works with English text

**Answer: B** — Vector search finds documents that are semantically similar to a query, but it has no concept of relationships between entities. Questions like "Who manages Alice?" or "What services depend on the auth service?" require traversing relationships (edges) between entities (nodes), which is exactly what knowledge graphs provide.

---

### Question 2 (Easy)

In a knowledge graph triple (subject, predicate, object), what does the predicate represent?

- A) The starting entity
- B) The ending entity
- C) The relationship type between the two entities
- D) The confidence score of the extraction

**Answer: C** — A triple like (Alice, manages, Bob) has Alice as the subject, "manages" as the predicate (the relationship), and Bob as the object. The predicate defines the type and direction of the relationship between two entities. Common predicates include "manages," "works_on," "depends_on," and "authored."

---

### Question 3 (Medium)

Why is predicate normalization important when building a knowledge graph?

- A) It makes the graph smaller
- B) Without normalization, the same relationship might be stored as "manages," "is the manager of," and "oversees" — making graph traversal miss valid connections
- C) It speeds up entity extraction
- D) It is required by all graph databases

**Answer: B** — If Alice "manages" Bob in one document and "oversees" Bob in another, without normalization these are two different edge types. A query for "who does Alice manage?" would miss the "oversees" edge. Predicate normalization maps synonymous relationship names to a canonical form, ensuring consistent traversal.

---

### Question 4 (Medium)

In the Graph RAG pattern, why run vector retrieval and graph retrieval in parallel rather than sequentially?

- A) Parallel is always faster
- B) They retrieve complementary information — vector retrieval finds relevant text passages while graph retrieval finds entity relationships. Neither subsumes the other, and both contribute to a complete answer
- C) Graph retrieval requires vector results as input
- D) Sequential retrieval causes errors

**Answer: B** — Vector retrieval and graph retrieval answer different aspects of a query. Vector retrieval finds detailed text passages that describe topics in depth. Graph retrieval finds structured relationship information (who manages whom, what depends on what). Running them in parallel is both faster and conceptually correct — they are independent retrieval strategies that produce complementary context for the LLM.

---

### Question 5 (Hard)

You have a knowledge base of 500 technical documents. Users mostly ask "how-to" questions ("How do I configure X?"), but 20% of queries are about service dependencies ("What happens if service Y goes down?"). What is the most cost-effective approach?

- A) Build a full Graph RAG system for all queries
- B) Use vector RAG only — the 20% dependency queries are not worth the graph complexity
- C) Use vector RAG as the primary system and add a lightweight entity-relationship index just for service dependencies, queried only when the question pattern matches
- D) Use graph RAG only — it can handle both types of queries

**Answer: C** — A full Graph RAG system for all 500 documents is expensive and unnecessary for "how-to" queries. Ignoring the 20% dependency queries leaves a significant gap. The optimal approach is vector RAG as the default (handles 80% of queries well) with a targeted graph just for service dependency relationships (handles the 20%). A router or classifier can detect dependency-type questions and activate graph retrieval only when needed, keeping costs low while improving quality where it matters most.

### Question 6 (Medium)

What advantage does an LSP-derived knowledge graph have over a manually constructed one for code?

a) LSP graphs support more entity types
b) LSP graphs are always correct and never stale because they are computed directly from the source code, eliminating the extraction pipeline and maintenance burden
c) LSP graphs are stored more efficiently
d) LSP graphs can represent business rules that code cannot express

**Answer: B**

**Explanation:** An LSP-derived graph is computed from the source of truth (the code itself). When code changes, the graph updates automatically — there is no extraction pipeline to run, no entity resolution to maintain, and no risk of stale data. A manually constructed knowledge graph requires re-extraction whenever code changes and can drift from reality between updates. The trade-off is flexibility: LSP only captures code structure, not business rules or architectural decisions.

---

### Question 7 (Hard)

You are deciding whether to build an explicit knowledge graph for a new domain. The domain has a well-defined API specification (OpenAPI) that describes all endpoints, schemas, and their relationships. What should you do?

a) Build a knowledge graph from the API documentation using entity extraction
b) Ignore the API specification and use vector search only
c) Use the OpenAPI spec directly as an implicit graph — it already encodes entities (endpoints, schemas) and relationships (request/response, dependencies) in a structured, machine-readable format
d) Convert the OpenAPI spec to unstructured text before building the graph

**Answer: C**

**Explanation:** The OpenAPI specification is an implicit graph that already encodes the entities and relationships you would extract. Endpoints are nodes, request/response schemas define edges, and dependencies between schemas capture relationships. Writing query adapters that translate graph questions into spec lookups is far cheaper and more reliable than building an explicit graph from the same information. The general principle: check for implicit graphs first, and only build explicit ones when no structured source exists.

---

## Exercises

### Exercise 1: Entity and Relationship Extraction Pipeline

**Objective:** Build an entity and relationship extraction pipeline that processes documents and builds a knowledge graph.

**Specification:**

1. Create `src/exercises/m12/ex01-knowledge-graph.ts`
2. Create a set of 5-10 sample documents describing a fictional software organization (teams, people, services, projects, dependencies)
3. Implement entity extraction using `generateText` with `Output.object` and a domain-specific schema
4. Implement relationship extraction as subject-predicate-object triples
5. Build an in-memory `KnowledgeGraph` from the extracted data
6. Implement graph traversal to answer these questions:
   - "Who manages [person]?" (1-hop traversal)
   - "What services does [project] depend on?" (1-hop traversal)
   - "How are [person1] and [person2] connected?" (path finding)
7. Print graph statistics and query results

**Expected output format:**

```
=== Knowledge Graph ===
Nodes: 24 (8 person, 4 team, 6 service, 4 project, 2 technology)
Edges: 35
Predicates: manages(6), works_on(8), depends_on(7), uses(5), reports_to(4), authored(5)

=== Query: Who manages Charlie Davis? ===
Traversal: charlie_davis -> [reports_to] -> alice_smith
Answer: Charlie Davis reports to Alice Smith (Engineering Director)

=== Query: What does Project Alpha depend on? ===
Traversal: project_alpha -> [depends_on] -> auth_service, embedding_service, postgresql
Answer: Project Alpha depends on: Authentication Service, Embedding Service, PostgreSQL
```

**Test specification:**

```typescript
// tests/exercises/m12/ex01-knowledge-graph.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 12: Knowledge Graph', () => {
  it('should extract entities from documents', async () => {
    const entities = await extractEntities(sampleDocuments)
    expect(entities.length).toBeGreaterThan(5)
  })

  it('should extract relationships', async () => {
    const relationships = await extractRelationships(sampleDocuments)
    expect(relationships.length).toBeGreaterThan(3)
  })

  it('should build a navigable graph', () => {
    const graph = buildGraph(entities, relationships)
    const stats = graph.getStats()
    expect(stats.nodeCount).toBeGreaterThan(0)
    expect(stats.edgeCount).toBeGreaterThan(0)
  })

  it('should answer 1-hop relationship queries', () => {
    const neighbors = graph.getNeighbors('person:charlie_davis')
    expect(neighbors.length).toBeGreaterThan(0)
  })

  it('should find paths between entities', () => {
    const path = findShortestPath(graph, 'person:charlie_davis', 'person:bob_johnson')
    expect(path).not.toBeNull()
    expect(path!.length).toBeGreaterThan(1)
  })
})
```

---

### Exercise 2: Graph-Augmented Retrieval

**Objective:** Combine graph traversal with vector search in a Graph RAG pipeline and compare it to vector-only retrieval.

**Specification:**

1. Create `src/exercises/m12/ex02-graph-rag.ts`
2. Build a knowledge graph from the sample documents (from Exercise 1)
3. Embed the document chunks in a vector store (simulated in-memory)
4. Implement Graph RAG that:
   - Extracts entities from the query
   - Runs vector search for text context
   - Traverses the graph for relationship context
   - Combines both contexts for the LLM
5. Create a test suite with two query types:
   - **Content queries** that vector RAG handles well
   - **Relationship queries** that need graph context
6. Show that Graph RAG outperforms vector-only RAG on relationship queries

**Test specification:**

```typescript
// tests/exercises/m12/ex02-graph-rag.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 12: Graph RAG', () => {
  it('should combine vector and graph context', async () => {
    const result = await graphRAG('Who manages the Platform team?', graph, vectorSearch)
    expect(result.vectorContext.length).toBeGreaterThan(0)
    expect(result.graphContext).toBeTruthy()
    expect(result.answer).toBeTruthy()
  })

  it('should find entities in the query', async () => {
    const entities = await extractQueryEntities('What does Project Alpha depend on?')
    expect(entities.entities).toContain('Project Alpha')
  })

  it('should outperform vector-only on relationship queries', async () => {
    const relationshipQueries = [
      'Who manages Charlie Davis?',
      'What services does Project Alpha depend on?',
      'Which team maintains the authentication service?',
    ]

    for (const query of relationshipQueries) {
      const graphResult = await graphRAG(query, graph, vectorSearch)
      const vectorResult = await vectorOnlyRAG(query, vectorSearch)

      // Graph RAG should include entity names that vector RAG misses
      expect(graphResult.entitiesFound.length).toBeGreaterThan(0)
    }
  })
})
```

> **Local Alternative (Ollama):** Entity extraction and relationship mapping use `generateText` with `Output.object` and Zod schemas — this works with `ollama('qwen3.5')`. Knowledge graph construction and graph-augmented retrieval are model-agnostic patterns. Cloud models (`qwen3.5:cloud`) will produce better entity extraction for complex documents.

---

## Summary

In this module, you learned:

1. **Why knowledge graphs:** Vector search finds similar text but cannot traverse relationships between entities. Knowledge graphs store and query structured relationships.
2. **Entity extraction:** LLMs extract entities (people, organizations, projects, technologies) from documents with structured output, including aliases and attributes.
3. **Relationship extraction:** Subject-predicate-object triples capture how entities are connected, with predicate normalization ensuring consistency.
4. **Building a graph:** An in-memory adjacency list with forward and reverse edges supports efficient traversal without a specialized graph database.
5. **Graph traversal:** Neighborhood retrieval, path finding, and subgraph extraction answer multi-hop relationship questions.
6. **Graph RAG:** Combining graph traversal with vector search provides both relationship context and text content for comprehensive answers.
7. **Graph consistency:** Entity deduplication and resolution handle the inevitable duplicate entities from multi-document extraction.
8. **Decision framework:** Use vector RAG for content questions, graph RAG for relationship questions, and hybrid for applications with both.
9. **Implicit knowledge graphs:** LSP provides a zero-maintenance knowledge graph over code — entities, relationships, and traversal derived directly from the source, always up-to-date.
10. **When graphs are free:** Code (LSP), databases (schema), and APIs (OpenAPI) provide inherent graph structure. Recognize these implicit graphs before investing in explicit graph construction from unstructured data.

In Module 13, you will extend your pipeline to handle multi-modal inputs — images, diagrams, screenshots, and audio — adding visual understanding to your retrieval and generation capabilities.
