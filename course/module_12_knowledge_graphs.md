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

> **Advanced Note:** The boundary between "graph question" and "vector question" is blurry. A well-structured RAG pipeline with good chunking and metadata can handle some relationship queries through careful prompt engineering. The decision to add a knowledge graph should be driven by observed failure patterns — if your users consistently ask relationship questions that your RAG pipeline cannot answer, a graph will help.

---

## Section 2: Entity Extraction with LLMs

### Defining Entities

An entity is a distinct, identifiable thing: a person, organization, project, technology, concept, or event. Entity extraction is the first step in building a knowledge graph — you cannot create relationships until you know what the entities are.

```typescript
// src/knowledge-graphs/entity-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Define entity types for your domain
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

async function extractEntities(text: string, domainContext?: string): Promise<Entity[]> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: EntityExtractionSchema }),
    system: `You are an entity extraction expert. Extract all distinct
entities from the text. For each entity:
- Use the most complete, canonical form of the name
- List all aliases (abbreviations, nicknames, alternative spellings)
- Classify by type
- Add relevant attributes as key-value pairs

${domainContext ? `Domain context: ${domainContext}` : ''}

Be thorough: extract even implicitly referenced entities.
Deduplicate: if the same entity is mentioned multiple ways, merge them.`,
    messages: [{ role: 'user', content: text }],
    temperature: 0,
  })

  return output.entities
}

// Batch extraction for processing multiple chunks
async function extractEntitiesFromChunks(
  chunks: string[],
  domainContext?: string
): Promise<{
  entities: Entity[]
  chunkEntityMap: Map<number, string[]> // chunk index -> entity names
}> {
  const allEntities: Entity[] = []
  const chunkEntityMap = new Map<number, string[]>()
  const seenEntities = new Map<string, Entity>() // name -> entity

  for (let i = 0; i < chunks.length; i++) {
    const entities = await extractEntities(chunks[i], domainContext)

    const chunkEntityNames: string[] = []

    for (const entity of entities) {
      const normalized = entity.name.toLowerCase().trim()

      if (seenEntities.has(normalized)) {
        // Merge aliases and attributes
        const existing = seenEntities.get(normalized)!
        existing.aliases = [...new Set([...existing.aliases, ...entity.aliases])]
        existing.attributes = {
          ...existing.attributes,
          ...entity.attributes,
        }
      } else {
        seenEntities.set(normalized, entity)
        allEntities.push(entity)
      }

      chunkEntityNames.push(entity.name)
    }

    chunkEntityMap.set(i, chunkEntityNames)
    console.log(`Chunk ${i + 1}/${chunks.length}: found ${entities.length} entities`)
  }

  return { entities: allEntities, chunkEntityMap }
}

export { extractEntities, extractEntitiesFromChunks, type Entity, type EntityType }
```

### Domain-Specific Entity Extraction

For specialized domains, provide explicit entity type definitions and examples to improve extraction accuracy.

```typescript
// src/knowledge-graphs/domain-entities.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Example: Software engineering domain
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

async function extractSoftwareEntities(text: string): Promise<z.infer<typeof SoftwareEntitySchema>['entities']> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: SoftwareEntitySchema }),
    system: `You are a software engineering knowledge graph builder.
Extract entities from the text. Entity types:

- developer: A person who writes code (include role: "frontend", "backend", etc.)
- team: A group of developers (include team_lead if mentioned)
- service: A running software service (include language, framework)
- api_endpoint: A specific API route (include method: GET/POST/etc.)
- database: A data store (include type: "postgresql", "redis", etc.)
- library: A software dependency (include version if mentioned)
- repository: A code repository (include platform: "github", etc.)
- deployment: A deployed environment (include env: "prod", "staging")
- incident: A production issue (include severity, date)
- feature: A product feature (include status: "shipped", "in-progress")

Be precise with naming. Use the most specific type available.`,
    messages: [{ role: 'user', content: text }],
    temperature: 0,
  })

  return output.entities
}

export { extractSoftwareEntities }
```

> **Beginner Note:** Start with a small number of broad entity types (person, organization, concept). You can always refine your taxonomy later. Over-specifying entity types (e.g., distinguishing "senior_engineer" from "junior_engineer") makes extraction harder without proportionally improving retrieval.

> **Advanced Note:** Entity extraction quality improves dramatically with few-shot examples specific to your domain. Include 2-3 example texts with their expected entity outputs in the system prompt. This grounds the model in your specific conventions (e.g., whether "Auth Service" and "authentication-service" should be merged).

---

## Section 3: Relationship Extraction

### Subject-Predicate-Object Triples

Relationships are expressed as triples: (subject, predicate, object). "Alice manages Bob" becomes (Alice, manages, Bob). "Project Alpha uses TypeScript" becomes (Project Alpha, uses, TypeScript).

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

async function extractRelationships(text: string, knownEntities?: string[]): Promise<Relationship[]> {
  const entityGuidance = knownEntities ? `\n\nKnown entities to look for: ${knownEntities.join(', ')}` : ''

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: RelationshipExtractionSchema }),
    system: `You are a relationship extraction expert. Extract all
relationships between entities in the text as subject-predicate-object
triples.

Guidelines:
- Use consistent predicate names (e.g., always "manages" not sometimes
  "manages" and sometimes "is manager of")
- Prefer active voice predicates: "manages" not "is managed by"
- Extract both explicit and implicit relationships
- Set confidence based on how directly stated the relationship is
- Mark bidirectional relationships (e.g., "collaborates_with")
${entityGuidance}

Common relationship types:
- manages, reports_to, works_on, belongs_to
- depends_on, uses, implements, extends
- authored, created, reviewed, approved
- located_in, part_of, related_to`,
    messages: [{ role: 'user', content: text }],
    temperature: 0,
  })

  return output.relationships
}

// Extract entities AND relationships together for consistency
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

async function extractEntitiesAndRelationships(text: string): Promise<z.infer<typeof EntityRelationshipSchema>> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: EntityRelationshipSchema }),
    system: `Extract all entities and relationships from the text.
Ensure every entity mentioned in a relationship also appears in
the entities list. Use consistent naming — if an entity appears
in both the entities list and a relationship, the names must match
exactly.`,
    messages: [{ role: 'user', content: text }],
    temperature: 0,
  })

  return output
}

export { extractRelationships, extractEntitiesAndRelationships, type Relationship }
```

### Predicate Normalization

Without normalization, you end up with "manages," "is the manager of," "leads," and "oversees" all representing the same relationship. Predicate normalization maps variations to canonical forms.

```typescript
// src/knowledge-graphs/predicate-normalization.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Static normalization for known predicates
const PREDICATE_MAP: Record<string, string> = {
  manages: 'manages',
  'is the manager of': 'manages',
  leads: 'manages',
  oversees: 'manages',
  supervises: 'manages',
  'reports to': 'reports_to',
  'reports into': 'reports_to',
  'works under': 'reports_to',
  'works on': 'works_on',
  'contributes to': 'works_on',
  'is assigned to': 'works_on',
  uses: 'uses',
  utilizes: 'uses',
  'depends on': 'depends_on',
  requires: 'depends_on',
  'is built with': 'uses',
}

function normalizePredicateStatic(predicate: string): string {
  const lower = predicate.toLowerCase().trim()
  return PREDICATE_MAP[lower] ?? lower.replace(/\s+/g, '_')
}

// LLM-based normalization for unknown predicates
const NormalizedPredicateSchema = z.object({
  normalized: z.string().describe('The canonical predicate form (lowercase, underscored)'),
  category: z.enum(['hierarchy', 'collaboration', 'dependency', 'authorship', 'location', 'temporal', 'other']),
})

async function normalizePredicateLLM(predicate: string, context: string): Promise<string> {
  // Try static first
  const staticResult = normalizePredicateStatic(predicate)
  if (PREDICATE_MAP[predicate.toLowerCase()]) {
    return staticResult
  }

  // Fall back to LLM
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: NormalizedPredicateSchema }),
    system: `Normalize this relationship predicate to a canonical form.
Use lowercase with underscores. Prefer common predicates:
manages, reports_to, works_on, depends_on, uses, authored,
created, part_of, located_in, related_to

Context for the relationship: ${context}`,
    messages: [{ role: 'user', content: `Predicate: "${predicate}"` }],
    temperature: 0,
  })

  return output.normalized
}

export { normalizePredicateStatic, normalizePredicateLLM }
```

> **Beginner Note:** Start by extracting entities and relationships together in a single LLM call (the `extractEntitiesAndRelationships` function). This ensures consistency — the model will use the same entity names in both the entity list and the relationships. Split extraction into separate steps only when your texts are too long for a single call.

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

class KnowledgeGraph {
  private nodes: Map<string, GraphNode> = new Map()
  private adjacencyList: Map<string, GraphEdge[]> = new Map()
  private reverseAdjacencyList: Map<string, GraphEdge[]> = new Map()
  private aliasIndex: Map<string, string> = new Map() // alias -> node ID

  // Add a node
  addNode(node: GraphNode): void {
    this.nodes.set(node.id, node)

    if (!this.adjacencyList.has(node.id)) {
      this.adjacencyList.set(node.id, [])
    }
    if (!this.reverseAdjacencyList.has(node.id)) {
      this.reverseAdjacencyList.set(node.id, [])
    }

    // Index aliases
    this.aliasIndex.set(node.name.toLowerCase(), node.id)
    for (const alias of node.aliases) {
      this.aliasIndex.set(alias.toLowerCase(), node.id)
    }
  }

  // Add an edge
  addEdge(edge: GraphEdge): void {
    // Ensure both nodes exist
    if (!this.nodes.has(edge.source) || !this.nodes.has(edge.target)) {
      console.warn(`Skipping edge: missing node (${edge.source} -> ${edge.target})`)
      return
    }

    // Add to forward adjacency
    const forward = this.adjacencyList.get(edge.source)
    if (forward) {
      // Check for duplicate edges
      const existing = forward.find(e => e.target === edge.target && e.predicate === edge.predicate)
      if (existing) {
        // Merge: increase weight
        existing.weight = Math.max(existing.weight, edge.weight)
        return
      }
      forward.push(edge)
    }

    // Add to reverse adjacency (for inbound queries)
    const reverse = this.reverseAdjacencyList.get(edge.target)
    if (reverse) {
      reverse.push(edge)
    }
  }

  // Find a node by name or alias
  findNode(nameOrAlias: string): GraphNode | undefined {
    const normalized = nameOrAlias.toLowerCase()
    const nodeId = this.aliasIndex.get(normalized)
    return nodeId ? this.nodes.get(nodeId) : undefined
  }

  // Get outgoing edges from a node
  getOutgoingEdges(nodeId: string): GraphEdge[] {
    return this.adjacencyList.get(nodeId) ?? []
  }

  // Get incoming edges to a node
  getIncomingEdges(nodeId: string): GraphEdge[] {
    return this.reverseAdjacencyList.get(nodeId) ?? []
  }

  // Get neighbors (nodes connected by outgoing edges)
  getNeighbors(nodeId: string): GraphNode[] {
    const edges = this.getOutgoingEdges(nodeId)
    return edges.map(e => this.nodes.get(e.target)).filter((n): n is GraphNode => n !== undefined)
  }

  // Get all nodes of a specific type
  getNodesByType(type: string): GraphNode[] {
    const results: GraphNode[] = []
    for (const node of this.nodes.values()) {
      if (node.type === type) results.push(node)
    }
    return results
  }

  // Get statistics
  getStats(): {
    nodeCount: number
    edgeCount: number
    nodeTypes: Record<string, number>
    predicateTypes: Record<string, number>
  } {
    const nodeTypes: Record<string, number> = {}
    for (const node of this.nodes.values()) {
      nodeTypes[node.type] = (nodeTypes[node.type] ?? 0) + 1
    }

    const predicateTypes: Record<string, number> = {}
    let totalEdges = 0
    for (const edges of this.adjacencyList.values()) {
      for (const edge of edges) {
        predicateTypes[edge.predicate] = (predicateTypes[edge.predicate] ?? 0) + 1
        totalEdges++
      }
    }

    return {
      nodeCount: this.nodes.size,
      edgeCount: totalEdges,
      nodeTypes,
      predicateTypes,
    }
  }

  // Serialize to JSON
  toJSON(): {
    nodes: GraphNode[]
    edges: GraphEdge[]
  } {
    const edges: GraphEdge[] = []
    for (const edgeList of this.adjacencyList.values()) {
      edges.push(...edgeList)
    }
    return {
      nodes: [...this.nodes.values()],
      edges,
    }
  }

  // Load from JSON
  static fromJSON(data: { nodes: GraphNode[]; edges: GraphEdge[] }): KnowledgeGraph {
    const graph = new KnowledgeGraph()
    for (const node of data.nodes) {
      graph.addNode(node)
    }
    for (const edge of data.edges) {
      graph.addEdge(edge)
    }
    return graph
  }
}

export { KnowledgeGraph, type GraphNode, type GraphEdge }
```

### Building a Graph from Extracted Data

```typescript
// src/knowledge-graphs/build-graph.ts

import { KnowledgeGraph, type GraphNode } from './graph.js'
import { extractEntitiesAndRelationships } from './relationship-extraction.js'
import { normalizePredicateStatic } from './predicate-normalization.js'

function generateNodeId(name: string, type: string): string {
  return `${type}:${name.toLowerCase().replace(/\s+/g, '_')}`
}

async function buildGraphFromText(texts: string[]): Promise<KnowledgeGraph> {
  const graph = new KnowledgeGraph()

  for (let i = 0; i < texts.length; i++) {
    console.log(`Processing text ${i + 1}/${texts.length}...`)

    const { entities, relationships } = await extractEntitiesAndRelationships(texts[i])

    // Add entities as nodes
    for (const entity of entities) {
      const nodeId = generateNodeId(entity.name, entity.type)
      const node: GraphNode = {
        id: nodeId,
        name: entity.name,
        type: entity.type,
        attributes: {},
        aliases: [],
      }
      graph.addNode(node)
    }

    // Add relationships as edges
    for (const rel of relationships) {
      // Find or create source and target nodes
      const sourceNode = graph.findNode(rel.subject)
      const targetNode = graph.findNode(rel.object)

      if (!sourceNode || !targetNode) {
        // Create missing nodes with generic type
        if (!sourceNode) {
          graph.addNode({
            id: generateNodeId(rel.subject, 'unknown'),
            name: rel.subject,
            type: 'unknown',
            attributes: {},
            aliases: [],
          })
        }
        if (!targetNode) {
          graph.addNode({
            id: generateNodeId(rel.object, 'unknown'),
            name: rel.object,
            type: 'unknown',
            attributes: {},
            aliases: [],
          })
        }
      }

      const sourceId = sourceNode?.id ?? generateNodeId(rel.subject, 'unknown')
      const targetId = targetNode?.id ?? generateNodeId(rel.object, 'unknown')

      graph.addEdge({
        source: sourceId,
        target: targetId,
        predicate: normalizePredicateStatic(rel.predicate),
        weight: rel.confidence,
        metadata: { sourceChunk: String(i) },
      })
    }
  }

  const stats = graph.getStats()
  console.log(`\nGraph built:`)
  console.log(`  Nodes: ${stats.nodeCount}`)
  console.log(`  Edges: ${stats.edgeCount}`)
  console.log(`  Node types: ${JSON.stringify(stats.nodeTypes)}`)
  console.log(`  Predicates: ${JSON.stringify(stats.predicateTypes)}`)

  return graph
}

export { buildGraphFromText, generateNodeId }
```

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

// Get all nodes within N hops of a starting node
function getNeighborhood(
  graph: KnowledgeGraph,
  startNodeId: string,
  maxHops: number = 2,
  predicateFilter?: string[]
): TraversalResult {
  const visitedNodes = new Set<string>()
  const collectedEdges: GraphEdge[] = []
  const collectedNodes: GraphNode[] = []
  const queue: Array<{ nodeId: string; depth: number }> = [{ nodeId: startNodeId, depth: 0 }]

  while (queue.length > 0) {
    const { nodeId, depth } = queue.shift()!

    if (visitedNodes.has(nodeId)) continue
    visitedNodes.add(nodeId)

    const node = graph.findNode(nodeId)
    if (node) collectedNodes.push(node)

    if (depth >= maxHops) continue

    // Traverse outgoing edges
    const outgoing = graph.getOutgoingEdges(nodeId)
    for (const edge of outgoing) {
      if (predicateFilter && !predicateFilter.includes(edge.predicate)) {
        continue
      }

      collectedEdges.push(edge)
      if (!visitedNodes.has(edge.target)) {
        queue.push({
          nodeId: edge.target,
          depth: depth + 1,
        })
      }
    }

    // Also traverse incoming edges for bidirectional discovery
    const incoming = graph.getIncomingEdges(nodeId)
    for (const edge of incoming) {
      if (predicateFilter && !predicateFilter.includes(edge.predicate)) {
        continue
      }

      collectedEdges.push(edge)
      if (!visitedNodes.has(edge.source)) {
        queue.push({
          nodeId: edge.source,
          depth: depth + 1,
        })
      }
    }
  }

  return {
    nodes: collectedNodes,
    edges: collectedEdges,
    paths: [],
  }
}

// Find shortest path between two nodes (BFS)
function findShortestPath(
  graph: KnowledgeGraph,
  startId: string,
  endId: string,
  maxDepth: number = 5
): string[] | null {
  const visited = new Set<string>()
  const queue: Array<{
    nodeId: string
    path: string[]
  }> = [{ nodeId: startId, path: [startId] }]

  while (queue.length > 0) {
    const { nodeId, path } = queue.shift()!

    if (nodeId === endId) return path
    if (path.length > maxDepth) continue
    if (visited.has(nodeId)) continue
    visited.add(nodeId)

    // Check outgoing edges
    const outgoing = graph.getOutgoingEdges(nodeId)
    for (const edge of outgoing) {
      if (!visited.has(edge.target)) {
        queue.push({
          nodeId: edge.target,
          path: [...path, edge.target],
        })
      }
    }

    // Check incoming edges
    const incoming = graph.getIncomingEdges(nodeId)
    for (const edge of incoming) {
      if (!visited.has(edge.source)) {
        queue.push({
          nodeId: edge.source,
          path: [...path, edge.source],
        })
      }
    }
  }

  return null // No path found
}

// Extract a subgraph related to a set of entity names
function extractSubgraph(graph: KnowledgeGraph, entityNames: string[], hops: number = 1): TraversalResult {
  const allNodes: GraphNode[] = []
  const allEdges: GraphEdge[] = []
  const seenNodeIds = new Set<string>()
  const seenEdgeKeys = new Set<string>()

  for (const name of entityNames) {
    const node = graph.findNode(name)
    if (!node) continue

    const neighborhood = getNeighborhood(graph, node.id, hops)

    for (const n of neighborhood.nodes) {
      if (!seenNodeIds.has(n.id)) {
        seenNodeIds.add(n.id)
        allNodes.push(n)
      }
    }

    for (const e of neighborhood.edges) {
      const key = `${e.source}-${e.predicate}-${e.target}`
      if (!seenEdgeKeys.has(key)) {
        seenEdgeKeys.add(key)
        allEdges.push(e)
      }
    }
  }

  return { nodes: allNodes, edges: allEdges, paths: [] }
}

// Convert traversal result to text for LLM context
function traversalToText(result: TraversalResult): string {
  const lines: string[] = []

  lines.push('Entities:')
  for (const node of result.nodes) {
    const attrs = Object.entries(node.attributes)
      .map(([k, v]) => `${k}=${v}`)
      .join(', ')
    lines.push(`- ${node.name} (${node.type})${attrs ? ` [${attrs}]` : ''}`)
  }

  lines.push('\nRelationships:')
  for (const edge of result.edges) {
    const sourceNode = result.nodes.find(n => n.id === edge.source)
    const targetNode = result.nodes.find(n => n.id === edge.target)
    lines.push(`- ${sourceNode?.name ?? edge.source} --[${edge.predicate}]--> ${targetNode?.name ?? edge.target}`)
  }

  return lines.join('\n')
}

export { getNeighborhood, findShortestPath, extractSubgraph, traversalToText, type TraversalResult }
```

> **Beginner Note:** Start with neighborhood retrieval (1-2 hops). Most relationship questions can be answered by looking at the immediate neighborhood of the mentioned entities. Path finding is useful for "how are X and Y connected?" questions but can be expensive on large graphs.

> **Advanced Note:** For large graphs (100K+ nodes), BFS traversal without bounds can be slow. Always set a `maxDepth` limit and consider using a priority queue (Dijkstra-like) that weights edges by confidence to find the most relevant paths first.

---

## Section 6: Graph RAG Pattern

### Combining Graph and Vector Retrieval

The Graph RAG pattern runs two retrieval strategies in parallel and merges the results:

1. **Vector retrieval:** Find text chunks relevant to the query (standard RAG)
2. **Graph retrieval:** Extract entities from the query, traverse the graph, get related context

The combined context gives the LLM both relevant text passages and structured relationship information.

```typescript
// src/knowledge-graphs/graph-rag.ts

import { generateText, Output, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { KnowledgeGraph } from './graph.js'
import { extractSubgraph, traversalToText } from './traversal.js'

// Step 1: Extract entities from the query
const QueryEntitiesSchema = z.object({
  entities: z.array(z.string()).describe('Entity names mentioned or implied in the query'),
  relationshipType: z.string().optional().describe('The type of relationship being asked about, if any'),
})

async function extractQueryEntities(query: string): Promise<z.infer<typeof QueryEntitiesSchema>> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: QueryEntitiesSchema }),
    system: `Extract entity names from the query that should be looked up
in a knowledge graph. Also identify the relationship type being asked about.

Examples:
- "Who manages Alice?" -> entities: ["Alice"], relationshipType: "manages"
- "What tech does Project Alpha use?" -> entities: ["Project Alpha"], relationshipType: "uses"
- "How are Alice and Bob connected?" -> entities: ["Alice", "Bob"]`,
    messages: [{ role: 'user', content: query }],
    temperature: 0,
  })

  return output
}

// Step 2: Full Graph RAG pipeline
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
  options: {
    vectorTopK?: number
    graphHops?: number
  } = {}
): Promise<GraphRAGResult> {
  const { vectorTopK = 5, graphHops = 2 } = options

  // Run vector and graph retrieval in parallel
  const [vectorResults, queryEntities] = await Promise.all([
    vectorSearch(query, vectorTopK),
    extractQueryEntities(query),
  ])

  console.log(`Query entities: ${queryEntities.entities.join(', ')}`)
  if (queryEntities.relationshipType) {
    console.log(`Relationship type: ${queryEntities.relationshipType}`)
  }

  // Graph retrieval: get subgraph around mentioned entities
  const subgraph = extractSubgraph(graph, queryEntities.entities, graphHops)
  const graphContext = traversalToText(subgraph)

  console.log(`Graph context: ${subgraph.nodes.length} nodes, ${subgraph.edges.length} edges`)
  console.log(`Vector context: ${vectorResults.length} chunks`)

  // Combine contexts and generate answer
  const combinedContext = `
=== Knowledge Graph Context ===
${graphContext}

=== Document Context ===
${vectorResults.map((chunk, i) => `[Document ${i + 1}]: ${chunk}`).join('\n\n')}
`

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `Answer the question using both the knowledge graph context
(entities and their relationships) and the document context (text passages).

The knowledge graph provides structured relationship information.
The document context provides detailed text passages.

Use both to provide a complete, accurate answer. If the graph and documents
contradict each other, note the discrepancy.`,
    messages: [
      {
        role: 'user',
        content: `${combinedContext}\n\nQuestion: ${query}`,
      },
    ],
  })

  return {
    answer: result.text,
    vectorContext: vectorResults,
    graphContext,
    entitiesFound: subgraph.nodes.map(n => n.name),
  }
}

export { extractQueryEntities, graphRAG, type GraphRAGResult }
```

### Example: Building and Querying a Graph

```typescript
// src/knowledge-graphs/example-usage.ts

import { buildGraphFromText } from './build-graph.js'
import { graphRAG } from './graph-rag.js'

async function main(): Promise<void> {
  // Sample documents about a software organization
  const documents = [
    `Alice Smith is the Engineering Director. She manages the Platform
team and the Product team. Alice reports to CEO Bob Johnson.
Alice has been with the company since 2019.`,

    `The Platform team is led by Charlie Davis. The team maintains
the authentication service (built with TypeScript and Hono) and
the data pipeline (built with Python). Charlie reports to Alice Smith.`,

    `The Product team is led by Diana Lee. They are building
Project Alpha, a new AI-powered search feature. Project Alpha
uses the authentication service and a PostgreSQL database.
Diana reports to Alice Smith.`,

    `Project Alpha depends on the embedding service, which is maintained
by the Platform team. The embedding service uses the OpenAI API
and stores vectors in Pinecone. Charlie Davis reviewed the
architecture for Project Alpha.`,
  ]

  // Build the knowledge graph
  const graph = await buildGraphFromText(documents)

  // Simulate a vector search function
  const vectorSearch = async (query: string, topK: number): Promise<string[]> => {
    // In practice, this would search an embedding index
    // For demonstration, return relevant documents
    return documents.slice(0, topK)
  }

  // Query 1: Multi-hop relationship
  console.log('\n=== Query 1: Multi-hop ===')
  const result1 = await graphRAG(
    'Who does Charlie Davis report to, and who does that person report to?',
    graph,
    vectorSearch
  )
  console.log('Answer:', result1.answer)

  // Query 2: Dependency tracking
  console.log('\n=== Query 2: Dependencies ===')
  const result2 = await graphRAG('What services does Project Alpha depend on?', graph, vectorSearch)
  console.log('Answer:', result2.answer)

  // Query 3: Connection finding
  console.log('\n=== Query 3: Connection ===')
  const result3 = await graphRAG('How is the Platform team connected to Project Alpha?', graph, vectorSearch)
  console.log('Answer:', result3.answer)
}

main().catch(console.error)
```

> **Beginner Note:** Start by building the graph from a small set of documents (10-20) and testing queries manually. Inspect the extracted entities and relationships to verify quality before scaling up. Bad entity extraction leads to a bad graph, which leads to bad retrieval.

> **Advanced Note:** For production Graph RAG, consider pre-computing and caching common subgraphs. Entity neighborhoods for frequently queried entities can be materialized, avoiding repeated traversals. Also consider embedding the graph context itself and including it in the vector index for unified retrieval.

---

## Section 7: Maintaining Graph Consistency

### The Deduplication Problem

Entity extraction from multiple documents produces duplicates. "Alice Smith," "Alice," "A. Smith," and "Engineering Director Alice" might all refer to the same person. Without deduplication, your graph has four separate nodes for one entity, and relationships are fragmented.

```typescript
// src/knowledge-graphs/entity-resolution.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import type { GraphNode } from './graph.js'

// Strategy 1: String similarity-based matching
function stringSimilarity(a: string, b: string): number {
  const aLower = a.toLowerCase()
  const bLower = b.toLowerCase()

  // Exact match
  if (aLower === bLower) return 1.0

  // One contains the other
  if (aLower.includes(bLower) || bLower.includes(aLower)) {
    return 0.8
  }

  // Levenshtein distance-based similarity
  const maxLen = Math.max(aLower.length, bLower.length)
  if (maxLen === 0) return 1.0

  const distance = levenshteinDistance(aLower, bLower)
  return 1 - distance / maxLen
}

function levenshteinDistance(a: string, b: string): number {
  const matrix: number[][] = []

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i]
  }
  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      const cost = b.charAt(i - 1) === a.charAt(j - 1) ? 0 : 1
      matrix[i][j] = Math.min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)
    }
  }

  return matrix[b.length][a.length]
}

// Find potential duplicates based on string similarity
function findPotentialDuplicates(nodes: GraphNode[], threshold: number = 0.7): Array<[GraphNode, GraphNode, number]> {
  const duplicates: Array<[GraphNode, GraphNode, number]> = []

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      // Only compare nodes of the same type
      if (nodes[i].type !== nodes[j].type) continue

      const similarity = stringSimilarity(nodes[i].name, nodes[j].name)
      if (similarity >= threshold) {
        duplicates.push([nodes[i], nodes[j], similarity])
      }

      // Also check aliases
      for (const alias of nodes[i].aliases) {
        const aliasSim = stringSimilarity(alias, nodes[j].name)
        if (aliasSim >= threshold) {
          duplicates.push([nodes[i], nodes[j], aliasSim])
          break
        }
      }
    }
  }

  return duplicates
}

// Strategy 2: LLM-based entity resolution
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
): Promise<z.infer<typeof EntityResolutionSchema>> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: EntityResolutionSchema }),
    system: `Determine if two entities refer to the same real-world thing.
Consider:
- Name similarity and common abbreviations
- Entity type (must match for them to be the same)
- Attributes and context
- Aliases

${context ? `Context: ${context}` : ''}`,
    messages: [
      {
        role: 'user',
        content: `Entity 1: "${entity1.name}" (${entity1.type}) aliases: [${entity1.aliases.join(', ')}] attributes: ${JSON.stringify(entity1.attributes)}

Entity 2: "${entity2.name}" (${entity2.type}) aliases: [${entity2.aliases.join(', ')}] attributes: ${JSON.stringify(entity2.attributes)}`,
      },
    ],
    temperature: 0,
  })

  return output
}

// Resolve and merge duplicate entities in a graph
import { KnowledgeGraph } from './graph.js'

async function deduplicateGraph(
  graph: KnowledgeGraph,
  useLLM: boolean = false
): Promise<{
  merged: number
  mergeLog: Array<{
    kept: string
    removed: string
    confidence: number
  }>
}> {
  const stats = graph.getStats()
  const allNodes: GraphNode[] = []

  // Collect all nodes by type
  for (const type of Object.keys(stats.nodeTypes)) {
    allNodes.push(...graph.getNodesByType(type))
  }

  const duplicates = findPotentialDuplicates(allNodes, 0.75)
  const mergeLog: Array<{
    kept: string
    removed: string
    confidence: number
  }> = []

  for (const [node1, node2, similarity] of duplicates) {
    let shouldMerge = similarity >= 0.9 // Auto-merge very high similarity
    let confidence = similarity

    if (!shouldMerge && useLLM) {
      // Use LLM for uncertain cases
      const resolution = await llmEntityResolution(node1, node2)
      shouldMerge = resolution.areSameEntity
      confidence = resolution.confidence
    }

    if (shouldMerge && confidence >= 0.8) {
      mergeLog.push({
        kept: node1.name,
        removed: node2.name,
        confidence,
      })
      // In a real implementation, you would:
      // 1. Merge node2's edges into node1
      // 2. Add node2's aliases to node1
      // 3. Remove node2
      console.log(`Merged: "${node2.name}" -> "${node1.name}" (confidence: ${confidence.toFixed(2)})`)
    }
  }

  return { merged: mergeLog.length, mergeLog }
}

export { findPotentialDuplicates, llmEntityResolution, deduplicateGraph }
```

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
    reasoning: 'Questions involve ownership, dependencies, and approval chains — all relationship queries.',
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

---

## Exercises

### Exercise 1: Entity and Relationship Extraction Pipeline

**Objective:** Build an entity and relationship extraction pipeline that processes documents and builds a knowledge graph.

**Specification:**

1. Create `src/exercises/ex12-knowledge-graph.ts`
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
// tests/ex12.test.ts
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

1. Create `src/exercises/ex12-graph-rag.ts`
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
// tests/ex12-graph-rag.test.ts
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

In Module 13, you will extend your pipeline to handle multi-modal inputs — images, diagrams, screenshots, and audio — adding visual understanding to your retrieval and generation capabilities.
