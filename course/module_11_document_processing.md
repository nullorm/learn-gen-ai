# Module 11: Document Processing

## Learning Objectives

- Handle diverse document types including PDF, HTML, markdown, and source code
- Extract clean text from PDFs, HTML pages, and structured documents
- Implement recursive character splitting that respects document structure
- Extract metadata (titles, authors, dates, headings) from documents automatically
- Use LLMs for structured extraction of tables, key-value pairs, and entities from documents
- Build document hierarchies with parent-child chunk relationships and section awareness
- Implement incremental processing that detects changes and updates embeddings efficiently
- Design strategies for processing large documents (500+ pages) without losing context

---

## Why Should I Care?

RAG systems are only as good as their input. You can have the most sophisticated retrieval pipeline in the world — HyDE, hybrid search, reranking, the full stack from Module 10 — and it will still fail if the documents are poorly processed. Garbage in, garbage out.

Real-world documents are messy. PDFs have headers, footers, page numbers, and multi-column layouts that produce gibberish when extracted naively. HTML pages have navigation bars, ads, and cookie banners mixed in with the actual content. Markdown files have code blocks, tables, and nested headings that need structural awareness. And source code has its own conventions — functions, classes, imports — that generic text splitters destroy.

This module teaches you how to turn raw documents into clean, well-structured chunks with rich metadata. The quality of your chunking and metadata directly determines the quality of your retrieval. A chunk that preserves the section heading ("## Refund Policy") and its parent context ("Company Policies > Customer Service") is far more useful than a chunk that is just a floating paragraph of text.

Document processing is also where you handle the practical realities of building a RAG system at scale: processing thousands of documents, detecting when documents change, updating embeddings incrementally rather than re-processing everything, and handling documents that are too large for any single LLM context window.

---

## Connection to Other Modules

This module extends the ingestion pipeline from **Module 9 (RAG Fundamentals)** and feeds directly into the advanced retrieval techniques from **Module 10 (Advanced RAG)**.

- **Module 8 (Embeddings & Similarity)** provides the embedding models used to convert processed chunks into vectors.
- **Module 3 (Structured Output)** introduced the `generateText` with `Output.object` pattern used here for structured extraction.
- **Module 12 (Knowledge Graphs)** builds on the entity and relationship extraction techniques introduced in Section 5.
- **Module 13 (Multi-modal)** extends document processing to images, diagrams, and audio.

Think of this module as building the factory that produces the raw materials for your retrieval pipeline.

---

## Section 1: Document Types

### The Document Landscape

In practice, you will encounter four major categories of documents:

| Type         | Common Formats                     | Challenges                                               |
| ------------ | ---------------------------------- | -------------------------------------------------------- |
| **PDF**      | Reports, papers, manuals, invoices | Layout extraction, tables, images, scanned text          |
| **HTML**     | Web pages, documentation, emails   | Boilerplate removal, navigation, ads, dynamic content    |
| **Markdown** | README files, docs, notes          | Nested structure, code blocks, tables, links             |
| **Code**     | .ts, .py, .go, .java               | Function boundaries, imports, comments, type definitions |

Each type requires different extraction and chunking strategies. A one-size-fits-all approach will produce poor results.

```typescript
// src/document-processing/types.ts

interface Document {
  id: string
  source: string // File path, URL, or identifier
  type: DocumentType
  content: string // Raw extracted text
  metadata: DocumentMetadata
  chunks: Chunk[]
  processedAt: Date
  hash: string // Content hash for change detection
}

type DocumentType = 'pdf' | 'html' | 'markdown' | 'code' | 'plaintext'

interface DocumentMetadata {
  title?: string
  author?: string
  date?: string
  language?: string
  wordCount: number
  pageCount?: number
  headings: string[]
  customFields: Record<string, string>
}

interface Chunk {
  id: string
  documentId: string
  content: string
  metadata: ChunkMetadata
  embedding?: number[]
}

interface ChunkMetadata {
  sectionHeading?: string
  parentHeadings: string[] // Breadcrumb of parent headings
  pageNumber?: number
  chunkIndex: number
  totalChunks: number
  startOffset: number // Character offset in original document
  endOffset: number
  type: 'text' | 'code' | 'table' | 'list'
}

export type { Document, DocumentType, DocumentMetadata, Chunk, ChunkMetadata }
```

> **Beginner Note:** Start with the document type you encounter most. If your knowledge base is mostly markdown documentation, focus on the markdown pipeline first. You can add PDF and HTML processing later. Each document type has its own ecosystem of tools and edge cases.

---

## Section 2: Text Extraction

### PDF Extraction

PDF is the most challenging format because PDFs are designed for display, not text extraction. The "text" in a PDF is a collection of positioned glyphs -- there is no guarantee that characters are in reading order, that spaces exist between words, or that paragraphs are semantically grouped.

You will use the `pdf-parse` library (`bun add pdf-parse`). Since it is a CommonJS module, you need a dynamic import.

```typescript
// src/document-processing/pdf-extraction.ts

import { readFile } from 'fs/promises'

interface PDFExtractionResult {
  text: string
  pageCount: number
  metadata: {
    title?: string
    author?: string
    creator?: string
    creationDate?: string
  }
  pages: string[] // Text per page
}

async function extractPDF(filePath: string): Promise<PDFExtractionResult>
```

Build `extractPDF` so that it:

1. Dynamically imports `pdf-parse` and reads the file into a buffer with `readFile`.
2. Calls `pdfParse(buffer)` and pulls out `.text`, `.numpages`, and `.info` (Title, Author, Creator, CreationDate).
3. Splits the raw text into per-page strings. Hint: `pdf-parse` inserts form feed characters (`\f`) between pages.
4. Returns the result matching `PDFExtractionResult`.

What character does `pdf-parse` use to separate pages, and why would filtering out empty pages matter?

### Cleaning PDF Text

Raw PDF text is messy -- page numbers floating on their own lines, double spaces, broken words from hyphenation at line ends, and single newlines inside paragraphs. Build a `cleanPDFText` function that applies regex-based fixes:

```typescript
function cleanPDFText(text: string): string
```

Think about these cleaning steps in order:

- How would you remove standalone page numbers (lines that are just a number)?
- How do you rejoin words split by end-of-line hyphenation (`"docu-\nment"` should become `"document"`)?
- How do you distinguish a newline inside a paragraph (should become a space) from a newline between paragraphs (should stay)?
- How do you collapse excessive blank lines?

### LLM-Assisted Cleaning

For badly OCR'd or garbled PDFs, regex is not enough. Build an `llmCleanPDFText` function that sends a page of raw text to an LLM with a system prompt instructing it to fix broken words, garbled characters, stray headers/footers, and lost table formatting. Use `generateText` with `temperature: 0` and `maxOutputTokens: 4000`.

```typescript
async function llmCleanPDFText(rawText: string, pageNumber?: number): Promise<string>
```

When would you choose regex cleaning vs LLM cleaning? What is the cost trade-off?

### HTML Extraction

HTML extraction requires stripping away all the non-content elements -- navigation bars, scripts, styles, ads, cookie banners -- while preserving the semantic structure of the actual content. You will use `cheerio` (`bun add cheerio`) for server-side DOM manipulation.

```typescript
// src/document-processing/html-extraction.ts

import * as cheerio from 'cheerio'
import type { Element } from 'domhandler'

interface HTMLExtractionResult {
  title: string
  content: string
  headings: Array<{ level: number; text: string }>
  links: Array<{ text: string; href: string }>
  metadata: {
    description?: string
    author?: string
    publishDate?: string
  }
}

function extractHTML(html: string): HTMLExtractionResult
```

Build `extractHTML` with these steps:

1. Load the HTML with `cheerio.load(html)`.
2. Remove non-content elements: `script`, `style`, `nav`, `header`, `footer`, `iframe`, and common class/ID selectors like `.sidebar`, `.advertisement`, `.cookie-banner`, `#navigation`.
3. Extract the title from `<h1>` (falling back to `<title>`), headings from `h1-h6` tags, and metadata from `<meta>` tags (description, author, `article:published_time`).
4. Find the main content area (`main`, `article`, `.content`, `#content`, `.post-body`) falling back to `body`.
5. Convert the content to clean text. Build a helper `htmlToCleanText` that walks child elements and converts headings to markdown-style `#` prefixes, `<p>` to text with blank lines, `<ul>`/`<ol>` to markdown lists, `<pre>`/`<code>` to fenced code blocks, and `<table>` to markdown table format.

How would you handle a `<table>` -- what helper would you write to convert rows and cells into markdown pipe-delimited format?

### Markdown Extraction

Markdown is the easiest format to work with because it is already structured text. The main challenge is preserving the hierarchy and handling embedded code blocks and tables correctly.

```typescript
// src/document-processing/markdown-extraction.ts

interface MarkdownSection {
  heading: string
  level: number
  content: string
  subsections: MarkdownSection[]
  parentHeadings: string[]
}

interface MarkdownExtractionResult {
  title: string
  sections: MarkdownSection[]
  frontmatter?: Record<string, string>
  codeBlocks: Array<{ language: string; code: string }>
}

function extractMarkdown(markdown: string): MarkdownExtractionResult
```

Build `extractMarkdown` to:

1. Parse YAML frontmatter if present (the `---` delimited block at the top). A simple `parseFrontmatter` helper that splits on newlines and colons will do.
2. Extract all fenced code blocks before section parsing -- otherwise `#` characters inside code get treated as headings. Replace them with `[CODE_BLOCK]` placeholders for section parsing.
3. Parse sections by scanning lines for heading patterns (`/^(#{1,6})\s+(.+)$/`). Maintain a heading stack to build the `parentHeadings` breadcrumb trail and nest `subsections` correctly.

Think about this: when you encounter a `## Heading` after a `### Subheading`, how do you know to pop the stack back to the right parent level?

> **Advanced Note:** For code files, consider using a proper AST parser (e.g., `@typescript-eslint/parser` for TypeScript, `tree-sitter` for multi-language support) instead of text-based extraction. AST-aware chunking preserves function boundaries, class definitions, and import relationships far better than character splitting.

---

## Section 3: Recursive Character Splitting

### Why Recursive Splitting?

The naive approach to chunking is to split text at a fixed character count -- every 1000 characters, create a new chunk. This produces terrible results because it splits mid-sentence, mid-paragraph, and mid-section. A chunk that starts with "...tion of the policy" and ends with "In the next sec..." is useless for retrieval.

Recursive character splitting tries to split at natural boundaries, in order of preference:

1. **Section breaks** (double newlines, heading boundaries)
2. **Paragraph breaks** (single newlines)
3. **Sentence breaks** (periods, question marks, exclamation marks)
4. **Word breaks** (spaces)
5. **Character breaks** (last resort)

### Implementation

```typescript
// src/document-processing/recursive-splitter.ts

interface SplitterOptions {
  maxChunkSize: number // Maximum characters per chunk
  minChunkSize: number // Minimum characters (avoid tiny chunks)
  overlap: number // Characters of overlap between chunks
  separators: string[] // Ordered list of separators to try
}

const DEFAULT_SEPARATORS = [
  '\n\n\n', // Section breaks (triple newline)
  '\n\n', // Paragraph breaks
  '\n', // Line breaks
  '. ', // Sentence endings
  '? ', // Question endings
  '! ', // Exclamation endings
  '; ', // Semicolons
  ', ', // Commas
  ' ', // Words
  '', // Characters (last resort)
]

function recursiveSplit(text: string, options: Partial<SplitterOptions> = {}): string[]
```

Build `recursiveSplit` with a recursive inner function `splitRecursive(textBlock, separatorIndex)`:

1. **Base case 1:** If the text block fits within `maxChunkSize`, return it (filtering out chunks below `minChunkSize`).
2. **Base case 2:** If you have exhausted all separators, hard-split at `maxChunkSize` with overlap.
3. **Recursive case:** Split by the current separator. If that separator does not split the text (only 1 part), try the next separator. Otherwise, merge adjacent parts into chunks that fit within `maxChunkSize`. If any single part is still too large, recursively split it with the next separator.
4. After generating raw chunks, apply overlap: for each chunk after the first, prepend the last `overlap` characters from the previous chunk, finding a clean word boundary.

What happens if you set `minChunkSize` too high -- say, 800 with a `maxChunkSize` of 1000? How would that affect short paragraphs?

### Structure-Aware Splitting

The recursive splitter above works on any text. But for documents with known structure (markdown headings, HTML sections), we can do better by splitting at structural boundaries.

```typescript
// src/document-processing/structure-aware-splitter.ts

interface StructuredChunk {
  content: string
  heading?: string
  parentHeadings: string[]
  type: 'text' | 'code' | 'table' | 'list'
}

function splitMarkdownByStructure(markdown: string, maxChunkSize: number = 1000): StructuredChunk[]
```

Build `splitMarkdownByStructure` to walk through the markdown line by line, tracking:

- Whether you are inside a fenced code block (toggled by ` ``` ` lines)
- The current heading and a heading stack for parent breadcrumbs
- The current content type (text, code, table, list) -- detect tables by lines starting/ending with `|`, detect lists by lines matching `[-*+]\s` or `\d+\.\s`

When you encounter a heading or a type change, flush the accumulated content as a `StructuredChunk`. Also flush when content exceeds `maxChunkSize`. This produces chunks that respect document structure rather than arbitrary character boundaries.

Why does tracking content type matter for retrieval? How would a chunk of type `'code'` be retrieved differently than a chunk of type `'text'`?

> **Beginner Note:** The overlap parameter is important. Without overlap, a sentence that straddles two chunks is split in half, and neither chunk has the complete thought. An overlap of 100-200 characters ensures that boundary sentences appear in both chunks, so retrieval can find them regardless of which chunk is matched.

> **Advanced Note:** The optimal chunk size depends on your embedding model and retrieval strategy. For models like `text-embedding-3-small`, 500-1000 characters per chunk works well. For models with larger context windows (like `text-embedding-3-large`), you can use 1000-2000 characters. Larger chunks preserve more context but reduce retrieval precision. Smaller chunks improve precision but may lose context. Test both on your data.

---

## Section 4: Metadata Extraction

### Why Metadata Matters

Metadata transforms a chunk from an anonymous blob of text into a contextualized piece of information. When a user asks "What was the refund policy in 2023?", metadata lets you filter by date before semantic search, dramatically improving precision.

Good metadata includes:

- **Structural metadata:** Section heading, parent headings, page number
- **Document metadata:** Title, author, date, source URL
- **Semantic metadata:** Topics, entities, key terms
- **Technical metadata:** Chunk index, content hash, processing timestamp

### Automatic Metadata Extraction

You will build two approaches: rule-based (fast, free) and LLM-based (slower, semantic).

```typescript
// src/document-processing/metadata-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface ExtractedMetadata {
  title: string
  dates: string[]
  emails: string[]
  urls: string[]
  headings: string[]
  keyTerms: string[]
}

function extractBasicMetadata(text: string): ExtractedMetadata
```

Build `extractBasicMetadata` using regex patterns only -- no LLM calls:

1. **Dates:** Match ISO format (`\d{4}-\d{2}-\d{2}`), US format (`\d{1,2}/\d{1,2}/\d{4}`), and English format (`Jan 15, 2024`). Deduplicate the results.
2. **Emails and URLs:** Standard patterns. What regex would you use for emails?
3. **Headings:** Match markdown heading lines and strip the `#` prefix.
4. **Title:** Use the first heading, or the first line truncated to 100 characters.
5. **Key terms:** Find capitalized multi-word phrases (likely proper nouns) with `/\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b/g`. Cap at 20 terms.

### LLM-Powered Metadata

For semantic fields that regex cannot extract, define a Zod schema and use `Output.object`:

```typescript
const DocumentMetadataSchema = z.object({
  title: z.string().describe('Document title'),
  summary: z.string().describe('2-3 sentence summary of the document'),
  topics: z.array(z.string()).describe('Main topics covered (3-5 topics)'),
  documentType: z.enum([
    'policy',
    'technical',
    'report',
    'tutorial',
    'reference',
    'meeting_notes',
    'email',
    'article',
    'other',
  ]),
  audience: z.string().describe('Intended audience'),
  keyEntities: z
    .array(
      z.object({
        name: z.string(),
        type: z.enum(['person', 'organization', 'product', 'technology', 'location', 'date', 'other']),
      })
    )
    .describe('Key entities mentioned'),
  dateReferences: z.array(z.string()).describe('Dates or time periods mentioned'),
})

async function extractLLMMetadata(text: string): Promise<z.infer<typeof DocumentMetadataSchema>>
```

Build `extractLLMMetadata` to truncate the input to the first 3000 characters (metadata extraction does not need the full document) and call `generateText` with `Output.object({ schema: DocumentMetadataSchema })`.

### Enriching Chunks

Build an `enrichChunkMetadata` function that takes a single chunk plus its document context (title, type, section heading) and uses an LLM to extract chunk-specific topics, entities, and a one-sentence summary. This per-chunk metadata enables fine-grained filtering at retrieval time.

```typescript
interface EnrichedChunk {
  content: string
  metadata: {
    sectionHeading: string
    parentHeadings: string[]
    documentTitle: string
    documentType: string
    topics: string[]
    entities: string[]
    chunkSummary: string
  }
}

async function enrichChunkMetadata(
  chunk: string,
  documentContext: { title: string; type: string; sectionHeading?: string; parentHeadings?: string[] }
): Promise<EnrichedChunk>
```

When would enriching every chunk be too expensive? How would you decide which chunks are worth the LLM call?

> **Beginner Note:** Start with rule-based metadata extraction (dates, emails, URLs, headings). It is free, fast, and reliable. Add LLM-based metadata extraction only for fields that rule-based approaches cannot handle -- like document type classification, topic extraction, or summarization.

> **Advanced Note:** Metadata-filtered retrieval is extremely powerful. Instead of searching all chunks, filter by document type ("policy"), date range ("2024"), or topic ("refund") before semantic search. This reduces the search space and dramatically improves precision. Most vector databases support metadata filtering natively.

---

## Section 5: Structured Extraction with LLMs

### Beyond Plain Text

Some documents contain structured information -- tables, key-value pairs, forms, specifications -- that is lost when extracted as plain text. An LLM can recover this structure and convert it into machine-readable formats.

### Table Extraction

```typescript
// src/document-processing/table-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const ExtractedTableSchema = z.object({
  tables: z.array(
    z.object({
      title: z.string().describe('Table title or caption if available'),
      headers: z.array(z.string()).describe('Column headers'),
      rows: z.array(z.array(z.string())).describe('Table rows, each as an array of cell values'),
      summary: z.string().describe('Brief description of what this table shows'),
    })
  ),
})

type ExtractedTable = z.infer<typeof ExtractedTableSchema>['tables'][0]

async function extractTables(text: string): Promise<ExtractedTable[]>
```

Build `extractTables` to call `generateText` with `Output.object({ schema: ExtractedTableSchema })`. The system prompt should instruct the LLM to find tables in various formats (markdown, ASCII, tab-separated, aligned columns) and convert each into the structured format. If a table has no explicit headers, the LLM should infer them.

Also build two conversion utilities:

```typescript
function tableToCSV(table: ExtractedTable): string // Escape cells containing commas
function tableToJSON(table: ExtractedTable): Array<Record<string, string>> // header -> cell value
```

How would you handle a cell that contains a comma in the CSV output? What about a cell that contains a double quote?

### Key-Value Pair Extraction

```typescript
// src/document-processing/kv-extraction.ts

const KeyValueSchema = z.object({
  pairs: z.array(
    z.object({
      key: z.string().describe('The field name or label'),
      value: z.string().describe('The value'),
      confidence: z.number().min(0).max(1).describe('How confident the extraction is'),
      source: z.string().describe('The original text this was extracted from'),
    })
  ),
})

async function extractKeyValuePairs(
  text: string,
  expectedFields?: string[]
): Promise<z.infer<typeof KeyValueSchema>['pairs']>
```

Build `extractKeyValuePairs` with a system prompt that looks for labeled fields, form-like data, configuration parameters, and specification values. When `expectedFields` is provided, include them in the prompt to guide extraction. The confidence field should reflect how explicitly the pair was stated: 1.0 for `"Name: John Smith"`, 0.7-0.9 for clearly implied but unlabeled, below 0.7 for uncertain.

Build a specialized `extractInvoiceData` that wraps `extractKeyValuePairs` with expected fields like Invoice Number, Date, Due Date, Customer Name, Total Amount, Tax, Subtotal -- and filters by confidence >= 0.7.

> **Beginner Note:** Structured extraction is most valuable for documents with consistent formats -- invoices, resumes, specifications, forms. For free-form text like blog posts or essays, metadata extraction (Section 4) is more appropriate.

> **Advanced Note:** For high-volume extraction of consistent document types (e.g., processing 10,000 invoices), fine-tune a smaller model on your specific format rather than using a large general model. The cost difference is significant at scale. Module 20 covers fine-tuning.

---

## Section 6: Document Hierarchies

### Parent-Child Chunk Relationships

In naive chunking, each chunk is independent. But real documents have hierarchy -- a section belongs to a chapter, which belongs to a document. When a chunk is retrieved, knowing its parent context dramatically improves answer quality.

The parent-child pattern stores chunks at multiple levels of granularity:

- **Parent chunks:** Large sections (1000-2000 characters) that provide context
- **Child chunks:** Small segments (200-500 characters) that are precise for retrieval
- **Linking:** Each child knows its parent, and retrieval can "expand" to include parent context

```typescript
// src/document-processing/document-hierarchy.ts

interface HierarchicalChunk {
  id: string
  content: string
  level: 'document' | 'section' | 'subsection' | 'paragraph'
  parentId: string | null
  childIds: string[]
  metadata: {
    heading?: string
    headingPath: string[] // ["Chapter 1", "Section 1.2", "Subsection 1.2.1"]
    documentId: string
    depth: number
  }
}

function buildHierarchy(markdown: string, documentId: string): HierarchicalChunk[]
function expandToParent(chunkId: string, allChunks: HierarchicalChunk[]): string
```

Build `buildHierarchy` to create a tree of chunks from markdown:

1. Create a document-level chunk (depth 0) with a preview of the first 500 characters.
2. Walk lines, detecting headings. Map heading levels 1-2 to `'section'` and 3+ to `'subsection'`.
3. Maintain a heading stack. When you encounter a heading, pop the stack to the correct parent level, then push the new heading. Use the stack to build `headingPath`.
4. Accumulate content between headings in a buffer. When you hit a new heading, flush the buffer into the current chunk's content.
5. For large content blocks (> 500 characters), split by `\n\n` and create `'paragraph'`-level children linked to their parent via `parentId`/`childIds`.

A helper `nextId()` using a counter and the `documentId` keeps IDs unique: `${documentId}_chunk_${counter++}`.

Build `expandToParent` for retrieval-time context expansion: given a chunk ID, reconstruct the heading path as a markdown heading breadcrumb, and if the chunk is a paragraph, include the parent section's full content instead of just the paragraph.

Why does expanding to the parent improve answer quality? What is the trade-off in terms of context window usage?

> **Beginner Note:** The simplest version of parent-child chunking is to store section headings as metadata on each chunk. When a chunk is retrieved, prepend its heading path ("Company Policies > Customer Service > Refund Policy") to give the LLM context about where this chunk lives in the document.

> **Advanced Note:** Some vector databases (like Weaviate and LlamaIndex) have built-in support for hierarchical indexing and parent-child retrieval. Using native support is more efficient than implementing it manually. The concept is the same: index small chunks for precision, retrieve parent chunks for context.

---

## Section 7: Incremental Processing

### Detecting Changes

When your document corpus changes -- new documents are added, existing ones are updated, old ones are deleted -- you do not want to re-process and re-embed everything. Incremental processing detects what changed and updates only the affected chunks and embeddings.

```typescript
// src/document-processing/incremental.ts

import { createHash } from 'crypto'
import { readFile, readdir } from 'fs/promises'
import { join, extname } from 'path'

interface DocumentRecord {
  path: string
  contentHash: string
  lastProcessed: Date
  chunkIds: string[]
  embeddingIds: string[]
}

interface ChangeDetectionResult {
  added: string[] // New documents
  modified: string[] // Changed documents
  deleted: string[] // Removed documents
  unchanged: string[] // No changes
}

function computeContentHash(content: string): string
async function detectChanges(
  directoryPath: string,
  existingRecords: Map<string, DocumentRecord>,
  supportedExtensions?: string[]
): Promise<ChangeDetectionResult>
```

Build `computeContentHash` using `createHash('sha256').update(content).digest('hex')`.

Build `detectChanges` to:

1. Recursively scan the directory for files matching `supportedExtensions` (default: `.md`, `.txt`, `.html`, `.pdf`).
2. For each file, read its content and compute the hash.
3. Compare against `existingRecords`: if the path is not in records, it is added. If the path is in records but the hash differs, it is modified. If it matches, it is unchanged.
4. Check for deleted files: paths in `existingRecords` that are not in the current file scan.

### Incremental Update Pipeline

Build an `incrementalUpdate` function that takes the change detection result and applies updates:

```typescript
interface VectorStore {
  upsert(chunks: Array<{ id: string; embedding: number[]; metadata: Record<string, unknown> }>): Promise<void>
  deleteByIds(ids: string[]): Promise<void>
}

async function incrementalUpdate(
  changes: ChangeDetectionResult,
  records: Map<string, DocumentRecord>,
  processFn: (path: string) => Promise<Array<{ id: string; content: string; embedding: number[] }>>,
  vectorStore: VectorStore
): Promise<{ processedDocs: number; addedChunks: number; deletedChunks: number }>
```

The function should:

- For added documents: process, upsert chunks, create a new record.
- For modified documents: delete old chunks from the vector store, process the updated document, upsert new chunks, update the record.
- For deleted documents: delete chunks from the vector store, remove the record.

What happens if the process function fails mid-way through a modified document -- the old chunks are deleted but the new ones are not yet inserted? How would you make this operation atomic?

> **Beginner Note:** The simplest change detection is file modification timestamp. Content hashing (SHA-256) is more reliable because it detects actual content changes rather than timestamp changes from file copying or backup restoration.

> **Advanced Note:** For large-scale deployments (thousands of documents), consider using a proper document database (MongoDB, PostgreSQL with JSONB) to store document records instead of in-memory Maps. Add a processing queue (like BullMQ or a simple file-based queue) to handle large batch updates asynchronously.

---

## Section 8: Large Document Strategies

### The 500-Page Problem

A 500-page PDF might contain 250,000 words -- far too much for any single LLM context window. You need strategies for processing, chunking, and querying documents of this size.

### Strategy 1: Hierarchical Summarization

Process the document in parts (map), then combine the results (reduce).

```typescript
// src/document-processing/large-documents.ts

import { generateText, embed } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { recursiveSplit } from './recursive-splitter.js'

interface LargeDocumentConfig {
  maxChunkSize: number
  overlapSize: number
  summaryBatchSize: number // How many chunks to summarize at once
  maxContextTokens: number // Model context limit
}

const DEFAULT_CONFIG: LargeDocumentConfig = {
  maxChunkSize: 1000,
  overlapSize: 200,
  summaryBatchSize: 10,
  maxContextTokens: 100000,
}

async function hierarchicalSummarize(
  text: string,
  config?: LargeDocumentConfig
): Promise<{ fullSummary: string; sectionSummaries: Array<{ section: string; summary: string }> }>
```

Build `hierarchicalSummarize` in three steps:

1. **Split** the text into manageable chunks using `recursiveSplit`.
2. **Map:** Group chunks into batches of `summaryBatchSize`. For each batch, call `generateText` with a system prompt like "Summarize this section of a larger document. Preserve key facts, numbers, names. This is section X of Y." Collect the batch summaries.
3. **Reduce:** Concatenate all batch summaries and call `generateText` again to synthesize them into a single comprehensive summary.

Why does the batch summarization prompt include "section X of Y"? How does that help the model?

### Strategy 2: Sliding Window Processing

```typescript
async function slidingWindowProcess(
  text: string,
  windowSize?: number, // default 5000
  stepSize?: number, // default 3000
  processFn: (windowText: string, windowIndex: number) => Promise<string>
): Promise<string[]>
```

Build this to move a window across the document with `windowSize` characters and `stepSize` advancement. The overlap (`windowSize - stepSize`) ensures continuity between windows. Each window is passed to `processFn` for whatever processing you need (extraction, summarization, analysis).

What happens if `stepSize > windowSize`? What gets missed?

### Strategy 3: Multi-Level Indexing

Create a two-level index: coarse (section summaries) for routing, and fine (individual chunks) for precise retrieval.

```typescript
interface MultiLevelIndex {
  documentSummary: string
  sections: Array<{
    heading: string
    summary: string
    summaryEmbedding: number[]
    chunks: Array<{ content: string; embedding: number[] }>
  }>
}

async function buildMultiLevelIndex(
  text: string,
  embeddingModel: Parameters<typeof embed>[0]['model'],
  config?: LargeDocumentConfig
): Promise<MultiLevelIndex>
```

Build `buildMultiLevelIndex` to:

1. Split text into sections by top-level headings.
2. For each section: summarize it (2-3 sentences), embed the summary, split the section into chunks, embed each chunk.
3. Create a document-level summary from the section summaries.

At retrieval time, you first search section summaries to find the relevant section, then search that section's chunks for the precise answer. This two-stage approach avoids searching all chunks in a 500-page document.

When would you prefer multi-level indexing over simply embedding all chunks in a flat index? What is the trade-off?

> **Beginner Note:** For most use cases, you do not need all these strategies. Hierarchical summarization (Strategy 1) is the simplest and works for most documents. Use multi-level indexing (Strategy 3) when you need to query large documents repeatedly -- the upfront cost is high but subsequent queries are fast.

> **Advanced Note:** Modern models like Claude have context windows of 200K tokens, which can fit approximately 150,000 words -- a full-length book. For documents within this range, you can sometimes skip chunking entirely and use the full document as context, relying on the model's attention mechanism to find relevant information. This approach ("long context RAG") trades compute cost for simplicity and can be surprisingly effective. Module 5 covers long context strategies.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Bounded Reading Within Token Budgets

Production file reading tools do not read entire files blindly. They enforce constraints:

- **Max token limit per read** — truncate content that exceeds the budget
- **Max file size in bytes** — skip files that are too large before even reading
- **Offset/limit parameters** — support paginated reading of large files
- **Binary file detection** — skip non-text files automatically
- **Image handling** — pass image files as visual content rather than text
- **PDF page ranges** — read specific pages instead of the entire document

Document processing is not just about extracting text — it is about reading intelligently within constraints. Every token you spend reading a file is a token you cannot spend on the LLM's response or other context.

```typescript
interface ReadOptions {
  maxTokens?: number
  offset?: number
  limit?: number
}

async function readWithBudget(path: string, maxTokens: number): Promise<string> {
  /* ... */
}
```

Build this function with a two-stage check: first, check the file size via `stat()` and reject files that exceed the budget (estimated as `maxTokens * 4` bytes). If the file is small enough to read, load its content and estimate the token count. If within budget, return the full content. Otherwise, truncate to fit and append a `[...truncated]` marker.

The `readWithBudget` pattern is essential for any document processing pipeline that operates under token constraints. Rather than reading everything and hoping it fits, you enforce budgets at the reading layer.

---

## Section 10: File Type Routing

Different document types need different processing strategies. Production systems detect the file type and route to the appropriate processor:

- **Markdown** — split by headings, extract YAML frontmatter, preserve code blocks
- **PDF** — extract by page, handle tables and layouts, support page range selection
- **Source code** — split by functions/classes, preserve imports, respect language syntax
- **Structured data (JSON, TOML, YAML)** — parse the structure, extract fields
- **Images** — pass as visual content for multimodal models
- **Plain text** — fall back to recursive character splitting

```typescript
type FileProcessor = (content: string, metadata: FileMetadata) => Chunk[]

const processors: Record<string, FileProcessor> = {
  '.md': markdownProcessor, // Heading-aware splitting
  '.pdf': pdfProcessor, // Page-based extraction
  '.ts': codeProcessor, // Function-level splitting
  '.json': structuredDataProcessor, // Schema-aware parsing
  '.txt': plainTextProcessor, // Recursive character splitting
}

function routeFile(path: string): FileProcessor {
  const ext = extname(path).toLowerCase()
  return processors[ext] ?? plainTextProcessor
}
```

The routing pattern normalizes diverse document types into a common `Chunk` format before they enter the embedding and retrieval pipeline. This separation of concerns — detection, routing, processing, normalization — keeps each processor focused and testable.

> **Note: LSP and Structure-Aware Processing** — This module teaches chunking strategies that try to respect code structure: split by functions, preserve class boundaries, keep imports together. Language Server Protocol provides the exact structural understanding that chunking approximates. LSP knows precisely where every function starts and ends, what its parameters are, and what it calls. When you chunk source code by function boundaries, you are approximating what a language server already knows exactly. This comparison motivates why structure-preserving chunking matters: the closer your chunks align with actual code structure, the better your retrieval quality.

---

## Quiz

### Question 1 (Easy)

Why is recursive character splitting better than fixed-size splitting?

- A) Recursive splitting produces more chunks
- B) Recursive splitting tries to split at natural boundaries (paragraphs, sentences) before falling back to character boundaries, preserving semantic coherence
- C) Recursive splitting is faster
- D) Recursive splitting produces smaller files

**Answer: B** — Fixed-size splitting cuts text at arbitrary character positions, often mid-sentence or mid-word. Recursive splitting tries increasingly fine-grained separators — section breaks, paragraph breaks, sentence breaks — before resorting to character-level splits. This means chunks are more likely to contain complete thoughts, which makes them more useful for embedding and retrieval.

---

### Question 2 (Easy)

What is the purpose of chunk overlap in text splitting?

- A) To make the chunks larger
- B) To ensure sentences at chunk boundaries appear in both adjacent chunks, so they can be found by retrieval regardless of which chunk is matched
- C) To reduce the total number of chunks
- D) To improve embedding model performance

**Answer: B** — Without overlap, a sentence that spans the boundary between two chunks is split in half, and neither chunk contains the complete sentence. An overlap of 100-200 characters duplicates the boundary region in both chunks, ensuring that boundary content is retrievable from either chunk. The tradeoff is slightly more storage and slightly higher embedding costs.

---

### Question 3 (Medium)

When should you use LLM-based metadata extraction instead of rule-based extraction?

- A) Always — LLMs are more accurate at everything
- B) When you need to extract fields that require semantic understanding, like document type classification, topic categorization, or summarization
- C) Only for PDF documents
- D) Only when processing fewer than 10 documents

**Answer: B** — Rule-based extraction is sufficient for structured patterns like dates, emails, URLs, and headings — it is free, fast, and deterministic. LLM-based extraction is needed when the metadata requires understanding meaning: classifying a document as a "policy" vs. "tutorial," extracting topics, or generating summaries. Use rule-based for what it handles well and reserve LLM-based extraction for semantic fields.

---

### Question 4 (Medium)

What is the main advantage of parent-child chunk hierarchies for retrieval?

- A) They reduce storage requirements
- B) They allow small, precise child chunks for accurate retrieval while providing broader parent context that helps the LLM generate better answers
- C) They make indexing faster
- D) They eliminate the need for metadata

**Answer: B** — Small chunks (200-500 characters) are better for precise retrieval — they match queries more specifically. But they often lack the surrounding context needed for the LLM to generate a good answer. Parent-child hierarchies solve this by retrieving the precise child chunk and then "expanding" to include the parent section, giving the LLM both precision (what was matched) and context (where it fits).

---

### Question 5 (Hard)

You are processing a 500-page technical manual for a RAG system. The manual has a clear table of contents, consistent heading structure, and many cross-references between sections. Which combination of strategies would be most effective?

- A) Fixed-size splitting with large overlap
- B) Structure-aware splitting by headings, parent-child hierarchy, and multi-level indexing with section summaries
- C) Recursive character splitting with BM25 keyword search only
- D) Summarize the entire document into one chunk

**Answer: B** — A well-structured technical manual with a clear table of contents is ideally suited for structure-aware splitting: split at heading boundaries (preserving section integrity), build a parent-child hierarchy (sections contain subsections), and create a multi-level index (section summaries for coarse retrieval, individual chunks for fine-grained retrieval). Cross-references can be preserved as metadata, linking related sections. Option A ignores structure, C discards the heading hierarchy, and D loses all detail.

### Question 6 (Medium)

Why should a production file reading tool enforce a maximum token budget per read rather than reading entire files?

a) Reading full files is slower than reading partial files
b) Every token spent reading a file is a token unavailable for the LLM's response or other context — unbounded reads can consume the entire context window on a single large file
c) Token budgets reduce disk I/O
d) LLMs cannot process files larger than 1000 tokens

**Answer: B**

**Explanation:** The context window is a shared resource. If a single file read consumes 80% of the window, there is little room left for the system prompt, conversation history, other retrieved chunks, or the model's response. Bounded reading with token budgets ensures no single file monopolizes the context. The `readWithBudget` pattern validates file size before reading, truncates if needed, and provides clear feedback about truncation.

---

### Question 7 (Hard)

Your document processing pipeline handles markdown, PDF, source code, and JSON files. Rather than using one splitting strategy for all types, you implement file type routing. What is the primary benefit of routing each type to a specialized processor?

a) It reduces the total number of chunks produced
b) Specialized processors respect each format's natural structure (headings in markdown, pages in PDF, functions in code, fields in JSON), producing higher-quality chunks that preserve semantic coherence within each chunk
c) File type routing eliminates the need for embeddings
d) It allows processing all files in parallel

**Answer: B**

**Explanation:** A markdown file has headings that define semantic boundaries. A source code file has functions and classes. A JSON file has nested fields. A one-size-fits-all recursive character splitter ignores these structures and may split mid-function or mid-JSON-object. Specialized processors know where the natural split points are for each format, producing chunks that contain complete, coherent units of information. This directly improves embedding quality and retrieval precision because each chunk represents a meaningful unit rather than an arbitrary text fragment.

---

## Exercises

### Exercise 1: Document Ingestion Pipeline

**Objective:** Build a complete document ingestion pipeline that processes PDF and markdown files, extracts metadata, and creates structured chunks ready for embedding.

**Specification:**

1. Create `src/exercises/m11/ex01-document-pipeline.ts`
2. Implement a `processDocument` function that:
   - Accepts a file path (PDF or markdown)
   - Detects the file type from the extension
   - Extracts text (using pdf-parse for PDF, direct read for markdown)
   - Cleans the extracted text
   - Extracts basic metadata (rule-based) and LLM metadata
   - Splits into structure-aware chunks with parent-child relationships
   - Enriches each chunk with contextual metadata (heading path, section, document title)
3. Export a `processDirectory` function that:
   - Scans a directory for supported files
   - Processes each file
   - Returns an array of all chunks with metadata
4. Print statistics: total files, total chunks, average chunk size, metadata summary

**Expected output format:**

```
=== Document Processing Report ===
Files processed: 12 (8 markdown, 4 PDF)
Total chunks: 347
Average chunk size: 623 characters

Metadata summary:
  Document types: 5 technical, 4 policy, 3 tutorial
  Date range: 2023-01-15 to 2024-06-30
  Total entities extracted: 89

Chunk hierarchy:
  Document-level: 12
  Section-level: 67
  Subsection-level: 134
  Paragraph-level: 134
```

**Test specification:**

```typescript
// tests/exercises/m11/ex01-document-pipeline.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 11: Document Processing Pipeline', () => {
  it('should process a markdown file', async () => {
    const result = await processDocument('test-docs/sample.md')
    expect(result.chunks.length).toBeGreaterThan(0)
    expect(result.metadata.title).toBeTruthy()
  })

  it('should extract headings as metadata', async () => {
    const result = await processDocument('test-docs/sample.md')
    expect(result.metadata.headings.length).toBeGreaterThan(0)
  })

  it('should create parent-child relationships', async () => {
    const result = await processDocument('test-docs/sample.md')
    const children = result.chunks.filter(c => c.parentId !== null)
    expect(children.length).toBeGreaterThan(0)

    // Every child should reference a valid parent
    for (const child of children) {
      const parent = result.chunks.find(c => c.id === child.parentId)
      expect(parent).toBeDefined()
    }
  })

  it('should handle a directory of files', async () => {
    const results = await processDirectory('test-docs/')
    expect(results.length).toBeGreaterThan(0)
  })
})
```

---

### Exercise 2: Incremental Update System

**Objective:** Build an incremental document processing system that detects changes and updates only the affected chunks.

**Specification:**

1. Create `src/exercises/m11/ex02-incremental.ts`
2. Implement a `DocumentTracker` class that:
   - Maintains a record of all processed documents (path, hash, chunk IDs)
   - Detects added, modified, and deleted documents
   - Processes only changed documents
   - Updates the vector store (simulated with an in-memory store)
   - Reports what changed and what was updated
3. Demonstrate the system with a test scenario:
   - Process an initial set of documents
   - Modify one document
   - Add a new document
   - Delete a document
   - Run incremental update and show that only changes are processed

**Test specification:**

```typescript
// tests/exercises/m11/ex02-incremental.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 11: Incremental Updates', () => {
  it('should detect new files', async () => {
    const tracker = new DocumentTracker()
    await tracker.processDirectory('test-docs/')

    // Simulate adding a new file
    const changes = await tracker.detectChanges('test-docs/')
    expect(changes.added.length).toBeGreaterThanOrEqual(0)
  })

  it('should detect modified files', async () => {
    const tracker = new DocumentTracker()
    await tracker.processDirectory('test-docs/')

    // After modifying an existing file
    const changes = await tracker.detectChanges('test-docs/')
    // Modified detection depends on content hash comparison
    expect(changes).toBeDefined()
  })

  it('should only reprocess changed files', async () => {
    const tracker = new DocumentTracker()
    await tracker.processDirectory('test-docs/')

    const result = await tracker.incrementalUpdate('test-docs/')
    // Unchanged files should not be reprocessed
    expect(result.processedDocs).toBeDefined()
  })
})
```

> **Local Alternative (Ollama):** Document processing (PDF extraction, chunking, metadata extraction) is mostly non-LLM work. Where the module uses LLMs for intelligent chunking or metadata extraction, `ollama('qwen3.5')` works. For structured extraction from documents, use `generateText` with `Output.object` and `ollama('qwen3.5')` — see Module 3's local alternative note.

---

### Exercise 3: Bounded Document Reader

**Objective:** Build a document reader that handles large files gracefully by enforcing token budgets, supporting pagination, and detecting file types.

**Specification:**

1. Create `src/exercises/m11/ex03-bounded-reader.ts`
2. Implement a `readWithBudget(path, maxTokens)` function that:
   - Checks file size before reading (skip files over a configurable byte limit)
   - Detects binary files and returns a placeholder instead of garbled content
   - Reads text content and truncates to the token budget with a `[...truncated]` marker
   - Supports `offset` and `limit` parameters for paginated reading of large files
   - Returns metadata alongside content: file size, token count, whether truncated, file type
3. Implement a `readDirectory(dirPath, totalBudget)` function that:
   - Reads all supported files in a directory
   - Allocates the token budget across files (equal split or proportional to file size)
   - Returns each file's content within its allocated budget
4. Test with files of varying sizes — small (fits in budget), medium (requires truncation), and large (skipped entirely)

```typescript
interface BoundedReadResult {
  path: string
  content: string
  tokenCount: number
  truncated: boolean
  fileType: string
  fileSizeBytes: number
}

async function readWithBudget(path: string, maxTokens: number): Promise<BoundedReadResult> {
  // TODO: Implement bounded reading
  throw new Error('Not implemented')
}
```

**Test specification:**

```typescript
// tests/exercises/m11/ex03-bounded-reader.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 11.3: Bounded Document Reader', () => {
  it('should read small files completely', async () => {
    const result = await readWithBudget('test-docs/small.md', 1000)
    expect(result.truncated).toBe(false)
    expect(result.tokenCount).toBeLessThanOrEqual(1000)
  })

  it('should truncate large files to budget', async () => {
    const result = await readWithBudget('test-docs/large.md', 100)
    expect(result.truncated).toBe(true)
    expect(result.content).toContain('[...truncated]')
  })

  it('should detect and skip binary files', async () => {
    const result = await readWithBudget('test-docs/image.png', 1000)
    expect(result.content).toContain('[binary file]')
  })
})
```

---

### Exercise 4: Incremental Processing with Content Hashing

**Objective:** Build a document processor that hashes content on first read, detects changes on subsequent reads, and only re-processes modified documents.

**Specification:**

1. Create `src/exercises/m11/ex04-hash-processing.ts`
2. Implement a `HashedDocumentProcessor` that:
   - Computes a content hash (SHA-256) for each processed document
   - Stores the hash alongside the document's chunk IDs in a manifest
   - On subsequent processing runs, compares current hashes to stored hashes
   - Only re-processes documents whose hash has changed
   - Removes chunks for deleted documents
   - Reports a change summary: added, modified, unchanged, deleted
3. Demonstrate with a scenario:
   - Process 5 documents initially (all new)
   - Modify 1 document and add 1 new document
   - Run incremental processing — only 2 documents should be processed
   - Delete 1 document
   - Run again — deleted document's chunks should be removed

```typescript
interface ChangeReport {
  added: string[]
  modified: string[]
  unchanged: string[]
  deleted: string[]
  chunksAdded: number
  chunksRemoved: number
}

class HashedDocumentProcessor {
  async processAll(paths: string[]): Promise<ChangeReport> {
    // TODO: Hash-based incremental processing
    throw new Error('Not implemented')
  }
}
```

**Test specification:**

```typescript
// tests/exercises/m11/ex04-hash-processing.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 11.4: Incremental Processing with Hashing', () => {
  it('should process all documents on first run', async () => {
    const processor = new HashedDocumentProcessor()
    const report = await processor.processAll(testPaths)
    expect(report.added.length).toBe(testPaths.length)
    expect(report.modified.length).toBe(0)
  })

  it('should skip unchanged documents on second run', async () => {
    const processor = new HashedDocumentProcessor()
    await processor.processAll(testPaths)
    const report = await processor.processAll(testPaths)
    expect(report.unchanged.length).toBe(testPaths.length)
    expect(report.added.length).toBe(0)
  })

  it('should detect modified documents by hash comparison', async () => {
    const processor = new HashedDocumentProcessor()
    await processor.processAll(testPaths)
    // After modifying a file's content
    const report = await processor.processAll(testPaths)
    expect(report.modified.length).toBeGreaterThanOrEqual(0)
  })
})
```

---

## Summary

In this module, you learned:

1. **Document types:** PDF, HTML, markdown, and code each require different extraction strategies.
2. **Text extraction:** PDF parsing, HTML cleaning with boilerplate removal, and markdown structural parsing.
3. **Recursive character splitting:** Split text at natural boundaries (sections, paragraphs, sentences) rather than fixed character counts, with configurable overlap.
4. **Metadata extraction:** Rule-based extraction for structured patterns (dates, emails, URLs) and LLM-based extraction for semantic fields (topics, document type, summaries).
5. **Structured extraction:** LLMs can recover tables, key-value pairs, and other structured data from unstructured text.
6. **Document hierarchies:** Parent-child chunk relationships provide precise retrieval with contextual expansion.
7. **Incremental processing:** Content hashing and change detection enable efficient updates when documents change.
8. **Large document strategies:** Hierarchical summarization, sliding windows, and multi-level indexing handle documents that exceed context windows.
9. **Bounded reading:** Enforcing token budgets, file size limits, and pagination at the reading layer prevents documents from consuming more context than they are worth.
10. **File type routing:** Detecting file types and routing to specialized processors (markdown heading splitter, PDF page extractor, code function splitter) produces higher-quality chunks than one-size-fits-all splitting.

In Module 12, you will learn how to extract entities and relationships from documents to build knowledge graphs — a complementary retrieval strategy that captures connections that vector search alone cannot.
