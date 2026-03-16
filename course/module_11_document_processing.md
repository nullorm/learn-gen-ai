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

PDF is the most challenging format because PDFs are designed for display, not text extraction. The "text" in a PDF is a collection of positioned glyphs — there is no guarantee that characters are in reading order, that spaces exist between words, or that paragraphs are semantically grouped.

```typescript
// src/document-processing/pdf-extraction.ts

// Using pdf-parse for basic PDF extraction
// Install: bun add pdf-parse

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

async function extractPDF(filePath: string): Promise<PDFExtractionResult> {
  // Dynamic import for pdf-parse (CommonJS module)
  const pdfParse = (await import('pdf-parse')).default

  const buffer = await readFile(filePath)
  const data = await pdfParse(buffer)

  // Split text by page breaks (pdf-parse uses form feed characters)
  const pages = data.text.split('\f').filter((page: string) => page.trim().length > 0)

  return {
    text: data.text,
    pageCount: data.numpages,
    metadata: {
      title: data.info?.Title,
      author: data.info?.Author,
      creator: data.info?.Creator,
      creationDate: data.info?.CreationDate,
    },
    pages,
  }
}

// Clean extracted PDF text
function cleanPDFText(text: string): string {
  return (
    text
      // Remove page numbers (common patterns)
      .replace(/^\s*\d+\s*$/gm, '')
      // Normalize whitespace
      .replace(/[ \t]+/g, ' ')
      // Fix broken words from line wrapping (hyphenation)
      .replace(/(\w)-\n(\w)/g, '$1$2')
      // Join lines that are part of the same paragraph
      .replace(/([^\n])\n([a-z])/g, '$1 $2')
      // Normalize multiple newlines
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  )
}

// Advanced: Use LLM to clean and structure messy PDF text
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function llmCleanPDFText(rawText: string, pageNumber?: number): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a document cleaning expert. Clean up the following text
extracted from a PDF. Fix:
- Broken words from line wrapping
- Garbled characters from poor OCR
- Headers/footers that appear mid-text
- Table formatting that was lost

Preserve the actual content and structure. Return ONLY the cleaned text.
Do not add commentary.`,
    messages: [
      {
        role: 'user',
        content: pageNumber ? `Page ${pageNumber}:\n\n${rawText}` : rawText,
      },
    ],
    temperature: 0,
    maxOutputTokens: 4000,
  })

  return result.text
}

export { extractPDF, cleanPDFText, llmCleanPDFText }
```

### HTML Extraction

HTML extraction requires stripping away all the non-content elements — navigation bars, scripts, styles, ads, cookie banners — while preserving the semantic structure of the actual content.

````typescript
// src/document-processing/html-extraction.ts

// Using cheerio for HTML parsing
// Install: bun add cheerio

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

function extractHTML(html: string): HTMLExtractionResult {
  const $ = cheerio.load(html)

  // Remove non-content elements
  $(
    'script, style, nav, header, footer, iframe, ' +
      '.sidebar, .advertisement, .cookie-banner, .nav, ' +
      '#navigation, #header, #footer, .menu'
  ).remove()

  // Extract title
  const title = $('h1').first().text().trim() || $('title').text().trim() || 'Untitled'

  // Extract headings with hierarchy
  const headings: Array<{ level: number; text: string }> = []
  $('h1, h2, h3, h4, h5, h6').each((_, el) => {
    const tagName = (el as Element).tagName
    const level = parseInt(tagName.charAt(1))
    headings.push({ level, text: $(el).text().trim() })
  })

  // Extract main content
  // Try to find the main content area first
  let contentElement = $('main, article, .content, #content, .post-body')
  if (contentElement.length === 0) {
    contentElement = $('body')
  }

  // Convert to clean text preserving structure
  const content = htmlToCleanText($, contentElement)

  // Extract metadata
  const metadata = {
    description: $('meta[name="description"]').attr('content') || $('meta[property="og:description"]').attr('content'),
    author: $('meta[name="author"]').attr('content') || $('[rel="author"]').text().trim(),
    publishDate: $('meta[property="article:published_time"]').attr('content') || $('time').attr('datetime'),
  }

  // Extract links
  const links: Array<{ text: string; href: string }> = []
  contentElement.find('a[href]').each((_, el) => {
    const href = $(el).attr('href')
    const text = $(el).text().trim()
    if (href && text) {
      links.push({ text, href })
    }
  })

  return { title, content, headings, links, metadata }
}

function htmlToCleanText($: cheerio.CheerioAPI, element: cheerio.Cheerio<cheerio.AnyNode>): string {
  const lines: string[] = []

  element.children().each((_, child) => {
    const el = $(child)
    const tagName = (child as Element).tagName?.toLowerCase() ?? ''

    if (['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tagName)) {
      const prefix = '#'.repeat(parseInt(tagName.charAt(1)))
      lines.push(`\n${prefix} ${el.text().trim()}\n`)
    } else if (tagName === 'p') {
      lines.push(el.text().trim())
      lines.push('')
    } else if (tagName === 'ul' || tagName === 'ol') {
      el.find('li').each((i, li) => {
        const prefix = tagName === 'ol' ? `${i + 1}.` : '-'
        lines.push(`${prefix} ${$(li).text().trim()}`)
      })
      lines.push('')
    } else if (tagName === 'pre' || tagName === 'code') {
      lines.push('```')
      lines.push(el.text().trim())
      lines.push('```')
      lines.push('')
    } else if (tagName === 'table') {
      lines.push(tableToText($, el))
      lines.push('')
    } else if (el.text().trim()) {
      lines.push(el.text().trim())
    }
  })

  return lines
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

function tableToText($: cheerio.CheerioAPI, table: cheerio.Cheerio<cheerio.AnyNode>): string {
  const rows: string[][] = []

  table.find('tr').each((_, tr) => {
    const cells: string[] = []
    $(tr)
      .find('td, th')
      .each((_, cell) => {
        cells.push($(cell).text().trim())
      })
    rows.push(cells)
  })

  if (rows.length === 0) return ''

  // Format as markdown table
  const tableHeader = rows[0]
  const separator = tableHeader.map(() => '---')
  const body = rows.slice(1)

  return [
    '| ' + tableHeader.join(' | ') + ' |',
    '| ' + separator.join(' | ') + ' |',
    ...body.map(row => '| ' + row.join(' | ') + ' |'),
  ].join('\n')
}

export { extractHTML, type HTMLExtractionResult }
````

### Markdown Extraction

Markdown is the easiest format to work with because it is already structured text. The main challenge is preserving the hierarchy and handling embedded code blocks and tables correctly.

````typescript
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

function extractMarkdown(markdown: string): MarkdownExtractionResult {
  // Parse YAML frontmatter if present
  let content = markdown
  let frontmatter: Record<string, string> | undefined

  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (frontmatterMatch) {
    frontmatter = parseFrontmatter(frontmatterMatch[1])
    content = frontmatterMatch[2]
  }

  // Extract code blocks before parsing sections
  // (to avoid treating # in code as headings)
  const codeBlocks: Array<{ language: string; code: string }> = []
  const codeBlockPattern = /```(\w*)\n([\s\S]*?)```/g
  let codeMatch

  while ((codeMatch = codeBlockPattern.exec(content)) !== null) {
    codeBlocks.push({
      language: codeMatch[1] || 'plaintext',
      code: codeMatch[2].trim(),
    })
  }

  // Replace code blocks with placeholders for section parsing
  const contentNoCode = content.replace(/```[\s\S]*?```/g, '[CODE_BLOCK]')

  // Parse sections
  const sections = parseMarkdownSections(contentNoCode, content)

  // Extract title
  const title = frontmatter?.title || sections[0]?.heading || content.match(/^#\s+(.+)$/m)?.[1] || 'Untitled'

  return { title, sections, frontmatter, codeBlocks }
}

function parseFrontmatter(raw: string): Record<string, string> {
  const result: Record<string, string> = {}
  for (const line of raw.split('\n')) {
    const colonIndex = line.indexOf(':')
    if (colonIndex > 0) {
      const key = line.slice(0, colonIndex).trim()
      const value = line
        .slice(colonIndex + 1)
        .trim()
        .replace(/^["']|["']$/g, '')
      result[key] = value
    }
  }
  return result
}

function parseMarkdownSections(textNoCode: string, _fullText: string): MarkdownSection[] {
  const lines = textNoCode.split('\n')
  const sections: MarkdownSection[] = []
  const headingStack: MarkdownSection[] = []

  let currentContent: string[] = []

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/)

    if (headingMatch) {
      // Save current content to the previous section
      if (headingStack.length > 0) {
        headingStack[headingStack.length - 1].content = currentContent.join('\n').trim()
      }
      currentContent = []

      const level = headingMatch[1].length
      const heading = headingMatch[2].trim()

      // Build parent heading trail
      const parentHeadings: string[] = []
      for (const parent of headingStack) {
        if (parent.level < level) {
          parentHeadings.push(parent.heading)
        }
      }

      const section: MarkdownSection = {
        heading,
        level,
        content: '',
        subsections: [],
        parentHeadings,
      }

      // Pop stack to find correct parent
      while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
        headingStack.pop()
      }

      if (headingStack.length > 0) {
        headingStack[headingStack.length - 1].subsections.push(section)
      } else {
        sections.push(section)
      }

      headingStack.push(section)
    } else {
      currentContent.push(line)
    }
  }

  // Save final section content
  if (headingStack.length > 0) {
    headingStack[headingStack.length - 1].content = currentContent.join('\n').trim()
  }

  return sections
}

export { extractMarkdown, type MarkdownSection, type MarkdownExtractionResult }
````

> **Advanced Note:** For code files, consider using a proper AST parser (e.g., `@typescript-eslint/parser` for TypeScript, `tree-sitter` for multi-language support) instead of text-based extraction. AST-aware chunking preserves function boundaries, class definitions, and import relationships far better than character splitting.

---

## Section 3: Recursive Character Splitting

### Why Recursive Splitting?

The naive approach to chunking is to split text at a fixed character count — every 1000 characters, create a new chunk. This produces terrible results because it splits mid-sentence, mid-paragraph, and mid-section. A chunk that starts with "...tion of the policy" and ends with "In the next sec..." is useless for retrieval.

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

function recursiveSplit(text: string, options: Partial<SplitterOptions> = {}): string[] {
  const { maxChunkSize = 1000, minChunkSize = 100, overlap = 200, separators = DEFAULT_SEPARATORS } = options

  function splitRecursive(textBlock: string, separatorIndex: number): string[] {
    if (textBlock.length <= maxChunkSize) {
      return [textBlock.trim()].filter(t => t.length >= minChunkSize)
    }

    if (separatorIndex >= separators.length) {
      // Last resort: hard split at maxChunkSize
      const result: string[] = []
      for (let i = 0; i < textBlock.length; i += maxChunkSize - overlap) {
        result.push(textBlock.slice(i, i + maxChunkSize).trim())
      }
      return result.filter(t => t.length >= minChunkSize)
    }

    const separator = separators[separatorIndex]
    const parts = textBlock.split(separator)

    if (parts.length === 1) {
      // This separator did not split the text — try the next one
      return splitRecursive(textBlock, separatorIndex + 1)
    }

    // Merge parts into chunks that fit within maxChunkSize
    const result: string[] = []
    let currentChunk = ''

    for (const part of parts) {
      const candidate = currentChunk ? currentChunk + separator + part : part

      if (candidate.length <= maxChunkSize) {
        currentChunk = candidate
      } else {
        // Current chunk is full
        if (currentChunk.trim().length >= minChunkSize) {
          result.push(currentChunk.trim())
        }

        // If the part itself is too large, recursively split it
        if (part.length > maxChunkSize) {
          const subChunks = splitRecursive(part, separatorIndex + 1)
          result.push(...subChunks)
          currentChunk = ''
        } else {
          currentChunk = part
        }
      }
    }

    // Do not forget the last chunk
    if (currentChunk.trim().length >= minChunkSize) {
      result.push(currentChunk.trim())
    }

    return result
  }

  const rawChunks = splitRecursive(text, 0)

  // Apply overlap
  if (overlap > 0 && rawChunks.length > 1) {
    const overlappedChunks: string[] = [rawChunks[0]]

    for (let i = 1; i < rawChunks.length; i++) {
      const prevChunk = rawChunks[i - 1]
      const overlapText = prevChunk.slice(-overlap)
      // Find a clean break point in the overlap
      const breakPoint = overlapText.lastIndexOf(' ')
      const cleanOverlap = breakPoint > 0 ? overlapText.slice(breakPoint + 1) : overlapText
      overlappedChunks.push(cleanOverlap + ' ' + rawChunks[i])
    }

    return overlappedChunks
  }

  return rawChunks
}

export { recursiveSplit, type SplitterOptions, DEFAULT_SEPARATORS }
```

### Structure-Aware Splitting

The recursive splitter above works on any text. But for documents with known structure (markdown headings, HTML sections), we can do better by splitting at structural boundaries.

````typescript
// src/document-processing/structure-aware-splitter.ts

interface StructuredChunk {
  content: string
  heading?: string
  parentHeadings: string[]
  type: 'text' | 'code' | 'table' | 'list'
}

function splitMarkdownByStructure(markdown: string, maxChunkSize: number = 1000): StructuredChunk[] {
  const chunks: StructuredChunk[] = []
  const lines = markdown.split('\n')

  let currentHeading = ''
  const headingStack: string[] = []
  let currentContent: string[] = []
  let currentType: 'text' | 'code' | 'table' | 'list' = 'text'
  let inCodeBlock = false

  for (const line of lines) {
    // Track code blocks
    if (line.startsWith('```')) {
      inCodeBlock = !inCodeBlock
      if (inCodeBlock) {
        // Flush current text content before code block
        if (currentContent.length > 0) {
          chunks.push({
            content: currentContent.join('\n').trim(),
            heading: currentHeading,
            parentHeadings: [...headingStack],
            type: currentType,
          })
          currentContent = []
        }
        currentType = 'code'
      }
      currentContent.push(line)
      if (!inCodeBlock) {
        // End of code block
        chunks.push({
          content: currentContent.join('\n').trim(),
          heading: currentHeading,
          parentHeadings: [...headingStack],
          type: 'code',
        })
        currentContent = []
        currentType = 'text'
      }
      continue
    }

    if (inCodeBlock) {
      currentContent.push(line)
      continue
    }

    // Detect headings
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/)
    if (headingMatch) {
      // Flush current content
      if (currentContent.length > 0) {
        const text = currentContent.join('\n').trim()
        if (text.length > 0) {
          chunks.push({
            content: text,
            heading: currentHeading,
            parentHeadings: [...headingStack],
            type: currentType,
          })
        }
        currentContent = []
        currentType = 'text'
      }

      const level = headingMatch[1].length
      const heading = headingMatch[2].trim()

      // Update heading stack
      while (headingStack.length > 0 && headingStack.length >= level) {
        headingStack.pop()
      }
      headingStack.push(heading)
      currentHeading = heading
      continue
    }

    // Detect tables
    if (line.startsWith('|') && line.endsWith('|')) {
      if (currentType !== 'table' && currentContent.length > 0) {
        chunks.push({
          content: currentContent.join('\n').trim(),
          heading: currentHeading,
          parentHeadings: [...headingStack],
          type: currentType,
        })
        currentContent = []
      }
      currentType = 'table'
    }

    // Detect lists
    if (/^[\s]*[-*+]\s/.test(line) || /^[\s]*\d+\.\s/.test(line)) {
      if (currentType !== 'list' && currentContent.length > 0) {
        const text = currentContent.join('\n').trim()
        if (text.length > 0) {
          chunks.push({
            content: text,
            heading: currentHeading,
            parentHeadings: [...headingStack],
            type: currentType,
          })
          currentContent = []
        }
      }
      currentType = 'list'
    }

    currentContent.push(line)

    // Check if current content exceeds max size
    const currentSize = currentContent.join('\n').length
    if (currentSize > maxChunkSize) {
      chunks.push({
        content: currentContent.join('\n').trim(),
        heading: currentHeading,
        parentHeadings: [...headingStack],
        type: currentType,
      })
      currentContent = []
      currentType = 'text'
    }
  }

  // Flush remaining content
  if (currentContent.length > 0) {
    const text = currentContent.join('\n').trim()
    if (text.length > 0) {
      chunks.push({
        content: text,
        heading: currentHeading,
        parentHeadings: [...headingStack],
        type: currentType,
      })
    }
  }

  return chunks
}

export { splitMarkdownByStructure, type StructuredChunk }
````

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

```typescript
// src/document-processing/metadata-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Rule-based metadata extraction (fast, no LLM needed)

interface ExtractedMetadata {
  title: string
  dates: string[]
  emails: string[]
  urls: string[]
  headings: string[]
  keyTerms: string[]
}

function extractBasicMetadata(text: string): ExtractedMetadata {
  // Extract dates (various formats)
  const datePatterns = [
    /\d{4}-\d{2}-\d{2}/g, // ISO: 2024-01-15
    /\d{1,2}\/\d{1,2}\/\d{4}/g, // US: 1/15/2024
    /(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}/gi,
  ]

  const dates: string[] = []
  for (const pattern of datePatterns) {
    const matches = text.match(pattern)
    if (matches) dates.push(...matches)
  }

  // Extract emails
  const emails = text.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g) ?? []

  // Extract URLs
  const urls = text.match(/https?:\/\/[^\s)>\]]+/g) ?? []

  // Extract headings (markdown-style)
  const headings = text.match(/^#{1,6}\s+(.+)$/gm)?.map(h => h.replace(/^#+\s+/, '')) ?? []

  // Extract title (first heading or first line)
  const title = headings[0] || text.split('\n')[0]?.slice(0, 100) || 'Untitled'

  // Extract key terms (capitalized phrases, likely proper nouns)
  const keyTermSet = new Set<string>()
  const termPattern = /\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b/g
  let termMatch
  while ((termMatch = termPattern.exec(text)) !== null) {
    keyTermSet.add(termMatch[0])
  }

  return {
    title,
    dates: [...new Set(dates)],
    emails: [...new Set(emails)],
    urls: [...new Set(urls)],
    headings,
    keyTerms: [...keyTermSet].slice(0, 20),
  }
}

// LLM-powered metadata extraction (slower but more accurate)

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

async function extractLLMMetadata(text: string): Promise<z.infer<typeof DocumentMetadataSchema>> {
  // Truncate to first 3000 characters for metadata extraction
  // (we do not need the full document)
  const truncated = text.length > 3000 ? text.slice(0, 3000) + '\n...[truncated]' : text

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: DocumentMetadataSchema }),
    system: `Extract metadata from the following document. Be precise
and extract only what is clearly present in the text.`,
    messages: [{ role: 'user', content: truncated }],
    temperature: 0,
  })

  return output
}

// Enrich chunks with contextual metadata
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
  documentContext: {
    title: string
    type: string
    sectionHeading?: string
    parentHeadings?: string[]
  }
): Promise<EnrichedChunk> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        topics: z.array(z.string()).describe('Topics in this specific chunk'),
        entities: z.array(z.string()).describe('Named entities in this chunk'),
        chunkSummary: z.string().describe('One sentence describing what this chunk contains'),
      }),
    }),
    system: `Extract metadata for this specific chunk of a larger document.
Document: "${documentContext.title}" (${documentContext.type})
Section: ${documentContext.sectionHeading ?? 'Unknown'}`,
    messages: [{ role: 'user', content: chunk }],
    temperature: 0,
  })

  return {
    content: chunk,
    metadata: {
      sectionHeading: documentContext.sectionHeading ?? '',
      parentHeadings: documentContext.parentHeadings ?? [],
      documentTitle: documentContext.title,
      documentType: documentContext.type,
      topics: output.topics,
      entities: output.entities,
      chunkSummary: output.chunkSummary,
    },
  }
}

export { extractBasicMetadata, extractLLMMetadata, enrichChunkMetadata, type ExtractedMetadata, type EnrichedChunk }
```

> **Beginner Note:** Start with rule-based metadata extraction (dates, emails, URLs, headings). It is free, fast, and reliable. Add LLM-based metadata extraction only for fields that rule-based approaches cannot handle — like document type classification, topic extraction, or summarization.

> **Advanced Note:** Metadata-filtered retrieval is extremely powerful. Instead of searching all chunks, filter by document type ("policy"), date range ("2024"), or topic ("refund") before semantic search. This reduces the search space and dramatically improves precision. Most vector databases support metadata filtering natively.

---

## Section 5: Structured Extraction with LLMs

### Beyond Plain Text

Some documents contain structured information — tables, key-value pairs, forms, specifications — that is lost when extracted as plain text. An LLM can recover this structure and convert it into machine-readable formats.

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

async function extractTables(text: string): Promise<ExtractedTable[]> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ExtractedTableSchema }),
    system: `Extract all tables from the following text. The text may
contain tables in various formats:
- Markdown tables
- ASCII tables
- Tab-separated data
- Aligned columns

Convert each table into a structured format with headers and rows.
If a table has no explicit headers, infer them from the data.`,
    messages: [{ role: 'user', content: text }],
    temperature: 0,
  })

  return output.tables
}

// Convert extracted table back to different formats

function tableToCSV(table: ExtractedTable): string {
  const escapeCell = (cell: string) => (cell.includes(',') ? `"${cell}"` : cell)
  const lines = [table.headers.map(escapeCell).join(','), ...table.rows.map(row => row.map(escapeCell).join(','))]
  return lines.join('\n')
}

function tableToJSON(table: ExtractedTable): Array<Record<string, string>> {
  return table.rows.map(row => {
    const obj: Record<string, string> = {}
    table.headers.forEach((header, i) => {
      obj[header] = row[i] ?? ''
    })
    return obj
  })
}

export { extractTables, tableToCSV, tableToJSON, type ExtractedTable }
```

### Key-Value Pair Extraction

```typescript
// src/document-processing/kv-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

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
): Promise<z.infer<typeof KeyValueSchema>['pairs']> {
  const fieldGuidance = expectedFields ? `\n\nLook specifically for these fields: ${expectedFields.join(', ')}` : ''

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: KeyValueSchema }),
    system: `Extract key-value pairs from the following text. Look for:
- Labeled fields (e.g., "Name: John Smith")
- Form-like data
- Configuration parameters
- Specification values
- Any structured data that can be represented as key-value pairs
${fieldGuidance}

Set confidence based on how clear the extraction is:
- 1.0: Explicitly labeled field
- 0.7-0.9: Clearly implied but not explicitly labeled
- Below 0.7: Uncertain extraction`,
    messages: [{ role: 'user', content: text }],
    temperature: 0,
  })

  return output.pairs
}

// Example: Extract invoice data
async function extractInvoiceData(invoiceText: string): Promise<Record<string, string>> {
  const pairs = await extractKeyValuePairs(invoiceText, [
    'Invoice Number',
    'Date',
    'Due Date',
    'Customer Name',
    'Total Amount',
    'Tax',
    'Subtotal',
  ])

  const result: Record<string, string> = {}
  for (const pair of pairs) {
    if (pair.confidence >= 0.7) {
      result[pair.key] = pair.value
    }
  }

  return result
}

export { extractKeyValuePairs, extractInvoiceData }
```

> **Beginner Note:** Structured extraction is most valuable for documents with consistent formats — invoices, resumes, specifications, forms. For free-form text like blog posts or essays, metadata extraction (Section 4) is more appropriate.

> **Advanced Note:** For high-volume extraction of consistent document types (e.g., processing 10,000 invoices), fine-tune a smaller model on your specific format rather than using a large general model. The cost difference is significant at scale. Module 20 covers fine-tuning.

---

## Section 6: Document Hierarchies

### Parent-Child Chunk Relationships

In naive chunking, each chunk is independent. But real documents have hierarchy — a section belongs to a chapter, which belongs to a document. When a chunk is retrieved, knowing its parent context dramatically improves answer quality.

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

function buildHierarchy(markdown: string, documentId: string): HierarchicalChunk[] {
  const chunks: HierarchicalChunk[] = []
  let chunkCounter = 0

  function nextId(): string {
    return `${documentId}_chunk_${chunkCounter++}`
  }

  // Create document-level chunk
  const docChunk: HierarchicalChunk = {
    id: nextId(),
    content: markdown.slice(0, 500) + '...', // Summary/preview
    level: 'document',
    parentId: null,
    childIds: [],
    metadata: {
      headingPath: [],
      documentId,
      depth: 0,
    },
  }
  chunks.push(docChunk)

  // Parse sections
  const lines = markdown.split('\n')
  let currentSection: HierarchicalChunk | null = null
  let currentSubsection: HierarchicalChunk | null = null
  let contentBuffer: string[] = []
  const headingStack: Array<{
    id: string
    heading: string
    level: number
  }> = []

  function flushContent(): void {
    const text = contentBuffer.join('\n').trim()
    if (text.length === 0) {
      contentBuffer = []
      return
    }

    const target = currentSubsection ?? currentSection ?? docChunk
    target.content += (target.content ? '\n\n' : '') + text

    // If the content is large, create paragraph-level children
    if (text.length > 500) {
      const paragraphs = text.split('\n\n').filter(p => p.trim())
      for (const para of paragraphs) {
        if (para.trim().length < 50) continue

        const paraChunk: HierarchicalChunk = {
          id: nextId(),
          content: para.trim(),
          level: 'paragraph',
          parentId: target.id,
          childIds: [],
          metadata: {
            heading: target.metadata.heading,
            headingPath: target.metadata.headingPath,
            documentId,
            depth: target.metadata.depth + 1,
          },
        }

        target.childIds.push(paraChunk.id)
        chunks.push(paraChunk)
      }
    }

    contentBuffer = []
  }

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/)

    if (headingMatch) {
      // Flush content buffer
      flushContent()

      const level = headingMatch[1].length
      const heading = headingMatch[2].trim()

      // Determine hierarchy level
      const hierarchyLevel = level <= 2 ? 'section' : 'subsection'

      // Update heading stack
      while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
        headingStack.pop()
      }

      const newChunk: HierarchicalChunk = {
        id: nextId(),
        content: '', // Will be filled when we flush
        level: hierarchyLevel,
        parentId: headingStack.length > 0 ? headingStack[headingStack.length - 1].id : docChunk.id,
        childIds: [],
        metadata: {
          heading,
          headingPath: [...headingStack.map(h => h.heading), heading],
          documentId,
          depth: headingStack.length + 1,
        },
      }

      // Link to parent
      const parent = chunks.find(c => c.id === newChunk.parentId)
      if (parent) {
        parent.childIds.push(newChunk.id)
      }

      chunks.push(newChunk)
      headingStack.push({ id: newChunk.id, heading, level })

      if (hierarchyLevel === 'section') {
        currentSection = newChunk
        currentSubsection = null
      } else {
        currentSubsection = newChunk
      }
    } else {
      contentBuffer.push(line)
    }
  }

  // Flush remaining content
  flushContent()

  return chunks
}

// Retrieve with parent expansion
function expandToParent(chunkId: string, allChunks: HierarchicalChunk[]): string {
  const chunk = allChunks.find(c => c.id === chunkId)
  if (!chunk) return ''

  // Build context from heading path
  const headingContext = chunk.metadata.headingPath.map((h, i) => '#'.repeat(i + 1) + ' ' + h).join('\n')

  // If this is a paragraph, include the parent section content
  if (chunk.level === 'paragraph' && chunk.parentId) {
    const parent = allChunks.find(c => c.id === chunk.parentId)
    if (parent) {
      return `${headingContext}\n\n${parent.content}`
    }
  }

  return `${headingContext}\n\n${chunk.content}`
}

export { buildHierarchy, expandToParent, type HierarchicalChunk }
```

> **Beginner Note:** The simplest version of parent-child chunking is to store section headings as metadata on each chunk. When a chunk is retrieved, prepend its heading path ("Company Policies > Customer Service > Refund Policy") to give the LLM context about where this chunk lives in the document.

> **Advanced Note:** Some vector databases (like Weaviate and LlamaIndex) have built-in support for hierarchical indexing and parent-child retrieval. Using native support is more efficient than implementing it manually. The concept is the same: index small chunks for precision, retrieve parent chunks for context.

---

## Section 7: Incremental Processing

### Detecting Changes

When your document corpus changes — new documents are added, existing ones are updated, old ones are deleted — you do not want to re-process and re-embed everything. Incremental processing detects what changed and updates only the affected chunks and embeddings.

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

function computeContentHash(content: string): string {
  return createHash('sha256').update(content).digest('hex')
}

async function detectChanges(
  directoryPath: string,
  existingRecords: Map<string, DocumentRecord>,
  supportedExtensions: string[] = ['.md', '.txt', '.html', '.pdf']
): Promise<ChangeDetectionResult> {
  const result: ChangeDetectionResult = {
    added: [],
    modified: [],
    deleted: [],
    unchanged: [],
  }

  // Scan current files
  const currentFiles = new Set<string>()

  async function scanDir(dir: string): Promise<void> {
    const entries = await readdir(dir, {
      withFileTypes: true,
    })
    for (const entry of entries) {
      const fullPath = join(dir, entry.name)
      if (entry.isDirectory()) {
        await scanDir(fullPath)
      } else if (supportedExtensions.includes(extname(entry.name))) {
        currentFiles.add(fullPath)
      }
    }
  }

  await scanDir(directoryPath)

  // Check each current file
  for (const filePath of currentFiles) {
    const content = await readFile(filePath, 'utf-8')
    const hash = computeContentHash(content)

    const existing = existingRecords.get(filePath)
    if (!existing) {
      result.added.push(filePath)
    } else if (existing.contentHash !== hash) {
      result.modified.push(filePath)
    } else {
      result.unchanged.push(filePath)
    }
  }

  // Check for deleted files
  for (const [path] of existingRecords) {
    if (!currentFiles.has(path)) {
      result.deleted.push(path)
    }
  }

  return result
}

// Incremental update pipeline
interface VectorStore {
  upsert(
    chunks: Array<{
      id: string
      embedding: number[]
      metadata: Record<string, unknown>
    }>
  ): Promise<void>
  deleteByIds(ids: string[]): Promise<void>
}

async function incrementalUpdate(
  changes: ChangeDetectionResult,
  records: Map<string, DocumentRecord>,
  processFn: (path: string) => Promise<Array<{ id: string; content: string; embedding: number[] }>>,
  vectorStore: VectorStore
): Promise<{
  processedDocs: number
  addedChunks: number
  deletedChunks: number
}> {
  let addedChunks = 0
  let deletedChunks = 0

  // Process added documents
  for (const path of changes.added) {
    console.log(`Adding: ${path}`)
    const chunks = await processFn(path)

    await vectorStore.upsert(
      chunks.map(c => ({
        id: c.id,
        embedding: c.embedding,
        metadata: { path, content: c.content },
      }))
    )

    const content = await readFile(path, 'utf-8')
    records.set(path, {
      path,
      contentHash: computeContentHash(content),
      lastProcessed: new Date(),
      chunkIds: chunks.map(c => c.id),
      embeddingIds: chunks.map(c => c.id),
    })

    addedChunks += chunks.length
  }

  // Process modified documents (delete old chunks, add new)
  for (const path of changes.modified) {
    console.log(`Updating: ${path}`)
    const existing = records.get(path)
    if (existing) {
      await vectorStore.deleteByIds(existing.chunkIds)
      deletedChunks += existing.chunkIds.length
    }

    const chunks = await processFn(path)
    await vectorStore.upsert(
      chunks.map(c => ({
        id: c.id,
        embedding: c.embedding,
        metadata: { path, content: c.content },
      }))
    )

    const content = await readFile(path, 'utf-8')
    records.set(path, {
      path,
      contentHash: computeContentHash(content),
      lastProcessed: new Date(),
      chunkIds: chunks.map(c => c.id),
      embeddingIds: chunks.map(c => c.id),
    })

    addedChunks += chunks.length
  }

  // Process deleted documents
  for (const path of changes.deleted) {
    console.log(`Deleting: ${path}`)
    const existing = records.get(path)
    if (existing) {
      await vectorStore.deleteByIds(existing.chunkIds)
      deletedChunks += existing.chunkIds.length
      records.delete(path)
    }
  }

  return {
    processedDocs: changes.added.length + changes.modified.length + changes.deleted.length,
    addedChunks,
    deletedChunks,
  }
}

export { detectChanges, incrementalUpdate, computeContentHash, type DocumentRecord, type ChangeDetectionResult }
```

> **Beginner Note:** The simplest change detection is file modification timestamp. Content hashing (SHA-256) is more reliable because it detects actual content changes rather than timestamp changes from file copying or backup restoration.

> **Advanced Note:** For large-scale deployments (thousands of documents), consider using a proper document database (MongoDB, PostgreSQL with JSONB) to store document records instead of in-memory Maps. Add a processing queue (like BullMQ or a simple file-based queue) to handle large batch updates asynchronously.

---

## Section 8: Large Document Strategies

### The 500-Page Problem

A 500-page PDF might contain 250,000 words — far too much for any single LLM context window. You need strategies for processing, chunking, and querying documents of this size.

### Map-Reduce Processing

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

// Strategy 1: Hierarchical summarization
// Summarize groups of chunks, then summarize the summaries

async function hierarchicalSummarize(
  text: string,
  config: LargeDocumentConfig = DEFAULT_CONFIG
): Promise<{
  fullSummary: string
  sectionSummaries: Array<{
    section: string
    summary: string
  }>
}> {
  // Step 1: Split into manageable chunks
  const chunks = recursiveSplit(text, {
    maxChunkSize: config.maxChunkSize,
    overlap: config.overlapSize,
  })

  console.log(`Split into ${chunks.length} chunks`)

  // Step 2: Summarize in batches
  const batchSummaries: string[] = []
  const totalBatches = Math.ceil(chunks.length / config.summaryBatchSize)

  for (let i = 0; i < chunks.length; i += config.summaryBatchSize) {
    const batch = chunks.slice(i, i + config.summaryBatchSize)
    const batchText = batch.join('\n\n---\n\n')
    const batchIndex = Math.floor(i / config.summaryBatchSize) + 1

    const result = await generateText({
      model: mistral('mistral-small-latest'),
      system: `Summarize the following section of a larger document.
Preserve key facts, numbers, names, and conclusions. Be concise
but thorough. This is section ${batchIndex} of ${totalBatches}.`,
      messages: [{ role: 'user', content: batchText }],
      maxOutputTokens: 500,
      temperature: 0,
    })

    batchSummaries.push(result.text)
    console.log(`Summarized batch ${batchIndex}/${totalBatches}`)
  }

  // Step 3: Synthesize batch summaries into final summary
  const allSummaries = batchSummaries.map((s, i) => `[Section ${i + 1}]\n${s}`).join('\n\n')

  const finalResult = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are given summaries of consecutive sections of a large
document. Synthesize these into a single comprehensive summary that:
1. Captures the main thesis and key arguments
2. Preserves all important facts, numbers, and conclusions
3. Maintains the logical flow of the document
4. Is structured with clear paragraphs`,
    messages: [{ role: 'user', content: allSummaries }],
    maxOutputTokens: 2000,
    temperature: 0,
  })

  return {
    fullSummary: finalResult.text,
    sectionSummaries: batchSummaries.map((s, i) => ({
      section: `Section ${i + 1}`,
      summary: s,
    })),
  }
}

// Strategy 2: Sliding window processing
// Process overlapping windows across the document

async function slidingWindowProcess(
  text: string,
  windowSize: number = 5000,
  stepSize: number = 3000,
  processFn: (windowText: string, windowIndex: number) => Promise<string>
): Promise<string[]> {
  const results: string[] = []
  let position = 0
  let windowIndex = 0

  while (position < text.length) {
    const windowEnd = Math.min(position + windowSize, text.length)
    const windowText = text.slice(position, windowEnd)

    const result = await processFn(windowText, windowIndex)
    results.push(result)

    position += stepSize
    windowIndex++

    console.log(`Processed window ${windowIndex}: chars ${position}-${windowEnd} of ${text.length}`)
  }

  return results
}

// Strategy 3: Multi-level indexing
// Create a two-level index: coarse (section summaries) + fine (chunks)

interface MultiLevelIndex {
  documentSummary: string
  sections: Array<{
    heading: string
    summary: string
    summaryEmbedding: number[]
    chunks: Array<{
      content: string
      embedding: number[]
    }>
  }>
}

async function buildMultiLevelIndex(
  text: string,
  embeddingModel: Parameters<typeof embed>[0]['model'],
  config: LargeDocumentConfig = DEFAULT_CONFIG
): Promise<MultiLevelIndex> {
  // Split into sections first
  const sections = text.split(/\n#{1,2}\s+/).filter(s => s.trim())

  const indexedSections: MultiLevelIndex['sections'] = []

  for (const section of sections) {
    // Get heading
    const headingMatch = section.match(/^(.+)\n/)
    const heading = headingMatch?.[1]?.trim() ?? 'Untitled Section'

    // Summarize section
    const summaryResult = await generateText({
      model: mistral('mistral-small-latest'),
      system: 'Summarize this section in 2-3 sentences.',
      messages: [
        {
          role: 'user',
          content: section.slice(0, 5000),
        },
      ],
      maxOutputTokens: 200,
      temperature: 0,
    })

    // Embed summary
    const { embedding: summaryEmbedding } = await embed({
      model: embeddingModel,
      value: summaryResult.text,
    })

    // Split section into chunks and embed each
    const sectionChunks = recursiveSplit(section, {
      maxChunkSize: config.maxChunkSize,
      overlap: config.overlapSize,
    })

    const embeddedChunks = await Promise.all(
      sectionChunks.map(async content => {
        const { embedding } = await embed({
          model: embeddingModel,
          value: content,
        })
        return { content, embedding }
      })
    )

    indexedSections.push({
      heading,
      summary: summaryResult.text,
      summaryEmbedding,
      chunks: embeddedChunks,
    })
  }

  // Create document-level summary
  const docSummary = await generateText({
    model: mistral('mistral-small-latest'),
    system: 'Provide a comprehensive summary of this document.',
    messages: [
      {
        role: 'user',
        content: `Section summaries:\n${indexedSections.map(s => `- ${s.heading}: ${s.summary}`).join('\n')}`,
      },
    ],
    maxOutputTokens: 500,
    temperature: 0,
  })

  return {
    documentSummary: docSummary.text,
    sections: indexedSections,
  }
}

export {
  hierarchicalSummarize,
  slidingWindowProcess,
  buildMultiLevelIndex,
  type LargeDocumentConfig,
  type MultiLevelIndex,
}
```

> **Beginner Note:** For most use cases, you do not need all these strategies. Hierarchical summarization (Strategy 1) is the simplest and works for most documents. Use multi-level indexing (Strategy 3) when you need to query large documents repeatedly — the upfront cost is high but subsequent queries are fast.

> **Advanced Note:** Modern models like Claude have context windows of 200K tokens, which can fit approximately 150,000 words — a full-length book. For documents within this range, you can sometimes skip chunking entirely and use the full document as context, relying on the model's attention mechanism to find relevant information. This approach ("long context RAG") trades compute cost for simplicity and can be surprisingly effective. Module 5 covers long context strategies.

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

---

## Exercises

### Exercise 1: Document Ingestion Pipeline

**Objective:** Build a complete document ingestion pipeline that processes PDF and markdown files, extracts metadata, and creates structured chunks ready for embedding.

**Specification:**

1. Create `src/exercises/ex11-document-pipeline.ts`
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
// tests/ex11.test.ts
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

1. Create `src/exercises/ex11-incremental.ts`
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
// tests/ex11-incremental.test.ts
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

In Module 12, you will learn how to extract entities and relationships from documents to build knowledge graphs — a complementary retrieval strategy that captures connections that vector search alone cannot.
