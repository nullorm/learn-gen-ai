# Module 13: Multi-modal

## Learning Objectives

- Understand what multi-modal models can see, hear, and read, and how they process different input types
- Send images to vision models using the Vercel AI SDK with base64, URL, and file-based inputs
- Build vision use cases including OCR, diagram understanding, and screenshot analysis
- Combine image and text inputs effectively with structured prompting strategies
- Integrate audio transcription using Whisper-based pipelines
- Design multi-modal RAG systems that embed images and support cross-modal search
- Extract structured data from images including tables, forms, and charts
- Recognize the limitations and failure modes of multi-modal models

---

## Why Should I Care?

The world is not text-only. The documents in your RAG pipeline contain diagrams, screenshots, charts, and photos. Your users paste images of error messages instead of typing them. Your knowledge base includes architecture diagrams, whiteboard photos, and PDF scans that are just images wrapped in a PDF container.

Until recently, all of this visual information was invisible to LLMs. You either ignored it (losing valuable context) or ran it through OCR (losing structure and meaning). Multi-modal models change this fundamentally. Claude, GPT-5.4, and Gemini can look at an image and understand what it shows — not just the text in it, but the layout, the relationships, the visual structure.

This module teaches you how to send images to multi-modal models via the Vercel AI SDK, build practical vision applications (OCR, diagram understanding, form extraction), combine visual and textual inputs, and handle audio through transcription pipelines. You will also learn how to build multi-modal RAG systems that can retrieve and reason over both text and images.

The practical impact is immediate. Instead of asking users to describe their error, you can ask them to screenshot it. Instead of manually transcribing whiteboard diagrams, you can photograph them. Instead of ignoring the charts in a report, you can extract the data they contain.

> **Provider Note:** Vision/image features require a multi-modal provider. This module uses Anthropic (`claude-sonnet-4-20250514`). Mistral's Pixtral models or OpenAI GPT-4 are alternatives.

---

## Connection to Other Modules

This module extends the document processing capabilities from **Module 11 (Document Processing)** with visual understanding and builds on the structured output patterns from **Module 3 (Structured Output)**.

- **Module 9 (RAG Fundamentals)** and **Module 10 (Advanced RAG)** provide the retrieval pipeline that multi-modal RAG extends.
- **Module 12 (Knowledge Graphs)** can be enriched with entities extracted from images and diagrams.
- **Module 6 (Streaming & Real-time)** covers streaming patterns used in real-time multi-modal applications.

Think of this module as giving your LLM applications the ability to see and hear, not just read.

---

## Section 1: Multi-modal Models

### What They Can Process

Multi-modal models accept multiple input types and reason across them. Current capabilities:

| Modality             | Models                           | What They Can Do                                                               |
| -------------------- | -------------------------------- | ------------------------------------------------------------------------------ |
| **Text + Image**     | Claude Sonnet 4, GPT-5.4, Gemini | See photos, diagrams, charts, screenshots, handwriting                         |
| **Text + Audio**     | GPT-5.4-audio, Gemini            | Hear speech, music, sound effects                                              |
| **Text + Video**     | Gemini (limited)                 | Watch short video clips                                                        |
| **Image Generation** | DALL-E, Stable Diffusion         | Generate images from text (not covered — this module focuses on understanding) |

The Vercel AI SDK supports image input across providers through a unified interface. Audio support varies by provider.

### Building a Model Capability Registry

Your first task is to build a capability registry that tracks which models support which modalities, so the rest of your multi-modal code can select the right model at runtime.

Create `src/multimodal/capabilities.ts` exporting:

```typescript
interface ModelCapability {
  model: string
  provider: string
  imageInput: boolean
  audioInput: boolean
  videoInput: boolean
  maxImages: number
  maxImageSize: string // e.g., "20MB"
  supportedFormats: string[]
}
```

Export a `MODEL_CAPABILITIES` array containing entries for at least three providers (Mistral's Pixtral, OpenAI, and Google Gemini). Each entry should populate all fields of `ModelCapability` based on each provider's actual limits.

Export a `selectModelForModality` function with this signature:

```typescript
function selectModelForModality(
  needs: { image?: boolean; audio?: boolean; video?: boolean },
  preferredProvider?: string
): ModelCapability | undefined
```

The function should find the first capability entry matching all requested modalities. If `preferredProvider` is given (and is not `'any'`), filter to that provider.

Think about: what should happen when no model matches all requested modalities? What if the preferred provider does not support video but another does?

> **Beginner Note:** The course default provider is Mistral, but Mistral does not support image or audio input. This module uses Anthropic for image examples because Claude has strong vision capabilities. If you are using Anthropic, you have image input but not audio. For audio, you will need the OpenAI provider or a separate transcription step (Section 5).
>
> **Important:** Vision/image input requires a multi-modal provider. The code in this module uses `anthropic('claude-sonnet-4-20250514')` for all image-related calls. If you prefer a different provider, OpenAI (`openai('gpt-4o')`) and Mistral's Pixtral (`mistral('pixtral-large-latest')`) also support image input. Non-vision code (text-only analysis, audio post-processing) continues to use your default provider.

---

## Section 2: Image Input with Vercel AI SDK

### Three Ways to Send Images

The Vercel AI SDK supports three methods for including images in messages:

1. **Base64 encoded** — Image data inline as a data URI string
2. **URL reference** — A publicly accessible URL wrapped in `new URL(...)`
3. **File-based** — A `Buffer` or `Uint8Array` read from the filesystem

All three use the same message structure — a content array mixing `{ type: 'image', image: ... }` and `{ type: 'text', text: ... }` parts:

```typescript
// The content array pattern used for all three methods
content: [
  { type: 'image', image: imageData },
  { type: 'text', text: prompt },
]
```

For base64, `imageData` is a string like `data:image/png;base64,<encoded>`. For URL, it is `new URL('https://...')`. For file-based, it is a `Buffer` from `readFile`.

Create `src/multimodal/image-input.ts` exporting four async functions:

```typescript
async function analyzeImageBase64(imagePath: string, prompt: string): Promise<string>
async function analyzeImageURL(imageUrl: string, prompt: string): Promise<string>
async function analyzeImageFile(imagePath: string, prompt: string): Promise<string>
async function compareImages(imagePaths: string[], prompt: string): Promise<string>
```

For `analyzeImageBase64`: read the file, convert to base64, determine the MIME type from the file extension (map `png`, `jpg`, `jpeg`, `gif`, `webp` to their `image/*` types), and construct the data URI. Pass it as the `image` field in a content part.

For `analyzeImageURL`: wrap the URL string in `new URL(...)` and pass it as the `image` field.

For `analyzeImageFile`: read the file into a buffer and pass the buffer directly as the `image` field. This is the simplest approach.

For `compareImages`: read all images into buffers, spread them as separate `{ type: 'image' }` content parts, then append a single `{ type: 'text' }` part with the prompt. This lets the model see multiple images in one message.

All functions should call `generateText` with `anthropic('claude-sonnet-4-20250514')` and return `result.text`.

### Image Validation Utility

Before sending images to the API, you should validate them. Create `src/multimodal/image-utils.ts` exporting:

```typescript
interface ImageValidation {
  isValid: boolean
  issues: string[]
  sizeBytes: number
  estimatedTokens: number
}

async function validateImage(imagePath: string): Promise<ImageValidation>
async function resizeForVision(imagePath: string, maxDimension?: number): Promise<Buffer>
```

For `validateImage`: use `stat` to get the file size, check that it is under 20MB, and verify the extension is one of the supported formats. Estimate token cost roughly — Claude charges about 1600 tokens for a 1568x1568 image, with smaller images costing proportionally less. A simple heuristic like `Math.min(1600, Math.ceil(sizeBytes / 750))` is a reasonable starting point.

For `resizeForVision`: use the `sharp` package (`bun add sharp`) to read image metadata. If neither dimension exceeds `maxDimension` (default 1568), return the original buffer. Otherwise, resize with `fit: 'inside'` and `withoutEnlargement: true` to maintain aspect ratio.

> **Beginner Note:** Start with the file-based method (`analyzeImageFile`). It is the most straightforward: read the file, pass the buffer. URL-based input is useful when images are hosted online. Base64 is useful when you receive images from APIs or user uploads as strings.

> **Advanced Note:** Image token costs are significant. A single high-resolution image can cost 1600+ tokens — equivalent to about 1200 words of text. For applications that process many images, resize them to the minimum resolution that preserves the information you need. A screenshot at 800x600 pixels costs much less than the same screenshot at 4K resolution, and for most analysis tasks, the results are identical.

---

## Section 3: Vision Use Cases

### OCR — Optical Character Recognition

Multi-modal models are surprisingly good at reading text from images. They handle printed text, handwriting, text in photos, and even text at odd angles.

Create `src/multimodal/ocr.ts` exporting two functions:

```typescript
async function extractText(imagePath: string): Promise<string>
async function structuredOCR(imagePath: string): Promise<z.infer<typeof OCRResultSchema>>
```

For `extractText`: read the image, send it to the model with a prompt that instructs it to extract ALL visible text while preserving formatting (line breaks, indentation, headers, bullet points). Tell it to mark unclear text with `[unclear]` and return only the extracted text with no commentary. Use `temperature: 0` for consistency.

For `structuredOCR`: define an `OCRResultSchema` using Zod with this shape:

```typescript
const OCRResultSchema = z.object({
  blocks: z.array(
    z.object({
      text: z.string(),
      type: z.enum(['heading', 'paragraph', 'list_item', 'table_cell', 'caption', 'code', 'handwriting', 'other']),
      position: z.enum(['top', 'middle', 'bottom', 'left', 'right', 'center']),
      confidence: z.number().min(0).max(1).describe('How confident in the text extraction'),
    })
  ),
  language: z.string().describe('Primary language of the text'),
  hasHandwriting: z.boolean(),
  imageDescription: z.string().describe('Brief description of the image context'),
})
```

Use `Output.object({ schema: OCRResultSchema })` to get structured output. What prompt would you give the model to extract text with structural information? How does it differ from the basic `extractText` prompt?

### Diagram Understanding

Multi-modal models can interpret diagrams — architecture diagrams, flowcharts, org charts, UML diagrams — and describe their structure and meaning.

Create `src/multimodal/diagram-analysis.ts` exporting:

```typescript
async function analyzeDiagram(imagePath: string, context?: string): Promise<z.infer<typeof DiagramAnalysisSchema>>
```

Define `DiagramAnalysisSchema` to capture the diagram type (architecture, flowchart, sequence, class, entity_relationship, network, org_chart, mindmap, other), a title, an array of components (each with name, type, and description), an array of connections (each with from, to, label, and direction), and a summary.

The function should read the image buffer, call `generateText` with `Output.object({ schema: DiagramAnalysisSchema })`, and use a prompt that asks the model to extract all components and their connections. If optional `context` is provided, include it in the prompt to help the model understand domain-specific terminology.

### Screenshot Analysis

Analyzing screenshots is one of the most practical vision applications — understanding UI states, error messages, and application behavior from visual evidence.

Create `src/multimodal/screenshot-analysis.ts` exporting:

```typescript
async function analyzeScreenshot(
  imagePath: string,
  userQuestion?: string
): Promise<z.infer<typeof ScreenshotAnalysisSchema>>
```

Define `ScreenshotAnalysisSchema` to capture: `applicationType` (string), `currentState` (string describing the page/view), `visibleElements` (array of element + state pairs), `errors` (array of objects with errorText, errorType enum, and suggestedFix), and `actionSuggestion` (what the user should do next).

Use a system prompt that positions the model as a technical support expert. If `userQuestion` is provided, use it as the user message text; otherwise default to a general "What is happening in this screenshot? Are there any errors?" question.

Think about: how would you handle screenshots that show no errors? How should the errorType enum help downstream code route issues?

> **Beginner Note:** Multi-modal models are remarkably good at "reading" screenshots, even complex ones with multiple panels, modals, and overlapping elements. However, they can struggle with very small text, low-resolution images, and unusual fonts. Always test with your actual use case before relying on vision analysis in production.

> **Advanced Note:** For OCR-heavy applications (processing thousands of document scans), consider using dedicated OCR services (Google Cloud Vision, AWS Textract) for the text extraction step and then using LLMs only for understanding and structuring the extracted text. Dedicated OCR is faster and cheaper per page than sending every image to a multi-modal LLM.

---

## Section 4: Image + Text Prompting

### Effective Multi-modal Prompts

Prompting with images requires different techniques than text-only prompting. The model needs guidance on what to look at, what to extract, and how to relate the image to the text.

Create `src/multimodal/image-text-prompting.ts` exporting four functions that demonstrate four distinct image+text prompting patterns.

**Pattern 1: Image with specific questions**

```typescript
async function askAboutImage(imagePath: string, questions: string[]): Promise<Record<string, string>>
```

This function sends all questions in a single call for efficiency. Format them as a numbered list in the prompt, asking the model to respond with numbered answers. Then parse the numbered answers from the response text back into a `Record<string, string>` keyed by the original question text.

How would you parse numbered answers from free-form text? What happens if the model includes sub-points under an answer?

**Pattern 2: Image with reference text**

```typescript
async function analyzeWithContext(imagePath: string, referenceText: string, task: string): Promise<string>
```

This function includes both the image and a block of reference text in the same message. Structure the text part with labeled sections: "Reference information:" followed by the reference text, then "Task:" followed by the task description, then an instruction to use both the image and the reference.

**Pattern 3: Sequential image analysis (before/after)**

```typescript
async function beforeAfterAnalysis(beforePath: string, afterPath: string, analysisPrompt: string): Promise<string>
```

This function sends two images with labels in a single message. The content array should alternate text labels and images: `'BEFORE:'`, then the first image, `'AFTER:'`, then the second image, then the analysis prompt. Labeling images helps the model track which is which.

**Pattern 4: Image grounding (verify claims against an image)**

```typescript
const ImageGroundingSchema = z.object({
  claims: z.array(
    z.object({
      claim: z.string(),
      supportedByImage: z.boolean(),
      evidence: z.string().describe('What in the image supports or contradicts this claim'),
    })
  ),
})

async function groundClaimsInImage(imagePath: string, claims: string[]): Promise<z.infer<typeof ImageGroundingSchema>>
```

This function takes a list of claims and asks the model to check each one against the image. Use `Output.object` for structured output. The prompt should present the claims as a numbered list and ask the model to determine if each is supported by what it can see.

> **Beginner Note:** When prompting with images, be specific about what you want the model to focus on. "Describe this image" produces vague responses. "List all error messages visible in this screenshot and explain what each one means" produces actionable output.

> **Advanced Note:** The order of image and text content in the message matters for some models. Placing the image first and the question after tends to produce better results than question-first, because the model processes the image before seeing what to look for. For multi-image prompts, label each image ("BEFORE:", "AFTER:" or "Image 1:", "Image 2:") to help the model track which image to reference.

---

## Section 5: Audio Transcription

### Whisper Pipeline

Audio input is not natively supported by all providers in the Vercel AI SDK. The standard approach is a two-step pipeline: transcribe audio to text using Whisper, then process the text with your LLM.

Create `src/multimodal/audio-transcription.ts` exporting three functions.

**Transcription function:**

```typescript
interface TranscriptionResult {
  text: string
  language: string
  duration: number
  segments: Array<{ start: number; end: number; text: string }>
}

async function transcribeAudio(
  audioPath: string,
  options?: { language?: string; prompt?: string }
): Promise<TranscriptionResult>
```

This function calls OpenAI's Whisper API directly (not via the Vercel AI SDK). Read the audio file, create a `FormData` with the file blob, model name (`'whisper-1'`), and response format (`'verbose_json'`). Optionally include language and prompt fields. POST to `https://api.openai.com/v1/audio/transcriptions` with the `Authorization: Bearer ${process.env.OPENAI_API_KEY}` header. Parse the JSON response into a `TranscriptionResult`.

What should happen if the API returns a non-OK status? How would you handle the case where `segments` is missing from the response?

**Audio processing pipeline:**

```typescript
async function processAudio(
  audioPath: string,
  analysisPrompt: string
): Promise<{ transcription: string; analysis: string }>
```

This is a two-step pipeline: first transcribe with `transcribeAudio`, then analyze the transcription text with `generateText` using your default provider. The system prompt should note that the text came from speech-to-text and may contain minor errors.

**Meeting notes extraction:**

```typescript
const MeetingNotesSchema = z.object({
  title: z.string().describe('Meeting topic/title'),
  participants: z.array(z.string()).describe('People mentioned or speaking'),
  summary: z.string().describe('2-3 sentence summary of the meeting'),
  keyDecisions: z.array(z.string()).describe('Decisions made during the meeting'),
  actionItems: z.array(
    z.object({
      task: z.string(),
      assignee: z.string().optional(),
      deadline: z.string().optional(),
    })
  ),
  openQuestions: z.array(z.string()).describe('Questions that were not resolved'),
})

async function extractMeetingNotes(audioPath: string): Promise<z.infer<typeof MeetingNotesSchema>>
```

Transcribe the audio, then use `Output.object({ schema: MeetingNotesSchema })` to extract structured meeting notes from the transcription text. The system prompt should instruct the model to be thorough with action items and to use `[Speaker N]` when a participant's name is unclear.

> **Beginner Note:** Whisper's API accepts files up to 25MB. For longer audio (meetings, lectures), split the audio into segments first. Many audio processing libraries (like `ffmpeg`) can split audio at silence boundaries for cleaner segments.

> **Advanced Note:** For real-time audio processing (live transcription), use the OpenAI Realtime API or WebSocket-based transcription services rather than the batch Whisper API. The Vercel AI SDK has experimental support for real-time audio through the OpenAI provider.

---

## Section 6: Multi-modal RAG

### Embedding Images for Retrieval

Standard RAG embeds text chunks for retrieval. Multi-modal RAG extends this to images. The approach: describe each image with text, embed the description, and store it alongside the image reference.

Create `src/multimodal/multimodal-rag.ts` exporting the following types and functions.

**Core type:**

```typescript
interface MultiModalDocument {
  id: string
  type: 'text' | 'image' | 'audio'
  content: string // Text content or file path for non-text
  description: string // Text description for embedding
  embedding: number[]
  metadata: Record<string, string>
}
```

**Image description for embedding:**

```typescript
async function describeImageForEmbedding(imagePath: string, documentContext?: string): Promise<string>
```

Send the image to the vision model with a prompt that asks for a detailed, search-optimized description. The description should cover: what the image shows, all visible text, key visual elements and relationships, data or information conveyed (for charts/diagrams), and technical details. If `documentContext` is provided, include it so the model can use domain vocabulary. Set `maxOutputTokens: 500` to keep descriptions focused.

Why does the quality of this description matter so much for retrieval? What information would help a text search find this image?

**Building the index:**

```typescript
async function buildMultiModalIndex(
  items: Array<{ id: string; type: 'text' | 'image'; content: string; metadata?: Record<string, string> }>,
  embeddingModel: Parameters<typeof embed>[0]['model']
): Promise<MultiModalDocument[]>
```

Iterate over items. For images, call `describeImageForEmbedding` to generate a text description. For text items, use the content directly as the description. Embed each description using the provided embedding model. Return an array of `MultiModalDocument` objects.

**Cross-modal search:**

```typescript
async function multiModalSearch(
  query: string,
  index: MultiModalDocument[],
  embeddingModel: Parameters<typeof embed>[0]['model'],
  topK?: number
): Promise<Array<{ document: MultiModalDocument; score: number }>>
```

Embed the query, compute `cosineSimilarity` against every document's embedding, sort by score descending, and return the top K results (default 5).

**RAG generation:**

```typescript
async function multiModalRAG(
  query: string,
  searchResults: Array<{ document: MultiModalDocument; score: number }>
): Promise<string>
```

Build a multi-modal message content array. For text results, include them as `{ type: 'text' }` parts with a `[Document]:` prefix. For image results, read the image file and include both a text label and the actual image buffer as content parts. Append the user's question at the end. Call `generateText` with the vision model and return the response.

Think about: when should you include the actual image vs. just its description in the RAG context? What are the token cost implications of including images?

> **Beginner Note:** The "describe then embed" approach is the simplest way to make images searchable. The description acts as a text proxy for the image, allowing standard text embedding and retrieval. The quality of the description directly determines retrieval quality — invest in a good description prompt.

> **Advanced Note:** For production multi-modal RAG, consider using CLIP or SigLIP embedding models that natively embed both text and images into the same vector space. This avoids the description step entirely and supports true cross-modal retrieval. The Vercel AI SDK does not natively support CLIP, but you can integrate it as a custom embedding provider.

---

## Section 7: Structured Extraction from Images

### Tables from Images

One of the most practical vision applications is extracting structured data from images of tables, forms, and charts.

Create `src/multimodal/image-extraction.ts` exporting three functions for the three extraction types.

**Table extraction:**

```typescript
const ImageTableSchema = z.object({
  tables: z.array(
    z.object({
      title: z.string().optional(),
      headers: z.array(z.string()),
      rows: z.array(z.array(z.string())),
      footnotes: z.array(z.string()).describe('Any footnotes or notes below the table'),
    })
  ),
  confidence: z.number().min(0).max(1).describe('How confident in the extraction accuracy'),
})

async function extractTableFromImage(imagePath: string): Promise<z.infer<typeof ImageTableSchema>>
```

Read the image, use `Output.object({ schema: ImageTableSchema })`, and prompt the model to identify column headers, extract each row's cell values, note footnotes, use `""` for empty cells, and use `"[unclear]"` (lowering confidence) for uncertain text. Use `temperature: 0`.

**Form extraction:**

```typescript
const FormDataSchema = z.object({
  formTitle: z.string().optional(),
  fields: z.array(
    z.object({
      label: z.string().describe('The field label/name'),
      value: z.string().describe('The filled-in or selected value'),
      fieldType: z.enum(['text', 'number', 'date', 'checkbox', 'radio', 'signature', 'dropdown', 'other']),
      isRequired: z.boolean(),
      isFilled: z.boolean(),
    })
  ),
  confidence: z.number().min(0).max(1),
})

async function extractFormData(imagePath: string): Promise<z.infer<typeof FormDataSchema>>
```

The prompt should ask the model to identify each field's label, read its value, determine the field type, note if it appears required (asterisk, bold), and note if it is filled or empty.

**Chart extraction:**

```typescript
const ChartDataSchema = z.object({
  chartType: z.enum(['bar', 'line', 'pie', 'scatter', 'area', 'histogram', 'other']),
  title: z.string().optional(),
  xAxisLabel: z.string().optional(),
  yAxisLabel: z.string().optional(),
  dataPoints: z.array(
    z.object({
      label: z.string(),
      value: z.number(),
      series: z.string().optional().describe('For multi-series charts'),
    })
  ),
  trends: z.array(z.string()).describe('Observable trends in the data'),
  confidence: z.number().min(0).max(1),
})

async function extractChartData(imagePath: string): Promise<z.infer<typeof ChartDataSchema>>
```

The prompt should instruct the model to read chart type, title, axis labels, and all data points (estimating values from the visual when exact numbers are not labeled). For bar/pie charts: extract category labels and values. For line charts: extract data points along the line. For scatter plots: estimate x,y coordinates.

Why is the confidence field important for chart extraction specifically? How does chart extraction differ from table extraction in terms of accuracy?

> **Beginner Note:** Chart data extraction is approximate — the model estimates values from the visual representation. For precise data extraction, always verify extracted numbers against the source data. For tables and forms, accuracy is much higher because the values are explicit text.

> **Advanced Note:** For high-accuracy table extraction from document scans, consider combining multi-modal LLM extraction with traditional computer vision techniques. Use OpenCV or similar libraries for table detection (finding table boundaries) and the LLM for cell content extraction. This hybrid approach is more reliable than either alone.

---

## Section 8: Limitations and Gotchas

### What Multi-modal Models Get Wrong

Multi-modal models have systematic failure modes that you need to know about and design around.

Create `src/multimodal/limitations.ts`. Start by defining and exporting a `KNOWN_LIMITATIONS` array of objects with this shape:

```typescript
interface LimitationExample {
  category: string
  description: string
  workaround: string
  severity: 'high' | 'medium' | 'low'
}
```

Populate it with entries for these eight failure modes:

1. **Spatial reasoning** (medium) — models may get left/right, above/below relationships wrong in complex layouts. Workaround: ask the model to describe spatial relationships explicitly and verify.
2. **Small text** (high) — text smaller than ~12px may be misread or missed. Workaround: crop and resize to make text larger.
3. **Counting** (medium) — models are bad at counting objects in images. Workaround: use object detection models (YOLO, etc.) instead.
4. **Hallucinated text** (high) — models may "read" text that is not in the image, especially when blurry. Workaround: cross-reference and use confidence scoring.
5. **Chart value estimation** (medium) — numerical values from charts are approximate. Workaround: verify against source data.
6. **Resolution sensitivity** (high) — resolution below 200 DPI degrades text extraction. Workaround: upscale to 300+ DPI.
7. **Multi-page documents** (low) — sending many pages as images is expensive and loses inter-page context. Workaround: use PDF extraction for text, vision only for images/diagrams.
8. **Handwriting** (medium) — recognition varies by quality. Workaround: flag uncertain words with `[unclear]`.

### Defensive Extraction

Next, export a `defensiveExtraction` function:

```typescript
const ValidatedExtractionSchema = z.object({
  extraction: z.string().describe('The extracted content'),
  confidence: z.number().min(0).max(1),
  uncertainAreas: z.array(z.string()).describe('Parts of the extraction that may be incorrect'),
  suggestedVerification: z.array(z.string()).describe('What a human should double-check'),
})

async function defensiveExtraction(
  imagePath: string,
  prompt: string
): Promise<z.infer<typeof ValidatedExtractionSchema>>
```

Use `Output.object` with the schema above. The system prompt should position the model as a careful document analyst: note blurry or unclear text in `uncertainAreas`, flag guessed values, set confidence based on overall quality, and suggest specific things a human should verify.

### Cost Estimation

Finally, export an `estimateImageCost` function:

```typescript
interface ImageCostEstimate {
  estimatedInputTokens: number
  estimatedCostUSD: number
  recommendation: string
}

function estimateImageCost(widthPx: number, heightPx: number, provider?: 'anthropic' | 'openai'): ImageCostEstimate
```

Implement the token estimation logic based on image tiling. Most providers tile images into fixed blocks. For Anthropic: images up to 384px on the longest side cost about 170 tokens (thumbnail), up to 768px cost about 800 tokens (medium), and larger images are tiled into 1568x1568 blocks at about 1600 tokens each. Compute the cost in USD based on the provider's input token pricing (e.g., $3/M for Claude Sonnet). Generate a recommendation string: suggest resizing if tokens exceed 3200, or suggest higher resolution if tokens are below 200 (text may be hard to read).

> **Beginner Note:** The biggest gotcha with multi-modal models is hallucinated text — the model "reads" text that is not actually in the image. Always verify critical extracted text (names, numbers, codes) against the original image. Never trust OCR output from a multi-modal model for financial, legal, or medical data without human verification.

> **Advanced Note:** For production applications, build a confidence-based workflow: extract with the multi-modal model, flag low-confidence items, route those to human reviewers. This gives you the speed of automation with the accuracy of human oversight. Track accuracy metrics over time to identify systematic failure patterns in your specific domain.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Image Preprocessing Pipeline

### Why Preprocess?

Production systems never send raw user images directly to the API. Images arrive in unexpected formats, at absurd resolutions, or with file sizes that blow through API limits. A preprocessing pipeline validates and normalizes images before they reach the model.

The pipeline has three stages:

1. **Validation** — check format (PNG, JPEG, GIF, WebP), reject unsupported types, verify the file is actually an image (not a renamed `.exe`)
2. **Resize** — large images waste tokens. Resize to stay within the model's optimal range while maintaining aspect ratio
3. **Optimize** — convert to an efficient format if needed, strip metadata, compress

```typescript
interface ImageValidationResult {
  valid: boolean
  format: string | null
  width: number
  height: number
  fileSizeBytes: number
  errors: string[]
}
```

The validation step should detect format from the file header (magic bytes), not the file extension. A `.png` file might actually be a JPEG. Check dimensions and file size against configurable limits. Return actionable error messages: "Image is 8000x6000 — max allowed is 4096x4096" is far more useful than "Invalid image."

For resizing, maintain the aspect ratio and cap the longest dimension. Most vision models get diminishing returns above 1568px on the longest side. A 4000x3000 photo resized to 1568x1176 looks identical to the model but uses significantly fewer tokens.

> **Beginner Note:** You do not need a heavy image processing library for basic validation. Reading the first few bytes of a file tells you the format (PNG starts with `\x89PNG`, JPEG with `\xFF\xD8`). For resizing, the `sharp` npm package is the standard choice in Node.js/Bun.

> **Advanced Note:** In production, run preprocessing asynchronously. When a user uploads an image, validate and resize in the background so the pipeline is ready when the LLM call happens. Cache preprocessed images to avoid re-processing the same image on retry.

---

## Section 10: Token Cost of Images

### Images Are Expensive

Image tokens are a significant cost driver in multi-modal applications. A single high-resolution image can consume as many tokens as several pages of text. Understanding this relationship lets you make informed trade-offs.

The token cost depends on resolution. Most providers tile large images into fixed-size chunks and charge per tile. Smaller images that fit in a single tile cost a flat amount. The relationship is roughly:

| Image Size | Approximate Tokens | Equivalent Text |
| ---------- | ------------------ | --------------- |
| 256x256    | ~200               | ~150 words      |
| 768x768    | ~800               | ~600 words      |
| 1568x1568  | ~1,600             | ~1,200 words    |
| 3000x3000  | ~6,400             | ~4,800 words    |

The practical implication: resizing a 3000x3000 image to 1568x1568 cuts token cost by 75% with negligible quality loss for most tasks (OCR, diagram understanding, screenshot analysis). Only keep full resolution when fine visual detail matters — reading tiny text, identifying small UI elements, or analyzing detailed charts.

```typescript
// Quick token estimate for planning
const estimateTokens = (width: number, height: number): number => {
  const maxDim = Math.max(width, height)
  if (maxDim <= 512) return 200
  if (maxDim <= 768) return 800
  const tiles = Math.ceil(width / 1568) * Math.ceil(height / 1568)
  return tiles * 1600
}
```

When building multi-modal applications, track image token usage separately from text tokens. This lets you identify which images are driving cost and whether resizing would help. A dashboard that shows "image tokens: 80% of total input" tells you exactly where to optimize.

---

## Section 11: File Type Routing for Multi-modal

### Connecting Document Processing to Vision

In **Module 11 (Document Processing)**, you built file readers that extract text from documents. Multi-modal models extend this: files that are images should not be read as text — they should be sent as visual content to a vision model.

A file type router sits at the entry point of your processing pipeline and decides how to handle each file:

- **Text files** (`.txt`, `.md`, `.csv`) — read as text, pass to text-based LLM
- **Image files** (`.png`, `.jpg`, `.gif`, `.webp`) — validate, preprocess, pass as visual content
- **PDF files** — extract text for text-heavy PDFs, or render pages as images for scanned/visual PDFs
- **Binary files** — skip with a warning

```typescript
type FileHandler = 'text' | 'image' | 'pdf' | 'skip'

function routeFile(filePath: string, mimeType: string): FileHandler {
  if (mimeType.startsWith('image/')) return 'image'
  if (mimeType === 'application/pdf') return 'pdf'
  if (mimeType.startsWith('text/')) return 'text'
  return 'skip'
}
```

The key insight is that the router makes multi-modal processing transparent to the rest of your pipeline. Upstream code sends files in, downstream code receives processed content — whether that content came from text extraction or image analysis.

> **Advanced Note:** For PDFs, the routing decision is not always clear. A scanned PDF is really a collection of images and should be processed with vision. A text-heavy PDF should be extracted as text. Production systems detect this by checking if the PDF has extractable text layers — if yes, use text extraction; if no, render pages as images.

---

## Quiz

### Question 1 (Easy)

Which three methods does the Vercel AI SDK support for sending images to multi-modal models?

- A) URL, file path string, and screenshot capture
- B) Base64 encoded data, URL reference, and file buffer (Uint8Array/Buffer)
- C) HTML img tags, SVG, and canvas elements
- D) Only URL references are supported

**Answer: B** — The Vercel AI SDK supports three image input methods: base64 data URIs (e.g., `data:image/png;base64,...`), URL references (as `new URL(...)` objects), and raw file buffers (Uint8Array or Buffer from `fs.readFile`). All three produce the same result — the choice depends on how you have the image data available.

---

### Question 2 (Easy)

Why is the "describe then embed" approach used for multi-modal RAG?

- A) Images cannot be embedded directly
- B) Standard text embedding models cannot process images, so you generate a text description of the image and embed that description for text-based retrieval
- C) It is faster than embedding images
- D) It produces more accurate results than CLIP

**Answer: B** — Standard text embedding models (like `text-embedding-3-small`) only accept text input. To make images searchable alongside text, you use a multi-modal LLM to generate a detailed text description of each image, then embed that description. Queries are matched against these descriptions using standard text similarity. This is simpler than using a native multi-modal embedding model like CLIP, though less precise for purely visual queries.

---

### Question 3 (Medium)

When extracting data from a chart image, why should you always verify the extracted numbers?

- A) Multi-modal models cannot see charts
- B) The model estimates values visually from the chart rendering — it reads the visual position of bars, lines, and points rather than the underlying data, so values are approximate
- C) Chart extraction only works with bar charts
- D) The model always returns random numbers for charts

**Answer: B** — When a model "reads" a chart, it is interpreting the visual rendering — the height of a bar, the position of a point on a line. These visual estimates are inherently approximate. A bar that represents $45.2M might be read as $45M or $46M depending on the chart's scale and resolution. For tables and forms, where values are explicit text, accuracy is much higher because the model is reading actual characters, not estimating positions.

---

### Question 4 (Hard)

You are building a support system where users paste screenshots of error messages. What is the most important failure mode to protect against?

- A) The model refusing to analyze screenshots
- B) Hallucinated text — the model may "read" text that is not actually in the screenshot, leading to incorrect error identification and wrong troubleshooting advice
- C) Slow processing time
- D) Images being too large to send

**Answer: B** — Hallucinated text is the most dangerous failure mode because it is invisible to the user. The model might read "Error 404: Page Not Found" when the screenshot actually shows "Error 403: Forbidden" — the hallucination is plausible enough that neither the user nor the system catches it. The troubleshooting advice will be wrong, and the user will not know why. Protect against this by asking the model to report confidence, flagging uncertain readings, and ideally showing the user what text was extracted so they can verify.

---

### Question 5 (Hard)

Your RAG system processes a mix of 1000 text documents and 200 architecture diagrams. Users ask questions like "Which service handles payment processing?" that could be answered by either text or diagrams. What is the most cost-effective approach to multi-modal RAG?

- A) Send all 200 diagram images with every query
- B) Generate text descriptions of all diagrams once, embed the descriptions, and include them in the same text-based vector index as the documents. Only fetch and send actual images to the model when a diagram is retrieved
- C) Ignore the diagrams and rely only on text documents
- D) Convert all text documents to images and use a vision-only approach

**Answer: B** — Generating descriptions once is a fixed upfront cost. After that, the descriptions live in the same vector index as your text chunks, making retrieval uniform and cheap. When a diagram description is retrieved as relevant, you fetch the actual image and include it in the LLM context alongside text chunks. This means you only pay the image token cost for the 1-3 diagrams actually relevant to each query, not all 200. Option A is extremely expensive (200 images per query), C loses valuable information, and D is wasteful and less accurate.

---

### Question 6 (Medium)

Why should an image preprocessing pipeline validate file format using magic bytes (file header) rather than the file extension?

a) Magic bytes are faster to read than file extensions
b) A file's extension can be wrong — a `.png` file might actually be a JPEG. Reading the first few bytes of the file reveals the true format, preventing processing errors and potential security issues
c) File extensions are not available on all operating systems
d) Magic bytes provide better image quality

**Answer: B**

**Explanation:** File extensions are user-controlled metadata that can be incorrect, either by accident (renaming a file) or intentionally (disguising file types). The file header (magic bytes) is embedded in the file itself — PNG files start with `\x89PNG`, JPEGs with `\xFF\xD8`. Validating via magic bytes ensures you process the file with the correct decoder and avoids cryptic errors when a "PNG" file is actually a JPEG or vice versa.

---

### Question 7 (Hard)

Your multi-modal RAG system processes a mix of text documents and architectural diagrams. A user asks "Which service handles authentication?" and a relevant diagram is retrieved. You need to include the diagram in the LLM context. What is the most token-efficient approach?

a) Always send the full-resolution original image
b) Send only the text description that was generated during indexing, without the image
c) Resize the image to the model's optimal resolution (around 1568px on the longest side) before including it — this reduces token cost by up to 75% while preserving enough detail for the model to read labels and understand the architecture
d) Convert the diagram to ASCII art

**Answer: C**

**Explanation:** Vision models get diminishing returns above approximately 1568px on the longest dimension. A 4000x3000 architecture diagram resized to 1568x1176 still has enough resolution for the model to read service names, arrows, and labels. The token savings are substantial — roughly 75% fewer image tokens. Option B loses visual information that the text description may not fully capture (spatial relationships, connections). Option A wastes tokens on resolution the model cannot effectively use.

---

## Exercises

### Exercise 1: Multi-modal Q&A System

**Objective:** Build a Q&A system that handles both text questions and questions with attached screenshots.

**Specification:**

1. Create `src/exercises/m13/ex01-multimodal-qa.ts`
2. Implement a `multiModalQA` function that:
   - Accepts a question (string) and an optional image path
   - If an image is provided, includes it in the context
   - Uses a system prompt appropriate for mixed text/image input
   - Returns a structured response with answer, confidence, and source (text vs image)
3. Implement three demo scenarios:
   - **Text-only question:** "What is the refund policy?" (answered from text context)
   - **Image-only question:** "What error is shown?" (with a screenshot)
   - **Text + image question:** "Is this error related to our API rate limiting?" (with reference docs and a screenshot)
4. For the image scenarios, create simple test images (or use provided sample images)
5. Print the response with confidence scores

**Expected output format:**

```
=== Scenario 1: Text Only ===
Question: What is the refund policy?
Answer: Full refunds within 30 days...
Confidence: 0.95
Source: text_context

=== Scenario 2: Image Only ===
Question: What error is shown?
[Image: screenshots/error.png]
Answer: The screenshot shows a 429 Too Many Requests error...
Confidence: 0.88
Source: image_analysis

=== Scenario 3: Text + Image ===
Question: Is this error related to our API rate limiting?
[Context: rate-limiting-docs.md]
[Image: screenshots/error.png]
Answer: Yes, the error shown matches our API rate limiting...
Confidence: 0.92
Source: combined
```

**Test specification:**

```typescript
// tests/exercises/m13/ex01-multimodal-qa.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 13: Multi-modal Q&A', () => {
  it('should answer text-only questions', async () => {
    const result = await multiModalQA('What is the refund policy?', undefined, textContext)
    expect(result.answer).toBeTruthy()
    expect(result.source).toBe('text_context')
  })

  it('should analyze images when provided', async () => {
    const result = await multiModalQA('What does this show?', 'test-images/sample.png')
    expect(result.answer).toBeTruthy()
    expect(result.confidence).toBeGreaterThan(0)
  })

  it('should combine text and image context', async () => {
    const result = await multiModalQA('Is this related to our documentation?', 'test-images/sample.png', textContext)
    expect(result.source).toBe('combined')
  })
})
```

---

### Exercise 2: Image Data Extraction Pipeline

**Objective:** Build a pipeline that extracts structured data from images of tables, forms, and charts, and makes the data queryable.

**Specification:**

1. Create `src/exercises/m13/ex02-image-extraction.ts`
2. Implement extractors for three image types:
   - **Table extractor:** Extracts headers and rows, outputs as JSON and CSV
   - **Form extractor:** Extracts field labels and values, outputs as key-value pairs
   - **Chart extractor:** Extracts data points and trends, outputs as JSON
3. Implement an `autoDetectAndExtract` function that:
   - Takes an image path
   - Determines the image type (table, form, chart, or other)
   - Applies the appropriate extractor
   - Returns structured data with confidence scores
4. Process a set of test images and print the extracted data
5. Include confidence-based flagging for uncertain extractions

**Test specification:**

```typescript
// tests/exercises/m13/ex02-image-extraction.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 13: Image Data Extraction', () => {
  it('should extract table data', async () => {
    const result = await extractTableFromImage('test-images/table.png')
    expect(result.tables.length).toBeGreaterThan(0)
    expect(result.tables[0].headers.length).toBeGreaterThan(0)
    expect(result.tables[0].rows.length).toBeGreaterThan(0)
  })

  it('should auto-detect image type', async () => {
    const result = await autoDetectAndExtract('test-images/table.png')
    expect(result.detectedType).toBe('table')
    expect(result.confidence).toBeGreaterThan(0.5)
  })

  it('should flag uncertain extractions', async () => {
    const result = await autoDetectAndExtract('test-images/blurry-chart.png')
    expect(result.uncertainAreas.length).toBeGreaterThanOrEqual(0)
  })
})
```

---

### Exercise 3: Image Preprocessing Pipeline

**Objective:** Build a preprocessing pipeline that validates, resizes, and optimizes images before sending them to a multi-modal model.

**Specification:**

1. Create `src/exercises/m13/ex03-image-preprocessing.ts`
2. Export an async function `preprocessImage(imagePath: string, options?: PreprocessOptions): Promise<PreprocessResult>`
3. Define the types:

```typescript
interface PreprocessOptions {
  maxDimension?: number // default: 1568
  maxFileSizeBytes?: number // default: 5_000_000 (5MB)
  allowedFormats?: string[] // default: ['png', 'jpg', 'jpeg', 'gif', 'webp']
  outputFormat?: 'png' | 'jpeg' | 'webp' // default: 'png'
}

interface PreprocessResult {
  valid: boolean
  originalWidth: number
  originalHeight: number
  originalSizeBytes: number
  processedWidth?: number
  processedHeight?: number
  processedSizeBytes?: number
  format: string
  wasResized: boolean
  wasConverted: boolean
  errors: string[]
  outputPath?: string // Path to the preprocessed image
}
```

4. Implement the pipeline:
   - **Validate format** — detect actual format from file header bytes (not extension). Reject unsupported formats with a clear error message
   - **Check dimensions** — read image width and height. If the longest dimension exceeds `maxDimension`, flag for resize
   - **Resize** — resize to fit within `maxDimension` while maintaining aspect ratio. Use the `sharp` package or equivalent
   - **Check file size** — reject files that exceed `maxFileSizeBytes` even after resize
   - **Output** — write the processed image to a temp file and return the path

5. Handle edge cases: corrupted files, zero-byte files, unsupported formats disguised with valid extensions

**Test specification:**

```typescript
// tests/exercises/m13/ex03-image-preprocessing.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 13: Image Preprocessing Pipeline', () => {
  it('should accept valid images within size limits', async () => {
    const result = await preprocessImage('test-images/small-valid.png')
    expect(result.valid).toBe(true)
    expect(result.wasResized).toBe(false)
    expect(result.errors).toHaveLength(0)
  })

  it('should resize oversized images', async () => {
    const result = await preprocessImage('test-images/large-4000x3000.png', {
      maxDimension: 1568,
    })
    expect(result.valid).toBe(true)
    expect(result.wasResized).toBe(true)
    expect(Math.max(result.processedWidth!, result.processedHeight!)).toBeLessThanOrEqual(1568)
  })

  it('should reject unsupported formats', async () => {
    const result = await preprocessImage('test-images/document.bmp')
    expect(result.valid).toBe(false)
    expect(result.errors.length).toBeGreaterThan(0)
  })

  it('should detect format from header, not extension', async () => {
    // A JPEG file with a .png extension
    const result = await preprocessImage('test-images/actually-jpeg.png')
    expect(result.format).toBe('jpeg')
    expect(result.valid).toBe(true)
  })
})
```

---

### Exercise 4: Resolution and Token Cost Experiment

**Objective:** Build a tool that sends the same image at different resolutions to a multi-modal model and compares the quality of responses against the token cost.

**Specification:**

1. Create `src/exercises/m13/ex04-token-cost-experiment.ts`
2. Export an async function `runTokenExperiment(imagePath: string, question: string, resolutions: number[]): Promise<ExperimentResult>`
3. Define the types:

```typescript
interface ResolutionTrial {
  maxDimension: number
  actualWidth: number
  actualHeight: number
  estimatedTokens: number
  response: string
  responseLength: number
  durationMs: number
}

interface ExperimentResult {
  question: string
  trials: ResolutionTrial[]
  recommendation: string // Which resolution offers the best quality/cost trade-off
}
```

4. For each resolution in the `resolutions` array:
   - Resize the image to that max dimension (reuse your preprocessing pipeline from Exercise 3 or build a simpler version)
   - Estimate the token cost using the `estimateImageCost` function from Section 7
   - Send the resized image to a multi-modal model with the same question
   - Record the response, token estimate, and duration

5. After all trials, compare responses and generate a recommendation: which resolution gives an acceptable response at the lowest cost?

6. Print a comparison table showing resolution, tokens, duration, and response preview

**Test specification:**

```typescript
// tests/exercises/m13/ex04-token-cost-experiment.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 13: Token Cost Experiment', () => {
  it('should run trials at multiple resolutions', async () => {
    const result = await runTokenExperiment(
      'test-images/chart.png',
      'What data does this chart show?',
      [256, 768, 1568]
    )
    expect(result.trials).toHaveLength(3)
  })

  it('should show increasing token estimates with resolution', async () => {
    const result = await runTokenExperiment('test-images/chart.png', 'Describe this image', [256, 1568])
    expect(result.trials[1].estimatedTokens).toBeGreaterThan(result.trials[0].estimatedTokens)
  })

  it('should produce a recommendation', async () => {
    const result = await runTokenExperiment('test-images/chart.png', 'What does this show?', [256, 768])
    expect(result.recommendation).toBeTruthy()
  })
})
```

> **Local Alternative (Ollama):** For vision tasks, use `ollama('ministral-3')` which has native vision support for image understanding, screenshot analysis, and visual question answering. For audio transcription, Whisper can be run locally via `whisper.cpp`. Multi-modal RAG works with local vision models for image understanding combined with `qwen3-embedding:0.6b` for text embeddings.

---

## Summary

In this module, you learned:

1. **Multi-modal models:** Current models can process text, images, and (some) audio. The Vercel AI SDK provides a unified interface across providers.
2. **Image input:** Three methods — base64, URL, and file buffer — for sending images to multi-modal models.
3. **Vision use cases:** OCR, diagram understanding, and screenshot analysis are practical applications that work today.
4. **Image + text prompting:** Effective multi-modal prompts are specific, labeled, and provide context for what the model should look at.
5. **Audio transcription:** Whisper-based pipelines convert audio to text for LLM processing, enabling meeting notes extraction and voice-based Q&A.
6. **Multi-modal RAG:** The "describe then embed" approach makes images searchable alongside text in standard vector stores.
7. **Structured extraction:** Tables, forms, and charts in images can be converted to structured data (JSON, CSV) with LLM vision.
8. **Limitations:** Hallucinated text, spatial reasoning errors, counting failures, and resolution sensitivity are systematic failure modes. Build confidence scoring and human verification into critical workflows.
9. **Image preprocessing:** Production systems validate format (via magic bytes, not extension), resize to optimal dimensions, and strip metadata before sending images to the model, preventing API errors and reducing token waste.
10. **Token cost of images:** Image tokens scale with resolution — resizing a 3000x3000 image to 1568x1568 cuts cost by roughly 75% with negligible quality loss for most vision tasks.
11. **File type routing for multi-modal:** A router directs text files to text extraction and image files to vision processing, making multi-modal handling transparent to the rest of the pipeline.

This completes Part III: Advanced Retrieval. You now have a comprehensive retrieval toolkit — vector search, hybrid search, reranking, knowledge graphs, document processing, and multi-modal understanding. In Part IV, you will learn to build agents that use these retrieval capabilities as tools, orchestrating complex multi-step workflows autonomously.
