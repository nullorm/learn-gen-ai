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

```typescript
// src/multimodal/capabilities.ts

// Multi-modal model capabilities and selection

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

const MODEL_CAPABILITIES: ModelCapability[] = [
  {
    model: 'pixtral-12b-2409',
    provider: 'mistral',
    imageInput: true,
    audioInput: false,
    videoInput: false,
    maxImages: 20,
    maxImageSize: '20MB',
    supportedFormats: ['png', 'jpg', 'gif', 'webp'],
  },
  {
    model: 'gpt-5.4',
    provider: 'openai',
    imageInput: true,
    audioInput: true,
    videoInput: false,
    maxImages: 20,
    maxImageSize: '20MB',
    supportedFormats: ['png', 'jpg', 'gif', 'webp'],
  },
  {
    model: 'gemini-1.5-pro', // TODO: Update to the latest Gemini model available (e.g., gemini-2.0-flash or gemini-2.5-pro)
    provider: 'google',
    imageInput: true,
    audioInput: true,
    videoInput: true,
    maxImages: 100,
    maxImageSize: '20MB',
    supportedFormats: ['png', 'jpg', 'gif', 'webp', 'mp4'],
  },
]

function selectModelForModality(
  needs: {
    image?: boolean
    audio?: boolean
    video?: boolean
  },
  preferredProvider: string = 'mistral'
): ModelCapability | undefined {
  return MODEL_CAPABILITIES.find(
    m =>
      (!needs.image || m.imageInput) &&
      (!needs.audio || m.audioInput) &&
      (!needs.video || m.videoInput) &&
      (m.provider === preferredProvider || preferredProvider === 'any')
  )
}

export { MODEL_CAPABILITIES, selectModelForModality }
```

> **Beginner Note:** The course default provider is Mistral, but Mistral does not support image or audio input. This module uses Anthropic for image examples because Claude has strong vision capabilities. If you are using Anthropic, you have image input but not audio. For audio, you will need the OpenAI provider or a separate transcription step (Section 5).

---

## Section 2: Image Input with Vercel AI SDK

### Three Ways to Send Images

The Vercel AI SDK supports three methods for including images in messages:

1. **Base64 encoded** — Image data inline in the message
2. **URL reference** — A publicly accessible URL pointing to the image
3. **File-based** — Read from the local filesystem

```typescript
// src/multimodal/image-input.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { readFile } from 'fs/promises'

// Method 1: Base64 encoded image
async function analyzeImageBase64(imagePath: string, prompt: string): Promise<string> {
  const imageBuffer = await readFile(imagePath)
  const base64Image = imageBuffer.toString('base64')

  // Determine MIME type from extension
  const extension = imagePath.split('.').pop()?.toLowerCase()
  const mimeTypes: Record<string, string> = {
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    gif: 'image/gif',
    webp: 'image/webp',
  }
  const mimeType = mimeTypes[extension ?? 'png'] ?? 'image/png'

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            image: `data:${mimeType};base64,${base64Image}`,
          },
          {
            type: 'text',
            text: prompt,
          },
        ],
      },
    ],
  })

  return result.text
}

// Method 2: URL reference
async function analyzeImageURL(imageUrl: string, prompt: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            image: new URL(imageUrl),
          },
          {
            type: 'text',
            text: prompt,
          },
        ],
      },
    ],
  })

  return result.text
}

// Method 3: File buffer (Uint8Array)
async function analyzeImageFile(imagePath: string, prompt: string): Promise<string> {
  const imageBuffer = await readFile(imagePath)

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            image: imageBuffer,
          },
          {
            type: 'text',
            text: prompt,
          },
        ],
      },
    ],
  })

  return result.text
}

// Multiple images in a single message
async function compareImages(imagePaths: string[], prompt: string): Promise<string> {
  const imageContents = await Promise.all(
    imagePaths.map(async path => {
      const buffer = await readFile(path)
      return {
        type: 'image' as const,
        image: buffer,
      }
    })
  )

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          ...imageContents,
          {
            type: 'text',
            text: prompt,
          },
        ],
      },
    ],
  })

  return result.text
}

// Example usage
async function main(): Promise<void> {
  // Analyze a local screenshot
  const description = await analyzeImageFile(
    'screenshots/error.png',
    'What error is shown in this screenshot? Describe the error message and suggest a fix.'
  )
  console.log('Analysis:', description)

  // Compare two UI designs
  const comparison = await compareImages(
    ['designs/v1.png', 'designs/v2.png'],
    'Compare these two UI designs. What changed between v1 and v2? List the differences.'
  )
  console.log('Comparison:', comparison)
}

main().catch(console.error)

export { analyzeImageBase64, analyzeImageURL, analyzeImageFile, compareImages }
```

### Image Input Best Practices

```typescript
// src/multimodal/image-utils.ts

import { readFile, stat } from 'fs/promises'

interface ImageValidation {
  isValid: boolean
  issues: string[]
  sizeBytes: number
  estimatedTokens: number
}

async function validateImage(imagePath: string): Promise<ImageValidation> {
  const issues: string[] = []

  // Check file exists and size
  const fileStat = await stat(imagePath)
  const sizeBytes = fileStat.size
  const sizeMB = sizeBytes / (1024 * 1024)

  if (sizeMB > 20) {
    issues.push(`Image is ${sizeMB.toFixed(1)}MB — exceeds 20MB limit`)
  }

  // Check extension
  const extension = imagePath.split('.').pop()?.toLowerCase()
  const supportedFormats = ['png', 'jpg', 'jpeg', 'gif', 'webp']
  if (!extension || !supportedFormats.includes(extension)) {
    issues.push(`Unsupported format: .${extension}. Use: ${supportedFormats.join(', ')}`)
  }

  // Estimate token cost (rough approximation)
  // Claude charges ~1600 tokens for a 1568x1568 image
  // Smaller images cost proportionally less
  const estimatedTokens = Math.min(1600, Math.ceil(sizeBytes / 750))

  return {
    isValid: issues.length === 0,
    issues,
    sizeBytes,
    estimatedTokens,
  }
}

// Resize image to reduce token cost (requires sharp)
// Install: bun add sharp
async function resizeForVision(imagePath: string, maxDimension: number = 1568): Promise<Buffer> {
  const sharp = (await import('sharp')).default

  const image = sharp(imagePath)
  const metadata = await image.metadata()

  const width = metadata.width ?? 0
  const height = metadata.height ?? 0

  if (width <= maxDimension && height <= maxDimension) {
    return await readFile(imagePath)
  }

  return await image
    .resize(maxDimension, maxDimension, {
      fit: 'inside',
      withoutEnlargement: true,
    })
    .toBuffer()
}

export { validateImage, resizeForVision }
```

> **Beginner Note:** Start with the file-based method (`analyzeImageFile`). It is the most straightforward: read the file, pass the buffer. URL-based input is useful when images are hosted online. Base64 is useful when you receive images from APIs or user uploads as strings.

> **Advanced Note:** Image token costs are significant. A single high-resolution image can cost 1600+ tokens — equivalent to about 1200 words of text. For applications that process many images, resize them to the minimum resolution that preserves the information you need. A screenshot at 800x600 pixels costs much less than the same screenshot at 4K resolution, and for most analysis tasks, the results are identical.

---

## Section 3: Vision Use Cases

### OCR — Optical Character Recognition

Multi-modal models are surprisingly good at reading text from images. They handle printed text, handwriting, text in photos, and even text at odd angles.

```typescript
// src/multimodal/ocr.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { readFile } from 'fs/promises'

// Basic OCR: extract all text from an image
async function extractText(imagePath: string): Promise<string> {
  const imageBuffer = await readFile(imagePath)

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Extract ALL text visible in this image. Preserve the
original formatting as much as possible:
- Maintain line breaks where they appear
- Preserve indentation and alignment
- Keep headers, bullet points, and numbered lists
- Note any text that is unclear with [unclear]

Return ONLY the extracted text, no commentary.`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return result.text
}

// Structured OCR: extract text with layout information
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

async function structuredOCR(imagePath: string): Promise<z.infer<typeof OCRResultSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: OCRResultSchema }),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: 'Extract all text from this image with structural information.',
          },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

export { extractText, structuredOCR }
```

### Diagram Understanding

Multi-modal models can interpret diagrams — architecture diagrams, flowcharts, org charts, UML diagrams — and describe their structure and meaning.

```typescript
// src/multimodal/diagram-analysis.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { readFile } from 'fs/promises'

const DiagramAnalysisSchema = z.object({
  diagramType: z.enum([
    'architecture',
    'flowchart',
    'sequence',
    'class',
    'entity_relationship',
    'network',
    'org_chart',
    'mindmap',
    'other',
  ]),
  title: z.string().describe('Title of the diagram if visible'),
  components: z.array(
    z.object({
      name: z.string(),
      type: z.string().describe('Type of component (e.g., service, database, user)'),
      description: z.string(),
    })
  ),
  connections: z.array(
    z.object({
      from: z.string(),
      to: z.string(),
      label: z.string().describe('Connection label or description'),
      direction: z.enum(['unidirectional', 'bidirectional', 'none']),
    })
  ),
  summary: z.string().describe('Overall description of what the diagram shows'),
})

async function analyzeDiagram(imagePath: string, context?: string): Promise<z.infer<typeof DiagramAnalysisSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: DiagramAnalysisSchema }),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Analyze this diagram. Extract all components and their connections.
${context ? `Context: ${context}` : ''}`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

export { analyzeDiagram }
```

### Screenshot Analysis

Analyzing screenshots is one of the most practical vision applications — understanding UI states, error messages, and application behavior from visual evidence.

```typescript
// src/multimodal/screenshot-analysis.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { readFile } from 'fs/promises'

const ScreenshotAnalysisSchema = z.object({
  applicationType: z.string().describe('What application is shown'),
  currentState: z.string().describe('What state/page the application is in'),
  visibleElements: z.array(
    z.object({
      element: z.string().describe('UI element description'),
      state: z.string().describe('Current state (e.g., active, disabled, error)'),
    })
  ),
  errors: z
    .array(
      z.object({
        errorText: z.string(),
        errorType: z.enum(['validation', 'network', 'permission', 'server', 'client', 'unknown']),
        suggestedFix: z.string(),
      })
    )
    .describe('Any error messages visible'),
  actionSuggestion: z.string().describe('What the user should do next'),
})

async function analyzeScreenshot(
  imagePath: string,
  userQuestion?: string
): Promise<z.infer<typeof ScreenshotAnalysisSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ScreenshotAnalysisSchema }),
    system: `You are a technical support expert. Analyze the screenshot
to understand the application state, identify any errors, and suggest
next steps.`,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: userQuestion ?? 'What is happening in this screenshot? Are there any errors?',
          },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

export { analyzeScreenshot }
```

> **Beginner Note:** Multi-modal models are remarkably good at "reading" screenshots, even complex ones with multiple panels, modals, and overlapping elements. However, they can struggle with very small text, low-resolution images, and unusual fonts. Always test with your actual use case before relying on vision analysis in production.

> **Advanced Note:** For OCR-heavy applications (processing thousands of document scans), consider using dedicated OCR services (Google Cloud Vision, AWS Textract) for the text extraction step and then using LLMs only for understanding and structuring the extracted text. Dedicated OCR is faster and cheaper per page than sending every image to a multi-modal LLM.

---

## Section 4: Image + Text Prompting

### Effective Multi-modal Prompts

Prompting with images requires different techniques than text-only prompting. The model needs guidance on what to look at, what to extract, and how to relate the image to the text.

```typescript
// src/multimodal/image-text-prompting.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { readFile } from 'fs/promises'

// Pattern 1: Image with specific questions
async function askAboutImage(imagePath: string, questions: string[]): Promise<Record<string, string>> {
  const imageBuffer = await readFile(imagePath)
  const results: Record<string, string> = {}

  // Ask all questions in a single call for efficiency
  const numberedQuestions = questions.map((q, i) => `${i + 1}. ${q}`).join('\n')

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Answer each of the following questions about this image.
Format your response as numbered answers matching the question numbers.

${numberedQuestions}`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  // Parse numbered answers
  const answerLines = result.text.split('\n')
  let currentNumber = 0
  let currentAnswer: string[] = []

  for (const line of answerLines) {
    const numberMatch = line.match(/^(\d+)\.\s*/)
    if (numberMatch) {
      if (currentNumber > 0 && currentNumber <= questions.length) {
        results[questions[currentNumber - 1]] = currentAnswer.join('\n').trim()
      }
      currentNumber = parseInt(numberMatch[1])
      currentAnswer = [line.replace(/^\d+\.\s*/, '')]
    } else {
      currentAnswer.push(line)
    }
  }
  // Save last answer
  if (currentNumber > 0 && currentNumber <= questions.length) {
    results[questions[currentNumber - 1]] = currentAnswer.join('\n').trim()
  }

  return results
}

// Pattern 2: Image with reference text
async function analyzeWithContext(imagePath: string, referenceText: string, task: string): Promise<string> {
  const imageBuffer = await readFile(imagePath)

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Reference information:
${referenceText}

Task: ${task}

Use both the image and the reference information to complete the task.`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return result.text
}

// Pattern 3: Sequential image analysis (before/after)
async function beforeAfterAnalysis(beforePath: string, afterPath: string, analysisPrompt: string): Promise<string> {
  const [beforeBuffer, afterBuffer] = await Promise.all([readFile(beforePath), readFile(afterPath)])

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'BEFORE:' },
          { type: 'image', image: beforeBuffer },
          { type: 'text', text: 'AFTER:' },
          { type: 'image', image: afterBuffer },
          {
            type: 'text',
            text: analysisPrompt,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return result.text
}

// Pattern 4: Image grounding — verify claims against image
const ImageGroundingSchema = z.object({
  claims: z.array(
    z.object({
      claim: z.string(),
      supportedByImage: z.boolean(),
      evidence: z.string().describe('What in the image supports or contradicts this claim'),
    })
  ),
})

async function groundClaimsInImage(imagePath: string, claims: string[]): Promise<z.infer<typeof ImageGroundingSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ImageGroundingSchema }),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `For each claim below, determine if it is supported by what you can see in the image.

Claims:
${claims.map((c, i) => `${i + 1}. ${c}`).join('\n')}`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

export { askAboutImage, analyzeWithContext, beforeAfterAnalysis, groundClaimsInImage }
```

> **Beginner Note:** When prompting with images, be specific about what you want the model to focus on. "Describe this image" produces vague responses. "List all error messages visible in this screenshot and explain what each one means" produces actionable output.

> **Advanced Note:** The order of image and text content in the message matters for some models. Placing the image first and the question after tends to produce better results than question-first, because the model processes the image before seeing what to look for. For multi-image prompts, label each image ("BEFORE:", "AFTER:" or "Image 1:", "Image 2:") to help the model track which image to reference.

---

## Section 5: Audio Transcription

### Whisper Pipeline

Audio input is not natively supported by all providers in the Vercel AI SDK. The standard approach is a two-step pipeline: transcribe audio to text using Whisper, then process the text with your LLM.

```typescript
// src/multimodal/audio-transcription.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { readFile } from 'fs/promises'

// Using OpenAI's Whisper API for transcription
// This is separate from the Vercel AI SDK's generateText

interface TranscriptionResult {
  text: string
  language: string
  duration: number
  segments: Array<{
    start: number
    end: number
    text: string
  }>
}

async function transcribeAudio(
  audioPath: string,
  options: {
    language?: string
    prompt?: string
  } = {}
): Promise<TranscriptionResult> {
  const audioBuffer = await readFile(audioPath)
  const audioBlob = new Blob([audioBuffer])

  const formData = new FormData()
  formData.append('file', audioBlob, audioPath.split('/').pop() ?? 'audio.wav')
  formData.append('model', 'whisper-1')
  formData.append('response_format', 'verbose_json')

  if (options.language) {
    formData.append('language', options.language)
  }
  if (options.prompt) {
    formData.append('prompt', options.prompt)
  }

  const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Transcription failed: ${response.statusText}`)
  }

  const data = (await response.json()) as {
    text: string
    language: string
    duration: number
    segments: Array<{
      start: number
      end: number
      text: string
    }>
  }

  return {
    text: data.text,
    language: data.language,
    duration: data.duration,
    segments: data.segments ?? [],
  }
}

// Full audio processing pipeline: transcribe -> analyze
async function processAudio(
  audioPath: string,
  analysisPrompt: string
): Promise<{
  transcription: string
  analysis: string
}> {
  // Step 1: Transcribe
  console.log('Transcribing audio...')
  const transcription = await transcribeAudio(audioPath)
  console.log(`Transcribed: ${transcription.text.length} characters, ${transcription.duration}s`)

  // Step 2: Analyze with LLM
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are analyzing a transcribed audio recording.
The transcription may contain minor errors from speech-to-text.`,
    messages: [
      {
        role: 'user',
        content: `Transcription:\n${transcription.text}\n\n${analysisPrompt}`,
      },
    ],
  })

  return {
    transcription: transcription.text,
    analysis: result.text,
  }
}

// Meeting notes extraction from audio
import { z } from 'zod'
import { generateText, Output } from 'ai'

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

async function extractMeetingNotes(audioPath: string): Promise<z.infer<typeof MeetingNotesSchema>> {
  const transcription = await transcribeAudio(audioPath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: MeetingNotesSchema }),
    system: `Extract structured meeting notes from this transcription.
Be thorough with action items and decisions. If a participant's name
is unclear, use [Speaker N].`,
    messages: [
      {
        role: 'user',
        content: transcription.text,
      },
    ],
    temperature: 0,
  })

  return output
}

export { transcribeAudio, processAudio, extractMeetingNotes }
```

> **Beginner Note:** Whisper's API accepts files up to 25MB. For longer audio (meetings, lectures), split the audio into segments first. Many audio processing libraries (like `ffmpeg`) can split audio at silence boundaries for cleaner segments.

> **Advanced Note:** For real-time audio processing (live transcription), use the OpenAI Realtime API or WebSocket-based transcription services rather than the batch Whisper API. The Vercel AI SDK has experimental support for real-time audio through the OpenAI provider.

---

## Section 6: Multi-modal RAG

### Embedding Images for Retrieval

Standard RAG embeds text chunks for retrieval. Multi-modal RAG extends this to images. The approach: describe each image with text, embed the description, and store it alongside the image reference.

```typescript
// src/multimodal/multimodal-rag.ts

import { generateText, embed, cosineSimilarity } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { readFile } from 'fs/promises'

interface MultiModalDocument {
  id: string
  type: 'text' | 'image' | 'audio'
  content: string // Text content or path for non-text
  description: string // Text description for embedding
  embedding: number[]
  metadata: Record<string, string>
}

// Generate a rich description of an image for embedding
async function describeImageForEmbedding(imagePath: string, documentContext?: string): Promise<string> {
  const imageBuffer = await readFile(imagePath)

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Provide a detailed description of this image for search indexing.
Include:
- What the image shows (type, subject matter)
- All visible text
- Key visual elements and their relationships
- Data or information conveyed (for charts/diagrams)
- Technical details visible

${documentContext ? `Document context: ${documentContext}` : ''}

Be specific and thorough. This description will be used to find this
image through text search.`,
          },
        ],
      },
    ],
    temperature: 0,
    maxOutputTokens: 500,
  })

  return result.text
}

// Build a multi-modal index
async function buildMultiModalIndex(
  items: Array<{
    id: string
    type: 'text' | 'image'
    content: string
    metadata?: Record<string, string>
  }>,
  embeddingModel: Parameters<typeof embed>[0]['model']
): Promise<MultiModalDocument[]> {
  const documents: MultiModalDocument[] = []

  for (const item of items) {
    let description: string

    if (item.type === 'image') {
      description = await describeImageForEmbedding(item.content)
      console.log(`Described image: ${item.id} (${description.slice(0, 80)}...)`)
    } else {
      description = item.content
    }

    const { embedding } = await embed({
      model: embeddingModel,
      value: description,
    })

    documents.push({
      id: item.id,
      type: item.type,
      content: item.content,
      description,
      embedding,
      metadata: item.metadata ?? {},
    })
  }

  return documents
}

// Search across text and images
async function multiModalSearch(
  query: string,
  index: MultiModalDocument[],
  embeddingModel: Parameters<typeof embed>[0]['model'],
  topK: number = 5
): Promise<
  Array<{
    document: MultiModalDocument
    score: number
  }>
> {
  const { embedding: queryEmbedding } = await embed({
    model: embeddingModel,
    value: query,
  })

  const scored = index.map(doc => ({
    document: doc,
    score: cosineSimilarity(queryEmbedding, doc.embedding),
  }))

  return scored.sort((a, b) => b.score - a.score).slice(0, topK)
}

// Generate answer from multi-modal context
async function multiModalRAG(
  query: string,
  searchResults: Array<{
    document: MultiModalDocument
    score: number
  }>
): Promise<string> {
  // Build multi-modal message content
  const contextParts: Array<{ type: 'text'; text: string } | { type: 'image'; image: Buffer }> = []

  for (const result of searchResults) {
    if (result.document.type === 'image') {
      // Include the actual image
      const imageBuffer = await readFile(result.document.content)
      contextParts.push({
        type: 'text',
        text: `[Image: ${result.document.description.slice(0, 100)}...]`,
      })
      contextParts.push({
        type: 'image',
        image: imageBuffer,
      })
    } else {
      contextParts.push({
        type: 'text',
        text: `[Document]: ${result.document.content}`,
      })
    }
  }

  contextParts.push({
    type: 'text',
    text: `\nQuestion: ${query}\n\nAnswer the question using the documents and images above.`,
  })

  const response = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      {
        role: 'user',
        content: contextParts,
      },
    ],
  })

  return response.text
}

export { describeImageForEmbedding, buildMultiModalIndex, multiModalSearch, multiModalRAG, type MultiModalDocument }
```

> **Beginner Note:** The "describe then embed" approach is the simplest way to make images searchable. The description acts as a text proxy for the image, allowing standard text embedding and retrieval. The quality of the description directly determines retrieval quality — invest in a good description prompt.

> **Advanced Note:** For production multi-modal RAG, consider using CLIP or SigLIP embedding models that natively embed both text and images into the same vector space. This avoids the description step entirely and supports true cross-modal retrieval. The Vercel AI SDK does not natively support CLIP, but you can integrate it as a custom embedding provider.

---

## Section 7: Structured Extraction from Images

### Tables from Images

One of the most practical vision applications is extracting structured data from images of tables, forms, and charts.

```typescript
// src/multimodal/image-extraction.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { readFile } from 'fs/promises'

// Extract table data from an image
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

async function extractTableFromImage(imagePath: string): Promise<z.infer<typeof ImageTableSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ImageTableSchema }),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Extract all tables from this image. For each table:
- Identify column headers
- Extract each row's cell values
- Note any footnotes or annotations
- If a cell is empty, use ""
- If a cell is unclear, use "[unclear]" and lower the confidence`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

// Extract form data from an image
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

async function extractFormData(imagePath: string): Promise<z.infer<typeof FormDataSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: FormDataSchema }),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Extract all form fields from this image. For each field:
- Identify the label
- Read the value (filled in or selected)
- Determine the field type
- Note if the field appears required (asterisk, bold, etc.)
- Note if the field is filled or empty`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

// Extract chart data from an image
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

async function extractChartData(imagePath: string): Promise<z.infer<typeof ChartDataSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ChartDataSchema }),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          {
            type: 'text',
            text: `Extract data from this chart. Read:
- Chart type and title
- Axis labels
- All data points (estimate values from the visual if exact numbers are not labeled)
- Observable trends

For bar/pie charts: extract category labels and values.
For line charts: extract data points along the line.
For scatter plots: estimate x,y coordinates of visible points.`,
          },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

export { extractTableFromImage, extractFormData, extractChartData }
```

> **Beginner Note:** Chart data extraction is approximate — the model estimates values from the visual representation. For precise data extraction, always verify extracted numbers against the source data. For tables and forms, accuracy is much higher because the values are explicit text.

> **Advanced Note:** For high-accuracy table extraction from document scans, consider combining multi-modal LLM extraction with traditional computer vision techniques. Use OpenCV or similar libraries for table detection (finding table boundaries) and the LLM for cell content extraction. This hybrid approach is more reliable than either alone.

---

## Section 8: Limitations and Gotchas

### What Multi-modal Models Get Wrong

Multi-modal models have systematic failure modes that you need to know about and design around.

```typescript
// src/multimodal/limitations.ts

interface LimitationExample {
  category: string
  description: string
  workaround: string
  severity: 'high' | 'medium' | 'low'
}

const KNOWN_LIMITATIONS: LimitationExample[] = [
  {
    category: 'Spatial reasoning',
    description: 'Models may get left/right, above/below relationships wrong, especially in complex layouts',
    workaround: 'Ask the model to describe spatial relationships explicitly and verify',
    severity: 'medium',
  },
  {
    category: 'Small text',
    description: 'Text smaller than ~12px in a screenshot may be misread or missed entirely',
    workaround: 'Crop the relevant area and resize to make text larger before sending',
    severity: 'high',
  },
  {
    category: 'Counting',
    description: "Models are bad at counting objects in images (e.g., 'how many people are in this photo?')",
    workaround: 'For counting tasks, use object detection models (YOLO, etc.) instead of multi-modal LLMs',
    severity: 'medium',
  },
  {
    category: 'Hallucinated text',
    description: "Models may 'read' text that is not actually in the image, especially when the image is blurry",
    workaround: 'Cross-reference extracted text with the original image; use confidence scoring',
    severity: 'high',
  },
  {
    category: 'Chart value estimation',
    description: 'Numerical values read from charts are approximate, not exact',
    workaround: 'Always verify chart-extracted numbers against source data; use wide tolerance',
    severity: 'medium',
  },
  {
    category: 'Resolution sensitivity',
    description: 'Image resolution below 200 DPI significantly degrades text extraction accuracy',
    workaround: 'Upscale low-resolution images before sending; target 300+ DPI for documents',
    severity: 'high',
  },
  {
    category: 'Multi-page documents',
    description: 'Sending many pages as separate images is expensive and may lose inter-page context',
    workaround: 'Use dedicated PDF extraction for text; use vision only for images and diagrams within PDFs',
    severity: 'low',
  },
  {
    category: 'Handwriting',
    description: 'Handwriting recognition varies by quality; messy handwriting may be misread',
    workaround: 'Ask the model to flag uncertain words with [unclear]; provide context',
    severity: 'medium',
  },
]

// Defensive multi-modal function with validation
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { readFile } from 'fs/promises'

const ValidatedExtractionSchema = z.object({
  extraction: z.string().describe('The extracted content'),
  confidence: z.number().min(0).max(1),
  uncertainAreas: z.array(z.string()).describe('Parts of the extraction that may be incorrect'),
  suggestedVerification: z.array(z.string()).describe('What a human should double-check'),
})

async function defensiveExtraction(
  imagePath: string,
  prompt: string
): Promise<z.infer<typeof ValidatedExtractionSchema>> {
  const imageBuffer = await readFile(imagePath)

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ValidatedExtractionSchema }),
    system: `You are a careful document analyst. Extract information from
the image, but be honest about uncertainty:
- If text is blurry or unclear, note it in uncertainAreas
- If you are guessing at a value, flag it
- Set confidence based on overall extraction quality
- Suggest specific things a human should verify`,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', image: imageBuffer },
          { type: 'text', text: prompt },
        ],
      },
    ],
    temperature: 0,
  })

  return output
}

export { KNOWN_LIMITATIONS, defensiveExtraction }
```

### Cost Considerations

```typescript
// src/multimodal/cost-estimation.ts

interface ImageCostEstimate {
  estimatedInputTokens: number
  estimatedCostUSD: number
  recommendation: string
}

function estimateImageCost(
  widthPx: number,
  heightPx: number,
  provider: 'anthropic' | 'openai' = 'anthropic'
): ImageCostEstimate {
  // Anthropic pricing: images are tiled into 1568x1568 blocks
  // Each block costs ~1600 tokens
  const maxDim = Math.max(widthPx, heightPx)
  let tokens: number

  if (maxDim <= 384) {
    tokens = 170 // Thumbnail
  } else if (maxDim <= 768) {
    tokens = 800 // Medium
  } else {
    // Large images are tiled
    const tilesX = Math.ceil(widthPx / 1568)
    const tilesY = Math.ceil(heightPx / 1568)
    tokens = tilesX * tilesY * 1600
  }

  // Claude Sonnet pricing: $3/M input tokens
  const costUSD = (tokens / 1_000_000) * 3.0

  let recommendation = 'Good size for analysis'
  if (tokens > 3200) {
    recommendation = 'Consider resizing — image uses many tokens. Reduce to 1568px max dimension.'
  }
  if (tokens < 200) {
    recommendation = 'Very small image — text may be hard to read. Consider using a higher resolution.'
  }

  return {
    estimatedInputTokens: tokens,
    estimatedCostUSD: costUSD,
    recommendation,
  }
}

export { estimateImageCost }
```

> **Beginner Note:** The biggest gotcha with multi-modal models is hallucinated text — the model "reads" text that is not actually in the image. Always verify critical extracted text (names, numbers, codes) against the original image. Never trust OCR output from a multi-modal model for financial, legal, or medical data without human verification.

> **Advanced Note:** For production applications, build a confidence-based workflow: extract with the multi-modal model, flag low-confidence items, route those to human reviewers. This gives you the speed of automation with the accuracy of human oversight. Track accuracy metrics over time to identify systematic failure patterns in your specific domain.

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

## Exercises

### Exercise 1: Multi-modal Q&A System

**Objective:** Build a Q&A system that handles both text questions and questions with attached screenshots.

**Specification:**

1. Create `src/exercises/ex13-multimodal-qa.ts`
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
// tests/ex13.test.ts
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

1. Create `src/exercises/ex13-image-extraction.ts`
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
// tests/ex13-extraction.test.ts
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

This completes Part III: Advanced Retrieval. You now have a comprehensive retrieval toolkit — vector search, hybrid search, reranking, knowledge graphs, document processing, and multi-modal understanding. In Part IV, you will learn to build agents that use these retrieval capabilities as tools, orchestrating complex multi-step workflows autonomously.
