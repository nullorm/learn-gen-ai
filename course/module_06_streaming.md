# Module 6: Streaming & Real-time

## Learning Objectives

- Understand why streaming matters for user experience and perceived performance
- Use `streamText` for real-time text generation with async iteration and callbacks
- Stream structured output with `streamText` and `Output.object()` for progressive data delivery
- Implement Server-Sent Events (SSE) endpoints for web-based streaming
- Handle backpressure, flow control, and error recovery in streaming pipelines
- Build UI patterns for streaming responses including typewriter effects and progressive disclosure

---

## Why Should I Care?

An LLM generating a 500-token response takes 2-5 seconds. If the user stares at a blank screen for 5 seconds, they assume the application is broken. If they see text appearing word by word within 200 milliseconds, they feel the application is fast and responsive — even though the total time is identical.

This is the core argument for streaming: **perceived performance matters more than actual performance.** Time-to-first-token (TTFT) — the delay before the first word appears — is the single most important latency metric for LLM applications. Streaming reduces TTFT from seconds to milliseconds because you start displaying output as soon as the first token is generated, rather than waiting for the complete response.

Beyond perception, streaming enables real architectural patterns. You can process tokens as they arrive, build progressive UIs that show partial structured data, pipe LLM output through transformation chains, and cancel generation mid-stream if the output is going off track. These patterns are essential for production applications.

---

## Connection to Other Modules

- **Module 1 (Setup)** introduced `streamText` briefly. This module dives deep into streaming mechanics.
- **Module 4 (Conversations)** used `generateText` for conversations. Streaming makes conversations feel instant.
- **Module 3 (Structured Output)** introduced `generateText` with `Output.object()`. Here you learn `streamText` with `Output.object()` for progressive structured data.
- **Module 7 (Tool Use)** involves tool calls during streaming, where the stream pauses for tool execution.
- **Module 5 (Long Context)** noted latency concerns with large contexts. Streaming mitigates this.

---

## Section 1: Why Streaming Matters

### Time-to-First-Token (TTFT)

TTFT measures how long the user waits before seeing any output. It is the most critical latency metric for interactive LLM applications:

```typescript
import { generateText, streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Non-streaming: user waits for the entire response
async function measureNonStreaming(): Promise<void> {
  const start = performance.now()

  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: 'Explain the theory of relativity in 3 paragraphs.',
  })

  const totalTime = performance.now() - start
  console.log(`Non-streaming: ${totalTime.toFixed(0)}ms to see anything`)
  console.log(`Response length: ${text.length} characters`)
}

// Streaming: user sees first token almost immediately
async function measureStreaming(): Promise<void> {
  const start = performance.now()
  let firstTokenTime: number | null = null

  const result = streamText({
    model: mistral('mistral-small-latest'),
    prompt: 'Explain the theory of relativity in 3 paragraphs.',
  })

  let fullText = ''
  for await (const chunk of result.textStream) {
    if (firstTokenTime === null) {
      firstTokenTime = performance.now() - start
    }
    fullText += chunk
  }

  const totalTime = performance.now() - start
  console.log(`Streaming TTFT: ${firstTokenTime?.toFixed(0)}ms`)
  console.log(`Streaming total: ${totalTime.toFixed(0)}ms`)
  console.log(`Response length: ${fullText.length} characters`)
}

await measureNonStreaming()
await measureStreaming()
```

Typical results:

| Metric          | Non-streaming | Streaming   |
| --------------- | ------------- | ----------- |
| TTFT            | 2000-5000ms   | 200-500ms   |
| Total time      | 2000-5000ms   | 2000-5000ms |
| User perception | "Slow"        | "Fast"      |

> **Beginner Note:** The total generation time is roughly the same for both approaches. The model still generates tokens at the same speed. The difference is that streaming shows each token as it is generated, while non-streaming waits for all tokens before showing anything.

### Perceived Performance

Research on user interface responsiveness shows:

- **< 100ms**: Feels instant
- **100-300ms**: Noticeable but acceptable
- **300-1000ms**: User notices the delay
- **> 1000ms**: User loses focus, may assume failure

Streaming brings TTFT into the 200-500ms range, keeping users engaged even for complex, multi-second generations.

---

## Section 2: streamText Deep Dive

### Basic Usage

The `streamText` function returns a stream result object with multiple ways to consume the stream:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Method 1: Async iteration over text chunks
async function streamWithAsyncIteration(): Promise<void> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    prompt: 'Write a haiku about TypeScript.',
  })

  // Each chunk is a string fragment (usually a few characters or a word)
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk)
  }
  console.log() // newline at end
}

await streamWithAsyncIteration()
```

### Consuming the Full Result

After streaming completes, you can access the full response and metadata:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function streamWithFullResult(): Promise<void> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    messages: [
      { role: 'system', content: 'You are a concise assistant.' },
      { role: 'user', content: 'What are the three laws of thermodynamics?' },
    ],
  })

  // Stream the text
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk)
  }
  console.log()

  // After streaming, access the complete result
  const fullText = await result.text
  const usage = await result.usage
  const finishReason = await result.finishReason

  console.log(`\n--- Metadata ---`)
  console.log(`Full text length: ${fullText.length}`)
  console.log(`Finish reason: ${finishReason}`)
  if (usage) {
    console.log(`Input tokens: ${usage.inputTokens}`)
    console.log(`Output tokens: ${usage.outputTokens}`)
  }
}

await streamWithFullResult()
```

### Streaming with Messages (Multi-turn)

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

async function streamConversation(messages: Message[]): Promise<string> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    messages,
  })

  let fullText = ''
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk)
    fullText += chunk
  }
  console.log()

  return fullText
}

// Build a streaming conversation
const conversation: Message[] = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'What is recursion in programming?' },
]

const response1 = await streamConversation(conversation)
conversation.push({ role: 'assistant', content: response1 })

conversation.push({ role: 'user', content: 'Give me a simple example in TypeScript.' })
const response2 = await streamConversation(conversation)
```

### Callbacks

`streamText` supports callbacks for fine-grained control:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Explain quantum entanglement.',

  onChunk({ chunk }) {
    // Called for each chunk (text, tool calls, etc.)
    if (chunk.type === 'text-delta') {
      // chunk.textDelta contains the text fragment
    }
  },

  onFinish({ text, usage, finishReason }) {
    console.log('\n--- Stream finished ---')
    console.log(`Total text: ${text.length} characters`)
    console.log(`Finish reason: ${finishReason}`)
    if (usage) {
      console.log(`Tokens: ${usage.inputTokens} in, ${usage.outputTokens} out`)
    }
  },

  onError({ error }) {
    console.error('Stream error:', error)
  },
})

// Still need to consume the stream
for await (const chunk of result.textStream) {
  process.stdout.write(chunk)
}
```

### Stream Timing Analysis

Measure token generation speed and timing:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface StreamMetrics {
  ttft: number // Time to first token (ms)
  totalTime: number // Total generation time (ms)
  tokenCount: number // Number of text chunks received
  tokensPerSecond: number // Generation speed
  chunks: Array<{ text: string; timestamp: number }>
}

async function measureStream(prompt: string): Promise<StreamMetrics> {
  const startTime = performance.now()
  let firstTokenTime = 0
  const chunks: Array<{ text: string; timestamp: number }> = []

  const result = streamText({
    model: mistral('mistral-small-latest'),
    prompt,
  })

  for await (const chunk of result.textStream) {
    const now = performance.now()
    if (chunks.length === 0) {
      firstTokenTime = now - startTime
    }
    chunks.push({ text: chunk, timestamp: now - startTime })
  }

  const totalTime = performance.now() - startTime

  return {
    ttft: firstTokenTime,
    totalTime,
    tokenCount: chunks.length,
    tokensPerSecond: (chunks.length / totalTime) * 1000,
    chunks,
  }
}

const metrics = await measureStream('List 10 programming languages and their primary use cases.')

console.log(`TTFT: ${metrics.ttft.toFixed(0)}ms`)
console.log(`Total time: ${metrics.totalTime.toFixed(0)}ms`)
console.log(`Chunks received: ${metrics.tokenCount}`)
console.log(`Speed: ${metrics.tokensPerSecond.toFixed(1)} chunks/sec`)
```

> **Advanced Note:** Token generation speed varies by model, prompt complexity, and server load. Claude models typically generate 50-100 tokens per second. The first few tokens may be slower due to prefill computation (processing your input). After prefill, generation speed is relatively constant.

---

## Section 3: Streaming Structured Output

### streamText with Output.object()

`streamText` with `Output.object()` lets you stream structured data as it is generated. Instead of waiting for the complete JSON object, you get partial updates as each field is populated:

```typescript
import { streamText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const RecipeSchema = z.object({
  name: z.string().describe('Name of the recipe'),
  prepTime: z.string().describe('Preparation time'),
  cookTime: z.string().describe('Cooking time'),
  servings: z.number().describe('Number of servings'),
  ingredients: z.array(
    z.object({
      item: z.string(),
      amount: z.string(),
    })
  ),
  steps: z.array(z.string()),
  tips: z.array(z.string()).optional(),
})

type Recipe = z.infer<typeof RecipeSchema>

async function streamRecipe(dish: string): Promise<Recipe> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: RecipeSchema }),
    prompt: `Create a detailed recipe for ${dish}.`,
  })

  // partialOutputStream yields the object as it's being built
  for await (const partialObject of result.partialOutputStream) {
    console.clear()
    console.log('--- Recipe (streaming) ---')

    if (partialObject.name) {
      console.log(`Name: ${partialObject.name}`)
    }
    if (partialObject.prepTime) {
      console.log(`Prep: ${partialObject.prepTime}`)
    }
    if (partialObject.cookTime) {
      console.log(`Cook: ${partialObject.cookTime}`)
    }
    if (partialObject.ingredients) {
      console.log(`Ingredients: ${partialObject.ingredients.length} items`)
    }
    if (partialObject.steps) {
      console.log(`Steps: ${partialObject.steps.length} so far`)
    }
  }

  // Get the final complete object
  const finalObject = await result.output
  return finalObject
}

const recipe = await streamRecipe('Thai green curry')
console.log('\n--- Final Recipe ---')
console.log(JSON.stringify(recipe, null, 2))
```

> **Beginner Note:** The partial object may have missing or incomplete fields. A string field might contain only the first few words. An array might have only the first item. Your UI should handle these partial states gracefully.

### Progressive UI Updates

```typescript
import { streamText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const AnalysisSchema = z.object({
  summary: z.string().describe('One paragraph summary'),
  sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']),
  keyTopics: z.array(z.string()).describe('Main topics discussed'),
  entities: z.array(
    z.object({
      name: z.string(),
      type: z.enum(['person', 'organization', 'location', 'product']),
    })
  ),
  actionItems: z.array(z.string()).optional(),
  confidence: z.number().min(0).max(1),
})

async function streamAnalysis(text: string): Promise<void> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: AnalysisSchema }),
    prompt: `Analyze the following text:\n\n${text}`,
  })

  for await (const partial of result.partialOutputStream) {
    // In a real UI, you would update DOM elements or send state updates
    const updates: string[] = []

    if (partial.sentiment) {
      updates.push(`Sentiment: ${partial.sentiment}`)
    }
    if (partial.keyTopics?.length) {
      updates.push(`Topics: ${partial.keyTopics.join(', ')}`)
    }
    if (partial.entities?.length) {
      updates.push(`Entities found: ${partial.entities.length}`)
    }
    if (partial.summary) {
      updates.push(`Summary: ${partial.summary.slice(0, 80)}...`)
    }
    if (partial.confidence !== undefined) {
      updates.push(`Confidence: ${(partial.confidence * 100).toFixed(0)}%`)
    }

    if (updates.length > 0) {
      console.log(updates.join(' | '))
    }
  }
}

await streamAnalysis('The quarterly results exceeded expectations with a 15% revenue increase...')
```

### Handling Partial Objects Safely

```typescript
import { z } from 'zod'

/**
 * When working with partial objects from streamText with Output.object(),
 * fields may be undefined even if they are required in the schema.
 * Use type-safe accessors.
 */
function safeDisplay<T extends Record<string, unknown>>(
  partial: Partial<T>,
  fields: Array<keyof T>
): Record<string, string> {
  const result: Record<string, string> = {}

  for (const field of fields) {
    const value = partial[field]
    if (value !== undefined && value !== null) {
      if (Array.isArray(value)) {
        result[field as string] = `${value.length} items`
      } else if (typeof value === 'object') {
        result[field as string] = JSON.stringify(value)
      } else {
        result[field as string] = String(value)
      }
    } else {
      result[field as string] = '(loading...)'
    }
  }

  return result
}
```

---

## Section 4: Server-Sent Events (SSE)

### What is SSE?

Server-Sent Events is a standard HTTP protocol for streaming data from server to client. Unlike WebSockets, SSE is unidirectional (server to client), uses standard HTTP, and automatically reconnects on failure.

SSE format:

```
data: {"text": "Hello"}

data: {"text": " world"}

data: [DONE]
```

### Building an SSE Streaming Endpoint

Using Bun's built-in HTTP server:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const server = Bun.serve({
  port: 3000,

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url)

    if (url.pathname === '/api/chat' && request.method === 'POST') {
      return handleChatStream(request)
    }

    if (url.pathname === '/') {
      return new Response(HTML_PAGE, {
        headers: { 'Content-Type': 'text/html' },
      })
    }

    return new Response('Not found', { status: 404 })
  },
})

async function handleChatStream(request: Request): Promise<Response> {
  const body = await request.json()
  const { messages } = body as { messages: Array<{ role: string; content: string }> }

  const result = streamText({
    model: mistral('mistral-small-latest'),
    messages: messages as any,
  })

  // Convert the AI SDK stream to an SSE response
  const encoder = new TextEncoder()

  const stream = new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of result.textStream) {
          const data = JSON.stringify({ type: 'text', content: chunk })
          controller.enqueue(encoder.encode(`data: ${data}\n\n`))
        }

        // Send completion signal
        const usage = await result.usage
        const doneData = JSON.stringify({
          type: 'done',
          usage: {
            inputTokens: usage?.inputTokens,
            outputTokens: usage?.outputTokens,
          },
        })
        controller.enqueue(encoder.encode(`data: ${doneData}\n\n`))
        controller.close()
      } catch (error) {
        const errorData = JSON.stringify({
          type: 'error',
          message: error instanceof Error ? error.message : 'Unknown error',
        })
        controller.enqueue(encoder.encode(`data: ${errorData}\n\n`))
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'Access-Control-Allow-Origin': '*',
    },
  })
}

const HTML_PAGE = `<!DOCTYPE html>
<html>
<head><title>Streaming Chat</title></head>
<body>
  <div id="chat" style="white-space: pre-wrap; font-family: monospace;"></div>
  <input id="input" type="text" placeholder="Type a message..." style="width: 80%;" />
  <button onclick="sendMessage()">Send</button>
  <script>
    const chatDiv = document.getElementById('chat');
    const input = document.getElementById('input');
    const messages = [];

    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;

      messages.push({ role: 'user', content: text });
      chatDiv.textContent += '\\nYou: ' + text + '\\n';
      input.value = '';

      chatDiv.textContent += 'Assistant: ';

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'text') {
              assistantText += data.content;
              chatDiv.textContent += data.content;
            }
          }
        }
      }

      messages.push({ role: 'assistant', content: assistantText });
      chatDiv.textContent += '\\n';
    }

    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') sendMessage();
    });
  </script>
</body>
</html>`

console.log(`Server running at http://localhost:${server.port}`)
```

### Using the AI SDK's Built-in Stream Helpers

The Vercel AI SDK provides utilities for converting streams to standard web responses:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function handleStreamRequest(request: Request): Promise<Response> {
  const body = await request.json()

  const result = streamText({
    model: mistral('mistral-small-latest'),
    messages: body.messages,
  })

  // The AI SDK can convert directly to a Response
  return result.toDataStreamResponse()
}

// This returns a response in the AI SDK's data stream format,
// which is compatible with the useChat hook in Next.js / React
```

> **Beginner Note:** The `toDataStreamResponse()` method creates a response in the Vercel AI SDK's own streaming protocol, which works seamlessly with its React hooks (`useChat`, `useCompletion`). If you are building a custom frontend, the manual SSE approach gives you full control.

> **Advanced Note:** The AI SDK data stream protocol encodes multiple data types (text deltas, tool calls, tool results, annotations) in a single stream. This is more capable than plain text SSE but requires the AI SDK client library to parse.

---

## Section 5: Backpressure and Flow Control

### What is Backpressure?

Backpressure occurs when the consumer (your code) processes data slower than the producer (the model) generates it. In streaming scenarios:

- The model generates tokens at ~50-100 tokens/second
- Your code might need to do work per token (database write, transformation, network send)
- If your work takes longer than the token generation interval, you need flow control

### Handling Backpressure in Node.js/Bun Streams

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * Example: Stream tokens while writing each one to a file.
 * The file write might be slower than token generation.
 */
async function streamToFile(prompt: string, outputPath: string): Promise<void> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    prompt,
  })

  const file = Bun.file(outputPath)
  const writer = file.writer()

  let tokenCount = 0

  for await (const chunk of result.textStream) {
    // The for-await-of loop naturally handles backpressure:
    // it will not request the next chunk until the current iteration completes
    writer.write(chunk)
    tokenCount++

    // Periodically flush to disk
    if (tokenCount % 50 === 0) {
      await writer.flush()
    }
  }

  await writer.end()
  console.log(`Wrote ${tokenCount} chunks to ${outputPath}`)
}

await streamToFile('Write a detailed guide to TypeScript generics.', './output/generics-guide.txt')
```

### Rate-Limited Streaming

Control the rate at which tokens are forwarded to the client:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * Buffer tokens and release them at a controlled rate.
 * Useful for creating consistent typewriter effects.
 */
async function* rateLimit(stream: AsyncIterable<string>, minIntervalMs: number): AsyncGenerator<string> {
  let lastEmit = 0

  for await (const chunk of stream) {
    const now = performance.now()
    const elapsed = now - lastEmit

    if (elapsed < minIntervalMs) {
      await new Promise(resolve => setTimeout(resolve, minIntervalMs - elapsed))
    }

    yield chunk
    lastEmit = performance.now()
  }
}

// Usage: emit at most one chunk every 30ms (for smooth typewriter effect)
const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Tell me about the history of computing.',
})

for await (const chunk of rateLimit(result.textStream, 30)) {
  process.stdout.write(chunk)
}
console.log()
```

### Buffered Streaming

Accumulate tokens into larger chunks before processing:

```typescript
/**
 * Buffer stream chunks into sentences or larger units.
 * Useful when downstream processing works better with complete sentences.
 */
async function* bufferBySentence(stream: AsyncIterable<string>): AsyncGenerator<string> {
  let buffer = ''
  const sentenceEnders = /[.!?]\s/

  for await (const chunk of stream) {
    buffer += chunk

    // Check if buffer contains a complete sentence
    const match = buffer.match(sentenceEnders)
    if (match && match.index !== undefined) {
      const endIndex = match.index + match[0].length
      const sentence = buffer.slice(0, endIndex)
      buffer = buffer.slice(endIndex)
      yield sentence
    }
  }

  // Flush remaining buffer
  if (buffer.trim()) {
    yield buffer
  }
}

// Usage: process complete sentences
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Explain photosynthesis in detail.',
})

let sentenceCount = 0
for await (const sentence of bufferBySentence(result.textStream)) {
  sentenceCount++
  console.log(`[Sentence ${sentenceCount}] ${sentence.trim()}`)
}
```

> **Advanced Note:** The `for await...of` loop over an async iterable provides natural backpressure. The producer (AI SDK stream) only generates the next value when the consumer is ready for it. This is built into the JavaScript async iteration protocol. You do not need explicit backpressure mechanisms unless you are bridging to a different streaming system.

---

## Section 6: Stream Transformation

### Filtering Tokens

Remove or modify tokens as they stream through:

````typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * Filter out specific patterns from the stream.
 * Example: remove markdown formatting for plain-text output.
 */
async function* stripMarkdown(stream: AsyncIterable<string>): AsyncGenerator<string> {
  let buffer = ''

  for await (const chunk of stream) {
    buffer += chunk

    // Process complete patterns in the buffer
    // Remove bold markers
    buffer = buffer.replace(/\*\*/g, '')
    // Remove italic markers (single *)
    buffer = buffer.replace(/(?<!\*)\*(?!\*)/g, '')
    // Remove headers
    buffer = buffer.replace(/^#{1,6}\s/gm, '')
    // Remove code block markers
    buffer = buffer.replace(/```\w*\n?/g, '')

    // Emit everything except the last few characters (might be partial pattern)
    if (buffer.length > 5) {
      const toEmit = buffer.slice(0, -5)
      buffer = buffer.slice(-5)
      yield toEmit
    }
  }

  // Flush remaining buffer
  if (buffer) {
    yield buffer
  }
}

const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Explain TypeScript interfaces with code examples.',
})

for await (const chunk of stripMarkdown(result.textStream)) {
  process.stdout.write(chunk)
}
console.log()
````

### Mapping Tokens

Transform each token in the stream:

```typescript
/**
 * Transform stream tokens — for example, convert to uppercase
 * or apply text processing.
 */
async function* mapStream(stream: AsyncIterable<string>, transform: (chunk: string) => string): AsyncGenerator<string> {
  for await (const chunk of stream) {
    yield transform(chunk)
  }
}

/**
 * Add annotations or metadata to the stream.
 */
async function* annotateStream(
  stream: AsyncIterable<string>
): AsyncGenerator<{ text: string; timestamp: number; index: number }> {
  let index = 0
  const startTime = performance.now()

  for await (const chunk of stream) {
    yield {
      text: chunk,
      timestamp: performance.now() - startTime,
      index: index++,
    }
  }
}

// Usage
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Write a short poem about coding.',
})

for await (const annotated of annotateStream(result.textStream)) {
  process.stdout.write(annotated.text)
  // Could log timing data to analytics
}
```

### Tee: Split a Stream

Send the same stream to multiple consumers:

```typescript
/**
 * Split a stream into two independent streams.
 * Both consumers receive all chunks.
 */
function teeStream<T>(stream: AsyncIterable<T>): [AsyncGenerator<T>, AsyncGenerator<T>] {
  const buffer1: T[] = []
  const buffer2: T[] = []
  let done = false
  let resolve1: (() => void) | null = null
  let resolve2: (() => void) | null = null

  // Start consuming the source stream
  ;(async () => {
    for await (const chunk of stream) {
      buffer1.push(chunk)
      buffer2.push(chunk)
      resolve1?.()
      resolve2?.()
    }
    done = true
    resolve1?.()
    resolve2?.()
  })()

  async function* consumer(buffer: T[]): AsyncGenerator<T> {
    while (true) {
      if (buffer.length > 0) {
        yield buffer.shift()!
      } else if (done) {
        return
      } else {
        await new Promise<void>(resolve => {
          if (buffer === buffer1) resolve1 = resolve
          else resolve2 = resolve
        })
      }
    }
  }

  return [consumer(buffer1), consumer(buffer2)]
}

// Usage: stream to both console and file simultaneously
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Explain WebSockets.',
})

const [displayStream, logStream] = teeStream(result.textStream)

// Consumer 1: Display to console
const displayPromise = (async () => {
  for await (const chunk of displayStream) {
    process.stdout.write(chunk)
  }
})()

// Consumer 2: Log to buffer
const logPromise = (async () => {
  let fullLog = ''
  for await (const chunk of logStream) {
    fullLog += chunk
  }
  return fullLog
})()

await displayPromise
const log = await logPromise
console.log(`\n\nLogged ${log.length} characters`)
```

---

## Section 7: Error Handling in Streams

### Types of Stream Errors

Streams can fail at several points:

1. **Connection errors**: Network failure before streaming starts
2. **Mid-stream errors**: Connection drops during generation
3. **Content filter errors**: Model refuses to continue
4. **Rate limit errors**: Too many concurrent requests
5. **Timeout errors**: Stream takes too long

### Comprehensive Error Handling

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface StreamResult {
  text: string
  completed: boolean
  error: string | null
  tokensGenerated: number
}

async function safeStream(prompt: string): Promise<StreamResult> {
  let text = ''
  let tokensGenerated = 0

  try {
    const result = streamText({
      model: mistral('mistral-small-latest'),
      prompt,
      maxOutputTokens: 4096,
      abortSignal: AbortSignal.timeout(30_000), // 30 second timeout
    })

    for await (const chunk of result.textStream) {
      text += chunk
      tokensGenerated++
      process.stdout.write(chunk)
    }

    return { text, completed: true, error: null, tokensGenerated }
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError' || error.name === 'TimeoutError') {
        console.error('\n[Stream timed out]')
        return {
          text,
          completed: false,
          error: 'Timeout: stream exceeded 30 seconds',
          tokensGenerated,
        }
      }

      if (error.message.includes('rate_limit')) {
        console.error('\n[Rate limited]')
        return {
          text,
          completed: false,
          error: 'Rate limited. Retry after delay.',
          tokensGenerated,
        }
      }

      console.error(`\n[Stream error: ${error.message}]`)
      return { text, completed: false, error: error.message, tokensGenerated }
    }

    return {
      text,
      completed: false,
      error: 'Unknown error',
      tokensGenerated,
    }
  }
}

const result = await safeStream('Write a comprehensive guide to error handling.')
console.log(`\n\nCompleted: ${result.completed}`)
console.log(`Tokens: ${result.tokensGenerated}`)
if (result.error) {
  console.log(`Error: ${result.error}`)
}
```

### Retry with Partial Recovery

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * If a stream fails mid-generation, retry with the partial output
 * included as context so the model continues from where it left off.
 */
async function resilientStream(prompt: string, maxRetries: number = 3): Promise<string> {
  let fullText = ''
  let retries = 0

  while (retries <= maxRetries) {
    try {
      const currentPrompt =
        retries === 0 ? prompt : `${prompt}\n\nNote: Continue from where this partial response left off:\n${fullText}`

      const result = streamText({
        model: mistral('mistral-small-latest'),
        prompt: currentPrompt,
        abortSignal: AbortSignal.timeout(30_000),
      })

      for await (const chunk of result.textStream) {
        fullText += chunk
        process.stdout.write(chunk)
      }

      return fullText // Completed successfully
    } catch (error) {
      retries++
      if (retries > maxRetries) {
        console.error(`\nFailed after ${maxRetries} retries.`)
        return fullText // Return what we have
      }

      console.error(`\n[Retry ${retries}/${maxRetries} after error]`)
      // Wait before retrying (exponential backoff)
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, retries)))
    }
  }

  return fullText
}
```

### Cancellation

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * Cancel a stream if it generates unwanted content
 * or exceeds a length threshold.
 */
async function streamWithCancellation(prompt: string, maxChars: number = 1000): Promise<string> {
  const abortController = new AbortController()

  const result = streamText({
    model: mistral('mistral-small-latest'),
    prompt,
    abortSignal: abortController.signal,
  })

  let text = ''

  try {
    for await (const chunk of result.textStream) {
      text += chunk
      process.stdout.write(chunk)

      // Cancel if response is too long
      if (text.length > maxChars) {
        console.log('\n[Cancelling: max length reached]')
        abortController.abort()
        break
      }

      // Cancel if response contains unwanted content
      if (text.includes('[CONFIDENTIAL]')) {
        console.log('\n[Cancelling: sensitive content detected]')
        abortController.abort()
        break
      }
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      // Expected when we abort — not a real error
    } else {
      throw error
    }
  }

  return text
}
```

> **Beginner Note:** Always use `AbortController` for cancellation rather than simply stopping iteration. Stopping iteration without aborting leaves the underlying HTTP connection open, wasting server resources and potentially accruing costs for tokens you are not using.

---

## Section 8: UI Patterns

### Typewriter Effect

The classic pattern where text appears character by character:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * Typewriter effect with configurable speed.
 * The stream provides chunks at model speed;
 * this function spaces them out for visual effect.
 */
async function typewriter(stream: AsyncIterable<string>, charDelayMs: number = 20): Promise<string> {
  let fullText = ''

  for await (const chunk of stream) {
    // Each chunk might contain multiple characters
    for (const char of chunk) {
      process.stdout.write(char)
      fullText += char
      await new Promise(resolve => setTimeout(resolve, charDelayMs))
    }
  }

  return fullText
}

const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Write a dramatic opening paragraph for a novel.',
})

const text = await typewriter(result.textStream, 15)
console.log(`\n\nGenerated ${text.length} characters`)
```

### Progressive Disclosure

Show content section by section as it is generated:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

/**
 * Group streamed content into sections (split by double newline)
 * and reveal each section with a visual transition.
 */
async function progressiveDisclosure(stream: AsyncIterable<string>): Promise<string[]> {
  let buffer = ''
  const sections: string[] = []

  for await (const chunk of stream) {
    buffer += chunk

    // Check for section breaks (double newline)
    const parts = buffer.split('\n\n')

    if (parts.length > 1) {
      // We have at least one complete section
      for (let i = 0; i < parts.length - 1; i++) {
        const section = parts[i].trim()
        if (section) {
          sections.push(section)
          console.log(`\n${'='.repeat(60)}`)
          console.log(`Section ${sections.length}:`)
          console.log('='.repeat(60))
          console.log(section)
        }
      }
      buffer = parts[parts.length - 1] // Keep the incomplete part
    }
  }

  // Handle the last section
  if (buffer.trim()) {
    sections.push(buffer.trim())
    console.log(`\n${'='.repeat(60)}`)
    console.log(`Section ${sections.length}:`)
    console.log('='.repeat(60))
    console.log(buffer.trim())
  }

  return sections
}

const result = streamText({
  model: mistral('mistral-small-latest'),
  prompt: 'Write a 5-section guide to learning TypeScript. Use double newlines between sections.',
})

const sections = await progressiveDisclosure(result.textStream)
console.log(`\nTotal sections: ${sections.length}`)
```

### Streaming Status Indicators

Show progress information alongside the streaming content:

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface StreamStatus {
  state: 'connecting' | 'streaming' | 'complete' | 'error'
  charsReceived: number
  elapsed: number
  speed: number // chars per second
}

async function streamWithStatus(prompt: string, onStatus: (status: StreamStatus) => void): Promise<string> {
  const startTime = performance.now()
  let text = ''

  onStatus({
    state: 'connecting',
    charsReceived: 0,
    elapsed: 0,
    speed: 0,
  })

  try {
    const result = streamText({
      model: mistral('mistral-small-latest'),
      prompt,
    })

    for await (const chunk of result.textStream) {
      text += chunk
      const elapsed = performance.now() - startTime

      onStatus({
        state: 'streaming',
        charsReceived: text.length,
        elapsed,
        speed: (text.length / elapsed) * 1000,
      })
    }

    const elapsed = performance.now() - startTime
    onStatus({
      state: 'complete',
      charsReceived: text.length,
      elapsed,
      speed: (text.length / elapsed) * 1000,
    })

    return text
  } catch (error) {
    const elapsed = performance.now() - startTime
    onStatus({
      state: 'error',
      charsReceived: text.length,
      elapsed,
      speed: text.length > 0 ? (text.length / elapsed) * 1000 : 0,
    })
    throw error
  }
}

// Usage with a status bar
const text = await streamWithStatus('Explain the concept of monads in functional programming.', status => {
  const statusLine = `[${status.state}] ${status.charsReceived} chars | ${status.elapsed.toFixed(0)}ms | ${status.speed.toFixed(0)} chars/sec`
  process.stderr.write(`\r${statusLine}`)
})

console.log(`\n\n${text}`)
```

> **Advanced Note:** In browser-based UIs, consider implementing a "thinking" indicator that appears during the TTFT gap (before the first token arrives). This gives users immediate feedback that their request is being processed. Combine this with streaming text display for the best perceived performance.

> **Local Alternative (Ollama):** All streaming patterns in this module (`streamText`, `streamText` with `Output.object()`, backpressure, AbortController) work identically with `ollama('qwen3.5')`. Local models actually benefit more from streaming since inference is slower — streaming lets users see output immediately rather than waiting for full generation. SSE endpoints work the same regardless of provider.

---

## Summary

In this module, you learned:

1. **Why streaming matters:** Time-to-first-token (TTFT) is the most important latency metric — streaming reduces perceived wait time from seconds to milliseconds.
2. **streamText:** How to use async iteration and callbacks to consume streaming text responses and measure timing metrics like TTFT and tokens per second.
3. **streamText with Output.object():** How to stream structured output progressively, enabling UIs to display partial data as it arrives rather than waiting for the complete object.
4. **Server-Sent Events (SSE):** How to build HTTP streaming endpoints that deliver token-by-token LLM output to web clients using the SSE protocol.
5. **Backpressure and flow control:** How to handle slow consumers, implement rate-limited streaming, and buffer tokens to prevent overwhelming downstream systems.
6. **Stream transformation:** How to filter, map, and split streams using transform streams and the tee pattern for parallel processing.
7. **Error handling:** How to detect and recover from mid-stream errors, implement timeouts, and provide fallback behavior when streaming connections fail.

In Module 7, you will learn how tools transform LLMs from text generators into agents that can take action in the real world.

---

## Quiz

### Question 1 (Easy)

What does TTFT stand for, and why is it the most important streaming metric?

A) Total Time For Transfer — it measures bandwidth
B) Time To First Token — it determines perceived responsiveness
C) Token Transfer Feedback Time — it measures throughput
D) Time To Full Text — it measures total generation time

**Answer: B**

TTFT stands for Time To First Token. It measures the delay between sending a request and receiving the first token of the response. This is the most important metric because it determines when the user first sees output. A TTFT of 200ms feels instant, while a TTFT of 3 seconds (common with non-streaming) feels slow, even if the total generation time is the same.

---

### Question 2 (Easy)

What is the key difference between `streamText` and `generateText`?

A) `streamText` is faster overall
B) `streamText` uses a different model
C) `streamText` yields tokens incrementally as they are generated
D) `streamText` costs less per token

**Answer: C**

`streamText` yields tokens one by one (or in small chunks) as the model generates them, allowing you to display output progressively. `generateText` waits for the complete response before returning. The total generation time, cost, and model are the same — only the delivery mechanism differs.

---

### Question 3 (Medium)

What is the purpose of `AbortController` when streaming?

A) To restart the stream from the beginning
B) To cancel the stream and free resources, stopping token generation
C) To pause the stream temporarily
D) To change the model mid-stream

**Answer: B**

`AbortController` sends a cancellation signal that stops the stream, closes the HTTP connection, and tells the provider to stop generating tokens. This is important for freeing resources and avoiding charges for tokens you do not need. Simply stopping iteration without aborting leaves the connection open.

---

### Question 4 (Medium)

What does `streamText` with `Output.object()` provide that `generateText` with `Output.object()` does not?

A) Better JSON validation
B) Partial objects that update progressively as fields are populated
C) Support for more complex schemas
D) Automatic error correction

**Answer: B**

`streamText` with `Output.object()` yields partial objects through its `partialOutputStream`. As the model generates the structured output, you receive intermediate versions with some fields populated and others still missing. This enables progressive UI updates where users see information appearing field by field. `generateText` with `Output.object()` waits for the complete, validated object.

---

### Question 5 (Hard)

What protocol does Server-Sent Events (SSE) use for streaming?

A) WebSocket (bidirectional TCP)
B) Standard HTTP (unidirectional server-to-client)
C) gRPC (HTTP/2 streams)
D) MQTT (pub/sub messaging)

**Answer: B**

SSE uses standard HTTP with a `Content-Type: text/event-stream` header. Data flows in one direction: server to client. Each event is a text line prefixed with `data: `. SSE is simpler than WebSockets (no upgrade handshake, automatic reconnection, works through proxies) and is ideal for streaming LLM responses where the data flow is inherently unidirectional.

---

## Exercises

### Exercise 1: Streaming Chat Endpoint with SSE

Build a complete streaming chat server with the following features:

**Requirements:**

1. HTTP server with a `POST /api/chat` endpoint that accepts `{ messages }` and returns an SSE stream
2. Each SSE event should include: `{ type: 'text' | 'usage' | 'done' | 'error', content: string, timestamp: number }`
3. Implement connection timeout (30 seconds max generation time)
4. Implement graceful cancellation when the client disconnects
5. Track and report token usage in the final SSE event
6. Serve a simple HTML page at `/` that demonstrates the streaming interface
7. Support multiple concurrent conversations (use a conversation ID)

**Starter code:**

```typescript
import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface ChatRequest {
  conversationId: string
  messages: Array<{ role: 'user' | 'assistant'; content: string }>
}

interface SSEEvent {
  type: 'text' | 'usage' | 'done' | 'error'
  content: string
  timestamp: number
}

// TODO: Implement the server
// TODO: Handle concurrent conversations
// TODO: Implement timeout and cancellation
// TODO: Track token usage
// TODO: Build the HTML frontend
```

### Exercise 2: Partial Structured Output

Build a streaming data extraction pipeline that shows results progressively.

**Requirements:**

1. Define a schema for extracting information from a text (entities, dates, amounts, relationships)
2. Use `streamText` with `Output.object()` to extract the information
3. Display a live dashboard that shows each field as it populates
4. Show a progress indicator (how many fields are populated vs total)
5. Handle the case where the model fails to populate some fields
6. After completion, validate the extracted data and highlight any missing or suspicious values

**Starter code:**

```typescript
import { streamText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const ExtractionSchema = z.object({
  title: z.string(),
  date: z.string().optional(),
  entities: z.array(
    z.object({
      name: z.string(),
      type: z.enum(['person', 'org', 'location', 'product']),
      mentions: z.number(),
    })
  ),
  amounts: z.array(
    z.object({
      value: z.number(),
      currency: z.string(),
      context: z.string(),
    })
  ),
  sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']),
  summary: z.string(),
  confidence: z.number().min(0).max(1),
})

// TODO: Implement progressive extraction
// TODO: Build the live dashboard display
// TODO: Validate and report completeness
```

**Evaluation criteria:**

- Dashboard updates in real-time as fields populate
- Progress indicator accurately reflects completion percentage
- Missing fields are clearly flagged after completion
- The pipeline handles both short and long input texts gracefully
