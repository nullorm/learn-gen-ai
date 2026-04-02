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

async function measureNonStreaming(): Promise<void>
async function measureStreaming(): Promise<void>
```

Build two functions that demonstrate the TTFT difference. `measureNonStreaming` uses `generateText` — record `performance.now()` before the call, then log the elapsed time after the response arrives. The user sees nothing until the entire response is ready.

`measureStreaming` uses `streamText` and iterates over `result.textStream` with `for await`. Record when the first chunk arrives (the TTFT) and when iteration completes (total time). How do you track first-token time? Set a variable to `null` initially, and only set it on the first iteration. What do you expect the total time to be relative to the non-streaming version?

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

After streaming completes, you can access the full response and metadata through promise properties on the result object:

```typescript
const fullText = await result.text
const usage = await result.usage
const finishReason = await result.finishReason
```

Build a function that first iterates over `result.textStream` (writing each chunk to stdout), then awaits these three properties and logs the full text length, finish reason, and token usage. These properties are promises that resolve once the stream is fully consumed.

### Streaming with Messages (Multi-turn)

```typescript
interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

async function streamConversation(messages: Message[]): Promise<string>
```

Build a function that takes a messages array, passes it to `streamText`, iterates over the text stream (writing to stdout and accumulating the full text), and returns the complete text. Then use it in a multi-turn pattern: create a conversation array, call `streamConversation`, push the assistant response back into the array, add a new user message, and call again. How does this differ from the `generateText` conversation pattern in Module 4?

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
interface StreamMetrics {
  ttft: number // Time to first token (ms)
  totalTime: number // Total generation time (ms)
  tokenCount: number // Number of text chunks received
  tokensPerSecond: number // Generation speed
  chunks: Array<{ text: string; timestamp: number }>
}

async function measureStream(prompt: string): Promise<StreamMetrics>
```

Build a function that measures detailed stream timing. Record `performance.now()` at the start. For each chunk, record a timestamp relative to start. Set `firstTokenTime` on the first chunk. After iteration, compute `tokensPerSecond` as `(chunks.length / totalTime) * 1000`. Return all metrics. Test with a prompt that generates a moderate-length response and log the TTFT, total time, chunk count, and speed.

> **Advanced Note:** Token generation speed varies by model, prompt complexity, and server load. Claude models typically generate 50-100 tokens per second. The first few tokens may be slower due to prefill computation (processing your input). After prefill, generation speed is relatively constant.

---

## Section 3: Streaming Structured Output

### streamText with Output.object()

`streamText` with `Output.object()` lets you stream structured data as it is generated. Instead of waiting for the complete JSON object, you get partial updates as each field is populated.

The key API pattern: pass `output: Output.object({ schema })` to `streamText`, then iterate over `result.partialOutputStream` for progressive updates and `await result.output` for the final validated object.

```typescript
import { streamText, Output } from 'ai'
import { z } from 'zod'

const RecipeSchema = z.object({
  name: z.string().describe('Name of the recipe'),
  prepTime: z.string().describe('Preparation time'),
  cookTime: z.string().describe('Cooking time'),
  servings: z.number().describe('Number of servings'),
  ingredients: z.array(z.object({ item: z.string(), amount: z.string() })),
  steps: z.array(z.string()),
  tips: z.array(z.string()).optional(),
})

type Recipe = z.infer<typeof RecipeSchema>

async function streamRecipe(dish: string): Promise<Recipe>
```

Build a function that streams a structured recipe. Call `streamText` with `output: Output.object({ schema: RecipeSchema })`. Iterate over `result.partialOutputStream` — each iteration yields a partial object where some fields may be populated and others still undefined. Display a live status showing which fields have arrived (name, prep time, ingredient count, step count). After iteration, `await result.output` gives the final complete, validated object. How should you handle fields that are still undefined in the partial object?

> **Beginner Note:** The partial object may have missing or incomplete fields. A string field might contain only the first few words. An array might have only the first item. Your UI should handle these partial states gracefully.

### Progressive UI Updates

```typescript
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

async function streamAnalysis(text: string): Promise<void>
```

Build a function that streams a text analysis using `Output.object()`. For each partial object from `partialOutputStream`, check which fields are populated and build a status line showing: sentiment, topic count, entity count, summary preview (first 80 chars), and confidence percentage. Only include fields that are non-null/non-undefined. In a real UI, these updates would drive DOM changes or state updates.

### Handling Partial Objects Safely

```typescript
function safeDisplay<T extends Record<string, unknown>>(
  partial: Partial<T>,
  fields: Array<keyof T>
): Record<string, string>
```

Build a generic helper for safely displaying partial objects. Iterate over the requested fields. For each field, check if the value is defined: if it is an array, show the count; if it is an object, stringify it; if it is a primitive, convert to string. If undefined or null, show `'(loading...)'`. This pattern is essential for any UI that displays streaming structured data, since required schema fields may still be undefined in partial objects.

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
async function handleChatStream(request: Request): Promise<Response>
```

Build an SSE streaming endpoint using Bun's HTTP server (`Bun.serve`). The server should handle `POST /api/chat` and serve a simple HTML page at `/`.

The `handleChatStream` function should:

1. Parse the request body to extract `messages`
2. Call `streamText` with those messages
3. Create a `new ReadableStream` with a `start(controller)` method
4. Inside start, iterate over `result.textStream` and for each chunk, encode an SSE event: `data: ${JSON.stringify({ type: 'text', content: chunk })}\n\n`
5. After iteration, send a final `done` event with usage stats
6. Wrap error handling in a try/catch that sends an `error` event
7. Return a `new Response(stream)` with headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: keep-alive`

For the HTML client, use `fetch` with `response.body.getReader()` to consume the SSE stream. Read chunks in a loop, decode them, split on newlines, and parse lines starting with `data: ` as JSON events. How do you detect the end of the stream on the client side?

### Using the AI SDK's Built-in Stream Helpers

The Vercel AI SDK provides utilities for converting streams to standard web responses:

As a shortcut, the AI SDK can convert a stream directly to a web Response:

```typescript
return result.toUIMessageStreamResponse()
```

This returns a response in the AI SDK's own streaming protocol, compatible with its React hooks (`useChat`, `useCompletion`). Use the manual SSE approach when building custom frontends.

> **Beginner Note:** The `toUIMessageStreamResponse()` method creates a response in the Vercel AI SDK's own streaming protocol, which works seamlessly with its React hooks (`useChat`, `useCompletion`). If you are building a custom frontend, the manual SSE approach gives you full control.

> **Advanced Note:** The AI SDK UI message stream protocol encodes multiple data types (text deltas, tool calls, tool results, annotations) in a single stream. This is more capable than plain text SSE but requires the AI SDK client library to parse.

---

## Section 5: Backpressure and Flow Control

### What is Backpressure?

Backpressure occurs when the consumer (your code) processes data slower than the producer (the model) generates it. In streaming scenarios:

- The model generates tokens at ~50-100 tokens/second
- Your code might need to do work per token (database write, transformation, network send)
- If your work takes longer than the token generation interval, you need flow control

### Handling Backpressure in Node.js/Bun Streams

```typescript
async function streamToFile(prompt: string, outputPath: string): Promise<void>
```

Build a function that streams LLM output directly to a file. Use `Bun.file(outputPath).writer()` to get a writer, then iterate over `result.textStream`. Write each chunk to the writer and periodically flush (e.g., every 50 chunks with `await writer.flush()`). Call `await writer.end()` after the loop. The `for await...of` loop naturally handles backpressure: it will not request the next chunk until the current iteration completes. Why is periodic flushing important rather than flushing every chunk?

### Rate-Limited Streaming

Control the rate at which tokens are forwarded to the client:

```typescript
async function* rateLimit(stream: AsyncIterable<string>, minIntervalMs: number): AsyncGenerator<string>
```

Build an async generator that wraps a stream and enforces a minimum interval between emitted chunks. Track the last emit time. For each chunk, if less than `minIntervalMs` has elapsed since the last emit, `await` a `setTimeout` for the remaining time. Then `yield` the chunk and update the timestamp. This creates a smooth, consistent typewriter effect. Test with a 30ms interval.

### Buffered Streaming

Accumulate tokens into larger chunks before processing:

```typescript
async function* bufferBySentence(stream: AsyncIterable<string>): AsyncGenerator<string>
```

Build an async generator that accumulates stream chunks into a buffer and yields complete sentences. Use a regex like `/[.!?]\s/` to detect sentence boundaries. When a match is found, yield everything up to and including the boundary, and keep the remainder in the buffer. After the stream ends, flush any remaining buffer content. This is useful when downstream processing (e.g., translation, text-to-speech) works better with complete sentences than individual tokens.

> **Advanced Note:** The `for await...of` loop over an async iterable provides natural backpressure. The producer (AI SDK stream) only generates the next value when the consumer is ready for it. This is built into the JavaScript async iteration protocol. You do not need explicit backpressure mechanisms unless you are bridging to a different streaming system.

---

## Section 6: Stream Transformation

### Filtering Tokens

Remove or modify tokens as they stream through:

```typescript
async function* stripMarkdown(stream: AsyncIterable<string>): AsyncGenerator<string>
```

Build an async generator that removes markdown formatting from a stream. Accumulate chunks into a buffer and apply regex replacements: remove `**` (bold), single `*` (italic), `#` headers, and triple-backtick code markers. The key challenge: a pattern like `**` might be split across two chunks. To handle this, always keep the last 5 characters in the buffer (they might be a partial pattern) and only emit content before that threshold. Flush the remaining buffer when the stream ends.

### Mapping Tokens

Transform each token in the stream:

```typescript
async function* mapStream(
  stream: AsyncIterable<string>,
  transform: (chunk: string) => string
): AsyncGenerator<string>

async function* annotateStream(
  stream: AsyncIterable<string>
): AsyncGenerator<{ text: string; timestamp: number; index: number }>
```

Build two general-purpose stream transformers. `mapStream` is the simplest: for each chunk, yield the result of applying `transform`. `annotateStream` enriches each chunk with metadata: the original text, a timestamp relative to stream start, and a sequential index. These are composable — you can pipe `annotateStream` into `mapStream` or chain multiple transforms. What other metadata might be useful to annotate in a production system?

### Tee: Split a Stream

Send the same stream to multiple consumers:

```typescript
function teeStream<T>(stream: AsyncIterable<T>): [AsyncGenerator<T>, AsyncGenerator<T>]
```

Build a function that splits a single async iterable into two independent consumers. Both consumers receive all chunks. The implementation needs:

1. Two separate buffers (one per consumer) and a `done` flag
2. An immediately-invoked async function that iterates the source stream, pushing each chunk to both buffers and resolving any waiting consumers
3. A `consumer` async generator that yields from its buffer, or waits (via a stored promise resolver) when the buffer is empty

The tricky part is coordinating the two consumers: each one may read at different speeds. When a consumer's buffer is empty and the stream is not done, it should `await` a promise that gets resolved when the next chunk arrives. Use this to stream to both console and a log file simultaneously.

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
interface StreamResult {
  text: string
  completed: boolean
  error: string | null
  tokensGenerated: number
}

async function safeStream(prompt: string): Promise<StreamResult>
```

Build a function that wraps streaming with comprehensive error handling. Pass `abortSignal: AbortSignal.timeout(30_000)` to `streamText` for a 30-second timeout. Wrap the entire iteration in a try/catch. On success, return `{ completed: true, error: null }`. In the catch block, check `error.name` for `'AbortError'` or `'TimeoutError'` (timeout), check `error.message` for `'rate_limit'` (throttled), and handle unknown errors. Always return whatever text was accumulated before the error — partial results are still useful. What should the caller do with a partial result?

### Retry with Partial Recovery

```typescript
async function resilientStream(prompt: string, maxRetries: number = 3): Promise<string>
```

Build a function that retries on failure with partial recovery. Track `fullText` across retries. On the first attempt, use the original prompt. On retries, append the partial output to the prompt with a continuation instruction so the model picks up where it left off. Use exponential backoff between retries (`1000 * Math.pow(2, retries)` ms). After `maxRetries`, return whatever text was accumulated. Why is exponential backoff important for rate limit errors?

### Cancellation

```typescript
async function streamWithCancellation(prompt: string, maxChars: number = 1000): Promise<string>
```

Build a function that cancels a stream based on content conditions. Create an `AbortController` and pass its `signal` to `streamText`. During iteration, check two conditions: (1) accumulated text exceeds `maxChars`, (2) text contains unwanted patterns like `[CONFIDENTIAL]`. When either triggers, call `abortController.abort()` and break. Wrap the iteration in a try/catch that silently handles `AbortError` (expected when we abort) but rethrows other errors. Why is calling `abort()` important rather than just breaking out of the loop?

> **Beginner Note:** Always use `AbortController` for cancellation rather than simply stopping iteration. Stopping iteration without aborting leaves the underlying HTTP connection open, wasting server resources and potentially accruing costs for tokens you are not using.

---

## Section 8: UI Patterns

### Typewriter Effect

The classic pattern where text appears character by character:

```typescript
async function typewriter(stream: AsyncIterable<string>, charDelayMs: number = 20): Promise<string>
```

Build a function that creates a typewriter effect. Each stream chunk might contain multiple characters, so iterate over each character individually. For each character, write it to stdout, accumulate it into `fullText`, and `await` a `setTimeout` of `charDelayMs`. Return the full text when done. Test with a 15-20ms delay for a smooth visual effect. How does this differ from the `rateLimit` generator in Section 5?

### Progressive Disclosure

Show content section by section as it is generated:

```typescript
async function progressiveDisclosure(stream: AsyncIterable<string>): Promise<string[]>
```

Build a function that groups streamed content into sections and reveals each one with a visual separator. Accumulate chunks into a buffer. Split the buffer on `\n\n` (double newline). When the split produces more than one part, all parts except the last are complete sections — display each with a header and separator. Keep the last part (potentially incomplete) in the buffer. After the stream ends, flush the remaining buffer as the final section. Return an array of all sections.

### Streaming Status Indicators

Show progress information alongside the streaming content:

```typescript
interface StreamStatus {
  state: 'connecting' | 'streaming' | 'complete' | 'error'
  charsReceived: number
  elapsed: number
  speed: number // chars per second
}

async function streamWithStatus(prompt: string, onStatus: (status: StreamStatus) => void): Promise<string>
```

Build a function that reports progress through a callback. Call `onStatus` with `'connecting'` before starting, `'streaming'` on each chunk (with chars received, elapsed time, and chars/sec), `'complete'` after iteration, and `'error'` if an exception occurs. Compute speed as `(text.length / elapsed) * 1000`. The caller can use `process.stderr.write('\r' + statusLine)` to display a self-updating status bar on a separate stream from the content output.

> **Advanced Note:** In browser-based UIs, consider implementing a "thinking" indicator that appears during the TTFT gap (before the first token arrives). This gives users immediate feedback that their request is being processed. Combine this with streaming text display for the best perceived performance.

> **Local Alternative (Ollama):** All streaming patterns in this module (`streamText`, `streamText` with `Output.object()`, backpressure, AbortController) work identically with `ollama('qwen3.5')`. Local models actually benefit more from streaming since inference is slower — streaming lets users see output immediately rather than waiting for full generation. SSE endpoints work the same regardless of provider.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: NDJSON Streaming

### Newline-Delimited JSON for Programmatic Consumers

SSE is designed for browser clients, but many LLM applications are consumed by other programs — CLIs, SDKs, CI pipelines, or backend services. For these consumers, NDJSON (newline-delimited JSON) is simpler and more natural. Each line is a complete, self-contained JSON object:

```
{"type":"text_delta","content":"Hello"}
{"type":"text_delta","content":" world"}
{"type":"tool_call","name":"search","args":{"query":"TypeScript"}}
{"type":"tool_result","name":"search","content":"..."}
{"type":"done","usage":{"input":150,"output":42}}
```

The consumer reads line by line, parses each line as JSON, and processes events incrementally — no buffering the entire response, no SSE protocol overhead, no event type headers. This is the format production CLIs and SDK modes use for structured streaming output.

### NDJSON vs SSE

| Feature      | SSE                           | NDJSON                           |
| ------------ | ----------------------------- | -------------------------------- |
| Protocol     | HTTP with `text/event-stream` | HTTP with `application/x-ndjson` |
| Client       | Browser `EventSource`         | Line-by-line reader              |
| Reconnection | Built-in                      | Manual                           |
| Best for     | Web UIs                       | CLIs, SDKs, pipelines            |

Both formats carry the same typed events — the difference is the transport encoding. A well-designed streaming system can emit either format based on the client's `Accept` header.

---

## Section 10: Abort Controller Trees

### Hierarchical Cancellation

The AbortController from Section 7 handles single-operation cancellation. Production streaming systems need hierarchical cancellation — aborting a parent operation should abort all its children (streaming, tool execution, sub-queries), and aborting a child should not affect the parent.

The pattern uses a parent-child relationship between AbortControllers:

```typescript
const parentController = new AbortController()
const childController = new AbortController()

// When parent aborts, abort the child
parentController.signal.addEventListener('abort', () => childController.abort())
```

In a streaming pipeline with tool calls, the hierarchy looks like:

1. **Session controller** — user presses Ctrl+C or cancel
2. **Request controller** — child of session, scoped to one `streamText` call
3. **Tool controller** — child of request, scoped to one tool execution

When the user cancels, the session controller aborts, which cascades to the request controller (stopping the stream) and the tool controller (stopping any in-progress tool execution). Each level cleans up its own resources.

### Why Not a Single Controller?

A single global AbortController means any cancellation aborts everything. With a hierarchy, you can cancel one tool execution without aborting the entire stream, or cancel the current request without ending the session.

---

## Section 11: Enhanced Backpressure with Buffered Writer

### Buffering Output Writes

Section 5 covered backpressure at the token level. A buffered writer is the concrete implementation: instead of writing every token immediately (which can block on I/O), batch small writes into larger chunks and flush them on a schedule or when the buffer is full.

```typescript
// Instead of writing each token individually:
for await (const chunk of result.textStream) {
  process.stdout.write(chunk) // blocks on I/O for each token
}

// Buffer and flush in batches:
let buffer = ''
for await (const chunk of result.textStream) {
  buffer += chunk
  if (buffer.length >= 256) {
    process.stdout.write(buffer)
    buffer = ''
  }
}
if (buffer.length > 0) process.stdout.write(buffer) // flush remainder
```

This prevents I/O from blocking the LLM processing loop. In terminal UIs, the batch size controls the visual "chunkiness" of output — smaller batches look smoother, larger batches are more efficient. Production systems tune this based on the output target (terminal, file, network socket).

---

## Section 12: Headless/CI Streaming Mode

### Non-Interactive Execution

Production streaming applications support a headless mode for CI/CD pipelines and programmatic use. Instead of an interactive terminal UI, the application accepts a prompt via command-line arguments or stdin, emits structured output (NDJSON or simplified text), and exits with a status code.

The key differences from interactive mode:

- **Input:** Prompt from CLI args or stdin pipe, not a REPL
- **Output:** NDJSON events or plain text, not cursor-controlled terminal UI
- **TTY detection:** Check `process.stdout.isTTY` to auto-switch between interactive and headless
- **Exit codes:** `0` for success, non-zero for errors — enables CI integration
- **No user prompts:** Actions that would require confirmation in interactive mode either auto-approve or fail, controlled by a policy flag

```typescript
const isHeadless = !process.stdout.isTTY || process.argv.includes('--headless')

if (isHeadless) {
  // Emit NDJSON events
  for await (const chunk of result.textStream) {
    console.log(JSON.stringify({ type: 'text_delta', content: chunk }))
  }
} else {
  // Interactive terminal display
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk)
  }
}
```

This pattern lets the same streaming application serve both human users and automated pipelines.

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
8. **NDJSON streaming:** A simpler alternative to SSE for programmatic consumers — each event is a complete JSON object on one line, ideal for CLIs and pipelines.
9. **Abort controller trees:** Hierarchical cancellation that propagates abort signals from parent to child operations, enabling granular cancellation in complex streaming pipelines.
10. **Buffered writers:** Batching streaming output writes for efficiency without blocking the LLM processing loop.
11. **Headless mode:** Non-interactive execution with structured output for CI/CD integration and programmatic use.

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

### Question 6 (Medium)

What is the primary advantage of NDJSON over SSE for streaming LLM output?

- A) NDJSON supports bidirectional communication
- B) NDJSON is simpler for programmatic consumers (CLIs, SDKs, pipelines) — each line is a self-contained JSON object with no protocol overhead
- C) NDJSON is faster because it uses binary encoding
- D) NDJSON has built-in automatic reconnection

**Answer: B** — NDJSON (newline-delimited JSON) is designed for programmatic consumers. Each line is a complete, parseable JSON object — no SSE protocol headers, no event type prefixes, no buffering the entire response. This makes it ideal for CLIs, SDKs, and CI pipelines where a line-by-line JSON reader is simpler than an SSE client.

---

### Question 7 (Hard)

In an abort controller tree with session, request, and tool controllers, what happens when the request controller is aborted?

- A) The session controller is also aborted, ending the entire session
- B) Only the current stream and its child tool executions are cancelled; the session controller remains active for future requests
- C) All controllers in the tree are aborted simultaneously
- D) The tool controller continues running to completion before the request is cancelled

**Answer: B** — In a hierarchical abort controller tree, aborting a parent cascades to its children, but not upward. Aborting the request controller stops the current `streamText` call and any in-progress tool executions (child controllers), but the session controller remains active. This allows the user to cancel one request without ending the session, and cancel a tool without aborting the stream.

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

### Exercise 3: Cancellation with Abort Controllers

Build a streaming pipeline with hierarchical cancellation that properly cleans up all in-progress operations.

**What to build:** Create `src/streaming/exercises/cancellation.ts`

**Requirements:**

1. Create a `CancellableStream` class that wraps a `streamText` call with a parent AbortController
2. Support spawning child operations (simulated tool calls) that each get their own child AbortController linked to the parent
3. When the parent is aborted, all children should be aborted and resources cleaned up
4. When a child is aborted, the parent and other children should continue unaffected
5. Track the state of each operation: `'running' | 'completed' | 'aborted'`
6. Expose a `cancel()` method on the parent and individual `cancelChild(id)` on children

**Expected behavior:**

- Starting a stream with 3 child tool operations and calling `cancel()` should abort all 4 operations and set all states to `'aborted'`
- Calling `cancelChild('tool-2')` should abort only that child; the stream and other children remain `'running'`
- After all operations complete or are cancelled, no abort listeners remain active (no memory leaks)

**File:** `src/streaming/exercises/cancellation.ts`

### Exercise 4: Streaming Tool Calls

Build a streaming handler that processes tool call parameters as they arrive, beginning preparation before the complete tool call is available.

**What to build:** Create `src/streaming/exercises/streaming-tools.ts`

**Requirements:**

1. Use `streamText` with tools defined (reuse tools from Module 7 or define simple ones)
2. Listen for `toolCallStreaming` events that provide partial tool call arguments as they stream in
3. Implement a `ToolPreparer` that begins setup work (e.g., validating a file path, resolving a URL) as soon as enough of the arguments are available — before the full tool call arrives
4. When the complete tool call arrives (`toolCall` event), execute using the pre-prepared state
5. Log the timeline: when streaming started, when preparation began, when the full call arrived, when execution completed
6. Measure the time saved by early preparation vs waiting for the complete call

**Expected behavior:**

- For a file-reading tool, preparation begins as soon as the `path` argument is partially streamed (e.g., path validation starts)
- The timeline log shows preparation starting before the tool call is fully specified
- Total latency is reduced compared to a baseline that waits for the complete call before doing any work

**File:** `src/streaming/exercises/streaming-tools.ts`
