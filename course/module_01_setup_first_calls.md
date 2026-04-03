# Module 1: Setup & First LLM Calls

## Learning Objectives

- Set up a TypeScript project with Bun and the Vercel AI SDK for LLM development
- Configure multiple LLM providers (Mistral, Groq, Anthropic, OpenAI, Ollama) and understand the provider abstraction
- Make synchronous and streaming LLM calls using `generateText` and `streamText`
- Understand message roles (system, user, assistant) and how they shape model behavior
- Handle errors, configure model parameters, and build resilient LLM applications

---

## Why Should I Care?

Every LLM-powered application — from a chatbot to an autonomous agent, from a code reviewer to a data pipeline — begins with a single API call. Before you can build retrieval-augmented generation systems, multi-step agents, or evaluation pipelines, you need to know how to reliably call a model, stream its response, and handle the inevitable failures. This module is where that journey starts.

The Vercel AI SDK provides a unified interface across dozens of model providers. Instead of learning the Mistral API, the Groq API, the Anthropic API, the OpenAI API, and the Ollama API separately, you learn one set of functions — `generateText`, `streamText`, `Output` — and swap providers with a single line of code. This abstraction is not just convenient; it is strategically important. Models improve rapidly, new providers emerge, and pricing changes constantly. Building on an abstraction layer means your application code stays stable while the AI landscape shifts underneath.

If you skip this foundation or cut corners here, every subsequent module will be harder. You will fight configuration issues when you should be learning prompt engineering. You will debug timeout errors when you should be building agents. Take the time now to build a solid project setup, understand the call lifecycle, and internalize the error handling patterns. Your future self will thank you.

---

## Connection to Other Modules

This module is the bedrock of the entire course. Every subsequent module assumes you have a working project with at least one configured provider and the ability to make `generateText` calls.

- **Module 2 (Prompt Engineering)** builds directly on the message roles and `generateText` calls introduced here.
- **Module 3 (Structured Output)** extends `generateText` with `Output.object()` and Zod schemas — you need to be comfortable with the basic call pattern first.
- **Module 4 (Conversational AI)** uses `streamText` extensively for real-time chat interfaces.
- **Module 5+ (Tools, Agents, RAG)** all build on the provider configuration and error handling patterns established in this module.

Think of this module as installing the engine in your car. Nothing else works until the engine runs.

---

## Section 1: Project Setup

### Initializing a Bun Project

We use [Bun](https://bun.sh) as our runtime and package manager. Bun is fast, has native TypeScript support, and simplifies the development experience compared to Node.js with a separate TypeScript compilation step.

> **Beginner Note:** If you have never used Bun before, install it with `curl -fsSL https://bun.sh/install | bash` on macOS/Linux or visit bun.sh for Windows instructions. Bun replaces both Node.js and npm/yarn/pnpm for our purposes.

Create your project directory and initialize it:

```bash
mkdir llm-engineering && cd llm-engineering
bun init -y
```

This creates a minimal project with `package.json`, `tsconfig.json`, and an `index.ts` entry point.

### Installing Dependencies

The Vercel AI SDK is split into a core package (`ai`) and provider-specific packages. Install the core SDK and the Mistral provider (our default):

```bash
bun add ai @ai-sdk/mistral
```

For additional providers (recommended for experimentation):

```bash
bun add @ai-sdk/groq @ai-sdk/anthropic @ai-sdk/openai ai-sdk-ollama
```

Your `package.json` should now look similar to this:

```json
{
  "name": "llm-engineering",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "bun run src/index.ts",
    "test": "bun test"
  },
  "dependencies": {
    "ai": "^6.0.0",
    "@ai-sdk/mistral": "^3.0.0",
    "@ai-sdk/groq": "^3.0.0",
    "@ai-sdk/anthropic": "^3.0.0",
    "@ai-sdk/openai": "^3.0.0",
    "ai-sdk-ollama": "^3.8.0"
  },
  "devDependencies": {
    "@types/bun": "latest"
  }
}
```

### TypeScript Configuration

Bun handles TypeScript natively, but we want a strict configuration for LLM development. Strict types catch entire categories of bugs when working with model responses, schemas, and prompt templates.

```json
{
  "compilerOptions": {
    "target": "ESNext",
    "module": "nodenext",
    "moduleResolution": "nodenext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInImports": true,
    "resolveJsonModule": true,
    "declaration": false,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "types": ["bun-types"]
  },
  "include": ["src/**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
```

> **Advanced Note:** The `"moduleResolution": "nodenext"` setting enforces strict ESM — TypeScript will error if you forget `.js` extensions on relative imports. This is stricter than `"bundler"` but catches issues early.

> **Important: ESM Import Extensions** — Always use `.js` extensions in relative imports, even though the source files are `.ts`. This is the correct ESM convention: `import { createModel } from './provider.js'`, not `from './provider'`. TypeScript resolves `.js` to the corresponding `.ts` file at compile time. Omitting extensions works in some bundlers but breaks in strict ESM environments.

### Project Structure

Create the following directory layout:

```bash
mkdir -p src/providers src/examples src/utils
touch src/index.ts src/providers/mistral.ts src/providers/groq.ts src/providers/anthropic.ts src/providers/openai.ts src/utils/env.ts
```

```
llm-engineering/
  src/
    index.ts              # Entry point
    providers/
      mistral.ts          # Mistral provider config (default)
      groq.ts             # Groq provider config
      anthropic.ts        # Anthropic provider config
      openai.ts           # OpenAI provider config
      ollama.ts           # Ollama provider config (local)
    examples/
      first-call.ts       # Section 3 examples
      roles.ts            # Section 4 examples
      streaming.ts        # Section 5 examples
      parameters.ts       # Section 6 examples
      error-handling.ts   # Section 7 examples
    utils/
      env.ts              # Environment variable helpers
  .env                    # API keys (never commit this)
  .env.example            # Template for required env vars
  package.json
  tsconfig.json
```

### Environment Variables

Create a `.env.example` file documenting the required keys:

```bash
# .env.example — copy to .env and fill in your keys
MISTRAL_API_KEY=...           # Free tier: 1 RPS, 500K tok/min, 1B tok/month — https://console.mistral.ai
GROQ_API_KEY=gsk_...          # Free tier: ~500K tokens/day — https://console.groq.com
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
# Ollama runs locally, no key needed
```

Create your actual `.env` file (this should be in `.gitignore`):

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Now create a utility to validate that keys are present at startup. Create a file `src/utils/env.ts` that exports these three functions:

```typescript
// src/utils/env.ts

export function requireEnv(name: string): string {
  /* ... */
}

export function optionalEnv(name: string, fallback: string = ''): string {
  /* ... */
}

export function validateEnv(): void {
  /* ... */
}
```

**What each function should do:**

- `requireEnv` — Read `process.env[name]`. If the value is missing or empty, throw an `Error` with a message that includes the variable name and a hint to copy `.env.example`. If present, return the value.
- `optionalEnv` — Return the environment variable value if it exists, otherwise return the `fallback`. This is a one-liner using `??`.
- `validateEnv` — Define an array of required variable names (at minimum `'MISTRAL_API_KEY'`). Filter to find which are missing from `process.env`. If any are missing, log each one and exit the process. If all are present, log a success message.

**Guiding questions:** What should `requireEnv` return if the variable exists? What operator lets you provide a default value for `undefined`? How do you exit a Node/Bun process with a non-zero status code?

> **Beginner Note:** Never hardcode API keys in your source files. Even in tutorials and experiments, use environment variables. It builds good habits and prevents accidental key exposure if you push code to GitHub.

### Verifying the Setup

Create a simple smoke test to confirm everything works. Create `src/index.ts` that:

1. Imports `validateEnv` from your `./utils/env.js` and calls it first
2. Imports `generateText` from `'ai'` and `mistral` from `'@ai-sdk/mistral'`
3. Defines an async `smokeTest` function that calls `generateText` with a simple prompt like `"Say 'hello' and nothing else."`
4. Logs the response text and usage information
5. Calls `smokeTest().catch(console.error)` at the bottom

The key API pattern is:

```typescript
const result = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Your prompt here',
})
console.log(result.text) // The generated text
console.log(result.usage) // { inputTokens, outputTokens, totalTokens }
```

Run it:

```bash
bun run src/index.ts
```

If you see `"hello"` (or some variation) printed to the console, your setup is complete. If you see an error, check the troubleshooting section at the end of this module.

---

## Section 2: Your First Provider

### What is a Provider?

In the Vercel AI SDK, a **provider** is an adapter that translates the SDK's unified interface into the specific API format expected by a model service. When you call `generateText`, the SDK does not talk directly to Anthropic or OpenAI — it delegates to a provider that handles authentication, request formatting, response parsing, and error mapping.

This abstraction has profound implications:

1. **Portability:** Switch from Claude to GPT-4 by changing one line of code.
2. **Testing:** Use a cheap/fast model for development, a powerful model for production.
3. **Resilience:** Fall back to a different provider if one is down.
4. **Cost management:** Route simple tasks to cheaper models, complex tasks to expensive ones.

### Setting Up the Mistral Provider

Mistral is our default provider throughout this course. Their generous free tier (1 billion tokens per month, all models, no credit card required) makes it the best starting point. The `@ai-sdk/mistral` package handles all the details.

Create `src/providers/mistral.ts`. It should import `mistral` from `'@ai-sdk/mistral'` and export:

- A pre-configured model constant `mistralSmall` using `mistral('mistral-small-latest')`
- A factory function `createMistralModel(modelId?: string)` that defaults to `'mistral-small-latest'`

The provider reads `MISTRAL_API_KEY` from the environment automatically. Free tier: 1 RPS, 500K tokens/min, 1B tokens/month per model, no credit card required. Sign up at https://console.mistral.ai.

Common model IDs: `"mistral-small-latest"` (fast, default), `"mistral-large-latest"` (most capable), `"codestral-latest"` (code-optimized), `"pixtral-12b-2409"` (multimodal/vision).

The pattern for creating a model instance is simply:

```typescript
import { mistral } from '@ai-sdk/mistral'
const model = mistral('mistral-small-latest')
```

> **Beginner Note:** The `mistral()` function returns a model object, not a response. Think of it as configuring _which_ model you want to talk to. The actual API call happens when you pass this model to `generateText` or `streamText`.

### Setting Up the Groq Provider

Groq provides ultra-fast inference on open-source models with a free tier (~500K tokens per day, no credit card required). It is an excellent alternative when you need speed.

Create `src/providers/groq.ts` following the same pattern. Import `groq` from `'@ai-sdk/groq'` and export a default constant and a factory function.

Groq reads `GROQ_API_KEY` from the environment automatically. Free tier: ~500K tokens/day, no credit card required. Sign up at https://console.groq.com.

Common model IDs: `"openai/gpt-oss-20b"` (fastest, cheapest), `"openai/gpt-oss-120b"` (code execution, reasoning), `"llama-3.3-70b-versatile"` (proven, larger).

### Setting Up the Anthropic Provider

Anthropic's Claude models are a premium option — highly capable but require a paid API key. Use Claude when you need top-tier reasoning and instruction following.

Create `src/providers/anthropic.ts` following the same pattern. Import `anthropic` from `'@ai-sdk/anthropic'`. Export constants for commonly used models and a factory function.

Anthropic reads `ANTHROPIC_API_KEY` from the environment automatically. Common model IDs: `"claude-sonnet-4-20250514"` (best speed/quality balance), `"claude-opus-4-20250514"` (most capable), `"claude-haiku-4-5-20251001"` (fastest, cheapest).

### Setting Up the OpenAI Provider

If you want to experiment with GPT models as an alternative:

Create `src/providers/openai.ts` following the same pattern. Import `openai` from `'@ai-sdk/openai'`.

OpenAI reads `OPENAI_API_KEY` from the environment automatically. Common model IDs: `"gpt-5.4"` (latest, fast and capable), `"gpt-5-mini"` (smaller, cheaper, faster).

### Setting Up Ollama (Local Models)

Ollama lets you run open-source models locally — no API key needed, no usage costs, and full data privacy. This is ideal for offline development and experimentation.

Create `src/providers/ollama.ts`. Import `ollama` from `'ai-sdk-ollama'` (note: not `@ai-sdk/ollama`). Export a default constant and a factory function.

Prerequisites: Install Ollama from https://ollama.com, pull a model with `ollama pull qwen3.5`, and the server runs on `http://localhost:11434` by default.

Recommended local models: `"qwen3.5"` (primary choice — best all-rounder), `"ministral-3"` (lightweight alternative). Cloud variants: `"qwen3.5:cloud"`, `"ministral-3:cloud"`, `"mistral-large-3:cloud"` (frontier-class).

> **Advanced Note:** Ollama models are significantly less capable than Claude or GPT-4 for complex tasks. Use them for quick iteration and testing, but validate important behavior with a frontier model. The quality gap is especially apparent in structured output (Module 3) and tool use (Module 7).

> **Gotcha: Thinking Mode on Qwen3/3.5** — Qwen3 and Qwen3.5 models default to "thinking mode," where the model spends tokens inside `<think>` tags before responding. The AI SDK strips these tags, so you often get an empty `text` response because all tokens were consumed by thinking. To disable it, pass `{ think: false }` as the second argument to the `ollama()` model constructor: `ollama('qwen3.5', { think: false })`. The `ai-sdk-ollama` provider handles this natively — no custom fetch hacks or `providerOptions` needed.

### A Unified Provider Factory

Here is a useful pattern that lets you select a provider at runtime. Create `src/providers/factory.ts` with these types and exports:

```typescript
// src/providers/factory.ts

import type { LanguageModel } from 'ai'

export type ProviderName = 'mistral' | 'groq' | 'anthropic' | 'openai' | 'ollama'

interface ModelConfig {
  provider: ProviderName
  modelId?: string
}

export function createModel(config: ModelConfig): LanguageModel {
  /* ... */
}
```

**What `createModel` should do:**

1. Import all five provider functions at the top of the file
2. Define a `DEFAULT_MODELS` record mapping each `ProviderName` to its default model ID string
3. Use `config.modelId ?? DEFAULT_MODELS[config.provider]` to resolve the model ID
4. Use a `switch` on `config.provider` to call the correct provider function with the model ID
5. For Ollama, pass `{ think: false }` as the second argument to disable thinking mode
6. Throw a descriptive error in the `default` case for unknown providers

**Guiding questions:** Why is `LanguageModel` the right return type? What would happen if you forgot the `default` case? Why does Ollama need `{ think: false }`?

Usage after you build it:

```typescript
const model = createModel({ provider: 'mistral' })
const model = createModel({ provider: 'anthropic', modelId: 'claude-haiku-4-5-20251001' })
```

This factory pattern becomes increasingly valuable as your application grows. You can read the provider name from a config file, a CLI argument, or an environment variable — keeping your application code provider-agnostic.

> **Scale Note: 75+ Providers via AI SDK** — The Vercel AI SDK supports 75+ providers through its unified interface — the same `generateText` and `streamText` functions you are learning here. Production coding agents use this exact pattern with the `models.dev` registry to support every major provider through a single SDK. This validates the provider-agnostic approach: your code stays the same whether you use Mistral, Anthropic, OpenAI, Google, or any other supported provider. Production systems also configure model **variants** with provider-specific options — for example, thinking budget levels for Anthropic models or reasoning effort levels for OpenAI — passed via `providerOptions` in `generateText`/`streamText` calls.

---

## Section 3: generateText — Your First Call

### The generateText Function

`generateText` is the most fundamental function in the Vercel AI SDK. It sends a prompt to a model and waits for the complete response before returning. This is the "request-response" pattern you already know from HTTP APIs.

```typescript
import { generateText } from 'ai'
```

The function signature accepts a configuration object with many options, but only two are required: `model` and either `prompt` or `messages`.

### Your Absolute First Call

Create `src/examples/first-call.ts`. It should:

1. Import `generateText` from `'ai'` and `mistral` from `'@ai-sdk/mistral'`
2. Define an async function that calls `generateText` with your Mistral model and a simple prompt like `'What is the capital of France?'`
3. Log `result.text` to the console

The core pattern is:

```typescript
const result = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is the capital of France?',
})
console.log(result.text)
```

Run it:

```bash
bun run src/examples/first-call.ts
```

That is it. One import, one function call, one result. Let us now unpack what happened.

### Understanding the Result Object

`generateText` returns a rich result object. Create `src/examples/result-object.ts` that makes a `generateText` call and logs each property of the result. The key properties to explore are:

```typescript
result.text // The generated text string
result.usage // { inputTokens: number, outputTokens: number, totalTokens: number }
result.finishReason // "stop" | "length" | "tool-calls" | "error"
result.response.id // Unique response ID from the provider
result.response.modelId // Which model actually responded
result.warnings // Array of warnings (e.g., unsupported features)
```

The `finishReason` is especially important: `"stop"` means the model completed naturally, while `"length"` means it was cut off by `maxOutputTokens`.

Build this file yourself — call `generateText` with a prompt, then log each of the properties above to see what the result object contains.

> **Beginner Note:** Always check `result.finishReason`. If it is `"length"`, the model was cut off mid-sentence because it hit the token limit. You will need to increase `maxOutputTokens` or shorten your prompt. This is one of the most common issues beginners encounter.

### Using prompt vs messages

There are two ways to specify what you want the model to do:

**Simple `prompt` (shorthand):**

```typescript
const result = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is 2 + 2?',
})
```

This is equivalent to sending a single user message. It is great for simple, one-off questions.

**Explicit `messages` (full control):**

```typescript
const result = await generateText({
  model: mistral('mistral-small-latest'),
  messages: [{ role: 'user', content: 'What is 2 + 2?' }],
})
```

The `messages` format gives you full control over the conversation, including system prompts and multi-turn history. We will use this format extensively from Section 4 onward.

### System Prompts

The `system` parameter sets the model's persona and instructions. The pattern is:

```typescript
const result = await generateText({
  model: mistral('mistral-small-latest'),
  system: 'You are a pirate captain. Respond in pirate speak. Keep responses under 50 words.',
  prompt: 'How do I learn to code?',
})
```

Create `src/examples/system-prompt.ts` — try a few different system prompts (a pirate, a poet, a strict teacher) and see how they change the model's response to the same user prompt. This is your first taste of how the system message shapes behavior.

### Multiple Examples with Different Prompts

Create `src/examples/various-prompts.ts` that demonstrates four different use cases of `generateText`:

1. **Factual question** — a simple prompt with no system message (e.g., "What are the three laws of thermodynamics?")
2. **Creative writing** — a system message setting a creative persona, with a story prompt
3. **Code generation** — a system message setting a developer persona, with a coding task
4. **Summarization** — a system message instructing the model to summarize, with a long passage as the prompt

For each, log the result text and `result.usage.totalTokens` so you can see how token usage varies by task.

**Tip:** Create the model object once and reuse it across calls:

```typescript
const model = mistral('mistral-small-latest')
```

> **Advanced Note:** The model object is stateless — it just holds configuration. Each `generateText` call is independent. There is no connection pooling or session management to worry about.

---

## Section 4: Understanding Roles

### The Three Roles

LLM conversations are structured as sequences of messages, each tagged with a **role**. The three roles are:

| Role        | Purpose                                         | Who writes it?                              |
| ----------- | ----------------------------------------------- | ------------------------------------------- |
| `system`    | Sets behavior, persona, and rules for the model | The developer                               |
| `user`      | Represents input from the end user              | The user (or developer simulating a user)   |
| `assistant` | Represents previous model responses             | The model (or developer providing examples) |

### System Messages: Setting the Stage

The system message is your primary control surface. It tells the model _how to behave_ before the user says anything. Think of it as writing an employee's job description.

Create `src/examples/roles-system.ts` that demonstrates how different system messages change the model's behavior. Use the `messages` array format instead of the `system` + `prompt` shorthand:

```typescript
const result = await generateText({
  model,
  messages: [
    { role: 'system', content: 'Your persona instructions here...' },
    { role: 'user', content: 'The user question here...' },
  ],
})
```

Build two functions in this file:

1. **Technical expert** — system message describes a senior software architect who gives precise answers and mentions trade-offs. Ask it a technical question.
2. **Friendly tutor** — system message describes a patient beginner tutor who uses analogies and avoids jargon. Ask it to explain a basic concept.

Run both and compare how the same type of question gets wildly different answers depending on the system message. This is the power of the system role.

### User Messages: The Input

User messages represent what the end user typed. In a chatbot, these come from actual user input. In a pipeline, you construct them programmatically.

```typescript
// Simple user message
{ role: 'user', content: 'Translate "hello world" to French.' }

// User message with context
{ role: 'user', content: `Given this error log:\n${errorLog}\n\nWhat is the root cause?` }
```

### Assistant Messages: Providing History

Assistant messages represent previous model responses. Including them creates the illusion of a continuous conversation. This is how multi-turn chat works — you replay the full conversation history with each API call.

Create `src/examples/roles-multiturn.ts` that simulates a multi-turn conversation by providing message history. The messages array should alternate between `user` and `assistant` roles, starting with an optional `system` message:

```typescript
messages: [
  { role: 'system', content: '...' },
  { role: 'user', content: '...' },
  { role: 'assistant', content: '...' }, // A previous model response you provide
  { role: 'user', content: '...' },
  { role: 'assistant', content: '...' }, // Another previous response
  { role: 'user', content: '...' }, // The new question — model responds to this
]
```

Build a conversation with at least 3 turns of history (e.g., a cooking assistant, a travel planner, or a debugging helper). The key insight: the model has no memory — you must include the full history each time. The model responds to the **last user message** in the context of everything above it.

> **Beginner Note:** The model has no memory between API calls. Every call is stateless. If you want the model to "remember" previous messages, you must include them in the `messages` array. This is why chatbots grow more expensive over time — each message includes the entire conversation history.

### Message Ordering Rules

The ordering of messages matters and follows strict rules:

1. **System message** (optional): Must be first if present.
2. **Messages must alternate** between user and assistant roles (with some flexibility).
3. **The last message should be from the user** — this is what the model responds to.

```typescript
// CORRECT ordering
const messages = [
  { role: 'system' as const, content: 'You are helpful.' },
  { role: 'user' as const, content: 'Hello' },
  { role: 'assistant' as const, content: 'Hi! How can I help?' },
  { role: 'user' as const, content: 'What is TypeScript?' },
]

// INCORRECT — will cause errors or unexpected behavior
const badMessages = [
  { role: 'user' as const, content: 'Hello' },
  { role: 'user' as const, content: 'Are you there?' }, // Two user messages in a row
  { role: 'system' as const, content: 'Be helpful.' }, // System after user — wrong position
]
```

### Practical Pattern: Building Conversation History

Now build a reusable pattern for managing conversation history incrementally. Create `src/examples/conversation-builder.ts` with a `Conversation` class:

```typescript
import type { ModelMessage } from 'ai'

class Conversation {
  private messages: ModelMessage[] = []

  constructor(systemPrompt?: string) {
    /* ... */
  }
  async say(userMessage: string): Promise<string> {
    /* ... */
  }
  getHistory(): ModelMessage[] {
    /* ... */
  }
  getTokenEstimate(): number {
    /* ... */
  }
}
```

**What each method should do:**

- **constructor** — If a `systemPrompt` is provided, push a system message onto the `messages` array. Also store a model instance.
- **say** — Push the user message, call `generateText` with the full `messages` array, push the assistant response, and return the text. This is the core loop of any chatbot.
- **getHistory** — Return a copy of the messages array (use spread: `[...this.messages]`).
- **getTokenEstimate** — Sum the character lengths of all message contents and divide by ~4 (a rough token estimate).

**Guiding questions:** Why do you push both the user message AND the assistant response onto the array? Why return a copy from `getHistory` instead of the array directly? What happens to the messages array as the conversation grows?

Test it by creating a conversation, calling `say()` a few times, and logging the history and token estimate.

> **Advanced Note:** This simple Conversation class grows unboundedly. In production, you need a strategy for managing context window limits: truncation, summarization, or sliding window. We cover these techniques in Module 4 (Conversations & Memory).

---

## Section 5: streamText — Real-time Responses

### Why Streaming Matters

When you call `generateText`, you wait for the _entire_ response before seeing anything. For a short answer, that is fine. But for a long response — a detailed explanation, a code review, a story — the user stares at a blank screen for several seconds. This feels slow and unresponsive.

**Streaming** solves this by delivering the response token-by-token as the model generates it. The user sees the first word within milliseconds, and the rest flows in smoothly. This is how ChatGPT, Claude.ai, and every modern chat interface works.

The key metric is **Time to First Token (TTFT)**. A 500-token response that takes 3 seconds feels instant when streamed but feels sluggish when buffered. TTFT is typically 200-800ms depending on the model and prompt length.

### The streamText Function

`streamText` has the same interface as `generateText` — same `model`, `messages`, `system`, and configuration options. The difference is in the return value: instead of a completed result, you get a stream.

Create `src/examples/streaming-basic.ts`. The key differences from `generateText`:

1. `streamText` is **not awaited** when called — it returns a stream object immediately
2. You consume the stream with `for await...of` on `result.textStream`
3. After the stream completes, properties like `result.usage` and `result.finishReason` are promises you must `await`

```typescript
const result = streamText({
  // No await here!
  model: mistral('mistral-small-latest'),
  prompt: 'Explain how a CPU works.',
})

for await (const textPart of result.textStream) {
  process.stdout.write(textPart) // Print each chunk as it arrives
}

const usage = await result.usage // Available after stream completes
```

Build this file — call `streamText`, consume the text stream with `process.stdout.write` so you see tokens appear in real-time, then log the usage and finish reason after the stream ends.

> **Beginner Note:** Notice that `streamText` is NOT awaited when called — it returns immediately with a stream object. You then consume the stream with `for await...of`. The `await` happens inside the loop as each chunk arrives.

### Stream Consumption Patterns

There are several ways to consume a stream. Create `src/examples/streaming-patterns.ts` that demonstrates all three patterns:

**Pattern 1: Token-by-token with `textStream`** — iterate with `for await...of` and write each chunk to stdout.

**Pattern 2: Collect the full text** — instead of iterating, simply `await result.text` to get the complete response as a single string once streaming finishes.

```typescript
const fullText = await result.text // Waits for the entire stream, returns the full string
```

**Pattern 3: Using callbacks** — pass `onChunk` and `onFinish` callbacks directly to `streamText`:

```typescript
const result = streamText({
  model,
  prompt: '...',
  onChunk(event) {
    /* called for each chunk */
  },
  onFinish(event) {
    /* called when stream completes, event.usage available */
  },
})
```

Build one function per pattern, run them sequentially in a `main()` function, and compare how each pattern feels. When would you use Pattern 2 vs Pattern 1? When would callbacks be useful?

### Streaming with Timing Information

Measuring token delivery speed helps you understand model performance and detect issues. Create `src/examples/streaming-timing.ts` that wraps a `streamText` call with timing instrumentation.

**What to track:**

- `startTime` — record with `performance.now()` before calling `streamText`
- `firstTokenTime` — record when the first chunk arrives (this gives you TTFT)
- `chunkCount` and `totalChars` — increment as you iterate the stream
- `endTime` — record after the stream completes

**The approach:** Use `performance.now()` to capture timestamps. Inside your `for await` loop, check if `firstTokenTime` is still `null` — if so, this is the first chunk, so record the time. After the loop, calculate TTFT, total time, stream duration, and chars/second.

**Guiding questions:** Why is `performance.now()` better than `Date.now()` for this? What is TTFT and why does it matter for user experience? How would you compute average characters per second from the values you tracked?

> **Advanced Note:** Time to First Token (TTFT) is a critical metric in production. It determines how quickly your user sees _something_ happen. TTFT varies by model (Haiku < Sonnet < Opus), prompt length (longer prompts = slower TTFT), and server load. Monitor TTFT in production to catch regressions.

### When to Use generateText vs streamText

| Scenario                      | Use            | Why                              |
| ----------------------------- | -------------- | -------------------------------- |
| User-facing chat              | `streamText`   | Perceived latency matters        |
| Background processing         | `generateText` | Simpler code, no stream handling |
| Pipeline/batch jobs           | `generateText` | Streaming adds no value          |
| Real-time UI updates          | `streamText`   | Progressive rendering            |
| Short responses (< 50 tokens) | `generateText` | Streaming overhead not worth it  |
| Long responses (> 200 tokens) | `streamText`   | User sees progress               |

---

## Section 6: Model Parameters

### Temperature

**Temperature** controls the randomness of the model's output. It is the single most important parameter to understand.

- **Temperature 0:** Deterministic. The model always picks the most likely next token. Ideal for factual Q&A, classification, and structured output.
- **Temperature 0.5:** Balanced. Some creativity while staying mostly coherent. Good for general-purpose use.
- **Temperature 1.0:** Creative. The model explores less likely tokens. Good for brainstorming and creative writing.
- **Temperature > 1.0:** Very random. Outputs become increasingly incoherent. Rarely useful.

Create `src/examples/temperature-comparison.ts` that empirically demonstrates temperature's effect. The approach:

1. Pick a creative prompt (e.g., "Write a one-sentence description of a sunset.")
2. Loop through several temperature values: `[0, 0.3, 0.7, 1.0]`
3. At each temperature, run the same prompt 3 times
4. Log all results so you can compare variation

The temperature parameter is passed directly to `generateText`:

```typescript
const result = await generateText({ model, prompt, temperature: 0.7 })
```

Build this and run it. You should observe:

- At temperature 0, all three runs produce identical (or nearly identical) output.
- At temperature 1.0, each run produces a distinctly different sentence.

### Top-P (Nucleus Sampling)

**Top-P** is an alternative to temperature for controlling randomness. Instead of scaling probabilities, it limits the pool of tokens the model considers.

- **Top-P 0.1:** Only consider the top 10% most likely tokens. Very focused.
- **Top-P 0.9:** Consider the top 90% of tokens. More diverse.
- **Top-P 1.0:** Consider all tokens (no filtering).

```typescript
const result = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Invent a name for a new programming language.',
  topP: 0.9,
})
```

> **Advanced Note:** Most practitioners use temperature OR top-P, not both simultaneously. Adjust temperature and leave top-P at its default. If you set both, the interaction between them can be unpredictable. Mistral will reject the request with a 400 error if you pass `topP` with `temperature: 0` — it requires `topP: 1` for greedy sampling. Other providers silently ignore `topP` in this case.

### Max Tokens

**maxOutputTokens** sets an upper limit on the response length. This is critical for cost control and preventing runaway responses.

Create `src/examples/max-tokens.ts` that demonstrates the effect of `maxOutputTokens`. Make two calls with the same prompt (e.g., "Explain the theory of relativity in detail."):

1. One with a very small limit (`maxOutputTokens: 50`) — the model will be cut off
2. One with a generous limit (`maxOutputTokens: 1000`) — the model finishes naturally

```typescript
const result = await generateText({ model, prompt, maxOutputTokens: 50 })
```

For each, log the text, `finishReason`, and `usage.outputTokens`. You should see `finishReason: "length"` for the short limit and `finishReason: "stop"` for the generous one.

> **Beginner Note:** Setting `maxOutputTokens` does not make the model _use_ that many tokens. It sets a ceiling. If the model can answer in 50 tokens, it will stop at 50 even if `maxOutputTokens` is 4000. Think of it as a safety net, not a target.

### Stop Sequences

**Stop sequences** tell the model to stop generating when it produces a specific string. This is useful for structured output and preventing the model from going off-topic.

Create `src/examples/stop-sequences.ts` that demonstrates how stop sequences work. For example, stopping at the first newline forces single-line output:

```typescript
const result = await generateText({
  model,
  prompt: 'Give me a motivational quote:',
  stopSequences: ['\n'],
})
```

Experiment with different stop sequences — what happens if you stop at a period? At a comma? Log `finishReason` to confirm the model stopped due to the sequence.

### Frequency and Presence Penalties

These parameters reduce repetition in generated text:

- **Frequency penalty** (0 to 2): Penalizes tokens based on how many times they have appeared. Higher values reduce word repetition.
- **Presence penalty** (0 to 2): Penalizes tokens based on whether they have appeared at all. Higher values encourage topic diversity.

Create `src/examples/penalties.ts` that compares outputs with and without frequency/presence penalties. Use a prompt that tends to produce repetition (e.g., "List 10 creative uses for a paperclip.") and make two calls:

1. One with `frequencyPenalty: 0, presencePenalty: 0` (default — may repeat ideas)
2. One with `frequencyPenalty: 0.5, presencePenalty: 0.5` (should produce more diverse ideas)

```typescript
const result = await generateText({ model, prompt, frequencyPenalty: 0.5, presencePenalty: 0.5 })
```

Compare the outputs — do you see more variety with penalties enabled?

> **Advanced Note:** Penalties are more commonly used with OpenAI models. Claude models tend to be less repetitive by default. Over-penalizing can make output disjointed and nonsensical. Start with values between 0 and 0.5 and adjust based on results.

### Parameter Reference Table

| Parameter          | Range           | Default         | Use Case                     |
| ------------------ | --------------- | --------------- | ---------------------------- |
| `temperature`      | 0 - 2           | ~1.0            | Control randomness           |
| `topP`             | 0 - 1           | ~1.0            | Alternative to temperature   |
| `maxOutputTokens`  | 1 - model limit | Model-dependent | Cost control, length control |
| `stopSequences`    | string[]        | []              | Stop at delimiters           |
| `frequencyPenalty` | 0 - 2           | 0               | Reduce word repetition       |
| `presencePenalty`  | 0 - 2           | 0               | Encourage topic diversity    |

### Recommended Starting Points

| Task             | Temperature | Max Tokens | Notes                |
| ---------------- | ----------- | ---------- | -------------------- |
| Classification   | 0           | 10         | Deterministic, short |
| Factual Q&A      | 0           | 500        | Consistent answers   |
| Summarization    | 0.3         | 300        | Slight variation OK  |
| General chat     | 0.7         | 1000       | Balanced             |
| Creative writing | 1.0         | 2000       | Maximum creativity   |
| Brainstorming    | 1.0         | 1000       | Diverse ideas        |

---

## Section 7: Error Handling

### Why Error Handling Matters for LLM Apps

LLM API calls fail in ways that traditional APIs do not. Models are expensive to run, so providers aggressively rate-limit. Responses can take 10-30 seconds, so timeouts are common. And because models are probabilistic, you sometimes get responses that are technically valid but semantically useless.

Robust error handling is not optional — it is the difference between a demo and a product.

### Common Error Types

```typescript
/**
 * Common error scenarios when calling LLM APIs:
 *
 * 1. Authentication errors (invalid API key)
 * 2. Rate limit errors (too many requests)
 * 3. Model not found (typo in model ID)
 * 4. Context length exceeded (input too long)
 * 5. Network errors (timeout, connection refused)
 * 6. Server errors (provider outage)
 */
```

### Basic Error Handling Pattern

Create `src/examples/error-handling-basic.ts` with a `safeGenerate` function:

```typescript
async function safeGenerate(prompt: string): Promise<string | null> {
  /* ... */
}
```

**What it should do:**

1. Wrap `generateText` in a `try/catch`
2. On success, check `result.finishReason` — if it is `'length'`, log a warning that the response was truncated
3. Return `result.text` on success
4. In the `catch`, inspect `error.message` to identify the error type:
   - Contains `'401'` or `'authentication'` — log an API key error
   - Contains `'429'` or `'rate'` — log a rate limit error
   - Contains `'timeout'` — log a timeout error
5. Return `null` on failure

**Guiding questions:** Why return `null` instead of throwing? What is the difference between a `"length"` finish reason and an actual error? How would you use `instanceof Error` to safely access `error.message`?

### Retry Logic with Exponential Backoff

Rate limits are the most common error in LLM applications. The standard approach is exponential backoff: wait, retry, wait longer, retry again.

Create `src/examples/error-handling-retry.ts` that implements retry with exponential backoff. You will need these types and helpers:

```typescript
import type { LanguageModel } from 'ai'

interface RetryOptions {
  maxRetries: number
  initialDelayMs: number
  maxDelayMs: number
  backoffMultiplier: number
}

function sleep(ms: number): Promise<void> {
  /* ... */
}
function isRetryable(error: unknown): boolean {
  /* ... */
}

async function generateTextWithRetry(
  params: { model: LanguageModel; prompt?: string; system?: string; maxOutputTokens?: number; temperature?: number },
  options?: Partial<RetryOptions>
): Promise<string> {
  /* ... */
}
```

**What each function should do:**

- **`sleep`** — Return a promise that resolves after `ms` milliseconds (use `setTimeout` inside `new Promise`).
- **`isRetryable`** — Check `error.message` for status codes. Return `true` for transient errors (429, 500, 502, 503, timeout, ECONNRESET). Return `false` for permanent errors (401, 403, 400, 404). Default to `true` for unknown errors.
- **`generateTextWithRetry`** — Loop from attempt 0 to `maxRetries`. On each attempt: if not the first attempt, sleep for the current delay then multiply the delay by `backoffMultiplier` (capped at `maxDelayMs`). Call `generateText`. If it succeeds, return the text. If it fails with a non-retryable error, throw immediately. If all retries are exhausted, throw with a descriptive message.

**Guiding questions:** Why should you NOT retry a 401 error? Why multiply the delay on each attempt instead of using a fixed delay? What should the default delay progression look like (1s, 2s, 4s, 8s...)?

### Timeout Handling

Long responses can hang if the provider is experiencing issues. Always set reasonable timeouts:

Create `src/examples/error-handling-timeout.ts` with a `generateWithTimeout` function:

```typescript
async function generateWithTimeout(prompt: string, timeoutMs: number = 30000): Promise<string> {
  /* ... */
}
```

**What it should do:**

1. Create an `AbortController` instance
2. Set a `setTimeout` that calls `controller.abort()` after `timeoutMs`
3. Pass `abortSignal: controller.signal` to `generateText`
4. In the `catch`, check if the error is an `AbortError` (check `error.name === 'AbortError'`) — if so, throw a descriptive timeout error
5. In the `finally` block, always `clearTimeout` to prevent the timer from firing after a successful response

**Key API pattern:**

```typescript
const controller = new AbortController()
const result = await generateText({ model, prompt, abortSignal: controller.signal })
```

**Guiding questions:** Why do you need the `finally` block to clear the timeout? What happens if you forget to clear it? How is `AbortError` different from other errors?

Test it with a reasonable timeout (30s) and an impossibly short one (100ms) to see both success and failure paths.

### Comprehensive Error Handler

Now build a production-grade wrapper that combines retry logic, timeouts, and structured error reporting. Create `src/utils/llm-client.ts` with these types and exports:

```typescript
// src/utils/llm-client.ts

import type { LanguageModel, ModelMessage } from 'ai'

interface LLMCallOptions {
  model?: LanguageModel
  system?: string
  messages?: ModelMessage[]
  prompt?: string
  maxOutputTokens?: number
  temperature?: number
  timeoutMs?: number
  maxRetries?: number
}

interface LLMResult {
  text: string
  usage: { inputTokens: number; outputTokens: number; totalTokens: number }
  finishReason: string
  durationMs: number
}

class LLMError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly retryable: boolean,
    public readonly originalError?: Error
  ) {
    /* call super(message), set this.name */
  }
}

function classifyError(error: unknown): LLMError {
  /* ... */
}

export async function llmCall(options: LLMCallOptions): Promise<LLMResult> {
  /* ... */
}
```

**What each piece should do:**

- **`LLMError`** — Extends `Error` with a `code` string (e.g., `'RATE_LIMIT'`, `'AUTH_ERROR'`, `'TIMEOUT'`, `'CONTEXT_LENGTH'`, `'SERVER_ERROR'`), a `retryable` boolean, and an optional `originalError`.
- **`classifyError`** — Inspect the error message to determine the type. Map `'401'`/`'authentication'` to `AUTH_ERROR` (not retryable), `'429'`/`'rate'` to `RATE_LIMIT` (retryable), `'500'`/`'502'`/`'503'` to `SERVER_ERROR` (retryable), `'timeout'`/`'AbortError'` to `TIMEOUT` (retryable), `'context'`/`'token'` to `CONTEXT_LENGTH` (not retryable). Default to `UNKNOWN` (retryable).
- **`llmCall`** — Combine everything: destructure options with defaults, loop with retry, create an `AbortController` with timeout for each attempt, call `generateText`, measure duration with `performance.now()`, classify errors on failure, only retry if `retryable` is true, use exponential backoff.

**Guiding questions:** Why create a new `AbortController` for each attempt? Why measure `durationMs` inside the function instead of letting the caller do it? What should happen if `options.model` is not provided?

> **Beginner Note:** You do not need to write this much error handling for every exercise. Start with a simple try/catch. Build toward the comprehensive version as your applications become more complex.

### Error Categorization

The comprehensive error handler above uses `classifyError` to turn raw exceptions into typed `LLMError` instances. This is a production-essential pattern: different error types require different responses, and treating all errors the same leads to poor user experience and wasted retries.

Here are the five error categories every LLM application should distinguish:

| Category             | HTTP Status         | Retryable?                | User-Facing Action                       |
| -------------------- | ------------------- | ------------------------- | ---------------------------------------- |
| **Rate Limit**       | 429                 | Yes (with backoff)        | "Please wait a moment and try again"     |
| **Authentication**   | 401, 403            | No                        | "Check your API key configuration"       |
| **Network**          | Timeout, ECONNRESET | Yes                       | "Connection issue — retrying"            |
| **Model Not Found**  | 404                 | No                        | "Invalid model ID — check configuration" |
| **Context Overflow** | 400 (token-related) | No (need to reduce input) | "Input too long — try a shorter message" |

The key insight is that **retryable errors** (rate limit, network, server errors) should trigger automatic retry with backoff, while **non-retryable errors** (auth, model not found, context overflow) should fail immediately with a clear message. Retrying a 401 wastes time; failing immediately on a 429 wastes an opportunity.

A typed error class makes downstream handling clean:

```typescript
// Pattern: typed error handling in application code
try {
  const result = await llmCall({ prompt: 'Hello' })
} catch (error) {
  if (error instanceof LLMError) {
    switch (error.code) {
      case 'RATE_LIMIT':
        // Already retried internally — show the user a wait message
        break
      case 'AUTH_ERROR':
        // Configuration problem — alert the developer
        break
      case 'CONTEXT_LENGTH':
        // Input too long — truncate and retry
        break
      default:
        // Unexpected — log for investigation
        break
    }
  }
}
```

> **Advanced Note:** Some providers include a `Retry-After` header in 429 responses, telling you exactly how long to wait. Production clients parse this header and use it as the backoff delay instead of a fixed exponential schedule. The Vercel AI SDK does not expose this header directly, but you can access it by inspecting the raw error object from the provider.

---

## Section 8: Resilient API Clients

### Why Resilience Matters

In production, your LLM application will make thousands of API calls per day. Transient failures — rate limits, server hiccups, network blips — are not exceptions, they are routine. A resilient API client handles these automatically so your application logic never sees them.

The three building blocks of a resilient client are:

1. **Retry with exponential backoff** — wait longer between each retry attempt
2. **Jitter** — add randomness to the delay so multiple clients do not all retry at the same instant
3. **Error classification** — only retry errors that are actually transient

### Exponential Backoff with Jitter

Plain exponential backoff has a problem: if 100 clients all get rate-limited at the same time, they all retry after 1 second, then 2 seconds, then 4 seconds — in perfect synchrony. This creates "retry storms" that keep hammering the server at regular intervals.

Jitter solves this by adding randomness. The "full jitter" strategy picks a random delay between 0 and the exponential backoff ceiling:

```typescript
// Exponential backoff with full jitter
function calculateDelay(attempt: number, baseDelayMs: number, maxDelayMs: number): number {
  const exponentialDelay = baseDelayMs * Math.pow(2, attempt)
  const cappedDelay = Math.min(exponentialDelay, maxDelayMs)
  // Full jitter: random value between 0 and the capped delay
  return Math.random() * cappedDelay
}

// attempt 0: random between 0 and 1000ms
// attempt 1: random between 0 and 2000ms
// attempt 2: random between 0 and 4000ms
// attempt 3: random between 0 and 8000ms (capped at maxDelayMs)
```

### Building the Retry Wrapper

A production retry wrapper combines backoff, jitter, error classification, and abort signal support:

```typescript
import type { LanguageModel } from 'ai'

interface RetryConfig {
  maxRetries: number
  baseDelayMs: number
  maxDelayMs: number
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 30000,
}
```

The wrapper should:

1. Accept any `generateText` parameters plus a `RetryConfig`
2. Classify each error to decide whether to retry
3. Calculate the delay with jitter for each attempt
4. Respect an optional `AbortSignal` for cancellation
5. Throw the last error if all retries are exhausted

> **Beginner Note:** Jitter may seem like a small detail, but it is the difference between a retry strategy that works in development and one that works at scale. AWS, Google Cloud, and every major API client library uses jitter for exactly this reason.

---

## Quiz

### Question 1 (Easy)

What is the primary difference between `generateText` and `streamText`?

- A) `generateText` uses the OpenAI API while `streamText` uses the Anthropic API
- B) `generateText` waits for the complete response; `streamText` delivers tokens incrementally
- C) `generateText` is async while `streamText` is synchronous
- D) `streamText` is faster because it uses a different model

**Answer: B** — `generateText` blocks until the full response is available, while `streamText` returns a stream that delivers text chunks as the model generates them. Both work with any provider.

---

### Question 2 (Easy)

Which environment variable does the `@ai-sdk/mistral` package read automatically?

- A) `MISTRAL_SECRET`
- B) `AI_API_KEY`
- C) `MISTRAL_API_KEY`
- D) `MISTRAL_TOKEN`

**Answer: C** — The Mistral provider reads `MISTRAL_API_KEY` from the environment. Each AI SDK provider follows the same convention: `@ai-sdk/openai` reads `OPENAI_API_KEY`, `@ai-sdk/anthropic` reads `ANTHROPIC_API_KEY`, etc.

---

### Question 3 (Medium)

If `result.finishReason` is `"length"`, what happened?

- A) The model encountered an error
- B) The response hit the `maxOutputTokens` limit and was truncated
- C) The model finished generating naturally
- D) A stop sequence was reached

**Answer: B** — A finish reason of `"length"` means the model was cut off because it reached the `maxOutputTokens` limit before it could complete its response. You should increase `maxOutputTokens` or shorten your prompt.

---

### Question 4 (Medium)

What temperature setting should you use for a classification task that must return consistent results?

- A) 1.0
- B) 0.7
- C) 0.5
- D) 0

**Answer: D** — Temperature 0 produces deterministic output, which is what you want for classification. The model will consistently pick the most probable token at each step, giving you the same result for the same input.

---

### Question 5 (Hard)

You have a production application that makes 100 LLM calls per minute. Occasionally you receive `429` errors. Which combination of strategies best addresses this?

- A) Increase temperature and add longer system prompts
- B) Exponential backoff with retry, plus an `AbortController` timeout
- C) Switch to a different model to avoid the rate limit
- D) Catch all errors silently and return empty strings

**Answer: B** — Rate limits (429) require retry with exponential backoff to avoid hammering the API. Adding `AbortController` timeouts prevents individual requests from hanging indefinitely. Option C might help short-term but does not address the underlying pattern. Option D silently loses data.

---

### Question 6 (Medium)

Why is jitter added to exponential backoff in a resilient API client?

- A) To make the retry delay shorter on average
- B) To prevent synchronized retry storms when many clients are rate-limited simultaneously
- C) To randomize which provider receives the retry request
- D) To ensure retries happen at exact power-of-two intervals

**Answer: B** — Without jitter, all clients that hit a rate limit at the same time will retry in lockstep (1s, 2s, 4s), creating periodic spikes that keep hammering the server. Jitter adds randomness to the delay so retries spread out over time, preventing these synchronized retry storms.

---

### Question 7 (Hard)

A resilient retry wrapper classifies errors before deciding whether to retry. Which of these errors should NOT be retried?

- A) HTTP 429 (Too Many Requests)
- B) HTTP 500 (Internal Server Error)
- C) HTTP 401 (Unauthorized — invalid API key)
- D) A network timeout (ETIMEDOUT)

**Answer: C** — A 401 error means the API key is invalid or missing. Retrying will never succeed because the credentials are wrong — this is a permanent error. Rate limits (429), server errors (500), and network timeouts are transient and may succeed on retry. Retrying permanent errors wastes time and resources.

---

## Exercises

### Exercise 1: Make a Successful generateText Call

**Objective:** Verify your setup by making a successful `generateText` call and inspecting the full result object.

**Specification:**

1. Create a file `src/exercises/m01/ex01-first-call.ts`
2. Export an async function `firstCall(): Promise<{ text: string; tokens: number; finishReason: string }>`
3. Use `generateText` with the Mistral provider to ask `"What is TypeScript? Answer in one sentence."`
4. Return an object with the `text`, `totalTokens` from usage, and `finishReason`
5. Log the full result to the console

**Test specification:**

```typescript
// tests/exercises/m01/ex01-first-call.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 1: First Call', () => {
  it('should return text, tokens, and finishReason', async () => {
    const result = await firstCall()
    expect(result.text).toBeTruthy()
    expect(result.tokens).toBeGreaterThan(0)
    expect(result.finishReason).toBe('stop')
  })
})
```

---

### Exercise 2: Provider Factory Function

**Objective:** Build a provider factory that creates model instances from a configuration object, demonstrating the power of the provider abstraction.

**Specification:**

1. Create a file `src/exercises/m01/ex02-provider-factory.ts`
2. Define a type `ProviderName = 'anthropic' | 'openai'`
3. Export a function `createModel(provider: ProviderName, modelId?: string): LanguageModel`
4. If no `modelId` is provided, use sensible defaults:
   - Anthropic: `"claude-sonnet-4-20250514"`
   - OpenAI: `"gpt-5.4"`
5. Throw a descriptive error for unknown providers
6. Export a function `testModel(provider: ProviderName, modelId?: string): Promise<string>` that uses `createModel` to generate a response to `"Say hello in one word."` and returns the text

**Test specification:**

```typescript
// tests/exercises/m01/ex02-provider-factory.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 2: Provider Factory', () => {
  it('should create a Mistral model with default ID', () => {
    const model = createModel('mistral')
    expect(model).toBeDefined()
  })

  it('should create a Mistral model with custom ID', () => {
    const model = createModel('mistral', 'mistral-large-latest')
    expect(model).toBeDefined()
  })

  it('should throw for unknown providers', () => {
    expect(() => createModel('gemini' as ProviderName)).toThrow()
  })

  it('should generate text using the factory', async () => {
    const response = await testModel('anthropic')
    expect(response.length).toBeGreaterThan(0)
  })
})
```

---

### Exercise 3: Temperature Comparison

**Objective:** Empirically observe how temperature affects output by comparing the same prompt at different temperature settings.

**Specification:**

1. Create a file `src/exercises/m01/ex03-temperature-comparison.ts`
2. Export an async function `compareTemperatures(prompt: string, temperatures: number[], runs: number): Promise<TemperatureResult[]>`
3. Define the `TemperatureResult` type:
   ```typescript
   interface TemperatureResult {
     temperature: number
     responses: string[]
     uniqueResponses: number
     averageLength: number
   }
   ```
4. For each temperature, run the prompt `runs` times and collect results
5. Calculate the number of unique responses and average response length
6. Print a formatted comparison table to the console

**Example usage:**

```typescript
await compareTemperatures('Name one color.', [0, 0.5, 1.0], 5)
```

**Expected output format:**

```
=== Temperature Comparison ===
Prompt: "Name one color."
Runs per temperature: 5

Temperature 0.0:
  Responses: Blue, Blue, Blue, Blue, Blue
  Unique: 1/5
  Avg length: 4 chars

Temperature 0.5:
  Responses: Blue, Red, Blue, Green, Blue
  Unique: 3/5
  Avg length: 4.2 chars

Temperature 1.0:
  Responses: Magenta, Teal, Crimson, Gold, Periwinkle
  Unique: 5/5
  Avg length: 6.4 chars
```

**Test specification:**

```typescript
// tests/exercises/m01/ex03-temperature-comparison.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 3: Temperature Comparison', () => {
  it('should return results for each temperature', async () => {
    const results = await compareTemperatures('Say a number 1-10.', [0, 1.0], 3)
    expect(results).toHaveLength(2)
  })

  it('should have fewer unique responses at temperature 0', async () => {
    const results = await compareTemperatures('Name a fruit.', [0, 1.0], 5)
    const lowTemp = results.find(r => r.temperature === 0)!
    const highTemp = results.find(r => r.temperature === 1.0)!
    expect(lowTemp.uniqueResponses).toBeLessThanOrEqual(highTemp.uniqueResponses)
  })
})
```

---

### Exercise 4: Streaming with Timing

**Objective:** Build a streaming response handler that displays tokens in real-time and reports detailed timing metrics.

**Specification:**

1. Create a file `src/exercises/m01/ex04-streaming-timing.ts`
2. Export an async function `streamWithTiming(prompt: string): Promise<StreamingMetrics>`
3. Define the `StreamingMetrics` type:
   ```typescript
   interface StreamingMetrics {
     text: string
     timeToFirstTokenMs: number
     totalTimeMs: number
     totalChunks: number
     totalCharacters: number
     charsPerSecond: number
     chunkTimestamps: number[] // ms since start for each chunk
   }
   ```
4. Use `streamText` to get the response
5. Print each chunk to stdout as it arrives (using `process.stdout.write`)
6. Record the timestamp of each chunk relative to the start
7. After the stream completes, print a timing report

**Expected output format:**

```
[Streaming response...]
The quick brown fox jumps over the lazy dog...

--- Timing Report ---
Time to first token: 245ms
Total time: 1,847ms
Chunks received: 42
Characters generated: 312
Speed: 168.9 chars/sec
```

**Test specification:**

```typescript
// tests/exercises/m01/ex04-streaming-timing.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 4: Streaming with Timing', () => {
  it('should return complete streaming metrics', async () => {
    const metrics = await streamWithTiming("Say 'hello world'.")
    expect(metrics.text).toBeTruthy()
    expect(metrics.timeToFirstTokenMs).toBeGreaterThan(0)
    expect(metrics.totalTimeMs).toBeGreaterThan(metrics.timeToFirstTokenMs)
    expect(metrics.totalChunks).toBeGreaterThan(0)
    expect(metrics.charsPerSecond).toBeGreaterThan(0)
  })

  it('should have TTFT less than total time', async () => {
    const metrics = await streamWithTiming('Count from 1 to 10.')
    expect(metrics.timeToFirstTokenMs).toBeLessThan(metrics.totalTimeMs)
  })

  it('should record a timestamp for each chunk', async () => {
    const metrics = await streamWithTiming('List 3 colors.')
    expect(metrics.chunkTimestamps.length).toBe(metrics.totalChunks)
    // Timestamps should be monotonically increasing
    for (let i = 1; i < metrics.chunkTimestamps.length; i++) {
      expect(metrics.chunkTimestamps[i]).toBeGreaterThanOrEqual(metrics.chunkTimestamps[i - 1])
    }
  })
})
```

---

### Exercise 5: Resilient API Client

**Objective:** Build a retry wrapper with exponential backoff and jitter that handles rate limits gracefully.

**Specification:**

1. Create a file `src/exercises/m01/ex05-retry-wrapper.ts`
2. Define a `RetryConfig` type:
   ```typescript
   interface RetryConfig {
     maxRetries: number
     baseDelayMs: number
     maxDelayMs: number
   }
   ```
3. Export a function `calculateDelay(attempt: number, baseDelayMs: number, maxDelayMs: number): number` that returns a jittered exponential backoff delay (random value between 0 and the capped exponential delay)
4. Export a function `isRetryableError(error: unknown): boolean` that returns `true` for transient errors (429, 500, 503, timeout) and `false` for permanent errors (401, 403, 400, 404)
5. Export an async function `generateTextWithRetry(params: { model: LanguageModel; prompt: string; system?: string }, config?: Partial<RetryConfig>): Promise<string>` that wraps `generateText` with retry logic using your `calculateDelay` and `isRetryableError` functions
6. The wrapper should respect an optional `AbortSignal` for cancellation
7. If all retries are exhausted, throw an error including the attempt count and last error message

**Test specification:**

```typescript
// tests/exercises/m01/ex05-retry-wrapper.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 5: Resilient API Client', () => {
  it('calculateDelay should return values within the expected range', () => {
    for (let i = 0; i < 100; i++) {
      const delay = calculateDelay(2, 1000, 30000)
      expect(delay).toBeGreaterThanOrEqual(0)
      expect(delay).toBeLessThanOrEqual(4000) // 1000 * 2^2 = 4000
    }
  })

  it('calculateDelay should cap at maxDelayMs', () => {
    for (let i = 0; i < 100; i++) {
      const delay = calculateDelay(10, 1000, 5000)
      expect(delay).toBeLessThanOrEqual(5000)
    }
  })

  it('isRetryableError should return true for rate limit errors', () => {
    expect(isRetryableError(new Error('429 Too Many Requests'))).toBe(true)
  })

  it('isRetryableError should return false for bad request errors', () => {
    expect(isRetryableError(new Error('400 Bad Request'))).toBe(false)
  })

  it('should successfully generate text with retry wrapper', async () => {
    const result = await generateTextWithRetry({
      model: mistral('mistral-small-latest'),
      prompt: 'Say hello in one word.',
    })
    expect(result.length).toBeGreaterThan(0)
  })
})
```

---

## Troubleshooting

### Common Setup Issues

| Problem                                                  | Cause                              | Solution                                        |
| -------------------------------------------------------- | ---------------------------------- | ----------------------------------------------- |
| `Missing required environment variable: MISTRAL_API_KEY` | `.env` file missing or key not set | Copy `.env.example` to `.env` and add your key  |
| `Error: 401 Unauthorized`                                | Invalid API key                    | Verify key at console.mistral.ai                |
| `Cannot find module 'ai'`                                | Dependencies not installed         | Run `bun install`                               |
| `TypeError: mistral is not a function`                   | Wrong import syntax                | Use `import { mistral } from '@ai-sdk/mistral'` |
| `ECONNREFUSED 127.0.0.1:11434`                           | Ollama not running                 | Start Ollama with `ollama serve`                |
| Response is empty string                                 | Model returned no content          | Check your prompt — it may be too vague         |

> **Looking Ahead: AI SDK Middleware** — The Vercel AI SDK supports a middleware system via `wrapLanguageModel()` that lets you intercept and transform model calls. You can add logging, caching, guardrails, or custom logic as composable wrappers around any model — without changing your application code. You'll see patterns throughout this course (observability in Module 23, guardrails in Module 21, caching in Module 22) that can all be implemented as middleware. We'll use direct implementations for clarity, but know that middleware is the idiomatic way to compose these concerns in production.

---

## Summary

In this module, you learned:

1. **Project setup:** How to initialize a Bun + TypeScript project with the Vercel AI SDK and configure environment variables.
2. **Providers:** How the provider abstraction works and how to configure Mistral, Groq, Anthropic, OpenAI, and Ollama.
3. **generateText:** How to make synchronous LLM calls, understand the result object, and use system prompts.
4. **Roles:** The system/user/assistant message roles and how to build multi-turn conversations.
5. **streamText:** How to stream responses for real-time output and measure timing metrics.
6. **Parameters:** How temperature, top-P, max tokens, and penalties shape model behavior.
7. **Error handling:** How to handle API errors, implement retries with exponential backoff, and set timeouts.
8. **Resilient API clients:** How to build production retry wrappers with exponential backoff and jitter to handle transient failures automatically.

You now have the foundation to build any LLM application. In Module 2, we will use these tools to master the art and science of prompt engineering.
