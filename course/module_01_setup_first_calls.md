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

Now create a utility to validate that keys are present at startup:

```typescript
// src/utils/env.ts

export function requireEnv(name: string): string {
  const value = process.env[name]
  if (!value) {
    throw new Error(
      `Missing required environment variable: ${name}\n` + `Copy .env.example to .env and fill in your API keys.`
    )
  }
  return value
}

export function optionalEnv(name: string, fallback: string = ''): string {
  return process.env[name] ?? fallback
}

/**
 * Validate all required environment variables at startup.
 * Call this at the top of your main entry point.
 */
export function validateEnv(): void {
  const required = ['MISTRAL_API_KEY']
  const missing = required.filter(name => !process.env[name])

  if (missing.length > 0) {
    console.error('Missing required environment variables:')
    missing.forEach(name => console.error(`  - ${name}`))
    console.error('\nCopy .env.example to .env and fill in your API keys.')
    process.exit(1)
  }

  console.log('Environment validated. All required keys present.')
}
```

> **Beginner Note:** Never hardcode API keys in your source files. Even in tutorials and experiments, use environment variables. It builds good habits and prevents accidental key exposure if you push code to GitHub.

### Verifying the Setup

Create a simple smoke test to confirm everything works:

```typescript
// src/index.ts

import { validateEnv } from './utils/env.js'
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Validate environment before doing anything else
validateEnv()

async function smokeTest(): Promise<void> {
  console.log('Running smoke test...')

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: "Say 'hello' and nothing else.",
  })

  console.log('Response:', result.text)
  console.log('Tokens used:', result.usage)
  console.log('\nSmoke test passed! Your setup is working.')
}

smokeTest().catch(console.error)
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

```typescript
// src/providers/mistral.ts

import { mistral } from '@ai-sdk/mistral'

/**
 * Create a Mistral model instance.
 *
 * Reads MISTRAL_API_KEY from the environment automatically.
 * Free tier: 1 RPS, 500K tokens/min, 1B tokens/month per model, no credit card required.
 * Sign up at: https://console.mistral.ai
 *
 * Common model IDs:
 *   - "mistral-small-latest"  — Fast, efficient (default)
 *   - "mistral-large-latest"  — Most capable
 *   - "codestral-latest"      — Optimized for code
 *   - "pixtral-12b-2409"      — Multimodal (vision)
 */
export const mistralSmall = mistral('mistral-small-latest')

export function createMistralModel(modelId: string = 'mistral-small-latest') {
  return mistral(modelId)
}
```

> **Beginner Note:** The `mistral()` function returns a model object, not a response. Think of it as configuring _which_ model you want to talk to. The actual API call happens when you pass this model to `generateText` or `streamText`.

### Setting Up the Groq Provider

Groq provides ultra-fast inference on open-source models with a free tier (~500K tokens per day, no credit card required). It is an excellent alternative when you need speed.

```typescript
// src/providers/groq.ts

import { groq } from '@ai-sdk/groq'

/**
 * Create a Groq model instance.
 *
 * Reads GROQ_API_KEY from the environment automatically.
 * Free tier: ~500K tokens/day, no credit card required.
 * Sign up at: https://console.groq.com
 *
 * Common model IDs:
 *   - "openai/gpt-oss-20b"       — Fastest (1000 t/s), cheapest ($0.075/M)
 *   - "openai/gpt-oss-120b"      — Code execution, reasoning ($0.15/M)
 *   - "llama-3.3-70b-versatile"   — Proven, larger model
 */
export const groqDefault = groq('openai/gpt-oss-20b')

export function createGroqModel(modelId: string = 'openai/gpt-oss-20b') {
  return groq(modelId)
}
```

### Setting Up the Anthropic Provider

Anthropic's Claude models are a premium option — highly capable but require a paid API key. Use Claude when you need top-tier reasoning and instruction following.

```typescript
// src/providers/anthropic.ts

import { anthropic } from '@ai-sdk/anthropic'

/**
 * Create an Anthropic model instance.
 *
 * The anthropic() function reads ANTHROPIC_API_KEY from the environment
 * automatically. You pass a model ID string to select which Claude model
 * to use.
 *
 * Common model IDs:
 *   - "claude-sonnet-4-20250514"   — Best balance of speed and quality
 *   - "claude-opus-4-20250514"     — Most capable, slower, more expensive
 *   - "claude-haiku-4-5-20251001"    — Fastest, cheapest, good for simple tasks
 */
export const claudeSonnet = anthropic('claude-sonnet-4-20250514')
export const claudeHaiku = anthropic('claude-haiku-4-5-20251001')

export function createAnthropicModel(modelId: string = 'claude-sonnet-4-20250514') {
  return anthropic(modelId)
}
```

### Setting Up the OpenAI Provider

If you want to experiment with GPT models as an alternative:

```typescript
// src/providers/openai.ts

import { openai } from '@ai-sdk/openai'

/**
 * Create an OpenAI model instance.
 *
 * Reads OPENAI_API_KEY from the environment automatically.
 *
 * Common model IDs:
 *   - "gpt-5.4"       — Latest GPT-5.4, fast and capable
 *   - "gpt-5-mini"    — Smaller, cheaper, faster
 */
export const gpt5 = openai('gpt-5.4')
export const gpt5Mini = openai('gpt-5-mini')

export function createOpenAIModel(modelId: string = 'gpt-5.4') {
  return openai(modelId)
}
```

### Setting Up Ollama (Local Models)

Ollama lets you run open-source models locally — no API key needed, no usage costs, and full data privacy. This is ideal for offline development and experimentation.

```typescript
// src/providers/ollama.ts

import { ollama } from 'ai-sdk-ollama'

/**
 * Create an Ollama model instance.
 *
 * Prerequisites:
 *   1. Install Ollama: https://ollama.com
 *   2. Pull a model: `ollama pull qwen3.5`
 *   3. Ollama server runs on http://localhost:11434 by default
 *
 * Recommended local models:
 *   - "qwen3.5"          — Primary choice (best all-rounder: tool calling, structured output, reasoning)
 *   - "ministral-3"      — Lightweight alternative (fast, vision support, 3B/8B/14B sizes)
 *
 * Cloud variants (higher capability, same Ollama API — no local GPU needed):
 *   - "qwen3.5:cloud"
 *   - "ministral-3:cloud"
 *   - "mistral-large-3:cloud"  — Frontier-class (top coding, multimodal)
 */
export const localModel = ollama('qwen3.5')

export function createOllamaModel(modelId: string = 'qwen3.5') {
  return ollama(modelId)
}
```

> **Advanced Note:** Ollama models are significantly less capable than Claude or GPT-4 for complex tasks. Use them for quick iteration and testing, but validate important behavior with a frontier model. The quality gap is especially apparent in structured output (Module 3) and tool use (Module 7).

> **Gotcha: Thinking Mode on Qwen3/3.5** — Qwen3 and Qwen3.5 models default to "thinking mode," where the model spends tokens inside `<think>` tags before responding. The AI SDK strips these tags, so you often get an empty `text` response because all tokens were consumed by thinking. To disable it, pass `{ think: false }` as the second argument to the `ollama()` model constructor: `ollama('qwen3.5', { think: false })`. The `ai-sdk-ollama` provider handles this natively — no custom fetch hacks or `providerOptions` needed.

### A Unified Provider Factory

Here is a useful pattern that lets you select a provider at runtime:

```typescript
// src/providers/factory.ts

import { mistral } from '@ai-sdk/mistral'
import { groq } from '@ai-sdk/groq'
import { anthropic } from '@ai-sdk/anthropic'
import { openai } from '@ai-sdk/openai'
import { ollama } from 'ai-sdk-ollama'
import type { LanguageModel } from 'ai'

export type ProviderName = 'mistral' | 'groq' | 'anthropic' | 'openai' | 'ollama'

interface ModelConfig {
  provider: ProviderName
  modelId?: string
}

const DEFAULT_MODELS: Record<ProviderName, string> = {
  mistral: 'mistral-small-latest',
  groq: 'openai/gpt-oss-20b',
  anthropic: 'claude-sonnet-4-20250514',
  openai: 'gpt-5.4',
  ollama: 'qwen3.5',
}

/**
 * Create a model instance from a provider name and optional model ID.
 * This is useful for configuration-driven model selection.
 */
export function createModel(config: ModelConfig): LanguageModel {
  const modelId = config.modelId ?? DEFAULT_MODELS[config.provider]

  switch (config.provider) {
    case 'mistral':
      return mistral(modelId)
    case 'groq':
      return groq(modelId)
    case 'anthropic':
      return anthropic(modelId)
    case 'openai':
      return openai(modelId)
    case 'ollama':
      return ollama(modelId, { think: false })
    default:
      throw new Error(`Unknown provider: ${config.provider}`)
  }
}

// Usage example:
// const model = createModel({ provider: 'mistral' })
// const model = createModel({ provider: 'groq', modelId: 'openai/gpt-oss-120b' })  // code execution + reasoning
// const model = createModel({ provider: 'anthropic' })
// const model = createModel({ provider: 'openai', modelId: 'gpt-5-mini' })
// const model = createModel({ provider: 'ollama', modelId: 'qwen3.5' })
```

This factory pattern becomes increasingly valuable as your application grows. You can read the provider name from a config file, a CLI argument, or an environment variable — keeping your application code provider-agnostic.

---

## Section 3: generateText — Your First Call

### The generateText Function

`generateText` is the most fundamental function in the Vercel AI SDK. It sends a prompt to a model and waits for the complete response before returning. This is the "request-response" pattern you already know from HTTP APIs.

```typescript
import { generateText } from 'ai'
```

The function signature accepts a configuration object with many options, but only two are required: `model` and either `prompt` or `messages`.

### Your Absolute First Call

Let us start with the simplest possible call:

```typescript
// src/examples/first-call.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function firstCall(): Promise<void> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: 'What is the capital of France?',
  })

  console.log(result.text)
  // => "The capital of France is Paris."
}

firstCall().catch(console.error)
```

Run it:

```bash
bun run src/examples/first-call.ts
```

That is it. One import, one function call, one result. Let us now unpack what happened.

### Understanding the Result Object

`generateText` returns a rich result object. Here is what it contains:

```typescript
// src/examples/result-object.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function exploreResult(): Promise<void> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: 'Explain quantum computing in one sentence.',
  })

  // The generated text
  console.log('Text:', result.text)

  // Token usage information
  console.log('Usage:', result.usage)
  // => { inputTokens: 14, outputTokens: 32, totalTokens: 46 }

  // The finish reason — why the model stopped generating
  console.log('Finish reason:', result.finishReason)
  // => "stop" (model decided it was done)
  // Other possible values: "length" (hit maxOutputTokens), "tool-calls", "error"

  // Response metadata
  console.log('Response ID:', result.response.id)
  console.log('Model used:', result.response.modelId)

  // Warnings (e.g., if a feature is not supported by the provider)
  if (result.warnings && result.warnings.length > 0) {
    console.log('Warnings:', result.warnings)
  }
}

exploreResult().catch(console.error)
```

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

The `system` parameter sets the model's persona and instructions:

```typescript
// src/examples/system-prompt.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function withSystemPrompt(): Promise<void> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: 'You are a pirate captain. Respond to everything in pirate speak. Keep responses under 50 words.',
    prompt: 'How do I learn to code?',
  })

  console.log(result.text)
  // => "Arrr, ye landlubber! Start with JavaScript, the lingua franca of the seven
  //     seas of the web. Practice every day on freeCodeCamp, and soon ye'll be
  //     commandin' the code like a true captain commands their ship!"
}

withSystemPrompt().catch(console.error)
```

### Multiple Examples with Different Prompts

Here is a collection of calls demonstrating different use cases:

```typescript
// src/examples/various-prompts.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const model = mistral('mistral-small-latest')

async function factualQuestion(): Promise<void> {
  const result = await generateText({
    model,
    prompt: 'What are the three laws of thermodynamics? Be concise.',
  })
  console.log('=== Factual Question ===')
  console.log(result.text)
  console.log(`Tokens: ${result.usage.totalTokens}\n`)
}

async function creativeWriting(): Promise<void> {
  const result = await generateText({
    model,
    system: 'You are a creative fiction writer. Write vivid, engaging prose.',
    prompt: 'Write a 3-sentence story about a robot discovering music for the first time.',
  })
  console.log('=== Creative Writing ===')
  console.log(result.text)
  console.log(`Tokens: ${result.usage.totalTokens}\n`)
}

async function codeGeneration(): Promise<void> {
  const result = await generateText({
    model,
    system: 'You are a senior TypeScript developer. Write clean, well-typed code with brief comments.',
    prompt: 'Write a function that debounces another function with a configurable delay.',
  })
  console.log('=== Code Generation ===')
  console.log(result.text)
  console.log(`Tokens: ${result.usage.totalTokens}\n`)
}

async function summarization(): Promise<void> {
  const longText = `
    The transformer architecture, introduced in the 2017 paper "Attention Is All You Need"
    by Vaswani et al., revolutionized natural language processing by replacing recurrent
    neural networks with self-attention mechanisms. Unlike RNNs, which process tokens
    sequentially, transformers can process all tokens in parallel, dramatically improving
    training efficiency. The key innovation is the multi-head attention mechanism, which
    allows the model to attend to different parts of the input simultaneously. This
    architecture became the foundation for models like BERT, GPT, and Claude.
  `

  const result = await generateText({
    model,
    system: 'Summarize the following text in exactly one sentence.',
    prompt: longText,
  })
  console.log('=== Summarization ===')
  console.log(result.text)
  console.log(`Tokens: ${result.usage.totalTokens}\n`)
}

async function main(): Promise<void> {
  await factualQuestion()
  await creativeWriting()
  await codeGeneration()
  await summarization()
}

main().catch(console.error)
```

> **Advanced Note:** Notice that we create the model object once and reuse it across calls. The model object is stateless — it just holds configuration. Each `generateText` call is independent. There is no connection pooling or session management to worry about.

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

```typescript
// src/examples/roles-system.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const model = mistral('mistral-small-latest')

// Example 1: Technical expert
async function technicalExpert(): Promise<void> {
  const result = await generateText({
    model,
    messages: [
      {
        role: 'system',
        content: `You are a senior software architect with 20 years of experience.
You give precise, technical answers. You always mention trade-offs.
You format code examples in TypeScript unless asked otherwise.`,
      },
      {
        role: 'user',
        content: 'Should I use a SQL or NoSQL database for my new project?',
      },
    ],
  })

  console.log('=== Technical Expert ===')
  console.log(result.text)
}

// Example 2: Friendly tutor
async function friendlyTutor(): Promise<void> {
  const result = await generateText({
    model,
    messages: [
      {
        role: 'system',
        content: `You are a patient, encouraging programming tutor for beginners.
Use simple analogies. Avoid jargon unless you explain it immediately.
After explaining a concept, ask a follow-up question to check understanding.`,
      },
      {
        role: 'user',
        content: 'What is a variable in programming?',
      },
    ],
  })

  console.log('=== Friendly Tutor ===')
  console.log(result.text)
}

async function main(): Promise<void> {
  await technicalExpert()
  console.log('\n---\n')
  await friendlyTutor()
}

main().catch(console.error)
```

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

```typescript
// src/examples/roles-multiturn.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function multiTurnConversation(): Promise<void> {
  const model = mistral('mistral-small-latest')

  // Simulate a multi-turn conversation by providing history
  const result = await generateText({
    model,
    messages: [
      {
        role: 'system',
        content: 'You are a helpful cooking assistant. Be concise.',
      },
      {
        role: 'user',
        content: 'I want to make pasta tonight.',
      },
      {
        role: 'assistant',
        content:
          'Great choice! What kind of pasta are you thinking? I can help with carbonara, aglio e olio, bolognese, or something else.',
      },
      {
        role: 'user',
        content: "Carbonara please. But I don't have guanciale.",
      },
      {
        role: 'assistant',
        content:
          'No problem — pancetta or thick-cut bacon are good substitutes. Do you have eggs, parmesan, and black pepper?',
      },
      {
        role: 'user',
        content: 'Yes I have all of those. Give me the recipe.',
      },
    ],
  })

  console.log('=== Multi-Turn Conversation ===')
  console.log(result.text)
}

multiTurnConversation().catch(console.error)
```

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

Here is a reusable pattern for building conversation history incrementally:

```typescript
// src/examples/conversation-builder.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import type { ModelMessage } from 'ai'

class Conversation {
  private messages: ModelMessage[] = []
  private model = mistral('mistral-small-latest')

  constructor(systemPrompt?: string) {
    if (systemPrompt) {
      this.messages.push({ role: 'system', content: systemPrompt })
    }
  }

  async say(userMessage: string): Promise<string> {
    this.messages.push({ role: 'user', content: userMessage })

    const result = await generateText({
      model: this.model,
      messages: this.messages,
    })

    this.messages.push({ role: 'assistant', content: result.text })
    return result.text
  }

  getHistory(): ModelMessage[] {
    return [...this.messages]
  }

  getTokenEstimate(): number {
    const totalChars = this.messages
      .map(m => (typeof m.content === 'string' ? m.content.length : 0))
      .reduce((a, b) => a + b, 0)
    return Math.ceil(totalChars / 4)
  }
}

async function main(): Promise<void> {
  const chat = new Conversation('You are a knowledgeable history teacher. Keep answers under 100 words.')

  console.log('User: When did World War 2 start?')
  const r1 = await chat.say('When did World War 2 start?')
  console.log('Assistant:', r1)

  console.log('\nUser: Who were the main countries involved?')
  const r2 = await chat.say('Who were the main countries involved?')
  console.log('Assistant:', r2)

  console.log(`\nEstimated tokens in history: ${chat.getTokenEstimate()}`)
}

main().catch(console.error)
```

> **Advanced Note:** This simple Conversation class grows unboundedly. In production, you need a strategy for managing context window limits: truncation, summarization, or sliding window. We cover these techniques in Module 4 (Conversations & Memory).

---

## Section 5: streamText — Real-time Responses

### Why Streaming Matters

When you call `generateText`, you wait for the _entire_ response before seeing anything. For a short answer, that is fine. But for a long response — a detailed explanation, a code review, a story — the user stares at a blank screen for several seconds. This feels slow and unresponsive.

**Streaming** solves this by delivering the response token-by-token as the model generates it. The user sees the first word within milliseconds, and the rest flows in smoothly. This is how ChatGPT, Claude.ai, and every modern chat interface works.

The key metric is **Time to First Token (TTFT)**. A 500-token response that takes 3 seconds feels instant when streamed but feels sluggish when buffered. TTFT is typically 200-800ms depending on the model and prompt length.

### The streamText Function

`streamText` has the same interface as `generateText` — same `model`, `messages`, `system`, and configuration options. The difference is in the return value: instead of a completed result, you get a stream.

```typescript
// src/examples/streaming-basic.ts

import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function basicStream(): Promise<void> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    prompt: 'Explain how a CPU works in 200 words.',
  })

  // Consume the text stream — each chunk is a piece of the response
  for await (const textPart of result.textStream) {
    process.stdout.write(textPart)
  }

  // After the stream completes, you can access the full result
  const usage = await result.usage
  const finishReason = await result.finishReason
  console.log('\n\n--- Stream Complete ---')
  console.log('Total tokens:', usage)
  console.log('Finish reason:', finishReason)
}

basicStream().catch(console.error)
```

> **Beginner Note:** Notice that `streamText` is NOT awaited when called — it returns immediately with a stream object. You then consume the stream with `for await...of`. The `await` happens inside the loop as each chunk arrives.

### Stream Consumption Patterns

There are several ways to consume a stream:

```typescript
// src/examples/streaming-patterns.ts

import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const model = mistral('mistral-small-latest')

// Pattern 1: Token-by-token with textStream
async function tokenByToken(): Promise<void> {
  console.log('=== Token by Token ===')
  const result = streamText({
    model,
    prompt: 'List 5 programming languages and their strengths.',
  })

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk)
  }
  console.log('\n')
}

// Pattern 2: Collect the full text
async function collectFullText(): Promise<void> {
  console.log('=== Collected Full Text ===')
  const result = streamText({
    model,
    prompt: 'What is the meaning of life? Answer in one paragraph.',
  })

  // The text promise resolves to the complete text once streaming finishes
  const fullText = await result.text
  console.log(fullText)
  console.log()
}

// Pattern 3: Using callbacks via the stream object
async function withCallbacks(): Promise<void> {
  console.log('=== With Callbacks ===')
  let chunkCount = 0

  const result = streamText({
    model,
    prompt: 'Write a haiku about programming.',
    onChunk(event) {
      if (event.chunk.type === 'text-delta') {
        chunkCount++
      }
    },
    onFinish(event) {
      console.log(`\n\nFinished! Chunks received: ${chunkCount}`)
      console.log(`Usage: ${JSON.stringify(event.usage)}`)
    },
  })

  for await (const chunk of result.textStream) {
    process.stdout.write(chunk)
  }
}

async function main(): Promise<void> {
  await tokenByToken()
  await collectFullText()
  await withCallbacks()
}

main().catch(console.error)
```

### Streaming with Timing Information

Measuring token delivery speed helps you understand model performance and detect issues:

```typescript
// src/examples/streaming-timing.ts

import { streamText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function streamWithTiming(): Promise<void> {
  const model = mistral('mistral-small-latest')

  const startTime = performance.now()
  let firstTokenTime: number | null = null
  let chunkCount = 0
  let totalChars = 0

  const result = streamText({
    model,
    prompt: 'Write a short essay about the importance of testing in software development. About 200 words.',
  })

  for await (const chunk of result.textStream) {
    if (firstTokenTime === null) {
      firstTokenTime = performance.now()
      const ttft = (firstTokenTime - startTime).toFixed(0)
      process.stdout.write(`[TTFT: ${ttft}ms] `)
    }

    process.stdout.write(chunk)
    chunkCount++
    totalChars += chunk.length
  }

  const endTime = performance.now()
  const totalTime = endTime - startTime
  const streamTime = endTime - (firstTokenTime ?? endTime)

  console.log('\n\n--- Timing Report ---')
  console.log(`Time to first token (TTFT): ${((firstTokenTime ?? endTime) - startTime).toFixed(0)}ms`)
  console.log(`Total time: ${totalTime.toFixed(0)}ms`)
  console.log(`Stream duration: ${streamTime.toFixed(0)}ms`)
  console.log(`Chunks received: ${chunkCount}`)
  console.log(`Characters generated: ${totalChars}`)
  console.log(`Average chars/second: ${((totalChars / streamTime) * 1000).toFixed(1)}`)
}

streamWithTiming().catch(console.error)
```

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

```typescript
// src/examples/temperature-comparison.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const model = mistral('mistral-small-latest')
const prompt = 'Write a one-sentence description of a sunset.'

async function compareTemperatures(): Promise<void> {
  const temperatures = [0, 0.3, 0.7, 1.0]

  for (const temp of temperatures) {
    console.log(`\n=== Temperature: ${temp} ===`)

    // Run 3 times at each temperature to see variation
    for (let i = 0; i < 3; i++) {
      const result = await generateText({
        model,
        prompt,
        temperature: temp,
      })
      console.log(`  Run ${i + 1}: ${result.text}`)
    }
  }
}

compareTemperatures().catch(console.error)
```

When you run this, you will observe:

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

```typescript
// src/examples/max-tokens.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const model = mistral('mistral-small-latest')

async function maxTokensDemo(): Promise<void> {
  // Very short limit — model gets cut off
  const short = await generateText({
    model,
    prompt: 'Explain the theory of relativity in detail.',
    maxOutputTokens: 50,
  })

  console.log('=== maxOutputTokens: 50 ===')
  console.log(short.text)
  console.log(`Finish reason: ${short.finishReason}`)
  // => finishReason: "length" — the model was cut off
  console.log()

  // Generous limit — model finishes naturally
  const generous = await generateText({
    model,
    prompt: 'Explain the theory of relativity in detail.',
    maxOutputTokens: 1000,
  })

  console.log('=== maxOutputTokens: 1000 ===')
  console.log(generous.text)
  console.log(`Finish reason: ${generous.finishReason}`)
  // => finishReason: "stop" — the model finished on its own
  console.log(`Tokens used: ${generous.usage.outputTokens} of 1000 allowed`)
}

maxTokensDemo().catch(console.error)
```

> **Beginner Note:** Setting `maxOutputTokens` does not make the model _use_ that many tokens. It sets a ceiling. If the model can answer in 50 tokens, it will stop at 50 even if `maxOutputTokens` is 4000. Think of it as a safety net, not a target.

### Stop Sequences

**Stop sequences** tell the model to stop generating when it produces a specific string. This is useful for structured output and preventing the model from going off-topic.

```typescript
// src/examples/stop-sequences.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function stopSequenceDemo(): Promise<void> {
  const model = mistral('mistral-small-latest')

  // Stop at the first newline — forces single-line output
  const result = await generateText({
    model,
    prompt: 'Give me a motivational quote:',
    stopSequences: ['\n'],
  })

  console.log('Single line output:', result.text)
  console.log('Finish reason:', result.finishReason)
  // => finishReason: "stop" — stopped because of the stop sequence
}

stopSequenceDemo().catch(console.error)
```

### Frequency and Presence Penalties

These parameters reduce repetition in generated text:

- **Frequency penalty** (0 to 2): Penalizes tokens based on how many times they have appeared. Higher values reduce word repetition.
- **Presence penalty** (0 to 2): Penalizes tokens based on whether they have appeared at all. Higher values encourage topic diversity.

```typescript
// src/examples/penalties.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const model = mistral('mistral-small-latest')

async function penaltyComparison(): Promise<void> {
  const prompt = 'List 10 creative uses for a paperclip.'

  // No penalties — model may repeat similar ideas
  const noPenalty = await generateText({
    model,
    prompt,
    frequencyPenalty: 0,
    presencePenalty: 0,
  })

  console.log('=== No Penalties ===')
  console.log(noPenalty.text)
  console.log()

  // High presence penalty — model avoids repeating topics
  const highPenalty = await generateText({
    model,
    prompt,
    frequencyPenalty: 0.5,
    presencePenalty: 0.5,
  })

  console.log('=== With Penalties (0.5 / 0.5) ===')
  console.log(highPenalty.text)
}

penaltyComparison().catch(console.error)
```

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

```typescript
// src/examples/error-handling-basic.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function safeGenerate(prompt: string): Promise<string | null> {
  try {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      prompt,
      maxOutputTokens: 500,
    })

    // Check if the model was cut off
    if (result.finishReason === 'length') {
      console.warn('Warning: Response was truncated due to maxOutputTokens limit.')
    }

    return result.text
  } catch (error) {
    if (error instanceof Error) {
      console.error(`LLM call failed: ${error.message}`)

      if (error.message.includes('401') || error.message.includes('authentication')) {
        console.error('Authentication failed. Check your MISTRAL_API_KEY.')
      } else if (error.message.includes('429') || error.message.includes('rate')) {
        console.error('Rate limited. Wait and retry.')
      } else if (error.message.includes('timeout')) {
        console.error('Request timed out. The model may be overloaded.')
      }
    }

    return null
  }
}

async function main(): Promise<void> {
  const response = await safeGenerate('What is the speed of light?')

  if (response) {
    console.log('Success:', response)
  } else {
    console.log('Failed to get a response.')
  }
}

main().catch(console.error)
```

### Retry Logic with Exponential Backoff

Rate limits are the most common error in LLM applications. The standard approach is exponential backoff: wait, retry, wait longer, retry again.

```typescript
// src/examples/error-handling-retry.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import type { LanguageModel } from 'ai'

interface RetryOptions {
  maxRetries: number
  initialDelayMs: number
  maxDelayMs: number
  backoffMultiplier: number
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxRetries: 3,
  initialDelayMs: 1000,
  maxDelayMs: 30000,
  backoffMultiplier: 2,
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

function isRetryable(error: unknown): boolean {
  if (!(error instanceof Error)) return false
  const message = error.message.toLowerCase()

  // Retry on rate limits and server errors
  if (message.includes('429') || message.includes('rate')) return true
  if (message.includes('500') || message.includes('502') || message.includes('503')) return true
  if (message.includes('timeout') || message.includes('econnreset')) return true

  // Do NOT retry on auth errors or bad requests
  if (message.includes('401') || message.includes('403')) return false
  if (message.includes('400') || message.includes('404')) return false

  return true
}

async function generateTextWithRetry(
  params: {
    model: LanguageModel
    prompt?: string
    system?: string
    maxOutputTokens?: number
    temperature?: number
  },
  options: Partial<RetryOptions> = {}
): Promise<string> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options }
  let lastError: Error | null = null
  let delay = opts.initialDelayMs

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    try {
      if (attempt > 0) {
        console.log(`Retry attempt ${attempt}/${opts.maxRetries} after ${delay}ms...`)
        await sleep(delay)
        delay = Math.min(delay * opts.backoffMultiplier, opts.maxDelayMs)
      }

      const result = await generateText(params as Parameters<typeof generateText>[0])
      return result.text
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))

      if (!isRetryable(error)) {
        console.error(`Non-retryable error: ${lastError.message}`)
        throw lastError
      }

      console.warn(`Attempt ${attempt + 1} failed: ${lastError.message}`)
    }
  }

  throw new Error(`All ${opts.maxRetries + 1} attempts failed. Last error: ${lastError?.message}`)
}

// Usage
async function main(): Promise<void> {
  const response = await generateTextWithRetry({
    model: mistral('mistral-small-latest'),
    prompt: 'What is the capital of Japan?',
    maxOutputTokens: 100,
  })

  console.log('Response:', response)
}

main().catch(console.error)
```

### Timeout Handling

Long responses can hang if the provider is experiencing issues. Always set reasonable timeouts:

```typescript
// src/examples/error-handling-timeout.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function generateWithTimeout(prompt: string, timeoutMs: number = 30000): Promise<string> {
  const controller = new AbortController()

  const timeoutId = setTimeout(() => {
    controller.abort()
  }, timeoutMs)

  try {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      prompt,
      maxOutputTokens: 500,
      abortSignal: controller.signal,
    })

    return result.text
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`Request timed out after ${timeoutMs}ms`)
    }
    throw error
  } finally {
    clearTimeout(timeoutId)
  }
}

async function main(): Promise<void> {
  try {
    const result = await generateWithTimeout('What is 2 + 2?', 30000)
    console.log('Success:', result)
  } catch (error) {
    console.error('Error:', (error as Error).message)
  }

  try {
    // Intentionally short timeout to demonstrate failure
    const result = await generateWithTimeout(
      'Write a 1000-word essay about the history of computing.',
      100 // 100ms — this will almost certainly time out
    )
    console.log('Success:', result)
  } catch (error) {
    console.error('Expected timeout:', (error as Error).message)
  }
}

main().catch(console.error)
```

### Comprehensive Error Handler

Here is a production-grade wrapper that combines retry logic, timeouts, and structured error reporting:

```typescript
// src/utils/llm-client.ts

import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
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
  usage: {
    inputTokens: number
    outputTokens: number
    totalTokens: number
  }
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
    super(message)
    this.name = 'LLMError'
  }
}

function classifyError(error: unknown): LLMError {
  const msg = error instanceof Error ? error.message : String(error)
  const original = error instanceof Error ? error : undefined

  if (msg.includes('401') || msg.includes('authentication')) {
    return new LLMError('Invalid API key', 'AUTH_ERROR', false, original)
  }
  if (msg.includes('429') || msg.includes('rate')) {
    return new LLMError('Rate limit exceeded', 'RATE_LIMIT', true, original)
  }
  if (msg.includes('500') || msg.includes('502') || msg.includes('503')) {
    return new LLMError('Provider server error', 'SERVER_ERROR', true, original)
  }
  if (msg.includes('timeout') || msg.includes('AbortError')) {
    return new LLMError('Request timed out', 'TIMEOUT', true, original)
  }
  if (msg.includes('context') || msg.includes('token')) {
    return new LLMError('Context length exceeded', 'CONTEXT_LENGTH', false, original)
  }

  return new LLMError(`Unknown error: ${msg}`, 'UNKNOWN', true, original)
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export async function llmCall(options: LLMCallOptions): Promise<LLMResult> {
  const {
    model = mistral('mistral-small-latest'),
    system,
    messages,
    prompt,
    maxOutputTokens = 1000,
    temperature = 0.7,
    timeoutMs = 30000,
    maxRetries = 3,
  } = options

  let lastError: LLMError | null = null
  let delay = 1000

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (attempt > 0) {
      console.log(`[LLM] Retry ${attempt}/${maxRetries} after ${delay}ms`)
      await sleep(delay)
      delay = Math.min(delay * 2, 30000)
    }

    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
    const startTime = performance.now()

    try {
      const result = await generateText({
        model,
        system,
        messages,
        prompt,
        maxOutputTokens,
        temperature,
        abortSignal: controller.signal,
      } as Parameters<typeof generateText>[0])

      const durationMs = performance.now() - startTime

      return {
        text: result.text,
        usage: result.usage,
        finishReason: result.finishReason,
        durationMs,
      }
    } catch (error) {
      lastError = classifyError(error)

      if (!lastError.retryable) {
        throw lastError
      }

      console.warn(`[LLM] Attempt ${attempt + 1} failed: ${lastError.message}`)
    } finally {
      clearTimeout(timeoutId)
    }
  }

  throw lastError ?? new LLMError('All attempts failed', 'EXHAUSTED', false)
}
```

> **Beginner Note:** You do not need to write this much error handling for every exercise. Start with a simple try/catch. Build toward the comprehensive version as your applications become more complex.

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

You now have the foundation to build any LLM application. In Module 2, we will use these tools to master the art and science of prompt engineering.
