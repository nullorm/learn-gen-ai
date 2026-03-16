# Module 4: Conversations & Memory

## Learning Objectives

- Build multi-turn conversational applications with proper message array management
- Understand context window limits across models and how they constrain conversation length
- Implement multiple memory strategies: sliding window, summarization, and hybrid approaches
- Persist conversation history to disk for long-running sessions
- Estimate token counts to stay within model limits and control costs

---

## Why Should I Care?

A single LLM call answers a question. A conversation solves a problem. The difference between a toy demo and a production application is almost always memory — the ability to maintain context across multiple turns so the model can reference what was said before, track the user's evolving intent, and build toward a complex goal.

But memory is not free. Every model has a finite context window, and every token you send costs money and time. A naive approach — sending the entire conversation history with every request — works for five messages and collapses at fifty. Production systems need strategies: which messages to keep, which to discard, which to compress. The choice of strategy affects response quality, latency, cost, and user experience.

This module teaches you to think about conversation memory as an engineering problem with concrete trade-offs. You will implement three distinct strategies, measure their behavior, and build a system flexible enough to switch between them. By the end, you will have a reusable conversation engine that can power chatbots, assistants, tutoring systems, and any other multi-turn application.

---

## Connection to Other Modules

- **Module 1 (Setup)** established the `generateText` call pattern. Here we extend it to multi-turn conversations.
- **Module 2 (Prompt Engineering)** taught you to craft effective system prompts. Those system prompts become the anchor of every conversation.
- **Module 3 (Structured Output)** showed `generateText` with `Output.object()`. In this module, you will use structured output to generate conversation summaries.
- **Module 5 (Long Context & Caching)** explores what happens when context windows grow very large and how caching reduces costs.
- **Module 6 (Streaming)** shows how to stream conversational responses for real-time UX.
- **Module 7 (Tool Use)** adds tool calls to conversations, where memory management becomes even more critical.

---

## Section 1: Multi-turn Conversations

### How LLMs See Conversations

LLMs are stateless. They do not remember previous calls. Every time you call `generateText`, the model sees only what you send in that specific request. The illusion of a conversation is created entirely by the client: you accumulate messages and send the full array each time.

A conversation is represented as an array of messages, each with a `role` and `content`:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
  {
    role: 'system',
    content: 'You are a helpful cooking assistant. Give concise recipes.',
  },
  {
    role: 'user',
    content: 'How do I make pasta carbonara?',
  },
  {
    role: 'assistant',
    content:
      'Classic carbonara: Cook spaghetti. Fry guanciale until crispy. Mix eggs, pecorino, and black pepper. Toss hot pasta with guanciale, then stir in the egg mixture off heat. The residual heat cooks the eggs into a creamy sauce.',
  },
  {
    role: 'user',
    content: 'Can I use bacon instead?',
  },
]
```

The model sees the entire conversation and understands that "instead" refers to guanciale from the previous turn.

### Making a Multi-turn Call

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

async function chat(messages: Message[]): Promise<string> {
  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    messages,
  })
  return text
}

// Build a conversation step by step
const conversation: Message[] = [
  { role: 'system', content: 'You are a math tutor. Explain step by step.' },
  { role: 'user', content: 'What is the derivative of x^3?' },
]

const response1 = await chat(conversation)
console.log('Assistant:', response1)

// Append the assistant's response and the next user message
conversation.push({ role: 'assistant', content: response1 })
conversation.push({ role: 'user', content: 'What about the second derivative?' })

const response2 = await chat(conversation)
console.log('Assistant:', response2)
```

> **Beginner Note:** The `role: 'assistant'` messages are responses from the model that you append back into the array. This is how the model "remembers" what it said. You are responsible for maintaining this array — the API does not do it for you.

### Role Alternation Rules

Most providers expect messages to follow specific patterns:

1. An optional `system` message at the start (or multiple system messages, depending on the provider)
2. Alternating `user` and `assistant` messages after that
3. The final message should be `user` (you are asking the model to respond)

```typescript
// Correct pattern
const validConversation: Message[] = [
  { role: 'system', content: 'You are helpful.' },
  { role: 'user', content: 'Hello' },
  { role: 'assistant', content: 'Hi! How can I help?' },
  { role: 'user', content: 'Tell me a joke' },
]

// This may cause errors with some providers:
// Two user messages in a row without an assistant response between them
const problematic: Message[] = [
  { role: 'user', content: 'Hello' },
  { role: 'user', content: 'Are you there?' }, // Some providers reject this
]
```

> **Advanced Note:** The Vercel AI SDK handles some provider-specific quirks internally, but understanding the underlying constraints helps you debug issues:
>
> - **Anthropic** requires the first non-system message to be a `user` message, and messages must strictly alternate user/assistant.
> - **Mistral** also requires strict user/assistant alternation. The system message is prepended to the first user message internally — two consecutive user messages or two consecutive assistant messages will cause errors.
> - **OpenAI** is more lenient and allows consecutive messages of the same role.
> - **Ollama (Qwen3.5)** follows the chat template of the underlying model. Qwen models expect user/assistant alternation (like Mistral), though Ollama is generally forgiving about edge cases.

### A Complete Conversational Loop

Here is a full interactive conversation using standard input:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

const conversation: Message[] = [
  {
    role: 'system',
    content: 'You are a friendly assistant. Keep responses under 3 sentences unless asked for detail.',
  },
]

async function getResponse(messages: Message[]): Promise<string> {
  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    messages,
  })
  return text
}

// Simple REPL loop
const reader = Bun.stdin.stream().getReader()
const decoder = new TextDecoder()

console.log('Chat started. Type "quit" to exit.\n')

process.stdout.write('You: ')
while (true) {
  const { value, done } = await reader.read()
  if (done) break

  const input = decoder.decode(value).trim()
  if (input.toLowerCase() === 'quit') break

  conversation.push({ role: 'user', content: input })

  const response = await getResponse(conversation)
  conversation.push({ role: 'assistant', content: response })

  console.log(`\nAssistant: ${response}\n`)
  process.stdout.write('You: ')
}

console.log(`\nConversation ended. Total messages: ${conversation.length}`)
```

---

## Section 2: Context Windows

### What is a Context Window?

The context window is the maximum number of tokens a model can process in a single request. This includes everything: the system prompt, the conversation history, and the model's response. Think of it as the model's working memory — it can only "see" what fits in the window.

### Token Limits by Model

Here are approximate context window sizes for popular models (as of early 2026):

| Model            | Context Window | Approx. Words |
| ---------------- | -------------- | ------------- |
| Claude Sonnet 4  | 200K tokens    | ~150K words   |
| Claude Haiku 4.5 | 200K tokens    | ~150K words   |
| Claude Opus 4    | 200K tokens    | ~150K words   |
| GPT-4o           | 128K tokens    | ~96K words    |
| GPT-4o mini      | 128K tokens    | ~96K words    |
| Qwen 3.5         | 128K tokens    | ~96K words    |
| Gemini 1.5 Pro   | 2M tokens      | ~1.5M words   |

> **Beginner Note:** A token is roughly 3/4 of a word in English. "Hello, world!" is about 4 tokens. Code tends to use more tokens per "word" because of punctuation, indentation, and special characters.

### Why Context Windows Matter for Conversations

A typical user message might be 50-200 tokens. An assistant response might be 100-500 tokens. A system prompt might be 200-1000 tokens. At those rates:

- A 4K token window: ~8-15 exchanges before you run out
- A 128K token window: ~250-500 exchanges
- A 200K token window: ~400-800 exchanges

These numbers seem generous until you factor in long messages, code blocks, tool call results, or pasted documents. In practice, even large context windows fill up faster than you expect.

```typescript
// Rough estimation of conversation token usage
interface TokenEstimate {
  systemPrompt: number
  averageUserMessage: number
  averageAssistantMessage: number
  maxResponseTokens: number
}

function estimateMaxTurns(contextWindow: number, estimate: TokenEstimate): number {
  const availableForHistory = contextWindow - estimate.systemPrompt - estimate.maxResponseTokens
  const tokensPerTurn = estimate.averageUserMessage + estimate.averageAssistantMessage
  return Math.floor(availableForHistory / tokensPerTurn)
}

const claude35Sonnet = estimateMaxTurns(200_000, {
  systemPrompt: 500,
  averageUserMessage: 100,
  averageAssistantMessage: 300,
  maxResponseTokens: 4096,
})

console.log(`Estimated max turns for Claude Sonnet 4: ${claude35Sonnet}`)
// ~488 turns — but real conversations are rarely this uniform
```

> **Advanced Note:** The context window includes both input and output tokens. When you set `maxOutputTokens` for the response, that reduces the space available for your input. A 200K context window with `maxOutputTokens: 8192` leaves 191,808 tokens for your messages.

---

## Section 3: Message History Management

### The Naive Approach and Its Problems

The simplest memory strategy is to keep every message forever:

```typescript
// Naive: keep everything
const allMessages: Message[] = [{ role: 'system', content: systemPrompt }]

function addExchange(userMsg: string, assistantMsg: string): void {
  allMessages.push({ role: 'user', content: userMsg })
  allMessages.push({ role: 'assistant', content: assistantMsg })
}
```

This works until it does not. Problems:

1. **Token overflow**: Eventually the conversation exceeds the context window
2. **Cost escalation**: You pay for every token sent, even irrelevant early messages
3. **Latency increase**: More tokens means slower time-to-first-token
4. **Relevance dilution**: Old, irrelevant messages can confuse the model

### The Core Challenge

Memory management is about answering one question: **which messages should the model see right now?** The answer depends on your application:

- A customer support bot needs to remember the current ticket but not last week's conversation
- A coding assistant needs the current file context but not the debugging session from an hour ago
- A tutoring system needs to track what concepts the student has mastered across sessions

### Building a Message Manager

Let us create a base class for message management:

```typescript
import { generateText, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface ConversationConfig {
  systemPrompt: string
  model?: LanguageModel
  maxContextTokens?: number
}

class ConversationManager {
  protected messages: Message[] = []
  protected systemPrompt: string
  protected model: LanguageModel
  protected maxContextTokens: number

  constructor(config: ConversationConfig) {
    this.systemPrompt = config.systemPrompt
    this.model = config.model ?? mistral('mistral-small-latest')
    this.maxContextTokens = config.maxContextTokens ?? 180_000
  }

  /** Add a user message and get the assistant's response */
  async send(userMessage: string): Promise<string> {
    this.messages.push({ role: 'user', content: userMessage })

    const contextMessages = this.buildContext()

    const { text } = await generateText({
      model: this.model,
      messages: contextMessages,
    })

    this.messages.push({ role: 'assistant', content: text })
    return text
  }

  /** Override this in subclasses to implement different memory strategies */
  protected buildContext(): Message[] {
    return [{ role: 'system', content: this.systemPrompt }, ...this.messages]
  }

  /** Get full message history */
  getHistory(): Message[] {
    return [...this.messages]
  }

  /** Get message count (excluding system prompt) */
  getMessageCount(): number {
    return this.messages.length
  }
}
```

This base class stores all messages and passes them all to the model. The subclasses we build in the following sections will override `buildContext()` to implement smarter strategies.

---

## Section 4: Sliding Window Strategy

### The Idea

Keep only the most recent N messages. Older messages are simply dropped. This is the simplest strategy that actually works in production.

```
Full history:   [S] [U1] [A1] [U2] [A2] [U3] [A3] [U4] [A4] [U5] [A5]
Window (N=6):   [S]                             [U3] [A3] [U4] [A4] [U5] [A5]
```

The system prompt (S) is always included. The window slides forward as new messages arrive.

### Implementation

```typescript
import { generateText, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

class SlidingWindowConversation {
  private messages: Message[] = []
  private systemPrompt: string
  private windowSize: number
  private model: LanguageModel

  constructor(config: {
    systemPrompt: string
    windowSize?: number // Number of recent messages to keep
    model?: LanguageModel
  }) {
    this.systemPrompt = config.systemPrompt
    this.windowSize = config.windowSize ?? 20
    this.model = config.model ?? mistral('mistral-small-latest')
  }

  async send(userMessage: string): Promise<string> {
    this.messages.push({ role: 'user', content: userMessage })

    const contextMessages = this.buildContext()

    const { text } = await generateText({
      model: this.model,
      messages: contextMessages,
    })

    this.messages.push({ role: 'assistant', content: text })
    return text
  }

  private buildContext(): Message[] {
    // Always include the system prompt
    const systemMsg: Message = { role: 'system', content: this.systemPrompt }

    // Take only the last N messages
    const recentMessages = this.messages.slice(-this.windowSize)

    // Ensure we start with a user message (not an orphaned assistant message)
    const startIndex = recentMessages.findIndex(m => m.role === 'user')
    const trimmed = startIndex > 0 ? recentMessages.slice(startIndex) : recentMessages

    return [systemMsg, ...trimmed]
  }

  /** Get stats about the current window */
  getStats(): { totalMessages: number; windowedMessages: number } {
    const windowed = Math.min(this.messages.length, this.windowSize)
    return {
      totalMessages: this.messages.length,
      windowedMessages: windowed,
    }
  }
}

// Usage
const chat = new SlidingWindowConversation({
  systemPrompt: 'You are a helpful assistant. Be concise.',
  windowSize: 10, // Keep last 10 messages (5 exchanges)
})

const response1 = await chat.send('My name is Alice.')
console.log('Response 1:', response1)

const response2 = await chat.send('What is 2 + 2?')
console.log('Response 2:', response2)

const response3 = await chat.send('What is my name?')
console.log('Response 3:', response3)
// With a large enough window, it remembers "Alice"

console.log('Stats:', chat.getStats())
```

### Choosing the Window Size

The right window size depends on your use case:

| Use Case         | Suggested Window | Rationale                                |
| ---------------- | ---------------- | ---------------------------------------- |
| Quick Q&A bot    | 4-6 messages     | Users ask independent questions          |
| Customer support | 20-30 messages   | Need to track issue context              |
| Coding assistant | 30-50 messages   | Need to track file changes and decisions |
| Tutoring system  | 40-60 messages   | Need to track learning progression       |

> **Beginner Note:** A "window size" of 20 messages means roughly 10 exchanges (user + assistant pairs). For most chatbots, 10-20 exchanges of context is sufficient.

### Trade-offs

**Pros:**

- Simple to implement and understand
- Predictable memory usage and cost
- Fast — no extra LLM calls needed

**Cons:**

- Abrupt information loss — the model suddenly forgets everything before the window
- No way to recall important early context (like the user's name)
- Window size is a fixed compromise between cost and context

---

## Section 5: Summarization Strategy

### The Idea

Instead of dropping old messages, use the LLM itself to summarize them. The summary replaces the detailed history, preserving key information in a compressed form.

```
Full history:   [S] [U1] [A1] [U2] [A2] [U3] [A3] [U4] [A4]
After summary:  [S] [Summary of U1-A2] [U3] [A3] [U4] [A4]
```

### Implementation

```typescript
import { generateText, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

class SummarizingConversation {
  private messages: Message[] = []
  private summary: string | null = null
  private systemPrompt: string
  private model: LanguageModel
  private summarizeThreshold: number // Summarize when messages exceed this count
  private keepRecent: number // Number of recent messages to keep unsummarized

  constructor(config: {
    systemPrompt: string
    model?: LanguageModel
    summarizeThreshold?: number
    keepRecent?: number
  }) {
    this.systemPrompt = config.systemPrompt
    this.model = config.model ?? mistral('mistral-small-latest')
    this.summarizeThreshold = config.summarizeThreshold ?? 20
    this.keepRecent = config.keepRecent ?? 6
  }

  async send(userMessage: string): Promise<string> {
    this.messages.push({ role: 'user', content: userMessage })

    // Check if we need to summarize
    if (this.messages.length > this.summarizeThreshold) {
      await this.summarizeOlderMessages()
    }

    const contextMessages = this.buildContext()

    const { text } = await generateText({
      model: this.model,
      messages: contextMessages,
    })

    this.messages.push({ role: 'assistant', content: text })
    return text
  }

  private async summarizeOlderMessages(): Promise<void> {
    // Messages to summarize: everything except the most recent ones
    const toSummarize = this.messages.slice(0, -this.keepRecent)

    if (toSummarize.length === 0) return

    // Build the text to summarize
    const conversationText = toSummarize.map(m => `${m.role}: ${m.content}`).join('\n')

    const existingSummaryContext = this.summary ? `Previous summary: ${this.summary}\n\n` : ''

    const { text: newSummary } = await generateText({
      model: this.model,
      messages: [
        {
          role: 'system',
          content: `You are a conversation summarizer. Create a concise summary that preserves:
- Key facts mentioned (names, numbers, preferences)
- Decisions made
- Current topic and context
- Any unresolved questions

Be concise but complete. This summary will be used as context for continuing the conversation.`,
        },
        {
          role: 'user',
          content: `${existingSummaryContext}Summarize this conversation:\n\n${conversationText}`,
        },
      ],
    })

    this.summary = newSummary

    // Keep only the recent messages
    this.messages = this.messages.slice(-this.keepRecent)

    console.log(`[Memory] Summarized ${toSummarize.length} messages`)
    console.log(`[Memory] Summary: ${this.summary.slice(0, 100)}...`)
  }

  private buildContext(): Message[] {
    const contextMessages: Message[] = [{ role: 'system', content: this.systemPrompt }]

    // Include summary as a system message if it exists
    if (this.summary) {
      contextMessages.push({
        role: 'system',
        content: `Previous conversation summary: ${this.summary}`,
      })
    }

    // Include recent messages
    contextMessages.push(...this.messages)

    return contextMessages
  }

  /** Get current memory state */
  getMemoryState(): {
    hasSummary: boolean
    summaryLength: number
    recentMessageCount: number
  } {
    return {
      hasSummary: this.summary !== null,
      summaryLength: this.summary?.length ?? 0,
      recentMessageCount: this.messages.length,
    }
  }
}

// Usage
const chat = new SummarizingConversation({
  systemPrompt: 'You are a project planning assistant.',
  summarizeThreshold: 10, // Summarize after 10 messages
  keepRecent: 4, // Keep last 4 messages unsummarized
})

// Simulate a long conversation
const exchanges = [
  'My name is Bob and I am planning a web application.',
  'It should be an e-commerce store for handmade crafts.',
  'The tech stack will be Next.js with PostgreSQL.',
  'We need user authentication and a shopping cart.',
  'The budget is $50,000 and the timeline is 3 months.',
  'Let us start by defining the database schema.',
  'What tables do we need for the product catalog?',
]

for (const msg of exchanges) {
  console.log(`\nUser: ${msg}`)
  const response = await chat.send(msg)
  console.log(`Assistant: ${response.slice(0, 150)}...`)
}

console.log('\nMemory state:', chat.getMemoryState())
```

### The Summarization Prompt Matters

The quality of your summary prompt directly affects how well the model maintains context. Here are patterns that work:

```typescript
// Good: Specific about what to preserve
const goodSummaryPrompt = `Summarize preserving:
1. All named entities (people, products, companies)
2. Numerical values (dates, amounts, quantities)
3. Decisions and agreements
4. Current task or topic
5. Open questions or pending items`

// Bad: Too vague
const badSummaryPrompt = `Summarize this conversation briefly.`
// This loses critical details
```

### Trade-offs

**Pros:**

- Preserves key information from the entire conversation
- Gracefully compresses long histories
- Can maintain context across very long sessions

**Cons:**

- Extra LLM call for each summarization (cost and latency)
- Summary quality depends on the summarization prompt
- Some detail is inevitably lost in compression
- Potential for "summary drift" where errors compound over multiple summarizations

> **Advanced Note:** Summary drift is a real problem in long conversations. Each summarization round can subtly distort facts. Consider periodically including a "fact sheet" of immutable facts (user name, project name, key decisions) that persists independently of the rolling summary.

---

## Section 6: Hybrid Approaches

### Combining Strategies

The most robust production systems combine multiple strategies. A common pattern:

1. **System prompt** with static instructions (always present)
2. **Fact sheet** with key persistent facts (always present)
3. **Rolling summary** of older conversation (updated periodically)
4. **Sliding window** of recent messages (last N turns)

```typescript
import { generateText, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface Fact {
  key: string
  value: string
  timestamp: number
}

class HybridConversation {
  private messages: Message[] = []
  private summary: string | null = null
  private facts: Map<string, Fact> = new Map()
  private systemPrompt: string
  private model: LanguageModel
  private windowSize: number
  private summarizeThreshold: number

  constructor(config: {
    systemPrompt: string
    model?: LanguageModel
    windowSize?: number
    summarizeThreshold?: number
  }) {
    this.systemPrompt = config.systemPrompt
    this.model = config.model ?? mistral('mistral-small-latest')
    this.windowSize = config.windowSize ?? 10
    this.summarizeThreshold = config.summarizeThreshold ?? 20
  }

  /** Store a persistent fact that survives summarization */
  setFact(key: string, value: string): void {
    this.facts.set(key, { key, value, timestamp: Date.now() })
  }

  /** Remove a persistent fact */
  removeFact(key: string): void {
    this.facts.delete(key)
  }

  async send(userMessage: string): Promise<string> {
    this.messages.push({ role: 'user', content: userMessage })

    // Summarize if history is getting long
    if (this.messages.length > this.summarizeThreshold) {
      await this.summarizeOlderMessages()
    }

    // Extract facts from the conversation (simplified — could use LLM for this)
    this.extractFacts(userMessage)

    const contextMessages = this.buildContext()

    const { text } = await generateText({
      model: this.model,
      messages: contextMessages,
    })

    this.messages.push({ role: 'assistant', content: text })
    return text
  }

  private extractFacts(message: string): void {
    // Simple pattern-based extraction (in production, use LLM for this)
    const nameMatch = message.match(/my name is (\w+)/i)
    if (nameMatch) {
      this.setFact('user_name', nameMatch[1])
    }

    const emailMatch = message.match(/my email is ([\w@.]+)/i)
    if (emailMatch) {
      this.setFact('user_email', emailMatch[1])
    }
  }

  private buildContext(): Message[] {
    const parts: Message[] = []

    // 1. System prompt (always first)
    parts.push({ role: 'system', content: this.systemPrompt })

    // 2. Fact sheet (if facts exist)
    if (this.facts.size > 0) {
      const factSheet = Array.from(this.facts.values())
        .map(f => `- ${f.key}: ${f.value}`)
        .join('\n')
      parts.push({
        role: 'system',
        content: `Known facts about the user:\n${factSheet}`,
      })
    }

    // 3. Rolling summary (if exists)
    if (this.summary) {
      parts.push({
        role: 'system',
        content: `Summary of earlier conversation:\n${this.summary}`,
      })
    }

    // 4. Recent messages (sliding window)
    const recentMessages = this.messages.slice(-this.windowSize)
    const startIndex = recentMessages.findIndex(m => m.role === 'user')
    const trimmed = startIndex > 0 ? recentMessages.slice(startIndex) : recentMessages
    parts.push(...trimmed)

    return parts
  }

  private async summarizeOlderMessages(): Promise<void> {
    const toSummarize = this.messages.slice(0, -this.windowSize)
    if (toSummarize.length === 0) return

    const conversationText = toSummarize.map(m => `${m.role}: ${m.content}`).join('\n')

    const existingSummary = this.summary ? `Previous summary:\n${this.summary}\n\n` : ''

    const { text: newSummary } = await generateText({
      model: this.model,
      messages: [
        {
          role: 'system',
          content: `Summarize this conversation. Preserve key facts, decisions, and context. Be concise.`,
        },
        {
          role: 'user',
          content: `${existingSummary}New messages to incorporate:\n\n${conversationText}`,
        },
      ],
    })

    this.summary = newSummary
    this.messages = this.messages.slice(-this.windowSize)
  }

  /** Get a full diagnostic of the memory state */
  diagnose(): {
    factCount: number
    facts: Record<string, string>
    hasSummary: boolean
    summaryPreview: string
    totalMessages: number
    windowedMessages: number
  } {
    const factObj: Record<string, string> = {}
    for (const [key, fact] of this.facts) {
      factObj[key] = fact.value
    }

    return {
      factCount: this.facts.size,
      facts: factObj,
      hasSummary: this.summary !== null,
      summaryPreview: this.summary?.slice(0, 100) ?? '',
      totalMessages: this.messages.length,
      windowedMessages: Math.min(this.messages.length, this.windowSize),
    }
  }
}

// Usage
const chat = new HybridConversation({
  systemPrompt: 'You are a personal assistant. Remember facts about the user and reference them naturally.',
  windowSize: 8,
  summarizeThreshold: 16,
})

// Pre-load some facts
chat.setFact('timezone', 'PST')
chat.setFact('preferred_language', 'TypeScript')

const r1 = await chat.send('My name is Sarah. I am working on a React project.')
console.log('Response:', r1)

const r2 = await chat.send('Can you help me with state management?')
console.log('Response:', r2)

console.log('\nDiagnosis:', chat.diagnose())
```

### When to Use Each Strategy

| Strategy       | Best For                     | Avoid When                          |
| -------------- | ---------------------------- | ----------------------------------- |
| Sliding window | Simple chatbots, Q&A         | Long-term context matters           |
| Summarization  | Long sessions, complex tasks | Low latency required                |
| Hybrid         | Production assistants        | Simple use cases (over-engineering) |
| Full history   | Short conversations          | Budget/latency sensitive            |

> **Advanced Note:** Some production systems use a "tiered memory" approach: hot memory (recent messages), warm memory (summaries), and cold memory (searchable archive stored in a vector database). Module 8 (Embeddings) and Module 9 (RAG) provide the tools to build the cold tier.

---

## Section 7: Conversation Persistence

### Why Persist Conversations?

In-memory conversations are lost when the process restarts. For production applications, you need to save conversations to disk or a database so users can:

- Continue conversations after closing the app
- Access conversation history across devices
- Resume sessions after server restarts

### Saving and Loading from Disk

```typescript
import { generateText, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface ConversationData {
  id: string
  systemPrompt: string
  messages: Message[]
  summary: string | null
  facts: Record<string, string>
  createdAt: string
  updatedAt: string
}

class PersistentConversation {
  private data: ConversationData
  private savePath: string
  private model: LanguageModel

  constructor(config: { id?: string; systemPrompt: string; saveDirectory?: string; model?: LanguageModel }) {
    const id = config.id ?? crypto.randomUUID()
    const saveDir = config.saveDirectory ?? './conversations'

    this.savePath = `${saveDir}/${id}.json`
    this.model = config.model ?? mistral('mistral-small-latest')

    this.data = {
      id,
      systemPrompt: config.systemPrompt,
      messages: [],
      summary: null,
      facts: {},
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }
  }

  /** Load a conversation from disk */
  static async load(filePath: string): Promise<PersistentConversation> {
    const file = Bun.file(filePath)
    const exists = await file.exists()

    if (!exists) {
      throw new Error(`Conversation file not found: ${filePath}`)
    }

    const data: ConversationData = await file.json()

    const conversation = new PersistentConversation({
      id: data.id,
      systemPrompt: data.systemPrompt,
    })
    conversation.data = data
    conversation.savePath = filePath

    return conversation
  }

  /** Save the conversation to disk */
  async save(): Promise<void> {
    this.data.updatedAt = new Date().toISOString()

    // Ensure directory exists
    const dir = this.savePath.substring(0, this.savePath.lastIndexOf('/'))
    await Bun.write(this.savePath, JSON.stringify(this.data, null, 2))

    console.log(`[Persistence] Saved to ${this.savePath}`)
  }

  /** Send a message and auto-save */
  async send(userMessage: string): Promise<string> {
    this.data.messages.push({ role: 'user', content: userMessage })

    const { text } = await generateText({
      model: this.model,
      messages: [{ role: 'system', content: this.data.systemPrompt }, ...this.data.messages],
    })

    this.data.messages.push({ role: 'assistant', content: text })

    // Auto-save after each exchange
    await this.save()

    return text
  }

  /** Get conversation metadata */
  getMetadata(): {
    id: string
    messageCount: number
    createdAt: string
    updatedAt: string
  } {
    return {
      id: this.data.id,
      messageCount: this.data.messages.length,
      createdAt: this.data.createdAt,
      updatedAt: this.data.updatedAt,
    }
  }
}

// Usage: Create a new conversation
const chat = new PersistentConversation({
  systemPrompt: 'You are a helpful assistant.',
  saveDirectory: './conversations',
})

await chat.send('Hello! My name is Alex.')
await chat.send('What is the capital of France?')

console.log('Metadata:', chat.getMetadata())

// Later: Load the conversation back
// const loaded = await PersistentConversation.load('./conversations/<id>.json');
// const response = await loaded.send('Do you remember my name?');
```

### Listing and Managing Conversations

```typescript
import { readdir, unlink } from 'node:fs/promises'

interface ConversationSummary {
  id: string
  filePath: string
  createdAt: string
  updatedAt: string
  messageCount: number
}

async function listConversations(directory: string): Promise<ConversationSummary[]> {
  const files = await readdir(directory)
  const jsonFiles = files.filter(f => f.endsWith('.json'))

  const summaries: ConversationSummary[] = []

  for (const file of jsonFiles) {
    const filePath = `${directory}/${file}`
    const data = await Bun.file(filePath).json()

    summaries.push({
      id: data.id,
      filePath,
      createdAt: data.createdAt,
      updatedAt: data.updatedAt,
      messageCount: data.messages.length,
    })
  }

  // Sort by most recently updated
  summaries.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())

  return summaries
}

async function deleteConversation(filePath: string): Promise<void> {
  await unlink(filePath)
  console.log(`Deleted conversation: ${filePath}`)
}

// Usage
const conversations = await listConversations('./conversations')
for (const conv of conversations) {
  console.log(`${conv.id} - ${conv.messageCount} messages - Updated: ${conv.updatedAt}`)
}
```

> **Beginner Note:** For production applications, you would typically use a database (PostgreSQL, SQLite, Redis) instead of JSON files. The pattern is the same — serialize the conversation data and store it with a unique ID. JSON files work well for development and small-scale applications.

> **Advanced Note:** When persisting conversations, consider GDPR and data retention requirements. Users may have the right to delete their conversation history. Build deletion into your persistence layer from the start.

---

## Section 8: Token Counting

### Why Count Tokens?

Token counting serves two purposes:

1. **Preventing context overflow**: Know when you are approaching the limit before the API returns an error
2. **Cost estimation**: Predict how much a conversation will cost before making the call

### Estimating Tokens

Exact token counting requires the model's actual tokenizer. For estimation, a simple heuristic works well:

```typescript
/**
 * Estimate token count for a string.
 * Rule of thumb: 1 token ≈ 4 characters in English.
 * This is an approximation — actual counts vary by model and content.
 */
function estimateTokens(text: string): number {
  // Simple character-based estimation
  return Math.ceil(text.length / 4)
}

/**
 * More accurate estimation using word-based heuristic.
 * English text averages ~1.3 tokens per word.
 * Code averages ~2 tokens per word due to punctuation.
 */
function estimateTokensAccurate(text: string, isCode: boolean = false): number {
  const words = text.split(/\s+/).filter(Boolean).length
  const multiplier = isCode ? 2.0 : 1.3
  return Math.ceil(words * multiplier)
}

// Test with known examples
console.log(estimateTokens('Hello, world!')) // ~4 (actual: ~4)
console.log(estimateTokens('The quick brown fox jumps over the lazy dog.'))
// ~12 (actual: ~10)
```

### Token Counting for Message Arrays

```typescript
interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface TokenBudget {
  contextWindow: number
  maxResponseTokens: number
  reservedForSystem: number
}

function countMessageTokens(messages: Message[]): number {
  let total = 0

  for (const message of messages) {
    // Each message has overhead for role formatting (~4 tokens)
    total += 4
    total += estimateTokens(message.content)
  }

  // Conversation framing overhead (~3 tokens)
  total += 3

  return total
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

function getRemainingBudget(
  messages: Message[],
  budget: TokenBudget
): {
  used: number
  remaining: number
  percentUsed: number
  canFitMoreMessages: boolean
} {
  const used = countMessageTokens(messages)
  const availableForMessages = budget.contextWindow - budget.maxResponseTokens - budget.reservedForSystem
  const remaining = availableForMessages - used

  return {
    used,
    remaining,
    percentUsed: (used / availableForMessages) * 100,
    canFitMoreMessages: remaining > 200, // At least 200 tokens for a new message
  }
}

// Usage
const messages: Message[] = [
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user', content: 'Explain quantum computing in simple terms.' },
  {
    role: 'assistant',
    content: 'Quantum computing uses quantum bits (qubits) that can be 0, 1, or both at once...',
  },
  { role: 'user', content: 'How is that useful practically?' },
]

const budget: TokenBudget = {
  contextWindow: 200_000, // Claude Sonnet 4
  maxResponseTokens: 4096,
  reservedForSystem: 500,
}

const status = getRemainingBudget(messages, budget)
console.log(`Tokens used: ${status.used}`)
console.log(`Tokens remaining: ${status.remaining}`)
console.log(`Percent used: ${status.percentUsed.toFixed(2)}%`)
console.log(`Can fit more: ${status.canFitMoreMessages}`)
```

### Token-Aware Sliding Window

Combine token counting with the sliding window to trim by token count rather than message count:

```typescript
import { generateText, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

function messageTokens(msg: Message): number {
  return 4 + estimateTokens(msg.content)
}

class TokenAwareConversation {
  private messages: Message[] = []
  private systemPrompt: string
  private model: LanguageModel
  private maxInputTokens: number
  private maxResponseTokens: number

  constructor(config: {
    systemPrompt: string
    model?: LanguageModel
    maxInputTokens?: number
    maxResponseTokens?: number
  }) {
    this.systemPrompt = config.systemPrompt
    this.model = config.model ?? mistral('mistral-small-latest')
    this.maxInputTokens = config.maxInputTokens ?? 180_000
    this.maxResponseTokens = config.maxResponseTokens ?? 4096
  }

  async send(userMessage: string): Promise<string> {
    this.messages.push({ role: 'user', content: userMessage })

    const contextMessages = this.buildContext()

    const { text, usage } = await generateText({
      model: this.model,
      messages: contextMessages,
      maxOutputTokens: this.maxResponseTokens,
    })

    this.messages.push({ role: 'assistant', content: text })

    // Log actual token usage for calibration
    if (usage) {
      console.log(`[Tokens] Input: ${usage.inputTokens}, Output: ${usage.outputTokens}`)
    }

    return text
  }

  private buildContext(): Message[] {
    const systemMsg: Message = {
      role: 'system',
      content: this.systemPrompt,
    }
    const systemTokens = messageTokens(systemMsg)
    const budgetForMessages = this.maxInputTokens - systemTokens - this.maxResponseTokens

    // Start from the most recent messages and work backward
    const selected: Message[] = []
    let usedTokens = 0

    for (let i = this.messages.length - 1; i >= 0; i--) {
      const msgTokens = messageTokens(this.messages[i])

      if (usedTokens + msgTokens > budgetForMessages) {
        break // No more room
      }

      selected.unshift(this.messages[i])
      usedTokens += msgTokens
    }

    // Ensure we start with a user message
    while (selected.length > 0 && selected[0].role === 'assistant') {
      selected.shift()
    }

    console.log(`[Context] Using ${selected.length}/${this.messages.length} messages (${usedTokens} tokens)`)

    return [systemMsg, ...selected]
  }

  /** Get token usage statistics */
  getTokenStats(): {
    totalHistoryTokens: number
    messageCount: number
    averageTokensPerMessage: number
  } {
    const total = this.messages.reduce((sum, m) => sum + messageTokens(m), 0)
    return {
      totalHistoryTokens: total,
      messageCount: this.messages.length,
      averageTokensPerMessage: this.messages.length > 0 ? total / this.messages.length : 0,
    }
  }
}

// Usage
const chat = new TokenAwareConversation({
  systemPrompt: 'You are a technical assistant for TypeScript developers.',
  maxInputTokens: 10_000, // Low limit for demonstration
  maxResponseTokens: 1024,
})

const questions = [
  'What is a generic type in TypeScript?',
  'Show me an example with arrays.',
  'How do conditional types work?',
  'What about mapped types?',
  'Can you combine them?',
]

for (const q of questions) {
  console.log(`\nUser: ${q}`)
  const response = await chat.send(q)
  console.log(`Assistant: ${response.slice(0, 100)}...`)
}

console.log('\nToken stats:', chat.getTokenStats())
```

### Using Actual Token Counts from the API

The Vercel AI SDK returns actual token usage in the response:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const { text, usage } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is TypeScript?',
})

if (usage) {
  console.log(`Input tokens: ${usage.inputTokens}`)
  console.log(`Output tokens: ${usage.outputTokens}`)
  console.log(`Total tokens: ${usage.totalTokens}`)

  // Calculate cost (example pricing — check current rates)
  const inputCostPer1M = 3.0 // $3 per 1M input tokens for Claude Sonnet
  const outputCostPer1M = 15.0 // $15 per 1M output tokens
  const inputCost = (usage.inputTokens / 1_000_000) * inputCostPer1M
  const outputCost = (usage.outputTokens / 1_000_000) * outputCostPer1M
  const totalCost = inputCost + outputCost

  console.log(`Estimated cost: $${totalCost.toFixed(6)}`)
}
```

> **Beginner Note:** Token counts from the API are exact. Your estimates will always be approximate. Use estimates for pre-flight checks (will this fit?) and API counts for post-flight accounting (what did this cost?).

> **Advanced Note:** For precise pre-flight token counting, you can use the `js-tiktoken` library (works for most tokenizers) or provider-specific token counting APIs. However, the character-based estimation (length/4) is accurate enough for most memory management decisions.

> **Local Alternative (Ollama):** All conversation patterns in this module work with `ollama('qwen3.5')`. Multi-turn conversations, sliding windows, and summarization strategies are model-agnostic — they manage the message array before sending it to any provider. Memory management is especially important with local models, which typically have smaller context windows (8K-32K tokens vs 200K for Claude).

---

## Summary

In this module, you learned:

1. **Multi-turn conversations:** LLMs are stateless — your application must manage the message array and send the full conversation history with every request.
2. **Context windows:** Each model has a finite token limit, and every message you send consumes tokens, creating a hard ceiling on conversation length.
3. **Message history management:** Naive approaches of sending all messages work for short conversations but break down as conversations grow in length and cost.
4. **Sliding window strategy:** Keeping only the most recent N messages is simple and predictable but loses early context that may be important.
5. **Summarization strategy:** Using the LLM to compress older messages into a running summary preserves key context while staying within token limits.
6. **Hybrid approaches:** Combining a summary of old messages with a sliding window of recent messages gives you the best balance of context retention and cost control.
7. **Conversation persistence:** Saving conversation history to disk enables long-running sessions that survive restarts and can be resumed later.
8. **Token estimation:** Character-based estimation (length/4) provides a practical way to manage token budgets without external libraries.

In Module 5, you will explore what happens when context windows are very large and how prompt caching can dramatically reduce costs for repeated context.

---

## Quiz

### Question 1 (Easy)

In LLM conversations, who is responsible for maintaining the message history between turns?

A) The LLM provider's API server
B) The client application (your code)
C) The model itself has built-in memory
D) The Vercel AI SDK automatically persists messages

**Answer: B**

LLMs are stateless. They do not remember previous calls. The client application must accumulate messages into an array and send the full conversation context with each request. Neither the API server nor the SDK maintains conversation state automatically.

---

### Question 2 (Medium)

A sliding window strategy with windowSize=10 will keep approximately how many complete user-assistant exchanges?

A) 10 exchanges
B) 5 exchanges
C) 20 exchanges
D) It depends on the token count

**Answer: B**

A window size of 10 messages means 10 individual messages. Since each exchange consists of a user message and an assistant message (2 messages), 10 messages equals approximately 5 complete exchanges.

---

### Question 3 (Medium)

What is the main disadvantage of the summarization memory strategy compared to sliding window?

A) It cannot preserve any context from older messages
B) It requires extra LLM calls, adding cost and latency
C) It uses more tokens than keeping all messages
D) It only works with OpenAI models

**Answer: B**

The summarization strategy requires an additional LLM call each time it compresses older messages into a summary. This adds both monetary cost and latency to the conversation. However, it compensates by preserving key information that sliding window would discard.

---

### Question 4 (Hard)

When building a token-aware context window, why do you process messages from newest to oldest?

A) Newer messages are always shorter
B) The model weights recent messages more heavily in its attention
C) It ensures the most relevant (recent) context is included if you run out of space
D) It is required by the Mistral API

**Answer: C**

Processing from newest to oldest ensures that the most recent messages (which are almost always the most relevant to the current query) are included first. If the token budget is exceeded, it is the oldest and least relevant messages that get dropped.

---

### Question 5 (Easy)

In a hybrid memory approach, what is the purpose of the "fact sheet" component?

A) To replace the system prompt
B) To store immutable key facts that should survive summarization without distortion
C) To count tokens accurately
D) To format messages for the API

**Answer: B**

The fact sheet stores critical persistent information (like the user's name, project details, or key decisions) in a structured format that is always included in the context. This prevents "summary drift" where important details get distorted or lost through repeated summarization cycles.

---

## Exercises

### Exercise 1: Configurable Memory Strategy Chatbot

Build a command-line chatbot that supports all three memory strategies (sliding window, summarization, hybrid) and can switch between them during a conversation.

**Requirements:**

1. Accept a `--strategy` flag: `window`, `summary`, or `hybrid`
2. Support a `--window-size` flag (default: 20)
3. Support a `--summarize-threshold` flag (default: 30)
4. Display token usage after each exchange
5. Support a `/stats` command that shows current memory state
6. Support a `/strategy <name>` command to switch strategies mid-conversation
7. Support a `/history` command that shows the raw message array
8. Persist the conversation to a JSON file, with auto-save after each turn

**Example usage:**

```bash
bun run chatbot.ts --strategy hybrid --window-size 10
```

**Starter code:**

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { parseArgs } from 'node:util'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

type StrategyName = 'window' | 'summary' | 'hybrid'

interface MemoryStrategy {
  name: StrategyName
  buildContext(systemPrompt: string, allMessages: Message[], options: StrategyOptions): Promise<Message[]>
}

interface StrategyOptions {
  windowSize: number
  summarizeThreshold: number
  currentSummary: string | null
  model: string
}

// Implement the three strategies
const windowStrategy: MemoryStrategy = {
  name: 'window',
  async buildContext(systemPrompt, allMessages, options) {
    // TODO: Implement sliding window
    throw new Error('Not implemented')
  },
}

const summaryStrategy: MemoryStrategy = {
  name: 'summary',
  async buildContext(systemPrompt, allMessages, options) {
    // TODO: Implement summarization
    throw new Error('Not implemented')
  },
}

const hybridStrategy: MemoryStrategy = {
  name: 'hybrid',
  async buildContext(systemPrompt, allMessages, options) {
    // TODO: Implement hybrid approach
    throw new Error('Not implemented')
  },
}

// Parse CLI arguments and start the chatbot
// TODO: Implement the main loop
```

**Evaluation criteria:**

- All three strategies produce different context arrays for the same conversation
- `/stats` accurately reports memory state for each strategy
- Conversation persists across restarts
- Token counting is within 20% of actual API-reported usage

### Exercise 2: Memory Strategy Comparison

Write a script that runs the same 20-turn conversation through all three memory strategies and compares:

1. Total tokens sent across all turns
2. Total estimated cost
3. Whether the model can recall information from turn 1 at turn 20
4. Response quality (manual evaluation)

**Requirements:**

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface BenchmarkResult {
  strategy: string
  totalInputTokens: number
  totalOutputTokens: number
  estimatedCost: number
  recallAccuracy: number // 0-1: can the model recall facts from early turns?
  turns: number
}

async function benchmarkStrategy(strategyName: string, messages: string[]): Promise<BenchmarkResult> {
  // TODO: Run the conversation through the specified strategy
  // TODO: After all turns, ask recall questions about early facts
  // TODO: Calculate total tokens and cost
  throw new Error('Not implemented')
}

// The test conversation
const testMessages = [
  'My name is Jordan and I live in Portland.',
  'I work as a data engineer at a startup called DataFlow.',
  'We use Python and Apache Spark for our data pipelines.',
  'Our biggest challenge is handling 10TB of daily event data.',
  'Can you suggest an architecture for real-time processing?',
  // ... 15 more messages about the project
  // Final: 'What was my name and what company do I work for?'
]

// TODO: Run benchmarks and print comparison table
```

**Output should look like:**

```
Strategy    | Input Tokens | Output Tokens | Cost    | Recall
------------|-------------|---------------|---------|-------
window      | 12,450      | 8,200         | $0.045  | 0.4
summary     | 15,800      | 9,100         | $0.062  | 0.9
hybrid      | 14,200      | 8,800         | $0.055  | 0.95
```
