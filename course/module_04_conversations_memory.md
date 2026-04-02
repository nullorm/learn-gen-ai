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

The `generateText` function accepts a `messages` array. To build a multi-turn conversation, you need a helper that takes the message array and returns the model's response:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

async function chat(messages: Message[]): Promise<string>
```

This function should call `generateText` with the messages array and return the `text` from the result.

To run a multi-turn conversation, start with a `Message[]` containing a system prompt and the first user message. Call `chat`, then push the assistant's response back into the array with `role: 'assistant'`. When the user sends another message, push it with `role: 'user'` and call `chat` again. Each call sends the entire accumulated array, so the model sees the full history.

What happens if you forget to push the assistant's response back into the array before adding the next user message?

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

To build an interactive chatbot, you need a REPL (read-eval-print loop). The structure is:

```typescript
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
```

Build a function `getResponse(messages: Message[]): Promise<string>` that calls `generateText` with the messages array and returns the text.

Then build the REPL loop. In Bun, you can read stdin with `Bun.stdin.stream().getReader()` and decode chunks with `TextDecoder`. The loop should:

1. Print a prompt (`You: `)
2. Read a line of input
3. If the input is `"quit"`, break out of the loop
4. Push the user message into the conversation array
5. Call `getResponse` with the conversation
6. Push the assistant response into the conversation array
7. Print the response and loop back to step 1

After the loop ends, print the total message count. How would you handle the case where `reader.read()` returns `done: true`?

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

To reason about this concretely, define a `TokenEstimate` interface with fields for `systemPrompt`, `averageUserMessage`, `averageAssistantMessage`, and `maxResponseTokens` (all numbers representing token counts).

```typescript
interface TokenEstimate {
  systemPrompt: number
  averageUserMessage: number
  averageAssistantMessage: number
  maxResponseTokens: number
}

function estimateMaxTurns(contextWindow: number, estimate: TokenEstimate): number
```

This function should calculate the available tokens for history (context window minus system prompt minus max response tokens), then divide by the tokens per turn (average user + average assistant message) to get the maximum number of turns. What happens to the estimate when `maxResponseTokens` is very large?

> **Advanced Note:** The context window includes both input and output tokens. When you set `maxOutputTokens` for the response, that reduces the space available for your input. A 200K context window with `maxOutputTokens: 8192` leaves 191,808 tokens for your messages.

---

## Section 3: Conversation State & Strategy Pattern

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

Different use cases need different strategies. And you might want to switch strategies at runtime — start with a simple window, upgrade to hybrid when the conversation gets complex. This means the strategy should be swappable without losing the conversation.

### Separating State from Strategy

The key design insight: **conversation state and memory strategy are separate concerns.** The state holds all the data (history, summaries, facts). The strategy decides which parts of that data the model should see.

```typescript
import type { ModelMessage } from 'ai'

interface ConversationState {
  history: ModelMessage[]
  summarization?: {
    summary: string
    until: number
  }
  facts?: {
    entries: Record<string, string>
    extractedUntil: number
  }
}

interface MemoryStrategy {
  buildContext(systemPrompt: string, state: ConversationState): ModelMessage[]
}
```

`ConversationState` is a plain data object. The `history` array holds every message ever sent. The optional `summarization` and `facts` fields are populated only when a strategy that uses them is active. The `until` / `extractedUntil` fields track how far processing has gone, so strategies know which messages still need summarizing or fact extraction.

`MemoryStrategy` is a single function: given the system prompt and the full state, return the message array to send to the model.

### The Conversation Manager

The manager composes state and strategy. It owns the state, delegates context building to the strategy, and allows hot-swapping strategies without losing any data.

```typescript
import type { ModelMessage } from 'ai'

class ConversationManager {
  private state: ConversationState = { history: [] }
  private strategy: MemoryStrategy
  private systemPrompt: string

  constructor(config: { systemPrompt: string; strategy: MemoryStrategy })

  addUserMessage(content: string): void
  addAssistantMessage(content: string): void
  buildContext(): ModelMessage[]
  setStrategy(strategy: MemoryStrategy): void
  getState(): ConversationState
  setState(state: ConversationState): void
  getHistory(): ModelMessage[]
  getMessageCount(): number
}
```

Build this class. The constructor should store the system prompt and strategy from the config. `addUserMessage` and `addAssistantMessage` push messages into `state.history` with the appropriate role. `buildContext` delegates to `this.strategy.buildContext(this.systemPrompt, this.state)`. `setStrategy` replaces the strategy reference — the state is untouched. `getHistory` should return a copy of the history array (not a reference to it). `getMessageCount` returns the length of the history.

Why does `setStrategy` not need to do anything with the state? What would go wrong if `getHistory` returned a direct reference instead of a copy?

---

## Section 4: Sliding Window Strategy

### The Idea

Keep only the most recent N messages. Older messages are simply dropped from context (but stay in state). This is the simplest strategy that actually works in production.

```
Full history:   [S] [U1] [A1] [U2] [A2] [U3] [A3] [U4] [A4] [U5] [A5]
Window (N=6):   [S]                             [U3] [A3] [U4] [A4] [U5] [A5]
```

The system prompt (S) is always included. The window slides forward as new messages arrive.

### Implementation

The windowing logic — take the last N messages, ensure we start on a user message — is a reusable building block. Extract it as a helper, then wrap it in a strategy.

First, build the helper function:

```typescript
import type { ModelMessage } from 'ai'

function getWindowedMessages(history: ModelMessage[], windowSize: number): ModelMessage[]
```

This function should slice the last `windowSize` messages from the history. Then find the first `user` message in that slice — if it is not at position 0, trim everything before it. Why? Because if the window boundary falls in the middle of an exchange, you might start with an orphaned assistant message, which violates the alternation rules.

Then build the strategy factory:

```typescript
function createSlidingWindowStrategy(windowSize: number = 20): MemoryStrategy
```

This factory returns a `MemoryStrategy` whose `buildContext` prepends the system prompt as a `{ role: 'system', content: systemPrompt }` message, then appends the result of `getWindowedMessages(state.history, windowSize)`.

`getWindowedMessages` is a pure function — it takes history and a window size, returns trimmed messages. The strategy factory wraps it with the system prompt. Later, the hybrid strategy will reuse this same helper.

The strategy only reads `state.history` — it ignores `summarization` and `facts` entirely. Those fields remain untouched in the state, available if you later switch to a strategy that uses them.

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

Unlike the sliding window, the full history stays in `state.history` — the summary is stored alongside it in `state.summarization`. The strategy uses the summary + recent messages to build context.

### Implementation

Like the windowing logic, the summarization checks are reusable helpers. Build two pure functions:

```typescript
import type { ModelMessage } from 'ai'

function needsSummarization(state: ConversationState, threshold: number): boolean

function getSummarizationText(state: ConversationState, keepRecent: number): string
```

`needsSummarization` should calculate how many messages in the history have not been summarized yet. The unsummarized count is the total history length minus `state.summarization?.until ?? 0`. If that count exceeds the threshold, return `true`.

`getSummarizationText` should build the text to send to the LLM for summarization. It needs to gather messages from the last summary point (`state.summarization?.until ?? 0`) up to but not including the most recent `keepRecent` messages. If a previous summary exists, prepend it with a `"Previous summary: "` prefix so the LLM can build on it. Format each message as `"role: content"` joined by newlines.

Then build the strategy factory:

```typescript
function createSummarizingStrategy(config: { summarizeThreshold?: number; keepRecent?: number }): MemoryStrategy & {
  needsSummarization(state: ConversationState): boolean
  getSummarizationText(state: ConversationState): string
}
```

The returned object should have three methods. `buildContext` assembles the context: start with the system prompt, then if `state.summarization` exists add a system message with the summary text, then append all messages from `state.summarization?.until ?? 0` onward. `needsSummarization` and `getSummarizationText` delegate to the helpers with the config values.

The chatbot loop calls `needsSummarization()` each turn, and when it returns true, sends `getSummarizationText()` to the LLM and stores the result:

```typescript
// In the chatbot loop:
if (strategy.needsSummarization(state)) {
  const text = strategy.getSummarizationText(state)
  const { text: summary } = await generateText({
    model,
    system: 'Summarize this conversation. Preserve key facts, decisions, and context.',
    prompt: text,
  })
  state.summarization = { summary, until: state.history.length - keepRecent }
}
```

Notice: the full history is never truncated. `state.history` always has every message. `summarization.until` tells the strategy where the summary covers up to, so `buildContext` only includes messages _after_ that point.

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

> **Advanced Note:** Summary drift is a real problem in long conversations. Each summarization round can subtly distort facts. Consider periodically including a "fact sheet" of immutable facts (user name, project name, key decisions) that persists independently of the rolling summary. The hybrid strategy in the next section addresses this.

> **Advanced Note:** In this module, summarization is triggered by the chatbot loop — the caller checks `needsSummarization()` and makes the LLM call. A cleaner production pattern is to give strategies a `maintain(state, model)` method that handles their own state upkeep (summarization, fact extraction, etc.) internally. The caller just calls `strategy.maintain()` before `buildContext()` without knowing the details. We keep it explicit here so you can see exactly what happens, but consider the `maintain` pattern when building real applications.

---

## Section 6: Hybrid Approaches

### Combining Strategies

The most robust production systems combine multiple strategies. A common pattern:

1. **System prompt** with static instructions (always present)
2. **Fact sheet** with key persistent facts (always present)
3. **Rolling summary** of older conversation (updated periodically)
4. **Sliding window** of recent messages (last N turns)

The fact sheet is the key addition. Facts are key-value pairs (like `name: Jordan`, `company: DataFlow`) that are always injected into context. Unlike summaries, facts don't get compressed or distorted — they survive indefinitely.

### Implementation

The hybrid strategy composes the helpers from Sections 4 and 5. No duplicated logic — `getWindowedMessages` handles the window, `needsSummarization` and `getSummarizationText` handle the summary checks. The only new code is the fact sheet injection.

```typescript
import type { ModelMessage } from 'ai'

function createHybridStrategy(config: { windowSize?: number; summarizeThreshold?: number }): MemoryStrategy & {
  needsSummarization(state: ConversationState): boolean
  getSummarizationText(state: ConversationState): string
}
```

The factory should default `windowSize` to 10 and `summarizeThreshold` to 20. The returned object needs three methods:

**`buildContext`** assembles four layers in order:

1. System prompt as the first system message (always first)
2. Fact sheet — if `state.facts?.entries` has any keys, format them as `"- key: value"` lines and add as a system message with the prefix `"Known facts about the user:"`
3. Rolling summary — if `state.summarization` exists, add as a system message with the prefix `"Summary of earlier conversation:"`
4. Recent messages — call `getWindowedMessages(state.history, windowSize)` from Section 4

**`needsSummarization`** and **`getSummarizationText`** reuse the helpers from Section 5, passing `windowSize` as the `keepRecent` parameter to `getSummarizationText`.

How does the fact sheet survive summarization? Why does `getWindowedMessages` use `windowSize` instead of `keepRecent` here?

Usage looks like:

```typescript
const manager = new ConversationManager({
  systemPrompt: 'You are a personal assistant.',
  strategy: createHybridStrategy({ windowSize: 8, summarizeThreshold: 16 }),
})

// Pre-load some facts
const state = manager.getState()
state.facts = { entries: { timezone: 'PST', preferred_language: 'TypeScript' }, extractedUntil: 0 }
```

### Strategy Switching in Action

Because state is separate from strategy, switching is trivial:

```typescript
// Start with a simple window
const manager = new ConversationManager({
  systemPrompt: 'You are a helpful assistant.',
  strategy: createSlidingWindowStrategy(10),
})

// ... 50 turns later, upgrade to hybrid
manager.setStrategy(createHybridStrategy({ windowSize: 10, summarizeThreshold: 20 }))
// All 50 messages are still in state.history — hybrid can summarize them
```

No history copying, no state reconstruction, no data loss. The new strategy just reads the existing state differently.

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

### Saving and Loading ConversationState

Because `ConversationState` is a plain data object, persistence is straightforward — serialize it to JSON. Define the on-disk format and the save/load functions:

```typescript
interface SavedConversation {
  id: string
  systemPrompt: string
  state: ConversationState
  createdAt: string
  updatedAt: string
}

async function saveConversation(
  filePath: string,
  id: string,
  systemPrompt: string,
  state: ConversationState,
  createdAt?: string
): Promise<void>

async function loadConversation(filePath: string): Promise<SavedConversation>
```

`saveConversation` should assemble a `SavedConversation` object (using the current timestamp for `updatedAt`, and `createdAt` defaults to now if not provided), then write it to the file path as formatted JSON using `Bun.write` and `JSON.stringify(data, null, 2)`.

`loadConversation` should read the file using `Bun.file(filePath)`. Check if the file exists first — if not, throw an error with a descriptive message. If it exists, parse and return it as `SavedConversation`.

To resume a conversation, load the state and pass it to a new manager:

```typescript
// Save after each turn
const id = crypto.randomUUID()
const savePath = `./data/conversations/${id}.json`

await saveConversation(savePath, id, systemPrompt, manager.getState())

// Later: resume
const saved = await loadConversation(savePath)
const manager = new ConversationManager({
  systemPrompt: saved.systemPrompt,
  strategy: createSlidingWindowStrategy(10),
})
manager.setState(saved.state)
```

Because the state includes everything — history, summaries, facts — the restored conversation has full fidelity regardless of which strategy you use when resuming. You can even resume with a _different_ strategy than the one used when saving.

### Listing and Managing Conversations

Build two more utility functions:

```typescript
import { readdir, unlink } from 'node:fs/promises'

interface ConversationSummary {
  id: string
  filePath: string
  createdAt: string
  updatedAt: string
  messageCount: number
}

async function listConversations(directory: string): Promise<ConversationSummary[]>

async function deleteConversation(filePath: string): Promise<void>
```

`listConversations` should read all `.json` files from the directory using `readdir`, load each one with `Bun.file`, and build a `ConversationSummary` for each (pulling `messageCount` from `data.state.history.length`). Sort the results by `updatedAt` descending (most recent first).

`deleteConversation` should remove the file using `unlink`.

> **Beginner Note:** For production applications, you would typically use a database (PostgreSQL, SQLite, Redis) instead of JSON files. The pattern is the same — serialize the conversation data and store it with a unique ID. JSON files work well for development and small-scale applications.

> **Advanced Note:** When persisting conversations, consider GDPR and data retention requirements. Users may have the right to delete their conversation history. Build deletion into your persistence layer from the start.

---

## Section 8: Token Counting

### Why Count Tokens?

Token counting serves two purposes:

1. **Preventing context overflow**: Know when you are approaching the limit before the API returns an error
2. **Cost estimation**: Predict how much a conversation will cost before making the call

### Estimating Tokens

Exact token counting requires the model's actual tokenizer. For estimation, a simple heuristic works well. Build two estimation functions:

```typescript
function estimateTokens(text: string): number

function estimateTokensAccurate(text: string, isCode?: boolean): number
```

`estimateTokens` uses the character-based heuristic: 1 token is approximately 4 characters in English. Divide `text.length` by 4 and round up.

`estimateTokensAccurate` uses a word-based heuristic. Split the text on whitespace, count the words, then multiply by a factor: ~1.3 for English prose, ~2.0 for code (because of punctuation, braces, and special characters). The `isCode` parameter (default `false`) selects the multiplier.

### Token Counting for Message Arrays

Build functions to count tokens for entire message arrays and track budget usage:

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

function countMessageTokens(messages: Message[]): number

function getRemainingBudget(
  messages: Message[],
  budget: TokenBudget
): {
  used: number
  remaining: number
  percentUsed: number
  canFitMoreMessages: boolean
}
```

`countMessageTokens` should iterate over the messages, adding ~4 tokens of overhead per message (for role formatting), plus `estimateTokens(message.content)` for the content. Add ~3 tokens at the end for conversation framing overhead.

`getRemainingBudget` calculates the available space for messages (`contextWindow - maxResponseTokens - reservedForSystem`), subtracts the used tokens from `countMessageTokens`, and returns the budget status. Set `canFitMoreMessages` to `true` if at least 200 tokens remain — enough for a minimal new message.

What happens to `percentUsed` if the available space is very small? Could `remaining` go negative?

### Token-Aware Sliding Window

Combine token counting with the sliding window to trim by token count rather than message count. Build a class that manages the conversation with token awareness:

```typescript
import { generateText, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
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
  })

  async send(userMessage: string): Promise<string>
  private buildContext(): Message[]
  getTokenStats(): {
    totalHistoryTokens: number
    messageCount: number
    averageTokensPerMessage: number
  }
}
```

The constructor stores config values, defaulting the model to `mistral('mistral-small-latest')`, `maxInputTokens` to 180,000, and `maxResponseTokens` to 4,096.

**`send`** pushes the user message, calls `buildContext` to get the context-windowed messages, calls `generateText` with that context, pushes the assistant response, and returns the text. If `usage` is available in the response, log the input/output token counts.

**`buildContext`** is the key method. It computes the token budget: `maxInputTokens` minus the system message tokens minus `maxResponseTokens`. Then it iterates from the most recent message backward, accumulating messages until adding the next one would exceed the budget. After selection, it trims any leading assistant messages (to maintain role alternation). Finally it prepends the system message.

Why iterate from newest to oldest instead of oldest to newest? What is the trade-off of using token-based windowing versus message-count-based windowing?

**`getTokenStats`** calculates the total tokens across all stored messages and returns summary statistics.

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

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: LLM-Assisted Memory Extraction

### Using the LLM to Decide What to Remember

The summarization strategy compresses entire conversation segments into summaries. But there is a more targeted approach: ask the LLM to extract only the _facts worth remembering_ from a conversation, and store those separately.

This is a production-essential pattern. Instead of a rolling summary that drifts over time, you maintain a curated set of structured memories — user preferences, project decisions, important names and dates — that persist across compaction cycles.

### The Extraction Pattern

After every N turns (or before compaction), send the recent conversation to the LLM with an extraction prompt:

```typescript
const extractionPrompt = `Review this conversation and extract facts worth remembering.
Categories: user preferences, project context, decisions made, named entities.
Return only new or updated facts — do not repeat known facts.`
```

The LLM returns structured data (using `Output.object()` from Module 3) with the extracted facts. These are stored in the `facts` field of `ConversationState` and injected into every future context window.

### Why This Beats Pure Summarization

Summaries compress everything — important and unimportant alike. Extraction is surgical: it identifies the high-value information and discards the rest. A summary might say "The user discussed their TypeScript project." Extraction captures: `language: TypeScript`, `framework: Next.js`, `deployment: Vercel`.

Extracted facts also do not drift. A summary of a summary can distort details. A fact like `name: Jordan` stays exact indefinitely.

> **Persistent Memory Directories** — Some production coding agents persist extracted memories to a dedicated directory (e.g., `~/.app/memories/`) that survives across sessions, restarts, and context compaction. Each memory is a file with metadata (timestamp, source, category). At session start, relevant memories are loaded and injected into the system prompt. This is fundamentally different from session persistence — it stores **curated knowledge**, not raw conversation history.

---

## Section 10: Token Budget Allocation

### The Context Window as a Budget

Production systems treat the context window as a budget to be allocated, not just a limit to avoid hitting. A typical allocation:

| Component            | Budget % | Purpose                     |
| -------------------- | -------- | --------------------------- |
| System prompt        | 5-15%    | Instructions, role, rules   |
| Memories/facts       | 5-10%    | Persistent knowledge        |
| Conversation summary | 10-20%   | Compressed older context    |
| Recent messages      | 40-60%   | Active conversation window  |
| Response tokens      | 10-20%   | Space for the model's reply |

The key insight is that these allocations are configurable and should be tuned per use case. A coding assistant needs more space for tool results. A creative writing assistant needs more space for recent messages.

### Budget Monitoring

Track token usage per component and trigger compaction when any component exceeds its allocation:

```typescript
// Monitor budget usage — compact when conversation exceeds its allocation
const conversationTokens = countMessageTokens(messages)
const budgetLimit = contextWindow * 0.6
if (conversationTokens > budgetLimit) {
  // Trigger compaction
}
```

Warning thresholds at 80% and auto-compact at 90% give your application a graceful degradation path rather than a hard failure when the context fills up.

---

## Section 11: Microcompaction

### Surgical Pruning Without Full Summarization

Full compaction (summarizing older messages) is a heavyweight operation — it requires an LLM call and replaces detailed history with a compressed summary. Microcompaction is a lighter-weight alternative that surgically removes low-value content without any LLM calls.

Microcompaction targets:

1. **Duplicate tool results** — if the same file was read three times, keep only the latest result
2. **Verbose outputs** — truncate long tool outputs to their first N lines
3. **Redundant assistant messages** — if the assistant said "I'll search for that" followed by the actual search results, the preamble can be removed
4. **Non-essential metadata** — timestamps, progress updates, and status messages

### The Microcompaction Pattern

```typescript
// Microcompaction: remove duplicate tool results, truncate verbose outputs
// No LLM call required — pure data transformation on the message array
```

The key difference from summarization: microcompaction is deterministic and free (no LLM call). It is applied _before_ checking whether full compaction is needed, often reducing token count enough to avoid the expensive summarization step entirely.

In practice, production systems layer these approaches: microcompaction runs every turn (cheap), summarization runs only when microcompaction is not enough (expensive).

> **Automatic Compaction Agents** — Some production systems run a hidden "compaction agent" — a separate `generateText` call with its own system prompt optimized for compression. Rather than heuristic truncation, the compaction agent produces a compressed summary preserving key decisions, tool results, and task state. It runs automatically and invisibly when the context fills up. This is LLM-assisted memory extraction taken further — the compaction agent maintains conversational coherence while dramatically reducing token count.

---

## Summary

In this module, you learned:

1. **Multi-turn conversations:** LLMs are stateless — your application must manage the message array and send the full conversation history with every request.
2. **Context windows:** Each model has a finite token limit, and every message you send consumes tokens, creating a hard ceiling on conversation length.
3. **State and strategy separation:** Keeping conversation state (history, summaries, facts) separate from the memory strategy (how to build context) enables clean strategy switching without data loss.
4. **Sliding window strategy:** Keeping only the most recent N messages is simple and predictable but loses early context that may be important.
5. **Summarization strategy:** Using the LLM to compress older messages into a running summary preserves key context while staying within token limits.
6. **Hybrid approaches:** Combining a fact sheet, rolling summary, and sliding window gives the best balance of context retention and cost control. Facts survive summarization without distortion.
7. **Conversation persistence:** Because `ConversationState` is a plain data object, serializing it to JSON gives full-fidelity save/restore that works with any strategy.
8. **Token estimation:** Character-based estimation (length/4) provides a practical way to manage token budgets without external libraries.
9. **LLM-assisted memory extraction:** Using the LLM to extract structured facts from conversations, producing curated knowledge that survives compaction without distortion.
10. **Token budget allocation:** Treating the context window as a budget with percentage allocations for system prompt, memories, summary, recent messages, and response tokens.
11. **Microcompaction:** Lightweight, deterministic pruning (removing duplicate tool results, truncating verbose outputs) that reduces token count without LLM calls.

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

### Question 6 (Medium)

What advantage does LLM-assisted memory extraction have over rolling summarization for preserving user preferences across long conversations?

- A) Extraction is cheaper because it uses fewer tokens
- B) Extraction captures structured facts (e.g., `language: TypeScript`) that stay exact across compaction cycles, while summaries can distort details over repeated compression
- C) Extraction does not require any LLM calls
- D) Summarization cannot handle conversations longer than 10 turns

**Answer: B** — Summaries compress everything and a summary of a summary can distort details over time ("summary drift"). Extraction is surgical — it identifies high-value facts and stores them in structured form. A fact like `name: Jordan` or `framework: Next.js` stays exact indefinitely, while a summary might gradually lose or alter such details.

---

### Question 7 (Hard)

A production system uses microcompaction before checking whether full summarization is needed. Which of the following is NOT a valid microcompaction target?

- A) Duplicate tool results where the same file was read multiple times
- B) Verbose tool outputs truncated to their first N lines
- C) The system prompt, condensed to save tokens
- D) Assistant preamble messages like "I'll search for that" when followed by actual results

**Answer: C** — Microcompaction targets low-value content in the message history: duplicate tool results, verbose outputs, and redundant assistant messages. The system prompt is never a microcompaction target — it contains the behavioral instructions the model needs and must remain intact. Microcompaction is deterministic and free (no LLM call), applied before the more expensive full summarization step.

---

## Exercises

### Exercise Prep: Shared Conversation Runner

Before the exercises, build a reusable conversation runner that both exercises will import. This avoids duplicating the summarization and fact extraction logic.

**File:** `src/memory/runner.ts`

**What to build:**

1. A `createStrategy(name, config)` factory function that returns the right strategy given a name (`'window'`, `'summary'`, or `'hybrid'`) and config (`windowSize`, `summarizeThreshold`)
2. A `runMaintenance(state, strategy, model)` function that checks if the strategy needs summarization, runs the LLM summarization call if so, and (for hybrid) extracts facts using structured output. This is the "check → summarize → extract" loop that both the chatbot and benchmark need.
3. A Zod schema for fact extraction — e.g., `z.object({ facts: z.record(z.string(), z.string()) })`

The key insight: `runMaintenance` encapsulates the side-effectful part (LLM calls for summarization and fact extraction) so that strategies stay pure. Both exercises call `runMaintenance(state, strategy, model)` each turn and don't need to know the details.

**Try it:** Import and use `createStrategy` and `runMaintenance` from a test script. Create a manager, add a few messages, call `runMaintenance` — verify it summarizes when the threshold is hit.

---

### Exercise 1: Configurable Memory Strategy Chatbot

Build a command-line chatbot that supports all three memory strategies. You will build this incrementally across four stages, each adding one layer of functionality. Get each stage working before moving to the next.

**File:** `src/exercises/m04/ex01-chatbot.ts`

**Run with:** `bun run src/exercises/m04/ex01-chatbot.ts --strategy window`

---

#### Stage 1: Core REPL Loop

Get a basic chatbot working with one strategy at a time.

**What to build:**

1. Use `parseArgs` from `node:util` to accept a `--strategy` flag (`window`, `summary`, or `hybrid`) and a `--window-size` flag (default: 20)
2. Use `createStrategy()` from the exercise prep to create a `ConversationManager` with the matching strategy
3. Set up a read loop (Bun treats `console` as an async iterable — `for await (const line of console)`)
4. Each turn: add the user message to the manager → call `buildContext()` → pass the result to `generateText` → print the response → add the assistant message to the manager
5. Support `/quit` to exit

**Try it:** Run with `--strategy window`, have a 3-turn conversation. Verify the model remembers what you said in turn 1.

---

#### Stage 2: Slash Commands & Token Display

Add observability so you can see what the memory strategy is actually doing.

**What to add:**

1. Intercept lines starting with `/` before they reach the LLM
2. `/history` — print the raw message array from `manager.getHistory()`
3. `/stats` — print the current memory state: strategy name, message count, and (for summarizing/hybrid) whether a summary exists
4. After each LLM response, display the token count of the context you sent — use `countMessageTokens` from `src/memory/tokens.ts`

**Try it:** Chat for 5+ turns, then run `/stats` and `/history`. Check that the token count grows with each turn.

---

#### Stage 3: Auto-Summarization & Fact Extraction

Make the summarizing and hybrid strategies actually summarize when the conversation grows long, and extract facts for hybrid.

**What to add:**

1. Accept a `--summarize-threshold` flag (default: 30)
2. Each turn, call `runMaintenance(state, strategy, model)` from the exercise prep before building context. This handles summarization and fact extraction transparently.
3. Update `/stats` to show summary info and (for hybrid) the fact sheet

**Try it:** Set `--summarize-threshold 6` and `--strategy hybrid`. Tell the bot your name, your job, and your favourite programming language. Chat until summarization triggers. Run `/stats` — the facts should be preserved even though the original messages were summarized.

---

#### Stage 4: Persistence & Strategy Switching

Make conversations survive restarts and let users switch strategies mid-conversation.

**What to add:**

1. After each turn (user message + assistant response), auto-save the conversation to a JSON file using `saveConversation()` from the teaching sections, or write your own serialization
2. On startup, check if a save file exists and load it with `loadConversation()`, then restore via `manager.setState(saved.state)`
3. `/strategy <name>` — switch to a different strategy mid-conversation using `manager.setStrategy()` and `createStrategy()`. Because state is separate from strategy, this is trivial — the history, summaries, and facts are all preserved automatically.

**Try it:** Start with `--strategy window`, chat for a few turns, `/quit`, restart — verify the conversation continues. Then try `/strategy hybrid` mid-conversation and confirm the history carries over.

---

**Evaluation criteria (all stages combined):**

- All three strategies produce different context arrays for the same conversation
- `/stats` accurately reports memory state for each strategy
- Token count is displayed after each exchange
- Hybrid strategy extracts and maintains a persistent fact sheet
- Conversation persists across restarts
- `/strategy` switching preserves conversation history

### Exercise 2: Memory Strategy Comparison

Write a benchmark function that runs a conversation through a memory strategy and tracks token usage.

**File:** `src/exercises/m04/ex02-benchmark.ts`

**What to build:**

Export a `benchmarkStrategy(name, messages, config?)` function that:

1. Creates a `ConversationManager` with `createStrategy(name, config)` from the exercise prep
2. Runs each message through the manager, calling `runMaintenance()` and `generateText` each turn
3. Tracks input/output tokens from `resp.usage` per turn (include maintenance tokens too)
4. Returns an object with `totalInputTokens`, `totalOutputTokens`, and the `manager` (so the test can run recall questions on the same conversation state afterward)

```typescript
import type { ConversationManager } from '../../memory/manager.js'
import type { StrategyName, StrategyConfig } from '../../memory/runner.js'

interface BenchmarkResult {
  totalInputTokens: number
  totalOutputTokens: number
  manager: ConversationManager
}

export function benchmarkStrategy(
  name: StrategyName,
  messages: string[],
  config?: StrategyConfig
): Promise<BenchmarkResult>
```

The test provides 24 hardcoded user messages with facts planted in the first 3 messages (name, location, company, language). It also runs recall questions after each benchmark to measure how well each strategy preserves early context. Your job is just the benchmark loop.

> **Note for Claude Code:** When writing the test file for this exercise, generate a realistic 20-30 message conversation yourself and hardcode it. Plant the `TestFacts` values (name, location, company, language) in the first 3 messages. Do not call the LLM to generate test messages at test time — that adds latency and nondeterminism.

**Try it:** Run `bun test tests/memory/benchmark.test.ts` to validate your implementation.

---

### Exercise 3: Session Memory with LLM Extraction

**Objective:** Build a `SessionMemory` class that uses the LLM to extract key facts from a conversation, persists them, and re-injects them into a fresh context after compaction.

**File:** `src/exercises/m04/ex03-session-memory.ts`

**What to build:**

1. Define a `MemoryEntry` type with: `key` (string), `value` (string), `category` (enum: `'preference' | 'fact' | 'decision' | 'context'`), `extractedAt` (ISO date string)
2. Export a class `SessionMemory` with:
   - `constructor(savePath: string)` — path to a JSON file for persistence
   - `async extractFromMessages(messages: Message[], model: LanguageModel): Promise<MemoryEntry[]>` — sends the messages to the LLM with an extraction prompt and uses `generateText` with `Output.object()` to get structured facts back. Returns only new or updated entries.
   - `addEntries(entries: MemoryEntry[]): void` — merges new entries into the memory store, overwriting existing entries with the same key
   - `getEntries(category?: string): MemoryEntry[]` — returns all entries, optionally filtered by category
   - `toPromptString(): string` — formats all entries as a string suitable for injection into a system prompt (e.g., `"- name: Jordan\n- language: TypeScript"`)
   - `async save(): Promise<void>` — persists the memory store to disk as JSON
   - `async load(): Promise<void>` — loads the memory store from disk if it exists
3. The extraction prompt should ask the LLM to identify: user preferences, project facts, decisions made, and important named entities

**Test specification:**

```typescript
// tests/exercises/m04/ex03-session-memory.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 3: Session Memory', () => {
  it('should extract facts from a conversation', async () => {
    const memory = new SessionMemory('./data/test-memory.json')
    const messages = [
      { role: 'user' as const, content: 'My name is Jordan and I work at DataFlow.' },
      { role: 'assistant' as const, content: 'Nice to meet you, Jordan! What can I help with?' },
      { role: 'user' as const, content: 'I prefer TypeScript and use Next.js for my projects.' },
    ]
    const entries = await memory.extractFromMessages(messages, model)
    expect(entries.length).toBeGreaterThan(0)
    const names = entries.map(e => e.key)
    expect(names.some(n => n.toLowerCase().includes('name'))).toBe(true)
  })

  it('should format entries as a prompt string', () => {
    const memory = new SessionMemory('./data/test-memory.json')
    memory.addEntries([
      { key: 'name', value: 'Jordan', category: 'fact', extractedAt: new Date().toISOString() },
      { key: 'language', value: 'TypeScript', category: 'preference', extractedAt: new Date().toISOString() },
    ])
    const prompt = memory.toPromptString()
    expect(prompt).toContain('name')
    expect(prompt).toContain('Jordan')
  })

  it('should overwrite existing entries with the same key', () => {
    const memory = new SessionMemory('./data/test-memory.json')
    memory.addEntries([{ key: 'city', value: 'Austin', category: 'fact', extractedAt: new Date().toISOString() }])
    memory.addEntries([{ key: 'city', value: 'Portland', category: 'fact', extractedAt: new Date().toISOString() }])
    expect(memory.getEntries().filter(e => e.key === 'city')).toHaveLength(1)
    expect(memory.getEntries().find(e => e.key === 'city')?.value).toBe('Portland')
  })
})
```

---

### Exercise 4: Auto-Compact Trigger

**Objective:** Build a token budget monitor that triggers compaction at configurable thresholds, connecting token counting (Section 8) with the summarization strategy (Section 5).

**File:** `src/exercises/m04/ex04-auto-compact.ts`

**What to build:**

1. Define a `CompactionConfig` type:
   ```typescript
   interface CompactionConfig {
     contextWindow: number
     maxResponseTokens: number
     warnThreshold: number // percentage (e.g., 0.8 for 80%)
     compactThreshold: number // percentage (e.g., 0.9 for 90%)
   }
   ```
2. Export a function `checkBudget(messages: Message[], config: CompactionConfig): { status: 'ok' | 'warn' | 'compact'; percentUsed: number; tokensUsed: number }` that estimates the token count of the message array and returns the budget status
3. Export an async function `autoCompact(messages: Message[], config: CompactionConfig, model: LanguageModel): Promise<{ messages: Message[]; compacted: boolean; summary?: string }>` that:
   - Calls `checkBudget` to determine if compaction is needed
   - If status is `'compact'`, summarizes the older messages (keeping the most recent 6) using `generateText` with a summarization prompt, and returns the compacted message array with the summary prepended as a system message
   - If status is `'ok'` or `'warn'`, returns the original messages unchanged
4. The summarization prompt should preserve key facts, decisions, and named entities

**Test specification:**

```typescript
// tests/exercises/m04/ex04-auto-compact.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 4: Auto-Compact Trigger', () => {
  it('should return ok status when under warn threshold', () => {
    const messages = [{ role: 'user' as const, content: 'Hello' }]
    const result = checkBudget(messages, {
      contextWindow: 100_000,
      maxResponseTokens: 4096,
      warnThreshold: 0.8,
      compactThreshold: 0.9,
    })
    expect(result.status).toBe('ok')
  })

  it('should return warn status when between thresholds', () => {
    // Generate messages large enough to hit 80-90% of a small context window
    const longMessage = 'x'.repeat(3200) // ~800 tokens
    const messages = Array.from({ length: 10 }, () => ({
      role: 'user' as const,
      content: longMessage,
    }))
    const result = checkBudget(messages, {
      contextWindow: 12_000,
      maxResponseTokens: 1000,
      warnThreshold: 0.8,
      compactThreshold: 0.9,
    })
    expect(result.status).toBe('warn')
  })

  it('should compact when over compact threshold', async () => {
    const longMessage = 'x'.repeat(4000) // ~1000 tokens
    const messages = Array.from({ length: 10 }, (_, i) => ({
      role: (i % 2 === 0 ? 'user' : 'assistant') as 'user' | 'assistant',
      content: `${longMessage} message ${i}`,
    }))
    const result = await autoCompact(
      messages,
      { contextWindow: 12_000, maxResponseTokens: 1000, warnThreshold: 0.8, compactThreshold: 0.9 },
      model
    )
    expect(result.compacted).toBe(true)
    expect(result.summary).toBeTruthy()
    expect(result.messages.length).toBeLessThan(messages.length)
  })
})
```
