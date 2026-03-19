# Module 14: Agent Fundamentals

## Learning Objectives

- Understand what an agent is and how it differs from a single LLM call
- Implement the ReAct (Reasoning + Acting) pattern with the Vercel AI SDK
- Build an agent loop that iterates through think, act, and observe phases
- Distinguish between planning-first and reactive agent strategies
- Implement tool selection logic and observation processing
- Add termination conditions including max steps, goal detection, and stuck detection
- Manage within-session agent memory for multi-step tasks
- Debug agents by tracing reasoning steps and diagnosing failures

---

## Why Should I Care?

A single `generateText` call can answer a question, but it cannot research a topic across multiple sources, verify its own claims, or adapt its approach when a first attempt fails. Agents can. An agent is what you get when you give an LLM the ability to use tools and let it run in a loop until it accomplishes a goal. This is where LLM applications stop being fancy autocomplete and start being genuinely useful software.

Every serious LLM product — coding assistants, research tools, customer support bots, data analysis pipelines — is built on agent patterns. Understanding the fundamentals in this module will let you build agents that are reliable, debuggable, and controllable rather than unpredictable black boxes.

If you skip this module, multi-agent systems (Module 15) and workflows (Module 16) will feel like magic you cannot debug. The patterns here are the foundation for everything that follows.

---

## Connection to Other Modules

- **Module 7 (Tool Use)** introduced tool definitions and single-step tool calls. This module extends that into multi-step loops.
- **Module 15 (Multi-Agent Systems)** builds directly on the agent loop pattern, running multiple agents that coordinate.
- **Module 16 (Workflows & Chains)** contrasts the autonomous agent approach with deterministic pipelines.
- **Module 17 (Code Generation)** applies agent patterns to iterative code writing and debugging.
- **Module 18 (Human-in-the-Loop)** adds approval gates and feedback into the agent loop.

---

## Section 1: What is an Agent?

### The Three Components

An agent is built from three elements working together:

1. **An LLM** — the reasoning engine that decides what to do
2. **Tools** — functions the LLM can call to interact with the world
3. **A loop** — the control flow that keeps the LLM running until the task is done

A single `generateText` call with tools is not an agent. It is a one-shot tool call. An agent runs that call repeatedly, feeding results back in, until it decides the task is complete.

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// This is NOT an agent — it is a single tool call
const singleCall = await generateText({
  model: mistral('mistral-small-latest'),
  tools: {
    searchWeb: {
      description: 'Search the web for information',
      parameters: z.object({
        query: z.string().describe('The search query'),
      }),
      execute: async ({ query }) => {
        // Simulated search
        return `Results for "${query}": Found 3 relevant articles.`
      },
    },
  },
  prompt: 'What is the population of Tokyo?',
})

// singleCall.text might be empty if the model wants to use a tool
// but there is no loop to process the tool result
```

> **Beginner Note:** Think of the difference between asking someone a question (single call) versus hiring someone to complete a project (agent). The project requires multiple steps, checking intermediate results, and adjusting the approach.

### From Single Call to Agent

The key insight is that `generateText` with `maxSteps` already provides an agent loop. The Vercel AI SDK will automatically re-call the model with tool results until the model either stops calling tools or hits the step limit:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// This IS an agent — maxSteps creates the loop
const agentResult = await generateText({
  model: mistral('mistral-small-latest'),
  maxSteps: 10,
  tools: {
    searchWeb: {
      description: 'Search the web for information',
      parameters: z.object({
        query: z.string().describe('The search query'),
      }),
      execute: async ({ query }) => {
        return `Results for "${query}": Tokyo population is approximately 14 million (city proper) or 37 million (greater metro area).`
      },
    },
  },
  prompt: 'What is the population of Tokyo? Use the search tool to find out.',
})

console.log('Final answer:', agentResult.text)
console.log('Steps taken:', agentResult.steps.length)

// Each step contains the model's reasoning and any tool calls
for (const step of agentResult.steps) {
  console.log(`Step ${agentResult.steps.indexOf(step) + 1}:`, {
    toolCalls: step.toolCalls.map(tc => tc.toolName),
    hasText: !!step.text,
  })
}
```

> **Advanced Note:** The `maxSteps` parameter is the simplest form of agent loop control in the Vercel AI SDK. For more complex agents that need custom logic between steps, you will build your own loop as shown in Section 3.

### What Makes a Good Agent

Not every task needs an agent. Use an agent when:

- The task requires multiple steps that depend on each other
- The LLM needs to gather information before answering
- The approach might need to change based on intermediate results
- The task involves verification or self-correction

Do not use an agent when:

- A single `generateText` call (with or without `Output.object`) suffices
- The steps are known in advance and do not depend on LLM decisions (use a workflow instead)
- Latency is critical and you cannot afford multiple LLM calls

---

## Section 2: The ReAct Pattern

### Reasoning + Acting

The ReAct pattern (Yao et al., 2022) is the most widely used agent pattern. It interleaves reasoning (thinking about what to do) with acting (using tools) in a structured cycle:

1. **Thought** — the model reasons about the current state and what to do next
2. **Action** — the model calls a tool
3. **Observation** — the tool result is fed back to the model

This cycle repeats until the model has enough information to give a final answer.

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// ReAct agent with explicit reasoning via system prompt
const reactAgent = await generateText({
  model: mistral('mistral-small-latest'),
  maxSteps: 8,
  system: `You are a research agent. For each step:
1. THINK: Reason about what you know and what you still need to find out.
2. ACT: Use a tool to gather information.
3. OBSERVE: Analyze the result and decide if you need more information.

When you have enough information, provide a comprehensive final answer.
Always explain your reasoning before using a tool.`,
  tools: {
    searchWeb: {
      description: 'Search the web for current information',
      parameters: z.object({
        query: z.string().describe('Search query'),
      }),
      execute: async ({ query }) => {
        // Simulated search results
        const results: Record<string, string> = {
          'TypeScript market share 2025':
            'TypeScript is used by 38.5% of developers according to Stack Overflow 2025 survey.',
          'TypeScript vs JavaScript performance':
            'TypeScript compiles to JavaScript so runtime performance is identical. Compilation adds a build step.',
          'TypeScript adoption Fortune 500':
            'Over 80% of Fortune 500 companies using JavaScript have adopted TypeScript for new projects.',
        }
        return results[query] || `No specific results found for "${query}". Try a different query.`
      },
    },
    calculator: {
      description: 'Perform mathematical calculations',
      parameters: z.object({
        expression: z.string().describe('Math expression to evaluate'),
      }),
      execute: async ({ expression }) => {
        try {
          if (!/^[\d\s+\-*/().]+$/.test(expression)) {
            return `Error: Invalid expression "${expression}"`
          }
          // WARNING: Function() is eval() in disguise. In production, use a math parser library like mathjs.
          const result = Function(`"use strict"; return (${expression})`)()
          return `${expression} = ${result}`
        } catch {
          return `Error evaluating: ${expression}`
        }
      },
    },
  },
  prompt:
    'Research the current state of TypeScript adoption. I want to know market share, performance implications, and enterprise adoption. Provide a summary with specific numbers.',
})

console.log('=== ReAct Agent Result ===')
console.log('Final answer:', reactAgent.text)
console.log('\n=== Reasoning Trace ===')
for (const [i, step] of reactAgent.steps.entries()) {
  console.log(`\nStep ${i + 1}:`)
  if (step.text) {
    console.log('  Thought:', step.text.slice(0, 200))
  }
  for (const tc of step.toolCalls) {
    console.log(`  Action: ${tc.toolName}(${JSON.stringify(tc.args)})`)
  }
  for (const tr of step.toolResults) {
    console.log(`  Observation: ${String(tr.result).slice(0, 150)}`)
  }
}
```

### Why ReAct Works

The power of ReAct comes from making the model's reasoning explicit. When the model must articulate its thought process before acting, it makes better decisions about which tools to use and how to interpret results.

Without explicit reasoning, an agent might:

- Call the same tool repeatedly with the same query
- Miss important information in tool results
- Fail to connect information across multiple tool calls

With reasoning, the agent can:

- Identify gaps in its knowledge
- Formulate targeted queries
- Synthesize information from multiple sources
- Recognize when it has enough information to answer

> **Beginner Note:** You do not need to implement ReAct from scratch. The combination of a good system prompt and the Vercel AI SDK's `maxSteps` gives you ReAct behavior. The system prompt encourages the model to reason explicitly, and `maxSteps` provides the loop.

> **Advanced Note:** Some models handle ReAct-style reasoning better than others. Claude models naturally tend to reason before acting. For models that rush to tool calls without thinking, you can use a "scratchpad" tool that the model calls to write down its thoughts before using action tools.

---

## Section 3: Agent Loop Implementation

### The Basic Loop

While `maxSteps` handles simple cases, building your own agent loop gives you full control over the process. Here is the fundamental pattern:

```typescript
import { generateText, type ModelMessage, type LanguageModel } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface AgentConfig {
  model: LanguageModel
  system: string
  tools: Record<string, any>
  maxSteps: number
}

interface AgentResult {
  answer: string
  steps: Array<{
    thought: string
    toolCalls: Array<{ name: string; args: Record<string, unknown> }>
    observations: string[]
  }>
  totalSteps: number
  finished: boolean
}

async function runAgent(config: AgentConfig, task: string): Promise<AgentResult> {
  const messages: ModelMessage[] = [{ role: 'user', content: task }]
  const steps: AgentResult['steps'] = []
  let finished = false

  for (let step = 0; step < config.maxSteps; step++) {
    // THINK + ACT: Call the model
    const response = await generateText({
      model: config.model,
      system: config.system,
      messages,
      tools: config.tools,
      maxSteps: 1, // Single step — we control the outer loop
    })

    // Record what happened in this step
    const currentStep = {
      thought: response.text || '',
      toolCalls: response.steps.flatMap(s =>
        s.toolCalls.map(tc => ({
          name: tc.toolName,
          args: tc.args as Record<string, unknown>,
        }))
      ),
      observations: response.steps.flatMap(s => s.toolResults.map(tr => String(tr.result))),
    }
    steps.push(currentStep)

    // Add assistant response to conversation
    // The response.response.messages contains all messages from this step
    for (const msg of response.response.messages) {
      messages.push(msg)
    }

    // CHECK: Did the model finish without calling tools?
    const lastStep = response.steps[response.steps.length - 1]
    if (lastStep && lastStep.toolCalls.length === 0) {
      finished = true
      return {
        answer: response.text,
        steps,
        totalSteps: step + 1,
        finished,
      }
    }

    console.log(`[Agent] Step ${step + 1}: Called ${currentStep.toolCalls.map(tc => tc.name).join(', ')}`)
  }

  // Hit max steps without finishing
  return {
    answer: steps[steps.length - 1]?.thought || 'Agent did not reach a conclusion.',
    steps,
    totalSteps: config.maxSteps,
    finished: false,
  }
}
```

### Using the Agent Loop

```typescript
const result = await runAgent(
  {
    model: mistral('mistral-small-latest'),
    system: `You are a research assistant. Use the available tools to gather information,
then provide a comprehensive answer. Think step by step.
When you have enough information, respond with your final answer WITHOUT calling any tools.`,
    tools: {
      search: {
        description: 'Search for information on a topic',
        parameters: z.object({
          query: z.string().describe('What to search for'),
        }),
        execute: async ({ query }: { query: string }) => {
          return `Search results for "${query}": [Relevant information would appear here]`
        },
      },
      getDetails: {
        description: 'Get detailed information about a specific item',
        parameters: z.object({
          item: z.string().describe('The item to get details for'),
        }),
        execute: async ({ item }: { item: string }) => {
          return `Details for "${item}": [Detailed information would appear here]`
        },
      },
    },
    maxSteps: 5,
  },
  'Compare the populations of the three largest cities in Japan.'
)

console.log('\n=== Agent Report ===')
console.log(`Finished: ${result.finished}`)
console.log(`Steps taken: ${result.totalSteps}`)
console.log(`Answer: ${result.answer}`)
```

> **Beginner Note:** The custom loop pattern gives you a place to insert logging, approval gates, memory management, and other custom logic between each agent step. The `maxSteps` approach in the Vercel AI SDK is simpler but less flexible.

### Adding Step Callbacks

A production agent needs visibility into what is happening at each step:

```typescript
import { generateText, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface StepEvent {
  stepNumber: number
  thought: string
  toolCalls: Array<{ name: string; args: Record<string, unknown> }>
  observations: string[]
  durationMs: number
}

type StepCallback = (event: StepEvent) => void | Promise<void>

async function runAgentWithCallbacks(
  task: string,
  tools: Record<string, any>,
  onStep: StepCallback,
  maxSteps = 10
): Promise<string> {
  const messages: ModelMessage[] = [{ role: 'user', content: task }]

  for (let step = 0; step < maxSteps; step++) {
    const startTime = Date.now()

    const response = await generateText({
      model: mistral('mistral-small-latest'),
      system: 'You are a helpful research agent. Use tools to gather information, then answer.',
      messages,
      tools,
      maxSteps: 1,
    })

    const durationMs = Date.now() - startTime

    const stepEvent: StepEvent = {
      stepNumber: step + 1,
      thought: response.text || '',
      toolCalls: response.steps.flatMap(s =>
        s.toolCalls.map(tc => ({
          name: tc.toolName,
          args: tc.args as Record<string, unknown>,
        }))
      ),
      observations: response.steps.flatMap(s => s.toolResults.map(tr => String(tr.result))),
      durationMs,
    }

    await onStep(stepEvent)

    for (const msg of response.response.messages) {
      messages.push(msg)
    }

    const lastStep = response.steps[response.steps.length - 1]
    if (lastStep && lastStep.toolCalls.length === 0) {
      return response.text
    }
  }

  return 'Max steps reached without conclusion.'
}

// Usage with a logging callback
const answer = await runAgentWithCallbacks(
  'What are the top 3 programming languages by popularity?',
  {
    search: {
      description: 'Search for information',
      parameters: z.object({ query: z.string() }),
      execute: async ({ query }: { query: string }) =>
        `Results for "${query}": Python, JavaScript, TypeScript are the top 3 in 2025.`,
    },
  },
  event => {
    console.log(`\n--- Step ${event.stepNumber} (${event.durationMs}ms) ---`)
    if (event.thought) console.log(`Thought: ${event.thought.slice(0, 100)}`)
    for (const tc of event.toolCalls) {
      console.log(`Action: ${tc.name}(${JSON.stringify(tc.args)})`)
    }
    for (const obs of event.observations) {
      console.log(`Observation: ${obs.slice(0, 100)}`)
    }
  }
)

console.log('\nFinal answer:', answer)
```

---

## Section 4: Planning vs Reacting

### Two Agent Strategies

Agents can approach tasks in two fundamentally different ways:

1. **Reactive (step-by-step)** — Decide what to do next based only on the current state. This is the basic ReAct pattern.
2. **Planning-first** — Create a plan before executing, then follow the plan (possibly revising it).

Neither is universally better. The right choice depends on the task.

### Reactive Agents

Reactive agents are simpler and work well when:

- The task is exploratory (you do not know what you will find)
- Each step's output heavily influences the next step
- The task is short (fewer than 5 steps)

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Reactive agent — decides step by step
const reactiveResult = await generateText({
  model: mistral('mistral-small-latest'),
  maxSteps: 6,
  system: `You are a reactive research agent. At each step:
1. Look at what you know so far
2. Decide what single piece of information would be most useful next
3. Use a tool to get that information
4. Repeat until you can answer the question

Do NOT plan ahead. Just take the best next step.`,
  tools: {
    search: {
      description: 'Search for information',
      parameters: z.object({ query: z.string() }),
      execute: async ({ query }) => `Results for "${query}": [simulated result]`,
    },
  },
  prompt: 'What caused the 2008 financial crisis?',
})
```

### Planning Agents

Planning agents create an explicit plan before executing. They work well when:

- The task has multiple independent sub-tasks
- You want the user to review the plan before execution begins
- The task is complex (more than 5 steps)
- Efficiency matters — a plan avoids redundant tool calls

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Step 1: Generate a plan
const planSchema = z.object({
  goal: z.string().describe('The overall goal'),
  steps: z
    .array(
      z.object({
        id: z.number(),
        description: z.string(),
        tool: z.string().describe('Which tool to use'),
        query: z.string().describe('What to search/look up'),
        dependsOn: z.array(z.number()).describe('IDs of steps this depends on'),
      })
    )
    .describe('Ordered list of steps to execute'),
})

type Plan = z.infer<typeof planSchema>

async function createPlan(task: string): Promise<Plan> {
  const { output: plan } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: planSchema }),
    prompt: `Create a research plan for the following task. Break it into concrete steps.
Available tools: search (web search), getPage (fetch a URL), calculate (math).
Task: ${task}`,
  })
  return plan
}

// Step 2: Execute the plan
async function executePlan(plan: Plan): Promise<Map<number, string>> {
  const results = new Map<number, string>()

  for (const step of plan.steps) {
    // Check dependencies are met
    const depsReady = step.dependsOn.every(depId => results.has(depId))
    if (!depsReady) {
      console.warn(`Skipping step ${step.id}: dependencies not met`)
      continue
    }

    console.log(`Executing step ${step.id}: ${step.description}`)

    // Gather context from dependencies
    const context = step.dependsOn.map(depId => `Step ${depId} result: ${results.get(depId)}`).join('\n')

    // Execute with context
    const response = await generateText({
      model: mistral('mistral-small-latest'),
      maxSteps: 2,
      tools: {
        search: {
          description: 'Search the web',
          parameters: z.object({ query: z.string() }),
          execute: async ({ query }) => `Results for "${query}": [simulated search results]`,
        },
      },
      prompt: `${context ? `Context from previous steps:\n${context}\n\n` : ''}Execute this step: ${step.description}\nQuery: ${step.query}`,
    })

    results.set(step.id, response.text)
  }

  return results
}

// Step 3: Synthesize final answer
async function synthesize(task: string, plan: Plan, results: Map<number, string>): Promise<string> {
  const stepResults = plan.steps
    .map(s => `Step ${s.id} (${s.description}): ${results.get(s.id) || 'No result'}`)
    .join('\n\n')

  const response = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: `Original task: ${task}

Research results:
${stepResults}

Synthesize these results into a comprehensive answer.`,
  })

  return response.text
}

// Run the planning agent
async function planningAgent(task: string): Promise<string> {
  console.log('=== Phase 1: Planning ===')
  const plan = await createPlan(task)
  console.log('Plan created:', JSON.stringify(plan, null, 2))

  console.log('\n=== Phase 2: Execution ===')
  const results = await executePlan(plan)

  console.log('\n=== Phase 3: Synthesis ===')
  const answer = await synthesize(task, plan, results)

  return answer
}

const answer = await planningAgent('Compare the economic policies of the US and EU regarding AI regulation')
console.log('\nFinal answer:', answer)
```

> **Advanced Note:** Hybrid approaches work well in practice. Start with a lightweight plan, execute reactively within each step, and revise the plan if new information invalidates it. This balances efficiency with flexibility.

---

## Section 5: Tool Selection

### How Agents Choose Tools

When an agent has multiple tools available, the LLM must decide which one to use. This decision is driven by:

1. **Tool descriptions** — clear, specific descriptions help the model match tools to needs
2. **Parameter schemas** — well-typed parameters with descriptions guide correct usage
3. **System prompt guidance** — explicit instructions about when to use each tool
4. **Context** — the current conversation state and what information is still needed

### Writing Good Tool Descriptions

Tool descriptions are prompts. They should be specific and include examples of when to use (and when not to use) the tool:

```typescript
import { z } from 'zod'

// BAD: Vague descriptions lead to misuse
const badTools = {
  search: {
    description: 'Search for stuff',
    parameters: z.object({ q: z.string() }),
    execute: async ({ q }: { q: string }) => `Results for ${q}`,
  },
}

// GOOD: Specific descriptions with clear scope
const goodTools = {
  webSearch: {
    description:
      'Search the web for current information, news, or facts. Use this for questions about recent events, statistics, or when you need up-to-date data. Returns a list of relevant snippets. Do NOT use this for calculations or code execution.',
    parameters: z.object({
      query: z.string().describe('A specific search query. Be precise — use keywords rather than full sentences.'),
      maxResults: z.number().optional().default(5).describe('Maximum number of results to return (1-10)'),
    }),
    execute: async ({ query, maxResults }: { query: string; maxResults?: number }) => {
      return `Top ${maxResults ?? 5} results for "${query}": [results]`
    },
  },
  calculator: {
    description:
      'Evaluate a mathematical expression. Use this for any arithmetic, percentages, unit conversions, or numerical comparisons. Input must be a valid JavaScript math expression. Do NOT use this for text processing.',
    parameters: z.object({
      expression: z
        .string()
        .describe('A valid JavaScript math expression, e.g., "1024 * 768", "Math.sqrt(144)", "(15/100) * 250"'),
    }),
    execute: async ({ expression }: { expression: string }) => {
      if (!/^[\d\s+\-*/().]+$/.test(expression)) {
        return `Error: Invalid expression "${expression}"`
      }
      // WARNING: Function() is eval() in disguise. In production, use a math parser library like mathjs.
      const result = Function(`"use strict"; return (${expression})`)()
      return `${expression} = ${result}`
    },
  },
  readFile: {
    description:
      'Read the contents of a file from the local filesystem. Use this when the user references a specific file or when you need to analyze file contents. Returns the full text content of the file.',
    parameters: z.object({
      path: z.string().describe("Absolute or relative file path, e.g., './data/report.csv'"),
    }),
    execute: async ({ path }: { path: string }) => {
      return `Contents of ${path}: [file contents]`
    },
  },
}
```

### Tool Selection via System Prompt

For complex tool sets, add explicit routing guidance in the system prompt:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const toolRoutingAgent = await generateText({
  model: mistral('mistral-small-latest'),
  maxSteps: 5,
  system: `You are an assistant with access to several tools. Choose the right tool for each task:

TOOL SELECTION GUIDE:
- webSearch: Use for factual questions, current events, statistics
- database: Use for internal company data, user records, transaction history
- calculator: Use for any math operations

IMPORTANT:
- Try the most specific tool first (database before webSearch for internal data)
- If a tool returns no results, try rephrasing before switching tools
- Never use untrusted input in calculations`,
  tools: {
    webSearch: {
      description: 'Search the public web for information',
      parameters: z.object({ query: z.string() }),
      execute: async ({ query }) => `Web results for "${query}"`,
    },
    database: {
      description: 'Query the internal database for company data',
      parameters: z.object({
        table: z.enum(['users', 'orders', 'products']),
        filter: z.string().describe('SQL-like WHERE clause'),
      }),
      execute: async ({ table, filter }) => `Database results from ${table} where ${filter}`,
    },
    calculator: {
      description: 'Evaluate a math expression',
      parameters: z.object({ expression: z.string() }),
      execute: async ({ expression }) => {
        if (!/^[\d\s+\-*/().]+$/.test(expression)) {
          return `Error: Invalid expression "${expression}"`
        }
        // WARNING: Function() is eval() in disguise. In production, use a math parser library like mathjs.
        const result = Function(`"use strict"; return (${expression})`)()
        return `Result: ${result}`
      },
    },
  },
  prompt: 'How many orders did user 12345 place last month, and what was the average order value?',
})
```

> **Beginner Note:** The model reads tool descriptions like a developer reads API documentation. The better your descriptions, the better the model's tool choices. Spend time on descriptions — they are the most important part of tool definitions.

---

## Section 6: Observation Processing

### Feeding Results Back

After a tool executes, its result becomes an "observation" that the agent uses for its next decision. The quality of observations directly affects agent performance.

### Structured Observations

Return structured data from tools so the agent can easily extract what it needs:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const tools = {
  searchProducts: {
    description: 'Search for products in the catalog',
    parameters: z.object({
      query: z.string(),
      category: z.string().optional(),
      maxPrice: z.number().optional(),
    }),
    execute: async ({ query, category, maxPrice }: { query: string; category?: string; maxPrice?: number }) => {
      // Simulated product search
      const products = [
        { name: 'Widget Pro', price: 29.99, rating: 4.5, inStock: true },
        { name: 'Widget Basic', price: 14.99, rating: 4.0, inStock: true },
        { name: 'Widget Ultra', price: 59.99, rating: 4.8, inStock: false },
      ]

      const filtered = products.filter(p => {
        if (maxPrice && p.price > maxPrice) return false
        return true
      })

      // Return structured observation — not just raw data
      return JSON.stringify({
        query,
        totalResults: filtered.length,
        results: filtered,
        note:
          filtered.length === 0
            ? 'No products found. Try broadening your search.'
            : `Found ${filtered.length} products matching "${query}".`,
      })
    },
  },
}

const result = await generateText({
  model: mistral('mistral-small-latest'),
  maxSteps: 4,
  system: `You are a shopping assistant. When you receive search results:
1. Analyze the results for relevance to the user's needs
2. Note any items that are out of stock
3. Consider price-to-rating ratio when making recommendations
4. If no results found, suggest alternative searches`,
  tools,
  prompt: 'I need a widget for under $40. What do you recommend?',
})

console.log(result.text)
```

### Observation Summarization

When tool results are large, the agent may struggle with long context. Summarize observations before feeding them back:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const toolsWithSummarization = {
  fetchWebPage: {
    description: 'Fetch and read a web page. Returns a summary of the content.',
    parameters: z.object({
      url: z.string().describe('URL to fetch'),
    }),
    execute: async ({ url }: { url: string }) => {
      // Simulated large page content
      const rawContent = `[Imagine 10,000 words of web page content from ${url}]`

      // Summarize the content before returning to the agent
      const summary = await generateText({
        model: mistral('mistral-small-latest'),
        prompt: `Summarize the following web page content in 3-5 bullet points.
Focus on key facts and data points.

Content: ${rawContent}`,
      })

      return JSON.stringify({
        url,
        summary: summary.text,
        contentLength: rawContent.length,
        note: 'Full content was summarized. Ask for specific details if needed.',
      })
    },
  },
}
```

> **Advanced Note:** Observation summarization introduces a trade-off: you use fewer tokens in the agent's context window, but you may lose details the agent needs later. A good strategy is to keep full observations for the most recent 2-3 steps and summarize older ones.

### Error Observations

Tools fail. When they do, return useful error observations that help the agent recover:

```typescript
import { z } from 'zod'

const resilientTools = {
  apiCall: {
    description: 'Call an external API endpoint',
    parameters: z.object({
      endpoint: z.string(),
      method: z.enum(['GET', 'POST']),
    }),
    execute: async ({ endpoint, method }: { endpoint: string; method: string }) => {
      try {
        // Simulated API call
        if (endpoint.includes('broken')) {
          throw new Error('Connection timeout after 5000ms')
        }
        return JSON.stringify({
          success: true,
          data: { result: 'API response data' },
        })
      } catch (error) {
        // Return a helpful error observation instead of throwing
        return JSON.stringify({
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          suggestion: 'The API may be down. Try again or use an alternative data source.',
          endpoint,
          method,
        })
      }
    },
  },
}
```

---

## Section 7: Termination Conditions

### When Should an Agent Stop?

An agent without proper termination conditions can run forever, wasting tokens and money. There are several reasons an agent should stop:

1. **Goal achieved** — the agent has enough information to answer
2. **Max steps reached** — safety limit to prevent runaway agents
3. **Stuck detection** — the agent is repeating actions without progress
4. **Error threshold** — too many consecutive failures
5. **Budget exhaustion** — token or cost limit reached

### Implementing Termination Conditions

```typescript
import { generateText, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface TerminationConfig {
  maxSteps: number
  maxConsecutiveErrors: number
  maxRepeatedActions: number
  maxTokens: number
}

interface AgentState {
  step: number
  consecutiveErrors: number
  actionHistory: string[]
  totalTokens: number
  messages: ModelMessage[]
}

function shouldTerminate(state: AgentState, config: TerminationConfig): { terminate: boolean; reason: string } {
  // 1. Max steps
  if (state.step >= config.maxSteps) {
    return {
      terminate: true,
      reason: `Max steps reached (${config.maxSteps})`,
    }
  }

  // 2. Too many consecutive errors
  if (state.consecutiveErrors >= config.maxConsecutiveErrors) {
    return {
      terminate: true,
      reason: `Too many consecutive errors (${state.consecutiveErrors})`,
    }
  }

  // 3. Stuck detection — same action repeated
  if (state.actionHistory.length >= config.maxRepeatedActions) {
    const recent = state.actionHistory.slice(-config.maxRepeatedActions)
    const allSame = recent.every(a => a === recent[0])
    if (allSame) {
      return {
        terminate: true,
        reason: `Agent stuck: repeated "${recent[0]}" ${config.maxRepeatedActions} times`,
      }
    }
  }

  // 4. Token budget
  if (state.totalTokens >= config.maxTokens) {
    return {
      terminate: true,
      reason: `Token budget exhausted (${state.totalTokens}/${config.maxTokens})`,
    }
  }

  return { terminate: false, reason: '' }
}

async function runAgentWithTermination(
  task: string,
  tools: Record<string, any>,
  config: TerminationConfig
): Promise<{ answer: string; reason: string; steps: number }> {
  const state: AgentState = {
    step: 0,
    consecutiveErrors: 0,
    actionHistory: [],
    totalTokens: 0,
    messages: [{ role: 'user', content: task }],
  }

  while (true) {
    // Check termination before each step
    const check = shouldTerminate(state, config)
    if (check.terminate) {
      return {
        answer: 'Agent terminated early: ' + check.reason,
        reason: check.reason,
        steps: state.step,
      }
    }

    const response = await generateText({
      model: mistral('mistral-small-latest'),
      system: 'You are a helpful research agent. Use tools to gather information.',
      messages: state.messages,
      tools,
      maxSteps: 1,
    })

    state.step++

    // Track tokens
    state.totalTokens += (response.usage?.inputTokens ?? 0) + (response.usage?.outputTokens ?? 0)

    // Track actions for stuck detection
    const toolNames = response.steps.flatMap(s => s.toolCalls.map(tc => `${tc.toolName}(${JSON.stringify(tc.args)})`))

    if (toolNames.length > 0) {
      state.actionHistory.push(...toolNames)
    }

    // Track errors
    const hasError = response.steps.some(s => s.toolResults.some(tr => String(tr.result).includes('"success":false')))
    if (hasError) {
      state.consecutiveErrors++
    } else {
      state.consecutiveErrors = 0
    }

    // Add messages to conversation
    for (const msg of response.response.messages) {
      state.messages.push(msg)
    }

    // Check if agent finished naturally (no tool calls)
    const lastStep = response.steps[response.steps.length - 1]
    if (lastStep && lastStep.toolCalls.length === 0) {
      return {
        answer: response.text,
        reason: 'Goal achieved',
        steps: state.step,
      }
    }
  }
}

// Usage
const result = await runAgentWithTermination(
  'What is the GDP of France?',
  {
    search: {
      description: 'Search for information',
      parameters: z.object({ query: z.string() }),
      execute: async ({ query }: { query: string }) =>
        JSON.stringify({
          success: true,
          data: `GDP of France: $3.05 trillion (2024)`,
        }),
    },
  },
  {
    maxSteps: 10,
    maxConsecutiveErrors: 3,
    maxRepeatedActions: 3,
    maxTokens: 50000,
  }
)

console.log(`Answer: ${result.answer}`)
console.log(`Terminated because: ${result.reason}`)
console.log(`Steps taken: ${result.steps.length}`)
```

> **Beginner Note:** Always set a `maxSteps` limit, even during development. An agent without a step limit can run up a significant API bill very quickly. Start with 5-10 steps and increase only if needed.

> **Advanced Note:** Stuck detection is more nuanced than checking for identical actions. A sophisticated agent might call the same tool with different parameters (which is fine) or alternate between two tools without making progress (which is not). Consider tracking the information gained at each step, not just the tools called.

---

## Section 8: Agent Memory

### Within-Session Context

An agent's "memory" within a session is the conversation history — the messages array that accumulates as the agent runs. Managing this memory well is critical for agent performance.

### Context Window Management

As an agent takes more steps, the conversation history grows. Eventually it can exceed the model's context window. Here are strategies to manage this:

```typescript
import { generateText, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface MemoryManager {
  messages: ModelMessage[]
  maxMessages: number
  summaryInterval: number
}

function createMemoryManager(maxMessages: number = 50, summaryInterval: number = 10): MemoryManager {
  return {
    messages: [],
    maxMessages,
    summaryInterval,
  }
}

async function addToMemory(memory: MemoryManager, newMessages: ModelMessage[]): Promise<void> {
  memory.messages.push(...newMessages)

  // If we have too many messages, summarize older ones
  if (memory.messages.length > memory.maxMessages) {
    await compactMemory(memory)
  }
}

async function compactMemory(memory: MemoryManager): Promise<void> {
  // Keep the first message (original task) and recent messages
  const keepRecent = Math.floor(memory.maxMessages / 2)
  const originalTask = memory.messages[0]
  const oldMessages = memory.messages.slice(1, -keepRecent)
  const recentMessages = memory.messages.slice(-keepRecent)

  // Summarize old messages
  const oldContent = oldMessages
    .map(m => {
      if (typeof m.content === 'string') return `${m.role}: ${m.content}`
      return `${m.role}: [complex content]`
    })
    .join('\n')

  const summaryResponse = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: `Summarize the following agent conversation history. Focus on:
- Key findings and facts discovered
- Tools used and their results
- Decisions made and their reasoning
Keep it concise but preserve all important information.

Conversation:
${oldContent}`,
  })

  // Replace old messages with summary
  memory.messages = [
    originalTask,
    {
      role: 'user' as const,
      content: `[Summary of previous steps: ${summaryResponse.text}]`,
    },
    ...recentMessages,
  ]
}

function getMessages(memory: MemoryManager): ModelMessage[] {
  return [...memory.messages]
}
```

### Working Memory: Key-Value Store

For agents that need to track specific facts across many steps, use a structured working memory:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface WorkingMemory {
  facts: Map<string, string>
  scratchpad: string[]
  currentGoal: string
  subGoals: string[]
  completedGoals: string[]
}

function createWorkingMemory(goal: string): WorkingMemory {
  return {
    facts: new Map(),
    scratchpad: [],
    currentGoal: goal,
    subGoals: [],
    completedGoals: [],
  }
}

function memoryToPrompt(memory: WorkingMemory): string {
  const factsStr =
    memory.facts.size > 0
      ? Array.from(memory.facts.entries())
          .map(([k, v]) => `  - ${k}: ${v}`)
          .join('\n')
      : '  (none yet)'

  const scratchpadStr =
    memory.scratchpad.length > 0 ? memory.scratchpad.map(note => `  - ${note}`).join('\n') : '  (empty)'

  return `## Working Memory

### Current Goal
${memory.currentGoal}

### Known Facts
${factsStr}

### Scratchpad (notes to self)
${scratchpadStr}

### Completed Sub-goals
${memory.completedGoals.map(g => `  - [x] ${g}`).join('\n') || '  (none)'}

### Remaining Sub-goals
${memory.subGoals.map(g => `  - [ ] ${g}`).join('\n') || '  (none)'}`
}

// Agent with working memory
async function agentWithMemory(task: string): Promise<string> {
  const memory = createWorkingMemory(task)

  const tools = {
    search: {
      description: 'Search for information',
      parameters: z.object({ query: z.string() }),
      execute: async ({ query }: { query: string }) => `Results for "${query}": [simulated results]`,
    },
    addFact: {
      description: 'Store a key fact in working memory for later reference',
      parameters: z.object({
        key: z.string().describe('Short label for the fact'),
        value: z.string().describe('The fact itself'),
      }),
      execute: async ({ key, value }: { key: string; value: string }) => {
        memory.facts.set(key, value)
        return `Fact stored: ${key} = ${value}`
      },
    },
    addNote: {
      description: 'Add a note to the scratchpad for planning',
      parameters: z.object({
        note: z.string().describe('A note about your reasoning or plan'),
      }),
      execute: async ({ note }: { note: string }) => {
        memory.scratchpad.push(note)
        return `Note added: ${note}`
      },
    },
    completeGoal: {
      description: 'Mark a sub-goal as completed',
      parameters: z.object({
        goal: z.string().describe('The sub-goal that was completed'),
      }),
      execute: async ({ goal }: { goal: string }) => {
        memory.completedGoals.push(goal)
        memory.subGoals = memory.subGoals.filter(g => g !== goal)
        return `Goal completed: ${goal}`
      },
    },
  }

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    maxSteps: 10,
    system: `You are a research agent with working memory. Your working memory is shown below.
Use the memory tools (addFact, addNote, completeGoal) to organize your knowledge.
Use the search tool to find information.

${memoryToPrompt(memory)}

Work methodically: search, store facts, track progress, then synthesize your final answer.`,
    tools,
    prompt: task,
  })

  return result.text
}

const answer = await agentWithMemory('Compare the populations and GDP of Germany, France, and Italy')
console.log(answer)
```

> **Beginner Note:** Working memory tools (addFact, addNote) are a way to help the agent organize information across many steps. Without them, the agent must rely on the full conversation history which can be noisy and hard to parse.

---

## Section 9: Debugging Agents

### Why Agent Debugging is Hard

Agents are non-deterministic and multi-step. A bug might be:

- The model choosing the wrong tool
- A tool returning unexpected data
- The model misinterpreting a tool result
- The model getting stuck in a loop
- The conversation history growing too long and confusing the model

### Building a Debug Tracer

```typescript
import { generateText, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface TraceEntry {
  timestamp: number
  stepNumber: number
  type: 'thought' | 'action' | 'observation' | 'error' | 'termination'
  content: string
  metadata?: Record<string, unknown>
}

class AgentTracer {
  private traces: TraceEntry[] = []
  private startTime: number

  constructor() {
    this.startTime = Date.now()
  }

  log(stepNumber: number, type: TraceEntry['type'], content: string, metadata?: Record<string, unknown>): void {
    const entry: TraceEntry = {
      timestamp: Date.now() - this.startTime,
      stepNumber,
      type,
      content,
      metadata,
    }
    this.traces.push(entry)

    // Real-time console output
    const icon =
      type === 'thought'
        ? '[THINK]'
        : type === 'action'
          ? '[ACT]  '
          : type === 'observation'
            ? '[OBS]  '
            : type === 'error'
              ? '[ERR]  '
              : '[END]  '
    console.log(`${icon} Step ${stepNumber} (+${entry.timestamp}ms): ${content.slice(0, 200)}`)
  }

  getTraces(): TraceEntry[] {
    return [...this.traces]
  }

  printSummary(): void {
    console.log('\n========== Agent Trace Summary ==========')
    console.log(`Total steps: ${new Set(this.traces.map(t => t.stepNumber)).size}`)
    console.log(`Total time: ${this.traces[this.traces.length - 1]?.timestamp ?? 0}ms`)
    console.log(`Actions taken: ${this.traces.filter(t => t.type === 'action').length}`)
    console.log(`Errors: ${this.traces.filter(t => t.type === 'error').length}`)

    // Tool usage breakdown
    const toolCounts = new Map<string, number>()
    for (const trace of this.traces.filter(t => t.type === 'action')) {
      const tool = (trace.metadata?.toolName as string) || 'unknown'
      toolCounts.set(tool, (toolCounts.get(tool) || 0) + 1)
    }
    console.log('\nTool usage:')
    for (const [tool, count] of toolCounts) {
      console.log(`  ${tool}: ${count} calls`)
    }
    console.log('==========================================')
  }
}

// Agent with tracing
async function tracedAgent(task: string): Promise<string> {
  const tracer = new AgentTracer()
  const messages: ModelMessage[] = [{ role: 'user', content: task }]
  const maxSteps = 8

  const tools = {
    search: {
      description: 'Search for information',
      parameters: z.object({ query: z.string() }),
      execute: async ({ query }: { query: string }) => {
        // Simulate occasional failures for demonstration
        if (query.includes('fail')) {
          throw new Error('Search service unavailable')
        }
        return `Results for "${query}": [simulated results]`
      },
    },
  }

  for (let step = 0; step < maxSteps; step++) {
    try {
      const response = await generateText({
        model: mistral('mistral-small-latest'),
        system: 'You are a research agent. Search for information, then answer.',
        messages,
        tools,
        maxSteps: 1,
      })

      // Log thoughts
      if (response.text) {
        tracer.log(step + 1, 'thought', response.text)
      }

      // Log actions and observations
      for (const s of response.steps) {
        for (const tc of s.toolCalls) {
          tracer.log(step + 1, 'action', `${tc.toolName}(${JSON.stringify(tc.args)})`, {
            toolName: tc.toolName,
            args: tc.args,
          })
        }
        for (const tr of s.toolResults) {
          tracer.log(step + 1, 'observation', String(tr.result).slice(0, 500))
        }
      }

      // Add messages
      for (const msg of response.response.messages) {
        messages.push(msg)
      }

      // Check if done
      const lastStep = response.steps[response.steps.length - 1]
      if (lastStep && lastStep.toolCalls.length === 0) {
        tracer.log(step + 1, 'termination', 'Agent finished naturally')
        tracer.printSummary()
        return response.text
      }
    } catch (error) {
      tracer.log(step + 1, 'error', error instanceof Error ? error.message : 'Unknown error')
    }
  }

  tracer.log(maxSteps, 'termination', 'Max steps reached')
  tracer.printSummary()
  return 'Agent did not finish within step limit.'
}

await tracedAgent('Research the history of TypeScript')
```

### Common Agent Failures and Fixes

| Symptom                                         | Likely Cause                                 | Fix                                                                                    |
| ----------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- |
| Agent calls same tool repeatedly with same args | Observation not being processed              | Check that tool results are added to messages correctly                                |
| Agent never uses tools                          | Tool descriptions do not match the task      | Rewrite descriptions to match the agent's goal                                         |
| Agent uses wrong tool                           | Descriptions are ambiguous                   | Add "Use this when..." and "Do NOT use this for..." to descriptions                    |
| Agent runs all steps without answering          | No clear termination signal in system prompt | Add "When you have enough information, respond with your answer without calling tools" |
| Agent gives shallow answers                     | Not enough steps allowed                     | Increase `maxSteps` and encourage thorough research                                    |
| Agent hallucinates despite having tools         | Model not using tools for factual claims     | Add "Always verify claims using tools" to system prompt                                |

> **Advanced Note:** For production agents, ship traces to an observability platform (Langfuse, LangSmith, or a custom solution). Console logging is fine for development but you need persistent, searchable traces for production debugging.

---

## Quiz

### Question 1 (Easy)

What are the three core components of an LLM agent?

- A) Prompt, response, and database
- B) LLM, tools, and a loop
- C) Input, processing, and output
- D) Model, embeddings, and vector store

**Answer: B** — An agent consists of an LLM (the reasoning engine), tools (functions it can call to interact with the world), and a loop (control flow that keeps the LLM running until the task is done). Without the loop, you have a single tool call, not an agent.

---

### Question 2 (Medium)

In the ReAct pattern, what is the correct order of phases?

- A) Act, Think, Observe
- B) Observe, Think, Act
- C) Think, Act, Observe
- D) Plan, Execute, Verify

**Answer: C** — ReAct stands for Reasoning + Acting. The cycle is: Think (reason about what to do), Act (call a tool), Observe (process the tool result). This cycle repeats until the agent has enough information. Option D describes a planning agent, not the ReAct pattern.

---

### Question 3 (Medium)

An agent keeps calling the `searchWeb` tool with the same query "TypeScript history" repeatedly. What is the most likely cause?

- A) The model's temperature is too high
- B) The tool results are not being correctly added back to the conversation messages
- C) The `maxSteps` value is too low
- D) The model does not support tool use

**Answer: B** — When tool results are not fed back into the conversation, the model has no memory of having already called the tool. It sees the same context every time and makes the same decision. This is the most common agent bug. The fix is to ensure `response.response.messages` are appended to the messages array after each step.

---

### Question 4 (Hard)

When should you prefer a planning agent over a reactive (step-by-step) agent?

- A) When the task is simple and requires fewer than 3 steps
- B) When you need the lowest possible latency
- C) When the task has multiple independent sub-tasks and you want to avoid redundant tool calls
- D) When the tools are unreliable and might fail

**Answer: C** — Planning agents excel when the task can be decomposed into independent sub-tasks because the plan can identify all needed information upfront and avoid redundant searches. For simple tasks (A), a reactive agent is simpler. Planning adds latency (B) because it requires an extra LLM call. Unreliable tools (D) actually favor reactive agents that can adapt on the fly.

---

### Question 5 (Hard)

What is the primary risk of using conversation history as the agent's only form of memory?

- A) The conversation history is not persistent across sessions
- B) The history can grow to exceed the model's context window, causing the agent to lose earlier information
- C) Other agents cannot access the conversation history
- D) The conversation history uses too much disk space

**Answer: B** — As an agent takes more steps, the conversation history grows. Once it exceeds the context window, earlier messages are truncated, and the agent loses access to information from early steps. This is why memory management strategies like summarization and working memory are important for long-running agents. Cross-session persistence (A) is a separate concern. Multi-agent access (C) is addressed in Module 15.

---

## Exercises

### Exercise 1: ReAct Agent for Web Search Research

**Objective:** Build a ReAct agent that can research questions by searching the web, gathering multiple pieces of information, and synthesizing a comprehensive answer.

**Specification:**

1. Create a file `src/exercises/m14/ex01-react-research-agent.ts`
2. Export an async function `researchAgent(question: string, options?: AgentOptions): Promise<ResearchResult>`
3. Define the types:

```typescript
interface AgentOptions {
  maxSteps?: number // default: 8
  verbose?: boolean // default: false — print trace to console
}

interface ResearchStep {
  stepNumber: number
  thought: string
  action?: {
    tool: string
    args: Record<string, unknown>
  }
  observation?: string
  durationMs: number
}

interface ResearchResult {
  answer: string
  steps: ResearchStep[]
  totalSteps: number
  totalDurationMs: number
  toolCallCount: number
  finished: boolean // true if agent finished naturally, false if hit maxSteps
}
```

4. Implement the following tools:
   - `webSearch` — takes a `query` string, returns simulated search results (you may use a Map of pre-defined results or call a real API)
   - `readPage` — takes a `url` string, returns simulated page content
   - `notepad` — takes a `note` string, stores it in an array and returns confirmation (this is the agent's working memory)

5. The agent must:
   - Use a system prompt that encourages the ReAct pattern (think before acting)
   - Track each step's thought, action, and observation
   - Stop when the model provides a final answer without calling tools
   - Stop if `maxSteps` is reached
   - Detect if the agent is stuck (same tool call 3 times in a row) and force termination

6. If `verbose` is true, print each step to the console as it happens

**Example usage:**

```typescript
const result = await researchAgent('What are the main differences between React and Vue.js in 2025?', {
  maxSteps: 8,
  verbose: true,
})

console.log(`Answer: ${result.answer}`)
console.log(`Steps: ${result.totalSteps}, Tool calls: ${result.toolCallCount}`)
```

**Expected output format (verbose mode):**

```
[Step 1] Thought: I need to search for current comparisons between React and Vue.js...
[Step 1] Action: webSearch({ query: "React vs Vue.js 2025 comparison" })
[Step 1] Observation: Found 5 results comparing React and Vue.js...
[Step 1] Duration: 1,204ms

[Step 2] Thought: Let me get more specific data on performance differences...
[Step 2] Action: webSearch({ query: "React vs Vue.js performance benchmarks 2025" })
[Step 2] Observation: Performance benchmarks show Vue 3 is slightly faster in...
[Step 2] Duration: 983ms

...

[Step 5] Thought: I now have enough information to provide a comprehensive answer.
[Step 5] Duration: 1,102ms

=== Research Complete ===
Steps: 5, Tool calls: 4, Duration: 5,892ms
```

**Test specification:**

```typescript
// tests/exercises/m14/ex01-react-research-agent.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: ReAct Research Agent', () => {
  it('should return a non-empty answer', async () => {
    const result = await researchAgent('What is TypeScript?')
    expect(result.answer).toBeTruthy()
    expect(result.answer.length).toBeGreaterThan(50)
  })

  it('should track steps correctly', async () => {
    const result = await researchAgent('Explain REST APIs', { maxSteps: 5 })
    expect(result.steps.length).toBeGreaterThan(0)
    expect(result.steps.length).toBeLessThanOrEqual(5)
    expect(result.totalSteps).toBe(result.steps.length)
  })

  it('should count tool calls', async () => {
    const result = await researchAgent('What is Node.js?')
    expect(result.toolCallCount).toBeGreaterThan(0)
    expect(result.toolCallCount).toBeLessThanOrEqual(result.totalSteps)
  })

  it('should respect maxSteps', async () => {
    const result = await researchAgent('Explain quantum computing in detail', { maxSteps: 3 })
    expect(result.totalSteps).toBeLessThanOrEqual(3)
  })

  it('should detect stuck agent', async () => {
    // This test would require a mock that always returns the same result
    // forcing the agent to repeat the same search
    const result = await researchAgent('Obscure topic with no results', {
      maxSteps: 10,
    })
    expect(result.totalSteps).toBeLessThan(10)
  })
})
```

---

### Exercise 2: Agent with Working Memory

**Objective:** Extend the research agent with a structured working memory that persists facts, tracks sub-goals, and helps the agent stay organized across many steps.

**Specification:**

1. Create a file `src/exercises/m14/ex02-memory-agent.ts`
2. Export an async function `memoryAgent(task: string, options?: MemoryAgentOptions): Promise<MemoryAgentResult>`
3. Define the types:

```typescript
interface MemoryAgentOptions {
  maxSteps?: number // default: 12
  verbose?: boolean // default: false
}

interface WorkingMemory {
  facts: Record<string, string>
  notes: string[]
  subGoals: string[]
  completedGoals: string[]
}

interface MemoryAgentResult {
  answer: string
  memory: WorkingMemory // Final state of working memory
  totalSteps: number
  finished: boolean
}
```

4. In addition to `webSearch` and `readPage`, implement these memory tools:
   - `storeFact` — takes `key` and `value`, stores in working memory
   - `addNote` — takes `note`, adds to notes array
   - `setSubGoals` — takes `goals` (string array), sets the sub-goals list
   - `completeGoal` — takes `goal`, moves it from subGoals to completedGoals

5. Inject the current state of working memory into the system prompt at each step

6. The agent should use working memory to organize a multi-part research task

**Test specification:**

```typescript
// tests/exercises/m14/ex02-memory-agent.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 14: Memory Agent', () => {
  it('should store facts in working memory', async () => {
    const result = await memoryAgent('Research the top 3 JavaScript frameworks')
    expect(Object.keys(result.memory.facts).length).toBeGreaterThan(0)
  })

  it('should track completed goals', async () => {
    const result = await memoryAgent('Compare Python and JavaScript: syntax, performance, and ecosystem')
    expect(result.memory.completedGoals.length).toBeGreaterThan(0)
  })

  it('should produce a final answer using stored facts', async () => {
    const result = await memoryAgent('What are the SOLID principles?')
    expect(result.answer).toBeTruthy()
    expect(result.finished).toBe(true)
  })
})
```

> **Local Alternative (Ollama):** ReAct agents work with `ollama('qwen3.5')`, which supports tool calling. The agent loop, observation-action cycles, and `maxSteps` are provider-agnostic. Local agents are slower but fully private. For complex reasoning tasks, consider `ollama('qwen3.5:cloud')` or `ollama('deepseek-r1')` for better planning capabilities.

---

## Summary

In this module, you learned:

1. **What an agent is:** An LLM plus tools plus a loop. The Vercel AI SDK's `maxSteps` provides the simplest agent loop, but custom loops give you more control.
2. **The ReAct pattern:** Think, act, observe — the fundamental cycle that makes agents effective. System prompts encourage explicit reasoning before tool use.
3. **Agent loop implementation:** How to build a custom loop with step tracking, message management, and callbacks for observability.
4. **Planning vs reacting:** Reactive agents work step by step; planning agents create a plan first. Choose based on task complexity and structure.
5. **Tool selection:** Good tool descriptions, typed parameters, and system prompt guidance help agents choose the right tool.
6. **Observation processing:** Structured tool results, summarization of large observations, and helpful error messages improve agent accuracy.
7. **Termination conditions:** Max steps, stuck detection, error thresholds, and token budgets prevent runaway agents.
8. **Agent memory:** Conversation history management, context window compaction, and structured working memory keep agents effective across many steps.
9. **Debugging agents:** Trace logging, step-by-step inspection, and common failure patterns help you diagnose and fix agent issues.

In Module 15, you will extend these patterns to build systems with multiple agents that coordinate, delegate, and communicate to solve complex tasks.
