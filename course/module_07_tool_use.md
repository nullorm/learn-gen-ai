# Module 7: Tool Use

## Learning Objectives

- Understand what tools are and how LLMs use them to interact with the external world
- Define tools with Zod schemas including names, descriptions, and typed parameters
- Execute single and multi-step tool calls with proper result handling
- Use the Vercel AI SDK's `maxSteps` for automatic tool execution loops
- Implement robust error handling and validation for tool inputs and outputs
- Apply tool design patterns for granularity, naming, and composability
- Address security considerations including input validation, sandboxing, and allowlists

---

## Why Should I Care?

Without tools, an LLM is a brain in a jar. It can think, reason, and generate text, but it cannot check the weather, query a database, send an email, or read a file. Tools are what transform a language model from a text generator into an agent that can take action in the real world.

Tool use is the single most important pattern for building useful LLM applications. A customer support bot without tools can only generate generic responses. With tools, it can look up the customer's order, check inventory, initiate a refund, and update the ticket — all within a single conversation. A coding assistant without tools can only suggest code. With tools, it can read files, run tests, search documentation, and deploy changes.

The Vercel AI SDK makes tool use straightforward with Zod-based tool definitions and automatic execution loops. But the real skill is in tool design: deciding what tools to expose, how to scope their capabilities, and how to handle the inevitable edge cases when the model calls tools incorrectly. This module covers both the mechanics and the design thinking.

---

## Connection to Other Modules

- **Module 1 (Setup)** established the provider configuration. Tools work with any configured provider.
- **Module 3 (Structured Output)** introduced Zod schemas. Tools use the same schema system for parameter definitions.
- **Module 4 (Conversations)** built multi-turn conversations. Tool calls add a new message type to the conversation flow.
- **Module 6 (Streaming)** showed `streamText`. Streaming with tools involves pausing the stream during tool execution.
- **Modules 10-12 (Agents)** build heavily on tool use. Tools are the actions that agents take.

---

## Section 1: What are Tools?

### The Concept

A tool is a function that the LLM can decide to call. The model does not execute the function itself — it generates a structured request ("call this function with these arguments"), your code executes the function, and you send the result back to the model. The model then incorporates the result into its response.

```
User: "What's the weather in Tokyo?"
  ↓
Model thinks: "I need to check the weather. I'll call the get_weather tool."
  ↓
Model outputs: tool_call(name="get_weather", args={location: "Tokyo"})
  ↓
Your code: runs getWeather("Tokyo") → { temp: 22, condition: "Sunny" }
  ↓
Your code: sends tool result back to the model
  ↓
Model: "The weather in Tokyo is 22°C and sunny."
```

### The Tool Call Lifecycle

1. **Definition**: You describe available tools (name, description, parameter schema)
2. **Selection**: The model decides whether to call a tool and which one
3. **Invocation**: The model generates the tool name and arguments
4. **Execution**: Your code runs the actual function
5. **Result**: You send the function's return value back to the model
6. **Integration**: The model uses the result to formulate its response

> **Beginner Note:** The model never runs your code directly. It generates a JSON object describing which function to call and what arguments to pass. You are always in control of what actually executes. This is a critical security property.

> **Under the Hood: Native Function Calling**
> Each LLM provider has its own native function calling API — Anthropic uses `tool_use` content blocks, OpenAI/Groq use `tool_calls` in the assistant message, and Mistral uses a similar `tool_calls` format. The Vercel AI SDK's `tools` parameter is a unified abstraction over all of these. When you define a tool in the SDK, it translates your Zod schema into the provider's expected format, sends it with the request, and parses the structured response back into a consistent shape. You never need to work with the raw provider APIs directly, but knowing this mapping exists helps when debugging or reading provider documentation.

### Why Not Just Use Prompt Engineering?

You might think: "Can I just tell the model to output structured commands and parse them myself?" You can, but tool use is better because:

1. **Schema enforcement**: The model is constrained to valid parameter types and structures
2. **Provider optimization**: The model is specifically trained to generate well-formed tool calls
3. **Automatic flow**: The SDK handles the call-result-response loop
4. **Reliability**: Tool calls have significantly lower error rates than free-form JSON in prompts

---

## Section 2: Defining Tools with Zod

### Basic Tool Definition

The Vercel AI SDK uses Zod schemas to define tool parameters. Each tool has a name, description, parameter schema, and an optional execute function:

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const weatherTool = tool({
  description: 'Get the current weather for a location',
  parameters: z.object({
    location: z.string().describe('City name or coordinates'),
    units: z.enum(['celsius', 'fahrenheit']).optional().default('celsius').describe('Temperature units'),
  }),
  execute: async ({ location, units }) => {
    // In production, this would call a weather API
    console.log(`[Tool] Getting weather for ${location} in ${units}`)

    // Simulated response
    return {
      location,
      temperature: 22,
      units,
      condition: 'Partly cloudy',
      humidity: 65,
      windSpeed: 12,
    }
  },
})

const { text } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is the weather like in Tokyo right now?',
  tools: { getWeather: weatherTool },
  maxSteps: 2, // Allow model to call tool and then respond
})

console.log(text)
```

### The Anatomy of a Tool Definition

```typescript
import { tool } from 'ai'
import { z } from 'zod'

const myTool = tool({
  // Description: tells the model WHEN and WHY to use this tool
  // This is critical — a bad description means the model won't call the tool correctly
  description: 'Search for products in the catalog by name, category, or price range',

  // Parameters: Zod schema defining what arguments the tool accepts
  // The .describe() on each field helps the model understand what to provide
  parameters: z.object({
    query: z.string().describe('Search query string'),
    category: z.enum(['electronics', 'clothing', 'food', 'books']).optional().describe('Filter by product category'),
    minPrice: z.number().optional().describe('Minimum price in USD'),
    maxPrice: z.number().optional().describe('Maximum price in USD'),
    limit: z.int().min(1).max(50).optional().default(10).describe('Maximum number of results to return'),
  }),

  // Execute: the function that runs when the model calls this tool
  // This is optional — you can handle execution yourself
  execute: async ({ query, category, minPrice, maxPrice, limit }) => {
    // Your implementation here
    return { results: [], total: 0 }
  },
})
```

### Multiple Tools

Define multiple tools and let the model choose which to call:

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const tools = {
  calculator: tool({
    description: 'Perform mathematical calculations. Use this for any arithmetic, algebra, or mathematical operations.',
    parameters: z.object({
      expression: z.string().describe('Mathematical expression to evaluate, e.g., "2 + 3 * 4"'),
    }),
    execute: async ({ expression }) => {
      try {
        // Validate expression contains only safe math characters
        if (!/^[\d\s+\-*/().]+$/.test(expression)) {
          return { expression, result: null, error: `Error: Invalid expression "${expression}"` }
        }
        // WARNING: Function() is eval() in disguise. In production, use a math parser library like mathjs.
        // Simple evaluation for basic math
        const result = Function(`"use strict"; return (${expression})`)()
        return { expression, result, error: null }
      } catch (error) {
        return {
          expression,
          result: null,
          error: `Failed to evaluate: ${expression}`,
        }
      }
    },
  }),

  unitConverter: tool({
    description: 'Convert between units of measurement (length, weight, temperature)',
    parameters: z.object({
      value: z.number().describe('The value to convert'),
      fromUnit: z.string().describe('Source unit (e.g., "km", "lb", "celsius")'),
      toUnit: z.string().describe('Target unit (e.g., "miles", "kg", "fahrenheit")'),
    }),
    execute: async ({ value, fromUnit, toUnit }) => {
      // Simplified conversion table
      const conversions: Record<string, Record<string, number>> = {
        km: { miles: 0.621371 },
        miles: { km: 1.60934 },
        kg: { lb: 2.20462 },
        lb: { kg: 0.453592 },
      }

      const factor = conversions[fromUnit]?.[toUnit]
      if (!factor) {
        return { error: `Cannot convert ${fromUnit} to ${toUnit}` }
      }

      return {
        input: `${value} ${fromUnit}`,
        output: `${(value * factor).toFixed(4)} ${toUnit}`,
      }
    },
  }),

  dateCalculator: tool({
    description: 'Calculate dates: days between dates, add/subtract days from a date',
    parameters: z.object({
      operation: z.enum(['daysBetween', 'addDays']),
      date1: z.string().describe('First date in ISO format (YYYY-MM-DD)'),
      date2OrDays: z.string().describe('Second date (for daysBetween) or number of days to add (for addDays)'),
    }),
    execute: async ({ operation, date1, date2OrDays }) => {
      const d1 = new Date(date1)

      if (operation === 'daysBetween') {
        const d2 = new Date(date2OrDays)
        const diffMs = Math.abs(d2.getTime() - d1.getTime())
        const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24))
        return { days: diffDays, from: date1, to: date2OrDays }
      } else {
        const days = parseInt(date2OrDays)
        const result = new Date(d1.getTime() + days * 24 * 60 * 60 * 1000)
        return {
          originalDate: date1,
          daysAdded: days,
          resultDate: result.toISOString().split('T')[0],
        }
      }
    },
  }),
}

const { text } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt:
    'I ran a 10km race. How many miles is that? Also, if the race was on 2025-03-15 and I want to run another one 90 days later, what date would that be?',
  tools,
  maxSteps: 5,
})

console.log(text)
```

> **Advanced Note:** The descriptions you write for tools are essentially prompts. The model reads them to decide when and how to use each tool. Invest time in writing clear, specific descriptions with examples of when the tool should and should not be used. This is one of the highest-leverage improvements you can make.

---

## Section 3: Single Tool Call

### Basic Flow

The simplest tool use pattern: the model calls one tool, gets the result, and responds.

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// A simple lookup tool
const dictionaryTool = tool({
  description: 'Look up the definition of a word',
  parameters: z.object({
    word: z.string().describe('The word to look up'),
  }),
  execute: async ({ word }) => {
    // Simulated dictionary lookup
    const definitions: Record<string, string> = {
      serendipity: 'The occurrence of events by chance in a happy way',
      ephemeral: 'Lasting for a very short time',
      ubiquitous: 'Present, appearing, or found everywhere',
    }

    const definition = definitions[word.toLowerCase()]
    if (definition) {
      return { word, definition, found: true }
    }
    return { word, definition: null, found: false }
  },
})

const { text, toolCalls, toolResults } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What does "serendipity" mean?',
  tools: { dictionary: dictionaryTool },
  maxSteps: 2,
})

console.log('Response:', text)
console.log('Tool calls made:', toolCalls?.length ?? 0)
```

### Inspecting Tool Call Details

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const stockTool = tool({
  description: 'Get the current stock price for a ticker symbol',
  parameters: z.object({
    symbol: z.string().describe('Stock ticker symbol (e.g., AAPL, GOOGL)'),
  }),
  execute: async ({ symbol }) => {
    // Simulated stock data
    const prices: Record<string, number> = {
      AAPL: 178.5,
      GOOGL: 142.3,
      MSFT: 415.2,
      AMZN: 185.6,
    }

    const price = prices[symbol.toUpperCase()]
    if (price) {
      return { symbol: symbol.toUpperCase(), price, currency: 'USD', timestamp: new Date().toISOString() }
    }
    return { symbol, error: 'Symbol not found' }
  },
})

const result = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is the current price of Apple stock?',
  tools: { getStockPrice: stockTool },
  maxSteps: 2,
})

// Inspect all steps
for (const step of result.steps) {
  console.log(`Step ${result.steps.indexOf(step) + 1}:`)
  console.log(`  Finish reason: ${step.finishReason}`)

  if (step.toolCalls && step.toolCalls.length > 0) {
    for (const call of step.toolCalls) {
      console.log(`  Tool call: ${call.toolName}`)
      console.log(`  Arguments: ${JSON.stringify(call.args)}`)
    }
  }

  if (step.toolResults && step.toolResults.length > 0) {
    for (const toolResult of step.toolResults) {
      console.log(`  Tool result: ${JSON.stringify(toolResult.result)}`)
    }
  }

  if (step.text) {
    console.log(`  Text: ${step.text.slice(0, 100)}`)
  }
}
```

---

## Section 4: Tool Execution

### Manual Tool Execution (Without execute Function)

Sometimes you want to handle tool execution yourself instead of providing an `execute` function:

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Define tool WITHOUT execute function
const fileReadTool = tool({
  description: 'Read the contents of a file',
  parameters: z.object({
    path: z.string().describe('Path to the file to read'),
  }),
  // No execute function — we handle it manually
})

// First call: model decides to call the tool
const step1 = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Read the file at ./config.json and tell me what port the server runs on.',
  tools: { readFile: fileReadTool },
})

// Check if the model wants to call a tool
if (step1.finishReason === 'tool-calls' && step1.toolCalls.length > 0) {
  const toolCall = step1.toolCalls[0]
  console.log(`Model wants to call: ${toolCall.toolName}`)
  console.log(`With args: ${JSON.stringify(toolCall.args)}`)

  // Execute the tool ourselves
  let toolResult: unknown
  try {
    const filePath = toolCall.args.path
    // Validate the path before reading (security!)
    if (!filePath.startsWith('./') && !filePath.startsWith('/tmp/')) {
      toolResult = { error: 'Access denied: path outside allowed directories' }
    } else {
      const content = await Bun.file(filePath).text()
      toolResult = { content, size: content.length }
    }
  } catch (error) {
    toolResult = { error: `Failed to read file: ${error}` }
  }

  // Second call: send the tool result back
  const step2 = await generateText({
    model: mistral('mistral-small-latest'),
    messages: [
      { role: 'user', content: 'Read the file at ./config.json and tell me what port the server runs on.' },
      {
        role: 'assistant',
        content: step1.toolCalls.map(tc => ({
          type: 'tool-call' as const,
          toolCallId: tc.toolCallId,
          toolName: tc.toolName,
          args: tc.args,
        })),
      },
      {
        role: 'tool',
        content: step1.toolCalls.map(tc => ({
          type: 'tool-result' as const,
          toolCallId: tc.toolCallId,
          result: toolResult,
        })),
      },
    ],
    tools: { readFile: fileReadTool },
  })

  console.log('Response:', step2.text)
}
```

### Async Tool Execution

Tools often involve async operations — API calls, database queries, file operations:

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Tool that calls an external API
const geocodeTool = tool({
  description: 'Convert a place name to latitude/longitude coordinates',
  parameters: z.object({
    placeName: z.string().describe('Name of a place (city, landmark, etc.)'),
  }),
  execute: async ({ placeName }) => {
    // Simulate an API call with async delay
    await new Promise(resolve => setTimeout(resolve, 500))

    // In production, call a geocoding API like Nominatim or Google Maps
    const coordinates: Record<string, { lat: number; lon: number }> = {
      tokyo: { lat: 35.6762, lon: 139.6503 },
      paris: { lat: 48.8566, lon: 2.3522 },
      'new york': { lat: 40.7128, lon: -74.006 },
      sydney: { lat: -33.8688, lon: 151.2093 },
    }

    const result = coordinates[placeName.toLowerCase()]
    if (result) {
      return { placeName, ...result, found: true }
    }
    return { placeName, found: false, error: 'Location not found' }
  },
})

const { text } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What are the coordinates of Tokyo and Paris?',
  tools: { geocode: geocodeTool },
  maxSteps: 5,
})

console.log(text)
```

> **Beginner Note:** Tool execution happens on your server, not in the model. The model only generates the request (function name + arguments). Your `execute` function runs the actual logic. This means you can do anything a normal function can do: call APIs, query databases, read files, run computations.

---

## Section 5: Multi-Step Tool Loops

### The Pattern

Some questions require multiple tool calls in sequence. The model calls one tool, examines the result, decides it needs more information, calls another tool, and so on until it has enough information to answer.

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Simulated database of employees
const employees: Record<string, { name: string; department: string; managerId: string | null; salary: number }> = {
  E001: { name: 'Alice', department: 'Engineering', managerId: null, salary: 150000 },
  E002: { name: 'Bob', department: 'Engineering', managerId: 'E001', salary: 120000 },
  E003: { name: 'Charlie', department: 'Marketing', managerId: 'E001', salary: 110000 },
  E004: { name: 'Diana', department: 'Engineering', managerId: 'E002', salary: 100000 },
  E005: { name: 'Eve', department: 'Marketing', managerId: 'E003', salary: 95000 },
}

const tools = {
  getEmployee: tool({
    description: 'Get employee details by their ID',
    parameters: z.object({
      employeeId: z.string().describe('Employee ID (e.g., E001)'),
    }),
    execute: async ({ employeeId }) => {
      const emp = employees[employeeId]
      if (emp) return { id: employeeId, ...emp }
      return { error: `Employee ${employeeId} not found` }
    },
  }),

  searchEmployees: tool({
    description: 'Search for employees by name or department',
    parameters: z.object({
      query: z.string().describe('Search query (name or department)'),
    }),
    execute: async ({ query }) => {
      const results = Object.entries(employees)
        .filter(
          ([_, emp]) =>
            emp.name.toLowerCase().includes(query.toLowerCase()) ||
            emp.department.toLowerCase().includes(query.toLowerCase())
        )
        .map(([id, emp]) => ({ id, name: emp.name, department: emp.department }))

      return { results, count: results.length }
    },
  }),

  getDirectReports: tool({
    description: 'Get all direct reports for a manager',
    parameters: z.object({
      managerId: z.string().describe('Manager employee ID'),
    }),
    execute: async ({ managerId }) => {
      const reports = Object.entries(employees)
        .filter(([_, emp]) => emp.managerId === managerId)
        .map(([id, emp]) => ({ id, name: emp.name, department: emp.department }))

      return { managerId, reports, count: reports.length }
    },
  }),
}

// This question requires multiple tool calls:
// 1. Search for Alice to get her ID
// 2. Get Alice's direct reports to find their IDs
// 3. Get details of each direct report for salary info
const { text, steps } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Who reports to Alice? What is the total salary budget for her direct reports?',
  tools,
  maxSteps: 10,
})

console.log('Final answer:', text)
console.log(`\nSteps taken: ${steps.length}`)

for (const [i, step] of steps.entries()) {
  console.log(`\nStep ${i + 1}: ${step.finishReason}`)
  if (step.toolCalls) {
    for (const tc of step.toolCalls) {
      console.log(`  Called: ${tc.toolName}(${JSON.stringify(tc.args)})`)
    }
  }
}
```

### Parallel Tool Calls

Some models can call multiple tools simultaneously when the calls are independent:

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const tools = {
  getWeather: tool({
    description: 'Get current weather for a city',
    parameters: z.object({ city: z.string() }),
    execute: async ({ city }) => {
      await new Promise(r => setTimeout(r, 300)) // Simulate API delay
      const data: Record<string, { temp: number; condition: string }> = {
        tokyo: { temp: 22, condition: 'Sunny' },
        london: { temp: 12, condition: 'Rainy' },
        'new york': { temp: 18, condition: 'Cloudy' },
      }
      return data[city.toLowerCase()] ?? { error: 'City not found' }
    },
  }),

  getTime: tool({
    description: 'Get the current local time in a city',
    parameters: z.object({ city: z.string() }),
    execute: async ({ city }) => {
      const timezones: Record<string, number> = {
        tokyo: 9,
        london: 0,
        'new york': -5,
      }
      const offset = timezones[city.toLowerCase()]
      if (offset === undefined) return { error: 'City not found' }

      const utc = new Date()
      const local = new Date(utc.getTime() + offset * 3600000)
      return { city, localTime: local.toISOString(), utcOffset: offset }
    },
  }),
}

// The model might call getWeather and getTime in parallel for each city
const { text, steps } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Compare the weather and current time in Tokyo, London, and New York.',
  tools,
  maxSteps: 10,
})

console.log(text)

// Check if parallel calls were made
for (const step of steps) {
  if (step.toolCalls && step.toolCalls.length > 1) {
    console.log(`\nParallel calls in step: ${step.toolCalls.map(tc => tc.toolName).join(', ')}`)
  }
}
```

---

## Section 6: maxSteps and Automatic Loops

### How maxSteps Works

The `maxSteps` parameter controls the maximum number of LLM call iterations. Each "step" is one call to the model. Without `maxSteps`, the model makes a single call and if it wants to use a tool, it stops with `finishReason: 'tool-calls'` and you must handle the loop yourself.

With `maxSteps`, the SDK automatically:

1. Calls the model
2. If the model makes tool calls, executes the tools
3. Sends the results back to the model
4. Repeats until the model produces a text response or `maxSteps` is reached

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Without maxSteps: single call, manual handling
const result1 = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is 15 * 23?',
  tools: {
    calculator: tool({
      description: 'Calculate a math expression',
      parameters: z.object({ expression: z.string() }),
      execute: async ({ expression }) => {
        // Validate expression contains only safe math characters
        if (!/^[\d\s+\-*/().]+$/.test(expression)) {
          return { result: null, error: `Error: Invalid expression "${expression}"` }
        }
        // WARNING: Function() is eval() in disguise. In production, use a math parser library like mathjs.
        // Simple math evaluation using Function constructor
        const result = Function(`"use strict"; return (${expression})`)()
        return { result }
      },
    }),
  },
  // No maxSteps: model returns tool_call, you must handle it yourself
})

console.log('Finish reason:', result1.finishReason)
// 'tool-calls' — model wants to call calculator but cannot proceed

// With maxSteps: automatic loop
const result2 = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is 15 * 23?',
  tools: {
    calculator: tool({
      description: 'Calculate a math expression',
      parameters: z.object({ expression: z.string() }),
      execute: async ({ expression }) => {
        // Validate expression contains only safe math characters
        if (!/^[\d\s+\-*/().]+$/.test(expression)) {
          return { result: null, error: `Error: Invalid expression "${expression}"` }
        }
        // WARNING: Function() is eval() in disguise. In production, use a math parser library like mathjs.
        const result = Function(`"use strict"; return (${expression})`)()
        return { result }
      },
    }),
  },
  maxSteps: 3, // Allow up to 3 model calls
})

console.log('Finish reason:', result2.finishReason)
// 'stop' — model got the result and formulated a text response
console.log('Response:', result2.text)
// "15 * 23 = 345"
```

### Choosing maxSteps

```typescript
// Guidelines for maxSteps:
//
// maxSteps: 1 — No tool use (tools are shown but model cannot act on results)
// maxSteps: 2 — Single tool call + response (most common)
// maxSteps: 3-5 — Multi-step reasoning (search → read → analyze)
// maxSteps: 5-10 — Complex workflows (iterative refinement)
// maxSteps: 10+ — Agent-like behavior (careful: cost and latency add up)

// Each step is a full model call, so:
// - Cost scales linearly with steps
// - Latency scales linearly with steps
// - Each step includes the full conversation history
```

### Monitoring Step Execution

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const { text, steps } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is the square root of 144, and then multiply that by 3?',
  tools: {
    calculator: tool({
      description: 'Evaluate a mathematical expression',
      parameters: z.object({
        expression: z.string().describe('Math expression to evaluate'),
      }),
      execute: async ({ expression }) => {
        // Validate expression contains only safe math characters
        if (!/^[\d\s+\-*/().]+$/.test(expression)) {
          return { expression, result: null, error: `Error: Invalid expression "${expression}"` }
        }
        // WARNING: Function() is eval() in disguise. In production, use a math parser library like mathjs.
        const result = Function(`"use strict"; return (${expression})`)()
        return { expression, result }
      },
    }),
  },
  maxSteps: 5,
})

// Analyze the steps
for (const [i, step] of steps.entries()) {
  console.log(`\n--- Step ${i + 1} ---`)
  console.log(`Finish reason: ${step.finishReason}`)

  if (step.toolCalls?.length) {
    for (const tc of step.toolCalls) {
      console.log(`Tool: ${tc.toolName}(${JSON.stringify(tc.args)})`)
    }
  }

  if (step.toolResults?.length) {
    for (const tr of step.toolResults) {
      console.log(`Result: ${JSON.stringify(tr.result)}`)
    }
  }

  if (step.text) {
    console.log(`Text: ${step.text.slice(0, 200)}`)
  }

  if (step.usage) {
    console.log(`Tokens: ${step.usage.inputTokens} in, ${step.usage.outputTokens} out`)
  }
}

console.log(`\nFinal answer: ${text}`)
console.log(`Total steps: ${steps.length}`)
```

> **Advanced Note:** Be careful with high `maxSteps` values. Each step sends the entire conversation history (including all previous tool calls and results) to the model, so token usage grows quadratically. A 10-step tool loop where each tool returns 500 tokens means the last step sends ~5000 tokens of tool results alone, on top of the original prompt and all intermediate model outputs.

---

## Section 7: Error Handling in Tools

### Tool Execution Errors

Tools can fail for many reasons: network errors, invalid input, permission denied, rate limits. How you handle these errors determines the user experience.

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const robustTools = {
  fetchUrl: tool({
    description: 'Fetch the content of a web page',
    parameters: z.object({
      url: z.url().describe('The URL to fetch'),
    }),
    execute: async ({ url }) => {
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 10_000)

        const response = await fetch(url, { signal: controller.signal })
        clearTimeout(timeout)

        if (!response.ok) {
          return {
            error: `HTTP ${response.status}: ${response.statusText}`,
            url,
            success: false,
          }
        }

        const text = await response.text()
        // Limit response size to avoid token explosion
        const truncated = text.slice(0, 5000)

        return {
          url,
          content: truncated,
          contentLength: text.length,
          truncated: text.length > 5000,
          success: true,
        }
      } catch (error) {
        if (error instanceof Error) {
          if (error.name === 'AbortError') {
            return { error: 'Request timed out after 10 seconds', url, success: false }
          }
          return { error: error.message, url, success: false }
        }
        return { error: 'Unknown error', url, success: false }
      }
    },
  }),
}

const { text } = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'What is on the homepage of example.com?',
  tools: robustTools,
  maxSteps: 3,
})

console.log(text)
```

### Error Reporting Patterns

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Pattern 1: Return errors as structured data (recommended)
// The model can read the error and respond intelligently
const safeTool = tool({
  description: 'Example tool with safe error handling',
  parameters: z.object({ input: z.string() }),
  execute: async ({ input }) => {
    try {
      const result = await processInput(input)
      return { success: true, data: result }
    } catch (error) {
      return { success: false, error: String(error) }
    }
  },
})

// Pattern 2: Throw errors (stops the tool loop)
// Use only for unrecoverable errors
const strictTool = tool({
  description: 'Example tool that throws on invalid input',
  parameters: z.object({ input: z.string() }),
  execute: async ({ input }) => {
    if (!isValid(input)) {
      throw new Error('Invalid arguments — this tool cannot process this request')
    }
    return await processInput(input)
  },
})

// Helper stubs for the example
async function processInput(input: string) {
  return { processed: input }
}
function isValid(input: string) {
  return input.length > 0
}
```

> **Beginner Note:** Always prefer returning errors as data (Pattern 1) over throwing exceptions (Pattern 2). When you return an error as data, the model reads it and can try a different approach, rephrase, or explain the failure to the user. When you throw an exception, the entire tool loop stops and the user gets a generic error message.

### Validation Before Execution

```typescript
import { tool } from 'ai'
import { z } from 'zod'

const databaseQueryTool = tool({
  description: 'Run a read-only SQL query against the database',
  parameters: z.object({
    query: z.string().describe('SQL SELECT query'),
  }),
  execute: async ({ query }) => {
    // Validate the query before running it
    const normalizedQuery = query.trim().toLowerCase()

    // Block dangerous operations
    const dangerousKeywords = ['drop', 'delete', 'update', 'insert', 'alter', 'truncate']
    for (const keyword of dangerousKeywords) {
      if (normalizedQuery.includes(keyword)) {
        return {
          error: `Query rejected: contains forbidden keyword '${keyword}'. Only SELECT queries are allowed.`,
        }
      }
    }

    // Verify it starts with SELECT
    if (!normalizedQuery.startsWith('select')) {
      return {
        error: 'Query rejected: must start with SELECT. Only read operations are allowed.',
      }
    }

    // Execute the validated query
    try {
      // const results = await database.query(query);
      const results = [{ id: 1, name: 'Example' }] // Simulated
      return { query, results, rowCount: results.length }
    } catch (error) {
      return { error: `Query execution failed: ${error}` }
    }
  },
})
```

---

## Section 8: Tool Design Patterns

### Granularity: Fine vs Coarse Tools

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Too fine-grained: model needs multiple calls for simple tasks
const tooFineGrained = {
  openFile: tool({
    description: 'Open a file handle',
    parameters: z.object({ path: z.string() }),
    execute: async () => ({ handle: 'file_1' }),
  }),
  readLine: tool({
    description: 'Read one line from an open file',
    parameters: z.object({ handle: z.string() }),
    execute: async () => ({ line: 'data' }),
  }),
  closeFile: tool({
    description: 'Close a file handle',
    parameters: z.object({ handle: z.string() }),
    execute: async () => ({ closed: true }),
  }),
}

// Too coarse-grained: model cannot do specific tasks
const tooCoarseGrained = {
  doEverything: tool({
    description: 'Perform any file operation',
    parameters: z.object({
      action: z.string(),
      path: z.string(),
      data: z.string().optional(),
    }),
    execute: async () => ({ result: 'done' }),
  }),
}

// Just right: task-level granularity
const justRight = {
  readFile: tool({
    description: 'Read the entire contents of a file',
    parameters: z.object({
      path: z.string().describe('Path to the file'),
    }),
    execute: async ({ path }) => {
      const content = await Bun.file(path).text()
      return { path, content, size: content.length }
    },
  }),
  writeFile: tool({
    description: 'Write content to a file (creates or overwrites)',
    parameters: z.object({
      path: z.string().describe('Path to the file'),
      content: z.string().describe('Content to write'),
    }),
    execute: async ({ path, content }) => {
      await Bun.write(path, content)
      return { path, bytesWritten: content.length }
    },
  }),
  listDirectory: tool({
    description: 'List files and directories in a path',
    parameters: z.object({
      path: z.string().describe('Directory path'),
    }),
    execute: async ({ path }) => {
      const { readdir } = await import('node:fs/promises')
      const entries = await readdir(path, { withFileTypes: true })
      return {
        path,
        entries: entries.map(e => ({
          name: e.name,
          type: e.isDirectory() ? 'directory' : 'file',
        })),
      }
    },
  }),
}
```

### Naming Conventions

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Good tool names: verb + noun, clear and specific
const goodNames = {
  searchProducts: tool({
    description: 'Search the product catalog',
    parameters: z.object({ query: z.string() }),
    execute: async () => ({ results: [] }),
  }),
  getOrderStatus: tool({
    description: 'Get the status of an order by ID',
    parameters: z.object({ orderId: z.string() }),
    execute: async () => ({ status: 'shipped' }),
  }),
  createTicket: tool({
    description: 'Create a support ticket',
    parameters: z.object({ title: z.string(), body: z.string() }),
    execute: async () => ({ ticketId: '123' }),
  }),
  sendEmail: tool({
    description: 'Send an email',
    parameters: z.object({ to: z.string(), subject: z.string(), body: z.string() }),
    execute: async () => ({ sent: true }),
  }),
  calculateTax: tool({
    description: 'Calculate tax for an amount',
    parameters: z.object({ amount: z.number(), region: z.string() }),
    execute: async () => ({ tax: 0 }),
  }),
}
```

### Description Best Practices

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Good: specific about what, when, and limitations
const goodTool = tool({
  description: `Search the product catalog by keyword, category, or price range.
Returns up to 20 matching products with name, price, and rating.
Use this when the user asks about products, availability, or pricing.
Does NOT check inventory — use checkInventory for stock availability.`,
  parameters: z.object({
    query: z.string().describe('Search keywords'),
  }),
  execute: async ({ query }) => ({ results: [] }),
})

// Bad: vague, no guidance on when to use
const badTool = tool({
  description: 'Search for stuff',
  parameters: z.object({
    q: z.string(), // No description, unclear name
  }),
  execute: async ({ q }) => ({ results: [] }),
})
```

### Composable Tool Sets

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Create tool sets that work together as a coherent unit
function createCRUDTools<T extends z.ZodType>(entityName: string, schema: T, store: Map<string, z.infer<T>>) {
  return {
    [`list${entityName}s`]: tool({
      description: `List all ${entityName.toLowerCase()}s. Returns an array of all entries.`,
      parameters: z.object({}),
      execute: async () => {
        const entries = Array.from(store.entries()).map(([id, data]) => ({
          id,
          ...data,
        }))
        return { entries, count: entries.length }
      },
    }),

    [`get${entityName}`]: tool({
      description: `Get a specific ${entityName.toLowerCase()} by ID.`,
      parameters: z.object({
        id: z.string().describe(`${entityName} ID`),
      }),
      execute: async ({ id }) => {
        const data = store.get(id)
        if (!data) return { error: `${entityName} not found: ${id}` }
        return { id, ...data }
      },
    }),

    [`create${entityName}`]: tool({
      description: `Create a new ${entityName.toLowerCase()}.`,
      parameters: schema as any,
      execute: async (data: z.infer<T>) => {
        const id = crypto.randomUUID().slice(0, 8)
        store.set(id, data)
        return { id, ...data, created: true }
      },
    }),

    [`delete${entityName}`]: tool({
      description: `Delete a ${entityName.toLowerCase()} by ID.`,
      parameters: z.object({
        id: z.string().describe(`${entityName} ID`),
      }),
      execute: async ({ id }) => {
        const existed = store.delete(id)
        return { id, deleted: existed }
      },
    }),
  }
}

// Usage
const todoSchema = z.object({
  title: z.string().describe('Todo title'),
  priority: z.enum(['low', 'medium', 'high']).describe('Priority level'),
  done: z.boolean().default(false),
})

const todoStore = new Map<string, z.infer<typeof todoSchema>>()
const todoTools = createCRUDTools('Todo', todoSchema, todoStore)
```

> **Advanced Note:** In production, limit the number of tools exposed to the model. Research shows that model accuracy for tool selection degrades with more than ~15-20 tools. If you have many tools, use a "tool router" pattern: a lightweight model first selects which tool category is relevant, then the main model works with only the relevant tools.

---

## Section 9: Security Considerations

### Input Validation

The model generates tool arguments, but you should never trust them blindly:

```typescript
import { tool } from 'ai'
import { z } from 'zod'
import { resolve, normalize } from 'node:path'

// Path traversal protection
const safeFileRead = tool({
  description: 'Read a file from the project directory',
  parameters: z.object({
    path: z.string().describe('Relative path within the project directory'),
  }),
  execute: async ({ path: inputPath }) => {
    const projectRoot = '/home/user/project'

    // Normalize and resolve the path
    const resolvedPath = resolve(projectRoot, normalize(inputPath))

    // Ensure path is within the project directory
    if (!resolvedPath.startsWith(projectRoot)) {
      return { error: 'Access denied: path outside project directory' }
    }

    // Block sensitive files
    const blockedPatterns = ['.env', 'secrets', 'credentials', '.key', '.pem']
    const lowerPath = resolvedPath.toLowerCase()
    for (const pattern of blockedPatterns) {
      if (lowerPath.includes(pattern)) {
        return { error: `Access denied: cannot read files matching pattern '${pattern}'` }
      }
    }

    try {
      const content = await Bun.file(resolvedPath).text()
      return { path: inputPath, content: content.slice(0, 10000) } // Limit size
    } catch {
      return { error: `File not found: ${inputPath}` }
    }
  },
})
```

### Allowlists

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Only allow specific, pre-approved operations
const allowedCommands = ['git status', 'git log --oneline -10', 'npm test', 'npm run lint', 'ls -la'] as const

const safeShellTool = tool({
  description: 'Run a whitelisted shell command',
  parameters: z.object({
    command: z.enum(allowedCommands).describe('The command to execute (must be from the allowed list)'),
  }),
  execute: async ({ command }) => {
    const parts = command.split(' ')
    const proc = Bun.spawn(parts, {
      cwd: '/home/user/project',
      stdout: 'pipe',
      stderr: 'pipe',
    })

    const stdout = await new Response(proc.stdout).text()
    const stderr = await new Response(proc.stderr).text()
    const exitCode = await proc.exited

    return {
      command,
      stdout: stdout.slice(0, 5000),
      stderr: stderr.slice(0, 1000),
      exitCode,
    }
  },
})
```

### Rate Limiting Tools

```typescript
import { tool } from 'ai'
import { z } from 'zod'

class ToolRateLimiter {
  private calls: Map<string, number[]> = new Map()
  private maxCalls: number
  private windowMs: number

  constructor(maxCalls: number, windowMs: number) {
    this.maxCalls = maxCalls
    this.windowMs = windowMs
  }

  check(toolName: string): boolean {
    const now = Date.now()
    const calls = this.calls.get(toolName) ?? []

    // Remove old calls outside the window
    const recentCalls = calls.filter(t => now - t < this.windowMs)
    this.calls.set(toolName, recentCalls)

    if (recentCalls.length >= this.maxCalls) {
      return false // Rate limited
    }

    recentCalls.push(now)
    return true
  }
}

const rateLimiter = new ToolRateLimiter(10, 60_000) // 10 calls per minute

function rateLimitedTool(name: string, config: Parameters<typeof tool>[0]): ReturnType<typeof tool> {
  const originalExecute = config.execute

  return tool({
    ...config,
    execute: async (args: any) => {
      if (!rateLimiter.check(name)) {
        return { error: 'Rate limit exceeded. Please wait before trying again.' }
      }

      if (originalExecute) {
        return originalExecute(args)
      }
      return { error: 'No execute function defined' }
    },
  })
}
```

### Sandboxing and Audit Logging

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Key security principles for tool use:
//
// 1. NEVER execute arbitrary code from tool arguments
// 2. ALWAYS validate and sanitize inputs
// 3. Use the principle of least privilege
// 4. Log all tool calls for audit
// 5. Implement rate limiting
// 6. Set resource limits (timeout, memory, output size)

// Audit logging wrapper
function auditedTool(name: string, config: Parameters<typeof tool>[0]): ReturnType<typeof tool> {
  const originalExecute = config.execute

  return tool({
    ...config,
    execute: async (args: any) => {
      const startTime = Date.now()
      console.log(`[AUDIT] Tool: ${name} | Args: ${JSON.stringify(args)} | Time: ${new Date().toISOString()}`)

      try {
        const result = originalExecute ? await originalExecute(args) : null
        const duration = Date.now() - startTime
        console.log(`[AUDIT] Tool: ${name} | Duration: ${duration}ms | Success`)
        return result
      } catch (error) {
        const duration = Date.now() - startTime
        console.log(`[AUDIT] Tool: ${name} | Duration: ${duration}ms | Error: ${error}`)
        throw error
      }
    },
  })
}
```

> **Beginner Note:** Security in tool use is about defense in depth. The model might generate unexpected or malicious arguments — not because it is adversarial, but because it is probabilistic and can make mistakes. Validate everything, limit everything, and log everything.

> **Advanced Note:** For high-security applications, consider running tool execution in a separate process or container with restricted permissions. Tools that interact with databases should use read-only connections unless writes are explicitly required. Tools that access the filesystem should use chroot or similar isolation.

> **Looking Ahead: Model Context Protocol (MCP)** — In this module, you defined tools by hand in your application code. Anthropic's [Model Context Protocol](https://modelcontextprotocol.io) is an open standard that lets tools be discovered dynamically from external servers. Instead of hardcoding a `weatherTool`, your agent connects to an MCP server and discovers available tools at runtime. The Vercel AI SDK supports this via `@ai-sdk/mcp`:
>
> ```typescript
> import { createMCPClient } from '@ai-sdk/mcp'
> const client = await createMCPClient({
>   transport: { type: 'http', url: 'http://localhost:3000/mcp' },
> })
> const tools = await client.tools() // discover tools dynamically
> const result = await generateText({ model, tools, prompt })
> await client.close()
> ```
>
> This means your agents can gain new capabilities without code changes — just connect to a new MCP server.

> **Local Alternative (Ollama):** Tool calling works with `ollama('qwen3.5')` — Qwen 3.5 has native function calling support. If tool calling fails with your local model, try the cloud variant (`ollama('qwen3.5:cloud')`) for better reliability. The multi-step tool loop and `maxSteps` work identically.

---

## Summary

In this module, you learned:

1. **What tools are:** Tools let LLMs interact with the external world by generating structured function calls that your application executes and returns results from.
2. **Tool definitions with Zod:** How to define tools with names, descriptions, and typed parameter schemas that guide the model toward correct usage.
3. **The tool call lifecycle:** The model proposes a tool call, your code executes it, and the result is sent back — the model never runs code directly.
4. **Single and multi-step tool calls:** How to handle one-shot tool use and iterative loops where the model chains multiple tool calls to accomplish complex tasks.
5. **maxSteps and automatic loops:** The Vercel AI SDK's `maxSteps` parameter automates the tool execution loop, letting the model call tools repeatedly until it has a final answer.
6. **Error handling:** How to catch tool execution errors, report them back to the model in a structured way, and validate inputs before execution.
7. **Tool design patterns:** Principles for granularity, naming, composability, and deciding between fine-grained and coarse-grained tool interfaces.
8. **Security considerations:** Input validation, sandboxing, allowlists, and running tool execution with restricted permissions to prevent misuse.

In Module 8, you will learn about embeddings — the vector representations that enable semantic search and form the foundation of RAG systems.

---

## Quiz

### Question 1 (Easy)

In the tool use lifecycle, who executes the actual function?

A) The LLM model
B) The model provider's API server
C) Your application code
D) The Vercel AI SDK runtime

**Answer: C**

The model generates a structured request (tool name + arguments) but never executes code directly. Your application code receives the tool call, runs the function in your environment, and sends the result back to the model. This separation is a critical security property — you always control what runs.

---

### Question 2 (Medium)

What does `maxSteps: 5` mean in a `generateText` call with tools?

A) The model can call a maximum of 5 tools total
B) The SDK will make up to 5 calls to the model, automatically handling tool calls and results
C) Each tool can be called at most 5 times
D) The model will generate at most 5 sentences

**Answer: B**

`maxSteps` controls the maximum number of model call iterations. The SDK will call the model, execute any requested tools, send results back, and repeat — up to 5 times. This allows multi-step tool chains where each step can involve one or more tool calls.

---

### Question 3 (Medium)

Why should tool errors be returned as data rather than thrown as exceptions?

A) Exceptions are not supported in async functions
B) Returning errors lets the model read them and try a different approach
C) Thrown exceptions cost more tokens
D) The AI SDK does not catch exceptions

**Answer: B**

When you return an error as structured data (e.g., `{ error: "File not found" }`), the model receives it as a tool result and can reason about it — trying an alternative approach, asking the user for clarification, or explaining the issue. When you throw an exception, the tool loop typically stops entirely, giving the model no chance to recover.

---

### Question 4 (Easy)

What is the recommended granularity level for tool design?

A) One tool per line of code (very fine)
B) One tool per user action or task (task-level)
C) One tool per application module (very coarse)
D) A single tool that handles everything

**Answer: B**

Task-level granularity aligns tools with meaningful user-facing actions. Too fine-grained (file open/read line/close) forces the model to orchestrate many calls for simple tasks. Too coarse (one do-everything tool) gives the model no meaningful choice and makes parameters complex. Task-level tools like `readFile`, `searchProducts`, `sendEmail` hit the sweet spot.

---

### Question 5 (Hard)

What is the primary security concern with tool use?

A) The model might generate incorrect text responses
B) The model-generated tool arguments should not be trusted without validation
C) Tools use too many tokens
D) Tool calls are slower than direct API calls

**Answer: B**

The model generates tool arguments probabilistically — it can produce unexpected, malformed, or potentially dangerous values. Path traversal attacks, SQL injection, and excessive resource consumption are all possible if you trust arguments without validation. Always validate, sanitize, and restrict tool inputs.

---

## Exercises

### Exercise 1: Multi-Tool Assistant

Build an interactive assistant with three tools: calculator, web search (simulated), and file reader.

**Requirements:**

1. Define three tools with complete Zod schemas:
   - `calculator`: evaluates mathematical expressions safely (use a parser, not eval)
   - `searchWeb`: simulates web search (returns predefined results for known queries)
   - `readFile`: reads files from a specified directory (with path validation)
2. Use `maxSteps: 10` for multi-step reasoning
3. Log each tool call with timing information
4. Handle errors gracefully (return structured error objects)
5. Implement path validation for `readFile` (restrict to a project directory)
6. Build a conversation loop that maintains context between turns

**Starter code:**

```typescript
import { generateText, tool } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

const PROJECT_DIR = './workspace'

const tools = {
  calculator: tool({
    description: 'TODO: Write a good description',
    parameters: z.object({
      // TODO: Define parameters
    }),
    execute: async args => {
      // TODO: Implement safe math evaluation (no eval!)
      // Consider using a simple parser for +, -, *, /, ^, sqrt, etc.
    },
  }),

  searchWeb: tool({
    description: 'TODO: Write a good description',
    parameters: z.object({
      // TODO: Define parameters
    }),
    execute: async args => {
      // TODO: Return simulated search results
      // Create a map of queries to predefined results
    },
  }),

  readFile: tool({
    description: 'TODO: Write a good description',
    parameters: z.object({
      // TODO: Define parameters
    }),
    execute: async args => {
      // TODO: Implement with path validation
      // Block paths outside PROJECT_DIR
      // Block sensitive files (.env, etc.)
      // Limit file size in response
    },
  }),
}

// TODO: Implement conversation loop
// TODO: Log tool calls and timing
// TODO: Handle multi-step reasoning
```

**Test scenarios:**

- "What is the square root of 256 multiplied by PI?"
- "Search for TypeScript best practices and then read the file ./workspace/config.json"
- "Read the project structure and tell me what files exist"

### Exercise 2: Tool Security Audit

Create a security test suite that validates your tool implementations handle adversarial inputs correctly.

**Requirements:**

1. Write test cases for path traversal attacks on `readFile` (e.g., paths with `..` components, absolute paths outside the project)
2. Write test cases for injection attacks on `calculator` (e.g., inputs that attempt to access process globals)
3. Write test cases for excessive resource consumption (e.g., very large inputs)
4. Verify that rate limiting works correctly
5. Verify that audit logging captures all tool calls

```typescript
import { describe, test, expect } from 'bun:test'

describe('Tool Security', () => {
  describe('readFile', () => {
    test('blocks path traversal with ..', async () => {
      // TODO: Verify that paths containing '..' are rejected
    })

    test('blocks absolute paths outside project', async () => {
      // TODO: Verify that absolute paths are rejected
    })

    test('blocks sensitive file patterns', async () => {
      // TODO: Verify .env, credentials files are blocked
    })
  })

  describe('calculator', () => {
    test('blocks code injection attempts', async () => {
      // TODO: Verify that non-mathematical expressions are rejected
    })

    test('handles extremely large numbers', async () => {
      // TODO: Verify graceful handling of oversized inputs
    })
  })

  describe('rate limiting', () => {
    test('enforces call limits', async () => {
      // TODO: Verify rate limiter blocks after threshold
    })
  })
})
```

**Evaluation criteria:**

- All path traversal attempts are blocked
- All injection attempts are blocked
- Error messages are informative but do not leak paths or system info
- Rate limiting kicks in after the configured threshold
- Audit log contains entries for every tool call attempt
