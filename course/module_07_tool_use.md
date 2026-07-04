# Module 7: Tool Use

## Learning Objectives

- Understand what tools are and how LLMs use them to interact with the external world
- Define tools with Zod schemas including names, descriptions, and typed parameters
- Execute single and multi-step tool calls with proper result handling
- Use the Vercel AI SDK's `stopWhen` with `isStepCount()` for automatic tool execution loops
- Implement robust error handling and validation for tool inputs and outputs
- Apply tool design patterns for granularity, naming, and composability
- Address security considerations including input validation, sandboxing, and allowlists

> *Module 7 is part of **Part II: Core Patterns**, building toward the **Core Patterns** badge — tools are what turn a chatbot into something that acts.*

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
- **Modules 16-17 (Agents)** build heavily on tool use. Tools are the actions that agents take.

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

> **Advanced Note: Native Function Calling**
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

The Vercel AI SDK uses Zod schemas to define tool parameters. Each tool has a name, description, parameter schema, and an optional execute function. The key imports are:

```typescript
import { generateText, tool, isStepCount } from 'ai'
import { z } from 'zod'
```

A tool definition has three parts:

- **`description`**: A string that tells the model when and why to use this tool. This is critical — a bad description means the model will not call the tool correctly.
- **`inputSchema`**: A Zod schema defining what arguments the tool accepts. Use `.describe()` on each field to help the model understand what to provide.
- **`execute`**: An async function that runs when the model calls this tool. This is optional — you can handle execution manually instead.

Here is the minimal shape:

```typescript
const myTool = tool({
  description: 'Get the current weather for a location',
  inputSchema: z.object({
    location: z.string().describe('City name or coordinates'),
  }),
  execute: async ({ location }) => {
    return { location, temperature: 22 }
  },
})
```

To use tools with `generateText`, pass them in the `tools` object and set `stopWhen: isStepCount(N)` to allow the model to call tools and then respond:

```typescript
const { text } = await generateText({
  model: yourModel,
  prompt: 'What is the weather like in Tokyo?',
  tools: { getWeather: myTool },
  stopWhen: isStepCount(2),
})
```

Build a weather tool that accepts a `location` (string) and optional `units` parameter (enum of `'celsius'` or `'fahrenheit'`, defaulting to `'celsius'`). The execute function should return a simulated weather object with `location`, `temperature`, `units`, `condition`, `humidity`, and `windSpeed`. Then wire it up with `generateText` and `stopWhen: isStepCount(2)`.

### The Anatomy of a Tool Definition

A more complete tool definition uses rich Zod schemas with optional fields, enums, numeric ranges, and descriptions on every parameter. Think about what a product search tool would need: a query string, an optional category enum, optional min/max price numbers, and a result limit with `z.int().min(1).max(50)`.

### Multiple Tools

You can define multiple tools in a single `tools` object and let the model choose which to call. Build a toolkit with three tools:

1. **`calculator`**: Accepts a math `expression` string. The execute function should validate that the expression contains only safe math characters (`/^[\d\s+\-*/().]+$/`), then evaluate it. What should happen if the expression contains non-math characters? Return a structured error, not a thrown exception.

   > **Gotcha:** `Function()` is `eval()` in disguise. In production, use a math parser library like `mathjs`.

2. **`unitConverter`**: Accepts a `value` (number), `fromUnit` (string), and `toUnit` (string). Use a conversion lookup table. What should the tool return when a conversion is not found?

3. **`dateCalculator`**: Accepts an `operation` enum (`'daysBetween'` or `'addDays'`), a `date1` string, and a `date2OrDays` string. For `daysBetween`, calculate the difference in days between two dates. For `addDays`, add N days to a date and return the result.

Pass all three tools to `generateText` with `stopWhen: isStepCount(5)` and a prompt that requires multiple tool calls.

> **Advanced Note:** The descriptions you write for tools are essentially prompts. The model reads them to decide when and how to use each tool. Invest time in writing clear, specific descriptions with examples of when the tool should and should not be used. This is one of the highest-leverage improvements you can make.

---

## Section 3: Single Tool Call

### Basic Flow

The simplest tool use pattern: the model calls one tool, gets the result, and responds.

Build a dictionary lookup tool. It should accept a `word` parameter (string) and look up the definition from a hardcoded `Record<string, string>` map. Include at least three words. When the word is found, return `{ word, definition, found: true }`. When it is not found, return `{ word, definition: null, found: false }`.

Wire it up with `generateText` and `stopWhen: isStepCount(2)`. The result object contains `text`, `toolCalls`, and `toolResults` — log all three to see the full lifecycle.

### Inspecting Tool Call Details

Build a stock price lookup tool that accepts a `symbol` parameter. Use a hardcoded map of ticker symbols to prices. The `generateText` result includes a `steps` array that reveals every step of the execution.

Iterate over `result.steps` and for each step log:

- `step.finishReason` — was it `'tool-calls'` or `'stop'`?
- `step.toolCalls` — what tool was called and with what arguments? Each call has `toolName`, `toolCallId`, and `input`.
- `step.toolResults` — what did the tool return? Each has an `output` property.
- `step.text` — any text the model generated in this step.

This step inspection pattern is how you debug tool use in practice. What do you expect the `finishReason` to be for the first step (when the model calls a tool) versus the last step (when it generates text)?

---

## Section 4: Tool Execution

### Manual Tool Execution (Without execute Function)

Sometimes you want to handle tool execution yourself instead of providing an `execute` function. Define a tool with only `description` and `inputSchema` — no `execute`.

When you call `generateText` without an `execute` function and without `stopWhen`, the model returns `finishReason: 'tool-calls'` and you must handle the loop manually. The workflow is:

1. Call `generateText` — check `step1.finishReason === 'tool-calls'`
2. Read `step1.toolCalls[0]` to get `toolName`, `toolCallId`, and `input`
3. Execute the tool yourself (e.g., read a file, call an API)
4. Call `generateText` again, passing the conversation history as `messages` — the original user message, the assistant's tool call (as a `'tool-call'` content block), and the tool result (as a `'tool-result'` content block)

Build a file reader tool (no `execute` function) that the model can request. When you handle execution manually, validate the path before reading — what should happen if the path points outside your allowed directory? Build the two-step flow: first call gets the tool request, your code executes it, second call sends the result back.

The message format for sending tool results back uses these content block types:

```typescript
// Assistant message content: array of tool-call blocks
{ type: 'tool-call' as const, toolCallId, toolName, input }

// Tool message content: array of tool-result blocks
{ type: 'tool-result' as const, toolCallId, toolName, output: { type: 'json', value: result } }
```

### Async Tool Execution

Tools often involve async operations — API calls, database queries, file operations. The `execute` function is always async, so you can `await` any operation inside it.

Build a geocoding tool that converts place names to latitude/longitude coordinates. Use a simulated lookup (a `Record<string, { lat: number; lon: number }>` map). Add a small `setTimeout` delay to simulate network latency. What should the tool return when a place name is not found?

Use `stopWhen: isStepCount(5)` and ask the model about coordinates for multiple cities to see how it handles multiple tool calls.

> **Beginner Note:** Your `execute` function can do anything a normal function can do: call APIs, query databases, read files, run computations.

---

## Section 5: Multi-Step Tool Loops

### The Pattern

Some questions require multiple tool calls in sequence. The model calls one tool, examines the result, decides it needs more information, calls another tool, and so on until it has enough information to answer.

Build an employee directory system with three tools:

1. **`getEmployee`**: Accepts an `employeeId` string, returns employee details from a hardcoded `Record` (name, department, managerId, salary). Return an error object if not found.
2. **`searchEmployees`**: Accepts a `query` string, filters employees by name or department (case-insensitive), returns matching entries with id, name, and department.
3. **`getDirectReports`**: Accepts a `managerId` string, finds all employees whose `managerId` matches, returns the list.

Use a dataset of at least 5 employees with a management hierarchy (some employees report to others).

Then ask the model: "Who reports to Alice? What is the total salary budget for her direct reports?" This question requires multiple tool calls in sequence — the model must search for Alice, get her direct reports, then look up each report's salary. Use `stopWhen: isStepCount(10)`.

After getting the result, iterate over `result.steps` to trace what the model did at each step. How many steps did it take? Did the model chain calls in the order you expected?

### Parallel Tool Calls

Some models can call multiple tools simultaneously when the calls are independent. Build two tools — `getWeather` (returns simulated weather for a city) and `getTime` (returns simulated local time for a city). Ask the model to compare the weather and time in three cities.

After execution, check the steps: if `step.toolCalls.length > 1` in any step, the model issued parallel calls. Did the model call both tools for the same city in parallel, or did it serialize them?

---

## Section 6: stopWhen and Automatic Loops

### How stopWhen with isStepCount Works

The `stopWhen` parameter with `isStepCount(N)` controls the maximum number of LLM call iterations. Each "step" is one call to the model. Without `stopWhen`, the model makes a single call and if it wants to use a tool, it stops with `finishReason: 'tool-calls'` and you must handle the loop yourself.

With `stopWhen: isStepCount(N)`, the SDK automatically:

1. Calls the model
2. If the model makes tool calls, executes the tools
3. Sends the results back to the model
4. Repeats until the model produces a text response or the step count is reached

Build a calculator tool and try calling `generateText` twice with the same prompt ("What is 15 \* 23?"): once **without** `stopWhen`, and once **with** `stopWhen: isStepCount(3)`. Compare the `finishReason` of each result. Without `stopWhen`, what is the `finishReason`? With `stopWhen`, what changes?

### Choosing Step Count

Guidelines for `isStepCount()`:

| Value               | Use Case                                                  | Example                                 |
| ------------------- | --------------------------------------------------------- | --------------------------------------- |
| `isStepCount(1)`    | No tool use (tools shown but model cannot act on results) | Preview which tool the model would pick |
| `isStepCount(2)`    | Single tool call + response (most common)                 | Dictionary lookup, weather check        |
| `isStepCount(3-5)`  | Multi-step reasoning                                      | Search, read, then analyze              |
| `isStepCount(5-10)` | Complex workflows                                         | Iterative refinement                    |
| `isStepCount(10+)`  | Agent-like behavior (careful: cost and latency add up)    | Open-ended research                     |

Each step is a full model call, so cost and latency scale linearly with steps. Each step includes the full conversation history.

> **Gotcha:** A high `isStepCount(N)` is a budget *ceiling*, not a target — and a cost trap. Because each step replays the entire conversation history (including every previous tool call and result), token usage grows quadratically with steps: a 10-step loop where each tool returns 500 tokens sends ~5000 tokens of tool results alone in the last step, on top of the original prompt and all intermediate model outputs. Set the ceiling as low as the task allows, and log the actual step count in production so one runaway request doesn't quietly 20× your bill.

### Monitoring Step Execution

You already built the step-inspection loop in Section 3 — iterating `result.steps` and reading each step's `finishReason`, `toolCalls`, `toolResults`, and `text`. Reuse that loop here; the one new element is **per-step token usage**: every step also carries a `step.usage` object with `inputTokens` and `outputTokens`.

Point your Section 3 inspection loop at a calculator tool with `stopWhen: isStepCount(5)` and a two-part math question (e.g., "What is the square root of 144, and then multiply that by 3?"), and add `step.usage` to what you record for each step.

Watch `inputTokens` climb from step 1 to the final step — each step replays the whole conversation so far. That growth curve is the cost mechanism the cost-trap Gotcha above warns about.

> **Gotcha (v7):** `step.usage` is per-step, but the top-level `result.usage` now **sums usage across every step** in AI SDK v7 (it reported only the final step before). Use `result.finalStep.usage` when you want just the last step. So adding up each `step.usage` and reading `result.usage` should now give the same total.

---

## Section 7: Error Handling in Tools

### Tool Execution Errors

Tools can fail for many reasons: network errors, invalid input, permission denied, rate limits. How you handle these errors determines the user experience.

Build a `fetchUrl` tool that accepts a `url` parameter (use `z.url()`). Your execute function should handle these concerns:

- Use `fetch` with an `AbortController` timeout (e.g., 10 seconds). What should happen when the request times out?
- Check `response.ok` and return a structured error if the HTTP status indicates failure.
- Truncate the response body to a maximum length (e.g., 5000 characters) to avoid token explosion. Include `truncated: true` in the result if you cut it short.
- Wrap everything in a try/catch. What should the error return shape look like so the model can understand what went wrong?

### Error Reporting Patterns

There are two patterns for handling errors in tools:

**Pattern 1: Return errors as structured data (recommended).** Wrap the execute body in try/catch and return `{ success: false, error: String(error) }` on failure. The model reads the error and can try a different approach.

**Pattern 2: Throw errors (stops the tool loop).** Use `throw new Error(...)` only for truly unrecoverable situations. When you throw, the entire tool loop stops and the user gets a generic error.

```typescript
// Pattern 1 shape:
return { success: true, data: result }
// or
return { success: false, error: 'what went wrong' }
```

Which pattern should you use by default, and why? Think about what happens to the conversation when a tool throws versus when it returns an error object.

### Validation Before Execution

Build a `databaseQuery` tool that accepts a SQL `query` string. Before executing any query, validate it:

1. Normalize the query (trim, lowercase)
2. Check for dangerous keywords (`drop`, `delete`, `update`, `insert`, `alter`, `truncate`) — if found, return a structured error explaining which keyword was blocked
3. Verify the query starts with `select` — if not, return an error
4. Only after validation passes, execute the query (simulated) and return results

What is the right return shape for a rejected query versus a successful one? Should you include the original query in both cases?

---

## Section 8: Tool Design Patterns

### Granularity: Fine vs Coarse Tools

Tool granularity is about finding the right level of abstraction. Consider three approaches to file operations:

**Too fine-grained** — separate tools for `openFile`, `readLine`, and `closeFile`. The model needs three calls just to read one file. Each call adds latency and cost.

**Too coarse-grained** — a single `doEverything` tool that accepts an `action` string parameter. The model has no structured choices and the parameter space is ambiguous.

**Just right** — task-level tools like `readFile`, `writeFile`, and `listDirectory`. Each tool does one complete, meaningful operation.

Build a set of file operation tools at the "just right" level. Define `readFile` (accepts `path`, returns content and size), `writeFile` (accepts `path` and `content`, writes the file, returns bytes written), and `listDirectory` (accepts `path`, returns entries with name and type). What makes this the right granularity? Think about how many tool calls the model needs for common file tasks.

### Naming Conventions

Good tool names follow the pattern **verb + noun**: `searchProducts`, `getOrderStatus`, `createTicket`, `sendEmail`, `calculateTax`. Each name immediately communicates what the tool does.

Bad names: `search`, `process`, `handle`, `doThing`. These force the model to rely entirely on the description.

### Description Best Practices

Compare these two descriptions:

- **Good**: "Search the product catalog by keyword, category, or price range. Returns up to 20 matching products with name, price, and rating. Use this when the user asks about products, availability, or pricing. Does NOT check inventory — use checkInventory for stock availability."
- **Bad**: "Search for stuff"

A good description answers three questions: _what_ does the tool do, _when_ should the model use it, and _what are its limitations_?

### Composable Tool Sets

Build a generic `createCRUDTools` function:

```typescript
function createCRUDTools<T extends z.ZodType>(
  entityName: string,
  schema: T,
  store: Map<string, z.infer<T>>
): Record<string, ReturnType<typeof tool>>
```

This function should generate four tools using the entity name: `list{Entity}s`, `get{Entity}`, `create{Entity}`, and `delete{Entity}`. Each tool gets an appropriate description, parameter schema, and execute function that operates on the provided `Map` store.

Then use it to create a todo management toolkit:

```typescript
const todoSchema = z.object({
  title: z.string().describe('Todo title'),
  priority: z.enum(['low', 'medium', 'high']).describe('Priority level'),
  done: z.boolean().default(false),
})
const todoStore = new Map<string, z.infer<typeof todoSchema>>()
const todoTools = createCRUDTools('Todo', todoSchema, todoStore)
```

How does the `list` tool work with `z.object({})`? How does `create` use the schema as its parameters? What should `get` and `delete` return when the ID is not found?

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
  inputSchema: z.object({
    path: z.string().describe('Relative path within the project directory'),
  }),
  execute: async ({ path: inputPath }) => {
    // Your implementation here
  },
})
```

Build the `execute` function with three layers of defense: (1) Resolve and normalize the path relative to a `projectRoot`, then check it stays within that directory using `startsWith`. (2) Check the path against a list of blocked patterns (`.env`, `secrets`, `credentials`, `.key`, `.pem`). (3) Read the file and cap the returned content at a max character limit. Return an `{ error }` object for denied or missing files, or `{ path, content }` on success.

Why is `resolve(projectRoot, normalize(inputPath))` necessary instead of just concatenating strings? What attack does it prevent?

### Allowlists

```typescript
import { tool } from 'ai'
import { z } from 'zod'

// Only allow specific, pre-approved operations
const allowedCommands = ['git status', 'git log --oneline -10', 'npm test', 'npm run lint', 'ls -la'] as const

const safeShellTool = tool({
  description: 'Run a whitelisted shell command',
  inputSchema: z.object({
    command: z.enum(allowedCommands).describe('The command to execute (must be from the allowed list)'),
  }),
  execute: async ({ command }) => {
    // Your implementation here
  },
})
```

Build the `execute` function. Split the command string into parts, spawn a subprocess using `Bun.spawn` with `stdout: 'pipe'` and `stderr: 'pipe'`, read both streams, and return `{ command, stdout, stderr, exitCode }`. Cap stdout and stderr at reasonable limits (e.g., 5000 and 1000 characters) to prevent token budget overflow.

Why does using `z.enum(allowedCommands)` in the parameters provide security at the schema level? What happens if the model tries to generate a command not in the allowlist?

### Rate Limiting Tools

```typescript
import { tool } from 'ai'
import { z } from 'zod'

class ToolRateLimiter {
  private calls: Map<string, number[]> = new Map()

  constructor(private maxCalls: number, private windowMs: number) {}

  check(toolName: string): boolean {
    /* ... */
  }
}

const rateLimiter = new ToolRateLimiter(10, 60_000) // 10 calls per minute

function rateLimitedTool(name: string, config: Parameters<typeof tool>[0]): ReturnType<typeof tool> {
  /* ... */
}
```

Build the `ToolRateLimiter` class with a `calls` map tracking timestamps per tool name. The `check` method should filter out timestamps outside the window, return `false` if at the limit, and record the current timestamp if allowed. Then build `rateLimitedTool` as a wrapper: extract `config.execute`, return a new `tool` that calls `rateLimiter.check(name)` before delegating to the original execute function. Return an error object if rate-limited.

Why use a sliding window (filtering by timestamp) rather than a simple counter that resets? What edge case does the sliding window handle better?

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
  /* ... */
}
```

Build `auditedTool` as a wrapper that intercepts the tool's `execute` function. Before calling the original execute, log the tool name, arguments (as JSON), and current timestamp. After execution, log the duration and whether it succeeded or failed. Use `try/catch` to capture errors — log them and re-throw so the caller still sees the error. Return a new `tool()` with the wrapped execute.

What information should you avoid logging (think about PII and credentials in tool arguments)? How would you make the logging configurable?

> **Beginner Note:** The model might generate unexpected or malicious arguments — not because it is adversarial, but because it is probabilistic and can make mistakes. Defend accordingly.

> **Advanced Note:** For high-security applications, consider running tool execution in a separate process or container with restricted permissions. Tools that interact with databases should use read-only connections unless writes are explicitly required. Tools that access the filesystem should use chroot or similar isolation.

> **Advanced Note: Model Context Protocol (MCP)** — In this module, you defined tools by hand in your application code. Anthropic's [Model Context Protocol](https://modelcontextprotocol.io) is an open standard that lets tools be discovered dynamically from external servers. Instead of hardcoding a `weatherTool`, your agent connects to an MCP server and discovers available tools at runtime. The Vercel AI SDK supports this via `@ai-sdk/mcp`:
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

> **Local Alternative (Ollama):** Tool calling works with `ollama('qwen3.5', { think: false })` — Qwen 3.5 has native function calling support. If tool calling fails with your local model, try the cloud variant (`ollama('qwen3.5:cloud', { think: false })`) for better reliability. The multi-step tool loop and `stopWhen: isStepCount()` work identically.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Going Further: Production Tool Patterns

The nine sections above are everything you need to define, call, loop over, and secure tools. These last patterns are how tool systems get organized in production agents — lifecycle hooks, permissions, user-defined tools. Reference material, not prerequisites.

### Production Tool Architecture

#### Self-Contained Tool Directories

When an application grows beyond a handful of tools, keeping tool definitions inline becomes unmaintainable. Production systems organize each tool as a self-contained directory:

```
tools/
├── SearchWeb/
│   ├── SearchWeb.ts    # Main implementation (execute function)
│   ├── prompt.ts       # System prompt fragment describing the tool's purpose
│   ├── schema.ts       # Zod input/output schemas
│   └── index.ts        # Re-exports
├── ReadFile/
│   ├── ReadFile.ts
│   ├── prompt.ts
│   ├── schema.ts
│   └── index.ts
```

Each tool owns three things: its **schema** (what arguments it accepts), its **prompt fragment** (instructions the LLM needs to use it well), and its **execution logic** (what happens when it runs). The system prompt is assembled by concatenating the prompt fragments from all registered tools — tools are self-describing.

This pattern scales to dozens of tools. Adding a new tool means adding a new directory, not modifying a central file. It also makes tools testable in isolation: you can unit test the schema, the execution logic, and the prompt fragment separately.

---

### Tool Lifecycle Hooks

#### Pre- and Post-Execution Hooks

Production tool systems run code before and after every tool execution. These hooks form a pipeline around the core execute function:

```
validate input (schema) → check permission → preToolUse hook → execute → postToolUse hook → return result
```

**PreToolUse hooks** run after schema validation but before execution. Uses include:

- Logging the tool call with arguments and timestamp
- Checking permissions (is this tool allowed in the current context?)
- Sanitizing inputs (stripping dangerous characters from shell commands)
- Starting a performance timer

**PostToolUse hooks** run after execution completes. Uses include:

- Logging the result and execution time
- Validating the output against expected formats
- Truncating large results before they consume context tokens
- Auditing (recording what the tool did for compliance)

Hooks can also **block execution**. A preToolUse hook that detects a disallowed operation returns an error result without ever calling the execute function. This is the mechanism behind permission systems.

```typescript
type PreToolHook = (call: { name: string; args: unknown }) => { allow: boolean; reason?: string }
type PostToolHook = (call: { name: string; args: unknown; result: unknown; durationMs: number }) => void
```

---

### Tool Result Management

#### Truncation for Context Budget

Tool results can be enormous — a file read might return thousands of lines, a search might return dozens of documents. Every token in a tool result consumes context window space that could be used for conversation or reasoning. Production systems enforce result size limits.

The naive approach is to cut the result at a character limit, but this loses structure. A better approach preserves the beginning and end while indicating what was truncated:

```typescript
function truncateResult(result: string, maxTokens: number): string {
  /* ... */
}
```

Build this function. Estimate the token count (roughly `result.length / 4`), and if within budget, return the result unchanged. Otherwise, compute a character budget from `maxTokens * 4`, split it between a head portion (~70%) and a tail portion (~20%) — the remaining ~10% of the budget is slack that absorbs the omission notice — and join them with a message indicating how many tokens were omitted. Return the concatenated head + omission notice + tail.

Why preserve both the head and tail instead of just truncating at the end? What kind of content typically appears at the end of a file or search result that would be worth preserving?

---

### Tool Permissions

#### Three-Tier Permission System

Production systems categorize tool operations into three permission levels:

- **Allow** — safe operations that execute without confirmation (reading files in the project directory, running tests)
- **Deny** — dangerous operations that are always blocked (deleting files outside the project, running destructive shell commands)
- **Ask** — operations requiring user confirmation before execution (writing files, running arbitrary commands)

Permission rules are declarative and checked before tool execution (in the preToolUse hook). This is the simplest version of the pattern — Module 24 (Safety) covers guardrails in depth.

#### Glob-Based Permission Rules

For finer-grained control, production systems match permission rules against the full command or argument string using glob patterns:

```typescript
const rules: PermissionRule[] = [
  { pattern: 'git status *', permission: 'allow' },
  { pattern: 'git push --force *', permission: 'deny' },
  { pattern: 'npm *', permission: 'ask' },
  { pattern: 'rm -rf *', permission: 'deny' },
]
```

The last matching rule wins. Rules can be scoped per-project (in a project config file) or globally. Unmatched commands fall through to a configurable default — typically `'ask'` in interactive mode and `'deny'` in automated mode.

---

### Custom Tools as TypeScript Files

#### User-Defined Tools with Zod Schemas

Some production systems let users extend the tool set by placing `.ts` files in a configuration directory. Each file exports one or more tools using the same `tool()` + Zod pattern the Vercel AI SDK uses:

```typescript
// tools/my-custom-tool.ts
import { tool } from 'ai'
import { z } from 'zod'

export const myCustomTool = tool({
  description: 'Looks up a user by email address',
  inputSchema: z.object({ email: z.email() }),
  execute: async ({ email }) => {
    // implementation
  },
})
```

The runtime discovers tools by scanning the directory at startup, importing each file, and registering any exported `tool()` instances. The filename and export name become the tool identity. This pattern supports hot-reloading — adding or modifying a tool file takes effect without restarting the application.

The `execute` function can receive a `context` parameter with session information (working directory, session ID, user identity), giving tools access to session state without global variables.

---

### Context-Providing Tools (LSP Pattern)

#### Tools That Inform Rather Than Act

Not all tools perform actions. Some tools exist to give the LLM better information for decision-making. The strongest example is Language Server Protocol (LSP) integration, where tools expose code intelligence operations: go-to-definition, find-references, hover info, and workspace symbol search.

The key insight is that these tools' output is not a final result — it is intelligence that informs the LLM's next action. When the LLM calls "find all references to function X" before renaming it, the tool result tells the LLM which files need updating. The LLM then uses that information to make accurate, complete edits.

Other examples of context-providing tools: file search (find relevant files before reading them), grep (locate patterns before editing), and diagnostics (discover errors after a change). Designing tools as context providers — not just action executors — produces more capable agents because the LLM can gather information before committing to an action.

---

## Summary

In this module, you learned:

1. **What tools are:** Tools let LLMs interact with the external world by generating structured function calls that your application executes and returns results from.
2. **Tool definitions with Zod:** How to define tools with names, descriptions, and typed parameter schemas that guide the model toward correct usage.
3. **The tool call lifecycle:** The model proposes a tool call, your code executes it, and the result is sent back — the model never runs code directly.
4. **Single and multi-step tool calls:** How to handle one-shot tool use and iterative loops where the model chains multiple tool calls to accomplish complex tasks.
5. **stopWhen and automatic loops:** The Vercel AI SDK's `stopWhen` parameter with `isStepCount()` automates the tool execution loop, letting the model call tools repeatedly until it has a final answer.
6. **Error handling:** How to catch tool execution errors, report them back to the model in a structured way, and validate inputs before execution.
7. **Tool design patterns:** Principles for granularity, naming, composability, and deciding between fine-grained and coarse-grained tool interfaces.
8. **Security considerations:** Input validation, sandboxing, allowlists, and running tool execution with restricted permissions to prevent misuse.
9. **Production tool architecture:** Self-contained tool directories with separate schema, prompt fragment, and execution logic for scalable tool systems.
10. **Lifecycle hooks:** Pre- and post-execution hooks for logging, timing, permission checks, and output validation.
11. **Result truncation:** Intelligently truncating large tool results to preserve context budget while keeping structure.
12. **Tool permissions:** Three-tier (allow/deny/ask) and glob-based permission systems for controlling tool execution.
13. **Custom tools:** User-defined tools as TypeScript files with Zod schemas, discovered and registered at runtime.
14. **Context-providing tools:** Tools that inform the LLM's decisions (like LSP) rather than just performing actions.

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

What does `stopWhen: isStepCount(5)` mean in a `generateText` call with tools?

A) The model can call a maximum of 5 tools total
B) The SDK will make up to 5 calls to the model, automatically handling tool calls and results
C) Each tool can be called at most 5 times
D) The model will generate at most 5 sentences

**Answer: B**

`stopWhen: isStepCount(5)` controls the maximum number of model call iterations. The SDK will call the model, execute any requested tools, send results back, and repeat — up to 5 times. This allows multi-step tool chains where each step can involve one or more tool calls.

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
   - `calculator`: evaluates mathematical expressions (use `Function()` with input validation as shown in the module, or the `mathjs` package)
   - `searchWeb`: simulates web search (returns predefined results for known queries)
   - `readFile`: reads files from a specified directory (with path validation)
2. Use `stopWhen: isStepCount(10)` for multi-step reasoning
3. Log each tool call with timing information
4. Handle errors gracefully (return structured error objects)
5. Implement path validation for `readFile` (restrict to a project directory)
6. Build a conversation loop that maintains context between turns — keep the history as a `ModelMessage[]` and route the system prompt through the `instructions` field, not a `role: 'system'` message

**Starter code:**

```typescript
import { generateText, tool, isStepCount, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Conversation history uses the SDK's ModelMessage type. Note: no 'system'
// role in this array — in AI SDK v7 the system prompt goes in the
// `instructions` field of generateText (a role:'system' message inside
// `messages` throws unless you set allowSystemInMessages: true).
const messages: ModelMessage[] = []

const PROJECT_DIR = './workspace'

const tools = {
  calculator: tool({
    description: 'TODO: Write a good description',
    inputSchema: z.object({
      // TODO: Define parameters
    }),
    execute: async args => {
      // TODO: Implement math evaluation
      // Use Function() with input validation (as shown in the module) or the mathjs package
    },
  }),

  searchWeb: tool({
    description: 'TODO: Write a good description',
    inputSchema: z.object({
      // TODO: Define parameters
    }),
    execute: async args => {
      // TODO: Return simulated search results
      // Create a map of queries to predefined results
    },
  }),

  readFile: tool({
    description: 'TODO: Write a good description',
    inputSchema: z.object({
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

// TODO: Implement the conversation loop:
//   - each turn: push the user message onto `messages`, call generateText with
//     { model, instructions, messages, tools, stopWhen: isStepCount(10) },
//     then push the response messages back onto `messages` to keep context
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

### Exercise 3: Tool Lifecycle Hooks

Build a hook system that wraps tool execution with pre- and post-hooks for logging, timing, and permission checking.

**What to build:** Create `src/tools/exercises/lifecycle-hooks.ts`

**Requirements:**

1. Define a `ToolRunner` that accepts a map of tools and arrays of pre/post hooks
2. Pre-hooks receive `{ name: string; args: unknown }` and return `{ allow: boolean; reason?: string }` — if any pre-hook returns `allow: false`, execution is blocked and the reason is returned as the tool result
3. Post-hooks receive `{ name: string; args: unknown; result: unknown; durationMs: number }` and run after execution completes (they cannot block)
4. Implement a logging pre-hook that records every tool call attempt with timestamp and arguments
5. Implement a timing post-hook that records execution duration for each tool
6. Implement a permission pre-hook that checks tool names against an allowlist and blocks unrecognized tools
7. The `ToolRunner.execute(name, args)` method runs the full pipeline: pre-hooks, execute, post-hooks

**Expected behavior:**

- Calling an allowed tool runs all pre-hooks (all return `allow: true`), executes the tool, runs all post-hooks, and returns the result
- Calling a blocked tool runs pre-hooks until one returns `allow: false`, skips execution and post-hooks, and returns the denial reason
- The logging hook captures all attempts (both allowed and blocked)
- The timing hook records accurate durations (within 10ms) for executed tools

**File:** `src/tools/exercises/lifecycle-hooks.ts`

### Exercise 4: Tool Result Truncation

Build a result truncation system that keeps tool results within a token budget while preserving useful structure.

**What to build:** Create `src/tools/exercises/result-truncation.ts`

**Requirements:**

1. Implement `truncateToolResult(result: string, maxTokens: number): string` that truncates results exceeding the token budget
2. Use a 70/30 head/tail split — preserve the first 70% and last 30% of the allowed content, with a truncation notice in between (a slightly different accounting than Going Further's 70/20-plus-notice-slack: here the notice rides on top of the content budget)
3. The truncation notice should state how many tokens were omitted
4. Use a simple token estimation: 1 token per 4 characters
5. If the result is within budget, return it unchanged
6. Handle edge cases: empty results, results exactly at the budget, very small budgets (under 20 tokens)

**Expected behavior:**

- A 1000-token result with a 500-token budget returns ~350 tokens of head, a truncation notice, and ~150 tokens of tail
- A 100-token result with a 500-token budget returns the original result unchanged
- The truncation notice reads something like `"... (500 tokens omitted) ..."`
- An empty result returns an empty string regardless of budget

**File:** `src/tools/exercises/result-truncation.ts`
