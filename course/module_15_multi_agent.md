# Module 15: Multi-Agent Systems

## Learning Objectives

- Understand why multiple agents outperform a single agent for complex tasks
- Implement the orchestrator-worker pattern for agent coordination
- Pass context between agents while maintaining separation of concerns
- Manage shared state and private state across agent boundaries
- Design delegation strategies based on capability and topic
- Run agents in parallel for independent sub-tasks
- Implement agent handoff for transferring conversations between specialists
- Handle sub-agent failures with retries, fallbacks, and graceful degradation

---

## Why Should I Care?

A single agent can research, write, and review — but it cannot do all three well at the same time. Just as software teams have frontend developers, backend developers, and QA engineers, multi-agent systems assign different roles to different agents. Each agent has a focused system prompt, a curated tool set, and a clear responsibility.

Multi-agent systems are how production LLM applications handle complex workflows: a customer support system with a triage agent, a billing specialist, and a technical support specialist; a content pipeline with research, writing, and editing agents; a coding assistant with a planner, implementer, and reviewer.

This module builds directly on the agent fundamentals from Module 14. If you can build one agent, you can build a system of agents that work together. The challenge is coordination — and that is what this module teaches.

---

## Connection to Other Modules

- **Module 14 (Agent Fundamentals)** provides the single-agent patterns that this module composes into multi-agent systems.
- **Module 16 (Workflows & Chains)** offers an alternative to multi-agent systems for deterministic pipelines.
- **Module 17 (Code Generation)** can use multi-agent patterns for plan-implement-review cycles.
- **Module 18 (Human-in-the-Loop)** adds human oversight to multi-agent coordination.

---

## Section 1: Why Multiple Agents?

### The Limits of a Single Agent

A single agent with many tools and a broad system prompt faces several problems:

1. **Prompt dilution** — a system prompt that covers research, writing, review, and formatting gives weak guidance on each
2. **Tool overload** — too many tools make tool selection harder and increase the chance of wrong picks
3. **Context pollution** — intermediate reasoning for one sub-task pollutes the context for another
4. **No separation of concerns** — everything happens in one conversation, making debugging harder

### The Multi-Agent Solution

Split a complex task into roles, give each role to a dedicated agent:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Single agent trying to do everything — gets overwhelmed
const doEverythingAgent = await generateText({
  model: mistral('mistral-small-latest'),
  maxSteps: 15,
  system: `You are a research-writer-reviewer agent. You must:
1. Research the topic using search tools
2. Write a well-structured article
3. Review the article for accuracy and style
4. Fix any issues found in review
This is a lot to handle in one prompt...`,
  tools: {
    /* 10+ tools */
  },
  prompt: 'Write a comprehensive article about renewable energy trends',
})

// Better: Three focused agents with clear responsibilities
async function researchAgent(topic: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    maxSteps: 5,
    system: `You are a research specialist. Your ONLY job is to gather facts,
statistics, and key points about a topic. Return a structured research brief.
Do NOT write articles. Do NOT review content. Just research.`,
    tools: {
      search: {
        description: 'Search for information on a topic',
        parameters: z.object({ query: z.string() }),
        execute: async ({ query }) => `Results for "${query}": [simulated research data]`,
      },
    },
    prompt: `Research this topic thoroughly: ${topic}`,
  })
  return result.text
}

async function writerAgent(researchBrief: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a professional writer. Given a research brief, write a clear,
well-structured article. Focus on readability and flow.
Do NOT do additional research. Use only the provided brief.`,
    prompt: `Write an article based on this research brief:\n\n${researchBrief}`,
  })
  return result.text
}

async function reviewerAgent(article: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are an editor. Review the article for:
- Factual accuracy (flag claims that seem unsupported)
- Clarity and readability
- Structure and flow
- Grammar and style
Return specific, actionable feedback.`,
    prompt: `Review this article:\n\n${article}`,
  })
  return result.text
}

// Run the pipeline
const research = await researchAgent('renewable energy trends 2025')
const draft = await writerAgent(research)
const feedback = await reviewerAgent(draft)

console.log('Research:', research.slice(0, 200))
console.log('\nDraft:', draft.slice(0, 200))
console.log('\nFeedback:', feedback.slice(0, 200))
```

> **Beginner Note:** Think of multi-agent systems like a team of people. You would not ask one person to simultaneously research, write, edit, and fact-check. You assign roles because specialists do better work than generalists trying to do everything at once.

> **Advanced Note:** Multi-agent systems are not always better. They add latency (more LLM calls), complexity (coordination logic), and cost (more tokens). Use them when the task genuinely benefits from separation of concerns. For simple tasks, a single well-prompted agent is faster, cheaper, and easier to debug.

---

## Section 2: Orchestrator-Worker Pattern

### The Core Pattern

The orchestrator-worker pattern has one "orchestrator" agent that understands the overall task and delegates sub-tasks to specialized "worker" agents. The orchestrator:

1. Analyzes the task
2. Breaks it into sub-tasks
3. Assigns each sub-task to the right worker
4. Collects and synthesizes results

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// --- Worker agents ---

async function researchWorker(query: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    maxSteps: 3,
    system: `You are a research agent. Search for factual information and return
a concise summary of findings. Include specific data points and sources where possible.`,
    tools: {
      search: {
        description: 'Search the web for information',
        parameters: z.object({ query: z.string() }),
        execute: async ({ query }) => `Results for "${query}": [simulated search data with relevant facts]`,
      },
    },
    prompt: query,
  })
  return result.text
}

async function analysisWorker(data: string, question: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a data analyst. Analyze the provided data and answer
the specific question. Be precise, use numbers when available, and note
any limitations in the data.`,
    prompt: `Data:\n${data}\n\nQuestion: ${question}`,
  })
  return result.text
}

async function writingWorker(content: string, format: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a professional writer. Transform the provided content
into the requested format. Maintain accuracy while improving readability.`,
    prompt: `Transform this content into ${format}:\n\n${content}`,
  })
  return result.text
}

// --- Orchestrator ---

interface SubTask {
  id: number
  type: 'research' | 'analysis' | 'writing'
  description: string
  input: string
  dependsOn: number[]
}

async function orchestrator(task: string): Promise<string> {
  // Step 1: Plan sub-tasks
  const { output: plan } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        subtasks: z.array(
          z.object({
            id: z.number(),
            type: z.enum(['research', 'analysis', 'writing']),
            description: z.string(),
            input: z.string().describe('The input/query for this sub-task'),
            dependsOn: z.array(z.number()),
          })
        ),
      }),
    }),
    prompt: `Break this task into sub-tasks. Available worker types:
- research: Searches for factual information
- analysis: Analyzes data and answers questions
- writing: Transforms content into a specific format

Task: ${task}`,
  })

  console.log(
    'Plan:',
    plan.subtasks.map(s => `${s.id}: [${s.type}] ${s.description}`)
  )

  // Step 2: Execute sub-tasks in dependency order
  const results = new Map<number, string>()

  for (const subtask of plan.subtasks) {
    // Wait for dependencies
    const depsReady = subtask.dependsOn.every(id => results.has(id))
    if (!depsReady) {
      console.warn(`Skipping subtask ${subtask.id}: dependencies not ready`)
      continue
    }

    // Build input with dependency results
    let input = subtask.input
    if (subtask.dependsOn.length > 0) {
      const depContext = subtask.dependsOn.map(id => results.get(id)).join('\n\n')
      input = `${depContext}\n\n${subtask.input}`
    }

    console.log(`Executing subtask ${subtask.id}: ${subtask.description}`)

    // Dispatch to the right worker
    let result: string
    switch (subtask.type) {
      case 'research':
        result = await researchWorker(input)
        break
      case 'analysis':
        result = await analysisWorker(input, subtask.description)
        break
      case 'writing':
        result = await writingWorker(input, subtask.description)
        break
    }

    results.set(subtask.id, result)
  }

  // Step 3: Synthesize final result
  const allResults = Array.from(results.entries())
    .map(([id, result]) => {
      const subtask = plan.subtasks.find(s => s.id === id)
      return `## Sub-task ${id}: ${subtask?.description}\n${result}`
    })
    .join('\n\n')

  const finalResult = await generateText({
    model: mistral('mistral-small-latest'),
    system: 'Synthesize these sub-task results into a coherent final answer.',
    prompt: `Original task: ${task}\n\nSub-task results:\n${allResults}`,
  })

  return finalResult.text
}

// Usage
const result = await orchestrator(
  'Analyze the current state of electric vehicle adoption in Europe, including market share, key players, and infrastructure challenges. Present as an executive summary.'
)
console.log('\n=== Final Result ===\n', result)
```

> **Beginner Note:** The orchestrator is like a project manager — it does not do the actual work, but it knows who should do what and in what order. Worker agents are the specialists who execute specific tasks.

### Dynamic Worker Selection

Instead of hardcoding worker types, let the orchestrator discover available workers at runtime:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface WorkerDefinition {
  name: string
  description: string
  capabilities: string[]
  execute: (input: string) => Promise<string>
}

class AgentRegistry {
  private workers = new Map<string, WorkerDefinition>()

  register(worker: WorkerDefinition): void {
    this.workers.set(worker.name, worker)
  }

  getWorker(name: string): WorkerDefinition | undefined {
    return this.workers.get(name)
  }

  listWorkers(): string {
    return Array.from(this.workers.values())
      .map(w => `- ${w.name}: ${w.description} (capabilities: ${w.capabilities.join(', ')})`)
      .join('\n')
  }

  async dispatch(workerName: string, input: string): Promise<string> {
    const worker = this.workers.get(workerName)
    if (!worker) {
      throw new Error(`Worker "${workerName}" not found`)
    }
    return worker.execute(input)
  }
}

// Set up registry
const registry = new AgentRegistry()

registry.register({
  name: 'researcher',
  description: 'Searches the web for factual information',
  capabilities: ['web search', 'fact finding', 'data gathering'],
  execute: async input => {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      maxSteps: 3,
      system: 'You are a research specialist. Search and summarize findings.',
      tools: {
        search: {
          description: 'Search for information',
          parameters: z.object({ query: z.string() }),
          execute: async ({ query }) => `Results for "${query}": [data]`,
        },
      },
      prompt: input,
    })
    return result.text
  },
})

registry.register({
  name: 'writer',
  description: 'Writes polished content from provided material',
  capabilities: ['article writing', 'summarization', 'formatting'],
  execute: async input => {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      system: 'You are a professional writer. Create clear, engaging content.',
      prompt: input,
    })
    return result.text
  },
})

registry.register({
  name: 'critic',
  description: 'Reviews content for quality, accuracy, and improvements',
  capabilities: ['editing', 'fact checking', 'quality review'],
  execute: async input => {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      system: 'You are a critical editor. Find issues and suggest improvements.',
      prompt: input,
    })
    return result.text
  },
})

// Orchestrator uses registry to delegate
async function dynamicOrchestrator(task: string, registry: AgentRegistry): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    maxSteps: 8,
    system: `You are an orchestrator. Break tasks into steps and delegate to workers.

Available workers:
${registry.listWorkers()}

Use the delegate tool to assign work. Collect results and synthesize a final answer.`,
    tools: {
      delegate: {
        description: 'Delegate a sub-task to a specific worker agent',
        parameters: z.object({
          worker: z.string().describe('Name of the worker to delegate to'),
          task: z.string().describe('The sub-task description'),
        }),
        execute: async ({ worker, task }: { worker: string; task: string }) => {
          try {
            const result = await registry.dispatch(worker, task)
            return JSON.stringify({
              worker,
              success: true,
              result,
            })
          } catch (error) {
            return JSON.stringify({
              worker,
              success: false,
              error: error instanceof Error ? error.message : 'Unknown error',
            })
          }
        },
      },
    },
    prompt: task,
  })

  return result.text
}

const output = await dynamicOrchestrator('Write a 500-word blog post about the future of AI in healthcare', registry)
console.log(output)
```

---

## Section 3: Agent Communication

### Passing Context Between Agents

Agents communicate by passing data — the output of one agent becomes the input of another. The key design decision is what to pass and in what format.

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Define a structured handoff format
const handoffSchema = z.object({
  summary: z.string().describe('Brief summary of what was accomplished'),
  findings: z
    .array(
      z.object({
        topic: z.string(),
        content: z.string(),
        confidence: z.enum(['high', 'medium', 'low']),
      })
    )
    .describe('Key findings from the research'),
  openQuestions: z.array(z.string()).describe('Questions that still need answers'),
  recommendations: z.array(z.string()).describe('Suggested next steps'),
})

type Handoff = z.infer<typeof handoffSchema>

// Research agent outputs a structured handoff
async function researchWithHandoff(topic: string): Promise<Handoff> {
  // First, do the research
  const research = await generateText({
    model: mistral('mistral-small-latest'),
    maxSteps: 4,
    system: 'You are a research specialist. Gather comprehensive information.',
    tools: {
      search: {
        description: 'Search for information',
        parameters: z.object({ query: z.string() }),
        execute: async ({ query }) => `Results for "${query}": [simulated comprehensive results about ${topic}]`,
      },
    },
    prompt: `Research this topic thoroughly: ${topic}`,
  })

  // Then, structure the handoff
  const { output: handoff } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: handoffSchema }),
    prompt: `Based on this research, create a structured handoff document:

${research.text}

Format it as a handoff for the next agent in the pipeline.`,
  })

  return handoff
}

// Writing agent receives structured handoff
async function writeFromHandoff(handoff: Handoff, format: string): Promise<string> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a professional writer. You receive structured research handoffs
and transform them into polished content. Use all high-confidence findings directly.
For medium-confidence findings, include with caveats. Skip low-confidence findings.`,
    prompt: `Create a ${format} from this research:

Summary: ${handoff.summary}

Key Findings:
${handoff.findings.map(f => `- [${f.confidence}] ${f.topic}: ${f.content}`).join('\n')}

Open Questions (mention these as areas for further investigation):
${handoff.openQuestions.map(q => `- ${q}`).join('\n')}`,
  })

  return result.text
}

// Usage
const handoff = await researchWithHandoff('sustainable agriculture technologies')
console.log('Handoff:', JSON.stringify(handoff, null, 2))

const article = await writeFromHandoff(handoff, 'blog post')
console.log('\nArticle:', article)
```

### Message-Passing Protocol

For more complex systems, define a formal message protocol:

```typescript
interface AgentMessage {
  from: string
  to: string
  type: 'request' | 'response' | 'error' | 'status'
  correlationId: string
  payload: unknown
  timestamp: number
}

class MessageBus {
  private handlers = new Map<string, (msg: AgentMessage) => Promise<void>>()
  private log: AgentMessage[] = []

  register(agentName: string, handler: (msg: AgentMessage) => Promise<void>): void {
    this.handlers.set(agentName, handler)
  }

  async send(message: AgentMessage): Promise<void> {
    this.log.push(message)
    console.log(`[MessageBus] ${message.from} -> ${message.to}: ${message.type}`)

    const handler = this.handlers.get(message.to)
    if (!handler) {
      throw new Error(`No handler registered for agent "${message.to}"`)
    }

    await handler(message)
  }

  getLog(): AgentMessage[] {
    return [...this.log]
  }
}

// Usage
const bus = new MessageBus()

bus.register('researcher', async msg => {
  if (msg.type === 'request') {
    // Process research request and send response
    const result = `Research results for: ${msg.payload}`
    await bus.send({
      from: 'researcher',
      to: msg.from,
      type: 'response',
      correlationId: msg.correlationId,
      payload: result,
      timestamp: Date.now(),
    })
  }
})

bus.register('orchestrator', async msg => {
  if (msg.type === 'response') {
    console.log(`Orchestrator received: ${msg.payload}`)
  }
})

// Orchestrator sends a request to researcher
await bus.send({
  from: 'orchestrator',
  to: 'researcher',
  type: 'request',
  correlationId: crypto.randomUUID(),
  payload: 'AI trends in 2025',
  timestamp: Date.now(),
})
```

> **Advanced Note:** Message buses add infrastructure complexity. Use direct function calls (agent A calls agent B directly) for simple systems. Introduce a message bus only when you need features like logging, replay, or decoupled agents.

---

## Section 4: Shared State

### What to Share, What Stays Private

In a multi-agent system, some information should be shared across agents and some should stay private:

| Shared State                | Private State                |
| --------------------------- | ---------------------------- |
| Task description and goals  | Internal reasoning traces    |
| Key findings and facts      | Intermediate drafts          |
| Current progress and status | Tool-specific parameters     |
| Error reports and blockers  | Working scratchpad           |
| Final outputs               | Model-specific prompt tricks |

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Shared state accessible to all agents
interface SharedState {
  task: string
  findings: Map<string, string>
  status: Map<string, 'pending' | 'in_progress' | 'completed' | 'failed'>
  errors: string[]
  finalOutputs: Map<string, string>
}

function createSharedState(task: string): SharedState {
  return {
    task,
    findings: new Map(),
    status: new Map(),
    errors: [],
    finalOutputs: new Map(),
  }
}

// Each agent gets read/write access to shared state through controlled methods
class AgentContext {
  constructor(
    private agentName: string,
    private shared: SharedState,
    private privateState: Map<string, unknown> = new Map()
  ) {}

  // Shared state methods
  addFinding(key: string, value: string): void {
    this.shared.findings.set(`${this.agentName}:${key}`, value)
  }

  getAllFindings(): Map<string, string> {
    return new Map(this.shared.findings)
  }

  setStatus(status: 'pending' | 'in_progress' | 'completed' | 'failed'): void {
    this.shared.status.set(this.agentName, status)
  }

  reportError(error: string): void {
    this.shared.errors.push(`[${this.agentName}] ${error}`)
  }

  setOutput(key: string, value: string): void {
    this.shared.finalOutputs.set(key, value)
  }

  getTask(): string {
    return this.shared.task
  }

  // Private state methods
  setPrivate(key: string, value: unknown): void {
    this.privateState.set(key, value)
  }

  getPrivate<T>(key: string): T | undefined {
    return this.privateState.get(key) as T | undefined
  }
}

// Agent that uses shared and private state
async function researchAgentWithState(ctx: AgentContext): Promise<void> {
  ctx.setStatus('in_progress')

  try {
    // Private: track search queries (other agents do not need this)
    const queriesUsed: string[] = []
    ctx.setPrivate('queries', queriesUsed)

    const result = await generateText({
      model: mistral('mistral-small-latest'),
      maxSteps: 4,
      system: `You are a research agent. Research the topic and report findings.`,
      tools: {
        search: {
          description: 'Search for information',
          parameters: z.object({ query: z.string() }),
          execute: async ({ query }) => {
            queriesUsed.push(query)
            return `Results for "${query}": [relevant findings]`
          },
        },
      },
      prompt: `Research: ${ctx.getTask()}`,
    })

    // Shared: add findings for other agents to use
    ctx.addFinding('research_summary', result.text)
    ctx.setOutput('research', result.text)
    ctx.setStatus('completed')
  } catch (error) {
    ctx.reportError(error instanceof Error ? error.message : 'Unknown error')
    ctx.setStatus('failed')
  }
}

// Usage
const shared = createSharedState('Analyze the impact of remote work on productivity')
const researchCtx = new AgentContext('researcher', shared)
const writerCtx = new AgentContext('writer', shared)

await researchAgentWithState(researchCtx)

console.log('Shared findings:', Object.fromEntries(shared.findings))
console.log('Status:', Object.fromEntries(shared.status))
console.log('Errors:', shared.errors)
```

> **Beginner Note:** Shared state is like a shared whiteboard that all team members can write on and read from. Private state is like each team member's personal notebook. Keep the shared whiteboard focused on what everyone needs to know.

---

## Section 5: Delegation Strategies

### By Capability

Route sub-tasks to agents based on what they can do:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface AgentCapability {
  name: string
  skills: string[]
  model: string
  execute: (task: string) => Promise<string>
}

const agents: AgentCapability[] = [
  {
    name: 'data-analyst',
    skills: ['data analysis', 'statistics', 'visualization', 'SQL'],
    model: 'mistral-small-latest',
    execute: async task => {
      const result = await generateText({
        model: mistral('mistral-small-latest'),
        system: 'You are a data analyst. Analyze data, compute statistics, and provide insights.',
        prompt: task,
      })
      return result.text
    },
  },
  {
    name: 'code-writer',
    skills: ['TypeScript', 'Python', 'code generation', 'debugging', 'algorithms'],
    model: 'mistral-small-latest',
    execute: async task => {
      const result = await generateText({
        model: mistral('mistral-small-latest'),
        system: 'You are a software engineer. Write clean, well-tested code.',
        prompt: task,
      })
      return result.text
    },
  },
  {
    name: 'content-creator',
    skills: ['writing', 'editing', 'blog posts', 'documentation', 'marketing copy'],
    model: 'mistral-small-latest',
    execute: async task => {
      const result = await generateText({
        model: mistral('mistral-small-latest'),
        system: 'You are a content creator. Write engaging, clear content.',
        prompt: task,
      })
      return result.text
    },
  },
]

// Capability-based routing
async function routeByCapability(task: string): Promise<string> {
  // Ask the LLM to analyze the task and match it to an agent
  const { output: routing } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        selectedAgent: z.string().describe('Name of the best agent for this task'),
        reasoning: z.string().describe('Why this agent is the best fit'),
        requiredSkills: z.array(z.string()).describe('Skills needed for this task'),
      }),
    }),
    prompt: `Given these available agents:
${agents.map(a => `- ${a.name}: ${a.skills.join(', ')}`).join('\n')}

Which agent should handle this task?
Task: ${task}`,
  })

  console.log(`Routing to: ${routing.selectedAgent} (${routing.reasoning})`)

  const agent = agents.find(a => a.name === routing.selectedAgent)
  if (!agent) {
    throw new Error(`Agent "${routing.selectedAgent}" not found`)
  }

  return agent.execute(task)
}

const result = await routeByCapability(
  'Write a TypeScript function that calculates the standard deviation of an array of numbers'
)
console.log(result)
```

### By Topic

Route based on the domain or topic of the request:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface TopicRouter {
  topics: Map<string, (input: string) => Promise<string>>
}

function createTopicRouter(): TopicRouter {
  const topics = new Map<string, (input: string) => Promise<string>>()

  topics.set('billing', async input => {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      system: `You are a billing specialist. You handle questions about invoices,
payments, subscriptions, and pricing. Be precise with numbers and dates.`,
      prompt: input,
    })
    return result.text
  })

  topics.set('technical', async input => {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      maxSteps: 3,
      system: `You are a technical support engineer. You help with API issues,
configuration problems, and debugging. Ask clarifying questions when needed.`,
      tools: {
        lookupDocs: {
          description: 'Search the documentation',
          parameters: z.object({ query: z.string() }),
          execute: async ({ query }) => `Documentation for "${query}": [relevant docs]`,
        },
      },
      prompt: input,
    })
    return result.text
  })

  topics.set('general', async input => {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      system: `You are a general customer support agent. Be helpful and friendly.
If the question is about billing or technical issues, note that a specialist
could provide better help.`,
      prompt: input,
    })
    return result.text
  })

  return { topics }
}

async function routeByTopic(router: TopicRouter, userMessage: string): Promise<{ topic: string; response: string }> {
  // Classify the topic
  const { output: classification } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        topic: z.enum(['billing', 'technical', 'general']).describe('The topic category'),
        confidence: z.number().min(0).max(1),
        reasoning: z.string(),
      }),
    }),
    prompt: `Classify this customer message into a topic:
"${userMessage}"

Categories: billing (invoices, payments, pricing), technical (API, bugs, configuration), general (everything else)`,
  })

  console.log(`Topic: ${classification.topic} (confidence: ${classification.confidence})`)

  const handler = router.topics.get(classification.topic)
  if (!handler) {
    throw new Error(`No handler for topic: ${classification.topic}`)
  }

  const response = await handler(userMessage)
  return { topic: classification.topic, response }
}

const router = createTopicRouter()
const result = await routeByTopic(router, 'My API key stopped working after I regenerated it')
console.log(`[${result.topic}] ${result.response}`)
```

---

## Section 6: Parallel Agent Execution

### Running Independent Agents Concurrently

When sub-tasks are independent, run them in parallel to reduce total latency:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface ParallelTask {
  name: string
  execute: () => Promise<string>
}

interface ParallelResult {
  name: string
  result: string
  durationMs: number
  success: boolean
  error?: string
}

async function runParallel(tasks: ParallelTask[], maxConcurrency: number = 5): Promise<ParallelResult[]> {
  const results: ParallelResult[] = []
  const startTime = Date.now()

  // Simple parallel execution with concurrency limit
  const chunks: ParallelTask[][] = []
  for (let i = 0; i < tasks.length; i += maxConcurrency) {
    chunks.push(tasks.slice(i, i + maxConcurrency))
  }

  for (const chunk of chunks) {
    const chunkResults = await Promise.allSettled(
      chunk.map(async task => {
        const taskStart = Date.now()
        try {
          const result = await task.execute()
          return {
            name: task.name,
            result,
            durationMs: Date.now() - taskStart,
            success: true,
          }
        } catch (error) {
          return {
            name: task.name,
            result: '',
            durationMs: Date.now() - taskStart,
            success: false,
            error: error instanceof Error ? error.message : 'Unknown error',
          }
        }
      })
    )

    for (const result of chunkResults) {
      if (result.status === 'fulfilled') {
        results.push(result.value)
      } else {
        results.push({
          name: 'unknown',
          result: '',
          durationMs: 0,
          success: false,
          error: result.reason?.message || 'Promise rejected',
        })
      }
    }
  }

  const totalMs = Date.now() - startTime
  console.log(
    `Parallel execution: ${tasks.length} tasks in ${totalMs}ms (vs ~${results.reduce((sum, r) => sum + r.durationMs, 0)}ms sequential)`
  )

  return results
}

// Example: Research multiple topics in parallel
const topics = ['AI in healthcare', 'AI in education', 'AI in finance', 'AI in transportation']

const tasks: ParallelTask[] = topics.map(topic => ({
  name: topic,
  execute: async () => {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      maxSteps: 3,
      system: 'You are a research specialist. Provide concise findings.',
      tools: {
        search: {
          description: 'Search for information',
          parameters: z.object({ query: z.string() }),
          execute: async ({ query }) => `Results for "${query}": [relevant findings about ${topic}]`,
        },
      },
      prompt: `Research the current state of ${topic}. Focus on key trends and challenges.`,
    })
    return result.text
  },
}))

const results = await runParallel(tasks, 3)

for (const result of results) {
  console.log(`\n[${result.name}] (${result.durationMs}ms, ${result.success ? 'OK' : 'FAILED'}):`)
  console.log(result.result.slice(0, 200))
}
```

### Map-Reduce with Agents

A common parallel pattern: map a task across multiple agents, then reduce the results:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function mapReduceAgents(
  items: string[],
  mapPrompt: (item: string) => string,
  reducePrompt: (results: string[]) => string,
  concurrency: number = 3
): Promise<string> {
  // MAP: Process each item in parallel
  console.log(`Map phase: processing ${items.length} items...`)
  const mapResults: string[] = []

  const chunks: string[][] = []
  for (let i = 0; i < items.length; i += concurrency) {
    chunks.push(items.slice(i, i + concurrency))
  }

  for (const chunk of chunks) {
    const results = await Promise.all(
      chunk.map(async item => {
        const result = await generateText({
          model: mistral('mistral-small-latest'),
          prompt: mapPrompt(item),
        })
        return result.text
      })
    )
    mapResults.push(...results)
  }

  // REDUCE: Synthesize all results
  console.log('Reduce phase: synthesizing results...')
  const finalResult = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: reducePrompt(mapResults),
  })

  return finalResult.text
}

// Example: Analyze multiple companies
const companies = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Meta']

const analysis = await mapReduceAgents(
  companies,
  company =>
    `Analyze ${company}'s AI strategy in 2025. Focus on: key products, investments, and competitive advantages. Be concise (3-5 bullet points).`,
  results =>
    `Synthesize these individual company analyses into a comparative overview of Big Tech AI strategies:

${results.map((r, i) => `### ${companies[i]}\n${r}`).join('\n\n')}

Provide: 1) Common themes, 2) Key differentiators, 3) Overall industry direction.`
)

console.log(analysis)
```

> **Advanced Note:** Be mindful of API rate limits when running agents in parallel. Most providers have tokens-per-minute and requests-per-minute limits. Set `maxConcurrency` based on your provider's rate limits. Consider adding exponential backoff for 429 (rate limit) errors.

---

## Section 7: Agent Handoff

### Transferring Conversations Between Agents

Sometimes an agent needs to transfer a conversation to a different specialist. This is common in customer support: a general agent detects a billing question and hands off to the billing specialist.

```typescript
import { generateText, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface HandoffRequest {
  targetAgent: string
  reason: string
  conversationSummary: string
  userIntent: string
  fullHistory: ModelMessage[]
}

interface AgentSpec {
  name: string
  system: string
  tools: Record<string, any>
  canHandoff: string[] // which agents this one can hand off to
}

class ConversationRouter {
  private agents = new Map<string, AgentSpec>()

  register(spec: AgentSpec): void {
    this.agents.set(spec.name, spec)
  }

  async runWithHandoff(
    startAgent: string,
    userMessage: string,
    maxHandoffs: number = 3
  ): Promise<{ finalAgent: string; response: string; handoffs: string[] }> {
    let currentAgent = startAgent
    const handoffHistory: string[] = []
    const messages: ModelMessage[] = [{ role: 'user', content: userMessage }]

    for (let i = 0; i <= maxHandoffs; i++) {
      const spec = this.agents.get(currentAgent)
      if (!spec) {
        throw new Error(`Agent "${currentAgent}" not found`)
      }

      console.log(`[Router] Current agent: ${currentAgent}`)

      // Add handoff tool if this agent can hand off
      const tools: Record<string, any> = { ...spec.tools }
      if (spec.canHandoff.length > 0) {
        tools.handoff = {
          description: `Transfer this conversation to a specialist agent. Available: ${spec.canHandoff.join(', ')}. Use this when the user's request is outside your expertise.`,
          parameters: z.object({
            targetAgent: z.enum(spec.canHandoff as [string, ...string[]]).describe('The agent to transfer to'),
            reason: z.string().describe('Why you are transferring'),
            summary: z.string().describe('Summary of the conversation so far for the receiving agent'),
          }),
        }
      }

      const result = await generateText({
        model: mistral('mistral-small-latest'),
        system: spec.system,
        messages,
        tools,
        maxSteps: 3,
      })

      // Check for handoff
      const handoffCall = result.steps.flatMap(s => s.toolCalls).find(tc => tc.toolName === 'handoff')

      if (handoffCall) {
        const args = handoffCall.args as {
          targetAgent: string
          reason: string
          summary: string
        }
        console.log(`[Router] Handoff: ${currentAgent} -> ${args.targetAgent} (${args.reason})`)
        handoffHistory.push(`${currentAgent} -> ${args.targetAgent}`)

        // Prepare messages for the new agent
        messages.length = 0
        messages.push({
          role: 'user',
          content: `[Transferred from ${currentAgent}. Context: ${args.summary}]\n\nOriginal request: ${userMessage}`,
        })

        currentAgent = args.targetAgent
        continue
      }

      // No handoff — this agent handled it
      return {
        finalAgent: currentAgent,
        response: result.text,
        handoffs: handoffHistory,
      }
    }

    return {
      finalAgent: currentAgent,
      response: 'Maximum handoffs reached. Please try rephrasing your request.',
      handoffs: handoffHistory,
    }
  }
}

// Set up agents
const router = new ConversationRouter()

router.register({
  name: 'triage',
  system: `You are a front-line support agent. Greet the user and understand their issue.
If it is about billing, hand off to the billing agent.
If it is about technical issues, hand off to the technical agent.
For general questions, answer directly.`,
  tools: {},
  canHandoff: ['billing', 'technical'],
})

router.register({
  name: 'billing',
  system: `You are a billing specialist. Handle all billing-related questions:
invoices, payments, subscriptions, refunds, pricing changes.
You have access to the billing system.`,
  tools: {
    lookupAccount: {
      description: "Look up a customer's billing account",
      parameters: z.object({ customerId: z.string() }),
      execute: async ({ customerId }: { customerId: string }) =>
        `Account ${customerId}: Plan: Pro, Balance: $0.00, Next billing: March 15`,
    },
  },
  canHandoff: ['technical'],
})

router.register({
  name: 'technical',
  system: `You are a technical support engineer. Handle API issues, configuration
problems, and debugging questions. You have access to logs and documentation.`,
  tools: {
    checkLogs: {
      description: 'Check system logs for a customer',
      parameters: z.object({ customerId: z.string() }),
      execute: async ({ customerId }: { customerId: string }) =>
        `Logs for ${customerId}: No errors in the last 24 hours.`,
    },
  },
  canHandoff: ['billing'],
})

// Test handoff
const result = await router.runWithHandoff('triage', 'Hi, I was charged twice on my last invoice. Can you help?')

console.log(`Final agent: ${result.finalAgent}`)
console.log(`Handoffs: ${result.handoffs.join(' -> ')}`)
console.log(`Response: ${result.response}`)
```

> **Beginner Note:** Agent handoff is like being transferred to a different department when you call customer support. The key is passing enough context so you do not have to repeat yourself — the receiving agent should know what you already discussed.

---

## Section 8: Error Handling

### Sub-Agent Failures

In a multi-agent system, individual agents can fail. The system must handle these failures gracefully:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface AgentResult {
  agentName: string
  success: boolean
  result?: string
  error?: string
  retries: number
}

async function runWithRetry(
  agentName: string,
  execute: () => Promise<string>,
  maxRetries: number = 2,
  fallback?: () => Promise<string>
): Promise<AgentResult> {
  let lastError: string | undefined

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      if (attempt > 0) {
        console.log(`[${agentName}] Retry ${attempt}/${maxRetries}...`)
      }

      const result = await execute()
      return {
        agentName,
        success: true,
        result,
        retries: attempt,
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : 'Unknown error'
      console.error(`[${agentName}] Attempt ${attempt + 1} failed: ${lastError}`)

      // Wait before retrying (exponential backoff)
      if (attempt < maxRetries) {
        const delayMs = Math.pow(2, attempt) * 1000
        await new Promise(resolve => setTimeout(resolve, delayMs))
      }
    }
  }

  // All retries exhausted — try fallback
  if (fallback) {
    console.log(`[${agentName}] Trying fallback...`)
    try {
      const fallbackResult = await fallback()
      return {
        agentName,
        success: true,
        result: fallbackResult,
        retries: maxRetries + 1,
      }
    } catch (error) {
      lastError = error instanceof Error ? error.message : 'Unknown error'
    }
  }

  return {
    agentName,
    success: false,
    error: lastError,
    retries: maxRetries,
  }
}

// Orchestrator with error handling
async function resilientOrchestrator(task: string): Promise<string> {
  const subtasks = [
    {
      name: 'researcher',
      execute: async () => {
        const result = await generateText({
          model: mistral('mistral-small-latest'),
          maxSteps: 3,
          system: 'You are a researcher. Gather key facts.',
          tools: {
            search: {
              description: 'Search for information',
              parameters: z.object({ query: z.string() }),
              execute: async ({ query }: { query: string }) => `Results for "${query}": [research findings]`,
            },
          },
          prompt: `Research: ${task}`,
        })
        return result.text
      },
      fallback: async () => {
        // Fallback: use the model's training data instead of search
        const result = await generateText({
          model: mistral('mistral-small-latest'),
          system: 'Provide what you know about this topic from your training data.',
          prompt: `What do you know about: ${task}`,
        })
        return `[From training data - not real-time] ${result.text}`
      },
    },
    {
      name: 'writer',
      execute: async () => {
        const result = await generateText({
          model: mistral('mistral-small-latest'),
          system: 'You are a writer. Create clear, engaging content.',
          prompt: `Write about: ${task}`,
        })
        return result.text
      },
      fallback: undefined,
    },
  ]

  // Run all sub-tasks with error handling
  const results: AgentResult[] = []
  for (const subtask of subtasks) {
    const result = await runWithRetry(subtask.name, subtask.execute, 2, subtask.fallback)
    results.push(result)

    if (!result.success) {
      console.error(`[Orchestrator] ${subtask.name} failed permanently: ${result.error}`)
    }
  }

  // Check overall status
  const failures = results.filter(r => !r.success)
  if (failures.length === results.length) {
    return 'All agents failed. Unable to complete the task.'
  }

  if (failures.length > 0) {
    console.warn(`[Orchestrator] Partial completion: ${failures.map(f => f.agentName).join(', ')} failed`)
  }

  // Synthesize from successful results
  const successfulResults = results
    .filter(r => r.success)
    .map(r => `[${r.agentName}]: ${r.result}`)
    .join('\n\n')

  const finalResult = await generateText({
    model: mistral('mistral-small-latest'),
    system: 'Synthesize the available results into a coherent answer. Note if any information is missing.',
    prompt: `Task: ${task}\n\nAvailable results:\n${successfulResults}\n\nFailed agents: ${failures.map(f => f.agentName).join(', ') || 'none'}`,
  })

  return finalResult.text
}

const output = await resilientOrchestrator('Summarize the latest trends in quantum computing')
console.log(output)
```

### Circuit Breaker Pattern

Prevent cascading failures by stopping calls to a consistently failing agent:

```typescript
class CircuitBreaker {
  private failures = 0
  private lastFailureTime = 0
  private state: 'closed' | 'open' | 'half-open' = 'closed'

  constructor(
    private threshold: number = 3,
    private resetTimeMs: number = 30000
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      // Check if enough time has passed to try again
      if (Date.now() - this.lastFailureTime > this.resetTimeMs) {
        this.state = 'half-open'
        console.log('[CircuitBreaker] Trying half-open...')
      } else {
        throw new Error('Circuit breaker is open — agent is unavailable')
      }
    }

    try {
      const result = await fn()
      // Success — reset
      this.failures = 0
      this.state = 'closed'
      return result
    } catch (error) {
      this.failures++
      this.lastFailureTime = Date.now()

      if (this.failures >= this.threshold) {
        this.state = 'open'
        console.log(`[CircuitBreaker] Opened after ${this.failures} failures`)
      }

      throw error
    }
  }

  getState(): string {
    return this.state
  }
}

// Usage: wrap agent calls in a circuit breaker
const researchBreaker = new CircuitBreaker(3, 30000)

try {
  const result = await researchBreaker.execute(async () => {
    const response = await generateText({
      model: mistral('mistral-small-latest'),
      prompt: 'Research AI trends',
    })
    return response.text
  })
  console.log(result)
} catch (error) {
  console.log(`Research agent unavailable (circuit: ${researchBreaker.getState()})`)
}
```

> **Advanced Note:** In production multi-agent systems, combine retries, circuit breakers, and fallbacks. The circuit breaker prevents wasting tokens on a consistently failing agent, retries handle transient errors, and fallbacks provide degraded but functional responses when an agent is down.

---

## Quiz

### Question 1 (Easy)

What is the main advantage of splitting a complex task across multiple agents instead of using one agent?

- A) Multiple agents use fewer tokens overall
- B) Each agent has a focused prompt and toolset, leading to better performance on its specific sub-task
- C) Multiple agents are always faster than a single agent
- D) Multiple agents do not need error handling

**Answer: B** — Each agent gets a focused system prompt and a curated set of tools for its specific role. This avoids prompt dilution (one prompt trying to cover everything) and tool overload (too many tools confusing the model). Multi-agent systems often use more tokens (A is wrong), are not always faster (C is wrong), and definitely need error handling (D is wrong).

---

### Question 2 (Medium)

In the orchestrator-worker pattern, what is the orchestrator's primary responsibility?

- A) Executing all the actual work itself
- B) Breaking the task into sub-tasks, delegating to workers, and synthesizing results
- C) Providing tools to the worker agents
- D) Storing the conversation history for all agents

**Answer: B** — The orchestrator is like a project manager: it analyzes the task, determines which workers should handle which parts, dispatches the work, collects results, and synthesizes them into a final answer. The workers execute the actual work (A). Tools belong to the workers (C). History management can be shared or handled separately (D).

---

### Question 3 (Medium)

When running multiple agents in parallel, what is the primary constraint you must consider?

- A) Parallel agents cannot share any state
- B) API rate limits from the model provider
- C) Parallel agents must all use the same model
- D) Results from parallel agents cannot be combined

**Answer: B** — Most API providers have tokens-per-minute and requests-per-minute limits. Running too many agents in parallel can trigger rate limiting (HTTP 429 errors). Parallel agents can share state through shared data structures (A is wrong), can use different models (C is wrong), and their results are typically combined by the orchestrator (D is wrong).

---

### Question 4 (Hard)

An agent handoff system transfers a user from a triage agent to a billing agent, but the billing agent asks the user to repeat their problem. What is the most likely cause?

- A) The billing agent has a different model than the triage agent
- B) The handoff did not include a conversation summary, so the billing agent has no context
- C) The billing agent's system prompt is too short
- D) The triage agent should not have tools

**Answer: B** — When transferring between agents, the receiving agent needs context about what was already discussed. Without a conversation summary or the original message, the billing agent starts fresh and has to ask the user to explain again. The fix is to pass a summary of the conversation and the user's original request as context to the receiving agent.

---

### Question 5 (Hard)

In a multi-agent system, why is a circuit breaker pattern useful?

- A) It limits the total number of agents that can run simultaneously
- B) It prevents repeatedly calling a consistently failing agent, saving tokens and reducing latency
- C) It ensures all agents complete before the orchestrator synthesizes results
- D) It encrypts communication between agents

**Answer: B** — A circuit breaker tracks consecutive failures for an agent. After a threshold is reached, it "opens" and immediately rejects further calls to that agent for a cooldown period. This prevents wasting tokens and time on an agent that is consistently failing (e.g., due to a downstream API outage). After the cooldown, it tries again ("half-open") to see if the issue resolved.

---

## Exercises

### Exercise 1: Orchestrator with Research, Writing, and Review Agents

**Objective:** Build a multi-agent system where an orchestrator delegates to three specialized agents: a researcher, a writer, and a reviewer. The system produces a polished article on a given topic.

**Specification:**

1. Create a file `src/exercises/m15/ex01-multi-agent-pipeline.ts`
2. Export an async function `multiAgentArticle(topic: string, options?: PipelineOptions): Promise<PipelineResult>`
3. Define the types:

```typescript
interface PipelineOptions {
  maxResearchSteps?: number // default: 5
  reviewRounds?: number // default: 1
  verbose?: boolean // default: false
}

interface AgentOutput {
  agentName: string
  output: string
  durationMs: number
  success: boolean
  error?: string
}

interface PipelineResult {
  finalArticle: string
  agentOutputs: AgentOutput[]
  totalDurationMs: number
  reviewFeedback: string[]
}
```

4. Implement three worker agents:
   - **Research Agent**: Uses search tools to gather information, returns a structured research brief
   - **Writing Agent**: Takes the research brief and produces a well-structured article
   - **Review Agent**: Reviews the article, provides feedback, and scores quality (1-10)

5. The orchestrator must:
   - Run the research agent first
   - Pass research results to the writing agent
   - Pass the draft to the review agent
   - If `reviewRounds > 1`, send the review feedback back to the writer for revision
   - Handle agent failures with retries (max 2 retries per agent)

6. If any agent fails after retries, the orchestrator should continue with available results and note the failure

**Example usage:**

```typescript
const result = await multiAgentArticle('The future of quantum computing', {
  maxResearchSteps: 5,
  reviewRounds: 2,
  verbose: true,
})

console.log(result.finalArticle)
console.log(`Agents used: ${result.agentOutputs.map(a => a.agentName).join(', ')}`)
console.log(`Review feedback: ${result.reviewFeedback.join('; ')}`)
```

**Test specification:**

```typescript
// tests/exercises/m15/ex01-multi-agent-pipeline.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 15: Multi-Agent Article Pipeline', () => {
  it('should produce a final article', async () => {
    const result = await multiAgentArticle('TypeScript best practices')
    expect(result.finalArticle).toBeTruthy()
    expect(result.finalArticle.length).toBeGreaterThan(200)
  })

  it('should record outputs from all three agents', async () => {
    const result = await multiAgentArticle('Cloud computing trends')
    const agentNames = result.agentOutputs.map(a => a.agentName)
    expect(agentNames).toContain('researcher')
    expect(agentNames).toContain('writer')
    expect(agentNames).toContain('reviewer')
  })

  it('should include review feedback', async () => {
    const result = await multiAgentArticle('Machine learning basics', {
      reviewRounds: 1,
    })
    expect(result.reviewFeedback.length).toBeGreaterThan(0)
  })

  it('should handle multiple review rounds', async () => {
    const result = await multiAgentArticle('Web development', {
      reviewRounds: 2,
    })
    expect(result.reviewFeedback.length).toBe(2)
  })

  it('should complete even if one agent fails', async () => {
    // With a mock that makes the review agent fail
    const result = await multiAgentArticle('Testing error handling')
    expect(result.finalArticle).toBeTruthy()
  })
})
```

---

### Exercise 2: Customer Support Routing System

**Objective:** Build a customer support system with agent handoff. A triage agent classifies incoming messages and routes them to the appropriate specialist.

**Specification:**

1. Create a file `src/exercises/m15/ex02-support-router.ts`
2. Export an async function `handleSupportRequest(message: string): Promise<SupportResult>`
3. Define the types:

```typescript
interface SupportResult {
  finalAgent: string
  response: string
  handoffChain: string[] // e.g., ["triage", "billing"]
  classification: string
  totalDurationMs: number
}
```

4. Implement at least three agents:
   - **Triage Agent**: Classifies the request and routes to the right specialist
   - **Billing Agent**: Handles billing, payment, and subscription questions
   - **Technical Agent**: Handles API issues, bugs, and configuration problems

5. Each specialist should have at least one tool relevant to their domain

6. The system must support handoffs between agents (a specialist can re-route if they realize the issue belongs to another specialist)

**Test specification:**

```typescript
// tests/exercises/m15/ex02-support-router.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 15: Support Router', () => {
  it('should route billing questions to the billing agent', async () => {
    const result = await handleSupportRequest('I was double-charged on my last invoice')
    expect(result.finalAgent).toBe('billing')
  })

  it('should route technical questions to the technical agent', async () => {
    const result = await handleSupportRequest('My API calls are returning 500 errors')
    expect(result.finalAgent).toBe('technical')
  })

  it('should track the handoff chain', async () => {
    const result = await handleSupportRequest('I need help with my account')
    expect(result.handoffChain.length).toBeGreaterThan(0)
    expect(result.handoffChain[0]).toBe('triage')
  })

  it('should always produce a response', async () => {
    const result = await handleSupportRequest('Hello, can you help me?')
    expect(result.response).toBeTruthy()
  })
})
```

> **Looking Ahead: Agent SDKs** — This module teaches multi-agent orchestration from scratch, which is valuable for understanding the patterns. In production, consider the official Agent SDKs: Anthropic's Claude Agent SDK, OpenAI's Agents SDK, and Mistral's Agents API all provide built-in primitives for structured handoffs between agents, built-in guardrails, tracing, and orchestration — handling many of the patterns you've implemented manually here. Mistral's Agents API additionally offers built-in connectors (web search, code execution, image generation), persistent memory across conversations, and native multi-agent orchestration.

> **Local Alternative (Ollama):** Multi-agent orchestration works with `ollama('qwen3.5')`. The orchestrator-worker pattern, delegation, and shared state are all code-level patterns independent of the model provider. You can even mix providers — use a capable API model as the orchestrator and local models as cheaper workers.

---

## Summary

In this module, you learned:

1. **Why multiple agents:** Single agents suffer from prompt dilution and tool overload. Multi-agent systems assign focused roles with curated tools and prompts.
2. **Orchestrator-worker pattern:** An orchestrator breaks tasks into sub-tasks, delegates to specialized workers, and synthesizes results. Workers can be registered dynamically.
3. **Agent communication:** Structured handoff documents and message-passing protocols enable agents to share context without coupling.
4. **Shared state:** Use shared state for task-level information (goals, findings, status) and private state for agent-internal details (reasoning traces, scratchpads).
5. **Delegation strategies:** Route by capability (matching skills to tasks) or by topic (matching domains to specialists).
6. **Parallel execution:** Run independent agents concurrently with concurrency limits and map-reduce patterns for throughput.
7. **Agent handoff:** Transfer conversations between specialists with context summaries so users do not repeat themselves.
8. **Error handling:** Retries with exponential backoff, fallback agents, and circuit breakers create resilient multi-agent systems.

In Module 16, you will learn about workflows and chains — a more deterministic alternative to autonomous multi-agent systems for tasks with well-defined steps.
