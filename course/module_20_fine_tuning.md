# Module 20: Fine-tuning

## Learning Objectives

- Determine when fine-tuning is the right approach compared to prompt engineering and RAG
- Prepare high-quality training datasets in JSONL format with proper conversation structure
- Apply best practices for training data diversity, balance, and quality filtering
- Use fine-tuning APIs from OpenAI and understand Anthropic's approaches
- Configure hyperparameters (epochs, learning rate, batch size) for optimal training
- Evaluate fine-tuned models against base models using held-out test sets
- Iterate on fine-tuning through dataset improvement and augmentation
- Perform cost analysis comparing training investment against inference savings

---

## Why Should I Care?

Prompt engineering gets you remarkably far. RAG adds external knowledge. But sometimes neither is enough. Your model needs to adopt a specific writing style, follow a complex output format consistently, handle domain-specific terminology, or perform a narrow task with high reliability — and the prompt is already 2,000 tokens long, eating into your context window and your budget on every single call.

Fine-tuning trains the model to internalize your instructions, your style, and your domain knowledge. A well fine-tuned model can replace a 2,000-token system prompt with a 50-token one. It can follow your output format without elaborate examples. It can handle domain jargon without long glossaries in the prompt.

But fine-tuning is not free. It requires high-quality training data, costs money to train, creates a model version you must maintain, and can go wrong in subtle ways (overfitting, catastrophic forgetting, data quality issues). Knowing when to fine-tune — and when not to — is as important as knowing how.

This module teaches the complete fine-tuning workflow: deciding whether to fine-tune, preparing data, training, evaluating, and iterating. We use the Vercel AI SDK to interact with fine-tuned models, and we cover the APIs for training itself.

---

## Connection to Other Modules

- **Module 2 (Prompt Engineering)** is the alternative to fine-tuning — always try prompt engineering first.
- **Module 9-10 (RAG)** is the other major alternative — RAG adds knowledge, fine-tuning changes behavior.
- **Module 19 (Evals)** provides the evaluation framework you need to measure fine-tuning success.
- **Module 22 (Cost Optimization)** connects to the cost reduction benefits of fine-tuning (shorter prompts, cheaper models).
- **Module 3 (Structured Output)** shows format requirements that fine-tuning can internalize.

---

## Section 1: When to Fine-tune

### The Decision Tree

Fine-tuning is one tool in a hierarchy of approaches. Each approach has different cost, effort, and capability profiles.

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Decision framework: should you fine-tune?
interface RequirementAnalysis {
  requirement: string
  promptEngineeringViable: boolean
  ragViable: boolean
  fineTuningNeeded: boolean
  reasoning: string
}

const requirements: RequirementAnalysis[] = [
  {
    requirement: 'Model should know about our internal product catalog',
    promptEngineeringViable: false,
    ragViable: true,
    fineTuningNeeded: false,
    reasoning:
      'External knowledge is best handled by RAG. Fine-tuning for knowledge is expensive and the knowledge becomes stale.',
  },
  {
    requirement: 'Model should always respond in a specific JSON format',
    promptEngineeringViable: true,
    ragViable: false,
    fineTuningNeeded: false,
    reasoning:
      'Structured output with Zod schemas (Module 3) handles this well. Fine-tune only if the model consistently fails to follow the format.',
  },
  {
    requirement: "Model should write in our brand's distinctive voice",
    promptEngineeringViable: true, // partially
    ragViable: false,
    fineTuningNeeded: true,
    reasoning:
      'Style is difficult to fully capture in a prompt. Fine-tuning on examples of your brand voice is highly effective.',
  },
  {
    requirement: 'Model should classify medical documents using ICD-10 codes',
    promptEngineeringViable: false,
    ragViable: true, // partially
    fineTuningNeeded: true,
    reasoning:
      'Domain-specific classification with hundreds of categories benefits enormously from fine-tuning on labeled examples.',
  },
  {
    requirement: 'Model should handle 10,000 requests per minute cheaply',
    promptEngineeringViable: false,
    ragViable: false,
    fineTuningNeeded: true,
    reasoning:
      'Fine-tuning a smaller model to replace a larger one with long prompts can dramatically reduce per-request cost at scale.',
  },
]

// Automated decision helper
async function analyzeFineTuningNeed(description: string): Promise<{
  recommendation: 'prompt_engineering' | 'rag' | 'fine_tuning' | 'combination'
  confidence: number
  reasoning: string
  prerequisites: string[]
}> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        recommendation: z.enum(['prompt_engineering', 'rag', 'fine_tuning', 'combination']),
        confidence: z.number().min(0).max(1),
        reasoning: z.string(),
        prerequisites: z.array(z.string()),
      }),
    }),
    system: `You are an ML engineering advisor. Analyze whether a use case requires fine-tuning.

Decision hierarchy (try in order):
1. Prompt engineering — cheapest, fastest iteration
2. RAG — when external/dynamic knowledge is needed
3. Fine-tuning — when behavior/style must change fundamentally
4. Combination — complex cases requiring multiple approaches

Fine-tuning is justified when:
- Prompt engineering alone cannot achieve the required quality
- You need to reduce prompt size for cost/latency
- Consistent style/format is critical and prompt-based approaches are unreliable
- Domain-specific performance requires specialized training
- You have at least 50-100 high-quality training examples`,
    prompt: description,
  })

  return output!
}
```

> **Beginner Note:** Think of it this way — prompt engineering tells the model what to do each time, RAG gives it information to work with, and fine-tuning changes what the model is. You prompt-engineer a model to write like Shakespeare; you fine-tune a model to be a Shakespeare-writing model.

### Cost-Benefit Analysis

```typescript
interface FineTuningCostBenefit {
  // Current approach costs
  currentPromptTokens: number
  currentRequestsPerDay: number
  currentCostPerToken: number

  // Fine-tuning costs
  trainingDataSize: number
  trainingCost: number
  fineTunedPromptTokens: number
  fineTunedCostPerToken: number

  // Computed
  currentDailyCost: number
  fineTunedDailyCost: number
  dailySavings: number
  breakEvenDays: number
}

function computeFineTuningROI(params: {
  currentPromptTokens: number
  currentRequestsPerDay: number
  currentCostPerMillionTokens: number
  trainingExamples: number
  avgExampleTokens: number
  trainingCostPerMillionTokens: number
  epochs: number
  fineTunedPromptTokens: number
  fineTunedCostPerMillionTokens: number
}): FineTuningCostBenefit {
  const currentCostPerToken = params.currentCostPerMillionTokens / 1_000_000
  const fineTunedCostPerToken = params.fineTunedCostPerMillionTokens / 1_000_000

  const currentDailyCost = params.currentPromptTokens * params.currentRequestsPerDay * currentCostPerToken

  const fineTunedDailyCost = params.fineTunedPromptTokens * params.currentRequestsPerDay * fineTunedCostPerToken

  const trainingTokens = params.trainingExamples * params.avgExampleTokens * params.epochs
  const trainingCost = trainingTokens * (params.trainingCostPerMillionTokens / 1_000_000)

  const dailySavings = currentDailyCost - fineTunedDailyCost
  const breakEvenDays = dailySavings > 0 ? Math.ceil(trainingCost / dailySavings) : Infinity

  return {
    currentPromptTokens: params.currentPromptTokens,
    currentRequestsPerDay: params.currentRequestsPerDay,
    currentCostPerToken,
    trainingDataSize: params.trainingExamples,
    trainingCost,
    fineTunedPromptTokens: params.fineTunedPromptTokens,
    fineTunedCostPerToken,
    currentDailyCost,
    fineTunedDailyCost,
    dailySavings,
    breakEvenDays,
  }
}

// Example calculation
const roi = computeFineTuningROI({
  currentPromptTokens: 2000, // Long system prompt with examples
  currentRequestsPerDay: 5000,
  currentCostPerMillionTokens: 3.0, // Claude Sonnet input pricing
  trainingExamples: 500,
  avgExampleTokens: 500,
  trainingCostPerMillionTokens: 25.0, // Training is more expensive
  epochs: 3,
  fineTunedPromptTokens: 200, // Much shorter prompt after fine-tuning
  fineTunedCostPerMillionTokens: 3.0,
})

console.log(`Current daily cost: $${roi.currentDailyCost.toFixed(2)}`)
console.log(`Fine-tuned daily cost: $${roi.fineTunedDailyCost.toFixed(2)}`)
console.log(`Daily savings: $${roi.dailySavings.toFixed(2)}`)
console.log(`Training cost: $${roi.trainingCost.toFixed(2)}`)
console.log(`Break-even: ${roi.breakEvenDays} days`)
```

> **Advanced Note:** Fine-tuning ROI calculations should also include engineering time for data preparation, training iteration, and ongoing maintenance. A model that costs less to run but requires a week of engineering time per update may not be worth it for a low-traffic application.

---

## Section 2: Dataset Preparation

### JSONL Format

The standard format for fine-tuning datasets is JSONL (JSON Lines) — one JSON object per line, each representing a training example.

```typescript
import { writeFileSync, readFileSync } from 'fs'

// OpenAI fine-tuning format
interface FineTuningMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

interface FineTuningExample {
  messages: FineTuningMessage[]
}

// Example: customer support fine-tuning data
const trainingExamples: FineTuningExample[] = [
  {
    messages: [
      {
        role: 'system',
        content: 'You are a helpful customer support agent for TechCorp.',
      },
      {
        role: 'user',
        content: 'My laptop screen is flickering after the latest update.',
      },
      {
        role: 'assistant',
        content: `I understand how frustrating a flickering screen can be. Let me help you resolve this.

This is a known issue with the recent display driver update. Here are the steps to fix it:

1. Right-click on the desktop and select "Display Settings"
2. Scroll down to "Advanced display settings"
3. Click "Display adapter properties"
4. Go to the "Driver" tab and select "Roll Back Driver"

If rolling back the driver does not resolve the flickering, please contact us at support@techcorp.com with your device model number, and we will arrange a remote diagnostic session.

Is there anything else I can help you with?`,
      },
    ],
  },
  {
    messages: [
      {
        role: 'system',
        content: 'You are a helpful customer support agent for TechCorp.',
      },
      {
        role: 'user',
        content: 'I want to cancel my subscription. This product is terrible.',
      },
      {
        role: 'assistant',
        content: `I'm sorry to hear you're not satisfied with your experience. I want to make sure we handle this properly for you.

I can process your cancellation right away. Before I do, would you mind sharing what specifically was not working well? Your feedback helps us improve.

To cancel your subscription:
- Your service will remain active until the end of your current billing period
- You will receive a confirmation email within 24 hours
- Any data you've stored will be available for download for 30 days after cancellation

Would you like me to proceed with the cancellation?`,
      },
    ],
  },
]

// Write to JSONL format
function writeTrainingData(examples: FineTuningExample[], outputPath: string): void {
  const lines = examples.map(ex => JSON.stringify(ex))
  writeFileSync(outputPath, lines.join('\n') + '\n')
  console.log(`Wrote ${examples.length} examples to ${outputPath}`)
}

writeTrainingData(trainingExamples, 'data/training-data/support-v1.jsonl')
```

### Multi-Turn Conversation Data

Fine-tuning on multi-turn conversations teaches the model how to handle context and follow-ups.

```typescript
const multiTurnExample: FineTuningExample = {
  messages: [
    {
      role: 'system',
      content: 'You are a helpful customer support agent for TechCorp.',
    },
    {
      role: 'user',
      content: 'How do I reset my password?',
    },
    {
      role: 'assistant',
      content: `You can reset your password by following these steps:

1. Go to techcorp.com/login
2. Click "Forgot Password" below the login form
3. Enter the email address associated with your account
4. Check your email for a reset link (it may take up to 5 minutes)
5. Click the link and create a new password

Your new password must be at least 12 characters and include a number and special character.

Would you like help with anything else?`,
    },
    {
      role: 'user',
      content: 'I did that but I never got the email.',
    },
    {
      role: 'assistant',
      content: `Let me help you troubleshoot the missing reset email.

Please check the following:

1. **Spam/Junk folder** — Reset emails sometimes get filtered
2. **Correct email address** — Make sure you entered the email associated with your TechCorp account
3. **Wait a few minutes** — Sometimes emails take up to 10 minutes to arrive

If you still have not received it after checking these, it is possible your account uses a different email address. I can look up your account if you provide your username or the last 4 digits of the payment method on file.

Which would you like to try?`,
    },
    {
      role: 'user',
      content: 'My username is john_doe_42',
    },
    {
      role: 'assistant',
      content: `Thank you. I have located your account under username john_doe_42.

I can see that your account is associated with a different email than you may have tried. For security, I can only share that it ends in "...@gmail.com".

I have resent the password reset link to your registered email address. Please check that inbox, including spam folders.

If you no longer have access to that email, I will need to verify your identity through our account recovery process. Would you like me to start that process?`,
    },
  ],
}
```

### Data Conversion Utilities

Convert existing data from various formats into the fine-tuning JSONL format.

```typescript
interface RawQAPair {
  question: string
  answer: string
  context?: string
}

interface CSVRow {
  input: string
  output: string
  category?: string
}

// Convert Q&A pairs to fine-tuning format
function qaToFineTuning(pairs: RawQAPair[], systemPrompt: string): FineTuningExample[] {
  return pairs.map(pair => ({
    messages: [
      { role: 'system', content: systemPrompt },
      {
        role: 'user',
        content: pair.context ? `Context: ${pair.context}\n\nQuestion: ${pair.question}` : pair.question,
      },
      { role: 'assistant', content: pair.answer },
    ],
  }))
}

// Convert CSV-style data to fine-tuning format
function csvToFineTuning(rows: CSVRow[], systemPrompt: string): FineTuningExample[] {
  return rows.map(row => ({
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: row.input },
      { role: 'assistant', content: row.output },
    ],
  }))
}

// Convert chat logs to fine-tuning format
interface ChatLog {
  messages: { sender: 'customer' | 'agent'; text: string; timestamp: string }[]
}

function chatLogToFineTuning(logs: ChatLog[], systemPrompt: string): FineTuningExample[] {
  return logs.map(log => {
    const messages: FineTuningMessage[] = [{ role: 'system', content: systemPrompt }]

    for (const msg of log.messages) {
      messages.push({
        role: msg.sender === 'customer' ? 'user' : 'assistant',
        content: msg.text,
      })
    }

    return { messages }
  })
}
```

> **Beginner Note:** JSONL is just regular JSON objects, one per line. Each line is a complete, valid JSON object. This format is efficient for streaming processing — you can process one example at a time without loading the entire file into memory.

---

## Section 3: Training Data Best Practices

### Data Quality Filtering

The quality of your fine-tuning output is directly proportional to the quality of your training data. Bad data in means bad model out.

```typescript
interface QualityFilter {
  name: string
  check: (example: FineTuningExample) => {
    passed: boolean
    reason?: string
  }
}

const qualityFilters: QualityFilter[] = [
  {
    name: 'minimum_length',
    check: ex => {
      const assistantMsg = ex.messages.find(m => m.role === 'assistant')
      if (!assistantMsg) return { passed: false, reason: 'No assistant message' }
      return {
        passed: assistantMsg.content.length >= 50,
        reason:
          assistantMsg.content.length < 50
            ? `Assistant response too short: ${assistantMsg.content.length} chars`
            : undefined,
      }
    },
  },
  {
    name: 'maximum_length',
    check: ex => {
      const totalTokens = ex.messages.reduce((sum, m) => sum + estimateTokenCount(m.content), 0)
      return {
        passed: totalTokens <= 4096,
        reason: totalTokens > 4096 ? `Total tokens (${totalTokens}) exceeds maximum` : undefined,
      }
    },
  },
  {
    name: 'has_system_prompt',
    check: ex => {
      const hasSystem = ex.messages.some(m => m.role === 'system')
      return {
        passed: hasSystem,
        reason: hasSystem ? undefined : 'Missing system prompt',
      }
    },
  },
  {
    name: 'valid_turn_order',
    check: ex => {
      const nonSystem = ex.messages.filter(m => m.role !== 'system')
      for (let i = 0; i < nonSystem.length - 1; i++) {
        if (nonSystem[i].role === nonSystem[i + 1].role) {
          return {
            passed: false,
            reason: `Consecutive ${nonSystem[i].role} messages at position ${i}`,
          }
        }
      }
      return { passed: true }
    },
  },
  {
    name: 'no_empty_messages',
    check: ex => {
      const emptyMsg = ex.messages.find(m => m.content.trim().length === 0)
      return {
        passed: !emptyMsg,
        reason: emptyMsg ? `Empty ${emptyMsg.role} message found` : undefined,
      }
    },
  },
  {
    name: 'assistant_ends_conversation',
    check: ex => {
      const lastMsg = ex.messages[ex.messages.length - 1]
      return {
        passed: lastMsg.role === 'assistant',
        reason: lastMsg.role !== 'assistant' ? 'Conversation must end with assistant message' : undefined,
      }
    },
  },
]

function estimateTokenCount(text: string): number {
  // Rough estimation: ~4 characters per token for English text
  return Math.ceil(text.length / 4)
}

interface FilterReport {
  total: number
  passed: number
  filtered: number
  filterBreakdown: Record<string, number>
  examples: FineTuningExample[]
}

function filterTrainingData(examples: FineTuningExample[], filters: QualityFilter[]): FilterReport {
  const filterBreakdown: Record<string, number> = {}
  const passed: FineTuningExample[] = []

  for (const ex of examples) {
    let passedAll = true

    for (const filter of filters) {
      const result = filter.check(ex)
      if (!result.passed) {
        filterBreakdown[filter.name] = (filterBreakdown[filter.name] ?? 0) + 1
        passedAll = false
        break // Fail fast
      }
    }

    if (passedAll) {
      passed.push(ex)
    }
  }

  return {
    total: examples.length,
    passed: passed.length,
    filtered: examples.length - passed.length,
    filterBreakdown,
    examples: passed,
  }
}
```

### Diversity and Balance

Training data should be diverse across categories, difficulty levels, and edge cases.

```typescript
interface DatasetStats {
  totalExamples: number
  categoryDistribution: Record<string, number>
  avgAssistantLength: number
  avgUserLength: number
  avgTurns: number
  lengthDistribution: {
    short: number // < 100 tokens
    medium: number // 100-500 tokens
    long: number // > 500 tokens
  }
}

function analyzeDataset(examples: FineTuningExample[]): DatasetStats {
  const categories: Record<string, number> = {}
  let totalAssistantLength = 0
  let totalUserLength = 0
  let totalTurns = 0
  const lengthDistribution = { short: 0, medium: 0, long: 0 }

  for (const ex of examples) {
    // Count categories based on system prompt or extract from metadata
    const systemMsg = ex.messages.find(m => m.role === 'system')
    const category = systemMsg?.content.substring(0, 50) ?? 'unknown'
    categories[category] = (categories[category] ?? 0) + 1

    // Analyze message lengths
    for (const msg of ex.messages) {
      const tokens = estimateTokenCount(msg.content)
      if (msg.role === 'assistant') {
        totalAssistantLength += tokens
        if (tokens < 100) lengthDistribution.short++
        else if (tokens < 500) lengthDistribution.medium++
        else lengthDistribution.long++
      } else if (msg.role === 'user') {
        totalUserLength += tokens
      }
    }

    totalTurns += ex.messages.filter(m => m.role !== 'system').length
  }

  const assistantMsgCount = examples.reduce(
    (sum, ex) => sum + ex.messages.filter(m => m.role === 'assistant').length,
    0
  )
  const userMsgCount = examples.reduce((sum, ex) => sum + ex.messages.filter(m => m.role === 'user').length, 0)

  return {
    totalExamples: examples.length,
    categoryDistribution: categories,
    avgAssistantLength: assistantMsgCount > 0 ? totalAssistantLength / assistantMsgCount : 0,
    avgUserLength: userMsgCount > 0 ? totalUserLength / userMsgCount : 0,
    avgTurns: examples.length > 0 ? totalTurns / examples.length : 0,
    lengthDistribution,
  }
}

// Balance dataset across categories
function balanceDataset(
  examples: FineTuningExample[],
  categoryExtractor: (ex: FineTuningExample) => string,
  targetPerCategory?: number
): FineTuningExample[] {
  // Group by category
  const groups = new Map<string, FineTuningExample[]>()
  for (const ex of examples) {
    const cat = categoryExtractor(ex)
    const group = groups.get(cat) ?? []
    group.push(ex)
    groups.set(cat, group)
  }

  // Determine target count per category
  const minCount = Math.min(...Array.from(groups.values()).map(g => g.length))
  const target = targetPerCategory ?? minCount

  // Sample from each category
  const balanced: FineTuningExample[] = []
  for (const [, group] of groups) {
    const shuffled = [...group].sort(() => Math.random() - 0.5)
    balanced.push(...shuffled.slice(0, target))
  }

  return balanced
}
```

### LLM-Assisted Data Enhancement

Use an LLM to improve or augment your training data.

```typescript
async function enhanceTrainingExample(example: FineTuningExample): Promise<FineTuningExample> {
  const assistantMsg = example.messages.find(m => m.role === 'assistant')
  const userMsg = example.messages.find(m => m.role === 'user')

  if (!assistantMsg || !userMsg) return example

  const { text } = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a training data quality improver. Given a user question and an existing assistant response, improve the response by:
1. Making it more helpful and comprehensive
2. Adding appropriate structure (bullet points, numbered steps)
3. Maintaining a professional, empathetic tone
4. Ensuring factual accuracy
5. Keeping a similar length (do not make it much longer)

Return ONLY the improved response, nothing else.`,
    prompt: `User question: ${userMsg.content}\n\nExisting response: ${assistantMsg.content}`,
  })

  return {
    messages: example.messages.map(m => (m.role === 'assistant' ? { ...m, content: text } : m)),
  }
}

// Generate synthetic training examples from a seed set
async function generateSyntheticExamples(
  seedExamples: FineTuningExample[],
  count: number,
  systemPrompt: string
): Promise<FineTuningExample[]> {
  const seedDescriptions = seedExamples
    .map(ex => {
      const user = ex.messages.find(m => m.role === 'user')
      return user?.content ?? ''
    })
    .join('\n- ')

  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        examples: z.array(
          z.object({
            userMessage: z.string(),
            assistantMessage: z.string(),
          })
        ),
      }),
    }),
    system: `Generate realistic training examples for a customer support chatbot.
The examples should be similar in style and domain to the seed examples but cover NEW scenarios.
Each assistant response should be helpful, structured, and empathetic.
Do NOT repeat any of the seed examples.`,
    prompt: `System prompt for the chatbot: ${systemPrompt}

Seed examples (generate similar but different ones):
- ${seedDescriptions}

Generate ${count} new examples.`,
  })

  return output!.examples.map(ex => ({
    messages: [
      { role: 'system' as const, content: systemPrompt },
      { role: 'user' as const, content: ex.userMessage },
      { role: 'assistant' as const, content: ex.assistantMessage },
    ],
  }))
}
```

> **Advanced Note:** Synthetic data generation is powerful but comes with risks. Models can introduce subtle biases, generate plausible-but-wrong examples, and produce overly uniform data. Always have a human review a sample of synthetic examples before including them in your training set.

---

## Section 4: Fine-tuning APIs

### OpenAI Fine-tuning with Vercel AI SDK

OpenAI provides the most accessible fine-tuning API. After training, you use the fine-tuned model through the Vercel AI SDK just like any other model.

```typescript
// Note: Fine-tuning API calls use the OpenAI client directly
// The Vercel AI SDK is used for inference with the fine-tuned model

import { openai as openaiProvider } from '@ai-sdk/openai'

// Step 1: Upload training file (using OpenAI SDK)
// This step uses the OpenAI SDK directly as the Vercel AI SDK
// does not wrap fine-tuning management APIs

interface FineTuningJob {
  id: string
  model: string
  status: 'queued' | 'running' | 'succeeded' | 'failed'
  fineTunedModel: string | null
  trainingFile: string
  hyperparameters: {
    nEpochs: number | 'auto'
    batchSize: number | 'auto'
    learningRateMultiplier: number | 'auto'
  }
}

// Prepare and validate training data before upload
async function prepareForFineTuning(
  examples: FineTuningExample[],
  outputPath: string
): Promise<{
  valid: boolean
  stats: DatasetStats
  warnings: string[]
}> {
  const warnings: string[] = []

  // Filter for quality
  const filtered = filterTrainingData(examples, qualityFilters)
  if (filtered.filtered > 0) {
    warnings.push(`Filtered ${filtered.filtered} examples: ${JSON.stringify(filtered.filterBreakdown)}`)
  }

  // Analyze the dataset
  const stats = analyzeDataset(filtered.examples)

  // Validation checks
  if (filtered.examples.length < 10) {
    warnings.push('CRITICAL: Less than 10 examples. OpenAI requires at least 10, recommends 50-100.')
  }

  if (stats.avgAssistantLength < 20) {
    warnings.push('Assistant responses are very short. Consider adding more detail.')
  }

  // Write JSONL file
  writeTrainingData(filtered.examples, outputPath)

  return {
    valid: filtered.examples.length >= 10,
    stats,
    warnings,
  }
}

// Step 2: Start fine-tuning (conceptual — actual API call)
async function startFineTuning(params: {
  trainingFilePath: string
  baseModel: string
  epochs?: number
  batchSize?: number
  learningRateMultiplier?: number
  suffix?: string
}): Promise<FineTuningJob> {
  // In production, you would use the OpenAI SDK:
  // const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  //
  // const file = await openai.files.create({
  //   file: fs.createReadStream(params.trainingFilePath),
  //   purpose: "fine-tune",
  // });
  //
  // const job = await openai.fineTuning.jobs.create({
  //   training_file: file.id,
  //   model: params.baseModel,
  //   hyperparameters: {
  //     n_epochs: params.epochs ?? "auto",
  //     batch_size: params.batchSize ?? "auto",
  //     learning_rate_multiplier: params.learningRateMultiplier ?? "auto",
  //   },
  //   suffix: params.suffix,
  // });

  // Placeholder return for illustration
  return {
    id: 'ftjob-abc123',
    model: params.baseModel,
    status: 'queued',
    fineTunedModel: null,
    trainingFile: 'file-xyz789',
    hyperparameters: {
      nEpochs: params.epochs ?? 'auto',
      batchSize: params.batchSize ?? 'auto',
      learningRateMultiplier: params.learningRateMultiplier ?? 'auto',
    },
  }
}

// Step 3: Use the fine-tuned model via Vercel AI SDK
async function useFineTunedModel(fineTunedModelId: string, prompt: string): Promise<string> {
  const { text } = await generateText({
    // Use the fine-tuned model ID directly
    model: openaiProvider(fineTunedModelId),
    prompt,
    // Note: fine-tuned models often need shorter (or no) system prompts
    system: 'You are a helpful customer support agent.',
  })

  return text
}

// Example: using a fine-tuned model
const response = await useFineTunedModel(
  'ft:gpt-5-mini-2026-01-15:my-org:support-v1:abc123',
  'My order has not arrived yet. What should I do?'
)
console.log(response)
```

### Anthropic's Approach to Customization

Anthropic offers different approaches to model customization. As of this writing, Anthropic provides fine-tuning through their API for enterprise customers.

```typescript
// Using Anthropic models with the Vercel AI SDK
// Anthropic fine-tuned models are used just like base models

async function compareBaseVsCustomized(): Promise<void> {
  const prompt = 'Draft a response to a customer who received a damaged product.'

  // Base model with detailed prompt
  const baseResult = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a customer support agent for TechCorp.

TONE: Empathetic, professional, solution-oriented
FORMAT:
1. Acknowledge the issue
2. Apologize sincerely
3. Offer specific resolution steps
4. Provide timeline
5. Offer additional help

RULES:
- Never blame the customer
- Always offer a replacement OR refund
- Include order lookup instructions
- Keep response under 200 words`,
    prompt,
  })

  console.log('Base model (with long prompt):')
  console.log(baseResult.text)
  console.log(`Input tokens: ${baseResult.usage?.inputTokens}`)

  // After fine-tuning, the model has internalized the style
  // and you need a much shorter prompt
  // (Using same model for illustration — in practice this would be
  // a fine-tuned model ID)
  const customResult = await generateText({
    model: mistral('mistral-small-latest'),
    system: 'You are a TechCorp support agent.',
    prompt,
  })

  console.log('\nCustomized model (short prompt):')
  console.log(customResult.text)
  console.log(`Input tokens: ${customResult.usage?.inputTokens}`)
}
```

> **Beginner Note:** The Vercel AI SDK abstracts away provider differences for inference. Whether you are using a base model or a fine-tuned model, the code is nearly identical — you just change the model ID. This makes it easy to swap between base and fine-tuned models for comparison.

---

## Section 5: Hyperparameters

### Understanding Key Hyperparameters

Hyperparameters control how the model learns from your training data. Getting them right is the difference between a model that performs well and one that overfits or underfits.

```typescript
interface HyperparameterConfig {
  epochs: number
  batchSize: number
  learningRateMultiplier: number
}

interface HyperparameterGuide {
  parameter: string
  description: string
  defaultValue: string
  recommendedRange: string
  datasetSizeGuidance: Record<string, string>
}

const hyperparameterGuides: HyperparameterGuide[] = [
  {
    parameter: 'epochs',
    description:
      'Number of complete passes through the training dataset. More epochs mean the model sees each example more times.',
    defaultValue: 'auto (typically 3-4)',
    recommendedRange: '1-10',
    datasetSizeGuidance: {
      'small (<100)': '4-8 epochs — small datasets need more passes',
      'medium (100-1000)': '2-4 epochs — balanced approach',
      'large (>1000)': '1-2 epochs — large datasets risk overfitting with more',
    },
  },
  {
    parameter: 'batchSize',
    description:
      'Number of examples processed together in one training step. Larger batches are more stable but may generalize less well.',
    defaultValue: 'auto (typically 1-8)',
    recommendedRange: '1-32',
    datasetSizeGuidance: {
      'small (<100)': '1-2 — small batches for small datasets',
      'medium (100-1000)': '4-8 — moderate batch sizes',
      'large (>1000)': '8-32 — larger batches for efficiency',
    },
  },
  {
    parameter: 'learningRateMultiplier',
    description: 'Scales the base learning rate. Higher values learn faster but risk overshooting optimal weights.',
    defaultValue: 'auto (typically 1.0-2.0)',
    recommendedRange: '0.1-5.0',
    datasetSizeGuidance: {
      'small (<100)': '0.5-1.0 — lower rates to avoid overfitting',
      'medium (100-1000)': '1.0-2.0 — standard range',
      'large (>1000)': '1.0-3.0 — can be more aggressive with more data',
    },
  },
]

// Suggest hyperparameters based on dataset characteristics
function suggestHyperparameters(
  datasetSize: number,
  avgExampleTokens: number,
  taskComplexity: 'simple' | 'moderate' | 'complex'
): HyperparameterConfig {
  // Epochs: inversely related to dataset size
  let epochs: number
  if (datasetSize < 100) epochs = 6
  else if (datasetSize < 500) epochs = 4
  else if (datasetSize < 2000) epochs = 3
  else epochs = 2

  // Adjust for complexity
  if (taskComplexity === 'complex') epochs = Math.min(epochs + 2, 10)
  if (taskComplexity === 'simple') epochs = Math.max(epochs - 1, 1)

  // Batch size: related to dataset size
  let batchSize: number
  if (datasetSize < 50) batchSize = 1
  else if (datasetSize < 200) batchSize = 4
  else if (datasetSize < 1000) batchSize = 8
  else batchSize = 16

  // Learning rate: inversely related to example complexity
  let learningRateMultiplier: number
  if (avgExampleTokens > 1000) learningRateMultiplier = 0.5
  else if (avgExampleTokens > 500) learningRateMultiplier = 1.0
  else learningRateMultiplier = 1.8

  return { epochs, batchSize, learningRateMultiplier }
}

// Example usage
const suggested = suggestHyperparameters(300, 400, 'moderate')
console.log('Suggested hyperparameters:', suggested)
// { epochs: 4, batchSize: 8, learningRateMultiplier: 1.0 }
```

### Hyperparameter Search

Run multiple training jobs with different configurations and compare results.

```typescript
interface HyperparameterSearchConfig {
  baseModel: string
  trainingFile: string
  validationFile: string
  configurations: HyperparameterConfig[]
}

interface SearchResult {
  config: HyperparameterConfig
  trainingLoss: number
  validationLoss: number
  evalScore: number
  overfitting: boolean
}

async function hyperparameterSearch(
  searchConfig: HyperparameterSearchConfig,
  evalTestCases: { input: string; expected: string }[]
): Promise<SearchResult[]> {
  const results: SearchResult[] = []

  for (const config of searchConfig.configurations) {
    console.log(`Training with epochs=${config.epochs}, batch=${config.batchSize}, lr=${config.learningRateMultiplier}`)

    // Start fine-tuning job with these hyperparameters
    const job = await startFineTuning({
      trainingFilePath: searchConfig.trainingFile,
      baseModel: searchConfig.baseModel,
      epochs: config.epochs,
      batchSize: config.batchSize,
      learningRateMultiplier: config.learningRateMultiplier,
    })

    // In production, you would poll for job completion
    // and then evaluate the fine-tuned model
    console.log(`Job ${job.id} started. Poll for completion...`)

    // After job completes, evaluate on test cases
    // (Placeholder — actual implementation would wait for training)
    const evalScore = await evaluateModel(job.fineTunedModel ?? searchConfig.baseModel, evalTestCases)

    results.push({
      config,
      trainingLoss: 0, // Would come from training metrics
      validationLoss: 0, // Would come from training metrics
      evalScore,
      overfitting: false, // Would be determined from loss curves
    })
  }

  // Sort by eval score, descending
  results.sort((a, b) => b.evalScore - a.evalScore)

  return results
}

async function evaluateModel(modelId: string, testCases: { input: string; expected: string }[]): Promise<number> {
  let totalScore = 0

  for (const tc of testCases) {
    const { text } = await generateText({
      model: openaiProvider(modelId as any),
      prompt: tc.input,
    })

    // Simple similarity-based scoring
    const score = text.toLowerCase().includes(tc.expected.toLowerCase()) ? 1.0 : 0.5
    totalScore += score
  }

  return totalScore / testCases.length
}

// Define search space
const searchSpace: HyperparameterConfig[] = [
  { epochs: 2, batchSize: 4, learningRateMultiplier: 1.0 },
  { epochs: 4, batchSize: 4, learningRateMultiplier: 1.0 },
  { epochs: 4, batchSize: 8, learningRateMultiplier: 1.0 },
  { epochs: 4, batchSize: 4, learningRateMultiplier: 2.0 },
  { epochs: 6, batchSize: 4, learningRateMultiplier: 0.5 },
]
```

> **Advanced Note:** Overfitting in fine-tuning manifests as the model memorizing training examples rather than learning generalizable behavior. Signs include: perfect performance on training examples but poor performance on new inputs, generating responses that are verbatim copies of training data, and training loss continuing to decrease while validation loss increases.

---

## Section 6: Evaluation

### Base vs Fine-tuned Comparison

The most important evaluation compares your fine-tuned model against the base model to confirm the fine-tuning actually improved performance.

```typescript
interface ModelComparisonResult {
  baseModelScores: number[]
  fineTunedModelScores: number[]
  baseMean: number
  fineTunedMean: number
  improvement: number
  improvementPercent: number
  byCategory: Record<string, { baseMean: number; fineTunedMean: number; improvement: number }>
  verdict: 'improved' | 'no_change' | 'degraded'
}

async function compareBaseVsFineTuned(
  baseModelId: string,
  fineTunedModelId: string,
  testCases: {
    input: string
    systemPrompt?: string
    category: string
    expectedOutput?: string
  }[],
  judge: (input: string, output: string) => Promise<number>
): Promise<ModelComparisonResult> {
  const baseScores: number[] = []
  const ftScores: number[] = []
  const categoryScores: Record<string, { base: number[]; ft: number[] }> = {}

  for (const tc of testCases) {
    // Generate outputs from both models
    const [baseOutput, ftOutput] = await Promise.all([
      generateText({
        model: openaiProvider(baseModelId as any),
        system: tc.systemPrompt,
        prompt: tc.input,
      }),
      generateText({
        model: openaiProvider(fineTunedModelId as any),
        // Fine-tuned model may need shorter prompt
        prompt: tc.input,
      }),
    ])

    // Score both outputs
    const [baseScore, ftScore] = await Promise.all([judge(tc.input, baseOutput.text), judge(tc.input, ftOutput.text)])

    baseScores.push(baseScore)
    ftScores.push(ftScore)

    // Track per-category
    if (!categoryScores[tc.category]) {
      categoryScores[tc.category] = { base: [], ft: [] }
    }
    categoryScores[tc.category].base.push(baseScore)
    categoryScores[tc.category].ft.push(ftScore)
  }

  const baseMean = baseScores.reduce((a, b) => a + b, 0) / baseScores.length
  const ftMean = ftScores.reduce((a, b) => a + b, 0) / ftScores.length
  const improvement = ftMean - baseMean

  // Per-category analysis
  const byCategory: ModelComparisonResult['byCategory'] = {}
  for (const [cat, scores] of Object.entries(categoryScores)) {
    const catBaseMean = scores.base.reduce((a, b) => a + b, 0) / scores.base.length
    const catFtMean = scores.ft.reduce((a, b) => a + b, 0) / scores.ft.length
    byCategory[cat] = {
      baseMean: catBaseMean,
      fineTunedMean: catFtMean,
      improvement: catFtMean - catBaseMean,
    }
  }

  let verdict: 'improved' | 'no_change' | 'degraded'
  if (improvement > 0.05) verdict = 'improved'
  else if (improvement < -0.05) verdict = 'degraded'
  else verdict = 'no_change'

  return {
    baseModelScores: baseScores,
    fineTunedModelScores: ftScores,
    baseMean,
    fineTunedMean: ftMean,
    improvement,
    improvementPercent: baseMean > 0 ? (improvement / baseMean) * 100 : 0,
    byCategory,
    verdict,
  }
}
```

### Held-Out Test Set Management

Never evaluate on data the model was trained on. Always maintain a separate test set.

```typescript
interface DataSplit {
  training: FineTuningExample[]
  validation: FineTuningExample[]
  test: FineTuningExample[]
}

function splitDataset(
  examples: FineTuningExample[],
  trainRatio: number = 0.8,
  validationRatio: number = 0.1,
  testRatio: number = 0.1,
  seed: number = 42
): DataSplit {
  // Validate ratios
  const totalRatio = trainRatio + validationRatio + testRatio
  if (Math.abs(totalRatio - 1.0) > 0.01) {
    throw new Error(`Ratios must sum to 1.0, got ${totalRatio}`)
  }

  // Deterministic shuffle using seed
  const shuffled = [...examples]
  let currentSeed = seed
  for (let i = shuffled.length - 1; i > 0; i--) {
    // Simple seeded random
    currentSeed = (currentSeed * 1664525 + 1013904223) & 0x7fffffff
    const j = currentSeed % (i + 1)
    ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
  }

  const trainEnd = Math.floor(examples.length * trainRatio)
  const validationEnd = trainEnd + Math.floor(examples.length * validationRatio)

  return {
    training: shuffled.slice(0, trainEnd),
    validation: shuffled.slice(trainEnd, validationEnd),
    test: shuffled.slice(validationEnd),
  }
}

// Usage
const allData = trainingExamples // Your full dataset
const split = splitDataset(allData, 0.8, 0.1, 0.1)

console.log(`Training: ${split.training.length} examples`)
console.log(`Validation: ${split.validation.length} examples`)
console.log(`Test: ${split.test.length} examples`)

// Write each split to a file
writeTrainingData(split.training, 'data/train.jsonl')
writeTrainingData(split.validation, 'data/validation.jsonl')
writeTrainingData(split.test, 'data/test.jsonl')
```

> **Beginner Note:** The training set is what the model learns from. The validation set is used during training to detect overfitting. The test set is used only after training is complete to get an unbiased estimate of real-world performance. Never peek at test set results during the training iteration cycle.

---

## Section 7: Iterating

### Improving Datasets Based on Evaluation

After your first fine-tuning run, evaluation results guide your next iteration.

```typescript
interface IterationPlan {
  currentScore: number
  targetScore: number
  weakCategories: string[]
  actions: IterationAction[]
  estimatedImpact: string
}

interface IterationAction {
  type: 'add_examples' | 'improve_examples' | 'remove_bad_examples' | 'adjust_hyperparameters' | 'augment_data'
  description: string
  priority: 'high' | 'medium' | 'low'
  effort: 'low' | 'medium' | 'high'
}

function planIteration(evalResults: ModelComparisonResult, targetScore: number): IterationPlan {
  const actions: IterationAction[] = []

  // Identify weak categories
  const weakCategories = Object.entries(evalResults.byCategory)
    .filter(([, scores]) => scores.fineTunedMean < targetScore)
    .map(([cat]) => cat)

  // If overall fine-tuned is worse than base, data quality is the issue
  if (evalResults.verdict === 'degraded') {
    actions.push({
      type: 'remove_bad_examples',
      description: 'Review and remove low-quality training examples. Fine-tuning degraded performance.',
      priority: 'high',
      effort: 'medium',
    })
    actions.push({
      type: 'adjust_hyperparameters',
      description: 'Reduce epochs and learning rate to prevent overfitting on bad data.',
      priority: 'high',
      effort: 'low',
    })
  }

  // If specific categories are weak, add more examples for those
  if (weakCategories.length > 0) {
    actions.push({
      type: 'add_examples',
      description: `Add more training examples for weak categories: ${weakCategories.join(', ')}`,
      priority: 'high',
      effort: 'medium',
    })
  }

  // If scores are close to target, try data augmentation
  if (evalResults.fineTunedMean > targetScore * 0.9 && evalResults.fineTunedMean < targetScore) {
    actions.push({
      type: 'augment_data',
      description: 'Use paraphrasing and synthetic generation to increase dataset diversity.',
      priority: 'medium',
      effort: 'medium',
    })
  }

  return {
    currentScore: evalResults.fineTunedMean,
    targetScore,
    weakCategories,
    actions,
    estimatedImpact:
      actions.length > 0
        ? `${actions.length} actions identified to improve score from ${evalResults.fineTunedMean.toFixed(3)} to ${targetScore}`
        : 'No clear improvement actions identified. Consider a larger base model.',
  }
}
```

### Data Augmentation Techniques

```typescript
// Paraphrase existing examples to increase diversity
async function paraphraseExamples(examples: FineTuningExample[], variations: number = 2): Promise<FineTuningExample[]> {
  const augmented: FineTuningExample[] = [...examples]

  for (const ex of examples) {
    const userMsg = ex.messages.find(m => m.role === 'user')
    if (!userMsg) continue

    const { output } = await generateText({
      model: mistral('mistral-small-latest'),
      output: Output.object({
        schema: z.object({
          paraphrases: z.array(z.string()),
        }),
      }),
      system:
        'Generate paraphrased versions of the given text. Keep the meaning identical but vary the wording, structure, and style. Each paraphrase should be clearly different from the others.',
      prompt: `Generate ${variations} paraphrases of: "${userMsg.content}"`,
    })

    for (const paraphrase of output!.paraphrases) {
      augmented.push({
        messages: ex.messages.map(m => (m.role === 'user' ? { ...m, content: paraphrase } : m)),
      })
    }
  }

  return augmented
}

// Add difficulty variations to existing examples
async function addDifficultyVariations(examples: FineTuningExample[]): Promise<FineTuningExample[]> {
  const augmented: FineTuningExample[] = [...examples]

  for (const ex of examples) {
    const userMsg = ex.messages.find(m => m.role === 'user')
    const assistantMsg = ex.messages.find(m => m.role === 'assistant')
    if (!userMsg || !assistantMsg) continue

    // Generate a harder version of the same scenario
    const { output } = await generateText({
      model: mistral('mistral-small-latest'),
      output: Output.object({
        schema: z.object({
          harderInput: z.string(),
          appropriateResponse: z.string(),
        }),
      }),
      system: `Given a customer support interaction, create a HARDER version of the same scenario.
Make the customer's question:
- More ambiguous or complex
- Include emotional language
- Combine multiple issues
- Include misspellings or informal language

Then write the appropriate response maintaining the same quality and style.`,
      prompt: `Original question: ${userMsg.content}\nOriginal response: ${assistantMsg.content}`,
    })

    augmented.push({
      messages: [
        ...ex.messages.filter(m => m.role === 'system'),
        { role: 'user', content: output!.harderInput },
        { role: 'assistant', content: output!.appropriateResponse },
      ],
    })
  }

  return augmented
}
```

> **Advanced Note:** Each iteration cycle should follow this loop: train, evaluate, analyze failures, plan improvements, update data, retrain. Track your iterations in a spreadsheet or experiment tracker. Most successful fine-tuning projects go through 3-5 iterations before reaching their target quality.

---

## Section 8: Cost Analysis

### Comprehensive Cost Model

Fine-tuning costs extend beyond the training API call. A complete analysis includes data preparation, training, inference, and maintenance.

```typescript
interface FineTuningCostModel {
  // Data preparation costs
  dataPrep: {
    humanAnnotationHours: number
    hourlyRate: number
    llmAugmentationCost: number
    totalDataPrepCost: number
  }

  // Training costs
  training: {
    trainingTokens: number
    costPerMillionTrainingTokens: number
    iterations: number
    totalTrainingCost: number
  }

  // Inference costs (monthly)
  inference: {
    requestsPerMonth: number
    baseModelPromptTokens: number
    fineTunedPromptTokens: number
    outputTokensPerRequest: number
    baseModelCostPerMonth: number
    fineTunedCostPerMonth: number
    monthlySavings: number
  }

  // Maintenance costs (annual)
  maintenance: {
    retrainingFrequency: 'monthly' | 'quarterly' | 'annual'
    retrainingCostPerCycle: number
    annualMaintenanceCost: number
  }

  // Summary
  summary: {
    upfrontCost: number
    monthlyRunningCost: number
    monthlySavings: number
    breakEvenMonths: number
    firstYearROI: number
  }
}

function buildCostModel(params: {
  // Data prep
  annotationHours: number
  hourlyRate: number
  augmentationApiCalls: number
  augmentationCostPerCall: number

  // Training
  trainingExamples: number
  avgTokensPerExample: number
  epochs: number
  trainingCostPerMillionTokens: number
  expectedIterations: number

  // Inference
  requestsPerMonth: number
  basePromptTokens: number
  fineTunedPromptTokens: number
  outputTokens: number
  baseCostPerMillionInputTokens: number
  baseCostPerMillionOutputTokens: number
  fineTunedCostPerMillionInputTokens: number
  fineTunedCostPerMillionOutputTokens: number

  // Maintenance
  retrainingFrequency: 'monthly' | 'quarterly' | 'annual'
}): FineTuningCostModel {
  // Data preparation
  const totalDataPrepCost =
    params.annotationHours * params.hourlyRate + params.augmentationApiCalls * params.augmentationCostPerCall

  // Training
  const trainingTokens = params.trainingExamples * params.avgTokensPerExample * params.epochs
  const singleTrainingCost = (trainingTokens / 1_000_000) * params.trainingCostPerMillionTokens
  const totalTrainingCost = singleTrainingCost * params.expectedIterations

  // Inference (monthly)
  const baseInputCost =
    ((params.basePromptTokens * params.requestsPerMonth) / 1_000_000) * params.baseCostPerMillionInputTokens
  const baseOutputCost =
    ((params.outputTokens * params.requestsPerMonth) / 1_000_000) * params.baseCostPerMillionOutputTokens
  const baseModelCostPerMonth = baseInputCost + baseOutputCost

  const ftInputCost =
    ((params.fineTunedPromptTokens * params.requestsPerMonth) / 1_000_000) * params.fineTunedCostPerMillionInputTokens
  const ftOutputCost =
    ((params.outputTokens * params.requestsPerMonth) / 1_000_000) * params.fineTunedCostPerMillionOutputTokens
  const fineTunedCostPerMonth = ftInputCost + ftOutputCost

  const monthlySavings = baseModelCostPerMonth - fineTunedCostPerMonth

  // Maintenance
  const retrainingMultiplier =
    params.retrainingFrequency === 'monthly' ? 12 : params.retrainingFrequency === 'quarterly' ? 4 : 1
  const annualMaintenanceCost = singleTrainingCost * retrainingMultiplier

  // Summary
  const upfrontCost = totalDataPrepCost + totalTrainingCost
  const monthlyRunningCost = fineTunedCostPerMonth + annualMaintenanceCost / 12
  const breakEvenMonths = monthlySavings > 0 ? Math.ceil(upfrontCost / monthlySavings) : Infinity
  const firstYearROI =
    upfrontCost > 0 ? ((monthlySavings * 12 - upfrontCost - annualMaintenanceCost) / upfrontCost) * 100 : 0

  return {
    dataPrep: {
      humanAnnotationHours: params.annotationHours,
      hourlyRate: params.hourlyRate,
      llmAugmentationCost: params.augmentationApiCalls * params.augmentationCostPerCall,
      totalDataPrepCost,
    },
    training: {
      trainingTokens,
      costPerMillionTrainingTokens: params.trainingCostPerMillionTokens,
      iterations: params.expectedIterations,
      totalTrainingCost,
    },
    inference: {
      requestsPerMonth: params.requestsPerMonth,
      baseModelPromptTokens: params.basePromptTokens,
      fineTunedPromptTokens: params.fineTunedPromptTokens,
      outputTokensPerRequest: params.outputTokens,
      baseModelCostPerMonth,
      fineTunedCostPerMonth,
      monthlySavings,
    },
    maintenance: {
      retrainingFrequency: params.retrainingFrequency,
      retrainingCostPerCycle: singleTrainingCost,
      annualMaintenanceCost,
    },
    summary: {
      upfrontCost,
      monthlyRunningCost,
      monthlySavings,
      breakEvenMonths,
      firstYearROI,
    },
  }
}

// Example cost analysis
const costModel = buildCostModel({
  annotationHours: 40,
  hourlyRate: 50,
  augmentationApiCalls: 500,
  augmentationCostPerCall: 0.02,

  trainingExamples: 500,
  avgTokensPerExample: 500,
  epochs: 4,
  trainingCostPerMillionTokens: 25.0,
  expectedIterations: 3,

  requestsPerMonth: 100_000,
  basePromptTokens: 2000,
  fineTunedPromptTokens: 200,
  outputTokens: 300,
  baseCostPerMillionInputTokens: 3.0,
  baseCostPerMillionOutputTokens: 15.0,
  fineTunedCostPerMillionInputTokens: 3.0,
  fineTunedCostPerMillionOutputTokens: 15.0,

  retrainingFrequency: 'quarterly',
})

console.log('\n=== Fine-tuning Cost Analysis ===')
console.log(`\nData Prep: $${costModel.dataPrep.totalDataPrepCost.toFixed(2)}`)
console.log(`Training: $${costModel.training.totalTrainingCost.toFixed(2)}`)
console.log(`Upfront Total: $${costModel.summary.upfrontCost.toFixed(2)}`)
console.log(`\nMonthly base cost: $${costModel.inference.baseModelCostPerMonth.toFixed(2)}`)
console.log(`Monthly fine-tuned cost: $${costModel.inference.fineTunedCostPerMonth.toFixed(2)}`)
console.log(`Monthly savings: $${costModel.summary.monthlySavings.toFixed(2)}`)
console.log(`Break-even: ${costModel.summary.breakEvenMonths} months`)
console.log(`First-year ROI: ${costModel.summary.firstYearROI.toFixed(1)}%`)
```

> **Beginner Note:** Fine-tuning is most cost-effective when you have high request volume and long system prompts. If you are making 100 requests per day with a 200-token prompt, the savings from fine-tuning will likely never offset the training cost. If you are making 100,000 requests per day with a 2,000-token prompt, fine-tuning can pay for itself within weeks.

> **Local Alternative (Ollama):** Fine-tuning concepts (when to fine-tune, dataset preparation, evaluation) apply universally. For hands-on practice, you can fine-tune smaller open models using tools like Unsloth or Axolotl, then serve them via Ollama with a custom Modelfile. This gives you the complete fine-tuning workflow — data prep, training, evaluation, deployment — entirely locally and for free.

---

## Summary

In this module, you learned:

1. **When to fine-tune:** Fine-tuning is appropriate when prompt engineering and RAG are insufficient — for consistent style adoption, complex format compliance, domain-specific terminology, or reducing long system prompts.
2. **Dataset preparation:** How to structure training data in JSONL format with proper conversation structure, including system prompts, user messages, and assistant responses.
3. **Data quality best practices:** Filtering low-quality examples, ensuring diversity and balance across categories, and using LLM-assisted data enhancement to expand limited datasets.
4. **Fine-tuning APIs:** How to use OpenAI's fine-tuning API through the Vercel AI SDK and understand Anthropic's approaches to model customization.
5. **Hyperparameters:** How epochs, learning rate, and batch size affect training outcomes, and how to run hyperparameter searches to find optimal configurations.
6. **Evaluation:** Comparing fine-tuned models against base models using held-out test sets, measuring improvements across multiple quality dimensions.
7. **Iterative improvement:** Improving datasets based on evaluation results, using data augmentation techniques to address specific weaknesses.
8. **Cost analysis:** Building a comprehensive cost model that weighs training investment against inference savings from shorter prompts and fewer tokens per request.

In Module 21, you will learn how to build safety guardrails that protect your LLM applications from prompt injection, jailbreaks, and data exfiltration.

---

## Quiz

**Question 1:** When is fine-tuning the WRONG approach?

A) When you need the model to adopt a specific writing style
B) When you need the model to know about events from last week
C) When you need the model to follow a complex output format consistently
D) When you need to reduce inference costs at scale

**Answer: B** — Fine-tuning trains knowledge into the model at training time. Information from last week cannot be in the training data quickly enough, and the model's knowledge will go stale. This is exactly the use case for RAG, which retrieves up-to-date information at inference time. Fine-tuning is for changing model behavior, not for keeping it current.

---

**Question 2:** What is the purpose of a validation set during fine-tuning?

A) To train the model on more diverse data
B) To detect overfitting during training
C) To test the model after training is complete
D) To generate synthetic training examples

**Answer: B** — The validation set is evaluated periodically during training to monitor whether the model is overfitting. If training loss continues to decrease but validation loss starts increasing, the model is memorizing training examples rather than learning generalizable patterns. This signal tells you to stop training or reduce the learning rate.

---

**Question 3:** What is a typical sign of overfitting in a fine-tuned model?

A) The model generates diverse outputs for the same input
B) The model generates verbatim copies of training examples
C) The model refuses to follow the system prompt
D) The model responds in a different language

**Answer: B** — When a model overfits, it memorizes training examples rather than learning the underlying patterns. A telltale sign is that the model reproduces training examples word-for-word when given similar inputs, rather than generating appropriate new responses. Other signs include perfect scores on training data but poor scores on new inputs.

---

**Question 4:** Why might a fine-tuned model need a shorter system prompt than the base model?

A) Fine-tuned models cannot process long prompts
B) The fine-tuned model has internalized the instructions from training data
C) API limits are lower for fine-tuned models
D) Shorter prompts are always better

**Answer: B** — One of the key benefits of fine-tuning is that the model learns to follow your desired behavior, style, and format from the training examples. Instructions that previously required a long system prompt are now part of the model's weights, so you can use a much shorter (or no) system prompt — reducing input token costs on every request.

---

**Question 5:** What is the recommended first step before deciding to fine-tune?

A) Collect 10,000 training examples
B) Try prompt engineering and evaluate whether it is sufficient
C) Set up a training infrastructure pipeline
D) Switch to the largest available model

**Answer: B** — Fine-tuning should be considered only after prompt engineering has been tried and found insufficient. Prompt engineering is faster, cheaper, and more flexible. Many tasks that seem to require fine-tuning can actually be solved with better prompts, few-shot examples, or structured output schemas. Fine-tune only when you have evidence that prompting is not enough.

---

## Exercises

### Exercise 1: Prepare a Fine-tuning Dataset

Prepare a complete fine-tuning dataset for a domain-specific task.

**Specification:**

1. Choose a domain (e.g., medical Q&A, legal document summarization, code review).

2. Create at least 30 training examples in JSONL format:
   - 20 manually written examples covering diverse scenarios
   - 10 examples generated synthetically using the `generateSyntheticExamples` function
   - Each example must have a system prompt, user message, and assistant response

3. Apply quality filters and fix any issues:
   - Run `filterTrainingData` with all quality filters
   - Fix or remove any examples that fail

4. Split the dataset:
   - 80% training, 10% validation, 10% test
   - Verify that all splits have representation from each category

5. Generate a dataset analysis report including:
   - Total examples per split
   - Category distribution
   - Average message lengths
   - Length distribution histogram

### Exercise 2: Train and Evaluate a Fine-tuned Model

Perform a complete fine-tuning cycle and evaluate the results.

**Specification:**

1. Using the dataset from Exercise 1, train a fine-tuned model:
   - Use suggested hyperparameters from `suggestHyperparameters`
   - Log the training configuration

2. After training completes, evaluate the fine-tuned model:
   - Run the held-out test set through both the base model and fine-tuned model
   - Use an LLM-as-judge evaluator with at least 3 criteria
   - Generate a `ModelComparisonResult`

3. Perform a cost analysis:
   - Calculate the break-even point for your use case
   - Determine whether fine-tuning is cost-justified at 1K, 10K, and 100K requests per month

4. Create an iteration plan based on the evaluation results:
   - Identify weak areas
   - Plan specific data improvements
   - Document what you would change for the next iteration

**Note:** If you do not have access to the OpenAI fine-tuning API, you can mock the training step and focus on the dataset preparation, evaluation framework, and cost analysis.
