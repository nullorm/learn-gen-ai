# Module 3: Structured Output

## Learning Objectives

- Understand why free-text LLM responses are insufficient for production applications
- Define schemas using Zod for type-safe structured output
- Use `generateText` with `Output.object()` to constrain LLM responses to exact TypeScript types
- Design nested schemas, enum constraints, and optional fields
- Apply schema design patterns for real-world data extraction tasks
- Validate and error-handle structured output with Zod's parse and safeParse
- Stream structured output with `streamText` and `Output.object()` for real-time partial results

---

## Why Should I Care?

Every serious LLM application needs to extract structured data from model responses. A chatbot needs to detect user intent as a typed enum. A data pipeline needs JSON objects, not paragraphs. A form assistant needs specific field values that match database schemas. Free text is fine for human consumption, but your TypeScript code needs types.

Without structured output, you end up writing fragile regex parsers and string-matching logic to extract data from free text. This is brittle, error-prone, and fails silently when the model changes its phrasing. Structured output eliminates this entire class of problems by constraining the model to produce valid, typed data structures.

The Vercel AI SDK's `generateText` function with `Output.object()`, combined with Zod schemas, gives you compile-time type safety and runtime validation in a single call. The model's output is guaranteed to match your schema or the call fails with a clear error. No parsing, no guessing, no silent corruption.

---

## Connection to Other Modules

- **Module 1 (Setup & First Calls)** introduced `generateText`. This module extends it with `Output.object()` for structured output.
- **Module 2 (Prompt Engineering)** taught you how to craft prompts. Here, you combine prompt engineering with schema definitions for maximum precision.
- **Module 7 (Tool Use)** defines tools using Zod schemas — the same pattern you learn here.
- **Module 9 (RAG Fundamentals)** uses structured output for citation extraction and metadata.
- **Module 14 (Agent Fundamentals)** uses structured output for planning and decision-making.

---

## Section 1: The Problem with Free Text

### Why Free Text Fails

Consider building an app that analyzes product reviews. You need three pieces of information from each review: the sentiment (positive, negative, neutral), a confidence score (0-1), and key topics mentioned. With `generateText`, you might write:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const result = await generateText({
  model: mistral('mistral-small-latest'),
  system: 'Analyze the review. Return sentiment, confidence, and topics.',
  prompt: 'This laptop is blazing fast but the battery dies after 2 hours. The screen is gorgeous though.',
})

console.log(result.text)
// Might return:
// "Sentiment: Mixed/Positive\nConfidence: 0.7\nTopics: performance, battery life, display"
// Or: "The sentiment is mostly positive with some negative aspects..."
// Or: "**Sentiment:** Positive (with caveats)\n**Confidence:** ~70%\n**Topics:** speed, battery, screen"
```

The model might return any of those formats — or something entirely different. Each format requires different parsing logic. And the next model update might change the format silently.

### The Parsing Nightmare

Attempting to parse free text leads to fragile code:

```typescript
// DON'T DO THIS — fragile and error-prone
function parseSentiment(text: string): { sentiment: string; confidence: number; topics: string[] } {
  const sentimentMatch = text.match(/sentiment[:\s]+(\w+)/i)
  const confidenceMatch = text.match(/confidence[:\s]+([\d.]+)/i)
  const topicsMatch = text.match(/topics[:\s]+(.+)/i)

  return {
    sentiment: sentimentMatch?.[1] ?? 'unknown',
    confidence: parseFloat(confidenceMatch?.[1] ?? '0'),
    topics: topicsMatch?.[1]?.split(',').map(t => t.trim()) ?? [],
  }
}
```

This function breaks when the model uses different phrasing, different ordering, or includes the data in prose instead of structured lines. It also has no type safety — the return type claims to be valid, but the data might be garbage.

### The Structured Output Solution

`generateText` with `Output.object()` eliminates all of this. You define a schema, and the model is constrained to produce output that matches it exactly:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const result = await generateText({
  model: mistral('mistral-small-latest'),
  output: Output.object({
    schema: z.object({
      sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']),
      confidence: z.number().min(0).max(1),
      topics: z.array(z.string()),
    }),
  }),
  prompt: 'This laptop is blazing fast but the battery dies after 2 hours. The screen is gorgeous though.',
})

// result.output is fully typed:
// { sentiment: 'mixed', confidence: 0.75, topics: ['performance', 'battery life', 'display'] }
console.log(result.output.sentiment) // TypeScript knows this is 'positive' | 'negative' | 'neutral' | 'mixed'
console.log(result.output.confidence) // TypeScript knows this is a number
```

No parsing. No regex. No ambiguity. The schema IS the contract.

---

## Section 2: Zod Schema Basics

### What is Zod?

Zod is a TypeScript-first schema validation library. It lets you define a schema once and get both TypeScript types AND runtime validation from the same definition. This is critical for LLM output — you need compile-time types for your code and runtime validation for the model's unpredictable output.

```typescript
import { z } from 'zod'

// Define a schema
const UserSchema = z.object({
  name: z.string(),
  age: z.int().positive(),
  email: z.email(),
})

// Infer the TypeScript type from the schema
type User = z.infer<typeof UserSchema>
// => { name: string; age: number; email: string }

// Validate data at runtime
const valid = UserSchema.parse({ name: 'Alice', age: 30, email: 'alice@example.com' })
// => { name: 'Alice', age: 30, email: 'alice@example.com' }

const invalid = UserSchema.safeParse({ name: 'Bob', age: -5, email: 'not-an-email' })
// => { success: false, error: ZodError }
```

### Core Zod Types

Here are the types you will use most with `Output.object()`:

```typescript
import { z } from 'zod'

// Primitives
z.string() // any string
z.number() // any number
z.boolean() // true or false
z.null() // null

// String constraints
z.string().min(1) // non-empty string
z.string().max(100) // max 100 characters
z.email() // valid email format
z.url() // valid URL format

// Number constraints
z.int() // integer only
z.number().positive() // > 0
z.number().min(0).max(1) // between 0 and 1 inclusive
z.number().min(1).max(10) // 1-10 rating scale

// Enums — constrain to specific values
z.enum(['small', 'medium', 'large'])
z.enum(['positive', 'negative', 'neutral'])

// Arrays
z.array(z.string()) // string array
z.array(z.number()).min(1) // non-empty number array
z.array(z.string()).max(5) // at most 5 items

// Objects
z.object({
  name: z.string(),
  score: z.number(),
})

// Optional fields
z.object({
  name: z.string(),
  nickname: z.string().optional(), // may be undefined
})

// Descriptions (help the LLM understand intent)
z.string().describe("The user's full name as it appears on their ID")
z.number().describe('Confidence score between 0 and 1')
```

> **Beginner Note:** The `.describe()` method is especially important for LLM output. It tells the model what each field should contain, acting as inline documentation that guides generation.

### Schema Descriptions Guide the Model

Adding `.describe()` to your schema fields dramatically improves output quality. The descriptions are sent to the model as part of the schema definition.

```typescript
const ReviewAnalysis = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']).describe('Overall sentiment of the review'),
  confidence: z.number().min(0).max(1).describe('How confident the analysis is, from 0 (uncertain) to 1 (certain)'),
  topics: z.array(z.string()).describe('Key topics or aspects mentioned in the review'),
  summary: z.string().max(200).describe('A one-sentence summary of the review'),
})
```

---

## Section 3: generateText with Output.object() — Constraining the Model

### The Output.object() Pattern

`generateText` with `Output.object()` adds a schema constraint to the model's output:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const result = await generateText({
  model: mistral('mistral-small-latest'),
  output: Output.object({ schema: z.object({ ... }) }),
  prompt: 'Your prompt here',
})

// result.output is the typed, validated output
// result.usage has token counts
// result.finishReason has the stop reason
```

### Your First Structured Output Call

```typescript
// src/examples/structured-first.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const CitySchema = z.object({
  name: z.string().describe('Name of the city'),
  country: z.string().describe('Country the city is in'),
  population: z.int().positive().describe('Approximate population'),
  knownFor: z.array(z.string()).describe('What the city is famous for (3-5 items)'),
  isCapital: z.boolean().describe('Whether the city is a national capital'),
})

type City = z.infer<typeof CitySchema>

async function main(): Promise<void> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: CitySchema }),
    prompt: 'Tell me about Tokyo.',
  })

  const city: City = result.output

  console.log(`City: ${city.name}`)
  console.log(`Country: ${city.country}`)
  console.log(`Population: ${city.population.toLocaleString()}`)
  console.log(`Known for: ${city.knownFor.join(', ')}`)
  console.log(`Capital: ${city.isCapital ? 'Yes' : 'No'}`)
  console.log(`\nTokens used: ${result.usage.totalTokens}`)
}

main().catch(console.error)
```

### System Prompts with Output.object()

You can combine system prompts with structured output. The system prompt guides _how_ the model fills in the schema:

```typescript
// src/examples/structured-with-system.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const SentimentSchema = z.object({
  sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().describe('Brief explanation of why this sentiment was chosen'),
  keyPhrases: z.array(z.string()).describe('Phrases from the text that indicate the sentiment'),
})

async function analyzeSentiment(text: string) {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: SentimentSchema }),
    system: `You are a sentiment analysis expert.
Analyze the given text carefully.
Consider both explicit statements and implied tone.
For mixed sentiment, lean toward the dominant emotion.`,
    prompt: text,
    temperature: 0,
  })

  return result.output
}

async function main(): Promise<void> {
  const reviews = [
    'Absolutely incredible experience! Best restaurant in the city.',
    'The food was good but the wait was over an hour. Would not return.',
    'Ordered the special. It arrived. I ate it.',
  ]

  for (const review of reviews) {
    const analysis = await analyzeSentiment(review)
    console.log(`\nReview: "${review}"`)
    console.log(`Sentiment: ${analysis.sentiment} (${(analysis.confidence * 100).toFixed(0)}% confidence)`)
    console.log(`Reasoning: ${analysis.reasoning}`)
    console.log(`Key phrases: ${analysis.keyPhrases.join(', ')}`)
  }
}

main().catch(console.error)
```

> **Beginner Note:** The `temperature: 0` setting is important for structured output. You want deterministic, consistent results — not creative variation. Always use temperature 0 for classification, extraction, and analysis tasks.

> **Provider Quirk:** Mistral requires `topP: 1` (or omitting `topP`) when using `temperature: 0` (greedy sampling). Other providers like Anthropic and OpenAI silently ignore `topP` during greedy decoding. If you pass both `temperature: 0` and `topP: 0.9` to Mistral, you'll get a 400 error. When writing provider-agnostic code, either omit `topP` when using temperature 0, or use a small non-zero temperature (e.g., `0.1`) when you need both parameters.

---

## Section 4: Nested Schemas

### Why Nested Schemas?

Real-world data is rarely flat. A product review might contain multiple aspects, each with its own sentiment. A job posting has company info, requirements, and benefits as separate sub-objects. Zod handles nesting naturally.

### Basic Nesting

```typescript
// src/examples/structured-nested.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const CompanySchema = z.object({
  name: z.string(),
  industry: z.string(),
  founded: z.int().describe('Year founded'),
})

const JobPostingSchema = z.object({
  title: z.string().describe('Job title'),
  company: CompanySchema,
  location: z.object({
    city: z.string(),
    state: z.string().optional(),
    country: z.string(),
    remote: z.boolean().describe('Whether the position allows remote work'),
  }),
  salary: z.object({
    min: z.number().positive().describe('Minimum salary in USD'),
    max: z.number().positive().describe('Maximum salary in USD'),
    currency: z.string().default('USD'),
  }),
  requirements: z.array(z.string()).describe('Required skills and qualifications'),
  benefits: z.array(z.string()).describe('Benefits offered'),
})

type JobPosting = z.infer<typeof JobPostingSchema>

async function extractJobPosting(text: string): Promise<JobPosting> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: JobPostingSchema }),
    system: 'Extract job posting information from the given text. Infer reasonable values if not explicitly stated.',
    prompt: text,
    temperature: 0,
  })

  return result.output
}

async function main(): Promise<void> {
  const posting = await extractJobPosting(`
    TechCorp (established 2015, fintech industry) is hiring a Senior TypeScript Developer
    in San Francisco, CA. Remote-friendly. Salary range: $150k-$200k.
    Must have 5+ years TypeScript, React, and Node.js experience.
    We offer health insurance, 401k matching, unlimited PTO, and equity.
  `)

  console.log(JSON.stringify(posting, null, 2))
}

main().catch(console.error)
```

### Arrays of Nested Objects

When you need to extract multiple instances of a complex structure:

```typescript
// src/examples/structured-array-nested.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const AspectSchema = z.object({
  aspect: z.string().describe('The product aspect being discussed (e.g., battery, screen, performance)'),
  sentiment: z.enum(['positive', 'negative', 'neutral']),
  quote: z.string().describe('The exact phrase from the review about this aspect'),
})

const DetailedReviewSchema = z.object({
  overallSentiment: z.enum(['positive', 'negative', 'neutral', 'mixed']),
  overallScore: z.number().min(1).max(5).describe('Rating on a 1-5 scale'),
  aspects: z.array(AspectSchema).min(1).describe('Individual aspects analyzed in the review'),
  wouldRecommend: z.boolean(),
  summary: z.string().max(100),
})

async function analyzeReview(review: string) {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: DetailedReviewSchema }),
    prompt: review,
    temperature: 0,
  })

  return result.output
}

async function main(): Promise<void> {
  const review = `
    This laptop is a mixed bag. The M3 chip is incredibly fast — compiling my TypeScript
    project takes half the time compared to my old machine. The display is stunning, easily
    the best I've used. However, the battery barely lasts 4 hours with my workload, which
    is disappointing for a $2000 machine. The keyboard feels great though. I'd recommend
    it for desk work but not for travel.
  `

  const analysis = await analyzeReview(review)

  console.log(`Overall: ${analysis.overallSentiment} (${analysis.overallScore}/5)`)
  console.log(`Would recommend: ${analysis.wouldRecommend}`)
  console.log(`Summary: ${analysis.summary}`)
  console.log('\nAspects:')
  for (const aspect of analysis.aspects) {
    console.log(`  ${aspect.aspect}: ${aspect.sentiment} — "${aspect.quote}"`)
  }
}

main().catch(console.error)
```

> **Advanced Note:** Deeply nested schemas (3+ levels) can sometimes confuse smaller models. If you encounter quality issues, flatten the schema or break the extraction into multiple `generateText` with `Output.object()` calls. Claude Sonnet handles 3-4 levels of nesting reliably; smaller open-source models may struggle beyond 2 levels.

---

## Section 5: Enum Constraints

### Why Enums Matter

Enums are the most powerful constraint in structured output. They reduce the model's output space to a finite set of values, eliminating the ambiguity of free-text classification.

Without enums:

```typescript
// The model might return "Positive", "positive", "POSITIVE", "pos", "good sentiment", etc.
sentiment: z.string()
```

With enums:

```typescript
// The model MUST return exactly one of these values
sentiment: z.enum(['positive', 'negative', 'neutral', 'mixed'])
```

### Practical Enum Patterns

```typescript
// src/examples/structured-enums.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Priority classification
const TicketSchema = z.object({
  title: z.string(),
  priority: z
    .enum(['critical', 'high', 'medium', 'low'])
    .describe('Urgency level based on impact and time sensitivity'),
  category: z.enum(['bug', 'feature', 'question', 'documentation', 'security']).describe('Type of support ticket'),
  estimatedEffort: z
    .enum(['trivial', 'small', 'medium', 'large', 'epic'])
    .describe('Estimated engineering effort to resolve'),
  assignTo: z
    .enum(['frontend', 'backend', 'devops', 'security', 'product'])
    .describe('Team best suited to handle this ticket'),
})

async function classifyTicket(description: string) {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: TicketSchema }),
    system: `You are a support ticket triage system.
Classify tickets based on their description.
For priority:
- critical: system down, data loss, security breach
- high: major feature broken, affecting many users
- medium: minor feature issue, workaround available
- low: cosmetic, nice-to-have, minor improvement`,
    prompt: description,
    temperature: 0,
  })

  return result.output
}

async function main(): Promise<void> {
  const tickets = [
    'Users cannot log in. The authentication service is returning 500 errors.',
    'The dark mode toggle does not update the sidebar color.',
    'Can we add an export-to-CSV button on the reports page?',
  ]

  for (const ticket of tickets) {
    const classified = await classifyTicket(ticket)
    console.log(`\nTicket: "${ticket.substring(0, 60)}..."`)
    console.log(`  Priority: ${classified.priority}`)
    console.log(`  Category: ${classified.category}`)
    console.log(`  Effort: ${classified.estimatedEffort}`)
    console.log(`  Assign: ${classified.assignTo}`)
  }
}

main().catch(console.error)
```

### Enum with Descriptions

When enum values need context, use `.describe()` on the enum itself or pair it with a reasoning field:

```typescript
const IntentSchema = z.object({
  intent: z
    .enum(['greeting', 'question', 'complaint', 'purchase', 'cancel', 'other'])
    .describe('The primary user intent'),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().describe('Brief explanation of why this intent was classified'),
})
```

> **Beginner Note:** Keep enum values as simple, lowercase strings. Avoid spaces, special characters, or long phrases. The model is more reliable with simple values like `'bug'` than `'Bug Report / Issue'`.

---

## Section 6: Optional Fields and Defaults

### When to Use Optional Fields

Not every piece of information is always present. Optional fields let the schema handle missing data gracefully instead of forcing the model to fabricate values.

```typescript
// src/examples/structured-optional.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const ContactSchema = z.object({
  name: z.string().describe('Full name'),
  email: z.email().optional().describe('Email address if mentioned'),
  phone: z.string().optional().describe('Phone number if mentioned'),
  company: z.string().optional().describe('Company or organization if mentioned'),
  role: z.string().optional().describe('Job title or role if mentioned'),
  preferredContact: z.enum(['email', 'phone', 'either', 'unknown']).describe('How they prefer to be contacted'),
})

type Contact = z.infer<typeof ContactSchema>

async function extractContact(text: string): Promise<Contact> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ContactSchema }),
    system: `Extract contact information from the text.
Only fill in fields that are explicitly mentioned.
Do not infer or fabricate missing information.
Use "unknown" for preferredContact if not stated.`,
    prompt: text,
    temperature: 0,
  })

  return result.output
}

async function main(): Promise<void> {
  const texts = [
    "Hi, I'm Sarah Chen from Acme Corp. Reach me at sarah@acme.com.",
    "Call me at 555-0123. Name's Bob. I prefer phone.",
    'My name is Alex.',
  ]

  for (const text of texts) {
    const contact = await extractContact(text)
    console.log(`\nInput: "${text}"`)
    console.log(`Name: ${contact.name}`)
    console.log(`Email: ${contact.email ?? '(not provided)'}`)
    console.log(`Phone: ${contact.phone ?? '(not provided)'}`)
    console.log(`Company: ${contact.company ?? '(not provided)'}`)
    console.log(`Preferred: ${contact.preferredContact}`)
  }
}

main().catch(console.error)
```

### Default Values

Use `.default()` for fields that should have a fallback value:

```typescript
const ConfigSchema = z.object({
  model: z.string().default('mistral-small-latest'),
  temperature: z.number().default(0.7),
  maxTokens: z.number().default(1000),
  language: z.string().default('en'),
  verbose: z.boolean().default(false),
})
```

### Nullable vs Optional

Understand the distinction:

```typescript
// Optional: field may be missing entirely (undefined)
z.string().optional() // string | undefined

// Nullable: field is present but may be null
z.string().nullable() // string | null

// Both: field may be missing or null
z.string().optional().nullable() // string | null | undefined
```

Use `optional()` when the information might not exist at all. Use `nullable()` when the field exists but might have no value (common in database schemas).

> **Advanced Note:** With `Output.object()`, the model tends to include all fields. If you want the model to genuinely omit optional fields when information is not available, explicitly instruct it in the system prompt: "Only include optional fields if the information is clearly present in the text."

---

## Section 7: Schema Design Patterns

### Pattern 1: Classification with Metadata

Pair a classification enum with supporting data:

```typescript
const ClassificationSchema = z.object({
  label: z.enum(['spam', 'ham']),
  confidence: z.number().min(0).max(1),
  signals: z.array(z.string()).describe('Specific indicators that led to this classification'),
})
```

### Pattern 2: Extraction with Source Attribution

Track where each piece of data came from:

```typescript
const EntitySchema = z.object({
  entities: z.array(
    z.object({
      value: z.string().describe('The extracted entity value'),
      type: z.enum(['person', 'organization', 'location', 'date', 'money']),
      source: z.string().describe('The exact text span this was extracted from'),
      startIndex: z.int().describe('Character position where this entity starts in the input'),
    })
  ),
})
```

### Pattern 3: Multi-Step Analysis

When the task requires reasoning before answering, include reasoning in the schema:

```typescript
const AnalysisSchema = z.object({
  observations: z.array(z.string()).describe('Key observations from the data'),
  analysis: z.string().describe('Synthesis of observations into an analysis'),
  conclusion: z.enum(['approve', 'reject', 'needs-review']).describe('Final decision based on the analysis'),
  confidence: z.number().min(0).max(1),
})
```

This is a structured form of chain-of-thought — the model generates its reasoning as part of the schema, and the reasoning informs the conclusion.

### Pattern 4: Comparison Output

When comparing multiple items:

```typescript
const ComparisonSchema = z.object({
  items: z
    .array(
      z.object({
        name: z.string(),
        pros: z.array(z.string()),
        cons: z.array(z.string()),
        score: z.number().min(1).max(10),
      })
    )
    .min(2),
  winner: z.string().describe('Name of the best item'),
  reasoning: z.string().describe('Why the winner was chosen'),
})
```

### Pattern 5: Structured Error Reporting

When the model should report issues it finds:

```typescript
const CodeAnalysisSchema = z.object({
  issues: z.array(
    z.object({
      severity: z.enum(['error', 'warning', 'info']),
      line: z.int().optional(),
      message: z.string(),
      suggestion: z.string().describe('How to fix this issue'),
    })
  ),
  overallQuality: z.enum(['good', 'acceptable', 'needs-work', 'critical']),
  summary: z.string().max(200),
})
```

> **Beginner Note:** When designing schemas, start simple and add fields incrementally. It is easier to add a field than to diagnose why a complex schema produces unexpected results. Test each addition with a few example inputs.

---

## Section 8: Validation and Error Handling

### parse vs safeParse

Zod provides two ways to validate data:

```typescript
import { z } from 'zod'

const Schema = z.object({
  name: z.string(),
  age: z.number().positive(),
})

// parse: throws on invalid data
try {
  const data = Schema.parse({ name: 'Alice', age: -5 })
} catch (error) {
  if (error instanceof z.ZodError) {
    console.error('Validation failed:', error.issues)
  }
}

// safeParse: returns a result object, never throws
const result = Schema.safeParse({ name: 'Alice', age: -5 })
if (result.success) {
  console.log('Valid:', result.data)
} else {
  console.error('Invalid:', result.error.issues)
}
```

### Handling Structured Output Failures

`generateText` with `Output.object()` validates the model's output against your schema. If validation fails, it throws. Here is how to handle it gracefully:

```typescript
// src/examples/structured-error-handling.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const StrictSchema = z.object({
  rating: z.int().min(1).max(5),
  category: z.enum(['electronics', 'clothing', 'food', 'other']),
  verified: z.boolean(),
})

async function safeExtract(text: string) {
  try {
    const result = await generateText({
      model: mistral('mistral-small-latest'),
      output: Output.object({ schema: StrictSchema }),
      prompt: text,
      temperature: 0,
    })

    return { success: true as const, data: result.output }
  } catch (error) {
    if (error instanceof Error) {
      return { success: false as const, error: error.message }
    }
    return { success: false as const, error: 'Unknown error' }
  }
}

async function main(): Promise<void> {
  // Good input — should succeed
  const good = await safeExtract('Great wireless headphones, 5 stars! Verified purchase.')
  console.log('Good input:', good)

  // Ambiguous input — model may struggle
  const ambiguous = await safeExtract('Hmm.')
  console.log('Ambiguous input:', ambiguous)
}

main().catch(console.error)
```

### Post-Validation: Additional Business Logic

Even when `generateText` with `Output.object()` succeeds, you may want additional validation:

```typescript
// src/examples/structured-post-validation.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const OrderSchema = z.object({
  items: z.array(
    z.object({
      name: z.string(),
      quantity: z.int().positive(),
      pricePerUnit: z.number().positive(),
    })
  ),
  subtotal: z.number().positive(),
  tax: z.number().min(0),
  total: z.number().positive(),
})

type Order = z.infer<typeof OrderSchema>

function validateOrderMath(order: Order): string[] {
  const errors: string[] = []

  // Check subtotal
  const calculatedSubtotal = order.items.reduce((sum, item) => sum + item.quantity * item.pricePerUnit, 0)

  if (Math.abs(calculatedSubtotal - order.subtotal) > 0.01) {
    errors.push(`Subtotal mismatch: calculated ${calculatedSubtotal.toFixed(2)}, got ${order.subtotal.toFixed(2)}`)
  }

  // Check total
  const calculatedTotal = order.subtotal + order.tax
  if (Math.abs(calculatedTotal - order.total) > 0.01) {
    errors.push(`Total mismatch: calculated ${calculatedTotal.toFixed(2)}, got ${order.total.toFixed(2)}`)
  }

  return errors
}

async function extractOrder(text: string): Promise<Order> {
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: OrderSchema }),
    system: 'Extract order details. Calculate subtotal, tax (8%), and total accurately.',
    prompt: text,
    temperature: 0,
  })

  const order = result.output
  const mathErrors = validateOrderMath(order)

  if (mathErrors.length > 0) {
    console.warn('Math validation warnings:', mathErrors)
    // In production, you might retry or correct the values
  }

  return order
}

async function main(): Promise<void> {
  const order = await extractOrder('I ordered 3 widgets at $12.99 each and 2 gadgets at $24.50 each.')
  console.log(JSON.stringify(order, null, 2))
}

main().catch(console.error)
```

> **Advanced Note:** LLMs are notoriously bad at precise arithmetic. Always validate calculated fields after extraction. For financial applications, use the model to identify the items and quantities, then compute totals in your application code rather than trusting the model's math.

### Retry on Validation Failure

When structured output fails validation, retrying with a more explicit prompt often succeeds:

```typescript
// src/examples/structured-retry.ts

import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

async function generateWithRetry<T extends z.ZodType>(
  schema: T,
  prompt: string,
  maxRetries: number = 2
): Promise<z.infer<T>> {
  const model = mistral('mistral-small-latest')
  let lastError: Error | null = null

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const enhancedPrompt =
        attempt === 0
          ? prompt
          : `${prompt}\n\nIMPORTANT: Your previous response did not match the required format. Please follow the schema exactly.`

      const result = await generateText({
        model,
        output: Output.object({ schema }),
        prompt: enhancedPrompt,
        temperature: 0,
      })

      return result.output
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error))
      console.warn(`Attempt ${attempt + 1} failed: ${lastError.message}`)
    }
  }

  throw new Error(`Failed after ${maxRetries + 1} attempts: ${lastError?.message}`)
}
```

---

## Section 9: Streaming Structured Output

### Why Stream Structured Output?

Streaming structured output serves two purposes:

1. **Progressive display:** Show partial results as they arrive (e.g., display the first few extracted entities while the model continues processing).
2. **Early exit:** If you see an unacceptable result in the first few fields, you can abort without waiting for the full generation.

### The streamText with Output.object() Pattern

`streamText` with `Output.object()` works like `generateText` with `Output.object()` but delivers partial objects as the model generates them:

```typescript
// src/examples/structured-streaming.ts

import { streamText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const ArticleSchema = z.object({
  title: z.string().describe('Article title'),
  author: z.string().describe('Author name'),
  publishDate: z.string().describe('Publication date'),
  summary: z.string().describe('Brief summary of the article (2-3 sentences)'),
  tags: z.array(z.string()).describe('Relevant topic tags'),
  readingTime: z.int().positive().describe('Estimated reading time in minutes'),
})

async function streamArticleAnalysis(text: string): Promise<void> {
  const result = streamText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ArticleSchema }),
    prompt: `Analyze this article:\n\n${text}`,
    temperature: 0,
  })

  // Stream partial objects as they arrive
  for await (const partialObject of result.partialOutputStream) {
    console.clear()
    console.log('=== Article Analysis (streaming) ===')

    if (partialObject.title) console.log(`Title: ${partialObject.title}`)
    if (partialObject.author) console.log(`Author: ${partialObject.author}`)
    if (partialObject.publishDate) console.log(`Date: ${partialObject.publishDate}`)
    if (partialObject.summary) console.log(`Summary: ${partialObject.summary}`)
    if (partialObject.tags) console.log(`Tags: ${partialObject.tags.join(', ')}`)
    if (partialObject.readingTime) console.log(`Reading time: ${partialObject.readingTime} min`)
  }

  // After streaming, get the final validated object
  const finalResult = await result.output
  console.log('\n=== Final (validated) ===')
  console.log(JSON.stringify(finalResult, null, 2))
}

async function main(): Promise<void> {
  await streamArticleAnalysis(`
    "The Future of TypeScript" by Jane Developer, published March 1, 2025.
    TypeScript continues to evolve with new features like the satisfies operator
    and improved type inference. The language's adoption has grown 40% year over year,
    making it the most popular transpiled language. This article explores upcoming
    features and their implications for enterprise development.
  `)
}

main().catch(console.error)
```

> **Beginner Note:** The `partialOutputStream` delivers incomplete objects — fields may be `undefined` even if they are required in the schema. Always check for field presence when rendering partial results. The final `result.output` is the fully validated output.

### Streaming with Callbacks

For more control over the streaming lifecycle:

```typescript
// src/examples/structured-streaming-callbacks.ts

import { streamText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const ExtractSchema = z.object({
  entities: z.array(
    z.object({
      name: z.string(),
      type: z.enum(['person', 'org', 'location']),
    })
  ),
  summary: z.string(),
})

async function streamWithCallbacks(text: string): Promise<void> {
  let updateCount = 0

  const result = streamText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: ExtractSchema }),
    prompt: text,
    temperature: 0,
    onFinish(event) {
      console.log(`\nStreaming complete. Updates: ${updateCount}`)
      console.log(`Tokens: ${JSON.stringify(event.usage)}`)
    },
  })

  for await (const partial of result.partialOutputStream) {
    updateCount++
    const entityCount = partial.entities?.length ?? 0
    process.stdout.write(`\rEntities found so far: ${entityCount}`)
  }

  const final = await result.output
  console.log('\n\nFinal result:')
  console.log(JSON.stringify(final, null, 2))
}

async function main(): Promise<void> {
  await streamWithCallbacks(`
    Tim Cook, CEO of Apple, met with Satya Nadella from Microsoft
    at the tech summit in San Francisco. They discussed AI partnerships
    with Google's Sundar Pichai, who joined remotely from Mountain View.
  `)
}

main().catch(console.error)
```

### When to Use streamText vs generateText with Output.object()

| Scenario                  | Use                                | Why                                           |
| ------------------------- | ---------------------------------- | --------------------------------------------- |
| Background processing     | `generateText` + `Output.object()` | Simpler code, no stream handling              |
| User-facing extraction    | `streamText` + `Output.object()`   | Show progress to user                         |
| Quick classification      | `generateText` + `Output.object()` | Small output, streaming overhead not worth it |
| Large schema (10+ fields) | `streamText` + `Output.object()`   | Show fields as they populate                  |
| Pipeline step             | `generateText` + `Output.object()` | Need complete object for next step            |
| With early abort logic    | `streamText` + `Output.object()`   | Can stop on bad partial results               |

---

## Quiz

### Question 1 (Easy)

What is the primary advantage of `generateText` with `Output.object()` over plain `generateText` for data extraction?

- A) It is faster
- B) It guarantees the output matches a typed schema
- C) It uses fewer tokens
- D) It works with more providers

**Answer: B** — `Output.object()` constrains the model's output to match a Zod schema exactly, giving you compile-time type safety and runtime validation. This eliminates the need for fragile text parsing and ensures your application receives data in the expected format.

---

### Question 2 (Easy)

Which Zod method adds a description that helps the LLM understand a field's purpose?

- A) `.label()`
- B) `.comment()`
- C) `.describe()`
- D) `.hint()`

**Answer: C** — The `.describe()` method adds a description string to the schema field. This description is included in the schema sent to the model and helps it understand what each field should contain, improving output accuracy.

---

### Question 3 (Medium)

You define a schema with `z.enum(['critical', 'high', 'medium', 'low'])`. What happens if the model tries to return `"urgent"`?

- A) `"urgent"` is silently accepted
- B) It is automatically mapped to the closest enum value
- C) The `generateText` call fails with a validation error
- D) The field is set to `undefined`

**Answer: C** — Zod enums are strict. If the model produces a value not in the enum list, validation fails and `generateText` with `Output.object()` throws an error. This is by design — enums guarantee your application only receives expected values.

---

### Question 4 (Medium)

When should you use `z.string().optional()` vs `z.string().nullable()` in a schema for LLM output?

- A) They are interchangeable
- B) `optional()` when the field might not exist at all; `nullable()` when it exists but may have no value
- C) `optional()` for strings, `nullable()` for numbers
- D) `nullable()` is only for database schemas

**Answer: B** — `optional()` means the field can be `undefined` (omitted entirely), while `nullable()` means the field is present but can be `null`. Use `optional()` when information might not be available in the input; use `nullable()` when the field should always be present in the output but may have an empty value.

---

### Question 5 (Hard)

You have a schema with a `total` field that should equal the sum of item prices. The model consistently returns a `total` that is off by a few cents. What is the best approach?

- A) Add `.describe('Must be exactly equal to sum of item prices')` to the total field
- B) Increase the temperature so the model explores different values
- C) Remove `total` from the schema and compute it in application code after extraction
- D) Add a post-validation step that corrects the model's total based on the extracted items

**Answer: C** — LLMs are unreliable at precise arithmetic. The most robust approach is to extract the raw data (items, quantities, prices) and compute derived values like totals in deterministic application code. Option D is acceptable but option C eliminates the error entirely.

---

## Exercises

### Exercise 1: Product Review Schema

**Objective:** Design and use a comprehensive product review analysis schema.

**Specification:**

1. Create a file `src/exercises/ex10-review-schema.ts`
2. Define a `ReviewAnalysis` schema with:
   - `sentiment`: enum (positive, negative, neutral, mixed)
   - `rating`: number 1-5
   - `aspects`: array of `{ aspect: string, sentiment: enum, quote: string }`
   - `recommendation`: boolean
   - `summary`: string (max 150 chars)
3. Export an async function `analyzeReview(text: string): Promise<ReviewAnalysis>`
4. Test with at least 3 reviews: one clearly positive, one clearly negative, one mixed

---

### Exercise 2: Entity Extractor

**Objective:** Build a named entity extractor that returns structured data from unstructured text.

**Specification:**

1. Create a file `src/exercises/ex11-entity-extractor.ts`
2. Define an `Extraction` schema with:
   - `entities`: array of `{ value: string, type: enum('person' | 'organization' | 'location' | 'date' | 'money'), source: string }`
   - `entityCount`: number
   - `dominantType`: the entity type that appears most frequently
3. Export `extractEntities(text: string): Promise<Extraction>`
4. Add post-validation: verify `entityCount` matches `entities.length`
5. Test with a news article paragraph containing multiple entity types

---

### Exercise 3: Form Filler

**Objective:** Build an AI form filler that extracts structured information from conversational text.

**Specification:**

1. Create a file `src/exercises/ex12-form-filler.ts`
2. Define a `RegistrationForm` schema with:
   - Required: `fullName`, `email`
   - Optional: `phone`, `company`, `role`, `dietaryRestrictions`
   - Enum: `ticketType` (general, vip, speaker)
3. Export `fillForm(conversationalText: string): Promise<{ form: RegistrationForm, completeness: number }>`
4. `completeness` should be the percentage of fields filled (0-100)
5. Test with inputs of varying completeness (full info, partial info, minimal info)

---

### Exercise 4: Streaming Structured Output Demo

**Objective:** Build a streaming structured output demo that shows progressive field population.

**Specification:**

1. Create a file `src/exercises/ex13-stream-object.ts`
2. Define a schema with at least 6 fields for a "Movie Analysis":
   - `title`, `year`, `genre` (enum), `rating` (1-10), `themes` (string array), `synopsis` (string)
3. Export `streamMovieAnalysis(description: string): Promise<MovieAnalysis>`
4. During streaming, log each field as it becomes available
5. After streaming completes, log the final validated object
6. Include timing: report how long streaming took and time-to-first-field

> **Local Alternative (Ollama):** Structured output with `generateText` and `Output.object()` works with Ollama models that support JSON mode. Use `ollama('qwen3.5')` — it handles JSON generation well for most schemas. Very complex nested schemas may need a larger model like `ollama('qwen3.5:cloud')`. If you encounter malformed JSON, add "Respond in valid JSON only" to your system prompt.

---

## Summary

In this module, you learned:

1. **The problem with free text:** Why parsing unstructured model output is fragile, and how `generateText` with `Output.object()` eliminates it.
2. **Zod basics:** How to define schemas with types, constraints, and descriptions that guide the model.
3. **generateText with Output.object():** How to constrain model output to exact TypeScript types with compile-time safety and runtime validation.
4. **Nested schemas:** How to model complex, hierarchical data structures.
5. **Enum constraints:** How to limit the model's output to a finite set of valid values.
6. **Optional fields and defaults:** How to handle missing information gracefully.
7. **Schema design patterns:** Classification, extraction, multi-step analysis, comparison, and error reporting patterns.
8. **Validation and error handling:** How to use parse/safeParse, post-validate computed fields, and retry on failure.
9. **Streaming structured output:** How to use `streamText` with `Output.object()` for progressive display and early abort.

In Module 4, you will combine structured output with conversation management to build multi-turn applications that maintain state across interactions.
