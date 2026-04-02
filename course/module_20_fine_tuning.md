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

interface RequirementAnalysis {
  requirement: string
  promptEngineeringViable: boolean
  ragViable: boolean
  fineTuningNeeded: boolean
  reasoning: string
}
```

Consider five common requirements and decide which approach fits each:

1. "Model should know about our internal product catalog" -- Is this a knowledge problem or a behavior problem? Which approach adds external knowledge at inference time?
2. "Model should always respond in a specific JSON format" -- Can structured output with Zod schemas (Module 3) handle this? When would fine-tuning be needed instead?
3. "Model should write in our brand's distinctive voice" -- Can you fully capture style in a prompt, or does the model need to internalize it?
4. "Model should classify medical documents using ICD-10 codes" -- With hundreds of categories and domain-specific terminology, how much can a prompt do?
5. "Model should handle 10,000 requests per minute cheaply" -- How does fine-tuning a smaller model reduce per-request cost?

Build a `RequirementAnalysis` array that captures your reasoning for each case.

Then build an automated decision helper:

```typescript
async function analyzeFineTuningNeed(description: string): Promise<{
  recommendation: 'prompt_engineering' | 'prompt_caching' | 'rag' | 'fine_tuning' | 'combination'
  confidence: number
  reasoning: string
  prerequisites: string[]
}>
```

Use `Output.object` with a Zod schema to get structured output. The system prompt should define the decision hierarchy: try prompt engineering first (cheapest, fastest), then prompt caching, then RAG (when external knowledge is needed), then fine-tuning (when behavior must change fundamentally), then a combination for complex cases. What criteria would you include in the system prompt to help the model decide?

> **Beginner Note:** Think of it this way — prompt engineering tells the model what to do each time, RAG gives it information to work with, and fine-tuning changes what the model is. You prompt-engineer a model to write like Shakespeare; you fine-tune a model to be a Shakespeare-writing model.

> **Prompt Caching as Middle Ground:** Before jumping to fine-tuning to reduce long system prompt costs, consider prompt caching. Providers like Anthropic and OpenAI cache repeated system prompt prefixes, so the same 2,000-token system prompt is only billed once and reused across requests. This achieves some of the same cost and latency benefits as fine-tuning (shorter effective prompt, faster responses) without any training. Prompt caching is the right choice when your system prompt is stable and your main cost concern is repeatedly sending the same instructions.

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
```

Build a `computeFineTuningROI` function:

```typescript
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
}): FineTuningCostBenefit
```

The key calculations are:

- Convert cost-per-million-tokens to cost-per-token by dividing by 1,000,000
- Current daily cost = prompt tokens x requests per day x cost per token
- Fine-tuned daily cost = fine-tuned prompt tokens x requests per day x fine-tuned cost per token
- Training tokens = training examples x avg tokens per example x epochs
- Training cost = training tokens x training cost per token
- Break-even days = training cost / daily savings (or `Infinity` if savings <= 0)

Try it with a realistic scenario: 2,000-token system prompt, 5,000 requests/day, $3/million tokens, 500 training examples at 500 tokens each, 3 epochs at $25/million training tokens, and a fine-tuned prompt of only 200 tokens. How many days until the training investment pays for itself?

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
```

Each training example is a conversation with a system prompt, one or more user messages, and one or more assistant messages. Build a `trainingExamples` array with at least two customer support examples. Each should have a system prompt establishing the agent's role, a customer question, and a high-quality assistant response that demonstrates the tone and structure you want the model to learn.

What makes a good training example? Think about: does the response acknowledge the customer's situation? Does it provide clear, actionable steps? Does it end by inviting further questions?

Then build a `writeTrainingData` function:

```typescript
function writeTrainingData(examples: FineTuningExample[], outputPath: string): void
```

Convert each example to a JSON string using `JSON.stringify`, join them with newlines, and write to the output path using `writeFileSync`. Why must each line be a complete JSON object rather than using a single JSON array for the whole file?

### Multi-Turn Conversation Data

Fine-tuning on multi-turn conversations teaches the model how to handle context and follow-ups. A multi-turn example simply has more alternating user/assistant messages after the system prompt.

Build a `multiTurnExample` that demonstrates a 3-turn support conversation (e.g., a password reset flow). The first turn handles the initial question, the second handles a follow-up complication, and the third resolves the issue. What makes multi-turn examples valuable for fine-tuning that single-turn examples cannot capture?

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

interface ChatLog {
  messages: { sender: 'customer' | 'agent'; text: string; timestamp: string }[]
}
```

Build three conversion functions:

1. `qaToFineTuning(pairs: RawQAPair[], systemPrompt: string): FineTuningExample[]` — Map each Q&A pair to the messages format. If a pair has `context`, how should you include it in the user message? Consider prefixing with `"Context: ..."`.

2. `csvToFineTuning(rows: CSVRow[], systemPrompt: string): FineTuningExample[]` — Map each row's `input` to a user message and `output` to an assistant message. Straightforward, but what about the `category` field — should it influence the system prompt?

3. `chatLogToFineTuning(logs: ChatLog[], systemPrompt: string): FineTuningExample[]` — Map each chat log to a conversation. The `sender` field needs role mapping: `'customer'` becomes `'user'`, `'agent'` becomes `'assistant'`. What should the first message in each example always be?

> **Beginner Note:** JSONL is just regular JSON objects, one per line. Each line is a complete, valid JSON object. This format is efficient for streaming processing — you can process one example at a time without loading the entire file into memory.

> **Production Insight: Implicit Training Data from Sessions.** Production AI tools generate high-quality fine-tuning data as a byproduct of normal operation. Every multi-turn session is a conversation with tool calls, results, and user approvals or denials baked in. Session summaries extract key patterns, and user feedback (approvals, rejections, edits) provides preference signals. If you run a production AI assistant, your logs are a potential fine-tuning dataset — the data you need may already exist.

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
```

Build an array of `QualityFilter` objects that catch common data problems. Each filter's `check` function examines a single training example and returns whether it passed. Think about what makes a training example invalid:

1. **minimum_length** — Is there an assistant message? Is it at least 50 characters? Responses shorter than that are unlikely to be helpful training signals.
2. **maximum_length** — Do the total tokens across all messages exceed 4,096? Use a rough estimate of ~4 characters per token.
3. **has_system_prompt** — Does at least one message have `role: 'system'`?
4. **valid_turn_order** — After filtering out system messages, do user and assistant messages alternate? What happens if two consecutive messages have the same role?
5. **no_empty_messages** — Does any message have empty content after trimming?
6. **assistant_ends_conversation** — Is the last message from the assistant? Why does this matter for training?

Also build a token estimation helper:

```typescript
function estimateTokenCount(text: string): number
```

A rough approximation of ~4 characters per token works for English text.

Then build the filtering function:

```typescript
interface FilterReport {
  total: number
  passed: number
  filtered: number
  filterBreakdown: Record<string, number>
  examples: FineTuningExample[]
}

function filterTrainingData(examples: FineTuningExample[], filters: QualityFilter[]): FilterReport
```

For each example, run all filters. If any filter fails, increment its count in `filterBreakdown` and exclude the example. Should you fail fast (stop at the first failing filter) or run all filters? What are the trade-offs?

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
```

Build an `analyzeDataset` function:

```typescript
function analyzeDataset(examples: FineTuningExample[]): DatasetStats
```

Iterate through all examples, counting categories (you can derive category from the system prompt or first 50 characters), computing average message lengths for assistant and user roles, counting non-system turns per example, and bucketing assistant message lengths into short/medium/long. What does a heavily skewed length distribution tell you about your training data?

Then build a `balanceDataset` function:

```typescript
function balanceDataset(
  examples: FineTuningExample[],
  categoryExtractor: (ex: FineTuningExample) => string,
  targetPerCategory?: number
): FineTuningExample[]
```

Group examples by category using the extractor function. If no `targetPerCategory` is given, use the minimum group size to avoid overrepresentation. Shuffle each group and take up to `target` examples. Why is balancing important? What happens when 80% of your training data is one category?

### LLM-Assisted Data Enhancement

Use an LLM to improve or augment your training data.

Build an `enhanceTrainingExample` function:

```typescript
async function enhanceTrainingExample(example: FineTuningExample): Promise<FineTuningExample>
```

Extract the user and assistant messages. Use `generateText` with a system prompt that instructs the model to improve the assistant response: make it more helpful, add structure (bullet points, numbered steps), maintain a professional tone, and keep a similar length. Return the example with the assistant message replaced by the improved version. How do you handle examples that have no assistant or user message?

Then build a synthetic data generator:

```typescript
async function generateSyntheticExamples(
  seedExamples: FineTuningExample[],
  count: number,
  systemPrompt: string
): Promise<FineTuningExample[]>
```

Use `Output.object` to generate structured output with a schema containing an array of `{ userMessage, assistantMessage }` objects. The prompt should include the seed examples (extract their user messages) so the model can generate similar but new scenarios. The system prompt should emphasize covering NEW scenarios, not repeating seeds.

Map the generated objects back to `FineTuningExample` format by wrapping each in the standard messages array with system, user, and assistant roles. What risks does synthetic data generation introduce? How would you validate that synthetic examples are not duplicates of seeds?

> **Advanced Note:** Synthetic data generation is powerful but comes with risks. Models can introduce subtle biases, generate plausible-but-wrong examples, and produce overly uniform data. Always have a human review a sample of synthetic examples before including them in your training set.

---

## Section 4: Fine-tuning APIs

### OpenAI Fine-tuning with Vercel AI SDK

OpenAI provides the most accessible fine-tuning API. After training, you use the fine-tuned model through the Vercel AI SDK just like any other model. Fine-tuning management (uploading files, starting jobs) uses the OpenAI SDK directly, while inference uses the Vercel AI SDK.

```typescript
import { openai as openaiProvider } from '@ai-sdk/openai'

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
```

The workflow has three steps:

**Step 1: Prepare and validate.** Build a `prepareForFineTuning` function:

```typescript
async function prepareForFineTuning(
  examples: FineTuningExample[],
  outputPath: string
): Promise<{ valid: boolean; stats: DatasetStats; warnings: string[] }>
```

Filter examples using your quality filters, analyze the filtered dataset, and check for critical issues. OpenAI requires at least 10 examples and recommends 50-100. What other checks should generate warnings? Think about average response length — very short assistant responses may not provide enough signal for training.

**Step 2: Start fine-tuning.** Build a `startFineTuning` function:

```typescript
async function startFineTuning(params: {
  trainingFilePath: string
  baseModel: string
  epochs?: number
  batchSize?: number
  learningRateMultiplier?: number
  suffix?: string
}): Promise<FineTuningJob>
```

In production, you would use the OpenAI SDK to upload the file (`openai.files.create`) and create a job (`openai.fineTuning.jobs.create`). For this exercise, return a placeholder `FineTuningJob` with `status: 'queued'`. What does the `suffix` parameter do? It helps you identify your fine-tuned model later (e.g., `ft:gpt-5-mini:my-org:support-v1:abc123`).

**Step 3: Use the fine-tuned model.** The key insight is that using a fine-tuned model through the Vercel AI SDK is identical to using a base model — you just change the model ID:

```typescript
const { text } = await generateText({
  model: openaiProvider('ft:gpt-5-mini-2026-01-15:my-org:support-v1:abc123'),
  system: 'You are a helpful customer support agent.',
  prompt: userQuestion,
})
```

Build a `useFineTunedModel` function that takes a model ID and prompt, calls `generateText`, and returns the text. Note that fine-tuned models often need shorter system prompts — why?

### Anthropic's Approach to Customization

Anthropic offers fine-tuning through their API for enterprise customers. The Vercel AI SDK abstracts away provider differences for inference, so fine-tuned Anthropic models are used identically to base models.

Build a `compareBaseVsCustomized` function that demonstrates the key benefit of fine-tuning: prompt size reduction. Use the same question with two different configurations:

1. A base model with a long system prompt (~150 words) specifying tone, format, and rules
2. A fine-tuned model (simulated with the same base model for now) with a short system prompt (~10 words)

Compare the `usage.inputTokens` between the two calls. How many tokens does the long system prompt add? At 100,000 requests per day, what would that cost difference be?

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
```

Build a `hyperparameterGuides` array that documents the three key parameters:

1. **Epochs** — Number of complete passes through the training dataset. More epochs mean the model sees each example more times. Default is auto (typically 3-4), range is 1-10. How does dataset size affect the ideal number? Small datasets (<100) need more passes (4-8), while large datasets (>1000) risk overfitting with more than 1-2.

2. **Batch size** — Number of examples processed together in one training step. Larger batches are more stable but may generalize less well. Default is auto (typically 1-8), range is 1-32. Why would you use smaller batches for smaller datasets?

3. **Learning rate multiplier** — Scales the base learning rate. Higher values learn faster but risk overshooting optimal weights. Default is auto (typically 1.0-2.0), range is 0.1-5.0. How does example complexity affect the ideal learning rate?

Then build a `suggestHyperparameters` function:

```typescript
function suggestHyperparameters(
  datasetSize: number,
  avgExampleTokens: number,
  taskComplexity: 'simple' | 'moderate' | 'complex'
): HyperparameterConfig
```

The logic follows these principles:

- **Epochs** are inversely related to dataset size: fewer examples need more passes. Start with size-based ranges (<100: 6, <500: 4, <2000: 3, else 2), then adjust for complexity (complex tasks add epochs, simple tasks subtract). Clamp to [1, 10].
- **Batch size** scales with dataset size: <50 examples use batch 1, <200 use 4, <1000 use 8, else 16.
- **Learning rate** is inversely related to example complexity (token count): longer examples (>1000 tokens) use 0.5, medium (>500) use 1.0, short use 1.8.

What would `suggestHyperparameters(300, 400, 'moderate')` return?

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
```

Build a `hyperparameterSearch` function:

```typescript
async function hyperparameterSearch(
  searchConfig: HyperparameterSearchConfig,
  evalTestCases: { input: string; expected: string }[]
): Promise<SearchResult[]>
```

For each configuration in the search space, start a fine-tuning job, wait for completion (in production you would poll), then evaluate the resulting model on the test cases. Sort results by eval score descending. How do you detect overfitting from the training and validation loss values?

Also build an `evaluateModel` helper:

```typescript
async function evaluateModel(modelId: string, testCases: { input: string; expected: string }[]): Promise<number>
```

For each test case, generate output and compute a simple score (e.g., does the output contain the expected text?). Return the average score across all test cases.

Define a search space that varies one parameter at a time from a baseline. For example, start with `{epochs: 4, batchSize: 4, learningRateMultiplier: 1.0}` and create variants that change epochs (2, 6), batch size (8), or learning rate (0.5, 2.0). Why is it important to vary one parameter at a time rather than trying random combinations?

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
```

Build a `compareBaseVsFineTuned` function:

```typescript
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
): Promise<ModelComparisonResult>
```

For each test case, generate outputs from both models in parallel using `Promise.all`. Note that the base model uses `tc.systemPrompt` while the fine-tuned model may need no system prompt (or a much shorter one) — why?

Score both outputs using the `judge` function. Track scores per category using a `Record<string, { base: number[], ft: number[] }>`. After processing all test cases:

- Compute the mean score for each model
- Compute the improvement (fine-tuned mean minus base mean)
- Compute improvement as a percentage: `(improvement / baseMean) * 100`
- Compute per-category means and improvements
- Determine the verdict: improvement > 0.05 means "improved", < -0.05 means "degraded", otherwise "no_change"

Why is 0.05 a reasonable threshold rather than 0? What role does noise play in small score differences?

### Held-Out Test Set Management

Never evaluate on data the model was trained on. Always maintain a separate test set.

```typescript
interface DataSplit {
  training: FineTuningExample[]
  validation: FineTuningExample[]
  test: FineTuningExample[]
}
```

Build a `splitDataset` function:

```typescript
function splitDataset(
  examples: FineTuningExample[],
  trainRatio: number = 0.8,
  validationRatio: number = 0.1,
  testRatio: number = 0.1,
  seed: number = 42
): DataSplit
```

First, validate that the ratios sum to 1.0 (within a small tolerance like 0.01). Then shuffle the examples deterministically using the seed — why is a deterministic shuffle important? Without it, running the split twice would give different splits, making experiments non-reproducible.

For the seeded shuffle, implement a Fisher-Yates shuffle using a linear congruential generator: `seed = (seed * 1664525 + 1013904223) & 0x7fffffff`, and use `seed % (i + 1)` for the swap index.

Split the shuffled array at the computed boundaries using `slice`. Write each split to a separate JSONL file using your `writeTrainingData` function.

What is the purpose of each split? The training set is what the model learns from. The validation set detects overfitting during training. The test set provides an unbiased estimate after training is complete. Why should you never look at test set results during the training iteration cycle?

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
```

Build a `planIteration` function:

```typescript
function planIteration(evalResults: ModelComparisonResult, targetScore: number): IterationPlan
```

This function analyzes evaluation results and recommends specific actions. The logic follows a diagnostic pattern:

1. **Identify weak categories** — Which categories in `evalResults.byCategory` have a fine-tuned mean below `targetScore`?
2. **If the model degraded** (verdict is `'degraded'`), the top priority is data quality: recommend removing bad examples and reducing epochs/learning rate. Why is data quality the most likely culprit when fine-tuning makes things worse?
3. **If specific categories are weak**, recommend adding more training examples for those categories. What priority should this have?
4. **If scores are close to target** (within 90%), recommend data augmentation to push past the plateau. Why might augmentation help when you are close but not there?
5. **If no actions are identified**, suggest considering a larger base model.

Generate an `estimatedImpact` string summarizing the plan. What information would be most useful to include?

### Data Augmentation Techniques

Build two augmentation functions:

```typescript
async function paraphraseExamples(examples: FineTuningExample[], variations: number = 2): Promise<FineTuningExample[]>
```

For each example, extract the user message and use `Output.object` to generate `variations` paraphrased versions. The system prompt should instruct the model to keep meaning identical while varying wording, structure, and style. For each paraphrase, create a new `FineTuningExample` with the user message replaced but the assistant message preserved. Start with the original examples in the result array and append paraphrased versions.

Why do you keep the same assistant message when paraphrasing the user message? What would happen if different phrasings of the same question got different responses?

```typescript
async function addDifficultyVariations(examples: FineTuningExample[]): Promise<FineTuningExample[]>
```

For each example, generate a harder version of the same scenario using `Output.object` with a schema containing `{ harderInput, appropriateResponse }`. The system prompt should instruct the model to make the customer's question more ambiguous, emotional, complex, or informal. Both the input and the response change in this case.

Create the augmented example by combining the original system messages with the new user and assistant messages. Why is it important to train on harder variations, not just paraphrases?

> **Advanced Note:** Each iteration cycle should follow this loop: train, evaluate, analyze failures, plan improvements, update data, retrain. Track your iterations in a spreadsheet or experiment tracker. Most successful fine-tuning projects go through 3-5 iterations before reaching their target quality.

---

## Section 8: Cost Analysis

### Comprehensive Cost Model

Fine-tuning costs extend beyond the training API call. A complete analysis includes data preparation, training, inference, and maintenance.

```typescript
interface FineTuningCostModel {
  dataPrep: {
    humanAnnotationHours: number
    hourlyRate: number
    llmAugmentationCost: number
    totalDataPrepCost: number
  }
  training: {
    trainingTokens: number
    costPerMillionTrainingTokens: number
    iterations: number
    totalTrainingCost: number
  }
  inference: {
    requestsPerMonth: number
    baseModelPromptTokens: number
    fineTunedPromptTokens: number
    outputTokensPerRequest: number
    baseModelCostPerMonth: number
    fineTunedCostPerMonth: number
    monthlySavings: number
  }
  maintenance: {
    retrainingFrequency: 'monthly' | 'quarterly' | 'annual'
    retrainingCostPerCycle: number
    annualMaintenanceCost: number
  }
  summary: {
    upfrontCost: number
    monthlyRunningCost: number
    monthlySavings: number
    breakEvenMonths: number
    firstYearROI: number
  }
}
```

Build a `buildCostModel` function:

```typescript
function buildCostModel(params: {
  annotationHours: number
  hourlyRate: number
  augmentationApiCalls: number
  augmentationCostPerCall: number
  trainingExamples: number
  avgTokensPerExample: number
  epochs: number
  trainingCostPerMillionTokens: number
  expectedIterations: number
  requestsPerMonth: number
  basePromptTokens: number
  fineTunedPromptTokens: number
  outputTokens: number
  baseCostPerMillionInputTokens: number
  baseCostPerMillionOutputTokens: number
  fineTunedCostPerMillionInputTokens: number
  fineTunedCostPerMillionOutputTokens: number
  retrainingFrequency: 'monthly' | 'quarterly' | 'annual'
}): FineTuningCostModel
```

The calculations break down into four cost centers:

1. **Data preparation** — Human annotation cost (hours x rate) plus LLM augmentation cost (API calls x cost per call). These are one-time upfront costs.

2. **Training** — Training tokens = examples x tokens per example x epochs. Multiply by cost per million training tokens. Then multiply by expected iterations (most projects need 3-5). Why do multiple iterations multiply the training cost?

3. **Inference (monthly)** — Compute monthly cost for both base and fine-tuned models. For each: input cost = (prompt tokens x requests / 1M) x cost per million input tokens; output cost follows the same pattern. Monthly savings = base cost minus fine-tuned cost. Where do the savings come from — shorter prompts, cheaper models, or both?

4. **Maintenance (annual)** — Retraining cost = single training cost x frequency multiplier (monthly=12, quarterly=4, annual=1). This is an ongoing cost that many teams forget.

5. **Summary** — Upfront cost = data prep + training. Monthly running cost = fine-tuned inference + maintenance/12. Break-even months = upfront cost / monthly savings (or `Infinity` if no savings). First-year ROI = `(savings*12 - upfront - maintenance) / upfront * 100`.

Try it with realistic numbers: 40 annotation hours at $50/hr, 500 training examples, 100K requests/month, base prompt of 2,000 tokens vs. fine-tuned prompt of 200 tokens. How many months until break-even?

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

**Question 6 (Medium):** A fine-tuned model performs well on the training set but poorly on the validation set. Which hyperparameter adjustment is most likely to help?

A) Increase the number of epochs to give the model more training time
B) Reduce the number of epochs or decrease the learning rate to prevent overfitting
C) Increase the batch size to process more examples at once
D) Add more duplicate examples to the training set

**Answer: B** — When training performance is good but validation performance is poor, the model is overfitting — memorizing training examples rather than learning generalizable patterns. Reducing epochs (fewer passes over the data) or decreasing the learning rate (smaller weight updates) both reduce the risk of overfitting. Increasing epochs (A) would worsen overfitting. Adding duplicates (D) would reinforce memorization.

---

**Question 7 (Hard):** You have 50 high-quality training examples and need 200 more. You use an LLM to generate synthetic examples based on the originals. What is the most critical quality check for the synthetic data?

A) Verify that synthetic examples are longer than the originals
B) Ensure synthetic examples do not duplicate originals and cover underrepresented categories while maintaining consistent quality
C) Check that all synthetic examples use the same response format
D) Confirm the LLM generated exactly 200 examples

**Answer: B** — Synthetic data generation risks two main problems: duplicating existing examples (which causes overfitting) and reinforcing category imbalances in the original set. The quality check should verify uniqueness (no near-duplicates of originals), diversity (coverage of underrepresented categories), and consistent quality (synthetic examples should match the quality bar of the hand-written ones). Format consistency (C) matters but is secondary to diversity and uniqueness.

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
