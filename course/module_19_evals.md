# Module 19: Evals & Testing

## Learning Objectives

- Understand why evaluation is essential for non-deterministic LLM applications
- Implement multiple evaluation types: exact match, fuzzy match, semantic similarity, and LLM-as-judge
- Build the LLM-as-judge pattern where one model evaluates another model's output
- Construct a reusable evaluation framework with test cases, scoring, and reporting
- Automate evaluation suites to run in CI pipelines
- Design regression tests that detect when prompt changes break existing behavior
- Create representative benchmarks with edge cases and adversarial inputs
- Compare human evaluation with automated evaluation and know when to use each
- Systematically A/B test prompt versions and measure improvements

---

## Why Should I Care?

Traditional software has a simple correctness model: given input X, the function should return Y. You write a unit test, assert equality, and move on. LLM applications shatter this model. Ask the same question twice with the same prompt and you may get different answers. Ask a slightly rephrased question and the output can change dramatically. Upgrade the model version and your carefully crafted prompts may behave differently.

Without evals, you are flying blind. You make a "small improvement" to a prompt and deploy it, only to discover days later that it broke a critical use case. You switch from one model to another for cost savings and have no idea whether quality degraded. You add retrieval context to your RAG pipeline and cannot tell whether the retrieved documents actually improved answers.

Evals are the bridge between "it seems to work" and "we have evidence it works." They transform LLM development from an art into an engineering discipline. Every production LLM application needs evals — not as a nice-to-have, but as a fundamental part of the development workflow.

This module teaches you to build eval systems from scratch using the Vercel AI SDK, so you can measure, track, and improve your LLM applications with confidence.

---

## Connection to Other Modules

- **Module 2 (Prompt Engineering)** provides the prompts you will evaluate and iterate on.
- **Module 3 (Structured Output)** makes evaluation easier — structured outputs are simpler to score than free-form text.
- **Module 9-10 (RAG)** produce pipelines that desperately need evaluation to ensure retrieval quality.
- **Module 14-15 (Agents)** create complex systems where evals catch subtle regressions.
- **Module 20 (Fine-tuning)** requires evals to compare base vs fine-tuned model performance.
- **Module 21 (Safety)** uses eval-like patterns to test guardrails and safety mechanisms.

---

## Section 1: Why Eval LLM Apps?

### The Non-Determinism Problem

Traditional software testing relies on deterministic behavior. A function that adds two numbers always returns the same result. LLM outputs are inherently non-deterministic — even with temperature set to 0, outputs can vary across API calls, model versions, and infrastructure changes.

Consider what happens when you call `generateText` with the same prompt three times in a row. With temperature 0, you might expect identical outputs, but infrastructure-level factors like floating-point precision and batching can still cause variation. The outputs might be semantically similar but differ in wording, structure, or length.

> **Beginner Note:** Temperature controls randomness in LLM outputs. Setting it to 0 makes the model as deterministic as possible, but it is not a guarantee of identical outputs.

Your task: build a function that demonstrates this non-determinism by calling the same prompt multiple times and comparing results.

```typescript
async function demonstrateNonDeterminism(): Promise<void>
```

Call `generateText` with `temperature: 0` and the same prompt three times. Collect the results into an array, then check whether all three are identical using `results.every(r => r === results[0])`. Print each result and the comparison. What do you observe?

### Regression Detection

The real danger is not that a single call varies — it is that a change you make (new prompt, new model, new retrieval strategy) causes systematic degradation that you do not notice until users complain.

```typescript
interface RegressionExample {
  input: string
  previousOutput: string
  currentOutput: string
  regressionDetected: boolean
}
```

Build a `checkForRegression` function that takes an old prompt, a new prompt, and an array of test inputs. For each input, generate outputs with both prompts and compare them.

```typescript
async function checkForRegression(
  oldPrompt: string,
  newPrompt: string,
  testInputs: string[]
): Promise<RegressionExample[]>
```

How would you detect that the new prompt is worse? You do not have a reference answer, so start with a simple heuristic. What metric could indicate degradation even without understanding the content? Consider: if the new prompt produces outputs that are drastically shorter or longer, that is a signal. What threshold for length difference would you choose, and why?

### What Makes LLM Evaluation Hard

LLM evaluation is fundamentally different from traditional testing:

1. **No single correct answer** — "Summarize this article" has many valid summaries.
2. **Quality is subjective** — What counts as "good" depends on context and user expectations.
3. **Evaluation itself requires intelligence** — You often need another LLM to judge quality.
4. **Distribution matters** — A single test case tells you little; you need statistical confidence.
5. **Metrics are multidimensional** — Accuracy, helpfulness, safety, and style all matter simultaneously.

> **Advanced Note:** Production eval systems typically track 5-10 metrics per evaluation, run hundreds of test cases, and maintain historical baselines. The investment pays off quickly — a single caught regression can save days of debugging and user churn.

---

## Section 2: Evaluation Types

### Exact Match

The simplest evaluation type. The output must exactly match an expected value. Useful for classification tasks, structured outputs, and deterministic extractions.

```typescript
interface EvalResult {
  testCase: string
  passed: boolean
  expected: string
  actual: string
  score: number
}
```

Build an `exactMatch` function with this signature:

```typescript
function exactMatch(expected: string, actual: string): EvalResult
```

Before comparing, normalize both strings. What variations should be ignored? Think about case differences and leading/trailing whitespace. The score should be 1.0 for a match, 0.0 otherwise.

Then build an `evalClassification` function that runs several sentiment classification test cases (`{ input: string, expected: string }[]`) through `generateText` and scores each result with `exactMatch`. What system prompt would constrain the model to produce output that exact match can evaluate?

### Fuzzy Match

When exact match is too strict, fuzzy matching allows for minor variations. The key algorithm here is **Levenshtein distance** — the minimum number of single-character edits (insertions, deletions, substitutions) needed to transform one string into another.

```typescript
function levenshteinDistance(a: string, b: string): number
```

Build this using dynamic programming with a 2D matrix of size `(a.length + 1) x (b.length + 1)`. Initialize the first row and column with incrementing values (0, 1, 2, ...). For each cell `[i][j]`, if characters match, copy the diagonal value. Otherwise, take the minimum of the three neighbors (diagonal + 1, left + 1, above + 1). What does each neighbor represent? Which edit operation does diagonal correspond to versus left versus above?

Then build the fuzzy match scorer:

```typescript
function fuzzyMatch(expected: string, actual: string, threshold?: number): EvalResult
```

Convert the distance to a similarity score using `1.0 - distance / maxLength`. Compare against a threshold (default 0.8). Before computing, should you normalize the strings? What score would `fuzzyMatch('San Francisco', 'san francisco, CA')` produce?

### Contains Match

Check whether the output contains expected keywords or phrases. Useful for factual recall evaluation.

```typescript
interface ContainsMatchOptions {
  caseSensitive?: boolean
  requireAll?: boolean
}

function containsMatch(expectedPhrases: string[], actual: string, options?: ContainsMatchOptions): EvalResult
```

Build this function. When `requireAll` is true (the default), every phrase must appear for the test to pass. When false, any single match is sufficient. The score should be the fraction of phrases found: `matchCount / expectedPhrases.length`. What normalization should you apply before checking if a phrase appears in the output?

### Semantic Similarity

> **Note:** This section uses OpenAI embeddings for semantic similarity. Substitute `mistral.embedding('mistral-embed')` if you only have a Mistral key.

Use embeddings to measure whether two texts have similar meaning, regardless of phrasing. The AI SDK provides `embed` and `cosineSimilarity` — see Module 8 for the math.

```typescript
import { embed, cosineSimilarity } from 'ai'
import { openai } from '@ai-sdk/openai'

async function semanticSimilarity(expected: string, actual: string, threshold?: number): Promise<EvalResult>
```

Build this function. Generate embeddings for both texts using `embed` with `openai.embedding('text-embedding-3-small')`, then compute `cosineSimilarity` between the two embedding vectors. The threshold defaults to 0.85. What happens when you compare "Regular exercise improves cardiovascular health" with "Working out consistently strengthens the heart"? Why does semantic similarity succeed where exact match fails?

> **Beginner Note:** Semantic similarity uses embedding models to convert text into numerical vectors, then measures the angle between those vectors. Two texts that mean the same thing will have vectors pointing in similar directions, giving a high cosine similarity score (close to 1.0).

### LLM-as-Judge (Preview)

The most flexible evaluation type. We dedicate the next section to this pattern. Here is the core idea — use `Output.object` with a Zod schema to get structured scores from the judge:

```typescript
const { output } = await generateText({
  model: mistral('mistral-small-latest'),
  output: Output.object({
    schema: z.object({ score: z.number().min(1).max(5), reasoning: z.string() }),
  }),
  system: `You are an expert evaluator. Score the output on: ${criteria}`,
  prompt: `Output to evaluate:\n${output}`,
})
```

> **Advanced Note:** Each evaluation type has different cost, speed, and accuracy profiles. Exact match is free and instant but rigid. Semantic similarity requires embedding API calls. LLM-as-judge is the most expensive but handles subjective quality best. Use the simplest evaluation type that works for each test case.

---

## Section 3: LLM-as-Judge Pattern

### Why Use an LLM to Judge?

Many LLM outputs cannot be evaluated with simple string matching. Summaries, creative writing, explanations, and conversational responses require understanding meaning, tone, completeness, and correctness. Using a powerful LLM to evaluate another LLM's output captures these nuances.

### Single-Criterion Judge

The simplest LLM judge evaluates on one dimension at a time. Build it using structured output:

```typescript
const JudgeResultSchema = z.object({
  score: z.number().min(1).max(5),
  reasoning: z.string(),
  suggestions: z.array(z.string()).optional(),
})

type JudgeResult = z.infer<typeof JudgeResultSchema>

async function singleCriterionJudge(
  criterion: string,
  criterionDescription: string,
  output: string,
  context?: string
): Promise<JudgeResult>
```

The system prompt should define a clear scoring scale (1-5) and include the criterion name and description. Use `Output.object({ schema: JudgeResultSchema })` to get structured results. The user prompt should include the context (if provided) and the output to evaluate.

Key design question: why is a 1-5 scale better than a binary pass/fail for LLM-as-judge? What information do you lose with pass/fail that a 5-point scale preserves? How does a graded scale help you track improvement across prompt iterations?

### Multi-Criterion Judge

Evaluate on multiple dimensions simultaneously for a comprehensive quality assessment.

```typescript
const MultiCriterionResultSchema = z.object({
  scores: z.record(z.string(), z.number().min(1).max(5)),
  overallScore: z.number().min(1).max(5),
  reasoning: z.string(),
  strengths: z.array(z.string()),
  weaknesses: z.array(z.string()),
})

type MultiCriterionResult = z.infer<typeof MultiCriterionResultSchema>

interface EvalCriterion {
  name: string
  description: string
  weight: number
}

async function multiCriterionJudge(
  criteria: EvalCriterion[],
  output: string,
  context: string
): Promise<MultiCriterionResult>
```

Build this function. The system prompt should list each criterion's name, description, and weight so the judge knows what to evaluate and how to compute the overall score as a weighted average.

Consider: for a RAG-generated answer, what criteria would you use? How would you weight relevance vs. accuracy vs. completeness vs. clarity? What happens if you give all criteria equal weight — does that match how users actually judge quality?

### Pairwise Comparison Judge

Compare two outputs directly to determine which is better. This eliminates absolute scoring biases.

```typescript
const PairwiseResultSchema = z.object({
  winner: z.enum(['A', 'B', 'tie']),
  confidence: z.number().min(0).max(1),
  reasoning: z.string(),
})

type PairwiseResult = z.infer<typeof PairwiseResultSchema>

async function pairwiseJudge(
  criterion: string,
  outputA: string,
  outputB: string,
  context: string
): Promise<PairwiseResult>
```

Build this function. Then build a `comparePromptVersions` function that uses it to compare two system prompts across multiple test inputs:

```typescript
async function comparePromptVersions(
  promptA: string,
  promptB: string,
  testInputs: string[],
  criterion: string
): Promise<{ winsA: number; winsB: number; ties: number }>
```

Critical detail: you must **randomize the presentation order** to reduce position bias. For each test input, flip a coin to decide which output is shown as "A" vs "B", then map the winner back to the actual prompt. How would you implement this randomization? What happens to your win counts if you skip this step?

> **Beginner Note:** Position bias is a real problem with LLM judges — they may prefer whichever output is presented first. Randomizing the order and running each comparison twice (swapping positions) helps mitigate this bias.

### Reference-Based Judge

When you have a gold-standard reference answer, the judge can compare the output against it.

```typescript
const ReferenceJudgeResultSchema = z.object({
  faithfulness: z.number().min(1).max(5),
  completeness: z.number().min(1).max(5),
  conciseness: z.number().min(1).max(5),
  overallScore: z.number().min(1).max(5),
  missingInformation: z.array(z.string()),
  incorrectInformation: z.array(z.string()),
  reasoning: z.string(),
})

type ReferenceJudgeResult = z.infer<typeof ReferenceJudgeResultSchema>

async function referenceBasedJudge(reference: string, output: string, question: string): Promise<ReferenceJudgeResult>
```

Build this function. The system prompt should define three scoring dimensions: faithfulness (does the output contain only correct information?), completeness (are all important points from the reference covered?), and conciseness (is there unnecessary information?).

Why is it important to track `missingInformation` and `incorrectInformation` separately rather than just giving a score? How do these lists help you improve your system in ways that a single number cannot?

> **Advanced Note:** LLM-as-judge has known failure modes: self-preference bias (models prefer their own outputs), verbosity bias (longer answers score higher), and position bias (first option preferred). Mitigate these with randomization, calibration sets, and using a different model family as the judge than the model being evaluated.

---

## Section 4: Building an Eval Framework

### Test Case Structure

A robust eval framework starts with well-defined test cases.

```typescript
interface TestCase {
  id: string
  name: string
  description?: string
  input: string
  systemPrompt?: string
  expectedOutput?: string
  expectedKeywords?: string[]
  category: string
  difficulty: 'easy' | 'medium' | 'hard'
  evaluators: EvaluatorType[]
  metadata?: Record<string, unknown>
}

type EvaluatorType = 'exact_match' | 'fuzzy_match' | 'contains' | 'semantic_similarity' | 'llm_judge' | 'custom'

interface EvalConfig {
  name: string
  description: string
  model: string
  testCases: TestCase[]
  evaluators: Map<EvaluatorType, EvaluatorFn>
  concurrency: number
  retries: number
}

type EvaluatorFn = (testCase: TestCase, output: string) => Promise<EvalResult>
```

### The EvalRunner

A central runner orchestrates test execution and scoring. Build an `EvalRunner` class with these result types:

```typescript
interface EvalRunResult {
  config: Pick<EvalConfig, 'name' | 'description' | 'model'>
  timestamp: string
  results: TestCaseResult[]
  summary: EvalSummary
  duration: number
}

interface TestCaseResult {
  testCase: TestCase
  output: string
  evalResults: EvalResult[]
  aggregateScore: number
  passed: boolean
  latencyMs: number
  tokenUsage: { input: number; output: number }
  error?: string
}

interface EvalSummary {
  totalTests: number
  passed: number
  failed: number
  averageScore: number
  scoresByCategory: Record<string, number>
  scoresByDifficulty: Record<string, number>
  averageLatencyMs: number
  totalTokens: number
}
```

The `EvalRunner` constructor takes an `EvalConfig` and exposes a single `run()` method that returns `Promise<EvalRunResult>`. Think about these design decisions as you build it:

1. **Concurrency control**: How do you process test cases in batches of `config.concurrency`? A helper `chunk<T>(array: T[], size: number): T[][]` keeps the batching logic clean — split the array and use `Promise.all` per chunk.
2. **Retry logic**: For each test case, retry the `generateText` call up to `config.retries` times with exponential backoff. What happens if all retries fail? Should you record an error or throw?
3. **Running evaluators**: For each test case, loop through its `testCase.evaluators` array, look up the evaluator function in the config map, and collect results. How do you handle an evaluator that throws?
4. **Aggregate scoring**: The aggregate score is the mean of all evaluator scores. A test case passes if the aggregate is >= 0.7. Why 0.7 and not 0.5?
5. **Summary computation**: Group results by `testCase.category` and `testCase.difficulty`, sum scores within each group, and divide by count. What data structure makes this grouping straightforward?

### Configuring and Running Evaluators

Wire up the evaluators and test cases into a complete eval configuration. Create a `Map<EvaluatorType, EvaluatorFn>` and populate it:

- `'exact_match'` calls your `exactMatch` function with `tc.expectedOutput`
- `'contains'` calls your `containsMatch` function with `tc.expectedKeywords`
- `'semantic_similarity'` calls your `semanticSimilarity` function
- `'llm_judge'` calls your `singleCriterionJudge` and normalizes the 1-5 score to 0-1

Then define test cases across different categories (sentiment classification with exact match, summarization with contains + llm_judge, Q&A with semantic similarity). Create an `EvalConfig`, instantiate `EvalRunner`, and call `run()`. Print the summary: total tests, passed, failed, average score, and duration.

---

## Section 5: Automated Eval Suites

### Structuring Evals for CI

Evaluation suites should run automatically on every PR that changes prompts, model configurations, or retrieval logic.

```typescript
import { readFileSync, writeFileSync, existsSync } from 'fs'

interface EvalSuiteConfig {
  suites: {
    name: string
    testCasesPath: string
    baselineScorePath?: string
    minimumScore: number
    failOnRegression: boolean
    regressionThreshold: number
  }[]
}
```

Build a `runEvalSuite` function:

```typescript
async function runEvalSuite(suiteConfig: EvalSuiteConfig): Promise<{
  passed: boolean
  results: Map<string, EvalRunResult>
  regressions: string[]
}>
```

For each suite in the config, this function should: load test cases from the JSON file, create and run an `EvalRunner`, check the average score against `minimumScore`, and if `failOnRegression` is true, compare against the baseline score file to detect drops exceeding `regressionThreshold`. Save each run's results to `data/eval-results/` for future baselines.

What should happen when the baseline file does not exist yet? Should the first run always pass, or should it fail until a baseline is established? What are the trade-offs of each approach?

### CI Integration Script

Build a `scripts/run-evals.ts` that configures multiple suites (core-quality, rag-accuracy, safety-compliance) with different minimum scores and regression thresholds, runs them, prints a formatted report, and exits with code 0 on success or 1 on failure. What information should the report include to make eval failures actionable?

### GitHub Actions Workflow

```yaml
# .github/workflows/evals.yml
name: LLM Evals
on:
  pull_request:
    paths:
      - 'src/prompts/**'
      - 'src/rag/**'
      - 'src/agents/**'
      - 'data/evals/**'

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v1
      - run: bun install
      - run: bun run scripts/run-evals.ts
        env:
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
      - uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: data/eval-results/
```

> **Advanced Note:** For expensive eval suites, consider running a "fast" subset on every PR and the full suite nightly. Tag test cases with `fast` vs `full` and filter accordingly. This keeps PR feedback loops short while maintaining comprehensive coverage.

---

## Section 6: Regression Testing for Prompts

### The Prompt Regression Problem

Prompts are code. When you change a prompt, you can break things just as surely as changing a function. Prompt regression testing compares the behavior of a new prompt against the old one to catch degradation.

```typescript
interface PromptVersion {
  id: string
  timestamp: string
  systemPrompt: string
  description: string
  author: string
}

interface RegressionReport {
  oldVersion: string
  newVersion: string
  testCases: number
  improvements: RegressionDetail[]
  regressions: RegressionDetail[]
  unchanged: number
  verdict: 'safe' | 'risky' | 'blocked'
}

interface RegressionDetail {
  testCaseId: string
  input: string
  oldScore: number
  newScore: number
  scoreDelta: number
  oldOutput: string
  newOutput: string
}
```

Build a `promptRegressionTest` function:

```typescript
async function promptRegressionTest(
  oldPrompt: PromptVersion,
  newPrompt: PromptVersion,
  testCases: TestCase[],
  criteria: EvalCriterion[]
): Promise<RegressionReport>
```

For each test case, generate outputs with both prompt versions (use `Promise.all` for parallelism), then evaluate both with your `multiCriterionJudge`. Compute the score delta (new minus old). Classify each result as improvement (delta > 0.5), regression (delta < -0.5), or unchanged.

How should you determine the verdict? Consider what thresholds make sense: if zero test cases regress, the change is clearly safe. But what percentage of regressions should trigger "risky" vs. "blocked"? What factors beyond the percentage matter — for example, should the severity of regressions matter?

### Tracking Prompt Versions

Build a `PromptRegistry` class that versions prompts in a JSON file. It should support:

- `register(promptName, version)` — append a version to the history
- `getLatest(promptName)` — return the most recent version
- `getPrevious(promptName)` — return the second-most-recent version
- `getVersion(promptName, versionId)` — find a specific version by ID
- `listVersions(promptName)` — return the full history

Use `readFileSync`/`writeFileSync` for persistence. The internal data structure is `Map<string, PromptVersion[]>`, serialized as a JSON object with prompt names as keys and version arrays as values.

How does the registry enable a regression testing workflow? Think about how you would wire `getLatest` and `getPrevious` into `promptRegressionTest` to automatically compare the two most recent versions of a prompt whenever a new version is registered.

> **Beginner Note:** Treat prompts like source code. Version them, review changes, and test before deploying. A prompt registry makes this workflow systematic rather than ad-hoc.

---

## Section 7: Benchmark Design

### Building Representative Test Sets

Good benchmarks are representative, diverse, and include edge cases. They should cover the full distribution of inputs your system will encounter in production.

```typescript
interface BenchmarkSpec {
  name: string
  description: string
  categories: BenchmarkCategory[]
  totalCases: number
  createdAt: string
  version: string
}

interface BenchmarkCategory {
  name: string
  description: string
  proportion: number // What fraction of total test cases
  testCases: TestCase[]
}
```

Build a `createBenchmark` function:

```typescript
function createBenchmark(spec: {
  name: string
  description: string
  categories: {
    name: string
    description: string
    proportion: number
    generator: () => TestCase[]
  }[]
}): BenchmarkSpec
```

Map each category spec to a `BenchmarkCategory` by calling its generator. Compute `totalCases` as the sum across categories. Set `createdAt` to the current ISO timestamp and `version` to `'1.0'`.

Design a benchmark for a customer support chatbot with four categories: common questions (40%), edge cases (30%), adversarial inputs (15%), and multilingual queries (15%). What evaluators would you assign to each category? Why might adversarial cases need `llm_judge` only while common questions can use `contains` + `llm_judge`?

### Edge Case Generation

Use an LLM to help generate edge cases for your benchmarks.

```typescript
async function generateEdgeCases(taskDescription: string, existingCases: TestCase[], count: number): Promise<TestCase[]>
```

Build this function using `Output.object` with a schema that captures the input, description, difficulty, and why it is an edge case. The system prompt should instruct the model to focus on ambiguous inputs, boundary conditions, special characters, contradictory requests, and inputs requiring missing context. Pass the existing test case inputs so the model avoids duplicates.

How do you map the LLM's generated objects to your `TestCase` format? What default evaluator makes sense for edge cases that have no expected output?

> **Advanced Note:** Benchmark contamination is a real risk. If your test cases are too similar to common training data, models may perform artificially well. Include domain-specific, novel scenarios that the model is unlikely to have memorized.

---

## Section 8: Human Eval vs Auto Eval

### When to Use Each

Human evaluation and automated evaluation serve different purposes. Understanding when to use each is critical for building reliable eval systems.

```typescript
interface EvalStrategy {
  evalType: 'human' | 'auto' | 'hybrid'
  rationale: string
  methods: string[]
  cost: 'low' | 'medium' | 'high'
  turnaroundTime: string
}
```

Build a `recommendEvalStrategy` function:

```typescript
function recommendEvalStrategy(
  taskType: string,
  subjectivity: 'low' | 'medium' | 'high',
  volume: number,
  frequency: 'once' | 'weekly' | 'per-pr'
): EvalStrategy
```

The logic follows two clear rules: high subjectivity with low volume (< 50) calls for human eval; low subjectivity with per-PR frequency requires automation. Everything else is a hybrid approach. What methods would you recommend for each strategy? When does `exact-match` suffice versus when do you need `llm-as-judge` versus when only a human expert will do? What `turnaroundTime` is realistic for each approach?

### Building a Human Eval Interface

Design the data structures for a human evaluation workflow:

```typescript
interface HumanEvalTask {
  id: string
  input: string
  output: string
  context?: string
  criteria: {
    name: string
    description: string
    scale: { min: number; max: number; labels: Record<number, string> }
  }[]
}

interface HumanEvalResponse {
  taskId: string
  evaluatorId: string
  scores: Record<string, number>
  freeformFeedback: string
  timestamp: string
}
```

Build two functions:

1. `createHumanEvalBatch` — takes an array of `{ input, output }` pairs and a criteria definition, returns `HumanEvalTask[]` with sequential IDs (e.g., `'task-1'`, `'task-2'`). How do you generate the ID and attach the criteria to each task?

2. `computeInterAnnotatorAgreement` — takes responses from multiple annotators (each annotator's responses as a separate array), groups by task, and computes agreement per criterion. A simple agreement metric: for each task, if the max score difference among annotators is <= 1, count it as agreement. Return the agreement rate per criterion as a fraction.

Why does inter-annotator agreement matter? If two humans disagree on whether an answer is good, what does that tell you about using human eval as ground truth?

### Calibrating Auto-Eval Against Human Eval

Use human evaluation results to validate and calibrate your automated evaluators.

```typescript
async function calibrateAutoEval(
  humanResults: HumanEvalResponse[],
  autoEvaluator: (input: string, output: string) => Promise<number>,
  testData: { input: string; output: string }[]
): Promise<{
  correlation: number
  bias: number
  recommendations: string[]
}>
```

Build this function. For each test data item, compute the average human score (from `humanResults` matching that task) and the auto score. Then compute the **Pearson correlation** between the two score arrays:

```
correlation = sum((h - meanH) * (a - meanA)) / sqrt(sum((h - meanH)^2) * sum((a - meanA)^2))
```

Also compute bias as `meanAuto - meanHuman`. Generate recommendations based on these thresholds: correlation < 0.7 means the auto-eval does not capture what humans value; bias > 0.5 means the auto-eval is too generous; correlation >= 0.85 with bias < 0.3 means good calibration. What recommendation would you give for each case?

> **Beginner Note:** Inter-annotator agreement measures how much different human evaluators agree with each other. If humans disagree, expecting an automated evaluator to match human judgment is unrealistic. Always check agreement before using human eval as your ground truth.

---

## Section 9: Prompt Version Testing

### A/B Testing Prompts Systematically

When you have a candidate prompt improvement, you need to test it rigorously against the current version before deploying.

```typescript
interface ABTestConfig {
  name: string
  controlPrompt: PromptVersion
  treatmentPrompt: PromptVersion
  testCases: TestCase[]
  criteria: EvalCriterion[]
  significanceLevel: number // e.g., 0.05
  runsPerTestCase: number // Repeat each test case N times for statistical power
}

interface ABTestResult {
  config: Pick<ABTestConfig, 'name' | 'significanceLevel'>
  controlScores: number[]
  treatmentScores: number[]
  controlMean: number
  treatmentMean: number
  pValue: number
  significant: boolean
  effectSize: number
  recommendation: string
  details: {
    testCaseId: string
    controlScore: number
    treatmentScore: number
    delta: number
  }[]
}

async function runABTest(config: ABTestConfig): Promise<ABTestResult> {
  /* ... */
}
```

Build `runABTest`. For each test case, run both the control and treatment prompts `runsPerTestCase` times using `generateText` and score each with `multiCriterionJudge`. Average the scores per test case, collect them into `controlScores` and `treatmentScores` arrays, then compute overall means. Use `pairedTTest` for the p-value and `computeCohenD` for effect size. Generate a recommendation string: "DEPLOY" if significant and treatment is better, "REJECT" if significant and worse, "HOLD" if not significant.

### Statistical Helpers

```typescript
function pairedTTest(a: number[], b: number[]): number {
  /* ... */
}

function computeCohenD(a: number[], b: number[]): number {
  /* ... */
}
```

Build these two statistical functions:

- `pairedTTest` — computes a paired t-test p-value. Calculate the pairwise differences (`b[i] - a[i]`), their mean and variance, then the t-statistic as `meanDiff / standardError`. Convert to a p-value using a t-distribution approximation (e.g., the regularized incomplete beta function). Return 1.0 if standard error is zero.
- `computeCohenD` — computes Cohen's d effect size. Calculate means and variances for both arrays, compute the pooled standard deviation, then return `(meanB - meanA) / pooledStd`. Handle zero pooled std by returning 0.

These are pure math functions — no LLM calls. You can implement the t-distribution approximation using a series expansion of the incomplete beta function, or use a simpler normal approximation for large sample sizes.

### Running a Full A/B Test

```typescript
// Compare two versions of a customer support prompt
const abResult = await runABTest({
  name: 'Customer Support v2.1 vs v2.0',
  controlPrompt: {
    id: 'v2.0',
    timestamp: '2025-01-01T00:00:00Z',
    systemPrompt: 'You are a customer support agent. Answer questions helpfully and accurately.',
    description: 'Baseline prompt',
    author: 'team',
  },
  treatmentPrompt: {
    id: 'v2.1',
    timestamp: '2025-01-15T00:00:00Z',
    systemPrompt: `You are an empathetic customer support agent.

When responding:
1. Acknowledge the customer's feelings first
2. Provide a clear, accurate answer
3. Offer additional help proactively

Always maintain a warm, professional tone.`,
    description: 'Added empathy and structured response format',
    author: 'team',
  },
  testCases: yourTestCases, // Use your test dataset from the eval framework
  criteria: [
    {
      name: 'Helpfulness',
      description: "Does the response solve the customer's problem?",
      weight: 0.4,
    },
    {
      name: 'Empathy',
      description: "Does the response acknowledge the customer's feelings?",
      weight: 0.3,
    },
    {
      name: 'Professionalism',
      description: 'Is the tone appropriate and professional?',
      weight: 0.3,
    },
  ],
  significanceLevel: 0.05,
  runsPerTestCase: 3,
})

console.log(`\nA/B Test: ${abResult.config.name}`)
console.log(`Control mean: ${abResult.controlMean.toFixed(3)}`)
console.log(`Treatment mean: ${abResult.treatmentMean.toFixed(3)}`)
console.log(`p-value: ${abResult.pValue.toFixed(4)}`)
console.log(`Effect size (Cohen's d): ${abResult.effectSize.toFixed(2)}`)
console.log(`Recommendation: ${abResult.recommendation}`)
```

> **Advanced Note:** Cohen's d values of 0.2, 0.5, and 0.8 are conventionally considered small, medium, and large effects. In LLM evaluation, even small effects (d = 0.2) can be meaningful at scale. Consider the business impact alongside statistical significance.

> **Production Note:** In production, eval suites become **deployment gates** — CI/CD steps that block merging if quality metrics drop below a threshold. The eval code you write in this module is exactly what runs in that gate. Maintain a golden dataset (50-100 curated examples), run your suite on every PR, and reject changes that regress accuracy. Tools like Promptfoo and Braintrust provide frameworks for this.

> **Local Alternative (Ollama):** Eval frameworks are model-agnostic — exact match, fuzzy match, and semantic similarity scorers work with any provider. For LLM-as-judge evaluation, `ollama('qwen3.5')` can serve as the judge model, though larger models produce more reliable judgments. Running evals locally means zero API cost for iterating on your test suite.

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 10: Cost as Evaluation Dimension

### Evaluating More Than Quality

A response that costs 10x more for 5% better quality may fail the eval. Production evaluation systems track cost alongside quality:

- **Input tokens** — How many tokens the prompt consumed
- **Output tokens** — How many tokens the response generated
- **Cost per request** — Calculated from the provider's pricing
- **Aggregate cost per session** — Total cost across multi-turn interactions

Cost becomes an evaluation dimension just like accuracy or helpfulness. A test case can pass on quality but fail on cost if the response exceeded a budget threshold.

```typescript
interface EvalResult {
  qualityScore: number
  costUsd: number
  latencyMs: number
  passesAll: boolean // true only if ALL dimensions pass
}
```

When comparing two prompt versions, the cheaper one wins if quality is equivalent. This prevents prompt bloat — the tendency to add more instructions that improve quality marginally but increase cost substantially.

---

## Section 11: Diagnostic Capture

### Debugging Eval Failures

When an eval fails, you need more than the score — you need the full context that led to the failure. Diagnostic capture records:

- The complete prompt (system + user messages)
- The model response (full text)
- All tool calls and their results
- Token usage (input, output, total)
- The evaluation scores per dimension

This context is written to a diagnostic report that makes failures reproducible. Instead of re-running the eval and hoping it fails the same way, you have a snapshot of exactly what happened.

The pattern is evaluation-as-debugging: apply eval techniques not just to measure quality, but to understand failures in production. When a user reports a bad response, the diagnostic capture lets you replay the exact conditions.

---

## Section 12: Feature Flag A/B Testing

### Practical A/B Testing with Feature Flags

The A/B testing in Section 9 compares prompt versions manually. Feature flags make this operational — different users get different prompt versions automatically, and metrics are collected per variant.

A simple in-process feature flag system assigns each request to a variant:

```typescript
function getVariant(userId: string, experimentId: string): 'control' | 'treatment' {
  /* ... */
}
```

The core logic: hash the concatenation of `userId` and `experimentId` into a number (a simple hash function that sums character codes works here), then use modulo to assign to a variant. This ensures each user consistently sees the same variant for a given experiment.

Each variant maps to a different prompt version. Evaluation metrics are tagged with the variant, enabling comparison after enough requests have been served. The flag system handles rollout control (percentage-based), so you can start at 5% treatment and increase as confidence grows.

> **Key Insight:** Feature flags decouple deployment from release. You deploy both prompt versions simultaneously, but the flag controls which one a given user sees. This eliminates the need to coordinate deploys with A/B test schedules.

---

## Section 13: LSP Diagnostics as Eval Signal

(See Module 10 Section 11 for LSP background.)

### Compilers and Linters as Evaluators

For generated code, deterministic tools — type checkers, linters, compilers — are better evaluators than LLM judges for syntactic and type correctness. They are faster, cheaper, and never hallucinate.

The pattern layers evaluations by cost and reliability:

1. **Deterministic checks first** — Run `tsc --noEmit` or ESLint. If the code has type errors or lint violations, it fails immediately.
2. **Heuristic checks second** — Run the test suite. If tests fail, the code is functionally incorrect.
3. **LLM-as-judge last** — Only for subjective quality (readability, idiomatic style, design quality). This is the most expensive check and should only run on code that already passes the cheaper gates.

```typescript
// Eval pipeline: cheapest checks first
const typeCheck = await run('tsc --noEmit')
if (typeCheck.exitCode !== 0) return { pass: false, reason: 'type-error', gate: 'compiler' }

const tests = await run('bun test')
if (tests.exitCode !== 0) return { pass: false, reason: 'test-failure', gate: 'tests' }

// Only invoke LLM judge if code compiles and tests pass
const judgeScore = await llmJudge(code)
```

This ordering saves significant cost — most bad code is caught by the compiler before the LLM judge is ever called.

---

## Summary

In this module, you learned:

1. **Why evals matter:** LLM applications are non-deterministic, making traditional unit testing insufficient — you need evaluation frameworks that measure quality across distributions of outputs.
2. **Evaluation types:** Exact match, fuzzy match, contains match, and semantic similarity each serve different use cases, from structured data validation to open-ended text assessment.
3. **LLM-as-judge:** Using one model to evaluate another model's output enables scalable assessment of qualities like helpfulness, accuracy, and tone that are hard to measure programmatically.
4. **Eval framework:** How to build a reusable evaluation runner with typed test cases, multiple scoring functions, and aggregate reporting for systematic quality measurement.
5. **Automated eval suites:** Structuring evaluations to run in CI pipelines so that every prompt change is automatically tested against regression benchmarks.
6. **Regression testing:** Designing test suites that detect when prompt or model changes break existing behavior, using baseline comparisons and threshold-based pass/fail criteria.
7. **A/B testing:** Systematically comparing prompt versions with statistical rigor, using effect size and confidence intervals to make data-driven improvement decisions.
8. **Human vs automated evaluation:** Understanding when automated metrics are sufficient and when human judgment is needed, and how to combine both approaches effectively.
9. **Human eval calibration:** Using inter-annotator agreement to measure human consistency, and correlating automated evaluator scores against human judgments to validate and calibrate your auto-eval pipeline.
10. **Cost as evaluation dimension:** Tracking cost alongside quality so a response that costs 10x more for marginal improvement can fail the eval, preventing prompt bloat.
11. **Diagnostic capture:** Recording the full context (prompt, response, tool calls, scores) for every eval failure to make failures reproducible without re-running.
12. **Feature flag A/B testing:** Using feature flags to assign users to prompt variants automatically, decoupling deployment from release and enabling percentage-based rollouts.
13. **LSP diagnostics as eval signal:** Layering deterministic checks (compiler, linter, tests) before expensive LLM-as-judge calls to catch most bad code cheaply.

In Module 20, you will learn when and how to fine-tune models to internalize behavior that prompt engineering alone cannot reliably achieve.

---

## Quiz

**Question 1:** Why is exact match evaluation insufficient for most LLM applications?

A) It is too slow to compute
B) LLMs produce varied valid outputs for the same input
C) It requires too much memory
D) It only works with numbers

**Answer: B** — LLM outputs are non-deterministic and there are often multiple valid ways to answer a question. "The capital of France is Paris" and "Paris is France's capital city" are both correct, but exact match would reject one of them. This is why more flexible evaluation methods like semantic similarity and LLM-as-judge are necessary for open-ended tasks.

---

**Question 2:** What is position bias in LLM-as-judge evaluation?

A) The judge model is biased toward certain topics
B) The judge tends to prefer whichever output is presented first (or last)
C) The judge scores longer outputs higher
D) The judge always gives the same score

**Answer: B** — Position bias means the judge model systematically favors the output presented in a certain position (often the first). This is mitigated by randomizing the order of outputs in pairwise comparisons and running each comparison twice with swapped positions, then averaging the results.

---

**Question 3:** When should you prefer human evaluation over automated evaluation?

A) When evaluating thousands of test cases per day
B) When the quality criteria are highly subjective and sample sizes are small
C) When running evals on every pull request
D) When evaluating classification accuracy

**Answer: B** — Human evaluation is best suited for subjective quality assessments where automated metrics struggle, particularly when sample sizes are manageable (dozens to low hundreds). For high-volume or high-frequency evaluation, automated methods are more practical, with periodic human review to calibrate the auto-evaluators.

---

**Question 4:** What is the purpose of a prompt registry in regression testing?

A) To store API keys securely
B) To version control prompts and enable comparison between versions
C) To limit the number of prompts used
D) To generate new prompts automatically

**Answer: B** — A prompt registry maintains a versioned history of all prompts, enabling you to compare a new prompt version against its predecessor. This makes regression testing systematic — you can always retrieve the previous version, run both against the same test cases, and determine whether the new version is better, worse, or equivalent.

---

**Question 5:** What does a p-value of 0.03 mean in an A/B test of two prompt versions?

A) The treatment prompt is 3% better
B) There is a 3% chance the observed difference is due to random chance
C) 3% of test cases showed improvement
D) The test has 3% statistical power

**Answer: B** — A p-value of 0.03 means there is a 3% probability of observing a difference this large (or larger) if the two prompts actually performed identically. Since this is below the conventional 0.05 threshold, we consider the difference statistically significant. However, statistical significance does not tell you the magnitude of the difference — that is what effect size (Cohen's d) measures.

---

**Question 6 (Medium):** Why should production evaluation systems track cost alongside quality scores?

A) Cost tracking is required by LLM provider terms of service
B) A response that costs 10x more for marginal quality improvement may fail the eval, preventing prompt bloat
C) Cost and quality are always inversely correlated
D) Cost tracking makes evals run faster

**Answer: B** — Without cost as an evaluation dimension, there is a natural tendency toward prompt bloat — adding more instructions that improve quality marginally but increase cost substantially. By including cost thresholds in the eval, a test case can pass on quality but fail on cost. When comparing two prompt versions with equivalent quality, the cheaper one wins. This keeps prompts lean and cost-efficient.

---

**Question 7 (Hard):** A code generation eval pipeline runs three evaluation layers: compiler type-checking, test suite execution, and LLM-as-judge for style. Why is this ordering important?

A) The LLM-as-judge must run first to provide context for the other checks
B) The ordering does not matter because all three checks are independent
C) Cheapest checks run first so most bad code is caught before invoking the expensive LLM judge
D) The compiler must run last because it is the most thorough check

**Answer: C** — Running deterministic, cheap checks (compiler, tests) before the expensive LLM-as-judge call saves significant cost. Most bad code has type errors or fails tests, and these are caught by the compiler and test suite without paying for an LLM call. The LLM judge only evaluates code that already compiles and passes tests, which is a small fraction of all generated code. This layered approach can reduce eval costs by an order of magnitude.

---

## Exercises

### Exercise 1: Build an Eval Framework with LLM-as-Judge

Build a complete evaluation framework that evaluates a question-answering system using multiple evaluation methods including LLM-as-judge.

**Specification:**

1. Create a `QAEvalFramework` class that:
   - Accepts a set of test cases with questions, reference answers, and categories
   - Runs each test case through the LLM to get an output
   - Evaluates each output using three methods: `contains_match`, `semantic_similarity`, and `llm_judge`
   - The LLM judge should evaluate on three criteria: accuracy, completeness, and clarity
   - Produces a JSON report with per-test-case and aggregate scores

2. Create at least 15 test cases across 3 categories:
   - Factual questions (5 cases) — use `exact_match` + `contains`
   - Explanation questions (5 cases) — use `semantic_similarity` + `llm_judge`
   - Opinion/analysis questions (5 cases) — use `llm_judge` only

3. Run the framework and analyze the results:
   - Which evaluation method correlates best with your own judgment?
   - Where do different evaluation methods disagree?

**Expected output:** A JSON report with scores, a summary table, and analysis of evaluator agreement.

### Exercise 2: Regression Test a RAG Pipeline

Build a regression testing system for a RAG-powered question-answering pipeline.

**Specification:**

1. Simulate a RAG pipeline that takes a question, retrieves relevant context, and generates an answer. Use a mock retriever that returns predefined context chunks.

2. Create two prompt versions:
   - **v1 (control):** A basic prompt that instructs the model to answer based on the provided context
   - **v2 (treatment):** An improved prompt that adds instructions for citing sources, acknowledging uncertainty, and structuring the answer

3. Build a regression testing system that:
   - Runs both prompts against 10+ test cases
   - Uses pairwise LLM-as-judge comparison (with position randomization)
   - Computes aggregate scores and statistical significance
   - Generates a regression report with verdict (safe/risky/blocked)

4. The report should include:
   - Per-test-case comparison with scores for both versions
   - Categories where v2 improves vs regresses
   - Overall recommendation with confidence level

**Expected output:** A regression report JSON file and a human-readable summary printed to the console.

### Exercise 3: Multi-Dimensional Eval

**Objective:** Build an evaluation that scores on multiple dimensions (quality, cost, latency, safety) and requires a response to pass ALL dimensions to succeed.

**Specification:**

1. Create a file `src/exercises/m19/ex03-multi-dim-eval.ts`
2. Export an async function `multiDimEval(testCases: MultiDimTestCase[], options?: MultiDimOptions): Promise<MultiDimReport>`
3. Define the types:

```typescript
interface MultiDimTestCase {
  input: string
  reference: string
  maxCostUsd?: number // default: 0.01
  maxLatencyMs?: number // default: 5000
}

interface DimensionResult {
  dimension: string
  score: number
  threshold: number
  passed: boolean
}

interface MultiDimCaseResult {
  input: string
  dimensions: DimensionResult[]
  passedAll: boolean
}

interface MultiDimReport {
  results: MultiDimCaseResult[]
  overallPassRate: number
  worstDimension: string // the dimension that failed most often
}
```

4. The eval must:
   - Score each test case on at least four dimensions: quality (LLM-as-judge or semantic similarity), cost (from token usage), latency (wall clock time), and safety (check for refusals or harmful content)
   - A test case passes only if ALL dimensions pass their thresholds
   - Report the overall pass rate and identify the worst-performing dimension

**Test specification:**

```typescript
// tests/exercises/m19/ex03-multi-dim-eval.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 19: Multi-Dimensional Eval', () => {
  it('should evaluate on multiple dimensions', async () => {
    const report = await multiDimEval([{ input: 'What is 2+2?', reference: '4' }])
    expect(report.results[0].dimensions.length).toBeGreaterThanOrEqual(4)
  })

  it('should fail if any dimension fails', async () => {
    const report = await multiDimEval([{ input: 'Write a novel', reference: 'A long story...', maxLatencyMs: 1 }])
    expect(report.results[0].passedAll).toBe(false)
  })

  it('should identify the worst dimension', async () => {
    const report = await multiDimEval([{ input: 'What is TypeScript?', reference: 'A typed superset of JavaScript' }])
    expect(report.worstDimension).toBeTruthy()
  })
})
```

### Exercise 4: Compiler-as-Eval

**Objective:** Build an eval pipeline that uses `tsc --noEmit` as the first evaluation gate for generated TypeScript code, only proceeding to LLM-as-judge if the code compiles.

**Specification:**

1. Create a file `src/exercises/m19/ex04-compiler-eval.ts`
2. Export an async function `compilerEval(codeSnippets: CodeSnippet[]): Promise<CompilerEvalReport>`
3. Define the types:

```typescript
interface CodeSnippet {
  description: string
  code: string
}

interface CompilerEvalResult {
  description: string
  compilerPass: boolean
  compilerErrors?: string[]
  judgeScore?: number // only present if compiler passed
  judgeReasoning?: string
  gate: 'compiler' | 'judge' | 'passed'
}

interface CompilerEvalReport {
  results: CompilerEvalResult[]
  caughtByCompiler: number
  reachedJudge: number
  passedAll: number
  costSaved: number // estimated cost saved by compiler catching errors early
}
```

4. The pipeline must:
   - Write each code snippet to a temporary file and run `tsc --noEmit` on it
   - If the compiler reports errors, mark the snippet as failed at the `compiler` gate without invoking the LLM
   - If the compiler passes, run an LLM-as-judge evaluation for code quality
   - Track how many snippets were caught by the compiler vs. reaching the LLM judge
   - Estimate cost saved by not invoking the LLM for compiler-rejected code

**Test specification:**

```typescript
// tests/exercises/m19/ex04-compiler-eval.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 19: Compiler-as-Eval', () => {
  it('should catch type errors without invoking LLM judge', async () => {
    const report = await compilerEval([{ description: 'bad types', code: 'const x: number = "hello"' }])
    expect(report.results[0].compilerPass).toBe(false)
    expect(report.results[0].gate).toBe('compiler')
    expect(report.results[0].judgeScore).toBeUndefined()
    expect(report.caughtByCompiler).toBe(1)
  })

  it('should invoke LLM judge for valid code', async () => {
    const report = await compilerEval([
      { description: 'valid code', code: 'const add = (a: number, b: number): number => a + b' },
    ])
    expect(report.results[0].compilerPass).toBe(true)
    expect(report.results[0].judgeScore).toBeDefined()
    expect(report.reachedJudge).toBe(1)
  })

  it('should report cost savings', async () => {
    const report = await compilerEval([
      { description: 'bad', code: 'const x: number = "hello"' },
      { description: 'good', code: 'const x: number = 42' },
    ])
    expect(report.costSaved).toBeGreaterThanOrEqual(0)
  })
})
```
