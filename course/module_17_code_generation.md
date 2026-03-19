# Module 17: Code Generation

## Learning Objectives

- Understand the capabilities and limitations of LLMs as code generators
- Write effective prompts that produce correct, well-structured code
- Extract code blocks from LLM responses using structured output
- Run generated code safely in sandboxed environments
- Implement test-driven code generation where the LLM writes code to pass tests
- Build iterative refinement loops that fix errors automatically
- Use LLMs for code review with actionable feedback
- Apply security best practices when executing LLM-generated code

---

## Why Should I Care?

Code generation is one of the most impactful applications of LLMs. From writing utility functions to scaffolding entire modules, LLMs can dramatically accelerate development — but only if you know how to use them effectively. Raw code generation is unreliable: the model might produce code that looks correct but fails on edge cases, uses deprecated APIs, or contains security vulnerabilities.

This module teaches you how to build systems that generate code reliably by combining LLM generation with automated testing, iterative refinement, and safety controls. The patterns here are used in production coding assistants, automated refactoring tools, and test generation systems.

If you treat code generation as a single `generateText` call, you will be disappointed. If you treat it as an iterative loop with tests, feedback, and validation, you will be impressed.

---

## Connection to Other Modules

- **Module 14 (Agent Fundamentals)** provides the agent loop pattern used for iterative code refinement.
- **Module 16 (Workflows & Chains)** provides chain patterns for generate-test-fix pipelines.
- **Module 3 (Structured Output)** is used to extract code blocks and metadata from LLM responses.
- **Module 18 (Human-in-the-Loop)** adds human review before executing generated code.

---

## Section 1: LLMs as Code Generators

### Capabilities

LLMs can generate code across dozens of languages with impressive fluency. They excel at:

- **Boilerplate**: Generating standard patterns, CRUD operations, type definitions
- **Translations**: Converting code between languages (Python to TypeScript, SQL to ORM)
- **Completions**: Finishing partially written functions based on context
- **Explanations**: Writing code with detailed comments explaining each step
- **Common patterns**: Implementing well-known algorithms and design patterns

### Limitations

LLMs also have systematic weaknesses:

- **Hallucinated APIs**: The model may call functions that do not exist or use wrong parameter names
- **Version confusion**: The model may use deprecated APIs from older library versions
- **Logic errors**: Code may pass basic tests but fail on edge cases
- **Security blind spots**: Generated code may have injection vulnerabilities, unsafe deserialization, or missing input validation
- **Context limits**: Large codebases cannot fit in the context window, leading to inconsistent imports or styles

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Demonstrate the difference between good and bad code generation prompts

// BAD: Vague prompt that produces unreliable code
const badResult = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Write a function to process data',
})
// This might return anything — Python? TypeScript? What kind of data?

// GOOD: Specific prompt with constraints
const goodResult = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: `Write a TypeScript function that:
- Name: parseCSVLine
- Input: a single CSV line as a string
- Output: an array of strings (the fields)
- Must handle quoted fields (fields containing commas wrapped in double quotes)
- Must handle escaped quotes (double-double quotes inside quoted fields)
- Do NOT use external libraries
- Include JSDoc comment with examples
- Include edge case handling for empty input

Return ONLY the function — no explanation, no tests.`,
})

console.log(goodResult.text)
```

> **Beginner Note:** LLMs generate code by pattern matching against their training data. They are very good at writing code that looks like code they have seen before. They are less reliable when the task requires novel logic or deep reasoning about edge cases.

> **Advanced Note:** The quality of generated code varies significantly by model. Claude models tend to produce well-structured TypeScript with good error handling. For critical code, always pair generation with automated testing — never trust generated code without verification.

---

## Section 2: Prompting for Code

### Effective Code Prompts

The quality of generated code depends heavily on the prompt. Include these elements:

1. **Language and runtime**: "TypeScript for Bun" or "Python 3.12"
2. **Function signature**: Name, parameters with types, return type
3. **Behavior specification**: What the function does, step by step
4. **Edge cases**: What should happen with empty input, null values, large datasets
5. **Constraints**: No external libraries, must be pure, must handle errors
6. **Examples**: Input/output pairs that define expected behavior

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

// Template for code generation prompts
function codePrompt(spec: {
  language: string
  functionName: string
  description: string
  parameters: Array<{ name: string; type: string; description: string }>
  returnType: string
  examples: Array<{ input: string; output: string }>
  constraints: string[]
  edgeCases: string[]
}): string {
  const params = spec.parameters.map(p => `  - ${p.name}: ${p.type} — ${p.description}`).join('\n')

  const examples = spec.examples.map(e => `  ${spec.functionName}(${e.input}) => ${e.output}`).join('\n')

  const constraints = spec.constraints.map(c => `- ${c}`).join('\n')

  const edgeCases = spec.edgeCases.map(e => `- ${e}`).join('\n')

  return `Write a ${spec.language} function with the following specification:

Function: ${spec.functionName}
Description: ${spec.description}

Parameters:
${params}

Return type: ${spec.returnType}

Examples:
${examples}

Constraints:
${constraints}

Edge cases to handle:
${edgeCases}

Return ONLY the function code with a JSDoc comment. No tests, no explanation, no markdown fences.`
}

// Usage
const prompt = codePrompt({
  language: 'TypeScript',
  functionName: 'deepMerge',
  description:
    'Recursively merge two objects. Arrays are concatenated. Nested objects are merged recursively. Primitive values from the second object overwrite the first.',
  parameters: [
    { name: 'target', type: 'Record<string, unknown>', description: 'The base object' },
    {
      name: 'source',
      type: 'Record<string, unknown>',
      description: 'The object to merge into target',
    },
  ],
  returnType: 'Record<string, unknown>',
  examples: [
    {
      input: '{ a: 1 }, { b: 2 }',
      output: '{ a: 1, b: 2 }',
    },
    {
      input: '{ a: { x: 1 } }, { a: { y: 2 } }',
      output: '{ a: { x: 1, y: 2 } }',
    },
    {
      input: '{ a: [1, 2] }, { a: [3, 4] }',
      output: '{ a: [1, 2, 3, 4] }',
    },
  ],
  constraints: [
    'Do NOT mutate the input objects',
    'Do NOT use any external libraries',
    'Must handle nested objects to arbitrary depth',
    'Must be type-safe with TypeScript generics',
  ],
  edgeCases: [
    'Empty objects: deepMerge({}, {}) returns {}',
    'Null values: null overwrites existing values',
    'Undefined values: undefined is ignored (does not overwrite)',
    'Mixed types: if target has an array and source has a primitive for the same key, source wins',
  ],
})

const result = await generateText({
  model: mistral('mistral-small-latest'),
  prompt,
})

console.log(result.text)
```

### System Prompts for Code Generation

Use system prompts to set global coding standards:

```typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

const codeGenSystemPrompt = `You are an expert TypeScript developer. When generating code:

1. STYLE:
   - Use modern TypeScript (ES2022+)
   - Prefer const over let, never use var
   - Use arrow functions for callbacks
   - Use template literals for string interpolation
   - Prefer early returns over deep nesting

2. TYPES:
   - Always include explicit type annotations for function parameters and return types
   - Use interfaces for object shapes, types for unions/intersections
   - Avoid \`any\` — use \`unknown\` when the type is truly unknown
   - Use generics when the function works with multiple types

3. ERROR HANDLING:
   - Validate all inputs at function boundaries
   - Use custom error classes for domain-specific errors
   - Never swallow errors silently — always log or rethrow

4. DOCUMENTATION:
   - Include JSDoc comments for all exported functions
   - Document parameters, return values, and thrown errors
   - Include usage examples in JSDoc @example tags

5. OUTPUT FORMAT:
   - Return ONLY the requested code
   - Do NOT include markdown code fences unless asked
   - Do NOT include test code unless asked
   - Do NOT include explanation text unless asked`

const result = await generateText({
  model: mistral('mistral-small-latest'),
  system: codeGenSystemPrompt,
  prompt: `Write a function called "retry" that:
- Takes an async function and retry options (maxRetries, delayMs, backoffMultiplier)
- Retries the function on failure with exponential backoff
- Returns the result on success
- Throws the last error after all retries are exhausted
- Supports an optional AbortSignal for cancellation`,
})

console.log(result.text)
```

> **Beginner Note:** Specific prompts produce specific code. If you tell the model "write a sorting function," you will get something generic. If you specify the language, algorithm, input type, edge cases, and constraints, you will get exactly what you need.

---

## Section 3: Structured Code Output

### Extracting Code from LLM Responses

LLMs often wrap code in markdown fences or add explanations. Use structured output to extract clean code:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const codeOutputSchema = z.object({
  code: z.string().describe('The complete, executable code'),
  language: z.string().describe('The programming language'),
  dependencies: z.array(z.string()).describe('Required npm packages (empty if none)'),
  exports: z
    .array(
      z.object({
        name: z.string(),
        type: z.enum(['function', 'class', 'constant', 'type', 'interface']),
        description: z.string(),
      })
    )
    .describe('Exported symbols'),
  testHints: z.array(z.string()).describe('Suggested test cases for this code'),
})

type CodeOutput = z.infer<typeof codeOutputSchema>

async function generateCode(spec: string): Promise<CodeOutput> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: codeOutputSchema }),
    system: `You are a TypeScript code generator. Generate clean, well-typed code.
The code field must contain complete, executable TypeScript — no markdown fences, no explanations.`,
    prompt: spec,
  })

  return output!
}

// Usage
const output = await generateCode(`
Write a TypeScript module that provides a simple in-memory cache with:
- get(key): Get a value by key
- set(key, value, ttlMs): Set a value with a time-to-live in milliseconds
- delete(key): Remove a key
- clear(): Remove all keys
- size(): Return the number of non-expired entries

The cache should automatically expire entries when their TTL has passed.
Use a Map internally. No external dependencies.
`)

console.log('Language:', output.language)
console.log('Dependencies:', output.dependencies)
console.log('Exports:', output.exports)
console.log('Test hints:', output.testHints)
console.log('\nCode:\n', output.code)
```

### Multi-File Code Generation

For larger features, generate multiple files:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const multiFileSchema = z.object({
  files: z.array(
    z.object({
      path: z.string().describe('Relative file path, e.g., src/cache.ts'),
      content: z.string().describe('Complete file content'),
      purpose: z.string().describe('What this file does'),
    })
  ),
  entryPoint: z.string().describe('Main file to run'),
  installCommand: z.string().optional().describe('Command to install dependencies, if any'),
})

async function generateMultiFile(spec: string) {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: multiFileSchema }),
    system: `Generate a TypeScript project with multiple files.
Each file should be complete and importable.
Use relative imports between files.
Follow standard project structure (src/ for code, types in separate files).`,
    prompt: spec,
  })

  return output!
}

const project = await generateMultiFile(`
Create a simple HTTP rate limiter module with:
1. A RateLimiter class (src/rate-limiter.ts) that implements token bucket algorithm
2. Type definitions (src/types.ts) for configuration and results
3. A middleware function (src/middleware.ts) that wraps a request handler
4. An example usage file (src/example.ts)

No external dependencies. Use Bun's built-in HTTP server for the example.
`)

for (const file of project.files) {
  console.log(`\n=== ${file.path} (${file.purpose}) ===`)
  console.log(file.content)
}
console.log(`\nEntry point: ${project.entryPoint}`)
```

> **Advanced Note:** Multi-file generation is less reliable than single-function generation because the model must maintain consistency across files (imports, types, naming). Always validate that generated imports resolve correctly and types match across files.

---

## Section 4: Isolated Subprocess Execution

> **Security Warning:** This subprocess approach provides timeout protection and output capture, but NOT security isolation. Generated code runs with the same OS permissions as your application. For true sandboxing in production, use container-based solutions (Docker, Firecracker) or WebAssembly runtimes.

### Running Generated Code Safely

Never run LLM-generated code directly in your main process. Always isolate it in a subprocess:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { spawn } from 'child_process'
import { writeFile, mkdir, rm } from 'fs/promises'
import { join } from 'path'
import { randomUUID } from 'crypto'

interface SandboxResult {
  stdout: string
  stderr: string
  exitCode: number
  durationMs: number
  timedOut: boolean
}

async function runInSandbox(
  code: string,
  options: {
    timeoutMs?: number
    workDir?: string
  } = {}
): Promise<SandboxResult> {
  const timeoutMs = options.timeoutMs ?? 10000
  const sandboxId = randomUUID().slice(0, 8)
  const workDir = options.workDir ?? join('/tmp', `sandbox-${sandboxId}`)

  try {
    // Create temporary directory
    await mkdir(workDir, { recursive: true })

    // Write the code to a file
    const filePath = join(workDir, 'generated.ts')
    await writeFile(filePath, code)

    // Run with Bun in a subprocess
    const startTime = Date.now()

    return await new Promise<SandboxResult>(resolve => {
      const child = spawn('bun', ['run', filePath], {
        cwd: workDir,
        timeout: timeoutMs,
        env: {
          ...process.env,
          // Remove sensitive environment variables
          MISTRAL_API_KEY: '',
          GROQ_API_KEY: '',
          ANTHROPIC_API_KEY: '',
          OPENAI_API_KEY: '',
          DATABASE_URL: '',
          HOME: workDir, // Prevent access to home directory
        },
      })

      let stdout = ''
      let stderr = ''
      let timedOut = false

      child.stdout.on('data', data => {
        stdout += data.toString()
      })

      child.stderr.on('data', data => {
        stderr += data.toString()
      })

      child.on('close', code => {
        resolve({
          stdout: stdout.trim(),
          stderr: stderr.trim(),
          exitCode: code ?? 1,
          durationMs: Date.now() - startTime,
          timedOut,
        })
      })

      child.on('error', error => {
        if (error.message.includes('ETIMEDOUT') || error.message.includes('killed')) {
          timedOut = true
        }
        resolve({
          stdout: stdout.trim(),
          stderr: error.message,
          exitCode: 1,
          durationMs: Date.now() - startTime,
          timedOut,
        })
      })
    })
  } finally {
    // Clean up
    try {
      await rm(workDir, { recursive: true, force: true })
    } catch {
      // Ignore cleanup errors
    }
  }
}

// Usage: Generate code and run it safely
const { output: generated } = await generateText({
  model: mistral('mistral-small-latest'),
  output: Output.object({
    schema: z.object({
      code: z.string(),
    }),
  }),
  prompt: `Write a TypeScript script that:
1. Generates the first 20 Fibonacci numbers
2. Prints each on its own line in the format "F(n) = value"
3. Calculates and prints the golden ratio approximation from the last two numbers

The script should be self-contained and print to console.`,
})

console.log('Generated code:')
console.log(generated!.code)

console.log('\n--- Running in sandbox ---')
const result = await runInSandbox(generated!.code, { timeoutMs: 5000 })

console.log(`Exit code: ${result.exitCode}`)
console.log(`Duration: ${result.durationMs}ms`)
console.log(`Timed out: ${result.timedOut}`)
console.log(`stdout:\n${result.stdout}`)
if (result.stderr) {
  console.log(`stderr:\n${result.stderr}`)
}
```

### Sandbox Security Checklist

When running generated code, always apply these protections:

```typescript
// Security practices for running generated code

// 1. ALWAYS run in a subprocess with limited permissions — never in the main process

// 2. Strip sensitive environment variables
const safeEnv: Record<string, string> = {
  PATH: process.env.PATH ?? '',
  NODE_ENV: 'sandbox',
  // Explicitly do NOT include API keys, database URLs, etc.
}

// 3. Set resource limits
const resourceLimits = {
  timeoutMs: 10_000, // Kill after 10 seconds
  maxOutputBytes: 1_000_000, // 1MB max output
}

// 4. Use a temporary directory that gets cleaned up
// const workDir = join('/tmp', `sandbox-${uuid}`);

// 5. NEVER let generated code access the network in production
// Use a container or VM for true network isolation

// 6. Validate output before using it
function validateOutput(output: string): boolean {
  // Check for suspicious patterns
  if (output.length > resourceLimits.maxOutputBytes) return false
  // Add domain-specific validation
  return true
}
```

> **Beginner Note:** Think of sandboxing like putting an unknown substance in a sealed container before testing it. You do not know what the generated code will do, so you run it in an isolated environment where it cannot damage your system or access sensitive data.

---

## Section 5: Test-Driven Generation

### Generate Code to Pass Tests

The most reliable code generation pattern: write the tests first, then have the LLM generate code that passes them.

````typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import { writeFile, mkdir, rm } from 'fs/promises'
import { join } from 'path'
import { randomUUID } from 'crypto'
import { spawn } from 'child_process'

interface TestSpec {
  functionName: string
  testCode: string
  description: string
}

interface GenerationResult {
  code: string
  testsPass: boolean
  testOutput: string
  attempts: number
}

async function runTests(
  implCode: string,
  testCode: string,
  workDir: string
): Promise<{ pass: boolean; output: string }> {
  // Write the implementation
  await writeFile(join(workDir, 'impl.ts'), implCode)

  // Write the test that imports the implementation
  const fullTestCode = `import { describe, it, expect } from "bun:test";
// Import all exports from the implementation
const impl = await import("./impl.ts");

// Make exports available as globals for the test
for (const [key, value] of Object.entries(impl)) {
  (globalThis as any)[key] = value;
}

${testCode}`

  await writeFile(join(workDir, 'test.test.ts'), fullTestCode)

  return new Promise(resolve => {
    const child = spawn('bun', ['test', 'test.test.ts'], {
      cwd: workDir,
      timeout: 15000,
      env: { ...process.env, MISTRAL_API_KEY: '', ANTHROPIC_API_KEY: '' },
    })

    let output = ''
    child.stdout.on('data', d => (output += d.toString()))
    child.stderr.on('data', d => (output += d.toString()))

    child.on('close', code => {
      resolve({
        pass: code === 0,
        output: output.trim(),
      })
    })

    child.on('error', err => {
      resolve({ pass: false, output: err.message })
    })
  })
}

async function generateToPassTests(spec: TestSpec, maxAttempts: number = 3): Promise<GenerationResult> {
  const workDir = join('/tmp', `tdd-${randomUUID().slice(0, 8)}`)
  await mkdir(workDir, { recursive: true })

  let lastCode = ''
  let lastOutput = ''

  try {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      console.log(`\n[TDD] Attempt ${attempt}/${maxAttempts}`)

      // Generate (or fix) the code
      const prompt =
        attempt === 1
          ? `Write a TypeScript implementation for the following:

Function: ${spec.functionName}
Description: ${spec.description}

The implementation must pass these tests:
\`\`\`typescript
${spec.testCode}
\`\`\`

Export the function. Return ONLY the implementation code, no test code.`
          : `The previous implementation failed the tests.

Previous code:
\`\`\`typescript
${lastCode}
\`\`\`

Test output (errors):
${lastOutput}

Fix the code so all tests pass. Return ONLY the corrected implementation code.`

      const result = await generateText({
        model: mistral('mistral-small-latest'),
        system: `You are a TypeScript developer. Generate ONLY implementation code.
Do NOT include test code, markdown fences, or explanations.
Export all functions that the tests import.`,
        prompt,
      })

      // Clean up any markdown fences the model might add
      lastCode = result.text
        .replace(/^```(?:typescript|ts)?\n/gm, '')
        .replace(/\n```$/gm, '')
        .trim()

      // Run tests
      const testResult = await runTests(lastCode, spec.testCode, workDir)
      lastOutput = testResult.output

      console.log(`[TDD] Tests ${testResult.pass ? 'PASSED' : 'FAILED'}`)

      if (testResult.pass) {
        return {
          code: lastCode,
          testsPass: true,
          testOutput: testResult.output,
          attempts: attempt,
        }
      }

      console.log('[TDD] Test output:', testResult.output.slice(0, 500))
    }

    return {
      code: lastCode,
      testsPass: false,
      testOutput: lastOutput,
      attempts: maxAttempts,
    }
  } finally {
    await rm(workDir, { recursive: true, force: true }).catch(() => {})
  }
}

// Usage
const result = await generateToPassTests({
  functionName: 'chunk',
  description: 'Split an array into chunks of a specified size. The last chunk may be smaller.',
  testCode: `
describe("chunk", () => {
  it("should split array into equal chunks", () => {
    expect(chunk([1, 2, 3, 4], 2)).toEqual([[1, 2], [3, 4]]);
  });

  it("should handle last chunk being smaller", () => {
    expect(chunk([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
  });

  it("should handle chunk size larger than array", () => {
    expect(chunk([1, 2], 5)).toEqual([[1, 2]]);
  });

  it("should return empty array for empty input", () => {
    expect(chunk([], 3)).toEqual([]);
  });

  it("should handle chunk size of 1", () => {
    expect(chunk([1, 2, 3], 1)).toEqual([[1], [2], [3]]);
  });
});`,
})

console.log(`\nTests passed: ${result.testsPass}`)
console.log(`Attempts: ${result.attempts}`)
console.log(`\nGenerated code:\n${result.code}`)
````

> **Beginner Note:** Test-driven generation is the most reliable way to generate code with LLMs. The tests define exactly what the code should do, and the iterative loop ensures the code actually works. Without tests, you are trusting the model's output blindly.

---

## Section 6: Iterative Refinement

### The Generate-Run-Fix Loop

The most powerful code generation pattern is iterative refinement: generate code, run it, check the output, and fix any issues:

````typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { writeFile, mkdir, rm } from 'fs/promises'
import { join } from 'path'
import { randomUUID } from 'crypto'
import { spawn } from 'child_process'

interface RefinementStep {
  attempt: number
  code: string
  exitCode: number
  stdout: string
  stderr: string
  action: 'initial' | 'fix_error' | 'improve_output'
}

async function executeCode(
  code: string,
  workDir: string
): Promise<{ exitCode: number; stdout: string; stderr: string }> {
  const filePath = join(workDir, 'code.ts')
  await writeFile(filePath, code)

  return new Promise(resolve => {
    const child = spawn('bun', ['run', filePath], {
      cwd: workDir,
      timeout: 10000,
      env: { ...process.env, MISTRAL_API_KEY: '', ANTHROPIC_API_KEY: '' },
    })

    let stdout = ''
    let stderr = ''
    child.stdout.on('data', d => (stdout += d.toString()))
    child.stderr.on('data', d => (stderr += d.toString()))

    child.on('close', code => {
      resolve({
        exitCode: code ?? 1,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
      })
    })
    child.on('error', err => {
      resolve({ exitCode: 1, stdout: '', stderr: err.message })
    })
  })
}

async function iterativeCodeGen(
  task: string,
  expectedOutput: string | ((output: string) => boolean),
  maxAttempts: number = 4
): Promise<{
  finalCode: string
  success: boolean
  steps: RefinementStep[]
}> {
  const workDir = join('/tmp', `refine-${randomUUID().slice(0, 8)}`)
  await mkdir(workDir, { recursive: true })

  const steps: RefinementStep[] = []
  let currentCode = ''

  try {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      // Generate or fix code
      let prompt: string
      let action: RefinementStep['action']

      if (attempt === 1) {
        action = 'initial'
        prompt = `Write a TypeScript script for this task:
${task}

The script should print its output to console.
Return ONLY the code, no markdown fences or explanations.`
      } else {
        const lastStep = steps[steps.length - 1]

        if (lastStep.exitCode !== 0) {
          action = 'fix_error'
          prompt = `Fix this TypeScript code that has errors.

Code:
${currentCode}

Error output:
${lastStep.stderr || lastStep.stdout}

Fix the errors and return ONLY the corrected code.`
        } else {
          action = 'improve_output'
          prompt = `This TypeScript code runs but the output does not match expectations.

Code:
${currentCode}

Current output:
${lastStep.stdout}

Expected: ${typeof expectedOutput === 'string' ? expectedOutput : 'Output did not pass validation'}

Fix the code to produce the expected output. Return ONLY the corrected code.`
        }
      }

      const result = await generateText({
        model: mistral('mistral-small-latest'),
        system: 'You are a TypeScript developer. Return ONLY executable code. No markdown fences, no explanations.',
        prompt,
      })

      currentCode = result.text
        .replace(/^```(?:typescript|ts)?\n/gm, '')
        .replace(/\n```$/gm, '')
        .trim()

      // Run the code
      const execution = await executeCode(currentCode, workDir)

      steps.push({
        attempt,
        code: currentCode,
        exitCode: execution.exitCode,
        stdout: execution.stdout,
        stderr: execution.stderr,
        action,
      })

      console.log(`[Refine] Attempt ${attempt}: exit=${execution.exitCode}, action=${action}`)

      // Check if output matches expectations
      if (execution.exitCode === 0) {
        const outputMatches =
          typeof expectedOutput === 'string'
            ? execution.stdout.trim() === expectedOutput.trim()
            : expectedOutput(execution.stdout)

        if (outputMatches) {
          return { finalCode: currentCode, success: true, steps }
        }
      }
    }

    return { finalCode: currentCode, success: false, steps }
  } finally {
    await rm(workDir, { recursive: true, force: true }).catch(() => {})
  }
}

// Usage
const result = await iterativeCodeGen(
  'Calculate the sum of all prime numbers below 100 and print the result',
  '1060', // Expected sum of primes below 100
  4
)

console.log(`\nSuccess: ${result.success}`)
console.log(`Attempts: ${result.steps.length}`)
console.log(`\nFinal code:\n${result.finalCode}`)
````

> **Advanced Note:** The iterative refinement loop is essentially an agent that generates code, tests it, and fixes errors. The difference from a general agent is that the steps are structured (generate, run, check, fix) rather than open-ended. This hybrid of chain structure and agent-like iteration produces the best results for code generation.

---

## Section 7: Code Review by LLM

### Automated Code Review

LLMs can review code for bugs, style issues, and improvement opportunities:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

const reviewSchema = z.object({
  overallQuality: z.number().min(1).max(10).describe('Overall code quality score'),
  issues: z.array(
    z.object({
      severity: z.enum(['critical', 'warning', 'suggestion']),
      line: z.number().optional().describe('Approximate line number'),
      description: z.string(),
      fix: z.string().optional().describe('Suggested fix'),
    })
  ),
  strengths: z.array(z.string()).describe('What the code does well'),
  securityConcerns: z.array(z.string()).describe('Any security issues found'),
  testSuggestions: z.array(z.string()).describe('Suggested test cases for this code'),
})

type CodeReview = z.infer<typeof reviewSchema>

async function reviewCode(code: string, context?: string): Promise<CodeReview> {
  const { output: review } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: reviewSchema }),
    system: `You are a senior TypeScript developer performing a code review.
Be thorough but constructive. Focus on:
1. Correctness: Does the code do what it should?
2. Edge cases: Are there inputs that would cause failures?
3. Security: Are there injection, validation, or access control issues?
4. Performance: Are there obvious inefficiencies?
5. Readability: Is the code clear and well-structured?
6. TypeScript best practices: Are types used effectively?`,
    prompt: `Review this TypeScript code:

\`\`\`typescript
${code}
\`\`\`

${context ? `Context: ${context}` : ''}`,
  })

  return review!
}

// Usage
const codeToReview = `
export async function fetchUserData(userId: string) {
  const response = await fetch(\`https://api.example.com/users/\${userId}\`);
  const data = await response.json();
  return data;
}

export function formatUserName(user: any) {
  return user.firstName + " " + user.lastName;
}

export async function deleteUser(userId: string) {
  await fetch(\`https://api.example.com/users/\${userId}\`, {
    method: "DELETE",
  });
  console.log("User deleted: " + userId);
}
`

const review = await reviewCode(codeToReview, 'User management module for a web application')

console.log(`Quality score: ${review.overallQuality}/10`)

console.log('\nIssues:')
for (const issue of review.issues) {
  console.log(`  [${issue.severity}] ${issue.description}`)
  if (issue.fix) console.log(`    Fix: ${issue.fix}`)
}

console.log('\nStrengths:', review.strengths)
console.log('\nSecurity concerns:', review.securityConcerns)
console.log('\nTest suggestions:', review.testSuggestions)
```

### Review-and-Fix Pipeline

Combine review with automatic fixing:

````typescript
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'

async function reviewAndFix(
  code: string,
  maxRounds: number = 2
): Promise<{
  code: string
  reviews: CodeReview[]
  rounds: number
}> {
  let currentCode = code
  const reviews: CodeReview[] = []

  for (let round = 0; round < maxRounds; round++) {
    // Review
    const review = await reviewCode(currentCode)
    reviews.push(review)

    console.log(`[Round ${round + 1}] Quality: ${review.overallQuality}/10, Issues: ${review.issues.length}`)

    // Check if quality is good enough
    const criticalIssues = review.issues.filter(i => i.severity === 'critical')
    if (criticalIssues.length === 0 && review.overallQuality >= 8) {
      console.log('Code meets quality threshold.')
      return { code: currentCode, reviews, rounds: round + 1 }
    }

    // Fix issues
    const issueList = review.issues
      .filter(i => i.severity !== 'suggestion')
      .map(i => `- [${i.severity}] ${i.description}${i.fix ? ` (Suggested fix: ${i.fix})` : ''}`)
      .join('\n')

    const fixResult = await generateText({
      model: mistral('mistral-small-latest'),
      system: 'You are a TypeScript developer fixing code review issues. Return ONLY the fixed code.',
      prompt: `Fix these issues in the code:

Issues:
${issueList}

Security concerns:
${review.securityConcerns.join('\n')}

Original code:
\`\`\`typescript
${currentCode}
\`\`\`

Return the complete fixed code.`,
    })

    currentCode = fixResult.text
      .replace(/^```(?:typescript|ts)?\n/gm, '')
      .replace(/\n```$/gm, '')
      .trim()
  }

  return { code: currentCode, reviews, rounds: maxRounds }
}

// Usage
const result = await reviewAndFix(
  `
export async function fetchUserData(userId: string) {
  const response = await fetch(\`https://api.example.com/users/\${userId}\`);
  const data = await response.json();
  return data;
}

export function formatUserName(user: any) {
  return user.firstName + " " + user.lastName;
}
`,
  2
)

console.log(`\nFinal code (after ${result.rounds} rounds):`)
console.log(result.code)
````

> **Beginner Note:** LLM code review catches different issues than a linter or type checker. Linters catch style violations. Type checkers catch type errors. LLMs catch logic errors, missing edge cases, and security issues that static analysis misses.

---

## Section 8: Security Considerations

### The Golden Rule

**Never execute untrusted code in a trusted environment.** LLM-generated code is untrusted code by definition.

### Security Layers

Implement defense in depth with multiple protection layers:

```typescript
// Layer 1: Static analysis before execution
function staticAnalysis(code: string): {
  safe: boolean
  warnings: string[]
} {
  const warnings: string[] = []

  // Check for dangerous patterns
  const dangerousPatterns = [
    { pattern: /process\.exit/g, reason: 'Code tries to exit the process' },
    {
      pattern: /require\s*\(\s*['"]child_process['"]\s*\)/g,
      reason: 'Code imports child_process',
    },
    {
      pattern: /require\s*\(\s*['"]fs['"]\s*\)/g,
      reason: 'Code imports fs module',
    },
    {
      pattern: /import.*from\s+['"]fs['"]/g,
      reason: 'Code imports fs module',
    },
    {
      pattern: /import.*from\s+['"]child_process['"]/g,
      reason: 'Code imports child_process',
    },
    { pattern: /fetch\s*\(/g, reason: 'Code makes network requests' },
    {
      pattern: /globalThis|global\./g,
      reason: 'Code accesses global scope',
    },
    {
      pattern: /Bun\.env|process\.env/g,
      reason: 'Code accesses environment variables',
    },
    {
      pattern: /Bun\.file|Bun\.write/g,
      reason: 'Code accesses file system via Bun',
    },
  ]

  for (const { pattern, reason } of dangerousPatterns) {
    if (pattern.test(code)) {
      warnings.push(reason)
    }
  }

  return {
    safe: warnings.length === 0,
    warnings,
  }
}

// Layer 2: Sandboxed execution (see Section 4)
// Always run in a subprocess with limited permissions

// Layer 3: Output validation
function validateOutput(output: string): {
  valid: boolean
  issues: string[]
} {
  const issues: string[] = []

  // Check output size
  if (output.length > 1_000_000) {
    issues.push('Output exceeds 1MB limit')
  }

  // Check for sensitive data leakage
  const sensitivePatterns = [
    /sk-ant-[a-zA-Z0-9]+/g, // Anthropic API keys
    /sk-[a-zA-Z0-9]+/g, // OpenAI API keys
    /password\s*[:=]\s*.+/gi, // Passwords
  ]

  for (const pattern of sensitivePatterns) {
    if (pattern.test(output)) {
      issues.push('Output may contain sensitive data')
    }
  }

  return {
    valid: issues.length === 0,
    issues,
  }
}

// Layer 4: Rate limiting
class ExecutionRateLimiter {
  private executions: number[] = []
  private maxPerMinute: number

  constructor(maxPerMinute: number = 10) {
    this.maxPerMinute = maxPerMinute
  }

  canExecute(): boolean {
    const now = Date.now()
    const oneMinuteAgo = now - 60_000

    // Clean old entries
    this.executions = this.executions.filter(t => t > oneMinuteAgo)

    return this.executions.length < this.maxPerMinute
  }

  recordExecution(): void {
    this.executions.push(Date.now())
  }
}

// Complete secure execution pipeline
async function secureCodeExecution(code: string): Promise<{
  executed: boolean
  result?: { stdout: string; exitCode: number }
  blocked?: { reason: string }
}> {
  // Layer 1: Static analysis
  const analysis = staticAnalysis(code)
  if (!analysis.safe) {
    return {
      executed: false,
      blocked: {
        reason: `Static analysis blocked: ${analysis.warnings.join('; ')}`,
      },
    }
  }

  // Layer 2: Rate limiting
  const rateLimiter = new ExecutionRateLimiter(10)
  if (!rateLimiter.canExecute()) {
    return {
      executed: false,
      blocked: { reason: 'Rate limit exceeded' },
    }
  }
  rateLimiter.recordExecution()

  // Layer 3: Sandboxed execution (simplified for demonstration)
  console.log('[Security] All checks passed. Executing in sandbox...')

  // In production, use the full sandbox from Section 4
  return {
    executed: true,
    result: {
      stdout: '[Sandboxed execution output would appear here]',
      exitCode: 0,
    },
  }
}

// Usage
const testCode = `
const sum = Array.from({ length: 10 }, (_, i) => i + 1).reduce((a, b) => a + b, 0);
console.log("Sum of 1-10:", sum);
`

const secResult = await secureCodeExecution(testCode)
console.log(secResult)
```

### Security Checklist for Code Generation Systems

| Risk                                     | Mitigation                                           |
| ---------------------------------------- | ---------------------------------------------------- |
| Generated code accesses filesystem       | Sandbox with no fs access; strip fs imports          |
| Generated code makes network requests    | Network-isolated sandbox; block outbound connections |
| Generated code accesses env variables    | Strip sensitive env vars in sandbox                  |
| Generated code runs forever              | Timeout limits (10s default)                         |
| Generated code produces massive output   | Output size limits (1MB default)                     |
| Generated code contains malware patterns | Static analysis before execution                     |
| Users inject prompt attacks via input    | Separate user input from code generation prompt      |
| Generated code leaks sensitive data      | Scan output for sensitive patterns                   |

> **Advanced Note:** For production code generation systems, use container-based sandboxing (Docker, Firecracker, or gVisor) instead of subprocess isolation. Subprocess isolation prevents the worst issues but does not provide true security boundaries. Container sandboxing gives you CPU/memory limits, network isolation, and filesystem restrictions.

---

## Quiz

### Question 1 (Easy)

What is the most reliable pattern for generating code with LLMs?

- A) Generate code with a single prompt and trust the output
- B) Generate code, then ask the LLM to review its own code
- C) Write tests first, then generate code that passes the tests iteratively
- D) Use a higher temperature for more creative code solutions

**Answer: C** — Test-driven generation is the most reliable pattern because the tests define exactly what the code should do, and the iterative loop (generate, test, fix) ensures the code actually works. Trusting a single generation (A) is unreliable. Self-review (B) can catch some issues but does not verify correctness. Higher temperature (D) increases creativity but decreases reliability.

---

### Question 2 (Medium)

Why should you NEVER execute LLM-generated code directly in your main process?

- A) LLM-generated code is always slower than hand-written code
- B) Generated code might access sensitive data, make network calls, modify files, or crash the process
- C) LLM-generated code uses too much memory
- D) Generated code cannot import npm packages

**Answer: B** — LLM-generated code is untrusted. It might contain code that reads environment variables (API keys), makes unauthorized network requests, deletes files, or causes the main process to crash. Running it in a sandboxed subprocess with stripped environment variables and resource limits prevents these risks.

---

### Question 3 (Medium)

When generating code with an LLM, which prompt element has the most impact on code quality?

- A) The model temperature setting
- B) Specific input/output examples that define expected behavior
- C) The length of the system prompt
- D) Whether you use `generateText` or `generateText` with `Output.object`

**Answer: B** — Input/output examples are the most effective way to communicate exactly what the code should do. They eliminate ambiguity about edge cases, expected formats, and behavior. Temperature (A) affects randomness but not fundamental quality. System prompt length (C) is less important than specificity. The API choice (D) affects output format but not the quality of the generated code itself.

---

### Question 4 (Hard)

In an iterative refinement loop (generate, run, fix), what information should you pass to the LLM when asking it to fix a failed attempt?

- A) Only the error message
- B) The previous code, the error message, and the expected output
- C) The original task description and the error message
- D) All previous attempts and all error messages

**Answer: B** — The model needs three things to fix code effectively: the code that failed (so it knows what to change), the error message (so it knows what went wrong), and the expected output (so it knows what success looks like). Only the error (A) leaves the model guessing about the code. Only the original task (C) might cause the model to start from scratch. All previous attempts (D) wastes tokens and can confuse the model.

---

### Question 5 (Hard)

A code generation system passes static analysis and runs in a sandbox, but the generated code outputs the value of `process.env.HOME`. What security layer failed?

- A) Static analysis should have caught `process.env`
- B) The sandbox should have stripped sensitive environment variables
- C) Output validation should have flagged the environment variable value
- D) All three layers should have contributed to preventing this

**Answer: D** — Defense in depth means all layers should contribute. Static analysis (A) should flag `process.env` access. The sandbox (B) should set `HOME` to the sandbox directory, not the real home path. Output validation (C) should scan for path-like strings that might reveal system information. In a well-designed system, multiple layers would catch this issue, so even if one fails, the others provide protection.

---

## Exercises

### Exercise 1: Test-Driven Code Generation Agent

**Objective:** Build a system that takes a function specification and test cases, then generates TypeScript code that passes all tests, iterating until the tests pass or a maximum number of attempts is reached.

**Specification:**

1. Create a file `src/exercises/m17/ex01-tdd-codegen.ts`
2. Export an async function `generateToPassTests(spec: FunctionSpec): Promise<CodeGenResult>`
3. Define the types:

```typescript
interface FunctionSpec {
  name: string
  description: string
  testCode: string // Bun test code (describe/it/expect)
  maxAttempts?: number // default: 3
  verbose?: boolean // default: false
}

interface CodeGenResult {
  code: string
  testsPass: boolean
  attempts: number
  history: Array<{
    attempt: number
    code: string
    testOutput: string
    passed: boolean
  }>
}
```

4. The system must:
   - Generate initial code from the spec description
   - Run the tests against the generated code
   - If tests fail, send the error output back to the LLM for fixing
   - Track all attempts in the history
   - Clean up temporary files after execution

5. Include static analysis to block dangerous code before execution

**Example usage:**

```typescript
const result = await generateToPassTests({
  name: 'flatten',
  description: 'Recursively flatten a nested array to a single level',
  testCode: `
describe("flatten", () => {
  it("should flatten nested arrays", () => {
    expect(flatten([1, [2, [3, [4]]]])).toEqual([1, 2, 3, 4]);
  });
  it("should handle empty arrays", () => {
    expect(flatten([])).toEqual([]);
  });
  it("should handle already flat arrays", () => {
    expect(flatten([1, 2, 3])).toEqual([1, 2, 3]);
  });
});`,
  maxAttempts: 3,
  verbose: true,
})

console.log(`Passed: ${result.testsPass}, Attempts: ${result.attempts}`)
```

**Test specification:**

```typescript
// tests/exercises/m17/ex01-tdd-codegen.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 17: TDD Code Generation', () => {
  it('should generate code that passes simple tests', async () => {
    const result = await generateToPassTests({
      name: 'add',
      description: 'Add two numbers',
      testCode: `
describe("add", () => {
  it("adds positive numbers", () => { expect(add(2, 3)).toBe(5); });
  it("adds negative numbers", () => { expect(add(-1, -2)).toBe(-3); });
  it("adds zero", () => { expect(add(0, 5)).toBe(5); });
});`,
    })
    expect(result.testsPass).toBe(true)
    expect(result.attempts).toBeLessThanOrEqual(3)
  })

  it('should track attempt history', async () => {
    const result = await generateToPassTests({
      name: 'reverse',
      description: 'Reverse a string',
      testCode: `
describe("reverse", () => {
  it("reverses a string", () => { expect(reverse("hello")).toBe("olleh"); });
  it("handles empty string", () => { expect(reverse("")).toBe(""); });
});`,
    })
    expect(result.history.length).toBeGreaterThan(0)
    expect(result.history[result.history.length - 1].passed).toBe(true)
  })

  it('should handle generation failure gracefully', async () => {
    const result = await generateToPassTests({
      name: 'impossible',
      description: 'A function that returns the meaning of life',
      testCode: `
describe("impossible", () => {
  it("returns 42 for any input", () => {
    expect(impossible("anything")).toBe(42);
    expect(impossible(123)).toBe(42);
  });
});`,
      maxAttempts: 2,
    })
    expect(result.attempts).toBeLessThanOrEqual(2)
  })
})
```

---

### Exercise 2: Code Review Pipeline

**Objective:** Build a pipeline that reviews code, fixes identified issues, and verifies the fixes — combining LLM code review with automated improvement.

**Specification:**

1. Create a file `src/exercises/m17/ex02-review-pipeline.ts`
2. Export an async function `reviewAndImprove(code: string, options?: ReviewOptions): Promise<ReviewResult>`
3. Define the types:

```typescript
interface ReviewOptions {
  maxRounds?: number // default: 2
  qualityThreshold?: number // default: 7 (out of 10)
  verbose?: boolean
}

interface ReviewIssue {
  severity: 'critical' | 'warning' | 'suggestion'
  description: string
  fix?: string
}

interface ReviewRound {
  round: number
  qualityScore: number
  issues: ReviewIssue[]
  codeAfterFix: string
}

interface ReviewResult {
  originalCode: string
  finalCode: string
  rounds: ReviewRound[]
  finalQualityScore: number
  totalIssuesFound: number
  totalIssuesFixed: number
}
```

4. The pipeline:
   - Reviews the code and scores quality (1-10)
   - If quality is below threshold, fixes critical and warning issues
   - Reviews the fixed code again
   - Repeats until quality threshold is met or maxRounds is reached

**Test specification:**

```typescript
// tests/exercises/m17/ex02-review-pipeline.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 17: Code Review Pipeline', () => {
  it('should review code and produce a quality score', async () => {
    const result = await reviewAndImprove(`
function add(a, b) { return a + b; }
    `)
    expect(result.finalQualityScore).toBeGreaterThan(0)
    expect(result.finalQualityScore).toBeLessThanOrEqual(10)
  })

  it('should improve code quality across rounds', async () => {
    const result = await reviewAndImprove(
      `
function processData(data: any) {
  var result = [];
  for (var i = 0; i < data.length; i++) {
    result.push(data[i].name);
  }
  return result;
}
    `,
      { maxRounds: 2 }
    )
    expect(result.finalQualityScore).toBeGreaterThanOrEqual(result.rounds[0].qualityScore)
  })

  it('should track total issues found and fixed', async () => {
    const result = await reviewAndImprove(`
export async function fetchData(url) {
  const response = await fetch(url);
  const data = await response.json();
  return data;
}
    `)
    expect(result.totalIssuesFound).toBeGreaterThan(0)
  })
})
```

> **Local Alternative (Ollama):** For code generation, `ollama('qwen3.5')` handles code tasks well. For specialized code work, use `ollama('qwen3-coder-next:cloud')`. The iterative refinement and test-driven generation patterns work with any model.

---

## Summary

In this module, you learned:

1. **LLMs as code generators:** They excel at boilerplate and common patterns but struggle with edge cases, security, and novel logic. Always verify generated code.
2. **Prompting for code:** Specific prompts with function signatures, examples, constraints, and edge cases produce dramatically better code than vague requests.
3. **Structured code output:** Use `generateText` with `Output.object` and schemas to extract clean code, dependencies, and metadata from LLM responses.
4. **Sandboxed execution:** Never run generated code in your main process. Use subprocesses with stripped environment variables, timeouts, and resource limits.
5. **Test-driven generation:** Write tests first, then generate code iteratively until the tests pass. This is the most reliable code generation pattern.
6. **Iterative refinement:** The generate-run-fix loop combines generation, execution, error analysis, and fixing into a powerful cycle that converges on working code.
7. **Code review by LLM:** LLMs catch logic errors, security issues, and style problems that static analysis misses. Combine review with automated fixing for a review-and-improve pipeline.
8. **Security considerations:** Defense in depth with static analysis, sandboxed execution, output validation, and rate limiting protects against the risks of running untrusted code.

In Module 18, you will learn how to add human oversight to these automated systems — approval gates for high-stakes actions, feedback integration for continuous improvement, and audit trails for compliance.
