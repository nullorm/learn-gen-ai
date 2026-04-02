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

The difference between good and bad code generation comes down to prompt specificity. Compare these two approaches:

```typescript
// BAD: Vague prompt — might return Python, JavaScript, or anything
const badResult = await generateText({
  model: mistral('mistral-small-latest'),
  prompt: 'Write a function to process data',
})
```

A good code generation prompt specifies the language, function name, input/output types, edge cases, constraints, and output format. When you tell the model exactly what to produce — including what NOT to include — the results improve dramatically.

Your task: build a function `generateCodeWithPrompt` in `src/codegen/basics.ts` that takes a detailed specification string and calls `generateText` with it. The function should accept a prompt string and return the LLM's text response. Think about: what happens if the model wraps the code in markdown fences? How would you strip those from the response?

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

Build a prompt template function that assembles these elements into a structured prompt. The function signature should be:

```typescript
function codePrompt(spec: {
  language: string
  functionName: string
  description: string
  parameters: Array<{ name: string; type: string; description: string }>
  returnType: string
  examples: Array<{ input: string; output: string }>
  constraints: string[]
  edgeCases: string[]
}): string
```

The function should combine all the spec fields into a clear, structured prompt string. Think about the ordering: which sections should come first to give the model the most context? How do you format the examples so the model understands input/output pairs?

The prompt should end with an instruction like: "Return ONLY the function code with a JSDoc comment. No tests, no explanation, no markdown fences."

### System Prompts for Code Generation

Beyond per-request prompts, a system prompt sets global coding standards. A good code generation system prompt covers:

1. **Style rules** — ES2022+, const over let, arrow functions for callbacks, early returns over nesting
2. **Type discipline** — explicit annotations, interfaces for object shapes, no `any` (use `unknown` instead)
3. **Error handling** — validate inputs at function boundaries, custom error classes, never swallow errors
4. **Documentation** — JSDoc comments, parameter/return docs, `@example` tags
5. **Output format** — no markdown fences, no test code, no explanation unless asked

Build a `codeGenSystemPrompt` string constant that covers these areas. Then use it as the `system` parameter in a `generateText` call. How does the system prompt interact with the per-request prompt? What happens when they conflict?

> **Beginner Note:** Specific prompts produce specific code. If you tell the model "write a sorting function," you will get something generic. If you specify the language, algorithm, input type, edge cases, and constraints, you will get exactly what you need.

---

## Section 3: Structured Code Output

### Extracting Code from LLM Responses

LLMs often wrap code in markdown fences or add explanations. Use structured output to extract clean code reliably. Define a Zod schema that captures not just the code, but metadata about what was generated:

```typescript
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
```

Build an async function `generateCode(spec: string): Promise<CodeOutput>` that uses `generateText` with `Output.object({ schema: codeOutputSchema })` to extract structured output. The system prompt should instruct the model that the `code` field must contain complete, executable TypeScript with no markdown fences.

Why is this better than parsing raw text? What happens when the model decides to add an explanation before the code block? How does structured output prevent that?

### Multi-File Code Generation

For larger features, you need to generate multiple files that are consistent with each other. Define a schema for multi-file output:

```typescript
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
```

Build a `generateMultiFile(spec: string)` function that uses this schema. The system prompt should instruct the model to use relative imports between files and follow standard project structure.

What are the risks of multi-file generation compared to single-function generation? How would you validate that imports between generated files actually resolve correctly?

> **Advanced Note:** Multi-file generation is less reliable than single-function generation because the model must maintain consistency across files (imports, types, naming). Always validate that generated imports resolve correctly and types match across files.

---

## Section 4: Isolated Subprocess Execution

> **Security Warning:** This subprocess approach provides timeout protection and output capture, but NOT security isolation. Generated code runs with the same OS permissions as your application. For true sandboxing in production, use container-based solutions (Docker, Firecracker) or WebAssembly runtimes.

### Running Generated Code Safely

Never run LLM-generated code directly in your main process. Always isolate it in a subprocess. Build a `runInSandbox` function with this signature:

```typescript
interface SandboxResult {
  stdout: string
  stderr: string
  exitCode: number
  durationMs: number
  timedOut: boolean
}

async function runInSandbox(code: string, options?: { timeoutMs?: number; workDir?: string }): Promise<SandboxResult>
```

The function should create a temporary workspace, write the code there, run it in a subprocess, and clean up afterwards. You will need `spawn` from `child_process` and file system helpers from `fs/promises`.

Think about these design questions:

- What environment variables should the subprocess inherit? Which ones are dangerous to expose (API keys, database URLs)? What minimal set does code need to run?
- Why would you set `HOME` to the sandbox directory instead of the real home directory?
- How do you distinguish a timeout from a normal error? What happens to the subprocess when your timer fires?
- When should cleanup happen -- only on success, or always? What language construct guarantees cleanup regardless of outcome?

### Sandbox Security Checklist

When running generated code, always apply these protections:

1. **ALWAYS** run in a subprocess — never in the main process
2. **Strip sensitive env vars** — only pass PATH and NODE_ENV='sandbox'
3. **Set resource limits** — timeout (10s), max output (1MB)
4. **Use a temporary directory** that gets cleaned up
5. **NEVER** let generated code access the network in production — use containers for network isolation
6. **Validate output** before using it — check for suspicious patterns and size limits

> **Beginner Note:** Think of sandboxing like putting an unknown substance in a sealed container before testing it. You do not know what the generated code will do, so you run it in an isolated environment where it cannot damage your system or access sensitive data.

---

## Section 5: Test-Driven Generation

### Generate Code to Pass Tests

The most reliable code generation pattern: write the tests first, then have the LLM generate code that passes them. Build a system with these types:

```typescript
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
```

You need two functions:

**`runTests(implCode, testCode, workDir)`** — Takes implementation code and test code as strings, writes them to files in a working directory, and runs the tests via `bun test`. The test file needs access to the implementation's exports. Return `{ pass: boolean; output: string }`.

How do you make exports from the implementation file available to test code that references functions by name? Consider dynamic import and `globalThis`. What file naming convention does Bun's test runner expect?

**`generateToPassTests(spec, maxAttempts)`** — The iterative loop. On the first attempt, the LLM gets the function description and test code. On subsequent attempts, it also gets its previous code and the error output. The loop terminates when tests pass or attempts run out.

Think about: what information does the LLM need to fix failing code effectively? Why is including the previous code alongside the error output better than sending just the error? How does the prompt differ between "write this from scratch" and "fix this broken code"?

> **Beginner Note:** Test-driven generation is the most reliable way to generate code with LLMs. The tests define exactly what the code should do, and the iterative loop ensures the code actually works. Without tests, you are trusting the model's output blindly.

---

## Section 6: Iterative Refinement

### The Generate-Run-Fix Loop

The most powerful code generation pattern is iterative refinement: generate code, run it, check the output, and fix any issues. This differs from test-driven generation in that you validate against expected output rather than test cases.

Build an `iterativeCodeGen` function:

```typescript
interface RefinementStep {
  attempt: number
  code: string
  exitCode: number
  stdout: string
  stderr: string
  action: 'initial' | 'fix_error' | 'improve_output'
}

async function iterativeCodeGen(
  task: string,
  expectedOutput: string | ((output: string) => boolean),
  maxAttempts?: number
): Promise<{
  finalCode: string
  success: boolean
  steps: RefinementStep[]
}>
```

The loop operates in three modes based on the previous step's outcome. Look at the `action` field in `RefinementStep` -- each mode needs a different prompt strategy. The first attempt generates from scratch; subsequent attempts either fix errors or adjust output depending on whether the code crashed or produced the wrong result.

Consider these questions:

- How do you decide whether the previous step was an error (non-zero exit code) vs. a wrong-output situation? What drives the choice of prompt?
- The `expectedOutput` parameter accepts either a string or a validation function. When would you prefer a function over an exact string match?
- How does this approach compare to test-driven generation from Section 5? When would you choose one over the other?
- Why record each attempt as a `RefinementStep`? What debugging value does the full step history provide?

> **Advanced Note:** The iterative refinement loop is essentially an agent that generates code, tests it, and fixes errors. The difference from a general agent is that the steps are structured (generate, run, check, fix) rather than open-ended. This hybrid of chain structure and agent-like iteration produces the best results for code generation.

---

## Section 7: Code Review by LLM

### Automated Code Review

LLMs can review code for bugs, style issues, and improvement opportunities. Define a review schema:

```typescript
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
```

Build an async function `reviewCode(code: string, context?: string): Promise<CodeReview>` that uses structured output with this schema. The system prompt should instruct the model to act as a senior TypeScript developer focusing on correctness, edge cases, security, performance, readability, and TypeScript best practices.

What makes LLM code review different from a linter or type checker? What kinds of issues can an LLM catch that static analysis tools miss?

### Review-and-Fix Pipeline

Now combine review with automated fixing. Build a `reviewAndFix` function:

```typescript
async function reviewAndFix(
  code: string,
  maxRounds?: number
): Promise<{
  code: string
  reviews: CodeReview[]
  rounds: number
}>
```

The pipeline alternates between reviewing and fixing until quality is good enough or you run out of rounds. You need to define what "good enough" means -- think about both the numeric score and the presence of critical issues.

Consider these design questions:

- What quality threshold makes a reasonable stopping condition? Should it be score-based, issue-severity-based, or both?
- How do you format review issues so the LLM can fix them effectively? What context does the fix prompt need?
- What is the risk of over-fixing -- could the LLM introduce new issues while fixing old ones? How would you detect this?
- Which issues should be auto-fixed vs. left as suggestions for the human?

> **Beginner Note:** LLM code review catches different issues than a linter or type checker. Linters catch style violations. Type checkers catch type errors. LLMs catch logic errors, missing edge cases, and security issues that static analysis misses.

---

## Section 8: Security Considerations

### The Golden Rule

**Never execute untrusted code in a trusted environment.** LLM-generated code is untrusted code by definition.

### Security Layers

Implement defense in depth with multiple protection layers. Build each layer as a separate function:

**Layer 1: Static analysis** — `staticAnalysis(code: string): { safe: boolean; warnings: string[] }`. Scans the code string for dangerous patterns before execution. What patterns indicate risk? Think about process control, file system access, network calls, environment variable reads, and global scope manipulation. Each match should produce a human-readable warning.

**Layer 2: Sandboxed execution** — Use the sandbox from Section 4. Always run in a subprocess with limited permissions.

**Layer 3: Output validation** — `validateOutput(output: string): { valid: boolean; issues: string[] }`. Inspects the subprocess output for suspicious content. What would indicate that generated code leaked sensitive data? Think about output size, key-shaped strings, and path-like values.

**Layer 4: Rate limiting** — `ExecutionRateLimiter` class that prevents abuse by limiting executions per time window. What data structure efficiently tracks "how many executions in the last N seconds"?

**Composing the layers** — `secureCodeExecution(code: string)` runs all four layers in sequence. Which layers should run before execution and which after? If any layer rejects, the pipeline should short-circuit. What return type captures both the result and any layer that blocked?

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

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Diff-Based Code Editing

### Surgical Edits over Full Regeneration

Production code generation systems rarely regenerate entire files. Instead, they produce targeted edits — find-replace pairs that modify specific locations while preserving surrounding code. This is safer, cheaper (fewer output tokens), and easier to review.

A diff-based code modifier works like this:

1. Read the current file content
2. Ask the LLM to produce one or more edit operations (each with `old_string` and `new_string`)
3. Validate that each `old_string` appears exactly once in the file (uniqueness check)
4. Apply the replacements sequentially

```typescript
interface CodeEdit {
  old_string: string
  new_string: string
}
```

The uniqueness constraint is critical — if `old_string` matches multiple locations, the edit is ambiguous and must be rejected. The caller should provide more surrounding context to disambiguate.

When a single logical change touches multiple locations (e.g., adding an import at the top and using it in a function below), you can either issue multiple edit operations or use a unified diff patch that contains multiple hunks applied atomically.

> **Key Insight:** String replacement is safest for single-site edits. Multi-hunk unified diff patches are better when one logical change spans multiple locations in a file — either all hunks apply or none do.

---

## Section 10: Safe Code Writing

### Production Safety for File Writes

Writing generated code to disk requires safety checks that prevent accidental damage:

1. **Read before write** — Always read the existing file content before overwriting. This prevents silently destroying code the LLM has never seen.
2. **Path validation** — Reject writes to paths outside the project directory. Resolve symlinks and reject paths containing `..` that escape the workspace.
3. **Overwrite confirmation** — If the target file already exists and has content, require explicit confirmation before replacing it.
4. **Parent directory creation** — Create intermediate directories as needed so the write does not fail on a missing parent.

```typescript
const safePath = resolve(projectRoot, requestedPath)
if (!safePath.startsWith(projectRoot)) throw new Error('Path escapes project root')
```

These checks form a write guard that wraps every file write in the code generation pipeline. The guard is a pure function — it takes a proposed write and returns either an approved write or an error, making it easy to test.

---

## Section 11: Edit History and Reversibility

### Tracking Changes for Undo

Every generated code change should be tracked and reversible. An edit history log records the file path, old content, new content, and timestamp for each edit. This enables:

- **Undo** — Revert the last edit by restoring the old content
- **Redo** — Re-apply a reverted edit
- **Debugging** — Walk through the sequence of changes to understand what the agent did

The key design choice is decoupling file state from conversation state. When the user undoes a code change, the file reverts but the conversation continues — the agent remembers what it wrote and why it was undone, so it can try a different approach.

```typescript
interface EditRecord {
  filePath: string
  oldContent: string
  newContent: string
  timestamp: number
}
```

A simple stack-based history (push on edit, pop on undo) covers most use cases. For production systems, consider persisting the history to disk so it survives across sessions.

---

## Section 12: Enhanced Sandboxing

### Production Execution Constraints

The sandboxed execution from Section 4 provides the foundation. Production systems add additional constraints:

- **Timeout limits** — Kill the subprocess after a configurable duration (e.g., 10 seconds) to prevent infinite loops
- **Output size limits** — Truncate stdout/stderr beyond a threshold (e.g., 1MB) to prevent memory exhaustion
- **Directory restrictions** — Confine file access to a temporary working directory; reject reads/writes outside it
- **Permission checks** — Validate the code does not request elevated privileges or access sensitive resources before execution

These constraints compose with the existing sandbox. Each is a guard that runs before or after execution, and each can be independently tested.

> **Advanced Note:** For full isolation, production systems use container-based sandboxing (Docker, Firecracker) that provides kernel-level enforcement of CPU, memory, network, and filesystem limits. Subprocess isolation is a starting point, not a finish line.

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

### Question 6 (Medium)

Why do production code generation systems prefer diff-based edits (find-replace pairs) over regenerating entire files?

- A) Diff-based edits use more tokens, which improves quality
- B) Regenerating files is not supported by the Vercel AI SDK
- C) Diff-based edits are safer, cheaper, and easier to review because they only modify specific locations
- D) LLMs cannot generate complete files

**Answer: C** — Diff-based editing produces targeted changes that modify only the necessary locations while preserving surrounding code. This is safer (less risk of breaking unrelated code), cheaper (fewer output tokens), and easier to review (the reviewer sees exactly what changed). The uniqueness constraint on the old_string ensures edits are unambiguous.

---

### Question 7 (Hard)

An edit history system tracks code changes with old/new content and supports undo. When a user undoes a code change, what should happen to the conversation state?

- A) The conversation should be rolled back to before the edit was proposed
- B) The conversation should continue unchanged — the agent remembers what it wrote and why it was undone
- C) The entire conversation history should be cleared to avoid confusion
- D) The undo should be blocked because it would make the conversation inconsistent

**Answer: B** — File state and conversation state should be decoupled. When the user undoes a code change, the file reverts but the conversation continues with full context. The agent remembers what it wrote and why it was undone, which allows it to try a different approach. Rolling back the conversation (A) would lose the context of why the change failed. Clearing history (C) or blocking undo (D) would harm the user experience.

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

### Exercise 3: Diff-Based Code Modifier

**Objective:** Build a code modification system that generates targeted find-replace edits instead of regenerating entire files.

**Specification:**

1. Create a file `src/exercises/m17/ex03-diff-modifier.ts`
2. Export an async function `diffModify(filePath: string, instruction: string): Promise<DiffModifyResult>`
3. Define the types:

```typescript
interface CodeEdit {
  old_string: string
  new_string: string
}

interface DiffModifyResult {
  edits: CodeEdit[]
  applied: number
  rejected: number
  errors: string[]
  originalContent: string
  finalContent: string
}
```

4. The system must:
   - Read the current file content
   - Ask the LLM to produce edit operations based on the instruction
   - Validate each edit: `old_string` must appear exactly once in the file
   - Apply valid edits sequentially, skip invalid ones with an error message
   - Return both the original and final content

**Test specification:**

```typescript
// tests/exercises/m17/ex03-diff-modifier.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 17: Diff-Based Code Modifier', () => {
  it('should apply a valid single edit', async () => {
    // Write a temp file, request a modification, verify the edit was applied
    const result = await diffModify(tempFile, 'Rename the function from foo to bar')
    expect(result.applied).toBeGreaterThan(0)
    expect(result.finalContent).toContain('bar')
    expect(result.finalContent).not.toContain('foo')
  })

  it('should reject edits where old_string is not found', async () => {
    const result = await diffModify(tempFile, 'Replace the nonexistent function baz')
    expect(result.rejected).toBeGreaterThan(0)
    expect(result.errors.length).toBeGreaterThan(0)
  })

  it('should preserve file content for non-edited regions', async () => {
    const result = await diffModify(tempFile, 'Add a return type annotation to the function')
    expect(result.finalContent).toContain(unchangedSection)
  })
})
```

---

### Exercise 4: Safe Code Writer

**Objective:** Build a code writer with production safety checks: read before write, path validation, overwrite confirmation, and parent directory creation.

**Specification:**

1. Create a file `src/exercises/m17/ex04-safe-writer.ts`
2. Export an async function `safeWrite(projectRoot: string, filePath: string, content: string, options?: SafeWriteOptions): Promise<SafeWriteResult>`
3. Define the types:

```typescript
interface SafeWriteOptions {
  allowOverwrite?: boolean // default: false
  createDirectories?: boolean // default: true
}

interface SafeWriteResult {
  written: boolean
  filePath: string
  previousContent: string | null
  error?: string
}
```

4. The system must:
   - Resolve the file path and reject any path outside `projectRoot`
   - Read existing content before writing (store it in `previousContent`)
   - Refuse to overwrite existing files unless `allowOverwrite` is true
   - Create parent directories if `createDirectories` is true
   - Return a result indicating success or failure with details

**Test specification:**

```typescript
// tests/exercises/m17/ex04-safe-writer.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 17: Safe Code Writer', () => {
  it('should write a new file successfully', async () => {
    const result = await safeWrite(tmpDir, 'src/utils.ts', 'export const x = 1')
    expect(result.written).toBe(true)
    expect(result.previousContent).toBeNull()
  })

  it('should reject paths outside project root', async () => {
    const result = await safeWrite(tmpDir, '../../etc/passwd', 'bad')
    expect(result.written).toBe(false)
    expect(result.error).toContain('outside')
  })

  it('should refuse to overwrite without allowOverwrite', async () => {
    await safeWrite(tmpDir, 'existing.ts', 'original')
    const result = await safeWrite(tmpDir, 'existing.ts', 'overwritten')
    expect(result.written).toBe(false)
  })

  it('should create parent directories', async () => {
    const result = await safeWrite(tmpDir, 'deep/nested/dir/file.ts', 'content')
    expect(result.written).toBe(true)
  })
})
```

---

### Exercise 5: Edit History with Undo/Redo

**Objective:** Build an edit history system that tracks all generated code changes and supports undo/redo operations while preserving conversation context.

**Specification:**

1. Create a file `src/exercises/m17/ex05-edit-history.ts`
2. Export the `EditHistory` class
3. Define the types:

```typescript
interface EditRecord {
  filePath: string
  oldContent: string
  newContent: string
  timestamp: number
  description: string
}

class EditHistory {
  constructor()
  record(edit: Omit<EditRecord, 'timestamp'>): void
  undo(): EditRecord | null // Returns the undone edit, or null if nothing to undo
  redo(): EditRecord | null // Returns the redone edit, or null if nothing to redo
  getHistory(): EditRecord[]
  canUndo(): boolean
  canRedo(): boolean
}
```

4. The system must:
   - Record every edit with a timestamp
   - Support undo by restoring the previous content (return the edit so the caller can apply the file revert)
   - Support redo by re-applying the most recently undone edit
   - Clear the redo stack when a new edit is recorded after an undo
   - Maintain the full history for debugging

**Test specification:**

```typescript
// tests/exercises/m17/ex05-edit-history.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 17: Edit History', () => {
  it('should record and retrieve edit history', () => {
    const history = new EditHistory()
    history.record({ filePath: 'a.ts', oldContent: 'old', newContent: 'new', description: 'edit 1' })
    expect(history.getHistory()).toHaveLength(1)
  })

  it('should undo the last edit', () => {
    const history = new EditHistory()
    history.record({ filePath: 'a.ts', oldContent: 'v1', newContent: 'v2', description: 'edit' })
    const undone = history.undo()
    expect(undone).not.toBeNull()
    expect(undone!.oldContent).toBe('v1')
  })

  it('should redo after undo', () => {
    const history = new EditHistory()
    history.record({ filePath: 'a.ts', oldContent: 'v1', newContent: 'v2', description: 'edit' })
    history.undo()
    const redone = history.redo()
    expect(redone).not.toBeNull()
    expect(redone!.newContent).toBe('v2')
  })

  it('should clear redo stack on new edit after undo', () => {
    const history = new EditHistory()
    history.record({ filePath: 'a.ts', oldContent: 'v1', newContent: 'v2', description: 'edit 1' })
    history.undo()
    history.record({ filePath: 'a.ts', oldContent: 'v1', newContent: 'v3', description: 'edit 2' })
    expect(history.canRedo()).toBe(false)
  })
})
```

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
9. **Diff-based code editing:** Producing targeted find-replace edits instead of regenerating entire files is safer, cheaper, and easier to review — with a uniqueness constraint to prevent ambiguous edits.
10. **Safe code writing:** Write guards that enforce read-before-write, path validation, overwrite confirmation, and parent directory creation prevent accidental damage from generated code.
11. **Edit history and reversibility:** Tracking every code change with old/new content enables undo, redo, and debugging while keeping file state decoupled from conversation state.
12. **Enhanced sandboxing:** Production execution adds timeout limits, output size caps, directory restrictions, and permission checks on top of basic subprocess isolation.

In Module 18, you will learn how to add human oversight to these automated systems — approval gates for high-stakes actions, feedback integration for continuous improvement, and audit trails for compliance.
