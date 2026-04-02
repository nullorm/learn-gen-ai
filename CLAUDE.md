# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A 24-module applied LLM engineering course taught interactively via Claude Code. Students build real LLM applications in TypeScript using the Vercel AI SDK with multi-provider support (Mistral, Groq, Claude, OpenAI, Ollama). Each module is launched with `/module-N` and follows a teach → quiz → exercises flow.

## Commands

```bash
bun test                              # Run all tests
bun test tests/config.test.ts         # Run a single test file
bun run tools/progress.ts             # Show student progress
bun run tools/progress.ts start N     # Start module N
bun run tools/progress.ts quiz N S T  # Record quiz (score S out of T)
bun run tools/progress.ts exercise N E # Record exercise E in module N
bun run tools/progress.ts complete N  # Mark module N complete
```

## Architecture

### Source Layers

Students build code incrementally across modules. Later modules import from earlier ones:

```
src/core/         → Provider setup, model configuration, shared utilities
src/prompts/      → Prompt templates, management, versioning
src/structured/   → Zod schemas, structured output patterns
src/memory/       → Conversation management, context strategies
src/streaming/    → Stream handlers, SSE utilities
src/tools/        → Tool definitions, execution patterns
src/embeddings/   → Embedding generation, similarity search
src/rag/          → Chunking, retrieval, RAG pipelines
src/agents/       → Agent loops, multi-agent orchestration
src/eval/         → Evaluation framework, LLM-as-judge
src/safety/       → Guardrails, input/output validation
```

### Module Teaching System

- `course/module_NN_*.md` — course content (concepts, code, quiz answers, exercises)
- `.claude/commands/module-N.md` — Claude Code slash commands defining the teaching flow
- `tools/progress*.ts` — XP/rank/badge/streak gamification engine, state in `progress.json` (gitignored)
- `src/config.ts` — loads `course/preferences.toml` (gitignored) for user preferences

### Test Structure

Tests mirror `src/` under `tests/`. Test runner config in `bunfig.toml` sets root to `./tests`. Uses `bun:test` exclusively.

## Code Conventions

- **Strict TypeScript** — `strict: true`, `noUncheckedIndexedAccess: true`
- **Pure functions over classes** — except where state is inherent
- **ESM imports only** — no `require()`, always use `.js` extensions in relative imports (e.g., `from './provider.js'`)
- **Ollama thinking mode** — Qwen3/3.5 models default to thinking mode which consumes all tokens in `<think>` tags. Disable via the model constructor: `ollama('qwen3.5', { think: false })`. The `ai-sdk-ollama` provider handles this natively
- **Prettier** — no semicolons, single quotes, trailing commas (es5), 120 char width
- **Vercel AI SDK patterns** — `generateText`, `streamText`, `Output.object()` for all LLM calls
- **Zod v4 patterns** — use top-level APIs: `z.int()`, `z.email()`, `z.url()`, `z.uuid()`, `z.iso.date()`. Chaining works: `z.int().min(1).max(10)`. Use Zod for all tool definitions, structured output, and validation
- **Provider-agnostic** — default provider is Mistral (free tier: 1 RPS, 500K tokens/min, 1B tokens/month per model), with Groq, Anthropic, OpenAI, and Ollama as alternatives

## Teaching Flow (for module commands)

Each `/module-N` command follows this sequence:

1. Read course content, preferences, and existing code
2. Teach section by section using the **student-builds-everything** approach (see below)
3. Quiz: 5 questions, one at a time, 80% to pass
4. Exercises: guided implementation, progress tracked per exercise
5. Completion: `bun run tools/progress.ts complete N`

User preferences (`course/preferences.toml`) control difficulty level and default provider. If the file doesn't exist, defaults to intermediate level with Anthropic provider.

### Student-Builds-Everything Approach

**The student writes ALL implementation code. The assistant writes tests and explains concepts.**

This is the core teaching philosophy. Never write implementation code for the student. Never create files they should create. Never fill in function bodies they should figure out.

For each section:

1. **Explain** the concept clearly (what it is, why it matters, how it works)
2. **Write a test** that defines the expected behavior — put it in `tests/` following the mirror structure
3. **Tell the student what to build** — specify the file path, exports, types, and behavior in plain language
4. **Wait** for the student to write the code and run the tests
5. **If tests pass** — briefly discuss the output, give an insight, move on
6. **If tests fail** — give a hint (not the answer), let them fix it

Rules:

- **NEVER** write implementation files (`src/`) for the student — only test files (`tests/`)
- **NEVER** create example files and run them yourself — describe what to build, let the student build and run it
- **NEVER** show complete function bodies in code blocks — show signatures, types, and describe the logic in words
- **ONE section at a time** — do not proceed until the student's code passes the current section's tests
- **Wait for student input** between every section — do not auto-advance
- When explaining code patterns, use short inline snippets (1-3 lines) to illustrate syntax, not full implementations
- Guide with questions: "What should happen when the env var is missing?" not "Here's the error handling code"
- If the student is stuck after 2 hints, offer to show a minimal skeleton (signature + comments only, no body)
