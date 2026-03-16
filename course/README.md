# Applied LLM Engineering with TypeScript

A comprehensive, project-based course teaching applied LLM engineering from first API calls through production deployment. Built in TypeScript with the Vercel AI SDK.

## How It Works

- **24 modules** across 6 parts, ~174-229 hours
- **Multi-provider** — learn with Mistral (default, free tier), Groq, Anthropic, OpenAI, or Ollama (local)
- **Vercel AI SDK** — provider-agnostic patterns that work everywhere
- **Gamification** — earn XP, unlock badges, climb ranks from Token to LLM Architect
- **Claude Code guided** — each module taught section-by-section with interactive exercises

## Getting Started

```bash
bun install
cp course/preferences.example.toml course/preferences.toml
# Edit preferences.toml to match your background and provider
```

To start a module, run the corresponding Claude Code command: `/module-1`

To check progress: `bun run progress`

## Curriculum

### Part I: First Contact

| # | Module | Topics | Hours |
|---|--------|--------|-------|
| 1 | Setup & First LLM Calls | Bun, AI SDK, providers, generateText, streamText, roles, temperature | 5-7 |
| 2 | Prompt Engineering | System prompts, few-shot, chain-of-thought, templates, versioning | 5-7 |
| 3 | Structured Output | Output.object, Zod schemas, type-safe responses, validation | 5-6 |

### Part II: Core Patterns

| # | Module | Topics | Hours |
|---|--------|--------|-------|
| 4 | Conversations & Memory | Multi-turn, context windows, sliding window, summarization | 6-8 |
| 5 | Long Context & Caching | Prompt caching, KV cache, context compression, chunked prefill | 6-8 |
| 6 | Streaming & Real-time | streamText, Output.object, SSE, backpressure, UI patterns | 6-8 |
| 7 | Tool Use | Zod tool definitions, execution, multi-step loops, maxSteps | 7-9 |
| 8 | Embeddings & Similarity | Embedding models, cosine similarity, vector stores, semantic search | 7-9 |
| 9 | RAG Fundamentals | Chunking, retrieval, context injection, citation, pipeline | 8-10 |

### Part III: Advanced Retrieval

| # | Module | Topics | Hours |
|---|--------|--------|-------|
| 10 | Advanced RAG | Hybrid search, reranking, HyDE, self-RAG, evaluation | 8-10 |
| 11 | Document Processing | PDF/HTML/markdown extraction, recursive chunking, metadata | 8-10 |
| 12 | Knowledge Graphs | Entity extraction, relationship mapping, graph RAG | 8-10 |
| 13 | Multi-modal | Vision models, image understanding, audio, multi-modal RAG | 8-10 |

### Part IV: Agents & Orchestration

| # | Module | Topics | Hours |
|---|--------|--------|-------|
| 14 | Agent Fundamentals | ReAct pattern, planning loops, tool selection, observation cycles | 8-10 |
| 15 | Multi-Agent Systems | Orchestrator-worker, delegation, shared state, communication | 7-9 |
| 16 | Workflows & Chains | Sequential/parallel pipelines, branching, composable chains | 7-9 |
| 17 | Code Generation | LLM-generated code, sandboxed execution, iterative refinement | 8-10 |
| 18 | Human-in-the-Loop | Approval flows, feedback integration, active learning | 7-9 |

### Part V: Quality & Safety

| # | Module | Topics | Hours |
|---|--------|--------|-------|
| 19 | Evals & Testing | LLM-as-judge, benchmarks, regression suites, prompt CI | 8-10 |
| 20 | Fine-tuning | When to fine-tune, dataset prep, training pipelines, evaluation | 8-10 |
| 21 | Safety & Guardrails | Input validation, output filtering, jailbreak prevention | 8-10 |
| 22 | Cost Optimization | Semantic caching, model routing, token budgets, fallback chains | 8-10 |

### Part VI: Production

| # | Module | Topics | Hours |
|---|--------|--------|-------|
| 23 | Observability | Logging, tracing, token tracking, latency metrics, debugging | 8-10 |
| 24 | Deployment | Hono server, auth, rate limiting, scaling, provider failover | 10-12 |

## Estimated Time

| Part | Modules | Hours |
|------|---------|-------|
| I. First Contact | 1-3 | 15-20 |
| II. Core Patterns | 4-9 | 40-52 |
| III. Advanced Retrieval | 10-13 | 32-40 |
| IV. Agents & Orchestration | 14-18 | 37-47 |
| V. Quality & Safety | 19-22 | 32-40 |
| VI. Production | 23-24 | 18-22 |
| **Total** | **1-24** | **~174-229** |

## Prerequisites

- TypeScript (comfortable with types, generics, async/await)
- Terminal comfort (running commands, environment variables)
- API key for at least one provider (Anthropic, OpenAI) OR Ollama installed locally
- No AI/ML background required — we start from first principles

## Module Dependencies

- Parts I-III are sequential (each builds on the previous)
- Parts IV and V can be done in either order after Part III
- Part VI requires all prior parts
