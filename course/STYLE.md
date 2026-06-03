# Course Style Guide

This is the source of truth for how course modules are written. The structural
rules here are enforced by `tools/lint-course.ts` (run `bun run tools/lint-course.ts`).
The cadence and callout rules keep modules engaging instead of monotonous.

## Module Skeleton

Every `course/module_NN_*.md` follows this order. The linter fails the build if it doesn't.

```
# Module N: Title

## Learning Objectives          (bulleted; what the student can do after)
                                 > You-are-here line (see below)
## Why Should I Care?           (motivation — keep it, do not gut it)
## Connection to Other Modules  (what it builds on / feeds into)

## Section 1: ...               (core arc — gap-free numbering from 1)
## Section 2: ...
   ...
## Section K: ...

## Going Further: <theme>       (OPTIONAL coda — advanced/framework material)
### Subsection                  (### under the coda, NOT "## Section N")
### Subsection

## Summary                      (the closing trio, always in this order)
## Quiz
## Exercises
```

Rules the linter checks:
- The opening triad must be present and in order.
- `## Section N:` numbering is gap-free starting at 1.
- A `## Going Further` section (if present) sits after the last `## Section` and before `## Summary`. Its subsections are `###`.
- `## Summary` → `## Quiz` → `## Exercises` are the **last three** `##` headings, in that order.
- Every `> **Label:**` callout uses an approved label (below).

**Core vs. Going Further:** a section stays in the core arc if a general LLM
engineer (not specifically a Claude Code user) needs it to understand the
module's topic. It moves to `## Going Further` if it is framework-specific or
advanced-operational. Merge/delete a section only if it is thin (< ~1 screen)
**and** redundant with something else.

## "You are here" Line

Immediately after `## Learning Objectives`, add one blockquote naming the Part
and its badge, so the reading experience connects to the XP/badge engine
(`tools/progress-types.ts`). Example:

```
> *Module 14 opens **Part IV: Agents & Orchestration**. Complete the Part to earn the **Agent Deployer** badge.*
```

Part → modules → badge (keep in sync with `PARTS` / `PART_BADGES` in `tools/progress-types.ts`):

| Part | Name | Modules | Badge |
|------|------|---------|-------|
| I | First Contact | 1–3 | First Contact |
| II | Core Patterns | 4–9 | Core Patterns |
| III | Advanced Retrieval | 10–13 | RAG Builder |
| IV | Agents & Orchestration | 14–18 | Agent Deployer |
| V | Quality & Safety | 19–22 | Quality Gate |
| VI | Production | 23–24 | Production Ready |

Ranks the student climbs by XP (for reference): Token → Prompter → Embedder →
Retriever → Tool Smith → Agent Builder → Eval Master → LLM Architect.

## Callout Vocabulary

Plain blockquotes, no emoji. These are the **only** label *roots* the linter allows.
Use **2–4 per module** — more than that and they stop breaking the rhythm and
become the rhythm. Format: `> **Label:** text`. A label may carry a topic suffix
after a `:` or a space — e.g. `> **Provider Tip: Native Citations**` or
`> **Local Alternative (Ollama)**` — but the leading root must be one of the
labels below. Bare off-vocabulary labels (`Note`, `Important`, `Key Insight`,
`Looking Ahead`, …) are rejected; map them to the closest root.

| Label | Use when | Example |
|-------|----------|---------|
| **Try it** | A quick experiment or prediction the student runs themselves | `> **Try it:** Predict whether streaming changes total latency, then run both and compare.` |
| **Gotcha** | A sharp edge / anti-pattern that bites in production | `> **Gotcha:** maxOutputTokens caps output, not context. Too low truncates mid-sentence.` |
| **Before / After** | Showing impact with a short real output or metric | `> **Before / After:** the vague prompt hedges across 3 sentences; the constrained one returns the single enum value.` |
| **Decision** | Choosing between approaches (pairs with the Decide archetype) | `> **Decision:** full context vs. RAG — decide by document size and query selectivity.` |
| **Beginner Note** | Aside for less-experienced readers | `> **Beginner Note:** a "token" is roughly 4 characters of English.` |
| **Advanced Note** | Aside that goes deeper than the section requires | `> **Advanced Note:** KV-cache reuse is why prefix caching is nearly free.` |
| **Production Patterns** | A short real-world / at-scale aside | `> **Production Patterns:** log the prompt hash, never the prompt, to keep PII out of logs.` |
| **Provider Tip** | Provider-specific guidance | `> **Provider Tip:** Groq caches prompt prefixes automatically; Anthropic needs explicit cache_control.` |
| **Local Alternative** | The Ollama / offline option | `> **Local Alternative:** swap the provider for ollama('qwen3.5', { think: false }) to run offline.` |

## Section Archetypes

Vary delivery so sections don't all read the same. Target mix (guidance, not a quota):

| Archetype | ~Share | How it's delivered | Test? |
|-----------|--------|--------------------|-------|
| **Build** | ~60% | Explain → failing test → student implements | Yes — failing test, `expect()` only |
| **Explore** | ~15% | Predict-then-run: student runs a scratch experiment and observes (compare outputs, measure tokens, watch a stream) | No test required |
| **Decide** | ~15% | Trade-off driven: decision matrix / "which would you pick and why" | No test required |
| **Debug** | ~10% | Present broken/anti-pattern code; student diagnoses and fixes | Optional: a failing test the fix turns green |

Open at least one section per module with a concrete hook (a one-line failure
story or a sharp question) rather than an abstract definition.

## Hard Guardrail: Engagement Never Leaks Into Tests

Explore / Before-After experiments live in **course prose and scratch runs only**.
Tests always assert with `expect()` — never `console.log`, `console.table`,
`process.stdout.write`, or any "look at the output" inspection. An Explore section
may say "run this and watch the output," but any *test* it introduces still asserts.

## Terminology

- Message type is **`ModelMessage`** (from `'ai'`). Never `CoreMessage` — it was removed.
- **Zod v4** top-level APIs: `z.int()`, `z.email()`, `z.url()`, `z.uuid()`, `z.iso.date()`. Chaining is fine: `z.int().min(1).max(10)`.
- Token cap is **`maxOutputTokens`** (AI SDK v5), not `maxTokens`.
- Default provider in examples is **Mistral** (`mistral-small-latest`); note Groq / Anthropic / OpenAI / Ollama as alternatives where relevant.
- All LLM calls use `generateText`, `streamText`, or `Output.object()`.
