# Final Consistency Audit

Audit of modules 1-6 after rewrite to remove complete implementations.
Checked for: leftover implementations, broken references, dangling prose, inconsistent style, exercise-section mismatch, import inconsistencies, cross-module references, quiz answer validity.

---

## Module 1: Setup & First LLM Calls

### Issues Found

1. **Spec-test mismatch: Exercise 2 `ProviderName` type vs test cases (lines 1242, 1257-1265)** — The specification defines `ProviderName = 'anthropic' | 'openai'` (only two providers), but the test specification calls `createModel('mistral')` and `createModel('mistral', 'mistral-large-latest')` — a provider not in the type. Either the spec should include `'mistral'` in the union type, or the tests should use `'anthropic'` or `'openai'`.

### Clean

- **No leftover implementations**: All teaching sections use signatures + prose + guiding questions. Function bodies are consistently `/* ... */` placeholders (e.g., `requireEnv`, `createModel`, `safeGenerate`, `generateTextWithRetry`, `generateWithTimeout`, `llmCall`, `classifyError`).
- **Consistent style**: Sections 1-8 all follow the same pattern: short API snippets (1-3 lines), signatures with `/* ... */`, prose describing what to build, and guiding questions.
- **No dangling prose**: No references to "as shown above" or "the implementation above".
- **Import statements**: All standard (`ai`, `@ai-sdk/mistral`, `@ai-sdk/groq`, etc.). No broken imports.
- **Cross-module references**: Valid forward references to Modules 2, 3, 4, 5+.
- **Quiz answers (Q1-Q7)**: All reference concepts (streaming vs non-streaming, environment variables, finish reasons, temperature, retry strategies, jitter, error classification). No references to removed code.
- **Exercises 1, 3, 4, 5**: Reference patterns from teaching sections correctly. Test specifications match the function signatures described.
- **Inline API patterns**: Short snippets like `const result = await generateText(...)` and `const result = streamText(...)` are appropriately illustrative (1-3 lines), not full implementations.
- **Section 8 (Resilient API Clients)**: The `calculateDelay` function (lines 1066-1070) is 5 lines but is a formula illustration, not a student-built function. Appropriate as a teaching example.

---

## Module 2: Prompt Engineering

### Issues Found

1. **Leftover implementation: `sentimentClassifier` in Section 3 (lines 239-268)** — Complete 30-line function with full `generateText` call, messages array with 3 few-shot examples, and `console.log`. Includes `sentimentClassifier().catch(console.error)` at the bottom. This is a fully runnable example, not a signature + prose pattern. Other examples in the same module (e.g., `fourComponentPrompt`, `comparePersonas`, `rulesDemo`, `formatDemo`, `structuredCoT`, `fewShotCoT`) correctly use empty bodies (`// Your implementation here`). The `sentimentClassifier` is the single exception and should be converted to match.

2. **Leftover implementation: `vulnerable` and `safer` functions in Section 7 (lines 709-733)** — Two complete functions (8 and 13 lines respectively) with full `generateText` calls. These demonstrate prompt injection defense. However, the teaching purpose here is showing a vulnerability pattern, not asking the student to build something — so this may be intentionally illustrative. Still, the `safer` function at 13 lines exceeds the >5 line threshold.

3. **Leftover implementation: `portableClassifier` in Section 8 (lines 905-925)** — Complete 21-line function with full `generateText` call, 3 few-shot examples in messages array, `temperature: 0`, and `result.text.trim().toLowerCase()`. This is a fully runnable example. Same pattern as `sentimentClassifier` — should be converted to signature + prose describing what to build.

4. **Broken reference: Exercise 4 references wrong file for `buildPrompt` (lines 1161, 1166)** — The exercise says "Build a named template registry on top of the `buildPrompt` and `createTemplate` functions you already built in `src/prompts/templates.ts`." However, `buildPrompt` and `createTemplate` are defined in Section 5 under `src/prompts/builder.ts` (line 518-521), not `src/prompts/templates.ts`. The import on line 1166 says `Import buildPrompt from '../prompts/templates.js'` which should be `'../prompts/builder.js'`. The file `src/prompts/templates.ts` exports `codeReviewPrompt` and `summarizePrompt` (different functions).

5. **Inconsistent style: Section 3 vs Sections 1-2, 4-6** — Sections 1-2 and 4-6 consistently use `// Your implementation here` for function bodies. Section 3 has the complete `sentimentClassifier` while `entityExtractor` in the same section correctly uses `// Your implementation here`. This inconsistency exists within a single section.

### Clean

- **No dangling prose**: No references to "as shown above" or "the implementation above".
- **Cross-module references**: Valid references to Modules 1, 3, 5, 7, 19, 21.
- **Quiz answers (Q1-Q7)**: All reference concepts (prompt components, few-shot pattern, chain-of-thought mechanics, prompt injection, provider differences, composition vs templates, hierarchical rules). No references to removed code.
- **Exercises 1-3, 5-6**: Reference patterns from teaching sections correctly.
- **Production Pattern sections (9-10)**: Mostly conceptual. Section 9 has a short `composeSystemPrompt` function (5 lines, lines 960-965) that is at the threshold but serves as a formula illustration. Section 10 is purely conceptual with no function bodies.
- **Import statements**: All standard (`ai`, `@ai-sdk/mistral`) except the broken Exercise 4 import noted above.

---

## Module 3: Structured Output

### Issues Found

1. **Leftover implementation: `streamArticleAnalysis` in Section 9 (lines 689-727)** — Complete 39-line function with full `streamText` + `Output.object()` call, `for await` loop over `partialOutputStream`, conditional field display, `await result.output`, JSON output, and a `main()` function with test data. This is a fully runnable script. The section also provides `streamWithCallbacks` (lines 752-785) as another complete 34-line implementation with `onFinish` callback, `updateCount` tracking, and `main()` entrypoint.

2. **Inconsistent style: Section 9 vs Sections 1-8** — Sections 1-8 consistently use signature + prose + guiding questions. Section 9 (Streaming Structured Output) provides two complete, runnable examples (`streamArticleAnalysis` and `streamWithCallbacks`) that the student could copy directly. Other sections in the module that teach streaming patterns (e.g., Section 3's `Output.object()` pattern) correctly show only the API pattern and ask the student to build the function.

### Clean

- **No dangling prose**: Clean throughout.
- **No broken references**: All imports are standard (`ai`, `@ai-sdk/mistral`, `zod`).
- **Cross-module references**: Valid references to Modules 1, 2, 7, 9, 14.
- **Quiz answers (Q1-Q7)**: All reference concepts (Output.object advantage, .describe() method, enum strictness, optional vs nullable, arithmetic unreliability, partial objects, schema descriptions). No references to removed code.
- **Exercises 1-5**: Reference patterns from teaching sections correctly. Exercise 4 (streaming structured output) does not conflict with the Section 9 implementations because it uses a different schema (MovieAnalysis vs ArticleSchema) and adds timing requirements.
- **Exercise 5 test specification**: Properly defined with schema shape assertions and LLM-based extraction tests.
- **Schema examples in Sections 2-7**: Appropriately illustrative (defining schemas is declarative, not implementation code). These are patterns for the student to understand and reuse, not function bodies.
- **Section 10**: Purely conceptual — guidelines for `.describe()` usage with short illustrative snippets. Clean.

---

## Module 4: Conversations & Memory

### Issues Found

1. **Implicit file path for teaching section code (Sections 3-8)** — The teaching sections define interfaces, classes, and functions (`ConversationState`, `MemoryStrategy`, `ConversationManager`, `getWindowedMessages`, `createSlidingWindowStrategy`, `needsSummarization`, `getSummarizationText`, `createSummarizingStrategy`, `createHybridStrategy`, `saveConversation`, `loadConversation`, `listConversations`, `deleteConversation`, `estimateTokens`, `estimateTokensAccurate`, `countMessageTokens`, `getRemainingBudget`, `TokenAwareConversation`) without specifying file paths for most of them. This contrasts with Modules 1-3 which specify explicit file paths. The Exercise Prep references `src/memory/runner.ts` and Exercise 1 Stage 2 references `countMessageTokens` from `src/memory/tokens.ts`, implying the student should organize code into `src/memory/` but the teaching sections never state this explicitly.

2. **Exercise 2 import assumes undeclared file path (line 1109)** — `import type { ConversationManager } from '../../memory/manager.js'` assumes the student placed the `ConversationManager` class in `src/memory/manager.js`. The teaching section (Section 3) defines the class but never tells the student to create `src/memory/manager.ts`. The student must infer this from the exercise import path.

### Clean

- **No leftover implementations**: All teaching sections use signatures + prose + guiding questions. The `ConversationManager` class (lines 278-293) is signatures only. Strategy factories return `MemoryStrategy` objects described in prose. Helper functions (`getWindowedMessages`, `needsSummarization`, `getSummarizationText`) are all signatures without bodies.
- **Consistent style**: Sections 1-8 all follow the same pattern. Sections 9-12 ("Production Patterns") are purely conceptual with no function bodies — only description and short illustrative snippets (2-3 lines). This is the cleanest Production Patterns implementation across all modules.
- **No dangling prose**: Clean throughout.
- **No broken references**: All imports are standard (`ai`, `@ai-sdk/mistral`, `zod`, `node:fs/promises`).
- **Cross-module references**: Valid references to Modules 1, 2, 3, 5, 6, 7, 8.
- **Quiz answers (Q1-Q7)**: All reference concepts (statelessness, window sizes, summarization cost, newest-to-oldest processing, fact sheet purpose, extraction vs summarization, microcompaction targets). No references to removed code.
- **Exercises 1-4**: Well-structured with clear stage progression (Exercise 1) and proper test specifications (Exercises 2-4). Exercise Prep pattern (shared runner) is good engineering.
- **The summarization loop snippet (lines 413-422)**: This is a 10-line code block showing how to call `needsSummarization` and `generateText` together. It is not a student-built function — it is glue code showing how the pieces connect. This is appropriate as a usage example, similar to showing how `generateText` is called.

---

## Module 5: Long Context & Caching

### Issues Found

1. **Inconsistent exercise file paths: Exercises 3-4 (lines 853, 877)** — Exercises 3-4 use `src/memory/exercises/auto-compact.ts` and `src/memory/exercises/context-budget.ts`, which breaks the convention established in Modules 1-4 of `src/exercises/mNN/exNN-*.ts`. This is the only module (other than Module 6) that uses a different path pattern for exercises.

2. **Missing exercise file paths: Exercises 1-2** — Exercises 1 and 2 do not specify file paths at all, unlike all exercises in Modules 1-4 which explicitly state the file to create. Exercise 1 has starter code but no `Create a file...` instruction. Exercise 2 has requirements but no file path.

### Clean

- **No leftover implementations**: All teaching sections use function signatures without bodies. Functions like `analyzeDocument`, `structuredLongContext`, `fullContextQA`, `chunkDocument`, `queryWithCaching`, `cachedDocumentQA`, `preprocessDocument`, `compressionStats`, `parseDocumentSections`, `selectRelevantSections`, `compressContext`, `structuredLongDocument`, `buildHierarchicalDocument`, `hierarchicalQA`, `calculateGroqCost`, `compareCachingSavings`, `recommendStrategy`, `hybridDocumentQA`, `estimateKVCacheSize` are all signatures + prose.
- **Consistent style**: Sections 1-8 all follow the same signature + prose + guiding questions pattern. Sections 9-12 ("Production Patterns") are conceptual with short illustrative snippets (2-5 lines). Clean throughout.
- **The `CachedConversation` class (lines 206-213)**: Signatures only — constructor and `send` method have no bodies. Clean.
- **No dangling prose**: Clean throughout.
- **No broken references**: All imports are standard (`ai`, `@ai-sdk/mistral`, `@ai-sdk/groq`, `node:fs/promises`).
- **Cross-module references**: Valid references to Modules 2, 4, 6, 8, 9. Module 3 reference in Section 5 (structured extraction) is valid.
- **Quiz answers (Q1-Q7)**: All reference concepts (caching benefits, page equivalents, lost-in-the-middle, break-even analysis, RAG vs caching, cache-friendly ordering, auto-compact design). No references to removed code.
- **Caching comparison table (lines 223-233)**: Appropriate reference data, not implementation code.
- **KV cache formula (Section 4)**: The formula description is prose with a signature — student builds the function. Clean.

---

## Module 6: Streaming & Real-time

### Issues Found

1. **Inconsistent exercise file paths: Exercises 3-4 (lines 852, 875)** — Exercises 3-4 use `src/streaming/exercises/cancellation.ts` and `src/streaming/exercises/streaming-tools.ts`, which breaks the convention established in Modules 1-4 of `src/exercises/mNN/exNN-*.ts`. Same issue as Module 5.

2. **Missing exercise file paths: Exercises 1-2** — Exercises 1 and 2 do not specify file paths at all. Exercise 1 has starter code but no `Create a file...` instruction. Exercise 2 has starter code but no file path.

3. **Forward reference to Module 7: Exercise 4 (line 862)** — Exercise 4 says "Use `streamText` with tools defined (reuse tools from Module 7 or define simple ones)." Since Module 6 comes before Module 7, students have not yet learned tool use. The "or define simple ones" escape clause mitigates this, but it is still a forward dependency. The exercise also references `toolCallStreaming` and `toolCall` events which are Module 7 concepts not taught in Module 6.

### Clean

- **No leftover implementations**: All teaching sections use function signatures without bodies. Functions like `measureNonStreaming`, `measureStreaming`, `streamConversation`, `measureStream`, `streamRecipe`, `streamAnalysis`, `safeDisplay`, `handleChatStream`, `streamToFile`, `rateLimit`, `bufferBySentence`, `stripMarkdown`, `mapStream`, `annotateStream`, `teeStream`, `safeStream`, `resilientStream`, `streamWithCancellation`, `typewriter`, `progressiveDisclosure`, `streamWithStatus` are all signatures + prose.
- **Consistent style**: Sections 1-8 all follow the same pattern. Callback examples in Section 2 (lines 135-163) show the API pattern with short bodies (just `if` checks and `console.log`) — these are API usage examples, not student implementations. Appropriate.
- **Sections 9-12 ("Production Patterns")**: Conceptual with short illustrative snippets. Section 11 has a 7-line buffering pattern (lines 562-576) that is an inline code contrast (before/after), not a student function. Section 12 has a 7-line TTY detection pattern. Both are appropriate illustrations.
- **No dangling prose**: Clean throughout.
- **No broken references**: All imports are standard (`ai`, `@ai-sdk/mistral`, `zod`).
- **Cross-module references**: Valid references to Modules 1, 3, 4, 5, 7.
- **Quiz answers (Q1-Q7)**: All reference concepts (TTFT, streaming vs non-streaming, AbortController, partial objects, SSE protocol, NDJSON advantages, abort controller trees). No references to removed code.
- **The `streamText` callbacks example (lines 135-163)**: This is an API pattern demonstration, not a student-built function. The code block shows how to wire `onChunk`, `onFinish`, and `onError` callbacks — this is reference documentation for the API, appropriate to show inline.

---

## Summary of Issues Across Modules 1-6

### Leftover Implementations

| Module | Location  | Function                | Lines                                                         |
| ------ | --------- | ----------------------- | ------------------------------------------------------------- |
| 2      | Section 3 | `sentimentClassifier`   | ~30 lines, complete with few-shot messages and console output |
| 2      | Section 7 | `safer`                 | ~13 lines, complete prompt injection defense                  |
| 2      | Section 8 | `portableClassifier`    | ~21 lines, complete few-shot classifier with messages         |
| 3      | Section 9 | `streamArticleAnalysis` | ~39 lines, complete streaming + Output.object demo            |
| 3      | Section 9 | `streamWithCallbacks`   | ~34 lines, complete streaming with callbacks demo             |

### Broken References

| Module | Location                           | Issue                                                                                                |
| ------ | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1      | Exercise 2 (lines 1242, 1257-1265) | `ProviderName` type excludes `'mistral'` but tests call `createModel('mistral')`                     |
| 2      | Exercise 4 (lines 1161, 1166)      | References `buildPrompt` in `src/prompts/templates.ts` but it is defined in `src/prompts/builder.ts` |

### Inconsistent Exercise File Paths

| Module | Exercises     | Path Used                  | Expected Convention                 |
| ------ | ------------- | -------------------------- | ----------------------------------- |
| 5      | Exercises 3-4 | `src/memory/exercises/`    | `src/exercises/m05/`                |
| 6      | Exercises 3-4 | `src/streaming/exercises/` | `src/exercises/m06/`                |
| 5      | Exercises 1-2 | No path specified          | Should specify `src/exercises/m05/` |
| 6      | Exercises 1-2 | No path specified          | Should specify `src/exercises/m06/` |

### No Issues Found In

- **Dangling prose**: No instances of "as shown above", "the implementation above", etc. across all 6 modules.
- **Quiz answer validity**: All 42 quiz answers (7 per module) reference concepts and principles, not removed code.
- **Cross-module references**: All valid. No references to implementations removed from other modules.
- **Import inconsistencies** (except the M02 Exercise 4 case): All imports use standard packages.
- **Module 4 Production Patterns (Sections 9-12)**: Cleanest implementation -- purely conceptual with no function bodies.
