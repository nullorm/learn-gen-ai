# Course Engagement & Consistency Pass — Design

> **Historical (pre-renumber):** this design uses the old 24-module layout; module numbers and filenames here do not match the current 27-module course.

**Date:** 2026-06-03
**Branch:** `course-polish` (off `main`)
**Scope:** Editorial pass across all 24 course modules + the `/module-N` command templates + supporting docs.

## Problem

The 24-module Applied LLM Engineering course has become tedious to work through. Two confirmed root causes (chosen by the course owner from a diagnostic):

1. **Monotonous cadence.** Every `## Section` follows an identical six-beat rhythm (definition → numbered build steps → Zod/TS snippet → embedded reflection question → optional trade-off table → Beginner/Advanced note). The course owns exactly **one** variety device — Beginner/Advanced Notes, used ~268×. `**Try it:**` boxes exist in **Module 4 only**. There are no emoji, no "predict-then-run" moments, and almost no error-case ("here's the failure, now fix it") walkthroughs. The `/module-N` command template *enforces* the monotony by requiring a failing test for every section, including purely conceptual ones.

2. **Bolted-on sections.** A past "Claude Code analysis" editorial pass appended ~15 coding-agent-framework sections across M14/15/16/18/24, plus scattered LSP / deployment / permission topics. Section counts ballooned from 8 (clean baseline) to 16 (M14, +100%) and 18 (M24, +125%). The appended sections read as a grab-bag that breaks each module's narrative arc. Modules **11** and **20** were untouched and remain clean — they are the template for what "coherent" looks like.

Secondary: **inconsistencies** — end-of-module ordering drifts (M10–18 bury `Summary` last; 15 other modules use `Summary → Quiz → Exercises`); duplicated content (LSP taught 3×: M10/M19/M23; headless-CI 2×: M16/M24); section-number gaps will appear after merges.

### Explicitly *not* the problem (owner's diagnostic)

- The opening triad (`Learning Objectives` → `Why Should I Care?` → `Connection to Other Modules`) is **fine** — keep it. Do not gut it.
- Module **length** is fine — do not aggressively cut for length's sake. Bias to merge/relocate/cluster over delete.

## Decisions (locked with owner)

- **Bolt-on strategy: Approach B — relocate + consolidate + cluster.** Keep all 24 modules and their numbers stable (preserves a mid-course student's map). No new module (rejected: structural, touches `progress-types`/README/commands/skills, disrupts the 24-module map).
- **Callout style: plain blockquotes, no emoji.** Match the existing `> **Beginner Note:**` house style.
- **Pilot first**, then roll out in Part-batches.

## Goals / success criteria

- Each bloated module reads as **core arc + one signposted `## Going Further` coda**, not "random Sections 9–18".
- Cadence visibly varies: sections are a mix of Build / Explore / Decide / Debug, and ~2–4 plain callouts per module break the rhythm.
- Cross-module duplication eliminated (one canonical source + cross-references).
- End-of-module order, section numbering, callout vocabulary, and terminology are consistent across all 24.
- The teach → quiz → exercises flow, the student-builds-everything philosophy, the quiz/exercise formats, the gamification, provider-agnosticism, and all 24 module numbers are **preserved**.
- `bun test` stays green; a structural check confirms every module is well-formed.

## Non-goals

- No new module; no renumbering modules; no structural redesign of the teaching flow.
- No rewriting of `src/` implementation code (that is the student's work, on `learner`).
- No deletion of substantive content for length alone.
- No change to quiz scoring (5Q, 80%), XP/rank/badge engine, or progress CLI.

---

## Component 1 — Engagement playbook (fixes monotonous cadence)

### 1a. Section archetypes

Each `## Section` is delivered as one of four archetypes so the rhythm varies. Target mix per module (guidance, not a quota):

| Archetype | ~Share | Delivery | Test? |
|-----------|--------|----------|-------|
| **Build** | ~60% | Concept → failing test → student implements (current default) | Yes — failing test, `expect()` only |
| **Explore** | ~15% | Predict-then-run: student runs a scratch experiment and observes (compare two prompts' outputs, measure tokens, watch a stream) | No test required; verification stays prose/observation |
| **Decide** | ~15% | Trade-off driven: decision matrix / "which would you pick and why" | No test required |
| **Debug** | ~10% | Present broken or anti-pattern code; student diagnoses and fixes | Optional: a failing test that the fix makes pass |

### 1b. Callout vocabulary (plain blockquotes, no emoji)

Fixed set, used ~2–4 per module (not per section — over-use recreates monotony):

- `> **Try it:**` — quick experiment or prediction (generalize Module 4's existing pattern course-wide).
- `> **Gotcha:**` — the sharp edge / anti-pattern that bites in production.
- `> **Before / After:**` — show impact with a short real output or metric, so the "why" is visceral.
- `> **Decision:**` — when to use X vs Y (pairs with the Decide archetype).
- **Kept as-is:** `> **Beginner Note:**`, `> **Advanced Note:**`, `> **Production Patterns:**`, `> **Provider Tip:**`, `> **Local Alternative:**`.

### 1c. Hard guardrail (must not violate test rules)

Explore / Before-After experiments live in **course prose and scratch runs only** — never in test files. The no-`console.log`-in-tests, `expect()`-only-assertions rule (CLAUDE.md, commit `caaebc6`) stays inviolate. An Explore section may say "run this and watch the output," but any *test* it introduces still asserts with `expect()`.

### 1d. Opening-hook variety

Vary *section* openings (not the module triad): some sections open with a one-line failure story or a concrete question instead of an abstract definition. Light touch; do not manufacture stories where they don't fit.

## Component 2 — Surface the progression arc (near-free engagement)

`tools/progress-types.ts` already defines six Parts (`First Contact` → `Production`), `PART_BADGES`, and 8 `RANKS` — never referenced in any module. Add a one-line **"You are here"** to each module opener (within or right after Learning Objectives), e.g.:

> *Module 14 opens **Part IV: Agents & Orchestration**. Complete the Part to earn the **Agent Deployer** badge.*

Source the Part/badge strings from `progress-types.ts` so they stay accurate.

## Component 3 — Bolt-on relocation map (Approach B)

Central device: the **`## Going Further` coda**. In each bloated module, the core arc keeps numbered sections; appended advanced/framework material moves below a single `## Going Further: <theme>` heading. This creates a narrative boundary ("core module ends; optional production depth follows").

### Consolidations (eliminate cross-module duplication)

- **LSP (taught 3×):** M10 §11 "LSP-Augmented Retrieval" becomes the **canonical** treatment (lightly expanded). M19 §13 and M23 §14 are **replaced by 2–3-sentence cross-references** framed for their context (eval signal / observability signal). Net −2 sections.
- **Headless CI (2×):** M24 is canonical; M16 §12 becomes a cross-reference. Net −1 section in M16.
- **Undo/redo vs edit-history (M16 §11 vs M17 §11):** distinct domains (workflow steps vs code edits) — keep both, but M17 references M16's reversibility pattern instead of re-deriving it.

### Per-module action table

| Module | State | Action |
|--------|-------|--------|
| **M1 Setup** | clean, *completed* | Engagement + standardize. Keep exercise specs functionally stable. |
| **M2 Prompt Eng** | clean, *completed* | Engagement + standardize. Keep exercise specs stable. |
| **M3 Structured Out** | clean, *completed* | Engagement + standardize. Keep exercise specs stable. |
| **M4 Conversations** | clean, *completed* | Engagement + standardize. (Already has `Try it:` — use as house model.) |
| **M5 Long Context** | *in progress — careful* | Engagement + standardize; do **not** disrupt exercise specs while student is mid-module. |
| **M6 Streaming** | clean | **PILOT (clean module).** Full engagement + standardize. |
| **M7 Tool Use** | mild | Engagement + standardize; verify §9 security isn't over-scoped. |
| **M8 Embeddings** | clean | Engagement + standardize. |
| **M9 RAG Fundamentals** | +50% | Cluster §11 Hierarchical-Config / §12 Lazy-Loading under `## Going Further`. Keep §9 RAG Assessment (distinct from M10) and §10 Context Priority in core. Standardize. |
| **M10 Advanced RAG** | +50% | Make §11 LSP canonical (expand). Cluster §12 Diagnostic-Driven Context with it under `## Going Further`. Keep §9–10 in core. **Fix end order** (Summary→Quiz→Exercises). |
| **M11 Doc Processing** | clean | Engagement + standardize only. **Fix end order.** |
| **M12 Knowledge Graphs** | +25% | Merge the useful bit of §9–10 (LSP-as-graph, ~150 words, weak analogy) into §8; delete the rest. **Fix end order.** |
| **M13 Multimodal** | +37% | Merge §9 Image Preprocessing into §2/§7; tighten §10 Token-Cost + cross-ref M22; cluster/merge §11 File-Type-Routing. **Fix end order.** |
| **M14 Agent Fundamentals** | **+100%** | **PILOT (bloated module).** Cluster §13 Extended Thinking / §14 Plan-Build Modes / §15 Max-Steps & Hidden Agents / §16 Trace Logging under `## Going Further: Production Agent Patterns`. Keep §9–12 in core. **Fix end order.** |
| **M15 Multi-Agent** | +75% | Cluster §11 Workspace Isolation / §12 Primary-Subagent / §13 Markdown Config / §14 @Mention under `## Going Further: Claude-Code-Style Agent Systems`. Keep §9–10. **Fix end order.** |
| **M16 Workflows** | +50% | Merge thin §10 Background Execution into §7; §11 Undo/Redo cross-refs M17; §12 Headless-CI → cross-ref M24; cluster §9 Middleware under `## Going Further`. **Fix end order.** |
| **M17 Code Generation** | +50% | §11 Edit-History cross-refs M16; cluster §12 Enhanced Sandboxing under `## Going Further`. Keep §9–10. **Fix end order.** |
| **M18 Human-in-the-Loop** | +62% | Collapse §9 Declarative-Rules + §10 Permission-Modes + §12 Approval-Modes + §13 Glob-Permissions → **2 consolidated sections** under `## Going Further`. Keep §11 Denial Adaptation in core. **Fix end order.** |
| **M19 Evals** | +62% | §10 Cost-as-Eval → cross-ref M22; §13 LSP → cross-ref M10; cluster §11 Diagnostic-Capture / §12 Feature-Flags under `## Going Further`. Keep §9. (Summary already first.) |
| **M20 Fine-tuning** | clean | Engagement + standardize only. |
| **M21 Safety** | +75% | Keep §13 OS-Sandboxing / §14 Network-Isolation in M21 (safety is the right home) but cluster under `## Going Further: Autonomous-Agent Security`. Keep §9–12 in core. |
| **M22 Cost Optimization** | +62% | Cluster §13 Reasoning-Effort under `## Going Further`; keep §11 Compaction (already cross-refs M4). Receives the cost-eval cross-ref from M19. |
| **M23 Observability** | +87% | §14 LSP → cross-ref M10; cluster §12 Enhanced-Logging / §13 Context-Viz / §15 Session-Sharing under `## Going Further`. Keep §9–11. |
| **M24 Deployment** | **+125%** | Cluster §14 Multi-Target / §15 Client-Server / §16 Headless-CI (canonical) / §17 MCP-Server / §18 Multi-Frontend under `## Going Further: Distribution & Integration`. Cluster/merge §11 NDJSON. Keep §9–13 core. |

Rule of thumb: a section stays in the **core arc** if a general LLM engineer (not specifically a Claude-Code user) needs it to understand the module's topic. It moves to **`## Going Further`** if it's framework-specific or advanced-operational. It is **merged/deleted** only if thin (<~1 screen) AND redundant.

## Component 4 — Standardization sweep

- **End-of-module order → `## Summary` → `## Quiz` → `## Exercises`** in all 24 (fix M10–18, which bury Summary last; verify M1–M8 conform).
- **Renumber `## Section N:`** within each touched module so numbering is gap-free after merges/moves. `## Going Further` subsections use `###`, not continued `## Section N`.
- **Callout labels** restricted to the Component-1b vocabulary; spread `Provider Tip` / `Local Alternative` (currently M9-only) where provider-specific guidance exists elsewhere.
- **Terminology lint (grep-swept across all 24):** `ModelMessage` never `CoreMessage`; Zod v4 top-level APIs (`z.int()`, `z.email()`, …); Mistral as default provider in examples; `maxOutputTokens` (AI SDK v5) not `maxTokens`.

## Component 5 — Command layer + durable style reference

- **Update the 24 `/module-N` command templates** (currently byte-identical, 66 lines): teach by archetype (Component 1a) rather than forcing a failing test on every section; allow conceptual sections to skip the test; reference the callout vocabulary; treat `## Going Further` as optional/advanced (offer to skip or go deep per `preferences.toml` level). Preserve everything else (quiz 5Q/80%, exercises, progress commands, student-builds rules).
- **Update `CLAUDE.md` Teaching Flow** to describe archetypes + the callout vocabulary, as source of truth.
- **Add `course/STYLE.md`** documenting: the callout vocabulary, the four archetypes, the standardized module skeleton (triad → sections → `Going Further` → Summary → Quiz → Exercises), and the terminology rules. Makes the consistency durable for future edits.

## Component 6 — Sequencing, safety, verification

### Sequencing

1. **Pilot:** fully transform **M14** (worst bolt-on sprawl + rich enough to exercise every engagement device) and **M6** (clean module — shows the pure engagement layer). Add `course/STYLE.md` during the pilot so it's validated against real edits.
2. **Owner check-in:** present M14 + M6 before/after; get sign-off on the *feel* before rolling out.
3. **Roll out in Part-batches** (using the existing `PARTS` grouping): Part I (M1–3), II (M4–9), III (M10–13), IV (M14–18), V (M19–22), VI (M23–24). Update the 24 command templates + CLAUDE.md once, near the end.

### Preserve / don't touch

- **Preserve:** student-builds-everything; quiz format (multiple-choice, Easy/Medium/Hard, inline answer + explanation, 5Q/80%); exercise format (`### Exercise N`, objective, spec, file paths under `src/exercises/mNN/`, test spec); gamification; provider-agnosticism (Mistral default); all 24 module numbers.
- **Don't touch:** `src/` implementation, `tests/` student solutions, the `learner` branch, `progress.json` (gitignored), `preferences.toml` (gitignored). Never `git add -A` — the working tree has untracked student work (`src/memory/`, `tests/memory/`); stage course/doc paths explicitly.
- **Mid-course safety:** M1–4 are completed and M5 is in progress on `learner`. Keep their **exercise specifications functionally stable** so existing student code isn't invalidated; apply engagement/consistency polish to their prose freely.

### Verification

- `bun test` stays green (tests cover tooling/config, not prose — edits should not affect them; confirms no accidental breakage).
- **Structural check** (grep-based, or a small optional `tools/lint-course.ts`): every module has the triad, exactly one each of `## Summary`/`## Quiz`/`## Exercises` in that order, gap-free `## Section N` numbering, and only callouts from the approved vocabulary.
- Spot-read each transformed module end-to-end once for flow.

## Risks

- **Over-clustering** could hide content a learner expects inline. Mitigation: the "general LLM engineer needs it?" rule; bias to keep in core when unsure.
- **Renumbering churn** makes diffs noisy. Mitigation: Part-batches, one module per commit, descriptive commit messages.
- **Engagement devices drifting into tests.** Mitigation: Component 1c guardrail, restated in `course/STYLE.md` and the command template.
- **Invalidating completed-module student code.** Mitigation: freeze M1–5 exercise specs.
