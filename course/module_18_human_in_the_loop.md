# Module 18: Human-in-the-Loop

## Learning Objectives

- Understand why human oversight is essential for trustworthy AI systems
- Implement approval flows where agents propose actions and humans approve or reject them
- Build confidence-based routing that auto-approves low-risk actions and escalates high-risk ones
- Integrate human feedback to improve future agent behavior
- Design active learning patterns where agents ask for help on uncertain cases
- Build CLI-based review interfaces for human decision-making
- Create audit trails that log all decisions for compliance and debugging
- Handle graceful degradation when no human is available

---

## Why Should I Care?

Autonomous agents are impressive, but they make mistakes. They hallucinate, misinterpret instructions, and sometimes take actions that are technically correct but contextually wrong. In high-stakes domains — financial transactions, medical recommendations, legal documents, customer communications, data deletion — an unchecked agent is a liability.

Human-in-the-loop (HITL) patterns add guardrails that keep agents useful without making them dangerous. The agent does the heavy lifting (research, drafting, analysis) while a human makes the final call on actions that matter. This is not a limitation — it is a feature. The best AI systems combine machine speed with human judgment.

This module teaches you how to add human oversight at the right points in an agent workflow: approval gates for high-stakes actions, confidence-based routing for efficiency, feedback loops for improvement, and audit trails for accountability.

---

## Connection to Other Modules

- **Module 14 (Agent Fundamentals)** provides the agent loop where HITL checkpoints are inserted.
- **Module 15 (Multi-Agent Systems)** uses HITL for orchestrator-level approvals.
- **Module 16 (Workflows & Chains)** adds approval gates between chain steps.
- **Module 17 (Code Generation)** uses human review before executing generated code.

---

## Section 1: Why Human-in-the-Loop?

### Three Reasons for Human Oversight

1. **Trust**: Users and stakeholders trust AI systems more when they know a human is in the loop for important decisions
2. **Accuracy**: Humans catch errors that automated systems miss — especially contextual errors, cultural nuances, and edge cases
3. **Compliance**: Many industries (healthcare, finance, legal) require human oversight for certain decisions by regulation

### The Autonomy Spectrum

Not every action needs human approval. Place each action on the autonomy spectrum:

| Level          | Description                      | Example                                 |
| -------------- | -------------------------------- | --------------------------------------- |
| Full autonomy  | Agent acts without asking        | Formatting text, looking up public data |
| Notify only    | Agent acts and informs the human | Sending a routine notification          |
| Approve/reject | Agent proposes, human decides    | Sending an email, modifying a database  |
| Human does it  | Agent assists but human executes | Financial transactions, legal filings   |

Build a function `classifyActionRisk` that uses structured output to classify an action into the appropriate level. Define a schema:

```typescript
const riskSchema = z.object({
  action: z.string(),
  riskLevel: z.enum(['low', 'medium', 'high', 'critical']),
  autonomyLevel: z.enum(['full_autonomy', 'notify_only', 'approve_reject', 'human_executes']),
  reasoning: z.string(),
  reversible: z.boolean().describe('Can this action be undone?'),
})
```

The function signature should be:

```typescript
async function classifyActionRisk(action: string, context: string): Promise<z.infer<typeof riskSchema>>
```

The function should use `generateText` with `Output.object({ schema: riskSchema })`. The system prompt needs to define what each risk level means and how it maps to an autonomy level.

Think about these questions:

- What makes an action "critical" vs. "high" risk? Consider reversibility -- can the action be undone? What about blast radius -- does it affect one record or thousands?
- How do you define clear criteria for each risk level so the model classifies consistently? What examples would you put in the system prompt?
- Why is the `reversible` field important for risk assessment? How would a reversible high-risk action differ from an irreversible one?

> **Beginner Note:** Human-in-the-loop is not about distrusting AI — it is about using AI and humans where each excels. AI is fast and tireless. Humans have judgment and contextual understanding. The best systems combine both.

---

## Section 2: Approval Flows

### Agent Proposes, Human Approves

The basic HITL pattern: the agent generates a proposed action, presents it to a human, and only executes if approved. You need these types:

```typescript
interface ProposedAction {
  type: string
  description: string
  details: Record<string, unknown>
  risk: 'low' | 'medium' | 'high' | 'critical'
  reasoning: string
}

interface ApprovalDecision {
  approved: boolean
  feedback?: string
  modifiedAction?: Partial<ProposedAction>
}
```

Build two things:

**`requestApproval(action: ProposedAction): Promise<ApprovalDecision>`** — A CLI-based function that displays the proposed action and asks the human to approve, reject, or modify it. Use Node's `readline.createInterface` for interactive input.

How do you structure the readline flow? Each response path (approve/reject/modify) needs different follow-up input. How do you wrap the callback-based `rl.question` in a Promise so the function can be awaited?

**`agentWithApproval(task: string): Promise<string>`** — An agent that uses `generateText` with tools, where high-risk tools require approval before executing and low-risk tools run automatically. Think about: how does a tool's `execute` function decide whether to request approval? What message should the tool return to the LLM if the human rejects the action?

> **Advanced Note:** In production, approval requests are typically sent to a queue (Slack, email, web dashboard) rather than blocking on stdin. The agent pauses, saves its state, and resumes when the approval comes back. This requires persistent state storage which we cover in the audit trail section.

### Programmatic Approval Callbacks

For automated testing and non-CLI environments, use callback-based approval. Define a callback type:

```typescript
type ApprovalCallback = (action: { toolName: string; args: Record<string, unknown> }) => Promise<{
  approved: boolean
  feedback?: string
}>
```

Build a function `agentWithCallback(task: string, onApprovalNeeded: ApprovalCallback)` that returns `{ response: string; approvals: Array<{ action: string; approved: boolean }> }`. The agent should use the callback to check with the caller before executing certain tools.

Think about: how do you decide which tools need approval? What data structure lets you look this up efficiently in each tool's `execute`? Why is the callback approach better for automated testing than the CLI approach? How would you write a test that verifies the agent correctly handles a rejection?

---

## Section 3: Confidence-Based Routing

### Auto-Approve High Confidence, Escalate Low Confidence

Not every action needs human review. Route based on the agent's confidence. Define the configuration and proposal types:

```typescript
interface ConfidenceConfig {
  autoApproveThreshold: number // Above this -> auto-approve (e.g., 0.9)
  escalateThreshold: number // Below this -> require human review (e.g., 0.7)
  // Between the two thresholds -> notify but proceed
}

const proposalSchema = z.object({
  action: z.string().describe('The proposed action'),
  confidence: z.number().min(0).max(1).describe('How confident you are this is the right action (0-1)'),
  reasoning: z.string().describe('Why this action and this confidence level'),
  alternatives: z.array(z.string()).describe('Other actions you considered'),
  riskAssessment: z.string().describe('What could go wrong'),
})

type Proposal = z.infer<typeof proposalSchema>
```

Build `confidenceBasedRouting(task, config, onApproval)` that returns `{ result, routing, proposal }` where `routing` is `'auto_approved' | 'notified' | 'human_reviewed'`. The function uses the LLM to propose an action with a confidence score, then routes based on that score and the config thresholds.

Think about:

- How do you get the LLM to produce an honest confidence score? What system prompt instructions prevent it from always reporting high confidence?
- What happens in each routing zone? Above the auto-approve threshold, between the two thresholds, and below the escalation threshold -- each needs different behavior.
- If the human rejects an escalated action, what should the function return?

### Dynamic Threshold Adjustment

Different action types deserve different thresholds. Build an `ActionThresholds` interface and a `getThresholds` function:

```typescript
interface ActionThresholds {
  readOnly: { autoApprove: number; escalate: number }
  modification: { autoApprove: number; escalate: number }
  deletion: { autoApprove: number; escalate: number }
  financial: { autoApprove: number; escalate: number }
  communication: { autoApprove: number; escalate: number }
}
```

What should the default thresholds be for each action type? Consider the cost of a mistake: a wrong read is cheap, a wrong financial transaction is expensive. How would you let callers override specific thresholds while keeping defaults for others? What TypeScript utility type supports partial overrides of a typed object?

> **Beginner Note:** Confidence-based routing is like a triage system in a hospital. Minor issues (high confidence, low risk) are handled automatically. Serious cases (low confidence, high risk) go to a specialist (human reviewer). This keeps the system efficient while maintaining safety.

---

## Section 4: Feedback Integration

### Learning from Human Corrections

When a human rejects or modifies an agent's proposal, that feedback is valuable. Store it and use it to improve future behavior. Define the feedback entry type:

```typescript
interface FeedbackEntry {
  id: string
  timestamp: number
  task: string
  proposedAction: string
  decision: 'approved' | 'rejected' | 'modified'
  humanFeedback?: string
  modifiedAction?: string
  category: string
}
```

Build a `FeedbackStore` class that stores feedback entries and makes them available for prompt injection. The class needs methods to add entries, query recent feedback by category, analyze rejection patterns, and calculate approval rates.

The most important method is one that formats recent feedback as prompt context -- converting past human decisions into text the LLM can learn from. How would you format approval/rejection history so the model understands what worked and what did not?

Now build `feedbackAwareAgent(task, category, feedbackStore, onApproval)` that incorporates feedback history into its system prompt. Think about:

- How do you inject the feedback context into the system prompt? Where does it go relative to the main instructions?
- What should happen when the approval rate drops below a threshold? How do you signal to the model that it needs to be more conservative?
- How does in-context learning from feedback differ from fine-tuning? What are the advantages and limitations of each?

> **Advanced Note:** In production, feedback stores should be persistent (database) and shared across agent instances. Use the feedback to fine-tune prompts, adjust confidence thresholds, and identify systematic issues. If a certain category of action is consistently rejected, that signals a prompt engineering problem.

---

## Section 5: Active Learning

### Agent Asks for Help on Uncertain Cases

Instead of waiting for the human to reject a proposal, a proactive agent can identify uncertainty and ask for guidance upfront. Define the types:

```typescript
interface ClarificationRequest {
  question: string
  options: string[]
  context: string
  impact: string
}

const uncertaintySchema = z.object({
  confident: z.boolean().describe('Are you confident enough to proceed?'),
  uncertainties: z
    .array(
      z.object({
        question: z.string().describe('What you are uncertain about'),
        options: z.array(z.string()).describe('Possible interpretations or options'),
        impact: z.string().describe('How this uncertainty affects the outcome'),
      })
    )
    .describe('Areas of uncertainty, if any'),
  proposedAction: z.string().describe('What you would do if proceeding'),
})
```

Build `activeLearningAgent(task, onClarification)` that works in two phases: first analyzing whether the task is clear enough, then acting (possibly after asking for clarification).

Think about:

- How does the system prompt help the model identify uncertainty? What common sources of ambiguity should it check for?
- If the model reports uncertainty, how do you convert its structured output into `ClarificationRequest` objects for the human?
- After receiving answers, how do you incorporate them into the follow-up prompt?
- Why is it better for the agent to ask upfront rather than act and be rejected? Compare the cost of a clarification question vs. a full rejected draft.

### Uncertainty Detection Heuristics

Beyond asking the model about its own confidence, build a `detectUncertainty(task: string): Promise<UncertaintySignals>` function that uses structured output to check for specific patterns:

```typescript
interface UncertaintySignals {
  hasAmbiguousPronouns: boolean
  missingNumbers: boolean
  multiplePossibleInterpretations: boolean
  domainSpecificTerms: boolean
  vagueDateReferences: boolean
  overallUncertainty: number // 0-1
}
```

Define a Zod schema matching this interface with descriptive `.describe()` strings for each field. The model should analyze the task text for ambiguity and return structured signals.

How would you use these signals downstream? Could the `overallUncertainty` score feed into the confidence-based routing from Section 3? What threshold would trigger a clarification request vs. a human escalation?

> **Beginner Note:** Active learning is the agent equivalent of asking a clarifying question before starting work. A junior employee who asks "Just to confirm, you mean the Q1 report for Acme Corp, right?" is more valuable than one who confidently delivers the wrong report.

---

## Section 6: Review Interfaces

### CLI-Based Review

Build a command-line review interface for human decision-making. Define the types:

```typescript
interface ReviewItem {
  id: string
  type: string
  content: string
  metadata: Record<string, unknown>
  risk: 'low' | 'medium' | 'high' | 'critical'
}

interface ReviewDecision {
  itemId: string
  decision: 'approve' | 'reject' | 'modify' | 'skip'
  feedback?: string
  modifiedContent?: string
  timestamp: number
  reviewerNote?: string
}
```

Build a `CLIReviewer` class that presents review items to a human via the terminal and collects decisions. The class needs methods for reviewing a single item and reviewing a batch with progress tracking.

Think about:

- How do you wrap `rl.question` in a Promise so the class methods can be async? Where should the readline interface be created and destroyed?
- For `reviewItem`, how do you display the item clearly and handle different decision paths (approve/reject/modify/skip)? Each path needs different follow-up input.
- For `reviewBatch`, what progress information is useful to show after each decision? How do you handle cleanup if the reviewer quits mid-batch?

### Programmatic Review Interface

For integrations with web dashboards or APIs, build a queue-based system:

```typescript
interface ReviewQueue {
  items: Map<string, ReviewItem & { status: 'pending' | 'reviewed' }>
  decisions: Map<string, ReviewDecision>
  waiters: Map<string, (decision: ReviewDecision) => void>
}
```

Build functions for creating, submitting to, and deciding on review queues: `createReviewQueue()`, `submitForReview(queue, item)`, `submitDecision(queue, decision)`, and `getPendingItems(queue)`.

The key pattern is a Promise-based bridge: when an item is submitted for review, the caller gets a Promise that resolves only when someone submits a decision. Think about:

- How do you create a Promise whose `resolve` function is stored externally and called later by a different function? This is the "deferred promise" pattern.
- What should `submitForReview` do if the item has already been reviewed? What about duplicate submissions?
- How does this architecture decouple the agent (which submits and waits) from the reviewer (which decides asynchronously)?

---

## Section 7: Audit Trails

### Logging Decisions for Compliance

Every decision in a HITL system should be logged with enough detail to reconstruct what happened and why. Define the entry type:

```typescript
interface AuditEntry {
  id: string
  timestamp: number
  sessionId: string
  eventType:
    | 'action_proposed'
    | 'action_approved'
    | 'action_rejected'
    | 'action_modified'
    | 'action_executed'
    | 'error'
    | 'clarification_requested'
    | 'clarification_received'
  agentId: string
  reviewerId?: string
  action: string
  details: Record<string, unknown>
  reasoning?: string
  feedback?: string
  taskDescription: string
  confidenceScore?: number
  riskLevel?: string
  policyApplied?: string
  complianceCheck?: boolean
}
```

Build an `AuditLog` class that records every decision point in a HITL workflow. The class needs methods for logging events (auto-generating IDs, timestamps, and session context), querying entries with filters, summarizing a session's statistics, and exporting the full log as JSON.

Now build `auditedAgent(task, audit, onApproval)` that wraps a simple agent with full audit logging at each decision point.

Think about:

- What events should be logged? Consider the lifecycle: task received, action proposed, human decision, action executed, errors.
- What auto-generated fields make the log useful without burdening the caller? Which fields should the caller provide vs. which should the class fill in?
- How do you calculate the approval rate from the event types? What edge case arises when no actions have been proposed yet?
- What fields are essential for compliance (GDPR, HIPAA) vs. nice-to-have for debugging?

> **Advanced Note:** For regulatory compliance (GDPR, HIPAA, SOX), audit logs typically need to be immutable (append-only), timestamped with a trusted clock, and stored in a tamper-evident manner. Consider using a dedicated audit logging service or write-once storage in production.

---

## Section 8: Graceful Degradation

### When No Human is Available

What happens when the agent needs approval but no human is online? Production systems need fallback behaviors. Define the configuration:

```typescript
interface DegradationConfig {
  approvalTimeoutMs: number
  fallbackBehavior: 'queue' | 'auto_approve_low_risk' | 'reject_all' | 'use_defaults'
  notifyOnDegradation: boolean
  maxQueueSize: number
}

interface DegradedAction {
  action: string
  originalRisk: string
  degradedDecision: string
  reason: string
}
```

Build a `GracefulDegradation` class that handles approval requests when humans may be unavailable or slow to respond. The class needs a method to request approval (trying the human first, falling back if unavailable or timed out), a private fallback method that implements the configured strategy, and a way to retrieve queued items.

Think about:

- How do you implement a timeout on the human's response? What concurrency primitive lets you race a callback against a timer?
- The config specifies four fallback strategies (`queue`, `auto_approve_low_risk`, `reject_all`, `use_defaults`). What does each one do? Which ones are safe for critical-risk actions?
- Why must `auto_approve_low_risk` NEVER auto-approve critical-risk actions? What is the cost/benefit analysis?
- What happens when the queue is full? Should new items be rejected or should old items be evicted?

### Escalation Chains

When the primary reviewer is unavailable, escalate to backups. Build an `escalatingApproval` function:

```typescript
interface Reviewer {
  id: string
  name: string
  available: () => boolean
  review: (action: string) => Promise<string>
}

async function escalatingApproval(
  action: string,
  reviewers: Reviewer[],
  timeoutMs?: number
): Promise<{ decision: string; reviewerId: string; escalated: boolean }>
```

The function should try reviewers in priority order, skipping unavailable ones and timing out slow ones. Think about:

- How do you determine whether the final decision came from an escalated reviewer vs. the primary one?
- What should happen if all reviewers are either unavailable or time out? What is the safest default?
- What should happen to queued actions when a human comes back online? How would you design a notification mechanism?

> **Beginner Note:** Graceful degradation means your system keeps working even when ideal conditions are not met. Just as a car still works without air conditioning (less comfortable but functional), your agent system should still work without a human reviewer (less capable but safe).

---

> **Production Patterns** — The following sections explore how the concepts above are applied in production systems. These are shorter and more conceptual than the hands-on sections above.

## Section 9: Declarative Permission Rules

### Rules Instead of Code

Production permission systems define access control declaratively — as data, not logic. Each rule specifies a pattern (glob or regex) and a decision (allow, deny, or ask). Rules are checked before every tool execution.

```typescript
interface PermissionRule {
  pattern: string // glob pattern for the resource
  decision: 'allow' | 'deny' | 'ask'
}
```

A rule set is evaluated in order, and the last matching rule wins. This lets you layer broad defaults with specific overrides:

```typescript
const rules: PermissionRule[] = [
  { pattern: '**', decision: 'ask' }, // default: ask for everything
  { pattern: 'src/**/*.ts', decision: 'allow' }, // allow reading project files
  { pattern: '.env*', decision: 'deny' }, // never touch env files
]
```

Declarative rules are easier to audit, version, and share than imperative permission logic. A project can ship baseline rules and let users override them.

---

## Section 10: Permission Modes

### Configurable Autonomy Levels

The autonomy spectrum from Section 1 becomes concrete through permission modes — named configurations that change which rules apply:

- **Restrictive** — Ask for most operations. Safe for exploration and untrusted tasks.
- **Normal** — Auto-approve reads and low-risk writes. Ask for shell commands, network access, and deletions.
- **Autonomous** — Auto-approve most operations within the project directory. Deny anything outside it.

Switching modes changes the active rule set, not the code. The permission engine stays the same; only the rules differ.

> **Key Insight:** Modes let you match the autonomy level to the task. A code review task needs restrictive mode; a scaffolding task can use autonomous mode. The user chooses the mode, not the agent.

---

## Section 11: Denial Adaptation

### Learning from "No"

When a human denies an action, the agent must not retry the same operation. Instead, it should:

1. Record the denial and the reason (if provided)
2. Include the denial in the next LLM call as context
3. Propose an alternative approach

This creates a feedback loop where the agent adapts its behavior within the session. The denial is appended to the conversation history, not used to truncate it — the agent remembers what it tried and why it was rejected.

```typescript
// After denial, the next message to the LLM includes:
// "The user denied your request to delete the file. Reason: 'Use git revert instead'. Propose an alternative."
```

The pattern is simple: denied actions go into a "do not retry" set for the session. If the agent generates the same tool call with the same parameters, it is blocked before reaching the user.

---

## Section 12: Three Approval Modes

### Suggest, Auto-Edit, and Full-Auto

Production coding agents implement three distinct autonomy levels that go beyond simple ask/don't-ask:

1. **Suggest mode** — Read-only. The agent can read files and analyze code, but requires explicit approval for all writes and all shell commands. Use this for exploration and review tasks.

2. **Auto-Edit mode** — File edits are auto-approved, but shell commands require approval. This is the sweet spot for code generation tasks where the agent should freely write code but not execute arbitrary commands.

3. **Full-Auto mode** — All operations are auto-approved, but with a critical safety constraint: network access is disabled and file writes are limited to the working directory. The agent can do anything locally but cannot exfiltrate data or reach external services.

The full-auto pattern demonstrates **compensating controls** — relaxing one constraint (human approval) while tightening another (network/filesystem scope). Full autonomy is safe when the blast radius is contained.

> **Advanced Note:** The network-disable-in-full-auto pattern is a significant safety innovation. No network means no data exfiltration, no supply chain attacks, and no unintended external side effects — even if the agent goes off the rails.

---

## Section 13: Glob-Based Command Permissions

### Fine-Grained Shell Command Control

For shell command execution, glob patterns provide precise control over what is allowed, what needs confirmation, and what is blocked:

```typescript
const commandRules = [
  { pattern: 'git status *', decision: 'allow' },
  { pattern: 'git push *', decision: 'ask' },
  { pattern: 'rm -rf *', decision: 'deny' },
  { pattern: 'npm test *', decision: 'allow' },
  { pattern: 'curl *', decision: 'deny' },
]
```

Rules are evaluated in order with last-match-wins semantics. This allows layering — a project defines baseline rules, and the user adds overrides on top. Glob patterns are more readable than regex and compose naturally.

The matcher compares the full command string against each rule's pattern. Wildcards (`*`) match any sequence of characters within a single argument. Double wildcards (`**`) are not typically needed for flat command strings.

---

## Quiz

### Question 1 (Easy)

What is the primary purpose of human-in-the-loop patterns in AI systems?

- A) To make AI systems slower and more expensive
- B) To add human oversight for trust, accuracy, and compliance
- C) To replace AI systems with human labor
- D) To reduce the number of API calls

**Answer: B** — HITL patterns add human judgment at critical decision points. This builds trust (users know a human checks important decisions), improves accuracy (humans catch contextual errors AI misses), and ensures compliance (regulations often require human oversight for certain actions). HITL systems are not about replacing AI but about combining machine speed with human judgment.

---

### Question 2 (Medium)

In confidence-based routing, what happens when the agent's confidence is between the auto-approve threshold and the escalation threshold?

- A) The action is rejected
- B) The action is auto-approved
- C) The agent proceeds with the action and notifies the human (but does not wait for approval)
- D) The system crashes

**Answer: C** — The "notify only" zone is between the escalation threshold and the auto-approve threshold. The agent proceeds with the action but logs it and notifies a human for awareness. This provides a middle ground between full autonomy (high confidence) and blocking on human approval (low confidence). It is appropriate for actions that are probably right but worth a human glancing at.

---

### Question 3 (Medium)

How does storing human feedback help improve agent behavior over time?

- A) The feedback is used to fine-tune the model weights in real-time
- B) Patterns from rejected proposals are included in the system prompt, teaching the agent what to avoid
- C) The feedback automatically creates new tools for the agent
- D) Human feedback has no effect on future agent behavior

**Answer: B** — Feedback entries (especially rejections and modifications) are formatted as context in the agent's system prompt. When the agent sees that similar proposals were rejected with specific reasons, it adjusts its future proposals accordingly. This is not fine-tuning the model weights (A) but rather in-context learning through prompt engineering.

---

### Question 4 (Hard)

An agent system handles customer emails. When a human reviewer rejects a draft, the system should both improve future drafts AND handle the current email. What is the correct order of operations?

- A) Store feedback, regenerate the draft with feedback context, send the new draft
- B) Send the rejected draft anyway, then store feedback
- C) Store feedback, regenerate the draft with feedback context, submit the new draft for approval, then send if approved
- D) Wait for the reviewer to write the email themselves

**Answer: C** — The correct flow is: (1) store the rejection feedback for future learning, (2) regenerate the draft incorporating the feedback, (3) submit the new draft for approval again (do not assume the regenerated version is correct), (4) send only after the new draft is approved. Option A skips the re-approval step. Option B sends a rejected draft. Option D defeats the purpose of having an agent.

---

### Question 5 (Hard)

In a graceful degradation system with `auto_approve_low_risk` fallback, a "critical" risk action is submitted but no human is available. What should happen?

- A) The action is auto-approved because it is an emergency
- B) The action is rejected because critical-risk actions require human approval
- C) The action is downgraded to "medium" risk and auto-approved
- D) The system retries until a human becomes available

**Answer: B** — With `auto_approve_low_risk` fallback behavior, only low and medium risk actions are auto-approved when no human is available. Critical-risk actions are rejected because the consequences of a wrong decision are too severe to auto-approve. This is the core principle of graceful degradation: maintain safety by reducing capability rather than reducing safety.

---

### Question 6 (Medium)

In a full-auto permission mode, the agent can execute all operations without human approval. What compensating control makes this safe?

- A) The agent uses a more powerful model that makes fewer mistakes
- B) Network access is disabled and file writes are restricted to the working directory, containing the blast radius
- C) The agent runs all operations twice and compares the results
- D) Full-auto mode is never safe and should not be used

**Answer: B** — Full-auto mode relaxes the human approval constraint but tightens the scope constraint. With network disabled, the agent cannot exfiltrate data, download malicious code, or reach external services. With file writes limited to the working directory, it cannot damage the system. This is the principle of compensating controls — relaxing one safeguard while tightening another to maintain overall safety.

---

### Question 7 (Hard)

A declarative permission system uses last-match-wins evaluation. Given these rules in order: `{ pattern: '**', decision: 'ask' }`, `{ pattern: 'src/**/*.ts', decision: 'allow' }`, `{ pattern: '.env*', decision: 'deny' }`, what happens when the agent tries to read `src/.env.local`?

- A) It is allowed because it matches `src/**/*.ts`
- B) It is denied because it matches `.env*`
- C) It triggers an ask because `**` is the broadest match
- D) It causes an error because two rules match

**Answer: B** — With last-match-wins evaluation, all matching rules are checked in order and the last one to match determines the decision. The file `src/.env.local` matches all three patterns: `**` (everything), `src/**/*.ts` would not match (no .ts extension), but `.env*` does match the filename. Since `.env*` is the last matching rule, the decision is deny. This demonstrates how layered rules can provide specific overrides on top of broad defaults.

---

## Exercises

### Exercise 1: Agent with Approval Gates for High-Stakes Actions

**Objective:** Build an agent that has multiple tools, some of which require human approval before execution. The agent should propose actions, wait for approval on high-risk ones, and proceed automatically on low-risk ones.

**Specification:**

1. Create a file `src/exercises/m18/ex01-approval-agent.ts`
2. Export an async function `approvalAgent(task: string, options?: ApprovalAgentOptions): Promise<ApprovalAgentResult>`
3. Define the types:

```typescript
type ApprovalCallback = (proposal: {
  tool: string
  args: Record<string, unknown>
  risk: string
  reasoning: string
}) => Promise<{
  approved: boolean
  feedback?: string
}>

interface ApprovalAgentOptions {
  maxSteps?: number // default: 8
  onApproval: ApprovalCallback
  verbose?: boolean // default: false
}

interface ApprovalEvent {
  tool: string
  risk: string
  approved: boolean
  feedback?: string
  timestamp: number
}

interface ApprovalAgentResult {
  response: string
  approvalEvents: ApprovalEvent[]
  totalSteps: number
  toolCallCount: number
  autoApproved: number
  humanApproved: number
  humanRejected: number
}
```

4. Implement the following tools:
   - `searchData` — low risk, auto-approved
   - `readFile` — low risk, auto-approved
   - `sendEmail` — high risk, requires human approval
   - `modifyRecord` — high risk, requires human approval
   - `deleteRecord` — critical risk, requires human approval
   - `calculateCost` — low risk, auto-approved

5. The agent must:
   - Automatically execute low-risk tools without approval
   - Request approval for high/critical-risk tools via the callback
   - If approval is rejected, inform the LLM and let it adjust its approach
   - Track all approval events in the result
   - Log each step if verbose is true

**Example usage:**

```typescript
const result = await approvalAgent(
  "Look up customer #789's account, calculate their outstanding balance, and send them a payment reminder email",
  {
    onApproval: async proposal => {
      console.log(`[REVIEW] ${proposal.tool}: ${proposal.reasoning}`)
      return { approved: true }
    },
    verbose: true,
  }
)

console.log(`Response: ${result.response}`)
console.log(`Auto-approved: ${result.autoApproved}`)
console.log(`Human-approved: ${result.humanApproved}`)
console.log(`Human-rejected: ${result.humanRejected}`)
```

**Test specification:**

```typescript
// tests/exercises/m18/ex01-approval-agent.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 18: Approval Agent', () => {
  it('should auto-approve low-risk tools', async () => {
    const result = await approvalAgent('Search for information about TypeScript', {
      onApproval: async () => ({ approved: true }),
    })
    expect(result.autoApproved).toBeGreaterThan(0)
    expect(result.humanApproved).toBe(0)
  })

  it('should request approval for high-risk tools', async () => {
    let approvalRequested = false
    const result = await approvalAgent('Send an email to user@example.com saying hello', {
      onApproval: async proposal => {
        approvalRequested = true
        expect(proposal.tool).toBe('sendEmail')
        return { approved: true }
      },
    })
    expect(approvalRequested).toBe(true)
    expect(result.humanApproved).toBeGreaterThan(0)
  })

  it('should handle rejection gracefully', async () => {
    const result = await approvalAgent('Delete all records from the users table', {
      onApproval: async () => ({
        approved: false,
        feedback: 'Bulk deletion not allowed',
      }),
    })
    expect(result.humanRejected).toBeGreaterThan(0)
    expect(result.response).toBeTruthy()
  })

  it('should track all approval events', async () => {
    const result = await approvalAgent('Search for data and send a summary email', {
      onApproval: async () => ({ approved: true }),
    })
    expect(result.approvalEvents.length).toBeGreaterThan(0)
    for (const event of result.approvalEvents) {
      expect(event.tool).toBeTruthy()
      expect(event.timestamp).toBeGreaterThan(0)
    }
  })
})
```

---

### Exercise 2: Audit Trail System

**Objective:** Build a comprehensive audit trail system that logs all agent actions, human decisions, and system events for compliance review.

**Specification:**

1. Create a file `src/exercises/m18/ex02-audit-trail.ts`
2. Export the `AuditTrail` class and `auditedPipeline` function
3. Define the types:

```typescript
interface AuditEvent {
  id: string
  timestamp: number
  sessionId: string
  eventType: string
  actor: string // "agent", "human", "system"
  action: string
  details: Record<string, unknown>
  risk?: string
  outcome?: string
}

class AuditTrail {
  constructor(sessionId?: string)
  log(event: Omit<AuditEvent, 'id' | 'timestamp' | 'sessionId'>): void
  getEvents(filter?: Partial<AuditEvent>): AuditEvent[]
  getSummary(): {
    totalEvents: number
    byActor: Record<string, number>
    byEventType: Record<string, number>
    approvalRate: number
    timeRange: { start: number; end: number }
  }
  exportJSON(): string
}

async function auditedPipeline(
  task: string,
  audit: AuditTrail,
  onApproval: ApprovalCallback
): Promise<{ result: string; audit: AuditTrail }>
```

4. The audit trail must:
   - Generate unique IDs for each event
   - Support filtering by any field
   - Calculate summary statistics
   - Export as JSON for compliance review
   - Be usable with any agent or pipeline

**Test specification:**

```typescript
// tests/exercises/m18/ex02-audit-trail.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 18: Audit Trail', () => {
  it('should log events with unique IDs', () => {
    const audit = new AuditTrail()
    audit.log({ eventType: 'test', actor: 'system', action: 'test action', details: {} })
    audit.log({ eventType: 'test', actor: 'system', action: 'another action', details: {} })

    const events = audit.getEvents()
    expect(events.length).toBe(2)
    expect(events[0].id).not.toBe(events[1].id)
  })

  it('should filter events', () => {
    const audit = new AuditTrail()
    audit.log({ eventType: 'action', actor: 'agent', action: 'search', details: {} })
    audit.log({ eventType: 'approval', actor: 'human', action: 'approve', details: {} })

    const agentEvents = audit.getEvents({ actor: 'agent' })
    expect(agentEvents.length).toBe(1)
  })

  it('should calculate summary statistics', () => {
    const audit = new AuditTrail()
    audit.log({ eventType: 'proposed', actor: 'agent', action: 'send email', details: {} })
    audit.log({ eventType: 'approved', actor: 'human', action: 'approve', details: {} })

    const summary = audit.getSummary()
    expect(summary.totalEvents).toBe(2)
    expect(summary.byActor['agent']).toBe(1)
    expect(summary.byActor['human']).toBe(1)
  })

  it('should export as valid JSON', () => {
    const audit = new AuditTrail()
    audit.log({ eventType: 'test', actor: 'system', action: 'test', details: { key: 'value' } })

    const json = audit.exportJSON()
    const parsed = JSON.parse(json)
    expect(parsed.entries.length).toBe(1)
    expect(parsed.summary.totalEvents).toBe(1)
  })
})
```

> **Local Alternative (Ollama):** Human-in-the-loop patterns (approval flows, feedback integration, corrections) are application-level logic independent of the model provider. All patterns in this module work with `ollama('qwen3.5')`. The confirmation prompts and active learning loops are the same regardless of which model generates the initial output.

---

### Exercise 3: Declarative Permission System

**Objective:** Build a permission checker that uses declarative rules with glob patterns to decide whether tool calls should be allowed, denied, or require human approval.

**Specification:**

1. Create a file `src/exercises/m18/ex03-permission-system.ts`
2. Export a function `createPermissionChecker(rules: PermissionRule[]): PermissionChecker`
3. Define the types:

```typescript
interface PermissionRule {
  pattern: string // glob pattern
  decision: 'allow' | 'deny' | 'ask'
}

interface PermissionChecker {
  check(resource: string): 'allow' | 'deny' | 'ask'
}
```

4. The checker must:
   - Evaluate rules in order, last matching rule wins
   - Support glob wildcards (`*` matches any characters, `**` matches across path separators)
   - Return `'ask'` if no rules match (safe default)
   - Handle edge cases: empty rule set, empty resource string

**Test specification:**

```typescript
// tests/exercises/m18/ex03-permission-system.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 18: Declarative Permission System', () => {
  it('should apply last-match-wins semantics', () => {
    const checker = createPermissionChecker([
      { pattern: '**', decision: 'deny' },
      { pattern: 'src/**', decision: 'allow' },
    ])
    expect(checker.check('src/index.ts')).toBe('allow')
    expect(checker.check('etc/passwd')).toBe('deny')
  })

  it('should default to ask when no rules match', () => {
    const checker = createPermissionChecker([])
    expect(checker.check('anything')).toBe('ask')
  })

  it('should handle specific deny overrides', () => {
    const checker = createPermissionChecker([
      { pattern: '**', decision: 'allow' },
      { pattern: '.env*', decision: 'deny' },
    ])
    expect(checker.check('.env')).toBe('deny')
    expect(checker.check('.env.local')).toBe('deny')
    expect(checker.check('src/app.ts')).toBe('allow')
  })
})
```

---

### Exercise 4: Denial Adaptation

**Objective:** Build an agent that adapts when the user denies an action — it proposes an alternative instead of retrying the denied operation.

**Specification:**

1. Create a file `src/exercises/m18/ex04-denial-adaptation.ts`
2. Export an async function `adaptiveAgent(task: string, options: AdaptiveAgentOptions): Promise<AdaptiveAgentResult>`
3. Define the types:

```typescript
interface AdaptiveAgentOptions {
  onApproval: (proposal: { tool: string; args: Record<string, unknown> }) => Promise<{
    approved: boolean
    feedback?: string
  }>
  maxSteps?: number // default: 6
}

interface AdaptiveAgentResult {
  response: string
  deniedActions: Array<{ tool: string; args: Record<string, unknown>; feedback?: string }>
  alternativesTried: number
  totalSteps: number
}
```

4. The agent must:
   - Track denied actions and never retry the same tool call with the same parameters
   - Include denial feedback in the next LLM call so the agent can adjust
   - Propose an alternative approach after a denial
   - Record all denied actions and the number of alternatives attempted

**Test specification:**

```typescript
// tests/exercises/m18/ex04-denial-adaptation.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 18: Denial Adaptation', () => {
  it('should not retry denied actions', async () => {
    const calls: string[] = []
    const result = await adaptiveAgent('Delete the temp files and clean up', {
      onApproval: async proposal => {
        calls.push(proposal.tool)
        if (proposal.tool === 'deleteFile') return { approved: false, feedback: 'Use trash instead' }
        return { approved: true }
      },
    })
    const deleteCalls = calls.filter(c => c === 'deleteFile')
    expect(deleteCalls.length).toBeLessThanOrEqual(1)
    expect(result.deniedActions.length).toBeGreaterThan(0)
  })

  it('should include denial feedback in result', async () => {
    const result = await adaptiveAgent('Send a notification email', {
      onApproval: async () => ({ approved: false, feedback: 'Use Slack instead of email' }),
    })
    expect(result.deniedActions[0].feedback).toContain('Slack')
  })
})
```

---

### Exercise 5: Permission Modes with Glob-Based Commands

**Objective:** Build a permission mode system that supports three autonomy levels (suggest/auto-edit/full-auto) with glob-based shell command rules.

**Specification:**

1. Create a file `src/exercises/m18/ex05-permission-modes.ts`
2. Export a function `createModeEngine(mode: PermissionMode): ModeEngine`
3. Define the types:

```typescript
type PermissionMode = 'suggest' | 'auto-edit' | 'full-auto'

interface CommandRule {
  pattern: string // glob pattern for command string
  decision: 'allow' | 'deny' | 'ask'
}

interface ModeEngine {
  mode: PermissionMode
  checkFileWrite(path: string): 'allow' | 'deny' | 'ask'
  checkFileRead(path: string): 'allow' | 'deny' | 'ask'
  checkCommand(command: string): 'allow' | 'deny' | 'ask'
}
```

4. Mode behaviors:
   - **suggest** — reads are allowed, all writes and commands require approval
   - **auto-edit** — reads and file writes within the project are allowed, commands require approval
   - **full-auto** — reads and writes within the project are allowed, safe commands (git status, npm test, etc.) are allowed, network commands (curl, wget, npm publish) are denied, destructive commands (rm -rf) are denied
5. Command rules use glob matching with last-match-wins semantics

**Test specification:**

```typescript
// tests/exercises/m18/ex05-permission-modes.test.ts
import { describe, it, expect } from 'bun:test'

describe('Exercise 18: Permission Modes', () => {
  it('suggest mode should require approval for all writes', () => {
    const engine = createModeEngine('suggest')
    expect(engine.checkFileWrite('src/app.ts')).toBe('ask')
    expect(engine.checkFileRead('src/app.ts')).toBe('allow')
    expect(engine.checkCommand('git status')).toBe('ask')
  })

  it('auto-edit mode should allow writes but ask for commands', () => {
    const engine = createModeEngine('auto-edit')
    expect(engine.checkFileWrite('src/app.ts')).toBe('allow')
    expect(engine.checkCommand('npm test')).toBe('ask')
  })

  it('full-auto mode should deny network commands', () => {
    const engine = createModeEngine('full-auto')
    expect(engine.checkFileWrite('src/app.ts')).toBe('allow')
    expect(engine.checkCommand('git status')).toBe('allow')
    expect(engine.checkCommand('curl http://example.com')).toBe('deny')
    expect(engine.checkCommand('rm -rf /')).toBe('deny')
  })
})
```

---

## Summary

In this module, you learned:

1. **Why human-in-the-loop:** Trust, accuracy, and compliance require human oversight for high-stakes AI actions. Place each action on the autonomy spectrum based on its risk level.
2. **Approval flows:** Agents propose actions and humans approve, reject, or modify them. Use callbacks for programmatic environments and stdin for CLI contexts.
3. **Confidence-based routing:** Auto-approve high-confidence, low-risk actions. Escalate low-confidence or high-risk actions to humans. Notify on moderate cases.
4. **Feedback integration:** Store human corrections and use them as context in future prompts. Track approval rates to identify systematic issues.
5. **Active learning:** Proactive agents identify their own uncertainty and ask for clarification before acting. This is more efficient than acting and being rejected.
6. **Review interfaces:** CLI-based and programmatic review queues let humans efficiently review batches of agent proposals.
7. **Audit trails:** Log every decision with enough detail for compliance review. Include who, what, when, why, and the outcome.
8. **Graceful degradation:** When no human is available, fall back to safe behaviors — auto-approve low-risk actions, queue high-risk ones, or reject everything. Never auto-approve critical actions.
9. **Declarative permission rules:** Defining access control as data (pattern + decision) instead of logic makes permissions auditable, versionable, and composable with last-match-wins semantics.
10. **Permission modes:** Named configurations (restrictive, normal, autonomous) let users match the autonomy level to the task by switching rule sets, not code.
11. **Denial adaptation:** Recording denied actions and including them as context prevents the agent from retrying rejected operations and teaches it to propose alternatives.
12. **Three approval modes:** Suggest (read-only), auto-edit (file writes allowed), and full-auto (all operations, but network disabled) provide distinct autonomy levels with compensating controls.
13. **Glob-based command permissions:** Fine-grained shell command control using glob patterns enables layered rules — project defaults with user overrides — for precise command execution governance.

This completes Part IV: Agents and Orchestration. You now have the patterns to build autonomous agents, coordinate multiple agents, design deterministic pipelines, generate code iteratively, and add human oversight — the full toolkit for production LLM applications.
