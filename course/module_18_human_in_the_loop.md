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

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

// Classify action risk to determine the approval level needed
const riskSchema = z.object({
  action: z.string(),
  riskLevel: z.enum(['low', 'medium', 'high', 'critical']),
  autonomyLevel: z.enum(['full_autonomy', 'notify_only', 'approve_reject', 'human_executes']),
  reasoning: z.string(),
  reversible: z.boolean().describe('Can this action be undone?'),
})

async function classifyActionRisk(action: string, context: string): Promise<z.infer<typeof riskSchema>> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: riskSchema }),
    system: `You are a risk assessment system. Classify actions by risk level:
- low: No real consequences if wrong (reading data, formatting)
- medium: Minor consequences, easily reversible (sending notifications, updating preferences)
- high: Significant consequences, hard to reverse (sending emails, modifying databases)
- critical: Major consequences, irreversible (financial transactions, data deletion, legal filings)

Map risk levels to autonomy levels:
- low -> full_autonomy
- medium -> notify_only
- high -> approve_reject
- critical -> human_executes`,
    prompt: `Action: ${action}\nContext: ${context}`,
  })

  return output!
}

// Usage
const risk = await classifyActionRisk(
  'Delete all user records older than 2 years',
  'Production database with 50,000 user records'
)

console.log(`Action: ${risk.action}`)
console.log(`Risk: ${risk.riskLevel}, Autonomy: ${risk.autonomyLevel}`)
console.log(`Reversible: ${risk.reversible}`)
console.log(`Reasoning: ${risk.reasoning}`)
```

> **Beginner Note:** Human-in-the-loop is not about distrusting AI — it is about using AI and humans where each excels. AI is fast and tireless. Humans have judgment and contextual understanding. The best systems combine both.

---

## Section 2: Approval Flows

### Agent Proposes, Human Approves

The basic HITL pattern: the agent generates a proposed action, presents it to a human, and only executes if approved.

```typescript
import { generateText, Output, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'
import * as readline from 'readline'

// Proposed action that needs approval
interface ProposedAction {
  type: string
  description: string
  details: Record<string, unknown>
  risk: 'low' | 'medium' | 'high' | 'critical'
  reasoning: string
}

// Human decision
interface ApprovalDecision {
  approved: boolean
  feedback?: string
  modifiedAction?: Partial<ProposedAction>
}

// Ask a human for approval via CLI
async function requestApproval(action: ProposedAction): Promise<ApprovalDecision> {
  console.log('\n' + '='.repeat(60))
  console.log('APPROVAL REQUIRED')
  console.log('='.repeat(60))
  console.log(`Action: ${action.type}`)
  console.log(`Description: ${action.description}`)
  console.log(`Risk level: ${action.risk}`)
  console.log(`Reasoning: ${action.reasoning}`)
  console.log(`Details: ${JSON.stringify(action.details, null, 2)}`)
  console.log('='.repeat(60))

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  })

  return new Promise(resolve => {
    rl.question('\nApprove this action? (y/n/m for modify): ', async answer => {
      const response = answer.trim().toLowerCase()

      if (response === 'y' || response === 'yes') {
        rl.close()
        resolve({ approved: true })
      } else if (response === 'm' || response === 'modify') {
        rl.question('Enter modified instructions: ', feedback => {
          rl.close()
          resolve({
            approved: true,
            feedback: feedback.trim(),
            modifiedAction: { description: feedback.trim() },
          })
        })
      } else {
        rl.question('Reason for rejection (optional): ', reason => {
          rl.close()
          resolve({
            approved: false,
            feedback: reason.trim() || undefined,
          })
        })
      }
    })
  })
}

// Agent with approval gates
async function agentWithApproval(task: string): Promise<string> {
  const messages: ModelMessage[] = [{ role: 'user', content: task }]

  const tools = {
    sendEmail: {
      description: 'Send an email to a recipient. REQUIRES HUMAN APPROVAL.',
      parameters: z.object({
        to: z.string().describe('Recipient email address'),
        subject: z.string().describe('Email subject'),
        body: z.string().describe('Email body'),
      }),
      execute: async ({ to, subject, body }: { to: string; subject: string; body: string }) => {
        const proposed: ProposedAction = {
          type: 'sendEmail',
          description: `Send email to ${to}: "${subject}"`,
          details: { to, subject, body },
          risk: 'high',
          reasoning: 'This action modifies external state and cannot be automatically undone.',
        }
        const decision = await requestApproval(proposed)
        if (!decision.approved) {
          return `Action denied by human reviewer.${decision.feedback ? ` Reason: ${decision.feedback}` : ''}`
        }
        // In production: actually send the email
        return `Email sent to ${to} with subject "${subject}".`
      },
    },
    searchData: {
      description: 'Search internal data. Does NOT require approval.',
      parameters: z.object({
        query: z.string(),
      }),
      execute: async ({ query }: { query: string }) => `Search results for "${query}": [data found]`,
    },
    deleteRecord: {
      description: 'Delete a database record. REQUIRES HUMAN APPROVAL.',
      parameters: z.object({
        table: z.string(),
        id: z.string(),
      }),
      execute: async ({ table, id }: { table: string; id: string }) => {
        const proposed: ProposedAction = {
          type: 'deleteRecord',
          description: `Delete record ${id} from table ${table}`,
          details: { table, id },
          risk: 'critical',
          reasoning: 'This action modifies external state and cannot be automatically undone.',
        }
        const decision = await requestApproval(proposed)
        if (!decision.approved) {
          return `Action denied by human reviewer.${decision.feedback ? ` Reason: ${decision.feedback}` : ''}`
        }
        // In production: actually delete the record
        return `Record ${id} deleted from table ${table}.`
      },
    },
  }

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a helpful assistant. Some actions require human approval before execution.
When you want to send an email or delete a record, propose the action and wait for approval.
For search actions, proceed automatically.`,
    messages,
    tools,
    maxSteps: 5,
  })

  return result.text
}

// Note: This example uses stdin for approval, which works in CLI contexts.
// For programmatic usage, replace requestApproval with a callback function.
```

> **Advanced Note:** In production, approval requests are typically sent to a queue (Slack, email, web dashboard) rather than blocking on stdin. The agent pauses, saves its state, and resumes when the approval comes back. This requires persistent state storage which we cover in the audit trail section.

### Programmatic Approval Callbacks

For automated testing and non-CLI environments, use callback-based approval:

```typescript
import { generateText, type ModelMessage } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

type ApprovalCallback = (action: { toolName: string; args: Record<string, unknown> }) => Promise<{
  approved: boolean
  feedback?: string
}>

async function agentWithCallback(
  task: string,
  onApprovalNeeded: ApprovalCallback
): Promise<{ response: string; approvals: Array<{ action: string; approved: boolean }> }> {
  const approvals: Array<{ action: string; approved: boolean }> = []
  const approvalRequired = new Set(['sendEmail', 'deleteRecord', 'modifyDatabase'])

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    maxSteps: 5,
    system: 'You are a helpful assistant with access to tools that may require approval.',
    tools: {
      sendEmail: {
        description: 'Send an email (requires approval)',
        parameters: z.object({
          to: z.string(),
          subject: z.string(),
          body: z.string(),
        }),
        execute: async args => {
          if (approvalRequired.has('sendEmail')) {
            const decision = await onApprovalNeeded({
              toolName: 'sendEmail',
              args,
            })
            approvals.push({
              action: 'sendEmail',
              approved: decision.approved,
            })

            if (!decision.approved) {
              return JSON.stringify({
                success: false,
                reason: `Action rejected by human reviewer: ${decision.feedback || 'no reason given'}`,
              })
            }
          }

          // Execute the action (simulated)
          return JSON.stringify({
            success: true,
            message: `Email sent to ${args.to}`,
          })
        },
      },
      searchData: {
        description: 'Search for data (no approval needed)',
        parameters: z.object({ query: z.string() }),
        execute: async ({ query }) => `Results for "${query}": [relevant data]`,
      },
    },
    prompt: task,
  })

  return { response: result.text, approvals }
}

// Usage with an auto-approve callback (for testing)
const testResult = await agentWithCallback(
  "Search for John Doe's email and send him a meeting invitation for Friday at 2pm",
  async action => {
    console.log(`Approval requested: ${action.toolName}`)
    console.log(`Args: ${JSON.stringify(action.args)}`)
    // Auto-approve for testing
    return { approved: true }
  }
)

console.log('Response:', testResult.response)
console.log('Approvals:', testResult.approvals)

// Usage with an auto-reject callback (for testing guardrails)
const rejectResult = await agentWithCallback('Delete all inactive users from the database', async action => {
  return {
    approved: false,
    feedback: 'Bulk deletions require manager approval',
  }
})

console.log('Response:', rejectResult.response)
console.log('Approvals:', rejectResult.approvals)
```

---

## Section 3: Confidence-Based Routing

### Auto-Approve High Confidence, Escalate Low Confidence

Not every action needs human review. Route based on the agent's confidence:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

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

async function confidenceBasedRouting(
  task: string,
  config: ConfidenceConfig,
  onApproval: (proposal: Proposal) => Promise<boolean>
): Promise<{
  result: string
  routing: 'auto_approved' | 'notified' | 'human_reviewed'
  proposal: Proposal
}> {
  // Step 1: Agent proposes an action with confidence
  const { output: proposal } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: proposalSchema }),
    system: `You are an assistant that proposes actions with confidence levels.
Be honest about your confidence:
- 0.9-1.0: Very confident, clear-cut case
- 0.7-0.9: Fairly confident, standard case with minor uncertainties
- 0.5-0.7: Moderate confidence, some ambiguity or missing context
- Below 0.5: Low confidence, significant uncertainty

Consider: Is the task clear? Do you have enough context? Could this go wrong?`,
    prompt: task,
  })

  console.log(`Proposal: ${proposal!.action}`)
  console.log(`Confidence: ${proposal!.confidence}`)
  console.log(`Reasoning: ${proposal!.reasoning}`)

  // Step 2: Route based on confidence
  let routing: 'auto_approved' | 'notified' | 'human_reviewed'

  if (proposal!.confidence >= config.autoApproveThreshold) {
    routing = 'auto_approved'
    console.log('[Router] Auto-approved (high confidence)')
  } else if (proposal!.confidence >= config.escalateThreshold) {
    routing = 'notified'
    console.log('[Router] Proceeding with notification (moderate confidence)')
  } else {
    routing = 'human_reviewed'
    console.log('[Router] Escalating to human review (low confidence)')

    const approved = await onApproval(proposal!)
    if (!approved) {
      return {
        result: 'Action rejected by human reviewer.',
        routing,
        proposal: proposal!,
      }
    }
  }

  // Step 3: Execute the action
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: `Execute this action: ${proposal!.action}\n\nContext: ${proposal!.reasoning}`,
  })

  return { result: result.text, routing, proposal: proposal! }
}

// Usage
const output = await confidenceBasedRouting(
  'Reply to the customer saying their refund has been processed for $49.99',
  {
    autoApproveThreshold: 0.9,
    escalateThreshold: 0.7,
  },
  async proposal => {
    console.log(`\n[HUMAN REVIEW NEEDED]`)
    console.log(`Action: ${proposal.action}`)
    console.log(`Confidence: ${proposal.confidence}`)
    console.log(`Risk: ${proposal.riskAssessment}`)
    // Simulating human approval
    return true
  }
)

console.log(`\nRouting: ${output.routing}`)
console.log(`Result: ${output.result}`)
```

### Dynamic Threshold Adjustment

Adjust thresholds based on the action type:

```typescript
import { z } from 'zod'

interface ActionThresholds {
  readOnly: { autoApprove: number; escalate: number }
  modification: { autoApprove: number; escalate: number }
  deletion: { autoApprove: number; escalate: number }
  financial: { autoApprove: number; escalate: number }
  communication: { autoApprove: number; escalate: number }
}

const defaultThresholds: ActionThresholds = {
  readOnly: { autoApprove: 0.5, escalate: 0.0 }, // Almost always auto-approve
  modification: { autoApprove: 0.85, escalate: 0.6 },
  deletion: { autoApprove: 0.95, escalate: 0.8 }, // Very high bar
  financial: { autoApprove: 1.0, escalate: 0.9 }, // Effectively always review
  communication: { autoApprove: 0.9, escalate: 0.7 },
}

function getThresholds(
  actionType: keyof ActionThresholds,
  overrides?: Partial<ActionThresholds>
): { autoApprove: number; escalate: number } {
  const thresholds = { ...defaultThresholds, ...overrides }
  return thresholds[actionType]
}

// Usage
const deleteThresholds = getThresholds('deletion')
console.log(`Delete auto-approve: ${deleteThresholds.autoApprove}`)
// 0.95 — practically never auto-approved

const readThresholds = getThresholds('readOnly')
console.log(`Read auto-approve: ${readThresholds.autoApprove}`)
// 0.5 — almost always auto-approved
```

> **Beginner Note:** Confidence-based routing is like a triage system in a hospital. Minor issues (high confidence, low risk) are handled automatically. Serious cases (low confidence, high risk) go to a specialist (human reviewer). This keeps the system efficient while maintaining safety.

---

## Section 4: Feedback Integration

### Learning from Human Corrections

When a human rejects or modifies an agent's proposal, that feedback is valuable. Store it and use it to improve future behavior:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

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

class FeedbackStore {
  private entries: FeedbackEntry[] = []

  add(entry: FeedbackEntry): void {
    this.entries.push(entry)
  }

  getRecentFeedback(category: string, limit: number = 5): FeedbackEntry[] {
    return this.entries
      .filter(e => e.category === category)
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, limit)
  }

  getRejectionPatterns(): Map<string, number> {
    const patterns = new Map<string, number>()
    for (const entry of this.entries.filter(e => e.decision === 'rejected')) {
      const key = entry.category
      patterns.set(key, (patterns.get(key) || 0) + 1)
    }
    return patterns
  }

  getApprovalRate(category?: string): number {
    const relevant = category ? this.entries.filter(e => e.category === category) : this.entries
    if (relevant.length === 0) return 0
    const approved = relevant.filter(e => e.decision === 'approved').length
    return approved / relevant.length
  }

  // Format recent feedback as context for the agent
  toPromptContext(category: string): string {
    const recent = this.getRecentFeedback(category, 3)
    if (recent.length === 0) return ''

    const examples = recent
      .map(e => {
        let text = `- Task: "${e.task}" -> Proposed: "${e.proposedAction}"`
        if (e.decision === 'rejected') {
          text += ` -> REJECTED: "${e.humanFeedback || 'no reason'}"`
        } else if (e.decision === 'modified') {
          text += ` -> MODIFIED TO: "${e.modifiedAction}"`
        } else {
          text += ` -> APPROVED`
        }
        return text
      })
      .join('\n')

    return `\nRecent feedback from human reviewers for similar tasks:
${examples}

Use this feedback to improve your proposals. Avoid patterns that were previously rejected.`
  }
}

// Agent that learns from feedback
async function feedbackAwareAgent(
  task: string,
  category: string,
  feedbackStore: FeedbackStore,
  onApproval: (action: string) => Promise<{
    decision: 'approved' | 'rejected' | 'modified'
    feedback?: string
    modification?: string
  }>
): Promise<string> {
  const feedbackContext = feedbackStore.toPromptContext(category)
  const approvalRate = feedbackStore.getApprovalRate(category)

  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: `You are a helpful assistant. Propose actions carefully.
${feedbackContext}

Current approval rate for ${category}: ${(approvalRate * 100).toFixed(0)}%.
${approvalRate < 0.7 ? 'Your proposals are being rejected frequently. Be more conservative and specific.' : ''}`,
    prompt: task,
  })

  // Request approval
  const decision = await onApproval(result.text)

  // Store feedback
  feedbackStore.add({
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    task,
    proposedAction: result.text,
    decision: decision.decision,
    humanFeedback: decision.feedback,
    modifiedAction: decision.modification,
    category,
  })

  if (decision.decision === 'modified' && decision.modification) {
    return decision.modification
  }

  return result.text
}

// Usage
const store = new FeedbackStore()

// Simulate past feedback
store.add({
  id: '1',
  timestamp: Date.now() - 3600000,
  task: 'Write a refund email',
  proposedAction: 'Dear Customer, your refund is done.',
  decision: 'rejected',
  humanFeedback: 'Too informal. Include the refund amount and expected timeline.',
  category: 'customer_email',
})

store.add({
  id: '2',
  timestamp: Date.now() - 1800000,
  task: 'Write an apology email',
  proposedAction:
    'Dear valued customer, we sincerely apologize for the inconvenience. Your refund of $29.99 will be processed within 3-5 business days.',
  decision: 'approved',
  category: 'customer_email',
})

const response = await feedbackAwareAgent(
  'Write a confirmation email for a subscription cancellation',
  'customer_email',
  store,
  async action => {
    console.log('Proposed action:', action)
    // Simulate approval
    return { decision: 'approved' }
  }
)

console.log('Response:', response)
console.log('\nApproval rate:', store.getApprovalRate('customer_email'))
```

> **Advanced Note:** In production, feedback stores should be persistent (database) and shared across agent instances. Use the feedback to fine-tune prompts, adjust confidence thresholds, and identify systematic issues. If a certain category of action is consistently rejected, that signals a prompt engineering problem.

---

## Section 5: Active Learning

### Agent Asks for Help on Uncertain Cases

Instead of waiting for the human to reject a proposal, a proactive agent can identify uncertainty and ask for guidance upfront:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

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

async function activeLearningAgent(
  task: string,
  onClarification: (requests: ClarificationRequest[]) => Promise<Record<string, string>>
): Promise<string> {
  // Step 1: Analyze the task for uncertainty
  const { output: analysis } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({ schema: uncertaintySchema }),
    system: `You are a careful assistant. Before acting, assess your confidence.
If anything is ambiguous, unclear, or has multiple valid interpretations,
flag it as an uncertainty. It is better to ask than to guess wrong.

Common sources of uncertainty:
- Ambiguous pronouns or references
- Missing context (dates, amounts, names)
- Multiple valid interpretations of the request
- Domain-specific terms that could mean different things
- Unstated preferences or constraints`,
    prompt: task,
  })

  // Step 2: If uncertain, ask for clarification
  if (!analysis!.confident && analysis!.uncertainties.length > 0) {
    console.log('[Agent] Requesting clarification on uncertain points...')

    const clarifications = await onClarification(
      analysis!.uncertainties.map(u => ({
        question: u.question,
        options: u.options,
        context: task,
        impact: u.impact,
      }))
    )

    // Step 3: Proceed with clarified context
    const clarificationContext = Object.entries(clarifications)
      .map(([q, a]) => `Q: ${q}\nA: ${a}`)
      .join('\n\n')

    const result = await generateText({
      model: mistral('mistral-small-latest'),
      prompt: `Original task: ${task}

Clarifications received:
${clarificationContext}

Now complete the task with these clarifications.`,
    })

    return result.text
  }

  // Step 4: Confident — proceed directly
  console.log('[Agent] Confident enough to proceed without clarification.')
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    prompt: task,
  })

  return result.text
}

// Usage
const response = await activeLearningAgent('Send a follow-up email to the client about the project', async requests => {
  console.log('\n=== Clarification Needed ===')
  const answers: Record<string, string> = {}

  for (const req of requests) {
    console.log(`\nQuestion: ${req.question}`)
    console.log(`Options: ${req.options.join(', ')}`)
    console.log(`Impact: ${req.impact}`)

    // Simulating human response
    answers[req.question] =
      'The client is Acme Corp. The project is the Q1 dashboard redesign. Send a status update, not a request for payment.'
  }

  return answers
})

console.log('\nFinal response:', response)
```

### Uncertainty Detection Heuristics

Beyond asking the model about its own confidence, use structural heuristics:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface UncertaintySignals {
  hasAmbiguousPronouns: boolean
  missingNumbers: boolean
  multiplePossibleInterpretations: boolean
  domainSpecificTerms: boolean
  vagueDateReferences: boolean
  overallUncertainty: number // 0-1
}

async function detectUncertainty(task: string): Promise<UncertaintySignals> {
  const { output } = await generateText({
    model: mistral('mistral-small-latest'),
    output: Output.object({
      schema: z.object({
        hasAmbiguousPronouns: z.boolean().describe('Are there pronouns (it, they, them) without clear referents?'),
        missingNumbers: z.boolean().describe('Are there missing quantities, amounts, or dates?'),
        multiplePossibleInterpretations: z
          .boolean()
          .describe('Could this task reasonably be interpreted in more than one way?'),
        domainSpecificTerms: z.boolean().describe('Are there industry-specific terms that could be ambiguous?'),
        vagueDateReferences: z.boolean().describe('Are there vague time references like "soon", "recently", "later"?'),
        overallUncertainty: z
          .number()
          .min(0)
          .max(1)
          .describe('Overall uncertainty score (0 = very clear, 1 = very ambiguous)'),
      }),
    }),
    prompt: `Analyze this task for ambiguity and missing information:
"${task}"`,
  })

  return output!
}

// Usage
const signals = await detectUncertainty('Send them the updated report when it is ready')

console.log('Uncertainty signals:', signals)
// Likely flags: ambiguous pronouns ("them", "it"), vague date ("when ready")
```

> **Beginner Note:** Active learning is the agent equivalent of asking a clarifying question before starting work. A junior employee who asks "Just to confirm, you mean the Q1 report for Acme Corp, right?" is more valuable than one who confidently delivers the wrong report.

---

## Section 6: Review Interfaces

### CLI-Based Review

Build a command-line review interface for human decision-making:

```typescript
import * as readline from 'readline'

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

class CLIReviewer {
  private rl: readline.Interface

  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    })
  }

  private ask(question: string): Promise<string> {
    return new Promise(resolve => {
      this.rl.question(question, answer => resolve(answer.trim()))
    })
  }

  async reviewItem(item: ReviewItem): Promise<ReviewDecision> {
    console.log('\n' + '-'.repeat(60))
    console.log(`Review Item: ${item.id}`)
    console.log(`Type: ${item.type} | Risk: ${item.risk}`)
    console.log('-'.repeat(60))
    console.log(item.content)
    console.log('-'.repeat(60))

    if (item.metadata && Object.keys(item.metadata).length > 0) {
      console.log('Metadata:', JSON.stringify(item.metadata, null, 2))
      console.log('-'.repeat(60))
    }

    const choices = '[a]pprove / [r]eject / [m]odify / [s]kip'
    const response = await this.ask(`Decision (${choices}): `)

    switch (response.toLowerCase()) {
      case 'a':
      case 'approve': {
        const note = await this.ask('Note (optional, press Enter to skip): ')
        return {
          itemId: item.id,
          decision: 'approve',
          reviewerNote: note || undefined,
          timestamp: Date.now(),
        }
      }

      case 'r':
      case 'reject': {
        const feedback = await this.ask('Reason for rejection: ')
        return {
          itemId: item.id,
          decision: 'reject',
          feedback,
          timestamp: Date.now(),
        }
      }

      case 'm':
      case 'modify': {
        const modified = await this.ask('Modified content: ')
        const note = await this.ask('Note about changes: ')
        return {
          itemId: item.id,
          decision: 'modify',
          modifiedContent: modified,
          feedback: note || undefined,
          timestamp: Date.now(),
        }
      }

      default:
        return {
          itemId: item.id,
          decision: 'skip',
          timestamp: Date.now(),
        }
    }
  }

  async reviewBatch(items: ReviewItem[]): Promise<ReviewDecision[]> {
    console.log(`\n${'='.repeat(60)}`)
    console.log(`Review Queue: ${items.length} items`)
    console.log(`${'='.repeat(60)}`)

    const decisions: ReviewDecision[] = []

    for (let i = 0; i < items.length; i++) {
      console.log(`\n[${i + 1}/${items.length}]`)
      const decision = await this.reviewItem(items[i])
      decisions.push(decision)

      // Show running stats
      const approved = decisions.filter(d => d.decision === 'approve').length
      const rejected = decisions.filter(d => d.decision === 'reject').length
      console.log(`Progress: ${approved} approved, ${rejected} rejected, ${items.length - i - 1} remaining`)
    }

    this.rl.close()
    return decisions
  }
}

// Usage example (would be run in a CLI context)
// const reviewer = new CLIReviewer();
// const decisions = await reviewer.reviewBatch([
//   {
//     id: "1",
//     type: "email_draft",
//     content: "Dear Customer, your refund of $49.99 has been processed...",
//     metadata: { recipient: "customer@example.com" },
//     risk: "high",
//   },
//   {
//     id: "2",
//     type: "data_update",
//     content: "UPDATE users SET status='inactive' WHERE last_login < '2024-01-01'",
//     metadata: { affectedRows: 1234 },
//     risk: "critical",
//   },
// ]);
```

### Programmatic Review Interface

For integrations with web dashboards or APIs:

```typescript
interface ReviewQueue {
  items: Map<string, ReviewItem & { status: 'pending' | 'reviewed' }>
  decisions: Map<string, ReviewDecision>
  waiters: Map<string, (decision: ReviewDecision) => void>
}

function createReviewQueue(): ReviewQueue {
  return {
    items: new Map(),
    decisions: new Map(),
    waiters: new Map(),
  }
}

function submitForReview(queue: ReviewQueue, item: ReviewItem): Promise<ReviewDecision> {
  queue.items.set(item.id, { ...item, status: 'pending' })

  return new Promise(resolve => {
    // Check if already reviewed (e.g., auto-approved)
    const existing = queue.decisions.get(item.id)
    if (existing) {
      resolve(existing)
      return
    }

    // Wait for a reviewer to submit a decision
    queue.waiters.set(item.id, resolve)
  })
}

function submitDecision(queue: ReviewQueue, decision: ReviewDecision): void {
  queue.decisions.set(decision.itemId, decision)

  const item = queue.items.get(decision.itemId)
  if (item) {
    item.status = 'reviewed'
  }

  // Resolve any waiting promise
  const waiter = queue.waiters.get(decision.itemId)
  if (waiter) {
    waiter(decision)
    queue.waiters.delete(decision.itemId)
  }
}

function getPendingItems(queue: ReviewQueue): ReviewItem[] {
  return Array.from(queue.items.values())
    .filter(item => item.status === 'pending')
    .map(({ status, ...item }) => item)
}

// Usage
const queue = createReviewQueue()

// Agent submits items for review
const agentTask = async () => {
  const decision = await submitForReview(queue, {
    id: 'action-1',
    type: 'email',
    content: 'Draft email to customer about refund',
    metadata: { amount: 49.99 },
    risk: 'high',
  })

  if (decision.decision === 'approve') {
    console.log('[Agent] Email approved, sending...')
  } else {
    console.log(`[Agent] Email ${decision.decision}: ${decision.feedback}`)
  }
}

// Reviewer checks the queue (in a separate process/interface)
const reviewerTask = async () => {
  // Simulate a short delay before the reviewer checks
  await new Promise(r => setTimeout(r, 100))

  const pending = getPendingItems(queue)
  console.log(`[Reviewer] ${pending.length} items pending`)

  for (const item of pending) {
    submitDecision(queue, {
      itemId: item.id,
      decision: 'approve',
      timestamp: Date.now(),
      reviewerNote: 'Looks good',
    })
  }
}

// Run both concurrently
await Promise.all([agentTask(), reviewerTask()])
```

---

## Section 7: Audit Trails

### Logging Decisions for Compliance

Every decision in a HITL system should be logged with enough detail to reconstruct what happened and why:

```typescript
interface AuditEntry {
  id: string
  timestamp: number
  sessionId: string

  // What happened
  eventType:
    | 'action_proposed'
    | 'action_approved'
    | 'action_rejected'
    | 'action_modified'
    | 'action_executed'
    | 'error'
    | 'clarification_requested'
    | 'clarification_received'

  // Who was involved
  agentId: string
  reviewerId?: string

  // Details
  action: string
  details: Record<string, unknown>
  reasoning?: string
  feedback?: string

  // Context
  taskDescription: string
  confidenceScore?: number
  riskLevel?: string

  // For compliance
  policyApplied?: string
  complianceCheck?: boolean
}

class AuditLog {
  private entries: AuditEntry[] = []
  private sessionId: string

  constructor(sessionId?: string) {
    this.sessionId = sessionId || crypto.randomUUID()
  }

  log(entry: Omit<AuditEntry, 'id' | 'timestamp' | 'sessionId'>): void {
    const fullEntry: AuditEntry = {
      ...entry,
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      sessionId: this.sessionId,
    }

    this.entries.push(fullEntry)

    // Also print for real-time monitoring
    console.log(`[AUDIT] ${fullEntry.eventType} | Agent: ${fullEntry.agentId} | ${fullEntry.action.slice(0, 80)}`)
  }

  getEntries(filter?: { eventType?: string; agentId?: string; after?: number }): AuditEntry[] {
    let results = [...this.entries]

    if (filter?.eventType) {
      results = results.filter(e => e.eventType === filter.eventType)
    }
    if (filter?.agentId) {
      results = results.filter(e => e.agentId === filter.agentId)
    }
    if (filter?.after) {
      results = results.filter(e => e.timestamp > filter.after)
    }

    return results
  }

  getSessionSummary(): {
    totalEvents: number
    proposed: number
    approved: number
    rejected: number
    modified: number
    errors: number
    approvalRate: number
    timeline: Array<{ time: number; event: string }>
  } {
    const proposed = this.entries.filter(e => e.eventType === 'action_proposed').length
    const approved = this.entries.filter(e => e.eventType === 'action_approved').length
    const rejected = this.entries.filter(e => e.eventType === 'action_rejected').length
    const modified = this.entries.filter(e => e.eventType === 'action_modified').length
    const errors = this.entries.filter(e => e.eventType === 'error').length

    return {
      totalEvents: this.entries.length,
      proposed,
      approved,
      rejected,
      modified,
      errors,
      approvalRate: proposed > 0 ? approved / proposed : 0,
      timeline: this.entries.map(e => ({
        time: e.timestamp,
        event: `${e.eventType}: ${e.action.slice(0, 50)}`,
      })),
    }
  }

  // Export for compliance — e.g., to save to a file or database
  exportJSON(): string {
    return JSON.stringify(
      {
        sessionId: this.sessionId,
        exportedAt: new Date().toISOString(),
        entries: this.entries,
        summary: this.getSessionSummary(),
      },
      null,
      2
    )
  }
}

// Agent with full audit trail
import { generateText } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

async function auditedAgent(
  task: string,
  audit: AuditLog,
  onApproval: (action: string) => Promise<{
    decision: 'approved' | 'rejected' | 'modified'
    feedback?: string
  }>
): Promise<string> {
  const agentId = 'support-agent-v1'

  // Log task start
  audit.log({
    eventType: 'action_proposed',
    agentId,
    action: 'Process customer request',
    details: { task },
    taskDescription: task,
  })

  // Generate response
  const result = await generateText({
    model: mistral('mistral-small-latest'),
    system: 'You are a customer support agent. Draft a response to the customer.',
    prompt: task,
  })

  // Request approval
  audit.log({
    eventType: 'action_proposed',
    agentId,
    action: 'Send customer response',
    details: { draft: result.text.slice(0, 200) },
    taskDescription: task,
    confidenceScore: 0.85,
    riskLevel: 'medium',
  })

  const decision = await onApproval(result.text)

  // Log the decision
  audit.log({
    eventType:
      decision.decision === 'approved'
        ? 'action_approved'
        : decision.decision === 'rejected'
          ? 'action_rejected'
          : 'action_modified',
    agentId,
    reviewerId: 'human-reviewer-1',
    action: 'Customer response review',
    details: { decision: decision.decision },
    feedback: decision.feedback,
    taskDescription: task,
    policyApplied: 'customer-communication-policy-v2',
    complianceCheck: true,
  })

  if (decision.decision === 'approved') {
    // Log execution
    audit.log({
      eventType: 'action_executed',
      agentId,
      action: 'Send customer response',
      details: { sent: true },
      taskDescription: task,
    })
    return result.text
  }

  return `Action ${decision.decision}: ${decision.feedback || 'No feedback'}`
}

// Usage
const audit = new AuditLog()

const response = await auditedAgent(
  'Customer wants to know the status of their refund for order #12345',
  audit,
  async action => {
    console.log('\n[Reviewer] Reviewing:', action.slice(0, 100))
    return { decision: 'approved' }
  }
)

// Print audit summary
const summary = audit.getSessionSummary()
console.log('\n=== Audit Summary ===')
console.log(`Total events: ${summary.totalEvents}`)
console.log(`Proposed: ${summary.proposed}`)
console.log(`Approved: ${summary.approved}`)
console.log(`Rejected: ${summary.rejected}`)
console.log(`Approval rate: ${(summary.approvalRate * 100).toFixed(0)}%`)

// Export for compliance
// await Bun.write("data/audit-log.json", audit.exportJSON());
```

> **Advanced Note:** For regulatory compliance (GDPR, HIPAA, SOX), audit logs typically need to be immutable (append-only), timestamped with a trusted clock, and stored in a tamper-evident manner. Consider using a dedicated audit logging service or write-once storage in production.

---

## Section 8: Graceful Degradation

### When No Human is Available

What happens when the agent needs approval but no human is online? Production systems need fallback behaviors:

```typescript
import { generateText, Output } from 'ai'
import { mistral } from '@ai-sdk/mistral'
import { z } from 'zod'

interface DegradationConfig {
  approvalTimeoutMs: number // How long to wait for human
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

class GracefulDegradation {
  private queue: Array<{
    action: string
    risk: string
    timestamp: number
    resolve: (decision: string) => void
  }> = []

  constructor(private config: DegradationConfig) {}

  async requestApproval(
    action: string,
    risk: string,
    humanAvailable: () => boolean,
    requestHuman: (action: string) => Promise<string>
  ): Promise<DegradedAction> {
    // If human is available, use normal flow
    if (humanAvailable()) {
      try {
        const decision = await Promise.race([
          requestHuman(action),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Approval timeout')), this.config.approvalTimeoutMs)
          ),
        ])

        return {
          action,
          originalRisk: risk,
          degradedDecision: decision,
          reason: 'Human reviewed',
        }
      } catch (error) {
        console.warn(`[Degradation] Human approval timed out after ${this.config.approvalTimeoutMs}ms`)
      }
    }

    // No human available or timed out — apply degradation strategy
    return this.applyFallback(action, risk)
  }

  private applyFallback(action: string, risk: string): DegradedAction {
    switch (this.config.fallbackBehavior) {
      case 'queue':
        // Save for later review
        if (this.queue.length < this.config.maxQueueSize) {
          console.log(`[Degradation] Queued for later review: ${action.slice(0, 50)}`)
          return {
            action,
            originalRisk: risk,
            degradedDecision: 'queued',
            reason: 'No human available — queued for later review',
          }
        }
        // Queue full — reject
        return {
          action,
          originalRisk: risk,
          degradedDecision: 'rejected',
          reason: 'Queue full and no human available',
        }

      case 'auto_approve_low_risk':
        if (risk === 'low' || risk === 'medium') {
          console.log(`[Degradation] Auto-approving ${risk}-risk action`)
          return {
            action,
            originalRisk: risk,
            degradedDecision: 'auto_approved',
            reason: `Auto-approved: ${risk} risk, no human available`,
          }
        }
        return {
          action,
          originalRisk: risk,
          degradedDecision: 'rejected',
          reason: `${risk}-risk action requires human approval — no human available`,
        }

      case 'reject_all':
        return {
          action,
          originalRisk: risk,
          degradedDecision: 'rejected',
          reason: 'All actions rejected when no human is available',
        }

      case 'use_defaults':
        return {
          action,
          originalRisk: risk,
          degradedDecision: 'default_applied',
          reason: 'Applied default safe action — no human available',
        }
    }
  }

  getPendingQueue(): Array<{ action: string; risk: string; timestamp: number }> {
    return this.queue.map(({ action, risk, timestamp }) => ({
      action,
      risk,
      timestamp,
    }))
  }
}

// Usage
const degradation = new GracefulDegradation({
  approvalTimeoutMs: 5000,
  fallbackBehavior: 'auto_approve_low_risk',
  notifyOnDegradation: true,
  maxQueueSize: 100,
})

// Simulate: human is not available
const result = await degradation.requestApproval(
  'Send confirmation email to customer@example.com',
  'medium',
  () => false, // Human not available
  async action => 'approved' // Would not be called
)

console.log(result)
// { degradedDecision: "auto_approved", reason: "Auto-approved: medium risk, no human available" }

// Simulate: high-risk action, human not available
const highRiskResult = await degradation.requestApproval(
  'Delete all inactive user accounts',
  'critical',
  () => false,
  async action => 'approved'
)

console.log(highRiskResult)
// { degradedDecision: "rejected", reason: "critical-risk action requires human approval" }
```

### Escalation Chains

When the primary reviewer is unavailable, escalate to backups:

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
  timeoutMs: number = 30000
): Promise<{ decision: string; reviewerId: string; escalated: boolean }> {
  for (let i = 0; i < reviewers.length; i++) {
    const reviewer = reviewers[i]

    if (!reviewer.available()) {
      console.log(`[Escalation] ${reviewer.name} unavailable, trying next...`)
      continue
    }

    try {
      const decision = await Promise.race([
        reviewer.review(action),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error('timeout')), timeoutMs)),
      ])

      return {
        decision,
        reviewerId: reviewer.id,
        escalated: i > 0,
      }
    } catch {
      console.log(`[Escalation] ${reviewer.name} timed out, trying next...`)
    }
  }

  return {
    decision: 'rejected',
    reviewerId: 'system',
    escalated: true,
  }
}

// Usage
const reviewers: Reviewer[] = [
  {
    id: 'primary',
    name: 'Primary Reviewer',
    available: () => false, // Simulating unavailable
    review: async action => 'approved',
  },
  {
    id: 'backup',
    name: 'Backup Reviewer',
    available: () => true,
    review: async action => {
      console.log(`[Backup] Reviewing: ${action.slice(0, 50)}`)
      return 'approved'
    },
  },
  {
    id: 'manager',
    name: 'Manager',
    available: () => true,
    review: async action => 'approved',
  },
]

const approval = await escalatingApproval('Approve $5,000 refund for customer complaint', reviewers)

console.log(`Decision: ${approval.decision}`)
console.log(`Reviewed by: ${approval.reviewerId}`)
console.log(`Escalated: ${approval.escalated}`)
```

> **Beginner Note:** Graceful degradation means your system keeps working even when ideal conditions are not met. Just as a car still works without air conditioning (less comfortable but functional), your agent system should still work without a human reviewer (less capable but safe).

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

This completes Part IV: Agents and Orchestration. You now have the patterns to build autonomous agents, coordinate multiple agents, design deterministic pipelines, generate code iteratively, and add human oversight — the full toolkit for production LLM applications.
