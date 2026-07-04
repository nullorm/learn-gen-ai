// TODO (Module 15, Section 9): implement bookingAgent(messages) with WorkflowAgent.
// - new WorkflowAgent({ model: mistral('mistral-small-latest'), instructions, tools })
// - one tool whose `execute` is a `'use step'` (durable). Optionally needsApproval: true.
// - return { messages: result.messages }. Needs MISTRAL_API_KEY to run.
import type { UIMessage } from 'ai'

export async function bookingAgent(_messages: UIMessage[]): Promise<{ messages: unknown[] }> {
  'use workflow'
  throw new Error('TODO: implement bookingAgent (Module 15, Section 9)')
}
