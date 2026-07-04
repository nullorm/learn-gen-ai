// TODO (Module 15, Section 6): implement approvalWorkflow using the module-level approvalHook.
// - approvalWorkflow(token) suspends on approvalHook.create({ token }) and returns
//   'approved' or 'rejected' based on the first event.
import { defineHook } from 'workflow'

export const approvalHook = defineHook<{ approved: boolean }>()

export async function approvalWorkflow(_token: string): Promise<string> {
  'use workflow'
  throw new Error('TODO: implement approvalWorkflow (Module 15, Section 6)')
}
