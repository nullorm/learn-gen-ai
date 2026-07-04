import { expect, test } from 'vitest'
import { start, resumeHook } from 'workflow/api'
import { waitForHook } from '@workflow/vitest'
import { approvalWorkflow } from '../src/approval'

// Note: against the empty stub this test times out (nothing ever suspends). Once you
// implement the hook, it suspends, `waitForHook` catches it, and the assertion runs.
test('approvalWorkflow suspends on a hook and resumes with the decision', async () => {
  const run = await start(approvalWorkflow, ['tok-1'])
  const hook = await waitForHook(run, { token: 'tok-1' })
  await resumeHook(hook.token, { approved: true })
  expect(await run.returnValue).toBe('approved')
})
