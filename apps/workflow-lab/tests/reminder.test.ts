import { expect, test } from 'vitest'
import { start, getRun } from 'workflow/api'
import { waitForSleep } from '@workflow/vitest'
import { reminderWorkflow } from '../src/reminder'

// Note: against the empty stub this test times out (nothing ever suspends). Once you
// implement the sleep, it suspends, `waitForSleep` catches it, and the assertion runs.
test('reminderWorkflow suspends on a durable sleep and resumes to done', async () => {
  const run = await start(reminderWorkflow, [])
  const sleepId = await waitForSleep(run)
  await getRun(run.runId).wakeUp({ correlationIds: [sleepId] })
  expect(await run.returnValue).toBe('done')
})
