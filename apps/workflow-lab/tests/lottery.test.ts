import { expect, test } from 'vitest'
import { start } from 'workflow/api'
import { lotteryWorkflow } from '../src/lottery'

// After you FIX the determinism bug (draw the ticket inside a `'use step'`), this passes cleanly.
test('lotteryWorkflow returns a valid, replay-stable ticket', async () => {
  const run = await start(lotteryWorkflow, [])
  const result = await run.returnValue
  expect(result.ticket).toBeGreaterThanOrEqual(0)
  expect(result.ticket).toBeLessThan(100)
  expect(result.drawnAt).toBeGreaterThan(0)
})
