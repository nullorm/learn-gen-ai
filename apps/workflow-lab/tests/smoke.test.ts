import { expect, test } from 'vitest'
import { start } from 'workflow/api'
import { addWorkflow } from '../src/smoke'

test('durable workflow runs a step and returns its result', async () => {
  const run = await start(addWorkflow, [2, 3])
  expect(await run.returnValue).toBe(5)
})
