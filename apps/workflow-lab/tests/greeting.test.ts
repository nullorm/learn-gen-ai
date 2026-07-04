import { expect, test } from 'vitest'
import { start } from 'workflow/api'
import { greetWorkflow } from '../src/greeting'

test('greetWorkflow runs a step and returns the greeting', async () => {
  const run = await start(greetWorkflow, ['Ada'])
  expect(await run.returnValue).toBe('Hello, Ada!')
})
