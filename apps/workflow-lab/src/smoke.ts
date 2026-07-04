export async function addWorkflow(a: number, b: number) {
  'use workflow'
  return await addStep(a, b)
}

async function addStep(a: number, b: number) {
  'use step'
  return a + b
}
