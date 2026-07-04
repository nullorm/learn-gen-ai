// TODO (Module 15, Section 2): implement greetWorkflow + a buildGreeting step.
// - greetWorkflow is a `'use workflow'` fn that calls the step buildGreeting(name).
// - buildGreeting is a `'use step'` fn returning `Hello, ${name}!`.
export async function greetWorkflow(_name: string): Promise<string> {
  'use workflow'
  throw new Error('TODO: implement greetWorkflow (Module 15, Section 2)')
}
