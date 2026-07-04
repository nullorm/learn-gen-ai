import { defineAgent } from 'eve'
import { mockModel } from 'eve/evals'

// Deterministic fixture model — `eve eval` runs offline with no provider key, and the
// course agent stays reproducible. It borrows a known model *identity*
// ('anthropic/claude-sonnet-5') only so eve's auto-compaction can find a context-window
// size; every response is scripted below, so nothing calls a real provider.
//
// Module 18 Section 6 shows the one-line swap to a real provider:
//   import { mistral } from '@ai-sdk/mistral'
//   export default defineAgent({ model: mistral('mistral-small-latest') }) // needs MISTRAL_API_KEY
export default defineAgent({
  model: mockModel({
    provider: 'anthropic',
    modelId: 'claude-sonnet-5',
    respond: ({ toolResults }) =>
      toolResults.length === 0
        ? { toolCalls: [{ name: 'get_weather', input: { city: 'Brooklyn' } }] }
        : `Weather in Brooklyn: ${JSON.stringify(toolResults[0]?.output)}`,
  }),
})
