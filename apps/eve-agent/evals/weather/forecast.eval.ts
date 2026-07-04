import { defineEval } from 'eve/evals'
import { includes } from 'eve/evals/expect'

export default defineEval({
  description: 'Uses get_weather and reports the condition',
  async test(t) {
    await t.send('What is the weather in Brooklyn?')
    t.succeeded()
    t.calledTool('get_weather')
    t.check(t.reply, includes('Sunny'))
  },
})
