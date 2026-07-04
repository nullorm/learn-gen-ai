// TODO (Module 15, Section 7): implement paymentWorkflow(order) + a chargeOnce step.
// - chargeOnce(order) is a `'use step'` that throws `FatalError` when order.amount <= 0,
//   otherwise returns { chargeId: 'ch_' + order.id }. Import FatalError from 'workflow'.
// - paymentWorkflow(order) calls chargeOnce — write NO retry loop; the SDK handles retries.
export interface Order {
  id: string
  amount: number
}

export async function paymentWorkflow(_order: Order): Promise<{ chargeId: string }> {
  'use workflow'
  throw new Error('TODO: implement paymentWorkflow (Module 15, Section 7)')
}
