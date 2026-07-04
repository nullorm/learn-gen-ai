// TODO (Module 15, Section 4): implement processOrder(orderId).
// - two steps: reserveStock(orderId) -> { reserved: true }, chargeCard(orderId) -> { chargeId: 'ch_' + orderId }
// - return { orderId, chargeId, reserved: true }
export async function processOrder(_orderId: string): Promise<{ orderId: string; chargeId: string; reserved: boolean }> {
  'use workflow'
  throw new Error('TODO: implement processOrder (Module 15, Section 4)')
}
