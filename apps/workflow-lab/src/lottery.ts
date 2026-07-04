// Module 15, Section 3 (Debug): this workflow VIOLATES the determinism rule.
// Math.random() and Date.now() in a `'use workflow'` body produce different values on
// every replay, corrupting a durable run. Diagnose why, then FIX it by moving the
// non-determinism into a `'use step'` (e.g. drawTicket()) so the values are drawn once
// and replayed. Keep the return shape { ticket, drawnAt }.
export async function lotteryWorkflow(): Promise<{ ticket: number; drawnAt: number }> {
  'use workflow'
  const ticket = Math.floor(Math.random() * 100) // BUG: non-deterministic in a workflow body
  const drawnAt = Date.now() // BUG: non-deterministic in a workflow body
  return { ticket, drawnAt }
}
