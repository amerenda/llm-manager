import type { Runner } from '../types'

/** Match backend `get_active_runners` (`last_seen` within 90s). Small slack for clock skew. */
export const RUNNER_SCHEDULER_STALE_MS = 95_000

/** Runners the scheduler and library aggregate APIs treat as active. */
export function isRunnerInSchedulerPool(r: Runner): boolean {
  if (r.enabled === false) return false
  if (!r.last_seen) return false
  return Date.now() - new Date(r.last_seen).getTime() < RUNNER_SCHEDULER_STALE_MS
}
