/** Match backend agent_version_compare.normalize_agent_version_tag */
export function normalizeAgentVersionTag(s: string | undefined): string {
  if (!s) return ''
  let t = s.trim().toLowerCase()
  if (t.startsWith('sha-')) t = t.slice(4)
  return t
}

/** True when running agent matches global target (ignores sha- prefix / case). */
export function agentVersionsEquivalent(a: string | undefined, b: string | undefined): boolean {
  return normalizeAgentVersionTag(a) === normalizeAgentVersionTag(b)
}
