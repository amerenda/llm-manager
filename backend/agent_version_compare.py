"""Compare llm-agent image version strings (build-time AGENT_VERSION vs global target)."""


def normalize_agent_version_tag(s: str) -> str:
    if not s:
        return ""
    t = str(s).strip().lower()
    if t.startswith("sha-"):
        t = t[4:]
    return t


def agent_versions_equivalent(a: str, b: str) -> bool:
    return normalize_agent_version_tag(a) == normalize_agent_version_tag(b)
