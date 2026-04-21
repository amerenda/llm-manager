"""
Ollama library scraper and model classification.

Scrapes https://ollama.com/library for the model catalog,
caches in postgres, classifies models as safe/unsafe.
"""

import asyncio
import fnmatch
import logging
import re
from html.parser import HTMLParser
from typing import Optional

import asyncpg
import httpx

import db

logger = logging.getLogger(__name__)

OLLAMA_LIBRARY_URL = "https://ollama.com/library"
OLLAMA_REGISTRY = "https://registry.ollama.ai"


# ── HTML parsing ──────────────────────────────────────────────────────────────

class LibraryPageParser(HTMLParser):
    """Parse the Ollama library index page to extract model cards.

    Key HTML attributes:
    - x-test-model-title: model card container
    - x-test-size: parameter size tags (8b, 70b, etc)
    - x-test-capability: capability tags (tools, vision, etc)
    - x-test-pull-count: download count
    """

    def __init__(self):
        super().__init__()
        self.models = []
        self._current = None
        self._in_card = False
        self._in_desc = False
        self._in_size = False
        self._in_capability = False
        self._in_pulls = False
        self._capture = ""

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href", "")

        # Model card link: /library/{name}
        if tag == "a" and href.startswith("/library/") and href.count("/") == 2:
            name = href.split("/")[-1]
            if name and name not in ("", "search"):
                self._current = {
                    "name": name,
                    "description": "",
                    "pulls": "",
                    "parameter_sizes": [],
                    "categories": [],
                }
                self._in_card = True

        if not self._current:
            return

        # Description paragraph
        if tag == "p" and "break-words" in attrs_dict.get("class", ""):
            self._in_desc = True
            self._capture = ""

        # Parameter size span (x-test-size attribute)
        if "x-test-size" in attrs_dict:
            self._in_size = True
            self._capture = ""

        # Capability span (x-test-capability attribute)
        if "x-test-capability" in attrs_dict:
            self._in_capability = True
            self._capture = ""

        # Pull count span (x-test-pull-count attribute)
        if "x-test-pull-count" in attrs_dict:
            self._in_pulls = True
            self._capture = ""

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_desc:
            self._capture += text
        elif self._in_size:
            self._capture += text
        elif self._in_capability:
            self._capture += text
        elif self._in_pulls:
            self._capture += text

    def handle_endtag(self, tag):
        if not self._current:
            return

        if self._in_desc and tag == "p":
            self._current["description"] = self._capture.strip()
            self._in_desc = False

        if self._in_size and tag == "span":
            val = self._capture.strip().lower()
            if val and val not in self._current["parameter_sizes"]:
                self._current["parameter_sizes"].append(val)
            self._in_size = False

        if self._in_capability and tag == "span":
            val = self._capture.strip().lower()
            if val and val not in self._current["categories"]:
                self._current["categories"].append(val)
            self._in_capability = False

        if self._in_pulls and tag == "span":
            self._current["pulls"] = self._capture.strip()
            self._in_pulls = False

        # End of card
        if tag == "a" and self._in_card and self._current:
            self._in_card = False
            if self._current["name"]:
                self.models.append(self._current)
            self._current = None


class ModelTagsParser(HTMLParser):
    """Parse a model's tag page to extract available tags with sizes."""

    def __init__(self):
        super().__init__()
        self.tags = []
        self._capture = ""
        self._in_row = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href", "")
        if tag == "a" and "/library/" in href and ":" in href.split("/")[-1]:
            self._in_row = True
            self._capture = ""

    def handle_data(self, data):
        if self._in_row:
            self._capture += " " + data.strip()

    def handle_endtag(self, tag):
        if tag == "a" and self._in_row:
            self._in_row = False
            text = self._capture.strip()
            if text:
                self._parse_tag_text(text)
            self._capture = ""

    def _parse_tag_text(self, text: str):
        """Parse tag text like 'qwen2.5:7b-q4_K_M 4.9 GB'."""
        parts = text.split()
        if not parts:
            return
        tag_name = parts[0]
        size_gb = None
        for i, p in enumerate(parts):
            if p in ("GB", "MB") and i > 0:
                try:
                    val = float(parts[i - 1])
                    size_gb = val if p == "GB" else val / 1024
                except ValueError:
                    pass
        # Extract quantization and param count
        quant = parse_quantization(tag_name)
        params = parse_param_count(tag_name)
        self.tags.append({
            "tag": tag_name,
            "size_gb": round(size_gb, 2) if size_gb else None,
            "quantization": quant,
            "parameter_count": params,
        })


# ── Name parsing helpers ──────────────────────────────────────────────────────

def parse_param_count(name: str) -> Optional[str]:
    """Extract parameter count from model name. e.g. 'llama3:8b' -> '8b'."""
    m = re.search(r'(\d+\.?\d*)[bB]', name)
    return m.group(0).lower() if m else None


def parse_quantization(name: str) -> Optional[str]:
    """Extract quantization from model tag. e.g. '7b-q4_K_M' -> 'q4_K_M'."""
    m = re.search(r'(q\d+_\w+|fp16|fp32|int8|int4)', name, re.IGNORECASE)
    return m.group(0) if m else None


# ── Scraping ──────────────────────────────────────────────────────────────────

async def scrape_library_index() -> list[dict]:
    """Fetch and parse the Ollama library index page."""
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.get(OLLAMA_LIBRARY_URL)
            r.raise_for_status()
        parser = LibraryPageParser()
        parser.feed(r.text)
        logger.info("Scraped %d models from Ollama library", len(parser.models))
        return parser.models
    except Exception:
        logger.exception("Failed to scrape Ollama library")
        return []


async def scrape_model_tags(model_name: str) -> list[dict]:
    """Fetch and parse a model's tag page."""
    try:
        url = f"{OLLAMA_LIBRARY_URL}/{model_name}/tags"
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.get(url)
            r.raise_for_status()
        parser = ModelTagsParser()
        parser.feed(r.text)
        return parser.tags
    except Exception:
        logger.exception("Failed to scrape tags for %s", model_name)
        return []


# ── Cache management ──────────────────────────────────────────────────────────

async def refresh_library_cache(pool: asyncpg.Pool, force: bool = False) -> dict:
    """Refresh the library cache. Returns status info."""
    if not force:
        age = await db.get_library_cache_age_hours(pool)
        if age is not None and age < 24:
            return {"status": "cached", "age_hours": round(age, 1)}

    models = await scrape_library_index()
    if not models:
        return {"status": "error", "message": "Failed to scrape library"}

    for m in models:
        await db.upsert_library_model(
            pool,
            name=m["name"],
            description=m.get("description", ""),
            pulls=m.get("pulls", ""),
            tags_json=m.get("tags", []),
            parameter_sizes=m.get("parameter_sizes", []),
            categories=m.get("categories", []),
        )

    await db.set_library_cache_meta(pool, "last_full_refresh", str(len(models)))
    logger.info("Library cache refreshed: %d models", len(models))
    return {"status": "refreshed", "model_count": len(models)}


async def ensure_model_tags(pool: asyncpg.Pool, model_name: str) -> list[dict]:
    """Ensure tags are loaded for a model. Scrape if missing."""
    row = await db.get_library_model(pool, model_name)
    if row and row.get("tags_json") and len(row["tags_json"]) > 0:
        return row["tags_json"]

    tags = await scrape_model_tags(model_name)
    if tags and row:
        await db.upsert_library_model(
            pool,
            name=model_name,
            description=row.get("description", ""),
            pulls=row.get("pulls", ""),
            tags_json=tags,
            parameter_sizes=row.get("parameter_sizes", []),
            categories=row.get("categories", []),
        )
    return tags


# ── Safety classification ────────────────────────────────────────────────────

# ── Remote manifest digest lookup (for "update available" detection) ────────

async def fetch_remote_manifest_digest(
    model_tag: str,
    max_retries: int = 2,
) -> tuple[str, Optional[str]]:
    """Return (digest, error) for a model:tag from registry.ollama.ai.

    Handles both library models (`qwen3.5:9b` → `library/qwen3.5:9b`) and
    user-namespaced models (`MFDoom/deepseek-r1-...:tag` → used as-is).

    Uses HEAD on the manifest endpoint so we get the content digest via the
    `Docker-Content-Digest` header without downloading the manifest body.
    Accepts the Ollama-specific manifest media type in addition to OCI.

    Retries with exponential backoff on 429 (rate limit) and 5xx.
    """
    # Split off tag
    if ":" in model_tag:
        path, tag = model_tag.rsplit(":", 1)
    else:
        path, tag = model_tag, "latest"

    # Library prefix for official models (no namespace)
    if "/" not in path:
        path = f"library/{path}"

    url = f"{OLLAMA_REGISTRY}/v2/{path}/manifests/{tag}"
    headers = {
        "Accept": "application/vnd.docker.distribution.manifest.v2+json, "
                  "application/vnd.oci.image.manifest.v1+json, "
                  "application/vnd.ollama.image.manifest+json, "
                  "application/json",
    }

    backoff = 1.0  # seconds — doubles on each retry
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
                r = await c.head(url, headers=headers)
                # 429 Too Many Requests, 503 Service Unavailable → back off
                if r.status_code in (429, 503) and attempt < max_retries:
                    # Honor Retry-After header if provided
                    retry_after = r.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                    logger.info("Registry %s on %s — backing off %.1fs (attempt %d/%d)",
                                r.status_code, model_tag, delay, attempt + 1, max_retries + 1)
                    await asyncio.sleep(delay)
                    backoff *= 2
                    continue
                if r.status_code == 404:
                    return "", "not found"
                if r.status_code >= 400:
                    return "", f"HTTP {r.status_code}"
                digest = r.headers.get("Docker-Content-Digest") or r.headers.get("docker-content-digest") or ""
                if not digest:
                    # Some registries don't send the header on HEAD — fall back to GET
                    r = await c.get(url, headers=headers)
                    digest = r.headers.get("Docker-Content-Digest", "")
                    if not digest and r.status_code == 200:
                        # Last resort: compute digest ourselves from the manifest JSON
                        import hashlib
                        digest = "sha256:" + hashlib.sha256(r.content).hexdigest()
                return digest, None
        except Exception as e:
            if attempt >= max_retries:
                return "", str(e)
            logger.info("Registry request failed (%s) — retrying in %.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff *= 2
    return "", "max retries exceeded"


async def refresh_remote_manifests(
    pool: asyncpg.Pool,
    model_names: list[str],
    min_age_seconds: int = 3600,
    inter_request_delay: float = 0.25,
) -> dict:
    """Fetch remote manifest digests for each model:tag and cache in
    ollama_remote_manifests.

    Rate-limit aware:
    - Skips tags checked successfully within the last min_age_seconds
      (default 1h) so repeated calls are cheap.
    - Sleeps inter_request_delay between requests to avoid bursts.
    - fetch_remote_manifest_digest itself retries with exponential backoff
      on 429/503.

    Returns {checked, errors, skipped}."""
    if not model_names:
        return {"checked": 0, "errors": 0, "skipped": 0}

    # Fetch current cache state to decide which tags to skip
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT model_name, digest, error,
                   EXTRACT(EPOCH FROM (NOW() - checked_at))::int AS age_sec
            FROM ollama_remote_manifests
            WHERE model_name = ANY($1::text[])
            """,
            list(set(model_names)),
        )
    cached = {r["model_name"]: r for r in rows}

    checked = 0
    errors = 0
    skipped = 0
    unique_names = sorted(set(model_names))
    for i, name in enumerate(unique_names):
        row = cached.get(name)
        # Skip recently-successful entries — re-fetch only if old, missing, or errored
        if row and row["age_sec"] < min_age_seconds and row["digest"] and not row["error"]:
            skipped += 1
            continue

        if i > 0:
            await asyncio.sleep(inter_request_delay)

        digest, err = await fetch_remote_manifest_digest(name)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ollama_remote_manifests (model_name, digest, checked_at, error)
                VALUES ($1, $2, NOW(), $3)
                ON CONFLICT (model_name) DO UPDATE SET
                    digest = EXCLUDED.digest,
                    checked_at = EXCLUDED.checked_at,
                    error = EXCLUDED.error
                """,
                name, digest, err,
            )
        if err:
            errors += 1
        else:
            checked += 1

    logger.info(
        "Remote manifests refreshed: %d checked, %d errors, %d skipped (cached)",
        checked, errors, skipped,
    )
    return {"checked": checked, "errors": errors, "skipped": skipped}


async def classify_model(pool: asyncpg.Pool, model_name: str) -> str:
    """Classify a model as 'safe' or 'unsafe' based on pattern rules."""
    tags = await db.get_safety_tags(pool)
    for tag in tags:
        if fnmatch.fnmatch(model_name.lower(), tag["pattern"].lower()):
            return tag["classification"]
    return "safe"  # default: untagged models are safe


async def classify_models_batch(pool: asyncpg.Pool, model_names: list[str]) -> dict[str, str]:
    """Classify multiple models in one DB query."""
    tags = await db.get_safety_tags(pool)
    result = {}
    for name in model_names:
        classification = "safe"
        for tag in tags:
            if fnmatch.fnmatch(name.lower(), tag["pattern"].lower()):
                classification = tag["classification"]
                break
        result[name] = classification
    return result
