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
