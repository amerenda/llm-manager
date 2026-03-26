"""Library browser and safety tag management routes."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import db
from library import (
    refresh_library_cache,
    ensure_model_tags,
    classify_models_batch,
    parse_param_count,
    parse_quantization,
)
from gpu import vram_for_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/library", tags=["library"])
safety_router = APIRouter(prefix="/api/safety-tags", tags=["safety"])


def _parse_jsonb(val):
    """asyncpg may return JSONB as strings — parse them."""

    if isinstance(val, str):
        import json
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return []
    return val if val is not None else []


def _get_pool(request: Request):
    return request.app.state.db


# ── Library browse ────────────────────────────────────────────────────────────

@router.get("")
async def browse_library(
    request: Request,
    search: Optional[str] = None,
    safety: Optional[str] = None,  # all | safe | unsafe
    fits: Optional[bool] = None,  # true = only models that fit
    downloaded: Optional[bool] = None,  # true = only downloaded
    sort: Optional[str] = "name",  # name | pulls | vram
):
    pool = _get_pool(request)

    # Ensure cache exists
    cache_age = await db.get_library_cache_age_hours(pool)
    if cache_age is None:
        await refresh_library_cache(pool, force=True)

    # Get library models
    library_models = await db.get_library_models(pool, search=search)

    # Get downloaded models from all runners
    downloaded_names = set()
    loaded_names = set()
    runner_vram = {}
    # Track per-runner downloads for richer UI data
    per_runner_downloads: dict[str, set[str]] = {}
    try:
        runners = await db.get_active_runners(pool)
        if runners:
            import httpx, re
            async with httpx.AsyncClient(timeout=10) as c:
                for r in runners:
                    addr = r["address"]
                    host = re.sub(r'^https?://', '', addr)
                    host = re.sub(r':\d+$', '', host)
                    ollama_base = f"http://{host}:11434"
                    hostname = r["hostname"]
                    per_runner_downloads[hostname] = set()
                    try:
                        tags_resp = await c.get(f"{ollama_base}/api/tags")
                        if tags_resp.status_code == 200:
                            for m in tags_resp.json().get("models", []):
                                downloaded_names.add(m["name"])
                                per_runner_downloads[hostname].add(m["name"])
                        ps_resp = await c.get(f"{ollama_base}/api/ps")
                        if ps_resp.status_code == 200:
                            for m in ps_resp.json().get("models", []):
                                loaded_names.add(m["name"])
                    except Exception:
                        logger.warning("Failed to query Ollama on runner %s at %s", hostname, ollama_base)
                    caps = r.get("capabilities", {})
                    if isinstance(caps, dict):
                        total = caps.get("gpu_vram_total_bytes", 0) / (1024**3)
                        if total > 0:
                            runner_vram[hostname] = round(total, 1)
    except Exception:
        logger.warning("Failed to get runner info for library browse")

    # Classify all models
    all_names = [m["name"] for m in library_models]
    safety_map = await classify_models_batch(pool, all_names)

    # Build response
    results = []
    for m in library_models:
        name = m["name"]
        param_sizes = _parse_jsonb(m.get("parameter_sizes", []))
        categories = _parse_jsonb(m.get("categories", []))
        # Estimate VRAM for the default (smallest) variant
        default_params = param_sizes[0] if param_sizes else None
        vram_est = vram_for_model(f"{name}:{default_params}") if default_params else vram_for_model(name)

        model_safety = safety_map.get(name, "safe")
        is_downloaded = any(d.startswith(name) for d in downloaded_names)
        fits_on = [
            {"runner": hostname, "vram_total_gb": vram}
            for hostname, vram in runner_vram.items()
            if vram >= vram_est
        ]
        model_fits = len(fits_on) > 0 or not runner_vram  # if no runners, don't filter

        # Apply filters
        if safety and safety != "all":
            if safety == "safe" and model_safety != "safe":
                continue
            if safety == "unsafe" and model_safety != "unsafe":
                continue
        if fits is not None:
            if fits and not model_fits:
                continue
            if not fits and model_fits:
                continue
        if downloaded is not None:
            if downloaded and not is_downloaded:
                continue
            if not downloaded and is_downloaded:
                continue

        # Which runners have this model downloaded?
        downloaded_on = [
            hostname for hostname, models in per_runner_downloads.items()
            if any(d.startswith(name) for d in models)
        ]

        results.append({
            "name": name,
            "description": m.get("description", ""),
            "pulls": m.get("pulls", ""),
            "parameter_sizes": param_sizes,
            "categories": categories,
            "safety": model_safety,
            "downloaded": is_downloaded,
            "downloaded_on": downloaded_on,
            "loaded": any(l.startswith(name) for l in loaded_names),
            "fits": model_fits,
            "fits_on": fits_on,
            "vram_estimate_gb": round(vram_est, 1),
        })

    # Sort results
    def _parse_pulls(p: str) -> float:
        """Parse '111.7M' -> 111700000."""
        if not p:
            return 0
        p = p.strip().upper()
        if p.endswith('B'):
            return float(p[:-1]) * 1e9
        if p.endswith('M'):
            return float(p[:-1]) * 1e6
        if p.endswith('K'):
            return float(p[:-1]) * 1e3
        try:
            return float(p)
        except ValueError:
            return 0

    if sort == "pulls":
        results.sort(key=lambda m: _parse_pulls(m["pulls"]), reverse=True)
    elif sort == "vram":
        results.sort(key=lambda m: m["vram_estimate_gb"])
    else:
        results.sort(key=lambda m: m["name"])

    return {
        "models": results,
        "total": len(results),
        "cache_age_hours": round(cache_age, 1) if cache_age else 0,
        "runners": list(runner_vram.keys()),
    }


@router.post("/refresh")
async def refresh_cache(request: Request):
    pool = _get_pool(request)
    result = await refresh_library_cache(pool, force=True)
    return result


@router.get("/{name}")
async def library_model_detail(name: str, request: Request):
    pool = _get_pool(request)
    model = await db.get_library_model(pool, name)
    if not model:
        raise HTTPException(404, f"Model '{name}' not found in library cache")

    # Lazy-load tags
    tags = await ensure_model_tags(pool, name)

    from library import classify_model
    safety = await classify_model(pool, name)

    return {
        "name": model["name"],
        "description": model.get("description", ""),
        "pulls": model.get("pulls", ""),
        "parameter_sizes": _parse_jsonb(model.get("parameter_sizes", [])),
        "categories": _parse_jsonb(model.get("categories", [])),
        "safety": safety,
        "tags": _parse_jsonb(tags) if isinstance(tags, str) else (tags or []),
    }


# ── Safety tags CRUD ──────────────────────────────────────────────────────────

class SafetyTagRequest(BaseModel):
    pattern: str
    classification: str = "unsafe"  # safe | unsafe
    reason: str = ""


@safety_router.get("")
async def list_safety_tags(request: Request):
    pool = _get_pool(request)
    return await db.get_safety_tags(pool)


@safety_router.post("")
async def create_safety_tag(body: SafetyTagRequest, request: Request):
    pool = _get_pool(request)
    if body.classification not in ("safe", "unsafe"):
        raise HTTPException(400, "classification must be 'safe' or 'unsafe'")
    tag_id = await db.create_safety_tag(pool, body.pattern, body.classification, body.reason)
    return {"ok": True, "id": tag_id}


@safety_router.put("/{tag_id}")
async def update_safety_tag(tag_id: int, body: SafetyTagRequest, request: Request):
    pool = _get_pool(request)
    ok = await db.update_safety_tag(pool, tag_id, body.pattern, body.classification, body.reason)
    if not ok:
        raise HTTPException(404, "Safety tag not found")
    return {"ok": True}


@safety_router.delete("/{tag_id}")
async def delete_safety_tag(tag_id: int, request: Request):
    pool = _get_pool(request)
    ok = await db.delete_safety_tag(pool, tag_id)
    if not ok:
        raise HTTPException(404, "Safety tag not found")
    return {"ok": True}


# ── App unsafe permission ─────────────────────────────────────────────────────

class AllowUnsafeRequest(BaseModel):
    allow_unsafe: bool


@router.put("/apps/{app_id}/allow-unsafe")
async def set_allow_unsafe(app_id: int, body: AllowUnsafeRequest, request: Request):
    pool = _get_pool(request)
    ok = await db.set_app_allow_unsafe(pool, app_id, body.allow_unsafe)
    if not ok:
        raise HTTPException(404, "App not found")
    return {"ok": True}
