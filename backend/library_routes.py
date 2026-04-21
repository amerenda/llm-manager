"""Library browser and safety tag management routes."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import db
from llm_agent import LLMAgentClient
from library import (
    refresh_library_cache,
    ensure_model_tags,
    classify_models_batch,
    parse_param_count,
    parse_quantization,
)
from gpu import vram_for_model

logger = logging.getLogger(__name__)


async def _get_runner_client(pool, runner_id: int) -> LLMAgentClient:
    runners = await db.get_active_runners(pool)
    r = next((x for x in runners if x["id"] == runner_id), None)
    if not r:
        raise HTTPException(503, "Runner not found")
    return LLMAgentClient(r["address"])


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

    # Read per-runner downloaded / loaded state from capabilities.
    # Agent heartbeats (every 30s) populate `downloaded_models` and
    # `loaded_models` in capabilities — authoritative and cheap. Stops the
    # old per-request fanout to every agent's /v1/models.
    downloaded_names: set[str] = set()
    loaded_names: set[str] = set()
    runner_vram: dict[str, float] = {}
    per_runner_downloads: dict[str, set[str]] = {}
    # digest lookup for update-available detection (Phase B below)
    per_runner_digests: dict[str, dict[str, str]] = {}
    try:
        runners = await db.get_active_runners(pool)
        for r in runners:
            hostname = r["hostname"]
            per_runner_downloads[hostname] = set()
            per_runner_digests[hostname] = {}

            caps = r.get("capabilities") or {}
            if isinstance(caps, str):
                try:
                    import json as _json
                    caps = _json.loads(caps)
                except Exception:
                    caps = {}

            # downloaded_models is a list of {name, digest, size_bytes, modified_at}
            # in recent agents; fall back to the old /v1/models live query if
            # an older agent is still checked in (no key present).
            downloaded_entries = caps.get("downloaded_models")
            if downloaded_entries is None:
                # Old agent — live poll once. Noisy but only for legacy runners.
                try:
                    client = await _get_runner_client(pool, r["id"])
                    result = await client.models()
                    downloaded_entries = [{"name": m.get("id", ""), "digest": "", "size_bytes": 0} for m in result.get("data", [])]
                except Exception:
                    logger.warning("Legacy-agent fallback failed for %s", hostname)
                    downloaded_entries = []

            for entry in downloaded_entries:
                mid = entry.get("name", "") if isinstance(entry, dict) else str(entry)
                if not mid:
                    continue
                downloaded_names.add(mid)
                per_runner_downloads[hostname].add(mid)
                digest = entry.get("digest", "") if isinstance(entry, dict) else ""
                if digest:
                    per_runner_digests[hostname][mid] = digest

            for name in caps.get("loaded_models", []) or []:
                loaded_names.add(name)

            total = caps.get("gpu_vram_total_bytes", 0) / (1024**3)
            if total > 0:
                runner_vram[hostname] = round(total, 1)
    except Exception:
        logger.warning("Failed to get runner info for library browse", exc_info=True)

    # Classify all models
    all_names = [m["name"] for m in library_models]
    safety_map = await classify_models_batch(pool, all_names)

    # Cached remote-manifest digests (populated by periodic refresh job).
    # Used to mark a runner's local copy as out-of-date when the remote
    # tag's content has changed.
    remote_digests: dict[str, str] = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT model_name, digest FROM ollama_remote_manifests WHERE digest != ''"
            )
            remote_digests = {r["model_name"]: r["digest"] for r in rows}
    except Exception:
        logger.debug("ollama_remote_manifests not yet populated")

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

        # Update-available detection: any runner with this model that has a
        # local digest differing from the remote_manifests cache.
        outdated_on: list[str] = []
        for hostname, digests in per_runner_digests.items():
            for model_tag, local_digest in digests.items():
                if not model_tag.startswith(name):
                    continue
                remote = remote_digests.get(model_tag)
                if remote and local_digest and remote != local_digest:
                    if hostname not in outdated_on:
                        outdated_on.append(hostname)

        # Per-size fit info for the UI badges
        size_info = {}
        for ps in param_sizes:
            sv = round(vram_for_model(f"{name}:{ps}"), 1)
            sf = [h for h, v in runner_vram.items() if v >= sv]
            size_info[ps] = {"vram_gb": sv, "fits": len(sf) > 0 or not runner_vram}

        results.append({
            "name": name,
            "description": m.get("description", ""),
            "pulls": m.get("pulls", ""),
            "parameter_sizes": param_sizes,
            "categories": categories,
            "safety": model_safety,
            "downloaded": is_downloaded,
            "downloaded_on": downloaded_on,
            "outdated_on": outdated_on,
            "loaded": any(l.startswith(name) for l in loaded_names),
            "fits": model_fits,
            "fits_on": fits_on,
            "vram_estimate_gb": round(vram_est, 1),
            "size_info": size_info,
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


@router.post("/refresh-remote-digests")
async def refresh_remote_digests_endpoint(request: Request):
    """Re-check each currently-downloaded tag against registry.ollama.ai
    and store the remote digest. Library view's `outdated_on` reads from
    this cache. Safe to call on-demand — polls only tags actually downloaded
    on a runner."""
    from library import refresh_remote_manifests
    pool = _get_pool(request)
    tags: set[str] = set()
    runners = await db.get_active_runners(pool)
    for r in runners:
        caps = r.get("capabilities") or {}
        if isinstance(caps, str):
            import json as _j
            try:
                caps = _j.loads(caps)
            except Exception:
                caps = {}
        for entry in (caps.get("downloaded_models") or []):
            name = entry.get("name") if isinstance(entry, dict) else str(entry)
            if name:
                tags.add(name)
    if not tags:
        return {"status": "idle", "message": "No downloaded models on any runner"}
    result = await refresh_remote_manifests(pool, sorted(tags))
    return {"status": "ok", **result, "tags_checked": len(tags)}


async def _pull_on_runner(request: Request, runner_id: int, model: str) -> dict:
    """Internal helper: kick off a pull on a specific runner. Reuses the
    existing /api/llm/models/pull endpoint semantics via its helper (so the
    op appears in background_ops and the UI's failed/progress card)."""
    from main import llm_pull_model, LLMPullRequest  # avoid circular import
    req = LLMPullRequest(model=model)
    return await llm_pull_model(req, runner_id=runner_id)


@router.post("/models/{name:path}/force-update")
async def force_update_model(name: str, request: Request, runner_id: Optional[int] = None):
    """Run `ollama pull {name}` regardless of local state. Use runner_id to
    target a specific runner; otherwise picks the first active runner that
    already has this model (falling back to the first active runner)."""
    pool = _get_pool(request)
    target: Optional[int] = runner_id
    if target is None:
        runners = await db.get_active_runners(pool)
        for r in runners:
            caps = r.get("capabilities") or {}
            if isinstance(caps, str):
                import json as _j
                try:
                    caps = _j.loads(caps)
                except Exception:
                    caps = {}
            names = {e.get("name") for e in (caps.get("downloaded_models") or []) if isinstance(e, dict)}
            if name in names:
                target = r["id"]
                break
        if target is None and runners:
            target = runners[0]["id"]
    if target is None:
        raise HTTPException(503, "No active runners to pull on")
    return await _pull_on_runner(request, target, name)


@router.post("/update-outdated")
async def update_outdated_models(request: Request):
    """Re-pull every downloaded tag whose local digest differs from the
    cached remote digest. Returns the list of pulls kicked off.

    Does NOT refresh remote digests first — call /refresh-remote-digests
    before this for a full freshness pass. (Separated so a batch update
    doesn't double-hit registry.ollama.ai.)"""
    pool = _get_pool(request)
    # Load remote digests
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT model_name, digest FROM ollama_remote_manifests WHERE digest != ''"
        )
        remote = {r["model_name"]: r["digest"] for r in rows}
    if not remote:
        return {"status": "idle", "message": "No remote digests cached — run /refresh-remote-digests first"}

    kicked: list[dict] = []
    runners = await db.get_active_runners(pool)
    for r in runners:
        caps = r.get("capabilities") or {}
        if isinstance(caps, str):
            import json as _j
            try:
                caps = _j.loads(caps)
            except Exception:
                caps = {}
        for entry in (caps.get("downloaded_models") or []):
            if not isinstance(entry, dict):
                continue
            tag = entry.get("name")
            local_digest = entry.get("digest")
            remote_digest = remote.get(tag)
            if not tag or not local_digest or not remote_digest:
                continue
            if local_digest == remote_digest:
                continue  # up-to-date
            # Stale copy — pull on THIS runner so the local blob is refreshed.
            try:
                resp = await _pull_on_runner(request, r["id"], tag)
                kicked.append({"runner": r["hostname"], "model": tag, "op_id": resp.get("op_id")})
            except Exception as e:
                kicked.append({"runner": r["hostname"], "model": tag, "error": str(e)})
    return {"status": "ok", "pulls": kicked, "count": len(kicked)}


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
