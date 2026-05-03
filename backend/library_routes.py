"""Library browser and safety tag management routes."""
import asyncio
import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import db
from llm_agent import client_from_runner_row
from library import (
    refresh_library_cache,
    ensure_model_tags,
    classify_models_batch,
    parse_param_count,
    parse_quantization,
)
from gpu import vram_for_model

logger = logging.getLogger(__name__)

# Library browse probes each runner's /v1/status; use a tighter connect budget than
# pulls so a dead agent does not block the UI for LLM_AGENT_CONNECT_TIMEOUT (minutes).
def _library_probe_connect_seconds() -> float:
    raw = (os.environ.get("LLM_AGENT_LIBRARY_PROBE_CONNECT_TIMEOUT") or "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 12.0


def _norm_digest(d: Optional[str]) -> str:
    """Normalize a manifest digest for comparison. Ollama's /api/tags reports
    bare hex ('6488c96fa...') while the registry's Docker-Content-Digest
    header returns prefixed form ('sha256:6488c96fa...'). Without this, every
    comparison is a false positive and the library looks perpetually outdated
    — which is exactly the bug that made "update all" do nothing useful."""
    if not d:
        return ""
    d = d.strip().lower()
    if d.startswith("sha256:"):
        d = d[len("sha256:"):]
    return d


async def _get_runner_client(pool, runner_id: int):
    runners = await db.get_active_runners(pool)
    r = next((x for x in runners if x["id"] == runner_id), None)
    if not r:
        raise HTTPException(503, "Runner not found")
    psk = os.environ.get("LLM_MANAGER_AGENT_PSK", "")
    return client_from_runner_row(r, psk)


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
    runner_id: Optional[int] = None,  # scope everything to this runner
):
    """Browse the Ollama library.

    When runner_id is set, everything scopes to that runner:
      - downloaded / downloaded_on / outdated_on reflect ONLY that runner
      - fits / fits_on use only that runner's VRAM
      - community models (downloaded on that runner but not in catalog)
        are NOT added here; caller uses /api/library/community?runner_id=...
        for those.
    When runner_id is None, behavior is the existing fleet-wide aggregate.
    """
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
    runners: list = []
    try:
        runners = await db.get_active_runners(pool)
        if runner_id is not None:
            runners = [r for r in runners if r["id"] == runner_id]
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

    # Overlay VRAM from live /v1/status so fits_on / filters work for unified-memory
    # (mac-mini-m4) and any host where heartbeat gpu_vram_total_bytes is 0 or stale.
    async def _live_total_gb(runner: dict) -> tuple[str, float]:
        hostname = runner["hostname"]
        try:
            psk = os.environ.get("LLM_MANAGER_AGENT_PSK", "")
            client = client_from_runner_row(runner, psk)
            _ac = _library_probe_connect_seconds()
            st = await client.status(
                timeout=httpx.Timeout(connect=_ac, read=25.0, write=_ac, pool=45.0)
            )
            g = st.get("gpu_vram_total_gb")
            if g is not None:
                return hostname, float(g)
        except Exception:
            logger.debug("library: live VRAM probe failed for %s", hostname, exc_info=True)
        return hostname, 0.0

    try:
        if runners:
            probed = await asyncio.gather(*(_live_total_gb(r) for r in runners))
            for hostname, gb in probed:
                if gb > 0:
                    runner_vram[hostname] = round(gb, 1)
    except Exception:
        logger.warning("library: live VRAM overlay failed", exc_info=True)

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
        # local digest differing from the remote_manifests cache. Normalize
        # both sides — local digest is bare hex from Ollama's /api/tags,
        # remote digest is sha256:-prefixed from the registry header.
        outdated_on: list[str] = []
        for hostname, digests in per_runner_digests.items():
            for model_tag, local_digest in digests.items():
                if not model_tag.startswith(name):
                    continue
                remote = remote_digests.get(model_tag)
                if remote and local_digest and _norm_digest(remote) != _norm_digest(local_digest):
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


@router.get("/community")
async def community_models(
    request: Request,
    runner_id: Optional[int] = None,
):
    """Models downloaded on a runner that aren't in the Ollama library
    catalog — community / user-namespaced / local models (e.g.
    MFDoom/deepseek-r1-tool-calling:... or imported from hf.co/...).

    Catalog scrape only covers ollama.com/library/*, so anything else
    never shows up in /api/library. This endpoint fills the gap: reads
    each runner's capabilities.downloaded_models, filters out anything
    whose base name appears in the catalog, returns what's left with
    runner+digest+outdated info.
    """
    pool = _get_pool(request)
    catalog_names = {m["name"] for m in await db.get_library_models(pool, search=None)}

    runners = await db.get_active_runners(pool)
    if runner_id is not None:
        runners = [r for r in runners if r["id"] == runner_id]

    # Build remote digest cache for outdated comparison
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT model_name, digest FROM ollama_remote_manifests WHERE digest != ''"
        )
        remote_digests = {r["model_name"]: r["digest"] for r in rows}

    # Aggregate by tag across runners
    tag_info: dict[str, dict] = {}
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
            if not tag:
                continue
            # Base name is everything before the first `:` (tag separator).
            # If it matches a catalog entry, skip — handled by /api/library.
            base = tag.split(":", 1)[0]
            if base in catalog_names:
                continue
            t = tag_info.setdefault(tag, {
                "name": tag,
                "downloaded_on": [],
                "outdated_on": [],
                "size_bytes": entry.get("size_bytes", 0),
                "digest": entry.get("digest", ""),
            })
            if r["hostname"] not in t["downloaded_on"]:
                t["downloaded_on"].append(r["hostname"])
            # Outdated detection
            rdig = remote_digests.get(tag)
            ldig = entry.get("digest", "")
            if rdig and ldig and _norm_digest(rdig) != _norm_digest(ldig):
                if r["hostname"] not in t["outdated_on"]:
                    t["outdated_on"].append(r["hostname"])

    return {"models": sorted(tag_info.values(), key=lambda m: m["name"])}


@router.post("/refresh-remote-digests")
async def refresh_remote_digests_endpoint(
    request: Request,
    force: bool = False,
    runner_id: Optional[int] = None,
):
    """Re-check each currently-downloaded tag against registry.ollama.ai
    and store the remote digest. Library view's `outdated_on` reads from
    this cache.

    force=false (default): tags with a successful cache entry newer than 1h
      are skipped to avoid hammering the registry. Perfect for the CronJob.
    force=true: re-fetch every tag. Use from the UI button when the user
      explicitly asks.
    runner_id=X: only refresh tags downloaded on that runner. Unset =
      fleet-wide.
    """
    from library import refresh_remote_manifests
    pool = _get_pool(request)
    runners = await db.get_active_runners(pool)
    if runner_id is not None:
        runners = [r for r in runners if r["id"] == runner_id]
    tags: set[str] = set()
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
        scope = f"runner {runner_id}" if runner_id else "any runner"
        return {"status": "idle", "message": f"No downloaded models on {scope}"}
    min_age = 0 if force else 3600
    result = await refresh_remote_manifests(pool, sorted(tags), min_age_seconds=min_age)
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
    target a specific runner; otherwise picks the first non-draining runner
    that already has this model, falling back to the first non-draining
    runner if none have it yet. Draining runners are skipped on the auto-
    pick path — admin intent wins."""
    pool = _get_pool(request)
    target: Optional[int] = runner_id
    if target is None:
        runners = await db.get_active_runners(pool)
        non_draining = [r for r in runners if not r.get("draining")]
        for r in non_draining:
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
        if target is None and non_draining:
            target = non_draining[0]["id"]
    if target is None:
        raise HTTPException(503, "No active runners to pull on (all draining or offline)")
    return await _pull_on_runner(request, target, name)


@router.post("/update-outdated")
async def update_outdated_models(
    request: Request,
    runner_id: Optional[int] = None,
    refresh: bool = True,
):
    """Re-pull every downloaded tag whose local digest differs from the
    cached remote digest.

    refresh=True (default): call refresh_remote_manifests(min_age=1h)
      first so the cache is warm. Recent entries are skipped — cheap on
      repeat calls, correct when the cache is empty.
    refresh=False: compare against current cache, no registry hits.
    runner_id=X: only scan that runner's downloaded_models and only
      pull on that runner. Unset = fleet-wide.
    """
    from library import refresh_remote_manifests
    pool = _get_pool(request)

    # Build the set of tags in scope (so auto-refresh only hits what we'll compare)
    runners = await db.get_active_runners(pool)
    if runner_id is not None:
        runners = [r for r in runners if r["id"] == runner_id]
    scope_tags: set[str] = set()
    per_runner_models: dict[int, list[dict]] = {}
    for r in runners:
        caps = r.get("capabilities") or {}
        if isinstance(caps, str):
            import json as _j
            try:
                caps = _j.loads(caps)
            except Exception:
                caps = {}
        models = [m for m in (caps.get("downloaded_models") or []) if isinstance(m, dict)]
        per_runner_models[r["id"]] = models
        for m in models:
            if m.get("name"):
                scope_tags.add(m["name"])

    if refresh and scope_tags:
        # 1h min_age skip keeps this cheap — already-fresh entries aren't re-fetched
        await refresh_remote_manifests(pool, sorted(scope_tags), min_age_seconds=3600)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT model_name, digest FROM ollama_remote_manifests WHERE digest != ''"
        )
        remote = {r["model_name"]: r["digest"] for r in rows}

    kicked: list[dict] = []
    skipped_no_remote: int = 0
    up_to_date: int = 0
    hostname_by_id = {r["id"]: r["hostname"] for r in runners}
    for rid, models in per_runner_models.items():
        host = hostname_by_id.get(rid, str(rid))
        for entry in models:
            tag = entry.get("name")
            local_digest = entry.get("digest")
            remote_digest = remote.get(tag)
            if not tag or not local_digest:
                continue
            if not remote_digest:
                # Remote never resolved (404 on registry, network blip, etc.)
                skipped_no_remote += 1
                continue
            if _norm_digest(local_digest) == _norm_digest(remote_digest):
                up_to_date += 1
                continue
            try:
                resp = await _pull_on_runner(request, rid, tag)
                kicked.append({"runner": host, "model": tag, "op_id": resp.get("op_id")})
            except Exception as e:
                kicked.append({"runner": host, "model": tag, "error": str(e)})
    return {
        "status": "ok",
        "pulls": kicked,
        "count": len(kicked),
        "up_to_date": up_to_date,
        "skipped_no_remote": skipped_no_remote,
        "scope": f"runner {runner_id}" if runner_id else "fleet",
    }


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
