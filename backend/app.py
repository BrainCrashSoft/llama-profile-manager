"""
app.py

FastAPI application exposing the REST + WebSocket API consumed by the
frontend. Kept intentionally simple (no ORM, no auth beyond localhost
binding) since this is a single-user local desktop tool.
"""

import asyncio
import json as _json
from datetime import datetime, timezone
import platform
import re
import subprocess
import traceback
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from . import __version__
from . import app_release, benchmark_runner, benchmarks as bench_store, command_builder, download_manager, gguf_meta, hf_avatar, hf_client, hf_search, llama_release, llama_server_download, models_preset, param_schema, process_manager, profiles, presets, router_dirs, scanner, settings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Llama Profile Manager")

# Populated by main.py once it knows which host/port uvicorn actually bound
# to, so the Settings page can tell the user whether - and where - this
# app's own UI (not llama-server's) is reachable from other devices.
runtime_info: Dict[str, Any] = {"bind_host": None, "port": None}

# Local-only tool served via pywebview/localhost; permissive CORS is fine here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):
    # Never leak raw stack traces to the UI; log server-side, return a clean message.
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": "Something went wrong on the backend. See server logs for details."})


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsUpdate(BaseModel):
    llama_servers: Optional[list] = None
    active_llama_server: Optional[str] = None
    llama_server_path: Optional[str] = None
    model_root_folders: Optional[list] = None
    default_host: Optional[str] = None
    default_port: Optional[int] = None
    theme: Optional[str] = None
    allow_lan_access: Optional[bool] = None
    verbose: Optional[bool] = None
    bench_prompt_tokens: Optional[int] = None
    bench_gen_tokens: Optional[int] = None
    bench_repetitions: Optional[int] = None
    bench_custom_prompt: Optional[str] = None


@app.get("/api/settings")
def get_settings():
    return settings.load_settings()


@app.put("/api/settings")
def put_settings(update: SettingsUpdate):
    data = {k: v for k, v in update.dict().items() if v is not None}
    return settings.update_settings(data)


@app.get("/api/settings/expected-binary-name")
def expected_binary_name():
    return {"name": settings.expected_binary_name(), "platform": platform.system()}


@app.post("/api/settings/validate-binary")
def validate_binary(payload: Dict[str, str]):
    path_str = payload.get("path", "")
    if not path_str:
        return {"valid": False, "message": "No path provided."}
    p = Path(path_str)
    if not p.exists():
        return {"valid": False, "message": "File does not exist at that path."}
    if not p.is_file():
        return {"valid": False, "message": "Path exists but is not a file."}
    return {"valid": True, "message": "Looks good."}


# ---------------------------------------------------------------------------
# llama-server (llama.cpp) - latest build check + one-click install
# ---------------------------------------------------------------------------

class LlamaServerDownloadRequest(BaseModel):
    force: bool = False       # bypass the 1 h release-info cache
    asset: Optional[str] = None  # specific asset name (a GPU variant); None = CPU


# Build number of a --version output, cached per (path, mtime, size) - same
# guard style as models_preset._flag_supported.
_llama_version_cache: Dict[Tuple[str, float, int], Optional[int]] = {}


def _active_binary_build(path_str: str) -> Optional[int]:
    """Run the active binary's --version and parse its build number.
    Returns None when unconfigured, missing, or unparsable."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        stat = p.stat()
        key = (str(p), stat.st_mtime, stat.st_size)
    except OSError:
        return None
    if key in _llama_version_cache:
        return _llama_version_cache[key]

    build: Optional[int] = None
    try:
        out = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=20,
                             env=process_manager.binary_env(p))
        build = llama_release.build_number_from_version_output(
            (out.stdout or "") + (out.stderr or "")
        )
    except Exception:
        build = None
    _llama_version_cache[key] = build
    if len(_llama_version_cache) > 32:  # bound the cache; entries are per-file
        _llama_version_cache.clear()
        _llama_version_cache[key] = build
    return build


@app.get("/api/llama-server/latest")
def llama_server_latest(force: bool = False):
    """
    Latest llama.cpp build release + every downloadable asset for this
    machine (CPU + GPU variants). `asset`/`url`/`size` are the default
    (CPU) choice for callers that don't care about the variant.
    """
    try:
        rel = llama_release.latest_release(force=force)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    try:
        variants = llama_release.list_platform_assets(rel["build"], rel["tag"], rel["assets"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not variants:
        raise HTTPException(status_code=400, detail="No llama.cpp release asset for this platform.")
    default = next((v for v in variants if v["is_cpu"]), variants[0])
    return {**rel, "variants": variants,
            "asset": default["name"], "url": default["url"], "size": default["size"]}


@app.get("/api/app/latest")
def app_latest(force: bool = False):
    """
    Latest release of THIS app on GitHub, for the Settings "new version
    available" badge. `is_newer` compares the latest tag against the
    running version (backend/__init__.py) with the app's own semver rules
    (pre-releases sort older than their release). Offline / no-repo-yet
    comes back as a 502 with a friendly message - same convention as
    /api/llama-server/latest - and the frontend then hides the badge.
    """
    try:
        rel = app_release.latest_version(force=force)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    rel = dict(rel)
    rel["is_newer"] = app_release.is_newer(rel["version"], __version__)
    return rel


@app.get("/api/llama-server/status")
def llama_server_status():
    s = settings.load_settings()
    active_name = s.get("active_llama_server", "")
    path_str = s.get("llama_server_path", "")
    entry = next(
        (e for e in (s.get("llama_servers") or [])
         if isinstance(e, dict) and e.get("name") == active_name),
        None,
    )
    return {
        **llama_server_download.installer.status(),
        "current": {
            "name": entry.get("name", "") if entry else "",
            "path": path_str,
            "build": _active_binary_build(path_str),
        },
    }


@app.post("/api/llama-server/download")
def llama_server_download_start(payload: LlamaServerDownloadRequest = LlamaServerDownloadRequest()):
    try:
        rel = llama_release.latest_release(force=payload.force)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    try:
        asset, url, _size = llama_release.resolve_platform_asset(
            payload.asset, rel["build"], rel["tag"], rel["assets"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return llama_server_download.installer.start(rel["tag"], asset, url)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/llama-server/download/cancel")
def llama_server_download_cancel():
    return llama_server_download.installer.cancel()


class LlamaServerRemoveRequest(BaseModel):
    path: str = ""


@app.post("/api/llama-server/remove")
def llama_server_remove(payload: LlamaServerRemoveRequest):
    """Delete the on-disk files of a version-list entry (see
    llama_server_download.remove_build_files for what exactly gets removed)."""
    path_str = (payload.path or "").strip()
    if not path_str:
        raise HTTPException(status_code=400, detail="No path given.")
    # A server running on this binary holds it open (Windows locks the file)
    # - stop it first.
    st = process_manager.manager.status()
    cmd = st.get("command") or []
    if st.get("state") in ("starting", "running") and cmd:
        try:
            if Path(cmd[0]).resolve() == Path(path_str).expanduser().resolve():
                raise HTTPException(
                    status_code=409,
                    detail="Stop the server that is using this build before removing it.",
                )
        except HTTPException:
            raise
        except OSError:
            pass  # unresolvable - let the deletion attempt surface the real error
    try:
        return llama_server_download.remove_build_files(path_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=409, detail=f"Could not delete the files: {e}")


# ---------------------------------------------------------------------------
# Scanner / library
# ---------------------------------------------------------------------------

@app.get("/api/models")
def get_models(rescan: bool = False):
    roots = settings.get_model_root_folders()
    if not roots:
        return {"scanned_at": None, "roots": [], "models": [], "errors": [], "total_models": 0, "total_files": 0}

    if not rescan:
        cached = scanner.load_cache()
        if cached is not None:
            return cached

    return scanner.scan_roots(roots)


@app.post("/api/models/rescan")
def rescan_models():
    roots = settings.get_model_root_folders()
    if not roots:
        raise HTTPException(status_code=400, detail="No model root folders configured. Add one in Settings first.")
    return scanner.scan_roots(roots)


@app.get("/api/mmproj/files")
def mmproj_files(model_path: str):
    """Candidate multimodal projector files in the given model file's folder,
    for the editor's --mmproj picker. The input still accepts any custom path."""
    return {"files": scanner.find_mmproj_files(model_path)}


@app.get("/api/chat-template/files")
def chat_template_files(model_path: str):
    """Candidate custom chat-template files (.jinja) in the given model
    file's folder, for the editor's --chat-template-file picker. The input
    still accepts any custom path."""
    return {"files": scanner.find_chat_template_files(model_path)}


@app.get("/api/gguf/facts")
def gguf_facts(path: str):
    """GGUF metadata facts for a model file (architecture, context_length,
    block_count, chat_template) via the `gguf` Python package - the profile
    editor uses block_count as the --n-cpu-moe slider's maximum."""
    try:
        return gguf_meta.read_gguf_facts(path)
    except gguf_meta.GgufMetaError as e:
        raise HTTPException(status_code=400, detail=str(e))


class ModelDeleteRequest(BaseModel):
    paths: List[str]


@app.post("/api/models/delete")
def delete_models(payload: ModelDeleteRequest):
    """Delete model files from disk. Every path must live inside one of the
    configured root folders - anything else is refused outright. Empty
    directories left behind are pruned back down to (not including) the
    root, and the scan cache is refreshed so the UI and the "already
    downloaded" flags match what's actually on disk."""
    roots = settings.get_model_root_folders()
    if not roots:
        raise HTTPException(status_code=400, detail="No model root folders configured.")
    resolved_roots = [Path(r).resolve() for r in roots]

    to_delete: List[Path] = []
    for path_str in payload.paths or []:
        p = Path(path_str).resolve()
        if not any(p.is_relative_to(root) for root in resolved_roots):
            raise HTTPException(
                status_code=400,
                detail=f"Refusing to delete {p} - it is outside your model root folders.",
            )
        to_delete.append(p)

    deleted = 0
    for p in to_delete:
        try:
            if p.is_file():
                p.unlink()
                deleted += 1
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"Could not delete {p.name}: {e}")

    # Prune directories left empty by the deletion, stopping at the root.
    for p in to_delete:
        d = p.parent
        while any(d.is_relative_to(root) and d != root for root in resolved_roots):
            try:
                d.rmdir()  # only removes empty directories
            except OSError:
                break
            d = d.parent

    scanner.scan_roots(roots)
    return {"deleted": deleted}


# ---------------------------------------------------------------------------
# Parameter schema
# ---------------------------------------------------------------------------

@app.get("/api/schema")
def get_schema():
    return param_schema.get_schema()


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class ProfileCreate(BaseModel):
    model_id: str
    model_path: str
    name: str
    params: Dict[str, Any] = {}
    custom_flags: str = ""
    notes: str = ""


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    custom_flags: Optional[str] = None
    notes: Optional[str] = None
    model_path: Optional[str] = None


class ProfileDuplicate(BaseModel):
    new_name: Optional[str] = None


class ProfileImport(BaseModel):
    model_id: str
    model_path: str
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Router presets
# ---------------------------------------------------------------------------

class PresetCreate(BaseModel):
    name: str
    profile_ids: List[str] = []
    models_max: int = 4
    autoload: bool = True
    load_on_startup: bool = True
    defaults: str = ""


class PresetUpdate(BaseModel):
    name: Optional[str] = None
    profile_ids: Optional[List[str]] = None
    models_max: Optional[int] = None
    autoload: Optional[bool] = None
    load_on_startup: Optional[bool] = None
    defaults: Optional[str] = None


@app.get("/api/presets")
def list_presets():
    return presets.list_presets()


# NOTE: declared before the /{preset_id} routes so FastAPI doesn't match
# "capability" as a preset id.
@app.get("/api/presets/capability")
def presets_capability():
    binary = settings.resolve_llama_server_path()
    supported = models_preset.router_mode_supported(binary)
    return {"supported": supported, "binary": binary}


@app.post("/api/presets")
def create_preset(payload: PresetCreate):
    return presets.create_preset(
        name=payload.name,
        profile_ids=payload.profile_ids,
        models_max=payload.models_max,
        autoload=payload.autoload,
        load_on_startup=payload.load_on_startup,
        defaults=payload.defaults,
    )


@app.get("/api/presets/{preset_id}")
def get_preset(preset_id: str):
    p = presets.get_preset(preset_id)
    if not p:
        raise HTTPException(status_code=404, detail="Preset not found.")
    return p


@app.put("/api/presets/{preset_id}")
def update_preset(preset_id: str, payload: PresetUpdate):
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    p = presets.update_preset(preset_id, updates)
    if not p:
        raise HTTPException(status_code=404, detail="Preset not found.")
    return p


@app.delete("/api/presets/{preset_id}")
def delete_preset(preset_id: str):
    if not presets.delete_preset(preset_id):
        raise HTTPException(status_code=404, detail="Preset not found.")
    return {"deleted": True}


@app.get("/api/presets/{preset_id}/preview")
def preview_preset(preset_id: str):
    preset = presets.get_preset(preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found.")
    data = models_preset.preview(preset, profiles.list_profiles())
    app_settings = settings.load_settings()
    host = app_settings.get("default_host", "127.0.0.1")
    port = app_settings.get("default_port", 8080)
    data["args"] = data["args"] + ["--host", str(host), "--port", str(port)]
    return data


# ---------------------------------------------------------------------------
# Router dirs (--models-dir)
# ---------------------------------------------------------------------------

class RouterDirCreate(BaseModel):
    name: str
    models_dir: str
    params: Dict[str, Any] = {}
    custom_flags: str = ""
    models_max: int = 4
    autoload: bool = True
    notes: str = ""


class RouterDirUpdate(BaseModel):
    name: Optional[str] = None
    models_dir: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    custom_flags: Optional[str] = None
    models_max: Optional[int] = None
    autoload: Optional[bool] = None
    notes: Optional[str] = None


@app.get("/api/router-dirs")
def list_router_dirs():
    return router_dirs.list_router_dirs()


@app.get("/api/router-dirs/capability")
def router_dirs_capability():
    binary = settings.resolve_llama_server_path()
    supported = models_preset.router_dir_supported(binary)
    return {"supported": supported, "binary": binary}


@app.post("/api/router-dirs")
def create_router_dir(payload: RouterDirCreate):
    return router_dirs.create_router_dir(
        name=payload.name,
        models_dir=payload.models_dir,
        params=payload.params,
        custom_flags=payload.custom_flags,
        models_max=payload.models_max,
        autoload=payload.autoload,
        notes=payload.notes,
    )


@app.get("/api/router-dirs/{router_dir_id}")
def get_router_dir(router_dir_id: str):
    p = router_dirs.get_router_dir(router_dir_id)
    if not p:
        raise HTTPException(status_code=404, detail="Router dir not found.")
    return p


@app.put("/api/router-dirs/{router_dir_id}")
def update_router_dir(router_dir_id: str, payload: RouterDirUpdate):
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    p = router_dirs.update_router_dir(router_dir_id, updates)
    if not p:
        raise HTTPException(status_code=404, detail="Router dir not found.")
    return p


@app.delete("/api/router-dirs/{router_dir_id}")
def delete_router_dir(router_dir_id: str):
    if not router_dirs.delete_router_dir(router_dir_id):
        raise HTTPException(status_code=404, detail="Router dir not found.")
    return {"deleted": True}


@app.get("/api/router-dirs/{router_dir_id}/command-preview")
def router_dir_command_preview(router_dir_id: str):
    p = router_dirs.get_router_dir(router_dir_id)
    if not p:
        raise HTTPException(status_code=404, detail="Router dir not found.")
    app_settings = settings.load_settings()
    host = app_settings.get("default_host", "127.0.0.1")
    port = app_settings.get("default_port", 8080)
    binary_name = settings.expected_binary_name()
    cmd = command_builder.preview_router_dir_command(
        binary_name, p["models_dir"], p.get("params", {}), p.get("custom_flags", ""),
        p.get("models_max", 4), p.get("autoload", True), host, port,
    )
    return {"command": cmd}


@app.get("/api/profiles")
def list_profiles(model_id: Optional[str] = None):
    return profiles.list_profiles(model_id)


def _store_model_folder_paths(params: Optional[Dict[str, Any]], model_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Persist model-folder file params (--mmproj, --chat-template-file) that
    are just a file name as the full path, so stored profiles match what
    actually gets launched (the command builder resolves them at launch too
    - this keeps the stored data self-contained).
    """
    if not params or not model_path:
        return params
    resolvers = {
        "mmproj": command_builder.resolve_mmproj_path,
        "chat_template_file": command_builder.resolve_chat_template_path,
    }
    changed = False
    for key, resolve in resolvers.items():
        v = params.get(key)
        if not isinstance(v, str):
            continue
        resolved = resolve(model_path, v)
        if resolved != v:
            if not changed:
                params = dict(params)
                changed = True
            params[key] = resolved
    return params


@app.post("/api/profiles")
def create_profile(payload: ProfileCreate):
    return profiles.create_profile(
        model_id=payload.model_id,
        model_path=payload.model_path,
        name=payload.name,
        params=_store_model_folder_paths(payload.params, payload.model_path),
        custom_flags=payload.custom_flags,
        notes=payload.notes,
    )


class ProfileImportAll(BaseModel):
    profiles: List[ProfileImport]


# NOTE: declared before /api/profiles/{profile_id} so FastAPI doesn't match
# "export-all" as a profile id (same reason presets/capability is placed early).
@app.get("/api/profiles/export-all")
def export_all_profiles():
    """Every profile in one versioned JSON file (backup / migration / sharing).

    `profiles` holds the same canonical export shape the per-profile Export
    button hands out (export_profile), so single-profile files, benchmark
    snapshots and export-all files are mutually compatible.
    """
    return {
        "app": "Llama Profile Manager",
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles.export_all_profiles(),
    }


@app.post("/api/profiles/import-all")
async def import_profiles_all(request: Request):
    """Import a whole collection: a list of {model_id, model_path, data}
    items (the frontend builds these from an export-all file or a bare
    profile array, re-rooting each entry at the picked local model).
    Structure is validated up front (400 with a clear message); per-item
    failures are reported in-band in the result, never aborting the batch."""
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid import file: the body must be a JSON object with a 'profiles' array.",
        )
    if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), list):
        raise HTTPException(
            status_code=400,
            detail="Invalid import file: expected a JSON object with a 'profiles' array of "
                   "{model_id, model_path, data} entries.",
        )
    try:
        payload = ProfileImportAll.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0] if e.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", ()) if str(x) != "body")
        raise HTTPException(
            status_code=400,
            detail="Invalid import file: "
                   f"{loc + ': ' if loc else ''}{first.get('msg', 'malformed entry')}. "
                   "Each entry needs model_id, model_path and a data object.",
        )
    # Same model-folder re-rooting as the single import: a bare --mmproj / 
    # --chat-template-file name in the file is stored as a full path relative
    # to the picked local model (full paths are left untouched - the command
    # builder resolves them at launch).
    items = []
    for p in payload.profiles:
        data = dict(p.data)
        data["params"] = _store_model_folder_paths(data.get("params"), p.model_path)
        items.append({"model_id": p.model_id, "model_path": p.model_path, "data": data})
    return profiles.import_profiles_batch(items)


@app.get("/api/profiles/{profile_id}")
def get_profile(profile_id: str):
    p = profiles.get_profile(profile_id)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return p


@app.put("/api/profiles/{profile_id}")
def update_profile(profile_id: str, payload: ProfileUpdate):
    updates = {k: v for k, v in payload.dict().items() if v is not None}
    if "params" in updates:
        existing = profiles.get_profile(profile_id)
        model_path = updates.get("model_path") or (existing or {}).get("model_path")
        updates["params"] = _store_model_folder_paths(updates["params"], model_path)
    p = profiles.update_profile(profile_id, updates)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return p


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str):
    ok = profiles.delete_profile(profile_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return {"deleted": True}


@app.post("/api/profiles/{profile_id}/duplicate")
def duplicate_profile(profile_id: str, payload: ProfileDuplicate):
    p = profiles.duplicate_profile(profile_id, payload.new_name)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return p


@app.get("/api/profiles/{profile_id}/export")
def export_profile(profile_id: str):
    data = profiles.export_profile(profile_id)
    if not data:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return data


@app.post("/api/profiles/import")
def import_profile(payload: ProfileImport):
    data = dict(payload.data)
    data["params"] = _store_model_folder_paths(data.get("params"), payload.model_path)
    return profiles.import_profile(payload.model_id, payload.model_path, data)


@app.get("/api/profiles/{profile_id}/command-preview")
def command_preview(profile_id: str):
    p = profiles.get_profile(profile_id)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found.")
    binary_name = settings.expected_binary_name()
    cmd = command_builder.preview_command(binary_name, p["model_path"], p["params"], p.get("custom_flags", ""))
    return {"command": cmd}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class BenchmarkRunRequest(BaseModel):
    # Exactly one of these: benchmark the live profile, or re-run a saved
    # record's params snapshot (BenchPlan §4.3).
    profile_id: Optional[str] = None
    benchmark_id: Optional[str] = None
    prompt_tokens: int = 512
    gen_tokens: int = 256
    repetitions: int = 5
    custom_prompt: Optional[str] = None


class BenchmarkImportRequest(BaseModel):
    mode: str = "new"          # "new" | "overwrite"
    name: Optional[str] = None
    profile_id: Optional[str] = None


@app.get("/api/benchmarks")
def list_benchmarks():
    return bench_store.list_benchmarks()


class BenchmarkCompareRequest(BaseModel):
    a: str
    b: str


@app.post("/api/benchmarks/compare")
def compare_benchmarks(payload: BenchmarkCompareRequest):
    """Were these two runs done with the same parameters?

    Each side returns the *benchmarkable* params - the normalized
    model_path/params/custom_flags subset that lands on the llama-server
    command line - computed with the same single source of truth
    (benchmarkable_params/params_hash) the staleness badge uses, so the
    verdict here can never disagree with the badge. `same` is recomputed
    from the two snapshots (identical semantics to the badge).
    """
    recs = {}
    for side in ("a", "b"):
        rec = bench_store.get_benchmark(getattr(payload, side))
        if not rec:
            raise HTTPException(
                status_code=404,
                detail=f"Benchmark {getattr(payload, side)!r} not found.",
            )
        recs[side] = rec
    if payload.a == payload.b:
        raise HTTPException(
            status_code=400, detail="Choose two different records to compare.")
    sides = {}
    for side in ("a", "b"):
        rec = recs[side]
        snap = rec.get("profile_params_snapshot") or {}
        sides[side] = {
            "id": rec["id"],
            "profile_name": rec.get("profile_name") or "",
            "model_path": rec.get("model_path") or "",
            "server_version": rec.get("server_version") or "",
            "status": rec.get("status") or "",
            "timestamp": rec.get("timestamp"),
            "prefill_tps": rec.get("prefill_tps"),
            "generation_tps": rec.get("generation_tps"),
            "params_hash": rec.get("params_hash") or "",
            "benchmarkable": bench_store.benchmarkable_params(snap),
        }
    sides["same"] = (bench_store.params_hash(recs["a"].get("profile_params_snapshot") or {})
                     == bench_store.params_hash(recs["b"].get("profile_params_snapshot") or {}))
    return sides


@app.get("/api/benchmarks/{benchmark_id}")
def get_benchmark(benchmark_id: str):
    rec = bench_store.get_benchmark(benchmark_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Benchmark not found.")
    return rec


@app.post("/api/benchmarks/run")
async def run_benchmark(payload: BenchmarkRunRequest):
    provided = sum(bool(x) for x in (payload.profile_id, payload.benchmark_id))
    if provided != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of profile_id or benchmark_id.",
        )
    options = {
        "prompt_tokens": min(max(int(payload.prompt_tokens or 512), 16), 32768),
        "gen_tokens": min(max(int(payload.gen_tokens or 256), 8), 8192),
        "repetitions": min(max(int(payload.repetitions or 5), 1), 20),
        "custom_prompt": payload.custom_prompt,
    }

    if payload.profile_id:
        p = profiles.get_profile(payload.profile_id)
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found.")
        source = {
            "profile_id": p["id"],
            "name": p["name"],
            "model_path": p["model_path"],
            "params": p.get("params") or {},
            "custom_flags": p.get("custom_flags") or "",
        }
        # The record's snapshot uses the app's canonical profile export
        # (BenchPlan §5.1), so imports/re-runs are always compatible.
        snapshot = profiles.export_profile(p["id"])
        re_ran_from = None
    else:
        src = bench_store.get_benchmark(payload.benchmark_id)
        if not src:
            raise HTTPException(status_code=404, detail="Benchmark not found.")
        snap = dict(src.get("profile_params_snapshot") or {})
        model_path = snap.get("model_path") or src.get("model_path") or ""
        if not model_path:
            raise HTTPException(
                status_code=400,
                detail="This benchmark has no model path to re-run.",
            )
        source = {
            "profile_id": src.get("profile_id"),   # may be dangling (profile deleted)
            "name": src.get("profile_name") or "(unknown profile)",
            "model_path": model_path,
            "params": snap.get("params") or {},
            "custom_flags": snap.get("custom_flags") or "",
        }
        snapshot = _json.loads(_json.dumps(snap))  # deep copy - never alias the stored record
        re_ran_from = src["id"]

    source["re_ran_from"] = re_ran_from
    try:
        # async def → we can hand the runner the event loop the log
        # subscribers live on (process_manager needs it for /ws/logs).
        record = benchmark_runner.runner.start(
            source, snapshot, options, asyncio.get_running_loop())
    except benchmark_runner.BenchmarkError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return record


@app.post("/api/benchmarks/{benchmark_id}/cancel")
def cancel_benchmark(benchmark_id: str):
    if benchmark_runner.runner.active_id != benchmark_id:
        raise HTTPException(status_code=400, detail="That benchmark isn't the one currently running.")
    benchmark_runner.runner.cancel()
    return {"cancelled": True}


@app.delete("/api/benchmarks/{benchmark_id}")
def delete_benchmark(benchmark_id: str):
    if benchmark_runner.runner.active_id == benchmark_id:
        raise HTTPException(status_code=409, detail="Stop the running benchmark before deleting its record.")
    if not bench_store.delete_benchmark(benchmark_id):
        raise HTTPException(status_code=404, detail="Benchmark not found.")
    # Drop any profile badges pointing at the now-missing record, so the UI
    # never renders a badge that can't open its row.
    for p in profiles.list_profiles():
        badge = p.get("benchmark_badge")
        if isinstance(badge, dict) and badge.get("benchmark_id") == benchmark_id:
            profiles.set_benchmark_badge(p["id"], None)
    return {"deleted": True}


@app.get("/api/benchmarks/{benchmark_id}/snapshot")
def benchmark_snapshot(benchmark_id: str):
    """The record's params snapshot in the canonical profile-export shape -
    the same thing the profile Export button hands out (BenchPlan §5.3)."""
    rec = bench_store.get_benchmark(benchmark_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Benchmark not found.")
    return rec.get("profile_params_snapshot") or {}


@app.post("/api/benchmarks/{benchmark_id}/import-as-profile")
def import_benchmark_as_profile(benchmark_id: str, payload: BenchmarkImportRequest):
    try:
        p = benchmark_runner.import_benchmark_as_profile(
            benchmark_id, mode=payload.mode, name=payload.name, profile_id=payload.profile_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not p:
        raise HTTPException(status_code=404, detail="Import failed - record not found.")
    return p


# ---------------------------------------------------------------------------
# Server control
# ---------------------------------------------------------------------------

class ServerStartRequest(BaseModel):
    profile_id: Optional[str] = None
    preset_id: Optional[str] = None
    router_dir_id: Optional[str] = None
    host_override: Optional[str] = None
    port_override: Optional[int] = None


@app.get("/api/server/status")
def server_status():
    return process_manager.manager.status()


@app.post("/api/server/start")
async def server_start(payload: ServerStartRequest):
    provided = sum(bool(x) for x in (payload.profile_id, payload.preset_id, payload.router_dir_id))
    if provided != 1:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly one of profile_id, preset_id, or router_dir_id.",
        )

    app_settings = settings.load_settings()
    binary_path = settings.resolve_llama_server_path()

    if payload.router_dir_id:
        return await _start_router_dir(payload, binary_path, app_settings)
    if payload.preset_id:
        return await _start_router(payload, binary_path, app_settings)
    return await _start_single(payload, binary_path, app_settings)


async def _start_single(payload: ServerStartRequest, binary_path: str,
                        app_settings: Dict[str, Any]) -> Dict[str, Any]:
    p = profiles.get_profile(payload.profile_id)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found.")

    # Precedence: explicit request override > the profile's own --host/--port
    # params > the app-wide default. Previously this always fell through to
    # the app-wide default and force-injected it into the command, silently
    # overwriting whatever host/port the profile itself had set.
    host = payload.host_override or p["params"].get("host") or app_settings.get("default_host", "127.0.0.1")
    port = payload.port_override or p["params"].get("port") or app_settings.get("default_port", 8080)

    args = command_builder.build_args(
        p["model_path"], p["params"], p.get("custom_flags", ""),
        host_override=host, port_override=port,
    )
    if app_settings.get("verbose"):
        args.append("-v")  # log everything, for debugging

    loop = asyncio.get_running_loop()
    try:
        # profile_id is tracked on the process so the UI can show which
        # profile owns the running server (and turn its Start into a Stop).
        status = process_manager.manager.start(
            binary_path, args, host, port, loop, mode="single",
            profile_id=payload.profile_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    profiles.mark_used(payload.profile_id)
    return status


async def _start_router(payload: ServerStartRequest, binary_path: str,
                        app_settings: Dict[str, Any]) -> Dict[str, Any]:
    preset = presets.get_preset(payload.preset_id)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found.")

    supported = models_preset.router_mode_supported(binary_path)
    if supported is False:
        raise HTTPException(
            status_code=400,
            detail="Your llama-server build doesn't support router mode (--models-preset). "
                   "Update llama-server to a recent build first.",
        )

    try:
        built = models_preset.build_launch_args(preset, profiles.list_profiles())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # In router mode the router owns host/port (per-model values are ignored),
    # so only the request override or the app-wide default applies.
    host = payload.host_override or app_settings.get("default_host", "127.0.0.1")
    port = payload.port_override or app_settings.get("default_port", 8080)

    args = built["args"] + ["--host", str(host), "--port", str(port)]
    if app_settings.get("verbose"):
        args.append("-v")  # log everything, for debugging

    loop = asyncio.get_running_loop()
    try:
        status = process_manager.manager.start(binary_path, args, host, port, loop, mode="router", preset_id=payload.preset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    for m in built["models"]:
        prof = profiles.get_profile(m["profile_id"])
        if prof:
            profiles.mark_used(prof["id"])
    return status


async def _start_router_dir(payload: ServerStartRequest, binary_path: str,
                            app_settings: Dict[str, Any]) -> Dict[str, Any]:
    rd = router_dirs.get_router_dir(payload.router_dir_id)
    if not rd:
        raise HTTPException(status_code=404, detail="Router dir not found.")
    if not (rd.get("models_dir") or "").strip():
        raise HTTPException(status_code=400, detail="This router dir has no models directory set.")

    supported = models_preset.router_dir_supported(binary_path)
    if supported is False:
        raise HTTPException(
            status_code=400,
            detail="Your llama-server build doesn't support --models-dir. "
                   "Update llama-server to a recent build first.",
        )

    host = payload.host_override or app_settings.get("default_host", "127.0.0.1")
    port = payload.port_override or app_settings.get("default_port", 8080)

    args = command_builder.build_router_dir_args(
        rd["models_dir"], rd.get("params", {}), rd.get("custom_flags", ""),
        rd.get("models_max", 4), rd.get("autoload", True),
        host_override=host, port_override=port,
    )
    if app_settings.get("verbose"):
        args.append("-v")  # log everything, for debugging

    loop = asyncio.get_running_loop()
    try:
        status = process_manager.manager.start(
            binary_path, args, host, port, loop, mode="router",
            router_dir_id=payload.router_dir_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return status


# ---------------------------------------------------------------------------
# Router model management (proxied to the running router)
# ---------------------------------------------------------------------------

class RouterModelRequest(BaseModel):
    model: str


def _router_base_url() -> str:
    st = process_manager.manager.status()
    if st.get("mode") != "router" or st["state"] not in ("running", "starting"):
        raise HTTPException(status_code=400, detail="No router server is running.")
    host = "127.0.0.1" if st["host"] == "0.0.0.0" else st["host"]
    return f"http://{host}:{st['port']}"


def _router_http(url: str, method: str = "GET", body: Optional[Dict[str, Any]] = None) -> Any:
    data = _json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            payload = _json.loads(e.read().decode("utf-8"))
            message = payload.get("error", {}).get("message") or _json.dumps(payload)
        except Exception:
            message = str(e)
        raise HTTPException(status_code=502, detail=f"Router request failed: {message}")
    except OSError as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the router: {e}")


@app.get("/api/server/router/models")
def router_models(reload: bool = False):
    url = _router_base_url() + "/models" + ("?reload=1" if reload else "")
    return _router_http(url)


@app.post("/api/server/router/models/load")
def router_models_load(payload: RouterModelRequest):
    return _router_http(_router_base_url() + "/models/load", "POST", {"model": payload.model})


@app.post("/api/server/router/models/unload")
def router_models_unload(payload: RouterModelRequest):
    return _router_http(_router_base_url() + "/models/unload", "POST", {"model": payload.model})


@app.post("/api/server/stop")
def server_stop():
    return process_manager.manager.stop()


@app.websocket("/ws/logs")
async def logs_ws(websocket: WebSocket):
    await websocket.accept()
    queue = process_manager.manager.subscribe()
    try:
        while True:
            line = await queue.get()
            await websocket.send_text(line)
    except WebSocketDisconnect:
        pass
    finally:
        process_manager.manager.unsubscribe(queue)


# ---------------------------------------------------------------------------
# System integration (open folder / open external link)
# ---------------------------------------------------------------------------

class OpenFolderRequest(BaseModel):
    path: str  # a file OR folder path; if a file, its parent folder is opened


@app.post("/api/system/open-folder")
def open_folder(payload: OpenFolderRequest):
    target = Path(payload.path)
    folder = target if target.is_dir() else target.parent
    if not folder.exists():
        raise HTTPException(status_code=400, detail="That folder doesn't exist.")

    try:
        system = platform.system()
        if system == "Windows":
            import os
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Couldn't open the folder: {e}")
    return {"opened": True}


class OpenUrlRequest(BaseModel):
    url: str


@app.post("/api/system/open-url")
def open_url(payload: OpenUrlRequest):
    # Restricted to the two external sites this app links out to: model repos
    # (huggingface.co) and this app's own releases (github.com). The endpoint
    # exists so links open in the user's real default browser instead of
    # inside the pywebview window - not as a general-purpose URL opener.
    if not (payload.url.startswith("https://huggingface.co/")
            or payload.url.startswith("https://github.com/")):
        raise HTTPException(status_code=400, detail="Only huggingface.co and github.com links can be opened this way.")
    try:
        webbrowser.open(payload.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Couldn't open the link: {e}")
    return {"opened": True}


def _detect_lan_ip() -> Optional[str]:
    """Best-effort local network IP, without requiring actual internet access
    (a UDP "connect" just consults the routing table, no packets need to
    actually be delivered)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


@app.get("/api/system/info")
def system_info():
    """
    App name + version. The version in backend/__init__.py is the single
    source of truth (mirrored by the git tag, e.g. 0.1.0 <-> v0.1.0);
    Settings shows it and uses it for the "new version available" check.
    """
    return {"app": "Llama Profile Manager", "version": __version__}


@app.get("/api/system/network-info")
def network_info():
    app_settings = settings.load_settings()
    allow_lan = bool(app_settings.get("allow_lan_access", False))
    bind_host = runtime_info.get("bind_host")
    port = runtime_info.get("port")
    lan_ip = _detect_lan_ip() if allow_lan else None
    return {
        "allow_lan_access": allow_lan,
        "bind_host": bind_host,
        "port": port,
        "currently_lan_reachable": bool(allow_lan and bind_host == "0.0.0.0"),
        "lan_url": f"http://{lan_ip}:{port}/" if (lan_ip and port and allow_lan and bind_host == "0.0.0.0") else None,
    }


# ---------------------------------------------------------------------------
# Hugging Face lookup
# ---------------------------------------------------------------------------

class HFResolveRequest(BaseModel):
    input: str


@app.post("/api/hf/resolve")
def hf_resolve(payload: HFResolveRequest):
    try:
        repo_id = hf_client.parse_repo_id(payload.input)
        result = hf_client.list_gguf_groups(repo_id)
        hf_client.mark_downloaded(repo_id, result["groups"])
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Same shape as a real HF namespace (word chars, dots, dashes); empty and
# the app's own "(ungrouped)" sentinel are rejected explicitly up front.
_HF_NAMESPACE_RE = re.compile(r"^[\w.\-]+$")


@app.get("/api/hf/avatar")
def hf_avatar_endpoint(namespace: str = ""):
    """
    The org/user avatar shown next to model rows. The namespace is the `org`
    part of an `org/repo` repo id - there is no per-repo icon on the Hub.
    Always 200 for a valid namespace: unknown namespaces and network
    failures both come back with null fields, so the frontend falls back to
    its initial-letter badge instead of surfacing an error.
    """
    ns = namespace.strip()
    if not ns or ns == "(ungrouped)" or not _HF_NAMESPACE_RE.match(ns):
        raise HTTPException(status_code=400, detail="Invalid Hugging Face namespace.")
    result = hf_avatar.fetch_avatar(ns)
    return {
        "namespace": ns,
        "avatarUrl": result["url"] if result else None,
        "fullName": result["name"] if result else None,
    }


class HFSearchRequest(BaseModel):
    query: str = ""
    limit: int = hf_search.DEFAULT_LIMIT
    sort: str = hf_search.DEFAULT_SORT


@app.post("/api/hf/search")
def hf_search_endpoint(payload: HFSearchRequest):
    sort = hf_search.normalize_sort(payload.sort)
    try:
        results = hf_search.search_models(payload.query, payload.limit, sort)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "query": payload.query,
        "limit": hf_search.clamp_limit(payload.limit),
        "sort": sort,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

class DownloadFile(BaseModel):
    path: str
    filename: str
    size_bytes: int = 0


class DownloadStartRequest(BaseModel):
    repo_id: str
    group_name: str = ""   # quantization group label; part of the job key
    files: List[DownloadFile]
    target_root: str


class DownloadCancelRequest(BaseModel):
    key: Optional[str] = None   # job key (repo_id::group_name); None = cancel all


@app.get("/api/downloads/status")
def downloads_status():
    return download_manager.manager.frame()


@app.post("/api/downloads/start")
async def downloads_start(payload: DownloadStartRequest):
    roots = settings.get_model_root_folders()
    if payload.target_root not in roots:
        raise HTTPException(status_code=400, detail="That target folder isn't one of your configured model root folders.")
    if "/" not in payload.repo_id:
        raise HTTPException(status_code=400, detail="Invalid repo id.")

    org, repo = payload.repo_id.split("/", 1)
    dest_dir = str(Path(payload.target_root) / org / repo)
    files = [f.dict() for f in payload.files]

    loop = asyncio.get_running_loop()
    try:
        key = download_manager.manager.start(payload.repo_id, payload.group_name, files, dest_dir, loop)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"key": key, **download_manager.manager.status_all().get(key, {})}


@app.post("/api/downloads/cancel")
def downloads_cancel(payload: Optional[DownloadCancelRequest] = None):
    return download_manager.manager.cancel(payload.key if payload else None)


@app.websocket("/ws/downloads")
async def downloads_ws(websocket: WebSocket):
    await websocket.accept()
    queue = download_manager.manager.subscribe()
    try:
        while True:
            message = await queue.get()
            await websocket.send_text(message)
    except WebSocketDisconnect:
        pass
    finally:
        download_manager.manager.unsubscribe(queue)


# ---------------------------------------------------------------------------
# Frontend static files (served last so /api and /ws take precedence)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
