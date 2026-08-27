"""
profiles.py

CRUD storage for launch profiles. Each model (identified by a stable
model_id derived from its path/group) can have multiple named profiles.

Profile JSON shape:
{
    "id": "uuid4-string",
    "model_id": "string",          # matches the group id from scanner.py
    "model_path": "string",        # resolved path to launch (first file of the group)
    "name": "string",              # e.g. "Default", "Long context"
    "params": { "ctx_size": 8192, "n_gpu_layers": "all", ... },  # keyed by param_schema keys
    "custom_flags": "string",      # raw extra CLI flags (advanced mode), space separated
    "notes": "string",
    "created_at": float,
    "updated_at": float,
    "last_used_at": float | None,
    "benchmark_badge": {           # optional - last benchmark result for this profile
        "prefill_tps": float,
        "generation_tps": float,
        "benchmark_id": "uuid",    # row in data/benchmarks.json
        "params_hash": "sha256",   # hash of the params at benchmark time
        "state": "fresh" | "stale" | "running" | "failed" | "none",
    } | absent,
}

All profiles live in a single JSON file (data/profiles.json) as a list,
which is simple and plenty fast for the number of profiles a single user
will realistically create.
"""

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import benchmarks as bench_store
from . import gguf_utils

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILES_FILE = DATA_DIR / "profiles.json"

# Guards the *entire* read-modify-write cycle of every mutation below, not
# just the final file write. Endpoints run on both the event loop (async def)
# and the threadpool (def), so two writers can genuinely interleave; holding
# the lock only around the write let a stale full-file rewrite clobber a
# concurrent save (e.g. mark_used from a server start racing a profile PUT).
# RLock because public mutation helpers may call each other (duplicate ->
# create, import -> create).
_lock = threading.RLock()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_all() -> List[Dict[str, Any]]:
    ensure_data_dir()
    if not PROFILES_FILE.exists():
        return []
    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_all(profiles: List[Dict[str, Any]]) -> None:
    ensure_data_dir()
    with _lock:
        tmp_path = PROFILES_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        tmp_path.replace(PROFILES_FILE)


def list_profiles(model_id: Optional[str] = None) -> List[Dict[str, Any]]:
    profiles = _read_all()
    if model_id:
        profiles = [p for p in profiles if p.get("model_id") == model_id]
    # most-recently-used first, then most-recently-updated
    profiles.sort(key=lambda p: (p.get("last_used_at") or 0, p.get("updated_at") or 0), reverse=True)
    return profiles


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    for p in _read_all():
        if p.get("id") == profile_id:
            return p
    return None


def _unique_name(model_id: str, name: str, profiles: List[Dict[str, Any]],
                 exclude_id: Optional[str] = None) -> str:
    """Make a name unique among the profiles of *this model*.

    Names are not identifiers (ids are UUIDs), so the same name may freely
    be used on different models. Within one model, duplicates are confusing
    (identical list rows, ambiguous export filenames), so on collision we
    append " (2)", " (3)", … instead of failing the save.
    """
    base = (name or "").strip() or "Default"
    taken = {
        (p.get("name") or "").strip().lower()
        for p in profiles
        if p.get("model_id") == model_id and p.get("id") != exclude_id
    }
    if base.lower() not in taken:
        return base
    n = 2
    while f"{base} ({n})".lower() in taken:
        n += 1
    return f"{base} ({n})"


def default_name_for_model(model_path: str) -> str:
    """Default profile name: the model file's name without the .gguf extension.

    Mirrors the scanner's grouping: for a multipart file
    (…-00001-of-00003.gguf) the part suffix is stripped, so the default name
    matches the model exactly as it is displayed in the library. Falls back to
    "Default" only if nothing usable can be derived from the path.
    """
    stem = Path(model_path or "").name
    if stem.lower().endswith(".gguf"):   # also covers a bare ".gguf"
        stem = stem[:-5]
    base_stem, _part = gguf_utils.split_multipart(stem)
    name = (base_stem if base_stem is not None else stem)
    return (name or "").strip() or "Default"


def create_profile(model_id: str, model_path: str, name: str,
                    params: Dict[str, Any], custom_flags: str = "", notes: str = "") -> Dict[str, Any]:
    now = time.time()
    with _lock:
        profiles = _read_all()
        # A blank name defaults to the model's filename (minus the .gguf
        # extension); the caller (editor) pre-fills the same value.
        effective_name = (name or "").strip() or default_name_for_model(model_path)
        profile = {
            "id": str(uuid.uuid4()),
            "model_id": model_id,
            "model_path": model_path,
            "name": _unique_name(model_id, effective_name, profiles),
            "params": params or {},
            "custom_flags": custom_flags or "",
            "notes": notes or "",
            "created_at": now,
            "updated_at": now,
            "last_used_at": None,
        }
        profiles.append(profile)
        _write_all(profiles)
    return profile


def update_profile(profile_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with _lock:
        profiles = _read_all()
        for p in profiles:
            if p.get("id") == profile_id:
                allowed = {"name", "params", "custom_flags", "notes", "model_path"}
                for k, v in (updates or {}).items():
                    if k in allowed:
                        p[k] = v
                if "name" in (updates or {}):
                    p["name"] = _unique_name(p.get("model_id"), p.get("name"), profiles,
                                             exclude_id=profile_id)
                p["updated_at"] = time.time()
                _refresh_badge_state(p)
                _write_all(profiles)
                return p
    return None


def _refresh_badge_state(p: Dict[str, Any]) -> None:
    """Re-check the benchmark badge's freshness against the profile's current
    parameters (BenchPlan §2). Called on every save:
      * hash matches  → state stays/turns "fresh" (green)
      * hash differs  → state flips to "stale" (amber); the old TPS numbers
        are kept, only the color/tooltip changes
    Name/notes edits cannot affect this because the hash only covers the
    parameter fields (see benchmarks.benchmarkable_params). A badge that is
    mid-run ("running") is left alone - the runner owns that state and
    re-checks the hash when the run finishes.
    """
    badge = p.get("benchmark_badge")
    if not isinstance(badge, dict) or badge.get("state") not in ("fresh", "stale"):
        return
    current = bench_store.params_hash(p)
    badge["state"] = "fresh" if current == badge.get("params_hash") else "stale"


def delete_profile(profile_id: str) -> bool:
    with _lock:
        profiles = _read_all()
        new_profiles = [p for p in profiles if p.get("id") != profile_id]
        if len(new_profiles) == len(profiles):
            return False
        _write_all(new_profiles)
    return True


def duplicate_profile(profile_id: str, new_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    original = get_profile(profile_id)
    if not original:
        return None
    created = create_profile(
        model_id=original["model_id"],
        model_path=original["model_path"],
        name=new_name or f"{original['name']} (copy)",
        params=dict(original.get("params", {})),
        custom_flags=original.get("custom_flags", ""),
        notes=original.get("notes", ""),
    )
    badge = original.get("benchmark_badge")
    if isinstance(badge, dict):
        # The copy has byte-identical params, so the badge's freshness
        # relation to the original benchmark carries over unchanged.
        set_benchmark_badge(created["id"], dict(badge))
    return created


def set_benchmark_badge(profile_id: str, badge: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Set (or clear, with None) a profile's benchmark_badge. Used by the
    benchmark runner at run start/finish and by the import-as-profile flow."""
    with _lock:
        profiles = _read_all()
        for p in profiles:
            if p.get("id") == profile_id:
                p["benchmark_badge"] = badge
                _write_all(profiles)
                return p
    return None


def replace_profile(profile_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Overwrite a profile's model + parameters from an external source
    (the benchmark import flow). Unlike update_profile, model_id may change
    too. "name" in data is applied when non-blank (existing name is kept
    otherwise)."""
    with _lock:
        profiles = _read_all()
        for p in profiles:
            if p.get("id") == profile_id:
                p["model_id"] = data.get("model_id", p.get("model_id"))
                p["model_path"] = data.get("model_path", p.get("model_path"))
                if (data.get("name") or "").strip():
                    p["name"] = _unique_name(p.get("model_id"), data["name"].strip(),
                                             profiles, exclude_id=profile_id)
                p["params"] = data.get("params") or {}
                p["custom_flags"] = data.get("custom_flags", "") or ""
                p["notes"] = data.get("notes", "") or ""
                p["updated_at"] = time.time()
                _write_all(profiles)
                return p
    return None


def mark_used(profile_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        profiles = _read_all()
        for p in profiles:
            if p.get("id") == profile_id:
                p["last_used_at"] = time.time()
                _write_all(profiles)
                return p
    return None


def _canonical_export(p: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the internal fields from a stored profile, leaving the app's ONE
    canonical profile serialization: the shape the Export button, the
    benchmark snapshots (BenchPlan §5) and export-all all share, so any of
    them can be imported back into a profile."""
    exported = dict(p)
    exported.pop("id", None)
    exported.pop("last_used_at", None)
    return exported


def export_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    """Return a copy suitable for writing to a standalone JSON file (no internal id).

    This is the app's ONE canonical profile serialization: the benchmark
    feature stores snapshots with this exact shape (BenchPlan §5), so
    importing a snapshot back into a profile is always compatible.
    """
    p = get_profile(profile_id)
    if not p:
        return None
    return _canonical_export(p)


def export_all_profiles() -> List[Dict[str, Any]]:
    """Canonical export of every profile (see export_profile for the shape).

    The single read happens under the lock so the batch is a consistent
    snapshot, unlike N successive export_profile() calls.
    """
    with _lock:
        return [_canonical_export(p) for p in _read_all()]


def import_profile(model_id: str, model_path: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return create_profile(
        model_id=model_id,
        model_path=model_path,
        name=data.get("name", "Imported profile"),
        params=data.get("params", {}),
        custom_flags=data.get("custom_flags", ""),
        notes=data.get("notes", ""),
    )


def import_profiles_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Import many profiles at once (the Import-all flow).

    Each item is {model_id, model_path, data} - the same contract as the
    single-profile import. The whole run is held under the same RLock the
    single mutations use, and each item is wrapped in try/except so one bad
    item never aborts the batch: failures are reported in `errors` (with the
    item's index and name when available) and the rest still import. Name
    collisions are auto-suffixed per model by create_profile/_unique_name.
    """
    imported: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    with _lock:
        for i, item in enumerate(items or []):
            data = item.get("data") if isinstance(item, dict) else None
            label = f"item {i}"
            if isinstance(data, dict) and (data.get("name") or "").strip():
                label += f" ({data['name'].strip()})"
            try:
                imported.append(import_profile(item["model_id"], item["model_path"], item["data"]))
            except Exception as e:  # noqa: BLE001 - one bad item must not sink the batch
                errors.append({"index": i, "name": label, "error": str(e)})
    return {"imported": imported, "errors": errors}
