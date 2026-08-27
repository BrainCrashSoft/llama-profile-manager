"""
hf_avatar.py

Resolves the Hugging Face avatar (org/user icon) for a namespace - the
`org` part of an `org/repo` repo id. This is exactly what huggingface.co
itself shows next to a repo: every repo under one org/user shares that
namespace's avatar, and there is no per-repo icon on the Hub.

The only endpoints that work (verified against the live Hub):
    GET /api/organizations/{ns}/overview  -> {"avatarUrl": "https://cdn-avatars.huggingface.co/v1/...", "fullname": ...}
    GET /api/users/{ns}/overview          -> same shape
A namespace is either an org or a user, so we try org first, then user on
404. (The naive https://huggingface.co/{ns}/avatar.png returns 401, and
/api/orgs/{ns} / /api/users/{ns} return 404.)

Results are cached in data/hf_avatars.json (same JSON-in-data/ convention
as settings.json / benchmarks.json), with a 30-day TTL for hits and 24h
for misses (so renamed/retired orgs re-resolve quickly), plus an in-memory
L1 dict so repeated renders never touch disk or the network. Network
failures are only held in the L1 dict (never persisted), so offline use
degrades to the frontend's initial-letter badge instead of poisoning the
disk cache.

Auth: the overview calls pass token=False explicitly - the app has no HF
token and must not pick one up from the local environment.
"""

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "hf_avatars.json"

HIT_TTL_S = 30 * 24 * 3600    # avatars/rarely change; a month is plenty
MISS_TTL_S = 24 * 3600        # re-check unknown namespaces once a day

# token=False: never pick up a token from the environment or cache.
_api = HfApi(token=False)

_lock = threading.RLock()
# L1 cache: namespace -> {"url", "name", "fetched_at", "mem_only"}
# ("mem_only" marks network-failure negatives that must not hit disk.)
_mem_cache: Dict[str, Dict[str, Any]] = {}

_NAMESPACE_RE = re.compile(r"^[\w.\-]+$")


def is_valid_namespace(namespace: Optional[str]) -> bool:
    """A real HF namespace: word chars, dots, dashes - not empty and not
    the app's own "(ungrouped)" sentinel."""
    ns = (namespace or "").strip()
    return bool(ns) and ns != "(ungrouped)" and bool(_NAMESPACE_RE.match(ns))


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _load_disk() -> Dict[str, Dict[str, Any]]:
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_disk(cache: Dict[str, Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp_path.replace(CACHE_FILE)


def _ttl(entry: Dict[str, Any]) -> int:
    return HIT_TTL_S if entry.get("url") else MISS_TTL_S


def _fresh(entry: Dict[str, Any]) -> bool:
    fetched_at = entry.get("fetched_at") or 0
    return (time.time() - fetched_at) < _ttl(entry)


# ---------------------------------------------------------------------------
# Remote lookup
# ---------------------------------------------------------------------------

def _overview_field(overview: Any, *keys: str) -> Optional[str]:
    for key in keys:
        value = overview.get(key) if isinstance(overview, dict) else getattr(overview, key, None)
        value = str(value).strip() if value else ""
        if value:
            return value
    return None


def _absolute_avatar_url(url: str) -> str:
    # Users with the Hub's default avatar get a *relative* path
    # ("/avatars/<hash>.svg"); orgs and custom avatars come back as full
    # cdn-avatars.huggingface.co URLs. Absolute the relative ones.
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return "https://huggingface.co" + url
    return url


def _fetch_remote(namespace: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Ask the Hub for the namespace's avatar. Tries the org endpoint first,
    then the user endpoint on 404. Returns {"url", "name"} on success,
    None if the namespace exists on neither endpoint, or raises on any
    other (transient) HF/network error - the caller decides how to treat
    those (never persisted).
    """
    for overview_fn in (_api.get_organization_overview, _api.get_user_overview):
        try:
            overview = overview_fn(namespace)
        except (HfHubHTTPError, RepositoryNotFoundError) as e:
            # 404 = "not this kind of namespace" -> try the other endpoint.
            status = getattr(e, "status_code", None)
            if status is None and getattr(e, "response", None) is not None:
                status = getattr(e.response, "status_code", None)
            if isinstance(e, RepositoryNotFoundError) or status == 404:
                continue
            raise
        # huggingface_hub deserializes these into Organization/User
        # dataclasses whose field is avatar_url (the raw JSON is avatarUrl).
        url = _overview_field(overview, "avatar_url", "avatarUrl", "avatar")
        name = _overview_field(overview, "fullname")
        return {"url": _absolute_avatar_url(url) if url else None, "name": name}
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_avatar(namespace: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    """
    Resolve the HF avatar for a namespace, honoring the L1 + disk caches.
    Returns {"url": str|None, "name": str|None} when the namespace is known
    (or was recently looked up), or None for invalid namespaces, unknown
    namespaces, and any network failure - the frontend renders its
    initial-letter badge in all of those cases.
    """
    ns = (namespace or "").strip()
    if not is_valid_namespace(ns):
        return None

    with _lock:
        entry = _mem_cache.get(ns)
        if entry is None:
            entry = _load_disk().get(ns)
            if entry is not None:
                _mem_cache[ns] = entry
        if entry is not None and _fresh(entry):
            return {"url": entry.get("url"), "name": entry.get("name")}

    # Stale (or uncached): go to the Hub.
    not_found = False
    remote: Optional[Dict[str, Optional[str]]] = None
    try:
        remote = _fetch_remote(ns)
        not_found = remote is None
    except Exception:
        # Network down / HF hiccup: negative held in memory only, so a
        # transient outage can't persist a 24h "unknown" into the file.
        not_found = False

    entry = {
        "url": (remote or {}).get("url") if remote is not None else None,
        "name": (remote or {}).get("name") if remote is not None else None,
        "fetched_at": time.time(),
        "mem_only": remote is None and not not_found,
    }
    with _lock:
        _mem_cache[ns] = entry
        if not entry["mem_only"]:
            cache = _load_disk()
            cache[ns] = {k: entry[k] for k in ("url", "name", "fetched_at")}
            _save_disk(cache)
    return {"url": entry["url"], "name": entry["name"]}
