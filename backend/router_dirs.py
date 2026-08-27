"""
router_dirs.py

CRUD storage for "router dir" profiles. A router dir launches llama-server's
router mode pointed at a folder of models, instead of a generated INI:

    llama-server --models-dir <path> [--models-max N] [--no-models-autoload] <defaults>

The models themselves are auto-discovered from the directory (no per-model
INI). ``params`` holds the *shared* default CLI parameters applied to every
model in the folder, edited with the same parameter configurator used for
regular profiles.

Router-dir JSON shape (stored as a list in data/router_dirs.json):
{
    "id": "uuid4-string",
    "name": "string",
    "models_dir": "string",     # value for --models-dir
    "params": {...},            # global defaults, keyed by param_schema keys
    "custom_flags": "string",   # raw extra CLI flags (advanced mode)
    "models_max": 4,            # --models-max (0 = unlimited)
    "autoload": true,           # false -> --no-models-autoload
    "notes": "string",
    "created_at": float,
    "updated_at": float,
}
"""

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ROUTER_DIRS_FILE = DATA_DIR / "router_dirs.json"

# Same lost-update lesson as profiles.py / presets.py: hold the lock across the
# entire read-modify-write cycle, not just around the final file write.
_lock = threading.RLock()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_all() -> List[Dict[str, Any]]:
    ensure_data_dir()
    if not ROUTER_DIRS_FILE.exists():
        return []
    try:
        with open(ROUTER_DIRS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_all(items: List[Dict[str, Any]]) -> None:
    ensure_data_dir()
    with _lock:
        tmp_path = ROUTER_DIRS_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        tmp_path.replace(ROUTER_DIRS_FILE)


def list_router_dirs() -> List[Dict[str, Any]]:
    items = _read_all()
    items.sort(key=lambda p: p.get("updated_at") or 0, reverse=True)
    return items


def get_router_dir(router_dir_id: str) -> Optional[Dict[str, Any]]:
    for p in _read_all():
        if p.get("id") == router_dir_id:
            return p
    return None


def create_router_dir(name: str, models_dir: str, params: Dict[str, Any],
                      custom_flags: str = "", models_max: int = 4,
                      autoload: bool = True, notes: str = "") -> Dict[str, Any]:
    now = time.time()
    with _lock:
        items = _read_all()
        item = {
            "id": str(uuid.uuid4()),
            "name": (name or "").strip() or "Router dir",
            "models_dir": (models_dir or "").strip(),
            "params": params or {},
            "custom_flags": custom_flags or "",
            "models_max": max(0, int(models_max or 0)),
            "autoload": bool(autoload),
            "notes": notes or "",
            "created_at": now,
            "updated_at": now,
        }
        items.append(item)
        _write_all(items)
    return item


def update_router_dir(router_dir_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with _lock:
        items = _read_all()
        for p in items:
            if p.get("id") == router_dir_id:
                allowed = {"name", "models_dir", "params", "custom_flags",
                           "models_max", "autoload", "notes"}
                for k, v in (updates or {}).items():
                    if k not in allowed:
                        continue
                    if k == "name":
                        p[k] = (v or "").strip() or p.get("name") or "Router dir"
                    elif k == "models_dir":
                        p[k] = (v or "").strip()
                    elif k == "models_max":
                        p[k] = max(0, int(v or 0))
                    elif k == "autoload":
                        p[k] = bool(v)
                    elif k in ("params", "custom_flags", "notes"):
                        p[k] = v if v is not None else p.get(k, "")
                p["updated_at"] = time.time()
                _write_all(items)
                return p
    return None


def delete_router_dir(router_dir_id: str) -> bool:
    with _lock:
        items = _read_all()
        new_items = [p for p in items if p.get("id") != router_dir_id]
        if len(new_items) == len(items):
            return False
        _write_all(new_items)
    return True
