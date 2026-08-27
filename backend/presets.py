"""
presets.py

CRUD storage for "router presets". A preset is a named collection of
launch profiles (by profile id) that together form one llama-server
*router* instance, started with:

    llama-server --models-preset <generated.ini> [--models-max N] …

Presets reference profiles by id - they never copy profile data - so
editing a profile applies to every preset that contains it.

Preset JSON shape (stored as a list in data/presets.json):
{
    "id": "uuid4-string",
    "name": "string",
    "profile_ids": ["profile-uuid", ...],
    "models_max": 4,            # --models-max (0 = unlimited)
    "autoload": true,           # false -> --no-models-autoload
    "load_on_startup": true,    # per-section "load-on-startup" in the INI
    "defaults": "string",       # raw "key = value" lines -> the [*] INI section
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
PRESETS_FILE = DATA_DIR / "presets.json"

# Same lost-update lesson as profiles.py: hold the lock across the entire
# read-modify-write cycle, not just around the final file write.
_lock = threading.RLock()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_all() -> List[Dict[str, Any]]:
    ensure_data_dir()
    if not PRESETS_FILE.exists():
        return []
    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_all(presets: List[Dict[str, Any]]) -> None:
    ensure_data_dir()
    with _lock:
        tmp_path = PRESETS_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(presets, f, indent=2)
        tmp_path.replace(PRESETS_FILE)


def list_presets() -> List[Dict[str, Any]]:
    presets = _read_all()
    presets.sort(key=lambda p: p.get("updated_at") or 0, reverse=True)
    return presets


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    for p in _read_all():
        if p.get("id") == preset_id:
            return p
    return None


def create_preset(name: str, profile_ids: List[str], models_max: int = 4,
                  autoload: bool = True, load_on_startup: bool = True,
                  defaults: str = "") -> Dict[str, Any]:
    now = time.time()
    with _lock:
        presets = _read_all()
        preset = {
            "id": str(uuid.uuid4()),
            "name": (name or "").strip() or "Router preset",
            "profile_ids": list(profile_ids or []),
            "models_max": max(0, int(models_max or 0)),
            "autoload": bool(autoload),
            "load_on_startup": bool(load_on_startup),
            "defaults": defaults or "",
            "created_at": now,
            "updated_at": now,
        }
        presets.append(preset)
        _write_all(presets)
    return preset


def update_preset(preset_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with _lock:
        presets = _read_all()
        for p in presets:
            if p.get("id") == preset_id:
                allowed = {"name", "profile_ids", "models_max", "autoload", "load_on_startup", "defaults"}
                for k, v in (updates or {}).items():
                    if k in allowed:
                        if k == "name":
                            p[k] = (v or "").strip() or p.get("name") or "Router preset"
                        elif k == "profile_ids":
                            p[k] = list(v or [])
                        elif k == "models_max":
                            p[k] = max(0, int(v or 0))
                        elif k in ("autoload", "load_on_startup"):
                            p[k] = bool(v)
                        elif k == "defaults":
                            p[k] = v or ""
                p["updated_at"] = time.time()
                _write_all(presets)
                return p
    return None


def delete_preset(preset_id: str) -> bool:
    with _lock:
        presets = _read_all()
        new_presets = [p for p in presets if p.get("id") != preset_id]
        if len(new_presets) == len(presets):
            return False
        _write_all(new_presets)
    return True
