"""
settings.py

Loads/saves app-wide settings (llama-server binary path, model root folders,
default host/port, theme) to a single JSON file in the data directory.
"""

import json
import platform
import threading
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

_lock = threading.Lock()

DEFAULT_SETTINGS: Dict[str, Any] = {
    # Named llama-server builds. Each entry: {"name": str, "path": str}.
    "llama_servers": [],
    # Name of the entry in llama_servers that is currently in use.
    "active_llama_server": "",
    # Effective path of the active build. Kept in sync with the list above
    # (see _normalize_llama_servers) so every consumer of this key - old and
    # new - always sees the binary that will actually be launched.
    "llama_server_path": "",          # full path to llama-server(.exe)
    "model_root_folders": [],         # list of strings (paths)
    "default_host": "127.0.0.1",
    "default_port": 8080,
    # Pass -v to llama-server on start (logs everything, for debugging).
    "verbose": False,
    "theme": "dark",                  # "dark" | "light"
    # Controls what this app's OWN management UI binds to (not llama-server's
    # host/port above, which is separate). False = 127.0.0.1 only, this
    # computer alone. True = 0.0.0.0, reachable from other devices on the
    # network. There is no login, so this is opt-in and takes effect on the
    # next app restart. See main.py.
    "allow_lan_access": False,
    # Benchmark run-form defaults (Benchmarks page): target prompt length,
    # generated-token count, repetitions, and the optional custom prompt.
    # Remembered so the form keeps the user's last values across restarts.
    "bench_prompt_tokens": 512,
    "bench_gen_tokens": 256,
    "bench_repetitions": 5,
    "bench_custom_prompt": "",
}


def _binary_name() -> str:
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> Dict[str, Any]:
    ensure_data_dir()
    if not SETTINGS_FILE.exists():
        # save_settings() takes the lock itself; don't hold it here too
        # (the lock is not reentrant), or this deadlocks on first run.
        save_settings(DEFAULT_SETTINGS)
        return dict(DEFAULT_SETTINGS)
    with _lock:
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = dict(DEFAULT_SETTINGS)
            merged.update(data or {})
            _normalize_llama_servers(merged)
            return merged
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable settings file: fall back to defaults but
            # don't clobber the broken file, so the user can inspect it.
            return dict(DEFAULT_SETTINGS)


def save_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    ensure_data_dir()
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings or {})
    with _lock:
        tmp_path = SETTINGS_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        tmp_path.replace(SETTINGS_FILE)
    return merged


def update_settings(partial: Dict[str, Any]) -> Dict[str, Any]:
    current = load_settings()
    current.update(partial or {})
    return save_settings(current)


def get_model_root_folders() -> List[str]:
    return load_settings().get("model_root_folders", [])


def _normalize_llama_servers(merged: Dict[str, Any]) -> None:
    """
    Keep the multi-version llama-server settings consistent, in memory:
      * migrate a legacy single ``llama_server_path`` into the list (first run
        after an upgrade), named after the file's stem (e.g. "b6120" for
        llama-server-b6120.exe);
      * drop malformed entries;
      * fall back to the first usable entry if the active name is stale/empty;
      * always sync ``llama_server_path`` to the active entry's path, so that
        key (used by launch code and command previews) never goes stale.
    """
    servers = merged.get("llama_servers")
    if not isinstance(servers, list):
        servers = []
    cleaned = [
        {"name": str(e.get("name", "")), "path": str(e.get("path", ""))}
        for e in servers
        if isinstance(e, dict) and (e.get("name") or e.get("path"))
    ]

    # Legacy migration: a single configured path with no list yet.
    if not cleaned and merged.get("llama_server_path"):
        legacy = str(merged["llama_server_path"])
        stem = Path(legacy).stem or "llama-server"
        cleaned.append({"name": stem, "path": legacy})

    active = str(merged.get("active_llama_server", ""))
    if active not in {e["name"] for e in cleaned}:
        # Stale or unset active: pick the first entry that has a path.
        first = next((e for e in cleaned if e["path"]), None)
        active = first["name"] if first else ""

    merged["llama_servers"] = cleaned
    merged["active_llama_server"] = active
    merged["llama_server_path"] = next(
        (e["path"] for e in cleaned if e["name"] == active), ""
    )


def resolve_llama_server_path() -> str:
    """Return the active llama-server build's path, or "" if none is set."""
    return load_settings().get("llama_server_path", "")


def expected_binary_name() -> str:
    return _binary_name()
