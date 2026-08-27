"""
benchmarks.py

Benchmark-record store - deliberately its own JSON file (data/benchmarks.json),
decoupled from the profiles store so benchmark history outlives profile edits
and deletions (records keep a name + params snapshot).

A benchmark record has the shape:
{
    "id": "uuid4-string",
    "profile_id": "uuid-string" | None,     # FK to the profile that started the run
                                              # (may dangle after the profile is deleted)
    "profile_name": "string",               # snapshot at run time
    "profile_params_snapshot": {...},       # full profile in the *export* serialization
                                            # (see profiles.export_profile) - what gets
                                            # re-run / imported / exported later
    "model_path": "string",                 # pulled out of the snapshot for easy display
    "model_name": "string",
    "server_version": "string",             # what `llama-server --version` reported
    "prefill_tps": float | None,
    "generation_tps": float | None,
    "prompt_tokens": int | None,            # actual token counts from the server
    "gen_tokens": int | None,
    "duration_s": float | None,
    "params_hash": "sha256-hex",
    "status": "running" | "completed" | "failed" | "cancelled",
    "progress": "string",                   # human-readable stage while running
    "error": "string",
    "re_ran_from": "uuid" | None,           # set when this run re-ran another record
    "started_at": float,
    "timestamp": float | None,              # when the run finished (None while running)
    "raw_output": "string",                 # JSON text: per-run timings, debug info
}

The two utility functions below - benchmarkable_params() and params_hash() -
are the single source of truth for "which fields of a profile count as its
parameters": everything that actually lands on the llama-server command line
(model path, structured params, custom flags) and nothing cosmetic (name,
notes, ids, timestamps). The benchmark runner hashes the snapshot it was
given with these, and the profile save path hashes the live profile with the
same functions, so staleness detection is an exact string compare and the
two code paths can never drift apart.
"""

import hashlib
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BENCHMARKS_FILE = DATA_DIR / "benchmarks.json"

# RLock, mirroring profiles.py: public helpers may nest (update_record is
# called from the runner while holding no other lock, but the store must
# stay safe for arbitrary call nesting).
_lock = threading.RLock()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Benchmarkable params + canonical hash
# ---------------------------------------------------------------------------

def _normalize_value(value: Any) -> Any:
    """Normalize one param value to what the command builder actually puts
    on the command line, so semantically-identical params hash identically:
      * None / "" - the command builder skips them → treated as absent
      * False - store-true flags are omitted when false → absent
      * 4096.0 === 4096 - JSON round-trips (file save, API boundary) can
        flip an int to a float and back; that must not read as a change
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value if value else None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return _normalize_container(value)
    return value


def _normalize_container(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value


def benchmarkable_params(profile_like: Dict[str, Any]) -> Dict[str, Any]:
    """
    The canonical "params only" object for anything shaped like a profile
    (a stored profile dict, or a benchmark record's params snapshot):
    exactly the fields that end up on the llama-server command line.

    Centralizing this in one place is what keeps the benchmark-run path and
    the staleness-check path in lockstep - both must call this, never their
    own field lists.
    """
    params = profile_like.get("params") or {}
    clean: Dict[str, Any] = {}
    for key, value in params.items():
        normalized = _normalize_value(value)
        if normalized is not None:
            clean[str(key)] = normalized
    return {
        "model_path": str(profile_like.get("model_path") or "").strip(),
        "params": clean,
        # whitespace-normalized so "--flag 1" and "--flag   1" aren't a change
        "custom_flags": " ".join(str(profile_like.get("custom_flags") or "").split()),
    }


def params_hash(profile_like: Dict[str, Any]) -> str:
    """
    SHA-256 over the canonicalized params: stable key order (sort_keys),
    normalized types (int/float, absent-vs-empty), so the hash only changes
    when something actually passed to the server changes. This is the
    value stored on the profile's benchmark_badge.params_hash.
    """
    canonical = benchmarkable_params(profile_like)
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def _read_all() -> List[Dict[str, Any]]:
    ensure_data_dir()
    if not BENCHMARKS_FILE.exists():
        return []
    try:
        with open(BENCHMARKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_all(records: List[Dict[str, Any]]) -> None:
    ensure_data_dir()
    with _lock:
        tmp_path = BENCHMARKS_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        tmp_path.replace(BENCHMARKS_FILE)


def list_benchmarks() -> List[Dict[str, Any]]:
    records = _read_all()
    records.sort(key=lambda r: r.get("started_at") or 0, reverse=True)
    return records


def get_benchmark(benchmark_id: str) -> Optional[Dict[str, Any]]:
    for r in _read_all():
        if r.get("id") == benchmark_id:
            return r
    return None


def create_record(*, profile_id: Optional[str], profile_name: str,
                  snapshot: Dict[str, Any], model_path: str,
                  params_hash: str, started_at: float,
                  server_version: str = "", re_ran_from: Optional[str] = None) -> Dict[str, Any]:
    record = {
        "id": str(uuid.uuid4()),
        "profile_id": profile_id,
        "profile_name": profile_name,
        "profile_params_snapshot": snapshot,
        "model_path": model_path,
        "model_name": Path(model_path).name if model_path else "",
        "server_version": server_version,
        "prefill_tps": None,
        "generation_tps": None,
        "prompt_tokens": None,
        "gen_tokens": None,
        "duration_s": None,
        "params_hash": params_hash,
        "status": "running",
        "progress": "preparing",
        "error": "",
        "re_ran_from": re_ran_from,
        "started_at": started_at,
        "timestamp": None,
        "raw_output": "",
    }
    with _lock:
        records = _read_all()
        records.append(record)
        _write_all(records)
    return record


def update_record(benchmark_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    with _lock:
        records = _read_all()
        for r in records:
            if r.get("id") == benchmark_id:
                r.update(fields)
                _write_all(records)
                return r
        return None


def delete_benchmark(benchmark_id: str) -> bool:
    with _lock:
        records = _read_all()
        kept = [r for r in records if r.get("id") != benchmark_id]
        if len(kept) == len(records):
            return False
        _write_all(kept)
    return True
