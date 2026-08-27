"""
gguf_meta.py

Reads GGUF file metadata with the `gguf` Python package (the llama.cpp
team's GGUF-format implementation - the Python equivalent of the retired
`gguf-dump` CLI). The profile editor uses this to learn a model's block
count (the slider maximum for --n-cpu-moe) and to make the model's
context length and chat template available.

Only the metadata header is touched (the reader memmaps the file and
never loads tensor data), so this is fast even for very large models.
Results are cached per (path, mtime_ns, size), both in memory and on disk
(data/gguf_meta_cache.json - same JSON-in-data/ convention as
settings.json / hf_avatars.json), so the first open of a model is fast
even right after an app restart.
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from gguf import GGUFReader

from . import gguf_utils

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "gguf_meta_cache.json"
MAX_ENTRIES = 1000

_cache_lock = threading.Lock()
# key "path|mtime_ns|size" -> facts. L1 in this dict; the same dict is
# mirrored to CACHE_FILE so restarts don't pay the parse cost again.
_cache: Dict[str, Dict[str, Any]] = {}
_disk_loaded = False


class GgufMetaError(ValueError):
    """Bad path or unreadable/corrupt GGUF file (maps to HTTP 400)."""


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _load_disk() -> Dict[str, Dict[str, Any]]:
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_disk(cache: Dict[str, Dict[str, Any]]) -> None:
    # The cache is pure speed: a write failure must never break a read.
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        tmp.replace(CACHE_FILE)
    except OSError:
        pass


def _ensure_disk_loaded() -> None:
    """Merge the on-disk cache into _cache once per process. The caller
    holds _cache_lock."""
    global _disk_loaded
    if not _disk_loaded:
        _cache.update(_load_disk())
        _disk_loaded = True


def _field_value(reader: GGUFReader, name: str) -> Any:
    """The Python value of a metadata field, or None if absent."""
    f = reader.get_field(name)
    if f is None or not f.types:
        return None
    return f.contents()


def _as_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _resolve_read_path(path: Path) -> Path:
    """
    If `path` is missing but is a multi-part GGUF part (-NNNNN-of-MMMMM),
    fall back to part 00001 in the same folder: every part of a split model
    carries the complete metadata header, so part 1's header is the model's
    header. Returns the original path when it exists (or no fallback
    applies).
    """
    if path.is_file():
        return path
    base_stem, part = gguf_utils.split_multipart(path.stem)
    if base_stem is not None and "/" in part:
        total = part.split("/", 1)[1]
        candidate = path.with_name(f"{base_stem}-00001-of-{total}.gguf")
        if candidate.is_file():
            return candidate
    return path


def read_gguf_facts(path: str) -> Dict[str, Any]:
    """
    Read the GGUF metadata facts the UI cares about from the file at
    `path`:

        {
          "path": str,             # the file actually read (may differ from
                                   # `path` via the multi-part fallback)
          "architecture": str | None,
          "context_length": int | None,
          "block_count": int | None,
          "expert_count": int | None,   # None/0 = dense model; >0 = MoE
          "chat_template": str | None,
        }

    Any of the facts may be None when the file doesn't define it.
    Field resolution mirrors llama.cpp's own conventions:
      * context_length - general.context_length, else <arch>.context_length
      * block_count    - <arch>.block_count
      * expert_count   - <arch>.expert_count, else general.expert_count
      * chat_template  - conversation.template, else tokenizer.chat_template
    Raises GgufMetaError for a non-.gguf path, a missing file, or a file
    that can't be parsed as GGUF.
    """
    p = Path(str(path).strip())
    if not p.name.lower().endswith(".gguf"):
        raise GgufMetaError(f"Not a .gguf file: {p.name or path!r}")

    read_path = _resolve_read_path(p)
    if not read_path.is_file():
        raise GgufMetaError(f"GGUF file not found: {read_path}")

    try:
        st = read_path.stat()
    except OSError as e:
        raise GgufMetaError(f"Can't stat {read_path}: {e}") from e

    # mtime_ns (int, not the float st_mtime) so the key round-trips
    # exactly through JSON.
    key = f"{read_path}|{st.st_mtime_ns}|{st.st_size}"
    with _cache_lock:
        _ensure_disk_loaded()
        cached = _cache.get(key)
    if cached is not None:
        return dict(cached)

    # np.memmap (used by GGUFReader) keeps the file open until the reader
    # object is gone, so build the result first and drop the reader before
    # caching - CPython frees it as soon as the function-local reference
    # does, releasing the file handle.
    try:
        reader = GGUFReader(str(read_path))
    except Exception as e:  # noqa: BLE001 - corrupt/partial downloads etc.
        raise GgufMetaError(
            f"Could not read GGUF metadata from {read_path.name}: {e}"
        ) from e

    try:
        arch = _field_value(reader, "general.architecture")
        arch = arch if isinstance(arch, str) and arch else None

        ctx = _field_value(reader, "general.context_length")
        if ctx is None and arch:
            ctx = _field_value(reader, f"{arch}.context_length")

        blocks = _field_value(reader, f"{arch}.block_count") if arch else None

        experts = _field_value(reader, f"{arch}.expert_count") if arch else None
        if experts is None:
            experts = _field_value(reader, "general.expert_count")

        # Classic exports use conversation.template; newer ones put the
        # Jinja template in tokenizer.chat_template. Check both.
        chat = _field_value(reader, "conversation.template")
        if not (isinstance(chat, str) and chat):
            chat = _field_value(reader, "tokenizer.chat_template")
    finally:
        del reader

    facts: Dict[str, Any] = {
        "path": str(read_path),
        "architecture": arch,
        "context_length": _as_positive_int(ctx),
        "block_count": _as_positive_int(blocks),
        "expert_count": _as_positive_int(experts),
        "chat_template": chat if isinstance(chat, str) and chat else None,
    }
    with _cache_lock:
        _ensure_disk_loaded()
        if len(_cache) >= MAX_ENTRIES:
            # Evict the oldest entries (dicts keep insertion order) so the
            # file can't grow without bound on a huge library.
            for old in list(_cache)[: len(_cache) - MAX_ENTRIES + 1]:
                del _cache[old]
        _cache[key] = dict(facts)
        _save_disk(_cache)
    return facts
