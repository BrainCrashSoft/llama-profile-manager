"""
scanner.py

Recursively scans configured root folders for .gguf files, groups multi-part
files (e.g. model-00001-of-00003.gguf) into a single logical model entry,
parses quantization type from the filename, and caches results to JSON so
subsequent app launches are instant.

Expected layout: root/org/repo/*.gguf (Hugging Face style), but flat
layouts (root/*.gguf or root/repo/*.gguf) are also handled gracefully.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import gguf_utils

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "scan_cache.json"

_lock = threading.Lock()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _scan_one_root(root: Path) -> List[Dict[str, Any]]:
    """
    Scan a single root folder. Returns a flat list of "model group" dicts:
    {
        "org": str,
        "repo": str,
        "model_name": str,           # grouped display name (multipart-safe)
        "quant": str | None,
        "files": [ { "path", "filename", "size_bytes", "modified", "part" } ],
        "total_size_bytes": int,
    }
    """
    if not root.exists() or not root.is_dir():
        return []

    # group_key -> group dict
    groups: Dict[tuple, Dict[str, Any]] = {}

    for gguf_path in root.rglob("*.gguf"):
        try:
            stat = gguf_path.stat()
        except OSError:
            continue

        rel = gguf_path.relative_to(root)
        parts = rel.parts

        # Infer org/repo from folder structure: root/org/repo/file.gguf
        if len(parts) >= 3:
            org, repo = parts[0], parts[1]
        elif len(parts) == 2:
            org, repo = "(ungrouped)", parts[0]
        else:
            org, repo = "(ungrouped)", "(ungrouped)"

        stem = gguf_path.stem
        base_stem, _part_label = gguf_utils.split_multipart(stem)
        is_multipart = base_stem is not None
        group_stem = base_stem if is_multipart else stem

        quant = gguf_utils.parse_quant(gguf_path.name)
        group_key = (org, repo, group_stem)

        entry = {
            "path": str(gguf_path),
            "filename": gguf_path.name,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
            "part": None,
        }
        if is_multipart:
            entry["part"] = _part_label

        if group_key not in groups:
            groups[group_key] = {
                "org": org,
                "repo": repo,
                "model_name": group_stem,
                "quant": quant,
                "files": [],
                "total_size_bytes": 0,
            }
        groups[group_key]["files"].append(entry)
        groups[group_key]["total_size_bytes"] += stat.st_size
        if groups[group_key]["quant"] is None and quant:
            groups[group_key]["quant"] = quant

    # sort files within each group by part number for stable ordering
    for g in groups.values():
        g["files"].sort(key=lambda f: f["filename"])
        g["is_multipart"] = len(g["files"]) > 1
        g["latest_modified"] = max((f["modified"] for f in g["files"]), default=0)

    return list(groups.values())


def scan_roots(roots: List[str]) -> Dict[str, Any]:
    """Scan all configured roots and return a combined result, also caching it."""
    all_groups: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for root_str in roots:
        root = Path(root_str)
        try:
            all_groups.extend(_scan_one_root(root))
        except PermissionError:
            errors.append({"root": root_str, "error": "Permission denied while scanning this folder."})
        except OSError as e:
            errors.append({"root": root_str, "error": f"Could not scan this folder: {e}"})

    result = {
        "scanned_at": time.time(),
        "roots": roots,
        "models": all_groups,
        "errors": errors,
        "total_models": len(all_groups),
        "total_files": sum(len(g["files"]) for g in all_groups),
    }
    _save_cache(result)
    return result


MMPROJ_EXTENSIONS = {".gguf", ".bin", ".safetensors"}

# Extensions that identify a custom Jinja chat-template file.
CHAT_TEMPLATE_EXTENSIONS = {".jinja"}


def find_mmproj_files(model_path: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Find multimodal projector files in the folder containing the given model
    file (plus one level of subfolders, since some repos nest them).

    A file qualifies when its name contains "mmproj" (case-insensitive) - the
    convention used by most HF repos (mmproj-model-f16.gguf, …). Returns
    [{ "path", "filename" }, …] sorted by filename; empty list if the folder
    can't be read or nothing matches.
    """
    model_file = Path(model_path)
    folder = model_file.parent if model_file.is_file() else model_file
    if not folder.exists() or not folder.is_dir():
        return []

    results: List[Dict[str, Any]] = []
    seen: set = set()

    def consider(p: Path) -> None:
        if len(results) >= max_results:
            return
        if p.suffix.lower() not in MMPROJ_EXTENSIONS or "mmproj" not in p.name.lower():
            return
        resolved = p.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        results.append({"path": str(p), "filename": p.name})

    try:
        entries = sorted(folder.iterdir(), key=lambda e: e.name.lower())
    except OSError:
        return []

    for entry in entries:
        if entry.is_file():
            consider(entry)
        elif entry.is_dir():
            try:
                for sub in sorted(entry.iterdir(), key=lambda e: e.name.lower()):
                    if sub.is_file():
                        consider(sub)
            except OSError:
                continue

    results.sort(key=lambda f: f["filename"].lower())
    return results


def find_chat_template_files(model_path: str, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Find candidate custom chat-template files in the folder containing the
    given model file (plus one level of subfolders). A file qualifies when
    it has a Jinja template extension (.jinja). Returns [{ "path",
    "filename" }, …] sorted by filename; empty list if the folder can't be
    read or nothing matches.
    """
    model_file = Path(model_path)
    folder = model_file.parent if model_file.is_file() else model_file
    if not folder.exists() or not folder.is_dir():
        return []

    results: List[Dict[str, Any]] = []
    seen: set = set()

    def consider(p: Path) -> None:
        if len(results) >= max_results:
            return
        if p.suffix.lower() not in CHAT_TEMPLATE_EXTENSIONS:
            return
        resolved = p.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        results.append({"path": str(p), "filename": p.name})

    try:
        entries = sorted(folder.iterdir(), key=lambda e: e.name.lower())
    except OSError:
        return []

    for entry in entries:
        if entry.is_file():
            consider(entry)
        elif entry.is_dir():
            try:
                for sub in sorted(entry.iterdir(), key=lambda e: e.name.lower()):
                    if sub.is_file():
                        consider(sub)
            except OSError:
                continue

    results.sort(key=lambda f: f["filename"].lower())
    return results


def _save_cache(result: Dict[str, Any]) -> None:
    ensure_data_dir()
    with _lock:
        tmp_path = CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        tmp_path.replace(CACHE_FILE)


def load_cache() -> Optional[Dict[str, Any]]:
    ensure_data_dir()
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
