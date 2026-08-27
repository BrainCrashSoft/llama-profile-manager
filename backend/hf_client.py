"""
hf_client.py

Talks to the public Hugging Face API (no auth) to list the GGUF files in a
model repo, so the user can pick a quantization to download without leaving
the app. Read-only except for what download_manager.py writes to disk.
"""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from . import gguf_utils, settings

HF_BASE = "https://huggingface.co"
USER_AGENT = "LlamaModelManager/1.0 (+local desktop app; https://github.com/ggml-org/llama.cpp)"

# Accepts a bare "org/repo" or a full huggingface.co URL (with or without a
# /tree/main, /blob/main/..., query string, or trailing slash).
_URL_RE = re.compile(r"^https?://huggingface\.co/([^/?#]+/[^/?#]+)", re.IGNORECASE)
_REPO_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


def parse_repo_id(raw: str) -> str:
    """Extract a normalized 'org/repo' id from a URL or bare id string."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Enter a Hugging Face repo (e.g. 'org/repo') or a huggingface.co URL.")

    m = _URL_RE.match(raw)
    if m:
        return m.group(1)

    if _REPO_ID_RE.match(raw):
        return raw

    raise ValueError(f"That doesn't look like a Hugging Face repo id or URL: {raw!r}")


def _api_get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError("Repository not found on Hugging Face. Check the name and try again.")
        if e.code in (401, 403):
            raise ValueError("This repository is private or gated - this app can't access it without authentication.")
        raise ValueError(f"Hugging Face returned an error (HTTP {e.code}).")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not reach Hugging Face: {e.reason}")
    except (TimeoutError, OSError) as e:
        raise ValueError(f"Timed out reaching Hugging Face: {e}")


def fetch_repo_tree(repo_id: str) -> List[Dict[str, Any]]:
    """Full recursive file tree for the repo's main branch."""
    url = f"{HF_BASE}/api/models/{repo_id}/tree/main?recursive=true"
    data = _api_get(url)
    if not isinstance(data, list):
        raise ValueError("Unexpected response from Hugging Face while listing files.")
    return data


def list_gguf_groups(repo_id: str) -> Dict[str, Any]:
    """
    Fetch the repo's file tree and group its .gguf files the same way the
    local scanner groups them (multi-part files collapsed into one entry).
    """
    tree = fetch_repo_tree(repo_id)
    gguf_entries = [e for e in tree if e.get("type") == "file" and e.get("path", "").lower().endswith(".gguf")]
    if not gguf_entries:
        raise ValueError("No .gguf files found in this repository.")

    groups: Dict[str, Dict[str, Any]] = {}
    for entry in gguf_entries:
        path = entry["path"]
        filename = path.rsplit("/", 1)[-1]
        size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
        stem = filename[:-5]  # strip trailing ".gguf"

        base_stem, part_label = gguf_utils.split_multipart(stem)
        group_key = base_stem if base_stem else stem
        quant = gguf_utils.parse_quant(filename)

        group = groups.setdefault(group_key, {
            "group_name": group_key,
            "quant": None,
            "files": [],
            "total_size_bytes": 0,
        })
        group["files"].append({"path": path, "filename": filename, "size_bytes": size, "part": part_label})
        group["total_size_bytes"] += size
        if group["quant"] is None and quant:
            group["quant"] = quant

    for g in groups.values():
        g["files"].sort(key=lambda f: f["filename"])
        g["is_multipart"] = len(g["files"]) > 1

    return {
        "repo_id": repo_id,
        "groups": sorted(groups.values(), key=lambda g: g["total_size_bytes"]),
    }


def mark_downloaded(repo_id: str, groups: List[Dict[str, Any]]) -> None:
    """
    Mutates `groups` in place, adding an `already_downloaded` bool to each -
    true if every file in the group already exists under {root}/org/repo/
    for any of the configured model root folders.

    Downloads mirror the HF repo layout, so files may live in subfolders
    (e.g. a per-quant folder holding a multi-part model). Each file is
    therefore matched by its repo-relative path first, with the bare file
    name as a fallback for flat (or manually rearranged) layouts.
    """
    org, repo = repo_id.split("/", 1)
    roots = settings.get_model_root_folders()

    existing_paths = set()
    existing_names = set()
    for root in roots:
        repo_dir = Path(root) / org / repo
        if repo_dir.is_dir():
            try:
                for f in repo_dir.rglob("*"):
                    if f.is_file():
                        existing_paths.add(f.relative_to(repo_dir).as_posix().lower())
                        existing_names.add(f.name.lower())
            except OSError:
                continue

    for g in groups:
        g["already_downloaded"] = bool(g["files"]) and all(
            f["path"].lower() in existing_paths
            or f["filename"].lower() in existing_names
            for f in g["files"]
        )
