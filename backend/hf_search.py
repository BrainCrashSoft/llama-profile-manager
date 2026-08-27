"""
hf_search.py

Model *search* (as opposed to hf_client.py, which resolves one specific
repo's file tree via the raw REST API). This uses the official
`huggingface_hub` client library, since search ranking/filtering is exactly
what it's built for. Results are restricted to repos tagged "gguf" - the
Hub auto-tags any repo containing .gguf files. Results can be ordered by a
selectable sort key (downloads by default), since this app only cares about
repos it can actually pull a GGUF from.
"""

from typing import Any, Dict, List, Optional

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

DEFAULT_LIMIT = 10
MAX_LIMIT = 50
DEFAULT_SORT = "downloads"
# Values accepted by the Hub's list_models(sort=...) parameter.
SORT_OPTIONS = ("downloads", "likes", "likes7d", "lastModified", "createdAt")

_api = HfApi()


def clamp_limit(limit: Optional[int]) -> int:
    try:
        n = int(limit) if limit is not None else DEFAULT_LIMIT
    except (TypeError, ValueError):
        n = DEFAULT_LIMIT
    return max(1, min(n, MAX_LIMIT))


def normalize_sort(sort: Optional[str]) -> str:
    sort = (sort or DEFAULT_SORT).strip()
    return sort if sort in SORT_OPTIONS else DEFAULT_SORT


def search_models(
    query: str,
    limit: Optional[int] = None,
    sort: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search the Hub for GGUF-tagged model repos, ordered by `sort`
    (downloads, likes, lastModified, or createdAt).
    `query` may be blank to just browse the repos by the chosen sort.
    """
    effective_limit = clamp_limit(limit)
    effective_sort = normalize_sort(sort)
    query = (query or "").strip()

    try:
        results = _api.list_models(
            search=query or None,
            filter="gguf",
            sort=effective_sort,
            limit=effective_limit,
            expand=["downloads", "likes", "lastModified", "tags", "pipeline_tag", "gated"],
        )
        return [_to_dict(m) for m in results]
    except HfHubHTTPError as e:
        raise ValueError(f"Hugging Face search failed: {e}")
    except Exception as e:
        raise ValueError(f"Could not reach Hugging Face: {e}")


def _to_dict(m: Any) -> Dict[str, Any]:
    last_modified = getattr(m, "last_modified", None)
    return {
        "repo_id": m.id,
        "downloads": getattr(m, "downloads", None) or 0,
        "likes": getattr(m, "likes", None) or 0,
        "last_modified": last_modified.isoformat() if last_modified else None,
        "tags": list(getattr(m, "tags", None) or []),
        "pipeline_tag": getattr(m, "pipeline_tag", None),
        "gated": bool(getattr(m, "gated", False)),
    }
