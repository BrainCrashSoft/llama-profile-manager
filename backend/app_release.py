"""
app_release.py

Checks whether a newer release of THIS app (LPM) exists on GitHub (no auth).
Mirrors the llama_release.py pattern: stdlib urllib only, a shared
certifi-fallback SSL context, a 1-hour in-memory TTL cache, and a friendly
ValueError on offline/bad data.

This app is run from a source checkout (start.bat -> venv -> python main.py),
so a newer version never applies itself: Settings shows a badge plus a link
to the release page, and the user updates with `git pull`. The check only
hits GitHub's public releases API for this app's own repo - no telemetry.

The repo to check is APP_REPO ("owner/repo"): the default below is a
placeholder until the public repo exists, and the LPM_REPO env var overrides
it (so forks can point the check at their own repo without code changes).
With the placeholder, /releases/latest 404s and the Settings badge simply
stays hidden - the check is silently disabled until a real repo is set.
"""

import json
import os
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

from .llama_release import _ssl_context, _use_certifi_context

# Placeholder until the public repo exists - set this (or the LPM_REPO env
# var, which wins) to the real "owner/repo". See module docstring for what
# happens while it's still a placeholder.
APP_REPO = os.environ.get("LPM_REPO") or "lpm/CHANGE_ME"
USER_AGENT = "LlamaProfileManager (local desktop app; in-app release check)"


def _urlopen(url: str, timeout: float = 20.0):
    """urlopen with the shared (certifi-fallback) SSL context (same as
    llama_release._urlopen)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError) and _use_certifi_context():
            return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())
        raise


def _urlopen_with(req: urllib.request.Request, timeout: float = 20.0):
    """Like _urlopen but for a pre-built Request (extra headers etc.)."""
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError) and _use_certifi_context():
            return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())
        raise


# ---------------------------------------------------------------------------
# Semver parsing / comparison
# ---------------------------------------------------------------------------

# Release tags look like "v0.1.0"; tolerate a missing patch segment
# ("v0.1" -> (0, 1, 0)) and an optional leading V. Pre-release / build
# suffixes ("-rc1", "+build") are not part of the numeric triple - they
# only affect ordering (see _version_key).
_VERSION_RE = re.compile(r"^[vV]?(\d+)(?:\.(\d+))?(?:\.(\d+))?")
# A "-" or "+" separator right after the numeric core marks a pre-release
# or build-metadata suffix ("0.2.0-rc1", "0.2.0+build.5").
_SUFFIX_RE = re.compile(r"^\d+(?:\.\d+){0,2}[-+]")


def parse_version(tag: str) -> Tuple[int, int, int]:
    """
    Parse a release tag into a comparable (major, minor, patch) tuple.

    "v0.2.0" -> (0, 2, 0); the "v" prefix is optional, a missing patch
    segment becomes 0, and pre-release/build suffixes are ignored here
    (is_newer/_version_key is where they make versions sort older).
    Raises ValueError for anything that isn't a version tag at all.
    """
    s = (tag or "").strip()
    m = _VERSION_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized version tag: {tag!r}")
    # The match must be the WHOLE tag up to an allowed suffix - a dangling
    # garbage tail like "0.1.0xyz" is not a version.
    rest = s[m.end():]
    if rest and not rest.startswith(("-", "+")):
        raise ValueError(f"Unrecognized version tag: {tag!r}")
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def _version_key(tag: str) -> Tuple[int, int, int, int]:
    """
    Sort key for a version tag. The trailing element is 1 for a plain
    release and 0 for a pre-release/build-suffixed tag, so a pre-release
    sorts OLDER than its own release: "v0.2.0-rc1" < "v0.2.0", while
    "v0.3.0-rc1" is still newer than "v0.2.0".
    """
    s = (tag or "").strip().lstrip("vV")
    return (*parse_version(s), 0 if _SUFFIX_RE.match(s) else 1)


def is_newer(latest: str, current: str) -> bool:
    """
    True when the `latest` release tag is a newer version than the
    `current` one (both compared as semver triples; pre-releases sort
    older than their release, see _version_key). Equal tags -> False.
    """
    return _version_key(latest) > _version_key(current)


# ---------------------------------------------------------------------------
# Latest-release lookup
# ---------------------------------------------------------------------------

# In-memory TTL cache for the releases payload (unauthenticated GitHub API
# = 60 req/h per IP, so we cache aggressively - same as llama_release).
_CACHE_TTL_SECONDS = 3600
_cache = {"fetched_at": 0.0, "data": None}
_cache_lock = threading.Lock()


def latest_version(force: bool = False) -> Dict[str, Any]:
    """
    Fetch (or return cached) the latest release of this app from GitHub.

    Returns:
        {
          "version": "0.2.0",           # normalized, no "v" prefix
          "tag": "v0.2.0",              # as published
          "html_url": "https://github.com/<owner>/<repo>/releases/tag/v0.2.0",
          "published_at": "2026-08-27T12:00:00Z",
        }
    Raises ValueError with a friendly message on network/API failure or a
    tag that isn't a version (same convention as llama_release).
    """
    with _cache_lock:
        if not force and _cache["data"] is not None:
            if time.time() - _cache["fetched_at"] < _CACHE_TTL_SECONDS:
                return _cache["data"]

    data = _fetch_latest()
    with _cache_lock:
        _cache["fetched_at"] = time.time()
        _cache["data"] = data
    return data


def _fetch_latest() -> Dict[str, Any]:
    url = f"https://api.github.com/repos/{APP_REPO}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with _urlopen_with(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"No releases found for {APP_REPO} yet - nothing to compare against.")
        raise ValueError(f"GitHub returned an error (HTTP {e.code}) while checking for a new LPM release.")
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise ValueError("Could not verify GitHub's TLS certificate on this machine - check your system CA store.")
        raise ValueError(f"Could not reach GitHub to check for a new LPM release: {reason}")
    except (TimeoutError, OSError) as e:
        raise ValueError(f"Timed out reaching GitHub while checking for a new LPM release: {e}")

    if not isinstance(payload, dict) or not payload.get("tag_name"):
        raise ValueError("Unexpected response from the GitHub release API.")

    tag = str(payload["tag_name"]).strip()
    version = parse_version(tag)  # ValueError with a clear message on bad tags
    return {
        "version": ".".join(str(n) for n in version),
        "tag": tag,
        "html_url": str(payload.get("html_url") or f"https://github.com/{APP_REPO}/releases/tag/{tag}"),
        "published_at": payload.get("published_at"),
    }
