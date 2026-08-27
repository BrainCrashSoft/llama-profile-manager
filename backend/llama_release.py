"""
llama_release.py

Discovers the latest llama.cpp build on GitHub (no auth) and maps it to the
downloadable asset for the current OS/architecture. Stdlib urllib only
(same precedent as hf_client.py).

Release-channel note (confirmed against the live API 2026-08-24): the
per-build binaries are published as PRERELEASES with tags like "b10612";
the stable /releases/latest tag (e.g. "v0.2.0") carries no binaries. So
latest_release() scans the releases list and picks the newest bXXXX tag.

All release-asset name matching lives in ASSET_NAME_RE so a change in
upstream naming is a one-line fix here.
"""

import json
import platform
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

GITHUB_RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15"
DOWNLOAD_BASE = "https://github.com/ggml-org/llama.cpp/releases/download"
USER_AGENT = "LlamaModelManager/1.0 (+local desktop app; https://github.com/ggml-org/llama.cpp)"

# Build release tags look like "b10612" (stable tags like "v0.2.0" are
# skipped - they ship no platform binaries).
_TAG_BUILD_RE = re.compile(r"^b(\d+)$")

# Downloadable per-platform asset for a build. Shape:
#   llama-b{build}-bin-{os}[{-variant[-ver]}]-{arch}.{ext}
# e.g.:
#   llama-b10612-bin-win-cpu-x64.zip            (Windows CPU)
#   llama-b10612-bin-win-cuda-12.4-x64.zip      (Windows CUDA 12.4)
#   llama-b10612-bin-ubuntu-vulkan-arm64.tar.gz (Linux Vulkan)
#   llama-b10612-bin-ubuntu-openvino-2026.3-x64.tar.gz
#   llama-b10612-bin-macos-arm64.tar.gz         (no variant = CPU)
# This one regex is the single place the naming convention is encoded;
# adjust here if upstream renames (confirmed against the live API 2026-08-24).
PLATFORM_ASSET_RE = re.compile(
    r"^llama-b(?P<build>\d+)-bin-(?P<os>win|macos|ubuntu)"
    r"(?P<mid>(?:-[A-Za-z0-9][A-Za-z0-9.]*)*?)"
    r"-(?P<arch>x64|arm64|s390x)"
    r"\.(?P<ext>zip|tar\.gz)$"
)

# Display labels for the variant middle section ("-cuda-12.4" -> "CUDA 12.4").
_VARIANT_LABELS = {
    "": "CPU", "cpu": "CPU", "vulkan": "Vulkan", "sycl": "SYCL",
    "sycl-fp16": "SYCL (FP16)", "sycl-fp32": "SYCL (FP32)",
    "opencl-adreno": "OpenCL (Adreno)",
}
_VARIANT_STYLES = {"cuda": "CUDA", "openvino": "OpenVINO", "rocm": "ROCm", "opencl": "OpenCL"}
# Variants whose trailing token is a version number (vs. a flavor suffix).
_VARIANT_VERSIONED = {"cuda", "openvino", "rocm"}


def variant_label(mid: str) -> str:
    """Human label for an asset variant middle section ("", "-cuda-12.4", ...)."""
    key = (mid or "").lstrip("-")
    if key in _VARIANT_LABELS:
        return _VARIANT_LABELS[key]
    first, _, rest = key.partition("-")
    if first in _VARIANT_STYLES:
        base = _VARIANT_STYLES[first]
        if rest:
            return f"{base} {rest}" if first in _VARIANT_VERSIONED else f"{base} ({rest})"
        return base
    return key.replace("-", " ").title()

# llama-server --version prints e.g.
#   "version: 0.2.0-dev (build 10612, commit 758443071)"
# (also tolerate an older "build: 10612" style with a colon).
_VERSION_BUILD_RE = re.compile(r"\bbuild[:\s]+(\d+)", re.IGNORECASE)

# In-memory TTL cache for the releases payload (unauthenticated GitHub API
# = 60 req/h per IP, so we cache aggressively).
_CACHE_TTL_SECONDS = 3600
_cache = {"fetched_at": 0.0, "data": None}
_cache_lock = threading.Lock()

# One shared SSL context, lazily built. On machines where the OS trust store
# is missing/broken (CERTIFICATE_VERIFY_FAILED) we fall back to the certifi
# bundle once it proves available, then keep using it.
_ssl_lock = threading.Lock()
_ssl_ctx: Optional[ssl.SSLContext] = None


def _ssl_context() -> ssl.SSLContext:
    global _ssl_ctx
    with _ssl_lock:
        if _ssl_ctx is None:
            _ssl_ctx = ssl.create_default_context()
        return _ssl_ctx


def _use_certifi_context() -> bool:
    """Rebuild the shared context on top of certifi's CA bundle; False if
    certifi isn't installed (in which case the original error stands)."""
    global _ssl_ctx
    try:
        import certifi
    except ImportError:
        return False
    with _ssl_lock:
        _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    return True


def _urlopen(url: str, timeout: float = 20.0):
    """urlopen with the shared (certifi-fallback) SSL context."""
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
# Release discovery
# ---------------------------------------------------------------------------

def latest_release(force: bool = False) -> Dict[str, Any]:
    """
    Fetch (or return cached) info about the latest llama.cpp build release.

    Returns:
        {
          "tag": "b10612",
          "build": 10612,
          "published_at": "2026-08-24T13:41:50Z",
          "assets": [{"name": ..., "size": ...}, ...],
        }
    Raises ValueError with a friendly message on network/API failure.
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
    req = urllib.request.Request(
        GITHUB_RELEASES_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    try:
        with _urlopen_with(req) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ValueError(f"GitHub returned an error (HTTP {e.code}) while checking for the latest llama.cpp build.")
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise ValueError("Could not verify GitHub's TLS certificate on this machine - check your system CA store.")
        raise ValueError(f"Could not reach GitHub to check for the latest llama.cpp build: {reason}")
    except (TimeoutError, OSError) as e:
        raise ValueError(f"Timed out reaching GitHub while checking for the latest llama.cpp build: {e}")

    if not isinstance(payload, list):
        raise ValueError("Unexpected response from the GitHub release API.")

    # The newest bXXXX prerelease wins; versioned tags (v0.2.0) carry no binaries.
    best = None
    for rel in payload:
        if not isinstance(rel, dict):
            continue
        m = _TAG_BUILD_RE.match(str(rel.get("tag_name") or "").strip())
        if not m:
            continue
        build = int(m.group(1))
        if best is None or build > best[0]:
            best = (build, rel)

    if best is None:
        raise ValueError("No bXXXX build release found on GitHub right now - try again later.")
    build, rel = best
    assets = [
        {"name": str(a.get("name", "")), "size": int(a.get("size") or 0)}
        for a in (rel.get("assets") or [])
        if isinstance(a, dict) and a.get("name")
    ]
    return {
        "tag": f"b{build}",
        "build": build,
        "published_at": rel.get("published_at"),
        "assets": assets,
    }


# ---------------------------------------------------------------------------
# Platform asset selection
# ---------------------------------------------------------------------------

def platform_tokens() -> Tuple[Optional[str], Optional[str]]:
    """
    Map platform.system()/platform.machine() to the (os, arch) tokens used
    in llama.cpp release asset names. (None, None) = no official build.
    """
    system = platform.system()
    machine = (platform.machine() or "").lower()

    if system == "Windows":
        if machine in ("amd64", "x86_64", "x64"):
            return "win", "x64"
        if machine == "arm64":
            return "win", "arm64"
        return None, None

    if system == "Darwin":
        if machine in ("x86_64", "amd64"):
            return "macos", "x64"
        if machine in ("arm64", "aarch64"):
            return "macos", "arm64"
        return None, None

    if system == "Linux":
        if machine in ("x86_64", "amd64"):
            return "ubuntu", "x64"
        # No plain linux-arm64 asset exists (only GPU variants).
        return None, None

    return None, None


def parse_asset(name: str) -> Optional[Dict[str, str]]:
    """Parse a llama.cpp release asset name; None when it doesn't match."""
    m = PLATFORM_ASSET_RE.match((name or "").strip())
    if not m:
        return None
    return {
        "build": m.group("build"),
        "os": m.group("os"),
        "mid": m.group("mid"),
        "arch": m.group("arch"),
        "ext": m.group("ext"),
    }


def list_platform_assets(build: int, tag: str,
                         assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    All downloadable assets of `build` that match THIS machine's OS+arch
    (CPU and every GPU flavor: CUDA, Vulkan, OpenVINO, ROCm, SYCL, ...).

    Returns [{"name", "url", "size", "label", "arch", "is_cpu"}], CPU first.
    Raises ValueError with a friendly message on unsupported platforms.
    """
    os_token, arch_token = platform_tokens()
    if os_token is None or arch_token is None:
        raise ValueError(
            f"Unsupported platform: {platform.system()} ({platform.machine()}). "
            "There is no official llama.cpp build for it yet - "
            "install llama-server manually under Settings → llama-server versions."
        )

    out: List[Dict[str, Any]] = []
    for a in assets or []:
        if not isinstance(a, dict):
            continue
        info = parse_asset(str(a.get("name", "")))
        if not info or info["build"] != str(build):
            continue
        if info["os"] != os_token or info["arch"] != arch_token:
            continue
        out.append({
            "name": a["name"],
            "url": f"{DOWNLOAD_BASE}/{tag}/{a['name']}",
            "size": int(a.get("size") or 0),
            "label": variant_label(info["mid"]),
            "arch": info["arch"],
            "is_cpu": info["mid"] in ("", "-cpu"),
        })
    out.sort(key=lambda v: (not v["is_cpu"], v["label"].lower(), v["name"]))
    return out


def resolve_platform_asset(name: Optional[str], build: int, tag: str,
                           assets: List[Dict[str, Any]]) -> Tuple[str, str, int]:
    """
    Resolve the asset to download: `name=None` -> this platform's CPU build;
    otherwise an exact asset name from the release, which must be a build
    for THIS machine's OS+arch (a macOS asset is refused on Windows, etc.).

    Returns (asset_name, download_url, size_bytes); ValueError otherwise.
    """
    variants = list_platform_assets(build, tag, assets)
    if name:
        v = next((v for v in variants if v["name"] == name), None)
        if v is None:
            raise ValueError(
                f"'{name}' is not a {tag} build for this platform. "
                "Pick one of the listed build variants."
            )
        return v["name"], v["url"], v["size"]
    cpu = next((v for v in variants if v["is_cpu"]), None)
    if cpu is None:
        if variants:  # no plain CPU asset, but GPU builds exist
            cpu = variants[0]
        else:
            raise ValueError(
                f"No llama.cpp release asset for this platform. "
                "You can still add a manually installed llama-server under Settings."
            )
    return cpu["name"], cpu["url"], cpu["size"]


def asset_for_platform(build: Optional[int] = None,
                       tag: Optional[str] = None,
                       assets: Optional[List[Dict[str, Any]]] = None
                       ) -> Tuple[str, str, int]:
    """
    Pick the downloadable CPU asset for the current OS/arch of the given
    (or latest) release. Returns (asset_name, download_url, size_bytes).
    """
    if build is None or tag is None or assets is None:
        rel = latest_release()
        build = build or rel["build"]
        tag = tag or rel["tag"]
        assets = assets if assets is not None else rel["assets"]
    return resolve_platform_asset(None, build, tag, assets)


def build_number_from_version_output(text: str) -> Optional[int]:
    """
    Parse the build number out of `llama-server --version` output, e.g.
    "version: 0.2.0-dev (build 10612, commit 758443071)". Returns None
    when not found.
    """
    if not text:
        return None
    m = _VERSION_BUILD_RE.search(text)
    return int(m.group(1)) if m else None
