"""
Tests for backend.app_release: the in-app "new LPM version" check
(GitHub /releases/latest) and its semver comparison helpers.

Run with:  pytest tests/test_app_release.py

Fully offline: the network layer (app_release._urlopen_with) is mocked,
same style as tests/test_hf_client.py. Also covers the TestClient level:
GET /api/app/latest (200 with the payload; 502 when offline).
"""

import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import app_release  # noqa: E402
from backend.app import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


class _FakeResponse:
    """Stand-in for the urlopen() context manager (a JSON document)."""

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


RELEASE_PAYLOAD = {
    "tag_name": "v0.2.0",
    "html_url": "https://github.com/owner/lpm/releases/tag/v0.2.0",
    "published_at": "2026-09-01T10:00:00Z",
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Every test starts with an empty release cache (TTL would otherwise
    leak results between tests)."""
    with app_release._cache_lock:
        app_release._cache["fetched_at"] = 0.0
        app_release._cache["data"] = None
    yield


def _mock_urlopen(payload):
    return mock.patch.object(app_release, "_urlopen_with",
                             return_value=_FakeResponse(payload))


# ---------------------------------------------------------------------------
# latest_version(): payload parsing, cache, offline
# ---------------------------------------------------------------------------

def test_latest_version_parses_release_payload():
    with _mock_urlopen(RELEASE_PAYLOAD):
        info = app_release.latest_version()
    assert info["version"] == "0.2.0"       # normalized, no "v" prefix
    assert info["tag"] == "v0.2.0"
    assert info["html_url"] == "https://github.com/owner/lpm/releases/tag/v0.2.0"
    assert info["published_at"] == "2026-09-01T10:00:00Z"


def test_latest_version_hits_cache_on_second_call():
    with _mock_urlopen(RELEASE_PAYLOAD) as m:
        first = app_release.latest_version()
        second = app_release.latest_version()
    assert first == second
    assert m.call_count == 1  # second call served from the TTL cache


def test_latest_version_force_bypasses_cache():
    with _mock_urlopen(RELEASE_PAYLOAD) as m:
        app_release.latest_version()
        app_release.latest_version(force=True)
    assert m.call_count == 2


def test_latest_version_offline_raises_valueerror():
    with mock.patch.object(app_release, "_urlopen_with",
                           side_effect=urllib.error.URLError("no network")):
        with pytest.raises(ValueError, match="Could not reach GitHub"):
            app_release.latest_version()


def test_latest_version_http_404_raises_valueerror():
    # The placeholder repo (no releases yet) lands here: the Settings badge
    # stays hidden and the app keeps working.
    err = urllib.error.HTTPError(
        "https://api.github.com/repos/lpm/CHANGE_ME/releases/latest",
        404, "Not Found", hdrs=None, fp=None)
    with mock.patch.object(app_release, "_urlopen_with", side_effect=err):
        with pytest.raises(ValueError, match="No releases found"):
            app_release.latest_version()


def test_latest_version_non_version_tag_raises_valueerror():
    with _mock_urlopen({"tag_name": "totally-not-a-version", "html_url": "x"}):
        with pytest.raises(ValueError, match="Unrecognized version tag"):
            app_release.latest_version()


# ---------------------------------------------------------------------------
# Semver parsing / comparison (parse_version / is_newer)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    ("v0.2.0", (0, 2, 0)),
    ("0.2.0", (0, 2, 0)),          # unprefixed parses identically
    ("V0.2.0", (0, 2, 0)),         # uppercase prefix tolerated
    ("v0.2", (0, 2, 0)),           # missing patch segment -> 0
    ("v0", (0, 0, 0)),             # major only
    ("v0.2.0-rc1", (0, 2, 0)),     # pre-release suffix ignored here
])
def test_parse_version(tag, expected):
    assert app_release.parse_version(tag) == expected


@pytest.mark.parametrize("tag", [
    "banana", "b10645", "", "   ", "1.2.3.4", "0.1.0xyz", "v", "v1.",
])
def test_parse_version_rejects_garbage(tag):
    with pytest.raises(ValueError):
        app_release.parse_version(tag)


@pytest.mark.parametrize("latest,current,expected", [
    ("0.2.0", "0.1.0", True),       # simple feature bump
    ("0.1.0", "0.1.0", False),      # equal
    ("0.10.0", "0.9.0", True),      # numeric (not lexicographic) compare
    ("v0.2.0", "0.2.0", False),     # v-prefixed == unprefixed
    ("0.2.0", "v0.2.0", False),
    ("v1.0.0", "v0.99.99", True),   # major beats a long minor
    ("v0.2.0-rc1", "v0.2.0", False),  # pre-release sorts older than its release
    ("v0.3.0-rc1", "v0.2.0", True),   # ...but still newer than the prior one
])
def test_is_newer_table(latest, current, expected):
    assert app_release.is_newer(latest, current) is expected


# ---------------------------------------------------------------------------
# GET /api/app/latest (TestClient, network mocked)
# ---------------------------------------------------------------------------

def test_api_app_latest_returns_payload():
    with _mock_urlopen(RELEASE_PAYLOAD):
        r = client.get("/api/app/latest")
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == "0.2.0"
    assert data["tag"] == "v0.2.0"
    assert data["html_url"].startswith("https://github.com/")


def test_api_app_latest_offline_returns_502():
    with mock.patch.object(app_release, "_urlopen_with",
                           side_effect=urllib.error.URLError("no network")):
        r = client.get("/api/app/latest")
    assert r.status_code == 502
    assert "Could not reach GitHub" in r.json()["detail"]
