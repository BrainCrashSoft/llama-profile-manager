"""
API-level test for GET /api/system/info (app name + version).

Run with:  pytest tests/test_system_info.py

The version endpoint is read-only, so (unlike
tests/test_profile_import_export.py) no data/ backup is needed.
"""

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import __version__  # noqa: E402
from backend.app import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


def test_system_info_shape():
    r = client.get("/api/system/info")
    assert r.status_code == 200
    data = r.json()
    assert data["app"] == "Llama Profile Manager"
    assert isinstance(data["version"], str) and data["version"]


def test_system_info_matches_backend_version():
    # The whole point of the endpoint: the UI (and the in-app update check)
    # must compare against the exact version the code is running.
    data = client.get("/api/system/info").json()
    assert data["version"] == __version__


def test_open_url_accepts_app_link_hosts():
    # The allowlist gates the bridge that opens links in the real browser;
    # the in-app update check needs github.com release pages in addition to
    # the huggingface.co model pages. webbrowser.open is mocked - the test
    # must not launch a browser.
    with mock.patch("webbrowser.open", return_value=True) as m:
        r = client.post("/api/system/open-url",
                        json={"url": "https://github.com/owner/lpm/releases/tag/v0.2.0"})
        assert r.status_code == 200
        assert client.post("/api/system/open-url",
                           json={"url": "https://huggingface.co/unsloth/Qwen3"}).status_code == 200
    assert m.call_count == 2


def test_open_url_rejects_other_hosts():
    r = client.post("/api/system/open-url", json={"url": "https://example.com/x"})
    assert r.status_code == 400
