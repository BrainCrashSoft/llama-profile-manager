"""
API test for the llama-server (llama.cpp) latest/download endpoints
(backend/app.py /api/llama-server/*).

Run with:  python tests/test_llama_server_api.py

Needs network access to api.github.com / github.com (real ~18 MB download
for the end-to-end section). The real data/settings.json is left updated
on purpose: the end-to-end install registers bXXXX as an EXTRA entry and
must not touch the user's active build.
"""

import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend import llama_server_download as lsd  # noqa: E402
from backend import settings  # noqa: E402
from backend.app import app  # noqa: E402

PASSED = 0


def ok(label: str) -> None:
    global PASSED
    PASSED += 1
    print(f"  PASS  {label}")


def check(cond: bool, label: str) -> None:
    if not cond:
        print(f"  FAIL  {label}")
        sys.exit(1)
    ok(label)


client = TestClient(app)

# ---------------------------------------------------------------------------
print("== /api/llama-server/latest ==")

r = client.get("/api/llama-server/latest")
check(r.status_code == 200, f"GET /latest -> 200 (got {r.status_code})")
data = r.json()
check(data["tag"].startswith("b") and data["build"] > 0, f"tag/build present ({data['tag']})")
check(data.get("asset") and data["asset"].startswith(f"llama-b{data['build']}-bin-"),
      f"asset for this platform: {data.get('asset')}")
check(str(data.get("url", "")).startswith("https://github.com/ggml-org/llama.cpp/releases/download/"),
      "download URL points at github.com")
check(data.get("size", 0) > 0, f"asset size reported ({data.get('size')})")

# The full per-platform variant list (CPU + GPU flavors).
check(isinstance(data.get("variants"), list) and data["variants"],
      f"variants list present ({len(data.get('variants') or [])} choices)")
cpu_v = next((v for v in data["variants"] if v["is_cpu"]), None)
check(cpu_v is not None and cpu_v["name"] == data["asset"], "default asset is the CPU build")
check(all(str(v["url"]).startswith("https://github.com/ggml-org/") for v in data["variants"]),
      "all variant URLs point at github.com")
check(data["variants"][0]["is_cpu"], "CPU variant listed first")

r = client.get("/api/llama-server/latest?force=true")
check(r.status_code == 200 and r.json()["tag"] == data["tag"], "force refresh works")

# ---------------------------------------------------------------------------
print("== /api/llama-server/download rejects bad assets ==")

r = client.post("/api/llama-server/download", json={"asset": "nope.zip"})
check(r.status_code == 400, f"unknown asset -> 400 (got {r.status_code})")
other = next((a["name"] for a in data["assets"]
              if a["name"].startswith(f"llama-b{data['build']}-bin-macos-")), None)
if other and platform.system() != "Darwin":
    r = client.post("/api/llama-server/download", json={"asset": other})
    check(r.status_code == 400, f"asset for another platform -> 400 (got {r.status_code})")

# ---------------------------------------------------------------------------
print("== /api/llama-server/status (fresh config -> build null) ==")

tmp_data = ROOT / "data" / "test-llama-api"
shutil.rmtree(tmp_data, ignore_errors=True)
tmp_data.mkdir(parents=True)
real_settings_file, real_data_dir = settings.SETTINGS_FILE, settings.DATA_DIR
settings.SETTINGS_FILE = tmp_data / "settings.json"
settings.DATA_DIR = tmp_data
try:
    settings.save_settings(dict(settings.DEFAULT_SETTINGS, llama_servers=[]))
    r = client.get("/api/llama-server/status")
    check(r.status_code == 200, f"GET /status -> 200 (got {r.status_code})")
    body = r.json()
    check(body["state"] == "idle", "job state idle")
    check(body["current"]["path"] == "", "current.path empty on fresh config")
    check(body["current"]["build"] is None, "current.build is null on fresh config")
finally:
    settings.SETTINGS_FILE, settings.DATA_DIR = real_settings_file, real_data_dir

# ---------------------------------------------------------------------------
print("== /api/llama-server/download 409 while a job is active ==")

with lsd.installer._lock:
    lsd.installer._status["state"] = "downloading"
try:
    r = client.post("/api/llama-server/download")
    check(r.status_code == 409, f"POST /download -> 409 (got {r.status_code})")
finally:
    with lsd.installer._lock:
        lsd.installer._status["state"] = "idle"

# ---------------------------------------------------------------------------
print("== end-to-end: POST /download, poll /status to done ==")

before = settings.load_settings()
before_active, before_path = before["active_llama_server"], before["llama_server_path"]

r = client.post("/api/llama-server/download")
check(r.status_code == 200, f"POST /download -> 200 (got {r.status_code}: {r.text[:200]})")
check(r.json()["state"] == "downloading", "job started in 'downloading'")

deadline = time.time() + 600
st = r.json()
while st["state"] in ("downloading", "extracting", "registering"):
    if time.time() > deadline:
        check(False, "timed out waiting for done")
    time.sleep(1.5)
    st = client.get("/api/llama-server/status").json()
    print(f"    ... {st['state']} {st['bytes_done']}/{st['bytes_total']}")
check(st["state"] == "done", f"job reached 'done' (got {st['state']} {st['error']})")
check(Path(st["installed_path"]).is_file(), "installed binary exists on disk")

s = settings.load_settings()
entry = next((e for e in s["llama_servers"] if e["path"] == st["installed_path"]), None)
check(entry is not None, "entry registered in settings.json")
check(entry and entry["name"] == st["entry_name"] == st["tag"], f"entry named {st['tag']}")

if before_path:
    check(s["active_llama_server"] == before_active, "user's active build untouched")
    check(s["llama_server_path"] == before_path, "llama_server_path untouched")
else:
    check(s["active_llama_server"] == st["tag"], "auto-activated (was the first build)")

r = client.get("/api/llama-server/status").json()
check(r["current"]["build"] is not None or before_path != "",
      "status reports current identity")

shutil.rmtree(tmp_data, ignore_errors=True)

print(f"\nALL {PASSED} CHECKS PASSED")
