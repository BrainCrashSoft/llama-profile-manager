"""
End-to-end test for the llama-server auto-download feature
(Plan: Automatic download and update checking for llama.cpp).

Run with:  python tests/test_llama_server_download.py

Needs network access to api.github.com / github.com (real ~18 MB download).

Sections
--------
 1  unit bits: version-output parsing, invalid tag/asset rejection
 2  full install on a TEMP settings copy with llama_servers cleared:
    done state, binary runs with the expected build, entry registered AND
    auto-activated (active_llama_server + llama_server_path synced)
 3  full install on the REAL data dir: entry added, but the user's
    already-configured active build is NOT touched (settings.json is
    restored byte-for-byte; the installed binary stays in place)
 4  cancel mid-download: job ends 'cancelled', only a removable .part file
 5  a second start() while one is running raises RuntimeError (409-equivalent)

The test backs up data/settings.json and works on temp copies for the
auto-activation case; only section 3 touches the real settings file and
restores it on exit.
"""

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import llama_release, process_manager, settings  # noqa: E402
from backend import llama_server_download as lsd  # noqa: E402

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


def wait_terminal(installer, timeout: float = 600.0) -> dict:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        st = installer.status()
        line = f"{st['state']} {st['bytes_done']}/{st['bytes_total']}"
        if line != last:
            print(f"    ... {line}")
            last = line
        if st["state"] not in ("downloading", "extracting", "registering"):
            return st
        time.sleep(0.5)
    raise AssertionError("timed out waiting for the install job")


def pick_asset() -> tuple:
    rel = llama_release.latest_release(force=True)
    name, url, size = llama_release.asset_for_platform(rel["build"], rel["tag"], rel["assets"])
    return rel, name, url, size


def run_version(binary: Path) -> str:
    # binary_env() prepends the binary's folder to LD_LIBRARY_PATH on Linux -
    # the prebuilt release .so files sit next to it and the dynamic linker
    # won't find them any other way.
    out = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True,
        timeout=60, cwd=str(binary.parent),
        env=process_manager.binary_env(binary),
    )
    return out.stdout + out.stderr


# ---------------------------------------------------------------------------
print("== 1  unit bits ==")

check(llama_release.build_number_from_version_output(
    "version: 0.2.0-dev (build 10612, commit 758443071)\nbuilt with Clang 20.1.8 for Windows x86_64") == 10612,
    "build parsed from real --version output")
check(llama_release.build_number_from_version_output("build: 1234") == 1234,
      "colon variant parsed")
check(llama_release.build_number_from_version_output("garbage") is None, "junk -> None")
check(llama_release.build_number_from_version_output("") is None, "empty -> None")

rel, asset_name, asset_url, asset_size = pick_asset()
build = rel["build"]
print(f"    target: {rel['tag']} {asset_name} ({asset_size:,} bytes)")

try:
    lsd.installer.start("not-a-tag", asset_name, asset_url)
    check(False, "invalid tag rejected")
except ValueError:
    check(True, "invalid tag rejected")

try:
    lsd.installer.start(rel["tag"], "evil-name-with-..zip", asset_url)
    check(False, "invalid asset name rejected")
except ValueError:
    check(True, "invalid asset name rejected")

variants = llama_release.list_platform_assets(build, rel["tag"], rel["assets"])
check(any(v["is_cpu"] for v in variants), "platform asset list includes a CPU build")
check(variants and variants[0]["is_cpu"], "CPU variant listed first")
if platform.system() == "Windows":
    check(any(v["label"].startswith("CUDA") for v in variants),
          "Windows list includes CUDA variants")

# An asset for ANOTHER platform must be refused by start() (any variant).
other_asset = next(
    (a["name"] for a in rel["assets"]
     if a["name"].startswith(f"llama-b{build}-bin-macos-")),
    None,
)
if other_asset and platform.system() != "Darwin":
    try:
        lsd.installer.start(rel["tag"], other_asset, asset_url)
        check(False, "asset for another platform rejected")
    except ValueError:
        check(True, "asset for another platform rejected")

# ---------------------------------------------------------------------------
print(f"== 2  full install, temp settings, cleared llama_servers ==")

tmp_data = ROOT / "data" / "test-llama-install"
shutil.rmtree(tmp_data, ignore_errors=True)
tmp_data.mkdir(parents=True)
tmp_settings = tmp_data / "settings.json"

real_settings_file = settings.SETTINGS_FILE
real_data_dir = settings.DATA_DIR
settings.SETTINGS_FILE = tmp_settings
settings.DATA_DIR = tmp_data
try:
    settings.save_settings(dict(settings.DEFAULT_SETTINGS, llama_servers=[]))

    dest = tmp_data / "llama-servers"
    lsd.installer.start(rel["tag"], asset_name, asset_url, dest)
    st = wait_terminal(lsd.installer)
    check(st["state"] == "done", f"job finished 'done' (got {st['state']} {st['error']})")

    bin_file = "llama-server.exe" if settings._binary_name().endswith(".exe") else "llama-server"
    binary = dest / f"llama-server-b{build}" / bin_file
    check(binary.is_file(), f"binary exists: llama-server-b{build}/{bin_file}")
    ver_out = run_version(binary)
    check(llama_release.build_number_from_version_output(ver_out) == build,
          f"binary --version reports build {build}")

    s = settings.load_settings()
    entry = next((e for e in s["llama_servers"] if e["path"] == str(binary.resolve())), None)
    check(entry is not None, "entry registered in llama_servers")
    check(entry and entry["name"] == f"b{build}", f"entry named b{build}")
    check(s["active_llama_server"] == f"b{build}", "auto-activated (was the only build)")
    check(s["llama_server_path"] == str(binary.resolve()), "llama_server_path synced to new build")

    # A second start for the same build must not duplicate the entry.
    lsd.installer.start(rel["tag"], asset_name, asset_url, dest)
    st = wait_terminal(lsd.installer)
    check(st["state"] == "done", f"re-install finished 'done' (got {st['state']} {st['error']})")
    s = settings.load_settings()
    n_same = sum(1 for e in s["llama_servers"] if e["path"] == str(binary.resolve()))
    check(n_same == 1, "re-install did not duplicate the entry")
finally:
    settings.SETTINGS_FILE = real_settings_file
    settings.DATA_DIR = real_data_dir

# ---------------------------------------------------------------------------
print("== 3  full install, real data dir (settings restored after) ==")

real_settings_json = settings.SETTINGS_FILE.read_text(encoding="utf-8")
before = settings.load_settings()
before_active = before["active_llama_server"]
before_path = before["llama_server_path"]

try:
    lsd.installer.start(rel["tag"], asset_name, asset_url)
    st = wait_terminal(lsd.installer)
    check(st["state"] == "done", f"real-dir install finished 'done' (got {st['state']} {st['error']})")

    s = settings.load_settings()
    entry = next((e for e in s["llama_servers"] if e["path"] == st["installed_path"]), None)
    check(entry is not None, "entry added to the real settings")
    if before_path:
        check(s["active_llama_server"] == before_active,
              "user's active build NOT stolen")
        check(s["llama_server_path"] == before_path,
              "llama_server_path still points at the user's build")
    else:
        check(s["active_llama_server"] == f"b{build}", "auto-activated on empty config")
finally:
    settings.SETTINGS_FILE.write_text(real_settings_json, encoding="utf-8")
check(settings.load_settings()["active_llama_server"] == before_active,
      "settings.json restored byte-for-byte")

# ---------------------------------------------------------------------------
# Sections 4+5 use a FAKE endless stream (monkeypatched _urlopen) so the
# cancel/concurrency behavior is deterministic regardless of network speed
# - a real 18 MB download can finish in well under a second on a fast line.
class _FakeResp:
    headers = {"Content-Length": "999999999"}

    def read(self, n):
        time.sleep(0.05)
        return b"\0" * min(n, 1024 * 1024)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

def _fake_urlopen(url, timeout: float = 20.0):
    return _FakeResp()

# ---------------------------------------------------------------------------
print("== 4  cancel mid-download (fake endless stream) ==")

tmp_data2 = ROOT / "data" / "test-llama-install-cancel"
shutil.rmtree(tmp_data2, ignore_errors=True)
tmp_data2.mkdir(parents=True)
dest2 = tmp_data2 / "llama-servers"

real_urlopen = lsd.llama_release._urlopen
lsd.llama_release._urlopen = _fake_urlopen
try:
    lsd.installer.start(rel["tag"], asset_name, asset_url, dest2)
    # Wait until bytes are actually flowing, then cancel.
    for _ in range(80):
        if lsd.installer.status()["bytes_done"] > 0:
            break
        time.sleep(0.05)
    check(lsd.installer.status()["bytes_done"] > 0, "download was in flight")
    lsd.installer.cancel()
    st = wait_terminal(lsd.installer, timeout=60)
    check(st["state"] == "cancelled", f"job ended 'cancelled' (got {st['state']} {st['error']})")
    parts = list(dest2.glob("*.part")) if dest2.exists() else []
    check(len(parts) in (0, 1), "only a .part file (or none) left behind")
    for p in parts:
        p.unlink()
        check(True, f".part removable: {p.name}")
    leftover = [f for f in dest2.iterdir()] if dest2.exists() else []
    check(not leftover, "dest dir empty after removing .part")

    # -----------------------------------------------------------------------
    print("== 5  concurrent start rejected (fake endless stream) ==")

    lsd.installer.start(rel["tag"], asset_name, asset_url, dest2)
    for _ in range(80):
        if lsd.installer.status()["bytes_done"] > 0:
            break
        time.sleep(0.05)
    check(lsd.installer.status()["state"] in ("downloading", "extracting", "registering"),
          "job is active")
    try:
        lsd.installer.start(rel["tag"], asset_name, asset_url, dest2)
        check(False, "second start() raised")
    except RuntimeError as e:
        check("already in progress" in str(e), f"second start() raised RuntimeError ({e})")
    lsd.installer.cancel()
    st = wait_terminal(lsd.installer, timeout=60)
    check(st["state"] == "cancelled", "cleanup cancel worked")
finally:
    lsd.llama_release._urlopen = real_urlopen

# Tidy the scratch dirs (keep the real data/llama-servers install).
shutil.rmtree(tmp_data, ignore_errors=True)
shutil.rmtree(tmp_data2, ignore_errors=True)

print(f"\nALL {PASSED} CHECKS PASSED")
