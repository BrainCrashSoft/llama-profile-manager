"""
API-level tests for profile import (single + all) and export-all.

Run with:  pytest tests/test_profile_import_export.py

Strategy (same pattern as tests/test_benchmark.py)
--------------------------------------------------
The real data/profiles.json (and data/benchmarks.json) are backed up BEFORE
the app is imported - importing backend.app runs
benchmark_runner.recover_interrupted_runs(), which must only ever see test
data - then every test works on a clean profile store, and the originals are
restored byte-for-byte when the module's tests finish (atexit is a belt-and-
braces backup if the run is interrupted).

Coverage
--------
 * GET  /api/profiles/export-all - wrapper shape, canonical entries
 * GET  export-all -> POST /api/profiles/import-all round-trip:
   params/custom_flags/notes preserved, ids regenerated
 * per-model name de-duplication on batch import (" (2)", " (3)", …)
 * model-folder params (--mmproj) re-rooted to the picked model_path
   (_store_model_folder_paths behavior, both import endpoints)
 * malformed payloads -> HTTP 400 with a clear message
 * per-item errors reported in-band; the rest of the batch still imports
"""

import atexit
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Back up the user's real data BEFORE importing the app
# ---------------------------------------------------------------------------
_BACKUP_DIR = Path(tempfile.mkdtemp(prefix="lpm-import-test-backup-"))
_BACKED_UP = []
for _name in ("profiles.json", "benchmarks.json"):
    _f = DATA / _name
    if _f.exists():
        shutil.copy2(_f, _BACKUP_DIR / _name)
        _BACKED_UP.append(_name)


def _restore_backup() -> None:
    if not _BACKUP_DIR.exists():
        return
    for _name in _BACKED_UP:
        shutil.copy2(_BACKUP_DIR / _name, DATA / _name)
    shutil.rmtree(_BACKUP_DIR, ignore_errors=True)
    _BACKUP_DIR.exists  # no-op; keep _BACKUP_DIR referenced after rmtree


atexit.register(_restore_backup)  # if the run dies before fixture teardown


from backend import profiles  # noqa: E402
from backend.app import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _restore_data_files():
    yield
    _restore_backup()


@pytest.fixture(autouse=True)
def _clean_profiles():
    (DATA / "profiles.json").write_text("[]", encoding="utf-8")
    yield
    (DATA / "profiles.json").write_text("[]", encoding="utf-8")


@pytest.fixture()
def model_folder(tmp_path: Path):
    """Two temp model folders, each with a model file and an mmproj file of
    the SAME bare name - the 'same model files, different folders' case."""
    folders = []
    for i, letter in enumerate(("A", "B")):
        folder = tmp_path / f"model-{letter}"
        folder.mkdir()
        (folder / f"model-{letter}.gguf").write_bytes(b"")
        (folder / "mmproj-x.gguf").write_bytes(b"")
        folders.append(folder)
    return folders


def make_profile(model_id, model_path, name, params=None, custom_flags="", notes=""):
    r = client.post("/api/profiles", json={
        "model_id": model_id, "model_path": model_path, "name": name,
        "params": params or {}, "custom_flags": custom_flags, "notes": notes,
    })
    assert r.status_code == 200, r.text
    return r.json()


def export_all():
    r = client.get("/api/profiles/export-all")
    assert r.status_code == 200, r.text
    return r.json()


def wrap_for_import(exported_profiles):
    """The client-side step the UI does: each exported (canonical) profile
    becomes one {model_id, model_path, data} item re-rooted at the target."""
    return [
        {"model_id": p["model_id"], "model_path": p["model_path"], "data": p}
        for p in exported_profiles
    ]


def import_all(items):
    return client.post("/api/profiles/import-all", json={"profiles": items})


# ---------------------------------------------------------------------------
# Export all
# ---------------------------------------------------------------------------

def test_export_all_shape():
    make_profile("o1::r1::Q1", "/m/1.gguf", "P1", {"ctx_size": 1024}, notes="n1")
    make_profile("o2::r2::Q2", "/m/2.gguf", "P2", {}, custom_flags="--flash-attn")

    body = export_all()

    assert set(body) == {"app", "version", "exported_at", "profiles"}
    assert body["app"] == "Llama Profile Manager"
    assert body["version"] == 1
    # exported_at is ISO-8601
    datetime.fromisoformat(body["exported_at"])
    assert len(body["profiles"]) == 2
    for p in body["profiles"]:
        assert "id" not in p and "last_used_at" not in p      # canonical shape
        assert {"model_id", "model_path", "name", "params",
                "custom_flags", "notes"} <= set(p)


def test_export_all_empty_collection():
    body = export_all()
    assert body["profiles"] == []
    assert body["version"] == 1


# ---------------------------------------------------------------------------
# Round-trip: export-all -> import-all
# ---------------------------------------------------------------------------

def test_roundtrip_preserves_data_and_regenerates_ids():
    originals = [
        make_profile("o1::r1::Q1", "/m/1.gguf", "RT1",
                     {"ctx_size": 8192, "n_gpu_layers": "all"}, "--fa on", "keep me"),
        make_profile("o2::r2::Q2", "/m/2.gguf", "RT2", {"temp": 0.7}, "", "n2"),
        make_profile("o3::r3::Q3", "/m/3.gguf", "RT3", {}, "  --x  ", "  spaced  "),
    ]
    old_ids = {p["id"] for p in originals}

    exported = export_all()
    assert len(exported["profiles"]) == 3

    # wipe, then import back (same-machine restore)
    (DATA / "profiles.json").write_text("[]", encoding="utf-8")
    r = import_all(wrap_for_import(exported["profiles"]))
    assert r.status_code == 200, r.text
    res = r.json()
    assert len(res["imported"]) == 3 and res["errors"] == []

    by_name = {p["name"]: p for p in client.get("/api/profiles").json()}
    assert set(by_name) == {"RT1", "RT2", "RT3"}
    for orig in originals:
        new = by_name[orig["name"]]
        assert new["params"] == orig["params"]
        assert new["custom_flags"] == orig["custom_flags"]
        assert new["notes"] == orig["notes"]
        assert new["model_id"] == orig["model_id"]
        assert new["model_path"] == orig["model_path"]
        assert new["id"] not in old_ids          # fresh UUIDs
        assert new["id"] != orig["id"]
    # and the ids are unique
    assert len({p["id"] for p in by_name.values()}) == 3


def test_import_all_empty_list_is_a_noop():
    make_profile("o1::r1::Q1", "/m/1.gguf", "S1", {"a": 1}, "cf", "nt")
    r = import_all([])
    assert r.status_code == 200, r.text
    assert r.json() == {"imported": [], "errors": []}
    assert [p["name"] for p in client.get("/api/profiles").json()] == ["S1"]


# ---------------------------------------------------------------------------
# Name de-duplication (per model, by create_profile/_unique_name)
# ---------------------------------------------------------------------------

def test_batch_name_dedup_per_model():
    make_profile("mA::r::Q", "/mA.gguf", "Dup", {"x": 1})      # pre-existing

    r = import_all([
        {"model_id": "mA::r::Q", "model_path": "/mA.gguf", "data": {"name": "Dup", "params": {}}},
        {"model_id": "mA::r::Q", "model_path": "/mA.gguf", "data": {"name": "Dup", "params": {}}},
        {"model_id": "mB::r::Q", "model_path": "/mB.gguf", "data": {"name": "Dup", "params": {}}},
    ])
    assert r.status_code == 200, r.text
    assert len(r.json()["imported"]) == 3 and r.json()["errors"] == []

    by_model = {}
    for p in client.get("/api/profiles").json():
        by_model.setdefault(p["model_id"], set()).add(p["name"])
    # model A: pre-existing "Dup" + two imports -> auto-suffixed per model
    # (the list endpoint sorts MRU, so compare as a set)
    assert by_model["mA::r::Q"] == {"Dup", "Dup (2)", "Dup (3)"}
    # model B: same name is fine - uniqueness is per model
    assert by_model["mB::r::Q"] == {"Dup"}


# ---------------------------------------------------------------------------
# Model-folder param re-rooting (_store_model_folder_paths via the endpoints)
# ---------------------------------------------------------------------------

def test_single_import_reroots_mmproj_to_target_model_folder(model_folder):
    folder_a, folder_b = model_folder
    # the file stores a BARE projector name (as the editor shows it)
    data = {"name": "WithProj", "model_id": "oA::r::Q",
            "model_path": str(folder_a / "model-A.gguf"),
            "params": {"mmproj": "mmproj-x.gguf", "ctx_size": 2048}}
    r = client.post("/api/profiles/import", json={
        "model_id": "oB::r::Q", "model_path": str(folder_b / "model-B.gguf"), "data": data,
    })
    assert r.status_code == 200, r.text
    stored = r.json()["params"]
    assert stored["ctx_size"] == 2048
    assert Path(stored["mmproj"]) == folder_b / "mmproj-x.gguf", stored["mmproj"]


def test_batch_import_reroots_mmproj_to_each_target(model_folder):
    folder_a, folder_b = model_folder
    r = import_all([
        {"model_id": "oB::r::Q", "model_path": str(folder_b / "model-B.gguf"),
         "data": {"name": "P1", "params": {"mmproj": "mmproj-x.gguf"}}},
        {"model_id": "oA::r::Q", "model_path": str(folder_a / "model-A.gguf"),
         "data": {"name": "P2", "params": {"mmproj": "mmproj-x.gguf"}}},
    ])
    assert r.status_code == 200, r.text
    stored = {p["name"]: p for p in client.get("/api/profiles").json()}
    assert Path(stored["P1"]["params"]["mmproj"]) == folder_b / "mmproj-x.gguf"
    assert Path(stored["P2"]["params"]["mmproj"]) == folder_a / "mmproj-x.gguf"


def test_full_path_mmproj_left_untouched(model_folder):
    folder_a, folder_b = model_folder
    # a value that already contains a path separator is not re-rooted - the
    # command builder resolves it at launch (foreign paths stay foreign)
    foreign = str(folder_a / "mmproj-x.gguf")
    r = client.post("/api/profiles/import", json={
        "model_id": "oB::r::Q", "model_path": str(folder_b / "model-B.gguf"),
        "data": {"name": "F", "params": {"mmproj": foreign}},
    })
    assert r.status_code == 200, r.text
    assert r.json()["params"]["mmproj"] == foreign


# ---------------------------------------------------------------------------
# Malformed payloads -> 400
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"profiles": "nope"},                                   # not a list
    {"profiles": {"profiles": []}},                         # not a list
    [1, 2, 3],                                              # top-level array
    {"nope": []},                                           # missing key
    {"profiles": [{}]},                                     # item missing everything
    {"profiles": [{"model_id": "a", "model_path": "b"}]},   # item missing data
    {"profiles": [{"model_id": "a", "model_path": "b", "data": "not-a-dict"}]},
    {"profiles": ["just a string"]},
])
def test_malformed_payload_rejected(payload):
    r = client.post("/api/profiles/import-all", json=payload)
    assert r.status_code == 400, (payload, r.status_code, r.text)
    assert "Invalid import file" in r.json()["detail"]


def test_non_json_body_rejected():
    r = client.post("/api/profiles/import-all", content=b"this is not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 400
    assert "Invalid import file" in r.json()["detail"]


def test_malformed_payload_creates_nothing():
    before = len(client.get("/api/profiles").json())
    client.post("/api/profiles/import-all", json={"profiles": "nope"})
    client.post("/api/profiles/import-all", json={"profiles": [{"model_id": "a"}]})
    assert len(client.get("/api/profiles").json()) == before


# ---------------------------------------------------------------------------
# Per-item errors: in-band, never abort the batch
# ---------------------------------------------------------------------------

def test_per_item_error_does_not_abort_batch(monkeypatch):
    orig = profiles.import_profile

    def flaky(model_id, model_path, data):
        if isinstance(data, dict) and data.get("name") == "Bad":
            raise RuntimeError("simulated store failure")
        return orig(model_id, model_path, data)

    monkeypatch.setattr(profiles, "import_profile", flaky)

    r = client.post("/api/profiles/import-all", json={"profiles": [
        {"model_id": "o1::r1::Q", "model_path": "/1.gguf", "data": {"name": "Good1", "params": {}}},
        {"model_id": "o2::r2::Q", "model_path": "/2.gguf", "data": {"name": "Bad", "params": {}}},
        {"model_id": "o3::r3::Q", "model_path": "/3.gguf", "data": {"name": "Good2"}},
    ]})
    assert r.status_code == 200, r.text
    res = r.json()
    assert len(res["imported"]) == 2
    assert len(res["errors"]) == 1
    err = res["errors"][0]
    assert err["index"] == 1
    assert "Bad" in err["name"]
    assert "simulated store failure" in err["error"]
    names = {p["name"] for p in client.get("/api/profiles").json()}
    assert names == {"Good1", "Good2"}          # the good ones survived


def test_every_item_failing_is_still_a_200_with_errors(monkeypatch):
    monkeypatch.setattr(profiles, "import_profile",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    r = client.post("/api/profiles/import-all", json={"profiles": [
        {"model_id": "o::r::Q", "model_path": "/x.gguf", "data": {"name": "Z"}},
    ]})
    assert r.status_code == 200
    res = r.json()
    assert res["imported"] == [] and len(res["errors"]) == 1
    assert client.get("/api/profiles").json() == []
