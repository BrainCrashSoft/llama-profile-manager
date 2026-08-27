"""
API-level tests for POST /api/benchmarks/compare - "were these two benchmark
runs done with the same parameters, and if not, what differs?".

Run with:  pytest tests/test_benchmark_compare.py

Strategy (same pattern as tests/test_profile_import_export.py)
--------------------------------------------------
The real data/profiles.json (and data/benchmarks.json) are backed up BEFORE
importing backend.app - the import triggers
benchmark_runner.recover_interrupted_runs(), which must only ever see test
data - then every test works on a clean benchmark store, and the originals
are restored byte-for-byte when the module's tests finish (atexit is a
belt-and-braces backup if the run is interrupted).

Records are created directly via bench_store.create_record(...) with
params_hash=bench_store.params_hash(snapshot), the same way the runner
creates them - no fake llama-server needed.

Coverage
--------
 * identical snapshots -> same: true, both benchmarkable objects equal
 * differing params.n_ctx (4096 vs 8192) -> same: false, values visible in
   the A/B benchmarkable objects
 * normalization (the same rules the staleness badge hashes by):
     4096 vs 4096.0, custom_flags "" vs key absent, a false param vs absent,
     whitespace-only custom_flags differences -> all same: true
 * differing model_path -> same: false
 * 404 (each id missing, named in the detail) and 400 (a == b)
 * full per-side field shape (step 1 of the plan)
"""

import atexit
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Back up the user's real data BEFORE importing the app
# ---------------------------------------------------------------------------
_BACKUP_DIR = Path(tempfile.mkdtemp(prefix="lpm-compare-test-backup-"))
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


from backend import benchmarks as bench_store  # noqa: E402
from backend.app import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _restore_data_files():
    yield
    _restore_backup()


@pytest.fixture(autouse=True)
def _clean_benchmarks():
    (DATA / "benchmarks.json").write_text("[]", encoding="utf-8")
    yield
    (DATA / "benchmarks.json").write_text("[]", encoding="utf-8")


def make_record(name="Rec", model_path="/m/model.gguf", params=None,
                custom_flags="", server_version="b123", status="completed"):
    """Create a finished benchmark record the way the runner would.
    custom_flags=None omits the key from the snapshot entirely."""
    snapshot = {
        "name": name,
        "model_path": model_path,
        "params": params or {},
    }
    if custom_flags is not None:
        snapshot["custom_flags"] = custom_flags
    rec = bench_store.create_record(
        profile_id=None,
        profile_name=name,
        snapshot=snapshot,
        model_path=model_path,
        params_hash=bench_store.params_hash(snapshot),
        started_at=1_700_000_000.0,
        server_version=server_version,
    )
    if status != "running":
        # update_record returns the stored (updated) record; re-fetch so the
        # caller sees the final state, not the running stub from create_record
        rec = bench_store.update_record(
            rec["id"], status=status, prefill_tps=800.0, generation_tps=30.0,
            timestamp=1_700_000_100.0)
    return rec


def compare(id_a, id_b):
    return client.post("/api/benchmarks/compare", json={"a": id_a, "b": id_b})


# ---------------------------------------------------------------------------
# Same verdict
# ---------------------------------------------------------------------------

def test_identical_snapshots_same_true():
    a = make_record("A", params={"n_ctx": 4096, "temp": 0.7})
    b = make_record("B", params={"n_ctx": 4096, "temp": 0.7})

    r = compare(a["id"], b["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same"] is True
    assert body["a"]["benchmarkable"] == body["b"]["benchmarkable"]
    assert body["a"]["benchmarkable"]["params"] == {"n_ctx": 4096, "temp": 0.7}
    assert body["a"]["benchmarkable"]["model_path"] == "/m/model.gguf"


def test_differing_n_ctx_same_false_values_visible():
    a = make_record("A", params={"n_ctx": 4096})
    b = make_record("B", params={"n_ctx": 8192})

    r = compare(a["id"], b["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same"] is False
    assert body["a"]["benchmarkable"]["params"]["n_ctx"] == 4096
    assert body["b"]["benchmarkable"]["params"]["n_ctx"] == 8192


@pytest.mark.parametrize("pa, pb", [
    # 4096 vs 4096.0 - JSON round-trips can flip int<->float; not a change
    ({"n_ctx": 4096}, {"n_ctx": 4096.0}),
    # a false param vs the key absent - store-true flags are omitted when false
    ({"flash_attn": False}, {}),
])
def test_normalization_not_a_change(pa, pb):
    a = make_record("A", params=pa)
    b = make_record("B", params=pb)
    r = compare(a["id"], b["id"])
    assert r.status_code == 200, r.text
    assert r.json()["same"] is True, (pa, pb, r.json())


def test_normalization_custom_flags_empty_vs_absent():
    a = make_record("A", custom_flags="")
    b = make_record("B", custom_flags=None)   # key absent from the snapshot
    r = compare(a["id"], b["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same"] is True
    # both sides normalize to the same (empty) flag string
    assert body["a"]["benchmarkable"]["custom_flags"] == \
        body["b"]["benchmarkable"]["custom_flags"] == ""


def test_normalization_whitespace_only_custom_flags():
    a = make_record("A", custom_flags="  --flash-attn   ")
    b = make_record("B", custom_flags="--flash-attn")
    r = compare(a["id"], b["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same"] is True
    assert body["a"]["benchmarkable"]["custom_flags"] == "--flash-attn"


def test_differing_model_path_same_false():
    a = make_record("A", model_path="/m/alpha.gguf")
    b = make_record("B", model_path="/m/beta.gguf")
    r = compare(a["id"], b["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same"] is False
    assert body["a"]["benchmarkable"]["model_path"] == "/m/alpha.gguf"
    assert body["b"]["benchmarkable"]["model_path"] == "/m/beta.gguf"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_404_a_missing_names_a():
    b = make_record("B")
    r = compare("missing-a-id", b["id"])
    assert r.status_code == 404
    assert "missing-a-id" in r.json()["detail"]


def test_404_b_missing_names_b():
    a = make_record("A")
    r = compare(a["id"], "missing-b-id")
    assert r.status_code == 404
    assert "missing-b-id" in r.json()["detail"]


def test_400_when_a_equals_b():
    a = make_record("A")
    r = compare(a["id"], a["id"])
    assert r.status_code == 400
    assert "different" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Response shape (step 1 of the plan)
# ---------------------------------------------------------------------------

SIDE_FIELDS = {"id", "profile_name", "model_path", "server_version", "status",
               "timestamp", "prefill_tps", "generation_tps", "params_hash",
               "benchmarkable"}
BENCHMARKABLE_FIELDS = {"model_path", "params", "custom_flags"}


def test_per_side_field_shape():
    a = make_record("Alpha", params={"n_ctx": 2048}, custom_flags="--fa",
                    server_version="b999")
    b = make_record("Beta", params={"temp": 0.5}, custom_flags="",
                    server_version="b1000")

    r = compare(a["id"], b["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"a", "b", "same"}
    assert isinstance(body["same"], bool)
    assert body["same"] is False        # different params

    for side, rec in (("a", a), ("b", b)):
        s = body[side]
        assert set(s) == SIDE_FIELDS
        assert set(s["benchmarkable"]) == BENCHMARKABLE_FIELDS
        assert s["id"] == rec["id"]
        assert s["profile_name"] == rec["profile_name"]
        assert s["model_path"] == rec["model_path"]
        assert s["server_version"] == rec["server_version"]
        assert s["status"] == rec["status"]
        assert s["timestamp"] == rec["timestamp"]
        assert s["prefill_tps"] == rec["prefill_tps"]
        assert s["generation_tps"] == rec["generation_tps"]
        assert s["params_hash"] == rec["params_hash"]

    # the benchmarkable subset is exactly what lands on the command line
    assert body["a"]["benchmarkable"] == {
        "model_path": "/m/model.gguf", "params": {"n_ctx": 2048},
        "custom_flags": "--fa"}
    assert body["b"]["benchmarkable"]["params"] == {"temp": 0.5}
    assert body["b"]["benchmarkable"]["custom_flags"] == ""
