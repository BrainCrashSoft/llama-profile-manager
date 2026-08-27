"""
Unit tests for backend.hf_client: group listing (list_gguf_groups) and the
already-downloaded matching (mark_downloaded).

Focus: the nested per-quant folder layout that some HF repos use for large
multi-part models, e.g.

    unsloth/Qwen3.8-Flash-Next-GGUF/
    ├── UD-Q2_K_XL/
    │   ├── Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf
    │   ├── Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf
    │   └── Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf
    └── Qwen3.8-Flash-Next-Q4_K_M.gguf

Downloads mirror that layout under {root}/org/repo/ (download_manager writes
each file at dest_root/<hf-relative path>), so mark_downloaded must match
files in subfolders, not just the repo folder's direct children.

Run with:  pytest tests/test_hf_client.py
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import hf_client

REPO_ID = "unsloth/Qwen3.8-Flash-Next-GGUF"

# HF tree for the layout above (as /api/models/<repo>/tree/main?recursive=true
# returns it): note the per-quant subfolder holding the multi-part files.
HF_TREE = [
    {"type": "file", "path": "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf", "size": 100},
    {"type": "file", "path": "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf", "size": 100},
    {"type": "file", "path": "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf", "size": 100},
    {"type": "file", "path": "Qwen3.8-Flash-Next-Q4_K_M.gguf", "size": 50},
]


def _groups():
    """The groups list_gguf_groups builds from HF_TREE (fetch mocked out)."""
    with mock.patch.object(hf_client, "fetch_repo_tree", return_value=HF_TREE):
        return hf_client.list_gguf_groups(REPO_ID)["groups"]


def _write(rel_path: Path) -> None:
    rel_path.parent.mkdir(parents=True, exist_ok=True)
    rel_path.write_bytes(b"x")


def _mark(root: Path, groups) -> None:
    with mock.patch.object(hf_client.settings, "get_model_root_folders",
                           return_value=[str(root)]):
        hf_client.mark_downloaded(REPO_ID, groups)


def test_list_gguf_groups_collapses_nested_multipart():
    groups = _groups()
    by_name = {g["group_name"]: g for g in groups}

    # Multi-part files in a subfolder collapse into ONE group keyed by the
    # file stem (same keying as the local scanner), keeping the subfolder in
    # each file's repo-relative path.
    assert set(by_name) == {"Qwen3.8-Flash-Next-UD-Q2_K_XL", "Qwen3.8-Flash-Next-Q4_K_M"}
    nested = by_name["Qwen3.8-Flash-Next-UD-Q2_K_XL"]
    assert nested["is_multipart"] is True
    assert nested["total_size_bytes"] == 300
    assert [f["path"] for f in nested["files"]] == [
        "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf",
        "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf",
        "UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf",
    ]


def test_mark_downloaded_matches_nested_multipart(tmp_path):
    groups = _groups()
    repo_dir = tmp_path / "unsloth" / "Qwen3.8-Flash-Next-GGUF"
    for f in HF_TREE:
        _write(repo_dir / Path(f["path"]))

    _mark(tmp_path, groups)
    by_name = {g["group_name"]: g["already_downloaded"] for g in groups}
    assert by_name == {
        "Qwen3.8-Flash-Next-UD-Q2_K_XL": True,
        "Qwen3.8-Flash-Next-Q4_K_M": True,
    }


def test_mark_downloaded_flat_layout_still_matches(tmp_path):
    """Back-compat: the classic flat {root}/org/repo/*.gguf layout."""
    groups = _groups()
    repo_dir = tmp_path / "unsloth" / "Qwen3.8-Flash-Next-GGUF"
    repo_dir.mkdir(parents=True)
    for f in HF_TREE:
        _write(repo_dir / f["path"].rsplit("/", 1)[-1])  # bare filenames

    _mark(tmp_path, groups)
    assert all(g["already_downloaded"] for g in groups)


def test_mark_downloaded_partial_group_is_not_downloaded(tmp_path):
    """All parts must exist: one missing part keeps the group downloadable."""
    groups = _groups()
    repo_dir = tmp_path / "unsloth" / "Qwen3.8-Flash-Next-GGUF"
    _write(repo_dir / "UD-Q2_K_XL" / "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf")
    _write(repo_dir / "UD-Q2_K_XL" / "Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf")
    _write(repo_dir / "Qwen3.8-Flash-Next-Q4_K_M.gguf")

    _mark(tmp_path, groups)
    by_name = {g["group_name"]: g["already_downloaded"] for g in groups}
    assert by_name["Qwen3.8-Flash-Next-UD-Q2_K_XL"] is False
    assert by_name["Qwen3.8-Flash-Next-Q4_K_M"] is True


def test_mark_downloaded_no_repo_dir(tmp_path):
    groups = _groups()
    _mark(tmp_path, groups)  # empty root - nothing on disk
    assert all(g["already_downloaded"] is False for g in groups)
