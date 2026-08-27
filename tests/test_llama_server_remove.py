"""
Unit tests for backend.llama_server_download.remove_build_files - the file
deletion behind the version list's "remove this version" button (the
/app/llama-server/remove endpoint).

Safety contract under test:
  * a binary inside an app-installed build folder removes the WHOLE folder
    (Windows launcher + its DLL siblings);
  * a binary the user pointed at elsewhere removes ONLY that file;
  * arbitrary directories are never deleted (only this app's own
    llama-server-b{N}[variant] install folders);
  * missing paths are not errors (stale entries can still be dropped);
  * empty/blank input raises ValueError.

Run with:  pytest tests/test_llama_server_remove.py
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import llama_server_download as lsd


def _app_build_dir(dest: Path, name="llama-server-b6120") -> Path:
    d = dest / name
    d.mkdir()
    (d / "llama-server.exe").write_bytes(b"x")
    (d / "ggml-base.dll").write_bytes(b"x")
    return d


def test_binary_in_app_build_folder_removes_whole_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)
    d = _app_build_dir(tmp_path)

    res = lsd.remove_build_files(str(d / "llama-server.exe"))

    assert res["deleted"] is True
    assert Path(res["removed"]).resolve() == d.resolve()
    assert not d.exists()


def test_variant_suffix_build_folder_is_recognized(tmp_path, monkeypatch):
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)
    d = _app_build_dir(tmp_path, name="llama-server-b6200-cuda-12.4")

    res = lsd.remove_build_files(str(d / "llama-server.exe"))

    assert res["deleted"] is True
    assert not d.exists()


def test_build_folder_path_directly_is_removed(tmp_path, monkeypatch):
    """An entry whose path is the install folder itself (not the binary)."""
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)
    d = _app_build_dir(tmp_path)

    res = lsd.remove_build_files(str(d))

    assert res["deleted"] is True
    assert not d.exists()


def test_user_binary_outside_dest_removes_only_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)
    user_dir = tmp_path / "user-tools"
    user_dir.mkdir()
    binary = user_dir / "llama-server-b6120.exe"
    binary.write_bytes(b"x")
    (user_dir / "readme.txt").write_bytes(b"keep me")

    res = lsd.remove_build_files(str(binary))

    assert res["deleted"] is True
    assert Path(res["removed"]).resolve() == binary.resolve()
    assert not binary.exists()
    assert (user_dir / "readme.txt").exists()  # siblings untouched


def test_binary_directly_in_dest_removes_only_the_file(tmp_path, monkeypatch):
    """No build subfolder: a loose file in the app's dest dir is not a build."""
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)
    binary = tmp_path / "llama-server-b6120.exe"
    binary.write_bytes(b"x")

    res = lsd.remove_build_files(str(binary))

    assert res["deleted"] is True
    assert not binary.exists()
    assert tmp_path.exists()


def test_arbitrary_folder_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)
    other = tmp_path / "important-stuff"
    other.mkdir()
    (other / "a.txt").write_bytes(b"x")

    with pytest.raises(ValueError):
        lsd.remove_build_files(str(other))
    assert other.exists()  # untouched


def test_missing_path_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)

    res = lsd.remove_build_files(str(tmp_path / "gone" / "llama-server-b1.exe"))

    assert res["deleted"] is False
    assert res["removed"] == ""


def test_blank_path_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(lsd, "DEFAULT_DEST", tmp_path)
    with pytest.raises(ValueError):
        lsd.remove_build_files("   ")
