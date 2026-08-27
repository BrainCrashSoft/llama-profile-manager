"""
llama_server_download.py

One-click installer for a llama.cpp release: downloads the per-platform
asset (zip or tar.gz) from GitHub into data/llama-servers/, extracts the
whole directory that contains llama-server(.exe) (the Windows build needs
its companion DLLs next to the launcher exe) as
llama-server-b{N}/llama-server(.exe), and registers it in the
llama_servers settings list (auto-activating only when no build is
currently configured).

Single-job variant of the download_manager.py pattern: one RLock-guarded
state dict, a worker thread, a cancel event checked between chunks, and a
finished job that stays reported until the next start().
"""

import os
import re
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from . import llama_release, settings

CHUNK_SIZE = 1024 * 1024  # 1 MB
DEFAULT_DEST = settings.DATA_DIR / "llama-servers"

STATE_IDLE = "idle"
STATE_DOWNLOADING = "downloading"
STATE_EXTRACTING = "extracting"
STATE_REGISTERING = "registering"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"


def _binary_name(build: int) -> str:
    suffix = ".exe" if settings._binary_name().endswith(".exe") else ""
    return f"llama-server-b{build}{suffix}"


def _binary_file_name() -> str:
    return settings._binary_name()  # llama-server(.exe)


def _initial_status() -> Dict[str, Any]:
    return {
        "state": STATE_IDLE,
        "bytes_done": 0,
        "bytes_total": 0,
        "tag": "",
        "error": "",
        "installed_path": "",
        "entry_name": "",
    }


class LlamaServerInstaller:
    """Owns the single active llama-server download/install job."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._status: Dict[str, Any] = _initial_status()
        self._cancel_event: Optional[threading.Event] = None
        self._worker: Optional[threading.Thread] = None

    # ---------- public API ----------

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def cancel(self) -> Dict[str, Any]:
        with self._lock:
            active = self._status["state"] in (
                STATE_DOWNLOADING, STATE_EXTRACTING, STATE_REGISTERING
            )
        if active and self._cancel_event is not None:
            self._cancel_event.set()
        return {"cancelled": active}

    def start(self, tag: str, asset_name: str, url: str,
              dest_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Start the download/install. Returns the initial status dict.
        Raises RuntimeError if a job is already active (409-equivalent),
        ValueError for a malformed tag/asset name.
        """
        m = re.match(r"^b(\d+)$", (tag or "").strip())
        if not m:
            raise ValueError(f"Invalid release tag: {tag!r}")
        build = int(m.group(1))

        # Re-validate the asset name before any bytes are written: it must be
        # a real llama.cpp asset for THIS machine's OS+arch (any variant -
        # CPU/CUDA/Vulkan/... - is fine, another platform's is not).
        info = llama_release.parse_asset(asset_name or "")
        os_token, arch_token = llama_release.platform_tokens()
        if (not info or info["build"] != str(build) or not os_token
                or info["os"] != os_token or info["arch"] != arch_token):
            raise ValueError(f"Invalid asset name for {tag}: {asset_name!r}")
        # "" for CPU (both the token-less "macos-x64" and "win-cpu-x64" forms),
        # "-cuda-12.4", "-vulkan", ... otherwise - used in dir + entry names.
        variant_suffix = "" if info["mid"] in ("", "-cpu") else info["mid"]
        # "zip" (Windows) or "tar.gz" (Linux/macOS) - the archive format is a
        # property of the chosen asset, not of the machine's filesystem.
        ext = info["ext"]

        dest = Path(dest_dir) if dest_dir else DEFAULT_DEST

        with self._lock:
            if self._status["state"] in (STATE_DOWNLOADING, STATE_EXTRACTING, STATE_REGISTERING):
                raise RuntimeError("A llama-server download is already in progress. Cancel it first.")
            self._status = _initial_status()
            self._status.update({"state": STATE_DOWNLOADING, "tag": f"b{build}"})
            self._cancel_event = threading.Event()

        self._worker = threading.Thread(
            target=self._run, args=(build, asset_name, url, dest, variant_suffix, ext),
            daemon=True, name=f"llama-server-install-{build}",
        )
        self._worker.start()
        return self.status()

    # ---------- internals ----------

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)

    def _cancelled(self) -> bool:
        ev = self._cancel_event
        return ev is not None and ev.is_set()

    def _run(self, build: int, asset_name: str, url: str, dest: Path,
             variant_suffix: str, ext: str) -> None:
        # The part name carries the real archive extension so the extract
        # step (and anyone peeking into data/llama-servers/) can tell the
        # format apart: ...-6120.zip.part vs ...-6120-vulkan.tar.gz.part.
        part = dest / f"llama-server-{build}{variant_suffix}.{ext}.part"
        try:
            dest.mkdir(parents=True, exist_ok=True)
            self._download(url, part)
            if self._cancelled():
                raise _Cancelled()
            self._set(state=STATE_EXTRACTING)
            binary = self._extract(part, dest, build, variant_suffix, ext)
            self._set(state=STATE_REGISTERING)
            entry_name = self._register(build, binary, variant_suffix)
            self._set(state=STATE_DONE, installed_path=str(binary.resolve()), entry_name=entry_name)
        except _Cancelled:
            part.unlink(missing_ok=True)
            self._set(state=STATE_CANCELLED, error="Cancelled by user.")
        except Exception as e:  # defensive: never let the worker die silently
            part.unlink(missing_ok=True)
            self._set(state=STATE_ERROR, error=str(e) or e.__class__.__name__)

    def _download(self, url: str, part: Path) -> None:
        last_publish = 0.0
        try:
            with llama_release._urlopen(url, timeout=60) as resp:
                content_length = resp.headers.get("Content-Length")
                total = int(content_length) if content_length else 0
                self._set(bytes_total=total)
                with open(part, "wb") as out:
                    while True:
                        if self._cancelled():
                            raise _Cancelled()
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        done = out.tell()
                        now = time.time()
                        if now - last_publish >= 0.25:
                            self._set(bytes_done=done)
                            last_publish = now
            self._set(bytes_done=part.stat().st_size)
        except _Cancelled:
            raise
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Download failed (HTTP {e.code}).")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error during download: {e.reason}")
        except OSError as e:
            raise RuntimeError(f"Could not write the download to disk: {e}")

    def _extract(self, archive: Path, dest: Path, build: int,
                 variant_suffix: str = "", ext: str = "zip") -> Path:
        """
        Pull the whole directory containing llama-server(.exe) out of the
        zip/tar.gz (whatever depth it sits at) and install it as
        dest/llama-server-b{N}{variant}/ - the Windows launcher exe needs
        its DLL siblings to run. `ext` ("zip" or "tar.gz") comes from the
        asset name, not the part file's suffix ("....tar.gz.part" ends in
        ".part", so a name check would misroute tar.gz to the zip reader).
        Returns the installed binary's path.
        """
        import posixpath

        bin_name = _binary_file_name()
        target_dir = dest / f"llama-server-b{build}{variant_suffix}"
        tmp_extract = dest / f".extract-{int(time.time() * 1000)}"

        def under(name: str, parent: str) -> bool:
            """True if archive entry `name` lives in `parent` (possibly root)."""
            if parent == "":
                return not name.startswith(("/", ".."))
            return name.startswith(parent + "/") and "/../" not in name

        def member_path(rel: str) -> Path:
            p = (tmp_extract / rel).resolve()
            if not str(p).startswith(str(tmp_extract.resolve()) + os.sep):
                raise RuntimeError(f"Refusing to extract archive entry outside its folder: {rel}")
            return p

        try:
            if ext == "tar.gz":
                with tarfile.open(archive, "r:gz") as tf:
                    # Regular files AND links: the sonamed libraries the
                    # binary is linked against (libllama-common.so.0 ->
                    # libllama-common.so.0.3.0, ...) are symlinks in the
                    # release tarballs, and dropping them leaves an
                    # installed llama-server that cannot start.
                    members = [m for m in tf.getmembers()
                               if m.isfile() or m.issym() or m.islnk()]
                    files = [m for m in members if m.isfile()]
                    bin_member = next((m for m in files if m.name.rsplit("/", 1)[-1] == bin_name), None)
                    if bin_member is None:
                        raise RuntimeError("The downloaded archive does not contain a llama-server binary.")
                    parent = posixpath.dirname(bin_member.name)
                    extracted: Dict[str, Path] = {}
                    for m in files:
                        if not under(m.name, parent):
                            continue
                        rel = m.name[len(parent) + 1:] if parent else m.name
                        out_path = member_path(rel)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        src = tf.extractfile(m)
                        if src is None:
                            continue
                        with open(out_path, "wb") as out:
                            shutil.copyfileobj(src, out)
                        # Keep each file's own permission bits (the
                        # sibling CLI tools are executables too).
                        try:
                            os.chmod(out_path, m.mode & 0o777)
                        except OSError:
                            pass
                        extracted[m.name] = out_path
                    for m in members:
                        if not (m.issym() or m.islnk()) or not under(m.name, parent):
                            continue
                        rel = m.name[len(parent) + 1:] if parent else m.name
                        out_path = member_path(rel)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.unlink(missing_ok=True)  # duplicate entries
                        if m.issym():
                            linkname = m.linkname or ""
                            # Never materialize absolute or parent-escaping
                            # links (same spirit as member_path() above).
                            if linkname.startswith(("/", "\\")) \
                                    or ".." in linkname.split("/"):
                                raise RuntimeError(
                                    f"Refusing to extract archive entry outside its folder: {m.name}")
                            os.symlink(linkname, out_path)
                        else:  # hard link: target must be an extracted member
                            target = extracted.get(m.linkname or "")
                            if target is None:
                                raise RuntimeError(
                                    f"Hard link target missing from archive: {m.linkname}")
                            os.link(target, out_path)
            else:
                with zipfile.ZipFile(archive) as zf:
                    infos = [i for i in zf.infolist() if not i.is_dir()]
                    bin_info = next((i for i in infos if i.filename.rsplit("/", 1)[-1] == bin_name), None)
                    if bin_info is None:
                        raise RuntimeError("The downloaded archive does not contain a llama-server binary.")
                    parent = posixpath.dirname(bin_info.filename)
                    for i in infos:
                        if not under(i.filename, parent):
                            continue
                        rel = i.filename[len(parent) + 1:] if parent else i.filename
                        out_path = member_path(rel)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(i) as src, open(out_path, "wb") as out:
                            shutil.copyfileobj(src, out)

            binary = tmp_extract / bin_name
            if not binary.is_file():
                raise RuntimeError("Extraction finished but the llama-server binary is missing.")
            if os.name == "posix":
                os.chmod(binary, 0o755)

            shutil.rmtree(target_dir, ignore_errors=True)  # re-install of same build
            shutil.move(str(tmp_extract), str(target_dir))
            return target_dir / bin_name
        finally:
            # Safe to clean up: tmp_extract was either moved away or is stale.
            archive.unlink(missing_ok=True)
            shutil.rmtree(tmp_extract, ignore_errors=True)

    def _register(self, build: int, binary: Path, variant_suffix: str = "") -> str:
        """
        Append {"name": "b{N}{variant}", "path": ...} to llama_servers if not
        already listed, and make it active ONLY when no entry currently has
        a path (never steal the active build from a configured user).
        Returns the entry name.
        """
        name = f"b{build}{variant_suffix}"
        path_str = str(binary.resolve())
        s = settings.load_settings()
        servers = s.get("llama_servers") or []

        ours = next(
            (e for e in servers
             if isinstance(e, dict) and (e.get("path") or "") == path_str),
            None,
        )
        if ours is None:
            ours = {"name": name, "path": path_str}
            servers.append(ours)

        # Only auto-activate when no OTHER entry has a usable path - never
        # steal the active build from a configured user.
        has_other_usable = any(
            (e.get("path") or "").strip()
            for e in servers if isinstance(e, dict) and e is not ours
        )
        update: Dict[str, Any] = {"llama_servers": servers}
        if not has_other_usable:
            update["active_llama_server"] = name
        settings.update_settings(update)
        return name


class _Cancelled(Exception):
    pass


# Name pattern of the per-build install folders this app creates under
# DEFAULT_DEST (llama-server-b6120, llama-server-b6200-cuda-12.4, ...).
_BUILD_DIR_RE = re.compile(r"^llama-server-b\d+", re.IGNORECASE)


def remove_build_files(path_str: str) -> Dict[str, Any]:
    """
    Delete the on-disk files of a llama-server build entry.

    * A binary inside one of this app's own per-build install folders
      ({data}/llama-servers/llama-server-b{N}[variant]/) removes the whole
      folder - the Windows launcher needs its DLL siblings, and nothing else
      lives in these folders.
    * Any other existing file removes that file only - the app never touches
      siblings of a binary the user pointed at in their own folder.
    * A directory that is not one of this app's build install folders is
      refused (ValueError) - arbitrary folders are never deleted.
    * A missing path is not an error: {deleted: False}, so the caller can
      still drop the stale settings entry.
    """
    p = Path((path_str or "").strip()).expanduser()
    if not p.parts:
        raise ValueError("No path given.")
    try:
        p = p.resolve()
    except OSError as e:
        raise ValueError(f"Could not resolve {path_str!r}: {e}")

    if not p.exists():
        return {"deleted": False, "removed": "", "message": "Not on disk - nothing to delete."}

    dest = DEFAULT_DEST.resolve()
    if p.is_dir():
        if p.parent == dest and _BUILD_DIR_RE.match(p.name):
            target, what = p, "the build folder"
        else:
            raise ValueError(
                "Refusing to delete a folder that is not one of this app's build installs."
            )
    else:
        # The build folder is the binary's parent; it is one of this app's
        # installs when it sits directly under DEFAULT_DEST with the right
        # name pattern.
        build_dir = p.parent
        if build_dir.parent == dest and _BUILD_DIR_RE.match(build_dir.name):
            target, what = build_dir, "the build folder"
        else:
            target, what = p, "the file"

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"deleted": True, "removed": str(target), "message": f"Deleted {what}: {target}"}


# Module-level singleton.
installer = LlamaServerInstaller()
