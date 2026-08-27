"""
download_manager.py

Downloads one or more GGUF files (a "group" - e.g. all parts of a
multi-part quant) from Hugging Face into the local model folder structure,
streaming to disk in chunks with progress published to subscribers
(consumed by the /ws/downloads WebSocket route).

Supports up to MAX_CONCURRENT_DOWNLOADS concurrent downloads: each job is
keyed "<repo_id>::<group_name>" and reported independently; extra starts
wait in a queue until a slot frees up. Mirrors the shape of
process_manager.py so the two background-job patterns stay consistent.
"""

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

STATE_QUEUED = "queued"
STATE_DOWNLOADING = "downloading"
STATE_DONE = "done"
STATE_ERROR = "error"
STATE_CANCELLED = "cancelled"
STATE_IDLE = "idle"  # legacy name, kept for import compatibility

ACTIVE_STATES = (STATE_QUEUED, STATE_DOWNLOADING)

MAX_CONCURRENT_DOWNLOADS = 3
# Finished jobs are dropped from memory after this long. The UI keeps its
# own copy of finished cards until the user dismisses them.
TERMINAL_TTL_SECONDS = 600

CHUNK_SIZE = 1024 * 1024  # 1 MB
PUBLISH_INTERVAL_SECONDS = 0.25
USER_AGENT = "LlamaModelManager/1.0 (+local desktop app; https://github.com/ggml-org/llama.cpp)"


class _Cancelled(Exception):
    pass


def _download_url(repo_id: str, path: str) -> str:
    encoded = "/".join(quote(part) for part in path.split("/"))
    return f"https://huggingface.co/{repo_id}/resolve/main/{encoded}"


def make_key(repo_id: str, group_name: str) -> str:
    return f"{repo_id}::{group_name}"


class _Job:
    """One download: a repo + a quantization group (one or more files)."""

    __slots__ = (
        "key", "repo_id", "group_name", "files", "dest_dir",
        "state", "current_index", "bytes_done_current", "bytes_total_current",
        "bytes_done_overall", "bytes_total_overall", "speed_bytes_per_sec",
        "error_message", "cancel_event", "created_at", "finished_at",
    )

    def __init__(self, key: str, repo_id: str, group_name: str,
                 files: List[Dict[str, Any]], dest_dir: str) -> None:
        self.key = key
        self.repo_id = repo_id
        self.group_name = group_name
        self.files = files
        self.dest_dir = dest_dir
        self.state = STATE_QUEUED
        self.current_index = -1
        self.bytes_done_current = 0
        self.bytes_total_current = 0
        self.bytes_done_overall = 0
        self.bytes_total_overall = sum(f.get("size_bytes", 0) for f in files)
        self.speed_bytes_per_sec = 0.0
        self.error_message = ""
        self.cancel_event = threading.Event()
        self.created_at = time.time()
        self.finished_at: Optional[float] = None

    def status_dict(self) -> Dict[str, Any]:
        current_file = None
        if 0 <= self.current_index < len(self.files):
            current_file = self.files[self.current_index].get("filename")
        return {
            "state": self.state,
            "repo_id": self.repo_id,
            "group_name": self.group_name,
            "current_file": current_file,
            "file_index": self.current_index + 1,
            "total_files": len(self.files),
            "bytes_done_current": self.bytes_done_current,
            "bytes_total_current": self.bytes_total_current,
            "bytes_done_overall": self.bytes_done_overall,
            "bytes_total_overall": self.bytes_total_overall,
            "speed_bytes_per_sec": self.speed_bytes_per_sec,
            "error_message": self.error_message,
        }


class DownloadManager:
    """Owns up to MAX_CONCURRENT_DOWNLOADS running download jobs (+ queue)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, _Job] = {}
        self._sem = threading.BoundedSemaphore(MAX_CONCURRENT_DOWNLOADS)
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.max_concurrent = MAX_CONCURRENT_DOWNLOADS

    # ---------- public status ----------

    def status_all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            return {key: job.status_dict() for key, job in self._jobs.items()}

    def frame(self) -> Dict[str, Any]:
        """Shape pushed over /ws/downloads and returned by /api/downloads/status."""
        with self._lock:
            self._prune_locked()
            return {
                "downloads": {key: job.status_dict() for key, job in self._jobs.items()},
                "max_concurrent": self.max_concurrent,
            }

    # ---------- lifecycle ----------

    def start(self, repo_id: str, group_name: str, files: List[Dict[str, Any]],
              dest_dir: str, loop: asyncio.AbstractEventLoop) -> str:
        """Start a download; returns its key (repo_id::group_name)."""
        key = make_key(repo_id, group_name or "")
        with self._lock:
            existing = self._jobs.get(key)
            if existing and existing.state in ACTIVE_STATES:
                raise RuntimeError(
                    f"Already downloading {repo_id} ({group_name or 'group'}). "
                    "Wait for it to finish or cancel it first."
                )
            if not files:
                raise ValueError("No files to download.")
            self._prune_locked()
            job = _Job(key, repo_id, group_name, files, dest_dir)
            self._jobs[key] = job
            self._loop = loop
        threading.Thread(target=self._run, args=(job,), daemon=True,
                         name=f"download-{key}").start()
        self._publish()
        return key

    def cancel(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Cancel one job (by key) or all of them (key=None)."""
        with self._lock:
            targets = [j for j in self._jobs.values() if key is None or j.key == key]
            for job in targets:
                if job.state in ACTIVE_STATES:
                    job.cancel_event.set()
        self._publish()
        return {"cancelled": len(targets)}

    # ---------- pub/sub ----------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(q)
        q.put_nowait(json.dumps(self.frame()))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self) -> None:
        with self._lock:
            payload = json.dumps(self.frame())
            subscribers = list(self._subscribers)
            loop = self._loop
        if loop is None:
            return
        for q in subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, payload)
            except RuntimeError:
                pass  # loop closed / shutting down

    # ---------- worker ----------

    def _run(self, job: _Job) -> None:
        acquired = self._acquire_slot(job)
        try:
            if acquired:
                job.state = STATE_DOWNLOADING
                self._publish()
                self._run_files(job)
                job.state = STATE_DONE
        except _Cancelled:
            job.state = STATE_CANCELLED
        except Exception as e:  # defensive: never let the worker thread die silently
            job.state = STATE_ERROR
            if not job.error_message:
                job.error_message = f"Download failed: {e}"
        finally:
            job.finished_at = time.time()
            if acquired:
                self._sem.release()
            self._publish()

    def _run_files(self, job: _Job) -> None:
        dest_root = Path(job.dest_dir)
        for idx, f in enumerate(job.files):
            job.current_index = idx
            job.bytes_done_current = 0
            job.bytes_total_current = f.get("size_bytes", 0)
            self._publish()
            self._download_one(job, f, dest_root)

    def _acquire_slot(self, job: _Job) -> bool:
        """Wait for one of the N concurrent-download slots (cancellable)."""
        while True:
            if job.cancel_event.is_set():
                return False
            if self._sem.acquire(timeout=0.25):
                if job.cancel_event.is_set():
                    self._sem.release()
                    return False
                return True

    def _download_one(self, job: _Job, f: Dict[str, Any], dest_root: Path) -> None:
        rel_path = f["path"]
        target_path = dest_root.joinpath(*rel_path.split("/"))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(target_path.name + ".part")

        url = _download_url(job.repo_id, rel_path)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

        last_publish_ts = 0.0
        last_speed_ts = time.time()
        last_speed_bytes = 0

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                # Prefer the real Content-Length if the tree API didn't give us a size.
                content_length = resp.headers.get("Content-Length")
                if content_length and not f.get("size_bytes"):
                    job.bytes_total_current = int(content_length)
                    job.bytes_total_overall += int(content_length)

                with open(tmp_path, "wb") as out:
                    while True:
                        if job.cancel_event.is_set():
                            raise _Cancelled()
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        n = len(chunk)
                        job.bytes_done_current += n
                        job.bytes_done_overall += n

                        now = time.time()
                        if now - last_publish_ts >= PUBLISH_INTERVAL_SECONDS:
                            elapsed = now - last_speed_ts
                            if elapsed > 0:
                                job.speed_bytes_per_sec = max(
                                    0.0, (job.bytes_done_overall - last_speed_bytes) / elapsed
                                )
                            last_speed_bytes = job.bytes_done_overall
                            last_speed_ts = now
                            last_publish_ts = now
                            self._publish()

            tmp_path.replace(target_path)

        except _Cancelled:
            tmp_path.unlink(missing_ok=True)
            raise
        except urllib.error.HTTPError as e:
            tmp_path.unlink(missing_ok=True)
            if e.code == 404:
                raise RuntimeError(f"File not found on Hugging Face: {f['filename']}")
            raise RuntimeError(f"Download failed (HTTP {e.code}) for {f['filename']}.")
        except urllib.error.URLError as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Network error downloading {f['filename']}: {e.reason}")
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not write {f['filename']} to disk: {e}")

    def _prune_locked(self) -> None:
        cutoff = time.time() - TERMINAL_TTL_SECONDS
        stale = [
            key for key, job in self._jobs.items()
            if job.finished_at is not None and job.finished_at < cutoff
        ]
        for key in stale:
            del self._jobs[key]


# Module-level singleton - up to MAX_CONCURRENT_DOWNLOADS jobs at a time.
manager = DownloadManager()
