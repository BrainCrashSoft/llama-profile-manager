"""
process_manager.py

Launches, monitors, and stops the llama-server subprocess. stdout/stderr are
read in a background thread and pushed into an asyncio.Queue per manager
instance, which the FastAPI WebSocket route drains to stream logs live to
the frontend.

Only one server instance is supported at a time in this version (see
README for the multi-instance note). Calling start() while a server is
already running raises RuntimeError.
"""

import atexit
import asyncio
import os
import platform
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_ERROR = "error"
STATE_STOPPING = "stopping"

# How long to wait after SIGTERM before escalating to SIGKILL/terminate()
GRACEFUL_STOP_TIMEOUT_SECONDS = 8


class LlamaServerProcess:
    """Owns at most one running llama-server process."""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None
        self._state: str = STATE_STOPPED
        self._error_message: str = ""
        self._started_at: Optional[float] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._log_lines: List[str] = []
        self._max_log_lines = 2000
        self._subscribers: List[asyncio.Queue] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.RLock()
        self._last_command: List[str] = []
        self._last_host_port: tuple = ("127.0.0.1", 8080)
        # "single" = one model via --model, "router" = multi-model router
        # via --models-preset. The frontend uses this to decide whether the
        # router model-management panel applies.
        self._mode: str = "single"
        # Which profile started the server (None for router mode / none yet).
        # The frontend uses this to turn the owning profile's Start button
        # into a Stop button.
        self._profile_id: Optional[str] = None
        # Which router preset started the server (None in single mode / none
        # yet). Same purpose as _profile_id, for the Router Presets page.
        self._preset_id: Optional[str] = None
        # Which router dir started the server (None in single mode / none yet).
        # Same purpose as _preset_id, for the Router Dir page.
        self._router_dir_id: Optional[str] = None
        # Set when the process exits or is stopped; stops the readiness probe.
        self._ready_event = threading.Event()

        # Safety net: the child process must never outlive this app.
        # atexit covers any normal interpreter exit (window closed, Ctrl+C,
        # unhandled crash); the signal handlers are belt-and-braces for
        # console runs. kill_on_exit() is idempotent, so overlapping
        # triggers are harmless.
        atexit.register(self.kill_on_exit)
        for _sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(_sig, self._signal_shutdown)
            except (ValueError, OSError, AttributeError):
                pass  # not the main thread, or unsupported on this platform
        # NOTE: this must stay an RLock, not a plain Lock. start()/stop() call
        # self.status() while already holding the lock, which would deadlock
        # on a non-reentrant lock.

    # ---------- public status ----------

    def status(self) -> Dict[str, Any]:
        with self._lock:
            pid = self._process.pid if self._process else None
            uptime = (time.time() - self._started_at) if (self._started_at and self._state in (STATE_STARTING, STATE_RUNNING)) else None
            return {
                "state": self._state,
                "pid": pid,
                "uptime_seconds": uptime,
                "error_message": self._error_message,
                "command": self._last_command,
                "host": self._last_host_port[0],
                "port": self._last_host_port[1],
                "mode": self._mode,
                "profile_id": self._profile_id,
                "preset_id": self._preset_id,
                "router_dir_id": self._router_dir_id,
            }

    def recent_logs(self, limit: int = 500) -> List[str]:
        with self._lock:
            return self._log_lines[-limit:]

    # ---------- lifecycle ----------

    def start(self, binary_path: str, args: List[str], host: str, port: int,
               loop: asyncio.AbstractEventLoop, mode: str = "single",
               profile_id: Optional[str] = None,
               preset_id: Optional[str] = None,
               router_dir_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self._state in (STATE_STARTING, STATE_RUNNING):
                raise RuntimeError("A server is already running. Stop it before starting another.")

            bin_path = Path(binary_path)
            if not binary_path:
                self._set_error("No llama-server executable path configured. Set it in Settings.")
                raise FileNotFoundError("llama-server path not configured")
            if not bin_path.exists():
                self._set_error(f"llama-server executable not found at: {binary_path}")
                raise FileNotFoundError(f"llama-server not found at {binary_path}")

            command = [str(bin_path)] + args
            self._last_command = command
            self._last_host_port = (host, port)
            self._mode = mode
            self._profile_id = profile_id
            self._preset_id = preset_id
            self._router_dir_id = router_dir_id
            self._log_lines = []
            self._error_message = ""
            self._state = STATE_STARTING
            self._loop = loop
            self._ready_event.clear()

            creationflags = 0
            preexec_fn = None
            if platform.system() == "Windows":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                preexec_fn = _new_process_group_posix

            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=creationflags,
                    preexec_fn=preexec_fn,
                )
            except FileNotFoundError:
                self._set_error(f"Could not execute llama-server at: {binary_path}")
                raise
            except PermissionError:
                self._set_error(f"Permission denied when trying to run: {binary_path}")
                raise
            except OSError as e:
                self._set_error(f"Failed to launch llama-server: {e}")
                raise

            self._started_at = time.time()
            # "starting" now doubles as the model-loading state: it stays this
            # way until the readiness probe below sees the API come up (or the
            # process crashes, which the reader thread flips to "error").
            self._state = STATE_STARTING

            self._reader_thread = threading.Thread(target=self._read_output_loop, daemon=True)
            self._reader_thread.start()
            self._ready_thread = threading.Thread(
                target=self._wait_until_ready, args=(host, port), daemon=True
            )
            self._ready_thread.start()

            return self.status()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._process or self._state in (STATE_STOPPED,):
                self._state = STATE_STOPPED
                self._profile_id = None
                self._preset_id = None
                self._router_dir_id = None
                return self.status()
            self._state = STATE_STOPPING
            self._profile_id = None
            self._preset_id = None
            self._router_dir_id = None
            proc = self._process

        try:
            if platform.system() == "Windows":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()  # SIGTERM
        except Exception:
            pass

        def _wait_and_force_kill():
            try:
                proc.wait(timeout=GRACEFUL_STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
            with self._lock:
                self._state = STATE_STOPPED
                self._started_at = None

        threading.Thread(target=_wait_and_force_kill, daemon=True).start()
        return self.status()

    # ---------- shutdown safety net ----------

    def kill_on_exit(self) -> None:
        """Synchronously terminate the child process (if any). Called when
        the app is going away (window close, atexit, signal) so no
        llama-server is left running behind a closed app.

        Idempotent and exception-safe: it must never raise, and calling it
        when nothing is running is a no-op."""
        try:
            with self._lock:
                proc = self._process
                self._process = None
                self._state = STATE_STOPPED
                self._started_at = None
                self._profile_id = None
                self._preset_id = None
                self._router_dir_id = None
                self._ready_event.set()  # release the readiness probe
            if proc is None:
                return
            try:
                if platform.system() == "Windows":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()  # SIGTERM
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._kill_tree(proc)  # escalation includes any child processes
        except Exception:
            pass  # last resort path - never let shutdown die here

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """Force-kill the process and everything it spawned."""
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=5)
            else:
                # the child calls setsid(), so its process group id == pid
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()
        except Exception:
            pass

    def _signal_shutdown(self, signum, frame):
        # Kill the child, then let Python run its normal shutdown (atexit
        # re-runs kill_on_exit, which is a no-op by then).
        self.kill_on_exit()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    # ---------- log streaming ----------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append(q)
            # replay recent history so a newly-connected client has context
            for line in self._log_lines[-200:]:
                q.put_nowait(line)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, line: str) -> None:
        with self._lock:
            self._log_lines.append(line)
            if len(self._log_lines) > self._max_log_lines:
                self._log_lines = self._log_lines[-self._max_log_lines:]
            subscribers = list(self._subscribers)
            loop = self._loop
        if loop is None:
            return
        for q in subscribers:
            try:
                loop.call_soon_threadsafe(q.put_nowait, line)
            except RuntimeError:
                pass  # loop closed / shutting down

    def _read_output_loop(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                self._publish(line)
                self._maybe_detect_error(line)
        except Exception as e:  # defensive: never let the reader thread crash silently
            self._publish(f"[model-manager] log reader stopped: {e}")
        finally:
            exit_code = proc.wait()
            with self._lock:
                if exit_code != 0 and self._state != STATE_STOPPING and self._state != STATE_STOPPED:
                    self._state = STATE_ERROR
                    if not self._error_message:
                        self._error_message = f"llama-server exited with code {exit_code}."
                else:
                    self._state = STATE_STOPPED
                self._started_at = None
                self._process = None
                self._ready_event.set()  # release the readiness probe
            self._publish(f"[model-manager] process exited with code {exit_code}")

    def _wait_until_ready(self, host: str, port: int) -> None:
        """Fallback readiness check for builds whose log format we don't
        recognize: if the process is still alive after a grace period and
        the port answers /health with 200, consider it ready. The normal
        path is the log-line detection in _maybe_detect_error, which fires
        as soon as llama.cpp prints "listening on …"."""
        grace_seconds = 30
        probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        url = f"http://{probe_host}:{port}/health"
        while not self._ready_event.wait(1.0):
            with self._lock:
                if self._state != STATE_STARTING or self._process is None:
                    return
                elapsed = time.time() - (self._started_at or time.time())
                if elapsed < grace_seconds:
                    continue
            try:
                with urllib.request.urlopen(url, timeout=0.5) as resp:
                    if resp.status == 200:
                        with self._lock:
                            if self._state == STATE_STARTING:
                                self._state = STATE_RUNNING
                                self._publish("[model-manager] model loaded and ready")
                        return
            except Exception:
                pass  # not up yet - loop re-checks

    def _maybe_detect_error(self, line: str) -> None:
        lower = line.lower()
        with self._lock:
            if "address already in use" in lower or "eaddrinuse" in lower:
                self._error_message = "Port is already in use. Choose a different port and try again."
            elif "cannot find" in lower and ".gguf" in lower:
                self._error_message = "Model file could not be found at the configured path."
            elif "out of memory" in lower or "cuda error" in lower and "memory" in lower:
                self._error_message = "The server ran out of GPU/CPU memory. Try lowering context size, batch size, or GPU layers."
            elif "failed to load model" in lower:
                self._error_message = "The model file failed to load. It may be corrupt, incomplete, or an unsupported format."

            # llama.cpp prints "listening on http://…" as soon as the model
            # is fully loaded and the API is up - the authoritative
            # "ready" signal for the starting → running transition. (A
            # /health probe alone can be answered by *another* server that
            # already sits on the same port, so the log line wins.)
            if "listening on http" in lower and self._state == STATE_STARTING:
                self._state = STATE_RUNNING
                self._publish("[model-manager] model loaded and ready")

    def _set_error(self, message: str) -> None:
        self._error_message = message
        self._state = STATE_ERROR


def _new_process_group_posix():
    import os
    os.setsid()


# Module-level singleton - the app only supports one running server instance.
manager = LlamaServerProcess()
