"""
main.py

Entrypoint for Llama Profile Manager (LPM).

Starts the FastAPI backend (via uvicorn) on a background thread, then opens
a native desktop window pointed at it using pywebview. If pywebview isn't
installed/available on this platform, falls back to opening the app in the
system's default web browser so the tool is still usable.

Run with:  python main.py
"""

import asyncio
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from backend import __version__
from backend.app import app, runtime_info
from backend.settings import load_settings

LOOPBACK_HOST = "127.0.0.1"
LAN_HOST = "0.0.0.0"
DEFAULT_PORT = 47291  # arbitrary high port to avoid clashing with llama-server itself

# Window/taskbar icon. pywebview (WinForms backend on Windows) loads it via
# System.Drawing.Icon, so it must be a real .ico (multi-size if possible).
# When the file is missing the backend falls back to the Python executable's
# own icon, so the app keeps working without it.
ICON_FILE = Path(__file__).resolve().parent / "assets" / "icon.ico"


class Api:
    """
    Exposed to the frontend as `window.pywebview.api.*`. Only available when
    running inside the native pywebview window - the browser-fallback mode
    has no equivalent, so the frontend checks for `window.pywebview` before
    calling any of these and falls back to plain text entry otherwise.
    """

    def _first(self, result):
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result

    @staticmethod
    def _dialog_kind(name: str):
        """File-dialog kind for this pywebview version: the FileDialog enum
        (newer releases) or the legacy *_DIALOG constants (older ones)."""
        import webview
        enum = getattr(webview, "FileDialog", None)
        if enum is not None:
            return getattr(enum, name)
        return getattr(webview, f"{name}_DIALOG")

    def pick_executable(self):
        """Native "open file" dialog, for choosing the llama-server binary."""
        import webview
        window = webview.windows[0]
        result = window.create_file_dialog(self._dialog_kind("OPEN"), allow_multiple=False)
        return self._first(result)

    def pick_folder(self):
        """Native "open folder" dialog, for choosing a model root folder."""
        import webview
        window = webview.windows[0]
        result = window.create_file_dialog(self._dialog_kind("FOLDER"), allow_multiple=False)
        return self._first(result)

    def open_text_file(self):
        """Native "open file" dialog that reads the picked file as UTF-8 text.
        The Import flow uses this to pull a profile JSON in. Returns
        {"path", "content"}, {"path", "error"} when the file can't be read,
        or None when the user cancels."""
        import webview
        window = webview.windows[0]
        result = window.create_file_dialog(self._dialog_kind("OPEN"), allow_multiple=False)
        path = self._first(result)
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {"path": path, "content": f.read()}
        except (OSError, UnicodeDecodeError) as e:
            return {"path": path, "error": f"Could not read that file as text: {e}"}

    def save_text_file(self, default_filename: str, content: str):
        """Native "save file" dialog; writes `content` to the chosen path."""
        import webview
        window = webview.windows[0]
        result = window.create_file_dialog(self._dialog_kind("SAVE"), save_filename=default_filename)
        path = self._first(result)
        if not path:
            return None
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path


def _find_free_port(preferred: int, bind_host: str) -> int:
    """Use the preferred port if free on `bind_host`, otherwise let the OS pick one."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((bind_host, port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Could not find a free port to bind the local web server to.")


# When a client drops a connection abruptly - the WebView2 window being
# reloaded/closed, the /ws/logs log socket disconnecting, the browser
# reclaiming an idle keep-alive socket - the event loop's transport fails
# to shut the already-dead socket down and raises ConnectionResetError
# (WinError 10054 on Windows). Nothing is lost: the connection is gone and
# the server keeps serving - but asyncio's default handler dumps a scary
# "Exception in callback _ProactorBasePipeTransport._call_connection_lost"
# traceback into the console each time. Swallow exactly that class of
# "client went away" noise; anything else still goes to the default handler.
def _quiet_loop_exception(loop, context):
    exc = context.get("exception")
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return
    loop.default_exception_handler(context)


def _run_server(bind_host: str, port: int) -> None:
    config = uvicorn.Config(app, host=bind_host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    async def _serve():
        # Same as server.run(), but with the exception handler installed on
        # the loop uvicorn actually runs on (run() creates it internally).
        asyncio.get_running_loop().set_exception_handler(_quiet_loop_exception)
        await server.serve(sockets=None)

    asyncio.run(_serve())


def _wait_for_server(url: str, timeout_seconds: float = 10.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def _detect_lan_ip():
    """Best-effort local network IP for the console message below."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return None


def _set_app_user_model_id() -> None:
    """Give this process a unique Windows AppUserModelID.

    Without one, Windows groups the window under the python.exe executable and
    renders *python.exe's* icon in the taskbar, ignoring the window icon that
    pywebview assigns (assets/icon.ico). Setting a distinct ID makes the taskbar
    (and jump lists / Win+Tab) treat the app as its own program and use the
    window icon instead. Must be called before the window is created. No-op off
    Windows."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "LPM.LlamaProfileManager"
        )
    except Exception:
        # Never let a cosmetic taskbar nicety break app startup.
        pass


def _stop_server_on_close(_event=None):
    """pywebview window-closing hook: make sure a llama-server started by
    this app doesn't keep running after the window is closed. Synchronous
    and short (~3 s graceful, then force-kill of the process tree). The
    atexit handler in process_manager is a backup if this never runs."""
    try:
        from backend import process_manager
        process_manager.manager.kill_on_exit()
    except Exception:
        pass


def main() -> None:
    print(f"Llama Profile Manager v{__version__}")
    app_settings = load_settings()
    allow_lan = bool(app_settings.get("allow_lan_access", False))
    bind_host = LAN_HOST if allow_lan else LOOPBACK_HOST

    port = _find_free_port(DEFAULT_PORT, bind_host)
    runtime_info["bind_host"] = bind_host
    runtime_info["port"] = port

    # The window/browser always points at loopback for local display -
    # 0.0.0.0 isn't itself a connectable address, it just means "accept
    # connections on any interface".
    url = f"http://{LOOPBACK_HOST}:{port}/"

    server_thread = threading.Thread(target=_run_server, args=(bind_host, port), daemon=True)
    server_thread.start()

    if not _wait_for_server(url):
        print(f"Warning: backend did not respond within the timeout; opening {url} anyway.", file=sys.stderr)

    if allow_lan:
        lan_ip = _detect_lan_ip()
        print(f"Remote access is enabled (Settings > Remote access).")
        if lan_ip:
            print(f"Reachable from other devices on your network at: http://{lan_ip}:{port}/")
        print("There is no login on this - only leave this enabled on networks you trust.")

    _set_app_user_model_id()

    try:
        import webview  # pywebview
        window = webview.create_window(
            "Llama Profile Manager", url, width=1500, height=960, min_size=(960, 640),
            js_api=Api(),
        )
        window.events.closing += _stop_server_on_close
        # `icon` is a start() argument (stored in pywebview's state and read
        # by the backend when it builds the window), not a create_window one.
        webview.start(icon=str(ICON_FILE) if ICON_FILE.is_file() else None)
    except ImportError:
        print("pywebview not available; opening in your default browser instead.")
        webbrowser.open(url)
        print(f"Llama Profile Manager is running at {url}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
