"""
End-to-end test for the benchmark feature (BenchPlan.md).

Run with:  python tests/test_benchmark.py

Strategy
--------
A FAKE llama-server stands in for the real binary:
  * `llama-server-fake.bat`  - answers `--version` and otherwise execs the
    fake python server with all real args (so command_builder output is
    exercised unchanged);
  * `fake_server.py`         - an HTTP server that prints the readiness
    line the process manager looks for ("listening on http://…"), answers
    POST /completion with DETERMINISTIC timings (prefill 812.4 tok/s,
    generation 34.2 tok/s, n_prompt = len(prompt)//4) and self-terminates
    when its `--stop-file` appears (or after 300 s as a backstop).

The test backs up the real data/ files (settings, profiles, benchmarks),
works on a clean slate, and restores everything byte-for-byte on exit.

Sections
--------
 1  hash canonicalization (int/float, whitespace, name/notes, False)
 2  first run: temp server, TPS parsing, version, tokens, badge, cleanup
 3  staleness on save (param flip, stale keeps TPS, notes/name no-flip)
 4  re-run from the saved snapshot (re_ran_from, different gen tokens)
 5  cancel mid-run (concurrent start rejected, badge restored)
 6  reusing the user's already-running server (stays up afterwards)
 7  busy slot: a different profile owns the server → refused
 8  import-as-profile (new + overwrite, badge pre-populated)
 9  app.py endpoints (snapshot export shape, delete clears badges)
10  startup recovery of interrupted runs
"""

import json
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fake llama-server fixtures
# ---------------------------------------------------------------------------

FAKE_SERVER_PY = r'''
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PREFILL_TPS = 812.4
GEN_TPS = 34.2

def arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default

host = arg("--host", "127.0.0.1")
port = int(arg("--port", "0") or 0)
stop_file = arg("--stop-file")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        if self.path != "/completion":
            self._send({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send({"error": "bad json"}, 400)
            return
        prompt = req.get("prompt") or ""
        max_tokens = max(1, int(req.get("max_tokens") or 16))
        n_prompt = max(1, len(prompt) // 4)
        time.sleep(max_tokens / 256.0)  # simulate the workload's duration
        self._send({
            "content": "fake completion",
            "n_prompt_tokens": n_prompt,
            "n_predicted_tokens": max_tokens,
            "timings": {
                "prompt_n": n_prompt,
                "prompt_ms": n_prompt / PREFILL_TPS * 1000.0,
                "prompt_per_second": PREFILL_TPS,
                "predicted_n": max_tokens,
                "predicted_ms": max_tokens / GEN_TPS * 1000.0,
                "predicted_per_second": GEN_TPS,
            },
        })

def main():
    srv = ThreadingHTTPServer((host, port), Handler)
    # the exact line process_manager looks for when marking the server ready
    print(f"server is listening on http://{host}:{port}", flush=True)

    def watchdog():
        deadline = time.time() + 300
        while time.time() < deadline:
            if stop_file and os.path.exists(stop_file):
                os._exit(0)
            time.sleep(0.2)
        os._exit(0)  # never leak a fake server past the test

    threading.Thread(target=watchdog, daemon=True).start()
    srv.serve_forever()

main()
'''

# IMPORTANT: label form, NOT `if (...)` blocks - an unescaped ")" inside an
# if-block's echo closes the block early and the wrong branch runs (the fake
# would then exit 0 immediately, before starting the python server).
FAKE_BAT = """@echo off
if "%~1"=="--version" goto :version
python "%~dp0fake_server.py" %*
exit /b %ERRORLEVEL%
:version
echo fake-server version: 999 (b9999)
exit /b 0
"""

PREFILL_TPS = 812.4
GEN_TPS = 34.2


# ---------------------------------------------------------------------------
# Tiny test framework
# ---------------------------------------------------------------------------

FAILED = []
CHECKS = [0]  # mutable counter

def check(name, cond, detail=""):
    CHECKS[0] += 1
    tag = "PASS" if cond else "FAIL"
    suffix = f"  -> {detail}" if (detail and not cond) else ""
    print(f"  [{tag}] {name}{suffix}")
    if not cond:
        FAILED.append(f"{name} {detail}")


def section(title):
    print(f"\n{title}")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def port_open(port):
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        return True
    except OSError:
        return False


def wait_port_gone(port, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not port_open(port):
            return True
        time.sleep(0.3)
    return not port_open(port)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- back up the user's real data BEFORE importing the app -----------
    backup_dir = Path(tempfile.mkdtemp(prefix="lpm-bench-backup-"))
    backup_names = []
    for name in ("settings.json", "profiles.json", "benchmarks.json"):
        f = DATA / name
        if f.exists():
            shutil.copy2(f, backup_dir / name)
            backup_names.append(name)

    # imports happen after the backup: importing benchmark_runner triggers
    # recover_interrupted_runs(), which must only ever see test data.
    from backend import benchmarks as bench_store
    from backend import benchmark_runner
    from backend import command_builder, profiles, settings as app_settings
    from backend.process_manager import manager
    from backend.benchmark_runner import BenchmarkError, runner

    (DATA / "benchmarks.json").unlink(missing_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="lpm-bench-fake-"))
    (tmp / "fake_server.py").write_text(FAKE_SERVER_PY, encoding="utf-8")
    bat = tmp / "llama-server-fake.bat"
    bat.write_text(FAKE_BAT, encoding="utf-8", newline="\r\n")

    def touch_stop(name):
        (tmp / name).touch()

    def ensure_server_gone(port, stop_flag_name):
        """The process manager's CTRL_BREAK kills the whole tree in normal
        operation; the per-profile stop-file is only a backstop. Important:
        stop files are shared across phases through the saved snapshots
        (the re-run uses run-1's custom flags), so they must only be touched
        here as a backstop, never unconditionally mid-test."""
        if wait_port_gone(port, timeout=8.0):
            return
        touch_stop(stop_flag_name)
        ok = wait_port_gone(port, timeout=8.0)
        check(f"fake server for '{stop_flag_name}' fully gone", ok, f"port {port} still open")

    def run_phase_server(profile, port):
        """Start a *user* server (as the UI would) and wait until ready."""
        args = command_builder.build_args(
            profile["model_path"], profile["params"], profile.get("custom_flags", ""),
            host_override="127.0.0.1", port_override=port)
        manager.start(str(bat), args, "127.0.0.1", port, None, mode="single", profile_id=profile["id"])
        deadline = time.time() + 30
        while time.time() < deadline:
            st = manager.status()
            if st["state"] == "running":
                return
            if st["state"] in ("stopped", "error"):
                raise SystemExit(f"phase server failed: {st} | logs: {manager.recent_logs(10)}")
            time.sleep(0.3)
        raise SystemExit("phase server did not become ready")

    def stop_phase_server(port, stop_flag_name):
        manager.stop()
        deadline = time.time() + 20
        while time.time() < deadline and manager.status()["state"] != "stopped":
            time.sleep(0.3)
        ensure_server_gone(port, stop_flag_name)

    def wait_terminal(rec_id, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = bench_store.get_benchmark(rec_id)
            if r and r["status"] != "running":
                return r
            time.sleep(0.5)
        r = bench_store.get_benchmark(rec_id)
        raise SystemExit(f"benchmark {rec_id} still running after {timeout}s: {r}")

    try:
        # ---- settings point at the fake binary ----------------------------
        # NOTE: _normalize_llama_servers re-syncs llama_server_path from the
        # ACTIVE named build, so the list must be replaced too - otherwise
        # the user's real llama-server would be used (and choke on
        # --stop-file).
        app_settings.save_settings({
            **app_settings.load_settings(),
            "llama_servers": [{"name": "fake-b9999", "path": str(bat)}],
            "active_llama_server": "fake-b9999",
            "llama_server_path": str(bat),
            "default_host": "127.0.0.1",
            "default_port": free_port(),
        })

        # =============== 1) hash canonicalization ==========================
        section("1) hash canonicalization")
        base = {"model_path": "C:\\m.gguf",
                "params": {"ctx_size": 4096, "n_gpu_layers": "all"},
                "custom_flags": "  --flash-attn   on "}
        alt = {"model_path": "C:\\m.gguf",
               "params": {"ctx_size": 4096.0, "n_gpu_layers": "all"},
               "custom_flags": "--flash-attn on"}
        check("int vs float + whitespace normalize to same hash",
              bench_store.params_hash(base) == bench_store.params_hash(alt))
        check("real param change produces a different hash",
              bench_store.params_hash(base) != bench_store.params_hash(
                  {**base, "params": {**base["params"], "ctx_size": 8192}}))
        check("name/notes are not part of the hash",
              bench_store.params_hash({**base, "name": "A", "notes": "x"})
              == bench_store.params_hash({**base, "name": "B", "notes": "y"}))
        check("bool False (flag omitted) == absent",
              bench_store.params_hash({"model_path": "x", "params": {"flash_attn": False},
                                       "custom_flags": ""})
              == bench_store.params_hash({"model_path": "x", "params": {},
                                          "custom_flags": ""}))

        # =============== 2) first benchmark run ============================
        section("2) first benchmark run (temporary server)")
        port1 = free_port()
        prof = profiles.create_profile(
            "test::org::repo::model", str(tmp / "model.gguf"), "Bench Test",
            {"ctx_size": 4096, "n_gpu_layers": "all", "port": port1},
            custom_flags=f'--stop-file {tmp / "stop-run1.flag"}')
        source = {"profile_id": prof["id"], "name": prof["name"],
                  "model_path": prof["model_path"], "params": prof["params"],
                  "custom_flags": prof["custom_flags"]}
        snapshot = profiles.export_profile(prof["id"])

        rec1 = runner.start(source, snapshot,
                            {"prompt_tokens": 512, "gen_tokens": 128,
                             "repetitions": 2, "custom_prompt": None}, None)
        r1 = wait_terminal(rec1["id"])
        check("status completed", r1["status"] == "completed",
              f'{r1["status"]} - {r1.get("error")}')
        check("prefill tps parsed", r1["prefill_tps"] == PREFILL_TPS, str(r1["prefill_tps"]))
        check("generation tps parsed", r1["generation_tps"] == GEN_TPS, str(r1["generation_tps"]))
        check("server version captured", "b9999" in (r1["server_version"] or ""),
              repr(r1["server_version"]))
        expected_n_prompt = max(1, len(benchmark_runner._synthetic_prompt(512)) // 4)
        check("prompt tokens recorded (actual, from server)",
              r1["prompt_tokens"] == expected_n_prompt,
              f'{r1["prompt_tokens"]} != {expected_n_prompt}')
        check("gen tokens summed over repetitions", r1["gen_tokens"] == 256,
              str(r1["gen_tokens"]))
        check("duration recorded", (r1["duration_s"] or 0) > 0, str(r1["duration_s"]))
        snap = r1["profile_params_snapshot"]
        check("snapshot stored in export shape",
              snap.get("name") == "Bench Test" and snap.get("params") == prof["params"]
              and "id" not in snap, json.dumps(snap)[:200])
        p = profiles.get_profile(prof["id"])
        badge = p.get("benchmark_badge") or {}
        check("badge fresh after run", badge.get("state") == "fresh", str(badge))
        check("badge links to the record", badge.get("benchmark_id") == rec1["id"], str(badge))
        check("badge keeps TPS", badge.get("generation_tps") == GEN_TPS, str(badge))
        time.sleep(2)  # let the runner's finally-block stop the temp server
        check("temp server stopped after run", manager.status()["state"] == "stopped",
              str(manager.status()))
        ensure_server_gone(port1, "stop-run1.flag")

        # =============== 3) staleness detection on save ====================
        section("3) staleness detection on save")
        profiles.update_profile(prof["id"], {"params": {**prof["params"], "ctx_size": 8192}})
        badge = (profiles.get_profile(prof["id"]).get("benchmark_badge") or {})
        check("param change -> stale", badge.get("state") == "stale", str(badge))
        check("stale keeps the old TPS", badge.get("generation_tps") == GEN_TPS, str(badge))
        profiles.update_profile(prof["id"], {"notes": "just a note"})
        badge = (profiles.get_profile(prof["id"]).get("benchmark_badge") or {})
        check("notes-only edit keeps stale (no flip)", badge.get("state") == "stale", str(badge))
        profiles.update_profile(prof["id"], {"name": "Bench Test 2"})
        badge = (profiles.get_profile(prof["id"]).get("benchmark_badge") or {})
        check("name-only edit keeps stale", badge.get("state") == "stale", str(badge))
        profiles.update_profile(prof["id"], {"params": prof["params"]})
        badge = (profiles.get_profile(prof["id"]).get("benchmark_badge") or {})
        check("params restored -> fresh again", badge.get("state") == "fresh", str(badge))

        # =============== 4) re-run from a saved snapshot ===================
        section("4) re-run from a saved snapshot")
        prof = profiles.get_profile(prof["id"])
        source2 = {"profile_id": prof["id"], "name": snapshot["name"],
                   "model_path": prof["model_path"], "params": prof["params"],
                   "custom_flags": prof["custom_flags"], "re_ran_from": rec1["id"]}
        rec2 = runner.start(source2, snapshot,
                            {"prompt_tokens": 512, "gen_tokens": 64,
                             "repetitions": 1, "custom_prompt": None}, None)
        r2 = wait_terminal(rec2["id"])
        check("re-run completed", r2["status"] == "completed",
              f'{r2["status"]} - {r2.get("error")}')
        check("re_ran_from recorded", r2["re_ran_from"] == rec1["id"], str(r2["re_ran_from"]))
        check("re-run gen tokens = 64", r2["gen_tokens"] == 64, str(r2["gen_tokens"]))
        ensure_server_gone(port1, "stop-run1.flag")

        # =============== 5) cancel mid-run =================================
        section("5) cancel mid-run")
        source3 = dict(source2, re_ran_from=rec2["id"])
        slow_opts = {"prompt_tokens": 256, "gen_tokens": 512,
                     "repetitions": 5, "custom_prompt": None}
        rec3 = runner.start(source3, snapshot, slow_opts, None)
        time.sleep(1.5)  # let it reach warmup / the first measurement
        try:
            runner.start(source3, snapshot, slow_opts, None)
            check("concurrent start rejected", False, "no exception raised")
        except BenchmarkError as e:
            check("concurrent start rejected", "already in progress" in str(e), str(e))
        runner.cancel()
        r3 = wait_terminal(rec3["id"], timeout=60)
        check("cancelled status", r3["status"] == "cancelled",
              f'{r3["status"]} - {r3.get("error")}')
        badge = (profiles.get_profile(prof["id"]).get("benchmark_badge") or {})
        check("badge restored after cancel (last-good kept)",
              badge.get("state") == "fresh"
              and badge.get("benchmark_id") == rec2["id"]
              and badge.get("generation_tps") == GEN_TPS, str(badge))
        time.sleep(2)
        check("cancelled run's temp server stopped", manager.status()["state"] == "stopped",
              str(manager.status()))
        ensure_server_gone(port1, "stop-run1.flag")

        # =============== 6) reusing the user's server ======================
        section("6) reusing the user's already-running server")
        run_phase_server(prof, port1)
        source4 = {"profile_id": prof["id"], "name": prof["name"],
                   "model_path": prof["model_path"], "params": prof["params"],
                   "custom_flags": prof["custom_flags"]}
        rec4 = runner.start(source4, snapshot,
                            {"prompt_tokens": 256, "gen_tokens": 64,
                             "repetitions": 1, "custom_prompt": None}, None)
        r4 = wait_terminal(rec4["id"])
        check("benchmark on reused server completed", r4["status"] == "completed",
              f'{r4["status"]} - {r4.get("error")}')
        raw = json.loads(r4["raw_output"] or "{}")
        check("reused_server flagged in raw output", raw.get("reused_server") is True,
              r4["raw_output"][:200])
        check("user's server still running after benchmark",
              manager.status()["state"] == "running", str(manager.status()))
        stop_phase_server(port1, "stop-run1.flag")

        # =============== 7) busy slot =======================================
        section("7) busy slot: another profile owns the server")
        port2 = free_port()
        prof2 = profiles.create_profile(
            "test::org::repo::model2", str(tmp / "model2.gguf"), "Busy Prof",
            {"ctx_size": 2048, "port": port2},
            custom_flags=f'--stop-file {tmp / "stop-busy.flag"}')
        run_phase_server(prof2, port2)
        try:
            runner.start(source4, snapshot,
                         {"prompt_tokens": 256, "gen_tokens": 64,
                          "repetitions": 1, "custom_prompt": None}, None)
            check("refused while a different server runs", False, "no exception raised")
        except BenchmarkError as e:
            check("refused while a different server runs", "already running" in str(e), str(e))
        stop_phase_server(port2, "stop-busy.flag")

        # =============== 8) import-as-profile ==============================
        section("8) import-as-profile")
        nb = benchmark_runner.import_benchmark_as_profile(rec4["id"], mode="new")
        check("new profile created", bool(nb.get("id")), str(nb)[:200])
        check("params copied from snapshot", nb.get("params") == (snapshot.get("params") or {}),
              str(nb.get("params")))
        check("default name pattern",
              nb.get("name", "").startswith(r4["profile_name"])
              and "benchmark" in nb.get("name", "").lower(), str(nb.get("name")))
        check("new profile badge pre-populated fresh",
              (nb.get("benchmark_badge") or {}).get("state") == "fresh"
              and (nb.get("benchmark_badge") or {}).get("benchmark_id") == rec4["id"],
              str(nb.get("benchmark_badge")))
        profiles.update_profile(nb["id"], {"params": {"ctx_size": 1}})
        ob = benchmark_runner.import_benchmark_as_profile(
            rec4["id"], mode="overwrite", profile_id=nb["id"])
        check("overwrite keeps the profile id", ob.get("id") == nb["id"], str(ob.get("id")))
        check("overwrite replaces params", ob.get("params") == (snapshot.get("params") or {}),
              str(ob.get("params")))
        check("overwrite badge fresh",
              (ob.get("benchmark_badge") or {}).get("state") == "fresh",
              str(ob.get("benchmark_badge")))

        # =============== 9) app.py endpoints ===============================
        section("9) app.py endpoints")
        from backend import app as app_module
        snap2 = app_module.benchmark_snapshot(rec4["id"])
        check("snapshot endpoint returns export shape",
              snap2 == snapshot and "params" in snap2 and "id" not in snap2,
              str(snap2)[:200])
        badge = (profiles.get_profile(prof["id"]).get("benchmark_badge") or {})
        check("precondition: badge points at rec", badge.get("benchmark_id") == rec4["id"],
              str(badge))
        app_module.delete_benchmark(rec4["id"])
        badge = profiles.get_profile(prof["id"]).get("benchmark_badge")
        check("delete clears dangling badge", badge in (None, {}), str(badge))
        check("record gone from DB", bench_store.get_benchmark(rec4["id"]) is None)

        # =============== 10) startup recovery ==============================
        section("10) startup recovery of interrupted runs")
        prof = profiles.get_profile(prof["id"])
        crash_rec = bench_store.create_record(
            profile_id=prof["id"], profile_name=prof["name"], snapshot=snapshot,
            model_path=prof["model_path"], params_hash=bench_store.params_hash(prof),
            started_at=time.time())
        profiles.set_benchmark_badge(prof["id"], {
            "prefill_tps": PREFILL_TPS, "generation_tps": GEN_TPS,
            "benchmark_id": crash_rec["id"],
            "params_hash": bench_store.params_hash(prof), "state": "running",
        })
        benchmark_runner.recover_interrupted_runs()
        rc = bench_store.get_benchmark(crash_rec["id"])
        check("orphaned running record marked failed",
              rc["status"] == "failed" and "Interrupted" in (rc.get("error") or ""),
              f'{rc["status"]} - {rc.get("error")}')
        badge = (profiles.get_profile(prof["id"]).get("benchmark_badge") or {})
        check("badge repaired (state re-derived, TPS kept)",
              badge.get("state") == "fresh" and badge.get("generation_tps") == GEN_TPS,
              str(badge))

    finally:
        # ---- cleanup -------------------------------------------------------
        section("cleanup")
        try:
            manager.stop()  # belt & braces: nothing may survive the test
        except Exception:
            pass
        try:
            touch_stop("stop-run1.flag")
            touch_stop("stop-busy.flag")
            time.sleep(1)
        except Exception:
            pass
        for name in backup_names:
            shutil.copy2(backup_dir / name, DATA / name)
        (DATA / "benchmarks.json").unlink(missing_ok=True)
        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"{CHECKS[0]} checks, {len(FAILED)} failed.")
    if FAILED:
        for f in FAILED:
            print("FAILED:", f)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
