"""
benchmark_runner.py

Runs benchmarks: brings up a llama-server for the profile's (or a saved
snapshot's) exact parameters, warms it up, fires a fixed set of
/completion requests, and records the resulting prefill/generation
throughput in the benchmark DB (backend/benchmarks.py), updating the
profile's benchmark_badge as it goes (BenchPlan §3).

Server lifecycle - the app supports exactly one server instance
(process_manager), so the runner follows the existing rules rather than
spawning side instances:
  * if the target profile already owns the running server, it is REUSED
    and left running afterwards;
  * otherwise a temporary server is started with the profile's own
    start logic (same command_builder path as the UI) and stopped again
    when the run finishes.
Only one benchmark can run at a time (it needs the one server slot); a
second request is rejected with a clear message.

Timing source: the server's own /completion response metadata
(`timings.prompt_per_second` / `predicted_per_second`, plus actual token
counts), i.e. the real model running with the profile's real flags -
including its sampler defaults, since those are server-level.
"""

import asyncio
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import benchmarks as bench_store
from . import command_builder, profiles, settings
from .process_manager import binary_env, manager

READY_TIMEOUT_S = 600      # model load can be long for big models
COMPLETION_TIMEOUT_S = 900
STOP_WAIT_S = 30


class BenchmarkError(RuntimeError):
    """Benchmark failures that map to an HTTP 4xx (nothing started, or
    started-but-refused)."""


class BenchmarkCancelled(Exception):
    """Internal: the user cancelled the run."""


def _average(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _tps_from_timing(t: Dict[str, Any], per_sec_key: str, ms_key: str, n_key: str) -> Optional[float]:
    """Prefer the per-second figure the server reports; fall back to
    deriving it from *_ms/*_n on builds that omit it."""
    v = t.get(per_sec_key)
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    ms, n = t.get(ms_key), t.get(n_key)
    if isinstance(ms, (int, float)) and ms > 0 and isinstance(n, (int, float)) and n > 0:
        return n / (ms / 1000.0)
    return None


def _synthetic_prompt(target_tokens: int) -> str:
    """
    A stand-in prompt of roughly `target_tokens` tokens: numbered,
    near-unique sentences. The exact token count doesn't need to match -
    the server reports the real prompt_tokens in the response, and that's
    what gets stored; the target just picks the workload size.
    """
    per_sentence = 14  # rough token count of one sentence at this shape
    n = max(4, int((target_tokens or 512) / per_sentence) + 1)
    places = ["river", "forest", "market", "harbor", "garden", "mountain", "valley", "coast"]
    parts = [
        f"Sentence {i} describes the {places[(i - 1) % len(places)]} near town {i}, and nothing else."
        for i in range(1, n + 1)
    ]
    return " ".join(parts)


class BenchmarkRunner:
    """Owns the single in-flight benchmark (if any)."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._active_id: Optional[str] = None
        # (binary path, mtime, size) -> version string; the binary is
        # interrogated once per build, not on every run.
        self._version_cache: Dict[tuple, str] = {}

    # ---------- public ----------

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    def is_running(self) -> bool:
        return self._active_id is not None

    def cancel(self) -> None:
        """Ask the in-flight run to stop. Checked between phases; the
        request currently in flight is allowed to finish."""
        with self._lock:
            if self._active_id is None:
                raise BenchmarkError("No benchmark is currently running.")
        self._cancel_event.set()

    def start(self, source: Dict[str, Any], snapshot: Dict[str, Any],
              options: Dict[str, Any],
              loop: Optional[asyncio.AbstractEventLoop] = None) -> Dict[str, Any]:
        """
        Kick off a benchmark run in a background thread.

        `source` - the profile-shaped dict to run:
            {profile_id, name, model_path, params, custom_flags, re_ran_from?}
        `snapshot` - the export-serialization of that profile (what gets
            stored on the record); for a profile run this is exactly
            profiles.export_profile(...), per BenchPlan §5.
        `options` - {prompt_tokens, gen_tokens, repetitions, custom_prompt}.
        `loop` - the asyncio loop log subscribers live on (captured by the
            caller, which runs on the event loop).

        Returns the freshly created record (status "running"). Raises
        BenchmarkError for refusal cases (already running, no binary, ...).
        """
        with self._lock:
            if self._active_id is not None:
                raise BenchmarkError(
                    "A benchmark is already in progress. Wait for it to finish "
                    "(or cancel it) before starting another.")

        binary_path = settings.resolve_llama_server_path()
        if not binary_path or not Path(binary_path).exists():
            raise BenchmarkError("No usable llama-server binary is configured. Set one in Settings first.")
        if not (source.get("model_path") or "").strip():
            raise BenchmarkError("This profile has no model path set - nothing to benchmark.")

        # Fast, synchronous rejection of the obvious "slot is taken" case so
        # the UI gets an immediate 409 instead of a record that fails a
        # moment later. The run thread re-checks authoritatively (TOCTOU).
        self._slot_check(source, wait_for_stopping=False)

        profile_id = source.get("profile_id")
        record = bench_store.create_record(
            profile_id=profile_id,
            profile_name=source.get("name") or "(unknown profile)",
            snapshot=snapshot,
            model_path=source.get("model_path") or "",
            params_hash=bench_store.params_hash(source),
            started_at=time.time(),
            re_ran_from=source.get("re_ran_from"),
        )

        # Optimistic "running" badge (BenchPlan §3.3) - a copy of the old
        # badge is kept in memory so a failed/cancelled run can restore it.
        prev_badge: Optional[Dict[str, Any]] = None
        if profile_id:
            p = profiles.get_profile(profile_id)
            if p:
                prev_badge = p.get("benchmark_badge")
                badge = dict(prev_badge) if isinstance(prev_badge, dict) else {}
                badge.update({"state": "running",
                              "benchmark_id": record["id"],
                              "params_hash": record["params_hash"]})
                profiles.set_benchmark_badge(profile_id, badge)

        self._cancel_event.clear()
        with self._lock:
            self._active_id = record["id"]

        threading.Thread(
            target=self._run_in_thread,
            args=(record["id"], source, options, prev_badge, binary_path, loop),
            daemon=True,
        ).start()
        return record

    # ---------- the run (background thread) ----------

    def _run_in_thread(self, record_id: str, source: Dict[str, Any],
                       options: Dict[str, Any], prev_badge: Optional[Dict[str, Any]],
                       binary_path: str, loop: Optional[asyncio.AbstractEventLoop]) -> None:
        record = bench_store.get_benchmark(record_id)
        params_hash = record.get("params_hash", "") if record else ""
        started_temp_server = False
        try:
            self._set_progress(record_id, "checking the server slot")

            # --- server lifecycle (BenchPlan §3.2) -------------------------
            reusing = self._slot_check(source, wait_for_stopping=True)
            app_settings = settings.load_settings()
            params = source.get("params") or {}
            host = str(params.get("host") or app_settings.get("default_host", "127.0.0.1"))
            port = int(params.get("port") or app_settings.get("default_port", 8080))

            if not reusing:
                self._set_progress(record_id, "starting server")
                args = command_builder.build_args(
                    source["model_path"], params, source.get("custom_flags", ""),
                    host_override=host, port_override=port,
                )
                if app_settings.get("verbose"):
                    args.append("-v")
                manager.start(binary_path, args, host, port, loop,
                              mode="single", profile_id=source.get("profile_id"))
                started_temp_server = True

            self._set_progress(record_id, "waiting for the server to be ready")
            self._wait_ready()

            # Server version, captured once per binary build (BenchPlan §3.5).
            bench_store.update_record(record_id, server_version=self._server_version(binary_path))

            base_url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
            prompt = (options.get("custom_prompt") or "").strip() \
                or _synthetic_prompt(int(options.get("prompt_tokens") or 512))
            gen_tokens = int(options.get("gen_tokens") or 128)
            repetitions = max(1, int(options.get("repetitions") or 1))

            bench_start = time.time()

            # --- warmup (BenchPlan §4: starting → warming up → running) ----
            # Mirrors a measurement request exactly (same prompt, token count,
            # ignore_eos, timeout) so the one-time costs are paid before the
            # first measured repetition. Its result is discarded.
            self._check_cancel()
            self._set_progress(record_id, "warming up")
            self._completion(base_url, prompt, gen_tokens, timeout=COMPLETION_TIMEOUT_S)

            # --- measurements ------------------------------------------------
            runs: List[Dict[str, Any]] = []
            for i in range(repetitions):
                self._check_cancel()
                self._set_progress(record_id, f"running measurement {i + 1}/{repetitions}")
                resp = self._completion(base_url, prompt, gen_tokens, timeout=COMPLETION_TIMEOUT_S)
                timings = resp.get("timings") or {}
                runs.append({
                    "prompt_tokens": resp.get("n_prompt_tokens", timings.get("prompt_n")),
                    "gen_tokens": resp.get("n_predicted_tokens", timings.get("predicted_n")),
                    "prefill_tps": _tps_from_timing(timings, "prompt_per_second", "prompt_ms", "prompt_n"),
                    "generation_tps": _tps_from_timing(timings, "predicted_per_second", "predicted_ms", "predicted_n"),
                })

            prefill = _average([r["prefill_tps"] for r in runs])
            gen = _average([r["generation_tps"] for r in runs])
            if prefill is None or gen is None:
                raise BenchmarkError(
                    "The server did not report usable timing data (is it a recent llama.cpp build?)")

            self._set_progress(record_id, "finishing")
            bench_store.update_record(
                record_id,
                status="completed",
                progress="completed",
                prefill_tps=round(prefill, 3),
                generation_tps=round(gen, 3),
                prompt_tokens=int(runs[0]["prompt_tokens"] or 0) or None,
                gen_tokens=int(sum(int(r["gen_tokens"] or 0) for r in runs)) or None,
                duration_s=round(time.time() - bench_start, 3),
                timestamp=time.time(),
                raw_output=json.dumps({
                    "reused_server": reusing,
                    "options": options,
                    "runs": runs,
                }, indent=2),
            )
            self._finish_badge(source.get("profile_id"), record_id, params_hash,
                               round(prefill, 3), round(gen, 3))
        except BenchmarkCancelled:
            bench_store.update_record(
                record_id, status="cancelled", progress="cancelled",
                error="Cancelled by the user.", timestamp=time.time())
            self._restore_badge(source.get("profile_id"), record_id, params_hash, prev_badge)
        except Exception as e:  # noqa: BLE001 - any failure lands in the record, not the app
            bench_store.update_record(
                record_id, status="failed", progress="failed",
                error=str(e) or e.__class__.__name__, timestamp=time.time())
            self._restore_badge(source.get("profile_id"), record_id, params_hash, prev_badge)
        finally:
            if started_temp_server:
                # Cleanup (BenchPlan §3.8): a server we started solely for
                # this benchmark comes back down. Reused user servers stay up.
                manager.stop()
                self._wait_stopped(STOP_WAIT_S)
            with self._lock:
                if self._active_id == record_id:
                    self._active_id = None

    # ---------- badge handling ----------

    def _finish_badge(self, profile_id: Optional[str], record_id: str, params_hash: str,
                      prefill: float, gen: float) -> None:
        """On completion: new TPS + hash on the badge. Freshness is decided
        against the profile's *current* params, so if the user edited them
        mid-run the badge lands as "stale" with the new numbers (BenchPlan
        §6: the run used the start-time snapshot, staleness checks the
        post-edit params)."""
        if not profile_id:
            return
        p = profiles.get_profile(profile_id)
        if not p:
            return  # profile deleted mid-run - record keeps its name snapshot
        state = "fresh" if bench_store.params_hash(p) == params_hash else "stale"
        profiles.set_benchmark_badge(profile_id, {
            "prefill_tps": prefill,
            "generation_tps": gen,
            "benchmark_id": record_id,
            "params_hash": params_hash,
            "state": state,
        })

    def _restore_badge(self, profile_id: Optional[str], record_id: str, params_hash: str,
                       prev_badge: Optional[Dict[str, Any]]) -> None:
        """On failure/cancel: never silently lose the last-good badge
        (BenchPlan §3.7). Restore the previous badge (re-checking its
        freshness, since params may have changed), or create a "failed"
        badge if there was nothing worth keeping."""
        if not profile_id:
            return
        p = profiles.get_profile(profile_id)
        if not p:
            return
        if isinstance(prev_badge, dict) and (
                prev_badge.get("prefill_tps") is not None
                or prev_badge.get("generation_tps") is not None):
            badge = dict(prev_badge)
            badge["state"] = ("fresh" if bench_store.params_hash(p) == badge.get("params_hash")
                              else "stale")
        else:
            badge = {
                "prefill_tps": None,
                "generation_tps": None,
                "benchmark_id": record_id,
                "params_hash": params_hash,
                "state": "failed",
            }
        profiles.set_benchmark_badge(profile_id, badge)

    # ---------- internals ----------

    def _slot_check(self, source: Dict[str, Any], wait_for_stopping: bool) -> bool:
        """
        Decide the server-slot situation for a run:
          * returns True when the target profile already owns the running
            server and it can be reused (live-profile runs only - a re-run
            must benchmark the SAVED snapshot, not the live params);
          * raises BenchmarkError when a different server occupies the slot;
          * when a server is "stopping", waits it out - unless
            wait_for_stopping is False (start() must stay fast; the run
            thread will do the waiting).
        """
        st = manager.status()
        reusing = (
            not source.get("re_ran_from")
            and source.get("profile_id")
            and st.get("profile_id") == source.get("profile_id")
            and st["state"] in ("starting", "running")
        )
        if not reusing and st["state"] in ("starting", "running"):
            raise BenchmarkError(
                "A server is already running. Stop it first (Server Console) "
                "to run a benchmark.")
        if st["state"] == "stopping":
            if not wait_for_stopping:
                return False
            if not self._wait_stopped(STOP_WAIT_S):
                raise BenchmarkError("The previous server is taking too long to stop.")
        return reusing

    def _set_progress(self, record_id: str, progress: str) -> None:
        bench_store.update_record(record_id, progress=progress)

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise BenchmarkCancelled()

    def _wait_ready(self, timeout: float = READY_TIMEOUT_S) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._check_cancel()
            st = manager.status()
            if st["state"] == "running":
                return
            if st["state"] in ("stopped", "error"):
                detail = (st.get("error_message")
                          or "The llama-server exited while starting. Check the Server Console logs.")
                # Surface the actual server output for generic exit codes
                # (e.g. "error: invalid argument: --x" for bad params).
                log_tail = [l for l in manager.recent_logs(10) if not l.startswith("[model-manager]")][-3:]
                if log_tail:
                    detail += " Server log: " + " … ".join(log_tail)
                raise BenchmarkError(detail[:1200])
            time.sleep(0.5)
        raise BenchmarkError("Timed out waiting for the llama-server to be ready.")

    def _wait_stopped(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if manager.status()["state"] == "stopped":
                return True
            time.sleep(0.3)
        return False

    @staticmethod
    def _completion(base_url: str, prompt: str, max_tokens: int, timeout: float) -> Dict[str, Any]:
        """One /completion request (non-streaming). ignore_eos forces the
        full max_tokens so every repetition measures the same workload."""
        body = json.dumps({
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "stream": False,
            "ignore_eos": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            base_url + "/completion", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            raise BenchmarkError(f"The server rejected the benchmark request (HTTP {e.code} {detail})".strip())
        except OSError as e:
            raise BenchmarkError(f"Could not reach the server at {base_url}: {e}")

    def _server_version(self, binary_path: str) -> str:
        """What `llama-server --version` reports (first line, e.g.
        'version: 0.1.2-dev (build 10488, commit 9d77fa172)'). Cached per
        binary path + build (mtime/size), so repeated runs don't
        re-interrogate the binary. Best effort - an empty string is fine."""
        p = Path(binary_path)
        try:
            st = p.stat()
            key = (str(p), st.st_mtime, st.st_size)
        except OSError:
            return ""
        if key in self._version_cache:
            return self._version_cache[key]
        line = ""
        try:
            out = subprocess.run([str(p), "--version"], capture_output=True, text=True, timeout=20,
                                 env=binary_env(p))
            for raw in ((out.stdout or "") + (out.stderr or "")).splitlines():
                s = " ".join(raw.split())
                if s:
                    line = s[:200]
                    break
        except Exception:
            pass
        self._version_cache[key] = line
        return line


# Module-level singleton - the app has one server slot, so one runner.
runner = BenchmarkRunner()


# ---------------------------------------------------------------------------
# Import a benchmark record back as a profile (BenchPlan §5.2)
# ---------------------------------------------------------------------------

def import_benchmark_as_profile(benchmark_id: str, mode: str = "new",
                                name: Optional[str] = None,
                                profile_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new profile from a record's params snapshot, or overwrite an
    existing one. The new profile's benchmark_badge is pre-populated as
    "fresh" (its params match the snapshot exactly, so the hash matches by
    construction). Raises ValueError for refusal cases.
    """
    rec = bench_store.get_benchmark(benchmark_id)
    if not rec:
        raise ValueError("Benchmark not found.")
    snap = rec.get("profile_params_snapshot") or {}
    model_id = snap.get("model_id") or ""
    model_path = snap.get("model_path") or ""
    if not model_path:
        raise ValueError("This benchmark record has no model path - nothing to import.")

    badge = {
        "prefill_tps": rec.get("prefill_tps"),
        "generation_tps": rec.get("generation_tps"),
        "benchmark_id": rec["id"],
        "params_hash": rec.get("params_hash") or "",
        "state": "fresh",
    }

    if mode == "overwrite":
        if not profile_id:
            raise ValueError("Pick the profile to overwrite.")
        if not profiles.get_profile(profile_id):
            raise ValueError("The profile to overwrite doesn't exist anymore.")
        profiles.replace_profile(profile_id, {
            "model_id": model_id,
            "model_path": model_path,
            "params": snap.get("params") or {},
            "custom_flags": snap.get("custom_flags") or "",
            "notes": snap.get("notes") or "",
        })
        profiles.set_benchmark_badge(profile_id, badge)
        return profiles.get_profile(profile_id) or {}

    if mode != "new":
        raise ValueError("mode must be 'new' or 'overwrite'.")

    base_name = rec.get("profile_name") or "Imported"
    when = time.strftime("%Y-%m-%d",
                         time.localtime(rec.get("timestamp") or rec.get("started_at") or time.time()))
    new_name = (name or "").strip() or f"{base_name} (from benchmark {when})"
    p = profiles.create_profile(
        model_id=model_id,
        model_path=model_path,
        name=new_name,
        params=snap.get("params") or {},
        custom_flags=snap.get("custom_flags") or "",
        notes=snap.get("notes") or "",
    )
    profiles.set_benchmark_badge(p["id"], badge)
    # Re-fetch: create_profile returns the pre-badge dict.
    return profiles.get_profile(p["id"]) or p


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------

def recover_interrupted_runs() -> None:
    """Called once at app import: a record still marked "running" was
    orphaned by an app restart (its thread is gone). Mark it failed and
    repair the profile badge it left on "running"."""
    for rec in bench_store.list_benchmarks():
        if rec.get("status") != "running":
            continue
        bench_store.update_record(
            rec["id"], status="failed", progress="failed",
            error="Interrupted - the app was restarted while the benchmark was running.",
            timestamp=time.time(),
        )
        pid = rec.get("profile_id")
        if not pid:
            continue
        p = profiles.get_profile(pid)
        if not p:
            continue
        badge = p.get("benchmark_badge")
        if not isinstance(badge, dict) or badge.get("benchmark_id") != rec["id"]:
            continue
        if badge.get("prefill_tps") is not None or badge.get("generation_tps") is not None:
            badge["state"] = ("fresh" if bench_store.params_hash(p) == badge.get("params_hash")
                              else "stale")
        else:
            badge = {
                "prefill_tps": None,
                "generation_tps": None,
                "benchmark_id": rec["id"],
                "params_hash": rec.get("params_hash") or "",
                "state": "failed",
            }
        profiles.set_benchmark_badge(pid, badge)


recover_interrupted_runs()
