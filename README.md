# Llama Profile Manager (LPM)

A local desktop app for browsing GGUF models on disk, building `llama-server` launch profiles, and starting/stopping/monitoring the server.

Built with Python + FastAPI (backend) and pywebview (native window) + plain HTML/CSS/JS (frontend, no Node/build step).

<img width="1486" height="953" alt="image" src="https://github.com/user-attachments/assets/261a7645-92d7-43a6-983e-7933a0c8cafe" />
<img width="1486" height="953" alt="image" src="https://github.com/user-attachments/assets/25347c4d-96e1-4c34-9220-c4d88c713b87" />
<img width="1486" height="953" alt="image" src="https://github.com/user-attachments/assets/c94b6ab3-fa7f-4802-8ad4-be71e7f458f5" />
<img width="1486" height="953" alt="image" src="https://github.com/user-attachments/assets/2cf36e86-7b39-4603-9598-1ef93930db9d" />

## Features

- **Model library** - scans root folders for `.gguf` files, groups multi-part files, parses quant type, caches scans. Quick links to folder and HF repo page.
- **Download page** - search Hugging Face for GGUF models or paste a repo id/URL. Lists quants with size/type badges, marks ones you already have, downloads with progress bar.
- **Launch profiles** - multiple named profiles per model (JSON on disk). Export/Import (single, bulk, or bare list with per-row matching), Export all, Copy to model.
- **Parameter configurator** - schema of `llama-server` CLI flags, grouped/collapsible, with tooltips, search, live command preview, and an advanced free-text field for raw flags.
- **Server control** - start/stop, live status (PID/uptime), live log streaming via WebSocket, readable error messages, link to server web UI.
- **Router Presets** (`--models-preset`) - bind several profiles to one router endpoint with shared defaults, optional `models_max` cap and autoload; live model list with load/unload.
- **Router Dir** (`--models-dir`) - auto-discovers all GGUFs in a folder with shared parameters; same controls as Router Presets.
- **Benchmarks** - measures prefill/generation throughput per profile, stores results separately from profiles, TPS badges (green/amber/blue/red), run history with re-run/import/export/delete, and build/version tracking per run.
- **Settings** - manage `llama-server` builds (with one-click latest-build download), model root folders, host/port, logging, theme, optional LAN access (off by default), app version/update badge.

## Requirements

- Python 3.11+
- `llama-server` binary from [llama.cpp](https://github.com/ggml-org/llama.cpp) - build it, grab a release, or use Settings → **Download latest**. Not bundled with the app.

## Setup

### Linux
Not working yes, fixing it ASAP

```bash
git clone https://github.com/BrainCrashSoft/llama-profile-manager.git
cd llama-profile-manager
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

pywebview needs a WebKit backend:

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-webkit2-4.1   # Debian/Ubuntu
```

Falls back to your default browser if pywebview can't start.

### Windows

```powershell
git clone https://github.com/BrainCrashSoft/llama-profile-manager.git
cd llama-profile-manager
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Uses the built-in Edge WebView2 runtime. Once the venv exists, `start.bat` is a one-click launcher.

## First run

1. In **Settings**, download or add a `llama-server` build (this becomes the active one).
2. Add model root folder(s) - settings autosave, triggers a rescan.
3. On **Profiles**, pick a model → **+ Profile** to open the parameter configurator.
4. Adjust parameters, name the profile, **Save Profile**.
5. Select the profile → **▶ Start**. Watch logs in **Server Console**.

## Updating

```bash
git pull
pip install -r requirements.txt   # only if it changed
```

`data/` (settings, profiles, builds) is gitignored and untouched by updates. Settings shows a "new version available" badge based on GitHub Releases; update manually via `git pull` (no auto-update).

## Versioning

Follows [SemVer](https://semver.org/) (pre-1.0: MINOR = features, PATCH = fixes). Version lives in `backend/__init__.py` (`__version__`); releases are tagged `vX.Y.Z`. See `CHANGELOG.md`. The `"version": 1` field in Export-all files is a separate data-format version.

## Project structure

```
llama-profile-manager/
├── main.py                       # entrypoint: FastAPI + pywebview window
├── backend/
│   ├── app.py                    # FastAPI app + routes
│   ├── scanner.py                # filesystem scanning
│   ├── gguf_utils.py             # quant-name parsing / multi-part grouping
│   ├── gguf_meta.py              # GGUF metadata (arch, context, blocks)
│   ├── hf_client.py              # HF repo lookup (available quants)
│   ├── hf_search.py              # HF model search
│   ├── hf_avatar.py              # org/user avatar lookup
│   ├── download_manager.py       # background GGUF download + progress
│   ├── app_release.py            # in-app "new LPM version" check
│   ├── llama_release.py          # latest llama.cpp release lookup
│   ├── llama_server_download.py  # one-click llama-server install
│   ├── profiles.py               # profile CRUD, import/export
│   ├── benchmarks.py             # benchmark record store
│   ├── benchmark_runner.py       # benchmark runs: temp server, timing
│   ├── param_schema.py           # llama.cpp parameter definitions
│   ├── command_builder.py        # profile params -> llama-server argv
│   ├── process_manager.py        # subprocess launch/monitor/stop
│   ├── models_preset.py          # --models-preset command building
│   ├── presets.py                # router preset CRUD
│   ├── router_dirs.py            # --models-dir preset CRUD
│   └── settings.py               # app settings persistence
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
├── assets/                        # app icons
├── data/                          # user config, profiles, cache (gitignored)
├── tests/                         # end-to-end tests
├── requirements.txt
└── start.bat                      # Windows one-click launcher
```

## Notes & limitations

- **Custom flags** in Advanced mode are tokenized with the platform's native CLI rules (Windows argv parsing / POSIX shell quoting), not a generic shell parser.
- **Parameter schema migration** - `--webui` and `-cb`/`--cont-batching` were removed (already defaults; use `--no-webui`/`-nocb`); `--mlock`/`--no-mmap` replaced by `-lm, --load-mode`. Old profiles using the retired keys need a manual check.
- **Search fields** treat space-separated words as AND. HF's own search (Download page) uses its own ranking via their API.
- **Remote access** - app UI binds to `127.0.0.1` by default; optional toggle for `0.0.0.0`. No login - only enable on trusted networks. Takes effect on restart. Separate from llama-server's own `--host`/`--port`.
- **HF search** uses `huggingface_hub`, restricted to `gguf`-tagged repos, capped at 50 results (default 10).
- **Downloads** - up to 3 concurrent jobs, extras queue. Multi-part quants download sequentially as one job. Files stream with a `.part` suffix until complete. Public API only - no private/gated repos.
- **Settings autosave** - no Save button; debounced ~0.5s for text, immediate for folders/theme.
- **Native file/folder pickers** only work in the pywebview window (`window.pywebview.api`); browser fallback uses manual path entry.
- **Profile import/export** also uses the native-dialog bridge in pywebview; browser fallback uses standard downloads/file-picker.
- **Single server instance** - only one `llama-server` runs at a time; starting another while one's active errors out.
- **No auth beyond localhost binding** - single-user tool. If exposing llama-server itself on your LAN, set an API key in the profile.
- **Parameter schema drift** - schema reflects llama-server's flags as of build time. Use Advanced mode for new flags, or extend `param_schema.py`.
- **Packaging** - structured for future PyInstaller freezing, not yet implemented.
