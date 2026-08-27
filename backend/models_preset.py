"""
models_preset.py

Generates the models.ini consumed by llama-server's router mode
(``llama-server --models-preset <file>``) from a router preset:
one INI section per profile.

INI format (see the llama.cpp server README, "Model presets"):
  - ``version = 1`` header
  - optional ``[*]`` global section (not used here - profiles are
    self-contained)
  - one section per model. The *section name* becomes the model's API name
    (the value clients put in the ``"model"`` field of requests).
  - keys are CLI arguments without leading dashes; bool flags use
    ``= true`` (negation would need the ``no-`` prefix form, but our
    schema's store-true flags are simply omitted when false)
  - preset-only keys: ``load-on-startup`` (bool), ``stop-timeout`` (int)

Args the *router itself* controls (host, port, API key, model alias, HF
repo) are stripped from sections on load, so we skip them here with a note.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import command_builder, gguf_utils, param_schema, presets as presets_store

ROUTER_DIR = Path(__file__).resolve().parent.parent / "data" / "router"

# Profile param keys the router takes over - setting them per model would be
# silently discarded by llama-server, so we skip them and tell the user.
ROUTER_CONTROLLED_KEYS = {"host", "port", "api_key"}

# Custom flags that are plain on/off switches (from the schema). Raw
# custom-flag text can't be type-checked, so these (and their ``no-``
# counterparts) are emitted as bare ``flag = true``; anything else is
# assumed to take a value.
_BOOL_FLAGS = {
    p["flag"].lstrip("-")
    for p in param_schema.PARAMETERS
    if p["type"] == "bool"
}


def _ini_key(flag: str) -> str:
    """``--n-gpu-layers`` / ``-ngl`` -> ``n-gpu-layers`` / ``ngl``."""
    return flag.lstrip("-")


def _truthy(value: Any) -> bool:
    """Bool-ish check that isn't fooled by the string 'False'/'0'."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return bool(value)


def _sanitize_section_name(name: str, fallback: str) -> str:
    """The section name doubles as the model's API identifier, and the
    router's name parser is picky: INI metachars (``[ ] ; #``), newlines
    AND spaces all make it silently fail to register the model (requests
    then die with "model '…' not found"). Replace any of those with a
    dash so the generated name is always accepted."""
    cleaned = re.sub(r"[\[\];#\s]+", "-", (name or "").strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("- ")
    return cleaned or fallback


def _model_stems(model_path: str) -> List[str]:
    """Name stems to try when matching sibling mmproj files, longest first:
    full stem, multipart suffix stripped, trailing quant token stripped
    (llava-7b-q4 -> llava-7b matches llava-7b-mmproj-f16.gguf)."""
    p = Path(model_path)
    stems = [p.stem]
    base, _part = gguf_utils.split_multipart(p.stem)
    if base:
        stems.append(base)
    for s in list(stems):
        quant = gguf_utils.parse_quant(s)
        if quant and s.upper().endswith("-" + quant):
            stems.append(s[: -(len(quant) + 1)])
    out: List[str] = []
    for s in stems:
        if s and s not in out:
            out.append(s)
    return out


def _find_mmproj(model_path: str) -> Optional[str]:
    """Heuristic mirroring llama.cpp's --mmproj-auto: find the projector
    next to the model. Prefers a file whose name contains the model's own
    stem (handles folders holding several models); falls back to the single
    *mmproj*.gguf in the directory if there's exactly one."""
    model_dir = Path(model_path).parent
    try:
        candidates = [str(p) for p in model_dir.glob("*mmproj*.gguf")]
    except OSError:
        return None
    if not candidates:
        return None
    for stem in _model_stems(model_path):
        stem_match = [c for c in candidates if stem in Path(c).name]
        if stem_match:
            return stem_match[0]
    return candidates[0] if len(candidates) == 1 else None


def _section_lines(profile: Dict[str, Any], load_on_startup: bool) -> tuple:
    """Returns (lines, notes) for one profile's INI section."""
    lines: List[str] = []
    notes: List[str] = []

    model_path = profile.get("model_path", "")
    if not model_path:
        return lines, ["profile has no model path"]
    lines.append(f"model = {model_path}")

    # Only auto-attach when the profile doesn't set --mmproj itself, so the
    # INI never carries two ``mmproj =`` lines for the same section.
    params_map = profile.get("params") or {}
    has_explicit_mmproj = any(
        k == "mmproj" and v is not None and v != "" for k, v in params_map.items()
    )
    mmproj = None if has_explicit_mmproj else _find_mmproj(model_path)
    if mmproj:
        lines.append(f"mmproj = {mmproj}")
        notes.append(f"attached mmproj: {Path(mmproj).name}")

    schema_by_key = {p["key"]: p for p in param_schema.PARAMETERS}
    ignored = []
    for key, value in (profile.get("params") or {}).items():
        if value is None or value == "":
            continue
        if key in ROUTER_CONTROLLED_KEYS:
            ignored.append(key)
            continue
        p = schema_by_key.get(key)
        if p is None:
            continue  # unknown key - not in the schema, can't map to a flag
        if key == "mmproj":
            # The router resolves INI paths against its working directory,
            # so a bare file name must be turned into the model-folder path.
            value = command_builder.resolve_mmproj_path(model_path, value)
        flag_key = _ini_key(p["flag"])
        if p["type"] == "bool":
            if _truthy(value):
                lines.append(f"{flag_key} = true")
            # false === absent for store-true flags
        else:
            lines.append(f"{flag_key} = {value}")
    if ignored:
        notes.append(f"ignored router-controlled params: {', '.join(sorted(ignored))}")

    # Raw custom flags: ``--flag value`` / ``-f value`` -> ``flag = value``;
    # known bare bool flags -> ``flag = true``.
    tokens = command_builder.split_custom_flags(profile.get("custom_flags") or "")
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            i += 1
            continue  # stray value without a flag - skip
        flag_key = _ini_key(tok)
        if flag_key in _BOOL_FLAGS or flag_key.startswith("no-"):
            lines.append(f"{flag_key} = true")
            i += 1
        elif i + 1 < len(tokens):
            lines.append(f"{flag_key} = {tokens[i + 1]}")
            i += 2
        else:
            lines.append(f"{flag_key} = true")  # dangling flag, assume on/off
            i += 1

    lines.append(f"load-on-startup = {'true' if load_on_startup else 'false'}")
    return lines, notes


def _default_lines(raw: str) -> List[str]:
    """Sanitize the user's raw "key = value" text for the [*] section:
    drop blanks and - importantly - any stray section header, which would
    otherwise split the INI into an unintended extra model entry."""
    lines: List[str] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s or s.startswith("["):
            continue
        lines.append(s)
    return lines


def generate(preset: Dict[str, Any], all_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the INI text (and launch info) for a preset from *live* profiles.

    Returns:
        {
            "ini": str,
            "models": [{"section", "profile_id", "profile_name", "model_path", "notes"}],
            "warnings": [str],
        }
    """
    by_id = {p.get("id"): p for p in (all_profiles or [])}
    models = []
    section_lines: List[tuple] = []
    warnings: List[str] = []
    seen_paths: Dict[str, str] = {}

    for idx, profile_id in enumerate(preset.get("profile_ids") or []):
        profile = by_id.get(profile_id)
        if profile is None:
            warnings.append(f"Profile {profile_id} no longer exists - skipped.")
            continue

        model_path = profile.get("model_path", "")
        if model_path and model_path in seen_paths:
            warnings.append(
                f"'{profile.get('name')}' points to the same model file as "
                f"'{seen_paths[model_path]}' - running the same file twice is pointless."
            )
        elif model_path:
            seen_paths[model_path] = profile.get("name", "?")

        name = profile.get("name") or f"model-{idx + 1}"
        section = _sanitize_section_name(name, f"model-{idx + 1}")
        lines, notes = _section_lines(profile, preset.get("load_on_startup", True))
        models.append({
            "section": section,
            "profile_id": profile_id,
            "profile_name": name,
            "model_path": model_path,
            "notes": notes,
        })
        section_lines.append((section, lines))

    ini_lines = ["version = 1"]
    default_lines = _default_lines(preset.get("defaults") or "")
    if default_lines:
        ini_lines.append("")
        ini_lines.append("[*]")
        ini_lines.extend(default_lines)
    ini_lines.append("")
    for section, lines in section_lines:
        ini_lines.append(f"[{section}]")
        ini_lines.extend(lines)
        ini_lines.append("")
    ini_text = "\n".join(ini_lines).rstrip() + "\n"

    return {"ini": ini_text, "models": models, "warnings": warnings}


def _cli_args(preset: Dict[str, Any], ini_path: Path) -> List[str]:
    args = ["--models-preset", str(ini_path)]
    args += ["--models-max", str(int(preset.get("models_max", 4)))]
    if not preset.get("autoload", True):
        args.append("--no-models-autoload")
    return args


def preview(preset: Dict[str, Any], all_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """generate() plus the CLI args that would be used at start (with the
    INI path the file would be written to), for the UI preview. No files
    are written."""
    built = generate(preset, all_profiles)
    built["args"] = _cli_args(preset, ROUTER_DIR / f"{preset['id']}.ini")
    return built


def build_launch_args(preset: Dict[str, Any], all_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Like generate(), but also writes the INI to data/router/<preset_id>.ini
    and returns the CLI args (without the binary, without host/port) to pass
    to llama-server. Raises ValueError if the preset yields no models.
    """
    built = generate(preset, all_profiles)
    if not built["models"]:
        raise ValueError(
            "This preset has no usable profiles (every referenced profile is missing). "
            "Edit the preset and add at least one profile."
        )

    presets_store.ensure_data_dir()
    ROUTER_DIR.mkdir(parents=True, exist_ok=True)
    ini_path = ROUTER_DIR / f"{preset['id']}.ini"
    with open(ini_path, "w", encoding="utf-8") as f:
        f.write(built["ini"])

    built["ini_path"] = str(ini_path)
    built["args"] = _cli_args(preset, ini_path)
    return built


def _flag_supported(binary_path: str, needle: str, cache: Dict[Any, Optional[bool]]) -> Optional[bool]:
    """
    Check whether a flag is present in the binary's --help output. Cached per
    (path, mtime, size). Returns True/False, or None when unknown (no binary
    configured, or the check itself failed) - in that case startup proceeds
    and the normal "not configured" / crash diagnostics apply.
    """
    import subprocess

    if not binary_path:
        return None
    p = Path(binary_path)
    if not p.exists():
        return None  # let startup produce the proper "not found" error
    try:
        stat = p.stat()
        key = (str(p), stat.st_mtime, stat.st_size)
    except OSError:
        return False
    if key in cache:
        return cache[key]

    result: Optional[bool] = None
    try:
        out = subprocess.run(
            [str(p), "--help"], capture_output=True, text=True, timeout=20,
        )
        combined = (out.stdout or "") + (out.stderr or "")
        result = needle in combined
    except Exception:
        result = None
    cache[key] = result
    return result


def router_mode_supported(binary_path: str) -> Optional[bool]:
    """Does the binary support router mode via --models-preset?"""
    return _flag_supported(binary_path, "--models-preset", router_mode_supported._cache)


def router_dir_supported(binary_path: str) -> Optional[bool]:
    """Does the binary support router mode via --models-dir?"""
    return _flag_supported(binary_path, "--models-dir", router_dir_supported._cache)


router_mode_supported._cache = {}  # type: ignore[attr-defined]
router_dir_supported._cache = {}  # type: ignore[attr-defined]
