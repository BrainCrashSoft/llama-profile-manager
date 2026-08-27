"""
command_builder.py

Turns a profile's structured params (+ optional raw custom flags) into the
actual argv list passed to subprocess.Popen, using param_schema.py to know
each parameter's CLI flag and type.
"""

import platform
import shlex
from typing import Any, Dict, List

from . import param_schema, scanner


def resolve_mmproj_path(model_path: str, value: Any) -> Any:
    """
    A profile may store a bare file name for --mmproj (that's what the
    editor's field shows). Resolve it against the model's own folder, where
    projector files conventionally live. Values that already contain a path
    separator - or that have no match next to the model - are returned
    unchanged, so the command still gets exactly what the user typed.
    """
    if not isinstance(value, str) or not value.strip() or not model_path:
        return value
    if "/" in value or "\\" in value:
        return value
    for f in scanner.find_mmproj_files(model_path):
        if f["filename"].lower() == value.strip().lower():
            return f["path"]
    return value


def resolve_chat_template_path(model_path: str, value: Any) -> Any:
    """
    A profile may store a bare file name for --chat-template-file (that's
    what the editor's field shows). Resolve it against the model's own
    folder, where custom chat-template files (.jinja) conventionally live.
    Values that already contain a path separator - or that have no match
    next to the model - are returned unchanged, so the command still gets
    exactly what the user typed.
    """
    if not isinstance(value, str) or not value.strip() or not model_path:
        return value
    if "/" in value or "\\" in value:
        return value
    for f in scanner.find_chat_template_files(model_path):
        if f["filename"].lower() == value.strip().lower():
            return f["path"]
    return value


def build_args(model_path: str, params: Dict[str, Any], custom_flags: str = "",
                host_override: str = None, port_override: int = None) -> List[str]:
    """
    Build the full argv (excluding the binary itself) for launching llama-server.

    `params` is keyed by param_schema keys (e.g. "ctx_size"), with values
    already in their final type (bool/int/float/str).
    """
    args: List[str] = ["--model", model_path]
    args += _emit_param_args(params, model_path)

    # host/port overrides (e.g. from the server-control panel) take precedence
    # over whatever is baked into the profile's own params.
    if host_override:
        _replace_or_append(args, "--host", host_override)
    if port_override:
        _replace_or_append(args, "--port", str(port_override))

    if custom_flags:
        args.extend(split_custom_flags(custom_flags))

    return args


def _emit_param_args(params: Dict[str, Any], model_path: str = "") -> List[str]:
    """
    Turn a param dict (keyed by param_schema keys) into its CLI flags, in the
    editor's display order (schema order, unknown keys last). Shared by the
    single-model and router-dir command builders so both emit flags
    identically. ``model_path`` is only used to resolve a bare --mmproj file
    name; pass "" when there is no single model (router dir) - the value is
    then left exactly as stored.
    """
    args: List[str] = []
    schema_by_key = {p["key"]: p for p in param_schema.PARAMETERS}
    params = params or {}

    # Emit flags in the editor's display order (schema order) rather than
    # the order the profile's params dict happens to store them, so the
    # command preview matches the Parameters list shown in the UI.
    # Unknown keys (skipped below) are ordered last but never emitted.
    ordered_keys = [p["key"] for p in param_schema.PARAMETERS if p["key"] in params]
    ordered_keys += [k for k in params if k not in schema_by_key]

    for key in ordered_keys:
        value = params[key]
        p = schema_by_key.get(key)
        if p is None:
            continue  # unknown key, silently skip (should not happen from the UI)
        if value is None or value == "":
            continue

        flag = p["flag"]
        ptype = p["type"]

        # A bare file name in a model-folder file param only makes sense
        # next to the model; llama-server resolves it against its own
        # working directory, so turn it into the real path here (also
        # covers profiles saved before the editor started resolving it).
        if key == "mmproj":
            value = resolve_mmproj_path(model_path, value)
        elif key == "chat_template_file":
            value = resolve_chat_template_path(model_path, value)

        if ptype == "bool":
            if value:
                args.append(flag)
            # llama.cpp flags are store-true style; "off" just means we omit the flag.
        else:
            args.append(flag)
            args.append(str(value))

    return args


def build_router_dir_args(models_dir: str, params: Dict[str, Any], custom_flags: str = "",
                          models_max: int = 4, autoload: bool = True,
                          host_override: str = None, port_override: int = None) -> List[str]:
    """
    Build the full argv (excluding the binary itself) for launching a
    router-mode server that auto-discovers every model in a directory:

        --models-dir <path> --models-max N [--no-models-autoload] <defaults>…

    ``params`` are the *shared* default parameters applied to every model in
    the folder. host/port are router-level, so the per-model --host/--port
    params (if any) are dropped in favour of the explicit override/app
    default, matching how the --models-preset router is started.
    """
    args: List[str] = ["--models-dir", models_dir]
    args += ["--models-max", str(int(models_max))]
    if not autoload:
        args.append("--no-models-autoload")

    p = dict(params or {})
    p.pop("host", None)
    p.pop("port", None)
    args += _emit_param_args(p, "")

    if host_override:
        _replace_or_append(args, "--host", host_override)
    if port_override:
        _replace_or_append(args, "--port", str(port_override))

    if custom_flags:
        args.extend(split_custom_flags(custom_flags))

    return args


def split_custom_flags(custom_flags: str) -> List[str]:
    """
    Tokenize the raw "custom flags" text the same way the *target platform's*
    own command line would. This matters because subprocess.Popen on Windows
    re-serializes a list of args back into a single command-line string via
    list2cmdline() before CreateProcess hands it to the child, and the child
    (a standard C/C++ program like llama-server.exe) then re-parses that
    string using the Windows C runtime's argv rules - which are NOT the same
    as POSIX shell quoting. Using shlex (POSIX-oriented) on Windows silently
    mangles anything with escaped quotes, e.g. JSON embedded in a flag value
    like --chat-template-kwargs "{\\"reasoning_effort\\":\\"medium\\"}",
    which is exactly what a user would type at a real Windows command
    prompt. See _windows_cmdline_split for the Windows-specific algorithm.
    """
    if platform.system() == "Windows":
        return _windows_cmdline_split(custom_flags)
    try:
        return shlex.split(custom_flags, posix=True)
    except ValueError:
        # Unbalanced quotes etc. - fall back to a naive split rather than crashing.
        return custom_flags.split()


def _windows_cmdline_split(s: str) -> List[str]:
    """
    Tokenize a string using the documented Microsoft C runtime command-line
    parsing rules (the same rules CommandLineToArgvW and every standard
    C/C++ program's argv follow), rather than shell-style quoting:
      - Whitespace outside quotes separates arguments.
      - A double quote toggles "inside quotes" mode; while inside, whitespace
        is literal.
      - A run of backslashes followed by a double quote collapses to half as
        many literal backslashes; if the run's length is odd, one literal
        double quote is emitted and the quote does NOT toggle mode; if even,
        the quote toggles mode as usual.
      - A run of backslashes NOT followed by a double quote is entirely
        literal.
    This is the correct inverse of Python's own subprocess.list2cmdline(),
    which is what actually builds the string CreateProcess sees - so
    round-tripping through this parser and list2cmdline reproduces exactly
    what the user typed.
    """
    args: List[str] = []
    current: List[str] = []
    in_quotes = False
    started = False
    i = 0
    n = len(s)

    while i < n:
        c = s[i]

        if c.isspace() and not in_quotes:
            if started:
                args.append("".join(current))
                current = []
                started = False
            i += 1
            continue

        started = True

        if c == "\\":
            run_start = i
            while i < n and s[i] == "\\":
                i += 1
            num_backslashes = i - run_start
            if i < n and s[i] == '"':
                current.append("\\" * (num_backslashes // 2))
                if num_backslashes % 2 == 1:
                    current.append('"')
                    i += 1
                else:
                    in_quotes = not in_quotes
                    i += 1
            else:
                current.append("\\" * num_backslashes)
            continue

        if c == '"':
            in_quotes = not in_quotes
            i += 1
            continue

        current.append(c)
        i += 1

    if started:
        args.append("".join(current))

    return args


def _replace_or_append(args: List[str], flag: str, value: str) -> None:
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            args[idx + 1] = value
            return
    args.append(flag)
    args.append(value)


def preview_command(binary_display_name: str, model_path: str, params: Dict[str, Any],
                     custom_flags: str = "", host_override: str = None,
                     port_override: int = None) -> str:
    """Human-readable command line string, for the live preview in the UI."""
    args = build_args(model_path, params, custom_flags, host_override, port_override)
    quoted = [binary_display_name] + [_quote(a) for a in args]
    return " ".join(quoted)


def preview_router_dir_command(binary_display_name: str, models_dir: str,
                                params: Dict[str, Any], custom_flags: str = "",
                                models_max: int = 4, autoload: bool = True,
                                host_override: str = None, port_override: int = None) -> str:
    """Human-readable command line string for a router-dir launch, for the
    live preview in the UI."""
    args = build_router_dir_args(models_dir, params, custom_flags, models_max, autoload,
                                 host_override, port_override)
    quoted = [binary_display_name] + [_quote(a) for a in args]
    return " ".join(quoted)


def _quote(token: str) -> str:
    if token == "" or any(c.isspace() for c in token):
        return f'"{token}"'
    return token
