"""
param_schema.py

Structured definition of the llama-server CLI parameters exposed in the
Parameter Configurator UI. This is the single source of truth for:
  - which flags are shown
  - what widget type to render for each
  - default values / valid ranges / enum choices
  - the plain-English help text shown in the info tooltip

The flag names, defaults and short descriptions below are sourced from the
llama.cpp server README (tools/server/README.md, ggml-org/llama.cpp,
"Common params" / "Sampling params" / "Server-specific params" tables) and
from `llama-server --help`. Help text has been rewritten in plain English
for this app rather than copied verbatim from the docs. Because llama.cpp
changes frequently, use the "Check for schema updates" action in Settings
to re-fetch the flag list, and use "Advanced mode" in the configurator to
pass any brand-new flag that isn't in this schema yet as a raw argument.

Each parameter entry has the shape:
{
    "key": str,                 # unique id, used as the JSON key in a profile's params dict
    "flag": str,                # the primary long-form CLI flag, e.g. "--ctx-size"
    "short_flag": str | None,   # short alias if one exists, e.g. "-c"
    "label": str,                # human friendly label for the form
    "type": "bool" | "int" | "float" | "string" | "enum",
    "default": Any,
    "min": number | None,        # for int/float
    "max": number | None,        # for int/float
    "step": number | None,       # for int/float: number-input spinner step (e.g. ctx-size +256);
                                 # also the step of a "slider" widget; floats without a
                                 # step stay free-typed (step="any")
    "options": [str] | None,     # for enum
    "suggestions": [number] | None,  # for int/float: datalist dropdown of common values;
                                     # the field stays a free-typed input (custom values allowed)
    "widget": str | None,         # optional UI override; "slider" renders the int as a range
                                  # input instead of a number box
    "slider_max_from": str | None,  # with widget "slider": name of the model GGUF fact
                                  # (from GET /api/gguf/facts, e.g. "block_count") that sets
                                  # the slider's maximum; the control is disabled when it
                                  # can't be read
    "slider_max_offset": int | None,  # with widget "slider": added to the fact before it is
                                  # used as the maximum (e.g. -1 for --n-cpu-moe:
                                  # max = block_count - 1)
    "unset_label": str | None,        # with widget "slider": text shown in the value slot while
                                  # unset (default "not set"); e.g. "Auto" for --ctx-size, since
                                  # an unset ctx-size means llama.cpp uses the model's own size
    "requires_model_fact": str | None,  # name of a model GGUF fact that must be a positive
                                  # integer for the parameter to be shown at all; the row is
                                  # hidden otherwise (e.g. --n-cpu-moe only on MoE models,
                                  # expert_count > 0)
    "help": str,                 # plain-English description + when to change it
    "category": str,             # which collapsible section it belongs to
}
"""

from typing import Any, Dict, List

CATEGORIES: List[Dict[str, str]] = [
    {"id": "model_context", "label": "Model & Context"},
    {"id": "multimodal", "label": "Multimodal Configuration"},
    {"id": "performance", "label": "Performance"},
    {"id": "speculative", "label": "Speculative Decoding"},
    {"id": "sampling", "label": "Sampling Defaults"},
    {"id": "server_network", "label": "Server & Network"},
    {"id": "memory_quant", "label": "Memory & Quantization"},
    {"id": "lora_advanced", "label": "LoRA & Advanced"},
]

PARAMETERS: List[Dict[str, Any]] = [
    # ---------------- Model & Context ----------------
    {
        "key": "ctx_size",
        "flag": "--ctx-size",
        "short_flag": "-c",
        "label": "Context Size",
        "type": "int",
        "default": 4096,
        "min": 0,
        "max": None,   # dynamic: the model's context_length from its GGUF metadata (slider max)
        "step": 256,
        "options": None,
        "widget": "slider",
        "slider_max_from": "context_length",
        "unset_label": "Auto",
        "help": "Maximum number of tokens (prompt + generated) the model can keep in its context "
                "window. The slider spans from 0 up to the model's own context length, read "
                "from the model file's GGUF metadata. Raise this if you need longer conversations or documents, "
                "but it increases memory (especially VRAM) usage roughly linearly, "
                "and very large values can slow down prompt processing. Unset for default (auto), "
                "The parameter is disabled when the model's context length can't be read.",
        "category": "model_context",
    },
    {
        "key": "n_gpu_layers",
        "flag": "--n-gpu-layers",
        "short_flag": "-ngl",
        "label": "GPU Layers",
        # Kept "string" on purpose: pre-slider profiles may store "auto"/"all",
        # which the CLI accepts and the slider displays until the user drags.
        "type": "string",
        "default": "auto",
        "min": 0,
        "max": None,   # dynamic: the model's block_count from its GGUF metadata (slider max)
        "options": None,
        "widget": "slider",
        "slider_max_from": "block_count",
        # Unset means "let llama.cpp decide based on available VRAM" - the
        # same as the stored value "auto" - so the slot says "auto", not the
        # generic "not set".
        "unset_label": "Auto",
        "help": "How many of the model's layers to offload to the GPU: the slider runs from 0 (all on "
                "the CPU) to the model's total block count (all on the GPU), read from the model file's "
                "GGUF metadata. Leave it not set (or 'auto') to let llama.cpp decide based on available "
                "VRAM; older profiles may show 'auto' or 'all' - dragging the slider replaces them with "
                "an exact number. Higher values mean more of the model runs on the GPU (faster) but use "
                "more VRAM; lower it if you get out-of-memory errors. The parameter is disabled when the "
                "block count can't be read.",
        "category": "model_context",
    },
    {
        "key": "n_cpu_moe",
        "flag": "--n-cpu-moe",
        "short_flag": "-ncmoe",
        "label": "CPU MoE Layers",
        "type": "int",
        "default": 0,
        "min": 0,
        "max": None,   # dynamic: the model's block_count from its GGUF metadata (slider max)
        "options": None,
        "widget": "slider",
        "slider_max_from": "block_count",
        "slider_max_offset": -1,
        "requires_model_fact": "expert_count",
        "help": "Keep the Mixture of Experts (MoE) weights of the first N layers on the CPU while the "
                "rest run on the GPU. Only shown for MoE models - those whose GGUF metadata has "
                "expert_count > 0. The slider spans 0 (all MoE weights on the GPU - the default) up "
                "to total block count, also read from the model file's "
                "GGUF metadata. Use it to relieve VRAM pressure on MoE models. The parameter is "
                "disabled when the block count can't be read from the model file.",
        "category": "model_context",
    },
    {
        "key": "n_predict",
        "flag": "--n-predict",
        "short_flag": "-n",
        "label": "Max Tokens to Predict",
        "type": "int",
        "default": -1,
        "min": -1,
        "max": 1048576,
        "options": None,
        "help": "Caps how many new tokens the server will generate per request before stopping on its "
                "own. -1 means no limit (generation stops only on an end-of-sequence token or a client "
                "stop condition).",
        "category": "model_context",
    },
    {
        "key": "rope_scaling",
        "flag": "--rope-scaling",
        "short_flag": None,
        "label": "RoPE Scaling Method",
        "type": "enum",
        "default": "none",
        "min": None,
        "max": None,
        "options": ["none", "linear", "yarn"],
        "help": "Selects the method used to stretch the model's positional encoding beyond its native "
                "training length. Leave as the model default unless you're deliberately running the "
                "model at a longer context than it was trained for.",
        "category": "model_context",
    },
    {
        "key": "rope_freq_base",
        "flag": "--rope-freq-base",
        "short_flag": None,
        "label": "RoPE Frequency Base",
        "type": "float",
        "default": None,
        "min": 0,
        "max": None,
        "options": None,
        "help": "Base frequency for RoPE (NTK-aware scaling). Defaults to the value stored in the model "
                "file. Only change this if you know you need custom context extension behavior.",
        "category": "model_context",
    },
    {
        "key": "rope_freq_scale",
        "flag": "--rope-freq-scale",
        "short_flag": None,
        "label": "RoPE Frequency Scale",
        "type": "float",
        "default": None,
        "min": 0,
        "max": None,
        "options": None,
        "help": "Scaling factor applied to RoPE frequency; effectively expands the usable context by a "
                "factor of 1/N. Usually left untouched and controlled instead via context size and the "
                "model's own metadata.",
        "category": "model_context",
    },
    {
        "key": "yarn_orig_ctx",
        "flag": "--yarn-orig-ctx",
        "short_flag": None,
        "label": "YaRN Original Context",
        "type": "int",
        "default": 0,
        "min": 0,
        "max": None,
        "options": None,
        "help": "The model's original training context size, used as a reference point by YaRN context "
                "scaling. 0 uses the model's own training context size.",
        "category": "model_context",
    },

    # ---------------- Multimodal Configuration ----------------
    {
        "key": "mmproj",
        "flag": "--mmproj",
        "short_flag": "-mm",
        "label": "Multimodal Projector File",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "suggestions": None,
        "help": "Path to a multimodal projector (.mmproj) file, for vision models. Leave blank for "
                "regular text models. When starting a model from a Hugging Face repo (-hf), the "
                "projector is downloaded automatically if available, so this can be omitted.",
        "category": "multimodal",
    },
    {
        "key": "mmproj_url",
        "flag": "--mmproj-url",
        "short_flag": "-mmu",
        "label": "Multimodal Projector URL",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "suggestions": None,
        "help": "URL to a multimodal projector file to fetch at startup. An alternative to pointing "
                "--mmproj at a local file.",
        "category": "multimodal",
    },
    {
        "key": "no_mmproj",
        "flag": "--no-mmproj",
        "short_flag": None,
        "label": "Disable Automatic Multimodal Projector",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "suggestions": None,
        "help": "Controls whether a multimodal projector file is used when one is available - mainly "
                "relevant when starting from a Hugging Face repo (-hf), where a projector is "
                "downloaded and attached automatically. Enabled by default; turn this on to "
                "disable that behavior (same as passing --no-mmproj / --no-mmproj-auto).",
        "category": "multimodal",
    },
    {
        "key": "no_mmproj_offload",
        "flag": "--no-mmproj-offload",
        "short_flag": None,
        "label": "Disable Projector GPU Offload",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "suggestions": None,
        "help": "The multimodal projector is offloaded to the GPU by default, along with the model. "
                "Turn this on to keep the projector on the CPU - saves VRAM at the cost of slower "
                "image processing (same as passing --no-mmproj-offload).",
        "category": "multimodal",
    },

    # ---------------- Performance ----------------
    {
        "key": "threads",
        "flag": "--threads",
        "short_flag": "-t",
        "label": "CPU Threads",
        "type": "int",
        "default": -1,
        "min": -1,
        "max": 256,
        "options": None,
        "help": "Number of CPU threads used during text generation. -1 lets llama.cpp pick a sensible "
                "default for your machine. Increase for CPU-bound inference on many-core machines; too "
                "high a value on a small CPU can actually slow things down due to contention.",
        "category": "performance",
    },
    {
        "key": "threads_batch",
        "flag": "--threads-batch",
        "short_flag": "-tb",
        "label": "CPU Threads (Batch/Prompt)",
        "type": "int",
        "default": None,
        "min": -1,
        "max": 256,
        "options": None,
        "help": "Number of threads used while processing the prompt (batch phase), as opposed to "
                "generating tokens one at a time. Defaults to the same value as CPU Threads.",
        "category": "performance",
    },
    {
        "key": "batch_size",
        "flag": "--batch-size",
        "short_flag": "-b",
        "label": "Batch Size",
        "type": "int",
        "default": 2048,
        "min": 0,
        "max": 65536,
        "step": 256,
        "options": None,
        "help": "The logical maximum number of tokens processed together in one batch during prompt "
                "processing. Larger values can speed up prompt ingestion at the cost of more memory.",
        "category": "performance",
    },
    {
        "key": "ubatch_size",
        "flag": "--ubatch-size",
        "short_flag": "-ub",
        "label": "Micro-batch Size",
        "type": "int",
        "default": 512,
        "min": 0,
        "max": 65536,
        "step": 128,
        "options": None,
        "help": "The physical maximum batch size actually submitted to the backend at once (a batch is "
                "split into micro-batches of this size). Lower it if you hit out-of-memory errors during "
                "prompt processing.",
        "category": "performance",
    },
    {
        "key": "flash_attn",
        "flag": "--flash-attn",
        "short_flag": "-fa",
        "label": "Flash Attention",
        "type": "enum",
        "default": "auto",
        "min": None,
        "max": None,
        "options": ["on", "off", "auto"],
        "help": "Enables the Flash Attention kernel, which usually reduces memory use and speeds up "
                "inference on supported hardware. 'auto' lets llama.cpp decide; force 'on' or 'off' if "
                "you're troubleshooting correctness or performance on a specific backend.",
        "category": "performance",
    },
    {
        "key": "parallel",
        "flag": "--parallel",
        "short_flag": "-np",
        "label": "Parallel Slots",
        "type": "int",
        "default": -1,
        "min": -1,
        "max": 256,
        "options": None,
        "help": "Number of concurrent request 'slots' the server maintains, enabling several requests to "
                "be processed together via continuous batching. -1 lets the server choose automatically. "
                "Raise this if multiple clients will hit the server at once; each slot needs its own "
                "share of the context/KV cache.",
        "category": "performance",
    },
    {
        "key": "no_cont_batching",
        "flag": "--no-cont-batching",
        "short_flag": "-nocb",
        "label": "Disable Continuous Batching",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "help": "Continuous batching - interleaving multiple in-flight requests instead of processing them "
                "strictly one at a time - is enabled by default and improves throughput under concurrent "
                "load. Turn this on to disable it, e.g. if you're debugging request-ordering issues.",
        "category": "performance",
    },

    # ---------------- Speculative Decoding ----------------
    {
        "key": "spec_type",
        "flag": "--spec-type",
        "short_flag": None,
        "label": "Speculative Decoding Type(s)",
        "type": "string",
        "default": "none",
        "min": None,
        "max": None,
        "options": None,
        "help": "Comma-separated list of speculative decoding methods to use, e.g. 'draft-simple' or "
                "'ngram-simple,ngram-cache'. Valid values: none, draft-simple, draft-eagle3, draft-mtp, "
                "draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, "
                "ngram-cache. 'none' disables speculative decoding. The draft-* methods need a separate "
                "draft model; the ngram-* methods generate draft tokens from the input/output text itself "
                "without a second model.",
        "category": "speculative",
    },
    {
        "key": "spec_draft_model",
        "flag": "--spec-draft-model",
        "short_flag": "-md",
        "label": "Draft Model Path",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "help": "Path to the smaller draft model file (.gguf) used by the draft-* speculative decoding methods. "
                "The draft model predicts a few tokens ahead that the main model verifies, speeding up "
                "generation when it usually agrees. Only needed for draft-based methods (see Speculative "
                "Decoding Type(s)); the ngram-* methods generate drafts from the text itself and need no "
                "second model. Leave blank to disable draft-model usage.",
        "category": "speculative",
    },
    {
        "key": "spec_draft_n_max",
        "flag": "--spec-draft-n-max",
        "short_flag": None,
        "label": "Max Draft Tokens",
        "type": "int",
        "default": 3,
        "min": 0,
        "max": None,
        "options": None,
        "help": "How many tokens to speculatively draft ahead per step when speculative decoding is "
                "enabled (see Speculative Decoding Type(s)). Higher values can speed things up further "
                "when the draft is usually accepted, but waste more work when it's rejected.",
        "category": "speculative",
    },

    # ---------------- Sampling Defaults ----------------
    {
        "key": "temp",
        "flag": "--temp",
        "short_flag": None,
        "label": "Temperature",
        "type": "float",
        "default": 0.8,
        "min": 0.0,
        "max": 5.0,
        "step": 0.1,
        "options": None,
        "help": "Controls randomness of token selection. Lower values (e.g. 0.2) make output more "
                "focused and deterministic; higher values (e.g. 1.2+) make it more varied and creative, "
                "at the cost of coherence.",
        "category": "sampling",
    },
    {
        "key": "top_k",
        "flag": "--top-k",
        "short_flag": None,
        "label": "Top-K",
        "type": "int",
        "default": 40,
        "min": 0,
        "max": 1000,
        "options": None,
        "help": "Restricts token choice to the K most likely next tokens before sampling. 0 disables this "
                "filter. Lower values make output safer/more repetitive; higher values allow more variety.",
        "category": "sampling",
    },
    {
        "key": "top_p",
        "flag": "--top-p",
        "short_flag": None,
        "label": "Top-P (Nucleus)",
        "type": "float",
        "default": 0.95,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "options": None,
        "help": "Keeps only the smallest set of tokens whose cumulative probability exceeds P, then "
                "samples from that set. 1.0 disables this filter. Commonly tuned together with "
                "temperature to shape output diversity.",
        "category": "sampling",
    },
    {
        "key": "min_p",
        "flag": "--min-p",
        "short_flag": None,
        "label": "Min-P",
        "type": "float",
        "default": 0.05,
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "options": None,
        "help": "Discards tokens whose probability is below this fraction of the most likely token's "
                "probability. 0.0 disables it. A modern, often more stable alternative/complement to "
                "top-p for keeping output sensible.",
        "category": "sampling",
    },
    {
        "key": "repeat_penalty",
        "flag": "--repeat-penalty",
        "short_flag": None,
        "label": "Repeat Penalty",
        "type": "float",
        "default": 1.0,
        "min": 0.0,
        "max": 3.0,
        "options": None,
        "help": "Penalizes tokens that already appeared recently, discouraging loops and verbatim "
                "repetition. 1.0 disables it. Raise slightly (e.g. 1.1) if the model tends to repeat "
                "itself; too high can make output stilted.",
        "category": "sampling",
    },
    {
        "key": "repeat_last_n",
        "flag": "--repeat-last-n",
        "short_flag": None,
        "label": "Repeat Penalty Window",
        "type": "int",
        "default": 64,
        "min": -1,
        "max": 1048576,
        "options": None,
        "help": "How many of the most recent tokens are considered when applying the repeat penalty. 0 "
                "disables it, -1 uses the full context size.",
        "category": "sampling",
    },
    {
        "key": "presence_penalty",
        "flag": "--presence-penalty",
        "short_flag": None,
        "label": "Presence Penalty",
        "type": "float",
        "default": 0.0,
        "min": -2.0,
        "max": 2.0,
        "options": None,
        "help": "Flat penalty applied to any token that has appeared at all so far, encouraging the "
                "model to bring up new topics. 0.0 disables it.",
        "category": "sampling",
    },
    {
        "key": "frequency_penalty",
        "flag": "--frequency-penalty",
        "short_flag": None,
        "label": "Frequency Penalty",
        "type": "float",
        "default": 0.0,
        "min": -2.0,
        "max": 2.0,
        "options": None,
        "help": "Penalty that scales with how often a token has already appeared, discouraging heavy "
                "reuse of the same words. 0.0 disables it.",
        "category": "sampling",
    },
    {
        "key": "seed",
        "flag": "--seed",
        "short_flag": "-s",
        "label": "RNG Seed",
        "type": "int",
        "default": -1,
        "min": -1,
        "max": 4294967295,
        "options": None,
        "help": "Seed for the random number generator used during sampling. -1 picks a random seed each "
                "run. Set a fixed value for reproducible output across runs (useful for debugging or "
                "benchmarking).",
        "category": "sampling",
    },

    # ---------------- Server & Network ----------------
    {
        "key": "host",
        "flag": "--host",
        "short_flag": None,
        "label": "Host",
        "type": "string",
        "default": "127.0.0.1",
        "min": None,
        "max": None,
        "options": None,
        "help": "IP address the server listens on. 127.0.0.1 only accepts connections from this machine; "
                "use 0.0.0.0 to accept connections from other devices on your network (make sure you "
                "understand the security implications first).",
        "category": "server_network",
    },
    {
        "key": "port",
        "flag": "--port",
        "short_flag": None,
        "label": "Port",
        "type": "int",
        "default": 8080,
        "min": 1,
        "max": 65535,
        "options": None,
        "help": "TCP port the server listens on. Change this if the default port is already in use, or "
                "if you're running multiple servers at once.",
        "category": "server_network",
    },
    {
        "key": "api_key",
        "flag": "--api-key",
        "short_flag": None,
        "label": "API Key",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "help": "Requires clients to send this key (as a Bearer token) to use the API. Leave blank for no "
                "authentication. Recommended if you set Host to 0.0.0.0 or otherwise expose the server "
                "beyond localhost.",
        "category": "server_network",
    },
    {
        "key": "embeddings",
        "flag": "--embeddings",
        "short_flag": None,
        "label": "Embeddings Mode",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "help": "Restricts the server to only serve embeddings, for use with dedicated embedding models "
                "rather than chat/completion models.",
        "category": "server_network",
    },
    {
        "key": "reranking",
        "flag": "--reranking",
        "short_flag": None,
        "label": "Reranking Endpoint",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "help": "Enables the document reranking endpoint. Requires a reranker model and is typically used "
                "together with Embeddings Mode.",
        "category": "server_network",
    },
    {
        "key": "metrics",
        "flag": "--metrics",
        "short_flag": None,
        "label": "Prometheus Metrics",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "help": "Exposes a Prometheus-compatible /metrics endpoint for monitoring dashboards.",
        "category": "server_network",
    },
    {
        "key": "no_webui",
        "flag": "--no-webui",
        "short_flag": None,
        "label": "Disable Web UI",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "help": "llama.cpp's built-in browser chat UI is served by default alongside the API. Turn this on "
                "to disable it if you only want the raw API endpoints.",
        "category": "server_network",
    },

    # ---------------- Memory & Quantization ----------------
    {
        "key": "fit",
        "flag": "--fit",
        "short_flag": "-fit",
        "label": "Fit to Device Memory",
        "type": "enum",
        "default": "on",
        "min": None,
        "max": None,
        "options": ["on", "off"],
        "help": "Whether to adjust unset arguments to fit in device memory ('on' or 'off', default: 'on'). "
                "When on, llama.cpp tunes any parameters you haven't explicitly set so the model fits "
                "within the available device memory; turn it off to disable that automatic adjustment.",
        "category": "memory_quant",
    },
    {
        "key": "load_mode",
        "flag": "--load-mode",
        "short_flag": "-lm",
        "label": "Model Load Mode",
        "type": "enum",
        "default": "auto",
        "min": None,
        "max": None,
        "options": ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"],
        "help": "Controls how the model file is loaded into memory. 'auto' uses mmap unless the device "
                "doesn't support it. 'mmap' memory-maps the file; 'mlock' forces it to stay resident in "
                "RAM rather than being swapped or compressed; 'mmap+mlock' combines both; 'none' skips any "
                "special loading mode; 'dio' uses Direct I/O where available. Replaces the older "
                "--mlock/--no-mmap flags.",
        "category": "memory_quant",
    },
    {
        "key": "cache_type_k",
        "flag": "--cache-type-k",
        "short_flag": "-ctk",
        "label": "KV Cache Type (K)",
        "type": "enum",
        "default": "f16",
        "min": None,
        "max": None,
        "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
        "help": "Data type used to store the K half of the attention KV cache. Quantized types (like "
                "q8_0) cut memory use substantially, letting you fit a longer context or more parallel "
                "slots in the same VRAM, at a small quality cost.",
        "category": "memory_quant",
    },
    {
        "key": "cache_type_v",
        "flag": "--cache-type-v",
        "short_flag": "-ctv",
        "label": "KV Cache Type (V)",
        "type": "enum",
        "default": "f16",
        "min": None,
        "max": None,
        "options": ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
        "help": "Data type used to store the V half of the attention KV cache. Same trade-off as KV Cache "
                "Type (K): quantized types save memory at a small quality cost. Quantized V cache "
                "typically also requires Flash Attention to be enabled.",
        "category": "memory_quant",
    },
    {
        "key": "no_kv_offload",
        "flag": "--no-kv-offload",
        "short_flag": "-nkvo",
        "label": "Disable KV Offload to GPU",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "help": "Keeps the KV cache on the CPU instead of the GPU. Enable this if you're offloading model "
                "layers to GPU but running low on VRAM specifically because of the KV cache.",
        "category": "memory_quant",
    },
    {
        "key": "split_mode",
        "flag": "--split-mode",
        "short_flag": "-sm",
        "label": "Multi-GPU Split Mode",
        "type": "enum",
        "default": "layer",
        "min": None,
        "max": None,
        "options": ["none", "layer", "row", "tensor"],
        "help": "How the model is divided across multiple GPUs. 'layer' pipelines whole layers across "
                "GPUs (default, works well generally); 'row' splits weights by row for more parallel "
                "throughput; 'none' uses a single GPU only. Irrelevant on single-GPU or CPU-only setups.",
        "category": "memory_quant",
    },
    {
        "key": "tensor_split",
        "flag": "--tensor-split",
        "short_flag": "-ts",
        "label": "Tensor Split Ratios",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "help": "Comma-separated proportions controlling how much of the model each GPU gets, e.g. '3,1' "
                "to give the first GPU 3x the share of the second. Leave blank to split evenly.",
        "category": "memory_quant",
    },

    # ---------------- LoRA & Advanced ----------------
    {
        "key": "lora",
        "flag": "--lora",
        "short_flag": None,
        "label": "LoRA Adapter Path(s)",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "help": "Path to one or more LoRA adapter files to apply on top of the base model (comma-separate "
                "multiple paths). Leave blank if you're not using a LoRA fine-tune.",
        "category": "lora_advanced",
    },
    {
        "key": "jinja",
        "flag": "--jinja",
        "short_flag": None,
        "label": "Use Jinja Templating",
        "type": "bool",
        "default": True,
        "min": None,
        "max": None,
        "options": None,
        "help": "Uses the full Jinja2 template engine to render chat prompts, matching how most modern "
                "instruct/chat models expect their prompt formatted. Leave enabled unless you have a "
                "specific reason to use the legacy formatter.",
        "category": "lora_advanced",
    },
    {
        "key": "chat_template",
        "flag": "--chat-template",
        "short_flag": None,
        "label": "Chat Template Override",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "help": "Forces a specific built-in Jinja chat template name instead of the one embedded in the "
                "model file. Leave blank to use the model's own template; only change this if chat "
                "formatting looks wrong for a particular model.",
        "category": "lora_advanced",
    },
    {
        "key": "chat_template_file",
        "flag": "--chat-template-file",
        "short_flag": None,
        "label": "Chat Template File",
        "type": "string",
        "default": "",
        "min": None,
        "max": None,
        "options": None,
        "help": "Path to a file containing a custom Jinja chat template - used instead of the template "
                "embedded in the model. Like --mmproj, the field shows a bare file name and offers "
                "candidate .jinja files from the model's own folder as a dropdown; any full path can "
                "also be typed. Leave blank to use the model's own template.",
        "category": "lora_advanced",
    },
    {
        "key": "log_disable",
        "flag": "--log-disable",
        "short_flag": None,
        "label": "Disable Logging",
        "type": "bool",
        "default": False,
        "min": None,
        "max": None,
        "options": None,
        "help": "Turns off llama.cpp's own log output entirely. Usually left off so you can see stdout in "
                "the log panel here; enable if the logs are too noisy.",
        "category": "lora_advanced",
    },
]


def get_schema() -> Dict[str, Any]:
    """Return the full schema (categories + parameters) as a plain dict for the API/frontend."""
    return {"categories": CATEGORIES, "parameters": PARAMETERS}


def get_parameter(key: str) -> Dict[str, Any] | None:
    for p in PARAMETERS:
        if p["key"] == key:
            return p
    return None
