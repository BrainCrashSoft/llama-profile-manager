"""
gguf_utils.py

Shared helpers for working with GGUF filenames: quant-type parsing and
multi-part file grouping. Used by both the local filesystem scanner
(scanner.py) and the Hugging Face client (hf_client.py) so quant detection
stays consistent between "what's on disk" and "what's available to
download".
"""

import re
from typing import Optional, Tuple

# Matches the multi-part suffix, e.g. "-00001-of-00003"
MULTIPART_RE = re.compile(r"-(\d{5})-of-(\d{5})$", re.IGNORECASE)

# Known llama.cpp / GGUF quantization tokens. Order matters: more specific
# tokens (like the "_XL" dynamic-quant variants popularized by unsloth) must
# come before their shorter prefixes so a full match is attempted first -
# though since matching requires a word boundary on both ends, a shorter
# token like "Q4_K" simply won't match inside "Q4_K_XL" at all (the "_" right
# after "K" isn't a word boundary), so omitting the XL variants was the
# actual bug rather than an ordering issue. Kept grouped here for clarity.
QUANT_TOKENS = [
    # "Dynamic"/XL variants (unsloth-style)
    "Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q5_K_XL", "Q6_K_XL", "Q8_K_XL",
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL",
    "Q2_K_S", "Q2_K", "Q3_K_XS", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K",
    "Q4_K_S", "Q4_K_M", "Q4_K", "Q4_0", "Q4_1",
    "Q5_K_S", "Q5_K_M", "Q5_K", "Q5_0", "Q5_1",
    "Q6_K", "Q8_0",
    # Ternary (BitNet-style) quantization
    "TQ1_0", "TQ2_0",
    # Newer low-bit float formats
    "NVFP4", "MXFP4", "FP8", "FP16", "FP32",
    "F16", "BF16", "F32",
]
QUANT_RE = re.compile(r"(?i)\b(" + "|".join(re.escape(t) for t in QUANT_TOKENS) + r")\b")


def parse_quant(filename: str) -> Optional[str]:
    """Extract the quantization token from a GGUF filename, if present."""
    match = QUANT_RE.search(filename)
    return match.group(1).upper() if match else None


def split_multipart(stem: str) -> Tuple[Optional[str], Optional[str]]:
    """
    If `stem` (filename without extension) ends with -NNNNN-of-MMMMM, return
    (base_stem, "NNNNN/MMMMM"). Otherwise return (None, None).
    """
    m = MULTIPART_RE.search(stem)
    if not m:
        return None, None
    return stem[: m.start()], f"{m.group(1)}/{m.group(2)}"
