"""
pytest configuration for the tests/ directory.

test_llama_server_api.py and test_llama_server_download.py are MANUAL
end-to-end scripts - run them with `python tests/<file>` (they need
network access and do a real ~18 MB download). They are NOT pytest
modules: they execute their checks at import time, so letting pytest
import them during collection would crash the whole run (SystemExit at
collection) and hit the live GitHub API. Keep the collected suite
hermetic and offline by excluding them here.
"""

collect_ignore = [
    "test_llama_server_api.py",
    "test_llama_server_download.py",
]
