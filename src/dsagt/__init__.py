"""
DSAgt — DataSmith Agent.

AI-assisted data pipeline builder for MCP-compatible agents.
"""

# Single source of truth for the package version: pyproject.toml reads this
# via `[tool.setuptools.dynamic] version = {attr = "dsagt.__version__"}`.
__version__ = "0.2.0"

# Cap CPU thread count for embedding / tokenization libraries before any
# heavy imports happen.  Without this, PyTorch / sentence-transformers /
# numpy+MKL default to using every available core, which pegs the
# machine and causes visible system unresponsiveness during embed bursts
# (kb_ingest, kb_search bursts, init's KB build).  Half the physical
# cores is a sensible default that leaves headroom for the OS, the
# agent process, OneDrive sync, IDE, browser, etc.  ``setdefault``
# preserves any value the user has already exported in their shell.
import os as _os

_default_threads = str(max(1, (_os.cpu_count() or 4) // 2))
_os.environ.setdefault("OMP_NUM_THREADS", _default_threads)
_os.environ.setdefault("MKL_NUM_THREADS", _default_threads)
# Silence the "tokenizers/parallelism" fork warning that fires when
# sentence-transformers' tokenizer is used after a fork (e.g. under
# pytest-xdist or DataLoader workers).
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# uv bundles its own OpenSSL that does not read the macOS Security keychain.
# On networks with an SSL-intercepting proxy (e.g. Zscaler), the corporate
# root CA lives only in the keychain, so HTTPS downloads (HuggingFace model
# weights, arXiv PDFs) fail with "unable to get local issuer certificate".
# truststore patches ssl.SSLContext to use the OS-native trust store
# (macOS Security framework / Windows Certificate Store), making the
# installed corporate CA visible to httpx, requests, and urllib3.
# inject_into_ssl() must run before any ssl.SSLContext is constructed.
try:
    import truststore as _truststore

    _truststore.inject_into_ssl()
    del _truststore
except ImportError:
    pass
del _os, _default_threads

from dsagt.registry import CodeRegistry

__all__ = ["CodeRegistry", "__version__"]
