"""Submodule: `import m3dc1.plot_field_basic` — exposes mutable shortlbl/label globals."""
from m3dc1 import eval_field, plot_field  # noqa: F401

# These module-level globals are set by calling code (e.g. plotsm3dc1.py)
# before invoking the plot functions.
shortlbl: dict = {}
label: dict = {}

__all__ = ["plot_field", "eval_field", "shortlbl", "label"]
