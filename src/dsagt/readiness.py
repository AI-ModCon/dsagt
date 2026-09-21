"""AI-readiness check: the AIDRIN quality baseline around every tabular stage.

The pipeline-builder instructions require a paired check before and after
every data operation, with reports in ``audit/``.  When a project keeps the
readiness check on (the default at ``dsagt init``), the check for a tabular
stage is the ``aidrin`` skill's quality baseline, run through the ``aidrin``
code that every project registers at init, so each run is an execution
record.  The setting adds one paragraph to the agent's instructions at the
per-operation check rule and changes nothing else: the ``aidrin`` package is
a dependency of dsagt, the skill is a base skill fetched at the release tag
of the installed package, and AIDRIN's own workflow serves a user who asks
for an assessment.

The ``readiness`` block of ``.dsagt/config.yaml`` holds the one setting::

    readiness:
      auto_assess: true
"""

from __future__ import annotations

#: The paragraph :func:`dsagt.agents.base._load_master_instructions` fills in
#: at the per-operation check rule when the check is on.
INSTRUCTIONS_PARAGRAPH = """\
#### AI-readiness check

For a stage whose input or output is a tabular file (CSV, TSV, Excel, JSON,
HDF5, Parquet, npz), the check is the `aidrin` skill's quality baseline: run
it on the data as it arrives and on the output of each transformation,
through the registered `aidrin` code's `executable` (never bare `aidrin`). A
JSON, HDF5, or NumPy file may hold nested or multi-dataset structure that
the AIDRIN baseline reads as one flat table; say so beside the numbers when
you report them. The report AIDRIN prints is saved to the run's record and
is retrieved with the `readiness_reports` MCP tool. Report the per-metric
change to the user before proposing the next step. Compare a score only with
an earlier report on the same table, or with the report of the table it was
derived from. Do not write a custom check for a metric AIDRIN provides."""


def aidrin_release_tag(version: str) -> str:
    """The AIDRIN git tag that holds *version* of the ``aidrin`` package.

    AIDRIN tags a release ``v<year>.<month>.<patch>`` with a two-digit month
    (``v2026.08.2``), while the installed package reports the normalized
    version (``2026.8.2``), so the month is zero-padded here.  Raises
    ``ValueError`` for a version that is not three integer components.
    """
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"aidrin version must be <year>.<month>.<patch>, got {version!r}"
        )
    year, month, patch = parts
    return f"v{year}.{int(month):02d}.{patch}"


def auto_assess_enabled(config: dict) -> bool:
    """Whether the project runs the readiness check; on when the config has no block."""
    block = config.get("readiness") or {}
    return bool(block.get("auto_assess", True))
