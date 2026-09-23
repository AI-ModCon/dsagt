"""AI-readiness check — the AIDRIN quality baseline around every tabular stage.

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

For a stage whose input or output is a table, the check is the `aidrin`
skill's quality baseline: run it on the file before and after the operation,
through the registered `aidrin` code's `executable` (never bare `aidrin`).
A table is a CSV, Parquet, Excel, or JSON-records file; an HDF5 or NumPy file
counts only once `aidrin summarize` shows it as one table, since AIDRIN reads
any HDF5 it can flatten and scores a simulation field as columns. Before a
check, call the `readiness_reports` tool on the file: a report from a run
after which the file is unchanged is current, and the post report of one
stage is the pre report of the next, so an unchanged file is not checked
twice. Run the baseline directly; do not ask the user about intent or confirm
a plan for these checks (the skill's full workflow is for assessments the user
asks for). The CLI prints its report to stdout and the record holds what it
printed, so redirect it into the audit file for a reader:
`... aidrin data-quality <file> --detail > audit/step_N_pre.aidrin.json`
before the operation and `> audit/step_N_post.aidrin.json` after it, then
report the per-metric change to the user before proposing the next step. Do not write a custom check for a
metric AIDRIN provides. A stage with a table as input or output gets this
check; every other stage keeps the check rule above."""


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


def readiness_block(auto_assess: bool) -> dict:
    """The ``readiness`` config block."""
    return {"auto_assess": bool(auto_assess)}


def auto_assess_enabled(config: dict) -> bool:
    """Whether the project runs the readiness check; on when the config has no block."""
    block = config.get("readiness") or {}
    return bool(block.get("auto_assess", True))
