"""Readiness assessment — AIDRIN as the check for every pipeline stage.

The pipeline-builder instructions require a paired check before and after
every data operation, with reports in ``audit/``.  This module makes that
check concrete: when a project opts in at ``dsagt init --readiness aidrin``,
DSAGT installs AIDRIN (AI Data Readiness Inspector) into a shared venv and
appends an instructions block telling the agent that ``check_[X]`` for
tabular data is an AIDRIN run of a fixed metric profile.  The ``aidrin``
skill itself is one of the base skills every project carries
(``skills.BASE_SKILLS``), fetched from the AIDRIN repository at init; the
assessment only makes the agent apply it on every stage.  Metric selection is by
named profile, not by prompt, so the agent runs the same metric set on every
stage of a pipeline.

Three pieces, all keyed on the ``readiness`` block of ``.dsagt/config.yaml``::

    readiness:
      tool: aidrin
      executable: ~/dsagt-projects/.tools/aidrin/bin/aidrin
      profile: quality

* :func:`ensure_aidrin` — the one-time install (``uv venv`` + ``uv pip``).
* :data:`PROFILES` — the metric sets the instructions name.
* :func:`instructions_block` — the text appended to the agent's instructions
  file by :func:`dsagt.agents.static_agent_record`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from dsagt.session import REGISTRY_DIR

#: Shared, machine-global install dir for readiness tooling (sibling of
#: ``kb_index/`` and ``.skill_sources/``).  One venv serves every project.
TOOLS_DIR = REGISTRY_DIR / ".tools"
AIDRIN_DIR = TOOLS_DIR / "aidrin"
AIDRIN_EXECUTABLE = AIDRIN_DIR / "bin" / "aidrin"

#: AIDRIN's ``develop`` branch holds the CLI and the ``aidrin`` skill.
AIDRIN_SPEC = "aidrin[mcp] @ git+https://github.com/idtlab/AIDRIN@develop"
#: AIDRIN requires Python >= 3.10; some of its dependencies have no 3.13+
#: wheels, so the venv is pinned below that.
AIDRIN_PYTHON = ">=3.10,<3.13"

#: Marker line the instructions block carries so the append is idempotent.
READINESS_MARKER = "DSAgt AI-Readiness Assessment"

#: Metric profiles the assessment instructions name.  ``quality`` runs on any
#: tabular file with no column arguments.  ``supervised`` adds the
#: target-dependent metrics: ``class-imbalance`` takes the target column and
#: ``feature-relevance`` takes the categorical and numerical column lists
#: plus the target.  Fairness-rate and privacy metrics are never in a
#: profile: they assume sensitive attributes or quasi-identifiers, which is
#: a per-dataset judgment the agent must make with the user.
PROFILES: dict[str, tuple[str, ...]] = {
    "quality": ("completeness", "duplicity", "outliers"),
    "supervised": (
        "completeness",
        "duplicity",
        "outliers",
        "class-imbalance",
        "feature-relevance",
    ),
}

TOOLS = ("aidrin",)


def readiness_block(
    tool: str, *, executable: str | Path | None = None, profile: str = "quality"
) -> dict:
    """The ``readiness`` config block for *tool*.

    ``executable`` defaults to the shared-venv path :func:`ensure_aidrin`
    installs to; pass an explicit path to use an existing AIDRIN install.
    """
    if tool not in TOOLS:
        raise ValueError(f"readiness tool must be one of {TOOLS}, got {tool!r}")
    if profile not in PROFILES:
        raise ValueError(
            f"readiness profile must be one of {tuple(PROFILES)}, got {profile!r}"
        )
    return {
        "tool": tool,
        "executable": str(executable or AIDRIN_EXECUTABLE),
        "profile": profile,
    }


def ensure_aidrin(install_dir: Path = AIDRIN_DIR) -> Path:
    """Install AIDRIN into *install_dir* (a venv) and return the ``aidrin`` path.

    Idempotent: returns immediately when the executable exists.  Raises
    ``RuntimeError`` carrying the failing command's stderr; a half-built
    venv is removed so a retry starts clean rather than seeing a venv with
    no ``aidrin`` in it.
    """
    executable = install_dir / "bin" / "aidrin"
    if executable.exists():
        return executable
    if shutil.which("uv") is None:
        raise RuntimeError(
            "uv is required to install AIDRIN (https://docs.astral.sh/uv/)"
        )
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        ["uv", "venv", "--python", AIDRIN_PYTHON, str(install_dir)],
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(install_dir / "bin" / "python"),
            AIDRIN_SPEC,
        ],
    ]
    for cmd in steps:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            shutil.rmtree(install_dir, ignore_errors=True)
            raise RuntimeError(
                f"AIDRIN install failed at `{' '.join(cmd)}`:\n{result.stderr.strip()}"
            )
    return executable


def instructions_block(readiness: dict) -> str:
    """The instructions appended to the agent's file when the assessment is enabled.

    Stated as rules the agent applies at every stage; the ``aidrin`` skill in
    ``skills/aidrin/`` carries the command forms and the argument order of
    each metric (``reference/metrics.md``).
    """
    profile = readiness.get("profile", "quality")
    metrics = ", ".join(PROFILES[profile])
    executable = readiness["executable"]
    return f"""# {READINESS_MARKER}

AIDRIN is enabled as the readiness check for this project. It replaces the
generic `check_[X]` in the per-operation check rule for every stage whose
input or output is a tabular file (CSV, Excel, JSON, HDF5, Parquet, npz).
The `aidrin` skill in `skills/aidrin/` documents the CLI; the executable for
this project is `{executable}`. The AIDRIN MCP tools are not available here:
use the CLI path. Every `aidrin` command in this project — `list`,
`summarize`, `run`, `batch`, whether or not it is an assessment run — goes through
dsagt-run so it is recorded:
`dsagt-run --code aidrin -- {executable} <aidrin args>`.

1. Before and after each data operation on a tabular file — including a merge,
   filter, or conversion you implement yourself — run every metric of the
   project profile (`{profile}`: {metrics}) on that file, one metric per
   call:
   `dsagt-run --code aidrin -- {executable} run <metric> <file> [args]`.
   Column arguments follow the skill's `reference/metrics.md`; ask the user
   for the target column once per pipeline. Collect the JSON outputs into
   `audit/step_N_pre.aidrin.json` / `audit/step_N_post.aidrin.json`, keyed
   by metric. `aidrin run` exits 0 on failure: a result holding an `Error`
   key is a failed metric — record it as such and continue with the rest.
2. The assessment is fixed: for assessment runs skip the skill's intent-elicitation and
   plan-confirmation steps. Use the skill's full workflow only when the
   user asks for a readiness assessment beyond the profile.
3. After the post-run, report the per-metric change between the pre and
   post reports to the user in one short table before proposing the next step.
4. Do not write a custom check code for a metric AIDRIN already provides.
   Metrics outside the profile (fairness rates, privacy, file-reference
   validation) run the same way only when the user confirms the dataset has
   the attributes they assume.
5. Stages whose input and output are not tabular (images, tar shards,
   model-ready tensors) keep the generic check rule; do not wrap them in an
   AIDRIN call.
"""
