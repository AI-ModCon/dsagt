"""
Pipeline reconstruction from tool execution records.

Reads wrapper records from trace_archive, builds a dependency graph
from input/output file overlap, and renders the pipeline as a
reproducible bash script or Snakemake workflow.

Only records with an execution layer (from dsagt-run) are used.
Proxy-only records (intent + report) don't carry file lists.

Usage::

    from dsagt.pipeline import reconstruct_pipeline

    script = reconstruct_pipeline(
        trace_dir=Path("runtime/my-project/trace_archive"),
        session_id="cheese-metagenome",
        fmt="bash",
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def load_pipeline_records(trace_dir: Path, session_id: str | None = None) -> list[dict]:
    """Load execution records that have the wrapper execution layer.

    Returns records sorted by execution start time.
    """
    if not trace_dir.is_dir():
        return []

    records = []
    for path in trace_dir.glob("*.json"):
        raw = json.loads(path.read_text())
        execution = raw.get("execution")
        if not execution:
            continue
        if session_id and raw.get("session_id") != session_id:
            continue
        records.append(raw)

    records.sort(key=lambda r: r["execution"].get("timestamp_start", ""))
    return records


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

def build_dependency_graph(records: list[dict]) -> dict[int, list[int]]:
    """Build a dependency graph from input/output file overlap.

    Returns adjacency list: {step_index: [indices of steps it depends on]}.
    A step depends on an earlier step if any of its input files appear
    in that earlier step's output files.
    """
    # Map output files to the step that produced them
    output_to_step: dict[str, int] = {}
    for i, record in enumerate(records):
        for f in record["execution"].get("output_files", []):
            output_to_step[f] = i

    # Build adjacency
    deps: dict[int, list[int]] = {i: [] for i in range(len(records))}
    for i, record in enumerate(records):
        for f in record["execution"].get("input_files", []):
            producer = output_to_step.get(f)
            if producer is not None and producer != i:
                if producer not in deps[i]:
                    deps[i].append(producer)

    return deps


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_bash(records: list[dict], deps: dict[int, list[int]]) -> str:
    """Render the pipeline as a bash script."""
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Pipeline reconstructed from DSAGT execution records",
        "",
    ]

    for i, record in enumerate(records):
        tool = record["tool_name"]
        execution = record["execution"]
        cmd = execution["exact_command"]
        rc = execution.get("return_code", 0)
        inputs = execution.get("input_files", [])
        outputs = execution.get("output_files", [])

        lines.append(f"# Step {i + 1}: {tool}")
        if inputs:
            lines.append(f"#   inputs:  {', '.join(inputs)}")
        if outputs:
            lines.append(f"#   outputs: {', '.join(outputs)}")
        if deps[i]:
            dep_names = [records[d]["tool_name"] for d in deps[i]]
            lines.append(f"#   depends: {', '.join(dep_names)}")
        if rc != 0:
            lines.append(f"#   WARNING: original run exited with code {rc}")

        # Shell-quote the command
        cmd_str = " ".join(_shell_quote(arg) for arg in cmd)
        lines.append(cmd_str)
        lines.append("")

    return "\n".join(lines)


def render_snakemake(records: list[dict], deps: dict[int, list[int]]) -> str:
    """Render the pipeline as a Snakemake workflow."""
    lines = [
        "# Snakemake workflow reconstructed from DSAGT execution records",
        "",
    ]

    rule_names = []
    for i, record in enumerate(records):
        tool = record["tool_name"]
        rule_name = f"{tool}_{i + 1}"
        rule_names.append(rule_name)

    # Add an 'all' rule that depends on every step's outputs
    all_outputs = []
    for record in records:
        all_outputs.extend(record["execution"].get("output_files", []))
    if all_outputs:
        lines.append("rule all:")
        lines.append("    input:")
        for f in all_outputs:
            lines.append(f'        "{f}",')
        lines.append("")

    for i, record in enumerate(records):
        execution = record["execution"]
        cmd = execution["exact_command"]
        inputs = execution.get("input_files", [])
        outputs = execution.get("output_files", [])

        lines.append(f"rule {rule_names[i]}:")
        if inputs:
            lines.append("    input:")
            for f in inputs:
                lines.append(f'        "{f}",')
        if outputs:
            lines.append("    output:")
            for f in outputs:
                lines.append(f'        "{f}",')

        cmd_str = " ".join(_shell_quote(arg) for arg in cmd)
        lines.append("    shell:")
        lines.append(f'        "{cmd_str}"')
        lines.append("")

    return "\n".join(lines)


def _shell_quote(s: str) -> str:
    """Quote a string for shell if it contains special characters."""
    if not s:
        return "''"
    safe = all(c.isalnum() or c in "-_./:=@+" for c in s)
    if safe:
        return s
    return "'" + s.replace("'", "'\\''") + "'"


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def reconstruct_pipeline(
    trace_dir: Path,
    session_id: str | None = None,
    fmt: str = "bash",
) -> str:
    """Reconstruct a pipeline from execution records.

    Parameters
    ----------
    trace_dir : Path
        Path to the trace_archive directory.
    session_id : str, optional
        Filter records to a specific session.
    fmt : str
        Output format: "bash" or "snakemake".

    Returns
    -------
    str
        The rendered pipeline script.
    """
    records = load_pipeline_records(trace_dir, session_id)
    if not records:
        return f"# No execution records found{' for session ' + session_id if session_id else ''}\n"

    deps = build_dependency_graph(records)

    if fmt == "snakemake":
        return render_snakemake(records, deps)
    return render_bash(records, deps)
