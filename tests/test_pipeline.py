"""
Tests for pipeline reconstruction from tool execution records.
"""

import json
from pathlib import Path


from dsagt.provenance import (
    build_dependency_graph,
    compute_terminal_outputs,
    load_pipeline_records,
    reconstruct_pipeline,
    render_bash,
    render_json,
    render_snakemake,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_record(trace_dir: Path, record: dict) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    rid = record.get("record_id", "r0")
    tool = record.get("code_name", "tool")
    path = trace_dir / f"{tool}_{rid}.json"
    path.write_text(json.dumps(record))
    return path


def _make_record(
    code_name: str,
    command: list[str],
    input_files: list[str] | None = None,
    output_files: list[str] | None = None,
    session_id: str = "s1",
    record_id: str = "r0",
    timestamp: str = "2024-01-15T10:00:00Z",
    return_code: int = 0,
) -> dict:
    return {
        "record_id": record_id,
        "code_name": code_name,
        "session_id": session_id,
        "execution": {
            "exact_command": command,
            "return_code": return_code,
            "stdout": "",
            "stderr": "",
            "timestamp_start": timestamp,
            "timestamp_end": timestamp,
            "input_files": input_files or [],
            "output_files": output_files or [],
        },
    }


# ---------------------------------------------------------------------------
# load_pipeline_records
# ---------------------------------------------------------------------------


class TestLoadPipelineRecords:

    def test_loads_wrapper_records(self, tmp_path):
        r = _make_record("fastp", ["fastp", "-q", "20"], record_id="r1")
        _write_record(tmp_path, r)

        records = load_pipeline_records(tmp_path)
        assert len(records) == 1
        assert records[0]["code_name"] == "fastp"

    def test_skips_proxy_only_records(self, tmp_path):
        """Records without execution layer are skipped."""
        proxy_record = {
            "record_id": "r1",
            "code_name": "fastp",
            "session_id": "s1",
            "intent": {"command": "fastp", "parameters": {}},
            "execution": None,
            "report": {"agent_output": "done"},
        }
        _write_record(tmp_path, proxy_record)

        records = load_pipeline_records(tmp_path)
        assert len(records) == 0

    def test_filters_by_session(self, tmp_path):
        _write_record(
            tmp_path, _make_record("a", ["a"], session_id="s1", record_id="r1")
        )
        _write_record(
            tmp_path, _make_record("b", ["b"], session_id="s2", record_id="r2")
        )

        records = load_pipeline_records(tmp_path, session_id="s1")
        assert len(records) == 1
        assert records[0]["code_name"] == "a"

    def test_sorted_by_timestamp(self, tmp_path):
        _write_record(
            tmp_path,
            _make_record(
                "late", ["late"], timestamp="2024-01-15T11:00:00Z", record_id="r2"
            ),
        )
        _write_record(
            tmp_path,
            _make_record(
                "early", ["early"], timestamp="2024-01-15T09:00:00Z", record_id="r1"
            ),
        )

        records = load_pipeline_records(tmp_path)
        assert records[0]["code_name"] == "early"
        assert records[1]["code_name"] == "late"

    def test_empty_directory(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        assert load_pipeline_records(tmp_path) == []

    def test_nonexistent_directory(self, tmp_path):
        assert load_pipeline_records(tmp_path / "missing") == []


# ---------------------------------------------------------------------------
# build_dependency_graph
# ---------------------------------------------------------------------------


class TestBuildDependencyGraph:

    def test_linear_dependency(self):
        """A → B: A produces file.fq, B consumes it."""
        records = [
            _make_record("fastp", ["fastp"], output_files=["clean.fq"]),
            _make_record("megahit", ["megahit"], input_files=["clean.fq"]),
        ]
        deps = build_dependency_graph(records)
        assert deps[0] == []
        assert deps[1] == [0]

    def test_no_dependencies(self):
        records = [
            _make_record("a", ["a"], output_files=["x.txt"]),
            _make_record("b", ["b"], output_files=["y.txt"]),
        ]
        deps = build_dependency_graph(records)
        assert deps[0] == []
        assert deps[1] == []

    def test_diamond_dependency(self):
        """A produces two files; B and C each consume one; D consumes both."""
        records = [
            _make_record("a", ["a"], output_files=["x.fq", "y.fq"]),
            _make_record("b", ["b"], input_files=["x.fq"], output_files=["bx.txt"]),
            _make_record("c", ["c"], input_files=["y.fq"], output_files=["cy.txt"]),
            _make_record("d", ["d"], input_files=["bx.txt", "cy.txt"]),
        ]
        deps = build_dependency_graph(records)
        assert deps[0] == []
        assert deps[1] == [0]
        assert deps[2] == [0]
        assert sorted(deps[3]) == [1, 2]

    def test_self_dependency_excluded(self):
        """A tool that lists the same file as input and output doesn't depend on itself."""
        records = [
            _make_record("a", ["a"], input_files=["x.txt"], output_files=["x.txt"]),
        ]
        deps = build_dependency_graph(records)
        assert deps[0] == []


# ---------------------------------------------------------------------------
# render_bash
# ---------------------------------------------------------------------------


class TestRenderBash:

    def test_basic_script(self):
        records = [_make_record("fastp", ["fastp", "-q", "20", "--in1", "reads.fq.gz"])]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps)

        assert "#!/usr/bin/env bash" in script
        assert "set -euo pipefail" in script
        assert "fastp -q 20 --in1 reads.fq.gz" in script
        assert "Step 1: fastp" in script

    def test_includes_file_comments(self):
        records = [
            _make_record(
                "fastp", ["fastp"], input_files=["raw.fq"], output_files=["clean.fq"]
            ),
        ]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps)

        assert "inputs:  raw.fq" in script
        assert "outputs: clean.fq" in script

    def test_includes_dependency_comments(self):
        records = [
            _make_record("fastp", ["fastp"], output_files=["clean.fq"]),
            _make_record("megahit", ["megahit"], input_files=["clean.fq"]),
        ]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps)

        assert "depends: fastp" in script

    def test_warns_on_nonzero_exit(self):
        records = [_make_record("bad", ["bad"], return_code=1)]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps)

        assert "WARNING: original run exited with code 1" in script

    def test_quotes_special_characters(self):
        records = [_make_record("echo", ["echo", "hello world", "it's"])]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps)

        assert "'hello world'" in script


# ---------------------------------------------------------------------------
# render_snakemake
# ---------------------------------------------------------------------------


class TestRenderSnakemake:

    def test_basic_workflow(self):
        records = [
            _make_record(
                "fastp",
                ["fastp", "-q", "20"],
                input_files=["raw.fq"],
                output_files=["clean.fq"],
            ),
        ]
        deps = build_dependency_graph(records)
        workflow = render_snakemake(records, deps)

        assert "rule fastp_1:" in workflow
        assert '"raw.fq"' in workflow
        assert '"clean.fq"' in workflow
        assert "rule all:" in workflow

    def test_multi_step(self):
        records = [
            _make_record("fastp", ["fastp"], output_files=["clean.fq"]),
            _make_record(
                "megahit",
                ["megahit"],
                input_files=["clean.fq"],
                output_files=["contigs.fa"],
            ),
        ]
        deps = build_dependency_graph(records)
        workflow = render_snakemake(records, deps)

        assert "rule fastp_1:" in workflow
        assert "rule megahit_2:" in workflow
        assert "rule all:" in workflow
        assert '"contigs.fa"' in workflow


# ---------------------------------------------------------------------------
# compute_terminal_outputs
# ---------------------------------------------------------------------------


class TestComputeTerminalOutputs:

    def test_diamond_dependency(self):
        """A produces two files; B and C each consume one; D combines them into
        a final output that nothing downstream consumes."""
        records = [
            _make_record("a", ["a"], output_files=["x.fq", "y.fq"]),
            _make_record("b", ["b"], input_files=["x.fq"], output_files=["bx.txt"]),
            _make_record("c", ["c"], input_files=["y.fq"], output_files=["cy.txt"]),
            _make_record(
                "d",
                ["d"],
                input_files=["bx.txt", "cy.txt"],
                output_files=["final.txt"],
            ),
        ]
        assert compute_terminal_outputs(records) == ["final.txt"]

    def test_independent_leaves(self):
        """Several unrelated single-step branches: every output is terminal."""
        records = [
            _make_record("a", ["a"], output_files=["a.out"]),
            _make_record("b", ["b"], output_files=["b.out"]),
            _make_record("c", ["c"], output_files=["c.out"]),
        ]
        assert compute_terminal_outputs(records) == ["a.out", "b.out", "c.out"]

    def test_linear_pipeline_only_last_output_terminal(self):
        records = [
            _make_record("a", ["a"], output_files=["x.txt"]),
            _make_record("b", ["b"], input_files=["x.txt"], output_files=["y.txt"]),
        ]
        assert compute_terminal_outputs(records) == ["y.txt"]

    def test_no_outputs(self):
        records = [_make_record("a", ["a"])]
        assert compute_terminal_outputs(records) == []

    def test_duplicate_output_deduplicated(self):
        """The same terminal file produced twice only appears once."""
        records = [
            _make_record("a", ["a"], output_files=["x.txt"], record_id="r1"),
            _make_record("b", ["b"], output_files=["x.txt"], record_id="r2"),
        ]
        assert compute_terminal_outputs(records) == ["x.txt"]


# ---------------------------------------------------------------------------
# render_json
# ---------------------------------------------------------------------------


class TestRenderJson:

    def test_structure(self):
        records = [
            _make_record(
                "fastp", ["fastp"], input_files=["raw.fq"], output_files=["clean.fq"]
            ),
        ]
        deps = build_dependency_graph(records)
        result = json.loads(render_json(records, deps))

        assert result["records"] == records
        assert result["dependency_graph"] == {"0": []}
        assert result["terminal_outputs"] == ["clean.fq"]

    def test_diamond_terminal_outputs(self):
        records = [
            _make_record("a", ["a"], output_files=["x.fq", "y.fq"]),
            _make_record("b", ["b"], input_files=["x.fq"], output_files=["bx.txt"]),
            _make_record("c", ["c"], input_files=["y.fq"], output_files=["cy.txt"]),
            _make_record(
                "d",
                ["d"],
                input_files=["bx.txt", "cy.txt"],
                output_files=["final.txt"],
            ),
        ]
        deps = build_dependency_graph(records)
        result = json.loads(render_json(records, deps))

        assert result["dependency_graph"] == {"0": [], "1": [0], "2": [0], "3": [1, 2]}
        assert result["terminal_outputs"] == ["final.txt"]


# ---------------------------------------------------------------------------
# reconstruct_pipeline (end-to-end)
# ---------------------------------------------------------------------------


class TestReconstructPipeline:

    def test_bash_format(self, tmp_path):
        _write_record(
            tmp_path, _make_record("fastp", ["fastp", "-q", "20"], record_id="r1")
        )
        script = reconstruct_pipeline(tmp_path, fmt="bash")

        assert "#!/usr/bin/env bash" in script
        assert "fastp -q 20" in script

    def test_snakemake_format(self, tmp_path):
        _write_record(
            tmp_path,
            _make_record(
                "fastp",
                ["fastp"],
                input_files=["raw.fq"],
                output_files=["clean.fq"],
                record_id="r1",
            ),
        )
        workflow = reconstruct_pipeline(tmp_path, fmt="snakemake")

        assert "rule fastp_1:" in workflow

    def test_json_format(self, tmp_path):
        _write_record(
            tmp_path,
            _make_record(
                "fastp",
                ["fastp"],
                input_files=["raw.fq"],
                output_files=["clean.fq"],
                record_id="r1",
            ),
        )
        result = json.loads(reconstruct_pipeline(tmp_path, fmt="json"))

        assert len(result["records"]) == 1
        assert result["records"][0]["code_name"] == "fastp"
        assert result["dependency_graph"] == {"0": []}
        assert result["terminal_outputs"] == ["clean.fq"]

    def test_empty_returns_comment(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        result = reconstruct_pipeline(tmp_path)
        assert "No execution records found" in result

    def test_empty_json_format(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        result = json.loads(reconstruct_pipeline(tmp_path, fmt="json"))
        assert result == {"records": [], "dependency_graph": {}, "terminal_outputs": []}

    def test_session_filter(self, tmp_path):
        _write_record(
            tmp_path, _make_record("a", ["a"], session_id="s1", record_id="r1")
        )
        _write_record(
            tmp_path, _make_record("b", ["b"], session_id="s2", record_id="r2")
        )

        script = reconstruct_pipeline(tmp_path, session_id="s1", fmt="bash")
        # Only the s1 record survives the filter: exactly one step (tool "a"),
        # the s2 record ("b") is excluded — so there is no second step.
        assert "Step 1: a" in script
        assert "Step 2:" not in script

    def test_full_pipeline_with_deps(self, tmp_path):
        """Three-step pipeline: fastp → megahit → quast."""
        _write_record(
            tmp_path,
            _make_record(
                "fastp",
                ["fastp", "-q", "20", "--in1", "raw.fq.gz"],
                output_files=["clean.fq.gz"],
                timestamp="2024-01-15T10:00:00Z",
                record_id="r1",
            ),
        )
        _write_record(
            tmp_path,
            _make_record(
                "megahit",
                ["megahit", "-r", "clean.fq.gz", "-o", "assembly"],
                input_files=["clean.fq.gz"],
                output_files=["assembly/final.contigs.fa"],
                timestamp="2024-01-15T10:10:00Z",
                record_id="r2",
            ),
        )
        _write_record(
            tmp_path,
            _make_record(
                "quast",
                ["quast", "assembly/final.contigs.fa", "-o", "quast_out"],
                input_files=["assembly/final.contigs.fa"],
                timestamp="2024-01-15T10:20:00Z",
                record_id="r3",
            ),
        )

        script = reconstruct_pipeline(tmp_path, session_id="s1", fmt="bash")

        # Steps in correct order
        lines = script.splitlines()
        step_lines = [line for line in lines if line.startswith("# Step")]
        assert "fastp" in step_lines[0]
        assert "megahit" in step_lines[1]
        assert "quast" in step_lines[2]

        # Dependencies noted
        assert "depends: fastp" in script
        assert "depends: megahit" in script
