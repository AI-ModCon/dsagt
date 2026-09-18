"""
Tests for pipeline reconstruction from tool execution records.
"""

import json
from pathlib import Path


from dsagt.provenance import (
    build_dependency_graph,
    load_pipeline_records,
    reconstruct_pipeline,
    render_bash,
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

    def test_failed_run_is_kept_as_a_comment(self):
        """A run that exited non-zero stays on the record as a comment; the
        script opens with ``set -e``, so a live failed step would stop it."""
        records = [
            _make_record("bad", ["bad", "--x"], return_code=1, record_id="r1"),
            _make_record(
                "good", ["good"], record_id="r2", timestamp="2024-01-15T11:00:00Z"
            ),
        ]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps)

        assert "#   failed with exit code 1; kept as a comment" in script
        assert "\n# bad --x\n" in script
        assert "\ngood\n" in script

    def test_paths_under_the_project_are_relative(self, tmp_path):
        """Absolute paths inside the project are written relative to it, so
        the script runs from the project directory or another checkout."""
        project = tmp_path / "proj"
        records = [
            _make_record(
                "conv",
                [
                    "python",
                    str(project / "codes/conv/x.py"),
                    str(project / "data/in.csv"),
                    "/tmp/out.csv",
                ],
                input_files=[str(project / "data/in.csv")],
                output_files=["/tmp/out.csv"],
            )
        ]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps, project_dir=project)

        assert "python codes/conv/x.py data/in.csv /tmp/out.csv" in script
        assert "inputs:  data/in.csv" in script
        assert str(project) not in script

    def test_quotes_special_characters(self):
        records = [_make_record("echo", ["echo", "hello world", "it's"])]
        deps = build_dependency_graph(records)
        script = render_bash(records, deps)

        assert "'hello world'" in script


# ---------------------------------------------------------------------------
# render_snakemake
# ---------------------------------------------------------------------------


class TestRenderBashReplays:
    """The script runs on a fresh copy of the project."""

    def test_output_directories_are_created_at_the_top(self):
        records = [
            _make_record(
                "plot", ["plot"], output_files=["plots/a.png"], record_id="r1"
            ),
            _make_record(
                "convert",
                ["convert"],
                output_files=["processed_data/tmp/x.json", "plots/b.png"],
                record_id="r2",
            ),
        ]
        script = render_bash(records, build_dependency_graph(records))
        head = script.split("# Step 1")[0]
        assert "mkdir -p plots processed_data/tmp" in head

    def test_a_repeated_output_is_removed_before_the_step_that_rewrites_it(self):
        records = [
            _make_record(
                "conv", ["conv", "a"], output_files=["out.json"], record_id="r1"
            ),
            _make_record(
                "conv", ["conv", "b"], output_files=["out.json"], record_id="r2"
            ),
        ]
        script = render_bash(records, build_dependency_graph(records))
        first, second = script.split("# Step 2")
        assert "rm -f out.json" not in first
        assert "rm -f out.json\nconv b" in second

    def test_a_stdout_file_is_a_redirect(self):
        record = _make_record(
            "aidrin",
            ["aidrin", "data-quality", "f.csv"],
            output_files=["audit/pre.json"],
            record_id="r1",
        )
        record["execution"]["stdout_file"] = "audit/pre.json"
        script = render_bash([record], build_dependency_graph([record]))
        assert "aidrin data-quality f.csv > audit/pre.json" in script

    def test_an_ad_hoc_run_is_named_as_one(self):
        record = _make_record("", ["python", "compare.py"], record_id="r1")
        script = render_bash([record], build_dependency_graph([record]))
        assert "# Step 1: ad-hoc run" in script
        assert "python compare.py" in script


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

    def test_empty_returns_comment(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        result = reconstruct_pipeline(tmp_path)
        assert "No execution records found" in result

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


# ---------------------------------------------------------------------------
# readiness_reports
# ---------------------------------------------------------------------------


class TestReadinessReports:

    def _aidrin_record(self, project, path, digest, record_id, ts, report):
        rec = _make_record(
            "aidrin",
            ["aidrin", "data-quality", path, "--detail"],
            input_files=[path],
            output_files=[report],
            record_id=record_id,
            timestamp=ts,
        )
        rec["execution"]["file_hashes"] = {path: digest}
        rec["execution"]["stdout_file"] = report
        _write_record(project / "trace_archive", rec)

    def test_reports_for_a_file_newest_first_with_change_status(self, tmp_path):
        import hashlib

        from dsagt.provenance import readiness_reports

        project = tmp_path
        (project / "data").mkdir()
        (project / "data" / "t.csv").write_text("a\n1\n")
        digest = hashlib.sha256(b"a\n1\n").hexdigest()
        self._aidrin_record(
            project,
            "data/t.csv",
            "0" * 64,
            "r1",
            "2026-01-01T00:00:00Z",
            "audit/pre.json",
        )
        self._aidrin_record(
            project,
            "data/t.csv",
            digest,
            "r2",
            "2026-01-01T01:00:00Z",
            "audit/post.json",
        )
        # A record for another file and a record from another code are left out.
        self._aidrin_record(
            project, "data/u.csv", digest, "r3", "2026-01-01T02:00:00Z", "audit/u.json"
        )
        _write_record(
            project / "trace_archive",
            _make_record(
                "convert",
                ["convert", "data/t.csv"],
                input_files=["data/t.csv"],
                record_id="r4",
            ),
        )
        reports = readiness_reports(project, "data/t.csv")
        assert [r["report"] for r in reports] == ["audit/post.json", "audit/pre.json"]
        assert reports[0]["unchanged"] is True
        assert reports[1]["unchanged"] is False

    def test_absolute_path_under_the_project_matches(self, tmp_path):
        from dsagt.provenance import readiness_reports

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "t.csv").write_text("x\n")
        self._aidrin_record(
            tmp_path,
            "data/t.csv",
            "0" * 64,
            "r1",
            "2026-01-01T00:00:00Z",
            "audit/pre.json",
        )
        assert len(readiness_reports(tmp_path, str(tmp_path / "data" / "t.csv"))) == 1

    def test_no_reports(self, tmp_path):
        from dsagt.provenance import readiness_reports

        assert readiness_reports(tmp_path, "data/none.csv") == []
