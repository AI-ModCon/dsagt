"""The headless-run checker separates dsagt's mechanical checks from outcomes."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dsagt.agents.base import _NATIVE_DESCRIPTION_CAP

spec = importlib.util.spec_from_file_location(
    "headless_check", Path(__file__).parent / "headless_check.py"
)
headless_check = importlib.util.module_from_spec(spec)
sys.modules["headless_check"] = headless_check  # the dataclass looks its module up
spec.loader.exec_module(headless_check)


def _project(root: Path, code_name: str) -> Path:
    (root / ".dsagt").mkdir(parents=True)
    (root / ".dsagt" / "config.yaml").write_text("project: t\n")
    for skill in (*headless_check.BASE_SKILLS, "convert"):
        (root / "skills" / skill).mkdir(parents=True)
        (root / "skills" / skill / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: d\n"
            + (
                "executable: dsagt-run --code convert -- python c.py\n"
                if skill == "convert"
                else ""
            )
            + "---\n"
        )
    (root / ".claude" / "skills").mkdir(parents=True)
    (root / ".claude" / "skills" / "convert").symlink_to("../../skills/convert")
    (root / "trace_archive").mkdir()
    record = {
        "record_id": "r1",
        "code_name": code_name,
        "execution": {
            "exact_command": ["python", "c.py"],
            "return_code": 0,
            "timestamp_start": "2026-01-01T00:00:00+00:00",
            "input_files": [],
            "output_files": [],
            "file_hashes": {},
        },
    }
    (root / "trace_archive" / "convert_r1.json").write_text(json.dumps(record))
    return root


def _failed(root: Path) -> list[str]:
    project = headless_check.Project.load(root)
    return [
        label
        for label, passed, _ in headless_check.common_checks(project)
        if passed is False
    ]


def _result(root: Path, label: str):
    project = headless_check.Project.load(root)
    return next(r for r in headless_check.common_checks(project) if r[0] == label)


TRACE_COUNT = "one code.execute trace per record"
MIRROR = "native skills mirror links every skill under the description cap"


@pytest.fixture(autouse=True)
def _local_store(monkeypatch):
    """The fixtures are serverless projects; a shared store is opted into."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


def test_a_record_of_a_registered_code_passes_all_but_the_trace_count(tmp_path):
    # No mlflow.db in the fixture, so the trace count is the one failure.
    assert _failed(_project(tmp_path, "convert")) == [TRACE_COUNT]


def test_the_trace_count_is_not_checked_against_a_shared_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example:5000")
    _, passed, detail = _result(_project(tmp_path, "convert"), TRACE_COUNT)
    assert passed is None and "http://mlflow.example:5000" in detail
    assert _failed(tmp_path) == []


def test_a_record_missing_a_key_is_reported_not_raised(tmp_path):
    root = _project(tmp_path, "convert")
    record_path = root / "trace_archive" / "convert_r1.json"
    record = json.loads(record_path.read_text())
    del record["execution"]["file_hashes"]
    record_path.write_text(json.dumps(record))
    assert "every record is complete" in _failed(root)


def test_a_skill_over_the_native_description_cap_may_be_a_copy(tmp_path):
    root = _project(tmp_path, "convert")
    spec = (
        f"---\nname: verbose\ndescription: {'x' * (_NATIVE_DESCRIPTION_CAP + 1)}\n---\n"
    )
    (root / "skills" / "verbose").mkdir()
    (root / "skills" / "verbose" / "SKILL.md").write_text(spec)
    (root / ".claude" / "skills" / "verbose").mkdir()
    (root / ".claude" / "skills" / "verbose" / "SKILL.md").write_text(spec)
    _, passed, detail = _result(root, MIRROR)
    assert passed is True and "verbose" in detail


def test_a_copied_skill_under_the_cap_is_a_failure(tmp_path):
    root = _project(tmp_path, "convert")
    (root / ".claude" / "skills" / "convert").unlink()
    (root / ".claude" / "skills" / "convert").mkdir()
    (root / ".claude" / "skills" / "convert" / "SKILL.md").write_text(
        "---\nx: 1\n---\n"
    )
    assert MIRROR in _failed(root)


def test_an_output_path_matches_however_the_agent_spelled_it(tmp_path):
    root = _project(tmp_path, "convert")
    record_path = root / "trace_archive" / "convert_r1.json"
    record = json.loads(record_path.read_text())
    record["execution"]["output_files"] = ["./out/table.csv", str(root / "out/two.csv")]
    record_path.write_text(json.dumps(record))
    project = headless_check.Project.load(root)
    assert project.outputs_of(record) == ["out/table.csv", "out/two.csv"]
    for path in ("out/table.csv", "out/two.csv"):
        assert headless_check.output_has_record(path)(project)[1] is True


def test_only_a_non_zero_record_counts_as_a_failed_run(tmp_path):
    root = _project(tmp_path, "convert")
    record = json.loads((root / "trace_archive" / "convert_r1.json").read_text())
    record["record_id"] = "r2"
    record["execution"]["return_code"] = 1
    (root / "trace_archive" / "convert_r2.json").write_text(json.dumps(record))
    project = headless_check.Project.load(root)
    observe = headless_check.failed_record_count("failed runs", r"convert")
    assert observe(project) == ("failed runs", "1 of 2")


def test_a_record_naming_no_registered_code_is_a_mechanical_failure(tmp_path):
    assert "every record names a registered code" in _failed(_project(tmp_path, "gone"))


def test_every_walkthrough_with_execution_prompts_declares_checks():
    use_cases = Path(__file__).parent.parent / "use_cases"
    with_prompts = {
        readme.parent.name
        for readme in use_cases.glob("*/README.md")
        if "## Post-Conditions" in readme.read_text()
    }
    assert with_prompts - {"plasma_turbulence"} <= set(headless_check.WALKTHROUGHS)
