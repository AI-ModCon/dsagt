"""The headless-run checker separates dsagt's mechanical checks from outcomes."""

import importlib.util
import json
import sys
from pathlib import Path

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
        if not passed
    ]


def test_a_record_of_a_registered_code_passes_all_but_the_trace_count(tmp_path):
    # No mlflow.db in the fixture, so the trace count is the one failure.
    assert _failed(_project(tmp_path, "convert")) == [
        "one code.execute trace per record"
    ]


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
