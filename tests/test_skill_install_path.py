"""One install path for every skill.

A skill reaches a project four ways: as a base skill at ``dsagt init``, as a
catalog skill through ``install_skill``, as an agent-authored skill through
``save_skill``, and, for one script with no workflow, through
``save_code_spec``.  Whichever way it comes, the project ends in one state:
the skill under ``skills/<name>/``, a registered code per script, the
skill's markdown rewritten to the stored commands, and the specs indexed.
These tests hold the four entry points to that one state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dsagt.registry import CodeRegistry, SkillRegistry, _parse_frontmatter
from dsagt.skills import register_skill_scripts

SKILL_MD = """\
---
name: trim-reads
description: Trim adapters from interleaved reads with the bundled script.
---

# trim-reads

Run the trimmer on a sample:

```bash
python3 scripts/trim.py --reads data/s1.fq.gz --out data/s1.trimmed.fq.gz
```

Then inspect with `python scripts/trim.py --help`.
"""

TRIM_PY = '''"""Trim adapters from an interleaved FASTQ."""
import argparse

p = argparse.ArgumentParser(description="Trim adapters from an interleaved FASTQ.")
p.add_argument("--reads", required=True, help="Interleaved FASTQ to trim")
p.add_argument("--out", required=True, help="Trimmed FASTQ to write")
p.add_argument("--min-length", type=int, default=50, help="Drop reads shorter than this")
a = p.parse_args()
open(a.out, "w").write(open(a.reads).read())
'''


def _write_skill(root: Path) -> Path:
    skill = root / "trim-reads"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text(SKILL_MD)
    (skill / "scripts" / "trim.py").write_text(TRIM_PY)
    (skill / "scripts" / "_helpers.py").write_text("X = 1\n")
    return skill


def _state(project: Path) -> dict:
    """What the project holds for the skill, for equality across entry points."""
    # Frontmatter and body, each stripped: save_skill and a copied file differ
    # only in the blank line between them.
    _, frontmatter, body = (
        (project / "skills" / "trim-reads" / "SKILL.md").read_text().split("---", 2)
    )
    skill_md = (frontmatter.strip(), body.strip())
    codes = {
        p.parent.name: _parse_frontmatter(p)
        for p in sorted((project / "skills").glob("*/SKILL.md"))
        if _parse_frontmatter(p).get("executable")
    }
    return {"skill_md": skill_md, "codes": codes}


@pytest.fixture
def fresh_project(tmp_path):
    def make(name: str) -> Path:
        project = tmp_path / name
        for sub in ("skills", "trace_archive", ".dsagt"):
            (project / sub).mkdir(parents=True)
        return project

    return make


def _via_copy(project: Path, skill_src: Path) -> list[str]:
    """The base-skill and catalog entry points: the directory is copied into
    ``skills/`` and registered (``install_into_project`` plus the shared
    registration)."""
    import shutil

    shutil.copytree(skill_src, project / "skills" / "trim-reads")
    return register_skill_scripts(project, "trim-reads")


def _via_save_skill(project: Path, skill_src: Path) -> list[str]:
    from dsagt.mcp.skill_tools import _handle_save_skill
    import asyncio

    spec = {
        "name": "trim-reads",
        "description": _parse_frontmatter(skill_src / "SKILL.md")["description"],
    }
    body = (skill_src / "SKILL.md").read_text().split("---", 2)[2].lstrip("\n")
    reply = asyncio.run(
        _handle_save_skill(
            {
                "spec": spec,
                "body": body,
                "reference_files": {
                    "scripts/trim.py": TRIM_PY,
                    "scripts/_helpers.py": "X = 1\n",
                },
            },
            skill_registry=SkillRegistry(runtime_dir=project, kb=None),
        )
    )
    assert "Run it as:" in reply
    return [line for line in reply.splitlines() if "dsagt-run" in line]


class TestOneState:

    def test_copy_and_save_skill_give_the_same_project(self, tmp_path, fresh_project):
        skill_src = _write_skill(tmp_path / "src")
        a = fresh_project("a")
        b = fresh_project("b")
        _via_copy(a, skill_src)
        _via_save_skill(b, skill_src)
        assert _state(a) == _state(b)

    def test_every_script_is_a_code_with_the_stored_command(
        self, tmp_path, fresh_project
    ):
        skill_src = _write_skill(tmp_path / "src")
        project = fresh_project("p")
        lines = _via_copy(project, skill_src)
        code = CodeRegistry(runtime_dir=project).get_code("trim-reads-trim")
        assert code is not None
        assert code["executable"] == (
            "dsagt-run --code trim-reads-trim -- python skills/trim-reads/scripts/trim.py"
        )
        assert code["tags"] == ["trim-reads"]
        # A helper module (leading underscore) is not a code.
        assert CodeRegistry(runtime_dir=project).get_code("trim-reads--helpers") is None
        assert lines == [code["executable"]]

    def test_argparse_gives_the_parameters(self, tmp_path, fresh_project):
        skill_src = _write_skill(tmp_path / "src")
        project = fresh_project("p")
        _via_copy(project, skill_src)
        code = CodeRegistry(runtime_dir=project).get_code("trim-reads-trim")
        assert code["description"] == "Trim adapters from an interleaved FASTQ."
        params = code["parameters"]
        assert params["reads"] == {
            "type": "string",
            "required": True,
            "cli": "--reads",
            "description": "Interleaved FASTQ to trim",
        }
        assert params["out"]["cli"] == "--out"
        assert params["min_length"] == {
            "type": "integer",
            "required": False,
            "cli": "--min-length",
            "default": 50,
            "description": "Drop reads shorter than this",
        }

    def test_bare_invocations_in_the_skill_text_become_the_stored_command(
        self, tmp_path, fresh_project
    ):
        skill_src = _write_skill(tmp_path / "src")
        project = fresh_project("p")
        _via_copy(project, skill_src)
        text = (project / "skills" / "trim-reads" / "SKILL.md").read_text()
        stored = "dsagt-run --code trim-reads-trim -- python skills/trim-reads/scripts/trim.py"
        assert f"{stored} --reads data/s1.fq.gz" in text
        assert f"`{stored} --help`" in text
        assert "python3 scripts/trim.py" not in text
        provenance = (project / "skills" / "trim-reads" / "PROVENANCE.txt").read_text()
        assert "scripts/trim.py" in provenance and stored in provenance

    def test_registration_is_idempotent(self, tmp_path, fresh_project):
        skill_src = _write_skill(tmp_path / "src")
        project = fresh_project("p")
        _via_copy(project, skill_src)
        first = _state(project)
        register_skill_scripts(project, "trim-reads")
        assert _state(project) == first

    def test_an_override_wins_over_derivation(self, tmp_path, fresh_project):
        skill_src = _write_skill(tmp_path / "src")
        project = fresh_project("p")
        override = {
            "name": "trim",
            "description": "Trim reads, the curated description.",
            "parameters": {
                "reads": {
                    "type": "string",
                    "required": True,
                    "cli": "--reads",
                    "role": "input",
                },
                "out": {
                    "type": "string",
                    "required": True,
                    "cli": "--out",
                    "role": "output",
                },
            },
        }
        (project / "skills" / "trim-reads").mkdir()
        import shutil

        shutil.copytree(
            skill_src, project / "skills" / "trim-reads", dirs_exist_ok=True
        )
        register_skill_scripts(
            project, "trim-reads", overrides={"scripts/trim.py": override}
        )
        code = CodeRegistry(runtime_dir=project).get_code("trim")
        assert code["description"] == "Trim reads, the curated description."
        assert code["parameters"]["reads"]["role"] == "input"
        text = (project / "skills" / "trim-reads" / "SKILL.md").read_text()
        assert (
            "dsagt-run --code trim -- python skills/trim-reads/scripts/trim.py --reads"
            in text
        )


class TestSaveCodeSpec:

    def test_a_code_spec_is_a_one_script_skill(self, fresh_project):
        """``save_code_spec`` writes the same object a skill's script becomes:
        a directory under ``skills/`` whose frontmatter declares the executable."""
        project = fresh_project("p")
        CodeRegistry(runtime_dir=project).save_tool(
            {
                "name": "greet",
                "description": "Print a greeting.",
                "executable": "python greet.py",
                "parameters": {},
            }
        )
        path = project / "skills" / "greet" / "SKILL.md"
        assert path.exists()
        spec = _parse_frontmatter(path)
        assert spec["executable"] == "dsagt-run --code greet -- python greet.py"
        assert (project / "skills" / "greet") in SkillRegistry(
            runtime_dir=project
        ).skill_dirs()
        assert not (project / "codes").exists()


class TestWhatIsAScript:

    def test_a_module_without_an_entry_point_is_not_a_code(
        self, tmp_path, fresh_project
    ):
        skill_src = _write_skill(tmp_path / "src")
        (skill_src / "scripts" / "models.py").write_text("class Card:\n    pass\n")
        (skill_src / "scripts" / "run.sh").write_text("echo hi\n")
        project = fresh_project("p")
        _via_copy(project, skill_src)
        registry = CodeRegistry(runtime_dir=project)
        assert registry.get_code("trim-reads-models") is None
        assert registry.get_code("trim-reads-run")["executable"].endswith(
            "-- bash skills/trim-reads/scripts/run.sh"
        )

    def test_a_mid_sentence_script_invocation_is_rewritten(
        self, tmp_path, fresh_project
    ):
        skill_src = _write_skill(tmp_path / "src")
        md = skill_src / "SKILL.md"
        md.write_text(
            md.read_text() + "\n- [ ] 2. Run python3 scripts/trim.py on the sample\n"
        )
        project = fresh_project("p")
        _via_copy(project, skill_src)
        text = (project / "skills" / "trim-reads" / "SKILL.md").read_text()
        assert (
            "Run dsagt-run --code trim-reads-trim -- python skills/trim-reads/scripts/trim.py on the sample"
            in text
        )


class TestGuessedWrapperLine:

    def test_a_wrapped_line_under_the_wrong_name_takes_the_stored_command(
        self, tmp_path, fresh_project
    ):
        skill_src = _write_skill(tmp_path / "src")
        md = skill_src / "SKILL.md"
        md.write_text(
            md.read_text()
            + "\n```bash\ndsagt-run --code trim-reads -- python scripts/trim.py --reads r.fq --out o.fq\n```\n"
        )
        project = fresh_project("p")
        _via_copy(project, skill_src)
        text = (project / "skills" / "trim-reads" / "SKILL.md").read_text()
        stored = "dsagt-run --code trim-reads-trim -- python skills/trim-reads/scripts/trim.py"
        assert f"{stored} --reads r.fq --out o.fq" in text
        assert "dsagt-run --code trim-reads --" not in text
        provenance = (project / "skills" / "trim-reads" / "PROVENANCE.txt").read_text()
        # Only the pairs that changed a line are on record, not all eight.
        assert provenance.count("rewritten by dsagt") == 2


class TestAnAgentSavedSpecIsKept:

    def test_a_second_registration_keeps_roles_and_dependencies(
        self, tmp_path, fresh_project
    ):
        """vasp, both agents: save_code_spec added dependencies, a later
        save_skill derived the spec again from argparse and dropped them."""
        skill_src = _write_skill(tmp_path / "src")
        project = fresh_project("p")
        _via_copy(project, skill_src)
        registry = CodeRegistry(runtime_dir=project)
        spec = registry.get_code("trim-reads-trim")
        registry.save_tool(
            {
                "name": "trim-reads-trim",
                "description": spec["description"],
                "executable": "python skills/trim-reads/scripts/trim.py",
                "parameters": spec["parameters"],
                "dependencies": ["pysam"],
            }
        )
        with_deps = CodeRegistry(runtime_dir=project).get_code("trim-reads-trim")
        assert "uv run --with pysam" in with_deps["executable"]
        register_skill_scripts(project, "trim-reads")
        again = CodeRegistry(runtime_dir=project).get_code("trim-reads-trim")
        assert again["executable"] == with_deps["executable"]
        # The skill's usage lines follow the stored command.
        text = (project / "skills" / "trim-reads" / "SKILL.md").read_text()
        assert f"{with_deps['executable']} --reads data/s1.fq.gz" in text

    def test_a_script_registered_under_its_own_name_is_not_registered_again(
        self, tmp_path, fresh_project
    ):
        """blastnet, Claude Code: save_code_spec registered a skill's script as
        well-sample-points with h5py and file roles; a later save_skill on the
        skill registered the same file again under the derived name, with
        neither, and that code failed wherever h5py was not installed."""
        import shutil

        skill_src = _write_skill(tmp_path / "src")
        project = fresh_project("p")
        shutil.copytree(skill_src, project / "skills" / "trim-reads")
        CodeRegistry(runtime_dir=project).save_tool(
            {
                "name": "trim",
                "description": "Trim adapters from an interleaved FASTQ.",
                "executable": "python ./skills/trim-reads/scripts/trim.py",
                "parameters": {
                    "reads": {"type": "string", "cli": "--reads", "role": "input"},
                    "out": {"type": "string", "cli": "--out", "role": "output"},
                },
                "dependencies": ["pysam"],
            }
        )
        stored = register_skill_scripts(project, "trim-reads")
        registry = CodeRegistry(runtime_dir=project)
        assert registry.get_code("trim-reads-trim") is None
        assert [c["name"] for c in registry.list_codes_raw()] == ["trim"]
        trim = registry.get_code("trim")
        assert "uv run --with pysam" in trim["executable"]
        assert stored == [trim["executable"]]
        text = (project / "skills" / "trim-reads" / "SKILL.md").read_text()
        assert f"{trim['executable']} --reads data/s1.fq.gz" in text
