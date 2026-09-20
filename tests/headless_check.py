"""Check a finished headless walkthrough run: mechanical checks, then outcome observations.

A headless run (``tests/headless_usecases.py``) leaves a project directory.
This script reads it and reports two things that are scored apart.
Mechanical checks are properties of dsagt that hold in every run whatever the
agent chose: the skills and codes are installed, every run of a registered
code has a complete record and a ``code.execute`` trace, the logs have no
error.  A mechanical failure is a dsagt regression and the exit code is 1.
Outcome observations depend on the agent (a value in a converted file, whether
a datacard validated, how many samples were assembled); they are printed as
values, never as pass or fail, and with several projects as a rate, because
one run moves by a post-condition or two with the same inputs.

Left out on purpose, because a headless session cannot show them: a step that
needs a second answer from a person, the last prompt's conversation trace
(collected at the next session start), and the content of a reply.

    python tests/headless_check.py use_cases/vasp_dft vasp-r1 [vasp-r2 ...]

Each walkthrough's checks are an entry in ``WALKTHROUGHS``, keyed by its
directory name; the checks shared by every walkthrough are in ``common_checks``.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dsagt.registry import _parse_frontmatter

BASE_SKILLS = ("skill-creator", "datacard-generator", "aidrin")
NATIVE_SKILL_DIRS = (".claude/skills", ".agents/skills", ".cline/skills")
PIPELINE_SCRIPT_NAMES = ("pipeline.sh", "dsagt_session_script.sh", "Snakefile")


@dataclass
class Project:
    """A finished run's directory, read once."""

    root: Path
    records: list[dict] = field(default_factory=list)
    codes: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "Project":
        if not (root / ".dsagt" / "config.yaml").exists():
            raise SystemExit(f"{root} is not a dsagt project (no .dsagt/config.yaml)")
        records = [
            json.loads(path.read_text())
            for path in sorted((root / "trace_archive").glob("*.json"))
        ]
        codes = {}
        for spec_path in sorted((root / "skills").glob("*/SKILL.md")):
            # dsagt's own parser, so a SKILL.md the registry reads is one the
            # checker reads.
            frontmatter = _parse_frontmatter(spec_path)
            if frontmatter.get("executable"):
                codes[frontmatter.get("name", spec_path.parent.name)] = frontmatter
        return cls(root=root, records=records, codes=codes)

    def records_of(self, pattern: str) -> list[dict]:
        return [r for r in self.records if re.search(pattern, r["code_name"])]

    def files(self, glob: str) -> list[Path]:
        return sorted(self.root.glob(glob))

    def query(self, database: str, sql: str) -> list[tuple]:
        """Rows from one of the project's sqlite files, opened read-only."""
        path = self.root / database
        if not path.exists():
            return []
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return connection.execute(sql).fetchall()
        finally:
            connection.close()


# ---------------------------------------------------------------------------
# Mechanical checks: each returns (name, passed, detail)
# ---------------------------------------------------------------------------


def common_checks(project: Project) -> list[tuple[str, bool, str]]:
    root = project.root
    results = []

    missing = [
        s for s in BASE_SKILLS if not (root / "skills" / s / "SKILL.md").exists()
    ]
    results.append(
        ("base skills installed", not missing, f"missing: {missing}" if missing else "")
    )

    results.append(("no codes/ directory", not (root / "codes").exists(), ""))

    mirrors = [root / d for d in NATIVE_SKILL_DIRS if (root / d).is_dir()]
    copied = [
        str(entry.relative_to(root))
        for mirror in mirrors
        for entry in mirror.iterdir()
        if entry.is_dir() and not entry.is_symlink()
    ]
    results.append(
        (
            "native skills mirror is symlinks",
            bool(mirrors) and not copied,
            f"copies: {copied}" if copied else "",
        )
    )

    unregistered = sorted(
        {r["code_name"] for r in project.records if r["code_name"] not in project.codes}
    )
    results.append(
        (
            "every record names a registered code",
            not unregistered,
            f"unregistered: {unregistered}" if unregistered else "",
        )
    )

    incomplete = [
        r["record_id"]
        for r in project.records
        if not {
            "exact_command",
            "return_code",
            "timestamp_start",
            "input_files",
            "output_files",
            "file_hashes",
        }
        <= set(r["execution"])
    ]
    results.append(
        (
            "every record is complete",
            not incomplete,
            f"incomplete: {incomplete}" if incomplete else "",
        )
    )

    unhashed = [
        f"{r['record_id']}:{f}"
        for r in project.records
        if r["execution"]["return_code"] == 0
        for f in r["execution"]["output_files"]
        if (root / f).is_file() and f not in r["execution"]["file_hashes"]
    ]
    results.append(
        (
            "every written output has a hash",
            not unhashed,
            f"unhashed: {unhashed[:5]}" if unhashed else "",
        )
    )

    traced = project.query(
        "mlflow.db",
        "select count(*) from trace_tags where key = 'dsagt.source' and value = 'execution'",
    )
    n_traced = traced[0][0] if traced else 0
    results.append(
        (
            "one code.execute trace per record",
            n_traced == len(project.records),
            f"{n_traced} traces, {len(project.records)} records",
        )
    )

    trace_log = root / ".dsagt" / "run_trace.log"
    failed = trace_log.read_text().strip() if trace_log.exists() else ""
    results.append(("run_trace.log is empty", not failed, failed[:200]))

    server_log = root / "dsagt_server.log"
    errors = (
        [
            line
            for line in server_log.read_text(errors="replace").splitlines()
            if re.search(r"Traceback|\bERROR\b", line)
        ]
        if server_log.exists()
        else []
    )
    results.append(
        ("server log has no error", not errors, errors[0][:200] if errors else "")
    )
    return results


def skills_installed(*names: str):
    def check(project: Project):
        missing = [
            n for n in names if not (project.root / "skills" / n / "SKILL.md").exists()
        ]
        unmirrored = [
            n
            for n in names
            if n not in missing
            and not any((project.root / d / n).exists() for d in NATIVE_SKILL_DIRS)
        ]
        detail = "; ".join(
            filter(
                None,
                [
                    f"missing: {missing}" if missing else "",
                    f"not mirrored: {unmirrored}" if unmirrored else "",
                ],
            )
        )
        return (
            f"skills installed and mirrored: {', '.join(names)}",
            not missing and not unmirrored,
            detail,
        )

    return check


def kb_collection(name: str):
    def check(project: Project):
        index = project.root / "kb_index"
        present = (index / name).is_dir()
        have = sorted(p.name for p in index.iterdir()) if index.is_dir() else []
        return (
            f"knowledge-base collection {name}",
            present,
            "" if present else f"have: {have}",
        )

    return check


def code_registered(pattern: str, at_least: int = 1):
    def check(project: Project):
        found = [n for n in project.codes if re.search(pattern, n)]
        return (
            f"registered code matching /{pattern}/ (at least {at_least})",
            len(found) >= at_least,
            f"found: {found}",
        )

    return check


def successful_records(pattern: str, at_least: int = 1, with_files: bool = True):
    def check(project: Project):
        good = [
            r
            for r in project.records_of(pattern)
            if r["execution"]["return_code"] == 0
            and (
                not with_files
                or r["execution"]["input_files"]
                or r["execution"]["output_files"]
            )
        ]
        return (
            f"successful records of /{pattern}/ naming files (at least {at_least})",
            len(good) >= at_least,
            f"{len(good)} of {len(project.records_of(pattern))}",
        )

    return check


def files_exist(glob: str, at_least: int = 1):
    def check(project: Project):
        found = project.files(glob)
        return (
            f"{glob} (at least {at_least})",
            len(found) >= at_least,
            f"{len(found)} found",
        )

    return check


def pipeline_script_saved(project: Project):
    found = [
        p
        for name in PIPELINE_SCRIPT_NAMES
        for p in project.root.rglob(name)
        if ".claude" not in p.parts
    ]
    return (
        "a reconstructed pipeline script is saved",
        bool(found),
        ", ".join(str(p.relative_to(project.root)) for p in found),
    )


def output_has_record(path: str):
    def check(project: Project):
        producers = [
            r["code_name"]
            for r in project.records
            if r["execution"]["return_code"] == 0
            and path in r["execution"]["output_files"]
        ]
        return (
            f"{path} is the output of a successful record",
            bool(producers),
            f"by: {sorted(set(producers))}",
        )

    return check


# ---------------------------------------------------------------------------
# Outcome observations: each returns (name, value)
# ---------------------------------------------------------------------------


def datacard_validation(project: Project):
    runs = project.records_of(r"datacard-validate")
    if not runs:
        return ("datacard validator, last exit code", "never run")
    return (
        "datacard validator, last exit code",
        f"{runs[-1]['execution']['return_code']} after {len(runs)} runs",
    )


def count_of(label: str, glob: str):
    def observe(project: Project):
        return (label, str(len(project.files(glob))))

    return observe


def grep_counts(path: str, *values: str):
    def observe(project: Project):
        target = project.root / path
        if not target.exists():
            return (f"{path} holds the expected values", "file absent")
        text = target.read_text(errors="replace")
        present = [v for v in values if v in text]
        return (f"{path} holds the expected values", f"{len(present)} of {len(values)}")

    return observe


def json_fields_match(path: str, reference: str, *fields: str):
    def lookup(data, dotted):
        for key in dotted.split("."):
            data = data.get(key) if isinstance(data, dict) else None
        return data

    def observe(project: Project):
        produced, expected = project.root / path, project.root / reference
        if not produced.exists() or not expected.exists():
            return (f"{path} against {reference}", "file absent")
        a, b = json.loads(produced.read_text()), json.loads(expected.read_text())
        same = [
            f
            for f in fields
            if lookup(a, f) is not None and lookup(a, f) == lookup(b, f)
        ]
        differing = [f for f in fields if f not in same]
        return (
            f"{path} against {reference}",
            f"{len(same)} of {len(fields)} fields equal"
            + (f"; differ: {differing}" if differing else ""),
        )

    return observe


def codes_with_parameter(parameter: str):
    def observe(project: Project):
        having = [
            name
            for name, spec in project.codes.items()
            if parameter in (spec.get("parameters") or {})
        ]
        return (
            f"codes with a {parameter} parameter",
            f"{len(having)} of {len(project.codes)}",
        )

    return observe


def record_stdout_contains(pattern: str, text: str):
    def observe(project: Project):
        hits = [
            r
            for r in project.records_of(pattern)
            if text in r["execution"].get("stdout", "")
        ]
        return (f"a /{pattern}/ record printed {text!r}", "yes" if hits else "no")

    return observe


WALKTHROUGHS = {
    "aidrin-ai-readiness": {
        "mechanical": [
            files_exist("skills/aidrin/PROVENANCE.txt"),
            successful_records(r"^aidrin$", at_least=13),
        ],
        "outcome": [
            count_of(
                "datacard at data/genesis_datacard_adult.md",
                "data/genesis_datacard_adult.md",
            ),
            datacard_validation,
        ],
    },
    "genesis_skills": {
        "mechanical": [
            kb_collection("skills_catalog__ai-modcon-genesis-skills"),
            skills_installed("croissant-validator"),
            files_exist("skills/croissant-validator/PROVENANCE.txt"),
            successful_records(r"datacard-introspect"),
            successful_records(r"croissant.*validat"),
        ],
        "outcome": [
            grep_counts(
                "audit/catalyst_screening_datacard.md",
                "250 °C",
                "GHSV",
                "CC-BY-4.0",
                "Single-run",
                "C2+",
                "relative",
            ),
            datacard_validation,
            record_stdout_contains(r"croissant.*validat", "mlcroissant parse OK"),
        ],
    },
    "cryoem": {
        "mechanical": [
            kb_collection("cryoppp"),
            code_registered(
                r".", at_least=8
            ),  # the four base-skill codes plus at least four of the agent's
            files_exist("data/cryoem/particles.csv"),
            files_exist("data/cryoem/particles_curated.csv"),
            output_has_record("data/cryoem/particles_curated.csv"),
            pipeline_script_saved,
        ],
        "outcome": [
            count_of(
                "check reports in audit/ (four stage boundaries)", "audit/*aidrin*.json"
            ),
            count_of("datacards", "**/*datacard*.md"),
            datacard_validation,
        ],
    },
    "combustion_simulation": {
        "mechanical": [
            files_exist("skills/blastnet-to-well/SKILL.md"),
            files_exist("skills/blastnet-to-well/references/*", at_least=2),
            files_exist("skills/blastnet-to-well/scripts/*.py"),
            code_registered(r"check-well-output"),
            code_registered(r"blastnet-to-well|convert-to-well"),
            files_exist("well_output/lifted_hydrogen_jet_traj_5000.hdf5"),
            successful_records(r"check-well-output"),
            pipeline_script_saved,
        ],
        "outcome": [
            datacard_validation,
            count_of(
                "failed checker runs on record",
                "trace_archive/check-well-output_*.json",
            ),
        ],
    },
    "vasp_dft": {
        "mechanical": [
            kb_collection("skills_catalog__k-dense-ai-scientific-agent-skills"),
            skills_installed("pymatgen", "vasp-to-isaac"),
            code_registered(r"vasp-neb-to-isaac"),
            output_has_record("audit/mock_slab_isaac.json"),
            files_exist("data/neb_record.json"),
            pipeline_script_saved,
        ],
        "outcome": [
            json_fields_match(
                "audit/mock_slab_isaac.json",
                "data/expected_isaac_record.json",
                "results.total_energy_eV",
                "results.energy_sigma0_eV",
                "computation.relaxation.ionic_steps",
                "system.configuration.code_version",
                "results.max_residual_force_eV_per_A",
            ),
        ],
    },
    "tokamak_stability": {
        "mechanical": [
            code_registered(
                r".", at_least=14
            ),  # the four base-skill codes plus the M3D-C1 functions
            files_exist("plots/*.png", at_least=3),
            files_exist("processed_data/pressure_spectrum.h5"),
            files_exist("processed_data/electrons.h5"),
            pipeline_script_saved,
        ],
        "outcome": [codes_with_parameter("output_json")],
    },
    "microbial_isolates": {
        "mechanical": [
            code_registered(r"^fastp"),
            code_registered(r"^megahit"),
            successful_records(r"^fastp"),
            successful_records(r"^megahit"),
            pipeline_script_saved,
        ],
        "outcome": [
            count_of("samples trimmed, of 11", "data/processed/*/"),
            count_of("samples assembled, of 11", "data/assemblies/*/final.contigs.fa"),
            datacard_validation,
        ],
    },
}


def check_project(walkthrough: str, root: Path) -> tuple[list, list]:
    project = Project.load(root)
    declared = WALKTHROUGHS[walkthrough]
    mechanical = common_checks(project) + [
        check(project) for check in declared["mechanical"]
    ]
    outcome = [observe(project) for observe in declared["outcome"]]
    return mechanical, outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "use_case_dir", help="the walkthrough directory, e.g. use_cases/vasp_dft"
    )
    parser.add_argument(
        "projects", nargs="+", help="project names under ~/dsagt-projects, or paths"
    )
    args = parser.parse_args()

    walkthrough = Path(args.use_case_dir).name
    if walkthrough not in WALKTHROUGHS:
        raise SystemExit(
            f"no checks declared for {walkthrough!r}; have {sorted(WALKTHROUGHS)}"
        )

    failures = 0
    outcomes: dict[str, list[str]] = {}
    for name in args.projects:
        root = (
            Path(name) if Path(name).is_dir() else Path.home() / "dsagt-projects" / name
        )
        mechanical, outcome = check_project(walkthrough, root)
        print(f"\n== {root.name}")
        for label, passed, detail in mechanical:
            failures += not passed
            print(
                f"  {'ok  ' if passed else 'FAIL'} {label}"
                + (f"  [{detail}]" if detail else "")
            )
        for label, value in outcome:
            outcomes.setdefault(label, []).append(value)
    print("\n== outcomes (values per run, not pass or fail)")
    for label, values in outcomes.items():
        print(f"  {label}: {' | '.join(values)}")
    print(f"\nmechanical failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
