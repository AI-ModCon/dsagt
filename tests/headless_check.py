"""Check a finished headless walkthrough run: mechanical checks, then outcome observations.

A headless run (``tests/headless_usecases.py``) leaves a project directory.
This script reads it and reports two things that are scored apart.
Mechanical checks are the walkthrough's declared post-conditions: dsagt's own
guarantees (the skills and codes are installed, every record is complete and
has a ``code.execute`` trace, the logs have no error) and the artifacts the
README's Post-Conditions name.  A failure means the run did not meet the
README, and the exit code is 1; the detail says which check and what it found,
which is what tells a dsagt regression from a run that went another way.  A
check this run's configuration puts out of reach, the trace count when the
store is a shared tracking server, is reported as not checked and fails
nothing.  Outcome observations are what the README leaves to the agent (a
value in a converted file, whether a validator ran through ``dsagt-run``, how
many samples were assembled); they are printed as values, never as pass or
fail, and with several projects as a rate, because one run moves by a
post-condition or two with the same inputs.

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
import os.path
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

# dsagt's own parser and its own native-mirror rule, so what the registry
# reads and what the mirror decides are what the checker reads and decides.
from dsagt.agents.base import _description_fits_native_cap
from dsagt.observability import resolve_tracking_uri
from dsagt.registry import _parse_frontmatter

#: This repository, which holds each walkthrough's reference fixtures.
REPO_ROOT = Path(__file__).resolve().parent.parent

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

    def relative(self, path: str) -> str:
        """*path* as it sits under the project root.

        A record holds the path the agent typed, so one file appears as
        ``data/x.csv`` in one record, ``./data/x.csv`` in the next and as an
        absolute path in a third; a comparison against a declared path has to
        bring them to one spelling first.  A path outside the project is
        returned as given.
        """
        if os.path.isabs(path):
            try:
                return str(Path(path).resolve().relative_to(self.root.resolve()))
            except ValueError:
                return path
        return os.path.normpath(path)

    def outputs_of(self, record: dict) -> list[str]:
        return [self.relative(f) for f in record["execution"].get("output_files", [])]

    def inputs_of(self, record: dict) -> list[str]:
        return [self.relative(f) for f in record["execution"].get("input_files", [])]

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
# Mechanical checks: each returns (name, passed, detail); passed is None for
# a check this run's configuration puts out of reach.
# ---------------------------------------------------------------------------


def common_checks(project: Project) -> list[tuple[str, bool | None, str]]:
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
    # A mirror entry with no project source is user-authored; the manifest
    # leaves those alone, so they are not dsagt's to link.
    copied = [
        entry
        for mirror in mirrors
        for entry in mirror.iterdir()
        if entry.is_dir()
        and not entry.is_symlink()
        and (root / "skills" / entry.name / "SKILL.md").exists()
    ]
    # dsagt copies rather than links a skill whose description exceeds the
    # native cap, because a link cannot be trimmed, and the walkthroughs
    # install catalog skills whose descriptions this repository does not
    # control.  That copy is the documented behavior, not a failure.
    over_cap = sorted(
        e.name
        for e in copied
        if not _description_fits_native_cap(root / "skills" / e.name / "SKILL.md")
    )
    unlinked = sorted(
        str(e.relative_to(root)) for e in copied if e.name not in over_cap
    )
    results.append(
        (
            "native skills mirror links every skill under the description cap",
            bool(mirrors) and not unlinked,
            (
                f"copies: {unlinked}"
                if unlinked
                else (f"copied for the description cap: {over_cap}" if over_cap else "")
            ),
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
        if r["execution"].get("return_code") == 0
        for f in r["execution"].get("output_files", [])
        if (root / f).is_file() and f not in r["execution"].get("file_hashes", {})
    ]
    results.append(
        (
            "every written output has a hash",
            not unhashed,
            f"unhashed: {unhashed[:5]}" if unhashed else "",
        )
    )

    # Traces go to the project's sqlite file only when MLFLOW_TRACKING_URI
    # names no shared server; against a server the checker has no store to
    # count, which is a different thing from a run that logged nothing.
    store = resolve_tracking_uri({"project_dir": str(root)})
    if store != f"sqlite:///{root.resolve() / 'mlflow.db'}":
        results.append(("one code.execute trace per record", None, f"store is {store}"))
    else:
        traced = project.query(
            "mlflow.db",
            "select count(*) from trace_tags "
            "where key = 'dsagt.source' and value = 'execution'",
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
            if r["execution"].get("return_code") == 0
            and (
                not with_files
                or r["execution"].get("input_files")
                or r["execution"].get("output_files")
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
            if r["execution"].get("return_code") == 0 and path in project.outputs_of(r)
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
    """The datacard validator's last exit code, over the runs that left a record.

    The label names the record because that is what is measured: a validator
    the agent ran outside ``dsagt-run`` validated the card and left nothing to
    read here.
    """
    label = "datacard-validate runs on record, last exit code"
    runs = project.records_of(r"datacard-validate")
    if not runs:
        return (label, "no record")
    return (label, f"{runs[-1]['execution'].get('return_code')} after {len(runs)} runs")


def successful_record_count(label: str, pattern: str):
    """How many runs of /pattern/ left a record that exited zero."""

    def observe(project: Project):
        runs = project.records_of(pattern)
        good = [r for r in runs if r["execution"].get("return_code") == 0]
        return (label, f"{len(good)} of {len(runs)}")

    return observe


def failed_record_count(label: str, pattern: str):
    """How many runs of /pattern/ left a record that exited non-zero."""

    def observe(project: Project):
        runs = project.records_of(pattern)
        bad = [r for r in runs if r["execution"].get("return_code") != 0]
        return (label, f"{len(bad)} of {len(runs)}")

    return observe


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
    """Compare a produced file against a fixture in this repository.

    *reference* is repository-relative.  The project's copy of the same
    fixture comes from the walkthrough's data bundle, which is published
    separately and can lag a correction made here; the repository's copy is
    the one review sees.
    """

    def lookup(data, dotted):
        for key in dotted.split("."):
            data = data.get(key) if isinstance(data, dict) else None
        return data

    def observe(project: Project):
        produced, expected = project.root / path, REPO_ROOT / reference
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


def tables_with_a_readiness_record(project: Project):
    """How many of the tables the pipeline wrote have an aidrin record that read them."""
    written = {
        f
        for r in project.records
        if r["code_name"] != "aidrin" and r["execution"].get("return_code") == 0
        for f in project.outputs_of(r)
        if f.lower().endswith((".csv", ".tsv", ".parquet", ".xlsx", ".xls"))
    }
    checked = {
        f
        for r in project.records_of(r"^aidrin$")
        if r["execution"].get("return_code") == 0
        for f in project.inputs_of(r)
    }
    return (
        "tables the pipeline wrote that have a readiness record",
        f"{len(written & checked)} of {len(written)}",
    )


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
            if text in (r["execution"].get("stdout") or "")
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
        ],
        "outcome": [
            successful_record_count(
                "croissant validation runs on record", r"croissant.*validat"
            ),
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
            tables_with_a_readiness_record,
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
            failed_record_count("failed checker runs on record", r"check-well-output"),
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
                "use_cases/vasp_dft/data/expected_isaac_record.json",
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
            failures += passed is False
            mark = {True: "ok  ", False: "FAIL", None: "--  "}[passed]
            print(f"  {mark} {label}" + (f"  [{detail}]" if detail else ""))
        for label, value in outcome:
            outcomes.setdefault(label, []).append(value)
    print("\n== outcomes (values per run, not pass or fail)")
    for label, values in outcomes.items():
        print(f"  {label}: {' | '.join(values)}")
    print(f"\nmechanical failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
