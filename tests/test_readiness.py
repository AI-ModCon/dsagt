"""AI-readiness check: config block, release tag, and the instructions paragraph."""

import re
from importlib.metadata import version

import pytest

from dsagt import readiness as rd
from dsagt.agents.base import _load_master_instructions


class TestConfigBlock:

    def test_block_holds_the_one_setting(self):
        assert rd.readiness_block(True) == {"auto_assess": True}
        assert rd.readiness_block(False) == {"auto_assess": False}

    def test_enabled_reads_the_block_and_defaults_on(self):
        assert rd.auto_assess_enabled({}) is True
        assert rd.auto_assess_enabled({"readiness": {"auto_assess": False}}) is False
        assert rd.auto_assess_enabled({"readiness": {"auto_assess": True}}) is True


class TestReleaseTag:

    def test_zero_pads_the_month(self):
        # importlib.metadata reports the normalized version; the tag keeps
        # the two-digit month AIDRIN releases under.
        assert rd.aidrin_release_tag("2026.8.2") == "v2026.08.2"
        assert rd.aidrin_release_tag("2026.08.2") == "v2026.08.2"
        assert rd.aidrin_release_tag("2026.11.1") == "v2026.11.1"

    def test_rejects_other_shapes(self):
        for bad in ("2026.8", "2026.8.2.dev1", "v2026.08.2", ""):
            with pytest.raises(ValueError, match="year"):
                rd.aidrin_release_tag(bad)

    def test_installed_package_maps_to_a_tag(self):
        tag = rd.aidrin_release_tag(version("aidrin"))
        assert re.fullmatch(r"v\d{4}\.\d{2}\.\d+", tag), tag


class TestInstructionsParagraph:

    def test_paragraph_sits_at_the_check_rule_when_on(self):
        text = _load_master_instructions(True)
        check_rule = text.index("### 4. Per-Operation Checks")
        paragraph = text.index("#### AI-readiness check")
        next_rule = text.index("### 5. File Organization")
        assert check_rule < paragraph < next_rule
        assert "quality baseline" in text
        assert "audit/step_N_pre.aidrin.json" in text
        # Through the registered code, never the bare binary.
        assert "registered `aidrin`" in text
        assert "<!--" not in text

    def test_paragraph_absent_when_off(self):
        text = _load_master_instructions(False)
        assert "AI-readiness check" not in text
        assert "<!--" not in text
        assert "### 4. Per-Operation Checks" in text


def test_docs_page_quotes_the_paragraph_verbatim():
    """docs/readiness.md shows the inserted paragraph as a quote block; the
    page drifts from the source unless a test holds them equal."""
    from pathlib import Path

    from dsagt.readiness import INSTRUCTIONS_PARAGRAPH

    page = Path(__file__).resolve().parents[1] / "docs" / "readiness.md"
    text = page.read_text()
    start = text.index("> #### AI-readiness check")
    end = text.index("\n## ", start)
    quoted = "\n".join(
        line[2:] if line.startswith("> ") else line[1:]
        for line in text[start:end].rstrip("\n").split("\n")
    )
    assert quoted == INSTRUCTIONS_PARAGRAPH.rstrip("\n")


def _dsagt_run_invocations(text: str) -> list[list[str]]:
    """The wrapper arguments of every ``dsagt-run`` command a code span names.

    Only code spans, because prose that mentions the wrapper has no separator
    to stop at.  Whitespace is flattened so a command wrapped across lines
    reads as one.
    """
    found = []
    for span in re.findall(r"`([^`]*)`", text):
        flat = " ".join(span.split())
        start = flat.find("dsagt-run ")
        if start == -1:
            continue
        head = flat[start:].split(" -- ", 1)[0]
        found.append(head.split()[1:])
    return found


def test_every_dsagt_run_the_shipped_text_names_parses():
    """A command dsagt tells an agent to run is one dsagt-run accepts.

    The paragraph and its docs page are compared to each other elsewhere, and
    they moved together when an option was removed and again when a merge
    brought the option back, so only the parser can settle whether the text
    names something that exists.
    """
    import dsagt
    from pathlib import Path

    from dsagt.commands.run_code import _parse_args

    docs = Path(__file__).resolve().parents[1] / "docs"
    package = Path(dsagt.__file__).resolve().parent
    sources = {
        "INSTRUCTIONS_PARAGRAPH": rd.INSTRUCTIONS_PARAGRAPH,
        "dsagt_instructions.md": (package / "dsagt_instructions.md").read_text(),
        "docs/readiness.md": (docs / "readiness.md").read_text(),
        "docs/provenance.md": (docs / "provenance.md").read_text(),
    }
    named = 0
    for name, text in sources.items():
        for wrapper_args in _dsagt_run_invocations(text):
            named += 1
            try:
                _parse_args(wrapper_args + ["--", "true"])
            except SystemExit:
                pytest.fail(f"{name} names dsagt-run {' '.join(wrapper_args)}")
    assert named, "no dsagt-run command found; the scan stopped matching"
