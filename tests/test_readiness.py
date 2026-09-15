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
