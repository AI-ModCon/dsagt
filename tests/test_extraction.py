"""
Tests for memory-extraction helpers.

Episodic extraction itself (``extract_session``) is a no-op stub post-proxy
— the proxy-shape trace reader and the LiteLLM extraction call were removed
with the proxy, and Phase 3 rebuilds extraction over the CanonicalTrace
pipeline.  The prompt builder and response parser are retained as Phase-3
building blocks and still have unit coverage here.
"""

import json

from dsagt.memory import (
    build_extraction_prompt,
    extract_session,
    parse_extraction_response,
)

# ---------------------------------------------------------------------------
# build_extraction_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:

    def _make_exchanges(self):
        return [
            {
                "timestamp": "2024-01-15T10:30:00Z",
                "model": "claude-sonnet-4-20250514",
                "new_messages": [
                    {"role": "user", "content": "Run fastp on sample1.fq.gz"}
                ],
                "response": [
                    {"type": "text", "text": "I'll run fastp with Q20 filtering."},
                    {
                        "type": "tool_use",
                        "name": "bash",
                        "input": {"cmd": "fastp -q 20 --in1 sample1.fq.gz"},
                    },
                ],
            },
            {
                "timestamp": "2024-01-15T10:31:00Z",
                "model": "claude-sonnet-4-20250514",
                "new_messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "98% reads passed",
                            },
                        ],
                    },
                ],
                "response": [
                    {
                        "type": "text",
                        "text": "Filtering complete. 98% of reads passed Q20.",
                    },
                ],
            },
        ]

    def test_includes_conversation(self):
        prompt = build_extraction_prompt(self._make_exchanges())
        assert "fastp" in prompt
        assert "sample1.fq.gz" in prompt
        assert "98% reads passed" in prompt

    def test_includes_categories(self):
        prompt = build_extraction_prompt(self._make_exchanges())
        assert "quality_control" in prompt
        assert "data_management" in prompt

    def test_custom_categories_merged(self):
        custom = {"my_category": "custom stuff"}
        prompt = build_extraction_prompt(self._make_exchanges(), categories=custom)
        assert "my_category" in prompt
        assert "custom stuff" in prompt
        assert "quality_control" in prompt  # stock still present

    def test_output_format_described(self):
        prompt = build_extraction_prompt(self._make_exchanges())
        assert '"facts"' in prompt
        assert '"summary"' in prompt
        assert '"insights"' in prompt

    def test_empty_exchanges(self):
        prompt = build_extraction_prompt([])
        assert "facts" in prompt  # prompt structure still present


# ---------------------------------------------------------------------------
# parse_extraction_response
# ---------------------------------------------------------------------------


class TestParseExtractionResponse:

    def test_valid_json(self):
        response = json.dumps(
            {
                "facts": [{"text": "fastp used Q20", "category": "quality_control"}],
                "summary": "Ran fastp on sample1.",
                "insights": [
                    {
                        "text": "Q20 is sufficient for isolates",
                        "category": "quality_control",
                    }
                ],
            }
        )
        result = parse_extraction_response(response)
        assert len(result["facts"]) == 1
        assert result["summary"] == "Ran fastp on sample1."
        assert len(result["insights"]) == 1

    def test_strips_markdown_fences(self):
        response = (
            "```json\n"
            + json.dumps(
                {
                    "facts": [],
                    "summary": "test",
                    "insights": [],
                }
            )
            + "\n```"
        )
        result = parse_extraction_response(response)
        assert result["summary"] == "test"

    def test_missing_fields_default(self):
        result = parse_extraction_response("{}")
        assert result["facts"] == []
        assert result["summary"] == ""
        assert result["insights"] == []


# ---------------------------------------------------------------------------
# extract_session — Phase-3 stub
# ---------------------------------------------------------------------------


class TestExtractSessionStub:
    """Until Phase 3 rebuilds extraction, ``extract_session`` is a no-op
    that reports unavailability without reading traces or calling an LLM."""

    def test_returns_unavailable_without_touching_kb_or_network(self):
        result = extract_session(
            project_name="proj",
            kb=None,
            api_key="test-key",
            session_id="test-session",
        )
        assert result["status"] == "extraction_unavailable"
        assert result["facts"] == 0
        assert result["insights"] == 0
        assert result["total_entries"] == 0
        assert result["session_id"] == "test-session"

    def test_stub_ignores_injected_exchanges(self):
        result = extract_session(
            project_name="proj",
            kb=None,
            api_key="test-key",
            session_id="s",
            exchanges=[{"new_messages": [], "response": []}],
        )
        assert result["status"] == "extraction_unavailable"
        assert result["total_entries"] == 0
