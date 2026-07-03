"""Guard the quickstart's bundled csv_summary fixture.

The README/docs quickstart has the agent register and run
``tests/smoke_test/csv_summary.py`` on ``samples.csv``; step 4's
null-column finding is the fact the explicit-memory step (5-6) stores and
recalls.  This pins that contract so the fixture can't silently drift out
from under the docs.
"""

import importlib.util
from pathlib import Path

_FIXTURE = Path(__file__).parent / "smoke_test" / "csv_summary.py"
_SAMPLES = Path(__file__).parent / "smoke_test" / "data" / "samples.csv"


def _load():
    spec = importlib.util.spec_from_file_location("csv_summary_fixture", _FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_reports_columns_and_row_count():
    result = _load().summarize(str(_SAMPLES))
    assert result["columns"] == ["id", "name", "status", "score", "timestamp"]
    assert result["row_count"] == 8


def test_summary_surfaces_the_null_columns_the_memory_step_stores():
    result = _load().summarize(str(_SAMPLES))
    # The quickstart's step 5 remembers "null values in the status and
    # timestamp columns" — it must come from here, not thin air.
    assert result["columns_with_nulls"] == ["status", "timestamp"]
    assert result["null_counts"]["status"] == 1
    assert result["null_counts"]["timestamp"] == 1


def test_summary_computes_numeric_stats_only_for_numeric_columns():
    result = _load().summarize(str(_SAMPLES))
    assert set(result["numeric_stats"]) == {"id", "score"}
    assert result["numeric_stats"]["score"]["max"] == 92.1
    # timestamp is date-like, not numeric — must not appear.
    assert "timestamp" not in result["numeric_stats"]
