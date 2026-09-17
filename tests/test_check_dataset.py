"""
Tests for the check-dataset built-in code (``src/dsagt/codes/check-dataset/``).

Each ``checks/*.py`` worker is loaded in-process via
``importlib.util.spec_from_file_location`` (matching
``test_csv_summary_fixture.py``'s pattern) and exercised through its
``run_check(...)`` function directly — these are unit tests of the check
logic, not of subprocess plumbing. A single end-to-end test drives the real
``check_dataset.py`` orchestrator through ``subprocess`` to pin the
argv/JSON contract between it and its workers.

Dataset/model fixtures (including the deliberately-buggy ones the issue's
acceptance criteria target) live in ``tests/fixtures_check_dataset.py`` so
they're importable by path, exactly like a real generated ``Dataset``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from dsagt.contract import save_contract
from dsagt.provenance import compute_pipeline_fingerprint, reconstruct_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKS_DIR = (
    REPO_ROOT / "src" / "dsagt" / "codes" / "check-dataset" / "scripts" / "checks"
)
ORCHESTRATOR_SCRIPT = (
    REPO_ROOT
    / "src"
    / "dsagt"
    / "codes"
    / "check-dataset"
    / "scripts"
    / "check_dataset.py"
)

MINIMAL_CONTRACT = {
    "version": 1,
    "mode": "standalone",
    "keys": {
        "features": {
            "dtype": "float32",
            "shape": [4],
            "role": "input",
            "value_range": [-1.0, 1.0],
            "collation": "stack",
        },
        "label": {
            "dtype": "int64",
            "shape": [],
            "role": "target",
            "collation": "stack",
        },
        "id": {
            "dtype": "string",
            "shape": [],
            "role": "metadata",
            "collation": "list",
        },
    },
    "normalization": {"owner": "dataset"},
    "split": {
        "strategy": "random",
        "seed": 0,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
    },
    "reconciliation": [],
}


def _load_check(name: str):
    spec = importlib.util.spec_from_file_location(
        f"check_dataset_{name}_check", CHECKS_DIR / f"{name}_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_contract(tmp_path: Path, contract: dict) -> Path:
    path = tmp_path / "dataset_contract.yaml"
    save_contract(path, contract)
    return path


def _write_record(
    trace_dir: Path,
    record_id: str,
    code_name: str,
    output_files: list,
    input_files: list | None = None,
    timestamp_start: str = "2024-01-01T00:00:00+00:00",
) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "code_name": code_name,
        "session_id": "s1",
        "execution": {
            "timestamp_start": timestamp_start,
            "exact_command": ["python", f"{code_name}.py"],
            "input_files": input_files or [],
            "output_files": output_files,
            "return_code": 0,
        },
    }
    (trace_dir / f"{record_id}.json").write_text(json.dumps(record))


# ---------------------------------------------------------------------------
# contract check
# ---------------------------------------------------------------------------


class TestContractCheck:
    def test_passing_dataset(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        mod = _load_check("contract")
        result = mod.run_check(
            {
                "contract": str(contract_path),
                "dataset": "tests.fixtures_check_dataset:GoodDataset",
            }
        )
        assert result["status"] == "passed", result
        assert result["detail"]["problems"] == {}

    def test_dtype_shape_and_range_mismatches_caught(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        mod = _load_check("contract")
        result = mod.run_check(
            {
                "contract": str(contract_path),
                "dataset": "tests.fixtures_check_dataset:BadContractDataset",
            }
        )
        assert result["status"] == "failed"
        problems = result["detail"]["problems"]
        assert "features" in problems
        assert any("dtype mismatch" in p for p in problems["features"])
        assert any("value_range violated" in p for p in problems["features"])
        assert "label" in problems
        assert any("shape" in p for p in problems["label"])


# ---------------------------------------------------------------------------
# determinism check
# ---------------------------------------------------------------------------


class TestDeterminismCheck:
    def test_deterministic_dataset_passes(self, tmp_path):
        mod = _load_check("determinism")
        result = mod.run_check({"dataset": "tests.fixtures_check_dataset:GoodDataset"})
        assert result["status"] == "passed"
        assert result["detail"]["samples_equal"] is True

    def test_nondeterministic_dataset_caught(self, tmp_path):
        mod = _load_check("determinism")
        result = mod.run_check(
            {"dataset": "tests.fixtures_check_dataset:NonDeterministicDataset"}
        )
        assert result["status"] == "failed"
        assert result["detail"]["samples_equal"] is False


# ---------------------------------------------------------------------------
# worker_equivalence check
# ---------------------------------------------------------------------------


def _run_worker_equivalence_subprocess(
    worker_args: dict, timeout_s: float = 60.0
) -> dict:
    """Run worker_equivalence_check.py as its own process, matching how
    check_dataset.py's orchestrator invokes it (this check spins up real
    ``DataLoader`` worker processes, so it must not share process state
    with other tests). Tolerates the child hanging *after* it has already
    printed its result: torch's ``spawn``-context DataLoader teardown can
    wedge on exit on some platforms even once iteration itself completed
    cleanly, so a bounded timeout plus the partial stdout it captures on
    kill is what a caller can rely on, not a clean return code.
    """
    script = CHECKS_DIR / "worker_equivalence_check.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, str(script), json.dumps(worker_args)],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
        )
        stdout = proc.stdout
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
    assert stdout.strip(), f"worker produced no output within {timeout_s}s"
    return json.loads(stdout)


class TestWorkerEquivalenceCheck:
    def test_map_style_dataset_is_skipped(self):
        mod = _load_check("worker_equivalence")
        result = mod.run_check({"dataset": "tests.fixtures_check_dataset:GoodDataset"})
        assert result["status"] == "skipped"

    def test_correctly_sharded_iterable_dataset_passes(self):
        result = _run_worker_equivalence_subprocess(
            {
                "dataset": "tests.fixtures_check_dataset:GoodIterableDataset",
                "dataset_args": {"n": 12},
                "id_key": "id",
            }
        )
        assert result["status"] == "passed", result

    def test_unsharded_iterable_dataset_duplication_caught(self):
        """Acceptance criterion: detects a deliberately introduced iterable-worker
        duplication bug (an IterableDataset that ignores get_worker_info())."""
        result = _run_worker_equivalence_subprocess(
            {
                "dataset": "tests.fixtures_check_dataset:BuggyIterableDataset",
                "dataset_args": {"n": 12},
                "id_key": "id",
            }
        )
        assert result["status"] == "failed"
        assert result["detail"]["duplicated_in_parallel"]


# ---------------------------------------------------------------------------
# split_leakage check
# ---------------------------------------------------------------------------


class TestSplitLeakageCheck:
    def _manifest(self, tmp_path, data: dict) -> str:
        path = tmp_path / "split_manifest.json"
        path.write_text(json.dumps(data))
        return str(path)

    def test_no_manifest_is_skipped(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        mod = _load_check("split_leakage")
        result = mod.run_check({"contract": str(contract_path)})
        assert result["status"] == "skipped"

    def test_clean_manifest_passes(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        manifest = {
            "train": [f"g{i}" for i in range(16)],
            "val": [f"g{i}" for i in range(16, 18)],
            "test": [f"g{i}" for i in range(18, 20)],
        }
        manifest_path = self._manifest(tmp_path, manifest)
        mod = _load_check("split_leakage")
        result = mod.run_check(
            {"contract": str(contract_path), "split_manifest": manifest_path}
        )
        assert result["status"] == "passed", result
        assert result["detail"]["leaked_ids"] == {}

    def test_group_repeated_across_splits_caught(self, tmp_path):
        """Acceptance criterion: detects a deliberately introduced group leak
        across splits."""
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        manifest = {
            "train": [f"g{i}" for i in range(16)] + ["g19"],
            "val": [f"g{i}" for i in range(16, 18)],
            "test": [f"g{i}" for i in range(18, 20)],
        }
        manifest_path = self._manifest(tmp_path, manifest)
        mod = _load_check("split_leakage")
        result = mod.run_check(
            {"contract": str(contract_path), "split_manifest": manifest_path}
        )
        assert result["status"] == "failed"
        assert "g19" in result["detail"]["leaked_ids"]

    def test_skewed_ratios_caught(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        manifest = {
            "train": [f"g{i}" for i in range(5)],
            "val": [f"g{i}" for i in range(5, 15)],
            "test": [f"g{i}" for i in range(15, 20)],
        }
        manifest_path = self._manifest(tmp_path, manifest)
        mod = _load_check("split_leakage")
        result = mod.run_check(
            {"contract": str(contract_path), "split_manifest": manifest_path}
        )
        assert result["status"] == "failed"
        assert "train" in result["detail"]["size_problems"]
        assert "val" in result["detail"]["size_problems"]


# ---------------------------------------------------------------------------
# throughput check
# ---------------------------------------------------------------------------


class TestThroughputCheck:
    def test_fast_dataset_passes(self):
        mod = _load_check("throughput")
        result = mod.run_check(
            {
                "dataset": "tests.fixtures_check_dataset:GoodDataset",
                "slow_threshold_s": 1.0,
            }
        )
        assert result["status"] == "passed"

    def test_slow_dataset_caught(self):
        mod = _load_check("throughput")
        result = mod.run_check(
            {
                "dataset": "tests.fixtures_check_dataset:SlowDataset",
                "n_samples": 5,
                "slow_threshold_s": 0.001,
            }
        )
        assert result["status"] == "failed"
        assert result["detail"]["slow_samples"]


# ---------------------------------------------------------------------------
# model_forward check
# ---------------------------------------------------------------------------


class TestModelForwardCheck:
    def test_no_model_is_skipped(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        mod = _load_check("model_forward")
        result = mod.run_check(
            {
                "contract": str(contract_path),
                "dataset": "tests.fixtures_check_dataset:GoodDataset",
            }
        )
        assert result["status"] == "skipped"

    def test_compatible_model_passes(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        mod = _load_check("model_forward")
        result = mod.run_check(
            {
                "contract": str(contract_path),
                "dataset": "tests.fixtures_check_dataset:GoodDataset",
                "model": "tests.fixtures_check_dataset:GoodModel",
            }
        )
        assert result["status"] == "passed", result

    def test_incompatible_model_reported_as_failed(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        mod = _load_check("model_forward")
        result = mod.run_check(
            {
                "contract": str(contract_path),
                "dataset": "tests.fixtures_check_dataset:GoodDataset",
                "model": "tests.fixtures_check_dataset:BadModel",
            }
        )
        assert result["status"] == "failed"
        assert "error" in result["detail"]


# ---------------------------------------------------------------------------
# staleness check
# ---------------------------------------------------------------------------


class TestStalenessCheck:
    def test_standalone_contract_is_skipped(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        mod = _load_check("staleness")
        result = mod.run_check(
            {"contract": str(contract_path), "project_dir": str(tmp_path)}
        )
        assert result["status"] == "skipped"

    def test_matching_fingerprint_passes(self, tmp_path):
        trace_dir = tmp_path / "trace_archive"
        _write_record(trace_dir, "r1", "load_csv", output_files=["raw.csv"])
        structured = json.loads(reconstruct_pipeline(trace_dir, fmt="json"))
        fingerprint = compute_pipeline_fingerprint(structured)

        pipeline_contract = copy.deepcopy(MINIMAL_CONTRACT)
        pipeline_contract["mode"] = "pipeline"
        pipeline_contract["pipeline_fingerprint"] = fingerprint
        contract_path = _write_contract(tmp_path, pipeline_contract)

        mod = _load_check("staleness")
        result = mod.run_check(
            {"contract": str(contract_path), "project_dir": str(tmp_path)}
        )
        assert result["status"] == "passed", result

    def test_upstream_pipeline_change_caught(self, tmp_path):
        """Acceptance criterion: detects an upstream pipeline change via
        fingerprint mismatch."""
        trace_dir = tmp_path / "trace_archive"
        _write_record(trace_dir, "r1", "load_csv", output_files=["raw.csv"])
        structured = json.loads(reconstruct_pipeline(trace_dir, fmt="json"))
        fingerprint = compute_pipeline_fingerprint(structured)

        pipeline_contract = copy.deepcopy(MINIMAL_CONTRACT)
        pipeline_contract["mode"] = "pipeline"
        pipeline_contract["pipeline_fingerprint"] = fingerprint
        contract_path = _write_contract(tmp_path, pipeline_contract)

        # Upstream pipeline gains a new step after the contract was written.
        _write_record(
            trace_dir,
            "r2",
            "normalize",
            output_files=["normalized.csv"],
            input_files=["raw.csv"],
            timestamp_start="2024-01-01T00:01:00+00:00",
        )

        mod = _load_check("staleness")
        result = mod.run_check(
            {"contract": str(contract_path), "project_dir": str(tmp_path)}
        )
        assert result["status"] == "failed"
        assert result["detail"]["current_fingerprint"] != fingerprint


# ---------------------------------------------------------------------------
# orchestrator end-to-end (real subprocess, pins the argv/JSON worker contract)
# ---------------------------------------------------------------------------


class TestOrchestratorEndToEnd:
    def test_runs_selected_checks_and_writes_report(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        output_path = tmp_path / "audit" / "report.json"

        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            [
                sys.executable,
                str(ORCHESTRATOR_SCRIPT),
                "--contract",
                str(contract_path),
                "--dataset",
                "tests.fixtures_check_dataset:GoodDataset",
                "--checks",
                "contract,determinism",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr

        report = json.loads(output_path.read_text())
        assert report["overall_passed"] is True
        assert set(report["checks"]) == {"contract", "determinism"}
        assert report["checks"]["contract"]["status"] == "passed"
        assert report["checks"]["determinism"]["status"] == "passed"

    def test_unknown_check_name_exits_with_error(self, tmp_path):
        contract_path = _write_contract(tmp_path, MINIMAL_CONTRACT)
        proc = subprocess.run(
            [
                sys.executable,
                str(ORCHESTRATOR_SCRIPT),
                "--contract",
                str(contract_path),
                "--dataset",
                "tests.fixtures_check_dataset:GoodDataset",
                "--checks",
                "bogus_check",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 2
        assert "Unknown check" in proc.stderr
