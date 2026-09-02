"""
Tests for the sample-contract schema (``dsagt.contract``).

Includes the two worked examples from the schema design: a tabular case
(standalone mode) and the XGC graph case
(``use_cases/fusion-fm/skills/xgc-ai-training/scripts/xgc_dataset.py``,
pipeline mode), confirming the schema expresses both.
"""

import copy

import pytest

from dsagt.contract import load_contract, save_contract, validate_contract

# ---------------------------------------------------------------------------
# Worked example: tabular case (standalone mode)
# ---------------------------------------------------------------------------

TABULAR_CONTRACT = {
    "version": 1,
    "mode": "standalone",
    "keys": {
        "features": {
            "dtype": "float32",
            "shape": [37],
            "role": "input",
            "value_range": [-3.0, 3.0],
            "collation": "stack",
        },
        "label": {
            "dtype": "int64",
            "shape": [],
            "role": "target",
            "collation": "stack",
        },
        "patient_id": {
            "dtype": "string",
            "shape": [],
            "role": "metadata",
            "collation": "list",
        },
    },
    "normalization": {"owner": "dataset"},
    "split": {
        "strategy": "group",
        "group_key": "patient_id",
        "seed": 42,
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    },
    "reconciliation": [
        {
            "key": "features",
            "producer": "float64 columns in the source CSV, unnormalized",
            "consumer": "float32 tensor, zero mean / unit variance expected by model.forward()",
            "resolution": "cast to float32 and standardize in __getitem__ using stats "
            "computed over the train split",
        },
        {
            "key": "label",
            "producer": "string category name",
            "consumer": "integer class index expected by the loss function",
            "resolution": "label encoding fit at contract-build time; mapping stored "
            "alongside the split manifest",
        },
    ],
}

# ---------------------------------------------------------------------------
# Worked example: XGC graph case (pipeline mode)
# ---------------------------------------------------------------------------
#
# Mirrors XGCGraphDataset.__getitem__ in
# use_cases/fusion-fm/skills/xgc-ai-training/scripts/xgc_dataset.py: a PyG
# Data graph (x, y, pos, edge_index, edge_attr) plus scalar metadata
# (leadtime, phi, step0, target_step), split by phi-plane group so every
# step from one plane stays in one split.

XGC_CONTRACT = {
    "version": 1,
    "mode": "pipeline",
    "pipeline_fingerprint": "sha256:6f1ea1e0c1a1c9e5c9b9f2e8f7d4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6",
    "keys": {
        "x": {
            "dtype": "float32",
            "shape": ["N", "n_steps", "7+F"],
            "role": "input",
            "collation": "graph_batch",
        },
        "pos": {
            "dtype": "float32",
            "shape": ["N", 2],
            "role": "input",
            "collation": "graph_batch",
        },
        "edge_index": {
            "dtype": "int64",
            "shape": [2, "E"],
            "role": "input",
            "collation": "graph_batch",
        },
        "edge_attr": {
            "dtype": "float32",
            "shape": ["E", 3],
            "role": "input",
            "collation": "graph_batch",
        },
        "leadtime": {
            "dtype": "float32",
            "shape": [1, 1],
            "role": "input",
            "collation": "stack",
        },
        "y": {
            "dtype": "float32",
            "shape": ["N", "F"],
            "role": "target",
            "collation": "graph_batch",
        },
        "phi": {
            "dtype": "int64",
            "shape": [],
            "role": "metadata",
            "collation": "list",
        },
        "step0": {
            "dtype": "int64",
            "shape": [],
            "role": "metadata",
            "collation": "list",
        },
        "target_step": {
            "dtype": "int64",
            "shape": [],
            "role": "metadata",
            "collation": "list",
        },
    },
    "normalization": {"owner": "pipeline"},
    "split": {
        "strategy": "group",
        "group_key": "phi",
        "seed": 7,
        "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
    },
    "reconciliation": [
        {
            "key": "x",
            "producer": "per-field npz arrays, one file per simulation step",
            "consumer": "single [N, n_steps, 7+F] tensor stacking static node "
            "context and per-step field values",
            "resolution": "__getitem__ loads the requested step_*.npz files and "
            "concatenates static_ctx with the field slice",
        },
        {
            "key": "edge_index",
            "producer": "triangular mesh connectivity in mesh.npz",
            "consumer": "undirected PyG edge_index expected by the message-passing layers",
            "resolution": "cells_to_edge_index(undirected=True), cached once as topology.pt",
        },
    ],
}


# ---------------------------------------------------------------------------
# Worked examples validate
# ---------------------------------------------------------------------------


class TestWorkedExamples:

    def test_tabular_contract_is_valid(self):
        validate_contract(TABULAR_CONTRACT)

    def test_xgc_contract_is_valid(self):
        validate_contract(XGC_CONTRACT)

    def test_tabular_and_xgc_round_trip_through_disk(self, tmp_path):
        for name, contract in (("tabular", TABULAR_CONTRACT), ("xgc", XGC_CONTRACT)):
            path = tmp_path / f"{name}_dataset_contract.yaml"
            save_contract(path, contract)
            assert load_contract(path) == contract


# ---------------------------------------------------------------------------
# Schema validation failures
# ---------------------------------------------------------------------------


class TestValidateContract:

    def _tabular(self) -> dict:
        return copy.deepcopy(TABULAR_CONTRACT)

    def test_missing_version(self):
        contract = self._tabular()
        del contract["version"]
        with pytest.raises(ValueError, match="version"):
            validate_contract(contract)

    def test_invalid_mode(self):
        contract = self._tabular()
        contract["mode"] = "bogus"
        with pytest.raises(ValueError, match="mode"):
            validate_contract(contract)

    def test_pipeline_mode_requires_fingerprint(self):
        contract = self._tabular()
        contract["mode"] = "pipeline"
        with pytest.raises(ValueError, match="pipeline_fingerprint"):
            validate_contract(contract)

    def test_standalone_mode_forbids_fingerprint(self):
        contract = copy.deepcopy(XGC_CONTRACT)
        contract["mode"] = "standalone"
        with pytest.raises(ValueError, match="pipeline_fingerprint"):
            validate_contract(contract)

    def test_empty_keys_rejected(self):
        contract = self._tabular()
        contract["keys"] = {}
        with pytest.raises(ValueError, match="keys"):
            validate_contract(contract)

    def test_key_missing_dtype(self):
        contract = self._tabular()
        del contract["keys"]["label"]["dtype"]
        with pytest.raises(ValueError, match="dtype"):
            validate_contract(contract)

    def test_key_invalid_role(self):
        contract = self._tabular()
        contract["keys"]["label"]["role"] = "bogus"
        with pytest.raises(ValueError, match="role"):
            validate_contract(contract)

    def test_key_shape_must_be_int_or_symbolic_name(self):
        contract = self._tabular()
        contract["keys"]["label"]["shape"] = [3.5]
        with pytest.raises(ValueError, match="shape"):
            validate_contract(contract)

    def test_key_symbolic_shape_accepted(self):
        contract = self._tabular()
        contract["keys"]["label"]["shape"] = ["N"]
        validate_contract(contract)

    def test_invalid_normalization_owner(self):
        contract = self._tabular()
        contract["normalization"]["owner"] = "bogus"
        with pytest.raises(ValueError, match="normalization"):
            validate_contract(contract)

    def test_invalid_split_strategy(self):
        contract = self._tabular()
        contract["split"]["strategy"] = "bogus"
        with pytest.raises(ValueError, match="strategy"):
            validate_contract(contract)

    def test_group_strategy_requires_group_key(self):
        contract = self._tabular()
        del contract["split"]["group_key"]
        with pytest.raises(ValueError, match="group_key"):
            validate_contract(contract)

    def test_ratios_must_sum_to_one(self):
        contract = self._tabular()
        contract["split"]["ratios"] = {"train": 0.9, "val": 0.3}
        with pytest.raises(ValueError, match="ratios"):
            validate_contract(contract)

    def test_reconciliation_key_must_exist(self):
        contract = self._tabular()
        contract["reconciliation"][0]["key"] = "not_a_key"
        with pytest.raises(ValueError, match="not_a_key"):
            validate_contract(contract)

    def test_reconciliation_missing_field(self):
        contract = self._tabular()
        del contract["reconciliation"][0]["resolution"]
        with pytest.raises(ValueError, match="resolution"):
            validate_contract(contract)


class TestSaveContractRejectsInvalid:

    def test_save_contract_validates_before_writing(self, tmp_path):
        contract = copy.deepcopy(TABULAR_CONTRACT)
        del contract["version"]
        path = tmp_path / "dataset_contract.yaml"
        with pytest.raises(ValueError):
            save_contract(path, contract)
        assert not path.exists()
