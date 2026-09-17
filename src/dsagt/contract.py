"""
Sample contract: the pivot artifact the dataset builder derives from.

A **sample contract** is a YAML file, persisted at ``<project>/dataset_contract.yaml``,
declaring what one training sample looks like: its keys (dtype, shape, value
range, semantic role), the collation rule per key, the split policy, and which
side (an upstream pipeline step, or ``__getitem__``) owns normalization. It is
written at the project root rather than under ``.dsagt/`` because it is a
user-facing artifact the scientist reviews and signs off on, not server-owned
state.

The contract reconciles two views of the same sample: the **producer**
contract (what is actually on disk) and the **consumer** contract (what
``model.forward()`` expects). The ``reconciliation`` section records where
they differ and how the generated ``Dataset`` adapts one to the other (dtype
casts, channel layout, normalization ownership, padding/ragged handling,
label encoding).

In **pipeline mode**, the contract also carries a ``pipeline_fingerprint``: a
hash (see ``provenance.compute_pipeline_fingerprint``) over the dependency
graph and terminal outputs of the upstream ``reconstruct_pipeline`` run that
produced the sample's inputs. A later staleness check recomputes the
fingerprint and flags the contract for review if the upstream pipeline has
changed. In **standalone mode** (no DSAgt pipeline; the data root was
characterized directly) there is nothing to fingerprint, so the field is
absent.

This module only defines and validates the schema; nothing here builds a
contract from a pipeline or a ``Dataset`` — that is the dataset-builder
skill's job.
"""

from __future__ import annotations

from pathlib import Path

import yaml

#: Filename the contract is persisted under, at the project root.
CONTRACT_FILENAME = "dataset_contract.yaml"

#: How the sample's inputs were produced.
#:   pipeline   - the inputs are terminal outputs of a DSAgt-tracked pipeline;
#:                the contract carries a pipeline_fingerprint (see below) for
#:                the staleness check.
#:   standalone - no DSAgt pipeline; the data root was characterized directly
#:                (e.g. via the scan-directory code). No pipeline to
#:                fingerprint, so the field must be absent.
VALID_MODES = {"pipeline", "standalone"}

#: What a sample key is for, i.e. what __getitem__ hands the training loop.
#:   input    - fed to model.forward().
#:   target   - the ground truth compared against the model's output in the
#:              loss.
#:   mask     - a boolean/float mask consumed alongside an input or target key
#:              (padding mask, loss mask, node/edge validity mask).
#:   metadata - carried on the sample for bookkeeping, splitting, or debugging
#:              (ids, provenance, plot coordinates) but never passed to
#:              forward() or the loss.
VALID_ROLES = {"input", "target", "mask", "metadata"}

#: Who has already applied normalization (centering/scaling) to a key's raw
#: values.
#:   pipeline - an upstream registered code already normalized the data on
#:              disk; __getitem__ passes the value through unchanged.
#:   dataset  - no upstream normalization; __getitem__ computes and applies it
#:              itself (e.g. using statistics fit over the train split).
VALID_NORMALIZATION_OWNERS = {"pipeline", "dataset"}

#: How split.ratios is turned into train/val/test membership.
#:   random - i.i.d. row-level assignment, seeded by split.seed. Leaks
#:            whenever samples share an identity (a patient, a simulation
#:            case, augmented copies of one image); use only when samples are
#:            genuinely independent.
#:   group  - every sample whose split.group_key takes the same value is
#:            assigned to the same split, so no group straddles a boundary.
#:   time   - samples are ordered by split.group_key's value (a timestamp or
#:            step index) and cut chronologically, so later time periods are
#:            held out rather than interleaved with training data.
VALID_SPLIT_STRATEGIES = {"random", "group", "time"}

#: Split strategies where split.group_key is required: "group" clusters equal
#: values into one split, "time" sorts by the same field before cutting
#: chronologically. Excludes "random", which has no notion of a key to
#: leak across.
_GROUPED_SPLIT_STRATEGIES = {"group", "time"}

_RATIO_TOLERANCE = 1e-6


def validate_contract(contract: dict) -> None:
    """Validate a sample contract dict against the schema. Raises ``ValueError``."""
    version = contract.get("version")
    if not isinstance(version, int):
        raise ValueError(f"'version' must be an int, got {version!r}")

    mode = contract.get("mode")
    if mode not in VALID_MODES:
        raise ValueError(f"'mode' must be one of {sorted(VALID_MODES)}, got {mode!r}")

    fingerprint = contract.get("pipeline_fingerprint")
    if mode == "pipeline":
        if not fingerprint or not isinstance(fingerprint, str):
            raise ValueError(
                "'pipeline_fingerprint' is required when mode is 'pipeline'"
            )
    elif fingerprint is not None:
        raise ValueError(
            "'pipeline_fingerprint' must be absent when mode is 'standalone'"
        )

    keys = contract.get("keys")
    if not isinstance(keys, dict) or not keys:
        raise ValueError(
            "'keys' must be a non-empty mapping of sample key name to spec"
        )
    for name, spec in keys.items():
        _validate_key_spec(name, spec)

    _validate_normalization(contract.get("normalization"))
    _validate_split(contract.get("split"))

    reconciliation = contract.get("reconciliation", [])
    _validate_reconciliation(reconciliation, keys)


def _validate_key_spec(name: str, spec: dict) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"keys.{name} must be a mapping, got {type(spec).__name__}")

    dtype = spec.get("dtype")
    if not dtype or not isinstance(dtype, str):
        raise ValueError(f"keys.{name}.dtype is required and must be a string")

    shape = spec.get("shape")
    if not isinstance(shape, list):
        raise ValueError(
            f"keys.{name}.shape must be a list (ints or symbolic dim names)"
        )
    for dim in shape:
        if not isinstance(dim, (int, str)) or isinstance(dim, bool):
            raise ValueError(
                f"keys.{name}.shape entries must be ints or symbolic dim names, got {dim!r}"
            )

    role = spec.get("role")
    if role not in VALID_ROLES:
        raise ValueError(
            f"keys.{name}.role must be one of {sorted(VALID_ROLES)}, got {role!r}"
        )

    # Not a closed enum: the collate_fn a key needs is domain-specific (stack,
    # pad, list, graph_batch, ...), and the dataset-builder skill (not this
    # module) is what maps a collation name to generated code. Only presence
    # is validated here.
    collation = spec.get("collation")
    if not collation or not isinstance(collation, str):
        raise ValueError(f"keys.{name}.collation is required and must be a string")

    value_range = spec.get("value_range")
    if value_range is not None:
        if len(value_range) != 2 or value_range[0] > value_range[1]:
            raise ValueError(f"keys.{name}.value_range must be a 2-element [min, max]")


def _validate_normalization(normalization: dict) -> None:
    if not isinstance(normalization, dict):
        raise ValueError("'normalization' must be a mapping with an 'owner' field")
    owner = normalization.get("owner")
    if owner not in VALID_NORMALIZATION_OWNERS:
        raise ValueError(
            f"normalization.owner must be one of {sorted(VALID_NORMALIZATION_OWNERS)}, "
            f"got {owner!r}"
        )


def _validate_split(split: dict) -> None:
    if not isinstance(split, dict):
        raise ValueError("'split' must be a mapping")

    strategy = split.get("strategy")
    if strategy not in VALID_SPLIT_STRATEGIES:
        raise ValueError(
            f"split.strategy must be one of {sorted(VALID_SPLIT_STRATEGIES)}, got {strategy!r}"
        )

    if not isinstance(split.get("seed"), int):
        raise ValueError("split.seed is required and must be an int")

    if strategy in _GROUPED_SPLIT_STRATEGIES and not split.get("group_key"):
        raise ValueError(
            f"split.group_key is required when split.strategy is '{strategy}'"
        )

    ratios = split.get("ratios")
    if not isinstance(ratios, dict) or not ratios:
        raise ValueError(
            "split.ratios must be a non-empty mapping of split name to fraction"
        )
    total = sum(ratios.values())
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        raise ValueError(f"split.ratios must sum to 1.0, got {total}")


def _validate_reconciliation(reconciliation: list, keys: dict) -> None:
    if not isinstance(reconciliation, list):
        raise ValueError("'reconciliation' must be a list")
    for i, entry in enumerate(reconciliation):
        if not isinstance(entry, dict):
            raise ValueError(f"reconciliation[{i}] must be a mapping")
        key = entry.get("key")
        if key not in keys:
            raise ValueError(
                f"reconciliation[{i}].key {key!r} is not a declared sample key"
            )
        for field in ("producer", "consumer", "resolution"):
            if not entry.get(field):
                raise ValueError(f"reconciliation[{i}].{field} is required")


def load_contract(path: str | Path) -> dict:
    """Read and validate a sample contract from disk."""
    contract = yaml.safe_load(Path(path).read_text()) or {}
    validate_contract(contract)
    return contract


def save_contract(path: str | Path, contract: dict) -> None:
    """Validate and write a sample contract to disk."""
    validate_contract(contract)
    Path(path).write_text(
        yaml.dump(contract, default_flow_style=False, sort_keys=False)
    )
