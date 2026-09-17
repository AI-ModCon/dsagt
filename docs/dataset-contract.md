# Dataset Contract

A **sample contract** is the pivot artifact for turning a data directory into a
training-ready PyTorch dataset package. It is a YAML file, persisted at
`<project>/dataset_contract.yaml` (project root, not `.dsagt/`, because it is a
user-facing artifact the scientist reviews and signs off on) declaring what one
training sample looks like, how samples collate into a batch, how the data
splits, and which side owns normalization.

This page documents the schema (`dsagt.contract`). The tooling that builds a
contract from a pipeline or a `Dataset` and consumes it (`check-dataset`, the
`dataset-builder` skill, the package scaffolder, the reference model) is
separate, forthcoming work; this schema is what they share.

## Two contracts, reconciled

A `Dataset`'s `__getitem__` is the adapter between two views of the same
sample:

- **Producer contract**: what is actually on disk.
- **Consumer contract**: what `model.forward()` expects.

The `reconciliation` section of the schema records where the two differ and
how the adapter resolves each difference: dtype casts, channel layout,
normalization ownership, padding and ragged handling, label encoding.

## Schema

```yaml
version: 1                    # schema version, int
mode: pipeline                # "pipeline" | "standalone"
pipeline_fingerprint: sha256:...   # required iff mode == pipeline; absent iff standalone

keys:
  <key_name>:
    dtype: float32             # framework dtype name
    shape: [N, 3]              # ints, or symbolic dim names (e.g. "N" for a
                                # variable node count)
    role: input                 # input | target | mask | metadata
    value_range: [0.0, 1.0]     # optional [min, max]
    collation: stack            # per-key collation rule (stack, graph_batch, list, ...)

normalization:
  owner: pipeline               # pipeline | dataset — whether a pipeline step
                                 # already normalized, or __getitem__ must

split:
  strategy: group                # random | group | time
  group_key: <key_name>          # required for group/time — the id samples
                                  # sharing it must not be split across
  seed: 42
  ratios:
    train: 0.7
    val: 0.15
    test: 0.15

reconciliation:
  - key: <key_name>
    producer: "what's actually on disk"
    consumer: "what model.forward() expects"
    resolution: "how __getitem__ bridges the two"
```

`dsagt.contract.validate_contract` enforces this shape; `load_contract` /
`save_contract` validate on the way in and out.

## Field reference

**`mode`**

| Value | Meaning |
|---|---|
| `pipeline` | The sample's inputs are terminal outputs of a DSAgt-tracked pipeline. The contract carries `pipeline_fingerprint`. |
| `standalone` | No DSAgt pipeline; the data root was characterized directly (e.g. via the `scan-directory` code). No pipeline to fingerprint — the field must be absent. |

**`keys.<name>.role`** — what `__getitem__` hands the training loop

| Value | Meaning |
|---|---|
| `input` | Fed to `model.forward()`. |
| `target` | The ground truth compared against the model's output in the loss. |
| `mask` | A boolean/float mask consumed alongside an input or target key (padding mask, loss mask, node/edge validity mask). |
| `metadata` | Carried for bookkeeping, splitting, or debugging (ids, provenance, plot coordinates) — never passed to `forward()` or the loss. |

**`keys.<name>.collation`** is a free-form string, not a closed enum — the collate rule a key needs is domain-specific, and mapping a name to generated `collate_fn` code is the dataset-builder skill's job, not this schema's. Common values seen in practice: `stack` (equal-shape tensors, `torch.stack`), `pad` (ragged sequences, pad to batch max), `list` (leave as a Python list, e.g. non-tensor metadata), `graph_batch` (PyG-style disjoint-union batching for graph keys).

**`normalization.owner`** — who has already applied centering/scaling to a key's raw values

| Value | Meaning |
|---|---|
| `pipeline` | An upstream registered code already normalized the data on disk; `__getitem__` passes the value through unchanged. |
| `dataset` | No upstream normalization; `__getitem__` computes and applies it itself (e.g. using statistics fit over the train split). |

**`split.strategy`** — how `split.ratios` is turned into train/val/test membership

| Value | `split.group_key` | Meaning |
|---|---|---|
| `random` | not used | i.i.d. row-level assignment, seeded by `split.seed`. Leaks whenever samples share an identity (a patient, a simulation case, augmented copies of one image); use only when samples are genuinely independent. |
| `group` | required | Every sample whose `group_key` takes the same value is assigned to the same split, so no group straddles a split boundary. |
| `time` | required | Samples are ordered by `group_key`'s value (a timestamp or step index) and cut chronologically, so later time periods are held out rather than interleaved with training data. |

## The pipeline fingerprint

In pipeline mode, the contract carries a hash over the upstream pipeline's
structural shape, computed by `provenance.compute_pipeline_fingerprint` from
`reconstruct_pipeline(..., fmt="json")`'s `dependency_graph` and
`terminal_outputs` (never `records` — timestamps and stdout vary rerun to
rerun even when the pipeline hasn't changed). A later staleness check
recomputes the fingerprint and flags the contract for review when it no
longer matches: a step was added or removed, or an output path changed.

Standalone mode (no DSAgt pipeline; the data root was characterized directly
with codes like `scan-directory`) has no pipeline to fingerprint, so the
field is absent.

## Worked example: tabular case (standalone)

A CSV of patient records: 37 numeric features, an integer label, grouped by
patient so repeated visits from the same patient never split across
train/val/test.

```yaml
version: 1
mode: standalone

keys:
  features:
    dtype: float32
    shape: [37]
    role: input
    value_range: [-3.0, 3.0]
    collation: stack
  label:
    dtype: int64
    shape: []
    role: target
    collation: stack
  patient_id:
    dtype: string
    shape: []
    role: metadata
    collation: list

normalization:
  owner: dataset

split:
  strategy: group
  group_key: patient_id
  seed: 42
  ratios:
    train: 0.7
    val: 0.15
    test: 0.15

reconciliation:
  - key: features
    producer: float64 columns in the source CSV, unnormalized
    consumer: float32 tensor, zero mean / unit variance expected by model.forward()
    resolution: cast to float32 and standardize in __getitem__ using stats computed
      over the train split
  - key: label
    producer: string category name
    consumer: integer class index expected by the loss function
    resolution: label encoding fit at contract-build time; mapping stored alongside
      the split manifest
```

## Worked example: XGC graph case (pipeline)

[`XGCGraphDataset`](https://github.com/AI-ModCon/dsagt/blob/main/use_cases/fusion-fm/skills/xgc-ai-training/scripts/xgc_dataset.py)
returns a PyG `Data` graph per `(phi_plane, start_timestep)` sample: node
features `x`, target field values `y`, node positions, mesh edges, and a few
scalar metadata fields. Splits are made by phi-plane group so every timestep
from one plane stays in one split. This confirms the schema handles a
variable node count (`N`), an edge dimension (`E`), and PyG's own graph
collation.

```yaml
version: 1
mode: pipeline
pipeline_fingerprint: sha256:6f1ea1e0c1a1c9e5c9b9f2e8f7d4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6

keys:
  x:
    dtype: float32
    shape: [N, n_steps, "7+F"]
    role: input
    collation: graph_batch
  pos:
    dtype: float32
    shape: [N, 2]
    role: input
    collation: graph_batch
  edge_index:
    dtype: int64
    shape: [2, E]
    role: input
    collation: graph_batch
  edge_attr:
    dtype: float32
    shape: [E, 3]
    role: input
    collation: graph_batch
  leadtime:
    dtype: float32
    shape: [1, 1]
    role: input
    collation: stack
  y:
    dtype: float32
    shape: [N, F]
    role: target
    collation: graph_batch
  phi:
    dtype: int64
    shape: []
    role: metadata
    collation: list
  step0:
    dtype: int64
    shape: []
    role: metadata
    collation: list
  target_step:
    dtype: int64
    shape: []
    role: metadata
    collation: list

normalization:
  owner: pipeline

split:
  strategy: group
  group_key: phi
  seed: 7
  ratios:
    train: 0.7
    val: 0.15
    test: 0.15

reconciliation:
  - key: x
    producer: per-field npz arrays, one file per simulation step
    resolution: __getitem__ loads the requested step_*.npz files and concatenates
      static_ctx with the field slice
    consumer: single [N, n_steps, 7+F] tensor stacking static node context and
      per-step field values
  - key: edge_index
    producer: triangular mesh connectivity in mesh.npz
    consumer: undirected PyG edge_index expected by the message-passing layers
    resolution: cells_to_edge_index(undirected=True), cached once as topology.pt
```

Both worked examples are validated as part of the test suite (`tests/test_contract.py`).
