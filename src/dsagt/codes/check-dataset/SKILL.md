---
name: check-dataset
description: Validate a PyTorch Dataset against its sample contract, running
  each check (contract, determinism, worker equivalence, split leakage,
  throughput, model forward, contract staleness) in an isolated subprocess
executable: dsagt-run --code check-dataset -- python codes/check-dataset/scripts/check_dataset.py
parameters:
  contract:
    type: string
    required: true
    cli: "--contract"
    description: Path to the dataset_contract.yaml being validated
  dataset:
    type: string
    required: true
    cli: "--dataset"
    description: Import path (module:attr) to the Dataset/IterableDataset class or a zero/kwarg factory
  dataset_args:
    type: string
    required: false
    cli: "--dataset-args"
    description: JSON object of kwargs passed when constructing the dataset
  output:
    type: string
    required: false
    cli: "--output"
    description: Report path (default audit/check_dataset_<timestamp>.json)
  checks:
    type: string
    required: false
    default: all
    cli: "--checks"
    description: Comma-separated subset of contract,determinism,worker_equivalence,split_leakage,throughput,model_forward,staleness
  seed:
    type: integer
    required: false
    default: 0
    cli: "--seed"
    description: Seed used by the determinism check
  sample_index:
    type: integer
    required: false
    default: 0
    cli: "--sample-index"
    description: Sample index read by the contract and determinism checks
  id_key:
    type: string
    required: false
    cli: "--id-key"
    description: Contract key used as per-sample identity by worker_equivalence (default the first role:metadata key)
  split_manifest:
    type: string
    required: false
    cli: "--split-manifest"
    description: JSON file for split_leakage {"<split_name>": [<group ids or row indices>]}
  n_throughput_samples:
    type: integer
    required: false
    default: 50
    cli: "--n-throughput-samples"
    description: Number of random-index __getitem__ calls timed by the throughput check
  slow_threshold_s:
    type: number
    required: false
    default: 5.0
    cli: "--slow-threshold-s"
    description: Per-item latency in seconds that fails the throughput check
  model:
    type: string
    required: false
    cli: "--model"
    description: Import path (module:attr) to a torch.nn.Module class or factory, for the model_forward check
  model_args:
    type: string
    required: false
    cli: "--model-args"
    description: JSON object of kwargs passed when constructing the model
  collate_fn:
    type: string
    required: false
    cli: "--collate-fn"
    description: Import path to the collate function used by model_forward (default torch's default_collate)
  batch_size:
    type: integer
    required: false
    default: 2
    cli: "--batch-size"
    description: Batch size used by the model_forward check
  project_dir:
    type: string
    required: false
    default: "."
    cli: "--project-dir"
    description: Project root containing trace_archive/, used by the staleness check
  check_timeout_s:
    type: number
    required: false
    default: 300.0
    cli: "--check-timeout-s"
    description: Seconds to wait for one check's subprocess before killing it and reporting status "error"
---

# check-dataset

Validate a generated PyTorch `Dataset` (or `IterableDataset`) against its
sample contract (`dataset_contract.yaml`, see `docs/dataset-contract.md`).
This is what separates the dataset-builder feature from a prompt that writes
a `Dataset`: every claim the contract makes is checked by execution, not by
inspection.

## Shell Command

```bash
dsagt-run --code check-dataset -- python codes/check-dataset/scripts/check_dataset.py \
  --contract dataset_contract.yaml \
  --dataset mypackage.dataset:MyDataset \
  --dataset-args '{"root": "data/processed"}'
```

## Checks

| Check | Catches | Needs |
|---|---|---|
| `contract` | `ds[i]` does not match declared keys, dtypes, shapes, ranges | dataset |
| `determinism` | Same seed produces a different sample | dataset |
| `worker_equivalence` | `num_workers=2` yields a different multiset than `num_workers=0` (iterable duplication); skipped for a map-style dataset | dataset, `--id-key` |
| `split_leakage` | Group ids appearing in more than one split; declared sizes wrong; skipped without `--split-manifest` | `--split-manifest` |
| `throughput` | Pathologically slow `__getitem__` | dataset |
| `model_forward` | Batch fails to pass through `model.forward()`; skipped without `--model` | dataset, `--model` |
| `staleness` | The upstream pipeline changed after the contract was written; skipped for a `standalone`-mode contract | `--project-dir` |

Each check runs in its own subprocess: importing the user's dataset into a
shared process would contaminate the fork-sensitive state
`worker_equivalence` measures (open file handles, a seeded global RNG, an
initialized CUDA context), and a segfault in a native reader (HDF5, ADIOS2)
must not take down the whole run. Select a subset with `--checks` when the
throughput check is too slow to run every time.

## Output

A JSON report to stdout and to `--output` (default
`audit/check_dataset_<timestamp>.json`):

```json
{
  "dataset": "mypackage.dataset:MyDataset",
  "contract": "dataset_contract.yaml",
  "checks_run": ["contract", "determinism", ...],
  "overall_passed": true,
  "checks": {
    "contract": {"status": "passed", "detail": {...}, "duration_ms": 12.3},
    "model_forward": {"status": "skipped", "reason": "--model not provided", "duration_ms": 1.1}
  }
}
```

`status` is one of `passed`, `failed`, `skipped`, or `error` (the worker
subprocess itself crashed, produced no parseable JSON, or was killed after
exceeding `--check-timeout-s`, rather than observing a check failure). Exit
code is 0 only if every requested, non-skipped check passed.
