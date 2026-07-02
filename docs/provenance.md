# Provenance

DSAgt makes every data operation a reproducible, auditable step. The agent registers a **code** — a CLI executable — and every run of that code is wrapped for provenance capture, so the whole pipeline can later be reconstructed from the record.

## Codes

Codes are CLI executables defined as markdown files with YAML frontmatter under `<project>/codes/`. The agent registers new codes via the MCP server's `save_code_spec` tool and finds existing ones via `search_registry`.

A code spec includes:

- A YAML frontmatter block describing the command, arguments, dependencies, and tags.
- A markdown body with usage examples and notes for the agent.

Example code spec:

```markdown
---
name: csvstat
command: csvstat
dependencies: []
tags: [csv, statistics]
---

Prints descriptive statistics for all columns in a CSV file.

Usage: csvstat [options] [FILE]
```

DSAgt wraps every registered code with `dsagt-run` for provenance capture and `uv run --with` for Python dependencies, so the agent can call any code without managing environments manually. It ships one bundled code, `scan-directory`, indexed for search by `dsagt init`.

## Execution capture

Every registered code runs through the `dsagt-run` wrapper. For each call it records the command, arguments, exit code, duration, input/output file counts, and truncated stderr to `<project>/trace_archive/<record_id>.json`, and emits a `code.execute` span to the trace store. The MCP server incrementally indexes those records into the `code_use` collection, so past executions are searchable.

The wrapper is the whole point of code-mediated data access: a direct shell or editor call leaves no record and breaks reconstruction.

## Pipeline reconstruction

The on-disk execution records are the canonical provenance chain. The agent calls `reconstruct_pipeline` to render the trace archive as a reproducible **bash script** (`format="bash"`) or **Snakemake workflow** (`format="snakemake"`). It flushes the latest records into the searchable index first, then walks the dependency graph inferred from each step's input/output files to order the steps.

## Try it

```bash
dsagt init            # follow the prompts: name it `demo`, then pick your agent
dsagt start demo      # launch the agent in the project
```

Then, in the agent:

1. > Register the csvkit codes `csvstat` and `csvcut`.
2. > Use `csvstat` from the registry on `data/samples.csv` and summarize the columns.
3. > Reconstruct the pipeline as a bash script.

Afterwards, inspect the trail:

```bash
ls ~/dsagt-projects/demo/{codes,trace_archive}          # the specs + execution records
mlflow ui --backend-store-uri sqlite:///$HOME/dsagt-projects/demo/mlflow.db   # code.execute spans
```

## In practice

See the [Use Cases](use-cases/index.md) for provenance-captured pipelines on real datasets — for example registering `fastp` and `megahit` as codes and reconstructing a genomics QC-and-assembly pipeline end to end.
