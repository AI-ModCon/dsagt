# CLI Reference

All commands are available after [installation](index.md#installation) and activating your virtual environment.

## Project Management

<!-- Shared with README.md — edit there, not here. -->
{%
   include-markdown "../README.md"
   start="<!-- md-shared:cli:start -->"
   end="<!-- md-shared:cli:end -->"
%}

### Deprecated `dsagt init` flags (backcompat)

The pre-menu flags still work — the automation/CI path — but are **deprecated** in favor of the interactive menu. Each one skips its corresponding prompt:

| Flag | Prompt it replaces |
|------|--------------------|
| `<name>` (positional) | Project name |
| `--agent <platform>` | Agent platform |
| `--location <path>` | Project location |
| `--include … \| --exclude …` | Knowledge collections / skill sources |
| `--episodic` | "Enable episodic memory?" |

New usage should prefer bare `dsagt init` and the menu.

## Project Location

The default project location is `~/dsagt-projects/<name>/`. 

## Viewing traces

The trace store is a serverless SQLite file — browse it with `dsagt traces`:

```bash
dsagt traces <name>   # runs: mlflow ui --backend-store-uri sqlite:///<project-path>/mlflow.db
```
