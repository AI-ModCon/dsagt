# Use Cases

End-to-end walkthroughs for representative scientific and data-readiness scenarios are located in [`use_cases/`](https://github.com/AI-ModCon/dsagt/tree/main/use_cases/). Each follows one layout (Prerequisites, Setup, Execution as pasted prompts with expected results, Post-Conditions, Coverage, Cleanup) and covers data acquisition, code or skill registration, pipeline construction, and agent-driven execution. Some run on real scientific datasets; the skill-management demos use small fixture data.

Fixture data of a few megabytes is in the use case's own `data/` folder in the repository; larger datasets are hosted in the [DSAgt use-case data folder](https://drive.google.com/drive/folders/1RWQAJeHaikIaD7CCf8ciJ71m55S1erp6), and each Setup section gives the copy or download command.

![DSAgt use cases](../assets/use-cases.png)

<!-- USE_CASES_TABLE -->

Each use-case folder is laid out by role: `README.md` is the walkthrough; `data/`
holds fixture data small enough to keep in the repository; `docs/`
holds documents the agent reads; `scripts/` holds code copied into the project;
`skills/` holds skills copied into the project; `reference/` holds expected outputs
and reference solutions that are not inputs to the demo. Only the folders a use case
needs are present.

Dependencies follow one rule per kind. Python packages from PyPI that a walkthrough
needs are an extra of dsagt named after the walkthrough (`vasp-dft`,
`combustion-simulation`, `tokamak-stability`, `plasma-turbulence`), so
`pip install "dsagt[<use-case>] @ git+https://github.com/AI-ModCon/dsagt.git"`
installs them with dsagt (`uv sync --all-extras` in a checkout). Anything else (conda-only tools, libraries built
from source) is installed by the use case's `scripts/setup_env.sh` into
`~/dsagt-projects/.tools/<use-case>/`, a shared tools directory beside the projects;
the README's Prerequisites list what the script installs and what it needs already
present. Larger data comes from the Google Drive folder linked above.

!!! note "Adding a use case"
    Drop a `README.md` with frontmatter (`title`, `domain`, `summary`) into a
    `use_cases/<name>/` folder; it is added to this table, its body is
    inlined as its own page, and it appears in the nav. A folder without
    frontmatter is left out. See
    [`hooks/gen_use_cases.py`](https://github.com/AI-ModCon/dsagt/blob/main/hooks/gen_use_cases.py).
