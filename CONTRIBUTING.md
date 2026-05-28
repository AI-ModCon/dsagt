# Contributing to dsagt

Thanks for your interest in contributing to dsagt.

## Getting Started

1. Fork the repository and create a feature branch from `main`.
2. Set up your environment and install dependencies.
3. Run tests and linting locally before opening a pull request.

See [developer.md](./developer.md) for project-specific developer workflows.

## Development Workflow

- Keep pull requests focused and small.
- Prefer clear commit messages that explain intent.
- Update documentation when behavior changes.

## Testing

Run non-integration tests before submitting:

```bash
uv run python -m pytest -m "not integration"
```

Integration tests require `.env` configuration, as documented in [developer.md](./developer.md).

## Linting

Use ruff for lint checks:

```bash
python -m ruff check src tests --select E9,F63,F7,F82
```

## Pull Requests

When opening a PR:

- Explain what changed and why.
- Link the related issue(s).
- Include test evidence for the changed behavior.

## AI/LLM-Assisted Contributions

AI tools are welcome for drafting code and docs, but contributors remain responsible for correctness and security.

Please ensure:

- Generated changes are reviewed and understood before submission.
- Outputs are validated with local linting/tests.
- No secrets, proprietary data, or sensitive information are included in prompts or commits.

## Code of Conduct

By participating, you agree to follow the project's [Code of Conduct](./CODE_OF_CONDUCT.md).
