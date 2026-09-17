.PHONY: help install install-dev test test-cov lint format docs clean

help:
	@echo "DSAgt - Development Tasks"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          Sync the project environment with uv"
	@echo "  make install-dev      Sync all dependency groups with uv"
	@echo "  make test             Run the non-integration test suite"
	@echo "  make test-cov         Run it with a coverage report"
	@echo "  make lint             Run linting checks with ruff"
	@echo "  make format           Format code with black"
	@echo "  make docs             Build the documentation site"
	@echo "  make clean            Remove build artifacts and cache files"
	@echo "  make help             Show this help message"

install:
	uv sync

install-dev:
	uv sync --all-groups

test:
	uv run --no-sync python -m pytest -m "not integration" -q

test-cov:
	uv run --no-sync python -m pytest -m "not integration" -q --cov=dsagt --cov-report=term-missing

lint:
	uv run ruff check src tests

format:
	uv run black src tests

docs:
	uv run mkdocs build --strict

clean:
	find . -type f -name '*.py[cod]' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf site/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
