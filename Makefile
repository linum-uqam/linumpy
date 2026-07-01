# Linumpy development Makefile
# All Python commands use uv so the virtual environment is managed automatically.

.PHONY: help install install-dev install-gpu lint format typecheck check test test-scripts test-all \
        nf-lint nf-test nf-all docs docs-clean deploy pre-commit

# ── Python venv ───────────────────────────────────────────────────────────────

install:          ## Install runtime dependencies and editable package
	uv sync
	uv pip install -e .

install-dev:      ## Install all dependencies including dev extras
	uv sync --all-extras
	uv pip install -e .

install-gpu:      ## Install with GPU extras (CUDA 13)
	uv sync --extra gpu --extra gds
	uv pip install -e .

# ── Code quality ──────────────────────────────────────────────────────────────

lint:             ## Run ruff linter (auto-fix enabled)
	uv run ruff check

format:           ## Run ruff formatter
	uv run ruff format

format-check:     ## Check formatting without making changes
	uv run ruff format --check

typecheck:        ## Run ty type checker
	uv run ty check

check: lint format-check typecheck  ## Run all quality checks (no auto-fix)

# ── Tests ─────────────────────────────────────────────────────────────────────

test:             ## Run linumpy unit tests
	uv run pytest linumpy/tests/ -x -v

test-scripts:     ## Run script entry-point tests
	uv run pytest scripts/tests/ -x -v

test-all:         ## Run all Python tests
	uv run pytest linumpy/tests/ scripts/tests/ -x -v

test-cov:         ## Run all Python tests with coverage report
	uv run pytest linumpy/tests/ scripts/tests/ --cov=linumpy --cov-report=term-missing

# ── Nextflow ──────────────────────────────────────────────────────────────────

nf-lint:          ## Lint all Nextflow pipeline files
	nextflow lint -project-dir workflows/preproc workflows/preproc/preproc_rawtiles.nf
	nextflow lint -project-dir workflows/reconst_3d workflows/reconst_3d/soct_3d_reconst.nf
	nextflow lint -project-dir workflows/reconst_2.5d workflows/reconst_2.5d/soct_2.5d_reconst.nf

nf-format:        ## Lint and auto-format all Nextflow pipeline files
	nextflow lint -project-dir workflows/preproc -format -harshil-alignment -sort-declarations workflows/preproc/preproc_rawtiles.nf
	nextflow lint -project-dir workflows/reconst_3d -format -harshil-alignment -sort-declarations workflows/reconst_3d/soct_3d_reconst.nf
	nextflow lint -project-dir workflows/reconst_2.5d -format -harshil-alignment -sort-declarations workflows/reconst_2.5d/soct_2.5d_reconst.nf

nf-test:          ## Run all nf-test stub-run tests
	nf-test test --stop-on-first-failure workflows/preproc/tests/ workflows/reconst_3d/tests/ workflows/reconst_2.5d/tests/

nf-test-preproc:  ## Run only preprocessing pipeline tests
	nf-test test --stop-on-first-failure workflows/preproc/tests/

nf-test-3d:       ## Run only 3D reconstruction pipeline tests
	nf-test test --stop-on-first-failure workflows/reconst_3d/tests/

nf-test-2.5d:     ## Run only 2.5D reconstruction pipeline tests
	nf-test test --stop-on-first-failure workflows/reconst_2.5d/tests/

nf-all: nf-lint nf-test  ## Lint + test all Nextflow pipelines

# ── Docs ──────────────────────────────────────────────────────────────────────

docs:             ## Build HTML documentation
	$(MAKE) -C docs html

docs-live:        ## Start live-reloading docs server
	$(MAKE) -C docs livehtml

docs-clean:       ## Remove docs build artifacts
	$(MAKE) -C docs clean

# ── Git / deploy ─────────────────────────────────────────────────────────────

pre-commit:       ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

# ── Help ──────────────────────────────────────────────────────────────────────

help:             ## Show this help message
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
