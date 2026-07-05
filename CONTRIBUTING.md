# Contributing to linumpy

Thank you for contributing to linumpy! This document covers the ground rules for human contributors. For agent-facing library reference, see `AGENTS.md`.

## Fork + PR Workflow

- All contributions go through personal forks. Clone the upstream repo, create a branch on your fork, and open a PR targeting `linum-uqam/main`.
- No direct pushes to `main` or `dev` — even for repository admins, changes go through PR review.
- Keep your fork's `dev` branch up to date with `linum-uqam/dev` before creating new branches.

## PR Size Guidelines

- Target **<15–20 files changed** and **<500 lines** per PR. Smaller PRs review faster and are less likely to introduce regressions.
- The `pr-size-check` GitHub workflow comments on oversized PRs as a reminder. Mechanical refactoring PRs (renames, formatting, test additions) get reviewer discretion.
- If a feature requires a large diff, split it into a stack of small PRs (e.g. one per module or subsystem).

## Coding Standards

- **PEP8**: We follow the [PEP8](https://peps.python.org/pep-0008/) coding standard, enforced by `ruff`.
- **OME-Zarr extension**: Use `.ome.zarr` as the file extension, as shown in the [NGFF documentation](https://ngff.openmicroscopy.org/0.4/index.html#bf2raw-layout).
- **Resolution & dimension docs**: All scripts and methods must explicitly document the expected resolutions and dimension units (e.g. microns, pixels, axis order).
- **Array axis order**: 3D volumes use **(Z, Y, X)**; 2D images use **(Y, X)**. See `AGENTS.md` for the full convention reference.

## Test Requirements

Every PR must independently pass before opening:

```bash
uv run ruff check
uv run ruff format --check
uv run ty check linumpy
uv run pytest linumpy/tests/ -x -v
```

For changes to `workflows/**`, also run:

```bash
nf-test test workflows/
nextflow lint workflows/<pipeline>.nf
```

## Docstring Sync

Modified docstrings must match the real function/method signature. If you change a parameter name, type, or default, update the docstring in the same commit.

## Review Process

- `CODEOWNERS` auto-assigns reviewers based on the files changed. Ensure the assigned reviewer is appropriate.
- Required conversations must be resolved before merge. If a reviewer leaves a comment, either address it or explicitly resolve the conversation.
- Reviewers should check: correctness, test coverage, docstring accuracy, and adherence to the coding standards above.

## Task Ownership

Before starting work on a task or issue, confirm with the team lead that it isn't already assigned to someone else. This avoids duplicate work and merge conflicts.

## Commit Message Style

- One-line subject in the imperative mood (e.g. "Fix GPU backend DCT implementation").
- No attribution trailers (no `Generated with [Devin]`, no `Co-Authored-By` lines).
- Keep the body (if any) focused on *why*, not *what* — the diff already shows what changed.
