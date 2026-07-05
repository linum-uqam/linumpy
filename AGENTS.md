# Agent Instructions
## Project Overview

**Linumpy** is the main library of the *Laboratoire d'Imagerie Numérique, Neurophotonique et Microscopie* (LINUM). It provides tools for serial histology and microscopy data: OCT acquisition, preprocessing, tile stitching, 3-D reconstruction, stacking, and analysis.

- **Repo**: github.com/linum-uqam/linumpy — default branch `main`, development on `dev`
- **Python**: 3.14+ (`.venv/bin/python` at repo root)
- **Key deps**: numpy 2.5+, scipy 1.18+ (via `[tool.uv] override-dependencies` — basicpy pins scipy<1.13 but we lift that ceiling for modern numpy/scipy), scikit-image, SimpleITK, BaSiCPy/basicpy>=2.0.0,<2.1 (PyTorch backend, torch constrained to >=2.12.0,<2.13), Zarr 3.0+, OME-Zarr 0.9+, napari
- **linum-basic**: git branch `modernisation` on Linum-BaSiC (not a PyPI release yet; uv.lock pins the resolved rev — switch to a tag when a proper release exists)

## Repository Layout

| Path | Purpose |
|------|---------|
| `linumpy/` | Core library (import as `linumpy`) |
| `linumpy/preproc/` | Preprocessing: XYZ correction (`xyzcorr.py`), intensity correction (`icorr.py`), normalization |
| `linumpy/stitching/` | Tile stitching, motor-based stacking, registration, interpolation |
| `linumpy/microscope/` | OCT data loader (`oct.py`) |
| `linumpy/io/` | OME-Zarr, NPZ, Thorlabs metadata readers |
| `linumpy/gpu/` | Optional CUDA-accelerated operations (CuPy); falls back to CPU |
| `linumpy/shifts/` | XY shift utilities for slice alignment |
| `linumpy/utils/` | Image quality, metrics, orientation, visualization |
| `scripts/` | ~90 `linum_*.py` CLI entry points; `scripts/tests/` holds their tests |
| `workflows/preproc/` | Nextflow preprocessing pipeline (`preproc_rawtiles.nf`) |
| `workflows/reconst_3d/` | 3-D reconstruction pipeline (`soct_3d_reconst.nf`) |
| `workflows/reconst_2.5d/` | 2.5-D reconstruction pipeline (`soct_2.5d_reconst.nf`) |

## Key Conventions

### Array axis order
- **3D volumes**: **(Z, Y, X)** — Z = depth/axial, Y = first lateral, X = second lateral
- **2D images**: **(Y, X)**
- OME-Zarr files store data as **(Z, Y, X)**; scripts transpose as needed

### File formats
- OME-Zarr: `.ome.zarr` (multi-resolution pyramids, NGFF spec)
- XY shifts: CSV with columns `fixed_id, moving_id, x_shift, y_shift, x_shift_mm, y_shift_mm`
- Reconstructed volumes: NIfTI (`.nii.gz`) or Zarr for intermediate steps

### Thread config import order
`linumpy._thread_config` **must be imported before** numpy, scipy, or SimpleITK. It is a first-party module placed in its own isort section for this reason.

### GPU support
GPU scripts have `_gpu` suffixes. The `linumpy/gpu/` subpackage wraps CuPy. Public GPU helpers use `np.ndarray` / `ArrayLike` for host arrays; CuPy/cupyx/numba/kvikio imports stay `Any` via `[tool.ty.analysis] replace-imports-with-any`. Internal duck-typed helpers remain loosely typed.

## Code Quality Checks

After every code change, run ruff and ty before considering the task done. **`ty` is blocking** in CI (`.github/workflows/python-app.yml`) and pre-commit — a nonzero exit fails the gate.

Full run:

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
```

Per file:

```bash
uv run ruff check <changed_file>
uv run ruff format --check <changed_file>
uv run ty check <changed_file>
```

Fix any errors reported before proceeding. For ruff, apply the suggested fix (e.g. collapse `if`/`else` into ternary when prompted by SIM108 etc.).

Ruff lint rule set: `E, F, W, I, UP, B, RUF, SIM, C4, PIE, PTH` — target Python 3.14, line length 127.
`ty` checks `linumpy/` and `scripts/` (see `[tool.ty.src]` in `pyproject.toml`); optional GPU deps are typed as `Any` via `replace-imports-with-any`.

Place as little ignore comments as possible. If you find yourself adding many ignores, consider whether the code can be refactored to be more compliant with the lint rules. Also reduce union types where possible to be clearer about the typing.

## Tests

Run the unit tests for modified modules:

```bash
uv run pytest linumpy/tests/ -x -v
```

For script changes also run:

```bash
uv run pytest scripts/tests/ -x -v
```

Use `uv run pytest` (not `.venv/bin/python -m pytest`) so the venv `bin/` is in PATH — required for subprocess-mode tests that resolve entry points via `shutil.which()`.

## Nextflow Pipeline Testing

nf-test is used for CI pipeline testing. All processes in `preproc_rawtiles.nf` and `soct_3d_reconst.nf` have `stub:` blocks so tests can run without real imaging data.

**Run tests locally** (requires `nf-test` and `nextflow` on PATH):

```bash
nf-test test workflows/preproc/tests/
nf-test test workflows/reconst_3d/tests/
```

**Linting issues must be fixed**

```bash
nextflow lint -project-dir workflows/preproc workflows/preproc/preproc_rawtiles.nf
nextflow lint -project-dir workflows/reconst_3d workflows/reconst_3d/soct_3d_reconst.nf
```

**Formatting** (`-format` only touches files with no errors; skipped if errors exist):

```bash
nextflow lint -project-dir workflows/preproc -format -harshil-alignment -sort-declarations workflows/preproc/preproc_rawtiles.nf
nextflow lint -project-dir workflows/reconst_3d -format -harshil-alignment -sort-declarations workflows/reconst_3d/soct_3d_reconst.nf
```

Always run nextflow tests and linting when the pipeline code is modified, even if the changes are small. The tests ensure that the processes run end-to-end and produce the expected outputs, even with stubbed data. This helps catch issues with process definitions, input/output handling, and config parsing early in development.

**Key files:**

| File | Purpose |
|------|---------|
| `nf-test.config` | Root nf-test configuration |
| `workflows/preproc/tests/preproc_rawtiles.nf.test` | Preproc pipeline stub-run test |
| `workflows/reconst_3d/tests/soct_3d_reconst.nf.test` | Reconst pipeline stub-run test |
| `workflows/preproc/tests/data/` | Minimal test input (empty tile dirs) |
| `workflows/reconst_3d/tests/data/` | Minimal test input (empty zarr dirs + shifts_xy.csv) |

Tests run in CI via `.github/workflows/nextflow-ci.yml` (triggers on `workflows/**` changes to `main`/`dev`).

**Adding stubs for new processes:** Each `process` block must have a `stub:` section that creates the expected outputs using `touch`/`mkdir -p`. The stub runs instead of the real `script:` when `-stub` is passed.

## Python Environment

Use `.venv/bin/python` (Python 3.14, virtual-env at repo root).
`uv` is used for dependency and virtual-env management:
```bash
uv sync          # install / update all dependencies from pyproject.toml
uv pip install -e .   # editable install of linumpy
```

## Remote Server & Deployment

**Server**: `132.207.157.41`
- Linumpy code: `/home/frans/code/linumpy/`
- Subject workspaces: `/scratch/workspace/sub-XX/`
  - `mosaic-grids/` — input tiles and `shifts_xy.csv`
  - `output/` — all pipeline outputs
  - `nextflow.config` — subject-specific parameter overrides

Ssh into the server to run commands. The reconstruction pipeline is executed on the server using Nextflow, which manages dependencies and parallelization. The workflow file can be different than the default `soct_3d_reconst.nf` due to development. Don't overwrite these changes.

### Deploying code changes

**Always use git to deploy code changes.** Do NOT use `scp` to copy files directly — it breaks file permissions (especially executable scripts) and risks overwriting in-progress server-side changes.

Keep commits brief, do not use the expand editor for detailed commit messages. The commit message should be a one-line description of the change.

When distributing commits from dev accross open PRs or to new PRs, rearange dev first to make the commits easier to move. For example, if you have 3 commits in dev but only want to move the first and third to a new PR, use `git rebase -i` to reorder them as 1, 3, 2 before creating the new PR. Also squash any closely related commits together on dev to make the history cleaner and easier to understand. Dev is easy to modify, PRs are not. Always make sure dev has a clean, logical commit history before branching PRs off of it. Dev is an independent branch that can be rearranged as needed to facilitate PR creation, the PR chain should match dev exactly when the PRs are merged back into main.

Before distributing commits, make sure the code in dev is clean and all tests pass. This ensures that the commits you are moving to PRs are stable and won't introduce issues. Also check the diff of the commits to ensure they only include the intended changes and don't have any unrelated modifications. Always check the PRs to ensure a correct match to the intended commits.

```bash
# 1. Commit and push locally
git add -A && git commit -m "description" && git push

# 2. Pull on server and reinstall
ssh 132.207.157.41 "export PATH=/home/frans/.local/bin:\$PATH; cd /home/frans/code/linumpy && git pull && uv pip install -e ."
```

### Running the reconstruction pipeline

```bash
nextflow run /home/frans/code/linumpy/workflows/reconst_3d/soct_3d_reconst.nf \
  -resume \
  -c /scratch/workspace/sub-18/nextflow.config \
  --input /scratch/workspace/sub-18/mosaic-grids/ \
  --output /scratch/workspace/sub-18/output/
```

- `-resume` reuses cached process outputs; use it by default to avoid re-running unchanged steps.
- Subject-specific config (e.g. `/scratch/workspace/sub-18/nextflow.config`) overrides the default `workflows/reconst_3d/nextflow.config`.

**Never run nextflow as an agent.** The pipeline takes hours to complete and will time out. The nextflow executable on the server (`/usr/local/bin/nextflow`) is owned by another user and may not be directly executable — the user must run it themselves in an interactive SSH session. Instead, provide the user with the exact `nextflow run` command to copy-paste.

### Downloading results locally

Use the repo-parameterized download script instead of the old per-subject
`~/Downloads/sub-XX/update_files.sh` copies. It reads the subject's remote
`nextflow.config` first, then downloads only the outputs that match the
enabled flags (common_space_previews, stack previews, diagnostics, zarr
archive, etc.):
```bash
scripts/subject/update_files.sh sub-18
scripts/subject/update_files.sh sub-22
```

See `docs/SUBJECT_TUNING.md` for the full subject tuning workflow.

Key output locations pulled by those scripts:

| Remote path | What it contains |
|-------------|-----------------|
| `output/common_space_previews/` | 3-panel PNG per slice (XY/XZ/YZ views) |
| `output/stack/` | Stacked zarr, z_matches.csv, stacking_decisions.csv, metrics |
| `output/correct_bias_field/` | Bias-corrected zarr + previews (when `correct_bias_field=true`) |
| `output/register_pairwise/` | Pairwise registration transforms |
| `output/align_to_ras/` | Atlas-aligned zarr + transform + preview |
| `output/detect_rehoming_events/` | Corrected shifts_xy_clean.csv + diagnostics |
| `mosaic-grids/previews/` | Per-tile mosaic preview PNGs |

## Reconstruction Pipeline: Diagnostics & Config Knowledge

Human cheat sheet: `docs/RECONSTRUCTION_TUNING_QUICKREF.md`  
Parameter deep-dives: `docs/RECONSTRUCTION_TUNING.md`  
Script-level diagnostics: `docs/RECONSTRUCTION_DIAGNOSTICS.md`

