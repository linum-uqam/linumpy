# Agent Instructions
Do not commit the agents.md file. It is for internal use only. Always check staged changes before committing to ensure that agents.md is not included. If you need to update it, do so locally and do not push those changes to the repository.

## Project Overview

**Linumpy** is the main library of the *Laboratoire d'Imagerie Numérique, Neurophotonique et Microscopie* (LINUM). It provides tools for serial histology and microscopy data: OCT acquisition, preprocessing, tile stitching, 3-D reconstruction, stacking, and analysis.

- **Repo**: github.com/linum-uqam/linumpy — default branch `main`, development on `dev`
- **Python**: 3.12+ (`.venv/bin/python` at repo root)
- **Key deps**: numpy <2.3, scipy <1.13, scikit-image, SimpleITK, JAX, Zarr 3.0+, OME-Zarr 0.9+, napari

## Repository Layout

| Path | Purpose |
|------|---------|
| `linumpy/` | Core library (import as `linumpy`) |
| `linumpy/preproc/` | Preprocessing: XYZ correction (`xyzcorr.py`), intensity correction (`icorr.py`), normalization |
| `linumpy/stitching/` | Tile stitching, motor-based stacking, registration, interpolation |
| `linumpy/microscope/` | OCT data loader (`oct.py`) |
| `linumpy/io/` | OME-Zarr, NPZ, Thorlabs metadata readers |
| `linumpy/gpu/` | Optional CUDA-accelerated operations (CuPy/JAX); falls back to CPU |
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
GPU scripts have `_gpu` suffixes. The `linumpy/gpu/` subpackage wraps CuPy/JAX; treat all CuPy/numba symbols as `Any` for type-checking purposes.

## Code Quality Checks

After every code change, run ruff and ty before considering the task done:

```bash
.venv/bin/ruff check <changed_file>
.venv/bin/ruff format --check <changed_file>
.venv/bin/ty check <changed_file>
```

Fix any errors reported before proceeding. For ruff, apply the suggested fix (e.g. collapse `if`/`else` into ternary when prompted by SIM108 etc.).

Ruff lint rule set: `E, F, W, I, UP, B, RUF, SIM, C4, PIE, PTH` — target Python 3.12, line length 127.
`ty` only type-checks `linumpy/` (not scripts); GPU duck-typed symbols are replaced with `Any`.

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

## Python Environment

Use `.venv/bin/python` (Python 3.12, virtual-env at repo root).
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

```bash
# 1. Commit and push locally
git add -A && git commit -m "description" && git push

# 2. Pull on server and reinstall
ssh 132.207.157.41 "export PATH=/home/frans/.local/bin:\$PATH; cd /home/frans/code/linumpy && git pull && uv pip install -e ."
```

Note: `uv` is at `/home/frans/.local/bin/uv` on the server and is not in the default non-interactive SSH PATH. Always export the PATH or use the full path.

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

Each subject folder under `~/Downloads/sub-XX/` has an `update_files.sh` script.  
It reads the remote `nextflow.config` first, then downloads only the outputs that match the enabled flags (common_space_previews, stack previews, diagnostics, zarr archive, etc.):
```bash
bash ~/Downloads/sub-18/update_files.sh
bash ~/Downloads/sub-22/update_files.sh
```

Key output locations pulled by those scripts:

| Remote path | What it contains |
|-------------|-----------------|
| `output/common_space_previews/` | 3-panel PNG per slice (XY/XZ/YZ views) |
| `output/stack/` | Stacked zarr, z_matches.csv, stacking_decisions.csv, metrics |
| `output/normalize_z_intensity/` | Z-intensity-normalized zarr + previews |
| `output/register_pairwise/` | Pairwise registration transforms |
| `output/align_to_ras/` | Atlas-aligned zarr + transform + preview |
| `output/detect_rehoming_events/` | Corrected shifts_xy_clean.csv + diagnostics |
| `mosaic-grids/previews/` | Per-tile mosaic preview PNGs |

## Reconstruction Pipeline: Diagnostics & Config Knowledge

### Diagnosing misalignment between stacked slices
1. Check `output/stack/stacking_decisions.csv` — look for slices where `transform_loaded=False` (gap in corrections)
2. Check `output/register_pairwise/` metrics — look for `mag` >100px or "optimizer boundary" warnings
3. Check raw motor shifts: `shifts_xy.csv` column `x_shift_mm` or `y_shift_mm`. Any step > 0.5mm is suspect.
4. Check if large steps are near multiples of `tile_fov_mm` (0.875mm for most subjects): `mag / 0.875`. Remainder < 10% → tile-column expansion event.

### tile_fov_mm and detect_rehoming
- **Always set `tile_fov_mm = 0.875`** for subjects acquired with the standard OCT tile grid. During acquisition, adding a new column of tiles produces a motor jump of exactly N × 0.875mm. Without correction, this mispositions all subsequent slices in common-space.
- **`detect_rehoming = true`** must accompany `tile_fov_mm`. Pass 1 corrects clean N×fov steps; Pass 2 corrects encoder glitch spikes. The output `shifts_xy_clean.csv` includes a `reliable` column: `reliable=0` means the original step magnitude exceeded `max_shift_mm` (the transition was unusual and may need image-based verification).
- To check if a step is a tile multiple: compute `x_shift_mm / 0.875` — if remainder < ~0.1, it's a tile step.

### common_space_refine settings
- `common_space_refine_unreliable = true` — enables image-based registration refinement for `reliable=0` transitions
- `common_space_refine_max_discrepancy_px = 0` — no limit on how far image estimate can differ from motor estimate (recommended; any limit risks rejecting good image-based corrections)
- Use these together for subjects with large or non-tile motor steps.

### Stack boundary filter
- `stack_max_pairwise_translation`: pairwise registrations with `mag > threshold * 0.95` are excluded from accumulation (their translation is zeroed, rotation kept)
- **Set to 0 to disable entirely.** Even a pairwise correction that hit the optimizer's 200px boundary is better than zero — the boundary value at least captures the direction of misalignment.
- Only use a nonzero value if specific slices have clearly erroneous translations that are worse than zero (very rare).

### Accumulation and drift
- `stack_max_cumulative_drift_px = 0` — disable drift cap (recommended). Accumulating pairwise corrections should converge if common_space is correct.
- `stack_translation_smooth_sigma = 3.0` — Gaussian smoothing of accumulated pairwise corrections (sigma in slice units). Reduces noise without removing large legitimate shifts.
- `load_transform_max_rotation = 4.0` — gate for loading rotation transforms. Default was 2.0°; raise to 4.0° for subjects with noisy pairwise rotations.

### Re-running after config changes
When server configs or code change, the pipeline must re-run from the first changed step:
- Code in `detect_rehoming` → re-run from `detect_rehoming` (cascades through common_space, register_pairwise, stack, normalize_z, align_to_ras)
- Config change to `stack_max_pairwise_translation` only → can re-run from `stack` only (use `-resume`)
- `-resume` caches all unchanged upstream processes automatically
