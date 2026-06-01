# Slice Configuration Feature


## Overview

`slice_config.csv` is the **single source of truth for per-slice pipeline decisions**. It controls which slices are included in the 3D reconstruction pipeline and records how every stage has acted on each slice, in a machine- and human-readable audit trail.

Typical uses:

- Exclude slices with quality issues (motion artifacts, bad cuts, calibration slices).
- Reconstruct only a subset of the data.
- Audit which slices had galvo shift corrections applied.
- Audit which slices were rehomed, auto-excluded, or interpolated.
- Debug reconstruction issues by inspecting per-slice flags and notes.

All reads and writes from Python code go through the shared `linumpy.io.slice_config` module (`read`, `write`, `stamp`, `stamp_many`, `merge_fragments`, `filter_slices_to_use`, `force_skip_slices`) so the schema stays consistent across scripts. **Raw numeric metrics no longer live in `slice_config.csv`** — only pipeline-relevant booleans, decisions, and short reasons. Full numeric diagnostics are written to the end-of-pipeline report and per-slice JSON files.

---

## Problem Statement

### Previous Behavior

The reconstruction pipeline would process all slices found in the input directory. If some slices were problematic, users had to:

1. Manually delete or move problematic slice files
2. Edit the `shifts_xy.csv` file to match remaining slices
3. Risk errors from mismatched slice IDs between files and shifts

### The Bug

The `linum-align-mosaics-3d-from-shifts` script had a critical bug where slice IDs extracted from filenames were used as array indices:

```python
# BUG: slice IDs used as array indices
slice_ids -= np.min(slice_ids)  # Normalize to start from 0
img, res = read_omezarr(mosaics_files[slice_ids[0]])  # Uses ID as index!
dx_cumsum[id]  # Assumes consecutive IDs
```

This caused errors when:
- Slices were non-consecutive (e.g., 0, 2, 5)
- The shifts file had more entries than slices being processed

---

## Solution

### Slice Configuration File

A CSV file (`slice_config.csv`) records per-slice pipeline decisions. Below is a fully-populated example after a complete run:

```text
slice_id,use,quality_score,galvo_confidence,galvo_fix,rehomed,rehoming_reliable,auto_excluded,auto_exclude_reason,interpolated,interpolation_failed,interpolation_method_used,interpolation_fallback_reason,notes
00,false,0.000,0.234,false,false,,false,,false,false,,,calibration_slice
01,true,0.812,0.891,true,false,,false,,false,false,,,
02,false,0.198,0.756,true,false,,false,,true,false,zmorph,,Motion artifact — interpolated
07,false,0.092,0.612,false,false,,false,,false,true,,low_overlap_ncc,zmorph hard-skipped — slot is a gap
03,true,0.756,0.512,false,true,true,false,,false,false,,,rehomed after tile-column expansion
04,true,0.744,0.189,false,false,,true,noisy cluster (3 consecutive),false,false,,,
05,true,0.834,0.923,true,false,,false,,false,false,,,
```

A minimal `slice_config.csv` (only required columns) still works — missing columns default to `false`/empty and are filled in as each pipeline stage stamps its flags.

### Canonical columns

The schema is defined by `linumpy.io.slice_config.CANONICAL_COLUMNS` and is enforced by every writer in the pipeline:

| Column | Type | Written by | Description |
|--------|------|------------|-------------|
| `slice_id` | string | generator | Two-digit slice identifier (e.g., "00", "01"). **Required key.** |
| `use` | boolean | generator / quality / galvo | Whether to include this slice. `false` → filtered out downstream. |
| `quality_score` | float | `linum_assess_slice_quality[_gpu].py` | Weighted quality score in [0, 1]. |
| `galvo_confidence` | float | `linum-generate-slice-config`, `linum-fix-galvo-shift-zarr` | Galvo detection confidence (0–1). |
| `galvo_fix` | boolean | `linum-generate-slice-config`, `linum-fix-galvo-shift-zarr` | Whether the galvo shift fix was applied. |
| `rehomed` | boolean | `linum-detect-rehoming` | `true` if the slice was corrected for a rehoming event (tile-column expansion or encoder glitch). |
| `rehoming_reliable` | boolean | `linum-detect-rehoming` | `true` if the motor-based correction was within `max_shift_mm`; `false` flags it for image-based verification. |
| `auto_excluded` | boolean | `linum-auto-exclude-slices` | `true` if the slice was automatically excluded by the low-quality cluster detector. |
| `auto_exclude_reason` | string | `linum-auto-exclude-slices` | Short human-readable reason (e.g. `noisy cluster (3 consecutive)`). |
| `interpolated` | boolean | `linum-interpolate-missing-slice --finalise` | `true` if this slice was successfully interpolated from its neighbours (a zarr was produced). |
| `interpolation_failed` | boolean | `linum-interpolate-missing-slice --finalise` | `true` if zmorph was attempted but hit a quality gate; no zarr was produced and the slot is a gap in the final volume. |
| `interpolation_method_used` | string | `linum-interpolate-missing-slice --finalise` | Method actually used (`zmorph`, `weighted`, `average`). Empty when `interpolation_failed=true`. |
| `interpolation_fallback_reason` | string | `linum-interpolate-missing-slice --finalise` | Reason zmorph hard-skipped (`low_overlap_ncc`, `reg_did_not_improve`, ...), or empty on success. |
| `notes` | string | any stage | Free-form human-readable annotation. Stages append with `; ` separators. |

> **Raw numeric metrics** (SSIM, edge score, variance ratio, NCC values, affine determinants, ...) are deliberately **not** in `slice_config.csv`. They live in per-slice JSON diagnostics and in the end-of-pipeline quality report.

### Boolean values

The `use` column (and all other booleans) accepts:
- **True**: `true`, `1`, `yes`
- **False**: `false`, `0`, `no`

All booleans are **written back as lowercase `true`/`false`** to stay consistent with Nextflow-style CSVs.

### Galvo detection columns

When `detect_galvo = true` in the preprocessing pipeline:
- **galvo_confidence**: Detection confidence (0.0 – 1.0)
  - High (≥0.6): Galvo artifact likely present
  - Low (<0.6): No clear artifact detected
- **galvo_fix**: Whether the fix would be applied during mosaic creation
  - `true`: Confidence ≥ threshold, fix applied
  - `false`: Confidence < threshold, fix skipped

### Rehoming columns

When `detect_rehoming = true` in the 3D reconstruction pipeline, `linum-detect-rehoming` stamps:
- **rehomed** (`true`/`false`) — the slice transition had an N×`tile_fov_mm` step or an encoder glitch that was corrected.
- **rehoming_reliable** (`true`/`false`) — whether the motor-based fix was within `max_shift_mm`. `false` means the downstream `common_space_refine_unreliable` stage should verify the correction against image data.

### Auto-exclusion columns

`linum-auto-exclude-slices` reads pairwise registration metrics and stamps:
- **auto_excluded** (`true`/`false`) — slice is in a noisy cluster.
- **auto_exclude_reason** (string) — short human-readable reason.

This replaces the old `auto_exclude.csv` side-file.

### Interpolation columns

After `finalise_interpolation`, any slice that reached `linum-interpolate-missing-slice` gets one of two states:

- **Success** — a reconstructed zarr was produced:
  - `interpolated=true`, `interpolation_failed=false`
  - `interpolation_method_used` = `zmorph` / `weighted` / `average`
  - `interpolation_fallback_reason` empty

- **Hard skip** — zmorph hit a quality gate, no zarr was produced, the slot stays a gap:
  - `interpolated=false`, `interpolation_failed=true`
  - `interpolation_method_used` empty
  - `interpolation_fallback_reason` = one of `low_overlap_ncc`, `no_foreground_planes`, `registration_exception`, `reg_did_not_improve`, `affine_determinant_non_positive`

The pipeline never fabricates a slice from a weighted blend when registration fails; blending two neighbours that could not be registered introduces ghost contours and would also be made-up data. See {doc}`SLICE_INTERPOLATION_FEATURE` for the interpolation algorithm details and rationale.

---

## Quality Assessment

### Automatic Quality Detection

The `linum-assess-slice-quality` script can analyze mosaic grids to detect quality issues and update the slice configuration. GPU acceleration is enabled by default (pass `--no-use_gpu` to disable).

**Quality Metrics:**
| Metric | Weight | Description |
|--------|--------|-------------|
| **SSIM** | 50% | Structural Similarity with neighboring slices |
| **Edge Preservation** | 30% | Correlation of edge maps with expected structure |
| **Variance Ratio** | 20% | Consistency of signal variance |

**Quality Score Formula:**
```
Quality Score = 0.5 × SSIM + 0.3 × EdgeScore + 0.2 × VarianceScore
```

### Calibration Slice Detection

The first slice in an acquisition is typically a **calibration slice** that is thicker than the others and should be excluded. The scripts support two methods:

1. **Automatic exclusion**: Use `--exclude_first N` to exclude the first N slices (default: 1)
2. **Thickness detection**: Use `--detect_calibration` to automatically detect slices that are significantly thicker than the median

### Quality Assessment Output

When quality assessment runs, the writer only populates canonical columns:

```text
slice_id,use,quality_score,notes
00,false,0.000,calibration_slice
01,true,0.812,
02,false,0.198,low_quality (quality_score<0.3)
03,true,0.923,
```

Raw per-metric numbers (SSIM, edge, variance, tissue depth) are available in the pipeline report and in the per-slice JSON diagnostics — they are *not* duplicated in `slice_config.csv`.

### Usage Examples

```bash
# Create new config with quality assessment (exclude first slice)
linum-assess-slice-quality /path/to/mosaics slice_config.csv

# Update existing config with quality info
linum-assess-slice-quality /path/to/mosaics slice_config.csv \
    --update_existing --existing_config existing_config.csv

# Automatically exclude low quality slices
linum-assess-slice-quality /path/to/mosaics slice_config.csv \
    --min_quality 0.3

# GPU-accelerated quality assessment (default; pass --no-use_gpu to disable)
linum-assess-slice-quality /path/to/mosaics slice_config.csv --use_gpu

# Report only (don't write file)
linum-assess-slice-quality /path/to/mosaics slice_config.csv --report_only
```

---

## Galvo Shift Detection Algorithm

### Background: The Galvo Artifact Problem

In serial OCT (SOCT) imaging, the galvo mirror physically sweeps across the sample during acquisition. At the end of each sweep, the mirror must return to its starting position. During this "galvo return" period, the system continues acquiring data, but this data represents the mirror's return path rather than the sample.

**The artifact occurs when:**
- The galvo return region is **not** at the edge of the raw tile data
- Instead, it appears somewhere in the middle of the A-line data
- This creates a visible intensity discontinuity (banding) in stitched images

**The fix:**
- Apply a circular shift to move the galvo return region to the edge
- The shift amount is determined by detecting where the intensity discontinuity occurs

### Detection Algorithm Overview

The detection happens in two stages:

1. **Shift Detection** (`detect_galvo_shift`): Find *where* the galvo return boundary is located
2. **Slice-level Detection** (`detect_galvo_for_slice`): Sample multiple tiles to find best detection

### Tile Sampling Strategy

When detecting galvo artifacts for a slice, the algorithm samples multiple tiles because:
- Background/empty tiles at the edges of the mosaic have low intensity
- Low-intensity tiles produce unreliable detection (low confidence)
- Tiles with actual tissue content give more reliable results

**Key parameters:**
- `n_samples`: Number of tiles to sample (default: 5)
- `min_intensity`: Minimum mean intensity threshold (default: 20.0)
- Tiles below `min_intensity` are skipped as background

The algorithm samples up to 3× more candidate tiles than `n_samples` to account for empty tiles, ensuring enough high-quality tiles are tested.

### Stage 1: Shift Detection

```python
def detect_galvo_shift(aip, n_pixel_return):
    # Compute average A-line profile from the AIP (Average Intensity Projection)
    profile = aip.mean(axis=1)
    
    # For each possible shift, compute intensity differences
    # between start and end of the shifted profile
    # The correct shift makes both boundaries of the galvo return 
    # show high differences (since they transition to/from image data)
    
    similarities = []
    for s in range(len(profile) - n_pixel_return):
        # Product of differences at both boundaries
        foo = differences[s] * differences[s + n_pixel_return]
        similarities.append(foo)
    
    # The shift with maximum similarity product is the correct one
    shift = argmax(similarities)
```

### Stage 2: Slice-Level Detection (`detect_galvo_for_slice`)

This function samples multiple tiles from a slice to find the most reliable galvo shift detection.

**Why sample multiple tiles?**
- Edge tiles in a mosaic often contain mostly background (air/mounting medium)
- Background tiles have low intensity and produce unreliable detection
- Tiles with actual tissue content give consistent, high-confidence results

**Sampling strategy:**
```python
def detect_galvo_for_slice(tiles, n_extra, threshold=0.6, n_samples=5,
                           axial_resolution=None, min_intensity=20.0):
    # Sample 3x more candidates to account for empty tiles
    n_candidates = n_samples * 3
    
    for idx in candidate_indices:
        tile = tiles[idx]
        
        # Skip low-intensity (background) tiles
        if tile.mean() < min_intensity:
            continue
        
        # Detect shift and confidence for this tile
        shift, confidence = detect_galvo_shift(tile_aip, n_extra)
        
        # Keep track of valid detections
        if confidence >= threshold:
            valid_detections.append((shift, confidence))
    
    # Return the detection with highest confidence
    return best_shift, best_confidence
```

**Key parameters:**
- `n_samples=5`: Number of valid (high-intensity) tiles to test
- `min_intensity=20.0`: Skip tiles with mean intensity below this value
- `threshold=0.6`: Minimum confidence required to consider detection valid

### Stage 3: Artifact Presence Detection

The key insight is that **not all slices have the galvo artifact**. Even when `fix_galvo_shift = true` in the pipeline, we should only apply the fix to slices that actually need it.

**Detection Method: Intensity Discontinuity Analysis**

The algorithm analyzes the raw tile AIP (Average Intensity Projection) looking for telltale signs of a misplaced galvo return region:

```
Image Region | Galvo Return | Image Region
-------------|--------------|-------------
  normal     |   different  |    normal
  intensity  |   intensity  |   intensity
```

**Three metrics are combined:**

1. **Boundary Contrast (50% weight)**
   - Compare intensity of the suspected return region vs. surrounding image regions
   - The galvo return can be **brighter OR darker** than the image (depends on sample)
   - Algorithm uses **absolute difference** to catch both cases
   - A 15%+ relative intensity difference → high score

2. **Edge Sharpness (30% weight)**
   - Sharp intensity transitions at the return region boundaries
   - Compare edge strength at boundaries vs. typical gradient in the image
   - Strong edges at expected locations → higher confidence

3. **Return Region Anomaly (20% weight)**
   - How statistically different is the return region intensity?
   - Uses z-score: `|image_intensity - return_intensity| / std`
   - z-score > 1.5 indicates significant anomaly

**Final Score Calculation:**
```python
artifact_score = (
    contrast_score * 0.50 +
    sharpness_score * 0.30 +
    anomaly_score * 0.20
)
```

### Score Interpretation

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 0.0 - 0.2 | No clear discontinuity | Skip correction |
| 0.4 - 0.5 | Borderline, likely clean | Skip correction |
| 0.5 - 0.6 | Mild discontinuity | Consider correction |
| 0.6 - 1.0 | Clear discontinuity | Apply correction |

**Default threshold:** 0.6 (configurable via `galvo_confidence_threshold`)

### Why Absolute Difference Matters

Early versions of the algorithm assumed the galvo return region would always be **darker** than the image. However, in practice:

- Galvo return can be **brighter** if surrounding tissue is dark
- Galvo return can be **darker** if surrounding tissue is bright
- The key feature is the **discontinuity**, not the direction

Using absolute difference ensures both cases are detected:
```python
# CORRECT: Detects both brighter and darker return regions
relative_diff = abs(image_intensity - return_intensity) / image_intensity
```

### Raw Tiles vs. Stitched Images

**Important:** The galvo artifact appears differently in:

| View | Appearance |
|------|------------|
| **Raw tile AIP** | Intensity band at galvo return position |
| **Stitched mosaic** | Horizontal banding across the full image |

The detection runs on **raw tiles** (single tile per slice) because:
- More memory efficient (no need to load full mosaic)
- Galvo shift is consistent within a slice
- Direct access to the underlying data structure

---

## Implementation

### New Script: `linum-generate-slice-config`

Generates a slice configuration file from existing data:

```bash
# From mosaic grids directory
linum-generate-slice-config /path/to/mosaics slice_config.csv

# From raw tiles directory
linum-generate-slice-config /path/to/raw_tiles slice_config.csv --from_tiles

# From existing shifts file
linum-generate-slice-config /path/to/shifts_xy.csv slice_config.csv --from_shifts

# With pre-excluded slices
linum-generate-slice-config /path/to/shifts_xy.csv slice_config.csv --from_shifts --exclude 2 5

# With galvo detection (requires raw tiles)
linum-generate-slice-config /path/to/raw_tiles slice_config.csv --from_tiles --detect_galvo

# From shifts file with galvo detection
linum-generate-slice-config /path/to/shifts_xy.csv slice_config.csv --from_shifts \
    --detect_galvo --tiles_dir /path/to/raw_tiles

# With custom galvo threshold
linum-generate-slice-config /path/to/raw_tiles slice_config.csv --from_tiles \
    --detect_galvo --galvo_threshold 0.4
```

### Updated Script: `linum-align-mosaics-3d-from-shifts`

Fixed bugs and added slice config support:

```bash
# Without slice config (backward compatible)
linum-align-mosaics-3d-from-shifts inputs shifts_xy.csv output

# With slice config
linum-align-mosaics-3d-from-shifts inputs shifts_xy.csv output --slice_config slice_config.csv
```

**Key Improvements:**
- Matches slice IDs from filenames to shifts file by ID (not array index)
- Properly accumulates shifts when intermediate slices are skipped
- Validates slices against shifts file
- Prints informative messages about processing

### Updated Preprocessing Pipeline

New parameters and process:

```groovy
// Parameters in nextflow.config
params.generate_slice_config = true  // Generate slice_config.csv
params.detect_galvo = false          // Include galvo detection in slice_config
params.galvo_confidence_threshold = 0.6  // Threshold for galvo fix

// Process
process generate_slice_config {
    publishDir "$params.output", mode: 'copy'
    
    input:
        tuple path(shifts_file), path(input_dir)
    
    output:
        path("slice_config.csv")
    
    script:
    String galvo_opts = params.detect_galvo ? 
        "--detect_galvo --tiles_dir ${input_dir} --galvo_threshold ${params.galvo_confidence_threshold}" : ""
    """
    linum-generate-slice-config ${shifts_file} slice_config.csv --from_shifts ${galvo_opts}
    """
}
```

### Updated Reconstruction Pipeline

```groovy
// Parameter in nextflow.config
params.slice_config = ""        // Optional path to slice_config.csv
params.auto_assess_quality = false  // If true, bootstrap a slice_config.csv from quality assessment

// Workflow logic (simplified)
def slice_config_path = params.slice_config ?: "${params.output}/slice_config.csv"
def has_slice_config = file(slice_config_path).exists() || params.auto_assess_quality

// Filter slices using slice_config.csv (python-side; the workflow just passes the path through).
// Each stage that has something to stamp takes the current slice_config as input and
// emits an updated one:
current_slice_config = Channel.fromPath(slice_config_path)
    | detect_rehoming_events          // stamps rehomed / rehoming_reliable
    | finalise_interpolation          // stamps interpolated / interpolation_method_used / ...
    | auto_exclude_slices             // stamps auto_excluded / auto_exclude_reason
    // ... the final slice_config.csv is what `stack` reads via --slice_config
```

All Python-side slice filtering (e.g. `linum-estimate-global-transform`) uses `linumpy.io.slice_config.filter_slices_to_use()` — there is no Groovy `parseSliceConfig` helper to maintain.

---

## Cumulative Shift Handling

### Problem

The shifts file contains pairwise shifts between consecutive slices:

```text
fixed_id,moving_id,x_shift,y_shift,x_shift_mm,y_shift_mm
0,1,10,5,0.01,0.005
1,2,8,3,0.008,0.003
2,3,12,7,0.012,0.007
```

If slice 2 is excluded, the shift from slice 1 to slice 3 must be the **sum** of shifts 1→2 and 2→3.

### Solution

The `build_cumulative_shifts()` function:

1. Builds cumulative shifts for **all** slices in the shifts file
2. Extracts only the values for selected slices

```python
def build_cumulative_shifts(shifts_df, selected_slice_ids, resolution):
    # Build cumulative for ALL slices first
    cumsum_all = {all_slice_ids[0]: (0.0, 0.0)}
    for i in range(len(all_slice_ids) - 1):
        fixed_id = all_slice_ids[i]
        moving_id = all_slice_ids[i + 1]
        dx_mm, dy_mm = shift_lookup[(fixed_id, moving_id)]
        prev_dx, prev_dy = cumsum_all[fixed_id]
        cumsum_all[moving_id] = (prev_dx + dx_mm, prev_dy + dy_mm)
    
    # Extract only for selected slices
    cumsum_selected = {}
    for slice_id in selected_slice_ids:
        cumsum_selected[slice_id] = cumsum_all[slice_id]
    
    return cumsum_selected
```

### Example

Original slices: 0, 1, 2, 3, 4, 5  
Shifts: 0→1: (10, 5), 1→2: (8, 3), 2→3: (12, 7), 3→4: (5, 2), 4→5: (9, 4)

Cumulative shifts for all slices:
- Slice 0: (0, 0)
- Slice 1: (10, 5)
- Slice 2: (18, 8)
- Slice 3: (30, 15)
- Slice 4: (35, 17)
- Slice 5: (44, 21)

If processing only slices 0, 2, 4:
- Slice 0: (0, 0)
- Slice 2: (18, 8) ← Correct! Sum of 0→1 + 1→2
- Slice 4: (35, 17) ← Correct! Sum of all up to slice 4

---

## Usage Workflow

### For New Datasets

1. Run preprocessing pipeline (generates `slice_config.csv` automatically)
2. Review and edit `slice_config.csv` if needed
3. Run reconstruction pipeline

```bash
# Preprocessing (generates slice_config.csv)
nextflow run preproc_rawtiles.nf --input /raw/data --output /output

# Review slice config
cat /output/slice_config.csv

# Edit if needed (set use=false for bad slices)
nano /output/slice_config.csv

# Reconstruction (the pipeline updates slice_config.csv in place as it runs,
# publishing the final version as slice_config_final.csv)
nextflow run soct_3d_reconst.nf \
    --input /output \
    --slice_config /output/slice_config.csv
```

### For Existing Datasets

1. Generate slice config from existing files
2. Edit to exclude problematic slices
3. Run reconstruction

```bash
# Generate config from existing shifts file
linum-generate-slice-config /output/shifts_xy.csv slice_config.csv --from_shifts

# Edit to exclude slice 2 (only canonical columns are needed)
# slice_id,use,notes
# 00,true,
# 01,true,
# 02,false,Bad quality
# 03,true,

# Reconstruction with slice config
nextflow run soct_3d_reconst.nf \
    --input /output \
    --slice_config slice_config.csv
```

### Inspecting the final audit trail

After a run, `${params.output}/slice_config_final.csv` contains the full per-slice trail: quality, galvo, rehoming, auto-exclusion, and interpolation flags. Combined with the pipeline report and per-slice JSON diagnostics it tells the complete story of what happened to each slice.

---

## Backward Compatibility

- **Slice config is optional**: If not provided, all slices are processed
- **Preprocessing parameter**: Set `generate_slice_config = false` to skip generation
- **Existing pipelines**: Continue to work without modification

---

## Validation

The pipeline performs validation:

1. **Slice config parse errors**: Reports malformed CSV files
2. **Missing slices in shifts**: Warns if selected slices aren't in shifts file
3. **Empty selection**: Errors if no slices remain after filtering
4. **Logging**: Prints which slices are included/excluded

---

## Python API: `linumpy.io.slice_config`

All code that reads or writes `slice_config.csv` goes through this module. This guarantees schema consistency and makes scripts trivial to update when the schema evolves.

```python
from linumpy.io import slice_config as slice_config_io

# Read (returns OrderedDict[slice_id -> row dict], keys preserved in file order).
rows = slice_config_io.read("slice_config.csv")

# Stamp a single slice: adds/overrides columns, extends canonical schema if needed.
rows = slice_config_io.stamp(rows, "03", {"interpolated": True, "interpolation_method_used": "zmorph"})

# Stamp many slices at once.
rows = slice_config_io.stamp_many(rows, {
    "03": {"rehomed": True, "rehoming_reliable": False},
    "07": {"rehomed": True, "rehoming_reliable": True},
})

# Merge a directory of per-slice manifest fragments (used by finalise_interpolation).
rows = slice_config_io.merge_fragments(rows, "interpolate_missing_slice/")

# Filter down to slices with use=true (respects auto_excluded via force_skip_slices).
to_process = slice_config_io.filter_slices_to_use(rows)

# Or, mark all auto_excluded+use=false slices as "skip" for stacking.
rows_for_stack = slice_config_io.force_skip_slices(rows)

# Write back: booleans serialised as lowercase true/false; canonical columns first.
slice_config_io.write("slice_config_next.csv", rows)
```

Key design rules:

- **Canonical columns come first** in the output, in the order defined by `CANONICAL_COLUMNS`. Extra columns from an older file are preserved at the end (but new code should not rely on extras).
- **Stamping is additive**: existing values are overwritten, but other columns are left alone. This lets the CSV accumulate information as it flows through the pipeline.
- **Appending notes**: stamping the `notes` column with the `append_notes=True` flag joins the new note with the existing one using `"; "` so earlier annotations survive.
- **No raw metrics**: writers are expected not to add columns like `ssim_mean` or `affine_determinant` — those belong in reports/diagnostics.

---

## Pipeline flow: `slice_config.csv` as a flowing artifact

In the 3D reconstruction pipeline (`workflows/reconst_3d/soct_3d_reconst.nf`), a single `slice_config.csv` channel flows through each stage that has something to stamp. Each stage takes the *current* `slice_config.csv` as input and emits an updated one:

```
[preproc slice_config.csv or auto-generated]
        │
        ▼
detect_rehoming_events        ──►  slice_config.csv (+rehomed, +rehoming_reliable)
        │
        ▼
[interpolate_missing_slice produces per-slice manifest fragments]
        │
        ▼
finalise_interpolation        ──►  slice_config_final.csv
                                   (+interpolated, +interpolation_method_used,
                                    +interpolation_fallback_reason)
        │
        ▼
auto_exclude_slices           ──►  slice_config.csv (+auto_excluded, +auto_exclude_reason)
        │
        ▼
stack (linum-stack-slices-motor --slice_config)
        │   — reads the final slice_config.csv, uses force_skip_slices()
        ▼
downstream stages (normalize_z, align_to_ras, generate_report)
        │   — read the same slice_config_final.csv
        ▼
[pipeline report shows the complete audit trail per slice]
```

There is no separate `auto_exclude.csv`, no separate interpolation manifest CSV, and no script dedicated to merging manifests into the config. Each stage stamps its own flags in place.

---

## Concurrency

The pipeline runs many processes in parallel, but `slice_config.csv` is **never written to concurrently**. The design is race-free by construction, not by locking:

1. **Single writer per stage.** Every Nextflow process that updates `slice_config.csv` takes the current version as an *input path* (staged into its own work directory) and emits a *new* CSV as an output. Nextflow copies outputs into the next consumer's work directory; nothing reads and writes the same file at the same time.

2. **Per-slice fan-out uses fragments, not shared writes.** Per-slice stages (interpolation, pairwise registration, ...) are the only ones that fan out. They each emit a small per-slice fragment (e.g. `slice_z{NN}_manifest.csv`) to their own work directory. Fragment filenames are unique per slice, so two parallel tasks cannot collide.

3. **Sequential consolidation.** Fragments are aggregated via `.collect()` and consumed by a single downstream process (e.g. `finalise_interpolation`) that runs one instance, calls `linumpy.io.slice_config.merge_fragments`, and writes a new CSV. All updates to `slice_config.csv` therefore funnel through a single writer.

4. **No in-place updates.** `linumpy.io.slice_config.stamp` / `merge_fragments` always write to a *new* `slice_config_out` path. A consumer reading the old version never observes a half-written file. This also keeps Nextflow's `-resume` cache semantics working — inputs to each stage are content-addressed and immutable.

5. **No file locking in `linumpy.io.slice_config`.** If you invoke these helpers from ad-hoc scripts outside of Nextflow, make sure each writer targets a distinct output path. The module relies on the pipeline's channel discipline for mutual exclusion.

See `linumpy/io/slice_config.py` for the concurrency contract in the module docstring.

---

## Files Changed

| File | Changes |
|------|---------|
| `linumpy/io/slice_config.py` | **NEW** — canonical schema + `read`/`write`/`stamp`/`stamp_many`/`merge_fragments`/`filter_slices_to_use`/`force_skip_slices`. |
| `scripts/analysis/linum_generate_slice_config.py` | Uses `linumpy.io.slice_config` for writes; canonical columns only. |
| `scripts/analysis/linum_assess_slice_quality[_gpu].py` | Refactored to use `linumpy.io.slice_config`; dropped `ssim_mean`/`edge_score`/`variance_score`/`depth` columns. |
| `scripts/analysis/linum_detect_rehoming.py` | Added `--slice_config_in`/`--slice_config_out`; stamps `rehomed` + `rehoming_reliable`. |
| `scripts/analysis/linum_auto_exclude_slices.py` | Stamps `auto_excluded` + `auto_exclude_reason` directly on `slice_config.csv` (no more side-file). |
| `scripts/stacking/linum_interpolate_missing_slice.py` | Added `--finalise` mode: merges per-slice manifest fragments into `slice_config.csv`. |
| `scripts/stacking/linum_stack_slices_motor.py` | Accepts `--slice_config`; uses `slice_config_io.force_skip_slices()`. `--force_skip_slices` removed. |
| `scripts/stitching/linum_align_mosaics_3d_from_shifts.py` | Fixed indexing bug, added `--slice_config`, now uses shared reader. |
| `scripts/stitching/linum_estimate_global_transform[_gpu].py`, `linum-analyze-stitch-affine`, `linum-fix-galvo-shift-zarr` | Switched to shared `linumpy.io.slice_config` reader/writer. |
| `scripts/linum_update_slice_config_with_interpolation.py` | **REMOVED** — replaced by `linum-interpolate-missing-slice --finalise`. |
| `workflows/preproc/preproc_rawtiles.nf` | Adds `generate_slice_config` process. |
| `workflows/reconst_3d/soct_3d_reconst.nf` | Threads `slice_config.csv` through `detect_rehoming_events` → `finalise_interpolation` → `auto_exclude_slices` → `stack`. |
| `workflows/reconst_3d/nextflow.config` | `slice_config` parameter; `auto_assess_quality` can bootstrap one. |

---

## Testing

```bash
# Test help
linum-generate-slice-config --help

# Test generation from shifts file
linum-generate-slice-config shifts_xy.csv test_config.csv --from_shifts

# Test with exclusions
linum-generate-slice-config shifts_xy.csv test_config.csv --from_shifts --exclude 1 2

# Test with galvo detection
linum-generate-slice-config /path/to/tiles test_config.csv --from_tiles --detect_galvo

# Test align script with config
linum-align-mosaics-3d-from-shifts inputs shifts_xy.csv output --slice_config test_config.csv
```

---

## Troubleshooting Galvo Detection

### Problem: Galvo fix not applied despite `galvo_fix=true`

**Symptoms:**
- `slice_config.csv` shows `galvo_fix=true` with confidence > 0.5
- Log shows "No galvo fix needed for slice" or shift=0
- Banding artifacts still visible in mosaic

**Cause:**
The detection during mosaic creation re-samples tiles and may sample different tiles than the original detection. If it samples background tiles (low intensity), it gets low confidence and returns shift=0.

**Solutions:**
1. The detection skips tiles with mean intensity < 20.0
2. Detection samples 3× more candidate tiles to find enough high-intensity ones

### Problem: Low confidence scores for all tiles

**Symptoms:**
- All tiles return confidence < 0.6
- `galvo_fix` set to `false` for all slices

**Possible causes:**
1. Tiles genuinely don't have galvo artifact (correct behavior)
2. Very homogeneous tissue with weak intensity variation
3. `n_extra` parameter doesn't match actual galvo return size

**Debugging:**
```python
from linumpy.preproc.xyzcorr import detect_galvo_for_slice

# Test with verbose output
shift, conf = detect_galvo_for_slice(
    tiles, n_extra=40, threshold=0.6, n_samples=5, min_intensity=20.0
)
print(f"shift={shift}, confidence={conf}")
```
