# Slice Configuration Feature


---

## Overview

The slice configuration feature allows users to control which slices are included in the 3D reconstruction pipeline. This is essential when:

- Some slices have quality issues (motion artifacts, bad cuts, etc.)
- You want to reconstruct only a subset of the data
- Debugging reconstruction issues with specific slices
- Reviewing which slices have galvo shift artifacts detected

---

## Problem Statement

### Previous Behavior

The reconstruction pipeline would process all slices found in the input directory. If some slices were problematic, users had to:

1. Manually delete or move problematic slice files
2. Edit the `shifts_xy.csv` file to match remaining slices
3. Risk errors from mismatched slice IDs between files and shifts

### The Bug

The `linum_align_mosaics_3d_from_shifts.py` script had a critical bug where slice IDs extracted from filenames were used as array indices:

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

A CSV file (`slice_config.csv`) controls which slices to include:

**Basic format:**
```csv
slice_id,use,notes
00,true,
01,true,
02,false,Motion artifact during cut
03,true,
04,true,
05,false,Bad image quality
```

**With galvo detection (when `detect_galvo = true`):**
```csv
slice_id,use,galvo_confidence,galvo_fix,notes
00,true,0.234,false,
01,true,0.891,true,
02,false,0.756,true,Motion artifact
03,true,0.512,false,
04,true,0.189,false,
05,true,0.923,true,
```

### File Format

| Column | Type | Description |
|--------|------|-------------|
| `slice_id` | string | Two-digit slice identifier (e.g., "00", "01") |
| `use` | boolean | `true`/`false` whether to include this slice |
| `galvo_confidence` | float | (Optional) Galvo detection confidence score (0-1) |
| `galvo_fix` | boolean | (Optional) Whether galvo fix would be applied |
| `notes` | string | Optional notes explaining exclusion |

### Boolean Values

The `use` column accepts:
- **True**: `true`, `1`, `yes`
- **False**: `false`, `0`, `no`

### Galvo Detection Columns

When `detect_galvo = true` in the preprocessing pipeline:
- **galvo_confidence**: Detection confidence (0.0 - 1.0)
  - High (≥0.6): Galvo artifact likely present
  - Low (<0.6): No clear artifact detected
- **galvo_fix**: Whether the fix would be applied during mosaic creation
  - `true`: Confidence ≥ threshold, fix applied
  - `false`: Confidence < threshold, fix skipped

---

## Quality Assessment

### Automatic Quality Detection

The `linum_assess_slice_quality.py` script (and GPU version `linum_assess_slice_quality_gpu.py`) can analyze mosaic grids to detect quality issues and update the slice configuration.

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

### Quality Assessment with Extended Format

When quality assessment is enabled, the config file includes additional columns:

```csv
slice_id,use,quality_score,ssim_mean,edge_score,variance_score,depth,exclude_reason
00,false,0.000,0.000,0.000,0.000,120,calibration_slice
01,true,0.756,0.891,0.612,0.534,85,
02,false,0.198,0.231,0.112,0.198,85,low_quality
03,true,0.812,0.923,0.701,0.645,85,
```

### Usage Examples

```bash
# Create new config with quality assessment (exclude first slice)
linum_assess_slice_quality.py /path/to/mosaics slice_config.csv

# Update existing config with quality info
linum_assess_slice_quality.py /path/to/mosaics slice_config.csv \
    --update_existing --existing_config existing_config.csv

# Automatically exclude low quality slices
linum_assess_slice_quality.py /path/to/mosaics slice_config.csv \
    --min_quality 0.3

# GPU-accelerated quality assessment
linum_assess_slice_quality_gpu.py /path/to/mosaics slice_config.csv

# Report only (don't write file)
linum_assess_slice_quality.py /path/to/mosaics slice_config.csv --report_only
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

### New Script: `linum_generate_slice_config.py`

Generates a slice configuration file from existing data:

```bash
# From mosaic grids directory
linum_generate_slice_config.py /path/to/mosaics slice_config.csv

# From raw tiles directory
linum_generate_slice_config.py /path/to/raw_tiles slice_config.csv --from_tiles

# From existing shifts file
linum_generate_slice_config.py /path/to/shifts_xy.csv slice_config.csv --from_shifts

# With pre-excluded slices
linum_generate_slice_config.py /path/to/shifts_xy.csv slice_config.csv --from_shifts --exclude 2 5

# With galvo detection (requires raw tiles)
linum_generate_slice_config.py /path/to/raw_tiles slice_config.csv --from_tiles --detect_galvo

# From shifts file with galvo detection
linum_generate_slice_config.py /path/to/shifts_xy.csv slice_config.csv --from_shifts \
    --detect_galvo --tiles_dir /path/to/raw_tiles

# With custom galvo threshold
linum_generate_slice_config.py /path/to/raw_tiles slice_config.csv --from_tiles \
    --detect_galvo --galvo_threshold 0.4
```

### Updated Script: `linum_align_mosaics_3d_from_shifts.py`

Fixed bugs and added slice config support:

```bash
# Without slice config (backward compatible)
linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv output

# With slice config
linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv output --slice_config slice_config.csv
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
    linum_generate_slice_config.py ${shifts_file} slice_config.csv --from_shifts ${galvo_opts}
    """
}
```

### Updated Reconstruction Pipeline

```groovy
// Parameter in nextflow.config
params.slice_config = ""  // Optional path to slice_config.csv

// Workflow logic
def slicesToUse = null
if (params.slice_config && params.slice_config != "") {
    slicesToUse = parseSliceConfig(params.slice_config)
    log.info "Slice config loaded: using ${slicesToUse.size()} slices"
}

// Filter input slices
inputSlices = channel
    .fromFilePairs(...)
    .filter { slice_id, _files ->
        if (slicesToUse != null) {
            return slicesToUse.contains(slice_id)
        }
        return true
    }
```

---

## Cumulative Shift Handling

### Problem

The shifts file contains pairwise shifts between consecutive slices:

```csv
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

# Reconstruction
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
linum_generate_slice_config.py /output/shifts_xy.csv slice_config.csv --from_shifts

# Edit to exclude slice 2
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

## Files Changed

| File | Changes |
|------|---------|
| `scripts/linum_generate_slice_config.py` | **NEW** - Generate slice config |
| `scripts/linum_align_mosaics_3d_from_shifts.py` | Fixed indexing bug, added `--slice_config` |
| `workflows/preproc/preproc_rawtiles.nf` | Added `generate_slice_config` process |
| `workflows/reconst_3d/soct_3d_reconst.nf` | Added slice filtering logic |
| `workflows/reconst_3d/nextflow.config` | Added `slice_config` parameter |

---

## Testing

```bash
# Test help
linum_generate_slice_config.py --help

# Test generation from shifts file
linum_generate_slice_config.py shifts_xy.csv test_config.csv --from_shifts

# Test with exclusions
linum_generate_slice_config.py shifts_xy.csv test_config.csv --from_shifts --exclude 1 2

# Test with galvo detection
linum_generate_slice_config.py /path/to/tiles test_config.csv --from_tiles --detect_galvo

# Test align script with config
linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv output --slice_config test_config.csv
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
