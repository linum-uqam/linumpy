# Slice Interpolation Feature

## Overview

The slice interpolation feature reconstructs missing slices in Serial OCT
datasets using information from adjacent slices. It is driven by
`slice_config.csv`: any slice with `use=false` is filtered out of the upstream
pipeline, producing a gap that the interpolator fills automatically. After
interpolation, the per-slice diagnostics are merged back into
`slice_config_final.csv` so the record of *what happened to each slice*
survives to the final quality report.

**Key features**:

- Primary reconstruction method: `zmorph` — z-aware morphing driven by the
  physical serial-section geometry.
- Automatic NCC-based boundary plane selection with foreground filtering and
  a minimum-correlation gate.
- Fractional affine transforms (`T**alpha`) computed via
  `scipy.linalg.fractional_matrix_power`, with a post-registration NCC
  improvement gate.
- **Hard-skip on failure: no fabricated data.** When any quality gate fails,
  zmorph produces *no* interpolated volume. The slot stays a genuine gap in
  the final reconstruction rather than being filled with a blended and
  therefore fabricated slice.
- Per-slice JSON diagnostics and per-slice manifest fragments are merged
  directly into `slice_config.csv` — the single source of truth for per-slice
  decisions, see {doc}`SLICE_CONFIG_FEATURE`.
  Successful interpolations stamp `interpolated=true`; failures stamp
  `interpolation_failed=true` plus the specific `fallback_reason`.
- Downstream propagation: `linum_register_pairwise.py` automatically marks
  any transform touching an interpolated slice as `reliable=0` so stacking
  can down-weight it. For hard-skipped slices there is no zarr at all, so
  pairwise simply bridges the two surviving neighbours directly.

**Important limitation**: interpolation only works for **single missing
slices**. When two or more consecutive slices are missing there is not enough
information to resolve the in-plane deformation across multiple cuts, and the
pipeline logs a warning instead of producing a fabricated volume.

---

## Background

### The problem

In Serial Optical Coherence Tomography (SOCT), as described in Lefebvre et
al. (2017)[^1], mouse brains are imaged after each cut through serial
sectioning (~200 µm slice interval). Occasionally a physical slice is:

- damaged during cutting,
- lost during handling, or
- unusable due to imaging artefacts.

When a single slice is missing, the gap can be filled by using the two
physically adjacent surfaces — the deepest imaged plane of the slice
*before* the cut and the shallowest imaged plane of the slice *after* — as
constraints.

### Driving the interpolator from `slice_config.csv`

The preprocessing pipeline generates an initial `slice_config.csv`
(see {doc}`SLICE_CONFIG_FEATURE`). The user
(or the automated quality assessment) may mark additional slices with
`use=false`. In the reconstruction pipeline those slices are filtered out
*before* common-space alignment, which leaves a gap in the numeric slice ID
sequence. `detectSingleGaps` in `soct_3d_reconst.nf` inspects the
post-filter list:

- `next_id - current_id == 2` → single-slice gap → interpolation job.
- `next_id - current_id > 2`  → multi-slice gap → warning, no interpolation.

So: **to request interpolation of slice `N`, just mark `N` as `use=false`**
in `slice_config.csv`. No separate flag is needed.

---

## Methods

The only scientifically motivated reconstruction method is `zmorph`.
`average` and `weighted` are simpler baselines kept for comparison and as
fallbacks; they do not use any 2D registration.

### Decision flow at a glance

```mermaid
flowchart TD
    START([Missing slice between<br/>vol_before and vol_after]) --> PLANES[find_best_overlap_planes<br/>foreground filter + NCC search]
    PLANES -->|no foreground planes| F1[fallback_reason:<br/>no_foreground_planes]
    PLANES -->|best NCC < min_overlap_correlation| F2[fallback_reason:<br/>low_overlap_ncc]
    PLANES -->|good pair| REG[2D ITK registration<br/>boundary plane → reference]
    REG -->|optimiser raised| F3[fallback_reason:<br/>registration_exception]
    REG -->|det T ≤ 0| F4[fallback_reason:<br/>affine_determinant_non_positive]
    REG -->|post-reg NCC ↑ < threshold| F5[fallback_reason:<br/>reg_did_not_improve]
    REG -->|gates pass| WARP[For each output plane at α = z / (nz_out-1):<br/>warp vol_before by T^α<br/>warp vol_after by T^(α-1)<br/>gaussian-feathered cross-fade]
    WARP --> OUT([Interpolated zarr<br/>+ manifest + diagnostics])
    F1 --> SKIP([Hard skip:<br/>no zarr written,<br/>genuine gap in stack])
    F2 --> SKIP
    F3 --> SKIP
    F4 --> SKIP
    F5 --> SKIP
```

### `zmorph` — z-aware morphing (default)

#### Physical model

In serial-block-face SOCT each slice is imaged *after* cutting. For a missing
slice between `vol_before` (slice `N-1`) and `vol_after` (slice `N+1`):

- `vol_before[-1]` = tissue surface exposed right **before** the missing
  slice was cut away.
- `vol_after[0]`   = tissue surface exposed right **after** the missing slice
  was cut away.

These two surfaces are separated only by the missing ~200 µm block and are
the only directly observable evidence of its content. The affine transform
`T` that maps one onto the other encodes the XY deformation (shift, rotation,
shear, mild scaling) accumulated across the missing block during cutting.

#### Algorithm

Using the Log-Euclidean one-parameter subgroup of the affine group[^2],
`T**alpha = exp(alpha · log T)` interpolates continuously between identity
(`alpha=0`) and `T` (`alpha=1`). We implement it via
`scipy.linalg.fractional_matrix_power`, which is numerically robust for
invertible affines and for matrices close to reflections (it returns a
complex matrix with negligible imaginary part in the well-conditioned case;
zmorph checks this and reports the max imaginary part in its diagnostics).

For each output plane at fractional depth
`alpha = z / (nz_out - 1) ∈ [0, 1]`:

| Step | Contribution |
|------|--------------|
| 1  | Register `vol_after[0]` to `vol_before[-1]` (slab-averaged for robustness) to obtain `T`. |
| 2a | Warp `vol_before[-1]` by `T**alpha` — identity at the top, full forward warp at the bottom. |
| 2b | Warp `vol_after[0]` by `T**(alpha - 1)` — full inverse warp at the top, identity at the bottom. |
| 2c | Cross-fade with weight `alpha` on *after* and `1 - alpha` on *before* (gaussian feathered in XY, per-plane z-weighted). |

Boundary conditions are **exact by construction**: the output's top plane
equals `vol_before[-1]` and its bottom plane equals `vol_after[0]` (up to
resampling error). Downstream pairwise registration therefore runs on real
tissue rather than on a blurred average.

#### Why zmorph is scientifically preferred

zmorph respects four constraints that matter for quantitative use:

1. **Exact boundary matching.** The top and bottom of the interpolated
   slice are the two physically adjacent imaged surfaces, not averaged
   features.
2. **Constant-rate deformation along Z.** Cutting deforms the block at a
   roughly constant rate between two consecutive cuts; a linear interpolation
   of the affine along z is the maximum-entropy prior consistent with this.
3. **No double-tissue artefacts.** A method that stacks the *interiors* of
   the two neighbours would mix volumes whose imaged content is spatially
   disjoint along the optical axis. zmorph never does this — every output
   plane is a single tissue surface morphed into the correct geometry.
4. **Out-of-distribution content stays absent.** Features visible only in
   the bulk of `vol_before` (e.g. fibres running along the optical axis)
   do not bleed into the interpolated slice because the interior of
   `vol_before` is never used.

The cost is that *micro-structure that was truly unique to the missing
block* cannot be recovered — it is physically unobservable from the
neighbours alone. This is a limitation shared by any interpolation method,
and is why interpolated slices stay flagged in `slice_config_final.csv` and
propagate `reliable=0` through `linum_register_pairwise.py` →
`linum_stack_slices_motor.py`.

### `weighted` and `average` — simple baselines (user-requested only)

- `weighted` produces a z-smoothed linear blend of `vol_before` and
  `vol_after`. No registration is performed.
- `average` is a plain 50/50 mean of the two neighbours.

Both are order-of-magnitude faster than zmorph but ignore the in-plane
deformation between adjacent slices. They are included for benchmarking and
for users who explicitly request a simple baseline. **They are never used
automatically** — zmorph never silently falls back to them because a
blended slice is also fabricated data and can introduce ghost contours
when the two neighbours differ.

### Failure handling: hard skip, no fabrication

When any quality gate fails, `interpolate_z_morph` returns
``(None, diagnostics)`` with ``diagnostics["interpolation_failed"] = True``
and a specific ``fallback_reason``. The CLI honours this by writing **no**
interpolated zarr, only a manifest fragment and a diagnostics JSON. The
Nextflow `interpolate_missing_slice` process declares the zarr output as
`optional: true` for this reason.

Possible failure reasons:

```
zmorph
  ├─ no_foreground_planes              (no boundary plane passed the foreground filter)
  ├─ low_overlap_ncc                   (best boundary NCC below min_overlap_correlation)
  ├─ registration_exception            (2D ITK optimiser raised)
  ├─ reg_did_not_improve               (post-reg NCC improvement < min_ncc_improvement)
  └─ affine_determinant_non_positive   (reflection / degenerate transform)
```

In every case the behaviour is identical: **no output zarr, manifest
fragment with `interpolation_failed=true`, `slice_config_final.csv`
stamps `interpolated=false, interpolation_failed=true`**. The stacked
volume has a genuine gap at that z position. Pairwise registration
automatically bridges the two surviving neighbours (the process takes
consecutive elements of the sorted `all_slices` channel; missing zarrs
are simply absent from that channel).

**Rationale.** A weighted blend of two neighbours that could not be
registered to each other is also fabricated data, with the extra failure
mode of ghost/double-contour artefacts whenever the tissue has moved
between cuts. Reporting "this slice could not be reconstructed" is
honest and keeps the final volume 100% measured.

### Boundary plane selection

`find_best_overlap_planes` searches the last `overlap_search_window` z-planes
of `vol_before` against the first `overlap_search_window` planes of
`vol_after` on the central ROI, after filtering by `min_foreground_fraction`
to discard agarose-only planes. The best NCC pair is used as the registration
reference. A minimum-correlation gate (`min_overlap_correlation`, default
0.3) falls back to a weighted average when no pair is similar enough to
register reliably.

Slab averaging (`reference_slab_size`, default 3) averages the chosen plane
with its immediate neighbours before running 2D registration, which makes
the fit robust to per-plane noise without requiring multi-plane
registration.

---

## Pipeline integration and per-slice trace

### Workflow placement

```
bring_to_common_space
        │
        ▼
[detectSingleGaps: single-slice gaps from slice_config.csv use=false]
        │
        ▼
interpolate_missing_slice  ──►  zarr + preview + diagnostics.json + manifest fragment
        │
        ▼  (one fragment per interpolated slice, collected)
        │
finalise_interpolation (linum_interpolate_missing_slice.py --finalise)
        │
        ├── input: current slice_config.csv (flowing from earlier steps)
        ├── input: per-slice manifest fragments
        ▼
slice_config_final.csv  (published at pipeline root; stamped interpolated/
                         interpolation_method_used/fallback_reason)
        │
        ▼
register_pairwise (flags transforms touching interpolated slices as reliable=0)
        │
        ▼
stack (down-weights reliable=0 transforms; reads slice_config_final via --slice_config)
```

### `slice_config_final.csv` — the per-slice audit trail

After interpolation the pipeline stamps the current `slice_config.csv` with
the interpolation outcome and publishes the result as
`slice_config_final.csv`. Only the trimmed, pipeline-relevant columns are
stored (raw NCC values and affine determinants live in the per-slice JSON
diagnostics, not in the CSV):

| Column | Meaning |
|--------|---------|
| `interpolated` | `true` if a reconstructed zarr was produced, `false` otherwise |
| `interpolation_failed` | `true` when zmorph hit a quality gate and no zarr was produced |
| `interpolation_method_used` | method actually used; empty string on `interpolation_failed=true` |
| `interpolation_fallback_reason` | reason the hard skip was triggered (``low_overlap_ncc``, ...), or empty |

All other canonical columns (`use`, `quality_score`, `galvo_confidence`,
`galvo_fix`, `rehomed`, `rehoming_reliable`, `auto_excluded`,
`auto_exclude_reason`, `notes`) carry over unchanged from the upstream
pipeline steps. For full numeric diagnostics consult the per-slice JSON
(see below).

If a manifest fragment refers to a slice not present in the input config
(e.g. an unexpected gap), `linumpy.io.slice_config.merge_fragments` appends
a new row with the interpolation trace columns populated and the rest of
the row left blank.

### Downstream reliability propagation

`linum_register_pairwise.py` detects `_interpolated.ome.zarr` inputs and
forces the resulting `pairwise_registration_metrics.json` to
`overall_status="error"` and `registration_confidence=0.0`.
`linum_stack_slices_motor.py` reads that flag via the `reliable` column and
down-weights those transforms during accumulation. Interpolated slices
therefore never masquerade as measured data in the stacked volume.

---

## Per-slice diagnostics JSON

Each interpolation run also emits a human-readable JSON file at
`${output}/interpolate_missing_slice/slice_z{NN}_interpolated_diagnostics.json`
with the full trace:

```json
{
  "method": "zmorph",
  "method_used": "zmorph",
  "fallback_reason": null,
  "ref_before": 47,
  "ref_after": 2,
  "pre_reg_ncc": 0.412,
  "post_reg_ncc": 0.687,
  "ncc_improvement": 0.275,
  "slab_before_range": [44, 50],
  "slab_after_range": [0, 5],
  "affine_matrix":   [[0.998, 0.011], [-0.011, 0.997]],
  "affine_translation": [1.23, -0.87],
  "affine_determinant": 0.995,
  "fractional_power_max_imag": 1.4e-14,
  "top_boundary_residual_mean": 0.0021,
  "bottom_boundary_residual_mean": 0.0018,
  "slice_id": "17",
  "slice_before_path": "...",
  "slice_after_path": "...",
  "output_path": "slice_z17_interpolated.ome.zarr"
}
```

Use these files when triaging a specific slice; use `slice_config_final.csv`
when reviewing the whole subject.

---

## Configuration

### Nextflow parameters (`workflows/reconst_3d/nextflow.config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interpolate_missing_slices` | `true` | Enable single-gap interpolation |
| `interpolation_method` | `'zmorph'` | `zmorph`, `average`, `weighted` |
| `interpolation_blend_method` | `'gaussian'` | `gaussian` (feathered), `linear` (50/50) |
| `interpolation_registration_metric` | `'MSE'` | `MSE`, `CC`, `MI` |
| `interpolation_max_iterations` | `1000` | Max iterations for boundary registration |
| `interpolation_overlap_search_window` | `5` | z-planes scanned at each boundary for the reference pair |
| `interpolation_min_overlap_correlation` | `0.3` | NCC gate; below this the method falls back to a weighted average |
| `interpolation_reference_slab_size` | `3` | planes averaged around the reference plane before registration |
| `interpolation_min_foreground_fraction` | `0.1` | minimum foreground fraction for a candidate boundary plane |
| `interpolation_min_ncc_improvement` | `0.05` | minimum post-reg NCC improvement to accept the transform |
| `interpolation_preview` | `false` | emit PNG previews next to each interpolated slice |

### Standalone script

```bash
# Recommended: z-morph method with diagnostics + manifest entry
linum_interpolate_missing_slice.py slice_z05.ome.zarr slice_z07.ome.zarr \
    slice_z06_interpolated.ome.zarr \
    --method zmorph \
    --blend_method gaussian \
    --slice_id 06 \
    --diagnostics slice_z06_diag.json \
    --manifest_entry slice_z06_manifest.csv

# Simple baselines (no 2D registration)
linum_interpolate_missing_slice.py slice_z05.ome.zarr slice_z07.ome.zarr \
    slice_z06_interpolated.ome.zarr --method weighted
```

### Post-hoc per-slice trace

```bash
# Merge a directory of per-slice manifest fragments into slice_config.csv.
linum_interpolate_missing_slice.py --finalise \
    --slice_config_in  slice_config.csv \
    --slice_config_out slice_config_final.csv \
    --fragments        interpolate_missing_slice
```

---

## Outputs

| Path (under `${params.output}`) | What it contains |
|---------------------------------|------------------|
| `interpolate_missing_slice/slice_z{NN}_interpolated.ome.zarr`                | the interpolated slice |
| `interpolate_missing_slice/slice_z{NN}_interpolated_preview.png`             | 1×4 preview (before / after / interpolated / XZ view) |
| `interpolate_missing_slice/slice_z{NN}_interpolated_diagnostics.json`        | per-slice diagnostics (see above) |
| `interpolate_missing_slice/slice_z{NN}_manifest.csv`                         | one-line per-slice manifest fragment (merged into slice_config_final.csv) |
| `slice_config_final.csv`                                                     | per-slice audit trail (canonical slice_config columns only) |

---

## Validation and quality checks

### Synthetic ground-truth benchmark

`linumpy/tests/test_stitching_interpolation.py` contains synthetic
ground-truth tests that build a three-slice stack, drop the middle slice,
and compare `zmorph` and the simple baselines against the held-out ground
truth via SSIM and PSNR. These tests exercise zmorph on drift-free and
drifting stacks and verify its exact-boundary property.

### Visual inspection

```bash
# Open the original neighbours and the interpolated slice side-by-side
napari slice_z05.ome.zarr slice_z06_interpolated.ome.zarr slice_z07.ome.zarr
```

### Aggregate review from a subject

```bash
column -s, -t < slice_config_final.csv | less -S
```

Columns to watch for:

- `interpolation_failed=true` → zmorph hit a quality gate; no zarr was
  produced and the slot is a gap in the final reconstruction. The
  specific reason is in `interpolation_fallback_reason`.
- Per-slice diagnostics JSON `post_reg_ncc` low (< 0.3) → the boundary
  planes are not similar enough for a trustworthy interpolation.

---

## Limitations

### Only single gaps

The method cannot interpolate when 2+ consecutive slices are missing:

- Insufficient information to estimate intermediate content,
- Deformation between non-adjacent slices is not well modelled by a single
  affine.

### Structural assumptions

- Adjacent slices have similar content.
- In-plane deformation between slices is well-approximated by an affine.
- No major structural changes between slices (the method will not
  hallucinate structures that only existed in the missing slice).

### Estimates, not measurements

Interpolated slices are *estimates*. They are flagged
`interpolated=true` in `slice_config_final.csv` and propagate `reliable=0`
through pairwise registration into stacking. Downstream analyses should
consider masking interpolated slices when computing volumetric statistics.

---

## References

[^1]: J. Lefebvre, A. Castonguay, P. Pouliot, M. Descoteaux, and F. Lesage,
    "Whole mouse brain imaging using optical coherence tomography:
    reconstruction, normalization, segmentation, and comparison with
    diffusion MRI," *Neurophotonics*, vol. 4, no. 4, p. 041501, July 2017,
    doi: 10.1117/1.NPh.4.4.041501.

[^2]: V. Arsigny, O. Commowick, X. Pennec, and N. Ayache, "A Log-Euclidean
    Framework for Statistics on Diffeomorphisms," in *Medical Image
    Computing and Computer-Assisted Intervention — MICCAI 2006*, Springer,
    2006.

### Additional references

- Lee, T. Y., & Wang, W. H. (2000). "Morphology-based three-dimensional
  interpolation." *IEEE TMI*, 19(7), 711-720.
- Raya, S. P., & Udupa, J. K. (1990). "Shape-Based Interpolation of
  Multidimensional Objects." *IEEE TMI*, 9(1), 32-42.
- Bao, W., et al. (2019). "Depth-Aware Video Frame Interpolation." *CVPR*.
- Penney, G. P., et al. (2004). "A comparison of similarity measures for
  use in 2-D-3-D medical image registration." *IEEE TMI*.

---

## Files

| File | Description |
|------|-------------|
| `linumpy/stitching/interpolation.py` | Interpolation algorithms (`interpolate_z_morph`, `interpolate_weighted`, `interpolate_average`, helpers) |
| `scripts/stacking/linum_interpolate_missing_slice.py` | Standalone CLI (also provides `--finalise` for merging manifest fragments) |
| `scripts/stitching/linum_register_pairwise.py` | Automatically flags registrations touching interpolated slices |
| `scripts/stacking/linum_stack_slices_motor.py` | Uses `reliable=0` to down-weight interpolated-slice transforms |
| `workflows/reconst_3d/soct_3d_reconst.nf` | `interpolate_missing_slice` + `finalise_interpolation` processes |
| `workflows/reconst_3d/nextflow.config` | Interpolation parameters |
| `linumpy/tests/test_stitching_interpolation.py` | Unit tests and synthetic ground-truth benchmarks |
| `scripts/tests/test_interpolate_missing_slice.py` | Script-level CLI tests |
