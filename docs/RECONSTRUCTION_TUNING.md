# Reconstruction Parameter Tuning Guide

A walkthrough of the parameters that most affect 3D reconstruction quality, what
they do, what raising/lowering them changes, and which knobs to reach for when
specific artifacts show up. Aimed at users who are new to the pipeline; if you
already know your way around it, the companion {doc}`Reconstruction Tuning
Quick Reference <RECONSTRUCTION_TUNING_QUICKREF>` is denser.

For the runtime-mechanics view of the same parameters, see the inline comments
in `workflows/reconst_3d/nextflow.config` (described in {doc}`NEXTFLOW_WORKFLOWS`).
For diagnostic *scripts* (rotation drift, tile dilation, motor-only stitching),
see {doc}`RECONSTRUCTION_DIAGNOSTICS`.

---

## How to approach tuning

Reconstruction problems almost always show up at slice boundaries: edges that
don't line up between slices, intensity steps, sudden Z-jumps, or rotated
sections. The pipeline runs in stages, and each artifact is usually produced
by one specific stage. The fastest tuning workflow is:

1. **Look at the previews first.** `output/common_space_previews/*.png` shows
   each slice in XY/XZ/YZ. Most artifacts are visible at a glance.
2. **Identify the stage.** Use the symptom map below to pick a stage.
3. **Adjust the smallest set of parameters that targets that stage.**
4. **Re-run with `-resume`.** Nextflow caches everything upstream of the
   change, so iteration is fast.
5. **Verify against previews and `output/stack/*.csv` metrics.**

The pipeline stages, in order:

```
Tile stitching → Slice quality → Common-space alignment → Missing-slice
  interpolation → Pairwise registration → Stacking → Bias correction → Atlas
```

A change to a parameter in stage *N* invalidates caches for stage *N* and all
later stages. Earlier stages are reused.

---

## Quick orientation: which stage produces which artifact?

| Artifact | Most likely stage |
|---|---|
| Tiles within a slice misaligned, seams visible | Tile stitching |
| Whole slice looks degraded, smeared, or missing tissue | Slice quality / preprocessing |
| Slice shifted left/right or up/down relative to neighbours | Common-space alignment |
| One slice has a clear gap then resumes | Missing-slice interpolation |
| Two consecutive slices are rotated relative to each other | Pairwise registration |
| Sudden Z-jump or tilt between slices | Stacking |
| Slow XY drift across many slices | Stacking (translation accumulation) |
| Visible step in brightness between slices | Bias correction |
| Atlas overlay is misaligned | Atlas registration |

---

## Profiles before parameters

Before reaching for individual parameters, pick a profile. Profiles set
sensible groups of parameters together, and you can override individual values
on top of a profile. They are defined in `workflows/reconst_3d/nextflow.config`
(see {doc}`NEXTFLOW_WORKFLOWS`).

- **`-profile conservative`** (recommended starting point): trusts motor
  positions for XY, applies only rotation from pairwise registration, skips
  any registration flagged warning/error, interpolates single-slice gaps.
  Fails gracefully when registration is unreliable.
- **`-profile aggressive`**: applies full pairwise transforms including XY
  translations and accumulates them. Best alignment when registration is
  reliable. Can compound errors when it is not.
- **`-profile minimal`**: ignores pairwise registration entirely
  (`apply_pairwise_transforms = false`). Most stable, fastest. Use when motor
  positions are reliable and registration repeatedly fails.

If the conservative profile produces a clean reconstruction, stop. Most of the
parameters below exist to handle specific failure modes you may not have.

---

## Stage 1: Tile stitching

Assembles the tiles within each slice into a single 2D image. The dominant
choice here is whether to trust motor positions or fit a transform from the
images.

| Parameter | Default | Effect |
|---|---|---|
| `use_motor_positions_for_stitching` | `true` | Use motor encoder positions for tile placement. Recommended — analysis showed image-based stitching introduces ~3% systematic compression. |
| `stitch_overlap_fraction` | `0.2` | Expected fraction of overlap between adjacent tiles. **Must match the acquisition setting.** |
| `stitch_blending_method` | `'diffusion'` | Tile-seam blending. `'diffusion'` is the smoothest, `'average'` is faster, `'none'` shows raw seams (useful when debugging tile placement). |
| `max_blend_refinement_px` | `10` | Sub-pixel refinement budget when blending seams. Larger values let blending hide small motor inaccuracies but can over-smear true features. |
| `stitch_global_transform` | `false` | Pool a single 2x2 affine across many slices and reuse it. Helps when individual slices have too few tiles for a stable per-slice fit. Enable if per-slice stitch transforms jitter. |

**If you see seams within a slice:**
1. Confirm `stitch_overlap_fraction` matches the acquisition.
2. Try `stitch_blending_method = 'diffusion'` if not already.
3. As a last resort, raise `max_blend_refinement_px` (10 → 20).

**If individual slices look fine but the whole slice is rotated/skewed
slightly compared to neighbours**: this is usually a per-slice stitch
transform problem. Try `stitch_global_transform = true`.

---

## Stage 2: Slice quality assessment

Optionally scores each slice and excludes the bad ones from common-space
alignment so they don't poison their neighbours.

| Parameter | Default | Effect |
|---|---|---|
| `auto_assess_quality` | `false` | Enable automatic quality scoring. |
| `auto_assess_min_quality` | `0.3` | Slices scored below this are excluded. Lower = more permissive. |
| `auto_assess_exclude_first` | `1` | Always exclude the first N calibration slices. |
| `auto_exclude_enabled` | `true` | Detect runs of consecutive low-quality registrations after pairwise (different mechanism). |

**When to enable:** if you see one or two badly degraded slices dragging the
common-space alignment of their neighbours.

---

## Stage 3: Common-space alignment

Aligns each stitched slice into a shared XY canvas using `shifts_xy.csv`
(motor positions). Most XY-misalignment problems are tuned here.

### Encoder glitch correction

`detect_rehoming` corrects two known motor artifacts:

1. **Encoder glitch spikes**: a single large step that self-cancels with the
   adjacent step. The motor moved cleanly; the encoder briefly reported a
   wrong position.
2. **Tile-FOV expansion events** (legacy data only): adding a new column of
   tiles at acquisition time produced a motor jump of exactly N × tile_FOV.

| Parameter | Default | Effect |
|---|---|---|
| `detect_rehoming` | `true` | Enable both passes. Almost always wanted. |
| `rehoming_max_shift_mm` | `0.5` | Steps below this magnitude aren't checked. Lower to catch smaller glitches at the cost of false positives. |
| `rehoming_return_fraction` | `0.4` | Detection sensitivity. Lower = more conservative. |
| `tile_fov_mm` | `null` | Set to the tile field-of-view (e.g. `0.875`) only when re-using older `shifts_xy.csv` files that contain mosaic-expansion artifacts. New shift files generated by `linum_estimate_xy_shift_from_metadata.py` no longer need this. |
| `tile_fov_tolerance` | `0.05` | Fractional tolerance when matching steps to tile-FOV multiples. |

### Image-based refinement of unreliable transitions

For transitions where the motor reading is flagged unreliable
(`reliable=0` in shifts CSV), the pipeline can fall back to image-based
phase correlation between the two slices.

| Parameter | Default | Effect |
|---|---|---|
| `common_space_refine_unreliable` | `false` | Enable image-based refinement. Turn on when slices grow significantly between acquisitions or when raw motor steps are noisy. |
| `common_space_refine_max_discrepancy_px` | `0` | Reject the image estimate if it disagrees with motor by more than this many pixels. `0` = accept all. Recommended `50` if you see image refinement producing wild values. |
| `common_space_refine_min_correlation` | `0.0` | Reject refinements with low correlation quality. Recommended `0.15`–`0.3`. |

**Recipe — large XY jumps between specific slices:**

1. Inspect `shifts_xy.csv` for the affected slice IDs. Look at `x_shift_mm`
   and `y_shift_mm`. Steps > 0.5 mm are suspect.
2. Check `output/detect_rehoming_events/shifts_xy_clean.csv`. If `reliable=0`
   for those rows, the corrected file already accounts for them.
3. If the misalignment persists, enable
   `common_space_refine_unreliable = true` and re-run.

### Excluded-slice interpolation

| Parameter | Default | Effect |
|---|---|---|
| `common_space_excluded_slice_mode` | `'local_median'` | How to fill XY positions for excluded slices. `'local_median'` averages neighbour shifts. |
| `common_space_excluded_slice_window` | `2` | Number of neighbours considered. |

---

## Stage 4: Missing-slice interpolation

Fills single-slice gaps so the stack has a continuous Z. The default is
`zmorph` — a 2D registration of the boundary planes is used to morph
fractional-affine intermediate planes between the two neighbours.

| Parameter | Default | Effect |
|---|---|---|
| `interpolate_missing_slices` | `true` | Enable interpolation. Disable to keep gaps explicit. |
| `interpolation_method` | `'zmorph'` | `'zmorph'` (registration-based morph), `'weighted'` (z-smoothed linear blend), `'average'` (50/50). |
| `interpolation_blend_method` | `'gaussian'` | `'gaussian'` (feathered) or `'linear'`. |
| `interpolation_min_overlap_correlation` | `0.3` | If the boundary-plane NCC is below this, falls back to `'weighted'`. |
| `interpolation_min_ncc_improvement` | `0.05` | If post-registration NCC doesn't improve by this much, falls back to `'weighted'`. |

**When to lower `interpolation_min_overlap_correlation`:** if zmorph keeps
falling back to weighted on slices that visually look fine, lower to e.g.
`0.2`. Watch for spurious deformations on noisier boundaries.

**When to disable interpolation:** if you specifically want to see where the
missing slices are (e.g. for QC), set `interpolate_missing_slices = false`.

---

## Stage 5: Pairwise registration

Computes small inter-slice corrections (rotation, sub-pixel translation). The
*main* alignment comes from motor positions; pairwise transforms are
refinements applied on top during stacking.

| Parameter | Default | Effect |
|---|---|---|
| `registration_transform` | `'euler'` | `'translation'` (XY only) or `'euler'` (XY + rotation). Use `'euler'` for any sample where slices are rotated relative to each other. |
| `registration_max_translation` | `200.0` | Optimizer bound on translation (px). Keep large; the actual *applied* translation is governed in stacking. Increase only if pairwise metrics report frequent boundary hits. |
| `registration_max_rotation` | `5.0` | Optimizer bound on rotation (deg). Raise (e.g. to `35.0`) for obliquely-mounted samples where slice-to-slice rotation can be large. |
| `registration_initial_alignment` | `'both'` | Initialization. `'both'` runs centre-of-mass and gradient inits and picks the better one. |

**Recipe — pairwise registration looks "stuck":**

1. Open a pairwise metrics JSON in `output/register_pairwise/`. Look for
   `mag` (translation magnitude in px) and `rotation`.
2. If many slices have `mag` ≈ `registration_max_translation` × 0.95+, the
   optimizer is hitting the boundary. Raise the bound (e.g. 200 → 400) but
   inspect: usually this means the input alignment is wrong upstream, not
   that the bound is too tight.
3. Keep `skip_warning_transforms = true` (in stacking) so unreliable
   boundary-hit transforms aren't applied.

---

## Stage 6: Stacking

Where pairwise corrections actually get applied to the volume. This is the
single most-tuned stage and the source of most reconstruction drift.

### Whether to apply transforms at all

| Parameter | Default | Effect |
|---|---|---|
| `apply_pairwise_transforms` | `true` | Master switch. Set `false` to stack motor-only (this is what `-profile minimal` does). |
| `apply_rotation_only` | `false` | Apply only the rotation component, keep XY from motor positions. The conservative profile sets this `true`. |
| `use_expected_z_overlap` | `true` | Use the expected slice thickness for Z spacing instead of correlation-based matching. Recommended; correlation matching is brittle at slice boundaries. |
| `max_rotation_deg` | `5.0` | Clamp applied rotations larger than this. Prevents single bad slices from rotating the entire stack downstream. |

### Per-slice transform gating

Each pairwise registration produces a status (ok/warning/error) and metrics.
The pipeline can refuse to apply low-quality transforms.

| Parameter | Default | Effect |
|---|---|---|
| `skip_error_transforms` | `true` | Skip transforms registered against interpolated slices etc. **Keep enabled.** |
| `skip_warning_transforms` | `true` | Skip transforms that hit the optimizer boundary. **Keep enabled.** Disabling causes Z-positioning errors. |
| `transform_confidence_high` | `0.6` | Above this, full transform applied. |
| `transform_confidence_low` | `0.3` | Below this, transform skipped entirely. Between low and high, rotation-only. |
| `load_transform_max_rotation` | `0.0` | Metric-based rotation gate. Set to e.g. `4.0` for noisy data — transforms with rotation > 4° are not loaded at all. |
| `load_transform_min_zcorr` | `0.0` | Metric-based Z-correlation gate. Pairs with `load_transform_max_rotation`. |

### Translation accumulation

Pairwise translations can be accumulated as cumulative canvas offsets — this
"steers the viewing plane" through the volume. Useful when the sample drifts
slowly across many slices.

| Parameter | Default | Effect |
|---|---|---|
| `stack_accumulate_translations` | `true` | Enable cumulative-offset accumulation. |
| `stack_confidence_weight_translations` | `true` | Weight each translation by its confidence before accumulating. Reduces noise from low-quality slices. |
| `stack_translation_smooth_sigma` | `3.0` | Gaussian smoothing (sigma in slices) over accumulated translations. Higher removes more jitter; too high and you lose legitimate trends. |
| `stack_max_pairwise_translation` | `0` | Translations exceeding this magnitude (px) are zeroed before accumulation. **Set to 0 to disable.** Even an optimizer-boundary value carries directional information; zeroing it is usually worse than keeping it. Use a non-zero value only if specific slices have clearly erroneous translations worse than zero. |
| `stack_max_cumulative_drift_px` | `50` | Cap on total accumulated drift. **Recommended: `0` (disabled).** If common-space alignment is correct, accumulated translations should converge naturally; a cap usually hides a real upstream problem. |
| `stack_smooth_window` | `5` | Moving-average window over per-slice rotations. Reduces visible jumps from outlier slices. |

**Recipe — slow XY drift across many slices in XZ/YZ view:**

1. Plot `output/stack/translation_per_slice.csv` (or read it). If translations
   trend monotonically, that's drift.
2. First check that common-space alignment is correct (XY view of each slice
   aligns). If common-space is the problem, fix it there first.
3. If common-space looks fine, raise `stack_translation_smooth_sigma`
   (3 → 5) to wash out noise.
4. Disable `stack_max_cumulative_drift_px` (set `0`) if it's clamping a real
   trend.

**Recipe — sudden tilt or jump at one specific slice:**

1. Open `output/stack/stacking_decisions.csv`. The affected row will usually
   show `transform_loaded=False` (a gap) or the pairwise registration metric
   will show a high rotation/translation outlier.
2. If `transform_loaded=False`: check `output/register_pairwise/` metrics for
   why it was rejected. Often `overall_status="error"` because a neighbour
   was interpolated.
3. If a noisy rotation: lower `load_transform_max_rotation` to a tighter
   value (e.g. 4.0) or raise the gate.

---

## Stage 7: Bias field correction (optional)

N4 bias correction applied after stacking. Removes depth-dependent attenuation
and slow inter-slice intensity drift.

| Parameter | Default | Effect |
|---|---|---|
| `correct_bias_field` | `false` | Master switch. |
| `bias_mode` | `'two_pass'` | `'per_section'`, `'global'`, or `'two_pass'` (per-section then global). `'two_pass'` is the recommended default. |
| `bias_strength` | `1.0` | Mixing strength. `0.0` = passthrough, `1.0` = full correction. Lower if N4 is over-correcting tissue features. |
| `bias_histogram_match_per_zplane` | `true` | Match each Z-plane to the global tissue distribution before N4. **Strongly reduces inter-slice intensity steps.** Roughly an order of magnitude better than chunked HM in tested cases. |
| `bias_tissue_threshold` | `0.005` | Voxels at or below this intensity are considered background and excluded from histogram matching. Lower if tissue is being treated as background. |
| `bias_zprofile_smooth_sigma` | `2.0` | Gaussian smoothing (sigma in Z-planes) of a residual scalar gain after HM. Eliminates the small inter-slice steps HM cannot remove. `0` = disabled. Typical range 2–4. |

**Recipe — visible intensity steps between slices:**

1. Enable `correct_bias_field = true`.
2. Keep `bias_histogram_match_per_zplane = true` and
   `bias_zprofile_smooth_sigma = 2.0`.
3. If steps remain, raise sigma (2 → 4).
4. If tissue features are getting flattened, lower `bias_strength` (1.0 → 0.7).

---

## Stage 8: Atlas registration (optional)

Registers the stacked volume to the Allen Mouse Brain Atlas.

| Parameter | Default | Effect |
|---|---|---|
| `align_to_ras_enabled` | `false` | Master switch. |
| `allen_resolution` | `25` | Atlas resolution (10/25/50/100 µm). Lower = slower & higher memory. |
| `allen_metric` | `'MI'` | `'MI'`, `'MSE'`, `'CC'`, `'AntsCC'`. `'MI'` is robust across modalities. |
| `allen_registration_level` | `2` | Pyramid level used for registration. Higher = faster but coarser. Output is always written at all pyramid resolutions. |
| `ras_input_orientation` | `''` | 3-letter code (e.g. `'PIR'`) telling the registrar what orientation the input is in. Crucial for correct atlas alignment. |
| `ras_initial_rotation` | `''` | `"Rx Ry Rz"` initial rotation hint in degrees. Use when MOMENTS-based init fails. |
| `allen_preview` | `true` | Save a 3×3 input/aligned/template comparison. Always check this. |
| `ras_orientation_preview` | `false` | Save a preview after orientation/initial-rotation are applied but before registration. **Enable when tuning `ras_input_orientation` and `ras_initial_rotation`** — you can verify orientation parameters without running the full registration. |

**Recipe — atlas overlay is mirrored or rotated 90°:**

1. Set `ras_orientation_preview = true` and re-run `align_to_ras` only.
2. Inspect the orientation preview: if the brain is rotated, adjust
   `ras_initial_rotation` (e.g. `"0.0 0.0 90.0"` for a 90° Z rotation).
3. If the brain is flipped, adjust `ras_input_orientation` (try the
   neighbour code: e.g. `'PIR'` → `'AIR'` if anterior/posterior is
   inverted).
4. Iterate until the orientation preview looks roughly aligned to the
   atlas template, then re-run with the full registration.

---

## Reference parameter sets

The canonical config and profile blocks already encode several known-good
combinations. For per-subject overrides see the deployed configs at
`/scratch/workspace/sub-XX/nextflow.config`. Common combinations:

**For data with mosaic-grid expansion events (legacy `shifts_xy.csv`):**
```groovy
detect_rehoming = true
tile_fov_mm = 0.875                              // your acquisition tile FOV
common_space_refine_unreliable = true
common_space_refine_max_discrepancy_px = 0
```

**For obliquely-mounted samples (large slice-to-slice rotations):**
```groovy
registration_transform = 'euler'
registration_max_rotation = 35.0
load_transform_max_rotation = 4.0
stack_smooth_window = 5
```

**For visible inter-slice intensity steps:**
```groovy
correct_bias_field = true
bias_mode = 'two_pass'
bias_histogram_match_per_zplane = true
bias_zprofile_smooth_sigma = 2.0
```

**For one or two bad slices contaminating the stack:**
```groovy
auto_assess_quality = true
auto_assess_min_quality = 0.3
auto_exclude_enabled = true
auto_exclude_consecutive = 3
auto_exclude_z_corr = 0.6
```

---

## See also

- {doc}`Reconstruction Tuning Quick Reference <RECONSTRUCTION_TUNING_QUICKREF>` —
  symptom→fix table for experienced users.
- {doc}`Reconstruction Diagnostics <RECONSTRUCTION_DIAGNOSTICS>` — diagnostic
  scripts and the `diagnostic_mode` master switch.
- {doc}`Pipeline Overview <PIPELINE_OVERVIEW>` — high-level stage descriptions.
- {doc}`Nextflow Workflows <NEXTFLOW_WORKFLOWS>` — running and configuring the
  pipeline.
- `workflows/reconst_3d/nextflow.config` — full canonical config with
  per-parameter inline comments (see {doc}`NEXTFLOW_WORKFLOWS`).
