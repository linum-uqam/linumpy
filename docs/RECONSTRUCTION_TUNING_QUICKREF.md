# Reconstruction Tuning — Quick Reference

Cheat sheet for tuning the 3D reconstruction pipeline. Assumes you know what
the stages do. For background and recipes, see the full {doc}`Reconstruction
Parameter Tuning Guide <RECONSTRUCTION_TUNING>`.

---

## Profiles first

| Profile | When to use |
|---|---|
| `-profile conservative` | Default starting point. Motor-XY + rotation-only. |
| `-profile aggressive` | Registration is reliable; want full pairwise corrections. |
| `-profile minimal` | Motor-only. Registration consistently fails. |

---

## Symptom → fix

| Symptom | First parameters to try |
|---|---|
| Seams within a slice | `stitch_overlap_fraction` (match acquisition); `stitch_blending_method='diffusion'`; raise `max_blend_refinement_px` |
| Per-slice stitch transform jitter | `stitch_global_transform=true` |
| One/two bad slices contaminating neighbours | `auto_assess_quality=true`, `auto_assess_min_quality=0.3` |
| Large XY jump at specific slices | Check `shifts_xy.csv`; `detect_rehoming=true`; `common_space_refine_unreliable=true` |
| Repeating large XY steps near tile-FOV multiples (legacy data) | `tile_fov_mm=<acquisition_fov_mm>`, `tile_fov_tolerance=0.05` |
| Image refinement producing wild values | `common_space_refine_max_discrepancy_px=50`, `common_space_refine_min_correlation=0.15` |
| Single-slice gap visible in Z | `interpolate_missing_slices=true` (default); lower `interpolation_min_overlap_correlation` if zmorph keeps falling back |
| Slice-to-slice rotation in oblique samples | `registration_transform='euler'`, raise `registration_max_rotation` (e.g. 35.0) |
| Sudden tilt/jump at one slice | Check `output/stack/stacking_decisions.csv` for `transform_loaded=False`; tighten `load_transform_max_rotation` (e.g. 4.0) |
| Slow XY drift across many slices | Raise `stack_translation_smooth_sigma` (3→5); set `stack_max_cumulative_drift_px=0` |
| Optimizer-boundary translation hits | Keep `skip_warning_transforms=true`; **don't** zero them via `stack_max_pairwise_translation` unless they're clearly worse than zero |
| Visible inter-slice intensity steps | `correct_bias_field=true`, `bias_histogram_match_per_zplane=true`, `bias_zprofile_smooth_sigma=2.0` (raise to 4 if persists) |
| Bias correction over-flattening tissue | Lower `bias_strength` (1.0 → 0.7) |
| Atlas overlay rotated 90° / flipped | `ras_orientation_preview=true`, then tune `ras_input_orientation` and `ras_initial_rotation` |

---

## Per-stage quick tables

### Tile stitching
| Param | Default | Lever |
|---|---|---|
| `use_motor_positions_for_stitching` | `true` | Keep on. |
| `stitch_overlap_fraction` | `0.2` | Must match acquisition. |
| `stitch_blending_method` | `'diffusion'` | `'none'` to debug seams. |
| `stitch_global_transform` | `false` | Pool affine across slices. |

### Common-space alignment
| Param | Default | Lever |
|---|---|---|
| `detect_rehoming` | `true` | Keep on. |
| `rehoming_max_shift_mm` | `0.5` | Lower to catch smaller glitches. |
| `tile_fov_mm` | `null` | Set only for legacy shift CSVs with FOV-multiple jumps. |
| `common_space_refine_unreliable` | `false` | Image-based fallback for `reliable=0` shifts. |
| `common_space_refine_max_discrepancy_px` | `0` | Recommended `50` to gate wild image estimates. |
| `common_space_refine_min_correlation` | `0.0` | Recommended `0.15`–`0.3`. |

### Pairwise registration
| Param | Default | Lever |
|---|---|---|
| `registration_transform` | `'euler'` | `'translation'` if no rotation expected. |
| `registration_max_rotation` | `5.0` | Raise for oblique samples (e.g. 35.0). |
| `registration_max_translation` | `200.0` | Keep large; not the actual applied bound. |

### Stacking
| Param | Default | Lever |
|---|---|---|
| `apply_pairwise_transforms` | `true` | `false` for motor-only. |
| `apply_rotation_only` | `false` | `true` (conservative): keep XY from motor. |
| `skip_warning_transforms` | `true` | **Keep on.** Disabling causes Z errors. |
| `skip_error_transforms` | `true` | **Keep on.** |
| `max_rotation_deg` | `5.0` | Clamp applied rotation. |
| `load_transform_max_rotation` | `0.0` | Tighten gate (e.g. 4.0) for noisy pairwise rotations. |
| `stack_accumulate_translations` | `true` | Cumulative canvas offsets. |
| `stack_translation_smooth_sigma` | `3.0` | Higher = smoother accumulated drift. |
| `stack_max_cumulative_drift_px` | `50` | Set `0` to disable cap (recommended). |
| `stack_max_pairwise_translation` | `0` | Set `0` to disable boundary zeroing (recommended). |
| `stack_smooth_window` | `5` | Moving avg over rotations. |
| `transform_confidence_high` | `0.6` | Full transform threshold. |
| `transform_confidence_low` | `0.3` | Skip threshold. Between = rotation-only. |

### Bias correction
| Param | Default | Lever |
|---|---|---|
| `correct_bias_field` | `false` | Master switch. |
| `bias_mode` | `'two_pass'` | `'per_section'` if per-slice issues only. |
| `bias_strength` | `1.0` | Lower if over-flattening. |
| `bias_histogram_match_per_zplane` | `true` | Keep on. |
| `bias_zprofile_smooth_sigma` | `2.0` | Range 2–4. |
| `bias_tissue_threshold` | `0.005` | Lower if tissue mistaken for bg. |

### Atlas registration
| Param | Default | Lever |
|---|---|---|
| `align_to_ras_enabled` | `false` | Master switch. |
| `allen_resolution` | `25` | Atlas µm. |
| `allen_registration_level` | `2` | Higher = faster, coarser. |
| `ras_input_orientation` | `''` | 3-letter code (e.g. `'PIR'`). |
| `ras_initial_rotation` | `''` | `"Rx Ry Rz"` deg hint. |
| `ras_orientation_preview` | `false` | Turn on while tuning orientation. |

---

## Diagnostic file map

| File | What to look at |
|---|---|
| `output/common_space_previews/*.png` | Per-slice XY/XZ/YZ — most artifacts visible here. |
| `output/stack/stacking_decisions.csv` | `transform_loaded` flags, applied rotations/translations. |
| `output/stack/z_matches.csv` | Z-overlap correlation per pair. |
| `output/register_pairwise/*_metrics.json` | Per-pair `mag`, `rotation`, `overall_status`, `z_correlation`. |
| `output/detect_rehoming_events/shifts_xy_clean.csv` | Corrected shifts; `reliable` column. |
| `shifts_xy.csv` | Raw motor steps (`x_shift_mm`, `y_shift_mm`). Steps > 0.5mm are suspect. |

---

## Rerun cascade rules

A change to a parameter invalidates that stage and all downstream stages.
Use `-resume` — Nextflow caches everything upstream automatically.

| Change to | Re-runs from |
|---|---|
| `detect_rehoming`, `tile_fov_mm` | `detect_rehoming` → all downstream |
| `common_space_*` | `common_space` → all downstream |
| `registration_*` | `register_pairwise` → all downstream |
| `apply_*`, `stack_*`, `load_transform_*`, `transform_confidence_*` | `stack` → all downstream |
| `bias_*`, `correct_bias_field` | `correct_bias_field` → atlas |
| `align_to_ras_*`, `allen_*`, `ras_*` | `align_to_ras` only |
