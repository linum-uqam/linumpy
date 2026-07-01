"""Deprecated ``linum_xxx.py`` command-name shims.

Each function is registered as the target of a legacy ``linum_xxx.py``
console-scripts entry point.  The canonical names are ``linum-xxx``
(hyphens, no ``.py`` suffix).

These shims emit a :class:`DeprecationWarning` then delegate to the
corresponding ``main()`` so all existing pipeline invocations keep working.
"""

import sys


def _warn(old: str, new: str) -> None:
    print(
        f"DeprecationWarning: Command '{old}' is deprecated; use '{new}' instead.",
        file=sys.stderr,
    )


# --- conversion ---
def linum_axis_xyz_to_zyx() -> None:
    _warn("linum_axis_xyz_to_zyx.py", "linum-axis-xyz-to-zyx")
    from scripts.conversion.linum_axis_xyz_to_zyx import main

    main()


def linum_convert_bin_to_nii() -> None:
    _warn("linum_convert_bin_to_nii.py", "linum-convert-bin-to-nii")
    from scripts.conversion.linum_convert_bin_to_nii import main

    main()


def linum_convert_nifti_to_nrrd() -> None:
    _warn("linum_convert_nifti_to_nrrd.py", "linum-convert-nifti-to-nrrd")
    from scripts.conversion.linum_convert_nifti_to_nrrd import main

    main()


def linum_convert_nifti_to_zarr() -> None:
    _warn("linum_convert_nifti_to_zarr.py", "linum-convert-nifti-to-zarr")
    from scripts.conversion.linum_convert_nifti_to_zarr import main

    main()


def linum_convert_omezarr_to_nifti() -> None:
    _warn("linum_convert_omezarr_to_nifti.py", "linum-convert-omezarr-to-nifti")
    from scripts.conversion.linum_convert_omezarr_to_nifti import main

    main()


def linum_convert_tiff_to_nifti() -> None:
    _warn("linum_convert_tiff_to_nifti.py", "linum-convert-tiff-to-nifti")
    from scripts.conversion.linum_convert_tiff_to_nifti import main

    main()


def linum_convert_tiff_to_omezarr() -> None:
    _warn("linum_convert_tiff_to_omezarr.py", "linum-convert-tiff-to-omezarr")
    from scripts.conversion.linum_convert_tiff_to_omezarr import main

    main()


def linum_convert_zarr_to_omezarr() -> None:
    _warn("linum_convert_zarr_to_omezarr.py", "linum-convert-zarr-to-omezarr")
    from scripts.conversion.linum_convert_zarr_to_omezarr import main

    main()


def linum_extract_pyramid_levels() -> None:
    _warn("linum_extract_pyramid_levels.py", "linum-extract-pyramid-levels")
    from scripts.conversion.linum_extract_pyramid_levels import main

    main()


def linum_fix_galvo_shift_zarr() -> None:
    _warn("linum_fix_galvo_shift_zarr.py", "linum-fix-galvo-shift-zarr")
    from scripts.conversion.linum_fix_galvo_shift_zarr import main

    main()


def linum_reorient_nifti_to_ras() -> None:
    _warn("linum_reorient_nifti_to_ras.py", "linum-reorient-nifti-to-ras")
    from scripts.conversion.linum_reorient_nifti_to_ras import main

    main()


def linum_resample_nifti() -> None:
    _warn("linum_resample_nifti.py", "linum-resample-nifti")
    from scripts.conversion.linum_resample_nifti import main

    main()


# --- mosaic ---
def linum_create_all_mosaic_grids_2d() -> None:
    _warn("linum_create_all_mosaic_grids_2d.py", "linum-create-all-mosaic-grids-2d")
    from scripts.mosaic.linum_create_all_mosaic_grids_2d import main

    main()


def linum_create_mosaic_grid_2d() -> None:
    _warn("linum_create_mosaic_grid_2d.py", "linum-create-mosaic-grid-2d")
    from scripts.mosaic.linum_create_mosaic_grid_2d import main

    main()


def linum_create_mosaic_grid_3d() -> None:
    _warn("linum_create_mosaic_grid_3d.py", "linum-create-mosaic-grid-3d")
    from scripts.mosaic.linum_create_mosaic_grid_3d import main

    main()


def linum_crop_3d_mosaic_below_interface() -> None:
    _warn("linum_crop_3d_mosaic_below_interface.py", "linum-crop-3d-mosaic-below-interface")
    from scripts.mosaic.linum_crop_3d_mosaic_below_interface import main

    main()


def linum_crop_tiles() -> None:
    _warn("linum_crop_tiles.py", "linum-crop-tiles")
    from scripts.mosaic.linum_crop_tiles import main

    main()


def linum_generate_mosaic_aips() -> None:
    _warn("linum_generate_mosaic_aips.py", "linum-generate-mosaic-aips")
    from scripts.mosaic.linum_generate_mosaic_aips import main

    main()


def linum_merge_slices_into_folders() -> None:
    _warn("linum_merge_slices_into_folders.py", "linum-merge-slices-into-folders")
    from scripts.mosaic.linum_merge_slices_into_folders import main

    main()


def linum_resample_mosaic_grid() -> None:
    _warn("linum_resample_mosaic_grid.py", "linum-resample-mosaic-grid")
    from scripts.mosaic.linum_resample_mosaic_grid import main

    main()


# --- illumination ---
def linum_compensate_illumination() -> None:
    _warn("linum_compensate_illumination.py", "linum-compensate-illumination")
    from scripts.illumination.linum_compensate_illumination import main

    main()


def linum_compute_attenuation_bias_field() -> None:
    _warn("linum_compute_attenuation_bias_field.py", "linum-compute-attenuation-bias-field")
    from scripts.illumination.linum_compute_attenuation_bias_field import main

    main()


def linum_correct_bias_field() -> None:
    _warn("linum_correct_bias_field.py", "linum-correct-bias-field")
    from scripts.illumination.linum_correct_bias_field import main

    main()


def linum_estimate_illumination() -> None:
    _warn("linum_estimate_illumination.py", "linum-estimate-illumination")
    from scripts.illumination.linum_estimate_illumination import main

    main()


def linum_fix_illumination_3d() -> None:
    _warn("linum_fix_illumination_3d.py", "linum-fix-illumination-3d")
    from scripts.illumination.linum_fix_illumination_3d import main

    main()


def linum_fix_illumination_basic() -> None:
    _warn("linum_fix_illumination_basic.py", "linum-fix-illumination-basic")
    from scripts.illumination.linum_fix_illumination_basic import main

    main()


def linum_intensity_normalization() -> None:
    _warn("linum_intensity_normalization.py", "linum-intensity-normalization")
    from scripts.illumination.linum_intensity_normalization import main

    main()


def linum_normalize_intensities_per_slice() -> None:
    _warn("linum_normalize_intensities_per_slice.py", "linum-normalize-intensities-per-slice")
    from scripts.illumination.linum_normalize_intensities_per_slice import main

    main()


def linum_sweep_illumination() -> None:
    _warn("linum_sweep_illumination.py", "linum-sweep-illumination")
    from scripts.illumination.linum_sweep_illumination import main

    main()


# --- attenuation ---
def linum_compensate_attenuation() -> None:
    _warn("linum_compensate_attenuation.py", "linum-compensate-attenuation")
    from scripts.attenuation.linum_compensate_attenuation import main

    main()


def linum_compensate_attenuation_inplace() -> None:
    _warn("linum_compensate_attenuation_inplace.py", "linum-compensate-attenuation-inplace")
    from scripts.attenuation.linum_compensate_attenuation_inplace import main

    main()


def linum_compensate_psf_from_model() -> None:
    _warn("linum_compensate_psf_from_model.py", "linum-compensate-psf-from-model")
    from scripts.attenuation.linum_compensate_psf_from_model import main

    main()


def linum_compensate_psf_model_free() -> None:
    _warn("linum_compensate_psf_model_free.py", "linum-compensate-psf-model-free")
    from scripts.attenuation.linum_compensate_psf_model_free import main

    main()


def linum_compute_attenuation() -> None:
    _warn("linum_compute_attenuation.py", "linum-compute-attenuation")
    from scripts.attenuation.linum_compute_attenuation import main

    main()


# --- stitching ---
def linum_align_mosaics_3d_from_shifts() -> None:
    _warn("linum_align_mosaics_3d_from_shifts.py", "linum-align-mosaics-3d-from-shifts")
    from scripts.stitching.linum_align_mosaics_3d_from_shifts import main

    main()


def linum_align_to_ras() -> None:
    _warn("linum_align_to_ras.py", "linum-align-to-ras")
    from scripts.stitching.linum_align_to_ras import main

    main()


def linum_apply_slices_transforms() -> None:
    _warn("linum_apply_slices_transforms.py", "linum-apply-slices-transforms")
    from scripts.stitching.linum_apply_slices_transforms import main

    main()


def linum_estimate_global_transform() -> None:
    _warn("linum_estimate_global_transform.py", "linum-estimate-global-transform")
    from scripts.stitching.linum_estimate_global_transform import main

    main()


def linum_estimate_transform() -> None:
    _warn("linum_estimate_transform.py", "linum-estimate-transform")
    from scripts.stitching.linum_estimate_transform import main

    main()


def linum_refine_manual_transforms() -> None:
    _warn("linum_refine_manual_transforms.py", "linum-refine-manual-transforms")
    from scripts.stitching.linum_refine_manual_transforms import main

    main()


def linum_register_pairwise() -> None:
    _warn("linum_register_pairwise.py", "linum-register-pairwise")
    from scripts.stitching.linum_register_pairwise import main

    main()


def linum_stitch_2d() -> None:
    _warn("linum_stitch_2d.py", "linum-stitch-2d")
    from scripts.stitching.linum_stitch_2d import main

    main()


def linum_stitch_3d() -> None:
    _warn("linum_stitch_3d.py", "linum-stitch-3d")
    from scripts.stitching.linum_stitch_3d import main

    main()


def linum_stitch_3d_refined() -> None:
    _warn("linum_stitch_3d_refined.py", "linum-stitch-3d-refined")
    from scripts.stitching.linum_stitch_3d_refined import main

    main()


# --- stacking ---
def linum_estimate_xy_shift_from_metadata() -> None:
    _warn("linum_estimate_xy_shift_from_metadata.py", "linum-estimate-xy-shift-from-metadata")
    from scripts.stacking.linum_estimate_xy_shift_from_metadata import main

    main()


def linum_interpolate_missing_slice() -> None:
    _warn("linum_interpolate_missing_slice.py", "linum-interpolate-missing-slice")
    from scripts.stacking.linum_interpolate_missing_slice import main

    main()


def linum_stack_slices_2d() -> None:
    _warn("linum_stack_slices_2d.py", "linum-stack-slices-2d")
    from scripts.stacking.linum_stack_slices_2d import main

    main()


def linum_stack_slices_3d() -> None:
    _warn("linum_stack_slices_3d.py", "linum-stack-slices-3d")
    from scripts.stacking.linum_stack_slices_3d import main

    main()


def linum_stack_slices_motor() -> None:
    _warn("linum_stack_slices_motor.py", "linum-stack-slices-motor")
    from scripts.stacking.linum_stack_slices_motor import main

    main()


# --- visualization ---
def linum_aip() -> None:
    _warn("linum_aip.py", "linum-aip")
    from scripts.visualization.linum_aip import main

    main()


def linum_aip_png() -> None:
    _warn("linum_aip_png.py", "linum-aip-png")
    from scripts.visualization.linum_aip_png import main

    main()


def linum_basic_preview() -> None:
    _warn("linum_basic_preview.py", "linum-basic-preview")
    from scripts.visualization.linum_basic_preview import main

    main()


def linum_screenshot_omezarr() -> None:
    _warn("linum_screenshot_omezarr.py", "linum-screenshot-omezarr")
    from scripts.visualization.linum_screenshot_omezarr import main

    main()


def linum_screenshot_omezarr_annotated() -> None:
    _warn("linum_screenshot_omezarr_annotated.py", "linum-screenshot-omezarr-annotated")
    from scripts.visualization.linum_screenshot_omezarr_annotated import main

    main()


def linum_view_oct_raw_tile() -> None:
    _warn("linum_view_oct_raw_tile.py", "linum-view-oct-raw-tile")
    from scripts.visualization.linum_view_oct_raw_tile import main

    main()


def linum_view_omezarr() -> None:
    _warn("linum_view_omezarr.py", "linum-view-omezarr")
    from scripts.visualization.linum_view_omezarr import main

    main()


def linum_view_zarr() -> None:
    _warn("linum_view_zarr.py", "linum-view-zarr")
    from scripts.visualization.linum_view_zarr import main

    main()


# --- analysis ---
def linum_analyze_shifts() -> None:
    _warn("linum_analyze_shifts.py", "linum-analyze-shifts")
    from scripts.analysis.linum_analyze_shifts import main

    main()


def linum_assess_slice_quality() -> None:
    _warn("linum_assess_slice_quality.py", "linum-assess-slice-quality")
    from scripts.analysis.linum_assess_slice_quality import main

    main()


def linum_auto_exclude_slices() -> None:
    _warn("linum_auto_exclude_slices.py", "linum-auto-exclude-slices")
    from scripts.analysis.linum_auto_exclude_slices import main

    main()


def linum_detect_focal_curvature() -> None:
    _warn("linum_detect_focal_curvature.py", "linum-detect-focal-curvature")
    from scripts.analysis.linum_detect_focal_curvature import main

    main()


def linum_detect_rehoming() -> None:
    _warn("linum_detect_rehoming.py", "linum-detect-rehoming")
    from scripts.analysis.linum_detect_rehoming import main

    main()


def linum_generate_pipeline_report() -> None:
    _warn("linum_generate_pipeline_report.py", "linum-generate-pipeline-report")
    from scripts.analysis.linum_generate_pipeline_report import main

    main()


def linum_generate_slice_config() -> None:
    _warn("linum_generate_slice_config.py", "linum-generate-slice-config")
    from scripts.analysis.linum_generate_slice_config import main

    main()


# --- segmentation ---
def linum_download_allen() -> None:
    _warn("linum_download_allen.py", "linum-download-allen")
    from scripts.segmentation.linum_download_allen import main

    main()


def linum_segment_brain_3d() -> None:
    _warn("linum_segment_brain_3d.py", "linum-segment-brain-3d")
    from scripts.segmentation.linum_segment_brain_3d import main

    main()


# --- utils ---
def linum_benchmark_kvikio_zarr() -> None:
    _warn("linum_benchmark_kvikio_zarr.py", "linum-benchmark-kvikio-zarr")
    from scripts.utils.linum_benchmark_kvikio_zarr import main

    main()


def linum_clean_raw_data() -> None:
    _warn("linum_clean_raw_data.py", "linum-clean-raw-data")
    from scripts.utils.linum_clean_raw_data import main

    main()


def linum_clip_percentile() -> None:
    _warn("linum_clip_percentile.py", "linum-clip-percentile")
    from scripts.utils.linum_clip_percentile import main

    main()


def linum_estimate_slices_transforms_gui() -> None:
    _warn("linum_estimate_slices_transforms_gui.py", "linum-estimate-slices-transforms-gui")
    from scripts.utils.linum_estimate_slices_transforms_gui import main

    main()


def linum_export_manual_align() -> None:
    _warn("linum_export_manual_align.py", "linum-export-manual-align")
    from scripts.utils.linum_export_manual_align import main

    main()


def linum_gpu_info() -> None:
    _warn("linum_gpu_info.py", "linum-gpu-info")
    from scripts.utils.linum_gpu_info import main

    main()


# --- diagnostics ---
def linum_aggregate_dilation_analysis() -> None:
    _warn("linum_aggregate_dilation_analysis.py", "linum-aggregate-dilation-analysis")
    from scripts.diagnostics.linum_aggregate_dilation_analysis import main

    main()


def linum_analyze_acquisition_rotation() -> None:
    _warn("linum_analyze_acquisition_rotation.py", "linum-analyze-acquisition-rotation")
    from scripts.diagnostics.linum_analyze_acquisition_rotation import main

    main()


def linum_analyze_registration_transforms() -> None:
    _warn("linum_analyze_registration_transforms.py", "linum-analyze-registration-transforms")
    from scripts.diagnostics.linum_analyze_registration_transforms import main

    main()


def linum_analyze_stitch_affine() -> None:
    _warn("linum_analyze_stitch_affine.py", "linum-analyze-stitch-affine")
    from scripts.diagnostics.linum_analyze_stitch_affine import main

    main()


def linum_analyze_tile_dilation() -> None:
    _warn("linum_analyze_tile_dilation.py", "linum-analyze-tile-dilation")
    from scripts.diagnostics.linum_analyze_tile_dilation import main

    main()


def linum_benchmark_gpu() -> None:
    _warn("linum_benchmark_gpu.py", "linum-benchmark-gpu")
    from scripts.diagnostics.linum_benchmark_gpu import main

    main()


def linum_benchmark_n4_gpu() -> None:
    _warn("linum_benchmark_n4_gpu.py", "linum-benchmark-n4-gpu")
    from scripts.diagnostics.linum_benchmark_n4_gpu import main

    main()


def linum_compare_stitching() -> None:
    _warn("linum_compare_stitching.py", "linum-compare-stitching")
    from scripts.diagnostics.linum_compare_stitching import main

    main()


def linum_diagnose_pipeline() -> None:
    _warn("linum_diagnose_pipeline.py", "linum-diagnose-pipeline")
    from scripts.diagnostics.linum_diagnose_pipeline import main

    main()


def linum_diagnose_reconstruction() -> None:
    _warn("linum_diagnose_reconstruction.py", "linum-diagnose-reconstruction")
    from scripts.diagnostics.linum_diagnose_reconstruction import main

    main()


def linum_n4_gpu_visual_compare() -> None:
    _warn("linum_n4_gpu_visual_compare.py", "linum-n4-gpu-visual-compare")
    from scripts.diagnostics.linum_n4_gpu_visual_compare import main

    main()


def linum_stack_motor_only() -> None:
    _warn("linum_stack_motor_only.py", "linum-stack-motor-only")
    from scripts.diagnostics.linum_stack_motor_only import main

    main()


def linum_stitch_motor_only() -> None:
    _warn("linum_stitch_motor_only.py", "linum-stitch-motor-only")
    from scripts.diagnostics.linum_stitch_motor_only import main

    main()


def linum_suggest_params() -> None:
    _warn("linum_suggest_params.py", "linum-suggest-params")
    from scripts.diagnostics.linum_suggest_params import main

    main()
