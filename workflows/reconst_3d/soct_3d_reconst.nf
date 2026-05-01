#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * 3D RECONSTRUCTION PIPELINE FOR SERIAL OCT DATA
 *
 * Input:  Directory containing mosaic_grid*.ome.zarr files + shifts_xy.csv
 * Output: 3D OME-Zarr volume with multi-resolution pyramid
 *
 * Channel patterns and authoring conventions: docs/NEXTFLOW_WORKFLOWS.md
 *
 * Helper functions (slice ID parsing, path utilities, CLI flag builders, stack
 * option builders) live in ./lib/Helpers.groovy and are auto-loaded by
 * Nextflow. Call sites use the `Helpers.` prefix.
 */

// =============================================================================
// SUB-WORKFLOW INCLUDES
// =============================================================================

// Diagnostic processes (analyze_rotation_drift, stitch_motor_only, stitch_refined,
// compare_stitching, stack_motor_only, analyze_acquisition_rotation) live in
// ./diagnostics.nf and are gated below by `params.diagnostic_mode` and
// per-stage flags.
include {
    analyze_rotation_drift;
    stitch_motor_only;
    stitch_refined;
    compare_stitching;
    stack_motor_only;
    analyze_acquisition_rotation;
} from './diagnostics.nf'

// =============================================================================
// PROCESSES
// =============================================================================

// -----------------------------------------------------------------------------
// Utility Processes
// -----------------------------------------------------------------------------

process README {
    publishDir { "${params.output}/${task.process}" }, mode: 'move'

    output:
    path "readme.txt"

    script:
    """
    echo "3D reconstruction pipeline" >> readme.txt
    echo "" >> readme.txt
    echo "[Params]" >> readme.txt
    for p in ${params}; do echo " \$p" >> readme.txt; done
    echo "" >> readme.txt
    echo "[Command-line]" >> readme.txt
    echo "${workflow.commandLine}" >> readme.txt
    echo "" >> readme.txt
    echo "[Configuration files]" >> readme.txt
    for c in ${workflow.configFiles}; do echo " \$c" >> readme.txt; done
    """

    stub:
    """
    touch readme.txt
    """
}

process analyze_shifts {
    input:
    path(shifts_file)

    output:
    path "shifts_analysis/*"

    script:
    """
    linum_analyze_shifts.py ${shifts_file} shifts_analysis \
        --resolution ${params.resolution} \
        --iqr_multiplier ${params.outlier_iqr_multiplier}
    """

    stub:
    """
    mkdir -p shifts_analysis
    touch shifts_analysis/placeholder.txt
    """
}

process generate_report {
    publishDir "${params.output}", mode: 'copy'

    input:
    tuple path(zarr), path(zip), path(png), path(annotated_png)
    val subject_name

    output:
    path "${subject_name}_quality_report.${params.report_format ?: 'html'}"

    script:
    def fmt          = params.report_format ?: 'html'
    def verbose_flag = params.report_verbose ? "--verbose" : ""
    def overview_arg = png          ? "--overview_png ${png}"          : ""
    def annotated_arg = annotated_png ? "--annotated_png ${annotated_png}" : ""
    """
    linum_generate_pipeline_report.py ${params.output} ${subject_name}_quality_report.${fmt} \
        --title "Quality Report: ${subject_name}" \
        --format ${fmt} ${verbose_flag} ${overview_arg} ${annotated_arg}
    """

    stub:
    """
    touch ${subject_name}_quality_report.${params.report_format ?: 'html'}
    """
}

// -----------------------------------------------------------------------------
// Preprocessing Processes
// -----------------------------------------------------------------------------

process resample_mosaic_grid {
    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_resampled.ome.zarr")

    script:
    def gpu_flag = params.use_gpu ? "--use_gpu" : "--no-use_gpu"
    """
    linum_resample_mosaic_grid.py ${mosaic_grid} "mosaic_grid_z${slice_id}_resampled.ome.zarr" \
        -r ${params.resolution} ${gpu_flag} -v
    """

    stub:
    """
    mkdir -p mosaic_grid_z${slice_id}_resampled.ome.zarr
    """
}

process fix_focal_curvature {
    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_focal_fix.ome.zarr")

    script:
    def gpu_flag = params.use_gpu ? "--use_gpu" : "--no-use_gpu"
    """
    linum_detect_focal_curvature.py ${mosaic_grid} "mosaic_grid_z${slice_id}_focal_fix.ome.zarr" ${gpu_flag}
    """

    stub:
    """
    mkdir -p mosaic_grid_z${slice_id}_focal_fix.ome.zarr
    """
}

process fix_illumination {
    cpus params.processes

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_illum_fix.ome.zarr")

    script:
    def gpu_flag = params.use_gpu ? "--use_gpu" : "--no-use_gpu"
    """
    linum_fix_illumination_3d.py ${mosaic_grid} "mosaic_grid_z${slice_id}_illum_fix.ome.zarr" \
        --n_processes ${params.processes} \
        --percentile_max ${params.clip_percentile_upper} ${gpu_flag}
    """

    stub:
    """
    mkdir -p mosaic_grid_z${slice_id}_illum_fix.ome.zarr
    """
}

// -----------------------------------------------------------------------------
// Stitching Processes
// -----------------------------------------------------------------------------

process estimate_global_transform {
    input:
    path("pool_input/*")
    path(slice_config)

    output:
    path("global_affine.npy"), emit: transform
    path("global_affine.json"), optional: true, emit: diagnostics

    script:
    def slice_config_arg = slice_config.name != 'NO_SLICE_CONFIG' ? "--slice_config ${slice_config}" : ""
    def histogram_arg = params.stitch_global_transform_histogram_match ? "--histogram_match" : ""
    def empty_arg = params.stitch_global_transform_max_empty_fraction != null
        ? "--max_empty_fraction ${params.stitch_global_transform_max_empty_fraction}"
        : ""
    def n_samples_arg = (params.stitch_global_transform_n_samples as int) > 0
        ? "--n_samples ${params.stitch_global_transform_n_samples as int}"
        : ""
    def include_arg = params.stitch_global_transform_slices?.trim()
        ? "--include_slice " + params.stitch_global_transform_slices.toString().split('[,\\s]+').join(' ')
        : ""
    def gpu_flag = params.use_gpu ? "--use_gpu" : "--no-use_gpu"
    """
    linum_estimate_global_transform.py pool_input global_affine.npy \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        ${slice_config_arg} \
        ${include_arg} \
        ${histogram_arg} \
        ${empty_arg} \
        ${n_samples_arg} \
        --seed ${params.stitch_global_transform_seed} \
        --diagnostics_json global_affine.json \
        -f ${gpu_flag}
    """

    stub:
    """
    touch global_affine.npy
    touch global_affine.json
    """
}

process stitch_3d_with_refinement {
    publishDir { "${params.output}/${task.process}" }, mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(mosaic_grid), path(input_transform)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr"), emit: stitched
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def transform_arg = input_transform.name != 'NO_TRANSFORM' ? "--input_transform ${input_transform}" : ""
    """
    linum_stitch_3d_refined.py ${mosaic_grid} "slice_z${slice_id}_stitch_3d.ome.zarr" \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        --blending_method ${params.stitch_blending_method} \
        --refinement_mode blend_shift \
        --max_refinement_px ${params.max_blend_refinement_px} \
        ${transform_arg} \
        -f
    """

    stub:
    """
    mkdir -p slice_z${slice_id}_stitch_3d.ome.zarr
    """
}

process generate_stitch_preview {
    publishDir "${params.output}/previews/stitched_slices", mode: 'copy'

    input:
    tuple val(slice_id), path(stitched_slice)

    output:
    path "slice_z${slice_id}_stitched.png"

    script:
    """
    linum_screenshot_omezarr.py ${stitched_slice} "slice_z${slice_id}_stitched.png" \
        --z_slice 0
    """

    stub:
    """
    touch slice_z${slice_id}_stitched.png
    """
}

// -----------------------------------------------------------------------------
// Correction Processes
// -----------------------------------------------------------------------------

process beam_profile_correction {
    publishDir { "${params.output}/${task.process}" }, mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(slice_3d)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_axial_corr.ome.zarr"), emit: corrected
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    """
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_axial_corr.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper}
    """

    stub:
    """
    mkdir -p slice_z${slice_id}_axial_corr.ome.zarr
    """
}

process crop_interface {
    publishDir { "${params.output}/${task.process}" }, mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_crop_interface.ome.zarr"), emit: cropped
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    """
    linum_crop_3d_mosaic_below_interface.py ${image} "slice_z${slice_id}_crop_interface.ome.zarr" \
        --depth ${params.crop_interface_out_depth} \
        --crop_before_interface \
        --percentile_max ${params.clip_percentile_upper}
    """

    stub:
    """
    mkdir -p slice_z${slice_id}_crop_interface.ome.zarr
    """
}

process normalize {
    publishDir { "${params.output}/${task.process}" }, mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_normalize.ome.zarr"), emit: normalized
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def gpu_flag = params.use_gpu ? "--use_gpu" : "--no-use_gpu"
    """
    linum_normalize_intensities_per_slice.py ${image} "slice_z${slice_id}_normalize.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper} ${gpu_flag}
    """

    stub:
    """
    mkdir -p slice_z${slice_id}_normalize.ome.zarr
    """
}

// -----------------------------------------------------------------------------
// Alignment Processes
// -----------------------------------------------------------------------------

process detect_rehoming_events {
    input:
    tuple path(shifts_csv), path(slice_config_in)

    output:
    path "shifts_xy_clean.csv",           emit: corrected_shifts
    path "slice_config.csv",              optional: true, emit: slice_config
    path "diagnostics/*",                 optional: true, emit: diagnostics

    script:
    def diag_arg = params.rehoming_diagnostics ? "--diagnostics diagnostics" : ""
    def frac_arg = params.rehoming_return_fraction ? "--return_fraction ${params.rehoming_return_fraction}" : ""
    def tile_fov_arg = params.tile_fov_mm ? "--tile_fov_mm ${params.tile_fov_mm}" : ""
    def tile_tol_arg = (params.tile_fov_mm && params.tile_fov_tolerance != null) ? "--tile_fov_tolerance ${params.tile_fov_tolerance}" : ""
    def max_shift_arg = params.rehoming_max_shift_mm ? "--max_shift_mm ${params.rehoming_max_shift_mm}" : ""
    def sc_args = slice_config_in.name != 'NO_SLICE_CONFIG'
        ? "--slice_config_in ${slice_config_in} --slice_config_out slice_config.csv"
        : ""
    """
    linum_detect_rehoming.py ${shifts_csv} shifts_xy_clean.csv \
        ${frac_arg} ${max_shift_arg} ${tile_fov_arg} ${tile_tol_arg} ${diag_arg} \
        ${sc_args}
    """

    stub:
    """
    printf 'fixed_id,moving_id,x_shift,y_shift,x_shift_mm,y_shift_mm,reliable\n' > shifts_xy_clean.csv
    """
}

// Auto-assess slice quality after normalization. An existing slice_config.csv
// (when supplied) is merged so manually-excluded slices stay excluded.
// See docs/NEXTFLOW_WORKFLOWS.md "Authoring Notes" for the two-input pattern.
process auto_assess_quality {
    input:
    path "inputs/*"
    path existing_slice_config

    output:
    path "slice_config.csv", emit: slice_config

    script:
    def update_args = existing_slice_config.name != 'NO_SLICE_CONFIG'
        ? "--update_existing --existing_config ${existing_slice_config}"
        : ""
    """
    linum_assess_slice_quality.py inputs slice_config.csv \\
        --min_quality ${params.auto_assess_min_quality} \\
        --exclude_first ${params.auto_assess_exclude_first} \\
        --roi_size ${params.auto_assess_roi_size} \\
        --processes ${params.processes} \\
        ${update_args} \\
        -f
    """

    stub:
    """
    printf 'slice_id,use\n' > slice_config.csv
    """
}

process bring_to_common_space {
    input:
    tuple path("inputs/*"), path("shifts_xy.csv"), path(slice_config)

    output:
    path "*.ome.zarr"

    script:
    def slice_config_arg = slice_config.name != 'NO_SLICE_CONFIG' ? "--slice_config ${slice_config}" : ""

    def excluded_args = params.common_space_excluded_slice_mode ?
        "--excluded_slice_mode ${params.common_space_excluded_slice_mode} --excluded_slice_window ${params.common_space_excluded_slice_window}" : ""

    def refine_arg = params.common_space_refine_unreliable ? "--refine_unreliable" : ""
    def discrepancy_arg = (params.common_space_refine_unreliable && params.common_space_refine_max_discrepancy_px > 0) ?
        "--refine_max_discrepancy_px ${params.common_space_refine_max_discrepancy_px}" : ""
    def min_corr_arg = (params.common_space_refine_unreliable && params.common_space_refine_min_correlation > 0) ?
        "--refine_min_correlation ${params.common_space_refine_min_correlation}" : ""

    """
    linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv common_space \
        ${slice_config_arg} ${excluded_args} ${refine_arg} ${discrepancy_arg} ${min_corr_arg}
    mv common_space/* .
    """

    stub:
    """
    for f in inputs/*.ome.zarr; do
        [ -e "\$f" ] || continue
        mkdir -p "\$(basename \$f)"
    done
    """
}

process generate_common_space_preview {
    publishDir "${params.output}/common_space_previews", mode: 'copy'

    input:
    tuple val(slice_id), path(slice_zarr)

    output:
    path "slice_z${slice_id}_preview.png"

    script:
    """
    linum_screenshot_omezarr.py ${slice_zarr} "slice_z${slice_id}_preview.png"
    """

    stub:
    """
    touch slice_z${slice_id}_preview.png
    """
}

// Interpolate a single missing slice via z-aware morphing (zmorph).
// On gate failure the zarr is omitted (hard skip); see
// docs/SLICE_INTERPOLATION_FEATURE.md for the full failure policy.
process interpolate_missing_slice {
    input:
    tuple val(missing_slice_id), path(slice_before), path(slice_after)

    output:
    path "slice_z${missing_slice_id}_interpolated.ome.zarr", optional: true, emit: zarr
    path "slice_z${missing_slice_id}_interpolated_preview.png", optional: true, emit: preview
    path "slice_z${missing_slice_id}_interpolated_diagnostics.json", emit: diagnostics
    path "slice_z${missing_slice_id}_manifest.csv", emit: manifest

    script:
    def preview_opt = params.interpolation_preview ? "--preview slice_z${missing_slice_id}_interpolated_preview.png" : ""
    def slab_opt = params.interpolation_reference_slab_size ? "--reference_slab_size ${params.interpolation_reference_slab_size}" : ""
    def fg_opt = params.interpolation_min_foreground_fraction != null ? "--min_foreground_fraction ${params.interpolation_min_foreground_fraction}" : ""
    def ncc_opt = params.interpolation_min_ncc_improvement != null ? "--min_ncc_improvement ${params.interpolation_min_ncc_improvement}" : ""
    """
    linum_interpolate_missing_slice.py ${slice_before} ${slice_after} \
        "slice_z${missing_slice_id}_interpolated.ome.zarr" \
        --method ${params.interpolation_method} \
        --blend_method ${params.interpolation_blend_method} \
        --registration_metric ${params.interpolation_registration_metric} \
        --max_iterations ${params.interpolation_max_iterations} \
        --overlap_search_window ${params.interpolation_overlap_search_window} \
        --min_overlap_correlation ${params.interpolation_min_overlap_correlation} \
        ${slab_opt} \
        ${fg_opt} \
        ${ncc_opt} \
        --slice_id ${missing_slice_id} \
        --diagnostics slice_z${missing_slice_id}_interpolated_diagnostics.json \
        --manifest_entry slice_z${missing_slice_id}_manifest.csv \
        ${preview_opt}
    """

    stub:
    """
    mkdir -p slice_z${missing_slice_id}_interpolated.ome.zarr
    echo '{}' > slice_z${missing_slice_id}_interpolated_diagnostics.json
    printf 'slice_id,interpolated\n${missing_slice_id},true\n' > slice_z${missing_slice_id}_manifest.csv
    """
}

// Merge per-slice interpolation manifest fragments into slice_config.csv.
// See docs/NEXTFLOW_WORKFLOWS.md "Authoring Notes" for the two-input pattern.
process finalise_interpolation {
    publishDir "${params.output}", mode: 'copy'

    input:
    path slice_config
    path "fragments/*"

    output:
    path "slice_config_final.csv"

    script:
    """
    linum_interpolate_missing_slice.py --finalise \\
        --slice_config_in ${slice_config} \\
        --slice_config_out slice_config_final.csv \\
        --fragments fragments
    """

    stub:
    """
    printf 'slice_id,use\n' > slice_config_final.csv
    """
}

// -----------------------------------------------------------------------------
// Registration Processes
// -----------------------------------------------------------------------------

process register_pairwise {
    input:
    tuple path(fixed_vol), path(moving_vol)

    output:
    path "*"

    script:
    def rotation_flag = params.registration_transform == 'translation' ? "--no_rotation" : "--enable_rotation"
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    linum_register_pairwise.py ${fixed_vol} ${moving_vol} \$dirname \
        --slicing_interval_mm ${params.registration_slicing_interval_mm} \
        --search_range_mm ${params.registration_allowed_drifting_mm} \
        --moving_z_index ${params.moving_slice_first_index} \
        --max_rotation_deg ${params.registration_max_rotation} \
        --max_translation_px ${params.registration_max_translation} \
        --initial_alignment ${params.registration_initial_alignment} \
        ${rotation_flag}
    """

    stub:
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    mkdir -p \$dirname
    touch \$dirname/transform.tfm
    """
}

// Optional: re-register slice pairs that have a manual transform, using the
// manual alignment as initialisation.  Produces a refined transform that
// combines the manual correction with a tight image-based residual correction.
// Only runs when params.refine_manual_transforms = true.
process refine_manual_transforms {
    input:
    tuple path(fixed_vol), path(moving_vol), path("auto_transforms")

    output:
    path "*"

    script:
    def manual_dir_opt = params.manual_transforms_dir ? "--manual_transforms_dir ${params.manual_transforms_dir}" : ""
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    linum_refine_manual_transforms.py ${fixed_vol} ${moving_vol} auto_transforms \$dirname \
        --max_translation_px ${params.refine_max_translation_px} \
        --max_rotation_deg ${params.refine_max_rotation_deg} \
        ${manual_dir_opt} -f
    """

    stub:
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    mkdir -p \$dirname
    touch \$dirname/transform.tfm
    """
}

// Auto-exclude clusters of consecutive low-quality registrations by stamping
// auto_excluded/auto_exclude_reason into slice_config.csv; stack reads them
// via --slice_config and treats those slices as motor-only.
// See docs/NEXTFLOW_WORKFLOWS.md "Authoring Notes" for the two-input pattern.
process auto_exclude_slices {
    input:
    path "transforms/*"
    path slice_config_in

    output:
    path "slice_config.csv", emit: slice_config

    script:
    """
    linum_auto_exclude_slices.py transforms ${slice_config_in} slice_config.csv \
        --consecutive_threshold ${params.auto_exclude_consecutive} \
        --z_corr_threshold ${params.auto_exclude_z_corr}
    """

    stub:
    """
    printf 'slice_id,use\n' > slice_config.csv
    """
}

// -----------------------------------------------------------------------------
// Stacking Processes
// -----------------------------------------------------------------------------

// Export lightweight data package for the manual alignment tool.
// Produces AIP images and copies pairwise transforms into a self-contained
// directory that can be downloaded and opened by the manual alignment widget.
process make_manual_align_package {
    input:
    tuple path("slices/*"), path("transforms/*")

    output:
    path("manual_align_package"), emit: pkg

    script:
    // When interpolation is enabled, interpolated slices live in a separate
    // publish dir (interpolate_missing_slice/) rather than bring_to_common_space/.
    // Pass that directory so the plugin's SSH reader can locate them.
    def interp_dir_opt = params.interpolate_missing_slices ?
        "--interpolated_slices_remote_dir ${params.output}/interpolate_missing_slice" : ""
    """
    linum_export_manual_align.py slices transforms manual_align_package \
        --level ${params.manual_align_level} \
        --slices_remote_dir ${params.output}/bring_to_common_space \
        ${interp_dir_opt}
    """

    stub:
    """
    mkdir -p manual_align_package
    """
}

// Stacking: assembles common-space slices into a 3D volume using motor positions
// for XY placement, pairwise registration for rotation/translation refinement,
// and correlation or physics-based Z-matching.
// publishDir mode is conditional: 'symlink' when a downstream step will produce
// the final output (preserves work-dir files for -resume); 'move' when this is last.
process stack {
    publishDir { "${params.output}/${task.process}" },
        mode: (params.correct_bias_field || params.align_to_ras_enabled) ? 'symlink' : 'move',
        saveAs: { fn -> fn.endsWith('.ome.zarr') ? null : fn }

    input:
    tuple path("slices/*"), path(shifts_file), path("transforms/*"), path(slice_config), val(subject_name), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png"), emit: volume
    path("*_metrics.json"), optional: true, emit: metrics
    path("z_matches.csv"), optional: true, emit: z_matches
    path("stacking_decisions.csv"), optional: true, emit: stacking_decisions

    script:
    def gpu_flag = params.use_gpu ? " --use_gpu" : " --no-use_gpu"
    def options = Helpers.stackBlendingArgs(params) +
                  Helpers.stackZMatchingArgs(params) +
                  Helpers.stackPairwiseTransformArgs(params) +
                  Helpers.stackSliceConfigArg(slice_config) +
                  Helpers.stackManualOverrideArg(params) +
                  Helpers.stackCumulativeArgs(params) +
                  Helpers.stackSmoothingArgs(params) +
                  " --no_xy_shift" +  // slices are already in common space
                  gpu_flag +
                  Helpers.pyramidArgs(params)

    def annotated_args = Helpers.annotatedScreenshotArgs(params, slice_ids_str)
    """
    linum_stack_slices_motor.py slices ${shifts_file} ${subject_name}.ome.zarr ${options}
    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr
    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png
    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png ${annotated_args}
    """

    stub:
    """
    mkdir -p ${subject_name}.ome.zarr
    touch ${subject_name}.ome.zarr.zip
    touch ${subject_name}.png
    touch ${subject_name}_annotated.png
    """
}

// Post-stacking N4 bias field correction.
// 'symlink' when align_to_ras follows; 'move' when this is the final output step.
process correct_bias_field {
    cpus params.processes

    publishDir { "${params.output}/${task.process}" },
        mode: params.align_to_ras_enabled ? 'symlink' : 'move',
        saveAs: { fn -> fn.endsWith('.ome.zarr') ? null : fn }

    input:
    tuple path(stacked_zarr), val(subject_name), val(n_slices), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png")

    script:
    def n_slices_opt = n_slices > 0 ? "--n_serial_slices ${n_slices}" : ""
    def annotated_args = Helpers.annotatedScreenshotArgs(params, slice_ids_str)
    def backend_flag = params.use_gpu ? "auto" : "cpu"
    def hm_perz_flag = params.bias_histogram_match_per_zplane ? "--histogram_match_per_zplane" : ""
    def tissue_thresh_flag = params.bias_tissue_threshold != null ? "--tissue_threshold ${params.bias_tissue_threshold}" : ""
    def zprofile_flag = params.bias_zprofile_smooth_sigma != null ? "--zprofile_smooth_sigma ${params.bias_zprofile_smooth_sigma}" : ""
    """
    linum_correct_bias_field.py ${stacked_zarr} ${subject_name}.ome.zarr \
        ${n_slices_opt} \
        --mode ${params.bias_mode} \
        --strength ${params.bias_strength} \
        --backend ${backend_flag} \
        --n_processes ${task.cpus} \
        ${hm_perz_flag} \
        ${tissue_thresh_flag} \
        ${zprofile_flag} \
        ${Helpers.pyramidArgs(params)}

    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr

    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png

    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png ${annotated_args}
    """

    stub:
    """
    mkdir -p ${subject_name}.ome.zarr
    touch ${subject_name}.ome.zarr.zip
    touch ${subject_name}.png
    touch ${subject_name}_annotated.png
    """
}

// Atlas registration to Allen Mouse Brain Atlas. Always the final step when enabled.
process align_to_ras {
    publishDir { "${params.output}/${task.process}" }, mode: 'move', saveAs: { fn ->
        fn.endsWith('.ome.zarr') ? null : fn
    }

    input:
    tuple path(stacked_zarr), path(zarr_zip), path(png), path(annotated_png)
    val subject_name

    output:
    path "${subject_name}_ras.ome.zarr"
    path "${subject_name}_ras.ome.zarr.zip"
    path "${subject_name}_ras_transform.tfm", optional: true
    path "${subject_name}_ras_preview.png", optional: true
    path "${subject_name}_ras_orientation_preview.png", optional: true

    script:
    def orientation_arg = params.ras_input_orientation ? "--input-orientation ${params.ras_input_orientation}" : ""
    def rotation_arg = params.ras_initial_rotation ? "--initial-rotation ${params.ras_initial_rotation}" : ""
    def preview_arg = params.allen_preview ? "--preview ${subject_name}_ras_preview.png" : ""
    def orientation_preview_arg = params.ras_orientation_preview ? "--orientation-preview ${subject_name}_ras_orientation_preview.png" : ""
    def ras_pyramid_opts = Helpers.pyramidArgs(params, '--n-levels')
    """
    linum_align_to_ras.py ${stacked_zarr} ${subject_name}_ras.ome.zarr \
        --allen-resolution ${params.allen_resolution} \
        --metric ${params.allen_metric} \
        --max-iterations ${params.allen_max_iterations} \
        --level ${params.allen_registration_level} \
        ${orientation_arg} ${rotation_arg} ${preview_arg} ${orientation_preview_arg} \
        ${ras_pyramid_opts}
    zip -r ${subject_name}_ras.ome.zarr.zip ${subject_name}_ras.ome.zarr
    """

    stub:
    """
    mkdir -p ${subject_name}_ras.ome.zarr
    touch ${subject_name}_ras.ome.zarr.zip
    """
}

// =============================================================================
// MAIN WORKFLOW
// =============================================================================

workflow {
    README()

    def inputDir = Helpers.normalizePath(params.input)
    def subject_name = Helpers.resolveSubjectName(inputDir, params.subject_name)
    log.info "Subject: ${subject_name}"
    log.info "GPU: ${params.use_gpu ? 'ENABLED' : 'DISABLED'}"

    def debugSlices = Helpers.parseDebugSlices(params.debug_slices)
    if (debugSlices) {
        log.info "DEBUG MODE: Processing only slices ${debugSlices.sort().join(', ')}"
    }

    // Shifts file
    def shifts_xy_path = params.shifts_xy ?: "${inputDir}/shifts_xy.csv"
    log.info "Shifts file: ${shifts_xy_path}"

    if (!file(shifts_xy_path).exists()) {
        error """
        Shifts file not found: ${shifts_xy_path}

        Please ensure shifts_xy.csv exists in your input directory,
        or specify the path with --shifts_xy /path/to/shifts_xy.csv
        """
    }
    // Value channel — fans out to many consumers; see "Authoring Notes" in
    // docs/NEXTFLOW_WORKFLOWS.md.
    shifts_xy = channel.value(file(shifts_xy_path))

    // Slice config (optional)
    def slice_config_path = params.slice_config ?: Helpers.joinPath(inputDir, "slice_config.csv")
    def slicesToUse = null
    if (file(slice_config_path).exists()) {
        log.info "Slice config: ${slice_config_path}"
        def parsed = Helpers.parseSliceConfig(slice_config_path)
        slicesToUse = parsed.use
        def total = slicesToUse.size() + parsed.excluded.size()
        log.info "Slice config: ${total} entries (${slicesToUse.size()} included, ${parsed.excluded.size()} excluded)"
    } else if (params.slice_config) {
        error("Slice config file not found: ${slice_config_path}")
    }

    // Discover input mosaic grids
    log.info "Looking for mosaic grids in: ${inputDir}"

    def inputDirFile = file(inputDir)
    def mosaicFiles = inputDirFile.listFiles()
        .findAll { f -> f.isDirectory() && f.name.startsWith('mosaic_grid') && f.name.endsWith('.ome.zarr') && f.name =~ /z\d+/ }
        .sort { f -> f.name }

    if (mosaicFiles.isEmpty()) {
        error("No mosaic grids found in ${inputDir}. Expected: mosaic_grid*_z00.ome.zarr")
    }

    def selectedIds = mosaicFiles.collect { f -> Helpers.extractSliceId(f) }.findAll { sid ->
        if (debugSlices != null) return debugSlices.contains(sid)
        if (slicesToUse != null) return slicesToUse.contains(sid)
        return true
    }
    def skippedCount = mosaicFiles.size() - selectedIds.size()
    if (skippedCount > 0) {
        def reason = debugSlices != null ? "debug_slices filter" : "slice_config"
        log.info "Found ${mosaicFiles.size()} mosaic grids; ${selectedIds.size()} selected, ${skippedCount} skipped (${reason})"
    } else {
        log.info "Found ${mosaicFiles.size()} mosaic grids; all selected"
    }

    inputSlices = channel
        .fromList(mosaicFiles)
        .map { f -> Helpers.toSliceTuple(f) }
        .filter { slice_id, _files ->
            if (debugSlices != null) {
                def included = debugSlices.contains(slice_id)
                if (!included) log.debug "Skipping slice ${slice_id} (not in debug_slices)"
                return included
            }
            if (slicesToUse != null) return slicesToUse.contains(slice_id)
            return true
        }

    def has_slice_config = file(slice_config_path).exists() || params.auto_assess_quality
    // Value channel — consumed by auto_assess, common_space, finalise, stack.
    slice_config_channel = channel.value(
        file(slice_config_path).exists() ? file(slice_config_path) : file('NO_SLICE_CONFIG')
    )

    if (params.analyze_shifts) {
        analyze_shifts(shifts_xy)
    }

    // Stage 1: Preprocessing
    resampled = params.resolution > 0 ? resample_mosaic_grid(inputSlices) : inputSlices
    focal_fixed = params.fix_curvature_enabled ? fix_focal_curvature(resampled) : resampled
    illum_fixed = params.fix_illum_enabled ? fix_illumination(focal_fixed) : focal_fixed

    // Stage 2: XY Stitching (image-registration-based blend refinement)
    if (params.stitch_global_transform) {
        pooled_mosaics = illum_fixed.map { _id, p -> p }.collect()
        estimate_global_transform(pooled_mosaics, slice_config_channel)
        stitch_inputs = illum_fixed.combine(estimate_global_transform.out.transform)
    } else {
        // Value channel so the placeholder can fan out to every per-slice tuple.
        no_transform = channel.value(file('NO_TRANSFORM'))
        stitch_inputs = illum_fixed.combine(no_transform)
    }
    stitch_3d_with_refinement(stitch_inputs)
    stitched_slices = stitch_3d_with_refinement.out.stitched

    if (params.stitch_preview) {
        generate_stitch_preview(stitched_slices)
    }

    // Stage 3: Corrections
    beam_profile_correction(stitched_slices)
    crop_interface(beam_profile_correction.out.corrected)
    normalize(crop_interface.out.cropped)

    // Stage 3.5: Auto slice quality assessment (optional). Generates a
    // slice_config.csv that marks degraded slices; an existing static
    // slice_config.csv is merged so manually-excluded slices stay excluded.
    // current_slice_config = the latest slice_config as it flows through the
    // pipeline; rebound by auto_assess / detect_rehoming when each runs.
    current_slice_config = slice_config_channel
    if (params.auto_assess_quality) {
        auto_assess_inputs = normalize.out.normalized
            .map { _id, norm_path -> norm_path }
            .collect()
        auto_assess_quality(auto_assess_inputs, slice_config_channel)
        current_slice_config = auto_assess_quality.out.slice_config
    }

    // Stage 4: Common Space Alignment.
    // detect_rehoming optionally corrects encoder-glitch spikes in the
    // shifts file and (when a real slice_config exists) stamps
    // rehomed/rehoming_reliable flags back into it.
    if (params.detect_rehoming) {
        detect_rehoming_input = shifts_xy.combine(current_slice_config)
        detect_rehoming_events(detect_rehoming_input)
        aligned_shifts = detect_rehoming_events.out.corrected_shifts
        if (has_slice_config) {
            current_slice_config = detect_rehoming_events.out.slice_config
        }
    } else {
        aligned_shifts = shifts_xy
    }

    common_space_input = normalize.out.normalized
        .toSortedList { a, b -> a[0] <=> b[0] }
        .flatten()
        .collate(2)
        .map { _meta, filename -> filename }
        .collect()
        .merge(aligned_shifts) { a, b -> tuple(a, b) }
        .merge(current_slice_config) { a, b -> tuple(a[0], a[1], b) }

    bring_to_common_space(common_space_input)

    slices_common_space = bring_to_common_space.out
        .flatten()
        .toSortedList { a, b -> a.getName() <=> b.getName() }

    if (params.common_space_preview) {
        preview_input = bring_to_common_space.out
            .flatten()
            .map { f -> Helpers.toSliceTuple(f) }
        generate_common_space_preview(preview_input)
    }

    // Stage 5: Missing Slice Interpolation (optional).
    // Single-slice gaps (use=false slices already filtered upstream) are
    // interpolated with zmorph; per-slice diagnostics are merged into
    // slice_config_final.csv. See docs/SLICE_INTERPOLATION_FEATURE.md.
    if (params.interpolate_missing_slices) {
        gaps_channel = slices_common_space
            .map { sliceList -> [Helpers.detectSingleGaps(sliceList), sliceList] }
            .flatMap { gapsAndSlices ->
                def gaps = gapsAndSlices[0]
                def sliceList = gapsAndSlices[1]
                if (gaps.isEmpty()) return []

                gaps.collect { gap ->
                    def (missingId, beforeId, afterId) = gap
                    def sliceBefore = sliceList.find { f -> f.getName().contains("slice_z${beforeId}") }
                    def sliceAfter = sliceList.find { f -> f.getName().contains("slice_z${afterId}") }
                    (sliceBefore && sliceAfter) ? tuple(missingId, sliceBefore, sliceAfter) : null
                }.findAll { item -> item != null }
            }

        interpolate_missing_slice(gaps_channel)

        // Publish slice_config_final.csv as an artifact for the report.
        // Intentionally NOT piped back into current_slice_config: when no
        // gaps exist, interpolate_missing_slice does not run and finalise's
        // output channel is empty, which would in turn empty out
        // current_slice_config and silently skip stack. Stack only reads
        // use/auto_excluded — neither column is modified here — so reading
        // the upstream config is equivalent.
        if (has_slice_config) {
            finalise_interpolation(
                current_slice_config,
                interpolate_missing_slice.out.manifest.collect(),
            )
        }

        all_slices = slices_common_space
            .mix(interpolate_missing_slice.out.zarr.collect())
            .flatten()
            .toSortedList { a, b -> a.getName() <=> b.getName() }
    } else {
        all_slices = slices_common_space
    }

    // Stage 6: Pairwise Registration
    log.info "Registering slices pairwise"

    fixed_slices = all_slices
        .map { list -> list.size() > 1 ? list.subList(0, list.size() - 1) : [] }
        .flatten()
    moving_slices = all_slices
        .map { list -> list.size() > 1 ? list.subList(1, list.size()) : [] }
        .flatten()
    pairs = fixed_slices.merge(moving_slices)

    register_pairwise(pairs)

    slices_collected = all_slices.flatten().collect()
    transforms_collected = register_pairwise.out.collect()

    // Stage 6.5: Export manual-alignment package (optional).
    if (params.export_manual_align) {
        export_input = slices_collected
            .combine(transforms_collected)
            .map { items -> Helpers.partitionSlicesAndTransforms(items) }
        make_manual_align_package(export_input)
    }

    // Stage 6.75: Refine manual transforms (optional). Re-runs pairwise
    // registration initialised from each manual transform; non-manual pairs
    // are copied unchanged. Refined outputs replace automated transforms.
    if (params.refine_manual_transforms && params.manual_transforms_dir) {
        log.info "Refining manual transforms from: ${params.manual_transforms_dir}"
        // Re-derive pairs from all_slices (value channel, safe to reuse)
        refine_fixed = all_slices
            .map { list -> list.size() > 1 ? list.subList(0, list.size() - 1) : [] }
            .flatten()
        refine_moving = all_slices
            .map { list -> list.size() > 1 ? list.subList(1, list.size()) : [] }
            .flatten()
        // Key pairs by moving zarr basename (= transform dir name)
        refine_pairs_keyed = refine_fixed
            .merge(refine_moving)
            .map { fixed, moving -> tuple(moving.getName().replace('.ome.zarr', ''), fixed, moving) }
        // Key auto transform dirs by dir name
        auto_transforms_keyed = register_pairwise.out
            .flatten()
            .filter { f -> !f.getName().endsWith('.ome.zarr') }
            .map { dir -> tuple(dir.getName(), dir) }
        // Join pairs with their corresponding auto transform dir
        refine_input = refine_pairs_keyed
            .join(auto_transforms_keyed)
            .map { _id, fixed, moving, auto_tfm -> tuple(fixed, moving, auto_tfm) }
        refine_manual_transforms(refine_input)
        transforms_for_stack = refine_manual_transforms.out.collect()
    } else {
        transforms_for_stack = transforms_collected
    }

    // Stage 7: Stacking
    log.info "Stacking slices with registration refinements"

    // Auto-exclude: detect clusters of consecutive low-quality registrations.
    // Stamps auto_excluded/auto_exclude_reason into slice_config so stack
    // sees them via --slice_config. Requires a real slice_config.
    stack_slice_config = current_slice_config
    if (params.auto_exclude_enabled && has_slice_config) {
        auto_exclude_slices(transforms_for_stack, current_slice_config)
        stack_slice_config = auto_exclude_slices.out.slice_config
    }

    // Build stack_input with `merge` (preserves list-vs-file structure of each
    // input). Earlier versions used `combine`, which flattens lists into a
    // single tuple and forced fragile filename-based dispatch in `.map`.
    stack_input = slices_collected
        .merge(shifts_xy) { s, x -> tuple(s, x) }
        .merge(transforms_for_stack) { acc, t -> tuple(acc[0], acc[1], t) }
        .merge(stack_slice_config) { acc, sc -> tuple(acc[0], acc[1], acc[2], sc) }
        .map { slices, shifts, transforms, sc ->
            tuple(slices, shifts, transforms, sc, subject_name, Helpers.extractSliceIdsString(slices))
        }

    stack(stack_input)
    stack_output = stack.out.volume
    stack_metadata = stack_input.map { _slices, _shifts, _transforms, _sc, name, ids_str ->
        tuple(name, ids_str.split(',').size(), ids_str)
    }

    // Stage 8: Bias Field Correction (optional)
    if (params.correct_bias_field) {
        log.info "Running N4 bias field correction (mode=${params.bias_mode})"
        znorm_input = stack_output
            .combine(stack_metadata)
            .map { zarr, _zip, _png, _annotated, name, n, ids_str -> tuple(zarr, name, n, ids_str) }
        correct_bias_field(znorm_input)
        final_stack_output = correct_bias_field.out
    } else {
        final_stack_output = stack_output
    }

    // Stage 9: Report Generation (optional)
    if (params.generate_report) {
        generate_report(final_stack_output, subject_name)
    }

    // Stage 10: Atlas Registration (optional)
    if (params.align_to_ras_enabled) {
        log.info "Registering to Allen Mouse Brain Atlas (RAS alignment)"
        align_to_ras(final_stack_output, subject_name)
    }

    // Stage 11: Diagnostics (optional). Toggle individually or via diagnostic_mode.
    if (params.diagnostic_mode) {
        log.info "DIAGNOSTIC MODE enabled (acq rotation, rotation drift, motor-only stitch/stack)"
    }

    if (Helpers.diagEnabled(params, 'analyze_acquisition_rotation')) {
        analyze_acquisition_rotation(shifts_xy, register_pairwise.out.collect())
    }

    if (Helpers.diagEnabled(params, 'analyze_rotation_drift')) {
        analyze_rotation_drift(register_pairwise.out.collect())
    }

    if (Helpers.diagEnabled(params, 'motor_only_stack')) {
        motor_only_stack_input = normalize.out.normalized
            .map { _id, slice_file -> slice_file }
            .collect()
        stack_motor_only(motor_only_stack_input, shifts_xy)
    }

    // motor_only_stitch is also a prerequisite for compare_stitching, so run it
    // whenever either is requested. A second `stitch_motor_only(illum_fixed)`
    // call would emit the same channel twice, which Nextflow forbids.
    def runMotorStitch = Helpers.diagEnabled(params, 'motor_only_stitch')
    def runComparison = params.compare_stitching || params.diagnostic_mode
    if (runMotorStitch || runComparison) {
        stitch_motor_only(illum_fixed)
    }

    if (runComparison) {
        log.info "Running stitching comparison (motor-only vs refined)..."

        stitch_refined(illum_fixed)

        motor_stitch_with_id = stitch_motor_only.out.map { f -> Helpers.toSliceTuple(f) }
        refined_stitch_with_id = stitch_refined.out[0].map { f -> Helpers.toSliceTuple(f) }

        comparison_input = motor_stitch_with_id
            .combine(refined_stitch_with_id, by: 0)

        compare_stitching(comparison_input)
    }
}
