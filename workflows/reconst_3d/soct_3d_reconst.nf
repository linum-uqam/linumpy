#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * 3D RECONSTRUCTION PIPELINE FOR SERIAL OCT DATA
 *
 * Input:  Directory containing mosaic_grid*.ome.zarr files + shifts_xy.csv
 * Output: 3D OME-Zarr volume with multi-resolution pyramid
 */

// -----------------------------------------------------------------------------
// Utility Processes
// -----------------------------------------------------------------------------

process README {
    publishDir "${params.output}/${task.process}", mode: 'move'

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
}

process analyze_shifts {
    publishDir "${params.output}/${task.process}", mode: 'copy'

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
}

// -----------------------------------------------------------------------------
// Diagnostic Processes
// -----------------------------------------------------------------------------

process analyze_rotation_drift {
    publishDir "${params.output}/diagnostics/rotation_analysis", mode: 'copy'

    input:
    path(reg_dirs)

    output:
    path "rotation_analysis/*"

    script:
    """
    mkdir -p register_pairwise
    for d in *; do
        if [ -d "\$d" ] && [[ "\$d" == slice_z* ]]; then
            ln -s "\$(pwd)/\$d" "register_pairwise/\$d"
        fi
    done

    linum_analyze_registration_transforms.py register_pairwise rotation_analysis \
        --resolution ${params.resolution} \
        --rotation_threshold ${params.diagnostic_rotation_threshold}
    """
}

process analyze_tile_dilation {
    publishDir "${params.output}/diagnostics/dilation_analysis/${slice_id}", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid), path(transform_xy)

    output:
    tuple val(slice_id), path("dilation_analysis_${slice_id}.json"), path("dilation_analysis_${slice_id}.png"), path("dilation_analysis_${slice_id}.txt")

    script:
    """
    linum_analyze_tile_dilation.py ${mosaic_grid} ${transform_xy} dilation_analysis \
        --resolution ${params.resolution} \
        --overlap_fraction ${params.motor_only_overlap} \
        --slice_id ${slice_id}

    mv dilation_analysis/dilation_analysis.json dilation_analysis_${slice_id}.json
    mv dilation_analysis/dilation_analysis.png dilation_analysis_${slice_id}.png
    mv dilation_analysis/dilation_analysis.txt dilation_analysis_${slice_id}.txt
    """
}

process aggregate_dilation_analysis {
    publishDir "${params.output}/diagnostics/aggregated_dilation", mode: 'copy'

    input:
    path(json_files)

    output:
    path "aggregated_dilation_analysis.json", emit: json
    path "per_slice_correction_factors.csv", emit: csv
    path "aggregated_dilation_report.txt", emit: report
    path "aggregated_dilation_analysis.png", emit: plot

    script:
    """
    mkdir -p dilation_input
    for f in *.json; do
        if [ -f "\$f" ]; then
            slice_id=\$(grep -o '"slice_id"[[:space:]]*:[[:space:]]*"[^"]*"' "\$f" | head -1 | sed 's/.*"\\([^"]*\\)".*/\\1/')
            if [ -z "\$slice_id" ]; then
                slice_id=\$(grep -o '"slice_id"[[:space:]]*:[[:space:]]*[0-9]*' "\$f" | head -1 | sed 's/.*[[:space:]]\\([0-9]*\\)/\\1/')
            fi
            if [ -n "\$slice_id" ]; then
                mkdir -p "dilation_input/\${slice_id}/dilation_analysis"
                ln -s "\$(pwd)/\$f" "dilation_input/\${slice_id}/dilation_analysis/dilation_analysis.json"
            fi
        fi
    done

    linum_aggregate_dilation_analysis.py dilation_input . \
        --pattern "*/dilation_analysis/dilation_analysis.json"
    """
}

process stitch_motor_only {
    publishDir "${params.output}/diagnostics/motor_only_stitch", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    path "slice_z${slice_id}_motor_only.ome.zarr"

    script:
    def blending = params.motor_only_stitch_blending ?: 'diffusion'
    """
    linum_stitch_motor_only.py ${mosaic_grid} "slice_z${slice_id}_motor_only.ome.zarr" \
        --overlap_fraction ${params.motor_only_overlap} \
        --blending_method ${blending}
    """
}

process stitch_refined {
    publishDir "${params.output}/diagnostics/refined_stitch", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    path "slice_z${slice_id}_refined.ome.zarr"
    path "slice_z${slice_id}_refinements.json", optional: true

    script:
    def refinement_out = params.save_refinement_data ? "--output_refinements slice_z${slice_id}_refinements.json" : ""
    """
    linum_stitch_3d_refined.py ${mosaic_grid} "slice_z${slice_id}_refined.ome.zarr" \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        --blending_method diffusion \
        --refinement_mode blend_shift \
        --max_refinement_px ${params.max_blend_refinement_px} \
        ${refinement_out} -f
    """
}

process compare_stitching {
    publishDir "${params.output}/diagnostics/stitch_comparison", mode: 'copy'

    input:
    tuple val(slice_id), path(motor_stitch), path(refined_stitch)

    output:
    path "slice_z${slice_id}_comparison/*"

    script:
    """
    linum_compare_stitching.py ${motor_stitch} ${refined_stitch} \
        "slice_z${slice_id}_comparison" \
        --label1 "Motor-only" --label2 "Refined" \
        --tile_step ${params.comparison_tile_step}
    """
}

process stack_motor_only {
    publishDir "${params.output}/diagnostics/motor_only_stack", mode: 'copy'

    input:
    path("slices/*")
    path(shifts_file)

    output:
    path "motor_only_stack.ome.zarr"
    path "motor_only_stack_preview.png", optional: true

    script:
    def blending_arg = params.motor_only_stack_blending ?: 'none'
    def preview_arg = "--preview motor_only_stack_preview.png"
    """
    linum_stack_motor_only.py slices ${shifts_file} motor_only_stack.ome.zarr \
        --blending ${blending_arg} \
        ${preview_arg}
    """
}

process run_full_diagnostics {
    publishDir "${params.output}/diagnostics", mode: 'copy'

    input:
    path(pipeline_output)

    output:
    path "full_diagnostics/*"

    script:
    """
    linum_diagnose_reconstruction.py ${pipeline_output} full_diagnostics \
        --resolution ${params.resolution} \
        --rotation_threshold ${params.diagnostic_rotation_threshold}
    """
}

process analyze_acquisition_rotation {
    publishDir "${params.output}/diagnostics/acquisition_rotation", mode: 'copy'

    input:
    path(shifts_file)
    path(reg_dirs)

    output:
    path "acquisition_rotation_analysis/*"

    script:
    """
    mkdir -p register_pairwise
    for d in *; do
        if [ -d "\$d" ] && [[ "\$d" == slice_z* ]]; then
            ln -s "\$(pwd)/\$d" "register_pairwise/\$d"
        fi
    done

    linum_analyze_acquisition_rotation.py ${shifts_file} acquisition_rotation_analysis \
        --registration_dir register_pairwise \
        --resolution ${params.resolution}
    """
}

process generate_report {
    publishDir "$params.output", mode: 'copy'

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
    def script_name = params.use_gpu ? "linum_resample_mosaic_grid_gpu.py" : "linum_resample_mosaic_grid.py"
    def gpu_flag = params.use_gpu ? "--use_gpu" : ""
    """
    ${script_name} ${mosaic_grid} "mosaic_grid_z${slice_id}_resampled.ome.zarr" \
        -r ${params.resolution} ${gpu_flag} -v
    """
}

process fix_focal_curvature {
    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_focal_fix.ome.zarr")

    script:
    """
    linum_detect_focal_curvature.py ${mosaic_grid} "mosaic_grid_z${slice_id}_focal_fix.ome.zarr"
    """
}

process fix_illumination {
    cpus params.processes

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_illum_fix.ome.zarr")

    script:
    def script_name = params.use_gpu ? "linum_fix_illumination_3d_gpu.py" : "linum_fix_illumination_3d.py"
    """
    ${script_name} ${mosaic_grid} "mosaic_grid_z${slice_id}_illum_fix.ome.zarr" \
        --n_processes ${params.processes} \
        --percentile_max ${params.clip_percentile_upper}
    """
}

// -----------------------------------------------------------------------------
// Stitching Processes
// -----------------------------------------------------------------------------

process generate_aip {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("mosaic_grid_z${slice_id}_aip.ome.zarr")

    script:
    """
    linum_aip.py ${mosaic_grid} "mosaic_grid_z${slice_id}_aip.ome.zarr"
    """
}

process estimate_xy_transformation {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(aip)

    output:
    tuple val(slice_id), path("z${slice_id}_transform_xy.npy"), emit: transform
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def script_name = params.use_gpu ? "linum_estimate_transform_gpu.py" : "linum_estimate_transform.py"
    def gpu_flag = params.use_gpu ? "--use_gpu" : ""
    def motor_flag = params.use_motor_positions_for_stitching ? "--use_motor_positions" : ""
    def overlap_arg = "--initial_overlap ${params.stitch_overlap_fraction}"
    """
    ${script_name} ${aip} "z${slice_id}_transform_xy.npy" \
        ${gpu_flag} ${motor_flag} ${overlap_arg}
    """
}

process stitch_3d {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(mosaic_grid), path(transform_xy)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr"), emit: stitched
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} "slice_z${slice_id}_stitch_3d.ome.zarr" \
        --blending_method ${params.stitch_blending_method}
    """
}

process stitch_3d_with_refinement {
    input:
    tuple val(slice_id), path(mosaic_grid)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr")

    script:
    """
    linum_stitch_3d_refined.py ${mosaic_grid} "slice_z${slice_id}_stitch_3d.ome.zarr" \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        --blending_method ${params.stitch_blending_method} \
        --refinement_mode blend_shift \
        --max_refinement_px ${params.max_blend_refinement_px} \
        -f
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
}

// -----------------------------------------------------------------------------
// Correction Processes
// -----------------------------------------------------------------------------

process beam_profile_correction {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

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
}

process crop_interface {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

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
}

process normalize {
    publishDir "${params.output}/${task.process}", mode: 'copy', pattern: "*_metrics.json"

    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_normalize.ome.zarr"), emit: normalized
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def script_name = params.use_gpu ? "linum_normalize_intensities_per_slice_gpu.py" : "linum_normalize_intensities_per_slice.py"
    def gpu_flag = params.use_gpu ? "--use_gpu" : ""
    """
    ${script_name} ${image} "slice_z${slice_id}_normalize.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper} \
        --min_contrast_fraction ${params.normalize_min_contrast} ${gpu_flag}
    """
}

// -----------------------------------------------------------------------------
// Alignment Processes
// -----------------------------------------------------------------------------

process detect_rehoming_events {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    path shifts_csv

    output:
    path "shifts_xy_clean.csv", emit: corrected_shifts
    path "diagnostics/*",       emit: diagnostics, optional: true

    script:
    def diag_arg = params.rehoming_diagnostics ? "--diagnostics diagnostics" : ""
    def frac_arg = params.rehoming_return_fraction ? "--return_fraction ${params.rehoming_return_fraction}" : ""
    def tile_fov_arg = params.tile_fov_mm ? "--tile_fov_mm ${params.tile_fov_mm}" : ""
    def max_shift_arg = params.rehoming_max_shift_mm ? "--max_shift_mm ${params.rehoming_max_shift_mm}" : ""
    """
    linum_detect_rehoming.py ${shifts_csv} shifts_xy_clean.csv ${frac_arg} ${max_shift_arg} ${tile_fov_arg} ${diag_arg}
    """
}

process auto_assess_quality {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    path "inputs/*"

    output:
    path "slice_config.csv", emit: slice_config

    script:
    """
    linum_assess_slice_quality.py inputs slice_config.csv \\
        --min_quality ${params.auto_assess_min_quality} \\
        --exclude_first ${params.auto_assess_exclude_first} \\
        --roi_size ${params.auto_assess_roi_size} \\
        --processes ${params.processes} \\
        -f
    """
}

process bring_to_common_space {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path("inputs/*"), path("shifts_xy.csv"), path(slice_config)

    output:
    path "*.ome.zarr"

    script:
    def slice_config_arg = slice_config.name != 'NO_SLICE_CONFIG' ? "--slice_config ${slice_config}" : ""

    def excluded_args = params.common_space_excluded_slice_mode ?
        "--excluded_slice_mode ${params.common_space_excluded_slice_mode} --excluded_slice_window ${params.common_space_excluded_slice_window}" : ""

    def refine_arg = params.common_space_refine_unreliable ? "--refine_unreliable" : ""

    """
    linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv common_space \
        ${slice_config_arg} ${excluded_args} ${refine_arg}
    mv common_space/* .
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
}

process interpolate_missing_slice {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple val(missing_slice_id), path(slice_before), path(slice_after)

    output:
    path "slice_z${missing_slice_id}_interpolated.ome.zarr", emit: zarr
    path "slice_z${missing_slice_id}_interpolated_preview.png", optional: true, emit: preview

    script:
    def preview_opt = params.interpolation_preview ? "--preview slice_z${missing_slice_id}_interpolated_preview.png" : ""
    """
    linum_interpolate_missing_slice.py ${slice_before} ${slice_after} \
        "slice_z${missing_slice_id}_interpolated.ome.zarr" \
        --method ${params.interpolation_method} \
        --blend_method ${params.interpolation_blend_method} \
        --registration_metric ${params.interpolation_registration_metric} \
        --max_iterations ${params.interpolation_max_iterations} \
        ${preview_opt}
    """
}

// -----------------------------------------------------------------------------
// Registration Processes
// -----------------------------------------------------------------------------

process create_registration_masks {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple val(slice_id), path(image)

    output:
    path("mask_slice_z${slice_id}.ome.zarr"), emit: masks
    path("mask_slice_z${slice_id}_preview.png"), optional: true, emit: previews
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def script_name = params.use_gpu ? "linum_create_masks_gpu.py" : "linum_create_masks.py"
    def gpu_flag = params.use_gpu ? "--use_gpu" : ""
    def normalize_flag = params.mask_normalize ? "--normalize" : ""
    def preview_flag = params.mask_preview ? "--preview mask_slice_z${slice_id}_preview.png" : ""
    def fill_holes_opt = params.mask_fill_holes ? "--fill_holes ${params.mask_fill_holes}" : ""
    """
    ${script_name} ${image} mask_slice_z${slice_id}.ome.zarr \
        --sigma ${params.mask_smoothing_sigma} \
        --selem_radius ${params.selem_radius} \
        --min_size ${params.min_size} \
        ${normalize_flag} ${fill_holes_opt} ${preview_flag} ${gpu_flag}
    """
}

process register_pairwise {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path(fixed_vol), path(moving_vol), path(moving_mask, stageAs: 'moving_mask*'), path(fixed_mask, stageAs: 'fixed_mask*')

    output:
    path "*"

    script:
    def rotation_flag = params.registration_transform == 'translation' ? "--no_rotation" : "--enable_rotation"
    def mask_opts = params.create_registration_masks ?
        "--use_masks --moving_mask ${moving_mask} --fixed_mask ${fixed_mask}" : ""
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    linum_register_pairwise.py ${fixed_vol} ${moving_vol} \$dirname \
        --slicing_interval_mm ${params.registration_slicing_interval_mm} \
        --search_range_mm ${params.registration_allowed_drifting_mm} \
        --moving_z_index ${params.moving_slice_first_index} \
        --max_rotation_deg ${params.registration_max_rotation} \
        --max_translation_px ${params.registration_max_translation} \
        ${rotation_flag} ${mask_opts}
    """
}

// -----------------------------------------------------------------------------
// Stacking Processes
// -----------------------------------------------------------------------------

// Motor-position-based stacking (shifts_xy.csv for XY + registration refinements).
// publishDir mode is conditional: 'symlink' when a downstream step will produce
// the final output (preserves work-dir files for -resume); 'move' when this is last.
process stack_motor {
    publishDir "$params.output/$task.process",
        mode: (params.normalize_z_slices || params.align_to_ras_enabled) ? 'symlink' : 'move',
        saveAs: { fn -> fn.endsWith('.ome.zarr') ? null : fn }

    input:
    tuple path("slices/*"), path(shifts_file), path("transforms/*"), val(subject_name), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png"), emit: volume
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def options = ""

    // Blending
    if (params.stack_blend_enabled) options += " --blend"
    if (params.blend_refinement_px > 0) options += " --blend_refinement_px ${params.blend_refinement_px}"
    if (params.stack_blend_z_refine_vox > 0) options += " --blend_z_refine_vox ${params.stack_blend_z_refine_vox}"

    // Z-matching
    options += " --slicing_interval_mm ${params.registration_slicing_interval_mm}"
    options += " --search_range_mm ${params.registration_allowed_drifting_mm}"
    options += " --moving_z_first_index ${params.moving_slice_first_index}"
    if (params.use_expected_z_overlap) options += " --use_expected_overlap"
    if (params.z_overlap_min_corr > 0) options += " --z_overlap_min_corr ${params.z_overlap_min_corr}"
    if (params.analyze_shifts) options += " --output_z_matches z_matches.csv"

    // Pairwise registration refinements
    if (params.apply_pairwise_transforms) {
        options += " --transforms_dir transforms"
        if (params.apply_rotation_only) options += " --rotation_only"
        options += " --max_rotation_deg ${params.max_rotation_deg}"
        if (params.skip_error_transforms) options += " --skip_error_transforms"
        if (params.skip_warning_transforms) options += " --skip_warning_transforms"
        options += " --confidence_high ${params.transform_confidence_high}"
        options += " --confidence_low ${params.transform_confidence_low}"
    }

    // Cumulative translation accumulation
    if (params.stack_accumulate_translations) {
        options += " --accumulate_translations"
        // stack_max_pairwise_translation > 0 filters clamped translations; 0 = keep all.
        // Set to 0 when skip_error_transforms=false to preserve re-homing boundary corrections.
        if (params.stack_max_pairwise_translation > 0)
            options += " --max_pairwise_translation ${params.stack_max_pairwise_translation}"
    }

    if (params.stack_smooth_window > 0) options += " --smooth_window ${params.stack_smooth_window}"

    // Slices are already in common space; skip redundant XY shifting
    options += " --no_xy_shift"

    // Pyramid
    if (params.pyramid_n_levels != null) {
        options += " --n_levels ${params.pyramid_n_levels}"
    } else {
        def base_res = params.resolution > 0 ? params.resolution : 10
        def valid_resolutions = params.pyramid_resolutions.findAll { it >= base_res }.sort()
        if (!valid_resolutions.contains(base_res)) valid_resolutions = [base_res] + valid_resolutions
        def pyramid_res_str = valid_resolutions.collect { it.toString() }.join(' ')
        options += " --pyramid_resolutions ${pyramid_res_str}"
        options += params.pyramid_make_isotropic ? " --make_isotropic" : " --no_isotropic"
    }

    def show_lines_flag = params.annotated_show_lines ? '--show_lines' : ''
    def orient = params.ras_input_orientation?.trim()?.replace("'", '') ?: ''
    def orientation_arg = orient ? "--orientation ${orient}" : ''
    """
    linum_stack_slices_motor.py slices ${shifts_file} ${subject_name}.ome.zarr ${options}
    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr
    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png
    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
        --slice_ids "${slice_ids_str}" \
        --label_every ${params.annotated_label_every} ${show_lines_flag} ${orientation_arg}
    """
}

// Traditional pairwise-registration-based stacking
process stack {
    publishDir "$params.output/$task.process", mode: 'move', saveAs: { fn ->
        fn.endsWith('.ome.zarr') ? null : fn
    }

    input:
    tuple path("mosaics/*"), path("transforms/*"), val(subject_name), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png"), emit: volume
    path("*_metrics.json"), optional: true, emit: metrics

    script:
    def options = ""

    if (params.stack_blend_enabled) {
        options += "--blend"
        if (params.stack_max_overlap > 0) options += " --overlap ${params.stack_max_overlap}"
    }
    if (params.stack_no_accumulate_transforms) options += " --no_accumulate_transforms"

    if (params.pyramid_n_levels != null) {
        options += " --n_levels ${params.pyramid_n_levels}"
    } else {
        def base_res = params.resolution > 0 ? params.resolution : 10
        def valid_resolutions = params.pyramid_resolutions.findAll { it >= base_res }.sort()
        if (!valid_resolutions.contains(base_res)) valid_resolutions = [base_res] + valid_resolutions
        def pyramid_res_str = valid_resolutions.collect { it.toString() }.join(' ')
        options += " --pyramid_resolutions ${pyramid_res_str}"
        options += params.pyramid_make_isotropic ? " --make_isotropic" : " --no_isotropic"
    }

    def show_lines_flag = params.annotated_show_lines ? '--show_lines' : ''
    def orient = params.ras_input_orientation?.trim()?.replace("'", '') ?: ''
    def orientation_arg = orient ? "--orientation ${orient}" : ''
    """
    linum_stack_slices_3d.py mosaics transforms ${subject_name}.ome.zarr ${options}
    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr
    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png
    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
        --slice_ids "${slice_ids_str}" \
        --label_every ${params.annotated_label_every} \
        ${show_lines_flag} ${orientation_arg}
    """
}

// Post-stacking Z-direction intensity normalization.
// 'symlink' when align_to_ras follows; 'move' when this is the final output step.
process normalize_z_intensity {
    publishDir "$params.output/$task.process",
        mode: params.align_to_ras_enabled ? 'symlink' : 'move',
        saveAs: { fn -> fn.endsWith('.ome.zarr') ? null : fn }

    input:
    tuple path(stacked_zarr), val(subject_name), val(n_slices), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png")

    script:
    def n_slices_opt = n_slices > 0 ? "--n_serial_slices ${n_slices}" : ""
    def show_lines_flag = params.annotated_show_lines ? '--show_lines' : ''
    def orient = params.ras_input_orientation?.trim()?.replace("'", '') ?: ''
    def orientation_arg = orient ? "--orientation ${orient}" : ''
    def znorm_mode_opts = ""
    if (params.znorm_mode == 'histogram') {
        znorm_mode_opts = "--mode histogram --strength ${params.znorm_strength} --tissue_threshold ${params.znorm_tissue_threshold}"
    } else {
        znorm_mode_opts = "--mode percentile --smooth_sigma ${params.znorm_smooth_sigma} --percentile ${params.znorm_percentile} --max_scale ${params.znorm_max_scale} --min_scale ${params.znorm_min_scale} --strength ${params.znorm_strength}"
    }
    def znorm_pyramid_opts = ""
    if (params.pyramid_n_levels != null) {
        znorm_pyramid_opts += " --n_levels ${params.pyramid_n_levels}"
    } else {
        def base_res = params.resolution > 0 ? params.resolution : 10
        def valid_resolutions = params.pyramid_resolutions.findAll { it >= base_res }.sort()
        if (!valid_resolutions.contains(base_res)) valid_resolutions = [base_res] + valid_resolutions
        def pyramid_res_str = valid_resolutions.collect { it.toString() }.join(' ')
        znorm_pyramid_opts += " --pyramid_resolutions ${pyramid_res_str}"
        znorm_pyramid_opts += params.pyramid_make_isotropic ? " --make_isotropic" : " --no_isotropic"
    }
    """
    linum_normalize_z_intensity.py ${stacked_zarr} ${subject_name}.ome.zarr \
        ${n_slices_opt} \
        ${znorm_mode_opts} \
        ${znorm_pyramid_opts}

    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr

    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png

    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
        --slice_ids "${slice_ids_str}" \
        --label_every ${params.annotated_label_every} ${show_lines_flag} ${orientation_arg}
    """
}

// Atlas registration to Allen Mouse Brain Atlas. Always the final step when enabled.
process align_to_ras {
    publishDir "$params.output/$task.process", mode: 'move', saveAs: { fn ->
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
    def ras_pyramid_opts = ""
    if (params.pyramid_n_levels != null) {
        ras_pyramid_opts += " --n-levels ${params.pyramid_n_levels}"
    } else {
        def base_res = params.resolution > 0 ? params.resolution : 10
        def valid_resolutions = params.pyramid_resolutions.findAll { it >= base_res }.sort()
        if (!valid_resolutions.contains(base_res)) valid_resolutions = [base_res] + valid_resolutions
        def pyramid_res_str = valid_resolutions.collect { it.toString() }.join(' ')
        ras_pyramid_opts += " --pyramid_resolutions ${pyramid_res_str}"
        ras_pyramid_opts += params.pyramid_make_isotropic ? " --make_isotropic" : " --no_isotropic"
    }
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
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Extract z## slice ID string from a filename; returns "unknown" if not found.
def extractSliceId(filename) {
    def name = filename instanceof Path ? filename.getName() : filename.toString()
    def matcher = name =~ /z(\d+)/
    return matcher ? matcher[0][1] : "unknown"
}

// Extract slice ID as integer; returns -1 if not found.
def extractSliceIdInt(filename) {
    def id = extractSliceId(filename)
    return id == "unknown" ? -1 : id.toInteger()
}

// Return tuple(slice_id, file) for a given file path.
def toSliceTuple(file_path) {
    tuple(extractSliceId(file_path), file_path)
}

// Return sorted, comma-separated slice IDs from a list of files (e.g. "01,02,03,05").
def extractSliceIdsString(fileList) {
    fileList
        .collect { extractSliceId(it) }
        .findAll { it != "unknown" }
        .sort { it.toInteger() }
        .join(',')
}

// Remove duplicate and trailing slashes from a path string.
def normalizePath(path) {
    return path.replaceAll('/+', '/').replaceAll('/$', '')
}

// Join path components safely.
def joinPath(base, filename) {
    return "${normalizePath(base)}/${filename}"
}

// Parse a slice_config.csv and return the set of slice IDs marked for use.
def parseSliceConfig(configPath) {
    def slicesToUse = [] as Set
    def slicesExcluded = [] as Set
    def file = new File(configPath)

    if (!file.exists()) error("Slice config file not found: ${configPath}")

    file.withReader { reader ->
        reader.readLine() // Skip header
        reader.eachLine { line ->
            def parts = line.split(',')
            if (parts.size() >= 2) {
                def sliceId = parts[0].trim()
                def use = parts[1].trim().toLowerCase()
                if (use in ['true', '1', 'yes']) slicesToUse.add(sliceId)
                else slicesExcluded.add(sliceId)
            }
        }
    }

    log.info "Slice config: ${slicesToUse.size()} to USE, ${slicesExcluded.size()} EXCLUDED"
    return slicesToUse
}

// Detect single-slice gaps in a sorted slice list.
// Returns a list of [missingId, beforeId, afterId] tuples.
def detectSingleGaps(sliceList) {
    def gaps = []
    def sliceIds = sliceList
        .collect { extractSliceIdInt(it) }
        .findAll { it >= 0 }
        .sort()

    for (int i = 0; i < sliceIds.size() - 1; i++) {
        def current = sliceIds[i]
        def next = sliceIds[i + 1]
        def gap = next - current

        if (gap == 2) {
            def missingId = String.format("%02d", current + 1)
            def beforeId = String.format("%02d", current)
            def afterId = String.format("%02d", next)
            gaps.add([missingId, beforeId, afterId])
            log.info "Gap detected: slice ${missingId} (between ${beforeId} and ${afterId})"
        } else if (gap > 2) {
            log.warn "Multiple missing slices between ${current} and ${next} - cannot interpolate"
        }
    }
    return gaps
}

// Parse debug_slices parameter; supports "25,26", "25-29", or "25,27-29".
// Returns a set of zero-padded slice IDs, or null if not specified.
def parseDebugSlices(debugSlicesStr) {
    if (!debugSlicesStr || debugSlicesStr.trim().isEmpty()) return null

    def sliceIds = [] as Set
    debugSlicesStr.split(',').each { part ->
        part = part.trim()
        if (part.contains('-')) {
            def rangeParts = part.split('-')
            if (rangeParts.size() == 2) {
                def start = rangeParts[0].trim().toInteger()
                def end = rangeParts[1].trim().toInteger()
                (start..end).each { sliceIds.add(String.format("%02d", it)) }
            }
        } else {
            sliceIds.add(String.format("%02d", part.toInteger()))
        }
    }
    return sliceIds
}

// =============================================================================
// MAIN WORKFLOW
// =============================================================================

workflow {
    README()

    def inputDir = normalizePath(params.input)

    // Resolve subject name from path if not explicitly set
    def subject_name = params.subject_name
    if (!subject_name) {
        def pathParts = inputDir.split('/')
        def subMatch = pathParts.find { it ==~ /sub-\w+/ }
        if (subMatch) {
            subject_name = subMatch
        } else {
            def inputFile = file(inputDir)
            def dirName = inputFile.getName()
            subject_name = (dirName in ['mosaic-grids', 'mosaics', 'mosaic_grids', 'input', 'data'])
                ? (inputFile.getParent()?.getName() ?: dirName)
                : dirName
        }
    }
    log.info "Subject: ${subject_name}"
    log.info "GPU: ${params.use_gpu ? 'ENABLED' : 'DISABLED'}"

    // =========================================================================
    // CONFIGURATION VALIDATION
    // Catch dangerous parameter combinations that are common causes of
    // Z-alignment failures. Logs warnings but does not abort the pipeline.
    // =========================================================================
    if (params.apply_pairwise_transforms && !params.skip_error_transforms) {
        log.warn """
[CONFIG WARNING] skip_error_transforms=false
  Transforms with overall_status='error' (failed registrations, e.g. against
  interpolated slices) will be applied. These often introduce large spurious
  rotations or translations.
  RECOMMENDATION: set skip_error_transforms=true
"""
    }
    if (params.apply_pairwise_transforms && !params.skip_warning_transforms) {
        log.warn """
[CONFIG WARNING] skip_warning_transforms=false
  Transforms with overall_status='warning' (optimizer hit boundary) will be
  applied. Their Z-offsets are unreliable and can cause Z-positioning errors.
  RECOMMENDATION: set skip_warning_transforms=true
"""
    }
    if (params.stacking_method == 'motor' && params.apply_pairwise_transforms && !params.apply_rotation_only) {
        log.warn """
[CONFIG WARNING] apply_rotation_only=false with stacking_method='motor'
  Registration XY translations will be added on top of motor positions.
  When registration fails, this drifts slices from their correct XY positions.
  RECOMMENDATION: set apply_rotation_only=true to keep XY from motor positions.
"""
    }
    if (!params.interpolate_missing_slices) {
        log.warn """
[CONFIG WARNING] interpolate_missing_slices=false
  Single-slice gaps (excluded slices, degraded sections) will create permanent
  holes in the reconstruction. Consider enabling interpolation.
  RECOMMENDATION: set interpolate_missing_slices=true
"""
    }
    if (params.stack_accumulate_translations && (params.stack_max_pairwise_translation == null || params.stack_max_pairwise_translation <= 0)) {
        log.warn """
[CONFIG WARNING] stack_accumulate_translations=true without max_pairwise_translation limit
  All pairwise translations will be accumulated, including failed registrations.
  RECOMMENDATION: set stack_max_pairwise_translation to a safe cap (e.g. 50).
"""
    }

    def debugSlices = parseDebugSlices(params.debug_slices)
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
    shifts_xy = channel.of(file(shifts_xy_path))

    // Slice config (optional)
    def slice_config_path = params.slice_config ?: joinPath(inputDir, "slice_config.csv")
    def slicesToUse = null
    if (file(slice_config_path).exists()) {
        slicesToUse = parseSliceConfig(slice_config_path)
        log.info "Slice config: ${slice_config_path}"
    } else if (params.slice_config) {
        error("Slice config file not found: ${slice_config_path}")
    }

    // Discover input mosaic grids
    log.info "Looking for mosaic grids in: ${inputDir}"

    def inputDirFile = file(inputDir)
    def mosaicFiles = inputDirFile.listFiles()
        .findAll { it.isDirectory() && it.name.startsWith('mosaic_grid') && it.name.endsWith('.ome.zarr') && it.name =~ /z\d+/ }
        .sort { it.name }

    if (mosaicFiles.isEmpty()) {
        error("No mosaic grids found in ${inputDir}. Expected: mosaic_grid*_z00.ome.zarr")
    }
    log.info "Found ${mosaicFiles.size()} mosaic grids"

    inputSlices = channel
        .fromList(mosaicFiles)
        .map { toSliceTuple(it) }
        .filter { slice_id, _files ->
            if (debugSlices != null) {
                def included = debugSlices.contains(slice_id)
                if (!included) log.debug "Skipping slice ${slice_id} (not in debug_slices)"
                return included
            }
            if (slicesToUse != null) return slicesToUse.contains(slice_id)
            return true
        }

    slice_config_channel = file(slice_config_path).exists()
        ? channel.fromPath(slice_config_path)
        : channel.of(file('NO_SLICE_CONFIG'))

    if (params.analyze_shifts) {
        analyze_shifts(shifts_xy)
    }

    // Stage 1: Preprocessing
    resampled = params.resolution > 0 ? resample_mosaic_grid(inputSlices) : inputSlices
    focal_fixed = params.fix_curvature_enabled ? fix_focal_curvature(resampled) : resampled
    illum_fixed = params.fix_illum_enabled ? fix_illumination(focal_fixed) : focal_fixed

    // Stage 2: XY Stitching
    // 'motor' path: image-registration-based blend refinement
    // 'registration' path: AIP-based XY transform estimation
    if (params.stacking_method == 'motor') {
        stitch_3d_with_refinement(illum_fixed)
        stitched_slices = stitch_3d_with_refinement.out
    } else {
        generate_aip(illum_fixed)
        estimate_xy_transformation(generate_aip.out)
        stitch_3d(illum_fixed.combine(estimate_xy_transformation.out.transform, by: 0))
        stitched_slices = stitch_3d.out.stitched
    }

    if (params.stitch_preview) {
        generate_stitch_preview(stitched_slices)
    }

    // Stage 3: Corrections
    beam_profile_correction(stitched_slices)
    crop_interface(beam_profile_correction.out.corrected)
    normalize(crop_interface.out.cropped)

    // Stage 3.5: Auto slice quality assessment (optional)
    // Runs after normalization, generates a slice_config.csv that marks
    // degraded slices. Replaces the static slice_config_channel for
    // bring_to_common_space when enabled.
    if (params.auto_assess_quality) {
        auto_assess_inputs = normalize.out.normalized
            .map { _id, norm_path -> norm_path }
            .collect()
        auto_assess_quality(auto_assess_inputs)
        effective_slice_config = auto_assess_quality.out.slice_config
    } else {
        effective_slice_config = slice_config_channel
    }

    // Stage 4: Common Space Alignment
    // Optionally correct encoder glitch spikes before alignment.
    if (params.detect_rehoming) {
        detect_rehoming_events(shifts_xy)
        aligned_shifts = detect_rehoming_events.out.corrected_shifts
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
        .merge(effective_slice_config) { a, b -> tuple(a[0], a[1], b) }

    bring_to_common_space(common_space_input)

    slices_common_space = bring_to_common_space.out
        .flatten()
        .toSortedList { a, b -> a.getName() <=> b.getName() }

    if (params.common_space_preview) {
        preview_input = bring_to_common_space.out
            .flatten()
            .map { toSliceTuple(it) }
        generate_common_space_preview(preview_input)
    }

    // Stage 5: Missing Slice Interpolation (optional)
    if (params.interpolate_missing_slices) {
        gaps_channel = slices_common_space
            .map { sliceList -> [detectSingleGaps(sliceList), sliceList] }
            .flatMap { gapsAndSlices ->
                def gaps = gapsAndSlices[0]
                def sliceList = gapsAndSlices[1]
                if (gaps.isEmpty()) return []

                gaps.collect { gap ->
                    def missingId = gap[0], beforeId = gap[1], afterId = gap[2]
                    def sliceBefore = sliceList.find { it.getName().contains("slice_z${beforeId}") }
                    def sliceAfter = sliceList.find { it.getName().contains("slice_z${afterId}") }
                    (sliceBefore && sliceAfter) ? tuple(missingId, sliceBefore, sliceAfter) : null
                }.findAll { it != null }
            }

        interpolate_missing_slice(gaps_channel)

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

    if (params.create_registration_masks) {
        mask_input = all_slices.flatten().map { toSliceTuple(it) }
        create_registration_masks(mask_input)

        all_masks = create_registration_masks.out.masks
            .collect()
            .map { list -> list.sort { it.getName() } }

        fixed_masks = all_masks
            .map { list -> list.size() > 1 ? list.subList(0, list.size() - 1) : [] }
            .flatten()
        moving_masks = all_masks
            .map { list -> list.size() > 1 ? list.subList(1, list.size()) : [] }
            .flatten()

        pairs = pairs
            .merge(moving_masks) { a, b -> tuple(a[0], a[1], b) }
            .merge(fixed_masks) { a, b -> tuple(a[0], a[1], a[2], b) }
    } else {
        pairs = pairs.map { a, b -> tuple(a, b, [], []) }
    }

    register_pairwise(pairs)

    // Stage 7: Stacking
    if (params.stacking_method == 'motor') {
        log.info "Using MOTOR-POSITION stacking with registration refinements"

        slices_collected = all_slices.flatten().collect()
        transforms_collected = register_pairwise.out.collect()

        motor_stack_input = slices_collected
            .combine(shifts_xy)
            .combine(transforms_collected)
            .map { items ->
                def slices = []
                def shifts = null
                def transforms = []

                items.each { item ->
                    def name = item.getName()
                    if (name.endsWith('.csv')) {
                        shifts = item
                    } else if (name.endsWith('.ome.zarr')) {
                        slices << item
                    } else {
                        transforms << item
                    }
                }

                def slice_ids_str = extractSliceIdsString(slices)
                tuple(slices, shifts, transforms, subject_name, slice_ids_str)
            }

        stack_motor(motor_stack_input)
        stack_output = stack_motor.out.volume
        stack_metadata = motor_stack_input.map { slices, shifts, transforms, name, ids_str ->
            tuple(name, ids_str.split(',').size(), ids_str)
        }

    } else {
        log.info "Using REGISTRATION stacking (pairwise transforms)"

        stack_input = all_slices
            .concat(register_pairwise.out.collect())
            .toList()
            .map { both_lists ->
                def slices = both_lists[0]
                def transforms = both_lists[1]
                def slice_ids_str = extractSliceIdsString(slices)
                tuple(slices, transforms, subject_name, slice_ids_str)
            }

        stack(stack_input)
        stack_output = stack.out.volume
        stack_metadata = stack_input.map { slices, transforms, name, ids_str ->
            tuple(name, ids_str.split(',').size(), ids_str)
        }
    }

    // Stage 8: Z-Intensity Normalization (optional)
    if (params.normalize_z_slices) {
        log.info "Normalizing Z-direction intensity drift (sigma=${params.znorm_smooth_sigma} slices)"
        znorm_input = stack_output
            .combine(stack_metadata)
            .map { zarr, zip, png, annotated, name, n, ids_str -> tuple(zarr, name, n, ids_str) }
        normalize_z_intensity(znorm_input)
        final_stack_output = normalize_z_intensity.out
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

    // Stage 11: Diagnostic Analyses (optional; enable via diagnostic_mode or individual flags)
    def runRotationAnalysis = params.diagnostic_mode || params.analyze_rotation_drift
    def runMotorOnlyStitch = params.diagnostic_mode || params.motor_only_stitch
    def runMotorOnlyStack = params.diagnostic_mode || params.motor_only_stack
    def runDilationDiagnostics = params.diagnostic_mode || params.analyze_tile_dilation
    def runAcquisitionRotation = params.diagnostic_mode || params.analyze_acquisition_rotation

    if (params.diagnostic_mode) {
        log.info "DIAGNOSTIC MODE enabled:"
        log.info "  - Acquisition rotation analysis"
        log.info "  - Registration rotation drift"
        log.info "  - Tile dilation analysis"
        log.info "  - Motor-only stitching (per-slice)"
        log.info "  - Motor-only stacking (3D volume)"
    }

    if (runAcquisitionRotation) {
        analyze_acquisition_rotation(shifts_xy, register_pairwise.out.collect())
    }

    if (runRotationAnalysis) {
        analyze_rotation_drift(register_pairwise.out.collect())
    }

    if (runDilationDiagnostics) {
        if (params.stacking_method == 'motor') {
            // estimate_xy_transformation is not run in the motor path, so no XY transform to analyze
            log.warn "Tile dilation analysis skipped: requires stacking_method='registration' (needs XY transform output)"
        } else if (params.use_motor_positions_for_stitching) {
            log.warn "Tile dilation analysis is not meaningful when use_motor_positions_for_stitching=true"
            log.warn "  The transform IS motor positions, so no dilation will be detected."
        } else {
            log.info "Running tile dilation analysis..."
            dilation_input_diag = illum_fixed.combine(estimate_xy_transformation.out.transform, by: 0)
            analyze_tile_dilation(dilation_input_diag)

            if (params.diagnostic_mode) {
                dilation_json_files_diag = analyze_tile_dilation.out
                    .map { slice_id, json, png, txt -> json }
                    .collect()
                aggregate_dilation_analysis(dilation_json_files_diag)
            }
        }
    }

    if (runMotorOnlyStitch) {
        stitch_motor_only(illum_fixed)
    }

    if (runMotorOnlyStack) {
        motor_only_stack_input = normalize.out.normalized
            .map { slice_id, file -> file }
            .collect()
        stack_motor_only(motor_only_stack_input, shifts_xy)
    }

    // Compare motor-only vs refined stitching
    def runStitchingComparison = params.compare_stitching || params.diagnostic_mode
    if (runStitchingComparison) {
        log.info "Running stitching comparison (motor-only vs refined)..."

        stitch_refined(illum_fixed)

        if (!runMotorOnlyStitch) {
            stitch_motor_only(illum_fixed)
        }

        motor_stitch_with_id = stitch_motor_only.out
            .map { file ->
                def match = file.getName() =~ /slice_z(\d+)/
                def slice_id = match ? match[0][1] : "unknown"
                tuple(slice_id, file)
            }

        refined_stitch_with_id = stitch_refined.out[0]
            .map { file ->
                def match = file.getName() =~ /slice_z(\d+)/
                def slice_id = match ? match[0][1] : "unknown"
                tuple(slice_id, file)
            }

        comparison_input = motor_stitch_with_id
            .combine(refined_stitch_with_id, by: 0)

        compare_stitching(comparison_input)
    }
}
