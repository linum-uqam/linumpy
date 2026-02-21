#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * =============================================================================
 * 3D RECONSTRUCTION PIPELINE FOR SERIAL OCT DATA
 * =============================================================================
 *
 * Converts raw S-OCT mosaic grids into a reconstructed 3D volume.
 *
 * Input:  Directory containing mosaic_grid*.ome.zarr files + shifts_xy.csv
 * Output: 3D OME-Zarr volume with multi-resolution pyramid
 *
 * Pipeline stages:
 *   1. Resample to target resolution
 *   2. Focal curvature correction (optional)
 *   3. Illumination correction (optional)
 *   4. XY tile stitching
 *   5. PSF/beam profile correction
 *   6. Interface cropping
 *   7. Intensity normalization
 *   8. Common space alignment (using shifts file)
 *   9. Missing slice interpolation (optional)
 *  10. Pairwise registration
 *  11. 3D volume stacking
 * =============================================================================
 */

// =============================================================================
// PROCESS DEFINITIONS
// =============================================================================

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
// Diagnostic Processes (for troubleshooting reconstruction artifacts)
// -----------------------------------------------------------------------------

process analyze_rotation_drift {
    publishDir "${params.output}/diagnostics/rotation_analysis", mode: 'copy'

    input:
    path(reg_dirs)

    output:
    path "rotation_analysis/*"

    script:
    """
    # Stage registration outputs into expected directory structure
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

    # Rename outputs with slice ID for unique file names
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
    # Create input directory structure expected by the aggregation script
    mkdir -p dilation_input
    for f in *.json; do
        if [ -f "\$f" ]; then
            # Extract slice_id from the JSON file
            slice_id=\$(grep -o '"slice_id"[[:space:]]*:[[:space:]]*"[^"]*"' "\$f" | head -1 | sed 's/.*"\\([^"]*\\)".*/\\1/')
            if [ -z "\$slice_id" ]; then
                # Try numeric format
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
    # Stage registration outputs into expected directory structure
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
    path "${subject_name}_quality_report.html"

    script:
    def verbose_flag = params.report_verbose ? "--verbose" : ""
    """
    linum_generate_pipeline_report.py ${params.output} ${subject_name}_quality_report.html \
        --title "Quality Report: ${subject_name}" \
        --format html ${verbose_flag}
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
// Segment Break Detection & Correction Processes
// -----------------------------------------------------------------------------

process detect_segment_breaks {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    path("slices/*")

    output:
    path "segment_corrections.json", emit: corrections
    path "rotations.csv",            emit: rotations
    path "segment_breaks.png",       emit: plot

    script:
    def script_name = params.use_gpu ? "linum_detect_segment_breaks_gpu.py" : "linum_detect_segment_breaks.py"
    def translation_arg = params.segment_break_translation_threshold > 0 ? "--translation_threshold ${params.segment_break_translation_threshold}" : ""
    def refine_arg = params.segment_break_refine_translations ? "--refine_translations" : ""
    """
    ${script_name} slices . \
        --rotation_threshold ${params.segment_break_rotation_threshold} \
        --max_rotation_search ${params.segment_break_max_rotation_search} \
        --local_window ${params.segment_break_local_window} \
        --metric_threshold ${params.segment_break_metric_threshold} \
        ${translation_arg} \
        ${refine_arg} \
        -f
    """
}

process apply_segment_corrections {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    path("slices/*")
    path(corrections)

    output:
    path "*.ome.zarr"

    script:
    def script_name = params.use_gpu ? "linum_apply_segment_corrections_gpu.py" : "linum_apply_segment_corrections.py"
    """
    ${script_name} slices ${corrections} corrected -f
    mv corrected/*.ome.zarr .
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
    input:
    tuple val(slice_id), path(aip)

    output:
    tuple val(slice_id), path("z${slice_id}_transform_xy.npy")

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
    input:
    tuple val(slice_id), path(mosaic_grid), path(transform_xy)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr")

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
    input:
    tuple val(slice_id), path(slice_3d)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_axial_corr.ome.zarr")

    script:
    """
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_axial_corr.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper}
    """
}

process crop_interface {
    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_crop_interface.ome.zarr")

    script:
    """
    linum_crop_3d_mosaic_below_interface.py ${image} "slice_z${slice_id}_crop_interface.ome.zarr" \
        --depth ${params.crop_interface_out_depth} \
        --crop_before_interface \
        --percentile_max ${params.clip_percentile_upper}
    """
}

process normalize {
    input:
    tuple val(slice_id), path(image)

    output:
    tuple val(slice_id), path("slice_z${slice_id}_normalize.ome.zarr")

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

process bring_to_common_space {
    publishDir "${params.output}/${task.process}", mode: 'copy'

    input:
    tuple path("inputs/*"), path("shifts_xy.csv"), path(slice_config)

    output:
    path "*.ome.zarr"

    script:
    def slice_config_arg = slice_config.name != 'NO_SLICE_CONFIG' ? "--slice_config ${slice_config}" : ""

    def outlier_args = params.filter_shift_outliers ? """--filter_outliers \\
        --max_shift_mm ${params.max_shift_mm} \\
        --outlier_method ${params.outlier_method} \\
        --iqr_multiplier ${params.outlier_iqr_multiplier} \\
        --max_step_mm ${params.common_space_max_step_mm} \\
        --step_window ${params.common_space_step_window} \\
        --step_method ${params.common_space_step_method} \\
        --step_mad_threshold ${params.common_space_step_mad_threshold}""" : ""

    def excluded_args = params.common_space_excluded_slice_mode ?
        "--excluded_slice_mode ${params.common_space_excluded_slice_mode} --excluded_slice_window ${params.common_space_excluded_slice_window}" : ""

    """
    linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv common_space \
        ${slice_config_arg} ${outlier_args} ${excluded_args}
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
    // Use the simplified registration script
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

// Motor-position-based stacking (uses shifts_xy.csv for XY alignment + optional refinements)
process stack_motor {
    publishDir "$params.output/$task.process", mode: 'move'

    input:
    tuple path("slices/*"), path(shifts_file), path("transforms/*"), val(subject_name), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png")

    script:
    def options = ""

    // Blending
    if (params.stack_blend_enabled) options += " --blend"

    // Z-matching
    options += " --slicing_interval_mm ${params.registration_slicing_interval_mm}"
    options += " --search_range_mm ${params.registration_allowed_drifting_mm}"
    options += " --moving_z_first_index ${params.moving_slice_first_index}"
    if (params.use_expected_z_overlap) options += " --use_expected_overlap"

    // Output z-matches for debugging (controlled by analyze_shifts)
    if (params.analyze_shifts) options += " --output_z_matches z_matches.csv"

    // Use registration refinements (rotation + small translation)
    if (params.apply_pairwise_transforms) {
        options += " --transforms_dir transforms"

        // Apply only rotation from registration (prevents XY jumps when motor positions are trusted)
        if (params.apply_rotation_only) options += " --rotation_only"

        // Clamp maximum rotation per slice to prevent registration errors from causing drift
        options += " --max_rotation_deg ${params.max_rotation_deg}"
    }

    // Accumulate pairwise translations cumulatively across slices
    if (params.stack_accumulate_translations) {
        options += " --accumulate_translations"
        // Pass registration boundary so accumulation can filter clamped translations
        options += " --max_pairwise_translation ${params.registration_max_translation}"
    }

    // Smooth cumulative translations to reduce XY jitter
    if (params.stack_smooth_window > 0) options += " --smooth_window ${params.stack_smooth_window}"

    // Skip XY shifting since slices are already in common space (from bring_to_common_space)
    options += " --no_xy_shift"

    // Pyramid configuration
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
    """
    # Stack slices using motor positions for XY + registration refinements
    linum_stack_slices_motor.py slices ${shifts_file} ${subject_name}.ome.zarr ${options}
    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr

    # Generate preview
    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png

    # Generate annotated preview with actual slice IDs
    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
        --slice_ids "${slice_ids_str}" \
        --label_every ${params.annotated_label_every} ${show_lines_flag}
    """
}

// Traditional pairwise-registration-based stacking
process stack {
    publishDir "$params.output/$task.process", mode: 'move'

    input:
    tuple path("mosaics/*"), path("transforms/*"), val(subject_name), val(slice_ids_str)

    output:
    tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png")

    script:
    def options = ""

    // Blending options
    if (params.stack_blend_enabled) {
        options += "--blend"
        if (params.stack_max_overlap > 0) options += " --overlap ${params.stack_max_overlap}"
    }

    // Transform accumulation
    if (params.stack_no_accumulate_transforms) options += " --no_accumulate_transforms"

    // Pyramid configuration
    if (params.pyramid_n_levels != null) {
        options += " --n_levels ${params.pyramid_n_levels}"
    } else {
        def base_res = params.resolution > 0 ? params.resolution : 10
        def valid_resolutions = params.pyramid_resolutions.findAll { it >= base_res }.sort()
        if (!valid_resolutions.contains(base_res)) valid_resolutions = [base_res] + valid_resolutions
        def pyramid_res_str = valid_resolutions.collect { it.toString() }.join(' ')
        options += " --pyramid_resolutions ${pyramid_res_str}"
        options += params.pyramid_make_isotropic ? " --make_isotropic" : " --no-make_isotropic"
    }

    def show_lines_flag = params.annotated_show_lines ? '--show_lines' : ''
    """
    # Stack slices and generate outputs
    linum_stack_slices_3d.py mosaics transforms ${subject_name}.ome.zarr ${options}
    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr
    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png

    # Generate annotated preview with pre-computed slice IDs
    linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
        --slice_ids "${slice_ids_str}" \
        --label_every ${params.annotated_label_every} \
        ${show_lines_flag}
    """
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Extract slice ID from a filename containing z## pattern
 * Works with both mosaic_grid_z01.ome.zarr and slice_z01_*.ome.zarr
 * Returns: String slice ID (e.g., "01") or "unknown" if not found
 */
def extractSliceId(filename) {
    def name = filename instanceof Path ? filename.getName() : filename.toString()
    def matcher = name =~ /z(\d+)/
    return matcher ? matcher[0][1] : "unknown"
}

/**
 * Extract slice ID as integer from a filename
 * Returns: Integer slice ID or -1 if not found
 */
def extractSliceIdInt(filename) {
    def id = extractSliceId(filename)
    return id == "unknown" ? -1 : id.toInteger()
}

/**
 * Create a tuple of (slice_id, file) from a file path
 * Useful for channel mapping operations
 */
def toSliceTuple(file_path) {
    tuple(extractSliceId(file_path), file_path)
}

/**
 * Extract sorted slice IDs from a list of files as comma-separated string
 * Returns: String like "01,02,03,05" (sorted numerically)
 */
def extractSliceIdsString(fileList) {
    fileList
        .collect { extractSliceId(it) }
        .findAll { it != "unknown" }
        .sort { it.toInteger() }
        .join(',')
}

/**
 * Normalize a file path by removing duplicate and trailing slashes
 */
def normalizePath(path) {
    return path.replaceAll('/+', '/').replaceAll('/$', '')
}

/**
 * Safely join path components, handling trailing slashes
 */
def joinPath(base, filename) {
    def normalizedBase = normalizePath(base)
    return "${normalizedBase}/${filename}"
}

/**
 * Parse slice configuration file to determine which slices to process
 * Returns: Set of slice IDs marked for use
 */
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

/**
 * Detect single-slice gaps that can be interpolated
 * Returns: List of [missingId, beforeId, afterId] tuples
 */
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

/**
 * Parse debug_slices parameter for subset processing
 * Supports formats: "25,26" or "25-29" or "25,27-29"
 * Returns: Set of zero-padded slice IDs, or null if not specified
 */
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
    // =========================================================================
    // INITIALIZATION
    // =========================================================================
    README()

    // Normalize input path
    def inputDir = normalizePath(params.input)

    // Determine subject name (auto-detect from path if not specified)
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

    // Parse debug_slices for subset processing
    def debugSlices = parseDebugSlices(params.debug_slices)
    if (debugSlices) {
        log.info "DEBUG MODE: Processing only slices ${debugSlices.sort().join(', ')}"
    }

    // -------------------------------------------------------------------------
    // Load shifts file
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Load slice configuration (optional)
    // -------------------------------------------------------------------------
    def slice_config_path = params.slice_config ?: joinPath(inputDir, "slice_config.csv")
    def slicesToUse = null
    if (file(slice_config_path).exists()) {
        slicesToUse = parseSliceConfig(slice_config_path)
        log.info "Slice config: ${slice_config_path}"
    } else if (params.slice_config) {
        error("Slice config file not found: ${slice_config_path}")
    }

    // -------------------------------------------------------------------------
    // Discover input mosaic grids
    // -------------------------------------------------------------------------
    log.info "Looking for mosaic grids in: ${inputDir}"

    def inputDirFile = file(inputDir)
    def mosaicFiles = inputDirFile.listFiles()
        .findAll { it.isDirectory() && it.name.startsWith('mosaic_grid') && it.name.endsWith('.ome.zarr') && it.name =~ /z\d+/ }
        .sort { it.name }

    if (mosaicFiles.isEmpty()) {
        error("No mosaic grids found in ${inputDir}. Expected: mosaic_grid*_z00.ome.zarr")
    }
    log.info "Found ${mosaicFiles.size()} mosaic grids"

    // Create input channel with slice filtering
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

    // -------------------------------------------------------------------------
    // Optional: Analyze shifts
    // -------------------------------------------------------------------------
    if (params.analyze_shifts) {
        analyze_shifts(shifts_xy)
    }

    // -------------------------------------------------------------------------
    // Stage 1: Preprocessing
    // -------------------------------------------------------------------------
    resampled = params.resolution > 0 ? resample_mosaic_grid(inputSlices) : inputSlices
    focal_fixed = params.fix_curvature_enabled ? fix_focal_curvature(resampled) : resampled
    illum_fixed = params.fix_illum_enabled ? fix_illumination(focal_fixed) : focal_fixed

    // -------------------------------------------------------------------------
    // Stage 2: XY Stitching
    // -------------------------------------------------------------------------
    if (params.use_refined_stitching) {
        // Use refined stitching with registration-based blend refinement
        // This helps reduce visible tile seams
        stitch_3d_with_refinement(illum_fixed)
        stitched_slices = stitch_3d_with_refinement.out
    } else {
        // Use standard stitching with motor-based transform
        generate_aip(illum_fixed)
        estimate_xy_transformation(generate_aip.out)
        stitch_3d(illum_fixed.combine(estimate_xy_transformation.out, by: 0))
        stitched_slices = stitch_3d.out
    }

    // Generate stitch previews if enabled
    if (params.stitch_preview) {
        generate_stitch_preview(stitched_slices)
    }

    // -------------------------------------------------------------------------
    // Stage 3: Corrections
    // -------------------------------------------------------------------------
    beam_profile_correction(stitched_slices)
    crop_interface(beam_profile_correction.out)
    normalize(crop_interface.out)

    // =========================================================================
    // STAGE 4: COMMON SPACE ALIGNMENT
    // =========================================================================
    common_space_input = normalize.out
        .toSortedList { a, b -> a[0] <=> b[0] }
        .flatten()
        .collate(2)
        .map { _meta, filename -> filename }
        .collect()
        .merge(shifts_xy) { a, b -> tuple(a, b) }
        .merge(slice_config_channel) { a, b -> tuple(a[0], a[1], b) }

    bring_to_common_space(common_space_input)

    // =========================================================================
    // STAGE 4b: SEGMENT BREAK DETECTION & CORRECTION (optional)
    // =========================================================================
    // Detects acquisition remounting events (sudden in-plane rotation or
    // translation jumps) and applies corrective rigid transforms to all slices
    // in the affected segment before pairwise registration runs.
    // Enable with --detect_segment_breaks true.
    if (params.detect_segment_breaks) {
        log.info "Segment break detection ENABLED " +
                 "(threshold: ${params.segment_break_rotation_threshold}°, " +
                 "search range: ±${params.segment_break_max_rotation_search}°)"

        common_slices_collected = bring_to_common_space.out.flatten().collect()

        detect_segment_breaks(common_slices_collected)

        apply_segment_corrections(
            common_slices_collected,
            detect_segment_breaks.out.corrections
        )

        slices_common_space = apply_segment_corrections.out
            .flatten()
            .toSortedList { a, b -> a.getName() <=> b.getName() }
    } else {
        slices_common_space = bring_to_common_space.out
            .flatten()
            .toSortedList { a, b -> a.getName() <=> b.getName() }
    }

    if (params.common_space_preview) {
        preview_input = bring_to_common_space.out
            .flatten()
            .map { toSliceTuple(it) }
        generate_common_space_preview(preview_input)
    }

    // =========================================================================
    // STAGE 5: MISSING SLICE INTERPOLATION (optional)
    // =========================================================================
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

    // =========================================================================
    // STAGE 6: PAIRWISE REGISTRATION
    // =========================================================================
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

    // =========================================================================
    // STAGE 7: STACKING
    // =========================================================================

    if (params.stacking_method == 'motor') {
        // Motor-position-based stacking with registration refinements
        // XY alignment: from motor positions (shifts_xy.csv)
        // Z-matching: from registration or correlation
        // Refinements: rotation + small translation from pairwise registration
        log.info "Using MOTOR-POSITION stacking with registration refinements"

        // Collect slices and transforms
        slices_collected = all_slices.flatten().collect()
        transforms_collected = register_pairwise.out.collect()

        // Build the input tuple for stack_motor process
        motor_stack_input = slices_collected
            .combine(shifts_xy)
            .combine(transforms_collected)
            .map { items ->
                // items is a flat list: [slice1, slice2, ..., shifts_file, transform1, transform2, ...]
                // We need to separate them into: tuple(slices, shifts, transforms, subject_name, slice_ids_str)
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
                        // Transform directories
                        transforms << item
                    }
                }

                // Extract slice IDs from slice filenames
                def slice_ids_str = extractSliceIdsString(slices)

                tuple(slices, shifts, transforms, subject_name, slice_ids_str)
            }

        stack_motor(motor_stack_input)
        stack_output = stack_motor.out

    } else {
        // Traditional pairwise-registration-based stacking
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
        stack_output = stack.out
    }

    // =========================================================================
    // STAGE 8: REPORT GENERATION
    // =========================================================================
    if (params.generate_report) {
        generate_report(stack_output, subject_name)
    }

    // =========================================================================
    // STAGE 9: DIAGNOSTIC ANALYSES (optional)
    // =========================================================================
    // Enable with diagnostic_mode=true or individual flags

    def runRotationAnalysis = params.diagnostic_mode || params.analyze_rotation_drift
    def runMotorOnlyStitch = params.diagnostic_mode || params.motor_only_stitch
    def runMotorOnlyStack = params.diagnostic_mode || params.motor_only_stack
    def runDilationDiagnostics = params.diagnostic_mode || params.analyze_tile_dilation

    if (params.diagnostic_mode) {
        log.info "DIAGNOSTIC MODE enabled:"
        log.info "  - Acquisition rotation analysis"
        log.info "  - Registration rotation drift"
        log.info "  - Tile dilation analysis"
        log.info "  - Motor-only stitching (per-slice)"
        log.info "  - Motor-only stacking (3D volume)"
    }

    if (params.diagnostic_mode) {
        analyze_acquisition_rotation(shifts_xy, register_pairwise.out.collect())
    }

    if (runRotationAnalysis) {
        analyze_rotation_drift(register_pairwise.out.collect())
    }

    if (runDilationDiagnostics) {
        if (params.use_motor_positions_for_stitching && !params.use_refined_stitching) {
            log.warn "Tile dilation analysis is not meaningful when use_motor_positions_for_stitching=true"
            log.warn "  The transform IS motor positions, so no dilation will be detected."
            log.warn "  Enable use_refined_stitching=true to analyze refinement-induced dilation."
        } else {
            log.info "Running tile dilation analysis..."
            dilation_input_diag = illum_fixed.combine(estimate_xy_transformation.out, by: 0)
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
        motor_only_stack_input = normalize.out
            .map { slice_id, file -> file }
            .collect()
        stack_motor_only(motor_only_stack_input, shifts_xy)
    }

    // Compare motor-only vs refined stitching
    def runStitchingComparison = params.compare_stitching || params.diagnostic_mode
    if (runStitchingComparison) {
        log.info "Running stitching comparison (motor-only vs refined)..."

        // Run refined stitching
        stitch_refined(illum_fixed)

        // Run motor-only stitching if not already done
        if (!runMotorOnlyStitch) {
            stitch_motor_only(illum_fixed)
        }

        // Join outputs by slice ID for comparison
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

        // Combine for comparison
        comparison_input = motor_stitch_with_id
            .combine(refined_stitch_with_id, by: 0)

        compare_stitching(comparison_input)
    }
}
