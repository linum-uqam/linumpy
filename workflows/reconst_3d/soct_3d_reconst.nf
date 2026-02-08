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
    output: path "readme.txt"

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
    input: path(shifts_file)
    output: path "shifts_analysis/*"

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
    input: path(reg_dirs)
    output: path "rotation_analysis/*"

    script:
    """
    # Create a directory structure that the script expects
    mkdir -p register_pairwise
    for d in ${reg_dirs}; do
        if [ -d "\$d" ]; then
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
    input: tuple val(slice_id), path(mosaic_grid), path(transform_xy)
    output: path "dilation_analysis/*"

    script:
    """
    linum_analyze_tile_dilation.py ${mosaic_grid} ${transform_xy} dilation_analysis \
        --resolution ${params.resolution} \
        --overlap_fraction ${params.motor_only_overlap} \
        --slice_id ${slice_id}
    """
}

process stitch_motor_only {
    publishDir "${params.output}/diagnostics/motor_only_stitch", mode: 'copy'
    input: tuple val(slice_id), path(mosaic_grid)
    output: path "slice_z${slice_id}_motor_only.ome.zarr"

    script:
    """
    linum_stitch_motor_only.py ${mosaic_grid} "slice_z${slice_id}_motor_only.ome.zarr" \
        --overlap_fraction ${params.motor_only_overlap} \
        --blending_method diffusion
    """
}

process run_full_diagnostics {
    publishDir "${params.output}/diagnostics", mode: 'copy'
    input: path(pipeline_output)
    output: path "full_diagnostics/*"

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
    output: path "*"

    script:
    """
    # Create a directory structure that the script expects
    mkdir -p register_pairwise
    for d in ${reg_dirs}; do
        if [ -d "\$d" ]; then
            ln -s "\$(pwd)/\$d" "register_pairwise/\$d"
        fi
    done

    linum_analyze_acquisition_rotation.py ${shifts_file} . \
        --registration_dir register_pairwise \
        --resolution ${params.resolution}
    """
}

process generate_report {
    publishDir "$params.output", mode: 'copy'
    input:
        tuple path(zarr), path(zip), path(png), path(annotated_png)
        val subject_name
    output: path "${subject_name}_quality_report.html"

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
    input: tuple val(slice_id), path(mosaic_grid)
    output: tuple val(slice_id), path("mosaic_grid_z${slice_id}_resampled.ome.zarr")

    script:
    def script = params.use_gpu ? "linum_resample_mosaic_grid_gpu.py" : "linum_resample_mosaic_grid.py"
    def gpu_opts = params.use_gpu ? "--use_gpu" : ""
    """
    ${script} ${mosaic_grid} "mosaic_grid_z${slice_id}_resampled.ome.zarr" -r ${params.resolution} ${gpu_opts} -v
    """
}

process fix_focal_curvature {
    input: tuple val(slice_id), path(mosaic_grid)
    output: tuple val(slice_id), path("mosaic_grid_z${slice_id}_focal_fix.ome.zarr")

    script:
    """
    linum_detect_focal_curvature.py ${mosaic_grid} "mosaic_grid_z${slice_id}_focal_fix.ome.zarr"
    """
}

process fix_illumination {
    cpus params.processes
    input: tuple val(slice_id), path(mosaic_grid)
    output: tuple val(slice_id), path("mosaic_grid_z${slice_id}_illum_fix.ome.zarr")

    script:
    def script = params.use_gpu ? "linum_fix_illumination_3d_gpu.py" : "linum_fix_illumination_3d.py"
    """
    ${script} ${mosaic_grid} "mosaic_grid_z${slice_id}_illum_fix.ome.zarr" \
        --n_processes ${params.processes} \
        --percentile_max ${params.clip_percentile_upper}
    """
}

// -----------------------------------------------------------------------------
// Stitching Processes
// -----------------------------------------------------------------------------

process generate_aip {
    publishDir "${params.output}/${task.process}", mode: 'copy'
    input: tuple val(slice_id), path(mosaic_grid)
    output: tuple val(slice_id), path("mosaic_grid_z${slice_id}_aip.ome.zarr")

    script:
    """
    linum_aip.py ${mosaic_grid} "mosaic_grid_z${slice_id}_aip.ome.zarr"
    """
}

process estimate_xy_transformation {
    input: tuple val(slice_id), path(aip)
    output: tuple val(slice_id), path("z${slice_id}_transform_xy.npy")

    script:
    def script = params.use_gpu ? "linum_estimate_transform_gpu.py" : "linum_estimate_transform.py"
    def gpu_opts = params.use_gpu ? "--use_gpu" : ""
    """
    ${script} ${aip} "z${slice_id}_transform_xy.npy" ${gpu_opts}
    """
}

process stitch_3d {
    input: tuple val(slice_id), path(mosaic_grid), path(transform_xy)
    output: tuple val(slice_id), path("slice_z${slice_id}_stitch_3d.ome.zarr")

    script:
    """
    linum_stitch_3d.py ${mosaic_grid} ${transform_xy} "slice_z${slice_id}_stitch_3d.ome.zarr"
    """
}

// -----------------------------------------------------------------------------
// Correction Processes
// -----------------------------------------------------------------------------

process beam_profile_correction {
    input: tuple val(slice_id), path(slice_3d)
    output: tuple val(slice_id), path("slice_z${slice_id}_axial_corr.ome.zarr")

    script:
    """
    linum_compensate_psf_model_free.py ${slice_3d} "slice_z${slice_id}_axial_corr.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper}
    """
}

process crop_interface {
    input: tuple val(slice_id), path(image)
    output: tuple val(slice_id), path("slice_z${slice_id}_crop_interface.ome.zarr")

    script:
    """
    linum_crop_3d_mosaic_below_interface.py ${image} "slice_z${slice_id}_crop_interface.ome.zarr" \
        --depth ${params.crop_interface_out_depth} \
        --crop_before_interface \
        --percentile_max ${params.clip_percentile_upper}
    """
}

process normalize {
    input: tuple val(slice_id), path(image)
    output: tuple val(slice_id), path("slice_z${slice_id}_normalize.ome.zarr")

    script:
    def script = params.use_gpu ? "linum_normalize_intensities_per_slice_gpu.py" : "linum_normalize_intensities_per_slice.py"
    def gpu_opts = params.use_gpu ? "--use_gpu" : ""
    """
    ${script} ${image} "slice_z${slice_id}_normalize.ome.zarr" \
        --percentile_max ${params.clip_percentile_upper} ${gpu_opts}
    """
}

// -----------------------------------------------------------------------------
// Alignment Processes
// -----------------------------------------------------------------------------

process bring_to_common_space {
    publishDir "${params.output}/${task.process}", mode: 'copy'
    input: tuple path("inputs/*"), path("shifts_xy.csv"), path(slice_config)
    output: path "*.ome.zarr"

    script:
    def slice_config_arg = slice_config.name != 'NO_SLICE_CONFIG' ? "--slice_config ${slice_config}" : ""
    def outlier_args = params.filter_shift_outliers ?
        "--filter_outliers --max_shift_mm ${params.max_shift_mm} --outlier_method ${params.outlier_method} --iqr_multiplier ${params.outlier_iqr_multiplier} " +
        "--max_step_mm ${params.common_space_max_step_mm} --step_window ${params.common_space_step_window} --step_method ${params.common_space_step_method}" : ""
    def excluded_args = params.common_space_excluded_slice_mode ?
        "--excluded_slice_mode ${params.common_space_excluded_slice_mode} --excluded_slice_window ${params.common_space_excluded_slice_window}" : ""
    """
    linum_align_mosaics_3d_from_shifts.py inputs shifts_xy.csv common_space ${slice_config_arg} ${outlier_args} ${excluded_args}
    mv common_space/* .
    """
}

process generate_common_space_preview {
    publishDir "${params.output}/common_space_previews", mode: 'copy'
    input: tuple val(slice_id), path(slice_zarr)
    output: path "slice_z${slice_id}_preview.png"

    script:
    """
    linum_screenshot_omezarr.py ${slice_zarr} "slice_z${slice_id}_preview.png"
    """
}

process interpolate_missing_slice {
    publishDir "${params.output}/${task.process}", mode: 'copy'
    input: tuple val(missing_slice_id), path(slice_before), path(slice_after)
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
    input: tuple val(slice_id), path(image)
    output:
        path("mask_slice_z${slice_id}.ome.zarr"), emit: masks
        path("mask_slice_z${slice_id}_preview.png"), optional: true, emit: previews

    script:
    def script = params.use_gpu ? "linum_create_masks_gpu.py" : "linum_create_masks.py"
    def gpu_opts = params.use_gpu ? "--use_gpu" : ""
    def normalize_flag = params.mask_normalize ? "--normalize" : ""
    def preview_flag = params.mask_preview ? "--preview mask_slice_z${slice_id}_preview.png" : ""
    def fill_holes_opt = params.mask_fill_holes ? "--fill_holes ${params.mask_fill_holes}" : ""
    """
    ${script} ${image} mask_slice_z${slice_id}.ome.zarr \
        --sigma ${params.mask_smoothing_sigma} \
        --selem_radius ${params.selem_radius} \
        --min_size ${params.min_size} \
        ${normalize_flag} ${fill_holes_opt} ${preview_flag} ${gpu_opts}
    """
}

process register_pairwise {
    publishDir "${params.output}/${task.process}", mode: 'copy'
    input: tuple path(fixed_vol), path(moving_vol), path(moving_mask, stageAs: 'moving_mask*'), path(fixed_mask, stageAs: 'fixed_mask*')
    output: path "*"

    script:
    def fixed_id = (fixed_vol.getName() =~ /slice_z(\d+)/)[0][1].toInteger()
    def moving_id = (moving_vol.getName() =~ /slice_z(\d+)/)[0][1].toInteger()
    def slice_gap = moving_id - fixed_id

    def opts = [
        "--moving_slice_index ${params.moving_slice_first_index}",
        "--transform ${params.registration_transform}",
        "--metric ${params.registration_metric}",
        "--max_translation ${params.registration_max_translation}",
        "--max_rotation ${params.registration_max_rotation}",
        "--slicing_interval ${params.registration_slicing_interval_mm}",
        "--allowed_drifting ${params.registration_allowed_drifting_mm}",
        "--z_bias ${params.registration_z_bias}"
    ]
    if (slice_gap > 1) opts << "--slice_gap_multiplier ${slice_gap}"


    def options_str = opts.join(' ')
    def mask_opts = params.create_registration_masks ? "--use_masks --moving_mask ${moving_mask} --fixed_mask ${fixed_mask} --mask_mode ${params.registration_mask_mode}" : ""
    """
    dirname=\$(basename ${moving_vol} .ome.zarr)
    linum_estimate_transform_pairwise.py ${fixed_vol} ${moving_vol} \$dirname ${options_str} ${mask_opts}
    """
}

// -----------------------------------------------------------------------------
// Stacking Process
// -----------------------------------------------------------------------------

process stack {
    publishDir "$params.output/$task.process", mode: 'move'
    input: tuple path("mosaics/*"), path("transforms/*"), val(subject_name)
    output: tuple path("${subject_name}.ome.zarr"), path("${subject_name}.ome.zarr.zip"), path("${subject_name}.png"), path("${subject_name}_annotated.png")

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
    """
    # Extract slice IDs from mosaic filenames (e.g., slice_z05_... -> 05)
    # Sort numerically so they appear in correct order in the preview
    slice_ids=\$(ls -1 mosaics/*.ome.zarr 2>/dev/null | sed -n 's/.*slice_z\\([0-9]*\\).*/\\1/p' | sort -n | tr '\\n' ',' | sed 's/,\$//')
    n_slices=\$(echo "\$slice_ids" | tr ',' '\\n' | wc -l | tr -d ' ')

    # Validate n_slices is a positive number
    if [ -z "\$n_slices" ] || [ "\$n_slices" -lt 1 ]; then
        echo "WARNING: Could not count input slices, trying alternative method"
        n_slices=\$(ls -d mosaics/*.ome.zarr 2>/dev/null | wc -l | tr -d ' ')
        slice_ids=""
    fi
    echo "DEBUG: Found \$n_slices input slices for annotated preview"
    echo "DEBUG: Slice IDs: \$slice_ids"

    linum_stack_slices_3d.py mosaics transforms ${subject_name}.ome.zarr ${options}
    zip -r ${subject_name}.ome.zarr.zip ${subject_name}.ome.zarr
    linum_screenshot_omezarr.py ${subject_name}.ome.zarr ${subject_name}.png

    # Pass slice_ids if available, otherwise fall back to n_slices
    if [ -n "\$slice_ids" ]; then
        linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
            --slice_ids "\$slice_ids" \
            --label_every ${params.annotated_label_every} \
            ${params.annotated_show_lines ? '--show_lines' : ''}
    else
        linum_screenshot_omezarr_annotated.py ${subject_name}.ome.zarr ${subject_name}_annotated.png \
            --n_slices \$n_slices \
            --label_every ${params.annotated_label_every} \
            ${params.annotated_show_lines ? '--show_lines' : ''}
    fi
    """
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

def normalizePath(path) {
    // Remove trailing slashes and normalize double slashes
    return path.replaceAll('/+', '/').replaceAll('/$', '')
}

def joinPath(base, filename) {
    // Safely join path components, handling trailing slashes
    def normalizedBase = normalizePath(base)
    return "${normalizedBase}/${filename}"
}

def parseSliceConfig(configPath) {
    def slicesToUse = [] as Set
    def slicesExcluded = [] as Set
    def file = new File(configPath)
    if (!file.exists()) error("Slice config file not found: ${configPath}")

    file.withReader { reader ->
        reader.readLine() // skip header
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

def detectSingleGaps(sliceList) {
    def gaps = []
    def pattern = ~/slice_z(\d+)/

    def sliceIds = sliceList.collect { file ->
        def matcher = pattern.matcher(file.getName())
        matcher.find() ? matcher.group(1).toInteger() : -1
    }.findAll { it >= 0 }.sort()

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

def parseDebugSlices(debugSlicesStr) {
    // Parse debug_slices parameter: "25,26" or "25-29" or "25,27-29"
    // Returns a Set of slice IDs as zero-padded strings (e.g., ["25", "26"])
    if (!debugSlicesStr || debugSlicesStr.trim().isEmpty()) return null

    def sliceIds = [] as Set
    debugSlicesStr.split(',').each { part ->
        part = part.trim()
        if (part.contains('-')) {
            // Range: "25-29"
            def rangeParts = part.split('-')
            if (rangeParts.size() == 2) {
                def start = rangeParts[0].trim().toInteger()
                def end = rangeParts[1].trim().toInteger()
                (start..end).each { sliceIds.add(String.format("%02d", it)) }
            }
        } else {
            // Single ID: "25"
            sliceIds.add(String.format("%02d", part.toInteger()))
        }
    }
    return sliceIds
}

// =============================================================================
// MAIN WORKFLOW
// =============================================================================

workflow {
    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    README()

    // Normalize input path (remove trailing slashes)
    def inputDir = normalizePath(params.input)

    // Auto-detect subject name from path (look for sub-* pattern or use parent directory)
    def subject_name = params.subject_name
    if (!subject_name) {
        // Try to find sub-* pattern in the path
        def pathParts = inputDir.split('/')
        def subMatch = pathParts.find { it ==~ /sub-\w+/ }
        if (subMatch) {
            subject_name = subMatch
        } else {
            // Fall back to parent directory if input is a subdirectory like mosaic-grids
            def inputFile = file(inputDir)
            def dirName = inputFile.getName()
            if (dirName in ['mosaic-grids', 'mosaics', 'mosaic_grids', 'input', 'data']) {
                subject_name = inputFile.getParent()?.getName() ?: dirName
            } else {
                subject_name = dirName
            }
        }
    }
    log.info "Subject: ${subject_name}"
    log.info "GPU: ${params.use_gpu ? 'ENABLED' : 'DISABLED'}"

    // Parse debug_slices parameter for quick testing
    def debugSlices = parseDebugSlices(params.debug_slices)
    if (debugSlices) {
        log.info "DEBUG MODE: Processing only slices ${debugSlices.sort().join(', ')}"
    }

    // Auto-detect shifts_xy path if not provided
    def shifts_xy_path = params.shifts_xy ? params.shifts_xy : "${inputDir}/shifts_xy.csv"
    log.info "Shifts file: ${shifts_xy_path}"

    // Verify file exists before creating channel
    if (!file(shifts_xy_path).exists()) {
        error """
        Shifts file not found: ${shifts_xy_path}

        Please ensure the shifts_xy.csv file exists in your input directory,
        or specify the path explicitly with --shifts_xy /path/to/shifts_xy.csv
        """
    }
    shifts_xy = channel.of(file(shifts_xy_path))

    // Auto-detect slice_config path if not provided (optional file)
    def slice_config_path = params.slice_config ?: joinPath(inputDir, "slice_config.csv")
    def slicesToUse = null
    if (file(slice_config_path).exists()) {
        slicesToUse = parseSliceConfig(slice_config_path)
        log.info "Slice config: ${slice_config_path}"
    } else if (params.slice_config) {
        error("Slice config file not found: ${slice_config_path}")
    }

    // Find mosaic grids (these are zarr directories)
    // Pattern matches: mosaic_grid_3d_z00.ome.zarr, mosaic_grid_z01.ome.zarr, etc.
    log.info "Looking for mosaic grids in: ${inputDir}"

    // Use Groovy file listing to find zarr directories
    def inputDirFile = file(inputDir)
    def mosaicFiles = inputDirFile.listFiles()
        .findAll { it.isDirectory() && it.name.startsWith('mosaic_grid') && it.name.endsWith('.ome.zarr') && it.name =~ /z\d+/ }
        .sort { it.name }

    if (mosaicFiles.isEmpty()) {
        error("No mosaic grids found in ${inputDir}. Expected files like: mosaic_grid*_z00.ome.zarr")
    }

    log.info "Found ${mosaicFiles.size()} mosaic grids"

    inputSlices = channel
        .fromList(mosaicFiles)
        .map { file_path ->
            // Extract slice ID - look for z followed by digits
            def matcher = file_path.getName() =~ /z(\d+)/
            def key = matcher ? matcher[0][1] : "unknown"
            tuple(key, file_path)
        }
        .filter { slice_id, _files ->
            // First check debug_slices (highest priority for quick testing)
            if (debugSlices != null) {
                def included = debugSlices.contains(slice_id)
                if (!included) log.debug "Skipping slice ${slice_id} (not in debug_slices)"
                return included
            }
            // Then check slice_config
            if (slicesToUse != null) {
                def included = slicesToUse.contains(slice_id)
                return included
            }
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
    generate_aip(illum_fixed)
    estimate_xy_transformation(generate_aip.out)
    stitch_3d(illum_fixed.combine(estimate_xy_transformation.out, by: 0))

    // -------------------------------------------------------------------------
    // Stage 3: Corrections
    // -------------------------------------------------------------------------
    beam_profile_correction(stitch_3d.out)
    crop_interface(beam_profile_correction.out)
    normalize(crop_interface.out)

    // -------------------------------------------------------------------------
    // Stage 4: Common Space Alignment
    // -------------------------------------------------------------------------
    common_space_input = normalize.out
        .toSortedList { a, b -> a[0] <=> b[0] }
        .flatten()
        .collate(2)
        .map { _meta, filename -> filename }
        .collect()
        .merge(shifts_xy) { a, b -> tuple(a, b) }
        .merge(slice_config_channel) { a, b -> tuple(a[0], a[1], b) }

    bring_to_common_space(common_space_input)

    slices_common_space = bring_to_common_space.out
        .flatten()
        .toSortedList { a, b -> a.getName() <=> b.getName() }

    if (params.common_space_preview) {
        preview_input = bring_to_common_space.out
            .flatten()
            .map { file_path ->
                def matcher = file_path.getName() =~ /slice_z(\d+)/
                tuple(matcher ? matcher[0][1] : "unknown", file_path)
            }
        generate_common_space_preview(preview_input)
    }

    // -------------------------------------------------------------------------
    // Stage 5: Missing Slice Interpolation (optional)
    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Stage 6: Pairwise Registration
    // -------------------------------------------------------------------------
    log.info "Registering slices pairwise"

    fixed_slices = all_slices.map { list -> list.size() > 1 ? list.subList(0, list.size() - 1) : [] }.flatten()
    moving_slices = all_slices.map { list -> list.size() > 1 ? list.subList(1, list.size()) : [] }.flatten()
    pairs = fixed_slices.merge(moving_slices)

    if (params.create_registration_masks) {
        mask_input = all_slices.flatten().map { file_path ->
            def matcher = file_path.getName() =~ /slice_z(\d+)/
            tuple(matcher ? matcher[0][1] : "unknown", file_path)
        }
        create_registration_masks(mask_input)

        all_masks = create_registration_masks.out.masks
            .collect()
            .map { list -> list.sort { it.getName() } }

        fixed_masks = all_masks.map { list -> list.size() > 1 ? list.subList(0, list.size() - 1) : [] }.flatten()
        moving_masks = all_masks.map { list -> list.size() > 1 ? list.subList(1, list.size()) : [] }.flatten()

        pairs = pairs
            .merge(moving_masks) { a, b -> tuple(a[0], a[1], b) }
            .merge(fixed_masks) { a, b -> tuple(a[0], a[1], a[2], b) }
    } else {
        pairs = pairs.map { a, b -> tuple(a, b, [], []) }
    }

    register_pairwise(pairs)

    // -------------------------------------------------------------------------
    // Stage 7: Stacking
    // -------------------------------------------------------------------------
    stack_input = all_slices
        .concat(register_pairwise.out.collect())
        .toList()
        .map { both_lists -> tuple(both_lists[0], both_lists[1], subject_name) }

    stack(stack_input)

    // -------------------------------------------------------------------------
    // Stage 8: Report Generation
    // -------------------------------------------------------------------------
    if (params.generate_report) {
        generate_report(stack.out, subject_name)
    }

    // -------------------------------------------------------------------------
    // Stage 9: Diagnostic Analyses (optional)
    // -------------------------------------------------------------------------
    // These analyses help troubleshoot reconstruction artifacts like edge
    // mismatches and "overhangs" in obliquely-mounted samples (e.g., 45° angle)
    //
    // diagnostic_mode=true enables ALL diagnostics (master switch)
    // Individual flags provide granular control when diagnostic_mode=false

    // Master switch logic: diagnostic_mode enables all, or use individual flags
    def runRotationAnalysis = params.diagnostic_mode || params.analyze_rotation_drift
    def runDilationAnalysis = params.diagnostic_mode || params.analyze_tile_dilation
    def runMotorOnlyStitch = params.diagnostic_mode || params.motor_only_stitch

    if (params.diagnostic_mode) {
        log.info "DIAGNOSTIC MODE ENABLED - Running all diagnostic analyses"
        log.info "  - Acquisition rotation analysis (from shifts)"
        log.info "  - Registration rotation drift analysis (mosaic-level)"
        log.info "  - Tile dilation analysis (tile-level)"
        log.info "  - Motor-only stitching"
    }

    // Acquisition rotation analysis: analyzes shift vectors for rotation patterns
    if (params.diagnostic_mode) {
        log.info "Running acquisition rotation analysis..."
        analyze_acquisition_rotation(shifts_xy, register_pairwise.out.collect())
    }

    // Registration rotation drift analysis: detects cumulative rotation between slices
    if (runRotationAnalysis) {
        log.info "Running registration rotation drift analysis..."
        analyze_rotation_drift(register_pairwise.out.collect())
    }

    // Tile dilation analysis: compares motor vs registration positions
    if (runDilationAnalysis) {
        log.info "Running tile dilation analysis..."
        dilation_input = illum_fixed.combine(estimate_xy_transformation.out, by: 0)
        analyze_tile_dilation(dilation_input)
    }

    // Motor-only stitching: creates slices using only motor positions
    if (runMotorOnlyStitch) {
        log.info "Creating motor-only stitched slices..."
        stitch_motor_only(illum_fixed)
    }
}
