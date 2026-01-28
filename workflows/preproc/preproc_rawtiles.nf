#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Convert raw S-OCT tiles into mosaic grids and xy shifts
// Input: Directory containing raw data set tiles
// Output: Mosaic grids and xy shifts
//
// Parameters are defined in nextflow.config

process create_mosaic_grid {
    cpus params.processes
    publishDir "$params.output", mode: 'link'  // Hard link: no duplication, file stays accessible
    
    input:
        tuple val(slice_id), path(tiles)
    output:
        tuple val(slice_id), path("*.ome.zarr")
    script:
    String options = ""
    options += params.fix_galvo_shift? "--fix_galvo_shift":"--no-fix_galvo_shift"
    options += " "
    options += params.fix_camera_shift? "--fix_camera_shift":"--no-fix_camera_shift"
    options += " "
    options += params.preprocess? "--preprocess":"--no-preprocess"
    // Select GPU or CPU script based on use_gpu parameter
    String script_name = params.use_gpu ? "linum_create_mosaic_grid_3d_gpu.py" : "linum_create_mosaic_grid_3d.py"
    String gpu_opts = params.use_gpu ? "--use_gpu --galvo_threshold ${params.galvo_confidence_threshold}" : ""
    """
    ${script_name} mosaic_grid_3d_z${slice_id}.ome.zarr --from_tiles_list $tiles --resolution ${params.resolution} --n_processes ${params.processes} --axial_resolution ${params.axial_resolution} --n_levels 0 --sharding_factor ${params.sharding_factor} ${options} ${gpu_opts}
    """
}

process generate_mosaic_preview {
    publishDir "$params.output/previews", mode: 'move'
    
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        path("mosaic_grid_z${slice_id}_preview.png")
    script:
    """
    linum_screenshot_omezarr.py ${mosaic_grid} mosaic_grid_z${slice_id}_preview.png
    """
}

process estimate_xy_shifts_from_metadata {
    cpus params.processes
    publishDir "$params.output", mode: 'copy' 
    input:
        path(input_dir)
    output:
        path("shifts_xy.csv")
    script:
    """
    linum_estimate_xy_shift_from_metadata.py ${input_dir} shifts_xy.csv --n_processes $params.processes
    """
}

process generate_slice_config {
    publishDir "$params.output", mode: 'copy'
    
    input:
        tuple path(shifts_file), path(input_dir)
    
    output:
        path("slice_config.csv")
    
    script:
    String galvo_opts = params.detect_galvo ? "--detect_galvo --tiles_dir ${input_dir} --galvo_threshold ${params.galvo_confidence_threshold}" : ""
    String exclude_first_opt = params.exclude_first_slices > 0 ? "--exclude_first ${params.exclude_first_slices}" : "--exclude_first 0"
    """
    linum_generate_slice_config.py ${shifts_file} slice_config.csv --from_shifts ${exclude_first_opt} ${galvo_opts}
    """
}

process assess_slice_quality {
    publishDir "$params.output", mode: 'copy'

    input:
        tuple path(slice_config), path(mosaics_dir)

    output:
        path("slice_config.csv")

    script:
    String script_name = params.use_gpu ? "linum_assess_slice_quality_gpu.py" : "linum_assess_slice_quality.py"
    String gpu_opts = params.use_gpu ? "--use_gpu" : ""
    String quality_opts = params.min_quality_score > 0 ? "--min_quality ${params.min_quality_score}" : ""
    """
    ${script_name} ${mosaics_dir} slice_config.csv \\
        --update_existing --existing_config ${slice_config} \\
        --exclude_first ${params.exclude_first_slices} \\
        --sample_depth ${params.quality_sample_depth} \\
        ${quality_opts} ${gpu_opts} -f
    """
}

workflow {
    if (params.use_old_folder_structure)
    {
        inputSlices = Channel.fromPath("$params.input/tile_x*_y*_z*/", type: 'dir')
                            .map{path -> tuple(path.toString().substring(path.toString().length() - 2), path)}
                            .groupTuple()
    }
    else
    {
        inputSlices = Channel.fromPath("$params.input/**/tile_x*_y*_z*/", type: 'dir')
                            .map{path -> tuple(path.toString().substring(path.toString().length() - 2), path)}
                            .groupTuple()
    }
    input_dir_channel = Channel.fromPath("$params.input", type: 'dir')
    output_dir_channel = Channel.fromPath("$params.output", type: 'dir')

    // Generate a 3D mosaic grid at full resolution
    create_mosaic_grid(inputSlices)

    // [Optional] Generate orthogonal view previews of mosaic grids
    if (params.generate_previews) {
        generate_mosaic_preview(create_mosaic_grid.out)
    }

    // Estimate XY shifts from metadata
    estimate_xy_shifts_from_metadata(input_dir_channel)

    // Generate slice configuration file (for controlling which slices to use in reconstruction)
    if (params.generate_slice_config) {
        // Combine shifts file with input directory for optional galvo detection
        slice_config_input = estimate_xy_shifts_from_metadata.out
            .combine(input_dir_channel)
        generate_slice_config(slice_config_input)

        // [Optional] Assess quality and update slice config
        if (params.assess_quality) {
            // Wait for all mosaics to be created, then run quality assessment
            all_mosaics_done = create_mosaic_grid.out.collect()
            quality_input = generate_slice_config.out
                .combine(output_dir_channel)
            assess_slice_quality(quality_input)
        }
    }
}
