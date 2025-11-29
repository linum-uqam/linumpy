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
    input:
        tuple val(slice_id), path(tiles)
    output:
        tuple val(slice_id), path("*.ome.zarr")
    script:
    String options = ""
    options += params.fix_galvo_shift? "--fix_galvo_shift":"--no-fix_galvo_shift"
    options += " "
    options += params.fix_camera_shift? "--fix_camera_shift":"--no-fix_camera_shift"
    """
    linum_create_mosaic_grid_3d.py mosaic_grid_3d_z${slice_id}.ome.zarr --from_tiles_list $tiles --resolution ${params.resolution} --n_processes ${params.processes} --axial_resolution ${params.axial_resolution} --n_levels 0 --sharding_factor ${params.sharding_factor} ${options}
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
        path(shifts_file)
    
    output:
        path("slice_config.csv")
    
    script:
    """
    linum_generate_slice_config.py ${shifts_file} slice_config.csv --from_shifts
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
        generate_slice_config(estimate_xy_shifts_from_metadata.out)
    }
}
