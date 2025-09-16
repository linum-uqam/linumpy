#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Workflow Description
// Convert raw S-OCT tiles into mosaic grids and xy shifts
// Input: Directory containing raw data set tiles
// Output: Mosaic grids and xy shifts

// Parameters
params.input = ""
params.output = "output"
params.use_old_folder_structure = false // Use the old folder structure where tiles are not stored in subfolders based on their Z
params.processes = 1 // Maximum number of python processes per nextflow process
params.axial_resolution = 1.5 // Axial resolution of imaging system in microns
params.resolution = -1 // resolution of mosaic grid. Defaults to full resolution.

process create_mosaic_grid {
    cpus params.processes
    input:
        tuple val(slice_id), path(tiles)
    output:
        tuple val(slice_id), path("*.ome.zarr")
    script:
    """
    linum_create_mosaic_grid_3d.py mosaic_grid_3d_z${slice_id}.ome.zarr --from_tiles_list $tiles --resolution ${params.resolution} --n_processes ${params.processes} --axial_resolution ${params.axial_resolution} --n_levels 0 --disable_fix_shift
    """
}

process compress_mosaic_grid {
    publishDir "$params.output"
    input:
        tuple val(slice_id), path(mosaic_grid)
    output:
        tuple val(slice_id), path("mosaic_grid_3d_z${slice_id}.ome.zarr.tar.gz")
    script:
    """
    tar -czvf mosaic_grid_3d_z${slice_id}.ome.zarr.tar.gz ${mosaic_grid}
    """
}

process estimate_xy_shifts_from_metadata {
    cpus params.processes
    publishDir "$params.output"
    input:
        path(input_dir)
    output:
        path("shifts_xy.csv")
    script:
    """
    linum_estimate_xy_shift_from_metadata.py ${input_dir} shifts_xy.csv --n_processes $params.processes
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

    // Compress to zip to reduce the number of files
    compress_mosaic_grid(create_mosaic_grid.out)

    // Estimate XY shifts from metadata
    estimate_xy_shifts_from_metadata(input_dir_channel)
}
