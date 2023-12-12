#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_directory = "/home/jreynaud/scratch/2023-07-13-Alexia--eau-10x-uqam-Mounier"
params.output_directory = "/home/jreynaud/scratch/converted_data"

process convert_bin_to_nifty {
    input:
        path tile_directory
    script:
    """
    z_value=\$(basename $tile_directory | sed -n "s/.*_z\\([0-9]\\+\\).*/\\1/p")
    output_dir=${params.output_directory}/slice_\${z_value}
    output_file=\${output_dir}/${tile_directory.baseName}.nii
    linum_convert_bin_to_nii.py $tile_directory \$output_file
    """
}

workflow{
    // Detect every tile directory
    tiles = channel.fromPath(params.input_directory + "/tile_*", type:"dir")
    convert_bin_to_nifty(tiles.flatten())
}