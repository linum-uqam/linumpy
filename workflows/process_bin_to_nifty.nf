#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.input_directory = "/home/jolefc/projects/def-jolefc/share/serial_oct/2023-07-13-Alexia--eau-10x-uqam-Mounier"
params.output_directory = "/home/jolefc/projects/def-jolefc/share/serial_oct/2023-07-13-Alexia--eau-10x-uqam-Mounier_NIFTY"

process convert_bin_to_nifty {
    input:
        path raw_data
    publishDir path: "${params.output_directory}", mode: 'copy'
    script:
    """
    linum_convert_oct_to_nii.py $raw_data ${params.output_directory} .nii
    """
    }

workflow{
    convert_bin_to_nifty(params.input_directory)
}