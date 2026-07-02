#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
 * Diagnostic processes for the 3D reconstruction pipeline.
 *
 * These are side-channel artefacts (rotation analyses, motor-only stitches /
 * stacks, motor-vs-refined comparisons). They are gated in the main workflow
 * by `params.diagnostic_mode` or per-stage flags
 * (analyze_rotation_drift, motor_only_stitch, motor_only_stack,
 *  analyze_acquisition_rotation, compare_stitching).
 *
 * Sub-workflow conventions: docs/NEXTFLOW_WORKFLOWS.md.
 */

process analyze_rotation_drift {
    publishDir "${params.output}/diagnostics/rotation_analysis", mode: 'copy'

    input:
    path "register_pairwise/*"

    output:
    path "rotation_analysis/*"

    script:
    """
    linum-analyze-registration-transforms register_pairwise rotation_analysis \
        --resolution ${params.resolution} \
        --rotation_threshold ${params.diagnostic_rotation_threshold}
    """

    stub:
    """
    mkdir -p rotation_analysis
    touch rotation_analysis/placeholder.txt
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
    linum-stitch-motor-only ${mosaic_grid} "slice_z${slice_id}_motor_only.ome.zarr" \
        --overlap_fraction ${params.motor_only_overlap} \
        --blending_method ${blending}
    """

    stub:
    """
    mkdir -p slice_z${slice_id}_motor_only.ome.zarr
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
    linum-stitch-3d-refined ${mosaic_grid} "slice_z${slice_id}_refined.ome.zarr" \
        --overlap_fraction ${params.stitch_overlap_fraction} \
        --blending_method diffusion \
        --refinement_mode blend_shift \
        --max_refinement_px ${params.max_blend_refinement_px} \
        ${refinement_out} -f
    """

    stub:
    """
    mkdir -p slice_z${slice_id}_refined.ome.zarr
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
    linum-compare-stitching ${motor_stitch} ${refined_stitch} \
        "slice_z${slice_id}_comparison" \
        --label1 "Motor-only" --label2 "Refined" \
        --tile_step ${params.comparison_tile_step}
    """

    stub:
    """
    mkdir -p slice_z${slice_id}_comparison
    touch slice_z${slice_id}_comparison/placeholder.txt
    """
}

process stack_motor_only {
    publishDir "${params.output}/diagnostics/motor_only_stack", mode: 'copy'

    input:
    path "slices/*"
    path shifts_file

    output:
    path "motor_only_stack.ome.zarr"
    path "motor_only_stack_preview.png", optional: true

    script:
    def blending_arg = params.motor_only_stack_blending ?: 'none'
    """
    linum-stack-motor-only slices ${shifts_file} motor_only_stack.ome.zarr \
        --blending ${blending_arg} \
        --preview motor_only_stack_preview.png
    """

    stub:
    """
    mkdir -p motor_only_stack.ome.zarr
    """
}

process analyze_acquisition_rotation {
    publishDir "${params.output}/diagnostics/acquisition_rotation", mode: 'copy'

    input:
    path shifts_file
    path "register_pairwise/*"

    output:
    path "acquisition_rotation_analysis/*"

    script:
    """
    linum-analyze-acquisition-rotation ${shifts_file} acquisition_rotation_analysis \
        --registration_dir register_pairwise \
        --resolution ${params.resolution}
    """

    stub:
    """
    mkdir -p acquisition_rotation_analysis
    touch acquisition_rotation_analysis/placeholder.txt
    """
}
