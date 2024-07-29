#!/usr/bin/env bash

# Parameters
DIRECTORY=/Users/jlefebvre/Downloads/2024-05-16-Mounier-15-Horizontal
WORKFLOW_FILE=$DIRECTORY/workflow_reconstruction_2.5d.nf
CONFIG_FILE=$DIRECTORY/reconstruction_2.5d_macbook_joel.config
SINGULARITY=$DIRECTORY/linumpy.sif

nextflow run $WORKFLOW_FILE -c $CONFIG_FILE --directory $DIRECTORY -resume
