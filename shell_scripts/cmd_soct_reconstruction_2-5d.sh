#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=92G
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lefebvre.joel@uqam.ca
#SBATCH --account=def-jolefc

module load nextflow
module load apptainer

# Parameters
DIRECTORY=/lustre04/scratch/jolefc/2024-06-05-S34-Coronal
WORKFLOW_FILE=$DIRECTORY/workflow_reconstruction_2.5d.nf
CONFIG_FILE=$DIRECTORY/reconstruction.config
SINGULARITY=$DIRECTORY/linumpy.sif

nextflow run $WORKFLOW_FILE -c $CONFIG_FILE -with-singularity $SINGULARITY --directory $DIRECTORY -resume
