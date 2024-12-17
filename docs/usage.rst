Usage
=====

The Nextflow workflows in the `workflow` directory can be used to perform reconstruction of the serial OCT data.

Cluster Reconstruction
----------------------
To submit a reconstruction on a DigitalAlliance cluster (e.g. Béluga or Narval), you need to prepare a bash script describing the job parameters.
Here's an example of such a script for a 2.5D reconstruction.

.. note::

    For this example, the nextflow pipeline file, nextflow configuration file and the Singularity image containing a compiled version of linumpy are in the same directory as the reconstruction directory. **See the workflow documentation to know how to adapt this example for your specific pipeline**

.. code-block:: bash

    #!/bin/sh
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=40
    #SBATCH --mem=92G
    #SBATCH --time=1:00:00
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=<YOUR_EMAIL_ADDRESS>
    #SBATCH --account=<DIG_ALLIANCE_PROJECT_ID>

    module load nextflow
    module load apptainer

    # Parameters
    DIRECTORY=<FULL_PATH_TO_THE_RAW_DATA>
    WORKFLOW_FILE=$DIRECTORY/workflow_reconstruction_2.5d.nf
    CONFIG_FILE=$DIRECTORY/reconstruction_2.5d_beluga.config
    SINGULARITY=$DIRECTORY/linumpy.sif

    nextflow run $WORKFLOW_FILE -c $CONFIG_FILE -with-singularity $SINGULARITY \
             --directory $DIRECTORY -resume




