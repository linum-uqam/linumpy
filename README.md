# Linumpy

**Linumpy** is the main library supporting research and development at the *Laboratoire d'Imagerie Numérique, Neurophotonique et Microscopie* ([LINUM]).

**Linumpy** contains tools and utilities to quickly work with serial histology and microscopy data. Those tools implement the recommended workflows and parameters used in the lab. The library and scripts can be installed locally by using:

To install the tool, clone the repository and then install them with

```
pip install -e .
```

To use the Napari viewer, you will need to install the following dependencies:

```
pip install napari[all]
``` 

We highly recommend working in a [Python Virtual Environment].

[LINUM]:https://linum.info.uqam.ca
[Python Virtual Environment]:https://virtualenv.pypa.io/en/latest/

**Linumpy** documentation is available: https://linumpy.readthedocs.io/en/latest/

# Execution

To execute the scripts, you can use the following command:

```
nextflow run workflow_soct_3d_slice_reconstruction.nf -resume
```
