# Linumpy

**Linumpy** is the main library supporting research and development at the *Laboratoire d'Imagerie Numérique, Neurophotonique et Microscopie* ([LINUM]).

**Linumpy** contains tools and utilities to quickly work with serial histology and microscopy data. Those tools implement the recommended workflows and parameters used in the lab. 

## Installation
*Although a version of the library is available on PyPI, it has not been updated in a while. We highly discourage its use.*

To install the library, clone the repository, change directory to the repository root and then install it with the command below:
```
pip install .
```

For development, use this command instead:
```
pip install -e .
```

To use the Napari viewer, also install the following dependencies:

```
pip install napari[all]
``` 

**We highly recommend working in a [Python Virtual Environment].**

### Troubleshooting
If the installation fails when building wheels for `asciitree`, define the following environment variable:
```
SETUPTOOLS_USE_DISTUTILS=stdlib
```
before installing (see [issue #45](https://github.com/linum-uqam/linumpy/issues/45)).

## Documentation
**Linumpy** documentation is available: https://linumpy.readthedocs.io/en/latest/

## Execution

To execute the scripts, you can use the following command:

```
nextflow run workflow_soct_3d_slice_reconstruction.nf -resume
```

[LINUM]:https://linum-lab.ca
[Python Virtual Environment]:https://virtualenv.pypa.io/en/latest/
