# -*- coding: utf-8 -*-

import glob

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''
_version_extra = ''

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Short description:
description = "linumpy: microscopy tools and utilities"
# Long description (for the pypi page)
long_description = """
Linumpy
=======
Linumpy is a small library containing tools and utilities to
quickly work with microscopy and serial histology data.

License
=======
``linumpy`` licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2021--, Laboratoire d'Imagerie Numérique, Neurophotonique et Microscopie [LINUM],
Université du Québec à Montréal (UQÀM).
"""

NAME = "linumpy"
MAINTAINER = "Joël Lefebvre"
MAINTAINER_EMAIL = "lefebvre.joel@uqam.ca"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/linum-uqam/linumpy"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "The LINUM developers"
AUTHOR_EMAIL = ""
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
SCRIPTS = glob.glob("scripts/*.py")

PREVIOUS_MAINTAINERS = []
