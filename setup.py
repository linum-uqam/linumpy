#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob

from setuptools import setup, find_packages

# Versions
_version_major = 0
_version_minor = 1
_version_micro = 1
_version_extra = ''

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)
__version__ = '.'.join(map(str, _ver))

# PyPi Package Information
CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Load long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Read the requirements
with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    for dependency in required_dependencies:
        if dependency[0:2] == '-e':
            repo_name = dependency.split('=')[-1]
            repo_url = dependency[3:]
            external_dependencies.append('{} @ {}'.format(repo_name, repo_url))
        else:
            external_dependencies.append(dependency)

# get all scripts in the script folder
SCRIPTS=glob.glob("scripts/*.py")

opts = dict(name="linumpy",
            maintainer="Joël Lefebvre",
            maintainer_email="lefebvre.joel@uqam.ca",
            description="linumpy: microscopy tools and utilities",
            long_description=long_description,
            long_description_content_type="text/markdown",
            url="https://github.com/linum-uqam/linumpy",
            license="GPLv3+",
            license_files=["LICENSE"],
            python_requires=">=3.11",
            download_url="",
            classifiers=CLASSIFIERS,
            author="The LINUM developers",
            author_email="",
            platforms="OS Independent",
            version=__version__,
            packages=find_packages(),
            setup_requires=['numpy'],
            entry_points={
                'console_scripts': ["{}=scripts.{}:main".format(
                    os.path.basename(s),
                    os.path.basename(s).split(".")[0]) for s in SCRIPTS]
            },
            install_requires=external_dependencies,
            include_package_data=True)

setup(**opts)
