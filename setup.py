#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import subprocess
import sys
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


class PostInstallCommand(install):
    """Post-installation command to apply patchelf fix for JAX CUDA."""

    def run(self):
        install.run(self)
        self._post_install()

    def _post_install(self):
        """Apply patchelf fix to JAX/jaxlib .so files if needed."""
        # Only run on Linux where patchelf is needed
        if sys.platform != 'linux':
            return

        # Check if patchelf is available
        try:
            subprocess.run(['patchelf', '--version'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("\n" + "=" * 70)
            print("NOTE: patchelf not installed - skipping JAX CUDA fix")
            print("If using GPU, install patchelf and run:")
            print("  source scripts/fix_jax_cuda_plugin.sh")
            print("=" * 70 + "\n")
            return

        # Try to find and patch jaxlib .so files
        try:
            import site
            sp = site.getsitepackages()[0]
            patched = 0

            # Patch jaxlib
            jaxlib_path = os.path.join(sp, 'jaxlib')
            if os.path.isdir(jaxlib_path):
                for so_file in Path(jaxlib_path).rglob("*.so"):
                    try:
                        subprocess.run(
                            ['patchelf', '--clear-execstack', str(so_file)],
                            capture_output=True, check=True
                        )
                        patched += 1
                    except subprocess.CalledProcessError:
                        pass

            # Patch jax_plugins
            jax_plugins_path = os.path.join(sp, 'jax_plugins')
            if os.path.isdir(jax_plugins_path):
                for so_file in Path(jax_plugins_path).rglob("*.so"):
                    try:
                        subprocess.run(
                            ['patchelf', '--clear-execstack', str(so_file)],
                            capture_output=True, check=True
                        )
                        patched += 1
                    except subprocess.CalledProcessError:
                        pass

            if patched > 0:
                print(f"\n✅ Applied patchelf fix to {patched} JAX .so files")
                print("   JAX CUDA should work on modern Linux kernels.\n")
        except Exception as e:
            print(f"\n⚠️  Could not apply patchelf fix: {e}")
            print("   Run manually: source scripts/fix_jax_cuda_plugin.sh\n")


class PostDevelopCommand(develop):
    """Post-develop command to apply patchelf fix for JAX CUDA."""

    def run(self):
        develop.run(self)
        # Use the same post-install logic
        PostInstallCommand._post_install(self)


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

# Optional dependencies
# NOTE: GPU extras are MUTUALLY EXCLUSIVE - install only ONE of: [gpu], [gpu-cuda11], [gpu-cuda13]
# Installing multiple GPU extras will cause JAX plugin conflicts!
#
# GPU support requires CuPy matching your CUDA version:
#   - CUDA 11.x: pip install linumpy[gpu-cuda11]
#   - CUDA 12.x: pip install linumpy[gpu]  (default)
#   - CUDA 13.x: pip install linumpy[gpu-cuda13]
#
# NOTE: BaSiCPy (fix_illumination) uses JAX which also supports GPU.
# JAX versions are constrained by basicpy's requirements (jax<=0.4.23).
# JAX CUDA 13 plugin requires jax>=0.4.36, so gpu-cuda13 uses CUDA 12 JAX packages.
extras_require = {
    'gpu': [
        'cupy-cuda12x>=12.0.0',  # Default: CUDA 12.x
        # JAX CUDA 12 packages (equivalent to jax[cuda12]<=0.4.23)
        'jax<=0.4.23',
        'jaxlib==0.4.23',
        'jax-cuda12-plugin==0.4.23; sys_platform == "linux"',
        'nvidia-cublas-cu12>=12.2.5.6; sys_platform == "linux"',
        'nvidia-cuda-cupti-cu12>=12.2.142; sys_platform == "linux"',
        'nvidia-cuda-nvcc-cu12>=12.2.140; sys_platform == "linux"',
        'nvidia-cuda-runtime-cu12>=12.2.140; sys_platform == "linux"',
        'nvidia-cudnn-cu12>=8.9; sys_platform == "linux"',
        'nvidia-cufft-cu12>=11.0.8.103; sys_platform == "linux"',
        'nvidia-cusolver-cu12>=11.5.2; sys_platform == "linux"',
        'nvidia-cusparse-cu12>=12.1.2.141; sys_platform == "linux"',
        'nvidia-nccl-cu12>=2.18.3; sys_platform == "linux"',
        'nvidia-nvjitlink-cu12>=12.2; sys_platform == "linux"',
    ],
    'gpu-cuda11': [
        'cupy-cuda11x>=11.0.0',
        # JAX CUDA 11 packages (equivalent to jax[cuda11_pip]<=0.4.23)
        'jax<=0.4.23',
        'jaxlib==0.4.23+cuda11.cudnn86; sys_platform == "linux"',
        'nvidia-cublas-cu11>=11.11; sys_platform == "linux"',
        'nvidia-cuda-cupti-cu11>=11.8; sys_platform == "linux"',
        'nvidia-cuda-nvcc-cu11>=11.8; sys_platform == "linux"',
        'nvidia-cuda-runtime-cu11>=11.8; sys_platform == "linux"',
        'nvidia-cudnn-cu11>=8.8; sys_platform == "linux"',
        'nvidia-cufft-cu11>=10.9; sys_platform == "linux"',
        'nvidia-cusolver-cu11>=11.4; sys_platform == "linux"',
        'nvidia-cusparse-cu11>=11.7; sys_platform == "linux"',
        'nvidia-nccl-cu11>=2.18.3; sys_platform == "linux"',
    ],
    'gpu-cuda13': [
        'cupy-cuda13x>=13.0.0',  # CUDA 13.x for CuPy (linumpy operations)
        # JAX 0.4.23 uses CUDA 12 driver API but links against library ABI version 11:
        #   - libcusolver.so.11, libcusparse.so.11, libcufft.so.10, libcublas.so.11
        # The -cu12 packages contain these .so.11 files (NOT .so.12!)
        # Non-suffixed packages contain .so.12/.so.13 which are INCOMPATIBLE.
        #
        # IMPORTANT: After installing, run: source scripts/fix_jax_cuda_plugin.sh
        # This applies patchelf fix and sets up LD_LIBRARY_PATH correctly.
        'jax==0.4.23',
        'jaxlib==0.4.23; sys_platform != "linux"',
        # Note: jaxlib CUDA wheel must be installed separately on Linux:
        # pip install jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        # Pinned nvidia package versions that JAX 0.4.23 was built with (Dec 2023)
        'nvidia-cublas-cu12==12.3.4.1; sys_platform == "linux"',
        'nvidia-cuda-cupti-cu12==12.3.101; sys_platform == "linux"',
        'nvidia-cuda-runtime-cu12==12.3.101; sys_platform == "linux"',
        'nvidia-cudnn-cu12==8.9.7.29; sys_platform == "linux"',
        'nvidia-cufft-cu12==11.0.12.1; sys_platform == "linux"',
        'nvidia-cusolver-cu12==11.5.4.101; sys_platform == "linux"',
        'nvidia-cusparse-cu12==12.2.0.103; sys_platform == "linux"',
        'nvidia-nccl-cu12==2.19.3; sys_platform == "linux"',
        'nvidia-nvjitlink-cu12==12.3.101; sys_platform == "linux"',
    ],
    'dev': [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
    ],
    'docs': [
        'sphinx>=6.0.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
}
# 'all' includes everything except GPU (since GPU requires specific CUDA version)
extras_require['all'] = extras_require['dev'] + extras_require['docs']

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
            extras_require=extras_require,
            include_package_data=True,
            # Post-install hooks to apply patchelf fix for JAX CUDA
            cmdclass={
                'install': PostInstallCommand,
                'develop': PostDevelopCommand,
            })

setup(**opts)
