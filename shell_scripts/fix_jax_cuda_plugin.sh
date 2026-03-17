#!/bin/bash
# Fix JAX CUDA plugin for JAX 0.4.23 (required by BaSiCPy)
#
# JAX 0.4.23 was compiled with CUDA 12 driver API but uses:
#   - cuSOLVER 11.x (libcusolver.so.11)
#   - cuSPARSE 11.x (libcusparse.so.11)
#   - cuFFT 10.x (libcufft.so.10 or .so.11)
#   - cuBLAS 11.x (libcublas.so.11)
#   - cuDNN 8.x (libcudnn.so.8)
#
# The nvidia-xxx-cu12 packages contain these .so.11 files.
# Non-suffixed packages (nvidia-cusolver) contain .so.12/.so.13 which are INCOMPATIBLE.
#
# This script:
# 1. Uninstalls conflicting packages
# 2. Installs JAX 0.4.23 with correct CUDA 12 packages
# 3. Applies patchelf fix for modern Linux kernels
# 4. Verifies the installation
#
# Usage:
#   source scripts/fix_jax_cuda_plugin.sh
#   # or
#   bash scripts/fix_jax_cuda_plugin.sh

# Don't use set -e as it can cause SSH disconnection issues
# Instead, handle errors explicitly where needed

echo "========================================================================"
echo " JAX CUDA Fix for JAX 0.4.23 (BaSiCPy compatibility)"
echo "========================================================================"
echo ""

# Check if running interactively (for prompts)
if [ -t 0 ]; then
    INTERACTIVE=1
else
    INTERACTIVE=0
    echo "Running in non-interactive mode (SSH/pipe detected)"
fi

# Find Python
PYTHON_CMD=""
if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -n "$PYENV_VIRTUAL_ENV" ] && [ -x "$PYENV_VIRTUAL_ENV/bin/python" ]; then
    PYTHON_CMD="$PYENV_VIRTUAL_ENV/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found"
    # Use return if sourced, exit if run as script
    (return 0 2>/dev/null) && return 1 || exit 1
fi

echo "Python: $PYTHON_CMD"
SP=$("$PYTHON_CMD" -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages: $SP"

# Check for patchelf
PATCHELF_AVAILABLE=0
if command -v patchelf &> /dev/null; then
    PATCHELF_AVAILABLE=1
else
    echo ""
    echo "⚠️  patchelf is required but not installed"
    echo "   Install with: sudo apt install patchelf"
    if [ $INTERACTIVE -eq 1 ]; then
        read -p "Continue without patchelf? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            # Use return if sourced, exit if run as script
            (return 0 2>/dev/null) && return 1 || exit 1
        fi
    else
        echo "   Continuing without patchelf (non-interactive mode)"
        echo "   You may need to run patchelf manually later."
    fi
fi

# Step 1: Clean up conflicting packages
echo ""
echo "=== Step 1: Removing conflicting packages ==="

# Remove non-suffixed nvidia packages that have wrong library versions
echo "Removing non-suffixed nvidia packages (contain .so.12/.so.13, incompatible)..."
"$PYTHON_CMD" -m pip uninstall -y \
    nvidia-cusolver nvidia-cufft nvidia-cusparse nvidia-cublas \
    nvidia-cuda-runtime nvidia-cudnn nvidia-nvjitlink nvidia-nccl \
    2>/dev/null || true

# Remove any CUDA 13 JAX plugins
echo "Removing CUDA 13 JAX plugins..."
"$PYTHON_CMD" -m pip uninstall -y \
    jax-cuda13-plugin jax-cuda13-pjrt \
    2>/dev/null || true

# Step 2: Install JAX with CUDA 12 support
echo ""
echo "=== Step 2: Installing JAX 0.4.23 with CUDA 12 support ==="

# Uninstall existing JAX and nvidia packages first
"$PYTHON_CMD" -m pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt 2>/dev/null || true

# Also uninstall all nvidia packages to avoid version conflicts
"$PYTHON_CMD" -m pip uninstall -y \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvcc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 \
    2>/dev/null || true

# Install JAX 0.4.23 with EXACT nvidia package versions it was built with
# These versions are from the JAX 0.4.23 release (December 2023)
echo "Installing JAX 0.4.23 with pinned nvidia package versions..."

# First install JAX without cuda extra to avoid pulling in wrong versions
"$PYTHON_CMD" -m pip install 'jax==0.4.23' 'jaxlib==0.4.23+cuda12.cudnn89' \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install the exact nvidia package versions that JAX 0.4.23 was built with
# These are the versions from late 2023 that have the correct .so versions
"$PYTHON_CMD" -m pip install \
    'nvidia-cublas-cu12==12.3.4.1' \
    'nvidia-cuda-cupti-cu12==12.3.101' \
    'nvidia-cuda-runtime-cu12==12.3.101' \
    'nvidia-cudnn-cu12==8.9.7.29' \
    'nvidia-cufft-cu12==11.0.12.1' \
    'nvidia-cusolver-cu12==11.5.4.101' \
    'nvidia-cusparse-cu12==12.2.0.103' \
    'nvidia-nccl-cu12==2.19.3' \
    'nvidia-nvjitlink-cu12==12.3.101'

echo "✓ JAX installed with pinned versions"

# Step 3: Verify -cu12 packages have correct library versions
echo ""
echo "=== Step 3: Verifying library versions ==="

"$PYTHON_CMD" << 'VERIFY_LIBS'
import os
import site
import glob

sp = site.getsitepackages()[0]

# Check for correct library versions from pinned nvidia packages
checks = [
    ("nvidia/cusolver/lib", "libcusolver.so.11", "nvidia-cusolver-cu12==11.5.4.101"),
    ("nvidia/cusparse/lib", "libcusparse.so.12", "nvidia-cusparse-cu12==12.2.0.103"),
    ("nvidia/cufft/lib", "libcufft.so.11", "nvidia-cufft-cu12==11.0.12.1"),
    ("nvidia/cublas/lib", "libcublas.so.12", "nvidia-cublas-cu12==12.3.4.1"),
    ("nvidia/cuda_runtime/lib", "libcudart.so.12", "nvidia-cuda-runtime-cu12==12.3.101"),
    ("nvidia/cudnn/lib", "libcudnn.so.8", "nvidia-cudnn-cu12==8.9.7.29"),
]

all_ok = True
for lib_path, lib_file, package in checks:
    full_path = os.path.join(sp, lib_path, lib_file)
    # Also check for any version of this library
    pattern = os.path.join(sp, lib_path, lib_file.rsplit('.so', 1)[0] + ".so*")
    found = glob.glob(pattern)
    if found:
        found_name = os.path.basename(sorted(found)[0])
        if os.path.exists(full_path):
            print(f"  ✓ {lib_file} found")
        else:
            print(f"  ⚠️ {found_name} found (expected {lib_file}) - version mismatch!")
            all_ok = False
    else:
        print(f"  ✗ {lib_file} NOT FOUND - install {package}")
        all_ok = False

if all_ok:
    print("\n✓ All nvidia packages have correct library versions")
else:
    print("\n⚠️  Some libraries have wrong versions - JAX may not work correctly")
    print("   Run this script again to reinstall correct versions")
VERIFY_LIBS

# Step 4: Apply patchelf fix
echo ""
echo "=== Step 4: Applying patchelf fix ==="

if [ $PATCHELF_AVAILABLE -eq 1 ]; then
    JAXLIB_PATH=$("$PYTHON_CMD" -c "import jaxlib; print(jaxlib.__path__[0])" 2>/dev/null || echo "")

    if [ -n "$JAXLIB_PATH" ] && [ -d "$JAXLIB_PATH" ]; then
        echo "Patching jaxlib at: $JAXLIB_PATH"
        find "$JAXLIB_PATH" -name "*.so" -type f -exec patchelf --clear-execstack {} \; 2>/dev/null || true
        echo "  ✓ Applied patchelf to jaxlib"
    fi

    JAX_PLUGINS_PATH="${SP}/jax_plugins"
    if [ -d "$JAX_PLUGINS_PATH" ]; then
        echo "Patching jax_plugins at: $JAX_PLUGINS_PATH"
        find "$JAX_PLUGINS_PATH" -name "*.so" -type f -exec patchelf --clear-execstack {} \; 2>/dev/null || true
        echo "  ✓ Applied patchelf to jax_plugins"
    fi
else
    echo "⚠️  Skipping patchelf (not installed)"
fi

# Step 5: Set up LD_LIBRARY_PATH
echo ""
echo "=== Step 5: Setting up LD_LIBRARY_PATH ==="

# Build LD_LIBRARY_PATH with -cu12 package paths
NEW_LD_PATH=""
for lib_dir in nvidia/cublas/lib nvidia/cuda_runtime/lib nvidia/cusolver/lib nvidia/cusparse/lib nvidia/cufft/lib nvidia/cudnn/lib nvidia/nvjitlink/lib; do
    full_path="${SP}/${lib_dir}"
    if [ -d "$full_path" ]; then
        if [ -n "$NEW_LD_PATH" ]; then
            NEW_LD_PATH="${NEW_LD_PATH}:${full_path}"
        else
            NEW_LD_PATH="${full_path}"
        fi
    fi
done

# Also check for system cuDNN 8.x
SYSTEM_CUDNN=""
for sys_path in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 /usr/lib64; do
    if [ -f "${sys_path}/libcudnn.so.8" ]; then
        SYSTEM_CUDNN="${sys_path}"
        break
    fi
done

if [ -n "$SYSTEM_CUDNN" ]; then
    echo "Found system cuDNN 8.x at: $SYSTEM_CUDNN"
    NEW_LD_PATH="${SYSTEM_CUDNN}:${NEW_LD_PATH}"
fi

# Append existing LD_LIBRARY_PATH
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="${NEW_LD_PATH}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${NEW_LD_PATH}"
fi

echo "LD_LIBRARY_PATH configured with $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | wc -l) paths"

# Step 6: Test JAX
echo ""
echo "=== Step 6: Testing JAX CUDA ==="

# First, show what library files actually exist
echo "Checking library files in nvidia packages..."
"$PYTHON_CMD" << 'CHECK_LIBS'
import os
import site
import glob

sp = site.getsitepackages()[0]

# Check each nvidia lib directory
lib_dirs = [
    'nvidia/cusolver/lib',
    'nvidia/cublas/lib',
    'nvidia/cusparse/lib',
    'nvidia/cufft/lib',
    'nvidia/cuda_runtime/lib',
    'nvidia/cudnn/lib',
]

for lib_dir in lib_dirs:
    full_path = os.path.join(sp, lib_dir)
    if os.path.isdir(full_path):
        files = [f for f in os.listdir(full_path) if '.so' in f]
        print(f"  {lib_dir}: {', '.join(sorted(files)[:3])}...")
CHECK_LIBS

echo ""

# Build LD_PRELOAD to force library loading before JAX initializes
# Use the correct .so versions from pinned nvidia packages
PRELOAD_LIBS=""
for lib in libcusolver.so.11 libcublas.so.12 libcublasLt.so.12 libcusparse.so.12 libcufft.so.11; do
    for search_path in ${SP}/nvidia/cusolver/lib ${SP}/nvidia/cublas/lib ${SP}/nvidia/cusparse/lib ${SP}/nvidia/cufft/lib; do
        if [ -f "${search_path}/${lib}" ]; then
            if [ -n "$PRELOAD_LIBS" ]; then
                PRELOAD_LIBS="${PRELOAD_LIBS}:${search_path}/${lib}"
            else
                PRELOAD_LIBS="${search_path}/${lib}"
            fi
            break
        fi
    done
done

if [ -n "$PRELOAD_LIBS" ]; then
    echo "Preloading CUDA libraries via LD_PRELOAD..."
    export LD_PRELOAD="$PRELOAD_LIBS"
fi

TEST_RESULT=$("$PYTHON_CMD" -c "
import os
import sys
import ctypes

# Preload CUDA libraries using ctypes BEFORE importing JAX
# This ensures cuSOLVER symbols are available when XLA initializes
ld_path = os.environ.get('LD_LIBRARY_PATH', '')
print('Preloading CUDA libraries...')

# Libraries to preload in order (dependencies first)
# These are the actual .so versions from the pinned nvidia packages
libs_to_load = [
    ('libcudart.so.12', 'CUDA runtime'),
    ('libcublas.so.12', 'cuBLAS'),
    ('libcublasLt.so.12', 'cuBLAS Lt'),
    ('libcusolver.so.11', 'cuSOLVER'),
    ('libcusparse.so.12', 'cuSPARSE'),
    ('libcufft.so.11', 'cuFFT'),
    ('libcudnn.so.8', 'cuDNN'),
]

loaded = set()
for lib_name, desc in libs_to_load:
    if desc in loaded:
        continue
    for path in ld_path.split(':'):
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                print(f'  ✓ {lib_name}')
                loaded.add(desc)
            except Exception as e:
                print(f'  ✗ {lib_name}: {e}')
            break

# Test JAX
try:
    import jax
    devices = jax.devices()
    print(f'JAX devices: {devices}')

    has_gpu = any('cuda' in str(d).lower() for d in devices)
    if not has_gpu:
        print('⚠️  No CUDA devices found')
        sys.exit(1)

    # Test SVD (used by BaSiCPy)
    import jax.numpy as jnp
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    u, s, v = jnp.linalg.svd(a)
    print(f'SVD test passed: singular values = {s}')
    print('✅ JAX CUDA is working!')

except Exception as e:
    print(f'❌ JAX test failed: {e}')
    sys.exit(1)
" 2>&1)

echo "$TEST_RESULT"

if echo "$TEST_RESULT" | grep -q "JAX CUDA is working"; then
    echo ""
    echo "========================================================================"
    echo " SUCCESS!"
    echo "========================================================================"
    echo ""
    echo "Before running JAX/BaSiCPy, set LD_LIBRARY_PATH:"
    echo ""
    echo "  export LD_LIBRARY_PATH=\"${NEW_LD_PATH}:\${LD_LIBRARY_PATH}\""
    echo ""
    echo "Or add to ~/.bashrc for permanent setup."
    echo ""
    echo "Verify with: linum_diagnose_pipeline.py --benchmark"
else
    echo ""
    echo "========================================================================"
    echo " SETUP FAILED"
    echo "========================================================================"
    echo ""
    echo "Common issues:"
    echo "  1. patchelf not installed: sudo apt install patchelf"
    echo "  2. Wrong cuDNN version: JAX 0.4.23 needs cuDNN 8.x (libcudnn.so.8)"
    echo "  3. CUDA driver too old: Need CUDA 12+ driver"
    echo ""
    echo "For diagnostics: linum_diagnose_pipeline.py --debug-cuda"
    # Use return if sourced, exit if run as script
    # This prevents SSH session termination when sourced
    (return 0 2>/dev/null) && return 1 || exit 1
fi
